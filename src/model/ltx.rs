//! LTX-Video text-to-video (v0.9.0, 2B): T5-XXL prompt embeddings, the 28-block video DiT (adaLN-single,
//! RMS-normed q/k, 3-axis interleaved RoPE, cross-attention over the projected T5 tokens), a flow-matching
//! Euler loop with dynamic shifting + terminal stretch, and the causal-3D VAE of [`crate::model::ltx_vae`].
//!
//! The transformer is a diffusers `transformer/` dir or a ComfyUI-style GGUF (city96 / calcuis `ltxv`)
//! read through FLUX's [`Store`] (quantized blocks stay mapped, decoded per GEMM panel). The T5 is the
//! base repo's `text_encoder/` shards or a `t5encoder` GGUF. The text stage runs first and is dropped
//! before the transformer is fetched, so the three models are never resident (or on disk) together
//! unless the caller keeps them.
//!
//! Reference: diffusers `models/transformers/transformer_ltx.py`, `pipelines/ltx/pipeline_ltx.py`.

use crate::runtime::sched;
use crate::model::flux::{read_json, t5_ids, t5_tokenizer, Lin, Store};
use crate::model::hub::{
    component_weights, hub_json, is_synthetic_model_id, list_repo_files, select_weight_files,
    split_weight_file, HuggingFaceHub, RepoFile,
};
use crate::model::image::ImageBuffer;
use crate::model::ltx_vae::LtxVae;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::runtime::sched::Rng;
use crate::model::t5::T5Encoder;
use crate::model::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod graph_program;

/// A tiny random-weight LTX transformer forward as a graph (the GPU-only matrix fixture).
pub fn tiny_dit_graph() -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
    graph_program::fixture::tiny_graph()
}

const HEADS: usize = 32;
const LATENT: usize = 128;
const TIME_DIM: usize = 256;
const TEXT_LEN: usize = 128;
const TEMPORAL: usize = 8;
const SPATIAL: usize = 32;
/// `LTXVideoRotaryPosEmbed` (`base_num_frames`, `base_height`, `base_width`, `theta`).
const ROPE_BASE: [f64; 3] = [20.0, 2048.0, 2048.0];
const ROPE_THETA: f64 = 10000.0;
/// `LTXPipeline.__call__` default `frame_rate`.
pub const FPS: u32 = 25;
/// T5-XXL `t5encoder` GGUF used with GGUF transformers unless the caller names another text encoder.
// ponytail: Q4_K_M is a fixed quality/size pick; `text_encoder` overrides it with another T5 GGUF quant (a diffusers repo always uses its own `text_encoder/`).
const DEFAULT_T5_GGUF: &str = "city96/t5-v1_1-xxl-encoder-gguf/t5-v1_1-xxl-encoder-Q4_K_M.gguf";

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

struct Block {
    a1: Attn,
    a2: Attn,
    ff1: Lin,
    ff2: Lin,
    /// `[6, d]`: shift/scale/gate for attention, shift/scale/gate for the MLP.
    sst: Vec<f32>,
}

pub struct LtxTransformer {
    proj_in: Lin,
    t1: Lin,
    t2: Lin,
    ada: Lin,
    c1: Lin,
    c2: Lin,
    blocks: Vec<Block>,
    /// `[2, d]`: output shift, scale.
    sst: Vec<f32>,
    proj_out: Lin,
    d: usize,
    gpu: graph_program::Cache,
}

/// `x * rsqrt(mean(x^2) + eps) * w` per `w.len()`-wide row.
fn rms_rows(x: &mut [f32], w: &[f32], eps: f32) {
    par_rows(x, w.len(), &|_, r| {
        let inv = 1.0 / (r.iter().map(|v| v * v).sum::<f32>() / r.len() as f32 + eps).sqrt();
        r.iter_mut().zip(w).for_each(|(v, g)| *v *= inv * g);
    });
}

/// `norm(x) * (1 + scale) + shift` per row, `norm` = RMS (`layer == false`) or LayerNorm without affine.
fn norm_mod(x: &[f32], d: usize, shift: &[f32], scale: &[f32], layer: bool) -> Vec<f32> {
    let mut y = x.to_vec();
    par_rows(&mut y, d, &|_, r| {
        let mean = if layer {
            r.iter().sum::<f32>() / d as f32
        } else {
            0.0
        };
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for ((v, sh), sc) in r.iter_mut().zip(shift).zip(scale) {
            *v = (*v - mean) * inv * (1.0 + sc) + sh;
        }
    });
    y
}

/// `x += gate * y` per row.
fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32]) {
    for (xr, yr) in x
        .chunks_exact_mut(gate.len())
        .zip(y.chunks_exact(gate.len()))
    {
        for ((x, y), g) in xr.iter_mut().zip(yr).zip(gate) {
            *x += g * y;
        }
    }
}

/// Interleaved-pair rotary tables, `[tokens, d]` each.
pub struct Rope {
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
    d: usize,
}

impl Rope {
    /// `LTXVideoRotaryPosEmbed` for an `f x h x w` latent grid at `fps`: per axis `d / 6` frequencies
    /// `theta^(j / (n - 1)) * pi / 2` times `2 * pos - 1`, axes interleaved, each angle repeated for its
    /// (even, odd) pair, `d % 6` leading identity entries.
    pub fn new(f: usize, h: usize, w: usize, fps: f64, d: usize) -> Self {
        let n = d / 6;
        let freqs: Vec<f64> = (0..n)
            .map(|j| ROPE_THETA.powf(j as f64 / (n - 1) as f64) * std::f64::consts::FRAC_PI_2)
            .collect();
        let scale = [TEMPORAL as f64 / fps, SPATIAL as f64, SPATIAL as f64];
        let (mut cos, mut sin) = (
            Vec::with_capacity(f * h * w * d),
            Vec::with_capacity(f * h * w * d),
        );
        for t in 0..f * h * w {
            let pos = [t / (h * w), (t / w) % h, t % w];
            cos.extend(std::iter::repeat(1.0).take(d % 6));
            sin.extend(std::iter::repeat(0.0).take(d % 6));
            for &fr in &freqs {
                for axis in 0..3 {
                    let g = pos[axis] as f64 * scale[axis] / ROPE_BASE[axis];
                    let a = fr * (g * 2.0 - 1.0);
                    cos.extend([a.cos() as f32; 2]);
                    sin.extend([a.sin() as f32; 2]);
                }
            }
        }
        Self { cos, sin, d }
    }

    fn apply(&self, x: &mut [f32]) {
        let d = self.d;
        par_rows(x, d, &|t, row| {
            let (c, s) = (&self.cos[t * d..(t + 1) * d], &self.sin[t * d..(t + 1) * d]);
            for (i, p) in row.chunks_exact_mut(2).enumerate() {
                let (a, b) = (p[0], p[1]);
                p[0] = a * c[2 * i] - b * s[2 * i];
                p[1] = b * c[2 * i + 1] + a * s[2 * i + 1];
            }
        });
    }
}

impl LtxTransformer {
    pub fn load(s: &Store) -> Result<Self, String> {
        // diffusers renames the original ComfyUI/LTX names.
        let diffusers = s.has("proj_in.weight");
        let nm = |n: &str| {
            if diffusers {
                n.replace("patchify_proj", "proj_in")
                    .replace("adaln_single", "time_embed")
                    .replace("q_norm", "norm_q")
                    .replace("k_norm", "norm_k")
            } else {
                n.to_string()
            }
        };
        let lin = |n: &str| s.lin(&nm(n));
        let vec = |n: &str| s.vec(&nm(n));
        if !s.has(&format!("{}.weight", nm("patchify_proj"))) {
            return Err("not an LTX-Video transformer (no patchify_proj / proj_in weight)".into());
        }
        let proj_in = lin("patchify_proj")?;
        let (d, c_in) = proj_in.dims();
        if c_in != LATENT || d % HEADS != 0 {
            return Err(format!(
                "LTX-Video transformer: patchify_proj is {d}x{c_in}, expected d x {LATENT}"
            ));
        }
        let attn = |p: &str| -> Result<Attn, String> {
            Ok(Attn {
                q: lin(&format!("{p}.to_q"))?,
                k: lin(&format!("{p}.to_k"))?,
                v: lin(&format!("{p}.to_v"))?,
                o: lin(&format!("{p}.to_out.0"))?,
                qn: vec(&format!("{p}.q_norm.weight"))?,
                kn: vec(&format!("{p}.k_norm.weight"))?,
            })
        };
        let mut blocks = Vec::new();
        while s.has(&format!(
            "transformer_blocks.{}.attn1.to_q.weight",
            blocks.len()
        )) {
            let b = format!("transformer_blocks.{}", blocks.len());
            blocks.push(Block {
                a1: attn(&format!("{b}.attn1"))?,
                a2: attn(&format!("{b}.attn2"))?,
                ff1: lin(&format!("{b}.ff.net.0.proj"))?,
                ff2: lin(&format!("{b}.ff.net.2"))?,
                sst: vec(&format!("{b}.scale_shift_table"))?,
            });
        }
        let sst = vec("scale_shift_table")?;
        let t = Self {
            proj_in,
            t1: lin("adaln_single.emb.timestep_embedder.linear_1")?,
            t2: lin("adaln_single.emb.timestep_embedder.linear_2")?,
            ada: lin("adaln_single.linear")?,
            c1: lin("caption_projection.linear_1")?,
            c2: lin("caption_projection.linear_2")?,
            blocks,
            sst,
            proj_out: lin("proj_out")?,
            d,
            gpu: Default::default(),
        };
        if t.blocks.is_empty() || t.sst.len() != 2 * d || t.ada.dims() != (6 * d, d) {
            return Err(format!(
                "LTX-Video transformer: inconsistent shapes (d={d}, {} blocks)",
                t.blocks.len()
            ));
        }
        if let Some(b) = t.blocks.iter().find(|b| {
            b.sst.len() != 6 * d
                || b.a1.qn.len() != d
                || b.a2.kn.len() != d
                || b.ff1.dims().0 != 4 * d
        }) {
            return Err(format!(
                "LTX-Video transformer: block tensors disagree with d={d} ({} scale_shift)",
                b.sst.len()
            ));
        }
        Ok(t)
    }

    /// `PixArtAlphaTextProjection`: T5 tokens `[n, 4096]` -> `[n, d]` (linear, tanh-GELU, linear).
    pub fn project_text(&self, ctx: &[f32], n: usize) -> Vec<f32> {
        let mut h = self.c1.fwd(ctx, n);
        nn::gelu_tanh(&mut h);
        self.c2.fwd(&h, n)
    }

    fn attention(
        &self,
        a: &Attn,
        h: &[f32],
        n: usize,
        kv: &[f32],
        nk: usize,
        rope: Option<&Rope>,
    ) -> Vec<f32> {
        let mut q = a.q.fwd(h, n);
        let mut k = a.k.fwd(kv, nk);
        let v = a.v.fwd(kv, nk);
        rms_rows(&mut q, &a.qn, 1e-5);
        rms_rows(&mut k, &a.kn, 1e-5);
        if let Some(r) = rope {
            r.apply(&mut q);
            r.apply(&mut k);
        }
        a.o.fwd(
            &nn::attention(&q, &k, &v, n, nk, HEADS, self.d / HEADS, false),
            n,
        )
    }

    /// `(emb, temb)` of `AdaLayerNormSingle` at model timestep `t`: the `[d]` embedding and its `[6d]` projection.
    fn time_embed(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        let mut emb = self
            .t1
            .fwd(&crate::model::flux::sinusoid(t as f64, TIME_DIM), 1);
        nn::silu(&mut emb);
        let emb = self.t2.fwd(&emb, 1);
        let mut a = emb.clone();
        nn::silu(&mut a);
        let temb = self.ada.fwd(&a, 1);
        (emb, temb)
    }

    /// Velocity for latent tokens `x` `[n, 128]` at model timestep `t` (= sigma * 1000) with projected text
    /// `ctx` `[nt, d]`. The first `clean` tokens (an image-conditioned first latent frame) run at timestep 0,
    /// as `LTXImageToVideoPipeline` does with `timestep * (1 - conditioning_mask)`.
    pub fn forward(
        &self,
        x: &[f32],
        n: usize,
        ctx: &[f32],
        nt: usize,
        t: f32,
        clean: usize,
        rope: &Rope,
    ) -> Vec<f32> {
        let d = self.d;
        // Token ranges and the timestep each runs at.
        let spans: Vec<(usize, usize, f32)> = if clean == 0 {
            vec![(0, n, t)]
        } else {
            vec![(0, clean, 0.0), (clean, n, t)]
        };
        let times: Vec<(Vec<f32>, Vec<f32>)> = spans.iter().map(|s| self.time_embed(s.2)).collect();
        // `norm(x) * (1 + scale) + shift` with each range's own modulation chunks `(shift, scale)`.
        let modulate =
            |x: &[f32], m: &[Vec<f32>], (i, j): (usize, usize), layer: bool| -> Vec<f32> {
                spans
                    .iter()
                    .zip(m)
                    .flat_map(|(s, m)| {
                        norm_mod(
                            &x[s.0 * d..s.1 * d],
                            d,
                            &m[i * d..(i + 1) * d],
                            &m[j * d..(j + 1) * d],
                            layer,
                        )
                    })
                    .collect()
            };
        let gate = |x: &mut [f32], y: &[f32], m: &[Vec<f32>], i: usize| {
            for (s, m) in spans.iter().zip(m) {
                gated_add(
                    &mut x[s.0 * d..s.1 * d],
                    &y[s.0 * d..s.1 * d],
                    &m[i * d..(i + 1) * d],
                );
            }
        };
        let mut x_h = self.proj_in.fwd(x, n);
        for b in &self.blocks {
            let m: Vec<Vec<f32>> = times
                .iter()
                .map(|(_, te)| b.sst.iter().zip(te).map(|(s, t)| s + t).collect())
                .collect();
            let h = modulate(&x_h, &m, (0, 1), false);
            gate(
                &mut x_h,
                &self.attention(&b.a1, &h, n, &h, n, Some(rope)),
                &m,
                2,
            );
            // Cross-attention reads the raw residual stream (no modulation) and is added ungated.
            let y = self.attention(&b.a2, &x_h, n, ctx, nt, None);
            nn::add_in_place(&mut x_h, &y);
            let h = modulate(&x_h, &m, (3, 4), false);
            let mut f = b.ff1.fwd(&h, n);
            nn::gelu_tanh(&mut f);
            gate(&mut x_h, &b.ff2.fwd(&f, n), &m, 5);
        }
        let out: Vec<Vec<f32>> = times
            .iter()
            .map(|(e, _)| {
                self.sst
                    .iter()
                    .zip(e.iter().chain(e))
                    .map(|(s, e)| s + e)
                    .collect()
            })
            .collect();
        self.proj_out.fwd(&modulate(&x_h, &out, (0, 1), true), n)
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

/// T5-XXL prompt embeddings: only the real tokens, `[n, 4096]` (the padding is masked out of the
/// transformer's cross-attention, so it never needs to be kept).
pub struct Prompt {
    emb: Vec<f32>,
    n: usize,
}

pub struct LtxText {
    t5: T5Encoder,
    tok: Tokenizer,
}

impl LtxText {
    /// `text_encoder/` shards + `tokenizer/` of a diffusers repo snapshot.
    pub fn from_dir(base: &Path) -> Result<Self, String> {
        Ok(Self {
            t5: crate::model::flux::t5_load(&base.join("text_encoder"))?,
            tok: t5_tokenizer(&base.join("tokenizer"))?,
        })
    }

    /// A `t5encoder` GGUF with the base repo's `text_encoder/config.json` and `tokenizer/`.
    pub fn from_gguf(gguf: &Path, config: &Path, tokenizer_dir: &Path) -> Result<Self, String> {
        Ok(Self {
            t5: T5Encoder::load_gguf(&Store::open(gguf)?, &read_json(config)?)?,
            tok: t5_tokenizer(tokenizer_dir)?,
        })
    }

    /// `_get_t5_prompt_embeds`: ids padded with 0 to 128 and encoded with every token attending to every
    /// other (the pipeline passes no attention mask to the encoder), the real tokens kept.
    pub fn encode(&self, prompt: &str) -> Prompt {
        let mut ids = t5_ids(&self.tok, prompt, TEXT_LEN);
        let n = ids.len();
        ids.resize(TEXT_LEN, 0);
        let mut emb = self.t5.encode(&ids);
        emb.truncate(n * self.t5.d_model());
        Prompt { emb, n }
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

pub struct LtxPipeline {
    tr: LtxTransformer,
    vae: LtxVae,
    sched: sched::Schedule,
}

pub struct Clip {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
}

impl Clip {
    /// 25 frames (4 latent frames) at 512x320, 30 steps, guidance 3: `LTXPipeline`'s documented range at a size a
    /// CPU finishes in minutes.
    pub fn new(seed: u64) -> Self {
        Self {
            frames: 25,
            width: 512,
            height: 320,
            steps: 30,
            guidance: 3.0,
            seed,
        }
    }
}

impl LtxPipeline {
    pub fn load(
        transformer: &Path,
        vae: &Path,
        scheduler_config: &Path,
        encoder: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            tr: LtxTransformer::load(&Store::open(transformer)?)?,
            vae: LtxVae::load(vae, encoder)?,
            sched: {
                let s = sched::Schedule::from_config(&read_json(scheduler_config)?);
                if s.mu != sched::Mu::Dynamic {
                    return Err("LTX-Video scheduler_config.json: expected use_dynamic_shifting".into());
                }
                s
            },
        })
    }

    /// The pipeline's `check_inputs` + `prepare_latents` geometry: `(latent frames, rows, columns)`.
    pub fn grid(c: &Clip) -> Result<(usize, usize, usize), String> {
        if c.width == 0 || c.height == 0 || c.width % SPATIAL != 0 || c.height % SPATIAL != 0 {
            return Err(format!(
                "width and height must be multiples of {SPATIAL}, got {}x{}",
                c.width, c.height
            ));
        }
        if c.frames == 0 || (c.frames - 1) % TEMPORAL != 0 {
            return Err(format!(
                "--frames must be 8k+1 (9, 17, 25, ...), got {}",
                c.frames
            ));
        }
        if !(1..=200).contains(&c.steps) {
            return Err(format!("steps must be in 1..=200, got {}", c.steps));
        }
        Ok((
            (c.frames - 1) / TEMPORAL + 1,
            c.height / SPATIAL,
            c.width / SPATIAL,
        ))
    }

    /// Denoises `c.frames` frames; `uncond` is the negative prompt (CFG runs when `guidance > 1`). With
    /// `image` (`LTXImageToVideoPipeline`) its VAE latent is frame 0 of the clip, held clean at timestep 0.
    pub fn generate(
        &self,
        cond: &Prompt,
        uncond: Option<&Prompt>,
        c: &Clip,
        image: Option<&ImageBuffer>,
        log: &dyn Fn(&str),
    ) -> Result<Vec<ImageBuffer>, String> {
        let (f, h, w) = Self::grid(c)?;
        let n = f * h * w;
        let sigmas = self.sched.sigmas(&sched::linspace_sigmas(c.steps), n, c.steps);
        let rope = Rope::new(f, h, w, FPS as f64, self.tr.d);
        // The GPU program runs the caption projection itself and takes the raw T5 tokens.
        let gpu = crate::model::dit_graph::use_metal();
        let ctx = |p: &Prompt| {
            if gpu {
                p.emb.clone()
            } else {
                self.tr.project_text(&p.emb, p.n)
            }
        };
        let (cc, uc) = (ctx(cond), uncond.map(ctx));
        let mut rng = Rng::new(c.seed);
        let mut x: Vec<f32> = (0..n * LATENT).map(|_| rng.gaussian()).collect();
        let clean = match image {
            Some(img) => {
                let start = Instant::now();
                let img = img.resize_lanczos(c.width, c.height);
                let z = self.vae.encode_image(&img)?;
                log(&format!(
                    "  image encoded: {:.2}s",
                    start.elapsed().as_secs_f64()
                ));
                x[..z.len()].copy_from_slice(&z);
                z.len() / LATENT
            }
            None => 0,
        };
        for i in 0..c.steps {
            let start = Instant::now();
            let t = sched::timestep(sigmas[i]);
            let vel = |ctx: &[f32], nt: usize| {
                if gpu {
                    self.tr.forward_gpu(&x, n, ctx, nt, t, clean, &rope)
                } else {
                    Ok(self.tr.forward(&x, n, ctx, nt, t, clean, &rope))
                }
            };
            let mut v = vel(&cc, cond.n)?;
            if let (Some(u), Some(un)) = (&uc, uncond) {
                let vu = vel(u, un.n)?;
                for (c_, u_) in v.iter_mut().zip(&vu) {
                    *c_ = u_ + c.guidance * (*c_ - u_);
                }
            }
            let k = clean * LATENT;
            sched::euler_step(&mut x[k..], &v[k..], sigmas[i], sigmas[i + 1]);
            log(&format!(
                "  step {}/{} (sigma {:.3}): {:.2}s",
                i + 1,
                c.steps,
                sigmas[i],
                start.elapsed().as_secs_f64()
            ));
        }
        let start = Instant::now();
        let frames = self.vae.decode_tokens(&x, f, h, w)?;
        log(&format!(
            "  VAE decode: {:.2}s",
            start.elapsed().as_secs_f64()
        ));
        Ok(frames)
    }
}

// ---------------------------------------------------------------------------
// Repos
// ---------------------------------------------------------------------------

/// Whether a repo file name belongs to the v0.9.0 generation (`ltx-video-2b-v0.9-*.gguf`, `ltxv_0.9_vae_*`):
/// later versions (`v0.9.1`, `0.9.6`, `13b-0.9.7`, ...) pair with other VAE decoders (timestep conditioning,
/// noise injection, wider blocks) that are not implemented. Transformer GGUFs (`marker`) must carry the
/// version; the original VAE files (`LTX-Video-VAE-*`) have none.
fn is_v090(name: &str, marker: bool) -> bool {
    let n = name.to_lowercase();
    match n.find("0.9") {
        Some(i) => {
            !(n[i + 3..].starts_with('.') && n[i + 4..].starts_with(|c: char| c.is_ascii_digit()))
        }
        None => !marker && !n.contains("13b"),
    }
}

/// Files of an LTX-Video GGUF repo: the transformer GGUF (the hub's usual quant choice among the v0.9.0
/// ones) and, when the repo carries its own, the matching VAE. `Err` carries the reason a repo is refused.
pub struct RepoPick {
    pub transformer: String,
    pub vae: Option<String>,
}

pub fn pick_gguf(files: &[RepoFile], quant: Option<&str>) -> Result<RepoPick, String> {
    let base = |p: &str| p.rsplit('/').next().unwrap_or(p).to_lowercase();
    let ggufs: Vec<&RepoFile> = files
        .iter()
        .filter(|f| {
            let b = base(&f.path);
            b.ends_with(".gguf") && !b.contains("vae") && !b.contains("t5")
        })
        .collect();
    if ggufs.is_empty() {
        return Err("no LTX-Video transformer GGUF in the repo".into());
    }
    let ok: Vec<RepoFile> = ggufs
        .iter()
        .filter(|f| is_v090(&base(&f.path), true))
        .map(|f| (*f).clone())
        .collect();
    if ok.is_empty() {
        let names: Vec<String> = ggufs.iter().take(3).map(|f| f.path.clone()).collect();
        return Err(format!(
            "LTX-Video GGUF without a v0.9.0 transformer (found {}): v0.9.1+ need their own VAE decoder \
             (timestep conditioning, noise injection), which is not implemented; only v0.9.0 loads",
            names.join(", ")
        ));
    }
    let transformer = select_weight_files(&ok, quant)?.swap_remove(0);
    let mut vaes: Vec<&RepoFile> = files
        .iter()
        .filter(|f| {
            let b = base(&f.path);
            b.ends_with(".safetensors")
                && b.contains("vae")
                && !b.contains("fp8")
                && is_v090(&b, false)
        })
        .collect();
    vaes.sort_by(|a, b| (a.size, &a.path).cmp(&(b.size, &b.path)));
    Ok(RepoPick {
        transformer,
        vae: vaes.first().map(|f| f.path.clone()),
    })
}

/// Where each model of one LTX-Video generation comes from.
#[derive(Clone)]
pub struct Plan {
    base: String,
    /// `Some((repo, file))`: a GGUF; `None`: the base repo's `transformer/`.
    gguf: Option<(String, String)>,
    vae: (String, String),
    label: String,
}

impl Plan {
    /// `repo/file.gguf` or `repo (diffusers)`.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// The loaded LTX-Video repo the TUI's `/video` and `/i2v` generate with.
pub type VideoSlot =
    std::sync::Arc<parking_lot::RwLock<Option<std::sync::Arc<(HuggingFaceHub, Plan)>>>>;

fn class_of(hub: &HuggingFaceHub, repo: &str) -> Option<String> {
    let p = hub.fetch_file(repo, "model_index.json").ok()?;
    read_json(&p).ok()?["_class_name"]
        .as_str()
        .map(str::to_string)
}

/// `Some` when `model_id` names an LTX-Video repo: the diffusers `LTXPipeline` repo itself, or a GGUF repo
/// (hub arch `ltxv`) whose card names such a base. Nothing is downloaded but the card and model index.
pub fn resolve(
    hub: &HuggingFaceHub,
    model_id: &str,
    quant: Option<&str>,
) -> Option<Result<Plan, String>> {
    let (repo, file) = split_weight_file(model_id);
    if repo.split('/').count() != 2 || is_synthetic_model_id(repo) || Path::new(model_id).exists() {
        return None;
    }
    let token = std::env::var("HF_TOKEN").ok();
    let info = hub_json(
        &format!("https://huggingface.co/api/models/{repo}"),
        token.as_deref(),
    )
    .ok()?;
    let class = info["config"]["diffusers"]["_class_name"].as_str();
    if matches!(class, Some("LTXPipeline" | "LTXImageToVideoPipeline")) {
        return Some(Ok(Plan {
            base: repo.into(),
            gguf: None,
            vae: (
                repo.into(),
                "vae/diffusion_pytorch_model.safetensors".into(),
            ),
            label: format!("{repo} (diffusers)"),
        }));
    }
    if info["gguf"]["architecture"] != "ltxv" {
        return None;
    }
    Some((|| {
        let files = list_repo_files(repo, token.as_deref())?;
        let mut pick = pick_gguf(&files, quant)?;
        if let Some(f) = file {
            if !files.iter().any(|x| x.path == f) {
                return Err(format!("{repo} has no file {f}"));
            }
            if !is_v090(f, true) {
                return Err(format!("{f}: only v0.9.0 LTX-Video GGUFs load (later versions need other VAE decoders)"));
            }
            pick.transformer = f.to_string();
        }
        let card = hub.fetch_file(repo, "README.md")?;
        let card =
            std::fs::read_to_string(&card).map_err(|e| format!("{}: {e}", card.display()))?;
        let base = crate::model::lora::card_base_model(&card).ok_or_else(|| {
            format!("{repo}: LTX-Video GGUF whose model card names no base_model")
        })?;
        if class_of(hub, &base).as_deref() != Some("LTXPipeline") {
            return Err(format!(
                "{repo}: base model {base} is not an LTXPipeline repo"
            ));
        }
        let vae = match pick.vae {
            Some(f) => (repo.to_string(), f),
            None => (
                base.clone(),
                "vae/diffusion_pytorch_model.safetensors".into(),
            ),
        };
        Ok(Plan {
            label: format!("{repo}/{}", pick.transformer),
            gguf: Some((repo.to_string(), pick.transformer)),
            base,
            vae,
        })
    })())
}

/// One whole text-to-video (or, with `image`, image-to-video) run. The T5 comes from `text_encoder` (a
/// `t5encoder` GGUF repo or `repo/file.gguf`) when given, else the base repo's `text_encoder/` shards.
pub fn run(
    hub: &HuggingFaceHub,
    plan: &Plan,
    prompt: &str,
    negative: &str,
    text_encoder: Option<&str>,
    image: Option<&ImageBuffer>,
    c: &Clip,
    log: &dyn Fn(&str),
) -> Result<Vec<ImageBuffer>, String> {
    LtxPipeline::grid(c)?;
    let fetch = |repo: &str, file: &str| -> Result<PathBuf, String> {
        log(&format!("Fetching {repo}/{file} (reused if cached)"));
        hub.fetch_file(repo, file)
            .map_err(|e| format!("{repo}/{file}: {e}"))
    };
    let t0 = Instant::now();
    let sched = fetch(&plan.base, "scheduler/scheduler_config.json")?;
    // A quantized transformer defaults to the quantized T5 (2.9 GB) rather than the base repo's 19 GB fp32 shards.
    let text_encoder = text_encoder.or(plan.gguf.as_ref().map(|_| DEFAULT_T5_GGUF));
    let (cond, uncond) = {
        let text = match text_encoder {
            Some(id) => {
                let (repo, file) = split_weight_file(id);
                let file = match file {
                    Some(f) => f.to_string(),
                    None => {
                        let files =
                            list_repo_files(repo, std::env::var("HF_TOKEN").ok().as_deref())?;
                        select_weight_files(&files, None)?.swap_remove(0)
                    }
                };
                let gguf = fetch(repo, &file)?;
                let config = fetch(&plan.base, "text_encoder/config.json")?;
                let spiece = fetch(&plan.base, "tokenizer/spiece.model")?;
                LtxText::from_gguf(&gguf, &config, spiece.parent().ok_or("tokenizer path")?)?
            }
            None => {
                let dir = hub.fetch_diffusers(&plan.base, &["transformer", "vae"])?;
                LtxText::from_dir(&dir)?
            }
        };
        log(&format!(
            "T5-XXL loaded in {:.2}s",
            t0.elapsed().as_secs_f64()
        ));
        let t1 = Instant::now();
        let cond = text.encode(prompt);
        let uncond = (c.guidance > 1.0).then(|| text.encode(negative));
        log(&format!(
            "Prompt encoded ({} tokens) in {:.2}s; text encoder released",
            cond.n,
            t1.elapsed().as_secs_f64()
        ));
        (cond, uncond)
    };
    let tr = match &plan.gguf {
        Some((repo, file)) => fetch(repo, file)?,
        None => {
            let dir = hub.fetch_diffusers(&plan.base, &["text_encoder", "vae"])?;
            component_weights(&dir.join("transformer"))?
        }
    };
    let vae = fetch(&plan.vae.0, &plan.vae.1)?;
    let t2 = Instant::now();
    let pipe = LtxPipeline::load(&tr, &vae, &sched, image.is_some())?;
    log(&format!(
        "LTX-Video {} loaded in {:.2}s",
        plan.label,
        t2.elapsed().as_secs_f64()
    ));
    pipe.generate(&cond, uncond.as_ref(), c, image, log)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> Vec<RepoFile> {
        names
            .iter()
            .map(|n| RepoFile {
                path: n.to_string(),
                size: None,
            })
            .collect()
    }

    #[test]
    fn only_v090_files_are_picked() {
        let p = pick_gguf(
            &files(&[
                "ltx-video-2b-v0.9-Q8_0.gguf",
                "ltx-video-2b-v0.9-Q4_0.gguf",
                "ltx-video-2b-v0.9.1-q4_0.gguf",
                "ltxv-13b-0.9.7-dev-q4_0.gguf",
                "LTX-Video-VAE-BF16.safetensors",
                "ltxv_0.9.5_vae_fp16.safetensors",
            ]),
            None,
        )
        .unwrap();
        assert_eq!(p.transformer, "ltx-video-2b-v0.9-Q4_0.gguf");
        assert_eq!(p.vae.as_deref(), Some("LTX-Video-VAE-BF16.safetensors"));
        let e = pick_gguf(
            &files(&[
                "ltxv-13b-0.9.7-dev-q4_0.gguf",
                "ltxv-2b-0.9.6-dev-q4_0.gguf",
            ]),
            None,
        )
        .err()
        .unwrap();
        assert!(e.contains("13b-0.9.7") && e.contains("v0.9.0"), "{e}");
    }

    #[test]
    fn rope_pads_and_pairs_angles() {
        let r = Rope::new(2, 2, 2, 25.0, 2048);
        assert_eq!(r.cos.len(), 8 * 2048);
        assert_eq!(
            (r.cos[0], r.sin[0], r.cos[1], r.sin[1]),
            (1.0, 0.0, 1.0, 0.0)
        );
        // Token 0 sits at the grid origin: every angle is freq * (0 * 2 - 1); pairs share one angle.
        assert_eq!(r.cos[2], r.cos[3]);
        let first = std::f64::consts::FRAC_PI_2;
        assert!(
            (r.cos[2] as f64 - first.cos()).abs() < 1e-6
                && (r.sin[2] as f64 + first.sin()).abs() < 1e-6
        );
        let mut x = vec![1.0f32; 8 * 2048];
        r.apply(&mut x);
        assert_eq!(&x[..2], &[1.0, 1.0]);
    }

    /// Writes F32 tensors as a safetensors file.
    fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, shape, v) in tensors {
            let start = data.len();
            v.iter().for_each(|x| data.extend(x.to_le_bytes()));
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend(data);
        std::fs::write(path, file).unwrap();
    }

    /// A 2-block, d = 64 LTX transformer under the original (ComfyUI/GGUF) tensor names and the diffusers
    /// ones gives the same velocity; running every token at timestep 0 through the image-conditioning split
    /// equals a plain run at t = 0.
    #[test]
    fn original_and_diffusers_names_agree_and_clean_split_matches_t0() {
        let d = 64;
        let mut rng = Rng::new(11);
        let mut w = |name: &str, shape: &[usize]| -> (String, Vec<usize>, Vec<f32>) {
            let n: usize = shape.iter().product();
            let scale = if name.ends_with("norm.weight") {
                0.0
            } else {
                0.05
            };
            let base = if name.ends_with("norm.weight") {
                1.0
            } else {
                0.0
            };
            (
                name.to_string(),
                shape.to_vec(),
                (0..n).map(|_| base + scale * rng.gaussian()).collect(),
            )
        };
        let mut t = vec![
            w("patchify_proj.weight", &[d, LATENT]),
            w("patchify_proj.bias", &[d]),
            w(
                "adaln_single.emb.timestep_embedder.linear_1.weight",
                &[d, TIME_DIM],
            ),
            w("adaln_single.emb.timestep_embedder.linear_1.bias", &[d]),
            w(
                "adaln_single.emb.timestep_embedder.linear_2.weight",
                &[d, d],
            ),
            w("adaln_single.emb.timestep_embedder.linear_2.bias", &[d]),
            w("adaln_single.linear.weight", &[6 * d, d]),
            w("adaln_single.linear.bias", &[6 * d]),
            w("caption_projection.linear_1.weight", &[d, 4096]),
            w("caption_projection.linear_1.bias", &[d]),
            w("caption_projection.linear_2.weight", &[d, d]),
            w("caption_projection.linear_2.bias", &[d]),
            w("scale_shift_table", &[2, d]),
            w("proj_out.weight", &[LATENT, d]),
            w("proj_out.bias", &[LATENT]),
        ];
        for b in 0..2 {
            for a in ["attn1", "attn2"] {
                for p in ["to_q", "to_k", "to_v", "to_out.0"] {
                    t.push(w(
                        &format!("transformer_blocks.{b}.{a}.{p}.weight"),
                        &[d, d],
                    ));
                    t.push(w(&format!("transformer_blocks.{b}.{a}.{p}.bias"), &[d]));
                }
                for p in ["q_norm", "k_norm"] {
                    t.push(w(&format!("transformer_blocks.{b}.{a}.{p}.weight"), &[d]));
                }
            }
            t.push(w(
                &format!("transformer_blocks.{b}.ff.net.0.proj.weight"),
                &[4 * d, d],
            ));
            t.push(w(
                &format!("transformer_blocks.{b}.ff.net.0.proj.bias"),
                &[4 * d],
            ));
            t.push(w(
                &format!("transformer_blocks.{b}.ff.net.2.weight"),
                &[d, 4 * d],
            ));
            t.push(w(&format!("transformer_blocks.{b}.ff.net.2.bias"), &[d]));
            t.push(w(
                &format!("transformer_blocks.{b}.scale_shift_table"),
                &[6, d],
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        let (orig, diff) = (
            dir.path().join("orig.safetensors"),
            dir.path().join("diff.safetensors"),
        );
        write_safetensors(&orig, &t);
        let renamed: Vec<_> = t
            .iter()
            .map(|(n, s, v)| {
                let n = n
                    .replace("patchify_proj", "proj_in")
                    .replace("adaln_single", "time_embed")
                    .replace("q_norm", "norm_q")
                    .replace("k_norm", "norm_k");
                (n, s.clone(), v.clone())
            })
            .collect();
        write_safetensors(&diff, &renamed);
        let (a, b) = (
            LtxTransformer::load(&Store::open(&orig).unwrap()).unwrap(),
            LtxTransformer::load(&Store::open(&diff).unwrap()).unwrap(),
        );
        let (f, h, wd) = (2, 2, 3);
        let n = f * h * wd;
        let x: Vec<f32> = (0..n * LATENT).map(|_| rng.gaussian()).collect();
        let text: Vec<f32> = (0..5 * 4096).map(|_| rng.gaussian()).collect();
        let rope = Rope::new(f, h, wd, FPS as f64, d);
        let (ca, cb) = (a.project_text(&text, 5), b.project_text(&text, 5));
        let (va, vb) = (
            a.forward(&x, n, &ca, 5, 700.0, 0, &rope),
            b.forward(&x, n, &cb, 5, 700.0, 0, &rope),
        );
        assert_eq!(va, vb);
        assert!(va.iter().all(|v| v.is_finite()) && va.iter().any(|v| v.abs() > 1e-6));
        // All tokens clean: the split path runs every token at timestep 0 (the `clean == n` single range).
        let split = a.forward(&x, n, &ca, 5, 700.0, n, &rope);
        let plain = a.forward(&x, n, &ca, 5, 0.0, 0, &rope);
        let err = split
            .iter()
            .zip(&plain)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-5, "split vs t=0: {err}");
        // A partial split really differs from the unconditioned run on the noisy tokens.
        let part = a.forward(&x, n, &ca, 5, 700.0, 4, &rope);
        assert!(part.iter().zip(&va).any(|(p, q)| (p - q).abs() > 1e-5));
    }

    fn cached(hub: &HuggingFaceHub, repo: &str, file: &str) -> Option<PathBuf> {
        let p = hub
            .cache_root
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots/main")
            .join(file);
        p.is_file().then_some(p)
    }

    fn mean_pool(p: &Prompt) -> Vec<f32> {
        let d = p.emb.len() / p.n;
        let mut m = vec![0.0f32; d];
        p.emb
            .chunks_exact(d)
            .for_each(|r| m.iter_mut().zip(r).for_each(|(a, b)| *a += b / p.n as f32));
        m
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt()
            * b.iter().map(|x| x * x).sum::<f32>().sqrt())
    }

    /// The Q4_K_M T5-XXL `t5encoder` GGUF gives finite embeddings in which a paraphrase is nearer than an
    /// unrelated sentence (no reference encoder is available to compare tensors against).
    #[test]
    fn t5_gguf_embeddings_when_cached() {
        let hub = HuggingFaceHub::new();
        let Some(gguf) = cached(
            &hub,
            "city96/t5-v1_1-xxl-encoder-gguf",
            "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
        ) else {
            println!("SKIP: city96/t5-v1_1-xxl-encoder-gguf Q4_K_M is not cached");
            return;
        };
        let base = "Lightricks/LTX-Video";
        let (Ok(cfg), Ok(spiece)) = (
            hub.fetch_file(base, "text_encoder/config.json"),
            hub.fetch_file(base, "tokenizer/spiece.model"),
        ) else {
            println!("SKIP: {base} tokenizer/config not reachable");
            return;
        };
        let text = LtxText::from_gguf(&gguf, &cfg, spiece.parent().unwrap()).unwrap();
        let [a, b, c] = [
            "a cat sits on a sofa",
            "a kitten rests on a couch",
            "quarterly stock market earnings report",
        ]
        .map(|p| text.encode(p));
        assert!(a
            .emb
            .iter()
            .chain(&b.emb)
            .chain(&c.emb)
            .all(|v| v.is_finite()));
        assert_eq!(a.emb.len(), a.n * 4096);
        let (ab, ac) = (
            cosine(&mean_pool(&a), &mean_pool(&b)),
            cosine(&mean_pool(&a), &mean_pool(&c)),
        );
        println!(
            "T5 GGUF tokens {} {} {}; cosine(cat, kitten) {ab:.3} vs cosine(cat, stocks) {ac:.3}",
            a.n, b.n, c.n
        );
        assert!(ab > ac, "{ab} <= {ac}");
    }

    /// The cached v0.9.0 transformer GGUF + VAE on a fixed pseudo-random prompt embedding (no T5 needed):
    /// finite, non-degenerate frames of the requested size.
    #[test]
    fn denoise_and_decode_when_cached() {
        let hub = HuggingFaceHub::new();
        let (Some(tr), Some(vae), Some(sched)) = (
            cached(&hub, "city96/LTX-Video-gguf", "ltx-video-2b-v0.9-Q4_0.gguf"),
            cached(
                &hub,
                "city96/LTX-Video-gguf",
                "LTX-Video-VAE-BF16.safetensors",
            ),
            cached(
                &hub,
                "Lightricks/LTX-Video",
                "scheduler/scheduler_config.json",
            ),
        ) else {
            println!(
                "SKIP: city96 LTX-Video Q4_0 GGUF, its VAE or the scheduler config is not cached"
            );
            return;
        };
        let pipe = LtxPipeline::load(&tr, &vae, &sched, false).unwrap();
        let mut rng = Rng::new(7);
        let prompt = Prompt {
            emb: (0..12 * 4096).map(|_| rng.gaussian()).collect(),
            n: 12,
        };
        let clip = Clip {
            frames: 9,
            width: 256,
            height: 160,
            steps: 3,
            guidance: 1.0,
            seed: 1,
        };
        let frames = pipe
            .generate(&prompt, None, &clip, None, &|s| println!("{s}"))
            .unwrap();
        assert_eq!(
            (frames.len(), frames[0].width, frames[0].height),
            (9, 256, 160)
        );
        let mean =
            frames[0].data.iter().map(|&v| v as f64).sum::<f64>() / frames[0].data.len() as f64;
        let var = frames[0]
            .data
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / frames[0].data.len() as f64;
        println!("frame 0 mean {mean:.1} std {:.1}", var.sqrt());
        assert!(var.sqrt() > 1.0, "constant frame");
        if let Ok(dir) = std::env::var("GIBBS_TEST_OUT") {
            crate::model::t2v::write_mp4(&frames, FPS, &Path::new(&dir).join("denoise.mp4"))
                .unwrap();
        }
    }

    /// A 9-frame 256x160 clip through the cached city96 v0.9.0 GGUF, its VAE and the T5 GGUF.
    #[test]
    fn real_weights_text_to_video_when_cached() {
        let hub = HuggingFaceHub::new();
        let ready = cached(
            &hub,
            "city96/t5-v1_1-xxl-encoder-gguf",
            "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
        )
        .is_some()
            && cached(&hub, "city96/LTX-Video-gguf", "ltx-video-2b-v0.9-Q4_0.gguf").is_some()
            && cached(
                &hub,
                "city96/LTX-Video-gguf",
                "LTX-Video-VAE-BF16.safetensors",
            )
            .is_some();
        let Some(Ok(plan)) = ready
            .then(|| resolve(&hub, "city96/LTX-Video-gguf", None))
            .flatten()
        else {
            println!("SKIP: the city96 LTX-Video GGUF, its VAE and the T5 GGUF are not all cached");
            return;
        };
        let clip = Clip {
            frames: 9,
            width: 256,
            height: 160,
            steps: 4,
            guidance: 3.0,
            seed: 1,
        };
        let frames = run(
            &hub,
            &plan,
            "a red balloon rising over a green field",
            "",
            Some("city96/t5-v1_1-xxl-encoder-gguf"),
            None,
            &clip,
            &|s| println!("{s}"),
        )
        .unwrap();
        assert_eq!(
            (frames.len(), frames[0].width, frames[0].height),
            (9, 256, 160)
        );
        let mean =
            |f: &ImageBuffer| f.data.iter().map(|&v| v as f64).sum::<f64>() / f.data.len() as f64;
        assert!(
            (5.0..250.0).contains(&mean(&frames[0])),
            "degenerate frame: mean {}",
            mean(&frames[0])
        );
    }
}
