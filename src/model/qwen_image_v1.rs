//! Qwen-Image 1 (diffusers `QwenImagePipeline`, e.g. `Qwen/Qwen-Image`, their GGUFs and finetunes)
//! on the CPU: the Qwen2.5-VL text tower (last hidden state, after its final norm, of the
//! system-prompt template without its first 34 tokens), the 60-block double-stream MMDiT
//! `QwenImageTransformer2DModel` (separate text and image streams, joint attention, 2x2-packed
//! latents), FlowMatchEuler with an exponential dynamic shift and true CFG with norm rescale,
//! and the 16-channel 8x VAE ([`crate::model::qwen_vae`]).
//! Text-to-image only: condition images, guidance-distilled variants and padding masks (one prompt
//! per pass, so no padding) are not implemented.

mod graph_program;

use crate::model::dit::{
    add_gated, norm_modulate, read_config, read_scheduler, rms_norm, silu_of, timestep_features,
};
use crate::runtime::sched;
use crate::model::flux::Store;
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::model::qwen3_5::Qwen3Encoder;
use crate::model::qwen_image::QLin;
use crate::model::qwen_vae::QwenVae;
use crate::runtime::sched::Rng;
use crate::model::text_to_image::{ImageDefaults, ImageGenerator, ImageRequest};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

/// The pipeline's prompt template around the user prompt.
const TEMPLATE_HEAD: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
const TEMPLATE_TAIL: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Template tokens (`prompt_template_encode_start_idx`) dropped from the hidden states.
const DROP: usize = 34;
/// `tokenizer_max_length`: prompt tokens kept after the dropped ones.
const MAX_TOKENS: usize = 1024;
const THETA: f64 = 10000.0;
/// Pixels per latent side (8x VAE).
const VAE_PX: usize = 8;
/// VAE latent channels.
const Z: usize = 16;
/// Qwen2 pre-tokenizer split (GPT-4 style).
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A Linear layer with bias, weights left in their stored encoding.
struct BLin {
    w: QLin,
    b: Vec<f32>,
}

impl BLin {
    fn load(s: &Store, name: &str) -> Result<Self, String> {
        let w = QLin::load(s, &format!("{name}.weight"))?;
        let b = s.vec(&format!("{name}.bias"))?;
        if b.len() != w.n {
            return Err(format!(
                "{name}: bias has {} entries for {} outputs",
                b.len(),
                w.n
            ));
        }
        Ok(Self { w, b })
    }

    /// x `[m, k]` -> `[m, n]`.
    fn fwd(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut y = self.w.fwd(x, m);
        par_rows(&mut y, self.b.len(), &|_, row| {
            row.iter_mut().zip(&self.b).for_each(|(v, b)| *v += b)
        });
        y
    }
}

/// One stream (image or text) of a double-stream block.
struct Stream {
    modulation: BLin,
    q: BLin,
    k: BLin,
    v: BLin,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    o: BLin,
    mlp_in: BLin,
    mlp_out: BLin,
}

struct Block {
    img: Stream,
    txt: Stream,
}

/// A stream's modulation `[shift1 | scale1 | gate1 | shift2 | scale2 | gate2]` (gates not `tanh`ed).
struct Mods(Vec<f32>);

const SHIFT1: usize = 0;
const SCALE1: usize = 1;
const GATE1: usize = 2;
const SHIFT2: usize = 3;
const SCALE2: usize = 4;
const GATE2: usize = 5;

impl Mods {
    fn part(&self, i: usize) -> &[f32] {
        let d = self.0.len() / 6;
        &self.0[i * d..(i + 1) * d]
    }

    /// `LayerNorm(x) * (1 + scale) + shift` of x `[rows, d]`.
    fn modulate(&self, x: &[f32], d: usize, shift: usize, scale: usize) -> Vec<f32> {
        norm_modulate(x, d, 1e-6, self.part(scale), self.part(shift))
    }
}

/// Rotary tables `[n, head_dim / 2]` over the joint sequence (text, then the image).
pub struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// `QwenEmbedRope` positions `(frame, height, width)` of a text prompt of `n_text` tokens and one
/// `h x w` grid of packed patches (`scale_rope`): text sits on all three axes after the grid's
/// larger half-side; the image is frame 0 on a grid centred on zero (negative positions are real
/// angles).
pub fn positions(n_text: usize, h: usize, w: usize) -> Vec<[i32; 3]> {
    let m = (h / 2).max(w / 2) as i32;
    let mut pos: Vec<[i32; 3]> = (0..n_text as i32).map(|t| [m + t; 3]).collect();
    let (oy, ox) = ((h - h / 2) as i32, (w - w / 2) as i32);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            pos.push([0, y - oy, x - ox]);
        }
    }
    pos
}

impl Rope {
    pub fn new(pos: &[[i32; 3]], axes: &[usize; 3]) -> Self {
        let half: usize = axes.iter().map(|a| a / 2).sum();
        let freqs: Vec<(usize, f32)> = axes
            .iter()
            .enumerate()
            .flat_map(|(a, &d)| {
                (0..d / 2).map(move |i| (a, (1.0 / THETA.powf(2.0 * i as f64 / d as f64)) as f32))
            })
            .collect();
        let (mut cos, mut sin) = (
            vec![0.0f32; pos.len() * half],
            vec![0.0f32; pos.len() * half],
        );
        for (r, p) in pos.iter().enumerate() {
            for (k, &(a, f)) in freqs.iter().enumerate() {
                let ang = p[a] as f32 * f;
                cos[r * half + k] = ang.cos();
                sin[r * half + k] = ang.sin();
            }
        }
        Self { cos, sin }
    }

    /// Rotates interleaved pairs of every head of x `[n, heads * head_dim]` (`apply_rotary_emb_qwen`).
    fn apply(&self, x: &mut [f32], width: usize, head_dim: usize) {
        let half = head_dim / 2;
        par_rows(x, width, &|r, row| {
            let (c, s) = (&self.cos[r * half..][..half], &self.sin[r * half..][..half]);
            for head in row.chunks_exact_mut(head_dim) {
                for k in 0..half {
                    let (a, b) = (head[2 * k], head[2 * k + 1]);
                    head[2 * k] = a * c[k] - b * s[k];
                    head[2 * k + 1] = a * s[k] + b * c[k];
                }
            }
        });
    }
}

/// `AdaLayerNormContinuous` (affine-free LayerNorm): `emb = [scale | shift]`, `LN(x) * (1 + scale) + shift`.
fn ada_norm_out(x: &[f32], d: usize, emb: &[f32]) -> Vec<f32> {
    norm_modulate(x, d, 1e-6, &emb[..d], &emb[d..])
}

pub struct Dit {
    dim: usize,
    heads: usize,
    head_dim: usize,
    /// Packed latent features per image token (`Z * 4`).
    in_ch: usize,
    pub axes: [usize; 3],
    img_in: BLin,
    txt_norm: Vec<f32>,
    txt_in: BLin,
    t_lin1: BLin,
    t_lin2: BLin,
    blocks: Vec<Block>,
    norm_out: BLin,
    proj_out: BLin,
    /// Metal programs per (text tokens, ph, pw) (GPU-only runs).
    pub gpu: graph_program::Cache,
}

impl Dit {
    /// Reads the transformer (diffusers names); sizes come from the tensors. `cfg` (a diffusers
    /// `config.json`, absent for GGUFs) only supplies the rotary axes and the unsupported options.
    pub fn load(s: &Store, cfg: Option<&Value>) -> Result<Self, String> {
        let null = Value::Null;
        let cfg = cfg.unwrap_or(&null);
        for k in [
            "guidance_embeds",
            "zero_cond_t",
            "use_additional_t_cond",
            "use_layer3d_rope",
        ] {
            if cfg[k] == true {
                return Err(format!("Qwen-Image 1: `{k}: true` is not implemented"));
            }
        }
        if cfg["patch_size"].as_u64().is_some_and(|p| p != 2) {
            return Err("Qwen-Image 1: only patch_size 2 is supported".into());
        }
        for (name, what) in [
            (
                "time_text_embed.guidance_embedder.linear_1.weight",
                "guidance_embeds",
            ),
            (
                "time_text_embed.addition_t_embedding.weight",
                "use_additional_t_cond",
            ),
        ] {
            if s.has(name) {
                return Err(format!(
                    "Qwen-Image 1: the weights use {what} ('{name}'), which is not implemented"
                ));
            }
        }
        let layers = (0..)
            .take_while(|i| s.has(&format!("transformer_blocks.{i}.attn.to_q.weight")))
            .count();
        if layers == 0 {
            return Err("Qwen-Image 1 transformer: no transformer_blocks".into());
        }
        let img_in = BLin::load(s, "img_in")?;
        let dim = img_in.w.n;
        let head_dim = s.vec("transformer_blocks.0.attn.norm_q.weight")?.len();
        if head_dim == 0 || dim % head_dim != 0 {
            return Err(format!(
                "Qwen-Image 1: dim {dim} is not a multiple of head_dim {head_dim}"
            ));
        }
        let axes: Vec<usize> = cfg["axes_dims_rope"]
            .as_array()
            .map_or(vec![16, 56, 56], |a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            });
        let axes: [usize; 3] = axes
            .try_into()
            .map_err(|_| "axes_dims_rope must have 3 entries")?;
        if axes.iter().sum::<usize>() != head_dim || axes.iter().any(|a| a % 2 != 0) {
            return Err(format!("Qwen-Image 1: axes_dims_rope {axes:?} must be 3 even sizes summing to head_dim {head_dim}"));
        }
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("transformer_blocks.{i}");
            let stream = |modulation: &str,
                          proj: [&str; 3],
                          norm: [&str; 2],
                          out: &str,
                          mlp: &str|
             -> Result<Stream, String> {
                let lin = |n: &str| BLin::load(s, &format!("{p}.{n}"));
                let vec = |n: &str| s.vec(&format!("{p}.{n}.weight"));
                Ok(Stream {
                    modulation: lin(modulation)?,
                    q: lin(proj[0])?,
                    k: lin(proj[1])?,
                    v: lin(proj[2])?,
                    norm_q: vec(norm[0])?,
                    norm_k: vec(norm[1])?,
                    o: lin(out)?,
                    mlp_in: lin(&format!("{mlp}.net.0.proj"))?,
                    mlp_out: lin(&format!("{mlp}.net.2"))?,
                })
            };
            blocks.push(Block {
                img: stream(
                    "img_mod.1",
                    ["attn.to_q", "attn.to_k", "attn.to_v"],
                    ["attn.norm_q", "attn.norm_k"],
                    "attn.to_out.0",
                    "img_mlp",
                )?,
                txt: stream(
                    "txt_mod.1",
                    ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                    ["attn.norm_added_q", "attn.norm_added_k"],
                    "attn.to_add_out",
                    "txt_mlp",
                )?,
            });
        }
        let dit = Self {
            dim,
            heads: dim / head_dim,
            head_dim,
            in_ch: img_in.w.k,
            axes,
            img_in,
            txt_norm: s.vec("txt_norm.weight")?,
            txt_in: BLin::load(s, "txt_in")?,
            t_lin1: BLin::load(s, "time_text_embed.timestep_embedder.linear_1")?,
            t_lin2: BLin::load(s, "time_text_embed.timestep_embedder.linear_2")?,
            blocks,
            norm_out: BLin::load(s, "norm_out.linear")?,
            proj_out: BLin::load(s, "proj_out")?,
            gpu: Default::default(),
        };
        let streams_ok = dit.blocks.iter().all(|b| {
            [&b.img, &b.txt].iter().all(|s| {
                s.modulation.w.n == 6 * dim
                    && [&s.q, &s.k, &s.v]
                        .iter()
                        .all(|l| l.w.n == dim && l.w.k == dim)
                    && s.o.w.k == dim
                    && s.o.w.n == dim
                    && s.norm_q.len() == head_dim
                    && s.norm_k.len() == head_dim
                    && s.mlp_in.w.k == dim
                    && s.mlp_out.w.n == dim
                    && s.mlp_in.w.n == s.mlp_out.w.k
            })
        });
        if !streams_ok
            || dit.t_lin1.w.k != 256
            || dit.t_lin1.w.n != dim
            || dit.t_lin2.w.n != dim
            || dit.txt_in.w.n != dim
            || dit.txt_in.w.k != dit.txt_norm.len()
            || dit.norm_out.w.n != 2 * dim
            || dit.proj_out.w.k != dim
        {
            return Err("Qwen-Image 1: tensor shapes do not match the transformer width".into());
        }
        Ok(dit)
    }

    fn text_width(&self) -> usize {
        self.txt_norm.len()
    }

    /// `txt_norm` (RMSNorm with an affine weight) and `txt_in` of text features `[n, text_width]`.
    pub fn project_text(&self, txt: &[f32], n: usize) -> Vec<f32> {
        let mut t = txt.to_vec();
        rms_norm(&mut t, self.text_width(), &self.txt_norm, 1e-6);
        self.txt_in.fwd(&t, n)
    }

    /// Timestep embedding of `sigma` (`Timesteps(scale = 1000)`).
    fn temb(&self, sigma: f32) -> Vec<f32> {
        let mut h = self.t_lin1.fwd(&timestep_features(1000.0 * sigma, 256), 1);
        nn::silu(&mut h);
        self.t_lin2.fwd(&h, 1)
    }

    /// Joint-attention q/k (QK-normed) and v of one stream's modulated rows `[rows, dim]`.
    fn qkv(&self, s: &Stream, h: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut q, mut k, v) = (s.q.fwd(h, rows), s.k.fwd(h, rows), s.v.fwd(h, rows));
        rms_norm(&mut q, self.head_dim, &s.norm_q, 1e-6);
        rms_norm(&mut k, self.head_dim, &s.norm_k, 1e-6);
        (q, k, v)
    }

    /// `x += gate2 * mlp(modulate(LN(x)))`.
    fn mlp(&self, s: &Stream, x: &mut [f32], m: &Mods) {
        let (d, rows) = (self.dim, x.len() / self.dim);
        let h = m.modulate(x, d, SHIFT2, SCALE2);
        let mut y = s.mlp_in.fwd(&h, rows);
        nn::gelu_tanh(&mut y);
        add_gated(x, d, Some(m.part(GATE2)), &s.mlp_out.fwd(&y, rows));
    }

    fn block(
        &self,
        b: &Block,
        img: &mut [f32],
        txt: &mut [f32],
        rope: &Rope,
        mi: &Mods,
        mt: &Mods,
    ) {
        let (d, hd) = (self.dim, self.head_dim);
        let (ni, nt) = (img.len() / d, txt.len() / d);
        let n = nt + ni;
        let (mut q, mut k, mut v) = self.qkv(&b.txt, &mt.modulate(txt, d, SHIFT1, SCALE1), nt);
        let (qi, ki, vi) = self.qkv(&b.img, &mi.modulate(img, d, SHIFT1, SCALE1), ni);
        // Joint sequence: text rows, then image rows.
        q.extend(qi);
        k.extend(ki);
        v.extend(vi);
        rope.apply(&mut q, d, hd);
        rope.apply(&mut k, d, hd);
        let a = nn::attention(&q, &k, &v, n, n, self.heads, hd, false);
        add_gated(img, d, Some(mi.part(GATE1)), &b.img.o.fwd(&a[nt * d..], ni));
        add_gated(txt, d, Some(mt.part(GATE1)), &b.txt.o.fwd(&a[..nt * d], nt));
        self.mlp(&b.img, img, mi);
        self.mlp(&b.txt, txt, mt);
    }

    /// Velocity of the packed latents `lat [n_img, in_ch]` at `sigma`, given the projected text
    /// `ctx [n_text, dim]` and the rotary tables of the joint sequence.
    pub fn forward(&self, lat: &[f32], ctx: &[f32], rope: &Rope, sigma: f32) -> Vec<f32> {
        let d = self.dim;
        let n_img = lat.len() / self.in_ch;
        let (mut img, mut txt) = (self.img_in.fwd(lat, n_img), ctx.to_vec());
        let cond = silu_of(&self.temb(sigma));
        for b in &self.blocks {
            let (mi, mt) = (
                Mods(b.img.modulation.fwd(&cond, 1)),
                Mods(b.txt.modulation.fwd(&cond, 1)),
            );
            self.block(b, &mut img, &mut txt, rope, &mi, &mt);
        }
        let y = ada_norm_out(&img, d, &self.norm_out.fwd(&cond, 1));
        self.proj_out.fwd(&y, n_img)
    }
}

/// Index into a channel-first `[c, h, w]` latent of every element of its `_pack_latents` form
/// (`[h/2 * w/2, c * 4]`, features ordered `(c, dy, dx)`).
fn pack_perm(c: usize, h: usize, w: usize) -> Vec<usize> {
    let (ph, pw) = (h / 2, w / 2);
    let mut perm = Vec::with_capacity(c * h * w);
    for py in 0..ph {
        for px in 0..pw {
            for ch in 0..c {
                for dy in 0..2 {
                    for dx in 0..2 {
                        perm.push(ch * h * w + (2 * py + dy) * w + 2 * px + dx);
                    }
                }
            }
        }
    }
    perm
}

fn pack(lat: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    pack_perm(c, h, w).into_iter().map(|i| lat[i]).collect()
}

/// `_unpack_latents`: packed tokens back to channel-first `[c, h, w]`.
fn unpack(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (p, i) in pack_perm(c, h, w).into_iter().enumerate() {
        out[i] = x[p];
    }
    out
}

/// True CFG with the pipeline's norm rescale, per token row of `ch` features:
/// `comb = neg + g * (pos - neg)`, `comb * |pos| / |comb|`.
fn cfg_rescale(pos: &[f32], neg: &[f32], g: f32, ch: usize) -> Vec<f32> {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut out = Vec::with_capacity(pos.len());
    for (p, n) in pos.chunks_exact(ch).zip(neg.chunks_exact(ch)) {
        let comb: Vec<f32> = p.iter().zip(n).map(|(p, n)| n + g * (p - n)).collect();
        let s = norm(p) / norm(&comb);
        out.extend(comb.iter().map(|c| c * s));
    }
    out
}

/// Everything after the text encoder: transformer, VAE and sampling schedule.
struct Core {
    dit: Dit,
    vae: QwenVae,
    sched: sched::Schedule,
}

/// A prompt's text-encoder features `[n, text_width]` and `n`.
type Features = (Vec<f32>, usize);

impl Core {
    /// Checks the transformer, VAE and (text width) `text_width` against each other.
    fn new(dit: Dit, vae: QwenVae, sched: sched::Schedule, text_width: usize) -> Result<Self, String> {
        if dit.in_ch != 4 * Z || dit.proj_out.w.n != dit.in_ch {
            return Err(format!(
                "Qwen-Image 1: the transformer takes {} and emits {} packed features, the VAE's {Z} channels need {}",
                dit.in_ch,
                dit.proj_out.w.n,
                4 * Z
            ));
        }
        if text_width != dit.text_width() {
            return Err(
                "Qwen-Image 1: text encoder width does not match the transformer's context width"
                    .into(),
            );
        }
        Ok(Self { dit, vae, sched })
    }

    fn defaults(&self) -> ImageDefaults {
        // The card's 50 steps at true CFG 4; ponytail: CPU-sized 512 default, the card uses 1328x.
        ImageDefaults {
            width: 512,
            height: 512,
            steps: 50,
            guidance: 4.0,
            multiple: 2 * VAE_PX,
        }
    }

    /// Denoises from noise and decodes. `encode` turns a prompt into text-encoder features; the
    /// negative prompt (" " when none is given) is encoded only for true CFG (`guidance > 1`).
    fn generate(
        &self,
        req: &ImageRequest,
        encode: &dyn Fn(&str) -> Result<Features, String>,
    ) -> Result<ImageBuffer, String> {
        let d = self.defaults();
        let (width, height) = (req.width.unwrap_or(d.width), req.height.unwrap_or(d.height));
        let steps = req.steps.unwrap_or(d.steps);
        let guidance = req.guidance.unwrap_or(d.guidance);
        if width == 0 || height == 0 || width % d.multiple != 0 || height % d.multiple != 0 {
            return Err(format!(
                "Qwen-Image 1: width/height must be non-zero multiples of {}, got {width}x{height}",
                d.multiple
            ));
        }
        if steps == 0 {
            return Err("Qwen-Image 1: steps must be at least 1".into());
        }
        let start = Instant::now();
        let (h, w) = (height / VAE_PX, width / VAE_PX);
        let (ph, pw) = (h / 2, w / 2);
        let (n_img, ch) = (ph * pw, self.dit.in_ch);
        // Text features -> (projected context, rotary tables of the joint sequence); on the GPU
        // the raw features go to the program, which projects them itself.
        let gpu = crate::model::dit_graph::use_metal();
        let prepare = |(txt, n): Features| {
            if n == 0 || txt.len() != n * self.dit.text_width() {
                return Err(format!(
                    "Qwen-Image 1: {} text features for {n} tokens of width {}",
                    txt.len(),
                    self.dit.text_width()
                ));
            }
            Ok(if gpu {
                (txt, None)
            } else {
                (
                    self.dit.project_text(&txt, n),
                    Some(Rope::new(&positions(n, ph, pw), &self.dit.axes)),
                )
            })
        };
        let velocity = |(ctx, rope): &(Vec<f32>, Option<Rope>), x: &[f32], sigma: f32| match rope {
            Some(rope) => Ok(self.dit.forward(x, ctx, rope, sigma)),
            None => self.dit.forward_gpu(&self.dit.gpu, x, ctx, ph, pw, sigma),
        };
        let pos = prepare(encode(&req.prompt)?)?;
        let neg = (guidance > 1.0)
            .then(|| encode(req.negative_prompt.as_deref().unwrap_or(" ")).and_then(prepare))
            .transpose()?;
        println!("  text encoder: {:.2}s", start.elapsed().as_secs_f64());

        let sigmas = self.sched.sigmas(&sched::linspace_sigmas(steps), n_img, steps);
        let mut rng = Rng::new(req.seed);
        let noise: Vec<f32> = (0..Z * h * w).map(|_| rng.gaussian()).collect();
        let mut x = pack(&noise, Z, h, w);
        for (i, s) in sigmas.windows(2).enumerate() {
            let t = Instant::now();
            let mut v = velocity(&pos, &x, s[0])?;
            if let Some(neg) = &neg {
                let vn = velocity(neg, &x, s[0])?;
                v = cfg_rescale(&v, &vn, guidance, ch);
            }
            sched::euler_step(&mut x, &v, s[0], s[1]);
            println!(
                "  step {}/{} (sigma {:.4}): {:.2}s",
                i + 1,
                sigmas.len() - 1,
                s[0],
                t.elapsed().as_secs_f64()
            );
        }
        self.vae.decode(&unpack(&x, Z, h, w), h, w)
    }
}

/// Qwen2Tokenizer from `<dir>/tokenizer.json`, or built from `vocab.json`, `merges.txt` and
/// `added_tokens.json` (Qwen-Image's `tokenizer/` ships no tokenizer.json): NFC, GPT-4-style split,
/// byte-level BPE, added tokens split out of the text at their stored ids.
fn load_tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer, String> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::NFC;
    use tokenizers::pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    };
    use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};
    let err = |e: tokenizers::Error| format!("{}: {e}", dir.display());
    let json = dir.join("tokenizer.json");
    if json.is_file() {
        return Tokenizer::from_file(&json).map_err(|e| format!("{}: {e}", json.display()));
    }
    let path = |n: &str| dir.join(n).to_string_lossy().into_owned();
    let bpe = BPE::from_file(&path("vocab.json"), &path("merges.txt"))
        .build()
        .map_err(err)?;
    let mut tok = Tokenizer::new(bpe);
    let split = Split::new(
        SplitPattern::Regex(SPLIT_PATTERN.into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(err)?;
    tok.with_normalizer(Some(NFC));
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::from(split),
        PreTokenizerWrapper::from(ByteLevel::new(false, false, false)),
    ])));
    let added_path = dir.join("added_tokens.json");
    let raw = std::fs::read_to_string(&added_path)
        .map_err(|e| format!("{}: {e}", added_path.display()))?;
    let map: serde_json::Map<String, Value> =
        serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", added_path.display()))?;
    let mut added: Vec<(u32, &str)> = Vec::with_capacity(map.len());
    for (content, id) in &map {
        let id = id
            .as_u64()
            .ok_or_else(|| format!("{}: id of {content} is not a number", added_path.display()))?;
        added.push((id as u32, content.as_str()));
    }
    // Added tokens take consecutive ids after the vocabulary, in the order they are added.
    added.sort();
    // ponytail: every added token is registered as special (tokenizer_config.json marks a few, e.g.
    // `<tool_call>`, non-special); matching in the text is identical, only decoding differs.
    let tokens: Vec<AddedToken> = added
        .iter()
        .map(|(_, c)| AddedToken::from(*c, true))
        .collect();
    tok.add_special_tokens(&tokens);
    for (id, c) in &added {
        if tok.token_to_id(c) != Some(*id) {
            return Err(format!(
                "{}: added token {c} does not land on id {id} (vocab.json has gaps?)",
                dir.display()
            ));
        }
    }
    Ok(tok)
}

pub struct QwenImagePipeline {
    enc: Qwen3Encoder,
    tok: tokenizers::Tokenizer,
    core: Core,
    label: String,
}

impl QwenImagePipeline {
    /// Loads a diffusers `QwenImagePipeline` snapshot directory.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_with(dir, None, None)
    }

    /// `dir` is the base repo snapshot (text encoder, tokenizer, VAE, scheduler, and the transformer
    /// unless `transformer` names a replacement file: a GGUF or single-file finetune in diffusers
    /// names). `scheduler` replaces the base's scheduler config.
    pub fn load_with(
        dir: &Path,
        transformer: Option<&Path>,
        scheduler: Option<Value>,
    ) -> Result<Self, String> {
        let te = dir.join("text_encoder");
        let tcfg = read_config(&te)?;
        if tcfg["model_type"] != "qwen2_5_vl" {
            return Err(format!(
                "Qwen-Image 1: text encoder is {}, expected qwen2_5_vl",
                tcfg["model_type"]
            ));
        }
        let text = if tcfg["text_config"].is_object() {
            &tcfg["text_config"]
        } else {
            &tcfg
        };
        let width = text["hidden_size"]
            .as_u64()
            .ok_or("text_encoder config: missing hidden_size")? as usize;
        let tok = load_tokenizer(&dir.join("tokenizer"))?;
        // The template head must be exactly the dropped tokens (special tokens split, digits single).
        let head = tok
            .encode(TEMPLATE_HEAD, false)
            .map_err(|e| format!("Qwen tokenizer: {e}"))?
            .len();
        if head != DROP {
            return Err(format!(
                "Qwen-Image 1: the prompt template tokenizes to {head} tokens, expected {DROP}"
            ));
        }
        let (path, cfg) = match transformer {
            Some(p) => (p.to_path_buf(), None),
            None => {
                let t = dir.join("transformer");
                (
                    crate::model::hub::component_weights(&t)?,
                    Some(read_config(&t)?),
                )
            }
        };
        let dit = Dit::load(&Store::open(&path)?, cfg.as_ref())?;
        let label = format!(
            "Qwen-Image 1 ({} double-stream blocks, dim {}; {}), Qwen2.5-VL text encoder + 16-ch VAE",
            dit.blocks.len(),
            dit.dim,
            path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())
        );
        let sched = sched::Schedule::from_config(&scheduler.map_or_else(|| read_scheduler(dir), Ok)?);
        let core = Core::new(dit, QwenVae::load(&dir.join("vae"))?, sched, width)?;
        Ok(Self {
            enc: Qwen3Encoder::open(&te)?,
            tok,
            core,
            label,
        })
    }
}

/// Prompt -> the text tower's last hidden state (after its final norm) for the prompt template,
/// minus its first [`DROP`] tokens; the template's tokens are cut at `DROP + MAX_TOKENS`.
fn encode(
    enc: &Qwen3Encoder,
    tok: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<Features, String> {
    let text = format!("{TEMPLATE_HEAD}{prompt}{TEMPLATE_TAIL}");
    let ids = tok
        .encode(text, false)
        .map_err(|e| format!("Qwen tokenizer: {e}"))?;
    let ids = &ids.get_ids()[..ids.len().min(DROP + MAX_TOKENS)];
    if ids.len() <= DROP {
        return Err(format!(
            "Qwen tokenizer: the template alone is {} tokens, expected more than {DROP}",
            ids.len()
        ));
    }
    let hs = enc.hidden_states_normed(ids)?;
    let w = hs.len() / ids.len();
    Ok((hs[DROP * w..].to_vec(), ids.len() - DROP))
}

impl ImageGenerator for QwenImagePipeline {
    fn summary(&self) -> String {
        self.label.clone()
    }

    fn defaults(&self) -> ImageDefaults {
        self.core.defaults()
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        self.core
            .generate(req, &|p| encode(&self.enc, &self.tok, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_tokens_are_centred_and_text_follows_the_half_side() {
        // 2 text tokens, then a 4x6 patch grid: text at max(2, 3) = 3 on every axis; image frame 0,
        // heights -2..1, widths -3..2.
        let pos = positions(2, 4, 6);
        assert_eq!(&pos[..2], &[[3, 3, 3], [4, 4, 4]]);
        assert_eq!(pos.len(), 2 + 24);
        assert_eq!(pos[2], [0, -2, -3]);
        assert_eq!(pos[2 + 5], [0, -2, 2]);
        assert_eq!(pos[2 + 6], [0, -1, -3]);
        assert_eq!(pos[2 + 23], [0, 1, 2]);
        // Odd sides: 3 rows centre on -2..0 (h - h/2 = 2 negative rows), text at max(1, 2) = 2.
        let odd = positions(1, 3, 5);
        assert_eq!(odd[0], [2, 2, 2]);
        assert_eq!(odd[1], [0, -2, -3]);
        assert_eq!(odd[odd.len() - 1], [0, 0, 1]);
    }

    #[test]
    fn rope_is_identity_at_position_zero_and_rotates_negative_positions() {
        let rope = Rope::new(&[[0, 0, 0], [0, -1, 0]], &[2, 2, 4]);
        let mut x: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect(); // 2 rows, 1 head of 8
        let orig = x.clone();
        rope.apply(&mut x, 8, 8);
        assert_eq!(&x[..8], &orig[..8]);
        // Row 1, pair 1 (height axis, inv_freq 1, position -1): rotated by -1 rad.
        let (a, b) = (orig[10], orig[11]);
        assert!((x[10] - (a * 1f32.cos() + b * 1f32.sin())).abs() < 1e-5);
        assert!((x[11] - (-a * 1f32.sin() + b * 1f32.cos())).abs() < 1e-5);
    }

    #[test]
    fn pack_and_unpack_round_trip() {
        let (c, h, w) = (2, 4, 6);
        let lat: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let packed = pack(&lat, c, h, w);
        assert_eq!(packed.len(), (h / 2) * (w / 2) * c * 4);
        // Token 0 holds the top-left 2x2 patch of channel 0, then of channel 1: features (c, dy, dx).
        assert_eq!(&packed[..8], &[0.0, 1.0, 6.0, 7.0, 24.0, 25.0, 30.0, 31.0]);
        // Token 1 is the next patch to the right.
        assert_eq!(packed[8], 2.0);
        assert_eq!(unpack(&packed, c, h, w), lat);
    }

    #[test]
    fn cfg_is_rescaled_to_the_conditional_norm_per_token() {
        // One token of 2 features: comb = neg + 2 (pos - neg) = [2, -1], rescaled to |pos| = 1.
        let got = cfg_rescale(&[1.0, 0.0], &[0.0, 1.0], 2.0, 2);
        let s = 5f32.sqrt();
        assert!(
            (got[0] - 2.0 / s).abs() < 1e-6 && (got[1] + 1.0 / s).abs() < 1e-6,
            "{got:?}"
        );
        // Two tokens are rescaled independently.
        let got = cfg_rescale(&[3.0, 4.0, 0.0, 2.0], &[0.0; 4], 3.0, 2);
        assert!((got[0] - 3.0).abs() < 1e-5 && (got[1] - 4.0).abs() < 1e-5);
        assert!(got[2].abs() < 1e-5 && (got[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn final_norm_takes_scale_before_shift_and_blocks_shift_before_scale() {
        // x = [1, 3] normalises to [-1, 1]; scale [1, 0], shift [10, 20]: [-1 * 2 + 10, 1 * 1 + 20].
        let y = ada_norm_out(&[1.0, 3.0], 2, &[1.0, 0.0, 10.0, 20.0]);
        assert!(
            (y[0] - 8.0).abs() < 1e-3 && (y[1] - 21.0).abs() < 1e-3,
            "{y:?}"
        );
        // Block modulation `[shift1, scale1, gate1, ...]` of d = 2.
        let m = Mods((0..12).map(|i| i as f32).collect());
        assert_eq!(m.part(SHIFT1), &[0.0, 1.0]);
        assert_eq!(m.part(SCALE1), &[2.0, 3.0]);
        assert_eq!(m.part(GATE1), &[4.0, 5.0]);
        assert_eq!(m.part(SHIFT2), &[6.0, 7.0]);
        assert_eq!(m.part(SCALE2), &[8.0, 9.0]);
        assert_eq!(m.part(GATE2), &[10.0, 11.0]);
        let y = m.modulate(&[1.0, 3.0], 2, SHIFT1, SCALE1);
        assert!(
            (y[0] - (-1.0 * 3.0 + 0.0)).abs() < 1e-3 && (y[1] - (1.0 * 4.0 + 1.0)).abs() < 1e-3,
            "{y:?}"
        );
    }

    #[test]
    fn tokenizer_is_built_from_vocab_and_merges_and_splits_out_special_tokens() {
        let dir = std::env::temp_dir().join(format!("gibbs-qwen1-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.json"), r#"{"h":0,"i":1,"Ġ":2,"hi":3}"#).unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\nh i\n").unwrap();
        std::fs::write(
            dir.join("added_tokens.json"),
            r#"{"<|im_end|>":5,"<|im_start|>":4}"#,
        )
        .unwrap();
        let tok = load_tokenizer(&dir).unwrap();
        let ids = |s: &str| tok.encode(s, false).unwrap().get_ids().to_vec();
        assert_eq!(ids("hi"), [3]);
        assert_eq!(ids("hi hi"), [3, 2, 3]);
        assert_eq!(ids("<|im_start|>hi<|im_end|>"), [4, 3, 5]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn env_path(k: &str) -> Option<std::path::PathBuf> {
        std::env::var_os(k).map(Into::into)
    }

    /// `GIBBS_QWEN1_TOKENIZER`: Qwen-Image's `tokenizer/` files (vocab.json, merges.txt,
    /// added_tokens.json) next to `ref_tokenizer.json` (Qwen2.5-VL's tokenizer.json): the tokenizer
    /// built from the former must give the latter's ids, and the template head must be `DROP` tokens.
    #[test]
    fn real_tokenizer_matches_the_reference_tokenizer_json() {
        let Some(dir) = env_path("GIBBS_QWEN1_TOKENIZER") else {
            eprintln!("SKIP: GIBBS_QWEN1_TOKENIZER not set");
            return;
        };
        let built = load_tokenizer(&dir).unwrap();
        let reference = tokenizers::Tokenizer::from_file(dir.join("ref_tokenizer.json")).unwrap();
        let ids =
            |t: &tokenizers::Tokenizer, s: &str| t.encode(s, false).unwrap().get_ids().to_vec();
        for s in [
            TEMPLATE_HEAD,
            "A cat 🐱 on 123456 books, 日本語のテキスト \n\n  indented\ttabs  ",
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            "don't won't I'LL we've; \"quoted\" (x) [y] ...",
            "<|endoftext|>x<|vision_start|><|image_pad|>",
            " ",
        ] {
            assert_eq!(ids(&built, s), ids(&reference, s), "{s:?}");
        }
        assert_eq!(ids(&built, TEMPLATE_HEAD).len(), DROP);
    }

    /// `GIBBS_QWEN2_DIR` (Qwen2.5-0.5B-Instruct: config.json, model.safetensors, tokenizer.json): the
    /// Qwen2 decoder (qkv bias, no qk-norm) the v1 text encoder shares with it must continue "The
    /// capital of France is" with " Paris", the tied embedding as LM head.
    #[test]
    fn real_qwen2_decoder_predicts_paris() {
        let Some(dir) = env_path("GIBBS_QWEN2_DIR") else {
            eprintln!("SKIP: GIBBS_QWEN2_DIR not set");
            return;
        };
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let ids = tok
            .encode("The capital of France is", false)
            .unwrap()
            .get_ids()
            .to_vec();
        let hs = Qwen3Encoder::open(&dir)
            .unwrap()
            .hidden_states_normed(&ids)
            .unwrap();
        let w = hs.len() / ids.len();
        let last = &hs[(ids.len() - 1) * w..];
        let embed = crate::model::weights::Weights::open(&dir.join("model.safetensors"))
            .unwrap()
            .vec("model.embed_tokens.weight")
            .unwrap();
        let score = |r: &[f32]| r.iter().zip(last).map(|(a, b)| a * b).sum::<f32>();
        let best = embed
            .chunks_exact(w)
            .enumerate()
            .max_by(|a, b| score(a.1).total_cmp(&score(b.1)))
            .unwrap()
            .0;
        assert_eq!(tok.decode(&[best as u32], false).unwrap(), " Paris");
    }

    /// `GIBBS_QWEN1_SLICE` (a few-block safetensors slice of the transformer in diffusers names) and
    /// `GIBBS_QWEN1_DUMP` (a directory): one forward pass on seeded inputs (6 text tokens, an odd 3x5
    /// patch grid so the centring and the text offset are not symmetric, sigma 0.6), written as raw f32 LE `lat.bin`, `txt.bin`, `out.bin` for a reference to replay.
    #[test]
    fn real_slice_forward_dump() {
        let (Some(slice), Some(dump)) =
            (env_path("GIBBS_QWEN1_SLICE"), env_path("GIBBS_QWEN1_DUMP"))
        else {
            eprintln!("SKIP: GIBBS_QWEN1_{{SLICE,DUMP}} not set");
            return;
        };
        let dit = Dit::load(&Store::open(&slice).unwrap(), None).unwrap();
        let (n_txt, ph, pw) = (6, 3, 5);
        let mut rng = Rng::new(1);
        let lat: Vec<f32> = (0..ph * pw * dit.in_ch).map(|_| rng.gaussian()).collect();
        let txt: Vec<f32> = (0..n_txt * dit.text_width())
            .map(|_| rng.gaussian())
            .collect();
        let ctx = dit.project_text(&txt, n_txt);
        let out = dit.forward(
            &lat,
            &ctx,
            &Rope::new(&positions(n_txt, ph, pw), &dit.axes),
            0.6,
        );
        let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for (name, v) in [("lat.bin", &lat), ("txt.bin", &txt), ("out.bin", &out)] {
            std::fs::write(dump.join(name), raw(v)).unwrap();
        }
        println!("{} blocks, out[..4] = {:?}", dit.blocks.len(), &out[..4]);
    }

    /// Real-weight run: `GIBBS_QWEN1_TRANSFORMER` (a Qwen-Image GGUF or safetensors in diffusers
    /// names), `GIBBS_QWEN1_VAE` (diffusers `vae/` dir), `GIBBS_QWEN1_FEATURES` (raw f32 LE text
    /// features `[n, 3584]`) and optionally `GIBBS_QWEN1_SIZE` (pixels, default 512) -> a PNG at
    /// `GIBBS_QWEN1_PNG`.
    #[test]
    fn real_transformer_and_vae_generate() {
        let (Some(tr), Some(vae), Some(feats), Some(png)) = (
            env_path("GIBBS_QWEN1_TRANSFORMER"),
            env_path("GIBBS_QWEN1_VAE"),
            env_path("GIBBS_QWEN1_FEATURES"),
            env_path("GIBBS_QWEN1_PNG"),
        ) else {
            eprintln!("SKIP: GIBBS_QWEN1_{{TRANSFORMER,VAE,FEATURES,PNG}} not set");
            return;
        };
        let raw = std::fs::read(feats).unwrap();
        let f: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = f.len() / 3584;
        let sched = serde_json::json!({"use_dynamic_shifting": true, "max_image_seq_len": 8192, "max_shift": 0.9, "shift_terminal": 0.02});
        let dit = Dit::load(&Store::open(&tr).unwrap(), None).unwrap();
        let core = Core::new(
            dit,
            QwenVae::load(&vae).unwrap(),
            sched::Schedule::from_config(&sched),
            3584,
        )
        .unwrap();
        let size = std::env::var("GIBBS_QWEN1_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let req = ImageRequest {
            prompt: "x".into(),
            width: Some(size),
            height: Some(size),
            seed: 7,
            ..Default::default()
        };
        let img = core.generate(&req, &|_| Ok((f.clone(), n))).unwrap();
        std::fs::write(&png, img.to_png()).unwrap();
        println!("wrote {} ({}x{})", png.display(), img.width, img.height);
    }
}
