//! Stable Diffusion 3 / 3.5 (diffusers `StableDiffusion3Pipeline`) on the CPU: CLIP-L + CLIP-G
//! (penultimate hidden states + pooled projections) + T5-XXL text encoders, the
//! `SD3Transformer2DModel` MMDiT (joint text/image attention, SD3.5 "MMDiT-X" dual attention and
//! QK RMSNorm when the checkpoint has them) on bf16-mapped weights, FlowMatchEulerDiscrete with a
//! static shift, and the 16-channel AutoencoderKL.

use crate::model::clip::{ClipTextTransformer, Tensors};
use crate::model::dit::{
    add_gated, component_weights, norm_modulate, read_config, rms_norm, silu_of, timestep_features,
};
use crate::model::flow;
use crate::model::hub::HuggingFaceHub;
use crate::model::image::ImageBuffer;
use crate::model::loaders::safetensors::SafetensorsWeightReader;
use crate::model::nn;
use crate::model::sd::Rng;
use crate::model::t5::T5Encoder;
use crate::model::text_to_image::{ImageDefaults, ImageGenerator, ImageRequest};
use crate::model::vae::SdVae;
use crate::model::weights::{Mat, Weights};
use std::path::Path;
use std::time::Instant;

const HEAD_DIM: usize = 64;
const EPS: f32 = 1e-6;
/// T5 tokens per prompt (pipeline `max_sequence_length` default).
const T5_LEN: usize = 256;
const T5_EOS: u32 = 1;
const T5_PAD: u32 = 0;

mod graph_program;

/// `nn.Linear` over a bf16 `[out, in]` matrix (kept here, not `dit::Lin`, so the graph program can bind
/// the matrix and bias by name).
struct Lin {
    w: Mat,
    b: Option<Vec<f32>>,
}

impl Lin {
    fn load(w: &Weights, p: &str) -> Result<Self, String> {
        let wt = w.mat(&format!("{p}.weight"))?;
        let bias = format!("{p}.bias");
        let b = if w.has(&bias) {
            Some(w.vec_n(&bias, wt.rows)?)
        } else {
            None
        };
        Ok(Self { w: wt, b })
    }

    fn out_dim(&self) -> usize {
        self.w.rows
    }

    fn in_dim(&self) -> usize {
        self.w.cols
    }

    /// x [rows, in] -> [rows, out].
    fn fwd(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * self.w.rows];
        nn::matmul_bf16(
            x,
            self.w.data(),
            self.b.as_deref(),
            rows,
            self.w.cols,
            self.w.rows,
            &mut out,
        );
        out
    }
}

/// q/k/v (+ output) projections of one attention stream, with optional per-head QK RMSNorm.
struct Proj {
    q: Lin,
    k: Lin,
    v: Lin,
    out: Option<Lin>,
    norm_q: Option<Vec<f32>>,
    norm_k: Option<Vec<f32>>,
}

impl Proj {
    /// `names`: q, k, v, out, norm_q, norm_k suffixes under `p`.
    fn load(w: &Weights, p: &str, names: [&str; 6]) -> Result<Self, String> {
        let lin = |n: &str| Lin::load(w, &format!("{p}.{n}"));
        let norm = |n: &str| -> Result<Option<Vec<f32>>, String> {
            let t = format!("{p}.{n}.weight");
            if w.has(&t) {
                w.vec_n(&t, HEAD_DIM).map(Some)
            } else {
                Ok(None)
            }
        };
        let out = format!("{p}.{}.weight", names[3]);
        Ok(Self {
            q: lin(names[0])?,
            k: lin(names[1])?,
            v: lin(names[2])?,
            out: if w.has(&out) {
                Some(lin(names[3])?)
            } else {
                None
            },
            norm_q: norm(names[4])?,
            norm_k: norm(names[5])?,
        })
    }

    fn qkv(&self, x: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut q, mut k) = (self.q.fwd(x, n), self.k.fwd(x, n));
        if let Some(g) = &self.norm_q {
            rms_norm(&mut q, HEAD_DIM, g, EPS);
        }
        if let Some(g) = &self.norm_k {
            rms_norm(&mut k, HEAD_DIM, g, EPS);
        }
        (q, k, self.v.fwd(x, n))
    }
}

const IMG: [&str; 6] = ["to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k"];
const CTX: [&str; 6] = [
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
];

/// FeedForward with tanh-GELU.
struct Ff {
    up: Lin,
    down: Lin,
}

impl Ff {
    fn load(w: &Weights, p: &str) -> Result<Self, String> {
        Ok(Self {
            up: Lin::load(w, &format!("{p}.net.0.proj"))?,
            down: Lin::load(w, &format!("{p}.net.2"))?,
        })
    }
    fn fwd(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut h = self.up.fwd(x, n);
        nn::gelu_tanh(&mut h);
        self.down.fwd(&h, n)
    }
}

struct Block {
    /// AdaLayerNormZero (6 chunks) or SD35AdaLayerNormZeroX (9 chunks, dual attention).
    norm1: Lin,
    /// AdaLayerNormZero, or AdaLayerNormContinuous (scale, shift) in the last, context-pre-only block.
    norm1_ctx: Lin,
    img: Proj,
    ctx: Proj,
    attn2: Option<Proj>,
    ff: Ff,
    /// `None` in the last block, whose context output is discarded.
    ff_ctx: Option<Ff>,
}

pub struct Mmdit {
    d: usize,
    heads: usize,
    patch: usize,
    in_ch: usize,
    pos_max: usize,
    patch_proj: Lin,
    pos_embed: Mat,
    t_lin1: Lin,
    t_lin2: Lin,
    text_lin1: Lin,
    text_lin2: Lin,
    context_embedder: Lin,
    blocks: Vec<Block>,
    norm_out: Lin,
    proj_out: Lin,
    gpu: graph_program::Cache,
}

/// Latents `[C, h, w]` as `[hp*wp, C*p*p]` patch rows in conv-weight order (c, ky, kx).
fn patchify(lat: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (hp, wp, pp) = (h / p, w / p, p * p);
    let mut patches = vec![0.0f32; hp * wp * c * pp];
    for i in 0..hp {
        for j in 0..wp {
            let row = &mut patches[(i * wp + j) * c * pp..][..c * pp];
            for ch in 0..c {
                for ky in 0..p {
                    for kx in 0..p {
                        row[ch * pp + ky * p + kx] = lat[(ch * h + i * p + ky) * w + j * p + kx];
                    }
                }
            }
        }
    }
    patches
}

/// Inverse layout of [`patchify`] for the projected rows `[hp*wp, p*p*C]` (features are (py, px, c)).
fn unpatchify(y: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (hp, wp, pp) = (h / p, w / p, p * p);
    let mut out = vec![0.0f32; c * h * w];
    for i in 0..hp {
        for j in 0..wp {
            let row = &y[(i * wp + j) * c * pp..][..c * pp];
            for py in 0..p {
                for px in 0..p {
                    for ch in 0..c {
                        out[(ch * h + i * p + py) * w + j * p + px] = row[(py * p + px) * c + ch];
                    }
                }
            }
        }
    }
    out
}

impl Mmdit {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = read_config(dir)?;
        let w = component_weights(dir)?;
        let get = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("transformer config: missing {k}"))
        };
        let (heads, hd, layers, patch, in_ch) = (
            get("num_attention_heads")?,
            get("attention_head_dim")?,
            get("num_layers")?,
            get("patch_size")?,
            get("in_channels")?,
        );
        if hd != HEAD_DIM || get("out_channels").unwrap_or(in_ch) != in_ch {
            return Err(format!("SD3 transformer: head_dim {hd} / out_channels unsupported (expected 64, = in_channels)"));
        }
        let pos_max = get("pos_embed_max_size")?;
        let d = heads * hd;
        let blocks = (0..layers)
            .map(|i| -> Result<Block, String> {
                let p = format!("transformer_blocks.{i}");
                let attn2 = format!("{p}.attn2.to_q.weight");
                let ff_ctx = format!("{p}.ff_context.net.0.proj.weight");
                Ok(Block {
                    norm1: Lin::load(&w, &format!("{p}.norm1.linear"))?,
                    norm1_ctx: Lin::load(&w, &format!("{p}.norm1_context.linear"))?,
                    img: Proj::load(&w, &format!("{p}.attn"), IMG)?,
                    ctx: Proj::load(&w, &format!("{p}.attn"), CTX)?,
                    attn2: if w.has(&attn2) {
                        Some(Proj::load(&w, &format!("{p}.attn2"), IMG)?)
                    } else {
                        None
                    },
                    ff: Ff::load(&w, &format!("{p}.ff"))?,
                    ff_ctx: if w.has(&ff_ctx) {
                        Some(Ff::load(&w, &format!("{p}.ff_context"))?)
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut pos_embed = w.mat("pos_embed.pos_embed")?;
        if pos_embed.rows * pos_embed.cols != pos_max * pos_max * d {
            return Err(format!("SD3 pos_embed is not [{pos_max}^2, {d}]"));
        }
        // Stored `[1, pos_max^2, d]`: view it as the `[pos_max^2, d]` table the graph gathers from.
        (pos_embed.rows, pos_embed.cols) = (pos_max * pos_max, d);
        let m = Self {
            d,
            heads,
            patch,
            in_ch,
            pos_max,
            patch_proj: Lin::load(&w, "pos_embed.proj")?,
            pos_embed,
            t_lin1: Lin::load(&w, "time_text_embed.timestep_embedder.linear_1")?,
            t_lin2: Lin::load(&w, "time_text_embed.timestep_embedder.linear_2")?,
            text_lin1: Lin::load(&w, "time_text_embed.text_embedder.linear_1")?,
            text_lin2: Lin::load(&w, "time_text_embed.text_embedder.linear_2")?,
            context_embedder: Lin::load(&w, "context_embedder")?,
            blocks,
            norm_out: Lin::load(&w, "norm_out.linear")?,
            proj_out: Lin::load(&w, "proj_out")?,
            gpu: Default::default(),
        };
        if m.patch_proj.in_dim() != in_ch * patch * patch
            || m.proj_out.out_dim() != in_ch * patch * patch
        {
            return Err("SD3 transformer: patch projection shapes do not match the config".into());
        }
        Ok(m)
    }

    /// Velocity for latents [C, h, w] at `timestep` (= sigma * 1000), context [n_ctx, joint_dim], pooled text.
    pub fn forward(
        &self,
        lat: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        ctx: &[f32],
        n_ctx: usize,
        pooled: &[f32],
    ) -> Vec<f32> {
        let (d, p, c) = (self.d, self.patch, self.in_ch);
        let (hp, wp) = (h / p, w / p);
        let n = hp * wp;
        let patches = patchify(lat, c, h, w, p);
        let mut x = self.patch_proj.fwd(&patches, n);
        let (top, left) = ((self.pos_max - hp) / 2, (self.pos_max - wp) / 2);
        let pe = self.pos_embed.data();
        for i in 0..hp {
            for j in 0..wp {
                let src = &pe[((top + i) * self.pos_max + left + j) * d..][..d];
                for (v, &e) in x[(i * wp + j) * d..][..d].iter_mut().zip(src) {
                    *v += crate::model::bf16_to_f32(e);
                }
            }
        }
        let mut temb = self.t_lin2.fwd(
            &silu_of(&self.t_lin1.fwd(&timestep_features(timestep, 256), 1)),
            1,
        );
        let pooled = self
            .text_lin2
            .fwd(&silu_of(&self.text_lin1.fwd(pooled, 1)), 1);
        temb.iter_mut().zip(&pooled).for_each(|(a, b)| *a += b);
        let st = silu_of(&temb);
        let mut cx = self.context_embedder.fwd(ctx, n_ctx);

        for b in &self.blocks {
            let e = b.norm1.fwd(&st, 1);
            let ch = |k: usize| &e[k * d..(k + 1) * d];
            // Chunks: shift, scale, gate (msa); shift, scale, gate (mlp); [shift, scale, gate (msa2)].
            let nx = norm_modulate(&x, d, EPS, ch(1), ch(0));
            // SD3.5 MMDiT-X: the second self-attention reads the pre-attention hidden states.
            let nx2 = b
                .attn2
                .as_ref()
                .map(|_| norm_modulate(&x, d, EPS, ch(7), ch(6)));
            let ce = b.norm1_ctx.fwd(&st, 1);
            let cc = |k: usize| &ce[k * d..(k + 1) * d];
            let nc = match b.ff_ctx {
                // AdaLayerNormContinuous: scale, shift.
                None => norm_modulate(&cx, d, EPS, cc(0), cc(1)),
                Some(_) => norm_modulate(&cx, d, EPS, cc(1), cc(0)),
            };

            let (q, k, v) = b.img.qkv(&nx, n);
            let (cq, ck, cv) = b.ctx.qkv(&nc, n_ctx);
            let cat = |a: Vec<f32>, b: Vec<f32>| [a, b].concat();
            let o = nn::attention(
                &cat(q, cq),
                &cat(k, ck),
                &cat(v, cv),
                n + n_ctx,
                n + n_ctx,
                self.heads,
                HEAD_DIM,
                false,
            );
            let (ao, co) = o.split_at(n * d);
            let out = |p: &Proj, a: &[f32], rows: usize| {
                p.out
                    .as_ref()
                    .map_or_else(|| a.to_vec(), |l| l.fwd(a, rows))
            };
            add_gated(&mut x, d, Some(ch(2)), &out(&b.img, ao, n));
            if let (Some(a2), Some(nx2)) = (&b.attn2, &nx2) {
                let (q, k, v) = a2.qkv(nx2, n);
                let o2 = nn::attention(&q, &k, &v, n, n, self.heads, HEAD_DIM, false);
                add_gated(&mut x, d, Some(ch(8)), &out(a2, &o2, n));
            }
            let nx = norm_modulate(&x, d, EPS, ch(4), ch(3));
            add_gated(&mut x, d, Some(ch(5)), &b.ff.fwd(&nx, n));

            if let Some(ff_ctx) = &b.ff_ctx {
                add_gated(&mut cx, d, Some(cc(2)), &out(&b.ctx, co, n_ctx));
                let nc = norm_modulate(&cx, d, EPS, cc(4), cc(3));
                add_gated(&mut cx, d, Some(cc(5)), &ff_ctx.fwd(&nc, n_ctx));
            }
        }
        let e = self.norm_out.fwd(&st, 1);
        let x = norm_modulate(&x, d, EPS, &e[..d], &e[d..]);
        let y = self.proj_out.fwd(&x, n);
        unpatchify(&y, c, h, w, p)
    }
}

/// One prompt's conditioning: `[77 + 256, joint_dim]` context and `[pooled_dim]` pooled text.
struct Cond {
    ctx: Vec<f32>,
    n_ctx: usize,
    pooled: Vec<f32>,
}

pub struct Sd3Pipeline {
    clip_l: ClipTextTransformer,
    clip_g: ClipTextTransformer,
    /// `None` when the snapshot has no `text_encoder_3` (the pipeline then feeds zeros).
    t5: Option<(T5Encoder, tokenizers::Tokenizer)>,
    mmdit: Mmdit,
    vae: SdVae,
    joint_dim: usize,
    shift: f64,
}

impl Sd3Pipeline {
    /// Loads a diffusers `StableDiffusion3Pipeline` snapshot directory.
    pub fn load(dir: &Path, hub: &HuggingFaceHub) -> Result<Self, String> {
        let sched = crate::model::dit::read_scheduler(dir)?;
        if sched["use_dynamic_shifting"].as_bool() == Some(true) {
            return Err("SD3 scheduler with use_dynamic_shifting is not supported".into());
        }
        let shift = sched["shift"].as_f64().unwrap_or(3.0);
        let tcfg = read_config(&dir.join("transformer"))?;
        let joint_dim = tcfg["joint_attention_dim"]
            .as_u64()
            .ok_or("transformer config: missing joint_attention_dim")?
            as usize;
        let clip_l = ClipTextTransformer::load_diffusers(
            hub,
            &dir.join("text_encoder"),
            &dir.join("tokenizer"),
        )?;
        let clip_g = ClipTextTransformer::load_diffusers(
            hub,
            &dir.join("text_encoder_2"),
            &dir.join("tokenizer_2"),
        )?;
        let t5_dir = dir.join("text_encoder_3");
        let t5 = if t5_dir.is_dir() {
            let cfg = read_config(&t5_dir)?;
            let enc = T5Encoder::load(&component_weights(&t5_dir)?, &cfg, "")?;
            let tok_path = dir.join("tokenizer_3").join("tokenizer.json");
            let tok = tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| format!("{}: {e}", tok_path.display()))?;
            Some((enc, tok))
        } else {
            None
        };
        let vae_dir = dir.join("vae");
        let vcfg = read_config(&vae_dir)?;
        let reader = SafetensorsWeightReader::new(crate::model::hub::component_weights(&vae_dir)?)?;
        let vae = SdVae::from_diffusers(&Tensors::safetensors(&reader)?)?.with_scaling(
            vcfg["scaling_factor"]
                .as_f64()
                .ok_or("vae config: missing scaling_factor")? as f32,
            vcfg["shift_factor"].as_f64().unwrap_or(0.0) as f32,
        );
        let mmdit = Mmdit::load(&dir.join("transformer"))?;
        if vae.latent_channels() != mmdit.in_ch {
            return Err(format!(
                "VAE has {} latent channels, transformer {}",
                vae.latent_channels(),
                mmdit.in_ch
            ));
        }
        if mmdit.text_lin1.in_dim() != clip_l.d_model() + clip_g.d_model()
            || mmdit.context_embedder.in_dim() != joint_dim
            || clip_l.d_model() + clip_g.d_model() > joint_dim
        {
            return Err("SD3: text encoder widths do not match the transformer".into());
        }
        Ok(Self {
            clip_l,
            clip_g,
            t5,
            mmdit,
            vae,
            joint_dim,
            shift,
        })
    }

    fn encode(&self, prompt: &str) -> Result<Cond, String> {
        let l = self.clip_l.encode(prompt, true)?;
        let g = self.clip_g.encode(prompt, true)?;
        let (dl, dg, jd) = (self.clip_l.d_model(), self.clip_g.d_model(), self.joint_dim);
        let clip_len = l.hidden.len() / dl;
        let mut ctx = vec![0.0f32; (clip_len + T5_LEN) * jd];
        for r in 0..clip_len {
            ctx[r * jd..][..dl].copy_from_slice(&l.hidden[r * dl..][..dl]);
            ctx[r * jd + dl..][..dg].copy_from_slice(&g.hidden[r * dg..][..dg]);
        }
        if let Some((t5, tok)) = &self.t5 {
            let ids = tok
                .encode(prompt, false)
                .map_err(|e| format!("T5 tokenizer: {e}"))?;
            let ids = t5_ids(ids.get_ids());
            let hs = t5.encode(&ids);
            if hs.len() != T5_LEN * jd {
                return Err(format!(
                    "T5 width {} != joint_attention_dim {jd}",
                    hs.len() / T5_LEN
                ));
            }
            ctx[clip_len * jd..].copy_from_slice(&hs);
        }
        Ok(Cond {
            ctx,
            n_ctx: clip_len + T5_LEN,
            pooled: [l.pooled, g.pooled].concat(),
        })
    }
}

/// T5 framing: ids truncated to 255, `</s>`, padded with `<pad>` to 256.
fn t5_ids(ids: &[u32]) -> Vec<u32> {
    let n = ids.len().min(T5_LEN - 1);
    let mut out = ids[..n].to_vec();
    out.push(T5_EOS);
    out.resize(T5_LEN, T5_PAD);
    out
}

impl ImageGenerator for Sd3Pipeline {
    fn summary(&self) -> String {
        format!(
            "Stable Diffusion 3 MMDiT ({} blocks{}), CLIP-L + CLIP-G{} + 16-ch VAE, bf16",
            self.mmdit.blocks.len(),
            if self.mmdit.blocks.iter().any(|b| b.attn2.is_some()) {
                ", SD3.5 dual attention"
            } else {
                ""
            },
            if self.t5.is_some() { " + T5-XXL" } else { "" }
        )
    }

    fn defaults(&self) -> ImageDefaults {
        // ponytail: CPU-sized 512 default (the checkpoints' native size is 1024; pass width/height for it).
        ImageDefaults {
            width: 512,
            height: 512,
            steps: 28,
            guidance: 7.0,
            multiple: 16,
        }
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        let dflt = self.defaults();
        let (width, height) = (
            req.width.unwrap_or(dflt.width),
            req.height.unwrap_or(dflt.height),
        );
        let steps = req.steps.unwrap_or(dflt.steps);
        let guidance = req.guidance.unwrap_or(dflt.guidance);
        let max = self.mmdit.pos_max * self.mmdit.patch * 8;
        if width == 0
            || height == 0
            || width % 16 != 0
            || height % 16 != 0
            || width > max
            || height > max
        {
            return Err(format!(
                "SD3: width/height must be multiples of 16 up to {max}, got {width}x{height}"
            ));
        }
        if !(1..=200).contains(&steps) {
            return Err(format!("SD3: steps must be in 1..=200, got {steps}"));
        }
        let (h, w) = (height / 8, width / 8);
        let cond = self.encode(&req.prompt)?;
        // diffusers runs CFG only when guidance_scale > 1.
        let uncond = if guidance > 1.0 {
            Some(self.encode(req.negative_prompt.as_deref().unwrap_or(""))?)
        } else {
            None
        };
        let sigmas = flow::default_sigmas(steps, self.shift);
        let mut rng = Rng::new(req.seed);
        let mut x: Vec<f32> = (0..self.mmdit.in_ch * h * w)
            .map(|_| rng.gaussian())
            .collect();
        for (i, s) in sigmas.windows(2).enumerate() {
            let start = Instant::now();
            let t = flow::timestep(s[0]);
            let run = |c: &Cond| -> Result<Vec<f32>, String> {
                if crate::model::dit_graph::use_metal() {
                    self.mmdit
                        .forward_gpu(&x, h, w, t, &c.ctx, c.n_ctx, &c.pooled)
                } else {
                    Ok(self.mmdit.forward(&x, h, w, t, &c.ctx, c.n_ctx, &c.pooled))
                }
            };
            let mut v = run(&cond)?;
            if let Some(u) = &uncond {
                let vu = run(u)?;
                v.iter_mut()
                    .zip(&vu)
                    .for_each(|(c, u)| *c = u + guidance * (*c - u));
            }
            flow::euler_step(&mut x, &v, s[0], s[1]);
            println!(
                "  step {}/{steps} (sigma {:.4}): {:.2}s",
                i + 1,
                s[0],
                start.elapsed().as_secs_f64()
            );
        }
        self.vae.decode(&x, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_framing_keeps_eos_and_pads() {
        assert_eq!(&t5_ids(&[5, 6])[..4], &[5, 6, T5_EOS, T5_PAD]);
        let long: Vec<u32> = (10..400).collect();
        let f = t5_ids(&long);
        assert_eq!(f.len(), T5_LEN);
        assert_eq!(f[254], 264);
        assert_eq!(f[255], T5_EOS);
    }
}
