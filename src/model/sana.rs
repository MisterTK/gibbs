//! Sana (diffusers `SanaPipeline`, e.g. `Efficient-Large-Model/Sana_600M_512px_diffusers`) on the
//! CPU: a Gemma 2 text encoder (prompt behind the pipeline's "complex human instruction"), the
//! `SanaTransformer2DModel` linear DiT (ReLU linear self-attention, cross-attention to the caption,
//! GLU MBConv feed-forward, adaLN-single) on bf16 weights, DPM-Solver++ 2M with flow sigmas, and
//! the 32x DC-AE decoder.

mod graph_program;

use crate::model::dcae::DcAeDecoder;
use crate::model::dit::{
    add_gated, component_weights, norm_modulate, read_config, rms_norm, silu_of, timestep_features,
    Lin,
};
use crate::model::flow::{self, DpmSolver2};
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::{par_for, par_rows, OutPtr};
use crate::model::sd::Rng;
use crate::model::text_to_image::{ImageDefaults, ImageGenerator, ImageRequest};
use crate::model::weights::{Mat, Weights};
use std::path::Path;
use std::time::Instant;

/// The pipeline's default `complex_human_instruction`, joined with newlines and prepended to the prompt.
const CHI: [&str; 8] = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
];
/// Pipeline `max_sequence_length`.
const MAX_SEQ: usize = 300;

// ---------------------------------------------------------------------------
// Gemma 2 encoder (last_hidden_state, final norm applied)
// ---------------------------------------------------------------------------

struct GLayer {
    norms: [Vec<f32>; 4],
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
}

pub struct Gemma2 {
    embed: Mat,
    layers: Vec<GLayer>,
    norm: Vec<f32>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    softcap: Option<f32>,
    scale: f32,
    inv_freq: Vec<f32>,
}

/// Gemma RMSNorm weight `w` stored as `1 + w`.
fn gemma_norm(w: &Weights, name: &str, d: usize) -> Result<Vec<f32>, String> {
    Ok(w.vec_n(name, d)?.into_iter().map(|v| 1.0 + v).collect())
}

impl Gemma2 {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let c = read_config(dir)?;
        if c["model_type"] != "gemma2" {
            return Err(format!(
                "Sana text encoder: model_type {} is not gemma2",
                c["model_type"]
            ));
        }
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("text_encoder config: missing {k}"))
        };
        let (d, heads, kv_heads, head_dim, n) = (
            get("hidden_size")?,
            get("num_attention_heads")?,
            get("num_key_value_heads")?,
            get("head_dim")?,
            get("num_hidden_layers")?,
        );
        let w = component_weights(dir)?;
        let p = if w.has("model.embed_tokens.weight") {
            "model."
        } else {
            ""
        };
        let layers = (0..n)
            .map(|i| -> Result<GLayer, String> {
                let l = format!("{p}layers.{i}");
                let lin = |s: &str| Lin::load(&w, &format!("{l}.{s}"));
                let nm = |s: &str| gemma_norm(&w, &format!("{l}.{s}.weight"), d);
                Ok(GLayer {
                    norms: [
                        nm("input_layernorm")?,
                        nm("post_attention_layernorm")?,
                        nm("pre_feedforward_layernorm")?,
                        nm("post_feedforward_layernorm")?,
                    ],
                    q: lin("self_attn.q_proj")?,
                    k: lin("self_attn.k_proj")?,
                    v: lin("self_attn.v_proj")?,
                    o: lin("self_attn.o_proj")?,
                    gate: lin("mlp.gate_proj")?,
                    up: lin("mlp.up_proj")?,
                    down: lin("mlp.down_proj")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let theta = c["rope_theta"].as_f64().unwrap_or(10000.0);
        Ok(Self {
            embed: w.mat(&format!("{p}embed_tokens.weight"))?,
            layers,
            norm: gemma_norm(&w, &format!("{p}norm.weight"), d)?,
            heads,
            kv_heads,
            head_dim,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            softcap: c["attn_logit_softcapping"].as_f64().map(|v| v as f32),
            scale: (c["query_pre_attn_scalar"]
                .as_f64()
                .unwrap_or(head_dim as f64))
            .powf(-0.5) as f32,
            inv_freq: (0..head_dim / 2)
                .map(|i| theta.powf(-(2.0 * i as f64) / head_dim as f64) as f32)
                .collect(),
        })
    }

    /// Causal, soft-capped GQA attention of q [n, heads*hd] over k/v [n, kv_heads*hd].
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
        let (hd, h, kvh) = (self.head_dim, self.heads, self.kv_heads);
        let mut out = vec![0.0f32; n * h * hd];
        let o = OutPtr::new(&mut out);
        par_for(h * n, &|t| {
            let (head, i) = (t / n, t % n);
            let g = head / (h / kvh);
            let qi = &q[i * h * hd + head * hd..][..hd];
            let mut s: Vec<f32> = (0..=i)
                .map(|j| {
                    let kj = &k[j * kvh * hd + g * hd..][..hd];
                    let x = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * self.scale;
                    self.softcap.map_or(x, |c| (x / c).tanh() * c)
                })
                .collect();
            crate::model::simd::vector_softmax(&mut s);
            // SAFETY: row i of head `head` belongs to task t.
            let dst = unsafe { o.slice(i * h * hd + head * hd, hd) };
            dst.fill(0.0);
            for (j, p) in s.iter().enumerate() {
                let vj = &v[j * kvh * hd + g * hd..][..hd];
                dst.iter_mut().zip(vj).for_each(|(d, v)| *d += p * v);
            }
        });
        out
    }

    fn rope(&self, x: &mut [f32], heads: usize) {
        let (hd, half) = (self.head_dim, self.head_dim / 2);
        par_rows(x, heads * hd, &|pos, row| {
            let (sin, cos): (Vec<f32>, Vec<f32>) = self
                .inv_freq
                .iter()
                .map(|f| (pos as f32 * f).sin_cos())
                .unzip();
            for head in row.chunks_exact_mut(hd) {
                let (a, b) = head.split_at_mut(half);
                for i in 0..half {
                    let (x0, x1) = (a[i], b[i]);
                    a[i] = x0 * cos[i] - x1 * sin[i];
                    b[i] = x1 * cos[i] + x0 * sin[i];
                }
            }
        });
    }

    pub fn encode(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let (n, d) = (ids.len(), self.embed.cols);
        let normalizer = crate::model::bf16_to_f32(crate::model::f32_to_bf16((d as f32).sqrt()));
        let mut h = Vec::with_capacity(n * d);
        for &id in ids {
            if id as usize >= self.embed.rows {
                return Err(format!(
                    "Gemma token id {id} out of vocab {}",
                    self.embed.rows
                ));
            }
            h.extend(
                self.embed
                    .row(id as usize)
                    .iter()
                    .map(|&v| crate::model::bf16_to_f32(v) * normalizer),
            );
        }
        for l in &self.layers {
            let mut x = h.clone();
            rms_norm(&mut x, d, &l.norms[0], self.eps);
            let (mut q, mut k, v) = (l.q.fwd(&x, n), l.k.fwd(&x, n), l.v.fwd(&x, n));
            self.rope(&mut q, self.heads);
            self.rope(&mut k, self.kv_heads);
            let mut a = l.o.fwd(&self.attention(&q, &k, &v, n), n);
            rms_norm(&mut a, d, &l.norms[1], self.eps);
            nn::add_in_place(&mut h, &a);
            let mut x = h.clone();
            rms_norm(&mut x, d, &l.norms[2], self.eps);
            let mut g = l.gate.fwd(&x, n);
            nn::gelu_tanh(&mut g);
            let up = l.up.fwd(&x, n);
            g.iter_mut().zip(&up).for_each(|(g, u)| *g *= u);
            let mut m = l.down.fwd(&g, n);
            rms_norm(&mut m, d, &l.norms[3], self.eps);
            nn::add_in_place(&mut h, &m);
        }
        rms_norm(&mut h, d, &self.norm, self.eps);
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Sana linear DiT
// ---------------------------------------------------------------------------

struct Block {
    sst: Vec<f32>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    cq: Lin,
    ck: Lin,
    cv: Lin,
    co: Lin,
    inverted: Lin,
    depth_w: Vec<f32>,
    depth_b: Vec<f32>,
    point: Lin,
}

pub struct SanaDit {
    d: usize,
    heads: usize,
    head_dim: usize,
    cross_heads: usize,
    in_ch: usize,
    patch_proj: Lin,
    t_lin1: Lin,
    t_lin2: Lin,
    t_mod: Lin,
    cap1: Lin,
    cap2: Lin,
    cap_norm: Vec<f32>,
    blocks: Vec<Block>,
    sst_out: Vec<f32>,
    proj_out: Lin,
    timestep_scale: f32,
    gpu: graph_program::Cache,
    sample_size: usize,
}

/// Depthwise 3x3 (pad 1) over token-major x [h*w, c]; weight [c, 9].
fn depthwise_tokens(x: &[f32], c: usize, h: usize, w: usize, wt: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    par_rows(&mut out, c, &|p, row| {
        let (y, xx) = ((p / w) as isize, (p % w) as isize);
        row.copy_from_slice(b);
        for ky in -1..=1isize {
            for kx in -1..=1isize {
                let (iy, ix) = (y + ky, xx + kx);
                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                    continue;
                }
                let src = &x[(iy as usize * w + ix as usize) * c..][..c];
                let k = ((ky + 1) * 3 + kx + 1) as usize;
                for (ch, (o, s)) in row.iter_mut().zip(src).enumerate() {
                    *o += wt[ch * 9 + k] * s;
                }
            }
        }
    });
    out
}

/// ReLU linear attention per head: out_i = (sum_j v_j k_j^T) q_i / (sum_j k_j . q_i + 1e-15).
fn linear_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let d = heads * hd;
    let mut out = vec![0.0f32; n * d];
    let o = OutPtr::new(&mut out);
    par_for(heads, &|h| {
        let mut kv = vec![0.0f64; hd * hd];
        let mut ks = vec![0.0f64; hd];
        for j in 0..n {
            let kj = &k[j * d + h * hd..][..hd];
            let vj = &v[j * d + h * hd..][..hd];
            for (dd, &kk) in kj.iter().enumerate() {
                let kk = kk.max(0.0) as f64;
                ks[dd] += kk;
                for (c, &vv) in vj.iter().enumerate() {
                    kv[c * hd + dd] += vv as f64 * kk;
                }
            }
        }
        for i in 0..n {
            let qi: Vec<f64> = q[i * d + h * hd..][..hd]
                .iter()
                .map(|v| v.max(0.0) as f64)
                .collect();
            let den = ks.iter().zip(&qi).map(|(a, b)| a * b).sum::<f64>() + 1e-15;
            // SAFETY: head h's columns of row i belong to task h.
            let dst = unsafe { o.slice(i * d + h * hd, hd) };
            for (c, dv) in dst.iter_mut().enumerate() {
                *dv = (kv[c * hd..][..hd]
                    .iter()
                    .zip(&qi)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / den) as f32;
            }
        }
    });
    out
}

impl SanaDit {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let c = read_config(dir)?;
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("transformer config: missing {k}"))
        };
        if get("patch_size")? != 1
            || c["guidance_embeds"].as_bool() == Some(true)
            || !c["qk_norm"].is_null()
            || !c["interpolation_scale"].is_null()
        {
            return Err("Sana: only patch_size 1 without guidance embeds, qk_norm or positional embeddings is supported".into());
        }
        let (heads, head_dim, in_ch) = (
            get("num_attention_heads")?,
            get("attention_head_dim")?,
            get("in_channels")?,
        );
        if get("out_channels").unwrap_or(in_ch) != in_ch {
            return Err("Sana: out_channels != in_channels is not supported".into());
        }
        let d = heads * head_dim;
        let w = component_weights(dir)?;
        let blocks = (0..get("num_layers")?)
            .map(|i| -> Result<Block, String> {
                let p = format!("transformer_blocks.{i}");
                let lin = |s: &str| Lin::load(&w, &format!("{p}.{s}"));
                Ok(Block {
                    sst: w.vec_n(&format!("{p}.scale_shift_table"), 6 * d)?,
                    q: lin("attn1.to_q")?,
                    k: lin("attn1.to_k")?,
                    v: lin("attn1.to_v")?,
                    o: lin("attn1.to_out.0")?,
                    cq: lin("attn2.to_q")?,
                    ck: lin("attn2.to_k")?,
                    cv: lin("attn2.to_v")?,
                    co: lin("attn2.to_out.0")?,
                    inverted: lin("ff.conv_inverted")?,
                    depth_w: w.vec(&format!("{p}.ff.conv_depth.weight"))?,
                    depth_b: w.vec(&format!("{p}.ff.conv_depth.bias"))?,
                    point: lin("ff.conv_point")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            d,
            heads,
            head_dim,
            cross_heads: get("num_cross_attention_heads")?,
            in_ch,
            patch_proj: Lin::load(&w, "patch_embed.proj")?,
            t_lin1: Lin::load(&w, "time_embed.emb.timestep_embedder.linear_1")?,
            t_lin2: Lin::load(&w, "time_embed.emb.timestep_embedder.linear_2")?,
            t_mod: Lin::load(&w, "time_embed.linear")?,
            cap1: Lin::load(&w, "caption_projection.linear_1")?,
            cap2: Lin::load(&w, "caption_projection.linear_2")?,
            cap_norm: w.vec_n("caption_norm.weight", d)?,
            blocks,
            sst_out: w.vec_n("scale_shift_table", 2 * d)?,
            proj_out: Lin::load(&w, "proj_out")?,
            timestep_scale: c["timestep_scale"].as_f64().unwrap_or(1.0) as f32,
            gpu: Default::default(),
            sample_size: get("sample_size")?,
        })
    }

    /// Projected, normed caption [n, d] from encoder states [n, caption_channels].
    pub fn caption(&self, enc: &[f32], n: usize) -> Vec<f32> {
        let mut c = self.cap1.fwd(enc, n);
        nn::gelu_tanh(&mut c);
        let mut c = self.cap2.fwd(&c, n);
        rms_norm(&mut c, self.d, &self.cap_norm, 1e-5);
        c
    }

    /// Flow prediction for latents [C, h, w] at integer `timestep`, caption [n_cap, d] (projected).
    pub fn forward(
        &self,
        lat: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        cap: &[f32],
        n_cap: usize,
    ) -> Vec<f32> {
        let (d, c) = (self.d, self.in_ch);
        let n = h * w;
        let mut x = self.patch_proj.fwd(&nn::transpose(lat, c, n), n);
        let emb = self.t_lin2.fwd(
            &silu_of(
                &self
                    .t_lin1
                    .fwd(&timestep_features(timestep * self.timestep_scale, 256), 1),
            ),
            1,
        );
        let tm = self.t_mod.fwd(&silu_of(&emb), 1);
        let ch_dim = self.d / self.cross_heads;
        for b in &self.blocks {
            let m: Vec<f32> = b.sst.iter().zip(&tm).map(|(a, b)| a + b).collect();
            let ch = |k: usize| &m[k * d..(k + 1) * d];
            // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            let nx = norm_modulate(&x, d, 1e-6, ch(1), ch(0));
            let a = linear_attention(
                &b.q.fwd(&nx, n),
                &b.k.fwd(&nx, n),
                &b.v.fwd(&nx, n),
                n,
                self.heads,
                self.head_dim,
            );
            add_gated(&mut x, d, Some(ch(2)), &b.o.fwd(&a, n));
            let a = nn::attention(
                &b.cq.fwd(&x, n),
                &b.ck.fwd(cap, n_cap),
                &b.cv.fwd(cap, n_cap),
                n,
                n_cap,
                self.cross_heads,
                ch_dim,
                false,
            );
            add_gated(&mut x, d, None, &b.co.fwd(&a, n));
            let nx = norm_modulate(&x, d, 1e-6, ch(4), ch(3));
            let mut t = b.inverted.fwd(&nx, n);
            nn::silu(&mut t);
            let c2 = b.inverted.out_dim();
            let t = depthwise_tokens(&t, c2, h, w, &b.depth_w, &b.depth_b);
            let half = c2 / 2;
            let mut g = vec![0.0f32; n * half];
            par_rows(&mut g, half, &|i, row| {
                let src = &t[i * c2..][..c2];
                for (o, (a, gt)) in row.iter_mut().zip(src[..half].iter().zip(&src[half..])) {
                    *o = a * gt / (1.0 + (-gt).exp());
                }
            });
            add_gated(&mut x, d, Some(ch(5)), &b.point.fwd(&g, n));
        }
        let shift: Vec<f32> = self.sst_out[..d]
            .iter()
            .zip(&emb)
            .map(|(a, b)| a + b)
            .collect();
        let scale: Vec<f32> = self.sst_out[d..]
            .iter()
            .zip(&emb)
            .map(|(a, b)| a + b)
            .collect();
        let y = self
            .proj_out
            .fwd(&norm_modulate(&x, d, 1e-6, &scale, &shift), n);
        nn::transpose(&y, n, c)
    }
}

pub struct SanaPipeline {
    encoder: Gemma2,
    tokenizer: tokenizers::Tokenizer,
    dit: SanaDit,
    vae: DcAeDecoder,
    flow_shift: f64,
}

impl SanaPipeline {
    /// Loads a diffusers `SanaPipeline` snapshot directory.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let s = crate::model::dit::read_scheduler(dir)?;
        if s["_class_name"] != "DPMSolverMultistepScheduler"
            || s["algorithm_type"] != "dpmsolver++"
            || s["prediction_type"] != "flow_prediction"
            || s["use_flow_sigmas"].as_bool() != Some(true)
            || s["solver_order"].as_u64() != Some(2)
            || s["solver_type"] != "midpoint"
            || s["final_sigmas_type"] != "zero"
        {
            return Err(format!(
                "Sana: scheduler {} is not the DPM-Solver++ 2M flow configuration this engine runs",
                s
            ));
        }
        let tok_path = dir.join("tokenizer").join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("{}: {e}", tok_path.display()))?;
        let p = Self {
            encoder: Gemma2::load(&dir.join("text_encoder"))?,
            tokenizer,
            dit: SanaDit::load(&dir.join("transformer"))?,
            vae: DcAeDecoder::load(&dir.join("vae"))?,
            flow_shift: s["flow_shift"].as_f64().unwrap_or(3.0),
        };
        if p.vae.latent_channels() != p.dit.in_ch || p.dit.cap1.in_dim() != p.encoder.embed.cols {
            return Err(
                "Sana: VAE channels / text encoder width do not match the transformer".into(),
            );
        }
        Ok(p)
    }

    fn ids(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Gemma tokenizer: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Gemma states for `prompt` before the caption projection (lower-cased, stripped): BOS plus
    /// the real tokens; with the instruction prefix the pipeline keeps BOS plus the last 299 of
    /// `chi + 298` positions, i.e. the instruction's last token onwards. Padding is masked out of
    /// cross-attention, so only real tokens are kept.
    fn encode_raw(&self, prompt: &str, instruct: bool) -> Result<(Vec<f32>, usize), String> {
        let prompt = prompt.to_lowercase();
        let prompt = prompt.trim();
        let chi = CHI.join("\n");
        let (text, max_len, keep_from) = if instruct {
            let chi_len = self.ids(&chi)?.len();
            (format!("{chi}{prompt}"), chi_len + MAX_SEQ - 2, chi_len - 1)
        } else {
            (prompt.to_string(), MAX_SEQ, 1)
        };
        let mut ids = self.ids(&text)?;
        ids.truncate(max_len);
        let hs = self.encoder.encode(&ids)?;
        let dm = self.encoder.embed.cols;
        let mut sel = hs[..dm].to_vec();
        sel.extend_from_slice(&hs[keep_from.min(ids.len()) * dm..]);
        let n = sel.len() / dm;
        Ok((sel, n))
    }

    /// Caption states for the transformer: the projected, normed caption on the host, or on the
    /// GPU path the raw encoder states (the Metal program applies the projection).
    fn encode(&self, prompt: &str, instruct: bool, gpu: bool) -> Result<(Vec<f32>, usize), String> {
        let (sel, n) = self.encode_raw(prompt, instruct)?;
        if gpu {
            Ok((sel, n))
        } else {
            Ok((self.dit.caption(&sel, n), n))
        }
    }
}

impl ImageGenerator for SanaPipeline {
    fn summary(&self) -> String {
        format!(
            "Sana linear DiT ({} blocks, dim {}), Gemma 2 text encoder + DC-AE f{}c{}, bf16",
            self.dit.blocks.len(),
            self.dit.d,
            self.vae.factor(),
            self.vae.latent_channels()
        )
    }

    fn defaults(&self) -> ImageDefaults {
        let side = self.dit.sample_size * self.vae.factor();
        ImageDefaults {
            width: side,
            height: side,
            steps: 20,
            guidance: 4.5,
            multiple: self.vae.factor(),
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
        let f = self.vae.factor();
        // ponytail: no resolution binning (the pipeline snaps to aspect-ratio bins and resizes back);
        // sizes must already be multiples of the VAE factor. 8x8 latents keep DC-AE on linear attention.
        if width % f != 0
            || height % f != 0
            || width < 8 * f
            || height < 8 * f
            || width > 4096
            || height > 4096
        {
            return Err(format!(
                "Sana: width/height must be multiples of {f} in {}..=4096, got {width}x{height}",
                8 * f
            ));
        }
        if !(1..=200).contains(&steps) {
            return Err(format!("Sana: steps must be in 1..=200, got {steps}"));
        }
        let (h, w) = (height / f, width / f);
        let start = Instant::now();
        let gpu = crate::model::dit_graph::use_metal();
        let (cap, n_cap) = self.encode(&req.prompt, true, gpu)?;
        let neg = if guidance > 1.0 {
            Some(self.encode(req.negative_prompt.as_deref().unwrap_or(""), false, gpu)?)
        } else {
            None
        };
        println!(
            "  text encoder: {n_cap} caption tokens, {:.2}s",
            start.elapsed().as_secs_f64()
        );
        let sigmas = flow::dpm_flow_sigmas(steps, self.flow_shift);
        let mut rng = Rng::new(req.seed);
        let mut x: Vec<f32> = (0..self.dit.in_ch * h * w)
            .map(|_| rng.gaussian())
            .collect();
        let mut dpm = DpmSolver2::default();
        for i in 0..steps {
            let start = Instant::now();
            let t = flow::timestep(sigmas[i]).floor();
            let fwd = |x: &[f32], cap: &[f32], n: usize| {
                if gpu {
                    self.dit.forward_gpu(&self.dit.gpu, x, h, w, t, cap, n)
                } else {
                    Ok(self.dit.forward(x, h, w, t, cap, n))
                }
            };
            let mut v = fwd(&x, &cap, n_cap)?;
            if let Some((nc, nn)) = &neg {
                let vu = fwd(&x, nc, *nn)?;
                v.iter_mut()
                    .zip(&vu)
                    .for_each(|(c, u)| *c = u + guidance * (*c - u));
            }
            dpm.step(&mut x, &v, &sigmas, i);
            println!(
                "  step {}/{steps} (sigma {:.4}): {:.2}s",
                i + 1,
                sigmas[i],
                start.elapsed().as_secs_f64()
            );
        }
        self.vae.decode(&x, h, w)
    }
}
