//! Gemma 4 vision tower (`gemma4_vision`) and its image processor: aspect-preserving resize to a
//! soft-token budget, 16 px patches with learned 2-D position embeddings, a bidirectional ViT with
//! axial 2-D RoPE, 3x3 average pooling, then `embed_vision` into the text width.

use super::ops::{gelu_mul, linear, rms_norm, rope, rope_table, Clipped};
use crate::model::image::ImageBuffer;
use crate::model::weights::{Mat, Weights};
use crate::model::{bf16_to_f32, nn};
use serde_json::Value;

struct Layer {
    q: Clipped,
    k: Clipped,
    v: Clipped,
    o: Clipped,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Clipped,
    up: Clipped,
    down: Clipped,
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
}

pub struct Vision {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    patch: usize,
    pool: usize,
    /// Soft tokens per image at most (`image_processor.max_soft_tokens`).
    max_soft: usize,
    eps: f32,
    input_proj: Mat,
    /// `[2, position_embedding_size * hidden]`: x table, y table.
    pos_table: Mat,
    layers: Vec<Layer>,
    inv_freq: Vec<f32>,
    /// `embed_vision.embedding_projection` (vision hidden -> text hidden).
    proj: Mat,
}

/// Largest `(h, w)` keeping the aspect ratio, both multiples of `pool * patch`, within
/// `max_patches` patches (transformers `get_aspect_ratio_preserving_size`).
pub fn target_size(
    height: usize,
    width: usize,
    patch: usize,
    max_patches: usize,
    pool: usize,
) -> Result<(usize, usize), String> {
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = pool * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    let max_side = (max_patches / (pool * pool)) * side;
    match (th, tw) {
        (0, 0) => return Err(format!("image {width}x{height} resizes to 0x0")),
        (0, _) => {
            th = side;
            tw = ((width / height) * side).min(max_side);
        }
        (_, 0) => {
            tw = side;
            th = ((height / width) * side).min(max_side);
        }
        _ => {}
    }
    if (th * tw) as f64 > target_px {
        return Err(format!(
            "image {width}x{height} exceeds {max_patches} patches"
        ));
    }
    Ok((th, tw))
}

/// Soft tokens image file `bytes` becomes: its [`target_size`] patch grid pooled `pool x pool`.
pub fn soft_tokens(
    bytes: &[u8],
    patch: usize,
    max_soft: usize,
    pool: usize,
) -> Result<usize, String> {
    let img = ImageBuffer::decode(bytes)?;
    let (th, tw) = target_size(img.height, img.width, patch, max_soft * pool * pool, pool)?;
    Ok((th / (patch * pool)) * (tw / (patch * pool)))
}

impl Vision {
    pub fn count(&self, bytes: &[u8]) -> Result<usize, String> {
        soft_tokens(bytes, self.patch, self.max_soft, self.pool)
    }

    pub fn load(w: &Weights, vc: &Value, processor: &Value) -> Result<Self, String> {
        let u = |k: &str| {
            vc[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let head_dim = u("head_dim")?;
        if vc["standardize"].as_bool().unwrap_or(false) {
            return Err("vision_config.standardize is not supported".into());
        }
        if u("num_key_value_heads")? != heads {
            return Err("vision tower with grouped KV heads is not supported".into());
        }
        let ip = &processor["image_processor"];
        let max_soft = ip["max_soft_tokens"].as_u64().unwrap_or(280) as usize;
        let p = "model.vision_tower";
        let n = u("num_hidden_layers")?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let lp = format!("{p}.encoder.layers.{i}");
            let c = |s: &str| Clipped::load(w, &format!("{lp}.{s}"));
            let v = |s: &str, d: usize| w.vec_n(&format!("{lp}.{s}.weight"), d);
            layers.push(Layer {
                q: c("self_attn.q_proj")?,
                k: c("self_attn.k_proj")?,
                v: c("self_attn.v_proj")?,
                o: c("self_attn.o_proj")?,
                q_norm: v("self_attn.q_norm", head_dim)?,
                k_norm: v("self_attn.k_norm", head_dim)?,
                gate: c("mlp.gate_proj")?,
                up: c("mlp.up_proj")?,
                down: c("mlp.down_proj")?,
                input_ln: v("input_layernorm", hidden)?,
                post_attn_ln: v("post_attention_layernorm", hidden)?,
                pre_ff_ln: v("pre_feedforward_layernorm", hidden)?,
                post_ff_ln: v("post_feedforward_layernorm", hidden)?,
            });
        }
        let theta = vc["rope_parameters"]["rope_theta"]
            .as_f64()
            .unwrap_or(100.0);
        // Axial RoPE: each of x and y rotates half the head with head_dim/4 frequencies.
        let spatial = head_dim / 2;
        let inv_freq = (0..spatial / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / spatial as f64)) as f32)
            .collect();
        let patch = u("patch_size")?;
        Ok(Self {
            hidden,
            heads,
            head_dim,
            patch,
            pool: u("pooling_kernel_size")?,
            max_soft,
            eps: vc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            input_proj: w.mat(&format!("{p}.patch_embedder.input_proj.weight"))?,
            pos_table: w.mat(&format!("{p}.patch_embedder.position_embedding_table"))?,
            layers,
            inv_freq,
            proj: w.mat("model.embed_vision.embedding_projection.weight")?,
        })
    }

    /// Encodes one image (file bytes) to `[n, text_hidden]` soft tokens; returns `(rows, n)`.
    pub fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let img = ImageBuffer::decode(bytes)?;
        let (p, k) = (self.patch, self.pool);
        let (th, tw) = target_size(img.height, img.width, p, self.max_soft * k * k, k)?;
        let img = if (th, tw) == (img.height, img.width) {
            img
        } else {
            img.resize(tw, th)
        };
        let (ph, pw) = (th / p, tw / p);
        let np = ph * pw;
        let d = self.hidden;
        // Patches: (py, px, c) inside each patch, rescaled to [0, 1] then to [-1, 1].
        let mut pix = vec![0.0f32; np * p * p * 3];
        for (i, patch) in pix.chunks_exact_mut(p * p * 3).enumerate() {
            let (gy, gx) = (i / pw, i % pw);
            for y in 0..p {
                for x in 0..p {
                    let src = ((gy * p + y) * tw + gx * p + x) * 3;
                    for c in 0..3 {
                        patch[(y * p + x) * 3 + c] = 2.0 * (img.data[src + c] as f32 / 255.0 - 0.5);
                    }
                }
            }
        }
        let mut h = linear(&pix, np, &self.input_proj);
        let table = self.pos_table.data();
        let tlen = self.pos_table.cols;
        for (i, row) in h.chunks_exact_mut(d).enumerate() {
            let (gx, gy) = (i % pw, i / pw);
            let xe = &table[gx * d..][..d];
            let ye = &table[tlen + gy * d..][..d];
            for ((v, &a), &b) in row.iter_mut().zip(xe).zip(ye) {
                *v += bf16_to_f32(a) + bf16_to_f32(b);
            }
        }
        let ropes: Vec<_> = (0..np)
            .map(|i| {
                (
                    rope_table((i % pw) as f32, &self.inv_freq),
                    rope_table((i / pw) as f32, &self.inv_freq),
                )
            })
            .collect();
        let (hd, nh) = (self.head_dim, self.heads);
        let half = hd / 2;
        for l in &self.layers {
            let mut a = h.clone();
            rms_norm(&mut a, d, Some(&l.input_ln), self.eps);
            let mut q = l.q.forward(&a, np);
            let mut kk = l.k.forward(&a, np);
            let mut v = l.v.forward(&a, np);
            rms_norm(&mut q, hd, Some(&l.q_norm), self.eps);
            rms_norm(&mut kk, hd, Some(&l.k_norm), self.eps);
            rms_norm(&mut v, hd, None, self.eps);
            for (i, ((cx, sx), (cy, sy))) in ropes.iter().enumerate() {
                for t in [&mut q, &mut kk] {
                    for head in t[i * nh * hd..(i + 1) * nh * hd].chunks_exact_mut(hd) {
                        rope(&mut head[..half], cx, sx);
                        rope(&mut head[half..], cy, sy);
                    }
                }
            }
            // Scale 1: undo `nn::attention`'s 1/sqrt(head_dim).
            let s = (hd as f32).sqrt();
            q.iter_mut().for_each(|v| *v *= s);
            let att = nn::attention(&q, &kk, &v, np, np, nh, hd, false);
            let mut o = l.o.forward(&att, np);
            rms_norm(&mut o, d, Some(&l.post_attn_ln), self.eps);
            h.iter_mut().zip(&o).for_each(|(h, o)| *h += o);
            let mut a = h.clone();
            rms_norm(&mut a, d, Some(&l.pre_ff_ln), self.eps);
            let mut g = l.gate.forward(&a, np);
            gelu_mul(&mut g, &l.up.forward(&a, np));
            let mut m = l.down.forward(&g, np);
            rms_norm(&mut m, d, Some(&l.post_ff_ln), self.eps);
            h.iter_mut().zip(&m).for_each(|(h, m)| *h += m);
        }
        // k x k average pooling over the patch grid (row-major pooled grid), times sqrt(hidden).
        let (oh, ow) = (ph / k, pw / k);
        let n = oh * ow;
        let mut pooled = vec![0.0f32; n * d];
        let scale = (d as f32).sqrt() / (k * k) as f32;
        for (i, row) in h.chunks_exact(d).enumerate() {
            let (gy, gx) = (i / pw, i % pw);
            let dst = &mut pooled[((gy / k) * ow + gx / k) * d..][..d];
            dst.iter_mut().zip(row).for_each(|(o, v)| *o += v * scale);
        }
        rms_norm(&mut pooled, d, None, self.eps);
        Ok((linear(&pooled, n, &self.proj), n))
    }
}

#[cfg(test)]
mod tests {
    use super::target_size;

    #[test]
    fn target_size_fills_the_budget_on_pool_multiples() {
        // 2520 patches = 280 soft tokens of 3x3 16-px patches.
        let (h, w) = target_size(480, 640, 16, 2520, 3).unwrap();
        assert_eq!((h % 48, w % 48), (0, 0));
        assert!(h * w <= 2520 * 256 && (h + 48) * (w + 48) > 2520 * 256);
        assert_eq!(target_size(768, 768, 16, 2520, 3).unwrap(), (768, 768));
        // Extreme aspect: one side clamps to a single pooled cell.
        assert_eq!(target_size(10, 4000, 16, 2520, 3).unwrap().0, 48);
    }
}
