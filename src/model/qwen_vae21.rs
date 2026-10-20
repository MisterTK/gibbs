//! Qwen-Image 2.1's `AutoencoderKLQwenImage21` decoder for one frame (diffusers
//! `autoencoder_kl_qwenimage21.py`): 64-channel latents at 1/16 resolution to RGBA. A single frame
//! never reaches Wan's temporal cache, so every 3-D op folds to its 2-D form: the first-frame
//! `upsample3d` skips `time_conv`, and the residual shortcut keeps the last of the two frames
//! `DupUp3D` emits (`first_chunk`). Weights stay f32; activations are channel-first `[c, h*w]`.

use crate::model::diff_graph::{ByShape, Gb, V};
use crate::model::dit::{component_weights, read_config};
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::graph::{Op, Tensor};
use crate::model::weights::Weights;
use std::path::Path;

struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
}

impl Conv {
    fn load(w: &Weights, p: &str, k: usize) -> Result<Self, String> {
        let shape = w.shape(&format!("{p}.weight"))?;
        let [c_out, c_in, kh, kw] = shape[..] else {
            return Err(format!(
                "VAE {p}.weight has shape {shape:?}, expected a 2-D conv"
            ));
        };
        if kh != k || kw != k {
            return Err(format!("VAE {p}.weight is {kh}x{kw}, expected {k}x{k}"));
        }
        Ok(Self {
            w: w.vec(&format!("{p}.weight"))?,
            b: w.vec_n(&format!("{p}.bias"), c_out)?,
            c_in,
            c_out,
            k,
        })
    }

    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        nn::conv2d(
            x,
            self.c_in,
            h,
            w,
            &self.w,
            Some(&self.b),
            self.c_out,
            self.k,
            1,
            self.k / 2,
        )
        .0
    }
}

/// `QwenImage21RMS_norm` then optionally SiLU, in place over `x [c, hw]`:
/// `x / max(||x_pixel||, 1e-12) * sqrt(c) * gamma` (L2 over channels, per pixel).
fn rms_norm(x: &mut [f32], hw: usize, gamma: &[f32], silu: bool) {
    let c = gamma.len();
    let mut ss = vec![0.0f32; hw];
    for ch in x.chunks_exact(hw) {
        ss.iter_mut().zip(ch).for_each(|(s, v)| *s += v * v);
    }
    let root_c = (c as f32).sqrt();
    ss.iter_mut()
        .for_each(|s| *s = root_c / s.sqrt().max(1e-12));
    par_rows(x, hw, &|ci, row| {
        let g = gamma[ci];
        row.iter_mut().zip(&ss).for_each(|(v, s)| *v *= s * g);
    });
    if silu {
        nn::silu(x);
    }
}

struct Resnet {
    norm1: Vec<f32>,
    conv1: Conv,
    norm2: Vec<f32>,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl Resnet {
    fn load(w: &Weights, p: &str) -> Result<Self, String> {
        let conv1 = Conv::load(w, &format!("{p}.conv1"), 3)?;
        let sc = format!("{p}.conv_shortcut");
        Ok(Self {
            norm1: w.vec_n(&format!("{p}.norm1.gamma"), conv1.c_in)?,
            norm2: w.vec_n(&format!("{p}.norm2.gamma"), conv1.c_out)?,
            conv2: Conv::load(w, &format!("{p}.conv2"), 3)?,
            shortcut: w
                .has(&format!("{sc}.weight"))
                .then(|| Conv::load(w, &sc, 1))
                .transpose()?,
            conv1,
        })
    }

    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let skip = self.shortcut.as_ref().map(|c| c.fwd(x, h, w));
        let mut t = x.to_vec();
        rms_norm(&mut t, h * w, &self.norm1, true);
        let mut t = self.conv1.fwd(&t, h, w);
        rms_norm(&mut t, h * w, &self.norm2, true);
        let mut t = self.conv2.fwd(&t, h, w);
        nn::add_in_place(&mut t, skip.as_deref().unwrap_or(x));
        t
    }
}

/// `QwenImage21AttentionBlock`: one head of width `c` over all `h*w` pixels.
struct Attn {
    norm: Vec<f32>,
    qkv: Conv,
    proj: Conv,
}

impl Attn {
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (c, hw) = (self.norm.len(), h * w);
        let mut t = x.to_vec();
        rms_norm(&mut t, hw, &self.norm, false);
        // [3c, hw] -> three [hw, c] tensors (q, k, v stacked along channels).
        let qkv = self.qkv.fwd(&t, h, w);
        let [q, k, v] = [0, 1, 2].map(|i| nn::transpose(&qkv[i * c * hw..(i + 1) * c * hw], c, hw));
        let a = nn::attention(&q, &k, &v, hw, hw, 1, c, false);
        let mut y = self.proj.fwd(&nn::transpose(&a, hw, c), h, w);
        nn::add_in_place(&mut y, x);
        y
    }
}

struct Up {
    resnets: Vec<Resnet>,
    /// `resample.1`: the 3x3 conv after the nearest 2x upsample.
    up: Option<Conv>,
    /// `DupUp3D` shortcut `(repeats, factor_t)` of an upsampling block.
    dup: Option<(usize, usize)>,
}

impl Up {
    fn fwd(&self, x: Vec<f32>, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let mut t = self.resnets[0].fwd(&x, h, w);
        for r in &self.resnets[1..] {
            t = r.fwd(&t, h, w);
        }
        let Some(conv) = &self.up else {
            return (t, h, w);
        };
        let c_in = conv.c_in;
        let mut y = conv.fwd(&nn::upsample_nearest2x(&t, c_in, h, w), 2 * h, 2 * w);
        let (repeats, ft) = self.dup.expect("upsampling blocks carry a shortcut");
        let c_out = conv.c_out;
        nn::add_in_place(&mut y, &dup_up(&x, c_out, h, w, repeats, ft));
        (y, 2 * h, 2 * w)
    }
}

/// `QwenImage21DupUp3D(first_chunk=True)` of one frame `x [c_in, h, w]`: channels repeated
/// `repeats` times, regrouped to `(c_out, factor_t, 2, 2)` and unfolded into 2x2 pixels, keeping the
/// last of the `factor_t` frames.
fn dup_up(
    x: &[f32],
    c_out: usize,
    h: usize,
    w: usize,
    repeats: usize,
    factor_t: usize,
) -> Vec<f32> {
    let (ow, mut out) = (2 * w, vec![0.0; c_out * 4 * h * w]);
    for (co, plane) in out.chunks_exact_mut(4 * h * w).enumerate() {
        for fy in 0..2 {
            for fx in 0..2 {
                let src = (((co * factor_t + factor_t - 1) * 2 + fy) * 2 + fx) / repeats;
                let s = &x[src * h * w..][..h * w];
                for y in 0..h {
                    for xx in 0..w {
                        plane[(2 * y + fy) * ow + 2 * xx + fx] = s[y * w + xx];
                    }
                }
            }
        }
    }
    out
}

// Graph emission (Metal program). Each `emit` mirrors the `fwd` above on `[c, h, w]` values.
impl Conv {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        b.conv2d(x, &self.w, Some(&self.b), self.c_out, self.k, 1, self.k / 2)
    }
}

impl Resnet {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        let skip = self.shortcut.as_ref().map_or(x, |c| c.emit(b, x));
        let t = b.rms_channels(x, &self.norm1);
        let t = b.silu(t);
        let t = self.conv1.emit(b, t);
        let t = b.rms_channels(t, &self.norm2);
        let t = b.silu(t);
        let t = self.conv2.emit(b, t);
        b.add(t, skip)
    }
}

impl Attn {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        let s = b.shape(x).to_vec();
        let (c, hw) = (self.norm.len(), s[1] * s[2]);
        let t = b.rms_channels(x, &self.norm);
        let flat = b.reshape(t, &[c, hw]);
        let tok = b.transpose(flat, &[1, 0]);
        let [q, k, v] = [0, 1, 2].map(|i| {
            b.linear_f32(tok, &self.qkv.w[i * c * c..(i + 1) * c * c], Some(&self.qkv.b[i * c..(i + 1) * c]), c, c)
        });
        let a = b.attention(q, k, v, 1);
        let o = b.linear_f32(a, &self.proj.w, Some(&self.proj.b), c, c);
        let o = b.transpose(o, &[1, 0]);
        let o = b.reshape(o, &s);
        b.add(x, o)
    }
}

impl Up {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        let mut t = self.resnets[0].emit(b, x);
        for r in &self.resnets[1..] {
            t = r.emit(b, t);
        }
        let Some(conv) = &self.up else {
            return t;
        };
        let u = b.upsample(t, 2, 2);
        let y = conv.emit(b, u);
        assert!(self.dup.is_some(), "upsampling blocks carry a shortcut");
        let s = b.shape(x).to_vec();
        let (h, w, c_out) = (s[1], s[2], conv.c_out);
        let idx = {
            let n = b.uniq("dup_idx");
            b.input_u32(&n, &[c_out * 4])
        };
        // `dup_up` as Gather + regroup: [c_out*4, h, w] -> [c_out, fy, fx, h, w] -> [c_out, h, fy, w, fx].
        let g = b.op("dup_gather", Op::Gather { axis: 0 }, &[x, idx], &[c_out * 4, h, w]);
        let g = b.reshape(g, &[c_out, 2, 2, h, w]);
        let g = b.transpose(g, &[0, 3, 1, 4, 2]);
        let g = b.reshape(g, &[c_out, 2 * h, 2 * w]);
        b.add(y, g)
    }
}

/// Gather indices of [`dup_up`]: output channel `co*4 + fy*2 + fx` reads input channel
/// `(((co*ft + ft-1)*2 + fy)*2 + fx) / repeats`.
fn dup_idx(c_out: usize, repeats: usize, ft: usize) -> Vec<u32> {
    (0..c_out * 4)
        .map(|i| (((i / 4 * ft + ft - 1) * 4 + i % 4) / repeats) as u32)
        .collect()
}

pub struct QwenVae21 {
    post_quant: Conv,
    conv_in: Conv,
    mid: (Resnet, Attn, Resnet),
    ups: Vec<Up>,
    norm_out: Vec<f32>,
    conv_out: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
    gpu: ByShape,
}

impl QwenVae21 {
    /// Reads a diffusers `vae/` directory (decoder and `post_quant_conv`; the encoder is not read).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = read_config(dir)?;
        if cfg["is_residual"].as_bool() != Some(true) || !cfg["patch_size"].is_null() {
            return Err(
                "Qwen-Image 2.1 VAE: only the residual, unpatched variant is supported".into(),
            );
        }
        let ints = |k: &str| -> Result<Vec<usize>, String> {
            cfg[k]
                .as_array()
                .ok_or_else(|| format!("vae config: missing {k}"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|v| v as usize)
                        .ok_or_else(|| format!("vae config: bad {k}"))
                })
                .collect()
        };
        let floats = |k: &str, n: usize| -> Result<Vec<f32>, String> {
            let v: Vec<f32> = cfg[k]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_f64().map(|v| v as f32))
                .collect();
            (v.len() == n)
                .then_some(v)
                .ok_or_else(|| format!("vae config: {k} is not {n} numbers"))
        };
        let (mult, z) = (
            ints("dim_mult")?,
            cfg["z_dim"].as_u64().unwrap_or(64) as usize,
        );
        let base = cfg["decoder_base_dim"]
            .as_u64()
            .or(cfg["base_dim"].as_u64())
            .ok_or("vae config: missing base_dim")? as usize;
        let blocks = cfg["num_res_blocks"].as_u64().unwrap_or(2) as usize;
        // Decoder levels run the encoder's temporal downsampling in reverse.
        let temporal: Vec<bool> = cfg["temperal_downsample"]
            .as_array()
            .ok_or("vae config: missing temperal_downsample")?
            .iter()
            .rev()
            .map(|v| v == true)
            .collect();
        if temporal.len() + 1 != mult.len() {
            return Err("vae config: temperal_downsample does not match dim_mult".into());
        }
        Self::open(
            &component_weights(dir)?,
            &mult,
            base,
            blocks,
            &temporal,
            floats("latents_mean", z)?,
            floats("latents_std", z)?,
        )
    }

    fn open(
        w: &Weights,
        mult: &[usize],
        base: usize,
        blocks: usize,
        temporal: &[bool],
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Self, String> {
        let dims: Vec<usize> = std::iter::once(mult[mult.len() - 1])
            .chain(mult.iter().rev().copied())
            .map(|m| m * base)
            .collect();
        let mid = |p: &str| Resnet::load(w, &format!("decoder.mid_block.resnets.{p}"));
        let attn = "decoder.mid_block.attentions.0";
        let mut ups = Vec::new();
        for (i, io) in dims.windows(2).enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            let up = i + 1 != mult.len();
            ups.push(Up {
                resnets: (0..=blocks)
                    .map(|j| Resnet::load(w, &format!("{p}.resnets.{j}")))
                    .collect::<Result<_, _>>()?,
                up: up
                    .then(|| Conv::load(w, &format!("{p}.upsampler.resample.1"), 3))
                    .transpose()?,
                dup: up.then(|| {
                    let ft = if temporal[i] { 2 } else { 1 };
                    (io[1] * ft * 4 / io[0], ft)
                }),
            });
        }
        let last = dims[dims.len() - 1];
        Ok(Self {
            post_quant: Conv::load(w, "post_quant_conv", 1)?,
            conv_in: Conv::load(w, "decoder.conv_in", 3)?,
            mid: (
                mid("0")?,
                Attn {
                    norm: w.vec_n(&format!("{attn}.norm.gamma"), dims[0])?,
                    qkv: Conv::load(w, &format!("{attn}.to_qkv"), 1)?,
                    proj: Conv::load(w, &format!("{attn}.proj"), 1)?,
                },
                mid("1")?,
            ),
            ups,
            norm_out: w.vec_n("decoder.norm_out.gamma", last)?,
            conv_out: Conv::load(w, "decoder.conv_out", 3)?,
            mean,
            std,
            gpu: ByShape::default(),
        })
    }

    pub fn latent_channels(&self) -> usize {
        self.mean.len()
    }

    /// Native CPU decoder over de-normalised latents `z [z, h, w]` -> `[4, 16h, 16w]` (RGBA planes).
    fn decode_cpu(&self, z: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        crate::hardware::cpu_compute::enter("qwen_vae21::decode");
        let x = self.post_quant.fwd(z, h, w);
        let mut x = self.conv_in.fwd(&x, h, w);
        x = self.mid.0.fwd(&x, h, w);
        x = self.mid.1.fwd(&x, h, w);
        x = self.mid.2.fwd(&x, h, w);
        let (mut h, mut w) = (h, w);
        for up in &self.ups {
            (x, h, w) = up.fwd(x, h, w);
        }
        rms_norm(&mut x, h * w, &self.norm_out, true);
        (self.conv_out.fwd(&x, h, w), h, w)
    }

    /// Metal decoder (same outputs as [`Self::decode_cpu`]); one program per latent size. The
    /// `DupUp3D` channel regrouping is a Gather whose U32 indices are extra program inputs.
    fn decode_gpu(&self, z: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize), String> {
        let net = self.gpu.get(&[h, w], || {
            let mut b = Gb::new();
            let x = b.input("z", &[self.latent_channels(), h, w]);
            let x = self.post_quant.emit(&mut b, x);
            let mut x = self.conv_in.emit(&mut b, x);
            x = self.mid.0.emit(&mut b, x);
            x = self.mid.1.emit(&mut b, x);
            x = self.mid.2.emit(&mut b, x);
            for up in &self.ups {
                x = up.emit(&mut b, x);
            }
            x = b.rms_channels(x, &self.norm_out);
            x = b.silu(x);
            x = self.conv_out.emit(&mut b, x);
            b.output(x);
            b.compile()
        })?;
        let mut ins = vec![Tensor::F32(z.to_vec())];
        ins.extend(self.ups.iter().filter_map(|u| u.up.as_ref().zip(u.dup)).map(|(c, (r, ft))| Tensor::U32(dup_idx(c.c_out, r, ft))));
        let out = net.run_t(ins)?.remove(0);
        let ups = self.ups.iter().filter(|u| u.up.is_some()).count();
        Ok((out, h << ups, w << ups))
    }

    /// Normalised latents `[z, h, w]` (the transformer's output space) -> an image of 16h x 16w:
    /// `latents * std + mean`, decode, clamp to [-1, 1], RGBA flattened over white (an RGB image
    /// cannot carry alpha).
    pub fn decode(&self, latents: &[f32], h: usize, w: usize) -> Result<ImageBuffer, String> {
        let z = self.latent_channels();
        if latents.len() != z * h * w {
            return Err(format!(
                "VAE decode: latents len {} != {z}x{h}x{w}",
                latents.len()
            ));
        }
        let mut z = Vec::with_capacity(latents.len());
        for (c, plane) in latents.chunks_exact(h * w).enumerate() {
            z.extend(plane.iter().map(|v| v * self.std[c] + self.mean[c]));
        }
        let (x, h, w) = if crate::model::diff_graph::gpu_mode() {
            self.decode_gpu(&z, h, w)?
        } else {
            self.decode_cpu(&z, h, w)
        };
        let px =
            |c: usize, i: usize| (x[c * h * w + i].clamp(-1.0, 1.0) / 2.0 + 0.5).clamp(0.0, 1.0);
        let mut data = Vec::with_capacity(3 * h * w);
        for i in 0..h * w {
            let a = px(3, i);
            data.extend((0..3).map(|c| ((a * px(c, i) + (1.0 - a)) * 255.0).round() as u8));
        }
        ImageBuffer::from_rgb(w, h, data)
    }
}

/// Tiny synthetic decoder (mult [1,2,2,4], base 4, mixed temporal flags) for the CPU-free fixtures and parity
/// tests (shared with `tests/cpu_free_matrix.rs`).
pub mod fixture {
    use super::*;
    use crate::model::diff_graph::testutil::{synth, Lcg};

    /// The decoder and its latent channel count.
    pub fn tiny() -> (QwenVae21, usize) {
        let (mult, base, zc, temporal) = ([1usize, 2, 2, 4], 4usize, 6usize, [true, false, true]);
        let mut s: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |s: &mut Vec<_>, p: &str, i: usize, o: usize, k: usize| {
            s.push((format!("{p}.weight"), vec![o, i, k, k]));
            s.push((format!("{p}.bias"), vec![o]));
        };
        let res = |s: &mut Vec<_>, p: &str, i: usize, o: usize| {
            s.push((format!("{p}.norm1.gamma"), vec![i]));
            conv(s, &format!("{p}.conv1"), i, o, 3);
            s.push((format!("{p}.norm2.gamma"), vec![o]));
            conv(s, &format!("{p}.conv2"), o, o, 3);
            if i != o {
                conv(s, &format!("{p}.conv_shortcut"), i, o, 1);
            }
        };
        let dims: Vec<usize> = std::iter::once(mult[3]).chain(mult.iter().rev().copied()).map(|m| m * base).collect();
        conv(&mut s, "post_quant_conv", zc, zc, 1);
        conv(&mut s, "decoder.conv_in", zc, dims[0], 3);
        res(&mut s, "decoder.mid_block.resnets.0", dims[0], dims[0]);
        res(&mut s, "decoder.mid_block.resnets.1", dims[0], dims[0]);
        let a = "decoder.mid_block.attentions.0";
        s.push((format!("{a}.norm.gamma"), vec![dims[0]]));
        conv(&mut s, &format!("{a}.to_qkv"), dims[0], 3 * dims[0], 1);
        conv(&mut s, &format!("{a}.proj"), dims[0], dims[0], 1);
        for (i, io) in dims.windows(2).enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            for j in 0..3 {
                res(&mut s, &format!("{p}.resnets.{j}"), if j == 0 { io[0] } else { io[1] }, io[1]);
            }
            if i + 1 != mult.len() {
                conv(&mut s, &format!("{p}.upsampler.resample.1"), io[1], io[1], 3);
            }
        }
        s.push(("decoder.norm_out.gamma".into(), vec![dims[4]]));
        conv(&mut s, "decoder.conv_out", dims[4], 4, 3);
        let mut r = synth(&s, 21);
        let mut g = Lcg(9);
        for (n, (_, v)) in r.0.iter_mut() {
            if n.ends_with(".gamma") {
                v.iter_mut().for_each(|x| *x = 1.0 + 0.2 * g.next());
            }
        }
        let w = Weights::from_f32(s, |n| Ok(r.0[n].1.clone())).unwrap();
        let (mean, std) = (Lcg(1).vec(zc, 0.5), Lcg(2).vec(zc, 0.5).iter().map(|v| 1.0 + v.abs()).collect());
        (QwenVae21::open(&w, &mult, base, 2, &temporal, mean, std).unwrap(), zc)
    }

    /// Public `decode` of the tiny net (Metal when the CPU tripwire is armed) plus a finiteness check of the
    /// raw Metal output.
    pub fn tiny_on_gpu() -> Result<(), String> {
        let (vae, zc) = tiny();
        let (h, w) = (3, 2);
        let z = Lcg(3).vec(zc * h * w, 1.0);
        let img = vae.decode(&z, h, w)?;
        let f = 1 << vae.ups.iter().filter(|u| u.up.is_some()).count();
        if (img.width, img.height) != (f * w, f * h) {
            return Err(format!("qwen_vae21: {}x{} image", img.width, img.height));
        }
        if vae.decode_gpu(&z, h, w)?.0.iter().any(|v| !v.is_finite()) {
            return Err("qwen_vae21: non-finite Metal output".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `DupUp3D` written as the reference's literal repeat_interleave / view / permute chain on a
    /// `[1, c_in, 1, h, w]` tensor, then `first_chunk` slicing.
    fn dup_up_reference(
        x: &[f32],
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        ft: usize,
    ) -> Vec<f32> {
        let repeats = c_out * ft * 4 / c_in;
        let rep: Vec<f32> = (0..c_in * repeats)
            .flat_map(|c| x[(c / repeats) * h * w..][..h * w].to_vec())
            .collect();
        // view(c_out, ft, 2, 2, 1, h, w) -> permute(0, 4, 1, 5, 2, 6, 3) -> (c_out, ft, h*2, w*2) frames.
        let idx = |co, f, fy, fx, y, xx| ((((co * ft + f) * 2 + fy) * 2 + fx) * h + y) * w + xx;
        let mut out = vec![0.0; c_out * h * w * 4];
        for co in 0..c_out {
            for y in 0..h {
                for fy in 0..2 {
                    for xx in 0..w {
                        for fx in 0..2 {
                            out[(co * 2 * h + 2 * y + fy) * 2 * w + 2 * xx + fx] =
                                rep[idx(co, ft - 1, fy, fx, y, xx)];
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn dup_up_matches_reference_permutation() {
        let (h, w) = (3, 2);
        for (c_in, c_out, ft) in [(8, 8, 2), (8, 4, 2), (4, 2, 1), (6, 3, 2)] {
            let x: Vec<f32> = (0..c_in * h * w).map(|i| i as f32).collect();
            let repeats = c_out * ft * 4 / c_in;
            assert_eq!(
                dup_up(&x, c_out, h, w, repeats, ft),
                dup_up_reference(&x, c_in, c_out, h, w, ft),
                "{c_in}->{c_out} ft{ft}"
            );
        }
    }

    #[test]
    fn rms_norm_is_unit_l2_times_sqrt_c() {
        // Two channels, two pixels: pixel 0 = (3, 4), pixel 1 = (0, 0) (clamped norm, stays 0).
        let mut x = vec![3.0, 0.0, 4.0, 0.0];
        rms_norm(&mut x, 2, &[1.0, 2.0], false);
        let s = 2f32.sqrt() / 5.0;
        assert!(
            (x[0] - 3.0 * s).abs() < 1e-6 && (x[2] - 4.0 * s * 2.0).abs() < 1e-6,
            "{x:?}"
        );
        assert_eq!((x[1], x[3]), (0.0, 0.0));
    }

    /// Decoding real latents needs the diffusers `vae/` of Qwen/Qwen-Image-2.1; without it this only
    /// reports a skip. A constant latent must decode to a finite, in-range RGB image of 16x the size.
    #[test]
    fn real_vae_decodes_a_latent() {
        let Some(dir) = ["disk".to_string()]
            .into_iter()
            .chain(std::env::var("GIBBS_TEST_STORAGE"))
            .chain(std::env::var("HF_HUB_CACHE"))
            .map(|r| Path::new(&r).join("models--Qwen--Qwen-Image-2.1/snapshots/main/vae"))
            .find(|d| d.join("config.json").exists())
        else {
            eprintln!(
                "SKIP: Qwen/Qwen-Image-2.1 vae/ not in disk/, GIBBS_TEST_STORAGE or HF_HUB_CACHE"
            );
            return;
        };
        let vae = QwenVae21::load(&dir).unwrap();
        let (h, w) = (4, 4);
        let img = vae
            .decode(&vec![0.0; vae.latent_channels() * h * w], h, w)
            .unwrap();
        assert_eq!(
            (img.width, img.height, img.data.len()),
            (16 * w, 16 * h, 3 * 256 * h * w)
        );
    }

    /// Tiny synthetic decoder (mult [1,2,2,4], base 4, mixed temporal flags): CPU vs the Metal
    /// graph program (f32 vs f32: 1e-3), with every node on the GPU.
    #[test]
    fn gpu_matches_cpu() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        use crate::model::diff_graph::testutil::{rel_err, Lcg};
        let (vae, zc) = fixture::tiny();
        let (h, wd) = (3, 2);
        let z = Lcg(3).vec(zc * h * wd, 1.0);
        let (cpu, ch, cw) = vae.decode_cpu(&z, h, wd);
        let (gpu, gh, gw) = vae.decode_gpu(&z, h, wd).unwrap();
        assert_eq!((ch, cw), (gh, gw));
        let e = rel_err(&gpu, &cpu);
        println!("qwen_vae21 decode rel err {e:e}");
        assert!(e < 1e-3, "decode rel err {e}");
        let net = vae.gpu.get(&[h, wd], || Err("cached".into())).unwrap();
        assert_eq!(net.cpu_node_count(), 0);
    }
}
