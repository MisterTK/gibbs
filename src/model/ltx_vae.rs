//! LTX-Video's causal 3D VAE (diffusers `AutoencoderKLLTXVideo`, v0.9.0 layout: no timestep
//! conditioning, no noise injection, non-causal decoder, causal encoder), f32 on the CPU.
//!
//! Reads the diffusers names (`vae/diffusion_pytorch_model.safetensors`) or the original ones
//! (`LTX-Video-VAE-*.safetensors`, `up_blocks.N.res_blocks`). Video tensors are `[frames][channels][h*w]`
//! so every frame is a `nn::conv2d` input; a 3x3x3 conv is three temporal taps of 2D convs.
//!
//! Reference: diffusers `models/autoencoders/autoencoder_kl_ltx.py`, `scripts/convert_ltx_to_diffusers.py`.

use crate::graph::Op;
use crate::model::diff_graph::{ByShape, Gb, V};
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::weights::Weights;
use std::path::Path;

#[cfg(test)]
mod spec_parity;

/// `[frames][channels][h * w]`.
pub struct Vid {
    pub d: Vec<f32>,
    pub c: usize,
    pub f: usize,
    pub h: usize,
    pub w: usize,
}

impl Vid {
    fn frame(&self, f: usize) -> &[f32] {
        let n = self.c * self.h * self.w;
        &self.d[f * n..(f + 1) * n]
    }
}

/// `Conv3d` weight `[co, ci, kt, k, k]` split into its `kt` temporal taps `[co, ci, k, k]`.
struct Conv {
    taps: Vec<Vec<f32>>,
    b: Vec<f32>,
    ci: usize,
    co: usize,
    k: usize,
    stride: usize,
    full: Vec<f32>,
}

impl Conv {
    fn load(w: &Weights, base: &str, stride: usize) -> Result<Self, String> {
        let shape = w.shape(&format!("{base}.weight"))?;
        let [co, ci, kt, k, k2] = shape[..] else {
            return Err(format!(
                "{base}.weight has shape {shape:?}, expected a 3D conv"
            ));
        };
        if kt != k || k != k2 || !matches!(k, 1 | 3) {
            return Err(format!(
                "{base}.weight: kernel {kt}x{k}x{k2} is not 1 or 3 cubed"
            ));
        }
        let all = w.vec(&format!("{base}.weight"))?;
        Ok(Self::new(
            &all,
            w.vec_n(&format!("{base}.bias"), co)?,
            [co, ci, k],
            stride,
        ))
    }

    /// `weight` `[co, ci, k, k, k]` split into its `k` temporal taps.
    fn new(weight: &[f32], b: Vec<f32>, [co, ci, k]: [usize; 3], stride: usize) -> Self {
        let kk = k * k;
        let taps = (0..k)
            .map(|t| {
                let mut v = Vec::with_capacity(co * ci * kk);
                for oi in 0..co * ci {
                    v.extend_from_slice(&weight[(oi * k + t) * kk..][..kk]);
                }
                v
            })
            .collect();
        Self {
            taps,
            b,
            ci,
            co,
            k,
            stride,
            full: weight.to_vec(),
        }
    }

    /// Replicate-padded temporal conv: `causal` pads `k - 1` copies of the first frame on the left,
    /// otherwise `(k - 1) / 2` copies of each end.
    fn fwd(&self, x: &Vid, causal: bool) -> Vid {
        assert_eq!(
            x.c, self.ci,
            "conv: {} input channels, expected {}",
            x.c, self.ci
        );
        let (k, s) = (self.k, self.stride);
        let left = if causal { k - 1 } else { (k - 1) / 2 };
        let padded = if k == 1 { x.f } else { x.f + k - 1 };
        let fo = (padded - k) / s + 1;
        let pad = k / 2;
        let (mut oh, mut ow) = (0, 0);
        let mut d = Vec::new();
        for t in 0..fo {
            let mut acc: Option<Vec<f32>> = None;
            for (kt, w) in self.taps.iter().enumerate() {
                let src = (t * s + kt).saturating_sub(left).min(x.f - 1);
                let (y, h, wd) = nn::conv2d(
                    x.frame(src),
                    self.ci,
                    x.h,
                    x.w,
                    w,
                    acc.is_none().then_some(&self.b[..]),
                    self.co,
                    k,
                    s,
                    pad,
                );
                (oh, ow) = (h, wd);
                match &mut acc {
                    Some(a) => nn::add_in_place(a, &y),
                    None => acc = Some(y),
                }
            }
            d.extend(acc.expect("a conv has at least one tap"));
        }
        Vid {
            d,
            c: self.co,
            f: fo,
            h: oh,
            w: ow,
        }
    }
}

/// Per pixel `x * rsqrt(mean_c(x^2) + eps)` (torch `RMSNorm` without affine over the channel dim).
fn rms_channels(x: &mut Vid, eps: f32) {
    let (hw, c) = (x.h * x.w, x.c);
    for fr in x.d.chunks_exact_mut(c * hw) {
        let mut ss = vec![0.0f32; hw];
        for ch in fr.chunks_exact(hw) {
            ss.iter_mut().zip(ch).for_each(|(s, v)| *s += v * v);
        }
        ss.iter_mut()
            .for_each(|s| *s = 1.0 / (*s / c as f32 + eps).sqrt());
        for ch in fr.chunks_exact_mut(hw) {
            ch.iter_mut().zip(&ss).for_each(|(v, s)| *v *= s);
        }
    }
}

/// Per pixel `LayerNorm` over channels with affine `g`, `b`.
fn layer_norm_channels(x: &mut Vid, g: &[f32], b: &[f32], eps: f32) {
    let (hw, c) = (x.h * x.w, x.c);
    for fr in x.d.chunks_exact_mut(c * hw) {
        let (mut mean, mut var) = (vec![0.0f32; hw], vec![0.0f32; hw]);
        for ch in fr.chunks_exact(hw) {
            mean.iter_mut().zip(ch).for_each(|(m, v)| *m += v);
        }
        mean.iter_mut().for_each(|m| *m /= c as f32);
        for ch in fr.chunks_exact(hw) {
            var.iter_mut()
                .zip(ch)
                .zip(&mean)
                .for_each(|((s, v), m)| *s += (v - m) * (v - m));
        }
        var.iter_mut()
            .for_each(|s| *s = 1.0 / (*s / c as f32 + eps).sqrt());
        for ((ch, gi), bi) in fr.chunks_exact_mut(hw).zip(g).zip(b) {
            for ((v, m), s) in ch.iter_mut().zip(&mean).zip(&var) {
                *v = (*v - m) * s * gi + bi;
            }
        }
    }
}

struct Res {
    c1: Conv,
    c2: Conv,
    /// `norm3` (gamma, beta) + 1x1x1 `conv_shortcut` when the channel count changes.
    short: Option<(Vec<f32>, Vec<f32>, Conv)>,
}

impl Res {
    fn fwd(&self, x: Vid, causal: bool) -> Vid {
        let mut h = Vid {
            d: x.d.clone(),
            ..x
        };
        let (c, f, hh, ww) = (x.c, x.f, x.h, x.w);
        rms_channels(&mut h, 1e-8);
        nn::silu(&mut h.d);
        let mut h = self.c1.fwd(&h, causal);
        rms_channels(&mut h, 1e-8);
        nn::silu(&mut h.d);
        let mut h = self.c2.fwd(&h, causal);
        let res = match &self.short {
            Some((g, b, conv)) => {
                let mut s = Vid {
                    d: x.d,
                    c,
                    f,
                    h: hh,
                    w: ww,
                };
                layer_norm_channels(&mut s, g, b, 1e-6);
                conv.fwd(&s, causal).d
            }
            None => x.d,
        };
        nn::add_in_place(&mut h.d, &res);
        h
    }
}

/// Where a module sits in the checkpoint: the diffusers path, or the original `up_blocks.N` /
/// `down_blocks.N` one with its running index.
#[derive(Clone, Copy)]
pub struct Names {
    pub diffusers: bool,
}

impl Names {
    fn conv(&self, w: &Weights, base: &str, stride: usize) -> Result<Conv, String> {
        Conv::load(w, &format!("{base}.conv"), stride)
    }

    /// A `LTXVideoResnetBlock3d` at `base` (`conv_in` blocks and `resnets.N`).
    fn res(&self, w: &Weights, base: &str, ci: usize, co: usize) -> Result<Res, String> {
        let short = if ci != co {
            let (norm, conv) = if self.diffusers {
                (
                    format!("{base}.norm3"),
                    format!("{base}.conv_shortcut.conv"),
                )
            } else {
                (
                    format!("{base}.norm3.norm"),
                    format!("{base}.conv_shortcut"),
                )
            };
            Some((
                w.vec_n(&format!("{norm}.weight"), ci)?,
                w.vec_n(&format!("{norm}.bias"), ci)?,
                Conv::load(w, &conv, 1)?,
            ))
        } else {
            None
        };
        Ok(Res {
            c1: self.conv(w, &format!("{base}.conv1"), 1)?,
            c2: self.conv(w, &format!("{base}.conv2"), 1)?,
            short,
        })
    }

    /// `n` resnets of one block: `base` is `decoder.mid_block` / `decoder.up_blocks.I` (diffusers) or
    /// `decoder.up_blocks.IDX` (original).
    fn resnets(&self, w: &Weights, base: &str, n: usize, c: usize) -> Result<Vec<Res>, String> {
        let list = if self.diffusers {
            "resnets"
        } else {
            "res_blocks"
        };
        (0..n)
            .map(|i| self.res(w, &format!("{base}.{list}.{i}"), c, c))
            .collect()
    }
}

/// One decoder stage: optional channel-changing resnet, optional 2x2x2 depth-to-space upsampler, resnets.
struct Up {
    conv_in: Option<Res>,
    up: Option<Conv>,
    res: Vec<Res>,
}

pub struct Decoder {
    conv_in: Conv,
    mid: Vec<Res>,
    ups: Vec<Up>,
    conv_out: Conv,
    patch: usize,
}

/// Widths of the checkpoints this module reads (diffusers `vae/config.json` of Lightricks/LTX-Video).
#[derive(Clone, Copy)]
pub struct Cfg {
    pub ch: [usize; 4],
    pub layers: [usize; 5],
    pub up: [bool; 4],
    pub patch: usize,
    pub latent: usize,
}

const V090: Cfg = Cfg {
    ch: [128, 256, 512, 512],
    layers: [4, 3, 3, 3, 4],
    up: [true, true, true, false],
    patch: 4,
    latent: 128,
};

impl Decoder {
    fn load(w: &Weights, n: Names, cfg: &Cfg) -> Result<Self, String> {
        let ch: Vec<usize> = cfg.ch.iter().rev().copied().collect();
        let layers: Vec<usize> = cfg.layers.iter().rev().copied().collect();
        let scale: Vec<bool> = cfg.up.iter().rev().copied().collect();
        let mut idx = 0; // original `decoder.up_blocks.IDX`: mid block, then conv_in / upsampler / resnets per stage
        let orig = |i: usize| format!("decoder.up_blocks.{i}");
        let mid_base = if n.diffusers {
            "decoder.mid_block".to_string()
        } else {
            orig(0)
        };
        let mid = n.resnets(w, &mid_base, layers[0], ch[0])?;
        let mut ups = Vec::new();
        let mut cin = ch[0];
        for i in 0..ch.len() {
            let cout = ch[i];
            let base = |idx: usize| {
                if n.diffusers {
                    format!("decoder.up_blocks.{i}")
                } else {
                    orig(idx)
                }
            };
            let conv_in = if cin != cout {
                idx += 1;
                let b = if n.diffusers {
                    format!("decoder.up_blocks.{i}.conv_in")
                } else {
                    orig(idx)
                };
                Some(n.res(w, &b, cin, cout)?)
            } else {
                None
            };
            let up = if scale[i] {
                idx += 1;
                let b = if n.diffusers {
                    format!("decoder.up_blocks.{i}.upsamplers.0")
                } else {
                    orig(idx)
                };
                Some(n.conv(w, &format!("{b}.conv"), 1)?)
            } else {
                None
            };
            idx += 1;
            let res = n.resnets(w, &base(idx), layers[i + 1], cout)?;
            ups.push(Up { conv_in, up, res });
            cin = cout;
        }
        Ok(Self {
            conv_in: n.conv(w, "decoder.conv_in", 1)?,
            mid,
            ups,
            conv_out: n.conv(w, "decoder.conv_out", 1)?,
            patch: cfg.patch,
        })
    }

    pub fn fwd(&self, z: Vid) -> Vid {
        crate::hardware::cpu_compute::enter("ltx_vae::decode");
        let causal = false;
        let mut x = self.conv_in.fwd(&z, causal);
        for r in &self.mid {
            x = r.fwd(x, causal);
        }
        for u in &self.ups {
            if let Some(r) = &u.conv_in {
                x = r.fwd(x, causal);
            }
            if let Some(c) = &u.up {
                x = depth_to_space(c.fwd(&x, causal));
            }
            for r in &u.res {
                x = r.fwd(x, causal);
            }
        }
        rms_channels(&mut x, 1e-8);
        nn::silu(&mut x.d);
        self.conv_out.fwd(&x, causal)
    }
}

/// `[f][c*8][h][w]` -> `[2f-1][c][2h][2w]`: channel `(c*2 + a)*4 + b*2 + d` is frame offset `a`, row offset
/// `b`, column offset `d`; the first output frame is dropped.
fn depth_to_space(x: Vid) -> Vid {
    let (c, f, h, w) = (x.c / 8, x.f, x.h, x.w);
    let (oh, ow) = (2 * h, 2 * w);
    let mut d = vec![0.0f32; (2 * f - 1) * c * oh * ow];
    for fo in 1..2 * f {
        let (src, a) = (fo / 2, fo % 2);
        let dst = &mut d[(fo - 1) * c * oh * ow..][..c * oh * ow];
        let fr = x.frame(src);
        for ch in 0..c {
            for b in 0..2 {
                for dd in 0..2 {
                    let plane = &fr[((ch * 2 + a) * 4 + b * 2 + dd) * h * w..][..h * w];
                    for y in 0..h {
                        for xx in 0..w {
                            dst[ch * oh * ow + (2 * y + b) * ow + 2 * xx + dd] = plane[y * w + xx];
                        }
                    }
                }
            }
        }
    }
    Vid {
        d,
        c,
        f: 2 * f - 1,
        h: oh,
        w: ow,
    }
}

/// One encoder stage: resnets, optional stride-2 downsampling conv, optional channel-changing resnet.
struct Down {
    res: Vec<Res>,
    down: Option<Conv>,
    conv_out: Option<Res>,
}

/// Causal encoder (`LTXVideoEncoder3d`): 4x4 patchify, stages, mid block, RMS-norm, SiLU, conv to 129 channels
/// (128 means + one shared log-variance channel; the posterior mode is the means).
struct Encoder {
    conv_in: Conv,
    downs: Vec<Down>,
    mid: Vec<Res>,
    conv_out: Conv,
    latent: usize,
}

impl Encoder {
    fn load(w: &Weights, n: Names, cfg: &Cfg) -> Result<Self, String> {
        let mut idx = 0; // original `encoder.down_blocks.IDX`: resnets, downsampler, conv_out per stage, then the mid block
        let mut cin = cfg.ch[0];
        let mut downs = Vec::new();
        for i in 0..cfg.ch.len() {
            let cout = cfg.ch[(i + 1).min(cfg.ch.len() - 1)];
            let mut next = |diffusers: String| {
                idx += 1;
                if n.diffusers {
                    diffusers
                } else {
                    format!("encoder.down_blocks.{}", idx - 1)
                }
            };
            let res = n.resnets(w, &next(format!("encoder.down_blocks.{i}")), cfg.layers[i], cin)?;
            let down = if cfg.up[i] {
                let b = next(format!("encoder.down_blocks.{i}.downsamplers.0"));
                Some(Conv::load(w, &format!("{b}.conv"), 2)?)
            } else {
                None
            };
            let conv_out = if cin != cout {
                let b = next(format!("encoder.down_blocks.{i}.conv_out"));
                Some(n.res(w, &b, cin, cout)?)
            } else {
                None
            };
            downs.push(Down {
                res,
                down,
                conv_out,
            });
            cin = cout;
        }
        let mid_base = if n.diffusers {
            "encoder.mid_block".to_string()
        } else {
            format!("encoder.down_blocks.{idx}")
        };
        Ok(Self {
            conv_in: n.conv(w, "encoder.conv_in", 1)?,
            downs,
            latent: cfg.latent,
            mid: n.resnets(w, &mid_base, cfg.layers[4], cin)?,
            conv_out: n.conv(w, "encoder.conv_out", 1)?,
        })
    }

    /// `x` is the patchified frame stack; returns the 128 mean channels.
    fn fwd(&self, x: Vid) -> Vid {
        crate::hardware::cpu_compute::enter("ltx_vae::encode");
        let causal = true;
        let mut x = self.conv_in.fwd(&x, causal);
        for dn in &self.downs {
            for r in &dn.res {
                x = r.fwd(x, causal);
            }
            if let Some(c) = &dn.down {
                x = c.fwd(&x, causal);
            }
            if let Some(r) = &dn.conv_out {
                x = r.fwd(x, causal);
            }
        }
        for r in &self.mid {
            x = r.fwd(x, causal);
        }
        rms_channels(&mut x, 1e-8);
        nn::silu(&mut x.d);
        let y = self.conv_out.fwd(&x, causal);
        let hw = y.h * y.w;
        let mut d = Vec::with_capacity(y.f * self.latent * hw);
        for f in 0..y.f {
            d.extend_from_slice(&y.frame(f)[..self.latent * hw]);
        }
        Vid {
            d,
            c: self.latent,
            f: y.f,
            h: y.h,
            w: y.w,
        }
    }
}

// Graph emission (Metal program) over `[c, t, h, w]` values; each `emit` mirrors the `fwd` above.
impl Gb {
    /// `x [c, ...]` -> tokens `[n, c]` (channels last).
    fn tok(&mut self, x: V) -> V {
        let s = self.shape(x).to_vec();
        let flat = self.reshape(x, &[s[0], s[1..].iter().product::<usize>()]);
        self.transpose(flat, &[1, 0])
    }

    /// Tokens `[n, c]` back to `shape` (`[c, ...]`).
    fn untok(&mut self, t: V, shape: &[usize]) -> V {
        let y = self.transpose(t, &[1, 0]);
        self.reshape(y, shape)
    }

    /// [`rms_channels`] over tokens `t [n, c]` (gain-less RmsNorm), result back in `shape`.
    fn rms_c(&mut self, t: V, shape: &[usize], eps: f32) -> V {
        let s = self.shape(t).to_vec();
        let y = self.op("rmsnorm", Op::RmsNorm { eps }, &[t], &s);
        self.untok(y, shape)
    }

    /// [`layer_norm_channels`] over tokens `t [n, c]`, result back in `shape`.
    fn ln_c(&mut self, t: V, shape: &[usize], g: &[f32], bias: &[f32], eps: f32) -> V {
        let y = self.layer_norm(t, g, bias, eps);
        self.untok(y, shape)
    }

    /// `rms_c` of a `[c, ...]` value.
    fn rms_x(&mut self, x: V, eps: f32) -> V {
        let s = self.shape(x).to_vec();
        let t = self.tok(x);
        self.rms_c(t, &s, eps)
    }
}

impl Conv {
    /// The replicate-padded temporal conv as Concat (first/last frame copies) + `Conv3d`.
    fn emit(&self, b: &mut Gb, x: V, causal: bool) -> V {
        let (k, s) = (self.k, self.stride);
        let left = if causal { k - 1 } else { (k - 1) / 2 };
        let right = k - 1 - left;
        let sh = b.shape(x).to_vec();
        let x = if left + right > 0 {
            // No dead or duplicate nodes: CSE + DCE leave stale node ids behind (see `passes/canonical.rs`).
            let first = (left > 0 || sh[1] == 1).then(|| b.slice(x, 1, 0, 1));
            let last = if sh[1] == 1 { first } else { (right > 0).then(|| b.slice(x, 1, sh[1] - 1, sh[1])) };
            let mut parts = vec![first; left];
            parts.push(Some(x));
            parts.extend(vec![last; right]);
            let parts: Vec<V> = parts.into_iter().flatten().collect();
            b.concat(&parts, 1)
        } else {
            x
        };
        let (t, pad) = (b.shape(x)[1], k / 2);
        let out = [
            self.co,
            (t - k) / s + 1,
            (sh[2] + 2 * pad - k) / s + 1,
            (sh[3] + 2 * pad - k) / s + 1,
        ];
        let w = b.wf32("conv3d.w", &[self.co, self.ci, k, k, k], self.full.clone());
        let bias = b.wf32("conv3d.b", &[self.co], self.b.clone());
        b.op(
            "conv3d",
            Op::Conv3d { stride: [s; 3], padding: [0, pad, pad, 0, pad, pad], dilation: [1; 3], groups: 1 },
            &[x, w, bias],
            &out,
        )
    }
}

impl Res {
    fn emit(&self, b: &mut Gb, x: V, causal: bool) -> V {
        let (sh, t) = (b.shape(x).to_vec(), b.tok(x));
        let h = b.rms_c(t, &sh, 1e-8);
        let h = b.silu(h);
        let h = self.c1.emit(b, h, causal);
        let h = b.rms_x(h, 1e-8);
        let h = b.silu(h);
        let h = self.c2.emit(b, h, causal);
        let res = match &self.short {
            Some((g, bias, conv)) => {
                let s = b.ln_c(t, &sh, g, bias, 1e-6);
                conv.emit(b, s, causal)
            }
            None => x,
        };
        b.add(h, res)
    }
}

/// [`depth_to_space`] on `[c*8, f, h, w]` -> `[c, 2f-1, 2h, 2w]`.
fn emit_depth_to_space(b: &mut Gb, x: V) -> V {
    let s = b.shape(x).to_vec();
    let (c, f, h, w) = (s[0] / 8, s[1], s[2], s[3]);
    let r = b.reshape(x, &[c, 2, 2, 2, f, h, w]); // [c, a, b, d, f, y, x]
    let t = b.transpose(r, &[0, 4, 1, 2, 3, 5, 6]); // [c, f, a, b, d, y, x]
    let t = b.transpose(t, &[0, 1, 2, 5, 3, 6, 4]); // [c, f, a, y, b, x, d]
    let m = b.reshape(t, &[c, 2 * f, 2 * h, 2 * w]);
    b.slice(m, 1, 1, 2 * f)
}

impl Decoder {
    fn emit(&self, b: &mut Gb, z: V) -> V {
        let causal = false;
        let mut x = self.conv_in.emit(b, z, causal);
        for r in &self.mid {
            x = r.emit(b, x, causal);
        }
        for u in &self.ups {
            if let Some(r) = &u.conv_in {
                x = r.emit(b, x, causal);
            }
            if let Some(c) = &u.up {
                let y = c.emit(b, x, causal);
                x = emit_depth_to_space(b, y);
            }
            for r in &u.res {
                x = r.emit(b, x, causal);
            }
        }
        x = b.rms_x(x, 1e-8);
        x = b.silu(x);
        self.conv_out.emit(b, x, causal)
    }
}

impl Encoder {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        let causal = true;
        let mut x = self.conv_in.emit(b, x, causal);
        for dn in &self.downs {
            for r in &dn.res {
                x = r.emit(b, x, causal);
            }
            if let Some(c) = &dn.down {
                x = c.emit(b, x, causal);
            }
            if let Some(r) = &dn.conv_out {
                x = r.emit(b, x, causal);
            }
        }
        for r in &self.mid {
            x = r.emit(b, x, causal);
        }
        x = b.rms_x(x, 1e-8);
        x = b.silu(x);
        let y = self.conv_out.emit(b, x, causal);
        b.slice(y, 0, 0, self.latent)
    }
}

impl LtxVae {
    /// Metal decoder over a `[f][c][h*w]` clip (same outputs as `Decoder::fwd`): one program per clip size.
    pub fn decode_gpu(&self, z: Vid) -> Result<Vid, String> {
        let net = self.gpu_dec.get(&[z.f, z.h, z.w], || {
            let mut b = Gb::new();
            let x = b.input("z", &[z.f, z.c, z.h, z.w]);
            let x = b.transpose(x, &[1, 0, 2, 3]);
            let y = self.dec.emit(&mut b, x);
            let y = b.transpose(y, &[1, 0, 2, 3]);
            b.output(y);
            b.compile()
        })?;
        let ups = self.dec.ups.iter().filter(|u| u.up.is_some()).count();
        let (mut f, mut h, mut w) = (z.f, z.h, z.w);
        for _ in 0..ups {
            (f, h, w) = (2 * f - 1, 2 * h, 2 * w);
        }
        let d = net.run(vec![z.d])?.remove(0);
        Ok(Vid { d, c: self.dec.conv_out.co, f, h, w })
    }

    /// Metal encoder over the patchified frame stack (same outputs as `Encoder::fwd`).
    fn encode_gpu(&self, enc: &Encoder, x: Vid) -> Result<Vid, String> {
        let net = self.gpu_enc.get(&[x.f, x.h, x.w], || {
            let mut b = Gb::new();
            let i = b.input("x", &[x.f, x.c, x.h, x.w]);
            let i = b.transpose(i, &[1, 0, 2, 3]);
            let y = enc.emit(&mut b, i);
            let y = b.transpose(y, &[1, 0, 2, 3]);
            b.output(y);
            b.compile()
        })?;
        let downs = enc.downs.iter().filter(|d| d.down.is_some()).count();
        let (mut f, mut h, mut w) = (x.f, x.h, x.w);
        for _ in 0..downs {
            // k = 3, stride 2, causal temporal pad of 2, spatial pad 1.
            (f, h, w) = ((f - 1) / 2 + 1, (h - 1) / 2 + 1, (w - 1) / 2 + 1);
        }
        let d = net.run(vec![x.d])?.remove(0);
        Ok(Vid { d, c: enc.latent, f, h, w })
    }
}

/// The decoder (and, for image-to-video, the encoder), plus the per-channel latent statistics the
/// pipeline normalises with.
pub struct LtxVae {
    pub dec: Decoder,
    enc: Option<Encoder>,
    mean: Vec<f32>,
    std: Vec<f32>,
    cfg: Cfg,
    gpu_dec: ByShape,
    gpu_enc: ByShape,
}

impl LtxVae {
    pub fn load(path: &Path, encoder: bool) -> Result<Self, String> {
        let w = Weights::open(path)?;
        let diffusers = w.has("decoder.mid_block.resnets.0.conv1.conv.weight");
        if !diffusers && !w.has("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight") {
            return Err(format!(
                "{}: not an LTX-Video VAE (no decoder resnets)",
                path.display()
            ));
        }
        // ponytail: v0.9.1+ VAEs (timestep-conditioned decoder, noise injection, other widths) are refused, not approximated.
        if [
            "decoder.time_embedder.timestep_embedder.linear_1.weight",
            "decoder.last_time_embedder.timestep_embedder.linear_1.weight",
        ]
        .iter()
        .any(|n| w.has(n))
            || w.shape("decoder.conv_in.conv.weight")? != [V090.ch[3], V090.latent, 3, 3, 3]
        {
            return Err(format!(
                "{}: only the v0.9.0 LTX-Video VAE (512-wide, no timestep conditioning) is implemented",
                path.display()
            ));
        }
        Self::open(&w, Names { diffusers }, encoder, V090)
    }

    pub fn open(w: &Weights, n: Names, encoder: bool, cfg: Cfg) -> Result<Self, String> {
        let (mean, std) = if n.diffusers {
            ("latents_mean", "latents_std")
        } else {
            (
                "per_channel_statistics.mean-of-means",
                "per_channel_statistics.std-of-means",
            )
        };
        Ok(Self {
            dec: Decoder::load(w, n, &cfg)?,
            enc: encoder.then(|| Encoder::load(w, n, &cfg)).transpose()?,
            mean: w.vec_n(mean, cfg.latent)?,
            std: w.vec_n(std, cfg.latent)?,
            cfg,
            gpu_dec: ByShape::default(),
            gpu_enc: ByShape::default(),
        })
    }

    /// Pipeline latents `tokens` `[f*h*w, 128]` (frame, row, column major; the transformer's output
    /// space) -> `8(f-1)+1` RGB frames of `32h x 32w`, after `latents * std / scaling_factor + mean`.
    pub fn decode_tokens(
        &self,
        tokens: &[f32],
        f: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<ImageBuffer>, String> {
        let lat = self.cfg.latent;
        if tokens.len() != f * h * w * lat {
            return Err(format!(
                "VAE decode: {} latents, expected {f}x{h}x{w}x{lat}",
                tokens.len()
            ));
        }
        let hw = h * w;
        let mut d = vec![0.0f32; tokens.len()];
        for (i, tok) in tokens.chunks_exact(lat).enumerate() {
            let (fr, p) = (i / hw, i % hw);
            for (c, v) in tok.iter().enumerate() {
                d[(fr * lat + c) * hw + p] = v * self.std[c] + self.mean[c]; // scaling_factor is 1.0
            }
        }
        let vid = Vid {
            d,
            c: lat,
            f,
            h,
            w,
        };
        let out = if crate::model::diff_graph::gpu_mode() {
            self.decode_gpu(vid)?
        } else {
            self.dec.fwd(vid)
        };
        let p = self.dec.patch;
        let (oh, ow) = (out.h * p, out.w * p);
        Ok((0..out.f)
            .map(|fr| {
                let src = out.frame(fr);
                let mut img = ImageBuffer::new(ow, oh);
                for y in 0..oh {
                    for x in 0..ow {
                        for ch in 0..3 {
                            // channel = c * p^2 + (x % p) * p + (y % p); see `LTXVideoDecoder3d.forward`.
                            let c = (ch * p + x % p) * p + y % p;
                            let v = src[c * out.h * out.w + (y / p) * out.w + x / p];
                            img.data[(y * ow + x) * 3 + ch] =
                                ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
                img
            })
            .collect())
    }

    /// One RGB image (`32h x 32w`) -> its normalised latent tokens `[h*w, 128]` (`(mode - mean) / std`), the first
    /// latent frame of an image-to-video clip.
    // ponytail: the posterior mode, not LTXImageToVideoPipeline's default `sample` (mean + exp(logvar / 2) * noise);
    // the logvar channel was not inspected, so this is deterministic but not proven identical to the reference.
    pub fn encode_image(&self, img: &ImageBuffer) -> Result<Vec<f32>, String> {
        let enc = self.enc.as_ref().ok_or("VAE loaded without its encoder")?;
        let p = self.cfg.patch;
        if img.width % (8 * p) != 0
            || img.height % (8 * p) != 0
            || img.data.len() != img.width * img.height * 3
        {
            return Err(format!(
                "VAE encode: {}x{} RGB image is not a multiple of {}",
                img.width,
                img.height,
                8 * p
            ));
        }
        let (h, w) = (img.height / p, img.width / p);
        // Patchify: channel (c * p + x % p) * p + y % p of pixel (y / p, x / p), values in [-1, 1].
        let mut d = vec![0.0f32; 3 * p * p * h * w];
        for y in 0..img.height {
            for x in 0..img.width {
                for c in 0..3 {
                    let ch = (c * p + x % p) * p + y % p;
                    d[ch * h * w + (y / p) * w + x / p] =
                        img.data[(y * img.width + x) * 3 + c] as f32 / 127.5 - 1.0;
                }
            }
        }
        let xin = Vid {
            d,
            c: 3 * p * p,
            f: 1,
            h,
            w,
        };
        let z = if crate::model::diff_graph::gpu_mode() {
            self.encode_gpu(enc, xin)?
        } else {
            enc.fwd(xin)
        };
        let hw = z.h * z.w;
        let lat = self.cfg.latent;
        let mut tokens = vec![0.0f32; hw * lat];
        for c in 0..lat {
            for q in 0..hw {
                tokens[q * lat + c] = (z.d[c * hw + q] - self.mean[c]) / self.std[c];
            }
        }
        Ok(tokens)
    }
}

/// Tiny synthetic v0.9.0-layout net for the CPU-free fixtures and parity tests (shared with
/// `tests/cpu_free_matrix.rs`).
pub mod fixture {
    use super::*;
    use crate::model::diff_graph::testutil::Lcg;

    /// Tiny v0.9.0-layout checkpoint (diffusers names): every module kind of the real net, small widths.
    fn tiny() -> (Cfg, Weights) {
        use crate::model::diff_graph::testutil::synth;
        let cfg = Cfg { ch: [4, 8, 16, 16], layers: [2, 1, 1, 1, 2], up: [true, true, true, false], patch: 2, latent: 8 };
        let mut s: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |s: &mut Vec<_>, p: &str, ci: usize, co: usize, k: usize| {
            s.push((format!("{p}.conv.weight"), vec![co, ci, k, k, k]));
            s.push((format!("{p}.conv.bias"), vec![co]));
        };
        let res = |s: &mut Vec<_>, p: &str, ci: usize, co: usize| {
            conv(s, &format!("{p}.conv1"), ci, co, 3);
            conv(s, &format!("{p}.conv2"), co, co, 3);
            if ci != co {
                s.push((format!("{p}.norm3.weight"), vec![ci]));
                s.push((format!("{p}.norm3.bias"), vec![ci]));
                conv(s, &format!("{p}.conv_shortcut"), ci, co, 1);
            }
        };
        let rev: Vec<usize> = cfg.ch.iter().rev().copied().collect();
        let l: Vec<usize> = cfg.layers.iter().rev().copied().collect();
        conv(&mut s, "decoder.conv_in", cfg.latent, rev[0], 3);
        for j in 0..l[0] {
            res(&mut s, &format!("decoder.mid_block.resnets.{j}"), rev[0], rev[0]);
        }
        let mut cin = rev[0];
        for i in 0..4 {
            let cout = rev[i];
            let b = format!("decoder.up_blocks.{i}");
            if cin != cout {
                res(&mut s, &format!("{b}.conv_in"), cin, cout);
            }
            if cfg.up[3 - i] {
                conv(&mut s, &format!("{b}.upsamplers.0.conv"), cout, cout * 8, 3);
            }
            for j in 0..l[i + 1] {
                res(&mut s, &format!("{b}.resnets.{j}"), cout, cout);
            }
            cin = cout;
        }
        conv(&mut s, "decoder.conv_out", cfg.ch[0], 3 * cfg.patch * cfg.patch, 3);
        conv(&mut s, "encoder.conv_in", 3 * cfg.patch * cfg.patch, cfg.ch[0], 3);
        let mut cin = cfg.ch[0];
        for i in 0..4 {
            let cout = cfg.ch[(i + 1).min(3)];
            let b = format!("encoder.down_blocks.{i}");
            for j in 0..cfg.layers[i] {
                res(&mut s, &format!("{b}.resnets.{j}"), cin, cin);
            }
            if cfg.up[i] {
                // `Encoder::load` reads the downsampler as `{b}.conv.weight` (one `.conv` less than `Names::conv`).
                s.push((format!("{b}.downsamplers.0.conv.weight"), vec![cin, cin, 3, 3, 3]));
                s.push((format!("{b}.downsamplers.0.conv.bias"), vec![cin]));
            }
            if cin != cout {
                res(&mut s, &format!("{b}.conv_out"), cin, cout);
            }
            cin = cout;
        }
        for j in 0..cfg.layers[4] {
            res(&mut s, &format!("encoder.mid_block.resnets.{j}"), cin, cin);
        }
        conv(&mut s, "encoder.conv_out", cin, cfg.latent + 1, 3);
        s.push(("latents_mean".into(), vec![cfg.latent]));
        s.push(("latents_std".into(), vec![cfg.latent]));
        let r = synth(&s, 31);
        (cfg, Weights::from_f32(s, |n| Ok(r.0[n].1.clone())).unwrap())
    }

    /// The tiny decoder + encoder.
    pub fn tiny_vae() -> LtxVae {
        let (cfg, w) = tiny();
        LtxVae::open(&w, Names { diffusers: true }, true, cfg).unwrap()
    }

    /// Public `encode_image` / `decode_tokens` of the tiny net (Metal when the CPU tripwire is armed), with
    /// finiteness checks of the encoder tokens and the raw Metal decoder output.
    pub fn tiny_on_gpu() -> Result<(), String> {
        let vae = tiny_vae();
        let (p, lat) = (vae.cfg.patch, vae.cfg.latent);
        // Three stride-2 stages: a 16 x 32 image is one 1 x 2 latent frame.
        let (ih, iw) = (8 * p, 16 * p);
        let mut img = ImageBuffer::new(iw, ih);
        let mut g = Lcg(4);
        img.data.iter_mut().for_each(|v| *v = ((g.next() + 1.0) * 127.5) as u8);
        let tokens = vae.encode_image(&img)?;
        if tokens.len() != 2 * lat || tokens.iter().any(|v| !v.is_finite()) {
            return Err(format!("ltx_vae: bad encoder tokens ({} values)", tokens.len()));
        }
        let (f, h, w) = (2, 2, 3);
        let frames = vae.decode_tokens(&Lcg(1).vec(f * h * w * lat, 1.0), f, h, w)?;
        if frames.len() != 9 || frames[0].width != 8 * p * w || frames[0].height != 8 * p * h {
            return Err(format!("ltx_vae: {} frames", frames.len()));
        }
        let z = Vid { d: Lcg(2).vec(f * lat * h * w, 1.0), c: lat, f, h, w };
        if vae.decode_gpu(z)?.d.iter().any(|v| !v.is_finite()) {
            return Err("ltx_vae: non-finite Metal output".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect()
        }
    }

    /// Direct 3D conv on the explicitly padded frame list, zero padding in space.
    fn naive(x: &Vid, w: &[f32], b: &[f32], co: usize, k: usize, s: usize, causal: bool) -> Vid {
        let first = x.frame(0).to_vec();
        let last = x.frame(x.f - 1).to_vec();
        let (l, r) = match (k, causal) {
            (1, _) => (0, 0),
            (_, true) => (k - 1, 0),
            (_, false) => ((k - 1) / 2, (k - 1) / 2),
        };
        let mut frames: Vec<Vec<f32>> = vec![first; l];
        frames.extend((0..x.f).map(|f| x.frame(f).to_vec()));
        frames.extend(vec![last; r]);
        let p = k / 2;
        let (oh, ow) = ((x.h + 2 * p - k) / s + 1, (x.w + 2 * p - k) / s + 1);
        let fo = (frames.len() - k) / s + 1;
        let mut d = vec![0.0f32; fo * co * oh * ow];
        for t in 0..fo {
            for o in 0..co {
                for y in 0..oh {
                    for xx in 0..ow {
                        let mut acc = b[o];
                        for i in 0..x.c {
                            for kt in 0..k {
                                for ky in 0..k {
                                    for kx in 0..k {
                                        let (yy, xp) = (
                                            (y * s + ky) as isize - p as isize,
                                            (xx * s + kx) as isize - p as isize,
                                        );
                                        if yy < 0
                                            || xp < 0
                                            || yy >= x.h as isize
                                            || xp >= x.w as isize
                                        {
                                            continue;
                                        }
                                        acc += w[(((o * x.c + i) * k + kt) * k + ky) * k + kx]
                                            * frames[t * s + kt]
                                                [i * x.h * x.w + yy as usize * x.w + xp as usize];
                                    }
                                }
                            }
                        }
                        d[((t * co + o) * oh + y) * ow + xx] = acc;
                    }
                }
            }
        }
        Vid {
            d,
            c: co,
            f: fo,
            h: oh,
            w: ow,
        }
    }

    #[test]
    fn temporal_taps_match_a_direct_3d_conv() {
        let mut g = Lcg(5);
        for (k, s, causal) in [(3, 1, false), (3, 1, true), (3, 2, true), (1, 1, false)] {
            let (ci, co) = (3, 4);
            let x = Vid {
                d: g.vec(5 * ci * 6 * 8),
                c: ci,
                f: 5,
                h: 6,
                w: 8,
            };
            let (w, b) = (g.vec(co * ci * k * k * k), g.vec(co));
            let got = Conv::new(&w, b.clone(), [co, ci, k], s).fwd(&x, causal);
            let want = naive(&x, &w, &b, co, k, s, causal);
            assert_eq!(
                (got.f, got.h, got.w),
                (want.f, want.h, want.w),
                "k{k} s{s} causal {causal}"
            );
            let err = got
                .d
                .iter()
                .zip(&want.d)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err < 1e-4, "k{k} s{s} causal {causal}: max error {err}");
        }
    }

    #[test]
    fn depth_to_space_places_frame_row_column_offsets() {
        // One channel group, 2 latent frames of 1x1: channel (a * 2 + b) * 2 + d holds value 10 * f + that index.
        let d: Vec<f32> = (0..2)
            .flat_map(|f| (0..8).map(move |c| (10 * f + c) as f32))
            .collect();
        let out = depth_to_space(Vid {
            d,
            c: 8,
            f: 2,
            h: 1,
            w: 1,
        });
        assert_eq!((out.c, out.f, out.h, out.w), (1, 3, 2, 2));
        // Output frame 0 is source frame 0 offset a = 1 (the a = 0 frame is dropped); rows/cols follow b, d.
        assert_eq!(out.frame(0), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(out.frame(1), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(out.frame(2), &[14.0, 15.0, 16.0, 17.0]);
    }

    /// A procedural 512x320 picture through the real v0.9.0 VAE: encode, decode, compare. The 32x32x8
    /// compression keeps a smooth scene's low frequencies, so the PSNR is a check of both halves.
    #[test]
    fn real_vae_roundtrip_when_cached() {
        let hub = crate::model::hub::HuggingFaceHub::new();
        let path = hub
            .cache_root
            .join("models--city96--LTX-Video-gguf/snapshots/main/LTX-Video-VAE-BF16.safetensors");
        if !path.is_file() {
            println!("SKIP: city96/LTX-Video-gguf LTX-Video-VAE-BF16.safetensors is not cached");
            return;
        }
        let (w, h) = (512, 320);
        let mut img = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sky = 255.0 * (1.0 - y as f32 / h as f32);
                let (dx, dy) = (x as f32 - 200.0, y as f32 - 120.0);
                let disc = dx * dx + dy * dy < 60.0 * 60.0;
                let hill = y as f32 > 200.0 + 25.0 * (x as f32 / 70.0).sin();
                let px = if disc {
                    (230, 40, 40)
                } else if hill {
                    (40, 150 + ((x / 8 + y / 8) % 2) as u8 * 40, 50)
                } else {
                    ((sky * 0.5) as u8, (sky * 0.7) as u8, 255)
                };
                img.set_pixel(x, y, px.0, px.1, px.2);
            }
        }
        let vae = LtxVae::load(&path, true).unwrap();
        let tokens = vae.encode_image(&img).unwrap();
        assert_eq!(tokens.len(), (h / 32) * (w / 32) * V090.latent);
        let back = vae.decode_tokens(&tokens, 1, h / 32, w / 32).unwrap();
        assert_eq!(back.len(), 1);
        let mse = img
            .data
            .iter()
            .zip(&back[0].data)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / img.data.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
        println!("VAE encode -> decode PSNR {psnr:.1} dB");
        assert!(psnr > 20.0, "PSNR {psnr}");
    }

    /// CPU vs Metal graph programs of the decoder (3 temporal 2x upsamplers) and the causal
    /// encoder (3 stride-2 downsamplers) on a tiny synthetic net (f32 vs f32: 1e-3).
    #[test]
    fn gpu_matches_cpu() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        use crate::model::diff_graph::testutil::{rel_err, Lcg};
        let vae = fixture::tiny_vae();
        let cfg = vae.cfg;
        let mk = |n: usize, c: usize, f: usize, h: usize, wd: usize, seed: u64| Vid {
            d: Lcg(seed).vec(n, 1.0),
            c,
            f,
            h,
            w: wd,
        };
        let (f, h, wd) = (2, 2, 3);
        let n = f * cfg.latent * h * wd;
        let cpu = vae.dec.fwd(mk(n, cfg.latent, f, h, wd, 1));
        let gpu = vae.decode_gpu(mk(n, cfg.latent, f, h, wd, 1)).unwrap();
        assert_eq!((cpu.c, cpu.f, cpu.h, cpu.w), (gpu.c, gpu.f, gpu.h, gpu.w));
        let e = rel_err(&gpu.d, &cpu.d);
        println!("ltx_vae decode rel err {e:e}");
        assert!(e < 1e-3, "decode rel err {e}");
        let net = vae.gpu_dec.get(&[f, h, wd], || Err("cached".into())).unwrap();
        assert_eq!(net.cpu_node_count(), 0);

        let enc = vae.enc.as_ref().unwrap();
        let (c, f, h, wd) = (3 * cfg.patch * cfg.patch, 3, 8, 16);
        let n = f * c * h * wd;
        let cpu = enc.fwd(mk(n, c, f, h, wd, 2));
        let gpu = vae.encode_gpu(enc, mk(n, c, f, h, wd, 2)).unwrap();
        assert_eq!((cpu.c, cpu.f, cpu.h, cpu.w), (gpu.c, gpu.f, gpu.h, gpu.w));
        let e = rel_err(&gpu.d, &cpu.d);
        println!("ltx_vae encode rel err {e:e}");
        assert!(e < 1e-3, "encode rel err {e}");
        let net = vae.gpu_enc.get(&[f, h, wd], || Err("cached".into())).unwrap();
        assert_eq!(net.cpu_node_count(), 0);
    }
}
