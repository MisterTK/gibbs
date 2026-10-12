//! Deep Compression Autoencoder decoder (diffusers `AutoencoderDC`, Sana's `dc-ae-f32c32`): 32x
//! spatial compression, ResBlocks at the high resolutions and EfficientViT blocks (multi-scale ReLU
//! linear attention + GLU MBConv) at the low ones, channel-wise RMSNorm with bias, upsampling by
//! nearest-interpolate + conv with a repeat/pixel-shuffle shortcut. Activations are `[C, H, W]` f32.

use crate::model::dit::read_config;
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::{par_for, OutPtr};
use crate::model::weights::Weights;
use std::path::Path;

const EPS: f32 = 1e-5;

fn conv(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    wt: &(Vec<f32>, Option<Vec<f32>>, usize, usize),
) -> Vec<f32> {
    let (wv, b, c_out, k) = wt;
    debug_assert_eq!(wv.len(), c_out * c * k * k);
    nn::conv2d(x, c, h, w, wv, b.as_deref(), *c_out, *k, 1, k / 2).0
}

/// (weight, bias, c_out, k) of a square conv `{p}`.
fn load_conv(w: &Weights, p: &str) -> Result<(Vec<f32>, Option<Vec<f32>>, usize, usize), String> {
    let name = format!("{p}.weight");
    let s = w.shape(&name)?.to_vec();
    if s.len() != 4 || s[2] != s[3] {
        return Err(format!(
            "DC-AE: {name} has shape {s:?}, expected a square conv"
        ));
    }
    let b = format!("{p}.bias");
    Ok((
        w.vec(&name)?,
        if w.has(&b) { Some(w.vec(&b)?) } else { None },
        s[0],
        s[2],
    ))
}

/// Channel-wise RMSNorm with weight and bias over x [C, HW].
fn rms_norm_ch(x: &mut [f32], c: usize, hw: usize, g: &[f32], b: &[f32]) {
    let mut ss = vec![0.0f32; hw];
    for ch in x.chunks_exact(hw) {
        ss.iter_mut().zip(ch).for_each(|(s, v)| *s += v * v);
    }
    let inv: Vec<f32> = ss
        .iter()
        .map(|s| 1.0 / (s / c as f32 + EPS).sqrt())
        .collect();
    for (ci, ch) in x.chunks_exact_mut(hw).enumerate() {
        for (v, i) in ch.iter_mut().zip(&inv) {
            *v = *v * i * g[ci] + b[ci];
        }
    }
}

/// Depthwise k x k conv (zero pad k/2) over x [C, H, W]; weight [C, 1, k, k].
fn depthwise(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    wt: &[f32],
    bias: Option<&[f32]>,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    let o = OutPtr::new(&mut out);
    let (hw, r) = (h * w, (k / 2) as isize);
    par_for(c, &|ci| {
        // SAFETY: channel ci belongs to task ci.
        let dst = unsafe { o.slice(ci * hw, hw) };
        let src = &x[ci * hw..][..hw];
        let kw = &wt[ci * k * k..][..k * k];
        for y in 0..h as isize {
            for xx in 0..w as isize {
                let mut s = bias.map_or(0.0, |b| b[ci]);
                for ky in 0..k as isize {
                    let iy = y + ky - r;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..k as isize {
                        let ix = xx + kx - r;
                        if ix >= 0 && ix < w as isize {
                            s += kw[(ky * k as isize + kx) as usize]
                                * src[(iy * w as isize + ix) as usize];
                        }
                    }
                }
                dst[(y * w as isize + xx) as usize] = s;
            }
        }
    });
    out
}

/// Per-pixel linear map of x [C_in, HW] by weight [C_out, C_in] (1x1 conv, no bias).
fn pointwise(x: &[f32], c_in: usize, hw: usize, wt: &[f32], c_out: usize) -> Vec<f32> {
    let t = nn::transpose(x, c_in, hw);
    nn::transpose(&nn::linear(&t, hw, wt, None, c_in, c_out), hw, c_out)
}

/// [C*4, H, W] -> [C, 2H, 2W].
fn pixel_shuffle(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ci in 0..c {
        for i in 0..2 {
            for j in 0..2 {
                let src = &x[((ci * 4 + i * 2 + j) * h) * w..][..h * w];
                for y in 0..h {
                    for xx in 0..w {
                        out[(ci * 2 * h + 2 * y + i) * 2 * w + 2 * xx + j] = src[y * w + xx];
                    }
                }
            }
        }
    }
    out
}

/// x [C, ...] with every channel repeated `r` times consecutively.
fn repeat_channels(x: &[f32], c: usize, r: usize) -> Vec<f32> {
    let hw = x.len() / c;
    x.chunks_exact(hw)
        .flat_map(|ch| std::iter::repeat(ch).take(r).flatten().copied())
        .collect()
}

struct Glu {
    inverted: (Vec<f32>, Option<Vec<f32>>, usize, usize),
    depth_w: Vec<f32>,
    depth_b: Vec<f32>,
    point: Vec<f32>,
    norm: (Vec<f32>, Vec<f32>),
}

struct EVit {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    inner: usize,
    ms_in: Vec<f32>,
    ms_k: usize,
    ms_out: Vec<f32>,
    out: Vec<f32>,
    norm: (Vec<f32>, Vec<f32>),
    glu: Glu,
}

enum Block {
    Res {
        conv1: (Vec<f32>, Option<Vec<f32>>, usize, usize),
        conv2: (Vec<f32>, Option<Vec<f32>>, usize, usize),
        norm: (Vec<f32>, Vec<f32>),
    },
    EVit(Box<EVit>),
    Up {
        conv: (Vec<f32>, Option<Vec<f32>>, usize, usize),
        c_out: usize,
    },
}

pub struct DcAeDecoder {
    latent: usize,
    conv_in: (Vec<f32>, Option<Vec<f32>>, usize, usize),
    /// Applied in order (diffusers runs `reversed(up_blocks)`).
    blocks: Vec<(usize, Block)>,
    norm_out: (Vec<f32>, Vec<f32>),
    conv_out: (Vec<f32>, Option<Vec<f32>>, usize, usize),
    head_dim: usize,
    pub scaling_factor: f32,
}

impl DcAeDecoder {
    /// `<snapshot>/vae` of a diffusers `AutoencoderDC`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        crate::graph::avenues::gate(crate::graph::avenues::Family::Vae)?;
        let cfg = read_config(dir)?;
        let usizes = |k: &str| -> Vec<usize> {
            cfg[k]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_default()
        };
        let chans = usizes("decoder_block_out_channels");
        let layers = usizes("decoder_layers_per_block");
        let types: Vec<String> = cfg["decoder_block_types"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if chans.is_empty()
            || chans.len() != layers.len()
            || chans.len() != types.len()
            || layers.contains(&0)
        {
            return Err(
                "DC-AE: decoder block config is missing or has empty levels (unsupported)".into(),
            );
        }
        if cfg["upsample_block_type"] != "interpolate"
            || cfg["decoder_norm_types"] != "rms_norm"
            || cfg["decoder_act_fns"] != "silu"
        {
            return Err(format!(
                "DC-AE: only interpolate upsampling, rms_norm and silu decoders are supported (config: {}, {}, {})",
                cfg["upsample_block_type"], cfg["decoder_norm_types"], cfg["decoder_act_fns"]
            ));
        }
        let w = crate::model::dit::component_weights(dir)?;
        let norm = |p: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((w.vec(&format!("{p}.weight"))?, w.vec(&format!("{p}.bias"))?))
        };
        let mut blocks = Vec::new();
        for i in (0..chans.len()).rev() {
            let c = chans[i];
            let mut j = 0;
            if i + 1 < chans.len() {
                blocks.push((
                    chans[i + 1],
                    Block::Up {
                        conv: load_conv(&w, &format!("decoder.up_blocks.{i}.0.conv"))?,
                        c_out: c,
                    },
                ));
                j = 1;
            }
            for l in 0..layers[i] {
                let p = format!("decoder.up_blocks.{i}.{}", j + l);
                let b = match types[i].as_str() {
                    "ResBlock" => Block::Res {
                        conv1: load_conv(&w, &format!("{p}.conv1"))?,
                        conv2: load_conv(&w, &format!("{p}.conv2"))?,
                        norm: norm(&format!("{p}.norm"))?,
                    },
                    "EfficientViTBlock" => {
                        let a = format!("{p}.attn");
                        let ms = format!("{a}.to_qkv_multiscale.0");
                        if w.has(&format!("{a}.to_qkv_multiscale.1.proj_in.weight")) {
                            return Err(
                                "DC-AE: more than one multiscale kernel is not supported".into()
                            );
                        }
                        let q = w.vec(&format!("{a}.to_q.weight"))?;
                        let inner = q.len() / c;
                        let ms_k = w.shape(&format!("{ms}.proj_in.weight"))?[2];
                        let g = format!("{p}.conv_out");
                        Block::EVit(Box::new(EVit {
                            q,
                            k: w.vec(&format!("{a}.to_k.weight"))?,
                            v: w.vec(&format!("{a}.to_v.weight"))?,
                            inner,
                            ms_in: w.vec(&format!("{ms}.proj_in.weight"))?,
                            ms_k,
                            ms_out: w.vec(&format!("{ms}.proj_out.weight"))?,
                            out: w.vec(&format!("{a}.to_out.weight"))?,
                            norm: norm(&format!("{a}.norm_out"))?,
                            glu: Glu {
                                inverted: load_conv(&w, &format!("{g}.conv_inverted"))?,
                                depth_w: w.vec(&format!("{g}.conv_depth.weight"))?,
                                depth_b: w.vec(&format!("{g}.conv_depth.bias"))?,
                                point: w.vec(&format!("{g}.conv_point.weight"))?,
                                norm: norm(&format!("{g}.norm"))?,
                            },
                        }))
                    }
                    t => return Err(format!("DC-AE: decoder block type {t} is not supported")),
                };
                blocks.push((c, b));
            }
        }
        Ok(Self {
            latent: cfg["latent_channels"]
                .as_u64()
                .ok_or("DC-AE config: missing latent_channels")? as usize,
            conv_in: load_conv(&w, "decoder.conv_in")?,
            blocks,
            norm_out: norm("decoder.norm_out")?,
            conv_out: load_conv(&w, "decoder.conv_out")?,
            head_dim: cfg["attention_head_dim"].as_u64().unwrap_or(32) as usize,
            scaling_factor: cfg["scaling_factor"]
                .as_f64()
                .ok_or("DC-AE config: missing scaling_factor")? as f32,
        })
    }

    pub fn latent_channels(&self) -> usize {
        self.latent
    }

    /// Spatial upsampling factor (32 for f32c32).
    pub fn factor(&self) -> usize {
        1 << self
            .blocks
            .iter()
            .filter(|(_, b)| matches!(b, Block::Up { .. }))
            .count()
    }

    fn evit(&self, e: &EVit, x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
        let hw = h * w;
        let (inner, hd) = (e.inner, self.head_dim);
        // [q | k | v] channels, then the depthwise + grouped-pointwise multi-scale branch.
        let qkv = [
            pointwise(x, c, hw, &e.q, inner),
            pointwise(x, c, hw, &e.k, inner),
            pointwise(x, c, hw, &e.v, inner),
        ]
        .concat();
        let ms = depthwise(&qkv, 3 * inner, h, w, &e.ms_in, None, e.ms_k);
        let groups = 3 * inner / hd;
        let mut ms2 = vec![0.0f32; ms.len()];
        for g in 0..groups {
            let src = &ms[g * hd * hw..][..hd * hw];
            let wt = &e.ms_out[g * hd * hd..][..hd * hd];
            ms2[g * hd * hw..][..hd * hw].copy_from_slice(&pointwise(src, hd, hw, wt, hd));
        }
        let all = [qkv, ms2].concat();
        // Chunks of 3*hd channels: (q, k, v) of one head; ReLU linear attention per head.
        let heads = all.len() / (3 * hd * hw);
        let mut att = vec![0.0f32; heads * hd * hw];
        let o = OutPtr::new(&mut att);
        par_for(heads, &|hi| {
            let base = &all[hi * 3 * hd * hw..][..3 * hd * hw];
            let relu = |s: &[f32]| s.iter().map(|v| v.max(0.0)).collect::<Vec<f32>>();
            let (q, k, v) = (
                relu(&base[..hd * hw]),
                relu(&base[hd * hw..2 * hd * hw]),
                &base[2 * hd * hw..],
            );
            // kv[c][d] = sum_j v[c][j] k[d][j]; ksum[d] = sum_j k[d][j].
            let mut kv = vec![0.0f64; hd * hd];
            let mut ksum = vec![0.0f64; hd];
            for d in 0..hd {
                let kd = &k[d * hw..][..hw];
                ksum[d] = kd.iter().map(|&v| v as f64).sum();
                for cc in 0..hd {
                    kv[cc * hd + d] = v[cc * hw..][..hw]
                        .iter()
                        .zip(kd)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                }
            }
            // SAFETY: head hi's channels belong to task hi.
            let dst = unsafe { o.slice(hi * hd * hw, hd * hw) };
            for i in 0..hw {
                let den: f64 = (0..hd).map(|d| ksum[d] * q[d * hw + i] as f64).sum::<f64>() + 1e-15;
                for cc in 0..hd {
                    let num: f64 = (0..hd)
                        .map(|d| kv[cc * hd + d] * q[d * hw + i] as f64)
                        .sum();
                    dst[cc * hw + i] = (num / den) as f32;
                }
            }
        });
        let mut y = pointwise(&att, heads * hd, hw, &e.out, c);
        rms_norm_ch(&mut y, c, hw, &e.norm.0, &e.norm.1);
        nn::add_in_place(&mut y, x);
        // GLU MBConv with residual.
        let g = &e.glu;
        let mut t = conv(&y, c, h, w, &g.inverted);
        nn::silu(&mut t);
        let c2 = g.inverted.2;
        let t = depthwise(&t, c2, h, w, &g.depth_w, Some(&g.depth_b), 3);
        let (a, gate) = t.split_at(c2 / 2 * hw);
        let mut gate = gate.to_vec();
        nn::silu(&mut gate);
        let m: Vec<f32> = a.iter().zip(&gate).map(|(a, g)| a * g).collect();
        let mut z = pointwise(&m, c2 / 2, hw, &g.point, c);
        rms_norm_ch(&mut z, c, hw, &g.norm.0, &g.norm.1);
        nn::add_in_place(&mut z, &y);
        z
    }

    /// Model-scale latents [C, h, w] (divided by `scaling_factor` here) -> RGB image of factor*h x factor*w.
    pub fn decode(&self, latents: &[f32], h: usize, w: usize) -> Result<ImageBuffer, String> {
        if latents.len() != self.latent * h * w {
            return Err(format!(
                "DC-AE decode: latents len {} != {}x{h}x{w}",
                latents.len(),
                self.latent
            ));
        }
        let z: Vec<f32> = latents.iter().map(|v| v / self.scaling_factor).collect();
        let c0 = self.conv_in.2;
        let mut x = conv(&z, self.latent, h, w, &self.conv_in);
        nn::add_in_place(&mut x, &repeat_channels(&z, self.latent, c0 / self.latent));
        let (mut c, mut h, mut w) = (c0, h, w);
        for (ch, b) in &self.blocks {
            debug_assert_eq!(*ch, c);
            match b {
                Block::Up { conv: cv, c_out } => {
                    let up = nn::upsample_nearest2x(&x, c, h, w);
                    let mut y = conv(&up, c, 2 * h, 2 * w, cv);
                    let sc = pixel_shuffle(&repeat_channels(&x, c, c_out * 4 / c), *c_out, h, w);
                    nn::add_in_place(&mut y, &sc);
                    x = y;
                    (c, h, w) = (*c_out, 2 * h, 2 * w);
                }
                Block::Res { conv1, conv2, norm } => {
                    let mut t = conv(&x, c, h, w, conv1);
                    nn::silu(&mut t);
                    let mut t = conv(&t, c, h, w, conv2);
                    rms_norm_ch(&mut t, c, h * w, &norm.0, &norm.1);
                    nn::add_in_place(&mut t, &x);
                    x = t;
                }
                Block::EVit(e) => x = self.evit(e, &x, c, h, w),
            }
        }
        rms_norm_ch(&mut x, c, h * w, &self.norm_out.0, &self.norm_out.1);
        x.iter_mut().for_each(|v| *v = v.max(0.0));
        let y = conv(&x, c, h, w, &self.conv_out);
        let mut img = ImageBuffer::new(w, h);
        let hw = h * w;
        let px = |v: f32| (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8;
        for i in 0..hw {
            img.set_pixel(i % w, i / w, px(y[i]), px(y[hw + i]), px(y[2 * hw + i]));
        }
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_shuffle_interleaves_subpixels() {
        // 4 channels of 1x1 -> one 2x2 channel in (i, j) order.
        assert_eq!(
            pixel_shuffle(&[1.0, 2.0, 3.0, 4.0], 1, 1, 1),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        // 4 channels of 1x2: channel (i*2+j) fills rows 2y+i, cols 2x+j.
        let x = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        assert_eq!(
            pixel_shuffle(&x, 1, 1, 2),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn depthwise_matches_dense_conv_on_one_channel() {
        let x: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let wt: Vec<f32> = (0..9).map(|i| (i as f32 - 4.0) * 0.3).collect();
        let want = nn::conv2d(&x, 1, 4, 5, &wt, Some(&[0.5]), 1, 3, 1, 1).0;
        let got = depthwise(&x, 1, 4, 5, &wt, Some(&[0.5]), 3);
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5));
    }
}
