//! Deep Compression Autoencoder decoder (diffusers `AutoencoderDC`, Sana's `dc-ae-f32c32`): 32x
//! spatial compression, ResBlocks at the high resolutions and EfficientViT blocks (multi-scale ReLU
//! linear attention + GLU MBConv) at the low ones, channel-wise RMSNorm with bias, upsampling by
//! nearest-interpolate + conv with a repeat/pixel-shuffle shortcut. Activations are `[C, H, W]` f32.

use crate::graph::Op;
use crate::model::diff_graph::{ByShape, Gb, V};
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

// Graph emission (Metal program) over `[c, h, w]` values; each helper mirrors a CPU function above.
type ConvW = (Vec<f32>, Option<Vec<f32>>, usize, usize);

fn emit_conv(b: &mut Gb, x: V, (w, bias, c_out, k): &ConvW) -> V {
    b.conv2d(x, w, bias.as_deref(), *c_out, *k, 1, k / 2)
}

/// [`depthwise`]: one group per channel, weight `[c, 1, k, k]`.
fn emit_depthwise(b: &mut Gb, x: V, w: &[f32], bias: Option<&[f32]>, k: usize) -> V {
    let s = b.shape(x).to_vec();
    let wv = b.wf32("dw.w", &[s[0], 1, k, k], w.to_vec());
    let mut ins = vec![x, wv];
    if let Some(bi) = bias {
        ins.push(b.wf32("dw.b", &[s[0]], bi.to_vec()));
    }
    let pad = k / 2;
    b.op("dwconv", Op::Conv2d { stride: [1; 2], padding: [pad; 4], dilation: [1; 2], groups: s[0] }, &ins, &s)
}

/// [`rms_norm_ch`]: RmsNorm with gain over channels-last tokens, then the bias.
fn emit_norm(b: &mut Gb, x: V, (g, bias): &(Vec<f32>, Vec<f32>)) -> V {
    let s = b.shape(x).to_vec();
    let (c, hw) = (s[0], s[1] * s[2]);
    let flat = b.reshape(x, &[c, hw]);
    let t = b.transpose(flat, &[1, 0]);
    let gv = b.wf32("rms.g", &[c], g.clone());
    let n = b.op("rmsnorm", Op::RmsNorm { eps: EPS }, &[t, gv], &[hw, c]);
    let y = b.transpose(n, &[1, 0]);
    let bv = b.wf32("rms.b", &[c, 1], bias.clone());
    let y = b.add_bcast(y, bv);
    b.reshape(y, &s)
}

/// [`repeat_channels`]: every channel repeated `r` times consecutively.
fn emit_repeat(b: &mut Gb, x: V, r: usize) -> V {
    if r == 1 {
        return x;
    }
    let s = b.shape(x).to_vec();
    let one = b.reshape(x, &[s[0], 1, s[1] * s[2]]);
    let rep = b.concat(&vec![one; r], 1);
    b.reshape(rep, &[s[0] * r, s[1], s[2]])
}

/// [`pixel_shuffle`]: `[C*4, h, w]` -> `[C, 2h, 2w]`.
fn emit_pixel_shuffle(b: &mut Gb, x: V) -> V {
    let s = b.shape(x).to_vec();
    let (c, h, w) = (s[0] / 4, s[1], s[2]);
    let r = b.reshape(x, &[c, 2, 2, h, w]); // [c, i, j, y, x]
    let t = b.transpose(r, &[0, 3, 1, 4, 2]); // [c, y, i, x, j]
    b.reshape(t, &[c, 2 * h, 2 * w])
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
    gpu: ByShape,
}

impl DcAeDecoder {
    /// `<snapshot>/vae` of a diffusers `AutoencoderDC`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = read_config(dir)?;
        Self::open(&crate::model::dit::component_weights(dir)?, &cfg)
    }

    fn open(w: &Weights, cfg: &serde_json::Value) -> Result<Self, String> {
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
                        conv: load_conv(w, &format!("decoder.up_blocks.{i}.0.conv"))?,
                        c_out: c,
                    },
                ));
                j = 1;
            }
            for l in 0..layers[i] {
                let p = format!("decoder.up_blocks.{i}.{}", j + l);
                let b = match types[i].as_str() {
                    "ResBlock" => Block::Res {
                        conv1: load_conv(w, &format!("{p}.conv1"))?,
                        conv2: load_conv(w, &format!("{p}.conv2"))?,
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
                                inverted: load_conv(w, &format!("{g}.conv_inverted"))?,
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
            conv_in: load_conv(w, "decoder.conv_in")?,
            blocks,
            norm_out: norm("decoder.norm_out")?,
            conv_out: load_conv(w, "decoder.conv_out")?,
            head_dim: cfg["attention_head_dim"].as_u64().unwrap_or(32) as usize,
            scaling_factor: cfg["scaling_factor"]
                .as_f64()
                .ok_or("DC-AE config: missing scaling_factor")? as f32,
            gpu: ByShape::default(),
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

    /// Native CPU decoder over `z = latents / scaling_factor` -> `[3, factor*h, factor*w]`.
    fn decode_cpu(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("dcae::decode");
        let c0 = self.conv_in.2;
        let mut x = conv(z, self.latent, h, w, &self.conv_in);
        nn::add_in_place(&mut x, &repeat_channels(z, self.latent, c0 / self.latent));
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
        conv(&x, c, h, w, &self.conv_out)
    }

    /// Metal decoder (same outputs as [`Self::decode_cpu`]); one program per latent size.
    fn decode_gpu(&self, z: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        let net = self.gpu.get(&[h, w], || {
            let mut b = Gb::new();
            let zv = b.input("z", &[self.latent, h, w]);
            let c0 = self.conv_in.2;
            let x = emit_conv(&mut b, zv, &self.conv_in);
            let rep = emit_repeat(&mut b, zv, c0 / self.latent);
            let mut x = b.add(x, rep);
            let mut c = c0;
            for (_, blk) in &self.blocks {
                match blk {
                    Block::Up { conv: cv, c_out } => {
                        let up = b.upsample(x, 2, 2);
                        let y = emit_conv(&mut b, up, cv);
                        let r = emit_repeat(&mut b, x, c_out * 4 / c);
                        let sc = emit_pixel_shuffle(&mut b, r);
                        x = b.add(y, sc);
                        c = *c_out;
                    }
                    Block::Res { conv1, conv2, norm } => {
                        let t = emit_conv(&mut b, x, conv1);
                        let t = b.silu(t);
                        let t = emit_conv(&mut b, t, conv2);
                        let t = emit_norm(&mut b, t, norm);
                        x = b.add(t, x);
                    }
                    Block::EVit(e) => x = self.emit_evit(&mut b, e, x),
                }
            }
            x = emit_norm(&mut b, x, &self.norm_out);
            let s = b.shape(x).to_vec();
            x = b.op("relu", Op::Relu, &[x], &s);
            x = emit_conv(&mut b, x, &self.conv_out);
            b.output(x);
            b.compile()
        })?;
        Ok(net.run(vec![z.to_vec()])?.remove(0))
    }

    /// [`Self::evit`] on `x [c, h, w]`: ReLU linear attention as batched MatMuls over `[heads, hd, hw]`.
    fn emit_evit(&self, b: &mut Gb, e: &EVit, x: V) -> V {
        let s = b.shape(x).to_vec();
        let (c, hw) = (s[0], s[1] * s[2]);
        let (inner, hd) = (e.inner, self.head_dim);
        let qkv = emit_conv(b, x, &([e.q.as_slice(), &e.k, &e.v].concat(), None, 3 * inner, 1));
        let ms = emit_depthwise(b, qkv, &e.ms_in, None, e.ms_k);
        let groups = 3 * inner / hd;
        let wv = b.wf32("ms.out", &[3 * inner, hd, 1, 1], e.ms_out.clone());
        let sh = b.shape(ms).to_vec();
        let ms2 = b.op(
            "ms_out",
            Op::Conv2d { stride: [1; 2], padding: [0; 4], dilation: [1; 2], groups },
            &[ms, wv],
            &sh,
        );
        let all = b.concat(&[qkv, ms2], 0);
        let heads = 6 * inner / (3 * hd);
        let all = b.reshape(all, &[heads, 3, hd, hw]);
        let mut part = |b: &mut Gb, i: usize| {
            let p = b.slice(all, 1, i, i + 1);
            b.reshape(p, &[heads, hd, hw])
        };
        let (q, k, v) = (part(b, 0), part(b, 1), part(b, 2));
        let sq = [heads, hd, hw];
        let q = b.op("relu", Op::Relu, &[q], &sq);
        let k = b.op("relu", Op::Relu, &[k], &sq);
        // kv[c][d] = sum_j v[c][j] k[d][j]; ksum[d] = sum_j k[d][j].
        let kv = b.op("kv", Op::MatMul { transpose_a: false, transpose_b: true }, &[v, k], &[heads, hd, hd]);
        let ksum = b.op(
            "ksum",
            Op::Reduce { kind: crate::graph::ReduceKind::Sum, axes: vec![2], keep_dims: true },
            &[k],
            &[heads, hd, 1],
        );
        let num = b.op("num", Op::MatMul { transpose_a: false, transpose_b: false }, &[kv, q], &sq);
        let den = b.op("den", Op::MatMul { transpose_a: true, transpose_b: false }, &[ksum, q], &[heads, 1, hw]);
        let den = b.add_scalar(den, 1e-15);
        let att = b.op("att", Op::Binary { kind: crate::graph::BinaryKind::Div }, &[num, den], &sq);
        let att = b.reshape(att, &[heads * hd, s[1], s[2]]);
        let y = emit_conv(b, att, &(e.out.clone(), None, c, 1));
        let y = emit_norm(b, y, &e.norm);
        let y = b.add(y, x);
        // GLU MBConv with residual.
        let g = &e.glu;
        let t = emit_conv(b, y, &g.inverted);
        let t = b.silu(t);
        let c2 = g.inverted.2;
        let t = emit_depthwise(b, t, &g.depth_w, Some(&g.depth_b), 3);
        let a = b.slice(t, 0, 0, c2 / 2);
        let gate = b.slice(t, 0, c2 / 2, c2);
        let gate = b.silu(gate);
        let m = b.mul(a, gate);
        let z = emit_conv(b, m, &(g.point.clone(), None, c, 1));
        let z = emit_norm(b, z, &g.norm);
        b.add(z, y)
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
        let y = if crate::model::diff_graph::gpu_mode() {
            self.decode_gpu(&z, h, w)?
        } else {
            self.decode_cpu(&z, h, w)
        };
        let (h, w) = (h * self.factor(), w * self.factor());
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

    /// Tiny f32c8-style decoder (ResBlock + 2 EfficientViT levels, 2 upsamples): CPU vs the Metal
    /// graph program (f32 vs f32: 1e-3), every node on the GPU.
    #[test]
    fn gpu_matches_cpu() {
        use crate::model::diff_graph::testutil::{rel_err, synth, Lcg};
        let (chans, hd, latent, ms_k) = ([8usize, 16, 32], 4usize, 8usize, 5usize);
        let cfg = serde_json::json!({
            "decoder_block_out_channels": chans, "decoder_layers_per_block": [1, 1, 1],
            "decoder_block_types": ["ResBlock", "EfficientViTBlock", "EfficientViTBlock"],
            "upsample_block_type": "interpolate", "decoder_norm_types": "rms_norm",
            "decoder_act_fns": "silu", "latent_channels": latent, "attention_head_dim": hd,
            "scaling_factor": 0.5,
        });
        let mut s: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |s: &mut Vec<_>, p: &str, ci: usize, co: usize, k: usize, bias: bool| {
            s.push((format!("{p}.weight"), vec![co, ci, k, k]));
            if bias {
                s.push((format!("{p}.bias"), vec![co]));
            }
        };
        let norm = |s: &mut Vec<_>, p: &str, c: usize| {
            s.push((format!("{p}.weight"), vec![c]));
            s.push((format!("{p}.bias"), vec![c]));
        };
        conv(&mut s, "decoder.conv_in", latent, chans[2], 3, true);
        // Level 2 (deepest, no upsample): EfficientViT; level 1: upsample (32 -> 16) + EfficientViT;
        // level 0: upsample (16 -> 8) + ResBlock.
        for i in (0..3).rev() {
            let c = chans[i];
            let j = if i + 1 < 3 {
                conv(&mut s, &format!("decoder.up_blocks.{i}.0.conv"), chans[i + 1], c, 3, true);
                1
            } else {
                0
            };
            let p = format!("decoder.up_blocks.{i}.{j}");
            if i == 0 {
                conv(&mut s, &format!("{p}.conv1"), c, c, 3, true);
                conv(&mut s, &format!("{p}.conv2"), c, c, 3, false);
                norm(&mut s, &format!("{p}.norm"), c);
            } else {
                let (a, inner, c2) = (format!("{p}.attn"), c, 4 * c);
                for n in ["to_q", "to_k", "to_v"] {
                    s.push((format!("{a}.{n}.weight"), vec![inner, c]));
                }
                s.push((format!("{a}.to_qkv_multiscale.0.proj_in.weight"), vec![3 * inner, 1, ms_k, ms_k]));
                s.push((format!("{a}.to_qkv_multiscale.0.proj_out.weight"), vec![3 * inner, hd, 1, 1]));
                s.push((format!("{a}.to_out.weight"), vec![c, 2 * inner]));
                norm(&mut s, &format!("{a}.norm_out"), c);
                let g = format!("{p}.conv_out");
                conv(&mut s, &format!("{g}.conv_inverted"), c, c2, 1, true);
                s.push((format!("{g}.conv_depth.weight"), vec![c2, 1, 3, 3]));
                s.push((format!("{g}.conv_depth.bias"), vec![c2]));
                s.push((format!("{g}.conv_point.weight"), vec![c, c2 / 2]));
                norm(&mut s, &format!("{g}.norm"), c);
            }
        }
        norm(&mut s, "decoder.norm_out", chans[0]);
        conv(&mut s, "decoder.conv_out", chans[0], 3, 3, true);
        let r = synth(&s, 41);
        let w = Weights::from_f32(s, |n| Ok(r.0[n].1.clone())).unwrap();
        let dec = DcAeDecoder::open(&w, &cfg).unwrap();
        let (h, wd) = (2, 3);
        let z = Lcg(3).vec(latent * h * wd, 1.0);
        let cpu = dec.decode_cpu(&z, h, wd);
        let gpu = dec.decode_gpu(&z, h, wd).unwrap();
        let e = rel_err(&gpu, &cpu);
        println!("dcae decode rel err {e:e}");
        assert!(e < 1e-3, "decode rel err {e}");
        let net = dec.gpu.get(&[h, wd], || Err("cached".into())).unwrap();
        assert_eq!(net.cpu_node_count(), 0);
    }
}
