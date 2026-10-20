//! The Qwen-Image / Wan 2.1 VAE (diffusers `AutoencoderKLQwenImage`: 16 latent channels, 8x spatial
//! compression, causal 3-D convolutions) for one still frame, on the CPU in f32. With a single
//! frame the causal convolutions only ever see their last temporal tap and the temporal upsamplers
//! are skipped (their first call just primes a cache), so every 3-D layer is a plain 2-D one.
//!
//! Reads the original/ComfyUI names (`decoder.upsamples.N.residual.K`) or diffusers' (`decoder.up_blocks.…`).

use crate::model::diff_graph::{ByShape, Gb, V};
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::weights::Weights;
use serde_json::Value;
use std::path::Path;

/// Wan 2.1 latent statistics (`latents_mean` / `latents_std` of the diffusers config).
const MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];
/// `base_dim 96`, `dim_mult [1, 2, 4, 4]`, 2 residual blocks per level: the only layout shipped.
const BASE: usize = 96;
const MULT: [usize; 4] = [1, 2, 4, 4];
const Z: usize = 16;

/// The original name's diffusers equivalent (`AutoencoderKLQwenImage` module paths).
fn diffusers_name(n: &str) -> String {
    let res = |r: &str| -> String {
        [
            ("residual.0.", "norm1."),
            ("residual.2.", "conv1."),
            ("residual.3.", "norm2."),
            ("residual.6.", "conv2."),
            ("shortcut.", "conv_shortcut."),
        ]
        .iter()
        .find_map(|(a, b)| r.strip_prefix(a).map(|t| format!("{b}{t}")))
        .unwrap_or_else(|| r.to_string())
    };
    // "N.rest" -> (N, rest)
    let idx = |s: &str| -> Option<(usize, String)> {
        let (i, r) = s.split_once('.')?;
        Some((i.parse().ok()?, r.to_string()))
    };
    for (a, b) in [
        ("conv1.", "quant_conv."),
        ("conv2.", "post_quant_conv."),
        ("decoder.conv1.", "decoder.conv_in."),
        ("decoder.head.0.", "decoder.norm_out."),
        ("decoder.head.2.", "decoder.conv_out."),
        ("encoder.conv1.", "encoder.conv_in."),
        ("encoder.head.0.", "encoder.norm_out."),
        ("encoder.head.2.", "encoder.conv_out."),
    ] {
        if let Some(t) = n.strip_prefix(a) {
            return format!("{b}{t}");
        }
    }
    for side in ["decoder", "encoder"] {
        if let Some((i, r)) = n.strip_prefix(&format!("{side}.middle.")).and_then(idx) {
            return match i {
                1 => format!("{side}.mid_block.attentions.0.{r}"),
                _ => format!("{side}.mid_block.resnets.{}.{}", i / 2, res(&r)),
            };
        }
    }
    if let Some((i, r)) = n.strip_prefix("decoder.upsamples.").and_then(idx) {
        let (s, j) = (i / 4, i % 4);
        return match j {
            3 => format!("decoder.up_blocks.{s}.upsamplers.0.{r}"),
            _ => format!("decoder.up_blocks.{s}.resnets.{j}.{}", res(&r)),
        };
    }
    if let Some((i, r)) = n.strip_prefix("encoder.downsamples.").and_then(idx) {
        return format!(
            "encoder.down_blocks.{i}.{}",
            res(&r).replace("resample.1.", "resample.1.")
        );
    }
    n.to_string()
}

/// `[c_out, c_in, k, k]` convolution (a 3-D one keeps its last temporal tap), zero padding `k / 2`.
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
}

impl Conv {
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

/// Channel-first RMS norm: `x / ‖x‖_c * sqrt(c) * gamma` per pixel.
fn rms(x: &mut [f32], c: usize, gamma: &[f32]) {
    let hw = x.len() / c;
    let mut ss = vec![0.0f32; hw];
    for row in x.chunks_exact(hw) {
        for (s, v) in ss.iter_mut().zip(row) {
            *s += v * v;
        }
    }
    let inv: Vec<f32> = ss
        .iter()
        .map(|s| (c as f32).sqrt() / s.sqrt().max(1e-12))
        .collect();
    for (row, g) in x.chunks_exact_mut(hw).zip(gamma) {
        for (v, i) in row.iter_mut().zip(&inv) {
            *v *= i * g;
        }
    }
}

struct Res {
    n1: Vec<f32>,
    c1: Conv,
    n2: Vec<f32>,
    c2: Conv,
    short: Option<Conv>,
}

impl Res {
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let skip = match &self.short {
            Some(s) => s.fwd(x, h, w),
            None => x.to_vec(),
        };
        let mut y = x.to_vec();
        rms(&mut y, self.c1.c_in, &self.n1);
        nn::silu(&mut y);
        let mut y = self.c1.fwd(&y, h, w);
        rms(&mut y, self.c2.c_in, &self.n2);
        nn::silu(&mut y);
        let mut y = self.c2.fwd(&y, h, w);
        nn::add_in_place(&mut y, &skip);
        y
    }
}

// Graph emission (Metal program). Each `emit` mirrors the `fwd` above on `[c, h, w]` values.
impl Gb {
    /// [`rms`] over the channels of `x [c, ...]`: a per-pixel RmsNorm after moving channels last
    /// (`x / sqrt(mean_c x² + eps) * gamma` equals `x / ‖x‖ * sqrt(c) * gamma`).
    pub(crate) fn rms_channels(&mut self, x: V, gamma: &[f32]) -> V {
        let s = self.shape(x).to_vec();
        let (c, hw) = (s[0], s[1..].iter().product::<usize>());
        let flat = if s.len() == 2 { x } else { self.reshape(x, &[c, hw]) };
        let t = self.transpose(flat, &[1, 0]);
        let g = self.wf32("rms.g", &[c], gamma.to_vec());
        let n = self.op("rmsnorm", crate::graph::Op::RmsNorm { eps: 1e-12 }, &[t, g], &[hw, c]);
        let y = self.transpose(n, &[1, 0]);
        if s.len() == 2 { y } else { self.reshape(y, &s) }
    }
}

impl Conv {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        b.conv2d(x, &self.w, Some(&self.b), self.c_out, self.k, 1, self.k / 2)
    }
}

impl Res {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        let skip = match &self.short {
            Some(s) => s.emit(b, x),
            None => x,
        };
        let y = b.rms_channels(x, &self.n1);
        let y = b.silu(y);
        let y = self.c1.emit(b, y);
        let y = b.rms_channels(y, &self.n2);
        let y = b.silu(y);
        let y = self.c2.emit(b, y);
        b.add(y, skip)
    }
}

impl Attn {
    fn emit(&self, b: &mut Gb, x: V) -> V {
        let s = b.shape(x).to_vec();
        let (c, hw) = (self.proj.c_in, s[1] * s[2]);
        let y = b.rms_channels(x, &self.norm);
        let flat = b.reshape(y, &[c, hw]);
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

/// Single-head spatial self-attention over all pixels.
struct Attn {
    norm: Vec<f32>,
    qkv: Conv,
    proj: Conv,
}

impl Attn {
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (c, hw) = (self.proj.c_in, h * w);
        let mut y = x.to_vec();
        rms(&mut y, c, &self.norm);
        let qkv = self.qkv.fwd(&y, h, w);
        let tok = |i: usize| nn::transpose(&qkv[i * c * hw..(i + 1) * c * hw], c, hw);
        let o = nn::attention(&tok(0), &tok(1), &tok(2), hw, hw, 1, c, false);
        let mut o = self.proj.fwd(&nn::transpose(&o, hw, c), h, w);
        nn::add_in_place(&mut o, x);
        o
    }
}

struct Net<'a> {
    w: &'a Weights,
    diffusers: bool,
}

impl Net<'_> {
    fn name(&self, n: &str) -> String {
        if self.diffusers {
            diffusers_name(n)
        } else {
            n.to_string()
        }
    }

    fn vec(&self, n: &str, len: usize) -> Result<Vec<f32>, String> {
        self.w.vec_n(&self.name(n), len)
    }

    /// `{p}.weight` [out, in, (kt,) k, k] and `{p}.bias`.
    fn conv(&self, p: &str, c_in: usize, c_out: usize, k: usize) -> Result<Conv, String> {
        let wn = self.name(&format!("{p}.weight"));
        let shape = self.w.shape(&wn)?;
        let w = self.w.vec(&wn)?;
        let kt = if shape.len() == 5 { shape[2] } else { 1 };
        if shape.len() < 4 || shape[0] != c_out || shape[1] != c_in || shape[shape.len() - 1] != k {
            return Err(format!(
                "VAE: '{wn}' has shape {shape:?}, expected [{c_out}, {c_in}, .., {k}, {k}]"
            ));
        }
        // The causal pad puts the current frame on the last temporal tap.
        let w = if kt > 1 {
            let kk = k * k;
            w.chunks_exact(kt * kk)
                .flat_map(|o| o[(kt - 1) * kk..].iter().copied())
                .collect()
        } else {
            w
        };
        Ok(Conv {
            w,
            b: self.vec(&format!("{p}.bias"), c_out)?,
            c_in,
            c_out,
            k,
        })
    }

    fn res(&self, p: &str, c_in: usize, c_out: usize) -> Result<Res, String> {
        Ok(Res {
            n1: self.vec(&format!("{p}.residual.0.gamma"), c_in)?,
            c1: self.conv(&format!("{p}.residual.2"), c_in, c_out, 3)?,
            n2: self.vec(&format!("{p}.residual.3.gamma"), c_out)?,
            c2: self.conv(&format!("{p}.residual.6"), c_out, c_out, 3)?,
            short: (c_in != c_out)
                .then(|| self.conv(&format!("{p}.shortcut"), c_in, c_out, 1))
                .transpose()?,
        })
    }

    fn attn(&self, p: &str, c: usize) -> Result<Attn, String> {
        Ok(Attn {
            norm: self.vec(&format!("{p}.norm.gamma"), c)?,
            qkv: self.conv(&format!("{p}.to_qkv"), c, 3 * c, 1)?,
            proj: self.conv(&format!("{p}.proj"), c, c, 1)?,
        })
    }
}

struct Up {
    res: Vec<Res>,
    /// Nearest 2x then a 3x3 convolution halving the channels (levels 0..=2).
    up: Option<Conv>,
}

pub struct QwenVae {
    post: Conv,
    conv_in: Conv,
    mid: (Res, Attn, Res),
    ups: Vec<Up>,
    norm_out: Vec<f32>,
    conv_out: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
    #[cfg(test)]
    enc: Result<Enc, String>,
    gpu: ByShape,
}

impl QwenVae {
    /// A diffusers `vae/` directory, or the ComfyUI `qwen_image_vae.safetensors` alone (no config:
    /// the Wan 2.1 statistics apply).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = crate::model::dit::read_config(dir).unwrap_or(Value::Null);
        if !cfg.is_null() {
            let ok = cfg["base_dim"].as_u64().unwrap_or(BASE as u64) == BASE as u64
                && cfg["z_dim"].as_u64().unwrap_or(Z as u64) == Z as u64
                && cfg["dim_mult"].as_array().is_none_or(|m| {
                    m.iter()
                        .filter_map(Value::as_u64)
                        .eq(MULT.iter().map(|&m| m as u64))
                })
                && cfg["num_res_blocks"].as_u64().unwrap_or(2) == 2
                && cfg["attn_scales"].as_array().is_none_or(Vec::is_empty);
            if !ok {
                return Err(format!(
                    "{}: only the 96-channel, [1, 2, 4, 4], 2-block, 16-latent Qwen-Image VAE is implemented",
                    dir.display()
                ));
            }
        }
        let stats = |k: &str, d: &[f32]| -> Vec<f32> {
            match cfg[k].as_array() {
                Some(a) if a.len() == Z => {
                    a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
                }
                _ => d.to_vec(),
            }
        };
        Self::open(
            &Weights::open(&crate::model::hub::component_weights(dir)?)?,
            stats("latents_mean", &MEAN),
            stats("latents_std", &STD),
            BASE,
        )
    }

    pub fn open(w: &Weights, mean: Vec<f32>, std: Vec<f32>, base: usize) -> Result<Self, String> {
        let net = Net {
            w,
            diffusers: !w.has("decoder.conv1.weight"),
        };
        let dims: Vec<usize> = [MULT[3]]
            .iter()
            .chain(MULT.iter().rev())
            .map(|m| base * m)
            .collect();
        let top = dims[0];
        let mut ups = Vec::new();
        for i in 0..4 {
            let (cin, cout) = (if i > 0 { dims[i] / 2 } else { dims[i] }, dims[i + 1]);
            let res = (0..3)
                .map(|j| {
                    net.res(
                        &format!("decoder.upsamples.{}", i * 4 + j),
                        if j == 0 { cin } else { cout },
                        cout,
                    )
                })
                .collect::<Result<_, _>>()?;
            let up = (i < 3)
                .then(|| {
                    net.conv(
                        &format!("decoder.upsamples.{}.resample.1", i * 4 + 3),
                        cout,
                        cout / 2,
                        3,
                    )
                })
                .transpose()?;
            ups.push(Up { res, up });
        }
        let last = dims[4];
        Ok(Self {
            post: net.conv("conv2", Z, Z, 1)?,
            conv_in: net.conv("decoder.conv1", Z, top, 3)?,
            mid: (
                net.res("decoder.middle.0", top, top)?,
                net.attn("decoder.middle.1", top)?,
                net.res("decoder.middle.2", top, top)?,
            ),
            ups,
            norm_out: net.vec("decoder.head.0.gamma", last)?,
            conv_out: net.conv("decoder.head.2", last, 3, 3)?,
            mean,
            std,
            #[cfg(test)]
            enc: Enc::open(&net),
            gpu: ByShape::default(),
        })
    }

    /// Normalized latents `[16, h, w]` (what the transformer produces) -> RGB image of 8h x 8w.
    /// A GPU-only process runs the decoder as a Metal graph program built once per latent size.
    pub fn decode(&self, lat: &[f32], h: usize, w: usize) -> Result<ImageBuffer, String> {
        if lat.len() != Z * h * w || h == 0 || w == 0 {
            return Err(format!(
                "VAE decode: latents len {} != {Z}x{h}x{w}",
                lat.len()
            ));
        }
        let z: Vec<f32> = lat
            .chunks_exact(h * w)
            .zip(self.mean.iter().zip(&self.std))
            .flat_map(|(c, (m, s))| c.iter().map(move |v| v * s + m))
            .collect();
        let x = if crate::model::diff_graph::gpu_mode() {
            self.decode_gpu(&z, h, w)?
        } else {
            self.decode_cpu(&z, h, w)
        };
        let (h, w) = (8 * h, 8 * w);
        let mut img = ImageBuffer::new(w, h);
        let hw = h * w;
        let px = |v: f32| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
        for i in 0..hw {
            img.set_pixel(i % w, i / w, px(x[i]), px(x[hw + i]), px(x[2 * hw + i]));
        }
        Ok(img)
    }

    /// Native CPU decoder over de-normalized latents `z [16, h, w]` -> `[3, 8h, 8w]`.
    pub fn decode_cpu(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("qwen_vae::decode");
        let mut x = self.conv_in.fwd(&self.post.fwd(z, h, w), h, w);
        let (mut h, mut w) = (h, w);
        let (r0, a, r1) = &self.mid;
        x = r1.fwd(&a.fwd(&r0.fwd(&x, h, w), h, w), h, w);
        for level in &self.ups {
            for r in &level.res {
                x = r.fwd(&x, h, w);
            }
            if let Some(up) = &level.up {
                x = nn::upsample_nearest2x(&x, up.c_in, h, w);
                (h, w) = (2 * h, 2 * w);
                x = up.fwd(&x, h, w);
            }
        }
        rms(&mut x, self.conv_out.c_in, &self.norm_out);
        nn::silu(&mut x);
        self.conv_out.fwd(&x, h, w)
    }

    /// Metal decoder (same outputs as [`Self::decode_cpu`]).
    pub fn decode_gpu(&self, z: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        let net = self.gpu.get(&[h, w], || {
            let mut b = Gb::new();
            let x = b.input("z", &[Z, h, w]);
            let x = self.post.emit(&mut b, x);
            let mut x = self.conv_in.emit(&mut b, x);
            let (r0, a, r1) = &self.mid;
            x = r0.emit(&mut b, x);
            x = a.emit(&mut b, x);
            x = r1.emit(&mut b, x);
            for level in &self.ups {
                for r in &level.res {
                    x = r.emit(&mut b, x);
                }
                if let Some(up) = &level.up {
                    x = b.upsample(x, 2, 2);
                    x = up.emit(&mut b, x);
                }
            }
            x = b.rms_channels(x, &self.norm_out);
            x = b.silu(x);
            x = self.conv_out.emit(&mut b, x);
            b.output(x);
            b.compile()
        })?;
        Ok(net.run(vec![z.to_vec()])?.remove(0))
    }
}

/// The encoder half, to round-trip real images through the real weights in tests.
#[cfg(test)]
struct Enc {
    conv_in: Conv,
    /// Per level: two residual blocks, then (levels 0..=2) a stride-2 convolution.
    levels: Vec<(Vec<Res>, Option<Conv>)>,
    mid: (Res, Attn, Res),
    norm_out: Vec<f32>,
    conv_out: Conv,
    quant: Conv,
}

#[cfg(test)]
impl Enc {
    fn open(net: &Net) -> Result<Self, String> {
        let dims: Vec<usize> = [1].iter().chain(MULT.iter()).map(|m| BASE * m).collect();
        let levels = (0..4)
            .map(|i| {
                let n = |j: usize| format!("encoder.downsamples.{}", i * 3 + j);
                let res = vec![
                    net.res(&n(0), dims[i], dims[i + 1])?,
                    net.res(&n(1), dims[i + 1], dims[i + 1])?,
                ];
                let down = (i < 3)
                    .then(|| net.conv(&format!("{}.resample.1", n(2)), dims[i + 1], dims[i + 1], 3))
                    .transpose()?;
                Ok((res, down))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            conv_in: net.conv("encoder.conv1", 3, BASE, 3)?,
            levels,
            mid: (
                net.res("encoder.middle.0", 384, 384)?,
                net.attn("encoder.middle.1", 384)?,
                net.res("encoder.middle.2", 384, 384)?,
            ),
            norm_out: net.vec("encoder.head.0.gamma", 384)?,
            conv_out: net.conv("encoder.head.2", 384, 2 * Z, 3)?,
            quant: net.conv("conv1", 2 * Z, 2 * Z, 1)?,
        })
    }
}

#[cfg(test)]
impl QwenVae {
    /// RGB image -> normalized latents `[16, h/8, w/8]` (posterior mean).
    fn encode(&self, img: &ImageBuffer) -> Vec<f32> {
        let (mut h, mut w) = (img.height, img.width);
        let mut x = vec![0.0f32; 3 * h * w];
        for i in 0..h * w {
            let (r, g, b) = img.get_pixel(i % w, i / w);
            [r, g, b]
                .iter()
                .enumerate()
                .for_each(|(c, &v)| x[c * h * w + i] = v as f32 / 127.5 - 1.0);
        }
        let e = self.enc.as_ref().expect("encoder weights");
        x = e.conv_in.fwd(&x, h, w);
        for (res, down) in &e.levels {
            for r in res {
                x = r.fwd(&x, h, w);
            }
            if let Some(d) = down {
                // ZeroPad2d((0, 1, 0, 1)) then a stride-2 valid convolution.
                let (ph, pw) = (h + 1, w + 1);
                let mut p = vec![0.0f32; d.c_in * ph * pw];
                for c in 0..d.c_in {
                    for y in 0..h {
                        p[(c * ph + y) * pw..][..w].copy_from_slice(&x[(c * h + y) * w..][..w]);
                    }
                }
                let (y, oh, ow) =
                    nn::conv2d(&p, d.c_in, ph, pw, &d.w, Some(&d.b), d.c_out, 3, 2, 0);
                (x, h, w) = (y, oh, ow);
            }
        }
        let (r0, a, r1) = &e.mid;
        x = r1.fwd(&a.fwd(&r0.fwd(&x, h, w), h, w), h, w);
        rms(&mut x, 384, &e.norm_out);
        nn::silu(&mut x);
        let x = e.quant.fwd(&e.conv_out.fwd(&x, h, w), h, w);
        x[..Z * h * w]
            .chunks_exact(h * w)
            .zip(self.mean.iter().zip(&self.std))
            .flat_map(|(c, (m, s))| c.iter().map(move |v| (v - m) / s))
            .collect()
    }
}

/// Tiny synthetic decoder for the CPU-free fixtures and parity tests (shared with `tests/cpu_free_matrix.rs`).
pub mod fixture {
    use super::*;
    use crate::model::diff_graph::testutil::{synth, Lcg};

    /// A tiny (base 4) decoder in the original names: one 4D conv of every kind, a real middle attention.
    pub fn tiny_specs(base: usize) -> Vec<(String, Vec<usize>)> {
        let mut s: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |s: &mut Vec<_>, p: &str, i: usize, o: usize, k: usize| {
            s.push((format!("{p}.weight"), vec![o, i, k, k]));
            s.push((format!("{p}.bias"), vec![o]));
        };
        let res = |s: &mut Vec<_>, p: &str, i: usize, o: usize| {
            s.push((format!("{p}.residual.0.gamma"), vec![i]));
            conv(s, &format!("{p}.residual.2"), i, o, 3);
            s.push((format!("{p}.residual.3.gamma"), vec![o]));
            conv(s, &format!("{p}.residual.6"), o, o, 3);
            if i != o {
                conv(s, &format!("{p}.shortcut"), i, o, 1);
            }
        };
        let dims: Vec<usize> = [MULT[3]].iter().chain(MULT.iter().rev()).map(|m| base * m).collect();
        conv(&mut s, "conv2", Z, Z, 1);
        conv(&mut s, "decoder.conv1", Z, dims[0], 3);
        res(&mut s, "decoder.middle.0", dims[0], dims[0]);
        let c = dims[0];
        s.push(("decoder.middle.1.norm.gamma".into(), vec![c]));
        conv(&mut s, "decoder.middle.1.to_qkv", c, 3 * c, 1);
        conv(&mut s, "decoder.middle.1.proj", c, c, 1);
        res(&mut s, "decoder.middle.2", c, c);
        for i in 0..4 {
            let (cin, cout) = (if i > 0 { dims[i] / 2 } else { dims[i] }, dims[i + 1]);
            for j in 0..3 {
                res(&mut s, &format!("decoder.upsamples.{}", i * 4 + j), if j == 0 { cin } else { cout }, cout);
            }
            if i < 3 {
                conv(&mut s, &format!("decoder.upsamples.{}.resample.1", i * 4 + 3), cout, cout / 2, 3);
            }
        }
        s.push(("decoder.head.0.gamma".into(), vec![dims[4]]));
        conv(&mut s, "decoder.head.2", dims[4], 3, 3);
        s
    }

    /// A base-4 decoder with gains near 1.
    pub fn tiny() -> QwenVae {
        let specs = tiny_specs(4);
        let mut r = synth(&specs, 11);
        let mut g = Lcg(5);
        for (n, (_, v)) in r.0.iter_mut() {
            if n.ends_with(".gamma") {
                v.iter_mut().for_each(|x| *x = 1.0 + 0.2 * g.next());
            }
        }
        let w = Weights::from_f32(specs, |n| Ok(r.0[n].1.clone())).unwrap();
        QwenVae::open(&w, MEAN.to_vec(), STD.to_vec(), 4).unwrap()
    }

    /// Public `decode` of the tiny net (Metal when the CPU tripwire is armed) plus a finiteness check of the
    /// raw Metal output.
    pub fn tiny_on_gpu() -> Result<(), String> {
        let vae = tiny();
        let (h, w) = (3, 2);
        let z = Lcg(3).vec(Z * h * w, 1.0);
        let img = vae.decode(&z, h, w)?;
        if (img.width, img.height) != (8 * w, 8 * h) {
            return Err(format!("qwen_vae: {}x{} image", img.width, img.height));
        }
        let raw = vae.decode_gpu(&vae_denorm(&vae, &z, h * w), h, w)?;
        if raw.iter().any(|v| !v.is_finite()) {
            return Err("qwen_vae: non-finite Metal output".into());
        }
        Ok(())
    }

    fn vae_denorm(vae: &QwenVae, lat: &[f32], hw: usize) -> Vec<f32> {
        lat.chunks_exact(hw)
            .zip(vae.mean.iter().zip(&vae.std))
            .flat_map(|(c, (m, s))| c.iter().map(move |v| v * s + m))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_names_follow_the_module_tree() {
        for (orig, want) in [
            (
                "decoder.upsamples.4.shortcut.weight",
                "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
            ),
            (
                "decoder.upsamples.11.resample.1.bias",
                "decoder.up_blocks.2.upsamplers.0.resample.1.bias",
            ),
            (
                "decoder.middle.2.residual.6.weight",
                "decoder.mid_block.resnets.1.conv2.weight",
            ),
            (
                "decoder.middle.1.to_qkv.weight",
                "decoder.mid_block.attentions.0.to_qkv.weight",
            ),
            ("decoder.head.0.gamma", "decoder.norm_out.gamma"),
            ("conv2.weight", "post_quant_conv.weight"),
        ] {
            assert_eq!(diffusers_name(orig), want);
        }
    }

    /// Encode then decode an image with the real weights: the autoencoder must give it back
    /// (a wrong layer in either half breaks the round trip).
    #[test]
    fn real_weights_round_trip() {
        let Some(path) = std::env::var_os("GIBBS_TEST_QWEN_VAE").map(std::path::PathBuf::from)
        else {
            println!("SKIP: set GIBBS_TEST_QWEN_VAE to a qwen_image_vae.safetensors");
            return;
        };
        let w = Weights::open(&path).unwrap();
        let vae = QwenVae::open(&w, MEAN.to_vec(), STD.to_vec(), BASE).unwrap();
        let (n, mut img) = (128usize, ImageBuffer::new(128, 128));
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f32 - 64.0, y as f32 - 64.0);
                let disc = dx * dx + dy * dy < 40.0 * 40.0;
                let stripe = (x / 16 + y / 16) % 2 == 0;
                let (r, g, b) = match (disc, stripe) {
                    (true, _) => (230, 60 + y as u8, 40),
                    (false, true) => (20 + x as u8, 90, 200),
                    _ => (250, 240, 120 + x as u8 / 2),
                };
                img.set_pixel(x, y, r, g, b);
            }
        }
        let lat = vae.encode(&img);
        let back = vae.decode(&lat, n / 8, n / 8).unwrap();
        let mse = img
            .data
            .iter()
            .zip(&back.data)
            .map(|(a, b)| (*a as f32 - *b as f32).powi(2))
            .sum::<f32>()
            / img.data.len() as f32;
        let psnr = 10.0 * (255.0f32 * 255.0 / mse.max(1e-9)).log10();
        println!("VAE round trip PSNR {psnr:.2} dB");
        assert!(psnr > 24.0, "PSNR {psnr}");
    }

    /// CPU decoder vs the Metal graph program on a tiny synthetic checkpoint (f32 vs f32: 1e-3).
    #[test]
    fn gpu_matches_cpu() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        use crate::model::diff_graph::testutil::{rel_err, Lcg};
        let vae = fixture::tiny();
        let (h, wd) = (3, 2);
        let z = Lcg(3).vec(Z * h * wd, 1.0);
        let cpu = vae.decode_cpu(&z, h, wd);
        let gpu = vae.decode_gpu(&z, h, wd).unwrap();
        let e = rel_err(&gpu, &cpu);
        assert!(e < 1e-3, "decode rel err {e}");
        println!("qwen_vae decode rel err {e:e}");
        let net = vae.gpu.get(&[h, wd], || Err("cached".into())).unwrap();
        assert_eq!(net.cpu_node_count(), 0);
    }
}
