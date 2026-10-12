//! ModelScope text-to-video UNet (diffusers `UNet3DConditionModel`) on real safetensors weights, f32 on CPU.
//!
//! Reuses the SD UNet blocks (`ResnetBlock2D` == LDM ResBlock, `Transformer2DModel` == SpatialTransformer)
//! and adds the two temporal pieces: `TemporalConvLayer` and `TransformerTemporalModel`. Activations are
//! frame-major `[frames, c, h*w]`; per-frame 2-D ops see one `[c, h*w]` frame at a time, temporal ops
//! regroup the whole clip.

use crate::model::loaders::{SafetensorsWeightReader, WeightReader};
use crate::model::nn;
use crate::model::unet::{
    add_into, frames_transpose, timestep_embedding, Conv, Linear, Norm, ResBlock, SpatialTransformer, Weights,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ponytail: the ali-vilab/text-to-video-ms-1.7b unet/config.json is fixed in code; read it if other checkpoints matter.
const IN_CH: usize = 4;
const BLOCK_OUT: [usize; 4] = [320, 640, 1280, 1280];
const LAYERS: usize = 2;
const CTX_DIM: usize = 1024;
/// `transformer_in` is `TransformerTemporalModel(num_attention_heads=8, attention_head_dim=64)`.
const TRANSFORMER_IN_WIDTH: usize = 8 * 64;
/// Three stride-2 downsamplers.
const DOWNSCALE: usize = 8;

/// diffusers `TemporalConvLayer`: four GroupNorm → SiLU → Conv3d (3,1,1) blocks plus identity.
/// Each conv is stored as its three frame taps stacked into one `[3c, c]` linear (bias on the centre tap).
struct TempConv(Vec<(Norm, Linear)>);

/// Conv3d `[c, c, 3, 1, 1]` weight + bias -> `[3c, c]` linear whose output rows are `[tap0 | tap1 | tap2]`.
fn stack_taps(w: &[f32], b: &[f32], c: usize) -> Linear {
    let mut stacked = vec![0.0; 3 * c * c];
    for co in 0..c {
        for ci in 0..c {
            for k in 0..3 {
                stacked[(k * c + co) * c + ci] = w[(co * c + ci) * 3 + k];
            }
        }
    }
    let mut bias = vec![0.0; 3 * c];
    bias[c..2 * c].copy_from_slice(b);
    Linear { w: nn::Wt::F32(stacked), b: Some(bias), in_f: c, out_f: 3 * c }
}

/// Conv3d (3,1,1) pad (1,0,0) epilogue. `z` rows are `(position, frame)` holding the per-tap products
/// `[tap0 | tap1 | tap2]`; `out[f] = tap1[f] + tap0[f-1] + tap2[f+1]`, zero outside the clip.
fn sum_taps(z: &[f32], frames: usize, c: usize) -> Vec<f32> {
    let rows = z.len() / (3 * c);
    let mut y = Vec::with_capacity(rows * c);
    for r in 0..rows {
        let f = r % frames;
        let start = y.len();
        y.extend_from_slice(&z[(3 * r + 1) * c..][..c]);
        if f > 0 {
            add_into(&mut y[start..], &z[3 * (r - 1) * c..][..c]);
        }
        if f + 1 < frames {
            add_into(&mut y[start..], &z[(3 * (r + 1) + 2) * c..][..c]);
        }
    }
    y
}

impl TempConv {
    /// `x [frames, c, hw]` -> same layout.
    fn forward(&self, x: &[f32], frames: usize, c: usize, hw: usize) -> Vec<f32> {
        let n = frames * hw;
        // (c, position, frame): GroupNorm statistics span every frame, as on diffusers' (B, C, F, H, W) tensor.
        let id = nn::transpose(x, frames, c * hw);
        let mut h = id.clone();
        for (norm, conv) in &self.0 {
            norm.group(&mut h, c, n, 1e-5);
            nn::silu(&mut h);
            let z = conv.run(&nn::transpose(&h, c, n), n);
            h = nn::transpose(&sum_taps(&z, frames, c), n, c);
        }
        add_into(&mut h, &id);
        nn::transpose(&h, c * hw, frames)
    }
}

/// `[frames, c, hw]` -> `(c, position, frame)` layout, GroupNorm'd with statistics over all frames.
fn norm_frames(x: &[f32], frames: usize, c: usize, hw: usize, norm: &Norm, eps: f32) -> Vec<f32> {
    let mut t = nn::transpose(x, frames, c * hw);
    norm.group(&mut t, c, hw * frames, eps);
    t
}

/// diffusers `TransformerTemporalModel`: the SpatialTransformer weight set, attending along frames
/// (attn1 and attn2 are both self-attention) with clip-wide GroupNorm statistics.
struct Temporal(SpatialTransformer);

impl Temporal {
    /// `x [frames, c, hw]` -> same layout.
    fn forward(&self, x: &[f32], frames: usize, c: usize, hw: usize) -> Vec<f32> {
        let t = &self.0;
        let n = frames * hw;
        let normed = nn::transpose(&norm_frames(x, frames, c, hw, &t.norm, 1e-6), c * hw, frames);
        let mut tok = t.proj_in.run(&frames_transpose(&normed, frames, c, hw), n);
        t.block(&mut tok, frames, hw, None);
        let mut out = frames_transpose(&t.proj_out.run(&tok, n), frames, hw, c);
        add_into(&mut out, x);
        out
    }
}

/// Clip activation `[frames, c, h*w]`.
#[derive(Clone)]
struct Vid {
    data: Vec<f32>,
    c: usize,
    h: usize,
    w: usize,
}

/// Runs `op` on each of `frames` equal chunks of `x` and concatenates the results.
fn per_frame(
    x: &[f32],
    frames: usize,
    mut op: impl FnMut(&[f32]) -> Result<Vec<f32>, String>,
) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for f in x.chunks_exact(x.len() / frames) {
        out.extend_from_slice(&op(f)?);
    }
    Ok(out)
}

fn conv_frames(conv: &Conv, data: &[f32], h: usize, w: usize, frames: usize, stride: usize) -> Result<Vid, String> {
    let data = per_frame(data, frames, |f| Ok(conv.run(f, h, w, stride)?.0))?;
    Ok(Vid { data, c: conv.c_out, h: h.div_ceil(stride), w: w.div_ceil(stride) })
}

struct Attn {
    spatial: SpatialTransformer,
    temporal: Temporal,
}

/// One resnet → temp_conv (→ attentions → temp_attentions) step of a 3-D block.
struct Layer3 {
    res: ResBlock,
    temp_conv: TempConv,
    attn: Option<Attn>,
}

/// Shared per-call inputs.
struct Cond<'a> {
    frames: usize,
    emb: &'a [f32],
    ctx: &'a [f32],
    ctx_len: usize,
}

impl Attn {
    fn forward(&self, data: &[f32], c: usize, hw: usize, k: &Cond) -> Result<Vec<f32>, String> {
        let s = self.spatial.forward(data, k.frames, c, hw, k.ctx, k.ctx_len)?;
        Ok(self.temporal.forward(&s, k.frames, c, hw))
    }
}

impl Layer3 {
    fn forward(&self, v: &Vid, k: &Cond) -> Result<Vid, String> {
        let (c, hw) = (self.res.out_conv.c_out, v.h * v.w);
        let data = per_frame(&v.data, k.frames, |f| self.res.forward(f, v.h, v.w, k.emb))?;
        let mut data = self.temp_conv.forward(&data, k.frames, c, hw);
        if let Some(a) = &self.attn {
            data = a.forward(&data, c, hw, k)?;
        }
        Ok(Vid { data, c, h: v.h, w: v.w })
    }
}

struct Stage {
    layers: Vec<Layer3>,
    /// Downsample2D (stride-2 conv) on the way down, Upsample2D (nearest 2x + conv) on the way up.
    resample: Option<Conv>,
}

/// diffusers `UNet3DConditionModel` (ModelScope text-to-video), epsilon prediction.
pub struct Unet3d {
    time1: Linear,
    time2: Linear,
    conv_in: Conv,
    transformer_in: Temporal,
    down: Vec<Stage>,
    /// `UNetMidBlock3DCrossAttn`: resnet/temp_conv, attentions, resnet/temp_conv.
    mid: (Layer3, Attn, Layer3),
    up: Vec<Stage>,
    norm_out: Norm,
    conv_out: Conv,
}

/// Strict safetensors loader: PyTorch shapes, F16/F32 -> f32, and a record of consumed tensors.
struct St {
    reader: SafetensorsWeightReader,
    shapes: HashMap<String, Vec<usize>>,
    used: RefCell<HashSet<String>>,
}

impl St {
    fn open(path: &Path) -> Result<Self, String> {
        let reader = SafetensorsWeightReader::new(path)?;
        let shapes = reader.list_tensors()?.into_iter().map(|m| (m.name, m.shape)).collect();
        Ok(Self { reader, shapes, used: RefCell::default() })
    }

    fn shape(&self, name: &str) -> Result<&[usize], String> {
        self.shapes.get(name).map(Vec::as_slice).ok_or_else(|| format!("missing UNet3D tensor {name}"))
    }

    /// Loads `name`, which must have exactly the PyTorch shape `want`.
    fn load(&self, name: &str, want: &[usize]) -> Result<Vec<f32>, String> {
        let got = self.shape(name)?;
        if got != want {
            return Err(format!("UNet3D tensor {name} has shape {got:?}, expected {want:?}"));
        }
        self.used.borrow_mut().insert(name.to_owned());
        self.reader.load_tensor(name).map_err(|e| format!("UNet3D tensor {name}: {e}"))
    }

    fn lin(&self, base: &str, in_f: usize, out_f: usize) -> Result<Linear, String> {
        let w = self.load(&format!("{base}.weight"), &[out_f, in_f])?;
        Ok(Linear { w: nn::Wt::F32(w), b: Some(self.vec(&format!("{base}.bias"), out_f)?), in_f, out_f })
    }

    fn conv(&self, base: &str, c_in: usize, c_out: usize, k: usize) -> Result<Conv, String> {
        let w = self.load(&format!("{base}.weight"), &[c_out, c_in, k, k])?;
        Ok(Conv { w: nn::Wt::F32(w), b: self.vec(&format!("{base}.bias"), c_out)?, c_in, c_out, k })
    }

    /// diffusers `ResnetBlock2D`; the 1x1 `conv_shortcut` exists exactly when channels change.
    fn res(&self, base: &str, c_in: usize, c_out: usize) -> Result<ResBlock, String> {
        let skip = if c_in != c_out { Some(self.conv(&format!("{base}.conv_shortcut"), c_in, c_out, 1)?) } else { None };
        Ok(ResBlock {
            in_norm: self.norm(&format!("{base}.norm1"), c_in)?,
            in_conv: self.conv(&format!("{base}.conv1"), c_in, c_out, 3)?,
            emb: self.lin(&format!("{base}.time_emb_proj"), 4 * BLOCK_OUT[0], c_out)?,
            out_norm: self.norm(&format!("{base}.norm2"), c_out)?,
            out_conv: self.conv(&format!("{base}.conv2"), c_out, c_out, 3)?,
            skip,
        })
    }

    /// `conv1` is Sequential(GroupNorm, SiLU, Conv3d); `conv2..4` insert Dropout before the Conv3d.
    fn temp_conv(&self, base: &str, c: usize) -> Result<TempConv, String> {
        let blocks = (1..=4)
            .map(|i| -> Result<(Norm, Linear), String> {
                let conv = format!("{base}.conv{i}.{}", if i == 1 { 2 } else { 3 });
                let w = self.load(&format!("{conv}.weight"), &[c, c, 3, 1, 1])?;
                let b = self.vec(&format!("{conv}.bias"), c)?;
                Ok((self.norm(&format!("{base}.conv{i}.0"), c)?, stack_taps(&w, &b, c)))
            })
            .collect::<Result<_, String>>()?;
        Ok(TempConv(blocks))
    }

    fn transformer_dims(&self, base: &str, c: usize, inner: usize, kv: usize) -> Result<SpatialTransformer, String> {
        let t = self.transformer(base)?;
        let got = (t.proj_in.in_f, t.proj_in.out_f, t.blocks[0].attn2.k.in_f);
        if got != (c, inner, kv) {
            return Err(format!("UNet3D {base}: (channels, width, attn2 kv) = {got:?}, expected {:?}", (c, inner, kv)));
        }
        Ok(t)
    }

    fn temporal(&self, base: &str, c: usize, inner: usize) -> Result<Temporal, String> {
        self.transformer_dims(base, c, inner, inner).map(Temporal)
    }

    fn attn(&self, base: &str, j: usize, c: usize) -> Result<Attn, String> {
        Ok(Attn {
            spatial: self.transformer_dims(&format!("{base}.attentions.{j}"), c, c, CTX_DIM)?,
            temporal: self.temporal(&format!("{base}.temp_attentions.{j}"), c, c)?,
        })
    }

    fn layer(&self, base: &str, j: usize, c_in: usize, c_out: usize, attn: bool) -> Result<Layer3, String> {
        Ok(Layer3 {
            res: self.res(&format!("{base}.resnets.{j}"), c_in, c_out)?,
            temp_conv: self.temp_conv(&format!("{base}.temp_convs.{j}"), c_out)?,
            attn: if attn { Some(self.attn(base, j, c_out)?) } else { None },
        })
    }

    /// Errors if the file holds tensors no block consumed (a mis-wired or missing block).
    fn finish(&self) -> Result<(), String> {
        let used = self.used.borrow();
        let mut extra: Vec<&str> = self.shapes.keys().filter(|n| !used.contains(*n)).map(String::as_str).collect();
        if extra.is_empty() {
            return Ok(());
        }
        extra.sort_unstable();
        Err(format!("UNet3D left {} tensors unused, e.g. {:?}", extra.len(), &extra[..extra.len().min(8)]))
    }
}

impl Weights for St {
    fn has(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }

    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        self.load(name, &[len])
    }

    fn linear(&self, base: &str, bias: bool) -> Result<Linear, String> {
        let name = format!("{base}.weight");
        let shape = self.shape(&name)?;
        let &[out_f, in_f] = shape else {
            return Err(format!("UNet3D tensor {name} has non-linear shape {shape:?}"));
        };
        let w = self.load(&name, &[out_f, in_f])?;
        let b = if bias { Some(self.vec(&format!("{base}.bias"), out_f)?) } else { None };
        Ok(Linear { w: nn::Wt::F32(w), b, in_f, out_f })
    }
}

impl Unet3d {
    /// diffusers UNet3DConditionModel from a safetensors file (F16 or F32, PyTorch shapes).
    /// Missing/mis-shaped tensor = Err naming it.
    pub fn load(path: &Path) -> Result<Self, String> {
        crate::graph::avenues::gate(crate::graph::avenues::Family::VideoModelScope)?;
        let st = St::open(path)?;
        let c0 = BLOCK_OUT[0];
        let top = BLOCK_OUT.len() - 1;

        // CrossAttnDownBlock3D x3, DownBlock3D; every block but the last downsamples.
        let mut down = Vec::with_capacity(BLOCK_OUT.len());
        let mut ch = c0;
        for (i, &out) in BLOCK_OUT.iter().enumerate() {
            let base = format!("down_blocks.{i}");
            let layers = (0..LAYERS)
                .map(|j| st.layer(&base, j, if j == 0 { ch } else { out }, out, i < top))
                .collect::<Result<_, _>>()?;
            let resample =
                if i < top { Some(st.conv(&format!("{base}.downsamplers.0.conv"), out, out, 3)?) } else { None };
            down.push(Stage { layers, resample });
            ch = out;
        }

        let c = BLOCK_OUT[top];
        let mid = (
            st.layer("mid_block", 0, c, c, false)?,
            st.attn("mid_block", 0, c)?,
            st.layer("mid_block", 1, c, c, false)?,
        );

        // UpBlock3D, CrossAttnUpBlock3D x3; every block but the last upsamples. Resnet j consumes the
        // skip of matching width (diffusers' res_skip_channels / resnet_in_channels).
        let mut up = Vec::with_capacity(BLOCK_OUT.len());
        let mut prev = c;
        for i in 0..BLOCK_OUT.len() {
            let base = format!("up_blocks.{i}");
            let out = BLOCK_OUT[top - i];
            let input = BLOCK_OUT[top - (i + 1).min(top)];
            let layers = (0..=LAYERS)
                .map(|j| {
                    let skip = if j == LAYERS { input } else { out };
                    let res_in = if j == 0 { prev } else { out };
                    st.layer(&base, j, res_in + skip, out, i > 0)
                })
                .collect::<Result<_, _>>()?;
            let resample =
                if i < top { Some(st.conv(&format!("{base}.upsamplers.0.conv"), out, out, 3)?) } else { None };
            up.push(Stage { layers, resample });
            prev = out;
        }

        let model = Self {
            time1: st.lin("time_embedding.linear_1", c0, 4 * c0)?,
            time2: st.lin("time_embedding.linear_2", 4 * c0, 4 * c0)?,
            conv_in: st.conv("conv_in", IN_CH, c0, 3)?,
            transformer_in: st.temporal("transformer_in", c0, TRANSFORMER_IN_WIDTH)?,
            down,
            mid,
            up,
            norm_out: st.norm("conv_norm_out", c0)?,
            conv_out: st.conv("conv_out", c0, IN_CH, 3)?,
        };
        st.finish()?;
        Ok(model)
    }

    /// x: [frames, 4, h, w] frame-major; t: DDPM timestep (0..=999); ctx: [ctx_len, 1024], shared by every frame.
    /// Returns eps in x's layout. h, w must be multiples of 8 (3 downsamples).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        frames: usize,
        h: usize,
        w: usize,
        t: f32,
        ctx: &[f32],
        ctx_len: usize,
    ) -> Result<Vec<f32>, String> {
        crate::hardware::cpu_compute::enter("unet3d::forward");
        if frames == 0 || h == 0 || w == 0 || h % DOWNSCALE != 0 || w % DOWNSCALE != 0 {
            return Err(format!("UNet3D needs frames > 0 and {h}x{w} non-zero multiples of {DOWNSCALE}"));
        }
        if x.len() != frames * IN_CH * h * w {
            return Err(format!("UNet3D input has {} values, expected {frames}x{IN_CH}x{h}x{w}", x.len()));
        }
        if ctx_len == 0 || ctx.len() != ctx_len * CTX_DIM {
            return Err(format!("UNet3D context has {} values, expected {ctx_len}x{CTX_DIM}", ctx.len()));
        }

        // Timesteps(320, flip_sin_to_cos=True, 0) -> linear_1, SiLU, linear_2. Every resnet starts its
        // time_emb_proj with SiLU, so apply it once; the embedding is shared by all frames.
        let mut emb = self.time1.run(&timestep_embedding(t, self.time1.in_f), 1);
        nn::silu(&mut emb);
        let mut emb = self.time2.run(&emb, 1);
        nn::silu(&mut emb);
        let k = Cond { frames, emb: &emb, ctx, ctx_len };

        let mut v = conv_frames(&self.conv_in, x, h, w, frames, 1)?;
        v.data = self.transformer_in.forward(&v.data, frames, v.c, h * w);

        let mut skips = vec![v.clone()];
        for stage in &self.down {
            for layer in &stage.layers {
                v = layer.forward(&v, &k)?;
                skips.push(v.clone());
            }
            if let Some(conv) = &stage.resample {
                v = conv_frames(conv, &v.data, v.h, v.w, frames, 2)?;
                skips.push(v.clone());
            }
        }

        let (first, attn, second) = &self.mid;
        v = first.forward(&v, &k)?;
        v.data = attn.forward(&v.data, v.c, v.h * v.w, &k)?;
        v = second.forward(&v, &k)?;

        for stage in &self.up {
            for layer in &stage.layers {
                let s = skips.pop().ok_or("UNet3D skip stack underflow")?;
                if (s.h, s.w) != (v.h, v.w) {
                    return Err(format!("UNet3D skip {}x{} != activation {}x{}", s.h, s.w, v.h, v.w));
                }
                // torch.cat([hidden, skip], dim=1) per frame.
                let (a, b) = (v.c * v.h * v.w, s.c * s.h * s.w);
                let mut data = Vec::with_capacity(frames * (a + b));
                for (hf, sf) in v.data.chunks_exact(a).zip(s.data.chunks_exact(b)) {
                    data.extend_from_slice(hf);
                    data.extend_from_slice(sf);
                }
                v = layer.forward(&Vid { data, c: v.c + s.c, ..v }, &k)?;
            }
            if let Some(conv) = &stage.resample {
                // nearest 2x treats the clip as frames*c channels.
                let data = nn::upsample_nearest2x(&v.data, frames * v.c, v.h, v.w);
                v = conv_frames(conv, &data, 2 * v.h, 2 * v.w, frames, 1)?;
            }
        }

        let hw = h * w;
        for f in v.data.chunks_exact_mut(v.c * hw) {
            self.norm_out.group(f, v.c, hw, 1e-5);
        }
        nn::silu(&mut v.data);
        Ok(conv_frames(&self.conv_out, &v.data, h, w, frames, 1)?.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    /// Deterministic uniform(-s, s) values.
    struct Lcg(std::cell::Cell<u64>);
    impl Lcg {
        fn next(&self, s: f32) -> f32 {
            let v = self.0.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0.set(v);
            ((v >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * s
        }
        fn vec(&self, n: usize, s: f32) -> Vec<f32> {
            (0..n).map(|_| self.next(s)).collect()
        }
    }

    /// Random transformer weights: channels `c`, width `inner`, ff width `2 * inner`.
    struct Rand {
        rng: Lcg,
        c: usize,
        inner: usize,
    }
    impl Weights for Rand {
        fn has(&self, name: &str) -> bool {
            name.contains("transformer_blocks.0.")
        }
        fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            let base = if name.ends_with("weight") { 1.0 } else { 0.0 };
            Ok((0..len).map(|_| base + self.rng.next(0.3)).collect())
        }
        fn linear(&self, base: &str, bias: bool) -> Result<Linear, String> {
            let (c, d) = (self.c, self.inner);
            let (out_f, in_f) = if base.ends_with("proj_in") {
                (d, c)
            } else if base.ends_with("proj_out") {
                (c, d)
            } else if base.ends_with("ff.net.0.proj") {
                (4 * d, d)
            } else if base.ends_with("ff.net.2") {
                (d, 2 * d)
            } else {
                (d, d)
            };
            let s = 1.0 / (in_f as f32).sqrt();
            Ok(Linear { w: nn::Wt::F32(self.rng.vec(out_f * in_f, s)), b: bias.then(|| self.rng.vec(out_f, 0.1)), in_f, out_f })
        }
    }

    fn ref_lin(l: &Linear, x: &[f32]) -> Vec<f32> {
        let w = l.w.f32();
        (0..l.out_f)
            .map(|o| l.b.as_ref().map_or(0.0, |b| b[o]) + (0..l.in_f).map(|i| w[o * l.in_f + i] * x[i]).sum::<f32>())
            .collect()
    }
    fn ref_ln(n: &Norm, x: &[f32]) -> Vec<f32> {
        let d = x.len() as f32;
        let m = x.iter().sum::<f32>() / d;
        let v = x.iter().map(|a| (a - m).powi(2)).sum::<f32>() / d;
        x.iter().enumerate().map(|(i, a)| (a - m) / (v + 1e-5).sqrt() * n.gamma[i] + n.beta[i]).collect()
    }
    /// GroupNorm on x[f][ch][p] with stats over (group channels, all f, all p).
    fn ref_gn(n: &Norm, x: &mut [Vec<Vec<f32>>], eps: f32) {
        let c = x[0].len();
        let cg = c / 32;
        for g in 0..32 {
            let vals: Vec<f32> = x.iter().flat_map(|fr| fr[g * cg..(g + 1) * cg].iter().flatten().copied()).collect();
            let m = vals.iter().sum::<f32>() / vals.len() as f32;
            let v = vals.iter().map(|a| (a - m).powi(2)).sum::<f32>() / vals.len() as f32;
            for fr in x.iter_mut() {
                for ch in g * cg..(g + 1) * cg {
                    for a in fr[ch].iter_mut() {
                        *a = (*a - m) / (v + eps).sqrt() * n.gamma[ch] + n.beta[ch];
                    }
                }
            }
        }
    }
    /// Self-attention over a sequence `xs` of tokens.
    fn ref_attn(a: &crate::model::unet::Attention, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let q: Vec<_> = xs.iter().map(|x| ref_lin(&a.q, x)).collect();
        let k: Vec<_> = xs.iter().map(|x| ref_lin(&a.k, x)).collect();
        let v: Vec<_> = xs.iter().map(|x| ref_lin(&a.v, x)).collect();
        let d = q[0].len();
        xs.iter()
            .enumerate()
            .map(|(i, _)| {
                let mut o = vec![0.0; d];
                for h in 0..d / 64 {
                    let r = h * 64..(h + 1) * 64;
                    let s: Vec<f32> = k
                        .iter()
                        .map(|kj| q[i][r.clone()].iter().zip(&kj[r.clone()]).map(|(a, b)| a * b).sum::<f32>() / 8.0)
                        .collect();
                    let mx = s.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for (j, vj) in v.iter().enumerate() {
                        for t in r.clone() {
                            o[t] += e[j] / z * vj[t];
                        }
                    }
                }
                ref_lin(&a.out, &o)
            })
            .collect()
    }

    #[test]
    fn temporal_matches_naive_reference() {
        let (c, inner, frames, hw) = (64, 128, 8, 40);
        let w = Rand { rng: Lcg(std::cell::Cell::new(5)), c, inner };
        let t = Temporal(w.transformer("t").unwrap());
        let x = w.rng.vec(frames * c * hw, 1.5);
        let got = t.forward(&x, frames, c, hw);

        let (st, b0) = (&t.0, &t.0.blocks[0]);
        let mut xn: Vec<Vec<Vec<f32>>> =
            (0..frames).map(|f| (0..c).map(|ch| x[(f * c + ch) * hw..][..hw].to_vec()).collect()).collect();
        ref_gn(&st.norm, &mut xn, 1e-6);
        for p in 0..hw {
            let mut seq: Vec<Vec<f32>> =
                (0..frames).map(|f| ref_lin(&st.proj_in, &(0..c).map(|ch| xn[f][ch][p]).collect::<Vec<_>>())).collect();
            for (norm, attn) in [(&b0.norm1, &b0.attn1), (&b0.norm2, &b0.attn2)] {
                let n: Vec<_> = seq.iter().map(|s| ref_ln(norm, s)).collect();
                for (s, a) in seq.iter_mut().zip(ref_attn(attn, &n)) {
                    add_into(s, &a);
                }
            }
            for s in seq.iter_mut() {
                let h = ref_lin(&b0.ff_in, &ref_ln(&b0.norm3, s));
                let d = h.len() / 2;
                let g: Vec<f32> = (0..d)
                    .map(|i| {
                        let mut v = [h[d + i]];
                        nn::gelu(&mut v);
                        h[i] * v[0]
                    })
                    .collect();
                add_into(s, &ref_lin(&b0.ff_out, &g));
            }
            for (f, s) in seq.iter().enumerate() {
                let o = ref_lin(&st.proj_out, s);
                for ch in 0..c {
                    let i = (f * c + ch) * hw + p;
                    assert!((got[i] - (o[ch] + x[i])).abs() < 1e-3, "f{f} c{ch} p{p}: {} vs {}", got[i], o[ch] + x[i]);
                }
            }
        }
    }

    #[test]
    fn temp_conv_matches_naive_reference() {
        let (c, frames, hw) = (64, 4, 3);
        let rng = Lcg(std::cell::Cell::new(9));
        let layers: Vec<(Norm, Vec<f32>, Vec<f32>)> = (0..4)
            .map(|_| {
                let n = Norm { gamma: rng.vec(c, 0.3).iter().map(|v| v + 1.0).collect(), beta: rng.vec(c, 0.3) };
                (n, rng.vec(c * c * 3, 0.2), rng.vec(c, 0.1))
            })
            .collect();
        let x = rng.vec(frames * c * hw, 1.0);
        let tc = TempConv(
            layers
                .iter()
                .map(|(n, w, b)| (Norm { gamma: n.gamma.clone(), beta: n.beta.clone() }, stack_taps(w, b, c)))
                .collect(),
        );
        let got = tc.forward(&x, frames, c, hw);

        let mut h: Vec<Vec<Vec<f32>>> =
            (0..frames).map(|f| (0..c).map(|ch| x[(f * c + ch) * hw..][..hw].to_vec()).collect()).collect();
        for (n, w, b) in &layers {
            ref_gn(n, &mut h, 1e-5);
            h.iter_mut().flatten().flatten().for_each(|v| *v = silu(*v));
            h = (0..frames)
                .map(|f| {
                    (0..c)
                        .map(|co| {
                            (0..hw)
                                .map(|p| {
                                    let mut s = b[co];
                                    for k in 0..3 {
                                        let src = f as isize + k as isize - 1;
                                        if (0..frames as isize).contains(&src) {
                                            for ci in 0..c {
                                                s += w[(co * c + ci) * 3 + k] * h[src as usize][ci][p];
                                            }
                                        }
                                    }
                                    s
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
        }
        for f in 0..frames {
            for ch in 0..c {
                for p in 0..hw {
                    let i = (f * c + ch) * hw + p;
                    let want = h[f][ch][p] + x[i];
                    assert!((got[i] - want).abs() < 1e-3, "f{f} c{ch} p{p}: {} vs {want}", got[i]);
                }
            }
        }
    }

    #[test]
    fn timestep_embedding_matches_diffusers_flip_sin_to_cos() {
        // get_timestep_embedding(t, 320, flip_sin_to_cos=True, downscale_freq_shift=0):
        // exponent = -ln(10000) * i / half, emb = cat([sin, cos]) then halves swapped -> [cos, sin].
        let (t, half) = (961.0f32, 160);
        let e = timestep_embedding(t, 2 * half);
        for i in 0..half {
            let arg = t * (-(10000f32.ln()) * i as f32 / half as f32).exp();
            assert!((e[i] - arg.cos()).abs() < 1e-6, "cos {i}");
            assert!((e[half + i] - arg.sin()).abs() < 1e-6, "sin {i}");
        }
    }

    #[test]
    fn norm_frames_spans_all_frames() {
        // c = 32 (one channel per group), hw = 1, 2 frames: per-frame stats would normalize each lone value
        // to 0; clip-wide stats over {1, 3} give -1 and +1.
        let c = 32;
        let x: Vec<f32> = [1.0f32; 32].into_iter().chain([3.0; 32]).collect();
        let norm = Norm { gamma: vec![1.0; c], beta: vec![0.0; c] };
        let y = norm_frames(&x, 2, c, 1, &norm, 1e-6);
        // Layout (c, position, frame).
        for ch in 0..c {
            assert!((y[2 * ch] + 1.0).abs() < 1e-5 && (y[2 * ch + 1] - 1.0).abs() < 1e-5, "{y:?}");
        }
    }

    #[test]
    fn temp_conv_taps_boundaries_and_residual() {
        let (c, frames) = (32, 3);
        let x: Vec<f32> = (0..frames * c).map(|i| (i / c * 100 + i % c) as f32).collect();
        let const_norm = || Norm { gamma: vec![0.0; c], beta: vec![1.0; c] };
        // Diagonal Conv3d weights [c, c, 3] with per-tap values.
        let diag = |taps: [f32; 3]| {
            let mut w = vec![0.0; c * c * 3];
            for ch in 0..c {
                w[(ch * c + ch) * 3..][..3].copy_from_slice(&taps);
            }
            w
        };
        let bias4 = vec![0.5; c];
        let layer = TempConv(vec![
            (const_norm(), stack_taps(&diag([0.0; 3]), &vec![0.0; c], c)),
            (const_norm(), stack_taps(&diag([0.0; 3]), &vec![0.0; c], c)),
            // GroupNorm output is 1 everywhere; only tap0 (previous frame) -> [0, s, s].
            (const_norm(), stack_taps(&diag([1.0, 0.0, 0.0]), &vec![0.0; c], c)),
            (Norm { gamma: vec![1.0; c], beta: vec![0.0; c] }, stack_taps(&diag([1.0, 10.0, 100.0]), &bias4, c)),
        ]);
        let y = layer.forward(&x, frames, c, 1);

        let s = silu(1.0);
        let v = [0.0, s, s];
        let mean = v.iter().sum::<f32>() / 3.0;
        let var = v.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / 3.0;
        let u: Vec<f32> = v.iter().map(|a| silu((a - mean) / (var + 1e-5).sqrt())).collect();
        let conv = [0.5 + 10.0 * u[0] + 100.0 * u[1], 0.5 + u[0] + 10.0 * u[1] + 100.0 * u[2], 0.5 + u[1] + 10.0 * u[2]];
        for f in 0..frames {
            for ch in 0..c {
                let want = x[f * c + ch] + conv[f];
                assert!((y[f * c + ch] - want).abs() < 1e-3, "frame {f} ch {ch}: {} vs {want}", y[f * c + ch]);
            }
        }
    }

    /// Zero weights for one C=320 transformer; attn2 keys/values read `kv`-wide input.
    struct Zeros {
        kv: usize,
    }

    impl Weights for Zeros {
        fn has(&self, name: &str) -> bool {
            name.contains("transformer_blocks.0.")
        }
        fn vec(&self, _: &str, len: usize) -> Result<Vec<f32>, String> {
            Ok(vec![0.0; len])
        }
        fn linear(&self, base: &str, bias: bool) -> Result<Linear, String> {
            let c = 320;
            let (out_f, in_f) = if base.ends_with("ff.net.0.proj") {
                (8 * c, c)
            } else if base.ends_with("ff.net.2") {
                (c, 4 * c)
            } else if base.ends_with("attn2.to_k") || base.ends_with("attn2.to_v") {
                (c, self.kv)
            } else {
                (c, c)
            };
            Ok(Linear { w: nn::Wt::F32(vec![0.0; out_f * in_f]), b: bias.then(|| vec![0.0; out_f]), in_f, out_f })
        }
    }

    #[test]
    fn transformers_at_320_have_five_heads_of_64() {
        let spatial = Zeros { kv: CTX_DIM }.transformer("s").unwrap();
        let temporal = Temporal(Zeros { kv: 320 }.transformer("t").unwrap());
        let (s0, t0) = (&spatial.blocks[0], &temporal.0.blocks[0]);
        for a in [&s0.attn1, &s0.attn2, &t0.attn1, &t0.attn2] {
            assert_eq!(a.heads(), 5);
        }
        // Zero projections leave only the residual; exercises both attention layouts end to end.
        let (frames, hw) = (3, 2);
        let x: Vec<f32> = (0..frames * 320 * hw).map(|i| i as f32 * 0.01).collect();
        let ctx = vec![0.1; 2 * CTX_DIM];
        assert_eq!(spatial.forward(&x, frames, 320, hw, &ctx, 2).unwrap(), x);
        assert_eq!(temporal.forward(&x, frames, 320, hw), x);
    }
}
