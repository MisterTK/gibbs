//! Stable Diffusion UNet (LDM `UNetModel` / diffusers `UNet2DConditionModel`): SD 1.x, 2.x and SDXL,
//! executed directly on real weights (GGUF, LDM single-file or diffusers safetensors).
//!
//! Structure is derived from the checkpoint's tensor names and shapes in the LDM namespace (diffusers
//! checkpoints are aliased onto it by [`alias_diffusers`]): channel counts come from weight shapes;
//! which blocks carry a ResBlock, SpatialTransformer (any depth), Downsample or Upsample comes from
//! which tensors exist. Activations stay in `[c, h*w]` layout and are only transposed to `[h*w, c]`
//! tokens inside the SpatialTransformer. Weights stay in their stored precision ([`nn::Wt`]).

use crate::model::clip::Tensors;
use crate::model::loaders::GgufWeightReader;
use crate::model::diff_graph::{gpu_mode, ByShape, Gb, GpuNet, V};
use crate::model::nn::{self, Wt};
use std::collections::HashSet;

const PREFIX: &str = "model.diffusion_model.";
const GN_GROUPS: usize = 32;
/// SD2 / SDXL / ModelScope `num_head_channels`.
const HEAD_DIM: usize = 64;

/// Feature map in `[c, h*w]` layout.
struct Act {
    data: Vec<f32>,
    c: usize,
    h: usize,
    w: usize,
}

pub(super) struct Linear {
    pub(super) w: Wt,
    pub(super) b: Option<Vec<f32>>,
    pub(super) in_f: usize,
    pub(super) out_f: usize,
}

impl Linear {
    pub(super) fn run(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.w.linear(x, rows, self.b.as_deref(), self.in_f, self.out_f)
    }
}

pub(super) struct Conv {
    pub(super) w: Wt,
    pub(super) b: Vec<f32>,
    pub(super) c_in: usize,
    pub(super) c_out: usize,
    pub(super) k: usize,
}

impl Conv {
    pub(super) fn run(&self, x: &[f32], h: usize, w: usize, stride: usize) -> Result<(Vec<f32>, usize, usize), String> {
        if x.len() != self.c_in * h * w {
            return Err(format!(
                "UNet conv expects {} input channels at {h}x{w}, got {} values",
                self.c_in,
                x.len()
            ));
        }
        Ok(nn::conv2d(x, self.c_in, h, w, &self.w.f32(), Some(&self.b), self.c_out, self.k, stride, self.k / 2))
    }
}

pub(super) struct Norm {
    pub(super) gamma: Vec<f32>,
    pub(super) beta: Vec<f32>,
}

impl Norm {
    pub(super) fn group(&self, x: &mut [f32], c: usize, hw: usize, eps: f32) {
        nn::group_norm(x, c, hw, GN_GROUPS, eps, &self.gamma, &self.beta);
    }
    fn layer(&self, x: &[f32], rows: usize) -> Vec<f32> {
        nn::layer_norm(x, rows, self.gamma.len(), &self.gamma, &self.beta, 1e-5)
    }
}

/// LDM `ResBlock` / diffusers `ResnetBlock2D` (same math) on one image `[c_in, h*w]`.
pub(super) struct ResBlock {
    pub(super) in_norm: Norm,
    pub(super) in_conv: Conv,
    pub(super) emb: Linear,
    pub(super) out_norm: Norm,
    pub(super) out_conv: Conv,
    pub(super) skip: Option<Conv>,
}

impl ResBlock {
    /// `emb_act` is SiLU(time embedding), shared by every ResBlock. Returns `[c_out, h*w]`.
    pub(super) fn forward(&self, x: &[f32], h: usize, w: usize, emb_act: &[f32]) -> Result<Vec<f32>, String> {
        let hw = h * w;
        let c_in = self.in_conv.c_in;
        if x.len() != c_in * hw {
            return Err(format!("UNet ResBlock expects {c_in} channels at {h}x{w}, got {} values", x.len()));
        }
        let mut t = x.to_vec();
        self.in_norm.group(&mut t, c_in, hw, 1e-5);
        nn::silu(&mut t);
        let (mut t, _, _) = self.in_conv.run(&t, h, w, 1)?;
        let e = self.emb.run(emb_act, 1);
        for (row, &bias) in t.chunks_exact_mut(hw).zip(&e) {
            row.iter_mut().for_each(|v| *v += bias);
        }
        let c = self.out_conv.c_out;
        self.out_norm.group(&mut t, self.in_conv.c_out, hw, 1e-5);
        nn::silu(&mut t);
        let (mut out, _, _) = self.out_conv.run(&t, h, w, 1)?;
        match &self.skip {
            Some(conv) => {
                let (s, _, _) = conv.run(x, h, w, 1)?;
                out.iter_mut().zip(&s).for_each(|(o, s)| *o += s);
            }
            None => {
                if c_in != c {
                    return Err(format!("UNet ResBlock identity skip with {c_in} -> {c} channels"));
                }
                out.iter_mut().zip(x).for_each(|(o, s)| *o += s);
            }
        }
        Ok(out)
    }
}

pub(super) struct Attention {
    pub(super) q: Linear,
    pub(super) k: Linear,
    pub(super) v: Linear,
    pub(super) out: Linear,
    pub(super) head_dim: usize,
}

impl Attention {
    pub(super) fn heads(&self) -> usize {
        self.q.out_f / self.head_dim
    }

    /// All `nx` rows of `x` attend to the shared `ctx` (cross-attention, or self when `ctx` is `x`).
    fn forward(&self, x: &[f32], nx: usize, ctx: &[f32], nctx: usize) -> Vec<f32> {
        let q = self.q.run(x, nx);
        let k = self.k.run(ctx, nctx);
        let v = self.v.run(ctx, nctx);
        let o = nn::attention(&q, &k, &v, nx, nctx, self.heads(), self.head_dim, false);
        self.out.run(&o, nx)
    }

    /// Self-attention within each of `frames` consecutive runs of `n` rows.
    fn per_frame(&self, x: &[f32], frames: usize, n: usize) -> Vec<f32> {
        let rows = frames * n;
        let (q, k, v) = (self.q.run(x, rows), self.k.run(x, rows), self.v.run(x, rows));
        let len = n * self.q.out_f;
        let mut o = Vec::with_capacity(rows * self.q.out_f);
        for ((q, k), v) in q.chunks_exact(len).zip(k.chunks_exact(len)).zip(v.chunks_exact(len)) {
            o.extend_from_slice(&nn::attention(q, k, v, n, n, self.heads(), self.head_dim, false));
        }
        self.out.run(&o, rows)
    }

    /// Self-attention along the frame axis. Rows are `(frame, position)`; viewed as
    /// `[frames, n*width]`, every (position, head) pair is an independent head over `frames` tokens,
    /// so all positions run in one attention call.
    fn across_frames(&self, x: &[f32], frames: usize, n: usize) -> Vec<f32> {
        let rows = frames * n;
        let (q, k, v) = (self.q.run(x, rows), self.k.run(x, rows), self.v.run(x, rows));
        let o = nn::attention(&q, &k, &v, frames, frames, n * self.heads(), self.head_dim, false);
        self.out.run(&o, rows)
    }
}

/// diffusers `BasicTransformerBlock`: self-attention, cross-attention, GEGLU feed-forward.
pub(super) struct BasicBlock {
    pub(super) norm1: Norm,
    pub(super) attn1: Attention,
    pub(super) norm2: Norm,
    pub(super) attn2: Attention,
    pub(super) norm3: Norm,
    pub(super) ff_in: Linear,
    pub(super) ff_out: Linear,
}

/// LDM `SpatialTransformer` / diffusers `Transformer2DModel` (linear or 1x1-conv projection) and the
/// weight set of `TransformerTemporalModel`: GroupNorm, proj_in, BasicTransformerBlocks, proj_out.
pub(super) struct SpatialTransformer {
    pub(super) norm: Norm,
    pub(super) proj_in: Linear,
    pub(super) blocks: Vec<BasicBlock>,
    pub(super) proj_out: Linear,
}

/// GEGLU: `x` is `[rows, 2*inner]` split as `[a, gate]`; returns `a * gelu(gate)`.
fn geglu(x: &[f32], rows: usize, inner: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * inner);
    for row in x.chunks_exact(2 * inner) {
        out.extend_from_slice(&row[inner..]);
    }
    nn::gelu(&mut out);
    for (o, row) in out.chunks_exact_mut(inner).zip(x.chunks_exact(2 * inner)) {
        o.iter_mut().zip(&row[..inner]).for_each(|(g, a)| *g *= a);
    }
    out
}

pub(super) fn add_into(dst: &mut [f32], src: &[f32]) {
    dst.iter_mut().zip(src).for_each(|(d, s)| *d += s);
}

/// Transposes each of `frames` consecutive `[rows, cols]` blocks of `x`.
pub(super) fn frames_transpose(x: &[f32], frames: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * rows * cols);
    for f in x.chunks_exact(rows * cols) {
        out.extend_from_slice(&nn::transpose(f, rows, cols));
    }
    out
}

impl SpatialTransformer {
    /// The BasicTransformerBlocks over tokens whose rows are `(frame, position)`, `n` positions per frame.
    /// `ctx: Some` = spatial (attn1 within each frame, attn2 cross-attends the shared context);
    /// `None` = temporal (attn1 and attn2 both self-attend across frames).
    pub(super) fn block(&self, tok: &mut [f32], frames: usize, n: usize, ctx: Option<(&[f32], usize)>) {
        let rows = frames * n;
        let self_attn = |a: &Attention, x: &[f32]| match ctx {
            Some(_) => a.per_frame(x, frames, n),
            None => a.across_frames(x, frames, n),
        };
        for b in &self.blocks {
            let x = b.norm1.layer(tok, rows);
            add_into(tok, &self_attn(&b.attn1, &x));
            let x = b.norm2.layer(tok, rows);
            let a = match ctx {
                Some((c, len)) => b.attn2.forward(&x, rows, c, len),
                None => self_attn(&b.attn2, &x),
            };
            add_into(tok, &a);
            let x = b.norm3.layer(tok, rows);
            let g = geglu(&b.ff_in.run(&x, rows), rows, b.ff_in.out_f / 2);
            add_into(tok, &b.ff_out.run(&g, rows));
        }
    }

    /// Spatial transformer over `frames` images `[frames, c, hw]` sharing `ctx [ctx_len, ctx_dim]`.
    pub(super) fn forward(
        &self,
        x: &[f32],
        frames: usize,
        c: usize,
        hw: usize,
        ctx: &[f32],
        ctx_len: usize,
    ) -> Result<Vec<f32>, String> {
        if c != self.proj_out.out_f || x.len() != frames * c * hw {
            return Err(format!(
                "UNet SpatialTransformer expects {} channels, got {c} ({} values for {frames}x{c}x{hw})",
                self.proj_out.out_f,
                x.len()
            ));
        }
        let mut t = x.to_vec();
        for f in t.chunks_exact_mut(c * hw) {
            self.norm.group(f, c, hw, 1e-6);
        }
        let rows = frames * hw;
        let mut tok = self.proj_in.run(&frames_transpose(&t, frames, c, hw), rows);
        self.block(&mut tok, frames, hw, Some((ctx, ctx_len)));
        let mut out = frames_transpose(&self.proj_out.run(&tok, rows), frames, hw, c);
        add_into(&mut out, x);
        Ok(out)
    }
}

enum Layer {
    ConvIn(Conv),
    Res(ResBlock),
    Transformer(SpatialTransformer),
    Down(Conv),
    Up(Conv),
}

impl Layer {
    fn forward(&self, x: Act, emb_act: &[f32], ctx: &[f32], ctx_len: usize) -> Result<Act, String> {
        match self {
            Layer::ConvIn(conv) => {
                let (data, h, w) = conv.run(&x.data, x.h, x.w, 1)?;
                Ok(Act { data, c: conv.c_out, h, w })
            }
            Layer::Res(r) => {
                let data = r.forward(&x.data, x.h, x.w, emb_act)?;
                Ok(Act { data, c: r.out_conv.c_out, ..x })
            }
            Layer::Transformer(t) => {
                let data = t.forward(&x.data, 1, x.c, x.h * x.w, ctx, ctx_len)?;
                Ok(Act { data, ..x })
            }
            Layer::Down(conv) => {
                let (data, h, w) = conv.run(&x.data, x.h, x.w, 2)?;
                Ok(Act { data, c: conv.c_out, h, w })
            }
            Layer::Up(conv) => {
                let up = nn::upsample_nearest2x(&x.data, x.c, x.h, x.w);
                let (data, h, w) = conv.run(&up, 2 * x.h, 2 * x.w, 1)?;
                Ok(Act { data, c: conv.c_out, h, w })
            }
        }
    }
}

// Graph emission (Metal program). Images are `[c, h, w]`, token streams `[rows, c]`; each `emit`
// mirrors the host `forward` of the same struct.
impl Linear {
    pub(super) fn emit(&self, b: &mut Gb, x: V) -> V {
        b.linear(x, &self.w, self.b.as_deref(), self.in_f, self.out_f)
    }
}

impl Conv {
    pub(super) fn emit(&self, b: &mut Gb, x: V, stride: usize) -> V {
        b.conv2d(x, &self.w.f32(), Some(&self.b), self.c_out, self.k, stride, self.k / 2)
    }
}

impl Norm {
    pub(super) fn emit_group(&self, b: &mut Gb, x: V, eps: f32) -> V {
        b.group_norm(x, &self.gamma, &self.beta, GN_GROUPS, eps)
    }
    pub(super) fn emit_layer(&self, b: &mut Gb, x: V) -> V {
        b.layer_norm(x, &self.gamma, &self.beta, 1e-5)
    }
}

impl ResBlock {
    /// `x [c_in, h, w]`; `emb_act` is `[1, emb_dim]` (SiLU of the time embedding). Returns `[c_out, h, w]`.
    pub(super) fn emit(&self, b: &mut Gb, x: V, emb_act: V) -> V {
        let t = self.in_norm.emit_group(b, x, 1e-5);
        let t = b.silu(t);
        let t = self.in_conv.emit(b, t, 1);
        let e = self.emb.emit(b, emb_act);
        let e = b.reshape(e, &[self.emb.out_f, 1, 1]);
        let t = b.add_bcast(t, e);
        let t = self.out_norm.emit_group(b, t, 1e-5);
        let t = b.silu(t);
        let t = self.out_conv.emit(b, t, 1);
        let skip = match &self.skip {
            Some(c) => c.emit(b, x, 1),
            None => x,
        };
        b.add(t, skip)
    }
}

impl Attention {
    /// `x [nx, d]` attends to `ctx [nctx, dctx]` (self-attention when `ctx == x`).
    pub(super) fn emit(&self, b: &mut Gb, x: V, ctx: V) -> V {
        let q = self.q.emit(b, x);
        let k = self.k.emit(b, ctx);
        let v = self.v.emit(b, ctx);
        let o = b.attention(q, k, v, self.heads());
        self.out.emit(b, o)
    }
}

impl SpatialTransformer {
    /// BasicTransformerBlocks over tokens `tok [n, c]` of one frame: self-attention, cross-attention
    /// over `ctx [len, dim]`, GEGLU feed-forward.
    pub(super) fn emit_blocks(&self, b: &mut Gb, mut tok: V, ctx: V) -> V {
        for blk in &self.blocks {
            let x = blk.norm1.emit_layer(b, tok);
            let a = blk.attn1.emit(b, x, x);
            tok = b.add(tok, a);
            let x = blk.norm2.emit_layer(b, tok);
            let a = blk.attn2.emit(b, x, ctx);
            tok = b.add(tok, a);
            let x = blk.norm3.emit_layer(b, tok);
            let g = blk.ff_in.emit(b, x);
            let inner = blk.ff_in.out_f / 2;
            // GEGLU: value is the first half, the second half goes through the exact GELU.
            let (a, gate) = (b.slice(g, 1, 0, inner), b.slice(g, 1, inner, 2 * inner));
            let gate = b.gelu(gate, false);
            let g = b.mul(a, gate);
            let f = blk.ff_out.emit(b, g);
            tok = b.add(tok, f);
        }
        tok
    }

    /// `x [c, h, w]` -> `[c, h, w]`.
    pub(super) fn emit(&self, b: &mut Gb, x: V, ctx: V) -> V {
        let s = b.shape(x).to_vec();
        let (c, hw) = (s[0], s[1] * s[2]);
        let t = self.norm.emit_group(b, x, 1e-6);
        let t = b.reshape(t, &[c, hw]);
        let t = b.transpose(t, &[1, 0]);
        let tok = self.proj_in.emit(b, t);
        let tok = self.emit_blocks(b, tok, ctx);
        let o = self.proj_out.emit(b, tok);
        let o = b.transpose(o, &[1, 0]);
        let o = b.reshape(o, &s);
        b.add(o, x)
    }
}

impl Layer {
    fn emit(&self, b: &mut Gb, x: V, emb_act: V, ctx: V) -> V {
        match self {
            Layer::ConvIn(conv) => conv.emit(b, x, 1),
            Layer::Res(r) => r.emit(b, x, emb_act),
            Layer::Transformer(t) => t.emit(b, x, ctx),
            Layer::Down(conv) => conv.emit(b, x, 2),
            Layer::Up(conv) => {
                let up = b.upsample(x, 2, 2);
                conv.emit(b, up, 1)
            }
        }
    }
}

/// Sinusoidal timestep embedding (LDM `timestep_embedding`, diffusers `Timesteps(flip_sin_to_cos=True,
/// downscale_freq_shift=0)`): `[cos(t*f), sin(t*f)]`.
pub(crate) fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut emb = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        emb[i] = arg.cos();
        emb[half + i] = arg.sin();
    }
    emb
}

/// Tensor source for blocks whose tensor names LDM and diffusers share.
/// Implementors return linears as PyTorch `[out, in]` row-major data whatever their on-disk dim order.
pub(super) trait Weights {
    fn has(&self, name: &str) -> bool;
    /// 1-D tensor of exactly `len` values.
    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String>;
    /// `<base>.weight` (and `<base>.bias` when `bias`) as a linear layer.
    fn linear(&self, base: &str, bias: bool) -> Result<Linear, String>;

    fn norm(&self, base: &str, c: usize) -> Result<Norm, String> {
        Ok(Norm { gamma: self.vec(&format!("{base}.weight"), c)?, beta: self.vec(&format!("{base}.bias"), c)? })
    }

    /// Head width 64; [`SdUnet`] resets it per its checkpoint's head rule.
    fn attention(&self, base: &str) -> Result<Attention, String> {
        Ok(Attention {
            q: self.linear(&format!("{base}.to_q"), false)?,
            k: self.linear(&format!("{base}.to_k"), false)?,
            v: self.linear(&format!("{base}.to_v"), false)?,
            out: self.linear(&format!("{base}.to_out.0"), true)?,
            head_dim: HEAD_DIM,
        })
    }

    /// GroupNorm + proj_in + every `transformer_blocks.N` + proj_out, with every linear's shape checked
    /// against `proj_in` (channels `c`, width `inner`) and attn2's key/value input width.
    fn transformer(&self, base: &str) -> Result<SpatialTransformer, String> {
        let proj_in = self.linear(&format!("{base}.proj_in"), true)?;
        let (c, inner) = (proj_in.in_f, proj_in.out_f);
        let proj_out = self.linear(&format!("{base}.proj_out"), true)?;
        if (proj_out.in_f, proj_out.out_f) != (inner, c) {
            return Err(format!(
                "UNet tensor {base}.proj_out.weight is [{}, {}] (out, in), expected [{c}, {inner}]",
                proj_out.out_f, proj_out.in_f
            ));
        }
        let mut blocks = Vec::new();
        while self.has(&format!("{base}.transformer_blocks.{}.norm1.weight", blocks.len())) {
            let tb = format!("{base}.transformer_blocks.{}", blocks.len());
            let b = BasicBlock {
                norm1: self.norm(&format!("{tb}.norm1"), inner)?,
                attn1: self.attention(&format!("{tb}.attn1"))?,
                norm2: self.norm(&format!("{tb}.norm2"), inner)?,
                attn2: self.attention(&format!("{tb}.attn2"))?,
                norm3: self.norm(&format!("{tb}.norm3"), inner)?,
                ff_in: self.linear(&format!("{tb}.ff.net.0.proj"), true)?,
                ff_out: self.linear(&format!("{tb}.ff.net.2"), true)?,
            };
            let (ff, kv) = (b.ff_out.in_f, b.attn2.k.in_f);
            let (a1, a2) = (&b.attn1, &b.attn2);
            let checks = [
                ("attn1.to_q", &a1.q, inner, inner),
                ("attn1.to_k", &a1.k, inner, inner),
                ("attn1.to_v", &a1.v, inner, inner),
                ("attn1.to_out.0", &a1.out, inner, inner),
                ("attn2.to_q", &a2.q, inner, inner),
                ("attn2.to_k", &a2.k, kv, inner),
                ("attn2.to_v", &a2.v, kv, inner),
                ("attn2.to_out.0", &a2.out, inner, inner),
                ("ff.net.0.proj", &b.ff_in, inner, 2 * ff),
                ("ff.net.2", &b.ff_out, ff, inner),
            ];
            for (name, l, in_f, out_f) in checks {
                if (l.in_f, l.out_f) != (in_f, out_f) {
                    return Err(format!(
                        "UNet tensor {tb}.{name}.weight is [{}, {}] (out, in), expected [{out_f}, {in_f}]",
                        l.out_f, l.in_f
                    ));
                }
            }
            blocks.push(b);
        }
        if blocks.is_empty() {
            return Err(format!("UNet {base} has no transformer_blocks"));
        }
        Ok(SpatialTransformer { norm: self.norm(&format!("{base}.norm"), c)?, proj_in, blocks, proj_out })
    }
}

/// Attention head layout of a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Heads {
    /// Fixed head width (SD 2.x, SDXL: 64).
    Dim(usize),
    /// Fixed head count (SD 1.x: 8).
    Count(usize),
}

impl Heads {
    fn head_dim(self, inner: usize) -> Result<usize, String> {
        let d = match self {
            Heads::Dim(d) => d,
            Heads::Count(n) if n > 0 => inner / n,
            Heads::Count(_) => 0,
        };
        if d == 0 || inner % d != 0 {
            return Err(format!("UNet attention width {inner} does not split into {self:?}"));
        }
        Ok(d)
    }

    /// From a diffusers UNet config: `num_attention_heads` (or the legacy `attention_head_dim`, which
    /// diffusers reads as a head count) per down block, over `block_out_channels`.
    pub fn from_diffusers(cfg: &serde_json::Value) -> Result<Self, String> {
        let heads = if cfg["num_attention_heads"].is_null() { &cfg["attention_head_dim"] } else { &cfg["num_attention_heads"] };
        if let Some(n) = heads.as_u64() {
            return Ok(Heads::Count(n as usize));
        }
        let chans: Vec<u64> = cfg["block_out_channels"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).collect();
        let heads: Vec<u64> = heads.as_array().into_iter().flatten().filter_map(|v| v.as_u64()).collect();
        let dims: HashSet<u64> = chans.iter().zip(&heads).filter(|(_, &h)| h > 0).map(|(c, h)| c / h).collect();
        match (heads.len() == chans.len() && !chans.is_empty(), dims.len()) {
            (true, 1) => Ok(Heads::Dim(*dims.iter().next().unwrap() as usize)),
            _ => Err(format!("UNet attention heads {heads:?} over channels {chans:?}: unsupported layout")),
        }
    }
}

/// Strict loader over one UNet namespace (`prefix` + LDM names) of a [`Tensors`].
struct Loader<'a> {
    t: &'a Tensors<'a>,
    prefix: &'a str,
    names: HashSet<String>,
    heads: Heads,
}

impl Loader<'_> {
    fn has_prefix(&self, prefix: &str) -> bool {
        let p = format!("{prefix}.");
        self.names.iter().any(|n| n.starts_with(&p))
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        let full = format!("{}{name}", self.prefix);
        self.t.shape(&full).map(<[usize]>::to_vec).ok_or_else(|| format!("missing UNet tensor {full}"))
    }

    fn wt(&self, name: &str) -> Result<Wt, String> {
        let full = format!("{}{name}", self.prefix);
        self.t.load_wt(&full).map_err(|e| format!("UNet: {e}"))
    }

    /// Conv weight `[out, in, k, k]`.
    fn conv(&self, base: &str) -> Result<Conv, String> {
        let name = format!("{base}.weight");
        let shape = self.shape(&name)?;
        let [c_out, c_in, kh, kw] = shape[..] else {
            return Err(format!("UNet tensor {}{name} has non-conv shape {shape:?}", self.prefix));
        };
        if kw != kh || kw % 2 == 0 {
            return Err(format!("UNet tensor {}{name} has unsupported kernel {kh}x{kw}", self.prefix));
        }
        Ok(Conv { w: self.wt(&name)?, b: self.vec(&format!("{base}.bias"), c_out)?, c_in, c_out, k: kw })
    }

    fn res_block(&self, base: &str) -> Result<ResBlock, String> {
        let in_conv = self.conv(&format!("{base}.in_layers.2"))?;
        let out_conv = self.conv(&format!("{base}.out_layers.3"))?;
        let skip_name = format!("{base}.skip_connection");
        let skip = if self.has(&format!("{skip_name}.weight")) { Some(self.conv(&skip_name)?) } else { None };
        Ok(ResBlock {
            in_norm: self.norm(&format!("{base}.in_layers.0"), in_conv.c_in)?,
            emb: self.linear(&format!("{base}.emb_layers.1"), true)?,
            out_norm: self.norm(&format!("{base}.out_layers.0"), in_conv.c_out)?,
            in_conv,
            out_conv,
            skip,
        })
    }

    /// Modules `<block>.0`, `<block>.1`, ... in order, typed by which tensors exist.
    fn block(&self, block: &str) -> Result<Vec<Layer>, String> {
        let mut layers = Vec::new();
        for j in 0.. {
            let sub = format!("{block}.{j}");
            if !self.has_prefix(&sub) {
                break;
            }
            let layer = if self.has(&format!("{sub}.in_layers.0.weight")) {
                Layer::Res(self.res_block(&sub)?)
            } else if self.has(&format!("{sub}.proj_in.weight")) {
                let mut t = self.transformer(&sub)?;
                let d = self.heads.head_dim(t.proj_in.out_f)?;
                for b in &mut t.blocks {
                    b.attn1.head_dim = d;
                    b.attn2.head_dim = d;
                }
                Layer::Transformer(t)
            } else if self.has(&format!("{sub}.op.weight")) {
                Layer::Down(self.conv(&format!("{sub}.op"))?)
            } else if self.has(&format!("{sub}.conv.weight")) {
                Layer::Up(self.conv(&format!("{sub}.conv"))?)
            } else if self.has(&format!("{sub}.weight")) {
                Layer::ConvIn(self.conv(&sub)?)
            } else {
                return Err(format!("UNet module {}{sub} has unrecognized tensors", self.prefix));
            };
            layers.push(layer);
        }
        if layers.is_empty() {
            return Err(format!("UNet block {}{block} has no modules", self.prefix));
        }
        Ok(layers)
    }

    fn blocks(&self, kind: &str) -> Result<Vec<Vec<Layer>>, String> {
        let mut blocks = Vec::new();
        while self.has_prefix(&format!("{kind}.{}", blocks.len())) {
            blocks.push(self.block(&format!("{kind}.{}", blocks.len()))?);
        }
        if blocks.is_empty() {
            return Err(format!("missing UNet tensors {}{kind}.0.*", self.prefix));
        }
        Ok(blocks)
    }
}

impl Weights for Loader<'_> {
    fn has(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let shape = self.shape(name)?;
        if shape != [len] {
            return Err(format!("UNet tensor {}{name} has shape {shape:?}, expected [{len}]", self.prefix));
        }
        self.t.load(&format!("{}{name}", self.prefix)).map_err(|e| format!("UNet: {e}"))
    }

    /// Linear weight `[out, in]`, or a 1x1 conv `[out, in, 1, 1]` (same row-major data).
    fn linear(&self, base: &str, bias: bool) -> Result<Linear, String> {
        let name = format!("{base}.weight");
        let shape = self.shape(&name)?;
        let (out_f, in_f) = match shape[..] {
            [o, i] | [o, i, 1, 1] => (o, i),
            _ => return Err(format!("UNet tensor {}{name} has non-linear shape {shape:?}", self.prefix)),
        };
        let b = if bias { Some(self.vec(&format!("{base}.bias"), out_f)?) } else { None };
        Ok(Linear { w: self.wt(&name)?, b, in_f, out_f })
    }
}

/// diffusers `UNet2DConditionModel` name -> LDM `UNetModel` name (as in SD single-file checkpoints).
/// `lpb` is `layers_per_block`; `up_attn(i)` tells whether `up_blocks.i` has attentions (its upsampler
/// is then module 2, else 1). `time_embedding.cond_proj` (LCM guidance embedding) has no LDM
/// counterpart and maps to `time_embed.cond_proj`.
pub fn ldm_unet_name(d: &str, lpb: usize, up_attn: &dyn Fn(usize) -> bool) -> Option<String> {
    let p: Vec<&str> = d.split('.').collect();
    let rest = |i: usize| p.get(i..).map(|r| r.join(".")).filter(|s| !s.is_empty());
    let res = |i: usize| -> Option<String> {
        let m = match *p.get(i)? {
            "norm1" => "in_layers.0",
            "conv1" => "in_layers.2",
            "time_emb_proj" => "emb_layers.1",
            "norm2" => "out_layers.0",
            "conv2" => "out_layers.3",
            "conv_shortcut" => "skip_connection",
            _ => return None,
        };
        Some(format!("{m}.{}", rest(i + 1)?))
    };
    let num = |i: usize| p.get(i)?.parse::<usize>().ok();
    Some(match p[..] {
        ["conv_in", ..] => format!("input_blocks.0.0.{}", rest(1)?),
        ["time_embedding", "linear_1", ..] => format!("time_embed.0.{}", rest(2)?),
        ["time_embedding", "linear_2", ..] => format!("time_embed.2.{}", rest(2)?),
        ["time_embedding", "cond_proj", ..] => format!("time_embed.cond_proj.{}", rest(2)?),
        ["add_embedding", "linear_1", ..] => format!("label_emb.0.0.{}", rest(2)?),
        ["add_embedding", "linear_2", ..] => format!("label_emb.0.2.{}", rest(2)?),
        ["conv_norm_out", ..] => format!("out.0.{}", rest(1)?),
        ["conv_out", ..] => format!("out.2.{}", rest(1)?),
        ["down_blocks", _, kind, ..] => {
            let (i, j) = (num(1)?, num(3)?);
            let n = 1 + i * (lpb + 1) + j;
            match kind {
                "resnets" => format!("input_blocks.{n}.0.{}", res(4)?),
                "attentions" => format!("input_blocks.{n}.1.{}", rest(4)?),
                "downsamplers" if p.get(4) == Some(&"conv") => {
                    format!("input_blocks.{}.0.op.{}", (i + 1) * (lpb + 1), rest(5)?)
                }
                _ => return None,
            }
        }
        ["mid_block", kind, j, ..] => match (kind, j) {
            ("resnets", "0") => format!("middle_block.0.{}", res(3)?),
            ("attentions", "0") => format!("middle_block.1.{}", rest(3)?),
            ("resnets", "1") => format!("middle_block.2.{}", res(3)?),
            _ => return None,
        },
        ["up_blocks", _, kind, ..] => {
            let (i, j) = (num(1)?, num(3)?);
            let n = i * (lpb + 1) + j;
            match kind {
                "resnets" => format!("output_blocks.{n}.0.{}", res(4)?),
                "attentions" => format!("output_blocks.{n}.1.{}", rest(4)?),
                "upsamplers" if p.get(4) == Some(&"conv") => {
                    let m = if up_attn(i) { 2 } else { 1 };
                    format!("output_blocks.{}.{m}.conv.{}", i * (lpb + 1) + lpb, rest(5)?)
                }
                _ => return None,
            }
        }
        _ => return None,
    })
}

/// Aliases every diffusers UNet tensor of `t` under its LDM name (prefix-free). Err on a tensor
/// with no LDM counterpart, so no weight is silently left out.
pub fn alias_diffusers(t: &mut Tensors, layers_per_block: usize) -> Result<(), String> {
    let names: Vec<String> = t.names().cloned().collect();
    let attn_ups: HashSet<usize> = names
        .iter()
        .filter_map(|n| {
            let r = n.strip_prefix("up_blocks.")?;
            let (i, rest) = r.split_once('.')?;
            rest.starts_with("attentions.").then(|| i.parse().ok())?
        })
        .collect();
    for n in names {
        let ldm = ldm_unet_name(&n, layers_per_block, &|i| attn_ups.contains(&i))
            .ok_or_else(|| format!("UNet tensor {n} has no LDM counterpart (unsupported UNet variant)"))?;
        t.alias(ldm, &n, None, false)?;
    }
    Ok(())
}

/// LDM `UNetModel` (SD 1.x / 2.x / SDXL), f32 activations on CPU.
pub struct SdUnet {
    model_channels: usize,
    in_channels: usize,
    context_dim: usize,
    /// Spatial divisor: 2^(number of downsamples).
    downscale: usize,
    time0: Linear,
    time2: Linear,
    /// LCM guidance embedding (`time_cond_proj_dim`): added to the sinusoidal timestep embedding.
    cond_proj: Option<Linear>,
    /// SDXL `add_embedding` (text_time): `[pooled text, time ids]` -> added to the time embedding.
    label_emb: Option<(Linear, Linear)>,
    input_blocks: Vec<Vec<Layer>>,
    middle: Vec<Layer>,
    output_blocks: Vec<Vec<Layer>>,
    out_norm: Norm,
    out_conv: Conv,
    /// Metal program per `(h, w, context_len)`.
    gpu: ByShape,
}

impl SdUnet {
    pub fn from_gguf_reader(reader: &GgufWeightReader) -> Result<Self, String> {
        Self::load(&Tensors::gguf(reader), PREFIX, Heads::Dim(HEAD_DIM))
    }

    /// The UNet under `prefix` (LDM names, e.g. `model.diffusion_model.` or `""` for aliased diffusers).
    pub fn load(t: &Tensors, prefix: &str, heads: Heads) -> Result<Self, String> {
        let names = t.names().filter_map(|n| n.strip_prefix(prefix)).map(str::to_owned).collect();
        let l = Loader { t, prefix, names, heads };

        let input_blocks = l.blocks("input_blocks")?;
        let (model_channels, in_channels) = match &input_blocks[0][..] {
            [Layer::ConvIn(c)] => (c.c_out, c.c_in),
            _ => return Err(format!("UNet {prefix}input_blocks.0 must be a single conv")),
        };
        if in_channels != 4 {
            return Err(format!(
                "UNet takes {in_channels} input channels: an inpainting / instruct-pix2pix UNet (needs a mask or image latents), not text-to-image"
            ));
        }
        let time0 = l.linear("time_embed.0", true)?;
        if time0.in_f != model_channels {
            return Err(format!(
                "UNet {prefix}time_embed.0.weight input {} != model channels {model_channels}",
                time0.in_f
            ));
        }
        let time2 = l.linear("time_embed.2", true)?;
        let cond_proj = match l.has("time_embed.cond_proj.weight") {
            true => Some(l.linear("time_embed.cond_proj", false)?),
            false => None,
        };
        if cond_proj.as_ref().is_some_and(|c| c.out_f != model_channels) {
            return Err(format!("UNet {prefix}time_embed.cond_proj does not output {model_channels} channels"));
        }
        let label_emb = if l.has("label_emb.0.0.weight") {
            Some((l.linear("label_emb.0.0", true)?, l.linear("label_emb.0.2", true)?))
        } else {
            None
        };
        let middle = l.block("middle_block")?;
        let output_blocks = l.blocks("output_blocks")?;
        let skips: usize = input_blocks.len();
        if output_blocks.len() != skips {
            return Err(format!("UNet has {skips} input blocks but {} output blocks", output_blocks.len()));
        }

        let mut context_dim = None;
        let layers = input_blocks.iter().chain(std::iter::once(&middle)).chain(&output_blocks).flatten();
        for layer in layers {
            if let Layer::Transformer(t) = layer {
                for b in &t.blocks {
                    let (k, v) = (b.attn2.k.in_f, b.attn2.v.in_f);
                    if k != v || context_dim.is_some_and(|d| d != k) {
                        return Err(format!("UNet cross-attention context dims disagree ({context_dim:?}, {k}, {v})"));
                    }
                    context_dim = Some(k);
                }
            }
        }
        let context_dim = context_dim.ok_or("UNet has no SpatialTransformer (attn2) tensors")?;
        let downs = input_blocks.iter().flatten().filter(|l| matches!(l, Layer::Down(_))).count();

        let out_conv = l.conv("out.2")?;
        Ok(Self {
            model_channels,
            in_channels,
            context_dim,
            downscale: 1 << downs,
            time0,
            time2,
            cond_proj,
            label_emb,
            out_norm: l.norm("out.0", out_conv.c_in)?,
            out_conv,
            input_blocks,
            middle,
            output_blocks,
            gpu: ByShape::default(),
        })
    }

    pub fn context_dim(&self) -> usize {
        self.context_dim
    }

    /// Input width of the SDXL `add_embedding` (`None` without one).
    pub fn add_embed_dim(&self) -> Option<usize> {
        self.label_emb.as_ref().map(|(l, _)| l.in_f)
    }

    /// Input width of the LCM guidance embedding projection (`None` without one).
    pub fn cond_dim(&self) -> Option<usize> {
        self.cond_proj.as_ref().map(|c| c.in_f)
    }

    /// Spatial divisor of the latent: 2^(number of downsamples).
    pub fn downscale(&self) -> usize {
        self.downscale
    }

    /// Native CPU forward (see [`Self::forward`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cpu(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        context: &[f32],
        context_len: usize,
        add: Option<&[f32]>,
        t_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        crate::hardware::cpu_compute::enter("unet::forward");
        self.check(x, h, w, context, context_len)?;

        let mut t_emb = timestep_embedding(timestep, self.model_channels);
        match (&self.cond_proj, t_cond) {
            (Some(p), Some(c)) if c.len() == p.in_f => add_into(&mut t_emb, &p.run(c, 1)),
            (None, None) => {}
            _ => return Err("UNet guidance embedding (time_cond_proj) missing or mis-sized".into()),
        }
        let mut emb = self.time0.run(&t_emb, 1);
        nn::silu(&mut emb);
        let mut emb = self.time2.run(&emb, 1);
        match (&self.label_emb, add) {
            (Some((l0, l2)), Some(a)) if a.len() == l0.in_f => {
                let mut e = l0.run(a, 1);
                nn::silu(&mut e);
                add_into(&mut emb, &l2.run(&e, 1));
            }
            (None, None) => {}
            _ => return Err("UNet added conditioning (SDXL text_time) missing or mis-sized".into()),
        }
        // Every ResBlock's emb_layers starts with SiLU; apply it once.
        nn::silu(&mut emb);

        let mut act = Act { data: x.to_vec(), c: self.in_channels, h, w };
        let mut skips = Vec::with_capacity(self.input_blocks.len());
        for block in &self.input_blocks {
            for layer in block {
                act = layer.forward(act, &emb, context, context_len)?;
            }
            skips.push(Act { data: act.data.clone(), ..act });
        }
        for layer in &self.middle {
            act = layer.forward(act, &emb, context, context_len)?;
        }
        for block in &self.output_blocks {
            let skip = skips.pop().ok_or("UNet skip stack underflow")?;
            if (skip.h, skip.w) != (act.h, act.w) {
                return Err(format!("UNet skip {}x{} != activation {}x{}", skip.h, skip.w, act.h, act.w));
            }
            act.data.extend_from_slice(&skip.data);
            act.c += skip.c;
            for layer in block {
                act = layer.forward(act, &emb, context, context_len)?;
            }
        }

        if act.c != self.out_conv.c_in {
            return Err(format!("UNet out conv expects {} channels, got {}", self.out_conv.c_in, act.c));
        }
        self.out_norm.group(&mut act.data, act.c, h * w, 1e-5);
        nn::silu(&mut act.data);
        let (eps, _, _) = self.out_conv.run(&act.data, h, w, 1)?;
        Ok(eps)
    }

    fn check(&self, x: &[f32], h: usize, w: usize, context: &[f32], context_len: usize) -> Result<(), String> {
        let d = self.downscale;
        if h == 0 || w == 0 || h % d != 0 || w % d != 0 {
            return Err(format!("UNet latent size {h}x{w} must be non-zero multiples of {d}"));
        }
        if x.len() != self.in_channels * h * w {
            return Err(format!("UNet input has {} values, expected {}x{h}x{w}", x.len(), self.in_channels));
        }
        if context_len == 0 || context.len() != context_len * self.context_dim {
            return Err(format!(
                "UNet context has {} values, expected {context_len}x{}",
                context.len(),
                self.context_dim
            ));
        }
        Ok(())
    }

    /// The Metal program for latents `h x w` and `ctx_len` context tokens. Inputs, in order: `x [4,h,w]`,
    /// `t_emb [1, model_channels]` (host sinusoid), `context`, then `add` (SDXL) and `t_cond` (LCM) when
    /// the checkpoint has them. The host arithmetic is the O(model_channels) sinusoid only.
    pub fn build_gpu(&self, h: usize, w: usize, ctx_len: usize) -> Result<GpuNet, String> {
        let mut b = Gb::new();
        let x = b.input("x", &[self.in_channels, h, w]);
        let t = b.input("t_emb", &[1, self.model_channels]);
        let ctx = b.input("context", &[ctx_len, self.context_dim]);
        let add = self.label_emb.as_ref().map(|(l0, _)| b.input("add", &[1, l0.in_f]));
        let t_cond = self.cond_proj.as_ref().map(|p| b.input("t_cond", &[1, p.in_f]));
        let mut t = t;
        if let (Some(p), Some(c)) = (&self.cond_proj, t_cond) {
            let pc = p.emit(&mut b, c);
            t = b.add(t, pc);
        }
        let e = self.time0.emit(&mut b, t);
        let e = b.silu(e);
        let mut e = self.time2.emit(&mut b, e);
        if let (Some((l0, l2)), Some(a)) = (&self.label_emb, add) {
            let l = l0.emit(&mut b, a);
            let l = b.silu(l);
            let l = l2.emit(&mut b, l);
            e = b.add(e, l);
        }
        let emb = b.silu(e);

        let mut act = x;
        let mut skips = Vec::new();
        for block in &self.input_blocks {
            for layer in block {
                act = layer.emit(&mut b, act, emb, ctx);
            }
            skips.push(act);
        }
        for layer in &self.middle {
            act = layer.emit(&mut b, act, emb, ctx);
        }
        for block in &self.output_blocks {
            let skip = skips.pop().ok_or("UNet skip stack underflow")?;
            act = b.concat(&[act, skip], 0);
            for layer in block {
                act = layer.emit(&mut b, act, emb, ctx);
            }
        }
        let t = self.out_norm.emit_group(&mut b, act, 1e-5);
        let t = b.silu(t);
        let eps = self.out_conv.emit(&mut b, t, 1);
        b.output(eps);
        b.compile()
    }

    /// [`Self::forward_cpu`] on Metal; one program per `(h, w, context_len)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        context: &[f32],
        context_len: usize,
        add: Option<&[f32]>,
        t_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        self.check(x, h, w, context, context_len)?;
        if add.map(<[f32]>::len) != self.add_embed_dim() {
            return Err("UNet added conditioning (SDXL text_time) missing or mis-sized".into());
        }
        if t_cond.map(<[f32]>::len) != self.cond_dim() {
            return Err("UNet guidance embedding (time_cond_proj) missing or mis-sized".into());
        }
        let net = self.gpu.get(&[h, w, context_len], || self.build_gpu(h, w, context_len))?;
        let mut inputs = vec![x.to_vec(), timestep_embedding(timestep, self.model_channels), context.to_vec()];
        inputs.extend(add.map(<[f32]>::to_vec));
        inputs.extend(t_cond.map(<[f32]>::to_vec));
        Ok(net.run(inputs)?.remove(0))
    }

    /// Predicts the model output `[4, h, w]` (eps or v) for latents `x [4, h, w]` at `timestep`,
    /// conditioned on `context [context_len, context_dim]`, SDXL `add` (`[pooled, time-id embeddings]`)
    /// and LCM `t_cond` (guidance embedding). A Metal program when the process is GPU-only.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        context: &[f32],
        context_len: usize,
        add: Option<&[f32]>,
        t_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if gpu_mode() {
            self.forward_gpu(x, h, w, timestep, context, context_len, add, t_cond)
        } else {
            self.forward_cpu(x, h, w, timestep, context, context_len, add, t_cond)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_matches_formula() {
        let e = timestep_embedding(999.0, 320);
        assert_eq!(e.len(), 320);
        assert!((e[0] - 999f32.cos()).abs() < 1e-6);
        assert!((e[160] - 999f32.sin()).abs() < 1e-6);
        let f = (-(10000f32.ln()) * 7.0 / 160.0).exp();
        assert!((e[7] - (999.0 * f).cos()).abs() < 1e-5);
        assert!((e[167] - (999.0 * f).sin()).abs() < 1e-5);
    }

    #[test]
    fn geglu_splits_value_then_gate() {
        // rows=2, inner=2: each row is [a0, a1, g0, g1].
        let x = [1.0, 2.0, 0.0, 3.0, -1.0, 0.5, -2.0, 1.0];
        let out = geglu(&x, 2, 2);
        let gelu = |v: f32| {
            let mut s = [v];
            nn::gelu(&mut s);
            s[0]
        };
        let want = [1.0 * gelu(0.0), 2.0 * gelu(3.0), -1.0 * gelu(-2.0), 0.5 * gelu(1.0)];
        for (o, w) in out.iter().zip(want) {
            assert!((o - w).abs() < 1e-6, "{o} vs {w}");
        }
        assert_eq!(out[0], 0.0); // gelu(0) = 0, proves gate (not value) went through gelu
        assert!((out[1] - 2.0 * 3.0 * 0.998_650_1).abs() < 1e-4);
    }

    /// The SD single-file layout (diffusers' convert_diffusers_to_original_stable_diffusion.py).
    #[test]
    fn diffusers_names_map_to_ldm() {
        let up_attn = |i: usize| i > 0; // SD 1.x/2.x: up_blocks.0 is UpBlock2D
        let m = |n: &str| ldm_unet_name(n, 2, &up_attn);
        let cases = [
            ("conv_in.weight", "input_blocks.0.0.weight"),
            ("time_embedding.linear_2.bias", "time_embed.2.bias"),
            ("down_blocks.0.resnets.1.conv1.weight", "input_blocks.2.0.in_layers.2.weight"),
            ("down_blocks.1.resnets.0.conv_shortcut.weight", "input_blocks.4.0.skip_connection.weight"),
            ("down_blocks.1.attentions.1.proj_in.weight", "input_blocks.5.1.proj_in.weight"),
            ("down_blocks.2.downsamplers.0.conv.weight", "input_blocks.9.0.op.weight"),
            ("mid_block.resnets.1.time_emb_proj.weight", "middle_block.2.emb_layers.1.weight"),
            (
                "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight",
                "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
            ),
            ("up_blocks.0.resnets.2.norm2.bias", "output_blocks.2.0.out_layers.0.bias"),
            ("up_blocks.0.upsamplers.0.conv.weight", "output_blocks.2.1.conv.weight"),
            ("up_blocks.1.upsamplers.0.conv.bias", "output_blocks.5.2.conv.bias"),
            ("up_blocks.3.attentions.2.proj_out.weight", "output_blocks.11.1.proj_out.weight"),
            ("conv_norm_out.weight", "out.0.weight"),
            ("add_embedding.linear_1.weight", "label_emb.0.0.weight"),
        ];
        for (d, want) in cases {
            assert_eq!(m(d).as_deref(), Some(want), "{d}");
        }
        assert_eq!(m("down_blocks.0.attentions.0.unknown"), Some("input_blocks.1.1.unknown".into()));
        assert_eq!(m("safety.weight"), None);
    }

    #[test]
    fn head_rules_from_diffusers_configs() {
        let sd15 = serde_json::json!({"attention_head_dim": 8, "block_out_channels": [320, 640, 1280, 1280]});
        assert_eq!(Heads::from_diffusers(&sd15).unwrap(), Heads::Count(8));
        let sdxl = serde_json::json!({"attention_head_dim": [5, 10, 20], "block_out_channels": [320, 640, 1280]});
        assert_eq!(Heads::from_diffusers(&sdxl).unwrap(), Heads::Dim(64));
        assert_eq!(Heads::Count(8).head_dim(640).unwrap(), 80);
    }

    type Specs = Vec<(String, Vec<usize>)>;

    fn lin(s: &mut Specs, p: &str, i: usize, o: usize, bias: bool) {
        s.push((format!("{p}.weight"), vec![o, i]));
        if bias {
            s.push((format!("{p}.bias"), vec![o]));
        }
    }
    fn conv(s: &mut Specs, p: &str, i: usize, o: usize, k: usize) {
        s.push((format!("{p}.weight"), vec![o, i, k, k]));
        s.push((format!("{p}.bias"), vec![o]));
    }
    fn nrm(s: &mut Specs, p: &str, c: usize) {
        s.push((format!("{p}.weight"), vec![c]));
        s.push((format!("{p}.bias"), vec![c]));
    }
    pub(crate) fn res(s: &mut Specs, p: &str, i: usize, o: usize, emb: usize) {
        nrm(s, &format!("{p}.in_layers.0"), i);
        conv(s, &format!("{p}.in_layers.2"), i, o, 3);
        lin(s, &format!("{p}.emb_layers.1"), emb, o, true);
        nrm(s, &format!("{p}.out_layers.0"), o);
        conv(s, &format!("{p}.out_layers.3"), o, o, 3);
        if i != o {
            conv(s, &format!("{p}.skip_connection"), i, o, 1);
        }
    }
    pub(crate) fn trans(s: &mut Specs, p: &str, c: usize, ctx: usize) {
        nrm(s, &format!("{p}.norm"), c);
        lin(s, &format!("{p}.proj_in"), c, c, true);
        let b = format!("{p}.transformer_blocks.0");
        for n in ["norm1", "norm2", "norm3"] {
            nrm(s, &format!("{b}.{n}"), c);
        }
        for (a, kv) in [("attn1", c), ("attn2", ctx)] {
            lin(s, &format!("{b}.{a}.to_q"), c, c, false);
            lin(s, &format!("{b}.{a}.to_k"), kv, c, false);
            lin(s, &format!("{b}.{a}.to_v"), kv, c, false);
            lin(s, &format!("{b}.{a}.to_out.0"), c, c, true);
        }
        lin(s, &format!("{b}.ff.net.0.proj"), c, 2 * 2 * c, true);
        lin(s, &format!("{b}.ff.net.2"), 2 * c, c, true);
        lin(s, &format!("{p}.proj_out"), c, c, true);
    }

    /// A 2-level LDM UNet (SD layout, 1 ResBlock per level) with SDXL `label_emb` and LCM `cond_proj`.
    fn tiny_specs(ctx: usize, add: usize, cond: usize) -> Specs {
        let (mc, emb) = (32, 128);
        let mut s = Specs::new();
        conv(&mut s, "input_blocks.0.0", 4, mc, 3);
        res(&mut s, "input_blocks.1.0", mc, mc, emb);
        trans(&mut s, "input_blocks.1.1", mc, ctx);
        conv(&mut s, "input_blocks.2.0.op", mc, mc, 3);
        res(&mut s, "input_blocks.3.0", mc, 64, emb);
        trans(&mut s, "input_blocks.3.1", 64, ctx);
        res(&mut s, "middle_block.0", 64, 64, emb);
        trans(&mut s, "middle_block.1", 64, ctx);
        res(&mut s, "middle_block.2", 64, 64, emb);
        // (skip channels popped, res in, res out, transformer width, upsample)
        let outs = [(64, 128, 64, 64, false), (32, 96, 64, 64, true), (32, 96, 32, 32, false), (32, 64, 32, 32, false)];
        for (i, (_, cin, cout, tc, up)) in outs.into_iter().enumerate() {
            res(&mut s, &format!("output_blocks.{i}.0"), cin, cout, emb);
            trans(&mut s, &format!("output_blocks.{i}.1"), tc, ctx);
            if up {
                conv(&mut s, &format!("output_blocks.{i}.2.conv"), cout, cout, 3);
            }
        }
        nrm(&mut s, "out.0", 32);
        conv(&mut s, "out.2", 32, 4, 3);
        lin(&mut s, "time_embed.0", mc, emb, true);
        lin(&mut s, "time_embed.2", emb, emb, true);
        lin(&mut s, "time_embed.cond_proj", cond, mc, false);
        lin(&mut s, "label_emb.0.0", add, emb, true);
        lin(&mut s, "label_emb.0.2", emb, emb, true);
        s
    }

    /// CPU UNet vs the Metal program: f32 vs f32, 1e-3.
    #[test]
    fn gpu_matches_cpu() {
        use crate::model::diff_graph::testutil::{rel_err, synth, Lcg};
        let (ctx, add, cond, h, w, ctx_len) = (48, 24, 16, 8, 12, 5);
        let r = synth(&tiny_specs(ctx, add, cond), 11);
        let t = Tensors::from_reader(&r).unwrap();
        let unet = SdUnet::load(&t, "", Heads::Dim(16)).unwrap();
        let mut rng = Lcg(2);
        let (x, c) = (rng.vec(4 * h * w, 1.0), rng.vec(ctx_len * ctx, 1.0));
        let (a, tc) = (rng.vec(add, 1.0), rng.vec(cond, 1.0));
        let cpu = unet.forward_cpu(&x, h, w, 301.0, &c, ctx_len, Some(&a), Some(&tc)).unwrap();
        let gpu = unet.forward_gpu(&x, h, w, 301.0, &c, ctx_len, Some(&a), Some(&tc)).unwrap();
        let e = rel_err(&gpu, &cpu);
        eprintln!("unet gpu-vs-cpu rel err {e:.2e}");
        assert!(e < 1e-3, "rel err {e}");
        // A second call at the same size reuses the program; another timestep changes the output.
        let gpu2 = unet.forward_gpu(&x, h, w, 5.0, &c, ctx_len, Some(&a), Some(&tc)).unwrap();
        let cpu2 = unet.forward_cpu(&x, h, w, 5.0, &c, ctx_len, Some(&a), Some(&tc)).unwrap();
        assert!(rel_err(&gpu2, &cpu2) < 1e-3 && rel_err(&gpu2, &gpu) > 1e-3);
        assert_eq!(unet.gpu.get(&[h, w, ctx_len], || Err("rebuilt".into())).unwrap().cpu_node_count(), 0);
    }
}
