//! Graph-building helpers shared by the Metal programs of the UNet / VAE / CLIP / video networks.
//!
//! Activations are f32; images are `[c, h, w]` (row-major NCHW, unbatched). Weights are copied out of the
//! host structs (which stay valid as the CPU reference), keeping linear weights in their stored f16/bf16
//! precision and widening conv/norm weights to f32 (the Metal conv/norm kernels are f32).
//! A program is compiled once per input resolution ([`GpuNet`]).

use crate::graph::program::GraphProgram;
use crate::graph::{BinaryKind, DType, Graph, Op, Tensor, UpsampleMode, ValueId, Weights};
use crate::hardware::{Device, PlacementMode};
use crate::model::nn::Wt;
use std::sync::Mutex;

pub type V = ValueId;

pub struct Gb {
    pub g: Graph,
    pub w: Weights,
    n: usize,
}

impl Default for Gb {
    fn default() -> Self {
        Self::new()
    }
}

impl Gb {
    pub fn new() -> Self {
        Self { g: Graph::new(), w: Weights::new(), n: 0 }
    }

    pub fn uniq(&mut self, base: &str) -> String {
        self.n += 1;
        format!("{base}.{}", self.n)
    }

    pub fn shape(&self, v: V) -> &[usize] {
        &self.g.value(v).shape
    }

    pub fn input(&mut self, name: &str, shape: &[usize]) -> V {
        self.g.input(name, shape.to_vec(), DType::F32)
    }

    pub fn input_u32(&mut self, name: &str, shape: &[usize]) -> V {
        self.g.input(name, shape.to_vec(), DType::U32)
    }

    pub fn output(&mut self, v: V) {
        self.g.output(v);
    }

    /// An f32 weight.
    pub fn wf32(&mut self, base: &str, shape: &[usize], data: Vec<f32>) -> V {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "weight {base} {shape:?}");
        let name = self.uniq(base);
        self.w.insert(name.clone(), Tensor::F32(data));
        self.g.weight(name, shape.to_vec(), DType::F32)
    }

    /// A weight in its stored precision (linear weights only: f16/bf16 stay packed).
    pub fn wt(&mut self, base: &str, shape: &[usize], w: &Wt) -> V {
        assert_eq!(w.len(), shape.iter().product::<usize>(), "weight {base} {shape:?}");
        let name = self.uniq(base);
        let (t, dt) = match w {
            Wt::F32(v) => (Tensor::F32(v.clone()), DType::F32),
            Wt::F16(v) => (Tensor::F16(v.clone()), DType::F16),
            Wt::BF16(v) => (Tensor::BF16(v.clone()), DType::BF16),
        };
        self.w.insert(name.clone(), t);
        self.g.weight(name, shape.to_vec(), dt)
    }

    pub fn op(&mut self, name: &str, op: Op, ins: &[V], shape: &[usize]) -> V {
        let name = self.uniq(name);
        self.g.add(name, op, ins, shape.to_vec(), DType::F32)
    }

    pub fn reshape(&mut self, x: V, shape: &[usize]) -> V {
        self.op("reshape", Op::Reshape { shape: Some(shape.to_vec()) }, &[x], shape)
    }

    /// `x [rows, in_f]` times a PyTorch `[out_f, in_f]` weight (+ bias).
    pub fn linear(&mut self, x: V, w: &Wt, b: Option<&[f32]>, in_f: usize, out_f: usize) -> V {
        let rows = self.shape(x)[0];
        let wv = self.wt("lin.w", &[out_f, in_f], w);
        let mut ins = vec![x, wv];
        if let Some(b) = b {
            ins.push(self.wf32("lin.b", &[out_f], b.to_vec()));
        }
        self.op("linear", Op::Linear, &ins, &[rows, out_f])
    }

    /// `x [c_in, h, w]` -> `[c_out, oh, ow]`; square kernel `k`, symmetric `pad`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        x: V,
        w: &[f32],
        b: Option<&[f32]>,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> V {
        let s = self.shape(x).to_vec();
        let (c_in, h, wd) = (s[0], s[1], s[2]);
        let (oh, ow) = ((h + 2 * pad - k) / stride + 1, (wd + 2 * pad - k) / stride + 1);
        let wv = self.wf32("conv.w", &[c_out, c_in, k, k], w.to_vec());
        let mut ins = vec![x, wv];
        if let Some(b) = b {
            ins.push(self.wf32("conv.b", &[c_out], b.to_vec()));
        }
        self.op(
            "conv2d",
            Op::Conv2d { stride: [stride; 2], padding: [pad; 4], dilation: [1; 2], groups: 1 },
            &ins,
            &[c_out, oh, ow],
        )
    }

    /// Conv2d with explicit `[top, left, bottom, right]` padding (e.g. the VAE's `(0,1,0,1)` downsample).
    pub fn conv2d_pad(
        &mut self,
        x: V,
        w: &[f32],
        b: Option<&[f32]>,
        c_out: usize,
        k: usize,
        stride: usize,
        padding: [usize; 4],
    ) -> V {
        let s = self.shape(x).to_vec();
        let (c_in, h, wd) = (s[0], s[1], s[2]);
        let oh = (h + padding[0] + padding[2] - k) / stride + 1;
        let ow = (wd + padding[1] + padding[3] - k) / stride + 1;
        let wv = self.wf32("conv.w", &[c_out, c_in, k, k], w.to_vec());
        let mut ins = vec![x, wv];
        if let Some(b) = b {
            ins.push(self.wf32("conv.b", &[c_out], b.to_vec()));
        }
        self.op(
            "conv2d",
            Op::Conv2d { stride: [stride; 2], padding, dilation: [1; 2], groups: 1 },
            &ins,
            &[c_out, oh, ow],
        )
    }

    /// GroupNorm over `x [c, ...]` (any rank >= 2 whose first axis is the channels); keeps the shape.
    pub fn group_norm(&mut self, x: V, gamma: &[f32], beta: &[f32], groups: usize, eps: f32) -> V {
        let s = self.shape(x).to_vec();
        let c = s[0];
        let hw: usize = s[1..].iter().product();
        let flat = if s.len() == 2 { x } else { self.reshape(x, &[c, hw]) };
        let (g, b) = (self.wf32("gn.g", &[c], gamma.to_vec()), self.wf32("gn.b", &[c], beta.to_vec()));
        let y = self.op("group_norm", Op::GroupNorm { groups, eps }, &[flat, g, b], &[c, hw]);
        if s.len() == 2 {
            y
        } else {
            self.reshape(y, &s)
        }
    }

    pub fn layer_norm(&mut self, x: V, gamma: &[f32], beta: &[f32], eps: f32) -> V {
        let s = self.shape(x).to_vec();
        let d = *s.last().unwrap();
        let (g, b) = (self.wf32("ln.g", &[d], gamma.to_vec()), self.wf32("ln.b", &[d], beta.to_vec()));
        self.op("layer_norm", Op::LayerNorm { eps }, &[x, g, b], &s)
    }

    pub fn silu(&mut self, x: V) -> V {
        let s = self.shape(x).to_vec();
        self.op("silu", Op::Unary { kind: crate::graph::UnaryKind::Silu }, &[x], &s)
    }

    pub fn gelu(&mut self, x: V, tanh: bool) -> V {
        let s = self.shape(x).to_vec();
        let kind = if tanh { crate::graph::UnaryKind::GeluTanh } else { crate::graph::UnaryKind::GeluErf };
        self.op("gelu", Op::Unary { kind }, &[x], &s)
    }

    pub fn add(&mut self, a: V, b: V) -> V {
        let s = self.shape(a).to_vec();
        self.op("add", Op::Binary { kind: BinaryKind::Add }, &[a, b], &s)
    }

    pub fn mul(&mut self, a: V, b: V) -> V {
        let s = self.shape(a).to_vec();
        self.op("mul", Op::Binary { kind: BinaryKind::Mul }, &[a, b], &s)
    }

    /// `a + b` with `b` broadcast (numpy rules) to `a`'s shape.
    pub fn add_bcast(&mut self, a: V, b: V) -> V {
        let s = self.shape(a).to_vec();
        self.op("addb", Op::Binary { kind: BinaryKind::Add }, &[a, b], &s)
    }

    pub fn transpose(&mut self, x: V, perm: &[usize]) -> V {
        let s = self.shape(x).to_vec();
        let out: Vec<usize> = perm.iter().map(|&p| s[p]).collect();
        self.op("transpose", Op::Transpose { perm: perm.to_vec() }, &[x], &out)
    }

    pub fn concat(&mut self, xs: &[V], axis: usize) -> V {
        let mut s = self.shape(xs[0]).to_vec();
        s[axis] = xs.iter().map(|&x| self.shape(x)[axis]).sum();
        self.op("concat", Op::Concat { axis: axis as isize }, xs, &s)
    }

    pub fn slice(&mut self, x: V, axis: usize, start: usize, end: usize) -> V {
        let mut s = self.shape(x).to_vec();
        s[axis] = end - start;
        self.op("slice", Op::Slice { axis: axis as isize, start: start as i64, end: end as i64 }, &[x], &s)
    }

    /// Nearest `scale`x upsample of the last two (or three, see `Op::Upsample`) axes.
    pub fn upsample(&mut self, x: V, scale: usize, spatial: usize) -> V {
        let mut s = self.shape(x).to_vec();
        let r = s.len();
        for d in &mut s[r - spatial..] {
            *d *= scale;
        }
        self.op("upsample", Op::Upsample { scale, mode: UpsampleMode::Nearest }, &[x], &s)
    }

    /// Multi-head attention over `[nq, heads*hd]` / `[nk, heads*hd]`.
    pub fn attention(&mut self, q: V, k: V, v: V, heads: usize) -> V {
        let s = self.shape(q).to_vec();
        self.op(
            "attn",
            Op::Attention { heads, kv_heads: heads, causal: false, window: None, soft_cap: None },
            &[q, k, v],
            &s,
        )
    }

    pub fn compile(self) -> Result<GpuNet, String> {
        GpuNet::compile(self.g, self.w)
    }
}

/// A compiled Metal program with host-side f32 in/out.
pub struct GpuNet(Mutex<GraphProgram>);

impl GpuNet {
    pub fn compile(g: Graph, w: Weights) -> Result<Self, String> {
        let p = GraphProgram::compile(g, w, PlacementMode::Pin(Device::Metal))?;
        if p.primary_device() != Device::Metal {
            return Err("gpu-only: program did not place on Metal".into());
        }
        Ok(Self(Mutex::new(p)))
    }

    /// Inputs in declaration order (f32), outputs in marking order.
    pub fn run(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
        self.run_t(inputs.into_iter().map(Tensor::F32).collect())
    }

    pub fn run_t(&self, inputs: Vec<Tensor>) -> Result<Vec<Vec<f32>>, String> {
        let outs = self.0.lock().map_err(|_| "GPU program poisoned".to_string())?.run(inputs)?;
        outs.into_iter().map(Tensor::into_f32).collect()
    }

    pub fn cpu_node_count(&self) -> usize {
        self.0.lock().map(|p| p.cpu_node_count()).unwrap_or(usize::MAX)
    }
}

/// Whether networks run as Metal programs: the process resolved to a GPU backend (`main` armed the
/// CPU tripwire). Library use and `--backend cpu` keep the native CPU forward.
pub fn gpu_mode() -> bool {
    crate::hardware::cpu_compute::banned() || FORCE_GPU.with(|f| f.get())
}

thread_local! {
    static FORCE_GPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with Metal dispatch on this thread without arming the process-wide tripwire (pipeline
/// parity tests that compare against a CPU run in the same process).
pub fn with_gpu<T>(f: impl FnOnce() -> T) -> T {
    let prev = FORCE_GPU.with(|c| c.replace(true));
    let out = f();
    FORCE_GPU.with(|c| c.set(prev));
    out
}

/// The program compiled for the last-seen input shape (one program per resolution; a new
/// resolution recompiles). // ponytail: one cached shape; an LRU if servers alternate sizes.
#[derive(Default)]
pub struct ByShape(Mutex<Option<(Vec<usize>, std::sync::Arc<GpuNet>)>>);

impl ByShape {
    pub fn get(
        &self,
        key: &[usize],
        build: impl FnOnce() -> Result<GpuNet, String>,
    ) -> Result<std::sync::Arc<GpuNet>, String> {
        let mut slot = self.0.lock().map_err(|_| "GPU program cache poisoned".to_string())?;
        if let Some((k, net)) = slot.as_ref() {
            if k == key {
                return Ok(net.clone());
            }
        }
        *slot = None;
        let net = std::sync::Arc::new(build()?);
        *slot = Some((key.to_vec(), net.clone()));
        Ok(net)
    }
}

impl Gb {
    /// `x [rows, in_f]` times an f32 `[out_f, in_f]` weight (+ bias).
    pub fn linear_f32(&mut self, x: V, w: &[f32], b: Option<&[f32]>, in_f: usize, out_f: usize) -> V {
        let rows = self.shape(x)[0];
        let wv = self.wf32("lin.w", &[out_f, in_f], w.to_vec());
        let mut ins = vec![x, wv];
        if let Some(b) = b {
            ins.push(self.wf32("lin.b", &[out_f], b.to_vec()));
        }
        self.op("linear", Op::Linear, &ins, &[rows, out_f])
    }

    /// `x + c` for a scalar.
    pub fn add_scalar(&mut self, x: V, c: f32) -> V {
        let k = self.wf32("const", &[1], vec![c]);
        self.add_bcast(x, k)
    }
}

/// Synthetic in-memory checkpoints and comparison helpers for CPU-vs-Metal parity tests.
pub mod testutil {
    use crate::model::loaders::{TensorMetadata, WeightReader};
    use std::collections::HashMap;

    /// `half`: tensors with >= 2 dims are served as F16 (values rounded through f16, packed bytes via
    /// `load_tensor_quantized`) like a real fp16 checkpoint, so `Tensors::load_wt` yields `Wt::F16`.
    pub struct MemReader(pub HashMap<String, (Vec<usize>, Vec<f32>)>, pub bool);

    impl WeightReader for MemReader {
        fn parse_header(&mut self) -> Result<HashMap<String, String>, String> {
            Ok(HashMap::new())
        }
        fn list_tensors(&self) -> Result<Vec<TensorMetadata>, String> {
            Ok(self
                .0
                .iter()
                .map(|(n, (s, _))| TensorMetadata {
                    name: n.clone(),
                    shape: s.clone(),
                    dtype: if self.1 && s.len() >= 2 { "F16" } else { "F32" }.into(),
                    byte_offset: 0,
                    byte_len: 0,
                })
                .collect())
        }
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
            let (s, v) = self.0.get(name).ok_or_else(|| format!("no tensor {name}"))?;
            Ok(if self.1 && s.len() >= 2 {
                v.iter().map(|&x| crate::model::f16_to_f32(crate::model::f32_to_f16(x))).collect()
            } else {
                v.clone()
            })
        }
        fn load_tensor_quantized(&self, name: &str) -> Result<Vec<u8>, String> {
            let (s, v) = self.0.get(name).ok_or_else(|| format!("no tensor {name}"))?;
            if !(self.1 && s.len() >= 2) {
                return Err(format!("no packed tensor {name}"));
            }
            Ok(v.iter().flat_map(|&x| crate::model::f32_to_f16(x).to_le_bytes()).collect())
        }
    }

    /// Deterministic uniform [-1, 1) stream.
    pub struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale).collect()
        }
    }

    /// Fills `(name, shape)` pairs: 1-D `*.weight` are norm gains (~1), `*.bias` are small, other
    /// weights are fan-in scaled so activations stay O(1) through a deep stack.
    pub fn synth(specs: &[(String, Vec<usize>)], seed: u64) -> MemReader {
        let mut r = Lcg(seed);
        let mut m = HashMap::new();
        for (n, s) in specs {
            let len = s.iter().product();
            let v = if s.len() == 1 && n.ends_with("weight") {
                (0..len).map(|_| 1.0 + 0.1 * r.next()).collect()
            } else if n.ends_with("bias") || s.len() == 1 {
                r.vec(len, 0.1)
            } else {
                let fan_in: usize = s[1..].iter().product::<usize>().max(1);
                r.vec(len, (3.0 / fan_in as f32).sqrt())
            };
            m.insert(n.clone(), (s.clone(), v));
        }
        MemReader(m, false)
    }

    /// max |a - b| / max(max |b|, 1e-3).
    pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let m = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        a.iter().zip(b).fold(0f32, |e, (x, y)| e.max((x - y).abs())) / m
    }
}
