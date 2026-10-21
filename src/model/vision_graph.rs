//! Graph-building helpers shared by vision/audio towers and
//! the GPU placement of their output rows into the text model's `inputs_embeds`.

use crate::graph::program::GraphProgram;
use crate::graph::{BinaryKind, DType, Graph, Op, Tensor, UnaryKind, ValueId, Weights};
use crate::hardware::{Device, PlacementMode};
use crate::model::weights::Mat;

/// A graph plus the weights its `weight` values bind to.
pub struct Gb {
    pub g: Graph,
    pub w: Weights,
}

impl Default for Gb {
    fn default() -> Self {
        Self::new()
    }
}

impl Gb {
    pub fn new() -> Self {
        Self {
            g: Graph::new(),
            w: Weights::new(),
        }
    }

    pub fn input(&mut self, name: &str, shape: Vec<usize>, dt: DType) -> ValueId {
        self.g.input(name, shape, dt)
    }

    /// An F32 weight (shared by name: declared once per name only).
    pub fn f32w(&mut self, name: &str, shape: Vec<usize>, data: Vec<f32>) -> ValueId {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        self.w.insert(name, Tensor::F32(data));
        self.g.weight(name, shape, DType::F32)
    }

    /// A borrowed/packed checkpoint matrix `[rows, cols]` (bf16, f16 or a GGUF block format).
    pub fn mat(&mut self, name: &str, m: &Mat) -> ValueId {
        let dt = match m.packed() {
            Some(p) => match p.format {
                crate::hardware::QuantFormat::FP32 => DType::F32,
                crate::hardware::QuantFormat::FP16 => DType::F16,
                crate::hardware::QuantFormat::BF16 => DType::BF16,
                f => DType::Quant(f),
            },
            None => DType::BF16,
        };
        self.w.insert(name, m.to_graph_tensor());
        self.g.weight(name, vec![m.rows, m.cols], dt)
    }

    /// An F32 node.
    pub fn op(&mut self, name: &str, op: Op, ins: &[ValueId], shape: Vec<usize>) -> ValueId {
        self.g.add(name, op, ins, shape, DType::F32)
    }

    pub fn reshape(&mut self, name: &str, x: ValueId, shape: Vec<usize>) -> ValueId {
        self.op(name, Op::Reshape { shape: Some(shape.clone()) }, &[x], shape)
    }

    /// Permutes `x` (declared shape `from`) by `perm`.
    pub fn transpose(&mut self, name: &str, x: ValueId, from: &[usize], perm: &[usize]) -> ValueId {
        let shape = perm.iter().map(|&p| from[p]).collect();
        self.op(name, Op::Transpose { perm: perm.to_vec() }, &[x], shape)
    }

    pub fn unary(&mut self, name: &str, kind: UnaryKind, x: ValueId, shape: Vec<usize>) -> ValueId {
        self.op(name, Op::Unary { kind }, &[x], shape)
    }

    pub fn binary(&mut self, name: &str, kind: BinaryKind, a: ValueId, b: ValueId, shape: Vec<usize>) -> ValueId {
        self.op(name, Op::Binary { kind }, &[a, b], shape)
    }

    pub fn add(&mut self, name: &str, a: ValueId, b: ValueId, shape: Vec<usize>) -> ValueId {
        self.op(name, Op::Binary { kind: BinaryKind::Add }, &[a, b], shape)
    }

    /// `x [rows, in]` · `w [out, in]`ᵀ (+ `bias`).
    pub fn linear(&mut self, name: &str, x: ValueId, w: ValueId, bias: Option<ValueId>, rows: usize, out: usize) -> ValueId {
        let mut ins = vec![x, w];
        ins.extend(bias);
        self.op(name, Op::Linear, &ins, vec![rows, out])
    }

    /// RMSNorm over the last `gamma.len()` elements; `None` = unit gain over `width`.
    pub fn rms(&mut self, name: &str, x: ValueId, gamma: Vec<f32>, eps: f32, shape: Vec<usize>) -> ValueId {
        let n = gamma.len();
        let g = self.f32w(&format!("{name}.weight"), vec![n], gamma);
        self.op(name, Op::RmsNorm { eps }, &[x, g], shape)
    }

    pub fn finish(mut self, outputs: &[ValueId]) -> (Graph, Weights) {
        for &o in outputs {
            self.g.output(o);
        }
        (self.g, self.w)
    }
}

/// Compiles `graph` for `mode` and returns the program.
pub fn compile(graph: Graph, weights: Weights, mode: PlacementMode) -> Result<GraphProgram, String> {
    GraphProgram::compile(graph, weights, mode)
}

/// `inputs_embeds` placement on the GPU: `base [n, h]` (the prompt's text rows) with `rows [m, h]`
/// scattered at the row indices `pos [m]` (`Scatter{axis: 0}`), run as a one-node program.
pub fn scatter_rows(base: Vec<f32>, n: usize, h: usize, pos: &[u32], rows: Vec<f32>, mode: PlacementMode) -> Result<Vec<f32>, String> {
    let m = pos.len();
    if m == 0 {
        return Ok(base);
    }
    debug_assert!(base.len() == n * h && rows.len() == m * h);
    let mut b = Gb::new();
    let base_v = b.input("base", vec![n, h], DType::F32);
    let idx = b.input("positions", vec![m], DType::U32);
    let upd = b.input("rows", vec![m, h], DType::F32);
    let out = b.op("inputs_embeds", Op::Scatter { axis: 0 }, &[base_v, idx, upd], vec![n, h]);
    let (g, w) = b.finish(&[out]);
    let mut prog = compile(g, w, mode)?;
    let mut res = prog.run(vec![Tensor::F32(base), Tensor::U32(pos.to_vec()), Tensor::F32(rows)])?;
    res.swap_remove(0).into_f32()
}

/// Whether `mode` resolves to the GPU (a program is built) rather than the explicit CPU opt-in.
pub fn is_gpu(mode: PlacementMode) -> bool {
    crate::graph::avenues::resolved_backend(mode) != Device::Cpu
}

/// Metal programs of one tower, compiled once per input shape (patch grid, frame count).
pub struct ProgramCache {
    pub mode: PlacementMode,
    progs: parking_lot::Mutex<std::collections::HashMap<Vec<usize>, std::sync::Arc<parking_lot::Mutex<GraphProgram>>>>,
}

impl ProgramCache {
    pub fn new(mode: PlacementMode) -> Self {
        Self { mode, progs: Default::default() }
    }

    /// The program for `key`, built by `build` and compiled on first use.
    pub fn get(
        &self,
        key: &[usize],
        build: impl FnOnce() -> Result<(Graph, Weights), String>,
    ) -> Result<std::sync::Arc<parking_lot::Mutex<GraphProgram>>, String> {
        let mut progs = self.progs.lock();
        if let Some(p) = progs.get(key) {
            return Ok(p.clone());
        }
        let (g, w) = build()?;
        let p = std::sync::Arc::new(parking_lot::Mutex::new(compile(g, w, self.mode)?));
        progs.insert(key.to_vec(), p.clone());
        Ok(p)
    }
}
