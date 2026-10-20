//! Gibbs graph IR: one model description shared by every executor (CPU today; Metal and
//! Core ML/ANE per PLAN.md).
//!
//! A [`Graph`] is pure structure in SSA form. Every value is a graph input, a named weight, or the
//! output of a node, and nodes are appended in topological order (a node can only reference values
//! that already exist). Weights are ordinary operands, never op fields, and their data lives
//! outside the graph in [`Weights`], so one graph can be fingerprinted or run.

pub mod avenues;
#[cfg(target_os = "macos")]
pub mod codec;
pub mod codegen;
pub mod coreml;
pub mod egraph;
pub mod cpu;
pub mod cuda;
pub mod level_zero;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod passes;
pub mod pattern;
pub mod placement;
pub mod program;
pub mod rocm;
pub mod search;
pub mod tune;
use crate::hardware::{Device, QuantFormat};
use std::collections::HashMap;

/// Whether `item` is in `GIBBS_DISABLE` (comma list, read once): per-item kill switch next to
/// `metal::fusion_policy` (here because the e-graph and loaders build without Metal). Items:
/// `deltanet` and loader items (README). Tests set [`TEST_DISABLE`] per thread.
pub(crate) fn disabled(item: &str) -> bool {
    #[cfg(test)]
    let list = TEST_DISABLE.with(|d| d.get());
    #[cfg(not(test))]
    let list = {
        static D: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| std::env::var("GIBBS_DISABLE").unwrap_or_default());
        D.as_str()
    };
    list.split(',').any(|s| s.trim() == item)
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_DISABLE: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Element type of a value. Quantized weights carry their block format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
    Quant(QuantFormat),
}

/// Physical layout of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Row-major in `shape` order; CPU images `[c, h*w]` are NCHW.
    #[default]
    RowMajor,
    /// Channels-last images (Metal `convolution2d`).
    Nhwc,
    /// `[1, C, 1, S]` (Core ML on the Neural Engine).
    Ane,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueKind {
    Input,
    /// Bound by name to [`Weights`] at execution.
    Weight,
    /// Persistent per-sequence buffer (e.g. KV cache slots, Gated DeltaNet recurrent state, causal conv state).
    State(String),
    Node(NodeId),
}
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: String,
    /// Empty when unknown (e.g. per-node ONNX lowering).
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub layout: Layout,
    pub kind: ValueKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairLayout {
    Adjacent,
    Half,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeVariant {
    /// Rotates adjacent pairs `(2i, 2i+1)`.
    Interleaved,
    /// Rotates `(i, i + rotary_dim/2)` (GPT-NeoX / Qwen layout).
    HalfSplit,
    /// Multimodal RoPE with temporal, height, width sections.
    MRope {
        sections: [u32; 3],
        interleaved: bool,
        pairs: PairLayout,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryKind {
    Exp,
    Sigmoid,
    Tanh,
    Softplus,
    Neg,
    Sqrt,
    Rsqrt,
    Abs,
    Relu,
    Relu2,
    Silu,
    GeluErf,
    GeluTanh,
    Scale(f32),
    Clamp(f32, f32),
    SoftCap(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsampleMode {
    Nearest,
    Bilinear,
    Bicubic,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MegakernelKind {
    /// Fused MLP: RmsNorm -> Linear(Gate) & Linear(Up) (or Linear(GateUp)) -> SwiGLU / GeGLU -> Linear(Down) -> Add
    Mlp {
        hidden_dim: usize,
        intermediate_dim: usize,
        eps: f32,
        is_geglu: bool,
        has_post_norm: bool,
    },
    /// Fused Attention: RmsNorm -> Linear(Q, K, V) -> Attention / RoPE / DeltaRule -> Linear(Out) -> Add
    Attention {
        hidden_dim: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
        is_delta: bool,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MegakernelDescriptor {
    pub kind: MegakernelKind,
    pub name: String,
}

/// Operations. Operand order is the node's `inputs` order, documented per variant.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `[table [rows, d], ids [n] (U32)]` → `[n, d]`.
    Embedding,
    /// `[x [n, in], w [out, in], bias? [out]]` → `[n, out]`.
    Linear,
    /// `[a, b]` → `op(a)·op(b)` for activation × activation products.
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
    },
    /// `[x [n, d], gamma [d], beta? [d]]`, over the last dim.
    LayerNorm {
        eps: f32,
    },
    /// `[x [n, d], gamma [d]]`.
    RmsNorm {
        eps: f32,
    },
    /// `[x [c, hw], gamma [c], beta [c]]`.
    GroupNorm {
        groups: usize,
        eps: f32,
    },
    /// `[x, gamma, beta]`, per instance.
    InstanceNorm {
        eps: f32,
    },
    /// `[x [n, heads*head_dim], positions [n] (U32)]`; rotates the first `rotary_dim` of each head.
    /// `[x [n, heads*head_dim], positions, inv_freq [rotary_dim/2] F32 weight]`.
    Rope {
        head_dim: usize,
        rotary_dim: usize,
        variant: RopeVariant,
        mscale: f32,
    },
    /// `[q [nq, heads*head_dim], k [nk, kv_heads*head_dim], v [nk, kv_heads*head_dim], bias?]` →
    /// `[nq, heads*head_dim]`, with `head_dim` taken from `q`'s shape; query head `h` reads kv head
    /// `h / (heads / kv_heads)`. Cross-attention is this op with `k`/`v` from another sequence.
    /// Scores are `q·k / sqrt(head_dim)`, then `soft_cap * tanh(s / soft_cap)`, then the optional
    /// additive `bias` (`[heads, nq, nk]`, `[nq, nk]` or `[nk]`; `-inf` masks, e.g. padding).
    /// `causal`: query `p` sees keys `k <= p` (top-left aligned when `nq != nk`). With `window = Some(w)`:
    /// causal sees `p - w < k <= p` (w keys), non-causal sees `|p - k| <= w / 2`. A query with no
    /// visible key outputs zeros.
    Attention {
        heads: usize,
        kv_heads: usize,
        causal: bool,
        window: Option<usize>,
        soft_cap: Option<f32>,
    },
    /// `[gate, up]` → `silu(gate) * up`.
    SwiGlu,
    /// `[gate, up]` → `gelu_pytorch_tanh(gate) * up`.
    GeGlu,
    /// `[x]` element-wise unary transformation.
    Unary {
        kind: UnaryKind,
    },
    /// `[a, b]` element-wise binary transformation with broadcasting.
    Binary {
        kind: BinaryKind,
    },
    /// `[q, k_new, v_new, k_state, v_state, positions, visible_end?]` evaluating causal attention and appending
    /// to state. The optional `visible_end` (`[m]` U32, each >= its row's position + 1) lets row `i` see keys
    /// `< visible_end[i]`; the sliding window still starts at the row's own position.
    CachedAttention {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        scale: f32,
        causal: bool,
        window: Option<usize>,
        soft_cap: Option<f32>,
        write: bool,
    },
    /// `[q, k_pool, v_pool, block_table, positions]` evaluating causal attention over paged KV memory.
    PagedCachedAttention {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        scale: f32,
        causal: bool,
        window: Option<usize>,
        soft_cap: Option<f32>,
        tokens_per_block: usize,
        max_blocks_per_seq: usize,
    },
    /// `[x, weight, bias?, conv_state]` 1D causal convolution updating state.
    CausalConv1dCached {
        kernel: usize,
        dilation: usize,
    },
    /// `[q, k, v, beta, log_decay, ssm_state]` Gated DeltaNet recurrence rule.
    DeltaRule {
        key_heads: usize,
        val_heads: usize,
        key_dim: usize,
        val_dim: usize,
    },
    Gather {
        axis: isize,
    },
    Scatter {
        axis: isize,
    },
    Pad {
        padding: Vec<usize>,
        value: f32,
    },
    AvgPool2d {
        kernel: [usize; 2],
        stride: [usize; 2],
    },
    Upsample {
        scale: usize,
        mode: UpsampleMode,
    },
    Conv3d {
        stride: [usize; 3],
        padding: [usize; 6],
        dilation: [usize; 3],
        groups: usize,
    },
    /// `[x [rows, n] f32]` → `[rows, k]` U32 indices of the `k` largest values per row, sorted by
    /// value descending; equal values keep the lower index first.
    TopK {
        k: usize,
    },
    /// `[x [rows, n] f32, idx [rows, k] U32]` → `[rows, k]` f32, `out[r, j] = x[r, idx[r, j]]`; a
    /// single-row `x` is shared by every row of `idx`.
    GatherCols,
    /// Expert-indexed linear `[x, bank, ids]`: `x` is `[m, in]` (every slot of a row reads the same
    /// row) or `[m * top_k, in]` (row `r` is slot `r`); `bank` is the `[experts * out, in]` stack of
    /// every expert's `[out, in]` weight (a GGUF `*_exps` tensor viewed as rows); `ids` is `[m, top_k]`
    /// U32 expert numbers. Output `[m * top_k, out]`: slot `s` is `x_s · bank[ids[s] * out ..][..out]ᵀ`.
    MoeLinear {
        experts: usize,
        top_k: usize,
    },
    Silu,
    /// Exact (erf) GELU, or the tanh approximation.
    Gelu {
        tanh: bool,
    },
    Relu,
    Softmax {
        axis: isize,
    },
    /// `[table [buckets, heads], q_pos [n] U32, k_pos [m] U32]` → `[heads, n, m]`, the T5 bias added
    /// to attention scores: `table[bucket(k_pos - q_pos)]` with HF `_relative_position_bucket`.
    RelPosBias {
        buckets: usize,
        max_distance: usize,
        bidirectional: bool,
        heads: usize,
    },
    /// `[state, parent [batch] U32]` permuting batch rows in-place.
    StateReorder,
    /// `[a, b]` element-wise.
    Add,
    /// `[a, b]` element-wise.
    Mul,
    /// `[x [c_in, h, w], w [c_out, c_in/groups, kh, kw], bias? [c_out]]`; padding is
    /// `[top, left, bottom, right]`.
    Conv2d {
        stride: [usize; 2],
        padding: [usize; 4],
        dilation: [usize; 2],
        groups: usize,
    },
    /// `[x [c_in, l], w [c_out, c_in/groups, k], bias? [c_out]]`; padding is `[begin, end]`.
    Conv1d {
        stride: usize,
        padding: [usize; 2],
        dilation: usize,
        groups: usize,
    },
    /// `[x]` with a static target shape, or `[x, shape]` when the shape is a runtime tensor (ONNX).
    Reshape {
        shape: Option<Vec<usize>>,
    },
    Transpose {
        perm: Vec<usize>,
    },
    Concat {
        axis: isize,
    },
    Slice {
        axis: isize,
        start: i64,
        end: i64,
    },
    Squeeze {
        axes: Vec<isize>,
    },
    Unsqueeze {
        axes: Vec<isize>,
    },
    Tile {
        reps: Vec<usize>,
    },
    Reduce {
        kind: ReduceKind,
        axes: Vec<isize>,
        keep_dims: bool,
    },
    /// `[x [n, d], mask [n] (U32)]` → mean of the rows whose mask is non-zero → `[d]`.
    MeanPool,
    /// `[x]` → `x / ‖x‖₂`.
    L2Normalize,
    /// Fused transformer block Megakernel single-dispatch operator.
    Megakernel(MegakernelDescriptor),
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Embedding => "Embedding",
            Op::Linear => "Linear",
            Op::MatMul { .. } => "MatMul",
            Op::LayerNorm { .. } => "LayerNorm",
            Op::RmsNorm { .. } => "RmsNorm",
            Op::GroupNorm { .. } => "GroupNorm",
            Op::InstanceNorm { .. } => "InstanceNorm",
            Op::Rope { .. } => "Rope",
            Op::Attention { .. } => "Attention",
            Op::SwiGlu => "SwiGlu",
            Op::TopK { .. } => "TopK",
            Op::GatherCols => "GatherCols",
            Op::MoeLinear { .. } => "MoeLinear",
            Op::Silu => "Silu",
            Op::Gelu { .. } => "Gelu",
            Op::Relu => "Relu",
            Op::Softmax { .. } => "Softmax",
            Op::RelPosBias { .. } => "RelPosBias",
            Op::StateReorder => "StateReorder",
            Op::Add => "Add",
            Op::Mul => "Mul",
            Op::Conv2d { .. } => "Conv2d",
            Op::Conv1d { .. } => "Conv1d",
            Op::Reshape { .. } => "Reshape",
            Op::Transpose { .. } => "Transpose",
            Op::Concat { .. } => "Concat",
            Op::Slice { .. } => "Slice",
            Op::Squeeze { .. } => "Squeeze",
            Op::Unsqueeze { .. } => "Unsqueeze",
            Op::Tile { .. } => "Tile",
            Op::Reduce { .. } => "Reduce",
            Op::MeanPool => "MeanPool",
            Op::L2Normalize => "L2Normalize",
            Op::GeGlu => "GeGlu",
            Op::Unary { .. } => "Unary",
            Op::Binary { .. } => "Binary",
            Op::CachedAttention { .. } => "CachedAttention",
            Op::CausalConv1dCached { .. } => "CausalConv1dCached",
            Op::PagedCachedAttention { .. } => "PagedCachedAttention",
            Op::DeltaRule { .. } => "DeltaRule",
            Op::Gather { .. } => "Gather",
            Op::Scatter { .. } => "Scatter",
            Op::Pad { .. } => "Pad",
            Op::AvgPool2d { .. } => "AvgPool2d",
            Op::Upsample { .. } => "Upsample",
            Op::Conv3d { .. } => "Conv3d",
            Op::Megakernel(_) => "Megakernel",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub name: String,
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    /// Placement chosen by the placement pass; builders leave it on the CPU.
    pub device: Device,
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub values: Vec<Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_value(
        &mut self,
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        kind: ValueKind,
    ) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(Value {
            name,
            shape,
            dtype,
            layout: Layout::RowMajor,
            kind,
        });
        id
    }

    /// Declares a graph input; inputs are bound by declaration order at execution.
    pub fn input(&mut self, name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> ValueId {
        let id = self.push_value(name.into(), shape, dtype, ValueKind::Input);
        self.inputs.push(id);
        id
    }

    /// Declares a weight, bound by `name` to [`Weights`] at execution.
    pub fn weight(&mut self, name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> ValueId {
        self.push_value(name.into(), shape, dtype, ValueKind::Weight)
    }

    /// Declares a persistent state tensor, bound to [`StateStore`] at execution.
    pub fn state(&mut self, name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> ValueId {
        let name = name.into();
        self.push_value(name.clone(), shape, dtype, ValueKind::State(name))
    }

    /// Appends a single-output node on the CPU and returns its output value.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        op: Op,
        inputs: &[ValueId],
        shape: Vec<usize>,
        dtype: DType,
    ) -> ValueId {
        let n = self.values.len() as u32;
        assert!(
            inputs.iter().all(|v| v.0 < n),
            "graph node references a value that does not exist yet"
        );
        let name = name.into();
        let node = NodeId(self.nodes.len() as u32);
        let out = self.push_value(name.clone(), shape, dtype, ValueKind::Node(node));
        self.nodes.push(Node {
            name,
            op,
            inputs: inputs.to_vec(),
            outputs: vec![out],
            device: Device::Cpu,
        });
        out
    }

    /// Marks `v` as a graph output; outputs are returned in marking order.
    pub fn output(&mut self, v: ValueId) {
        self.outputs.push(v);
    }

    pub fn value(&self, v: ValueId) -> &Value {
        &self.values[v.0 as usize]
    }

    pub fn node(&self, n: NodeId) -> &Node {
        &self.nodes[n.0 as usize]
    }

    /// The node that produces `v`, if `v` is a node output.
    pub fn producer(&self, v: ValueId) -> Option<&Node> {
        match &self.value(v).kind {
            ValueKind::Node(n) => Some(self.node(*n)),
            _ => None,
        }
    }

    /// Nodes that read `v` (linear scan).
    pub fn consumers(&self, v: ValueId) -> impl Iterator<Item = &Node> + '_ {
        self.nodes.iter().filter(move |n| n.inputs.contains(&v))
    }
}

/// Tensor data for weights, graph inputs, and graph outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    F32(Vec<f32>),
    U32(Vec<u32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    Mat(crate::model::weights::Mat),
    Quantized(std::sync::Arc<crate::model::layers::QuantizedWeight>),
}

impl Tensor {
    pub fn len(&self) -> usize {
        match self {
            Tensor::F32(v) => v.len(),
            Tensor::U32(v) => v.len(),
            Tensor::F16(v) => v.len(),
            Tensor::BF16(v) => v.len(),
            Tensor::Mat(m) => m.rows * m.cols,
            Tensor::Quantized(q) => q.num_elements(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Result<&[f32], String> {
        match self {
            Tensor::F32(v) => Ok(v.as_slice()),
            Tensor::U32(_) => Err("expected an f32 tensor, got u32".into()),
            Tensor::F16(_) => Err("expected an f32 tensor, got f16".into()),
            Tensor::BF16(_) => Err("expected an f32 tensor, got bf16".into()),
            Tensor::Mat(_) => Err("expected an f32 tensor, got Mat".into()),
            Tensor::Quantized(_) => Err("expected an f32 tensor, got Quantized".into()),
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Tensor::F32(v) => v.clone(),
            Tensor::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Tensor::F16(v) => v.iter().map(|&x| crate::model::f16_to_f32(x)).collect(),
            Tensor::BF16(v) => v.iter().map(|&x| crate::model::bf16_to_f32(x)).collect(),
            Tensor::Mat(m) => m.to_f32(),
            Tensor::Quantized(q) => q.to_f32(),
        }
    }

    pub fn into_f32(self) -> Result<Vec<f32>, String> {
        match self {
            Tensor::F32(v) => Ok(v),
            other => Ok(other.to_f32()),
        }
    }

    pub fn as_u32(&self) -> Result<&[u32], String> {
        match self {
            Tensor::U32(v) => Ok(v.as_slice()),
            _ => Err("expected a u32 tensor".into()),
        }
    }
}

/// Persistent state buffers mapped by name for sequential / autoregressive decoding.
#[derive(Debug, Clone, Default)]
pub struct StateStore {
    pub buffers: HashMap<String, Tensor>,
    #[cfg(target_os = "macos")]
    pub metal_buffers: HashMap<String, (std::sync::Arc<::metal::Buffer>, usize)>,
    /// Names in `metal_buffers` whose elements are f16 (attention KV caches); the rest are f32. The
    /// element count in `metal_buffers` counts elements of either type.
    #[cfg(target_os = "macos")]
    pub metal_f16: std::collections::HashSet<String>,
}

impl StateStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.buffers.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.buffers.get_mut(name)
    }

    pub fn insert(&mut self, name: impl Into<String>, t: Tensor) {
        self.buffers.insert(name.into(), t);
    }

    /// Rewinds to `pos` positions. Full KV states need nothing (attention reads only the rows
    /// below the position it is given, and later writes overwrite the rest); a recurrent state
    /// (Gated DeltaNet `ssm_state`, causal-conv `conv_state`) cannot rewind and errs: `restore`
    /// a [`Self::snapshot`] taken at `pos` instead.
    pub fn truncate(&mut self, pos: usize) -> Result<(), String> {
        let recurrent = |n: &String| n.contains("conv_state") || n.contains("ssm_state");
        let names = self.buffers.keys().chain(self.metal_names());
        match names.into_iter().find(|n| recurrent(n)) {
            Some(n) => Err(format!(
                "cannot rewind recurrent state {n} to position {pos}: restore a snapshot"
            )),
            None => Ok(()),
        }
    }

    #[cfg(target_os = "macos")]
    fn metal_names(&self) -> impl Iterator<Item = &String> {
        self.metal_buffers.keys()
    }

    #[cfg(not(target_os = "macos"))]
    fn metal_names(&self) -> impl Iterator<Item = &String> {
        std::iter::empty()
    }

    /// Device buffer `name` as f32 (f16 buffers are widened).
    #[cfg(target_os = "macos")]
    pub fn read_metal(&self, name: &str) -> Option<Vec<f32>> {
        let (buf, n) = self.metal_buffers.get(name)?;
        // SAFETY: shared-storage buffer of `n` elements of the recorded type; programs run
        // synchronously, so the GPU is idle.
        Some(unsafe {
            if self.metal_f16.contains(name) {
                std::slice::from_raw_parts(buf.contents() as *const u16, *n)
                    .iter()
                    .map(|&h| crate::model::f16_to_f32(h))
                    .collect()
            } else {
                std::slice::from_raw_parts(buf.contents() as *const f32, *n).to_vec()
            }
        })
    }

    /// Copies of the named states (device-resident ones are read back from their buffers).
    pub fn snapshot(&self, names: &[&str]) -> HashMap<String, Tensor> {
        let mut snap = HashMap::new();
        for &name in names {
            #[cfg(target_os = "macos")]
            if let Some(v) = self.read_metal(name) {
                snap.insert(name.to_string(), Tensor::F32(v));
                continue;
            }
            if let Some(t) = self.buffers.get(name) {
                snap.insert(name.to_string(), t.clone());
            }
        }
        snap
    }

    /// Writes `snapshot` back: into the device buffer of a device-resident state.
    pub fn restore(&mut self, snapshot: HashMap<String, Tensor>) {
        for (name, t) in snapshot {
            #[cfg(target_os = "macos")]
            if let Some((buf, n)) = self.metal_buffers.get(&name) {
                if let Tensor::F32(v) = &t {
                    // The device buffer may have grown past the snapshot (states grow with the context).
                    assert!(
                        v.len() <= *n,
                        "snapshot of state {name} is larger than its buffer"
                    );
                    // SAFETY: as in `read_metal`; `v` has v.len() <= n elements.
                    unsafe {
                        if self.metal_f16.contains(&name) {
                            let dst = std::slice::from_raw_parts_mut(buf.contents() as *mut u16, v.len());
                            for (d, &x) in dst.iter_mut().zip(v) {
                                *d = crate::model::f32_to_f16(x.clamp(-65504.0, 65504.0));
                            }
                        } else {
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr(),
                                buf.contents() as *mut f32,
                                v.len(),
                            )
                        }
                    };
                    self.buffers.remove(&name);
                    continue;
                }
            }
            self.buffers.insert(name, t);
        }
    }

    #[cfg(target_os = "macos")]
    pub fn sync_metal_to_cpu(&mut self) {
        let synced: Vec<_> = self
            .metal_buffers
            .keys()
            .filter_map(|name| Some((name.clone(), self.read_metal(name)?)))
            .collect();
        for (name, v) in synced {
            self.buffers.insert(name, Tensor::F32(v));
        }
    }
}

/// Weight data by name, owned outside the graph.
#[derive(Debug, Clone, Default)]
pub struct Weights(HashMap<String, Tensor>);

impl Weights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, data: Tensor) {
        self.0.insert(name.into(), data);
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.0.get(name)
    }
    pub fn contains_key(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        self.0.remove(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Total element count across all weights.
    pub fn parameter_count(&self) -> usize {
        self.0.values().map(Tensor::len).sum()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    /// Full KV rewinds by position; a recurrent state refuses and comes back from a snapshot; a
    /// device-resident state is read from and written back into its buffer.
    #[test]
    fn truncate_snapshot_restore() {
        let mut st = StateStore::new();
        st.insert("layers.3.k_cache", Tensor::F32(vec![1.0; 8]));
        assert!(st.truncate(2).is_ok());
        st.insert("layers.0.ssm_state", Tensor::F32(vec![1.0, 2.0]));
        assert!(st.truncate(2).unwrap_err().contains("ssm_state"));
        let snap = st.snapshot(&["layers.0.ssm_state", "missing"]);
        assert_eq!(snap.len(), 1);
        st.insert("layers.0.ssm_state", Tensor::F32(vec![9.0, 9.0]));
        st.restore(snap);
        assert_eq!(
            st.get("layers.0.ssm_state").unwrap().as_f32().unwrap(),
            &[1.0, 2.0]
        );
        #[cfg(target_os = "macos")]
        if let Some(dev) = ::metal::Device::system_default() {
            let buf = dev.new_buffer(8, ::metal::MTLResourceOptions::StorageModeShared);
            // SAFETY: shared 8-byte buffer = 2 f32.
            unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, 2) }
                .copy_from_slice(&[5.0, 6.0]);
            st.metal_buffers
                .insert("layers.1.conv_state".into(), (std::sync::Arc::new(buf), 2));
            let snap = st.snapshot(&["layers.1.conv_state"]);
            let live = st.metal_buffers["layers.1.conv_state"].0.contents() as *mut f32;
            // SAFETY: as above.
            unsafe { live.write(0.0) };
            st.restore(snap);
            assert_eq!(unsafe { *live }, 5.0);
        }
    }
}
