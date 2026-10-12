//! GPU-only enforcement: the avenue ladder every node must climb before it may leave the GPU, the
//! per-family gates for loaders that still run on the CPU, and the op list each family needs.
//!
//! `explore_node` tries six avenues in order and stops at the first that really has a code path.
//! A graph with an unlowered node fails to place with a [`GpuReport`] naming every avenue's reason.

use crate::graph::{DType, Op, PairLayout, RopeVariant, BinaryKind, ReduceKind, UnaryKind, UpsampleMode};
use crate::hardware::Device;
use crate::hardware::QuantFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avenue {
    MetalKernel,
    Decompose,
    DtypeAdapt,
    GeneratedKernel,
    MatmulPrimitive,
    CoreMlAne,
}

impl Avenue {
    pub const ALL: [Avenue; 6] = [
        Avenue::MetalKernel,
        Avenue::Decompose,
        Avenue::DtypeAdapt,
        Avenue::GeneratedKernel,
        Avenue::MatmulPrimitive,
        Avenue::CoreMlAne,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Avenue::MetalKernel => "metal-kernel",
            Avenue::Decompose => "decompose",
            Avenue::DtypeAdapt => "dtype-adapt",
            Avenue::GeneratedKernel => "generated-kernel",
            Avenue::MatmulPrimitive => "matmul-primitive",
            Avenue::CoreMlAne => "core-ml-ane",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Lowered(&'static str),
    Rejected(String),
}

#[derive(Clone, Debug)]
pub struct Attempt {
    pub avenue: Avenue,
    pub verdict: Verdict,
}

#[derive(Clone, Debug)]
pub struct NodeReport {
    pub node: String,
    pub op: String,
    pub dtype: DType,
    pub attempts: Vec<Attempt>,
}

impl NodeReport {
    /// The avenue that lowered the node, if any.
    pub fn lowered_by(&self) -> Option<(Avenue, &'static str)> {
        self.attempts.iter().find_map(|a| match a.verdict {
            Verdict::Lowered(how) => Some((a.avenue, how)),
            Verdict::Rejected(_) => None,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct GpuReport {
    pub total_nodes: usize,
    pub unlowered: Vec<NodeReport>,
}

impl GpuReport {
    pub fn error(&self) -> String {
        let mut s = format!(
            "gpu-only: {} of {} graph nodes have no Metal lowering\n",
            self.unlowered.len(),
            self.total_nodes
        );
        for n in &self.unlowered {
            s.push_str(&format!("  {} [{} {:?}]\n", n.node, n.op, n.dtype));
            for a in &n.attempts {
                if let Verdict::Rejected(reason) = &a.verdict {
                    s.push_str(&format!("    ✗ {}: {}\n", a.avenue.name(), reason));
                }
            }
        }
        s.push_str("run with --backend cpu to accept the CPU executor for this model");
        s
    }
}

fn metal_kernel(op: &Op, dtype: DType) -> Verdict {
    if crate::graph::placement::device_supports_op(Device::Metal, op, dtype) {
        Verdict::Lowered("hand-written Metal kernel")
    } else {
        Verdict::Rejected(format!("no Metal kernel for {} at {:?}", op.name(), dtype))
    }
}

fn decompose(op: &Op, dtype: DType) -> Verdict {
    match crate::graph::passes::decompose::rule_for(op, dtype) {
        Some(how) => Verdict::Lowered(how),
        None => Verdict::Rejected(format!("no rewrite rule for {}", op.name())),
    }
}

/// Whether a quantized weight with no Metal codec is dequantized at load (`metal.rs` weight upload).
fn load_time_dequant(op: &Op, dtype: DType) -> bool {
    matches!(op, Op::Linear | Op::Embedding | Op::Megakernel(_)) && matches!(dtype, DType::Quant(_))
}

fn dtype_adapt(op: &Op, dtype: DType) -> Verdict {
    if load_time_dequant(op, dtype) {
        return Verdict::Lowered("load-time dequantize/re-encode per policy");
    }
    if matches!(dtype, DType::F16 | DType::BF16)
        && crate::graph::passes::dtype_adapt::can_cast(op, dtype)
    {
        return Verdict::Lowered("GPU Cast around the F32 kernel");
    }
    Verdict::Rejected(format!(
        "operand dtype {dtype:?} cannot be cast or re-encoded"
    ))
}

fn generated(op: &Op, dtype: DType) -> Verdict {
    if crate::graph::codegen::can_generate(op, dtype) {
        Verdict::Lowered("generated MSL kernel")
    } else {
        Verdict::Rejected(format!(
            "op {} is not covered by the generated elementwise/movement/reduction templates",
            op.name()
        ))
    }
}

/// `MatMul` through a simdgroup_matrix / matmul2d kernel. No such lowering exists yet (G4.1 step 5).
fn matmul_primitive(op: &Op, _dtype: DType) -> Verdict {
    Verdict::Rejected(format!(
        "no simdgroup_matrix/matmul2d lowering for {}",
        op.name()
    ))
}

fn core_ml(op: &Op) -> Verdict {
    if crate::graph::coreml::supports_node(op) {
        Verdict::Lowered("Core ML program")
    } else {
        Verdict::Rejected(format!("no Core ML lowering for {}", op.name()))
    }
}

/// Tries the six avenues in order and stops at the first `Lowered`.
pub fn explore_node(name: &str, op: &Op, dtype: DType) -> NodeReport {
    let mut attempts = Vec::with_capacity(Avenue::ALL.len());
    for avenue in Avenue::ALL {
        let verdict = match avenue {
            Avenue::MetalKernel => metal_kernel(op, dtype),
            Avenue::Decompose => decompose(op, dtype),
            Avenue::DtypeAdapt => dtype_adapt(op, dtype),
            Avenue::GeneratedKernel => generated(op, dtype),
            Avenue::MatmulPrimitive => matmul_primitive(op, dtype),
            Avenue::CoreMlAne => core_ml(op),
        };
        let done = matches!(verdict, Verdict::Lowered(_));
        attempts.push(Attempt { avenue, verdict });
        if done {
            break;
        }
    }
    NodeReport {
        node: name.to_string(),
        op: format!("{op:?}"),
        dtype,
        attempts,
    }
}

// ---------------------------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    GraphLm,
    Gemma4,
    Qwen35,
    Gemma4Media,
    Qwen35Vision,
    DraftersMtp,
    Ninfer,
    T5,
    Seq2Seq,
    Idefics3Siglip,
    ClassifierBert,
    ClassifierModern,
    ClassifierDecoder,
    System1,
    DiffusionUnet,
    DiffusionDit,
    Vae,
    VideoLtx,
    VideoModelScope,
    Audio,
}

impl Family {
    pub const ALL: [Family; 20] = [
        Family::GraphLm,
        Family::Gemma4,
        Family::Qwen35,
        Family::Gemma4Media,
        Family::Qwen35Vision,
        Family::DraftersMtp,
        Family::Ninfer,
        Family::T5,
        Family::Seq2Seq,
        Family::Idefics3Siglip,
        Family::ClassifierBert,
        Family::ClassifierModern,
        Family::ClassifierDecoder,
        Family::System1,
        Family::DiffusionUnet,
        Family::DiffusionDit,
        Family::Vae,
        Family::VideoLtx,
        Family::VideoModelScope,
        Family::Audio,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Gpu,
    Gated(&'static str),
}

/// Observed status. A family becomes `Gpu` only in the commit that makes its fixture test in
/// `tests/cpu_free_matrix.rs` pass.
pub fn family_status(f: Family) -> Status {
    use Family::*;
    match f {
        GraphLm | Gemma4 | Qwen35 | System1 => Status::Gpu,
        Gemma4Media => Status::Gated(
            "builder: vision/audio prefill bypasses the program (host Rust vision tower)",
        ),
        Qwen35Vision => {
            Status::Gated("builder: vision tower and MRope prefill are host Rust, no Metal program")
        }
        DraftersMtp => Status::Gated(
            "builder: drafter/MTP forward is host Rust; no state snapshot/restore on the Metal program",
        ),
        Ninfer => Status::Gated(
            "builder: .ninfer catalog emits Moe with host-side experts; no Metal expert operands",
        ),
        T5 => Status::Gated(
            "builder: no T5 graph builder (encoder-decoder with cross-attention state)",
        ),
        Seq2Seq => Status::Gated("builder: no seq2seq graph builder (beam search, positions)"),
        Idefics3Siglip => Status::Gated("builder: SigLIP forward is host Rust, no graph builder"),
        ClassifierBert => {
            Status::Gated("builder: Head::apply and the BERT classifier forward are host Rust")
        }
        ClassifierModern => {
            Status::Gated("builder: DeBERTa/ModernBERT forward is host Rust, no graph builder")
        }
        ClassifierDecoder => {
            Status::Gated("builder: decoder seq-cls forward is host Rust, no graph builder")
        }
        DiffusionUnet => Status::Gpu,
        DiffusionDit => Status::Gated("builder: DiT forward is host Rust, no graph builder"),
        Vae => Status::Gpu,
        VideoLtx => Status::Gated("builder: LTX forward is host Rust, no graph builder"),
        VideoModelScope => Status::Gpu,
        Audio => Status::Gated("builder: conformer audio forward is host Rust; no Metal program yet"),
    }
}

/// `Ok` when `backend` is the explicit CPU opt-in or the family has a Metal program.
pub fn family_gate(f: Family, backend: Device) -> Result<(), String> {
    if backend == Device::Cpu {
        return Ok(());
    }
    match family_status(f) {
        Status::Gpu => Ok(()),
        Status::Gated(reason) => Err(format!(
            "gpu-only: {f:?} has no Metal program yet ({reason}); run with --backend cpu to accept the CPU executor"
        )),
    }
}

/// `family_gate` against the process backend: the CPU when the tripwire is not armed (library use,
/// `--backend cpu`), Metal when `main` resolved to a GPU backend.
pub fn gate(f: Family) -> Result<(), String> {
    let backend = if crate::hardware::cpu_compute::banned() { Device::Metal } else { Device::Cpu };
    family_gate(f, backend)
}

/// The device a placement mode runs on: the pinned one, or the default GPU when `Auto`.
pub fn resolved_backend(mode: crate::hardware::PlacementMode) -> Device {
    match mode {
        crate::hardware::PlacementMode::Pin(d) => d,
        crate::hardware::PlacementMode::Auto => crate::hardware::default_device(),
    }
}

fn rms() -> Op {
    Op::RmsNorm { eps: 1e-6 }
}
fn ln() -> Op {
    Op::LayerNorm { eps: 1e-5 }
}
fn slice() -> Op {
    Op::Slice {
        axis: 0,
        start: 0,
        end: 1,
    }
}
fn rope(variant: RopeVariant) -> Op {
    Op::Rope {
        head_dim: 16,
        rotary_dim: 16,
        variant,
        mscale: 1.0,
    }
}
fn attn() -> Op {
    Op::Attention {
        heads: 2,
        kv_heads: 2,
        causal: false,
        window: None,
        soft_cap: None,
    }
}
fn cached_attn() -> Op {
    Op::CachedAttention {
        heads: 2,
        kv_heads: 1,
        head_dim: 16,
        v_head_dim: 16,
        scale: 1.0,
        causal: true,
        window: None,
        soft_cap: None,
        write: true,
    }
}
fn conv2d() -> Op {
    Op::Conv2d {
        stride: [1, 1],
        padding: [1, 1, 1, 1],
        dilation: [1, 1],
        groups: 1,
    }
}
fn conv3d() -> Op {
    Op::Conv3d {
        stride: [1, 1, 1],
        padding: [0, 0, 1, 1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
    }
}
fn unary(kind: UnaryKind) -> Op {
    Op::Unary { kind }
}
fn group_norm() -> Op {
    Op::GroupNorm {
        groups: 8,
        eps: 1e-5,
    }
}
fn reshape() -> Op {
    Op::Reshape { shape: None }
}

/// Ops the family's forward pass needs, as `(op, dtype)` samples. These are the ops whose Metal
/// lowering decides the family's status: a `Gpu` family has none unlowered; a `Gated` family has at
/// least one. Linear/Embedding weights are listed at the dtypes the loaders produce.
pub fn required_ops(f: Family) -> Vec<(Op, DType)> {
    use DType::{BF16, F32};
    let q4 = DType::Quant(QuantFormat::GGUF_Q4_0);
    let q8 = DType::Quant(QuantFormat::GGUF_Q8_0);
    let rope_hs = || rope(RopeVariant::HalfSplit);
    let mut ops: Vec<(Op, DType)> = Vec::new();
    // The decoder core shared by GraphLm, Gemma 4 and Qwen 3.5.
    let decoder = |ops: &mut Vec<(Op, DType)>| {
        for dt in [F32, BF16, q4, q8] {
            ops.push((Op::Linear, dt));
        }
        ops.push((Op::Embedding, F32));
        ops.extend([
            (rms(), F32),
            (ln(), F32),
            (slice(), F32),
            (rope_hs(), F32),
            (cached_attn(), F32),
            (Op::SwiGlu, F32),
            (Op::Gelu { tanh: true }, F32),
            (Op::Add, F32),
        ]);
    };
    match f {
        Family::GraphLm => decoder(&mut ops),
        Family::Gemma4 => {
            decoder(&mut ops);
            ops.extend([
                (Op::GeGlu, F32),
                (Op::Mul, F32),
                (unary(UnaryKind::SoftCap(30.0)), F32),
            ]);
        }
        Family::Qwen35 => {
            decoder(&mut ops);
            ops.extend([
                (
                    Op::DeltaRule {
                        key_heads: 2,
                        val_heads: 2,
                        key_dim: 16,
                        val_dim: 16,
                    },
                    F32,
                ),
                (
                    Op::CausalConv1dCached {
                        kernel: 4,
                        dilation: 1,
                    },
                    F32,
                ),
                (Op::Silu, F32),
                (Op::Mul, F32),
                (unary(UnaryKind::Sigmoid), F32),
                (unary(UnaryKind::Softplus), F32),
                (Op::L2Normalize, F32),
            ]);
        }
        Family::System1 => ops.extend([
            (Op::Embedding, F32),
            (Op::Linear, F32),
            (ln(), F32),
            (attn(), F32),
            (Op::Gelu { tanh: false }, F32),
            (Op::Add, F32),
            (Op::Mul, F32),
            (Op::MeanPool, F32),
            (Op::L2Normalize, F32),
        ]),
        Family::Gemma4Media => ops.extend([
            (conv2d(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (reshape(), F32),
            (ln(), F32),
            (Op::Gelu { tanh: true }, F32),
            (
                Op::MatMul {
                    transpose_a: false,
                    transpose_b: true,
                },
                F32,
            ),
        ]),
        Family::Qwen35Vision => ops.extend([
            (conv3d(), F32),
            (reshape(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (
                rope(RopeVariant::MRope {
                    sections: [8, 4, 4],
                    interleaved: true,
                    pairs: PairLayout::Half,
                }),
                F32,
            ),
            (attn(), F32),
            (ln(), F32),
        ]),
        Family::DraftersMtp => ops.extend([
            (Op::Embedding, F32),
            (Op::Concat { axis: -1 }, F32),
            (Op::Linear, F32),
            (rms(), F32),
            (cached_attn(), F32),
        ]),
        Family::Ninfer => ops.extend([
            (Op::Linear, F32),
            (
                Op::Moe {
                    experts: 4,
                    top_k: 2,
                },
                F32,
            ),
            (
                Op::DeltaRule {
                    key_heads: 2,
                    val_heads: 2,
                    key_dim: 16,
                    val_dim: 16,
                },
                F32,
            ),
        ]),
        Family::T5 => ops.extend([
            (Op::Embedding, F32),
            (rms(), F32),
            (Op::Linear, F32),
            (
                Op::RelPosBias {
                    buckets: 32,
                    max_distance: 128,
                    bidirectional: true,
                    heads: 2,
                },
                F32,
            ),
            (attn(), F32),
            (Op::Relu, F32),
            (Op::Add, F32),
        ]),
        Family::Seq2Seq => ops.extend([
            (Op::Embedding, F32),
            (ln(), F32),
            (Op::Linear, F32),
            (attn(), F32),
            (Op::StateReorder, F32),
            (Op::Gather { axis: 0 }, F32),
            (Op::Relu, F32),
            (Op::Add, F32),
        ]),
        Family::Idefics3Siglip => ops.extend([
            (conv2d(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (Op::Scatter { axis: 0 }, F32),
            (ln(), F32),
            (Op::Gelu { tanh: true }, F32),
            (Op::Linear, F32),
        ]),
        Family::ClassifierBert => ops.extend([
            (Op::Embedding, F32),
            (ln(), F32),
            (Op::Linear, F32),
            (attn(), F32),
            (Op::Gelu { tanh: false }, F32),
            (Op::Add, F32),
            (unary(UnaryKind::Tanh), F32),
            (slice(), F32),
        ]),
        Family::ClassifierModern => ops.extend([
            (Op::Embedding, F32),
            (ln(), F32),
            (Op::Linear, F32),
            (attn(), F32),
            (Op::GeGlu, F32),
            (rope_hs(), F32),
            (Op::Gather { axis: 0 }, F32),
            (unary(UnaryKind::Tanh), F32),
        ]),
        Family::ClassifierDecoder => ops.extend([
            (Op::Embedding, F32),
            (rms(), F32),
            (Op::Linear, F32),
            (cached_attn(), F32),
            (Op::SwiGlu, F32),
            (Op::Gather { axis: 0 }, F32),
        ]),
        Family::DiffusionUnet => ops.extend([
            (conv2d(), F32),
            (group_norm(), F32),
            (Op::Silu, F32),
            (
                Op::Upsample {
                    scale: 2,
                    mode: UpsampleMode::Nearest,
                },
                F32,
            ),
            (Op::Concat { axis: 0 }, F32),
            (attn(), F32),
            (Op::Attention { heads: 2, kv_heads: 2, causal: true, window: None, soft_cap: None }, F32),
            (Op::Linear, F32),
            (Op::Linear, DType::F16),
            (Op::Linear, BF16),
            (Op::Add, F32),
            (Op::Mul, F32),
            (Op::Binary { kind: BinaryKind::Add }, F32),
            (Op::Gelu { tanh: false }, F32),
            (unary(UnaryKind::Scale(1.702)), F32),
            (unary(UnaryKind::Sigmoid), F32),
            (ln(), F32),
            (slice(), F32),
            (reshape(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (Op::Embedding, F32),
            (Op::Gather { axis: 0 }, F32),
        ]),
        Family::DiffusionDit => ops.extend([
            (Op::Linear, F32),
            (attn(), F32),
            (ln(), F32),
            (Op::Mul, F32),
            (Op::Add, F32),
            (reshape(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (Op::Gelu { tanh: true }, F32),
            (
                rope(RopeVariant::MRope {
                    sections: [8, 4, 4],
                    interleaved: false,
                    pairs: PairLayout::Half,
                }),
                F32,
            ),
        ]),
        Family::Vae => ops.extend([
            (conv2d(), F32),
            (conv3d(), F32),
            (group_norm(), F32),
            (Op::Silu, F32),
            (
                Op::Upsample {
                    scale: 2,
                    mode: UpsampleMode::Nearest,
                },
                F32,
            ),
            (attn(), F32),
            (Op::Add, F32),
            (Op::Binary { kind: BinaryKind::Add }, F32),
            (Op::Binary { kind: BinaryKind::Div }, F32),
            (Op::Relu, F32),
            (rms(), F32),
            (Op::Gather { axis: 0 }, F32),
            (Op::MatMul { transpose_a: false, transpose_b: true }, F32),
            (Op::Reduce { kind: ReduceKind::Sum, axes: vec![2], keep_dims: true }, F32),
            (unary(UnaryKind::Scale(0.5)), F32),
            (reshape(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (slice(), F32),
            (Op::Concat { axis: 0 }, F32),
        ]),
        Family::VideoLtx => ops.extend([
            (Op::Linear, F32),
            (attn(), F32),
            (conv3d(), F32),
            (rope_hs(), F32),
            (reshape(), F32),
        ]),
        Family::VideoModelScope => ops.extend([
            (conv3d(), F32),
            (group_norm(), F32),
            (attn(), F32),
            (Op::Concat { axis: 0 }, F32),
            (conv2d(), F32),
            (Op::Silu, F32),
            (Op::Linear, F32),
            (Op::Linear, DType::F16),
            (Op::Add, F32),
            (Op::Mul, F32),
            (Op::Binary { kind: BinaryKind::Add }, F32),
            (Op::Gelu { tanh: false }, F32),
            (ln(), F32),
            (slice(), F32),
            (reshape(), F32),
            (Op::Transpose { perm: vec![1, 0] }, F32),
            (
                Op::Upsample {
                    scale: 2,
                    mode: UpsampleMode::Nearest,
                },
                F32,
            ),
        ]),
        Family::Audio => ops.extend([
            (
                Op::Conv1d {
                    stride: 1,
                    padding: [1, 1],
                    dilation: 1,
                    groups: 1,
                },
                F32,
            ),
            (ln(), F32),
            (Op::Silu, F32),
            (Op::Linear, F32),
            (attn(), F32),
            (Op::Mul, F32),
        ]),
    }
    ops
}
