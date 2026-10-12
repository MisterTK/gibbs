//! Every graph op at every GPU activation dtype is lowered by some avenue, or is listed in
//! `KNOWN_UNLOWERED`. The list may only shrink: lowering an entry makes this test demand its removal.

use gibbs_engine::graph::avenues::{explore_node, Verdict};
use gibbs_engine::graph::{
    BinaryKind, DType, MegakernelDescriptor, MegakernelKind, Op, PairLayout, ReduceKind,
    RopeVariant, UnaryKind, UpsampleMode,
};

/// `(op key, dtype, reason)`; the reason is the avenue verdict text of the first rejected avenue.
const KNOWN_UNLOWERED: &[(&str, &str, &str)] = &[
    ("MatMul", "F32", "no Metal kernel for MatMul at F32"),
    ("MatMul", "F16", "no Metal kernel for MatMul at F16"),
    ("MatMul", "BF16", "no Metal kernel for MatMul at BF16"),
    ("GroupNorm", "F32", "no Metal kernel for GroupNorm at F32"),
    ("GroupNorm", "F16", "no Metal kernel for GroupNorm at F16"),
    ("GroupNorm", "BF16", "no Metal kernel for GroupNorm at BF16"),
    ("InstanceNorm", "F32", "no Metal kernel for InstanceNorm at F32"),
    ("InstanceNorm", "F16", "no Metal kernel for InstanceNorm at F16"),
    ("InstanceNorm", "BF16", "no Metal kernel for InstanceNorm at BF16"),
    ("Gather", "F32", "no Metal kernel for Gather at F32"),
    ("Gather", "F16", "no Metal kernel for Gather at F16"),
    ("Gather", "BF16", "no Metal kernel for Gather at BF16"),
    ("Scatter", "F32", "no Metal kernel for Scatter at F32"),
    ("Scatter", "F16", "no Metal kernel for Scatter at F16"),
    ("Scatter", "BF16", "no Metal kernel for Scatter at BF16"),
    ("Pad", "F32", "no Metal kernel for Pad at F32"),
    ("Pad", "F16", "no Metal kernel for Pad at F16"),
    ("Pad", "BF16", "no Metal kernel for Pad at BF16"),
    ("AvgPool2d", "F32", "no Metal kernel for AvgPool2d at F32"),
    ("AvgPool2d", "F16", "no Metal kernel for AvgPool2d at F16"),
    ("AvgPool2d", "BF16", "no Metal kernel for AvgPool2d at BF16"),
    ("Upsample", "F32", "no Metal kernel for Upsample at F32"),
    ("Upsample", "F16", "no Metal kernel for Upsample at F16"),
    ("Upsample", "BF16", "no Metal kernel for Upsample at BF16"),
    ("Conv3d", "F32", "no Metal kernel for Conv3d at F32"),
    ("Conv3d", "F16", "no Metal kernel for Conv3d at F16"),
    ("Conv3d", "BF16", "no Metal kernel for Conv3d at BF16"),
    ("Moe", "F32", "no Metal kernel for Moe at F32"),
    ("Moe", "F16", "no Metal kernel for Moe at F16"),
    ("Moe", "BF16", "no Metal kernel for Moe at BF16"),
    ("Relu", "F32", "no Metal kernel for Relu at F32"),
    ("Relu", "F16", "no Metal kernel for Relu at F16"),
    ("Relu", "BF16", "no Metal kernel for Relu at BF16"),
    ("Conv1d", "F32", "no Metal kernel for Conv1d at F32"),
    ("Conv1d", "F16", "no Metal kernel for Conv1d at F16"),
    ("Conv1d", "BF16", "no Metal kernel for Conv1d at BF16"),
    ("Reshape", "F32", "no Metal kernel for Reshape at F32"),
    ("Reshape", "F16", "no Metal kernel for Reshape at F16"),
    ("Reshape", "BF16", "no Metal kernel for Reshape at BF16"),
    ("Transpose", "F32", "no Metal kernel for Transpose at F32"),
    ("Transpose", "F16", "no Metal kernel for Transpose at F16"),
    ("Transpose", "BF16", "no Metal kernel for Transpose at BF16"),
    ("Concat", "F32", "no Metal kernel for Concat at F32"),
    ("Concat", "F16", "no Metal kernel for Concat at F16"),
    ("Concat", "BF16", "no Metal kernel for Concat at BF16"),
    ("Squeeze", "F32", "no Metal kernel for Squeeze at F32"),
    ("Squeeze", "F16", "no Metal kernel for Squeeze at F16"),
    ("Squeeze", "BF16", "no Metal kernel for Squeeze at BF16"),
    ("Unsqueeze", "F32", "no Metal kernel for Unsqueeze at F32"),
    ("Unsqueeze", "F16", "no Metal kernel for Unsqueeze at F16"),
    ("Unsqueeze", "BF16", "no Metal kernel for Unsqueeze at BF16"),
    ("Tile", "F32", "no Metal kernel for Tile at F32"),
    ("Tile", "F16", "no Metal kernel for Tile at F16"),
    ("Tile", "BF16", "no Metal kernel for Tile at BF16"),
    ("Unary::Exp", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Exp", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Exp", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Tanh", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Tanh", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Tanh", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Neg", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Neg", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Neg", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Sqrt", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Sqrt", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Sqrt", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Rsqrt", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Rsqrt", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Rsqrt", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Abs", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Abs", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Abs", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Relu", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Relu", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Relu", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Relu2", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Relu2", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Relu2", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Silu", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Silu", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Silu", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::GeluErf", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::GeluErf", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::GeluErf", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::GeluTanh", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::GeluTanh", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::GeluTanh", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Scale(0.5)", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Scale(0.5)", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Scale(0.5)", "BF16", "no Metal kernel for Unary at BF16"),
    ("Unary::Clamp(-1.0, 1.0)", "F32", "no Metal kernel for Unary at F32"),
    ("Unary::Clamp(-1.0, 1.0)", "F16", "no Metal kernel for Unary at F16"),
    ("Unary::Clamp(-1.0, 1.0)", "BF16", "no Metal kernel for Unary at BF16"),
    ("Binary::Add", "F32", "no Metal kernel for Binary at F32"),
    ("Binary::Add", "F16", "no Metal kernel for Binary at F16"),
    ("Binary::Add", "BF16", "no Metal kernel for Binary at BF16"),
    ("Binary::Sub", "F32", "no Metal kernel for Binary at F32"),
    ("Binary::Sub", "F16", "no Metal kernel for Binary at F16"),
    ("Binary::Sub", "BF16", "no Metal kernel for Binary at BF16"),
    ("Binary::Mul", "F32", "no Metal kernel for Binary at F32"),
    ("Binary::Mul", "F16", "no Metal kernel for Binary at F16"),
    ("Binary::Mul", "BF16", "no Metal kernel for Binary at BF16"),
    ("Binary::Div", "F32", "no Metal kernel for Binary at F32"),
    ("Binary::Div", "F16", "no Metal kernel for Binary at F16"),
    ("Binary::Div", "BF16", "no Metal kernel for Binary at BF16"),
    ("Binary::Max", "F32", "no Metal kernel for Binary at F32"),
    ("Binary::Max", "F16", "no Metal kernel for Binary at F16"),
    ("Binary::Max", "BF16", "no Metal kernel for Binary at BF16"),
    ("Binary::Min", "F32", "no Metal kernel for Binary at F32"),
    ("Binary::Min", "F16", "no Metal kernel for Binary at F16"),
    ("Binary::Min", "BF16", "no Metal kernel for Binary at BF16"),
    ("Reduce::Sum", "F32", "no Metal kernel for Reduce at F32"),
    ("Reduce::Sum", "F16", "no Metal kernel for Reduce at F16"),
    ("Reduce::Sum", "BF16", "no Metal kernel for Reduce at BF16"),
    ("Reduce::Mean", "F32", "no Metal kernel for Reduce at F32"),
    ("Reduce::Mean", "F16", "no Metal kernel for Reduce at F16"),
    ("Reduce::Mean", "BF16", "no Metal kernel for Reduce at BF16"),
    ("Reduce::Max", "F32", "no Metal kernel for Reduce at F32"),
    ("Reduce::Max", "F16", "no Metal kernel for Reduce at F16"),
    ("Reduce::Max", "BF16", "no Metal kernel for Reduce at BF16"),
    ("Reduce::Min", "F32", "no Metal kernel for Reduce at F32"),
    ("Reduce::Min", "F16", "no Metal kernel for Reduce at F16"),
    ("Reduce::Min", "BF16", "no Metal kernel for Reduce at BF16"),
];

const DTYPES: [(DType, &str); 3] = [(DType::F32, "F32"), (DType::F16, "F16"), (DType::BF16, "BF16")];

/// One index per `Op` variant. The match has no wildcard arm: adding a variant breaks the build here.
fn ordinal(op: &Op) -> usize {
    match op {
        Op::Embedding => 0,
        Op::Linear => 1,
        Op::MatMul { .. } => 2,
        Op::LayerNorm { .. } => 3,
        Op::RmsNorm { .. } => 4,
        Op::GroupNorm { .. } => 5,
        Op::InstanceNorm { .. } => 6,
        Op::Rope { .. } => 7,
        Op::Attention { .. } => 8,
        Op::SwiGlu => 9,
        Op::GeGlu => 10,
        Op::Unary { .. } => 11,
        Op::Binary { .. } => 12,
        Op::CachedAttention { .. } => 13,
        Op::PagedCachedAttention { .. } => 14,
        Op::CausalConv1dCached { .. } => 15,
        Op::DeltaRule { .. } => 16,
        Op::Gather { .. } => 17,
        Op::Scatter { .. } => 18,
        Op::Pad { .. } => 19,
        Op::AvgPool2d { .. } => 20,
        Op::Upsample { .. } => 21,
        Op::Conv3d { .. } => 22,
        Op::Moe { .. } => 23,
        Op::Silu => 24,
        Op::Gelu { .. } => 25,
        Op::Relu => 26,
        Op::Softmax { .. } => 27,
        Op::TopKSoftmax { .. } => 28,
        Op::RelPosBias { .. } => 29,
        Op::StateReorder => 30,
        Op::Add => 31,
        Op::Mul => 32,
        Op::Conv2d { .. } => 33,
        Op::Conv1d { .. } => 34,
        Op::Reshape { .. } => 35,
        Op::Transpose { .. } => 36,
        Op::Concat { .. } => 37,
        Op::Slice { .. } => 38,
        Op::Squeeze { .. } => 39,
        Op::Unsqueeze { .. } => 40,
        Op::Tile { .. } => 41,
        Op::Reduce { .. } => 42,
        Op::MeanPool => 43,
        Op::L2Normalize => 44,
        Op::Megakernel(_) => 45,
    }
}
const VARIANTS: usize = 46;

/// Key used in `KNOWN_UNLOWERED`: the op name, plus the kind for ops whose support varies by kind.
fn key(op: &Op) -> String {
    match op {
        Op::Unary { kind } => format!("Unary::{kind:?}"),
        Op::Binary { kind } => format!("Binary::{kind:?}"),
        Op::Reduce { kind, .. } => format!("Reduce::{kind:?}"),
        Op::Gelu { tanh } => format!("Gelu::{}", if *tanh { "tanh" } else { "erf" }),
        other => other.name().to_string(),
    }
}

fn all_ops() -> Vec<Op> {
    let mut ops = vec![
        Op::Embedding,
        Op::Linear,
        Op::MatMul { transpose_a: false, transpose_b: true },
        Op::LayerNorm { eps: 1e-5 },
        Op::RmsNorm { eps: 1e-6 },
        Op::GroupNorm { groups: 8, eps: 1e-5 },
        Op::InstanceNorm { eps: 1e-5 },
        Op::Rope { head_dim: 16, rotary_dim: 16, variant: RopeVariant::HalfSplit, mscale: 1.0 },
        Op::Rope { head_dim: 16, rotary_dim: 16, variant: RopeVariant::Interleaved, mscale: 1.0 },
        Op::Rope {
            head_dim: 16,
            rotary_dim: 16,
            variant: RopeVariant::MRope { sections: [8, 4, 4], interleaved: true, pairs: PairLayout::Half },
            mscale: 1.0,
        },
        Op::Attention { heads: 2, kv_heads: 2, causal: true, window: None, soft_cap: None },
        Op::SwiGlu,
        Op::GeGlu,
        Op::CachedAttention {
            heads: 2, kv_heads: 1, head_dim: 16, v_head_dim: 16, scale: 1.0,
            causal: true, window: Some(4), soft_cap: Some(30.0), write: true,
        },
        Op::PagedCachedAttention {
            heads: 2, kv_heads: 1, head_dim: 16, v_head_dim: 16, scale: 1.0, causal: true,
            window: None, soft_cap: None, tokens_per_block: 16, max_blocks_per_seq: 4,
        },
        Op::CausalConv1dCached { kernel: 4, dilation: 1 },
        Op::DeltaRule { key_heads: 2, val_heads: 2, key_dim: 16, val_dim: 16 },
        Op::Gather { axis: 0 },
        Op::Scatter { axis: 0 },
        Op::Pad { padding: vec![0, 1], value: 0.0 },
        Op::AvgPool2d { kernel: [2, 2], stride: [2, 2] },
        Op::Upsample { scale: 2, mode: UpsampleMode::Nearest },
        Op::Conv3d { stride: [1, 1, 1], padding: [0, 0, 1, 1, 1, 1], dilation: [1, 1, 1], groups: 1 },
        Op::Moe { experts: 4, top_k: 2 },
        Op::Silu,
        Op::Gelu { tanh: true },
        Op::Gelu { tanh: false },
        Op::Relu,
        Op::Softmax { axis: -1 },
        Op::TopKSoftmax { axis: -1, k: 2 },
        Op::RelPosBias { buckets: 32, max_distance: 128, bidirectional: true, heads: 2 },
        Op::StateReorder,
        Op::Add,
        Op::Mul,
        Op::Conv2d { stride: [1, 1], padding: [1, 1, 1, 1], dilation: [1, 1], groups: 1 },
        Op::Conv1d { stride: 1, padding: [1, 1], dilation: 1, groups: 1 },
        Op::Reshape { shape: Some(vec![2, 2]) },
        Op::Transpose { perm: vec![1, 0] },
        Op::Concat { axis: 0 },
        Op::Slice { axis: 0, start: 0, end: 1 },
        Op::Squeeze { axes: vec![0] },
        Op::Unsqueeze { axes: vec![0] },
        Op::Tile { reps: vec![2] },
        Op::MeanPool,
        Op::L2Normalize,
        Op::Megakernel(MegakernelDescriptor {
            kind: MegakernelKind::Mlp {
                hidden_dim: 64, intermediate_dim: 128, eps: 1e-6, is_geglu: false, has_post_norm: false,
            },
            name: "mlp".into(),
        }),
    ];
    for kind in [
        UnaryKind::Exp, UnaryKind::Sigmoid, UnaryKind::Tanh, UnaryKind::Softplus, UnaryKind::Neg,
        UnaryKind::Sqrt, UnaryKind::Rsqrt, UnaryKind::Abs, UnaryKind::Relu, UnaryKind::Relu2,
        UnaryKind::Silu, UnaryKind::GeluErf, UnaryKind::GeluTanh, UnaryKind::Scale(0.5),
        UnaryKind::Clamp(-1.0, 1.0), UnaryKind::SoftCap(30.0),
    ] {
        ops.push(Op::Unary { kind });
    }
    for kind in [BinaryKind::Add, BinaryKind::Sub, BinaryKind::Mul, BinaryKind::Div, BinaryKind::Max, BinaryKind::Min] {
        ops.push(Op::Binary { kind });
    }
    for kind in [ReduceKind::Sum, ReduceKind::Mean, ReduceKind::Max, ReduceKind::Min] {
        ops.push(Op::Reduce { kind, axes: vec![-1], keep_dims: true });
    }
    ops
}

#[test]
fn every_op_is_lowered_or_known_unlowered() {
    let ops = all_ops();
    let mut seen = [false; VARIANTS];
    for op in &ops {
        seen[ordinal(op)] = true;
    }
    assert!(seen.iter().all(|&s| s), "all_ops() misses Op variant(s): {:?}", seen);

    let mut unexpected = Vec::new();
    let mut stale = Vec::new();
    let mut actual_unlowered = Vec::new();
    for op in &ops {
        for (dtype, dname) in DTYPES {
            let report = explore_node("sample", op, dtype);
            let k = key(op);
            let listed = KNOWN_UNLOWERED.iter().any(|(o, d, _)| *o == k && *d == dname);
            match report.lowered_by() {
                Some(_) if listed => stale.push(format!("remove {k} {dname} from KNOWN_UNLOWERED")),
                Some(_) => {}
                None => {
                    let reason = match &report.attempts[0].verdict {
                        Verdict::Rejected(r) => r.clone(),
                        Verdict::Lowered(_) => unreachable!(),
                    };
                    actual_unlowered.push(format!("    (\"{k}\", \"{dname}\", \"{reason}\"),"));
                    if !listed {
                        unexpected.push(format!("{k} {dname}: {reason}"));
                    }
                }
            }
        }
    }
    println!("unlowered pairs: {}\n{}", actual_unlowered.len(), actual_unlowered.join("\n"));
    assert!(stale.is_empty(), "{}", stale.join("\n"));
    assert!(
        unexpected.is_empty(),
        "ops with no lowering and no KNOWN_UNLOWERED entry:\n{}",
        unexpected.join("\n")
    );
}
