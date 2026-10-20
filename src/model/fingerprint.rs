//! # Architecture Fingerprinting
//!
//! Provides dynamic structural fingerprinting of model architectures directly from the
//! [`Graph`] IR without relying on hardcoded model name heuristics.

use crate::graph::{Graph, Node, Op, RopeVariant, ValueId};
use serde::{Deserialize, Serialize};

/// Structural fingerprint of a transformer architecture extracted dynamically from graph IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureFingerprint {
    /// Discovered architecture family ("llama", "deepseek_v3", "qwen2", "glm", "gemma", "custom").
    pub family: String,
    /// Multi-Head Latent Attention (DeepSeek low-rank key-value joint compression).
    pub has_mla: bool,
    /// Mixture of Experts routing.
    pub has_moe: bool,
    /// Number of routed experts (0 if dense).
    pub expert_count: usize,
    /// Rotary Position Embedding layout ("interleaved", "half_split", "none").
    pub rope_variant: String,
    /// SwiGLU gated activation feed-forward network.
    pub has_swiglu: bool,
    /// Sliding window attention span if configured.
    pub sliding_window: Option<usize>,
    /// Logit soft-capping threshold (e.g. 50.0 in Gemma 2).
    pub attention_soft_capping: Option<f32>,
}

/// Inspects operations, SSA connectivity, and parameters in a [`Graph`] to dynamically
/// discover model architecture characteristics without relying on string names.
pub fn fingerprint_graph(graph: &Graph) -> ArchitectureFingerprint {
    let producer_is = |v: ValueId, f: fn(&Op) -> bool| graph.producer(v).filter(|n| f(&n.op));
    let linear = |v: ValueId| producer_is(v, |op| matches!(op, Op::Linear));
    let feeds_mul = |n: &Node| graph.consumers(n.outputs[0]).any(|c| matches!(c.op, Op::Binary { kind: crate::graph::BinaryKind::Mul }));
    // An attention operand's projection, looking through RoPE and (fused-QKV) slices.
    let projection = |mut v: ValueId| {
        while let Some(n) = producer_is(v, |op| matches!(op, Op::Rope { .. } | Op::Slice { .. })) {
            v = n.inputs[0];
        }
        linear(v)
    };

    let mut has_rmsnorm = false;
    let mut has_layernorm = false;
    let mut has_swiglu = false;
    let mut has_geglu = false;
    let mut has_attention = false;
    let mut attn_has_bias = false;
    let mut has_mla = false;
    let mut rope_variant = "none".to_string();
    let mut decoupled_rope = false;
    let mut sliding_window = None;
    let mut attention_soft_capping = None;
    let mut has_moe = false;
    let mut expert_count = 0;

    for node in &graph.nodes {
        match &node.op {
            Op::RmsNorm { .. } => has_rmsnorm = true,
            Op::LayerNorm { .. } => has_layernorm = true,
            Op::SwiGlu => has_swiglu = true,
            // Decomposed SwiGLU / GeGLU: the activation gates a Mul.
            Op::Unary { kind: crate::graph::UnaryKind::Silu } => has_swiglu |= feeds_mul(node),
            Op::Unary {
                kind: crate::graph::UnaryKind::GeluErf | crate::graph::UnaryKind::GeluTanh,
            } => has_geglu |= feeds_mul(node),
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                ..
            } => {
                rope_variant = match variant {
                    RopeVariant::Interleaved => "interleaved",
                    RopeVariant::HalfSplit => "half_split",
                    RopeVariant::MRope { .. } => "mrope",
                }
                .to_string();
                decoupled_rope |= rotary_dim < head_dim;
            }
            Op::Attention {
                window, soft_cap, ..
            } => {
                has_attention = true;
                sliding_window = sliding_window.or(*window);
                attention_soft_capping = attention_soft_capping.or(*soft_cap);
                for proj in node.inputs.iter().filter_map(|&v| projection(v)) {
                    attn_has_bias |= proj.inputs.len() == 3;
                    // MLA low-rank latent bottleneck: Linear -> [RMSNorm ->] Linear -> Attention.
                    let x = proj.inputs[0];
                    has_mla |= linear(x).is_some()
                        || producer_is(x, |op| matches!(op, Op::RmsNorm { .. }))
                            .is_some_and(|norm| linear(norm.inputs[0]).is_some());
                }
            }
            Op::MoeLinear { experts, .. } if *experts > 1 => {
                has_moe = true;
                expert_count = expert_count.max(*experts);
            }
            _ => {}
        }
    }

    if !has_mla && decoupled_rope && has_attention && (has_moe || has_rmsnorm) {
        // Decoupled RoPE combined with standard attention is another MLA signature
        has_mla = true;
    }

    // Family Classification
    let family = if has_mla {
        // DeepSeek V2 / V3 / R1 are the primary architectures featuring MLA
        "deepseek_v3".to_string()
    } else if has_layernorm && (has_swiglu || has_geglu || attn_has_bias) {
        // GLM architectures feature LayerNorm
        "glm".to_string()
    } else if attention_soft_capping.is_some() || (has_geglu && has_rmsnorm && !has_swiglu) {
        // Gemma / Gemma 2 uses attention soft-capping and GeGLU with RMSNorm
        "gemma".to_string()
    } else if has_rmsnorm
        && (rope_variant == "half_split" || attn_has_bias)
        && has_swiglu
        && (attn_has_bias || sliding_window.is_some())
    {
        // Qwen2 uses half-split RoPE, QKV attention bias, RMSNorm, and SwiGLU
        "qwen2".to_string()
    } else if has_rmsnorm && has_swiglu {
        // Canonical LLaMA family: RMSNorm + SwiGLU + interleaved RoPE
        "llama".to_string()
    } else {
        "custom".to_string()
    };

    ArchitectureFingerprint {
        family,
        has_mla,
        has_moe,
        expert_count,
        rope_variant,
        has_swiglu,
        sliding_window,
        attention_soft_capping,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    const F: DType = DType::F32;

    fn linear(g: &mut Graph, name: &str, x: ValueId, bias: bool) -> ValueId {
        let mut ins = vec![x, g.weight(format!("{name}.weight"), vec![], F)];
        if bias {
            ins.push(g.weight(format!("{name}.bias"), vec![], F));
        }
        g.add(name, Op::Linear, &ins, vec![], F)
    }

    fn norm(g: &mut Graph, x: ValueId, op: Op) -> ValueId {
        let gamma = g.weight("norm.weight", vec![], F);
        g.add("norm", op, &[x, gamma], vec![], F)
    }

    fn rope(head_dim: usize, rotary_dim: usize, variant: RopeVariant) -> Op {
        Op::Rope {
            head_dim,
            rotary_dim,
            variant,
            mscale: 1.0,
        }
    }

    fn attn(heads: usize, window: Option<usize>, soft_cap: Option<f32>) -> Op {
        Op::Attention {
            heads,
            kv_heads: heads,
            causal: true,
            window,
            soft_cap,
        }
    }

    /// `x -> q/k/v Linear -> Rope(q, k) -> attention -> o Linear`.
    fn attention_block(g: &mut Graph, x: ValueId, bias: bool, rope: Op, attn: Op) -> ValueId {
        let pos = g.input("positions", vec![], DType::U32);
        let [q, k, v] = ["q", "k", "v"].map(|n| linear(g, n, x, bias));
        let q = g.add("rope_q", rope.clone(), &[q, pos], vec![], F);
        let k = g.add("rope_k", rope, &[k, pos], vec![], F);
        let a = g.add("attn", attn, &[q, k, v], vec![], F);
        linear(g, "o", a, false)
    }

    /// `x -> gate/up Linear -> SwiGlu -> down Linear`.
    fn swiglu(g: &mut Graph, x: ValueId) -> ValueId {
        let gate = linear(g, "gate", x, false);
        let up = linear(g, "up", x, false);
        let s = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![], F);
        linear(g, "down", s, false)
    }

    #[test]
    fn test_fingerprint_synthetic_llama() {
        let mut g = Graph::new();
        let x = g.input("x", vec![], F);
        let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-5 });
        // Interleaved RoPE, no bias, no sliding window, no soft capping
        let a = attention_block(
            &mut g,
            n,
            false,
            rope(128, 128, RopeVariant::Interleaved),
            attn(32, None, None),
        );
        swiglu(&mut g, a);

        let fp = fingerprint_graph(&g);
        assert_eq!(fp.family, "llama");
        assert!(!fp.has_mla);
        assert!(!fp.has_moe);
        assert_eq!(fp.expert_count, 0);
        assert_eq!(fp.rope_variant, "interleaved");
        assert!(fp.has_swiglu);
        assert_eq!(fp.sliding_window, None);
        assert_eq!(fp.attention_soft_capping, None);
    }

    #[test]
    fn test_fingerprint_synthetic_deepseek_mla() {
        let mut g = Graph::new();
        let x = g.input("x", vec![], F);
        let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-6 });
        // KV compression to a 512 latent and decompression back
        let kv_down = linear(&mut g, "kv_down", n, false);
        let kv = linear(&mut g, "kv_up", kv_down, false);
        let pos = g.input("positions", vec![], DType::U32);
        let q = linear(&mut g, "q", n, false);
        // Decoupled RoPE: rotary_dim 64 < head_dim 128
        let q = g.add("rope_q", rope(128, 64, RopeVariant::Interleaved), &[q, pos], vec![], F);
        let a = g.add("attn_mla", attn(64, None, None), &[q, kv, kv], vec![], F);
        // DeepSeek MoE FFN (64 experts, top 8)
        let router = g.weight("router", vec![], F);
        g.add("moe_ffn", Op::MoeLinear { experts: 64, top_k: 8 }, &[a, router], vec![], F);

        let fp = fingerprint_graph(&g);
        assert_eq!(fp.family, "deepseek_v3");
        assert!(fp.has_mla);
        assert!(fp.has_moe);
        assert_eq!(fp.expert_count, 64);
        assert_eq!(fp.rope_variant, "interleaved");
    }

    #[test]
    fn test_fingerprint_synthetic_moe() {
        let mut g = Graph::new();
        let x = g.input("x", vec![], F);
        let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-5 });
        let a = attention_block(
            &mut g,
            n,
            false,
            rope(128, 128, RopeVariant::Interleaved),
            attn(32, None, None),
        );
        // MoE routing with 8 experts (e.g. Mixtral 8x7B), then a SwiGLU expert block
        let router = g.weight("router", vec![], F);
        let m = g.add("moe", Op::MoeLinear { experts: 8, top_k: 2 }, &[a, router], vec![], F);
        swiglu(&mut g, m);

        let fp = fingerprint_graph(&g);
        assert!(!fp.has_mla);
        assert!(fp.has_moe);
        assert_eq!(fp.expert_count, 8);
        assert!(fp.has_swiglu);
    }

    #[test]
    fn test_fingerprint_synthetic_qwen2() {
        let mut g = Graph::new();
        let x = g.input("x", vec![], F);
        let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-6 });
        let a = attention_block(
            &mut g,
            n,
            true,
            rope(128, 128, RopeVariant::HalfSplit),
            attn(28, Some(4096), None),
        );
        swiglu(&mut g, a);

        let fp = fingerprint_graph(&g);
        assert_eq!(fp.family, "qwen2");
        assert_eq!(fp.rope_variant, "half_split");
        assert_eq!(fp.sliding_window, Some(4096));
        assert!(fp.has_swiglu);
    }

    #[test]
    fn test_fingerprint_synthetic_gemma() {
        let mut g = Graph::new();
        let x = g.input("x", vec![], F);
        let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-6 });
        let a = attention_block(
            &mut g,
            n,
            false,
            rope(256, 256, RopeVariant::HalfSplit),
            attn(8, Some(4096), Some(50.0)),
        );
        // GeGLU: gelu(gate) * up
        let gate = linear(&mut g, "gate", a, false);
        let up = linear(&mut g, "up", a, false);
        let act = g.add("gelu", Op::Unary { kind: crate::graph::UnaryKind::GeluTanh }, &[gate], vec![], F);
        g.add("mul", Op::Binary { kind: crate::graph::BinaryKind::Mul }, &[act, up], vec![], F);

        let fp = fingerprint_graph(&g);
        assert_eq!(fp.family, "gemma");
        assert_eq!(fp.attention_soft_capping, Some(50.0));
        assert_eq!(fp.sliding_window, Some(4096));
    }

    #[test]
    fn test_fingerprint_synthetic_glm() {
        let mut g = Graph::new();
        let x = g.input("x", vec![], F);
        let n = norm(&mut g, x, Op::LayerNorm { eps: 1e-5 });
        let a = attention_block(
            &mut g,
            n,
            true,
            rope(128, 128, RopeVariant::Interleaved),
            attn(32, None, None),
        );
        swiglu(&mut g, a);

        let fp = fingerprint_graph(&g);
        assert_eq!(fp.family, "glm");
    }
}
