//! # Architecture Fingerprinting & Per-Layer Quantization Mapping
//!
//! Provides dynamic structural fingerprinting of model architectures directly from the
//! [`Graph`] IR without relying on hardcoded model name heuristics.
//! In addition, extracts fine-grained per-layer, per-projection quantization mappings
//! from checkpoint manifests ([`ModelManifest`]).

use crate::graph::{Graph, Node, Op, RopeVariant, ValueId};
use crate::hardware::QuantFormat;
use crate::model::ModelManifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    let feeds_mul = |n: &Node| graph.consumers(n.outputs[0]).any(|c| matches!(c.op, Op::Mul));
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
            Op::Silu => has_swiglu |= feeds_mul(node),
            Op::Gelu { .. } => has_geglu |= feeds_mul(node),
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

/// Mapping from layer index and operation name to its concrete quantization format.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LayerQuantizationMap {
    /// Layer index -> (Op name -> QuantFormat)
    pub layer_formats: HashMap<usize, HashMap<String, QuantFormat>>,
}

impl LayerQuantizationMap {
    /// Create a new empty layer quantization map.
    pub fn new() -> Self {
        Self {
            layer_formats: HashMap::new(),
        }
    }

    /// Insert or update a quantization format for a layer and operation.
    pub fn insert(&mut self, layer: usize, op: impl Into<String>, format: QuantFormat) {
        self.layer_formats
            .entry(layer)
            .or_default()
            .insert(op.into(), format);
    }

    /// Retrieve the quantization format for a specific layer and operation.
    pub fn get_format(&self, layer: usize, op: &str) -> Option<QuantFormat> {
        self.layer_formats.get(&layer)?.get(op).copied()
    }

    /// Total number of distinct layers tracked in the map.
    pub fn num_layers(&self) -> usize {
        self.layer_formats.len()
    }

    /// Whether the map contains any layer quantization entries.
    pub fn is_empty(&self) -> bool {
        self.layer_formats.is_empty()
    }
}

/// Extract layer index and canonical projection op name from a tensor name.
pub fn parse_tensor_layer_op(name: &str) -> Option<(usize, String)> {
    // 1. Extract layer index
    // Check delimiter split (e.g., "model.layers.0.self_attn.q_proj.weight" or "blk.0.attn_q.weight")
    let parts: Vec<&str> = name.split(|c| c == '.' || c == '/').collect();
    let mut layer_idx = None;

    for (i, &part) in parts.iter().enumerate() {
        if matches!(part, "layers" | "layer" | "blk" | "block" | "blocks" | "h") {
            if i + 1 < parts.len() {
                if let Ok(idx) = parts[i + 1].parse::<usize>() {
                    layer_idx = Some(idx);
                    break;
                }
            }
        }
    }

    // Fallback: split by '_' if no layer prefix was found with '.'
    if layer_idx.is_none() {
        let under_parts: Vec<&str> = name.split('_').collect();
        for (i, &part) in under_parts.iter().enumerate() {
            if matches!(part, "layers" | "layer" | "blk") {
                if i + 1 < under_parts.len() {
                    if let Ok(idx) = under_parts[i + 1].parse::<usize>() {
                        layer_idx = Some(idx);
                        break;
                    }
                }
            }
        }
    }

    let l = layer_idx?;

    // 2. Extract canonical op name
    let lower = name.to_lowercase();
    let op = if lower.contains("q_proj")
        || lower.contains("attn_q.")
        || lower.ends_with("attn_q.weight")
        || lower.ends_with("attn_q.bias")
        || lower.contains(".wq.")
        || lower.ends_with(".wq.weight")
    {
        "q_proj"
    } else if lower.contains("k_proj")
        || lower.contains("attn_k.")
        || lower.ends_with("attn_k.weight")
        || lower.ends_with("attn_k.bias")
        || lower.contains(".wk.")
        || lower.ends_with(".wk.weight")
    {
        "k_proj"
    } else if lower.contains("v_proj")
        || lower.contains("attn_v.")
        || lower.ends_with("attn_v.weight")
        || lower.ends_with("attn_v.bias")
        || lower.contains(".wv.")
        || lower.ends_with(".wv.weight")
    {
        "v_proj"
    } else if lower.contains("o_proj")
        || lower.contains("attn_output")
        || lower.contains("attn_out")
        || lower.contains(".wo.")
        || lower.ends_with(".wo.weight")
        || lower.contains("out_proj")
    {
        "o_proj"
    } else if lower.contains("gate_proj")
        || lower.contains("ffn_gate")
        || lower.contains(".w1.")
        || lower.ends_with(".w1.weight")
        || lower.contains("c_fc")
    {
        "gate_proj"
    } else if lower.contains("up_proj")
        || lower.contains("ffn_up")
        || lower.contains(".w3.")
        || lower.ends_with(".w3.weight")
    {
        "up_proj"
    } else if lower.contains("down_proj")
        || lower.contains("ffn_down")
        || lower.contains(".w2.")
        || lower.ends_with(".w2.weight")
        || (lower.contains("mlp") && lower.contains("c_proj"))
    {
        "down_proj"
    } else if lower.contains("q_a_proj") {
        "q_a_proj"
    } else if lower.contains("q_b_proj") {
        "q_b_proj"
    } else if lower.contains("kv_a_proj") {
        "kv_a_proj"
    } else if lower.contains("kv_b_proj") {
        "kv_b_proj"
    } else {
        // Check for any part ending with _proj
        let mut custom_proj = None;
        for part in &parts {
            if part.ends_with("_proj") {
                custom_proj = Some(*part);
                break;
            }
        }
        if let Some(cp) = custom_proj {
            cp
        } else {
            return None;
        }
    };

    Some((l, op.to_string()))
}

/// Build a per-layer quantization map from a parsed [`ModelManifest`].
pub fn build_quant_map(manifest: &ModelManifest) -> LayerQuantizationMap {
    let mut map = LayerQuantizationMap::new();

    for (name, tensor) in &manifest.tensors {
        if let Some((layer_idx, op_name)) = parse_tensor_layer_op(name) {
            let is_weight = name.ends_with(".weight")
                || name.ends_with("_weight")
                || !name.contains(".bias");

            let layer_map = map.layer_formats.entry(layer_idx).or_default();

            // Always prefer weight format over bias format
            if is_weight || !layer_map.contains_key(&op_name) {
                layer_map.insert(op_name, tensor.quant_format);
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;
    use crate::model::{ModelManifest, SourceFormat, VirtualTensor};

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
    fn test_layer_quantization_map_building() {
        let mut manifest = ModelManifest::default();
        manifest.format = SourceFormat::Safetensors;

        let make_vt = |name: &str, qf: QuantFormat| VirtualTensor {
            name: name.to_string(),
            shape: vec![4096, 4096],
            quant_format: qf,
            byte_offset: 0,
            byte_len: 1024,
            dtype: "F16".into(),
            shard_index: 0,
        };

        // Layer 0 - Hugging Face naming
        manifest.tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            make_vt("model.layers.0.self_attn.q_proj.weight", QuantFormat::GGUF_Q4_K_M),
        );
        // Bias tensor should not overwrite weight quantization format
        manifest.tensors.insert(
            "model.layers.0.self_attn.q_proj.bias".into(),
            make_vt("model.layers.0.self_attn.q_proj.bias", QuantFormat::FP32),
        );
        manifest.tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            make_vt("model.layers.0.self_attn.k_proj.weight", QuantFormat::GGUF_Q4_K_M),
        );
        manifest.tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            make_vt("model.layers.0.self_attn.v_proj.weight", QuantFormat::GGUF_Q6_K),
        );
        manifest.tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            make_vt("model.layers.0.self_attn.o_proj.weight", QuantFormat::FP16),
        );
        manifest.tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".into(),
            make_vt("model.layers.0.mlp.gate_proj.weight", QuantFormat::NVFP4),
        );
        manifest.tensors.insert(
            "model.layers.0.mlp.up_proj.weight".into(),
            make_vt("model.layers.0.mlp.up_proj.weight", QuantFormat::NVFP4),
        );
        manifest.tensors.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            make_vt("model.layers.0.mlp.down_proj.weight", QuantFormat::NVFP4),
        );

        // Layer 1 - GGUF naming
        manifest.tensors.insert(
            "blk.1.attn_q.weight".into(),
            make_vt("blk.1.attn_q.weight", QuantFormat::GGUF_Q8_0),
        );
        manifest.tensors.insert(
            "blk.1.attn_output.weight".into(),
            make_vt("blk.1.attn_output.weight", QuantFormat::GGUF_Q8_0),
        );
        manifest.tensors.insert(
            "blk.1.ffn_down.weight".into(),
            make_vt("blk.1.ffn_down.weight", QuantFormat::GGUF_Q4_0),
        );

        let quant_map = build_quant_map(&manifest);
        assert_eq!(quant_map.num_layers(), 2);

        // Layer 0 checks
        assert_eq!(quant_map.get_format(0, "q_proj"), Some(QuantFormat::GGUF_Q4_K_M));
        assert_eq!(quant_map.get_format(0, "k_proj"), Some(QuantFormat::GGUF_Q4_K_M));
        assert_eq!(quant_map.get_format(0, "v_proj"), Some(QuantFormat::GGUF_Q6_K));
        assert_eq!(quant_map.get_format(0, "o_proj"), Some(QuantFormat::FP16));
        assert_eq!(quant_map.get_format(0, "gate_proj"), Some(QuantFormat::NVFP4));
        assert_eq!(quant_map.get_format(0, "up_proj"), Some(QuantFormat::NVFP4));
        assert_eq!(quant_map.get_format(0, "down_proj"), Some(QuantFormat::NVFP4));

        // Layer 1 checks
        assert_eq!(quant_map.get_format(1, "q_proj"), Some(QuantFormat::GGUF_Q8_0));
        assert_eq!(quant_map.get_format(1, "o_proj"), Some(QuantFormat::GGUF_Q8_0));
        assert_eq!(quant_map.get_format(1, "down_proj"), Some(QuantFormat::GGUF_Q4_0));
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
        let act = g.add("gelu", Op::Gelu { tanh: true }, &[gate], vec![], F);
        g.add("mul", Op::Mul, &[act, up], vec![], F);

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
