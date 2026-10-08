//! Integration tests for universal loaders, architecture fingerprinting,
//! and configuration extraction.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

use gibbs_engine::graph::{DType, Graph, Op, RopeVariant, ValueId};
use gibbs_engine::model::fingerprint::fingerprint_graph;
use gibbs_engine::model::loaders::config::{
    ConfigExtractor, GgufConfigExtractor, HuggingFaceConfigExtractor, OnnxConfigExtractor,
};
use gibbs_engine::model::loaders::gguf::GgufWeightReader;
use gibbs_engine::model::loaders::onnx::OnnxWeightReader;
use gibbs_engine::model::loaders::safetensors::SafetensorsWeightReader;
use gibbs_engine::model::loaders::{load_with_format, WeightReader};

/// Helper to construct a synthetic valid Safetensors file.
fn create_synthetic_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    let file_path = dir.join("test_model.safetensors");
    let mut file = File::create(&file_path).expect("Failed to create safetensors file");

    // 4 floats: [1.0, 2.0, 3.0, 4.0]
    let tensor_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut raw_bytes = Vec::new();
    for val in tensor_data {
        raw_bytes.extend_from_slice(&val.to_le_bytes());
    }

    let header_json = serde_json::json!({
        "weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, raw_bytes.len()]
        }
    });

    let header_str = serde_json::to_string(&header_json).unwrap();
    let header_bytes = header_str.as_bytes();
    let header_len = header_bytes.len() as u64;

    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header_bytes).unwrap();
    file.write_all(&raw_bytes).unwrap();
    file.flush().unwrap();

    file_path
}

/// Helper to construct a minimal synthetic GGUF v2 file.
fn create_synthetic_gguf_v2() -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic "GGUF"
    buf.extend_from_slice(b"GGUF");
    // Version 2 (u32)
    buf.extend_from_slice(&2u32.to_le_bytes());
    // Tensor count (u64): 1
    buf.extend_from_slice(&1u64.to_le_bytes());
    // KV count (u64): 2 ("general.architecture" -> "llama", "general.alignment" -> 32)
    buf.extend_from_slice(&2u64.to_le_bytes());

    // Helper: write GGUF v2 string (u64 len + utf8)
    let write_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };

    // KV 1: "general.architecture" (type 8 = string)
    write_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    write_str(&mut buf, "llama");

    // KV 2: "general.alignment" (type 4 = uint32)
    write_str(&mut buf, "general.alignment");
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes());

    // Tensor Info 0:
    // Name
    write_str(&mut buf, "tensor_f32");
    // n_dims: 2 (u32)
    buf.extend_from_slice(&2u32.to_le_bytes());
    // dims (u64 each): [2, 2]
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    // type: 0 = F32 (u32)
    buf.extend_from_slice(&0u32.to_le_bytes());
    // offset: 0 (u64)
    buf.extend_from_slice(&0u64.to_le_bytes());

    // Align to 32 bytes
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.extend(std::iter::repeat(0u8).take(pad));

    // Tensor payload (4 x f32 = 16 bytes)
    for val in [10.0f32, 20.0f32, 30.0f32, 40.0f32] {
        buf.extend_from_slice(&val.to_le_bytes());
    }

    buf
}

#[test]
fn test_universal_loader_safetensors_interface() {
    let tmp = tempdir().unwrap();
    let sf_path = create_synthetic_safetensors(tmp.path());

    // Test direct SafetensorsWeightReader
    let mut reader = SafetensorsWeightReader::new(&sf_path).expect("Failed to open safetensors");
    let kv = reader.parse_header().expect("Failed to parse header");
    assert!(kv.is_empty());

    let tensors = reader.list_tensors().expect("Failed to list tensors");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "weight");
    assert_eq!(tensors[0].shape, vec![2, 2]);
    assert_eq!(tensors[0].dtype, "F32");

    let values = reader.load_tensor("weight").expect("Failed to load tensor");
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);

    let quant_bytes = reader
        .load_tensor_quantized("weight")
        .expect("Failed to load quantized tensor");
    assert_eq!(quant_bytes.len(), 16);

    // Test polymorphic load_with_format detection
    let boxed_reader = load_with_format(&sf_path, None).expect("load_with_format failed");
    let boxed_tensors = boxed_reader.list_tensors().expect("list_tensors failed");
    assert_eq!(boxed_tensors.len(), 1);
    assert_eq!(boxed_tensors[0].name, "weight");
}

#[test]
fn test_universal_loader_gguf_interface() {
    let raw_gguf = create_synthetic_gguf_v2();
    let tmp = tempdir().unwrap();
    let gguf_path = tmp.path().join("model.gguf");
    std::fs::write(&gguf_path, &raw_gguf).unwrap();

    let mut reader = GgufWeightReader::new(raw_gguf.clone()).expect("Failed to parse GGUF bytes");
    let kv = reader.parse_header().expect("Failed to parse header");
    assert_eq!(
        kv.get("general.architecture").map(|s| s.as_str()),
        Some("llama")
    );

    let tensors = reader.list_tensors().expect("Failed to list tensors");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "tensor_f32");
    assert_eq!(tensors[0].shape, vec![2, 2]);
    assert_eq!(tensors[0].dtype, "F32");

    let loaded = reader
        .load_tensor("tensor_f32")
        .expect("Failed to dequantize F32 tensor");
    assert_eq!(loaded, vec![10.0, 20.0, 30.0, 40.0]);

    // Test load_with_format on GGUF
    let boxed_reader =
        load_with_format(&gguf_path, Some("gguf")).expect("load_with_format gguf failed");
    let boxed_tensors = boxed_reader.list_tensors().unwrap();
    assert_eq!(boxed_tensors[0].name, "tensor_f32");
}

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
        base: 10000.0,
        variant,
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

/// `x -> q/k/v Linear -> Rope(q, k) -> attention`.
fn attention_block(g: &mut Graph, x: ValueId, bias: bool, rope: Op, attn: Op) -> ValueId {
    let pos = g.input("positions", vec![], DType::U32);
    let [q, k, v] = ["q", "k", "v"].map(|n| linear(g, n, x, bias));
    let q = g.add("rope_q", rope.clone(), &[q, pos], vec![], F);
    let k = g.add("rope_k", rope, &[k, pos], vec![], F);
    g.add("attn", attn, &[q, k, v], vec![], F)
}

fn swiglu(g: &mut Graph, x: ValueId) -> ValueId {
    let gate = linear(g, "gate", x, false);
    let up = linear(g, "up", x, false);
    g.add("swiglu", Op::SwiGlu, &[gate, up], vec![], F)
}

#[test]
fn test_architecture_fingerprinting_suite() {
    // 1. LLaMA (RMSNorm + SwiGLU + interleaved RoPE)
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
    swiglu(&mut g, a);
    let fp_llama = fingerprint_graph(&g);
    assert_eq!(fp_llama.family, "llama");
    assert!(fp_llama.has_swiglu);
    assert_eq!(fp_llama.rope_variant, "interleaved");

    // 2. DeepSeek-MLA (latent attention low-rank compression)
    let mut g = Graph::new();
    let x = g.input("x", vec![], F);
    let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-6 });
    let down = linear(&mut g, "down", n, false);
    let kv = linear(&mut g, "up", down, false);
    let pos = g.input("positions", vec![], DType::U32);
    let q = linear(&mut g, "q", n, false);
    let q = g.add("rope_q", rope(128, 64, RopeVariant::Interleaved), &[q, pos], vec![], F);
    let a = g.add("attn", attn(64, None, None), &[q, kv, kv], vec![], F);
    let router = g.weight("router", vec![], F);
    g.add("moe", Op::Moe { experts: 64, top_k: 8 }, &[a, router], vec![], F);
    let fp_ds = fingerprint_graph(&g);
    assert_eq!(fp_ds.family, "deepseek_v3");
    assert!(fp_ds.has_mla);
    assert!(fp_ds.has_moe);
    assert_eq!(fp_ds.expert_count, 64);

    // 3. MoE (Mixture of Experts)
    let mut g = Graph::new();
    let x = g.input("x", vec![], F);
    let n = norm(&mut g, x, Op::RmsNorm { eps: 1e-5 });
    let router = g.weight("router", vec![], F);
    let m = g.add("moe", Op::Moe { experts: 8, top_k: 2 }, &[n, router], vec![], F);
    swiglu(&mut g, m);
    let fp_moe = fingerprint_graph(&g);
    assert!(fp_moe.has_moe);
    assert_eq!(fp_moe.expert_count, 8);

    // 4. GLM (LayerNorm + SwiGLU/bias)
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
    let fp_glm = fingerprint_graph(&g);
    assert_eq!(fp_glm.family, "glm");

    // 5. Gemma (Soft-capping + RMSNorm)
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
    let gelu = g.add("gelu", Op::Gelu { tanh: true }, &[a], vec![], F);
    g.add("mul", Op::Mul, &[gelu, a], vec![], F);
    let fp_gemma = fingerprint_graph(&g);
    assert_eq!(fp_gemma.family, "gemma");
    assert_eq!(fp_gemma.attention_soft_capping, Some(50.0));
}

#[test]
fn test_corrupted_file_handling() {
    let tmp = tempdir().unwrap();

    // 1. Corrupted Safetensors: File smaller than 8 bytes
    let tiny_file = tmp.path().join("tiny.safetensors");
    std::fs::write(&tiny_file, b"too_few").unwrap();
    assert!(SafetensorsWeightReader::new(&tiny_file).is_err());

    // 2. Corrupted Safetensors: Truncated JSON header
    let trunc_sf = tmp.path().join("trunc.safetensors");
    let mut f = File::create(&trunc_sf).unwrap();
    let header_len = 1000u64;
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(b"{\"partial\": true").unwrap();
    f.flush().unwrap();
    assert!(SafetensorsWeightReader::new(&trunc_sf).is_err());

    // 3. Corrupted GGUF: Invalid magic
    let bad_magic = b"NOT_GGUF_HEADER_BYTES_12345678".to_vec();
    assert!(GgufWeightReader::new(bad_magic).is_err());

    // 4. Corrupted GGUF: Unsupported version
    let mut bad_version = Vec::new();
    bad_version.extend_from_slice(b"GGUF");
    bad_version.extend_from_slice(&99u32.to_le_bytes()); // invalid version
    bad_version.extend_from_slice(&[0u8; 32]);
    assert!(GgufWeightReader::new(bad_version).is_err());

    // 5. Corrupted ONNX: Invalid protobuf bytes
    let bad_onnx_bytes = b"garbage_non_protobuf_onnx_file_bytes";
    assert!(OnnxWeightReader::from_bytes(bad_onnx_bytes).is_err());
}

#[test]
fn test_model_config_extraction() {
    // 1. HuggingFace JSON config extraction
    let hf_json = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "model_type": "llama"
    });
    let hf_extractor = HuggingFaceConfigExtractor {
        config_json: hf_json,
    };
    let hf_config = hf_extractor
        .extract_config()
        .expect("HF config extraction failed");
    assert_eq!(hf_config.vocab_size, 32000);
    assert_eq!(hf_config.hidden_dim, 4096);
    assert_eq!(hf_config.num_layers, 32);
    assert_eq!(hf_config.num_heads, 32);
    assert_eq!(hf_config.head_dim, 128);
    assert_eq!(hf_config.intermediate_dim, 11008);
    assert_eq!(hf_config.model_type, "llama");
    assert!(hf_config.validate().is_ok());

    // 2. GGUF KV config extraction
    let mut gguf_kv = HashMap::new();
    gguf_kv.insert("llama.vocab_size".to_string(), "32000".to_string());
    gguf_kv.insert("llama.embedding_length".to_string(), "4096".to_string());
    gguf_kv.insert("llama.block_count".to_string(), "32".to_string());
    gguf_kv.insert("llama.attention.head_count".to_string(), "32".to_string());
    gguf_kv.insert("llama.feed_forward_length".to_string(), "11008".to_string());
    gguf_kv.insert("llama.context_length".to_string(), "4096".to_string());
    gguf_kv.insert("llama.rope.dimension_count".to_string(), "128".to_string());
    gguf_kv.insert("llama.rope.freq_base".to_string(), "10000.0".to_string());
    gguf_kv.insert(
        "llama.attention.layer_norm_rms_epsilon".to_string(),
        "0.00001".to_string(),
    );

    let gguf_extractor = GgufConfigExtractor { kv_pairs: gguf_kv };
    let gguf_config = gguf_extractor
        .extract_config()
        .expect("GGUF config extraction failed");
    assert_eq!(gguf_config.vocab_size, 32000);
    assert_eq!(gguf_config.hidden_dim, 4096);
    assert_eq!(gguf_config.num_layers, 32);
    assert_eq!(gguf_config.num_heads, 32);
    assert_eq!(gguf_config.head_dim, 128);
    assert_eq!(gguf_config.intermediate_dim, 11008);
    assert!(gguf_config.validate().is_ok());

    // 3. ONNX config extraction
    let onnx_extractor = OnnxConfigExtractor {
        graph_inputs: vec![("input_ids".to_string(), vec![32000, 128])],
        graph_outputs: vec![("logits".to_string(), vec![1, 128, 4096])],
        node_attrs: {
            let mut m = HashMap::new();
            let mut attrs = HashMap::new();
            attrs.insert("num_heads".to_string(), "32".to_string());
            m.insert("attention_node".to_string(), attrs);
            m
        },
    };
    let onnx_config = onnx_extractor
        .extract_config()
        .expect("ONNX config extraction failed");
    assert_eq!(onnx_config.vocab_size, 32000);
    assert_eq!(onnx_config.hidden_dim, 4096);
    assert_eq!(onnx_config.num_heads, 32);
    assert_eq!(onnx_config.head_dim, 128);
    assert!(onnx_config.validate().is_ok());
}
