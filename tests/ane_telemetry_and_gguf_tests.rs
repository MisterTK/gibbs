use foundry_engine::model::hub::HuggingFaceHub;
use foundry_engine::model::graph_lm::{GraphLm, GraphLmConfig};
use foundry_engine::model::tokenizer::Tokenizer;
use foundry_engine::model::UniversalModelLoader;
use foundry_engine::telemetry::EngineTelemetry;
use gibbs_engine::foundry_engine;
use std::time::Duration;

#[test]
fn test_rolling_tps_and_latency_percentiles() {
    let telem = EngineTelemetry::new();
    assert_eq!(telem.current_tps(), 0.0);

    // Tokens/s is the token count over the 3 s window.
    telem.record_tokens(10);
    telem.record_tokens(20);
    assert!((telem.current_tps() - 10.0).abs() < 1e-3);

    for ms in (1..=10).map(|i| i * 10) {
        telem.record_ttft(Duration::from_millis(ms));
        telem.record_itl(Duration::from_millis(ms / 10));
    }
    let s = telem.get_summary();
    assert_eq!(s.total_tokens, 30);
    assert_eq!((s.ttft_p50_ms, s.ttft_p90_ms), (60.0, 90.0));
    assert_eq!((s.itl_p50_ms, s.itl_p90_ms), (6.0, 9.0));
}

#[test]
fn test_gguf_tokenizer_and_decode_robustness() {
    // 1. Verify Tokenizer::from_tokens builds valid vocabulary
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "Hello".to_string(),
        " world".to_string(),
        "!".to_string(),
    ];
    let tok = Tokenizer::from_tokens(tokens);
    assert_eq!(tok.vocab_size(), 6);
    assert!(tok.is_special(0));
    assert!(tok.is_special(1));
    assert!(tok.is_special(2));

    // 2. Verify decode_token space normalization
    let decoded_hello = tok.decode_token(3);
    assert_eq!(decoded_hello, "Hello");

    let decoded_world = tok.decode_token(4);
    assert_eq!(decoded_world, " world");

    // 3. Verify robust non-empty fallback for out-of-vocab token IDs
    let fallback = tok.decode_token(9999);
    assert_eq!(
        fallback, " <tok_9999>",
        "Out of vocab token must decode to readable non-empty representation"
    );

    // 4. Verify GGUF buffer parsing with real local GGUF if available
    let gguf_path = std::path::Path::new("disk/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/main/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    if gguf_path.exists() {
        let bytes = std::fs::read(gguf_path).expect("Failed to read test GGUF file");
        let extracted_tok = UniversalModelLoader::extract_gguf_tokenizer(&bytes);
        assert!(
            extracted_tok.is_ok(),
            "GGUF tokenizer extraction failed: {:?}",
            extracted_tok.err()
        );
        let real_tok = extracted_tok.unwrap();
        assert_eq!(
            real_tok.vocab_size(),
            49152,
            "Expected 49152 tokens from SmolLM2 GGUF"
        );
        println!(
            "Successfully extracted {} tokens from SmolLM2-135M GGUF!",
            real_tok.vocab_size()
        );
    }
}

#[test]
fn test_gguf_stories15m_and_smollm2_tensor_execution() {
    // 1. stories15M.gguf: parse GGUF bytes, inspect tensor count (>50), verify blk.0.attn_q.weight & token_embd.weight, and execute GEMV dot product
    let stories15m_path =
        std::path::Path::new("disk/models--ggml-org--tiny-llamas/snapshots/main/stories15M.gguf");
    if stories15m_path.exists() {
        let bytes = std::fs::read(stories15m_path).expect("Failed to read stories15M.gguf");
        let manifest = UniversalModelLoader::parse_gguf_bytes(&bytes)
            .expect("Failed to parse stories15M.gguf");

        assert!(
            manifest.tensors.len() > 50,
            "Expected >50 tensors in stories15M.gguf, got {}",
            manifest.tensors.len()
        );
        assert!(
            manifest.tensors.contains_key("blk.0.attn_q.weight"),
            "Missing blk.0.attn_q.weight in stories15M.gguf"
        );
        assert!(
            manifest.tensors.contains_key("token_embd.weight"),
            "Missing token_embd.weight in stories15M.gguf"
        );

        // Extract weights and execute GEMV dot product
        let q_weights = manifest
            .extract_tensor_f32(&bytes, "blk.0.attn_q.weight")
            .expect("Failed to extract blk.0.attn_q.weight");
        assert_eq!(q_weights.len(), 288 * 288);

        let input_vec = vec![0.1f32; 288];
        let row_0 = &q_weights[0..288];
        let dot_product = foundry_engine::model::vector_dot_product(&input_vec, row_0);
        assert!(!dot_product.is_nan(), "GEMV dot product produced NaN");
        println!(
            "stories15M.gguf blk.0.attn_q.weight GEMV dot product: {}",
            dot_product
        );
    }

    // 2. stories15M-q4_0.gguf: verify Q4_0 tensor loading, metadata architecture is llama, and vocabulary size is 32000
    let stories15m_q4_path = std::path::Path::new(
        "disk/models--ggml-org--tiny-llamas/snapshots/main/stories15M-q4_0.gguf",
    );
    if stories15m_q4_path.exists() {
        let bytes = std::fs::read(stories15m_q4_path).expect("Failed to read stories15M-q4_0.gguf");
        let manifest = UniversalModelLoader::parse_gguf_bytes(&bytes)
            .expect("Failed to parse stories15M-q4_0.gguf");

        assert_eq!(
            manifest
                .metadata
                .get("general.architecture")
                .map(|s| s.as_str()),
            Some("llama"),
            "Expected architecture 'llama' in stories15M-q4_0.gguf metadata"
        );

        let tok = UniversalModelLoader::extract_gguf_tokenizer(&bytes)
            .expect("Failed to extract tokenizer from stories15M-q4_0.gguf");
        assert_eq!(
            tok.vocab_size(),
            32000,
            "Expected 32000 vocabulary size for stories15M-q4_0.gguf"
        );

        // Verify Q4_0 tensor loading
        assert!(manifest.tensors.contains_key("token_embd.weight"));
        let token_embd = manifest.tensors.get("token_embd.weight").unwrap();
        assert_eq!(token_embd.dtype, "Q4_0");
        assert_eq!(token_embd.shape, vec![288, 32000]);

        let dequantized = manifest
            .extract_tensor_f32(&bytes, "token_embd.weight")
            .expect("Failed to dequantize Q4_0 token_embd.weight");
        assert_eq!(dequantized.len(), 288 * 32000);
        assert!(!dequantized[0].is_nan());
    }

    // 3. SmolLM2-135M-Instruct-Q4_K_M.gguf: extract tokenizer, verify vocab size is 49152, verify output.weight or token_embd.weight tensor metadata
    let smollm2_path = std::path::Path::new("disk/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/main/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    if smollm2_path.exists() {
        let bytes = std::fs::read(smollm2_path).expect("Failed to read SmolLM2 Q4_K_M GGUF");
        let tok = UniversalModelLoader::extract_gguf_tokenizer(&bytes)
            .expect("Failed to extract tokenizer from SmolLM2 GGUF");
        assert_eq!(
            tok.vocab_size(),
            49152,
            "Expected vocab size 49152 for SmolLM2"
        );

        let manifest = UniversalModelLoader::parse_gguf_bytes(&bytes)
            .expect("Failed to parse SmolLM2 GGUF manifest");
        let has_output = manifest.tensors.contains_key("output.weight");
        let has_token_embd = manifest.tensors.contains_key("token_embd.weight");
        assert!(
            has_output || has_token_embd,
            "Expected output.weight or token_embd.weight in SmolLM2 GGUF manifest"
        );
        if let Some(tensor) = manifest.tensors.get("token_embd.weight") {
            assert_eq!(tensor.shape, vec![576, 49152]);
        }

        if let Some(t) = manifest.tensors.get("blk.0.attn_q.weight") {
            println!(
                "blk.0.attn_q.weight shape: {:?}, dtype: {}",
                t.shape, t.dtype
            );
            let q_weights = manifest
                .extract_tensor_f32(&bytes, "blk.0.attn_q.weight")
                .expect("Failed to dequantize blk.0.attn_q.weight from SmolLM2 Q4_K");
            let expected_len = t.shape.iter().product::<usize>();
            assert_eq!(q_weights.len(), expected_len);
            assert!(q_weights.iter().any(|&v| v != 0.0));
            assert!(!q_weights[0].is_nan());
            println!(
                "Successfully dequantized SmolLM2 Q4_K tensor blk.0.attn_q.weight (len: {})",
                q_weights.len()
            );
        }
    }
}

#[test]
fn test_sub_2b_headliner_gguf_loading_and_architecture_support() {
    // Helper to generate a valid binary GGUF buffer for sub-2B headliner architectures
    fn build_synthetic_gguf_bytes(arch: &str, is_gemma: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3

        let h = 64usize;
        let kv_h = 32usize;
        let inter = 128usize;
        let vocab = 5usize;

        let mut tensor_list: Vec<(&str, Vec<usize>, Vec<f32>)> = Vec::new();
        tensor_list.push((
            "token_embd.weight",
            vec![h, vocab],
            vec![0.02f32; h * vocab],
        ));
        tensor_list.push(("blk.0.attn_norm.weight", vec![h], vec![1.0f32; h]));
        tensor_list.push(("blk.0.attn_q.weight", vec![h, h], vec![0.01f32; h * h]));
        tensor_list.push((
            "blk.0.attn_k.weight",
            vec![kv_h, h],
            vec![0.01f32; kv_h * h],
        ));
        tensor_list.push((
            "blk.0.attn_v.weight",
            vec![kv_h, h],
            vec![0.01f32; kv_h * h],
        ));

        if !is_gemma {
            // Qwen QKV separate biases
            tensor_list.push(("blk.0.attn_q.bias", vec![h], vec![0.05f32; h]));
            tensor_list.push(("blk.0.attn_k.bias", vec![kv_h], vec![0.02f32; kv_h]));
            tensor_list.push(("blk.0.attn_v.bias", vec![kv_h], vec![0.03f32; kv_h]));
        }

        tensor_list.push(("blk.0.attn_output.weight", vec![h, h], vec![0.01f32; h * h]));

        if is_gemma {
            // Gemma2 post-attention norm
            tensor_list.push(("blk.0.post_attention_norm.weight", vec![h], vec![1.0f32; h]));
        }

        tensor_list.push(("blk.0.ffn_norm.weight", vec![h], vec![1.0f32; h]));
        tensor_list.push((
            "blk.0.ffn_gate.weight",
            vec![inter, h],
            vec![0.01f32; inter * h],
        ));
        tensor_list.push((
            "blk.0.ffn_up.weight",
            vec![inter, h],
            vec![0.01f32; inter * h],
        ));
        tensor_list.push((
            "blk.0.ffn_down.weight",
            vec![h, inter],
            vec![0.01f32; h * inter],
        ));

        if is_gemma {
            // Gemma2 post-ffw norm
            tensor_list.push(("blk.0.post_ffw_norm.weight", vec![h], vec![1.0f32; h]));
        }

        tensor_list.push(("output_norm.weight", vec![h], vec![1.0f32; h]));
        tensor_list.push(("output.weight", vec![h, vocab], vec![0.01f32; h * vocab]));

        let tensor_count = tensor_list.len() as u64;
        let kv_count = 7u64;
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());

        let write_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        // KV 1: general.architecture (type 8 = string)
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_str(&mut buf, arch);

        // KV 2: {arch}.attention.head_count (type 4 = u32)
        write_str(&mut buf, &format!("{}.attention.head_count", arch));
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        // KV 3: {arch}.attention.head_count_kv (type 4 = u32)
        write_str(&mut buf, &format!("{}.attention.head_count_kv", arch));
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());

        // KV 4: {arch}.embedding_length (type 4 = u32)
        write_str(&mut buf, &format!("{}.embedding_length", arch));
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(h as u32).to_le_bytes());

        // KV 5: {arch}.block_count (type 4 = u32)
        write_str(&mut buf, &format!("{}.block_count", arch));
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());

        // KV 6: general.alignment (type 4 = u32)
        write_str(&mut buf, "general.alignment");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());

        // KV 7: tokenizer.ggml.tokens (type 9 = array, elem_type = 8)
        write_str(&mut buf, "tokenizer.ggml.tokens");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let token_strings = ["<unk>", "<|im_start|>", "<|im_end|>", "hello", "world"];
        buf.extend_from_slice(&(token_strings.len() as u64).to_le_bytes());
        for tok in &token_strings {
            write_str(&mut buf, tok);
        }

        // Tensor headers
        let mut current_offset = 0u64;
        for (name, shape, data) in &tensor_list {
            write_str(&mut buf, name);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // raw_type 0 = F32
            buf.extend_from_slice(&current_offset.to_le_bytes());
            current_offset += (data.len() * 4) as u64;
        }

        // Align to 32 bytes before tensor binary data
        let alignment = 32usize;
        let rem = buf.len() % alignment;
        if rem != 0 {
            buf.resize(buf.len() + (alignment - rem), 0u8);
        }

        for (_name, _shape, data) in &tensor_list {
            for &val in data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        buf
    }

    // 1. Verify Qwen GGUF loading, metadata parsing, tokenizer extraction, and QKV bias
    let qwen_bytes = build_synthetic_gguf_bytes("qwen2", false);
    let qwen_manifest = UniversalModelLoader::parse_gguf(&qwen_bytes)
        .expect("Failed to parse Qwen2 GGUF binary buffer");
    assert_eq!(
        qwen_manifest
            .metadata
            .get("general.architecture")
            .map(|s| s.as_str()),
        Some("qwen2")
    );
    assert_eq!(
        qwen_manifest
            .metadata
            .get("qwen2.attention.head_count")
            .map(|s| s.as_str()),
        Some("4")
    );
    assert_eq!(
        qwen_manifest
            .metadata
            .get("qwen2.attention.head_count_kv")
            .map(|s| s.as_str()),
        Some("2")
    );
    assert!(qwen_manifest.tensors.contains_key("blk.0.attn_q.bias"));
    assert!(qwen_manifest.tensors.contains_key("blk.0.attn_k.bias"));
    assert!(qwen_manifest.tensors.contains_key("blk.0.attn_v.bias"));

    let qwen_tok = UniversalModelLoader::extract_gguf_tokenizer(&qwen_bytes)
        .expect("Failed to extract tokenizer from Qwen2 GGUF");
    assert_eq!(qwen_tok.vocab_size(), 5);
    assert_eq!(qwen_tok.decode_token(3), "hello");
    assert_eq!(qwen_tok.decode_token(4), "world");

    let qwen_cfg = GraphLmConfig {
        vocab_size: 5,
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        head_dim: 16,
        max_seq_len: 256,
        ..Default::default()
    };
    let qwen_model = GraphLm::from_gguf_data(qwen_cfg.clone(), &qwen_bytes)
        .expect("Failed to construct GraphLm from Qwen2 GGUF");
    assert!(
        qwen_model.layers[0].qkv_proj.bias.is_some(),
        "Qwen2 model layer must have QKV bias populated"
    );
    let qwen_logits = qwen_model.forward(&[0, 1]);
    assert_eq!(qwen_logits.len(), 5);
    assert!(!qwen_logits[0].is_nan());

    // 2. Verify Gemma2 GGUF loading, metadata parsing, and post-attention/post-ffw norms
    let gemma_bytes = build_synthetic_gguf_bytes("gemma2", true);
    let gemma_manifest = UniversalModelLoader::parse_gguf(&gemma_bytes)
        .expect("Failed to parse Gemma2 GGUF binary buffer");
    assert_eq!(
        gemma_manifest
            .metadata
            .get("general.architecture")
            .map(|s| s.as_str()),
        Some("gemma2")
    );
    assert!(gemma_manifest
        .tensors
        .contains_key("blk.0.post_attention_norm.weight"));
    assert!(gemma_manifest
        .tensors
        .contains_key("blk.0.post_ffw_norm.weight"));

    let gemma_cfg = GraphLmConfig {
        vocab_size: 5,
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        head_dim: 16,
        max_seq_len: 256,
        ..Default::default()
    };
    let gemma_model = GraphLm::from_gguf_data(gemma_cfg, &gemma_bytes)
        .expect("Failed to construct GraphLm from Gemma2 GGUF");
    assert!(
        gemma_model.layers[0].post_attention_norm.is_some(),
        "Gemma2 model layer must have post_attention_norm populated"
    );
    assert!(
        gemma_model.layers[0].post_ffw_norm.is_some(),
        "Gemma2 model layer must have post_ffw_norm populated"
    );
    let gemma_logits = gemma_model.forward(&[2, 3]);
    assert_eq!(gemma_logits.len(), 5);
    assert!(!gemma_logits[0].is_nan());

    // 3. Verify sub-2B headliner candidate generation & parameter estimation in HuggingFaceHub
    let hub = HuggingFaceHub::new().with_budget(usize::MAX);

    // Qwen2.5 0.5B / 1.5B (and Coder variants)
    assert_eq!(
        hub.locate_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
            .estimated_params,
        500_000_000
    );
    assert_eq!(
        hub.locate_model("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
            .estimated_params,
        1_500_000_000
    );
    assert_eq!(
        hub.locate_model("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
            .estimated_params,
        500_000_000
    );
    assert_eq!(
        hub.locate_model("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
            .estimated_params,
        1_500_000_000
    );

    // Gemma 2B variants
    assert_eq!(
        hub.locate_model("unsloth/gemma-2-2b-it-GGUF")
            .estimated_params,
        2_000_000_000
    );
    assert_eq!(
        hub.locate_model("google/gemma-4-2b-it-GGUF")
            .estimated_params,
        2_000_000_000
    );

    // GLM-Edge 1.5B
    assert_eq!(
        hub.locate_model("THUDM/glm-edge-1.5b-chat-GGUF")
            .estimated_params,
        1_500_000_000
    );

    // Kimi-Chat-Sub4B
    assert_eq!(
        hub.locate_model("moonshotai/Kimi-Chat-Sub4B-GGUF")
            .estimated_params,
        3_000_000_000
    );

    // MiniCPM variants
    assert_eq!(
        hub.locate_model("openbmb/MiniCPM-2B-sft-bf16-GGUF")
            .estimated_params,
        2_000_000_000
    );
    assert_eq!(
        hub.locate_model("openbmb/MiniCPM3-4B-GGUF")
            .estimated_params,
        4_000_000_000
    );

    // SmolLM2 variants
    assert_eq!(
        hub.locate_model("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF")
            .estimated_params,
        135_000_000
    );
    assert_eq!(
        hub.locate_model("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF")
            .estimated_params,
        360_000_000
    );

    // 4. Verify resolve_or_synthesize_tiny configures accurate architecture parameters for these families
    let temp_dir = tempfile::tempdir().unwrap();
    let tiny_hub =
        HuggingFaceHub::with_cache_dir(temp_dir.path().to_path_buf()).with_budget(usize::MAX);

    let families = [
        ("foundry/gemma-2b", "gemma2", 2_000_000_000usize),
        ("foundry/qwen2.5-0.5b", "qwen2", 500_000_000usize),
        ("foundry/qwen2.5-1.5b", "qwen2", 1_500_000_000usize),
        ("foundry/glm-edge-1.5b", "chatglm", 1_500_000_000usize),
        ("foundry/kimi-chat-sub4b", "kimi", 3_000_000_000usize),
        ("foundry/minicpm-2b", "minicpm", 2_400_000_000usize),
        ("foundry/minicpm3-4b", "minicpm", 4_000_000_000usize),
    ];

    for (model_id, expected_arch, expected_params) in families {
        let artifacts = tiny_hub
            .resolve_or_synthesize_tiny(model_id)
            .unwrap_or_else(|e| panic!("Failed to resolve/synthesize {}: {}", model_id, e));
        assert!(artifacts.is_cached);
        let cfg_path = artifacts.config_path.expect("Missing config_path");
        let cfg_data = std::fs::read_to_string(&cfg_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&cfg_data).unwrap();
        assert_eq!(
            json["model_type"].as_str(),
            Some(expected_arch),
            "Mismatch model_type for {}",
            model_id
        );
        assert_eq!(
            json["num_parameters"].as_u64(),
            Some(expected_params as u64),
            "Mismatch num_parameters for {}",
            model_id
        );
        assert!(json["num_key_value_heads"].as_u64().is_some());
    }
}
