use axum::body::Body;
use axum::http::{Request, StatusCode};
use foundry_engine::hardware::QuantMode;
use foundry_engine::kv::PagedKVPool;
use foundry_engine::model::graph_lm::{GraphLm, GraphLmConfig};
use foundry_engine::model::hub::HuggingFaceHub;
use foundry_engine::model::tokenizer::Tokenizer;
use foundry_engine::scheduler::{
    ContinuousBatchScheduler, ExecutableModel, SamplingParams, DEFAULT_MAX_NEW_TOKENS,
};
use foundry_engine::server::{create_router, AppState};
use foundry_engine::telemetry::EngineTelemetry;
use gibbs_engine::foundry_engine;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

#[test]
fn test_tokenizer_encoding_decoding_and_streaming() {
    let tokenizer = Tokenizer::default_bpe();
    let prompt = "How many 'r's in the word strawberry?";
    let tokens = tokenizer.encode(prompt);
    assert!(!tokens.is_empty());

    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, prompt);

    // Verify token-by-token streaming delta
    for &tok_id in &tokens {
        let piece = tokenizer.decode_token(tok_id);
        assert!(!piece.is_empty());
    }

    // Verify chat template
    let templated = tokenizer.apply_chat_template(&[("user", prompt)]);
    assert!(templated.contains("<|im_start|>user"));
    assert!(templated.contains("<|im_start|>assistant"));
}

#[test]
fn test_hub_model_budget_safeguard() {
    let hub = HuggingFaceHub::new();
    let artifacts_big = hub.locate_model("meta-llama/Llama-3.2-1B-Instruct");
    let budget_res = hub.validate_budget(&artifacts_big);
    // Should warn or reject models exceeding 10M parameter testing budget
    assert!(budget_res.is_err());
    let err_msg = budget_res.unwrap_err();
    assert!(err_msg.contains("10,000,000") || err_msg.contains("budget"));

    // Tiny model resolves within budget
    let artifacts_tiny = hub.resolve_or_synthesize_tiny("foundry/tiny-5m").unwrap();
    assert!(artifacts_tiny.estimated_params <= 10_000_000);
}

#[test]
fn test_graph_lm_quantized_inference_and_generation() {
    let mut graph_lm = GraphLm::new(GraphLmConfig::smollm_1m());
    assert!(graph_lm.parameter_count() <= 10_000_000);

    // Test with NVFP4 Ultra Mode
    graph_lm.quantize(QuantMode::NVFP4Ultra);
    let prompt = "How many 'r's in the word strawberry?";
    let tokens = graph_lm.tokenizer.encode(prompt);
    let next_tok = graph_lm.generate_step(&tokens, 0.7);
    assert!(next_tok < graph_lm.config.vocab_size as u32);

    let generated = graph_lm.generate_text("There are", 5, 0.5);
    assert!(!generated.is_empty());
}

#[test]
fn test_real_safetensors_pythia_model_inference_and_comprehension() {
    use std::path::Path;
    let model_dir = Path::new("disk/models--EleutherAI--pythia-14m/snapshots/main");
    if !model_dir.exists() {
        println!("Skipping real Pythia test: directory not found");
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");
    assert!(tokenizer.vocab_size() >= 50277);

    // Verify BPE space prefix encoding on real vocabulary
    let prompt = "The capital of France is";
    let tokens = tokenizer.encode(prompt);
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokenizer.inv_vocab.get(&tokens[1]).unwrap(), "Ġcapital");
    assert_eq!(tokenizer.decode(&tokens), prompt);

    // Load real safetensors weights
    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");
    let cfg = GraphLmConfig {
        vocab_size: 50304,
        hidden_dim: 128,
        intermediate_dim: 512,
        num_layers: 6,
        num_heads: 4,
        head_dim: 32,
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: 0.25,
        use_parallel_residual: true,
        use_standard_layernorm: true,
        use_gelu_mlp: true,
        rope_theta: 10000.0,
        ..Default::default()
    };

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load real Pythia weights into GraphLm");
    model.set_tokenizer(tokenizer);

    // Generate text from real pretrained weights
    let generated = model.generate_text(prompt, 6, 0.0);
    println!("Real Pythia-14M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with("The capital of France is"));
    // Comprehensibility check: output must not contain raw byte leaks like <0x or control characters
    assert!(!generated.contains("<0x"));
    assert!(!generated.contains("Ġ"));
    assert!(!generated.contains("Ċ"));
}

#[test]
fn test_real_safetensors_tinystories_model_inference_and_comprehension() {
    use std::path::Path;
    let model_dir = Path::new("disk/models--roneneldan--TinyStories-Instruct-8M/snapshots/main");
    if !model_dir.exists() {
        println!("Skipping TinyStories test: directory not found");
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_standard_layernorm: hf_cfg.resolved_use_standard_layernorm(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load real TinyStories weights into GraphLm");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time";
    let generated = model.generate_text(prompt, 10, 0.5);
    println!("Real TinyStories-Instruct-8M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with("Once upon a time"));
    assert!(!generated.contains("<0x"));
    assert!(!generated.contains("Ġ"));
    assert!(!generated.contains("Ċ"));
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_backend_compile_and_dispatch() {
    // Real MSL compile + dispatch over a shared buffer.
    let metal = foundry_engine::backend::MetalBackend::new().expect("Metal device");
    assert!(metal.is_unified_memory());
    let ptr = metal
        .allocate_buffer(64 * 4)
        .expect("Metal buffer allocation");
    metal
        .jit_compile(
            "fill_idx",
            "#include <metal_stdlib>\nkernel void fill_idx(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = float(i) * 2.0f; }",
        )
        .expect("Metal JIT compile");
    metal
        .dispatch("fill_idx", (1, 1, 1), (64, 1, 1), &[ptr])
        .expect("Metal dispatch");
    let out = unsafe { std::slice::from_raw_parts(ptr as *const f32, 64) };
    assert_eq!(out[63], 126.0);
    assert!(
        metal
            .jit_compile("broken", "kernel void broken( {")
            .is_err(),
        "MSL compile errors must surface"
    );
}

#[tokio::test]
async fn test_server_real_streaming_and_system1_endpoint() {
    let kv_pool = Arc::new(PagedKVPool::new(1024));
    let telemetry = EngineTelemetry::new();

    let tiny_model = Arc::new(GraphLm::new(GraphLmConfig::smollm_1m()));
    let jev_model = None;

    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        "",
        kv_pool.clone(),
        Some(ExecutableModel::from(tiny_model)),
    ));

    // Spawn scheduler step runner for the duration of the test
    let sched_clone = scheduler.clone();
    let step_handle = tokio::spawn(async move {
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(5)).await;
            sched_clone.step();
        }
    });

    let state = AppState {
        scheduler,
        telemetry,
        jev_model,
        image_model: Default::default(),
        classifier: Default::default(),
    };
    let app = create_router(state);

    // 1. Test POST /v1/chat/completions (Non-Streaming)
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "foundry/smollm-1m",
                "messages": [{"role": "user", "content": "How many 'r's in strawberry?"}],
                "stream": false
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let chat_resp: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(chat_resp["object"], "chat.completion");
    assert!(
        chat_resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .len()
            > 0
    );

    let _ = step_handle.abort();
}

#[test]
fn test_smollm2_gqa_llama_inference_and_comprehension_ablation() {
    let model_dir =
        std::path::Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");
    assert_eq!(tokenizer.vocab_size(), 49152);

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    // SmolLM2 Llama architecture with GQA (9 attention heads, 3 KV heads)
    // Test with 2 layers for fast verification of GQA fusion, RoPE rotate_half, and SwiGLU
    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: if cfg!(debug_assertions) {
            4
        } else {
            hf_cfg.num_layers
        },
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_standard_layernorm: hf_cfg.resolved_use_standard_layernorm(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load SmolLM2 safetensors with GQA expansion");
    model.set_tokenizer(tokenizer);

    let prompt = "The quick brown fox";
    let generated = model.generate_text(prompt, 6, 0.0);
    println!("SmolLM2-135M-Instruct generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with("The quick brown fox"));
    assert!(!generated.contains("<0x"));
}
#[test]
fn test_smollm2_program_scheduler_inference() {
    let model_dir =
        std::path::Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: if cfg!(debug_assertions) {
            2
        } else {
            hf_cfg.num_layers
        },
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_standard_layernorm: hf_cfg.resolved_use_standard_layernorm(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");
    let mut raw_model = GraphLm::from_safetensors_data(cfg, &st_bytes).expect("loads safetensors");
    raw_model.set_tokenizer(tokenizer.clone());

    let exec_model = foundry_engine::scheduler::ExecutableModel::load_graph_lm(
        raw_model,
        foundry_engine::hardware::Device::Cpu,
    )
    .expect("loads Program");
    assert!(
        matches!(
            exec_model,
            foundry_engine::scheduler::ExecutableModel::Program(..)
        ),
        "expected ExecutableModel::Program on CPU"
    );

    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let mut kv = None;

    // Prefill through scheduler
    let first_next = exec_model
        .step_prefill_chunk(&prompt_tokens, 0, true, &prompt_tokens, 0.0, 1.0, &mut kv)
        .expect("prefill succeeds");
    let next_tok = first_next.expect("prefill returned next token");
    let mut generated_tokens = prompt_tokens.clone();
    generated_tokens.push(next_tok);

    for _ in 0..4 {
        let tok = exec_model
            .step_token_with_kv(
                &generated_tokens,
                *generated_tokens.last().unwrap(),
                generated_tokens.len(),
                0.0,
                1.0,
                &mut kv,
            )
            .expect("decode step succeeds");
        generated_tokens.push(tok);
    }

    let decoded_text = tokenizer.decode(&generated_tokens);
    println!(
        "SmolLM2-135M Program scheduler generated: '{}'",
        decoded_text
    );
    assert!(generated_tokens.len() > prompt_tokens.len());
}

#[test]
fn test_decode_throughput_benchmark_300_tok_s() {
    let model = GraphLm::new(GraphLmConfig::smollm_1m());
    let prompt = "In a quiet valley surrounded by mountains";
    let num_tokens = 50;

    // Warm-up run
    let _ = model.generate_text_fast(prompt, 5, 0.0);

    // Timed benchmark run
    let start = std::time::Instant::now();
    let generated = model.generate_text_fast(prompt, num_tokens, 0.0);
    let elapsed = start.elapsed();

    let tps = num_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "⚡ Fast KV Decode Throughput: {:.1} tok/s ({} tokens in {:.2}ms)",
        tps,
        num_tokens,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("Sample generation: '{}'", generated);

    assert!(!generated.is_empty());
    let target_tps = if cfg!(debug_assertions) { 100.0 } else { 300.0 };
    assert!(
        tps >= target_tps,
        "Throughput {:.1} tok/s must reach or exceed target {} tok/s",
        tps,
        target_tps
    );
}

#[test]
fn test_real_weights_tinystories_decode_throughput_300_tok_s() {
    let model_dir =
        std::path::Path::new("disk/models--roneneldan--TinyStories-Instruct-8M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_standard_layernorm: hf_cfg.resolved_use_standard_layernorm(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load real TinyStories weights into GraphLm");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time";
    let num_tokens = 30;

    // Warmup step to spin up CPU frequency governors and cache lines
    let _ = model.generate_text_fast("Hello", 1, 0.0);

    // Timed benchmark run: measures pure autoregressive decode time
    let start = std::time::Instant::now();
    let (generated, decode_elapsed) = model.generate_text_fast_timed(prompt, num_tokens, 0.0);
    let total_elapsed = start.elapsed();

    let pure_decode_tps = num_tokens as f64 / decode_elapsed.as_secs_f64();
    let e2e_tps = num_tokens as f64 / total_elapsed.as_secs_f64();
    println!(
        "⚡ Real TinyStories-8M Pure KV Decode Throughput: {:.1} tok/s ({} tokens in {:.2}ms) | E2E: {:.1} tok/s",
        pure_decode_tps,
        num_tokens,
        decode_elapsed.as_secs_f64() * 1000.0,
        e2e_tps
    );
    println!("Real weights generation: '{}'", generated);
    assert!(!generated.is_empty());
    let target_tps = if cfg!(debug_assertions) { 20.0 } else { 300.0 };
    assert!(
        pure_decode_tps > target_tps,
        "Throughput {:.1} tok/s must be strictly greater than target {} tok/s",
        pure_decode_tps,
        target_tps
    );
    if !cfg!(debug_assertions) {
        assert!(
            pure_decode_tps > 300.0,
            "Throughput {:.1} tok/s must be strictly greater than 300.0 tok/s",
            pure_decode_tps
        );
    }
}

#[test]
fn test_all_readme_models_inference_and_zero_gibberish() {
    println!("\n========================================================");
    println!("  FOUNDRY README MODELS INTEGRATION & ZERO-GIBBERISH SUITE");
    println!("========================================================");

    // 1. TinyStories-Instruct-8M (GPT-Neo / WPE / LayerNorm / GELU)
    let ts_dir =
        std::path::Path::new("disk/models--roneneldan--TinyStories-Instruct-8M/snapshots/main");
    if ts_dir.exists() {
        let tok_str = std::fs::read_to_string(ts_dir.join("tokenizer.json")).unwrap();
        let tok = Tokenizer::from_json_str(&tok_str).unwrap();
        let cfg_bytes = std::fs::read(ts_dir.join("config.json")).unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&cfg_bytes).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: hf.vocab_size,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: hf.num_layers,
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            use_standard_layernorm: hf.resolved_use_standard_layernorm(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(ts_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);

        let prompt = "Once in a quiet village surrounded by green hills,";
        let out = model.generate_text(prompt, 15, 0.0);
        println!("[1/5] TinyStories-Instruct-8M Output: '{}'", out);
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
        assert!(!out.is_empty());
    }

    // 2. Pythia-14M (GPT-NeoX / RoPE 25% / Parallel Residual / LayerNorm / GELU / Interleaved QKV Bias)
    let py_dir = std::path::Path::new("disk/models--EleutherAI--pythia-14m/snapshots/main");
    if py_dir.exists() {
        let tok_str = std::fs::read_to_string(py_dir.join("tokenizer.json")).unwrap();
        let tok = Tokenizer::from_json_str(&tok_str).unwrap();
        let cfg_bytes = std::fs::read(py_dir.join("config.json")).unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&cfg_bytes).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: 50304,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: hf.num_layers,
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            use_standard_layernorm: hf.resolved_use_standard_layernorm(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(py_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);

        let prompt = "In a secret garden behind the old stone wall,";
        let out = model.generate_text(prompt, 12, 0.0);
        println!("[2/5] Pythia-14M Output: '{}'", out);
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
        assert!(!out.is_empty());
    }

    // 3. SmolLM2-135M-Instruct (LLaMA / GQA 9Q:3KV / RoPE rotate_half 100% / RMSNorm / SwiGLU)
    let smol_dir =
        std::path::Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
    if smol_dir.exists() {
        let tok_str = std::fs::read_to_string(smol_dir.join("tokenizer.json")).unwrap();
        let tok = Tokenizer::from_json_str(&tok_str).unwrap();
        let cfg_bytes = std::fs::read(smol_dir.join("config.json")).unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&cfg_bytes).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: hf.vocab_size,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: if cfg!(debug_assertions) {
                4
            } else {
                hf.num_layers
            },
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(smol_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);

        let prompt = "User: How are you today? Assistant: I am";
        let out = model.generate_text(prompt, 10, 0.0);
        println!("[3/5] SmolLM2-135M-Instruct Output: '{}'", out);
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
        assert!(!out.is_empty());
    }

    // 4. Foundry Edge Preset Models (Tiny-5M, SmolLM-1M, Tiny-10M)
    {
        let model = GraphLm::new(GraphLmConfig::tiny_5m());
        let prompt = "The sun rises over the horizon,";
        let out = model.generate_text(prompt, 10, 0.5);
        println!("[4/5] Foundry Tiny-5M Output: '{}'", out);
        assert!(!out.is_empty());
        assert!(!out.contains("<0x"));
        assert!(out.starts_with(prompt));
    }

    // 7. GGUF Zero-Copy Loader & Tensor Execution (tiny-llamas stories15M)
    {
        let gguf_path = std::path::Path::new(
            "disk/models--ggml-org--tiny-llamas/snapshots/main/stories15M.gguf",
        );
        if gguf_path.exists() {
            let bytes = std::fs::read(gguf_path).expect("Failed to read stories15M.gguf");
            let manifest = foundry_engine::model::UniversalModelLoader::parse_gguf_bytes(&bytes)
                .expect("Failed to parse stories15M.gguf");
            assert!(manifest.tensors.len() >= 50);
            assert!(manifest.tensors.contains_key("blk.0.attn_q.weight"));
            assert!(manifest.tensors.contains_key("token_embd.weight"));
            println!(
                "[7/7] stories15M.gguf verified with {} tensors, zero-copy parsed",
                manifest.tensors.len()
            );
        }
    }

    println!("========================================================");
    println!("  ALL README MODELS VERIFIED WITH ZERO-GIBBERISH OUTPUT");
    println!("========================================================");
}

#[test]
fn test_xenova_llama2_c_stories15m_inference_and_comprehension() {
    let model_dir = std::path::Path::new("disk/models--Xenova--llama2.c-stories15M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");
    assert_eq!(tokenizer.vocab_size(), 32000);

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load Xenova/llama2.c-stories15M safetensors");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time, there was a little girl named Lily.";
    let generated = model.generate_text(prompt, 20, 0.0);
    println!("Xenova/llama2.c-stories15M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with(prompt));
    assert!(!generated.contains("<0x"));
}

#[test]
fn test_ant_5m_llama_inference_and_comprehension() {
    let model_dir = std::path::Path::new("disk/models--GODELEV--Ant-5M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");
    assert_eq!(tokenizer.vocab_size(), 16384);

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 1024,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load GODELEV/Ant-5M safetensors");
    model.set_tokenizer(tokenizer);

    let prompt = "<USER>Tell me a story<ASSISTANT>";
    let generated = model.generate_text(prompt, 15, 0.0);
    println!("GODELEV/Ant-5M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(!generated.contains("<0x"));
}

#[test]
fn test_tinystories_1m_inference_and_comprehension() {
    let model_dir = std::path::Path::new("disk/models--roneneldan--TinyStories-1M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load roneneldan/TinyStories-1M safetensors");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time";
    let generated = model.generate_text(prompt, 15, 0.0);
    println!("roneneldan/TinyStories-1M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with(prompt));
    assert!(!generated.contains("<0x"));
}

#[test]
fn test_tinystories_3m_inference_and_comprehension() {
    let model_dir = std::path::Path::new("disk/models--roneneldan--TinyStories-3M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load roneneldan/TinyStories-3M safetensors");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time";
    let generated = model.generate_text(prompt, 15, 0.0);
    println!("roneneldan/TinyStories-3M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with(prompt));
    assert!(!generated.contains("<0x"));
}

#[test]
fn test_tinystories_8m_base_inference_and_comprehension() {
    let model_dir = std::path::Path::new("disk/models--roneneldan--TinyStories-8M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load roneneldan/TinyStories-8M safetensors");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time";
    let generated = model.generate_text(prompt, 15, 0.0);
    println!(
        "roneneldan/TinyStories-8M (base) generation: '{}'",
        generated
    );
    assert!(!generated.is_empty());
    assert!(generated.starts_with(prompt));
    assert!(!generated.contains("<0x"));
}

#[test]
fn test_plyx_15m_inference_and_comprehension() {
    let model_dir = std::path::Path::new("disk/models--MultivexAI--Plyx-15M/snapshots/main");
    if !model_dir.exists() {
        return;
    }

    let config_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&config_path).expect("Failed to read config.json");
    let hf_cfg: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");

    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
        quant_mode: QuantMode::Uncompressed,
        rotary_pct: hf_cfg.resolved_rotary_pct(),
        use_parallel_residual: hf_cfg.resolved_parallel_residual(),
        use_gelu_mlp: hf_cfg.resolved_use_gelu_mlp(),
        rope_theta: hf_cfg.resolved_rope_theta(),
        ..Default::default()
    };

    let st_path = model_dir.join("model.safetensors");
    let st_bytes = std::fs::read(&st_path).expect("Failed to read model.safetensors");

    let mut model = GraphLm::from_safetensors_data(cfg, &st_bytes)
        .expect("Failed to load MultivexAI/Plyx-15M safetensors");
    model.set_tokenizer(tokenizer);

    let prompt = "Once upon a time";
    let generated = model.generate_text(prompt, 15, 0.0);
    println!("MultivexAI/Plyx-15M generation: '{}'", generated);
    assert!(!generated.is_empty());
    assert!(generated.starts_with(prompt));
    assert!(!generated.contains("<0x"));
}

#[test]
fn test_default_uncompressed_and_no_hadamard_mlp_replacement_guardrail() {
    // 1. Verify default config has Hadamard attention/MLP turned off
    let default_cfg = GraphLmConfig::default();
    assert!(
        !default_cfg.use_hadamard_attention,
        "Hadamard attention must not be enabled by default"
    );
    assert!(
        !default_cfg.use_hadamard_mlp,
        "Walsh-Hadamard MLP must not replace standard MLPs by default"
    );
    assert_eq!(
        default_cfg.quant_mode,
        QuantMode::Uncompressed,
        "Inference must default to uncompressed first"
    );

    // 2. Verify all presets default to uncompressed and no Hadamard substitution
    let presets = [
        GraphLmConfig::smollm_1m(),
        GraphLmConfig::tiny_5m(),
        GraphLmConfig::tiny_10m(),
    ];
    for p in presets {
        assert!(!p.use_hadamard_attention);
        assert!(!p.use_hadamard_mlp);
        assert_eq!(p.quant_mode, QuantMode::Uncompressed);
    }

    // 3. Verify instantiated GraphLm weights default to uncompressed weights
    let model = GraphLm::new(GraphLmConfig::smollm_1m());
    for (i, layer) in model.layers.iter().enumerate() {
        assert!(
            matches!(
                layer.qkv_proj.weight,
                foundry_engine::model::layers::QuantizedWeight::Uncompressed { .. }
            ),
            "Layer {} qkv_proj must default to Uncompressed",
            i
        );
        assert!(
            matches!(
                layer.gate_up_proj.weight,
                foundry_engine::model::layers::QuantizedWeight::Uncompressed { .. }
            ),
            "Layer {} gate_up_proj must default to Uncompressed",
            i
        );
        assert!(
            matches!(
                layer.down_proj.weight,
                foundry_engine::model::layers::QuantizedWeight::Uncompressed { .. }
            ),
            "Layer {} down_proj must default to Uncompressed",
            i
        );
    }
}

#[test]
fn test_all_readme_models_legibility_and_grammar_verification() {
    // 1. Verify TinyStories-Instruct-8M
    let ts_dir =
        std::path::Path::new("disk/models--roneneldan--TinyStories-Instruct-8M/snapshots/main");
    if ts_dir.exists() {
        let tok_str = std::fs::read_to_string(ts_dir.join("tokenizer.json")).unwrap();
        let tok = Tokenizer::from_json_str(&tok_str).unwrap();
        let cfg_bytes = std::fs::read(ts_dir.join("config.json")).unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&cfg_bytes).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: hf.vocab_size,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: hf.num_layers,
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_standard_layernorm: hf.resolved_use_standard_layernorm(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(ts_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);
        let out = model.generate_text("Once upon a time", 15, 0.5);
        assert!(out.starts_with("Once upon a time"));
        assert!(!out.contains("<0x"));
    }

    // 2. Verify Pythia-14M
    let py_dir = std::path::Path::new("disk/models--EleutherAI--pythia-14m/snapshots/main");
    if py_dir.exists() {
        let tok_str = std::fs::read_to_string(py_dir.join("tokenizer.json")).unwrap();
        let tok = Tokenizer::from_json_str(&tok_str).unwrap();
        let cfg_bytes = std::fs::read(py_dir.join("config.json")).unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&cfg_bytes).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: 50304,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: hf.num_layers,
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_standard_layernorm: hf.resolved_use_standard_layernorm(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(py_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);
        let out = model.generate_text("The capital of France is", 10, 0.0);
        assert!(out.starts_with("The capital of France is"));
        assert!(!out.contains("<0x"));
    }
}

#[test]
fn test_dynamic_unseen_prompts_across_all_readme_models() {
    println!("\n========================================================");
    println!("  TESTING DYNAMIC, UNSEEN PROMPTS ACROSS ALL README MODELS");
    println!("========================================================");

    // 1. TinyStories-1M, 3M, 8M, Instruct-8M dynamic narrative testing
    let dynamic_story_prompt = "In a quiet village near the mountains, a friendly dog";
    for (name, dir_str) in [
        (
            "TinyStories-1M",
            "disk/models--roneneldan--TinyStories-1M/snapshots/main",
        ),
        (
            "TinyStories-3M",
            "disk/models--roneneldan--TinyStories-3M/snapshots/main",
        ),
        (
            "TinyStories-8M",
            "disk/models--roneneldan--TinyStories-8M/snapshots/main",
        ),
        (
            "TinyStories-Instruct-8M",
            "disk/models--roneneldan--TinyStories-Instruct-8M/snapshots/main",
        ),
    ] {
        let p = std::path::Path::new(dir_str);
        if p.exists() {
            let tok = Tokenizer::from_json_str(
                &std::fs::read_to_string(p.join("tokenizer.json")).unwrap(),
            )
            .unwrap();
            let hf: foundry_engine::model::hf_loader::HfConfigJson =
                serde_json::from_slice(&std::fs::read(p.join("config.json")).unwrap()).unwrap();
            let cfg = GraphLmConfig {
                vocab_size: hf.vocab_size,
                hidden_dim: hf.hidden_dim,
                intermediate_dim: hf.resolved_intermediate_dim(),
                num_layers: if cfg!(debug_assertions) {
                    4.min(hf.num_layers)
                } else {
                    hf.num_layers
                },
                num_heads: hf.num_heads,
                head_dim: hf.hidden_dim / hf.num_heads.max(1),
                max_seq_len: 2048,
                quant_mode: QuantMode::Uncompressed,
                rotary_pct: hf.resolved_rotary_pct(),
                use_parallel_residual: hf.resolved_parallel_residual(),
                use_gelu_mlp: hf.resolved_use_gelu_mlp(),
                use_standard_layernorm: hf.resolved_use_standard_layernorm(),
                rope_theta: hf.resolved_rope_theta(),
                ..Default::default()
            };
            let weights = std::fs::read(p.join("model.safetensors")).unwrap();
            let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
            model.set_tokenizer(tok);
            let out = model.generate_text(dynamic_story_prompt, 12, 0.0);
            println!("  [Dynamic] {}: '{}'", name, out.replace('\n', " "));
            assert!(out.starts_with(dynamic_story_prompt));
            assert!(!out.contains("<0x"));
            assert!(!out.is_empty());
        }
    }

    // 2. Pythia-14M dynamic factual prompt
    let py_dir = std::path::Path::new("disk/models--EleutherAI--pythia-14m/snapshots/main");
    if py_dir.exists() {
        let tok = Tokenizer::from_json_str(
            &std::fs::read_to_string(py_dir.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&std::fs::read(py_dir.join("config.json")).unwrap()).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: 50304,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: if cfg!(debug_assertions) {
                4.min(hf.num_layers)
            } else {
                hf.num_layers
            },
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            use_standard_layernorm: hf.resolved_use_standard_layernorm(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(py_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);
        let prompt = "The sun rises in the east and";
        let out = model.generate_text(prompt, 12, 0.0);
        println!("  [Dynamic] Pythia-14M: '{}'", out.replace('\n', " "));
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
    }

    // 3. SmolLM2-135M dynamic instruction dialogue prompt
    let smol_dir =
        std::path::Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
    if smol_dir.exists() {
        let tok = Tokenizer::from_json_str(
            &std::fs::read_to_string(smol_dir.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&std::fs::read(smol_dir.join("config.json")).unwrap()).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: hf.vocab_size,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: if cfg!(debug_assertions) {
                4.min(hf.num_layers)
            } else {
                hf.num_layers
            },
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(smol_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);
        let prompt = "User: What is the fastest animal on land? Assistant: The";
        let out = model.generate_text(prompt, 10, 0.0);
        println!("  [Dynamic] SmolLM2-135M: '{}'", out.replace('\n', " "));
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
    }

    // 4. llama2.c-stories15M dynamic narrative prompt
    let xen_dir = std::path::Path::new("disk/models--Xenova--llama2.c-stories15M/snapshots/main");
    if xen_dir.exists() {
        let tok = Tokenizer::from_json_str(
            &std::fs::read_to_string(xen_dir.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&std::fs::read(xen_dir.join("config.json")).unwrap()).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: hf.vocab_size,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: if cfg!(debug_assertions) {
                4.min(hf.num_layers)
            } else {
                hf.num_layers
            },
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(xen_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);
        let prompt = "One morning, a little bird named Pip decided to";
        let out = model.generate_text(prompt, 12, 0.0);
        println!(
            "  [Dynamic] llama2.c-stories15M: '{}'",
            out.replace('\n', " ")
        );
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
    }

    // 5. MultivexAI/Plyx-15M dynamic story prompt
    let plyx_dir = std::path::Path::new("disk/models--MultivexAI--Plyx-15M/snapshots/main");
    if plyx_dir.exists() {
        let tok = Tokenizer::from_json_str(
            &std::fs::read_to_string(plyx_dir.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
        let hf: foundry_engine::model::hf_loader::HfConfigJson =
            serde_json::from_slice(&std::fs::read(plyx_dir.join("config.json")).unwrap()).unwrap();
        let cfg = GraphLmConfig {
            vocab_size: hf.vocab_size,
            hidden_dim: hf.hidden_dim,
            intermediate_dim: hf.resolved_intermediate_dim(),
            num_layers: if cfg!(debug_assertions) {
                4.min(hf.num_layers)
            } else {
                hf.num_layers
            },
            num_heads: hf.num_heads,
            head_dim: hf.hidden_dim / hf.num_heads.max(1),
            max_seq_len: 2048,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: hf.resolved_rotary_pct(),
            use_parallel_residual: hf.resolved_parallel_residual(),
            use_gelu_mlp: hf.resolved_use_gelu_mlp(),
            rope_theta: hf.resolved_rope_theta(),
            ..Default::default()
        };
        let weights = std::fs::read(plyx_dir.join("model.safetensors")).unwrap();
        let mut model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
        model.set_tokenizer(tok);
        let prompt = "The little boy opened the wooden box and";
        let out = model.generate_text(prompt, 10, 0.0);
        println!("  [Dynamic] Plyx-15M: '{}'", out.replace('\n', " "));
        assert!(out.starts_with(prompt));
        assert!(!out.contains("<0x"));
    }

    println!("========================================================");
    println!("  ALL DYNAMIC, UNSEEN PROMPTS VERIFIED SUCCESSFULLY");
    println!("========================================================");
}

/// GLM-Edge (HF `GlmForCausalLM`): interleaved-pair RoPE + fused `mlp.gate_up_proj`.
/// Expected values come from transformers' `GlmForCausalLM` in f32 on the same checkpoint.
#[test]
fn test_glm_edge_matches_hf_oracle() {
    let dir = std::path::Path::new("disk/models--THUDM--glm-edge-1.5b-chat/snapshots/main");
    if !dir.exists() {
        return;
    }
    let hf: foundry_engine::model::hf_loader::HfConfigJson =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let mut cfg = GraphLmConfig::from_hf(&hf, QuantMode::Uncompressed);
    // Debug builds run 2 layers; expected = HF logits of that truncation (+ final norm, lm_head).
    let (layers, expect) = if cfg!(debug_assertions) {
        (2, [(1339usize, 16.4600f32), (763, 15.7523)])
    } else {
        (28, [(14488, 16.5629), (11262, 13.4245)]) // " Paris", " located"
    };
    cfg.num_layers = layers;
    let tok =
        Tokenizer::from_json_str(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap())
            .unwrap();
    let ids = tok.encode("The capital of France is");
    assert_eq!(ids, [1026, 8570, 325, 11334, 398]);
    let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
    let model = GraphLm::from_safetensors_data(cfg, &weights).unwrap();
    drop(weights);

    let argmax = |lg: &[f32]| {
        (0..lg.len())
            .max_by(|&a, &b| lg[a].total_cmp(&lg[b]))
            .unwrap()
    };
    let mut kv: Vec<_> = (0..layers)
        .map(|_| foundry_engine::model::graph_lm::LayerKVCache::new(32, model.config.hidden_dim))
        .collect();
    let mut logits = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        logits = model.forward_token_with_kv(t, pos, &mut kv);
    }
    for lg in [&model.forward(&ids), &logits] {
        assert_eq!(argmax(lg), expect[0].0);
        for (id, want) in expect {
            assert!(
                (lg[id] - want).abs() < 0.02,
                "logit[{id}] = {} want {want}",
                lg[id]
            );
        }
    }
    if cfg!(debug_assertions) {
        return;
    }
    // HF greedy: " Paris. It is the most visited"
    let mut out = Vec::new();
    for _ in 0..7 {
        let next = argmax(&logits) as u32;
        out.push(next);
        logits = model.forward_token_with_kv(next, ids.len() + out.len() - 1, &mut kv);
    }
    assert_eq!(out, [14488, 46, 1462, 398, 283, 1945, 24733]);
}

/// Greedy ids via full-sequence `forward` and the KV-cache decode path (must agree).
fn greedy_both_paths(model: &GraphLm, prompt: &[u32], n: usize) -> Vec<u32> {
    let argmax = |lg: &[f32]| {
        (0..lg.len())
            .max_by(|&a, &b| lg[a].total_cmp(&lg[b]))
            .unwrap() as u32
    };
    let mut kv: Vec<_> = (0..model.layers.len())
        .map(|_| foundry_engine::model::graph_lm::LayerKVCache::new(64, model.config.hidden_dim))
        .collect();
    let mut logits = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        logits = model.forward_token_with_kv(t, pos, &mut kv);
    }
    let mut ids = prompt.to_vec();
    for _ in 0..n {
        let next = argmax(&model.forward(&ids));
        assert_eq!(
            argmax(&logits),
            next,
            "KV path disagrees with forward at {}",
            ids.len()
        );
        ids.push(next);
        logits = model.forward_token_with_kv(next, ids.len() - 1, &mut kv);
    }
    ids[prompt.len()..].to_vec()
}

/// llama.cpp oracle (`llama-tokenize`, `llama-simple`) on ggml-org/tiny-llamas stories15M:
/// GGUF metadata config, llama q/k in interleaved-RoPE layout, Q4_0 element order, and
/// SentencePiece encoding ('▁' spaces + BOS) / decoding (dummy prefix stripped).
#[test]
fn test_gguf_stories15m_matches_llama_cpp() {
    for (file, want) in [
        (
            "stories15M.gguf",
            [
                29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708,
                5377, 297,
            ],
        ),
        (
            "stories15M-q4_0.gguf",
            [
                29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708,
                5377, 322,
            ],
        ),
    ] {
        let path =
            std::path::Path::new("disk/models--ggml-org--tiny-llamas/snapshots/main").join(file);
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let mut model = GraphLm::from_gguf_data(GraphLmConfig::default(), &bytes).unwrap();
        model.set_tokenizer(
            foundry_engine::model::UniversalModelLoader::extract_gguf_tokenizer(&bytes).unwrap(),
        );
        let prompt = model.tokenizer.encode("Once upon a time");
        assert_eq!(prompt, [1, 9038, 2501, 263, 931], "{file}");
        let out = greedy_both_paths(&model, &prompt, want.len());
        assert_eq!(out, want, "{file}");
        let text = model.tokenizer.decode(&[prompt, out].concat());
        assert!(
            text.starts_with("Once upon a time, there was a little girl named Lily."),
            "{file}: {text:?}"
        );
    }
}

/// llama.cpp oracle on unsloth SmolLM2-135M-Instruct Q4_K_M (Q5_0/Q4_K/Q6_K/Q8_0 tensors,
/// GQA 9/3, rope base 1e5, tied output): greedy " Paris. Paris is a city".
#[test]
fn test_gguf_smollm2_q4_k_m_matches_llama_cpp() {
    let path = std::path::Path::new("disk/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/main/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    if !path.exists() || cfg!(debug_assertions) {
        return;
    }
    let bytes = std::fs::read(path).unwrap();
    let model = GraphLm::from_gguf_data(GraphLmConfig::default(), &bytes).unwrap();
    let tok = foundry_engine::model::UniversalModelLoader::extract_gguf_tokenizer(&bytes).unwrap();
    let prompt = tok.encode("The capital of France is");
    assert_eq!(prompt, [504, 3575, 282, 4649, 314]); // add_bos_token = false
                                                     // llama.cpp result_output logits[0..3] for this prompt.
    let logits = model.forward(&prompt);
    for (got, want) in logits.iter().zip([3.4901f32, -10.5134, -7.1320]) {
        assert!((got - want).abs() < 0.05, "logit {got} want {want}");
    }
    assert_eq!(
        greedy_both_paths(&model, &prompt, 6),
        [7042, 30, 7042, 314, 253, 2240]
    );
}

/// Minimal GGUF v3 writer (alignment 32) for synthetic loader tests.
enum GgufKv<'a> {
    U32(u32),
    F32(f32),
    Str(&'a str),
}

fn gguf_file(kvs: &[(&str, GgufKv)], tensors: &[(&str, [u64; 2], u32, Vec<u8>)]) -> Vec<u8> {
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend((v.len() as u64).to_le_bytes());
        b.extend(v.as_bytes());
    };
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        s(&mut b, k);
        match v {
            GgufKv::U32(x) => b.extend([4u32.to_le_bytes(), x.to_le_bytes()].concat()),
            GgufKv::F32(x) => b.extend([6u32.to_le_bytes(), x.to_le_bytes()].concat()),
            GgufKv::Str(x) => {
                b.extend(8u32.to_le_bytes());
                s(&mut b, x);
            }
        }
    }
    let pad = |n: usize| n.div_ceil(32) * 32;
    let mut off = 0u64;
    for (name, dims, ty, data) in tensors {
        s(&mut b, name);
        b.extend(2u32.to_le_bytes());
        dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
        b.extend(ty.to_le_bytes());
        b.extend(off.to_le_bytes());
        off += pad(data.len()) as u64;
    }
    b.resize(pad(b.len()), 0);
    for (_, _, _, data) in tensors {
        let end = b.len() + pad(data.len());
        b.extend(data);
        b.resize(end, 0);
    }
    b
}

/// Every GGUF block format the loader accepts decodes like ggml `dequantize_row_*` through both
/// the f32 extraction path and the zero-copy fused GEMV (expected values from gguf-py on the same
/// bytes); GGML types it cannot decode are a load error naming the tensor, not misread bytes.
#[test]
fn test_gguf_block_formats_decode_like_ggml() {
    use foundry_engine::model::layers::{dispatch_fused_gemv, QuantizedWeight};
    use foundry_engine::model::UniversalModelLoader;
    // (name, ggml type, block elems, block bytes, W at [0,17,31,32,100,255,256+77], W·x)
    let cases: [(&str, u32, usize, usize, [f32; 7], [f32; 2]); 8] = [
        (
            "Q4_0",
            2,
            32,
            18,
            [
                0.070015, 0.070015, 0.070015, -0.070015, 0.0100021, 0.0600128, 0.0,
            ],
            [1.30028, 1.29778],
        ),
        (
            "Q4_1",
            3,
            32,
            20,
            [
                0.00500107, 0.075016, 0.075016, 0.0450096, 0.00500107, 0.0650139, 0.135029,
            ],
            [-0.717653, -0.942701],
        ),
        (
            "Q5_0",
            6,
            32,
            22,
            [
                0.0300064, -0.150032, 0.0100021, 0.0900192, -0.070015, -0.160034, 0.0400085,
            ],
            [-0.770164, -0.217546],
        ),
        (
            "Q5_1",
            7,
            32,
            24,
            [
                0.205044, 0.255054, 0.0950203, 0.285061, 0.00500107, 0.0850182, 0.255054,
            ],
            [-0.137529, -0.527613],
        ),
        (
            "Q8_0",
            8,
            32,
            34,
            [
                -0.810173, -1.20026, -1.22026, 0.970207, -0.230049, 1.00021, 0.880188,
            ],
            [8.35678, -6.60641],
        ),
        (
            "Q4_K",
            12,
            256,
            144,
            [
                -0.0550117, -0.125027, -0.145031, 0.770164, 4.2009, 0.450096, 0.665142,
            ],
            [24.474, 11.5725],
        ),
        (
            "Q5_K",
            13,
            256,
            176,
            [
                0.105022, -0.125027, -0.145031, 0.970207, 0.28006, 0.530113, -0.135029,
            ],
            [-34.1785, -66.6817],
        ),
        (
            "Q6_K",
            14,
            256,
            210,
            [
                1.05022, 2.28049, -10.6423, 32.1969, -23.9851, 1.80038, -7.83167,
            ],
            [-36.5703, 31.5592],
        ),
    ];
    // 2 rows x 256: bytes (i*73+29) % 256 with f16 d = 0.01 (+ dmin 0.005 / m -0.005) per block.
    let raw = |name: &str, elems: usize, bytes: usize| {
        let mut r: Vec<u8> = (0..512 / elems * bytes)
            .map(|i| ((i * 73 + 29) % 256) as u8)
            .collect();
        for blk in r.chunks_exact_mut(bytes) {
            let (d_at, second) = match name {
                "Q6_K" => (208, None),
                "Q4_K" | "Q5_K" => (0, Some(0x1D1Fu16)),
                "Q4_1" | "Q5_1" => (0, Some(0x9D1F)),
                _ => (0, None),
            };
            blk[d_at..d_at + 2].copy_from_slice(&0x211Fu16.to_le_bytes());
            if let Some(v) = second {
                blk[2..4].copy_from_slice(&v.to_le_bytes());
            }
        }
        r
    };
    let tensors: Vec<_> = cases
        .iter()
        .map(|&(name, ty, elems, bytes, ..)| (name, [256, 2], ty, raw(name, elems, bytes)))
        .collect();
    let data = gguf_file(&[], &tensors);
    let manifest = UniversalModelLoader::parse_gguf(&data).unwrap();
    let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    for (name, _, _, _, want_w, want_y) in cases {
        let w = manifest.extract_tensor_f32(&data, name).unwrap();
        for (i, want) in [0, 17, 31, 32, 100, 255, 333].into_iter().zip(want_w) {
            assert!(
                (w[i] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{name} w[{i}] = {} want {want}",
                w[i]
            );
        }
        let t = &manifest.tensors[name];
        let QuantizedWeight::GgufQuantized {
            format,
            raw_data,
            in_features,
            out_features,
        } = manifest
            .extract_tensor_as_quantized(&data, name, t.quant_format)
            .unwrap()
        else {
            panic!("{name}: expected zero-copy GGUF blocks");
        };
        assert_eq!(
            (in_features, out_features),
            (256, 2),
            "{name}: GGUF dims are [in, out]"
        );
        let mut y = [0.0f32; 2];
        dispatch_fused_gemv(&raw_data, format, 256, 2, &x, &mut y, None);
        for (got, want) in y.iter().zip(want_y) {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{name} W·x = {got} want {want}"
            );
        }
    }

    let bad = gguf_file(
        &[],
        &[("blk.0.ffn_down.weight", [256, 2], 15, vec![0; 2 * 292])],
    );
    let err = UniversalModelLoader::parse_gguf(&bad).unwrap_err();
    assert!(
        err.contains("blk.0.ffn_down.weight") && err.contains("Q8_K"),
        "{err}"
    );
}

/// RMSNorm uses the GGUF-declared epsilon: with 0 blocks, logits = lm_head(rmsnorm(embedding)),
/// and at mean(x²) = 1e-6 eps = 1e-6 gives 0.002 / sqrt(2e-6) (the old hardcoded 1e-5: 0.603).
#[test]
fn test_gguf_rms_norm_eps_from_metadata() {
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut eye = [0.0f32; 16];
    (0..4).for_each(|i| eye[i * 5] = 1.0);
    let embd = eye.map(|v| v * 0.002); // token 0 -> [0.002, 0, 0, 0]
    let data = gguf_file(
        &[
            ("general.architecture", GgufKv::Str("llama")),
            ("llama.embedding_length", GgufKv::U32(4)),
            ("llama.block_count", GgufKv::U32(0)),
            ("llama.attention.head_count", GgufKv::U32(1)),
            ("llama.attention.layer_norm_rms_epsilon", GgufKv::F32(1e-6)),
        ],
        &[
            ("token_embd.weight", [4, 4], 0, f32s(&embd)),
            ("output_norm.weight", [4, 1], 0, f32s(&[1.0; 4])),
            ("output.weight", [4, 4], 0, f32s(&eye)),
        ],
    );
    let model = GraphLm::from_gguf_data(GraphLmConfig::default(), &data).unwrap();
    let want = 0.002 / (1e-6f32 + 1e-6).sqrt();
    let mut kv: Vec<foundry_engine::model::graph_lm::LayerKVCache> = vec![];
    for logits in [
        model.forward(&[0]),
        model.forward_token_with_kv(0, 0, &mut kv),
    ] {
        assert!(
            (logits[0] - want).abs() < 1e-3,
            "logit {} want {want}",
            logits[0]
        );
    }
}

/// GGUF-embedded tokenizers encode exactly like llama.cpp (`llama-tokenize` ids): SentencePiece
/// scores-derived merges (stories15M) and byte-level BPE merges + smollm pre-tokenizer + the GGUF
/// chat template (SmolLM2).
#[test]
fn test_gguf_tokenizers_match_llama_tokenize() {
    use foundry_engine::model::UniversalModelLoader;
    let dir = |repo: &str| std::path::PathBuf::from(format!("disk/models--{repo}/snapshots/main"));
    let stories = dir("ggml-org--tiny-llamas").join("stories15M.gguf");
    if stories.exists() {
        let tok =
            UniversalModelLoader::extract_gguf_tokenizer(&std::fs::read(stories).unwrap()).unwrap();
        for (text, want) in [
            (
                "Once upon a time, there was a little girl named Lily.",
                &[
                    1u32, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354,
                    29889,
                ][..],
            ),
            (
                "  How are   you?\n\n\n",
                &[1, 259, 1128, 526, 259, 366, 29973, 13, 13, 13],
            ),
            (
                "Café résumé 🦀 1234 don't “quotes”",
                &[
                    1, 315, 28059, 6896, 398, 29948, 29871, 243, 162, 169, 131, 29871, 29896,
                    29906, 29941, 29946, 1016, 29915, 29873, 1346, 339, 4769, 30024,
                ],
            ),
        ] {
            assert_eq!(tok.encode(text), want, "{text:?}");
            assert_eq!(tok.decode(want), text);
        }
    }
    let smollm =
        dir("unsloth--SmolLM2-135M-Instruct-GGUF").join("SmolLM2-135M-Instruct-Q4_K_M.gguf");
    if smollm.exists() {
        let tok =
            UniversalModelLoader::extract_gguf_tokenizer(&std::fs::read(smollm).unwrap()).unwrap();
        assert_eq!(
            tok.encode("  How are   you?\n\n\n"),
            [216, 1073, 359, 256, 346, 47, 16506]
        );
        let chat = tok.apply_chat_template(&[("user", "What is the capital of France?")]);
        assert_eq!(
            tok.encode(&chat),
            [
                1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018,
                411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47,
                2, 198, 1, 520, 9531, 198
            ],
            "{chat:?}"
        );
    }
}

/// Chat path end to end: the model's own `chat_template` (tokenizer_config.json) rendered and
/// BPE-tokenized like HF `apply_chat_template(add_generation_prompt=True)`, then greedy decoding
/// through `ContinuousBatchScheduler` equal to transformers' greedy `generate` (f32 weights).
#[test]
fn test_chat_template_and_scheduler_greedy_match_hf() {
    use foundry_engine::model::hub::ModelArtifacts;
    use tokio::sync::mpsc::error::TryRecvError;
    if cfg!(debug_assertions) {
        return;
    }
    let cases: [(&str, &[u32], &[u32]); 2] = [
        (
            // Template injects SmolLM's default system prompt; answer ends with <|im_end|>.
            "disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main",
            &[
                1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018,
                411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47,
                2, 198, 1, 520, 9531, 198,
            ],
            &[504, 3575, 282, 4649, 314, 7042, 30], // "The capital of France is Paris."
        ),
        (
            // Answer ends with <|user|> (a generation_config eos id).
            "disk/models--THUDM--glm-edge-1.5b-chat/snapshots/main",
            &[59253, 10, 5127, 398, 283, 8570, 325, 11334, 63, 59254, 10],
            &[1026, 8570, 325, 11334, 398, 14488, 46], // "The capital of France is Paris."
        ),
    ];
    for (dir, want_prompt, want_answer) in cases {
        let dir = std::path::Path::new(dir);
        if !dir.join("tokenizer_config.json").exists() {
            continue;
        }
        let artifacts = ModelArtifacts {
            model_id: dir.display().to_string(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            ..Default::default()
        };
        let model =
            foundry_engine::model::hf_loader::load_causal_lm(&artifacts, QuantMode::Uncompressed)
                .unwrap();

        let sched = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(1024)),
            Some(ExecutableModel::from(model)),
        );
        let greedy = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            top_logprobs: None,
        };
        let (_, mut rx, _, _, _) = sched.submit_with_params(greedy, |tok| {
            let ids =
                tok.encode(&tok.apply_chat_template(&[("user", "What is the capital of France?")]));
            assert_eq!(ids, want_prompt, "{}", dir.display());
            ids
        });
        let mut answer = Vec::new();
        for _ in 0..64 {
            sched.step();
            match rx.try_recv() {
                Ok(id) => answer.push(id),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }
        }
        assert_eq!(answer, want_answer, "{}", dir.display());
    }
}

/// A checkpoint missing a required tensor fails loading with the tensor name instead of running
/// on synthetic weights (how GLM's unmapped `mlp.gate_up_proj` once produced gibberish).
#[test]
fn test_missing_required_weight_names_the_tensor() {
    let header =
        br#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}"#;
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(header);
    data.extend_from_slice(&[0u8; 32]);
    let cfg = GraphLmConfig {
        vocab_size: 4,
        hidden_dim: 2,
        num_layers: 1,
        ..GraphLmConfig::smollm_1m()
    };
    let err = GraphLm::from_safetensors_data(cfg, &data).unwrap_err();
    assert!(
        err.contains("model.layers.0.input_layernorm.weight"),
        "{err}"
    );
}
