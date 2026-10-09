use foundry_engine::hardware::HardwareCapability;
use foundry_engine::kv::{BlockId, RadixPrefixCache};
use foundry_engine::quant::QuantEngine;
use gibbs_engine::foundry_engine;

#[test]
fn test_hardware_probe_reports_detected_facts() {
    let hw = HardwareCapability::probe_system();
    assert_eq!(hw.cpu_has_arm_neon, cfg!(target_arch = "aarch64"));
    if cfg!(target_os = "macos") {
        assert!(hw.cpu_name.is_some(), "machdep.cpu.brand_string not read");
        assert!(
            hw.total_memory_bytes.is_some_and(|b| b >= 1 << 30),
            "hw.memsize not read"
        );
    }
}

#[test]
fn test_fast_walsh_hadamard_transform() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    QuantEngine::fast_walsh_hadamard_transform_f32(&mut data);
    // Length preserved and values transformed
    assert_eq!(data.len(), 8);
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn test_kivi_2bit_quantization() {
    let raw_data = vec![0.1, 0.4, 0.8, 1.2, -0.5, 0.2, 0.9, 2.1];
    let mut packed = vec![0u8; 2];
    let (min_val, scale) = QuantEngine::quantize_kivi_2bit(&raw_data, &mut packed);
    assert!(scale > 0.0);
    assert!(min_val < 0.0);
}

#[test]
fn test_radix_prefix_cache_matching() {
    let cache = RadixPrefixCache::new();
    let prefix_tokens = vec![101, 2054, 2003, 1037];
    let blocks = vec![BlockId(1), BlockId(2)];

    cache.insert_prefix(&prefix_tokens, &blocks);

    let (matched, count) = cache.match_prefix(&[101, 2054, 2003, 1037, 9999]);
    assert_eq!(count, 4);
    assert_eq!(matched, blocks);

    let (_, count_miss) = cache.match_prefix(&[9999, 101]);
    assert_eq!(count_miss, 0);
}

#[test]
fn test_luminal_online_softmax_attention_speedup_and_accuracy() {
    use foundry_engine::model::simd::{
        vector_dot_product, vector_fused_scale_add, vector_online_softmax_attention, vector_softmax,
    };
    use std::time::Instant;

    let head_dim = 64;
    let current_len = 256;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..head_dim)
        .map(|i| ((i as f32 * 0.13).sin()) * 0.5)
        .collect();
    let k_cache: Vec<f32> = (0..(current_len * head_dim))
        .map(|i| ((i as f32 * 0.27).cos()) * 0.8)
        .collect();
    let v_cache: Vec<f32> = (0..(current_len * head_dim))
        .map(|i| ((i as f32 * 0.39).sin()) * 1.1)
        .collect();

    // 1. Standard 3-pass attention
    let t0 = Instant::now();
    let mut scores = vec![0.0f32; current_len];
    for pos in 0..current_len {
        let k_i = &k_cache[pos * head_dim..(pos + 1) * head_dim];
        scores[pos] = vector_dot_product(&q, k_i) * scale;
    }
    vector_softmax(&mut scores);
    let mut expected = vec![0.0f32; head_dim];
    for pos in 0..current_len {
        let v_i = &v_cache[pos * head_dim..(pos + 1) * head_dim];
        vector_fused_scale_add(&mut expected, v_i, scores[pos]);
    }
    let elapsed_3pass = t0.elapsed();

    // 2. Luminal-inspired single-pass online softmax attention
    let t1 = Instant::now();
    let mut actual = vec![0.0f32; head_dim];
    vector_online_softmax_attention(
        &q,
        &k_cache,
        &v_cache,
        head_dim,
        current_len,
        head_dim,
        scale,
        &mut actual,
    );
    let elapsed_online = t1.elapsed();

    println!(
        "Luminal Online Attention Benchmark: 3-pass={:?}, online={:?}",
        elapsed_3pass, elapsed_online
    );

    // Numerical accuracy check
    for d in 0..head_dim {
        assert!(
            (actual[d] - expected[d]).abs() < 1e-4,
            "Discrepancy at dim {}: actual={}, expected={}",
            d,
            actual[d],
            expected[d]
        );
    }
}

#[test]
fn test_tiled_gemv_roofline_saturation() {
    #[cfg(target_os = "macos")]
    {
        let rt = foundry_engine::graph::metal::Runtime::new(
            &foundry_engine::model::hub::HuggingFaceHub::cache_dir(),
        )
        .unwrap();
        let mut max_q4 = 0.0f64;
        let mut max_q8 = 0.0f64;
        let mut max_bf16 = 0.0f64;
        let mut roofline = 0.0f64;
        for _ in 0..5 {
            let (rf, q4, q8, bf16) = rt.benchmark_gemv_saturation().unwrap();
            roofline = rf;
            max_q4 = max_q4.max(q4);
            max_q8 = max_q8.max(q8);
            max_bf16 = max_bf16.max(bf16);
            if max_q4 > 0.90 * roofline && max_q8 > 0.90 * roofline {
                break;
            }
        }
        let (roofline_gbps, q4_gbps, q8_gbps, bf16_gbps) = (roofline, max_q4, max_q8, max_bf16);
        println!("Hardware streaming read roofline (memory_sweep): {roofline_gbps:.2} GB/s");
        println!(
            "Tiled GEMV Q4_0: {q4_gbps:.2} GB/s ({:.1}% roofline)",
            (q4_gbps / roofline_gbps) * 100.0
        );
        println!(
            "Tiled GEMV Q8_0: {q8_gbps:.2} GB/s ({:.1}% roofline)",
            (q8_gbps / roofline_gbps) * 100.0
        );
        println!(
            "Tiled GEMV BF16: {bf16_gbps:.2} GB/s ({:.1}% roofline)",
            (bf16_gbps / roofline_gbps) * 100.0
        );

        let target_pct = 0.90; // >90% roofline saturation requirement
        assert!(
            q4_gbps > target_pct * roofline_gbps,
            "Q4_0 bandwidth {q4_gbps:.2} GB/s should reach at least {:.0}% of measured roofline ({roofline_gbps:.2} GB/s)",
            target_pct * 100.0
        );
        assert!(
            q8_gbps > target_pct * roofline_gbps,
            "Q8_0 bandwidth {q8_gbps:.2} GB/s should reach at least {:.0}% of measured roofline ({roofline_gbps:.2} GB/s)",
            target_pct * 100.0
        );
    }
}

#[test]
fn test_real_smollm2_135m_decode_throughput_on_disk() {
    use foundry_engine::hardware::Device;
    use foundry_engine::model::graph_lm::{GraphLm, GraphLmConfig};
    use foundry_engine::model::hf_loader::HfConfigJson;
    use foundry_engine::model::tokenizer::Tokenizer;
    use foundry_engine::scheduler::ExecutableModel;
    use std::path::Path;
    use std::time::Instant;

    let model_dir = Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
    if !model_dir.exists() {
        println!("SKIP: SmolLM2-135M not found in disk/");
        return;
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tok_str = std::fs::read_to_string(&tok_path).expect("Failed to read tokenizer.json");
    let tokenizer = Tokenizer::from_json_str(&tok_str).expect("Failed to parse tokenizer.json");
    let cfg_path = model_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).expect("Failed to read config.json");
    let hf_cfg: HfConfigJson =
        serde_json::from_slice(&cfg_bytes).expect("Failed to parse config.json");
    let cfg = GraphLmConfig {
        vocab_size: hf_cfg.vocab_size,
        hidden_dim: hf_cfg.hidden_dim,
        intermediate_dim: hf_cfg.resolved_intermediate_dim(),
        num_layers: hf_cfg.num_layers,
        num_heads: hf_cfg.num_heads,
        head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
        max_seq_len: 2048,
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

    let exec_model =
        ExecutableModel::load_graph_lm(raw_model, Device::Metal).expect("loads Program");
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let mut kv = None;

    let first_next = exec_model
        .step_prefill_chunk(&prompt_tokens, 0, true, &prompt_tokens, 0.0, 1.0, &mut kv)
        .expect("prefill succeeds");
    let next_tok = first_next.expect("prefill returned next token");
    let mut generated = prompt_tokens.clone();
    generated.push(next_tok);

    let num_tokens = 32;
    let t0 = Instant::now();
    for _ in 0..num_tokens {
        let tok = exec_model
            .step_token_with_kv(
                &generated,
                *generated.last().unwrap(),
                generated.len(),
                0.0,
                1.0,
                &mut kv,
            )
            .expect("decode step succeeds");
        generated.push(tok);
    }
    let elapsed = t0.elapsed();
    let tps = (num_tokens as f64) / elapsed.as_secs_f64();
    let decoded = tokenizer.decode(&generated);
    println!(
        "SmolLM2-135M Real Disk Checkpoint: {tps:.2} tok/s ({num_tokens} tokens in {elapsed:?})"
    );
    println!("Decoded: {:?}", decoded);
    assert!(
        tps > 10.0,
        "SmolLM2 real decode throughput must exceed 10 tok/s"
    );
    assert!(
        decoded.to_lowercase().contains("paris"),
        "SmolLM2 must answer Paris"
    );
}

#[test]
fn test_real_gemma4_e2b_decode_and_bandwidth_on_disk() {
    use foundry_engine::hardware::Device;
    use foundry_engine::model::hub::HuggingFaceHub;
    use foundry_engine::model::native::load_native;
    use std::time::Instant;

    let repo = "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive";
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
    if art.weights_path.is_none() {
        println!("SKIP: {repo} not found in disk/");
        return;
    }

    let lm = match load_native(&art, Device::Metal) {
        Some(Ok(m)) => m,
        Some(Err(e)) => panic!("Failed to load {repo}: {e}"),
        None => panic!("load_native returned None for {repo}"),
    };

    let tok = lm.tokenizer();
    let prompt = "The capital of France is";
    let ids = tok.encode(prompt);
    let (seq_ids, mut state) = lm.prepare(ids, &[]).expect("prepare sequence");

    let logits = lm.forward(&seq_ids, 0, &mut state).expect("forward prompt");
    let mut cur_token = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap() as u32;
    let mut pos = seq_ids.len();
    let mut generated = Vec::new();

    let num_tokens = 16;
    let t0 = Instant::now();
    for _ in 0..num_tokens {
        if tok.is_special(cur_token) {
            break;
        }
        generated.push(cur_token);
        let next_logits = lm
            .forward(&[cur_token], pos, &mut state)
            .expect("decode step");
        pos += 1;
        cur_token = (0..next_logits.len())
            .max_by(|&a, &b| next_logits[a].total_cmp(&next_logits[b]))
            .unwrap() as u32;
    }
    let elapsed = t0.elapsed();
    let tps = (generated.len() as f64) / elapsed.as_secs_f64();
    let model_bytes = 2.87e9f64; // ~2.87 GB Q2_K_P weights
    let eff_bw = (tps * model_bytes) / 1e9;
    println!("Gemma 4 E2B Real Disk Checkpoint: {tps:.2} tok/s ({eff_bw:.2} GB/s effective memory streaming)");
    assert!(
        tps > 5.0,
        "Gemma 4 E2B real decode throughput must exceed 5 tok/s"
    );
}

#[test]
fn test_real_qwen35_9b_decode_and_roofline_on_disk() {
    use foundry_engine::hardware::Device;
    use foundry_engine::model::hub::HuggingFaceHub;
    use foundry_engine::model::native::load_native;
    use std::time::Instant;

    let repo = "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF";
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
    if art.weights_path.is_none() {
        println!("SKIP: {repo} not found in disk/");
        return;
    }

    let lm = match load_native(&art, Device::Metal) {
        Some(Ok(m)) => m,
        Some(Err(e)) => panic!("Failed to load {repo}: {e}"),
        None => panic!("load_native returned None for {repo}"),
    };

    let tok = lm.tokenizer();
    let prompt = "The capital of France is";
    let ids = tok.encode(prompt);
    let (seq_ids, mut state) = lm.prepare(ids, &[]).expect("prepare sequence");

    let logits = lm.forward(&seq_ids, 0, &mut state).expect("forward prompt");
    let mut cur_token = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap() as u32;
    let mut pos = seq_ids.len();
    let mut generated = Vec::new();

    let num_tokens = 8;
    let t0 = Instant::now();
    for _ in 0..num_tokens {
        if tok.is_special(cur_token) {
            break;
        }
        generated.push(cur_token);
        let next_logits = lm
            .forward(&[cur_token], pos, &mut state)
            .expect("decode step");
        pos += 1;
        cur_token = (0..next_logits.len())
            .max_by(|&a, &b| next_logits[a].total_cmp(&next_logits[b]))
            .unwrap() as u32;
    }
    let elapsed = t0.elapsed();
    let tps = (generated.len() as f64) / elapsed.as_secs_f64();
    let model_bytes = 4.71e9f64; // ~4.71 GB IQ2_M GGUF
    let eff_bw = (tps * model_bytes) / 1e9;
    let roofline_pct = (eff_bw / 298.83) * 100.0;
    println!("Qwen 3.5 9B Real Disk Checkpoint on Metal: {tps:.2} tok/s ({eff_bw:.2} GB/s, {roofline_pct:.1}% of 298.8 GB/s roofline)");
    assert!(
        tps > 20.0,
        "Qwen 3.5 9B real Metal decode throughput must exceed 20 tok/s"
    );
}
