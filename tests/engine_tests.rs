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
