use foundry_engine::hardware::{QuantFormat, QuantMode};
use foundry_engine::model::layers::QuantizedLinearLayer;
use foundry_engine::model::nano::{NanoTransformerConfig, NanoTransformerModel};
use foundry_engine::quant::exl3::{
    dequantize_exl3_tensor, gemv_exl3_f32, quantize_exl3_tensor, Exl3BitRate,
};
use foundry_engine::quant::nvfp4::{
    dequantize_nvfp4_tensor, gemv_nvfp4_f32, quantize_nvfp4_tensor,
};
use gibbs_engine::foundry_engine;

#[test]
fn test_nvfp4_e2m1_quantization_and_fused_gemv() {
    let rows = 32;
    let cols = 32;
    let mut weights = vec![0.0f32; rows * cols];
    for (i, w) in weights.iter_mut().enumerate() {
        *w = ((i as f32 * 0.1).sin() * 4.0).clamp(-6.0, 6.0);
    }

    let tensor = quantize_nvfp4_tensor(&weights, rows, cols).expect("NVFP4 quantization failed");
    assert_eq!(tensor.rows, rows);
    assert_eq!(tensor.cols, cols);

    // Test dequantize
    let mut dequant = vec![0.0f32; rows * cols];
    dequantize_nvfp4_tensor(&tensor, &mut dequant);
    assert_eq!(dequant.len(), weights.len());

    // Test fused GEMV against reference
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).cos()).collect();
    let mut out_gemv = vec![0.0f32; rows];
    gemv_nvfp4_f32(&tensor, &x, &mut out_gemv);

    let mut out_ref = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += dequant[r * cols + c] * x[c];
        }
        out_ref[r] = sum;
    }

    for (g, r) in out_gemv.iter().zip(&out_ref) {
        assert!((g - r).abs() < 1e-4, "GEMV output mismatch: {} vs {}", g, r);
    }
}

#[test]
fn test_exl3_all_bitrates_and_fused_gemv() {
    let rows = 32;
    let cols = 32;
    let mut weights = vec![0.0f32; rows * cols];
    for (i, w) in weights.iter_mut().enumerate() {
        *w = (i as f32 * 0.2).cos() * 2.5;
    }

    for rate in [Exl3BitRate::Bits3, Exl3BitRate::Bits4, Exl3BitRate::Bits5] {
        let tensor = quantize_exl3_tensor(&weights, rows, cols, rate).expect("EXL3 quant failed");
        let mut dequant = vec![0.0f32; rows * cols];
        dequantize_exl3_tensor(&tensor, &mut dequant);

        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut out_gemv = vec![0.0f32; rows];
        gemv_exl3_f32(&tensor, &x, &mut out_gemv);

        let mut out_ref = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += dequant[r * cols + c] * x[c];
            }
            out_ref[r] = sum;
        }

        for (g, r) in out_gemv.iter().zip(&out_ref) {
            assert!((g - r).abs() < 1e-4, "EXL3 {:?} GEMV mismatch", rate);
        }
    }
}

#[test]
fn test_quantized_linear_layer_cross_format() {
    let in_feat = 64;
    let out_feat = 32;
    let weights: Vec<f32> = (0..(in_feat * out_feat))
        .map(|i| (i as f32 * 0.05).sin())
        .collect();

    let x: Vec<f32> = (0..in_feat).map(|i| (i as f32 * 0.02).cos()).collect();

    // 1. Uncompressed
    let layer_uncomp =
        QuantizedLinearLayer::from_f32("uncomp", in_feat, out_feat, &weights, QuantFormat::FP32)
            .unwrap();
    let mut out_uncomp = vec![0.0f32; out_feat];
    layer_uncomp.forward(&x, &mut out_uncomp);
    assert_eq!(layer_uncomp.bits_per_weight(), 32.0); // stored as f32

    // 2. NVFP4
    let layer_nvfp4 = QuantizedLinearLayer::from_f32(
        "nvfp4",
        in_feat,
        out_feat,
        &weights,
        QuantFormat::NVFP4_E2M1,
    )
    .unwrap();
    let mut out_nvfp4 = vec![0.0f32; out_feat];
    layer_nvfp4.forward(&x, &mut out_nvfp4);
    assert_eq!(layer_nvfp4.bits_per_weight(), 4.25);

    // 3. EXL3 4-Bit
    let layer_exl3 =
        QuantizedLinearLayer::from_f32("exl3", in_feat, out_feat, &weights, QuantFormat::EXL3_4Bit)
            .unwrap();
    let mut out_exl3 = vec![0.0f32; out_feat];
    layer_exl3.forward(&x, &mut out_exl3);
    assert_eq!(layer_exl3.bits_per_weight(), 4.125);

    // Outputs should correlate positively with uncompressed
    let dot_nvfp4: f32 = out_uncomp.iter().zip(&out_nvfp4).map(|(u, n)| u * n).sum();
    assert!(
        dot_nvfp4 > 0.0,
        "NVFP4 output should positively correlate with reference"
    );
}

#[test]
fn test_nano_model_all_quant_modes_forward() {
    let cfg = NanoTransformerConfig {
        vocab_size: 256,
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        eps: 1e-5,
    };

    let mut model = NanoTransformerModel::new(cfg);

    for mode in [
        QuantMode::Uncompressed,
        QuantMode::NVFP4Ultra,
        QuantMode::EXL3Aggressive,
        QuantMode::EXL3Balanced,
        QuantMode::EXL3Quality,
        QuantMode::AdaptiveHybrid,
        QuantMode::TurboSqueeze,
    ] {
        model.quantize_model(mode);
        let tok = model.forward_token(42, 0);
        assert!(
            tok < 256,
            "Mode {:?} produced out of bounds token {}",
            mode,
            tok
        );
    }
}

#[test]
fn test_hadamard_needle_and_napier_factorization() {
    use foundry_engine::quant::needle::{
        fast_hadamard_transform_f32, hadamard_attention_f32, hadamard_mlp_f32, napier_exp2_f32,
        napier_factorize_dot, napier_log2_f32,
    };

    // 1. FWHT Orthogonal Involution (H * H = I)
    let mut vec = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, 5.0, 0.5];
    let orig = vec.clone();
    fast_hadamard_transform_f32(&mut vec);
    assert_eq!(vec.len(), 8);
    // Second application inverts the transform back to original
    fast_hadamard_transform_f32(&mut vec);
    for (a, b) in vec.iter().zip(&orig) {
        assert!(
            (a - b).abs() < 1e-4,
            "FWHT Involution failed: {} vs {}",
            a,
            b
        );
    }

    // 2. Hadamard Attention (Outlier-Free Structured Attention)
    let head_dim = 16;
    let current_len = 8;
    let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.2).sin()).collect();
    let k_cache: Vec<f32> = (0..(current_len * head_dim))
        .map(|i| (i as f32 * 0.3).cos())
        .collect();
    let v_cache: Vec<f32> = (0..(current_len * head_dim))
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let mut out = vec![0.0f32; head_dim];
    hadamard_attention_f32(
        &q,
        &k_cache,
        &v_cache,
        head_dim,
        current_len,
        head_dim,
        &mut out,
    );
    assert_eq!(out.len(), head_dim);
    for &v in &out {
        assert!(!v.is_nan() && !v.is_infinite());
    }

    // 3. Hadamard MLP (O(N log N) butterfly structured layer)
    let x = vec![0.5f32, -0.2, 1.2, -0.8, 0.1, 0.4, -0.5, 0.9];
    let diag = vec![1.0f32; 8];
    let mut mlp_out = vec![0.0f32; 8];
    hadamard_mlp_f32(&x, &diag, &mut mlp_out);
    assert_eq!(mlp_out.len(), 8);
    for &v in &mlp_out {
        assert!(!v.is_nan() && !v.is_infinite());
    }

    // 4. Napier Logarithmic Factorization
    let v1 = [2.0f32, 4.0, 8.0, 16.0];
    let v2 = [0.5f32, 0.25, 0.125, 0.0625];
    // Reference: 2*0.5 + 4*0.25 + 8*0.125 + 16*0.0625 = 1 + 1 + 1 + 1 = 4.0
    let approx = napier_factorize_dot(&v1, &v2);
    assert!(
        (approx - 4.0).abs() < 0.2,
        "Napier dot product approximation discrepancy: got {}, expected ~4.0",
        approx
    );
    assert!((napier_log2_f32(8.0) - 3.0).abs() < 0.05);
    assert!((napier_exp2_f32(3.0) - 8.0).abs() < 0.1);
}

#[test]
fn test_solidsf_jesse_weightless_lut_layer() {
    use foundry_engine::model::layers::{QuantizedLinearLayer, QuantizedWeight};
    use foundry_engine::quant::jesse::WeightlessJesseLayer;

    let in_features = 32;
    let out_features = 16;
    let address_bits = 6;
    let weights: Vec<f32> = (0..(in_features * out_features))
        .map(|i| ((i as f32 * 0.17).cos() * 2.0).clamp(-4.0, 4.0))
        .collect();

    let jesse = WeightlessJesseLayer::from_f32_weights(
        "jesse_layer",
        in_features,
        out_features,
        address_bits,
        &weights,
    );
    assert_eq!(jesse.in_features, in_features);
    assert_eq!(jesse.out_features, out_features);
    assert!(jesse.bits_per_weight() <= 1.0);

    // Zero-multiplication forward pass execution
    let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.05).sin()).collect();
    let mut out = vec![0.0f32; out_features];
    jesse.forward(&x, &mut out);
    assert_eq!(out.len(), out_features);
    assert!(out.iter().any(|&v| v != 0.0));
    for &v in &out {
        assert!(!v.is_nan() && !v.is_infinite());
    }

    // Integration into QuantizedLinearLayer
    let q_layer = QuantizedLinearLayer {
        name: "test_jesse_linear".to_string(),
        in_features,
        out_features,
        weight: QuantizedWeight::WeightlessJesse(jesse),
        bias: None,
    };
    let mut q_out = vec![0.0f32; out_features];
    q_layer.forward(&x, &mut q_out);
    assert_eq!(q_out, out);
}
