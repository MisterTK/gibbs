use foundry_engine::hardware::QuantFormat;
use foundry_engine::model::layers::QuantizedLinearLayer;
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

    // Outputs should correlate positively with uncompressed
    let dot_nvfp4: f32 = out_uncomp.iter().zip(&out_nvfp4).map(|(u, n)| u * n).sum();
    assert!(
        dot_nvfp4 > 0.0,
        "NVFP4 output should positively correlate with reference"
    );
}
