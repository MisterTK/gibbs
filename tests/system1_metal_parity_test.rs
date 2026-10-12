use gibbs_engine::hardware::{available_devices, Device};
use gibbs_engine::model::jev::{BertDims, BertEncoder};

#[test]
fn test_system1_metal_vs_cpu_parity() {
    if !available_devices().contains(&Device::Metal) {
        println!("SKIP: Metal not available");
        return;
    }

    let dims = BertDims {
        hidden_size: 16,
        num_layers: 1,
        num_heads: 2,
        intermediate_size: 32,
        max_positions: 32,
        vocab_size: 64,
        type_vocab_size: 2,
        layer_norm_eps: 1e-5,
    };

    let weight_val = 0.05f32;
    let tensor_fn = |_name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
        let len: usize = shape.iter().product();
        Ok(vec![weight_val; len])
    };

    let cpu_enc = BertEncoder::load(dims.clone(), Device::Cpu, tensor_fn).expect("cpu load");
    let metal_enc = BertEncoder::load(dims, Device::Metal, tensor_fn).expect("metal load");

    let ids = vec![1, 10, 25, 4];
    let pos = vec![0, 1, 2, 3];
    let types = vec![0, 0, 0, 0];
    let cpu_out = cpu_enc.run(&ids, &pos, &types, true).expect("cpu run");
    let metal_out = metal_enc.run(&ids, &pos, &types, true).expect("metal run");

    assert_eq!(cpu_out.len(), 16);
    assert_eq!(metal_out.len(), 16);

    let dot: f32 = cpu_out
        .iter()
        .zip(metal_out.iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let norm_c: f32 = cpu_out.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let norm_m: f32 = metal_out.iter().map(|&b| b * b).sum::<f32>().sqrt();
    let cos = dot / (norm_c * norm_m).max(1e-9);

    println!("System 1 CPU vs Metal cosine similarity: {cos:.6}");
    assert!(
        cos > 0.999,
        "Cosine similarity {cos} is below threshold 0.999"
    );
}
