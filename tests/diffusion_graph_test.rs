//! A toy diffusion family spec run through `runtime::diffusion::Net` + `runtime::sched`: a
//! conditioned denoiser network stepped by the flow Euler scheduler, then a convolutional VAE
//! decoder network, on the CPU and (when present) Metal, which must agree.

use gibbs_engine::arch::expr::Val;
use gibbs_engine::arch::{load, Registry};
use gibbs_engine::graph::Tensor;
use gibbs_engine::hardware::{available_devices, Device, PlacementMode};
use gibbs_engine::model::weight_format::{ReencodeConfig, ReencodePolicy};
use gibbs_engine::runtime::diffusion::{dir_artifacts, Net};
use gibbs_engine::runtime::sched::{euler_step, linspace_sigmas, shifted_sigmas};
use std::path::Path;
use std::sync::Arc;

const SPEC: &str = r#"{
  family: "toy_diffusion", pipeline: "diffusion", match: {hf_model_type: ["toy_diffusion"]},
  components: {
    denoiser: {
      vars: ["c", "side", "nt"],
      inputs: [{name: "x", dtype: "f32", shape: ["c * side * side"]},
               {name: "context", dtype: "f32", shape: ["nt", 16]},
               {name: "t_emb", dtype: "f32", shape: [16], gen: "timestep_embedding", args: {t: "step:sigma * 1000", dim: 16}}],
      body: [
        {op: "Reduce", args: {kind: "Mean", axes: [0], keep_dims: false}, in: ["i:context"], out: "cond"},
        {op: "Binary", args: {kind: "Add"}, in: ["cond", "i:t_emb"], out: "cond"},
        {op: "Unsqueeze", args: {axes: [0]}, in: ["cond"], out: "cond"},
        {op: "Linear", in: ["cond", "w:cond_proj.weight"], out: "proj"},
        {op: "Reshape", args: {shape: ["c * side * side"]}, in: ["proj"], out: "proj"},
        {op: "Binary", args: {kind: "Add"}, in: ["i:x", "proj"], out: "h"},
        {op: "Unary", args: {kind: "Silu"}, in: ["h"], out: "h"},
        {op: "Unsqueeze", args: {axes: [0]}, in: ["h"], out: "h"},
        {op: "Linear", in: ["h", "w:out_proj.weight"], out: "v"},
        {op: "Reshape", args: {shape: ["c * side * side"]}, in: ["v"], out: "v"},
      ],
      outputs: ["v"],
    },
    decoder: {
      vars: ["c", "side"],
      inputs: [{name: "z", dtype: "f32", shape: ["c", "side", "side"]}],
      body: [{op: "Conv2d", args: {stride: [1, 1], padding: [0, 0, 0, 0], dilation: [1, 1], groups: 1},
              in: ["i:z", "w:vae.conv.weight", "w:vae.conv.bias"], out: "img"}],
      outputs: ["img"],
    },
  },
}"#;

/// A minimal F32 safetensors file.
fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
    for (name, shape, v) in tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

fn run(dir: &Path, dev: Device) -> Vec<f32> {
    let (c, side, nt) = (4usize, 8usize, 3usize);
    let reg = Registry::from_files(&[("families/toy_diffusion.json5".into(), SPEC.into())], &[]).unwrap();
    let reencode = ReencodeConfig { policy: ReencodePolicy::BitExact, ..Default::default() };
    let m = Arc::new(load::load_with(&dir_artifacts(dir), &reg, None, reencode).unwrap());
    let den = Net::new(m.clone(), "denoiser", PlacementMode::Pin(dev));
    let dec = Net::new(m.clone(), "decoder", PlacementMode::Pin(dev));
    let ctx: Vec<f32> = (0..nt * 16).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
    let mut x = vec![0.5f32; c * side * side];
    let sigmas = shifted_sigmas(&linspace_sigmas(4), 3.0);
    let vars = [("c", Val::Num(c as f64)), ("side", Val::Num(side as f64)), ("nt", Val::Num(nt as f64))];
    for i in 0..4 {
        let given = [("x", Tensor::F32(x.clone())), ("context", Tensor::F32(ctx.clone()))];
        let v = den.run_one(&vars, &given, &[("sigma", sigmas[i] as f64)], None).unwrap();
        assert_eq!(v.len(), x.len());
        euler_step(&mut x, &v, sigmas[i], sigmas[i + 1]);
    }
    let img = dec
        .run_one(&vars[..2], &[("z", Tensor::F32(x))], &[], None)
        .unwrap();
    assert_eq!(img.len(), 3 * side * side);
    assert!(gibbs_engine::runtime::diffusion::unconsumed(&m, &[&den, &dec], &[]).is_empty());
    img
}

#[test]
fn toy_spec_denoising_loop_and_vae_decode() {
    let dir = std::env::temp_dir().join(format!("gibbs-toy-diffusion-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let w = |n: usize, s: f32| (0..n).map(|i| ((i * 7919 % 101) as f32 / 101.0 - 0.5) * s).collect::<Vec<f32>>();
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            ("cond_proj.weight", vec![256, 16], w(256 * 16, 0.1)),
            ("out_proj.weight", vec![256, 256], w(256 * 256, 0.05)),
            ("vae.conv.weight", vec![3, 4, 1, 1], w(12, 0.5)),
            ("vae.conv.bias", vec![3], vec![0.1, -0.1, 0.0]),
        ],
    );
    std::fs::write(dir.join("config.json"), r#"{"model_type": "toy_diffusion"}"#).unwrap();
    let cpu = run(&dir, Device::Cpu);
    assert!(cpu.iter().all(|v| v.is_finite()));
    // On macOS the Metal leg is required (no silent CPU-only pass).
    assert!(
        !cfg!(target_os = "macos") || available_devices().contains(&Device::Metal),
        "no Metal device on macOS"
    );
    if available_devices().contains(&Device::Metal) {
        let gpu = run(&dir, Device::Metal);
        let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("Metal vs CPU max diff {worst}");
        assert!(worst < 1e-4, "Metal vs CPU max diff {worst}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
