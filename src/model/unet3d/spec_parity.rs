//! `arch/families/modelscope_t2v.json5` vs the hand-written UNet3D on the `diffb` golden geometry
//! (widths [64, 128, 128, 128], cross width 96, transformer_in 128; 2 frames of 8x8, 3 tokens).

use super::*;
use crate::arch::expr::Val;
use crate::arch::testing::{compare, Runner, Weights as W};
use crate::graph::Tensor;
use crate::hardware::Device;

#[test]
fn unet3d_spec_matches_old_forward() {
    let cfg = Cfg { in_ch: 4, block_out: [64, 128, 128, 128], ctx_dim: 96, tin_width: 128 };
    let dir = std::env::temp_dir().join(format!("gibbs-unet3d-spec-{}", std::process::id()));
    let mut w = W::new(9);
    for (n, s) in fixture::specs(cfg) {
        if s.len() == 1 && n.ends_with("weight") {
            w.norm(n, s[0], 1.0);
        } else if s.len() == 1 {
            w.s(n, &s, 0.1);
        } else {
            // fan-in scale over every dim but the first (conv kernels included).
            let fan: usize = s[1..].iter().product();
            w.s(n, &s, 1.0 / (fan as f32).sqrt());
        }
    }
    w.write_checkpoint(
        &dir,
        &serde_json::json!({
            "_class_name": "UNet3DConditionModel", "in_channels": 4, "out_channels": 4,
            "block_out_channels": cfg.block_out, "cross_attention_dim": 96, "attention_head_dim": 64,
            "layers_per_block": 2, "norm_num_groups": 32,
        }),
        4,
    );
    let old = Unet3d::from_st(St::open(&dir.join("model.safetensors"), cfg).unwrap()).unwrap();
    let (frames, h, wd, nt) = (2usize, 8usize, 8usize, 3usize);
    let mut r = crate::arch::testing::Rng(16);
    let x: Vec<f32> = (0..frames * 4 * h * wd).map(|_| r.unit()).collect();
    let ctx: Vec<f32> = (0..nt * 96).map(|_| r.unit()).collect();
    let _cpu = crate::hardware::cpu_compute::allow();
    let num = |v: usize| Val::Num(v as f64);
    let vars = [("frames", num(frames)), ("h", num(h)), ("w", num(wd)), ("nt", num(nt))];
    for dev in [Device::Cpu, Device::Metal] {
        if dev == Device::Metal && !crate::hardware::available_devices().contains(&Device::Metal) {
            continue;
        }
        let mut run = Runner::load(&dir, Some("modelscope_t2v"), dev, 0).unwrap();
        for t in [900.0f32, 400.0] {
            let want = old.forward_cpu(&x, frames, h, wd, t, &ctx, nt).unwrap();
            let temb = crate::model::unet::timestep_embedding(t, 64);
            let ins = vec![Tensor::F32(x.clone()), Tensor::F32(temb), Tensor::F32(ctx.clone())];
            let got = run.run("denoiser", &vars, ins).unwrap().remove("eps").unwrap().into_f32().unwrap();
            let (e, c) = compare(&got, &want);
            eprintln!("{dev:?} t={t}: rel {e:.2e} cos {c:.8}");
            assert!(e < 1e-3 && c > 0.9999, "{dev:?} t={t}: rel {e} cos {c}");
        }
        assert!(run.unconsumed().is_empty(), "{:?}", run.unconsumed());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
