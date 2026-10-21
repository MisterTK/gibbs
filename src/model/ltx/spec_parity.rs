//! `arch/families/ltx.json5` vs the hand-written LTX transformer (host forward and Metal program)
//! on the `diffb` golden geometry: original names, 2 blocks, d 128, 3x3x4 grid, three clean splits.

use super::*;
use crate::arch::expr::Val;
use crate::arch::testing::{compare, Runner, Weights};
use crate::graph::Tensor;
use crate::hardware::Device;

fn fixture(dir: &Path) -> PathBuf {
    const D: usize = 128;
    let mut w = Weights::new(5);
    let lin = |w: &mut Weights, p: &str, o: usize, i: usize| {
        w.w(format!("{p}.weight"), &[o, i]);
        w.s(format!("{p}.bias"), &[o], 0.1);
    };
    lin(&mut w, "patchify_proj", D, LATENT);
    lin(&mut w, "adaln_single.emb.timestep_embedder.linear_1", D, 256);
    lin(&mut w, "adaln_single.emb.timestep_embedder.linear_2", D, D);
    lin(&mut w, "adaln_single.linear", 6 * D, D);
    lin(&mut w, "caption_projection.linear_1", D, 48);
    lin(&mut w, "caption_projection.linear_2", D, D);
    for i in 0..2 {
        let b = format!("transformer_blocks.{i}");
        for a in ["attn1", "attn2"] {
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                lin(&mut w, &format!("{b}.{a}.{n}"), D, D);
            }
            w.norm(format!("{b}.{a}.q_norm.weight"), D, 1.0);
            w.norm(format!("{b}.{a}.k_norm.weight"), D, 1.0);
        }
        lin(&mut w, &format!("{b}.ff.net.0.proj"), 4 * D, D);
        lin(&mut w, &format!("{b}.ff.net.2"), D, 4 * D);
        w.s(format!("{b}.scale_shift_table"), &[6, D], 0.5);
    }
    w.s("scale_shift_table", &[2, D], 0.5);
    lin(&mut w, "proj_out", LATENT, D);
    w.write_checkpoint(dir, &serde_json::json!({}), 4);
    dir.join("model.safetensors")
}

/// The spec's `t_feat` rows: one per span, clean (t = 0) first.
pub(crate) fn t_feat(n: usize, clean: usize, t: f32) -> Vec<f32> {
    [(clean > 0, 0.0), (clean < n, t as f64)]
        .into_iter()
        .filter(|s| s.0)
        .flat_map(|s| crate::model::flux::sinusoid(s.1, TIME_DIM))
        .collect()
}

#[test]
fn ltx_spec_matches_old_forward() {
    let dir = std::env::temp_dir().join(format!("gibbs-ltx-spec-{}", std::process::id()));
    let file = fixture(&dir);
    let old = LtxTransformer::load(&Store::open(&file).unwrap()).unwrap();
    let (f, h, w, nt) = (3usize, 3usize, 4usize, 24usize);
    let n = f * h * w;
    let mut r = crate::arch::testing::Rng(11);
    let x: Vec<f32> = (0..n * LATENT).map(|_| r.unit()).collect();
    let raw: Vec<f32> = (0..nt * 48).map(|_| r.unit()).collect();
    let rope = Rope::new(f, h, w, FPS as f64, 128);
    for dev in [Device::Cpu, Device::Metal] {
        if dev == Device::Metal && !crate::hardware::available_devices().contains(&Device::Metal) {
            continue;
        }
        let _cpu = crate::hardware::cpu_compute::allow();
        let mut run = Runner::load(&dir, Some("ltx"), dev, 0).unwrap();
        for (t, clean) in [(700.0f32, 0usize), (350.0, 5), (700.0, n)] {
            let want = old.forward(&x, n, &old.project_text(&raw, nt), nt, t, clean, &rope);
            let vars = [
                ("frames", Val::Num(f as f64)),
                ("h", Val::Num(h as f64)),
                ("w", Val::Num(w as f64)),
                ("nt", Val::Num(nt as f64)),
                ("clean", Val::Num(clean as f64)),
            ];
            let ins = vec![Tensor::F32(x.clone()), Tensor::F32(raw.clone()), Tensor::F32(t_feat(n, clean, t))];
            let got = run.run("denoiser", &vars, ins).unwrap().remove("v").unwrap().into_f32().unwrap();
            let (e, c) = compare(&got, &want);
            eprintln!("{dev:?} t={t} clean={clean}: rel {e:.2e} cos {c:.8}");
            assert!(e < 1e-3 && c > 0.9999, "{dev:?} t={t} clean={clean}: rel {e} cos {c}");
        }
        assert!(run.unconsumed().is_empty(), "{:?}", run.unconsumed());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
