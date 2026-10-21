//! Diffusion/video goldens, group B: Z-Image, Qwen-Image 1 / 2.1, Krea 2, LTX-Video transformers,
//! the Qwen-Image / Qwen-Image 2.1 / LTX VAE decoders and the ModelScope UNet3D. Each case writes a
//! tiny random checkpoint, loads it with the family's loader and records one forward: the host
//! forward on `Device::Cpu`, the Metal graph program on `Device::Metal`.

use crate::util::*;
use gibbs_engine::hardware::Device;
use gibbs_engine::model::flux::{Loras, Store};
use gibbs_engine::model::weights::Weights as Mw;
use serde_json::{json, Value};

type Spec = Vec<(String, Vec<usize>)>;

fn sp(s: &mut Spec, name: impl Into<String>, shape: &[usize]) {
    s.push((name.into(), shape.to_vec()));
}

/// `{p}.weight [o, i]` (+ `{p}.bias [o]`).
fn lin(s: &mut Spec, p: &str, o: usize, i: usize, bias: bool) {
    sp(s, format!("{p}.weight"), &[o, i]);
    if bias {
        sp(s, format!("{p}.bias"), &[o]);
    }
}

/// Rank-1 `*weight` / `*gamma` are gains 1 ± 0.1, other rank-1 tensors ± 0.1, the rest uniform with
/// variance 1 / fan_in (fan_in = product of all but the first dim).
fn synth(seed: u64, specs: &Spec) -> Vec<T> {
    let mut r = Rng::new(seed);
    specs
        .iter()
        .map(|(n, s)| {
            let len: usize = s.iter().product();
            let v = if s.len() == 1 && (n.ends_with("weight") || n.ends_with("gamma")) {
                r.around(len, 1.0, 0.1)
            } else if s.len() == 1 {
                r.vec(len, 0.1)
            } else {
                let fan: usize = s[1..].iter().product::<usize>().max(1);
                r.vec(len, (3.0 / fan as f32).sqrt())
            };
            (n.clone(), s.clone(), v)
        })
        .collect()
}

fn set(t: &mut [T], name: &str, v: Vec<f32>) {
    let e = t.iter_mut().find(|e| e.0 == name).expect(name);
    assert_eq!(e.2.len(), v.len(), "{name}");
    e.2 = v;
}

fn rnd(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    Rng::new(seed).vec(n, scale)
}

/// Every k-th value, at most 4096.
fn sub(v: &[f32]) -> Vec<f32> {
    v.iter().step_by(v.len().div_ceil(4096)).copied().collect()
}

fn metal(d: Device) -> bool {
    matches!(d, Device::Metal)
}

/// Per-channel latent statistics `(mean, std)`.
fn stats(seed: u64, c: usize) -> (Vec<f32>, Vec<f32>) {
    let mut r = Rng::new(seed);
    let mean = r.vec(c, 0.5);
    let std = (0..c).map(|_| 1.0 + 0.5 * r.unit().abs()).collect();
    (mean, std)
}

/// `x[c, ..] * std[c] + mean[c]` over channel-major planes of `plane` values.
fn denorm(x: &[f32], plane: usize, mean: &[f32], std: &[f32]) -> Vec<f32> {
    x.chunks_exact(plane)
        .enumerate()
        .flat_map(|(c, p)| p.iter().map(move |v| v * std[c] + mean[c]))
        .collect()
}

// ---- spec-path helpers ------------------------------------------------------------------------

use gibbs_engine::arch::expr::Val;
use gibbs_engine::arch::testing::Runner;
use gibbs_engine::graph::Tensor::{self, F32};

fn num(n: usize) -> Val {
    Val::Num(n as f64)
}

/// A spec runner over a fixture directory whose weights are `weights`.
fn spec_runner(dir: &std::path::Path, weights: &str, family: Option<&str>, dev: Device) -> Runner {
    let a = gibbs_engine::model::hub::ModelArtifacts {
        model_id: "test/diffb".into(),
        local_dir: Some(dir.to_path_buf()),
        config_path: Some(dir.join("config.json")).filter(|p| p.is_file()),
        weights_path: Some(dir.join(weights)),
        is_cached: true,
        ..Default::default()
    };
    Runner::from_artifacts(&a, family, dev, 64).unwrap_or_else(|e| panic!("{e}"))
}

/// diffusers `_pack_latents`: [c, h, w] -> [(h/2)*(w/2), 4c] in (c, py, px) order (the runtime's
/// `pack2x2`); `unpack` inverts it.
fn pack2x2(lat: &[f32], c: usize, h: usize, w: usize, unpack: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; lat.len()];
    for i in 0..h / 2 {
        for j in 0..w / 2 {
            for ch in 0..c {
                for py in 0..2 {
                    for px in 0..2 {
                        let p = ((i * (w / 2) + j) * c + ch) * 4 + py * 2 + px;
                        let u = (ch * h + 2 * i + py) * w + 2 * j + px;
                        if unpack { out[u] = lat[p] } else { out[p] = lat[u] }
                    }
                }
            }
        }
    }
    out
}

/// Output `out` of `component` (f32).
fn run1(r: &mut Runner, component: &str, vars: &[(&str, Val)], ins: Vec<Tensor>, out: &str) -> Vec<f32> {
    let mut o = r.run(component, vars, ins).unwrap_or_else(|e| panic!("{e}"));
    match o.remove(out) {
        Some(F32(v)) => v,
        other => panic!("output `{out}`: {other:?}"),
    }
}

/// Z-Image S3-DiT (`ZImageTransformer2DModel`): 2 noise refiners, 2 context refiners, 2 layers,
/// dim 64 / 2 heads, axes [8, 12, 12]; one forward with caption/image padding (21 -> 32 caption,
/// 36 -> 64 patches) and one without.
const ZIMAGE_CASES: [(usize, usize, usize, f32); 2] = [(12, 12, 21, 0.7), (16, 16, 32, 0.25)];

fn zimage_fixture(case: &str) -> std::path::PathBuf {
    const DIM: usize = 64;
    const HD: usize = 32;
    const CAP: usize = 48;
    const IN: usize = 4;
    const ADA: usize = 64;
    const FF: usize = 128;
    let dir = fixture_dir(case);
    let mut s = Spec::new();
    for (fam, modulated) in [
        ("noise_refiner", true),
        ("context_refiner", false),
        ("layers", true),
    ] {
        for i in 0..2 {
            let p = format!("{fam}.{i}");
            if modulated {
                lin(&mut s, &format!("{p}.adaLN_modulation.0"), 4 * DIM, ADA, true);
            }
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                lin(&mut s, &format!("{p}.attention.{n}"), DIM, DIM, false);
            }
            lin(&mut s, &format!("{p}.feed_forward.w1"), FF, DIM, false);
            lin(&mut s, &format!("{p}.feed_forward.w3"), FF, DIM, false);
            lin(&mut s, &format!("{p}.feed_forward.w2"), DIM, FF, false);
            for (n, len) in [
                ("attention.norm_q", HD),
                ("attention.norm_k", HD),
                ("attention_norm1", DIM),
                ("attention_norm2", DIM),
                ("ffn_norm1", DIM),
                ("ffn_norm2", DIM),
            ] {
                sp(&mut s, format!("{p}.{n}.weight"), &[len]);
            }
        }
    }
    lin(&mut s, "t_embedder.mlp.0", 96, 256, true);
    lin(&mut s, "t_embedder.mlp.2", ADA, 96, true);
    lin(&mut s, "all_x_embedder.2-1", DIM, 4 * IN, true);
    sp(&mut s, "cap_embedder.0.weight", &[CAP]);
    lin(&mut s, "cap_embedder.1", DIM, CAP, true);
    lin(&mut s, "all_final_layer.2-1.adaLN_modulation.1", DIM, ADA, true);
    lin(&mut s, "all_final_layer.2-1.linear", 4 * IN, DIM, true);
    sp(&mut s, "x_pad_token", &[1, DIM]);
    sp(&mut s, "cap_pad_token", &[1, DIM]);
    write_st(&dir.join("diffusion_pytorch_model.safetensors"), &synth(1, &s));
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "ZImageTransformer2DModel", "dim": DIM, "n_heads": 2, "n_kv_heads": 2,
            "in_channels": IN, "all_patch_size": [2], "all_f_patch_size": [1], "qk_norm": true,
            "axes_dims": [8, 12, 12], "axes_lens": [64, 64, 64], "cap_feat_dim": CAP,
            "n_refiner_layers": 2, "n_layers": 2, "norm_eps": 1e-5, "t_scale": 1000.0,
            "rope_theta": 256.0
        }),
    );
    dir
}

#[test]
fn zimage() {
    use gibbs_engine::model::zimage::ZDit;
    const CAP: usize = 48;
    const IN: usize = 4;
    let dit = ZDit::load(&zimage_fixture("diffb_zimage")).expect("Z-Image fixture");
    // runner: internal gibbs_engine::model::zimage::ZDit::{forward, forward_gpu}
    golden("diffb_zimage", TOL, |dev| {
        let mut o = Out::default();
        for (k, (h, w, n_cap, sigma)) in ZIMAGE_CASES.into_iter().enumerate() {
            let lat = rnd(10 + k as u64, IN * h * w, 1.0);
            let cap = rnd(20 + k as u64, n_cap * CAP, 1.0);
            let y = if metal(dev) {
                dit.forward_gpu(&lat, h, w, sigma, &cap, n_cap).unwrap()
            } else {
                dit.forward(&lat, h, w, sigma, &cap, n_cap)
            };
            o.f(format!("out_{h}x{w}_cap{n_cap}_s{sigma}"), &y);
        }
        o
    });
}

/// The `zimage` spec's `denoiser` on the same fixture.
#[test]
fn zimage_spec() {
    const CAP: usize = 48;
    const IN: usize = 4;
    let dir = zimage_fixture("diffb_zimage_spec");
    golden("diffb_zimage", TOL, |dev| {
        let mut r = spec_runner(&dir, "diffusion_pytorch_model.safetensors", None, dev);
        let mut o = Out::default();
        for (k, (h, w, n_cap, sigma)) in ZIMAGE_CASES.into_iter().enumerate() {
            let lat = rnd(10 + k as u64, IN * h * w, 1.0);
            let cap = rnd(20 + k as u64, n_cap * CAP, 1.0);
            r.step = vec![("sigma".into(), Val::Num(sigma as f64))];
            let vars = [("h", num(h)), ("w", num(w)), ("nt", num(n_cap))];
            let y = run1(&mut r, "denoiser", &vars, vec![F32(lat), F32(cap)], "out");
            o.f(format!("out_{h}x{w}_cap{n_cap}_s{sigma}"), &y);
        }
        o
    });
}

/// Qwen-Image 1 double-stream MMDiT (`QwenImageTransformer2DModel`): 2 blocks, dim 64 / 2 heads,
/// axes [8, 12, 12], 64 packed latent features, text width 48; an odd 5x7 patch grid, 21 tokens.
#[test]
fn qwen_image_v1() {
    qwen_image_v1_run(false)
}

/// The `qwen_image` spec's `denoiser` on the same fixture.
#[test]
fn qwen_image_v1_spec() {
    qwen_image_v1_run(true)
}

fn qwen_image_v1_run(spec: bool) {
    use gibbs_engine::model::qwen_image_v1::{positions, Dit, Rope};
    const D: usize = 64;
    const HD: usize = 32;
    const FF: usize = 128;
    const TW: usize = 48;
    const IN: usize = 64;
    let dir = fixture_dir(if spec { "diffb_qwen_image_v1_spec" } else { "diffb_qwen_image_v1" });
    let mut s = Spec::new();
    lin(&mut s, "img_in", D, IN, true);
    sp(&mut s, "txt_norm.weight", &[TW]);
    lin(&mut s, "txt_in", D, TW, true);
    lin(&mut s, "time_text_embed.timestep_embedder.linear_1", D, 256, true);
    lin(&mut s, "time_text_embed.timestep_embedder.linear_2", D, D, true);
    for i in 0..2 {
        let p = format!("transformer_blocks.{i}");
        for (m, q, k, v, nq, nk, o, mlp) in [
            (
                "img_mod.1", "attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k",
                "attn.to_out.0", "img_mlp",
            ),
            (
                "txt_mod.1", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
                "attn.norm_added_q", "attn.norm_added_k", "attn.to_add_out", "txt_mlp",
            ),
        ] {
            lin(&mut s, &format!("{p}.{m}"), 6 * D, D, true);
            for n in [q, k, v, o] {
                lin(&mut s, &format!("{p}.{n}"), D, D, true);
            }
            sp(&mut s, format!("{p}.{nq}.weight"), &[HD]);
            sp(&mut s, format!("{p}.{nk}.weight"), &[HD]);
            lin(&mut s, &format!("{p}.{mlp}.net.0.proj"), FF, D, true);
            lin(&mut s, &format!("{p}.{mlp}.net.2"), D, FF, true);
        }
    }
    lin(&mut s, "norm_out.linear", 2 * D, D, true);
    lin(&mut s, "proj_out", IN, D, true);
    let file = dir.join("diffusion_pytorch_model.safetensors");
    write_st(&file, &synth(2, &s));
    let cfg = json!({
        "_class_name": "QwenImageTransformer2DModel", "patch_size": 2, "in_channels": IN,
        "out_channels": 16, "num_layers": 2, "attention_head_dim": HD, "num_attention_heads": 2,
        "joint_attention_dim": TW, "guidance_embeds": false, "axes_dims_rope": [8, 12, 12]
    });
    write_json(&dir.join("config.json"), &cfg);
    let dit = (!spec).then(|| Dit::load(&Store::open(&file).unwrap(), Some(&cfg)).expect("Qwen-Image 1 fixture"));
    let (ph, pw, nt) = (5usize, 7usize, 21usize);
    let lat = rnd(11, ph * pw * IN, 1.0);
    let txt = rnd(12, nt * TW, 1.0);
    // runner: internal gibbs_engine::model::qwen_image_v1::Dit::{project_text, forward, forward_gpu}
    golden("diffb_qwen_image_v1", TOL, |dev| {
        let mut o = Out::default();
        let mut r = spec.then(|| spec_runner(&dir, "diffusion_pytorch_model.safetensors", None, dev));
        for sigma in [0.8f32, 0.15] {
            let y = if let Some(r) = &mut r {
                r.step = vec![("sigma".into(), Val::Num(sigma as f64))];
                let vars = [("h", num(2 * ph)), ("w", num(2 * pw)), ("nt", num(nt))];
                run1(r, "denoiser", &vars, vec![F32(lat.clone()), F32(txt.clone())], "out")
            } else if metal(dev) {
                let dit = dit.as_ref().unwrap();
                dit.forward_gpu(&dit.gpu, &lat, &txt, ph, pw, sigma).unwrap()
            } else {
                let dit = dit.as_ref().unwrap();
                let rope = Rope::new(&positions(nt, ph, pw), &dit.axes);
                dit.forward(&lat, &dit.project_text(&txt, nt), &rope, sigma)
            };
            o.f(format!("out_s{sigma}"), &y);
        }
        o
    });
}

/// Qwen-Image 2.1 single-stream DiT (`QwenImage21Transformer2DModel`): 2 blocks, dim 64 / 2 heads,
/// axes [8, 12, 12], 64 latent features, text width 48; a 5x7 grid, 21 tokens. The same weights
/// with the SwiGLU's gate/up fused (`img_mlp.gate_up`, as some GGUFs store them) give `fused_*`.
#[test]
fn qwen_image21() {
    qwen_image21_run(false)
}

/// The `qwen_image21` spec's `denoiser` on the same fixtures (split and fused).
#[test]
fn qwen_image21_spec() {
    qwen_image21_run(true)
}

fn qwen_image21_run(spec: bool) {
    use gibbs_engine::model::qwen_image21::{t2i_positions, Dit21, Rope};
    const D: usize = 64;
    const HD: usize = 32;
    const FF: usize = 128;
    const TW: usize = 48;
    const IN: usize = 64;
    let mut s = Spec::new();
    lin(&mut s, "img_in", D, IN, false);
    sp(&mut s, "txt_in.text_norm.weight", &[TW]);
    lin(&mut s, "txt_in.in_layer", D, TW, false);
    lin(&mut s, "txt_in.out_layer", D, D, false);
    lin(&mut s, "time_text_embed.timestep_embedder.linear_1", D, 256, false);
    lin(&mut s, "time_text_embed.timestep_embedder.linear_2", D, D, false);
    lin(&mut s, "modulation.1", 4 * D, D, false);
    for i in 0..2 {
        let p = format!("transformer_blocks.{i}");
        sp(&mut s, format!("{p}.attn.norm_q.weight"), &[HD]);
        sp(&mut s, format!("{p}.attn.norm_k.weight"), &[HD]);
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            lin(&mut s, &format!("{p}.attn.{n}"), D, D, false);
        }
        lin(&mut s, &format!("{p}.img_mlp.gate_layer"), FF, D, false);
        lin(&mut s, &format!("{p}.img_mlp.proj"), FF, D, false);
        lin(&mut s, &format!("{p}.img_mlp.out"), D, FF, false);
    }
    lin(&mut s, "norm_out.linear", D, D, false);
    lin(&mut s, "proj_out", IN, D, false);
    let mut split = synth(3, &s);
    // The checkpoint stores the zero-centred text norm scale (the loader adds 1).
    set(&mut split, "txt_in.text_norm.weight", rnd(30, TW, 0.1));
    // Fused copy: `gate_up = [gate; up]` replaces the two halves.
    let mut fused: Vec<T> = Vec::new();
    for t in &split {
        if let Some(p) = t.0.strip_suffix(".img_mlp.gate_layer.weight") {
            let up = &split
                .iter()
                .find(|u| u.0 == format!("{p}.img_mlp.proj.weight"))
                .unwrap()
                .2;
            fused.push((
                format!("{p}.img_mlp.gate_up.weight"),
                vec![2 * FF, D],
                [t.2.as_slice(), up].concat(),
            ));
        } else if !t.0.ends_with(".img_mlp.proj.weight") {
            fused.push(t.clone());
        }
    }
    let cfg = json!({
        "_class_name": "QwenImage21Transformer2DModel", "patch_size": 1, "in_channels": IN,
        "num_layers": 2, "attention_head_dim": HD, "num_attention_heads": 2,
        "joint_attention_dim": TW, "causal_condition": true, "axes_dims_rope": [8, 12, 12]
    });
    let sfx = if spec { "_spec" } else { "" };
    let load = |case: &str, t: &[T]| {
        let dir = fixture_dir(&format!("{case}{sfx}"));
        let file = dir.join("diffusion_pytorch_model.safetensors");
        write_st(&file, t);
        write_json(&dir.join("config.json"), &cfg);
        let dit = (!spec).then(|| Dit21::load(&Store::open(&file).unwrap(), Some(&cfg)).expect("Qwen-Image 2.1 fixture"));
        (dir, dit)
    };
    let dits = [
        ("split", load("diffb_qwen_image21", &split)),
        ("fused", load("diffb_qwen_image21_fused", &fused)),
    ];
    let (h, w, nt) = (5usize, 7usize, 21usize);
    let lat = rnd(11, h * w * IN, 1.0);
    let txt = rnd(12, nt * TW, 1.0);
    // runner: internal gibbs_engine::model::qwen_image21::Dit21::{project_text, forward, forward_gpu}
    golden("diffb_qwen_image21", TOL, |dev| {
        let mut o = Out::default();
        for (name, (dir, dit)) in &dits {
            let mut r = spec.then(|| spec_runner(dir, "diffusion_pytorch_model.safetensors", None, dev));
            for sigma in [0.8f32, 0.15] {
                let y = if let Some(r) = &mut r {
                    r.step = vec![("sigma".into(), Val::Num(sigma as f64))];
                    let vars = [("h", num(h)), ("w", num(w)), ("nt", num(nt))];
                    run1(r, "denoiser", &vars, vec![F32(lat.clone()), F32(txt.clone())], "out")
                } else if metal(dev) {
                    let dit = dit.as_ref().unwrap();
                    dit.forward_gpu(&dit.gpu, &lat, &txt, h, w, sigma).unwrap()
                } else {
                    let dit = dit.as_ref().unwrap();
                    let rope = Rope::new(&t2i_positions(nt, h, w), &dit.axes);
                    dit.forward(&lat, &dit.project_text(&txt, nt), &rope, sigma)
                };
                o.f(format!("{name}_s{sigma}"), &y);
            }
        }
        o
    });
}

/// Krea 2 (original/ComfyUI names): text fusion (2 layerwise + 2 refiner blocks over 12 encoder
/// taps of width 32) and the 2-block DiT (dim 64, 4 query / 2 kv heads of 16, 16 latent channels).
#[test]
fn krea2() {
    krea2_run(false)
}

/// The `krea2` spec's `fuse` + `denoiser` on the same fixture.
#[test]
fn krea2_spec() {
    krea2_run(true)
}

fn krea2_run(spec: bool) {
    use gibbs_engine::model::krea2::Dit;
    let (dim, hd, kv, mlp, td, tmlp) = (64usize, 16usize, 32usize, 96usize, 32usize, 48usize);
    let dir = fixture_dir(if spec { "diffb_krea2_spec" } else { "diffb_krea2" });
    let mut s = Spec::new();
    lin(&mut s, "first", dim, 64, true);
    lin(&mut s, "tmlp.0", dim, 16, true);
    lin(&mut s, "tmlp.2", dim, dim, true);
    lin(&mut s, "tproj.1", 6 * dim, dim, true);
    lin(&mut s, "txtfusion.projector", 1, 12, false);
    sp(&mut s, "txtmlp.0.scale", &[td]);
    lin(&mut s, "txtmlp.1", dim, td, true);
    lin(&mut s, "txtmlp.3", dim, dim, true);
    lin(&mut s, "last.linear", 64, dim, true);
    sp(&mut s, "last.norm.scale", &[dim]);
    sp(&mut s, "last.modulation.lin", &[2, dim]);
    let block = |s: &mut Spec, p: String, d: usize, a: usize, kvw: usize, m: usize, modulated: bool| {
        if modulated {
            sp(s, format!("{p}.mod.lin"), &[6 * d]);
        }
        for (n, sh) in [
            ("attn.wq.weight", vec![a, d]),
            ("attn.wk.weight", vec![kvw, d]),
            ("attn.wv.weight", vec![kvw, d]),
            ("attn.gate.weight", vec![d, d]),
            ("attn.wo.weight", vec![d, a]),
            ("attn.qknorm.qnorm.scale", vec![hd]),
            ("attn.qknorm.knorm.scale", vec![hd]),
            ("prenorm.scale", vec![d]),
            ("postnorm.scale", vec![d]),
            ("mlp.gate.weight", vec![m, d]),
            ("mlp.up.weight", vec![m, d]),
            ("mlp.down.weight", vec![d, m]),
        ] {
            sp(s, format!("{p}.{n}"), &sh);
        }
    };
    for i in 0..2 {
        block(&mut s, format!("blocks.{i}"), dim, dim, kv, mlp, true);
        for kind in ["layerwise_blocks", "refiner_blocks"] {
            block(&mut s, format!("txtfusion.{kind}.{i}"), td, td, td / 2, tmlp, false);
        }
    }
    let file = dir.join("krea2_tiny.safetensors");
    write_st(&file, &synth(4, &s));
    let dit = (!spec).then(|| Dit::load(&file, &mut Loras::default(), &Value::Null).expect("Krea 2 fixture"));
    let n = 24usize;
    let taps: Vec<Vec<f32>> = (0..12).map(|l| rnd(100 + l, n * td, 1.0)).collect();
    let (h, w) = (12usize, 12usize);
    let lat = rnd(7, 16 * h * w, 1.0);
    // runner: internal gibbs_engine::model::krea2::Dit::{fuse, forward, fuse_gpu, forward_gpu}
    golden("diffb_krea2", TOL, |dev| {
        let mut o = Out::default();
        if spec {
            let mut r = spec_runner(&dir, "krea2_tiny.safetensors", Some("krea2"), dev);
            // Taps stacked per token, layer-minor (the runtime's context layout).
            let stacked: Vec<f32> = (0..n).flat_map(|t| taps.iter().flat_map(move |l| l[t * td..][..td].iter().copied())).collect();
            let ctx = run1(&mut r, "fuse", &[("nt", num(n))], vec![F32(stacked)], "out");
            o.f("fuse", &ctx);
            for sigma in [0.8f32, 0.3] {
                r.step = vec![("sigma".into(), Val::Num(sigma as f64))];
                let vars = [("h", num(h)), ("w", num(w)), ("nt", num(n))];
                let y = run1(&mut r, "denoiser", &vars, vec![F32(pack2x2(&lat, 16, h, w, false)), F32(ctx.clone())], "out");
                o.f(format!("forward_s{sigma}"), &pack2x2(&y, 16, h, w, true));
            }
            return o;
        }
        let dit = dit.as_ref().unwrap();
        let ctx = if metal(dev) {
            dit.fuse_gpu(&dit.gpu_fuse, &taps, n).unwrap()
        } else {
            dit.fuse(&taps, n)
        };
        o.f("fuse", &ctx);
        for sigma in [0.8f32, 0.3] {
            let y = if metal(dev) {
                dit.forward_gpu(&dit.gpu, &lat, h, w, sigma, &ctx).unwrap()
            } else {
                dit.forward(&lat, h, w, sigma, &ctx)
            };
            o.f(format!("forward_s{sigma}"), &y);
        }
        o
    });
}

/// LTX-Video transformer (original names): 2 blocks, dim 128 (32 heads of 4), 128 latent channels,
/// caption width 48; a 3x3x4 latent grid (36 tokens), 24 text tokens, three image-conditioning
/// splits (none, the first 5 tokens clean, all clean).
#[test]
fn ltx() {
    use gibbs_engine::model::ltx::{LtxTransformer, Rope, FPS};
    const D: usize = 128;
    const LATENT: usize = 128;
    const TI: usize = 48;
    let dir = fixture_dir("diffb_ltx");
    let mut s = Spec::new();
    lin(&mut s, "patchify_proj", D, LATENT, true);
    lin(&mut s, "adaln_single.emb.timestep_embedder.linear_1", D, 256, true);
    lin(&mut s, "adaln_single.emb.timestep_embedder.linear_2", D, D, true);
    lin(&mut s, "adaln_single.linear", 6 * D, D, true);
    lin(&mut s, "caption_projection.linear_1", D, TI, true);
    lin(&mut s, "caption_projection.linear_2", D, D, true);
    for i in 0..2 {
        let b = format!("transformer_blocks.{i}");
        for a in ["attn1", "attn2"] {
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                lin(&mut s, &format!("{b}.{a}.{n}"), D, D, true);
            }
            sp(&mut s, format!("{b}.{a}.q_norm.weight"), &[D]);
            sp(&mut s, format!("{b}.{a}.k_norm.weight"), &[D]);
        }
        lin(&mut s, &format!("{b}.ff.net.0.proj"), 4 * D, D, true);
        lin(&mut s, &format!("{b}.ff.net.2"), D, 4 * D, true);
        sp(&mut s, format!("{b}.scale_shift_table"), &[6, D]);
    }
    sp(&mut s, "scale_shift_table", &[2, D]);
    lin(&mut s, "proj_out", LATENT, D, true);
    let mut t = synth(5, &s);
    for (k, e) in t.iter_mut().enumerate() {
        if e.0.ends_with("scale_shift_table") {
            e.2 = rnd(500 + k as u64, e.2.len(), 0.5);
        }
    }
    let file = dir.join("ltxv_tiny.safetensors");
    write_st(&file, &t);
    let tr = LtxTransformer::load(&Store::open(&file).unwrap()).expect("LTX fixture");
    let (f, h, w, nt) = (3usize, 3usize, 4usize, 24usize);
    let n = f * h * w;
    let x = rnd(11, n * LATENT, 1.0);
    let raw = rnd(12, nt * TI, 1.0);
    let rope = Rope::new(f, h, w, FPS as f64, D);
    // runner: internal gibbs_engine::model::ltx::LtxTransformer::{project_text, forward, forward_gpu}
    golden("diffb_ltx", TOL, |dev| {
        let mut o = Out::default();
        for (t, clean) in [(700.0f32, 0usize), (350.0, 5), (700.0, n)] {
            let y = if metal(dev) {
                tr.forward_gpu(&x, n, &raw, nt, t, clean, &rope).unwrap()
            } else {
                tr.forward(&x, n, &tr.project_text(&raw, nt), nt, t, clean, &rope)
            };
            o.f(format!("t{t}_clean{clean}"), &sub(&y));
        }
        o
    });
}

/// Qwen-Image VAE decoder (ComfyUI `qwen_image_vae.safetensors` names), base 4, mult [1, 2, 4, 4],
/// 16 latent channels: `latents * std + mean` of a 3x2 latent -> `[3, 24, 16]` before clamping.
#[test]
fn qwen_vae() {
    qwen_vae_run(false)
}

/// The `qwen_vae` spec's `decoder` on the same fixture.
#[test]
fn qwen_vae_spec() {
    qwen_vae_run(true)
}

fn qwen_vae_run(spec: bool) {
    use gibbs_engine::model::qwen_vae::QwenVae;
    const Z: usize = 16;
    let (base, mult) = (4usize, [1usize, 2, 4, 4]);
    let dir = fixture_dir(if spec { "diffb_qwen_vae_spec" } else { "diffb_qwen_vae" });
    let conv = |s: &mut Spec, p: &str, i: usize, o: usize, k: usize| {
        sp(s, format!("{p}.weight"), &[o, i, k, k]);
        sp(s, format!("{p}.bias"), &[o]);
    };
    let res = |s: &mut Spec, p: &str, i: usize, o: usize| {
        sp(s, format!("{p}.residual.0.gamma"), &[i]);
        conv(s, &format!("{p}.residual.2"), i, o, 3);
        sp(s, format!("{p}.residual.3.gamma"), &[o]);
        conv(s, &format!("{p}.residual.6"), o, o, 3);
        if i != o {
            conv(s, &format!("{p}.shortcut"), i, o, 1);
        }
    };
    let dims: Vec<usize> = [mult[3]]
        .iter()
        .chain(mult.iter().rev())
        .map(|m| base * m)
        .collect();
    let mut s = Spec::new();
    conv(&mut s, "conv2", Z, Z, 1);
    conv(&mut s, "decoder.conv1", Z, dims[0], 3);
    res(&mut s, "decoder.middle.0", dims[0], dims[0]);
    let c = dims[0];
    sp(&mut s, "decoder.middle.1.norm.gamma", &[c]);
    conv(&mut s, "decoder.middle.1.to_qkv", c, 3 * c, 1);
    conv(&mut s, "decoder.middle.1.proj", c, c, 1);
    res(&mut s, "decoder.middle.2", c, c);
    for i in 0..4 {
        let (cin, cout) = (if i > 0 { dims[i] / 2 } else { dims[i] }, dims[i + 1]);
        for j in 0..3 {
            let p = format!("decoder.upsamples.{}", i * 4 + j);
            res(&mut s, &p, if j == 0 { cin } else { cout }, cout);
        }
        if i < 3 {
            let p = format!("decoder.upsamples.{}.resample.1", i * 4 + 3);
            conv(&mut s, &p, cout, cout / 2, 3);
        }
    }
    sp(&mut s, "decoder.head.0.gamma", &[dims[4]]);
    conv(&mut s, "decoder.head.2", dims[4], 3, 3);
    let file = dir.join("qwen_image_vae.safetensors");
    write_st(&file, &synth(6, &s));
    let (mean, std) = stats(60, Z);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLQwenImage", "base_dim": base, "z_dim": Z,
            "dim_mult": mult, "num_res_blocks": 2, "attn_scales": [],
            "latents_mean": mean, "latents_std": std
        }),
    );
    // runner: internal gibbs_engine::model::qwen_vae::QwenVae::{open, decode_cpu, decode_gpu}
    // (`QwenVae::load` only accepts the 96-wide production config).
    let vae = (!spec).then(|| {
        QwenVae::open(&Mw::open(&file).unwrap(), mean.clone(), std.clone(), base).expect("Qwen VAE fixture")
    });
    let (h, w) = (3usize, 2usize);
    let z = denorm(&rnd(13, Z * h * w, 1.0), h * w, &mean, &std);
    golden("diffb_qwen_vae", TOL, |dev| {
        let vae = vae.as_ref();
        let y = if spec {
            let mut r = spec_runner(&dir, "qwen_image_vae.safetensors", None, dev);
            run1(&mut r, "decoder", &[("h", num(h)), ("w", num(w))], vec![F32(z.clone())], "image")
        } else if metal(dev) {
            let vae = vae.unwrap();
            vae.decode_gpu(&z, h, w).unwrap()
        } else {
            vae.unwrap().decode_cpu(&z, h, w)
        };
        let mut o = Out::default();
        o.f("decode_3x2", &sub(&y));
        o
    });
}

/// Qwen-Image 2.1 VAE decoder (diffusers names, residual, unpatched): base 4, mult [1, 2, 2, 4],
/// 6 latent channels, temporal flags [true, false, true]: a 3x2 latent -> RGBA `[4, 24, 16]`.
#[test]
fn qwen_vae21() {
    qwen_vae21_run(false)
}

/// The `qwen_vae21` spec's `decoder` on the same fixture.
#[test]
fn qwen_vae21_spec() {
    qwen_vae21_run(true)
}

fn qwen_vae21_run(spec: bool) {
    use gibbs_engine::model::qwen_vae21::QwenVae21;
    let (mult, base, zc) = ([1usize, 2, 2, 4], 4usize, 6usize);
    let dir = fixture_dir(if spec { "diffb_qwen_vae21_spec" } else { "diffb_qwen_vae21" });
    let conv = |s: &mut Spec, p: &str, i: usize, o: usize, k: usize| {
        sp(s, format!("{p}.weight"), &[o, i, k, k]);
        sp(s, format!("{p}.bias"), &[o]);
    };
    let res = |s: &mut Spec, p: &str, i: usize, o: usize| {
        sp(s, format!("{p}.norm1.gamma"), &[i]);
        conv(s, &format!("{p}.conv1"), i, o, 3);
        sp(s, format!("{p}.norm2.gamma"), &[o]);
        conv(s, &format!("{p}.conv2"), o, o, 3);
        if i != o {
            conv(s, &format!("{p}.conv_shortcut"), i, o, 1);
        }
    };
    let dims: Vec<usize> = std::iter::once(mult[3])
        .chain(mult.iter().rev().copied())
        .map(|m| m * base)
        .collect();
    let mut s = Spec::new();
    conv(&mut s, "post_quant_conv", zc, zc, 1);
    conv(&mut s, "decoder.conv_in", zc, dims[0], 3);
    res(&mut s, "decoder.mid_block.resnets.0", dims[0], dims[0]);
    res(&mut s, "decoder.mid_block.resnets.1", dims[0], dims[0]);
    let a = "decoder.mid_block.attentions.0";
    sp(&mut s, format!("{a}.norm.gamma"), &[dims[0]]);
    conv(&mut s, &format!("{a}.to_qkv"), dims[0], 3 * dims[0], 1);
    conv(&mut s, &format!("{a}.proj"), dims[0], dims[0], 1);
    for (i, io) in dims.windows(2).enumerate() {
        let p = format!("decoder.up_blocks.{i}");
        for j in 0..3 {
            let cin = if j == 0 { io[0] } else { io[1] };
            res(&mut s, &format!("{p}.resnets.{j}"), cin, io[1]);
        }
        if i + 1 != mult.len() {
            conv(&mut s, &format!("{p}.upsampler.resample.1"), io[1], io[1], 3);
        }
    }
    sp(&mut s, "decoder.norm_out.gamma", &[dims[4]]);
    conv(&mut s, "decoder.conv_out", dims[4], 4, 3);
    write_st(&dir.join("diffusion_pytorch_model.safetensors"), &synth(7, &s));
    let (mean, std) = stats(70, zc);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLQwenImage", "is_residual": true, "base_dim": base,
            "z_dim": zc, "dim_mult": mult, "num_res_blocks": 2,
            "temperal_downsample": [true, false, true], "attn_scales": [],
            "latents_mean": mean, "latents_std": std
        }),
    );
    let vae = (!spec).then(|| QwenVae21::load(&dir).expect("Qwen VAE 2.1 fixture"));
    let (h, w) = (3usize, 2usize);
    let z = denorm(&rnd(14, zc * h * w, 1.0), h * w, &mean, &std);
    // runner: internal gibbs_engine::model::qwen_vae21::QwenVae21::{decode_cpu, decode_gpu}
    golden("diffb_qwen_vae21", TOL, |dev| {
        let (y, oh, ow) = if spec {
            let mut r = spec_runner(&dir, "diffusion_pytorch_model.safetensors", Some("qwen_vae21"), dev);
            let y = run1(&mut r, "decoder", &[("h", num(h)), ("w", num(w))], vec![F32(z.clone())], "image");
            let n = y.len() / 4;
            let s = ((n / (h * w)) as f64).sqrt() as usize;
            (y, s * h, s * w)
        } else if metal(dev) {
            vae.as_ref().unwrap().decode_gpu(&z, h, w).unwrap()
        } else {
            vae.as_ref().unwrap().decode_cpu(&z, h, w)
        };
        assert_eq!((oh, ow), (8 * h, 8 * w)); // 3 upsamplers in this tiny config
        let mut o = Out::default();
        o.f("decode_3x2", &sub(&y));
        o
    });
}

/// LTX-Video causal-3D VAE decoder (diffusers names, v0.9.0 layout): widths [4, 8, 16, 16],
/// layers [2, 1, 1, 1, 2], 3 depth-to-space stages, patch 2, 8 latent channels; a 2x2x3 latent
/// clip (`tokens * std + mean`) -> 9 frames of `[12, 16, 24]` (patchified RGB) before clamping.
#[test]
fn ltx_vae() {
    use gibbs_engine::model::ltx_vae::{Cfg, LtxVae, Names, Vid};
    let cfg = Cfg {
        ch: [4, 8, 16, 16],
        layers: [2, 1, 1, 1, 2],
        up: [true, true, true, false],
        patch: 2,
        latent: 8,
    };
    let dir = fixture_dir("diffb_ltx_vae");
    let conv = |s: &mut Spec, p: &str, ci: usize, co: usize, k: usize| {
        sp(s, format!("{p}.conv.weight"), &[co, ci, k, k, k]);
        sp(s, format!("{p}.conv.bias"), &[co]);
    };
    let res = |s: &mut Spec, p: &str, ci: usize, co: usize| {
        conv(s, &format!("{p}.conv1"), ci, co, 3);
        conv(s, &format!("{p}.conv2"), co, co, 3);
        if ci != co {
            sp(s, format!("{p}.norm3.weight"), &[ci]);
            sp(s, format!("{p}.norm3.bias"), &[ci]);
            conv(s, &format!("{p}.conv_shortcut"), ci, co, 1);
        }
    };
    let rev: Vec<usize> = cfg.ch.iter().rev().copied().collect();
    let l: Vec<usize> = cfg.layers.iter().rev().copied().collect();
    let mut s = Spec::new();
    conv(&mut s, "decoder.conv_in", cfg.latent, rev[0], 3);
    for j in 0..l[0] {
        res(&mut s, &format!("decoder.mid_block.resnets.{j}"), rev[0], rev[0]);
    }
    let mut cin = rev[0];
    for i in 0..4 {
        let cout = rev[i];
        let b = format!("decoder.up_blocks.{i}");
        if cin != cout {
            res(&mut s, &format!("{b}.conv_in"), cin, cout);
        }
        if cfg.up[3 - i] {
            conv(&mut s, &format!("{b}.upsamplers.0.conv"), cout, cout * 8, 3);
        }
        for j in 0..l[i + 1] {
            res(&mut s, &format!("{b}.resnets.{j}"), cout, cout);
        }
        cin = cout;
    }
    conv(&mut s, "decoder.conv_out", cfg.ch[0], 3 * cfg.patch * cfg.patch, 3);
    sp(&mut s, "latents_mean", &[cfg.latent]);
    sp(&mut s, "latents_std", &[cfg.latent]);
    let mut t = synth(8, &s);
    let (mean, std) = stats(80, cfg.latent);
    set(&mut t, "latents_mean", mean.clone());
    set(&mut t, "latents_std", std.clone());
    let file = dir.join("diffusion_pytorch_model.safetensors");
    write_st(&file, &t);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLLTXVideo", "block_out_channels": cfg.ch,
            "layers_per_block": cfg.layers, "spatio_temporal_scaling": cfg.up,
            "patch_size": cfg.patch, "patch_size_t": 1, "latent_channels": cfg.latent,
            "decoder_causal": false, "scaling_factor": 1.0
        }),
    );
    // runner: internal gibbs_engine::model::ltx_vae::LtxVae::{open, dec.fwd, decode_gpu}
    // (`LtxVae::load` only accepts the 512-wide v0.9.0 production widths).
    let vae = LtxVae::open(&Mw::open(&file).unwrap(), Names { diffusers: true }, false, cfg)
        .expect("LTX VAE fixture");
    let (f, h, w, lat) = (2usize, 2usize, 3usize, cfg.latent);
    let hw = h * w;
    let tokens = rnd(15, f * hw * lat, 1.0);
    let mut d = vec![0.0f32; tokens.len()];
    for (i, tok) in tokens.chunks_exact(lat).enumerate() {
        for (c, v) in tok.iter().enumerate() {
            d[((i / hw) * lat + c) * hw + i % hw] = v * std[c] + mean[c];
        }
    }
    golden("diffb_ltx_vae", TOL, |dev| {
        let z = Vid { d: d.clone(), c: lat, f, h, w };
        let y = if metal(dev) {
            vae.decode_gpu(z).unwrap()
        } else {
            vae.dec.fwd(z)
        };
        assert_eq!((y.f, y.c, y.h, y.w), (9, 12, 8 * h, 8 * w));
        let mut o = Out::default();
        o.f("decode_2x2x3", &sub(&y.d));
        o
    });
}

/// ModelScope text-to-video `UNet3DConditionModel`: block widths [64, 128, 128, 128] (heads of 64),
/// cross-attention width 96, `transformer_in` width 128 (2 heads; the production model has 8);
/// 2 frames of 8x8 latents, 3 context tokens, two timesteps.
#[test]
fn unet3d() {
    use gibbs_engine::model::unet3d::{Cfg, St, Unet3d};
    let cfg = Cfg {
        in_ch: 4,
        block_out: [64, 128, 128, 128],
        ctx_dim: 96,
        tin_width: 128,
    };
    let dir = fixture_dir("diffb_unet3d");
    // Arguments are (in, out), as in the in-crate fixture.
    let s_lin = |s: &mut Spec, base: &str, i: usize, o: usize, bias: bool| lin(s, base, o, i, bias);
    let s_norm = |s: &mut Spec, base: &str, c: usize| {
        sp(s, format!("{base}.weight"), &[c]);
        sp(s, format!("{base}.bias"), &[c]);
    };
    let s_conv = |s: &mut Spec, base: &str, i: usize, o: usize, k: usize| {
        sp(s, format!("{base}.weight"), &[o, i, k, k]);
        sp(s, format!("{base}.bias"), &[o]);
    };
    let s_trans = |s: &mut Spec, p: &str, c: usize, inner: usize, kv: usize| {
        s_norm(s, &format!("{p}.norm"), c);
        s_lin(s, &format!("{p}.proj_in"), c, inner, true);
        s_lin(s, &format!("{p}.proj_out"), inner, c, true);
        let t = format!("{p}.transformer_blocks.0");
        for n in ["norm1", "norm2", "norm3"] {
            s_norm(s, &format!("{t}.{n}"), inner);
        }
        for (a, kv) in [("attn1", inner), ("attn2", kv)] {
            s_lin(s, &format!("{t}.{a}.to_q"), inner, inner, false);
            s_lin(s, &format!("{t}.{a}.to_k"), kv, inner, false);
            s_lin(s, &format!("{t}.{a}.to_v"), kv, inner, false);
            s_lin(s, &format!("{t}.{a}.to_out.0"), inner, inner, true);
        }
        s_lin(s, &format!("{t}.ff.net.0.proj"), inner, 4 * inner, true);
        s_lin(s, &format!("{t}.ff.net.2"), 2 * inner, inner, true);
    };
    let s_layer = |s: &mut Spec, base: &str, j: usize, ci: usize, co: usize, attn: bool| {
        let r = format!("{base}.resnets.{j}");
        s_norm(s, &format!("{r}.norm1"), ci);
        s_conv(s, &format!("{r}.conv1"), ci, co, 3);
        s_lin(s, &format!("{r}.time_emb_proj"), 4 * cfg.block_out[0], co, true);
        s_norm(s, &format!("{r}.norm2"), co);
        s_conv(s, &format!("{r}.conv2"), co, co, 3);
        if ci != co {
            s_conv(s, &format!("{r}.conv_shortcut"), ci, co, 1);
        }
        let tc = format!("{base}.temp_convs.{j}");
        for i in 1..=4 {
            s_norm(s, &format!("{tc}.conv{i}.0"), co);
            let conv = format!("{tc}.conv{i}.{}", if i == 1 { 2 } else { 3 });
            sp(s, format!("{conv}.weight"), &[co, co, 3, 1, 1]);
            sp(s, format!("{conv}.bias"), &[co]);
        }
        if attn {
            s_trans(s, &format!("{base}.attentions.{j}"), co, co, cfg.ctx_dim);
            s_trans(s, &format!("{base}.temp_attentions.{j}"), co, co, co);
        }
    };
    let (bo, c0, top) = (cfg.block_out, cfg.block_out[0], cfg.block_out.len() - 1);
    let mut s = Spec::new();
    s_lin(&mut s, "time_embedding.linear_1", c0, 4 * c0, true);
    s_lin(&mut s, "time_embedding.linear_2", 4 * c0, 4 * c0, true);
    s_conv(&mut s, "conv_in", cfg.in_ch, c0, 3);
    s_trans(&mut s, "transformer_in", c0, cfg.tin_width, cfg.tin_width);
    let mut ch = c0;
    for (i, &out) in bo.iter().enumerate() {
        let base = format!("down_blocks.{i}");
        for j in 0..2 {
            s_layer(&mut s, &base, j, if j == 0 { ch } else { out }, out, i < top);
        }
        if i < top {
            s_conv(&mut s, &format!("{base}.downsamplers.0.conv"), out, out, 3);
        }
        ch = out;
    }
    let c = bo[top];
    s_layer(&mut s, "mid_block", 0, c, c, false);
    s_trans(&mut s, "mid_block.attentions.0", c, c, cfg.ctx_dim);
    s_trans(&mut s, "mid_block.temp_attentions.0", c, c, c);
    s_layer(&mut s, "mid_block", 1, c, c, false);
    let mut prev = c;
    for i in 0..=top {
        let base = format!("up_blocks.{i}");
        let out = bo[top - i];
        let input = bo[top - (i + 1).min(top)];
        for j in 0..=2 {
            let skip = if j == 2 { input } else { out };
            let res_in = if j == 0 { prev } else { out };
            s_layer(&mut s, &base, j, res_in + skip, out, i > 0);
        }
        if i < top {
            s_conv(&mut s, &format!("{base}.upsamplers.0.conv"), out, out, 3);
        }
        prev = out;
    }
    s_norm(&mut s, "conv_norm_out", c0);
    s_conv(&mut s, "conv_out", c0, cfg.in_ch, 3);
    let file = dir.join("diffusion_pytorch_model.safetensors");
    write_st(&file, &synth(9, &s));
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "UNet3DConditionModel", "in_channels": cfg.in_ch,
            "out_channels": cfg.in_ch, "block_out_channels": bo,
            "cross_attention_dim": cfg.ctx_dim, "attention_head_dim": 64, "layers_per_block": 2,
            "norm_num_groups": 32, "sample_size": 32,
            "down_block_types": ["CrossAttnDownBlock3D", "CrossAttnDownBlock3D",
                                 "CrossAttnDownBlock3D", "DownBlock3D"],
            "up_block_types": ["UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D",
                               "CrossAttnUpBlock3D"]
        }),
    );
    // runner: internal gibbs_engine::model::unet3d::{St::open, Unet3d::from_st, forward_cpu, forward_gpu}
    // (`Unet3d::load` is fixed to the production widths).
    let m = Unet3d::from_st(St::open(&file, cfg).unwrap()).expect("UNet3D fixture");
    let (frames, h, w, ctx_len) = (2usize, 8usize, 8usize, 3usize);
    let x = rnd(16, frames * cfg.in_ch * h * w, 1.0);
    let ctx = rnd(17, ctx_len * cfg.ctx_dim, 1.0);
    golden("diffb_unet3d", TOL, |dev| {
        let mut o = Out::default();
        for t in [900.0f32, 400.0] {
            let y = if metal(dev) {
                m.forward_gpu(&x, frames, h, w, t, &ctx, ctx_len).unwrap()
            } else {
                m.forward_cpu(&x, frames, h, w, t, &ctx, ctx_len).unwrap()
            };
            o.f(format!("eps_t{t}"), &y);
        }
        o
    });
}

/// The `qwen_image21` spec on the real Qwen-Image 2.1 Q4_0 GGUF (one denoiser step, 4x4 latent
/// grid, 12 text rows) against the old `Dit21::forward_gpu` output captured before its deletion
/// (`diffb_qwen_image21_real.json`, from `real_gguf_step_matches_cpu`). Needs the ~4 GB file.
#[test]
#[ignore]
fn qwen_image21_real_gguf_spec() {
    // `dit_graph::fixture::rand`, the old test's inputs.
    fn rand(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }
    let path = std::path::PathBuf::from(std::env::var("GIBBS_QWEN21_TRANSFORMER").unwrap_or_else(|_| {
        "/Users/keas/code/rust-inference-server/disk/models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF/snapshots/main/qwen-image-2.1-UC-Q4_0.gguf".into()
    }));
    let want: Vec<f32> = serde_json::from_str(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/diffb_qwen_image21_real.json")).unwrap(),
    )
    .unwrap();
    let mut r = spec_runner(path.parent().unwrap(), path.file_name().unwrap().to_str().unwrap(), Some("qwen_image21"), Device::Metal);
    let var = |r: &Runner, k: &str| r.model.env.var(k).unwrap().usize().unwrap();
    let (in_ch, tw) = (var(&r, "in_ch"), var(&r, "text_width"));
    let (h, w, nt) = (4usize, 4usize, 12usize);
    r.step = vec![("sigma".into(), Val::Num(0.7f32 as f64))];
    let got = run1(
        &mut r,
        "denoiser",
        &[("h", num(h)), ("w", num(w)), ("nt", num(nt))],
        vec![F32(rand(h * w * in_ch, 21, 1.0)), F32(rand(nt * tw, 22, 1.0))],
        "out",
    );
    let (e, c) = gibbs_engine::arch::testing::compare(&got, &want);
    eprintln!("qwen_image21 real GGUF spec vs old Metal: rel {e:.3e} cos {c:.7}");
    assert!(e < 1e-3 && c > 0.9999, "rel {e} cos {c}");
}

/// Qwen2.5-VL text tower (`model.language_model.*`, qkv biases, GQA 4/2 heads of 16, 2 layers) as
/// Qwen-Image 1's text encoder: the final-normed hidden states of 11 tokens.
#[test]
fn qwen2_5_vl_text() {
    qwen2_5_vl_text_run(false)
}

/// The `qwen2_5_vl_text` spec's `text` component (head `hidden`) on the same fixture.
#[test]
fn qwen2_5_vl_text_spec() {
    qwen2_5_vl_text_run(true)
}

fn qwen2_5_vl_text_run(spec: bool) {
    let (d, ff, v) = (64usize, 128usize, 32usize);
    let dir = fixture_dir(if spec { "diffb_qwen2_5_vl_text_spec" } else { "diffb_qwen2_5_vl_text" });
    let mut s = Spec::new();
    let p = "model.language_model";
    sp(&mut s, format!("{p}.embed_tokens.weight"), &[v, d]);
    for i in 0..2 {
        let l = format!("{p}.layers.{i}");
        sp(&mut s, format!("{l}.input_layernorm.weight"), &[d]);
        lin(&mut s, &format!("{l}.self_attn.q_proj"), d, d, true);
        lin(&mut s, &format!("{l}.self_attn.k_proj"), 32, d, true);
        lin(&mut s, &format!("{l}.self_attn.v_proj"), 32, d, true);
        lin(&mut s, &format!("{l}.self_attn.o_proj"), d, d, false);
        sp(&mut s, format!("{l}.post_attention_layernorm.weight"), &[d]);
        lin(&mut s, &format!("{l}.mlp.gate_proj"), ff, d, false);
        lin(&mut s, &format!("{l}.mlp.up_proj"), ff, d, false);
        lin(&mut s, &format!("{l}.mlp.down_proj"), d, ff, false);
    }
    sp(&mut s, format!("{p}.norm.weight"), &[d]);
    write_st(&dir.join("model.safetensors"), &synth(8, &s));
    write_json(
        &dir.join("config.json"),
        &json!({"model_type": "qwen2_5_vl", "text_config": {"hidden_size": d, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": ff, "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0, "vocab_size": v, "tie_word_embeddings": false}}),
    );
    let ids: Vec<u32> = (0..11).map(|i| (i * 7 + 3) % v as u32).collect();
    // runner: internal gibbs_engine::model::qwen3_5::Qwen3Encoder::hidden_states_normed (CPU only)
    golden("diffb_qwen2_5_vl_text", TOL, |dev| {
        let y = if spec {
            spec_runner(&dir, "model.safetensors", Some("qwen2_5_vl_text"), dev).lm(&ids, 0, "hidden", "hidden", vec![])
        } else {
            let _cpu = gibbs_engine::hardware::cpu_compute::allow();
            gibbs_engine::model::qwen3_5::Qwen3Encoder::open(&dir).unwrap().hidden_states_normed(&ids).unwrap()
        };
        let mut o = Out::default();
        o.f("hidden", &y);
        o
    });
}

/// The pipeline `runtime` sections of the image families deserialize as `runtime::diffusion`
/// reads them.
#[test]
fn dit_runtimes_parse() {
    let reg = gibbs_engine::arch::registry().unwrap();
    for f in ["zimage", "qwen_image", "qwen_image21", "krea2"] {
        let spec = reg.spec(f, &Default::default()).unwrap_or_else(|e| panic!("{f}: {e}"));
        spec.runtime_as::<gibbs_engine::runtime::diffusion::Runtime>()
            .unwrap_or_else(|e| panic!("{f}: {e}"));
    }
}
