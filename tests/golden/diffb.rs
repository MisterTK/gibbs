//! Diffusion/video goldens, group B: Z-Image, Qwen-Image 1 / 2.1, Krea 2, LTX-Video transformers,
//! the Qwen-Image / Qwen-Image 2.1 / LTX VAE decoders and the ModelScope UNet3D. Each case writes a
//! tiny random checkpoint, runs its family spec's networks (or the whole pipeline through the
//! public image loaders) and records the outputs: the graph CPU executor on `Device::Cpu`, the
//! Metal program on `Device::Metal`.

use crate::util::*;
use gibbs_engine::hardware::Device;
use serde_json::{json, Value};
use gibbs_engine::arch::expr::Val;
use gibbs_engine::graph::Tensor;
use gibbs_engine::runtime::diffusion::{unconsumed, Net};

type Spec = Vec<(String, Vec<usize>)>;

/// [`golden`] for cases whose JSON was written by an implementation since deleted (the legacy
/// Z-Image / Qwen-Image / Krea 2 / Qwen VAE code): checked, never rewritten (`GIBBS_GOLDEN_WRITE`
/// skips them).
fn frozen(case: &str, tol: Tol, run: impl Fn(Device) -> Out) {
    if writing() {
        eprintln!("golden: {case} is frozen (its oracle implementation is deleted); not rewritten");
        return;
    }
    golden(case, tol, run);
}

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

use gibbs_engine::arch::testing::Runner;
use gibbs_engine::graph::Tensor::F32;

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

/// Every checkpoint tensor was read by some instance (or is `ignore`d).
fn assert_consumed(r: &Runner) {
    let left = r.unconsumed();
    assert!(left.is_empty(), "unconsumed tensors: {left:?}");
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
    const CAP: usize = 48;
    const IN: usize = 4;
    let dir = zimage_fixture("diffb_zimage");
    // runner: arch::testing::Runner, spec `zimage` component `denoiser`
    frozen("diffb_zimage", TOL, |dev| {
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
        assert_consumed(&r);
        o
    });
}

/// Qwen-Image 1 double-stream MMDiT (`QwenImageTransformer2DModel`): 2 blocks, dim 64 / 2 heads,
/// axes [8, 12, 12], 64 packed latent features, text width 48; an odd 5x7 patch grid, 21 tokens.
#[test]
fn qwen_image_v1() {
    qwen_image_v1_run()
}

/// Qwen-Image 1 MMDiT `diffusion_pytorch_model.safetensors` + config in `dir`: 2 blocks, dim 64 / 2
/// heads, 64 packed latent features, text width 48.
fn write_qwen_image_v1(dir: &std::path::Path) -> Value {
    const D: usize = 64;
    const HD: usize = 32;
    const FF: usize = 128;
    const TW: usize = 48;
    const IN: usize = 64;
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
    cfg
}

fn qwen_image_v1_run() {
    const TW: usize = 48;
    const IN: usize = 64;
    let dir = fixture_dir("diffb_qwen_image_v1");
    write_qwen_image_v1(&dir);
    let (ph, pw, nt) = (5usize, 7usize, 21usize);
    let lat = rnd(11, ph * pw * IN, 1.0);
    let txt = rnd(12, nt * TW, 1.0);
    // runner: arch::testing::Runner, spec `qwen_image` component `denoiser`
    frozen("diffb_qwen_image_v1", TOL, |dev| {
        let mut o = Out::default();
        let mut r = spec_runner(&dir, "diffusion_pytorch_model.safetensors", None, dev);
        for sigma in [0.8f32, 0.15] {
            r.step = vec![("sigma".into(), Val::Num(sigma as f64))];
            let vars = [("h", num(2 * ph)), ("w", num(2 * pw)), ("nt", num(nt))];
            let y = run1(&mut r, "denoiser", &vars, vec![F32(lat.clone()), F32(txt.clone())], "out");
            o.f(format!("out_s{sigma}"), &y);
        }
        assert_consumed(&r);
        o
    });
}

/// Qwen-Image 2.1 single-stream DiT (`QwenImage21Transformer2DModel`): 2 blocks, dim 64 / 2 heads,
/// axes [8, 12, 12], 64 latent features, text width 48; a 5x7 grid, 21 tokens. The same weights
/// with the SwiGLU's gate/up fused (`img_mlp.gate_up`, as some GGUFs store them) give `fused_*`.
#[test]
fn qwen_image21() {
    qwen_image21_run()
}

/// Qwen-Image 2.1 single-stream DiT tensors (split and gate/up-fused) and config: 2 blocks, dim 64 /
/// 2 heads, 64 latent features, text width 48.
fn qwen_image21_fixture() -> (Vec<T>, Vec<T>, Value) {
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
    (split, fused, cfg)
}

fn qwen_image21_run() {
    const TW: usize = 48;
    const IN: usize = 64;
    let (split, fused, cfg) = qwen_image21_fixture();
    let load = |case: &str, t: &[T]| {
        let dir = fixture_dir(case);
        write_st(&dir.join("diffusion_pytorch_model.safetensors"), t);
        write_json(&dir.join("config.json"), &cfg);
        dir
    };
    let dirs = [("split", load("diffb_qwen_image21", &split)), ("fused", load("diffb_qwen_image21_fused", &fused))];
    let (h, w, nt) = (5usize, 7usize, 21usize);
    let lat = rnd(11, h * w * IN, 1.0);
    let txt = rnd(12, nt * TW, 1.0);
    // runner: arch::testing::Runner, spec `qwen_image21` component `denoiser`
    frozen("diffb_qwen_image21", TOL, |dev| {
        let mut o = Out::default();
        for (name, dir) in &dirs {
            let mut r = spec_runner(dir, "diffusion_pytorch_model.safetensors", None, dev);
            for sigma in [0.8f32, 0.15] {
                r.step = vec![("sigma".into(), Val::Num(sigma as f64))];
                let vars = [("h", num(h)), ("w", num(w)), ("nt", num(nt))];
                let y = run1(&mut r, "denoiser", &vars, vec![F32(lat.clone()), F32(txt.clone())], "out");
                o.f(format!("{name}_s{sigma}"), &y);
            }
            assert_consumed(&r);
        }
        o
    });
}

/// Krea 2 (original/ComfyUI names): text fusion (2 layerwise + 2 refiner blocks over 12 encoder
/// taps of width 32) and the 2-block DiT (dim 64, 4 query / 2 kv heads of 16, 16 latent channels).
#[test]
fn krea2() {
    krea2_run()
}

/// The tiny Krea 2 checkpoint `krea2_tiny.safetensors` in fixture dir `case`.
fn krea2_fixture(case: &str) -> std::path::PathBuf {
    let (dim, hd, kv, mlp, td, tmlp) = (64usize, 16usize, 32usize, 96usize, 32usize, 48usize);
    let dir = fixture_dir(case);
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
    dir
}

fn krea2_run() {
    let td = 32usize;
    let dir = krea2_fixture("diffb_krea2");
    let n = 24usize;
    let taps: Vec<Vec<f32>> = (0..12).map(|l| rnd(100 + l, n * td, 1.0)).collect();
    let (h, w) = (12usize, 12usize);
    let lat = rnd(7, 16 * h * w, 1.0);
    // runner: arch::testing::Runner, spec `krea2` components `fuse` + `denoiser` (2x2-packed latents)
    frozen("diffb_krea2", TOL, |dev| {
        let mut o = Out::default();
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
        assert_consumed(&r);
        o
    });
}

/// LTX-Video transformer (original names): 2 blocks, dim 128 (32 heads of 4), 128 latent channels,
/// caption width 48; a 3x3x4 latent grid (36 tokens), 24 text tokens, three image-conditioning
/// splits (none, the first 5 tokens clean, all clean).
#[test]
fn ltx() {
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
    let (f, h, w, nt) = (3usize, 3usize, 4usize, 24usize);
    let n = f * h * w;
    let x = rnd(11, n * LATENT, 1.0);
    let raw = rnd(12, nt * TI, 1.0);
    // runner: gibbs_engine::runtime::diffusion::Net (spec `ltx`, net `denoiser`; timestep features per step).
    golden("diffb_ltx", TOL, |dev| {
        let net = spec_net(&dir, "ltxv_tiny.safetensors", "ltx", "denoiser", dev);
        let mut o = Out::default();
        for (t, clean) in [(700.0f32, 0usize), (350.0, 5), (700.0, n)] {
            let vars = [("frames", f), ("h", h), ("w", w), ("nt", nt), ("clean", clean)].map(|(k, v)| (k, Val::Num(v as f64)));
            let given = [("x", Tensor::F32(x.clone())), ("context", Tensor::F32(raw.clone()))];
            o.f(format!("t{t}_clean{clean}"), &sub(&net.run_one(&vars, &given, &[("t", t as f64)], None).unwrap()));
        }
        assert!(unconsumed(&net.model, &[&net], &[]).is_empty());
        o
    });
}

/// Qwen-Image VAE decoder (ComfyUI `qwen_image_vae.safetensors` names), base 4, mult [1, 2, 4, 4],
/// 16 latent channels: `latents * std + mean` of a 3x2 latent -> `[3, 24, 16]` before clamping.
#[test]
fn qwen_vae() {
    qwen_vae_run()
}

/// Qwen-Image VAE decoder (original Wan names) `qwen_image_vae.safetensors` + config in `dir`, widths
/// `base * mult`, `z` latent channels; returns the latent (mean, std).
fn write_qwen_vae(dir: &std::path::Path, base: usize, mult: [usize; 4], z: usize) -> (Vec<f32>, Vec<f32>) {
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
    conv(&mut s, "conv2", z, z, 1);
    conv(&mut s, "decoder.conv1", z, dims[0], 3);
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
    sp(&mut s, "decoder.head.0.gamma", &[dims[mult.len()]]);
    conv(&mut s, "decoder.head.2", dims[mult.len()], 3, 3);
    let file = dir.join("qwen_image_vae.safetensors");
    write_st(&file, &synth(6, &s));
    let (mean, std) = stats(60, z);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLQwenImage", "base_dim": base, "z_dim": z,
            "dim_mult": mult, "num_res_blocks": 2, "attn_scales": [],
            "latents_mean": mean, "latents_std": std
        }),
    );
    (mean, std)
}

fn qwen_vae_run() {
    const Z: usize = 16;
    let (base, mult) = (4usize, [1usize, 2, 4, 4]);
    let dir = fixture_dir("diffb_qwen_vae");
    let (mean, std) = write_qwen_vae(&dir, base, mult, Z);
    let (h, w) = (3usize, 2usize);
    let z = denorm(&rnd(13, Z * h * w, 1.0), h * w, &mean, &std);
    // runner: arch::testing::Runner, spec `qwen_vae` component `decoder`
    frozen("diffb_qwen_vae", TOL, |dev| {
        let mut r = spec_runner(&dir, "qwen_image_vae.safetensors", None, dev);
        let y = run1(&mut r, "decoder", &[("h", num(h)), ("w", num(w))], vec![F32(z.clone())], "image");
        assert_consumed(&r);
        let mut o = Out::default();
        o.f("decode_3x2", &sub(&y));
        o
    });
}

/// Qwen-Image 2.1 VAE decoder (diffusers names, residual, unpatched): base 4, mult [1, 2, 2, 4],
/// 6 latent channels, temporal flags [true, false, true]: a 3x2 latent -> RGBA `[4, 24, 16]`.
#[test]
fn qwen_vae21() {
    qwen_vae21_run()
}

/// Qwen-Image 2.1 VAE decoder (diffusers names, residual, unpatched) in `dir`: widths
/// `base * mult`, `zc` latent channels, `temporal` flags (one per downsampling level); returns the
/// latent (mean, std).
fn write_qwen_vae21(dir: &std::path::Path, mult: &[usize], base: usize, zc: usize, temporal: &[bool]) -> (Vec<f32>, Vec<f32>) {
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
    let dims: Vec<usize> = std::iter::once(mult[mult.len() - 1])
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
    sp(&mut s, "decoder.norm_out.gamma", &[dims[mult.len()]]);
    conv(&mut s, "decoder.conv_out", dims[mult.len()], 4, 3);
    write_st(&dir.join("diffusion_pytorch_model.safetensors"), &synth(7, &s));
    let (mean, std) = stats(70, zc);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKLQwenImage", "is_residual": true, "base_dim": base,
            "z_dim": zc, "dim_mult": mult, "num_res_blocks": 2,
            "temperal_downsample": temporal, "attn_scales": [],
            "latents_mean": mean, "latents_std": std
        }),
    );
    (mean, std)
}

fn qwen_vae21_run() {
    let (mult, base, zc) = ([1usize, 2, 2, 4], 4usize, 6usize);
    let dir = fixture_dir("diffb_qwen_vae21");
    let (mean, std) = write_qwen_vae21(&dir, &mult, base, zc, &[true, false, true]);
    let (h, w) = (3usize, 2usize);
    let z = denorm(&rnd(14, zc * h * w, 1.0), h * w, &mean, &std);
    // runner: arch::testing::Runner, spec `qwen_vae21` component `decoder`
    frozen("diffb_qwen_vae21", TOL, |dev| {
        let mut r = spec_runner(&dir, "diffusion_pytorch_model.safetensors", Some("qwen_vae21"), dev);
        let y = run1(&mut r, "decoder", &[("h", num(h)), ("w", num(w))], vec![F32(z.clone())], "image");
        assert_consumed(&r);
        assert_eq!(y.len(), 4 * (8 * h) * (8 * w)); // RGBA; 3 upsamplers in this tiny config
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
    struct Cfg {
        ch: [usize; 4],
        layers: [usize; 5],
        up: [bool; 4],
        patch: usize,
        latent: usize,
    }
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
    let (f, h, w, lat) = (2usize, 2usize, 3usize, cfg.latent);
    let hw = h * w;
    let tokens = rnd(15, f * hw * lat, 1.0);
    // Pipeline latents `[f, c, h*w]`; the decoder applies `* std + mean` itself.
    let mut z = vec![0.0f32; tokens.len()];
    for (i, tok) in tokens.chunks_exact(lat).enumerate() {
        for (c, v) in tok.iter().enumerate() {
            z[((i / hw) * lat + c) * hw + i % hw] = *v;
        }
    }
    let _ = (&mean, &std);
    // runner: gibbs_engine::runtime::diffusion::Net (spec `ltx_vae`, net `decoder`, patchified output `y`).
    golden("diffb_ltx_vae", TOL, |dev| {
        let net = spec_net(&dir, "diffusion_pytorch_model.safetensors", "ltx_vae", "decoder", dev);
        let num = |v: usize| Val::Num(v as f64);
        let vars = [("frames", num(f)), ("h", num(h)), ("w", num(w))];
        let y = net.run_one(&vars, &[("z", Tensor::F32(z.clone()))], &[], Some("y")).unwrap();
        assert_eq!(y.len(), 9 * 12 * 8 * h * 8 * w);
        let mut o = Out::default();
        o.f("decode_2x2x3", &sub(&y));
        o
    });
}

/// The `diffb_unet3d` UNet3D checkpoint (`diffusion_pytorch_model.safetensors` + `config.json`) in `dir`.
fn write_unet3d(dir: &std::path::Path, ctx_dim: usize) {
    struct Cfg {
        in_ch: usize,
        block_out: [usize; 4],
        ctx_dim: usize,
        tin_width: usize,
    }
    let cfg = Cfg {
        in_ch: 4,
        block_out: [64, 128, 128, 128],
        ctx_dim,
        tin_width: 128,
    };
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
}

/// ModelScope text-to-video `UNet3DConditionModel`: block widths [64, 128, 128, 128] (heads of 64),
/// cross-attention width 96, `transformer_in` width 128 (2 heads; the production model has 8);
/// 2 frames of 8x8 latents, 3 context tokens, two timesteps.
#[test]
fn unet3d() {
    let dir = fixture_dir("diffb_unet3d");
    write_unet3d(&dir, 96);
    let (in_ch, ctx_dim) = (4, 96);
    let (frames, h, w, ctx_len) = (2usize, 8usize, 8usize, 3usize);
    let x = rnd(16, frames * in_ch * h * w, 1.0);
    let ctx = rnd(17, ctx_len * ctx_dim, 1.0);
    // runner: gibbs_engine::runtime::diffusion::Net (spec `modelscope_t2v`, net `denoiser`).
    golden("diffb_unet3d", TOL, |dev| {
        let net = spec_net(&dir, "diffusion_pytorch_model.safetensors", "modelscope_t2v", "denoiser", dev);
        let num = |v: usize| Val::Num(v as f64);
        let vars = [("frames", num(frames)), ("h", num(h)), ("w", num(w)), ("nt", num(ctx_len))];
        let mut o = Out::default();
        for t in [900.0f32, 400.0] {
            let given = [("x", Tensor::F32(x.clone())), ("context", Tensor::F32(ctx.clone()))];
            o.f(format!("eps_t{t}"), &net.run_one(&vars, &given, &[("t", t as f64)], None).unwrap());
        }
        assert!(unconsumed(&net.model, &[&net], &[]).is_empty());
        o
    });
}

/// Network `net` of `family` loaded from fixture `dir` (weights `weights`), placed on `dev`.
fn spec_net(dir: &std::path::Path, weights: &str, family: &str, net: &str, dev: Device) -> Net {
    let m = gibbs_engine::arch::load::load(&artifacts(dir, "test/spec", weights), gibbs_engine::arch::registry().unwrap(), Some(family))
        .expect(family);
    gibbs_engine::arch::load::check(&m).unwrap();
    Net::new(std::sync::Arc::new(m), net, gibbs_engine::hardware::PlacementMode::Pin(dev))
}

/// ModelScope text-to-video through the public video pipeline (`SpecDiffusion::generate_video`): a
/// tiny `TextToVideoSDPipeline` snapshot (2-layer CLIP, the `diffb_unet3d` UNet3D with cross width
/// 64, a 4-level SD VAE decoding every frame), 16 frames of 64x64, 2 DDIM steps; the frames rule
/// refuses 8 frames. Detected and loaded through `model::video_repo` from the hub cache.
/// Shape/sanity checks only (no recorded golden).
#[test]
fn modelscope_pipeline() {
    use gibbs_engine::model::video_repo;
    use gibbs_engine::runtime::diffusion::VideoRequest;
    let root = fixture_dir("diffb_modelscope_pipeline");
    let hub = crate::diffa::clip_hub(&root);
    // The snapshot sits in the hub cache, so detection and fetches resolve offline.
    let dir = root.join("hub/models--golden--tiny-t2v/snapshots/main");
    write_json(
        &dir.join("model_index.json"),
        &json!({"_class_name": "TextToVideoSDPipeline", "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"], "unet": ["diffusers", "UNet3DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"], "scheduler": ["diffusers", "DDIMScheduler"]}),
    );
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({"_class_name": "DDIMScheduler", "beta_schedule": "scaled_linear", "beta_start": 0.00085,
                "beta_end": 0.012, "num_train_timesteps": 1000, "steps_offset": 1, "clip_sample": false,
                "set_alpha_to_one": false, "prediction_type": "epsilon"}),
    );
    crate::diffa::write_clip(&dir, "text_encoder", "tokenizer", true, false, 21);
    write_unet3d(&dir.join("unet"), 64);
    crate::diffa::write_sd_vae(&dir, &[32, 32, 32, 32]);
    let plan = video_repo::detect(&hub, "golden/tiny-t2v", None).expect("a video repo").unwrap();
    assert_eq!(plan.family, "modelscope_t2v");
    let req = VideoRequest {
        prompt: "t5 t17 t200".into(),
        frames: Some(16),
        width: Some(64),
        height: Some(64),
        steps: Some(2),
        guidance: Some(5.0),
        seed: 3,
        ..Default::default()
    };
    for dev in devices() {
        let mode = gibbs_engine::hardware::PlacementMode::Pin(dev);
        let (frames, fps, _) = video_repo::generate(&hub, &plan, mode, None, &req, None, &|_| {}).unwrap();
        assert_eq!((frames.len(), fps), (16, 8));
        assert!(frames.iter().all(|f| (f.width, f.height, f.data.len()) == (64, 64, 64 * 64 * 3)));
        assert!(frames[0].data.iter().any(|&v| v != frames[0].data[0]), "{dev:?}: flat frame");
        let p = video_repo::load(&hub, &plan, mode, None).unwrap();
        let e = p.generate_video(&VideoRequest { frames: Some(8), ..req.clone() }, None, &|_| {}).unwrap_err();
        assert!(e.contains("16..=24"), "{e}");
        let e = p.generate_video(&req, Some(&frames[0]), &|_| {}).unwrap_err();
        assert!(e.contains("does not take an input image"), "{e}");
    }
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
    qwen2_5_vl_text_run()
}

fn qwen2_5_vl_text_run() {
    let (d, ff, v) = (64usize, 128usize, 32usize);
    let dir = fixture_dir("diffb_qwen2_5_vl_text");
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
    // runner: arch::testing::Runner, spec `qwen2_5_vl_text` component `text` (head `hidden`)
    frozen("diffb_qwen2_5_vl_text", TOL, |dev| {
        let mut r = spec_runner(&dir, "model.safetensors", Some("qwen2_5_vl_text"), dev);
        let y = r.lm(&ids, 0, "hidden", "hidden", vec![]);
        assert_consumed(&r);
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

/// Old `krea2::Dit` + `Loras` output on the `pair` LoRA (spec matched it at rel 1.95e-7 before
/// the old code was deleted): every 64th of `sub(out)`.
#[rustfmt::skip]
const KREA2_LORA_PIN: [f32; 36] = [0.6458024, -0.68495977, 1.2038748, 0.78530777, -0.8388921, -0.30057698, -0.5283634, -1.3309352, -0.6637603, 0.857429, 1.5352197, -1.841813, -0.32170674, 0.8646051, 1.4135798, -1.467209, -0.14866997, -0.3009609, 0.8847723, -0.25245777, -0.008546151, 0.94894725, -0.026348034, -0.5299631, 0.60384136, -0.54123783, 0.68916553, -0.18912089, -0.037548088, -0.48464662, 0.20832507, 0.7079825, 0.28712678, 1.2895193, -0.23603027, 0.6485835];

/// Krea 2 LoRA files through the shared `LoraFile` loader on the spec path (original/ComfyUI key
/// forms under `diffusion_model.`): a low-rank pair changes the output, a `.diff_b` bias delta
/// alone (zero up weight) changes it too, and a module naming a block the model lacks is an error.
#[test]
fn krea2_spec_lora() {
    use gibbs_engine::model::lora::{LoraFile, Part};
    let dir = krea2_fixture("diffb_krea2_lora");
    let reg = gibbs_engine::arch::registry().unwrap();
    let (h, w, n) = (12usize, 12usize, 24usize);
    let lat = pack2x2(&rnd(7, 16 * h * w, 1.0), 16, h, w, false);
    let ctx = rnd(8, n * 64, 1.0);
    let lora = |name: &str, t: Vec<T>| {
        let p = dir.join(name);
        write_st(&p, &t);
        LoraFile::open(&p).unwrap()
    };
    let pair = |m: &str, up: f32| -> Vec<T> {
        vec![
            (format!("diffusion_model.{m}.lora_down.weight"), vec![4, 64], rnd(30, 4 * 64, 0.5)),
            (format!("diffusion_model.{m}.lora_up.weight"), vec![64, 4], rnd(31, 64 * 4, up)),
        ]
    };
    let forward = |l: Option<&LoraFile>| -> Vec<f32> {
        let mut r = spec_runner(&dir, "krea2_tiny.safetensors", Some("krea2"), Device::Cpu);
        if let Some(l) = l {
            assert_eq!(l.apply(Part::Denoiser, &mut r.model, reg, 1.0).unwrap(), 1);
        }
        r.step = vec![("sigma".into(), Val::Num(0.5))];
        let vars = [("h", num(h)), ("w", num(w)), ("nt", num(n))];
        run1(&mut r, "denoiser", &vars, vec![F32(lat.clone()), F32(ctx.clone())], "out")
    };
    let base = forward(None);
    // Parity of the low-rank pair against the old Krea 2 LoRA loader (pinned: subsampled output of
    // `krea2::Dit::load` + `Loras::add_file` on this fixture, captured before its deletion).
    let pair_file = lora("pair.safetensors", pair("blocks.1.attn.wq", 0.5));
    let got = forward(Some(&pair_file));
    let pin = sub(&got).iter().step_by(64).copied().collect::<Vec<f32>>();
    let (e, c) = gibbs_engine::arch::testing::compare(&pin, &KREA2_LORA_PIN);
    assert!(e < 1e-3 && c > 0.9999, "LoRA vs pinned old output: rel {e} cos {c}");
    let changed = |y: &[f32]| y.iter().zip(&base).any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(changed(&got));
    let mut bias = pair("first", 0.0);
    bias.push(("diffusion_model.first.diff_b".into(), vec![64], rnd(32, 64, 0.5)));
    assert!(changed(&forward(Some(&lora("bias.safetensors", bias)))), "diff_b bias delta not applied");
    let stray = lora("stray.safetensors", pair("blocks.9.attn.wq", 0.5));
    let mut r = spec_runner(&dir, "krea2_tiny.safetensors", Some("krea2"), Device::Cpu);
    let e = stray.apply(Part::Denoiser, &mut r.model, reg, 1.0).unwrap_err();
    assert!(e.contains("matches no weight"), "{e}");
}

// ---------------------------------------------------------------------------------------------
// Tiny text-to-image pipelines (tokenizer + Qwen text tower + DiT + VAE) through the public image
// loader: prompt template, prefix drop, schedule, guidance, latent packing and VAE affine.
// ---------------------------------------------------------------------------------------------

/// A WordLevel stand-in for the Qwen BPE: `<|im_start|>` / `<|im_end|>` special, words and every
/// punctuation mark / newline one token (so the Qwen-Image system head is 34 tokens, like the real
/// tokenizer), `tK` = prompt words.
fn qwen_tokenizer_json() -> Value {
    let words = "system user assistant Describe the image by detailing color shape size texture quantity text spatial relationships of objects and background Comprehend analyze provided prompt";
    let mut vocab = serde_json::Map::new();
    for (i, w) in ["<|im_start|>", "<|im_end|>", "<unk>", "\n", ",", ":", ".", "<|endoftext|>"]
        .into_iter()
        .chain(words.split(' '))
        .enumerate()
    {
        vocab.insert(w.into(), json!(i));
    }
    let n = vocab.len();
    for k in 0..32 {
        vocab.insert(format!("t{k}"), json!(n + k));
    }
    let added: Vec<Value> = [("<|im_start|>", 0), ("<|im_end|>", 1), ("<|endoftext|>", 7)]
        .iter()
        .map(|(s, id)| {
            json!({"id": id, "content": s, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"String": " "}, "behavior": "Removed", "invert": false},
            {"type": "Split", "pattern": {"Regex": "[^\\w\\s]|\\n"}, "behavior": "Isolated", "invert": false}
        ]},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
}

/// A dense Qwen text tower in `dir` (bf16): `kind` `qwen3` (q/k norm), `qwen3_vl` (nested
/// `text_config`, interleaved M-RoPE, `model.language_model.`) or `qwen2_5_vl` (qkv biases).
fn write_qwen_te(dir: &std::path::Path, kind: &str, hidden: usize, layers: usize, seed: u64) {
    let (heads, kv, hd, inter, vocab) = (4usize, 2usize, 16usize, 64usize, 96usize);
    let p = if kind == "qwen3_vl" { "model.language_model" } else { "model" };
    let mut w = Weights::new(seed);
    w.w(format!("{p}.embed_tokens.weight"), &[vocab, hidden]);
    for l in 0..layers {
        let lp = format!("{p}.layers.{l}");
        let a = format!("{lp}.self_attn");
        w.norm(format!("{lp}.input_layernorm.weight"), hidden, 1.0)
            .norm(format!("{lp}.post_attention_layernorm.weight"), hidden, 1.0)
            .w(format!("{a}.q_proj.weight"), &[heads * hd, hidden])
            .w(format!("{a}.k_proj.weight"), &[kv * hd, hidden])
            .w(format!("{a}.v_proj.weight"), &[kv * hd, hidden])
            .w(format!("{a}.o_proj.weight"), &[hidden, heads * hd]);
        if kind == "qwen2_5_vl" {
            w.s(format!("{a}.q_proj.bias"), &[heads * hd], 0.2)
                .s(format!("{a}.k_proj.bias"), &[kv * hd], 0.2)
                .s(format!("{a}.v_proj.bias"), &[kv * hd], 0.2);
        } else {
            w.norm(format!("{a}.q_norm.weight"), hd, 1.0).norm(format!("{a}.k_norm.weight"), hd, 1.0);
        }
        w.w(format!("{lp}.mlp.gate_proj.weight"), &[inter, hidden])
            .w(format!("{lp}.mlp.up_proj.weight"), &[inter, hidden])
            .w(format!("{lp}.mlp.down_proj.weight"), &[hidden, inter]);
    }
    w.norm(format!("{p}.norm.weight"), hidden, 1.0);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    let mut text = json!({
        "model_type": kind, "hidden_size": hidden, "vocab_size": vocab, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
        "intermediate_size": inter, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": true
    });
    let cfg = match kind {
        "qwen3_vl" => {
            text["model_type"] = json!("qwen3_vl_text");
            text["rope_scaling"] = json!({"rope_type": "default", "mrope_interleaved": true, "mrope_section": [4, 2, 2]});
            json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"], "text_config": text})
        }
        "qwen2_5_vl" => {
            text["architectures"] = json!(["Qwen2_5_VLForConditionalGeneration"]);
            text["rope_scaling"] = json!({"type": "mrope", "mrope_section": [4, 2, 2]});
            text
        }
        _ => {
            text["architectures"] = json!(["Qwen3Model"]);
            text
        }
    };
    write_json(&dir.join("config.json"), &cfg);
}

fn flow_scheduler(dir: &std::path::Path, dynamic: bool) {
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000,
                "shift": 3.0, "use_dynamic_shifting": dynamic, "base_shift": 0.5, "max_shift": 0.9,
                "base_image_seq_len": 256, "max_image_seq_len": 8192, "shift_terminal": if dynamic { json!(0.02) } else { Value::Null }}),
    );
}

/// Runs `req` through the public image loader on every device; pixels every k-th value.
fn pipeline_golden(case: &str, dir: &std::path::Path, req: gibbs_engine::runtime::diffusion::ImageRequest) {
    use crate::diffa::{on, pin, stride};
    use gibbs_engine::model::hub::HuggingFaceHub;
    let hub = HuggingFaceHub::with_cache_dir(dir.join("hub"));
    let art = artifacts(dir, &format!("golden/{case}"), "none");
    let (w, h) = (req.width.unwrap(), req.height.unwrap());
    frozen(case, Tol { rel: 2e-3, cos: 0.99999 }, |dev| {
        // runner: runtime::diffusion::load_image (the public image loader), pinned to `dev`
        let p = gibbs_engine::runtime::diffusion::load_image(&art, &hub, pin(dev))
            .expect("image repo")
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        let img = on(dev, || p.text_to_image(&req)).unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!((img.width, img.height), (w, h));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}

fn request(prompt: &str, guidance: f32, seed: u64) -> gibbs_engine::runtime::diffusion::ImageRequest {
    gibbs_engine::runtime::diffusion::ImageRequest {
        prompt: prompt.into(),
        width: Some(32),
        height: Some(32),
        steps: Some(2),
        guidance: Some(guidance),
        seed,
        ..Default::default()
    }
}

/// Z-Image: Qwen3 tower (second-to-last layer), S3-DiT with in-graph patchify, SD VAE (4 latent
/// channels); CFG with `guidance + 1` and the negated velocity.
#[test]
fn zimage_pipeline() {
    let case = "diffb_zimage_pipeline";
    let dir = fixture_dir(case);
    write_json(&dir.join("model_index.json"), &json!({"_class_name": "ZImagePipeline"}));
    flow_scheduler(&dir, false);
    write_json(&dir.join("tokenizer/tokenizer.json"), &qwen_tokenizer_json());
    write_qwen_te(&dir.join("text_encoder"), "qwen3", 48, 3, 0x5A01);
    zimage_fixture(&format!("{case}/transformer"));
    crate::diffa::write_sd_vae(&dir, &[32, 32, 32, 32]);
    pipeline_golden(case, &dir, request("t3 t7 t11", 0.5, 9));
}

/// Qwen-Image 1: Qwen2.5-VL tower (final-normed, 34-token head dropped), MMDiT on 2x2-packed
/// latents, rescaled CFG, dynamic shift with a terminal, Qwen VAE.
#[test]
fn qwen_image_v1_pipeline() {
    let case = "diffb_qwen_image_v1_pipeline";
    let dir = fixture_dir(case);
    write_json(&dir.join("model_index.json"), &json!({"_class_name": "QwenImagePipeline"}));
    flow_scheduler(&dir, true);
    write_json(&dir.join("tokenizer/tokenizer.json"), &qwen_tokenizer_json());
    write_qwen_te(&dir.join("text_encoder"), "qwen2_5_vl", 48, 3, 0x5A02);
    write_qwen_image_v1(&fixture_dir(&format!("{case}/transformer")));
    write_qwen_vae(&fixture_dir(&format!("{case}/vae")), 4, [1, 2, 4, 4], 16);
    pipeline_golden(case, &dir, request("t3 t7 t11", 4.0, 10));
}

/// Qwen-Image 2.1: Qwen3-VL tower (last layer before its norm, system turn dropped), single-stream
/// DiT on token latents, CFG, 64-channel residual VAE (16x), RGBA over white.
#[test]
fn qwen_image21_pipeline() {
    let case = "diffb_qwen_image21_pipeline";
    let dir = fixture_dir(case);
    write_qwen21_pipeline(&dir);
    pipeline_golden(case, &dir, request("t3 t7 t11", 2.0, 11));
}

/// The tiny Qwen-Image 2.1 diffusers pipeline in `dir`.
fn write_qwen21_pipeline(dir: &std::path::Path) {
    write_json(&dir.join("model_index.json"), &json!({"_class_name": "QwenImage21Pipeline"}));
    flow_scheduler(dir, true);
    write_json(&dir.join("processor/tokenizer.json"), &qwen_tokenizer_json());
    write_qwen_te(&dir.join("text_encoder"), "qwen3_vl", 48, 3, 0x5A03);
    let (split, _, cfg) = qwen_image21_fixture();
    let t = dir.join("transformer");
    write_st(&t.join("diffusion_pytorch_model.safetensors"), &split);
    write_json(&t.join("config.json"), &cfg);
    write_qwen_vae21(&dir.join("vae"), &[1, 1, 1, 1, 1], 4, 64, &[false, true, false, false]);
}

/// Krea 2 (Raw): 12 taps of a 36-layer Qwen3-VL tower fused in-model, the 2-block DiT on 2x2-packed
/// latents, CFG with `guidance + 1`, dynamic mu, Qwen VAE.
#[test]
fn krea2_pipeline() {
    let case = "diffb_krea2_pipeline";
    let dir = fixture_dir(case);
    write_json(&dir.join("model_index.json"), &json!({"_class_name": "Krea2Pipeline", "is_distilled": false}));
    flow_scheduler(&dir, true);
    write_json(&dir.join("tokenizer/tokenizer.json"), &qwen_tokenizer_json());
    write_qwen_te(&dir.join("text_encoder"), "qwen3_vl", 32, 36, 0x5A04);
    krea2_fixture(&format!("{case}/transformer"));
    write_qwen_vae(&fixture_dir(&format!("{case}/vae")), 4, [1, 2, 4, 4], 16);
    pipeline_golden(case, &dir, request("t3 t7 t11", 4.5, 12));
}

fn pixels(img: gibbs_engine::model::image::ImageBuffer) -> Vec<f32> {
    img.data.iter().map(|&v| v as f32).collect()
}

/// A derived Qwen-Image 2.1 repo (a transformer file, a card naming its base, its own scheduler
/// config and card SIGMAS) through `load_image`: it runs as the base pipeline with that transformer,
/// the repo's scheduler config and the card's sigmas (steps default to their count), unlike the
/// base's own schedule.
#[test]
fn qwen_image21_derived_repo() {
    use crate::diffa::{on, pin};
    use gibbs_engine::arch::testing::compare;
    use gibbs_engine::model::hub::HuggingFaceHub;
    use gibbs_engine::runtime::diffusion::{load_image, Extras, ImageGenerator, SpecDiffusion};
    let dir = fixture_dir("diffb_qwen_image21_derived");
    let base = dir.join("hub/models--golden--qwen21-base/snapshots/main");
    write_qwen21_pipeline(&base);
    let ft = dir.join("ft");
    std::fs::create_dir_all(&ft).unwrap();
    let file = ft.join("qwen21-ft.safetensors");
    let (split, _, _) = qwen_image21_fixture();
    // A "finetune": the base transformer, every weight scaled.
    let tuned: Vec<T> = split.iter().map(|(n, s, v)| (n.clone(), s.clone(), v.iter().map(|x| x * 0.9).collect())).collect();
    write_st(&file, &tuned);
    let card = "---\nbase_model: golden/qwen21-base\n---\nSTEPS, SIGMAS = 3, [1.0, 0.7, 0.3]\n";
    std::fs::write(ft.join("README.md"), card).unwrap();
    let sched = json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 1.0,
                       "use_dynamic_shifting": false});
    write_json(&ft.join("scheduler/scheduler_config.json"), &sched);
    let hub = HuggingFaceHub::with_cache_dir(dir.join("hub"));
    let derived = artifacts(&ft, "golden/qwen21-ft", "qwen21-ft.safetensors");
    let base_art = artifacts(&base, "golden/qwen21-base", "none");
    for dev in devices() {
        let p = load_image(&derived, &hub, pin(dev)).expect("derived repo").unwrap();
        let extra = Extras {
            weights: vec![("transformer".into(), file.clone())],
            scheduler: Some(sched.clone()),
            card: Some(card.into()),
            ..Default::default()
        };
        let want = SpecDiffusion::load(&base_art, &hub, pin(dev), Some("qwen_image21"), &extra).unwrap();
        let plain = SpecDiffusion::load(&base_art, &hub, pin(dev), Some("qwen_image21"), &Extras {
            weights: vec![("transformer".into(), file.clone())],
            ..Default::default()
        })
        .unwrap();
        let mut req = request("t3 t7 t11", 0.0, 4);
        req.steps = None;
        let got = pixels(on(dev, || p.text_to_image(&req)).unwrap());
        req.steps = Some(3);
        let (e, c) = compare(&got, &pixels(on(dev, || want.text_to_image(&req)).unwrap()));
        assert!(e < 1e-6 && c > 0.999999, "{dev:?}: derived vs base + repo schedule: rel {e} cos {c}");
        let (e, _) = compare(&got, &pixels(on(dev, || plain.text_to_image(&req)).unwrap()));
        assert!(e > 1e-3, "{dev:?}: the repo schedule changed nothing (rel {e})");
    }
}

/// ComfyUI's split Krea 2 files (`diffusion_models/`, `text_encoders/`, `vae/`, no configs; the
/// spec's `runtime.split` supplies them, here patched to the tiny text tower) load through
/// `load_image` and render what the same weights do as a diffusers pipeline; the bf16 text encoder
/// file is preferred over a sibling, and `turbo` in the file name selects the distilled defaults.
#[test]
fn krea2_split_files() {
    use crate::diffa::{on, pin};
    use gibbs_engine::arch::testing::compare;
    use gibbs_engine::arch::{Registry, EMBEDDED};
    use gibbs_engine::model::hub::HuggingFaceHub;
    use gibbs_engine::runtime::diffusion::load_image_in;
    let case = "diffb_krea2_split";
    let dir = fixture_dir(case);
    let twin = dir.join("diffusers");
    write_json(&twin.join("model_index.json"), &json!({"_class_name": "Krea2Pipeline", "is_distilled": false}));
    write_json(&twin.join("tokenizer/tokenizer.json"), &qwen_tokenizer_json());
    write_qwen_te(&twin.join("text_encoder"), "qwen3_vl", 32, 36, 0x5A04);
    krea2_fixture(&format!("{case}/diffusers/transformer"));
    write_qwen_vae(&twin.join("vae"), 4, [1, 2, 4, 4], 16);
    // Split VAE files carry no config: both layouts use the default latent statistics.
    std::fs::remove_file(twin.join("vae/config.json")).unwrap();
    let split = dir.join("split");
    let cp = |from: &str, to: &str| {
        std::fs::create_dir_all(split.join(to).parent().unwrap()).unwrap();
        std::fs::copy(twin.join(from), split.join(to)).unwrap();
    };
    cp("transformer/krea2_tiny.safetensors", "diffusion_models/krea2_raw_bf16.safetensors");
    cp("transformer/krea2_tiny.safetensors", "diffusion_models/krea2_turbo_bf16.safetensors");
    // Sorts first; would fail to load as the text encoder.
    cp("vae/qwen_image_vae.safetensors", "text_encoders/a_fp8.safetensors");
    cp("text_encoder/model.safetensors", "text_encoders/qwen3vl_tiny_bf16.safetensors");
    cp("vae/qwen_image_vae.safetensors", "vae/qwen_image_vae.safetensors");
    write_json(&dir.join("hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/main/tokenizer.json"), &qwen_tokenizer_json());
    let hub = HuggingFaceHub::with_cache_dir(dir.join("hub"));
    let mut files: Vec<(String, String)> = EMBEDDED.iter().map(|(p, t)| (p.to_string(), t.to_string())).collect();
    files.push(("overrides/tiny_krea2_split.json5".into(), r#"{applies_to: {family: "krea2"}, patch: {runtime: {split: {files: {
        "text_encoder/config.json": {text_config: {hidden_size: 32, intermediate_size: 64, num_hidden_layers: 36,
          num_attention_heads: 4, num_key_value_heads: 2, head_dim: 16, rms_norm_eps: 1e-6, rope_theta: 10000,
          rope_scaling: {mrope_interleaved: true, mrope_section: [4, 2, 2], rope_type: "default"}}}}}}}}"#.into()));
    let reg = Registry::from_files(&files, &[]).unwrap();
    let raw = artifacts(&split, "golden/krea-split", "diffusion_models/krea2_raw_bf16.safetensors");
    let turbo = artifacts(&split, "golden/krea-split", "diffusion_models/krea2_turbo_bf16.safetensors");
    let twin_art = artifacts(&twin, "golden/krea-diffusers", "none");
    let req = request("t3 t7 t11", 4.5, 12);
    for dev in devices() {
        let load = |a| load_image_in(&reg, a, &hub, pin(dev)).expect("image repo").unwrap_or_else(|e| panic!("{e}"));
        let (p, t, w) = (load(&raw), load(&turbo), load(&twin_art));
        assert_eq!((p.defaults().steps, t.defaults().steps), (28, 8), "raw / turbo defaults");
        let got = pixels(on(dev, || p.text_to_image(&req)).unwrap());
        let (e, c) = compare(&got, &pixels(on(dev, || w.text_to_image(&req)).unwrap()));
        assert!(e < 1e-6 && c > 0.999999, "{dev:?}: split vs diffusers: rel {e} cos {c}");
    }
}
