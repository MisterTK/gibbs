//! Parity of the `gemma4` / `gemma4_unified` vision specs with `model::gemma4::{vision, unified}`
//! (deleted with them).

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::Tensor;
use crate::hardware::{available_devices, Device, PlacementMode};
use crate::model::gemma4::{unified::VisionEmbedder, vision::Vision};
use crate::model::image::ImageBuffer;
use crate::model::vision_graph::ProgramCache;
use crate::runtime::media::{self, Expand};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A golden fixture dir (`cargo test --test golden <case>` writes it), `None` when absent.
fn fixture(case: &str) -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/golden-fixtures").join(case);
    if !d.exists() {
        eprintln!("skip {case}: run `cargo test --test golden {}` first", case.trim_start_matches("gemma_"));
    }
    d.exists().then_some(d)
}

fn json(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

fn png(w: usize, h: usize) -> Vec<u8> {
    let mut img = ImageBuffer::new(w, h);
    for (i, px) in img.data.iter_mut().enumerate() {
        *px = ((i * 37 + (i / 7) * 11) % 251) as u8;
    }
    img.to_png()
}

fn old_weights(dir: &Path) -> (crate::model::weights::Weights, Value, Value) {
    let w = crate::model::weights::Weights::open(&dir.join("model.safetensors")).unwrap();
    (w, json(&dir.join("config.json")), json(&dir.join("processor_config.json")))
}

/// The old tower graph on the CPU executor.
fn old_cache() -> ProgramCache {
    ProgramCache::new(PlacementMode::Pin(Device::Cpu))
}

/// Per device: the spec's `runtime.media.image` preprocessing of `bytes` (must equal `old_pix`),
/// then its `vision` instance against `want`; every tensor of the tower is read.
fn check(case: &str, dir: &Path, bytes: &[u8], old_pix: &[f32], want: &[f32]) {
    let mut devs = vec![Device::Cpu];
    if available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    for dev in devs {
        let mut r = Runner::load(dir, None, dev, 64).unwrap();
        let m = media::resolve(&r.model.spec.runtime["media"], &r.model.env).unwrap();
        let img = &m["image"];
        let cfg = json(&dir.join("config.json"));
        let id = |k: &str| cfg[k].as_u64().unwrap() as u32;
        assert!(
            matches!(img.expand, Expand::RepeatBetween { placeholder, open, close }
                if (placeholder, open, close) == (id("image_token_id"), id("boi_token_id"), id("eoi_token_id"))),
            "{case}: {:?}",
            img.expand
        );
        let bidir = cfg["text_config"]["use_bidirectional_attention"] == "vision";
        assert_eq!(img.bidirectional, bidir, "{case}: bidirectional");
        let p = media::preprocess(&img.preprocess, bytes).unwrap();
        assert_eq!(p.data.len(), old_pix.len(), "{case}: preprocessed length");
        let d = p.data.iter().zip(old_pix).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(d < 1e-6, "{case}: preprocessing max |d| {d}");
        let vars: Vec<(&str, Val)> = p.grid.vars.iter().map(|&(k, v)| (k, Val::Num(v as f64))).collect();
        let mut out = r.run("vision", &vars, vec![Tensor::F32(p.data)]).unwrap();
        let got = match out.remove("embeds") {
            Some(Tensor::F32(v)) => v,
            o => panic!("{case}: output {o:?}"),
        };
        assert_eq!(got.len(), want.len(), "{case}: {} tokens", p.grid.tokens);
        let (rel, cos) = compare(&got, want);
        eprintln!("{case} {dev:?}: {} tokens ({:?}), rel {rel:.3e} cos {cos:.7}", p.grid.tokens, p.grid.vars);
        assert!(rel < 1e-3 && cos > 0.9999, "{case} {dev:?}: rel {rel:e} cos {cos}");
        let unread: Vec<_> = r.unconsumed().into_iter().filter(|n| n.contains("vision")).collect();
        assert!(unread.is_empty(), "{case}: unread {unread:?}");
    }
}

/// `gemma4_vision` tower against `Vision::build_graph` (golden fixture `gemma_gemma4_vision`).
#[test]
fn vision_matches_old_tower() {
    let _g = crate::hardware::cpu_compute::allow();
    let Some(dir) = fixture("gemma_gemma4_vision") else { return };
    let (w, cfg, processor) = old_weights(&dir);
    let v = Vision::load(&w, &cfg["vision_config"], &processor).unwrap();
    for (iw, ih) in [(32, 24), (20, 44)] {
        let bytes = png(iw, ih);
        let (pix, ph, pw) = v.preprocess(&bytes).unwrap();
        let want = v.forward_gpu(&old_cache(), &pix, ph, pw).unwrap();
        check(&format!("gemma4 vision {iw}x{ih}"), &dir, &bytes, &pix, &want);
    }
}

/// `gemma4_unified` encoder-free embedder against `VisionEmbedder::build_graph` (golden fixture
/// `gemma_gemma4_unified_vision`).
#[test]
fn unified_vision_matches_old_embedder() {
    let _g = crate::hardware::cpu_compute::allow();
    let Some(dir) = fixture("gemma_gemma4_unified_vision") else { return };
    let (w, cfg, processor) = old_weights(&dir);
    let v = VisionEmbedder::load(&w, &cfg["vision_config"], &processor).unwrap();
    for (iw, ih) in [(48, 36), (30, 70)] {
        let bytes = png(iw, ih);
        let (pix, n, gw) = v.patches(&bytes).unwrap();
        let want = v.forward_gpu(&old_cache(), &pix, n, gw).unwrap();
        check(&format!("gemma4_unified vision {iw}x{ih}"), &dir, &bytes, &pix, &want);
    }
}

/// The real E2B GGUF + `mmproj`: the renamed `gemma4v` tower builds with every weight present,
/// reads every vision tensor, and runs a 6x6 patch grid on the CPU executor.
#[test]
#[ignore = "real E2B GGUF + mmproj, ~18 s in a debug build"]
fn e2b_gguf_mmproj_vision_tower_loads() {
    let _g = crate::hardware::cpu_compute::allow();
    let dir = Path::new("/Users/keas/code/rust-inference-server/disk/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/main");
    if !dir.exists() {
        eprintln!("skip: no E2B GGUF");
        return;
    }
    let a = crate::model::hub::ModelArtifacts {
        model_id: "unsloth/gemma-4-E2B-it-qat-GGUF".into(),
        local_dir: Some(dir.to_path_buf()),
        weights_path: Some(dir.join("gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf")),
        is_cached: true,
        ..Default::default()
    };
    let mut r = Runner::from_artifacts(&a, None, Device::Cpu, 64).unwrap();
    assert_eq!(r.model.spec.family, "gemma4");
    let m = media::resolve(&r.model.spec.runtime["media"], &r.model.env).unwrap();
    eprintln!("e2b image media: {:?}", m["image"]);
    // No config.json (no `config_path`): the ids come from `token(...)` over the GGUF vocabulary.
    assert!(r.model.ckpt.cfg.is_null());
    assert!(matches!(m["image"].expand, Expand::RepeatBetween { placeholder: 258880, open: 255999, close: 258882 }));
    let unmapped: Vec<_> = r.model.renames.iter().filter(|(raw, c)| raw == c && (raw.starts_with("v.") || raw.starts_with("mm.input"))).collect();
    assert!(unmapped.is_empty(), "unmapped mmproj vision tensors {unmapped:?}");
    let vars = [("grid_h", Val::Num(6.0)), ("grid_w", Val::Num(6.0))];
    let inst = r.instance("vision", &vars).unwrap();
    let vision: Vec<_> = r.model.renames.iter().map(|(_, c)| c).filter(|c| c.starts_with("model.vision_tower") || c.starts_with("model.embed_vision")).collect();
    // Clamp bounds fold into the graph as constants (`values()`), every other tensor is a weight.
    let bound = |c: &str| c.ends_with("_min") || c.ends_with("_max");
    let unread: Vec<_> = vision.iter().filter(|c| !bound(c) && !inst.weights.contains(**c)).collect();
    assert!(unread.is_empty(), "vision tensors the tower does not read: {unread:?}");
    let clamps = inst.graph.nodes.iter().filter(|n| n.name.ends_with("_clamp")).count();
    assert_eq!(clamps, 2 * vision.iter().filter(|c| c.ends_with("input_min")).count(), "clamp nodes");
    let v_in = inst.inputs[0].shape[1];
    let pix: Vec<f32> = (0..36 * v_in).map(|i| ((i * 37 % 251) as f32 / 125.0) - 1.0).collect();
    let out = r.run("vision", &vars, vec![Tensor::F32(pix)]).unwrap();
    let Some(Tensor::F32(e)) = out.get("embeds") else { panic!("no embeds") };
    let rms = (e.iter().map(|x| x * x).sum::<f32>() / e.len() as f32).sqrt();
    eprintln!("e2b vision: {} tensors, 4 tokens x {}, rms {rms}", inst.weights.len(), e.len() / 4);
    assert!(e.len() % 4 == 0 && e.iter().all(|x| x.is_finite()) && rms > 0.0);
}
