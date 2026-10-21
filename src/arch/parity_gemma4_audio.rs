//! Parity of the `gemma4` / `gemma4_unified` audio specs with `model::gemma4::{audio, unified}`
//! (deleted with them): the spec's `runtime.media.audio` preprocessing and tower instance against
//! the old encoders on the same WAVs.

use super::expr::Val;
use super::testing::{compare, Runner};
use crate::graph::Tensor;
use crate::hardware::{available_devices, Device, PlacementMode};
use crate::model::gemma4::audio::{tiny, Audio, Features};
use crate::model::gemma4::unified::AudioEmbedder;
use crate::model::gemma4::Encoder;
use crate::runtime::media::{self, Expand};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The first golden fixture dir of `cases` that exists (same checkpoint under several case names;
/// `cargo test --test golden <case>` writes them), `None` when none does.
fn fixture(cases: &[&str]) -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/golden-fixtures");
    let d = cases.iter().map(|c| root.join(c)).find(|d| d.exists());
    if d.is_none() {
        eprintln!("skip: run `cargo test --test golden {}` first", cases[0].trim_start_matches("gemma_"));
    }
    d
}

fn json(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

/// Two tones at 16 kHz (every frame valid), a shorter 16 kHz tone (padding frames and padded
/// attention rows) and a 22.05 kHz tone (resampled).
fn clips() -> Vec<Vec<u8>> {
    use crate::model::gemma4::unified::tests::wav;
    vec![tiny::wav(), wav(16_000, 2600), wav(22_050, 3000)]
}

fn devices() -> Vec<Device> {
    let mut devs = vec![Device::Cpu];
    if available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    devs
}

/// Per device: the spec's `runtime.media.audio` (expand ids from the config, preprocessing equal
/// to `old_input(clip)`), then its tower instance against `old(clip, on_metal)`; every audio
/// tensor is read.
fn check(
    case: &str,
    dir: &Path,
    old_input: &dyn Fn(&[u8]) -> Vec<f32>,
    old: &dyn Fn(&[u8], bool) -> (Vec<f32>, usize),
) {
    let cfg = json(&dir.join("config.json"));
    let id = |k: &str| cfg[k].as_u64().or_else(|| cfg["eoa_token_index"].as_u64()).unwrap() as u32;
    for dev in devices() {
        let mut r = Runner::load(dir, None, dev, 64).unwrap();
        let m = media::resolve(&r.model.spec.runtime["media"], &r.model.env).unwrap();
        let a = &m["audio"];
        assert!(
            matches!(a.expand, Expand::RepeatBetween { placeholder, open, close }
                if (placeholder, open, close) == (id("audio_token_id"), id("boa_token_id"), id("eoa_token_id"))),
            "{case}: {:?}",
            a.expand
        );
        for (i, clip) in clips().iter().enumerate() {
            let p = media::preprocess(&a.preprocess, clip).unwrap();
            assert_eq!(p.data, old_input(clip), "{case} clip {i}: preprocessing");
            let vars: Vec<(&str, Val)> = p.grid.vars.iter().map(|&(k, v)| (k, Val::Num(v as f64))).collect();
            let mut out = r.run(&a.component, &vars, vec![Tensor::F32(p.data)]).unwrap();
            let got = match out.remove("embeds") {
                Some(Tensor::F32(v)) => v,
                o => panic!("{case}: output {o:?}"),
            };
            for on_metal in [false, dev == Device::Metal] {
                let (want, n) = old(clip, on_metal);
                assert_eq!((p.grid.tokens, got.len()), (n, want.len()), "{case} clip {i}: tokens");
                let (rel, cos) = compare(&got, &want);
                let old_dev = if on_metal { "old Metal" } else { "old host" };
                eprintln!("{case} {dev:?} vs {old_dev}, clip {i}: {n} tokens {:?}, rel {rel:.3e} cos {cos:.7}", p.grid.vars);
                assert!(rel < 1e-3 && cos > 0.9999, "{case} {dev:?} clip {i}: rel {rel:e} cos {cos}");
            }
        }
        let unread: Vec<_> = r.unconsumed().into_iter().filter(|n| n.contains("audio")).collect();
        assert!(unread.is_empty(), "{case}: unread {unread:?}");
    }
}

/// `gemma4_audio` conformer against `Audio::encode` (host forward and Metal graph).
#[test]
fn audio_matches_old_conformer() {
    let _g = crate::hardware::cpu_compute::allow();
    let Some(dir) = fixture(&["gemma_gemma4_audio", "gemma_gemma4_vision", "gemma_gemma4_decode"]) else { return };
    let (cfg, processor) = (json(&dir.join("config.json")), json(&dir.join("processor_config.json")));
    let w = crate::model::weights::Weights::open(&dir.join("model.safetensors")).unwrap();
    let host = Audio::load(&w, &cfg["audio_config"], &processor).unwrap();
    let metal = Audio::load(&w, &cfg["audio_config"], &processor).unwrap().on_gpu(PlacementMode::Pin(Device::Metal));
    let fe = Features::new(&processor["feature_extractor"]).unwrap();
    check(
        "gemma4 audio",
        &dir,
        &|clip| fe.frames(clip).unwrap().0,
        &|clip, gpu| (if gpu { metal.encode(clip) } else { host.encode(clip) }).unwrap(),
    );
}

/// `gemma4_unified` raw-frame embedder against `AudioEmbedder::encode`.
#[test]
fn unified_audio_matches_old_embedder() {
    let _g = crate::hardware::cpu_compute::allow();
    let Some(dir) = fixture(&["gemma_gemma4_unified_audio", "gemma_gemma4_unified_vision"]) else { return };
    let (cfg, processor) = (json(&dir.join("config.json")), json(&dir.join("processor_config.json")));
    let w = crate::model::weights::Weights::open(&dir.join("model.safetensors")).unwrap();
    let load = || AudioEmbedder::load(&w, &cfg["audio_config"], &processor).unwrap();
    let (host, metal) = (load(), load().on_gpu(PlacementMode::Pin(Device::Metal)));
    let frame = processor["feature_extractor"]["audio_samples_per_token"].as_u64().unwrap() as usize;
    check(
        "gemma4_unified audio",
        &dir,
        &|clip| {
            let (pcm, rate) = crate::model::audio::read_wav(clip).unwrap();
            let mut x = crate::model::audio::resample(&pcm, rate, 16_000);
            x.resize(x.len().div_ceil(frame) * frame, 0.0);
            x
        },
        &|clip, gpu| (if gpu { metal.encode(clip) } else { host.encode(clip) }).unwrap(),
    );
}

/// The real E2B GGUF + `mmproj`: the renamed `gemma4a` conformer builds with every weight present,
/// reads every audio tensor, and runs a 0.25 s clip on the CPU executor.
#[test]
#[ignore = "reads the 1 GB E2B mmproj"]
fn e2b_gguf_mmproj_audio_tower_loads() {
    let _g = crate::hardware::cpu_compute::allow();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("disk/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/main");
    if !dir.exists() {
        eprintln!("skip: no E2B GGUF");
        return;
    }
    let a = crate::model::hub::ModelArtifacts {
        model_id: "unsloth/gemma-4-E2B-it-qat-GGUF".into(),
        local_dir: Some(dir.clone()),
        weights_path: Some(dir.join("gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf")),
        is_cached: true,
        ..Default::default()
    };
    let mut r = Runner::from_artifacts(&a, None, Device::Cpu, 64).unwrap();
    assert_eq!(r.model.spec.family, "gemma4");
    let m = media::resolve(&r.model.spec.runtime["media"], &r.model.env).unwrap();
    let audio = &m["audio"];
    eprintln!("e2b audio media: {audio:?}");
    let unmapped: Vec<_> = r.model.renames.iter().filter(|(raw, c)| raw == c && (raw.starts_with("a.") || raw.starts_with("mm.a."))).collect();
    assert!(unmapped.is_empty(), "unmapped mmproj audio tensors {unmapped:?}");
    let p = media::preprocess(&audio.preprocess, &tiny::wav()).unwrap();
    let vars: Vec<(&str, Val)> = p.grid.vars.iter().map(|&(k, v)| (k, Val::Num(v as f64))).collect();
    let inst = r.instance("audio", &vars).unwrap();
    let towers: Vec<_> = r.model.renames.iter().map(|(_, c)| c).filter(|c| c.starts_with("model.audio_tower") || c.starts_with("model.embed_audio")).collect();
    let unread: Vec<_> = towers.iter().filter(|c| !inst.weights.contains(**c)).collect();
    assert!(unread.is_empty(), "audio tensors the tower does not read: {unread:?}");
    let out = r.run("audio", &vars, vec![Tensor::F32(p.data)]).unwrap();
    let Some(Tensor::F32(e)) = out.get("embeds") else { panic!("no embeds") };
    let rms = (e.iter().map(|x| x * x).sum::<f32>() / e.len() as f32).sqrt();
    eprintln!("e2b audio: {} tensors, {} tokens x {}, rms {rms}", inst.weights.len(), p.grid.tokens, e.len() / p.grid.tokens);
    assert!(e.len() % p.grid.tokens == 0 && e.iter().all(|x| x.is_finite()) && rms > 0.0);
}
