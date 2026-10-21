//! Metal vs CPU parity of the encoder classifiers, end to end through `SequenceClassifier::load` /
//! `classify` on tiny synthetic repos (random f32 weights, a word-level tokenizer): the same family
//! spec instance compiled for each device. f32 vs f32, so the bound is a relative 1e-3 of the
//! largest score; per-architecture errors are printed.

use super::*;
use crate::hardware::{available_devices, cpu_compute};

use super::synth::*;

fn repo(name: &str, cfg: Value, ts: &Ts) -> HuggingFaceHub {
    HuggingFaceHub::with_cache_dir(write_repo(name, cfg, ts))
}

/// Loads `name` on the CPU (host forward) and on Metal and compares every label's score for
/// (`TEXT`, none) and (`TEXT`, `PAIR`).
fn parity(name: &str, cfg: Value, ts: Ts) {
    if !available_devices().contains(&Device::Metal) {
        eprintln!("SKIP {name}: no Metal device");
        return;
    }
    // Every graph run here reproduces `graph::shape::infer`.
    let _shapes = crate::graph::shape::CheckCompiled::new();
    let hub = repo(name, cfg, &ts);
    let id = format!("t/{name}");
    let _g = cpu_compute::allow();
    let cpu = SequenceClassifier::load(&hub, &id, Device::Cpu).unwrap();
    let gpu = SequenceClassifier::load(&hub, &id, Device::Metal).unwrap();
    assert_eq!(
        (cpu.device(), gpu.device()),
        (Device::Cpu, Device::Metal),
        "{name}"
    );
    for pair in [None, Some(PAIR)] {
        let (want, n) = cpu.classify(TEXT, pair).unwrap();
        let (got, m) = gpu.classify(TEXT, pair).unwrap();
        assert_eq!(n, m);
        let key = |l: &[LabelScore]| {
            let mut v: Vec<(String, f32)> = l.iter().map(|x| (x.label.clone(), x.score)).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        let (want, got) = (key(&want), key(&got));
        let scale = want.iter().fold(1e-3f32, |m, (_, s)| m.max(s.abs()));
        let err = want
            .iter()
            .zip(&got)
            .fold(0f32, |m, (a, b)| m.max((a.1 - b.1).abs()))
            / scale;
        eprintln!(
            "{name} pair={}: max rel err {err:.2e}  cpu {want:?}  metal {got:?}",
            pair.is_some()
        );
        assert!(
            err < 1e-3,
            "{name}: Metal scores differ from the CPU forward by {err}: {want:?} vs {got:?}"
        );
    }
}

#[test]
fn bert_softmax_head_on_metal() {
    let (cfg, ts) = bert(None);
    parity("bert", cfg, ts);
}

#[test]
fn bert_multi_label_sigmoid_head_on_metal() {
    let (cfg, ts) = bert(Some("multi_label_classification"));
    parity("bert-ml", cfg, ts);
}

#[test]
fn modernbert_cls_pooling_windowed_attention_on_metal() {
    let (cfg, ts) = modernbert("cls", false);
    parity("modernbert-cls", cfg, ts);
}

/// Mean pooling and a YaRN attention factor (cos/sin scale != 1).
#[test]
fn modernbert_mean_pooling_yarn_rope_on_metal() {
    let (cfg, ts) = modernbert("mean", true);
    parity("modernbert-mean", cfg, ts);
}

#[test]
fn electra_projected_embeddings_on_metal() {
    let (cfg, ts) = electra();
    parity("electra", cfg, ts);
}

#[test]
fn albert_shared_layers_on_metal() {
    let (cfg, ts) = albert();
    parity("albert", cfg, ts);
}

#[test]
fn deberta_v2_log_buckets_on_metal() {
    let (cfg, ts) = deberta_v2();
    parity("deberta-v2", cfg, ts);
}

#[test]
fn deberta_v1_absolute_positions_on_metal() {
    let (cfg, ts) = deberta_v1();
    parity("deberta-v1", cfg, ts);
}
