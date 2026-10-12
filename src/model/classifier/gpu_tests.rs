//! Metal vs CPU parity of the encoder classifiers, end to end through `SequenceClassifier::load` /
//! `classify` on tiny synthetic repos (random f32 weights, a word-level tokenizer): the CPU body is
//! the native host forward (the reference), the Metal body is the graph program. f32 vs f32, so the
//! bound is a relative 1e-3 of the largest score; per-architecture errors are printed.

use super::*;
use crate::hardware::{available_devices, cpu_compute};
use serde_json::json;


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
    let hub = repo(name, cfg, &ts);
    let id = format!("t/{name}");
    let _g = cpu_compute::allow();
    let cpu = SequenceClassifier::load(&hub, &id, Device::Cpu).unwrap();
    let gpu = SequenceClassifier::load(&hub, &id, Device::Metal).unwrap();
    assert_eq!((cpu.device(), gpu.device()), (Device::Cpu, Device::Metal), "{name}");
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
        let err = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max((a.1 - b.1).abs())) / scale;
        eprintln!("{name} pair={}: max rel err {err:.2e}  cpu {want:?}  metal {got:?}", pair.is_some());
        assert!(err < 1e-3, "{name}: Metal scores differ from the CPU forward by {err}: {want:?} vs {got:?}");
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

/// SetFit pooling (mean/cls/max, optional normalise) and sklearn `predict_proba` forms on the
/// BERT encoder graph vs the host `pool` + `Logistic::proba`.
#[test]
fn setfit_pool_and_logistic_head_on_metal() {
    use super::setfit::{pool, Logistic, Pool};
    if !available_devices().contains(&Device::Metal) {
        return;
    }
    let dims = BertDims {
        hidden_size: 16,
        num_layers: 2,
        num_heads: 2,
        intermediate_size: 24,
        max_positions: 32,
        vocab_size: VOCAB,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
    };
    let ts = bert(None).1;
    let get = |name: &str, _: &[usize]| -> Result<Vec<f32>, String> {
        let key = format!("bert.{name}");
        Ok(ts.t.iter().find(|t| t.0 == key).unwrap().2.clone())
    };
    let _g = cpu_compute::allow();
    let cpu = BertEncoder::load(dims, Device::Cpu, get).unwrap();
    let (ids, pos, types) = ([3u32, 7, 1, 9, 2, 5, 11, 4, 6], (0..9).collect::<Vec<u32>>(), [0u32; 9]);
    let hidden = cpu.run(&ids, &pos, &types, false).unwrap();
    for (p, normalize, rows, ovr) in [
        (Pool::Mean, true, 1, true),
        (Pool::Max, false, 3, false),
        (Pool::Cls, false, 3, true),
        (Pool::Mean, false, 1, false),
    ] {
        let mut rng = Ts::new(rows as u64 + 7);
        let head = Logistic {
            coef: (0..rows * 16).map(|_| rng.rnd() as f64).collect(),
            intercept: (0..rows).map(|_| rng.rnd() as f64).collect(),
            features: 16,
            ovr,
        };
        let want = head.proba(&pool(&hidden, 16, p, normalize));
        let gpu = BertEncoder::load_with_head(dims, Device::Metal, get, Some(head.graph(p, normalize))).unwrap();
        let got = gpu.run_head(&ids, &pos, &types).unwrap();
        assert_eq!(got.len(), want.len());
        let err = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("setfit {p:?} norm={normalize} rows={rows} ovr={ovr}: max abs err {err:.2e}  {want:?} vs {got:?}");
        assert!(err < 1e-3, "{p:?} {normalize} {rows} {ovr}: {want:?} vs {got:?}");
    }
}
