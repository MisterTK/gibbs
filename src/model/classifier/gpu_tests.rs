//! Metal vs CPU parity of the encoder classifiers, end to end through `SequenceClassifier::load` /
//! `classify` on tiny synthetic repos (random f32 weights, a word-level tokenizer): the same family
//! spec instance compiled for each device. f32 vs f32, so the bound is a relative 1e-3 of the
//! largest score; per-architecture errors are printed.

use serde_json::json;
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
fn llama_decoder_seqcls_on_metal() {
    let (cfg, ts) = llama_seqcls();
    parity("llama-seqcls", cfg, ts);
}

#[test]
fn gpt_neox_decoder_seqcls_on_metal() {
    let (cfg, ts) = gpt_neox_seqcls();
    parity("gpt-neox-seqcls", cfg, ts);
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

/// A PEFT LoRA SEQ_CLS adapter (q/v LoRA under the task prefix `bert.`, a `modules_to_save`
/// classifier) on a bare `BertModel` base, merged by `peft::prepare_seq_cls`, scores like a plain
/// classifier repo holding the merged weights. CPU, and Metal when present.
#[test]
fn peft_seq_cls_adapter_on_bare_base_matches_merged_repo() {
    let (_, full) = bert(None);
    let bare = |n: &str| n.strip_prefix("bert.").map(str::to_string);
    let mut base = Ts::new(0);
    base.t = full.t.iter().filter_map(|(n, s, v)| Some((bare(n)?, s.clone(), v.clone()))).collect();
    let mut ad = Ts::new(9);
    let (r, alpha, h) = (2usize, 4.0f32, 16usize);
    let mut merged = full.t.clone();
    for l in 0..2 {
        for m in ["query", "value"] {
            let p = format!("bert.encoder.layer.{l}.attention.self.{m}");
            ad.w(&format!("base_model.model.{p}.lora_A.weight"), &[r, h]);
            ad.w(&format!("base_model.model.{p}.lora_B.weight"), &[h, r]);
            let (a, b) = (&ad.t[ad.t.len() - 2].2, &ad.t[ad.t.len() - 1].2);
            let w = &mut merged.iter_mut().find(|t| t.0 == format!("{p}.weight")).unwrap().2;
            for o in 0..h {
                for i in 0..h {
                    w[o * h + i] += alpha / r as f32 * (0..r).map(|k| b[o * r + k] * a[k * h + i]).sum::<f32>();
                }
            }
        }
    }
    ad.lin("base_model.model.classifier", 3, h);
    for (n, s, v) in &ad.t[ad.t.len() - 2..] {
        let n = n.strip_prefix("base_model.model.").unwrap().to_string();
        match merged.iter_mut().find(|t| t.0 == n) {
            Some(t) => t.2 = v.clone(),
            None => merged.push((n, s.clone(), v.clone())),
        }
    }
    let mut reference = Ts::new(0);
    reference.t = merged;

    let base_cfg = json!({"architectures": ["BertModel"], "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "intermediate_size": 24, "max_position_embeddings": 32, "type_vocab_size": 2});
    let cache = write_repo("peft-base", base_cfg.clone(), &base);
    let mut ref_cfg = base_cfg.clone();
    ref_cfg["architectures"] = json!(["BertForSequenceClassification"]);
    let ref_hub = repo("peft-ref", ref_cfg, &reference);
    // The adapter repo: its config names no architecture (labels only), so the base resolves it.
    let tmp = write_repo("peft-ad", json!({}), &ad);
    let (from, to) = (tmp.join("models--t--peft-ad"), cache.join("models--t--peft-ad"));
    let _ = std::fs::remove_dir_all(&to);
    std::fs::rename(&from, &to).unwrap();
    let snap = to.join("snapshots/main");
    std::fs::rename(snap.join("model.safetensors"), snap.join("adapter_model.safetensors")).unwrap();
    std::fs::write(
        snap.join("adapter_config.json"),
        json!({"peft_type": "LORA", "task_type": "SEQ_CLS", "r": r, "lora_alpha": alpha,
            "base_model_name_or_path": "t/peft-base"}).to_string(),
    )
    .unwrap();
    let hub = HuggingFaceHub::with_cache_dir(cache);
    let _g = cpu_compute::allow();
    let mut devs = vec![Device::Cpu];
    if available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    for dev in devs {
        let want = SequenceClassifier::load(&ref_hub, "t/peft-ref", dev).unwrap();
        let got = SequenceClassifier::load(&hub, "t/peft-ad", dev).unwrap_or_else(|e| panic!("{dev:?}: {e}"));
        assert_eq!(got.architecture, "BertForSequenceClassification");
        for pair in [None, Some(PAIR)] {
            let (a, _) = want.classify(TEXT, pair).unwrap();
            let (b, _) = got.classify(TEXT, pair).unwrap();
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.label, y.label, "{dev:?}");
                assert!((x.score - y.score).abs() < 1e-5, "{dev:?}: {a:?} vs {b:?}");
            }
        }
    }
}

/// Decoder seq-cls on the spec (`runtime::seqcls`) against the softmax scores of `TEXT` the
/// deleted GraphLm path gave (`GraphLmSeq`: host `final_hidden` · `score.weight`, commit
/// b73c897), CPU and Metal. GPT-NeoX differs by ~1e-4: GraphLm ran HF `gelu` as the tanh
/// approximation, the spec as exact erf GELU (transformers' `gelu`).
#[test]
fn decoder_seqcls_matches_old_graph_lm() {
    const OLD: [(&str, [f32; 3]); 2] = [
        ("llama", [0.8506544, 0.0595768, 0.08976873]),
        ("gpt-neox", [0.39762998, 0.3666832, 0.23568688]),
    ];
    let mut devs = vec![Device::Cpu];
    if available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    for ((name, want), (cfg, ts)) in OLD.into_iter().zip([llama_seqcls(), gpt_neox_seqcls()]) {
        let name = format!("{name}-seqcls-old");
        let hub = repo(&name, cfg, &ts);
        for &dev in &devs {
            let _g = cpu_compute::allow();
            let c = SequenceClassifier::load(&hub, &format!("t/{name}"), dev).unwrap();
            let (got, _) = c.classify(TEXT, None).unwrap();
            let g: Vec<f32> = ["A", "B", "C"].iter().map(|l| got.iter().find(|s| &s.label == l).unwrap().score).collect();
            let (rel, cos) = crate::arch::testing::compare(&g, &want);
            eprintln!("{name} {dev:?}: rel {rel:.2e} cos {cos:.8}");
            assert!(rel < 1e-3 && cos > 0.9999, "{name} {dev:?}: {g:?} vs {want:?}");
        }
    }
}
