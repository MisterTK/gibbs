//! RoBERTa and DistilBERT sequence classifiers through the `bert` spec (`SequenceClassifier::load`),
//! CPU and Metal, against the old host classifier's scores (`classify` on the CPU at 251be4c,
//! softmax over labels A, B, C) pinned below. `TEXT` holds token 1 (RoBERTa's pad id) on purpose.

use super::testing::compare;
use crate::hardware::{available_devices, cpu_compute, Device};
use crate::model::hub::HuggingFaceHub;
use crate::model::classifier::synth::{write_repo, Ts, PAIR, TEXT, VOCAB};
use crate::model::classifier::SequenceClassifier;
use serde_json::{json, Value};

fn roberta() -> (Value, Ts) {
    let mut ts = Ts::new(7);
    ts.w("roberta.embeddings.word_embeddings.weight", &[VOCAB, 16]);
    ts.w("roberta.embeddings.position_embeddings.weight", &[34, 16]);
    ts.w("roberta.embeddings.token_type_embeddings.weight", &[1, 16]);
    ts.ln("roberta.embeddings.LayerNorm", 16);
    for l in 0..2 {
        let p = format!("roberta.encoder.layer.{l}");
        for s in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"] {
            ts.lin(&format!("{p}.{s}"), 16, 16);
        }
        ts.ln(&format!("{p}.attention.output.LayerNorm"), 16);
        ts.lin(&format!("{p}.intermediate.dense"), 24, 16);
        ts.lin(&format!("{p}.output.dense"), 16, 24);
        ts.ln(&format!("{p}.output.LayerNorm"), 16);
    }
    ts.lin("classifier.dense", 16, 16);
    ts.lin("classifier.out_proj", 3, 16);
    let cfg = json!({"architectures": ["RobertaForSequenceClassification"], "model_type": "roberta",
        "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 24,
        "max_position_embeddings": 34, "type_vocab_size": 1, "pad_token_id": 1, "layer_norm_eps": 1e-5});
    (cfg, ts)
}

fn distilbert() -> (Value, Ts) {
    let mut ts = Ts::new(11);
    ts.w("distilbert.embeddings.word_embeddings.weight", &[VOCAB, 16]);
    ts.w("distilbert.embeddings.position_embeddings.weight", &[32, 16]);
    ts.ln("distilbert.embeddings.LayerNorm", 16);
    for l in 0..2 {
        let p = format!("distilbert.transformer.layer.{l}");
        for s in ["attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin"] {
            ts.lin(&format!("{p}.{s}"), 16, 16);
        }
        ts.ln(&format!("{p}.sa_layer_norm"), 16);
        ts.lin(&format!("{p}.ffn.lin1"), 24, 16);
        ts.lin(&format!("{p}.ffn.lin2"), 16, 24);
        ts.ln(&format!("{p}.output_layer_norm"), 16);
    }
    ts.lin("pre_classifier", 16, 16);
    ts.lin("classifier", 3, 16);
    let cfg = json!({"architectures": ["DistilBertForSequenceClassification"], "model_type": "distilbert",
        "dim": 16, "n_layers": 2, "n_heads": 2, "hidden_dim": 24, "max_position_embeddings": 32});
    (cfg, ts)
}

/// (fixture, pair input, old scores for A, B, C).
const PINNED: [(&str, bool, [f32; 3]); 4] = [
    ("roberta", false, [0.23316306, 0.5075492, 0.2592877]),
    ("roberta", true, [0.23566867, 0.50305873, 0.26127255]),
    ("distilbert", false, [0.4171669, 0.20411296, 0.37872016]),
    ("distilbert", true, [0.41865396, 0.20233509, 0.37901092]),
];

#[test]
fn bert_variants_match_old_classifier() {
    let mut devs = vec![Device::Cpu];
    if available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    let _g = cpu_compute::allow();
    for (fixture, (cfg, ts)) in [("roberta", roberta()), ("distilbert", distilbert())] {
        let name = format!("spec-variant-{fixture}");
        let hub = HuggingFaceHub::with_cache_dir(write_repo(&name, cfg, &ts));
        for &dev in &devs {
            let c = SequenceClassifier::load(&hub, &format!("t/{name}"), dev).unwrap_or_else(|e| panic!("{fixture}: {e}"));
            assert_eq!(c.device(), dev, "{fixture}");
            for pair in [false, true] {
                let (mut l, _) = c.classify(TEXT, pair.then_some(PAIR)).unwrap();
                assert_eq!(l.len(), 3, "{fixture}");
                l.sort_by(|a, b| a.label.cmp(&b.label));
                let got: Vec<f32> = l.iter().map(|x| x.score).collect();
                let want = PINNED.iter().find(|p| p.0 == fixture && p.1 == pair).unwrap().2;
                let (rel, cos) = compare(&got, &want);
                eprintln!("{fixture} pair={pair} {dev:?}: rel {rel:.2e} cos {cos:.8} got {got:?}");
                assert!(rel < 1e-3 && cos > 0.9999, "{fixture} pair={pair} {dev:?}: {got:?} vs {want:?}");
            }
        }
    }
}
