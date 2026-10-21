//! llama.cpp BERT GGUF encoder spec (`bert_gguf`) against the old `GgufClassifier::logits` CPU
//! forward (jev `BertEncoder` + host head, commit 251be4c; outputs pinned below) on tiny GGUF
//! fixtures: a BERT export (2 token types, bare `cls.output`, 3 labels) and an XLM-R-like one (1-D
//! 1-row token-type table, `cls` + tanh pooler, 1-D single-label `cls.output`). CPU and Metal.

use super::expr::Val;
use super::testing::{compare, Rng, Runner};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::hub::ModelArtifacts;
use crate::model::loaders::gguf::test_gguf_file;

const H: usize = 16;
const HEADS: u32 = 2;
const INTER: usize = 32;
const LAYERS: usize = 2;
const POS: usize = 16;
const TOKENS: [&str; 8] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "\u{2581}hello", "\u{2581}world", "\u{2581}the", "ing"];

fn s(v: &str) -> Vec<u8> {
    let mut b = (v.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(v.as_bytes());
    b
}

/// A GGUF of the llama.cpp BERT layout: `(name, row-major shape, GGUF dims, values)` per tensor.
fn fixture(types: usize, pooler: bool, labels: usize) -> Vec<u8> {
    let mut rng = Rng(7 + types as u64 + labels as u64);
    let mut ts: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut push = |name: String, shape: &[usize], gain: bool| {
        let n: usize = shape.iter().product();
        let v = (0..n).map(|_| if gain { 1.0 + 0.1 * rng.unit() } else { 0.3 * rng.unit() }).collect();
        // GGUF dims fastest first; the llama.cpp writer drops trailing unit dims.
        let mut dims: Vec<usize> = shape.iter().rev().copied().collect();
        while dims.len() > 1 && dims.last() == Some(&1) {
            dims.pop();
        }
        ts.push((name, dims, v));
    };
    push("token_embd.weight".into(), &[TOKENS.len(), H], false);
    push("position_embd.weight".into(), &[POS, H], false);
    push("token_types.weight".into(), &[types, H], false);
    push("token_embd_norm.weight".into(), &[H], true);
    push("token_embd_norm.bias".into(), &[H], false);
    for l in 0..LAYERS {
        for (m, o, i) in [("attn_q", H, H), ("attn_k", H, H), ("attn_v", H, H), ("attn_output", H, H), ("ffn_up", INTER, H), ("ffn_down", H, INTER)] {
            push(format!("blk.{l}.{m}.weight"), &[o, i], false);
            push(format!("blk.{l}.{m}.bias"), &[o], false);
        }
        for m in ["attn_output_norm", "layer_output_norm"] {
            push(format!("blk.{l}.{m}.weight"), &[H], true);
            push(format!("blk.{l}.{m}.bias"), &[H], false);
        }
    }
    if pooler {
        push("cls.weight".into(), &[H, H], false);
        push("cls.bias".into(), &[H], false);
    }
    push("cls.output.weight".into(), &[labels, H], false);
    push("cls.output.bias".into(), &[labels], false);

    let arr = |elem: u32, items: Vec<Vec<u8>>| {
        let mut b = elem.to_le_bytes().to_vec();
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        items.into_iter().for_each(|i| b.extend(i));
        b
    };
    let u32b = |v: u32| v.to_le_bytes().to_vec();
    let kv = [
        ("general.architecture", 8, s("bert")),
        ("bert.embedding_length", 4, u32b(H as u32)),
        ("bert.block_count", 4, u32b(LAYERS as u32)),
        ("bert.attention.head_count", 4, u32b(HEADS)),
        ("bert.feed_forward_length", 4, u32b(INTER as u32)),
        ("bert.attention.layer_norm_epsilon", 6, 1e-12f32.to_le_bytes().to_vec()),
        ("tokenizer.ggml.model", 8, s("bert")),
        ("tokenizer.ggml.tokens", 9, arr(8, TOKENS.iter().map(|t| s(t)).collect())),
        // Specials are control tokens (3), the rest normal (1).
        ("tokenizer.ggml.token_type", 9, arr(5, (0..TOKENS.len()).map(|i| u32b(if i < 4 { 3 } else { 1 })).collect())),
    ];
    let tensors: Vec<(&str, &[usize], &[f32])> = ts.iter().map(|(n, d, v)| (n.as_str(), d.as_slice(), v.as_slice())).collect();
    test_gguf_file(&kv, &tensors)
}

#[test]
fn bert_gguf_spec_matches_old_classifier() {
    let mut devs = vec![Device::Cpu];
    if crate::hardware::available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    let _g = crate::hardware::cpu_compute::allow();
    // Old `GgufClassifier` tokenization of "hello world" / ("hello the", "world") under TOKENS.
    let cases: [(Vec<u32>, Vec<u32>); 2] = [(vec![2, 4, 5, 3], vec![0; 4]), (vec![2, 4, 6, 3, 5, 3], vec![0, 0, 0, 0, 1, 1])];
    // Old `GgufClassifier::logits` (CPU, raw) per case.
    let old: [(&str, usize, bool, usize, [&[f32]; 2]); 2] = [
        ("bert", 2, false, 3, [&[0.38424462, 0.3168, -0.10926959], &[0.2953179, 0.6068759, 0.06610335]]),
        ("xlmr", 1, true, 1, [&[0.55778867], &[0.5102308]]),
    ];
    for (name, types, pooler, labels, wants) in old {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, fixture(types, pooler, labels)).unwrap();
        let a = ModelArtifacts {
            model_id: format!("test/{name}-gguf"),
            local_dir: Some(dir.path().to_path_buf()),
            weights_path: Some(path.clone()),
            is_cached: true,
            ..Default::default()
        };
        for (c, ((ids, tt), want)) in cases.iter().zip(wants).enumerate() {
            let n = ids.len();
            for &dev in &devs {
                let mut r = Runner::from_artifacts(&a, Some("bert_gguf"), dev, 64).unwrap_or_else(|e| panic!("{name}: {e}"));
                let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("classify".into()))];
                let mut ins = vec![Tensor::U32(ids.clone()), Tensor::U32((0..n as u32).collect())];
                if types > 1 {
                    ins.push(Tensor::U32(tt.clone()));
                }
                let got = match r.run("encoder", &vars, ins).unwrap_or_else(|e| panic!("{name}: {e}")).remove("logits") {
                    Some(Tensor::F32(v)) => v,
                    o => panic!("{name}: logits {o:?}"),
                };
                let (rel, cos) = compare(&got, want);
                eprintln!("{name} pair={} {dev:?}: rel {rel:.2e} cos {cos:.8}", c == 1);
                assert!(rel < 1e-3 && cos > 0.9999, "{name} {dev:?}: {got:?} vs {want:?}");
                assert!(r.unconsumed().is_empty(), "{name}: unconsumed {:?}", r.unconsumed());
            }
        }
    }
}
