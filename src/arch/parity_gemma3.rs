//! Parity of the family spec with the pre-spec implementation (deleted with it).

use super::testing::{compare, Runner, Weights};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::classifier::DecoderBackbone;
use crate::model::{bf16_to_f32, f32_to_bf16};
use serde_json::{json, Map, Value as J};
use std::path::Path;

fn close(bad: &mut Vec<String>, what: &str, got: &[f32], want: &[f32]) {
    let (rel, cos) = compare(got, want);
    eprintln!("{what}: rel {rel:.3e} cos {cos:.6}");
    if !(rel < 1e-3 && cos > 0.9999) {
        bad.push(format!("{what}: rel {rel:.3e} cos {cos:.6}"));
    }
}

/// `w` as a BF16 `dir/model.safetensors` (values rounded to bf16 in `w` too).
fn write_bf16(w: &mut Weights, dir: &Path) {
    let (mut header, mut data) = (Map::new(), Vec::<u8>::new());
    for (name, shape, v) in &mut w.tensors {
        let start = data.len();
        for x in v.iter_mut() {
            let b = f32_to_bf16(*x);
            *x = bf16_to_f32(b);
            data.extend(b.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&J::Object(header)).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

const V: usize = 96;

/// The `gemma_gemma3_seqcls` golden fixture (tests/golden/gemma.rs) with hidden `h`, bf16 or f32
/// weights and tight (saturating) soft caps: 3 layers (two sliding, window 3, one global with
/// yarn rope), GQA 4/2, `query_pre_attn_scalar` 24, 3 labels.
fn fixture(dir: &Path, h: usize, bf16: bool) -> Weights {
    let (nh, kvh, hd, inter) = (4, 2, 16, 96);
    let mut w = Weights::new(0x6e6d_6133);
    w.w("model.embed_tokens.weight", &[V, h]);
    for l in 0..3 {
        let p = format!("model.layers.{l}");
        w.w(format!("{p}.self_attn.q_proj.weight"), &[nh * hd, h]);
        w.w(format!("{p}.self_attn.k_proj.weight"), &[kvh * hd, h]);
        w.w(format!("{p}.self_attn.v_proj.weight"), &[kvh * hd, h]);
        w.w(format!("{p}.self_attn.o_proj.weight"), &[h, nh * hd]);
        w.norm(format!("{p}.self_attn.q_norm.weight"), hd, 0.0);
        w.norm(format!("{p}.self_attn.k_norm.weight"), hd, 0.0);
        w.w(format!("{p}.mlp.gate_proj.weight"), &[inter, h]);
        w.w(format!("{p}.mlp.up_proj.weight"), &[inter, h]);
        w.w(format!("{p}.mlp.down_proj.weight"), &[h, inter]);
        for n in [
            "input",
            "post_attention",
            "pre_feedforward",
            "post_feedforward",
        ] {
            w.norm(format!("{p}.{n}_layernorm.weight"), h, 0.0);
        }
    }
    w.norm("model.norm.weight", h, 0.0);
    w.w("score.weight", &[3, h]);
    let cfg = json!({
        "architectures": ["Gemma3TextForSequenceClassification"], "model_type": "gemma3_text",
        "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 3,
        "num_attention_heads": nh, "num_key_value_heads": kvh, "head_dim": hd,
        "vocab_size": V, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        "sliding_window": 3, "query_pre_attn_scalar": 24, "attn_logit_softcapping": ATTN_CAP,
        "final_logit_softcapping": FINAL_CAP, "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 1e6, "rope_local_base_freq": 10000.0,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16},
        "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1,
        "id2label": {"0": "neg", "1": "neu", "2": "pos"},
    });
    w.write_checkpoint(dir, &cfg, V);
    if bf16 {
        write_bf16(&mut w, dir);
    }
    w
}

const ATTN_CAP: f32 = 1.0;
const FINAL_CAP: f32 = 0.5;

/// The spec's text instance against `Gemma3Text`'s host forward (`last_hidden`): tied soft-capped
/// logits of a 6-row prefill (heads `all` and `last`) and 3 cached decode steps, the final hidden
/// (head `hidden`) and the seq-cls `score` head (row `last_index`, also with trailing padding).
#[test]
fn gemma3_text_cpu() {
    text_and_seqcls(Device::Cpu);
}

/// Run through the central scheduler (Metal).
#[test]
fn gemma3_text_metal() {
    if crate::hardware::available_devices().contains(&Device::Metal) {
        text_and_seqcls(Device::Metal);
    }
}

fn text_and_seqcls(dev: Device) {
    let _g = crate::hardware::cpu_compute::allow();
    let mut bad = Vec::new();
    // bf16 with a square hidden (bf16(sqrt(64)) = 8); f32 with sqrt(40) unrounded.
    for (h, bf16) in [(64, true), (40, false)] {
        let case = format!("h{h} bf16 {bf16}");
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let w = fixture(dir, h, bf16);
        let tensor = |n: &str| w.tensors.iter().find(|t| t.0 == n).unwrap().2.clone();
        let (embed, score) = (tensor("model.embed_tokens.weight"), tensor("score.weight"));
        let cfg: J =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let host =
            crate::model::gemma3_text::Gemma3Text::load(&super::testing::artifacts(dir), &cfg)
                .unwrap();
        let dot = |m: &[f32], h: &[f32]| -> Vec<f32> {
            m.chunks(h.len())
                .map(|r| r.iter().zip(h).map(|(a, b)| a * b).sum())
                .collect()
        };
        let logits = |ids: &[u32]| -> Vec<f32> {
            dot(&embed, &host.last_hidden(ids).unwrap())
                .into_iter()
                .map(|x| (x / FINAL_CAP).tanh() * FINAL_CAP)
                .collect()
        };
        let seq: [u32; 9] = [2, 5, 17, 9, 40, 77, 31, 64, 8];
        let (prompt, decode) = seq.split_at(6);
        {
            let mut r = Runner::load(dir, None, dev, 64).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(r.model.spec.family, "gemma3", "detected family");
            let all = r.lm(prompt, 0, "all", "logits", vec![]);
            for (i, row) in all.chunks(V).enumerate() {
                close(
                    &mut bad,
                    &format!("{case} {dev:?} prefill row {i}"),
                    row,
                    &logits(&prompt[..=i]),
                );
            }
            let last = r.lm(prompt, 0, "last", "logits", vec![]);
            close(
                &mut bad,
                &format!("{case} {dev:?} prefill last"),
                &last,
                &logits(prompt),
            );
            for (i, &t) in decode.iter().enumerate() {
                let got = r.lm(&[t], prompt.len() + i, "last", "logits", vec![]);
                close(
                    &mut bad,
                    &format!("{case} {dev:?} decode {i}"),
                    &got,
                    &logits(&seq[..prompt.len() + i + 1]),
                );
            }
            let hidden = r.lm(&seq, 0, "hidden", "hidden", vec![]);
            let want = host.last_hidden(&seq).unwrap();
            close(
                &mut bad,
                &format!("{case} {dev:?} hidden"),
                &hidden[hidden.len() - want.len()..],
                &want,
            );
            // Seq-cls: last row, then row 5 of the prompt followed by two pad ids.
            let got = r.lm(prompt, 0, "score", "logits", vec![Tensor::U32(vec![5])]);
            close(
                &mut bad,
                &format!("{case} {dev:?} score"),
                &got,
                &dot(&score, &host.last_hidden(prompt).unwrap()),
            );
            let padded: Vec<u32> = prompt.iter().copied().chain([0, 0]).collect();
            let got = r.lm(&padded, 0, "score", "logits", vec![Tensor::U32(vec![5])]);
            close(
                &mut bad,
                &format!("{case} {dev:?} score padded"),
                &got,
                &dot(&score, &host.last_hidden(prompt).unwrap()),
            );
            assert_eq!(
                r.unconsumed(),
                Vec::<String>::new(),
                "unconsumed checkpoint tensors"
            );
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// `src` safetensors plus `extra` tensors (BF16, values rounded in place) as `dst`.
fn append_bf16(src: &Path, dst: &Path, extra: &mut [(String, Vec<usize>, Vec<f32>)]) {
    let bytes = std::fs::read(src).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: Map<String, J> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let mut data = bytes[8 + n..].to_vec();
    for (name, shape, v) in extra {
        let start = data.len();
        for x in v.iter_mut() {
            let b = f32_to_bf16(*x);
            *x = bf16_to_f32(b);
            data.extend(b.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&J::Object(header)).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(dst, out).unwrap();
}

#[test]
fn gemma4_seqcls_cpu() {
    gemma4_seqcls(Device::Cpu);
}

/// Run through the central scheduler (Metal).
#[test]
fn gemma4_seqcls_metal() {
    if crate::hardware::available_devices().contains(&Device::Metal) {
        gemma4_seqcls(Device::Metal);
    }
}

/// gemma4's `head: 'score'` (Gemma4ForSequenceClassification: final norm, the head's own
/// `norm.weight`, `score`) and `head: 'hidden'` (with that norm) against `Gemma4Seq`'s host
/// forward, on the gemma4 golden fixtures (`cargo test --test golden gemma4_*`) plus a head.
fn gemma4_seqcls(dev: Device) {
    let _g = crate::hardware::cpu_compute::allow();
    let mut bad = Vec::new();
    // Not `gemma_gemma4_moe`: the old host forward (`TextModel::last_hidden`) disagrees with the
    // old decode graph there (cos 0.73), which the spec matches exactly (parity_gemma4_text).
    for case in ["gemma_gemma4_decode", "gemma_gemma4_ple"] {
        let src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/golden-fixtures")
            .join(case);
        if !src.exists() {
            eprintln!("skip {case}: run the golden test first");
            continue;
        }
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for f in ["config.json", "tokenizer.json", "processor_config.json"] {
            std::fs::copy(src.join(f), dir.join(f)).unwrap();
        }
        let cfg: J =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let h = cfg["text_config"]["hidden_size"].as_u64().unwrap() as usize;
        let mut w = Weights::new(0x5c0e);
        w.norm("norm.weight", h, 1.0).w("score.weight", &[3, h]);
        append_bf16(
            &src.join("model.safetensors"),
            &dir.join("model.safetensors"),
            &mut w.tensors,
        );
        let score = w.tensors[1].2.clone();
        let host =
            crate::model::gemma4::seqcls::Gemma4Seq::load(&super::testing::artifacts(dir), &cfg)
                .unwrap();
        let dot = |m: &[f32], h: &[f32]| -> Vec<f32> {
            m.chunks(h.len())
                .map(|r| r.iter().zip(h).map(|(a, b)| a * b).sum())
                .collect()
        };
        let ids: [u32; 7] = [2, 10, 37, 54, 8, 21, 5];
        let mut r = Runner::load(dir, Some("gemma4"), dev, 64).unwrap_or_else(|e| panic!("{e}"));
        let ve = |n: usize| Tensor::U32((1..=n as u32).collect());
        let want = host.last_hidden(&ids).unwrap();
        let hidden = r.lm(&ids, 0, "hidden", "hidden", vec![ve(ids.len())]);
        close(
            &mut bad,
            &format!("{case} {dev:?} hidden"),
            &hidden[hidden.len() - h..],
            &want,
        );
        let got = r.lm(
            &ids,
            0,
            "score",
            "logits",
            vec![Tensor::U32(vec![6]), ve(ids.len())],
        );
        close(
            &mut bad,
            &format!("{case} {dev:?} score"),
            &got,
            &dot(&score, &want),
        );
        let got = r.lm(
            &ids,
            0,
            "score",
            "logits",
            vec![Tensor::U32(vec![3]), ve(ids.len())],
        );
        close(
            &mut bad,
            &format!("{case} {dev:?} score row 3"),
            &got,
            &dot(&score, &host.last_hidden(&ids[..4]).unwrap()),
        );
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
