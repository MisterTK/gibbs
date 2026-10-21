//! Encoder-decoder goldens: T5 (`arch/families/t5.json5`) and the BART-family seq2seq models
//! (`bart.json5`, `marian.json5`, `m2m100.json5`: Marian, BART, M2M100, MBart-50), each loaded
//! from an on-disk HF checkpoint (config.json, generation_config.json, model.safetensors and the
//! family's own tokenizer files) through `runtime::encdec`. Recorded from the former `model::t5`
//! / `model::seq2seq` implementations.

use crate::util::*;
use gibbs_engine::hardware::{Device, PlacementMode};
use gibbs_engine::model::native::NativeLm;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

fn load(dir: &Path, dev: Device) -> Arc<dyn NativeLm> {
    let a = artifacts(dir, "golden/encdec", "model.safetensors");
    let reg = gibbs_engine::arch::registry().expect("registry");
    gibbs_engine::runtime::encdec::load(&a, reg, None, PlacementMode::Pin(dev))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// Greedy pick; when writing, refuses a near-tie (top-2 gap < 1e-2) that would make the exact id
/// check hinge on rounding.
fn argmax(v: &[f32]) -> u32 {
    let i = (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    if writing() {
        let second = (0..v.len()).filter(|&j| j != i).map(|j| v[j]).fold(f32::MIN, f32::max);
        assert!(v[i] - second > 1e-2, "near-tie: top-2 gap {}", v[i] - second);
    }
    i as u32
}

// ---------------------------------------------------------------------------
// T5 (shapes of `t5.rs` `tiny`: 2+2 layers, d_model 32, 4 heads of 8, d_ff 48, 8 buckets)
// ---------------------------------------------------------------------------

const T5_VOCAB: usize = 40;
/// Weight amplitude of `t5.rs` `tiny` (its `Rng::mat`).
const AMP: f32 = 0.4;

fn t5_checkpoint(
    dir: &Path,
    model_type: &str,
    gated: bool,
    tied: bool,
    per_block_rel: bool,
    dtype: StDtype,
) {
    let (v, d, inner, ff, buckets, heads) = (T5_VOCAB, 32, 32, 48, 8, 4);
    let mut w = Weights::new(0x75_0000 + gated as u64 * 2 + tied as u64 * 4 + per_block_rel as u64);
    for stack in ["encoder", "decoder"] {
        for i in 0..2 {
            let b = format!("{stack}.block.{i}.layer");
            let attn = |w: &mut Weights, l: usize, m: &str| {
                w.norm(format!("{b}.{l}.layer_norm.weight"), d, 1.0);
                for p in ["q", "k", "v"] {
                    w.s(format!("{b}.{l}.{m}.{p}.weight"), &[inner, d], AMP);
                }
                w.s(format!("{b}.{l}.{m}.o.weight"), &[d, inner], AMP);
            };
            attn(&mut w, 0, "SelfAttention");
            if i == 0 || per_block_rel {
                w.s(
                    format!("{b}.0.SelfAttention.relative_attention_bias.weight"),
                    &[buckets, heads],
                    1.0,
                );
            }
            let f = if stack == "decoder" {
                attn(&mut w, 1, "EncDecAttention");
                2
            } else {
                1
            };
            w.norm(format!("{b}.{f}.layer_norm.weight"), d, 1.0);
            if gated {
                w.s(format!("{b}.{f}.DenseReluDense.wi_0.weight"), &[ff, d], AMP);
                w.s(format!("{b}.{f}.DenseReluDense.wi_1.weight"), &[ff, d], AMP);
            } else {
                w.s(format!("{b}.{f}.DenseReluDense.wi.weight"), &[ff, d], AMP);
            }
            w.s(format!("{b}.{f}.DenseReluDense.wo.weight"), &[d, ff], AMP);
        }
        w.norm(format!("{stack}.final_layer_norm.weight"), d, 1.0);
    }
    // Larger than the layers' AMP: at AMP the tied head's top-2 logit gaps fall below 1e-3.
    w.s("shared.weight", &[v, d], 1.0);
    if !tied {
        w.s("lm_head.weight", &[v, d], AMP);
    }
    write_st_as(&dir.join("model.safetensors"), &w.tensors, dtype);
    let arch = match model_type {
        "umt5" => "UMT5ForConditionalGeneration",
        _ => "T5ForConditionalGeneration",
    };
    write_json(
        &dir.join("config.json"),
        &json!({
            "architectures": [arch], "model_type": model_type, "vocab_size": v, "d_model": d,
            "d_kv": 8, "num_heads": heads, "d_ff": ff, "num_layers": 2, "num_decoder_layers": 2,
            "relative_attention_num_buckets": buckets, "relative_attention_max_distance": 16,
            "layer_norm_epsilon": 1e-6,
            "feed_forward_proj": if gated { "gated-gelu" } else { "relu" },
            "tie_word_embeddings": tied, "decoder_start_token_id": 0, "pad_token_id": 0,
            "eos_token_id": 1, "n_positions": 80,
        }),
    );
    write_tokenizer(dir, v);
}

/// Encoder over 10 source ids (`prepare`, which appends EOS), then 8 greedy decoder steps from
/// the start token: logits of steps 0-3, the greedy ids, and the logits of a rewind that replays
/// decoder tokens 5..7 at position 5.
fn t5_case(case: &str, model_type: &str, gated: bool, tied: bool, per_block_rel: bool, dtype: StDtype) {
    let dir = fixture_dir(case);
    t5_checkpoint(&dir, model_type, gated, tied, per_block_rel, dtype);
    golden(case, TOL, |dev| {
        let m = load(&dir, dev);
        let src: Vec<u32> = (0..10).map(|i| 2 + (i * 7) % 30).collect();
        let (prompt, mut st) = m.prepare(src, &[]).unwrap();
        let mut out = Out::default();
        let mut toks = prompt.clone();
        for step in 0..8 {
            let l = m.forward(&toks[step..step + 1], step, &mut st).unwrap();
            assert_eq!(l.len(), T5_VOCAB);
            if step < 4 {
                out.f(format!("step{step}"), &l);
            }
            toks.push(argmax(&l));
        }
        let l = m.forward(&toks[5..7], 5, &mut st).unwrap();
        out.f("rewind5", &l).ids("greedy", &toks[1..]);
        out
    });
}

#[test]
fn t5_gated_tied() {
    t5_case("encdec_t5_gated_tied", "t5", true, true, false, StDtype::F32);
}

#[test]
fn t5_relu_untied() {
    t5_case("encdec_t5_relu_untied", "t5", false, false, false, StDtype::F32);
}

#[test]
fn umt5_gated_untied_per_block_bias() {
    t5_case("encdec_umt5_per_block_bias", "umt5", true, false, true, StDtype::F32);
}

/// bf16 safetensors stay mapped bf16 (`Lin::Bf16` on the CPU, bf16 Linear on Metal).
#[test]
fn t5_gated_tied_bf16() {
    t5_case("encdec_t5_gated_tied_bf16", "t5", true, true, false, StDtype::BF16);
}

// ---------------------------------------------------------------------------
// Seq2seq (shapes of `seq2seq.rs` `tiny`: 2+2 layers, d_model 32, 4 heads, FFN 64, vocab 40,
// 80 positions)
// ---------------------------------------------------------------------------

const V: usize = 40;
const D: usize = 32;
const MAX_POS: usize = 80;

#[derive(Clone, Copy)]
struct Fam {
    arch: &'static str,
    model_type: &'static str,
    act: &'static str,
    scale_embedding: bool,
    /// M2M100 / MBart: pre-LN with a final `layer_norm` per stack.
    pre_ln: bool,
    /// MBart / BART: learned positions and `layernorm_embedding`.
    learned_pos: bool,
    tied: bool,
    logits_bias: bool,
    /// bos, pad, eos, decoder_start.
    special: [u32; 4],
    forced_bos: Option<u32>,
}

fn s2s_weights(path: &Path, f: Fam, seed: u64) {
    let (d, ffn) = (D, 64);
    let mut w = Weights::new(seed);
    w.s("model.shared.weight", &[V, d], 0.3);
    for side in ["encoder", "decoder"] {
        let p = format!("model.{side}");
        if f.learned_pos {
            w.s(format!("{p}.embed_positions.weight"), &[MAX_POS + 2, d], 0.5);
            w.norm(format!("{p}.layernorm_embedding.weight"), d, 1.0);
            w.s(format!("{p}.layernorm_embedding.bias"), &[d], 0.1);
        }
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), d, 1.0);
            w.s(format!("{n}.bias"), &[d], 0.1);
        };
        let lin = |w: &mut Weights, n: String, rows: usize, cols: usize| {
            // 0.6 (vs `tiny`'s 0.3): at 0.3 greedy and beam paths repeat one token.
            w.s(format!("{n}.weight"), &[rows, cols], 0.6);
            w.s(format!("{n}.bias"), &[rows], 0.1);
        };
        for i in 0..2 {
            let l = format!("{p}.layers.{i}");
            let mut attns = vec!["self_attn"];
            if side == "decoder" {
                attns.push("encoder_attn");
            }
            for a in attns {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    lin(&mut w, format!("{l}.{a}.{proj}"), d, d);
                }
                ln(&mut w, format!("{l}.{a}_layer_norm"));
            }
            lin(&mut w, format!("{l}.fc1"), ffn, d);
            lin(&mut w, format!("{l}.fc2"), d, ffn);
            ln(&mut w, format!("{l}.final_layer_norm"));
        }
        if f.pre_ln {
            ln(&mut w, format!("{p}.layer_norm"));
        }
    }
    if !f.tied {
        w.s("lm_head.weight", &[V, d], 0.5);
    }
    if f.logits_bias {
        w.s("final_logits_bias", &[1, V], 0.2);
    }
    w.write(path);
}

fn s2s_config(dir: &Path, f: Fam, beams: usize, max_length: usize) {
    let [bos, pad, eos, start] = f.special;
    let mut cfg = json!({
        "architectures": [f.arch], "model_type": f.model_type, "vocab_size": V, "d_model": D,
        "max_position_embeddings": MAX_POS, "encoder_layers": 2, "decoder_layers": 2,
        "encoder_attention_heads": 4, "decoder_attention_heads": 4, "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64, "activation_function": f.act, "scale_embedding": f.scale_embedding,
        "normalize_before": f.pre_ln, "tie_word_embeddings": f.tied, "bos_token_id": bos,
        "pad_token_id": pad, "eos_token_id": eos, "decoder_start_token_id": start,
    });
    if let Some(b) = f.forced_bos {
        cfg["forced_bos_token_id"] = json!(b);
    }
    write_json(&dir.join("config.json"), &cfg);
    write_json(
        &dir.join("generation_config.json"),
        &json!({"num_beams": beams, "max_length": max_length}),
    );
}

/// SentencePiece pieces after the specials: `▁`, `t`, digits, `▁t`, then `▁t0`, `▁t1`, ...
fn spm_pieces(n: usize) -> Vec<String> {
    let mut p: Vec<String> = ["\u{2581}", "t"].map(String::from).to_vec();
    p.extend((0..10).map(|i| i.to_string()));
    p.push("\u{2581}t".into());
    p.extend((0..).map(|i| format!("\u{2581}t{i}")).take(n - p.len()));
    p
}

/// A minimal SentencePiece `ModelProto`: pieces `<unk>` (UNKNOWN), `<s>`, `</s>` (CONTROL), then
/// `normal` with score `-index`; Unigram or BPE trainer spec; dummy prefix and whitespace
/// cleanup on, no precompiled charsmap.
fn write_spm(path: &Path, normal: &[String], bpe: bool) {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push(v as u8 | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn msg(out: &mut Vec<u8>, field: u64, body: &[u8]) {
        varint(out, field << 3 | 2);
        varint(out, body.len() as u64);
        out.extend_from_slice(body);
    }
    let mut out = Vec::new();
    let specials = [("<unk>", 2u64), ("<s>", 3), ("</s>", 3)];
    let all = specials
        .iter()
        .map(|&(s, k)| (s.to_string(), k))
        .chain(normal.iter().map(|s| (s.clone(), 1)));
    for (i, (text, kind)) in all.enumerate() {
        let mut p = Vec::new();
        msg(&mut p, 1, text.as_bytes());
        varint(&mut p, 2 << 3 | 5);
        p.extend_from_slice(&(-(i as f32)).to_le_bytes());
        varint(&mut p, 3 << 3);
        varint(&mut p, kind);
        msg(&mut out, 1, &p);
    }
    let mut trainer = Vec::new();
    varint(&mut trainer, 3 << 3);
    varint(&mut trainer, if bpe { 2 } else { 1 });
    msg(&mut out, 2, &trainer);
    let mut norm = Vec::new();
    for field in [3u64, 4] {
        varint(&mut norm, field << 3);
        varint(&mut norm, 1);
    }
    msg(&mut out, 3, &norm);
    std::fs::write(path, out).unwrap();
}

fn vocab_json(dir: &Path, entries: impl IntoIterator<Item = (String, u32)>) {
    let m: serde_json::Map<String, Value> = entries.into_iter().map(|(k, v)| (k, json!(v))).collect();
    write_json(&dir.join("vocab.json"), &Value::Object(m));
}

/// Marian (opus-mt layout): Unigram `source.spm` + `vocab.json` (`</s>` 0, `<unk>` 1, pieces,
/// `<pad>` 39).
fn marian_tokenizer(dir: &Path) {
    let normal = spm_pieces(V - 3);
    write_spm(&dir.join("source.spm"), &normal, false);
    let pieces = normal.into_iter().zip(2..);
    vocab_json(
        dir,
        [("</s>".to_string(), 0), ("<unk>".to_string(), 1), ("<pad>".to_string(), 39)]
            .into_iter()
            .chain(pieces),
    );
}

const FAIRSEQ_SPECIALS: [&str; 4] = ["<s>", "<pad>", "</s>", "<unk>"];

/// M2M100: BPE `sentencepiece.bpe.model` + `vocab.json` (fairseq specials, pieces 4..37) and
/// the language codes `__en__` 38, `__fr__` 39 from special_tokens_map.json.
fn m2m100_tokenizer(dir: &Path) {
    let normal = spm_pieces(V - 6);
    write_spm(&dir.join("sentencepiece.bpe.model"), &normal, true);
    let specials = FAIRSEQ_SPECIALS.iter().map(|s| s.to_string()).zip(0..);
    vocab_json(dir, specials.chain(normal.into_iter().zip(4..)));
    write_json(
        &dir.join("special_tokens_map.json"),
        &json!({"additional_special_tokens": ["__en__", "__fr__"]}),
    );
    write_json(
        &dir.join("tokenizer_config.json"),
        &json!({"tokenizer_class": "M2M100Tokenizer", "src_lang": "en", "tgt_lang": "fr"}),
    );
}

/// MBart-50 slow layout: BPE `sentencepiece.bpe.model` only (fairseq specials, spm ids shifted
/// by one: 4..37), language codes `en_XX` 38, `fr_XX` 39.
fn mbart_tokenizer(dir: &Path) {
    write_spm(&dir.join("sentencepiece.bpe.model"), &spm_pieces(V - 6), true);
    write_json(
        &dir.join("special_tokens_map.json"),
        &json!({"additional_special_tokens": ["en_XX", "fr_XX"]}),
    );
    write_json(
        &dir.join("tokenizer_config.json"),
        &json!({"tokenizer_class": "MBart50Tokenizer", "src_lang": "en_XX", "tgt_lang": "fr_XX"}),
    );
}

/// BART: byte-level BPE `vocab.json` + `merges.txt` (fairseq specials, `Ġ`, `t`, digits, `Ġt`,
/// `Ġt0..9`, `t0..9`, `Ġt12`, `Ġt15`, `Ġt20`).
fn bart_tokenizer(dir: &Path) {
    let digits: Vec<String> = (0..10).map(|i| i.to_string()).collect();
    let mut merges = vec![("Ġ".to_string(), "t".to_string())];
    merges.extend(digits.iter().map(|g| ("Ġt".to_string(), g.clone())));
    merges.extend(digits.iter().map(|g| ("t".to_string(), g.clone())));
    merges.extend([("Ġt1", "2"), ("Ġt1", "5"), ("Ġt2", "0")].map(|(a, b)| (a.into(), b.into())));
    let mut tokens: Vec<String> = FAIRSEQ_SPECIALS.iter().map(|s| s.to_string()).collect();
    tokens.extend(["Ġ", "t"].map(String::from));
    tokens.extend(digits);
    tokens.extend(merges.iter().map(|(a, b)| format!("{a}{b}")));
    assert_eq!(tokens.len(), V);
    vocab_json(dir, tokens.into_iter().zip(0..));
    let text: String = std::iter::once("#version: 0.2".to_string())
        .chain(merges.iter().map(|(a, b)| format!("{a} {b}")))
        .map(|l| l + "\n")
        .collect();
    std::fs::write(dir.join("merges.txt"), text).unwrap();
}

/// Writes the greedy checkpoint `fixture_dir(case)` (num_beams 1, max_length 30) and the beam one
/// `fixture_dir(case_beam)` (num_beams 3, max_length 13): same weights, tokenizer and config.
fn s2s_fixture(case: &str, f: Fam, seed: u64, tokenizer: fn(&Path)) -> [std::path::PathBuf; 2] {
    let dirs = [fixture_dir(case), fixture_dir(&format!("{case}_beam"))];
    for (dir, (beams, max_len)) in dirs.iter().zip([(1, 30), (3, 13)]) {
        s2s_weights(&dir.join("model.safetensors"), f, seed);
        s2s_config(dir, f, beams, max_len);
        tokenizer(dir);
    }
    dirs
}

/// The source text (bytes, as the seq2seq byte-passthrough tokenizer hands it to `prepare`).
/// Greedy model: per-token log-probs (after logits processors, as `forward` returns them) of the
/// first 3 output steps and 8 greedy ids. Beam model (`prepare` runs the 3-beam search, `forward`
/// replays it): the replayed ids, and their per-token log-probs teacher-forced through the
/// greedy model (summed, the hypothesis' beam score).
fn s2s_case(case: &str, f: Fam, seed: u64, tokenizer: fn(&Path), text: &str) {
    let [dir, beam_dir] = s2s_fixture(case, f, seed, tokenizer);
    let eos = f.special[2];
    let src: Vec<u32> = text.bytes().map(u32::from).collect();
    // The first output position's log-probs sit at the display aliases `V + id`.
    let region = |l: &[f32], first: bool| -> Vec<f32> {
        assert_eq!(l.len(), 2 * V);
        l[if first { V..2 * V } else { 0..V }].to_vec()
    };
    golden(case, TOL, |dev| {
        let mut out = Out::default();
        let m = load(&dir, dev);
        let (prompt, mut st) = m.prepare(src.clone(), &[]).unwrap();
        out.ids("prompt", &prompt);
        let mut l = m.forward(&prompt, 0, &mut st).unwrap();
        let mut greedy = Vec::new();
        for step in 0..8 {
            let lp = region(&l, step == 0);
            if step < 3 {
                out.f(format!("step{step}"), &lp);
            }
            let t = argmax(&lp);
            greedy.push(t);
            l = m.forward(&[t], prompt.len() + step, &mut st).unwrap();
        }
        out.ids("greedy", &greedy);

        let mb = load(&beam_dir, dev);
        let (bprompt, mut bst) = mb.prepare(src.clone(), &[]).unwrap();
        assert_eq!(bprompt, prompt);
        let mut l = mb.forward(&prompt, 0, &mut bst).unwrap();
        let mut beam = Vec::new();
        loop {
            let t = argmax(&l) % V as u32;
            beam.push(t);
            // A hypothesis cut at max_length 13 has no EOS; replay past it would invent one.
            if t == eos || prompt.len() + beam.len() >= 13 {
                break;
            }
            l = mb.forward(&[t], prompt.len() + beam.len() - 1, &mut bst).unwrap();
        }
        out.ids("beam", &beam);

        let (_, mut st) = m.prepare(src.clone(), &[]).unwrap();
        let mut l = m.forward(&prompt, 0, &mut st).unwrap();
        let mut lps = Vec::new();
        for (i, &t) in beam.iter().enumerate() {
            lps.push(region(&l, i == 0)[t as usize]);
            if i + 1 < beam.len() {
                l = m.forward(&[t], prompt.len() + i, &mut st).unwrap();
            }
        }
        out.f("beam_logprobs", &lps);
        out
    });
}

const TEXT: &str = "t5 t9 t12 t7 t20 t3 t15 t11";

#[test]
fn marian() {
    let f = Fam {
        arch: "MarianMTModel",
        model_type: "marian",
        act: "relu",
        scale_embedding: true,
        pre_ln: false,
        learned_pos: false,
        tied: true,
        logits_bias: true,
        special: [0, 39, 0, 39],
        forced_bos: None,
    };
    s2s_case("encdec_marian", f, 0x5e_0001, marian_tokenizer, TEXT);
}

#[test]
fn bart() {
    let f = Fam {
        arch: "BartForConditionalGeneration",
        model_type: "bart",
        act: "gelu",
        scale_embedding: false,
        pre_ln: false,
        learned_pos: true,
        tied: true,
        logits_bias: true,
        special: [0, 1, 2, 2],
        forced_bos: Some(0),
    };
    s2s_case("encdec_bart", f, 0x5e_0002, bart_tokenizer, TEXT);
}

#[test]
fn m2m100() {
    let f = Fam {
        arch: "M2M100ForConditionalGeneration",
        model_type: "m2m_100",
        act: "silu",
        scale_embedding: true,
        pre_ln: true,
        learned_pos: false,
        tied: false,
        logits_bias: false,
        special: [0, 1, 2, 2],
        forced_bos: None,
    };
    s2s_case("encdec_m2m100", f, 0x5e_0003, m2m100_tokenizer, &format!(">>fr<< {TEXT}"));
}

#[test]
fn mbart() {
    let f = Fam {
        arch: "MBartForConditionalGeneration",
        model_type: "mbart",
        act: "gelu",
        scale_embedding: false,
        pre_ln: true,
        learned_pos: true,
        tied: true,
        logits_bias: false,
        special: [0, 1, 2, 2],
        forced_bos: None,
    };
    s2s_case("encdec_mbart", f, 0x5e_0004, mbart_tokenizer, &format!(">>fr_XX<< {TEXT}"));
}
