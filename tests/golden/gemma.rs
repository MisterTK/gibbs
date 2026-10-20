use crate::util::*;
use gibbs_engine::graph::Tensor;
use gibbs_engine::hardware::{Device, PlacementMode};
use gibbs_engine::model::native::{load_native, MediaInput, NativeLm};
use gibbs_engine::model::weight_format::{set_reencode_override, ReencodeConfig, ReencodePolicy};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const V: usize = 128;
const IMG: u32 = 120;
const AUD: u32 = 121;
const BOI: u32 = 122;
const EOI: u32 = 123;
const BOA: u32 = 124;
const EOA: u32 = 125;

fn argmax(l: &[f32]) -> u32 {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

/// Every `k`-th value so at most `max` remain.
fn strided(v: &[f32], max: usize) -> Vec<f32> {
    v.iter().step_by(v.len().div_ceil(max).max(1)).copied().collect()
}

/// A WordLevel tokenizer of `n` ids whose `names` ids are added special tokens.
fn tokenizer(dir: &Path, n: usize, names: &[(u32, &str)]) {
    write_tokenizer(dir, n);
    let p = dir.join("tokenizer.json");
    let mut t: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    for &(id, name) in names {
        let vocab = t["model"]["vocab"].as_object_mut().unwrap();
        vocab.remove(&format!("t{id}"));
        vocab.insert(name.into(), json!(id));
        t["added_tokens"].as_array_mut().unwrap().push(json!({"id": id, "content": name,
            "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
    }
    write_json(&p, &t);
}

fn media_tokens(dir: &Path) {
    tokenizer(
        dir,
        V,
        &[
            (IMG, "<|image|>"),
            (AUD, "<|audio|>"),
            (BOI, "<|image>"),
            (EOI, "<image|>"),
            (BOA, "<|audio>"),
            (EOA, "<audio|>"),
        ],
    );
}

/// `w x h` RGB PNG with a deterministic gradient pattern.
fn png(w: usize, h: usize) -> Vec<u8> {
    let mut img = gibbs_engine::model::image::ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.set_pixel(
                x,
                y,
                ((x * 7 + y * 3) % 256) as u8,
                ((x * y * 5) % 256) as u8,
                (200 - (x * 3 + y * 2) % 200) as u8,
            );
        }
    }
    img.to_png()
}

/// 0.5 s of 16 kHz mono 16-bit PCM: 440 Hz + 1750 Hz tones.
fn wav() -> Vec<u8> {
    let pcm: Vec<u8> = (0..8000)
        .flat_map(|i| {
            let x = i as f32 / 16000.0;
            let v = 0.4 * (std::f32::consts::TAU * 440.0 * x).sin()
                + 0.2 * (std::f32::consts::TAU * 1750.0 * x).sin();
            ((v * 32767.0) as i16).to_le_bytes()
        })
        .collect();
    let mut b = b"RIFF".to_vec();
    b.extend((36 + pcm.len() as u32).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend([1, 0, 1, 0]);
    b.extend(16000u32.to_le_bytes());
    b.extend(32000u32.to_le_bytes());
    b.extend([2, 0, 16, 0]);
    b.extend(b"data");
    b.extend((pcm.len() as u32).to_le_bytes());
    b.extend(pcm);
    b
}

/// Bit-exact weights on every device: the golden must not depend on the user's re-encode settings.
fn load(dir: &Path, dev: Device) -> Arc<dyn NativeLm> {
    set_reencode_override(Some(ReencodeConfig {
        policy: ReencodePolicy::BitExact,
        cache_dir: None,
        ..Default::default()
    }));
    let art = artifacts(dir, "golden/gemma4", "model.safetensors");
    let m = load_native(&art, PlacementMode::Pin(dev))
        .expect("gemma4 is a native architecture")
        .unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    // No silent host fallback: a Metal pin must run the Metal program.
    assert_eq!(m.device(), dev, "{}: {}", dir.display(), m.backend_label());
    m
}

#[derive(Clone, Copy, Default)]
struct G4 {
    ple: bool,
    moe: bool,
    bidir: bool,
}

/// `Gemma4ForConditionalGeneration` checkpoint (E2B layout at toy size): text decoder (sliding +
/// full layers with per-type head dims, proportional rope, KV sharing on the last 2 layers with
/// double-wide MLPs; or 2 MoE layers), the `gemma4_vision` tower and the `gemma4_audio` conformer.
fn gemma4_fixture(case: &str, o: G4) -> PathBuf {
    let dir = fixture_dir(case);
    let (h, nh, kvh, hd, ghd, inter, pd) = (64, 2, 1, 16, 32, 96, 8);
    let (types, shared): (&[&str], usize) = if o.moe {
        (&["sliding_attention", "full_attention"], 0)
    } else {
        (
            &["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            2,
        )
    };
    let (nl, (experts, top_k, ff)) = (types.len(), (8, 2, 32));
    let mut w = Weights::new(0x6e6d_6134 + o.ple as u64 * 2 + o.moe as u64 * 3 + o.bidir as u64 * 5);
    let p = "model.language_model";
    w.w(format!("{p}.embed_tokens.weight"), &[V, h]);
    w.norm(format!("{p}.norm.weight"), h, 1.0);
    if o.ple {
        w.s(format!("{p}.embed_tokens_per_layer.weight"), &[V, nl * pd], 0.5);
        w.w(format!("{p}.per_layer_model_projection.weight"), &[nl * pd, h]);
        w.norm(format!("{p}.per_layer_projection_norm.weight"), pd, 1.0);
    }
    for (i, t) in types.iter().enumerate() {
        let l = format!("{p}.layers.{i}");
        let d = if *t == "sliding_attention" { hd } else { ghd };
        let mlp = if i >= nl - shared { 2 * inter } else { inter };
        w.w(format!("{l}.self_attn.q_proj.weight"), &[nh * d, h]);
        if i < nl - shared {
            w.w(format!("{l}.self_attn.k_proj.weight"), &[kvh * d, h]);
            w.w(format!("{l}.self_attn.v_proj.weight"), &[kvh * d, h]);
            w.norm(format!("{l}.self_attn.k_norm.weight"), d, 1.0);
        }
        w.w(format!("{l}.self_attn.o_proj.weight"), &[h, nh * d]);
        w.norm(format!("{l}.self_attn.q_norm.weight"), d, 1.0);
        w.w(format!("{l}.mlp.gate_proj.weight"), &[mlp, h]);
        w.w(format!("{l}.mlp.up_proj.weight"), &[mlp, h]);
        w.w(format!("{l}.mlp.down_proj.weight"), &[h, mlp]);
        for n in ["input", "post_attention", "pre_feedforward", "post_feedforward"] {
            w.norm(format!("{l}.{n}_layernorm.weight"), h, 1.0);
        }
        w.norm(format!("{l}.layer_scalar"), 1, 1.0);
        if o.ple {
            w.w(format!("{l}.per_layer_input_gate.weight"), &[pd, h]);
            w.w(format!("{l}.per_layer_projection.weight"), &[h, pd]);
            w.norm(format!("{l}.post_per_layer_input_norm.weight"), h, 1.0);
        }
        if o.moe {
            w.w(format!("{l}.router.proj.weight"), &[experts, h]);
            w.norm(format!("{l}.router.scale"), h, 1.0);
            w.norm(format!("{l}.router.per_expert_scale"), experts, 1.0);
            for n in ["pre_feedforward_layernorm_2", "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"] {
                w.norm(format!("{l}.{n}.weight"), h, 1.0);
            }
            w.w(format!("{l}.mlp.experts.gate_up_proj"), &[experts, 2 * ff, h]);
            w.w(format!("{l}.mlp.experts.down_proj"), &[experts, h, ff]);
        }
    }
    // Vision tower: hidden 16, 2 layers, 4 px patches, 2x2 pooling.
    let (vd, vi, vhd, pos) = (16, 32, 8, 8);
    let vp = "model.vision_tower";
    w.w(format!("{vp}.patch_embedder.input_proj.weight"), &[vd, 4 * 4 * 3]);
    w.s(format!("{vp}.patch_embedder.position_embedding_table"), &[2, pos, vd], 0.3);
    w.w("model.embed_vision.embedding_projection.weight", &[h, vd]);
    for i in 0..2 {
        let l = format!("{vp}.encoder.layers.{i}");
        for (n, r, c) in [
            ("self_attn.q_proj", vd, vd),
            ("self_attn.k_proj", vd, vd),
            ("self_attn.v_proj", vd, vd),
            ("self_attn.o_proj", vd, vd),
            ("mlp.gate_proj", vi, vd),
            ("mlp.up_proj", vi, vd),
            ("mlp.down_proj", vd, vi),
        ] {
            w.w(format!("{l}.{n}.linear.weight"), &[r, c]);
        }
        for n in ["q_proj", "o_proj"] {
            for (b, v) in [("input_min", -1.5), ("input_max", 1.5), ("output_min", -2.5), ("output_max", 2.5)] {
                w.put(format!("{l}.self_attn.{n}.{b}"), &[1], vec![v]);
            }
        }
        w.norm(format!("{l}.self_attn.q_norm.weight"), vhd, 1.0);
        w.norm(format!("{l}.self_attn.k_norm.weight"), vhd, 1.0);
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            w.norm(format!("{l}.{n}.weight"), vd, 1.0);
        }
    }
    // Audio conformer: hidden 16, 2 layers, 16 mel bins.
    let (ad, ap) = (16, "model.audio_tower");
    let sc = format!("{ap}.subsample_conv_projection");
    w.s(format!("{sc}.layer0.conv.weight"), &[4, 1, 3, 3], 0.3);
    w.norm(format!("{sc}.layer0.norm.weight"), 4, 1.0);
    w.s(format!("{sc}.layer1.conv.weight"), &[8, 4, 3, 3], 0.2);
    w.norm(format!("{sc}.layer1.norm.weight"), 8, 1.0);
    w.w(format!("{sc}.input_proj_linear.weight"), &[ad, 32]);
    w.w(format!("{ap}.output_proj.weight"), &[12, ad]);
    w.s(format!("{ap}.output_proj.bias"), &[12], 0.1);
    w.w("model.embed_audio.embedding_projection.weight", &[h, 12]);
    for i in 0..2 {
        let l = format!("{ap}.layers.{i}");
        for f in ["feed_forward1", "feed_forward2"] {
            w.norm(format!("{l}.{f}.pre_layer_norm.weight"), ad, 1.0);
            w.norm(format!("{l}.{f}.post_layer_norm.weight"), ad, 1.0);
            w.w(format!("{l}.{f}.ffw_layer_1.linear.weight"), &[32, ad]);
            w.w(format!("{l}.{f}.ffw_layer_2.linear.weight"), &[ad, 32]);
        }
        for n in ["norm_pre_attn", "norm_post_attn", "norm_out", "lconv1d.pre_layer_norm", "lconv1d.conv_norm"] {
            w.norm(format!("{l}.{n}.weight"), ad, 1.0);
        }
        for n in ["q_proj", "k_proj", "v_proj", "post"] {
            w.w(format!("{l}.self_attn.{n}.linear.weight"), &[ad, ad]);
        }
        w.w(format!("{l}.self_attn.relative_k_proj.weight"), &[ad, ad]);
        w.s(format!("{l}.self_attn.per_dim_scale"), &[ad / 2], 0.3);
        w.w(format!("{l}.lconv1d.linear_start.linear.weight"), &[2 * ad, ad]);
        w.s(format!("{l}.lconv1d.depthwise_conv1d.weight"), &[ad, 1, 5], 0.4);
        w.w(format!("{l}.lconv1d.linear_end.linear.weight"), &[ad, ad]);
        for (b, v) in [("input_min", -2.0), ("input_max", 2.0), ("output_min", -3.0), ("output_max", 3.0)] {
            w.put(format!("{l}.feed_forward1.ffw_layer_1.{b}"), &[1], vec![v]);
        }
    }
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);

    let mut text = json!({
        "model_type": "gemma4_text", "hidden_size": h, "num_hidden_layers": nl,
        "num_attention_heads": nh, "num_key_value_heads": kvh, "head_dim": hd,
        "global_head_dim": ghd, "num_global_key_value_heads": kvh, "intermediate_size": inter,
        "num_kv_shared_layers": shared, "hidden_size_per_layer_input": if o.ple { pd } else { 0 },
        "layer_types": types, "sliding_window": 4, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0, "pad_token_id": 0,
        "vocab_size": V, "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6},
        },
    });
    if o.bidir {
        text["use_bidirectional_attention"] = json!("vision");
    }
    if o.moe {
        text["enable_moe_block"] = json!(true);
        text["num_experts"] = json!(experts);
        text["top_k_experts"] = json!(top_k);
        text["moe_intermediate_size"] = json!(ff);
    }
    write_json(
        &dir.join("config.json"),
        &json!({
            "model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": text,
            "vision_config": {
                "model_type": "gemma4_vision", "hidden_size": vd, "intermediate_size": vi,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": vhd,
                "num_hidden_layers": 2, "patch_size": 4, "pooling_kernel_size": 2,
                "position_embedding_size": pos, "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 100.0},
            },
            "audio_config": {
                "model_type": "gemma4_audio", "hidden_size": ad, "num_attention_heads": 2,
                "num_hidden_layers": 2, "attention_context_left": 5, "attention_context_right": 0,
                "conv_kernel_size": 5, "rms_norm_eps": 1e-6, "attention_logit_cap": 50.0,
                "residual_weight": 0.5, "hidden_act": "silu", "output_proj_dims": 12,
            },
            "image_token_id": IMG, "audio_token_id": AUD, "boi_token_id": BOI,
            "eoi_token_id": EOI, "boa_token_id": BOA, "eoa_token_id": EOA,
        }),
    );
    write_json(
        &dir.join("processor_config.json"),
        &json!({
            "image_processor": {"max_soft_tokens": 9, "patch_size": 4, "pooling_kernel_size": 2},
            "feature_extractor": {
                "sampling_rate": 16000, "frame_length": 160, "fft_length": 256, "hop_length": 80,
                "feature_size": 16, "mel_floor": 1e-3, "min_frequency": 0.0, "max_frequency": 8000.0,
            },
        }),
    );
    media_tokens(&dir);
    dir
}

/// `Gemma4UnifiedForConditionalGeneration` (gemma-4-12B layout at toy size): sliding + full layer
/// with `attention_k_eq_v`, bidirectional image blocks, the encoder-free 12 px patch embedder
/// (4 px patches, 3x3 merge) and the raw 640-sample audio frame embedder.
fn unified_fixture(case: &str) -> PathBuf {
    let dir = fixture_dir(case);
    let (h, inter, pdim, d, pos, frame) = (32, 64, 12 * 12 * 3, 32, 8, 640);
    let mut w = Weights::new(0x6e6d_6175);
    let p = "model.language_model";
    w.w(format!("{p}.embed_tokens.weight"), &[V, h]);
    w.norm(format!("{p}.norm.weight"), h, 1.0);
    for (i, (hd, kv_eq)) in [(16, false), (32, true)].into_iter().enumerate() {
        let l = format!("{p}.layers.{i}");
        w.w(format!("{l}.self_attn.q_proj.weight"), &[2 * hd, h]);
        w.w(format!("{l}.self_attn.k_proj.weight"), &[hd, h]);
        if !kv_eq {
            w.w(format!("{l}.self_attn.v_proj.weight"), &[hd, h]);
        }
        w.w(format!("{l}.self_attn.o_proj.weight"), &[h, 2 * hd]);
        w.norm(format!("{l}.self_attn.q_norm.weight"), hd, 1.0);
        w.norm(format!("{l}.self_attn.k_norm.weight"), hd, 1.0);
        w.w(format!("{l}.mlp.gate_proj.weight"), &[inter, h]);
        w.w(format!("{l}.mlp.up_proj.weight"), &[inter, h]);
        w.w(format!("{l}.mlp.down_proj.weight"), &[h, inter]);
        for n in ["input", "post_attention", "pre_feedforward", "post_feedforward"] {
            w.norm(format!("{l}.{n}_layernorm.weight"), h, 1.0);
        }
        w.norm(format!("{l}.layer_scalar"), 1, 1.0);
    }
    let v = "model.vision_embedder";
    w.w(format!("{v}.patch_dense.weight"), &[d, pdim]);
    w.s(format!("{v}.patch_dense.bias"), &[d], 0.1);
    for (n, dim) in [("patch_ln1", pdim), ("patch_ln2", d), ("pos_norm", d)] {
        w.norm(format!("{v}.{n}.weight"), dim, 1.0);
        w.s(format!("{v}.{n}.bias"), &[dim], 0.1);
    }
    w.s(format!("{v}.pos_embedding"), &[pos, 2, d], 0.5);
    w.w("model.embed_vision.embedding_projection.weight", &[h, d]);
    w.w("model.embed_audio.embedding_projection.weight", &[h, frame]);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    write_json(
        &dir.join("config.json"),
        &json!({
            "model_type": "gemma4_unified", "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": {
                "model_type": "gemma4_unified_text", "hidden_size": h, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "num_global_key_value_heads": 1, "attention_k_eq_v": true,
                "use_bidirectional_attention": "vision", "hidden_size_per_layer_input": 0,
                "intermediate_size": inter, "vocab_size": V, "pad_token_id": 0,
                "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 4,
                "max_position_embeddings": 4096, "final_logit_softcapping": 30.0,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6},
                },
            },
            "vision_config": {"patch_size": 4, "pooling_kernel_size": 3, "model_patch_size": 12,
                              "rms_norm_eps": 1e-6},
            "audio_config": {"audio_samples_per_token": frame, "rms_norm_eps": 1e-6},
            "image_token_id": IMG, "audio_token_id": AUD, "boi_token_id": BOI,
            "eoi_token_id": EOI, "boa_token_id": BOA, "eoa_token_index": EOA,
        }),
    );
    write_json(
        &dir.join("processor_config.json"),
        &json!({
            "image_processor": {"max_soft_tokens": 9},
            "feature_extractor": {"sampling_rate": 16000, "audio_samples_per_token": frame},
            "audio_seq_length": 750,
        }),
    );
    media_tokens(&dir);
    dir
}

/// Logits after a one-shot prefill of `prompt`, after 3 decode steps (argmax fed back), then 8 more
/// greedy ids. `fwd(tokens, pos)` returns the logits after the last token.
fn decode_run(mut fwd: impl FnMut(&[u32], usize) -> Vec<f32>, prompt: &[u32]) -> Out {
    let mut out = Out::default();
    let mut l = fwd(prompt, 0);
    out.f("prefill_last", &l);
    let mut pos = prompt.len();
    let mut ids = Vec::new();
    for i in 0..11 {
        let t = argmax(&l);
        ids.push(t);
        l = fwd(&[t], pos);
        pos += 1;
        if i < 3 {
            out.f(format!("decode_{i}"), &l);
        }
    }
    out.ids("greedy", &ids);
    out
}

const PROMPT: [u32; 20] = [2, 10, 37, 54, 8, 91, 23, 66, 12, 105, 41, 7, 88, 19, 73, 30, 99, 5, 62, 44];

fn text_case(case: &str, o: G4) {
    let dir = gemma4_fixture(case, o);
    golden(case, TOL, |dev| {
        let m = load(&dir, dev);
        let (ids, mut st) = m.prepare(PROMPT.to_vec(), &[]).unwrap();
        decode_run(|t, p| m.forward(t, p, &mut st).unwrap(), &ids)
    });
}

/// Media prompt through `prepare`: the expanded ids, the one-shot prefill logits, every position's
/// logits from a token-by-token pass (strided to 4096), then 3 decode steps.
fn media_case(case: &str, dir: &Path, prompt: &[u32], media: MediaInput) {
    golden(case, TOL, |dev| {
        let m = load(dir, dev);
        let media = [media.clone()];
        let (ids, mut st) = m.prepare(prompt.to_vec(), &media).unwrap();
        let mut out = decode_run(|t, p| m.forward(t, p, &mut st).unwrap(), &ids);
        out.ids("expanded", &ids);
        let (_, mut st) = m.prepare(prompt.to_vec(), &media).unwrap();
        let steps: Vec<f32> = ids
            .iter()
            .enumerate()
            .flat_map(|(p, &t)| m.forward(&[t], p, &mut st).unwrap())
            .collect();
        out.f("per_position", &strided(&steps, 4096));
        out
    });
}

#[test]
fn gemma4_decode() {
    text_case("gemma_gemma4_decode", G4::default());
}

#[test]
fn gemma4_ple() {
    text_case("gemma_gemma4_ple", G4 { ple: true, ..G4::default() });
}

/// The MoE block has no host forward: the CPU reference is the decode graph (built from the same
/// checkpoint) on the CPU graph executor, stepped token by token; Metal is the `load_native` model.
#[test]
fn gemma4_moe() {
    use gibbs_engine::model::gemma4::text::TextModel;
    let case = "gemma_gemma4_moe";
    let dir = gemma4_fixture(case, G4 { moe: true, ..G4::default() });
    golden(case, TOL, |dev| {
        if dev != Device::Cpu {
            let m = load(&dir, dev);
            let (ids, mut st) = m.prepare(PROMPT.to_vec(), &[]).unwrap();
            return decode_run(|t, p| m.forward(t, p, &mut st).unwrap(), &ids);
        }
        // runner: internal gemma4::text::TextModel::load + to_decode_graph + graph::cpu::run
        let cfg: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let w = gibbs_engine::model::weights::Weights::open(&dir.join("model.safetensors")).unwrap();
        let tm = TextModel::load(&w, &cfg["text_config"]).unwrap();
        let (g, gw) = tm.to_decode_graph().unwrap();
        let mut state = gibbs_engine::graph::StateStore::new();
        decode_run(
            |toks, pos| {
                let mut last = Vec::new();
                for (i, &t) in toks.iter().enumerate() {
                    let mut e = vec![0.0f32; tm.hidden];
                    tm.embed_row(t, &mut e);
                    let inputs = vec![Tensor::F32(e), Tensor::U32(vec![(pos + i) as u32])];
                    let r = gibbs_engine::graph::cpu::run(&g, &gw, &mut state, inputs).unwrap();
                    last = r[0].as_f32().unwrap().to_vec();
                }
                last
            },
            &PROMPT,
        )
    });
}

/// Bidirectional image block: one-shot prefill vs 5-token chunks (the first chunk ends inside the
/// block, so the pass runs on to the block end and the next chunk returns its cached logits).
#[test]
fn gemma4_image_prefill() {
    let case = "gemma_gemma4_image_prefill";
    let dir = gemma4_fixture(case, G4 { bidir: true, ..G4::default() });
    let prompt = [2, 10, 11, IMG, 12, 13, 14];
    let media = [MediaInput::Image(png(32, 24))];
    golden(case, TOL, |dev| {
        let m = load(&dir, dev);
        let (ids, mut st) = m.prepare(prompt.to_vec(), &media).unwrap();
        let mut out = decode_run(|t, p| m.forward(t, p, &mut st).unwrap(), &ids);
        out.ids("expanded", &ids);
        let (_, mut st) = m.prepare(prompt.to_vec(), &media).unwrap();
        let chunks: Vec<f32> = ids
            .chunks(5)
            .enumerate()
            .flat_map(|(i, c)| m.forward(c, i * 5, &mut st).unwrap())
            .collect();
        out.f("prefill_chunks5", &chunks);
        out
    });
}

#[test]
fn gemma4_vision() {
    let case = "gemma_gemma4_vision";
    let dir = gemma4_fixture(case, G4::default());
    media_case(case, &dir, &[2, 10, 11, IMG, 12, 13], MediaInput::Image(png(32, 24)));
}

#[test]
fn gemma4_audio() {
    let case = "gemma_gemma4_audio";
    let dir = gemma4_fixture(case, G4::default());
    media_case(case, &dir, &[2, 10, AUD, 12, 13], MediaInput::Audio(wav()));
}

#[test]
fn gemma4_unified_vision() {
    let case = "gemma_gemma4_unified_vision";
    let dir = unified_fixture(case);
    media_case(case, &dir, &[2, 10, 11, IMG, 12, 13], MediaInput::Image(png(48, 36)));
}

#[test]
fn gemma4_unified_audio() {
    let case = "gemma_gemma4_unified_audio";
    let dir = unified_fixture(case);
    media_case(case, &dir, &[2, 10, AUD, 12, 13], MediaInput::Audio(wav()));
}

/// `Gemma3TextForSequenceClassification` hub snapshot: 3 layers (two sliding, window 3, one global
/// with yarn rope), GQA 4/2, `query_pre_attn_scalar` 24, attention softcap, 3 softmax labels.
#[test]
fn gemma3_seqcls() {
    use gibbs_engine::model::classifier::SequenceClassifier;
    use gibbs_engine::model::hub::HuggingFaceHub;
    let case = "gemma_gemma3_seqcls";
    let root = fixture_dir(case);
    // A repo id that cannot exist on the Hub: every file the loader asks for is in the snapshot.
    let dir = root.join("models--gibbs-golden--gemma3-seqcls-tiny/snapshots/main");
    let (h, v, nh, kvh, hd, inter) = (64, 96, 4, 2, 16, 96);
    let mut w = Weights::new(0x6e6d_6133);
    w.w("model.embed_tokens.weight", &[v, h]);
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
        for n in ["input", "post_attention", "pre_feedforward", "post_feedforward"] {
            w.norm(format!("{p}.{n}_layernorm.weight"), h, 0.0);
        }
    }
    w.norm("model.norm.weight", h, 0.0);
    w.w("score.weight", &[3, h]);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    write_json(
        &dir.join("config.json"),
        &json!({
            "architectures": ["Gemma3TextForSequenceClassification"], "model_type": "gemma3_text",
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 3,
            "num_attention_heads": nh, "num_key_value_heads": kvh, "head_dim": hd,
            "vocab_size": v, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
            "sliding_window": 3, "query_pre_attn_scalar": 24, "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": null, "hidden_activation": "gelu_pytorch_tanh",
            "rope_theta": 1e6, "rope_local_base_freq": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16},
            "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1,
            "id2label": {"0": "neg", "1": "neu", "2": "pos"},
            "label2id": {"neg": 0, "neu": 1, "pos": 2},
        }),
    );
    write_tokenizer(&dir, v);
    let texts = [
        "t5 t17 t9",
        "t2 t11 t23 t40 t8 t77 t31 t64 t19 t50 t6 t90 t42",
    ];
    golden(case, TOL, |dev| {
        let hub = HuggingFaceHub::with_cache_dir(root.clone());
        let c = SequenceClassifier::load(&hub, "gibbs-golden/gemma3-seqcls-tiny", dev)
            .unwrap_or_else(|e| panic!("gemma3 seq-cls load: {e}"));
        assert_eq!(c.device(), dev, "gemma3 seq-cls body fell back");
        let mut out = Out::default();
        for (i, t) in texts.iter().enumerate() {
            let (scores, _) = c.classify(t, None).unwrap();
            let by_label: Vec<f32> = c
                .labels
                .iter()
                .map(|l| scores.iter().find(|s| &s.label == l).unwrap().score)
                .collect();
            out.f(format!("scores_{i}"), &by_label);
        }
        out
    });
}
