//! Host text-encoder goldens: `Qwen3Encoder` (the diffusion pipelines' qwen3 / Qwen3-VL /
//! Qwen2.5-VL text towers) on tiny bf16 safetensors checkpoints, through `Qwen3Encoder::open`.
//! Outputs: the hidden states the pipelines read (`hidden_states`, `hidden_states_normed`,
//! `hidden_states_at` with a padded mask). Host-only, so CPU only.

use crate::util::*;
use gibbs_engine::hardware::Device;
use gibbs_engine::model::qwen3_5::Qwen3Encoder;
use serde_json::{json, Value};

const H: usize = 64;
const V: usize = 64;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 16;
const INTER: usize = 96;
const LAYERS: usize = 3;
const IDS: [u32; 7] = [5, 17, 33, 48, 2, 12, 60];

/// Dense decoder under prefix `p`: `qk_norm` (qwen3 / Qwen3-VL) or qkv biases (Qwen2.5-VL).
fn weights(seed: u64, p: &str, qk_norm: bool) -> Weights {
    let mut w = Weights::new(seed);
    w.w(format!("{p}.embed_tokens.weight"), &[V, H]);
    for l in 0..LAYERS {
        let lp = format!("{p}.layers.{l}");
        let a = format!("{lp}.self_attn");
        w.norm(format!("{lp}.input_layernorm.weight"), H, 1.0)
            .norm(format!("{lp}.post_attention_layernorm.weight"), H, 1.0)
            .w(format!("{a}.q_proj.weight"), &[HEADS * HD, H])
            .w(format!("{a}.k_proj.weight"), &[KV * HD, H])
            .w(format!("{a}.v_proj.weight"), &[KV * HD, H])
            .w(format!("{a}.o_proj.weight"), &[H, HEADS * HD]);
        if qk_norm {
            w.norm(format!("{a}.q_norm.weight"), HD, 1.0)
                .norm(format!("{a}.k_norm.weight"), HD, 1.0);
        } else {
            w.s(format!("{a}.q_proj.bias"), &[HEADS * HD], 0.2)
                .s(format!("{a}.k_proj.bias"), &[KV * HD], 0.2)
                .s(format!("{a}.v_proj.bias"), &[KV * HD], 0.2);
        }
        w.w(format!("{lp}.mlp.gate_proj.weight"), &[INTER, H])
            .w(format!("{lp}.mlp.up_proj.weight"), &[INTER, H])
            .w(format!("{lp}.mlp.down_proj.weight"), &[H, INTER]);
    }
    w.norm(format!("{p}.norm.weight"), H, 1.0);
    w
}

fn text_config(model_type: &str) -> Value {
    json!({
        "model_type": model_type, "hidden_size": H, "vocab_size": V, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD,
        "intermediate_size": INTER, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": true
    })
}

fn run(case: &str, cfg: &Value, w: &Weights) {
    let dir = fixture_dir(case);
    write_json(&dir.join("config.json"), cfg);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    let enc = Qwen3Encoder::open(&dir).unwrap_or_else(|e| panic!("{case}: {e}"));
    let mut out = Out::default();
    out.f("hidden_2", &enc.hidden_states(&IDS, 2).unwrap())
        .f("normed", &enc.hidden_states_normed(&IDS).unwrap());
    // Last two rows padding (mask 1 1 1 1 1 0 0); taps: embeddings, layer 1, every layer.
    for (k, h) in [0, 1, LAYERS].iter().zip(enc.hidden_states_at(&IDS, 5, &[0, 1, LAYERS]).unwrap()) {
        out.f(format!("masked_{k}"), &h);
    }
    check(case, Device::Cpu, &out, TOL);
}

#[test]
fn qwen3_encoder() {
    let mut cfg = text_config("qwen3");
    cfg["architectures"] = json!(["Qwen3ForCausalLM"]);
    run("text_enc_qwen3", &cfg, &weights(0x5157_454E_0E01, "model", true));
}

/// Nested `text_config` with interleaved M-RoPE sections, decoder under `model.language_model`.
#[test]
fn qwen3_vl_encoder() {
    let mut text = text_config("qwen3_vl_text");
    text["rope_scaling"] = json!({"rope_type": "default", "mrope_interleaved": true, "mrope_section": [4, 2, 2]});
    let cfg = json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"], "text_config": text});
    run("text_enc_qwen3_vl", &cfg, &weights(0x5157_454E_0E02, "model.language_model", true));
}

/// Qwen2.5-VL text tower: qkv biases, no q/k norm, `mrope` rope type (text positions: plain RoPE).
#[test]
fn qwen2_5_vl_encoder() {
    let mut cfg = text_config("qwen2_5_vl");
    cfg["architectures"] = json!(["Qwen2_5_VLForConditionalGeneration"]);
    cfg["rope_scaling"] = json!({"type": "mrope", "mrope_section": [4, 2, 2]});
    run("text_enc_qwen2_5_vl", &cfg, &weights(0x5157_454E_0E03, "model", false));
}

/// The spec text towers the diffusion pipelines reference (`qwen3`, `qwen3_vl_text`,
/// `qwen2_5_vl_text`) on the same fixtures, against the `Qwen3Encoder` goldens: `hidden_2` (tap
/// `layer.1`), `normed` (head `hidden`), and the masked taps after layers 1 and 3 on the 5 real rows.
/// (`masked_0`, the raw embeddings, has no spec tap; the llama-family Qwen2.5-VL tower has no
/// masked pass, and Qwen-Image 1 pads nothing.)
fn run_spec(case: &str, family: &str, cfg: &Value, w: &Weights, masked: bool) {
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::arch::testing::{compare, Runner};
    use gibbs_engine::graph::Tensor;
    if writing() {
        return; // the encoder runner owns the JSON
    }
    let dir = fixture_dir(&format!("{case}_spec"));
    write_json(&dir.join("config.json"), cfg);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    let g: Value = serde_json::from_slice(&std::fs::read(format!("{}/tests/golden/{case}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
    let want = |k: &str| -> Vec<f32> { g["f32"][k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
    for dev in devices() {
        let a = gibbs_engine::model::hub::ModelArtifacts {
            model_id: "test/text_enc".into(),
            local_dir: Some(dir.clone()),
            config_path: Some(dir.join("config.json")),
            weights_path: Some(dir.join("model.safetensors")),
            is_cached: true,
            ..Default::default()
        };
        let mut r = Runner::from_artifacts(&a, Some(family), dev, 64).unwrap_or_else(|e| panic!("{case}: {e}"));
        let taps = |l: &[usize]| Val::List(l.iter().map(|i| Val::Str(format!("layer.{i}"))).collect());
        let n = IDS.len();
        let ins = || vec![Tensor::U32(IDS.to_vec()), Tensor::U32((0..n as u32).collect())];
        let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("hidden".into())), ("taps", taps(&[1]))];
        let mut o = r.run("text", &vars, ins()).unwrap_or_else(|e| panic!("{case}: {e}"));
        let mut got = |k: &str| match o.remove(k) {
            Some(Tensor::F32(v)) => v,
            x => panic!("{case}: output {k}: {x:?}"),
        };
        let mut errs = Vec::new();
        for (k, v) in [("hidden_2", got("layer.1")), ("normed", got("hidden"))] {
            let (e, c) = compare(&v, &want(k));
            if !(e < TOL.rel && c > TOL.cos) {
                errs.push(format!("{k}: rel {e:.3e} cos {c:.6}"));
            }
        }
        if !masked {
            assert!(r.unconsumed().is_empty(), "{case}: unconsumed {:?}", r.unconsumed());
            assert!(errs.is_empty(), "{case} spec on {dev:?}:\n  {}", errs.join("\n  "));
            continue;
        }
        let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("hidden".into())), ("taps", taps(&[0, LAYERS - 1])),
            ("masked", Val::Bool(true))];
        let mut mi = ins();
        mi.push(Tensor::F32((0..n).map(|i| if i < 5 { 0.0 } else { f32::NEG_INFINITY }).collect()));
        let mut o = r.run("text", &vars, mi).unwrap_or_else(|e| panic!("{case}: {e}"));
        for (k, t) in [("masked_1", "layer.0"), ("masked_3", format!("layer.{}", LAYERS - 1).leak() as &str)] {
            let Some(Tensor::F32(v)) = o.remove(t) else { panic!("{case}: {t}") };
            let (e, c) = compare(&v[..5 * H], &want(k)[..5 * H]);
            if !(e < TOL.rel && c > TOL.cos) {
                errs.push(format!("{k}: rel {e:.3e} cos {c:.6}"));
            }
        }
        assert!(r.unconsumed().is_empty(), "{case}: unconsumed {:?}", r.unconsumed());
        assert!(errs.is_empty(), "{case} spec on {dev:?}:\n  {}", errs.join("\n  "));
    }
}

#[test]
fn qwen3_encoder_spec() {
    let mut cfg = text_config("qwen3");
    cfg["architectures"] = json!(["Qwen3ForCausalLM"]);
    run_spec("text_enc_qwen3", "qwen3", &cfg, &weights(0x5157_454E_0E01, "model", true), true);
}

#[test]
fn qwen3_vl_encoder_spec() {
    let mut text = text_config("qwen3_vl_text");
    text["rope_scaling"] = json!({"rope_type": "default", "mrope_interleaved": true, "mrope_section": [4, 2, 2]});
    let cfg = json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"], "text_config": text});
    run_spec("text_enc_qwen3_vl", "qwen3_vl_text", &cfg, &weights(0x5157_454E_0E02, "model.language_model", true), true);
}

#[test]
fn qwen2_5_vl_encoder_spec() {
    let mut cfg = text_config("qwen2_5_vl");
    cfg["architectures"] = json!(["Qwen2_5_VLForConditionalGeneration"]);
    cfg["rope_scaling"] = json!({"type": "mrope", "mrope_section": [4, 2, 2]});
    run_spec("text_enc_qwen2_5_vl", "qwen2_5_vl_text", &cfg, &weights(0x5157_454E_0E03, "model", false), false);
}
