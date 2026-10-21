//! Host text-encoder goldens: `Qwen3Encoder` (the diffusion pipelines' qwen3 / Qwen3-VL /
//! Qwen2.5-VL text towers) on tiny bf16 safetensors checkpoints, through `Qwen3Encoder::open`.
//! Outputs: the hidden states the pipelines read (`hidden_states`, `hidden_states_normed`,
//! `hidden_states_at` with a padded mask). Host-only, so CPU only.

use crate::util::*;
use gibbs_engine::hardware::Device;
use gibbs_engine::model::qwen3_5::Qwen3Encoder;
use serde_json::{json, Value};

pub(crate) const H: usize = 64;
pub(crate) const V: usize = 64;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 16;
const INTER: usize = 96;
const LAYERS: usize = 3;
const IDS: [u32; 7] = [5, 17, 33, 48, 2, 12, 60];

/// Dense decoder under prefix `p` with `layers` layers: `qk_norm` (qwen3 / Qwen3-VL) or qkv
/// biases (Qwen2.5-VL).
pub(crate) fn weights(seed: u64, p: &str, qk_norm: bool, layers: usize) -> Weights {
    let mut w = Weights::new(seed);
    w.w(format!("{p}.embed_tokens.weight"), &[V, H]);
    for l in 0..layers {
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

pub(crate) fn text_config(model_type: &str, layers: usize) -> Value {
    json!({
        "model_type": model_type, "hidden_size": H, "vocab_size": V, "num_hidden_layers": layers,
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
    let mut cfg = text_config("qwen3", LAYERS);
    cfg["architectures"] = json!(["Qwen3ForCausalLM"]);
    run("text_enc_qwen3", &cfg, &weights(0x5157_454E_0E01, "model", true, LAYERS));
}

/// Nested `text_config` with interleaved M-RoPE sections, decoder under `model.language_model`.
#[test]
fn qwen3_vl_encoder() {
    let mut text = text_config("qwen3_vl_text", LAYERS);
    text["rope_scaling"] = json!({"rope_type": "default", "mrope_interleaved": true, "mrope_section": [4, 2, 2]});
    let cfg = json!({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"], "text_config": text});
    run("text_enc_qwen3_vl", &cfg, &weights(0x5157_454E_0E02, "model.language_model", true, LAYERS));
}

/// Qwen2.5-VL text tower: qkv biases, no q/k norm, `mrope` rope type (text positions: plain RoPE).
#[test]
fn qwen2_5_vl_encoder() {
    let mut cfg = text_config("qwen2_5_vl", LAYERS);
    cfg["architectures"] = json!(["Qwen2_5_VLForConditionalGeneration"]);
    cfg["rope_scaling"] = json!({"type": "mrope", "mrope_section": [4, 2, 2]});
    run("text_enc_qwen2_5_vl", &cfg, &weights(0x5157_454E_0E03, "model", false, LAYERS));
}
