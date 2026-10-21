//! GraphLm decoders (llama, granite, LFM2, q/k-norm, wide GQA), BitNet and Nano.

use crate::util::*;
use gibbs_engine::hardware::{Device, HardwareCapability, QuantMode};
use gibbs_engine::model::hf_loader::load_causal_lm;
use gibbs_engine::model::nano::{NanoTransformerConfig, NanoTransformerModel};
use gibbs_engine::model::native::load_native;
use gibbs_engine::model::weight_format::{load_causal_lm_with, ReencodeConfig};
use gibbs_engine::scheduler::ExecutableModel;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const V: usize = 256;
const PROMPT: [u32; 6] = [5, 17, 42, 99, 133, 200];
/// Forced tokens decoded one at a time after `PROMPT`.
const DECODE: [u32; 3] = [7, 64, 250];

/// 20 tokens: on Metal a chunk of >= 16 runs the batched prefill graph.
fn long_prompt() -> Vec<u32> {
    (0..20u32).map(|i| (i * 37 + 11) % 252 + 4).collect()
}

struct Dims {
    h: usize,
    heads: usize,
    kv: usize,
    hd: usize,
    inter: usize,
    layers: usize,
}

#[derive(Default, Clone, Copy)]
struct Opts {
    qk_norm: bool,
    shared_mlp: bool,
    bitnet: bool,
    untied: bool,
}

fn config(model_type: &str, arch: &str, d: &Dims, extra: Value) -> Value {
    let mut c = json!({
        "model_type": model_type, "architectures": [arch], "vocab_size": V,
        "hidden_size": d.h, "num_hidden_layers": d.layers, "num_attention_heads": d.heads,
        "num_key_value_heads": d.kv, "head_dim": d.hd, "intermediate_size": d.inter,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu",
        "max_position_embeddings": 128, "bos_token_id": 1, "eos_token_id": 2,
        "tie_word_embeddings": true,
    });
    for (k, v) in extra.as_object().unwrap() {
        c[k] = v.clone();
    }
    c
}

/// A projection: random, or (BitNet) ternary `{-a, 0, a}` with `a = 1/sqrt(in)`, which online
/// ternarization maps to `{-m, 0, m}` (m = mean |W|) far from any rounding threshold.
fn proj(w: &mut Weights, name: String, shape: [usize; 2], ternary: bool) {
    if !ternary {
        w.w(name, &shape);
        return;
    }
    let a = 1.0 / (shape[1] as f32).sqrt();
    let v = (0..shape[0] * shape[1])
        .map(|_| a * ((w.rng.next_u64() % 3) as f32 - 1.0))
        .collect();
    w.put(name, &shape, v);
}

/// HF Llama tensor names (`model.layers.{l}.self_attn.q_proj.weight`, ...).
fn llama_weights(seed: u64, d: &Dims, o: Opts) -> Weights {
    let mut w = Weights::new(seed);
    // Small next to the block outputs: a tied head over large embeddings just repeats the input.
    w.s("model.embed_tokens.weight", &[V, d.h], 0.05);
    w.norm("model.norm.weight", d.h, 1.0);
    if o.untied {
        w.w("lm_head.weight", &[V, d.h]);
    }
    let (a, kvd) = (d.heads * d.hd, d.kv * d.hd);
    for l in 0..d.layers {
        let p = format!("model.layers.{l}");
        w.norm(format!("{p}.input_layernorm.weight"), d.h, 1.0);
        w.norm(format!("{p}.post_attention_layernorm.weight"), d.h, 1.0);
        for (n, shape) in [
            ("q_proj", [a, d.h]),
            ("k_proj", [kvd, d.h]),
            ("v_proj", [kvd, d.h]),
            ("o_proj", [d.h, a]),
        ] {
            proj(&mut w, format!("{p}.self_attn.{n}.weight"), shape, o.bitnet);
        }
        if o.qk_norm {
            w.norm(format!("{p}.self_attn.q_norm.weight"), d.hd, 1.0);
            w.norm(format!("{p}.self_attn.k_norm.weight"), d.hd, 1.0);
        }
        if o.shared_mlp {
            // Granite-4 fused [gate; up] rows.
            w.w(
                format!("{p}.shared_mlp.input_linear.weight"),
                &[2 * d.inter, d.h],
            );
            w.w(
                format!("{p}.shared_mlp.output_linear.weight"),
                &[d.h, d.inter],
            );
        } else {
            for (n, shape) in [
                ("gate_proj", [d.inter, d.h]),
                ("up_proj", [d.inter, d.h]),
                ("down_proj", [d.h, d.inter]),
            ] {
                proj(&mut w, format!("{p}.mlp.{n}.weight"), shape, o.bitnet);
            }
        }
        if o.bitnet {
            w.norm(format!("{p}.self_attn.attn_sub_norm.weight"), a, 1.0);
            w.norm(format!("{p}.mlp.ffn_sub_norm.weight"), d.inter, 1.0);
        }
    }
    w
}

fn fixture(case: &str, cfg: &Value, w: &Weights) -> PathBuf {
    let dir = fixture_dir(case);
    write_json(&dir.join("config.json"), cfg);
    w.write(&dir.join("model.safetensors"));
    write_tokenizer(&dir, V);
    dir
}

/// runner: the CLI load dispatch (`main.rs`): `load_native` declines GraphLm families, then
/// `load_causal_lm_with` (`--quant` f32: no host-dependent format pick or re-encode) and
/// `ExecutableModel::load_graph_lm` on `dev`.
fn load_lm(dir: &Path, dev: Device) -> ExecutableModel {
    let a = artifacts(dir, "golden/lm", "model.safetensors");
    assert!(load_native(&a, dev).is_none(), "routed to a native loader");
    let (llm, _) = load_causal_lm_with(
        &a,
        dev,
        Some(QuantMode::Uncompressed),
        &HardwareCapability::default(),
        &ReencodeConfig::default(),
    )
    .unwrap();
    let m = ExecutableModel::load_graph_lm(llm, dev).unwrap();
    assert_eq!(m.llm_backend(), dev, "{}", m.backend_label());
    m
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32
}

/// Last-position logits after a 6- and a 20-token prefill, after each of 3 forced tokens
/// following the 6-token prompt, and the 8-token greedy continuation of the 6-token prompt.
fn run_lm(m: &ExecutableModel, dev: Device) -> Out {
    let cpu0 = gibbs_engine::hardware::cpu_compute::entered();
    let mut o = Out::default();
    o.f("prefill6", &m.prompt_logits(PROMPT.to_vec()).unwrap());
    o.f("prefill20", &m.prompt_logits(long_prompt()).unwrap());
    // Prompts shorter than 16 run token by token through the decode program (KV state carried),
    // so these are the logits of 3 cached decode steps after the 6-token prompt.
    let seq = [&PROMPT[..], &DECODE[..]].concat();
    for i in 0..DECODE.len() {
        let l = m
            .prompt_logits(seq[..PROMPT.len() + i + 1].to_vec())
            .unwrap();
        o.f(format!("decode{i}"), &l);
    }
    let (mut ids, mut kv) = (PROMPT.to_vec(), None);
    for _ in 0..8 {
        let n = ids.len();
        let t = m
            .step_token_with_kv(&ids, ids[n - 1], n, 0.0, 1.0, None, &mut kv)
            .map(|(t, _)| t)
            .unwrap();
        ids.push(t);
    }
    // The greedy path (device argmax after the first token) agrees with the logits' argmax.
    for k in 0..8 {
        let l = m.prompt_logits(ids[..PROMPT.len() + k].to_vec()).unwrap();
        assert_eq!(
            argmax(&l),
            ids[PROMPT.len() + k],
            "argmax vs greedy step {k}"
        );
    }
    if dev == Device::Metal {
        // A silent CPU fallback would compare CPU against CPU.
        let n = gibbs_engine::hardware::cpu_compute::entered() - cpu0;
        assert_eq!(n, 0, "Metal run entered CPU compute");
    }
    o.ids("greedy8", &ids[PROMPT.len()..]);
    o
}

fn graph_lm_case(case: &str, cfg: Value, w: Weights) {
    let dir = fixture(case, &cfg, &w);
    golden(case, TOL, |dev| run_lm(&load_lm(&dir, dev), dev));
}

/// SmolLM-style llama: GQA (4 heads, 2 kv), tied head.
#[test]
fn graphlm_llama() {
    let d = Dims {
        h: 64,
        heads: 4,
        kv: 2,
        hd: 16,
        inter: 128,
        layers: 2,
    };
    graph_lm_case(
        "lm_graphlm_llama",
        config("llama", "LlamaForCausalLM", &d, json!({})),
        llama_weights(11, &d, Opts::default()),
    );
}

/// Granite-4 dense (`GraniteMoeHybrid`, attention-only layers): attention/embedding/residual/logit
/// multipliers, fused `shared_mlp`, untied head.
#[test]
fn graphlm_granite() {
    let d = Dims {
        h: 64,
        heads: 4,
        kv: 2,
        hd: 16,
        inter: 128,
        layers: 2,
    };
    let cfg = config(
        "granitemoehybrid",
        "GraniteMoeHybridForCausalLM",
        &d,
        json!({
            "attention_multiplier": 0.1, "embedding_multiplier": 3.0,
            "residual_multiplier": 0.5, "logits_scaling": 4.0,
            "shared_intermediate_size": d.inter, "num_local_experts": 0,
            "layer_types": ["attention", "attention"], "position_embedding_type": "rope",
            "tie_word_embeddings": false,
        }),
    );
    let o = Opts {
        shared_mlp: true,
        untied: true,
        ..Default::default()
    };
    graph_lm_case("lm_graphlm_granite", cfg, llama_weights(12, &d, o));
}

/// LFM2 hybrid: gated short conv (kernel 3) / attention with q/k layernorms and GQA / conv.
#[test]
fn graphlm_lfm2() {
    let d = Dims {
        h: 32,
        heads: 4,
        kv: 2,
        hd: 8,
        inter: 64,
        layers: 3,
    };
    let (conv, k) = ([true, false, true], 3);
    let cfg = config(
        "lfm2",
        "Lfm2ForCausalLM",
        &d,
        json!({
            "norm_eps": 1e-5, "conv_L_cache": k, "conv_bias": false,
            "block_auto_adjust_ff_dim": false, "rope_theta": 1000000.0,
            "layer_types": ["conv", "full_attention", "conv"],
        }),
    );
    let mut w = Weights::new(13);
    w.s("model.embed_tokens.weight", &[V, d.h], 0.05);
    w.norm("model.embedding_norm.weight", d.h, 1.0);
    for (l, &is_conv) in conv.iter().enumerate() {
        let p = format!("model.layers.{l}");
        w.norm(format!("{p}.operator_norm.weight"), d.h, 1.0);
        w.norm(format!("{p}.ffn_norm.weight"), d.h, 1.0);
        if is_conv {
            w.w(format!("{p}.conv.in_proj.weight"), &[3 * d.h, d.h]);
            w.w(format!("{p}.conv.conv.weight"), &[d.h, 1, k]);
            w.w(format!("{p}.conv.out_proj.weight"), &[d.h, d.h]);
        } else {
            let (a, kvd) = (d.heads * d.hd, d.kv * d.hd);
            w.w(format!("{p}.self_attn.q_proj.weight"), &[a, d.h]);
            w.w(format!("{p}.self_attn.k_proj.weight"), &[kvd, d.h]);
            w.w(format!("{p}.self_attn.v_proj.weight"), &[kvd, d.h]);
            w.w(format!("{p}.self_attn.out_proj.weight"), &[d.h, a]);
            w.norm(format!("{p}.self_attn.q_layernorm.weight"), d.hd, 1.0);
            w.norm(format!("{p}.self_attn.k_layernorm.weight"), d.hd, 1.0);
        }
        w.w(format!("{p}.feed_forward.w1.weight"), &[d.inter, d.h]);
        w.w(format!("{p}.feed_forward.w3.weight"), &[d.inter, d.h]);
        w.w(format!("{p}.feed_forward.w2.weight"), &[d.h, d.inter]);
    }
    graph_lm_case("lm_graphlm_lfm2", cfg, w);
}

/// Qwen3-style per-head q/k RMSNorm before RoPE on llama names.
#[test]
fn graphlm_qk_norm() {
    let d = Dims {
        h: 32,
        heads: 2,
        kv: 2,
        hd: 16,
        inter: 64,
        layers: 2,
    };
    let o = Opts {
        qk_norm: true,
        ..Default::default()
    };
    graph_lm_case(
        "lm_graphlm_qk_norm",
        config("llama", "LlamaForCausalLM", &d, json!({})),
        llama_weights(14, &d, o),
    );
}

/// Attention wider than hidden (explicit `head_dim`: 4 x 16 = 64 > 32) with GQA (ERNIE-4.5 / Qwen3 shape).
#[test]
fn graphlm_gqa_wide() {
    let d = Dims {
        h: 32,
        heads: 4,
        kv: 2,
        hd: 16,
        inter: 64,
        layers: 2,
    };
    graph_lm_case(
        "lm_graphlm_gqa_wide",
        config("llama", "LlamaForCausalLM", &d, json!({})),
        llama_weights(15, &d, Opts::default()),
    );
}

/// transformers BitNet (`quant_method: bitnet`, online ternarization, int8 BitLinear activations),
/// relu² MLP, attention/MLP sub-norms. CPU only: the reference is GraphLm's host forward
/// (`final_hidden`); its graph (CPU program and Metal) is known wrong for BitNet.
#[test]
fn bitnet() {
    let case = "lm_bitnet";
    let d = Dims {
        h: 64,
        heads: 4,
        kv: 2,
        hd: 16,
        inter: 128,
        layers: 2,
    };
    let cfg = config(
        "bitnet",
        "BitNetForCausalLM",
        &d,
        json!({
            "hidden_act": "relu2",
            "quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear",
                                    "quantization_mode": "online"},
        }),
    );
    let o = Opts {
        bitnet: true,
        ..Default::default()
    };
    let dir = fixture(case, &cfg, &llama_weights(16, &d, o));
    // runner: internal GraphLm host forward (`load_causal_lm` + `GraphLm::forward`/`generate_step`)
    let llm = load_causal_lm(
        &artifacts(&dir, "golden/lm", "model.safetensors"),
        QuantMode::Uncompressed,
    )
    .unwrap();
    let c = &llm.config;
    assert_eq!(
        (c.act_quant_bits, c.use_relu2_mlp, c.ternarize_weights),
        (8, true, true)
    );
    let mut out = Out::default();
    out.f("prefill6", &llm.forward(&PROMPT));
    let seq = [&PROMPT[..], &DECODE[..]].concat();
    for i in 0..DECODE.len() {
        out.f(
            format!("decode{i}"),
            &llm.forward(&seq[..PROMPT.len() + i + 1]),
        );
    }
    let mut ids = PROMPT.to_vec();
    for _ in 0..8 {
        ids.push(llm.generate_step(&ids, 0.0));
    }
    out.ids("greedy8", &ids[PROMPT.len()..]);
    check(case, Device::Cpu, &out, TOL);
}

/// Nano (host-only toy decoder, no attention mixing): its closed-form init weights written under
/// its own names and layouts (`[in, out]` row-major: `qkv_weights.{l}` [h, 3h], `out_weights.{l}`
/// [h, h] reading q only, `mlp_gate_up.{l}` [h, 2*inter] on the un-normed residual, `mlp_down.{l}`
/// [inter, h], `lm_head` [h, v]). 10-step greedy trajectory from token 42 and the logits of the
/// last step. CPU only (no device path).
#[test]
fn nano() {
    let case = "lm_nano";
    let (v, h, inter, layers) = (256usize, 64usize, 128usize, 2usize);
    let mut w = Weights::new(0);
    let f = |n: usize, g: &dyn Fn(usize) -> f32| (0..n).map(g).collect::<Vec<f32>>();
    w.put(
        "embedding_table",
        &[v, h],
        f(v * h, &|i| (i as f32 * 0.01).sin() * 0.1),
    );
    for l in 0..layers {
        let qkv = f(h * 3 * h, &|i| ((i + l) as f32 * 0.02).cos() * 0.05);
        w.put(format!("qkv_weights.{l}"), &[h, 3 * h], qkv);
        let out = f(h * h, &|i| ((i + l * 2) as f32 * 0.03).sin() * 0.05);
        w.put(format!("out_weights.{l}"), &[h, h], out);
        let gu = f(h * 2 * inter, &|i| ((i + l * 3) as f32 * 0.01).cos() * 0.05);
        w.put(format!("mlp_gate_up.{l}"), &[h, 2 * inter], gu);
        let down = f(inter * h, &|i| ((i + l * 4) as f32 * 0.02).sin() * 0.05);
        w.put(format!("mlp_down.{l}"), &[inter, h], down);
        w.put(format!("norm_weights.{l}"), &[h], vec![1.0; h]);
    }
    w.put("final_norm", &[h], vec![1.0; h]);
    w.put(
        "lm_head",
        &[h, v],
        f(h * v, &|i| (i as f32 * 0.005).sin() * 0.1),
    );
    let dir = fixture_dir(case);
    write_json(
        &dir.join("config.json"),
        &json!({"hidden_size": h, "num_hidden_layers": layers, "num_attention_heads": 4,
                "intermediate_size": inter, "vocab_size": v}),
    );
    w.write(&dir.join("model.safetensors"));
    // runner: internal `NanoTransformerModel` fields filled from the fixture file (the removed
    // `HuggingFaceModelDir` loader did the same) + `forward_token` / `forward_logits`
    let cfg = NanoTransformerConfig {
        vocab_size: v,
        hidden_dim: h,
        intermediate_dim: inter,
        num_layers: layers,
        num_heads: 4,
        head_dim: 16,
        eps: 1e-5,
    };
    let (m, data) =
        gibbs_engine::model::loaders::read_checkpoint(&dir.join("model.safetensors")).unwrap();
    let t = |n: &str| m.extract_tensor_f32(&data, n).unwrap();
    let mut model = NanoTransformerModel::new(cfg.clone());
    model.embedding_table = t("embedding_table");
    for l in 0..layers {
        model.qkv_weights[l] = t(&format!("qkv_weights.{l}"));
        model.out_weights[l] = t(&format!("out_weights.{l}"));
        model.mlp_gate_up[l] = t(&format!("mlp_gate_up.{l}"));
        model.mlp_down[l] = t(&format!("mlp_down.{l}"));
        model.norm_weights[l] = t(&format!("norm_weights.{l}"));
    }
    model.final_norm = t("final_norm");
    model.lm_head = t("lm_head");
    model.quantize_model(QuantMode::Uncompressed);
    let trajectory = |m: &NanoTransformerModel| {
        let mut t = 42u32;
        (0..10)
            .map(|pos| {
                t = m.forward_token(t, pos);
                t
            })
            .collect::<Vec<u32>>()
    };
    let ids = trajectory(&model);
    // The fixture is exactly Nano's built-in init (nano_model_tests' config).
    assert_eq!(ids, trajectory(&NanoTransformerModel::new(cfg)));
    let mut out = Out::default();
    out.ids("greedy10", &ids);
    out.f("final_logits", &model.forward_logits(ids[8]));
    check(case, Device::Cpu, &out, TOL);
}
