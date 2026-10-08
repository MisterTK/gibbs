//! Block drafters for speculative decoding, conditioned on target hidden states: z-lab
//! `DFlashDraftModel` / `DFlash2DraftModel` and the vLLM speculators `dflash` / `dflash2` /
//! `dspark` formats (plus dense `Qwen3DSparkModel`). One non-causal pass of a few Qwen3 layers
//! over `[anchor, mask, ...]` whose keys/values also cover every committed position (projected
//! from the target's tapped hidden states) predicts a whole block. DFlash2 adds grouped dynamic
//! convolutions around each sublayer and a top-k candidate selector walked left to right;
//! DSpark adds a low-rank Markov bias on the previous token.
//!
//! References: github.com/z-lab/dflash `dflash/model.py`; vllm-project/speculators
//! `models/{dflash,dflash2,dspark}`; vLLM `model_executor/models/qwen3_dspark.py` and
//! `v1/worker/gpu/spec_decode/dspark/speculator.py`.

use super::{linear, linear_bf16, rms_norm, rms_norm_rows, vector_dot, Linear};
use crate::model::nn::add_in_place;
use crate::model::par::{par_for, par_rows, OutPtr};
use crate::model::simd::{vector_fused_scale_add, vector_softmax, vector_swiglu};
use crate::model::weights::{Mat, Weights};
use crate::model::ModelManifest;
use crate::scheduler::spec::{DraftState, Drafter};
use serde_json::Value;
use std::sync::Arc;

/// Grouped dynamic causal convolution around one sublayer (DFlash2).
struct Conv {
    /// `[2, kernel, hidden]`: input-side then output-side base kernel.
    base: Vec<f32>,
    /// hidden -> `[2, kernel, groups]` per-position kernel deltas.
    proj: Linear,
    kernel: usize,
    group: usize,
}

impl Conv {
    /// `out[i] = sum_t x[i-t] * (base[side][t] + delta[i][side][t][group])` within the block.
    fn apply(&self, x: &[f32], n: usize, delta: &[f32], side: usize) -> Vec<f32> {
        let d = x.len() / n;
        let (k, groups) = (self.kernel, d / self.group);
        let mut out = vec![0.0f32; n * d];
        par_rows(&mut out, d, &|i, row| {
            let dl = &delta[i * 2 * k * groups + side * k * groups..][..k * groups];
            for t in 0..k.min(i + 1) {
                let src = &x[(i - t) * d..][..d];
                let base = &self.base[(side * k + t) * d..][..d];
                for (c, o) in row.iter_mut().enumerate() {
                    *o += src[c] * (base[c] + dl[t * groups + c / self.group]);
                }
            }
        });
        out
    }
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q: Linear,
    /// `k_proj | v_proj`.
    kv: Linear,
    o: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// `gate_proj | up_proj`.
    gate_up: Linear,
    down: Linear,
    /// Mask positions see only earlier mask positions (sliding layers of most drafters).
    causal: bool,
    window: Option<usize>,
    /// DFlash2 `(attention_conv, mlp_conv)`.
    conv: Option<(Conv, Conv)>,
}

/// How draft hidden rows become target-vocabulary token choices.
enum Pick {
    /// Target lm_head, argmax/sample per row (DFlash).
    Plain,
    /// Target lm_head, then the top-k candidates rescored by a predecessor-conditioned bilinear
    /// term (DFlash2 `CandidateSelector`).
    Selector {
        pred: Mat,
        succ: Mat,
        proj: Linear,
        top_k: usize,
    },
    /// Own draft-vocab head (`d2t[i]` = target id of draft id i, when the vocab is pruned) plus the
    /// vanilla Markov bias `w2 . w1[previous token]` (DSpark).
    Markov {
        head: Option<Linear>,
        d2t: Option<Vec<u32>>,
        w1: Mat,
        w2: Mat,
    },
}

struct BlockDrafter {
    label: String,
    taps: Vec<usize>,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    inv_freq: Vec<f32>,
    block: usize,
    mask: u32,
    /// DSpark `sample_from_anchor`: every block row predicts the token after it (block drafts);
    /// else the anchor row is context and mask rows predict their own position (block - 1).
    from_anchor: bool,
    fc: Linear,
    hidden_norm: Vec<f32>,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    pick: Pick,
    /// Target token embeddings and output head (`[vocab, hidden]`).
    embed: Mat,
    lm_head: Mat,
}

/// Committed context projected to each layer's keys (normed, rotated) and values.
struct Ctx {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

/// Architectures [`load`] reads.
pub const ARCHITECTURES: &[&str] = &[
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DSparkDraftModel",
    "Qwen3DSparkModel",
];

/// Builds the drafter in checkpoint `ck` (z-lab / speculators tensor names; `cfg` is its
/// config.json, or [`gguf_checkpoint`]'s equivalent) against the target checkpoint `tw` (config
/// `tcfg`), whose embeddings and lm_head it shares. `name` labels it.
pub fn load(
    name: &str,
    ck: &Weights,
    cfg: &Value,
    tw: &Weights,
    tcfg: &Value,
) -> Result<Arc<dyn Drafter>, String> {
    let arch = cfg["architectures"][0].as_str().unwrap_or("");
    // speculators checkpoints nest the Qwen3 layer config and count taps as "input of layer i".
    let spec = cfg["speculators_model_type"].as_str();
    let (c, dc) = match spec {
        Some(_) => (&cfg["transformer_layer_config"], cfg),
        None if cfg["dflash_config"].is_object() => (cfg, &cfg["dflash_config"]),
        None => (cfg, cfg),
    };
    let field = |k: &str| {
        if dc[k].is_null() {
            &cfg[k]
        } else {
            &dc[k]
        }
    };
    let get = |k: &str| {
        c[k].as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("drafter config: {k} missing"))
    };
    for k in [
        "output_multiplier",
        "final_logit_softcapping",
        "input_embedding_scale",
    ] {
        if !field(k).is_null() && field(k).as_f64() != Some(1.0) {
            return Err(format!("drafter config sets {k}, which is not implemented"));
        }
    }
    let taps: Vec<usize> = match (spec, field("target_layer_ids").as_array()) {
        (None, Some(ids)) => ids
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| v as usize)
            .collect(),
        (Some(_), _) => cfg["aux_hidden_state_layer_ids"]
            .as_array()
            .ok_or("speculators drafter has no aux_hidden_state_layer_ids")?
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| {
                (v as usize)
                    .checked_sub(1)
                    .ok_or("aux hidden-state layer 0")
            })
            .collect::<Result<_, _>>()?,
        (None, None) => return Err("drafter config has no target_layer_ids".into()),
    };
    let block = field("block_size")
        .as_u64()
        .ok_or("drafter config has no block_size")? as usize;
    let mask = field("mask_token_id")
        .as_u64()
        .ok_or("drafter config has no mask_token_id")? as u32;
    let dspark = spec == Some("dspark") || arch.contains("DSpark");
    let from_anchor = field("sample_from_anchor").as_bool().unwrap_or(dspark);
    if block < 2 {
        return Err(format!("block_size {block} drafts nothing"));
    }

    let tc = if tcfg["text_config"].is_object() {
        &tcfg["text_config"]
    } else {
        tcfg
    };
    let t_layers = tc["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    if let Some(&bad) = taps.iter().find(|&&t| t >= t_layers) {
        return Err(format!(
            "drafter taps target layer {bad}; the target has {t_layers} layers"
        ));
    }

    let (hidden, heads, kv_heads) = (
        get("hidden_size")?,
        get("num_attention_heads")?,
        get("num_key_value_heads")?,
    );
    let head_dim = get("head_dim").unwrap_or(hidden / heads);
    let rope = if c["rope_parameters"].is_object() {
        &c["rope_parameters"]
    } else {
        &c["rope_scaling"]
    };
    let rope_type = rope["rope_type"].as_str().unwrap_or("default");
    if !matches!(rope_type, "default" | "llama3")
        || rope["partial_rotary_factor"]
            .as_f64()
            .is_some_and(|f| f != 1.0)
    {
        return Err(format!("drafter rope {rope} is not implemented"));
    }
    let theta = rope["rope_theta"]
        .as_f64()
        .or(c["rope_theta"].as_f64())
        .ok_or("drafter config has no rope_theta")?;
    let llama3 = |f: f64| -> f64 {
        // transformers `_compute_llama3_parameters`: long wavelengths slowed by `factor`,
        // short ones kept, the band between interpolated.
        let p = |k: &str| rope[k].as_f64().unwrap_or(1.0);
        let (factor, lo, hi, orig) = (
            p("factor"),
            p("low_freq_factor"),
            p("high_freq_factor"),
            p("original_max_position_embeddings"),
        );
        let wavelen = 2.0 * std::f64::consts::PI / f;
        if wavelen < orig / hi {
            f
        } else if wavelen > orig / lo {
            f / factor
        } else {
            let s = (orig / wavelen - lo) / (hi - lo);
            (1.0 - s) * f / factor + s * f
        }
    };
    let inv_freq = (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .map(|f| if rope_type == "llama3" { llama3(f) } else { f } as f32)
        .collect();
    let n_layers = get("num_hidden_layers")?;
    let types = c["layer_types"].as_array();
    // speculators: sliding layers are causal unless `sliding_window_non_causal`; z-lab:
    // `is_causal` overrides, else sliding layers are causal. Full layers are bidirectional.
    let non_causal = spec.is_some() && cfg["sliding_window_non_causal"] == true;
    let window = c["sliding_window"].as_u64().map(|w| w as usize);

    let vec = |n: &str| ck.vec(n);
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let l = format!("layers.{i}");
        let a = format!("{l}.self_attn");
        let sliding = types.is_some_and(|t| t.get(i).is_some_and(|v| v == "sliding_attention"));
        let causal = match cfg["is_causal"].as_bool() {
            Some(b) if spec.is_none() => b,
            _ => sliding && !non_causal,
        };
        let conv = if ck.has(&format!("{l}.attention_conv.base_kernel")) {
            let conv = |p: &str| -> Result<Conv, String> {
                let kernel = ck.shape(&format!("{l}.{p}.base_kernel"))?[1];
                let group = field("conv_group_size")
                    .as_u64()
                    .ok_or("DFlash2 drafter has no conv_group_size")?
                    as usize;
                Ok(Conv {
                    base: vec(&format!("{l}.{p}.base_kernel"))?,
                    proj: linear(&ck, &[&format!("{l}.{p}.kernel_projection")], false)?,
                    kernel,
                    group,
                })
            };
            Some((conv("attention_conv")?, conv("mlp_conv")?))
        } else {
            None
        };
        layers.push(Layer {
            in_norm: vec(&format!("{l}.input_layernorm.weight"))?,
            post_norm: vec(&format!("{l}.post_attention_layernorm.weight"))?,
            q: linear(&ck, &[&format!("{a}.q_proj")], false)?,
            kv: linear(
                &ck,
                &[&format!("{a}.k_proj"), &format!("{a}.v_proj")],
                false,
            )?,
            o: linear(&ck, &[&format!("{a}.o_proj")], false)?,
            q_norm: vec(&format!("{a}.q_norm.weight"))?,
            k_norm: vec(&format!("{a}.k_norm.weight"))?,
            gate_up: linear(
                &ck,
                &[&format!("{l}.mlp.gate_proj"), &format!("{l}.mlp.up_proj")],
                false,
            )?,
            down: linear(&ck, &[&format!("{l}.mlp.down_proj")], false)?,
            causal,
            window: window.filter(|_| sliding),
            conv,
        });
    }

    let embed_name = [
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ]
    .into_iter()
    .find(|n| tw.has(n))
    .ok_or("target checkpoint has no embed_tokens")?;
    let embed = tw.mat(embed_name)?;
    let lm_head = if tw.has("lm_head.weight") {
        tw.mat("lm_head.weight")?
    } else {
        embed.clone()
    };
    let t_vocab = embed.rows;
    let pick = if ck.has("candidate_selector.predecessor_codebook") {
        Pick::Selector {
            pred: ck.mat("candidate_selector.predecessor_codebook")?,
            succ: ck.mat("candidate_selector.successor_codebook")?,
            proj: linear(&ck, &["candidate_selector.hidden_projection"], false)?,
            top_k: field("selector_top_k")
                .as_u64()
                .ok_or("DFlash2 drafter has no selector_top_k")? as usize,
        }
    } else if ck.has("markov_head.markov_w1.weight") {
        if field("markov_head_type").as_str().unwrap_or("vanilla") != "vanilla" {
            return Err(format!(
                "DSpark markov_head_type {} is not implemented (vanilla only)",
                field("markov_head_type")
            ));
        }
        let head = if ck.has("lm_head.weight") {
            Some(linear(&ck, &["lm_head"], false)?)
        } else {
            None
        };
        let d_vocab = head.as_ref().map_or(t_vocab, |h| h.rows);
        let d2t = if d_vocab < t_vocab {
            let off = vec("d2t").map_err(|_| "pruned-vocab DSpark drafter ships no d2t")?;
            if off.len() != d_vocab {
                return Err(format!(
                    "d2t has {} entries for {d_vocab} draft ids",
                    off.len()
                ));
            }
            Some(
                off.iter()
                    .enumerate()
                    .map(|(i, &o)| i as u32 + o as u32)
                    .collect::<Vec<u32>>(),
            )
        } else {
            None
        };
        if d2t
            .as_ref()
            .is_some_and(|m| m.iter().any(|&t| t as usize >= t_vocab))
        {
            return Err("d2t maps outside the target vocabulary".into());
        }
        Pick::Markov {
            head,
            d2t,
            w1: ck.mat("markov_head.markov_w1.weight")?,
            w2: ck.mat("markov_head.markov_w2.weight")?,
        }
    } else {
        Pick::Plain
    };
    let m = BlockDrafter {
        label: format!(
            "{} {name} (block {block})",
            match &pick {
                Pick::Plain => "DFlash",
                Pick::Selector { .. } => "DFlash2",
                Pick::Markov { .. } => "DSpark",
            }
        ),
        fc: linear(&ck, &["fc"], false)?,
        hidden_norm: vec("hidden_norm.weight")?,
        norm: vec("norm.weight")?,
        taps,
        hidden,
        heads,
        kv_heads,
        head_dim,
        inter: get("intermediate_size")?,
        eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        inv_freq,
        block,
        mask,
        from_anchor,
        layers,
        pick,
        embed,
        lm_head,
    };
    m.check(tc, t_vocab)?;
    Ok(Arc::new(m))
}

/// A llama.cpp `dflash` GGUF sidecar (`convert_hf_to_gguf.py` `DFlashModel` / `DSparkModel`) as the
/// checkpoint and `config.json` equivalent [`load`] reads: tensors renamed to z-lab's names and
/// dequantized to bf16, `dflash.*` metadata mapped back to the config fields the converter wrote
/// it from. Everything the drafter graph here does not run (own token embeddings, post-norms,
/// attention sinks, value / attention scales, pruned draft vocabularies) is an Err, never ignored.
pub fn gguf_checkpoint(m: &ModelManifest, data: &[u8]) -> Result<(Weights, Value), String> {
    let a = m
        .metadata
        .get("general.architecture")
        .map_or("", String::as_str);
    if a != "dflash" {
        return Err(format!(
            "GGUF architecture '{a}' is not llama.cpp's 'dflash' draft layout"
        ));
    }
    let md = |k: &str| m.metadata.get(&format!("dflash.{k}"));
    let num = |k: &str| md(k).and_then(|v| v.parse::<f64>().ok());
    let int = |k: &str| {
        num(k)
            .map(|v| v as usize)
            .ok_or_else(|| format!("GGUF metadata dflash.{k} missing"))
    };
    for k in ["attention.value_scale", "attention.scale"] {
        if num(k).is_some_and(|v| v != 1.0) {
            return Err(format!(
                "drafter GGUF sets dflash.{k}, which is not implemented"
            ));
        }
    }
    if let Some(t) = md("rope.scaling.type").filter(|t| *t != "none") {
        return Err(format!("drafter RoPE scaling '{t}' is not implemented"));
    }
    let (hidden, heads, layers) = (
        int("embedding_length")?,
        int("attention.head_count")?,
        int("block_count")?,
    );
    let head_dim = int("attention.key_length").unwrap_or(hidden / heads.max(1));
    let rot = int("rope.dimension_count").unwrap_or(head_dim);
    // `dflash.target_layers` are the converter's `target_layer_ids + 1` (hidden_states indices).
    let taps = md("target_layers")
        .ok_or("GGUF metadata dflash.target_layers missing")?
        .split(',')
        .map(|v| {
            v.parse::<usize>()
                .ok()
                .and_then(|v| v.checked_sub(1))
                .ok_or_else(|| format!("dflash.target_layers entry '{v}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let window = int("attention.sliding_window").ok().filter(|&w| w > 0);
    let pattern: Vec<bool> = md("attention.sliding_window_pattern")
        .map(|s| s.split(',').map(|v| v == "1").collect())
        .unwrap_or_default();
    let types: Vec<&str> = (0..layers)
        .map(
            |i| match window.is_some() && pattern.get(i) == Some(&true) {
                true => "sliding_attention",
                false => "full_attention",
            },
        )
        .collect();
    let markov = m.tensors.contains_key("markov_w1.weight");
    let selector = m.tensors.contains_key("selector_hidden.weight");
    let mut cfg = serde_json::json!({
        "architectures": [match (markov, selector) {
            (true, _) => "DSparkDraftModel",
            (_, true) => "DFlash2DraftModel",
            _ => "DFlashDraftModel",
        }],
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": int("attention.head_count_kv")?,
        "head_dim": head_dim,
        "intermediate_size": int("feed_forward_length")?,
        "rms_norm_eps": num("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        "rope_parameters": {
            "rope_theta": num("rope.freq_base").ok_or("GGUF metadata dflash.rope.freq_base missing")?,
            "partial_rotary_factor": rot as f64 / head_dim as f64,
        },
        "layer_types": types,
        "sliding_window": window,
        "block_size": int("block_size")?,
        "mask_token_id": m.metadata.get("tokenizer.ggml.mask_token_id").and_then(|v| v.parse::<u64>().ok()),
        "target_layer_ids": taps,
        "output_multiplier": num("logit_scale"),
        "final_logit_softcapping": num("final_logit_softcapping"),
        "input_embedding_scale": num("embedding_scale"),
        "conv_group_size": int("conv_group_size").ok(),
        "selector_top_k": int("selector_top_k").ok(),
    });
    if let Some(c) = md("attention.causal") {
        cfg["is_causal"] = (c == "1").into();
    }
    if markov {
        // llama.cpp: absent means the anchor slot is a prediction slot too.
        cfg["sample_from_anchor"] = (md("sample_from_anchor").map_or(true, |v| v == "1")).into();
    }
    let mut gguf_name = std::collections::HashMap::new();
    let mut specs = Vec::with_capacity(m.tensors.len());
    for (name, t) in &m.tensors {
        let Some(hf) = drafter_tensor(name)? else {
            continue;
        };
        // GGUF lists dims fastest first.
        specs.push((hf.clone(), t.shape.iter().rev().copied().collect()));
        gguf_name.insert(hf, name.as_str());
    }
    let ck = Weights::from_f32(specs, |hf| m.extract_tensor_f32(data, gguf_name[hf]))?;
    Ok((ck, cfg))
}

/// llama.cpp `dflash` tensor name -> z-lab / speculators name (`None`: read by nothing here).
fn drafter_tensor(name: &str) -> Result<Option<String>, String> {
    let top = match name {
        "fc.weight" => "fc.weight",
        "enc.output_norm.weight" => "hidden_norm.weight",
        "output_norm.weight" => "norm.weight",
        "output.weight" => "lm_head.weight",
        "markov_w1.weight" => "markov_head.markov_w1.weight",
        "markov_w2.weight" => "markov_head.markov_w2.weight",
        "selector_predecessor.weight" => "candidate_selector.predecessor_codebook",
        "selector_successor.weight" => "candidate_selector.successor_codebook",
        "selector_hidden.weight" => "candidate_selector.hidden_projection.weight",
        // The DSpark confidence head only gates llama.cpp's own drafting.
        "conf_proj.weight" | "conf_proj.bias" => return Ok(None),
        _ => "",
    };
    if !top.is_empty() {
        return Ok(Some(top.into()));
    }
    let unsupported =
        || format!("GGUF tensor '{name}' is not part of the drafter graph this loader runs");
    let (layer, rest) = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .ok_or_else(unsupported)?;
    let hf = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_conv_base" => "attention_conv.base_kernel",
        "attn_conv_proj.weight" => "attention_conv.kernel_projection.weight",
        "ffn_conv_base" => "mlp_conv.base_kernel",
        "ffn_conv_proj.weight" => "mlp_conv.kernel_projection.weight",
        _ => return Err(unsupported()),
    };
    Ok(Some(format!("layers.{layer}.{hf}")))
}

impl BlockDrafter {
    /// Shape agreement between drafter, target and config, so a mismatched pair errs at attach.
    fn check(&self, tc: &Value, t_vocab: usize) -> Result<(), String> {
        let t_hidden = tc["hidden_size"].as_u64().unwrap_or(0) as usize;
        if self.embed.cols != self.hidden || self.lm_head.cols != self.hidden {
            return Err(format!(
                "drafter hidden {} != target embedding width {}",
                self.hidden, self.embed.cols
            ));
        }
        if self.fc.cols != self.taps.len() * t_hidden || self.fc.rows != self.hidden {
            return Err(format!(
                "drafter fc takes {} features; {} taps of the target's hidden {t_hidden} give {}",
                self.fc.cols,
                self.taps.len(),
                self.taps.len() * t_hidden
            ));
        }
        if self.mask as usize >= t_vocab {
            return Err(format!(
                "mask token {} is outside the target vocabulary",
                self.mask
            ));
        }
        let (hd, q, kv) = (
            self.head_dim,
            self.heads * self.head_dim,
            self.kv_heads * self.head_dim,
        );
        for (i, l) in self.layers.iter().enumerate() {
            let conv_ok = l.conv.as_ref().map_or(true, |(a, b)| {
                [a, b].iter().all(|c| {
                    self.hidden % c.group == 0
                        && c.base.len() == 2 * c.kernel * self.hidden
                        && c.proj.rows == 2 * c.kernel * (self.hidden / c.group)
                })
            });
            if l.q.rows != q
                || l.kv.rows != 2 * kv
                || l.o.cols != q
                || l.q_norm.len() != hd
                || l.gate_up.rows != 2 * self.inter
                || !conv_ok
            {
                return Err(format!(
                    "drafter layer {i}: shapes disagree with config.json"
                ));
            }
        }
        match &self.pick {
            Pick::Selector {
                pred,
                succ,
                proj,
                top_k,
            } => {
                if pred.rows != t_vocab || succ.rows != t_vocab || proj.rows != pred.cols {
                    return Err("candidate selector shapes disagree with the target vocab".into());
                }
                if *top_k == 0 || *top_k > t_vocab {
                    return Err(format!("selector_top_k {top_k} is out of range"));
                }
            }
            Pick::Markov { head, w1, w2, .. } => {
                let d_vocab = head.as_ref().map_or(t_vocab, |h| h.rows);
                if w1.rows != t_vocab || w2.rows != d_vocab || w1.cols != w2.cols {
                    return Err("Markov head shapes disagree with the vocabularies".into());
                }
            }
            Pick::Plain => {}
        }
        Ok(())
    }

    /// Rotate-half RoPE of one head at `pos`.
    fn rope(&self, x: &mut [f32], pos: usize) {
        let half = self.inv_freq.len();
        for (i, &f) in self.inv_freq.iter().enumerate() {
            let (s, c) = (pos as f32 * f).sin_cos();
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * c - b * s;
            x[i + half] = b * c + a * s;
        }
    }

    /// `[k | v]` rows -> per-kv-head normed, rotated keys (in place) at positions `start..`.
    fn keys(&self, l: &Layer, kv: &mut [f32], start: usize) {
        let (hd, kvw) = (self.head_dim, self.kv_heads * self.head_dim);
        for (i, row) in kv.chunks_exact_mut(2 * kvw).enumerate() {
            for k in row[..kvw].chunks_exact_mut(hd) {
                rms_norm(k, &l.k_norm, self.eps);
                self.rope(k, start + i);
            }
        }
    }

    /// Appends committed rows of tapped target hidden states (`[rows, taps * hidden]`).
    fn extend(&self, ctx: &mut Ctx, hidden: &[f32]) {
        let rows = hidden.len() / self.fc.cols;
        if rows == 0 {
            return;
        }
        let mut c = self.fc.forward(hidden, rows);
        rms_norm_rows(&mut c, &self.hidden_norm, self.eps);
        let kvw = self.kv_heads * self.head_dim;
        for (l, (k, v)) in self
            .layers
            .iter()
            .zip(ctx.k.iter_mut().zip(ctx.v.iter_mut()))
        {
            let mut kv = l.kv.forward(&c, rows);
            self.keys(l, &mut kv, ctx.len);
            for row in kv.chunks_exact(2 * kvw) {
                k.extend_from_slice(&row[..kvw]);
                v.extend_from_slice(&row[kvw..]);
            }
        }
        ctx.len += rows;
    }

    /// Attention of block rows `x` (positions `start..`) over the context plus the block.
    fn attend(
        &self,
        l: &Layer,
        x: &[f32],
        n: usize,
        start: usize,
        ck: &[f32],
        cv: &[f32],
    ) -> Vec<f32> {
        let (hd, heads, kvh) = (self.head_dim, self.heads, self.kv_heads);
        let (qw, kvw) = (heads * hd, kvh * hd);
        let mut q = l.q.forward(x, n);
        for (i, row) in q.chunks_exact_mut(qw).enumerate() {
            for h in row.chunks_exact_mut(hd) {
                rms_norm(h, &l.q_norm, self.eps);
                self.rope(h, start + i);
            }
        }
        let mut kv = l.kv.forward(x, n);
        self.keys(l, &mut kv, start);
        let ctx = ck.len() / kvw;
        let mut out = vec![0.0f32; n * qw];
        let o = OutPtr::new(&mut out);
        let scale = 1.0 / (hd as f32).sqrt();
        par_for(heads * n, &|t| {
            let (h, i) = (t / n, t % n);
            let g = h / (heads / kvh);
            let qh = &q[i * qw + h * hd..][..hd];
            let qpos = start + i;
            // Keys at positions 0..ctx (context), then start.. (block).
            let visible = |kpos: usize, block: Option<usize>| {
                (!l.causal || block.map_or(true, |j| j <= i))
                    && l.window.map_or(true, |w| {
                        qpos.saturating_sub(kpos) < w && (l.causal || kpos.saturating_sub(qpos) < w)
                    })
            };
            let mut p = Vec::with_capacity(ctx + n);
            let mut vrows: Vec<&[f32]> = Vec::with_capacity(ctx + n);
            for j in 0..ctx {
                if visible(j, None) {
                    p.push(vector_dot(qh, &ck[j * kvw + g * hd..][..hd]) * scale);
                    vrows.push(&cv[j * kvw + g * hd..][..hd]);
                }
            }
            for j in 0..n {
                if visible(start + j, Some(j)) {
                    let row = &kv[j * 2 * kvw..];
                    p.push(vector_dot(qh, &row[g * hd..][..hd]) * scale);
                    vrows.push(&row[kvw + g * hd..][..hd]);
                }
            }
            vector_softmax(&mut p);
            // SAFETY: task (h, i) alone writes head h of row i.
            let dst = unsafe { o.slice(i * qw + h * hd, hd) };
            for (&pj, v) in p.iter().zip(vrows) {
                vector_fused_scale_add(dst, v, pj);
            }
        });
        l.o.forward(&out, n)
    }

    /// The draft pass over block ids at positions `start..`: final-normed rows `[n, hidden]`.
    fn run(&self, ctx: &Ctx, ids: &[u32], start: usize) -> Vec<f32> {
        let (n, d) = (ids.len(), self.hidden);
        let mut h = vec![0.0f32; n * d];
        for (row, &t) in h.chunks_exact_mut(d).zip(ids) {
            for (o, &w) in row.iter_mut().zip(self.embed.row(t as usize)) {
                *o = f32::from_bits((w as u32) << 16);
            }
        }
        // ponytail: a conv's two kernels come from one projection of its sublayer input.
        let sub = |x: &[f32], conv: Option<&Conv>, f: &dyn Fn(&[f32]) -> Vec<f32>| match conv {
            None => f(x),
            Some(c) => {
                let delta = c.proj.forward(x, n);
                c.apply(&f(&c.apply(x, n, &delta, 0)), n, &delta, 1)
            }
        };
        for (li, l) in self.layers.iter().enumerate() {
            let mut x = h.clone();
            rms_norm_rows(&mut x, &l.in_norm, self.eps);
            let a = sub(&x, l.conv.as_ref().map(|c| &c.0), &|x| {
                self.attend(l, x, n, start, &ctx.k[li], &ctx.v[li])
            });
            add_in_place(&mut h, &a);
            let mut x = h.clone();
            rms_norm_rows(&mut x, &l.post_norm, self.eps);
            let m = sub(&x, l.conv.as_ref().map(|c| &c.1), &|x| {
                let gu = l.gate_up.forward(x, n);
                let mut act = vec![0.0f32; n * self.inter];
                let inter = self.inter;
                par_rows(&mut act, inter, &|i, row| {
                    vector_swiglu(&gu[i * 2 * inter..][..2 * inter], row, inter)
                });
                l.down.forward(&act, n)
            });
            add_in_place(&mut h, &m);
        }
        rms_norm_rows(&mut h, &self.norm, self.eps);
        h
    }
}

fn lm(m: &Mat, x: &[f32], n: usize) -> Vec<f32> {
    linear_bf16(&[m.data()], m.cols, None, x, n)
}

fn bf16_row(m: &Mat, r: usize) -> Vec<f32> {
    m.row(r)
        .iter()
        .map(|&w| f32::from_bits((w as u32) << 16))
        .collect()
}

/// Indices of the `k` largest values (any order).
fn top_k(x: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| x[b].total_cmp(&x[a]));
    idx.truncate(k);
    idx
}

impl Drafter for BlockDrafter {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn max_k(&self) -> usize {
        // ponytail: verify of <= 16 rows is bit-exact with one-token decode (qwen3_5 multi-row
        // GEMV); larger blocks are capped there until a wider exact path exists.
        (self.block - usize::from(!self.from_anchor)).min(15)
    }

    fn tap_layers(&self) -> Vec<usize> {
        self.taps.clone()
    }

    fn new_state(&self) -> DraftState {
        Box::new(Ctx {
            k: vec![Vec::new(); self.layers.len()],
            v: vec![Vec::new(); self.layers.len()],
            len: 0,
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<Vec<u32>, String> {
        let ctx = st
            .downcast_mut::<Ctx>()
            .ok_or("draft state was not built by this drafter")?;
        if hidden.len() % self.fc.cols != 0 {
            return Err(format!(
                "tapped hidden states are not whole rows of {}",
                self.fc.cols
            ));
        }
        self.extend(ctx, hidden);
        let (&anchor, committed) = tokens.split_last().ok_or("empty sequence")?;
        if ctx.len != committed.len() {
            return Err(format!(
                "drafter context holds {} positions; the target has run {}",
                ctx.len,
                committed.len()
            ));
        }
        let first = usize::from(!self.from_anchor);
        let m = k.min(self.block - first);
        if m == 0 {
            return Ok(Vec::new());
        }
        let mut ids = vec![self.mask; self.block];
        ids[0] = anchor;
        let h = self.run(ctx, &ids, committed.len());
        let rows = &h[first * self.hidden..(first + m) * self.hidden];
        let mut out = Vec::with_capacity(m);
        match &self.pick {
            Pick::Plain => {
                let logits = lm(&self.lm_head, rows, m);
                for row in logits.chunks_exact(self.lm_head.rows) {
                    out.push(pick(row));
                }
            }
            Pick::Selector {
                pred,
                succ,
                proj,
                top_k: kk,
            } => {
                let v = self.lm_head.rows;
                let unary = lm(&self.lm_head, rows, m);
                let hp = proj.forward(rows, m);
                let r = proj.rows;
                let mut prev = anchor;
                let mut row = vec![f32::NEG_INFINITY; v];
                for i in 0..m {
                    let u = &unary[i * v..][..v];
                    let c: Vec<f32> = bf16_row(pred, prev as usize)
                        .iter()
                        .zip(&hp[i * r..][..r])
                        .map(|(a, b)| a * b)
                        .collect();
                    row.fill(f32::NEG_INFINITY);
                    for cand in top_k(u, *kk) {
                        row[cand] = u[cand] + vector_dot(&c, &bf16_row(succ, cand));
                    }
                    prev = pick(&row);
                    out.push(prev);
                }
            }
            Pick::Markov { head, d2t, w1, w2 } => {
                let base = match head {
                    Some(hd) => hd.forward(rows, m),
                    None => lm(&self.lm_head, rows, m),
                };
                let dv = base.len() / m;
                let mut full = vec![f32::NEG_INFINITY; self.lm_head.rows];
                let mut prev = anchor;
                for i in 0..m {
                    let bias = lm(w2, &bf16_row(w1, prev as usize), 1);
                    let mut row = base[i * dv..][..dv].to_vec();
                    add_in_place(&mut row, &bias);
                    let t = match d2t {
                        Some(map) => {
                            for (&t, &l) in map.iter().zip(&row) {
                                full[t as usize] = l;
                            }
                            pick(&full)
                        }
                        None => pick(&row),
                    };
                    prev = t;
                    out.push(t);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::loaders::gguf::{open, test_gguf_file};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    const H: usize = 8;
    const V: usize = 16;

    type Kv = (&'static str, u32, Vec<u8>);
    fn kv_u32(k: &'static str, v: u32) -> Kv {
        (k, 4, v.to_le_bytes().to_vec())
    }
    fn kv_f32(k: &'static str, v: f32) -> Kv {
        (k, 6, v.to_le_bytes().to_vec())
    }
    fn kv_str(k: &'static str, v: &str) -> Kv {
        let mut p = (v.len() as u64).to_le_bytes().to_vec();
        p.extend_from_slice(v.as_bytes());
        (k, 8, p)
    }
    fn kv_ints(k: &'static str, v: &[i32]) -> Kv {
        let mut p = 5u32.to_le_bytes().to_vec();
        p.extend_from_slice(&(v.len() as u64).to_le_bytes());
        v.iter().for_each(|x| p.extend_from_slice(&x.to_le_bytes()));
        (k, 9, p)
    }

    /// `(llama.cpp name, z-lab name, row-major shape)` of a 1-layer DSpark drafter
    /// (hidden 8, 2 heads x 4 dims, 1 kv head, ffn 8, Markov rank 4, 2 taps).
    fn layout() -> Vec<(&'static str, &'static str, Vec<usize>)> {
        vec![
            ("fc.weight", "fc.weight", vec![H, 2 * H]),
            ("enc.output_norm.weight", "hidden_norm.weight", vec![H]),
            ("output_norm.weight", "norm.weight", vec![H]),
            (
                "markov_w1.weight",
                "markov_head.markov_w1.weight",
                vec![V, 4],
            ),
            (
                "markov_w2.weight",
                "markov_head.markov_w2.weight",
                vec![V, 4],
            ),
            (
                "blk.0.attn_norm.weight",
                "layers.0.input_layernorm.weight",
                vec![H],
            ),
            (
                "blk.0.ffn_norm.weight",
                "layers.0.post_attention_layernorm.weight",
                vec![H],
            ),
            (
                "blk.0.attn_q.weight",
                "layers.0.self_attn.q_proj.weight",
                vec![8, H],
            ),
            (
                "blk.0.attn_k.weight",
                "layers.0.self_attn.k_proj.weight",
                vec![4, H],
            ),
            (
                "blk.0.attn_v.weight",
                "layers.0.self_attn.v_proj.weight",
                vec![4, H],
            ),
            (
                "blk.0.attn_output.weight",
                "layers.0.self_attn.o_proj.weight",
                vec![H, 8],
            ),
            (
                "blk.0.attn_q_norm.weight",
                "layers.0.self_attn.q_norm.weight",
                vec![4],
            ),
            (
                "blk.0.attn_k_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                vec![4],
            ),
            (
                "blk.0.ffn_gate.weight",
                "layers.0.mlp.gate_proj.weight",
                vec![8, H],
            ),
            (
                "blk.0.ffn_up.weight",
                "layers.0.mlp.up_proj.weight",
                vec![8, H],
            ),
            (
                "blk.0.ffn_down.weight",
                "layers.0.mlp.down_proj.weight",
                vec![H, 8],
            ),
        ]
    }

    /// Deterministic values in [-0.5, 0.5); norms near 1.
    fn values(name: &str, n: usize) -> Vec<f32> {
        let mut s = name.bytes().fold(0x9E37_79B9u32, |a, b| {
            a.wrapping_mul(31).wrapping_add(b as u32)
        });
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let v = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                if name.contains("norm") {
                    1.0 + 0.2 * v
                } else {
                    v
                }
            })
            .collect()
    }

    fn target() -> Weights {
        let shape = vec![V, H];
        Weights::from_f32(vec![("model.embed_tokens.weight".into(), shape)], |n| {
            Ok(values(n, V * H))
        })
        .unwrap()
    }

    fn run(d: &Arc<dyn Drafter>) -> (Vec<u32>, Vec<Vec<f32>>) {
        let hidden = values("hidden", 2 * 2 * H);
        let mut st = d.new_state();
        let mut rows = Vec::new();
        let out = d
            .propose(&mut st, &[1, 2, 3], &hidden, 3, &mut |l| {
                rows.push(l.to_vec());
                l.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            })
            .unwrap();
        (out, rows)
    }

    /// A llama.cpp `dflash` GGUF (DSpark: Markov head) maps onto the z-lab names, config and
    /// layouts [`load`] reads: the drafter built from it behaves exactly like the same weights
    /// stored under the safetensors names.
    #[test]
    fn gguf_sidecar_maps_to_the_safetensors_drafter() {
        let kv = [
            kv_str("general.architecture", "dflash"),
            kv_u32("dflash.block_count", 1),
            kv_u32("dflash.embedding_length", H as u32),
            kv_u32("dflash.feed_forward_length", 8),
            kv_u32("dflash.attention.head_count", 2),
            kv_u32("dflash.attention.head_count_kv", 1),
            kv_u32("dflash.attention.key_length", 4),
            kv_f32("dflash.attention.layer_norm_rms_epsilon", 1e-6),
            kv_f32("dflash.rope.freq_base", 10000.0),
            kv_u32("dflash.block_size", 4),
            kv_ints("dflash.target_layers", &[2, 3]),
            kv_u32("tokenizer.ggml.mask_token_id", 3),
        ];
        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = layout()
            .into_iter()
            .map(|(g, _, shape)| {
                let v = values(g, shape.iter().product());
                (g, shape.into_iter().rev().collect(), v)
            })
            .collect();
        let all: Vec<(&str, &[usize], &[f32])> = tensors
            .iter()
            .map(|(n, d, v)| (*n, &d[..], &v[..]))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dflash.gguf");
        std::fs::write(&path, test_gguf_file(&kv, &all)).unwrap();
        let (m, data) = open(&path).unwrap();
        let (ck, cfg) = gguf_checkpoint(&m, &data).unwrap();
        assert_eq!(cfg["architectures"][0], "DSparkDraftModel");
        assert_eq!(cfg["target_layer_ids"], serde_json::json!([1, 2]));
        assert_eq!(
            ck.shape("layers.0.self_attn.k_proj.weight").unwrap(),
            [4, H]
        );
        let tcfg = serde_json::json!({"num_hidden_layers": 4, "hidden_size": H});
        let from_gguf = load("gguf", &ck, &cfg, &target(), &tcfg).unwrap();

        let by_name: HashMap<&str, &str> = layout().iter().map(|(g, z, _)| (*z, *g)).collect();
        let ck = Weights::from_f32(
            layout()
                .into_iter()
                .map(|(_, z, s)| (z.to_string(), s))
                .collect(),
            |z| {
                Ok(values(
                    by_name[z],
                    layout()
                        .iter()
                        .find(|l| l.1 == z)
                        .unwrap()
                        .2
                        .iter()
                        .product(),
                ))
            },
        )
        .unwrap();
        let cfg = serde_json::json!({
            "architectures": ["DSparkDraftModel"], "hidden_size": H, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "intermediate_size": 8, "rms_norm_eps": 1e-6, "block_size": 4, "mask_token_id": 3,
            "rope_parameters": {"rope_theta": 10000.0}, "target_layer_ids": [1, 2],
            "sample_from_anchor": true,
        });
        let from_st = load("st", &ck, &cfg, &target(), &tcfg).unwrap();
        let (a, b) = (run(&from_gguf), run(&from_st));
        assert_eq!(a.0.len(), 3);
        assert_eq!(a, b);

        // Tensors the graph here does not run are refused by name, not skipped.
        assert!(drafter_tensor("blk.0.attn_post_norm.weight").is_err());
        assert!(drafter_tensor("token_embd.weight").is_err());
        assert_eq!(drafter_tensor("conf_proj.bias").unwrap(), None);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (n(a) * n(b))
    }

    /// satgeze/Qwen3.5-0.8B-DSpark: its llama.cpp GGUF (`Qwen3.5-0.8B-DSpark.gguf`) against the
    /// same checkpoint's `model.safetensors` (embed / lm_head dropped so both read the target's),
    /// on one GGUF target, same hidden states: identical weights, identical drafter logits.
    #[test]
    fn real_gguf_sidecar_matches_its_safetensors_twin() {
        let disk = Path::new(env!("CARGO_MANIFEST_DIR")).join("disk");
        let gg = disk
            .join("models--satgeze--Qwen3.5-0.8B-DSpark/snapshots/main/Qwen3.5-0.8B-DSpark.gguf");
        let twin = disk.join("models--twin--dspark08/snapshots/main");
        let target_dir = disk.join("models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/main");
        let target: Option<PathBuf> = std::fs::read_dir(&target_dir).ok().and_then(|d| {
            d.flatten()
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|e| e == "gguf"))
        });
        let (true, true, Some(target)) =
            (gg.exists(), twin.join("model.safetensors").exists(), target)
        else {
            println!(
                "SKIP: DSpark GGUF, safetensors twin or Qwen3.5-0.8B GGUF target not in disk/"
            );
            return;
        };
        let (tcfg, tw) = crate::model::qwen3_5::gguf_target(&target).unwrap();
        let (m, data) = open(&gg).unwrap();
        let (ck, cfg) = gguf_checkpoint(&m, &data).unwrap();
        let a = load("gguf", &ck, &cfg, &tw, &tcfg).unwrap();
        let tcfg_json: Value =
            serde_json::from_slice(&std::fs::read(twin.join("config.json")).unwrap()).unwrap();
        let st = Weights::open(&twin.join("model.safetensors")).unwrap();
        let b = load("twin", &st, &tcfg_json, &tw, &tcfg).unwrap();
        let (ha, hb) = (a.tap_layers(), b.tap_layers());
        assert_eq!(ha, hb);
        let rows = 5;
        let hidden: Vec<f32> = values("real-hidden", rows * ha.len() * 1024)
            .iter()
            .map(|v| v * 8.0)
            .collect();
        let ids: Vec<u32> = (0..=rows as u32).map(|i| 1000 + 37 * i).collect();
        let go = |d: &Arc<dyn Drafter>| {
            let mut st = d.new_state();
            let mut logits = Vec::new();
            let out = d
                .propose(&mut st, &ids, &hidden, 6, &mut |l| {
                    logits.push(l.to_vec());
                    l.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32
                })
                .unwrap();
            (out, logits)
        };
        let (x, y) = (go(&a), go(&b));
        let worst =
            x.1.iter()
                .zip(&y.1)
                .map(|(p, q)| cosine(p, q))
                .fold(1.0f64, f64::min);
        println!(
            "DSpark GGUF vs safetensors twin: tokens {:?} / {:?}, worst logit cosine {worst:.9}",
            x.0, y.0
        );
        assert_eq!(x.0, y.0);
        assert!(worst > 0.999999, "{worst}");
    }
}
