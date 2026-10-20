//! llama.cpp `gemma4` GGUFs (`general.architecture = gemma4`) as a [`Gemma4`]: the decoder's
//! tensors renamed to their transformers names and dequantized to bf16 (the per-layer embedding
//! table stays quantized in the mapped file, rows decoded on demand), the `config.json` equivalent
//! read from the `gemma4.*` metadata, and the `mmproj` GGUF beside it (arch `clip`, projectors
//! `gemma4v` / `gemma4a`) renamed the same way for the vision and audio towers.

use super::audio::Audio;
use super::text::{PleTable, TextModel};
use super::vision::Vision;
use super::{Encoder, Gemma4, NoEncoder};
use crate::model::hub::{local_mmproj, ModelArtifacts};
use crate::model::native::NativeLm;
use crate::model::weights::Weights;
use crate::model::{ModelManifest, UniversalModelLoader};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// `None` unless the artifacts' weights are a `gemma4` GGUF (the other loaders take the rest).
pub fn load_gguf(
    artifacts: &ModelArtifacts,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    let path = artifacts
        .weights_path
        .as_deref()
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))?;
    let (manifest, data) = crate::model::loaders::gguf::open(path).ok()?;
    let data = Arc::new(data);
    (manifest.metadata.get("general.architecture")? == "gemma4").then(|| {
        model(&data, &manifest, path, mode).map_err(|e| format!("{}: {e}", path.display()))
    })
}

/// Decode graph and weights of the text decoder of the `gemma4` GGUF at `path`, with its scaled token
/// embedding lookup: what a parity test needs to run the same graph on the CPU executor and Metal.
#[allow(clippy::type_complexity)]
pub fn gguf_decode_fixture(
    path: &Path,
) -> Result<
    (
        crate::graph::Graph,
        crate::graph::Weights,
        Box<dyn Fn(u32) -> Vec<f32>>,
    ),
    String,
> {
    let (manifest, data) = crate::model::loaders::gguf::open(path)?;
    let text = load_text(&Arc::new(data), &manifest)?;
    let (g, w) = text.to_decode_graph()?;
    let hidden = text.hidden;
    let embed = move |id: u32| {
        let mut row = vec![0.0f32; hidden];
        text.embed_row(id, &mut row);
        row
    };
    Ok((g, w, Box::new(embed)))
}

/// Metadata value(s) of `key`: one number, or a comma-joined array ([`UniversalModelLoader::parse_gguf`]).
fn nums(md: &HashMap<String, String>, key: &str) -> Option<Vec<f64>> {
    md.get(key)?
        .split(',')
        .map(|v| v.parse().ok())
        .collect::<Option<_>>()
}

fn num(md: &HashMap<String, String>, key: &str) -> Result<f64, String> {
    nums(md, key)
        .and_then(|v| v.first().copied())
        .ok_or_else(|| format!("GGUF metadata {key} missing"))
}

/// The text decoder of a `gemma4` GGUF.
pub(super) fn load_text(
    map: &Arc<crate::model::loaders::gguf::GgufData>,
    m: &ModelManifest,
) -> Result<Arc<TextModel>, String> {
    let (cfg, first_shared) = text_config(m, map)?;
    let ple = m
        .tensors
        .get("per_layer_token_embd.weight")
        .filter(|_| cfg["hidden_size_per_layer_input"].as_u64() > Some(0))
        .map(|t| ple_table(map, t))
        .transpose()?;
    let w = Weights::from_gguf(
        map,
        m,
        |name| text_hf_name(name, first_shared),
        |_| false,
        |_, _| {},
    )?;
    Ok(Arc::new(
        TextModel::load_with_ple(&w, &cfg, ple).map_err(|e| format!("gemma4 text: {e}"))?,
    ))
}

fn model(
    map: &Arc<crate::model::loaders::gguf::GgufData>,
    m: &ModelManifest,
    path: &Path,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let text = load_text(map, m)?;
    let mut tokenizer = UniversalModelLoader::extract_gguf_tokenizer(map)?;
    // llama.cpp ends generation on `<eos>` and `<turn|>` whichever one the file declares.
    let eos = num(&m.metadata, "tokenizer.ggml.eos_token_id")? as u32;
    let ends = ["<eos>", "<turn|>"].map(|s| tokenizer.vocab.get(s).copied());
    tokenizer.stop_ids = Some(
        std::iter::once(eos)
            .chain(ends.into_iter().flatten())
            .collect(),
    );
    let tid = |s: &str| {
        tokenizer
            .vocab
            .get(s)
            .copied()
            .ok_or_else(|| format!("GGUF vocabulary has no '{s}' token"))
    };
    let (image_token, audio_token) = (tid("<|image|>")?, tid("<|audio|>")?);
    let (boi, eoi, boa, eoa) = (
        tid("<|image>")?,
        tid("<image|>")?,
        tid("<|audio>")?,
        tid("<audio|>")?,
    );
    let mode = mode.into();
    let (vision, audio) = match local_mmproj(path) {
        Some(p) => projectors(&p, mode)?,
        None => {
            let msg = format!(
                "{} has no mmproj*.gguf beside it, so it takes text only",
                path.display()
            );
            (
                Box::new(NoEncoder(msg.clone())) as Box<dyn Encoder>,
                Box::new(NoEncoder(msg)) as Box<dyn Encoder>,
            )
        }
    };
    let program = super::compile_gemma4_program(&text, mode)?;
    Ok(Arc::new(Gemma4 {
        tokenizer,
        text,
        program,
        unified: false,
        vision,
        audio,
        image_token,
        audio_token,
        boi,
        eoi,
        boa,
        eoa,
    }))
}
/// `text_config` of the GGUF and the first layer that reuses an earlier layer's KV cache.
fn text_config(m: &ModelManifest, data: &[u8]) -> Result<(Value, usize), String> {
    let md = &m.metadata;
    let u = |k: &str| num(md, &format!("gemma4.{k}")).map(|v| v as usize);
    let layers = u("block_count")?;
    let sliding: Vec<bool> = nums(md, "gemma4.attention.sliding_window_pattern")
        .filter(|p| p.len() == layers)
        .ok_or("GGUF metadata gemma4.attention.sliding_window_pattern missing or short")?
        .into_iter()
        .map(|v| v != 0.0)
        .collect();
    let heads = nums(md, "gemma4.attention.head_count")
        .ok_or("GGUF metadata gemma4.attention.head_count missing")?;
    if heads.iter().any(|&h| h != heads[0]) {
        return Err("per-layer attention head counts are not supported".into());
    }
    let kv = nums(md, "gemma4.attention.head_count_kv")
        .filter(|k| k.len() == 1 || k.len() == layers)
        .ok_or("GGUF metadata gemma4.attention.head_count_kv missing or short")?;
    let kv_at = |i: usize| kv[i.min(kv.len() - 1)] as usize;
    let first = |swa: bool| sliding.iter().position(|&s| s == swa);
    let kv_swa = kv_at(first(true).unwrap_or(0));
    let kv_full = first(false).map_or(kv_swa, kv_at);
    if (0..layers).any(|i| kv_at(i) != if sliding[i] { kv_swa } else { kv_full }) {
        return Err("KV head counts vary within the sliding or the full-attention layers".into());
    }
    let hd_full = u("attention.key_length")?;
    let hd_swa = u("attention.key_length_swa").unwrap_or(hd_full);
    if u("attention.value_length").unwrap_or(hd_full) != hd_full
        || u("attention.value_length_swa").unwrap_or(hd_swa) != hd_swa
        || u("rope.dimension_count").unwrap_or(hd_full) != hd_full
        || u("rope.dimension_count_swa").unwrap_or(hd_swa) != hd_swa
    {
        // ponytail: the Gemma 4 models all rotate sliding layers fully and size values like keys.
        return Err(
            "partially rotated sliding layers or value heads of another width are not supported"
                .into(),
        );
    }
    let shared = u("attention.shared_kv_layers").unwrap_or(0).min(layers);
    let first_shared = layers - shared;
    // Full-attention layers without a value projection use their keys as values.
    let has_v: Vec<bool> = (0..first_shared)
        .filter(|&i| !sliding[i])
        .map(|i| m.tensors.contains_key(&format!("blk.{i}.attn_v.weight")))
        .collect();
    if has_v.iter().any(|&v| v) && has_v.iter().any(|&v| !v) {
        return Err("only some full-attention layers have a value projection".into());
    }
    // llama.cpp stores "proportional" RoPE as per-pair frequency factors: 1 for the rotated
    // pairs, 1e30 (never rotating) for the rest.
    let partial = match m.tensors.contains_key("rope_freqs.weight") {
        true => {
            let f = m.extract_tensor_f32(data, "rope_freqs.weight")?;
            let rotated = f.iter().take_while(|&&v| v == 1.0).count();
            if f.len() != hd_full / 2 || f[rotated..].iter().any(|&v| v < 1e20) {
                return Err("rope_freqs.weight is not a 1 / 1e30 partial-rotation mask".into());
            }
            2.0 * rotated as f64 / hd_full as f64
        }
        false if first(false).is_some() => {
            return Err("GGUF has full-attention layers but no rope_freqs.weight".into())
        }
        false => 1.0,
    };
    let mut c = json!({
        "hidden_size": u("embedding_length")?,
        "num_hidden_layers": layers,
        "num_attention_heads": heads[0] as usize,
        "num_key_value_heads": kv_swa,
        "num_global_key_value_heads": kv_full,
        "head_dim": hd_swa,
        "global_head_dim": hd_full,
        "hidden_size_per_layer_input": u("embedding_length_per_layer_input").unwrap_or(0),
        "num_kv_shared_layers": shared,
        "rms_norm_eps": num(md, "gemma4.attention.layer_norm_rms_epsilon")?,
        "layer_types": sliding
            .iter()
            .map(|&s| if s { "sliding_attention" } else { "full_attention" })
            .collect::<Vec<_>>(),
        "sliding_window": u("attention.sliding_window")?,
        "max_position_embeddings": u("context_length")?,
        "attention_k_eq_v": !has_v.is_empty() && !has_v[0],
        "pad_token_id": md.get("tokenizer.ggml.padding_token_id").and_then(|v| v.parse::<u64>().ok()).unwrap_or(0),
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": num(md, "gemma4.rope.freq_base_swa")?},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": partial,
                               "rope_theta": num(md, "gemma4.rope.freq_base")?},
        },
    });
    // Sparse layers (26B-A4B): the routed experts beside the dense MLP, in every layer that has a router.
    let sparse = (0..layers)
        .filter(|i| {
            m.tensors
                .contains_key(&format!("blk.{i}.ffn_gate_inp.weight"))
        })
        .count();
    if sparse != 0 {
        if sparse != layers {
            return Err(format!("{sparse} of {layers} layers have a router: mixed dense and sparse layers are not supported"));
        }
        c["enable_moe_block"] = true.into();
        c["num_experts"] = u("expert_count")?.into();
        c["top_k_experts"] = u("expert_used_count")?.into();
        c["moe_intermediate_size"] = u("expert_feed_forward_length")?.into();
    }
    if let Ok(cap) = num(md, "gemma4.final_logit_softcapping") {
        c["final_logit_softcapping"] = cap.into();
    }
    Ok((c, first_shared))
}

/// transformers name of GGUF decoder tensor `name` (`Ok(None)`: not loaded into the weights).
fn text_hf_name(name: &str, first_shared: usize) -> Result<Option<String>, String> {
    const P: &str = "model.language_model";
    let top = match name {
        "token_embd.weight" => "embed_tokens.weight",
        "per_layer_model_proj.weight" => "per_layer_model_projection.weight",
        "per_layer_proj_norm.weight" => "per_layer_projection_norm.weight",
        "output_norm.weight" => "norm.weight",
        // The table is read row by row ([`ple_table`]); the mask is folded into the config.
        "per_layer_token_embd.weight" | "rope_freqs.weight" => return Ok(None),
        "output.weight" => return Err("an untied output.weight is not supported".into()),
        _ => "",
    };
    if !top.is_empty() {
        return Ok(Some(format!("{P}.{top}")));
    }
    let unknown = || format!("GGUF tensor '{name}' is not part of the gemma4 graph");
    let (layer, rest) = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .ok_or_else(unknown)?;
    let hf = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        // KV-shared layers attend to an earlier layer's cache: their own k/v are never read.
        "attn_k.weight" | "attn_v.weight" | "attn_k_norm.weight"
            if layer.parse::<usize>().is_ok_and(|l| l >= first_shared) =>
        {
            return Ok(None)
        }
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
        "ffn_gate_inp.weight" => "router.proj.weight",
        "ffn_gate_inp.scale" => "router.scale",
        "ffn_down_exps.scale" => "router.per_expert_scale",
        "pre_ffw_norm_2.weight" => "pre_feedforward_layernorm_2.weight",
        "post_ffw_norm_1.weight" => "post_feedforward_layernorm_1.weight",
        "post_ffw_norm_2.weight" => "post_feedforward_layernorm_2.weight",
        // Expert banks take no `.weight`: `Weights::from_gguf` keeps `.mlp.experts.*` mapped.
        "ffn_gate_exps.weight" => "mlp.experts.gate_proj",
        "ffn_up_exps.weight" => "mlp.experts.up_proj",
        "ffn_gate_up_exps.weight" => "mlp.experts.gate_up_proj",
        "ffn_down_exps.weight" => "mlp.experts.down_proj",
        "inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" => "post_per_layer_input_norm.weight",
        "layer_output_scale.weight" => "layer_scalar",
        _ => return Err(unknown()),
    };
    Ok(Some(format!("{P}.layers.{layer}.{hf}")))
}

/// The per-layer embedding table `t` (`[vocab, layers * dim]`) as rows decoded from the mapping.
fn ple_table(
    map: &Arc<crate::model::loaders::gguf::GgufData>,
    t: &crate::model::VirtualTensor,
) -> Result<PleTable, String> {
    let &[cols, rows] = t.shape.as_slice() else {
        return Err("per_layer_token_embd.weight is not a matrix".into());
    };
    let (map, row_bytes, format, dtype, base) = (
        map.clone(),
        t.byte_len / rows,
        t.quant_format,
        t.dtype.clone(),
        t.byte_offset,
    );
    let row = move |id: usize| -> Result<Vec<f32>, String> {
        let raw = &map[base + id * row_bytes..][..row_bytes];
        match crate::model::layers::dequantize_gguf_blocks(format, raw) {
            Some(v) => Ok(v),
            None => crate::model::decode_dtype(raw, &dtype),
        }
    };
    row(0).map_err(|e| format!("per_layer_token_embd.weight: {e}"))?;
    Ok(PleTable::lazy(rows, cols, move |id, out| {
        // Decodable: the first row was.
        out.copy_from_slice(&row(id).expect("per-layer embedding row")[..cols]);
    }))
}

/// transformers name of an `mmproj` tensor of the `gemma4v` / `gemma4a` projectors (`None`: not
/// part of either graph).
fn mm_hf_name(name: &str) -> Option<String> {
    match name {
        "v.patch_embd.weight" => {
            return Some("model.vision_tower.patch_embedder.input_proj.weight".into())
        }
        "v.position_embd.weight" => {
            return Some("model.vision_tower.patch_embedder.position_embedding_table".into())
        }
        "mm.input_projection.weight" => {
            return Some("model.embed_vision.embedding_projection.weight".into())
        }
        "a.input_projection.weight" => {
            return Some(
                "model.audio_tower.subsample_conv_projection.input_proj_linear.weight".into(),
            )
        }
        "a.pre_encode.out.weight" => return Some("model.audio_tower.output_proj.weight".into()),
        "a.pre_encode.out.bias" => return Some("model.audio_tower.output_proj.bias".into()),
        "mm.a.input_projection.weight" => {
            return Some("model.embed_audio.embedding_projection.weight".into())
        }
        _ => {}
    }
    if let Some(r) = name.strip_prefix("a.conv1d.") {
        let (i, rest) = r.split_once('.')?;
        return match rest {
            "weight" => Some(format!(
                "model.audio_tower.subsample_conv_projection.layer{i}.conv.weight"
            )),
            "norm.weight" => Some(format!(
                "model.audio_tower.subsample_conv_projection.layer{i}.norm.weight"
            )),
            _ => None,
        };
    }
    let (tower, r) = match name.split_once(".blk.")? {
        ("v", r) => ("model.vision_tower.encoder", r),
        ("a", r) => ("model.audio_tower", r),
        _ => return None,
    };
    let (layer, rest) = r.split_once('.')?;
    let (stem, suffix) = rest.rsplit_once('.')?;
    let vision = tower.contains("vision");
    // ClippableLinear: weight under `.linear`, clamps beside it.
    let linear = match (vision, stem) {
        (true, "attn_q") => "self_attn.q_proj",
        (true, "attn_k") => "self_attn.k_proj",
        (true, "attn_v") => "self_attn.v_proj",
        (true, "attn_out") => "self_attn.o_proj",
        (true, "ffn_gate") => "mlp.gate_proj",
        (true, "ffn_up") => "mlp.up_proj",
        (true, "ffn_down") => "mlp.down_proj",
        (false, "attn_q") => "self_attn.q_proj",
        (false, "attn_k") => "self_attn.k_proj",
        (false, "attn_v") => "self_attn.v_proj",
        (false, "attn_out") => "self_attn.post",
        (false, "ffn_up") => "feed_forward1.ffw_layer_1",
        (false, "ffn_down") => "feed_forward1.ffw_layer_2",
        (false, "ffn_up_1") => "feed_forward2.ffw_layer_1",
        (false, "ffn_down_1") => "feed_forward2.ffw_layer_2",
        (false, "conv_pw1") => "lconv1d.linear_start",
        (false, "conv_pw2") => "lconv1d.linear_end",
        _ => "",
    };
    if !linear.is_empty() {
        return match suffix {
            "weight" => Some(format!("{tower}.layers.{layer}.{linear}.linear.weight")),
            "input_min" | "input_max" | "output_min" | "output_max" => {
                Some(format!("{tower}.layers.{layer}.{linear}.{suffix}"))
            }
            _ => None,
        };
    }
    let plain = match (vision, stem) {
        (true, "attn_q_norm") => "self_attn.q_norm",
        (true, "attn_k_norm") => "self_attn.k_norm",
        (true, "ln1") => "input_layernorm",
        (true, "attn_post_norm") => "post_attention_layernorm",
        (true, "ln2") => "pre_feedforward_layernorm",
        (true, "ffn_post_norm") => "post_feedforward_layernorm",
        (false, "ffn_norm") => "feed_forward1.pre_layer_norm",
        (false, "ffn_post_norm") => "feed_forward1.post_layer_norm",
        (false, "ffn_norm_1") => "feed_forward2.pre_layer_norm",
        (false, "ffn_post_norm_1") => "feed_forward2.post_layer_norm",
        (false, "attn_pre_norm") => "norm_pre_attn",
        (false, "attn_post_norm") => "norm_post_attn",
        (false, "ln2") => "norm_out",
        (false, "norm_conv") => "lconv1d.conv_norm",
        (false, "conv_norm") => "lconv1d.pre_layer_norm",
        (false, "conv_dw") => "lconv1d.depthwise_conv1d",
        (false, "attn_k_rel") => "self_attn.relative_k_proj",
        // Stored softplus'ed; the audio tower applies softplus itself ([`mm_from_gguf`]).
        (false, "per_dim_scale") if suffix == "weight" => {
            return Some(format!("{tower}.layers.{layer}.self_attn.per_dim_scale"))
        }
        _ => return None,
    };
    (suffix == "weight").then(|| format!("{tower}.layers.{layer}.{plain}.weight"))
}

/// Undoes llama.cpp's `mmproj` conversion of tensor `name` (`shape`: its GGUF dims reversed) in
/// place.
fn mm_from_gguf(name: &str, shape: &[usize], v: &mut [f32]) {
    if name == "v.patch_embd.weight" {
        // [out, c, y, x] conv kernel -> the patch embedder's [out, (y, x, c)] linear.
        let (o, p) = (shape[0], shape[2]);
        let src = v.to_vec();
        for oi in 0..o {
            for c in 0..3 {
                for yx in 0..p * p {
                    v[oi * p * p * 3 + yx * 3 + c] = src[(oi * 3 + c) * p * p + yx];
                }
            }
        }
    } else if name.ends_with(".per_dim_scale.weight") {
        // Stored softplus'ed (the audio tower applies softplus to the raw parameter itself).
        v.iter_mut()
            .for_each(|x| *x = if *x > 20.0 { *x } else { x.exp_m1().ln() });
    }
}

/// The vision and audio encoders of the `mmproj` GGUF at `path`. A modality the file does not
/// carry, or whose projector is not implemented, becomes an encoder that says so when a prompt
/// attaches it.
fn projectors(
    path: &Path,
    mode: crate::hardware::PlacementMode,
) -> Result<(Box<dyn Encoder>, Box<dyn Encoder>), String> {
    let gpu = crate::model::vision_graph::is_gpu(mode);
    let at = |e: String| format!("{}: {e}", path.display());
    let file = std::fs::File::open(path).map_err(|e| at(e.to_string()))?;
    // SAFETY: as in [`load_gguf`].
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| at(e.to_string()))?;
    let m = UniversalModelLoader::parse_gguf(&map).map_err(at)?;
    let md = &m.metadata;
    if md.get("general.architecture").map(String::as_str) != Some("clip") {
        return Err(at("not an mmproj (architecture 'clip') GGUF".into()));
    }
    // Ok when the file carries `modality` with the `want` projector, else why not.
    let kind = |modality: &str, want: &str| -> Result<(), String> {
        let has = md
            .get(&format!("clip.has_{modality}_encoder"))
            .map(String::as_str);
        match (has, md.get(&format!("clip.{modality}.projector_type"))) {
            (Some("1"), Some(t)) if t == want => Ok(()),
            (Some("1"), t) => Err(at(format!(
                "{modality} projector '{}' is not implemented (only '{want}'; the encoder-free gemma4uv / gemma4ua of the 12B models are not)",
                t.map_or("?", String::as_str)
            ))),
            _ => Err(at(format!("no {modality} encoder"))),
        }
    };
    let (mut vision, mut audio) = (kind("vision", "gemma4v"), kind("audio", "gemma4a"));
    // A tensor this loader has no graph for (e.g. the 26B-A4B tower's `v.std_bias`) disables its
    // modality, with the reason shown when a prompt attaches it; it never blocks the text model.
    for name in m.tensors.keys().filter(|n| mm_hf_name(n).is_none()) {
        let why = at(format!(
            "tensor '{name}' is not part of the gemma4v / gemma4a graph"
        ));
        if name.starts_with("v.") || name.starts_with("mm.input") {
            vision = vision.and(Err(why));
        } else {
            audio = audio.and(Err(why));
        }
    }
    let (mut specs, mut names) = (Vec::new(), HashMap::new());
    for (name, t) in &m.tensors {
        let is_vision = name.starts_with("v.") || name.starts_with("mm.input");
        if (is_vision && vision.is_err()) || (!is_vision && audio.is_err()) {
            continue;
        }
        let hf = mm_hf_name(name).expect("unmapped tensors disabled their modality above");
        let mut shape: Vec<usize> = t.shape.iter().rev().copied().collect();
        if name == "v.patch_embd.weight" {
            shape = vec![shape[0], shape[1] * shape[2] * shape[3]];
        }
        specs.push((hf.clone(), shape));
        names.insert(hf, name.clone());
    }
    let w = Weights::from_f32(specs, |hf| {
        let name = &names[hf];
        let mut v = m.extract_tensor_f32(&map, name)?;
        let shape: Vec<usize> = m.tensors[name].shape.iter().rev().copied().collect();
        mm_from_gguf(name, &shape, &mut v);
        Ok(v)
    })?;
    let n = |k: &str| num(md, k).map_err(at);
    let vision_tower = || -> Result<Box<dyn Encoder>, String> {
        let heads = n("clip.vision.attention.head_count")? as usize;
        let vc = json!({
            "hidden_size": n("clip.vision.embedding_length")? as usize,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "head_dim": m.tensors["v.blk.0.attn_q_norm.weight"].shape[0],
            "num_hidden_layers": n("clip.vision.block_count")? as usize,
            "patch_size": n("clip.vision.patch_size")? as usize,
            "pooling_kernel_size": n("clip.vision.projector.scale_factor").unwrap_or(3.0) as usize,
            "rms_norm_eps": n("clip.vision.attention.layer_norm_epsilon")?,
            // llama.cpp hard-codes the vision RoPE base (clip.cpp, gemma4v).
            "rope_parameters": {"rope_theta": 100.0},
        });
        // ponytail: the HF processor's soft-token budget; llama.cpp's 70..1120 range is its own
        // preprocessing choice.
        let processor = json!({"image_processor": {"max_soft_tokens": 280}});
        let v = Vision::load(&w, &vc, &processor).map_err(|e| at(format!("gemma4 vision: {e}")))?;
        Ok(Box::new(if gpu { v.on_gpu(mode) } else { v }))
    };
    let audio_tower = || -> Result<Box<dyn Encoder>, String> {
        let ac = json!({
            "hidden_size": n("clip.audio.embedding_length")? as usize,
            "num_attention_heads": n("clip.audio.attention.head_count")? as usize,
            "num_hidden_layers": n("clip.audio.block_count")? as usize,
            "conv_kernel_size": m.tensors["a.blk.0.conv_dw.weight"].shape[0],
            // llama.cpp hard-codes the chunked local attention (12 frames, 12 past), logit cap 50,
            // half-step residuals and the 1e-6 norm epsilon (gemma4a.cpp).
            "attention_context_left": 13,
            "attention_context_right": 0,
            "attention_logit_cap": 50.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 1e-6,
        });
        // Gemma4AudioFeatureExtractor as every gemma-4 checkpoint's processor_config.json sets it.
        let processor = json!({"feature_extractor": {
            "feature_size": 128, "sampling_rate": 16000, "frame_length": 320, "hop_length": 160,
            "fft_length": 512, "min_frequency": 0.0, "max_frequency": 8000.0, "mel_floor": 0.001}});
        let a = Audio::load(&w, &ac, &processor).map_err(|e| at(format!("gemma4 audio: {e}")))?;
        Ok(Box::new(if gpu { a.on_gpu(mode) } else { a }))
    };
    let encoder =
        |has: Result<(), String>, load: &dyn Fn() -> Result<Box<dyn Encoder>, String>| match has {
            Ok(()) => load(),
            Err(why) => Ok(Box::new(NoEncoder(why)) as Box<dyn Encoder>),
        };
    Ok((
        encoder(vision, &vision_tower)?,
        encoder(audio, &audio_tower)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::QuantFormat;
    use crate::model::{SourceFormat, VirtualTensor};

    fn tensor(name: &str, shape: &[usize], offset: usize, bytes: usize) -> (String, VirtualTensor) {
        let t = VirtualTensor {
            name: name.into(),
            shape: shape.to_vec(),
            quant_format: QuantFormat::FP32,
            byte_offset: offset,
            byte_len: bytes,
            dtype: "F32".into(),
            shard_index: 0,
        };
        (name.into(), t)
    }

    /// A 4-layer Gemma 4 header: sliding/full pattern `1,0,1,0`, 2 shared-KV layers, per-type
    /// head counts as arrays, and a `rope_freqs` mask rotating 4 of 16 pairs.
    fn manifest() -> (ModelManifest, Vec<u8>) {
        let md: HashMap<String, String> = [
            ("block_count", "4"),
            ("embedding_length", "64"),
            ("attention.head_count", "4"),
            ("attention.head_count_kv", "2,1,2,1"),
            ("attention.key_length", "32"),
            ("attention.key_length_swa", "16"),
            ("rope.dimension_count", "32"),
            ("rope.dimension_count_swa", "16"),
            ("attention.shared_kv_layers", "2"),
            ("attention.sliding_window_pattern", "1,0,1,0"),
            ("embedding_length_per_layer_input", "8"),
            ("attention.sliding_window", "8"),
            ("context_length", "128"),
            ("attention.layer_norm_rms_epsilon", "0.000001"),
            ("rope.freq_base", "1000000"),
            ("rope.freq_base_swa", "10000"),
            ("final_logit_softcapping", "30"),
        ]
        .into_iter()
        .map(|(k, v)| (format!("gemma4.{k}"), v.to_string()))
        .collect();
        let mask: Vec<u8> = (0..16)
            .flat_map(|i| (if i < 4 { 1.0f32 } else { 1e30 }).to_le_bytes())
            .collect();
        let m = ModelManifest {
            format: SourceFormat::Gguf,
            tensors: [tensor("rope_freqs.weight", &[16], 0, 64)]
                .into_iter()
                .collect(),
            metadata: md,
            ..Default::default()
        };
        (m, mask)
    }

    /// A 2-layer 26B-A4B-style GGUF (sliding + full layer, dense MLP beside 8 experts, fused Q4_0
    /// gate/up bank, Q4_0 down bank, F32 router and per-expert scale): the sparse layers through the
    /// real GGUF path, Metal program against the CPU graph executor.
    #[test]
    fn sparse_gguf_fused_banks_stay_packed_and_match_cpu_graph() {
        use crate::model::layers::quantize_f32_to_q4_0;
        let (h, vocab, nh, hd, inter) = (64usize, 128usize, 2usize, 32usize, 48usize);
        let (experts, top_k, ff) = (8usize, 2usize, 32usize);
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };
        let f32s = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        let mut put = |n: String, dims: &[usize], ty: u32, bytes: Vec<u8>| {
            tensors.push((n, dims.to_vec(), ty, bytes))
        };
        put(
            "token_embd.weight".into(),
            &[h, vocab],
            0,
            f32s(rnd(h * vocab, 0.8)),
        );
        put(
            "output_norm.weight".into(),
            &[h],
            0,
            f32s(rnd(h, 0.2).iter().map(|v| v + 1.0).collect()),
        );
        put(
            "rope_freqs.weight".into(),
            &[hd / 2],
            0,
            f32s(vec![1.0; hd / 2]),
        );
        for l in 0..2 {
            let p = format!("blk.{l}");
            let gain = |n: usize, r: &mut dyn FnMut(usize, f32) -> Vec<f32>| {
                f32s(r(n, 0.2).iter().map(|v| v + 1.0).collect())
            };
            for n in [
                "attn_norm",
                "post_attention_norm",
                "ffn_norm",
                "post_ffw_norm",
                "pre_ffw_norm_2",
                "post_ffw_norm_1",
                "post_ffw_norm_2",
            ] {
                put(format!("{p}.{n}.weight"), &[h], 0, gain(h, &mut rnd));
            }
            for n in ["attn_q_norm", "attn_k_norm"] {
                put(format!("{p}.{n}.weight"), &[hd], 0, gain(hd, &mut rnd));
            }
            for (n, dims) in [
                ("attn_q", [h, nh * hd]),
                ("attn_k", [h, nh * hd]),
                ("attn_v", [h, nh * hd]),
                ("attn_output", [nh * hd, h]),
                ("ffn_gate", [h, inter]),
                ("ffn_up", [h, inter]),
                ("ffn_down", [inter, h]),
            ] {
                put(
                    format!("{p}.{n}.weight"),
                    &dims,
                    0,
                    f32s(rnd(dims[0] * dims[1], 0.8)),
                );
            }
            put(
                format!("{p}.layer_output_scale.weight"),
                &[1],
                0,
                f32s(vec![0.9]),
            );
            put(
                format!("{p}.ffn_gate_inp.weight"),
                &[h, experts],
                0,
                f32s(rnd(h * experts, 1.6)),
            );
            put(
                format!("{p}.ffn_gate_inp.scale"),
                &[h],
                0,
                gain(h, &mut rnd),
            );
            put(
                format!("{p}.ffn_down_exps.scale"),
                &[experts],
                0,
                gain(experts, &mut rnd),
            );
            put(
                format!("{p}.ffn_gate_up_exps.weight"),
                &[h, 2 * ff, experts],
                2,
                quantize_f32_to_q4_0(&rnd(experts * 2 * ff * h, 0.8)),
            );
            put(
                format!("{p}.ffn_down_exps.weight"),
                &[ff, h, experts],
                2,
                quantize_f32_to_q4_0(&rnd(experts * h * ff, 0.8)),
            );
        }
        let u32_kv = |k: &str, v: u32| (format!("gemma4.{k}"), 4u32, v.to_le_bytes().to_vec());
        let f32_kv = |k: &str, v: f32| (format!("gemma4.{k}"), 6u32, v.to_le_bytes().to_vec());
        let mut arch = (6u64).to_le_bytes().to_vec();
        arch.extend_from_slice(b"gemma4");
        // `attention.sliding_window_pattern`: an int32 array, layer 0 sliding and layer 1 full.
        let mut pattern = 5u32.to_le_bytes().to_vec();
        pattern.extend_from_slice(&2u64.to_le_bytes());
        pattern.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let kvs = vec![
            ("general.architecture".to_string(), 8u32, arch),
            u32_kv("block_count", 2),
            u32_kv("embedding_length", h as u32),
            u32_kv("attention.head_count", nh as u32),
            u32_kv("attention.head_count_kv", nh as u32),
            u32_kv("attention.key_length", hd as u32),
            u32_kv("attention.sliding_window", 8),
            (
                "gemma4.attention.sliding_window_pattern".to_string(),
                9u32,
                pattern,
            ),
            u32_kv("context_length", 64),
            f32_kv("attention.layer_norm_rms_epsilon", 1e-6),
            f32_kv("rope.freq_base", 10000.0),
            f32_kv("rope.freq_base_swa", 10000.0),
            f32_kv("final_logit_softcapping", 30.0),
            u32_kv("expert_count", experts as u32),
            u32_kv("expert_used_count", top_k as u32),
            u32_kv("expert_feed_forward_length", ff as u32),
        ];
        let file = crate::model::loaders::gguf::test_gguf_blocks(&kvs, &tensors);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny-gemma4-moe.gguf");
        std::fs::write(&path, &file).unwrap();

        let (g, w, embed) = gguf_decode_fixture(&path).unwrap();
        crate::graph::shape::assert_shapes(&g);
        assert!(
            g.values
                .iter()
                .any(|v| v.name.ends_with("experts.gate_up_proj.weight")),
            "the fused gate/up bank is one MoeLinear weight"
        );
        let mut prog = crate::graph::program::GraphProgram::compile(
            g.clone(),
            w.clone(),
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
        )
        .expect("Metal program of the sparse GGUF");
        assert_eq!(prog.cpu_node_count(), 0);
        let mut cpu_state = crate::graph::StateStore::new();
        let mut tok = 5u32;
        for pos in 0..6usize {
            let emb = embed(tok);
            let inputs = || {
                vec![
                    crate::graph::Tensor::F32(emb.clone()),
                    crate::graph::Tensor::U32(vec![pos as u32]),
                ]
            };
            let want = crate::graph::cpu::run(&g, &w, &mut cpu_state, inputs()).unwrap();
            let want = want[0].as_f32().unwrap();
            let got = prog.run(inputs()).unwrap();
            let got = got[0].as_f32().unwrap();
            let dot: f64 = want
                .iter()
                .zip(got)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let n = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (n(want) * n(got));
            assert!(cos >= 0.999, "step {pos}: Metal vs CPU graph cosine {cos}");
            tok = want
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
        }
    }

    #[test]
    fn config_follows_the_gguf_metadata() {
        let (mut m, data) = manifest();
        let (c, first_shared) = text_config(&m, &data).unwrap();
        assert_eq!(first_shared, 2);
        assert_eq!(
            c["layer_types"],
            serde_json::json!([
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ])
        );
        assert_eq!(
            (
                c["num_key_value_heads"].as_u64(),
                c["num_global_key_value_heads"].as_u64()
            ),
            (Some(2), Some(1))
        );
        assert_eq!(
            (c["head_dim"].as_u64(), c["global_head_dim"].as_u64()),
            (Some(16), Some(32))
        );
        assert_eq!(
            c["rope_parameters"]["full_attention"]["partial_rotary_factor"],
            0.25
        );
        // Full layer 1 owns its KV cache but has no value projection: values are the keys.
        assert_eq!(c["attention_k_eq_v"], true);
        m.tensors
            .extend([tensor("blk.1.attn_v.weight", &[32, 64], 0, 0)]);
        assert_eq!(text_config(&m, &data).unwrap().0["attention_k_eq_v"], false);
        // A router in some layers only is refused; in every layer it makes the block sparse.
        m.tensors
            .extend([tensor("blk.0.ffn_gate_inp.weight", &[4, 64], 0, 0)]);
        assert!(text_config(&m, &data)
            .unwrap_err()
            .contains("mixed dense and sparse"));
        for l in 1..4 {
            m.tensors.extend([tensor(
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[4, 64],
                0,
                0,
            )]);
        }
        m.metadata.extend(
            [
                ("expert_count", "128"),
                ("expert_used_count", "8"),
                ("expert_feed_forward_length", "704"),
            ]
            .map(|(k, v)| (format!("gemma4.{k}"), v.to_string())),
        );
        let c = text_config(&m, &data).unwrap().0;
        assert_eq!(c["enable_moe_block"], true);
        assert_eq!(
            (
                c["num_experts"].as_u64(),
                c["top_k_experts"].as_u64(),
                c["moe_intermediate_size"].as_u64()
            ),
            (Some(128), Some(8), Some(704))
        );
    }

    #[test]
    fn tensor_names_map_to_transformers() {
        let t = |n: &str| text_hf_name(n, 2).unwrap();
        assert_eq!(
            t("blk.1.post_norm.weight").unwrap(),
            "model.language_model.layers.1.post_per_layer_input_norm.weight"
        );
        assert_eq!(
            t("blk.1.layer_output_scale.weight").unwrap(),
            "model.language_model.layers.1.layer_scalar"
        );
        // KV-shared layers never read their own k/v; the table and the RoPE mask are not weights.
        assert_eq!(
            (
                t("blk.2.attn_k.weight"),
                t("rope_freqs.weight"),
                t("per_layer_token_embd.weight")
            ),
            (None, None, None)
        );
        // Sparse layers: router, norms, per-expert scale; banks carry no `.weight` so they stay mapped.
        for (gguf, hf) in [
            ("ffn_gate_inp.weight", "router.proj.weight"),
            ("ffn_gate_inp.scale", "router.scale"),
            ("ffn_down_exps.scale", "router.per_expert_scale"),
            (
                "pre_ffw_norm_2.weight",
                "pre_feedforward_layernorm_2.weight",
            ),
            (
                "post_ffw_norm_1.weight",
                "post_feedforward_layernorm_1.weight",
            ),
            ("ffn_gate_exps.weight", "mlp.experts.gate_proj"),
            ("ffn_gate_up_exps.weight", "mlp.experts.gate_up_proj"),
            ("ffn_down_exps.weight", "mlp.experts.down_proj"),
        ] {
            assert_eq!(
                t(&format!("blk.1.{gguf}")).unwrap(),
                format!("model.language_model.layers.1.{hf}")
            );
        }
        assert!(text_hf_name("blk.0.ssm_a", 2).is_err());
        let m = |n: &str| mm_hf_name(n).unwrap_or_default();
        assert_eq!(
            m("v.blk.3.attn_q.input_max"),
            "model.vision_tower.encoder.layers.3.self_attn.q_proj.input_max"
        );
        assert_eq!(
            m("v.blk.1.ln2.weight"),
            "model.vision_tower.encoder.layers.1.pre_feedforward_layernorm.weight"
        );
        assert_eq!(
            m("a.blk.2.ffn_down_1.weight"),
            "model.audio_tower.layers.2.feed_forward2.ffw_layer_2.linear.weight"
        );
        assert_eq!(
            m("a.blk.0.per_dim_scale.weight"),
            "model.audio_tower.layers.0.self_attn.per_dim_scale"
        );
        assert_eq!(
            m("a.conv1d.1.norm.weight"),
            "model.audio_tower.subsample_conv_projection.layer1.norm.weight"
        );
        assert_eq!(mm_hf_name("v.std_bias"), None);
    }

    /// The converter's conv-kernel and softplus rewrites are undone exactly.
    #[test]
    fn mmproj_conversions_are_inverted() {
        // 1 output, 3 channels, 2x2 patch: GGUF [c][y][x] -> HF [(y, x, c)].
        let mut v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        mm_from_gguf("v.patch_embd.weight", &[1, 3, 2, 2], &mut v);
        assert_eq!(
            v,
            [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
        );
        let raw = [-0.7f32, 0.0, 2.5];
        let mut v: Vec<f32> = raw.iter().map(|x| (1.0 + x.exp()).ln()).collect();
        mm_from_gguf("a.blk.0.per_dim_scale.weight", &[3], &mut v);
        assert!(
            v.iter().zip(raw).all(|(a, b)| (a - b).abs() < 1e-5),
            "{v:?}"
        );
    }

    /// Finds the first node where the Metal session departs from the CPU executor on the real Gemma 4
    /// checkpoint (`GIBBS_GEMMA_GGUF`, default E2B QAT) (every node output is exposed as a graph output). Run with `--ignored --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs the real E2B QAT checkpoint under disk/"]
    fn gemma4_real_metal_cpu_bisect() {
        use crate::graph::{StateStore, Tensor};
        let default = "disk/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/main/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf".to_string();
        let path = std::env::var("GIBBS_GEMMA_GGUF").unwrap_or(default);
        let path = std::path::Path::new(&path);
        if !path.exists() {
            eprintln!("SKIP: {path:?} not found");
            return;
        }
        let (manifest, data) = crate::model::loaders::gguf::open(path).unwrap();
        let data = Arc::new(data);
        let text = load_text(&data, &manifest).unwrap();
        let (mut g, w) = text.to_decode_graph().unwrap();
        crate::graph::passes::canonical::run(&mut g);
        for n in 0..g.nodes.len() {
            let o = g.nodes[n].outputs[0];
            if !g.outputs.contains(&o) {
                g.outputs.push(o);
            }
        }
        let tokens = [
            2u32, 105, 2364, 3041, 563, 506, 5279, 1855, 529, 5946, 38102, 236764, 710, 2364,
        ];
        let build = |tok: u32, pos: usize| {
            let mut emb = vec![0.0f32; text.hidden];
            text.embed_row(tok, &mut emb);
            let mut inputs = vec![Tensor::F32(emb), Tensor::U32(vec![pos as u32])];
            if let Some(p) = &text.ple {
                let mut e = vec![0.0f32; text.layers_n * p.dim];
                let id = if (tok as usize) < p.embed.rows {
                    tok
                } else {
                    text.pad_id
                };
                p.embed.get_row(id as usize, &mut e);
                let sc = (p.dim as f32).sqrt();
                e.iter_mut().for_each(|v| *v *= sc);
                inputs.push(Tensor::F32(e));
            }
            inputs
        };
        let mut cpu_state = StateStore::new();
        let mut native_cache = text.new_cache();
        let mut mg = g.clone();
        crate::graph::passes::dtype_adapt::retype_uncoded_weights(&mut mg, &|f| {
            crate::graph::metal::supports_op(
                &crate::graph::Op::Linear,
                crate::graph::DType::Quant(f),
            )
        })
        .unwrap();
        let rt = crate::graph::metal::Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir())
            .unwrap();
        let session = crate::graph::metal::Session::new_for_graph(rt, &w, Some(&mg)).unwrap();
        let plan = session.prepare_plan(&mg).unwrap();
        let mut metal_state = StateStore::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            let cpu = crate::graph::cpu::run(&g, &w, &mut cpu_state, build(tok, pos)).unwrap();
            let native_logits;
            {
                // Native reference: the hand-written forward the CLI's CPU path uses.
                let mut emb = vec![0.0f32; text.hidden];
                text.embed_row(tok, &mut emb);
                let native = text.forward(emb, &[tok], pos, &mut native_cache, &[]);
                let gl = cpu[0].as_f32().unwrap();
                let dot: f32 = native.iter().zip(gl).map(|(a, b)| a * b).sum();
                let na: f32 = native.iter().map(|a| a * a).sum::<f32>().sqrt();
                let nb: f32 = gl.iter().map(|a| a * a).sum::<f32>().sqrt();
                println!(
                    "pos {pos}: native vs graph-cpu logits cos {:.5}",
                    dot / (na * nb)
                );
                native_logits = native;
            }
            let met = session
                .run_prepared(&mg, &plan, Some(&mut metal_state), build(tok, pos))
                .unwrap();
            {
                let ml = met[0].as_f32().unwrap();
                let dot: f32 = native_logits.iter().zip(ml).map(|(a, b)| a * b).sum();
                let na: f32 = native_logits.iter().map(|a| a * a).sum::<f32>().sqrt();
                let nb: f32 = ml.iter().map(|a| a * a).sum::<f32>().sqrt();
                println!(
                    "pos {pos}: native vs metal logits cos {:.5}",
                    dot / (na * nb)
                );
            }
            let mut first_bad = None;
            for (i, (c, m)) in cpu.iter().zip(&met).enumerate() {
                if i == 0 {
                    continue; // the logits, compared via the node list below
                }
                let (Ok(c), Ok(m)) = (c.as_f32(), m.as_f32()) else {
                    continue;
                };
                let scale = c.iter().fold(0f32, |a, &x| a.max(x.abs())).max(1e-6);
                let d = c
                    .iter()
                    .zip(m)
                    .fold(0f32, |a, (&x, &y)| a.max((x - y).abs()))
                    / scale;
                if d > 2e-3 && first_bad.is_none() {
                    first_bad = Some((i, d));
                }
            }
            match first_bad {
                Some((i, d)) => {
                    let n = &g.nodes[(i - 1).min(g.nodes.len() - 1)];
                    println!(
                        "pos {pos}: first divergence output #{i} (node '{}' {:?}) rel diff {d:.4}",
                        n.name, n.op
                    );
                    if let (Ok(c), Ok(m)) = (cpu[i].as_f32(), met[i].as_f32()) {
                        let bad: Vec<usize> = (0..c.len())
                            .filter(|&j| !((c[j] - m[j]).abs() <= 1e-2 * (1.0 + c[j].abs())))
                            .collect();
                        println!(
                            "  {} of {} elements differ; first {:?}; e.g. cpu {:?} metal {:?}",
                            bad.len(),
                            c.len(),
                            &bad[..bad.len().min(6)],
                            bad.iter().take(3).map(|&j| c[j]).collect::<Vec<_>>(),
                            bad.iter().take(3).map(|&j| m[j]).collect::<Vec<_>>()
                        );
                    }
                }
                None => println!("pos {pos}: all {} outputs within 2e-3", cpu.len()),
            }
        }
    }

    #[test]
    fn test_gemma4_gguf_packed_parity() {
        gemma4_e2b_gguf_metal_parity();
    }

    #[test]
    fn gemma4_e2b_gguf_metal_parity() {
        #[cfg(not(target_os = "macos"))]
        return;
        if crate::hardware::available_devices().contains(&crate::hardware::Device::Metal) == false {
            return;
        }
        let snap = std::path::Path::new("disk/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF/snapshots/main/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf");
        if !snap.exists() {
            eprintln!("SKIP: snapshot not found at {:?}", snap);
            return;
        }
        let mut artifacts = crate::model::hub::ModelArtifacts::default();
        artifacts.weights_path = Some(snap.to_path_buf());
        let cpu = load_gguf(
            &artifacts,
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
        )
        .expect("load_gguf cpu is None")
        .expect("load_gguf cpu failed");
        let metal = load_gguf(
            &artifacts,
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
        )
        .expect("load_gguf metal is None")
        .expect("load_gguf metal failed");

        let tok = cpu.tokenizer();
        let prompt =
            "<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n";
        let (prompt_tokens, mut cpu_state) = cpu.prepare(tok.encode(prompt), &[]).unwrap();
        let (_, mut metal_state) = metal.prepare(tok.encode(prompt), &[]).unwrap();

        let cpu_logits = cpu.forward(&prompt_tokens, 0, &mut cpu_state).unwrap();
        let metal_logits = metal.forward(&prompt_tokens, 0, &mut metal_state).unwrap();

        let (cpu_argmax, _) = cpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (metal_argmax, _) = metal_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let mut max_diff = 0.0f32;
        for (c, m) in cpu_logits.iter().zip(&metal_logits) {
            max_diff = max_diff.max((c - m).abs());
        }
        println!(
            "E2B Prefill: CPU argmax={} ({:?}), Metal argmax={} ({:?}), max diff={max_diff}",
            cpu_argmax,
            tok.decode(&[cpu_argmax as u32]),
            metal_argmax,
            tok.decode(&[metal_argmax as u32])
        );

        assert_eq!(cpu_argmax, metal_argmax, "prefill argmax mismatch");
        assert!(max_diff < 0.1, "prefill max diff {max_diff} exceeds 0.1");

        let mut cur_tok = cpu_argmax as u32;
        let mut pos = prompt_tokens.len();
        for step in 0..3 {
            let c_logits = cpu.forward(&[cur_tok], pos, &mut cpu_state).unwrap();
            let m_logits = metal.forward(&[cur_tok], pos, &mut metal_state).unwrap();

            let (c_next, _) = c_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let (m_next, _) = m_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let mut diff = 0.0f32;
            for (c, m) in c_logits.iter().zip(&m_logits) {
                diff = diff.max((c - m).abs());
            }
            println!(
                "Step {step} pos {pos}: CPU={} ({:?}), Metal={} ({:?}), max diff={diff}",
                c_next,
                tok.decode(&[c_next as u32]),
                m_next,
                tok.decode(&[m_next as u32])
            );
            assert_eq!(c_next, m_next, "step {step} argmax mismatch");
            assert!(diff < 0.1, "step {step} max diff {diff} exceeds 0.1");

            cur_tok = c_next as u32;
            pos += 1;
        }
        let decode_steps = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..decode_steps {
            let m_logits = metal.forward(&[cur_tok], pos, &mut metal_state).unwrap();
            let (m_next, _) = m_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            cur_tok = m_next as u32;
            pos += 1;
        }
        let metal_elapsed = t0.elapsed();
        let metal_tps = decode_steps as f64 / metal_elapsed.as_secs_f64();
        println!(
            "E2B Metal Decode: {decode_steps} tokens in {:.3}s ({:.1} tok/s)",
            metal_elapsed.as_secs_f64(),
            metal_tps
        );
    }
}
