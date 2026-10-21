//! Qwen3.5 (`qwen3_5`: hybrid Gated DeltaNet / gated attention decoder + ViT) and Qwen3-VL
//! (`qwen3_vl`: dense Qwen3 decoder + ViT with deepstack feature injection) as [`NativeLm`]s:
//! text and images in, text out; plus the text-only dense decoders sharing that stack, Qwen3
//! (`qwen3`) and HunYuan dense v1 (`hunyuan_v1_dense`). CPU, bf16 weights borrowed from the mapped
//! checkpoint. Qwen3, HunYuan dense and Qwen3.5 (text only) also load from llama.cpp GGUFs.
//!
//! Reference: transformers `models/qwen3_5/modeling_qwen3_5.py`, `models/qwen3_vl/modeling_qwen3_vl.py`,
//! `models/qwen3/modeling_qwen3.py`, `models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py`
//! and the `Qwen2VLImageProcessor` the VL models share.

mod draft;
mod fixture;
#[cfg(test)]
mod gpu_tests;
mod text;
pub use fixture::{tiny_mtp_graph, tiny_ninfer_decode_graph, tiny_vision_tower_graph};
pub use text::{tiny_decode_graph, tiny_moe_decode_graph};
pub mod vision;

pub use draft::{
    gguf_checkpoint as gguf_drafter, load as load_drafter, ARCHITECTURES as DRAFT_ARCHITECTURES,
};

use crate::hardware::Device;
use crate::model::classifier::DecoderBackbone;
use crate::model::hf_loader::{load_json_tokenizer, tokenizer_from_files};
use crate::model::hub::ModelArtifacts;
use crate::model::image::ImageBuffer;
use crate::model::native::{MediaInput, NativeLm, SeqState};
use crate::model::par::{dot_bf16, par_for, par_rows, OutPtr};
use crate::model::tokenizer::Tokenizer;
use crate::model::weights::{Mat, Weights};
use crate::scheduler::spec::Drafter;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use text::{Arch, LayerState, MtpDrafter, MtpGpuDrafter, TextModel};
use vision::{Preprocessor, Vision};

/// The mapped safetensors checkpoint of `artifacts` (pre-quantized schemes vetted) and its
/// directory; Errs off the CPU.
fn open_checkpoint(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    _mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<(Weights, std::path::PathBuf), String> {
    let model_type = cfg["model_type"].as_str().unwrap_or("qwen3_5");
    let weights = artifacts
        .weights_path
        .as_deref()
        .ok_or("no safetensors weights in the model directory")?;
    let dir = weights.parent().unwrap_or(Path::new("."));
    let name = weights.to_string_lossy();
    if !(name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")) {
        return Err(format!(
            "{}: {model_type} loads safetensors checkpoints only",
            weights.display()
        ));
    }
    crate::model::prequant::check(artifacts.config_path.as_deref(), dir)?;
    let config = artifacts.config_path.as_deref();
    let ck = crate::model::prequant::open_weights(weights, config)?;
    Ok((ck, dir.to_path_buf()))
}

/// Loads a `qwen3_5` / `qwen3_5_text` / `qwen3_vl` checkpoint (`config.json` already parsed as
/// `cfg`). Text-only exports (flat `qwen3_5_text` configs, or checkpoints without `model.visual`)
/// load without the vision tower and refuse images.
pub fn load(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let mode = mode.into();
    let (ck, dir) = open_checkpoint(artifacts, cfg, mode)?;
    let arch = match cfg["model_type"].as_str() {
        Some("qwen3_vl") => Arch::Qwen3Vl,
        Some("qwen4_exp" | "qwen4_exp_text") => Arch::Qwen4Exp,
        _ => Arch::Qwen35,
    };
    let text = Arc::new(TextModel::load(&ck, text_config(cfg), arch)?);
    // ponytail: Qwen4-Exp's ViT (an unmodified Qwen3-VL one) is not wired in: text only.
    let vision = if arch == Arch::Qwen4Exp {
        None
    } else {
        load_vision(&ck, cfg, &text, || {
            Preprocessor::load(&dir.join("preprocessor_config.json"))
        })?
    };

    // The matrices keep the mapping alive; the index itself is no longer needed.
    drop(ck);
    let tok = load_json_tokenizer(&dir.join("tokenizer.json"))?;
    assemble(tok, text, vision, mode)
}

/// A [`Qwen35`] over `text` with its Metal programs (decoder; vision tower per image grid; the
/// MTP head when the checkpoint has one), or the host forward under `--backend cpu`.
fn assemble(
    tok: Tokenizer,
    text: Arc<TextModel>,
    vision: Option<(Vision, Preprocessor, u32)>,
    mode: crate::hardware::PlacementMode,
) -> Result<Arc<dyn NativeLm>, String> {
    let opts = text::DecodeOpts {
        mrope: vision.is_some(),
        deepstack: vision.as_ref().map_or(0, |(v, ..)| v.deepstack_layers()),
        tap: text.mtp.is_some(),
    };
    let program = compile_qwen_program(&text, mode, opts)?;
    let vision = vision
        .map(|(v, p, id)| Ok::<_, String>((v.on_gpu(mode)?, p, id)))
        .transpose()?;
    let drafter = text
        .mtp
        .is_some()
        .then(|| mtp_drafter(&text, &program, mode))
        .transpose()?;
    Ok(Arc::new(Qwen35 {
        tok,
        text,
        vision,
        drafter,
        program,
    }))
}

/// Test hook: the model [`load_ninfer`] / [`load`] assemble from an in-memory checkpoint `ck` with
/// config `cfg` (no tokenizer file: ids map to `t{id}`).
#[cfg(test)]
pub(crate) fn from_checkpoint(
    ck: &Weights,
    cfg: &Value,
    mode: crate::hardware::PlacementMode,
) -> Result<Arc<dyn NativeLm>, String> {
    let text = Arc::new(TextModel::load(ck, text_config(cfg), Arch::Qwen35)?);
    let vision = load_vision(ck, cfg, &text, || {
        Err("test checkpoint has no preprocessor".into())
    })?;
    let tok = Tokenizer::from_tokens((0..text.vocab).map(|i| format!("t{i}")).collect());
    assemble(tok, text, vision, mode)
}

/// The MTP drafter of `text`: its head as a Metal program next to the target's `program`, or the
/// host forward under `--backend cpu`.
fn mtp_drafter(
    text: &Arc<TextModel>,
    program: &Option<Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>>,
    mode: crate::hardware::PlacementMode,
) -> Result<Arc<dyn Drafter>, String> {
    if program.is_none() {
        return Ok(Arc::new(MtpDrafter(text.clone())));
    }
    let (graph, weights) = text
        .mtp_decode_graph(text.decode_max_ctx)
        .map_err(|e| format!("Qwen3.5 MTP: {e}"))?;
    let head = crate::graph::program::GraphProgram::compile(graph, weights, mode)
        .map_err(|e| format!("Qwen3.5 MTP: {e}"))?;
    Ok(Arc::new(MtpGpuDrafter::new(
        text.clone(),
        Arc::new(parking_lot::Mutex::new(head)),
    )))
}

/// The vision tower, its preprocessor and the image placeholder id of a checkpoint that has
/// `model.visual` (`pre` reads its `preprocessor_config.json`); `None` for text-only ones.
fn load_vision(
    ck: &Weights,
    cfg: &Value,
    text: &TextModel,
    pre: impl FnOnce() -> Result<Preprocessor, String>,
) -> Result<Option<(Vision, Preprocessor, u32)>, String> {
    if !ck.has("model.visual.patch_embed.proj.weight") {
        return Ok(None);
    }
    let vision = Vision::load(ck, &cfg["vision_config"])?;
    if vision.out_hidden() != text.hidden() {
        return Err(format!(
            "vision out_hidden_size {} != text hidden_size {}",
            vision.out_hidden(),
            text.hidden()
        ));
    }
    let image_token = cfg["image_token_id"]
        .as_u64()
        .ok_or("config.json has no image_token_id")? as u32;
    Ok(Some((vision, pre()?, image_token)))
}

/// `text_config`, or the whole config when it is flat (`qwen3_5_text`).
fn text_config(cfg: &Value) -> &Value {
    if cfg["text_config"].is_object() {
        &cfg["text_config"]
    } else {
        cfg
    }
}

/// Loads a text-only `qwen3` / `hunyuan_v1_dense` safetensors checkpoint.
pub fn load_dense(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let mode = mode.into();
    let (ck, dir) = open_checkpoint(artifacts, cfg, mode)?;
    let arch = if cfg["model_type"] == "qwen3" {
        Arch::Qwen3
    } else {
        Arch::HunYuan
    };
    let text = Arc::new(TextModel::load(&ck, cfg, arch)?);
    let tok = load_json_tokenizer(&dir.join("tokenizer.json"))?;
    let program = compile_qwen_program(&text, mode.into(), text::DecodeOpts::default())?;
    Ok(Arc::new(Dense { tok, text, program }))
}

/// The text decoder of a `Qwen3ForSequenceClassification` (`qwen3`) or
/// `Qwen3_5[Text]ForSequenceClassification` (`qwen3_5` with `text_config`, or flat
/// `qwen3_5_text`) checkpoint, for scoring the final-norm hidden state of the last token.
pub fn backbone(
    artifacts: &ModelArtifacts,
    cfg: &Value,
) -> Result<Box<dyn DecoderBackbone>, String> {
    let arch = match cfg["model_type"].as_str() {
        Some("qwen3") => Arch::Qwen3,
        Some("qwen3_5" | "qwen3_5_text") => Arch::Qwen35,
        other => return Err(format!("model_type {other:?} has no decoder backbone here")),
    };
    let (ck, dir) = open_checkpoint(artifacts, cfg, Device::Cpu)?;
    let text = TextModel::load(&ck, text_config(cfg), arch)?;
    let tok = load_json_tokenizer(&dir.join("tokenizer.json"))?;
    Ok(Box::new(Dense {
        tok,
        text: Arc::new(text),
        program: None,
    }))
}

impl DecoderBackbone for Dense {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn context_len(&self) -> usize {
        // The compiled seq-cls program's KV state holds `decode_max_ctx` positions.
        if self.program.is_some() {
            self.text.decode_max_ctx
        } else {
            self.text.context_len()
        }
    }

    fn hidden(&self) -> usize {
        self.text.hidden()
    }

    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        crate::hardware::cpu_compute::enter("decoder::last_hidden");
        let d = self.text.hidden();
        let mut h = vec![0.0f32; ids.len() * d];
        for (&t, row) in ids.iter().zip(h.chunks_exact_mut(d)) {
            self.text.embed(t, row)?;
        }
        // Text-only (M-)RoPE: every axis at the sequence index.
        let pos: Vec<[u32; 3]> = (0..ids.len() as u32).map(|i| [i; 3]).collect();
        let mut state = self.text.new_state();
        Ok(self.text.last_hidden(h, &pos, 0, &mut state, &[], &[]))
    }

    /// Compiles the decode graph with the score head for Metal. Dense's `program` (never a
    /// `NativeLm` here: `backbone` hands out a `DecoderBackbone`) then runs one token per step
    /// with KV / DeltaNet state, so the hybrid DeltaNet layers need no separate prefill graph.
    fn attach_score_gpu(&mut self, score: &[f32], labels: usize) -> Result<(), String> {
        if score.len() != labels * self.text.hidden() {
            return Err(format!(
                "score head is not [{labels}, {}]",
                self.text.hidden()
            ));
        }
        let gpu_only = |e: String| format!("gpu-only: Qwen3.5 seq-cls: {e}");
        let (graph, weights) = self.text.to_seqcls_graph(score).map_err(gpu_only)?;
        let mode = crate::hardware::PlacementMode::Pin(Device::Metal);
        let program =
            crate::graph::program::GraphProgram::compile(graph, weights, mode).map_err(gpu_only)?;
        self.program = Some(Arc::new(parking_lot::Mutex::new(program)));
        Ok(())
    }

    // ponytail: n sequential decode steps (one GPU dispatch each) over a fresh KV state sized to
    // `decode_max_ctx`; a batched prefill graph (DeltaRule over n rows) would cut the launches.
    fn score_gpu(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let prog = self
            .program
            .as_ref()
            .ok_or("no GPU score program attached")?;
        if ids.is_empty() || ids.len() > self.text.decode_max_ctx {
            return Err(format!(
                "{} tokens: the Metal program holds 1..={}",
                ids.len(),
                self.text.decode_max_ctx
            ));
        }
        let mut p = prog.lock();
        let saved = std::mem::replace(&mut p.state, crate::graph::StateStore::new());
        let run = |p: &mut crate::graph::program::GraphProgram| -> Result<Vec<f32>, String> {
            let mut logits = Vec::new();
            for (pos, &tok) in ids.iter().enumerate() {
                let mut emb = vec![0.0f32; self.text.hidden()];
                self.text.embed(tok, &mut emb)?;
                let inputs = vec![
                    crate::graph::Tensor::F32(emb),
                    crate::graph::Tensor::U32(vec![pos as u32]),
                ];
                logits = p
                    .run(inputs)?
                    .into_iter()
                    .next()
                    .ok_or("seq-cls program produced no output")?
                    .into_f32()?;
            }
            Ok(logits)
        };
        let logits = run(&mut p);
        p.state = saved;
        logits
    }
}

/// A `qwen3` checkpoint directory (`config.json`, `model.safetensors[.index.json]`) used as a
/// text encoder (e.g. by diffusion pipelines).
pub struct Qwen3Encoder {
    text: TextModel,
}

impl Qwen3Encoder {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg_path = dir.join("config.json");
        let cfg: Value = std::fs::read(&cfg_path)
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", cfg_path.display()))?;
        let index = dir.join("model.safetensors.index.json");
        let weights = if index.exists() {
            index
        } else {
            dir.join("model.safetensors")
        };
        // A Qwen3-VL checkpoint (Qwen-Image 2.1's encoder) and a Qwen2.5-VL one (Qwen-Image 1's)
        // are read for their text decoders only; plain `qwen2` shares that decoder.
        let arch = match cfg["model_type"].as_str() {
            Some("qwen3_vl") => Arch::Qwen3Vl,
            Some("qwen2" | "qwen2_5_vl") => Arch::Qwen2,
            _ => Arch::Qwen3,
        };
        let text = TextModel::load(&Weights::open(&weights)?, text_config(&cfg), arch)?;
        Ok(Self { text })
    }

    /// The text tower of a Qwen3-VL checkpoint file or shard index (`model.language_model.*`; its
    /// vision tower stays unread), `cfg` being the model's `config.json` (`text_config` nested).
    pub fn from_weights(weights: &Path, cfg: &Value) -> Result<Self, String> {
        let text = TextModel::load(&Weights::open(weights)?, text_config(cfg), Arch::Qwen3Vl)?;
        Ok(Self { text })
    }

    /// `[ids.len(), hidden]` after the first `layers` decoder layers, before the final norm.
    pub fn hidden_states(&self, ids: &[u32], layers: usize) -> Result<Vec<f32>, String> {
        Ok(self.hidden_states_at(ids, ids.len(), &[layers])?.remove(0))
    }

    /// transformers' `hidden_states[-1]` of `ids`: `[ids.len(), hidden]` after the final norm.
    pub fn hidden_states_normed(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.text.normed_hidden_states(ids)
    }

    /// transformers' `hidden_states[k]` for each `k` in `layers` (ascending) with attention mask
    /// `[1] * valid + [0] * (len - valid)`, in one pass.
    pub fn hidden_states_at(
        &self,
        ids: &[u32],
        valid: usize,
        layers: &[usize],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.text.hidden_states_at(ids, valid, layers)
    }
}

#[cfg(test)]
impl Qwen3Encoder {
    /// The text tower of a llama.cpp `qwen3vl` GGUF: a quantized stand-in for an encoder too
    /// large to download in full (tests only).
    pub(crate) fn from_gguf(path: &Path) -> Result<Self, String> {
        let (manifest, data) = crate::model::loaders::gguf::open(path)?;
        let (ck, cfg, arch) = gguf_checkpoint(&Arc::new(data), &manifest)?;
        Ok(Self {
            text: TextModel::load(&ck, &cfg, arch)?,
        })
    }
}

/// A llama.cpp `qwen3`, `qwen3moe`, `hunyuan-dense`, `qwen35`, `qwen35moe` or `qwen4exp` GGUF file as a
/// [`NativeLm`]: tensors renamed to their transformers names (llama.cpp's conversion transforms
/// undone) and dequantized to bf16, config from the `<arch>.*` metadata; MoE expert banks stay
/// packed in the file data (mapped for a single file). `None` for anything else (the GraphLm path
/// takes it).
pub fn load_gguf(
    artifacts: &ModelArtifacts,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    let path = artifacts
        .weights_path
        .as_deref()
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))?;
    // Non-native architectures fall through to the GraphLm path without reading the (split) set.
    let arch = crate::model::loaders::gguf::architecture(path)?;
    matches!(
        arch.as_str(),
        "qwen3" | "qwen3moe" | "hunyuan-dense" | "qwen35" | "qwen35moe" | "qwen4exp"
    )
    .then(|| {
        let (manifest, data) = crate::model::loaders::gguf::open(path)?;
        // ponytail: a split set is one heap buffer (expert banks included); a single file stays mapped.
        gguf_model(&Arc::new(data), &manifest, mode)
    })
    .map(|r| r.map_err(|e| format!("{}: {e}", path.display())))
}

/// Decode graph and weights of a GGUF [`load_gguf`] takes, built on the host without a device
/// (graph-compiler tests on real checkpoints).
pub fn gguf_decode_graph(
    path: &std::path::Path,
) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
    let (manifest, data) = crate::model::loaders::gguf::open(path)?;
    let (ck, cfg, arch) = gguf_checkpoint(&Arc::new(data), &manifest)?;
    TextModel::load(&ck, &cfg, arch)?.to_decode_graph()
}

fn gguf_model(
    map: &Arc<crate::model::loaders::gguf::GgufData>,
    m: &crate::model::ModelManifest,
    _mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let mode = _mode.into();
    let (ck, cfg, arch) = gguf_checkpoint(map, m)?;
    let text = Arc::new(TextModel::load(&ck, &cfg, arch)?);
    let tok = crate::model::UniversalModelLoader::extract_gguf_tokenizer(map)?;
    let program = compile_qwen_program(&text, mode, text::DecodeOpts::default())?;
    Ok(if matches!(arch, Arch::Qwen35 | Arch::Qwen4Exp) {
        // llama.cpp ships the vision tower as a separate mmproj GGUF: images are refused.
        Arc::new(Qwen35 {
            tok,
            text,
            vision: None,
            drafter: None,
            program,
        })
    } else {
        Arc::new(Dense { tok, text, program })
    })
}

/// A `.ninfer` artifact ([`crate::model::ninfer`]: dense Qwen3.5 / 3.6 / 3.8 text model and, in v3,
/// its vision tower; speculative components left unread) as a [`NativeLm`]; `None` for any other
/// weights file.
pub fn load_ninfer(
    artifacts: &ModelArtifacts,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    let path = artifacts
        .weights_path
        .as_deref()
        .filter(|p| p.to_string_lossy().ends_with(crate::model::ninfer::EXT))?;
    let mode = mode.into();
    let load = || -> Result<Arc<dyn NativeLm>, String> {
        let art = crate::model::ninfer::Artifact::open(path)?;
        let (ck, cfg) = art.checkpoint()?;
        let text = TextModel::load(&ck, &cfg, Arch::Qwen35)?;
        let vision = load_vision(&ck, &cfg, &text, || {
            let json = art
                .resource("preprocessor_config.json")
                .ok_or("no preprocessor_config.json resource")?;
            Preprocessor::parse(&json, "preprocessor_config.json")
        })?;
        let json = art
            .resource("tokenizer.json")
            .ok_or("no tokenizer.json resource")?;
        let tok = tokenizer_from_files(&String::from_utf8_lossy(&json), &|n| art.resource(n))?;
        assemble(tok, Arc::new(text), vision, mode)
    };
    Some(load().map_err(|e| format!("{}: {e}", path.display())))
}

/// `config.json` equivalent, decoder and layer count of a GGUF that [`load_gguf`] takes.
fn gguf_config(m: &crate::model::ModelManifest) -> Result<(Value, Arch, usize), String> {
    let a = m
        .metadata
        .get("general.architecture")
        .map_or("", String::as_str);
    let arch = match a {
        "qwen3" | "qwen3moe" => Arch::Qwen3,
        "hunyuan-dense" => Arch::HunYuan,
        "qwen35" | "qwen35moe" => Arch::Qwen35,
        "qwen3vl" => Arch::Qwen3Vl,
        "qwen4exp" => {
            if !m.tensors.contains_key("blk.0.hc_attn_norm.weight") {
                return Err("this qwen4exp file is a detached MTP head (no trunk tensors); load the trunk GGUF".into());
            }
            Arch::Qwen4Exp
        }
        _ => return Err(format!("GGUF architecture '{a}' is not a native decoder")),
    };
    let num = |k: &str| {
        m.metadata
            .get(&format!("{a}.{k}"))
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| format!("GGUF metadata {a}.{k} missing"))
    };
    let int = |k: &str| num(k).map(|v| v as usize);
    if let Some(t) = m
        .metadata
        .get(&format!("{a}.rope.scaling.type"))
        .filter(|t| *t != "none")
    {
        return Err(format!("{a} RoPE scaling '{t}' is not implemented"));
    }
    let (hidden, heads) = (int("embedding_length")?, int("attention.head_count")?);
    let head_dim = int("attention.key_length").unwrap_or(hidden / heads.max(1));
    // Multi-token-prediction (`nextn`) blocks follow the decoder's; they are not loaded.
    let layers = int("block_count")?.saturating_sub(int("nextn_predict_layers").unwrap_or(0));
    let mut cfg = serde_json::json!({
        "hidden_size": hidden,
        "intermediate_size": if matches!(a, "qwen3moe" | "qwen35moe" | "qwen4exp") { 0 } else { int("feed_forward_length")? },
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": int("attention.head_count_kv")?,
        "head_dim": head_dim,
        "rms_norm_eps": num("attention.layer_norm_rms_epsilon")?,
        "rope_theta": num("rope.freq_base")?,
        "max_position_embeddings": int("context_length")?,
        "tie_word_embeddings": !m.tensors.contains_key("output.weight"),
    });
    match arch {
        // The converter bakes HunYuan's NTK alpha into `rope.freq_base`; llama.cpp scales it once
        // more only when the file also stores `rope.scaling.alpha`.
        Arch::HunYuan => {
            if let Some(alpha) = num("rope.scaling.alpha").ok().filter(|&v| v > 0.0) {
                cfg["rope_scaling"] = serde_json::json!({"type": "dynamic", "alpha": alpha});
            }
        }
        Arch::Qwen35 | Arch::Qwen4Exp => {
            let rot = int("rope.dimension_count")?;
            let v_heads = int("ssm.time_step_rank")?;
            let kinds: Vec<&str> = (0..layers)
                .map(
                    |i| match m.tensors.contains_key(&format!("blk.{i}.ssm_a")) {
                        true => "linear_attention",
                        false => "full_attention",
                    },
                )
                .collect();
            let extra = serde_json::json!({
                // Text only (no vision tower): every M-RoPE axis carries the sequence position,
                // so the section split (`rope.dimension_sections`) cannot change a rotation.
                "rope_parameters": {
                    "rope_theta": cfg["rope_theta"],
                    "partial_rotary_factor": rot as f64 / head_dim as f64,
                    "mrope_interleaved": true,
                    "mrope_section": [rot / 2, 0, 0],
                },
                "layer_types": kinds,
                "linear_num_key_heads": int("ssm.group_count")?,
                "linear_num_value_heads": v_heads,
                "linear_key_head_dim": int("ssm.state_size")?,
                "linear_value_head_dim": int("ssm.inner_size")? / v_heads.max(1),
                "linear_conv_kernel_dim": int("ssm.conv_kernel")?,
                "attn_output_gate": true,
            });
            if matches!(a, "qwen35moe" | "qwen4exp") {
                cfg["num_experts"] = int("expert_count")?.into();
                cfg["num_experts_per_tok"] = int("expert_used_count")?.into();
            }
            if arch == Arch::Qwen4Exp {
                let list = |k: &str| -> Vec<u64> {
                    m.metadata
                        .get(&format!("{a}.{k}"))
                        .map_or_else(Vec::new, |s| {
                            s.split(',').filter_map(|v| v.parse().ok()).collect()
                        })
                };
                let ratios = list("attention.compress_ratios");
                let ratio = ratios
                    .iter()
                    .take(layers)
                    .copied()
                    .find(|&r| r > 0)
                    .unwrap_or(0);
                cfg["output_gate_type"] = "sigmoid".into();
                cfg["hc_count"] = int("hyper_connection.count")?.into();
                cfg["hc_lowrank"] = int("hyper_connection.low_rank")?.into();
                cfg["indexer_n_heads"] = int("attention.indexer.head_count")?.into();
                cfg["indexer_head_dim"] = int("attention.indexer.key_length")?.into();
                cfg["indexer_budget"] = int("attention.indexer.top_k")?.into();
                cfg["indexer_compress_ratio"] = ratio.into();
                if let Some(&l) = list("ple.layers").first() {
                    cfg["ple_layer_ids"] = serde_json::json!([l + 1]);
                    cfg["ngram_size"] = int("ple.ngram_size")?.into();
                    cfg["heads_per_ngram"] = int("ple.heads_per_ngram")?.into();
                    cfg["ple_conv_kernel_size"] = int("ple.conv_kernel")?.into();
                    cfg["eos_token_id"] = int("ple.eos_token_id")?.into();
                    cfg["ple_hash"] = serde_json::json!({
                        "multipliers": list("ple.layer_multipliers"),
                        "offsets": list("ple.head_offsets"),
                        "vocab_sizes": list("ple.head_vocab_sizes"),
                    });
                }
            }
            for (k, v) in extra.as_object().expect("object literal") {
                cfg[k] = v.clone();
            }
        }
        Arch::Qwen3 | Arch::Qwen2 => {
            if a == "qwen3moe" {
                cfg["num_experts"] = int("expert_count")?.into();
                cfg["num_experts_per_tok"] = int("expert_used_count")?.into();
            }
        }
        // Text tower of a Qwen3-VL GGUF (the vision tower is a separate mmproj file): text-only
        // positions put every M-RoPE axis at the sequence index, so the section split is moot.
        Arch::Qwen3Vl => {
            cfg["rope_scaling"] = serde_json::json!({
                "mrope_interleaved": true,
                "mrope_section": [head_dim / 2, 0, 0],
            });
        }
    }
    Ok((cfg, arch, layers))
}

/// The transformers-named checkpoint, `config.json` equivalent and decoder of a GGUF that
/// [`load_gguf`] takes.
fn gguf_checkpoint(
    data: &Arc<crate::model::loaders::gguf::GgufData>,
    m: &crate::model::ModelManifest,
) -> Result<(Weights, Value, Arch), String> {
    let (cfg, arch, layers) = gguf_config(m)?;
    let a = &m.metadata["general.architecture"];
    let mut gguf_name = std::collections::HashMap::new();
    let mut specs = Vec::with_capacity(m.tensors.len());
    // Expert banks: (HF name, byte offset, format, `[experts, rows, cols]`), left in the mapping.
    let mut packed = Vec::new();
    let (kh, vh, vd, qk) = if matches!(arch, Arch::Qwen35 | Arch::Qwen4Exp) {
        let k = cfg["linear_num_key_heads"].as_u64().unwrap_or(1) as usize;
        let v = cfg["linear_num_value_heads"].as_u64().unwrap_or(1) as usize;
        let d = cfg["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
        let kd = cfg["linear_key_head_dim"].as_u64().unwrap_or(128) as usize;
        (k, v, d, 2 * k * kd)
    } else {
        (1, 1, 128, 0)
    };
    let mut permuted_packed: Vec<(String, crate::model::weights::Packed)> = Vec::new();
    for (name, t) in &m.tensors {
        let block = name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, _)| l.parse::<usize>().ok());
        if block.is_some_and(|l| l >= layers) {
            continue;
        }
        // The PLE n-gram table is one global tensor (read lazily, row by row).
        if name == "per_layer_token_embd.weight" {
            let l = cfg["ple_layer_ids"][0]
                .as_u64()
                .ok_or("PLE table without `ple.layers`")?
                - 1;
            let [hd, rows] = t.shape[..] else {
                return Err(format!("{name}: expected [row dim, rows]"));
            };
            let hf = format!("model.layers.{l}.ple.ple_embedding.table");
            packed.push((hf, t.byte_offset, t.quant_format, vec![1, rows, hd]));
            continue;
        }
        let hf = gguf_to_hf(arch, name)
            .ok_or_else(|| format!("GGUF tensor '{name}' is not part of the {a} graph"))?;
        // GGUF lists dims fastest first.
        let mut shape: Vec<usize> = t.shape.iter().rev().copied().collect();
        let is_permuted_linear = kh != vh
            && shape.len() == 2
            && (hf.ends_with(".linear_attn.in_proj_z.weight")
                || hf.ends_with(".linear_attn.in_proj_qkv.weight"));

        if is_permuted_linear {
            let cols = shape[1];
            let rows = shape[0];
            let row_bytes = t.quant_format.tensor_bytes(cols);
            let total_bytes = rows * row_bytes;
            if t.byte_offset + total_bytes <= data.len() {
                let slice = &data[t.byte_offset..t.byte_offset + total_bytes];
                let prefix_rows = if hf.ends_with(".linear_attn.in_proj_qkv.weight") {
                    qk.min(rows)
                } else {
                    0
                };
                let permuted = untile_packed_rows(slice, row_bytes, vd, kh, vh, prefix_rows);
                permuted_packed.push((
                    hf,
                    crate::model::weights::Packed {
                        map: Arc::new(crate::model::loaders::gguf::GgufData::Buf(permuted)),
                        off: 0,
                        format: t.quant_format,
                        shape,
                    },
                ));
                continue;
            }
        }

        let legacy = crate::graph::disabled("qwen_packed");
        let needs_transform = matches!(arch, Arch::Qwen35 | Arch::Qwen4Exp)
            && qwen35_needs_transform(&hf, kh, vh, is_permuted_linear, legacy);
        // Matrices stay packed in BF16/FP16 and in any block format the GPU codec registry decodes (a
        // dequantized copy streams 2-3x the bytes per token). Expert banks keep the wider block-codec
        // predicate: dequantized they would not fit in memory.
        let block_ok = |cols: usize| {
            crate::model::layers::gguf_block_codec(t.quant_format)
                .is_some_and(|(elems, _, _)| cols % elems == 0)
        };
        let packable = |cols: usize| {
            matches!(
                t.quant_format,
                crate::hardware::QuantFormat::BF16 | crate::hardware::QuantFormat::FP16
            ) || (crate::graph::codec::codec(t.quant_format).is_some() && block_ok(cols))
        };
        let bank = hf.contains(".mlp.experts.");
        let can_pack = ((shape.len() == 2 && packable(shape[1]))
            || (bank && (packable(shape[shape.len() - 1]) || block_ok(shape[shape.len() - 1]))))
            && !needs_transform;
        // Value heads tiled across `out_proj` columns: untile whole quant blocks per row when each
        // `vd`-column head is block aligned.
        if needs_transform
            && !legacy
            && kh != vh
            && hf.ends_with(".linear_attn.out_proj.weight")
            && shape.len() == 2
            && packable(shape[1])
        {
            let row_bytes = t.quant_format.tensor_bytes(shape[1]);
            let seg = t.quant_format.tensor_bytes(vd);
            let total = shape[0] * row_bytes;
            if packable(vd) && seg * vh == row_bytes && t.byte_offset + total <= data.len() {
                let permuted: Vec<u8> = data[t.byte_offset..t.byte_offset + total]
                    .chunks_exact(row_bytes)
                    .flat_map(|row| untile_packed_rows(row, seg, 1, kh, vh, 0))
                    .collect();
                permuted_packed.push((
                    hf,
                    crate::model::weights::Packed {
                        map: Arc::new(crate::model::loaders::gguf::GgufData::Buf(permuted)),
                        off: 0,
                        format: t.quant_format,
                        shape,
                    },
                ));
                continue;
            }
            // Value heads split quant blocks: keep the GGUF (tiled) column order; the graph tiles the input instead.
            packed.push((
                hf.replace(".out_proj.weight", ".out_proj_tiled.weight"),
                t.byte_offset,
                t.quant_format,
                shape,
            ));
            continue;
        }

        if can_pack {
            packed.push((hf, t.byte_offset, t.quant_format, shape));
            continue;
        }
        // The shared expert's gate is one `[hidden]` row.
        if hf.ends_with(".shared_expert_gate.weight") {
            shape.insert(0, 1);
        }
        specs.push((hf.clone(), shape));
        gguf_name.insert(hf, name.as_str());
    }
    let ck = Weights::from_f32(specs, |hf| {
        let mut v = m.extract_tensor_f32(data, gguf_name[hf])?;
        if matches!(arch, Arch::Qwen35 | Arch::Qwen4Exp) {
            qwen35_from_gguf(hf, &mut v, &cfg);
        }
        Ok(v)
    })?
    .with_gguf_packed(data, packed);
    let mut ck = ck;
    for (name, p) in permuted_packed {
        ck.insert_packed(name, p);
    }
    Ok((ck, cfg, arch))
}

/// The `config.json` equivalent and embedding / lm_head tensors (transformers names, bf16) of a
/// native GGUF target, for drafters that share them: only those two tensors are decoded.
pub fn gguf_target(path: &Path) -> Result<(Value, Weights), String> {
    let (m, data) = crate::model::loaders::gguf::open(path)?;
    let (mut cfg, arch, _) = gguf_config(&m).map_err(|e| format!("{}: {e}", path.display()))?;
    cfg["model_type"] = match arch {
        Arch::Qwen35 => "qwen3_5",
        Arch::HunYuan => "hunyuan_v1_dense",
        _ => "qwen3",
    }
    .into();
    let embd = m
        .tensors
        .get("token_embd.weight")
        .ok_or_else(|| format!("{}: no token_embd.weight", path.display()))?;
    cfg["vocab_size"] = embd.shape.last().copied().into();
    let specs = [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output.weight", "lm_head.weight"),
    ]
    .into_iter()
    .filter_map(|(g, hf)| {
        let t = m.tensors.get(g)?;
        Some((
            hf.to_string(),
            t.shape.iter().rev().copied().collect::<Vec<usize>>(),
            g,
        ))
    })
    .collect::<Vec<_>>();
    let names: std::collections::HashMap<&str, &str> =
        specs.iter().map(|(hf, _, g)| (hf.as_str(), *g)).collect();
    let ck = Weights::from_f32(
        specs
            .iter()
            .map(|(hf, s, _)| (hf.clone(), s.clone()))
            .collect(),
        |hf| m.extract_tensor_f32(&data, names[hf]),
    )?;
    Ok((cfg, ck))
}

/// [`qwen35_transforms`], or with `legacy` (`GIBBS_DISABLE=qwen_packed`) the predicate before it:
/// every `.linear_attn.` tensor that is not a permuted linear, plus `out_proj`, dequantizes.
fn qwen35_needs_transform(hf: &str, kh: usize, vh: usize, permuted: bool, legacy: bool) -> bool {
    if !legacy {
        return qwen35_transforms(hf, kh, vh);
    }
    hf.ends_with("norm.weight")
        || hf.contains(".ple.norm_")
        || hf.ends_with(".linear_attn.out_proj.weight")
        || (hf.contains(".linear_attn.") && !permuted)
}

/// Whether [`qwen35_from_gguf`] changes tensor `hf` (key / value head counts `kh` / `vh`).
fn qwen35_transforms(hf: &str, kh: usize, vh: usize) -> bool {
    match hf.rsplit_once(".linear_attn.") {
        None => hf.ends_with("norm.weight") || hf.contains(".ple.norm_"),
        Some((_, rest)) => {
            kh != vh
                && matches!(
                    rest,
                    "in_proj_qkv.weight"
                        | "conv1d.weight"
                        | "in_proj_z.weight"
                        | "in_proj_a.weight"
                        | "in_proj_b.weight"
                        | "neg_A"
                        | "dt_bias"
                        | "out_proj.weight"
                )
        }
    }
}

/// Undoes llama.cpp's Qwen3.5 conversion (`Qwen3NextModel` and `_LinearAttentionVReorderBase`
/// `modify_tensors`) on tensor `hf` (transformers name, config `c`) in place: `1 + w` RMSNorm
/// weights back to `w` (the gated DeltaNet norm is stored as is),
/// and value heads from llama.cpp's tiled order back to transformers' grouped-by-key-head order.
fn qwen35_from_gguf(hf: &str, v: &mut [f32], c: &Value) {
    let Some((_, rest)) = hf.rsplit_once(".linear_attn.") else {
        if hf.ends_with("norm.weight") || hf.contains(".ple.norm_") {
            v.iter_mut().for_each(|w| *w -= 1.0);
        }
        return;
    };
    // `ssm_a` is `-exp(A_log)` already (`neg_A`): only the head order below changes.
    let u = |k: &str| c[k].as_u64().unwrap_or(1) as usize;
    let (kh, vh) = (u("linear_num_key_heads"), u("linear_num_value_heads"));
    let (qk, vd, n) = (
        2 * kh * u("linear_key_head_dim"),
        u("linear_value_head_dim"),
        v.len(),
    );
    if kh == vh {
        return;
    }
    match rest {
        // q and k channels lead, then the value heads; `per` = inputs or conv taps per channel.
        "in_proj_qkv.weight" | "conv1d.weight" => {
            let per = n / (qk + vh * vd);
            untile_heads(&mut v[qk * per..], kh, vd * per)
        }
        "in_proj_z.weight" | "in_proj_a.weight" | "in_proj_b.weight" | "neg_A" | "dt_bias" => {
            untile_heads(v, kh, n / vh)
        }
        "out_proj.weight" => v
            .chunks_exact_mut(vh * vd)
            .for_each(|row| untile_heads(row, kh, vd)),
        _ => {}
    }
}

/// `x`, value heads of `block` values each in llama.cpp's tiled order (`[v_per_k, k_heads]`),
/// rewritten in transformers' grouped order (`[k_heads, v_per_k]`).
fn untile_heads(x: &mut [f32], k_heads: usize, block: usize) {
    let per_k = x.len() / (block * k_heads);
    let tiled = x.to_vec();
    for k in 0..k_heads {
        for r in 0..per_k {
            x[(k * per_k + r) * block..][..block]
                .copy_from_slice(&tiled[(r * k_heads + k) * block..][..block]);
        }
    }
}
/// Untiles block-row slices directly in quantized packed byte buffer.
fn untile_packed_rows(
    data: &[u8],
    row_bytes: usize,
    rows_per_block: usize,
    k_heads: usize,
    v_heads: usize,
    prefix_rows: usize,
) -> Vec<u8> {
    let mut out = data.to_vec();
    let per_k = v_heads / k_heads;
    let block_bytes = rows_per_block * row_bytes;
    let prefix_bytes = prefix_rows * row_bytes;
    if prefix_bytes >= data.len() {
        return out;
    }
    let tiled = &data[prefix_bytes..];
    let dest = &mut out[prefix_bytes..];
    for k in 0..k_heads {
        for r in 0..per_k {
            let src = (r * k_heads + k) * block_bytes;
            let dst = (k * per_k + r) * block_bytes;
            if src + block_bytes <= tiled.len() && dst + block_bytes <= dest.len() {
                dest[dst..dst + block_bytes].copy_from_slice(&tiled[src..src + block_bytes]);
            }
        }
    }
    out
}

/// llama.cpp's `qwen3` / `hunyuan-dense` / `qwen35[moe]` / `qwen4exp` tensor names -> transformers'.
fn gguf_to_hf(arch: Arch, name: &str) -> Option<String> {
    let top = match name {
        "token_embd.weight" => Some("model.embed_tokens.weight"),
        "output_norm.weight" => Some("model.norm.weight"),
        "output.weight" => Some("lm_head.weight"),
        "output_hc_norm.weight" => Some("model.hyper_connection_mixer.hc_norm.weight"),
        "output_hc_down.weight" => {
            Some("model.hyper_connection_mixer.input_mix_weight_down.weight")
        }
        "output_hc_up.weight" => Some("model.hyper_connection_mixer.input_mix_weight_up.weight"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.into());
    }
    let (layer, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    let q35 = matches!(arch, Arch::Qwen35 | Arch::Qwen4Exp);
    let hf = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" if !q35 => "post_attention_layernorm.weight",
        "post_attention_norm.weight" if q35 => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" if arch == Arch::HunYuan => "self_attn.query_layernorm.weight",
        "attn_k_norm.weight" if arch == Arch::HunYuan => "self_attn.key_layernorm.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "ffn_gate_inp_shexp.weight" if q35 => "mlp.shared_expert_gate.weight",
        "ffn_gate_shexp.weight" if q35 => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" if q35 => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" if q35 => "mlp.shared_expert.down_proj.weight",
        // Expert banks have no `.weight` in HF's fused layout (`experts.gate_up_proj`).
        "ffn_gate_exps.weight" => "mlp.experts.gate_proj",
        "ffn_up_exps.weight" => "mlp.experts.up_proj",
        "ffn_down_exps.weight" => "mlp.experts.down_proj",
        "ffn_gate_up_exps.weight" => "mlp.experts.gate_up_proj",
        "attn_qkv.weight" if q35 => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" if q35 => "linear_attn.in_proj_z.weight",
        "ssm_beta.weight" if q35 => "linear_attn.in_proj_b.weight",
        "ssm_alpha.weight" if q35 => "linear_attn.in_proj_a.weight",
        "ssm_a" if q35 => "linear_attn.neg_A",
        "ssm_dt.bias" if q35 => "linear_attn.dt_bias",
        "ssm_conv1d.weight" if q35 => "linear_attn.conv1d.weight",
        "ssm_norm.weight" if q35 => "linear_attn.norm.weight",
        "ssm_out.weight" if q35 => "linear_attn.out_proj.weight",
        "hc_attn_norm.weight" if q35 => "attn_hyper_connection.hc_norm.weight",
        "hc_attn_down.weight" if q35 => "attn_hyper_connection.input_mix_weight_down.weight",
        "hc_attn_up.weight" if q35 => "attn_hyper_connection.input_mix_weight_up.weight",
        "hc_attn_inject.weight" if q35 => "attn_hyper_connection.block_inject_weight.weight",
        "hc_ffn_norm.weight" if q35 => "mlp_hyper_connection.hc_norm.weight",
        "hc_ffn_down.weight" if q35 => "mlp_hyper_connection.input_mix_weight_down.weight",
        "hc_ffn_up.weight" if q35 => "mlp_hyper_connection.input_mix_weight_up.weight",
        "hc_ffn_inject.weight" if q35 => "mlp_hyper_connection.block_inject_weight.weight",
        "indexer.q_proj.weight" if q35 => "self_attn.indexer.index_q_proj.weight",
        "indexer.k_proj.weight" if q35 => "self_attn.indexer.index_k_proj.weight",
        "indexer.q_norm.weight" if q35 => "self_attn.indexer.q_layernorm.weight",
        "indexer.k_norm.weight" if q35 => "self_attn.indexer.k_layernorm.weight",
        "ple_key.weight" if q35 => "ple.key_proj.weight",
        "ple_value.weight" if q35 => "ple.value_proj.weight",
        "ple_norm_key.weight" if q35 => "ple.norm_key.weight",
        "ple_norm_query.weight" if q35 => "ple.norm_query.weight",
        "ple_norm_conv.weight" if q35 => "ple.norm_conv.weight",
        "ple_conv1d.weight" if q35 => "ple.conv1d.weight",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{hf}"))
}

/// Text-only dense decoder (Qwen3, HunYuan dense v1); also the decoder body of the Qwen3 /
/// Qwen3.5 sequence classifiers ([`backbone`]).
fn compile_qwen_program(
    text: &Arc<TextModel>,
    mode: crate::hardware::PlacementMode,
    opts: text::DecodeOpts,
) -> Result<Option<Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>>, String> {
    if matches!(mode, crate::hardware::PlacementMode::Pin(Device::Cpu)) {
        return Ok(None);
    }
    let (graph, weights) = text
        .to_decode_graph_with(text.decode_max_ctx, opts)
        .map_err(|e| format!("Qwen3.5: {e}"))?;
    crate::graph::program::GraphProgram::compile(graph, weights, mode)
        .map(|prog| Some(Arc::new(parking_lot::Mutex::new(prog))))
        .map_err(|e| format!("Qwen3.5: {e}"))
}

/// Text-only dense decoder (Qwen3, HunYuan dense v1); also the decoder body of the Qwen3 /
/// Qwen3.5 sequence classifiers ([`backbone`]).
struct Dense {
    tok: Tokenizer,
    text: Arc<TextModel>,
    program: Option<Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>>,
}

struct DenseSeq {
    layers: Vec<LayerState>,
    program_state: crate::graph::StateStore,
    len: usize,
    /// Decoder layers whose outputs every forward records into `tapped` (see `set_taps`).
    taps: Vec<usize>,
    tapped: Vec<f32>,
}

impl NativeLm for Dense {
    fn backend_label(&self) -> String {
        if let Some(prog) = &self.program {
            return prog.lock().backend_label();
        }
        crate::scheduler::cpu_backend_label()
    }

    fn device(&self) -> Device {
        if let Some(prog) = &self.program {
            return prog.lock().primary_device();
        }
        Device::Cpu
    }

    fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        Some(self.program.as_ref()?.lock().placement_summary())
    }

    fn summary(&self) -> String {
        self.text.summary()
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn context_len(&self) -> usize {
        if self.program.is_some() {
            self.text.decode_max_ctx
        } else {
            self.text.context_len()
        }
    }

    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        if !media.is_empty() {
            return Err("this model takes text only".into());
        }
        let seq = DenseSeq {
            layers: self.text.new_state(),
            program_state: crate::graph::StateStore::new(),
            len: 0,
            taps: Vec::new(),
            tapped: Vec::new(),
        };
        Ok((ids, Box::new(seq)))
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        crate::model::native::logits_of(self.run(tokens, start_pos, state, false, false)?)
    }

    fn forward_greedy(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<crate::graph::Tensor, String> {
        self.run(tokens, start_pos, state, false, true)
    }

    fn forward_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        crate::model::native::logits_of(self.run(tokens, start_pos, state, true, false)?)
    }

    fn truncate(&self, state: &mut SeqState, len: usize) -> Result<(), String> {
        let seq = Self::seq(state)?;
        if len > seq.len {
            return Err(format!("cannot truncate {} positions to {len}", seq.len));
        }

        self.text.truncate(&mut seq.layers, len)?;
        seq.len = len;
        Ok(())
    }

    fn set_taps(&self, state: &mut SeqState, layers: &[usize]) -> Result<(), String> {
        if let Some(&l) = layers.iter().find(|&&l| l >= self.text.layers()) {
            return Err(format!(
                "tap layer {l}: the decoder has {} layers",
                self.text.layers()
            ));
        }
        Self::seq(state)?.taps = layers.to_vec();
        Ok(())
    }

    fn take_taps(&self, state: &mut SeqState) -> Vec<f32> {
        Self::seq(state).map_or_else(|_| Vec::new(), |s| std::mem::take(&mut s.tapped))
    }
}

impl Dense {
    fn seq(state: &mut SeqState) -> Result<&mut DenseSeq, String> {
        state
            .downcast_mut::<DenseSeq>()
            .ok_or_else(|| "sequence state was not built by this model".into())
    }

    /// Runs `tokens` from `start_pos` (rewinding the KV caches to it); logits after the last
    /// token, or after every token when `all`. `greedy`: the program may return the last
    /// position's device-picked token instead (`U32([token])`).
    fn run(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
        all: bool,
        greedy: bool,
    ) -> Result<crate::graph::Tensor, String> {
        use crate::graph::Tensor;
        let seq = Self::seq(state)?;
        if start_pos > seq.len || tokens.is_empty() {
            return Err(format!(
                "cannot run {} tokens from position {start_pos} (state holds {})",
                tokens.len(),
                seq.len
            ));
        }
        if let Some(prog) = &self.program {
            if seq.taps.is_empty() {
                let mut p = prog.lock();
                std::mem::swap(&mut p.state, &mut seq.program_state);
                let mut logits = Tensor::F32(Vec::new());
                for (i, &tok) in tokens.iter().enumerate() {
                    let pos = start_pos + i;
                    if pos >= self.text.decode_max_ctx {
                        std::mem::swap(&mut p.state, &mut seq.program_state);
                        return Err(format!(
                            "position {pos} exceeds compiled decode context limit {}",
                            self.text.decode_max_ctx
                        ));
                    }
                    let mut emb = vec![0.0f32; self.text.hidden()];
                    self.text.embed(tok, &mut emb)?;
                    let inputs = vec![Tensor::F32(emb), Tensor::U32(vec![pos as u32])];
                    let last = i + 1 == tokens.len();
                    let step_out = if greedy && last {
                        p.run_greedy(inputs, self.tok.vocab_size())
                    } else {
                        p.run(inputs)
                    };
                    match step_out {
                        Ok(res) => match (res.into_iter().next(), &mut logits) {
                            (Some(t @ Tensor::U32(_)), l) => *l = t,
                            (Some(Tensor::F32(v)), Tensor::F32(l)) if all || last => {
                                l.extend_from_slice(&v)
                            }
                            _ => {}
                        },
                        Err(e) => {
                            std::mem::swap(&mut p.state, &mut seq.program_state);
                            return Err(e);
                        }
                    }
                }
                std::mem::swap(&mut p.state, &mut seq.program_state);
                seq.len = start_pos + tokens.len();
                return Ok(logits);
            }
        }
        self.text.truncate(&mut seq.layers, start_pos)?;
        let (n, d) = (tokens.len(), self.text.hidden());
        let mut h = vec![0.0f32; n * d];
        for (&t, row) in tokens.iter().zip(h.chunks_exact_mut(d)) {
            self.text.embed(t, row)?;
        }
        let pos: Vec<[u32; 3]> = (start_pos..start_pos + n).map(|p| [p as u32; 3]).collect();
        let h = self.text.forward_rows(
            h,
            &pos,
            start_pos,
            &mut seq.layers,
            0,
            usize::MAX,
            &[],
            &[],
            false,
            &seq.taps,
            &mut seq.tapped,
        );
        seq.len = start_pos + n;
        Ok(Tensor::F32(if all {
            self.text.logits(&h, n)
        } else {
            self.text.logits(&h[(n - 1) * d..], 1)
        }))
    }
}

struct Qwen35 {
    tok: Tokenizer,
    text: Arc<TextModel>,
    /// The vision tower, its preprocessor and the image placeholder id; `None` for text-only
    /// checkpoints.
    vision: Option<(Vision, Preprocessor, u32)>,
    drafter: Option<Arc<dyn Drafter>>,
    program: Option<Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>>,
}

/// No image feature at this prompt position.
const TEXT_ROW: u32 = u32::MAX;

/// One sequence: decoder state plus the prompt's image features and M-RoPE positions.
struct Seq {
    layers: Vec<LayerState>,
    program_state: crate::graph::StateStore,
    /// Positions already run.
    len: usize,
    /// (t, h, w) rotary positions of the prompt; empty for text-only prompts (all three = index).
    pos3: Vec<[u32; 3]>,
    /// Position offset for tokens past the prompt (`get_rope_index`'s `mrope_position_deltas`).
    delta: i64,
    /// Per prompt position: row of `feats` replacing the token embedding, or [`TEXT_ROW`].
    feat_row: Vec<u32>,
    /// Merged image features `[rows, hidden]`, all images in prompt order.
    feats: Vec<f32>,
    /// Qwen3-VL deepstack features, one `[rows, hidden]` buffer per injected decoder layer.
    deepstack: Vec<Vec<f32>>,
    /// Decoder layers whose outputs every forward records into `tapped` (see `set_taps`).
    taps: Vec<usize>,
    tapped: Vec<f32>,
    /// Program path: recurrent states after each position of the last checkpointing forward
    /// (`ckpt[i]` follows position `ckpt_from + i`), for `truncate`.
    ckpt: Vec<std::collections::HashMap<String, crate::graph::Tensor>>,
    ckpt_from: usize,
}

impl NativeLm for Qwen35 {
    fn backend_label(&self) -> String {
        if let Some(prog) = &self.program {
            return prog.lock().backend_label();
        }
        crate::scheduler::cpu_backend_label()
    }

    fn device(&self) -> Device {
        if let Some(prog) = &self.program {
            return prog.lock().primary_device();
        }
        Device::Cpu
    }

    fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        Some(self.program.as_ref()?.lock().placement_summary())
    }

    fn summary(&self) -> String {
        let mtp = if self.drafter.is_some() {
            ", MTP head"
        } else {
            ""
        };
        match &self.vision {
            Some((v, ..)) => format!("{}{mtp}, {}", self.text.summary(), v.summary()),
            None => format!("{}{mtp}, text only", self.text.summary()),
        }
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn context_len(&self) -> usize {
        if self.program.is_some() {
            self.text.decode_max_ctx
        } else {
            self.text.context_len()
        }
    }

    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        let (images, _) = self.size_images(&ids, media)?;
        let mut seq = Seq {
            // The program path keeps its state in `program_state`.
            layers: if self.program.is_some() {
                Vec::new()
            } else {
                self.text.new_state()
            },
            len: 0,
            program_state: crate::graph::StateStore::new(),
            pos3: Vec::new(),
            delta: 0,
            feat_row: Vec::new(),
            feats: Vec::new(),
            deepstack: Vec::new(),
            taps: Vec::new(),
            tapped: Vec::new(),
            ckpt: Vec::new(),
            ckpt_from: 0,
        };
        if images.is_empty() {
            return Ok((ids, Box::new(seq)));
        }
        let (vision, pre, image_token) = self
            .vision
            .as_ref()
            .expect("size_images refuses images without a vision tower");
        let mut encoded = images
            .iter()
            .map(|img| vision.encode(img, pre))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        seq.deepstack = vec![Vec::new(); vision.deepstack_layers()];
        let mut out = Vec::with_capacity(ids.len());
        let mut cur = 0u32;
        for &t in &ids {
            if t != *image_token {
                out.push(t);
                seq.pos3.push([cur; 3]);
                seq.feat_row.push(TEXT_ROW);
                cur += 1;
                continue;
            }
            // `get_rope_index`: t = start, h = start + row, w = start + col over the merged grid,
            // then text resumes at start + max(rows, cols).
            let e = encoded.next().expect("one encoding per placeholder");
            let first = (seq.feats.len() / self.text.hidden()) as u32;
            for r in 0..e.rows {
                for c in 0..e.cols {
                    out.push(t);
                    seq.pos3.push([cur, cur + r as u32, cur + c as u32]);
                    seq.feat_row.push(first + (r * e.cols + c) as u32);
                }
            }
            cur += e.rows.max(e.cols) as u32;
            seq.feats.extend_from_slice(&e.embeds);
            for (d, f) in seq.deepstack.iter_mut().zip(&e.deepstack) {
                d.extend_from_slice(f);
            }
        }
        seq.delta = cur as i64 - out.len() as i64;
        Ok((out, Box::new(seq)))
    }

    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        Ok(self.size_images(&ids, media)?.1)
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        crate::model::native::logits_of(self.run(tokens, start_pos, state, false, false)?)
    }

    fn forward_greedy(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<crate::graph::Tensor, String> {
        self.run(tokens, start_pos, state, false, true)
    }

    fn forward_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        if self.text.is_hc() {
            return Err("this model cannot verify draft tokens".into());
        }
        crate::model::native::logits_of(self.run(tokens, start_pos, state, true, false)?)
    }

    fn truncate(&self, state: &mut SeqState, len: usize) -> Result<(), String> {
        let seq = Self::seq(state)?;
        if len > seq.len {
            return Err(format!("cannot truncate {} positions to {len}", seq.len));
        }
        if len == seq.len {
            self.text.drop_checkpoints(&mut seq.layers);
            seq.ckpt.clear();
        } else if self
            .program
            .as_ref()
            .is_some_and(|p| p.lock().recurrent_states().is_empty())
        {
            seq.len = len; // attention-only: KV rows past `len` are overwritten
        } else if self.program.is_some() {
            // Full KV needs nothing (attention reads only the rows below the position it is
            // given); the recurrent states come back from the snapshot taken after position len-1.
            let held = seq.ckpt.len();
            if len <= seq.ckpt_from || len > seq.ckpt_from + held {
                return Err(format!(
                    "recurrent state rewinds only into its last verified span {}..={}, not to {len}",
                    seq.ckpt_from + 1,
                    seq.ckpt_from + held
                ));
            }
            let snap = seq.ckpt.swap_remove(len - seq.ckpt_from - 1);
            seq.program_state.restore(snap);
            seq.ckpt.clear();
            seq.len = len;
        } else {
            self.text.truncate(&mut seq.layers, len)?;
            seq.len = len;
        }
        Ok(())
    }

    fn set_taps(&self, state: &mut SeqState, layers: &[usize]) -> Result<(), String> {
        if self.text.is_hc() {
            return Err("this model cannot expose hidden states to a drafter".into());
        }
        if let Some(&l) = layers.iter().find(|&&l| l >= self.text.layers()) {
            return Err(format!(
                "tap layer {l}: the decoder has {} layers",
                self.text.layers()
            ));
        }
        // The decode program outputs one tap: the final-normed hidden row, which is what the MTP
        // head reads (its host twin taps the last layer and norms it itself).
        if self.program.is_some() && layers != [self.text.layers() - 1] {
            return Err(format!(
                "the Metal program taps only the last decoder layer's final-normed output, not {layers:?}"
            ));
        }
        Self::seq(state)?.taps = layers.to_vec();
        Ok(())
    }

    fn take_taps(&self, state: &mut SeqState) -> Vec<f32> {
        Self::seq(state).map_or_else(|_| Vec::new(), |s| std::mem::take(&mut s.tapped))
    }

    fn drafter(&self) -> Option<Arc<dyn Drafter>> {
        self.drafter.clone()
    }
}

impl Qwen35 {
    fn seq(state: &mut SeqState) -> Result<&mut Seq, String> {
        state
            .downcast_mut::<Seq>()
            .ok_or_else(|| "sequence state was not built by this model".into())
    }

    /// The program path of [`Self::run`]: `p.state` is the sequence's state. Image positions feed
    /// their feature row for the token embedding, the `[t, h, w]` rope positions and, per deepstack
    /// layer, the row added after it. `all` snapshots the recurrent states `refs` after every
    /// position for [`NativeLm::truncate`]; `greedy` runs the last position through
    /// [`crate::graph::program::GraphProgram::run_greedy`].
    #[allow(clippy::too_many_arguments)]
    fn program_steps(
        &self,
        p: &mut crate::graph::program::GraphProgram,
        seq: &mut Seq,
        tokens: &[u32],
        start_pos: usize,
        all: bool,
        greedy: bool,
        refs: &[&str],
    ) -> Result<crate::graph::Tensor, String> {
        use crate::graph::Tensor;
        let d = self.text.hidden();
        let ds_layers = self
            .vision
            .as_ref()
            .map_or(0, |(v, ..)| v.deepstack_layers());
        let (mut logits, mut picked) = (Vec::new(), None);
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            if pos >= self.text.decode_max_ctx {
                return Err(format!(
                    "position {pos} exceeds compiled decode context limit {}",
                    self.text.decode_max_ctx
                ));
            }
            let feat = seq.feat_row.get(pos).copied().filter(|&r| r != TEXT_ROW);
            let mut emb = vec![0.0f32; d];
            match feat {
                Some(r) => emb.copy_from_slice(&seq.feats[r as usize * d..][..d]),
                None => self.text.embed(tok, &mut emb)?,
            }
            let mut inputs = vec![Tensor::F32(emb), Tensor::U32(vec![pos as u32])];
            if self.vision.is_some() {
                let p3 = seq
                    .pos3
                    .get(pos)
                    .copied()
                    .unwrap_or([(pos as i64 + seq.delta) as u32; 3]);
                inputs.push(Tensor::U32(p3.to_vec()));
                for k in 0..ds_layers {
                    let mut row = vec![0.0f32; d];
                    if let (Some(r), Some(f)) = (feat, seq.deepstack.get(k)) {
                        row.copy_from_slice(&f[r as usize * d..][..d]);
                    }
                    inputs.push(Tensor::F32(row));
                }
            }
            let last = i + 1 == tokens.len();
            let mut res = if greedy && last {
                p.run_greedy(inputs, self.tok.vocab_size())?
            } else {
                p.run(inputs)?
            }
            .into_iter();
            match res.next() {
                Some(Tensor::U32(t)) => picked = t.first().copied(),
                Some(Tensor::F32(v)) if all || last => logits.extend_from_slice(&v),
                _ => {}
            }
            if !seq.taps.is_empty() {
                match res.next() {
                    Some(Tensor::F32(h)) => seq.tapped.extend_from_slice(&h),
                    _ => return Err("the decode program has no hidden-state output".into()),
                }
            }
            if all {
                seq.ckpt.push(p.state.snapshot(refs));
            }
        }
        Ok(match picked {
            Some(t) => Tensor::U32(vec![t]),
            None => Tensor::F32(logits),
        })
    }

    /// Runs `tokens` from `start_pos`; logits after the last token, or after every token (with
    /// recurrent checkpoints for `truncate`) when `all`. `greedy`: the program may return the last
    /// position's device-picked token instead (`U32([token])`).
    fn run(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
        all: bool,
        greedy: bool,
    ) -> Result<crate::graph::Tensor, String> {
        let seq = Self::seq(state)?;
        if start_pos != seq.len {
            return Err(format!(
                "Qwen3.5 state holds {} positions; cannot run from position {start_pos} (recurrent state does not rewind)",
                seq.len
            ));
        }
        if let Some(prog) = &self.program {
            // ponytail: one token per program step (prefill included); an n-row prefill graph is the upgrade.
            // Consecutive checkpointing forwards extend one span; anything else starts over.
            if !all || seq.ckpt.is_empty() || seq.ckpt_from + seq.ckpt.len() != start_pos {
                seq.ckpt.clear();
                seq.ckpt_from = start_pos;
            }
            let mut p = prog.lock();
            let names = p.recurrent_states();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            std::mem::swap(&mut p.state, &mut seq.program_state);
            let out = self.program_steps(&mut p, seq, tokens, start_pos, all, greedy, &refs);
            std::mem::swap(&mut p.state, &mut seq.program_state);
            if out.is_ok() {
                seq.len = start_pos + tokens.len();
            }
            return out;
        }
        if tokens.is_empty() {
            return Err("forward needs at least one token".into());
        }
        let d = self.text.hidden();
        let mut h = vec![0.0f32; tokens.len() * d];
        let mut pos = Vec::with_capacity(tokens.len());
        let mut visual = Vec::new();
        for (i, (&t, row)) in tokens.iter().zip(h.chunks_exact_mut(d)).enumerate() {
            let p = start_pos + i;
            match seq.feat_row.get(p) {
                Some(&r) if r != TEXT_ROW => {
                    row.copy_from_slice(&seq.feats[r as usize * d..][..d]);
                    visual.push((i, r as usize));
                }
                _ => self.text.embed(t, row)?,
            }
            pos.push(match seq.pos3.get(p) {
                Some(&p3) => p3,
                None => [(p as i64 + seq.delta) as u32; 3],
            });
        }
        self.text.set_ple_tokens(&mut seq.layers, tokens);
        let h = self.text.forward_rows(
            h,
            &pos,
            start_pos,
            &mut seq.layers,
            0,
            usize::MAX,
            &visual,
            &seq.deepstack,
            all,
            &seq.taps,
            &mut seq.tapped,
        );
        seq.len += tokens.len();
        let n = tokens.len();
        let w = self.text.width();
        Ok(crate::graph::Tensor::F32(if all {
            self.text.logits(&h, n)
        } else {
            self.text.logits(&h[(n - 1) * w..], 1)
        }))
    }
}

impl Qwen35 {
    /// Decodes the prompt's images and sizes them (no vision tower): the images and the prompt's
    /// length once each placeholder expands to its merged grid. Errs on a placeholder/image count
    /// mismatch, audio, or a prompt the context can't hold.
    fn size_images(
        &self,
        ids: &[u32],
        media: &[MediaInput],
    ) -> Result<(Vec<ImageBuffer>, usize), String> {
        let Some((vision, pre, image_token)) = &self.vision else {
            if !media.is_empty() {
                return Err("this checkpoint has no vision tower: text input only".into());
            }
            return Ok((Vec::new(), ids.len()));
        };
        let images = media
            .iter()
            .map(|m| match m {
                MediaInput::Image(b) => ImageBuffer::decode(b),
                MediaInput::Audio(_) => Err("this model takes text and images, not audio".into()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pads = ids.iter().filter(|&&t| t == *image_token).count();
        if pads != images.len() {
            return Err(format!(
                "prompt has {pads} image placeholder(s) for {} image(s)",
                images.len()
            ));
        }
        let mut total = ids.len() - pads;
        for img in &images {
            let (r, c) = vision.grid(img, pre)?;
            total += r * c;
        }
        if !images.is_empty() && total >= self.context_len() {
            return Err(format!(
                "prompt with its {} image(s) is {total} tokens; the context holds {}",
                images.len(),
                self.context_len()
            ));
        }
        Ok((images, total))
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// `prefixes[i].weight` stacked along the output dim (one pass for several projections),
/// borrowed from the mapped checkpoint, with `.bias` when `bias`.
fn linear(w: &Weights, prefixes: &[&str], bias: bool) -> Result<Linear, String> {
    let (mut parts, mut b) = (Vec::with_capacity(prefixes.len()), Vec::new());
    for p in prefixes {
        let m = w.mat(&format!("{p}.weight"))?;
        if let Some(first) = parts.first().map(|f: &Mat| f.cols) {
            if m.cols != first {
                return Err(format!("{p}.weight: {} inputs, expected {first}", m.cols));
            }
        }
        if bias {
            b.extend(w.vec_n(&format!("{p}.bias"), m.rows)?);
        }
        parts.push(m);
    }
    Ok(Linear {
        rows: parts.iter().map(|m| m.rows).sum(),
        cols: parts.first().map_or(0, |m| m.cols),
        parts,
        bias: bias.then_some(b),
    })
}

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

/// `y = x W^T (+ b)` with `W` the bf16 `parts` stacked along the output dim, `[rows, cols]`.
pub(crate) struct Linear {
    pub(crate) parts: Vec<Mat>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bias: Option<Vec<f32>>,
}

impl Linear {
    /// `x` `[n, cols]` -> `[n, rows]`.
    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        if self.parts.iter().any(|m| m.packed().is_some()) {
            let mut out = vec![0.0f32; n * self.rows];
            if n == 1 && self.parts.len() == 1 {
                self.parts[0].gemv_f32(x, &mut out);
                if let Some(b) = &self.bias {
                    for r in 0..self.rows {
                        out[r] += b[r];
                    }
                }
                return out;
            }
            let mut row_offset = 0;
            for part in &self.parts {
                let p_rows = part.rows;
                let w_f32 = part.to_f32();
                for sample in 0..n {
                    let xs = &x[sample * self.cols..(sample + 1) * self.cols];
                    for r in 0..p_rows {
                        let wr = &w_f32[r * self.cols..(r + 1) * self.cols];
                        let mut dot = 0.0f32;
                        for c in 0..self.cols {
                            dot += xs[c] * wr[c];
                        }
                        out[sample * self.rows + row_offset + r] = dot;
                    }
                }
                row_offset += p_rows;
            }
            if let Some(b) = &self.bias {
                for sample in 0..n {
                    for r in 0..self.rows {
                        out[sample * self.rows + r] += b[r];
                    }
                }
            }
            out
        } else {
            let parts: Vec<&[u16]> = self.parts.iter().map(Mat::data).collect();
            linear_bf16(&parts, self.cols, self.bias.as_deref(), x, n)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn to_graph_tensor(&self) -> crate::graph::Tensor {
        if self.parts.len() == 1 {
            self.parts[0].to_graph_tensor()
        } else {
            let mut all = Vec::with_capacity(self.rows * self.cols);
            for p in &self.parts {
                all.extend(p.to_f32());
            }
            crate::graph::Tensor::F32(all)
        }
    }

    /// `Op::Linear` of `x` `[n, cols]` through this single-matrix weight (and bias) as graph
    /// weights `{name}.weight` / `{name}.bias`.
    pub(crate) fn graph_node(
        &self,
        g: &mut crate::graph::Graph,
        w: &mut crate::graph::Weights,
        name: &str,
        x: crate::graph::ValueId,
        n: usize,
    ) -> crate::graph::ValueId {
        use crate::graph::{DType, Op, Tensor};
        use crate::hardware::QuantFormat as Q;
        let [m] = &self.parts[..] else {
            panic!("{name}: graph_node needs a single-matrix linear");
        };
        let dtype = match m.packed().map(|p| p.format) {
            Some(Q::FP32) => DType::F32,
            Some(Q::FP16) => DType::F16,
            Some(Q::BF16) | None => DType::BF16,
            Some(f) => DType::Quant(f),
        };
        let wn = g.weight(format!("{name}.weight"), vec![self.rows, self.cols], dtype);
        w.insert(format!("{name}.weight"), m.to_graph_tensor());
        let mut ins = vec![x, wn];
        if let Some(b) = &self.bias {
            ins.push(g.weight(format!("{name}.bias"), vec![b.len()], DType::F32));
            w.insert(format!("{name}.bias"), Tensor::F32(b.clone()));
        }
        g.add(name, Op::Linear, &ins, vec![n, self.rows], DType::F32)
    }
}

/// Weight rows one task converts to f32 and multiplies against every input row.
const ROW_BLOCK: usize = 16;
/// Weight rows per task when there is a single input row (memory-bound GEMV).
const GEMV_BLOCK: usize = 64;

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `x` `[n, k]` times the row-stacked bf16 `parts` (each `[rows_p, k]`) transposed, plus `bias`
/// -> `[n, sum rows_p]`: NEON GEMV for up to [`EXACT_ROWS`] rows (each row's result is
/// bit-identical to the one-row GEMV, so speculative verify reproduces plain decode exactly),
/// 4x4 register tiles over f32-converted weight blocks for more.
fn linear_bf16(parts: &[&[u16]], k: usize, bias: Option<&[f32]>, x: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), n * k, "linear input is not n x cols");
    let rows: usize = parts.iter().map(|p| p.len() / k.max(1)).sum();
    let mut out = vec![0.0f32; n * rows];
    let o = OutPtr::new(&mut out);
    let gemv = n <= EXACT_ROWS;
    let block = if gemv { GEMV_BLOCK } else { ROW_BLOCK };
    // (part, first row within it, first output column, row count): blocks never span parts.
    let mut tasks = Vec::new();
    let mut g = 0;
    for (pi, p) in parts.iter().enumerate() {
        let pr = p.len() / k.max(1);
        for r0 in (0..pr).step_by(block) {
            tasks.push((pi, r0, g + r0, block.min(pr - r0)));
        }
        g += pr;
    }
    par_for(tasks.len(), &|t| {
        let (pi, r0, c0, rb) = tasks[t];
        let w = &parts[pi][r0 * k..(r0 + rb) * k];
        // SAFETY: task t alone writes columns c0..c0+rb of every output row.
        let put = |i: usize, j: usize, v: f32| unsafe { o.slice(i * rows + c0 + j, 1)[0] = v };
        if gemv {
            let xr = |i: usize| &x[i * k..][..k];
            for j in 0..rb {
                let wj = &w[j * k..][..k];
                let emit = |i: usize, r: &[f32]| {
                    for (a, &v) in r.iter().enumerate() {
                        put(i + a, j, v);
                    }
                };
                for i in (0..n).step_by(4) {
                    match n - i {
                        1 => put(i, j, dot_bf16(wj, xr(i))),
                        2 => emit(i, &dot_bf16_rows(wj, [xr(i), xr(i + 1)])),
                        3 => emit(i, &dot_bf16_rows(wj, [xr(i), xr(i + 1), xr(i + 2)])),
                        _ => emit(
                            i,
                            &dot_bf16_rows(wj, [xr(i), xr(i + 1), xr(i + 2), xr(i + 3)]),
                        ),
                    }
                }
            }
        } else {
            SCRATCH.with_borrow_mut(|wf| {
                wf.resize(rb * k, 0.0);
                for (d, &s) in wf.iter_mut().zip(w) {
                    *d = f32::from_bits((s as u32) << 16);
                }
                let wr = |j: usize| &wf[j * k..][..k];
                let xr = |i: usize| &x[i * k..][..k];
                let (n4, rb4) = (n / 4 * 4, rb / 4 * 4);
                for i in (0..n4).step_by(4) {
                    for j in (0..rb4).step_by(4) {
                        let tile = tile4x4(
                            [xr(i), xr(i + 1), xr(i + 2), xr(i + 3)],
                            [wr(j), wr(j + 1), wr(j + 2), wr(j + 3)],
                            k,
                        );
                        for (a, row) in tile.iter().enumerate() {
                            for (b, &v) in row.iter().enumerate() {
                                put(i + a, j + b, v);
                            }
                        }
                    }
                    for j in rb4..rb {
                        for a in 0..4 {
                            put(i + a, j, vector_dot(xr(i + a), wr(j)));
                        }
                    }
                }
                for i in n4..n {
                    for j in 0..rb {
                        put(i, j, vector_dot(xr(i), wr(j)));
                    }
                }
            });
        }
        if let Some(b) = bias {
            for i in 0..n {
                for j in 0..rb {
                    // SAFETY: as above.
                    unsafe { o.slice(i * rows + c0 + j, 1)[0] += b[c0 + j] };
                }
            }
        }
    });
    out
}

use crate::model::simd::vector_dot_product as vector_dot;

/// Input rows up to which [`linear_bf16`] runs the exact GEMV path (a verify pass of up to 15
/// draft tokens).
const EXACT_ROWS: usize = 16;

/// [`dot_bf16`] of one weight row against `R` input rows, reading the weights once; every result
/// is bit-identical to `dot_bf16(w, x[r])` (same lanes, FMA order, reduction tree and tail).
#[cfg(target_arch = "aarch64")]
fn dot_bf16_rows<const R: usize>(w: &[u16], x: [&[f32]; R]) -> [f32; R] {
    use core::arch::aarch64::*;
    let n = w.len();
    assert!(x.iter().all(|s| s.len() >= n));
    let mut out = [0.0f32; R];
    let mut i = 0;
    // SAFETY: every load is at i + 16 <= n, bounded by the assert above.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; R];
        while i + 16 <= n {
            let a = vld1q_u16(w.as_ptr().add(i));
            let b = vld1q_u16(w.as_ptr().add(i + 8));
            let wv = [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(a))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(a)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(b))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(b)),
            ];
            for (acc, xs) in acc.iter_mut().zip(&x) {
                for q in 0..4 {
                    acc[q] = vfmaq_f32(acc[q], wv[q], vld1q_f32(xs.as_ptr().add(i + 4 * q)));
                }
            }
            i += 16;
        }
        for (o, acc) in out.iter_mut().zip(&acc) {
            *o = vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ));
        }
    }
    for (o, xs) in out.iter_mut().zip(&x) {
        for j in i..n {
            *o += crate::model::bf16_to_f32(w[j]) * xs[j];
        }
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_bf16_rows<const R: usize>(w: &[u16], x: [&[f32]; R]) -> [f32; R] {
    x.map(|xs| dot_bf16(w, &xs[..w.len()]))
}

/// `out[a][b] = dot(x[a], w[b])` over the first `k` elements: 16 accumulators, 8 loads per 16 FMAs.
#[cfg(target_arch = "aarch64")]
fn tile4x4(x: [&[f32]; 4], w: [&[f32]; 4], k: usize) -> [[f32; 4]; 4] {
    use core::arch::aarch64::*;
    assert!(x.iter().chain(&w).all(|s| s.len() >= k));
    let mut out = [[0.0f32; 4]; 4];
    let mut p = 0;
    // SAFETY: every load is at p + 4 <= k, bounded by the assert above.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
        while p + 4 <= k {
            let xv = x.map(|s| vld1q_f32(s.as_ptr().add(p)));
            let wv = w.map(|s| vld1q_f32(s.as_ptr().add(p)));
            for a in 0..4 {
                for b in 0..4 {
                    acc[a][b] = vfmaq_f32(acc[a][b], xv[a], wv[b]);
                }
            }
            p += 4;
        }
        for a in 0..4 {
            for b in 0..4 {
                out[a][b] = vaddvq_f32(acc[a][b]);
            }
        }
    }
    for a in 0..4 {
        for b in 0..4 {
            for q in p..k {
                out[a][b] += x[a][q] * w[b][q];
            }
        }
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
fn tile4x4(x: [&[f32]; 4], w: [&[f32]; 4], k: usize) -> [[f32; 4]; 4] {
    x.map(|xa| w.map(|wb| vector_dot(&xa[..k], &wb[..k])))
}

/// RMSNorm of every row in place (`weight` already includes Qwen3.5's `1 +` offset).
fn rms_norm_rows(x: &mut [f32], weight: &[f32], eps: f32) {
    let d = weight.len();
    par_rows(x, d, &|_, row| rms_norm(row, weight, eps));
}

fn rms_norm(row: &mut [f32], weight: &[f32], eps: f32) {
    let inv = 1.0 / (vector_dot(row, row) / row.len() as f32 + eps).sqrt();
    for (v, w) in row.iter_mut().zip(weight) {
        *v *= inv * w;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `qwen3moe` metadata becomes a Qwen3 decoder with routed experts (no dense FFN width), and its
    /// router / expert tensors map to the names `Moe::load` reads.
    #[test]
    fn qwen3moe_gguf_config_and_tensor_names() {
        let md: std::collections::HashMap<String, String> = [
            ("general.architecture", "qwen3moe"),
            ("qwen3moe.embedding_length", "2048"),
            ("qwen3moe.attention.head_count", "32"),
            ("qwen3moe.attention.head_count_kv", "4"),
            ("qwen3moe.attention.key_length", "128"),
            ("qwen3moe.attention.layer_norm_rms_epsilon", "0.000001"),
            ("qwen3moe.rope.freq_base", "10000000"),
            ("qwen3moe.context_length", "262144"),
            ("qwen3moe.block_count", "48"),
            ("qwen3moe.feed_forward_length", "6144"),
            ("qwen3moe.expert_count", "128"),
            ("qwen3moe.expert_used_count", "8"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let m = crate::model::ModelManifest {
            metadata: md,
            ..Default::default()
        };
        let (cfg, arch, layers) = gguf_config(&m).unwrap();
        assert!(arch == Arch::Qwen3 && layers == 48);
        assert_eq!(cfg["intermediate_size"], 0);
        assert_eq!(
            (
                cfg["num_experts"].as_u64(),
                cfg["num_experts_per_tok"].as_u64()
            ),
            (Some(128), Some(8))
        );
        assert_eq!(cfg["head_dim"], 128);
        for (gguf, hf) in [
            ("ffn_gate_inp.weight", "mlp.gate.weight"),
            ("ffn_gate_exps.weight", "mlp.experts.gate_proj"),
            ("ffn_up_exps.weight", "mlp.experts.up_proj"),
            ("ffn_down_exps.weight", "mlp.experts.down_proj"),
            ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ] {
            assert_eq!(
                gguf_to_hf(Arch::Qwen3, &format!("blk.5.{gguf}")).unwrap(),
                format!("model.layers.5.{hf}")
            );
        }
    }

    /// A tiny `qwen3moe` GGUF with Q4_0 expert banks, through the real GGUF path: the banks stay
    /// packed in the mapping, and the Metal program (banks read in place by `MoeLinear`) matches the
    /// host forward (rows decoded from the same mapping).
    #[test]
    fn qwen3moe_gguf_banks_stay_packed_and_match_host() {
        use crate::model::layers::quantize_f32_to_q4_0;
        let (hidden, vocab, heads, kv_heads, hd) = (64usize, 128usize, 4usize, 2usize, 16usize);
        let (experts, top_k, ff, layers) = (8usize, 2usize, 32usize, 2usize);
        let mut state = 0x2545F4914F6CDD1Du64;
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
        // (name, GGML dims fastest first, type, bytes): F32 = 0, Q4_0 = 2.
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        let f32s = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut add_f32 = |n: &str, dims: &[usize], v: Vec<f32>| {
            tensors.push((n.into(), dims.to_vec(), 0, f32s(v)))
        };
        add_f32(
            "token_embd.weight",
            &[hidden, vocab],
            rnd(hidden * vocab, 0.8),
        );
        add_f32(
            "output_norm.weight",
            &[hidden],
            rnd(hidden, 0.2).iter().map(|v| v + 1.0).collect(),
        );
        add_f32("output.weight", &[hidden, vocab], rnd(hidden * vocab, 0.8));
        for l in 0..layers {
            let p = format!("blk.{l}");
            for n in ["attn_norm", "ffn_norm"] {
                add_f32(
                    &format!("{p}.{n}.weight"),
                    &[hidden],
                    rnd(hidden, 0.2).iter().map(|v| v + 1.0).collect(),
                );
            }
            add_f32(
                &format!("{p}.attn_q.weight"),
                &[hidden, heads * hd],
                rnd(hidden * heads * hd, 0.8),
            );
            add_f32(
                &format!("{p}.attn_k.weight"),
                &[hidden, kv_heads * hd],
                rnd(hidden * kv_heads * hd, 0.8),
            );
            add_f32(
                &format!("{p}.attn_v.weight"),
                &[hidden, kv_heads * hd],
                rnd(hidden * kv_heads * hd, 0.8),
            );
            add_f32(
                &format!("{p}.attn_output.weight"),
                &[heads * hd, hidden],
                rnd(hidden * heads * hd, 0.8),
            );
            for n in ["attn_q_norm", "attn_k_norm"] {
                add_f32(
                    &format!("{p}.{n}.weight"),
                    &[hd],
                    rnd(hd, 0.2).iter().map(|v| v + 1.0).collect(),
                );
            }
            add_f32(
                &format!("{p}.ffn_gate_inp.weight"),
                &[hidden, experts],
                rnd(hidden * experts, 1.6),
            );
        }
        let mut bank = |n: String, dims: &[usize], v: Vec<f32>| {
            tensors.push((n, dims.to_vec(), 2, quantize_f32_to_q4_0(&v)))
        };
        for l in 0..layers {
            let p = format!("blk.{l}");
            bank(
                format!("{p}.ffn_gate_exps.weight"),
                &[hidden, ff, experts],
                rnd(experts * ff * hidden, 0.8),
            );
            bank(
                format!("{p}.ffn_up_exps.weight"),
                &[hidden, ff, experts],
                rnd(experts * ff * hidden, 0.8),
            );
            bank(
                format!("{p}.ffn_down_exps.weight"),
                &[ff, hidden, experts],
                rnd(experts * hidden * ff, 0.8),
            );
        }
        let u32_kv = |k: &str, v: u32| (format!("qwen3moe.{k}"), 4u32, v.to_le_bytes().to_vec());
        let f32_kv = |k: &str, v: f32| (format!("qwen3moe.{k}"), 6u32, v.to_le_bytes().to_vec());
        let mut arch_name = (8u64).to_le_bytes().to_vec();
        arch_name.extend_from_slice(b"qwen3moe");
        let kvs = vec![
            ("general.architecture".to_string(), 8u32, arch_name),
            u32_kv("embedding_length", hidden as u32),
            u32_kv("attention.head_count", heads as u32),
            u32_kv("attention.head_count_kv", kv_heads as u32),
            u32_kv("attention.key_length", hd as u32),
            f32_kv("attention.layer_norm_rms_epsilon", 1e-6),
            f32_kv("rope.freq_base", 10000.0),
            u32_kv("context_length", 64),
            u32_kv("block_count", layers as u32),
            u32_kv("feed_forward_length", 96),
            u32_kv("expert_count", experts as u32),
            u32_kv("expert_used_count", top_k as u32),
        ];
        let file = crate::model::loaders::gguf::test_gguf_blocks(&kvs, &tensors);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny-qwen3moe.gguf");
        std::fs::write(&path, &file).unwrap();

        let (manifest, data) = crate::model::loaders::gguf::open(&path).unwrap();
        let (ck, cfg, arch) = gguf_checkpoint(&Arc::new(data), &manifest).unwrap();
        for part in ["gate_proj", "up_proj", "down_proj"] {
            assert!(
                ck.packed_gguf(&format!("model.layers.1.mlp.experts.{part}"))
                    .is_some(),
                "{part} bank stays packed in the file mapping"
            );
        }
        let text = Arc::new(TextModel::load(&ck, &cfg, arch).unwrap());
        let program = compile_qwen_program(
            &text,
            crate::hardware::PlacementMode::Pin(Device::Metal),
            text::DecodeOpts::default(),
        )
        .unwrap()
        .expect("a Metal program");
        let mut prog = program.lock();
        assert_eq!(prog.cpu_node_count(), 0);
        let mut host_state = text.new_state();
        let mut tok = 11u32;
        for pos in 0..6usize {
            let mut emb = vec![0.0f32; hidden];
            text.embed(tok, &mut emb).unwrap();
            let hid = text.forward_rows(
                emb.clone(),
                &[[pos as u32; 3]],
                pos,
                &mut host_state,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let want = text.logits(&hid, 1);
            let got = prog
                .run(vec![
                    crate::graph::Tensor::F32(emb),
                    crate::graph::Tensor::U32(vec![pos as u32]),
                ])
                .unwrap();
            let got = got[0].as_f32().unwrap();
            let dot: f64 = want
                .iter()
                .zip(got)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let n = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (n(&want) * n(got));
            assert!(cos >= 0.999, "step {pos}: Metal vs host cosine {cos}");
            tok = want
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
        }
    }

    /// Hybrid Qwen3.5 stack (DeltaNet + gated GQA full attention, qk-norm, partial rotary): the
    /// Metal seq-cls program against the host forward. The text weights are bf16 on both sides and
    /// accumulate in f32, so the bound is the 1e-3 f32 one.
    #[test]
    fn seqcls_gpu_matches_host_qwen3_5() {
        let (text, h) = text::tiny_model_scaled(0.3);
        let mut llm = Dense {
            tok: crate::model::tokenizer::Tokenizer::default_bpe(),
            text: Arc::new(text),
            program: None,
        };
        let score = crate::model::classifier::decoder_gpu::fill(3 * h, 60, 0.5);
        let seqs: [&[u32]; 3] = [&[3, 17, 5], &[9, 8, 7, 6, 5, 4, 3, 2, 1, 100], &[42]];
        let e = crate::model::classifier::decoder_gpu::tests::parity(&mut llm, &score, 3, &seqs);
        eprintln!("qwen3.5 hybrid max rel err {e:?}");
    }

    /// Plain Qwen3 (all full-attention layers, GQA, qk-norm, full rotary, tied embeddings).
    #[test]
    fn seqcls_gpu_matches_host_qwen3() {
        let (h, vocab, heads, kv, hd, inter) = (64, 80, 4, 2, 16, 96);
        let cfg = serde_json::json!({
            "model_type": "qwen3", "hidden_size": h, "vocab_size": vocab,
            "num_hidden_layers": 2, "num_attention_heads": heads, "num_key_value_heads": kv,
            "head_dim": hd, "intermediate_size": inter, "max_position_embeddings": 64,
            "rope_theta": 10000.0, "tie_word_embeddings": true
        });
        let mut specs = vec![
            ("model.embed_tokens.weight".to_string(), vec![vocab, h]),
            ("model.norm.weight".to_string(), vec![h]),
        ];
        for l in 0..2 {
            let p = format!("model.layers.{l}");
            for (n, s) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![heads * hd, h]),
                ("self_attn.k_proj.weight", vec![kv * hd, h]),
                ("self_attn.v_proj.weight", vec![kv * hd, h]),
                ("self_attn.o_proj.weight", vec![h, heads * hd]),
                ("self_attn.q_norm.weight", vec![hd]),
                ("self_attn.k_norm.weight", vec![hd]),
                ("mlp.gate_proj.weight", vec![inter, h]),
                ("mlp.up_proj.weight", vec![inter, h]),
                ("mlp.down_proj.weight", vec![h, inter]),
            ] {
                specs.push((format!("{p}.{n}"), s));
            }
        }
        let sizes: std::collections::HashMap<_, usize> = specs
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().product()))
            .collect();
        let ck = Weights::from_f32(specs, |name| {
            let seed = name.bytes().map(|b| b as usize).sum::<usize>();
            let norm = name.contains("norm");
            Ok(crate::model::classifier::decoder_gpu::fill(
                sizes[name],
                seed,
                if norm { 0.2 } else { 0.3 },
            )
            .into_iter()
            .map(|v| if norm { v + 1.0 } else { v })
            .collect())
        })
        .unwrap();
        let mut llm = Dense {
            tok: crate::model::tokenizer::Tokenizer::default_bpe(),
            text: Arc::new(TextModel::load(&ck, &cfg, Arch::Qwen3).unwrap()),
            program: None,
        };
        let score = crate::model::classifier::decoder_gpu::fill(2 * h, 61, 0.5);
        let seqs: [&[u32]; 3] = [&[3, 17, 5], &[9, 8, 7, 6, 5, 4, 3, 2, 1, 70, 11, 12], &[42]];
        let e = crate::model::classifier::decoder_gpu::tests::parity(&mut llm, &score, 2, &seqs);
        eprintln!("qwen3 dense max rel err {e:?}");
    }

    #[test]
    fn linear_matches_reference_for_gemv_and_tiled_gemm() {
        let (rows, cols) = (37, 70);
        let w: Vec<u16> = (0..rows * cols)
            .map(|i| crate::model::f32_to_bf16(((i * 7919) % 13) as f32 / 7.0 - 0.9))
            .collect();
        let bias: Vec<f32> = (0..rows).map(|r| r as f32 * 0.1).collect();
        // Two stacked parts (20 + 17 rows): blocks restart at the part boundary.
        let parts = [&w[..20 * cols], &w[20 * cols..]];
        for n in [1, 2, 3, 9, 16, 21] {
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i * 31) % 17) as f32 / 9.0 - 1.0)
                .collect();
            let y = linear_bf16(&parts, cols, Some(&bias), &x, n);
            for i in 0..n {
                for r in 0..rows {
                    let want: f32 = bias[r]
                        + (0..cols)
                            .map(|c| {
                                f32::from_bits((w[r * cols + c] as u32) << 16) * x[i * cols + c]
                            })
                            .sum::<f32>();
                    let got = y[i * rows + r];
                    assert!(
                        (got - want).abs() < 1e-3,
                        "n={n} row {i} col {r}: {got} vs {want}"
                    );
                }
                // Up to EXACT_ROWS a row's output does not depend on its batch (speculative
                // verify must reproduce one-token decode bit for bit).
                if n <= EXACT_ROWS {
                    let alone = linear_bf16(&parts, cols, Some(&bias), &x[i * cols..][..cols], 1);
                    assert_eq!(alone, y[i * rows..][..rows], "n={n} row {i}");
                }
            }
        }
    }

    /// `<root>/models--<org>--<repo>/snapshots/main` for root `disk`, then `GIBBS_TEST_STORAGE`
    /// (a second cache, e.g. for GGUF downloads), or None when `file` is in neither.
    fn snapshot(repo: &str, file: &str) -> Option<std::path::PathBuf> {
        ["disk".to_string()]
            .into_iter()
            .chain(std::env::var("GIBBS_TEST_STORAGE"))
            .find_map(|root| {
                let dir = Path::new(&root)
                    .join(format!("models--{}", repo.replace('/', "--")))
                    .join("snapshots/main");
                dir.join(file).exists().then_some(dir)
            })
    }

    fn map_gguf(p: &Path) -> Arc<crate::model::loaders::gguf::GgufData> {
        let f = std::fs::File::open(p).unwrap();
        // SAFETY: read-only mapping of a cache file.
        Arc::new(crate::model::loaders::gguf::GgufData::Map(
            unsafe { memmap2::Mmap::map(&f) }.unwrap(),
        ))
    }

    fn dense(dir: &Path, arch: Arch) -> Dense {
        let cfg: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let ck = Weights::open(&dir.join("model.safetensors")).unwrap();
        Dense {
            tok: load_json_tokenizer(&dir.join("tokenizer.json")).unwrap(),
            text: Arc::new(TextModel::load(&ck, &cfg, arch).unwrap()),
            program: None,
        }
    }

    /// Greedy continuation of `ids`, up to `n` tokens or a generation end.
    fn greedy(lm: &dyn NativeLm, ids: Vec<u32>, n: usize) -> Vec<u32> {
        let (ids, mut st) = lm.prepare(ids, &[]).unwrap();
        let mut logits = lm.forward(&ids, 0, &mut st).unwrap();
        let mut out = Vec::new();
        for _ in 0..n {
            let t = argmax(&logits);
            out.push(t);
            if lm.tokenizer().ends_generation(t) {
                break;
            }
            logits = lm
                .forward(&[t], ids.len() + out.len() - 1, &mut st)
                .unwrap();
        }
        out
    }

    fn argmax(v: &[f32]) -> u32 {
        (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32
    }

    /// The chat-templated prompt (one user turn, `enable_thinking` as given) as token ids.
    fn chat_ids(tok: &Tokenizer, q: &str, thinking: bool) -> Vec<u32> {
        let msgs = [crate::model::hf_loader::ChatMessage::new("user", q)];
        let kwargs = serde_json::json!({ "enable_thinking": thinking });
        let spec = tok.chat_template.as_ref().unwrap();
        tok.encode(&spec.render(&msgs, None, true, &kwargs).unwrap())
    }

    /// Greedy chat answer with `enable_thinking` set as given.
    fn chat(lm: &dyn NativeLm, q: &str, thinking: bool, n: usize) -> String {
        let ids = chat_ids(lm.tokenizer(), q, thinking);
        lm.tokenizer().decode(&greedy(lm, ids, n))
    }

    /// Tokenizer parity strings: CJK, kana, digits runs, emoji, whitespace runs, accents, code.
    const PARITY_TEXTS: [&str; 12] = [
        "Hello, world! The quick brown fox jumps over the lazy dog.",
        "我喜欢在周末读书。今天天气很好，我们去公园散步吧！",
        "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：",
        "日本語のテキスト、カタカナとひらがな。",
        "Price: $1234567.89 (tax 20%), call 555-0123 or a1b22c333d4444",
        "Emoji 😀🎉👍🏽 and a flag 🇩🇪!",
        "  leading and    multiple   spaces\t\ttabs\n\n\nnewlines \r\n trailing   ",
        "Ünïcödé façade naïve café — résumé, e\u{301}cole",
        "def f(x):\n    return x**2  # square\n",
        "Привет, как дела? مرحبا 안녕하세요 สวัสดี",
        "don't won't I'm they're WE'LL it'S",
        "Translate the following segment into German, without additional explanation.\n\nThe cat sleeps.",
    ];

    /// Asserts `a` and `b` encode [`PARITY_TEXTS`] and `special` to the same ids.
    fn assert_same_ids(a: &Tokenizer, b: &Tokenizer, special: &str) {
        let mut ids = 0;
        for s in PARITY_TEXTS.iter().chain([&special]) {
            let x = a.encode(s);
            assert_eq!(x, b.encode(s), "{s:?}");
            ids += x.len();
        }
        let n = PARITY_TEXTS.len() + 1;
        eprintln!("tokenizer parity: {n} strings, {ids} ids identical");
    }

    /// Greedy decoding of `ids` on `a` and `b` in lockstep for up to `n` tokens: the tokens both
    /// pick, then at the first disagreement (if any) the picks `(ta, tb)` with each model's logit
    /// margin of its own pick over the other's.
    fn lockstep(
        a: &dyn NativeLm,
        b: &dyn NativeLm,
        ids: Vec<u32>,
        n: usize,
    ) -> (Vec<u32>, Option<(u32, u32, f32, f32)>) {
        let (ids, mut sa) = a.prepare(ids, &[]).unwrap();
        let (_, mut sb) = b.prepare(ids.clone(), &[]).unwrap();
        let mut la = a.forward(&ids, 0, &mut sa).unwrap();
        let mut lb = b.forward(&ids, 0, &mut sb).unwrap();
        let mut out = Vec::new();
        for _ in 0..n {
            let (ta, tb) = (argmax(&la), argmax(&lb));
            if ta != tb {
                let gap = |l: &[f32], x: u32, y: u32| l[x as usize] - l[y as usize];
                return (out, Some((ta, tb, gap(&la, ta, tb), gap(&lb, tb, ta))));
            }
            out.push(ta);
            if a.tokenizer().ends_generation(ta) {
                break;
            }
            let pos = ids.len() + out.len() - 1;
            la = a.forward(&[ta], pos, &mut sa).unwrap();
            lb = b.forward(&[ta], pos, &mut sb).unwrap();
        }
        (out, None)
    }

    /// Lossless relation: the bf16 GGUF of Qwen3-0.6B (same bf16 weights, llama.cpp names and
    /// tokenizer metadata) tokenizes and decodes greedily exactly like the safetensors repo.
    #[test]
    fn qwen3_safetensors_and_bf16_gguf_decode_alike() {
        let (Some(st), Some(gg)) = (
            snapshot("Qwen/Qwen3-0.6B", "model.safetensors"),
            snapshot("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-BF16.gguf"),
        ) else {
            eprintln!("SKIP: Qwen/Qwen3-0.6B or unsloth/Qwen3-0.6B-GGUF BF16 not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let st = dense(&st, Arch::Qwen3);
        let data = map_gguf(&gg.join("Qwen3-0.6B-BF16.gguf"));
        let m = crate::model::UniversalModelLoader::parse_gguf(&data).unwrap();
        let gg = gguf_model(&data, &m, Device::Cpu).unwrap();
        let prompt = "The theory of general relativity was developed by";
        let ids = st.tok.encode(prompt);
        assert_eq!(gg.tokenizer().encode(prompt), ids);
        let a = greedy(&st, ids.clone(), 12);
        assert_eq!(greedy(gg.as_ref(), ids, 12), a);
        let text = st.tok.decode(&a);
        assert!(text.starts_with(" Einstein"), "{text}");
    }

    #[test]
    fn qwen3_answers_with_thinking_on_and_off() {
        let Some(dir) = snapshot("Qwen/Qwen3-0.6B", "model.safetensors") else {
            eprintln!("SKIP: Qwen/Qwen3-0.6B not in disk/");
            return;
        };
        let lm = dense(&dir, Arch::Qwen3);
        let q = "What is the capital of France? Answer briefly.";
        let off = chat(&lm, q, false, 8);
        assert!(off.contains("Paris") && !off.contains("<think>"), "{off}");
        let on = chat(&lm, q, true, 6);
        assert!(on.starts_with("<think>"), "{on}");
    }

    /// Scalar f32 Qwen3 forward written straight from transformers' `modeling_qwen3.py`
    /// (`Qwen3Attention`: q/k RMSNorm per head, then rotate-half RoPE; tied embeddings), sharing
    /// no kernel with [`TextModel`]. Returns the logits after the last of `ids`.
    fn reference_logits(dir: &Path, ids: &[u32]) -> Vec<f32> {
        let c: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let w = Weights::open(&dir.join("model.safetensors")).unwrap();
        let u = |k: &str| c[k].as_u64().unwrap() as usize;
        let (d, nh, nkv, hd) = (
            u("hidden_size"),
            u("num_attention_heads"),
            u("num_key_value_heads"),
            u("head_dim"),
        );
        let eps = c["rms_norm_eps"].as_f64().unwrap() as f32;
        let theta = c["rope_theta"].as_f64().unwrap();
        let n = ids.len();
        let rms = |x: &[f32], g: &[f32]| -> Vec<f32> {
            let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
            x.iter().zip(g).map(|(v, g)| v / s * g).collect()
        };
        // x [n, k] @ W[out, k]^T
        let mm = |x: &[Vec<f32>], name: &str| -> Vec<Vec<f32>> {
            let m = w.mat(name).unwrap();
            let wf = m.to_f32();
            x.iter()
                .map(|r| {
                    (0..m.rows)
                        .map(|o| {
                            r.iter()
                                .zip(&wf[o * m.cols..][..m.cols])
                                .map(|(a, b)| a * b)
                                .sum()
                        })
                        .collect()
                })
                .collect()
        };
        let rope = |v: &mut [f32], p: usize| {
            let half = hd / 2;
            for i in 0..half {
                let f = (p as f64 / theta.powf(2.0 * i as f64 / hd as f64)) as f32;
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * f.cos() - b * f.sin();
                v[i + half] = b * f.cos() + a * f.sin();
            }
        };
        let emb = w.mat("model.embed_tokens.weight").unwrap().to_f32();
        let mut h: Vec<Vec<f32>> = ids
            .iter()
            .map(|&t| emb[t as usize * d..][..d].to_vec())
            .collect();
        for l in 0..u("num_hidden_layers") {
            let p = format!("model.layers.{l}");
            let v = |s: &str| w.vec(&format!("{p}.{s}.weight")).unwrap();
            let x: Vec<Vec<f32>> = h.iter().map(|r| rms(r, &v("input_layernorm"))).collect();
            let (qn, kn) = (v("self_attn.q_norm"), v("self_attn.k_norm"));
            let mut q = mm(&x, &format!("{p}.self_attn.q_proj.weight"));
            let mut k = mm(&x, &format!("{p}.self_attn.k_proj.weight"));
            let vv = mm(&x, &format!("{p}.self_attn.v_proj.weight"));
            for i in 0..n {
                for hh in 0..nh {
                    let s = &mut q[i][hh * hd..][..hd];
                    let normed = rms(s, &qn);
                    s.copy_from_slice(&normed);
                    rope(s, i);
                }
                for hh in 0..nkv {
                    let s = &mut k[i][hh * hd..][..hd];
                    let normed = rms(s, &kn);
                    s.copy_from_slice(&normed);
                    rope(s, i);
                }
            }
            let mut att = vec![vec![0.0f32; nh * hd]; n];
            for i in 0..n {
                for hh in 0..nh {
                    let g = hh / (nh / nkv);
                    let qh = &q[i][hh * hd..][..hd];
                    let sc: Vec<f32> = (0..=i)
                        .map(|j| {
                            qh.iter()
                                .zip(&k[j][g * hd..][..hd])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / (hd as f32).sqrt()
                        })
                        .collect();
                    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for (j, ej) in e.iter().enumerate() {
                        for t in 0..hd {
                            att[i][hh * hd + t] += ej / z * vv[j][g * hd + t];
                        }
                    }
                }
            }
            let o = mm(&att, &format!("{p}.self_attn.o_proj.weight"));
            for (r, o) in h.iter_mut().zip(&o) {
                r.iter_mut().zip(o).for_each(|(a, b)| *a += b);
            }
            let x: Vec<Vec<f32>> = h
                .iter()
                .map(|r| rms(r, &v("post_attention_layernorm")))
                .collect();
            let g = mm(&x, &format!("{p}.mlp.gate_proj.weight"));
            let up = mm(&x, &format!("{p}.mlp.up_proj.weight"));
            let a: Vec<Vec<f32>> = g
                .iter()
                .zip(&up)
                .map(|(g, u)| {
                    g.iter()
                        .zip(u)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect()
                })
                .collect();
            let dn = mm(&a, &format!("{p}.mlp.down_proj.weight"));
            for (r, o) in h.iter_mut().zip(&dn) {
                r.iter_mut().zip(o).for_each(|(a, b)| *a += b);
            }
        }
        let last = rms(&h[n - 1], &w.vec("model.norm.weight").unwrap());
        (0..emb.len() / d)
            .map(|t| {
                last.iter()
                    .zip(&emb[t * d..][..d])
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    /// The fast decoder (prefill of the prompt, then one cached decode step) matches the scalar
    /// reference forward to f32 rounding.
    #[test]
    fn qwen3_matches_scalar_reference() {
        let Some(dir) = snapshot("Qwen/Qwen3-0.6B", "model.safetensors") else {
            eprintln!("SKIP: Qwen/Qwen3-0.6B not in disk/");
            return;
        };
        let lm = dense(&dir, Arch::Qwen3);
        let ids = lm.tok.encode("The capital of Japan is");
        let (prompt, mut st) = lm.prepare(ids[..ids.len() - 1].to_vec(), &[]).unwrap();
        lm.forward(&prompt, 0, &mut st).unwrap();
        let got = lm
            .forward(&ids[ids.len() - 1..], prompt.len(), &mut st)
            .unwrap();
        let want = reference_logits(&dir, &ids);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        eprintln!(
            "max |logit diff| {err}, top token {:?}",
            lm.tok.decode(&[argmax(&want) as u32])
        );
        assert!(err < 2e-2, "max |logit diff| {err}");
        assert_eq!(argmax(&got), argmax(&want));
    }

    /// HunYuan dense v1: qk-norm after RoPE, dynamic-NTK alpha base.
    #[test]
    fn hunyuan_dense_answers() {
        let Some(dir) = snapshot("tencent/Hunyuan-0.5B-Instruct", "model.safetensors") else {
            eprintln!("SKIP: tencent/Hunyuan-0.5B-Instruct not in disk/");
            return;
        };
        let lm = dense(&dir, Arch::HunYuan);
        let a = chat(
            &lm,
            "What is the capital of France? Answer briefly.",
            false,
            12,
        );
        assert!(a.contains("Paris"), "{a}");
    }

    /// llama.cpp's Qwen3.5 conversion applied to synthetic transformers tensors (2 key heads, 3
    /// value heads per key head, so `_reorder_v_heads` is no identity), then undone by
    /// [`qwen35_from_gguf`]: every tensor comes back.
    #[test]
    fn qwen35_gguf_transforms_invert_the_converter() {
        let (kh, vh, kd, vd, cols, kernel) = (2, 6, 2, 2, 3, 4);
        let c = serde_json::json!({"linear_num_key_heads": kh, "linear_num_value_heads": vh,
            "linear_key_head_dim": kd, "linear_value_head_dim": vd});
        // `_reorder_v_heads`: blocks grouped by key head `[kh, per_k]` -> tiled `[per_k, kh]`.
        let tile = |x: &[f32], block: usize| -> Vec<f32> {
            let per_k = x.len() / (block * kh);
            let mut out = x.to_vec();
            for k in 0..kh {
                for r in 0..per_k {
                    out[(r * kh + k) * block..][..block]
                        .copy_from_slice(&x[(k * per_k + r) * block..][..block]);
                }
            }
            out
        };
        let qk = 2 * kh * kd;
        let v_tail =
            |x: &[f32], per: usize| [&x[..qk * per], &tile(&x[qk * per..], vd * per)].concat();
        let p = "model.layers.0";
        let cases: [(String, usize, Box<dyn Fn(&[f32]) -> Vec<f32>>); 10] = [
            (
                format!("{p}.linear_attn.in_proj_qkv.weight"),
                (qk + vh * vd) * cols,
                Box::new(|x| v_tail(x, cols)),
            ),
            (
                format!("{p}.linear_attn.conv1d.weight"),
                (qk + vh * vd) * kernel,
                Box::new(|x| v_tail(x, kernel)),
            ),
            (
                format!("{p}.linear_attn.in_proj_z.weight"),
                vh * vd * cols,
                Box::new(|x| tile(x, vd * cols)),
            ),
            (
                format!("{p}.linear_attn.in_proj_a.weight"),
                vh * cols,
                Box::new(|x| tile(x, cols)),
            ),
            (
                format!("{p}.linear_attn.in_proj_b.weight"),
                vh * cols,
                Box::new(|x| tile(x, cols)),
            ),
            (
                format!("{p}.linear_attn.dt_bias"),
                vh,
                Box::new(|x| tile(x, 1)),
            ),
            (
                format!("{p}.linear_attn.neg_A"),
                vh,
                Box::new(|x| tile(x, 1)),
            ),
            (
                format!("{p}.linear_attn.out_proj.weight"),
                3 * vh * vd,
                Box::new(|x| x.chunks(vh * vd).flat_map(|r| tile(r, vd)).collect()),
            ),
            (
                format!("{p}.linear_attn.norm.weight"),
                vd,
                Box::new(|x| x.to_vec()),
            ),
            (
                format!("{p}.self_attn.q_norm.weight"),
                4,
                Box::new(|x| x.iter().map(|w| w + 1.0).collect()),
            ),
        ];
        for (name, n, convert) in cases {
            let hf: Vec<f32> = (0..n).map(|i| i as f32 / 16.0 - 0.5).collect();
            let mut v = convert(&hf);
            qwen35_from_gguf(&name, &mut v, &c);
            let err = hf
                .iter()
                .zip(&v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err < 1e-5, "{name}: {hf:?} -> {v:?}");
        }
    }

    /// Tensors [`qwen35_from_gguf`] leaves unchanged stay packed, and the packed `out_proj`
    /// column untile matches the f32 one.
    #[test]
    fn qwen35_identity_transforms_stay_packed() {
        let c = |kh: usize, vh: usize| {
            serde_json::json!({"linear_num_key_heads": kh, "linear_num_value_heads": vh,
                "linear_key_head_dim": 2, "linear_value_head_dim": 2})
        };
        let p = "model.layers.0";
        for (kh, vh) in [(2, 2), (2, 6)] {
            for n in [
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "in_proj_a.weight",
                "in_proj_b.weight",
                "out_proj.weight",
                "conv1d.weight",
                "neg_A",
                "dt_bias",
                "norm.weight",
            ] {
                let hf = format!("{p}.linear_attn.{n}");
                let len = (4 * kh + vh * 2) * 4 * vh;
                let x: Vec<f32> = (0..len).map(|i| i as f32).collect();
                let mut v = x.clone();
                qwen35_from_gguf(&hf, &mut v, &c(kh, vh));
                if !qwen35_transforms(&hf, kh, vh) {
                    assert_eq!(v, x, "{hf} {kh}/{vh}");
                }
            }
            assert!(qwen35_transforms(
                &format!("{p}.input_layernorm.weight"),
                kh,
                vh
            ));
        }
        assert!(!qwen35_transforms(
            &format!("{p}.linear_attn.out_proj.weight"),
            16,
            16
        ));
        // out_proj rows of 6 heads x 2 columns, one byte per column.
        let (kh, vh, vd) = (2, 6, 2);
        let rows: Vec<f32> = (0..3 * vh * vd).map(|i| i as f32).collect();
        let mut want = rows.clone();
        qwen35_from_gguf(
            &format!("{p}.linear_attn.out_proj.weight"),
            &mut want,
            &c(kh, vh),
        );
        let bytes: Vec<u8> = rows.iter().map(|&v| v as u8).collect();
        let got: Vec<f32> = bytes
            .chunks_exact(vh * vd)
            .flat_map(|r| untile_packed_rows(r, vd, 1, kh, vh, 0))
            .map(f32::from)
            .collect();
        assert_eq!(got, want);
    }

    /// `GIBBS_DISABLE=qwen_packed`: identity `.linear_attn.` tensors and `out_proj` dequantize again.
    #[test]
    fn qwen_packed_disabled_restores_dequantize_predicate() {
        let p = "model.layers.0.linear_attn";
        let needs = |n: &str, permuted: bool| {
            let legacy = crate::graph::disabled("qwen_packed");
            qwen35_needs_transform(&format!("{p}.{n}"), 16, 16, permuted, legacy)
        };
        assert!(!needs("in_proj_z.weight", false) && !needs("out_proj.weight", false));
        crate::graph::TEST_DISABLE.with(|d| d.set("views,qwen_packed"));
        assert!(needs("in_proj_z.weight", false) && needs("out_proj.weight", false));
        assert!(needs("dt_bias", false) && !needs("in_proj_qkv.weight", true));
    }

    /// HY-MT1.5-1.8B from its llama.cpp `hunyuan-dense` GGUF: the GGUF vocabulary (pre-tokenizer
    /// "hunyuan-dense") encodes like the original tokenizer.json, and the model card's prompts
    /// translate (greedy, the GGUF's own chat template).
    #[test]
    fn hunyuan_dense_gguf_tokenizes_and_translates() {
        let file = "HY-MT1.5-1.8B-Q4_K_M.gguf";
        let (Some(gg), Some(hf)) = (
            snapshot("tencent/HY-MT1.5-1.8B-GGUF", file),
            snapshot("tencent/HY-MT1.5-1.8B", "tokenizer.json"),
        ) else {
            eprintln!("SKIP: tencent/HY-MT1.5-1.8B-GGUF {file} or the HF tokenizer.json not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let data = map_gguf(&gg.join(file));
        let m = crate::model::UniversalModelLoader::parse_gguf(&data).unwrap();
        let lm = gguf_model(&data, &m, Device::Cpu).unwrap();
        let hf = load_json_tokenizer(&hf.join("tokenizer.json")).unwrap();
        let special = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>你好<｜hy_Assistant｜>";
        assert_same_ids(lm.tokenizer(), &hf, special);
        for (q, want) in [
            (
                "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\nThe weather is nice today, so we are going to the park.",
                ["天气", "公园"],
            ),
            (
                "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n我喜欢在周末读书。",
                ["read", "weekend"],
            ),
            (
                "Translate the following segment into German, without additional explanation.\n\nThe cat is sleeping on the sofa.",
                ["Katze", "Sofa"],
            ),
        ] {
            let a = chat(lm.as_ref(), q, false, 40);
            eprintln!("{q:?} -> {a:?}");
            assert!(want.iter().all(|w| a.contains(w)), "{a}");
        }
    }

    /// Qwen3.5-0.8B from unsloth's llama.cpp `qwen35` GGUFs against the bf16 safetensors: every
    /// tensor (llama.cpp's transforms undone) by cosine, the tokenizer, greedy decoding of three
    /// chats with the GGUF's own template (tokens identical, or a near tie at the first
    /// divergence), and images refused.
    #[test]
    fn qwen35_gguf_matches_safetensors() {
        let repo = "unsloth/Qwen3.5-0.8B-GGUF";
        let files: Vec<_> = [
            ("Qwen3.5-0.8B-BF16.gguf", 0.99999),
            ("Qwen3.5-0.8B-Q8_0.gguf", 0.999),
        ]
        .into_iter()
        .filter_map(|(f, c)| Some((snapshot(repo, f)?.join(f), c)))
        .collect();
        let st = snapshot("Qwen/Qwen3.5-0.8B", "config.json");
        let (Some(st), false) = (st, files.is_empty()) else {
            eprintln!("SKIP: Qwen/Qwen3.5-0.8B or a {repo} BF16/Q8_0 file not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let cfg: Value =
            serde_json::from_slice(&std::fs::read(st.join("config.json")).unwrap()).unwrap();
        let ck = Weights::open(&st.join("model.safetensors.index.json")).unwrap();
        let reference = Qwen35 {
            tok: load_json_tokenizer(&st.join("tokenizer.json")).unwrap(),
            text: Arc::new(TextModel::load(&ck, text_config(&cfg), Arch::Qwen35).unwrap()),
            vision: None,
            drafter: None,
            program: None,
        };
        for (path, min_cos) in files {
            let data = map_gguf(&path);
            let m = crate::model::UniversalModelLoader::parse_gguf(&data).unwrap();
            let (gk, _, arch) = gguf_checkpoint(&data, &m).unwrap();
            let (mut worst, mut count) = ((1.0f64, String::new()), 0);
            for name in m.tensors.keys() {
                let Some(hf) = gguf_to_hf(arch, name).filter(|h| gk.has(h)) else {
                    continue;
                };
                let theirs = ck.vec(
                    &hf.replacen("model.", "model.language_model.", 1)
                        .replace("neg_A", "A_log"),
                );
                let (a, b) = (
                    gk.vec(&hf).unwrap(),
                    match hf.ends_with(".neg_A") {
                        true => theirs.unwrap().into_iter().map(|v| -v.exp()).collect(),
                        false => theirs.unwrap(),
                    },
                );
                assert_eq!(a.len(), b.len(), "{hf}");
                let dot = |x: &[f32], y: &[f32]| -> f64 {
                    x.iter().zip(y).map(|(&p, &q)| p as f64 * q as f64).sum()
                };
                let cos = dot(&a, &b) / (dot(&a, &a) * dot(&b, &b)).sqrt();
                if cos < worst.0 {
                    worst = (cos, hf);
                }
                count += 1;
            }
            drop(gk);
            let file = path.file_name().unwrap().to_string_lossy();
            eprintln!(
                "{file}: {count} tensors, min cosine {:.7} ({})",
                worst.0, worst.1
            );
            assert!(worst.0 >= min_cos, "{file}: {worst:?}");

            let gg = gguf_model(&data, &m, Device::Cpu).unwrap();
            assert_same_ids(
                gg.tokenizer(),
                &reference.tok,
                "<|im_start|>user\nhi<|im_end|>",
            );
            for q in [
                "What is the capital of France? Answer briefly.",
                "Translate into German: The cat is sleeping on the sofa.",
                "Write one sentence about the ocean.",
            ] {
                let ids = chat_ids(gg.tokenizer(), q, false);
                assert_eq!(
                    ids,
                    chat_ids(&reference.tok, q, false),
                    "{q}: chat template"
                );
                let (same, split) = lockstep(&reference, gg.as_ref(), ids, 32);
                let text = reference.tok.decode(&same);
                match split {
                    None => eprintln!("{file} {q:?}: {} tokens identical: {text:?}", same.len()),
                    Some((a, b, ga, gb)) => eprintln!(
                        "{file} {q:?}: identical for {} tokens ({text:?}), then safetensors {:?} vs GGUF {:?}; margins {ga:.3} / {gb:.3}",
                        same.len(),
                        reference.tok.decode(&[a]),
                        reference.tok.decode(&[b]),
                    ),
                }
                // Quantization may only flip a near tie.
                assert!(split.is_none_or(|(_, _, ga, gb)| ga.max(gb) < 0.5), "{q}");
            }
            let img = MediaInput::Image(b"\x89PNG\r\n\x1a\n".to_vec());
            let err = gg.prepare(vec![1, 2], &[img]).err().unwrap();
            assert!(err.contains("no vision tower"), "{err}");
        }
    }

    /// The first four layers (three Gated DeltaNet, one gated attention) of a real 4-6 bit NInfer
    /// artifact against the same layers of the source bf16 checkpoint: the last-position logits
    /// agree, which a wrong row order, Q/gate interleave or convolution layout would break.
    #[test]
    fn ninfer_first_layers_match_source() {
        let (Some(n), Some(r)) = (
            snapshot("v381654729/qwen3.5-0.8b-ninfer", "qwen3_5_0_8b.ninfer"),
            snapshot("Qwen/Qwen3.5-0.8B", "model.safetensors"),
        ) else {
            eprintln!("SKIP: the 0.8B NInfer artifact or the Qwen/Qwen3.5-0.8B layer subset is not in disk/");
            return;
        };
        let art = crate::model::ninfer::Artifact::open(&n.join("qwen3_5_0_8b.ninfer")).unwrap();
        let (qck, mut cfg) = art.checkpoint().unwrap();
        cfg["num_hidden_layers"] = 4.into();
        cfg["layer_types"] = cfg["layer_types"].as_array().unwrap()[..4].to_vec().into();
        let json = art.resource("tokenizer.json").unwrap();
        let tok = |_: ()| {
            tokenizer_from_files(&String::from_utf8_lossy(&json), &|n| art.resource(n)).unwrap()
        };
        let quant = Qwen35 {
            tok: tok(()),
            text: Arc::new(TextModel::load(&qck, &cfg, Arch::Qwen35).unwrap()),
            vision: None,
            drafter: None,
            program: None,
        };
        cfg["tie_word_embeddings"] = true.into();
        let source = Qwen35 {
            tok: tok(()),
            text: Arc::new(
                TextModel::load(
                    &Weights::open(&r.join("model.safetensors")).unwrap(),
                    &cfg,
                    Arch::Qwen35,
                )
                .unwrap(),
            ),
            vision: None,
            drafter: None,
            program: None,
        };
        let ids = quant
            .tok
            .encode("The quick brown fox jumps over the lazy dog. Paris is the capital of");
        let cosine = |a: &[f32], b: &[f32]| {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                let (x, y) = (x as f64, y as f64);
                (dot, na, nb) = (dot + x * y, na + x * x, nb + y * y);
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        let taps = [1, 2, 3, 4];
        let (ha, hb) = (
            quant.text.hidden_states_at(&ids, ids.len(), &taps).unwrap(),
            source
                .text
                .hidden_states_at(&ids, ids.len(), &taps)
                .unwrap(),
        );
        // Group-64 4-5 bit weights are ~0.994 cosine each; the stream decays slowly from there.
        // A wrong row order or layout would collapse it at the first layer.
        for (k, (x, y)) in ha.iter().zip(&hb).enumerate() {
            let cos = cosine(x, y);
            println!("hidden after layer {}: cosine {cos:.5}", k + 1);
            assert!(cos > 0.97, "hidden after layer {}: cosine {cos}", k + 1);
        }
        let last = |lm: &Qwen35| {
            let (ids, mut st) = lm.prepare(ids.clone(), &[]).unwrap();
            lm.forward(&ids, 0, &mut st).unwrap()
        };
        let cos = cosine(&last(&quant), &last(&source));
        println!("4-layer logits, NInfer vs source: cosine {cos:.5}");
        assert!(cos > 0.9, "cosine {cos}");
    }

    /// Real Qwen3.5-MoE weights from a llama.cpp `qwen35moe` GGUF (a 93-expert pruned 35B-A3B
    /// in MXFP4 experts / Q8_0 rest): experts stay packed in the file mapping, and the greedy
    /// answer to a factual question is right.
    #[test]
    fn qwen35moe_gguf_answers_a_factual_question() {
        let repo = "tvall43/Qwen3.5-14B-A3B-Claude-4.6-Opus-Reasoning-Distilled-reap-gguf";
        let file = "Qwen3.5-14B-A3B-Claude-Opus-Reasoning-Distilled-4.6-MXFP4_MOE.gguf";
        let Some(dir) = snapshot(repo, file) else {
            eprintln!("SKIP: {repo} {file} not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let data = map_gguf(&dir.join(file));
        let m = crate::model::UniversalModelLoader::parse_gguf(&data).unwrap();
        let lm = gguf_model(&data, &m, Device::Cpu).unwrap();
        assert!(lm.summary().contains("experts"), "{}", lm.summary());
        let ids = lm
            .tokenizer()
            .encode("Q: What is the capital of France?\nA: The capital of France is");
        let a = lm.tokenizer().decode(&greedy(lm.as_ref(), ids, 4));
        assert!(a.contains("Paris"), "{a}");
    }

    /// The Qwen4-Exp micro GGUF (random weights: hyper-connections, GDN, MoE, PLE and QSA all
    /// run): one-shot prefill, chunked prefill and token-by-token decode agree.
    #[test]
    fn qwen4exp_gguf_prefill_chunked_and_incremental_agree() {
        let file = "micro-qwen4exp-realqsa-Q4_K_M.gguf";
        let Some(dir) = snapshot("Lynxpda/micro-qwen4exp", file) else {
            eprintln!("SKIP: Lynxpda/micro-qwen4exp {file} not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let data = map_gguf(&dir.join(file));
        let m = crate::model::UniversalModelLoader::parse_gguf(&data).unwrap();
        let lm = gguf_model(&data, &m, Device::Cpu).unwrap();
        let cos = |a: &[f32], b: &[f32]| {
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| (*x as f64) * (*y as f64))
                .sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        };
        // ponytail: the file's random decay constants grow with length, so no run past ~100 tokens
        // stays finite; selection (past the 2048-key budget) is covered by the transformers fixture.
        for len in [12usize, 40] {
            let ids: Vec<u32> = (0..len as u32)
                .map(|i| (i * 7919 + 13) % 32000 + 3)
                .collect();
            let (ids, mut s1) = lm.prepare(ids, &[]).unwrap();
            let whole = lm.forward(&ids, 0, &mut s1).unwrap();
            assert!(
                whole.iter().all(|v| v.is_finite()),
                "{len} tokens: non-finite logits"
            );
            let (_, mut s2) = lm.prepare(ids.clone(), &[]).unwrap();
            let cut = len / 3;
            lm.forward(&ids[..cut], 0, &mut s2).unwrap();
            let mut last = lm.forward(&ids[cut..cut + 5], cut, &mut s2).unwrap();
            for (i, t) in ids.iter().enumerate().skip(cut + 5) {
                last = lm.forward(&[*t], i, &mut s2).unwrap();
            }
            let c = cos(&whole, &last);
            assert!(
                c > 0.9999 && argmax(&whole) == argmax(&last),
                "{len} tokens: cosine {c}"
            );
        }
    }

    /// The `qwen4_exp` safetensors fixture (malaiwah/qwen4-exp-tiny-random-bf16: four hyper-
    /// connection streams, MoE, bigram/trigram PLE, one QSA layer with a 2-block budget) against
    /// the hidden states transformers 5.16.1 captured for it (malaiwah/qfs-fixture-root-captures-v1,
    /// 64-token windows, bf16): final hidden row of each of the first 63 positions.
    #[test]
    fn qwen4exp_safetensors_matches_transformers_hidden_states() {
        let repo = "malaiwah/qwen4-exp-tiny-random-bf16";
        let Some(dir) = snapshot(repo, "model.safetensors") else {
            eprintln!("SKIP: {repo} not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let cap = ["disk".to_string()].into_iter().chain(std::env::var("GIBBS_TEST_STORAGE")).map(|r| {
            Path::new(&r)
                .join("datasets--malaiwah--qfs-fixture-root-captures-v1/snapshots/main/roots/qwen4_exp/first/capture")
        });
        let Some(cap) = cap
            .into_iter()
            .find(|c| c.join("hidden_0000.safetensors").exists())
        else {
            eprintln!("SKIP: malaiwah/qfs-fixture-root-captures-v1 captures not in disk/ or GIBBS_TEST_STORAGE");
            return;
        };
        let cfg: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let ck = Weights::open(&dir.join("model.safetensors")).unwrap();
        let text = TextModel::load(&ck, text_config(&cfg), Arch::Qwen4Exp).unwrap();
        for w in 0..4 {
            // `final-000N.tokens.npy`: 128-byte header, then 64 little-endian i32.
            let npy =
                std::fs::read(dir.join(format!("panel/arrays/final-000{w}.tokens.npy"))).unwrap();
            let ids: Vec<u32> = npy[128..]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
                .collect();
            assert_eq!(ids.len(), 64);
            let want = Weights::open(&cap.join(format!("hidden_000{w}.safetensors"))).unwrap();
            let want = want.vec("hidden_states").unwrap();
            let mut h = vec![0.0f32; ids.len() * 64];
            for (&t, row) in ids.iter().zip(h.chunks_exact_mut(64)) {
                text.embed(t, row).unwrap();
            }
            let pos: Vec<[u32; 3]> = (0..ids.len() as u32).map(|p| [p; 3]).collect();
            let mut state = text.new_state();
            text.set_ple_tokens(&mut state, &ids);
            let h = text.forward_rows(
                h,
                &pos,
                0,
                &mut state,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let got = text.head_hidden(&h, ids.len());
            let (a, b) = (&got[..63 * 64], &want[..63 * 64]);
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| (*x as f64) * (*y as f64))
                .sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (n(a) * n(b));
            eprintln!("window {w}: cosine {cos:.6}");
            assert!(cos > 0.995, "window {w}: cosine {cos}");
        }
    }
}
