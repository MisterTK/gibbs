//! Causal LMs implemented outside [`GraphLm`](crate::model::graph_lm::GraphLm): architectures it does
//! not express (hybrid linear attention, per-layer embeddings, ...) and multimodal models whose
//! prompts carry images or audio. The scheduler drives them through [`NativeLm`]; [`load_native`]
//! picks the implementation from `config.json`.

use crate::hardware::{Device, PlacementMode};
use crate::model::hub::ModelArtifacts;
use crate::model::tokenizer::Tokenizer;
use crate::scheduler::spec::Drafter;
use std::any::Any;
use std::sync::Arc;

/// A media item attached to a prompt, as received (encoded file bytes), in prompt order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaInput {
    /// PNG/JPEG/... bytes; [`crate::model::image::ImageBuffer::decode`] reads them.
    Image(Vec<u8>),
    /// WAV/... bytes.
    Audio(Vec<u8>),
}

/// One sequence's model state: KV caches, recurrent state, encoded media.
pub type SeqState = Box<dyn Any + Send + Sync>;

/// What the request allows [`NativeLm::prepare_ctx`]: models that decode inside `prepare` (seq2seq
/// beam search) stop at `max_new_tokens` and when `cancel` is set (client gone).
#[derive(Default, Clone)]
pub struct PrepareCtx {
    pub max_new_tokens: usize,
    pub cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// The logits of a forward result; a device-picked token (`U32`, [`NativeLm::forward_greedy`])
/// where logits are expected is an error, never a one-element row.
pub fn logits_of(t: crate::graph::Tensor) -> Result<Vec<f32>, String> {
    match t {
        crate::graph::Tensor::F32(v) => Ok(v),
        _ => Err("expected f32 logits, got a device-picked token".into()),
    }
}

pub trait NativeLm: Send + Sync {
    /// Compute label for logs and the TUI header, e.g. `CPU (NEON SIMD), 15 threads`.
    fn backend_label(&self) -> String;
    /// Hardware device this model runs on.
    fn device(&self) -> Device {
        Device::Cpu
    }
    /// Weights and shape, for logs and the TUI (e.g. `bf16 weights, 24 layers, hidden 1024`).
    fn summary(&self) -> String;
    /// The model's graph nodes by op kind, device and avenue; `None` when it runs no graph program.
    fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        None
    }
    fn tokenizer(&self) -> &Tokenizer;
    /// Positions one sequence may hold (prompt + output).
    fn context_len(&self) -> usize;
    /// Output cap when the caller sets none (seq2seq: generation_config `max_length`).
    fn default_max_new_tokens(&self) -> Option<usize> {
        None
    }
    /// Turns the template-rendered prompt `ids` (one placeholder per media item, in order) into
    /// the ids the model runs (placeholders expanded to each item's token count) and builds the
    /// sequence's state. Errs on media the model can't take.
    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String>;
    /// [`Self::prepare`] under the request's limits; the scheduler calls this one.
    fn prepare_ctx(
        &self,
        ids: Vec<u32>,
        media: &[MediaInput],
        _ctx: &PrepareCtx,
    ) -> Result<(Vec<u32>, SeqState), String> {
        self.prepare(ids, media)
    }
    /// Length of the ids [`Self::prepare`] would return, without building state. Override when
    /// the count is cheaper than running the media encoders (it is for every image model).
    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        Ok(self.prepare(ids, media)?.0.len())
    }
    /// Runs `tokens` at positions `start_pos..` against `state`; returns the logits after the last.
    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String>;
    /// [`Self::forward`] for a greedy step: `U32([token])` when the device picked the first largest
    /// logit within the tokenizer's vocabulary ([`crate::graph::program::GraphProgram::run_greedy`]),
    /// else `F32(logits)`.
    fn forward_greedy(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<crate::graph::Tensor, String> {
        self.forward(tokens, start_pos, state)
            .map(crate::graph::Tensor::F32)
    }

    /// The next token of a greedy device decode stream (see
    /// [`crate::graph::program::GraphProgram::stream_next`]): `last` at `start_pos`, at most
    /// `max_new` tokens queued. `None` when this model or state cannot stream.
    fn stream_greedy_token(
        &self,
        _last: u32,
        _start_pos: usize,
        _max_new: usize,
        _state: &mut SeqState,
    ) -> Result<Option<u32>, String> {
        Ok(None)
    }

    /// Ends the state's device decode stream, if any.
    fn stream_end(&self, _state: &mut SeqState) -> Result<(), String> {
        Ok(())
    }

    // Speculative decoding (see `crate::scheduler::spec`): a model supports it by verifying
    // several positions in one pass and rewinding the ones the target rejects.

    /// [`Self::forward`] returning the logits after every token, `[tokens.len(), vocab]`.
    fn forward_all(
        &self,
        _tokens: &[u32],
        _start_pos: usize,
        _state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        Err("this model cannot verify draft tokens".into())
    }

    /// Drops positions `len..` from `state`, exactly as if they had never run. Recurrent models
    /// may only rewind into the span of their last [`Self::forward_all`].
    fn truncate(&self, _state: &mut SeqState, _len: usize) -> Result<(), String> {
        Err("this model cannot rewind its sequence state".into())
    }

    /// From now on every forward of this sequence records, per position, the output of each
    /// decoder layer in `layers` (0-based, the residual stream before the final norm: HF
    /// `hidden_states[i + 1]`) as `[positions, layers.len() * hidden]`, drained by
    /// [`Self::take_taps`].
    fn set_taps(&self, _state: &mut SeqState, _layers: &[usize]) -> Result<(), String> {
        Err("this model cannot expose hidden states to a drafter".into())
    }

    fn take_taps(&self, _state: &mut SeqState) -> Vec<f32> {
        Vec::new()
    }

    /// The drafter the checkpoint ships (e.g. multi-token-prediction heads).
    fn drafter(&self) -> Option<Arc<dyn Drafter>> {
        None
    }
}

/// `model_type`s [`load_native`] implements. The hub picker lets their repos load whatever their
/// pipeline tag (`image-text-to-text`, `any-to-any`, ...).
pub const NATIVE_MODEL_TYPES: &[&str] = &[
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen4_exp",
    "qwen4_exp_text",
    "qwen3_vl",
    "gemma4",
    "gemma4_unified",
    "idefics3",
    "smolvlm",
    "qwen3",
    "hunyuan_v1_dense",
    "t5",
    "mt5",
    "umt5",
];

/// [`NATIVE_MODEL_TYPES`] whose loaders also read GGUF weights: llama.cpp `qwen3` /
/// `hunyuan-dense` / `qwen35` / `qwen35moe` / `qwen4exp` files ([`crate::model::qwen3_5::load_gguf`]),
/// `gemma4` files with their `mmproj` ([`crate::model::gemma4::load_gguf`]) and candle-style T5
/// GGUFs ([`crate::model::t5::load`]). The hub picker refuses GGUF repos of the other types.
pub const NATIVE_GGUF_TYPES: &[&str] = &[
    "gemma4",
    "gemma4_unified",
    "qwen3",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen4_exp",
    "qwen4_exp_text",
    "hunyuan_v1_dense",
    "t5",
    "mt5",
    "umt5",
];

/// Loads `artifacts` as a [`NativeLm`] when its `config.json` names an architecture implemented
/// here; `None` sends it down the GraphLm path.
///
/// On Metal the effective re-encode policy (`ReencodeConfig::effective`) applies to the model's
/// GGUF projections, behind the whole-model KL gate of [`crate::arch::load::gated`].
pub fn load_native(
    artifacts: &ModelArtifacts,
    mode: impl Into<PlacementMode>,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    let mode: PlacementMode = mode.into();
    let on_metal = match mode {
        PlacementMode::Pin(d) => d == Device::Metal,
        PlacementMode::Auto => crate::hardware::default_device() == Device::Metal,
    };
    crate::arch::load::gated(
        on_metal,
        || load_native_once(artifacts, mode),
        |reference, candidate| {
            crate::model::weight_format::native_kl_gate(&**reference, &**candidate)
        },
    )
}

/// One load of a native model under whatever re-encode configuration is currently in effect.
fn load_native_once(
    artifacts: &ModelArtifacts,
    mode: PlacementMode,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    if let Some(r) = crate::model::qwen3_5::load_gguf(artifacts, mode)
        .or_else(|| crate::model::qwen3_5::load_ninfer(artifacts, mode))
    {
        return Some(r);
    }
    if let Some(r) = crate::model::gemma4::load_gguf(artifacts, mode) {
        return Some(r);
    }
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.config_path.as_ref()?).ok()?).ok()?;
    if crate::model::seq2seq::ARCHITECTURES
        .contains(&cfg["architectures"][0].as_str().unwrap_or(""))
    {
        return Some(crate::model::seq2seq::load(artifacts, &cfg, mode));
    }
    // google/umt5-* configs name only the architecture.
    match cfg["model_type"]
        .as_str()
        .or(cfg["architectures"][0].as_str())?
    {
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen4_exp"
        | "qwen4_exp_text" | "qwen3_vl" => Some(crate::model::qwen3_5::load(artifacts, &cfg, mode)),
        "qwen3" | "hunyuan_v1_dense" => {
            Some(crate::model::qwen3_5::load_dense(artifacts, &cfg, mode))
        }
        "gemma4" | "gemma4_unified" => Some(crate::model::gemma4::load(artifacts, &cfg, mode)),
        "idefics3" | "smolvlm" => Some(crate::model::idefics3::load(artifacts, &cfg, mode)),
        "t5" | "mt5" | "umt5" | "UMT5ForConditionalGeneration" => {
            Some(crate::model::t5::load(artifacts, &cfg, mode))
        }
        t => crate::arch::unsupported(t).map(|why| Err(format!("{t}: {why}"))),
    }
}
