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

pub trait NativeLm: Send + Sync {
    /// Compute label for logs and the TUI header, e.g. `CPU (NEON SIMD), 15 threads`.
    fn backend_label(&self) -> String;
    /// Hardware device this model runs on.
    fn device(&self) -> Device {
        Device::Cpu
    }
    /// Weights and shape, for logs and the TUI (e.g. `bf16 weights, 24 layers, hidden 1024`).
    fn summary(&self) -> String;
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

/// Why no loader here runs `model_type` (a family a Hub repo names that is not implemented), for
/// the hub picker and [`load_native`] alike; `None` for every other type.
pub fn unsupported_architecture(model_type: &str) -> Option<&'static str> {
    if model_type.starts_with("deepseek_v4") {
        return Some(
            "DeepSeek-V4 family: needs multi-query attention with q/o low-rank projections and compressed-KV sparse indexing (compress_ratios, index_topk), hyper-connections (hc_mult), 256-384 routed experts with sqrtsoftplus gating and multi-token-prediction heads (V4.1 adds Engram n-gram embeddings); none is implemented, and every real checkpoint exceeds host memory (smallest: 57.8B parameters)",
        );
    }
    if model_type.starts_with("deepseek_v2")
        || model_type.starts_with("deepseek_v3")
        || model_type.starts_with("deepseek2")
    {
        return Some("architecture 'deepseek' uses Multi-Head Latent Attention (MLA), which is not implemented");
    }
    if model_type.starts_with("glm_moe_dsa") {
        return Some(
            "architecture 'glm-dsa' uses Dynamic Sparse Attention (DSA), which is not implemented",
        );
    }
    if model_type.starts_with("wan") {
        return Some("text-to-video model, architecture 'wan' (not supported)");
    }
    None
}

/// Loads `artifacts` as a [`NativeLm`] when its `config.json` names an architecture implemented
/// here; `None` sends it down the GraphLm path.
///
/// On Metal the effective re-encode policy (`ReencodeConfig::effective`) applies to the model's
/// GGUF projections, behind the same whole-model gate as GraphLm loads: a candidate that re-encoded
/// anything is compared with the bit-exact load on the probe text, and a rejected one falls back
/// max-speed -> max-speed (FFN only) -> low-bit-native -> bit-exact.
pub fn load_native(
    artifacts: &ModelArtifacts,
    mode: impl Into<PlacementMode>,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    use crate::model::reencode::{native_stats, reset_native_stats, set_native_applied};
    use crate::model::weight_format::{
        gate_passes, native_kl_gate, reencode_override, set_reencode_override, ReencodeConfig,
        ReencodePolicy::{self, BitExact, LowBitNative, MaxSpeed},
    };
    let mode: PlacementMode = mode.into();
    let user = ReencodeConfig::effective();
    let on_metal = match mode {
        PlacementMode::Pin(d) => d == Device::Metal,
        PlacementMode::Auto => crate::hardware::default_device() == Device::Metal,
    };
    let attempts: &[(&str, ReencodePolicy, bool)] = match user.policy {
        _ if !on_metal => &[("bit-exact", BitExact, false)],
        MaxSpeed => &[
            ("max-speed", MaxSpeed, false),
            ("max-speed (FFN only)", MaxSpeed, true),
            ("low-bit-native", LowBitNative, false),
            ("bit-exact", BitExact, false),
        ],
        LowBitNative => &[
            ("low-bit-native", LowBitNative, false),
            ("bit-exact", BitExact, false),
        ],
        BitExact => &[("bit-exact", BitExact, false)],
    };
    let log = |m: String| {
        tracing::info!("{m}");
        if let Some(f) = &user.log {
            f(&m);
        }
    };
    let previous = reencode_override();
    let load_as = |policy: ReencodePolicy, ffn_only: bool| {
        set_reencode_override(Some(ReencodeConfig {
            policy,
            ffn_only,
            ..user.clone()
        }));
        reset_native_stats();
        load_native_once(artifacts, mode)
    };
    let mut reference: Option<Arc<dyn NativeLm>> = None;
    let mut result = None;
    for &(label, policy, ffn_only) in attempts {
        let Some(loaded) = load_as(policy, ffn_only) else {
            result = None; // not a native architecture: the GraphLm path applies its own policy
            break;
        };
        let cand = match loaded {
            Ok(c) => c,
            Err(e) => {
                result = Some(Err(e));
                break;
            }
        };
        let stats = native_stats();
        let saved = stats.bytes_before.saturating_sub(stats.bytes_after);
        if policy == BitExact || stats.reencoded == 0 {
            set_native_applied(
                if stats.reencoded == 0 {
                    BitExact
                } else {
                    policy
                },
                saved,
            );
            if stats.reencoded > 0 || user.policy != BitExact {
                log(format!(
                    "reencode: {label} applied ({} of {} tensors re-encoded)",
                    stats.reencoded, stats.tensors
                ));
            }
            result = Some(Ok(cand));
            break;
        }
        if reference.is_none() {
            match load_as(BitExact, false) {
                Some(Ok(r)) => reference = Some(r),
                other => {
                    log(format!(
                        "reencode: no bit-exact reference to gate against ({}); using bit-exact",
                        other.and_then(Result::err).unwrap_or_default()
                    ));
                    result = other_exact(&load_as);
                    break;
                }
            }
        }
        // `load_as(BitExact)` above reset the stats: the candidate's own were read first.
        let gate = native_kl_gate(reference.as_deref().expect("loaded above"), &*cand);
        match gate {
            Ok(r) if gate_passes(policy, user.max_kl, &r) => {
                log(format!(
                    "reencode: {label} accepted (mean KL {:.4} nats, top-1 {:.1}%; {} tensors, {:.0} MB saved)",
                    r.mean_kl, r.top1 * 100.0, stats.reencoded, saved as f64 / 1e6
                ));
                set_native_applied(policy, saved);
                result = Some(Ok(cand));
                break;
            }
            Ok(r) => log(format!(
                "reencode: {label} rejected (mean KL {:.4} nats, top-1 {:.1}%)",
                r.mean_kl,
                r.top1 * 100.0
            )),
            Err(e) => {
                log(format!("reencode: gate unavailable ({e}); using bit-exact"));
                result = other_exact(&load_as);
                break;
            }
        }
    }
    set_reencode_override(previous);
    result
}

/// The bit-exact load, as the last resort when the gate cannot run.
fn other_exact(
    load_as: &dyn Fn(
        crate::model::weight_format::ReencodePolicy,
        bool,
    ) -> Option<Result<Arc<dyn NativeLm>, String>>,
) -> Option<Result<Arc<dyn NativeLm>, String>> {
    let r = load_as(crate::model::weight_format::ReencodePolicy::BitExact, false);
    crate::model::reencode::set_native_applied(
        crate::model::weight_format::ReencodePolicy::BitExact,
        0,
    );
    r
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
        t => unsupported_architecture(t).map(|why| Err(format!("{t}: {why}"))),
    }
}
