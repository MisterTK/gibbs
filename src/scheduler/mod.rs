pub mod spec;

use crate::hardware::{Device, QuantMode};
use crate::kv::{BlockId, PagedKVPool, TOKENS_PER_BLOCK};
use crate::model::graph_lm::{sample_logits, GraphLm};
use crate::model::native::{MediaInput, NativeLm, PrepareCtx, SeqState};
use crate::model::tokenizer::Tokenizer;
use crate::telemetry::{EngineTelemetry, GapSample};
use parking_lot::{Mutex, RwLock};
use spec::{Drafter, KTuner, SpecSeq};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// Prefix of the error that ends a sequence aborted by [`Scheduler::swap_model`]; the server
/// answers such a request with 503 (retry) rather than 500.
pub const MODEL_SWAPPED: &str = "model swapped to ";

/// Label for CPU execution, shared by the banner, logs, and TUI.
pub fn cpu_backend_label() -> String {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    format!("CPU (NEON SIMD), {threads} threads")
}

/// Per-sequence KV state, owned by the backend that produced it.
pub enum SeqKvCache {
    /// StateStore for GraphProgram execution
    Program(crate::graph::StateStore),
    /// Built by [`NativeLm::prepare`] at submission.
    Native(SeqState),
}
#[derive(Clone)]
pub enum ExecutableModel {
    /// Multi-segment GraphProgram execution.
    Program(
        Arc<parking_lot::RwLock<crate::graph::program::GraphProgram>>,
        Arc<parking_lot::RwLock<GraphLm>>,
    ),
    /// A causal LM outside GraphLm (new architectures, multimodal); see [`crate::model::native`].
    Native(Arc<dyn NativeLm>),
}

/// Last 64 tokens: the window the sampler sees.
fn recent(tokens: &[u32]) -> &[u32] {
    &tokens[tokens.len().saturating_sub(64)..]
}

const WRONG_KV: &str = "sequence KV cache belongs to a different compute backend";

impl ExecutableModel {
    /// Places a loaded GraphLm on `mode` (Metal uploads the weights and compiles the kernels).
    pub fn load_graph_lm(
        model: GraphLm,
        mode: impl Into<crate::hardware::PlacementMode>,
    ) -> Result<Self, String> {
        let mode = mode.into();
        let target_device = match mode {
            crate::hardware::PlacementMode::Pin(d) => d,
            crate::hardware::PlacementMode::Auto => crate::hardware::default_device(),
        };
        tracing::info!(
            "Model 'GraphLm ({} layers, {} hidden)' placed on {target_device}",
            model.config.num_layers,
            model.config.hidden_dim,
        );

        match target_device {
            Device::Cpu => {
                let (decode_g, weights) = model.to_decode_graph();
                let prog = crate::graph::program::GraphProgram::compile(
                    decode_g,
                    weights,
                    crate::hardware::PlacementMode::Pin(Device::Cpu),
                )?;
                Ok(Self::Program(
                    Arc::new(parking_lot::RwLock::new(prog)),
                    Arc::new(parking_lot::RwLock::new(model)),
                ))
            }
            #[cfg(target_os = "macos")]
            Device::Metal => {
                let (decode_g, weights) = model.to_decode_graph();
                let prog = crate::graph::program::GraphProgram::compile(
                    decode_g,
                    weights,
                    crate::hardware::PlacementMode::Pin(Device::Metal),
                )?;
                Ok(Self::Program(
                    Arc::new(parking_lot::RwLock::new(prog)),
                    Arc::new(parking_lot::RwLock::new(model)),
                ))
            }
            #[cfg(not(target_os = "macos"))]
            Device::Metal => Err("the Metal backend requires macOS".into()),
        }
    }

    pub fn llm_backend(&self) -> Device {
        match self {
            Self::Program(prog, _) => prog.read().primary_device(),
            Self::Native(m) => m.device(),
        }
    }

    /// e.g. `CPU (NEON SIMD), 14 threads` or `Metal (Apple M5 Pro)` (device name from the Metal API).
    pub fn backend_label(&self) -> String {
        match self {
            Self::Program(prog, _) => prog.read().backend_label(),
            Self::Native(m) => m.backend_label(),
        }
    }
    /// Graph nodes by op kind, device and avenue; `None` for models that do not run a graph
    /// program (CPU executors, seq2seq, vision models).
    pub fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        match self {
            Self::Program(prog, _) => Some(prog.read().placement_summary()),
            Self::Native(m) => m.placement_summary(),
        }
    }
    /// Decodes `last_token` against the sequence's KV cache (prefilling `all_tokens` first when the
    /// sequence has none) and samples the next token. `logprobs: Some(k)` also reports the chosen
    /// token's probability, the top `k` candidates and the entropy.
    pub fn step_token_with_kv(
        &self,
        all_tokens: &[u32],
        last_token: u32,
        pos: usize,
        temp: f32,
        top_p: f32,
        logprobs: Option<usize>,
        kv_cache: &mut Option<SeqKvCache>,
    ) -> Result<(u32, Option<TokenLogprobInfo>), String> {
        // Greedy (as `sample_logits` decides it) without logprobs: programs may pick the token on
        // the device.
        let device_pick = temp <= 1e-4 && logprobs.is_none();
        let logits = match self {
            // `pos` counts `last_token` itself.
            Self::Native(m) => match kv_cache {
                Some(SeqKvCache::Native(state)) if device_pick => {
                    match m.forward_greedy(&[last_token], pos - 1, state)? {
                        crate::graph::Tensor::U32(t) if t.len() == 1 => return Ok((t[0], None)),
                        t => crate::model::native::logits_of(t)?,
                    }
                }
                Some(SeqKvCache::Native(state)) => m.forward(&[last_token], pos - 1, state)?,
                _ => return Err(WRONG_KV.into()),
            },
            Self::Program(prog, m) => {
                let mut p = prog.write();
                match kv_cache {
                    Some(SeqKvCache::Program(state)) => {
                        std::mem::swap(&mut p.state, state);
                        let actual_pos = (pos - 1) as u32;
                        let inputs = vec![
                            crate::graph::Tensor::U32(vec![last_token]),
                            crate::graph::Tensor::U32(vec![actual_pos]),
                        ];
                        let out = if device_pick {
                            p.run_greedy(inputs, m.read().tokenizer.vocab_size())
                        } else {
                            p.run(inputs)
                        };
                        std::mem::swap(&mut p.state, state);
                        let out = out?;
                        match out.into_iter().next() {
                            Some(crate::graph::Tensor::F32(v)) => v,
                            Some(crate::graph::Tensor::U32(t)) if t.len() == 1 => {
                                return Ok((t[0], None))
                            }
                            _ => return Err("GraphProgram did not return F32 logits".into()),
                        }
                    }
                    None => {
                        p.state = crate::graph::StateStore::new();
                        let mut last_logits = Vec::new();
                        for (i, &tok) in all_tokens.iter().enumerate() {
                            let inputs = vec![
                                crate::graph::Tensor::U32(vec![tok]),
                                crate::graph::Tensor::U32(vec![i as u32]),
                            ];
                            let out = p.run(inputs)?;
                            match out.into_iter().next() {
                                Some(crate::graph::Tensor::F32(v)) => last_logits = v,
                                _ => return Err("GraphProgram did not return F32 logits".into()),
                            }
                        }
                        *kv_cache = Some(SeqKvCache::Program(p.state.clone()));
                        last_logits
                    }
                    _ => return Err(WRONG_KV.into()),
                }
            }
        };
        let chosen = self.sample(&logits, recent(all_tokens), temp, top_p);
        Ok((chosen, logprobs.map(|k| logprob_info(&logits, chosen, k))))
    }

    /// The next greedy token of a device decode stream ([`GraphProgram::stream_next`]): `last` is
    /// the newest token, `pos` counts it (as in [`Self::step_token_with_kv`]), and at most
    /// `max_new` more tokens are queued. `None` when this model or sequence cannot stream (use
    /// `step_token_with_kv`). A stream keeps one speculative token in flight: end it with
    /// [`Self::stream_end`] before the sequence state is reused.
    ///
    /// [`GraphProgram::stream_next`]: crate::graph::program::GraphProgram::stream_next
    pub fn stream_greedy_token(
        &self,
        last: u32,
        pos: usize,
        max_new: usize,
        kv: &mut Option<SeqKvCache>,
    ) -> Result<Option<u32>, String> {
        match (self, kv) {
            (Self::Program(prog, m), Some(SeqKvCache::Program(state))) => {
                let mut p = prog.write();
                let vocab = m.read().tokenizer.vocab_size();
                std::mem::swap(&mut p.state, state);
                let out = p.stream_next(last, (pos - 1) as u32, max_new, vocab);
                std::mem::swap(&mut p.state, state);
                out
            }
            (Self::Native(m), Some(SeqKvCache::Native(state))) => {
                m.stream_greedy_token(last, pos - 1, max_new, state)
            }
            _ => Ok(None),
        }
    }

    /// Ends the sequence's device decode stream, if any (waits for its in-flight tokens).
    pub fn stream_end(&self, kv: &mut Option<SeqKvCache>) -> Result<(), String> {
        match (self, kv) {
            (Self::Program(prog, _), Some(SeqKvCache::Program(state))) => {
                let mut p = prog.write();
                std::mem::swap(&mut p.state, state);
                let out = p.stream_end();
                std::mem::swap(&mut p.state, state);
                out
            }
            (Self::Native(m), Some(SeqKvCache::Native(state))) => m.stream_end(state),
            _ => Ok(()),
        }
    }

    /// Evaluates a chunk of prompt tokens. If this chunk completes the prompt,
    /// returns `Some(next_token)`. If more prompt tokens remain, returns `None`.
    pub fn step_prefill_chunk(
        &self,
        chunk_tokens: &[u32],
        start_pos: usize,
        is_last_chunk: bool,
        all_prompt_tokens: &[u32],
        temp: f32,
        top_p: f32,
        kv_cache: &mut Option<SeqKvCache>,
    ) -> Result<Option<u32>, String> {
        if chunk_tokens.is_empty() {
            return Ok(None);
        }
        let logits = self.prefill_logits(chunk_tokens, start_pos, kv_cache)?;
        Ok(is_last_chunk.then(|| self.sample(&logits, recent(all_prompt_tokens), temp, top_p)))
    }

    /// Logits at the last position of `chunk_tokens` (non-empty), advancing `kv_cache`.
    pub(crate) fn prefill_logits(
        &self,
        chunk_tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut Option<SeqKvCache>,
    ) -> Result<Vec<f32>, String> {
        Ok(match self {
            Self::Native(m) => match kv_cache {
                Some(SeqKvCache::Native(state)) => m.forward(chunk_tokens, start_pos, state)?,
                _ => return Err(WRONG_KV.into()),
            },
            Self::Program(prog, m) => {
                let mut p = prog.write();
                let seq_state = match kv_cache
                    .get_or_insert_with(|| SeqKvCache::Program(crate::graph::StateStore::new()))
                {
                    SeqKvCache::Program(s) => s,
                    _ => return Err(WRONG_KV.into()),
                };
                std::mem::swap(&mut p.state, seq_state);
                let chunk_len = chunk_tokens.len();
                let is_metal = p.is_single_metal();

                let last_logits = if chunk_len >= 16 && is_metal {
                    let model = m.read();
                    let prefill_g = model.to_prefill_graph_structure(chunk_len);
                    let pos_vec: Vec<u32> =
                        (0..chunk_len).map(|i| (start_pos + i) as u32).collect();
                    let inputs = vec![
                        crate::graph::Tensor::U32(chunk_tokens.to_vec()),
                        crate::graph::Tensor::U32(pos_vec),
                    ];
                    // `run_reshaped` caches the plan per chunk shape (an uncached plan costs ~6 ms).
                    let res = p.run_reshaped(&prefill_g, inputs).and_then(|out| {
                        match out.into_iter().next() {
                            Some(crate::graph::Tensor::F32(v)) => {
                                let v_size = model.config.vocab_size;
                                Ok(v[(chunk_len - 1) * v_size..chunk_len * v_size].to_vec())
                            }
                            _ => Err("GraphProgram did not return F32 logits".to_string()),
                        }
                    });
                    match res {
                        Ok(l) => l,
                        Err(e) => {
                            std::mem::swap(&mut p.state, seq_state);
                            return Err(e);
                        }
                    }
                } else {
                    let mut last = Vec::new();
                    let mut err = None;
                    for (i, &tok) in chunk_tokens.iter().enumerate() {
                        let inputs = vec![
                            crate::graph::Tensor::U32(vec![tok]),
                            crate::graph::Tensor::U32(vec![(start_pos + i) as u32]),
                        ];
                        match p.run(inputs) {
                            Ok(out) => match out.into_iter().next() {
                                Some(crate::graph::Tensor::F32(v)) => last = v,
                                _ => {
                                    err =
                                        Some("GraphProgram did not return F32 logits".to_string());
                                    break;
                                }
                            },
                            Err(e) => {
                                err = Some(e);
                                break;
                            }
                        }
                    }
                    if let Some(e) = err {
                        std::mem::swap(&mut p.state, seq_state);
                        return Err(e);
                    }
                    last
                };
                std::mem::swap(&mut p.state, seq_state);
                last_logits
            }
        })
    }

    /// Last-position logits for a text prompt (already tokenized, template applied by the caller).
    pub fn prompt_logits(&self, ids: Vec<u32>) -> Result<Vec<f32>, String> {
        let (ids, mut kv) = self.prepare(ids, &[], &PrepareCtx::default())?;
        if ids.is_empty() {
            return Err("the prompt encodes to no tokens".into());
        }
        self.prefill_logits(&ids, 0, &mut kv)
    }

    /// Top-k 50 / top-p sampling without repetition penalty; `temp <= 0` is greedy argmax.
    fn sample(&self, logits: &[f32], recent: &[u32], temp: f32, top_p: f32) -> u32 {
        match self {
            Self::Program(_, m) => m
                .read()
                .sample_next_token_with_penalty(logits, recent, temp, 50, top_p, 1.0),
            Self::Native(m) => {
                let vocab = m.tokenizer().vocab_size();
                let valid = if vocab > 0 {
                    logits.len().min(vocab)
                } else {
                    logits.len()
                };
                sample_logits(&logits[..valid], recent, temp, 50, top_p, 1.0)
            }
        }
    }

    /// One-shot generation through a private [`ContinuousBatchScheduler`]: (text, decode-step
    /// tokens, decode time). `drafter` speculates; native models otherwise use the drafter they ship.
    pub fn generate_text_measured(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        drafter: Option<Arc<dyn Drafter>>,
    ) -> Result<(String, usize, std::time::Duration), String> {
        let s =
            ContinuousBatchScheduler::new("", Arc::new(PagedKVPool::new(512)), Some(self.clone()));
        if drafter.is_some() {
            s.set_drafter(drafter);
        }
        let g = s.generate(prompt, max_new_tokens, temperature)?;
        Ok((g.text, g.decode.1, g.decode.0))
    }

    /// [`Self::generate_text_measured`] with drafters off: one forward pass per generated token,
    /// the quantity the bench's roofline is defined over.
    pub fn generate_text_measured_plain(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<(String, usize, std::time::Duration), String> {
        let s =
            ContinuousBatchScheduler::new("", Arc::new(PagedKVPool::new(512)), Some(self.clone()));
        s.set_drafter(None);
        let g = s.generate(prompt, max_new_tokens, temperature)?;
        Ok((g.text, g.decode.1, g.decode.0))
    }

    pub fn tokenizer(&self) -> Tokenizer {
        match self {
            Self::Program(_, m) => m.read().tokenizer.clone(),
            Self::Native(m) => m.tokenizer().clone(),
        }
    }

    /// GraphLm weight format; `None` for native models, which keep their own representation.
    pub fn current_quant_mode(&self) -> Option<QuantMode> {
        match self {
            Self::Program(_, m) => Some(m.read().current_quant_mode()),
            Self::Native(_) => None,
        }
    }

    /// Maximum positions the model's KV cache holds; `None` when the model has no fixed window.
    pub fn context_len(&self) -> Option<usize> {
        match self {
            Self::Program(_, m) => Some(m.read().config.max_seq_len.max(1)),
            Self::Native(m) => Some(m.context_len().max(1)),
        }
    }

    /// Expands `ids` for the model and builds the sequence's initial state: native models
    /// [`NativeLm::prepare`] (media allowed); every other model takes text only.
    fn prepare(
        &self,
        ids: Vec<u32>,
        media: &[MediaInput],
        ctx: &PrepareCtx,
    ) -> Result<(Vec<u32>, Option<SeqKvCache>), String> {
        match self {
            Self::Native(m) => {
                let (ids, state) = m.prepare_ctx(ids, media, ctx)?;
                Ok((ids, Some(SeqKvCache::Native(state))))
            }
            _ if media.is_empty() => Ok((ids, None)),
            _ => Err("this model takes text only (no image or audio input)".into()),
        }
    }

    /// [`Self::prepare`]'s id count without building state ([`NativeLm::prompt_len`]).
    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        match self {
            Self::Native(m) => m.prompt_len(ids, media),
            _ => self
                .prepare(ids, media, &PrepareCtx::default())
                .map(|(ids, _)| ids.len()),
        }
    }
}

/// `chosen`'s probability, the top `top_k` candidates and the exact entropy of `logits`.
fn logprob_info(logits: &[f32], chosen: u32, top_k: usize) -> TokenLogprobInfo {
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let lse = max_l + sum_exp.ln();
    let entropy = -logits
        .iter()
        .map(|&l| {
            let p = (l - lse).exp();
            if p > 1e-12 {
                p * (l - lse)
            } else {
                0.0
            }
        })
        .sum::<f32>();
    // Top k in O(V + k log k).
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    let k_take = top_k.min(indexed.len()).max(1);
    if k_take < indexed.len() {
        indexed.select_nth_unstable_by(k_take - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let top_slice = &mut indexed[..k_take];
    top_slice.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_candidates = top_slice
        .iter()
        .map(|&(tok, l)| (tok, (l - lse).exp()))
        .collect();
    let chosen_prob = (logits.get(chosen as usize).copied().unwrap_or(max_l) - lse)
        .exp()
        .min(1.0);
    TokenLogprobInfo {
        chosen_id: chosen,
        chosen_prob,
        top_candidates,
        entropy,
    }
}

/// What [`ContinuousBatchScheduler::generate`] produced and what it cost.
pub struct Generation {
    pub text: String,
    /// (time, prompt tokens) of prefill.
    pub prefill: (Duration, usize),
    /// (time, tokens emitted by decode steps); the first token comes from prefill.
    pub decode: (Duration, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// Default output cap for requests that do not set `max_tokens`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 64;
/// Sampling temperature for requests that do not set `temperature` (the OpenAI default).
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
/// Nucleus cutoff for requests that do not set `top_p`.
pub const DEFAULT_TOP_P: f32 = 0.9;
/// Most prompt tokens one scheduler step prefills, across all sequences.
pub const MAX_PREFILL_CHUNK: usize = 2048;
/// Prefill budget before the first prefill has been timed.
const INITIAL_PREFILL_CHUNK: usize = 64;
/// With nothing decoding, a step's prefill is sized to take about this long, so a request that
/// arrives mid-prefill waits at most about this long before it is admitted.
const IDLE_STEP: Duration = Duration::from_millis(100);

/// Measured costs that size the prefill budget. While sequences decode, a step's prefill is
/// sized to cost about one decode round (at least 1 token), so prompt ingestion takes roughly
/// half of each step; on CPU one prompt token can cost more than a decode round, so the 1-token
/// floor can make a step longer than that. Otherwise prefill is sized to cost [`IDLE_STEP`].
/// Backends that batch prefill (Metal) measure a lower cost per token than per-token CPU
/// prefill, so they get larger budgets.
// ponytail: one EWMA of cost per token, though batched prefill gets cheaper per token as chunks
// grow; fit cost(chunk) if budgets oscillate.
#[derive(Default)]
struct PrefillTuner {
    /// EWMA of prefill time per prompt token (ns); 0 until measured.
    prefill_ns_per_token: f64,
    /// EWMA of one step's decode work over all decoding sequences (ns); 0 until measured.
    decode_round_ns: f64,
}

impl PrefillTuner {
    fn budget(&self, decoding: bool) -> usize {
        if self.prefill_ns_per_token == 0.0 {
            return INITIAL_PREFILL_CHUNK;
        }
        let target_ns = match (decoding, self.decode_round_ns) {
            // Decode cost not timed yet: give the decoding sequences the step.
            (true, d) if d == 0.0 => return 1,
            (true, d) => d,
            (false, _) => IDLE_STEP.as_nanos() as f64,
        };
        ((target_ns / self.prefill_ns_per_token) as usize).clamp(1, MAX_PREFILL_CHUNK)
    }

    /// `prefill` = (time, prompt tokens) and `decode` = (time, decoded tokens) spent in one step.
    fn record(&mut self, prefill: (Duration, usize), decode: (Duration, usize)) {
        let ewma = |old: f64, new: f64| {
            if old == 0.0 {
                new
            } else {
                0.75 * old + 0.25 * new
            }
        };
        if prefill.1 > 0 {
            let per_token = prefill.0.as_nanos() as f64 / prefill.1 as f64;
            self.prefill_ns_per_token = ewma(self.prefill_ns_per_token, per_token);
        }
        if decode.1 > 0 {
            self.decode_round_ns = ewma(self.decode_round_ns, decode.0.as_nanos() as f64);
        }
    }
}

/// Per-sequence sampling controls (OpenAI `temperature` / `top_p` / `max_tokens`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// `None` uses the model's generation_config default, else [`DEFAULT_TEMPERATURE`]; `<= 0` is
    /// exact greedy argmax.
    pub temperature: Option<f32>,
    /// `None` uses the model's generation_config default, else [`DEFAULT_TOP_P`].
    pub top_p: Option<f32>,
    pub max_new_tokens: usize,
    pub top_logprobs: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobInfo {
    pub chosen_id: u32,
    pub chosen_prob: f32,
    pub top_candidates: Vec<(u32, f32)>,
    pub entropy: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            top_logprobs: None,
        }
    }
}

impl SamplingParams {
    /// An unset `max_tokens` runs to EOS or the context end (`submit_with_media` clamps it).
    pub fn from_request(temperature: Option<f32>, max_tokens: Option<usize>) -> Self {
        Self {
            temperature,
            top_p: None,
            max_new_tokens: max_tokens.unwrap_or(usize::MAX),
            top_logprobs: None,
        }
    }
}

/// Why a sequence stopped, in OpenAI `finish_reason` terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS/special token, or no model.
    Stop,
    /// `max_new_tokens` or the model's context length was reached.
    Length,
    /// The engine failed mid-request (prefill/decode error); the message is the cause.
    Error(String),
}

/// Set once by the scheduler before the token stream closes; unset means `Stop`.
pub type FinishSignal = Arc<std::sync::OnceLock<FinishReason>>;

/// OpenAI `finish_reason` for a closed stream, or the engine error that aborted it.
pub fn finish_outcome(signal: &FinishSignal) -> Result<&'static str, String> {
    match signal.get() {
        None | Some(FinishReason::Stop) => Ok("stop"),
        Some(FinishReason::Length) => Ok("length"),
        Some(FinishReason::Error(e)) => Err(e.clone()),
    }
}

pub struct Sequence {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub allocated_blocks: Vec<BlockId>,
    pub params: SamplingParams,
    pub state: SequenceState,
    pub submitted: Instant,
    /// When the last output token was sent; `None` before the first.
    pub last_emit: Option<Instant>,
    pub token_sender: Option<UnboundedSender<u32>>,
    pub kv_cache: Option<SeqKvCache>,
    pub prefilled_tokens: usize,
    pub finish: FinishSignal,
    /// Speculative decoding state; set when the sequence starts prefill with a drafter.
    pub spec: Option<SpecSeq>,
}

impl Sequence {
    pub fn new(id: u64, prompt_tokens: Vec<u32>) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: Vec::new(),
            allocated_blocks: Vec::new(),
            params: SamplingParams::default(),
            state: SequenceState::Waiting,
            submitted: Instant::now(),
            last_emit: None,
            token_sender: None,
            kv_cache: None,
            prefilled_tokens: 0,
            finish: FinishSignal::default(),
            spec: None,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Appends `toks` (one decode step's output), streams them, and records their latency: time
    /// to first token, else the gap since the previous step shared evenly by the step's tokens
    /// (returned).
    fn emit(&mut self, toks: &[u32], telemetry: &EngineTelemetry) -> Option<Duration> {
        self.output_tokens.extend_from_slice(toks);
        if let Some(sender) = &self.token_sender {
            for &t in toks {
                let _ = sender.send(t);
            }
        }
        let now = Instant::now();
        match self.last_emit.replace(now) {
            None => {
                telemetry.record_ttft(now - self.submitted);
                None
            }
            Some(prev) => {
                let itl = (now - prev) / toks.len().max(1) as u32;
                for _ in toks {
                    telemetry.record_itl(itl);
                }
                (!toks.is_empty()).then_some(itl)
            }
        }
    }
}

pub struct BatchStepOutput {
    pub tokens_generated: usize,
    pub active_sequences_count: usize,
    /// (time, prompt tokens) this step prefilled.
    pub prefill: (Duration, usize),
    /// (time, tokens emitted) of this step's decode work.
    pub decode: (Duration, usize),
}

/// One admitted sequence, as of the end of the last step.
#[derive(Debug, Clone)]
pub struct SeqView {
    pub id: u64,
    pub state: SequenceState,
    pub prompt: usize,
    pub prefilled: usize,
    pub generated: usize,
    pub max_new: usize,
    pub submitted: Instant,
}

/// What the scheduler did in its last step, published for dashboards without touching the
/// locks a step holds.
#[derive(Debug, Clone, Default)]
pub struct SchedulerView {
    pub seqs: Vec<SeqView>,
    pub waiting: usize,
    /// Prefill budget (prompt tokens) the last step used; see [`PrefillTuner`].
    pub prefill_budget: usize,
    /// Duration of the last step that ran a forward pass.
    pub busy_step: Duration,
    /// The speculative drafter, when one is active.
    pub spec: Option<SpecView>,
}

/// Speculative decoding as of the last step; rates are in [`EngineTelemetry`]'s snapshot.
#[derive(Debug, Clone)]
pub struct SpecView {
    pub label: String,
    /// Draft tokens per step the tuner currently rates best (0 = plain decoding pays off);
    /// `None` until it has measured a step.
    pub k: Option<usize>,
}

pub struct ContinuousBatchScheduler {
    model_id: RwLock<String>,
    pub kv_pool: Arc<PagedKVPool>,
    model: RwLock<Option<ExecutableModel>>,
    tokenizer: RwLock<Arc<Tokenizer>>,
    pub telemetry: Arc<EngineTelemetry>,
    pub waiting_queue: RwLock<VecDeque<Sequence>>,
    pub active_sequences: RwLock<Vec<Sequence>>,
    pub step_counter: AtomicU32,
    next_seq_id: AtomicU64,
    prefill_tuner: Mutex<PrefillTuner>,
    view: Mutex<SchedulerView>,
    /// Drafter for speculative decoding: the model's own (MTP heads) unless replaced.
    drafter: RwLock<Option<Arc<dyn Drafter>>>,
    k_tuner: Mutex<KTuner>,
    pub logprob_tx: Arc<RwLock<Option<UnboundedSender<TokenLogprobInfo>>>>,
}

/// The drafter `model` ships, unless `GIBBS_SPEC=off`.
fn builtin_drafter(model: Option<&ExecutableModel>) -> Option<Arc<dyn Drafter>> {
    match model? {
        ExecutableModel::Native(m) if spec_enabled() => m.drafter(),
        _ => None,
    }
}

fn spec_enabled() -> bool {
    std::env::var("GIBBS_SPEC").map_or(true, |v| v != "off")
}

impl ContinuousBatchScheduler {
    pub fn new(
        model_id: impl Into<String>,
        kv_pool: Arc<PagedKVPool>,
        model: Option<ExecutableModel>,
    ) -> Self {
        Self::with_telemetry(model_id, kv_pool, model, EngineTelemetry::new())
    }

    pub fn with_telemetry(
        model_id: impl Into<String>,
        kv_pool: Arc<PagedKVPool>,
        model: Option<ExecutableModel>,
        telemetry: Arc<EngineTelemetry>,
    ) -> Self {
        let tokenizer = Arc::new(
            model
                .as_ref()
                .map(ExecutableModel::tokenizer)
                .unwrap_or_else(Tokenizer::default_bpe),
        );
        let has_model = model.is_some();
        let s = Self {
            model_id: RwLock::new(model_id.into()),
            kv_pool,
            drafter: RwLock::new(builtin_drafter(model.as_ref())),
            model: RwLock::new(model),
            tokenizer: RwLock::new(tokenizer),
            telemetry,
            waiting_queue: RwLock::new(VecDeque::new()),
            active_sequences: RwLock::new(Vec::new()),
            step_counter: AtomicU32::new(0),
            next_seq_id: AtomicU64::new(0),
            prefill_tuner: Mutex::new(PrefillTuner::default()),
            view: Mutex::new(SchedulerView::default()),
            k_tuner: Mutex::new(KTuner::default()),
            logprob_tx: Arc::new(RwLock::new(None)),
        };
        if has_model {
            s.log_drafter();
        }
        s
    }

    /// Says which drafter new sequences speculate with, if any.
    fn log_drafter(&self) {
        match self.drafter.read().as_ref() {
            Some(d) => self.telemetry.log_info(&format!(
                "speculative decoding: {} drafter, k picked from measured speed (0..={}); GIBBS_SPEC=off disables it",
                d.label(),
                d.max_k()
            )),
            None if !spec_enabled() => self
                .telemetry
                .log_info("speculative decoding: off (GIBBS_SPEC=off)"),
            None => {}
        }
    }

    /// Tokenizer belonging to the currently executing model.
    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.read().clone()
    }

    pub fn has_model(&self) -> bool {
        self.model.read().is_some()
    }

    /// The executing model (a cheap clone: every variant shares its weights); `None` without an LLM.
    pub fn model(&self) -> Option<ExecutableModel> {
        self.model.read().clone()
    }

    /// Folds one plain decode token into the telemetry gap: its inter-token latency `itl` against
    /// the newest GPU run sample. Nothing is recorded without one (CPU, or a profiled token).
    #[cfg(target_os = "macos")]
    fn record_gap(&self, itl: Duration) {
        let Some(run) = crate::graph::metal::take_run_samples().last().copied() else {
            return;
        };
        let ceiling = crate::model::variant::host_ceiling_gbps();
        let weights_ms = if ceiling > 0.0 {
            self.telemetry.bytes_per_token() as f64 / 1e9 / ceiling * 1e3
        } else {
            0.0
        };
        self.telemetry.record_gap(GapSample {
            wall_ms: itl.as_secs_f32() * 1e3,
            run_ms: run.run_ns as f32 / 1e6,
            gpu_ms: run.gpu_ns as f32 / 1e6,
            weights_ms: weights_ms as f32,
            dispatches: run.dispatches,
        });
    }

    #[cfg(not(target_os = "macos"))]
    fn record_gap(&self, _itl: Duration) {}

    /// Backend the current model executes on (CPU when no LLM is loaded).
    pub fn llm_backend(&self) -> Device {
        self.model
            .read()
            .as_ref()
            .map_or(Device::Cpu, ExecutableModel::llm_backend)
    }

    /// Truthful compute label for the current model (CPU when no LLM is loaded).
    pub fn backend_label(&self) -> String {
        self.model
            .read()
            .as_ref()
            .map_or_else(cpu_backend_label, ExecutableModel::backend_label)
    }

    /// Id of the model this engine serves (the `--model` id, or the last hot-swapped one).
    pub fn model_id(&self) -> String {
        self.model_id.read().clone()
    }

    /// Prefill budget (prompt tokens per step) the engine chose for its last step.
    pub fn prefill_chunk(&self) -> usize {
        self.view.lock().prefill_budget
    }

    /// Sequences, queue, and step timing as of the end of the last step.
    pub fn view(&self) -> SchedulerView {
        self.view.lock().clone()
    }

    /// Weight format of the loaded LLM; `None` when no LLM is loaded.
    pub fn quant_mode(&self) -> Option<QuantMode> {
        self.model
            .read()
            .as_ref()
            .and_then(ExecutableModel::current_quant_mode)
    }

    /// Replaces the executing model and its tokenizer. In-flight sequences are aborted
    /// (their token streams end with an error) because their KV caches belong to the old weights.
    pub fn swap_model(&self, model_id: &str, model: ExecutableModel) {
        let mut active = self.active_sequences.write();
        let mut waiting = self.waiting_queue.write();
        self.fail_all(
            &mut active,
            &mut waiting,
            &format!("{MODEL_SWAPPED}{model_id}"),
        );
        *self.tokenizer.write() = Arc::new(model.tokenizer());
        *self.drafter.write() = builtin_drafter(Some(&model));
        *self.model.write() = Some(model);
        *self.model_id.write() = model_id.to_string();
        // Prefill and decode costs were measured on the old model.
        *self.prefill_tuner.lock() = PrefillTuner::default();
        *self.k_tuner.lock() = KTuner::default();
        self.log_drafter();
    }

    /// Replaces the speculative drafter (`None` turns speculation off); sequences that started
    /// before keep decoding as they were. `GIBBS_SPEC=off` keeps it off.
    pub fn set_drafter(&self, drafter: Option<Arc<dyn Drafter>>) {
        *self.drafter.write() = drafter.filter(|_| spec_enabled());
        *self.k_tuner.lock() = KTuner::default();
        self.log_drafter();
    }

    /// Ends every in-flight sequence with `reason`; their streams close with an error.
    pub fn abort_all(&self, reason: &str) {
        let mut active = self.active_sequences.write();
        let mut waiting = self.waiting_queue.write();
        self.fail_all(&mut active, &mut waiting, reason);
    }

    fn fail_all(&self, active: &mut Vec<Sequence>, waiting: &mut VecDeque<Sequence>, reason: &str) {
        let n = active.len() + waiting.len();
        for seq in active.drain(..).chain(waiting.drain(..)) {
            let _ = seq.finish.set(FinishReason::Error(reason.to_string()));
            for &b in &seq.allocated_blocks {
                self.kv_pool.release_block(b);
            }
        }
        if n > 0 {
            self.telemetry
                .log_warn(&format!("{n} in-flight request(s) aborted: {reason}"));
        }
    }

    /// Builds prompt ids with the current model's tokenizer (via `encode`, e.g. template + encode) and enqueues
    /// them atomically with respect to `swap_model` (both hold `waiting_queue`), so a sequence never carries
    /// another vocabulary's ids. Returns the tokenizer to decode the stream with and the prompt token count.
    pub fn submit_with(
        &self,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> (u64, UnboundedReceiver<u32>, Arc<Tokenizer>, usize) {
        let (id, rx, tok, n, _) = self.submit_with_params(SamplingParams::default(), encode);
        (id, rx, tok, n)
    }

    /// `submit_with` honoring per-request sampling params; `max_new_tokens` is clamped so prompt plus
    /// output fits the model's context window. The returned `FinishSignal` is final once the stream
    /// closes; a prompt the model can't prepare closes at once with that error.
    pub fn submit_with_params(
        &self,
        params: SamplingParams,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> (
        u64,
        UnboundedReceiver<u32>,
        Arc<Tokenizer>,
        usize,
        FinishSignal,
    ) {
        self.submit_with_media(params, &[], encode)
            .unwrap_or_else(|e| {
                let id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
                let finish = FinishSignal::default();
                let _ = finish.set(FinishReason::Error(e));
                (id, unbounded_channel().1, self.tokenizer(), 0, finish)
            })
    }

    /// [`Self::submit_with_params`] with images/audio for the prompt (one template placeholder per
    /// item, in order). Errs, before anything is queued, when the model can't take the media or
    /// was swapped while the prompt was prepared. Preparing (e.g. encoding images) runs outside
    /// the queue lock so it never stalls a step.
    pub fn submit_with_media(
        &self,
        params: SamplingParams,
        media: &[MediaInput],
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<
        (
            u64,
            UnboundedReceiver<u32>,
            Arc<Tokenizer>,
            usize,
            FinishSignal,
        ),
        String,
    > {
        self.submit_cancellable(params, media, None, encode)
    }

    /// [`Self::submit_with_media`]; `cancel` (set when the client goes away) stops a model that
    /// decodes inside `prepare` (seq2seq beam search).
    pub fn submit_cancellable(
        &self,
        mut params: SamplingParams,
        media: &[MediaInput],
        cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<
        (
            u64,
            UnboundedReceiver<u32>,
            Arc<Tokenizer>,
            usize,
            FinishSignal,
        ),
        String,
    > {
        let ctx = PrepareCtx {
            max_new_tokens: params.max_new_tokens,
            cancel,
        };
        let (tokenizer, model, ids, kv_cache) = self.prepare_prompt(media, &ctx, encode)?;
        let n = ids.len();
        if let Some(ctx) = model.as_ref().and_then(ExecutableModel::context_len) {
            params.max_new_tokens = params.max_new_tokens.min(ctx.saturating_sub(n));
        }
        if let Some((t, p)) = tokenizer.sampling_defaults {
            params.temperature.get_or_insert(t);
            params.top_p.get_or_insert(p);
        }
        let mut waiting = self.waiting_queue.write();
        if !Arc::ptr_eq(&tokenizer, &self.tokenizer()) {
            return Err("model swapped while the prompt was prepared".into());
        }
        let id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = unbounded_channel();
        let mut seq = Sequence::new(id, ids);
        seq.params = params;
        seq.token_sender = Some(tx);
        seq.state = SequenceState::Waiting;
        seq.kv_cache = kv_cache;
        let finish = seq.finish.clone();
        waiting.push_back(seq);
        drop(waiting);
        Ok((id, rx, tokenizer, n, finish))
    }

    /// Prompt ids the model would prefill for `encode` + `media` (images/audio expanded), without
    /// queueing or encoding media: the Anthropic `count_tokens` path.
    pub fn prompt_len(
        &self,
        media: &[MediaInput],
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<usize, String> {
        let (tokenizer, model) = self.model_pair();
        let ids = Self::encode_nonempty(&tokenizer, encode);
        match &model {
            Some(m) => m.prompt_len(ids, media),
            None if media.is_empty() => Ok(ids.len()),
            None => Err("no model loaded to take image or audio input".into()),
        }
    }

    /// `swap_model` replaces both under the queue's write lock: read them as one pair.
    fn model_pair(&self) -> (Arc<Tokenizer>, Option<ExecutableModel>) {
        let _queue = self.waiting_queue.read();
        (self.tokenizer(), self.model.read().clone())
    }

    fn encode_nonempty(
        tokenizer: &Tokenizer,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Vec<u32> {
        let mut ids = encode(tokenizer);
        if ids.is_empty() {
            ids.push(1);
        }
        ids
    }

    /// Encodes the prompt and runs the model's `prepare` against one consistent
    /// (tokenizer, model) pair.
    fn prepare_prompt(
        &self,
        media: &[MediaInput],
        ctx: &PrepareCtx,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<
        (
            Arc<Tokenizer>,
            Option<ExecutableModel>,
            Vec<u32>,
            Option<SeqKvCache>,
        ),
        String,
    > {
        let (tokenizer, model) = self.model_pair();
        let ids = Self::encode_nonempty(&tokenizer, encode);
        let (ids, kv_cache) = match &model {
            Some(m) => m.prepare(ids, media, ctx)?,
            None if media.is_empty() => (ids, None),
            None => return Err("no model loaded to take image or audio input".into()),
        };
        Ok((tokenizer, model, ids, kv_cache))
    }

    /// Submit request and return sequence ID plus live token streaming receiver
    pub fn submit_request_stream(&self, prompt_tokens: Vec<u32>) -> (u64, UnboundedReceiver<u32>) {
        let id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = unbounded_channel();
        let mut seq = Sequence::new(id, prompt_tokens);
        seq.token_sender = Some(tx);
        seq.state = SequenceState::Waiting;
        let mut waiting = self.waiting_queue.write();
        waiting.push_back(seq);
        drop(waiting);
        (id, rx)
    }

    /// Runs one request to completion on this thread (CLI `--prompt`, bench): the prompt goes
    /// through the tokenizer's chat template when it declares one, sampling uses `temperature`
    /// and [`DEFAULT_TOP_P`]. Prints the prefill line unless the TUI owns the terminal.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Generation, String> {
        let params = SamplingParams {
            temperature: Some(temperature),
            top_p: Some(DEFAULT_TOP_P),
            max_new_tokens,
            top_logprobs: None,
        };
        let (_, mut rx, tok, _, finish) =
            self.submit_with_media(params, &[], |t| match &t.chat_template {
                Some(_) => t.encode(&t.apply_chat_template(&[("user", prompt)])),
                None => t.encode(prompt),
            })?;
        let (mut prefill, mut decode) = ((Duration::ZERO, 0), (Duration::ZERO, 0));
        let mut out = Vec::new();
        'run: loop {
            let s = self.step();
            prefill = (prefill.0 + s.prefill.0, prefill.1 + s.prefill.1);
            decode = (decode.0 + s.decode.0, decode.1 + s.decode.1);
            loop {
                match rx.try_recv() {
                    Ok(t) => out.push(t),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break 'run,
                }
            }
        }
        finish_outcome(&finish)?;
        if !crate::telemetry::TUI_OWNS_TERMINAL.load(Ordering::Relaxed) {
            println!(
                "⚡ Prefill: {} tokens in {:.2}ms ({:.1} tok/s)",
                prefill.1,
                prefill.0.as_secs_f64() * 1000.0,
                prefill.1 as f64 / prefill.0.as_secs_f64().max(1e-9)
            );
        }
        Ok(Generation {
            text: tok.decode(&out),
            prefill,
            decode,
        })
    }

    pub fn step(&self) -> BatchStepOutput {
        let step_started = Instant::now();
        // Lock order (active, waiting, model) matches `swap_model`, so a step never mixes models.
        let mut active = self.active_sequences.write();
        let mut waiting = self.waiting_queue.write();
        let model = self.model.read().clone();
        let tokenizer = self.tokenizer();
        let drafter = self.drafter.read().clone();
        // Sequences end at the model's context length: its KV cache holds no more positions.
        let context_len = model
            .as_ref()
            .and_then(ExecutableModel::context_len)
            .unwrap_or(usize::MAX);
        let current_step = self.step_counter.fetch_add(1, Ordering::Relaxed) as usize;
        let decoding = active.iter().any(|s| s.state == SequenceState::Decoding);
        let prefill_budget = self.prefill_tuner.lock().budget(decoding);

        // 1. Promote waiting sequences if KV blocks are available
        while let Some(mut seq) = waiting.pop_front() {
            let (matched_blocks, matched_tokens) =
                self.kv_pool.radix_cache.match_prefix(&seq.prompt_tokens);

            seq.allocated_blocks = matched_blocks;
            let needed_tokens = seq
                .prompt_tokens
                .len()
                .saturating_sub(matched_tokens)
                .min(prefill_budget);
            let needed_blocks = (needed_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;

            let mut all_allocated = true;
            for _ in 0..needed_blocks {
                if let Some(block_id) = self.kv_pool.allocate_block(current_step) {
                    seq.allocated_blocks.push(block_id);
                } else {
                    all_allocated = false;
                    break;
                }
            }

            if all_allocated {
                seq.prefilled_tokens = 0;
                seq.state = SequenceState::Prefilling;
                active.push(seq);
            } else {
                for &b in &seq.allocated_blocks {
                    self.kv_pool.release_block(b);
                }
                waiting.push_front(seq);
                break;
            }
        }

        // Release waiting queue lock cleanly so concurrent ingestion is not blocked during forward compute
        let waiting_len = waiting.len();
        drop(waiting);
        // Show admitted sequences now: on CPU one prefill step can take seconds.
        self.publish(&active, waiting_len, prefill_budget, None);

        // 2. Execute chunked prefill and forward decode across active sequences.
        // Prefill is handed out shortest remaining prompt first, so a short prompt queued behind a
        // long one gets its few tokens (and its first output token) this step. The oldest
        // prefilling sequence keeps up to half the budget, so a stream of short prompts cannot
        // hold a long one back forever.
        active.sort_by_key(|s| match s.state {
            SequenceState::Prefilling => s.prompt_tokens.len().saturating_sub(s.prefilled_tokens),
            _ => 0,
        });
        let oldest = active
            .iter()
            .filter(|s| s.state == SequenceState::Prefilling)
            .min_by_key(|s| s.submitted);
        let mut reserved = oldest.map_or(0, |s| {
            (s.prompt_tokens.len() - s.prefilled_tokens).min(prefill_budget.div_ceil(2))
        });
        let oldest = oldest.map(|s| s.id);
        // ponytail: chunked prefill caps prefill tokens per step with global active lock; fine-grained sequence locks if concurrency scales.
        let mut tokens_gen = 0;

        let mut prefill_budget_left = prefill_budget;
        let (mut prefill_time, mut prefill_tokens) = (Duration::ZERO, 0);
        let (mut decode_time, mut decoded, mut decode_emitted) = (Duration::ZERO, 0, 0);

        for seq in active.iter_mut() {
            if seq.state == SequenceState::Finished {
                continue;
            }
            // Nobody reads the stream any more (client gone, TUI cancel): stop computing it.
            if seq.token_sender.as_ref().is_some_and(|tx| tx.is_closed()) {
                seq.state = SequenceState::Finished;
                continue;
            }

            let Some(m) = &model else {
                // No autoregressive model loaded: end the stream instead of inventing tokens.
                seq.state = SequenceState::Finished;
                continue;
            };

            if seq.state == SequenceState::Prefilling {
                let available = if Some(seq.id) == oldest {
                    reserved = 0;
                    prefill_budget_left
                } else {
                    prefill_budget_left.saturating_sub(reserved)
                };
                if available == 0 {
                    // Step's prefill budget exhausted; keep sequence in Prefilling for next step.
                    continue;
                }

                let remaining = seq.prompt_tokens.len().saturating_sub(seq.prefilled_tokens);
                let chunk_len = remaining.min(available);
                if chunk_len == 0 {
                    seq.state = SequenceState::Decoding;
                } else {
                    // Ensure sufficient KV blocks are allocated for this chunk
                    let needed_capacity = seq.prefilled_tokens + chunk_len;
                    let current_capacity = seq.allocated_blocks.len() * TOKENS_PER_BLOCK;
                    if needed_capacity > current_capacity {
                        let blocks_needed = (needed_capacity - current_capacity + TOKENS_PER_BLOCK
                            - 1)
                            / TOKENS_PER_BLOCK;
                        for _ in 0..blocks_needed {
                            if let Some(b) = self.kv_pool.allocate_block(current_step) {
                                seq.allocated_blocks.push(b);
                            } else {
                                break;
                            }
                        }
                    }

                    if seq.prefilled_tokens == 0 && seq.spec.is_none() {
                        if let Some(d) = &drafter {
                            let nanos = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_or(0, |t| t.as_nanos() as u64);
                            let seed = nanos ^ seq.id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            match SpecSeq::start(m, d.clone(), &mut seq.kv_cache, seed) {
                                Ok(s) => seq.spec = Some(s),
                                Err(e) => self.telemetry.log_warn(&format!(
                                    "sequence {}: no speculative decoding: {e}",
                                    seq.id
                                )),
                            }
                        }
                    }
                    let is_last = (seq.prefilled_tokens + chunk_len) >= seq.prompt_tokens.len();
                    let chunk =
                        &seq.prompt_tokens[seq.prefilled_tokens..seq.prefilled_tokens + chunk_len];
                    let started = Instant::now();
                    let next_tok_opt = match m.step_prefill_chunk(
                        chunk,
                        seq.prefilled_tokens,
                        is_last,
                        &seq.prompt_tokens,
                        seq.params.temperature.unwrap_or(DEFAULT_TEMPERATURE),
                        seq.params.top_p.unwrap_or(DEFAULT_TOP_P),
                        &mut seq.kv_cache,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            self.telemetry
                                .log_warn(&format!("sequence {} aborted in prefill: {e}", seq.id));
                            let _ = seq
                                .finish
                                .set(FinishReason::Error(format!("prefill failed: {e}")));
                            seq.state = SequenceState::Finished;
                            continue;
                        }
                    };
                    prefill_time += started.elapsed();
                    prefill_tokens += chunk_len;
                    seq.prefilled_tokens += chunk_len;
                    prefill_budget_left = prefill_budget_left.saturating_sub(chunk_len);

                    if is_last {
                        seq.state = SequenceState::Decoding;
                        self.kv_pool
                            .radix_cache
                            .insert_prefix(&seq.prompt_tokens, &seq.allocated_blocks);
                        if seq.params.max_new_tokens == 0 {
                            let _ = seq.finish.set(FinishReason::Length);
                            seq.state = SequenceState::Finished;
                        } else if let Some(next_tok) = next_tok_opt {
                            if tokenizer.ends_generation(next_tok) {
                                seq.state = SequenceState::Finished;
                            } else {
                                seq.emit(&[next_tok], &self.telemetry);
                                tokens_gen += 1;
                                if seq.total_tokens() >= context_len
                                    || seq.output_tokens.len() >= seq.params.max_new_tokens
                                {
                                    let _ = seq.finish.set(FinishReason::Length);
                                    seq.state = SequenceState::Finished;
                                }
                            }
                        }
                    }
                }
            } else if seq.state == SequenceState::Decoding {
                let temp = seq.params.temperature.unwrap_or(DEFAULT_TEMPERATURE);
                let top_p = seq.params.top_p.unwrap_or(DEFAULT_TOP_P);
                let mut all = seq.prompt_tokens.clone();
                all.extend_from_slice(&seq.output_tokens);
                // Draft no more than the sequence may still emit (the target adds one) or hold.
                let spec_k = seq.spec.as_ref().map(|sp| {
                    self.k_tuner
                        .lock()
                        .choose(sp.max_k())
                        .min(
                            seq.params
                                .max_new_tokens
                                .saturating_sub(seq.output_tokens.len() + 1),
                        )
                        .min(context_len.saturating_sub(all.len()))
                });
                // Ensure sufficient KV blocks are allocated for the positions this step may add.
                let needed_capacity = seq.total_tokens() + 1 + spec_k.unwrap_or(0);
                while needed_capacity > seq.allocated_blocks.len() * TOKENS_PER_BLOCK {
                    match self.kv_pool.allocate_block(current_step) {
                        Some(b) => seq.allocated_blocks.push(b),
                        None => break,
                    }
                }

                let started = Instant::now();
                let result = match (spec_k, seq.spec.as_mut()) {
                    (Some(k), Some(sp)) => {
                        let vocab = tokenizer.vocab_size();
                        spec::decode(m, sp, &all, &mut seq.kv_cache, temp, top_p, vocab, k).map(
                            |s| {
                                if s.synced {
                                    let msg = self.k_tuner.lock().record(
                                        k,
                                        s.tokens.len(),
                                        started.elapsed(),
                                    );
                                    if let Some(msg) = msg {
                                        self.telemetry.log_info(&msg);
                                    }
                                }
                                if s.drafted > 0 {
                                    self.telemetry.record_spec(
                                        s.drafted,
                                        s.accepted,
                                        s.tokens.len(),
                                    );
                                }
                                s.tokens
                            },
                        )
                    }
                    _ => {
                        let last = all.last().copied().unwrap_or(1);
                        let logprobs = seq.params.top_logprobs;
                        let streamed = if temp <= 1e-4 && logprobs.is_none() {
                            m.stream_greedy_token(
                                last,
                                all.len(),
                                seq.params
                                    .max_new_tokens
                                    .saturating_sub(seq.output_tokens.len())
                                    .min(context_len.saturating_sub(all.len())),
                                &mut seq.kv_cache,
                            )
                        } else {
                            Ok(None)
                        };
                        match streamed {
                            Ok(Some(t)) => Ok(vec![t]),
                            Ok(None) => m
                                .step_token_with_kv(
                                    &all,
                                    last,
                                    all.len(),
                                    temp,
                                    top_p,
                                    logprobs,
                                    &mut seq.kv_cache,
                                )
                                .map(|(tok, info)| {
                                    if let (Some(info), Some(tx)) =
                                        (info, self.logprob_tx.read().as_ref())
                                    {
                                        let _ = tx.send(info);
                                    }
                                    vec![tok]
                                }),
                            Err(e) => Err(e),
                        }
                    }
                };
                let tokens = match result {
                    Ok(t) => t,
                    Err(e) => {
                        self.telemetry
                            .log_warn(&format!("sequence {} aborted in decode: {e}", seq.id));
                        let _ = seq
                            .finish
                            .set(FinishReason::Error(format!("decode failed: {e}")));
                        seq.state = SequenceState::Finished;
                        continue;
                    }
                };
                decode_time += started.elapsed();
                decoded += 1;

                // Emit up to EOS, max_new_tokens or the context end, whichever comes first.
                let room = seq
                    .params
                    .max_new_tokens
                    .saturating_sub(seq.output_tokens.len())
                    .min(context_len.saturating_sub(seq.total_tokens()));
                let eos = tokens.iter().position(|&t| tokenizer.ends_generation(t));
                let n = eos.unwrap_or(tokens.len()).min(room);
                if n > 0 {
                    let itl = seq.emit(&tokens[..n], &self.telemetry);
                    tokens_gen += n;
                    decode_emitted += n;
                    // A plain decode token (no drafter, one forward) against the run behind it.
                    if let (Some(itl), None, 1) = (itl, spec_k, n) {
                        self.record_gap(itl);
                    }
                }
                if seq.total_tokens() >= context_len
                    || seq.output_tokens.len() >= seq.params.max_new_tokens
                {
                    let _ = seq.finish.set(FinishReason::Length);
                    seq.state = SequenceState::Finished;
                } else if eos == Some(n) {
                    seq.state = SequenceState::Finished;
                }
            }
        }
        self.prefill_tuner
            .lock()
            .record((prefill_time, prefill_tokens), (decode_time, decoded));
        if tokens_gen > 0 {
            self.telemetry.record_tokens(tokens_gen);
        }

        // 4. Release blocks for finished sequences
        active.retain(|seq| {
            if seq.state == SequenceState::Finished {
                for &b in &seq.allocated_blocks {
                    self.kv_pool.release_block(b);
                }
                self.telemetry.record_request_completed();
                false
            } else {
                true
            }
        });

        let busy = (prefill_tokens + decoded > 0).then(|| step_started.elapsed());
        self.publish(&active, waiting_len, prefill_budget, busy);

        BatchStepOutput {
            tokens_generated: tokens_gen,
            active_sequences_count: active.len(),
            prefill: (prefill_time, prefill_tokens),
            decode: (decode_time, decode_emitted),
        }
    }

    /// Publishes the dashboard view; `busy` (the step's duration) only when it ran a forward pass.
    fn publish(&self, active: &[Sequence], waiting: usize, budget: usize, busy: Option<Duration>) {
        let mut view = self.view.lock();
        view.seqs.clear();
        view.seqs.extend(active.iter().map(|s| SeqView {
            id: s.id,
            state: s.state,
            prompt: s.prompt_tokens.len(),
            prefilled: s.prefilled_tokens,
            generated: s.output_tokens.len(),
            max_new: s.params.max_new_tokens,
            submitted: s.submitted,
        }));
        view.waiting = waiting;
        view.prefill_budget = budget;
        view.spec = self.drafter.read().as_ref().map(|d| SpecView {
            label: d.label(),
            k: self.k_tuner.lock().best(),
        });
        if let Some(d) = busy {
            view.busy_step = d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph_lm::{GraphLmConfig, LayerKVCache};

    /// Host-forward KV cache: the reference the program-path tests compare against.
    fn new_cpu_kv(model: &GraphLm) -> Vec<LayerKVCache> {
        let max_seq_len = model.config.max_seq_len.max(1);
        (0..model.layers.len())
            .map(|_| LayerKVCache::new(max_seq_len, model.config.hidden_dim))
            .collect()
    }

    fn cpu_exec(model: GraphLm) -> ExecutableModel {
        ExecutableModel::load_graph_lm(model, Device::Cpu).unwrap()
    }

    /// smollm_1m with room for prompts spanning several prefill budgets; never emits EOS.
    fn long_context_model() -> GraphLm {
        let mut model = GraphLm::new(GraphLmConfig {
            max_seq_len: 4096,
            ..GraphLmConfig::smollm_1m()
        });
        model.tokenizer.special_tokens.clear();
        model.tokenizer.inv_vocab.clear();
        model
    }

    /// While a sequence decodes, prompt prefill is cut to a measured budget (and the decoding
    /// sequence still gets a token every step); once it finishes, the budget grows back.
    #[test]
    fn test_prefill_budget_tracks_decoding_load() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(long_context_model())),
        );
        let greedy = |n| SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: n,
            top_logprobs: None,
        };
        let (_, mut rx1, _, _, _) = s.submit_with_params(greedy(40), |_| vec![1, 2, 3, 4]);
        s.step();
        assert!(rx1.try_recv().is_ok());

        let (_, _long, _, _, _) = s.submit_with_params(greedy(1), |_| vec![42; 3000]);
        let (mut prefilled, mut busiest) = (0, 0);
        while s
            .active_sequences
            .read()
            .iter()
            .any(|q| q.state == SequenceState::Decoding)
        {
            s.step();
            let chunk = s.prefill_chunk();
            busiest = busiest.max(chunk);
            let now = s.active_sequences.read().last().unwrap().prefilled_tokens;
            assert_eq!(now - prefilled, chunk, "prefill must use the whole budget");
            prefilled = now;
            assert!(
                rx1.try_recv().is_ok(),
                "decoding sequence starved by prefill"
            );
        }
        s.step();
        let idle = s.prefill_chunk();
        assert!(
            idle > busiest,
            "budget {idle} with nothing decoding should exceed {busiest} while decoding"
        );
    }

    /// A short prompt queued behind a long one is prefilled first and answers in the same step.
    #[test]
    fn test_short_prompt_not_stuck_behind_long_prefill() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(long_context_model())),
        );
        let greedy = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 4,
            top_logprobs: None,
        };
        let (_, _long, _, _, _) = s.submit_with_params(greedy, |_| vec![42; 3000]);
        let (_, mut short, _, _, _) = s.submit_with_params(greedy, |_| vec![1, 2, 3, 4]);
        s.step();
        assert!(
            short.try_recv().is_ok(),
            "short prompt waited behind the long prefill"
        );
    }

    /// The oldest prompt keeps prefilling while newer, shorter prompts keep arriving.
    #[test]
    fn test_long_prompt_not_starved_by_short_prompts() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(long_context_model())),
        );
        let greedy = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1,
            top_logprobs: None,
        };
        let (long, _long, _, _, _) = s.submit_with_params(greedy, |_| vec![42; 3000]);
        let (mut streams, mut prefilled) = (Vec::new(), 0);
        for step in 0..6 {
            // Each newcomer has less left to prefill than the long prompt, so it sorts first.
            streams.push(s.submit_with_params(greedy, |_| vec![7; 2000]).1);
            s.step();
            // A fast host may finish the long prefill (and retire it) within these steps.
            let now = s
                .view()
                .seqs
                .iter()
                .find(|q| q.id == long)
                .map_or(3000, |q| q.prefilled);
            assert!(
                now > prefilled,
                "step {step}: the long prompt got no prefill"
            );
            if now >= 3000 {
                break;
            }
            prefilled = now;
        }
    }

    /// Dropping the token receiver (client gone, TUI cancel) ends the sequence on the next step.
    #[test]
    fn test_dropped_receiver_frees_slot() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(never_eos_model())),
        );
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1000,
            top_logprobs: None,
        };
        let (_, rx, _, _, _) = s.submit_with_params(params, |_| vec![1, 2, 3]);
        s.step();
        assert_eq!(s.view().seqs.len(), 1);
        drop(rx);
        s.step();
        assert!(s.active_sequences.read().is_empty() && s.waiting_queue.read().is_empty());
    }

    /// Counting a prompt (Anthropic count_tokens) neither queues a sequence nor takes an id.
    #[test]
    fn test_prompt_len_queues_nothing() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(never_eos_model())),
        );
        assert_eq!(s.prompt_len(&[], |_| vec![1, 2, 3]).unwrap(), 3);
        assert!(s.waiting_queue.read().is_empty());
        let (id, ..) = s.submit_with_params(SamplingParams::default(), |_| vec![1, 2, 3]);
        assert_eq!(id, 0, "prompt_len consumed a sequence id");
        assert_eq!(s.waiting_queue.read().len(), 1);
    }

    /// A model swap fails in-flight requests instead of closing them like finished ones.
    #[test]
    fn test_swap_model_fails_in_flight_requests() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(never_eos_model())),
        );
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1000,
            top_logprobs: None,
        };
        let (_, _rx, _, _, running) = s.submit_with_params(params, |_| vec![1, 2, 3]);
        s.step();
        let (_, _rx2, _, _, queued) = s.submit_with_params(params, |_| vec![4, 5]);
        s.swap_model("other", cpu_exec(never_eos_model()));
        assert!(finish_outcome(&running).is_err() && finish_outcome(&queued).is_err());
    }

    #[test]
    fn test_prompt_ingestion_lock_cleanliness_and_state() {
        let kv_pool = Arc::new(PagedKVPool::new(64));
        let scheduler = ContinuousBatchScheduler::new("", kv_pool, None);

        // submit_request_stream must drop waiting lock cleanly
        let (_, _rx) = scheduler.submit_request_stream(vec![10, 20, 30]);

        // Verify waiting lock can immediately be acquired (no leaked lock)
        let waiting = scheduler.waiting_queue.try_write();
        assert!(
            waiting.is_some(),
            "waiting_queue lock must be released cleanly"
        );
        let w = waiting.unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].state, SequenceState::Waiting);
        assert_eq!(w[0].prefilled_tokens, 0);
    }

    /// GraphLm whose tokenizer marks no id special, so generation can only stop on `max_new_tokens`.
    fn never_eos_model() -> GraphLm {
        let mut model = GraphLm::new(GraphLmConfig::smollm_1m());
        model.tokenizer.special_tokens.clear();
        model.tokenizer.inv_vocab.clear();
        model
    }

    fn run_to_completion(
        scheduler: &ContinuousBatchScheduler,
        mut rx: UnboundedReceiver<u32>,
    ) -> Vec<u32> {
        for _ in 0..1000 {
            scheduler.step();
            if scheduler.active_sequences.read().is_empty()
                && scheduler.waiting_queue.read().is_empty()
            {
                break;
            }
        }
        let mut out = Vec::new();
        while let Ok(t) = rx.try_recv() {
            out.push(t);
        }
        out
    }

    #[test]
    fn test_max_new_tokens_stops_exactly() {
        for n in [1usize, 7, 100] {
            let scheduler = ContinuousBatchScheduler::new(
                "",
                Arc::new(PagedKVPool::new(512)),
                Some(cpu_exec(never_eos_model())),
            );
            let params = SamplingParams {
                temperature: Some(0.0),
                top_p: None,
                max_new_tokens: n,
                top_logprobs: None,
            };
            let (_, rx, _, _, finish) = scheduler.submit_with_params(params, |_| vec![1, 2, 3]);
            assert_eq!(
                run_to_completion(&scheduler, rx).len(),
                n,
                "max_new_tokens={n}"
            );
            assert_eq!(
                finish.get(),
                Some(&FinishReason::Length),
                "max_new_tokens={n}"
            );
        }
    }

    /// Generation ends at the model's context length (its KV cache capacity), not past it.
    #[test]
    fn test_generation_stops_at_model_context_length() {
        let model = never_eos_model();
        let ctx = model.config.max_seq_len;
        let scheduler = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(model)),
        );
        // Unclamped submit path: only the step's context check can stop this sequence.
        let (_, rx) = scheduler.submit_request_stream(vec![7u32; ctx - 12]);
        assert_eq!(run_to_completion(&scheduler, rx).len(), 12);
    }

    #[test]
    fn test_temperature_zero_picks_argmax() {
        let model = never_eos_model();
        let prompt = [5u32, 9, 13];
        let mut cache: Vec<crate::model::graph_lm::LayerKVCache> = (0..model.layers.len())
            .map(|_| {
                crate::model::graph_lm::LayerKVCache::new(
                    model.config.max_seq_len,
                    model.config.hidden_dim,
                )
            })
            .collect();
        let mut logits = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            logits = model.forward_token_with_kv(t, p, &mut cache);
        }
        // The sampler only considers ids the tokenizer can decode.
        let vocab = match model.tokenizer.vocab_size() {
            0 => logits.len(),
            v => v.min(logits.len()),
        };
        let argmax = logits[..vocab]
            .iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (i, &v)| if v > b.1 { (i, v) } else { b },
            )
            .0 as u32;

        let scheduler = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(model)),
        );
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1,
            top_logprobs: None,
        };
        let (_, rx, _, _, _) = scheduler.submit_with_params(params, |_| prompt.to_vec());
        assert_eq!(run_to_completion(&scheduler, rx), vec![argmax]);
    }

    /// Synthetic weights tie the top-2 logits to within one f32 ulp (margin 0..4.8e-7 at |logit|
    /// 5.64), so greedy token streams may legitimately diverge between backends. Compare
    /// teacher-forced logits instead; 80 positions cross the GPU KV cache's 64-row growth.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_metal_scheduler_serves_max_new_tokens() {
        let model = ExecutableModel::load_graph_lm(never_eos_model(), Device::Metal).unwrap();
        assert_eq!(model.llm_backend(), Device::Metal);
        let scheduler =
            ContinuousBatchScheduler::new("", Arc::new(PagedKVPool::new(512)), Some(model));
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 12,
            top_logprobs: None,
        };
        let (_, rx, _, _, _) = scheduler.submit_with_params(params, |_| vec![5, 9, 13, 2, 77]);
        let out = run_to_completion(&scheduler, rx);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|&t| (t as usize) < 2048));
    }
    #[test]
    fn test_program_scheduler_prefill_and_decode_parity() {
        let raw_model = never_eos_model();
        let vocab = raw_model.config.vocab_size;
        let prompt = vec![5u32, 9, 13, 2];

        // 1. Reference: GraphLm forward
        let exp_next = raw_model.forward(&prompt);
        let exp_tok =
            raw_model.sample_next_token_with_penalty(&exp_next, &prompt, 0.0, 50, 1.0, 1.0);

        // 2. Load as ExecutableModel::Program via CPU load_graph_lm
        let exec_model = ExecutableModel::load_graph_lm(raw_model.clone(), Device::Cpu).unwrap();
        assert!(
            matches!(exec_model, ExecutableModel::Program(..)),
            "expected ExecutableModel::Program variant on CPU load_graph_lm"
        );
        assert_eq!(exec_model.llm_backend(), Device::Cpu);

        // 2a. Verify chunked prefill path
        let mut chunk_kv = None;
        let chunk1_tok = exec_model
            .step_prefill_chunk(&prompt[..2], 0, false, &prompt, 0.0, 1.0, &mut chunk_kv)
            .expect("chunk 1 prefill succeeds");
        assert_eq!(chunk1_tok, None);

        let chunk2_tok = exec_model
            .step_prefill_chunk(&prompt[2..], 2, true, &prompt, 0.0, 1.0, &mut chunk_kv)
            .expect("chunk 2 prefill succeeds");
        assert_eq!(
            chunk2_tok,
            Some(exp_tok),
            "chunked prefill greedy token must match reference"
        );

        // 2b. Verify step_token_with_kv single-call prefill + decode
        let mut kv = None;
        let next_tok = exec_model
            .step_token_with_kv(
                &prompt,
                *prompt.last().unwrap(),
                prompt.len(),
                0.0,
                1.0,
                None,
                &mut kv,
            )
            .map(|(t, _)| t)
            .expect("step_token_with_kv succeeds");

        assert!(
            (next_tok as usize) < vocab,
            "token must be < vocab: {next_tok} vs {vocab}"
        );
        assert_eq!(
            next_tok, exp_tok,
            "Program greedy token must match GraphLm forward"
        );

        // Step decode
        let mut full_prompt = prompt.clone();
        full_prompt.push(next_tok);
        let dec_tok = exec_model
            .step_token_with_kv(
                &full_prompt,
                next_tok,
                full_prompt.len(),
                0.0,
                1.0,
                None,
                &mut kv,
            )
            .map(|(t, _)| t)
            .expect("decode step succeeds");
        assert!((dec_tok as usize) < vocab, "decode token must be < vocab");
        // Compare next token generated by Program against forward_token_with_kv
        let mut ref_kv = new_cpu_kv(&raw_model);
        for (p, &t) in prompt.iter().enumerate() {
            let _ = raw_model.forward_token_with_kv(t, p, &mut ref_kv);
        }
        let ref_logits = raw_model.forward_token_with_kv(next_tok, prompt.len(), &mut ref_kv);
        let exp_dec_tok =
            raw_model.sample_next_token_with_penalty(&ref_logits, &full_prompt, 0.0, 50, 1.0, 1.0);
        // Note: As noted in line 1833, synthetic weights tie top-2 logits to within 1 ULP (margin ~0),
        // so compare logit margin / logit tolerance:
        let prog_val = ref_logits[dec_tok as usize];
        let exp_val = ref_logits[exp_dec_tok as usize];
        assert!(
            (prog_val - exp_val).abs() < 1e-4,
            "Program picked token {dec_tok} (logit {prog_val}) vs reference {exp_dec_tok} (logit {exp_val})"
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_program_scheduler_metal_parity() {
        let raw_model = never_eos_model();
        let vocab = raw_model.config.vocab_size;
        let prompt = vec![5u32, 9, 13, 2];

        // 1. Reference: GraphLm CPU forward
        let exp_next = raw_model.forward(&prompt);
        let exp_tok =
            raw_model.sample_next_token_with_penalty(&exp_next, &prompt, 0.0, 50, 1.0, 1.0);

        // 2. Load as ExecutableModel::Program on Metal
        let exec_model = ExecutableModel::load_graph_lm(raw_model.clone(), Device::Metal).unwrap();
        assert!(
            matches!(exec_model, ExecutableModel::Program(..)),
            "expected ExecutableModel::Program variant on Metal load_graph_lm"
        );
        assert_eq!(exec_model.llm_backend(), Device::Metal);
        assert!(exec_model.backend_label().starts_with("Metal ("));

        // 2a. Chunked prefill on Metal
        let mut chunk_kv = None;
        let chunk1_tok = exec_model
            .step_prefill_chunk(&prompt[..2], 0, false, &prompt, 0.0, 1.0, &mut chunk_kv)
            .expect("Metal chunk 1 prefill succeeds");
        assert_eq!(chunk1_tok, None);

        let chunk2_tok = exec_model
            .step_prefill_chunk(&prompt[2..], 2, true, &prompt, 0.0, 1.0, &mut chunk_kv)
            .expect("Metal chunk 2 prefill succeeds");
        // Synthetic weights tie the top logits to within an ulp, so compare the picked token's reference
        // logit, not its id (as the sibling test does).
        let tied = |tok: u32, what: &str| {
            let (got, want) = (exp_next[tok as usize], exp_next[exp_tok as usize]);
            assert!(
                (got - want).abs() < 1e-4,
                "{what}: picked token {tok} (reference logit {got}) vs reference {exp_tok} (logit {want})"
            );
        };
        tied(
            chunk2_tok.expect("last chunk samples a token"),
            "Metal chunked prefill greedy token must match CPU reference",
        );

        // 2b. Single-call prefill + decode on Metal
        let mut kv = None;
        let next_tok = exec_model
            .step_token_with_kv(
                &prompt,
                *prompt.last().unwrap(),
                prompt.len(),
                0.0,
                1.0,
                None,
                &mut kv,
            )
            .map(|(t, _)| t)
            .expect("Metal step_token_with_kv succeeds");
        tied(
            next_tok,
            "Metal Program greedy token must match GraphLm forward",
        );

        // 2c. Sequential decode step on Metal
        let mut full_prompt = prompt.clone();
        full_prompt.push(next_tok);
        let dec_tok = exec_model
            .step_token_with_kv(
                &full_prompt,
                next_tok,
                full_prompt.len(),
                0.0,
                1.0,
                None,
                &mut kv,
            )
            .map(|(t, _)| t)
            .expect("Metal decode step succeeds");
        assert!((dec_tok as usize) < vocab);

        let mut ref_kv = new_cpu_kv(&raw_model);
        for (p, &t) in prompt.iter().enumerate() {
            let _ = raw_model.forward_token_with_kv(t, p, &mut ref_kv);
        }
        let ref_logits = raw_model.forward_token_with_kv(next_tok, prompt.len(), &mut ref_kv);
        let prog_logits = match &exec_model {
            ExecutableModel::Program(p, _) => {
                let mut prog = p.write();
                let inps = vec![
                    crate::graph::Tensor::U32(vec![next_tok]),
                    crate::graph::Tensor::U32(vec![prompt.len() as u32]),
                ];
                match prog.run(inps).unwrap().remove(0) {
                    crate::graph::Tensor::F32(v) => v,
                    _ => panic!("expected f32 logits"),
                }
            }
            _ => unreachable!(),
        };
        let max_diff = prog_logits
            .iter()
            .zip(&ref_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "Max logits difference {max_diff} exceeds 1e-3 tolerance"
        );
        let exp_dec_tok =
            raw_model.sample_next_token_with_penalty(&ref_logits, &full_prompt, 0.0, 50, 1.0, 1.0);
        let prog_val = ref_logits[dec_tok as usize];
        let exp_val = ref_logits[exp_dec_tok as usize];
        assert!(
            (prog_val - exp_val).abs() < 1e-3,
            "Metal Program picked token {dec_tok} (logit {prog_val}) vs reference {exp_dec_tok} (logit {exp_val})"
        );

        // 2d. Batched prefill with m >= 16 (chunk of 20 tokens) on Metal
        let long_prompt: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % vocab as u32).collect();
        let mut batched_kv = None;
        let batched_tok = exec_model
            .step_prefill_chunk(
                &long_prompt,
                0,
                true,
                &long_prompt,
                0.0,
                1.0,
                &mut batched_kv,
            )
            .expect("Batched 20-token prefill succeeds");

        let ref_batched_logits = raw_model.forward(&long_prompt);
        let exp_batched_tok = raw_model.sample_next_token_with_penalty(
            &ref_batched_logits,
            &long_prompt,
            0.0,
            50,
            1.0,
            1.0,
        );
        let prog_batched_val = ref_batched_logits[batched_tok.unwrap() as usize];
        let exp_batched_val = ref_batched_logits[exp_batched_tok as usize];
        assert!(
            (prog_batched_val - exp_batched_val).abs() < 1e-3,
            "Batched prefill token logit {prog_batched_val} vs reference {exp_batched_val}"
        );

        // Decode 1 step after batched prefill
        let next_b_tok = batched_tok.unwrap();
        let mut full_long = long_prompt.clone();
        full_long.push(next_b_tok);
        let dec_b_tok = exec_model
            .step_token_with_kv(
                &full_long,
                next_b_tok,
                full_long.len(),
                0.0,
                1.0,
                None,
                &mut batched_kv,
            )
            .map(|(t, _)| t)
            .expect("Decode step after batched prefill succeeds");
        assert!((dec_b_tok as usize) < vocab);
    }
}
