//! Weight-format selection, run at model load before the backend compiles its kernels.
//!
//! Stage 1 times every weight format the backend has LLM kernels for on one decoder layer of the
//! model being served: decode streams each weight once per token, so that GEMV time ranks the
//! formats on this host. Stage 2 quantizes only the formats that beat f32 there, teacher-forces
//! a probe text through the whole model, and keeps the fastest whose next-token distribution
//! stays within [`MAX_MEAN_KL`] of f32. `--quant` overrides the choice. Decisions are cached per
//! host CPU, backend, model weights, and engine binary under `<cache>/host/`, so a rebuilt
//! engine (new kernels) or a changed checkpoint is measured again.

use crate::hardware::{default_cache_dir, Device, HardwareCapability, QuantMode};
use crate::model::graph_lm::{projection_formats, requantized, GraphLm, LayerKVCache};
use crate::model::hf_loader::load_causal_lm;
use crate::model::hub::ModelArtifacts;
use crate::model::layers::QuantizedLinearLayer;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Quality floor: largest mean KL divergence (nats) of a format's next-token distribution from
/// the f32 one over the probe. Roughly where 6-bit block formats land; 4-bit ones exceed it.
pub const MAX_MEAN_KL: f64 = 0.01;
/// A format replaces the current pick only if it is this much faster; near-ties keep the more
/// exact format.
const MIN_SPEEDUP: f64 = 1.05;
/// Teacher-forced probe positions; the first `WARMUP` are not timed.
const PROBE_TOKENS: usize = 48;
const WARMUP: usize = 4;
/// After this many positions, a format that is not already faster than the pick is abandoned.
const EARLY_EXIT: usize = 16;
/// Runs per GEMV in stage 1; the fastest counts.
const GEMV_REPS: usize = 8;
const PROBE_TEXT: &str = "The engine measures each weight format on this machine before it serves \
    the model. It keeps the fastest format whose predictions stay close to full precision, because \
    a format that is fast but wrong does not help anyone who reads the answers it writes.";

/// CPU weight formats, most exact first. TurboSqueeze stores the same NVFP4 weights as
/// NVFP4Ultra, so it is not a separate candidate.
const CPU_CANDIDATES: &[QuantMode] = &[
    QuantMode::Uncompressed,
    QuantMode::F16,
    QuantMode::BF16,
    QuantMode::Q8_0,
    QuantMode::EXL3Quality,
    QuantMode::EXL3Balanced,
    QuantMode::AdaptiveHybrid,
    QuantMode::NVFP4Ultra,
    QuantMode::EXL3Aggressive,
    QuantMode::WeightlessJesse,
];

/// Weight formats `backend` has LLM kernels for; the first is the f32 reference.
pub fn candidates(backend: Device) -> &'static [QuantMode] {
    match backend {
        Device::Cpu => CPU_CANDIDATES,
        // ponytail: Metal LLM kernels read f32 weights only; f16/bf16/int8 GEMV (PLAN P5) join here.
        _ => &[QuantMode::Uncompressed],
    }
}

/// One measured format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub mode: QuantMode,
    /// Stage 1: GEMV time over one decoder layer's projections (µs).
    pub layer_gemv_us: f64,
    /// Stage 2: decode time per token over the probe; `None` when not probed to the end.
    pub ms_per_token: Option<f64>,
    /// Stage 2: mean KL divergence from f32 (nats); `None` when not probed to the end.
    pub mean_kl: Option<f64>,
    /// Stage 2: share of probe positions whose argmax equals f32's.
    pub top1: Option<f64>,
    pub verdict: String,
}

/// The format the engine serves a model in, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    key: String,
    pub mode: QuantMode,
    pub trials: Vec<Trial>,
    #[serde(skip)]
    pub cache_hit: bool,
    /// Set by `--quant` rather than measured.
    #[serde(skip)]
    pub overridden: bool,
}

impl Choice {
    /// How the format was picked, e.g. `chosen by the engine (cached)` or `set by --quant`.
    pub fn origin(&self) -> &'static str {
        match (self.overridden, self.cache_hit) {
            (true, _) => "set by --quant",
            (false, true) => "chosen by the engine (cached)",
            (false, false) => "chosen by the engine",
        }
    }

    /// One line for the startup log.
    pub fn summary(&self) -> String {
        let head = format!(
            "{:?} ({:.2} bpw), {}",
            self.mode,
            self.mode.effective_bits_per_weight(),
            self.origin()
        );
        if self.overridden {
            return head;
        }
        let trials: Vec<String> = self
            .trials
            .iter()
            .map(|t| format!("{:?}: {}", t.mode, t.verdict))
            .collect();
        format!("{head} [{}]", trials.join("; "))
    }
}

/// Loads `artifacts` for `backend` in `quant_override`, or in the format [`choose`] picks.
pub fn load_causal_lm_for(
    artifacts: &ModelArtifacts,
    backend: Device,
    quant_override: Option<QuantMode>,
    hw: &HardwareCapability,
) -> Result<(GraphLm, Choice), String> {
    if let Some(mode) = quant_override {
        // TurboSqueeze stores NVFP4Ultra's weights, so it is not a separate CPU candidate.
        let supported = candidates(backend).contains(&mode)
            || (backend == Device::Cpu && mode == QuantMode::TurboSqueeze);
        if !supported {
            return Err(format!(
                "--quant {mode:?}: {backend} LLM kernels read {:?} weights only",
                candidates(backend)
            ));
        }
        let llm = load_causal_lm(artifacts, mode)?;
        let choice = Choice {
            key: String::new(),
            mode,
            trials: Vec::new(),
            cache_hit: false,
            overridden: true,
        };
        return Ok((llm, choice));
    }
    let params = artifacts.estimated_params as u64;
    if let Some(total) = hw.total_memory_bytes.filter(|&t| !f32_fits(params, t)) {
        return load_without_f32(artifacts, backend, params, total);
    }
    let mut llm = load_causal_lm(artifacts, QuantMode::Uncompressed)?;
    let cache = default_cache_dir().ok();
    let choice = choose(&llm, &model_key(artifacts), backend, hw, cache.as_deref());
    if choice.mode != QuantMode::Uncompressed {
        llm.quantize(choice.mode);
    }
    Ok((llm, choice))
}

/// An f32 copy of the weights must leave room for the format probe's quantized copies.
pub fn f32_fits(params: u64, total_memory: u64) -> bool {
    params.saturating_mul(4) <= total_memory / 2
}

/// The most exact CPU format of at least 3 bits per weight (the 0.5-bit lookup-table format does
/// not preserve weights) whose size, plus 10 % for caches and activations, fits in 70 % of
/// memory: what a model too large for f32 loads as.
pub fn fitting_mode(params: u64, total_memory: u64) -> Option<QuantMode> {
    CPU_CANDIDATES[1..].iter().copied().find(|m| {
        let bpw = f64::from(m.effective_bits_per_weight());
        bpw >= 3.0 && params as f64 * bpw / 8.0 * 1.1 <= total_memory as f64 * 0.7
    })
}

/// Loads a model whose f32 copy would not fit straight into [`fitting_mode`]: every projection is
/// encoded as it is read, so the f32 model never exists. The format probe needs that f32
/// reference, so the format is picked by size alone (ponytail: no GEMV timing or KL gate here).
fn load_without_f32(
    artifacts: &ModelArtifacts,
    backend: Device,
    params: u64,
    total: u64,
) -> Result<(GraphLm, Choice), String> {
    let gb = |b: f64| b / (1u64 << 30) as f64;
    if backend != Device::Cpu {
        return Err(format!(
            "needs ~{:.0} GB as f32 and {backend} LLM kernels read f32 weights only; load it with --backend cpu, which holds quantized weights",
            gb(params as f64 * 4.0)
        ));
    }
    let mode = fitting_mode(params, total).ok_or_else(|| {
        format!(
            "needs ~{:.0} GB even in the smallest weight format; host has {:.0} GB",
            gb(params as f64 * 3.125 / 8.0),
            gb(total as f64)
        )
    })?;
    let llm = load_causal_lm(artifacts, mode)?;
    let verdict = format!(
        "f32 weights (~{:.0} GB) do not fit in {:.0} GB; encoded while loading, not probed",
        gb(params as f64 * 4.0),
        gb(total as f64)
    );
    let trial = Trial {
        mode,
        layer_gemv_us: 0.0,
        ms_per_token: None,
        mean_kl: None,
        top1: None,
        verdict,
    };
    let choice = Choice {
        key: String::new(),
        mode,
        trials: vec![trial],
        cache_hit: false,
        overridden: false,
    };
    Ok((llm, choice))
}

/// Identity of the weights being served: model id plus weight-file size and mtime.
pub fn model_key(artifacts: &ModelArtifacts) -> String {
    let file = artifacts
        .weights_path
        .as_deref()
        .and_then(|p| p.metadata().ok())
        .map_or_else(|| "no-weight-file".into(), |m| file_stamp(&m));
    format!("{}|{file}", artifacts.model_id)
}

/// Picks the weight format for `reference` (f32 weights) on `backend`, from the cache under
/// `cache_dir` when this exact host, backend, model, and engine build were measured before.
pub fn choose(
    reference: &GraphLm,
    model_key: &str,
    backend: Device,
    hw: &HardwareCapability,
    cache_dir: Option<&Path>,
) -> Choice {
    let formats = candidates(backend);
    if formats.len() == 1 {
        return Choice {
            key: String::new(),
            mode: formats[0],
            trials: vec![Trial {
                mode: formats[0],
                layer_gemv_us: 0.0,
                ms_per_token: None,
                mean_kl: None,
                top1: None,
                verdict: format!("the only weight format {backend} LLM kernels read"),
            }],
            cache_hit: false,
            overridden: false,
        };
    }

    let key = format!(
        "{model_key}|{backend}|{}|{}|{formats:?}|{MAX_MEAN_KL}",
        hw.cpu_name.as_deref().unwrap_or("unknown-cpu"),
        engine_build()
    );
    let path = cache_dir.map(|d| {
        d.join("host")
            .join(format!("weights-{:016x}.json", fnv1a(key.as_bytes())))
    });
    let cached = path
        .as_deref()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Choice>(&b).ok())
        .filter(|c| c.key == key);
    if let Some(c) = cached {
        return Choice {
            cache_hit: true,
            ..c
        };
    }

    let choice = measure(reference, formats, key, hw.total_memory_bytes);
    if let Some(p) = &path {
        // A failed write only costs a re-measure on the next load.
        let _ =
            std::fs::create_dir_all(p.parent().expect("cache file has a parent")).and_then(|()| {
                std::fs::write(p, serde_json::to_vec_pretty(&choice).unwrap_or_default())
            });
    }
    choice
}

fn measure(
    reference: &GraphLm,
    formats: &[QuantMode],
    key: String,
    total_memory: Option<u64>,
) -> Choice {
    // Stage 1: GEMV time of one decoder layer in each format.
    let layer = &reference.layers[reference.layers.len() / 2];
    let layer_us = |mode| {
        let (attn, ffn, _) = projection_formats(mode);
        gemv_us(&[
            requantized(&layer.qkv_proj, attn),
            requantized(&layer.out_proj, attn),
            requantized(&layer.gate_up_proj, ffn),
            requantized(&layer.down_proj, ffn),
        ])
    };
    // The reference is timed first and last and keeps its faster run, so a cold first
    // measurement (thread pool, caches, clocks) does not count against f32.
    let first_us = layer_us(formats[0]);
    let others: Vec<f64> = formats[1..].iter().map(|&m| layer_us(m)).collect();
    let f32_us = first_us.min(layer_us(formats[0]));
    let trial = |mode, us: f64, verdict| Trial {
        mode,
        layer_gemv_us: us,
        ms_per_token: None,
        mean_kl: None,
        top1: None,
        verdict,
    };
    let mut trials = vec![trial(
        formats[0],
        f32_us,
        format!("{f32_us:.0} µs/layer GEMV, reference"),
    )];
    let mut survivors = Vec::new();
    for (&mode, &us) in formats[1..].iter().zip(&others) {
        let faster = us * MIN_SPEEDUP <= f32_us;
        if faster {
            survivors.push(trials.len());
        }
        let note = if faster {
            "faster than f32"
        } else {
            "not faster than f32"
        };
        trials.push(trial(
            mode,
            us,
            format!("{us:.0} µs/layer GEMV ({:.2}x f32), {note}", us / f32_us),
        ));
    }
    let done = |trials, mode| Choice {
        key,
        mode,
        trials,
        cache_hit: false,
        overridden: false,
    };
    if survivors.is_empty() {
        return done(trials, formats[0]);
    }

    // Stage 2: whole-model probe of the survivors. Each holds an f32 copy beside the reference.
    let f32_bytes = reference.parameter_count() as u64 * 4;
    if total_memory.is_some_and(|total| 2 * f32_bytes > total / 2) {
        for &i in &survivors {
            trials[i].verdict += "; not probed: two f32 copies exceed half of memory";
        }
        return done(trials, formats[0]);
    }
    let tokens = probe_tokens(reference);
    let (ref_logp, ref_ms) = reference_probe(reference, &tokens);
    trials[0].ms_per_token = Some(ref_ms);
    trials[0].verdict += &format!(", probe {ref_ms:.2} ms/token");
    let mut best = (formats[0], ref_ms);

    for i in survivors {
        let mode = trials[i].mode;
        let timed = probe_format(reference, mode, &tokens, &ref_logp, best.1 / MIN_SPEEDUP);
        let t = &mut trials[i];
        let Some((ms, kl, top1)) = timed else {
            t.verdict += &format!("; probe abandoned: not faster than {:?}", best.0);
            continue;
        };
        (t.ms_per_token, t.mean_kl, t.top1) = (Some(ms), Some(kl), Some(top1));
        t.verdict += &format!(
            "; probe {ms:.2} ms/token, mean KL {kl:.4}, top-1 {:.0}%",
            top1 * 100.0
        );
        // Negated so a NaN KL (non-finite logits) is rejected too.
        if !(kl <= MAX_MEAN_KL) {
            t.verdict += &format!(", rejected: KL above {MAX_MEAN_KL}");
        } else if ms * MIN_SPEEDUP <= best.1 {
            t.verdict += &format!(", faster than {:?}", best.0);
            best = (mode, ms);
        } else {
            t.verdict += &format!(", not faster than {:?}", best.0);
        }
    }
    done(trials, best.0)
}

/// Summed best-of-[`GEMV_REPS`] time (µs) of one GEMV through each of `layers`.
fn gemv_us(layers: &[QuantizedLinearLayer]) -> f64 {
    layers
        .iter()
        .map(|l| {
            let x: Vec<f32> = (0..l.in_features)
                .map(|i| (i % 17) as f32 / 17.0 - 0.5)
                .collect();
            let mut out = vec![0.0f32; l.out_features];
            (0..GEMV_REPS)
                .map(|_| {
                    let t = Instant::now();
                    l.forward(&x, &mut out);
                    t.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum()
}

/// `PROBE_TEXT` tokens, repeated to `PROBE_TOKENS` and kept inside the model's vocab and context.
fn probe_tokens(model: &GraphLm) -> Vec<u32> {
    let vocab = model.config.vocab_size.max(1) as u32;
    let mut ids = model.tokenizer.encode(PROBE_TEXT);
    if ids.is_empty() {
        ids = (1..=PROBE_TOKENS as u32).collect();
    }
    let n = PROBE_TOKENS
        .min(model.config.max_seq_len)
        .max(EARLY_EXIT + 1);
    ids.iter().cycle().take(n).map(|t| t % vocab).collect()
}

/// Teacher-forces `tokens` through `model`, handing each position's logits to `visit`. Returns
/// forward-pass ms per token after `WARMUP` (`visit` is not timed), or `None` once that pace at
/// `EARLY_EXIT` exceeds `bail_ms`.
fn probe(
    model: &GraphLm,
    tokens: &[u32],
    bail_ms: f64,
    mut visit: impl FnMut(usize, &[f32]),
) -> Option<f64> {
    let mut kv: Vec<LayerKVCache> = (0..model.layers.len())
        .map(|_| LayerKVCache::new(model.config.max_seq_len.max(1), model.config.hidden_dim))
        .collect();
    let mut forward = Duration::ZERO;
    let pace = |t: Duration, n: usize| t.as_secs_f64() * 1e3 / n as f64;
    for (pos, &tok) in tokens.iter().enumerate() {
        let started = Instant::now();
        let logits = model.forward_token_with_kv(tok, pos, &mut kv);
        if pos >= WARMUP {
            forward += started.elapsed();
        }
        visit(pos, &logits);
        if pos + 1 == EARLY_EXIT && pace(forward, EARLY_EXIT - WARMUP) > bail_ms {
            return None;
        }
    }
    Some(pace(forward, tokens.len() - WARMUP))
}

/// f32 log-probabilities at every probe position, and the f32 decode ms per token.
fn reference_probe(reference: &GraphLm, tokens: &[u32]) -> (Vec<Vec<f32>>, f64) {
    let mut logp = Vec::with_capacity(tokens.len());
    let ms = probe(reference, tokens, f64::INFINITY, |_, logits| {
        logp.push(log_softmax(logits))
    })
    .expect("an unbounded probe always finishes");
    (logp, ms)
}

/// `reference` quantized to `mode` and teacher-forced over `tokens`: (ms per token, mean KL
/// from `ref_logp`, top-1 agreement), or `None` if abandoned as slower than `bail_ms`.
fn probe_format(
    reference: &GraphLm,
    mode: QuantMode,
    tokens: &[u32],
    ref_logp: &[Vec<f32>],
    bail_ms: f64,
) -> Option<(f64, f64, f64)> {
    let mut model = reference.clone();
    model.quantize(mode);
    let (mut kl, mut agree) = (0.0f64, 0usize);
    let ms = probe(&model, tokens, bail_ms, |pos, logits| {
        let q = log_softmax(logits);
        kl += kl_divergence(&ref_logp[pos], &q);
        agree += usize::from(argmax(&ref_logp[pos]) == argmax(&q));
    })?;
    let n = tokens.len() as f64;
    Some((ms, kl / n, agree as f64 / n))
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits
        .iter()
        .map(|&x| ((x - max) as f64).exp())
        .sum::<f64>()
        .ln() as f32;
    logits.iter().map(|&x| x - max - log_sum).collect()
}

/// KL(p ‖ q) from log-probabilities.
fn kl_divergence(log_p: &[f32], log_q: &[f32]) -> f64 {
    log_p
        .iter()
        .zip(log_q)
        .map(|(&p, &q)| (p as f64).exp() * (p - q) as f64)
        .sum()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &x)| if x > b.1 { (i, x) } else { b },
        )
        .0
}

fn file_stamp(m: &std::fs::Metadata) -> String {
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    format!("{}-{mtime}", m.len())
}

/// The running binary's size and mtime: a rebuild may change kernels, so it re-measures.
fn engine_build() -> String {
    std::env::current_exe()
        .and_then(|p| p.metadata())
        .map_or_else(|_| format!("{:?}", SystemTime::now()), |m| file_stamp(&m))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph_lm::GraphLmConfig;

    /// The quality floor on real weights: f16 and Q8_0 pass it, 4-bit NVFP4 and the LUT do not.
    #[test]
    fn quality_gate_accepts_near_lossless_and_rejects_lossy_formats() {
        if cfg!(debug_assertions) {
            return;
        }
        let dir = Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
        if !dir.join("model.safetensors").exists() {
            eprintln!("SKIP: SmolLM2-135M-Instruct not cached in disk/");
            return;
        }
        let artifacts = ModelArtifacts {
            model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".into(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            ..Default::default()
        };
        let reference = load_causal_lm(&artifacts, QuantMode::Uncompressed).unwrap();
        let tokens = probe_tokens(&reference);
        let (ref_logp, _) = reference_probe(&reference, &tokens);
        let kl = |mode| {
            probe_format(&reference, mode, &tokens, &ref_logp, f64::INFINITY)
                .unwrap()
                .1
        };
        for mode in [QuantMode::F16, QuantMode::Q8_0] {
            let k = kl(mode);
            eprintln!("{mode:?}: mean KL {k:.6}");
            assert!(
                k <= MAX_MEAN_KL,
                "{mode:?}: mean KL {k} should pass the gate"
            );
        }
        for mode in [QuantMode::NVFP4Ultra, QuantMode::WeightlessJesse] {
            let k = kl(mode);
            eprintln!("{mode:?}: mean KL {k:.6}");
            assert!(
                k > MAX_MEAN_KL,
                "{mode:?}: mean KL {k} should fail the gate"
            );
        }
    }

    /// A decision is reused for the same model and re-measured for different weights.
    #[test]
    fn choice_is_cached_per_model() {
        let cache = tempfile::tempdir().unwrap();
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let hw = HardwareCapability::probe_system();
        let pick = |key| choose(&model, key, Device::Cpu, &hw, Some(cache.path()));

        let first = pick("model-a|1-1");
        assert!(!first.cache_hit);
        assert_eq!(first.trials.len(), CPU_CANDIDATES.len());
        let again = pick("model-a|1-1");
        assert!(again.cache_hit);
        assert_eq!(again.mode, first.mode);
        assert!(
            !pick("model-a|2-2").cache_hit,
            "changed weights must re-measure"
        );
    }

    /// A model whose f32 copy does not fit loads straight into the most exact format that does
    /// (Q8_0 for 135M parameters in 400 MB) and still predicts what f32 predicts.
    #[test]
    fn model_too_big_for_f32_loads_quantized() {
        let dir = Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
        if !dir.join("model.safetensors").exists() {
            eprintln!("SKIP: SmolLM2-135M-Instruct not cached in disk/");
            return;
        }
        let artifacts = ModelArtifacts {
            model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".into(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            estimated_params: 135_000_000,
            ..Default::default()
        };
        let mut hw = HardwareCapability::probe_system();
        hw.total_memory_bytes = Some(400 << 20);
        assert_eq!(fitting_mode(135_000_000, 400 << 20), Some(QuantMode::Q8_0));
        let (q, choice) = load_causal_lm_for(&artifacts, Device::Cpu, None, &hw).unwrap();
        assert_eq!(choice.mode, QuantMode::Q8_0);
        assert_eq!(q.current_quant_mode(), QuantMode::Q8_0);
        let f = load_causal_lm(&artifacts, QuantMode::Uncompressed).unwrap();
        let ids = f.tokenizer.encode("The capital of France is");
        assert_eq!(argmax(&q.forward(&ids)), argmax(&f.forward(&ids)));
        let err = load_causal_lm_for(&artifacts, Device::Metal, None, &hw)
            .err()
            .unwrap();
        assert!(err.contains("--backend cpu"), "{err}");
    }
}
