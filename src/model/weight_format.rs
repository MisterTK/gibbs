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
use crate::model::hf_loader::{load_causal_lm, open_causal_lm};
use crate::model::hub::ModelArtifacts;
use crate::model::layers::QuantizedLinearLayer;
use crate::model::reencode::{self, Encoder, Mode, ReencodeProgress, ReencodeReport};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Quality floor: largest mean KL divergence (nats) of a format's next-token distribution from
/// the f32 one over the probe. Roughly where 6-bit block formats land; 4-bit ones exceed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReencodePolicy {
    MaxSpeed,
    LowBitNative,
    BitExact,
}

impl ReencodePolicy {
    /// The CLI / settings spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::MaxSpeed => "max-speed",
            Self::LowBitNative => "low-bit-native",
            Self::BitExact => "bit-exact",
        }
    }
}

impl Default for ReencodePolicy {
    fn default() -> Self {
        Self::MaxSpeed
    }
}

impl std::str::FromStr for ReencodePolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "max-speed" | "maxspeed" | "speed" => Ok(Self::MaxSpeed),
            "low-bit-native" | "lowbitnative" | "lowbit" => Ok(Self::LowBitNative),
            "bit-exact" | "bitexact" | "exact" => Ok(Self::BitExact),
            _ => Err(format!("unknown reencode policy '{s}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariantPolicy {
    Fastest,
    Fastest4Bit,
    BestQuality,
}

impl VariantPolicy {
    /// The settings / `--variant` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fastest => "fastest",
            Self::Fastest4Bit => "fastest-4bit",
            Self::BestQuality => "best-quality",
        }
    }
}

impl Default for VariantPolicy {
    fn default() -> Self {
        Self::Fastest
    }
}

impl std::str::FromStr for VariantPolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fastest" | "fast" => Ok(Self::Fastest),
            "fastest-4bit" | "fastest4bit" => Ok(Self::Fastest4Bit),
            "best-quality" | "bestquality" | "quality" => Ok(Self::BestQuality),
            _ => Err(format!("unknown variant policy '{s}'")),
        }
    }
}

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
    /// What a GPU load's re-encode policy did (`None` for CPU loads, which use `trials`).
    #[serde(skip)]
    pub reencode: Option<ReencodeReport>,
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
        if let Some(r) = &self.reencode {
            let formats: Vec<String> = r
                .formats
                .iter()
                .map(|(f, b)| format!("{f} {:.0} MB", *b as f64 / 1e6))
                .collect();
            return format!(
                "{} ({}; gate: {}{}) [{}]",
                r.applied,
                formats.join(", "),
                r.gate,
                if r.cache_hit { "; cached" } else { "" },
                r.attempts.join("; ")
            );
        }
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

/// How a GPU load re-encodes its weights (S3.5).
#[derive(Clone)]
pub struct ReencodeConfig {
    pub policy: ReencodePolicy,
    /// Overrides the KL limit of the whole-model gate.
    pub max_kl: Option<f64>,
    /// Cache root (entries live in `<dir>/reencoded/`); `None` disables the cache.
    pub cache_dir: Option<PathBuf>,
    /// Filled while tensors are re-encoded, for the load-stage text.
    pub progress: Option<Arc<ReencodeProgress>>,
    /// Receives every `reencode: ...` log line (the CLI routes it to its telemetry log).
    pub log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    /// Native loaders: only feed-forward projections are re-encoded (the second gate attempt).
    pub ffn_only: bool,
}

impl Default for ReencodeConfig {
    fn default() -> Self {
        Self {
            policy: ReencodePolicy::default(),
            max_kl: None,
            cache_dir: default_cache_dir().ok(),
            progress: None,
            log: None,
            ffn_only: false,
        }
    }
}

/// The configuration loaders use instead of the settings file when set ([`set_reencode_override`]).
static OVERRIDE: parking_lot::RwLock<Option<ReencodeConfig>> = parking_lot::RwLock::new(None);

/// Pins the re-encode configuration every loader (GraphLm and native) reads through
/// [`ReencodeConfig::effective`]; `None` returns to the settings file.
pub fn set_reencode_override(c: Option<ReencodeConfig>) {
    *OVERRIDE.write() = c;
}

/// The pinned configuration, if any (to restore after a temporary override).
pub fn reencode_override() -> Option<ReencodeConfig> {
    OVERRIDE.read().clone()
}

impl ReencodeConfig {
    /// `~/.cache/gibbs/settings.json` (`reencode`, `reencode_max_kl`), which the TUI writes.
    pub fn from_settings() -> Self {
        let s = crate::tui::settings::load_settings();
        Self {
            policy: s.reencode.parse().unwrap_or_default(),
            max_kl: s.reencode_max_kl.map(f64::from),
            ..Self::default()
        }
    }

    /// The override ([`set_reencode_override`]) when set, else the settings file.
    pub fn effective() -> Self {
        OVERRIDE.read().clone().unwrap_or_else(Self::from_settings)
    }

    /// CLI flags (`--reencode`, `--reencode-max-kl`) over the settings file.
    pub fn resolve(cli_policy: Option<&str>, cli_max_kl: Option<f32>) -> Result<Self, String> {
        let mut c = Self::from_settings();
        if let Some(p) = cli_policy {
            c.policy = p.parse()?;
        }
        if let Some(k) = cli_max_kl {
            c.max_kl = Some(f64::from(k));
        }
        Ok(c)
    }
}

/// Loads `artifacts` for `backend` in `quant_override`, or in the format [`choose`] picks (CPU) /
/// the re-encode policy chooses (Metal), with the policy from `~/.cache/gibbs/settings.json`.
pub fn load_causal_lm_for(
    artifacts: &ModelArtifacts,
    backend: Device,
    quant_override: Option<QuantMode>,
    hw: &HardwareCapability,
) -> Result<(GraphLm, Choice), String> {
    load_causal_lm_with(
        artifacts,
        backend,
        quant_override,
        hw,
        &ReencodeConfig::effective(),
    )
}

/// [`load_causal_lm_for`] with an explicit re-encode configuration.
pub fn load_causal_lm_with(
    artifacts: &ModelArtifacts,
    backend: Device,
    quant_override: Option<QuantMode>,
    hw: &HardwareCapability,
    reencode: &ReencodeConfig,
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
            reencode: None,
        };
        return Ok((llm, choice));
    }
    // A Metal-placed load never measures formats on the CPU: the policy picks them and the GPU gates them.
    if backend == Device::Metal {
        return load_metal(artifacts, hw, reencode);
    }
    // `--backend cpu` (and hosts without Metal): measure the CPU formats.
    let params = artifacts.estimated_params as u64;
    if let Some(total) = hw.total_memory_bytes.filter(|&t| !f32_fits(params, t)) {
        return load_without_f32(artifacts, backend, params, total);
    }
    let mut llm = load_causal_lm(artifacts, QuantMode::Uncompressed)?;
    let choice = choose(
        &llm,
        &model_key(artifacts),
        backend,
        hw,
        reencode.cache_dir.as_deref(),
    );
    if choice.mode != QuantMode::Uncompressed {
        llm.quantize(choice.mode);
    }
    Ok((llm, choice))
}

/// Mean KL limit and top-1 floor of the whole-model gate.
fn gate_limits(policy: ReencodePolicy, max_kl: Option<f64>) -> (f64, f64) {
    let (kl, top1) = match policy {
        ReencodePolicy::MaxSpeed => (0.10, 0.90),
        _ => (MAX_MEAN_KL, 0.99),
    };
    (max_kl.unwrap_or(kl), top1)
}

/// Whether a checkpoint tensor is a gather table (never streamed as a matrix).
fn gathered(name: &str) -> bool {
    ["embd", "embed", "wte", "wpe"]
        .iter()
        .any(|s| name.contains(s))
}

fn load_metal(
    artifacts: &ModelArtifacts,
    hw: &HardwareCapability,
    cfg: &ReencodeConfig,
) -> Result<(GraphLm, Choice), String> {
    let src = open_causal_lm(artifacts, QuantMode::Uncompressed)?;
    let threads = hw
        .performance_cores
        .map_or_else(
            || std::thread::available_parallelism().map_or(2, |n| n.get()),
            |c| c as usize,
        )
        .saturating_sub(1)
        .max(1);
    let log = |m: String| {
        tracing::info!("{m}");
        if let Some(f) = &cfg.log {
            f(&m);
        }
    };
    let progress = cfg.progress.clone().unwrap_or_default();
    progress
        .total
        .store(src.projection_count(), std::sync::atomic::Ordering::Relaxed);
    let key = artifacts
        .weights_path
        .as_deref()
        .and_then(|p| reencode::cache_key(p, cfg.policy, cfg.max_kl));
    let done = |llm: GraphLm, report: ReencodeReport, trials: Vec<Trial>| {
        let choice = Choice {
            key: String::new(),
            mode: QuantMode::Uncompressed,
            trials,
            cache_hit: report.cache_hit,
            overridden: false,
            reencode: Some(report),
        };
        (llm, choice)
    };
    let trial = |verdict: String, kl: Option<f64>, top1: Option<f64>| Trial {
        mode: QuantMode::Uncompressed,
        layer_gemv_us: 0.0,
        ms_per_token: None,
        mean_kl: kl,
        top1,
        verdict,
    };

    // Cache: a hit skips encoding and the gate. A cached tensor that does not match this model
    // (or a corrupt entry) deletes the entry and falls through to a fresh encode.
    if let (Some(dir), Some(k)) = (&cfg.cache_dir, key) {
        if let Some(reader) = reencode::CacheReader::open(dir, k) {
            let report = reader.report();
            let enc = Encoder::new(Mode::Cache(Arc::new(reader)), threads, None);
            let llm = src.build(Some(&enc))?;
            if !enc.cache_miss.load(std::sync::atomic::Ordering::Relaxed) {
                log(format!(
                    "reencode: cache hit ({}), gate: {}",
                    report.applied, report.gate
                ));
                let t = trial(
                    format!("cached: {}", report.gate),
                    report.mean_kl,
                    report.top1,
                );
                return Ok(done(llm, report, vec![t]));
            }
            let (bin, json) = reencode::cache_paths(dir, k);
            let _ = (std::fs::remove_file(bin), std::fs::remove_file(json));
            log("reencode: cache entry does not match this model; re-encoding".into());
        }
    }

    let mut report = ReencodeReport {
        requested: cfg.policy.name().into(),
        ..Default::default()
    };
    let policy_encoder = |policy, ffn_only, rmse_limit| {
        progress.done.store(0, std::sync::atomic::Ordering::Relaxed);
        Encoder::new(
            Mode::Policy {
                policy,
                ffn_only,
                rmse_limit,
            },
            threads,
            Some(progress.clone()),
        )
    };
    let file_bytes = artifacts
        .weights_path
        .as_deref()
        .and_then(|p| p.metadata().ok())
        .map_or(0, |m| m.len());
    // Only tensors a projection can read: `*.weight` matrices (companions such as `.absmax`,
    // `.scales`, `_packed`, `_zero_point` are not weights), counted codec-less unless the GPU reads
    // their format or they repack into a codec.
    let codec_less = src.checkpoint().is_some_and(|(m, data)| {
        m.tensors.values().any(|t| {
            t.shape.len() == 2
                && t.name.ends_with(".weight")
                && !gathered(&t.name)
                && !reencode::has_gpu_codec(reencode::declared_format(t))
                && !matches!(crate::model::repack::repack(m, data, &t.name), Ok(Some(_)))
        })
    });
    let too_big = hw
        .total_memory_bytes
        .is_some_and(|ram| file_bytes > ram / 2);

    // Per-tensor gate: no source program to compare against.
    if codec_less || too_big {
        let why = if codec_less {
            "a source tensor has no GPU codec".to_string()
        } else {
            format!(
                "source {:.1} GB > 50% of {:.1} GB RAM",
                file_bytes as f64 / 1e9,
                hw.total_memory_bytes.unwrap_or(0) as f64 / 1e9
            )
        };
        log(format!("reencode gate: per-tensor ({why})"));
        log(format!(
            "re-encoding for this GPU ({}): {} tensors",
            cfg.policy.name(),
            progress.total.load(std::sync::atomic::Ordering::Relaxed)
        ));
        let enc = policy_encoder(cfg.policy, false, Some(reencode::TENSOR_RMSE_LIMIT));
        let llm = src.build(Some(&enc))?;
        report.applied = cfg.policy.name().into();
        report.gate = format!("per-tensor RMSE <= {} ({why})", reencode::TENSOR_RMSE_LIMIT);
        report.from_records(&enc.records());
        let verdict = format!("{}: {}", report.applied, report.gate);
        write_cache_entry(cfg, key, &llm, &report);
        return Ok(done(llm, report, vec![trial(verdict, None, None)]));
    }

    // Whole-model gate: teacher-force the probe text through the source program and each candidate,
    // both compiled on Metal. The source model is dropped once its log-probabilities are kept.
    let attempts: &[(&str, ReencodePolicy, bool)] = match cfg.policy {
        ReencodePolicy::MaxSpeed => &[
            ("max-speed", ReencodePolicy::MaxSpeed, false),
            ("max-speed (FFN only)", ReencodePolicy::MaxSpeed, true),
            ("low-bit-native", ReencodePolicy::LowBitNative, false),
            ("bit-exact", ReencodePolicy::BitExact, false),
        ],
        ReencodePolicy::LowBitNative => &[
            ("low-bit-native", ReencodePolicy::LowBitNative, false),
            ("bit-exact", ReencodePolicy::BitExact, false),
        ],
        ReencodePolicy::BitExact => &[("bit-exact", ReencodePolicy::BitExact, false)],
    };
    let mut reference: Option<(Vec<u32>, Vec<Vec<f32>>)> = None;
    let mut trials = Vec::new();
    for (i, &(label, policy, ffn_only)) in attempts.iter().enumerate() {
        log(format!(
            "re-encoding for this GPU ({}): {} tensors (attempt {}/{})",
            policy.name(),
            progress.total.load(std::sync::atomic::Ordering::Relaxed),
            i + 1,
            attempts.len()
        ));
        let enc = policy_encoder(policy, ffn_only, None);
        let llm = src.build(Some(&enc))?;
        let records = enc.records();
        let finish =
            |mut report: ReencodeReport, gate: String, kl, top1, mut trials: Vec<Trial>| {
                report.applied = label.into();
                report.gate = gate.clone();
                (report.mean_kl, report.top1) = (kl, top1);
                report.from_records(&records);
                trials.push(trial(format!("{label}: {gate}"), kl, top1));
                write_cache_entry(cfg, key, &llm, &report);
                (report, trials)
            };
        // Native storage everywhere (or the last resort) needs no gate.
        let lossy = records.iter().any(|r| r.lossy);
        if policy == ReencodePolicy::BitExact || !lossy {
            report.attempts.push(format!("{label}: nothing re-encoded"));
            let gate = if lossy {
                "bit-exact (fallback)"
            } else {
                "bit-exact"
            };
            let (r, t) = finish(report, gate.into(), None, None, trials);
            log(format!("reencode: {label} applied ({gate})"));
            return Ok(done(llm, r, t));
        }
        if reference.is_none() {
            let exact = src.build(Some(&Encoder::new(Mode::Source, threads, None)))?;
            let tokens = probe_tokens(&exact);
            let logp = gpu_logprobs(&exact, &tokens)?;
            reference = Some((tokens, logp));
        }
        let (tokens, ref_logp) = reference.as_ref().expect("set above");
        let cand = gpu_logprobs(&llm, tokens)?;
        let n = tokens.len() as f64;
        let kl = ref_logp
            .iter()
            .zip(&cand)
            .map(|(p, q)| kl_divergence(p, q))
            .sum::<f64>()
            / n;
        let top1 = ref_logp
            .iter()
            .zip(&cand)
            .filter(|(p, q)| argmax(p) == argmax(q))
            .count() as f64
            / n;
        let (kl_limit, top1_floor) = gate_limits(policy, cfg.max_kl);
        let gate = format!("mean KL {kl:.4} nats, top-1 {:.1}%", top1 * 100.0);
        // Negated so a NaN KL (non-finite logits) is rejected too.
        if kl <= kl_limit && top1 >= top1_floor {
            report.attempts.push(format!("{label}: {gate}, accepted"));
            log(format!("reencode: {label} accepted ({gate})"));
            let (r, t) = finish(report, gate, Some(kl), Some(top1), trials);
            return Ok(done(llm, r, t));
        }
        let next = attempts[i + 1].0;
        log(format!(
            "reencode: {label} rejected (mean KL {kl:.4} nats, top-1 {:.1}%); trying {next}",
            top1 * 100.0
        ));
        report.attempts.push(format!("{label}: {gate}, rejected"));
        trials.push(trial(
            format!("{label}: {gate}, rejected (limits KL {kl_limit}, top-1 {top1_floor})"),
            Some(kl),
            Some(top1),
        ));
    }
    unreachable!("the last attempt (bit-exact) always returns")
}

fn write_cache_entry(
    cfg: &ReencodeConfig,
    key: Option<u64>,
    llm: &GraphLm,
    report: &ReencodeReport,
) {
    if let (Some(dir), Some(k)) = (&cfg.cache_dir, key) {
        if let Err(e) = reencode::write_cache(dir, k, llm, report) {
            tracing::warn!("reencode: cache not written: {e}");
        }
    }
}

/// Log-probabilities of the next token at every position of teacher-forced `tokens`, from the model
/// compiled as a GraphProgram pinned to Metal (the CPU forward is never used).
#[cfg(target_os = "macos")]
fn gpu_logprobs(model: &GraphLm, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    use crate::graph::{program::GraphProgram, Tensor};
    use crate::hardware::PlacementMode;
    let (graph, weights) = model.to_decode_graph();
    let mut prog = GraphProgram::compile(graph, weights, PlacementMode::Pin(Device::Metal))?;
    if prog.cpu_node_count() != 0 {
        return Err("reencode gate: the probe program has CPU nodes".into());
    }
    let mut out = Vec::with_capacity(tokens.len());
    for (pos, &tok) in tokens.iter().enumerate() {
        let r = prog.run(vec![Tensor::U32(vec![tok]), Tensor::U32(vec![pos as u32])])?;
        match r.into_iter().next() {
            Some(Tensor::F32(logits)) => out.push(log_softmax(&logits)),
            _ => return Err("reencode gate: the probe program did not return f32 logits".into()),
        }
    }
    Ok(out)
}

#[cfg(not(target_os = "macos"))]
fn gpu_logprobs(_: &GraphLm, _: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    Err("the Metal backend requires macOS".into())
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
        reencode: None,
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
            reencode: None,
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
        reencode: None,
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

/// Result of comparing a candidate model's next-token distributions with the source model's.
#[derive(Debug, Clone, Copy)]
pub struct GateReport {
    pub mean_kl: f64,
    pub top1: f64,
    pub tokens: usize,
}

/// Teacher-forces [`PROBE_TEXT`] one token at a time through `source` and `candidate` (both on
/// Metal) and compares the next-token distributions: mean KL(source ‖ candidate) in nats and top-1
/// agreement, over the same positions the GraphLm gate uses.
pub fn native_kl_gate(
    source: &dyn crate::model::native::NativeLm,
    candidate: &dyn crate::model::native::NativeLm,
) -> Result<GateReport, String> {
    let tok = source.tokenizer();
    let vocab = tok.vocab_size().max(1) as u32;
    let mut ids = tok.encode(PROBE_TEXT);
    if ids.is_empty() {
        ids = (1..=PROBE_TOKENS as u32).collect();
    }
    let n = PROBE_TOKENS
        .min(source.context_len().min(candidate.context_len()))
        .max(2);
    let tokens: Vec<u32> = ids.iter().cycle().take(n).map(|t| t % vocab).collect();
    let run = |m: &dyn crate::model::native::NativeLm| -> Result<Vec<Vec<f32>>, String> {
        let (_, mut state) = m.prepare(vec![tokens[0]], &[])?;
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| {
                let logits = m.forward(&[t], pos, &mut state)?;
                Ok(log_softmax(&logits[..logits.len().min(vocab as usize)]))
            })
            .collect()
    };
    let (want, got) = (run(source)?, run(candidate)?);
    let nf = n as f64;
    let mean_kl = want
        .iter()
        .zip(&got)
        .map(|(p, q)| kl_divergence(p, q))
        .sum::<f64>()
        / nf;
    let top1 = want
        .iter()
        .zip(&got)
        .filter(|(p, q)| argmax(p) == argmax(q))
        .count() as f64
        / nf;
    Ok(GateReport {
        mean_kl,
        top1,
        tokens: n,
    })
}

/// Whether a gate result passes the policy's limits (a NaN KL never does).
pub fn gate_passes(policy: ReencodePolicy, max_kl: Option<f64>, r: &GateReport) -> bool {
    let (kl_limit, top1_floor) = gate_limits(policy, max_kl);
    r.mean_kl <= kl_limit && r.top1 >= top1_floor
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
    fn host_weight_caches(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir.join("host"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("weights-"))
            .collect()
    }

    fn smollm_artifacts() -> Option<ModelArtifacts> {
        let dir = Path::new("disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main");
        dir.join("model.safetensors")
            .exists()
            .then(|| ModelArtifacts {
                model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".into(),
                config_path: Some(dir.join("config.json")),
                tokenizer_path: Some(dir.join("tokenizer.json")),
                weights_path: Some(dir.join("model.safetensors")),
                ..Default::default()
            })
    }

    /// A Metal-placed load applies the re-encode policy (gated on Metal) and never runs the CPU
    /// format probe, so no `host/weights-*.json` appears.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_load_applies_policy_without_cpu_probe() {
        if !crate::hardware::available_devices().contains(&Device::Metal) {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let progress = Arc::new(ReencodeProgress::default());
        let cfg = ReencodeConfig {
            policy: ReencodePolicy::MaxSpeed,
            max_kl: None,
            cache_dir: Some(tmp.path().to_path_buf()),
            progress: Some(progress.clone()),
            log: None,
            ffn_only: false,
        };
        let artifacts = ModelArtifacts {
            model_id: "gibbs/smollm-1m".into(),
            ..Default::default()
        };
        let hw = HardwareCapability::probe_system();
        let (llm, choice) =
            load_causal_lm_with(&artifacts, Device::Metal, None, &hw, &cfg).unwrap();

        assert!(
            host_weight_caches(tmp.path()).is_empty(),
            "Metal load wrote a CPU format-probe cache"
        );
        assert!(
            !tmp.path().join("host").exists(),
            "Metal load touched the CPU probe cache dir"
        );
        let r = choice
            .reencode
            .clone()
            .expect("a Metal load reports its re-encode");
        assert!(!r.attempts.is_empty() && !r.applied.is_empty(), "{r:?}");
        assert!(choice.summary().starts_with(&r.applied));
        // 4 layers x 4 projections + LM head, every one counted
        assert_eq!(
            progress.total.load(std::sync::atomic::Ordering::Relaxed),
            17
        );
        assert_eq!(progress.done.load(std::sync::atomic::Ordering::Relaxed), 17);
        assert_eq!(r.tensors, 17);
        // whatever the gate kept, the model is an f32-free GPU model unless it fell back to bit-exact
        if r.applied != "bit-exact" {
            assert_ne!(
                llm.lm_head.quant_format(),
                crate::hardware::QuantFormat::FP32
            );
        }
    }

    /// The GPU kernels for the formats `max-speed` picks (Q4_K, block-32 Q4_0) agree with the CPU
    /// dequantizing GEMV on the same encoded weights, so a high KL in the gate is quantization error,
    /// not a kernel fault.
    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_matches_cpu_on_max_speed_weights() {
        let Some(artifacts) = smollm_artifacts() else {
            eprintln!("SKIP: SmolLM2-135M-Instruct not cached in disk/");
            return;
        };
        if cfg!(debug_assertions) || !crate::hardware::available_devices().contains(&Device::Metal)
        {
            return;
        }
        let src = open_causal_lm(&artifacts, QuantMode::Uncompressed).unwrap();
        let enc = Encoder::new(
            Mode::Policy {
                policy: ReencodePolicy::MaxSpeed,
                ffn_only: false,
                rmse_limit: None,
            },
            4,
            None,
        );
        let cand = src.build(Some(&enc)).unwrap();
        let formats: std::collections::BTreeSet<_> =
            enc.records().iter().map(|r| r.stored.name()).collect();
        eprintln!("stored formats: {formats:?}");
        let tokens = probe_tokens(&cand);
        let (cpu, _) = reference_probe(&cand, &tokens);
        let gpu = gpu_logprobs(&cand, &tokens).unwrap();
        let kl = cpu
            .iter()
            .zip(&gpu)
            .map(|(p, q)| kl_divergence(p, q))
            .sum::<f64>()
            / tokens.len() as f64;
        eprintln!("CPU vs GPU on the re-encoded model: mean KL {kl:.6}");
        assert!(
            kl < 1e-3,
            "GPU and CPU disagree on identical weights: KL {kl}"
        );
    }

    /// Control for the test above: a CPU load does measure formats and writes the probe cache into the
    /// configured directory, so the Metal assertion has teeth.
    #[test]
    fn cpu_load_writes_the_format_probe_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = ReencodeConfig {
            cache_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let artifacts = ModelArtifacts {
            model_id: "gibbs/smollm-1m".into(),
            ..Default::default()
        };
        let hw = HardwareCapability::probe_system();
        load_causal_lm_with(&artifacts, Device::Cpu, None, &hw, &cfg).unwrap();
        assert_eq!(host_weight_caches(tmp.path()).len(), 1);
    }

    /// Real weights: the first Metal load encodes, gates and caches; the second is a cache hit that
    /// reproduces the same GPU log-probabilities, and neither touches the CPU probe cache.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_load_caches_reencoded_weights() {
        let Some(artifacts) = smollm_artifacts() else {
            eprintln!("SKIP: SmolLM2-135M-Instruct not cached in disk/");
            return;
        };
        if cfg!(debug_assertions) || !crate::hardware::available_devices().contains(&Device::Metal)
        {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let cfg = ReencodeConfig {
            policy: ReencodePolicy::MaxSpeed,
            max_kl: None,
            cache_dir: Some(tmp.path().to_path_buf()),
            progress: None,
            log: None,
            ffn_only: false,
        };
        let hw = HardwareCapability::probe_system();
        let (first, c1) = load_causal_lm_with(&artifacts, Device::Metal, None, &hw, &cfg).unwrap();
        let r1 = c1.reencode.clone().unwrap();
        eprintln!("first load: {}", c1.summary());
        assert!(!r1.cache_hit);
        let entries: Vec<String> = std::fs::read_dir(tmp.path().join("reencoded"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().any(|n| n.ends_with(".bin"))
                && entries.iter().any(|n| n.ends_with(".json")),
            "{entries:?}"
        );
        assert!(!entries.iter().any(|n| n.ends_with(".tmp")), "{entries:?}");

        let (second, c2) = load_causal_lm_with(&artifacts, Device::Metal, None, &hw, &cfg).unwrap();
        eprintln!("second load: {}", c2.summary());
        assert!(c2.reencode.as_ref().unwrap().cache_hit);
        assert_eq!(c2.reencode.as_ref().unwrap().applied, r1.applied);
        let tokens = probe_tokens(&first);
        assert_eq!(
            gpu_logprobs(&first, &tokens).unwrap(),
            gpu_logprobs(&second, &tokens).unwrap()
        );
        assert!(host_weight_caches(tmp.path()).is_empty());
    }

    /// A tight `--reencode-max-kl` rejects every lossy attempt and ends at the bit-exact fallback.
    #[cfg(target_os = "macos")]
    #[test]
    fn kl_gate_falls_back_to_bit_exact() {
        let Some(artifacts) = smollm_artifacts() else {
            eprintln!("SKIP: SmolLM2-135M-Instruct not cached in disk/");
            return;
        };
        if cfg!(debug_assertions) || !crate::hardware::available_devices().contains(&Device::Metal)
        {
            return;
        }
        let cfg = ReencodeConfig {
            policy: ReencodePolicy::MaxSpeed,
            max_kl: Some(1e-9),
            cache_dir: None,
            progress: None,
            log: None,
            ffn_only: false,
        };
        let hw = HardwareCapability::probe_system();
        let (_, c) = load_causal_lm_with(&artifacts, Device::Metal, None, &hw, &cfg).unwrap();
        let r = c.reencode.clone().unwrap();
        eprintln!("{}", c.summary());
        assert_eq!(r.applied, "bit-exact", "{r:?}");
        assert_eq!(
            r.attempts.iter().filter(|a| a.contains("rejected")).count(),
            3,
            "{r:?}"
        );
    }

    /// Gate thresholds: max-speed KL <= 0.10 and top-1 >= 0.90; low-bit-native KL <= 0.01 (top-1 >= 0.99);
    /// a NaN KL never passes; an explicit `max_kl` overrides the KL limit only.
    #[test]
    fn gate_thresholds_per_policy() {
        let r = |mean_kl: f64, top1: f64| GateReport { mean_kl, top1, tokens: 8 };
        use ReencodePolicy::*;
        assert!(gate_passes(MaxSpeed, None, &r(0.10, 0.90)));
        assert!(!gate_passes(MaxSpeed, None, &r(0.1001, 0.95)));
        assert!(!gate_passes(MaxSpeed, None, &r(0.01, 0.8999)));
        assert!(gate_passes(LowBitNative, None, &r(0.01, 0.99)));
        assert!(!gate_passes(LowBitNative, None, &r(0.011, 1.0)));
        assert!(!gate_passes(MaxSpeed, None, &r(f64::NAN, 1.0)));
        assert!(gate_passes(LowBitNative, Some(0.05), &r(0.04, 0.99)));
        assert!(!gate_passes(MaxSpeed, Some(0.02), &r(0.05, 1.0)));

    }
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
        // On Metal the same model loads through the policy with the per-tensor gate (its source
        // program would not fit), never as f32.
        let cfg = ReencodeConfig {
            cache_dir: None,
            ..Default::default()
        };
        let (m, choice) = load_causal_lm_with(&artifacts, Device::Metal, None, &hw, &cfg).unwrap();
        let r = choice.reencode.unwrap();
        assert!(r.gate.starts_with("per-tensor RMSE"), "{r:?}");
        assert_ne!(m.lm_head.quant_format(), crate::hardware::QuantFormat::FP32);
        assert!(r.lossy > 0 && r.tensors == 4 * m.layers.len() + 1, "{r:?}");
        // and it still predicts what the f32 model predicts (on the GPU: no CPU forward)
        #[cfg(target_os = "macos")]
        if crate::hardware::available_devices().contains(&Device::Metal) {
            let last = |lp: Vec<Vec<f32>>| argmax(lp.last().unwrap());
            assert_eq!(
                last(gpu_logprobs(&m, &ids).unwrap()),
                last(gpu_logprobs(&f, &ids).unwrap())
            );
        }
    }
}
