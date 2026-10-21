use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDefaults {
    #[serde(default = "default_prompt")]
    pub prompt: String,
    #[serde(default = "default_decode_tokens")]
    pub decode_tokens: usize,
    #[serde(default = "default_runs")]
    pub runs: usize,
    #[serde(default = "default_reencode")]
    pub reencode: String,
}

/// The prompt a benchmark decodes from unless the suite names one.
pub fn default_prompt() -> String {
    "Explain the second law of thermodynamics to a curious student.".into()
}
fn default_decode_tokens() -> usize {
    128
}
fn default_runs() -> usize {
    3
}
/// The re-encode policy a model loads under unless the suite names one.
pub fn default_reencode() -> String {
    "max-speed".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchModel {
    pub id: String,
    #[serde(default)]
    pub min_tok_s: Option<f64>,
    #[serde(default)]
    pub reference: bool,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub decode_tokens: Option<usize>,
    #[serde(default)]
    pub runs: Option<usize>,
    #[serde(default)]
    pub reencode: Option<String>,
    /// METRIC slug override (default: the file stem of `id`).
    #[serde(default)]
    pub name: Option<String>,
    /// Prompt text file, relative to the suite file's directory; overrides `prompt`.
    #[serde(default)]
    pub prompt_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSuite {
    #[serde(default)]
    pub defaults: Option<BenchDefaults>,
    pub models: Vec<BenchModel>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBenchResult {
    pub model: String,
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_mix: Option<String>,
    pub bytes_per_token: u64,
    /// `streamed` (matrices a decode step reads) or `file` (whole checkpoint; overstates).
    pub bytes_basis: &'static str,
    pub ceiling_gbps: f64,
    pub decode_tok_s: f64,
    pub roofline_tok_s: f64,
    pub roofline_pct: f64,
    /// Median wall time per decode token (1000 / tok/s).
    pub wall_ms: f64,
    /// Median GPU time of the token's command buffer (absent when the model has no Metal run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_ms: Option<f64>,
    /// Median time inside `run_prepared` (inputs, encode, GPU wait, output readback).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_ms: Option<f64>,
    /// `wall_ms - gpu_ms`: everything that is not the GPU executing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dispatches: Option<f64>,
    /// Time the streamed bytes take at the measured ceiling: the floor of a token.
    pub weights_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tok_s: Option<f64>,
    pub rss_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_kernels: Option<Vec<String>>,
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

/// A model loaded for measurement and what its decode step streams.
pub struct BenchLoad {
    pub exec: crate::scheduler::ExecutableModel,
    /// The re-encode policy whose formats the model's projections ended up in: what the KL
    /// fallback chain applied (GraphLm loads and, via the same chain, native loads).
    pub policy: crate::model::weight_format::ReencodePolicy,
    /// Bytes per token that native loaders saved against the source blocks under `policy`
    /// (those are counted from the header with [`ReencodePolicy::BitExact`]); 0 for GraphLm loads.
    pub saved_bytes: u64,
}

/// `(policy, saved bytes per token)` a finished load applied: the roofline basis of
/// [`roofline_bytes`]. `choice` is a GraphLm load's weight-format choice (`requested` when it
/// names no policy); `None` reads the native loader's record, where tensors keep their source
/// formats (bit-exact) unless it saved bytes.
pub fn applied_policy(
    choice: Option<&crate::model::weight_format::Choice>,
    requested: crate::model::weight_format::ReencodePolicy,
) -> (crate::model::weight_format::ReencodePolicy, u64) {
    use crate::model::weight_format::ReencodePolicy;
    let Some(choice) = choice else {
        let (policy, saved) =
            crate::model::reencode::native_applied().unwrap_or((ReencodePolicy::BitExact, 0));
        return (
            if saved > 0 {
                ReencodePolicy::BitExact
            } else {
                policy
            },
            saved,
        );
    };
    // `applied` reads `max-speed`, `max-speed (FFN only)`, `low-bit-native` or `bit-exact`.
    // ponytail: the FFN-only attempt is costed as full max-speed; exact bytes need per-tensor records.
    let policy = choice
        .reencode
        .as_ref()
        .and_then(|r| r.applied.split(" (").next()?.parse::<ReencodePolicy>().ok())
        .unwrap_or(requested);
    (policy, 0)
}

/// Loads `artifacts` on Metal under the `reencode` policy name (the effective policy of every
/// loader during the load).
pub fn load_for_bench(
    artifacts: &crate::model::hub::ModelArtifacts,
    reencode: &str,
) -> Result<BenchLoad, String> {
    let cfg = crate::model::weight_format::ReencodeConfig {
        policy: reencode.parse()?,
        ..Default::default()
    };
    let policy = cfg.policy;
    let (exec, choice) = crate::arch::dispatch::load_lm(
        artifacts,
        &crate::arch::LoadOpts {
            placement: crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
            token: None,
            quant: None,
            quant_name: None,
            hw: &crate::hardware::HardwareCapability::probe_system(),
            reencode: cfg,
            family: None,
            stage: &|_| {},
            cli_routes: false,
            admit: &|_| Ok(()),
        },
    )?;
    let (policy, saved_bytes) = match &choice {
        None => applied_policy(None, policy),
        Some(c) => (applied_policy(Some(c), policy).0, 0),
    };
    Ok(BenchLoad {
        exec,
        policy,
        saved_bytes,
    })
}

/// The roofline numerator for `artifacts` loaded as `load` describes: matrices a decode step
/// streams after the applied policy (gathered tables excluded) and `"streamed"`, or the file size
/// and `"file"` when no header can be read.
pub fn roofline_bytes(
    artifacts: &crate::model::hub::ModelArtifacts,
    policy: crate::model::weight_format::ReencodePolicy,
    saved_bytes: u64,
) -> (u64, &'static str) {
    let streamed = artifacts
        .weights_path
        .as_ref()
        .and_then(|p| crate::model::variant::read_local_header(p).ok())
        .map(|h| {
            h.streamed_bytes_per_token(policy)
                .saturating_sub(saved_bytes)
        })
        .filter(|&b| b > 0);
    match streamed {
        Some(b) => (b, "streamed"),
        None => (bench_file_bytes(artifacts), "file"),
    }
}

/// Warm-up plus `runs` timed generations of `decode_tokens` tokens with drafters off; medians of
/// the per-token wall, GPU and host times against the roofline given by `bytes_per_token`.
/// `model` and `file` are left for the caller to fill.
pub fn measure_loaded(
    exec: &crate::scheduler::ExecutableModel,
    prompt: &str,
    decode_tokens: usize,
    runs: usize,
    bytes_per_token: u64,
    ceiling_gbps: f64,
) -> Result<ModelBenchResult, String> {
    if bytes_per_token == 0 || ceiling_gbps <= 0.0 {
        return Err("no weight bytes found, cannot compute a roofline".into());
    }
    exec.generate_text_measured_plain(prompt, 16, 0.0)?;
    let (mut speeds, mut walls) = (Vec::new(), Vec::new());
    let (mut gpu, mut run, mut enc, mut disp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..runs {
        crate::graph::metal::take_run_samples();
        let (_, passes, dur) = exec.generate_text_measured_plain(prompt, decode_tokens, 0.0)?;
        let s = dur.as_secs_f64();
        if s > 0.0 && passes > 0 {
            speeds.push(passes as f64 / s);
            walls.push(s * 1000.0 / passes as f64);
        }
        // Prefill samples come first; the last `passes` are the decode steps.
        let samples = crate::graph::metal::take_run_samples();
        for r in samples.iter().rev().take(passes) {
            gpu.push(r.gpu_ns as f64 / 1e6);
            run.push(r.run_ns as f64 / 1e6);
            enc.push(r.encode_ns as f64 / 1e3);
            disp.push(r.dispatches as f64);
        }
    }
    let tok_s = median(&mut speeds).ok_or("no decode pass completed")?;
    let wall_ms = median(&mut walls).unwrap_or(0.0);
    let gpu_ms = median(&mut gpu);
    let roofline_tok_s = ceiling_gbps / (bytes_per_token as f64 / 1e9);
    Ok(ModelBenchResult {
        model: String::new(),
        file: String::new(),
        codec_mix: None,
        bytes_per_token,
        bytes_basis: "streamed",
        ceiling_gbps,
        decode_tok_s: tok_s,
        roofline_tok_s,
        roofline_pct: tok_s / roofline_tok_s * 100.0,
        wall_ms,
        gpu_ms,
        run_ms: median(&mut run),
        // GPU and wall are separate medians; at tiny token times GPU can exceed wall by noise.
        host_ms: gpu_ms.map(|g| (wall_ms - g).max(0.0)),
        encode_us: median(&mut enc),
        dispatches: median(&mut disp),
        weights_ms: bytes_per_token as f64 / 1e9 / ceiling_gbps * 1e3,
        prefill_tok_s: None,
        rss_gb: get_peak_rss_gb(),
        top_kernels: None,
    })
}

fn get_peak_rss_gb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        rusage.ru_maxrss as f64 / 1e9 // macOS returns bytes
    }
    #[cfg(not(target_os = "macos"))]
    0.0
}

/// Bytes of the weights file (or checkpoint directory): the roofline numerator of last resort.
fn bench_file_bytes(artifacts: &crate::model::hub::ModelArtifacts) -> u64 {
    artifacts
        .weights_path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .or_else(|| {
            artifacts.local_dir.as_ref().map(|d| {
                std::fs::read_dir(d)
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum::<u64>()
            })
        })
        .unwrap_or(0)
}

/// Routes `tracing` events (`tune:`, `reencode:` lines) to stderr when `RUST_LOG` is set.
fn init_logging() {
    if std::env::var_os("RUST_LOG").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
    }
}

/// Runs a bench suite file and prints JSON lines plus METRIC lines.
pub fn run_bench_suite(path: &str) -> Result<i32, String> {
    init_logging();
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read bench suite '{path}': {e}"))?;
    let suite: BenchSuite = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse bench suite '{path}': {e}"))?;

    let defaults = suite.defaults.unwrap_or(BenchDefaults {
        prompt: default_prompt(),
        decode_tokens: default_decode_tokens(),
        runs: default_runs(),
        reencode: default_reencode(),
    });

    let mut had_error = false;
    let mut missed_floor = false;
    let mut roofline_pcts = Vec::new();
    let mut slug_speeds = Vec::new();
    let suite_dir = std::path::Path::new(path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();

    let hub = crate::model::hub::HuggingFaceHub::new();
    let rt = crate::graph::metal::Runtime::shared().ok();
    if let Some(r) = rt.as_ref().filter(|r| !r.codec_rates_current()) {
        // The roofline ceiling and the re-encode rule both read measured codec rates.
        let _lock = crate::hardware::gpu_lock()?;
        let n = r.run_codec_bench()?;
        println!("codec-bench: measured {n} formats");
    }
    // Measured streaming ceiling of this GPU (host profile / codec bench), not the spec-sheet figure.
    let ceiling_gbps = match crate::model::variant::host_ceiling_gbps() {
        c if c > 0.0 => c,
        _ => crate::hardware::HardwareProfile::detect().memory_bandwidth_gbps,
    };

    for model_cfg in &suite.models {
        let decode_tokens = model_cfg.decode_tokens.unwrap_or(defaults.decode_tokens);
        let runs = model_cfg.runs.unwrap_or(defaults.runs);
        let reencode = model_cfg.reencode.as_ref().unwrap_or(&defaults.reencode);
        let fail = |e: &str| println!("{}", serde_json::json!({"model": model_cfg.id, "error": e}));
        let prompt = match &model_cfg.prompt_file {
            Some(f) => match std::fs::read_to_string(suite_dir.join(f)) {
                Ok(p) => p,
                Err(e) => {
                    fail(&format!("prompt_file '{f}': {e}"));
                    had_error = true;
                    continue;
                }
            },
            None => model_cfg
                .prompt
                .clone()
                .unwrap_or_else(|| defaults.prompt.clone()),
        };

        let artifacts = match hub.auto_download_and_cache(&model_cfg.id, None) {
            Ok(a) => a,
            Err(e) => {
                fail(&e);
                had_error = true;
                continue;
            }
        };

        // Held until this model's row is printed: concurrent benchmarks would corrupt each other.
        let _lock = crate::hardware::gpu_lock()?;
        let load = match load_for_bench(&artifacts, reencode) {
            Ok(x) => x,
            Err(e) => {
                fail(&e);
                had_error = true;
                continue;
            }
        };
        let model_exec = load.exec;
        let (bytes_per_token, bytes_basis) =
            roofline_bytes(&artifacts, load.policy, load.saved_bytes);
        let mut res = match measure_loaded(
            &model_exec,
            &prompt,
            decode_tokens,
            runs,
            bytes_per_token,
            ceiling_gbps,
        ) {
            Ok(r) => r,
            Err(e) => {
                fail(&e);
                had_error = true;
                continue;
            }
        };
        res.bytes_basis = bytes_basis;
        res.model = model_cfg.id.clone();
        res.file = artifacts
            .config_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| model_cfg.id.clone());
        roofline_pcts.push(res.roofline_pct);
        let slug = model_cfg.name.clone().unwrap_or_else(|| {
            model_cfg
                .id
                .split('/')
                .next_back()
                .unwrap_or(&model_cfg.id)
                .replace(".gguf", "")
        });
        slug_speeds.push((slug, res.decode_tok_s));
        if model_cfg
            .min_tok_s
            .is_some_and(|floor| res.decode_tok_s < floor)
        {
            missed_floor = true;
        }
        if let Ok(line) = serde_json::to_string(&res) {
            println!("{line}");
        }
    }

    if !roofline_pcts.is_empty() {
        let geomean = (roofline_pcts.iter().map(|&p| p.max(0.1).ln()).sum::<f64>()
            / roofline_pcts.len() as f64)
            .exp();
        let min_pct = roofline_pcts.iter().copied().fold(f64::INFINITY, f64::min);
        println!("METRIC geomean_roofline_pct={geomean:.2}");
        println!("METRIC min_roofline_pct={min_pct:.2}");
    }

    for (slug, speed) in slug_speeds {
        println!("METRIC {slug}_tok_s={speed:.1}");
    }

    if had_error {
        Ok(2)
    } else if missed_floor {
        Ok(3)
    } else {
        Ok(0)
    }
}

/// Runs parity comparison between Metal program and CPU execution for reference models.
pub fn run_bench_parity(path: &str) -> Result<i32, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read bench suite '{path}': {e}"))?;
    let suite: BenchSuite = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse bench suite '{path}': {e}"))?;

    let hub = crate::model::hub::HuggingFaceHub::new();
    let mut compared = 0usize;

    for model_cfg in &suite.models {
        if !model_cfg.reference {
            continue;
        }

        let artifacts = match hub.auto_download_and_cache(&model_cfg.id, None) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("parity: resolve error for {}: {e}", model_cfg.id);
                return Ok(4);
            }
        };
        let _lock = crate::hardware::gpu_lock()?;

        // Load Native Metal and CPU models for bit-parity verification
        // The CPU leg is the deliberate reference: only its load and forwards run under `allow`, so the
        // Metal leg stays under the GPU-only tripwire.
        let hw = crate::hardware::HardwareCapability::probe_system();
        let load = |dev| {
            let opts = crate::arch::LoadOpts {
                placement: crate::hardware::PlacementMode::Pin(dev),
                token: None,
                quant: None,
                quant_name: None,
                hw: &hw,
                reencode: crate::model::weight_format::ReencodeConfig::effective(),
                family: None,
                stage: &|_| {},
                cli_routes: false,
                admit: &|_| Ok(()),
            };
            crate::arch::dispatch::load_lm(&artifacts, &opts).map(|(e, _)| match e {
                crate::scheduler::ExecutableModel::Native(m) => Some(m),
                crate::scheduler::ExecutableModel::Program(..) => None,
            })
        };
        let loaded = load(crate::hardware::Device::Metal).and_then(|m| match m {
            Some(m) => {
                let _cpu = crate::hardware::cpu_compute::allow();
                let c = load(crate::hardware::Device::Cpu)?;
                Ok(c.map(|c| (m, c)))
            }
            None => Ok(None),
        });
        let (metal_exec, cpu_exec) = match loaded {
            Ok(Some(pair)) => pair,
            Ok(None) => {
                // GraphLm checkpoint: no separate native forward, so compare the Metal decode
                // program with the CPU graph executor on the very same graph and weights.
                match graph_lm_parity(&artifacts, &suite, model_cfg) {
                    Ok(min_cos) => {
                        eprintln!(
                            "parity: {} ok (graph executor vs Metal, 16 steps, min cos {min_cos:.5})",
                            model_cfg.id
                        );
                        compared += 1;
                        continue;
                    }
                    Err(e) => {
                        eprintln!("parity: {} FAILED: {e}", model_cfg.id);
                        return Ok(4);
                    }
                }
            }
            Err(e) => {
                eprintln!("parity: load error for {}: {e}", model_cfg.id);
                return Ok(4);
            }
        };

        let prompt = "Explain physics";
        let prompt_ids = metal_exec.tokenizer().encode(prompt);
        let (prompt_tokens, mut m_state) = metal_exec.prepare(prompt_ids, &[])?;
        let (_, mut c_state) = {
            let _cpu = crate::hardware::cpu_compute::allow();
            cpu_exec.prepare(prompt_tokens.clone(), &[])?
        };

        let mut metal_tokens = prompt_tokens.clone();
        let mut cpu_tokens = prompt_tokens.clone();
        let mut m_logits = metal_exec.forward(&prompt_tokens, 0, &mut m_state)?;
        let mut c_logits = {
            let _cpu = crate::hardware::cpu_compute::allow();
            cpu_exec.forward(&prompt_tokens, 0, &mut c_state)?
        };
        let mut min_cos = 1.0f32;

        for i in 0..8 {
            let dot: f32 = m_logits
                .iter()
                .zip(c_logits.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let norm_m: f32 = m_logits.iter().map(|&a| a * a).sum::<f32>().sqrt();
            let norm_c: f32 = c_logits.iter().map(|&b| b * b).sum::<f32>().sqrt();
            let cos = dot / (norm_m * norm_c).max(1e-9);

            let (m_top, _) = m_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            let (c_top, _) = c_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();

            min_cos = min_cos.min(cos);
            if m_top != c_top || cos < 0.999 {
                eprintln!(
                    "parity: {} step {i}: metal {m_top} != cpu {c_top} (cos {cos:.5})",
                    model_cfg.id
                );
                return Ok(4);
            }

            metal_tokens.push(m_top as u32);
            cpu_tokens.push(c_top as u32);

            m_logits = metal_exec.forward(&[m_top as u32], metal_tokens.len() - 1, &mut m_state)?;
            c_logits = {
                let _cpu = crate::hardware::cpu_compute::allow();
                cpu_exec.forward(&[c_top as u32], cpu_tokens.len() - 1, &mut c_state)?
            };
        }
        eprintln!(
            "parity: {} ok (8 steps, min cos {min_cos:.5})",
            model_cfg.id
        );
        compared += 1;
    }

    eprintln!("parity: {compared} compared");
    if compared == 0 {
        eprintln!("parity: nothing was compared");
        return Ok(4);
    }
    Ok(0)
}

/// Greedy 16-step comparison of the Metal decode program against the CPU graph executor for a
/// GraphLm checkpoint (same graph, same weights). Returns the minimum logits cosine.
fn graph_lm_parity(
    artifacts: &crate::model::hub::ModelArtifacts,
    suite: &BenchSuite,
    model_cfg: &BenchModel,
) -> Result<f32, String> {
    use crate::graph::{program::GraphProgram, Tensor};
    let hw = crate::hardware::HardwareCapability::probe_system();
    let default_reencode = suite
        .defaults
        .as_ref()
        .map_or_else(default_reencode, |d| d.reencode.clone());
    let cfg = model_cfg
        .reencode
        .as_ref()
        .unwrap_or(&default_reencode)
        .parse()
        .map(|policy| crate::model::weight_format::ReencodeConfig {
            policy,
            ..Default::default()
        })?;
    let (lm, _) = crate::model::weight_format::load_causal_lm_with(
        artifacts,
        crate::hardware::Device::Metal,
        None,
        &hw,
        &cfg,
    )?;
    let (g, w) = lm.to_decode_graph();
    let mut metal = GraphProgram::compile(
        g.clone(),
        w.clone(),
        crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
    )?;
    let _allow = crate::hardware::cpu_compute::allow();
    let mut cpu = GraphProgram::compile(
        g,
        w,
        crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
    )?;
    let logits = |p: &mut GraphProgram, tok: u32, pos: usize| -> Result<Vec<f32>, String> {
        match p
            .run(vec![Tensor::U32(vec![tok]), Tensor::U32(vec![pos as u32])])?
            .into_iter()
            .next()
        {
            Some(Tensor::F32(v)) => Ok(v),
            _ => Err("program did not return f32 logits".into()),
        }
    };
    // Fixed small token ids (valid in every vocabulary), then greedy continuation.
    let prompt = [1u32, 450, 3250, 1129, 29, 94, 311, 20];
    let mut min_cos = 1.0f32;
    let mut tok = 0u32;
    for pos in 0..prompt.len() + 8 {
        if pos < prompt.len() {
            tok = prompt[pos];
        }
        let (m, c) = (logits(&mut metal, tok, pos)?, logits(&mut cpu, tok, pos)?);
        let dot: f32 = m.iter().zip(&c).map(|(a, b)| a * b).sum();
        let nm = m.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nc = c.iter().map(|a| a * a).sum::<f32>().sqrt();
        let cos = dot / (nm * nc).max(1e-9);
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i)
        };
        min_cos = min_cos.min(cos);
        if cos < 0.999 {
            return Err(format!(
                "step {pos}: cos {cos:.5} (metal top {}, cpu top {})",
                top(&m),
                top(&c)
            ));
        }
        tok = top(&c) as u32;
    }
    Ok(min_cos)
}

/// `--probe-top5`: raw (no chat template) prompt logits of `model_id` on Metal, as one JSON line
/// `{"model", "ids", "top5": [[id, logit], ...]}`; the llama.cpp oracle compares against it.
pub fn probe_top5(model_id: &str, prompt: &str) -> Result<(), String> {
    let artifacts =
        crate::model::hub::HuggingFaceHub::new().auto_download_and_cache(model_id, None)?;
    let _lock = crate::hardware::gpu_lock()?;
    let exec = load_for_bench(&artifacts, "max-speed")?.exec;
    let ids = exec.tokenizer().encode(prompt);
    let logits = exec.prompt_logits(ids.clone())?;
    let vocab = exec.tokenizer().vocab_size();
    let valid = if vocab > 0 {
        logits.len().min(vocab)
    } else {
        logits.len()
    };
    let mut top: Vec<(usize, f32)> = logits[..valid].iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top.truncate(5);
    println!(
        "{}",
        serde_json::json!({"model": model_id, "ids": ids, "top5": top})
    );
    Ok(())
}

/// `--explain-placement`: loads `model_id` on Metal under the effective re-encode policy and
/// prints where its graph nodes run (op kind, device, avenue) and whether that is 100% on the GPU.
/// Returns the process exit code: 0 only when every node runs on Metal ([`placement_exit_code`]).
pub fn explain_placement(
    hub: &crate::model::hub::HuggingFaceHub,
    model_id: &str,
) -> Result<i32, String> {
    use crate::graph::placement::{placement_table, placement_verdict};
    let artifacts = hub.auto_download_and_cache(model_id, None)?;
    let _lock = crate::hardware::gpu_lock()?;
    let policy = crate::model::weight_format::ReencodeConfig::effective().policy;
    let exec = load_for_bench(&artifacts, policy.name())?.exec;
    let rows = exec
        .placement_summary()
        .ok_or("this model runs no graph program: there is no placement to explain")?;
    println!("{}", placement_table(&rows));
    let code = placement_exit_code(&rows);
    println!("{}", placement_verdict(&rows).unwrap_or_else(|cpu| cpu));
    Ok(code)
}

/// 0 when every placed node runs on Metal, else 1 (CPU nodes, another device, or no nodes).
pub fn placement_exit_code(rows: &[crate::graph::placement::PlacementRow]) -> i32 {
    let metal = crate::hardware::Device::Metal;
    i32::from(rows.is_empty() || rows.iter().any(|r| r.device != metal))
}

#[cfg(test)]
mod placement_exit_tests {
    use super::placement_exit_code;
    use crate::graph::placement::PlacementRow;
    use crate::hardware::Device;

    fn row(device: Device) -> PlacementRow {
        PlacementRow { op: "Linear".into(), device, avenue: "kernel".into(), count: 1 }
    }

    #[test]
    fn explain_placement_exits_zero_only_when_everything_is_metal() {
        assert_eq!(placement_exit_code(&[row(Device::Metal), row(Device::Metal)]), 0);
        assert_eq!(placement_exit_code(&[row(Device::Metal), row(Device::Cpu)]), 1);
        assert_eq!(placement_exit_code(&[row(Device::Cpu)]), 1);
        assert_eq!(placement_exit_code(&[]), 1);
    }
}

#[cfg(test)]
mod suite_tests {
    /// Every committed suite parses as `run_bench_suite` reads it, with a valid re-encode policy per row.
    #[test]
    fn bench_suites_parse() {
        for path in ["bench/loop.json", "bench/reference-7b.json"] {
            let text = std::fs::read_to_string(path).unwrap();
            let suite: super::BenchSuite = serde_json::from_str(&text).unwrap();
            assert!(!suite.models.is_empty(), "{path}: no models");
            for m in &suite.models {
                let policy = m
                    .reencode
                    .clone()
                    .or_else(|| suite.defaults.as_ref().map(|d| d.reencode.clone()))
                    .unwrap_or_else(super::default_reencode);
                policy
                    .parse::<crate::model::weight_format::ReencodePolicy>()
                    .unwrap();
            }
        }
    }
}
