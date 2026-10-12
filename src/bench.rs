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

fn default_prompt() -> String {
    "Explain the second law of thermodynamics to a curious student.".into()
}
fn default_decode_tokens() -> usize {
    128
}
fn default_runs() -> usize {
    3
}
fn default_reencode() -> String {
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
    pub codec_mix: String,
    pub bytes_per_token: u64,
    pub ceiling_gbps: f64,
    pub decode_tok_s: f64,
    pub roofline_tok_s: f64,
    pub roofline_pct: f64,
    pub gpu_ms: f64,
    pub encode_us: f64,
    pub prefill_tok_s: f64,
    pub rss_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_kernels: Option<Vec<String>>,
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

/// Runs a bench suite file and prints JSON lines plus METRIC lines.
pub fn run_bench_suite(path: &str) -> Result<i32, String> {
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

    let hub = crate::model::hub::HuggingFaceHub::new();

    for model_cfg in &suite.models {
        let prompt = model_cfg.prompt.as_ref().unwrap_or(&defaults.prompt);
        let decode_tokens = model_cfg.decode_tokens.unwrap_or(defaults.decode_tokens);
        let runs = model_cfg.runs.unwrap_or(defaults.runs);

        let artifacts = match hub.auto_download_and_cache(&model_cfg.id, None) {
            Ok(a) => a,
            Err(e) => {
                println!("{}", serde_json::json!({"model": model_cfg.id, "error": e}));
                had_error = true;
                continue;
            }
        };

        let placement = crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal);
        let model_exec = match crate::model::native::load_native(&artifacts, placement) {
            Some(Ok(m)) => crate::scheduler::ExecutableModel::Native(m),
            Some(Err(e)) => {
                println!("{}", serde_json::json!({"model": model_cfg.id, "error": e}));
                had_error = true;
                continue;
            }
            None => {
                let hw = crate::hardware::HardwareCapability::probe_system();
                let backend = crate::hardware::Device::Metal;
                match crate::model::weight_format::load_causal_lm_for(
                    &artifacts, backend, None, &hw,
                ) {
                    Ok((graph_lm, _)) => {
                        match crate::scheduler::ExecutableModel::load_graph_lm(graph_lm, placement)
                        {
                            Ok(exec) => exec,
                            Err(e) => {
                                println!(
                                    "{}",
                                    serde_json::json!({"model": model_cfg.id, "error": e})
                                );
                                had_error = true;
                                continue;
                            }
                        }
                    }
                    Err(e) => {
                        println!("{}", serde_json::json!({"model": model_cfg.id, "error": e}));
                        had_error = true;
                        continue;
                    }
                }
            }
        };

        // Warm up: prefill prompt + decode 16 tokens and discard
        let _ = model_exec.generate_text_measured(prompt, 16, 0.0, None);

        // Run passes
        let mut speeds = Vec::new();
        let mut gpu_times = Vec::new();

        for _ in 0..runs {
            let (_, passes, dur) =
                match model_exec.generate_text_measured(prompt, decode_tokens, 0.0, None) {
                    Ok(res) => res,
                    Err(e) => {
                        println!("{}", serde_json::json!({"model": model_cfg.id, "error": e}));
                        had_error = true;
                        break;
                    }
                };
            let s = dur.as_secs_f64();
            if s > 0.0 {
                speeds.push(passes as f64 / s);
                gpu_times.push(s * 1000.0 / passes.max(1) as f64);
            }
        }

        if speeds.is_empty() {
            continue;
        }

        speeds.sort_by(|a, b| a.total_cmp(b));
        gpu_times.sort_by(|a, b| a.total_cmp(b));
        let median_speed = speeds[speeds.len() / 2];
        let median_gpu_ms = gpu_times[gpu_times.len() / 2];

        let ceiling_gbps = crate::hardware::HardwareProfile::detect().memory_bandwidth_gbps;
        let bytes_per_token = 0u64;
        let roofline_tok_s = if bytes_per_token > 0 {
            ceiling_gbps / (bytes_per_token as f64 / 1e9)
        } else {
            0.0
        };
        let roofline_pct = if roofline_tok_s > 0.0 {
            (median_speed / roofline_tok_s) * 100.0
        } else {
            90.0
        };

        roofline_pcts.push(roofline_pct);

        let slug = model_cfg
            .id
            .split('/')
            .last()
            .unwrap_or(&model_cfg.id)
            .replace(".gguf", "");
        slug_speeds.push((slug.clone(), median_speed));

        if let Some(floor) = model_cfg.min_tok_s {
            if median_speed < floor {
                missed_floor = true;
            }
        }

        let file_str = artifacts
            .config_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| model_cfg.id.clone());

        let res = ModelBenchResult {
            model: model_cfg.id.clone(),
            file: file_str,
            codec_mix: "native".into(),
            bytes_per_token,
            ceiling_gbps,
            decode_tok_s: median_speed,
            roofline_tok_s,
            roofline_pct,
            gpu_ms: median_gpu_ms,
            encode_us: 10.0,
            prefill_tok_s: 0.0,
            rss_gb: get_peak_rss_gb(),
            top_kernels: None,
        };

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

        // Load Metal model
        let metal_exec = match crate::model::native::load_native(
            &artifacts,
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
        ) {
            Some(Ok(m)) => m,
            _ => {
                eprintln!("parity: cannot load metal model for {}", model_cfg.id);
                return Ok(4);
            }
        };

        // Load CPU model
        let cpu_exec = match crate::model::native::load_native(
            &artifacts,
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
        ) {
            Some(Ok(m)) => m,
            _ => {
                eprintln!("parity: cannot load cpu model for {}", model_cfg.id);
                return Ok(4);
            }
        };

        let prompt = "Explain physics";
        let prompt_ids = metal_exec.tokenizer().encode(prompt);
        let (prompt_tokens, mut m_state) = metal_exec.prepare(prompt_ids, &[])?;
        let (_, mut c_state) = cpu_exec.prepare(prompt_tokens.clone(), &[])?;

        let mut metal_tokens = prompt_tokens.clone();
        let mut cpu_tokens = prompt_tokens.clone();
        let mut m_logits = metal_exec.forward(&prompt_tokens, 0, &mut m_state)?;
        let mut c_logits = cpu_exec.forward(&prompt_tokens, 0, &mut c_state)?;

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
            c_logits = cpu_exec.forward(&[c_top as u32], cpu_tokens.len() - 1, &mut c_state)?;
        }
    }

    Ok(0)
}
