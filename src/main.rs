use clap::Parser;
use gibbs_engine::hardware::{Device, HardwareCapability, QuantMode};
use gibbs_engine::kv::PagedKVPool;
use gibbs_engine::model::classifier::{self, ClassifierSlot, SequenceClassifier};
use gibbs_engine::model::hub::{is_synthetic_model_id, HuggingFaceHub, StorageBackend};
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::jev::JevSystem1Model;
use gibbs_engine::model::load_log;
use gibbs_engine::model::ltx;
use gibbs_engine::model::native::load_native;
use gibbs_engine::model::t2v::{write_mp4, TextToVideoPipeline};
use gibbs_engine::model::text_to_image::{
    load_image_model, ImageGenerator, ImageRequest, ImageSlot,
};
use gibbs_engine::model::weight_format::{load_causal_lm_with, ReencodeConfig};
use gibbs_engine::scheduler::{cpu_backend_label, ContinuousBatchScheduler, ExecutableModel};
use gibbs_engine::server::{create_router, AppState};
use gibbs_engine::telemetry::EngineTelemetry;
use gibbs_engine::tui::TuiApp;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "gibbs",
    author = "Gibbs Systems <engineering@gibbs.ai>",
    version = "0.1.0",
    about = "Gibbs: local inference engine (CPU + Apple Metal)"
)]
struct Cli {
    #[arg(short, long, default_value = "HuggingFaceTB/SmolLM2-135M-Instruct")]
    model: String,

    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    #[arg(long, default_value_t = false)]
    tui: bool,

    #[arg(long, default_value_t = false)]
    system1: bool,

    /// Hugging Face BERT sentence encoder backing --system1 (needs config.json, tokenizer.json, model.safetensors).
    #[arg(long, default_value = "sentence-transformers/all-MiniLM-L6-v2")]
    system1_model: String,

    /// Weight format override (uncompressed, f16, bf16, q8_0, nvfp4, exl3-3/4/5, hybrid, turbo,
    /// jesse). Without it the engine measures the formats its kernels support on this host and
    /// picks the fastest one that stays within its quality floor.
    #[arg(long)]
    quant: Option<String>,

    /// Compute backend: auto (Metal on Apple Silicon, CPU only where no GPU exists), metal, ane
    /// (System 1), cuda, rocm, level-zero, or cpu. `cpu` is the explicit opt-in to the CPU executor;
    /// cuda, rocm and level-zero are stubs that fail at startup.
    #[arg(long, default_value = "auto")]
    backend: String,

    #[arg(long, default_value = "disk")]
    storage: String,

    #[arg(long)]
    hf_token: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    /// Speculative decoding drafter for --model: a DFlash / DFlash2 / DSpark repo trained for it,
    /// or a small causal LM with the same tokenizer.
    #[arg(long)]
    draft: Option<String>,

    #[arg(long)]
    image: Option<String>,

    /// Output file: a .png for text-to-image (default output.png), a required .mp4 for text-to-video.
    #[arg(long)]
    output: Option<String>,

    /// Denoising steps: default from the image model (e.g. 1 for turbo, 25 for SD), 25 for text-to-video.
    #[arg(long)]
    steps: Option<usize>,

    #[arg(long, default_value_t = 0.7)]
    strength: f32,

    /// Image/video seed; defaults to a clock-based seed that is printed so runs can be reproduced.
    #[arg(long)]
    seed: Option<u64>,

    /// Text-to-video: number of frames (ModelScope 16..=24, default 16; LTX-Video 8k+1, default 25).
    #[arg(long)]
    frames: Option<usize>,

    /// Classifier-free guidance scale (CFG runs when > 1); default from the model (9 for text-to-video).
    #[arg(long)]
    guidance: Option<f32>,

    /// Text-to-image / text-to-video: negative prompt for classifier-free guidance.
    #[arg(long)]
    negative_prompt: Option<String>,

    /// LTX-Video: T5-XXL text encoder GGUF (`org/repo` or `org/repo/file.gguf`); GGUF transformers default to
    /// city96/t5-v1_1-xxl-encoder-gguf Q4_K_M, diffusers repos to their own 19 GB `text_encoder/`.
    #[arg(long)]
    text_encoder: Option<String>,

    /// Text-to-image width / height in pixels (default from the model).
    #[arg(long)]
    width: Option<usize>,

    #[arg(long)]
    height: Option<usize>,

    #[arg(long)]
    max_tokens: Option<usize>,

    #[arg(long, default_value_t = 8192)]
    ctx: usize,

    /// GPU weight re-encode policy: max-speed | low-bit-native | bit-exact (default: settings.json, else max-speed).
    #[arg(long)]
    reencode: Option<String>,

    /// Mean-KL limit (nats) of the GPU re-encode gate; overrides the policy's own.
    #[arg(long)]
    reencode_max_kl: Option<f32>,

    /// Variant policy of the model picker and downloads: fastest | fastest-4bit | best-quality
    /// (default: settings.json, else fastest).
    #[arg(long)]
    variant: Option<String>,

    #[arg(long)]
    bench: Option<String>,

    #[arg(long)]
    bench_parity: Option<String>,

    /// Print the top-5 next-token logits (JSON) of --model for this raw prompt on Metal, then exit.
    #[arg(long)]
    probe_top5: Option<String>,

    #[arg(long)]
    tui_frame_log: Option<String>,
}

const VIDEO_SIZE: usize = 256;
const VIDEO_FPS: u32 = 8;

fn clock_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// `--quant` spellings.
fn parse_quant(s: &str) -> Result<QuantMode, String> {
    Ok(match s.to_lowercase().as_str() {
        "uncompressed" | "none" | "f32" | "fp32" => QuantMode::Uncompressed,
        "f16" | "fp16" => QuantMode::F16,
        "bf16" => QuantMode::BF16,
        "q8_0" | "q8" | "int8" => QuantMode::Q8_0,
        "nvfp4" | "nvfp4-ultra" | "fp4" => QuantMode::NVFP4Ultra,
        "exl3-3bit" | "exl3-3" => QuantMode::EXL3Aggressive,
        "exl3-4bit" | "exl3" | "exl3-4" => QuantMode::EXL3Balanced,
        "exl3-5bit" | "exl3-5" => QuantMode::EXL3Quality,
        "hybrid" | "adaptive" | "adaptivehybrid" => QuantMode::AdaptiveHybrid,
        "turbo" | "turbosqueeze" => QuantMode::TurboSqueeze,
        "jesse" | "weightless" | "lut" => QuantMode::WeightlessJesse,
        other => return Err(format!("Unknown --quant '{other}'")),
    })
}

/// One CLI text-to-video generation with an LTX-Video pipeline (diffusers repo or `ltxv` GGUF).
fn run_ltx(cli: &Cli, hub: &HuggingFaceHub, plan: &ltx::Plan) -> Result<(), String> {
    let model = &cli.model;
    let Some(prompt) = cli.prompt.as_deref() else {
        return Err(format!(
            "'{model}' is a text-to-video model; the HTTP server, daemon and TUI have no video endpoint. \
             Run one generation instead: gibbs --model {model} --prompt \"...\" --output clip.mp4"
        ));
    };
    let output = cli
        .output
        .as_deref()
        .filter(|o| o.ends_with(".mp4"))
        .ok_or("text-to-video needs --output <file.mp4>")?;
    let d = ltx::Clip::new(cli.seed.unwrap_or_else(clock_seed));
    let clip = ltx::Clip {
        frames: cli.frames.unwrap_or(d.frames),
        width: cli.width.unwrap_or(d.width),
        height: cli.height.unwrap_or(d.height),
        steps: cli.steps.unwrap_or(d.steps),
        guidance: cli.guidance.unwrap_or(d.guidance),
        ..d
    };
    println!("Prompt: {prompt:?}");
    println!(
        "Seed: {}  Steps: {}  Frames: {}  Guidance: {}  Size: {}x{}",
        clip.seed, clip.steps, clip.frames, clip.guidance, clip.width, clip.height
    );
    let start = Instant::now();
    let negative = cli.negative_prompt.as_deref().unwrap_or("");
    let image = cli
        .image
        .as_deref()
        .map(ImageBuffer::from_base64_or_path)
        .transpose()?;
    let log = |s: &str| println!("{s}");
    let frames = ltx::run(
        hub,
        plan,
        prompt,
        negative,
        cli.text_encoder.as_deref(),
        image.as_ref(),
        &clip,
        &log,
    )?;
    write_mp4(&frames, ltx::FPS, Path::new(output))?;
    let bytes = std::fs::metadata(output)
        .map_err(|e| format!("{output}: {e}"))?
        .len();
    println!(
        "Generated {} frames in {:.2}s",
        frames.len(),
        start.elapsed().as_secs_f64()
    );
    println!("Saved {output} ({bytes} bytes, {} fps)", ltx::FPS);
    Ok(())
}

/// One CLI text-to-video generation (ModelScope TextToVideoSDPipeline checkpoints).
fn run_text_to_video(cli: &Cli, hub: &HuggingFaceHub, backend: Device) -> Result<(), String> {
    let model = &cli.model;
    gibbs_engine::graph::avenues::family_gate(
        gibbs_engine::graph::avenues::Family::VideoModelScope,
        backend,
    )?;
    let Some(prompt) = cli.prompt.as_deref() else {
        return Err(format!(
            "'{model}' is a text-to-video model; the HTTP server, daemon and TUI have no video endpoint. \
             Run one generation instead: gibbs --model {model} --prompt \"...\" --output clip.mp4"
        ));
    };
    if cli.image.is_some() {
        return Err(format!(
            "--image is not supported with the text-to-video model '{model}'"
        ));
    }
    let output = cli
        .output
        .as_deref()
        .filter(|o| o.ends_with(".mp4"))
        .ok_or("text-to-video needs --output <file.mp4>")?;
    let steps = cli.steps.unwrap_or(25);
    let seed = cli.seed.unwrap_or_else(clock_seed);

    let t0 = Instant::now();
    let t2v = TextToVideoPipeline::load(hub, model)?;
    println!(
        "Text-to-video pipeline (CLIP + UNet3D + VAE) loaded in {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    println!("Prompt: {prompt:?}");
    let guidance = cli.guidance.unwrap_or(9.0);
    let frames = cli.frames.unwrap_or(16);
    println!(
        "Seed: {seed}  Steps: {steps}  Frames: {frames}  Guidance: {guidance}  Size: {VIDEO_SIZE}x{VIDEO_SIZE}"
    );
    let start = Instant::now();
    let frames = t2v.generate(
        prompt, frames, VIDEO_SIZE, VIDEO_SIZE, steps, guidance, seed,
    )?;
    let total = start.elapsed().as_secs_f64();
    write_mp4(&frames, VIDEO_FPS, Path::new(output))?;
    let bytes = std::fs::metadata(output)
        .map_err(|e| format!("{output}: {e}"))?
        .len();
    println!(
        "Generated {} frames in {total:.2}s ({:.2}s/step average, VAE decode included)",
        frames.len(),
        total / steps as f64
    );
    println!("Saved {output} ({bytes} bytes, {VIDEO_FPS} fps)");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(v) = &cli.variant {
        gibbs_engine::model::variant::set_policy_override(Some(v.parse()?));
    }
    if cli.reencode.is_some() || cli.reencode_max_kl.is_some() {
        gibbs_engine::model::weight_format::set_reencode_override(Some(ReencodeConfig::resolve(
            cli.reencode.as_deref(),
            cli.reencode_max_kl,
        )?));
    }
    if let Some(bench_path) = &cli.bench {
        let code = gibbs_engine::bench::run_bench_suite(bench_path)?;
        std::process::exit(code);
    }
    if let Some(parity_path) = &cli.bench_parity {
        let code = gibbs_engine::bench::run_bench_parity(parity_path)?;
        std::process::exit(code);
    }
    if let Some(prompt) = &cli.probe_top5 {
        gibbs_engine::bench::probe_top5(&cli.model, prompt)?;
        std::process::exit(0);
    }
    let telemetry = EngineTelemetry::new();
    #[cfg(target_os = "macos")]
    if let Ok(rt) = gibbs_engine::graph::metal::Runtime::shared() {
        let prof = rt.profile();
        telemetry.set_hardware_profile(&prof.device, prof.memory_bandwidth_gbps as f32);
        // S3.6: measured codec efficiencies for the variant predictor; background, never blocks the load.
        // Only in the long-lived TUI: a one-shot `--prompt` run would share the GPU with it and exit
        // before it finishes (the result is saved only on completion).
        if cli.prompt.is_none() {
            rt.spawn_codec_bench();
        }
    }

    // 1. Probe Hardware
    let hw = HardwareCapability::probe_system();
    let mut placement: gibbs_engine::hardware::PlacementMode = cli.backend.parse()?;
    // GPU-only by default: `auto` resolves to Metal wherever it exists; the CPU is an explicit `--backend cpu`.
    if placement == gibbs_engine::hardware::PlacementMode::Auto
        && gibbs_engine::hardware::default_device() == Device::Metal
    {
        placement = gibbs_engine::hardware::PlacementMode::Pin(Device::Metal);
    }
    let backend = match placement {
        gibbs_engine::hardware::PlacementMode::Pin(d) => d,
        gibbs_engine::hardware::PlacementMode::Auto => Device::Cpu,
    };
    let backend_label = match placement {
        gibbs_engine::hardware::PlacementMode::Auto => "auto (cost-modeled placement)".to_string(),
        gibbs_engine::hardware::PlacementMode::Pin(d) => match d {
            Device::Cpu => cpu_backend_label(),
            #[cfg(target_os = "macos")]
            Device::Metal => format!(
                "Metal ({})",
                gibbs_engine::backend::metal::system_device_name()?
            ),
            #[cfg(not(target_os = "macos"))]
            Device::Metal => return Err("--backend metal requires macOS".into()),
            #[cfg(target_os = "macos")]
            Device::Ane => "Apple Neural Engine (Core ML)".into(),
            #[cfg(not(target_os = "macos"))]
            Device::Ane => return Err("--backend ane requires macOS".into()),
            Device::Cuda | Device::Rocm | Device::LevelZero => {
                return Err(d
                    .stub_error()
                    .expect("cuda/rocm/level-zero are stubs")
                    .into())
            }
        },
    };
    // GPU-only tripwire: from here on, any CPU compute outside `--backend cpu` panics.
    if matches!(backend, Device::Metal | Device::Ane) {
        gibbs_engine::hardware::cpu_compute::ban(true);
    }
    if backend == Device::Ane && !cli.system1 {
        return Err("--backend ane runs System 1 only (add --system1)".into());
    }
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GIBBS INFERENCE ENGINE // v0.1.0");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Compute backend: {backend_label}");
    for line in hw.summary() {
        println!("{line}");
    }

    let quant_override = cli.quant.as_deref().map(parse_quant).transpose()?;

    // 2. Initialize Memory Hierarchy and KV Pool
    let kv_pool = Arc::new(PagedKVPool::new(16384));

    // 4. Model Resolution & Cache Inspection (Auto-Download & Cloud Storage)
    let storage_backend = StorageBackend::from_uri(&cli.storage)?;
    let hub = HuggingFaceHub::new()
        .with_storage(storage_backend)
        .with_budget(usize::MAX);

    // Text-to-video is chosen by the checkpoint's model_index.json / hub GGUF architecture, never by the model id.
    if !cli.system1 {
        if let Some(plan) = ltx::resolve(&hub, &cli.model, cli.quant.as_deref()) {
            return run_ltx(&cli, &hub, &plan?).map_err(Into::into);
        }
    }
    if !cli.system1 && TextToVideoPipeline::is_t2v_repo(&hub, &cli.model) {
        return run_text_to_video(&cli, &hub, backend).map_err(Into::into);
    }
    let token = cli.hf_token.or_else(|| std::env::var("HF_TOKEN").ok());

    // A `*ForSequenceClassification` config.json makes `--model` a classifier (/v1/classify).
    let classifier_arch = if cli.system1 {
        None
    } else {
        classifier::detect(&hub, &cli.model)
    };
    // `--system1` loads its own encoder (JevSystem1Model::load) and a classifier its own files;
    // `--model` is then not resolved as an LLM.
    let artifacts = if cli.system1 || classifier_arch.is_some() {
        None
    } else {
        telemetry.log_info(&format!(
            "Checking model source: '{}' (storage: {:?})",
            cli.model, hub.storage
        ));
        // A PEFT LoRA CAUSAL_LM adapter becomes a merged snapshot of its base first.
        gibbs_engine::model::peft::prepare_causal_lm(&hub, &cli.model, token.as_deref())?;
        let a = hub
            .auto_download_and_cache(&cli.model, token.as_deref())
            .map_err(|e| load_log::record(&hub, &cli.model, "downloading", &e))?;
        telemetry.log_info(&format!(
            "Model Artifacts Resolved: cached={}, local_dir={:?}, weights={:?}",
            a.is_cached, a.local_dir, a.weights_path
        ));
        Some(a)
    };

    // Image vs LLM is decided by the resolved files (model_index.json, SD single-file, LoRA, SD GGUF),
    // never by the model id.
    let image_model: Option<Arc<dyn ImageGenerator>> = match artifacts
        .as_ref()
        .map(|a| (a, Instant::now()))
        .and_then(|(a, t0)| Some((load_image_model(a, &hub)?, t0)))
    {
        Some((r, t0)) => {
            let g = r.map_err(|e| {
                format!(
                    "Cannot load image model '{}': {}",
                    cli.model,
                    load_log::record(&hub, &cli.model, "loading image model", &e)
                )
            })?;
            let d = g.defaults();
            telemetry.log_info(&format!(
                "Image model: {} loaded in {:.2}s (default {}x{}, {} steps, guidance {})",
                g.summary(),
                t0.elapsed().as_secs_f64(),
                d.width,
                d.height,
                d.steps,
                d.guidance
            ));
            Some(g)
        }
        None => None,
    };
    let is_image = image_model.is_some();
    // Check memory requirements of the LLM and warn if it exceeds physical memory.
    // Skipped for image models (real checkpoint size is logged at load) and System 1 (no LLM).
    if let (Some(a), Some(system_mem)) = (
        artifacts.as_ref().filter(|_| !is_image),
        hw.total_memory_bytes,
    ) {
        // Weights are read into f32 before any re-encoding.
        let estimated_weight_bytes = a.estimated_params as u64 * 4;
        if estimated_weight_bytes > system_mem {
            telemetry.log_warn(&format!(
                "Model '{}' (~{} params, {:.2} GB in f32) exceeds physical system memory ({:.2} GB). Offloading will be active.",
                cli.model,
                a.estimated_params,
                estimated_weight_bytes as f64 / 1e9,
                system_mem as f64 / 1e9
            ));
        } else {
            telemetry.log_info(&format!(
                "Model '{}' resolved (~{} params, {:.2} MB in f32, system memory {:.2} GB)",
                cli.model,
                a.estimated_params,
                estimated_weight_bytes as f64 / 1e6,
                system_mem as f64 / 1e9
            ));
        }
    }
    if cli.image.is_some() && !is_image {
        return Err(format!(
            "--image requires a text-to-image model; '{}' is not one",
            cli.model
        )
        .into());
    }
    if cli.image.is_some() && cli.prompt.is_none() {
        return Err("--image requires --prompt".into());
    }

    if let (Some(sd), Some(prompt)) = (&image_model, &cli.prompt) {
        let req = ImageRequest {
            prompt: prompt.clone(),
            negative_prompt: cli.negative_prompt.clone(),
            width: cli.width,
            height: cli.height,
            steps: cli.steps,
            guidance: cli.guidance,
            seed: cli.seed.unwrap_or_else(clock_seed),
        };
        let d = sd.defaults();
        let steps = req.steps.unwrap_or(d.steps);
        println!("Prompt: {prompt:?}");
        println!(
            "Seed: {}  Steps: {steps}  Guidance: {}  Negative: {:?}",
            req.seed,
            req.guidance.unwrap_or(d.guidance),
            req.negative_prompt.as_deref().unwrap_or("")
        );
        let start = Instant::now();
        let img = if let Some(init_path) = &cli.image {
            let init = gibbs_engine::model::image::ImageBuffer::from_base64_or_path(init_path)
                .map_err(|e| format!("Cannot read --image {init_path}: {e}"))?;
            println!(
                "Mode: image-to-image from {init_path} ({}x{}), strength {:.2}",
                init.width, init.height, cli.strength
            );
            sd.image_to_image(&req, &init, cli.strength)?
        } else {
            println!(
                "Mode: text-to-image {}x{}",
                req.width.unwrap_or(d.width),
                req.height.unwrap_or(d.height)
            );
            sd.text_to_image(&req)?
        };
        let gen_secs = start.elapsed().as_secs_f64();
        let png_bytes = img.to_png();
        let output = cli.output.as_deref().unwrap_or("output.png");
        std::fs::write(output, &png_bytes)?;
        println!(
            "Generated {}x{} in {gen_secs:.2}s ({:.2}s/step incl. text encoding and VAE)",
            img.width,
            img.height,
            gen_secs / steps as f64
        );
        println!("Saved {output} ({} bytes)", png_bytes.len());
        return Ok(());
    }

    let mut weights = None;
    let mut drafter = None;
    let classifier_slot = ClassifierSlot::default();
    let (model_exec, jev_model) = if is_image {
        (None, None)
    } else if cli.system1 {
        telemetry.log_info(&format!(
            "Loading System 1 sentence encoder '{}'",
            cli.system1_model
        ));
        let t0 = std::time::Instant::now();
        let jev_backend = if backend == Device::Cpu
            && gibbs_engine::hardware::available_devices().contains(&Device::Metal)
        {
            Device::Metal
        } else {
            backend
        };
        let jev = JevSystem1Model::load(&hub, &cli.system1_model, jev_backend)
            .map_err(|e| format!("Cannot load System 1 encoder '{}': {e}", cli.system1_model))?;
        if let Some(report) = jev.encoder.executor_report() {
            if report.starts_with("Core ML") {
                telemetry.log_info(&report);
            } else {
                telemetry.log_info(&format!("Metal: {report}"));
            }
        }
        telemetry.log_info(&format!(
            "System 1 encoder '{}' ready on {}: {} layers, hidden {}, {} heads, {:.1}M params, loaded in {:.2}s (zero-shot cosine decisions)",
            jev.repo_id,
            jev.encoder.device(),
            jev.encoder.dims.num_layers,
            jev.encoder.dims.hidden_size,
            jev.encoder.dims.num_heads,
            jev.encoder.parameter_count() as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        ));
        (None, Some(Arc::new(jev)))
    } else if classifier_arch.is_some() {
        let t0 = Instant::now();
        let c = SequenceClassifier::load(&hub, &cli.model, backend)
            .map_err(|e| format!("Cannot load classifier '{}': {e}", cli.model))?;
        telemetry.log_info(&format!(
            "Classifier '{}' ready: {}, loaded in {:.2}s",
            cli.model,
            c.summary(),
            t0.elapsed().as_secs_f64()
        ));
        if let Some(prompt) = &cli.prompt {
            let (text, pair) = classifier::split_pair(prompt);
            let (labels, tokens) = c.classify(text, pair)?;
            println!("{tokens} tokens");
            for l in labels {
                println!("{:>10.6}  {}", l.score, l.label);
            }
            return Ok(());
        }
        *classifier_slot.write() = Some(Arc::new(c));
        (None, None)
    } else {
        let artifacts = artifacts
            .as_ref()
            .expect("artifacts are resolved unless --system1 or a classifier");
        let model_size = artifacts.weights_path.as_ref().map_or(0, |p| {
            gibbs_engine::model::variant::decode_bytes_per_token(
                p,
                gibbs_engine::model::variant::reencode_policy(),
            )
        });
        telemetry.set_model_size_bytes(model_size);
        let model_exec = match load_native(artifacts, placement) {
            Some(native) => {
                let m = native.map_err(|e| {
                    format!(
                        "Cannot load '{}': {}",
                        cli.model,
                        load_log::record(&hub, &cli.model, "loading native model", &e)
                    )
                })?;
                telemetry.log_info(&format!("Model: {}", m.summary()));
                ExecutableModel::Native(m)
            }
            None => {
                let (graph_lm, choice) =
                    load_causal_lm_with(artifacts, backend, quant_override, &hw, &{
                        let mut cfg =
                            ReencodeConfig::resolve(cli.reencode.as_deref(), cli.reencode_max_kl)?;
                        let t = telemetry.clone();
                        cfg.log = Some(Arc::new(move |m: &str| t.log_info(m)));
                        cfg
                    })
                    .map_err(|e| {
                        format!(
                            "Cannot load '{}': {}",
                            cli.model,
                            load_log::record(&hub, &cli.model, "loading weights", &e)
                        )
                    })?;
                if is_synthetic_model_id(&cli.model) {
                    telemetry.log_warn(&format!(
                        "'{}' is a synthetic random-weight model for kernel benchmarking: its text output is noise",
                        cli.model
                    ));
                }
                let model_exec = ExecutableModel::load_graph_lm(graph_lm, placement)
                    .map_err(|e| format!("Cannot run '{}' on {backend_label}: {e}", cli.model))?;
                telemetry.log_info(&format!("Weight format: {}", choice.summary()));
                weights = Some(choice);
                model_exec
            }
        };
        telemetry.log_info(&format!("Compute backend: {}", model_exec.backend_label()));
        if let Some(d) = &cli.draft {
            let da = hub.auto_download_and_cache(d, token.as_deref())?;
            let dr = gibbs_engine::model::draft::load_drafter(&da, artifacts, placement)
                .map_err(|e| format!("Cannot load drafter '{d}': {e}"))?;
            telemetry.log_info(&format!("Speculative decoding drafter: {}", dr.label()));
            drafter = Some(dr);
        }

        if let Some(prompt) = &cli.prompt {
            println!("💬 Generating from prompt: \"{}\"", prompt);
            let cap = match &model_exec {
                ExecutableModel::Native(m) => m.default_max_new_tokens(),
                _ => None,
            };
            let max_toks = cli.max_tokens.or(cap).unwrap_or(64);
            // Models that declare greedy decoding (T5, seq2seq: transformers' default) are not sampled.
            let temperature = match model_exec.tokenizer().sampling_defaults {
                Some((t, _)) if t <= 1e-4 => 0.0,
                _ => 0.7,
            };
            let (output, passes, decode) = model_exec.generate_text_measured(
                prompt,
                max_toks,
                temperature,
                drafter.clone(),
            )?;
            println!("✨ Output:\n{}", output);
            let tok_s = passes as f64 / decode.as_secs_f64().max(1e-9);
            println!(
                "⚡ Decode: {passes} tokens in {:.2}ms ({:.1} tok/s) on {}",
                decode.as_secs_f64() * 1000.0,
                tok_s,
                model_exec.backend_label()
            );

            // Roofline reporting
            let bytes_per_token = telemetry.bytes_per_token();
            let ceiling_gbps = telemetry.get_summary().memory_bandwidth_gbps as f64;
            if bytes_per_token > 0 && ceiling_gbps > 0.0 {
                let roofline_tok_s = ceiling_gbps / (bytes_per_token as f64 / 1e9);
                let pct = (tok_s / roofline_tok_s) * 100.0;
                println!(
                    "📈 Roofline: {pct:.1}% of {roofline_tok_s:.1} tok/s ({bytes_per_token} B/token at {ceiling_gbps:.1} GB/s)"
                );
            }
            return Ok(());
        }

        (Some(model_exec), None)
    };

    // 5. Initialize Continuous Batch Scheduler with Real Streaming Channel & Telemetry
    let scheduler = Arc::new(ContinuousBatchScheduler::with_telemetry(
        cli.model.clone(),
        kv_pool.clone(),
        model_exec,
        telemetry.clone(),
    ));
    if cli.draft.is_some() && drafter.is_none() {
        return Err("--draft needs a text-generation --model".into());
    }
    if drafter.is_some() {
        if std::env::var("GIBBS_SPEC").as_deref() == Ok("off") {
            return Err("--draft: speculative decoding is disabled (GIBBS_SPEC=off)".into());
        }
        scheduler.set_drafter(drafter);
    }

    // 6. Continuous batching loop on its own thread: a step is blocking compute, and on a tokio
    // worker it kept the streams it woke from running until it yielded (bursty SSE, stalls).
    // A panicking step fails the in-flight requests (their clients get an error) and the loop
    // keeps serving, instead of the thread dying and every later request hanging.
    let sched_clone = scheduler.clone();
    std::thread::spawn(move || loop {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sched_clone.step())) {
            Ok(out) if out.active_sequences_count == 0 => {
                std::thread::sleep(Duration::from_millis(5))
            }
            Ok(_) => {}
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                sched_clone.abort_all(&format!("engine step panicked: {msg}"));
            }
        }
    });

    // 7. HTTP/SSE server. Bound here so a busy port fails before the TUI takes the terminal.
    let image_slot = ImageSlot::default();
    *image_slot.write() = image_model;
    let state = AppState {
        scheduler: scheduler.clone(),
        telemetry: telemetry.clone(),
        jev_model: jev_model.clone(),
        image_model: image_slot.clone(),
        classifier: classifier_slot.clone(),
    };
    let app = create_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("Cannot listen on {addr}: {e}"))?;
    println!("🚀 Gibbs HTTP/SSE Server listening on http://{}", addr);
    println!("   Endpoints: /v1/chat/completions, /v1/completions, /v1/responses (OpenAI), /v1/messages (Anthropic), /api/chat, /api/generate (Ollama), /v1/images/generations, /v1/images/edits, /v1/embeddings, /v1/decisions, /v1/classify, /v1/rerank, /v1/models, /health");
    println!(
        "   Agents: {}  ·  OLLAMA_HOST={}  ·  TUI: /connect",
        gibbs_engine::server::connect::agent_command("claude", addr, &scheduler.model_id())?,
        gibbs_engine::server::connect::client_addr(addr)
    );
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("HTTP server stopped: {e}");
        }
    });

    // 8. Interactive TUI or Headless Daemon Mode
    if cli.tui {
        let mut tui_app = TuiApp::new(telemetry, kv_pool)
            .with_runtime(scheduler, hub, token, jev_model, image_slot);
        tui_app.quant_override = quant_override;
        tui_app.weights = weights;
        tui_app.frame_log_path = cli.tui_frame_log;
        tui_app.classifier = classifier_slot;
        tui_app.http_addr = Some(addr);
        tui_app.placement = placement;
        tokio::task::spawn_blocking(move || tui_app.run()).await??;
    } else {
        println!("Gibbs Engine running in daemon mode. Press Ctrl+C to stop.");
        tokio::signal::ctrl_c().await?;
        println!("Shutting down Gibbs Engine cleanly.");
    }

    Ok(())
}
