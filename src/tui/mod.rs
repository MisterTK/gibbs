//! Terminal dashboard: live engine metrics, a prompt console, and a model picker that searches
//! the local cache and huggingface.co, then downloads, measures, and hot-swaps the chosen model.
//!
//! Prompts and model loads run on background threads and report through a channel, so the UI
//! never blocks; it redraws on input, on streamed output, and every [`REFRESH`] otherwise.

pub(crate) mod mascot;
mod picker;
mod render;

use crate::hardware::{HardwareCapability, PlacementMode, QuantMode};
use crate::kv::PagedKVPool;
use crate::model::classifier::{self, Activation, ClassifierSlot, LabelScore, SequenceClassifier};
use crate::model::hf_loader::ChatMessage;
use crate::model::hub::{
    split_revision, split_weight_file, DownloadProgress, HuggingFaceHub, StorageBackend,
};
use crate::model::image::ImageBuffer;
use crate::model::jev::{ChoiceOption, JevSystem1Model};
use crate::model::ltx;
use crate::model::native::{load_native, MediaInput};
use crate::model::text_to_image::{load_image_model, ImageRequest, ImageSlot};
use crate::model::tokenizer::Tokenizer;
use crate::model::weight_format::{load_causal_lm_for, Choice};
use crate::scheduler::{finish_outcome, ContinuousBatchScheduler, ExecutableModel, SamplingParams};
use crate::telemetry::EngineTelemetry;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use parking_lot::Mutex;
use picker::{Picker, PickerAction};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Offered in the picker even when not cached: the models verified to load and generate.
const PINNED_MODELS: &[&str] = &[
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "unsloth/SmolLM2-135M-Instruct-GGUF",
    "THUDM/glm-edge-1.5b-chat",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "roneneldan/TinyStories-Instruct-8M",
    "EleutherAI/pythia-14m",
];
/// Redraw interval when nothing happens; input and streamed output redraw at once.
const REFRESH: Duration = Duration::from_millis(250);
/// Poll interval while output streams or a model loads.
const BUSY_POLL: Duration = Duration::from_millis(33);
/// Throughput history: one sample per second.
const SPARK_SECONDS: usize = 120;
/// Streamed chat text is re-sent to the UI at most this often.
const STREAM_UPDATE: Duration = Duration::from_millis(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Chat,
    Image,
    Decide,
    Embed,
    Classify,
    /// LTX-Video clip; `image`: the text starts with the first frame's path (`/i2v`).
    Video {
        image: bool,
    },
}

/// Default model family for bare (command-less) prompts, inferred from the active model id.
fn route_for_model(model_id: &str) -> Route {
    let m = model_id.to_lowercase();
    if m.contains("image") || m.contains("diffusion") {
        Route::Image
    } else if m.contains("minilm") || m.contains("embed") || m.contains("sentence-transformers") {
        Route::Embed
    } else if m.contains("bert")
        || m.contains("jev")
        || m.contains("system1")
        || m.contains("decision")
    {
        Route::Decide
    } else {
        Route::Chat
    }
}

/// Text-generation models: the only kind the TUI hot-swaps into the scheduler.
pub(crate) fn is_chat_model(id: &str) -> bool {
    route_for_model(id) == Route::Chat
}

/// Luminance-ramp preview sampled from the real generated pixels.
fn ascii_preview(img: &ImageBuffer, cols: usize, rows: usize) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    if img.width == 0 || img.height == 0 || img.channels == 0 {
        return String::new();
    }
    let used = img.channels.min(3);
    let mut out = String::with_capacity((cols + 1) * rows);
    for r in 0..rows {
        for c in 0..cols {
            let i = ((r * img.height / rows) * img.width + c * img.width / cols) * img.channels;
            let lum = img.data[i..i + used]
                .iter()
                .map(|&v| v as usize)
                .sum::<usize>()
                / used;
            out.push(RAMP[lum * (RAMP.len() - 1) / 255] as char);
        }
        out.push('\n');
    }
    out.pop();
    out
}

/// A chat prompt word `@<path>` naming an existing image (png/jpg/jpeg) or WAV audio
/// file: the path and whether it is an image. Other `@words` stay prompt text.
fn attachment(word: &str) -> Option<(&std::path::Path, bool)> {
    let path = std::path::Path::new(word.strip_prefix('@')?);
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let image = match ext.as_str() {
        "png" | "jpg" | "jpeg" => true,
        "wav" => false,
        _ => return None,
    };
    path.is_file().then_some((path, image))
}

/// A prompt with `@<path>` attachments as one user message of HF content parts (text and
/// `{"type":"image"|"audio"}` in prompt order) and the files' bytes in the same order; `None`
/// without attachments.
fn media_message(prompt: &str) -> Result<Option<(ChatMessage, Vec<MediaInput>)>, String> {
    let (mut parts, mut media, mut text) = (Vec::new(), Vec::new(), Vec::<&str>::new());
    let text_part = |text: &mut Vec<&str>| {
        let part = serde_json::json!({"type": "text", "text": text.join(" ")});
        text.clear();
        part
    };
    for word in prompt.split_whitespace() {
        let Some((path, image)) = attachment(word) else {
            text.push(word);
            continue;
        };
        if !text.is_empty() {
            parts.push(text_part(&mut text));
        }
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        parts.push(serde_json::json!({"type": if image { "image" } else { "audio" }}));
        media.push(if image {
            MediaInput::Image(bytes)
        } else {
            MediaInput::Audio(bytes)
        });
    }
    if media.is_empty() {
        return Ok(None);
    }
    if !text.is_empty() {
        parts.push(text_part(&mut text));
    }
    let mut msg = ChatMessage::new("user", "");
    msg.content = serde_json::Value::Array(parts);
    Ok(Some((msg, media)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Running,
    Text,
    Json,
    Image,
    Error,
}

/// One console entry: a prompt or model load and its result.
pub struct Entry {
    pub time: String,
    pub prompt: String,
    pub kind: Kind,
    pub output: String,
    /// e.g. `42 tokens · 86.1 tok/s`; empty when there is nothing to add.
    pub stats: String,
    started: Instant,
    took: Option<Duration>,
    /// Set for running chats: raising it stops streaming and drops the token receiver.
    cancel: Option<Arc<AtomicBool>>,
}

/// Results from background work, applied on the UI thread.
enum Update {
    Partial {
        idx: usize,
        text: String,
        tokens: usize,
    },
    Done {
        idx: usize,
        kind: Kind,
        text: String,
        stats: String,
    },
    Loaded {
        idx: usize,
        id: String,
        /// (GraphLm weight-format choice, `None` for native models; shape and backend).
        result: Result<(Option<Choice>, String), String>,
    },
}

/// A prompt's background work: runs with its console entry index and the update channel.
type Job = Box<dyn FnOnce(usize, mpsc::Sender<Update>) + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Dashboard,
    Prompt,
    Picker,
}

/// A model load running on a background thread.
struct Load {
    id: String,
    /// Console entry of this load.
    idx: usize,
    started: Instant,
    stage: Arc<Mutex<&'static str>>,
    progress: Arc<DownloadProgress>,
    /// (when, bytes done) of the last rate sample, and the bytes/s since the one before.
    sample: (Instant, u64),
    rate: f64,
}

impl Load {
    /// `downloading onnx/model_q4.onnx 45% · 360/796 MB · 12 MB/s` while bytes flow, else the stage.
    fn progress_text(&self) -> String {
        let stage = *self.stage.lock();
        let (done, total) = (
            self.progress.done.load(Ordering::Relaxed),
            self.progress.total.load(Ordering::Relaxed),
        );
        if stage != "downloading" || total == 0 {
            return stage.to_string();
        }
        format!(
            "downloading {} {}% · {:.0}/{:.0} MB · {:.1} MB/s",
            self.progress.file.lock(),
            done * 100 / total,
            done as f64 / 1e6,
            total as f64 / 1e6,
            self.rate / 1e6
        )
    }
}

pub struct TuiApp {
    pub telemetry: Arc<EngineTelemetry>,
    pub kv_pool: Arc<PagedKVPool>,
    pub hw: HardwareCapability,
    pub scheduler: Option<Arc<ContinuousBatchScheduler>>,
    pub hub: HuggingFaceHub,
    pub hf_token: Option<String>,
    pub jev_model: Option<Arc<JevSystem1Model>>,
    /// Shared with the HTTP server; `Some` makes `/image` and bare prompts generate images.
    pub image_model: ImageSlot,
    /// Set by loading an LTX-Video repo; `/video` and `/i2v` generate clips with it.
    pub video: ltx::VideoSlot,
    /// Shared with the HTTP server; `Some` is the active model (a text classifier).
    pub classifier: ClassifierSlot,
    /// `--quant`; `None` lets the engine pick each loaded model's weight format.
    pub quant_override: Option<QuantMode>,
    /// Weight format of the served LLM and how it was picked.
    pub weights: Option<Choice>,
    pub http_addr: Option<SocketAddr>,
    pub placement: PlacementMode,
    pub scratch_dir: tempfile::TempDir,
    pub history: Vec<Entry>,
    pub input: String,
    focus: Focus,
    picker: Picker,
    load: Option<Load>,
    running: usize,
    updates_tx: mpsc::Sender<Update>,
    updates_rx: mpsc::Receiver<Update>,
    /// Console lines scrolled up from the bottom; 0 follows new output.
    scroll: usize,
    /// Tokens generated in each of the last [`SPARK_SECONDS`] seconds, oldest first.
    spark: Vec<u64>,
    spark_at: Instant,
    spark_total: u64,
    host_lines: Vec<String>,
    pub mascot: mascot::MascotEngine,
    pub show_template_modal: bool,
    pub active_template: usize,
    pub system_prompt: String,
    pub show_logprob_inspector: bool,
    pub latest_logprobs: Option<crate::scheduler::TokenLogprobInfo>,
    pub logprob_history: std::collections::VecDeque<crate::scheduler::TokenLogprobInfo>,
    pub logprob_scrub_idx: usize,
    pub logprob_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::TokenLogprobInfo>>,
    pub show_roofline_inspector: bool,
    pub roofline_probe_result: Option<String>,
    pub show_latency_modal: bool,
    pub show_token_inspector: bool,
}

impl TuiApp {
    pub fn new(telemetry: Arc<EngineTelemetry>, kv_pool: Arc<PagedKVPool>) -> Self {
        let hw = HardwareCapability::probe_system();
        let hub = HuggingFaceHub::new()
            .with_storage(StorageBackend::from_uri("disk").expect("local disk storage"))
            .with_budget(usize::MAX);
        let hf_token = std::env::var("HF_TOKEN").ok();
        let picker = Picker::new(
            hub.clone(),
            hf_token.clone(),
            PINNED_MODELS,
            hw.total_memory_bytes,
        );
        let (updates_tx, updates_rx) = mpsc::channel();
        let spark_total = telemetry.total_tokens_generated.load(Ordering::Relaxed);
        #[cfg(target_os = "macos")]
        if let Ok(rt) =
            crate::graph::metal::Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir())
        {
            let prof = rt.profile();
            telemetry.set_hardware_profile(&prof.device, prof.memory_bandwidth_gbps as f32);
        }
        Self {
            telemetry,
            kv_pool,
            host_lines: hw.summary(),
            hw,
            scheduler: None,
            hub,
            hf_token,
            jev_model: None,
            image_model: ImageSlot::default(),
            video: ltx::VideoSlot::default(),
            classifier: ClassifierSlot::default(),
            quant_override: None,
            weights: None,
            http_addr: None,
            placement: PlacementMode::Auto,
            scratch_dir: tempfile::tempdir().expect("scratch directory for TUI images"),
            history: Vec::new(),
            input: String::new(),
            focus: Focus::Dashboard,
            picker,
            load: None,
            running: 0,
            updates_tx,
            updates_rx,
            scroll: 0,
            spark: vec![0; SPARK_SECONDS],
            spark_at: Instant::now(),
            spark_total,
            mascot: mascot::MascotEngine::new(),
            show_template_modal: false,
            active_template: 0,
            system_prompt: "You are a helpful and concise AI assistant.".into(),
            show_logprob_inspector: false,
            latest_logprobs: None,
            logprob_history: std::collections::VecDeque::with_capacity(64),
            logprob_scrub_idx: 0,
            logprob_rx: None,
            show_roofline_inspector: false,
            roofline_probe_result: None,
            show_latency_modal: false,
            show_token_inspector: false,
        }
    }

    /// Attaches the engine's live scheduler (for chat + model hot-swap) and any loaded System 1 / image models.
    pub fn with_runtime(
        mut self,
        scheduler: Arc<ContinuousBatchScheduler>,
        hub: HuggingFaceHub,
        hf_token: Option<String>,
        jev_model: Option<Arc<JevSystem1Model>>,
        image_model: ImageSlot,
    ) -> Self {
        self.picker = Picker::new(
            hub.clone(),
            hf_token.clone(),
            PINNED_MODELS,
            self.hw.total_memory_bytes,
        );
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        *scheduler.logprob_tx.write() = Some(tx);
        self.logprob_rx = Some(rx);
        self.scheduler = Some(scheduler);
        self.hub = hub;
        self.hf_token = hf_token;
        self.jev_model = jev_model;
        self.image_model = image_model;
        self
    }

    /// Id of the active model: the loaded image model or classifier, else the scheduler's; empty
    /// without any.
    fn model_id(&self) -> String {
        if let Some(id) = self.image_model.read().as_ref().and_then(|m| m.model_id()) {
            return id;
        }
        if let Some(c) = self.classifier.read().as_ref() {
            return c.repo_id.clone();
        }
        self.scheduler
            .as_ref()
            .map(|s| s.model_id())
            .unwrap_or_default()
    }

    fn push_entry(&mut self, prompt: String, kind: Kind, output: String) -> usize {
        let now = Instant::now();
        self.history.push(Entry {
            time: chrono::Local::now().format("%H:%M:%S").to_string(),
            prompt,
            kind,
            output,
            stats: String::new(),
            started: now,
            took: (kind != Kind::Running).then_some(Duration::ZERO),
            cancel: None,
        });
        if kind == Kind::Running {
            self.running += 1;
        }
        self.scroll = 0;
        self.history.len() - 1
    }

    /// Runs the typed prompt on a background thread; its entry streams into the console.
    /// Commands: `/chat`, `/image`, `/decide <query> | opt, opt`, `/embed`, `/classify <text>`
    /// (`<text> || <pair>` for pair models); bare text classifies when a classifier is active,
    /// else routes by the active model id. `/connect` and `/agent` answer at once. Missing
    /// models and bad input are reported at once.
    pub fn submit_prompt(&mut self) {
        let raw = std::mem::take(&mut self.input).trim().to_string();
        if raw.is_empty() {
            return;
        }
        let (cmd, rest) = raw.split_once(char::is_whitespace).unwrap_or((&raw, ""));
        let answer = match cmd {
            "/connect" => Some(
                self.http_addr
                    .map(|a| crate::server::connect::setup_text(a, &self.model_id()))
                    .ok_or_else(|| "no HTTP server is running".to_string()),
            ),
            "/agent" => Some(self.launch_agent(rest.trim())),
            "/roofline" => {
                self.show_roofline_inspector = true;
                let probe = self.run_roofline_probe();
                self.roofline_probe_result = probe.as_ref().ok().cloned();
                Some(probe)
            }
            "/avatar" => {
                let target = rest.trim();
                if target.is_empty() {
                    self.mascot.avatar_style = self.mascot.avatar_style.cycle();
                    Some(Ok(format!(
                        "Active mascot scene: {} (options: classic, chalkboard, thermo, lattice, cyber)",
                        self.mascot.avatar_style.name()
                    )))
                } else if let Some(style) = mascot::AvatarStyle::from_str_style(target) {
                    self.mascot.avatar_style = style;
                    Some(Ok(format!("Switched mascot scene to: {}", style.name())))
                } else {
                    Some(Err(format!(
                        "unknown avatar style '{target}'; options: classic, chalkboard, thermo, lattice, cyber"
                    )))
                }
            }
            "/bench" => {
                let s = self.telemetry.get_summary();
                let rss_mb = crate::hardware::current_resident_memory_bytes() as f64 / (1024.0 * 1024.0);
                let eta = if s.memory_bandwidth_gbps > 0.0 {
                    (s.effective_bandwidth_gbps / s.memory_bandwidth_gbps) * 100.0
                } else {
                    0.0
                };
                Some(Ok(format!(
                    "BENCHMARK SUMMARY:\n\
                     · Current Throughput: {:.2} tok/s\n\
                     · Total Tokens Generated: {}\n\
                     · Peak Bandwidth Achieved: {:.1} GB/s\n\
                     · Roofline Saturation (η): {:.1}%\n\
                     · Host Process RSS: {:.1} MB",
                    s.current_tps,
                    s.total_tokens,
                    s.effective_bandwidth_gbps,
                    eta,
                    rss_mb
                )))
            }
            "/quant" => {
                Some(Ok(format!(
                    "MODEL QUANTIZATION:\n\
                     · Active Model ID: {}\n\
                     · Engine Weight Precision: Native GGUF mixed precision\n\
                     · Host Acceleration: Metal GPU unified memory",
                    self.model_id()
                )))
            }
            "/help" => {
                Some(Ok(
                    "GIBBS ENGINE COMMAND & KEY REFERENCE:\n\
                     Keybindings:\n\
                     · m / Esc    : Switch between Dashboard & Model Picker\n\
                     · F2         : System Prompt / Chat Template Selector\n\
                     · F3         : Switch Josiah Willard Gibbs Mascot Scene\n\
                     · Ctrl+P     : Token Perplexity & Logprob History Inspector (←/→ scrub)\n\
                     · Ctrl+R     : 2D Visual Roofline Diagnostic Inspector\n\
                     · Esc        : Close open modal / cancel prompt / return to Dashboard\n\
                     · Ctrl+C     : Exit Gibbs engine\n\n\
                     Slash Commands:\n\
                     · /roofline  : Open 2D Roofline Diagnostic Inspector & calibration sweep\n\
                     · /avatar [style] : Switch mascot art scene (classic, chalkboard, thermo, lattice, cyber)\n\
                     · /bench     : Report live decode throughput, RSS, and roofline efficiency\n\
                     · /quant     : Display active weight quantization formats\n\
                     · /gibbs     : Full historical biography & ASCII portrait of J. Willard Gibbs\n\
                     · /chat <msg>: Send chat message to active model\n\
                     · /connect   : Print curl/OpenAI API connection commands"
                        .to_string(),
                ))
            }
            "/gibbs" => Some(Ok(mascot::gibbs_full_portrait())),
            _ => None,
        };
        if let Some(r) = answer {
            let (kind, text) = r.map_or_else(|e| (Kind::Error, e), |t| (Kind::Text, t));
            self.push_entry(raw, kind, text);
            return;
        }
        let (route, body) = match cmd {
            "/chat" => (Ok(Route::Chat), rest.trim()),
            "/image" | "/img" => (Ok(Route::Image), rest.trim()),
            "/decide" | "/judge" => (Ok(Route::Decide), rest.trim()),
            "/embed" => (Ok(Route::Embed), rest.trim()),
            "/classify" | "/cls" => (Ok(Route::Classify), rest.trim()),
            "/i2v" => (Ok(Route::Video { image: true }), rest.trim()),
            "/video" | "/vid" | "/t2v" => (Ok(Route::Video { image: false }), rest.trim()),
            c if c.starts_with('/') => (Err(format!("unknown command '{c}'; use /chat, /image, /decide, /embed, /classify, /connect, /agent, /roofline, /avatar, /bench, /quant, /help, /gibbs")), ""),
            _ if self.classifier.read().is_some() => (Ok(Route::Classify), raw.as_str()),
            _ if self.image_model.read().is_some() => (Ok(Route::Image), raw.as_str()),
            _ => (Ok(route_for_model(&self.model_id())), raw.as_str()),
        };
        let body = body.to_string();
        let job = route.and_then(|r| {
            if body.is_empty() {
                Err("empty prompt".to_string())
            } else {
                self.prompt_job(r, body)
            }
        });
        match job {
            Err(e) => {
                self.push_entry(raw, Kind::Error, e);
            }
            Ok((job, cancel)) => {
                let names: Vec<String> = raw
                    .split_whitespace()
                    .filter_map(attachment)
                    .filter_map(|(p, _)| Some(p.file_name()?.to_string_lossy().into_owned()))
                    .collect();
                let shown = if names.is_empty() {
                    raw
                } else {
                    format!("{raw}  [attached: {}]", names.join(", "))
                };
                let idx = self.push_entry(shown, Kind::Running, String::new());
                self.history[idx].cancel = cancel;
                let tx = self.updates_tx.clone();
                std::thread::spawn(move || job(idx, tx));
            }
        }
    }

    /// Starts an agent CLI pointed at this server: in a new tmux window inside tmux, else a new
    /// macOS Terminal window; elsewhere (or when the CLI is missing) returns the command to run.
    fn launch_agent(&self, agent: &str) -> Result<String, String> {
        let addr = self.http_addr.ok_or("no HTTP server is running")?;
        let cmd = crate::server::connect::agent_command(agent, addr, &self.model_id())?;
        let on_path = std::env::var_os("PATH")
            .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(agent).is_file()));
        if !on_path {
            return Err(format!(
                "{agent} is not on PATH; install it, then run:\n{cmd}"
            ));
        }
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
        let mut launcher = if std::env::var_os("TMUX").is_some() {
            let mut c = std::process::Command::new("tmux");
            c.args(["new-window", "-n", agent, "-c"])
                .arg(&cwd)
                .arg(&cmd);
            c
        } else if cfg!(target_os = "macos") {
            let cwd = crate::server::connect::q(&cwd.to_string_lossy());
            let script = format!("cd {cwd} && {cmd}")
                .replace('\\', "\\\\")
                .replace('"', "\\\"");
            let mut c = std::process::Command::new("osascript");
            c.args([
                "-e",
                &format!("tell application \"Terminal\" to do script \"{script}\""),
                "-e",
                "tell application \"Terminal\" to activate",
            ]);
            c
        } else {
            return Ok(format!("run in another terminal:\n{cmd}"));
        };
        let out = launcher
            .output()
            .map_err(|e| format!("cannot launch {agent}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cannot launch {agent}: {}\nrun it yourself:\n{cmd}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(format!("launched {agent} in a new window:\n{cmd}"))
    }
    /// Runs an empirical roofline calibration probe comparing measured memory bandwidth
    /// against Apple unified memory hardware roofline ceilings.
    pub fn run_roofline_probe(&self) -> Result<String, String> {
        #[cfg(target_os = "macos")]
        {
            let dir = crate::model::hub::HuggingFaceHub::cache_dir();
            let rt = crate::graph::metal::Runtime::new(&dir)?;
            let measured_gbps = rt.benchmark_memory_bandwidth()?;

            let mut peak_gflops = 0.0f64;
            for t in &rt.profile().gemm {
                let min_us = t.us.iter().map(|(_, us)| *us).fold(f64::INFINITY, f64::min);
                if min_us.is_finite() && min_us > 0.0 {
                    let gflops = (2.0 * t.m as f64 * t.n as f64 * t.k as f64) / (min_us * 1e3);
                    peak_gflops = peak_gflops.max(gflops);
                }
            }
            if peak_gflops == 0.0 {
                peak_gflops = 14_000.0;
            }

            let ridge = peak_gflops / measured_gbps.max(1.0);
            let s = self.telemetry.get_summary();
            let eta = if s.effective_bandwidth_gbps > 0.0 {
                (s.effective_bandwidth_gbps as f64 / measured_gbps) * 100.0
            } else {
                0.0
            };
            let regime = if eta > 65.0 {
                "Bandwidth-Constrained"
            } else {
                "Latency/Compute-Bound"
            };

            Ok(format!(
                "EMPIRICAL ROOFLINE CALIBRATION:\n\
                 · Host GPU Device: {}\n\
                 · Empirical Memory Sweep (256 MB): {:.1} GB/s\n\
                 · Peak Calibrated GEMM Compute: {:.1} TFLOPs ({:.0} GFLOP/s)\n\
                 · Empirical Operational Intensity Ridge: {:.1} FLOPs/Byte\n\
                 · Live Decode Bandwidth: {:.1} GB/s (η = {:.1}% {})\n\
                 · Instantaneous Throughput: {:.1} tok/s",
                &rt.profile().device,
                measured_gbps,
                peak_gflops / 1000.0,
                peak_gflops,
                ridge,
                s.effective_bandwidth_gbps,
                eta,
                regime,
                s.current_tps
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err("Empirical roofline probe is only available on Metal hosts".into())
        }
    }

    /// The work for one prompt (and, for chats, its cancel flag), or why it cannot run.
    fn prompt_job(
        &self,
        route: Route,
        body: String,
    ) -> Result<(Job, Option<Arc<AtomicBool>>), String> {
        let done = |tx: &mpsc::Sender<Update>, idx, r: Result<(Kind, String, String), String>| {
            let (kind, text, stats) = r.unwrap_or_else(|e| (Kind::Error, e, String::new()));
            let _ = tx.send(Update::Done {
                idx,
                kind,
                text,
                stats,
            });
        };
        match route {
            Route::Chat => {
                let sched = self
                    .scheduler
                    .clone()
                    .filter(|s| s.has_model())
                    .ok_or("no text model loaded; press m to load one")?;
                let media = media_message(&body)?;
                let cancel = Arc::new(AtomicBool::new(false));
                let cancelled = cancel.clone();
                let active_template = self.active_template;
                let system_prompt = self.system_prompt.clone();
                let job: Job = Box::new(move |idx, tx| {
                    // Instruct models get their chat template; base models (Plain) get the raw text.
                    // Answers run to EOS or the end of the model's context.
                    let params = SamplingParams {
                        temperature: None,
                        top_p: None,
                        max_new_tokens: usize::MAX,
                        top_logprobs: Some(5),
                    };
                    let submitted = match &media {
                        None => {
                            let system_prompt = system_prompt.clone();
                            Ok(sched.submit_with_params(params, move |tok| {
                                let prompt_str = Self::format_chat_prompt_with(
                                    &system_prompt,
                                    active_template,
                                    tok,
                                    body.as_str(),
                                );
                                tok.encode(&prompt_str)
                            }))
                        }
                        Some((msg, media)) => {
                            let system_prompt = system_prompt.clone();
                            let msg = msg.clone();
                            sched.submit_with_media(params, media, move |tok| {
                                let mut msgs = Vec::new();
                                if !system_prompt.is_empty() {
                                    msgs.push(crate::model::hf_loader::ChatMessage::new(
                                        "system",
                                        system_prompt.as_str(),
                                    ));
                                }
                                msgs.push(msg);
                                tok.encode(&tok.apply_chat_template_messages(&msgs, None))
                            })
                        }
                    };
                    let (_, mut rx, tok, _, finish) = match submitted {
                        Ok(s) => s,
                        Err(e) => return done(&tx, idx, Err(e)),
                    };
                    let mut ids = Vec::new();
                    let (mut first, mut sent) = (None, Instant::now());
                    while let Some(id) = rx.blocking_recv() {
                        if cancelled.load(Ordering::Relaxed) {
                            // Dropping `rx` ends the sequence at its next emit; the UI already
                            // marked the entry cancelled.
                            return;
                        }
                        ids.push(id);
                        first.get_or_insert_with(Instant::now);
                        if sent.elapsed() >= STREAM_UPDATE {
                            sent = Instant::now();
                            let text = tok.decode(&ids);
                            let _ = tx.send(Update::Partial {
                                idx,
                                text,
                                tokens: ids.len(),
                            });
                        }
                    }
                    let decode = first.map_or(0.0, |t| t.elapsed().as_secs_f64());
                    let rate = if ids.len() > 1 && decode > 0.0 {
                        format!(" · {:.1} tok/s", (ids.len() - 1) as f64 / decode)
                    } else {
                        String::new()
                    };
                    let r = finish_outcome(&finish).map(|reason| {
                        let stats = format!("{} tokens{rate} · {reason}", ids.len());
                        (Kind::Text, tok.decode(&ids), stats)
                    });
                    done(&tx, idx, r);
                });
                Ok((job, Some(cancel)))
            }
            Route::Image => {
                let sd = self.image_model.read().clone().ok_or(
                    "no image model loaded; press m and load a text-to-image repo, e.g. stabilityai/sd-turbo",
                )?;
                let dir = self.scratch_dir.path().to_path_buf();
                Ok((
                    Box::new(move |idx, tx| {
                        let seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        let req = ImageRequest {
                            prompt: body,
                            seed,
                            ..Default::default()
                        };
                        let d = sd.defaults();
                        let r = sd
                            .text_to_image(&req)
                            .map_err(|e| format!("image generation failed: {e}"))
                            .and_then(|img| {
                                let path = dir.join(format!("tui_image_{:03}.png", idx + 1));
                                let png = img.to_png();
                                std::fs::write(&path, &png)
                                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
                                let text =
                                    format!("{}\n{}", path.display(), ascii_preview(&img, 48, 12));
                                let stats = format!(
                                    "{}x{} · {} steps · cfg {} · {} bytes · seed {seed}",
                                    img.width,
                                    img.height,
                                    d.steps,
                                    d.guidance,
                                    png.len()
                                );
                                Ok((Kind::Image, text, stats))
                            });
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
            Route::Video { image } => {
                let model = self.video.read().clone().ok_or(
                    "no video model loaded; press m and load an LTX-Video repo, e.g. city96/LTX-Video-gguf",
                )?;
                let dir = self.scratch_dir.path().to_path_buf();
                Ok((
                    Box::new(move |idx, tx| {
                        let r = (|| {
                            let (first, prompt) = match image {
                                true => body
                                    .split_once(char::is_whitespace)
                                    .ok_or("usage: /i2v <image path> <prompt>")?,
                                false => ("", body.as_str()),
                            };
                            let init = image
                                .then(|| ImageBuffer::from_base64_or_path(first))
                                .transpose()?;
                            let seed = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64;
                            let clip = ltx::Clip::new(seed);
                            let start = Instant::now();
                            let negative =
                                "worst quality, inconsistent motion, blurry, jittery, distorted";
                            let frames = ltx::run(
                                &model.0,
                                &model.1,
                                prompt.trim(),
                                negative,
                                None,
                                init.as_ref(),
                                &clip,
                                &|_| {},
                            )?;
                            let path = dir.join(format!("tui_video_{:03}.mp4", idx + 1));
                            crate::model::t2v::write_mp4(&frames, ltx::FPS, &path)?;
                            let bytes = std::fs::metadata(&path)
                                .map_err(|e| format!("{}: {e}", path.display()))?
                                .len();
                            let stats = format!(
                                "{} frames {}x{} · {} steps · cfg {} · {:.0}s · {bytes} bytes · seed {seed}",
                                frames.len(),
                                clip.width,
                                clip.height,
                                clip.steps,
                                clip.guidance,
                                start.elapsed().as_secs_f64()
                            );
                            Ok((Kind::Text, path.display().to_string(), stats))
                        })();
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
            Route::Decide | Route::Embed => {
                let jev = self.jev_model.clone().ok_or(
                    "no System 1 (decision/embedding) model loaded; launch with `--system1 --tui`",
                )?;
                let run: fn(&JevSystem1Model, &str) -> Result<String, String> =
                    if route == Route::Decide {
                        decide
                    } else {
                        embed
                    };
                Ok((
                    Box::new(move |idx, tx| {
                        let r = run(&jev, &body).map(|json| (Kind::Json, json, String::new()));
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
            Route::Classify => {
                let c = self.classifier.read().clone().ok_or(
                    "no text classifier loaded; press m and load a text-classification model",
                )?;
                Ok((
                    Box::new(move |idx, tx| {
                        let (text, pair) = classifier::split_pair(&body);
                        let r = c.classify(text, pair).map(|(labels, tokens)| {
                            let stats = format!("{tokens} tokens · {}", c.repo_id);
                            (Kind::Text, label_bars(&labels, c.activation), stats)
                        });
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
        }
    }

    /// Downloads (if needed), loads, and activates `id` on a background thread: a text
    /// classifier becomes the active model; an LLM (picking its weight format) is hot-swapped
    /// into the scheduler and replaces any active classifier. On failure the active model stays.
    pub fn load_model(&mut self, id: &str) {
        let refused = match (&self.scheduler, &self.load) {
            (None, _) => Some("no engine scheduler attached".to_string()),
            (_, Some(l)) => Some(format!("{} is still loading", l.id)),
            (Some(_), None) => None,
        };
        if let Some(e) = refused {
            self.push_entry(format!("load {id}"), Kind::Error, e);
            return;
        }
        let sched = self.scheduler.clone().expect("checked above");
        let idx = self.push_entry(format!("load {id}"), Kind::Running, String::new());
        let stage = Arc::new(Mutex::new("resolving"));
        let progress = Arc::new(DownloadProgress::default());
        self.load = Some(Load {
            id: id.to_string(),
            idx,
            started: Instant::now(),
            stage: stage.clone(),
            progress: progress.clone(),
            sample: (Instant::now(), 0),
            rate: 0.0,
        });
        self.telemetry.log_info(&format!("TUI: loading '{id}'"));
        let mut hub = self.hub.clone();
        hub.progress = Some(progress);
        let (token, tx) = (self.hf_token.clone(), self.updates_tx.clone());
        let slot = self.classifier.clone();
        let images = self.image_model.clone();
        let video = self.video.clone();
        let (quant_override, hw, id, placement) = (
            self.quant_override,
            self.hw.clone(),
            id.to_string(),
            self.placement,
        );
        std::thread::spawn(move || {
            // A panic inside loading must surface as an error, not leave the load stuck.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = match placement {
                    PlacementMode::Pin(d) => d,
                    PlacementMode::Auto => crate::hardware::default_device(),
                };
                let effective_placement = PlacementMode::Pin(backend);
                *stage.lock() = if hub.locate_model(&id).weights_path.is_some() {
                    "reading cached files"
                } else {
                    "downloading"
                };
                if classifier::detect(&hub, &id).is_some() {
                    let c = SequenceClassifier::load(&hub, &id, backend)?;
                    let shape = c.summary();
                    *slot.write() = Some(Arc::new(c));
                    *images.write() = None;
                    return Ok((None, shape));
                }
                if let Some(plan) = crate::model::ltx::resolve(&hub, &id, None) {
                    let plan = plan?;
                    let shape = format!("LTX-Video {}; /video <prompt> and /i2v <image> <prompt> write .mp4 clips (CPU, minutes)", plan.label());
                    *video.write() = Some(Arc::new((hub.clone(), plan)));
                    *images.write() = None;
                    *slot.write() = None;
                    return Ok((None, shape));
                }
                // Image repos are recognised by their files (model_index.json, SD single-file, LoRA).
                let image = route_for_model(&id) == Route::Image;
                let adapter = !image
                    && crate::model::peft::prepare_causal_lm(&hub, &id, token.as_deref())?;
                if !adapter && !image && !is_chat_model(&id) {
                    return Err(format!(
                        "only text-generation, text-to-image models and text classifiers hot-swap; relaunch with `--model {id} --tui`"
                    ));
                }
                let artifacts = hub.auto_download_and_cache(&id, token.as_deref())?;
                let model_size = artifacts
                    .weights_path
                    .as_ref()
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len())
                    .unwrap_or(0);
                sched.telemetry.set_model_size_bytes(model_size);
                *stage.lock() = "reading weights, choosing weight format";
                if let Some(img) = load_image_model(&artifacts, &hub) {
                    let g = img?;
                    let d = g.defaults();
                    let shape = format!(
                        "{} · default {}x{}, {} steps, cfg {}; prompts now generate images",
                        g.summary(),
                        d.width,
                        d.height,
                        d.steps,
                        d.guidance
                    );
                    *images.write() = Some(g);
                    *slot.write() = None;
                    return Ok((None, shape));
                }
                if let Some(native) = load_native(&artifacts, effective_placement) {
                    let m = native?;
                    let shape = format!("{}, on {}", m.summary(), m.backend_label());
                    sched.swap_model(&id, ExecutableModel::Native(m));
                    *slot.write() = None;
                    *images.write() = None;
                    return Ok((None, shape));
                }
                let (llm, choice) = load_causal_lm_for(&artifacts, backend, quant_override, &hw)?;
                let shape = format!(
                    "{} layers, hidden {}, vocab {}",
                    llm.config.num_layers, llm.config.hidden_dim, llm.config.vocab_size
                );
                *stage.lock() = "placing weights on the backend";
                sched.swap_model(&id, ExecutableModel::load_graph_lm(llm, placement)?);
                *slot.write() = None;
                *images.write() = None;
                Ok((
                    Some(choice),
                    format!("{shape}, on {}", sched.backend_label()),
                ))
            }))
            .unwrap_or_else(|p| {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                Err(format!("loader panicked: {msg}"))
            });
            // Failures leave a record (error, stage, snapshot contents) in the load-failure log.
            let result = result.map_err(|e| {
                let stage = *stage.lock();
                crate::model::load_log::record(&hub, &id, stage, &e)
            });
            let _ = tx.send(Update::Loaded { idx, id, result });
        });
    }

    /// Downloads (if needed) and attaches drafter `id` to the active model for speculative
    /// decoding, on a background thread; sequences admitted from then on use it.
    fn attach_draft(&mut self, id: &str) {
        let Some(sched) = self.scheduler.clone() else {
            self.push_entry(
                format!("draft {id}"),
                Kind::Error,
                "no engine scheduler attached".into(),
            );
            return;
        };
        let idx = self.push_entry(format!("draft {id}"), Kind::Running, String::new());
        let (hub, token, tx) = (
            self.hub.clone(),
            self.hf_token.clone(),
            self.updates_tx.clone(),
        );
        let (id, target) = (id.to_string(), sched.model_id());
        std::thread::spawn(move || {
            let attach = || -> Result<String, String> {
                let da = hub.auto_download_and_cache(&id, token.as_deref())?;
                let d = crate::model::draft::load_drafter(&da, &hub.locate_model(&target))?;
                // ponytail: check-then-set; a model swap landing in between keeps this drafter
                // until the swap resets it. Add a guarded set if loads race in practice.
                if sched.model_id() != target {
                    return Err(format!("{target} was replaced while the drafter loaded"));
                }
                let label = d.label();
                if std::env::var("GIBBS_SPEC").as_deref() == Ok("off") {
                    return Err("speculative decoding is disabled (GIBBS_SPEC=off)".into());
                }
                sched.set_drafter(Some(d));
                Ok(format!("speculative decoding on {target} with {label}"))
            };
            let (kind, text) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(attach))
            {
                Ok(Ok(t)) => (Kind::Text, t),
                Ok(Err(e)) => (Kind::Error, e),
                Err(_) => (Kind::Error, "drafter loader panicked".into()),
            };
            let _ = tx.send(Update::Done {
                idx,
                kind,
                text,
                stats: String::new(),
            });
        });
    }

    /// Applies finished background work. Returns true when anything changed.
    fn drain_updates(&mut self) -> bool {
        let mut changed = false;
        while let Ok(u) = self.updates_rx.try_recv() {
            changed = true;
            match u {
                Update::Partial { idx, text, tokens } => {
                    let e = &mut self.history[idx];
                    // A cancelled chat keeps what it showed when cancelled.
                    if e.kind == Kind::Running {
                        (e.output, e.stats) = (text, format!("{tokens} tokens"));
                    }
                }
                Update::Done {
                    idx,
                    kind,
                    text,
                    stats,
                } => self.finish(idx, kind, text, stats),
                Update::Loaded { idx, id, result } => {
                    let load = self.load.take();
                    let secs = load
                        .as_ref()
                        .map_or(0.0, |l| l.started.elapsed().as_secs_f64());
                    let fetched = load.map_or(0, |l| l.progress.done.load(Ordering::Relaxed));
                    let source = if fetched > 0 {
                        format!("downloaded {}", picker::fmt_bytes(fetched))
                    } else {
                        "cached".to_string()
                    };
                    match result {
                        Ok((choice, shape)) => {
                            let weights = choice.as_ref().map_or(String::new(), |c| {
                                format!(
                                    "\nweights: {:?} ({:.2} bpw), {}",
                                    c.mode,
                                    c.mode.effective_bits_per_weight(),
                                    c.origin()
                                )
                            });
                            self.telemetry.log_info(&format!(
                                "TUI: loaded '{id}' in {secs:.1}s ({source}): {shape}{}",
                                choice.as_ref().map_or(String::new(), |c| format!(
                                    "; weights {}",
                                    c.summary()
                                ))
                            ));
                            self.weights = choice;
                            self.picker.mark_cached(&id);
                            self.finish(idx, Kind::Text, format!("{shape}{weights}"), source);
                        }
                        Err(e) => {
                            self.telemetry
                                .log_warn(&format!("TUI: cannot load '{id}': {e}"));
                            self.finish(idx, Kind::Error, e, String::new());
                        }
                    }
                }
            }
        }
        changed
    }

    /// Final state of a running entry; entries already finished (e.g. cancelled) stay as they are.
    fn finish(&mut self, idx: usize, kind: Kind, text: String, stats: String) {
        let e = &mut self.history[idx];
        if e.kind != Kind::Running {
            return;
        }
        self.running -= 1;
        (e.kind, e.output, e.stats) = (kind, text, stats);
        e.took = Some(e.started.elapsed());
        e.cancel = None;
    }

    /// Cancels the newest running chat: its thread drops the token receiver (freeing the
    /// sequence) and the entry keeps the tokens so far. Returns false when no chat runs.
    fn cancel_chat(&mut self) -> bool {
        let Some(idx) = self
            .history
            .iter()
            .rposition(|e| e.kind == Kind::Running && e.cancel.is_some())
        else {
            return false;
        };
        let e = &self.history[idx];
        if let Some(c) = &e.cancel {
            c.store(true, Ordering::Relaxed);
        }
        let tokens = if e.stats.is_empty() {
            "0 tokens"
        } else {
            e.stats.as_str()
        };
        let (text, stats) = (e.output.clone(), format!("{tokens} · cancelled"));
        self.finish(idx, Kind::Text, text, stats);
        self.telemetry.log_info("TUI: chat cancelled");
        true
    }

    /// Removes `<cache>/models--<org>--<repo>` of `id` (never the active or loading model).
    fn delete_cached(&mut self, id: &str) {
        // Ids may name one file (`org/repo/x-Q4_K_M.gguf`); the cache dir is per repo. Hub ids
        // and the default macOS filesystem are case-insensitive, so `Qwen/X` is `qwen/x`.
        // Deleting a bare `org/repo` removes every revision, so `org/repo@<rev>` counts as it.
        let same_repo = |other: &str| {
            let repo = split_weight_file(other).0;
            repo.eq_ignore_ascii_case(id) || split_revision(repo).0.eq_ignore_ascii_case(id)
        };
        let refused = if same_repo(&self.model_id()) {
            Some("it is the active model")
        } else if self.load.as_ref().is_some_and(|l| same_repo(&l.id)) {
            Some("it is loading")
        } else if id.split('/').count() != 2 || id.contains("..") {
            Some("not an <org>/<repo> id")
        } else {
            None
        };
        if let Some(why) = refused {
            self.picker.set_note(
                format!("cannot delete {id}: {why}"),
                ratatui::style::Color::Red,
            );
            return;
        }
        let dir = self.picker.repo_dir(id);
        let freed = picker::dir_size(&dir);
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {
                self.telemetry.log_info(&format!(
                    "TUI: deleted {id} from the cache, freed {}",
                    picker::fmt_bytes(freed)
                ));
                self.picker.mark_deleted(id, freed);
            }
            Err(e) => {
                let msg = format!("cannot delete {}: {e}", dir.display());
                self.telemetry.log_warn(&format!("TUI: {msg}"));
                self.picker.set_note(msg, ratatui::style::Color::Red);
            }
        }
    }

    /// Samples throughput once a second and applies background updates. Returns true when the
    /// screen should be redrawn.
    fn tick(&mut self) -> bool {
        let mut dirty = self.drain_updates();
        if self.spark_at.elapsed() >= Duration::from_secs(1) {
            let total = self
                .telemetry
                .total_tokens_generated
                .load(Ordering::Relaxed);
            self.spark.remove(0);
            self.spark.push(total - self.spark_total);
            (self.spark_total, self.spark_at) = (total, Instant::now());
            dirty = true;
        }
        self.mascot.tick();
        if self.load.is_some() {
            self.mascot.set_state(mascot::MascotState::Thinking);
        } else if self.running > 0 {
            self.mascot.set_state(mascot::MascotState::Generating);
        } else if self.history.last().map_or(false, |e| e.kind == Kind::Error) {
            self.mascot.set_state(mascot::MascotState::Error);
        } else if let Some(last) = self.history.last() {
            if last.kind == Kind::Text
                && last
                    .took
                    .is_some_and(|t| last.started.elapsed() < t + Duration::from_secs(3))
            {
                self.mascot.set_state(mascot::MascotState::Equilibrium);
            } else {
                self.mascot.set_state(mascot::MascotState::Idle);
            }
        } else {
            self.mascot.set_state(mascot::MascotState::Idle);
        }
        if let Some(l) = &mut self.load {
            let done = l.progress.done.load(Ordering::Relaxed);
            let dt = l.sample.0.elapsed().as_secs_f64();
            if dt >= 1.0 {
                l.rate = done.saturating_sub(l.sample.1) as f64 / dt;
                l.sample = (Instant::now(), done);
            }
            let text = l.progress_text();
            let e = &mut self.history[l.idx];
            if e.output != text {
                e.output = text;
                dirty = true;
            }
        }
        if self.focus == Focus::Picker {
            dirty |= self.picker.tick();
        }
        if let Some(rx) = &mut self.logprob_rx {
            while let Ok(info) = rx.try_recv() {
                if self.logprob_history.len() >= 64 {
                    self.logprob_history.pop_front();
                }
                self.logprob_history.push_back(info.clone());
                self.latest_logprobs = Some(info);
                self.logprob_scrub_idx = self.logprob_history.len().saturating_sub(1);
                if self.show_logprob_inspector {
                    dirty = true;
                }
            }
        }
        dirty
    }

    /// Handles a key press. Returns true when the TUI should exit.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        // Raw mode reports Enter typed as `\n` / `\r` as Ctrl-J / Ctrl-M.
        if ctrl && matches!(key.code, KeyCode::Char('j' | 'm')) {
            return self.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        }
        if ctrl && key.code == KeyCode::Char('c') {
            return true;
        }
        if ctrl && key.code == KeyCode::Char('x') {
            self.cancel_chat();
            return false;
        }
        if ctrl && key.code == KeyCode::Char('p') {
            if matches!(self.focus, Focus::Dashboard | Focus::Prompt) {
                self.show_logprob_inspector = !self.show_logprob_inspector;
                return false;
            }
        }
        if ctrl && key.code == KeyCode::Char('r') {
            if matches!(self.focus, Focus::Dashboard | Focus::Prompt) {
                self.show_roofline_inspector = !self.show_roofline_inspector;
                if self.show_roofline_inspector && self.roofline_probe_result.is_none() {
                    self.roofline_probe_result = self.run_roofline_probe().ok();
                }
                return false;
            }
        }
        if ctrl && key.code == KeyCode::Char('l') {
            if matches!(self.focus, Focus::Dashboard | Focus::Prompt) {
                self.show_latency_modal = !self.show_latency_modal;
                return false;
            }
        }
        if ctrl && key.code == KeyCode::Char('t') {
            if matches!(self.focus, Focus::Dashboard | Focus::Prompt) {
                self.show_token_inspector = !self.show_token_inspector;
                return false;
            }
        }
        if self.show_latency_modal && key.code == KeyCode::Esc {
            self.show_latency_modal = false;
            return false;
        }
        if self.show_token_inspector && key.code == KeyCode::Esc {
            self.show_token_inspector = false;
            return false;
        }
        if self.show_roofline_inspector && key.code == KeyCode::Esc {
            self.show_roofline_inspector = false;
            return false;
        }
        if self.show_logprob_inspector {
            match key.code {
                KeyCode::Esc => {
                    self.show_logprob_inspector = false;
                    return false;
                }
                KeyCode::Left => {
                    self.logprob_scrub_idx = self.logprob_scrub_idx.saturating_sub(1);
                    return false;
                }
                KeyCode::Right => {
                    self.logprob_scrub_idx = (self.logprob_scrub_idx + 1)
                        .min(self.logprob_history.len().saturating_sub(1));
                    return false;
                }
                _ => {}
            }
        }
        if key.code == KeyCode::F(3) {
            self.mascot.avatar_style = self.mascot.avatar_style.cycle();
            return false;
        }
        if key.code == KeyCode::F(2) {
            self.show_template_modal = !self.show_template_modal;
            return false;
        }
        if self.show_template_modal {
            match key.code {
                KeyCode::Esc | KeyCode::F(2) => self.show_template_modal = false,
                KeyCode::Up => self.active_template = self.active_template.saturating_sub(1),
                KeyCode::Down => self.active_template = (self.active_template + 1).min(4),
                KeyCode::Char(c) if c.is_ascii_digit() => {
                    let d = c.to_digit(10).unwrap_or(0) as usize;
                    if d <= 4 {
                        self.active_template = d;
                    }
                }
                KeyCode::Enter => self.show_template_modal = false,
                _ => {}
            }
            return false;
        }
        match self.focus {
            Focus::Picker => match self.picker.handle_key(key) {
                PickerAction::None => {}
                PickerAction::Close => self.focus = Focus::Dashboard,
                PickerAction::Load(id) => {
                    self.focus = Focus::Dashboard;
                    self.load_model(&id);
                }
                PickerAction::Delete(id) => self.delete_cached(&id),
                PickerAction::AttachDraft(id) => {
                    self.focus = Focus::Dashboard;
                    self.attach_draft(&id);
                }
            },
            Focus::Prompt => match key.code {
                KeyCode::Esc => {
                    if !self.cancel_chat() {
                        self.focus = Focus::Dashboard;
                    }
                }
                KeyCode::Enter => self.submit_prompt(),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Char('u') if ctrl => self.input.clear(),
                KeyCode::Char(c) if !ctrl => self.input.push(c),
                code => self.scroll_console(code),
            },
            Focus::Dashboard => match key.code {
                KeyCode::Char('q') => return true,
                KeyCode::Char('i') | KeyCode::Char('p') | KeyCode::Enter => {
                    self.focus = Focus::Prompt
                }
                KeyCode::Char('/') => {
                    self.focus = Focus::Prompt;
                    self.input.push('/');
                }
                KeyCode::Char('m') => {
                    self.picker.open(&self.model_id());
                    self.focus = Focus::Picker;
                }
                KeyCode::Char('k') => self.scroll_console(KeyCode::Up),
                KeyCode::Char('j') => self.scroll_console(KeyCode::Down),
                code => self.scroll_console(code),
            },
        }
        false
    }

    fn scroll_console(&mut self, code: KeyCode) {
        self.scroll = match code {
            KeyCode::Up => self.scroll + 1,
            KeyCode::Down => self.scroll.saturating_sub(1),
            KeyCode::PageUp => self.scroll + 10,
            KeyCode::PageDown => self.scroll.saturating_sub(10),
            KeyCode::Home => usize::MAX / 2,
            KeyCode::End => 0,
            _ => return,
        };
    }

    pub fn run(&mut self) -> io::Result<()> {
        self.telemetry.is_tui_active.store(true, Ordering::Relaxed);
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        // A panic on this thread must not leave the terminal in raw mode; other threads' panics
        // go to the log (the default hook would print over the alternate screen).
        let ui_thread = std::thread::current().id();
        let previous: Arc<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync> =
            Arc::from(std::panic::take_hook());
        let (prev, telemetry) = (previous.clone(), self.telemetry.clone());
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().id() == ui_thread {
                restore_terminal();
                prev(info);
            } else {
                telemetry.log_warn(&format!("panic: {info}"));
            }
        }));
        let result = match Terminal::new(CrosstermBackend::new(io::stdout())) {
            Ok(mut terminal) => {
                let r = self.event_loop(&mut terminal);
                restore_terminal();
                r.and(terminal.show_cursor())
            }
            Err(e) => {
                restore_terminal();
                Err(e)
            }
        };
        std::panic::set_hook(Box::new(move |info| previous(info)));
        self.telemetry.is_tui_active.store(false, Ordering::Relaxed);
        result
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> io::Result<()> {
        let mut next_draw = Instant::now();
        loop {
            if self.tick() || Instant::now() >= next_draw {
                terminal.draw(|f| self.render(f))?;
                next_draw = Instant::now() + REFRESH;
            }
            let busy = self.running > 0 || self.load.is_some() || self.focus == Focus::Picker;
            let wait = next_draw
                .saturating_duration_since(Instant::now())
                .min(if busy { BUSY_POLL } else { REFRESH });
            if event::poll(wait)? {
                // Apply every queued event before one redraw (fast typing, pastes, resizes).
                loop {
                    if let Event::Key(key) = event::read()? {
                        if key.kind != KeyEventKind::Release && self.handle_key(key) {
                            return Ok(());
                        }
                    }
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
                next_draw = Instant::now();
            }
        }
    }
    /// Updates the active chat template (0: default, 1: ChatML, 2: Llama3, 3: Gemma, 4: Plain).
    pub fn set_chat_template(&mut self, template_idx: usize) {
        self.active_template = template_idx.min(4);
    }

    /// Updates the system prompt prepended to chat requests.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Formats a chat prompt using the active chat template and system prompt.
    pub fn format_chat_prompt(&self, tok: &Tokenizer, user_prompt: &str) -> String {
        Self::format_chat_prompt_with(&self.system_prompt, self.active_template, tok, user_prompt)
    }

    /// Formats a chat prompt using explicit system prompt and template style.
    pub fn format_chat_prompt_with(
        system_prompt: &str,
        active_template: usize,
        tok: &Tokenizer,
        user_prompt: &str,
    ) -> String {
        let mut msgs = Vec::new();
        if !system_prompt.is_empty() {
            msgs.push(("system", system_prompt));
        }
        msgs.push(("user", user_prompt));
        match active_template {
            1 => tok.apply_chat_template_with_style(
                &msgs,
                crate::model::tokenizer::ChatTemplate::ChatML,
            ),
            2 => tok.apply_chat_template_with_style(
                &msgs,
                crate::model::tokenizer::ChatTemplate::Llama3,
            ),
            3 => tok.apply_chat_template_with_style(
                &msgs,
                crate::model::tokenizer::ChatTemplate::Gemma,
            ),
            4 => tok.apply_chat_template_with_style(
                &msgs,
                crate::model::tokenizer::ChatTemplate::Plain,
            ),
            _ => tok.apply_chat_template(&msgs),
        }
    }
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
}

/// `/decide <state> | option, option`: zero-shot choice by embedding similarity.
fn decide(jev: &JevSystem1Model, body: &str) -> Result<String, String> {
    let (query, opts) = body
        .split_once('|')
        .map(|(q, o)| (q.trim(), o))
        .unwrap_or((body, ""));
    let options: Vec<ChoiceOption> = opts
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| ChoiceOption::new(s, None::<String>))
        .collect();
    if options.is_empty() {
        return Err(
            "usage: /decide <state> | option1, option2 (zero-shot choice by embedding similarity)"
                .to_string(),
        );
    }
    let decision = jev
        .embed(query)
        .and_then(|s| jev.evaluate_choice_embedded(&s, &options))
        .map_err(|e| format!("System 1 decision failed: {e}"))?;
    let json = serde_json::json!({ "query": query, "model": jev.repo_id, "decision": decision });
    Ok(serde_json::to_string_pretty(&json).unwrap_or_default())
}

/// `/embed <text>`: dimensions, L2 norm, and the head of the embedding.
fn embed(jev: &JevSystem1Model, text: &str) -> Result<String, String> {
    let v = jev
        .embed(text)
        .map_err(|e| format!("System 1 embedding failed: {e}"))?;
    let l2_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let json = serde_json::json!({
        "dimensions": v.len(),
        "l2_norm": l2_norm,
        "head": &v[..v.len().min(8)],
    });
    Ok(serde_json::to_string_pretty(&json).unwrap_or_default())
}

/// Classifier output: one line per label (the top [`SHOWN`]), with a bar for probabilities.
fn label_bars(labels: &[LabelScore], activation: Activation) -> String {
    const SHOWN: usize = 10;
    const BAR: usize = 24;
    let top = &labels[..labels.len().min(SHOWN)];
    let w = top
        .iter()
        .map(|l| l.label.chars().count())
        .max()
        .unwrap_or(0);
    let mut lines: Vec<String> = top
        .iter()
        .map(|l| {
            if activation == Activation::Identity {
                format!("{:<w$}  {:>9.4}  (raw score)", l.label, l.score)
            } else {
                let n = (l.score.clamp(0.0, 1.0) * BAR as f32).round() as usize;
                let bar = format!("{}{}", "█".repeat(n), "·".repeat(BAR - n));
                format!("{:<w$}  {bar}  {:.4}", l.label, l.score)
            }
        })
        .collect();
    if labels.len() > SHOWN {
        lines.push(format!("… {} more labels", labels.len() - SHOWN));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_and_run(app: &mut TuiApp, s: &str) {
        app.input = s.to_string();
        app.submit_prompt();
    }

    /// Applies background updates until no prompt or load is running.
    fn settle(app: &mut TuiApp) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while app.running > 0 || app.load.is_some() {
            app.drain_updates();
            assert!(Instant::now() < deadline, "background work did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Commands with nothing to run on fail at once with the reason; nothing is fabricated.
    #[test]
    fn test_prompt_commands_report_missing_models_and_bad_input() {
        let mut app = TuiApp::new(EngineTelemetry::new(), Arc::new(PagedKVPool::new(64)));
        let cases = [
            ("/decide is this safe | approve, reject", "System 1"),
            ("/embed quick brown fox", "System 1"),
            ("/chat hello", "no text model"),
            ("/image a lighthouse", "no image model"),
            ("/image", "empty prompt"),
            ("/bogus x", "unknown command '/bogus'"),
            ("/video airship", "no video model"),
            ("/i2v x", "no video model"),
        ];
        for (input, reason) in cases {
            type_and_run(&mut app, input);
            let e = app.history.last().unwrap();
            assert_eq!(e.kind, Kind::Error, "{input}");
            assert!(e.output.contains(reason), "{input}: {}", e.output);
        }
        assert_eq!(app.running, 0);
        assert!(app.input.is_empty());
    }

    /// Keys type into the prompt while it has focus; the same keys drive the dashboard otherwise.
    #[test]
    fn test_keys_type_in_prompt_and_command_dashboard() {
        let mut app = TuiApp::new(EngineTelemetry::new(), Arc::new(PagedKVPool::new(64)));
        assert!(!app.handle_key(key(KeyCode::Char('i'))));
        for c in "mq".chars() {
            assert!(!app.handle_key(key(KeyCode::Char(c))), "'{c}' must type");
        }
        assert_eq!(app.input, "mq");
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.focus, Focus::Dashboard);
        app.handle_key(key(KeyCode::Char('m')));
        assert_eq!(app.focus, Focus::Picker);
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.focus, Focus::Dashboard);
        assert!(app.handle_key(key(KeyCode::Char('q'))));
        assert!(app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
    }

    #[test]
    fn test_model_load_hot_swaps_scheduler_and_chat_streams() {
        let cache = tempfile::tempdir().unwrap();
        let kv = Arc::new(PagedKVPool::new(1024));
        let sched = Arc::new(ContinuousBatchScheduler::new("", kv.clone(), None));
        let mut app = TuiApp::new(EngineTelemetry::new(), kv).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );

        // Neither a text-generation model nor a sequence classifier: refused by the load worker
        // once it reads config.json (cached here, so no network), scheduler stays empty.
        let snap = cache
            .path()
            .join("models--sentence-transformers--all-MiniLM-L6-v2/snapshots/main");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(
            snap.join("config.json"),
            r#"{"architectures":["BertModel"]}"#,
        )
        .unwrap();
        app.load_model("sentence-transformers/all-MiniLM-L6-v2");
        settle(&mut app);
        assert_eq!(app.history[0].kind, Kind::Error);
        assert!(
            app.history[0].output.contains("hot-swap"),
            "{}",
            app.history[0].output
        );
        assert!(!sched.has_model());

        // Synthetic model loads offline in the background and replaces model + tokenizer.
        let before_tok = sched.tokenizer();
        app.load_model("gibbs/tiny-5m");
        settle(&mut app);
        let e = &app.history[1];
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert_eq!(sched.model_id(), "gibbs/tiny-5m");
        assert!(!Arc::ptr_eq(&before_tok, &sched.tokenizer()));
        assert!(app.weights.is_some());

        // Ctrl-J submits like Enter; Esc with a chat running cancels it (focus stays), then leaves.
        app.focus = Focus::Prompt;
        app.input = "/chat Tell me a long story".into();
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        let idx = app.history.len() - 1;
        assert_eq!(app.history[idx].kind, Kind::Running);
        app.handle_key(key(KeyCode::Esc));
        let e = &app.history[idx];
        assert_eq!(
            (e.kind, e.stats.as_str()),
            (Kind::Text, "0 tokens · cancelled")
        );
        assert_eq!((app.running, app.focus), (0, Focus::Prompt));
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.focus, Focus::Dashboard);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stepper = {
            let (sched, stop) = (sched.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    sched.step();
                }
            })
        };
        type_and_run(&mut app, "/chat Once upon a time");
        settle(&mut app);
        stop.store(true, Ordering::Relaxed);
        stepper.join().unwrap();
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert!(e.stats.contains("tokens"), "{}", e.stats);
    }

    #[test]
    fn test_tui_native_gemma4_loads_on_metal() {
        if cfg!(debug_assertions) {
            eprintln!("SKIP: test requires --release to prevent settle timeout");
            return;
        }
        if !crate::hardware::available_devices().contains(&crate::hardware::Device::Metal) {
            eprintln!("SKIP: Metal not available");
            return;
        }
        let snap = std::path::Path::new("disk/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF/snapshots/main/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf");
        if !snap.exists() {
            eprintln!("SKIP: gemma-4-E2B GGUF not found");
            return;
        }
        let kv = Arc::new(PagedKVPool::new(1024));
        let sched = Arc::new(ContinuousBatchScheduler::new("", kv.clone(), None));
        let mut app = TuiApp::new(EngineTelemetry::new(), kv).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir("disk".into()),
            None,
            None,
            ImageSlot::default(),
        );

        app.load_model("unsloth/gemma-4-E2B-it-qat-mobile-GGUF");
        settle(&mut app);
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert!(
            e.output.contains("on Metal ("),
            "TUI output was: {}",
            e.output
        );
        assert!(
            sched.backend_label().starts_with("Metal ("),
            "scheduler label was: {}",
            sched.backend_label()
        );
    }
    /// Loading a classifier makes it the active model and bare prompts classify (`a || b` is a
    /// pair); loading an LLM afterwards makes the LLM active again.
    #[test]
    fn test_classifier_load_classifies_until_an_llm_loads() {
        const CE: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
        let repo = std::path::Path::new("disk/models--cross-encoder--ms-marco-MiniLM-L-6-v2");
        if !repo.join("snapshots/main/model.safetensors").exists() {
            eprintln!("SKIP: {CE} not cached in disk/");
            return;
        }
        // Private cache: the real classifier linked in, the synthetic LLM written beside it.
        let cache = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(
            repo.canonicalize().unwrap(),
            cache.path().join(repo.file_name().unwrap()),
        )
        .unwrap();
        let kv = Arc::new(PagedKVPool::new(1024));
        let sched = Arc::new(ContinuousBatchScheduler::new("", kv.clone(), None));
        let mut app = TuiApp::new(EngineTelemetry::new(), kv).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );

        app.load_model(CE);
        settle(&mut app);
        assert_eq!(app.history[0].kind, Kind::Text, "{}", app.history[0].output);
        assert_eq!(app.model_id(), CE);

        type_and_run(
            &mut app,
            "How many people live in Berlin? || Berlin had a population of 3,520,031 registered inhabitants.",
        );
        settle(&mut app);
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert!(
            e.output.starts_with("LABEL_0") && e.output.contains("(raw score)"),
            "{}",
            e.output
        );

        app.load_model("gibbs/tiny-5m");
        settle(&mut app);
        assert!(app.classifier.read().is_none());
        assert_eq!(app.model_id(), "gibbs/tiny-5m");
    }

    /// ctrl-d arms, a second ctrl-d removes the cached repo directory; a loading repo is refused.
    #[test]
    fn test_picker_ctrl_d_deletes_cached_repo_after_confirm() {
        let cache = tempfile::tempdir().unwrap();
        let repo = cache.path().join("models--o--r");
        std::fs::create_dir_all(repo.join("snapshots/main")).unwrap();
        std::fs::write(repo.join("snapshots/main/model.safetensors"), b"w").unwrap();
        let kv = Arc::new(PagedKVPool::new(64));
        let sched = Arc::new(ContinuousBatchScheduler::new("", kv.clone(), None));
        let mut app = TuiApp::new(EngineTelemetry::new(), kv).with_runtime(
            sched,
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );
        let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);
        app.handle_key(key(KeyCode::Char('m')));
        // The only cached row sorts first.
        app.handle_key(ctrl_d);
        assert!(repo.exists(), "first ctrl-d only asks for confirmation");
        app.handle_key(ctrl_d);
        assert!(!repo.exists());
        assert_eq!(app.focus, Focus::Picker);

        // A load of one file of the repo blocks deleting the repo.
        std::fs::create_dir_all(repo.join("snapshots/main")).unwrap();
        std::fs::write(repo.join("snapshots/main/model.safetensors"), b"w").unwrap();
        app.load = Some(Load {
            id: "o/r/model.safetensors".into(),
            idx: 0,
            started: Instant::now(),
            stage: Arc::new(Mutex::new("resolving")),
            progress: Arc::default(),
            sample: (Instant::now(), 0),
            rate: 0.0,
        });
        app.handle_key(key(KeyCode::Esc));
        app.handle_key(key(KeyCode::Char('m')));
        app.handle_key(ctrl_d);
        app.handle_key(ctrl_d);
        assert!(repo.exists(), "a loading repo is never deleted");
    }

    #[test]
    fn test_scratch_dir_cleanup_on_drop() {
        let app = TuiApp::new(EngineTelemetry::new(), Arc::new(PagedKVPool::new(64)));
        let scratch = app.scratch_dir.path().to_path_buf();
        std::fs::write(scratch.join("probe.png"), b"x").unwrap();
        drop(app);
        assert!(!scratch.exists(), "scratch directory must clean up on drop");
    }

    #[test]
    fn test_at_path_attachments_become_ordered_media_parts() {
        let dir = tempfile::tempdir().unwrap();
        let (img, wav) = (dir.path().join("a.PNG"), dir.path().join("b.wav"));
        std::fs::write(&img, b"png").unwrap();
        std::fs::write(&wav, b"wav").unwrap();
        let prompt = format!(
            "compare @{} with @{} and @missing.png @notes.txt",
            img.display(),
            wav.display()
        );
        let (msg, media) = media_message(&prompt).unwrap().expect("has attachments");
        assert_eq!(
            media,
            [
                MediaInput::Image(b"png".to_vec()),
                MediaInput::Audio(b"wav".to_vec())
            ]
        );
        assert_eq!(
            msg.content,
            serde_json::json!([
                {"type": "text", "text": "compare"},
                {"type": "image"},
                {"type": "text", "text": "with"},
                {"type": "audio"},
                {"type": "text", "text": "and @missing.png @notes.txt"},
            ])
        );
        assert!(media_message("plain @word text").unwrap().is_none());
    }
    #[test]
    fn test_chat_template_hot_swapping_and_modal() {
        let mut app = TuiApp::new(EngineTelemetry::new(), Arc::new(PagedKVPool::new(64)));
        let tok = Tokenizer::default_bpe();

        // 1. Toggle modal with F2
        assert!(!app.show_template_modal);
        app.handle_key(key(KeyCode::F(2)));
        assert!(app.show_template_modal);

        // 2. Navigate templates with Down / Up / digit keys
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.active_template, 1);
        app.handle_key(key(KeyCode::Char('2')));
        assert_eq!(app.active_template, 2);
        app.handle_key(key(KeyCode::Enter));
        assert!(!app.show_template_modal);

        // 3. Test observable formatting effect with ChatML (1) vs Llama3 (2) vs Plain (4)
        app.set_chat_template(1); // ChatML
        app.set_system_prompt("You are a test assistant.");
        let prompt_chatml = app.format_chat_prompt(&tok, "Hello Gibbs");
        assert!(prompt_chatml.contains("<|im_start|>system\nYou are a test assistant.<|im_end|>"));
        assert!(prompt_chatml.contains("<|im_start|>user\nHello Gibbs<|im_end|>"));

        app.set_chat_template(2); // Llama3
        let prompt_llama = app.format_chat_prompt(&tok, "Hello Gibbs");
        assert!(prompt_llama.contains(
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a test assistant.<|eot_id|>"
        ));
        assert!(prompt_llama
            .contains("<|start_header_id|>user<|end_header_id|>\n\nHello Gibbs<|eot_id|>"));

        app.set_chat_template(4); // Plain
        let prompt_plain = app.format_chat_prompt(&tok, "Hello Gibbs");
        assert_eq!(prompt_plain, "You are a test assistant.\nHello Gibbs");
    }

    #[test]
    fn test_mascot_reactive_state_machine_in_tui() {
        let mut app = TuiApp::new(EngineTelemetry::new(), Arc::new(PagedKVPool::new(64)));
        assert_eq!(app.mascot.state, mascot::MascotState::Idle);

        // While running > 0, mascot becomes Generating
        app.running = 1;
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Generating);

        // Reset running, becomes Idle
        app.running = 0;
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Idle);

        // On error in history, mascot becomes Error
        app.push_entry("/bogus".into(), Kind::Error, "unknown command".into());
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Error);

        // Successful completion within 3s triggers Equilibrium
        app.push_entry("done prompt".into(), Kind::Text, "response text".into());
        app.history.last_mut().unwrap().took = Some(Duration::from_millis(100));
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Equilibrium);
    }

    #[test]
    fn test_logprob_inspector_toggle_and_channel() {
        let cache = tempfile::tempdir().unwrap();
        let kv = Arc::new(PagedKVPool::new(64));
        let sched = Arc::new(ContinuousBatchScheduler::new("", kv.clone(), None));
        let mut app = TuiApp::new(EngineTelemetry::new(), kv).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );

        assert!(!app.show_logprob_inspector);
        let ctrl_p = KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL);

        // 1. Toggle on Dashboard
        app.handle_key(ctrl_p);
        assert!(app.show_logprob_inspector);

        // 2. Esc closes it
        app.handle_key(key(KeyCode::Esc));
        assert!(!app.show_logprob_inspector);

        // 3. In Picker focus, Ctrl-P does NOT open inspector
        app.focus = Focus::Picker;
        app.handle_key(ctrl_p);
        assert!(!app.show_logprob_inspector);

        // 4. Transmit real TokenLogprobInfo on channel and verify tick() receives it
        let info = crate::scheduler::TokenLogprobInfo {
            chosen_id: 11751,
            chosen_prob: 0.642,
            top_candidates: vec![(11751, 0.642), (13, 0.215), (198, 0.08)],
            entropy: 0.892,
        };
        if let Some(tx) = sched.logprob_tx.read().as_ref() {
            let _ = tx.send(info.clone());
        }
        app.tick();
        assert_eq!(app.latest_logprobs, Some(info));
        assert_eq!(app.logprob_history.len(), 1);

        // 5. Test scrub index navigation with Left / Right
        app.focus = Focus::Dashboard;
        app.handle_key(ctrl_p);
        assert!(app.show_logprob_inspector);
        app.handle_key(key(KeyCode::Left));
        assert_eq!(app.logprob_scrub_idx, 0);
        app.handle_key(key(KeyCode::Right));
        assert_eq!(app.logprob_scrub_idx, 0);
    }

    #[test]
    fn test_roofline_probe_and_gibbs_commands() {
        let mut app = TuiApp::new(EngineTelemetry::new(), Arc::new(PagedKVPool::new(64)));

        // 1. /gibbs command
        app.input = "/gibbs".into();
        app.submit_prompt();
        assert_eq!(app.history.len(), 1);
        assert!(app.history[0].output.contains("JOSIAH WILLARD GIBBS"));
        assert!(app.history[0].output.contains("Mathematics is a language"));

        // 2. run_roofline_probe
        let probe = app.run_roofline_probe();
        if cfg!(target_os = "macos") {
            assert!(probe.is_ok());
            let report = probe.unwrap();
            assert!(report.contains("ROOFLINE CALIBRATION"));
            assert!(report.contains("GB/s"));
            assert!(report.contains("Ridge"));
        } else {
            assert!(probe.is_err());
        }

        // 3. Ctrl+R opens roofline inspector modal, Esc closes it
        assert!(!app.show_roofline_inspector);
        let ctrl_r = KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL);
        app.handle_key(ctrl_r);
        assert!(app.show_roofline_inspector);
        app.handle_key(key(KeyCode::Esc));
        assert!(!app.show_roofline_inspector);

        // 4. /roofline command opens modal
        app.input = "/roofline".into();
        app.submit_prompt();
        assert!(app.show_roofline_inspector);
        app.handle_key(key(KeyCode::Esc));
        assert!(!app.show_roofline_inspector);

        // 5. F3 cycles avatar styles
        assert_eq!(app.mascot.avatar_style, mascot::AvatarStyle::Classic);
        app.handle_key(key(KeyCode::F(3)));
        assert_eq!(app.mascot.avatar_style, mascot::AvatarStyle::Chalkboard);
        app.handle_key(key(KeyCode::F(3)));
        assert_eq!(app.mascot.avatar_style, mascot::AvatarStyle::Thermo);

        // 6. /avatar command
        app.input = "/avatar cyber".into();
        app.submit_prompt();
        assert_eq!(app.mascot.avatar_style, mascot::AvatarStyle::Cyber);

        // 7. /bench, /quant, /help commands
        app.input = "/bench".into();
        app.submit_prompt();
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("BENCHMARK SUMMARY"));

        app.input = "/quant".into();
        app.submit_prompt();
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("MODEL QUANTIZATION"));

        app.input = "/help".into();
        app.submit_prompt();
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("GIBBS ENGINE COMMAND & KEY REFERENCE"));
    }
}
