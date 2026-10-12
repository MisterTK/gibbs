use chrono::Local;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LogLevel {
    Info,
    Warn,
    Debug,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Debug => "DEBUG",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub message: String,
}

/// Latency samples kept per metric; percentiles cover the most recent this many tokens.
const LATENCY_WINDOW: usize = 512;
/// Window of the rolling tokens/s figure.
const TPS_WINDOW: Duration = Duration::from_secs(3);

/// Recent latency samples (µs), oldest first.
#[derive(Default)]
struct Latencies {
    ttft_us: VecDeque<u32>,
    itl_us: VecDeque<u32>,
}

fn push_sample(q: &mut VecDeque<u32>, d: Duration) {
    if q.len() == LATENCY_WINDOW {
        q.pop_front();
    }
    q.push_back(d.as_micros().min(u32::MAX as u128) as u32);
}

/// (p50, p90) in ms, or zeros without samples.
fn percentiles(q: &VecDeque<u32>) -> (f32, f32) {
    if q.is_empty() {
        return (0.0, 0.0);
    }
    let mut v: Vec<u32> = q.iter().copied().collect();
    v.sort_unstable();
    let at = |p: f32| v[((v.len() - 1) as f32 * p).round() as usize] as f32 / 1000.0;
    (at(0.5), at(0.9))
}

pub struct EngineTelemetry {
    pub total_tokens_generated: AtomicU64,
    pub total_requests_completed: AtomicU64,
    pub is_tui_active: AtomicBool,
    pub logs: RwLock<VecDeque<LogEntry>>,
    token_events: Mutex<VecDeque<(Instant, usize)>>,
    latencies: Mutex<Latencies>,
    /// Speculative decoding: (draft tokens proposed, accepted, verify forwards, tokens they emitted).
    spec: Mutex<[u64; 4]>,
    model_size_bytes: AtomicU64,
    hardware_bandwidth_gbps: RwLock<f32>,
    active_device: RwLock<String>,
}

impl Default for EngineTelemetry {
    fn default() -> Self {
        Self {
            total_tokens_generated: AtomicU64::new(0),
            total_requests_completed: AtomicU64::new(0),
            is_tui_active: AtomicBool::new(false),
            logs: RwLock::new(VecDeque::with_capacity(500)),
            token_events: Mutex::new(VecDeque::with_capacity(1024)),
            latencies: Mutex::new(Latencies::default()),
            spec: Mutex::new([0; 4]),
            model_size_bytes: AtomicU64::new(0),
            hardware_bandwidth_gbps: RwLock::new(0.0),
            active_device: RwLock::new(String::new()),
        }
    }
}

impl EngineTelemetry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn log(&self, level: LogLevel, message: &str) {
        let timestamp = Local::now().format("%H:%M:%S%.3f").to_string();

        // Outside the TUI, logs go to the terminal; inside it, the TUI draws them.
        if !self.is_tui_active.load(Ordering::Relaxed) {
            match level {
                LogLevel::Info => println!("\x1b[32m[{}] [INFO]\x1b[0m {}", timestamp, message),
                LogLevel::Warn => eprintln!("\x1b[33;1m[{}] [WARN]\x1b[0m {}", timestamp, message),
                LogLevel::Debug => println!("\x1b[90m[{}] [DEBUG]\x1b[0m {}", timestamp, message),
            }
        }

        let mut queue = self.logs.write();
        if queue.len() >= 500 {
            queue.pop_front();
        }
        queue.push_back(LogEntry {
            timestamp,
            level,
            message: message.to_string(),
        });
    }

    pub fn log_info(&self, msg: &str) {
        self.log(LogLevel::Info, msg);
    }

    pub fn log_warn(&self, msg: &str) {
        self.log(LogLevel::Warn, msg);
    }

    pub fn log_debug(&self, msg: &str) {
        self.log(LogLevel::Debug, msg);
    }

    pub fn get_recent_logs(&self, limit: usize) -> Vec<LogEntry> {
        let queue = self.logs.read();
        let start = queue.len().saturating_sub(limit);
        queue.iter().skip(start).cloned().collect()
    }

    pub fn record_tokens(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.total_tokens_generated
            .fetch_add(count as u64, Ordering::Relaxed);
        let now = Instant::now();
        let mut events = self.token_events.lock();
        events.push_back((now, count));
        while events
            .front()
            .is_some_and(|&(t, _)| now.duration_since(t) > TPS_WINDOW)
        {
            events.pop_front();
        }
    }

    /// Tokens generated per second over the last [`TPS_WINDOW`].
    pub fn current_tps(&self) -> f32 {
        let now = Instant::now();
        let events = self.token_events.lock();
        let tokens: usize = events
            .iter()
            .filter(|&&(t, _)| now.duration_since(t) <= TPS_WINDOW)
            .map(|&(_, c)| c)
            .sum();
        tokens as f32 / TPS_WINDOW.as_secs_f32()
    }

    pub fn record_request_completed(&self) {
        self.total_requests_completed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Time from a request's submission to its first output token.
    pub fn record_ttft(&self, d: Duration) {
        push_sample(&mut self.latencies.lock().ttft_us, d);
    }

    /// Gap between consecutive output tokens of one request.
    pub fn record_itl(&self, d: Duration) {
        push_sample(&mut self.latencies.lock().itl_us, d);
    }

    /// Returns the most recent `limit` token-to-token latency samples in milliseconds (oldest first).
    pub fn recent_itl_ms(&self, limit: usize) -> Vec<f32> {
        let l = self.latencies.lock();
        let start = l.itl_us.len().saturating_sub(limit);
        l.itl_us
            .iter()
            .skip(start)
            .map(|&us| us as f32 / 1000.0)
            .collect()
    }

    /// One speculative step: `drafted` draft tokens, `accepted` of them kept, `emitted` output
    /// tokens from one target forward.
    pub fn record_spec(&self, drafted: usize, accepted: usize, emitted: usize) {
        let mut s = self.spec.lock();
        s[0] += drafted as u64;
        s[1] += accepted as u64;
        s[2] += 1;
        s[3] += emitted as u64;
    }

    pub fn set_hardware_profile(&self, device: &str, bandwidth_gbps: f32) {
        *self.active_device.write() = device.to_string();
        *self.hardware_bandwidth_gbps.write() = bandwidth_gbps;
    }

    pub fn set_bytes_per_token(&self, bytes: u64) {
        self.model_size_bytes.store(bytes, Ordering::Relaxed);
    }
    pub fn set_model_size_bytes(&self, bytes: u64) {
        self.model_size_bytes.store(bytes, Ordering::Relaxed);
    }
    pub fn bytes_per_token(&self) -> u64 {
        self.model_size_bytes.load(Ordering::Relaxed)
    }
    pub fn model_size_bytes(&self) -> u64 {
        self.model_size_bytes.load(Ordering::Relaxed)
    }

    pub fn get_summary(&self) -> TelemetrySnapshot {
        let (ttft, itl) = {
            let l = self.latencies.lock();
            (percentiles(&l.ttft_us), percentiles(&l.itl_us))
        };
        let [drafted, accepted, forwards, emitted] = *self.spec.lock();
        let model_size = self.model_size_bytes.load(Ordering::Relaxed) as f32 / 1e9;
        let bw = *self.hardware_bandwidth_gbps.read();
        let current_tps = self.current_tps();
        let effective_bandwidth_gbps = if model_size > 0.0 {
            current_tps * model_size
        } else {
            0.0
        };
        let roofline_pct = if bw > 0.0 && effective_bandwidth_gbps > 0.0 {
            let pct = effective_bandwidth_gbps / bw * 100.0;
            if pct > 100.5 {
                static LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "roofline > 100%: ceiling {bw:.1} GB/s is below the achieved {effective_bandwidth_gbps:.1} GB/s; re-run the host profile"
                    );
                }
            }
            pct
        } else {
            0.0
        };
        let device_name = self.active_device.read().clone();
        TelemetrySnapshot {
            total_tokens: self.total_tokens_generated.load(Ordering::Relaxed),
            completed_requests: self.total_requests_completed.load(Ordering::Relaxed),
            current_tps,
            ttft_p50_ms: ttft.0,
            ttft_p90_ms: ttft.1,
            itl_p50_ms: itl.0,
            itl_p90_ms: itl.1,
            spec_drafted: drafted,
            spec_accepted: accepted,
            spec_forwards: forwards,
            spec_accept_rate: accepted as f32 / drafted.max(1) as f32,
            spec_tokens_per_forward: emitted as f32 / forwards.max(1) as f32,
            memory_bandwidth_gbps: bw,
            effective_bandwidth_gbps,
            roofline_pct,
            device_name,
        }
    }
}

/// Engine counters; latency percentiles cover the last [`LATENCY_WINDOW`] samples.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TelemetrySnapshot {
    pub total_tokens: u64,
    pub completed_requests: u64,
    pub current_tps: f32,
    pub ttft_p50_ms: f32,
    pub ttft_p90_ms: f32,
    pub itl_p50_ms: f32,
    pub itl_p90_ms: f32,
    /// Speculative decoding since start: draft tokens proposed and accepted, target forwards
    /// that verified them, accepted / drafted, and output tokens per such forward.
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub spec_forwards: u64,
    pub spec_accept_rate: f32,
    pub spec_tokens_per_forward: f32,
    #[serde(default)]
    pub memory_bandwidth_gbps: f32,
    #[serde(default)]
    pub effective_bandwidth_gbps: f32,
    #[serde(default)]
    pub roofline_pct: f32,
    #[serde(default)]
    pub device_name: String,
}
