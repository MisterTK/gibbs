//! Re-encode policy (sprint-2 S3.5): which format each streamed 2-D weight matrix (Linear weights,
//! LM head) is stored in on a GPU load, how a re-encode is gated per tensor, and the on-disk cache of
//! the result. The whole-model KL gate and the fallback chain live in [`crate::model::weight_format`].
//!
//! Rules, per projection, from its source format `src`:
//! * `MaxSpeed`: Q4_K when `src` is above 4.5 bpw or has no GPU codec, else native.
//! * `LowBitNative`: float sources become Q8_0; a codec-less source becomes the smallest of
//!   Q4_K / Q5_K / Q6_K / Q8_0 whose bpw is at least the source's; else native.
//! * `BitExact`: native when a codec exists; a codec-less source follows the `LowBitNative` rule.
//!
//! "Native" carries the checkpoint's own blocks (GGUF block formats and the repacked source encodings
//! of [`crate::model::repack`]) byte for byte; float sources are re-stored at their own dtype, which
//! is exact. Where blocks cannot be carried (mixed formats inside one fused projection, reordered
//! rows) the f32 values are encoded in the source format when an encoder exists (exact for Q4_0 and
//! Q8_0, one more quantization step for the K-quants, recorded as `lossy`).

use crate::graph::codec::{codec, CODEC_VERSION};
use crate::hardware::QuantFormat as Q;
use crate::model::graph_lm::{GraphLm, ProjRole, Projection, ProjectionEncoder};
use crate::model::layers::{gguf_block_codec, QuantizedLinearLayer, QuantizedWeight};
use crate::model::weight_format::ReencodePolicy;
use crate::model::VirtualTensor;
use bytes::Bytes;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Re-encode formats from smallest to largest.
pub const LADDER: [Q; 4] = [Q::GGUF_Q4_K_M, Q::GGUF_Q5_K_M, Q::GGUF_Q6_K, Q::GGUF_Q8_0];
/// Per-tensor gate: relative RMSE of the re-encoded weights against the source values.
pub const TENSOR_RMSE_LIMIT: f64 = 0.08;
/// Values per row run a native re-encode trial decodes and encodes at a time (8 MB of f32).
const TRIAL_VALUES: usize = 1 << 21;

pub fn is_float(f: Q) -> bool {
    matches!(f, Q::FP32 | Q::FP16 | Q::BF16)
}

/// Whether Metal reads `f` directly (float kernels, a registered codec, or a source encoding that
/// [`crate::model::repack`] turns into one).
pub fn has_gpu_codec(f: Q) -> bool {
    is_float(f)
        || codec(f).is_some()
        || matches!(
            f,
            Q::FP8_E4M3 | Q::FP8_E5M2 | Q::BNB_NF4 | Q::BNB_FP4 | Q::GPTQ_W4A16 | Q::AWQ_W4A16
        )
}

/// Formats [`QuantizedLinearLayer::from_f32`] can produce.
pub fn encodable(f: Q) -> bool {
    is_float(f) || matches!(f, Q::GGUF_Q4_0 | Q::GGUF_Q8_0) || LADDER[..3].contains(&f)
}

fn block_elems(f: Q) -> usize {
    gguf_block_codec(f).map_or(1, |c| c.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Native,
    Encode(Q),
}

/// The smallest ladder format whose bpw is at least `bpw`.
pub fn smallest_ladder_ge(bpw: f32) -> Q {
    LADDER
        .into_iter()
        .find(|f| f.bits_per_weight() >= bpw)
        .unwrap_or(Q::GGUF_Q8_0)
}

/// The policy's decision for a projection stored in `src`; `MaxSpeed` reads this host's measured
/// codec rates (falling back to the static rule where none are measured yet).
pub fn rule(policy: ReencodePolicy, src: Q) -> Target {
    rule_with(policy, src, &crate::model::variant::host_gbps)
}

/// A candidate must need under this fraction of the incumbent's time per weight to replace it:
/// run-to-run noise in the measured rates must not flip a tensor's format (and the re-encode cache).
const TIME_MARGIN: f64 = 0.9;

/// [`rule`] with explicit rates (`GB/s` of a format's GEMV; `None` = not measured).
///
/// `MaxSpeed` picks the target minimizing `bpw / GB/s` (time to stream one weight) among the
/// source format (when a GPU codec reads it), the ladder and Q4_0, restricted to formats the
/// encoder can produce. A source above 4.5 bpw never drops below 4 bpw; a source at or below
/// 4.5 bpw only moves to equal-or-higher precision, so a lossy re-encode never loses bits.
pub fn rule_with(policy: ReencodePolicy, src: Q, rate: &dyn Fn(Q) -> Option<f64>) -> Target {
    let base = static_rule(policy, src);
    if policy != ReencodePolicy::MaxSpeed {
        return base;
    }
    let bpw = src.bits_per_weight();
    let floor = if bpw > 4.5 { 4.0 } else { bpw - 0.01 };
    let time = |f: Q| {
        rate(f)
            .map(|g| f.bits_per_weight() as f64 / g)
            .filter(|t| t.is_finite() && *t > 0.0)
    };
    let incumbent = match base {
        Target::Native => src,
        Target::Encode(f) => f,
    };
    let (mut best, mut best_t) = (incumbent, time(incumbent));
    let readable = has_gpu_codec(src);
    for f in std::iter::once(src).chain(LADDER).chain([Q::GGUF_Q4_0]) {
        let allowed = (f == src && readable) || encodable(f);
        if f == incumbent || !allowed || f.bits_per_weight() < floor {
            continue;
        }
        if let Some(t) = time(f) {
            if best_t.is_none_or(|bt| t < bt * TIME_MARGIN) {
                (best, best_t) = (f, Some(t));
            }
        }
    }
    if best == src && readable {
        Target::Native
    } else {
        Target::Encode(best)
    }
}

/// The rate-independent rule (what runs before the codec benchmark has measured this host).
fn static_rule(policy: ReencodePolicy, src: Q) -> Target {
    let codec_less = !has_gpu_codec(src);
    match policy {
        ReencodePolicy::MaxSpeed if src.bits_per_weight() > 4.5 || codec_less => {
            Target::Encode(Q::GGUF_Q4_K_M)
        }
        ReencodePolicy::LowBitNative if is_float(src) => Target::Encode(Q::GGUF_Q8_0),
        ReencodePolicy::LowBitNative | ReencodePolicy::BitExact if codec_less => {
            Target::Encode(smallest_ladder_ge(src.bits_per_weight()))
        }
        _ => Target::Native,
    }
}

/// The next ladder format above `f`.
pub fn bump(f: Q) -> Option<Q> {
    LADDER
        .into_iter()
        .find(|g| g.bits_per_weight() > f.bits_per_weight())
}

/// `f`, or the next larger format whose blocks tile `cols` (a float dtype when none does).
pub fn fit(mut f: Q, cols: usize, float: Q) -> Q {
    // llama.cpp's convention for rows that hold no whole 256-element block: 4-bit K -> block-32 Q4_0.
    if f == Q::GGUF_Q4_K_M && cols % 256 != 0 && cols % 32 == 0 {
        return Q::GGUF_Q4_0;
    }
    while cols % block_elems(f) != 0 {
        f = bump(f).unwrap_or(float);
    }
    f
}

/// The format a checkpoint tensor is stored in.
pub fn declared_format(t: &VirtualTensor) -> Q {
    if gguf_block_codec(t.quant_format).is_some() {
        return t.quant_format;
    }
    match t.dtype.as_str() {
        "F32" | "F64" => Q::FP32,
        "F16" => Q::FP16,
        "BF16" => Q::BF16,
        "F8_E4M3" => Q::FP8_E4M3,
        "F8_E5M2" => Q::FP8_E5M2,
        // Packed integers under a float tag: a quantized checkpoint (EXL3, GLQ, ...) no codec reads.
        // TurboQuant is the codec-less low-bit tag: the ladder lifts it to Q4_K_M.
        "U8" | "I8" | "I16" | "U16" | "I32" | "U32" | "I64" | "U64" if is_float(t.quant_format) => {
            Q::TurboQuant
        }
        _ => t.quant_format,
    }
}

/// Counts finished projections for the load-stage text.
#[derive(Default)]
pub struct ReencodeProgress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
}

impl ReencodeProgress {
    /// `re-encoding for this GPU (max-speed): 12/197 tensors`
    pub fn text(&self, policy: ReencodePolicy) -> String {
        format!(
            "re-encoding for this GPU ({}): {}/{} tensors",
            policy.name(),
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed)
        )
    }
}

/// One projection's outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub name: String,
    pub source: Q,
    pub stored: Q,
    /// The checkpoint's blocks were copied unchanged.
    pub carried: bool,
    /// The stored values differ from the source's (a lossy encode, or a K-quant requantized from f32).
    pub lossy: bool,
    pub rmse: Option<f64>,
    pub bytes: u64,
}

pub enum Mode {
    /// The exact reference: carry the checkpoint's blocks, else keep (or exactly re-store) the f32 values.
    Source,
    Policy {
        policy: ReencodePolicy,
        /// Attention and LM head keep their source format.
        ffn_only: bool,
        /// Gate each re-encode on this relative RMSE ([`TENSOR_RMSE_LIMIT`] in production), bumping to
        /// the next larger format until it passes.
        rmse_limit: Option<f64>,
    },
    /// Weights read back from a cache entry.
    Cache(Arc<CacheReader>),
}

pub struct Encoder {
    pub mode: Mode,
    pub threads: usize,
    pub progress: Option<Arc<ReencodeProgress>>,
    records: Mutex<Vec<Record>>,
    /// A cached tensor was missing: the entry does not match this model.
    pub cache_miss: AtomicBool,
}

impl Encoder {
    pub fn new(mode: Mode, threads: usize, progress: Option<Arc<ReencodeProgress>>) -> Self {
        Self {
            mode,
            threads: threads.max(1),
            progress,
            records: Mutex::new(Vec::new()),
            cache_miss: AtomicBool::new(false),
        }
    }

    pub fn records(&self) -> Vec<Record> {
        self.records.lock().clone()
    }

    /// Encodes `layer.weight` (f32) as `f`, falling back by `fit` when `f`'s blocks do not tile.
    fn encode_f32(&self, p: &Projection<'_>, f: Q) -> (Q, QuantizedWeight) {
        let (in_f, out_f) = (p.layer.in_features, p.layer.out_features);
        let QuantizedWeight::Uncompressed { data, .. } = &p.layer.weight else {
            unreachable!("loaders hand encoders f32 layers")
        };
        let f = fit(f, in_f, Q::FP32);
        (f, encode_weight(data, in_f, out_f, f, self.threads))
    }

    /// Native storage: the checkpoint's blocks, or the f32 values re-stored in the source format.
    fn native(
        &self,
        p: &Projection<'_>,
        src: Q,
        carried: Option<(Q, Vec<u8>)>,
        exact_only: bool,
        rec: &mut Record,
    ) -> QuantizedWeight {
        let (in_f, out_f) = (p.layer.in_features, p.layer.out_features);
        if let Some((f, bytes)) = carried {
            (rec.stored, rec.carried) = (f, true);
            return gguf_weight(f, Bytes::from(bytes), in_f, out_f);
        }
        let QuantizedWeight::Uncompressed { data, .. } = &p.layer.weight else {
            unreachable!("loaders hand encoders f32 layers")
        };
        match src {
            Q::FP32 => p.layer.weight.clone(),
            Q::FP16 | Q::BF16 | Q::GGUF_Q4_0 | Q::GGUF_Q8_0 if in_f % block_elems(src) == 0 => {
                rec.stored = src;
                encode_weight(data, in_f, out_f, src, self.threads)
            }
            _ if exact_only => p.layer.weight.clone(),
            f if encodable(f) && in_f % block_elems(f) == 0 => {
                (rec.stored, rec.lossy) = (f, true);
                encode_weight(data, in_f, out_f, f, self.threads)
            }
            _ => {
                // No encoder for the source format: the smallest encodable format at least as large.
                let (f, w) = self.encode_f32(p, smallest_ladder_ge(src.bits_per_weight()));
                (rec.stored, rec.lossy) = (f, true);
                w
            }
        }
    }
}

impl ProjectionEncoder for Encoder {
    fn encode(&self, p: Projection<'_>) -> QuantizedLinearLayer {
        let (in_f, out_f) = (p.layer.in_features, p.layer.out_features);
        // The carried blocks' own format when there are any, else what the checkpoint declares.
        let carried = if matches!(self.mode, Mode::Cache(_)) {
            None
        } else {
            carry(&p)
        };
        let src = carried
            .as_ref()
            .map_or_else(|| declared_source(&p), |c| c.0);
        let mut rec = Record {
            name: p.layer.name.clone(),
            source: src,
            stored: Q::FP32,
            carried: false,
            lossy: false,
            rmse: None,
            bytes: 0,
        };
        let weight = match &self.mode {
            Mode::Cache(c) => c.weight(&p.layer.name, in_f, out_f).unwrap_or_else(|| {
                self.cache_miss.store(true, Ordering::Relaxed);
                p.layer.weight.clone()
            }),
            Mode::Source => self.native(&p, src, carried, true, &mut rec),
            Mode::Policy {
                policy,
                ffn_only,
                rmse_limit,
            } => {
                let target = if *ffn_only && p.role != ProjRole::Ffn {
                    Target::Native
                } else {
                    rule(*policy, src)
                };
                match target {
                    Target::Native => self.native(&p, src, carried, false, &mut rec),
                    Target::Encode(f) => {
                        let (mut f, mut w) = self.encode_f32(&p, f);
                        rec.lossy = true;
                        if let Some(limit) = *rmse_limit {
                            let QuantizedWeight::Uncompressed { data, .. } = &p.layer.weight else {
                                unreachable!()
                            };
                            let mut rmse = relative_rmse(data, &w.to_f32());
                            while rmse > limit {
                                let Some(next) = bump(f) else { break };
                                (f, w) = self.encode_f32(&p, next);
                                rmse = relative_rmse(data, &w.to_f32());
                            }
                            rec.rmse = Some(rmse);
                        }
                        rec.stored = f;
                        w
                    }
                }
            }
        };
        rec.bytes = weight_bytes(&weight) as u64;
        if let Some(pr) = &self.progress {
            pr.done.fetch_add(1, Ordering::Relaxed);
        }
        self.records.lock().push(rec);
        let mut layer =
            QuantizedLinearLayer::from_quantized_weight(&p.layer.name, in_f, out_f, weight);
        layer.bias = p.layer.bias;
        layer
    }
}

/// `sqrt(Σ(x-y)² / Σx²)`.
pub fn relative_rmse(x: &[f32], y: &[f32]) -> f64 {
    let mut err = (0.0, 0.0);
    add_sq_err(&mut err, x, y);
    rmse_of(err)
}

/// Adds `(Σ(x-y)², Σx²)` of `x`, `y` to `err`.
fn add_sq_err(err: &mut (f64, f64), x: &[f32], y: &[f32]) {
    for (&a, &b) in x.iter().zip(y) {
        err.0 += (a as f64 - b as f64).powi(2);
        err.1 += (a as f64).powi(2);
    }
}

fn rmse_of((num, den): (f64, f64)) -> f64 {
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

/// The largest-bpw format among the checkpoint tensors a projection came from (FP32 when unknown).
fn declared_source(p: &Projection<'_>) -> Q {
    let Some((m, _)) = p.checkpoint else {
        return Q::FP32;
    };
    p.sources
        .iter()
        .filter_map(|n| m.tensors.get(n))
        .map(declared_format)
        .max_by(|a, b| a.bits_per_weight().total_cmp(&b.bits_per_weight()))
        .unwrap_or(Q::FP32)
}

fn gguf_weight(f: Q, bytes: Bytes, in_f: usize, out_f: usize) -> QuantizedWeight {
    if matches!(f, Q::FP16 | Q::BF16) {
        QuantizedWeight::ZeroCopySlice {
            data: bytes,
            in_features: in_f,
            out_features: out_f,
            format: f,
        }
    } else {
        QuantizedWeight::GgufQuantized {
            format: f,
            raw_data: bytes,
            in_features: in_f,
            out_features: out_f,
        }
    }
}

pub fn weight_bytes(w: &QuantizedWeight) -> usize {
    match w {
        QuantizedWeight::Uncompressed { data, .. } => data.len() * 4,
        QuantizedWeight::ZeroCopySlice { data, .. } => data.len(),
        QuantizedWeight::GgufQuantized { raw_data, .. } => raw_data.len(),
        other => (other.num_elements() as f32 * other.bits_per_weight() / 8.0) as usize,
    }
}

/// Encodes row-major `w` as `f` (a float or an encodable block format whose blocks tile `in_f`),
/// splitting rows over `threads` workers at utility QoS; row chunks concatenate exactly.
pub(crate) fn encode_weight(w: &[f32], in_f: usize, out_f: usize, f: Q, threads: usize) -> QuantizedWeight {
    if f == Q::FP32 {
        return QuantizedWeight::Uncompressed {
            data: w.to_vec(),
            in_features: in_f,
            out_features: out_f,
        };
    }
    let workers = if w.len() < 1 << 16 {
        1
    } else {
        threads.min(out_f)
    };
    let rows = out_f.div_ceil(workers);
    let raw = |c: &[f32]| -> Bytes {
        let l = QuantizedLinearLayer::from_f32("", in_f, c.len() / in_f, c, f)
            .unwrap_or_else(|e| panic!("cannot encode as {f:?}: {e}"));
        match l.weight {
            QuantizedWeight::GgufQuantized { raw_data, .. } => raw_data,
            QuantizedWeight::ZeroCopySlice { data, .. } => data,
            _ => unreachable!("{f:?} encodes to blocks or a half float"),
        }
    };
    let parts: Vec<Bytes> = if workers == 1 {
        vec![raw(w)]
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = w
                .chunks(rows * in_f)
                .map(|c| {
                    s.spawn(move || {
                        crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
                        raw(c)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("encoder thread panicked"))
                .collect()
        })
    };
    let bytes = if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        Bytes::from(parts.concat())
    };
    if matches!(f, Q::FP16 | Q::BF16) {
        QuantizedWeight::ZeroCopySlice {
            data: bytes,
            in_features: in_f,
            out_features: out_f,
            format: f,
        }
    } else {
        QuantizedWeight::GgufQuantized {
            format: f,
            raw_data: bytes,
            in_features: in_f,
            out_features: out_f,
        }
    }
}

/// The checkpoint's own blocks for a projection, when every constituent tensor is stored in one
/// block format (a GGUF type or a [`crate::model::repack`] encoding): whole tensors, or GQA k/v
/// tensors repeated per head exactly as the loader replicated their f32 rows.
fn carry(p: &Projection<'_>) -> Option<(Q, Vec<u8>)> {
    let (m, data) = p.checkpoint?;
    let (in_f, out_f) = (p.layer.in_features, p.layer.out_features);
    if p.sources.is_empty() {
        return None;
    }
    let mut parts: Vec<(Q, std::borrow::Cow<'_, [u8]>, usize)> = Vec::new();
    for name in &p.sources {
        let t = m.tensors.get(name)?;
        let part = match crate::model::repack::repack(m, data, name) {
            Ok(Some(r)) if r.cols == in_f => (r.format, std::borrow::Cow::Owned(r.bytes), r.rows),
            Ok(Some(_)) => return None,
            Ok(None) => {
                let (elems, block_bytes, _) = gguf_block_codec(t.quant_format)?;
                if in_f % elems != 0 || t.byte_len % (in_f / elems * block_bytes) != 0 {
                    return None;
                }
                let bytes = data.get(t.byte_offset..t.byte_offset + t.byte_len)?;
                (
                    t.quant_format,
                    std::borrow::Cow::Borrowed(bytes),
                    t.byte_len / (in_f / elems * block_bytes),
                )
            }
            Err(_) => return None,
        };
        parts.push(part);
    }
    let f = parts[0].0;
    if parts.iter().any(|p| p.0 != f) {
        return None;
    }
    let (elems, block_bytes, _) = gguf_block_codec(f)?;
    let row_bytes = in_f / elems * block_bytes;
    // fused q/k/v: k and v hold fewer rows than q when the heads were grouped.
    let q_rows = parts[0].2;
    let mut out = Vec::with_capacity(out_f * row_bytes);
    for (i, (_, bytes, rows)) in parts.iter().enumerate() {
        let rep = if parts.len() == 3 && i > 0 && *rows < q_rows {
            q_rows / rows
        } else {
            1
        };
        if rep > 1 && (q_rows % rows != 0 || rows % p.head_dim.max(1) != 0) {
            return None;
        }
        if rep == 1 {
            out.extend_from_slice(bytes);
        } else {
            for head in bytes.chunks(p.head_dim * row_bytes) {
                for _ in 0..rep {
                    out.extend_from_slice(head);
                }
            }
        }
    }
    if out.len() != out_f * row_bytes {
        return None;
    }
    // The blocks must decode to exactly the f32 rows the loader built: this rejects any layout the
    // loader changed after reading (reordered heads, transposes, ternarized rows) without special cases.
    let QuantizedWeight::Uncompressed { data: want, .. } = &p.layer.weight else {
        return None;
    };
    let decode = gguf_block_codec(f)?.2;
    let mut y = vec![0.0f32; elems];
    for (b, w) in out.chunks_exact(block_bytes).zip(want.chunks_exact(elems)) {
        decode(b, &mut y);
        if y.iter().zip(w).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return None;
        }
    }
    Some((f, out))
}

// ---- native loaders -------------------------------------------------------------------------

/// What the native loaders did to their tensors since [`reset_native_stats`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeStats {
    /// Projection matrices considered.
    pub tensors: usize,
    /// Matrices re-encoded to another format (all lossy: the source blocks differ from the stored ones).
    pub reencoded: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
}

static NATIVE_STATS: Mutex<NativeStats> = Mutex::new(NativeStats {
    tensors: 0,
    reencoded: 0,
    bytes_before: 0,
    bytes_after: 0,
});

pub fn reset_native_stats() {
    *NATIVE_STATS.lock() = NativeStats::default();
}

pub fn native_stats() -> NativeStats {
    *NATIVE_STATS.lock()
}

/// Test hook: pretend the current load re-encoded `n` tensors.
#[cfg(test)]
pub(crate) fn fake_native_reencoded(n: usize) {
    let mut s = NATIVE_STATS.lock();
    (s.tensors, s.reencoded) = (n, n);
}

static NATIVE_APPLIED: Mutex<Option<(ReencodePolicy, u64)>> = Mutex::new(None);

/// Records the policy the last native load ended up with and the bytes per token it saved.
pub fn set_native_applied(policy: ReencodePolicy, saved_bytes: u64) {
    *NATIVE_APPLIED.lock() = Some((policy, saved_bytes));
}

/// `(policy, bytes saved over the source blocks)` of the last native load that reached the gate.
pub fn native_applied() -> Option<(ReencodePolicy, u64)> {
    *NATIVE_APPLIED.lock()
}

/// Whether `name` (an HF-style tensor name) is a feed-forward projection.
fn is_ffn_name(name: &str) -> bool {
    ["mlp", "ffn", "feed_forward"]
        .iter()
        .any(|s| name.contains(s))
}

/// Gather tables (token / per-layer embeddings, which a tied head shares) keep their source blocks:
/// a gathered row's precision is not a streaming cost.
fn is_gather_table(name: &str) -> bool {
    ["embed", "token_embd", "ple"]
        .iter()
        .any(|s| name.contains(s))
}

/// Whether `f` still reads faster than the stored `src` by the rule's margin under `rate`; without
/// measured rates, whether it at least holds fewer bits.
fn beats_source(f: Q, src: Q, rate: &dyn Fn(Q) -> Option<f64>) -> bool {
    let time = |q: Q| rate(q).map(|g| q.bits_per_weight() as f64 / g);
    match (time(f), time(src)) {
        (Some(a), Some(b)) => a < b * TIME_MARGIN,
        _ => f.bits_per_weight() < src.bits_per_weight(),
    }
}

/// A native loader's `rows x cols` projection `name`, stored as `src` blocks in `raw`, under the
/// effective policy `cfg`: `Some((format, blocks))` when the policy re-stores it in a faster format
/// (every such re-encode is lossy and counted in [`native_stats`]), `None` to keep the stored blocks.
pub fn reencode_native(
    cfg: &crate::model::weight_format::ReencodeConfig,
    name: &str,
    src: Q,
    rows: usize,
    cols: usize,
    raw: &[u8],
) -> Option<(Q, Vec<u8>)> {
    let ffn = match &cfg.ffn_weights {
        Some(pats) => pats.iter().any(|p| crate::arch::glob(p, name)),
        None => is_ffn_name(name),
    };
    if is_gather_table(name) || (cfg.ffn_only && !ffn) {
        return None;
    }
    let mut stats = NATIVE_STATS.lock();
    stats.tensors += 1;
    drop(stats);
    let Target::Encode(f) = rule(cfg.policy, src) else {
        return None;
    };
    let f = fit(f, cols, Q::FP32);
    if f == src || !encodable(f) {
        return None;
    }
    // The source's own decoder, so the values are exactly what the GPU would have read.
    let (elems, block_bytes, decode) = gguf_block_codec(src)?;
    let row_bytes = cols / elems * block_bytes;
    if cols % elems != 0 || raw.len() != rows * row_bytes {
        return None;
    }
    let threads = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .saturating_sub(1)
        .max(1);
    // Runs of rows, not the whole tensor, and blocks kept only for the format that is stored: the
    // allocator keeps freed large blocks resident, so whole-tensor trial buffers (decoded source,
    // round trip, rejected blocks) would stay in the process's resident memory after the load.
    let run = (TRIAL_VALUES / cols).max(1);
    let mut x = vec![0.0f32; run.min(rows) * cols];
    // `f`'s relative RMSE over the tensor, and with `keep` its blocks (`None`: no block format).
    let mut encode = |f: Q, keep: bool| {
        let mut err = (0.0, 0.0);
        let mut blocks = (keep && !is_float(f)).then(|| Vec::with_capacity(rows * f.tensor_bytes(cols)));
        for part in raw.chunks(run * row_bytes) {
            let x = &mut x[..part.len() / block_bytes * elems];
            for (b, out) in part.chunks_exact(block_bytes).zip(x.chunks_exact_mut(elems)) {
                decode(b, out);
            }
            let w = encode_weight(x, cols, x.len() / cols, f, threads);
            add_sq_err(&mut err, x, &w.to_f32());
            blocks = blocks.and_then(|mut b: Vec<u8>| match &w {
                QuantizedWeight::GgufQuantized { raw_data, .. } => {
                    b.extend_from_slice(raw_data);
                    Some(b)
                }
                _ => None,
            });
        }
        (rmse_of(err), blocks)
    };
    let (mut f, mut rmse) = (f, encode(f, false).0);
    while rmse > TENSOR_RMSE_LIMIT {
        let Some(next) = bump(f).map(|n| fit(n, cols, Q::FP32)) else {
            break;
        };
        (f, rmse) = (next, encode(next, false).0);
    }
    // The RMSE bump may have walked past the source's own cost (Q4_K -> Q4_0 failing the gate lands on
    // Q5_K, bigger and slower than what was stored): then the stored blocks stay.
    if !beats_source(f, src, &crate::model::variant::host_gbps) {
        return None;
    }
    // A float target stores no blocks.
    let blocks = encode(f, true).1?;
    let mut stats = NATIVE_STATS.lock();
    stats.reencoded += 1;
    (stats.bytes_before, stats.bytes_after) = (
        stats.bytes_before + raw.len() as u64,
        stats.bytes_after + blocks.len() as u64,
    );
    tracing::debug!("reencode {name} {rows}x{cols}: {src:?} -> {f:?} (rmse {rmse:.4})");
    Some((f, blocks))
}

// ---- cache ----------------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReencodeReport {
    pub requested: String,
    /// The attempt that was kept: `max-speed`, `max-speed (FFN only)`, `low-bit-native`, `bit-exact`.
    pub applied: String,
    pub mean_kl: Option<f64>,
    pub top1: Option<f64>,
    /// `mean KL 0.0123 nats, top-1 97%`, `bit-exact`, or `per-tensor RMSE <= 0.08 (reason)`.
    pub gate: String,
    pub attempts: Vec<String>,
    /// Resident bytes per stored format.
    pub formats: BTreeMap<String, u64>,
    pub tensors: usize,
    pub carried: usize,
    pub lossy: usize,
    pub cache_hit: bool,
}

impl ReencodeReport {
    pub fn from_records(&mut self, records: &[Record]) {
        self.formats.clear();
        for r in records {
            *self.formats.entry(r.stored.name().to_string()).or_default() += r.bytes;
        }
        self.tensors = records.len();
        self.carried = records.iter().filter(|r| r.carried).count();
        self.lossy = records.iter().filter(|r| r.lossy).count();
    }
}

#[derive(Serialize, Deserialize)]
struct TensorIndex {
    format: Q,
    offset: u64,
    bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct Index {
    codec_version: u32,
    key: String,
    report: ReencodeReport,
    tensors: BTreeMap<String, TensorIndex>,
}

const PAGE: u64 = 16384;

pub fn cache_paths(dir: &Path, key: u64) -> (PathBuf, PathBuf) {
    let d = dir.join("reencoded");
    (
        d.join(format!("{key:016x}.bin")),
        d.join(format!("{key:016x}.json")),
    )
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// `fnv1a(path, size, mtime, policy, CODEC_VERSION)` (plus the KL override, which changes the outcome).
pub fn cache_key(weights: &Path, policy: ReencodePolicy, max_kl: Option<f64>) -> Option<u64> {
    let m = weights.metadata().ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let s = format!(
        "{}|{}|{mtime}|{}|{CODEC_VERSION}|{max_kl:?}",
        weights.display(),
        m.len(),
        policy.name()
    );
    Some(fnv1a(s.as_bytes()))
}

struct MapSlice {
    map: Arc<memmap2::Mmap>,
    off: usize,
    len: usize,
}

impl AsRef<[u8]> for MapSlice {
    fn as_ref(&self) -> &[u8] {
        &self.map[self.off..self.off + self.len]
    }
}

pub struct CacheReader {
    map: Arc<memmap2::Mmap>,
    index: Index,
}

impl CacheReader {
    /// The entry for `key`; a missing `.bin`, unreadable or mismatched index, or a blob outside the
    /// file deletes both files (the caller re-encodes) and returns `None`.
    pub fn open(dir: &Path, key: u64) -> Option<Self> {
        let (bin, json) = cache_paths(dir, key);
        if !json.exists() && !bin.exists() {
            return None;
        }
        let read = || -> Option<Self> {
            let index: Index = serde_json::from_slice(&std::fs::read(&json).ok()?).ok()?;
            if index.codec_version != CODEC_VERSION || index.key != format!("{key:016x}") {
                return None;
            }
            let file = std::fs::File::open(&bin).ok()?;
            // SAFETY: read-only mapping of a file this module only ever replaces by rename.
            let map = Arc::new(unsafe { memmap2::Mmap::map(&file) }.ok()?);
            let ok = index
                .tensors
                .values()
                .all(|t| (t.offset + t.bytes) as usize <= map.len());
            ok.then_some(Self { map, index })
        };
        let r = read();
        if r.is_none() {
            let _ = std::fs::remove_file(&bin);
            let _ = std::fs::remove_file(&json);
        }
        r
    }

    pub fn report(&self) -> ReencodeReport {
        ReencodeReport {
            cache_hit: true,
            ..self.index.report.clone()
        }
    }

    /// The stored weight of `name`, zero-copy out of the mapping (f32 tensors are copied).
    pub fn weight(&self, name: &str, in_f: usize, out_f: usize) -> Option<QuantizedWeight> {
        let t = self.index.tensors.get(name)?;
        let (off, len) = (t.offset as usize, t.bytes as usize);
        if t.format == Q::FP32 {
            let data = self.map[off..off + len]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>();
            return (data.len() == in_f * out_f).then_some(QuantizedWeight::Uncompressed {
                data,
                in_features: in_f,
                out_features: out_f,
            });
        }
        let bytes = Bytes::from_owner(MapSlice {
            map: self.map.clone(),
            off,
            len,
        });
        (t.format.tensor_bytes(in_f * out_f) == len)
            .then(|| gguf_weight(t.format, bytes, in_f, out_f))
    }
}

/// Writes the weights of `model` and `report` as cache entry `key` under `dir`: page-aligned blobs in
/// `<key>.bin`, an index in `<key>.json`, each written to a `.tmp` file and renamed (index last).
pub fn write_cache(
    dir: &Path,
    key: u64,
    model: &GraphLm,
    report: &ReencodeReport,
) -> Result<(), String> {
    let (bin, json) = cache_paths(dir, key);
    let d = bin.parent().expect("cache file has a parent");
    std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    let (bin_tmp, json_tmp) = (
        bin.with_extension("bin.tmp"),
        json.with_extension("json.tmp"),
    );
    let mut layers: Vec<&QuantizedLinearLayer> = Vec::new();
    for l in &model.layers {
        layers.extend([&l.qkv_proj, &l.out_proj, &l.gate_up_proj, &l.down_proj]);
    }
    layers.push(&model.lm_head);
    let io = |e: std::io::Error| e.to_string();
    let mut file = std::io::BufWriter::new(std::fs::File::create(&bin_tmp).map_err(io)?);
    let (mut tensors, mut at) = (BTreeMap::new(), 0u64);
    for l in layers {
        use std::io::Write;
        let f32_bytes;
        let (format, bytes): (Q, &[u8]) = match &l.weight {
            QuantizedWeight::Uncompressed { data, .. } => {
                f32_bytes = data
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>();
                (Q::FP32, &f32_bytes)
            }
            QuantizedWeight::ZeroCopySlice { data, format, .. } => (*format, data),
            QuantizedWeight::GgufQuantized {
                raw_data, format, ..
            } => (*format, raw_data),
            _ => return Err(format!("{}: format is not cacheable", l.name)),
        };
        file.write_all(bytes).map_err(io)?;
        tensors.insert(
            l.name.clone(),
            TensorIndex {
                format,
                offset: at,
                bytes: bytes.len() as u64,
            },
        );
        let padded = (bytes.len() as u64).div_ceil(PAGE) * PAGE;
        file.write_all(&vec![0u8; (padded - bytes.len() as u64) as usize])
            .map_err(io)?;
        at += padded;
    }
    {
        use std::io::Write;
        file.flush().map_err(io)?;
    }
    drop(file);
    let index = Index {
        codec_version: CODEC_VERSION,
        key: format!("{key:016x}"),
        report: report.clone(),
        tensors,
    };
    std::fs::write(
        &json_tmp,
        serde_json::to_vec_pretty(&index).map_err(|e| e.to_string())?,
    )
    .map_err(io)?;
    std::fs::rename(&bin_tmp, &bin).map_err(io)?;
    std::fs::rename(&json_tmp, &json).map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ReencodePolicy::*;

    /// Rates in GB/s as the codec benchmark measured them on the dev machine (TQ2_0 is the slow one).
    fn rates(f: Q) -> Option<f64> {
        Some(match f {
            Q::GGUF_TQ2_0 => 110.0,
            Q::GGUF_Q4_0 => 300.0,
            Q::GGUF_Q4_K_M => 250.0,
            Q::GGUF_Q5_K_M => 240.0,
            Q::GGUF_Q6_K => 230.0,
            Q::GGUF_Q8_0 => 380.0,
            Q::BF16 | Q::FP16 => 200.0,
            _ => return None,
        })
    }

    /// Smaller is not faster: slow Q4_K (4.5/100) loses to the larger Q5_K (5.5/240); once Q4_K is fast it wins back.
    #[test]
    fn max_speed_picks_min_bytes_over_gbps_not_min_bytes() {
        let slow_q4k = |f: Q| if f == Q::GGUF_Q4_K_M { Some(100.0) } else { rates(f) };
        // Q4_K 4.5/100 = .045; Q8_0 8.5/380 = .0224; Q4_0 4.5/300 = .015 is best of all.
        assert_eq!(rule_with(MaxSpeed, Q::BF16, &slow_q4k), Target::Encode(Q::GGUF_Q4_0));
        // Without Q4_0: Q5_K (5.5/240) beats slow Q4_K; Q8_0 (8.5/380) is inside the noise margin of it.
        let no_q4_0 = |f: Q| if f == Q::GGUF_Q4_0 { None } else { slow_q4k(f) };
        assert_eq!(rule_with(MaxSpeed, Q::BF16, &no_q4_0), Target::Encode(Q::GGUF_Q5_K_M));
        let fast_q4k = |f: Q| if f == Q::GGUF_Q4_K_M { Some(400.0) } else { no_q4_0(f) };
        assert_eq!(rule_with(MaxSpeed, Q::BF16, &fast_q4k), Target::Encode(Q::GGUF_Q4_K_M));
    }

    #[test]
    fn max_speed_moves_a_slow_low_bit_source_to_the_faster_target() {
        // 2.06 bpw / 110 GB/s is slower per weight than 4.5 bpw / 300 GB/s.
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_TQ2_0, &rates),
            Target::Encode(Q::GGUF_Q4_0)
        );
        // Q8_0 (8.5 bpw / 380) takes longer per weight than Q4_0 (4.5 / 300): never "faster by bits".
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_Q8_0, &rates),
            Target::Encode(Q::GGUF_Q4_0)
        );
        // The source stays when nothing beats it by the margin (ties keep the source).
        assert_eq!(rule_with(MaxSpeed, Q::GGUF_Q4_0, &rates), Target::Native);
    }

    /// Measured on this host (E2B Q4_K_M, GB/s): a Q4_K tensor the RMSE gate bumps to Q5_K must stay Q4_K.
    #[test]
    fn a_bumped_encode_keeps_the_source_unless_it_is_still_faster() {
        let measured = |f: Q| match f {
            Q::GGUF_Q4_K_M => Some(230.0),
            Q::GGUF_Q5_K_M => Some(171.0),
            Q::GGUF_Q6_K => Some(151.0),
            Q::GGUF_Q4_0 => Some(255.0),
            _ => None,
        };
        assert!(!beats_source(Q::GGUF_Q5_K_M, Q::GGUF_Q4_K_M, &measured));
        assert!(beats_source(Q::GGUF_Q5_K_M, Q::GGUF_Q6_K, &measured));
        assert!(beats_source(Q::GGUF_Q4_0, Q::GGUF_Q6_K, &measured));
        // Unmeasured: only fewer bits count.
        assert!(!beats_source(Q::GGUF_Q5_K_M, Q::GGUF_Q4_K_M, &|_| None));
        assert!(beats_source(Q::GGUF_Q4_0, Q::GGUF_Q6_K, &|_| None));
    }

    #[test]
    fn max_speed_ignores_unmeasured_targets_and_keeps_precision() {
        let only_q8 = |f: Q| match f {
            Q::GGUF_TQ2_0 => Some(20.0),
            Q::GGUF_Q8_0 => Some(380.0),
            _ => None,
        };
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_TQ2_0, &only_q8),
            Target::Encode(Q::GGUF_Q8_0)
        );
        for src in [
            Q::GGUF_TQ2_0,
            Q::GGUF_IQ2_M,
            Q::GGUF_Q4_K_M,
            Q::GGUF_IQ4_XS,
            Q::GGUF_Q6_K,
            Q::BF16,
        ] {
            let stored = match rule_with(MaxSpeed, src, &rates) {
                Target::Native => src,
                Target::Encode(f) => f,
            };
            let floor = if src.bits_per_weight() > 4.5 {
                4.0
            } else {
                src.bits_per_weight() - 0.01
            };
            assert!(stored.bits_per_weight() >= floor, "{src:?} -> {stored:?}");
        }
        // No rates at all: the static rule.
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_TQ2_0, &|_| None),
            Target::Native
        );
        assert_eq!(
            rule_with(MaxSpeed, Q::BF16, &|_| None),
            Target::Encode(Q::GGUF_Q4_K_M)
        );
    }

    #[test]
    fn tq2_0_to_q4_0_is_bit_exact() {
        use crate::model::layers::{gguf_block_codec, quantize_f32_to_q4_0};
        let (elems, bytes, tq) = gguf_block_codec(Q::GGUF_TQ2_0).unwrap();
        let (_, _, q4) = gguf_block_codec(Q::GGUF_Q4_0).unwrap();
        let mut s = 0x9E3779B97F4A7C15u64;
        for _ in 0..64 {
            let mut blk = vec![0u8; bytes];
            for b in &mut blk[..bytes - 2] {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (s >> 33) as u8;
            }
            blk[bytes - 2..].copy_from_slice(
                &crate::model::f32_to_f16(0.0371 + (s >> 60) as f32 * 0.01).to_le_bytes(),
            );
            let mut want = vec![0.0f32; elems];
            tq(&blk, &mut want);
            let enc = quantize_f32_to_q4_0(&want);
            let mut got = vec![0.0f32; 32];
            for (i, b) in enc.chunks_exact(18).enumerate() {
                q4(b, &mut got);
                for (a, w) in got.iter().zip(&want[i * 32..]) {
                    assert_eq!(a, w, "block {i}");
                }
            }
        }
    }

    #[test]
    fn rules_follow_the_policy_table() {
        crate::model::variant::with_unmeasured_host(|| {
            // MaxSpeed: > 4.5 bpw or codec-less -> Q4_K, else native.
            for src in [
                Q::BF16,
                Q::FP16,
                Q::FP32,
                Q::GGUF_Q8_0,
                Q::GGUF_Q6_K,
                Q::GGUF_Q5_K_M,
                Q::GGUF_Q4_1,
            ] {
                assert_eq!(
                    rule(MaxSpeed, src),
                    Target::Encode(Q::GGUF_Q4_K_M),
                    "{src:?}"
                );
            }
            for src in [
                Q::GGUF_Q4_0,
                Q::GGUF_Q4_K_M,
                Q::GGUF_IQ4_XS,
                Q::GGUF_Q2_K,
                Q::GGUF_IQ2_M,
                Q::GGUF_TQ2_0,
            ] {
                assert_eq!(rule(MaxSpeed, src), Target::Native, "{src:?}");
            }
            assert_eq!(rule(MaxSpeed, Q::TurboQuant), Target::Encode(Q::GGUF_Q4_K_M));
            // LowBitNative: float -> Q8_0; codec-less -> smallest ladder format >= source bpw.
            for src in [Q::BF16, Q::FP16, Q::FP32] {
                assert_eq!(rule(LowBitNative, src), Target::Encode(Q::GGUF_Q8_0));
            }
            assert_eq!(rule(LowBitNative, Q::GGUF_Q4_K_M), Target::Native);
            assert_eq!(
                rule(LowBitNative, Q::TurboQuant),
                Target::Encode(Q::GGUF_Q4_K_M)
            );
            // BitExact: native whenever a codec exists.
            for src in [Q::BF16, Q::FP32, Q::GGUF_Q8_0, Q::GGUF_Q4_0] {
                assert_eq!(rule(BitExact, src), Target::Native);
            }
            assert_eq!(rule(BitExact, Q::TurboQuant), Target::Encode(Q::GGUF_Q4_K_M));
        });
    }

    #[test]
    fn fit_bumps_until_blocks_tile() {
        assert_eq!(fit(Q::GGUF_Q4_K_M, 512, Q::FP32), Q::GGUF_Q4_K_M);
        // 576 = 2.25 x 256 (SmolLM2 hidden): block-32 Q4_0, not an 8.5 bpw Q8_0
        assert_eq!(fit(Q::GGUF_Q4_K_M, 576, Q::FP32), Q::GGUF_Q4_0);
        assert_eq!(fit(Q::GGUF_Q5_K_M, 576, Q::FP32), Q::GGUF_Q8_0);
        assert_eq!(fit(Q::GGUF_Q4_K_M, 64, Q::FP32), Q::GGUF_Q4_0);
        assert_eq!(fit(Q::GGUF_Q4_K_M, 48, Q::FP32), Q::FP32);
        assert_eq!(bump(Q::GGUF_Q8_0), None);
    }

    fn layer(name: &str, in_f: usize, out_f: usize, seed: u32) -> QuantizedLinearLayer {
        let mut s = seed;
        let w: Vec<f32> = (0..in_f * out_f)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.2
            })
            .collect();
        QuantizedLinearLayer::from_f32(name, in_f, out_f, &w, Q::FP32).unwrap()
    }

    fn encode_with(
        mode: Mode,
        l: QuantizedLinearLayer,
        role: ProjRole,
    ) -> (QuantizedLinearLayer, Record) {
        let e = Encoder::new(mode, 3, None);
        let out = e.encode(Projection {
            role,
            layer: l,
            sources: vec![],
            head_dim: 8,
            checkpoint: None,
        });
        let rec = e.records().remove(0);
        (out, rec)
    }

    #[test]
    fn policy_encodes_and_ffn_only_keeps_attention() {
        crate::model::variant::with_unmeasured_host(|| {
            let pol = |ffn_only| Mode::Policy {
                policy: MaxSpeed,
                ffn_only,
                rmse_limit: Some(TENSOR_RMSE_LIMIT),
            };
            let (q, rec) = encode_with(pol(false), layer("a", 512, 24, 1), ProjRole::Ffn);
            assert_eq!((q.quant_format(), rec.lossy), (Q::GGUF_Q4_K_M, true));
            assert!(rec.rmse.unwrap() <= TENSOR_RMSE_LIMIT);
            // the same matrix in attention under FFN-only stays f32 (source unknown = FP32, native)
            let (q, rec) = encode_with(pol(true), layer("a", 512, 24, 1), ProjRole::Attn);
            assert_eq!((q.quant_format(), rec.lossy), (Q::FP32, false));
            // 64 columns cannot hold Q4_K blocks: block-32 Q4_0 keeps the bit width
            let (q, _) = encode_with(pol(false), layer("b", 64, 8, 2), ProjRole::Ffn);
            assert_eq!(q.quant_format(), Q::GGUF_Q4_0);
        })
    }

    #[test]
    fn per_tensor_gate_bumps_a_format_that_misses_the_limit() {
        crate::model::variant::with_unmeasured_host(|| {
            let gate = |limit| Mode::Policy {
                policy: MaxSpeed,
                ffn_only: false,
                rmse_limit: Some(limit),
            };
            let (q4, r4) = encode_with(
                gate(TENSOR_RMSE_LIMIT),
                layer("c", 256, 16, 3),
                ProjRole::Ffn,
            );
            assert_eq!(q4.quant_format(), Q::GGUF_Q4_K_M);
            assert!(
                r4.rmse.unwrap() > 0.01,
                "uniform weights quantize to ~6% at 4 bits: {:?}",
                r4.rmse
            );
            // a limit Q4_K misses sends the tensor up the ladder until it passes
            let (q, r) = encode_with(gate(0.02), layer("c", 256, 16, 3), ProjRole::Ffn);
            assert!(
                q.quant_format().bits_per_weight() > Q::GGUF_Q4_K_M.bits_per_weight(),
                "{:?}",
                q.quant_format()
            );
            assert!(r.rmse.unwrap() <= 0.02, "{:?}", r.rmse);
        })
    }

    /// Fused GQA q/k/v built from GGUF blocks: the carried bytes dequantize to exactly the f32 rows
    /// the loader would have built (k and v repeated per head), and BitExact keeps them unchanged.
    #[test]
    fn carry_copies_grouped_query_blocks() {
        crate::model::variant::with_unmeasured_host(|| {
            use crate::model::layers::{
                dequantize_gguf_blocks, quantize_k_rows, quantize_row_q4_k_ref,
            };
            use crate::model::{ModelManifest, VirtualTensor};
            let (in_f, head_dim, q_rows, kv_rows) = (256usize, 8usize, 32usize, 16usize);
            let blocks = |rows: usize, seed: u32| {
                let l = layer("t", in_f, rows, seed);
                let QuantizedWeight::Uncompressed { data, .. } = l.weight else {
                    unreachable!()
                };
                quantize_k_rows(&data, in_f, 144, quantize_row_q4_k_ref)
            };
            let (q, k, v) = (blocks(q_rows, 1), blocks(kv_rows, 2), blocks(kv_rows, 3));
            let mut manifest = ModelManifest::default();
            let mut data = Vec::new();
            for (name, bytes, rows) in [("q", &q, q_rows), ("k", &k, kv_rows), ("v", &v, kv_rows)] {
                manifest.tensors.insert(
                    name.into(),
                    VirtualTensor {
                        name: name.into(),
                        shape: vec![rows, in_f],
                        quant_format: Q::GGUF_Q4_K_M,
                        byte_offset: data.len(),
                        byte_len: bytes.len(),
                        dtype: "Q4_K".into(),
                        shard_index: 0,
                    },
                );
                data.extend_from_slice(bytes);
            }
            let f32s = |b: &[u8]| dequantize_gguf_blocks(Q::GGUF_Q4_K_M, b).unwrap();
            let expand = |b: &[u8]| -> Vec<f32> {
                f32s(b)
                    .chunks(head_dim * in_f)
                    .flat_map(|h| h.iter().chain(h).copied().collect::<Vec<_>>())
                    .collect()
            };
            let want: Vec<f32> = [f32s(&q), expand(&k), expand(&v)].concat();
            fn proj<'a>(
                m: &'a ModelManifest,
                data: &'a [u8],
                want: &[f32],
                in_f: usize,
                head_dim: usize,
            ) -> Projection<'a> {
                let l = QuantizedLinearLayer::from_f32(
                    "layer_0_qkv",
                    in_f,
                    want.len() / in_f,
                    want,
                    Q::FP32,
                )
                .unwrap();
                Projection {
                    role: ProjRole::Attn,
                    layer: l,
                    sources: vec!["q".into(), "k".into(), "v".into()],
                    head_dim,
                    checkpoint: Some((m, data)),
                }
            }
            let mk = || proj(&manifest, &data, &want, in_f, head_dim);
            let (f, bytes) = carry(&mk()).expect("same-format blocks carry");
            assert_eq!(f, Q::GGUF_Q4_K_M);
            assert_eq!(f32s(&bytes), want);

            let e = Encoder::new(
                Mode::Policy {
                    policy: BitExact,
                    ffn_only: false,
                    rmse_limit: None,
                },
                1,
                None,
            );
            let out = e.encode(mk());
            let rec = e.records().remove(0);
            assert_eq!(
                (out.quant_format(), rec.carried, rec.lossy),
                (Q::GGUF_Q4_K_M, true, false)
            );
            // MaxSpeed leaves a 4.5 bpw Q4_K source alone too
            let e = Encoder::new(
                Mode::Policy {
                    policy: MaxSpeed,
                    ffn_only: false,
                    rmse_limit: None,
                },
                1,
                None,
            );
            let _ = e.encode(mk());
            assert!(e.records()[0].carried);
            // mixed formats inside one projection cannot be carried
            let mut mixed = manifest.clone();
            mixed.tensors.get_mut("v").unwrap().quant_format = Q::GGUF_Q5_K_M;
            assert!(carry(&proj(&mixed, &data, &want, in_f, head_dim)).is_none());
        })
    }

    /// The hand-off from `carry` to `repack`: a bitsandbytes NF4 projection stays NF4 blocks (carried,
    /// bit-equal to its f32 dequant) instead of falling through to a lossy Q4_K re-encode.
    #[test]
    fn carry_hands_bnb_weights_to_repack() {
        use crate::model::layers::dequantize_gguf_blocks;
        let (f, want) = crate::model::repack::bnb::tests::build(false, 64, false, 3, 512, 9);
        let layer = QuantizedLinearLayer::from_f32("layer_0_out", 512, 3, &want, Q::FP32).unwrap();
        let e = Encoder::new(
            Mode::Policy {
                policy: BitExact,
                ffn_only: false,
                rmse_limit: None,
            },
            1,
            None,
        );
        let out = e.encode(Projection {
            role: ProjRole::Attn,
            layer,
            sources: vec!["w.weight".into()],
            head_dim: 8,
            checkpoint: Some((&f.m, &f.data[..])),
        });
        let rec = e.records().remove(0);
        assert_eq!(
            (out.quant_format(), rec.carried, rec.lossy),
            (Q::BNB_NF4_B64, true, false)
        );
        let QuantizedWeight::GgufQuantized { raw_data, .. } = &out.weight else {
            panic!()
        };
        let got = dequantize_gguf_blocks(Q::BNB_NF4_B64, raw_data).unwrap();
        assert!(got
            .iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn threaded_encode_equals_single_thread() {
        let w: Vec<f32> = (0..512 * 160)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.003)
            .collect();
        let one = encode_weight(&w, 512, 160, Q::GGUF_Q5_K_M, 1);
        let many = encode_weight(&w, 512, 160, Q::GGUF_Q5_K_M, 7);
        assert_eq!(one.to_f32(), many.to_f32());
        let (
            QuantizedWeight::GgufQuantized { raw_data: a, .. },
            QuantizedWeight::GgufQuantized { raw_data: b, .. },
        ) = (&one, &many)
        else {
            panic!()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn cache_round_trips_and_rejects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let mut model = GraphLm::new(crate::model::graph_lm::GraphLmConfig::smollm_1m());
        let enc = Encoder::new(
            Mode::Policy {
                policy: LowBitNative,
                ffn_only: false,
                rmse_limit: None,
            },
            2,
            None,
        );
        model.encode_projections(&enc);
        let mut report = ReencodeReport {
            requested: "low-bit-native".into(),
            ..Default::default()
        };
        report.from_records(&enc.records());
        write_cache(dir.path(), 7, &model, &report).unwrap();
        let (bin, json) = cache_paths(dir.path(), 7);
        assert!(bin.exists() && json.exists() && !bin.with_extension("bin.tmp").exists());

        let reader = Arc::new(CacheReader::open(dir.path(), 7).expect("cache hit"));
        assert!(reader.report().cache_hit);
        let mut again = GraphLm::new(crate::model::graph_lm::GraphLmConfig::smollm_1m());
        let cached = Encoder::new(Mode::Cache(reader), 1, None);
        again.encode_projections(&cached);
        assert!(!cached.cache_miss.load(Ordering::Relaxed));
        assert_eq!(
            again.layers[0].qkv_proj.weight.to_f32(),
            model.layers[0].qkv_proj.weight.to_f32()
        );
        assert_eq!(again.lm_head.weight.to_f32(), model.lm_head.weight.to_f32());

        // a truncated .bin deletes both files
        std::fs::write(&bin, b"short").unwrap();
        assert!(CacheReader::open(dir.path(), 7).is_none());
        assert!(!bin.exists() && !json.exists());
    }

    /// The row-run trial picks the same format and blocks as one whole-tensor encode, over a
    /// tensor spanning several runs with a short last one.
    #[test]
    fn native_trial_in_row_runs_matches_whole_tensor() {
        let (cols, src) = (256, Q::GGUF_Q8_0);
        let rows = 2 * (TRIAL_VALUES / cols) + 5;
        let mut s = 0x9e3779b97f4a7c15u64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let QuantizedWeight::GgufQuantized { raw_data: raw, .. } = encode_weight(&w, cols, rows, src, 4) else {
            panic!("Q8_0 encodes to blocks");
        };
        let cfg = crate::model::weight_format::ReencodeConfig {
            policy: MaxSpeed,
            cache_dir: None,
            ..Default::default()
        };
        // The whole-tensor procedure the row runs replace.
        let whole = || -> Option<(Q, Vec<u8>)> {
            let Target::Encode(f) = rule(cfg.policy, src) else { return None };
            let f = fit(f, cols, Q::FP32);
            let (elems, block_bytes, decode) = gguf_block_codec(src)?;
            let mut data = vec![0.0f32; rows * cols];
            for (b, o) in raw.chunks_exact(block_bytes).zip(data.chunks_exact_mut(elems)) {
                decode(b, o);
            }
            let (mut f, mut q) = (f, encode_weight(&data, cols, rows, f, 4));
            while relative_rmse(&data, &q.to_f32()) > TENSOR_RMSE_LIMIT {
                let Some(next) = bump(f).map(|n| fit(n, cols, Q::FP32)) else { break };
                (f, q) = (next, encode_weight(&data, cols, rows, next, 4));
            }
            if !beats_source(f, src, &crate::model::variant::host_gbps) {
                return None;
            }
            let QuantizedWeight::GgufQuantized { raw_data, .. } = q else { return None };
            Some((f, raw_data.to_vec()))
        };
        let want = whole();
        assert!(want.is_some(), "MaxSpeed re-stores Q8_0 here: the test compares blocks");
        assert_eq!(reencode_native(&cfg, "layers.0.mlp.up.weight", src, rows, cols, &raw), want);
    }
}
