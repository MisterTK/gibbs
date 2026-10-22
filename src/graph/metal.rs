//! Metal executor for the graph IR (macOS). One run is one command buffer (tier T1 in PLAN.md):
//! activations live in one arena buffer planned from value lifetimes, weights are uploaded once per
//! [`Session`], and a Linear whose only reader is an exact GELU or a residual Add runs that op in
//! its GEMM epilogue (tier T0). Compiled pipelines are cached in an `MTLBinaryArchive` under
//! `<cache>/metal/`; the measured host profile (MPP check, dispatch cost, GEMM timings) under
//! `<cache>/host/`. Ops, shapes, or dtypes it does not implement are errors; nothing falls back to
//! the CPU.

// objc 0.2's `msg_send!` expands a `cfg(feature = "cargo-clippy")` check.
#![allow(unexpected_cfgs)]

use super::codec;
use super::{DType, Graph, Op, Tensor, ValueId, ValueKind, Weights};
pub fn commit_and_wait(cb: &CommandBufferRef) -> Result<(), String> {
    cb.commit();
    cb.wait_until_completed();
    match cb.status() {
        metal::MTLCommandBufferStatus::Completed => Ok(()),
        status => {
            let desc = unsafe {
                let err: *mut metal::objc::runtime::Object = metal::objc::msg_send![cb, error];
                if !err.is_null() {
                    let d: *mut metal::objc::runtime::Object =
                        metal::objc::msg_send![err, localizedDescription];
                    if !d.is_null() {
                        let s: *const std::ffi::c_char = metal::objc::msg_send![d, UTF8String];
                        if !s.is_null() {
                            std::ffi::CStr::from_ptr(s).to_string_lossy().to_string()
                        } else {
                            "no utf8".into()
                        }
                    } else {
                        "no description".into()
                    }
                } else {
                    "no error object".into()
                }
            };
            Err(format!("Metal execution failed ({status:?}): {desc}"))
        }
    }
}

pub fn new_shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, String> {
    let buf = device.new_buffer(
        bytes.max(4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    if buf.contents().is_null() {
        return Err(format!("Metal could not allocate {bytes} bytes"));
    }
    Ok(buf)
}

pub fn set_bytes<T>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const std::ffi::c_void,
    );
}

/// Compute encoder that counts `dispatch_thread_groups` calls (the only dispatch form used).
/// Everything else goes through `Deref` to the raw encoder.
pub(crate) struct Counted<'a> {
    inner: &'a ComputeCommandEncoderRef,
    dispatches: std::cell::Cell<u32>,
}

impl<'a> Counted<'a> {
    pub(crate) fn new(inner: &'a ComputeCommandEncoderRef) -> Self {
        Self {
            inner,
            dispatches: std::cell::Cell::new(0),
        }
    }
    pub(crate) fn dispatches(&self) -> u32 {
        self.dispatches.get()
    }
    pub(crate) fn dispatch_thread_groups(&self, groups: MTLSize, threads: MTLSize) {
        self.dispatches.set(self.dispatches.get() + 1);
        self.inner.dispatch_thread_groups(groups, threads);
    }
}

impl std::ops::Deref for Counted<'_> {
    type Target = ComputeCommandEncoderRef;
    fn deref(&self) -> &ComputeCommandEncoderRef {
        self.inner
    }
}

/// One `run_prepared` call (non-profiler path), for honest per-token bench rows.
#[derive(Clone, Copy, Default, Debug)]
pub struct RunSample {
    pub gpu_ns: u64,
    pub run_ns: u64,
    pub encode_ns: u64,
    pub dispatches: u32,
}

const RUN_SAMPLE_CAP: usize = 8192;
// ponytail: process-global ring; per-session if two sessions decode concurrently.
static RUN_SAMPLES: Mutex<std::collections::VecDeque<RunSample>> =
    Mutex::new(std::collections::VecDeque::new());

fn push_run_sample(s: RunSample) {
    let mut q = RUN_SAMPLES.lock();
    RUN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if q.len() >= RUN_SAMPLE_CAP {
        q.pop_front();
    }
    q.push_back(s);
}

/// Drains the run-sample ring (oldest first).
pub fn take_run_samples() -> Vec<RunSample> {
    RUN_SAMPLES.lock().drain(..).collect()
}

/// The newest run sample and how many samples were pushed before it, without draining the ring
/// (the scheduler's telemetry reads every token; a bench drains its own window).
pub fn last_run_sample() -> Option<(u64, RunSample)> {
    let q = RUN_SAMPLES.lock();
    q.back()
        .map(|s| (RUN_SEQ.load(std::sync::atomic::Ordering::Relaxed), *s))
}

static RUN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// GEMV shapes resolved by the tuner this process, and how many came from the persisted cache.
static TUNE_SHAPES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TUNE_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(cache hits, shapes)` of the GEMV tuner so far.
pub fn tune_status() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (TUNE_HITS.load(Relaxed), TUNE_SHAPES.load(Relaxed))
}

pub use crate::telemetry::KernelRow;

/// The last per-kind kernel profile (slowest kind first) and when it was taken.
static LAST_PROFILE: Mutex<Option<(Instant, Vec<KernelRow>)>> = Mutex::new(None);
/// Set by [`request_kernel_profile`]; the next decode run consumes it.
static PROFILE_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The last kernel profile, from `GIBBS_PROFILE_KERNELS=1` or a [`request_kernel_profile`] token.
pub fn last_kernel_profile() -> Option<(Instant, Vec<KernelRow>)> {
    LAST_PROFILE.lock().clone()
}

/// Profiles the next decode token (each step in its own command buffer, so that token is slow):
/// the result lands in [`last_kernel_profile`] and nothing is printed.
pub fn request_kernel_profile() {
    PROFILE_NEXT.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a one-shot kernel profile is pending (a fed stream yields the next token to it).
pub(crate) fn profile_requested() -> bool {
    PROFILE_NEXT.load(std::sync::atomic::Ordering::Relaxed)
}

/// A greedy decode stream ([`Session::submit_fed`]): each token's command buffers are queued before
/// the previous token is read back, and token ids flow between tokens on the device through `ring`.
pub struct FedStream {
    /// Slot `t % 4` (16 bytes at `16 * (t % 4)`) holds token `t`'s `[position, picked token, -, -]`;
    /// byte 64 holds the stream's first input token.
    ring: Buffer,
    state_bufs: Vec<Arc<Buffer>>,
    inflight: std::collections::VecDeque<InFlight>,
    /// Tokens submitted so far, at most `budget` (the positions the state buffers were sized for).
    pub(crate) submitted: usize,
    pub(crate) budget: usize,
    pos0: u32,
    /// The position and input token the caller's next step must pass.
    pub(crate) next_pos: u32,
    pub(crate) last: u32,
    /// When the last [`Session::wait_fed`] returned.
    last_wait: Option<Instant>,
}

impl FedStream {
    pub(crate) fn in_flight(&self) -> usize {
        self.inflight.len()
    }
}

struct InFlight {
    cbs: Vec<metal::CommandBuffer>,
    slot: usize,
    encode_ns: u64,
    dispatches: u32,
    started: Instant,
}

/// A program state's [`FedStream`]. A cloned state starts without one, so clones never share
/// in-flight work.
#[derive(Default)]
pub struct StreamSlot(pub(crate) Option<Box<FedStream>>);

impl Clone for StreamSlot {
    fn clone(&self) -> Self {
        StreamSlot(None)
    }
}

impl std::fmt::Debug for StreamSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StreamSlot")
    }
}

/// Codec micro-benchmark progress: 0 idle, 1 running, 2 done, 3 failed.
static CODEC_BENCH_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static CODEC_BENCH_DONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CODEC_BENCH_TOTAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(state, formats measured, formats to measure)` of the codec micro-benchmark; state is 0 idle,
/// 1 running, 2 done (this process ran it, or the host profile already held current rates), 3 failed.
pub fn codec_bench_status() -> (u8, usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        CODEC_BENCH_STATE.load(Relaxed),
        CODEC_BENCH_DONE.load(Relaxed),
        CODEC_BENCH_TOTAL.load(Relaxed),
    )
}

pub fn system_device() -> Result<Device, String> {
    Device::system_default()
        .ok_or_else(|| "no Metal GPU (MTLCreateSystemDefaultDevice returned nil)".to_string())
}

pub fn system_device_name() -> Result<String, String> {
    Ok(system_device()?.name().to_string())
}

pub fn system_has_unified_memory() -> Result<bool, String> {
    Ok(system_device()?.has_unified_memory())
}
use crate::hardware::{sysctl, QuantFormat};

use super::tune::{self, GemvKernel, GemvVariant};
use metal::foreign_types::ForeignType;
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    BinaryArchiveDescriptor, Buffer, CommandBufferRef, CommandQueue, CompileOptions,
    ComputeCommandEncoderRef, ComputePipelineDescriptor, ComputePipelineDescriptorRef,
    ComputePipelineState, Device, DeviceRef, Library, MTLComputePipelineState, MTLPipelineOption,
    MTLResourceOptions, MTLSize, URL,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Returns whether the Metal executor implements `op` for the given data type.
/// Formats supported for graph weights uploaded to Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum WeightFormat {
    F32,
    U32,
    Q4_0,
    Q8_0,
    BF16,
    F16,
    IQ2_M,
    IQ4_XS,
    Q4_K,
    Q2_K,
    IQ3_S,
    /// A format with no hand kernel: read through its [`codec::Codec`] by a generated GEMV.
    Gen(QuantFormat),
}

impl WeightFormat {
    /// The storage format a weight of codec/float format `f` is uploaded in.
    fn of_quant(f: QuantFormat) -> WeightFormat {
        match f {
            QuantFormat::GGUF_Q4_0 => WeightFormat::Q4_0,
            QuantFormat::GGUF_Q8_0 => WeightFormat::Q8_0,
            QuantFormat::GGUF_IQ4_XS => WeightFormat::IQ4_XS,
            QuantFormat::GGUF_IQ2_M | QuantFormat::GGUF_IQ2_S => WeightFormat::IQ2_M,
            QuantFormat::GGUF_Q4_K_M => WeightFormat::Q4_K,
            QuantFormat::GGUF_Q2_K => WeightFormat::Q2_K,
            QuantFormat::GGUF_IQ3_S => WeightFormat::IQ3_S,
            QuantFormat::BF16 => WeightFormat::BF16,
            QuantFormat::FP16 => WeightFormat::F16,
            f if codec::codec(f).is_some() => WeightFormat::Gen(f),
            _ => WeightFormat::F32,
        }
    }

    /// The codec format a generated GEMV reads this storage with (hand-kernel formats included:
    /// both read the checkpoint's blocks as stored). `None` for floats and the auxiliary-plane formats.
    fn codec_format(self) -> Option<QuantFormat> {
        Some(match self {
            WeightFormat::Q4_0 => QuantFormat::GGUF_Q4_0,
            WeightFormat::Q8_0 => QuantFormat::GGUF_Q8_0,
            WeightFormat::IQ4_XS => QuantFormat::GGUF_IQ4_XS,
            WeightFormat::IQ2_M => QuantFormat::GGUF_IQ2_M,
            WeightFormat::Q4_K => QuantFormat::GGUF_Q4_K_M,
            WeightFormat::Q2_K => QuantFormat::GGUF_Q2_K,
            WeightFormat::IQ3_S => QuantFormat::GGUF_IQ3_S,
            WeightFormat::Gen(f) => f,
            _ => return None,
        })
    }

    /// Name of the hand-written GEMV kernel, when this format has one.
    fn hand_kernel(self) -> Option<&'static str> {
        Some(match self {
            WeightFormat::Q4_0 => "gemv_q4_0",
            WeightFormat::Q8_0 => "gemv_q8_0",
            WeightFormat::IQ4_XS => "gemv_iq4_xs",
            WeightFormat::IQ2_M => "gemv_iq2_m",
            WeightFormat::Q4_K => "gemv_q4_k",
            WeightFormat::Q2_K => "gemv_q2_k",
            WeightFormat::IQ3_S => "gemv_iq3_s",
            WeightFormat::BF16 => "gemv_bf16",
            WeightFormat::F16 => "gemv_f16",
            _ => return None,
        })
    }

    /// Today's kernel for this format: the hand one where it exists (except `IQ4_XS`, `IQ2_M`, and
    /// `IQ3_S`, whose `Wide` codecs are faster at 2 rows), else generated at 2 rows.
    fn default_variant(self) -> GemvVariant {
        if self.hand_kernel().is_some()
            && !matches!(
                self,
                WeightFormat::IQ4_XS | WeightFormat::IQ2_M | WeightFormat::IQ3_S
            )
        {
            GemvVariant::HAND
        } else {
            GemvVariant::generated(2)
        }
    }

    /// Every kernel variant that can read this storage format.
    fn gemv_variants(self) -> Vec<GemvVariant> {
        let generated = self
            .codec_format()
            .is_some_and(|f| crate::graph::codec::codec(f).is_some());
        let mut v = Vec::new();
        if self.hand_kernel().is_some() {
            v.push(GemvVariant::HAND);
        }
        if generated {
            v.extend(tune::GEN_ROWS.map(GemvVariant::generated));
        }
        v
    }

    /// Whether `m >= 16` packed linears of this storage run as a batched GEMM (the hand kernels, or the
    /// codec expansion plus the f32 GEMM) instead of one GEMV per row.
    fn batched_gemm(self) -> bool {
        matches!(
            self,
            WeightFormat::BF16 | WeightFormat::F16 | WeightFormat::Q4_0 | WeightFormat::Q8_0
        ) || self
            .codec_format()
            .is_some_and(|f| crate::graph::codec::codec(f).is_some())
    }
}

/// Returns whether the Metal executor implements `op` for the given data type.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
    if ext::supports(op, dtype) {
        return true;
    }
    match op {
        Op::Linear => match dtype {
            DType::F32 | DType::F16 | DType::BF16 => true,
            DType::Quant(f) => crate::graph::codec::codec(f).is_some(),
            DType::U32 => false,
        },
        Op::Embedding => matches!(dtype, DType::F32 | DType::U32),
        Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::SwiGlu
        | Op::GeGlu
        | Op::Attention { .. }
        | Op::CachedAttention { .. }
        | Op::PagedCachedAttention { .. }
        | Op::Rope { .. }
        | Op::Slice { .. }
        | Op::CausalConv1dCached { .. }
        | Op::DeltaRule { .. }
        | Op::Softmax { .. }
        | Op::MeanPool
        | Op::Conv2d { .. }
        | Op::RelPosBias { .. }
        | Op::StateReorder
        | Op::L2Normalize => dtype == DType::F32,
        Op::Megakernel(_) => {
            matches!(
                dtype,
                DType::F32
                    | DType::F16
                    | DType::BF16
                    | DType::Quant(QuantFormat::GGUF_Q4_0)
                    | DType::Quant(QuantFormat::GGUF_Q8_0)
                    | DType::Quant(QuantFormat::GGUF_IQ4_XS)
                    | DType::Quant(QuantFormat::GGUF_IQ2_M)
                    | DType::Quant(QuantFormat::GGUF_IQ2_S)
                    | DType::Quant(QuantFormat::GGUF_Q4_K_M)
                    | DType::Quant(QuantFormat::GGUF_Q2_K)
                    | DType::Quant(QuantFormat::GGUF_IQ3_S)
            )
        }
        _ => false,
    }
}

mod ext;
mod fuse;
#[cfg(test)]
mod testutil;

const BASE_MSL: &str = include_str!("metal_kernels.metal");
/// Core kernels plus every lowering group's (`ext`).
static KERNELS_MSL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{BASE_MSL}\n{}", ext::msl()));
/// `MTLLanguageVersion4_0`; the metal crate's enum stops at 3.x.
const MSL_4_0: u64 = 0x40000;
const ARENA_ALIGN: usize = 256;
/// Largest finite f16. K/V values beyond it saturate (kernels and host uploads alike) rather than become inf.
const F16_MAX: f32 = 65504.0;

/// Whether attention KV caches default to f16 storage: only with `GIBBS_KV_DTYPE=f16` (opt-in). The
/// f16 kernels accumulate and softmax in f32 and saturate stores at +-65504, but decode with them was
/// 0.5-24% slower per model (ledger row s3-abba-kv-f16; M8 at 4k context worst), so f32 is the default.
pub(crate) fn kv_f16_default() -> bool {
    match std::env::var("GIBBS_KV_DTYPE").as_deref() {
        Ok("f16") => true,
        Ok("f32") | Err(_) => false,
        Ok(other) => {
            eprintln!("GIBBS_KV_DTYPE={other:?} is not f32 or f16; using f32");
            false
        }
    }
}
/// Threads per threadgroup: element-wise and row kernels, and the 4-simdgroup GEMM/attention kernels.
const TG: u64 = 256;
const TG4: u64 = 128;
/// Linear shapes `[m, n, k]` timed by the host profile: few and many rows, square, wide, and tall.
// ponytail: shapes are System 1's (hidden 384, FFN 1536); add the LLM/SD shape classes with P3/P5.
const GEMM_BENCH_SHAPES: [[usize; 3]; 6] = [
    [16, 384, 384],
    [16, 1536, 384],
    [16, 384, 1536],
    [256, 384, 384],
    [256, 1536, 384],
    [256, 384, 1536],
];
const ATTN_MAX_HEAD_DIM: usize = 512;
/// Decode attention splits its keys across threadgroups from this many visible keys, about
/// `ATTN_SPLIT_KEYS` per slice and at most `ATTN_MAX_SPLITS` slices.
const ATTN_SPLIT_MIN_KEYS: usize = 48;
const ATTN_SPLIT_KEYS: usize = 32;
const ATTN_MAX_SPLITS: usize = 32;
const ATTN_TARGET_TGS: usize = 64;
/// Query heads a split-attention threadgroup owns (`ATTN_HPT` in the kernel source).
const ATTN_HPT: usize = 4;
const BASE_KERNELS: [&str; 80] = [
    "embedding_f32",
    "layer_norm_f32",
    "mean_pool_f32",
    "l2_normalize_f32",
    "noop",
    "memory_sweep",
    "memory_sweep_ulong4",
    "memory_sweep_grid_stride",
    "gemm_sg_none",
    "gemm_sg_gelu",
    "gemm_sg_residual",
    "gemm_rows_none",
    "gemm_rows_gelu",
    "gemm_rows_residual",
    "rmsnorm_f32",
    "rmsnorm_residual_f32",
    "rmsnorm_fused_f32",
    "graph_rope_f32",
    "rmsnorm_batch_f32",
    "rope_batch_f32",
    "swiglu_f32",
    "geglu_f32",
    "split_glu_f32",
    "moe_weighted_sum_f32",
    "gemv_q4_0",
    "gemv_q4_0_half",
    "gemv_q8_0",
    "gemv_bf16",
    "gemv_f16",
    "gemv_tq2_0",
    "gemv_iq4_xs",
    "gemv_iq2_m",
    "gemv_q4_k",
    "gemv_q2_k",
    "gemv_iq3_s",
    "gemv_q4_0_half_pair",
    "gemv_q8_0_pair",
    "gemv_q4_k_pair",
    "gemv_bf16_pair",
    "gemm_batched_bf16_none",
    "gemm_batched_bf16_gelu",
    "gemm_batched_bf16_residual",
    "gemm_batched_f16_none",
    "gemm_batched_f16_gelu",
    "gemm_batched_f16_residual",
    "gemm_batched_q4_0_none",
    "gemm_batched_q4_0_gelu",
    "gemm_batched_q4_0_residual",
    "gemm_batched_q8_0_none",
    "gemm_batched_q8_0_gelu",
    "gemm_batched_q8_0_residual",
    "rope_general_f32",
    "kv_append_batched",
    "kv_append_batched_f16",
    "attn_cached_f32",
    "flash_attn_cached_f32_1",
    "flash_attn_cached_f32_2",
    "flash_attn_cached_f32_4",
    "flash_attn_cached_f32_8",
    "flash_attn_cached_f32",
    "flash_attn_cached_f16_1",
    "flash_attn_cached_f16_2",
    "flash_attn_cached_f16_4",
    "flash_attn_cached_f16_8",
    "flash_attn_cached_f16",
    "attn_cached_gqa_split_f32_1",
    "attn_cached_gqa_split_f32_2",
    "attn_cached_gqa_split_f32_4",
    "attn_cached_gqa_split_f32_8",
    "attn_cached_gqa_split_f32_16",
    "attn_cached_gqa_split_f16_1",
    "attn_cached_gqa_split_f16_2",
    "attn_cached_gqa_split_f16_4",
    "attn_cached_gqa_split_f16_8",
    "attn_cached_gqa_split_f16_16",
    "attn_cached_merge_f32",
    "paged_attn_cached_f32",
    "causal_conv1d_f32",
    "delta_rule_f32",
    "gemv_nvfp4",
];
const MPP_KERNELS: [&str; 3] = ["gemm_mpp_none", "gemm_mpp_gelu", "gemm_mpp_residual"];

/// GEMM implementation for `Op::Linear`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GemmKernel {
    /// MetalPerformancePrimitives `matmul2d` (MSL 4.0; the GPU's matrix units on M5).
    Mpp,
    /// `simdgroup_matrix` 8×8 tiles; every Apple GPU.
    Simdgroup,
    /// One simdgroup per output column; for few rows.
    Rows,
}

impl GemmKernel {
    const ALL: [GemmKernel; 3] = [GemmKernel::Mpp, GemmKernel::Simdgroup, GemmKernel::Rows];

    /// Pipeline name prefix, threadgroups for an `[m, n]` output, and threads per threadgroup.
    fn spec(self, m: usize, n: usize) -> (&'static str, MTLSize, u64) {
        let tiles =
            |tm: u64, tn: u64| MTLSize::new((n as u64).div_ceil(tn), (m as u64).div_ceil(tm), 1);
        match self {
            GemmKernel::Mpp => ("gemm_mpp", tiles(64, 32), TG4),
            GemmKernel::Simdgroup => ("gemm_sg", tiles(32, 32), TG4),
            GemmKernel::Rows => ("gemm_rows", MTLSize::new((n as u64).div_ceil(8), 1, 1), TG),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epi {
    None,
    Gelu,
    Residual,
    Silu,
    SwiGlu,
    GeGlu,
    UpSwiGlu,
    UpGeGlu,
}

impl Epi {
    #[inline]
    pub fn as_u32(self) -> u32 {
        match self {
            Epi::None => 0,
            Epi::Gelu => 1,
            Epi::Residual => 2,
            Epi::Silu => 3,
            Epi::SwiGlu => 4,
            Epi::GeGlu => 5,
            Epi::UpSwiGlu => 6,
            Epi::UpGeGlu => 7,
        }
    }
}

/// Measured once per (GPU, OS build, kernel source) and cached as JSON under `<cache>/host/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostProfile {
    pub device: String,
    pub os_build: String,
    /// fnv1a of the kernel source the timings were measured with; edits re-measure.
    pub kernels: u64,
    /// Measured streaming read throughput on the host GPU in GB/s.
    #[serde(default)]
    pub memory_bandwidth_gbps: f64,
    #[serde(default)]
    pub stream_gbps: f64,
    #[serde(default)]
    pub ceiling_gbps: f64,
    /// `Ok(max abs error)` when MPP `matmul2d` compiled and matched `nn::linear` on this GPU
    /// (edge tiles included); otherwise why it is not used.
    pub mpp: Result<f32, String>,
    /// GPU time of one dispatch boundary (back-to-back empty kernels), µs.
    pub dispatch_us: f64,
    /// GPU time per Linear at representative shapes; the executor picks the fastest kernel for
    /// the nearest shape.
    pub gemm: Vec<GemmTiming>,
    /// Per-codec GEMV efficiency (`GB/s / ceiling`) keyed by the codec source hash; filled by the
    /// background [`Runtime::benchmark_codecs`] run, absent until it finishes.
    #[serde(default)]
    pub codec: Option<crate::model::variant::CodecEff>,
}

/// `(rows, cols)` of the GEMVs [`Runtime::benchmark_codecs`] times (sprint-2 S3.2).
const CODEC_BENCH_SHAPES: [(usize, usize); 4] =
    [(4096, 4096), (14336, 4096), (4096, 14336), (262144, 1536)];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmTiming {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// µs per kernel available on this host.
    pub us: Vec<(GemmKernel, f64)>,
}

impl HostProfile {
    /// The faster kernel at the timed shape nearest to `[m, n, k]` (log-scale distance per dim).
    pub fn gemm_for(&self, [m, n, k]: [usize; 3]) -> GemmKernel {
        let d = |a: usize, b: usize| ((a.max(1) as f64).ln() - (b.max(1) as f64).ln()).abs();
        let t = self.gemm.iter().min_by(|a, b| {
            let (da, db) = (
                d(a.m, m) + d(a.n, n) + d(a.k, k),
                d(b.m, m) + d(b.n, n) + d(b.k, k),
            );
            da.total_cmp(&db)
        });
        t.and_then(|t| t.us.iter().min_by(|a, b| a.1.total_cmp(&b.1)))
            .map_or(GemmKernel::Simdgroup, |&(kind, _)| kind)
    }

    /// Calculate roofline generation rate (tokens/second) for a model with resident size in bytes.
    pub fn roofline_tok_s(&self, model_size_bytes: u64) -> f64 {
        if model_size_bytes == 0 || self.memory_bandwidth_gbps <= 0.0 {
            return 0.0;
        }
        let size_gb = model_size_bytes as f64 / 1e9;
        self.memory_bandwidth_gbps / size_gb
    }
}

/// Compiled kernels and the host profile for the system GPU; shared by every [`Session`].
pub struct Runtime {
    device: Device,
    queue: CommandQueue,
    psos: HashMap<&'static str, ComputePipelineState>,
    profile: HostProfile,
    /// Whether this load's pipelines came from the binary archive without compiling.
    pub archive_hit: bool,
    /// One-line load summary for logs.
    pub report: String,
    /// Where `profile` is persisted.
    profile_path: std::path::PathBuf,
    /// Generated GEMV pipelines by (codec format, rows per simdgroup, simdgroups), compiled on first use.
    gen_psos: Mutex<HashMap<(QuantFormat, u8, u8), ComputePipelineState>>,
    /// Measured GEMV variant winners ([`tune`]) and where they persist.
    tune: Mutex<tune::TuneCache>,
    tune_path: std::path::PathBuf,
}

impl Runtime {
    /// Compiles the kernels (MSL 4.0 + MPP when available), restores or records the binary
    /// archive, and loads or measures the host profile, all under `cache`.
    pub fn new(cache: &Path) -> Result<Arc<Self>, String> {
        let t0 = Instant::now();
        let device = system_device()?;
        let (lib, msl4) = compile(&device)?;
        let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
        let os_build = sysctl::string(c"kern.osversion").unwrap_or_else(|| "unknown-os".into());
        let tag: String = format!("{}-{os_build}", device.name())
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let mut names = BASE_KERNELS.to_vec();
        names.extend(ext::kernel_names());
        if msl4.is_ok() {
            names.extend(MPP_KERNELS);
        }
        let src_hash = fnv1a(KERNELS_MSL.as_bytes());
        let archive = cache.join("metal").join(format!(
            "{tag}-{}-{src_hash:016x}.bin",
            if msl4.is_ok() { "mpp" } else { "base" },
        ));
        let t1 = Instant::now();
        let (psos, archive_hit, archive_note) = pipelines(&device, &lib, &names, &archive)?;
        if !archive_hit {
            // Archives and profiles of other kernel sources: ABBA runs alternate two binaries, so
            // keep the newest few instead of deleting what the other binary will need.
            prune_cache(&cache.join("metal"), &tag, &archive, 8);
        }
        let pso_ms = t1.elapsed().as_secs_f64() * 1e3;
        for (name, pso) in &psos {
            if pso.thread_execution_width() != 32 || pso.max_total_threads_per_threadgroup() < TG {
                return Err(format!(
                    "Metal pipeline '{name}' needs SIMD width 32 and >= {TG} threads per threadgroup (got {} / {})",
                    pso.thread_execution_width(),
                    pso.max_total_threads_per_threadgroup()
                ));
            }
        }
        let mut rt = Runtime {
            queue: device.new_command_queue(),
            device,
            psos,
            profile: HostProfile {
                device: String::new(),
                os_build: os_build.clone(),
                kernels: src_hash,
                memory_bandwidth_gbps: 0.0,
                stream_gbps: 0.0,
                ceiling_gbps: 0.0,
                mpp: msl4.clone().map(|_| 0.0),
                dispatch_us: 0.0,
                gemm: Vec::new(),
                codec: None,
            },
            archive_hit,
            report: String::new(),
            profile_path: std::path::PathBuf::new(),
            gen_psos: Mutex::new(HashMap::new()),
            tune: Mutex::new(tune::TuneCache::default()),
            tune_path: std::path::PathBuf::new(),
        };
        let profile_hash = src_hash ^ codec::source_hash().rotate_left(13);
        let profile_path = cache
            .join("host")
            .join(format!("{tag}-{profile_hash:016x}.json"));
        prune_cache(&cache.join("host"), &tag, &profile_path, 8);
        let cached = std::fs::read(&profile_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<HostProfile>(&b).ok())
            .filter(|p| {
                p.os_build == os_build
                    && p.kernels == src_hash
                    && p.device == rt.device.name()
                    && p.memory_bandwidth_gbps > 0.0
                    && p.gemm.iter().map(|t| [t.m, t.n, t.k]).eq(GEMM_BENCH_SHAPES)
            });
        let profile_note = match cached {
            Some(p) => {
                rt.profile = p;
                format!("host profile from {}", profile_path.display())
            }
            None => {
                let busy = cfg!(not(test)) && tune::foreign_busy();
                rt.profile = rt.measure(msl4)?;
                let ch = codec::source_hash();
                if let Ok(entries) = std::fs::read_dir(cache.join("host")) {
                    let prefix = format!("{tag}-");
                    let mut best = None;
                    for e in entries.flatten() {
                        let name = e.file_name();
                        let Some(s) = name.to_str() else { continue };
                        if !s.starts_with(&prefix) || !s.ends_with(".json") {
                            continue;
                        }
                        let mtime = e
                            .metadata()
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                        let matched = std::fs::read(e.path())
                            .ok()
                            .and_then(|b| serde_json::from_slice::<HostProfile>(&b).ok())
                            .filter(|prev| {
                                prev.os_build == os_build
                                    && prev.memory_bandwidth_gbps > 0.0
                                    && prev.stream_gbps > 0.0
                                    && prev.ceiling_gbps > 0.0
                            })
                            .and_then(|prev| {
                                let c = prev.codec.filter(|c| c.codec_hash == ch)?;
                                Some((
                                    prev.memory_bandwidth_gbps,
                                    prev.stream_gbps,
                                    prev.ceiling_gbps,
                                    c,
                                ))
                            });
                        if let Some(vals) = matched {
                            let key = (mtime, s.to_owned());
                            if best.as_ref().is_none_or(|(k, _)| key > *k) {
                                best = Some((key, vals));
                            }
                        }
                    }
                    if let Some((_, (bw, stream, ceil, c))) = best {
                        rt.profile.memory_bandwidth_gbps = bw;
                        rt.profile.stream_gbps = stream;
                        rt.profile.ceiling_gbps = ceil;
                        rt.profile.codec = Some(c);
                    }
                }
                if busy {
                    format!(
                        "host profile measured under foreign GPU load; not saved to {}",
                        profile_path.display()
                    )
                } else {
                    let saved = profile_path
                        .parent()
                        .map_or(Ok(()), std::fs::create_dir_all)
                        .and_then(|_| {
                            std::fs::write(&profile_path, serde_json::to_vec_pretty(&rt.profile)?)
                        });
                    match saved {
                        Ok(()) => {
                            format!("host profile measured, saved {}", profile_path.display())
                        }
                        Err(e) => format!(
                            "host profile measured; saving {} failed: {e}",
                            profile_path.display()
                        ),
                    }
                }
            }
        };
        let codec_eff = rt
            .profile
            .codec
            .clone()
            .filter(|c| c.codec_hash == codec::source_hash());
        if codec_eff.is_some() || crate::model::variant::host_ceiling_gbps() == 0.0 {
            let p = &rt.profile;
            crate::model::variant::set_host_rates(p.ceiling_gbps.max(p.stream_gbps), codec_eff);
        }
        rt.profile_path = profile_path;
        rt.tune_path = cache.join("host").join(format!("tune-{tag}.json"));
        rt.tune = Mutex::new(tune::TuneCache::load(&rt.tune_path));
        let p = &rt.profile;
        let gemm: Vec<String> = p
            .gemm
            .iter()
            .map(|t| {
                let us: Vec<String> =
                    t.us.iter()
                        .map(|(kind, u)| format!("{kind:?} {u:.1}"))
                        .collect();
                format!("{}x{}x{}: {} µs", t.m, t.n, t.k, us.join(" / "))
            })
            .collect();
        rt.report = format!(
            "{}: kernels compiled in {compile_ms:.0} ms ({}); pipelines in {pso_ms:.0} ms, {archive_note}; {profile_note}: \
             streaming read {:.1} GB/s, MPP {}, dispatch {:.2} µs, GEMM {}",
            rt.device.name(),
            if p.mpp.is_ok() { "MSL 4.0 + MetalPerformancePrimitives" } else { "default MSL, no MPP" },
            p.memory_bandwidth_gbps,
            match &p.mpp {
                Ok(e) => format!("verified (max abs err {e:.1e})"),
                Err(e) => format!("unused: {e}"),
            },
            p.dispatch_us,
            gemm.join("; "),
        );
        Ok(Arc::new(rt))
    }

    /// The process-wide runtime under [`crate::hardware::default_cache_dir`]: one pipeline set, host
    /// profile and tune cache for every session (separate runtimes would each re-measure the host
    /// and could disagree about its ceiling).
    pub fn shared() -> Result<Arc<Self>, String> {
        static SHARED: std::sync::LazyLock<Result<Arc<Runtime>, String>> =
            std::sync::LazyLock::new(|| Runtime::new(&crate::hardware::default_cache_dir()?));
        SHARED.clone()
    }

    pub fn profile(&self) -> &HostProfile {
        &self.profile
    }

    fn pso(&self, name: &str) -> Result<&ComputePipelineState, String> {
        self.psos
            .get(name)
            .ok_or_else(|| format!("Metal pipeline '{name}' is not available on this host"))
    }

    /// Measures host GPU streaming read bandwidth in GB/s taking the max over 3 kernels
    /// on a 1 GiB shared buffer across 20 reps.
    pub fn benchmark_memory_bandwidth(&self) -> Result<f64, String> {
        let pso_sweep = self.pso("memory_sweep")?;
        let pso_ulong4 = self.pso("memory_sweep_ulong4")?;
        let pso_grid = self.pso("memory_sweep_grid_stride")?;

        let buf_bytes = 1024 * 1024 * 1024usize; // 1 GiB
        let in_buf = self
            .device
            .new_buffer(buf_bytes as u64, MTLResourceOptions::StorageModeShared);
        let out_buf = self
            .device
            .new_buffer(64, MTLResourceOptions::StorageModeShared);
        if in_buf.contents().is_null() || out_buf.contents().is_null() {
            return Err("Failed to allocate 1 GiB buffer for memory bandwidth benchmark".into());
        }
        unsafe {
            std::ptr::write_bytes(out_buf.contents() as *mut u8, 0, 64);
        }

        let reps = 20;
        let gb = (buf_bytes as f64) / 1e9;
        let tg_size = 256usize;
        let mut max_gbps = 0.0f64;

        // 1. memory_sweep: count = ulong2 count (16 B each), 4 ulong2 per thread (64 B/thread)
        {
            let count = (buf_bytes / 16) as u32;
            let count_buf = self.device.new_buffer_with_data(
                &count as *const u32 as *const c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let total_threads = (count / 4) as usize;
            let threadgroups = (total_threads + tg_size - 1) / tg_size;
            let us = self.timed(reps, |enc| {
                enc.set_compute_pipeline_state(pso_sweep);
                enc.set_buffer(0, Some(&in_buf), 0);
                enc.set_buffer(1, Some(&out_buf), 0);
                enc.set_buffer(2, Some(&count_buf), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(threadgroups as u64, 1, 1),
                    MTLSize::new(tg_size as u64, 1, 1),
                );
                Ok(())
            })?;
            let s = us * 1e-6;
            if s > 0.0 {
                max_gbps = max_gbps.max(gb / s);
            }
        }

        // 2. memory_sweep_ulong4: count = ulong4 count (32 B each), 2 ulong4 per thread (64 B/thread)
        {
            let count = (buf_bytes / 32) as u32;
            let count_buf = self.device.new_buffer_with_data(
                &count as *const u32 as *const c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let total_threads = (count / 2) as usize;
            let threadgroups = (total_threads + tg_size - 1) / tg_size;
            let us = self.timed(reps, |enc| {
                enc.set_compute_pipeline_state(pso_ulong4);
                enc.set_buffer(0, Some(&in_buf), 0);
                enc.set_buffer(1, Some(&out_buf), 0);
                enc.set_buffer(2, Some(&count_buf), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(threadgroups as u64, 1, 1),
                    MTLSize::new(tg_size as u64, 1, 1),
                );
                Ok(())
            })?;
            let s = us * 1e-6;
            if s > 0.0 {
                max_gbps = max_gbps.max(gb / s);
            }
        }

        // 3. memory_sweep_grid_stride: grid-stride 16 B/thread variant with 4 iterations per thread
        {
            let count = (buf_bytes / 16) as u32;
            let count_buf = self.device.new_buffer_with_data(
                &count as *const u32 as *const c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let total_threads = ((count / 4) as usize).max(tg_size);
            let threadgroups = (total_threads + tg_size - 1) / tg_size;
            let us = self.timed(reps, |enc| {
                enc.set_compute_pipeline_state(pso_grid);
                enc.set_buffer(0, Some(&in_buf), 0);
                enc.set_buffer(1, Some(&out_buf), 0);
                enc.set_buffer(2, Some(&count_buf), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(threadgroups as u64, 1, 1),
                    MTLSize::new(tg_size as u64, 1, 1),
                );
                Ok(())
            })?;
            let s = us * 1e-6;
            if s > 0.0 {
                max_gbps = max_gbps.max(gb / s);
            }
        }

        let _ = unsafe { std::ptr::read_volatile(out_buf.contents() as *const u32) };
        Ok(max_gbps)
    }

    /// Micro-benchmarks the M=1 GEMV of every format that has one (each registered codec, BF16,
    /// FP16) at the [`CODEC_BENCH_SHAPES`], in GPU time, and returns `eff = GB/s / ceiling` per
    /// format with `ceiling = max(stream bandwidth, best GEMV GB/s)`. A format that fails to
    /// build or run is logged and left out (its efficiency stays the default).
    pub fn benchmark_codecs(self: &Arc<Self>) -> Result<crate::model::variant::CodecEff, String> {
        use crate::model::variant::CodecEff;
        let mut gbps = std::collections::BTreeMap::new();
        let floats = [QuantFormat::BF16, QuantFormat::FP16];
        let formats: Vec<QuantFormat> = codec::formats().into_iter().chain(floats).collect();
        CODEC_BENCH_DONE.store(0, std::sync::atomic::Ordering::Relaxed);
        CODEC_BENCH_TOTAL.store(formats.len(), std::sync::atomic::Ordering::Relaxed);
        for f in formats {
            // Each codec's rate is its best variant's: that is the kernel a load would run.
            let variants = WeightFormat::of_quant(f).gemv_variants();
            match self.bench_gemv(f, &CODEC_BENCH_SHAPES, &variants) {
                Ok(t) => {
                    let best = t.iter().map(|x| x.1).fold(0.0f64, f64::max);
                    gbps.insert(CodecEff::key(f), best);
                }
                Err(e) => tracing::warn!("codec bench {f:?}: {e}"),
            }
            CODEC_BENCH_DONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let best = gbps.values().fold(0.0f64, |a, &b| a.max(b));
        if best <= 0.0 {
            return Err("no GEMV format could be measured".into());
        }
        let ceiling_gbps = self.profile.stream_gbps.max(best);
        let eff = gbps
            .iter()
            .map(|(k, g)| (k.clone(), g / ceiling_gbps))
            .collect();
        Ok(CodecEff {
            codec_hash: codec::source_hash(),
            ceiling_gbps,
            gbps,
            eff,
        })
    }

    /// GB/s of the M=1 GEMV in `f` for each of `variants`: weight bytes over GPU time summed across
    /// `shapes` (`(rows, cols)`). Each shape runs on enough distinct copies of the weights (one
    /// command buffer) to overflow the system cache, so the number is DRAM streaming, not cache hits.
    fn bench_gemv(
        self: &Arc<Self>,
        f: QuantFormat,
        shapes: &[(usize, usize)],
        variants: &[GemvVariant],
    ) -> Result<Vec<(GemvVariant, f64)>, String> {
        let float = matches!(f, QuantFormat::BF16 | QuantFormat::FP16);
        let (elems, block_bytes) = codec::block_geometry(f).ok_or("no codec")?;
        let mut pattern = vec![0u8; 1 << 20];
        let mut s = 0x9E3779B97F4A7C15u64;
        for b in &mut pattern {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 56) as u8;
        }
        let mut weights = Weights::new();
        let mut runs = Vec::new();
        for (si, (n, k)) in shapes.iter().copied().enumerate() {
            if k % elems != 0 {
                continue;
            }
            let bytes = n * (k / elems) * block_bytes;
            let copies = ((192usize << 20) / bytes).max(1);
            let mut raw = Vec::with_capacity(copies * bytes);
            while raw.len() < copies * bytes {
                raw.extend_from_slice(&pattern[..(copies * bytes - raw.len()).min(pattern.len())]);
            }
            let halves: Vec<u16> = if float {
                raw[..bytes]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            } else {
                Vec::new()
            };
            let map = Arc::new(crate::model::loaders::gguf::GgufData::Buf(raw));
            let mut g = Graph::new();
            let x = g.input("x", vec![1, k], DType::F32);
            for c in 0..copies {
                let name = format!("w{si}_{c}");
                let (dtype, t) = match f {
                    QuantFormat::BF16 => (DType::BF16, Tensor::BF16(halves.clone())),
                    QuantFormat::FP16 => (DType::F16, Tensor::F16(halves.clone())),
                    _ => {
                        let p = crate::model::weights::Packed {
                            map: map.clone(),
                            off: c * bytes,
                            format: f,
                            shape: vec![n, k],
                        };
                        (
                            DType::Quant(f),
                            Tensor::Mat(crate::model::weights::Mat::from_packed(p, n, k)),
                        )
                    }
                };
                let w = g.weight(&name, vec![n, k], dtype);
                let y = g.add(
                    &format!("lin{c}"),
                    Op::Linear,
                    &[x, w],
                    vec![1, n],
                    DType::F32,
                );
                g.output(y);
                weights.insert(name, t);
            }
            runs.push((g, k, copies * bytes));
        }
        if runs.is_empty() {
            return Err("no benchmark shape tiles this format's blocks".into());
        }
        let mut session = Session::new(self.clone(), &weights)?;
        let mut out = Vec::new();
        for &v in variants {
            session.set_variant_override(Some(v));
            let (mut total_bytes, mut total_s) = (0.0f64, 0.0f64);
            for (g, k, bytes) in &runs {
                let plan = session.prepare_plan(g)?;
                let x = Tensor::F32(vec![0.5; *k]);
                let mut best = f64::INFINITY;
                for rep in 0..9 {
                    session.run_prepared(g, &plan, None, vec![x.clone()])?;
                    if rep > 0 {
                        best = best.min(session.last_gpu_seconds());
                    }
                }
                total_bytes += *bytes as f64;
                total_s += best;
            }
            out.push((v, total_bytes / 1e9 / total_s));
        }
        Ok(out)
    }

    /// The generated packed-GEMV pipeline for `f` with `params`, compiled on first use.
    fn gen_pso(
        &self,
        f: QuantFormat,
        params: crate::graph::codegen::GemvParams,
    ) -> Result<ComputePipelineState, String> {
        let c = codec::codec(f).ok_or_else(|| format!("no codec for {f:?}"))?;
        let key = (c.format, params.rows_per_sg as u8, params.simdgroups as u8);
        if let Some(p) = self.gen_psos.lock().get(&key) {
            return Ok(p.clone());
        }
        let mut src = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
        src.push_str(crate::graph::codegen::gemv_common(&KERNELS_MSL));
        src.push_str(&crate::graph::codegen::gemv_packed(c, params));
        let name = crate::graph::codegen::gemv_packed_name(c, params);
        let pso = self.compile_generated(&src, &name)?;
        self.gen_psos.lock().insert(key, pso.clone());
        Ok(pso)
    }

    /// Compiles one generated kernel `name` from `src` (fast-math off, as for the hand kernels).
    fn compile_generated(&self, src: &str, name: &str) -> Result<ComputePipelineState, String> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = self
            .device
            .new_library_with_source(src, &opts)
            .map_err(|e| format!("generated kernel compilation failed: {e}"))?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| format!("generated function '{name}': {e}"))?;
        self.device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("generated pipeline '{name}': {e}"))
    }

    /// The generated dequantize-to-f32 pipeline for `f`, compiled on first use (cached under the
    /// otherwise unused `rows_per_sg == 0` slot).
    fn dequant_pso(&self, f: QuantFormat) -> Result<ComputePipelineState, String> {
        let c = codec::codec(f).ok_or_else(|| format!("no codec for {f:?}"))?;
        let key = (c.format, 0, 0);
        if let Some(p) = self.gen_psos.lock().get(&key) {
            return Ok(p.clone());
        }
        let mut src = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
        src.push_str(&crate::graph::codegen::dequant_packed(c));
        let pso = self.compile_generated(&src, &crate::graph::codegen::dequant_packed_name(c))?;
        self.gen_psos.lock().insert(key, pso.clone());
        Ok(pso)
    }

    /// Hash of the kernel sources behind every variant of `fmt`: an edit invalidates only the
    /// tune entries of the formats whose kernels it changed.
    fn variants_hash(&self, fmt: WeightFormat) -> u64 {
        use crate::graph::codegen::{gemv_common, gemv_packed};
        let mut h = fnv1a(gemv_common(&KERNELS_MSL).as_bytes());
        let mut mix =
            |s: &str| h = (h.rotate_left(5) ^ fnv1a(s.as_bytes())).wrapping_mul(0x100000001b3);
        if let Some(name) = fmt.hand_kernel() {
            mix(tune::kernel_source(&KERNELS_MSL, name));
        }
        if let Some(c) = fmt.codec_format().and_then(codec::codec) {
            for v in fmt.gemv_variants() {
                if v.kernel != GemvKernel::Hand {
                    mix(&gemv_packed(c, v.gen_params()));
                }
            }
        }
        h
    }

    /// The fastest GEMV variant of `fmt` at `rows x cols`: the persisted winner, else every variant
    /// timed on this device (the winner is persisted for the next process).
    fn tuned_variant(self: &Arc<Self>, fmt: WeightFormat, rows: usize, cols: usize) -> GemvVariant {
        let variants = fmt.gemv_variants();
        let default = fmt.default_variant();
        let Some(q) = fmt.codec_format().filter(|_| variants.len() > 1) else {
            return default;
        };
        TUNE_SHAPES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = tune::TuneCache::key(&format!("{q:?}"), rows, cols, self.variants_hash(fmt));
        let cached = self.tune.lock().get(&key);
        if let Some(t) = cached {
            TUNE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return t.variant;
        }
        // Foreign GPU load, sampled before this process times anything, or rounds that disagree /
        // drift > 5%: the winner is used here but never persisted to disk.
        let busy = tune::foreign_busy();
        let mut rounds = Vec::new();
        for round in 0..2 {
            let order: Vec<GemvVariant> = if round == 0 {
                variants.clone()
            } else {
                variants.iter().rev().copied().collect()
            };
            match self.bench_gemv(q, &[(rows, cols)], &order) {
                Ok(t) => rounds.push(t),
                Err(e) => {
                    tracing::warn!("tune {q:?} {rows}x{cols}: {e}");
                    return default;
                }
            }
        }
        let Some(best) = tune::decide(&rounds, default, busy) else {
            return default;
        };
        let consistent = !busy && tune::rounds_consistent(&rounds, best);
        let gbps = rounds
            .iter()
            .filter_map(|r| r.iter().find(|t| t.0 == best).map(|t| t.1))
            .sum::<f64>()
            / rounds.len() as f64;
        let bytes = weight_format_bytes(fmt, rows * cols) as f64;
        tracing::info!(
            "tune: {q:?} {rows}x{cols}: {} {gbps:.0} GB/s ({} variants)",
            best.label(),
            variants.len()
        );
        let tuned = tune::Tuned {
            variant: best,
            us: bytes / (gbps * 1e9) * 1e6,
            gbps,
        };
        let mut cache = self.tune.lock();
        if consistent {
            cache.insert(key, tuned);
            cache.save(&self.tune_path);
        } else {
            tracing::warn!(
                "tune {q:?} {rows}x{cols}: GPU busy or rounds inconsistent, winner not persisted"
            );
            cache.insert_contended(key, tuned);
        }
        best
    }

    /// Starts [`Self::benchmark_codecs`] on a background thread unless this host profile already
    /// holds results for the current codec sources, then stores them in the profile JSON and
    /// publishes them to the variant predictor. Never blocks the caller; once per process.
    pub fn spawn_codec_bench(self: &Arc<Self>) {
        static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let current = self
            .profile
            .codec
            .as_ref()
            .is_some_and(|c| c.codec_hash == codec::source_hash());
        use std::sync::atomic::Ordering::Relaxed;
        if current {
            let n = self.profile.codec.as_ref().map_or(0, |c| c.gbps.len());
            CODEC_BENCH_DONE.store(n, Relaxed);
            CODEC_BENCH_TOTAL.store(n, Relaxed);
            CODEC_BENCH_STATE.store(2, Relaxed);
            return;
        }
        if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        CODEC_BENCH_STATE.store(1, Relaxed);
        let rt = self.clone();
        let spawned = std::thread::Builder::new()
            .name("codec-bench".into())
            .spawn(move || {
                crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
                if let Err(e) = rt.run_codec_bench() {
                    tracing::warn!("codec bench: {e}");
                }
            });
        if let Err(e) = spawned {
            CODEC_BENCH_STATE.store(3, Relaxed);
            tracing::warn!("codec bench: cannot start thread: {e}");
        }
    }
    /// True when the cached host profile has codec rates measured with the current codec sources.
    pub fn codec_rates_current(&self) -> bool {
        self.profile
            .codec
            .as_ref()
            .is_some_and(|c| c.codec_hash == codec::source_hash())
    }

    /// Runs [`Self::benchmark_codecs`], saves the profile and publishes the rates; returns the
    /// number of formats measured.
    pub fn run_codec_bench(self: &Arc<Self>) -> Result<usize, String> {
        use std::sync::atomic::Ordering::Relaxed;
        CODEC_BENCH_STATE.store(1, Relaxed);
        let run = || -> Result<usize, String> {
            let e = self.benchmark_codecs()?;
            let n = e.gbps.len();
            let mut p = self.profile.clone();
            p.ceiling_gbps = e.ceiling_gbps;
            p.codec = Some(e.clone());
            if let Err(err) = std::fs::write(
                &self.profile_path,
                serde_json::to_vec_pretty(&p).unwrap_or_default(),
            ) {
                tracing::warn!("codec bench: saving {}: {err}", self.profile_path.display());
            }
            crate::model::variant::set_host_rates(e.ceiling_gbps, Some(e));
            Ok(n)
        };
        let result = run();
        CODEC_BENCH_STATE.store(if result.is_ok() { 2 } else { 3 }, Relaxed);
        result
    }

    /// Host profile micro-benchmarks: MPP correctness vs `nn::linear`, dispatch cost, GEMM timings.
    fn measure(&self, msl4: Result<(), String>) -> Result<HostProfile, String> {
        let upload = |v: &[f32]| -> Buffer {
            self.device.new_buffer_with_data(
                v.as_ptr() as *const c_void,
                (v.len().max(1) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let rand = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let mpp = msl4.and_then(|()| {
            let (m, n, k) = (70, 33, 129);
            let (x, w, b) = (rand(m * k, 1), rand(n * k, 2), rand(n, 3));
            let want = crate::model::nn::linear(&x, m, &w, Some(&b), k, n);
            let (xb, wb, bb, ob) = (
                upload(&x),
                upload(&w),
                upload(&b),
                upload(&vec![0.0; m * n]),
            );
            self.timed(1, |enc| {
                self.encode_gemm(
                    enc,
                    GemmKernel::Mpp,
                    Epi::None,
                    (&xb, 0),
                    (&wb, 0),
                    Some((&bb, 0)),
                    None,
                    (&ob, 0),
                    [m, n, k],
                )
            })?;
            // SAFETY: `ob` holds m*n f32 and the GPU work that wrote it has completed.
            let got = unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, m * n) };
            let err = max_abs_diff(got, &want);
            if err <= 1e-3 {
                Ok(err)
            } else {
                Err(format!("matmul2d max abs error {err:e} vs nn::linear"))
            }
        });
        let gemm_time = |[m, n, k]: [usize; 3], kind: GemmKernel, reps: usize| {
            let (xb, wb, bb, ob) = (
                upload(&rand(m * k, 4)),
                upload(&rand(n * k, 5)),
                upload(&rand(n, 6)),
                upload(&vec![0.0; m * n]),
            );
            self.timed(reps, |enc| {
                self.encode_gemm(
                    enc,
                    kind,
                    Epi::None,
                    (&xb, 0),
                    (&wb, 0),
                    Some((&bb, 0)),
                    None,
                    (&ob, 0),
                    [m, n, k],
                )
            })
        };
        // ~50 ms of GPU work first so the clocks have ramped before anything is timed.
        gemm_time([256, 384, 1536], GemmKernel::Simdgroup, 100)?;
        let scratch = upload(&[0.0; 4]);
        let noop = self.pso("noop")?;
        let reps = 1000;
        let dispatch_us = self.timed(reps, |enc| {
            enc.set_compute_pipeline_state(noop);
            enc.set_buffer(0, Some(&scratch), 0);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            Ok(())
        })?;
        let mut gemm = Vec::new();
        for shape @ [m, n, k] in GEMM_BENCH_SHAPES {
            let kinds = GemmKernel::ALL
                .into_iter()
                .filter(|&k| k != GemmKernel::Mpp || mpp.is_ok());
            let us = kinds
                .map(|kind| Ok((kind, gemm_time(shape, kind, 50)?)))
                .collect::<Result<_, String>>()?;
            gemm.push(GemmTiming { m, n, k, us });
        }
        let stream_gbps = self.benchmark_memory_bandwidth().unwrap_or(0.0);
        let memory_bandwidth_gbps = stream_gbps;
        let ceiling_gbps = stream_gbps;
        Ok(HostProfile {
            device: self.device.name().to_string(),
            os_build: self.profile.os_build.clone(),
            kernels: self.profile.kernels,
            memory_bandwidth_gbps,
            stream_gbps,
            ceiling_gbps,
            mpp,
            dispatch_us,
            gemm,
            codec: None,
        })
    }

    /// Encodes `encode` `reps` times into one command buffer and returns the GPU time per
    /// repetition in µs: the fastest of 3 such buffers after one warm-up dispatch.
    fn timed(
        &self,
        reps: usize,
        encode: impl Fn(&Counted) -> Result<(), String>,
    ) -> Result<f64, String> {
        let mut best = f64::INFINITY;
        for n in [1, reps, reps, reps] {
            let per = metal::objc::rc::autoreleasepool(|| {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for _ in 0..n {
                    encode(&Counted::new(enc))?;
                }
                enc.end_encoding();
                commit_and_wait(cb)?;
                Ok::<f64, String>(gpu_seconds(cb) * 1e6 / n as f64)
            })?;
            if n == reps {
                best = best.min(per);
            }
        }
        Ok(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gemm(
        &self,
        enc: &Counted,
        kind: GemmKernel,
        epi: Epi,
        x: (&Buffer, usize),
        w: (&Buffer, usize),
        bias: Option<(&Buffer, usize)>,
        res: Option<(&Buffer, usize)>,
        out: (&Buffer, usize),
        [m, n, k]: [usize; 3],
    ) -> Result<(), String> {
        let (prefix, groups, threads) = kind.spec(m, n);
        let suffix = match epi {
            Epi::None => "none",
            Epi::Gelu => "gelu",
            Epi::Residual => "residual",
            other => {
                return Err(format!(
                    "Metal GEMM kernel '{prefix}' does not support epilogue {other:?}"
                ));
            }
        };
        enc.set_compute_pipeline_state(self.pso(&format!("{prefix}_{suffix}"))?);
        // Absent bias/residual slots are bound to `out` and never read (has_bias = 0, no residual epilogue).
        for (i, (b, off)) in [x, w, bias.unwrap_or(out), res.unwrap_or(out), out]
            .into_iter()
            .enumerate()
        {
            enc.set_buffer(i as u64, Some(b), off as u64);
        }
        set_bytes(
            enc,
            5,
            &[u32_of(m)?, u32_of(n)?, u32_of(k)?, bias.is_some() as u32],
        );
        enc.dispatch_thread_groups(groups, MTLSize::new(threads, 1, 1));
        Ok(())
    }
}

/// Graph weights on the GPU plus a reusable activation arena; runs are serialized.
pub struct Session {
    rt: Arc<Runtime>,
    /// Owners of the host memory the no-copy weight buffers wrap (a packed weight built at
    /// compile time, e.g. a fused QKV, has no other owner once compilation returns).
    _maps: Vec<Arc<crate::model::loaders::gguf::GgufData>>,
    /// Buffer, byte offset, element count, and weight storage format.
    weights: HashMap<String, (Buffer, usize, usize, WeightFormat)>,
    arena: Mutex<Option<Buffer>>,
    state_buffers: Mutex<HashMap<String, Arc<Buffer>>>,
    mlp_scratch: Mutex<Option<Buffer>>,
    /// f32 expansion of the packed weight a batched GEMM step is reading (grown to the largest).
    dequant_scratch: Mutex<Option<Buffer>>,
    /// Per-slice partials of the split decode attention (grown to the largest).
    attn_scratch: Mutex<Option<Buffer>>,
    /// Per-threadgroup (value, index) partials of the wide-row argmax (`ext::moe`).
    argmax_scratch: Mutex<Option<Buffer>>,
    /// Set while [`Self::submit_fed`] encodes: token ids come from the device, not the host.
    fed_ids: std::sync::atomic::AtomicBool,
    /// Last command buffer of the newest fed-stream token; host arena writes wait for it.
    fed_tail: Mutex<Option<metal::CommandBuffer>>,
    gemm: Option<GemmKernel>,
    /// Forces every packed GEMV that can run it onto this variant (benchmarks and tests).
    variant_override: Option<GemvVariant>,
    last_gpu_s: std::sync::atomic::AtomicU64,
    /// Store attention KV caches (states read only as `CachedAttention` k/v operands) as f16.
    kv_f16: bool,
}

impl Session {
    /// Uploads every weight once. Preserves packed formats without f32 expansion.
    pub fn new(rt: Arc<Runtime>, weights: &Weights) -> Result<Self, String> {
        Self::new_for_graph(rt, weights, None)
    }

    /// Uploads only weights referenced by `graph` (or all if `graph` is None).
    pub fn new_for_graph(
        rt: Arc<Runtime>,
        weights: &Weights,
        graph: Option<&Graph>,
    ) -> Result<Self, String> {
        let active_names: Option<std::collections::HashSet<&str>> = graph.map(|g| {
            g.values
                .iter()
                .filter(|v| v.kind == ValueKind::Weight)
                .map(|v| v.name.as_str())
                .collect()
        });
        let mut map_buffers: HashMap<*const u8, Buffer> = HashMap::new();
        let mut maps = Vec::new();
        let mut bufs = HashMap::new();
        for (name, t) in weights.iter() {
            if let Some(ref active) = active_names {
                if !active.contains(name) {
                    continue;
                }
            }
            if let Tensor::Mat(m) = t {
                if let Some(p) = m.packed() {
                    let map_ptr = p.map.as_ptr();
                    let map_len = p.map.len();
                    let num_elements = m.rows * m.cols;
                    let fmt = match WeightFormat::of_quant(p.format) {
                        WeightFormat::F32 | WeightFormat::Q2_K => None,
                        WeightFormat::Gen(f) if codec::planar(f).is_some() => None,
                        wfmt => Some(wfmt),
                    };
                    if let Some(fmt) = fmt {
                        if !map_buffers.contains_key(&map_ptr) {
                            let buf = rt.device.new_buffer_with_bytes_no_copy(
                                map_ptr as *const c_void,
                                map_len as u64,
                                MTLResourceOptions::StorageModeShared,
                                None,
                            );
                            if !buf.contents().is_null() {
                                map_buffers.insert(map_ptr, buf);
                                maps.push(p.map.clone());
                            }
                        }
                        if let Some(map_buf) = map_buffers.get(&map_ptr) {
                            bufs.insert(
                                name.to_string(),
                                (map_buf.clone(), p.off, num_elements, fmt),
                            );
                            continue;
                        }
                    }
                }
            }
            let (_f32_buf, ptr, num_elements, byte_len, fmt) = match t {
                Tensor::F32(v) => (
                    None,
                    v.as_ptr() as *const c_void,
                    v.len(),
                    v.len() * 4,
                    WeightFormat::F32,
                ),
                Tensor::U32(v) => (
                    None,
                    v.as_ptr() as *const c_void,
                    v.len(),
                    v.len() * 4,
                    WeightFormat::U32,
                ),
                Tensor::BF16(v) => (
                    None,
                    v.as_ptr() as *const c_void,
                    v.len(),
                    v.len() * 2,
                    WeightFormat::BF16,
                ),
                Tensor::F16(v) => (
                    None,
                    v.as_ptr() as *const c_void,
                    v.len(),
                    v.len() * 2,
                    WeightFormat::F16,
                ),
                Tensor::Mat(m) => {
                    if let Some(p) = m.packed() {
                        let bytes = &p.map[p.off..];
                        let num_elements = m.rows * m.cols;
                        match WeightFormat::of_quant(p.format) {
                            WeightFormat::F32 => {
                                let fmt = p.format;
                                tracing::info!(
                                    "weight {name}: {fmt:?} has no GPU codec; dequantized to f32 at load (streams f32, not {:.2} bpw)",
                                    fmt.bits_per_weight()
                                );
                                let v = m.to_f32();
                                let ptr = v.as_ptr() as *const c_void;
                                let len = v.len();
                                (Some(v), ptr, len, len * 4, WeightFormat::F32)
                            }
                            wfmt => (
                                None,
                                bytes.as_ptr() as *const c_void,
                                num_elements,
                                weight_format_bytes(wfmt, num_elements),
                                wfmt,
                            ),
                        }
                    } else {
                        let data = m.data();
                        let ptr = data.as_ptr() as *const c_void;
                        let len = data.len();
                        (None, ptr, len, len * 2, WeightFormat::BF16)
                    }
                }
            };
            // Planar codecs are stored in the device layout, never as the checkpoint's bytes (no zero-copy map).
            let planar_bytes;
            let (ptr, byte_len) = match fmt {
                WeightFormat::Gen(f) if byte_len > 0 => match (codec::planar(f), codec::codec(f)) {
                    (Some(pl), Some(c)) => {
                        // SAFETY: `ptr` addresses `byte_len` readable bytes of the source tensor.
                        let raw = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_len) };
                        planar_bytes = pl.repack(raw, c.block_bytes as usize);
                        (planar_bytes.as_ptr() as *const c_void, planar_bytes.len())
                    }
                    _ => (ptr, byte_len),
                },
                _ => (ptr, byte_len),
            };
            let buf = if byte_len == 0 {
                new_shared_buffer(&rt.device, 4)?
            } else {
                rt.device.new_buffer_with_data(
                    ptr,
                    byte_len as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            if buf.contents().is_null() {
                return Err(format!(
                    "Metal could not allocate weight '{name}' ({num_elements} elements, {byte_len} bytes)"
                ));
            }
            bufs.insert(name.to_string(), (buf, 0, num_elements, fmt));
        }
        // The default (2 rows/simdgroup) kernel of every generated format, compiled up front so the
        // first token never waits on a compiler.
        let gen_formats: std::collections::HashSet<QuantFormat> = bufs
            .values()
            .filter_map(|(_, _, _, f)| match f {
                WeightFormat::Gen(q) => Some(*q),
                _ => None,
            })
            .collect();
        for q in gen_formats {
            rt.gen_pso(q, crate::graph::codegen::GemvParams::default())?;
        }
        Ok(Self {
            rt,
            _maps: maps,
            weights: bufs,
            arena: Mutex::new(None),
            state_buffers: Mutex::new(HashMap::new()),
            mlp_scratch: Mutex::new(None),
            dequant_scratch: Mutex::new(None),
            attn_scratch: Mutex::new(None),
            argmax_scratch: Mutex::new(None),
            fed_ids: std::sync::atomic::AtomicBool::new(false),
            fed_tail: Mutex::new(None),
            gemm: None,
            variant_override: None,
            last_gpu_s: std::sync::atomic::AtomicU64::new(0),
            kv_f16: kv_f16_default(),
        })
    }
    /// GPU seconds of the last completed (non-profiled) [`Self::run`] command buffer.
    pub fn last_gpu_seconds(&self) -> f64 {
        f64::from_bits(self.last_gpu_s.load(std::sync::atomic::Ordering::Relaxed))
    }
    /// Stores attention KV caches as f16 (`true`) or f32 for plans prepared after this call;
    /// the default is f32 unless `GIBBS_KV_DTYPE=f16`.
    pub fn set_kv_f16(&mut self, on: bool) {
        self.kv_f16 = on;
    }
    /// Uses `kind` for every Linear instead of the host profile's choice.
    pub fn force_gemm(&mut self, kind: GemmKernel) -> Result<(), String> {
        self.rt.pso(&format!("{}_none", kind.spec(1, 1).0))?;
        self.gemm = Some(kind);
        Ok(())
    }

    /// Forces every packed GEMV onto `v` where the format can run it (`None`: tuned/default).
    pub(crate) fn set_variant_override(&mut self, v: Option<GemvVariant>) {
        self.variant_override = v;
    }

    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    /// Bytes a step moves for `v`: a weight in its stored format, anything else as f32 elements.
    fn value_bytes(&self, graph: &Graph, v: ValueId) -> u64 {
        let val = graph.value(v);
        let n: usize = val.shape.iter().product();
        if val.kind == ValueKind::Weight {
            if let Some((_, _, len, fmt)) = self.weights.get(&val.name) {
                return weight_format_bytes(*fmt, *len) as u64;
            }
        }
        (n * 4) as u64
    }

    /// The pipeline, rows per threadgroup and threads per threadgroup of `format`'s packed GEMV in `variant`.
    fn gemv_pipeline(
        &self,
        format: WeightFormat,
        variant: GemvVariant,
        cols: usize,
    ) -> Result<(ComputePipelineState, u64, u64), String> {
        use crate::graph::codegen::{gemv_rows_per_tg, gemv_threads};
        let generated =
            |q: QuantFormat, v: GemvVariant| -> Result<(ComputePipelineState, u64, u64), String> {
                let p = v.gen_params();
                Ok((
                    self.rt.gen_pso(q, p)?,
                    gemv_rows_per_tg(p) as u64,
                    gemv_threads(p) as u64,
                ))
            };
        match (format, variant.kernel) {
            (WeightFormat::Gen(q), GemvKernel::Hand) => generated(q, GemvVariant::generated(2)),
            (_, GemvKernel::Hand) => match format.hand_kernel() {
                Some(_) if format == WeightFormat::Q4_0 && (cols & 1023) != 0 => {
                    Ok((self.rt.pso("gemv_q4_0_half")?.clone(), 16, 256))
                }
                Some(name) => Ok((self.rt.pso(name)?.clone(), 16, 256)),
                None => Err(format!(
                    "Metal executor: unsupported packed GEMV format {format:?}"
                )),
            },
            (_, GemvKernel::Gen) => match format.codec_format() {
                Some(q) => generated(q, variant),
                None => Err(format!("no generated GEMV for {format:?}")),
            },
        }
    }
    /// Picks each m = 1 packed GEMV's kernel variant: the session override, else the tuner's
    /// winner for its `(format, rows, cols)` (`GIBBS_TUNE=off`: the format's default).
    fn assign_variants(&self, steps: &mut [Planned]) {
        let mut chosen: HashMap<(WeightFormat, usize, usize), GemvVariant> = HashMap::new();
        for s in steps.iter_mut() {
            let Step::MegakernelMlp {
                gate_name,
                up_name,
                down_name,
                hidden,
                inter,
                variants,
                ..
            } = &mut s.op
            else {
                continue;
            };
            for (v, (name, rows, cols)) in variants.iter_mut().zip([
                (&*gate_name, *inter, *hidden),
                (&*up_name, *inter, *hidden),
                (&*down_name, *hidden, *inter),
            ]) {
                let Some(f) = self
                    .weights
                    .get(name)
                    .map(|w| w.3)
                    .filter(|f| !f.gemv_variants().is_empty())
                else {
                    continue;
                };
                let allowed = f.gemv_variants();
                *v = match self.variant_override {
                    Some(o) if allowed.contains(&o) => o,
                    Some(_) => f.default_variant(),
                    None if !matches!(f, WeightFormat::Q4_0 | WeightFormat::Q4_K)
                        && tune::enabled()
                        && allowed.len() > 1 =>
                    {
                        *chosen
                            .entry((f, rows, cols))
                            .or_insert_with(|| self.rt.tuned_variant(f, rows, cols))
                    }
                    None => f.default_variant(),
                };
            }
        }
        for s in steps.iter_mut() {
            let Step::GemvPacked {
                m,
                rows,
                cols,
                format,
                variant,
                ..
            } = &mut s.op
            else {
                continue;
            };
            let allowed = format.gemv_variants();
            *variant = if let Some(o) = self.variant_override {
                if allowed.contains(&o) {
                    o
                } else {
                    format.default_variant()
                }
            } else if *m == 1 && tune::enabled() && allowed.len() > 1 {
                *chosen
                    .entry((*format, *rows, *cols))
                    .or_insert_with(|| self.rt.tuned_variant(*format, *rows, *cols))
            } else {
                format.default_variant()
            };
        }
    }

    /// Prepares the planned steps and activation arena allocation for `graph` once.
    pub fn prepare_plan(&self, graph: &Graph) -> Result<PreparedPlan, String> {
        let (steps, aliases) = plan(graph)?;
        let mut steps = batch_horizontal(steps, &aliases)?;
        if std::env::var_os("GIBBS_DUMP_PLAN").is_some() {
            for (i, s) in steps.iter().enumerate() {
                eprintln!("{}", plan_line(i, s));
            }
        }
        for s in &steps {
            for &v in &s.reads {
                let val = graph.value(v);
                if val.kind == ValueKind::Weight {
                    let (_, _, len, fmt) = self
                        .weights
                        .get(&val.name)
                        .ok_or_else(|| format!("missing weight '{}'", val.name))?;
                    let expected_fmt = match val.dtype {
                        DType::F32 => WeightFormat::F32,
                        DType::U32 => WeightFormat::U32,
                        DType::BF16 => WeightFormat::BF16,
                        DType::F16 => WeightFormat::F16,
                        DType::Quant(QuantFormat::GGUF_Q4_0) => WeightFormat::Q4_0,
                        DType::Quant(QuantFormat::GGUF_Q8_0) => WeightFormat::Q8_0,
                        DType::Quant(QuantFormat::GGUF_IQ4_XS) => WeightFormat::IQ4_XS,
                        DType::Quant(QuantFormat::GGUF_IQ2_M)
                        | DType::Quant(QuantFormat::GGUF_IQ2_S) => WeightFormat::IQ2_M,
                        DType::Quant(QuantFormat::GGUF_Q4_K_M) => WeightFormat::Q4_K,
                        DType::Quant(QuantFormat::GGUF_Q2_K) => WeightFormat::Q2_K,
                        DType::Quant(QuantFormat::GGUF_IQ3_S) => WeightFormat::IQ3_S,
                        DType::Quant(f) if crate::graph::codec::codec(f).is_some() => {
                            WeightFormat::Gen(f)
                        }
                        _ => WeightFormat::F32,
                    };
                    if *len != numel(graph, v)? || *fmt != expected_fmt {
                        return Err(format!(
                            "weight '{}' has {len} {:?} elements, graph declares {:?} {:?}",
                            val.name, fmt, val.dtype, val.shape
                        ));
                    }
                }
            }
        }
        self.assign_variants(&mut steps);
        let f16_states = if self.kv_f16 {
            assign_kv_dtypes(graph, &mut steps)
        } else {
            HashSet::new()
        };
        let (offsets, arena_size) = allocate(graph, &steps, &aliases)?;
        pair_small_hand_gemvs(
            graph,
            &mut steps,
            &aliases,
            &offsets,
            self.variant_override == Some(GemvVariant::HAND),
        );
        let mut arena = self.arena.lock();
        if arena
            .as_ref()
            .is_none_or(|b| (b.length() as usize) < arena_size)
        {
            *arena = Some(new_shared_buffer(&self.rt.device, arena_size)?);
        }
        let is_decode = graph
            .inputs
            .iter()
            .all(|&v| graph.value(v).shape.first().copied().unwrap_or(1) <= 1);
        let mut weight_bytes = 0u64;
        let mut state_reads = Vec::new();
        let mut flops_per_run = 0u64;

        for s in &steps {
            for &v in &s.reads {
                let val = graph.value(v);
                if val.kind == ValueKind::Weight {
                    if let Some(&(_, _, len, fmt)) = self.weights.get(&val.name) {
                        // Stored bytes of this tensor (not its backing buffer, which may be the
                        // whole mmap'd checkpoint).
                        let total_b = weight_format_bytes(fmt, len) as u64;
                        if matches!(s.op, Step::Embedding { .. }) {
                            let rows = val.shape.first().copied().unwrap_or(1);
                            let row_bytes = if rows > 0 { total_b / rows as u64 } else { 0 };
                            let ids_count = s
                                .reads
                                .first()
                                .map(|&iv| graph.value(iv).shape.iter().product::<usize>() as u64)
                                .unwrap_or(1);
                            weight_bytes += ids_count * row_bytes;
                        } else {
                            weight_bytes += total_b;
                        }
                    }
                }
            }

            match &s.op {
                Step::CachedAttention {
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    window,
                    kv_f16,
                    ..
                } => {
                    let eb = if *kv_f16 { 2u64 } else { 4 };
                    for (suffix, width) in [("k", *head_dim), ("v", *v_head_dim)] {
                        state_reads.push(StateRead {
                            name: format!("{}_{suffix}", s.name),
                            bytes_per_pos: (*kv_heads * width) as u64 * eb,
                            window: *window,
                            fixed_bytes: 0,
                        });
                    }
                    let nq = 1u64;
                    let nk = 1u64;
                    flops_per_run += 4 * nq * nk * (*head_dim as u64) * (*heads as u64);
                }
                Step::Linear { m, n, k, .. } => {
                    flops_per_run += 2 * (*m as u64) * (*n as u64) * (*k as u64);
                }
                Step::GemvPacked { rows, cols, .. } => {
                    flops_per_run += 2 * (*rows as u64) * (*cols as u64);
                }
                Step::GemvPackedPair { cols, legs, .. } => {
                    flops_per_run += 2 * ((legs[0].rows + legs[1].rows) as u64) * (*cols as u64);
                }
                _ => {}
            }
        }
        let mut input_views: Vec<(ValueId, usize, usize)> = graph
            .inputs
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i, 0))
            .collect();
        for (&v, &(mut root, mut off)) in &aliases {
            while let Some(&(r, o)) = aliases.get(&root) {
                (root, off) = (r, off + o);
            }
            if let Some(i) = graph.inputs.iter().position(|&x| x == root) {
                input_views.push((v, i, off));
            }
        }
        let mlp_scratch_size = steps
            .iter()
            .filter_map(|s| match s.op {
                Step::MegakernelMlp { hidden, inter, .. } => Some(hidden * 4 + 2 * inter * 4),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let mut embedding_checks = Vec::new();
        let mut attn_pos_inputs = Vec::new();
        let mut state_write_checks = Vec::new();
        let mut state_vars: Vec<(ValueId, String)> = Vec::new();
        let mut state_idx_map: HashMap<String, usize> = HashMap::new();
        for s in &steps {
            if let Step::Embedding { ids, rows, .. } = s.op {
                let pos = graph.inputs.iter().position(|&i| i == ids).ok_or_else(|| {
                    format!("node '{}': Metal embedding ids must be a graph input (range-checked on the host)", s.name)
                })?;
                embedding_checks.push((pos, rows, s.name.clone()));
            }
            // `StateWrite` positions are host-checked against the declared rows (as on the CPU).
            if let Step::Ext(ext::ExtStep::Movement(ext::movement::Step::StateWrite {
                pos0,
                m,
                cap,
                ..
            })) = s.op
            {
                let i = graph.inputs.iter().position(|&x| x == pos0).ok_or_else(|| {
                    format!("node '{}': Metal StateWrite pos0 must be a graph input (range-checked on the host)", s.name)
                })?;
                state_write_checks.push((i, m, cap, s.name.clone()));
            }
            // Positions this step writes from a graph input: the state buffers grow to hold them.
            let pos_rows = match &s.op {
                Step::CachedAttention {
                    pos: Some(p), m, ..
                } => Some((*p, *m)),
                Step::Ext(ext::ExtStep::Movement(ext::movement::Step::StateWrite {
                    pos0,
                    m,
                    ..
                })) => Some((*pos0, *m)),
                _ => None,
            };
            if let Some((p, m)) = pos_rows {
                if let Some(i) = graph.inputs.iter().position(|&x| x == p) {
                    if !attn_pos_inputs.contains(&(i, m)) {
                        attn_pos_inputs.push((i, m));
                    }
                }
            }
            for &v in &s.reads {
                if let ValueKind::State { name, .. } = &graph.value(v).kind {
                    if !state_idx_map.contains_key(name) {
                        let idx = state_vars.len();
                        state_idx_map.insert(name.clone(), idx);
                        state_vars.push((v, name.clone()));
                    }
                }
            }
        }
        let mut value_locs = Vec::with_capacity(graph.values.len());
        for (idx, val) in graph.values.iter().enumerate() {
            let v = ValueId(idx as u32);
            let loc = if let ValueKind::State { name, .. } = &val.kind {
                if let Some(&state_idx) = state_idx_map.get(name) {
                    PlanValueLoc::State(state_idx)
                } else {
                    let state_idx = state_vars.len();
                    state_idx_map.insert(name.clone(), state_idx);
                    state_vars.push((v, name.clone()));
                    PlanValueLoc::State(state_idx)
                }
            } else if let Some(&off) = offsets.get(&v) {
                PlanValueLoc::Arena(off)
            } else if let Some((buf, off, _, _)) = self.weights.get(&val.name) {
                PlanValueLoc::Weight(buf.clone(), *off)
            } else {
                PlanValueLoc::Unbound
            };
            value_locs.push(loc);
        }

        Ok(PreparedPlan {
            steps,
            offsets,
            arena_size,
            is_decode,
            weight_bytes,
            state_reads,
            flops_per_run,
            input_views,
            mlp_scratch_size,
            state_write_checks,
            embedding_checks,
            attn_pos_inputs,
            state_vars,
            value_locs,
            f16_states,
        })
    }

    /// Returns the raw pointer to the currently allocated arena buffer contents, if allocated.
    pub fn arena_buffer_ptr(&self) -> Option<*mut std::ffi::c_void> {
        self.arena.lock().as_ref().map(|b| b.contents())
    }

    /// Runs `graph` with `inputs` bound positionally to `graph.inputs`; returns `graph.outputs`.
    pub fn run(&self, graph: &Graph, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        self.run_with_state(graph, None, inputs)
    }

    /// Runs `graph` with `inputs` and persistent `state`; returns `graph.outputs`.
    pub fn run_with_state(
        &self,
        graph: &Graph,
        state: Option<&mut crate::graph::StateStore>,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        #[cfg(test)]
        crate::graph::shape::note(graph);
        let plan = self.prepare_plan(graph)?;
        self.run_prepared(graph, &plan, state, inputs)
    }

    /// The state buffers of `plan` in `state` (else the session's own), grown to cover `need_rows`
    /// positions.
    fn resolve_state_bufs(
        &self,
        graph: &Graph,
        plan: &PreparedPlan,
        mut state: Option<&mut crate::graph::StateStore>,
        need_rows: usize,
    ) -> Result<Vec<Arc<Buffer>>, String> {
        let mut state_bufs: Vec<Arc<Buffer>> = Vec::with_capacity(plan.state_vars.len());
        if let Some(st) = &mut state {
            for &(v, ref name) in &plan.state_vars {
                let n = state_len(graph, v, need_rows)?;
                let f16 = plan.f16_states.contains(name);
                let eb = if f16 { 2 } else { 4 };
                if st.metal_buffers.contains_key(name) && st.metal_f16.contains(name) != f16 {
                    return Err(format!(
                        "state '{name}' is stored as {} but this program needs {}",
                        if f16 { "f32" } else { "f16" },
                        if f16 { "f16" } else { "f32" },
                    ));
                }
                let grown = st.metal_buffers.get(name).is_some_and(|(_, old)| *old < n);
                if grown || !st.metal_buffers.contains_key(name) {
                    let buf = new_shared_buffer(&self.rt.device, (n * eb).max(ARENA_ALIGN))?;
                    if let Some((old, old_n)) = st.metal_buffers.get(name) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                old.contents() as *const u8,
                                buf.contents() as *mut u8,
                                *old_n * eb,
                            );
                        }
                    } else if let Some(Tensor::F32(cur_data)) = st.buffers.get(name) {
                        let copy_len = cur_data.len().min(n);
                        if f16 {
                            // SAFETY: the buffer holds n >= copy_len u16s.
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(buf.contents() as *mut u16, copy_len)
                            };
                            for (d, &x) in dst.iter_mut().zip(cur_data) {
                                *d = crate::model::f32_to_f16(x.clamp(-F16_MAX, F16_MAX));
                            }
                        } else {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    cur_data.as_ptr(),
                                    buf.contents() as *mut f32,
                                    copy_len,
                                );
                            }
                        }
                    }
                    if f16 {
                        st.metal_f16.insert(name.clone());
                    }
                    let arc_buf = Arc::new(buf);
                    st.metal_buffers.insert(name.clone(), (arc_buf.clone(), n));
                    state_bufs.push(arc_buf);
                } else {
                    state_bufs.push(st.metal_buffers[name].0.clone());
                }
            }
        } else {
            let mut sbufs = self.state_buffers.lock();
            for &(v, ref name) in &plan.state_vars {
                let n = state_len(graph, v, need_rows)?;
                let want =
                    (n * if plan.f16_states.contains(name) { 2 } else { 4 }).max(ARENA_ALIGN);
                let have = sbufs.get(name).map_or(0, |b| b.length() as usize);
                if have < want {
                    let buf = new_shared_buffer(&self.rt.device, want)?;
                    if let Some(old) = sbufs.get(name) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                old.contents() as *const u8,
                                buf.contents() as *mut u8,
                                have.min(want),
                            );
                        }
                    }
                    let arc_buf = Arc::new(buf);
                    sbufs.insert(name.clone(), arc_buf.clone());
                    state_bufs.push(arc_buf);
                } else {
                    state_bufs.push(sbufs[name].clone());
                }
            }
        }
        Ok(state_bufs)
    }

    /// Runs `graph` using a pre-planned execution schedule.
    pub fn run_prepared(
        &self,
        graph: &Graph,
        plan: &PreparedPlan,
        state: Option<&mut crate::graph::StateStore>,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let run_start = Instant::now();
        // A streamed token may still be running on the arena this call writes.
        if let Some(cb) = self.fed_tail.lock().take() {
            cb.wait_until_completed();
        }
        let mut sample: Option<RunSample> = None;
        if inputs.len() != graph.inputs.len() {
            return Err(format!(
                "graph has {} inputs, got {}",
                graph.inputs.len(),
                inputs.len()
            ));
        }
        for (&v, t) in graph.inputs.iter().zip(&inputs) {
            let val = graph.value(v);
            let ok_type = matches!(
                (val.dtype, t),
                (DType::F32, Tensor::F32(_)) | (DType::U32, Tensor::U32(_))
            );
            if !ok_type || numel(graph, v)? != t.len() {
                return Err(format!(
                    "input '{}' expects {:?} {:?}, got {} elements",
                    val.name,
                    val.dtype,
                    val.shape,
                    t.len()
                ));
            }
        }
        let steps = &plan.steps;
        let offsets = &plan.offsets;
        for &(pos, rows, ref name) in &plan.embedding_checks {
            let Tensor::U32(v) = &inputs[pos] else {
                unreachable!("dtype checked above")
            };
            if let Some((r, &id)) = v.iter().enumerate().find(|&(_, &id)| id as usize >= rows) {
                return Err(format!(
                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                ));
            }
        }
        for &(pos, m, cap, ref name) in &plan.state_write_checks {
            let Tensor::U32(v) = &inputs[pos] else {
                unreachable!("dtype checked above")
            };
            let p0 = v.first().map_or(0, |&p| p as usize);
            if p0 + m > cap {
                return Err(format!(
                    "node '{name}': writing rows {p0}..{} past the state's {cap} rows",
                    p0 + m
                ));
            }
        }
        // Positions this call touches: 2-D states keep at least 4096 rows and grow in 4096-row steps.
        let need_rows = plan
            .attn_pos_inputs
            .iter()
            .filter_map(|&(i, m)| match &inputs[i] {
                Tensor::U32(v) => v.first().map(|&x| x as usize + m),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let state_bufs = self.resolve_state_bufs(graph, plan, state, need_rows)?;

        let size = plan.arena_size;
        let mut arena = self.arena.lock();
        if arena.as_ref().is_none_or(|b| (b.length() as usize) < size) {
            *arena = Some(new_shared_buffer(&self.rt.device, size)?);
        }
        let arena = arena.as_ref().expect("allocated above");
        let base = arena.contents() as *mut u8;
        for (&v, t) in graph.inputs.iter().zip(&inputs) {
            let (_f32_buf, src, len) = match t {
                Tensor::F32(x) => (None, x.as_ptr() as *const u8, x.len()),
                Tensor::U32(x) => (None, x.as_ptr() as *const u8, x.len()),
                other => {
                    let v = other.to_f32();
                    let ptr = v.as_ptr() as *const u8;
                    let len = v.len();
                    (Some(v), ptr, len)
                }
            };
            // SAFETY: `allocate` reserved len*4 bytes at this offset inside the arena.
            unsafe { std::ptr::copy_nonoverlapping(src, base.add(offsets[&v]), len * 4) };
        }
        let loc = |v: ValueId| -> (&Buffer, usize) {
            match &plan.value_locs[v.0 as usize] {
                PlanValueLoc::State(idx) => (&state_bufs[*idx], 0),
                PlanValueLoc::Arena(off) => (arena, *off),
                PlanValueLoc::Weight(buf, off) => (buf, *off),
                PlanValueLoc::Unbound => {
                    let w = &self.weights[&graph.value(v).name];
                    (&w.0, w.1)
                }
            }
        };
        let mlp_scratch_buf = self.mlp_scratch_for(plan)?;
        metal::objc::rc::autoreleasepool(|| {
            let print_profile = std::env::var_os("GIBBS_PROFILE_KERNELS").is_some();
            // A requested sample is one real decode token; the printing profiler profiles every run.
            let one_shot = !print_profile
                && plan.is_decode
                && !plan.state_reads.is_empty()
                && PROFILE_NEXT.swap(false, std::sync::atomic::Ordering::Relaxed);
            if print_profile || one_shot {
                let mut timings = Vec::new();
                for s in steps {
                    // The printing profiler runs each step twice: pass 0 faults in memory-mapped
                    // weight pages and warms caches, pass 1 is timed. A requested sample runs each
                    // step once, so the token it profiles is still computed exactly once.
                    let mut gpu_s = 0.0;
                    let cold = std::env::var_os("GIBBS_PROFILE_COLD").is_some();
                    for _ in 0..if print_profile && !cold { 2 } else { 1 } {
                        let cb = self.rt.queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        self.encode(&Counted::new(enc), s, mlp_scratch_buf.as_ref(), &loc)?;
                        enc.end_encoding();
                        commit_and_wait(cb)?;
                        gpu_s = gpu_seconds(cb);
                    }
                    let bytes: u64 = s
                        .reads
                        .iter()
                        .chain(std::iter::once(&s.out))
                        .map(|&v| self.value_bytes(graph, v))
                        .sum();
                    timings.push((s.name.clone(), format!("{:?}", s.op), gpu_s, bytes));
                }
                let ceiling = self.rt.profile.ceiling_gbps.max(1.0);
                let rows = kernel_rows(&timings, ceiling);
                if print_profile {
                    print_kernel_profile(&timings, &rows, ceiling);
                }
                *LAST_PROFILE.lock() = Some((Instant::now(), rows));
            } else {
                let (cbs, dispatches, encode_dur) =
                    self.encode_chunks(steps, &loc, mlp_scratch_buf.as_ref())?;
                let last = cbs.last().expect("a plan has at least one step");
                commit_and_wait(last)?;
                for cb in &cbs[..cbs.len() - 1] {
                    if cb.status() != metal::MTLCommandBufferStatus::Completed {
                        return Err(format!(
                            "Metal command buffer ended with status {:?}",
                            cb.status()
                        ));
                    }
                }
                let span = gpu_span_seconds(&cbs[0], last);
                sample = Some(RunSample {
                    gpu_ns: (span * 1e9) as u64,
                    run_ns: 0,
                    encode_ns: encode_dur.as_nanos() as u64,
                    dispatches,
                });
                self.last_gpu_s
                    .store(span.to_bits(), std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
        })?;
        let outs = graph
            .outputs
            .iter()
            .map(|&v| {
                let (buf, off) = loc(v);
                let n = numel(graph, v)?;
                // SAFETY: the value occupies n f32 at `off` and the command buffer has completed.
                let s = unsafe {
                    std::slice::from_raw_parts(
                        (buf.contents() as *const u8).add(off) as *const f32,
                        n,
                    )
                };
                // u32 values (ids, tokens) are stored as their bits.
                Ok(match graph.value(v).dtype {
                    DType::U32 => Tensor::U32(s.iter().map(|x| x.to_bits()).collect()),
                    _ => Tensor::F32(s.to_vec()),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if let Some(mut s) = sample {
            s.run_ns = run_start.elapsed().as_nanos() as u64;
            push_run_sample(s);
        }
        Ok(outs)
    }

    /// Whether `plan` (of `graph`) can run as a [`FedStream`]: inputs exactly the `[1]` u32 token id
    /// and position, every embedding gather reading input 0, attention positions from input 1, and
    /// one output.
    pub(crate) fn fed_stream_ok(graph: &Graph, plan: &PreparedPlan) -> bool {
        let one_u32 = |v: ValueId| {
            let val = graph.value(v);
            val.dtype == DType::U32 && val.shape.iter().product::<usize>() == 1
        };
        let view_of = |v: ValueId, input: usize| {
            plan.input_views
                .iter()
                .any(|&(x, i, _)| x == v && i == input)
        };
        graph.inputs.len() == 2
            && graph.inputs.iter().all(|&v| one_u32(v))
            && graph.outputs.len() == 1
            && plan.attn_pos_inputs.iter().all(|&(i, _)| i == 1)
            && plan.steps.iter().all(|s| match &s.op {
                Step::Embedding { ids, .. }
                | Step::Ext(ext::ExtStep::Experts(ext::moe::Step::Embed { ids, .. })) => {
                    view_of(*ids, 0)
                }
                _ => true,
            })
    }

    /// A [`FedStream`] from input token `last` at position `pos`, its state buffers in `state`
    /// sized for `budget` tokens.
    pub(crate) fn new_fed_stream(
        &self,
        graph: &Graph,
        plan: &PreparedPlan,
        state: &mut crate::graph::StateStore,
        last: u32,
        pos: u32,
        budget: usize,
    ) -> Result<FedStream, String> {
        let ring = new_shared_buffer(&self.rt.device, 128)?;
        // SAFETY: a fresh 128-byte shared buffer no command buffer uses yet.
        unsafe { *(ring.contents() as *mut u32).add(16) = last };
        let state_bufs =
            self.resolve_state_bufs(graph, plan, Some(state), pos as usize + budget)?;
        Ok(FedStream {
            ring,
            state_bufs,
            inflight: std::collections::VecDeque::new(),
            submitted: 0,
            budget,
            pos0: pos,
            next_pos: pos,
            last,
            last_wait: None,
        })
    }

    /// Encodes and commits the stream's next token: its id is the previous token's device pick
    /// (the first token's from the ring), its position host-written into its ring slot; a blit
    /// copies the picked id (output 0) into that slot. Nothing waits.
    pub(crate) fn submit_fed(
        &self,
        graph: &Graph,
        plan: &PreparedPlan,
        s: &mut FedStream,
    ) -> Result<(), String> {
        let started = Instant::now();
        let t = s.submitted;
        let slot = t % 4;
        let ids_at = if t == 0 { 64 } else { 16 * ((t - 1) % 4) + 4 };
        let out = *graph
            .outputs
            .first()
            .ok_or("a fed stream needs an output")?;
        // SAFETY: slot `t % 4` last served token t - 4, which completed (at most two tokens are in
        // flight when one is submitted).
        unsafe { *(s.ring.contents() as *mut u32).add(4 * slot) = s.pos0 + t as u32 };
        let (cbs, dispatches, encode_dur) = {
            let mut arena = self.arena.lock();
            if arena
                .as_ref()
                .is_none_or(|b| (b.length() as usize) < plan.arena_size)
            {
                *arena = Some(new_shared_buffer(&self.rt.device, plan.arena_size)?);
            }
            let arena = arena.as_ref().expect("allocated above");
            let mlp_scratch_buf = self.mlp_scratch_for(plan)?;
            let (ring, state_bufs) = (&s.ring, &s.state_bufs);
            let loc = |v: ValueId| -> (&Buffer, usize) {
                if let Some(&(_, i, off)) = plan.input_views.iter().find(|x| x.0 == v) {
                    return (ring, if i == 0 { ids_at } else { 16 * slot } + off);
                }
                match &plan.value_locs[v.0 as usize] {
                    PlanValueLoc::State(idx) => (&state_bufs[*idx], 0),
                    PlanValueLoc::Arena(off) => (arena, *off),
                    PlanValueLoc::Weight(buf, off) => (buf, *off),
                    PlanValueLoc::Unbound => {
                        let w = &self.weights[&graph.value(v).name];
                        (&w.0, w.1)
                    }
                }
            };
            metal::objc::rc::autoreleasepool(|| {
                use std::sync::atomic::Ordering::Relaxed;
                self.fed_ids.store(true, Relaxed);
                let r = self.encode_chunks(&plan.steps, &loc, mlp_scratch_buf.as_ref());
                self.fed_ids.store(false, Relaxed);
                let (cbs, dispatches, encode_dur) = r?;
                let last = cbs.last().expect("encode_chunks returns a buffer");
                let (ob, oo) = loc(out);
                let blit = last.new_blit_command_encoder();
                blit.copy_from_buffer(ob, oo as u64, ring, (16 * slot + 4) as u64, 4);
                blit.end_encoding();
                last.commit();
                *self.fed_tail.lock() = Some(last.clone());
                Ok::<_, String>((cbs, dispatches, encode_dur))
            })?
        };
        s.inflight.push_back(InFlight {
            cbs,
            slot,
            encode_ns: encode_dur.as_nanos() as u64,
            dispatches,
            started,
        });
        s.submitted += 1;
        Ok(())
    }

    /// Waits for the stream's oldest in-flight token and returns its picked id.
    pub(crate) fn wait_fed(&self, s: &mut FedStream) -> Result<u32, String> {
        let f = s
            .inflight
            .pop_front()
            .ok_or("no streamed token in flight")?;
        let last = f.cbs.last().expect("a submitted token has a buffer");
        last.wait_until_completed();
        if let Some(cb) = f
            .cbs
            .iter()
            .find(|cb| cb.status() != metal::MTLCommandBufferStatus::Completed)
        {
            return Err(format!(
                "Metal command buffer ended with status {:?}",
                cb.status()
            ));
        }
        // SAFETY: the token's blit wrote this word and its command buffers completed.
        let tok = unsafe { *(s.ring.contents() as *const u32).add(4 * f.slot + 1) };
        let span = gpu_span_seconds(&f.cbs[0], last);
        self.last_gpu_s
            .store(span.to_bits(), std::sync::atomic::Ordering::Relaxed);
        let now = Instant::now();
        push_run_sample(RunSample {
            gpu_ns: (span * 1e9) as u64,
            run_ns: (now - s.last_wait.unwrap_or(f.started)).as_nanos() as u64,
            encode_ns: f.encode_ns,
            dispatches: f.dispatches,
        });
        s.last_wait = Some(now);
        Ok(tok)
    }

    /// Waits for every in-flight token of `s` (no samples, ids discarded).
    pub(crate) fn drain_fed(&self, s: &mut FedStream) {
        for f in s.inflight.drain(..) {
            if let Some(cb) = f.cbs.last() {
                cb.wait_until_completed();
            }
        }
    }

    /// The MLP megakernel scratch `plan` needs, grown to fit.
    fn mlp_scratch_for(&self, plan: &PreparedPlan) -> Result<Option<Buffer>, String> {
        if plan.mlp_scratch_size == 0 {
            return Ok(None);
        }
        let need = plan.mlp_scratch_size.max(256);
        let mut guard = self.mlp_scratch.lock();
        if guard.as_ref().is_none_or(|b| (b.length() as usize) < need) {
            *guard = Some(new_shared_buffer(&self.rt.device, need)?);
        }
        Ok(guard.clone())
    }

    /// Encodes `steps` into command buffers cut into chunks, committing every chunk but the last as
    /// soon as it is encoded; returns the buffers (the last uncommitted), dispatches and encode time.
    fn encode_chunks<'a>(
        &self,
        steps: &[Planned],
        loc: &dyn Fn(ValueId) -> (&'a Buffer, usize),
        mlp_scratch: Option<&Buffer>,
    ) -> Result<(Vec<metal::CommandBuffer>, u32, std::time::Duration), String> {
        // The GPU runs the first `CB_FIRST_STEPS` steps while the host encodes the rest (encoding costs
        // ~0.4 us per dispatch, a step on the GPU ~10 us). Buffers on one queue run in order and
        // the arena is hazard-tracked. luminal: luminal_metal/src/command_buffer.rs:21-202.
        // Plans under `CB_CHUNK_MIN_STEPS` steps stay one buffer: measured on a 147-dispatch 1B-parameter
        // decode, splitting added ~0.14 ms of GPU span (buffer boundaries) for ~0.06 ms of encoding to hide.
        let (first, chunk) = if steps.len() >= CB_CHUNK_MIN_STEPS {
            CB_CHUNKING
        } else {
            (usize::MAX, usize::MAX)
        };
        let mut cbs: Vec<metal::CommandBuffer> = Vec::new();
        let mut err = None;
        let mut dispatches = 0u32;
        let encode_start = std::time::Instant::now();
        let mut at = 0;
        while at < steps.len() && err.is_none() {
            let end = if cbs.is_empty() {
                first
            } else {
                at.saturating_add(chunk)
            }
            .min(steps.len());
            let cb = self.rt.queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder();
            let counted = Counted::new(enc);
            for s in &steps[at..end] {
                if let Err(e) = self.encode(&counted, s, mlp_scratch, &loc) {
                    err = Some(e);
                    break;
                }
            }
            enc.end_encoding();
            dispatches += counted.dispatches();
            if end < steps.len() && err.is_none() {
                cb.commit();
            }
            cbs.push(cb);
            at = end;
        }
        let encode_dur = encode_start.elapsed();
        if let Some(e) = err {
            // Earlier chunks are already on the GPU: let them finish before reporting.
            if cbs.len() > 1 {
                cbs[cbs.len() - 2].wait_until_completed();
            }
            return Err(e);
        }
        if cbs.is_empty() {
            let cb = self.rt.queue.new_command_buffer().to_owned();
            cb.new_compute_command_encoder().end_encoding();
            cbs.push(cb);
        }
        Ok((cbs, dispatches, encode_dur))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gemv_or_gemm(
        &self,
        enc: &Counted,
        x: (&Buffer, usize),
        w_loc: (&Buffer, usize),
        variant: GemvVariant,
        graph_val_name: &str,
        res: Option<(&Buffer, usize)>,
        out: (&Buffer, usize),
        rows: usize,
        cols: usize,
        epi: Epi,
    ) -> Result<(), String> {
        let format = self
            .weights
            .get(graph_val_name)
            .map(|w| w.3)
            .unwrap_or(WeightFormat::F32);
        match format {
            WeightFormat::F32 => {
                let kind = self
                    .gemm
                    .unwrap_or_else(|| self.rt.profile.gemm_for([1, rows, cols]));
                self.rt
                    .encode_gemm(enc, kind, epi, x, w_loc, None, res, out, [1, rows, cols])
            }
            WeightFormat::BF16
            | WeightFormat::F16
            | WeightFormat::Q8_0
            | WeightFormat::Q4_0
            | WeightFormat::IQ4_XS
            | WeightFormat::IQ2_M
            | WeightFormat::Q4_K
            | WeightFormat::Q2_K
            | WeightFormat::IQ3_S
            | WeightFormat::Gen(_) => {
                let (pso, rows_per_tg, threads) = self.gemv_pipeline(format, variant, cols)?;
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(w_loc.0), w_loc.1 as u64);
                enc.set_buffer(1, Some(x.0), x.1 as u64);
                enc.set_buffer(2, Some(out.0), out.1 as u64);
                #[repr(C)]
                struct PackedGemvParams(u32, u32, u32, u32, u32);
                let epi_u32 = epi.as_u32();
                let p = PackedGemvParams(u32_of(rows)?, u32_of(cols)?, 1, 0, epi_u32);
                set_bytes(enc, 3, &p);
                enc.set_buffer(4, Some(out.0), out.1 as u64); // dummy bias
                if let Some(r) = res {
                    enc.set_buffer(5, Some(r.0), r.1 as u64);
                } else {
                    enc.set_buffer(5, Some(out.0), out.1 as u64);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(rows_per_tg), 1, 1),
                    MTLSize::new(threads, 1, 1),
                );
                Ok(())
            }
            WeightFormat::U32 => Err("U32 weights not supported for GEMV".into()),
        }
    }

    fn encode<'a>(
        &self,
        enc: &Counted,
        s: &Planned,
        mlp_scratch_buf: Option<&Buffer>,
        loc: &impl Fn(ValueId) -> (&'a Buffer, usize),
    ) -> Result<(), String> {
        let rt = &self.rt;
        let bind = |i: u64, v: ValueId| {
            let (b, off) = loc(v);
            enc.set_buffer(i, Some(b), off as u64);
        };
        let grid1 = |n: usize| {
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(TG), 1, 1),
                MTLSize::new(TG, 1, 1),
            )
        };
        match s.op {
            Step::Ext(ref st) => {
                let e = ext::Enc { enc, loc: &loc };
                ext::encode(self, &e, s, st)?;
            }
            Step::Linear {
                x,
                w,
                bias,
                res,
                epi,
                m,
                n,
                k,
            } => {
                let kind = self.gemm.unwrap_or_else(|| rt.profile.gemm_for([m, n, k]));
                rt.encode_gemm(
                    enc,
                    kind,
                    epi,
                    loc(x),
                    loc(w),
                    bias.map(loc),
                    res.map(loc),
                    loc(s.out),
                    [m, n, k],
                )?;
            }
            Step::Embedding {
                table,
                ids,
                n,
                d,
                rows,
                ..
            } => {
                enc.set_compute_pipeline_state(rt.pso("embedding_f32")?);
                bind(0, table);
                bind(1, ids);
                bind(2, s.out);
                set_bytes(enc, 3, &[u32_of(n)?, u32_of(d)?, u32_of(rows)?]);
                grid1(n * d);
            }
            Step::LayerNorm {
                x,
                g,
                b,
                rows,
                d,
                eps,
            } => {
                enc.set_compute_pipeline_state(rt.pso("layer_norm_f32")?);
                bind(0, x);
                bind(1, g);
                bind(2, b);
                bind(3, s.out);
                set_bytes(enc, 4, &[u32_of(d)?, eps.to_bits()]);
                enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(TG, 1, 1));
            }
            Step::MeanPool { x, mask, n, d } => {
                enc.set_compute_pipeline_state(rt.pso("mean_pool_f32")?);
                bind(0, x);
                bind(1, mask);
                bind(2, s.out);
                set_bytes(enc, 3, &[u32_of(n)?, u32_of(d)?]);
                grid1(d);
            }
            Step::L2Normalize { x, n } => {
                enc.set_compute_pipeline_state(rt.pso("l2_normalize_f32")?);
                bind(0, x);
                bind(1, s.out);
                set_bytes(enc, 2, &u32_of(n)?);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(TG, 1, 1));
            }
            Step::RmsNorm {
                x,
                g,
                res,
                scale,
                chain,
                rope,
                rows,
                d,
                eps,
            } => {
                if scale.is_some() || chain.is_some() || rope.is_some() {
                    enc.set_compute_pipeline_state(rt.pso("rmsnorm_fused_f32")?);
                    bind(0, x);
                    bind(1, g);
                    bind(2, res.unwrap_or(x));
                    bind(3, scale.unwrap_or(x));
                    bind(4, s.out);
                    if let Some((g2, out2, _)) = chain {
                        bind(5, g2);
                        bind(6, out2);
                    } else {
                        bind(5, g);
                        bind(6, s.out);
                    }
                    #[repr(C)]
                    struct RmsNormFusedParams {
                        rows: u32,
                        d: u32,
                        eps: f32,
                        eps2: f32,
                        flags: u32,
                        n_seq: u32,
                        rotary_dim: u32,
                        pair_stride: u32,
                        mscale: f32,
                    }
                    let flags = (res.is_some() as u32)
                        | ((scale.is_some() as u32) << 1)
                        | ((chain.is_some() as u32) << 2)
                        | ((rope.is_some() as u32) << 3);
                    let p = RmsNormFusedParams {
                        rows: u32_of(rows)?,
                        d: u32_of(d)?,
                        eps,
                        eps2: chain.map_or(0.0, |c| c.2),
                        flags,
                        n_seq: rope.map_or(1, |r| r.n_seq as u32),
                        rotary_dim: rope.map_or(0, |r| r.rotary_dim as u32),
                        pair_stride: rope.map_or(1, |r| r.pair_stride as u32),
                        mscale: rope.map_or(1.0, |r| r.mscale),
                    };
                    set_bytes(enc, 7, &p);
                    if let Some(rp) = rope {
                        bind(8, rp.pos);
                        bind(9, rp.inv_freq);
                    } else {
                        bind(8, x);
                        bind(9, x);
                    }
                    enc.dispatch_thread_groups(
                        MTLSize::new(rows as u64, 1, 1),
                        MTLSize::new(TG, 1, 1),
                    );
                } else {
                    enc.set_compute_pipeline_state(rt.pso(if res.is_some() {
                        "rmsnorm_residual_f32"
                    } else {
                        "rmsnorm_f32"
                    })?);
                    bind(0, x);
                    bind(1, g);
                    let mut slot = 2;
                    if let Some(r) = res {
                        bind(slot, r);
                        slot += 1;
                    }
                    bind(slot, s.out);
                    #[repr(C)]
                    struct RmsNormParams(u32, u32, f32);
                    let p = RmsNormParams(u32_of(rows)?, u32_of(d)?, eps);
                    set_bytes(enc, slot + 1, &p);
                    enc.dispatch_thread_groups(
                        MTLSize::new(rows as u64, 1, 1),
                        MTLSize::new(TG, 1, 1),
                    );
                }
            }
            Step::GemvPacked {
                x,
                w,
                weight_name: _,
                bias,
                res,
                epi,
                m,
                rows,
                cols,
                format,
                variant,
            } => {
                let (pso, rows_per_tg, threads) = self.gemv_pipeline(format, variant, cols)?;
                enc.set_compute_pipeline_state(&pso);
                bind(0, w);
                bind(1, x);
                bind(2, s.out);
                #[repr(C)]
                struct PackedGemvParams(u32, u32, u32, u32, u32);
                let epi_u32 = epi.as_u32();
                let p = PackedGemvParams(
                    u32_of(rows)?,
                    u32_of(cols)?,
                    u32_of(m)?,
                    bias.is_some() as u32,
                    epi_u32,
                );
                set_bytes(enc, 3, &p);
                if let Some(b) = bias {
                    bind(4, b);
                } else {
                    bind(4, s.out);
                }
                if let Some(r) = res {
                    bind(5, r);
                } else {
                    bind(5, s.out);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(rows_per_tg), m as u64, 1),
                    MTLSize::new(threads, 1, 1),
                );
            }
            Step::GemvPackedPair {
                x,
                m,
                cols,
                format,
                ref legs,
            } => {
                #[repr(C)]
                #[derive(Clone, Copy)]
                struct PackedGemvParams(u32, u32, u32, u32, u32);
                #[repr(C)]
                struct PackedGemvPairParams {
                    p0: PackedGemvParams,
                    p1: PackedGemvParams,
                    tgs0: u32,
                    pad0: u32,
                    pad1: u32,
                    pad2: u32,
                }
                let pair_kernel = if m == 1
                    && legs[0].variant.kernel == GemvKernel::Hand
                    && legs[1].variant.kernel == GemvKernel::Hand
                {
                    match format {
                        WeightFormat::BF16 => Some("gemv_bf16_pair"),
                        WeightFormat::Q4_0 => Some("gemv_q4_0_half_pair"),
                        WeightFormat::Q8_0 => Some("gemv_q8_0_pair"),
                        WeightFormat::Q4_K => Some("gemv_q4_k_pair"),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(kname) = pair_kernel {
                    let l0 = &legs[0];
                    let l1 = &legs[1];
                    enc.set_compute_pipeline_state(rt.pso(kname)?);
                    bind(0, l0.w);
                    bind(1, l1.w);
                    bind(2, x);
                    bind(3, l0.out);
                    bind(4, l1.out);
                    let tgs0 = (l0.rows as u64).div_ceil(16);
                    let tgs1 = (l1.rows as u64).div_ceil(16);
                    let pp = PackedGemvPairParams {
                        p0: PackedGemvParams(
                            u32_of(l0.rows)?,
                            u32_of(cols)?,
                            u32_of(m)?,
                            l0.bias.is_some() as u32,
                            l0.epi.as_u32(),
                        ),
                        p1: PackedGemvParams(
                            u32_of(l1.rows)?,
                            u32_of(cols)?,
                            u32_of(m)?,
                            l1.bias.is_some() as u32,
                            l1.epi.as_u32(),
                        ),
                        tgs0: u32_of(tgs0 as usize)?,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    };
                    set_bytes(enc, 5, &pp);
                    bind(6, l0.bias.unwrap_or(l0.out));
                    bind(7, l1.bias.unwrap_or(l1.out));
                    bind(8, l0.res.unwrap_or(l0.out));
                    bind(9, l1.res.unwrap_or(l1.out));
                    enc.dispatch_thread_groups(
                        MTLSize::new(tgs0 + tgs1, m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                } else {
                    for leg in legs {
                        let (pso, rows_per_tg, threads) =
                            self.gemv_pipeline(format, leg.variant, cols)?;
                        enc.set_compute_pipeline_state(&pso);
                        bind(0, leg.w);
                        bind(1, x);
                        bind(2, leg.out);
                        let p = PackedGemvParams(
                            u32_of(leg.rows)?,
                            u32_of(cols)?,
                            u32_of(m)?,
                            leg.bias.is_some() as u32,
                            leg.epi.as_u32(),
                        );
                        set_bytes(enc, 3, &p);
                        bind(4, leg.bias.unwrap_or(leg.out));
                        bind(5, leg.res.unwrap_or(leg.out));
                        enc.dispatch_thread_groups(
                            MTLSize::new((leg.rows as u64).div_ceil(rows_per_tg), m as u64, 1),
                            MTLSize::new(threads, 1, 1),
                        );
                    }
                }
            }
            Step::GemmBatchedPacked {
                x,
                w,
                bias,
                res,
                epi,
                m,
                n,
                k,
                format,
            } => {
                let hand = match format {
                    WeightFormat::BF16 => Some("bf16"),
                    WeightFormat::F16 => Some("f16"),
                    WeightFormat::Q4_0 => Some("q4_0"),
                    WeightFormat::Q8_0 => Some("q8_0"),
                    _ => None,
                };
                if let Some(fmt_name) = hand {
                    let suffix = match epi {
                        Epi::None => "none",
                        Epi::Gelu => "gelu",
                        Epi::Residual => "residual",
                        _ => "none",
                    };
                    enc.set_compute_pipeline_state(
                        rt.pso(&format!("gemm_batched_{fmt_name}_{suffix}"))?,
                    );
                    bind(0, x);
                    bind(1, w);
                    if let Some(b) = bias {
                        bind(2, b);
                    } else {
                        bind(2, s.out);
                    }
                    if let Some(r) = res {
                        bind(3, r);
                    } else {
                        bind(3, s.out);
                    }
                    bind(4, s.out);
                    #[repr(C)]
                    struct GemmParams(u32, u32, u32, u32);
                    let p = GemmParams(u32_of(m)?, u32_of(n)?, u32_of(k)?, bias.is_some() as u32);
                    set_bytes(enc, 5, &p);
                    enc.dispatch_thread_groups(
                        MTLSize::new((n as u64).div_ceil(32), (m as u64).div_ceil(32), 1),
                        MTLSize::new(TG4, 1, 1),
                    );
                } else {
                    // Any other codec format: expand the weight to f32 in a session scratch buffer (one
                    // pass over the packed blocks through the codec's `decode8`), then run the f32 GEMM.
                    // luminal: crates/luminal_metal (dequantize-then-matmul for prefill shapes).
                    let f = format.codec_format().ok_or_else(|| {
                        format!("Metal executor: unsupported batched packed format {format:?}")
                    })?;
                    let bytes = n * k * 4;
                    let mut guard = self.dequant_scratch.lock();
                    if guard
                        .as_ref()
                        .map_or(true, |b| (b.length() as usize) < bytes)
                    {
                        *guard = Some(new_shared_buffer(&rt.device, bytes)?);
                    }
                    let scratch = guard.as_ref().unwrap();
                    let pso = rt.dequant_pso(f)?;
                    enc.set_compute_pipeline_state(&pso);
                    bind(0, w);
                    enc.set_buffer(1, Some(scratch), 0);
                    set_bytes(enc, 2, &[u32_of(n)?, u32_of(k)?]);
                    grid1(n * (k / 8));
                    let kind = self.gemm.unwrap_or_else(|| rt.profile.gemm_for([m, n, k]));
                    rt.encode_gemm(
                        enc,
                        kind,
                        epi,
                        loc(x),
                        (scratch, 0),
                        bias.map(loc),
                        res.map(loc),
                        loc(s.out),
                        [m, n, k],
                    )?;
                }
            }
            Step::SwiGlu { gate, up, n } => {
                enc.set_compute_pipeline_state(rt.pso("swiglu_f32")?);
                bind(0, gate);
                bind(1, up);
                bind(2, s.out);
                set_bytes(enc, 3, &u32_of(n)?);
                grid1(n);
            }
            Step::GeGlu { gate, up, n } => {
                enc.set_compute_pipeline_state(rt.pso("geglu_f32")?);
                bind(0, gate);
                bind(1, up);
                bind(2, s.out);
                set_bytes(enc, 3, &u32_of(n)?);
                grid1(n);
            }
            Step::SplitGlu {
                x,
                rows,
                half_cols,
                kind,
            } => {
                enc.set_compute_pipeline_state(rt.pso("split_glu_f32")?);
                bind(0, x);
                bind(1, s.out);
                set_bytes(enc, 2, &[u32_of(rows)?, u32_of(half_cols)?, kind, 0u32]);
                grid1(rows * half_cols);
            }
            Step::MoeWeightedSum {
                down,
                weights,
                n,
                k,
                h,
            } => {
                enc.set_compute_pipeline_state(rt.pso("moe_weighted_sum_f32")?);
                bind(0, down);
                bind(1, weights);
                bind(2, s.out);
                set_bytes(enc, 3, &[u32_of(n)?, u32_of(k)?, u32_of(h)?, 0u32]);
                grid1(n * h);
            }
            Step::Rope {
                x,
                pos,
                inv_freq,
                n_seq,
                n_heads,
                head_dim,
                pairs: _,
                pair_stride,
                pair_gap: _,
                rotary_dim,
                mscale,
            } => {
                enc.set_compute_pipeline_state(rt.pso("rope_general_f32")?);
                bind(0, x);
                bind(1, s.out);
                bind(2, pos);
                bind(3, inv_freq);
                #[repr(C)]
                struct GraphRopeParams2 {
                    n_heads: u32,
                    head_dim: u32,
                    pairs: u32,
                    pair_stride: u32,
                    pair_gap: u32,
                    rotary_dim: u32,
                    mscale: f32,
                    n_seq: u32,
                }
                let half = (rotary_dim / 2) as u32;
                let (stride, gap) = if pair_stride == 2 {
                    (2u32, 1u32)
                } else {
                    (1u32, half)
                };
                let p = GraphRopeParams2 {
                    n_heads: n_heads as u32,
                    head_dim: head_dim as u32,
                    pairs: half,
                    pair_stride: stride,
                    pair_gap: gap,
                    rotary_dim: rotary_dim as u32,
                    mscale,
                    n_seq: n_seq as u32,
                };
                set_bytes(enc, 4, &p);
                grid1(n_seq * n_heads * head_dim);
            }
            Step::RmsNormBatch { ref segs, rope } => {
                enc.set_compute_pipeline_state(rt.pso("rmsnorm_batch_f32")?);
                #[repr(C)]
                #[derive(Default)]
                struct RmsNormBatchParams {
                    rows: [u32; 4],
                    d: [u32; 4],
                    eps: [f32; 4],
                    nseg: u32,
                    rope_mask: u32,
                    n_seq: u32,
                    rotary_dim: u32,
                    pair_stride: u32,
                    mscale: f32,
                }
                let mut p = RmsNormBatchParams {
                    nseg: segs.len() as u32,
                    n_seq: rope.map_or(1, |r| r.n_seq as u32),
                    rotary_dim: rope.map_or(0, |r| r.rotary_dim as u32),
                    pair_stride: rope.map_or(1, |r| r.pair_stride as u32),
                    mscale: rope.map_or(1.0, |r| r.mscale),
                    ..Default::default()
                };
                let mut total = 0usize;
                for i in 0..4 {
                    // Unused slots rebind segment 0 (never read: `nseg` bounds the row search).
                    let sg = segs.get(i).unwrap_or(&segs[0]);
                    bind(i as u64 * 3, sg.x);
                    bind(i as u64 * 3 + 1, sg.g);
                    bind(i as u64 * 3 + 2, sg.out);
                    if let Some(sg) = segs.get(i) {
                        p.rows[i] = u32_of(sg.rows)?;
                        p.d[i] = u32_of(sg.d)?;
                        p.eps[i] = sg.eps;
                        if sg.has_rope {
                            p.rope_mask |= 1u32 << i;
                        }
                        total += sg.rows;
                    }
                }
                set_bytes(enc, 12, &p);
                if let Some(rp) = rope {
                    bind(13, rp.pos);
                    bind(14, rp.inv_freq);
                } else {
                    bind(13, segs[0].x);
                    bind(14, segs[0].x);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new(total as u64, 1, 1),
                    MTLSize::new(TG, 1, 1),
                );
            }
            Step::RopeBatch {
                pos,
                inv_freq,
                n_seq,
                head_dim,
                rotary_dim,
                pair_stride,
                mscale,
                ref segs,
            } => {
                enc.set_compute_pipeline_state(rt.pso("rope_batch_f32")?);
                #[repr(C)]
                struct RopeBatchParams {
                    n_heads: [u32; 4],
                    nseg: u32,
                    head_dim: u32,
                    rotary_dim: u32,
                    pair_stride: u32,
                    mscale: f32,
                    n_seq: u32,
                }
                let mut p = RopeBatchParams {
                    n_heads: [0; 4],
                    nseg: segs.len() as u32,
                    head_dim: u32_of(head_dim)?,
                    rotary_dim: u32_of(rotary_dim)?,
                    pair_stride: if pair_stride == 2 { 2 } else { 1 },
                    mscale,
                    n_seq: u32_of(n_seq)?,
                };
                let mut total = 0usize;
                for i in 0..4 {
                    let (x, out, _) = *segs.get(i).unwrap_or(&segs[0]);
                    bind(i as u64 * 2, x);
                    bind(i as u64 * 2 + 1, out);
                    if let Some(&(_, _, heads)) = segs.get(i) {
                        p.n_heads[i] = u32_of(heads)?;
                        total += n_seq * heads * head_dim;
                    }
                }
                bind(8, pos);
                bind(9, inv_freq);
                set_bytes(enc, 10, &p);
                grid1(total);
            }
            Step::CachedAttention {
                q,
                k,
                v,
                k_state,
                v_state,
                pos,
                vis,
                rope,
                heads,
                head_dim,
                kv_heads,
                v_head_dim: _,
                scale,
                window,
                soft_cap,
                write,
                m,
                kv_f16,
            } => {
                let start_pos = if let Some(p_val) = pos {
                    let (p_buf, p_off) = loc(p_val);
                    unsafe { *((p_buf.contents() as *const u8).add(p_off) as *const u32) }
                } else {
                    0
                };
                let group = (heads / kv_heads.max(1)).max(1);
                let kv_dim = kv_heads * head_dim;
                let stride = kv_dim;

                // State buffers are physically capped (4096 rows); never write or read past them.
                if let Some(ks) = k_state {
                    let (kb, ko) = loc(ks);
                    let rows =
                        (kb.length() as usize - ko) / (stride * if kv_f16 { 2 } else { 4 }).max(1);
                    if start_pos as usize + m > rows {
                        return Err(format!(
                            "KV cache holds {rows} positions on this backend; need {}",
                            start_pos as usize + m
                        ));
                    }
                }
                let gqa_decode = m == 1 && vis.is_none();
                let fuse_kv_write = gqa_decode && write && k_state.is_some() && v_state.is_some();
                if write && k_state.is_some() && v_state.is_some() && !fuse_kv_write {
                    let ks = k_state.unwrap();
                    let vs = v_state.unwrap();
                    enc.set_compute_pipeline_state(rt.pso(if kv_f16 {
                        "kv_append_batched_f16"
                    } else {
                        "kv_append_batched"
                    })?);
                    bind(0, k);
                    bind(1, v);
                    bind(2, ks);
                    bind(3, vs);
                    #[repr(C)]
                    struct KvAppendParams {
                        m: u32,
                        kv_dim: u32,
                        stride: u32,
                        start_pos: u32,
                    }
                    let append_p = KvAppendParams {
                        m: m as u32,
                        kv_dim: kv_dim as u32,
                        stride: stride as u32,
                        start_pos,
                    };
                    set_bytes(enc, 4, &append_p);
                    grid1(m * kv_dim);
                }

                let (k_buf_id, v_buf_id) = if let (Some(ks), Some(vs)) = (k_state, v_state) {
                    (ks, vs)
                } else {
                    (k, v)
                };

                // Long decode cache: one threadgroup per (KV head group, key slice) plus a merge, instead of
                // one per head scanning every key (8 threadgroups on a 20-core GPU). Slices of at least
                // `ATTN_SPLIT_KEYS` keys, enough of them for about `ATTN_TARGET_TGS` threadgroups.
                let len = start_pos as usize + 1;
                let lo = window.map_or(0, |w| len.saturating_sub(w));
                let n_keys = len - lo;
                let n_splits = attn_splits(n_keys, head_dim, kv_heads, group, gqa_decode);
                let mut scratch_guard = self.attn_scratch.lock();
                if n_splits > 1 {
                    let bytes = heads * n_splits * (head_dim + 2) * 4;
                    if scratch_guard
                        .as_ref()
                        .map_or(true, |b| (b.length() as usize) < bytes)
                    {
                        *scratch_guard = Some(new_shared_buffer(&rt.device, bytes)?);
                    }
                }
                // [KV dtype f32, f16][head_dim <= 32, 64, 128, 256, 512].
                const GQA: [[&str; 5]; 2] = [
                    [
                        "attn_cached_gqa_split_f32_1",
                        "attn_cached_gqa_split_f32_2",
                        "attn_cached_gqa_split_f32_4",
                        "attn_cached_gqa_split_f32_8",
                        "attn_cached_gqa_split_f32_16",
                    ],
                    [
                        "attn_cached_gqa_split_f16_1",
                        "attn_cached_gqa_split_f16_2",
                        "attn_cached_gqa_split_f16_4",
                        "attn_cached_gqa_split_f16_8",
                        "attn_cached_gqa_split_f16_16",
                    ],
                ];
                const FLASH: [[&str; 5]; 2] = [
                    [
                        "flash_attn_cached_f32_1",
                        "flash_attn_cached_f32_2",
                        "flash_attn_cached_f32_4",
                        "flash_attn_cached_f32_8",
                        "flash_attn_cached_f32",
                    ],
                    [
                        "flash_attn_cached_f16_1",
                        "flash_attn_cached_f16_2",
                        "flash_attn_cached_f16_4",
                        "flash_attn_cached_f16_8",
                        "flash_attn_cached_f16",
                    ],
                ];
                let nj = match head_dim.div_ceil(32) {
                    0..=1 => 0,
                    2 => 1,
                    3..=4 => 2,
                    5..=8 => 3,
                    _ => 4,
                };
                enc.set_compute_pipeline_state(if gqa_decode {
                    rt.pso(GQA[kv_f16 as usize][nj])?
                } else if kv_f16 {
                    rt.pso(FLASH[1][nj])?
                } else {
                    rt.pso(FLASH[0][nj])
                        .or_else(|_| rt.pso("attn_cached_f32"))?
                });
                bind(0, q);
                bind(1, k_buf_id);
                bind(2, v_buf_id);
                match scratch_guard.as_ref().filter(|_| n_splits > 1) {
                    Some(part) => enc.set_buffer(3, Some(part), 0),
                    None => bind(3, s.out),
                }
                if gqa_decode {
                    bind(5, k);
                    bind(6, v);
                    bind(7, rope.map_or(q, |r| r.inv_freq));
                } else {
                    // Rows see keys `< vis[row]` when the operand is present (default `pos + row + 1`).
                    bind(5, vis.unwrap_or(q));
                    bind(6, k);
                    bind(7, v);
                }
                #[repr(C)]
                struct AttnCachedParams {
                    n_heads: u32,
                    head_dim: u32,
                    stride: u32,
                    group: u32,
                    scale: f32,
                    window: u32,
                    softcap: f32,
                    start_pos: u32,
                    m: u32,
                    has_vis: u32,
                    n_splits: u32,
                    chunk: u32,
                    write_kv: u32,
                    rotary_dim: u32,
                    pair_stride: u32,
                    mscale: f32,
                }
                let p = AttnCachedParams {
                    n_heads: heads as u32,
                    head_dim: head_dim as u32,
                    stride: stride as u32,
                    group: group as u32,
                    scale,
                    window: window.unwrap_or(0) as u32,
                    softcap: soft_cap.unwrap_or(0.0),
                    start_pos,
                    m: m as u32,
                    has_vis: vis.is_some() as u32,
                    n_splits: n_splits as u32,
                    chunk: n_keys.div_ceil(n_splits) as u32,
                    write_kv: fuse_kv_write as u32,
                    rotary_dim: rope.map_or(0, |r| r.rotary_dim as u32),
                    pair_stride: rope.map_or(1, |r| r.pair_stride as u32),
                    mscale: rope.map_or(1.0, |r| r.mscale),
                };
                set_bytes(enc, 4, &p);
                let tgs_x = if gqa_decode {
                    kv_heads * group.div_ceil(ATTN_HPT)
                } else {
                    heads
                };
                enc.dispatch_thread_groups(
                    MTLSize::new(tgs_x as u64, m as u64, n_splits as u64),
                    MTLSize::new(TG4, 1, 1),
                );
                if let Some(part) = scratch_guard.as_ref().filter(|_| n_splits > 1) {
                    enc.set_compute_pipeline_state(rt.pso("attn_cached_merge_f32")?);
                    enc.set_buffer(0, Some(part), 0);
                    bind(1, s.out);
                    set_bytes(enc, 2, &p);
                    enc.dispatch_thread_groups(
                        MTLSize::new(heads as u64, m as u64, 1),
                        MTLSize::new(TG4, 1, 1),
                    );
                }
            }
            Step::PagedCachedAttention {
                q,
                k_pool,
                v_pool,
                block_table,
                pos,
                heads,
                kv_heads,
                head_dim,
                v_head_dim: _,
                scale,
                window,
                soft_cap,
                tokens_per_block,
                max_blocks_per_seq,
                m,
            } => {
                let start_pos = if let Some(p_val) = pos {
                    let (p_buf, p_off) = loc(p_val);
                    unsafe { *((p_buf.contents() as *const u8).add(p_off) as *const u32) }
                } else {
                    0
                };
                let group = (heads / kv_heads.max(1)).max(1);
                let stride = kv_heads * head_dim;

                enc.set_compute_pipeline_state(rt.pso("paged_attn_cached_f32")?);
                bind(0, q);
                bind(1, k_pool);
                bind(2, v_pool);
                bind(3, block_table);
                bind(4, s.out);
                #[repr(C)]
                struct PagedAttnCachedParams {
                    n_heads: u32,
                    head_dim: u32,
                    stride: u32,
                    group: u32,
                    scale: f32,
                    window: u32,
                    softcap: f32,
                    start_pos: u32,
                    m: u32,
                    tokens_per_block: u32,
                    max_blocks_per_seq: u32,
                }
                let p = PagedAttnCachedParams {
                    n_heads: heads as u32,
                    head_dim: head_dim as u32,
                    stride: stride as u32,
                    group: group as u32,
                    scale,
                    window: window.unwrap_or(0) as u32,
                    softcap: soft_cap.unwrap_or(0.0),
                    start_pos,
                    m: m as u32,
                    tokens_per_block: tokens_per_block as u32,
                    max_blocks_per_seq: max_blocks_per_seq as u32,
                };
                set_bytes(enc, 5, &p);
                enc.dispatch_thread_groups(
                    MTLSize::new(heads as u64, m as u64, 1),
                    MTLSize::new(TG4, 1, 1),
                );
            }
            Step::CausalConv1dCached {
                x,
                w,
                state,
                n,
                channels,
                kernel,
                dilation,
            } => {
                enc.set_compute_pipeline_state(rt.pso("causal_conv1d_f32")?);
                bind(0, x);
                bind(1, w);
                if let Some(st) = state {
                    bind(2, st);
                } else {
                    bind(2, s.out);
                }
                bind(3, s.out);
                let hist = (kernel - 1) * dilation;
                #[repr(C)]
                struct Conv1dParams {
                    n: u32,
                    channels: u32,
                    kernel_size: u32,
                    dilation: u32,
                    hist: u32,
                    has_state: u32,
                }
                let p = Conv1dParams {
                    n: n as u32,
                    channels: channels as u32,
                    kernel_size: kernel as u32,
                    dilation: dilation as u32,
                    hist: hist as u32,
                    has_state: state.is_some() as u32,
                };
                set_bytes(enc, 4, &p);
                grid1(channels);
            }
            Step::DeltaRule {
                q,
                k,
                v,
                beta,
                log_decay,
                state,
                n,
                key_heads,
                val_heads,
                key_dim,
                val_dim,
            } => {
                enc.set_compute_pipeline_state(rt.pso("delta_rule_f32")?);
                bind(0, q);
                bind(1, k);
                bind(2, v);
                bind(3, beta);
                bind(4, log_decay);
                let temp_state_buf;
                if let Some(st) = state {
                    bind(5, st);
                } else {
                    let sz = (val_heads * key_dim * val_dim * 4).max(256);
                    temp_state_buf = new_shared_buffer(&rt.device, sz)?;
                    enc.set_buffer(5, Some(&temp_state_buf), 0);
                }
                bind(6, s.out);
                let rep = (val_heads / key_heads.max(1)).max(1);
                #[repr(C)]
                struct DeltaRuleParams {
                    n: u32,
                    key_heads: u32,
                    val_heads: u32,
                    key_dim: u32,
                    val_dim: u32,
                    rep: u32,
                }
                let p = DeltaRuleParams {
                    n: n as u32,
                    key_heads: key_heads as u32,
                    val_heads: val_heads as u32,
                    key_dim: key_dim as u32,
                    val_dim: val_dim as u32,
                    rep: rep as u32,
                };
                set_bytes(enc, 7, &p);
                if key_dim > 512 {
                    return Err(format!(
                        "node '{}': delta rule key_dim {key_dim} exceeds 512",
                        s.name
                    ));
                }
                enc.dispatch_thread_groups(
                    MTLSize::new((val_dim as u64).div_ceil(32), val_heads as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            Step::MegakernelMlp {
                x,
                norm_w,
                gate_w,
                up_w,
                down_w,
                post_norm_w,
                pre_res,
                ref gate_name,
                ref up_name,
                ref down_name,
                variants,
                hidden,
                inter,
                eps,
                is_geglu,
            } => {
                let bytes_norm = hidden * 4;
                let bytes_inter = inter * 4;
                let total_scratch = bytes_norm + 2 * bytes_inter;

                let fallback_scratch;
                let scratch = if let Some(b) =
                    mlp_scratch_buf.filter(|b| (b.length() as usize) >= total_scratch)
                {
                    b
                } else {
                    let mut guard = self.mlp_scratch.lock();
                    if guard
                        .as_ref()
                        .is_none_or(|b| (b.length() as usize) < total_scratch)
                    {
                        let buf = new_shared_buffer(&self.rt.device, total_scratch.max(256))?;
                        *guard = Some(buf);
                    }
                    fallback_scratch = guard.as_ref().expect("allocated above").clone();
                    &fallback_scratch
                };
                let off_norm = 0usize;
                let off_gate = bytes_norm;
                let off_up = bytes_norm + bytes_inter;

                // 1. RMSNorm: x * norm_w -> scratch[off_norm] (optionally fused with preceding post-attn norm/residual -> x)
                if let Some((x0, g0, res0, scale0, eps0)) = pre_res {
                    #[repr(C)]
                    struct RmsNormFusedParams {
                        rows: u32,
                        d: u32,
                        eps: f32,
                        eps2: f32,
                        flags: u32,
                        n_seq: u32,
                        rotary_dim: u32,
                        pair_stride: u32,
                        mscale: f32,
                    }
                    enc.set_compute_pipeline_state(rt.pso("rmsnorm_fused_f32")?);
                    bind(0, x0);
                    bind(1, g0);
                    bind(2, res0.unwrap_or(x0));
                    bind(3, scale0.unwrap_or(x0));
                    bind(4, x);
                    bind(5, norm_w);
                    enc.set_buffer(6, Some(scratch), off_norm as u64);
                    let flags = (res0.is_some() as u32) | ((scale0.is_some() as u32) << 1) | 4;
                    let p = RmsNormFusedParams {
                        rows: 1,
                        d: u32_of(hidden)?,
                        eps: eps0,
                        eps2: eps,
                        flags,
                        n_seq: 1,
                        rotary_dim: 0,
                        pair_stride: 1,
                        mscale: 1.0,
                    };
                    set_bytes(enc, 7, &p);
                    bind(8, x0);
                    bind(9, x0);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(TG, 1, 1));
                } else {
                    enc.set_compute_pipeline_state(rt.pso("rmsnorm_f32")?);
                    bind(0, x);
                    bind(1, norm_w);
                    enc.set_buffer(2, Some(scratch), off_norm as u64);
                    #[repr(C)]
                    struct RmsNormParams(u32, u32, f32);
                    let p = RmsNormParams(1, u32_of(hidden)?, eps);
                    set_bytes(enc, 3, &p);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(TG, 1, 1));
                }
                // 2. Gate GEMV: scratch[off_norm] * gate_w -> scratch[off_gate]
                self.encode_gemv_or_gemm(
                    enc,
                    (scratch, off_norm),
                    loc(gate_w),
                    variants[0],
                    gate_name,
                    None,
                    (scratch, off_gate),
                    inter,
                    hidden,
                    Epi::None,
                )?;

                // 3. Up GEMV: scratch[off_norm] * up_w -> scratch[off_up]
                self.encode_gemv_or_gemm(
                    enc,
                    (scratch, off_norm),
                    loc(up_w),
                    variants[1],
                    up_name,
                    None,
                    (scratch, off_up),
                    inter,
                    hidden,
                    Epi::None,
                )?;

                // 4. SwiGLU / GeGLU: scratch[off_gate] * scratch[off_up] -> scratch[off_gate]
                let pso_name = if is_geglu { "geglu_f32" } else { "swiglu_f32" };
                enc.set_compute_pipeline_state(rt.pso(pso_name)?);
                enc.set_buffer(0, Some(scratch), off_gate as u64);
                enc.set_buffer(1, Some(scratch), off_up as u64);
                enc.set_buffer(2, Some(scratch), off_gate as u64);
                set_bytes(enc, 3, &u32_of(inter)?);
                grid1(inter);

                // 5. Down GEMV with residual addition: scratch[off_gate] * down_w + x -> s.out
                if let Some(pn) = post_norm_w {
                    // 5. Down GEMV into scratch[off_norm]: scratch[off_gate] * down_w -> scratch[off_norm]
                    self.encode_gemv_or_gemm(
                        enc,
                        (scratch, off_gate),
                        loc(down_w),
                        variants[2],
                        down_name,
                        None,
                        (scratch, off_norm),
                        hidden,
                        inter,
                        Epi::None,
                    )?;

                    // 6. Post-down RMSNorm with residual add: scratch[off_norm] * pn + x -> s.out
                    enc.set_compute_pipeline_state(rt.pso("rmsnorm_residual_f32")?);
                    enc.set_buffer(0, Some(scratch), off_norm as u64);
                    bind(1, pn);
                    bind(2, x);
                    bind(3, s.out);
                    #[repr(C)]
                    struct RmsNormParams(u32, u32, f32);
                    let p = RmsNormParams(1, u32_of(hidden)?, eps);
                    set_bytes(enc, 4, &p);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(TG, 1, 1));
                } else {
                    // 5. Down GEMV with residual addition: scratch[off_gate] * down_w + x -> s.out
                    self.encode_gemv_or_gemm(
                        enc,
                        (scratch, off_gate),
                        loc(down_w),
                        variants[2],
                        down_name,
                        Some(loc(x)),
                        loc(s.out),
                        hidden,
                        inter,
                        Epi::Residual,
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Step {
    /// A step owned by a lowering group in `ext`.
    Ext(ext::ExtStep),
    Embedding {
        table: ValueId,
        ids: ValueId,
        n: usize,
        d: usize,
        rows: usize,
    },
    LayerNorm {
        x: ValueId,
        g: ValueId,
        b: ValueId,
        rows: usize,
        d: usize,
        eps: f32,
    },
    Linear {
        x: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        res: Option<ValueId>,
        epi: Epi,
        m: usize,
        n: usize,
        k: usize,
    },
    MeanPool {
        x: ValueId,
        mask: ValueId,
        n: usize,
        d: usize,
    },
    L2Normalize {
        x: ValueId,
        n: usize,
    },
    RmsNorm {
        x: ValueId,
        g: ValueId,
        /// Added to the normalised output (`res + norm(x) * g`) when the norm's only reader was an Add.
        res: Option<ValueId>,
        /// Scalar multiplier (`(norm(x) * g [+ res]) * scale[0]`) when the only reader was a scalar Mul.
        scale: Option<ValueId>,
        /// Chained pre-norm `(g2, out2, eps2)` over this step's output `out`.
        chain: Option<(ValueId, ValueId, f32)>,
        /// Fused RoPE applied per head after normalisation when the norm's only reader was a Rope.
        rope: Option<NormRope>,
        rows: usize,
        d: usize,
        eps: f32,
    },
    GemvPacked {
        x: ValueId,
        w: ValueId,
        weight_name: String,
        bias: Option<ValueId>,
        res: Option<ValueId>,
        epi: Epi,
        m: usize,
        rows: usize,
        cols: usize,
        format: WeightFormat,
        /// Kernel variant, set by [`Session::prepare_plan`] from the tuner (or the format default).
        variant: GemvVariant,
    },
    GemvPackedPair {
        x: ValueId,
        m: usize,
        cols: usize,
        format: WeightFormat,
        legs: [GemvPairLeg; 2],
    },
    GemmBatchedPacked {
        x: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        res: Option<ValueId>,
        epi: Epi,
        m: usize,
        n: usize,
        k: usize,
        format: WeightFormat,
    },
    SwiGlu {
        gate: ValueId,
        up: ValueId,
        n: usize,
    },
    GeGlu {
        gate: ValueId,
        up: ValueId,
        n: usize,
    },
    SplitGlu {
        x: ValueId,
        rows: usize,
        half_cols: usize,
        kind: u32,
    },
    MoeWeightedSum {
        down: ValueId,
        weights: ValueId,
        n: usize,
        k: usize,
        h: usize,
    },
    Rope {
        x: ValueId,
        pos: ValueId,
        inv_freq: ValueId,
        n_seq: usize,
        n_heads: usize,
        head_dim: usize,
        pairs: usize,
        pair_stride: usize,
        pair_gap: usize,
        rotary_dim: usize,
        mscale: f32,
    },
    /// Horizontal batch of up to 4 independent RmsNorm segments (`batch_horizontal`).
    RmsNormBatch {
        segs: Vec<NormSeg>,
        rope: Option<NormRope>,
    },
    /// Horizontal batch of up to 4 Rope segments `(x, out, n_heads)` sharing pos, inv_freq and geometry.
    RopeBatch {
        pos: ValueId,
        inv_freq: ValueId,
        n_seq: usize,
        head_dim: usize,
        rotary_dim: usize,
        pair_stride: usize,
        mscale: f32,
        segs: Vec<(ValueId, ValueId, usize)>,
    },
    CachedAttention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        k_state: Option<ValueId>,
        v_state: Option<ValueId>,
        pos: Option<ValueId>,
        vis: Option<ValueId>,
        rope: Option<NormRope>,
        heads: usize,
        head_dim: usize,
        kv_heads: usize,
        v_head_dim: usize,
        scale: f32,
        window: Option<usize>,
        soft_cap: Option<f32>,
        write: bool,
        m: usize,
        /// The k/v state buffers hold f16 (set by `assign_kv_dtypes`).
        kv_f16: bool,
    },
    PagedCachedAttention {
        q: ValueId,
        k_pool: ValueId,
        v_pool: ValueId,
        block_table: ValueId,
        pos: Option<ValueId>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        scale: f32,
        window: Option<usize>,
        soft_cap: Option<f32>,
        tokens_per_block: usize,
        max_blocks_per_seq: usize,
        m: usize,
    },
    CausalConv1dCached {
        x: ValueId,
        w: ValueId,
        state: Option<ValueId>,
        n: usize,
        channels: usize,
        kernel: usize,
        dilation: usize,
    },
    DeltaRule {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        beta: ValueId,
        log_decay: ValueId,
        state: Option<ValueId>,
        n: usize,
        key_heads: usize,
        val_heads: usize,
        key_dim: usize,
        val_dim: usize,
    },
    MegakernelMlp {
        x: ValueId,
        norm_w: ValueId,
        gate_w: ValueId,
        up_w: ValueId,
        down_w: ValueId,
        post_norm_w: Option<ValueId>,
        pre_res: Option<(ValueId, ValueId, Option<ValueId>, Option<ValueId>, f32)>,
        gate_name: String,
        up_name: String,
        down_name: String,
        /// gate / up / down GEMV kernels, set by [`Session::prepare_plan`] like `GemvPacked::variant`.
        variants: [GemvVariant; 3],
        hidden: usize,
        inter: usize,
        eps: f32,
        is_geglu: bool,
    },
}

/// One dispatch: the step, the value it writes, and every value it reads.
#[derive(Debug, Clone)]
pub struct Planned {
    pub name: String,
    pub op: Step,
    pub out: ValueId,
    /// Further values this step writes (multi-output fusions); empty for every single-output step.
    pub extra_outs: Vec<ValueId>,
    pub reads: Vec<ValueId>,
}

#[derive(Debug, Clone)]
pub struct StateRead {
    pub name: String,
    pub bytes_per_pos: u64,
    pub window: Option<usize>,
    pub fixed_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) enum PlanValueLoc {
    Arena(usize),
    Weight(Buffer, usize),
    State(usize),
    Unbound,
}

/// A pre-planned execution schedule and arena allocation for a static graph.
#[derive(Debug, Clone)]
pub struct PreparedPlan {
    pub steps: Vec<Planned>,
    pub offsets: HashMap<ValueId, usize>,
    pub arena_size: usize,
    pub is_decode: bool,
    pub weight_bytes: u64,
    pub state_reads: Vec<StateRead>,
    pub flops_per_run: u64,
    /// Every value viewing a graph input: (value, input index, byte offset into the input).
    pub(crate) input_views: Vec<(ValueId, usize, usize)>,
    pub(crate) mlp_scratch_size: usize,
    pub(crate) embedding_checks: Vec<(usize, usize, String)>,
    /// `StateWrite` (pos0 input index, rows written, declared state rows, node name).
    pub(crate) state_write_checks: Vec<(usize, usize, usize, String)>,
    pub(crate) attn_pos_inputs: Vec<(usize, usize)>,
    pub(crate) state_vars: Vec<(ValueId, String)>,
    pub(crate) value_locs: Vec<PlanValueLoc>,
    /// Names of the states stored as f16 (see [`assign_kv_dtypes`]).
    pub f16_states: HashSet<String>,
}

/// The one f16 KV cache policy (`Session::set_kv_f16`, `GIBBS_KV_DTYPE`): decides which states are
/// stored as f16 and flags their `CachedAttention` steps. A state qualifies when every step reading it
/// uses it as the k or v state of a `CachedAttention` (any other reader, e.g. a plain `Attention` or
/// `StateReorder` over the same cache, needs f32), and both states of a step agree. Returns the f16
/// state names.
fn assign_kv_dtypes(graph: &Graph, steps: &mut [Planned]) -> HashSet<String> {
    let name_of = |v: ValueId| match &graph.value(v).kind {
        ValueKind::State { name: n, .. } => Some(n.clone()),
        _ => None,
    };
    let kv_pair = |op: &Step| match op {
        Step::CachedAttention {
            k_state: Some(k),
            v_state: Some(v),
            ..
        } => Some((*k, *v)),
        _ => None,
    };
    let (mut f16, mut other) = (HashSet::new(), HashSet::new());
    for s in steps.iter() {
        let pair = kv_pair(&s.op);
        for &r in &s.reads {
            if let Some(n) = name_of(r) {
                if pair.is_some_and(|(k, v)| r == k || r == v) {
                    f16.insert(n);
                } else {
                    other.insert(n);
                }
            }
        }
    }
    f16.retain(|n| !other.contains(n));
    // A step's two states share one kernel dtype: drop a state whose partner dropped out.
    loop {
        let before = f16.len();
        for s in steps.iter() {
            if let Some((k, v)) = kv_pair(&s.op) {
                let (kn, vn) = (
                    name_of(k).unwrap_or_default(),
                    name_of(v).unwrap_or_default(),
                );
                if !(f16.contains(&kn) && f16.contains(&vn)) {
                    f16.remove(&kn);
                    f16.remove(&vn);
                }
            }
        }
        if f16.len() == before {
            break;
        }
    }
    for s in steps.iter_mut() {
        if let Some((k, _)) = kv_pair(&s.op) {
            let on = name_of(k).is_some_and(|n| f16.contains(&n));
            if let Step::CachedAttention { kv_f16, .. } = &mut s.op {
                *kv_f16 = on;
            }
        }
    }
    f16
}

/// Element count of the physical buffer for a state value: 2-D states with more than 4096 declared rows
/// keep at least 4096 rows (a declared 64k-row cache would waste GBs) and grow in 4096-row steps to
/// cover `need_rows` positions, never beyond the declared rows.
fn state_len(g: &Graph, v: ValueId, need_rows: usize) -> Result<usize, String> {
    let val = g.value(v);
    if val.shape.len() == 2 && val.shape[0] > 4096 {
        let rows = (need_rows.div_ceil(4096) * 4096).max(4096);
        return Ok(rows.min(val.shape[0]) * val.shape[1]);
    }
    numel(g, v)
}

fn numel(g: &Graph, v: ValueId) -> Result<usize, String> {
    let val = g.value(v);
    if val.shape.is_empty() {
        return Err(format!(
            "value '{}' has no static shape (the Metal executor needs one)",
            val.name
        ));
    }
    Ok(val.shape.iter().product())
}

fn u32_of(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("dimension {n} exceeds the Metal kernels' u32 range"))
}

/// Whether planner fusions are enabled (`false` when `GIBBS_FUSION=off|none|0`).
pub(crate) fn fusion_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(
            std::env::var("GIBBS_FUSION").as_deref(),
            Ok("none" | "off" | "0")
        )
    });
    *ON
}

/// RoPE parameters fused into [`Step::RmsNorm`] / [`Step::RmsNormBatch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormRope {
    pub pos: ValueId,
    pub inv_freq: ValueId,
    pub n_seq: usize,
    pub rotary_dim: usize,
    pub pair_stride: usize,
    pub mscale: f32,
}

/// One segment of [`Step::RmsNormBatch`].
#[derive(Debug, Clone)]
pub struct NormSeg {
    pub x: ValueId,
    pub g: ValueId,
    pub out: ValueId,
    pub rows: usize,
    pub d: usize,
    pub eps: f32,
    pub has_rope: bool,
}

/// One projection leg of [`Step::GemvPackedPair`].
#[derive(Debug, Clone)]
pub struct GemvPairLeg {
    pub w: ValueId,
    pub weight_name: String,
    pub bias: Option<ValueId>,
    pub res: Option<ValueId>,
    pub epi: Epi,
    pub rows: usize,
    pub out: ValueId,
    pub variant: GemvVariant,
}

/// Pairs already-adjacent small-row (`rows <= 512`) `HAND` `Step::GemvPacked` projections sharing
/// `x`, `cols`, and `format` after variant assignment and arena allocation, verifying disjoint
/// arena ranges for outputs and inputs.
fn pair_small_hand_gemvs(
    graph: &Graph,
    steps: &mut Vec<Planned>,
    aliases: &HashMap<ValueId, (ValueId, usize)>,
    offsets: &HashMap<ValueId, usize>,
    allow_quant_pair: bool,
) {
    if !fusion_enabled() {
        return;
    }
    let root = |v: ValueId| aliases.get(&v).map_or(v, |a| a.0);
    let range_of = |v: ValueId| -> Option<(usize, usize)> {
        let off = *offsets.get(&v)?;
        let len = graph.value(v).shape.iter().product::<usize>() * 4;
        Some((off, off + len))
    };
    let overlaps = |(a0, a1): (usize, usize), (b0, b1): (usize, usize)| a0 < b1 && b0 < a1;
    let can_pair = |a: &Planned, b: &Planned| -> bool {
        if !a.extra_outs.is_empty() || !b.extra_outs.is_empty() {
            return false;
        }
        let (
            Step::GemvPacked {
                x: x0,
                m: 1,
                rows: r0,
                cols: c0,
                format: f0,
                variant: v0,
                ..
            },
            Step::GemvPacked {
                x: x1,
                m: 1,
                rows: r1,
                cols: c1,
                format: f1,
                variant: v1,
                ..
            },
        ) = (&a.op, &b.op)
        else {
            return false;
        };
        if x0 != x1
            || f0 != f1
            || c0 != c1
            || *v0 != GemvVariant::HAND
            || *v1 != GemvVariant::HAND
            || *r0 > 512
            || *r1 > 512
        {
            return false;
        }
        let fmt_ok = *f0 == WeightFormat::BF16
            || (allow_quant_pair
                && (*f0 == WeightFormat::Q4_K
                    || *f0 == WeightFormat::Q8_0
                    || (*f0 == WeightFormat::Q4_0 && *c0 >= 512 && (*c0 & 255) == 0)));
        if !fmt_ok
            || b.reads
                .iter()
                .any(|&r| r == a.out || root(r) == root(a.out))
        {
            return false;
        }
        let (Some(out0_r), Some(out1_r)) = (range_of(a.out), range_of(b.out)) else {
            return false;
        };
        if overlaps(out0_r, out1_r) {
            return false;
        }
        if a.reads
            .iter()
            .filter_map(|&r| range_of(r))
            .any(|rr| overlaps(out1_r, rr))
        {
            return false;
        }
        if b.reads
            .iter()
            .filter_map(|&r| range_of(r))
            .any(|rr| overlaps(out0_r, rr))
        {
            return false;
        }
        true
    };
    let same_rows = |a: &Planned, b: &Planned| -> bool {
        matches!(
            (&a.op, &b.op),
            (
                Step::GemvPacked { rows: r0, .. },
                Step::GemvPacked { rows: r1, .. }
            ) if r0 == r1
        )
    };
    let old = std::mem::take(steps);
    steps.reserve(old.len());
    let mut i = 0;
    while i < old.len() {
        let defer_to_next = i + 2 < old.len()
            && can_pair(&old[i], &old[i + 1])
            && !same_rows(&old[i], &old[i + 1])
            && can_pair(&old[i + 1], &old[i + 2])
            && same_rows(&old[i + 1], &old[i + 2]);
        if i + 1 < old.len() && !defer_to_next && can_pair(&old[i], &old[i + 1]) {
            let s0 = old[i].clone();
            let s1 = old[i + 1].clone();
            let (
                Step::GemvPacked {
                    x,
                    w: w0,
                    weight_name: wn0,
                    bias: bias0,
                    res: res0,
                    epi: epi0,
                    m,
                    rows: rows0,
                    cols,
                    format,
                    variant: var0,
                },
                Step::GemvPacked {
                    w: w1,
                    weight_name: wn1,
                    bias: bias1,
                    res: res1,
                    epi: epi1,
                    rows: rows1,
                    variant: var1,
                    ..
                },
            ) = (s0.op, s1.op)
            else {
                unreachable!("checked by can_pair");
            };
            let mut reads = s0.reads;
            for r in s1.reads {
                if !reads.contains(&r) {
                    reads.push(r);
                }
            }
            steps.push(Planned {
                name: format!("{}+{}", s0.name, s1.name),
                op: Step::GemvPackedPair {
                    x,
                    m,
                    cols,
                    format,
                    legs: [
                        GemvPairLeg {
                            w: w0,
                            weight_name: wn0,
                            bias: bias0,
                            res: res0,
                            epi: epi0,
                            rows: rows0,
                            out: s0.out,
                            variant: var0,
                        },
                        GemvPairLeg {
                            w: w1,
                            weight_name: wn1,
                            bias: bias1,
                            res: res1,
                            epi: epi1,
                            rows: rows1,
                            out: s1.out,
                            variant: var1,
                        },
                    ],
                },
                out: s0.out,
                extra_outs: vec![s1.out],
                reads,
            });
            i += 2;
        } else {
            steps.push(old[i].clone());
            i += 1;
        }
    }
}

/// Merges independent steps into one dispatch as [`fuse::batches`] extracts them: runs (<= 4) of
/// RmsNorms without a residual (any rows/d/eps) or of Ropes with identical geometry (any head
/// count), with independent GEMVs between norm members moved ahead of the run; an RmsNorm with a
/// residual or scale chains the plain RmsNorm of its output, or the MegakernelMlp reading a one-row
/// one absorbs it. Anything else is left untouched.
fn batch_horizontal(
    steps: Vec<Planned>,
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<Vec<Planned>, String> {
    if !fusion_enabled() {
        return Ok(steps);
    }
    let b = fuse::batches(&steps, aliases)?;
    let n = steps.len();
    let mut out: Vec<Planned> = Vec::with_capacity(n);
    let mut steps: Vec<Option<Planned>> = steps.into_iter().map(Some).collect();
    let mut take = |p: usize| steps[p].take().expect("each step is taken once");
    let mut p = 0;
    while p < n {
        let run: Vec<Planned> = match b.runs.get(&p) {
            Some(members) => {
                let last = members[members.len() - 1];
                for q in (p..=last).filter(|q| !members.contains(q)) {
                    out.push(take(q));
                }
                p = last + 1;
                members.iter().map(|&q| take(q)).collect()
            }
            None => {
                let mut first = take(p);
                p += 1;
                if b.chain.contains(&(p - 1)) {
                    let next = take(p);
                    p += 1;
                    if let (
                        Step::RmsNorm { ref mut chain, .. },
                        Step::RmsNorm {
                            g: g2, eps: eps2, ..
                        },
                    ) = (&mut first.op, &next.op)
                    {
                        *chain = Some((*g2, next.out, *eps2));
                        first.extra_outs.push(next.out);
                        if !first.reads.contains(g2) {
                            first.reads.push(*g2);
                        }
                        first.name = format!("{}+{}", first.name, next.name);
                    }
                } else if b.absorb.contains(&(p - 1)) {
                    let mut mlp = take(p);
                    p += 1;
                    if let (
                        &Step::RmsNorm {
                            x,
                            g,
                            res,
                            scale,
                            eps,
                            ..
                        },
                        Step::MegakernelMlp {
                            ref mut pre_res, ..
                        },
                    ) = (&first.op, &mut mlp.op)
                    {
                        *pre_res = Some((x, g, res, scale, eps));
                    }
                    mlp.extra_outs.push(first.out);
                    for &r in &first.reads {
                        if !mlp.reads.contains(&r) {
                            mlp.reads.push(r);
                        }
                    }
                    mlp.name = format!("{}+{}", first.name, mlp.name);
                    first = mlp;
                }
                out.push(first);
                continue;
            }
        };
        let name = run
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join("+");
        let outs: Vec<ValueId> = run.iter().map(|s| s.out).collect();
        let mut reads: Vec<ValueId> = Vec::new();
        for &r in run.iter().flat_map(|s| &s.reads) {
            if !reads.contains(&r) {
                reads.push(r);
            }
        }
        let op = match &run[0].op {
            Step::RmsNorm { .. } => Step::RmsNormBatch {
                rope: run.iter().find_map(|s| match s.op {
                    Step::RmsNorm { rope, .. } => rope,
                    _ => None,
                }),
                segs: run
                    .iter()
                    .map(|s| match s.op {
                        Step::RmsNorm {
                            x,
                            g,
                            rows,
                            d,
                            eps,
                            rope,
                            ..
                        } => NormSeg {
                            x,
                            g,
                            out: s.out,
                            rows,
                            d,
                            eps,
                            has_rope: rope.is_some(),
                        },
                        _ => unreachable!("run members share a kind"),
                    })
                    .collect(),
            },
            Step::Rope {
                pos,
                inv_freq,
                n_seq,
                head_dim,
                rotary_dim,
                pair_stride,
                mscale,
                ..
            } => Step::RopeBatch {
                pos: *pos,
                inv_freq: *inv_freq,
                n_seq: *n_seq,
                head_dim: *head_dim,
                rotary_dim: *rotary_dim,
                pair_stride: *pair_stride,
                mscale: *mscale,
                segs: run
                    .iter()
                    .map(|s| match s.op {
                        Step::Rope { x, n_heads, .. } => (x, s.out, n_heads),
                        _ => unreachable!("run members share a kind"),
                    })
                    .collect(),
            },
            _ => unreachable!("only RmsNorm and Rope runs are built"),
        };
        out.push(Planned {
            name,
            op,
            out: outs[0],
            extra_outs: outs[1..].to_vec(),
            reads,
        });
    }
    Ok(out)
}

/// The packed format a Linear over the `[n, k]` weight `w` runs in; `None`: the f32 kernels.
fn linear_format(g: &Graph, w: ValueId) -> Option<WeightFormat> {
    let &[_, k] = &g.value(w).shape[..] else {
        return None;
    };
    let format = match g.value(w).dtype {
        DType::Quant(QuantFormat::GGUF_Q8_0) => WeightFormat::Q8_0,
        DType::Quant(QuantFormat::GGUF_Q4_0) => WeightFormat::Q4_0,
        DType::Quant(QuantFormat::GGUF_IQ4_XS) => WeightFormat::IQ4_XS,
        DType::Quant(QuantFormat::GGUF_IQ2_M) | DType::Quant(QuantFormat::GGUF_IQ2_S) => {
            WeightFormat::IQ2_M
        }
        DType::Quant(QuantFormat::GGUF_Q4_K_M) => WeightFormat::Q4_K,
        DType::Quant(QuantFormat::GGUF_Q2_K) => WeightFormat::Q2_K,
        DType::Quant(QuantFormat::GGUF_IQ3_S) => WeightFormat::IQ3_S,
        DType::Quant(f) if crate::graph::codec::codec(f).is_some() => WeightFormat::Gen(f),
        DType::BF16 => WeightFormat::BF16,
        DType::F16 => WeightFormat::F16,
        _ => return None,
    };
    let fits = match format.codec_format() {
        Some(f) => k % crate::graph::codec::codec(f).expect("checked").block_elems as usize == 0,
        None => true,
    };
    fits.then_some(format)
}

/// The [`NormRope`] of the RoPE node `r` (inputs `x, pos, inv_freq`) over `n_seq` positions.
fn norm_rope(r: &crate::graph::Node, n_seq: usize) -> NormRope {
    let Op::Rope {
        rotary_dim,
        variant,
        mscale,
        ..
    } = &r.op
    else {
        unreachable!("planner rules fuse Rope nodes only")
    };
    NormRope {
        pos: r.inputs[1],
        inv_freq: r.inputs[2],
        n_seq,
        rotary_dim: *rotary_dim,
        pair_stride: if matches!(variant, crate::graph::RopeVariant::Interleaved) {
            2
        } else {
            1
        },
        mscale: *mscale,
    }
}

/// Lowers nodes to dispatches, validating ops and shapes. The fusions [`fuse::decide`] extracts
/// (GEMV epilogues, RmsNorm residual/scale/RoPE, attention q/k RoPE) lower inside the absorbing
/// node's step; absorbed nodes emit nothing.
fn plan(g: &Graph) -> Result<(Vec<Planned>, HashMap<ValueId, (ValueId, usize)>), String> {
    let decided = fuse::decide(g)?;
    let mut fused = vec![false; g.nodes.len()];
    for j in decided.absorbed() {
        fused[j] = true;
    }
    let mut steps = Vec::new();
    let mut aliases: HashMap<ValueId, (ValueId, usize)> = HashMap::new();
    for (i, node) in g.nodes.iter().enumerate() {
        if fused[i] {
            continue;
        }
        let name = &node.name;
        let arg = |k: usize| {
            node.inputs
                .get(k)
                .copied()
                .ok_or_else(|| format!("node '{name}': {} needs operand {k}", node.op.name()))
        };
        let n_of = |v: ValueId| numel(g, v);
        let mut out = *node
            .outputs
            .first()
            .ok_or_else(|| format!("node '{name}' has no output"))?;
        let ext_plan = ext::plan(g, node, &mut aliases);
        let (op, out_len) = match &node.op {
            _ if ext_plan.is_some() => match ext_plan.expect("checked")? {
                ext::PlanOut::Alias => {
                    fused[i] = true;
                    continue;
                }
                ext::PlanOut::Step(st, n) => (Step::Ext(st), n),
            },
            Op::Embedding => {
                let (table, ids) = (arg(0)?, arg(1)?);
                let ts = &g.value(table).shape;
                let [rows, d] = ts[..] else {
                    return Err(format!(
                        "node '{name}': embedding table shape {ts:?} is not [rows, d]"
                    ));
                };
                if g.value(ids).dtype != DType::U32 || g.value(table).dtype != DType::F32 {
                    return Err(format!(
                        "node '{name}': embedding needs an f32 table and u32 ids"
                    ));
                }
                let n = n_of(ids)?;
                (
                    Step::Embedding {
                        table,
                        ids,
                        n,
                        d,
                        rows,
                    },
                    n * d,
                )
            }
            Op::Linear => {
                let (x, w) = (arg(0)?, arg(1)?);
                let bias = node.inputs.get(2).copied();
                let ws = &g.value(w).shape;
                let [n, k] = ws[..] else {
                    return Err(format!(
                        "node '{name}': weight shape {ws:?} is not [out, in]"
                    ));
                };
                let xl = n_of(x)?;
                if k == 0 || xl % k != 0 {
                    return Err(format!(
                        "node '{name}': x of {xl} elements does not match weight {ws:?}"
                    ));
                }
                if let Some(b) = bias {
                    if n_of(b)? != n {
                        return Err(format!("node '{name}': bias is not [{n}]"));
                    }
                }
                let m = xl / k;
                let format = linear_format(g, w);
                let (mut epi, mut res) = (Epi::None, None);
                if let Some(&(j, e, side)) = decided.epi.get(&i) {
                    (epi, res) = (e, side);
                    out = g.nodes[j].outputs[0];
                }
                if let Some(fmt) = format {
                    if m >= 16 && fmt.batched_gemm() {
                        (
                            Step::GemmBatchedPacked {
                                x,
                                w,
                                bias,
                                res,
                                epi,
                                m,
                                n,
                                k,
                                format: fmt,
                            },
                            m * n,
                        )
                    } else {
                        (
                            Step::GemvPacked {
                                x,
                                w,
                                weight_name: g.value(w).name.clone(),
                                bias,
                                res,
                                epi,
                                m,
                                rows: n,
                                cols: k,
                                format: fmt,
                                variant: fmt.default_variant(),
                            },
                            m * n,
                        )
                    }
                } else {
                    (
                        Step::Linear {
                            x,
                            w,
                            bias,
                            res,
                            epi,
                            m,
                            n,
                            k,
                        },
                        m * n,
                    )
                }
            }
            Op::LayerNorm { eps } => {
                if node.inputs.len() < 3 {
                    return Err(format!(
                        "node '{name}': LayerNorm without beta is not implemented"
                    ));
                }
                let (x, gm, b) = (arg(0)?, arg(1)?, arg(2)?);
                let (xl, d) = (n_of(x)?, n_of(gm)?);
                if d == 0 || n_of(b)? != d || xl % d != 0 {
                    return Err(format!(
                        "node '{name}': x/gamma/beta lengths {xl}/{d}/{} disagree",
                        n_of(b)?
                    ));
                }
                (
                    Step::LayerNorm {
                        x,
                        g: gm,
                        b,
                        rows: xl / d,
                        d,
                        eps: *eps,
                    },
                    xl,
                )
            }
            Op::MeanPool => {
                let (x, mask) = (arg(0)?, arg(1)?);
                let d = g.value(x).shape.last().copied().unwrap_or(0);
                let n = n_of(mask)?;
                if d == 0 || n_of(x)? != n * d || g.value(mask).dtype != DType::U32 {
                    return Err(format!(
                        "node '{name}': x of {} elements does not match a u32 mask of {n}",
                        n_of(x)?
                    ));
                }
                (Step::MeanPool { x, mask, n, d }, d)
            }
            Op::L2Normalize => {
                let x = arg(0)?;
                (Step::L2Normalize { x, n: n_of(x)? }, n_of(x)?)
            }
            Op::RmsNorm { eps } => {
                let (x, g_val) = (arg(0)?, arg(1)?);
                let d = g.value(g_val).shape.last().copied().unwrap_or(0);
                let xl = n_of(x)?;
                if d == 0 || xl % d != 0 {
                    return Err(format!("node '{name}': RmsNorm shapes mismatch"));
                }
                // Absorbed `+ r` (`rmsnorm_residual_f32`), then `* s` (scalar output scale), or a
                // RoPE over the same head_dim (per-head norm + rotary embedding).
                let fz = decided.norm.get(&i).copied().unwrap_or_default();
                let mut absorb = |j: usize| out = g.nodes[j].outputs[0];
                let res = fz.res.map(|(j, r)| {
                    absorb(j);
                    r
                });
                let scale = fz.scale.map(|(k, s)| {
                    absorb(k);
                    s
                });
                let rope = match fz.rope {
                    Some(j) => {
                        absorb(j);
                        Some(norm_rope(&g.nodes[j], n_of(g.nodes[j].inputs[1])?))
                    }
                    None => None,
                };
                (
                    Step::RmsNorm {
                        x,
                        g: g_val,
                        res,
                        scale,
                        chain: None,
                        rope,
                        rows: xl / d,
                        d,
                        eps: *eps,
                    },
                    xl,
                )
            }
            Op::SwiGlu => {
                let (gate, up) = (arg(0)?, arg(1)?);
                let n = n_of(gate)?;
                (Step::SwiGlu { gate, up, n }, n)
            }
            Op::GeGlu => {
                let (gate, up) = (arg(0)?, arg(1)?);
                let n = n_of(gate)?;
                (Step::GeGlu { gate, up, n }, n)
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                mscale,
            } => {
                let (x, pos, inv_freq) = (arg(0)?, arg(1)?, arg(2)?);
                let xl = n_of(x)?;
                let hd = *head_dim;
                let hd_nz = hd.max(1);
                let n_heads = xl / hd_nz;
                let n_seq = n_of(pos)?;
                let heads_per_seq = n_heads / n_seq.max(1);
                let pairs = rotary_dim / 2;
                let (pair_stride, pair_gap) = match variant {
                    crate::graph::RopeVariant::Interleaved => (2, 1),
                    crate::graph::RopeVariant::HalfSplit => (1, pairs),
                    crate::graph::RopeVariant::MRope {
                        pairs: p_layout, ..
                    } => match p_layout {
                        crate::graph::PairLayout::Adjacent => (2, 1),
                        crate::graph::PairLayout::Half => (1, pairs),
                    },
                };
                (
                    Step::Rope {
                        x,
                        pos,
                        inv_freq,
                        n_seq,
                        n_heads: heads_per_seq,
                        head_dim: hd,
                        pairs,
                        pair_stride,
                        pair_gap,
                        rotary_dim: *rotary_dim,
                        mscale: *mscale,
                    },
                    xl,
                )
            }
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
                scale,
                causal: _,
                window,
                soft_cap,
                write,
            } => {
                let (q, k, rope) = match decided.attn_rope.get(&i) {
                    Some(&(qi, ki)) => (
                        g.nodes[qi].inputs[0],
                        g.nodes[ki].inputs[0],
                        Some(norm_rope(&g.nodes[qi], 1)),
                    ),
                    None => (arg(0)?, arg(1)?, None),
                };
                let v = arg(2)?;
                let k_state = if node.inputs.len() >= 4 {
                    Some(arg(3)?)
                } else {
                    None
                };
                let v_state = if node.inputs.len() >= 5 {
                    Some(arg(4)?)
                } else {
                    None
                };
                let pos = if node.inputs.len() >= 6 {
                    Some(arg(5)?)
                } else {
                    None
                };
                let vis = if node.inputs.len() >= 7 {
                    Some(arg(6)?)
                } else {
                    None
                };
                if (*head_dim).max(*v_head_dim) > ATTN_MAX_HEAD_DIM {
                    return Err(format!(
                        "node '{name}': head_dim {} exceeds the Metal kernel's {ATTN_MAX_HEAD_DIM}",
                        (*head_dim).max(*v_head_dim)
                    ));
                }
                let ql = n_of(q)?;
                let m = ql / (*heads * *head_dim).max(1);
                (
                    Step::CachedAttention {
                        q,
                        k,
                        v,
                        k_state,
                        v_state,
                        pos,
                        vis,
                        rope,
                        heads: *heads,
                        head_dim: *head_dim,
                        kv_heads: *kv_heads,
                        v_head_dim: *v_head_dim,
                        scale: *scale,
                        window: *window,
                        soft_cap: *soft_cap,
                        write: *write,
                        m,
                        kv_f16: false,
                    },
                    m * *heads * *v_head_dim,
                )
            }
            Op::PagedCachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
                scale,
                causal: _,
                window,
                soft_cap,
                tokens_per_block,
                max_blocks_per_seq,
            } => {
                let q = arg(0)?;
                let k_pool = arg(1)?;
                let v_pool = arg(2)?;
                let block_table = arg(3)?;
                let pos = if node.inputs.len() >= 5 {
                    Some(arg(4)?)
                } else {
                    None
                };
                if (*head_dim).max(*v_head_dim) > ATTN_MAX_HEAD_DIM {
                    return Err(format!(
                        "node '{name}': head_dim {} exceeds the Metal kernel's {ATTN_MAX_HEAD_DIM}",
                        (*head_dim).max(*v_head_dim)
                    ));
                }
                let ql = n_of(q)?;
                let m = ql / (*heads * *head_dim).max(1);
                (
                    Step::PagedCachedAttention {
                        q,
                        k_pool,
                        v_pool,
                        block_table,
                        pos,
                        heads: *heads,
                        kv_heads: *kv_heads,
                        head_dim: *head_dim,
                        v_head_dim: *v_head_dim,
                        scale: *scale,
                        window: *window,
                        soft_cap: *soft_cap,
                        tokens_per_block: *tokens_per_block,
                        max_blocks_per_seq: *max_blocks_per_seq,
                        m,
                    },
                    m * *heads * *v_head_dim,
                )
            }
            Op::CausalConv1dCached { kernel, dilation } => {
                let x = arg(0)?;
                let w = arg(1)?;
                let state = if node.inputs.len() >= 3 {
                    Some(arg(2)?)
                } else {
                    None
                };
                let k = *kernel;
                let d = *dilation;
                let xl = n_of(x)?;
                let wl = n_of(w)?;
                let channels = wl / k.max(1);
                let n = xl / channels.max(1);
                (
                    Step::CausalConv1dCached {
                        x,
                        w,
                        state,
                        n,
                        channels,
                        kernel: k,
                        dilation: d,
                    },
                    xl,
                )
            }
            Op::DeltaRule {
                key_heads,
                val_heads,
                key_dim,
                val_dim,
            } => {
                let q = arg(0)?;
                let k = arg(1)?;
                let v = arg(2)?;
                let beta = arg(3)?;
                let log_decay = arg(4)?;
                let state = if node.inputs.len() >= 6 {
                    Some(arg(5)?)
                } else {
                    None
                };
                let kh = *key_heads;
                let vh = *val_heads;
                let kd = *key_dim;
                let vd = *val_dim;
                let ql = n_of(q)?;
                let n = ql / (kh * kd).max(1);
                (
                    Step::DeltaRule {
                        q,
                        k,
                        v,
                        beta,
                        log_decay,
                        state,
                        n,
                        key_heads: kh,
                        val_heads: vh,
                        key_dim: kd,
                        val_dim: vd,
                    },
                    n * vh * vd,
                )
            }
            Op::Megakernel(desc) => match &desc.kind {
                crate::graph::MegakernelKind::Mlp {
                    hidden_dim,
                    intermediate_dim,
                    eps,
                    is_geglu,
                    has_post_norm,
                } => {
                    let (x, norm_w, gate_w, up_w, down_w) =
                        (arg(0)?, arg(1)?, arg(2)?, arg(3)?, arg(4)?);
                    let post_norm_w = if *has_post_norm { Some(arg(5)?) } else { None };
                    let n = n_of(x)?;

                    let gate_name = g.value(gate_w).name.clone();
                    let up_name = g.value(up_w).name.clone();
                    let down_name = g.value(down_w).name.clone();
                    (
                        Step::MegakernelMlp {
                            x,
                            norm_w,
                            gate_w,
                            up_w,
                            down_w,
                            post_norm_w,
                            pre_res: None,
                            gate_name,
                            up_name,
                            down_name,
                            variants: [GemvVariant::HAND; 3],
                            hidden: *hidden_dim,
                            inter: *intermediate_dim,
                            eps: *eps,
                            is_geglu: *is_geglu,
                        },
                        n,
                    )
                }
            },
            op => {
                return Err(format!(
                    "Metal executor does not implement {} (node '{name}')",
                    op.name()
                ))
            }
        };
        if n_of(out)? != out_len {
            return Err(format!(
                "node '{name}': output declared {:?}, computed {out_len} elements",
                g.value(out).shape
            ));
        }
        let reads = match op {
            Step::Ext(ref st) => ext::reads(st),
            Step::Embedding { table, ids, .. } => vec![table, ids],
            Step::L2Normalize { x, .. } => vec![x],
            Step::LayerNorm { x, g, b, .. } => vec![x, g, b],
            Step::Linear {
                x, w, bias, res, ..
            } => [Some(x), Some(w), bias, res]
                .into_iter()
                .flatten()
                .collect(),
            Step::GemvPacked {
                x, w, bias, res, ..
            } => [Some(x), Some(w), bias, res]
                .into_iter()
                .flatten()
                .collect(),
            Step::GemmBatchedPacked {
                x, w, bias, res, ..
            } => [Some(x), Some(w), bias, res]
                .into_iter()
                .flatten()
                .collect(),
            Step::MeanPool { x, mask, .. } => vec![x, mask],
            Step::RmsNorm {
                x,
                g,
                res,
                scale,
                chain,
                rope,
                ..
            } => [
                Some(x),
                Some(g),
                res,
                scale,
                chain.map(|c| c.0),
                rope.map(|r| r.pos),
                rope.map(|r| r.inv_freq),
            ]
            .into_iter()
            .flatten()
            .collect(),
            Step::SwiGlu { gate, up, .. } | Step::GeGlu { gate, up, .. } => vec![gate, up],
            Step::Rope {
                x, pos, inv_freq, ..
            } => vec![x, pos, inv_freq],
            Step::CachedAttention {
                q,
                k,
                v,
                k_state,
                v_state,
                pos,
                vis,
                rope,
                ..
            } => [
                Some(q),
                Some(k),
                Some(v),
                k_state,
                v_state,
                pos,
                vis,
                rope.map(|r| r.inv_freq),
            ]
            .into_iter()
            .flatten()
            .collect(),
            Step::PagedCachedAttention {
                q,
                k_pool,
                v_pool,
                block_table,
                pos,
                ..
            } => [Some(q), Some(k_pool), Some(v_pool), Some(block_table), pos]
                .into_iter()
                .flatten()
                .collect(),
            Step::CausalConv1dCached { x, w, state, .. } => {
                [Some(x), Some(w), state].into_iter().flatten().collect()
            }
            Step::DeltaRule {
                q,
                k,
                v,
                beta,
                log_decay,
                state,
                ..
            } => [
                Some(q),
                Some(k),
                Some(v),
                Some(beta),
                Some(log_decay),
                state,
            ]
            .into_iter()
            .flatten()
            .collect(),
            // Built only by post-passes (`fuse_moe_subgraphs` / `batch_horizontal`).
            Step::RmsNormBatch { .. }
            | Step::RopeBatch { .. }
            | Step::SplitGlu { .. }
            | Step::MoeWeightedSum { .. }
            | Step::GemvPackedPair { .. } => {
                unreachable!("post-pass steps are created after node lowering")
            }
            Step::MegakernelMlp {
                x,
                norm_w,
                gate_w,
                up_w,
                down_w,
                post_norm_w,
                ..
            } => {
                let mut v = vec![x, norm_w, gate_w, up_w, down_w];
                if let Some(pn) = post_norm_w {
                    v.push(pn);
                }
                v
            }
        };
        // Operands are f32 except embedding ids, pooling mask (u32), and packed weights.
        let is_non_f32_allowed = |v: ValueId| -> bool {
            match &op {
                Step::Ext(st) => ext::allows_non_f32(st, v),
                Step::Embedding { ids, .. } => v == *ids,
                Step::MeanPool { mask, .. } => v == *mask,
                Step::GemvPacked { w, .. } | Step::GemmBatchedPacked { w, .. } => v == *w,
                Step::Rope { pos, .. } => v == *pos,
                Step::RmsNorm { rope: Some(r), .. } => v == r.pos,
                Step::CachedAttention { pos, vis, .. } => Some(v) == *pos || Some(v) == *vis,
                Step::PagedCachedAttention {
                    block_table, pos, ..
                } => v == *block_table || Some(v) == *pos,
                Step::MegakernelMlp {
                    gate_w,
                    up_w,
                    down_w,
                    ..
                } => v == *gate_w || v == *up_w || v == *down_w,
                _ => false,
            }
        };
        if let Some(&v) = reads
            .iter()
            .find(|&&v| !is_non_f32_allowed(v) && g.value(v).dtype != DType::F32)
        {
            return Err(format!(
                "node '{name}': operand '{}' is {:?}; the Metal kernels take f32",
                g.value(v).name,
                g.value(v).dtype
            ));
        }
        steps.push(Planned {
            name: name.clone(),
            op,
            out,
            extra_outs: Vec::new(),
            reads,
        });
    }
    fuse_moe_subgraphs(g, &mut steps, &aliases)?;
    ext::deltanet::fuse(g, &mut steps)?;
    Ok((steps, aliases))
}

/// Applies the MoE subgraph fusions [`fuse::moe`] extracts: router TopK+GatherCols+Softmax
/// (+GatherCols+Mul), fused gate/up Slice+Slice+GeGlu/SwiGlu, and expert down Mul+Reduce(Sum).
fn fuse_moe_subgraphs(
    g: &Graph,
    steps: &mut Vec<Planned>,
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<(), String> {
    if !fusion_enabled() {
        return Ok(());
    }
    let decided = fuse::moe(g, steps, aliases)?;
    let mut old = std::mem::take(steps).into_iter().enumerate();
    while let Some((p, mut st)) = old.next() {
        let Some(&f) = decided.get(&p) else {
            steps.push(st);
            continue;
        };
        let mut rest: Vec<Planned> = old.by_ref().take(f.len() - 1).map(|(_, s)| s).collect();
        match f {
            fuse::MoeFusion::TopKSoftmax { weights } => {
                if let Step::Ext(ext::ExtStep::Experts(ext::moe::Step::TopK {
                    ref mut weights_out,
                    ..
                })) = st.op
                {
                    *weights_out = Some(weights);
                }
                st.extra_outs.push(weights);
            }
            fuse::MoeFusion::TopKSoftmaxScaled { scale: sc, weights } => {
                if let Step::Ext(ext::ExtStep::Experts(ext::moe::Step::TopK {
                    ref mut weights_out,
                    ref mut scale,
                    ..
                })) = st.op
                {
                    *weights_out = Some(weights);
                    *scale = Some(sc);
                }
                st.extra_outs.push(weights);
                if !st.reads.contains(&sc) {
                    st.reads.push(sc);
                }
            }
            fuse::MoeFusion::SplitGlu {
                x,
                rows,
                half_cols,
                kind,
            } => {
                let glu = rest.pop().expect("SplitGlu spans 3 steps");
                st = Planned {
                    name: glu.name,
                    op: Step::SplitGlu {
                        x,
                        rows,
                        half_cols,
                        kind,
                    },
                    out: glu.out,
                    extra_outs: glu.extra_outs,
                    reads: vec![x],
                };
            }
            fuse::MoeFusion::WeightedSum {
                down,
                weights,
                n,
                k,
                h,
            } => {
                let red = rest.pop().expect("WeightedSum spans 2 steps");
                st = Planned {
                    name: red.name,
                    op: Step::MoeWeightedSum {
                        down,
                        weights,
                        n,
                        k,
                        h,
                    },
                    out: red.out,
                    extra_outs: Vec::new(),
                    reads: vec![down, weights],
                };
            }
        }
        steps.push(st);
    }
    Ok(())
}

/// First-fit arena offsets for graph inputs and step outputs; a value's space is reused once its
/// last reader has run (graph outputs live to the end). Returns (offsets, arena bytes).
fn allocate(
    g: &Graph,
    steps: &[Planned],
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<(HashMap<ValueId, usize>, usize), String> {
    let mut last: HashMap<ValueId, usize> = HashMap::new();
    for (s, st) in steps.iter().enumerate() {
        for &v in &st.reads {
            last.insert(v, s);
            if let Some(&(src, _)) = aliases.get(&v) {
                let prev = last.get(&src).copied().unwrap_or(0);
                last.insert(src, prev.max(s));
            }
        }
    }
    for &v in &g.outputs {
        last.insert(v, usize::MAX);
        if let Some(&(src, _)) = aliases.get(&v) {
            last.insert(src, usize::MAX);
        }
    }
    let mut live: Vec<(usize, usize, ValueId)> = Vec::new();
    let mut offsets = HashMap::new();
    let mut peak = 0;
    let mut alloc = |v: ValueId, live: &mut Vec<(usize, usize, ValueId)>| -> Result<(), String> {
        let size = (numel(g, v)? * 4).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
        live.sort_unstable_by_key(|l| l.0);
        let mut off = 0;
        for &(o, sz, _) in live.iter() {
            if off + size <= o {
                break;
            }
            off = off.max(o + sz);
        }
        live.push((off, size, v));
        offsets.insert(v, off);
        peak = peak.max(off + size);
        Ok(())
    };
    for &v in &g.inputs {
        alloc(v, &mut live)?;
    }
    for (s, st) in steps.iter().enumerate() {
        alloc(st.out, &mut live)?;
        for &v in &st.extra_outs {
            alloc(v, &mut live)?;
        }
        live.retain(|&(_, _, v)| last.get(&v).is_some_and(|&l| l > s));
    }
    for (&val, &(src, byte_off)) in aliases {
        if let Some(&src_off) = offsets.get(&src) {
            offsets.insert(val, src_off + byte_off);
        }
    }
    Ok((offsets, peak.max(4)))
}

/// Compiles the kernel library as MSL 4.0 (with MPP); on failure, with the default language
/// version (no MPP). The second value says why MPP is unavailable.
fn compile(device: &Device) -> Result<(Library, Result<(), String>), String> {
    let opts = || {
        let o = CompileOptions::new();
        // Precise exp/sqrt keep GPU results within float rounding of the CPU kernels.
        o.set_fast_math_enabled(false);
        o
    };
    let o4 = opts();
    // SAFETY: `setLanguageVersion:` takes an NSUInteger enum; 0x40000 is MTLLanguageVersion4_0.
    let () = unsafe { msg_send![&*o4, setLanguageVersion: MSL_4_0] };
    match device.new_library_with_source(&format!("#define GIBBS_MPP 1\n{}", &*KERNELS_MSL), &o4) {
        Ok(lib) => Ok((lib, Ok(()))),
        Err(e4) => {
            let lib = device
                .new_library_with_source(&KERNELS_MSL, &opts())
                .map_err(|e| format!("Metal shader compilation failed: {e}"))?;
            Ok((lib, Err(format!("MSL 4.0 compile failed: {e4}"))))
        }
    }
}

/// Stored size of `n` weight elements in `fmt`.
fn weight_format_bytes(fmt: WeightFormat, n: usize) -> usize {
    match fmt {
        WeightFormat::F32 | WeightFormat::U32 => n * 4,
        WeightFormat::BF16 | WeightFormat::F16 => n * 2,
        WeightFormat::Q4_0
        | WeightFormat::Q8_0
        | WeightFormat::IQ4_XS
        | WeightFormat::IQ2_M
        | WeightFormat::Q4_K
        | WeightFormat::Q2_K
        | WeightFormat::IQ3_S
        | WeightFormat::Gen(_) => fmt.codec_format().expect("codec format").tensor_bytes(n),
    }
}

/// Pipelines for `names`, built only from the binary archive at `archive` when it holds them all
/// (`FailOnBinaryArchiveMiss`); otherwise compiled, recorded into a new archive, and saved there.
#[allow(clippy::type_complexity)]
fn pipelines(
    device: &Device,
    lib: &Library,
    names: &[&'static str],
    archive: &Path,
) -> Result<(HashMap<&'static str, ComputePipelineState>, bool, String), String> {
    let url = URL::new_with_string(&format!("file://{}", archive.display()));
    // metal 0.33 wraps the autoreleased `URLWithString:` result as owned and releases it on drop;
    // take the reference that drop gives back.
    // SAFETY: `url` is a live NSURL.
    let () = unsafe { msg_send![&*url, retain] };
    let descriptor = |name: &str| -> Result<ComputePipelineDescriptor, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("Metal function '{name}': {e}"))?;
        let d = ComputePipelineDescriptor::new();
        d.set_compute_function(Some(&f));
        Ok(d)
    };
    if archive.exists() {
        let ad = BinaryArchiveDescriptor::new();
        ad.set_url(&url);
        if let Ok(ar) = device.new_binary_archive_with_descriptor(&ad) {
            let restored: Result<HashMap<_, _>, String> = names
                .iter()
                .map(|&n| {
                    let d = descriptor(n)?;
                    d.set_binary_archives(&[&*ar]);
                    Ok((n, pipeline_from_archive(device, &d)?))
                })
                .collect();
            if let Ok(psos) = restored {
                return Ok((
                    psos,
                    true,
                    format!("binary archive hit ({})", archive.display()),
                ));
            }
        }
    }
    let ar = device.new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new())?;
    let mut psos = HashMap::new();
    for &n in names {
        let d = descriptor(n)?;
        ar.add_compute_pipeline_functions_with_descriptor(&d)
            .map_err(|e| format!("binary archive: add '{n}': {e}"))?;
        psos.insert(
            n,
            device
                .new_compute_pipeline_state(&d)
                .map_err(|e| format!("Metal pipeline '{n}': {e}"))?,
        );
    }
    let saved = archive
        .parent()
        .map_or(Ok(()), |p| {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())
        })
        .and_then(|()| ar.serialize_to_url(&url).map(|_| ()));
    let note = match saved {
        Ok(()) => format!("binary archive miss, recorded {}", archive.display()),
        Err(e) => format!(
            "binary archive miss; saving {} failed: {e}",
            archive.display()
        ),
    };
    Ok((psos, false, note))
}

/// `newComputePipelineStateWithDescriptor:options:FailOnBinaryArchiveMiss` — errors instead of compiling.
fn pipeline_from_archive(
    device: &DeviceRef,
    d: &ComputePipelineDescriptorRef,
) -> Result<ComputePipelineState, String> {
    let mut err: *mut Object = std::ptr::null_mut();
    let reflection: *mut *mut Object = std::ptr::null_mut();
    // SAFETY: valid device and descriptor; a nil reflection pointer is allowed; on failure the
    // method returns nil and sets `err`.
    let pso: *mut MTLComputePipelineState = unsafe {
        msg_send![device, newComputePipelineStateWithDescriptor: d
                                                        options: MTLPipelineOption::FailOnBinaryArchiveMiss
                                                     reflection: reflection
                                                          error: &mut err]
    };
    if pso.is_null() {
        return Err(if err.is_null() {
            "binary archive miss".into()
        } else {
            // SAFETY: `err` is an NSError; localizedDescription is an NSString.
            unsafe {
                let desc: *mut Object = msg_send![err, localizedDescription];
                let c: *const c_char = msg_send![desc, UTF8String];
                CStr::from_ptr(c).to_string_lossy().into_owned()
            }
        });
    }
    // SAFETY: a +1 retained pipeline state returned by a `new...` method.
    Ok(unsafe { ComputePipelineState::from_ptr(pso) })
}

/// The kernel kind of a step's `Debug` text: its innermost variant (`Ext(Norm(RmsNorm { .. }))` is
/// `RmsNorm`, `GemvPacked { .. }` is `GemvPacked`).
fn kernel_kind(desc: &str) -> &str {
    let mut rest = desc;
    loop {
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let (ident, tail) = rest.split_at(end);
        match tail.strip_prefix('(') {
            // A wrapper variant (`Ext(`, `Norm(`): the kind is the variant inside it.
            Some(inner) if inner.starts_with(|c: char| c.is_ascii_uppercase()) => rest = inner,
            _ => return if ident.is_empty() { "?" } else { ident },
        }
    }
}

/// One `GIBBS_DUMP_PLAN` line.
fn plan_line(i: usize, s: &Planned) -> String {
    format!(
        "plan {i:4} {:<46} {}",
        s.name,
        kernel_kind(&format!("{:?}", s.op))
    )
}

/// Key slices a cached-attention step splits `n_keys` visible keys into; above 1 the step adds a
/// merge dispatch.
fn attn_splits(
    n_keys: usize,
    head_dim: usize,
    kv_heads: usize,
    group: usize,
    gqa_decode: bool,
) -> usize {
    let (min_keys, split_keys) = if head_dim >= 256 {
        (32, 16)
    } else {
        (ATTN_SPLIT_MIN_KEYS, ATTN_SPLIT_KEYS)
    };
    if gqa_decode && n_keys >= min_keys {
        let head_tgs = kv_heads * group.div_ceil(ATTN_HPT);
        ATTN_TARGET_TGS
            .div_ceil(head_tgs)
            .min(n_keys / split_keys)
            .clamp(2, ATTN_MAX_SPLITS)
    } else {
        1
    }
}

/// CPU-only listing of what [`Session::prepare_plan`] schedules for `g`: `plan` (MoE subgraph
/// fusion included), `batch_horizontal`, the arena `allocate` and `pair_small_hand_gemvs`, without a
/// device, weights or the tuner. A GEMV whose variant the tuner picks prints `variant=unassigned`
/// (pairing is unaffected: without a variant override it pairs only BF16, which has one kernel).
/// Dispatches per step follow the encoder: attention with `ctx` keys in the cache, and
/// `MegakernelMlp`'s norm, gate, up, activation, down and optional post-norm.
pub fn plan_dump(g: &Graph, ctx: usize) -> Result<String, String> {
    use std::fmt::Write;
    let (steps, aliases) = plan(g)?;
    let mut steps = batch_horizontal(steps, &aliases)?;
    let (offsets, arena) = allocate(g, &steps, &aliases)?;
    pair_small_hand_gemvs(g, &mut steps, &aliases, &offsets, false);
    let mut out = String::new();
    let mut kinds: HashMap<String, usize> = HashMap::new();
    let (mut layer, mut dispatches, mut unassigned) = (None, 0usize, 0usize);
    for (i, s) in steps.iter().enumerate() {
        *kinds
            .entry(kernel_kind(&format!("{:?}", s.op)).to_string())
            .or_default() += 1;
        // The layer index: the first all-digit segment of the step name.
        let l = s.name.split('.').find_map(|p| p.parse::<usize>().ok());
        if i == 0 || l != layer {
            layer = l;
            let _ = writeln!(out, "-- layer {}", l.map_or("-".into(), |l| l.to_string()));
        }
        let v = g.value(s.out);
        let mut line = format!("{} -> {:?} {:?}", plan_line(i, s), v.shape, v.dtype);
        if !s.extra_outs.is_empty() {
            let _ = write!(line, " +{} outs", s.extra_outs.len());
        }
        for &r in &s.reads {
            let w = g.value(r);
            if w.kind == ValueKind::Weight {
                let _ = write!(line, " w:{:?}{:?}", w.dtype, w.shape);
            }
        }
        let mut n = 1;
        match &s.op {
            Step::GemvPacked { m, format, .. } | Step::GemvPackedPair { m, format, .. } => {
                if *m == 1 && tune::enabled() && format.gemv_variants().len() > 1 {
                    unassigned += 1;
                    let _ = write!(line, " fmt={format:?} variant=unassigned");
                } else {
                    let d = format.default_variant();
                    let _ = write!(
                        line,
                        " fmt={format:?} variant={:?}{}",
                        d.kernel, d.rows_per_sg
                    );
                }
            }
            &Step::CachedAttention {
                vis,
                k_state,
                v_state,
                heads,
                kv_heads,
                head_dim,
                window,
                write,
                m,
                ..
            } => {
                // Mirrors the encoder: kv_append unless fused into the GQA decode kernel, then the
                // attention, then a merge when the keys are split.
                let gqa_decode = m == 1 && vis.is_none();
                let n_keys = window.map_or(ctx, |w| ctx.min(w));
                let group = (heads / kv_heads.max(1)).max(1);
                let splits = attn_splits(n_keys, head_dim, kv_heads, group, gqa_decode);
                n = usize::from(write && k_state.is_some() && v_state.is_some() && !gqa_decode)
                    + 1
                    + usize::from(splits > 1);
                let _ = write!(line, " window={window:?} keys@ctx={n_keys} splits={splits}");
            }
            Step::MegakernelMlp { post_norm_w, .. } => n = 5 + usize::from(post_norm_w.is_some()),
            _ => {}
        }
        dispatches += n;
        if let ValueKind::Node(id) = v.kind {
            let node = &g.nodes[id.0 as usize];
            let src = |k: usize| node.inputs.get(k).map(|&x| g.value(x).shape.clone());
            match &node.op {
                Op::Slice { axis, start, end } => {
                    let shape = src(0).unwrap_or_default();
                    let ax = if *axis < 0 {
                        *axis + shape.len() as isize
                    } else {
                        *axis
                    } as usize;
                    let outer: usize = shape.iter().take(ax).product();
                    let _ = write!(
                        line,
                        " slice axis={ax} [{start}..{end}] of {shape:?} contiguous={}",
                        outer == 1
                    );
                }
                Op::Concat { axis } => {
                    let ins: Vec<_> = (0..node.inputs.len()).filter_map(src).collect();
                    let _ = write!(line, " concat axis={axis} of {ins:?}");
                }
                Op::Transpose { perm } => {
                    let _ = write!(
                        line,
                        " transpose {perm:?} of {:?}",
                        src(0).unwrap_or_default()
                    );
                }
                _ => {}
            }
        }
        if n != 1 {
            let _ = write!(line, " dispatches={n}");
        }
        let _ = writeln!(out, "{line}");
    }
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let _ = writeln!(out, "-- step kinds");
    for (k, c) in &kinds {
        let _ = writeln!(out, "kind {c:5} {k}");
    }
    let _ = writeln!(
        out,
        "plan-dump: {} steps, {dispatches} dispatches at ctx {ctx}, arena {arena} B, \
         {unassigned} GEMV steps unassigned (tuner)",
        steps.len()
    );
    Ok(out)
}

/// Per-step GPU timings `(step, op description, seconds, bytes)` grouped by kernel kind, slowest
/// first, with each kind's rate against `ceiling_gbps`.
fn kernel_rows(timings: &[(String, String, f64, u64)], ceiling_gbps: f64) -> Vec<KernelRow> {
    let mut by_kind: HashMap<&str, (usize, f64, u64)> = HashMap::new();
    for (_, desc, s, bytes) in timings {
        let e = by_kind.entry(kernel_kind(desc)).or_default();
        e.0 += 1;
        e.1 += s;
        e.2 += bytes;
    }
    let mut rows: Vec<KernelRow> = by_kind
        .into_iter()
        .map(|(kind, (steps, s, bytes))| {
            let gbps = if s > 0.0 { bytes as f64 / s / 1e9 } else { 0.0 };
            KernelRow {
                kind: kind.to_string(),
                steps,
                us: s * 1e6,
                gbps,
                pct: gbps / ceiling_gbps * 100.0,
            }
        })
        .collect();
    rows.sort_by(|a, b| b.us.total_cmp(&a.us).then_with(|| a.kind.cmp(&b.kind)));
    rows
}

/// `GIBBS_PROFILE_KERNELS=1`: the per-kind table and the ten slowest steps, on stderr.
fn print_kernel_profile(
    timings: &[(String, String, f64, u64)],
    rows: &[KernelRow],
    ceiling_gbps: f64,
) {
    eprintln!("profiler: per-command-buffer timing (bytes = operands read + output written)");
    eprintln!(
        "{:<20} | {:<25} | {:>8} | {:>8} | {:>8} | {:>8}",
        "step", "kernel", "µs", "bytes", "GB/s", "% ceiling"
    );
    eprintln!("by kernel kind: kind | steps | total µs | GB/s | % ceiling");
    for r in rows.iter().take(8) {
        eprintln!(
            "  {:<22} {:>4} {:>9.0} {:>7.1} {:>6.1}%",
            r.kind, r.steps, r.us, r.gbps, r.pct
        );
    }
    let mut slowest: Vec<_> = timings.iter().collect();
    slowest.sort_by(|a, b| b.2.total_cmp(&a.2));
    let show = std::env::var("GIBBS_PROFILE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    for (name, op_desc, s, bytes) in slowest.into_iter().take(show) {
        let gbps = if *s > 0.0 {
            *bytes as f64 / s / 1e9
        } else {
            0.0
        };
        eprintln!(
            "{:<20} | {:<25} | {:>8.1} | {:>8} | {:>8.1} | {:>7.1}%",
            name,
            op_desc.chars().take(25).collect::<String>(),
            s * 1e6,
            bytes,
            gbps,
            gbps / ceiling_gbps * 100.0
        );
    }
}

const CB_CHUNK_MIN_STEPS: usize = 220;

/// Steps in the first command buffer and in each later one.
const CB_CHUNKING: (usize, usize) = (32, usize::MAX);

/// GPU time from the start of the first to the end of the last of consecutive command buffers.
fn gpu_span_seconds(first: &CommandBufferRef, last: &CommandBufferRef) -> f64 {
    // SAFETY: GPUStartTime/GPUEndTime are CFTimeInterval (f64) properties of completed buffers.
    let (t0, t1): (f64, f64) =
        unsafe { (msg_send![first, GPUStartTime], msg_send![last, GPUEndTime]) };
    t1 - t0
}

/// GPU execution time of a completed command buffer, in seconds.
fn gpu_seconds(cb: &CommandBufferRef) -> f64 {
    // SAFETY: GPUStartTime/GPUEndTime are CFTimeInterval (f64) properties of a completed buffer.
    let (t0, t1): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    t1 - t0
}

/// max |a - b|; NaN anywhere makes the result NaN (`f32::max` would drop it).
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |m, d| if d > m || d.is_nan() { d } else { m })
}

/// Removes all but the `keep` most recently modified files in `dir` named `tag*`, never `keep_path`.
fn prune_cache(dir: &Path, tag: &str, keep_path: &Path, keep: usize) {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p != keep_path
                && p.file_name()
                    .is_some_and(|f| f.to_string_lossy().starts_with(tag))
        })
        .filter_map(|p| Some((p.metadata().ok()?.modified().ok()?, p)))
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, p) in files.into_iter().skip(keep) {
        let _ = std::fs::remove_file(p);
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::cpu;
    use crate::graph::StateStore;
    use std::sync::LazyLock;

    /// One runtime (one compile and host-profile run) shared by the tests.
    static RT: LazyLock<Arc<Runtime>> =
        LazyLock::new(|| Runtime::new(tempfile::tempdir().unwrap().path()).unwrap());

    fn rt() -> Arc<Runtime> {
        RT.clone()
    }

    fn kinds() -> Vec<GemmKernel> {
        let mut k = vec![GemmKernel::Simdgroup, GemmKernel::Rows];
        if rt().profile().mpp.is_ok() {
            k.push(GemmKernel::Mpp);
        }
        k
    }

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Runs `g` on the CPU executor (`nn.rs` kernels) and on Metal; returns the first outputs.
    fn both(
        g: &Graph,
        w: &Weights,
        inputs: Vec<Tensor>,
        gemm: Option<GemmKernel>,
    ) -> (Vec<f32>, Vec<f32>) {
        let want = cpu::run_stateless(g, w, inputs.clone())
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        let mut s = Session::new(rt(), w).unwrap();
        if let Some(k) = gemm {
            s.force_gemm(k).unwrap();
        }
        let got = s.run(g, inputs).unwrap().remove(0).into_f32().unwrap();
        assert_eq!(got.len(), want.len());
        (want, got)
    }

    /// max |a - b| relative to max(1, max |a|).
    fn rel_err(want: &[f32], got: &[f32]) -> f32 {
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        max_abs_diff(want, got) / scale
    }

    #[test]
    fn linear_and_epilogues_match_cpu_on_every_gemm_kernel() {
        let f = DType::F32;
        for kind in kinds() {
            for (m, n, k) in [
                (1, 1, 1),
                (17, 50, 37),
                (70, 33, 129),
                (5, 384, 384),
                (130, 96, 64),
            ] {
                // 0: no bias, 1: bias, 2: fused GELU, 3: fused residual, 4: two readers (standalone GELU + Add).
                for case in 0..5 {
                    let mut g = Graph::new();
                    let mut w = Weights::new();
                    let x = g.input("x", vec![m, k], f);
                    let r = g.input("r", vec![m, n], f);
                    let wv = g.weight("w", vec![n, k], f);
                    let bv = g.weight("b", vec![n], f);
                    let ins = if case == 0 {
                        vec![x, wv]
                    } else {
                        vec![x, wv, bv]
                    };
                    let lin = g.add("lin", Op::Linear, &ins, vec![m, n], f);
                    let out = match case {
                        0 | 1 => lin,
                        2 => g.add(
                            "gelu",
                            Op::Unary {
                                kind: crate::graph::UnaryKind::GeluErf,
                            },
                            &[lin],
                            vec![m, n],
                            f,
                        ),
                        3 => g.add(
                            "res",
                            Op::Binary {
                                kind: crate::graph::BinaryKind::Add,
                            },
                            &[r, lin],
                            vec![m, n],
                            f,
                        ),
                        _ => {
                            let a = g.add(
                                "gelu",
                                Op::Unary {
                                    kind: crate::graph::UnaryKind::GeluErf,
                                },
                                &[lin],
                                vec![m, n],
                                f,
                            );
                            g.add(
                                "sum",
                                Op::Binary {
                                    kind: crate::graph::BinaryKind::Add,
                                },
                                &[a, lin],
                                vec![m, n],
                                f,
                            )
                        }
                    };
                    g.output(out);
                    w.insert("w", Tensor::F32(rand(n * k, 1)));
                    w.insert("b", Tensor::F32(rand(n, 2)));
                    let inputs = vec![Tensor::F32(rand(m * k, 3)), Tensor::F32(rand(m * n, 4))];
                    let (want, got) = both(&g, &w, inputs, Some(kind));
                    let e = rel_err(&want, &got);
                    assert!(e < 1e-5, "{kind:?} {m}x{n}x{k} case {case}: rel err {e:e}");
                }
            }
        }
    }

    #[test]
    fn test_graph_fused_epilogue_matches_cpu() {
        let f = DType::F32;
        let bf = DType::BF16;
        let (m, n, k) = (1, 64, 32);
        let bf_weights: Vec<u16> = (0..n * k)
            .map(|i| crate::model::f32_to_bf16(0.01 * ((i % 7) as f32 - 3.0)))
            .collect();
        // 1. Linear -> Silu
        {
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![m, k], f);
            let wv = g.weight("w", vec![n, k], bf);
            let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], f);
            let silu = g.add(
                "silu",
                Op::Unary {
                    kind: crate::graph::UnaryKind::Silu,
                },
                &[lin],
                vec![m, n],
                f,
            );
            g.output(silu);

            w.insert("w", Tensor::BF16(bf_weights.clone()));
            let inputs = vec![Tensor::F32(vec![0.1f32; m * k])];
            let (want, got) = both(&g, &w, inputs, None);
            assert!(rel_err(&want, &got) < 1e-4, "Fused Silu parity failed");
        }
        // 2. Linear -> Gelu
        {
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![m, k], f);
            let wv = g.weight("w", vec![n, k], bf);
            let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], f);
            let gelu = g.add(
                "gelu",
                Op::Unary {
                    kind: crate::graph::UnaryKind::GeluErf,
                },
                &[lin],
                vec![m, n],
                f,
            );
            g.output(gelu);

            w.insert("w", Tensor::BF16(bf_weights.clone()));
            let inputs = vec![Tensor::F32(vec![0.1f32; m * k])];
            let (want, got) = both(&g, &w, inputs, None);
            assert!(rel_err(&want, &got) < 1e-4, "Fused Gelu parity failed");
        }
        // 3. Linear -> SwiGlu
        {
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![m, k], f);
            let up_val = g.input("up", vec![m, n], f);
            let wv = g.weight("w", vec![n, k], bf);
            let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], f);
            let swiglu = g.add("swiglu", Op::SwiGlu, &[lin, up_val], vec![m, n], f);
            g.output(swiglu);

            w.insert("w", Tensor::BF16(bf_weights));
            let inputs = vec![
                Tensor::F32(vec![0.1f32; m * k]),
                Tensor::F32(vec![0.5f32; m * n]),
            ];
            let (want, got) = both(&g, &w, inputs, None);
            assert!(rel_err(&want, &got) < 1e-4, "Fused SwiGlu parity failed");
        }
    }

    #[test]
    fn test_generic_buffer_lifetime_arena_zero_reallocations() {
        let f = DType::F32;
        let (m, n, k) = (1, 64, 64);
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![m, k], f);
        let w1 = g.weight("w1", vec![n, k], f);
        let w2 = g.weight("w2", vec![k, n], f);
        let l1 = g.add("l1", Op::Linear, &[x, w1], vec![m, n], f);
        let a1 = g.add(
            "a1",
            Op::Unary {
                kind: crate::graph::UnaryKind::GeluErf,
            },
            &[l1],
            vec![m, n],
            f,
        );
        let l2 = g.add("l2", Op::Linear, &[a1, w2], vec![m, k], f);
        g.output(l2);

        w.insert("w1", Tensor::F32(vec![0.02f32; n * k]));
        w.insert("w2", Tensor::F32(vec![0.02f32; k * n]));

        let s = Session::new(rt(), &w).unwrap();
        let plan = s.prepare_plan(&g).unwrap();
        let initial_ptr = s
            .arena_buffer_ptr()
            .expect("arena buffer pre-allocated during prepare_plan");

        for step in 0..100 {
            let inputs = vec![Tensor::F32(vec![0.05f32 * (step as f32); m * k])];
            let out = s
                .run_prepared(&g, &plan, None, inputs)
                .expect("decode step succeeds");
            assert_eq!(out.len(), 1);
            let cur_ptr = s
                .arena_buffer_ptr()
                .expect("arena buffer remains allocated");
            assert_eq!(
                cur_ptr, initial_ptr,
                "Arena buffer must never be reallocated across decode steps (step {step})"
            );
        }
    }

    #[test]
    fn layer_norm_and_gelu_match_cpu() {
        let f = DType::F32;
        for (rows, d) in [(7, 384), (3, 33), (1, 1)] {
            let mut g = Graph::new();
            let x = g.input("x", vec![rows, d], f);
            let (gv, bv) = (g.weight("g", vec![d], f), g.weight("b", vec![d], f));
            let ln = g.add(
                "ln",
                Op::LayerNorm { eps: 1e-12 },
                &[x, gv, bv],
                vec![rows, d],
                f,
            );
            g.output(ln);
            let mut w = Weights::new();
            w.insert("g", Tensor::F32(rand(d, 5)));
            w.insert("b", Tensor::F32(rand(d, 6)));
            let xs: Vec<f32> = rand(rows * d, 7).iter().map(|v| v * 10.0 + 3.0).collect();
            let (want, got) = both(&g, &w, vec![Tensor::F32(xs)], None);
            assert!(rel_err(&want, &got) < 1e-5, "{rows}x{d}");
        }
        // GELU over [-8, 8]: the f32 erf fit tracks the CPU's f64 one.
        let mut g = Graph::new();
        let x = g.input("x", vec![1601], f);
        let y = g.add(
            "gelu",
            Op::Unary {
                kind: crate::graph::UnaryKind::GeluErf,
            },
            &[x],
            vec![1601],
            f,
        );
        g.output(y);
        let xs: Vec<f32> = (0..1601).map(|i| i as f32 / 100.0 - 8.0).collect();
        let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(xs)], None);
        let e = max_abs_diff(&want, &got);
        assert!(e < 2e-6, "gelu max abs err {e:e}");
    }

    #[test]
    fn attention_matches_cpu() {
        let f = DType::F32;
        for (nq, nk, heads, hd, causal) in [
            (37, 37, 3, 24, false),
            (37, 37, 3, 24, true),
            (5, 5, 12, 32, false),
            (2, 9, 2, 64, false),
            (1, 1, 1, 256, false),
        ] {
            let w = heads * hd;
            let mut g = Graph::new();
            let q = g.input("q", vec![nq, w], f);
            let k = g.input("k", vec![nk, w], f);
            let v = g.input("v", vec![nk, w], f);
            let op = Op::Attention {
                heads,
                kv_heads: heads,
                causal,
                window: None,
                soft_cap: None,
                scale: None,
            };
            let a = g.add("attn", op, &[q, k, v], vec![nq, w], f);
            g.output(a);
            let inputs = vec![
                Tensor::F32(rand(nq * w, 8)),
                Tensor::F32(rand(nk * w, 9)),
                Tensor::F32(rand(nk * w, 10)),
            ];
            let (want, got) = both(&g, &Weights::new(), inputs, None);
            let e = rel_err(&want, &got);
            assert!(
                e < 1e-5,
                "nq {nq} nk {nk} heads {heads} hd {hd} causal {causal}: {e:e}"
            );
        }
    }

    #[test]
    fn embedding_add_pool_normalize_match_cpu() {
        let (f, u) = (DType::F32, DType::U32);
        for mask in [vec![1, 1, 0, 1, 0], vec![0, 0, 0, 0, 0]] {
            let mut g = Graph::new();
            let ids = g.input("ids", vec![5], u);
            let pos = g.input("pos", vec![5], u);
            let mv = g.input("mask", vec![5], u);
            let (t1, t2) = (
                g.weight("t1", vec![10, 6], f),
                g.weight("t2", vec![5, 6], f),
            );
            let e1 = g.add("e1", Op::Embedding, &[t1, ids], vec![5, 6], f);
            let e2 = g.add("e2", Op::Embedding, &[t2, pos], vec![5, 6], f);
            let s = g.add(
                "sum",
                Op::Binary {
                    kind: crate::graph::BinaryKind::Add,
                },
                &[e1, e2],
                vec![5, 6],
                f,
            );
            let p = g.add("pool", Op::MeanPool, &[s, mv], vec![6], f);
            let o = g.add("l2", Op::L2Normalize, &[p], vec![6], f);
            g.output(o);
            let mut w = Weights::new();
            w.insert("t1", Tensor::F32(rand(60, 11)));
            w.insert("t2", Tensor::F32(rand(30, 12)));
            let inputs = vec![
                Tensor::U32(vec![9, 0, 3, 3, 7]),
                Tensor::U32((0..5).collect()),
                Tensor::U32(mask),
            ];
            let (want, got) = both(&g, &w, inputs, None);
            assert!(rel_err(&want, &got) < 1e-6);
        }
    }

    /// One Gemma-4-E2B sliding-layer decode block (f16 dense weights, hidden 1536, 8 heads, 1 KV head,
    /// head_dim 256, intermediate 6144, 128-key cache): the fixture `scripts/fusion_vs_xla.py` compares
    /// with XLA. Prints `FUSION_BLOCK {"us": <median GPU us>, "dispatches": <n>}`; run once per
    /// `GIBBS_FUSION` value (`cargo test --release --lib fusion_block_bench -- --ignored --nocapture`).
    #[test]
    #[ignore = "measurement"]
    fn fusion_block_bench() {
        let (h, nh, hd, ff, t) = (1536usize, 8usize, 256usize, 6144usize, 128usize);
        let f = DType::F32;
        let (mut g, mut w) = (Graph::new(), Weights::new());
        let mut dense = |g: &mut Graph, name: &str, rows: usize, cols: usize, seed: u64| {
            let scale = 1.0 / (cols as f32).sqrt();
            let data: Vec<u16> = rand(rows * cols, seed)
                .into_iter()
                .map(|v| crate::model::f32_to_f16(v * scale))
                .collect();
            w.insert(name.to_string(), Tensor::F16(data));
            g.weight(name, vec![rows, cols], DType::F16)
        };
        let x0 = g.input("x", vec![1, h], f);
        let pos = g.input("pos", vec![1], DType::U32);
        let ks = g.state("k_state", vec![t, hd], f, false);
        let vs = g.state("v_state", vec![t, hd], f, false);
        let (wq, wk, wv) = (
            dense(&mut g, "wq", nh * hd, h, 1),
            dense(&mut g, "wk", hd, h, 2),
            dense(&mut g, "wv", hd, h, 3),
        );
        let (wo, wg, wu, wd) = (
            dense(&mut g, "wo", h, nh * hd, 4),
            dense(&mut g, "wg", ff, h, 5),
            dense(&mut g, "wu", ff, h, 6),
            dense(&mut g, "wd", h, ff, 7),
        );
        let mut ones = |g: &mut Graph, name: &str, n: usize| {
            w.insert(name.to_string(), Tensor::F32(vec![1.0; n]));
            g.weight(name, vec![n], f)
        };
        let (g1, g2) = (ones(&mut g, "g1", h), ones(&mut g, "g2", h));
        let (qn, kn) = (ones(&mut g, "qn", hd), ones(&mut g, "kn", hd));
        w.insert(
            "inv_freq".to_string(),
            Tensor::F32(
                (0..hd / 2)
                    .map(|i| 10000f32.powf(-(i as f32) / (hd / 2) as f32))
                    .collect(),
            ),
        );
        let inv = g.weight("inv_freq", vec![hd / 2], f);
        let eps = Op::RmsNorm { eps: 1e-6 };
        let rope = Op::Rope {
            head_dim: hd,
            rotary_dim: hd,
            variant: crate::graph::RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        let a = g.add("a", eps.clone(), &[x0, g1], vec![1, h], f);
        let q = g.add("q", Op::Linear, &[a, wq], vec![1, nh * hd], f);
        let k = g.add("k", Op::Linear, &[a, wk], vec![1, hd], f);
        let v = g.add("v", Op::Linear, &[a, wv], vec![1, hd], f);
        let q = g.add("qn", eps.clone(), &[q, qn], vec![1, nh * hd], f);
        let k = g.add("kn", eps.clone(), &[k, kn], vec![1, hd], f);
        let q = g.add("qr", rope.clone(), &[q, pos, inv], vec![1, nh * hd], f);
        let k = g.add("kr", rope, &[k, pos, inv], vec![1, hd], f);
        let att = Op::CachedAttention {
            heads: nh,
            kv_heads: 1,
            head_dim: hd,
            v_head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        };
        let o = g.add("attn", att, &[q, k, v, ks, vs, pos], vec![1, nh * hd], f);
        let o = g.add("o", Op::Linear, &[o, wo], vec![1, h], f);
        let x1 = g.add(
            "res1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[x0, o],
            vec![1, h],
            f,
        );
        let b = g.add("b", eps, &[x1, g2], vec![1, h], f);
        let gate = g.add("gate", Op::Linear, &[b, wg], vec![1, ff], f);
        let up = g.add("up", Op::Linear, &[b, wu], vec![1, ff], f);
        let act = g.add("act", Op::GeGlu, &[gate, up], vec![1, ff], f);
        let d = g.add("down", Op::Linear, &[act, wd], vec![1, h], f);
        let out = g.add(
            "res2",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[x1, d],
            vec![1, h],
            f,
        );
        g.output(out);

        let s = Session::new(rt(), &w).unwrap();
        let plan = s.prepare_plan(&g).unwrap();
        let mut state = StateStore::new();
        let mut run = || {
            let inputs = vec![Tensor::F32(rand(h, 9)), Tensor::U32(vec![100])];
            s.run_prepared(&g, &plan, Some(&mut state), inputs).unwrap();
        };
        for _ in 0..20 {
            run();
        }
        take_run_samples();
        for _ in 0..200 {
            run();
        }
        let mut samples = take_run_samples();
        assert_eq!(samples.len(), 200);
        let mut run: Vec<u64> = samples.iter().map(|s| s.run_ns).collect();
        run.sort_unstable();
        samples.sort_by_key(|s| s.gpu_ns);
        // `run_us` (host wall per call, dispatch and sync included) is the basis comparable with XLA's timing.
        println!(
            "FUSION_BLOCK {{\"us\": {:.1}, \"run_us\": {:.1}, \"dispatches\": {}}}",
            samples[100].gpu_ns as f64 / 1e3,
            run[100] as f64 / 1e3,
            samples[100].dispatches
        );
    }

    /// A two-output step's outputs are both live until their own last reader: later steps never
    /// get arena space that overlaps a still-live extra output.
    #[test]
    fn allocate_gives_extra_outs_distinct_live_ranges() {
        let f = DType::F32;
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], f);
        let gv = g.weight("g", vec![4], f);
        let a = g.add("a", Op::RmsNorm { eps: 1e-5 }, &[x, gv], vec![2, 4], f);
        let b = g.add(
            "b",
            Op::Unary {
                kind: crate::graph::UnaryKind::Silu,
            },
            &[x],
            vec![2, 4],
            f,
        );
        let c = g.add("c", Op::RmsNorm { eps: 1e-5 }, &[a, gv], vec![2, 4], f);
        let d = g.add(
            "d",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[b, c],
            vec![2, 4],
            f,
        );
        g.output(d);
        let norm = |x, out, extra_outs: Vec<ValueId>, reads| Planned {
            name: String::new(),
            op: Step::RmsNorm {
                x,
                g: gv,
                res: None,
                scale: None,
                chain: None,
                rope: None,
                rows: 2,
                d: 4,
                eps: 1e-5,
            },
            out,
            extra_outs,
            reads,
        };
        let steps = vec![
            norm(x, a, vec![b], vec![x, gv]),
            norm(a, c, vec![], vec![a, gv]),
            norm(b, d, vec![], vec![b, c]),
        ];
        let (offsets, _) = allocate(&g, &steps, &HashMap::new()).unwrap();
        let range = |v: ValueId| (offsets[&v], offsets[&v] + 2 * 4 * 4);
        let disjoint =
            |p: ValueId, q: ValueId| range(p).1 <= range(q).0 || range(q).1 <= range(p).0;
        // a is read by step 1, b by step 2, c by step 2, d is the graph output.
        for (p, q) in [(a, b), (a, c), (b, c), (b, d), (c, d)] {
            assert!(
                disjoint(p, q),
                "{p:?} and {q:?} overlap: {:?} {:?}",
                range(p),
                range(q)
            );
        }
    }

    /// Single-reader Rope after RmsNorm (and RmsNormBatch) and single-reader Q/K Rope feeding
    /// decode CachedAttention fuse into their producer/consumer steps and match the CPU executor.
    #[test]
    fn rmsnorm_and_cached_attention_fuse_rope_and_match_cpu() {
        let f = DType::F32;
        let (heads, kv_heads, hd) = (4usize, 2usize, 8usize);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let rope = Op::Rope {
            head_dim: hd,
            rotary_dim: hd,
            variant: crate::graph::RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        // 1. Q/K/V RmsNorms + Q/K Ropes batch into one RmsNormBatch with rope_mask.
        let mut g1 = Graph::new();
        let q_in = g1.input("q", vec![1, q_dim], f);
        let k_in = g1.input("k", vec![1, kv_dim], f);
        let v_in = g1.input("v", vec![1, kv_dim], f);
        let pos = g1.input("pos", vec![1], DType::U32);
        let gw = g1.weight("gw", vec![hd], f);
        let inv = g1.weight("inv", vec![hd / 2], f);
        let qn = g1.add(
            "qn",
            Op::RmsNorm { eps: 1e-5 },
            &[q_in, gw],
            vec![1, q_dim],
            f,
        );
        let kn = g1.add(
            "kn",
            Op::RmsNorm { eps: 1e-5 },
            &[k_in, gw],
            vec![1, kv_dim],
            f,
        );
        let vn = g1.add(
            "vn",
            Op::RmsNorm { eps: 1e-5 },
            &[v_in, gw],
            vec![1, kv_dim],
            f,
        );
        let qr = g1.add("qr", rope.clone(), &[qn, pos, inv], vec![1, q_dim], f);
        let kr = g1.add("kr", rope.clone(), &[kn, pos, inv], vec![1, kv_dim], f);
        let ks = g1.state("ks", vec![16, kv_dim], f, false);
        let vs = g1.state("vs", vec![16, kv_dim], f, false);
        let att = Op::CachedAttention {
            heads,
            kv_heads,
            head_dim: hd,
            v_head_dim: hd,
            scale: 0.25,
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        };
        let out1 = g1.add(
            "att",
            att.clone(),
            &[qr, kr, vn, ks, vs, pos],
            vec![1, q_dim],
            f,
        );
        g1.output(out1);
        let mut w = Weights::new();
        w.insert("gw", Tensor::F32(rand(hd, 1)));
        w.insert("inv", Tensor::F32(rand(hd / 2, 2)));
        let mut sess1 = Session::new(rt(), &w).unwrap();
        let prep1 = sess1.prepare_plan(&g1).unwrap();
        assert_eq!(
            prep1.steps.len(),
            2,
            "RmsNormBatch(with Q/K Rope) + CachedAttention"
        );
        assert!(matches!(
            prep1.steps[0].op,
            Step::RmsNormBatch { rope: Some(_), .. }
        ));

        // 2. Without Q/K RmsNorms, Q/K Ropes fuse directly into decode CachedAttention.
        let mut g2 = Graph::new();
        let q2 = g2.input("q", vec![1, q_dim], f);
        let k2 = g2.input("k", vec![1, kv_dim], f);
        let v2 = g2.input("v", vec![1, kv_dim], f);
        let pos2 = g2.input("pos", vec![1], DType::U32);
        let inv2 = g2.weight("inv", vec![hd / 2], f);
        let qr2 = g2.add("qr", rope.clone(), &[q2, pos2, inv2], vec![1, q_dim], f);
        let kr2 = g2.add("kr", rope, &[k2, pos2, inv2], vec![1, kv_dim], f);
        let ks2 = g2.state("ks", vec![16, kv_dim], f, false);
        let vs2 = g2.state("vs", vec![16, kv_dim], f, false);
        let out2 = g2.add(
            "att",
            att,
            &[qr2, kr2, v2, ks2, vs2, pos2],
            vec![1, q_dim],
            f,
        );
        g2.output(out2);
        let mut sess2 = Session::new(rt(), &w).unwrap();
        let prep2 = sess2.prepare_plan(&g2).unwrap();
        assert_eq!(prep2.steps.len(), 1, "Q/K Rope fused into CachedAttention");
        assert!(matches!(
            prep2.steps[0].op,
            Step::CachedAttention { rope: Some(_), .. }
        ));

        // f16 KV covers the RoPE-fused f16 store; its rounding (<= 2^-11 of each K/V) sets the bound.
        for (kv_f16, tol) in [(false, 1e-5), (true, 1e-3)] {
            sess1.set_kv_f16(kv_f16);
            sess2.set_kv_f16(kv_f16);
            for (g, sess) in [(&g1, &sess1), (&g2, &sess2)] {
                let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
                for step in 0..3u32 {
                    let inp = vec![
                        Tensor::F32(rand(q_dim, 10 + step as u64)),
                        Tensor::F32(rand(kv_dim, 20 + step as u64)),
                        Tensor::F32(rand(kv_dim, 30 + step as u64)),
                        Tensor::U32(vec![step]),
                    ];
                    let want = cpu::run(g, &w, &mut cs, inp.clone()).unwrap();
                    let got = sess.run_with_state(g, Some(&mut ms), inp).unwrap();
                    let e = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                    assert!(e < tol, "f16 KV {kv_f16} step {step}: {e}");
                }
            }
        }
    }

    /// A residual RmsNorm (`norm(x0) * g0 + res0`) feeding a MegakernelMlp fuses into the MLP's
    /// `pre_res` (`rmsnorm_fused_f32`) while keeping `x` live for the MLP down-projection residual.
    #[test]
    fn megakernel_mlp_fuses_pre_residual_norm_and_matches_cpu() {
        let f = DType::F32;
        let (h, inter) = (96usize, 192usize);
        let mut g = Graph::new();
        let x0 = g.input("x0", vec![1, h], f);
        let res0 = g.input("res0", vec![1, h], f);
        let g0 = g.weight("g0", vec![h], f);
        let norm_w = g.weight("norm_w", vec![h], f);
        let gate_w = g.weight("gate_w", vec![inter, h], f);
        let up_w = g.weight("up_w", vec![inter, h], f);
        let down_w = g.weight("down_w", vec![h, inter], f);
        let post_w = g.weight("post_w", vec![h], f);

        let attn_norm = g.add(
            "attn_norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x0, g0],
            vec![1, h],
            f,
        );
        let x = g.add(
            "res_attn",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[res0, attn_norm],
            vec![1, h],
            f,
        );
        let ff_in = g.add(
            "pre_ff_ln",
            Op::RmsNorm { eps: 1e-6 },
            &[x, norm_w],
            vec![1, h],
            f,
        );
        let gate = g.add("gate", Op::Linear, &[ff_in, gate_w], vec![1, inter], f);
        let up = g.add("up", Op::Linear, &[ff_in, up_w], vec![1, inter], f);
        let act = g.add("act", Op::GeGlu, &[gate, up], vec![1, inter], f);
        let down = g.add("down", Op::Linear, &[act, down_w], vec![1, h], f);
        let down_norm = g.add(
            "post_ff_ln",
            Op::RmsNorm { eps: 1e-5 },
            &[down, post_w],
            vec![1, h],
            f,
        );
        let out = g.add(
            "res_ff",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[x, down_norm],
            vec![1, h],
            f,
        );
        g.output(out);
        crate::graph::egraph::fuse_rule(&mut g, None, crate::graph::egraph::Rule::MegakernelMlp);

        let mut w = Weights::new();
        for (k, s) in [("g0", 1u64), ("norm_w", 2), ("post_w", 3)] {
            w.insert(k, Tensor::F32(rand(h, s)));
        }
        for (k, n, s) in [
            ("gate_w", inter * h, 4u64),
            ("up_w", inter * h, 5),
            ("down_w", h * inter, 6),
        ] {
            w.insert(k, Tensor::F32(rand(n, s)));
        }
        let sess = Session::new(rt(), &w).unwrap();
        let prep = sess.prepare_plan(&g).unwrap();
        assert_eq!(prep.steps.len(), 1);
        assert!(matches!(
            prep.steps[0].op,
            Step::MegakernelMlp {
                pre_res: Some(_),
                ..
            }
        ));
        let inp = vec![Tensor::F32(rand(h, 7)), Tensor::F32(rand(h, 8))];
        let (want, got) = both(&g, &w, inp, None);
        assert!(rel_err(&want, &got) < 1e-4);
    }

    /// Adjacent independent RmsNorms (different rows and eps) and Ropes (different head counts) become
    /// one batch step each; a norm reading a batch member's output stays outside it. All outputs match
    /// the CPU executor.
    #[test]
    fn horizontal_batches_match_cpu() {
        let f = DType::F32;
        let mut g = Graph::new();
        let xs = [(3usize, 1e-5f32), (1, 1e-6), (2, 1e-3)]
            .map(|(rows, eps)| (g.input("x", vec![rows, 4], f), rows, eps));
        let gv = g.weight("g", vec![4], f);
        let mut outs: Vec<ValueId> = xs
            .iter()
            .map(|&(x, rows, eps)| g.add("n", Op::RmsNorm { eps }, &[x, gv], vec![rows, 4], f))
            .collect();
        // Depends on the first member: cannot join the run.
        outs.push(g.add(
            "dep",
            Op::RmsNorm { eps: 1e-5 },
            &[outs[0], gv],
            vec![3, 4],
            f,
        ));
        let pos = g.input("pos", vec![1], DType::U32);
        let inv = g.weight("inv", vec![2], f);
        let rope = Op::Rope {
            head_dim: 4,
            rotary_dim: 4,
            variant: crate::graph::RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        let (xa, xb) = (g.input("xa", vec![1, 8], f), g.input("xb", vec![1, 4], f));
        outs.push(g.add("ra", rope.clone(), &[xa, pos, inv], vec![1, 8], f));
        outs.push(g.add("rb", rope, &[xb, pos, inv], vec![1, 4], f));
        for &o in &outs {
            g.output(o);
        }
        let (steps, aliases) = plan(&g).unwrap();
        let steps = batch_horizontal(steps, &aliases).unwrap();
        let kinds: Vec<_> = steps
            .iter()
            .map(|s| kernel_kind(&format!("{:?}", s.op)).to_string())
            .collect();
        assert_eq!(kinds, ["RmsNormBatch", "RmsNorm", "RopeBatch"]);

        let mut w = Weights::new();
        w.insert("g", Tensor::F32(vec![0.5, 1.2, -0.8, 2.1]));
        w.insert("inv", Tensor::F32(vec![1.0, 0.1]));
        let inputs = vec![
            Tensor::F32(rand(12, 1)),
            Tensor::F32(rand(4, 2)),
            Tensor::F32(rand(8, 3)),
            Tensor::U32(vec![5]),
            Tensor::F32(rand(8, 4)),
            Tensor::F32(rand(4, 5)),
        ];
        let want = cpu::run_stateless(&g, &w, inputs.clone()).unwrap();
        let got = Session::new(rt(), &w).unwrap().run(&g, inputs).unwrap();
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            let e = rel_err(a.as_f32().unwrap(), b.as_f32().unwrap());
            assert!(e < 1e-5, "output {i}: {e}");
        }
    }

    #[test]
    fn rmsnorm_swiglu_and_geglu_match_cpu() {
        let f = DType::F32;
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], f);
        let u_val = g.input("u", vec![2, 4], f);
        let gv = g.weight("g", vec![4], f);
        let norm = g.add("norm", Op::RmsNorm { eps: 1e-5 }, &[x, gv], vec![2, 4], f);
        let swiglu = g.add("swiglu", Op::SwiGlu, &[norm, u_val], vec![2, 4], f);
        let geglu = g.add("geglu", Op::GeGlu, &[norm, u_val], vec![2, 4], f);
        let out = g.add(
            "sum",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[swiglu, geglu],
            vec![2, 4],
            f,
        );
        g.output(out);

        let mut w = Weights::new();
        w.insert("g", Tensor::F32(vec![0.5, 1.2, -0.8, 2.1]));
        let inputs = vec![
            Tensor::F32(vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5]),
            Tensor::F32(vec![0.2, -0.4, 1.1, -0.7, 0.9, -1.2, 0.3, 0.8]),
        ];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-4);
    }

    /// `norm(x) * g + r` with the Add as the norm's only reader is one fused step; a norm that
    /// is also read elsewhere stays a plain norm. Both match the CPU executor.
    #[test]
    fn rmsnorm_residual_fuses_and_matches_cpu() {
        let f = DType::F32;
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 4], f);
        let r = g.input("r", vec![3, 4], f);
        let gv = g.weight("g", vec![4], f);
        let n1 = g.add("n1", Op::RmsNorm { eps: 1e-5 }, &[x, gv], vec![3, 4], f);
        let s1 = g.add(
            "s1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[r, n1],
            vec![3, 4],
            f,
        );
        let n2 = g.add("n2", Op::RmsNorm { eps: 1e-5 }, &[s1, gv], vec![3, 4], f);
        let s2 = g.add(
            "s2",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[n2, s1],
            vec![3, 4],
            f,
        );
        let o = g.add(
            "o",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[n2, s2],
            vec![3, 4],
            f,
        );
        g.output(o);
        let (steps, _) = plan(&g).unwrap();
        // n1+s1 fuse; n2 has two readers (s2 and o), so it and s2 stay separate: norm, norm, add, mul.
        assert_eq!(steps.len(), 4, "norm+add, norm, add, mul");
        assert!(matches!(steps[0].op, Step::RmsNorm { res: Some(_), .. }));
        assert!(matches!(steps[1].op, Step::RmsNorm { res: None, .. }));

        let mut w = Weights::new();
        w.insert("g", Tensor::F32(vec![0.5, 1.2, -0.8, 2.1]));
        let inputs = vec![
            Tensor::F32((0..12).map(|i| (i as f32 * 0.7).sin() * 2.0).collect()),
            Tensor::F32((0..12).map(|i| (i as f32 * 1.3).cos()).collect()),
        ];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-5);
    }

    /// `(norm(x) * g + r) * s` followed by `norm(out) * g2` fuses into one `rmsnorm_fused_f32`
    /// dispatch writing both the scaled residual stream and the pre-normalised activation.
    #[test]
    fn rmsnorm_scale_and_chain_fuse_and_match_cpu() {
        let f = DType::F32;
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 8], f);
        let r = g.input("r", vec![2, 8], f);
        let g1 = g.weight("g1", vec![8], f);
        let g2 = g.weight("g2", vec![8], f);
        let sc = g.weight("sc", vec![1], f);
        let n1 = g.add("n1", Op::RmsNorm { eps: 1e-5 }, &[x, g1], vec![2, 8], f);
        let a1 = g.add(
            "a1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[r, n1],
            vec![2, 8],
            f,
        );
        let m1 = g.add(
            "m1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[a1, sc],
            vec![2, 8],
            f,
        );
        let n2 = g.add("n2", Op::RmsNorm { eps: 1e-6 }, &[m1, g2], vec![2, 8], f);
        let out = g.add(
            "out",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[m1, n2],
            vec![2, 8],
            f,
        );
        g.output(out);

        let mut w = Weights::new();
        w.insert(
            "g1",
            Tensor::F32((0..8).map(|i| 0.5 + i as f32 * 0.1).collect()),
        );
        w.insert(
            "g2",
            Tensor::F32((0..8).map(|i| 1.2 - i as f32 * 0.08).collect()),
        );
        w.insert("sc", Tensor::F32(vec![0.70710677]));
        let sess = Session::new(rt(), &w).unwrap();
        let prep = sess.prepare_plan(&g).unwrap();
        assert_eq!(
            prep.steps.len(),
            2,
            "fused norm+add+scale+prenorm, then add"
        );
        assert!(matches!(
            prep.steps[0].op,
            Step::RmsNorm {
                res: Some(_),
                scale: Some(_),
                chain: Some(_),
                ..
            }
        ));
        let inputs = vec![
            Tensor::F32((0..16).map(|i| (i as f32 * 0.4).sin()).collect()),
            Tensor::F32((0..16).map(|i| (i as f32 * 0.9).cos()).collect()),
        ];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-5);
    }

    /// Every kernel source this build emits compiles to a pipeline: the hand library as MSL 4.0
    /// with MPP (each hand, group and MPP kernel as a pipeline with SIMD width 32 and at least the
    /// threads its encoder dispatches), each generated GEMV (`gen` and `gen64`) and dequant variant,
    /// each expert-GEMV grid binding, and the fused GeGlu→GEMV chain. Compile only: nothing is
    /// dispatched and no runtime (host-profile measurement) is created.
    #[test]
    fn every_kernel_source_compiles() {
        use crate::graph::codegen::{self, GemvParams};
        let device = Device::system_default().expect("Metal device");
        let (lib, mpp) = compile(&device).expect("kernel library");
        mpp.expect("MSL 4.0 kernel library");
        // `Runtime::new` checks only the `TG` floor; the fused delta-rule step dispatches more.
        let need = |name: &str| -> u64 {
            if name.starts_with("deltanet_step_") {
                ext::deltanet::THREADS as u64
            } else {
                TG
            }
        };
        let check = |name: &str, pso: &ComputePipelineState| {
            assert_eq!(pso.thread_execution_width(), 32, "{name}");
            assert!(
                pso.max_total_threads_per_threadgroup() >= need(name),
                "{name}: at most {} threads per threadgroup",
                pso.max_total_threads_per_threadgroup()
            );
        };
        for name in BASE_KERNELS
            .iter()
            .copied()
            .chain(ext::kernel_names())
            .chain(MPP_KERNELS)
        {
            let f = lib
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let pso = device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            check(name, &pso);
        }
        let build = |src: &str, name: &str| {
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(false);
            let lib = device
                .new_library_with_source(src, &opts)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let f = lib
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let pso = device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            check(name, &pso);
        };
        let head = "#include <metal_stdlib>\nusing namespace metal;\n";
        let common = codegen::gemv_common(&KERNELS_MSL);
        let mut codecs: Vec<&codec::Codec> = Vec::new();
        for f in [
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_1,
            QuantFormat::GGUF_Q5_0,
            QuantFormat::GGUF_Q5_1,
            QuantFormat::GGUF_Q8_0,
            QuantFormat::GGUF_Q2_K,
            QuantFormat::GGUF_Q3_K,
            QuantFormat::GGUF_Q4_K_M,
            QuantFormat::GGUF_Q5_K_M,
            QuantFormat::GGUF_Q6_K,
            QuantFormat::GGUF_IQ4_NL,
            QuantFormat::GGUF_IQ4_XS,
            QuantFormat::GGUF_IQ3_XXS,
            QuantFormat::GGUF_IQ3_S,
            QuantFormat::GGUF_IQ2_XXS,
            QuantFormat::GGUF_IQ2_XS,
            QuantFormat::GGUF_IQ2_S,
            QuantFormat::GGUF_IQ2_M,
            QuantFormat::GGUF_IQ1_S,
            QuantFormat::GGUF_TQ1_0,
            QuantFormat::GGUF_TQ2_0,
            QuantFormat::GGUF_NVFP4,
        ] {
            if let Some(c) = codec::codec(f) {
                if !codecs.iter().any(|d| d.format == c.format) {
                    codecs.push(c);
                }
            }
        }
        for &c in &codecs {
            let variants = tune::GEN_ROWS.map(tune::GemvVariant::generated);
            for v in variants {
                let p = v.gen_params();
                build(
                    &format!("{head}{common}{}", codegen::gemv_packed(c, p)),
                    &codegen::gemv_packed_name(c, p),
                );
            }
            build(
                &format!("{head}{}", codegen::dequant_packed(c)),
                &codegen::dequant_packed_name(c),
            );
            for r in [2, 4] {
                let p = GemvParams {
                    rows_per_sg: r,
                    ..Default::default()
                };
                build(
                    &format!("{head}{}", codegen::gemv_indexed(c, p)),
                    &codegen::gemv_indexed_name(c, p),
                );
            }
        }
    }

    /// Every packed format (generated and hand kernels): the GPU GEMV equals the CPU executor
    /// (which dequantizes through `dequantize_*_block`), with and without a bias epilogue.
    #[test]
    fn gemv_matches_cpu_for_every_packed_format() {
        use crate::model::layers::gguf_block_codec;
        let formats = [
            QuantFormat::GGUF_Q4_1,
            QuantFormat::GGUF_Q5_0,
            QuantFormat::GGUF_Q5_1,
            QuantFormat::GGUF_Q3_K,
            QuantFormat::GGUF_Q5_K_M,
            QuantFormat::GGUF_Q6_K,
            QuantFormat::GGUF_IQ4_NL,
            QuantFormat::GGUF_IQ3_XXS,
            QuantFormat::GGUF_IQ2_XXS,
            QuantFormat::GGUF_IQ2_XS,
            QuantFormat::GGUF_IQ1_S,
            QuantFormat::GGUF_TQ1_0,
            QuantFormat::GGUF_TQ2_0,
            // hand kernels
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q8_0,
            QuantFormat::GGUF_IQ4_XS,
            QuantFormat::GGUF_IQ2_M,
            QuantFormat::GGUF_IQ2_S,
            QuantFormat::GGUF_Q4_K_M,
            QuantFormat::GGUF_Q2_K,
            QuantFormat::GGUF_IQ3_S,
            QuantFormat::GGUF_NVFP4,
        ];
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        // Finite f16 scale (|x| in 2^-6..2^4) so the reference and the GPU never meet NaN/inf.
        let f16 = |next: &mut dyn FnMut() -> u32| -> [u8; 2] {
            let exp = 9 + (next() % 14) as u16;
            (((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16).to_le_bytes()
        };
        // m = 3 runs the GEMV kernels; m = 20 (>= 16) the batched GEMM (hand kernels, or codec expansion + f32 GEMM).
        for (f, m) in formats.into_iter().flat_map(|f| [(f, 3usize), (f, 20)]) {
            let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
            let scale_at: &[usize] = match f {
                QuantFormat::GGUF_Q4_1
                | QuantFormat::GGUF_Q5_1
                | QuantFormat::GGUF_Q5_K_M
                | QuantFormat::GGUF_Q4_K_M => &[0, 2],
                QuantFormat::GGUF_Q2_K => &[80, 82],
                QuantFormat::GGUF_Q6_K => &[208],
                QuantFormat::GGUF_Q3_K => &[108],
                QuantFormat::GGUF_TQ1_0 => &[52],
                QuantFormat::GGUF_TQ2_0 => &[64],
                _ => &[0],
            };
            let (n, k) = (37usize, be * 3);
            let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                .map(|_| (next() >> 4) as u8)
                .collect();
            for blk in raw.chunks_exact_mut(bb) {
                for &o in scale_at {
                    blk[o..o + 2].copy_from_slice(&f16(&mut next));
                }
            }
            let mut g = Graph::new();
            let x = g.input("x", vec![m, k], DType::F32);
            let w_val = g.weight("w", vec![n, k], DType::Quant(f));
            let b_val = g.weight("b", vec![n], DType::F32);
            let y = g.add(
                "lin",
                Op::Linear,
                &[x, w_val, b_val],
                vec![m, n],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert(
                "w",
                Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
            );
            w.insert(
                "b",
                Tensor::F32((0..n).map(|i| i as f32 * 0.01 - 0.2).collect()),
            );
            let xs: Vec<f32> = (0..m * k)
                .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let want = cpu::run_stateless(&g, &w, vec![Tensor::F32(xs.clone())])
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            // Every variant of the format (hand and generated at each rows/simdgroup); formats with
            // no variant choice (NVFP4) run their one kernel.
            let variants = WeightFormat::of_quant(f).gemv_variants();
            let variants: Vec<Option<GemvVariant>> = if variants.is_empty() {
                vec![None]
            } else {
                variants.into_iter().map(Some).collect()
            };
            for v in variants {
                let mut s = Session::new(rt(), &w).unwrap();
                s.set_variant_override(v);
                let got = s
                    .run(&g, vec![Tensor::F32(xs.clone())])
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                let err = rel_err(&want, &got);
                assert!(
                    err < 1e-4,
                    "{f:?} m={m} {v:?}: packed linear differs from CPU by {err}"
                );
            }
        }
    }

    #[test]
    fn megakernel_mlp_matches_cpu_for_every_variant() {
        use crate::model::layers::gguf_block_codec;
        for f in [
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_K_M,
            QuantFormat::GGUF_Q8_0,
        ] {
            for post_norm in [false, true] {
                let (hidden, inter) = (256usize, 512usize);
                let mut seed = 0x1234_5678_9ABC_DEF1u64 ^ f as u64;
                let mut next = move || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed >> 33) as u32
                };
                let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
                let scale_at: &[usize] = match f {
                    QuantFormat::GGUF_Q4_K_M => &[0, 2],
                    _ => &[0],
                };
                let mut w = Weights::new();
                let mut g = Graph::new();
                let mut quant = |g: &mut Graph, w: &mut Weights, name: &str, n: usize, k: usize| {
                    let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                        .map(|_| (next() >> 4) as u8)
                        .collect();
                    for blk in raw.chunks_exact_mut(bb) {
                        for &o in scale_at {
                            let exp = 9 + (next() % 14) as u16;
                            let d =
                                ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
                            blk[o..o + 2].copy_from_slice(&d.to_le_bytes());
                        }
                    }
                    w.insert(
                        name,
                        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
                    );
                    g.weight(name, vec![n, k], DType::Quant(f))
                };
                let x = g.input("x", vec![1, hidden], DType::F32);
                let norm_w = g.weight("norm.weight", vec![hidden], DType::F32);
                w.insert("norm.weight", Tensor::F32(vec![1.0; hidden]));
                let normed = g.add(
                    "norm",
                    Op::RmsNorm { eps: 1e-5 },
                    &[x, norm_w],
                    vec![1, hidden],
                    DType::F32,
                );
                let gw = quant(&mut g, &mut w, "gate.weight", inter, hidden);
                let uw = quant(&mut g, &mut w, "up.weight", inter, hidden);
                let dw = quant(&mut g, &mut w, "down.weight", hidden, inter);
                let gate = g.add(
                    "gate",
                    Op::Linear,
                    &[normed, gw],
                    vec![1, inter],
                    DType::F32,
                );
                let up = g.add("up", Op::Linear, &[normed, uw], vec![1, inter], DType::F32);
                let act = g.add(
                    "swiglu",
                    Op::SwiGlu,
                    &[gate, up],
                    vec![1, inter],
                    DType::F32,
                );
                let mut down = g.add("down", Op::Linear, &[act, dw], vec![1, hidden], DType::F32);
                if post_norm {
                    let pn = g.weight("post_norm.weight", vec![hidden], DType::F32);
                    w.insert("post_norm.weight", Tensor::F32(vec![0.9; hidden]));
                    down = g.add(
                        "post_norm",
                        Op::RmsNorm { eps: 1e-5 },
                        &[down, pn],
                        vec![1, hidden],
                        DType::F32,
                    );
                }
                let out = g.add(
                    "add",
                    Op::Binary {
                        kind: crate::graph::BinaryKind::Add,
                    },
                    &[x, down],
                    vec![1, hidden],
                    DType::F32,
                );
                g.output(out);
                let unfused = g.clone();
                assert_eq!(
                    crate::graph::egraph::fuse_rule(
                        &mut g,
                        None,
                        crate::graph::egraph::Rule::MegakernelMlp
                    ),
                    1
                );
                let xs: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
                let want = cpu::run_stateless(&unfused, &w, vec![Tensor::F32(xs.clone())])
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                for v in WeightFormat::of_quant(f).gemv_variants() {
                    let mut s = Session::new(rt(), &w).unwrap();
                    s.set_variant_override(Some(v));
                    let got = s
                        .run(&g, vec![Tensor::F32(xs.clone())])
                        .unwrap()
                        .remove(0)
                        .into_f32()
                        .unwrap();
                    let err = rel_err(&want, &got);
                    assert!(
                        err < 1e-3,
                        "{f:?} post_norm={post_norm} {v:?}: megakernel MLP differs from CPU by {err}"
                    );
                }
            }
        }
    }

    /// TQ2_0 lives in the planar device layout (groups of 8 blocks): GEMV (m = 1, every variant), the
    /// dequantizing batched path (m = 20) equal the CPU executor at shapes where rows are not whole groups
    /// (5, 9 blocks per row), n is not a multiple of the rows per threadgroup and the total block count is
    /// not a multiple of 8.
    #[test]
    fn tq2_0_planar_layout_matches_cpu() {
        let f = QuantFormat::GGUF_TQ2_0;
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for (n, k) in [(37usize, 1280usize), (101, 2304), (19, 2048), (5, 768)] {
            let mut raw: Vec<u8> = (0..n * (k / 256) * 66)
                .map(|_| (next() >> 4) as u8)
                .collect();
            for blk in raw.chunks_exact_mut(66) {
                // Finite f16 scale in 2^-6..2^4.
                let exp = 9 + (next() % 14) as u16;
                let d = ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
                blk[64..].copy_from_slice(&d.to_le_bytes());
            }
            let mut w = Weights::new();
            w.insert(
                "w",
                Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
            );
            for m in [1usize, 20] {
                let mut g = Graph::new();
                let x = g.input("x", vec![m, k], DType::F32);
                let w_val = g.weight("w", vec![n, k], DType::Quant(f));
                let y = g.add("lin", Op::Linear, &[x, w_val], vec![m, n], DType::F32);
                g.output(y);
                let xs: Vec<f32> = (0..m * k)
                    .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
                    .collect();
                let want = cpu::run_stateless(&g, &w, vec![Tensor::F32(xs.clone())])
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                for v in WeightFormat::of_quant(f).gemv_variants() {
                    let mut s = Session::new(rt(), &w).unwrap();
                    s.set_variant_override(Some(v));
                    let got = s
                        .run(&g, vec![Tensor::F32(xs.clone())])
                        .unwrap()
                        .remove(0)
                        .into_f32()
                        .unwrap();
                    let err = rel_err(&want, &got);
                    assert!(
                        err < 1e-4,
                        "TQ2_0 {n}x{k} m={m} {v:?}: differs from CPU by {err}"
                    );
                }
            }
        }
    }

    /// Generated-GEMV bandwidth at lm_head-like shapes (`cargo test --release -- --ignored --nocapture`).
    #[test]
    #[ignore = "bandwidth measurement"]
    fn generated_gemv_bandwidth() {
        use crate::model::layers::gguf_block_codec;
        for f in [
            QuantFormat::GGUF_TQ2_0,
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q8_0,
            QuantFormat::GGUF_Q6_K,
            QuantFormat::GGUF_Q5_K_M,
        ] {
            let (be, bb, _) = gguf_block_codec(f).unwrap();
            for (n, k) in [
                (262144usize, 1536usize),
                (6144, 1536),
                (1536, 6144),
                (14336, 4096),
            ] {
                let raw = vec![0u8; n * (k / be) * bb];
                let bytes = raw.len();
                let mut g = Graph::new();
                let x = g.input("x", vec![1, k], DType::F32);
                let wv = g.weight("w", vec![n, k], DType::Quant(f));
                let y = g.add("lin", Op::Linear, &[x, wv], vec![1, n], DType::F32);
                g.output(y);
                let mut w = Weights::new();
                w.insert(
                    "w",
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
                );
                let s = Session::new(rt(), &w).unwrap();
                let xs = Tensor::F32(vec![0.5; k]);
                s.run(&g, vec![xs.clone()]).unwrap();
                // GPU-side time of the command buffer (host copy-in/out excluded).
                let mut dt = 0.0;
                for _ in 0..20 {
                    s.run(&g, vec![xs.clone()]).unwrap();
                    dt += s.last_gpu_seconds() / 20.0;
                }
                println!(
                    "{f:?} {n}x{k}: {:.0} us, {:.1} GB/s",
                    dt * 1e6,
                    bytes as f64 / dt / 1e9
                );
            }
        }
    }

    /// Best GEMV rate per format at the projection shapes of a ~2B model, DRAM-streaming conditions
    /// (`cargo test --release --lib shape_rates -- --ignored --nocapture`; `SHAPE_RATES_FMT=TQ2_0`
    /// keeps only the formats whose name contains the text).
    #[test]
    #[ignore = "bandwidth measurement"]
    fn shape_rates() {
        for f in [
            QuantFormat::GGUF_TQ2_0,
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_K_M,
            QuantFormat::GGUF_Q6_K,
            QuantFormat::GGUF_Q8_0,
            QuantFormat::GGUF_IQ4_XS,
            QuantFormat::GGUF_IQ2_XXS,
            QuantFormat::GGUF_Q5_K_M,
            QuantFormat::GGUF_Q3_K,
            QuantFormat::GGUF_IQ4_NL,
        ] {
            if std::env::var("SHAPE_RATES_FMT").is_ok_and(|s| !format!("{f:?}").contains(&s)) {
                continue;
            }
            let fmt = WeightFormat::of_quant(f);
            for shape in [
                (6144usize, 1536usize),
                (1536, 6144),
                (2048, 1536),
                (1536, 2048),
                (1536, 2816),
                (2816, 768),
                (262144, 1536),
                (262144, 2816),
            ] {
                let t = rt().bench_gemv(f, &[shape], &fmt.gemv_variants()).unwrap();
                let best = t.iter().map(|x| x.1).fold(0.0, f64::max);
                println!(
                    "{f:?} {shape:?}: best {best:.0} GB/s, {} variants {:?}",
                    t.len(),
                    t.iter()
                        .map(|x| (x.0.label(), x.1 as u32))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// Linear -> Silu/SwiGlu/GeGlu on a BF16 weight at m below and above the batched-GEMM cut (16):
    /// the activation must never be dropped.
    #[test]
    fn packed_linear_activation_epilogues_match_cpu_at_every_m() {
        let (n, k) = (40usize, 64usize);
        let bf: Vec<u16> = (0..n * k)
            .map(|i| crate::model::f32_to_bf16(0.05 * ((i % 11) as f32 - 5.0)))
            .collect();
        for m in [1usize, 15, 16, 64] {
            for act in 0..3 {
                let mut g = Graph::new();
                let x = g.input("x", vec![m, k], DType::F32);
                let up = g.input("up", vec![m, n], DType::F32);
                let wv = g.weight("w", vec![n, k], DType::BF16);
                let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], DType::F32);
                let y = match act {
                    0 => g.add(
                        "a",
                        Op::Unary {
                            kind: crate::graph::UnaryKind::Silu,
                        },
                        &[lin],
                        vec![m, n],
                        DType::F32,
                    ),
                    1 => g.add("a", Op::SwiGlu, &[lin, up], vec![m, n], DType::F32),
                    _ => g.add("a", Op::GeGlu, &[lin, up], vec![m, n], DType::F32),
                };
                g.output(y);
                let mut w = Weights::new();
                w.insert("w", Tensor::BF16(bf.clone()));
                let inputs = vec![Tensor::F32(rand(m * k, 7)), Tensor::F32(rand(m * n, 8))];
                let (want, got) = both(&g, &w, inputs, None);
                let err = rel_err(&want, &got);
                assert!(err < 1e-3, "m={m} act={act}: rel err {err}");
            }
        }
    }

    #[test]
    fn gemv_q8_0_matches_cpu() {
        let (m, n, k) = (3, 37, 32 * 40);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q8_0));
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);

        let n_blocks = n * (k / 32);
        let mut raw = Vec::with_capacity(n_blocks * 34);
        for b in 0..n_blocks {
            let d = crate::model::f32_to_f16(0.125 * ((b % 5) as f32 + 1.0));
            raw.extend_from_slice(&d.to_le_bytes());
            for j in 0..32 {
                let q = (((j as i32 * 7 + b as i32 * 3) % 255) - 127) as i8;
                raw.push(q as u8);
            }
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                QuantFormat::GGUF_Q8_0,
                raw,
                n,
                k,
            )),
        );
        w.insert("b", Tensor::F32(rand(n, 17)));
        let inputs = vec![Tensor::F32(rand(m * k, 42))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }

    #[test]
    fn gemv_q4_0_matches_cpu() {
        let (m, n, k) = (2, 37, 32 * 40);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q4_0));
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);

        let n_blocks = n * (k / 32);
        let mut raw = Vec::with_capacity(n_blocks * 18);
        for b in 0..n_blocks {
            let d = crate::model::f32_to_f16(0.25 * ((b % 3) as f32 + 1.0));
            raw.extend_from_slice(&d.to_le_bytes());
            for j in 0..16 {
                let low = (j * 3 + b) % 16;
                let high = (j * 5 + b * 2) % 16;
                raw.push((low | (high << 4)) as u8);
            }
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                QuantFormat::GGUF_Q4_0,
                raw,
                n,
                k,
            )),
        );
        w.insert("b", Tensor::F32(rand(n, 23)));
        let inputs = vec![Tensor::F32(rand(m * k, 99))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }
    #[test]
    fn gemv_q4_0_residual_matches_cpu() {
        let (m, n, k) = (1, 37, 32 * 40);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let res_in = g.input("res", vec![m, n], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q4_0));
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        let y_res = g.add(
            "res_add",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[y, res_in],
            vec![m, n],
            DType::F32,
        );
        g.output(y_res);

        let n_blocks = n * (k / 32);
        let mut raw = Vec::with_capacity(n_blocks * 18);
        for b in 0..n_blocks {
            let d = crate::model::f32_to_f16(0.25 * ((b % 3) as f32 + 1.0));
            raw.extend_from_slice(&d.to_le_bytes());
            for j in 0..16 {
                let low = (j * 3 + b) % 16;
                let high = (j * 5 + b * 2) % 16;
                raw.push((low | (high << 4)) as u8);
            }
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                QuantFormat::GGUF_Q4_0,
                raw,
                n,
                k,
            )),
        );
        w.insert("b", Tensor::F32(rand(n, 23)));
        let inputs = vec![Tensor::F32(rand(m * k, 99)), Tensor::F32(rand(m * n, 77))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }

    #[test]
    fn gemv_iq4_xs_matches_cpu() {
        let (m, n, k) = (1, 16, 256 * 2);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_IQ4_XS));
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);

        let n_blocks = n * (k / 256);
        let mut raw = Vec::with_capacity(n_blocks * 136);
        for b in 0..n_blocks {
            let d = crate::model::f32_to_f16(0.0625 * ((b % 5) as f32 + 1.0));
            raw.extend_from_slice(&d.to_le_bytes());
            let scales_h = (b as u16 * 17) & 0xFFFF;
            raw.extend_from_slice(&scales_h.to_le_bytes());
            for j in 0..4 {
                raw.push(((j * 7 + b) % 256) as u8);
            }
            for j in 0..128 {
                let low = (j * 3 + b) % 16;
                let high = (j * 5 + b * 2) % 16;
                raw.push((low | (high << 4)) as u8);
            }
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                QuantFormat::GGUF_IQ4_XS,
                raw,
                n,
                k,
            )),
        );
        w.insert("b", Tensor::F32(rand(n, 19)));
        let inputs = vec![Tensor::F32(rand(m * k, 43))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }

    #[test]
    fn gemv_iq2_m_matches_cpu() {
        let (m, n, k) = (1, 16, 256 * 2);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_IQ2_M));
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);

        let n_blocks = n * (k / 256);
        let mut raw = Vec::with_capacity(n_blocks * 82);
        for b in 0..n_blocks {
            let d = crate::model::f32_to_f16(0.0625 * ((b % 5) as f32 + 1.0));
            raw.extend_from_slice(&d.to_le_bytes());
            for j in 0..32 {
                raw.push(((j * 3 + b) % 256) as u8);
            }
            for j in 0..32 {
                raw.push(((j * 5 + b) % 256) as u8);
            }
            for j in 0..8 {
                raw.push(((j * 7 + b) % 256) as u8);
            }
            for j in 0..8 {
                raw.push(((j * 11 + b) % 256) as u8);
            }
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                QuantFormat::GGUF_IQ2_M,
                raw,
                n,
                k,
            )),
        );
        w.insert("b", Tensor::F32(rand(n, 23)));
        let inputs = vec![Tensor::F32(rand(m * k, 45))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }

    #[test]
    fn gemv_nvfp4_matches_cpu() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let rt = rt();
        let (rows, cols) = (16usize, 64usize);
        let n_blocks = rows * (cols / 16);
        let byte_stride = cols / 2;

        let mut wq = vec![0u8; rows * byte_stride];
        for (i, b) in wq.iter_mut().enumerate() {
            let low = (i * 3 + 1) % 16;
            let high = (i * 5 + 2) % 16;
            *b = (low | (high << 4)) as u8;
        }

        let mut scales = vec![0u8; n_blocks];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 7 + 0x30) % 0x7E) as u8;
        }
        let global_scale = 1.5f32;
        let x_data = rand(cols, 42);

        let lut = crate::quant::nvfp4::NVFP4_E2M1_LUT;
        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for blk in 0..(cols / 16) {
                let s = crate::model::f8_e4m3_to_f32(scales[r * (cols / 16) + blk]) * global_scale;
                for i in 0..8 {
                    let byte_val = wq[r * byte_stride + blk * 8 + i];
                    let v0 = lut[(byte_val & 0x0F) as usize] * s;
                    let v1 = lut[(byte_val >> 4) as usize] * s;
                    let c0 = blk * 16 + 2 * i;
                    let c1 = blk * 16 + 2 * i + 1;
                    acc += v0 * x_data[c0] + v1 * x_data[c1];
                }
            }
            expected[r] = acc;
        }

        let wq_buf = new_shared_buffer(&rt.device, wq.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(wq.as_ptr(), wq_buf.contents() as *mut u8, wq.len())
        };

        let scales_buf = new_shared_buffer(&rt.device, scales.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                scales.as_ptr(),
                scales_buf.contents() as *mut u8,
                scales.len(),
            )
        };

        let x_buf = new_shared_buffer(&rt.device, cols * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x_data.as_ptr(), x_buf.contents() as *mut f32, cols)
        };

        let y_buf = new_shared_buffer(&rt.device, rows * 4).unwrap();

        metal::objc::rc::autoreleasepool(|| {
            let cb = rt.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(rt.pso("gemv_nvfp4").unwrap());
            enc.set_buffer(0, Some(&wq_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            #[repr(C)]
            struct PackedGemvParams {
                rows: u32,
                cols: u32,
                batch_size: u32,
                has_bias: u32,
                epi: u32,
            }
            let p = PackedGemvParams {
                rows: rows as u32,
                cols: cols as u32,
                batch_size: 1,
                has_bias: 0,
                epi: 0,
            };
            set_bytes(enc, 3, &p);
            enc.set_buffer(4, Some(&y_buf), 0);
            enc.set_buffer(5, Some(&y_buf), 0);
            enc.set_buffer(6, Some(&scales_buf), 0);
            set_bytes(enc, 7, &global_scale);

            enc.dispatch_thread_groups(
                MTLSize::new((rows as u64).div_ceil(16), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            commit_and_wait(cb).unwrap();
        });

        let mut got = vec![0.0f32; rows];
        unsafe {
            std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, got.as_mut_ptr(), rows)
        };

        for (r, (&e, &g)) in expected.iter().zip(&got).enumerate() {
            assert!(
                (e - g).abs() < 1e-4 || (e - g).abs() / (e.abs().max(g.abs())) < 1e-4,
                "row {r}: expected {e}, got {g}"
            );
        }
    }

    #[test]
    fn gemv_bf16_matches_cpu() {
        let (m, n, k) = (4, 37, 1000);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::BF16);
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);

        let f32_vals = rand(n * k, 77);
        let bf16_vals: Vec<u16> = f32_vals
            .iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();
        let mut w = Weights::new();
        w.insert("w", Tensor::BF16(bf16_vals));
        w.insert("b", Tensor::F32(rand(n, 31)));
        let inputs = vec![Tensor::F32(rand(m * k, 123))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }
    #[test]
    fn gemv_f16_matches_cpu() {
        let (m, n, k) = (3, 37, 1000);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::F16);
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);

        let f32_vals = rand(n * k, 77);
        let f16_vals: Vec<u16> = f32_vals
            .iter()
            .map(|&v| crate::model::f32_to_f16(v))
            .collect();
        let mut w = Weights::new();
        w.insert("w", Tensor::F16(f16_vals));
        w.insert("b", Tensor::F32(rand(n, 31)));
        let inputs = vec![Tensor::F32(rand(m * k, 123))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "want {:?} vs got {:?}",
            &want[..4],
            &got[..4]
        );
    }
    #[test]
    #[ignore = "benchmark: measures hardware roofline saturation, run via cargo test --lib test_tiled_gemv_roofline_saturation -- --ignored --test-threads=1"]
    fn test_tiled_gemv_roofline_saturation() {
        use metal::MTLResourceOptions;
        let rt = Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir()).unwrap();
        let roofline_gbps = rt.benchmark_memory_bandwidth().unwrap_or(298.0);
        println!(
            "Hardware streaming read roofline (memory_sweep): {:.2} GB/s",
            roofline_gbps
        );

        #[repr(C)]
        struct PackedGemvParams(u32, u32, u32, u32, u32);

        // 1. Q4_0: rows = 16384, cols = 8192 (~75.5 MB)
        {
            let (rows, cols) = (16384usize, 8192usize);
            let n_blocks = rows * (cols / 32);
            let bytes = n_blocks * 18;
            let w_buf = rt
                .device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let x_buf = rt
                .device
                .new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
            let y_buf = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
            let pso = rt.pso("gemv_q4_0").unwrap();
            let p = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);

            let us = rt
                .timed(20, |enc| {
                    enc.set_compute_pipeline_state(pso);
                    enc.set_buffer(0, Some(&w_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&y_buf), 0);
                    set_bytes(enc, 3, &p);
                    enc.set_buffer(4, Some(&y_buf), 0);
                    enc.set_buffer(5, Some(&y_buf), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new((rows as u64).div_ceil(16), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Ok(())
                })
                .unwrap();

            let gb_per_sec = (bytes as f64 / 1e9) / (us * 1e-6);
            println!(
                "Tiled GEMV Q4_0: {:.2} GB/s ({:.1}% roofline)",
                gb_per_sec,
                (gb_per_sec / roofline_gbps) * 100.0
            );
            let target_pct = 0.85;
            assert!(
                gb_per_sec > target_pct * roofline_gbps,
                "Q4_0 bandwidth {:.2} GB/s should reach at least {:.0}% of measured roofline ({:.2} GB/s)",
                gb_per_sec, target_pct * 100.0, target_pct * roofline_gbps
            );
        }

        // 2. Q8_0: rows = 8192, cols = 8192 (~71.3 MB)
        {
            let (rows, cols) = (8192usize, 8192usize);
            let n_blocks = rows * (cols / 32);
            let bytes = n_blocks * 34;
            let w_buf = rt
                .device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let x_buf = rt
                .device
                .new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
            let y_buf = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
            let pso = rt.pso("gemv_q8_0").unwrap();
            let p = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);

            let us = rt
                .timed(20, |enc| {
                    enc.set_compute_pipeline_state(pso);
                    enc.set_buffer(0, Some(&w_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&y_buf), 0);
                    set_bytes(enc, 3, &p);
                    enc.set_buffer(4, Some(&y_buf), 0);
                    enc.set_buffer(5, Some(&y_buf), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new((rows as u64).div_ceil(8), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Ok(())
                })
                .unwrap();

            let gb_per_sec = (bytes as f64 / 1e9) / (us * 1e-6);
            println!(
                "Tiled GEMV Q8_0: {:.2} GB/s ({:.1}% roofline)",
                gb_per_sec,
                (gb_per_sec / roofline_gbps) * 100.0
            );
            let target_pct = 0.85;
            assert!(
                gb_per_sec > target_pct * roofline_gbps,
                "Q8_0 bandwidth {:.2} GB/s should reach at least {:.0}% of measured roofline ({:.2} GB/s)",
                gb_per_sec, target_pct * 100.0, target_pct * roofline_gbps
            );
        }

        // 3. BF16: rows = 8192, cols = 4096 (~67.1 MB)
        {
            let (rows, cols) = (8192usize, 4096usize);
            let bytes = rows * cols * 2;
            let w_buf = rt
                .device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let x_buf = rt
                .device
                .new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
            let y_buf = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
            let pso = rt.pso("gemv_bf16").unwrap();
            let p = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);

            let us = rt
                .timed(20, |enc| {
                    enc.set_compute_pipeline_state(pso);
                    enc.set_buffer(0, Some(&w_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&y_buf), 0);
                    set_bytes(enc, 3, &p);
                    enc.set_buffer(4, Some(&y_buf), 0);
                    enc.set_buffer(5, Some(&y_buf), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new((rows as u64).div_ceil(16), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Ok(())
                })
                .unwrap();

            let gb_per_sec = (bytes as f64 / 1e9) / (us * 1e-6);
            println!(
                "Tiled GEMV BF16: {:.2} GB/s ({:.1}% roofline)",
                gb_per_sec,
                (gb_per_sec / roofline_gbps) * 100.0
            );
            let target_pct = 0.85;
            assert!(
                gb_per_sec > target_pct * roofline_gbps,
                "BF16 bandwidth {:.2} GB/s should reach at least {:.0}% of measured roofline ({:.2} GB/s)",
                gb_per_sec, target_pct * 100.0, target_pct * roofline_gbps
            );
        }
    }

    #[test]
    fn gemm_batched_packed_matches_cpu() {
        let (m, n, k) = (32, 64, 64);
        // 1. BF16
        {
            let mut g = Graph::new();
            let x = g.input("x", vec![m, k], DType::F32);
            let w_val = g.weight("w", vec![n, k], DType::BF16);
            let b_val = g.weight("b", vec![n], DType::F32);
            let y = g.add(
                "lin",
                Op::Linear,
                &[x, w_val, b_val],
                vec![m, n],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            let raw_w: Vec<u16> = rand(n * k, 123)
                .into_iter()
                .map(crate::model::f32_to_bf16)
                .collect();
            w.insert("w", Tensor::BF16(raw_w));
            w.insert("b", Tensor::F32(rand(n, 456)));
            let inputs = vec![Tensor::F32(rand(m * k, 789))];
            let (want, got) = both(&g, &w, inputs, None);
            assert!(rel_err(&want, &got) < 1e-3, "BF16 batched mismatch");
        }
        // 2. Q8_0
        {
            let mut g = Graph::new();
            let x = g.input("x", vec![m, k], DType::F32);
            let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q8_0));
            let y = g.add("lin", Op::Linear, &[x, w_val], vec![m, n], DType::F32);
            g.output(y);
            let n_blocks = n * (k / 32);
            let mut raw = Vec::with_capacity(n_blocks * 34);
            for b in 0..n_blocks {
                let d = crate::model::f32_to_f16(0.125 * ((b % 5) as f32 + 1.0));
                raw.extend_from_slice(&d.to_le_bytes());
                for j in 0..32 {
                    let q = (((j as i32 * 7 + b as i32 * 3) % 255) - 127) as i8;
                    raw.push(q as u8);
                }
            }
            let mut w = Weights::new();
            w.insert(
                "w",
                Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                    QuantFormat::GGUF_Q8_0,
                    raw,
                    n,
                    k,
                )),
            );
            let inputs = vec![Tensor::F32(rand(m * k, 999))];
            let (want, got) = both(&g, &w, inputs, None);
            assert!(rel_err(&want, &got) < 1e-3, "Q8_0 batched mismatch");
        }
    }
    #[test]
    fn test_metal_batched_prefill_throughput() {
        let (n, k) = (576, 576);
        let runtime = rt();

        for &fmt in &[WeightFormat::BF16, WeightFormat::Q4_0] {
            let mut w = Weights::new();
            let dtype = match fmt {
                WeightFormat::BF16 => {
                    let raw_w: Vec<u16> = rand(n * k, 42)
                        .into_iter()
                        .map(crate::model::f32_to_bf16)
                        .collect();
                    w.insert("w", Tensor::BF16(raw_w));
                    DType::BF16
                }
                WeightFormat::Q4_0 => {
                    let n_blocks = n * (k / 32);
                    let mut raw = Vec::with_capacity(n_blocks * 18);
                    for b in 0..n_blocks {
                        let d = crate::model::f32_to_f16(0.125 * ((b % 5) as f32 + 1.0));
                        raw.extend_from_slice(&d.to_le_bytes());
                        for _ in 0..16 {
                            raw.push(0x88);
                        }
                    }
                    w.insert(
                        "w",
                        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                            QuantFormat::GGUF_Q4_0,
                            raw,
                            n,
                            k,
                        )),
                    );
                    DType::Quant(QuantFormat::GGUF_Q4_0)
                }
                _ => unreachable!(),
            };
            let session = Session::new(runtime.clone(), &w).unwrap();

            for &m in &[512, 1024, 2048] {
                let mut g = Graph::new();
                let x = g.input("x", vec![m, k], DType::F32);
                let w_val = g.weight("w", vec![n, k], dtype);
                let y = g.add("lin", Op::Linear, &[x, w_val], vec![m, n], DType::F32);
                g.output(y);

                let inp = vec![Tensor::F32(vec![0.1f32; m * k])];

                // Warmup
                let _ = session.run(&g, inp.clone()).unwrap();

                // Benchmark
                let iters = 5;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let _ = session.run(&g, inp.clone()).unwrap();
                }
                let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
                let throughput = m as f64 / elapsed;

                println!(
                    "Batched Prefill [{:?}] M={:<4} N={:<4} K={:<4}: {:.2} ms | Throughput: {:>8.1} tok/s",
                    fmt, m, n, k, elapsed * 1000.0, throughput
                );
                assert!(
                    throughput > 100.0,
                    "Throughput {:.1} tok/s must be > 100 tok/s for M={m} ({:?})",
                    throughput,
                    fmt
                );
            }
        }
    }

    #[test]
    fn unsupported_ops_bad_ids_and_missing_weights_are_errors() {
        let f = DType::F32;
        let s = Session::new(rt(), &Weights::new()).unwrap();
        // Relu lowers at F32 only; a u32 value has no kernel.
        for op in [Op::Unary {
            kind: crate::graph::UnaryKind::Relu,
        }] {
            let mut g = Graph::new();
            let x = g.input("x", vec![4], DType::U32);
            let y = g.add("act", op, &[x], vec![4], DType::U32);
            g.output(y);
            let e = s.run(&g, vec![Tensor::F32(vec![0.0; 4])]).unwrap_err();
            assert!(e.contains("the Metal kernels take f32"), "{e}");
        }
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 2], f);
        let wv = g.weight("absent", vec![3, 2], f);
        let y = g.add("lin", Op::Linear, &[x, wv], vec![1, 3], f);
        g.output(y);
        assert!(s
            .run(&g, vec![Tensor::F32(vec![0.0; 2])])
            .unwrap_err()
            .contains("missing weight 'absent'"));

        let mut w = Weights::new();
        w.insert("t", Tensor::F32(vec![0.0; 6]));
        let s = Session::new(rt(), &w).unwrap();
        let mut g = Graph::new();
        let ids = g.input("ids", vec![2], DType::U32);
        let t = g.weight("t", vec![3, 2], f);
        let e = g.add("emb", Op::Embedding, &[t, ids], vec![2, 2], f);
        g.output(e);
        let err = s.run(&g, vec![Tensor::U32(vec![0, 3])]).unwrap_err();
        assert!(err.contains("id 3 out of range"), "{err}");

        let mut g = Graph::new();
        let a = g.input("a", vec![2], DType::U32);
        let b = g.input("b", vec![2], f);
        let y = g.add(
            "sum",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, b],
            vec![2],
            f,
        );
        g.output(y);
        let err = s
            .run(&g, vec![Tensor::U32(vec![1, 2]), Tensor::F32(vec![0.0; 2])])
            .unwrap_err();
        assert!(err.contains("the Metal kernels take f32"), "{err}");
    }

    #[test]
    fn second_load_restores_pipelines_and_profile_from_cache() {
        let dir = tempfile::tempdir().unwrap();
        let first = Runtime::new(dir.path()).unwrap();
        assert!(!first.archive_hit, "{}", first.report);
        let second = Runtime::new(dir.path()).unwrap();
        assert!(second.archive_hit, "{}", second.report);
        assert!(
            second.report.contains("host profile from"),
            "{}",
            second.report
        );
    }

    #[test]
    fn test_dynamic_roofline_scan() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = Runtime::new(dir.path()).unwrap();
        let profile = runtime.profile();
        assert!(
            profile.memory_bandwidth_gbps > 0.0,
            "measured bandwidth must be > 0: got {}",
            profile.memory_bandwidth_gbps
        );

        // For Gemma 4 E2B Q4_0 (~1.44 GB = 1_440_000_000 bytes):
        let gemma4_e2b_bytes = 1_440_000_000u64;
        let roofline = profile.roofline_tok_s(gemma4_e2b_bytes);
        println!("Host Metal GPU: {}", profile.device);
        println!(
            "Measured streaming bandwidth: {:.2} GB/s",
            profile.memory_bandwidth_gbps
        );
        println!("Gemma 4 E2B Q4_0 Roofline: {:.1} tok/s", roofline);
        assert!(
            roofline > 0.0,
            "roofline rate must be dynamic and > 0, got {roofline}"
        );
        let expected_roofline = profile.memory_bandwidth_gbps / 1.44;
        assert!((roofline - expected_roofline).abs() < 1e-4);
    }
    #[test]
    fn test_attention_and_state_ops_match_cpu() {
        // 1. Slice
        {
            let mut g = Graph::new();
            let x = g.input("x", vec![2, 8], DType::F32);
            let y = g.add(
                "sl",
                Op::Slice {
                    axis: -1,
                    start: 2,
                    end: 6,
                },
                &[x],
                vec![2, 4],
                DType::F32,
            );
            g.output(y);
            let inp = vec![Tensor::F32(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0,
            ])];
            let (want, got) = both(&g, &Weights::new(), inp, None);
            assert_eq!(want, got);
        }
        // 2. Rope (Interleaved and HalfSplit)
        for variant in [
            crate::graph::RopeVariant::Interleaved,
            crate::graph::RopeVariant::HalfSplit,
        ] {
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![1, 8], DType::F32);
            let pos = g.input("pos", vec![1], DType::U32);
            let inv_freq = g.weight("rope.inv_freq.default", vec![4], DType::F32);
            w.insert(
                "rope.inv_freq.default".to_string(),
                Tensor::F32(vec![1.0, 0.1, 0.01, 0.001]),
            );
            let y = g.add(
                "rope",
                Op::Rope {
                    head_dim: 8,
                    rotary_dim: 8,
                    variant,
                    mscale: 1.0,
                },
                &[x, pos, inv_freq],
                vec![1, 8],
                DType::F32,
            );
            g.output(y);
            let inp = vec![
                Tensor::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                Tensor::U32(vec![3]),
            ];
            let (want, got) = both(&g, &w, inp, None);
            assert!(
                rel_err(&want, &got) < 1e-4,
                "Rope mismatch: {want:?} vs {got:?}"
            );
        }

        // 3. CachedAttention
        {
            let mut g = Graph::new();
            let q = g.input("q", vec![1, 4], DType::F32);
            let k = g.input("k", vec![1, 4], DType::F32);
            let v = g.input("v", vec![1, 4], DType::F32);
            let y = g.add(
                "attn",
                Op::CachedAttention {
                    heads: 1,
                    kv_heads: 1,
                    head_dim: 4,
                    v_head_dim: 4,
                    scale: 0.5,
                    causal: true,
                    window: None,
                    soft_cap: None,
                    write: false,
                },
                &[q, k, v],
                vec![1, 4],
                DType::F32,
            );
            g.output(y);
            let inp = vec![
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![0.5, 1.5, 2.5, 3.5]),
            ];
            let (want, got) = both(&g, &Weights::new(), inp, None);
            assert!(rel_err(&want, &got) < 1e-4, "CachedAttention mismatch");
        }

        // 3b. PagedCachedAttention
        {
            let mut g = Graph::new();
            let q = g.input("q", vec![1, 4], DType::F32);
            let k_pool = g.input("k_pool", vec![16, 4], DType::F32);
            let v_pool = g.input("v_pool", vec![16, 4], DType::F32);
            let block_table = g.input("block_table", vec![4], DType::U32);
            let pos = g.input("pos", vec![1], DType::U32);
            let y = g.add(
                "paged_attn",
                Op::PagedCachedAttention {
                    heads: 1,
                    kv_heads: 1,
                    head_dim: 4,
                    v_head_dim: 4,
                    scale: 0.5,
                    causal: true,
                    window: None,
                    soft_cap: None,
                    tokens_per_block: 16,
                    max_blocks_per_seq: 4,
                },
                &[q, k_pool, v_pool, block_table, pos],
                vec![1, 4],
                DType::F32,
            );
            g.output(y);
            let mut k_data = vec![0.0f32; 16 * 4];
            let mut v_data = vec![0.0f32; 16 * 4];
            k_data[0..4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            v_data[0..4].copy_from_slice(&[0.5, 1.5, 2.5, 3.5]);
            let inp = vec![
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(k_data),
                Tensor::F32(v_data),
                Tensor::U32(vec![0, 0, 0, 0]),
                Tensor::U32(vec![0]),
            ];
            let (want, got) = both(&g, &Weights::new(), inp, None);
            assert!(rel_err(&want, &got) < 1e-4, "PagedCachedAttention mismatch");
        }

        // 4. CausalConv1dCached
        {
            let mut g = Graph::new();
            let x = g.input("x", vec![3, 1], DType::F32);
            let w_val = g.weight("w", vec![2], DType::F32);
            let y = g.add(
                "conv",
                Op::CausalConv1dCached {
                    kernel: 2,
                    dilation: 1,
                },
                &[x, w_val],
                vec![3, 1],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert("w", Tensor::F32(vec![0.5, 1.0]));
            let inp = vec![Tensor::F32(vec![2.0, 4.0, 6.0])];
            let (want, got) = both(&g, &w, inp, None);
            assert!(
                rel_err(&want, &got) < 1e-4,
                "Conv1d mismatch: {want:?} vs {got:?}"
            );
        }

        // 5. DeltaRule
        {
            let mut g = Graph::new();
            let q = g.input("q", vec![1, 2], DType::F32);
            let k = g.input("k", vec![1, 2], DType::F32);
            let v = g.input("v", vec![1, 2], DType::F32);
            let beta = g.input("beta", vec![1, 1], DType::F32);
            let log_decay = g.input("log_decay", vec![1], DType::F32);
            let y = g.add(
                "delta",
                Op::DeltaRule {
                    key_heads: 1,
                    val_heads: 1,
                    key_dim: 2,
                    val_dim: 2,
                },
                &[q, k, v, beta, log_decay],
                vec![1, 2],
                DType::F32,
            );
            g.output(y);
            let inp = vec![
                Tensor::F32(vec![1.0, 0.5]),
                Tensor::F32(vec![0.2, 0.8]),
                Tensor::F32(vec![0.3, -0.4]),
                Tensor::F32(vec![0.9]),
                Tensor::F32(vec![-0.1]),
            ];
            let (want, got) = both(&g, &Weights::new(), inp, None);
            assert!(
                rel_err(&want, &got) < 1e-4,
                "DeltaRule mismatch: {want:?} vs {got:?}"
            );
        }
    }
    #[test]
    fn test_zero_copy_slice_aliasing_m1_matches_cpu() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 64], DType::F32);
        let w_large = g.weight("w_large", vec![192, 64], DType::F32);
        let lin = g.add("lin", Op::Linear, &[x, w_large], vec![1, 192], DType::F32);

        // 3-way slice
        let v0 = g.add(
            "sl0",
            Op::Slice {
                axis: -1,
                start: 0,
                end: 64,
            },
            &[lin],
            vec![1, 64],
            DType::F32,
        );
        let v1 = g.add(
            "sl1",
            Op::Slice {
                axis: -1,
                start: 64,
                end: 128,
            },
            &[lin],
            vec![1, 64],
            DType::F32,
        );
        let v2 = g.add(
            "sl2",
            Op::Slice {
                axis: -1,
                start: 128,
                end: 192,
            },
            &[lin],
            vec![1, 64],
            DType::F32,
        );

        // Slice of a slice: check transitive composition
        let v0_sub = g.add(
            "sl0_sub",
            Op::Slice {
                axis: -1,
                start: 16,
                end: 48,
            },
            &[v0],
            vec![1, 32],
            DType::F32,
        );

        // Downstream steps consuming each slice view
        let w1 = g.weight("w1", vec![64, 64], DType::F32);
        let step1 = g.add("step1", Op::Linear, &[v1, w1], vec![1, 64], DType::F32);
        let step2 = g.add(
            "step2",
            Op::Unary {
                kind: crate::graph::UnaryKind::GeluErf,
            },
            &[v2],
            vec![1, 64],
            DType::F32,
        );
        let step3 = g.add(
            "step3",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[step1, step2],
            vec![1, 64],
            DType::F32,
        );

        // Later large allocation to exercise arena reuse
        let w_huge = g.weight("w_huge", vec![1024, 64], DType::F32);
        let huge_step = g.add(
            "huge_step",
            Op::Linear,
            &[step3, w_huge],
            vec![1, 1024],
            DType::F32,
        );

        // Second parent: NOT a graph output! Tests the alias-read lifetime extension directly
        let w_lin2 = g.weight("w_lin2", vec![1024, 64], DType::F32);
        let lin2 = g.add("lin2", Op::Linear, &[x, w_lin2], vec![1, 1024], DType::F32);
        let a = g.add(
            "a",
            Op::Slice {
                axis: -1,
                start: 0,
                end: 64,
            },
            &[lin2],
            vec![1, 64],
            DType::F32,
        );
        let b = g.add(
            "b",
            Op::Slice {
                axis: -1,
                start: 64,
                end: 128,
            },
            &[lin2],
            vec![1, 64],
            DType::F32,
        );

        // Intermediate allocation smaller than lin2: fits in lin2's arena space if lin2 is freed!
        let w_inter = g.weight("w_inter", vec![512, 64], DType::F32);
        let inter = g.add("inter", Op::Linear, &[x, w_inter], vec![1, 512], DType::F32);
        let inter_sub = g.add(
            "inter_sub",
            Op::Slice {
                axis: -1,
                start: 0,
                end: 64,
            },
            &[inter],
            vec![1, 64],
            DType::F32,
        );

        // Consume a and b AFTER the inter allocation
        let add_a = g.add(
            "add_a",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, inter_sub],
            vec![1, 64],
            DType::F32,
        );
        let gelu_b = g.add(
            "gelu_b",
            Op::Unary {
                kind: crate::graph::UnaryKind::GeluErf,
            },
            &[b],
            vec![1, 64],
            DType::F32,
        );

        g.output(v0);
        g.output(v0_sub);
        g.output(huge_step);
        g.output(add_a);
        g.output(gelu_b);

        let mut w = Weights::new();
        w.insert("w_large", Tensor::F32(rand(192 * 64, 11)));
        w.insert("w1", Tensor::F32(rand(64 * 64, 22)));
        w.insert("w_huge", Tensor::F32(rand(1024 * 64, 33)));
        w.insert("w_lin2", Tensor::F32(rand(1024 * 64, 55)));
        w.insert("w_inter", Tensor::F32(rand(512 * 64, 66)));
        let inp = vec![Tensor::F32(rand(64, 44))];
        let want = cpu::run_stateless(&g, &w, inp.clone()).unwrap();
        let s = Session::new(rt(), &w).unwrap();
        let got = s.run(&g, inp).unwrap();
        assert_eq!(want.len(), got.len());
        for (i, (w_t, g_t)) in want.iter().zip(&got).enumerate() {
            let w_slice = w_t.as_f32().unwrap();
            let g_slice = g_t.as_f32().unwrap();
            let diff = rel_err(w_slice, g_slice);
            assert!(
                diff < 1e-4,
                "Output {i} mismatch: want {:?} vs got {:?}, diff = {diff}",
                &w_slice[..4],
                &g_slice[..4]
            );
        }
    }
    #[test]
    fn test_measure_llama_decode_dispatches_per_layer() {
        use crate::arch::expr::Val;
        // The llama spec's decode instance on a synthetic 4-layer checkpoint.
        let num_layers = 4;
        let dir = tempfile::tempdir().unwrap();
        let a = crate::arch::testing::synthetic_llama(
            dir.path(),
            serde_json::json!({"num_hidden_layers": num_layers}),
        );
        let mut r = crate::arch::testing::Runner::from_artifacts(
            &a,
            Some("llama"),
            crate::hardware::Device::Metal,
            512,
        )
        .unwrap();
        let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
        let inst = r.instance("text", &vars).unwrap();
        let weights = r.model.store.materialize(&inst.weights).unwrap();
        let decode_g = inst.graph;

        let s = Session::new(rt(), &weights).unwrap();
        let plan = s.prepare_plan(&decode_g).unwrap();

        let total_steps = plan.steps.len();
        let non_layer_steps = 3;
        let layer_steps = total_steps.saturating_sub(non_layer_steps);
        let steps_per_layer = layer_steps as f64 / num_layers as f64;

        println!("=== llama spec decode dispatch count ===");
        println!("Total graph nodes: {}", decode_g.nodes.len());
        println!("Total GPU dispatches: {}", total_steps);
        println!(
            "Dispatches per layer: {:.1} ({} layers)",
            steps_per_layer, num_layers
        );
        for (i, step) in plan.steps.iter().enumerate() {
            println!("  Step {i:02}: {}", step.name);
        }
        assert!(
            steps_per_layer <= 10.0,
            "Dispatches per layer must be <= 10, got {steps_per_layer}"
        );
    }

    #[test]
    fn test_stateful_cached_attention_parity() {
        for (kv_f16, tol) in KV_DTYPES {
            stateful_cached_attention_parity(kv_f16, tol);
        }
    }

    fn stateful_cached_attention_parity(kv_f16: bool, tol: f32) {
        let (heads, kv_heads, hd) = (4, 2, 8);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let max_seq = 32;
        let scale = 1.0 / (hd as f32).sqrt();
        let window = Some(3);
        let soft_cap = Some(15.0);

        let make_graph = |m: usize| {
            let mut g = Graph::new();
            let q = g.input("q", vec![m, q_dim], DType::F32);
            let k = g.input("k", vec![m, kv_dim], DType::F32);
            let v = g.input("v", vec![m, kv_dim], DType::F32);
            let k_state = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
            let v_state = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
            let pos = g.input("pos", vec![m], DType::U32);
            let out = g.add(
                "cached_attn",
                Op::CachedAttention {
                    heads,
                    kv_heads,
                    head_dim: hd,
                    v_head_dim: hd,
                    scale,
                    causal: true,
                    window,
                    soft_cap,
                    write: true,
                },
                &[q, k, v, k_state, v_state, pos],
                vec![m, q_dim],
                DType::F32,
            );
            g.output(out);
            g
        };

        let prefill_g = make_graph(5);
        let decode_g = make_graph(1);

        let mut cpu_state = StateStore::new();
        let mut metal_state = StateStore::new();
        let mut session = Session::new(rt(), &Weights::new()).unwrap();
        session.set_kv_f16(kv_f16);

        // 1. Prefill chunk of m = 5 (positions 0..5)
        {
            let q_data = rand(5 * q_dim, 101);
            let k_data = rand(5 * kv_dim, 202);
            let v_data = rand(5 * kv_dim, 303);
            let pos_data = vec![0u32, 1, 2, 3, 4];

            let inputs = vec![
                Tensor::F32(q_data),
                Tensor::F32(k_data),
                Tensor::F32(v_data),
                Tensor::U32(pos_data),
            ];

            let cpu_out =
                cpu::run(&prefill_g, &Weights::new(), &mut cpu_state, inputs.clone()).unwrap();
            let metal_out = session
                .run_with_state(&prefill_g, Some(&mut metal_state), inputs)
                .unwrap();

            let cpu_out_f32 = cpu_out[0].as_f32().unwrap();
            let metal_out_f32 = metal_out[0].as_f32().unwrap();
            assert!(
                rel_err(cpu_out_f32, metal_out_f32) < tol,
                "Prefill output mismatch: max rel err {}",
                rel_err(cpu_out_f32, metal_out_f32)
            );

            metal_state.sync_metal_to_cpu();
            let cpu_k = cpu_state.get("k_state").unwrap().as_f32().unwrap();
            let metal_k = metal_state.get("k_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_k, metal_k) < tol,
                "Prefill k_state mismatch: max rel err {}",
                rel_err(cpu_k, metal_k)
            );

            let cpu_v = cpu_state.get("v_state").unwrap().as_f32().unwrap();
            let metal_v = metal_state.get("v_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_v, metal_v) < tol,
                "Prefill v_state mismatch: max rel err {}",
                rel_err(cpu_v, metal_v)
            );
        }

        // 2. 3 decode steps (m = 1, positions 5, 6, 7)
        for step in 0..3 {
            let cur_pos = (5 + step) as u32;
            let q_data = rand(q_dim, 400 + step as u64);
            let k_data = rand(kv_dim, 500 + step as u64);
            let v_data = rand(kv_dim, 600 + step as u64);
            let pos_data = vec![cur_pos];

            let inputs = vec![
                Tensor::F32(q_data),
                Tensor::F32(k_data),
                Tensor::F32(v_data),
                Tensor::U32(pos_data),
            ];

            let cpu_out =
                cpu::run(&decode_g, &Weights::new(), &mut cpu_state, inputs.clone()).unwrap();
            let metal_out = session
                .run_with_state(&decode_g, Some(&mut metal_state), inputs)
                .unwrap();

            let cpu_out_f32 = cpu_out[0].as_f32().unwrap();
            let metal_out_f32 = metal_out[0].as_f32().unwrap();
            assert!(
                rel_err(cpu_out_f32, metal_out_f32) < tol,
                "Decode step {step} output mismatch: max rel err {}",
                rel_err(cpu_out_f32, metal_out_f32)
            );
            metal_state.sync_metal_to_cpu();

            let cpu_k = cpu_state.get("k_state").unwrap().as_f32().unwrap();
            let metal_k = metal_state.get("k_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_k, metal_k) < tol,
                "Decode step {step} k_state mismatch: max rel err {}",
                rel_err(cpu_k, metal_k)
            );

            let cpu_v = cpu_state.get("v_state").unwrap().as_f32().unwrap();
            let metal_v = metal_state.get("v_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_v, metal_v) < tol,
                "Decode step {step} v_state mismatch: max rel err {}",
                rel_err(cpu_v, metal_v)
            );
        }
    }

    /// (KV dtype is f16?, rel_err bound against the f32 CPU executor): f16 rounds each stored K/V to 11 bits
    /// (<= 2^-11 relative, 2.4e-4 absolute on |x| <= 1 inputs).
    const KV_DTYPES: [(bool, f32); 2] = [(false, 1e-4), (true, 1e-3)];

    /// Smallest cosine between matching `hd`-wide head rows of `want` and `got`.
    fn min_head_cosine(want: &[f32], got: &[f32], hd: usize) -> f32 {
        want.chunks(hd)
            .zip(got.chunks(hd))
            .map(|(a, b)| {
                let dot = |x: &[f32], y: &[f32]| {
                    x.iter().zip(y).map(|(&p, &q)| (p * q) as f64).sum::<f64>()
                };
                (dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-30)) as f32
            })
            .fold(1.0, f32::min)
    }

    /// Decode over a cache long enough to split across threadgroups (full and sliding window, soft cap,
    /// grouped KV heads): the split + merge path equals the CPU executor, with f32 and f16 KV caches
    /// (f16: every head row at cosine >= 0.9999).
    #[test]
    fn split_decode_attention_matches_cpu() {
        for (kv_f16, tol) in KV_DTYPES {
            split_decode_attention_matches_cpu_for(kv_f16, tol);
        }
    }

    fn split_decode_attention_matches_cpu_for(kv_f16: bool, tol: f32) {
        // (cache size, prefill length) pairs are looped below.
        // Query heads per KV head: 2 (a partial head tile), 8 (two full tiles), 1, 6 (tiles of 4 and 2),
        // 3 (a single tile of 3); head dims below 512 exercise the q-tile zero fill.
        for (heads, kv_heads, hd) in [
            (4usize, 2usize, 32usize),
            (8, 1, 64),
            (3, 3, 32),
            (6, 1, 32),
            (6, 2, 128),
            // E2B's sliding (hd 256) and global (hd 512) layers: the 8- and 16-value kernel variants.
            (8, 1, 256),
            (8, 1, 512),
        ] {
            // Short caches for every shape; for the two widest, a cache past the 4096-row minimum of the
            // physical state buffers: allocated large at prefill (4200), and grown with a copy mid-decode
            // (4060 prefill, the decode at +40 crosses 4096).
            for (max_seq, prefill) in [(1024usize, 700usize), (4400, 4200), (4400, 4060)] {
                if prefill > 1000 && !matches!(hd, 64 | 128) {
                    continue;
                }
                let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
                for window in [None, Some(300), Some(90)] {
                    let make = |m: usize| {
                        let mut g = Graph::new();
                        let q = g.input("q", vec![m, q_dim], DType::F32);
                        let k = g.input("k", vec![m, kv_dim], DType::F32);
                        let v = g.input("v", vec![m, kv_dim], DType::F32);
                        let ks = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
                        let vs = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
                        let pos = g.input("pos", vec![m], DType::U32);
                        let out = g.add(
                            "attn",
                            Op::CachedAttention {
                                heads,
                                kv_heads,
                                head_dim: hd,
                                v_head_dim: hd,
                                scale: 1.0 / (hd as f32).sqrt(),
                                causal: true,
                                window,
                                soft_cap: Some(30.0),
                                write: true,
                            },
                            &[q, k, v, ks, vs, pos],
                            vec![m, q_dim],
                            DType::F32,
                        );
                        g.output(out);
                        g
                    };
                    let (mut cpu_state, mut metal_state) = (StateStore::new(), StateStore::new());
                    let mut session = Session::new(rt(), &Weights::new()).unwrap();
                    session.set_kv_f16(kv_f16);
                    let step = |g: &Graph,
                                m: usize,
                                at: usize,
                                seed: u64,
                                cs: &mut StateStore,
                                ms: &mut StateStore| {
                        let inputs = vec![
                            Tensor::F32(rand(m * q_dim, seed)),
                            Tensor::F32(rand(m * kv_dim, seed + 1)),
                            Tensor::F32(rand(m * kv_dim, seed + 2)),
                            Tensor::U32((at..at + m).map(|p| p as u32).collect()),
                        ];
                        let want = cpu::run(g, &Weights::new(), cs, inputs.clone()).unwrap();
                        let got = session.run_with_state(g, Some(ms), inputs).unwrap();
                        let (want, got) = (want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                        let cos = min_head_cosine(want, got, hd);
                        assert!(
                            cos >= 0.9999,
                            "f16 KV {kv_f16} at {at} {window:?}: min cosine {cos}"
                        );
                        rel_err(want, got)
                    };
                    let e = step(
                        &make(prefill),
                        prefill,
                        0,
                        1,
                        &mut cpu_state,
                        &mut metal_state,
                    );
                    assert!(e < tol, "f16 KV {kv_f16} prefill {window:?}: {e}");
                    let (kbuf, n) = &metal_state.metal_buffers["k_state"];
                    assert_eq!(metal_state.metal_f16.contains("k_state"), kv_f16);
                    assert_eq!(
                        kbuf.length() as usize,
                        (n * if kv_f16 { 2 } else { 4 }).max(ARENA_ALIGN)
                    );
                    let decode = make(1);
                    // 700..: more than 128 keys visible, so the sliding windows of 300 and 90 straddle the cut-off.
                    for (i, at) in [prefill, prefill + 1, prefill + 40].into_iter().enumerate() {
                        let e = step(
                            &decode,
                            1,
                            at,
                            10 + 3 * i as u64,
                            &mut cpu_state,
                            &mut metal_state,
                        );
                        assert!(e < tol, "f16 KV {kv_f16} decode at {at} {window:?}: {e}");
                    }
                }
            }
        }
    }

    /// E1 probe: a chain of dependent f16 GEMVs (each reads 1536x1536 weights, feeding the next) as
    /// separate dispatches versus one persistent kernel with an atomic grid barrier between steps.
    /// Prints per-step time for both and checks the results agree
    /// (`cargo test --release --lib persistent_chain_probe -- --ignored --nocapture`).
    #[test]
    #[ignore = "measurement"]
    fn persistent_chain_probe() {
        const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
inline float row_dot(device const half* w, device const float* x, uint cols, uint lane) {
    float acc = 0.0f;
    for (uint c = lane * 4; c < cols; c += 128) {
        acc += dot(float4(*(device const half4*)(w + c)), *(device const float4*)(x + c));
    }
    return simd_sum(acc);
}
kernel void sep(device const half* w [[buffer(0)]], device const float* x [[buffer(1)]],
                device float* y [[buffer(2)]], constant uint2& p [[buffer(3)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 8 + sg;
    if (row >= p.x) return;
    float s = row_dot(w + (ulong)row * p.y, x, p.y, lane);
    if (lane == 0) y[row] = s;
}
kernel void persist(device const half* w [[buffer(0)]], device float* xa [[buffer(1)]],
                    device float* xb [[buffer(2)]], device atomic_uint* ctr [[buffer(3)]],
                    constant uint4& p [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    for (uint s = 0; s < p.z; s++) {
        device const half* ws = w + (ulong)s * p.x * p.y;
        device float* x = (s & 1) ? xb : xa;
        device float* y = (s & 1) ? xa : xb;
        for (uint row = tg * 8 + sg; row < p.x; row += p.w * 8) {
            float v = row_dot(ws + (ulong)row * p.y, x, p.y, lane);
            if (lane == 0) y[row] = v;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
            // Spin cap: a co-residency mistake ends the kernel (wrong results) instead of hanging the GPU.
            for (uint spin = 0; spin < 20000000u && atomic_load_explicit(ctr, memory_order_relaxed) < (s + 1) * p.w; spin++) {}
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"#;
        let rt = rt();
        let dev = rt.device.clone();
        let lib = dev
            .new_library_with_source(SRC, &metal::CompileOptions::new())
            .unwrap();
        let pso = |n: &str| {
            dev.new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
                .unwrap()
        };
        let (sep, persist) = (pso("sep"), pso("persist"));
        let n = std::env::var("PROBE_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1536usize);
        let steps = 64usize;
        let w = new_shared_buffer(&dev, steps * n * n * 2).unwrap();
        unsafe {
            let p = w.contents() as *mut u16;
            // Random f16 weights, variance 1/n: the chain keeps O(1) magnitudes and every row differs.
            let mut s = 0x9E3779B97F4A7C15u64;
            let amp = 3f32.sqrt() / (n as f32).sqrt();
            for i in 0..steps * n * n {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
                *p.add(i) = crate::model::f32_to_f16(u * amp);
            }
        }
        let xa = new_shared_buffer(&dev, n * 4).unwrap();
        let xb = new_shared_buffer(&dev, n * 4).unwrap();
        let ctr = new_shared_buffer(&dev, 256).unwrap();
        let init = |b: &Buffer| unsafe {
            for i in 0..n {
                *(b.contents() as *mut f32).add(i) = ((i % 7) as f32 - 3.0) * 0.1;
            }
        };
        let q = dev.new_command_queue();
        let read = |b: &Buffer| -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
        };
        let run_sep = || -> f64 {
            init(&xa);
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&sep);
            for s in 0..steps {
                let (x, y) = if s % 2 == 0 { (&xa, &xb) } else { (&xb, &xa) };
                enc.set_buffer(0, Some(&w), (s * n * n * 2) as u64);
                enc.set_buffer(1, Some(x), 0);
                enc.set_buffer(2, Some(y), 0);
                set_bytes(enc, 3, &[n as u32, n as u32]);
                enc.dispatch_thread_groups(
                    MTLSize::new((n as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            enc.end_encoding();
            let t = Instant::now();
            commit_and_wait(cb).unwrap();
            t.elapsed().as_secs_f64()
        };
        let run_persist = |tgs: u64| -> f64 {
            init(&xa);
            unsafe { *(ctr.contents() as *mut u32) = 0 };
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&persist);
            enc.set_buffer(0, Some(&w), 0);
            enc.set_buffer(1, Some(&xa), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&ctr), 0);
            set_bytes(enc, 4, &[n as u32, n as u32, steps as u32, tgs as u32]);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            let t = Instant::now();
            commit_and_wait(cb).unwrap();
            t.elapsed().as_secs_f64()
        };
        let bytes = (n * n * 2) as f64;
        run_sep();
        let t_sep = (0..5).map(|_| run_sep()).fold(f64::MAX, f64::min);
        let want = read(if steps % 2 == 0 { &xa } else { &xb });
        let mag = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(mag > 1e-2, "degenerate chain: max |y| {mag}");
        println!(
            "n {n}: separate {:.1} us/step = {:.0} GB/s (max |y| {mag:.2})",
            t_sep / steps as f64 * 1e6,
            bytes * steps as f64 / t_sep / 1e9
        );
        for tgs in [20u64, 40, 80, 120] {
            run_persist(tgs);
            let t = (0..5).map(|_| run_persist(tgs)).fold(f64::MAX, f64::min);
            let got = read(if steps % 2 == 0 { &xa } else { &xb });
            let err = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "persistent {tgs:>3} TGs: {:.1} us/step = {:.0} GB/s, max |diff| {err:e}",
                t / steps as f64 * 1e6,
                bytes * steps as f64 / t / 1e9
            );
        }
    }

    /// Decode attention GPU time over a long cache for the Gemma-4 layer shapes (8 query heads on one KV
    /// head: full attention hd 512, sliding hd 256 with a 512 window)
    /// (`cargo test --release --lib attention_decode_rates -- --ignored --nocapture`).
    #[test]
    #[ignore = "bandwidth measurement"]
    fn attention_decode_rates() {
        for (heads, kv_heads, hd, window, at) in [
            (8usize, 1usize, 512usize, None, 4096usize),
            (8, 1, 256, Some(512usize), 4096),
            (8, 4, 128, None, 4096),
        ] {
            let (q_dim, kv_dim, max_seq) = (heads * hd, kv_heads * hd, 8192);
            let mut g = Graph::new();
            let q = g.input("q", vec![1, q_dim], DType::F32);
            let k = g.input("k", vec![1, kv_dim], DType::F32);
            let v = g.input("v", vec![1, kv_dim], DType::F32);
            let ks = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
            let vs = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
            let pos = g.input("pos", vec![1], DType::U32);
            let out = g.add(
                "attn",
                Op::CachedAttention {
                    heads,
                    kv_heads,
                    head_dim: hd,
                    v_head_dim: hd,
                    scale: 1.0,
                    causal: true,
                    window,
                    soft_cap: None,
                    write: true,
                },
                &[q, k, v, ks, vs, pos],
                vec![1, q_dim],
                DType::F32,
            );
            g.output(out);
            let session = Session::new(rt(), &Weights::new()).unwrap();
            let mut state = StateStore::new();
            let mut best = f64::INFINITY;
            for i in 0..12 {
                let inputs = vec![
                    Tensor::F32(rand(q_dim, 1 + i)),
                    Tensor::F32(rand(kv_dim, 2)),
                    Tensor::F32(rand(kv_dim, 3)),
                    Tensor::U32(vec![at as u32]),
                ];
                session
                    .run_with_state(&g, Some(&mut state), inputs)
                    .unwrap();
                if i >= 2 {
                    best = best.min(session.last_gpu_seconds());
                }
            }
            let keys = window.map_or(at + 1, |w| w.min(at + 1));
            let bytes = (keys * kv_dim * 2 * 4) as f64;
            println!(
                "attention heads {heads} kv {kv_heads} hd {hd} window {window:?} keys {keys}: {:.0} us, {:.0} GB/s of KV",
                best * 1e6,
                bytes / best / 1e9
            );
        }
    }

    /// A bidirectional block: rows inside it see keys up to the block's end (`visible_end`), the sliding
    /// window still starting at the row's own position. Metal against the CPU executor.
    #[test]
    fn cached_attention_visible_end_matches_cpu() {
        let (heads, kv_heads, hd, m) = (4, 2, 8, 6);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let mut g = Graph::new();
        let q = g.input("q", vec![m, q_dim], DType::F32);
        let k = g.input("k", vec![m, kv_dim], DType::F32);
        let v = g.input("v", vec![m, kv_dim], DType::F32);
        let ks = g.state("k_state", vec![16, kv_dim], DType::F32, false);
        let vs = g.state("v_state", vec![16, kv_dim], DType::F32, false);
        let pos = g.input("pos", vec![m], DType::U32);
        let vis = g.input("vis", vec![m], DType::U32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
                window: Some(4),
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos, vis],
            vec![m, q_dim],
            DType::F32,
        );
        g.output(out);
        let inputs = vec![
            Tensor::F32(rand(m * q_dim, 11)),
            Tensor::F32(rand(m * kv_dim, 22)),
            Tensor::F32(rand(m * kv_dim, 33)),
            Tensor::U32((0..m as u32).collect()),
            Tensor::U32(vec![1, 2, 3, 6, 6, 6]),
        ];
        let want = cpu::run(&g, &Weights::new(), &mut StateStore::new(), inputs.clone()).unwrap();
        for (kv_f16, tol) in KV_DTYPES {
            let mut session = Session::new(rt(), &Weights::new()).unwrap();
            session.set_kv_f16(kv_f16);
            let got = session
                .run_with_state(&g, Some(&mut StateStore::new()), inputs.clone())
                .unwrap();
            let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
            assert!(err < tol, "visible_end mismatch (f16 KV {kv_f16}): {err}");
        }
    }

    /// `m` rows of cached attention over `[max_seq, kv_heads * hd]` k/v states named `k_state` / `v_state`.
    fn kv_attn_graph(
        heads: usize,
        kv_heads: usize,
        hd: usize,
        window: Option<usize>,
        max_seq: usize,
        m: usize,
    ) -> Graph {
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let mut g = Graph::new();
        let q = g.input("q", vec![m, q_dim], DType::F32);
        let k = g.input("k", vec![m, kv_dim], DType::F32);
        let v = g.input("v", vec![m, kv_dim], DType::F32);
        let ks = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
        let vs = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
        let pos = g.input("pos", vec![m], DType::U32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
                window,
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos],
            vec![m, q_dim],
            DType::F32,
        );
        g.output(out);
        g
    }

    /// f16 KV caches against the f32 CPU executor: prefills of 1, 63, 64, 65 and 4096 rows then two decode
    /// steps (the last prefill's first decode crosses the 4096-row physical buffer: grown by a byte copy and
    /// split across threadgroups), grouped / ungrouped KV heads and a sliding window. The stored caches are
    /// exactly the round-to-nearest-even f16 of what the CPU executor stored (prefill via `kv_append`,
    /// decode via the split kernel's fused write).
    #[test]
    fn f16_kv_attention_matches_cpu_across_lengths() {
        use crate::model::{f16_to_f32, f32_to_f16};
        let max_seq = 4400;
        for (heads, kv_heads, hd, window) in [
            (4usize, 2usize, 32usize, None),
            (4, 4, 32, None),
            (8, 1, 64, Some(40usize)),
        ] {
            let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
            for prefill in [1usize, 63, 64, 65, 4096] {
                let mut session = Session::new(rt(), &Weights::new()).unwrap();
                session.set_kv_f16(true);
                let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
                let mut at = 0;
                for (i, m) in [prefill, 1, 1].into_iter().enumerate() {
                    let g = kv_attn_graph(heads, kv_heads, hd, window, max_seq, m);
                    let seed = 100 * i as u64 + 7;
                    let inputs = vec![
                        Tensor::F32(rand(m * q_dim, seed)),
                        Tensor::F32(rand(m * kv_dim, seed + 1)),
                        Tensor::F32(rand(m * kv_dim, seed + 2)),
                        Tensor::U32((at..at + m).map(|p| p as u32).collect()),
                    ];
                    let want = cpu::run(&g, &Weights::new(), &mut cs, inputs.clone()).unwrap();
                    let got = session.run_with_state(&g, Some(&mut ms), inputs).unwrap();
                    let (want, got) = (want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                    let (e, cos) = (rel_err(want, got), min_head_cosine(want, got, hd));
                    assert!(
                        e < 1e-3 && cos >= 0.9999,
                        "heads {heads}/{kv_heads} hd {hd} {window:?} prefill {prefill} chunk {i}: {e}, cos {cos}"
                    );
                    at += m;
                }
                assert!(ms.metal_f16.contains("k_state") && ms.metal_f16.contains("v_state"));
                let n_buf = ms.metal_buffers["k_state"].1;
                assert_eq!(ms.metal_buffers["k_state"].0.length() as usize, n_buf * 2);
                ms.sync_metal_to_cpu();
                for name in ["k_state", "v_state"] {
                    let want = &cs.get(name).unwrap().as_f32().unwrap()[..at * kv_dim];
                    let got = &ms.get(name).unwrap().as_f32().unwrap()[..at * kv_dim];
                    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
                        assert_eq!(
                            g,
                            f16_to_f32(f32_to_f16(w)),
                            "{name}[{i}] of {at} positions: stored {g}, f32 {w}"
                        );
                    }
                }
            }
        }
    }

    /// Out-of-range K/V saturate at +-65504 instead of becoming inf; NaN stays NaN; tiny values flush to 0.
    /// Covers `kv_append_batched_f16` (m = 2 prefill) and the split kernel's fused f16 write (m = 1).
    #[test]
    fn f16_kv_saturates_out_of_range_values() {
        let k = [1e6, -1e6, 70000.0, f32::NAN];
        let v = [65504.0, 65520.0, 1e-9, -2.5];
        for m in [1usize, 2] {
            let g = kv_attn_graph(1, 1, 4, None, 16, m);
            let mut session = Session::new(rt(), &Weights::new()).unwrap();
            session.set_kv_f16(true);
            let mut st = StateStore::new();
            let inputs = vec![
                Tensor::F32(vec![1.0; 4 * m]),
                Tensor::F32(k.repeat(m)),
                Tensor::F32(v.repeat(m)),
                Tensor::U32((0..m as u32).collect()),
            ];
            session.run_with_state(&g, Some(&mut st), inputs).unwrap();
            let (ks, vs) = (
                st.read_metal("k_state").unwrap(),
                st.read_metal("v_state").unwrap(),
            );
            for row in 0..m {
                let (kr, vr) = (&ks[row * 4..row * 4 + 4], &vs[row * 4..row * 4 + 4]);
                assert_eq!(kr[..3], [65504.0, -65504.0, 65504.0], "m {m} row {row}");
                assert!(kr[3].is_nan(), "m {m} row {row}");
                assert_eq!(vr, [65504.0, 65504.0, 0.0, -2.5], "m {m} row {row}");
            }
        }
    }

    /// CPU-only: a state is f16 only when every reader is a `CachedAttention` k/v operand. Layers sharing
    /// a cache stay f16; a plain `Attention` over the cache keeps it f32.
    #[test]
    fn kv_dtype_follows_the_states_readers() {
        let (heads, hd, max_seq) = (2, 8, 16);
        let f = DType::F32;
        let attn = |write| Op::CachedAttention {
            heads,
            kv_heads: heads,
            head_dim: hd,
            v_head_dim: hd,
            scale: 0.3,
            causal: true,
            window: None,
            soft_cap: None,
            write,
        };
        let build = |shared: bool, plain_reader: bool| {
            let mut g = Graph::new();
            let q = g.input("q", vec![1, heads * hd], f);
            let k = g.input("k", vec![1, heads * hd], f);
            let v = g.input("v", vec![1, heads * hd], f);
            let pos = g.input("pos", vec![1], DType::U32);
            let ks = g.state("k_state", vec![max_seq, heads * hd], f, false);
            let vs = g.state("v_state", vec![max_seq, heads * hd], f, false);
            let a = g.add(
                "a",
                attn(true),
                &[q, k, v, ks, vs, pos],
                vec![1, heads * hd],
                f,
            );
            g.output(a);
            if shared {
                let (ks2, vs2) = (
                    g.state("k_state", vec![max_seq, heads * hd], f, false),
                    g.state("v_state", vec![max_seq, heads * hd], f, false),
                );
                let b = g.add(
                    "b",
                    attn(false),
                    &[a, k, v, ks2, vs2, pos],
                    vec![1, heads * hd],
                    f,
                );
                g.output(b);
            }
            if plain_reader {
                let plain = Op::Attention {
                    heads,
                    kv_heads: heads,
                    causal: false,
                    window: None,
                    soft_cap: None,
                    scale: None,
                };
                let c = g.add("c", plain, &[q, ks, vs], vec![1, heads * hd], f);
                g.output(c);
            }
            g
        };
        let assign = |g: &Graph| {
            let (mut steps, _) = plan(g).unwrap();
            let f16 = assign_kv_dtypes(g, &mut steps);
            let flags: Vec<bool> = steps
                .iter()
                .filter_map(|s| match s.op {
                    Step::CachedAttention { kv_f16, .. } => Some(kv_f16),
                    _ => None,
                })
                .collect();
            (f16.len(), flags)
        };
        assert_eq!(assign(&build(false, false)), (2, vec![true]));
        assert_eq!(assign(&build(true, false)), (2, vec![true, true]));
        assert_eq!(assign(&build(false, true)), (0, vec![false]));
        assert_eq!(assign(&build(true, true)), (0, vec![false, false]));
    }

    #[test]
    fn kernel_rows_group_steps_by_innermost_kind_slowest_first() {
        assert_eq!(kernel_kind("Ext(Norm(RmsNorm { eps: 1e-6 }))"), "RmsNorm");
        assert_eq!(kernel_kind("GemvPacked { rows: 8 }"), "GemvPacked");
        assert_eq!(kernel_kind("?"), "?");
        // Field values never name the kind: `ValueId(..)`, `Some(ValueId(..))` and `Gen(..)` follow it.
        assert_eq!(
            kernel_kind("GemmBatchedPacked { x: ValueId(1), res: Some(ValueId(4)), format: Gen(GGUF_Q4_K_M) }"),
            "GemmBatchedPacked"
        );
        assert_eq!(kernel_kind("Ext(Moe(Embed))"), "Embed");
        assert_eq!(kernel_kind("Softmax(3)"), "Softmax");
        let t = |name: &str, desc: &str, s: f64, bytes: u64| {
            (name.to_string(), desc.to_string(), s, bytes)
        };
        let rows = kernel_rows(
            &[
                t("a", "Ext(Norm(RmsNorm { eps: 1e-6 }))", 1e-6, 1_000),
                t("b", "GemvPacked { rows: 8 }", 8e-6, 2_000_000),
                t("c", "Ext(Norm(RmsNorm { eps: 1e-6 }))", 1e-6, 1_000),
            ],
            100.0,
        );
        let kinds: Vec<_> = rows.iter().map(|r| (r.kind.as_str(), r.steps)).collect();
        assert_eq!(kinds, [("GemvPacked", 1), ("RmsNorm", 2)]);
        assert!((rows[0].us - 8.0).abs() < 1e-9);
        // 2 MB in 8 µs = 250 GB/s = 250% of a 100 GB/s ceiling.
        assert!((rows[0].gbps - 250.0).abs() < 1e-6 && (rows[0].pct - 250.0).abs() < 1e-6);
    }

    /// A requested profile times one real decode token step by step: the token's output is the
    /// one an unprofiled run gives, and the per-kind rows land in `last_kernel_profile`.
    #[test]
    fn requested_kernel_profile_samples_a_decode_token_and_keeps_its_output() {
        let (heads, kv_heads, hd) = (4, 2, 8);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let mut g = Graph::new();
        let q = g.input("q", vec![1, q_dim], DType::F32);
        let k = g.input("k", vec![1, kv_dim], DType::F32);
        let v = g.input("v", vec![1, kv_dim], DType::F32);
        let ks = g.state("k_state", vec![16, kv_dim], DType::F32, false);
        let vs = g.state("v_state", vec![16, kv_dim], DType::F32, false);
        let pos = g.input("pos", vec![1], DType::U32);
        let out = g.add(
            "cached_attn",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
                window: None,
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos],
            vec![1, q_dim],
            DType::F32,
        );
        g.output(out);
        let inputs = || {
            vec![
                Tensor::F32(rand(q_dim, 1)),
                Tensor::F32(rand(kv_dim, 2)),
                Tensor::F32(rand(kv_dim, 3)),
                Tensor::U32(vec![0]),
            ]
        };
        let session = Session::new(rt(), &Weights::new()).unwrap();
        let want = session
            .run_with_state(&g, Some(&mut StateStore::new()), inputs())
            .unwrap();
        // Another test's decode run may take the one-shot request: ask again until ours is sampled.
        let start = Instant::now();
        let mut got = None;
        for _ in 0..100 {
            request_kernel_profile();
            let out = session
                .run_with_state(&g, Some(&mut StateStore::new()), inputs())
                .unwrap();
            if last_kernel_profile().is_some_and(|(at, _)| at >= start) {
                got = Some(out);
                break;
            }
        }
        let got = got.expect("a requested decode token is profiled");
        assert!(rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap()) < 1e-6);
        let (_, rows) = last_kernel_profile().unwrap();
        assert!(rows.iter().map(|r| r.steps).sum::<usize>() >= 1, "{rows:?}");
        assert!(
            rows.iter().all(|r| r.us >= 0.0 && r.gbps >= 0.0),
            "{rows:?}"
        );
    }


    #[test]
    fn moe_subgraph_fusions_match_cpu() {
        use crate::graph::ReduceKind;

        for m in [1usize, 2] {
            let (h, ff, experts, top_k) = (16, 16, 6, 3);
            let f = DType::F32;

            // 1. MoE subgraph starting from router logits (12 raw steps -> 5 fused steps).
            let mut g = Graph::new();
            let x = g.input("x", vec![m, h], f);
            let logits = g.input("logits", vec![m, experts], f);
            let scale_w = g.weight("scale", vec![1, experts], f);
            let gu_w = g.weight("gate_up", vec![experts * 2 * ff, h], f);
            let down_w = g.weight("down", vec![experts * h, ff], f);

            let ids = g.add(
                "topk",
                Op::TopK { k: top_k },
                &[logits],
                vec![m, top_k],
                DType::U32,
            );
            let picked = g.add("picked", Op::GatherCols, &[logits, ids], vec![m, top_k], f);
            let probs = g.add(
                "probs",
                Op::Softmax { axis: -1 },
                &[picked],
                vec![m, top_k],
                f,
            );
            let sc = g.add("sc", Op::GatherCols, &[scale_w, ids], vec![m, top_k], f);
            let weights_v = g.add(
                "weights",
                Op::Binary {
                    kind: crate::graph::BinaryKind::Mul,
                },
                &[probs, sc],
                vec![m, top_k],
                f,
            );
            let gu = g.add(
                "gu",
                Op::MoeLinear { experts, top_k },
                &[x, gu_w, ids],
                vec![m * top_k, 2 * ff],
                f,
            );
            let gate = g.add(
                "gate",
                Op::Slice {
                    axis: 1,
                    start: 0,
                    end: ff as i64,
                },
                &[gu],
                vec![m * top_k, ff],
                f,
            );
            let up = g.add(
                "up",
                Op::Slice {
                    axis: 1,
                    start: ff as i64,
                    end: (2 * ff) as i64,
                },
                &[gu],
                vec![m * top_k, ff],
                f,
            );
            let act = g.add("act", Op::GeGlu, &[gate, up], vec![m * top_k, ff], f);
            let down = g.add(
                "down",
                Op::MoeLinear { experts, top_k },
                &[act, down_w, ids],
                vec![m * top_k, h],
                f,
            );
            let w_col = g.add(
                "w_col",
                Op::Reshape {
                    shape: Some(vec![m * top_k, 1]),
                },
                &[weights_v],
                vec![m * top_k, 1],
                f,
            );
            let weighted = g.add(
                "weighted",
                Op::Binary {
                    kind: crate::graph::BinaryKind::Mul,
                },
                &[down, w_col],
                vec![m * top_k, h],
                f,
            );
            let slots = g.add(
                "slots",
                Op::Reshape {
                    shape: Some(vec![m, top_k, h]),
                },
                &[weighted],
                vec![m, top_k, h],
                f,
            );
            let out = g.add(
                "out",
                Op::Reduce {
                    kind: ReduceKind::Sum,
                    axes: vec![1],
                    keep_dims: false,
                },
                &[slots],
                vec![m, h],
                f,
            );
            g.output(out);

            let mut w = Weights::new();
            w.insert("scale", Tensor::F32(rand(experts, 11)));
            w.insert("gate_up", Tensor::F32(rand(experts * 2 * ff * h, 22)));
            w.insert("down", Tensor::F32(rand(experts * h * ff, 33)));

            let sess = Session::new(rt(), &w).unwrap();
            let prep = sess.prepare_plan(&g).unwrap();
            assert_eq!(prep.steps.len(), 5, "12 raw MoE steps fuse to 5");
            assert!(matches!(
                prep.steps[0].op,
                Step::Ext(ext::ExtStep::Experts(ext::moe::Step::TopK {
                    weights_out: Some(_),
                    scale: Some(_),
                    ..
                }))
            ));
            assert!(matches!(prep.steps[2].op, Step::SplitGlu { kind: 1, .. }));
            assert!(matches!(prep.steps[4].op, Step::MoeWeightedSum { .. }));

            let inputs = vec![
                Tensor::F32(rand(m * h, 44)),
                Tensor::F32(rand(m * experts, 55)),
            ];
            let want = cpu::run_stateless(&g, &w, inputs.clone())
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            let got = sess
                .run_prepared(&g, &prep, None, inputs)
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            assert!(
                rel_err(&want, &got) < 1e-5,
                "rel_err = {}",
                rel_err(&want, &got)
            );
        }
    }

    #[test]
    fn gemv_packed_pair_small_rows_matches_cpu() {
        use crate::model::layers::gguf_block_codec;
        for f in [
            QuantFormat::GGUF_Q4_K_M,
            QuantFormat::GGUF_Q8_0,
            QuantFormat::GGUF_Q4_0,
        ] {
            for (rows0, rows1, cols) in [(256usize, 256usize, 512usize), (16, 16, 2048)] {
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (f as u64) ^ (cols as u64);
                let mut next = move || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed >> 33) as u32
                };
                let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
                let scale_at: &[usize] = match f {
                    QuantFormat::GGUF_Q4_K_M => &[0, 2],
                    _ => &[0],
                };
                let mut w = Weights::new();
                let mut g = Graph::new();
                let mut quant = |g: &mut Graph, w: &mut Weights, name: &str, n: usize, k: usize| {
                    let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                        .map(|_| (next() >> 4) as u8)
                        .collect();
                    for blk in raw.chunks_exact_mut(bb) {
                        for &o in scale_at {
                            let exp = 9 + (next() % 14) as u16;
                            let d =
                                ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
                            blk[o..o + 2].copy_from_slice(&d.to_le_bytes());
                        }
                    }
                    w.insert(
                        name,
                        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
                    );
                    g.weight(name, vec![n, k], DType::Quant(f))
                };
                let x = g.input("x", vec![1, cols], DType::F32);
                let kw = quant(&mut g, &mut w, "k_w", rows0, cols);
                let vw = quant(&mut g, &mut w, "v_w", rows1, cols);
                let kb = g.weight("k_b", vec![rows0], DType::F32);
                w.insert("k_b", Tensor::F32(rand(rows0, 11)));
                let k_out = g.add(
                    "k_proj",
                    Op::Linear,
                    &[x, kw, kb],
                    vec![1, rows0],
                    DType::F32,
                );
                let v_out = g.add("v_proj", Op::Linear, &[x, vw], vec![1, rows1], DType::F32);
                g.output(k_out);
                g.output(v_out);

                let mut s = Session::new(rt(), &w).unwrap();
                s.set_variant_override(Some(GemvVariant::HAND));
                let plan = s.prepare_plan(&g).unwrap();
                assert_eq!(plan.steps.len(), 1);
                assert!(matches!(plan.steps[0].op, Step::GemvPackedPair { .. }));

                let inp = vec![Tensor::F32(rand(cols, 42))];
                let mut cs = StateStore::new();
                let want = cpu::run(&g, &w, &mut cs, inp.clone()).unwrap();
                let got = s.run_prepared(&g, &plan, None, inp).unwrap();
                for (i, (wt, gt)) in want.iter().zip(&got).enumerate() {
                    let err = rel_err(wt.as_f32().unwrap(), gt.as_f32().unwrap());
                    assert!(
                        err < 1e-5,
                        "{f:?} ({rows0},{rows1},{cols}) out {i}: err {err}"
                    );
                }
                if f == QuantFormat::GGUF_Q4_K_M && (rows0, rows1, cols) == (16, 16, 2048) {
                    s.set_variant_override(Some(GemvVariant::generated(1)));
                    assert_eq!(s.prepare_plan(&g).unwrap().steps.len(), 2);
                }
            }
        }
        for (rows0, rows1, cols) in [(256usize, 256usize, 512usize), (16, 16, 2048)] {
            let mut w = Weights::new();
            let mut g = Graph::new();
            let x = g.input("x", vec![1, cols], DType::F32);
            let kw = g.weight("k_w", vec![rows0, cols], DType::BF16);
            let vw = g.weight("v_w", vec![rows1, cols], DType::BF16);
            let kb = g.weight("k_b", vec![rows0], DType::F32);
            let vb = g.weight("v_b", vec![rows1], DType::F32);
            let to_bf16 = |vals: Vec<f32>| -> Vec<u16> {
                vals.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
            };
            w.insert("k_w", Tensor::BF16(to_bf16(rand(rows0 * cols, 71))));
            w.insert("v_w", Tensor::BF16(to_bf16(rand(rows1 * cols, 73))));
            w.insert("k_b", Tensor::F32(rand(rows0, 11)));
            w.insert("v_b", Tensor::F32(rand(rows1, 13)));
            let k_out = g.add(
                "k_proj",
                Op::Linear,
                &[x, kw, kb],
                vec![1, rows0],
                DType::F32,
            );
            let v_raw = g.add("v_proj", Op::Linear, &[x, vw], vec![1, rows1], DType::F32);
            let v_out = g.add(
                "v_biased",
                Op::Binary {
                    kind: crate::graph::BinaryKind::Add,
                },
                &[v_raw, vb],
                vec![1, rows1],
                DType::F32,
            );
            g.output(k_out);
            g.output(v_out);

            let s = Session::new(rt(), &w).unwrap();
            let plan = s.prepare_plan(&g).unwrap();
            assert_eq!(plan.steps.len(), 1);
            assert!(matches!(plan.steps[0].op, Step::GemvPackedPair { .. }));

            let inp = vec![Tensor::F32(rand(cols, 42))];
            let mut cs = StateStore::new();
            let want = cpu::run(&g, &w, &mut cs, inp.clone()).unwrap();
            let got = s.run_prepared(&g, &plan, None, inp).unwrap();
            for (i, (wt, gt)) in want.iter().zip(&got).enumerate() {
                let err = rel_err(wt.as_f32().unwrap(), gt.as_f32().unwrap());
                assert!(
                    err < 1e-5,
                    "BF16 ({rows0},{rows1},{cols}) out {i}: err {err}"
                );
            }
        }
    }

    #[test]
    fn plan_value_locs_resolve_without_per_token_hashmap() {
        let f = DType::F32;
        let mut g = Graph::new();
        let ids = g.input("ids", vec![1], DType::U32);
        let pos = g.input("pos", vec![1], DType::U32);
        let emb_w = g.weight("emb", vec![8, 4], f);
        let x = g.add("emb_lookup", Op::Embedding, &[emb_w, ids], vec![1, 4], f);
        let wq = g.weight("wq", vec![4, 4], f);
        let wk = g.weight("wk", vec![4, 4], f);
        let wv = g.weight("wv", vec![4, 4], f);
        let q = g.add("q", Op::Linear, &[x, wq], vec![1, 4], f);
        let k = g.add("k", Op::Linear, &[x, wk], vec![1, 4], f);
        let v = g.add("v", Op::Linear, &[x, wv], vec![1, 4], f);
        let ks = g.state("k_cache", vec![16, 4], f, false);
        let vs = g.state("v_cache", vec![16, 4], f, false);
        let attn = g.add(
            "attn",
            Op::CachedAttention {
                heads: 1,
                kv_heads: 1,
                head_dim: 4,
                v_head_dim: 4,
                scale: 0.5,
                causal: true,
                window: None,
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos],
            vec![1, 4],
            f,
        );
        g.output(attn);

        let mut w = Weights::new();
        w.insert("emb", Tensor::F32(rand(8 * 4, 1)));
        w.insert("wq", Tensor::F32(rand(16, 2)));
        w.insert("wk", Tensor::F32(rand(16, 3)));
        w.insert("wv", Tensor::F32(rand(16, 4)));

        let mut s = Session::new(rt(), &w).unwrap();
        s.set_kv_f16(false);
        let plan = s.prepare_plan(&g).unwrap();

        assert_eq!(plan.value_locs.len(), g.values.len());
        assert_eq!(plan.embedding_checks.len(), 1);
        assert_eq!(plan.embedding_checks[0].0, 0);
        assert_eq!(plan.embedding_checks[0].1, 8);
        assert_eq!(plan.attn_pos_inputs, vec![(1, 1)]);
        assert_eq!(plan.state_vars.len(), 2);
        assert!(matches!(
            plan.value_locs[ids.0 as usize],
            PlanValueLoc::Arena(_)
        ));
        assert!(matches!(
            plan.value_locs[emb_w.0 as usize],
            PlanValueLoc::Weight(_, _)
        ));
        assert!(matches!(
            plan.value_locs[ks.0 as usize],
            PlanValueLoc::State(0)
        ));
        assert!(matches!(
            plan.value_locs[vs.0 as usize],
            PlanValueLoc::State(1)
        ));

        let mut m_state = StateStore::new();
        let mut c_state = StateStore::new();
        for step_pos in 0..3u32 {
            let inp = vec![Tensor::U32(vec![step_pos + 1]), Tensor::U32(vec![step_pos])];
            let want = cpu::run(&g, &w, &mut c_state, inp.clone()).unwrap();
            let got = s.run_prepared(&g, &plan, Some(&mut m_state), inp).unwrap();
            let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
            assert!(err < 1e-4, "step {step_pos}: rel_err {err}");
        }
    }


    /// CPU only: `plan_dump` groups steps by layer and describes a materialized strided slice.
    #[test]
    fn plan_dump_describes_layers_and_slices() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 8], DType::F32);
        let op = Op::Slice {
            axis: 1,
            start: 0,
            end: 4,
        };
        let s = g.add("blk.3.slice", op, &[x], vec![2, 4], DType::F32);
        let neg = Op::Unary {
            kind: crate::graph::UnaryKind::Neg,
        };
        let y = g.add("blk.3.neg", neg, &[s], vec![2, 4], DType::F32);
        g.output(y);
        let d = plan_dump(&g, 1).unwrap();
        assert!(d.contains("-- layer 3\n"), "{d}");
        assert!(
            d.contains("slice axis=1 [0..4] of [2, 8] contiguous=false"),
            "{d}"
        );
        assert!(
            d.contains("plan-dump: 2 steps, 2 dispatches at ctx 1"),
            "{d}"
        );
    }

    #[test]
    fn batch_horizontal_commutes_interleaved_gemv_and_rmsnorm() {
        let f = DType::F32;
        let (heads, kv_heads, hd, h) = (4usize, 2usize, 8usize, 32usize);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let rope = Op::Rope {
            head_dim: hd,
            rotary_dim: hd,
            variant: crate::graph::RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        let mut g = Graph::new();
        let x = g.input("x", vec![1, h], f);
        let pos = g.input("pos", vec![1], DType::U32);
        let wq = g.weight("wq", vec![q_dim, h], DType::F16);
        let wk = g.weight("wk", vec![kv_dim, h], DType::F16);
        let wv = g.weight("wv", vec![kv_dim, h], DType::F16);
        let gw = g.weight("gw", vec![hd], f);
        let inv = g.weight("inv", vec![hd / 2], f);

        let q = g.add("q", Op::Linear, &[x, wq], vec![1, q_dim], f);
        let qn = g.add("qn", Op::RmsNorm { eps: 1e-5 }, &[q, gw], vec![1, q_dim], f);
        let qr = g.add("qr", rope.clone(), &[qn, pos, inv], vec![1, q_dim], f);

        let k = g.add("k", Op::Linear, &[x, wk], vec![1, kv_dim], f);
        let kn = g.add(
            "kn",
            Op::RmsNorm { eps: 1e-5 },
            &[k, gw],
            vec![1, kv_dim],
            f,
        );
        let kr = g.add("kr", rope, &[kn, pos, inv], vec![1, kv_dim], f);

        let v = g.add("v", Op::Linear, &[x, wv], vec![1, kv_dim], f);
        let vn = g.add(
            "vn",
            Op::RmsNorm { eps: 1e-6 },
            &[v, gw],
            vec![1, kv_dim],
            f,
        );

        let ks = g.state("ks", vec![16, kv_dim], f, false);
        let vs = g.state("vs", vec![16, kv_dim], f, false);
        let att = g.add(
            "att",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 0.25,
                causal: true,
                window: None,
                soft_cap: None,
                write: true,
            },
            &[qr, kr, vn, ks, vs, pos],
            vec![1, q_dim],
            f,
        );
        g.output(att);

        let to_f16 = |vals: Vec<f32>| -> Vec<u16> {
            vals.into_iter()
                .map(|v| crate::model::f32_to_f16(v * 0.2))
                .collect()
        };
        let mut w = Weights::new();
        w.insert("wq", Tensor::F16(to_f16(rand(q_dim * h, 1))));
        w.insert("wk", Tensor::F16(to_f16(rand(kv_dim * h, 2))));
        w.insert("wv", Tensor::F16(to_f16(rand(kv_dim * h, 3))));
        w.insert("gw", Tensor::F32(rand(hd, 4)));
        w.insert("inv", Tensor::F32(rand(hd / 2, 5)));

        let mut sess = Session::new(rt(), &w).unwrap();
        sess.set_kv_f16(false);
        let plan = sess.prepare_plan(&g).unwrap();
        assert_eq!(plan.steps.len(), 5);
        assert!(matches!(plan.steps[0].op, Step::GemvPacked { .. }));
        assert!(matches!(plan.steps[1].op, Step::GemvPacked { .. }));
        assert!(matches!(plan.steps[2].op, Step::GemvPacked { .. }));
        match &plan.steps[3].op {
            Step::RmsNormBatch {
                rope: Some(_),
                segs,
            } => {
                assert_eq!(segs.len(), 3);
                assert_eq!(
                    [segs[0].has_rope, segs[1].has_rope, segs[2].has_rope],
                    [true, true, false]
                );
            }
            other => panic!("expected RmsNormBatch, got {other:?}"),
        }
        assert!(matches!(plan.steps[4].op, Step::CachedAttention { .. }));

        let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
        for step in 0..3u32 {
            let inp = vec![
                Tensor::F32(rand(h, 10 + step as u64)),
                Tensor::U32(vec![step]),
            ];
            let want = cpu::run(&g, &w, &mut cs, inp.clone()).unwrap();
            let got = sess.run_prepared(&g, &plan, Some(&mut ms), inp).unwrap();
            let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
            assert!(err < 1e-4, "step {step}: rel_err {err}");
        }
    }

    #[test]
    fn readback_span_timing_probe() {
        let n = 262_144usize;
        let buf = new_shared_buffer(&rt().device, n * 4).unwrap();
        let ptr = buf.contents() as *mut f32;
        unsafe {
            for i in 0..n {
                *ptr.add(i) = (i as f32) * 0.001;
            }
        }
        let s = unsafe { std::slice::from_raw_parts(ptr as *const f32, n) };
        let mut to_vec_us = Vec::with_capacity(100);
        for _ in 0..100 {
            let t0 = Instant::now();
            let v = std::hint::black_box(s.to_vec());
            to_vec_us.push(t0.elapsed().as_secs_f64() * 1e6);
            drop(v);
        }
        let mut reuse = vec![0.0f32; n];
        let mut reuse_us = Vec::with_capacity(100);
        for _ in 0..100 {
            let t0 = Instant::now();
            reuse.copy_from_slice(std::hint::black_box(s));
            std::hint::black_box(&reuse);
            reuse_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        to_vec_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        reuse_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "readback_to_vec_med_us={:.2} readback_reuse_med_us={:.2}",
            to_vec_us[50], reuse_us[50]
        );
    }

    #[test]
    fn prepared_plan_precomputes_mlp_scratch_size() {
        let f = DType::F32;
        let (h, inter) = (64usize, 128usize);
        let mut g = Graph::new();
        let x = g.input("x", vec![1, h], f);
        let norm_w = g.weight("norm_w", vec![h], f);
        let gate_w = g.weight("gate_w", vec![inter, h], f);
        let up_w = g.weight("up_w", vec![inter, h], f);
        let down_w = g.weight("down_w", vec![h, inter], f);
        let ff_in = g.add("ln", Op::RmsNorm { eps: 1e-5 }, &[x, norm_w], vec![1, h], f);
        let gate = g.add("gate", Op::Linear, &[ff_in, gate_w], vec![1, inter], f);
        let up = g.add("up", Op::Linear, &[ff_in, up_w], vec![1, inter], f);
        let act = g.add("act", Op::SwiGlu, &[gate, up], vec![1, inter], f);
        let down = g.add("down", Op::Linear, &[act, down_w], vec![1, h], f);
        let out = g.add(
            "res",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[x, down],
            vec![1, h],
            f,
        );
        g.output(out);
        assert_eq!(
            crate::graph::egraph::fuse_rule(
                &mut g,
                None,
                crate::graph::egraph::Rule::MegakernelMlp
            ),
            1
        );
        let mut w = Weights::new();
        w.insert("norm_w", Tensor::F32(rand(h, 1)));
        w.insert("gate_w", Tensor::F32(rand(inter * h, 2)));
        w.insert("up_w", Tensor::F32(rand(inter * h, 3)));
        w.insert("down_w", Tensor::F32(rand(h * inter, 4)));
        let s = Session::new(rt(), &w).unwrap();
        let plan = s.prepare_plan(&g).unwrap();
        assert_eq!(plan.mlp_scratch_size, h * 4 + 2 * inter * 4);
    }

    /// Bits of a graph output: u32 ids as themselves, f32 as `to_bits`.
    fn out_bits(t: &Tensor) -> Vec<u32> {
        match t {
            Tensor::U32(v) => v.clone(),
            Tensor::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
            other => panic!("unexpected output {other:?}"),
        }
    }

    /// Runs `g` on the CPU executor and on Metal, each from the states `init`, once per input set in
    /// `runs` (state persisting between runs); returns every run's (CPU, Metal) outputs and the final
    /// (CPU, Metal) value of each state in `init`.
    #[allow(clippy::type_complexity)]
    fn both_stateful(
        g: &Graph,
        w: &Weights,
        init: &[(&str, Vec<f32>)],
        runs: Vec<Vec<Tensor>>,
    ) -> (Vec<(Vec<Tensor>, Vec<Tensor>)>, Vec<(Vec<f32>, Vec<f32>)>) {
        let (mut cs, mut gs) = (StateStore::new(), StateStore::new());
        for (name, v) in init {
            cs.insert(*name, Tensor::F32(v.clone()));
            gs.insert(*name, Tensor::F32(v.clone()));
        }
        let s = Session::new(rt(), w).unwrap();
        let outs = runs
            .into_iter()
            .map(|inputs| {
                let c = cpu::run(g, w, &mut cs, inputs.clone()).unwrap();
                let m = s.run_with_state(g, Some(&mut gs), inputs).unwrap();
                (c, m)
            })
            .collect();
        let states = init
            .iter()
            .map(|(name, _)| {
                let c = cs.get(name).unwrap().as_f32().unwrap().to_vec();
                (c, gs.read_metal(name).unwrap())
            })
            .collect();
        (outs, states)
    }

    /// `StateWrite` writes rows into a persistent state at `pos0` on both executors: the token is
    /// `pos0 + m`, a later node of the same graph reads the new rows, and the state carries into the
    /// next run.
    #[test]
    fn state_write_matches_cpu() {
        let (cap, w, m) = (6, 3, 2);
        let mut g = Graph::new();
        let st = g.state("cache", vec![cap, w], DType::F32, false);
        let rows = g.input("rows", vec![m, w], DType::F32);
        let pos0 = g.input("pos0", vec![1], DType::U32);
        let all = g.input("all", vec![cap], DType::U32);
        let token = g.push("write", Op::StateWrite, &[st, rows, pos0]).unwrap();
        let read = g.push("read", Op::Gather { axis: 0 }, &[st, all]).unwrap();
        g.output(token);
        g.output(read);
        let init = rand(cap * w, 3);
        let run = |p: u32, seed| {
            vec![
                Tensor::F32(rand(m * w, seed)),
                Tensor::U32(vec![p]),
                Tensor::U32((0..cap as u32).collect()),
            ]
        };
        let (outs, states) = both_stateful(
            &g,
            &Weights::new(),
            &[("cache", init.clone())],
            vec![run(3, 4), run(0, 5), run(4, 6)],
        );
        let mut want = init;
        for ((c, gpu), (p, seed)) in outs.iter().zip([(3usize, 4u64), (0, 5), (4, 6)]) {
            want[p * w..(p + m) * w].copy_from_slice(&rand(m * w, seed));
            assert_eq!(c[0], Tensor::U32(vec![(p + m) as u32]));
            assert_eq!(out_bits(&gpu[0]), [(p + m) as u32]);
            assert_eq!(c[1], Tensor::F32(want.clone()), "CPU reads the write");
            assert_eq!(gpu[1], Tensor::F32(want.clone()), "Metal reads the write");
        }
        assert_eq!(states[0].0, want);
        assert_eq!(states[0].1, want);
        // Dead-node elimination keeps the write even when nothing reads its token.
        let mut g2 = Graph::new();
        let st = g2.state("cache", vec![cap, w], DType::F32, false);
        let rows = g2.input("rows", vec![m, w], DType::F32);
        let pos0 = g2.input("pos0", vec![1], DType::U32);
        g2.push("write", Op::StateWrite, &[st, rows, pos0]).unwrap();
        let out = g2
            .push(
                "other",
                Op::Unary {
                    kind: crate::graph::UnaryKind::Neg,
                },
                &[rows],
            )
            .unwrap();
        g2.output(out);
        crate::graph::passes::canonical::eliminate_dead_nodes(&mut g2);
        assert!(g2.nodes.iter().any(|n| n.op == Op::StateWrite));
        // Writing past the declared rows is the same error on both executors.
        let past = || {
            vec![
                Tensor::F32(rand(m * w, 1)),
                Tensor::U32(vec![5]),
                Tensor::U32((0..cap as u32).collect()),
            ]
        };
        let mut st = StateStore::new();
        let ce = cpu::run(&g, &Weights::new(), &mut st, past()).unwrap_err();
        let me = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(&g, past())
            .unwrap_err();
        assert!(ce.contains("past the state's 6 rows"), "{ce}");
        assert_eq!(ce, me);
    }

    /// `QsaSelect` / `QsaBias` on Metal equal the CPU executor bit for bit: random scores with ties,
    /// signed zeros, infinities and NaN, rows below and above `top_blocks` complete blocks, and a
    /// score width past one threadgroup.
    #[test]
    fn qsa_select_and_bias_match_cpu() {
        for (n, b, ratio, top) in [
            (7usize, 9usize, 4usize, 2usize),
            (5, 700, 2, 37),
            (9, 300, 1, 1),
            (4, 64, 8, 0),
            (3, 33, 3, 32),
        ] {
            let mut score: Vec<f32> = rand(n * b, n as u64 + b as u64)
                .into_iter()
                .map(|v| (v * 4.0).round() * 0.5)
                .collect();
            for (i, special) in [
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                -f32::NAN,
            ]
            .into_iter()
            .enumerate()
            {
                score[(i * 7) % (n * b)] = special;
            }
            let pos: Vec<u32> = rand(n, 9)
                .into_iter()
                .map(|v| ((v + 1.0) * 0.5 * (b * ratio) as f32) as u32 % (b * ratio) as u32)
                .collect();
            for select in [true, false] {
                let mut g = Graph::new();
                let sc = g.input("score", vec![n, b], DType::F32);
                let p = g.input("pos", vec![n], DType::U32);
                let op = if select {
                    Op::QsaSelect {
                        ratio,
                        top_blocks: top,
                    }
                } else {
                    Op::QsaBias {
                        ratio,
                        top_blocks: top,
                    }
                };
                let y = g.push("qsa", op, &[sc, p]).unwrap();
                g.output(y);
                let inputs = vec![Tensor::F32(score.clone()), Tensor::U32(pos.clone())];
                let want = cpu::run_stateless(&g, &Weights::new(), inputs.clone()).unwrap();
                let got = Session::new(rt(), &Weights::new())
                    .unwrap()
                    .run(&g, inputs)
                    .unwrap();
                assert_eq!(
                    out_bits(&want[0]),
                    out_bits(&got[0]),
                    "n {n} B {b} ratio {ratio} top {top} select {select}"
                );
            }
        }
    }

    /// `DeltaRule` and `CausalConv1dCached` with state over n = 5 rows in one call, starting from a
    /// non-zero state: Metal matches the CPU executor in every output row and in the final state, over
    /// two consecutive calls.
    #[test]
    fn multi_row_recurrent_ops_with_state_match_cpu() {
        let n = 5;
        for (kh, vh, kd, vd) in [
            (1usize, 1usize, 4usize, 4usize),
            (2, 4, 16, 32),
            (2, 2, 128, 128),
        ] {
            let mut g = Graph::new();
            let f = DType::F32;
            let q = g.input("q", vec![n, kh * kd], f);
            let k = g.input("k", vec![n, kh * kd], f);
            let v = g.input("v", vec![n, vh * vd], f);
            let beta = g.input("beta", vec![n, vh], f);
            let ld = g.input("log_decay", vec![n, vh], f);
            let st = g.state("ssm", vec![vh * kd * vd], f, true);
            let op = Op::DeltaRule {
                key_heads: kh,
                val_heads: vh,
                key_dim: kd,
                val_dim: vd,
            };
            let y = g.push("delta", op, &[q, k, v, beta, ld, st]).unwrap();
            g.output(y);
            let run = |seed: u64| {
                vec![
                    Tensor::F32(rand(n * kh * kd, seed)),
                    Tensor::F32(rand(n * kh * kd, seed + 1)),
                    Tensor::F32(rand(n * vh * vd, seed + 2)),
                    Tensor::F32(
                        rand(n * vh, seed + 3)
                            .iter()
                            .map(|x| 0.5 + 0.4 * x)
                            .collect(),
                    ),
                    Tensor::F32(
                        rand(n * vh, seed + 4)
                            .iter()
                            .map(|x| -0.3 + 0.2 * x)
                            .collect(),
                    ),
                ]
            };
            let init = rand(vh * kd * vd, 7);
            let (outs, states) = both_stateful(
                &g,
                &Weights::new(),
                &[("ssm", init.clone())],
                vec![run(10), run(20)],
            );
            for (c, gpu) in &outs {
                let (c, gpu) = (c[0].as_f32().unwrap(), gpu[0].as_f32().unwrap());
                assert_eq!(c.len(), n * vh * vd);
                assert!(rel_err(c, gpu) < 1e-4, "DeltaRule {kh}/{vh}/{kd}/{vd} rows");
            }
            assert!(
                rel_err(&states[0].0, &states[0].1) < 1e-4,
                "DeltaRule state"
            );
            assert!(max_abs_diff(&states[0].0, &init) > 1e-3, "the state moved");
        }
        for (ch, kernel, dilation) in [(7usize, 4usize, 1usize), (33, 3, 2), (1, 2, 1)] {
            let mut g = Graph::new();
            let x = g.input("x", vec![n, ch], DType::F32);
            let wv = g.weight("w", vec![ch, kernel], DType::F32);
            let hist = (kernel - 1) * dilation;
            let st = g.state("conv", vec![hist, ch], DType::F32, true);
            let op = Op::CausalConv1dCached { kernel, dilation };
            let y = g.push("conv", op, &[x, wv, st]).unwrap();
            g.output(y);
            let mut w = Weights::new();
            w.insert("w", Tensor::F32(rand(ch * kernel, 3)));
            let init = rand(hist * ch, 5);
            let runs = vec![
                vec![Tensor::F32(rand(n * ch, 1))],
                vec![Tensor::F32(rand(n * ch, 2))],
            ];
            let (outs, states) = both_stateful(&g, &w, &[("conv", init)], runs);
            for (c, gpu) in &outs {
                let (c, gpu) = (c[0].as_f32().unwrap(), gpu[0].as_f32().unwrap());
                assert_eq!(c.len(), n * ch);
                assert!(rel_err(c, gpu) < 1e-5, "conv {ch}/{kernel}/{dilation} rows");
            }
            assert!(rel_err(&states[0].0, &states[0].1) < 1e-6, "conv state");
        }
    }
}
