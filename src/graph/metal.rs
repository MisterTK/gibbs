//! Metal executor for the graph IR (macOS). One run is one command buffer (tier T1 in PLAN.md):
//! activations live in one arena buffer planned from value lifetimes, weights are uploaded once per
//! [`Session`], and a Linear whose only reader is an exact GELU or a residual Add runs that op in
//! its GEMM epilogue (tier T0). Compiled pipelines are cached in an `MTLBinaryArchive` under
//! `<cache>/metal/`; the measured host profile (MPP check, dispatch cost, GEMM timings) under
//! `<cache>/host/`. Ops, shapes, or dtypes it does not implement are errors; nothing falls back to
//! the CPU.

// objc 0.2's `msg_send!` expands a `cfg(feature = "cargo-clippy")` check.
#![allow(unexpected_cfgs)]

use super::{DType, Graph, Op, Tensor, ValueId, ValueKind, Weights};
use crate::backend::metal::{commit_and_wait, new_shared_buffer, set_bytes, system_device};
use crate::hardware::{sysctl, QuantFormat};

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
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Returns whether the Metal executor implements `op` for the given data type.
/// Formats supported for graph weights uploaded to Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    U32,
    Q4_0,
    Q8_0,
    BF16,
    F16,
}

/// Returns whether the Metal executor implements `op` for the given data type.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
    match op {
        Op::Linear => matches!(
            dtype,
            DType::F32
                | DType::F16
                | DType::BF16
                | DType::Quant(QuantFormat::GGUF_Q4_0)
                | DType::Quant(QuantFormat::GGUF_Q8_0)
        ),
        Op::Embedding => matches!(dtype, DType::F32 | DType::U32),
        Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::SwiGlu
        | Op::GeGlu
        | Op::Attention { .. }
        | Op::CachedAttention { .. }
        | Op::Rope { .. }
        | Op::Slice { .. }
        | Op::CausalConv1dCached { .. }
        | Op::DeltaRule { .. }
        | Op::Add
        | Op::Mul
        | Op::Softmax { .. }
        | Op::TopKSoftmax { .. }
        | Op::MeanPool
        | Op::L2Normalize => dtype == DType::F32,
        Op::Unary {
            kind: crate::graph::UnaryKind::SoftCap(_),
        } => dtype == DType::F32,
        Op::Gelu { tanh } => !*tanh && dtype == DType::F32,
        _ => false,
    }
}

const KERNELS_MSL: &str = include_str!("metal_kernels.metal");
/// `MTLLanguageVersion4_0`; the metal crate's enum stops at 3.x.
const MSL_4_0: u64 = 0x40000;
const ARENA_ALIGN: usize = 256;
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
const ATTN_MAX_HEAD_DIM: usize = 256;
const BASE_KERNELS: [&str; 45] = [
    "embedding_f32",
    "add_f32",
    "gelu_erf_f32",
    "layer_norm_f32",
    "attention_f32",
    "mean_pool_f32",
    "l2_normalize_f32",
    "noop",
    "memory_sweep",
    "gemm_sg_none",
    "gemm_sg_gelu",
    "gemm_sg_residual",
    "gemm_rows_none",
    "gemm_rows_gelu",
    "gemm_rows_residual",
    "rmsnorm_f32",
    "graph_rope_f32",
    "swiglu_f32",
    "geglu_f32",
    "gemv_q4_0",
    "gemv_q8_0",
    "gemv_bf16",
    "gemv_f16",
    "gemv_tq2_0",
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
    "slice_f32",
    "rope_general_f32",
    "kv_append_batched",
    "attn_cached_f32",
    "causal_conv1d_f32",
    "delta_rule_f32",
    "topk_softmax_f32",
    "mul_f32",
    "softcap_f32",
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
    /// `Ok(max abs error)` when MPP `matmul2d` compiled and matched `nn::linear` on this GPU
    /// (edge tiles included); otherwise why it is not used.
    pub mpp: Result<f32, String>,
    /// GPU time of one dispatch boundary (back-to-back empty kernels), µs.
    pub dispatch_us: f64,
    /// GPU time per Linear at representative shapes; the executor picks the fastest kernel for
    /// the nearest shape.
    pub gemm: Vec<GemmTiming>,
}

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
            // Archives of earlier kernel sources for this GPU and OS can never hit again.
            let stale = std::fs::read_dir(cache.join("metal"))
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path());
            for p in stale.filter(|p| {
                *p != archive
                    && p.file_name()
                        .is_some_and(|f| f.to_string_lossy().starts_with(&tag))
            }) {
                let _ = std::fs::remove_file(p);
            }
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
                mpp: msl4.clone().map(|_| 0.0),
                dispatch_us: 0.0,
                gemm: Vec::new(),
            },
            archive_hit,
            report: String::new(),
        };
        let profile_path = cache.join("host").join(format!("{tag}.json"));
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
                rt.profile = rt.measure(msl4)?;
                let saved = profile_path
                    .parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|_| {
                        std::fs::write(&profile_path, serde_json::to_vec_pretty(&rt.profile)?)
                    });
                match saved {
                    Ok(()) => format!("host profile measured, saved {}", profile_path.display()),
                    Err(e) => format!(
                        "host profile measured; saving {} failed: {e}",
                        profile_path.display()
                    ),
                }
            }
        };
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

    pub fn profile(&self) -> &HostProfile {
        &self.profile
    }

    fn pso(&self, name: &str) -> Result<&ComputePipelineState, String> {
        self.psos
            .get(name)
            .ok_or_else(|| format!("Metal pipeline '{name}' is not available on this host"))
    }

    /// Measures host GPU streaming read bandwidth in GB/s using a 256 MB memory sweep.
    pub fn benchmark_memory_bandwidth(&self) -> Result<f64, String> {
        let pso = self.pso("memory_sweep")?;
        let buf_bytes = 256 * 1024 * 1024usize; // 256 MB
        let in_buf = self
            .device
            .new_buffer(buf_bytes as u64, MTLResourceOptions::StorageModeShared);
        let out_buf = self
            .device
            .new_buffer(64, MTLResourceOptions::StorageModeShared);
        if in_buf.contents().is_null() || out_buf.contents().is_null() {
            return Err("Failed to allocate 256 MB buffer for memory bandwidth benchmark".into());
        }
        unsafe {
            std::ptr::write_bytes(out_buf.contents() as *mut u8, 0, 64);
        }
        let count = (buf_bytes / 16) as u32; // number of ulong2 elements (16 bytes each)
        let count_buf = self.device.new_buffer_with_data(
            &count as *const u32 as *const c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let total_threads = (count / 4) as usize; // each thread reads 4 x ulong2 = 64 bytes
        let tg_size = 256usize;
        let threadgroups = (total_threads + tg_size - 1) / tg_size;

        let reps = 10;
        let us_per_rep = self.timed(reps, |enc| {
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(&in_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&count_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(threadgroups as u64, 1, 1),
                MTLSize::new(tg_size as u64, 1, 1),
            );
            Ok(())
        })?;
        let _ = unsafe { std::ptr::read_volatile(out_buf.contents() as *const u32) };

        let gb = (buf_bytes as f64) / 1e9;
        let s = us_per_rep * 1e-6;
        let gbps = if s > 0.0 { gb / s } else { 0.0 };
        Ok(gbps)
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
        let memory_bandwidth_gbps = self.benchmark_memory_bandwidth().unwrap_or(0.0);
        Ok(HostProfile {
            device: self.device.name().to_string(),
            os_build: self.profile.os_build.clone(),
            kernels: self.profile.kernels,
            memory_bandwidth_gbps,
            mpp,
            dispatch_us,
            gemm,
        })
    }

    /// Encodes `encode` `reps` times into one command buffer and returns the GPU time per
    /// repetition in µs: the fastest of 3 such buffers after one warm-up dispatch.
    fn timed(
        &self,
        reps: usize,
        encode: impl Fn(&ComputeCommandEncoderRef) -> Result<(), String>,
    ) -> Result<f64, String> {
        let mut best = f64::INFINITY;
        for n in [1, reps, reps, reps] {
            let per = metal::objc::rc::autoreleasepool(|| {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for _ in 0..n {
                    encode(enc)?;
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
        enc: &ComputeCommandEncoderRef,
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
    /// Buffer, element count, and weight storage format.
    weights: HashMap<String, (Buffer, usize, WeightFormat)>,
    arena: Mutex<Option<Buffer>>,
    state_buffers: Mutex<HashMap<String, Arc<Buffer>>>,
    gemm: Option<GemmKernel>,
}
impl Session {
    /// Uploads every weight once. Preserves packed formats (Q4_0, Q8_0, BF16, F16) without f32 expansion.
    pub fn new(rt: Arc<Runtime>, weights: &Weights) -> Result<Self, String> {
        let mut bufs = HashMap::new();
        for (name, t) in weights.iter() {
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
                Tensor::Quantized(q) => match q.as_ref() {
                    crate::model::layers::QuantizedWeight::ZeroCopySlice {
                        data,
                        format,
                        in_features,
                        out_features,
                    } if *format == QuantFormat::GGUF_Q4_0 => (
                        None,
                        data.as_ptr() as *const c_void,
                        in_features * out_features,
                        data.len(),
                        WeightFormat::Q4_0,
                    ),
                    crate::model::layers::QuantizedWeight::ZeroCopySlice {
                        data,
                        format,
                        in_features,
                        out_features,
                    } if *format == QuantFormat::GGUF_Q8_0 => (
                        None,
                        data.as_ptr() as *const c_void,
                        in_features * out_features,
                        data.len(),
                        WeightFormat::Q8_0,
                    ),
                    crate::model::layers::QuantizedWeight::ZeroCopySlice {
                        data,
                        format,
                        in_features,
                        out_features,
                    } if *format == QuantFormat::BF16 => (
                        None,
                        data.as_ptr() as *const c_void,
                        in_features * out_features,
                        data.len(),
                        WeightFormat::BF16,
                    ),
                    crate::model::layers::QuantizedWeight::ZeroCopySlice {
                        data,
                        format,
                        in_features,
                        out_features,
                    } if *format == QuantFormat::FP16 => (
                        None,
                        data.as_ptr() as *const c_void,
                        in_features * out_features,
                        data.len(),
                        WeightFormat::F16,
                    ),
                    crate::model::layers::QuantizedWeight::GgufQuantized {
                        raw_data,
                        format,
                        in_features,
                        out_features,
                    } if *format == QuantFormat::GGUF_Q4_0 => (
                        None,
                        raw_data.as_ptr() as *const c_void,
                        in_features * out_features,
                        raw_data.len(),
                        WeightFormat::Q4_0,
                    ),
                    crate::model::layers::QuantizedWeight::GgufQuantized {
                        raw_data,
                        format,
                        in_features,
                        out_features,
                    } if *format == QuantFormat::GGUF_Q8_0 => (
                        None,
                        raw_data.as_ptr() as *const c_void,
                        in_features * out_features,
                        raw_data.len(),
                        WeightFormat::Q8_0,
                    ),
                    _ => {
                        let v = t.to_f32();
                        let ptr = v.as_ptr() as *const c_void;
                        let len = v.len();
                        (Some(v), ptr, len, len * 4, WeightFormat::F32)
                    }
                },
                Tensor::Mat(m) => {
                    if let Some(p) = m.packed() {
                        let bytes = &p.map[p.off..];
                        let num_elements = m.rows * m.cols;
                        match p.format {
                            QuantFormat::GGUF_Q4_0 => (
                                None,
                                bytes.as_ptr() as *const c_void,
                                num_elements,
                                (num_elements / 32) * 18,
                                WeightFormat::Q4_0,
                            ),
                            QuantFormat::GGUF_Q8_0 => (
                                None,
                                bytes.as_ptr() as *const c_void,
                                num_elements,
                                (num_elements / 32) * 34,
                                WeightFormat::Q8_0,
                            ),
                            QuantFormat::BF16 => (
                                None,
                                bytes.as_ptr() as *const c_void,
                                num_elements,
                                num_elements * 2,
                                WeightFormat::BF16,
                            ),
                            QuantFormat::FP16 => (
                                None,
                                bytes.as_ptr() as *const c_void,
                                num_elements,
                                num_elements * 2,
                                WeightFormat::F16,
                            ),
                            _ => {
                                let v = m.to_f32();
                                let ptr = v.as_ptr() as *const c_void;
                                let len = v.len();
                                (Some(v), ptr, len, len * 4, WeightFormat::F32)
                            }
                        }
                    } else {
                        let v = m.to_f32();
                        let ptr = v.as_ptr() as *const c_void;
                        let len = v.len();
                        (Some(v), ptr, len, len * 4, WeightFormat::F32)
                    }
                }
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
            bufs.insert(name.to_string(), (buf, num_elements, fmt));
        }
        Ok(Self {
            rt,
            weights: bufs,
            arena: Mutex::new(None),
            state_buffers: Mutex::new(HashMap::new()),
            gemm: None,
        })
    }

    /// Uses `kind` for every Linear instead of the host profile's choice.
    pub fn force_gemm(&mut self, kind: GemmKernel) -> Result<(), String> {
        self.rt.pso(&format!("{}_none", kind.spec(1, 1).0))?;
        self.gemm = Some(kind);
        Ok(())
    }

    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    /// Prepares the planned steps and activation arena allocation for `graph` once.
    pub fn prepare_plan(&self, graph: &Graph) -> Result<PreparedPlan, String> {
        let (steps, aliases) = plan(graph)?;
        for s in &steps {
            for &v in &s.reads {
                let val = graph.value(v);
                if val.kind == ValueKind::Weight {
                    let (_, len, fmt) = self
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
        let (offsets, arena_size) = allocate(graph, &steps, &aliases)?;
        Ok(PreparedPlan {
            steps,
            offsets,
            arena_size,
        })
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
        let plan = self.prepare_plan(graph)?;
        self.run_prepared(graph, &plan, state, inputs)
    }

    /// Runs `graph` using a pre-planned execution schedule.
    pub fn run_prepared(
        &self,
        graph: &Graph,
        plan: &PreparedPlan,
        mut state: Option<&mut crate::graph::StateStore>,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
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
        let size = plan.arena_size;
        for s in steps {
            if let Step::Embedding { ids, rows, .. } = s.op {
                let pos = graph.inputs.iter().position(|&i| i == ids).ok_or_else(|| {
                    format!("node '{}': Metal embedding ids must be a graph input (range-checked on the host)", s.name)
                })?;
                let Tensor::U32(v) = &inputs[pos] else {
                    unreachable!("dtype checked above")
                };
                if let Some((r, &id)) = v.iter().enumerate().find(|&(_, &id)| id as usize >= rows) {
                    return Err(format!(
                        "node '{}': row {r} id {id} out of range for table of {rows} rows",
                        s.name
                    ));
                }
            }
        }
        let mut state_map: HashMap<String, Arc<Buffer>> = HashMap::new();
        if let Some(st) = &mut state {
            for s in steps {
                for &v in &s.reads {
                    if let ValueKind::State(name) = &graph.value(v).kind {
                        let n = numel(graph, v)?;
                        if !st.metal_buffers.contains_key(name) {
                            let buf = new_shared_buffer(&self.rt.device, (n * 4).max(ARENA_ALIGN))?;
                            if let Some(Tensor::F32(cur_data)) = st.buffers.get(name) {
                                let copy_len = cur_data.len().min(n);
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        cur_data.as_ptr(),
                                        buf.contents() as *mut f32,
                                        copy_len,
                                    );
                                    if n > copy_len {
                                        std::ptr::write_bytes(
                                            (buf.contents() as *mut f32).add(copy_len),
                                            0,
                                            (n - copy_len) * 4,
                                        );
                                    }
                                }
                            } else {
                                unsafe {
                                    std::ptr::write_bytes(buf.contents() as *mut u8, 0, n * 4);
                                }
                            }
                            st.metal_buffers.insert(name.clone(), (Arc::new(buf), n));
                        }
                        state_map.insert(name.clone(), st.metal_buffers[name].0.clone());
                    }
                }
            }
        } else {
            let mut sbufs = self.state_buffers.lock();
            for s in steps {
                for &v in &s.reads {
                    if let ValueKind::State(name) = &graph.value(v).kind {
                        let n = numel(graph, v)?;
                        if !sbufs.contains_key(name) {
                            let buf = new_shared_buffer(&self.rt.device, (n * 4).max(ARENA_ALIGN))?;
                            unsafe {
                                std::ptr::write_bytes(buf.contents() as *mut u8, 0, n * 4);
                            }
                            sbufs.insert(name.clone(), Arc::new(buf));
                        }
                        state_map.insert(name.clone(), sbufs[name].clone());
                    }
                }
            }
        }

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
            if let ValueKind::State(name) = &graph.value(v).kind {
                (&state_map[name], 0)
            } else if let Some(&off) = offsets.get(&v) {
                (arena, off)
            } else {
                (&self.weights[&graph.value(v).name].0, 0)
            }
        };
        metal::objc::rc::autoreleasepool(|| {
            let cb = self.rt.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let mut err = None;
            for s in steps {
                if let Err(e) = self.encode(enc, s, &loc) {
                    err = Some(e);
                    break;
                }
            }
            enc.end_encoding();
            if let Some(e) = err {
                return Err(e);
            }
            commit_and_wait(cb)
        })?;

        graph
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
                Ok(Tensor::F32(s.to_vec()))
            })
            .collect()
    }

    fn encode<'a>(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &Planned,
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
                table, ids, n, d, ..
            } => {
                enc.set_compute_pipeline_state(rt.pso("embedding_f32")?);
                bind(0, table);
                bind(1, ids);
                bind(2, s.out);
                set_bytes(enc, 3, &[u32_of(n)?, u32_of(d)?]);
                grid1(n * d);
            }
            Step::Add { a, b, n } => {
                enc.set_compute_pipeline_state(rt.pso("add_f32")?);
                bind(0, a);
                bind(1, b);
                bind(2, s.out);
                set_bytes(enc, 3, &u32_of(n)?);
                grid1(n);
            }
            Step::Gelu { x, n } => {
                enc.set_compute_pipeline_state(rt.pso("gelu_erf_f32")?);
                bind(0, x);
                bind(1, s.out);
                set_bytes(enc, 2, &u32_of(n)?);
                grid1(n);
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
            Step::Attention {
                q,
                k,
                v,
                nq,
                nk,
                heads,
                head_dim,
                causal,
            } => {
                enc.set_compute_pipeline_state(rt.pso("attention_f32")?);
                bind(0, q);
                bind(1, k);
                bind(2, v);
                bind(3, s.out);
                #[repr(C)]
                struct AttnParams(u32, u32, u32, u32, u32, f32);
                let p = AttnParams(
                    u32_of(nq)?,
                    u32_of(nk)?,
                    u32_of(heads)?,
                    u32_of(head_dim)?,
                    causal as u32,
                    1.0 / (head_dim as f32).sqrt(),
                );
                set_bytes(enc, 4, &p);
                enc.dispatch_thread_groups(
                    MTLSize::new(heads as u64, nq as u64, 1),
                    MTLSize::new(TG4, 1, 1),
                );
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
            Step::RmsNorm { x, g, rows, d, eps } => {
                enc.set_compute_pipeline_state(rt.pso("rmsnorm_f32")?);
                bind(0, x);
                bind(1, g);
                bind(2, s.out);
                #[repr(C)]
                struct RmsNormParams(u32, u32, f32);
                let p = RmsNormParams(u32_of(rows)?, u32_of(d)?, eps);
                set_bytes(enc, 3, &p);
                enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(TG, 1, 1));
            }
            Step::GemvPacked {
                x,
                w,
                bias,
                res,
                epi,
                m,
                rows,
                cols,
                format,
            } => {
                let kernel_name = match format {
                    WeightFormat::Q4_0 => "gemv_q4_0",
                    WeightFormat::Q8_0 => "gemv_q8_0",
                    WeightFormat::BF16 => "gemv_bf16",
                    WeightFormat::F16 => "gemv_f16",
                    _ => {
                        return Err(format!(
                            "Metal executor: unsupported packed GEMV format {format:?}"
                        ))
                    }
                };
                enc.set_compute_pipeline_state(rt.pso(kernel_name)?);
                bind(0, w);
                bind(1, x);
                bind(2, s.out);
                #[repr(C)]
                struct PackedGemvParams(u32, u32, u32, u32, u32);
                let epi_u32 = match epi {
                    Epi::None => 0,
                    Epi::Gelu => 1,
                    Epi::Residual => 2,
                };
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
                let rows_per_tg = match format {
                    WeightFormat::Q8_0 => 8,
                    _ => 16,
                };
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(rows_per_tg), m as u64, 1),
                    MTLSize::new(256, 1, 1),
                );
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
                let fmt_name = match format {
                    WeightFormat::BF16 => "bf16",
                    WeightFormat::F16 => "f16",
                    WeightFormat::Q4_0 => "q4_0",
                    WeightFormat::Q8_0 => "q8_0",
                    _ => {
                        return Err(format!(
                            "Metal executor: unsupported batched packed format {format:?}"
                        ))
                    }
                };
                let suffix = match epi {
                    Epi::None => "none",
                    Epi::Gelu => "gelu",
                    Epi::Residual => "residual",
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
            Step::Slice {
                x,
                rows,
                in_cols,
                start,
                out_cols,
            } => {
                enc.set_compute_pipeline_state(rt.pso("slice_f32")?);
                bind(0, x);
                bind(1, s.out);
                #[repr(C)]
                struct SliceParams {
                    rows: u32,
                    in_cols: u32,
                    start: u32,
                    out_cols: u32,
                }
                let p = SliceParams {
                    rows: rows as u32,
                    in_cols: in_cols as u32,
                    start: start as u32,
                    out_cols: out_cols as u32,
                };
                set_bytes(enc, 2, &p);
                grid1(rows * out_cols);
            }
            Step::Rope {
                x,
                pos,
                n_seq,
                n_heads,
                head_dim,
                pairs: _,
                pair_stride,
                pair_gap: _,
                rotary_dim,
                base,
            } => {
                enc.set_compute_pipeline_state(rt.pso("rope_general_f32")?);
                bind(0, x);
                bind(1, s.out);
                bind(2, pos);
                #[repr(C)]
                struct GraphRopeParams2 {
                    n_heads: u32,
                    head_dim: u32,
                    pairs: u32,
                    pair_stride: u32,
                    pair_gap: u32,
                    rotary_dim: u32,
                    base: f32,
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
                    base,
                    n_seq: n_seq as u32,
                };
                set_bytes(enc, 3, &p);
                grid1(n_seq * n_heads * head_dim);
            }
            Step::CachedAttention {
                q,
                k,
                v,
                k_state,
                v_state,
                pos,
                heads,
                head_dim,
                kv_heads,
                v_head_dim: _,
                scale,
                window,
                soft_cap,
                write,
                m,
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

                if write && k_state.is_some() && v_state.is_some() {
                    let ks = k_state.unwrap();
                    let vs = v_state.unwrap();
                    enc.set_compute_pipeline_state(rt.pso("kv_append_batched")?);
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

                enc.set_compute_pipeline_state(rt.pso("attn_cached_f32")?);
                bind(0, q);
                bind(1, k_buf_id);
                bind(2, v_buf_id);
                bind(3, s.out);
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
                };
                set_bytes(enc, 4, &p);
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
                grid1(val_heads);
            }
            Step::TopKSoftmax { x, rows, cols, k } => {
                enc.set_compute_pipeline_state(rt.pso("topk_softmax_f32")?);
                bind(0, x);
                bind(1, s.out);
                #[repr(C)]
                struct TopKSoftmaxParams {
                    rows: u32,
                    cols: u32,
                    k: u32,
                }
                let p = TopKSoftmaxParams {
                    rows: rows as u32,
                    cols: cols as u32,
                    k: k as u32,
                };
                set_bytes(enc, 2, &p);
                grid1(rows);
            }
            Step::Mul {
                a,
                b,
                total,
                a_len,
                b_len,
                row_broadcast,
                d,
            } => {
                enc.set_compute_pipeline_state(rt.pso("mul_f32")?);
                bind(0, a);
                bind(1, b);
                bind(2, s.out);
                #[repr(C)]
                struct MulParams {
                    total: u32,
                    a_len: u32,
                    b_len: u32,
                    row_broadcast: u32,
                    d: u32,
                }
                let p = MulParams {
                    total: total as u32,
                    a_len: a_len as u32,
                    b_len: b_len as u32,
                    row_broadcast: row_broadcast as u32,
                    d: d as u32,
                };
                set_bytes(enc, 3, &p);
                grid1(total);
            }
            Step::SoftCap { x, cap, n } => {
                enc.set_compute_pipeline_state(rt.pso("softcap_f32")?);
                bind(0, x);
                bind(1, s.out);
                set_bytes(enc, 2, &cap);
                set_bytes(enc, 3, &(n as u32));
                grid1(n);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Step {
    Embedding {
        table: ValueId,
        ids: ValueId,
        n: usize,
        d: usize,
        rows: usize,
    },
    Add {
        a: ValueId,
        b: ValueId,
        n: usize,
    },
    Gelu {
        x: ValueId,
        n: usize,
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
    Attention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        nq: usize,
        nk: usize,
        heads: usize,
        head_dim: usize,
        causal: bool,
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
        rows: usize,
        d: usize,
        eps: f32,
    },
    GemvPacked {
        x: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        res: Option<ValueId>,
        epi: Epi,
        m: usize,
        rows: usize,
        cols: usize,
        format: WeightFormat,
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
    Slice {
        x: ValueId,
        rows: usize,
        in_cols: usize,
        start: usize,
        out_cols: usize,
    },
    Rope {
        x: ValueId,
        pos: ValueId,
        n_seq: usize,
        n_heads: usize,
        head_dim: usize,
        pairs: usize,
        pair_stride: usize,
        pair_gap: usize,
        rotary_dim: usize,
        base: f32,
    },
    CachedAttention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        k_state: Option<ValueId>,
        v_state: Option<ValueId>,
        pos: Option<ValueId>,
        heads: usize,
        head_dim: usize,
        kv_heads: usize,
        v_head_dim: usize,
        scale: f32,
        window: Option<usize>,
        soft_cap: Option<f32>,
        write: bool,
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
    TopKSoftmax {
        x: ValueId,
        rows: usize,
        cols: usize,
        k: usize,
    },
    Mul {
        a: ValueId,
        b: ValueId,
        total: usize,
        a_len: usize,
        b_len: usize,
        row_broadcast: bool,
        d: usize,
    },
    SoftCap {
        x: ValueId,
        cap: f32,
        n: usize,
    },
}

/// One dispatch: the step, the value it writes, and every value it reads.
#[derive(Debug, Clone)]
pub struct Planned {
    pub name: String,
    pub op: Step,
    pub out: ValueId,
    pub reads: Vec<ValueId>,
}

/// A pre-planned execution schedule and arena allocation for a static graph.
#[derive(Debug, Clone)]
pub struct PreparedPlan {
    pub steps: Vec<Planned>,
    pub offsets: HashMap<ValueId, usize>,
    pub arena_size: usize,
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

/// Lowers nodes to dispatches, validating ops and shapes. A Linear whose output has exactly one
/// reader, not a graph output, absorbs that reader when it is an exact GELU or an Add whose other
/// operand already exists (a residual).
fn plan(g: &Graph) -> Result<(Vec<Planned>, HashMap<ValueId, (ValueId, usize)>), String> {
    let mut readers: HashMap<ValueId, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &v in &n.inputs {
            readers.entry(v).or_default().push(i);
        }
    }
    let exists_before = |v: ValueId, i: usize| match &g.value(v).kind {
        ValueKind::Input | ValueKind::Weight | ValueKind::State(_) => true,
        ValueKind::Node(n) => (n.0 as usize) < i,
    };
    let mut fused = vec![false; g.nodes.len()];
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
        let (op, out_len) = match &node.op {
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
                let mut format = match g.value(w).dtype {
                    DType::Quant(QuantFormat::GGUF_Q8_0) => Some(WeightFormat::Q8_0),
                    DType::Quant(QuantFormat::GGUF_Q4_0) => Some(WeightFormat::Q4_0),
                    DType::BF16 => Some(WeightFormat::BF16),
                    DType::F16 => Some(WeightFormat::F16),
                    _ => None,
                };
                if (format == Some(WeightFormat::Q4_0) || format == Some(WeightFormat::Q8_0))
                    && k % 32 != 0
                {
                    format = None;
                }
                let (mut epi, mut res) = (Epi::None, None);
                if let Some([j]) = readers.get(&out).map(Vec::as_slice) {
                    let c = &g.nodes[*j];
                    if !g.outputs.contains(&out) {
                        match c.op {
                            Op::Gelu { tanh: false } => epi = Epi::Gelu,
                            Op::Add if c.inputs.len() == 2 => {
                                let r = if c.inputs[0] == out {
                                    c.inputs[1]
                                } else {
                                    c.inputs[0]
                                };
                                if r != out && exists_before(r, i) && n_of(r)? == m * n {
                                    (epi, res) = (Epi::Residual, Some(r));
                                }
                            }
                            _ => {}
                        }
                        if epi != Epi::None {
                            fused[*j] = true;
                            out = c.outputs[0];
                        }
                    }
                }
                if let Some(fmt) = format {
                    if m >= 16 {
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
                                bias,
                                res,
                                epi,
                                m,
                                rows: n,
                                cols: k,
                                format: fmt,
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
            Op::Attention {
                heads,
                kv_heads,
                causal,
                window,
                soft_cap,
            } => {
                if kv_heads != heads || window.is_some() || soft_cap.is_some() {
                    return Err(format!(
                        "node '{name}': Attention with kv_heads != heads, window, or soft_cap is not implemented"
                    ));
                }
                let (q, k, v) = (arg(0)?, arg(1)?, arg(2)?);
                let width = g.value(q).shape.last().copied().unwrap_or(0);
                let (ql, kl) = (n_of(q)?, n_of(k)?);
                if *heads == 0
                    || width == 0
                    || width % heads != 0
                    || ql % width != 0
                    || kl % width != 0
                    || kl == 0
                    || n_of(v)? != kl
                {
                    return Err(format!(
                        "node '{name}': q/k/v shapes do not fit {heads} heads"
                    ));
                }
                let head_dim = width / heads;
                if head_dim > ATTN_MAX_HEAD_DIM {
                    return Err(format!("node '{name}': head_dim {head_dim} exceeds the Metal kernel's {ATTN_MAX_HEAD_DIM}"));
                }
                (
                    Step::Attention {
                        q,
                        k,
                        v,
                        nq: ql / width,
                        nk: kl / width,
                        heads: *heads,
                        head_dim,
                        causal: *causal,
                    },
                    ql,
                )
            }
            Op::Gelu { tanh: false } => {
                let x = arg(0)?;
                (Step::Gelu { x, n: n_of(x)? }, n_of(x)?)
            }
            Op::Add => {
                let (a, b) = (arg(0)?, arg(1)?);
                let (al, bl) = (n_of(a)?, n_of(b)?);
                if al != bl {
                    return Err(format!("node '{name}': Add of {al} and {bl} elements"));
                }
                (Step::Add { a, b, n: al }, al)
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
                (
                    Step::RmsNorm {
                        x,
                        g: g_val,
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
            Op::Slice {
                axis: _,
                start,
                end,
            } => {
                let x = arg(0)?;
                let x_shape = &g.value(x).shape;
                let (rows, in_cols) = if x_shape.len() >= 2 {
                    let cols = *x_shape.last().unwrap();
                    (n_of(x)? / cols, cols)
                } else {
                    (1, n_of(x)?)
                };
                let st = if *start < 0 {
                    (in_cols as i64 + *start).max(0) as usize
                } else {
                    *start as usize
                };
                let en = if *end < 0 {
                    (in_cols as i64 + *end).max(0) as usize
                } else {
                    (*end as usize).min(in_cols)
                };
                let out_cols = en.saturating_sub(st);
                if rows == 1 {
                    let (root_src, base_off) = aliases.get(&x).copied().unwrap_or((x, 0));
                    aliases.insert(out, (root_src, base_off + st * 4));
                    fused[i] = true;
                    continue;
                }
                (
                    Step::Slice {
                        x,
                        rows,
                        in_cols,
                        start: st,
                        out_cols,
                    },
                    rows * out_cols,
                )
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                base,
                variant,
            } => {
                let (x, pos) = (arg(0)?, arg(1)?);
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
                };
                (
                    Step::Rope {
                        x,
                        pos,
                        n_seq,
                        n_heads: heads_per_seq,
                        head_dim: hd,
                        pairs,
                        pair_stride,
                        pair_gap,
                        rotary_dim: *rotary_dim,
                        base: *base,
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
                let q = arg(0)?;
                let k = arg(1)?;
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
                        heads: *heads,
                        head_dim: *head_dim,
                        kv_heads: *kv_heads,
                        v_head_dim: *v_head_dim,
                        scale: *scale,
                        window: *window,
                        soft_cap: *soft_cap,
                        write: *write,
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
            Op::Softmax { axis: _ } | Op::TopKSoftmax { axis: _, .. } => {
                let x = arg(0)?;
                let x_shape = &g.value(x).shape;
                let cols = x_shape.last().copied().unwrap_or(1);
                let xl = n_of(x)?;
                let rows = xl / cols.max(1);
                let k_val = match node.op {
                    Op::TopKSoftmax { k, .. } => k,
                    _ => cols,
                };
                (
                    Step::TopKSoftmax {
                        x,
                        rows,
                        cols,
                        k: k_val,
                    },
                    xl,
                )
            }
            Op::Mul => {
                let (a, b) = (arg(0)?, arg(1)?);
                let (al, bl) = (n_of(a)?, n_of(b)?);
                let a_shape = &g.value(a).shape;
                let b_shape = &g.value(b).shape;
                let (row_broadcast, d) = if a_shape.len() >= 2
                    && b_shape.len() >= 2
                    && b_shape.last() == Some(&1)
                    && a_shape[0] == b_shape[0]
                {
                    (true, al / bl.max(1))
                } else {
                    (false, 1)
                };
                let total = al.max(bl);
                (
                    Step::Mul {
                        a,
                        b,
                        total,
                        a_len: al,
                        b_len: bl,
                        row_broadcast,
                        d,
                    },
                    total,
                )
            }
            Op::Unary {
                kind: crate::graph::UnaryKind::SoftCap(cap),
            } => {
                let x = arg(0)?;
                let n = n_of(x)?;
                (Step::SoftCap { x, cap: *cap, n }, n)
            }
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
            Step::Embedding { table, ids, .. } => vec![table, ids],
            Step::Add { a, b, .. } => vec![a, b],
            Step::Gelu { x, .. } | Step::L2Normalize { x, .. } => vec![x],
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
            Step::Attention { q, k, v, .. } => vec![q, k, v],
            Step::MeanPool { x, mask, .. } => vec![x, mask],
            Step::RmsNorm { x, g, .. } => vec![x, g],
            Step::SwiGlu { gate, up, .. } | Step::GeGlu { gate, up, .. } => vec![gate, up],
            Step::Slice { x, .. } => vec![x],
            Step::Rope { x, pos, .. } => vec![x, pos],
            Step::CachedAttention {
                q,
                k,
                v,
                k_state,
                v_state,
                pos,
                ..
            } => [Some(q), Some(k), Some(v), k_state, v_state, pos]
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
            Step::TopKSoftmax { x, .. } => vec![x],
            Step::Mul { a, b, .. } => vec![a, b],
            Step::SoftCap { x, .. } => vec![x],
        };
        // Operands are f32 except embedding ids, pooling mask (u32), and packed weights.
        let non_f32_operand = match op {
            Step::Embedding { ids, .. } => Some(ids),
            Step::MeanPool { mask, .. } => Some(mask),
            Step::GemvPacked { w, .. } | Step::GemmBatchedPacked { w, .. } => Some(w),
            Step::Rope { pos, .. } => Some(pos),
            Step::CachedAttention { pos: Some(p), .. } => Some(p),
            _ => None,
        };
        if let Some(&v) = reads
            .iter()
            .find(|&&v| Some(v) != non_f32_operand && g.value(v).dtype != DType::F32)
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
            reads,
        });
    }
    Ok((steps, aliases))
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
    match device.new_library_with_source(&format!("#define GIBBS_MPP 1\n{KERNELS_MSL}"), &o4) {
        Ok(lib) => Ok((lib, Ok(()))),
        Err(e4) => {
            let lib = device
                .new_library_with_source(KERNELS_MSL, &opts())
                .map_err(|e| format!("Metal shader compilation failed: {e}"))?;
            Ok((lib, Err(format!("MSL 4.0 compile failed: {e4}"))))
        }
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
                        2 => g.add("gelu", Op::Gelu { tanh: false }, &[lin], vec![m, n], f),
                        3 => g.add("res", Op::Add, &[r, lin], vec![m, n], f),
                        _ => {
                            let a = g.add("gelu", Op::Gelu { tanh: false }, &[lin], vec![m, n], f);
                            g.add("sum", Op::Add, &[a, lin], vec![m, n], f)
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
        let y = g.add("gelu", Op::Gelu { tanh: false }, &[x], vec![1601], f);
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
            let s = g.add("sum", Op::Add, &[e1, e2], vec![5, 6], f);
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
        let out = g.add("sum", Op::Add, &[swiglu, geglu], vec![2, 4], f);
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
        let qw = crate::model::layers::QuantizedWeight::GgufQuantized {
            format: QuantFormat::GGUF_Q8_0,
            raw_data: bytes::Bytes::from(raw),
            in_features: k,
            out_features: n,
        };
        let mut w = Weights::new();
        w.insert("w", Tensor::Quantized(Arc::new(qw)));
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
        let qw = crate::model::layers::QuantizedWeight::GgufQuantized {
            format: QuantFormat::GGUF_Q4_0,
            raw_data: bytes::Bytes::from(raw),
            in_features: k,
            out_features: n,
        };
        let mut w = Weights::new();
        w.insert("w", Tensor::Quantized(Arc::new(qw)));
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
        let y_res = g.add("res_add", Op::Add, &[y, res_in], vec![m, n], DType::F32);
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
        let qw = crate::model::layers::QuantizedWeight::GgufQuantized {
            format: QuantFormat::GGUF_Q4_0,
            raw_data: bytes::Bytes::from(raw),
            in_features: k,
            out_features: n,
        };
        let mut w = Weights::new();
        w.insert("w", Tensor::Quantized(Arc::new(qw)));
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
            let qw = crate::model::layers::QuantizedWeight::GgufQuantized {
                format: QuantFormat::GGUF_Q8_0,
                raw_data: bytes::Bytes::from(raw),
                in_features: k,
                out_features: n,
            };
            let mut w = Weights::new();
            w.insert("w", Tensor::Quantized(Arc::new(qw)));
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
                    let qw = crate::model::layers::QuantizedWeight::GgufQuantized {
                        format: QuantFormat::GGUF_Q4_0,
                        raw_data: bytes::Bytes::from(raw),
                        in_features: k,
                        out_features: n,
                    };
                    w.insert("w", Tensor::Quantized(Arc::new(qw)));
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
        for op in [Op::Silu, Op::Gelu { tanh: true }] {
            let mut g = Graph::new();
            let x = g.input("x", vec![4], f);
            let y = g.add("act", op, &[x], vec![4], f);
            g.output(y);
            let e = s.run(&g, vec![Tensor::F32(vec![0.0; 4])]).unwrap_err();
            assert!(e.contains("Metal executor does not implement"), "{e}");
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
        let y = g.add("sum", Op::Add, &[a, b], vec![2], f);
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
            let x = g.input("x", vec![1, 8], DType::F32);
            let pos = g.input("pos", vec![1], DType::U32);
            let y = g.add(
                "rope",
                Op::Rope {
                    head_dim: 8,
                    rotary_dim: 8,
                    base: 10000.0,
                    variant,
                },
                &[x, pos],
                vec![1, 8],
                DType::F32,
            );
            g.output(y);
            let inp = vec![
                Tensor::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                Tensor::U32(vec![3]),
            ];
            let (want, got) = both(&g, &Weights::new(), inp, None);
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
            Op::Gelu { tanh: false },
            &[v2],
            vec![1, 64],
            DType::F32,
        );
        let step3 = g.add("step3", Op::Add, &[step1, step2], vec![1, 64], DType::F32);

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
        let add_a = g.add("add_a", Op::Add, &[a, inter_sub], vec![1, 64], DType::F32);
        let gelu_b = g.add(
            "gelu_b",
            Op::Gelu { tanh: false },
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
    fn test_measure_graph_lm_dispatches_per_layer() {
        use crate::model::graph_lm::{GraphLm, GraphLmConfig};
        let cfg = GraphLmConfig::smollm_1m();
        let num_layers = cfg.num_layers;
        let model = GraphLm::new(cfg);
        let (decode_g, weights) = model.to_decode_graph();

        let s = Session::new(rt(), &weights).unwrap();
        let plan = s.prepare_plan(&decode_g).unwrap();

        let total_steps = plan.steps.len();
        let non_layer_steps = 3;
        let layer_steps = total_steps.saturating_sub(non_layer_steps);
        let steps_per_layer = layer_steps as f64 / num_layers as f64;

        println!("=== GraphLm Decode Dispatch Count ===");
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
            let k_state = g.state("k_state", vec![max_seq, kv_dim], DType::F32);
            let v_state = g.state("v_state", vec![max_seq, kv_dim], DType::F32);
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
        let session = Session::new(rt(), &Weights::new()).unwrap();

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
                rel_err(cpu_out_f32, metal_out_f32) < 1e-4,
                "Prefill output mismatch: max rel err {}",
                rel_err(cpu_out_f32, metal_out_f32)
            );

            metal_state.sync_metal_to_cpu();
            let cpu_k = cpu_state.get("k_state").unwrap().as_f32().unwrap();
            let metal_k = metal_state.get("k_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_k, metal_k) < 1e-4,
                "Prefill k_state mismatch: max rel err {}",
                rel_err(cpu_k, metal_k)
            );

            let cpu_v = cpu_state.get("v_state").unwrap().as_f32().unwrap();
            let metal_v = metal_state.get("v_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_v, metal_v) < 1e-4,
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
                rel_err(cpu_out_f32, metal_out_f32) < 1e-4,
                "Decode step {step} output mismatch: max rel err {}",
                rel_err(cpu_out_f32, metal_out_f32)
            );
            metal_state.sync_metal_to_cpu();

            let cpu_k = cpu_state.get("k_state").unwrap().as_f32().unwrap();
            let metal_k = metal_state.get("k_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_k, metal_k) < 1e-4,
                "Decode step {step} k_state mismatch: max rel err {}",
                rel_err(cpu_k, metal_k)
            );

            let cpu_v = cpu_state.get("v_state").unwrap().as_f32().unwrap();
            let metal_v = metal_state.get("v_state").unwrap().as_f32().unwrap();
            assert!(
                rel_err(cpu_v, metal_v) < 1e-4,
                "Decode step {step} v_state mismatch: max rel err {}",
                rel_err(cpu_v, metal_v)
            );
        }
    }
}
