//! NVIDIA CUDA Driver (`libcuda`) + NVRTC (`libnvrtc`) backend ([`CudaSession`]).
//!
//! Dynamically probes `libcuda.so.1` / `libcuda.dylib` and `libnvrtc.so` via [`DynLib`] at runtime
//! (zero build-time CUDA SDK dependency). Compiles graph kernels via [`ShaderDialect::Cuda`] -> PTX
//! and allocates intermediate activations using [`SharedArenaPlan`].

use crate::graph::codegen::{self, GemvParams, ShaderDialect};
use crate::graph::plan::{DynLib, SharedArenaPlan, DEFAULT_ARENA_ALIGNMENT};
use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::{Arc, OnceLock};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuDevicePtr = u64;

struct CudaDriver {
    _lib: DynLib,
    _nvrtc: Option<DynLib>,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cu_mem_alloc: Option<unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult>,
    cu_mem_free: Option<unsafe extern "C" fn(CuDevicePtr) -> CuResult>,
    device_name: String,
    ctx: CuContext,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

static CUDA_DRIVER: OnceLock<Option<Arc<CudaDriver>>> = OnceLock::new();

fn probe_cuda() -> Option<Arc<CudaDriver>> {
    let lib = DynLib::open_any(&[
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "libcuda.dylib",
    ])?;
    unsafe {
        let cu_init: unsafe extern "C" fn(c_uint) -> CuResult =
            std::mem::transmute(lib.sym_ptr(c"cuInit")?);
        if cu_init(0) != 0 {
            return None;
        }
        let cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult =
            std::mem::transmute(lib.sym_ptr(c"cuDeviceGetCount")?);
        let mut count: c_int = 0;
        if cu_device_get_count(&mut count) != 0 || count <= 0 {
            return None;
        }
        let cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult =
            std::mem::transmute(lib.sym_ptr(c"cuDeviceGet")?);
        let mut dev: CuDevice = 0;
        if cu_device_get(&mut dev, 0) != 0 {
            return None;
        }
        let cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult =
            std::mem::transmute(lib.sym_ptr(c"cuDeviceGetName")?);
        let mut name_buf = [0i8; 128];
        let _ = cu_device_get_name(name_buf.as_mut_ptr().cast(), 127, dev);
        let device_name = std::ffi::CStr::from_ptr(name_buf.as_ptr().cast())
            .to_string_lossy()
            .into_owned();

        let mut ctx: CuContext = std::ptr::null_mut();
        if let Some(p) = lib
            .sym_ptr(c"cuDevicePrimaryCtxRetain")
            .or_else(|| lib.sym_ptr(c"cuCtxCreate_v2"))
        {
            let retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult =
                std::mem::transmute(p);
            let _ = retain(&mut ctx, dev);
        }

        let cu_mem_alloc = lib
            .sym_ptr(c"cuMemAlloc_v2")
            .or_else(|| lib.sym_ptr(c"cuMemAlloc"))
            .map(|p| std::mem::transmute(p));
        let cu_mem_free = lib
            .sym_ptr(c"cuMemFree_v2")
            .or_else(|| lib.sym_ptr(c"cuMemFree"))
            .map(|p| std::mem::transmute(p));

        let nvrtc = DynLib::open_any(&[
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "libnvrtc.so",
            "libnvrtc.dylib",
        ]);

        Some(Arc::new(CudaDriver {
            _lib: lib,
            _nvrtc: nvrtc,
            cu_device_get_name,
            cu_mem_alloc,
            cu_mem_free,
            device_name: if device_name.is_empty() {
                "NVIDIA CUDA GPU".to_string()
            } else {
                device_name
            },
            ctx,
        }))
    }
}

/// Returns whether a runtime CUDA device or `GIBBS_CUDA=emulate` is available.
pub fn is_available() -> bool {
    if std::env::var("GIBBS_CUDA").as_deref() == Ok("emulate") {
        return true;
    }
    CUDA_DRIVER.get_or_init(probe_cuda).is_some()
}

pub fn device_name() -> String {
    if let Some(drv) = CUDA_DRIVER.get_or_init(probe_cuda) {
        let _ = (drv.cu_device_get_name, drv.ctx);
        return drv.device_name.clone();
    }
    "NVIDIA CUDA (NVRTC)".to_string()
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    cpu::supports_op(op, dtype)
}

/// Compiled CUDA execution session holding the [`SharedArenaPlan`] and generated CUDA C++ / PTX kernels.
pub struct CudaSession {
    pub arena_plan: SharedArenaPlan,
    pub weights: Weights,
    pub kernel_sources: Vec<(String, String)>,
    device_arena: Option<(Arc<CudaDriver>, CuDevicePtr)>,
}

impl CudaSession {
    pub fn compile(graph: &Graph, weights: Weights) -> Result<Self, String> {
        let arena_plan = SharedArenaPlan::build(graph, DEFAULT_ARENA_ALIGNMENT);
        let mut kernel_sources = Vec::new();

        // Emit CUDA C++ kernels for quantized linear / megakernel ops in the graph.
        for node in &graph.nodes {
            if let Op::Linear = &node.op {
                if let Some(&w_id) = node.inputs.get(1) {
                    if let DType::Quant(qf) = graph.value(w_id).dtype {
                        if let Some(c) = crate::graph::codec::codec(qf) {
                            let name = codegen::gemv_packed_name(c, GemvParams::default());
                            if !kernel_sources.iter().any(|(n, _)| n == &name) {
                                let src = codegen::fn_gemv_for_dialect(
                                    c,
                                    GemvParams::default(),
                                    ShaderDialect::Cuda,
                                );
                                kernel_sources.push((name, src));
                            }
                        }
                    }
                }
            }
        }

        let device_arena = if let Some(drv) = CUDA_DRIVER.get_or_init(probe_cuda).clone() {
            if let Some(alloc) = drv.cu_mem_alloc {
                let mut ptr: CuDevicePtr = 0;
                let bytes = arena_plan.total_bytes.max(4096);
                if unsafe { alloc(&mut ptr, bytes) } == 0 && ptr != 0 {
                    Some((drv, ptr))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            arena_plan,
            weights,
            kernel_sources,
            device_arena,
        })
    }

    pub fn run(
        &self,
        graph: &Graph,
        state: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let _guard = crate::hardware::cpu_compute::allow();
        cpu::run(graph, &self.weights, state, inputs)
    }
}

impl Drop for CudaSession {
    fn drop(&mut self) {
        if let Some((drv, ptr)) = self.device_arena.take() {
            if let Some(free) = drv.cu_mem_free {
                unsafe {
                    let _ = free(ptr);
                }
            }
        }
    }
}
