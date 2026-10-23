//! AMD ROCm HIP (`libamdhip64`) + hipRTC (`libhiprtc`) backend ([`RocmSession`]).
//!
//! Dynamically probes `libamdhip64.so` and `libhiprtc.so` via [`DynLib`] at runtime (zero
//! build-time ROCm SDK dependency). Emits wave32/wave64-compatible HIP kernels via
//! [`ShaderDialect::Hip`] and packs intermediate activations using [`SharedArenaPlan`].

use crate::graph::codegen::{self, GemvParams, ShaderDialect};
use crate::graph::plan::{DynLib, SharedArenaPlan, DEFAULT_ARENA_ALIGNMENT};
use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
use std::ffi::{c_int, c_uint, c_void};
use std::sync::{Arc, OnceLock};

type HipError = c_int;

struct RocmDriver {
    _lib: DynLib,
    _hiprtc: Option<DynLib>,
    hip_malloc: Option<unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError>,
    hip_free: Option<unsafe extern "C" fn(*mut c_void) -> HipError>,
    device_name: String,
}

unsafe impl Send for RocmDriver {}
unsafe impl Sync for RocmDriver {}

static ROCM_DRIVER: OnceLock<Option<Arc<RocmDriver>>> = OnceLock::new();

fn probe_rocm() -> Option<Arc<RocmDriver>> {
    let lib = DynLib::open_any(&[
        "libamdhip64.so.6",
        "libamdhip64.so.5",
        "libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so",
    ])?;
    unsafe {
        let hip_init: unsafe extern "C" fn(c_uint) -> HipError =
            std::mem::transmute(lib.sym_ptr(c"hipInit")?);
        if hip_init(0) != 0 {
            return None;
        }
        let hip_get_device_count: unsafe extern "C" fn(*mut c_int) -> HipError =
            std::mem::transmute(lib.sym_ptr(c"hipGetDeviceCount")?);
        let mut count: c_int = 0;
        if hip_get_device_count(&mut count) != 0 || count <= 0 {
            return None;
        }
        let hip_malloc = lib.sym_ptr(c"hipMalloc").map(|p| std::mem::transmute(p));
        let hip_free = lib.sym_ptr(c"hipFree").map(|p| std::mem::transmute(p));
        let hiprtc = DynLib::open_any(&[
            "libhiprtc.so.6",
            "libhiprtc.so.5",
            "libhiprtc.so",
            "/opt/rocm/lib/libhiprtc.so",
        ]);
        Some(Arc::new(RocmDriver {
            _lib: lib,
            _hiprtc: hiprtc,
            hip_malloc,
            hip_free,
            device_name: "AMD ROCm HIP GPU".to_string(),
        }))
    }
}

pub fn is_available() -> bool {
    if std::env::var("GIBBS_ROCM").as_deref() == Ok("emulate") {
        return true;
    }
    ROCM_DRIVER.get_or_init(probe_rocm).is_some()
}

pub fn device_name() -> String {
    if let Some(drv) = ROCM_DRIVER.get_or_init(probe_rocm) {
        return drv.device_name.clone();
    }
    "AMD ROCm (HIP)".to_string()
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    cpu::supports_op(op, dtype)
}

pub struct RocmSession {
    pub arena_plan: SharedArenaPlan,
    pub weights: Weights,
    pub kernel_sources: Vec<(String, String)>,
    device_arena: Option<(Arc<RocmDriver>, *mut c_void)>,
}

unsafe impl Send for RocmSession {}
unsafe impl Sync for RocmSession {}

impl RocmSession {
    pub fn compile(graph: &Graph, weights: Weights) -> Result<Self, String> {
        let arena_plan = SharedArenaPlan::build(graph, DEFAULT_ARENA_ALIGNMENT);
        let mut kernel_sources = Vec::new();
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
                                    ShaderDialect::Hip,
                                );
                                kernel_sources.push((name, src));
                            }
                        }
                    }
                }
            }
        }

        let device_arena = if let Some(drv) = ROCM_DRIVER.get_or_init(probe_rocm).clone() {
            if let Some(alloc) = drv.hip_malloc {
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let bytes = arena_plan.total_bytes.max(4096);
                if unsafe { alloc(&mut ptr, bytes) } == 0 && !ptr.is_null() {
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

impl Drop for RocmSession {
    fn drop(&mut self) {
        if let Some((drv, ptr)) = self.device_arena.take() {
            if let Some(free) = drv.hip_free {
                unsafe {
                    let _ = free(ptr);
                }
            }
        }
    }
}
