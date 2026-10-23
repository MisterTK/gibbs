//! Intel Arc / Data Center Max XPU (`libOpenCL` / `libze_loader`) backend ([`IntelXpuSession`]).
//!
//! Dynamically probes OpenCL / Level Zero runtime libraries via [`DynLib`], emits subgroup-accelerated
//! OpenCL C / SYCL kernels via [`ShaderDialect::OpenClSycl`], and manages intermediate tensor memory
//! with [`SharedArenaPlan`].

use crate::graph::codegen::{self, GemvParams, ShaderDialect};
use crate::graph::plan::{DynLib, SharedArenaPlan, DEFAULT_ARENA_ALIGNMENT};
use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
use std::ffi::{c_int, c_uint, c_void};
use std::sync::{Arc, OnceLock};

struct IntelXpuDriver {
    _lib: DynLib,
    pub name: String,
}

static INTEL_XPU_DRIVER: OnceLock<Option<Arc<IntelXpuDriver>>> = OnceLock::new();

fn probe_intel_xpu() -> Option<Arc<IntelXpuDriver>> {
    // Prefer Level Zero (`libze_loader`) on Linux/Windows Intel XPU systems; OpenCL (`libOpenCL`)
    // only when an Intel GPU device is present.
    if let Some(ze) = DynLib::open_any(&["libze_loader.so.1", "libze_loader.so"]) {
        unsafe {
            let ze_init: unsafe extern "C" fn(c_uint) -> c_int =
                std::mem::transmute(ze.sym_ptr(c"zeInit")?);
            if ze_init(0) == 0 {
                return Some(Arc::new(IntelXpuDriver {
                    _lib: ze,
                    name: "Intel XPU (Level Zero)".to_string(),
                }));
            }
        }
    }
    let cl = DynLib::open_any(&["libOpenCL.so.1", "libOpenCL.so"])?;
    unsafe {
        let get_platforms: unsafe extern "C" fn(c_uint, *mut *mut c_void, *mut c_uint) -> c_int =
            std::mem::transmute(cl.sym_ptr(c"clGetPlatformIDs")?);
        let mut n_plat: c_uint = 0;
        if get_platforms(0, std::ptr::null_mut(), &mut n_plat) != 0 || n_plat == 0 {
            return None;
        }
        Some(Arc::new(IntelXpuDriver {
            _lib: cl,
            name: "Intel XPU (OpenCL/SYCL)".to_string(),
        }))
    }
}

pub fn is_available() -> bool {
    if std::env::var("GIBBS_INTEL_XPU").as_deref() == Ok("emulate") {
        return true;
    }
    INTEL_XPU_DRIVER.get_or_init(probe_intel_xpu).is_some()
}

pub fn device_name() -> String {
    if let Some(d) = INTEL_XPU_DRIVER.get_or_init(probe_intel_xpu) {
        return d.name.clone();
    }
    "Intel XPU (OpenCL / SYCL)".to_string()
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    cpu::supports_op(op, dtype)
}

pub struct IntelXpuSession {
    pub arena_plan: SharedArenaPlan,
    pub weights: Weights,
    pub kernel_sources: Vec<(String, String)>,
}

impl IntelXpuSession {
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
                                    ShaderDialect::OpenClSycl,
                                );
                                kernel_sources.push((name, src));
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            arena_plan,
            weights,
            kernel_sources,
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
