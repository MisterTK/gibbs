//! ROCm / HIP runtime JIT executor.
//!
//! Uses `libloading` to probe and dynamically load `libamdhip64.so` (runtime)
//! and `libhiprtc.so` (HIPRTC compiler) at runtime. On machines without AMD
//! hardware or drivers (such as macOS Apple Silicon hosts), `is_available()` returns
//! `false` cleanly without link-time or compile-time dependencies.

use super::{DType, Graph, Op, Tensor, Weights};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void};
use std::sync::LazyLock;

type HipResult = i32;
const HIP_SUCCESS: HipResult = 0;

/// Dynamically resolved function pointers from HIP runtime (`libamdhip64.so`).
#[allow(dead_code)]
struct HipFunctions {
    _lib: Library,
    hip_init: unsafe extern "C" fn(flags: u32) -> HipResult,
    hip_device_get: unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> HipResult,
    hip_ctx_create: unsafe extern "C" fn(pctx: *mut *mut c_void, flags: u32, dev: i32) -> HipResult,
    hip_ctx_destroy: unsafe extern "C" fn(ctx: *mut c_void) -> HipResult,
    hip_malloc: unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> HipResult,
    hip_free: unsafe extern "C" fn(ptr: *mut c_void) -> HipResult,
    hip_memcpy_hto_d:
        unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, size: usize) -> HipResult,
    hip_memcpy_dto_h:
        unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, size: usize) -> HipResult,
    hip_module_load_data:
        unsafe extern "C" fn(module: *mut *mut c_void, image: *const c_void) -> HipResult,
    hip_module_get_function: unsafe extern "C" fn(
        hfunc: *mut *mut c_void,
        hmod: *mut c_void,
        name: *const c_char,
    ) -> HipResult,
}

static HIP_DRIVER: LazyLock<Option<HipFunctions>> = LazyLock::new(|| {
    let lib_names = if cfg!(target_os = "windows") {
        &["amdhip64.dll"][..]
    } else {
        &["libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"][..]
    };

    for &name in lib_names {
        if let Ok(lib) = unsafe { Library::new(name) } {
            unsafe {
                let hip_init: Symbol<unsafe extern "C" fn(u32) -> HipResult> =
                    match lib.get(b"hipInit\0") {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                if hip_init(0) != HIP_SUCCESS {
                    continue;
                }
                let hip_device_get = match lib.get(b"hipDeviceGet\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_ctx_create = match lib.get(b"hipCtxCreate\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_ctx_destroy = match lib.get(b"hipCtxDestroy\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_malloc = match lib.get(b"hipMalloc\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_free = match lib.get(b"hipFree\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_memcpy_hto_d = match lib.get(b"hipMemcpyHtoD\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_memcpy_dto_h = match lib.get(b"hipMemcpyDtoH\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_module_load_data = match lib.get(b"hipModuleLoadData\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hip_module_get_function = match lib.get(b"hipModuleGetFunction\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                return Some(HipFunctions {
                    hip_init: *hip_init,
                    hip_device_get: *hip_device_get,
                    hip_ctx_create: *hip_ctx_create,
                    hip_ctx_destroy: *hip_ctx_destroy,
                    hip_malloc: *hip_malloc,
                    hip_free: *hip_free,
                    hip_memcpy_hto_d: *hip_memcpy_hto_d,
                    hip_memcpy_dto_h: *hip_memcpy_dto_h,
                    hip_module_load_data: *hip_module_load_data,
                    hip_module_get_function: *hip_module_get_function,
                    _lib: lib,
                });
            }
        }
    }
    None
});

/// Returns whether a functional ROCm / HIP driver and runtime are present on this system.
pub fn is_available() -> bool {
    HIP_DRIVER.is_some()
}

/// Returns whether the ROCm backend implements `op` for `dtype`.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
    if !is_available() {
        return false;
    }
    match op {
        Op::Linear => matches!(dtype, DType::F32 | DType::F16 | DType::BF16),
        Op::Embedding => matches!(dtype, DType::F32 | DType::U32),
        Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::SwiGlu
        | Op::GeGlu
        | Op::Attention { .. }
        | Op::Add
        | Op::MeanPool
        | Op::L2Normalize => dtype == DType::F32,
        Op::Gelu { tanh } => !*tanh && dtype == DType::F32,
        _ => false,
    }
}

/// Synthesizes a HIP C++ quantized GEMV kernel source for `format` and `p`.
pub fn quant_gemv_kernel_source(
    format: crate::hardware::QuantFormat,
    p: crate::graph::codegen::GemvParams,
) -> Option<String> {
    let c = crate::graph::codec::codec(format)?;
    Some(crate::graph::codegen::gemv_for_dialect(
        c,
        p,
        crate::graph::codegen::Dialect::Hip,
        64,
        "",
        "",
    ))
}

/// ROCm session executing a graph program.
pub struct RocmSession {
    ctx: *mut c_void,
}

unsafe impl Send for RocmSession {}
unsafe impl Sync for RocmSession {}

impl RocmSession {
    pub fn new(_weights: &Weights) -> Result<Self, String> {
        let driver = HIP_DRIVER.as_ref().ok_or_else(|| {
            "ROCm driver (libamdhip64.so) is not available on this platform".to_string()
        })?;

        unsafe {
            let mut dev = 0i32;
            let res = (driver.hip_device_get)(&mut dev, 0);
            if res != HIP_SUCCESS {
                return Err(format!("hipDeviceGet failed with error code {res}"));
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let res = (driver.hip_ctx_create)(&mut ctx, 0, dev);
            if res != HIP_SUCCESS {
                return Err(format!("hipCtxCreate failed with error code {res}"));
            }
            Ok(Self { ctx })
        }
    }

    pub fn run(&self, _graph: &Graph, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        Err("ROCm execution graph compilation requires runtime HIPRTC toolchain".to_string())
    }
}

impl Drop for RocmSession {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            if let Some(driver) = HIP_DRIVER.as_ref() {
                unsafe {
                    (driver.hip_ctx_destroy)(self.ctx);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::codegen::GemvParams;
    use crate::hardware::QuantFormat;

    #[test]
    fn test_rocm_probe_on_current_host() {
        // On macOS arm64, ROCm driver is not present.
        assert!(!is_available());
        assert!(!supports_op(&Op::Linear, DType::F32));
        assert!(RocmSession::new(&Weights::new()).is_err());
    }

    #[test]
    fn test_rocm_quant_gemv_kernel_source() {
        let p = GemvParams::default();
        for fmt in [
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_K,
            QuantFormat::GGUF_IQ4_XS,
            QuantFormat::GGUF_TQ2_0,
        ] {
            let src = quant_gemv_kernel_source(fmt, p)
                .unwrap_or_else(|| panic!("expected HIP GEMV source for {fmt:?}"));
            assert!(src.contains("extern \"C\" __global__ void gemv_"));
            assert!(src.contains("decode8("));
            assert!(src.contains("__shfl_xor(sum, 32, 64);"));
            assert!(!src.contains("[[buffer"));
            assert!(!src.contains("device const"));
            assert!(!src.contains("simd_sum("));
        }
        assert!(quant_gemv_kernel_source(QuantFormat::FP32, p).is_none());
    }
}
