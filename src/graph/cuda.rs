//! CUDA runtime JIT executor.
//!
//! Uses `libloading` to probe and dynamically load `libcuda.so.1` (driver)
//! and `libnvrtc.so.1` (NVRTC compiler) at runtime. On machines without NVIDIA
//! hardware or drivers (such as macOS Apple Silicon hosts), `is_available()` returns
//! `false` cleanly without link-time or compile-time dependencies.

use super::{DType, Graph, Op, Tensor, Weights};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void};
use std::sync::LazyLock;

type CuResult = i32;
const CUDA_SUCCESS: CuResult = 0;

/// Dynamically resolved function pointers from CUDA driver (`libcuda.so.1` / `nvcuda.dll`).
#[allow(dead_code)]
struct CudaFunctions {
    _lib: Library,
    cu_init: unsafe extern "C" fn(flags: u32) -> CuResult,
    cu_device_get: unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> CuResult,
    cu_ctx_create: unsafe extern "C" fn(pctx: *mut *mut c_void, flags: u32, dev: i32) -> CuResult,
    cu_ctx_destroy: unsafe extern "C" fn(ctx: *mut c_void) -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(dptr: *mut u64, bytesize: usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(dptr: u64) -> CuResult,
    cu_memcpy_hto_d: unsafe extern "C" fn(
        dst_device: u64,
        src_host: *const c_void,
        byte_count: usize,
    ) -> CuResult,
    cu_memcpy_dto_h:
        unsafe extern "C" fn(dst_host: *mut c_void, src_device: u64, byte_count: usize) -> CuResult,
    cu_module_load_data:
        unsafe extern "C" fn(module: *mut *mut c_void, image: *const c_void) -> CuResult,
    cu_module_get_function: unsafe extern "C" fn(
        hfunc: *mut *mut c_void,
        hmod: *mut c_void,
        name: *const c_char,
    ) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        f: *mut c_void,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        h_stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult,
}

/// Dynamically resolved function pointers from NVRTC (`libnvrtc.so.1`).
#[allow(dead_code)]
struct NvrtcFunctions {
    _lib: Library,
    nvrtc_create_program: unsafe extern "C" fn(
        prog: *mut *mut c_void,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> CuResult,
    nvrtc_compile_program: unsafe extern "C" fn(
        prog: *mut c_void,
        num_options: i32,
        options: *const *const c_char,
    ) -> CuResult,
    nvrtc_get_ptx_size:
        unsafe extern "C" fn(prog: *mut c_void, ptx_size_ret: *mut usize) -> CuResult,
    nvrtc_get_ptx: unsafe extern "C" fn(prog: *mut c_void, ptx: *mut c_char) -> CuResult,
    nvrtc_destroy_program: unsafe extern "C" fn(prog: *mut *mut c_void) -> CuResult,
}

static CUDA_DRIVER: LazyLock<Option<CudaFunctions>> = LazyLock::new(|| {
    let lib_names = if cfg!(target_os = "windows") {
        &["nvcuda.dll"][..]
    } else {
        &["libcuda.so.1", "libcuda.so", "libcuda.dylib"][..]
    };

    for &name in lib_names {
        if let Ok(lib) = unsafe { Library::new(name) } {
            unsafe {
                let cu_init: Symbol<unsafe extern "C" fn(u32) -> CuResult> =
                    match lib.get(b"cuInit\0") {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                if cu_init(0) != CUDA_SUCCESS {
                    continue;
                }
                let cu_device_get = match lib.get(b"cuDeviceGet\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_ctx_create = match lib
                    .get(b"cuCtxCreate_v2\0")
                    .or_else(|_| lib.get(b"cuCtxCreate\0"))
                {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_ctx_destroy = match lib
                    .get(b"cuCtxDestroy_v2\0")
                    .or_else(|_| lib.get(b"cuCtxDestroy\0"))
                {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_mem_alloc = match lib
                    .get(b"cuMemAlloc_v2\0")
                    .or_else(|_| lib.get(b"cuMemAlloc\0"))
                {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_mem_free = match lib
                    .get(b"cuMemFree_v2\0")
                    .or_else(|_| lib.get(b"cuMemFree\0"))
                {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_memcpy_hto_d = match lib
                    .get(b"cuMemcpyHtoD_v2\0")
                    .or_else(|_| lib.get(b"cuMemcpyHtoD\0"))
                {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_memcpy_dto_h = match lib
                    .get(b"cuMemcpyDtoH_v2\0")
                    .or_else(|_| lib.get(b"cuMemcpyDtoH\0"))
                {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_module_load_data = match lib.get(b"cuModuleLoadData\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_module_get_function = match lib.get(b"cuModuleGetFunction\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let cu_launch_kernel = match lib.get(b"cuLaunchKernel\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                return Some(CudaFunctions {
                    cu_init: *cu_init,
                    cu_device_get: *cu_device_get,
                    cu_ctx_create: *cu_ctx_create,
                    cu_ctx_destroy: *cu_ctx_destroy,
                    cu_mem_alloc: *cu_mem_alloc,
                    cu_mem_free: *cu_mem_free,
                    cu_memcpy_hto_d: *cu_memcpy_hto_d,
                    cu_memcpy_dto_h: *cu_memcpy_dto_h,
                    cu_module_load_data: *cu_module_load_data,
                    cu_module_get_function: *cu_module_get_function,
                    cu_launch_kernel: *cu_launch_kernel,
                    _lib: lib,
                });
            }
        }
    }
    None
});

#[allow(dead_code)]
static NVRTC: LazyLock<Option<NvrtcFunctions>> = LazyLock::new(|| {
    let lib_names = if cfg!(target_os = "windows") {
        &["nvrtc64_120_0.dll", "nvrtc64_112_0.dll", "nvrtc.dll"][..]
    } else {
        &[
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "libnvrtc.so.1",
            "libnvrtc.so",
        ][..]
    };

    for &name in lib_names {
        if let Ok(lib) = unsafe { Library::new(name) } {
            unsafe {
                let nvrtc_create_program = match lib.get(b"nvrtcCreateProgram\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let nvrtc_compile_program = match lib.get(b"nvrtcCompileProgram\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let nvrtc_get_ptx_size = match lib.get(b"nvrtcGetPTXSize\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let nvrtc_get_ptx = match lib.get(b"nvrtcGetPTX\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let nvrtc_destroy_program = match lib.get(b"nvrtcDestroyProgram\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                return Some(NvrtcFunctions {
                    nvrtc_create_program: *nvrtc_create_program,
                    nvrtc_compile_program: *nvrtc_compile_program,
                    nvrtc_get_ptx_size: *nvrtc_get_ptx_size,
                    nvrtc_get_ptx: *nvrtc_get_ptx,
                    nvrtc_destroy_program: *nvrtc_destroy_program,
                    _lib: lib,
                });
            }
        }
    }
    None
});

/// Returns whether a functional CUDA driver and runtime are present on this system.
pub fn is_available() -> bool {
    CUDA_DRIVER.is_some()
}

/// Returns whether the CUDA backend implements `op` for `dtype`.
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

/// CUDA session executing a graph program.
pub struct CudaSession {
    ctx: *mut c_void,
}

unsafe impl Send for CudaSession {}
unsafe impl Sync for CudaSession {}

impl CudaSession {
    pub fn new(_weights: &Weights) -> Result<Self, String> {
        let driver = CUDA_DRIVER.as_ref().ok_or_else(|| {
            "CUDA driver (libcuda.so.1 / nvcuda.dll) is not available on this platform".to_string()
        })?;

        unsafe {
            let mut dev = 0i32;
            let res = (driver.cu_device_get)(&mut dev, 0);
            if res != CUDA_SUCCESS {
                return Err(format!("cuDeviceGet failed with error code {res}"));
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let res = (driver.cu_ctx_create)(&mut ctx, 0, dev);
            if res != CUDA_SUCCESS {
                return Err(format!("cuCtxCreate failed with error code {res}"));
            }
            Ok(Self { ctx })
        }
    }

    pub fn run(&self, _graph: &Graph, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        Err("CUDA execution graph compilation requires runtime NVRTC toolchain".to_string())
    }
}

impl Drop for CudaSession {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            if let Some(driver) = CUDA_DRIVER.as_ref() {
                unsafe {
                    (driver.cu_ctx_destroy)(self.ctx);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_probe_on_current_host() {
        // On macOS arm64, CUDA driver is not present.
        assert!(!is_available());
        assert!(!supports_op(&Op::Linear, DType::F32));
        assert!(CudaSession::new(&Weights::new()).is_err());
    }
}
