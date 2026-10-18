//! CUDA runtime JIT executor.
//!
//! Uses `libloading` to probe and dynamically load `libcuda.so.1` (driver)
//! and `libnvrtc.so.1` (NVRTC compiler) at runtime. On machines without NVIDIA
//! hardware or drivers (such as macOS Apple Silicon hosts), `is_available()` returns
//! `false` cleanly without link-time or compile-time dependencies.

use super::{DType, Graph, Op, Tensor, ValueId, ValueKind, Weights};
use libloading::{Library, Symbol};
use std::collections::HashMap;
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
    match op {
        Op::Linear => match dtype {
            DType::F32 | DType::F16 | DType::BF16 => true,
            DType::Quant(fmt) => crate::graph::codec::codec(fmt).is_some(),
            _ => false,
        },
        Op::MoeLinear { .. } => match dtype {
            DType::F32 => true,
            DType::Quant(fmt) => crate::graph::codec::codec(fmt).is_some(),
            _ => false,
        },
        Op::Embedding => matches!(dtype, DType::F32 | DType::U32),
        Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::SwiGlu
        | Op::GeGlu
        | Op::TopK { .. }
        | Op::GatherCols
        | Op::Softmax { .. }
        | Op::Attention { .. }
        | Op::Add
        | Op::MeanPool
        | Op::L2Normalize => dtype == DType::F32,
        Op::Gelu { tanh } => !*tanh && dtype == DType::F32,
        _ => false,
    }
}

/// Epilogue fused into a packed quantized CUDA GEMV dispatch (`PackedGemvParams.epi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaEpi {
    None,
    Gelu,
    Residual,
    Silu,
    SwiGlu,
    GeGlu,
    UpSwiGlu,
    UpGeGlu,
}

impl CudaEpi {
    pub fn code(self) -> u32 {
        match self {
            CudaEpi::None => 0,
            CudaEpi::Gelu => 1,
            CudaEpi::Residual => 2,
            CudaEpi::Silu => 3,
            CudaEpi::SwiGlu => 4,
            CudaEpi::GeGlu => 5,
            CudaEpi::UpSwiGlu => 6,
            CudaEpi::UpGeGlu => 7,
        }
    }
}

/// Planned execution step on the CUDA backend.
#[derive(Debug, Clone)]
pub enum CudaStep {
    QuantGemv {
        x: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        res: Option<ValueId>,
        out: ValueId,
        format: crate::hardware::QuantFormat,
        rows: usize,
        cols: usize,
        batch: usize,
        epi: CudaEpi,
        params: crate::graph::codegen::GemvParams,
    },
    TopK {
        x: ValueId,
        out: ValueId,
        rows: usize,
        cols: usize,
        k: usize,
    },
    GatherCols {
        x: ValueId,
        idx: ValueId,
        out: ValueId,
        total: usize,
        cols: usize,
        k: usize,
        shared: bool,
    },
    QuantMoeGemv {
        x: ValueId,
        bank: ValueId,
        ids: ValueId,
        out: ValueId,
        format: crate::hardware::QuantFormat,
        rows: usize,
        cols: usize,
        slots: usize,
        top_k: usize,
        experts: usize,
        x_per_slot: bool,
        params: crate::graph::codegen::GemvParams,
    },
    Other {
        op: Op,
        inputs: Vec<ValueId>,
        out: ValueId,
    },
}

/// Lowers `g.nodes` into [`CudaStep`]s, fusing single-reader GELU, SiLU, residual Add,
/// SwiGLU, and GeGLU epilogues into preceding packed quantized `Op::Linear` steps.
pub fn plan_graph(g: &Graph) -> Result<Vec<CudaStep>, String> {
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
    let numel = |v: ValueId| -> Result<usize, String> {
        let s = &g.value(v).shape;
        if s.is_empty() {
            return Err(format!("value '{}' has unknown shape", g.value(v).name));
        }
        Ok(s.iter().product())
    };

    let mut fused = vec![false; g.nodes.len()];
    let mut steps = Vec::with_capacity(g.nodes.len());

    for (i, node) in g.nodes.iter().enumerate() {
        if fused[i] {
            continue;
        }
        let name = &node.name;
        let mut out = *node
            .outputs
            .first()
            .ok_or_else(|| format!("node '{name}' has no output"))?;

        if node.op == Op::Linear && node.inputs.len() >= 2 {
            let x = node.inputs[0];
            let w = node.inputs[1];
            let bias = node.inputs.get(2).copied();
            if let DType::Quant(fmt) = g.value(w).dtype {
                if let Some(c) = crate::graph::codec::codec(fmt) {
                    let ws = &g.value(w).shape;
                    let [rows, cols] = ws[..] else {
                        return Err(format!(
                            "node '{name}': weight shape {ws:?} is not [out, in]"
                        ));
                    };
                    let xl = numel(x)?;
                    if cols == 0 || xl % cols != 0 {
                        return Err(format!(
                            "node '{name}': x of {xl} elements does not match weight {ws:?}"
                        ));
                    }
                    if let Some(b) = bias {
                        if numel(b)? != rows {
                            return Err(format!("node '{name}': bias is not [{rows}]"));
                        }
                    }
                    let batch = xl / cols;
                    if cols % (c.block_elems as usize) == 0 {
                        let (mut epi, mut res) = (CudaEpi::None, None);
                        if !g.outputs.contains(&out) {
                            if let Some(&[j]) = readers.get(&out).map(Vec::as_slice) {
                                if j > i && !fused[j] {
                                    let consumer = &g.nodes[j];
                                    match &consumer.op {
                                        Op::Gelu { tanh: false }
                                        | Op::Unary {
                                            kind: super::UnaryKind::GeluErf,
                                        } => {
                                            epi = CudaEpi::Gelu;
                                        }
                                        Op::Silu
                                        | Op::Unary {
                                            kind: super::UnaryKind::Silu,
                                        } => {
                                            epi = CudaEpi::Silu;
                                        }
                                        Op::SwiGlu if consumer.inputs.len() == 2 => {
                                            if consumer.inputs[0] == out {
                                                let up = consumer.inputs[1];
                                                if up != out
                                                    && exists_before(up, i)
                                                    && numel(up)? == batch * rows
                                                {
                                                    epi = CudaEpi::SwiGlu;
                                                    res = Some(up);
                                                }
                                            } else if consumer.inputs[1] == out {
                                                let gate = consumer.inputs[0];
                                                if gate != out
                                                    && exists_before(gate, i)
                                                    && numel(gate)? == batch * rows
                                                {
                                                    epi = CudaEpi::UpSwiGlu;
                                                    res = Some(gate);
                                                }
                                            }
                                        }
                                        Op::GeGlu if consumer.inputs.len() == 2 => {
                                            if consumer.inputs[0] == out {
                                                let up = consumer.inputs[1];
                                                if up != out
                                                    && exists_before(up, i)
                                                    && numel(up)? == batch * rows
                                                {
                                                    epi = CudaEpi::GeGlu;
                                                    res = Some(up);
                                                }
                                            } else if consumer.inputs[1] == out {
                                                let gate = consumer.inputs[0];
                                                if gate != out
                                                    && exists_before(gate, i)
                                                    && numel(gate)? == batch * rows
                                                {
                                                    epi = CudaEpi::UpGeGlu;
                                                    res = Some(gate);
                                                }
                                            }
                                        }
                                        Op::Add
                                        | Op::Binary {
                                            kind: super::BinaryKind::Add,
                                        } if consumer.inputs.len() == 2 => {
                                            let r = if consumer.inputs[0] == out {
                                                consumer.inputs[1]
                                            } else {
                                                consumer.inputs[0]
                                            };
                                            if r != out
                                                && exists_before(r, i)
                                                && numel(r)? == batch * rows
                                            {
                                                epi = CudaEpi::Residual;
                                                res = Some(r);
                                            }
                                        }
                                        _ => {}
                                    }
                                    if epi != CudaEpi::None {
                                        fused[j] = true;
                                        out = *consumer.outputs.first().ok_or_else(|| {
                                            format!("node '{}' has no output", consumer.name)
                                        })?;
                                    }
                                }
                            }
                        }
                        steps.push(CudaStep::QuantGemv {
                            x,
                            w,
                            bias,
                            res,
                            out,
                            format: fmt,
                            rows,
                            cols,
                            batch,
                            epi,
                            params: crate::graph::codegen::GemvParams::default(),
                        });
                        continue;
                    }
                }
            }
        }

        match &node.op {
            Op::TopK { k } => {
                let x = *node
                    .inputs
                    .first()
                    .ok_or_else(|| format!("node '{name}': TopK needs operand 0"))?;
                let cols = g.value(x).shape.last().copied().unwrap_or(0);
                if *k == 0 || *k > cols {
                    return Err(format!(
                        "node '{name}': TopK needs 1 <= k <= the row width {cols}, got k={k}"
                    ));
                }
                let n = numel(x)?;
                steps.push(CudaStep::TopK {
                    x,
                    out,
                    rows: n / cols,
                    cols,
                    k: *k,
                });
                continue;
            }
            Op::GatherCols => {
                if node.inputs.len() < 2 {
                    return Err(format!("node '{name}': GatherCols needs 2 operands"));
                }
                let (x, idx) = (node.inputs[0], node.inputs[1]);
                let cols = g.value(x).shape.last().copied().unwrap_or(0);
                let k = g.value(idx).shape.last().copied().unwrap_or(0);
                let total = numel(idx)?;
                let x_rows = numel(x)?.checked_div(cols).unwrap_or(0);
                if cols == 0
                    || k == 0
                    || (x_rows != total / k && x_rows != 1)
                    || g.value(idx).dtype != DType::U32
                {
                    return Err(format!(
                        "node '{name}': GatherCols needs x [rows, n] (or one shared row) and u32 idx [rows, k]"
                    ));
                }
                steps.push(CudaStep::GatherCols {
                    x,
                    idx,
                    out,
                    total,
                    cols,
                    k,
                    shared: x_rows == 1,
                });
                continue;
            }
            Op::MoeLinear { experts, top_k } => {
                if node.inputs.len() != 3 {
                    return Err(format!(
                        "node '{name}': MoeLinear needs [x, bank, ids], got {} operands",
                        node.inputs.len()
                    ));
                }
                let (x, bank, ids) = (node.inputs[0], node.inputs[1], node.inputs[2]);
                if *experts == 0 || *top_k == 0 {
                    return Err(format!(
                        "node '{name}': MoeLinear needs experts >= 1 and top_k >= 1"
                    ));
                }
                let [x_rows, cols] = g.value(x).shape[..] else {
                    return Err(format!(
                        "node '{name}': MoeLinear needs x [m, in] and a bank [experts * out, in]"
                    ));
                };
                let [bank_rows, bank_cols] = g.value(bank).shape[..] else {
                    return Err(format!(
                        "node '{name}': MoeLinear needs x [m, in] and a bank [experts * out, in]"
                    ));
                };
                if bank_cols != cols || bank_rows % experts != 0 || bank_rows == 0 {
                    return Err(format!(
                        "node '{name}': bank [{bank_rows}, {bank_cols}] is not {experts} stacked [out, {cols}] matrices"
                    ));
                }
                let slots = numel(ids)?;
                if g.value(ids).dtype != DType::U32 || slots % top_k != 0 {
                    return Err(format!("node '{name}': ids must be u32 [m, {top_k}]"));
                }
                if x_rows != slots && x_rows * top_k != slots {
                    return Err(format!(
                        "node '{name}': x has {x_rows} rows, neither m nor m * {top_k} for {slots} slots"
                    ));
                }
                let rows = bank_rows / experts;
                if numel(out)? != slots * rows {
                    return Err(format!("node '{name}': output must be [{slots}, {rows}]"));
                }
                if let DType::Quant(fmt) = g.value(bank).dtype {
                    if let Some(c) = crate::graph::codec::codec(fmt) {
                        if cols % (c.block_elems as usize) == 0 {
                            steps.push(CudaStep::QuantMoeGemv {
                                x,
                                bank,
                                ids,
                                out,
                                format: fmt,
                                rows,
                                cols,
                                slots,
                                top_k: *top_k,
                                experts: *experts,
                                x_per_slot: x_rows == slots,
                                params: crate::graph::codegen::GemvParams::default(),
                            });
                            continue;
                        }
                    }
                }
            }
            _ => {}
        }

        steps.push(CudaStep::Other {
            op: node.op.clone(),
            inputs: node.inputs.clone(),
            out,
        });
    }

    Ok(steps)
}

/// Compiles a CUDA C++ translation unit into PTX via runtime NVRTC (`libnvrtc`).
pub fn compile_ptx(src: &str, name: &str) -> Result<Vec<u8>, String> {
    let nvrtc = NVRTC.as_ref().ok_or_else(|| {
        "NVRTC library (libnvrtc.so.1 / nvrtc.dll) is not available on this platform".to_string()
    })?;
    let c_src =
        std::ffi::CString::new(src).map_err(|e| format!("NVRTC source contains NUL byte: {e}"))?;
    let c_name =
        std::ffi::CString::new(name).map_err(|e| format!("NVRTC name contains NUL byte: {e}"))?;
    unsafe {
        let mut prog: *mut c_void = std::ptr::null_mut();
        let res = (nvrtc.nvrtc_create_program)(
            &mut prog,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if res != CUDA_SUCCESS {
            return Err(format!("nvrtcCreateProgram failed with error code {res}"));
        }
        let compile_res = (nvrtc.nvrtc_compile_program)(prog, 0, std::ptr::null());
        if compile_res != CUDA_SUCCESS {
            let _ = (nvrtc.nvrtc_destroy_program)(&mut prog);
            return Err(format!(
                "nvrtcCompileProgram({name}) failed with error code {compile_res}"
            ));
        }
        let mut ptx_size = 0usize;
        let size_res = (nvrtc.nvrtc_get_ptx_size)(prog, &mut ptx_size);
        if size_res != CUDA_SUCCESS || ptx_size == 0 {
            let _ = (nvrtc.nvrtc_destroy_program)(&mut prog);
            return Err(format!("nvrtcGetPTXSize failed with error code {size_res}"));
        }
        let mut ptx = vec![0u8; ptx_size];
        let get_res = (nvrtc.nvrtc_get_ptx)(prog, ptx.as_mut_ptr() as *mut c_char);
        let _ = (nvrtc.nvrtc_destroy_program)(&mut prog);
        if get_res != CUDA_SUCCESS {
            return Err(format!("nvrtcGetPTX failed with error code {get_res}"));
        }
        Ok(ptx)
    }
}

/// Synthesizes a self-contained CUDA C++ quantized GEMV translation unit for `format` and `p`.
pub fn quant_gemv_kernel_source(
    format: crate::hardware::QuantFormat,
    p: crate::graph::codegen::GemvParams,
) -> Option<String> {
    let c = crate::graph::codec::codec(format)?;
    Some(crate::graph::codegen::c_dialect_translation_unit(
        c,
        p,
        crate::graph::codegen::Dialect::Cuda,
        "",
        "",
    ))
}

/// Synthesizes a self-contained CUDA C++ quantized expert-indexed GEMV translation unit for `format` and `p`.
pub fn quant_moe_gemv_kernel_source(
    format: crate::hardware::QuantFormat,
    p: crate::graph::codegen::GemvParams,
) -> Option<String> {
    let c = crate::graph::codec::codec(format)?;
    Some(crate::graph::codegen::c_dialect_indexed_translation_unit(
        crate::graph::codegen::Dialect::Cuda,
        c,
        p,
    ))
}

/// JIT-compiled CUDA quantized GEMV module and kernel function handle.
pub struct CudaQuantGemv {
    pub module: *mut c_void,
    pub func: *mut c_void,
    pub kernel_name: String,
    pub ptx: Vec<u8>,
}

unsafe impl Send for CudaQuantGemv {}
unsafe impl Sync for CudaQuantGemv {}

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

    pub fn run(&self, graph: &Graph, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        let steps = plan_graph(graph)?;
        let mut _compiled = Vec::new();
        for step in &steps {
            match step {
                CudaStep::QuantGemv { format, params, .. } => {
                    _compiled.push(self.compile_quant_gemv(*format, *params)?);
                }
                CudaStep::QuantMoeGemv { format, params, .. } => {
                    _compiled.push(self.compile_quant_moe_gemv(*format, *params)?);
                }
                _ => {}
            }
        }
        Err("CUDA execution graph compilation requires runtime NVRTC toolchain".to_string())
    }

    /// JIT-compiles a packed `Codec` quantized GEMV kernel via `c_dialect_translation_unit` and NVRTC.
    pub fn compile_quant_gemv(
        &self,
        fmt: crate::hardware::QuantFormat,
        p: crate::graph::codegen::GemvParams,
    ) -> Result<CudaQuantGemv, String> {
        let codec = crate::graph::codec::codec(fmt)
            .ok_or_else(|| format!("no packed GPU codec registered for {fmt:?}"))?;
        let src = crate::graph::codegen::c_dialect_translation_unit(
            codec,
            p,
            crate::graph::codegen::Dialect::Cuda,
            "",
            "",
        );
        let format_str = format!("{:?}", codec.format).to_lowercase();
        let kernel_name = format!("gemv_{format_str}");
        let ptx = compile_ptx(&src, &kernel_name)?;
        let driver = CUDA_DRIVER.as_ref().ok_or_else(|| {
            "CUDA driver (libcuda.so.1 / nvcuda.dll) is not available on this platform".to_string()
        })?;
        unsafe {
            let mut module: *mut c_void = std::ptr::null_mut();
            let res = (driver.cu_module_load_data)(&mut module, ptx.as_ptr() as *const c_void);
            if res != CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData failed with error code {res}"));
            }
            let c_func = std::ffi::CString::new(kernel_name.as_str())
                .map_err(|e| format!("invalid kernel name: {e}"))?;
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = (driver.cu_module_get_function)(&mut func, module, c_func.as_ptr());
            if res != CUDA_SUCCESS {
                return Err(format!(
                    "cuModuleGetFunction({kernel_name}) failed with error code {res}"
                ));
            }
            Ok(CudaQuantGemv {
                module,
                func,
                kernel_name,
                ptx,
            })
        }
    }

    /// JIT-compiles a packed `Codec` quantized MoE indexed GEMV kernel via `c_dialect_indexed_translation_unit` and NVRTC.
    pub fn compile_quant_moe_gemv(
        &self,
        fmt: crate::hardware::QuantFormat,
        p: crate::graph::codegen::GemvParams,
    ) -> Result<CudaQuantGemv, String> {
        let codec = crate::graph::codec::codec(fmt)
            .ok_or_else(|| format!("no packed GPU codec registered for {fmt:?}"))?;
        let src = crate::graph::codegen::c_dialect_indexed_translation_unit(
            crate::graph::codegen::Dialect::Cuda,
            codec,
            p,
        );
        let kernel_name = crate::graph::codegen::gemv_indexed_name(codec, p);
        let ptx = compile_ptx(&src, &kernel_name)?;
        let driver = CUDA_DRIVER.as_ref().ok_or_else(|| {
            "CUDA driver (libcuda.so.1 / nvcuda.dll) is not available on this platform".to_string()
        })?;
        unsafe {
            let mut module: *mut c_void = std::ptr::null_mut();
            let res = (driver.cu_module_load_data)(&mut module, ptx.as_ptr() as *const c_void);
            if res != CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData failed with error code {res}"));
            }
            let c_func = std::ffi::CString::new(kernel_name.as_str())
                .map_err(|e| format!("invalid kernel name: {e}"))?;
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = (driver.cu_module_get_function)(&mut func, module, c_func.as_ptr());
            if res != CUDA_SUCCESS {
                return Err(format!(
                    "cuModuleGetFunction({kernel_name}) failed with error code {res}"
                ));
            }
            Ok(CudaQuantGemv {
                module,
                func,
                kernel_name,
                ptx,
            })
        }
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
    use crate::graph::codegen::GemvParams;
    use crate::hardware::QuantFormat;

    #[test]
    fn test_cuda_probe_on_current_host() {
        // On macOS arm64, CUDA driver is not present.
        assert!(!is_available());
        assert!(supports_op(&Op::Linear, DType::F32));
        assert!(CudaSession::new(&Weights::new()).is_err());
    }

    #[test]
    fn test_cuda_quant_gemv_kernel_source() {
        let p = GemvParams::default();
        for fmt in [
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_K,
            QuantFormat::GGUF_IQ4_XS,
            QuantFormat::GGUF_TQ2_0,
        ] {
            let src = quant_gemv_kernel_source(fmt, p)
                .unwrap_or_else(|| panic!("expected CUDA GEMV source for {fmt:?}"));
            assert!(src.contains("extern \"C\" __global__ void gemv_"));
            assert!(src.contains("decode8("));
            assert!(src.contains("__shfl_xor_sync(0xffffffffu, sum, 16);"));
            assert!(!src.contains("[[buffer"));
            assert!(!src.contains("device const"));
            assert!(!src.contains("simd_sum("));
        }
        assert!(quant_gemv_kernel_source(QuantFormat::FP32, p).is_none());
    }

    #[test]
    fn test_cuda_plan_quant_gemv_and_epilogue_fusion() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 256], DType::F32);
        let res_in = g.input("res_in", vec![1, 256], DType::F32);
        let up_in = g.input("up_in", vec![1, 256], DType::F32);
        let w_gelu = g.weight(
            "w_gelu",
            vec![256, 256],
            DType::Quant(QuantFormat::GGUF_Q4_K),
        );
        let w_add = g.weight(
            "w_add",
            vec![256, 256],
            DType::Quant(QuantFormat::GGUF_IQ4_XS),
        );
        let w_gate = g.weight(
            "w_gate",
            vec![256, 256],
            DType::Quant(QuantFormat::GGUF_Q4_K),
        );
        let w_up = g.weight(
            "w_up",
            vec![256, 256],
            DType::Quant(QuantFormat::GGUF_IQ4_XS),
        );

        let lin_gelu = g.add(
            "lin_gelu",
            Op::Linear,
            &[x, w_gelu],
            vec![1, 256],
            DType::F32,
        );
        let act_gelu = g.add(
            "act_gelu",
            Op::Gelu { tanh: false },
            &[lin_gelu],
            vec![1, 256],
            DType::F32,
        );

        let lin_add = g.add(
            "lin_add",
            Op::Linear,
            &[act_gelu, w_add],
            vec![1, 256],
            DType::F32,
        );
        let out_add = g.add(
            "out_add",
            Op::Add,
            &[lin_add, res_in],
            vec![1, 256],
            DType::F32,
        );

        let lin_gate = g.add(
            "lin_gate",
            Op::Linear,
            &[out_add, w_gate],
            vec![1, 256],
            DType::F32,
        );
        let out_swiglu = g.add(
            "out_swiglu",
            Op::SwiGlu,
            &[lin_gate, up_in],
            vec![1, 256],
            DType::F32,
        );

        let gate2 = g.add(
            "gate2",
            Op::Linear,
            &[out_swiglu, w_gate],
            vec![1, 256],
            DType::F32,
        );
        let up2 = g.add(
            "up2",
            Op::Linear,
            &[out_swiglu, w_up],
            vec![1, 256],
            DType::F32,
        );
        let out_upswiglu = g.add(
            "out_upswiglu",
            Op::SwiGlu,
            &[gate2, up2],
            vec![1, 256],
            DType::F32,
        );
        g.output(out_upswiglu);

        let steps = plan_graph(&g).expect("plan_graph should succeed");
        assert_eq!(steps.len(), 5);

        let expected = [
            (QuantFormat::GGUF_Q4_K, CudaEpi::Gelu, 1u32, None, act_gelu),
            (
                QuantFormat::GGUF_IQ4_XS,
                CudaEpi::Residual,
                2u32,
                Some(res_in),
                out_add,
            ),
            (
                QuantFormat::GGUF_Q4_K,
                CudaEpi::SwiGlu,
                4u32,
                Some(up_in),
                out_swiglu,
            ),
            (QuantFormat::GGUF_Q4_K, CudaEpi::None, 0u32, None, gate2),
            (
                QuantFormat::GGUF_IQ4_XS,
                CudaEpi::UpSwiGlu,
                6u32,
                Some(gate2),
                out_upswiglu,
            ),
        ];

        for (step, (exp_fmt, exp_epi, exp_code, exp_res, exp_out)) in
            steps.iter().zip(expected.iter())
        {
            match step {
                CudaStep::QuantGemv {
                    format,
                    rows,
                    cols,
                    batch,
                    epi,
                    res,
                    out,
                    params,
                    ..
                } => {
                    assert_eq!(*format, *exp_fmt);
                    assert_eq!(*rows, 256);
                    assert_eq!(*cols, 256);
                    assert_eq!(*batch, 1);
                    assert_eq!(*epi, *exp_epi);
                    assert_eq!(epi.code(), *exp_code);
                    assert_eq!(*res, *exp_res);
                    assert_eq!(*out, *exp_out);
                    let src = quant_gemv_kernel_source(*format, *params)
                        .expect("expected CUDA kernel source");
                    assert!(src.contains("extern \"C\" __global__ void gemv_"));
                    assert!(src.contains("apply_gemv_epi"));
                }
                other => panic!("expected QuantGemv step, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_cuda_plan_moe_quant_gemv() {
        assert!(supports_op(&Op::Linear, DType::F32));
        assert!(supports_op(&Op::TopK { k: 2 }, DType::F32));
        assert!(supports_op(&Op::GatherCols, DType::F32));
        assert!(supports_op(&Op::Softmax { axis: -1 }, DType::F32));
        assert!(supports_op(
            &Op::MoeLinear {
                experts: 8,
                top_k: 2,
            },
            DType::F32
        ));
        assert!(supports_op(
            &Op::MoeLinear {
                experts: 8,
                top_k: 2,
            },
            DType::Quant(QuantFormat::GGUF_IQ2_XXS)
        ));
        assert!(supports_op(
            &Op::MoeLinear {
                experts: 8,
                top_k: 2,
            },
            DType::Quant(QuantFormat::GGUF_Q4_K_M)
        ));

        let (m, d_in, d_ff, experts, top_k) = (1usize, 256usize, 256usize, 8usize, 2usize);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, d_in], DType::F32);
        let w_router = g.weight("w_router", vec![experts, d_in], DType::F32);
        let bank_gu = g.weight(
            "bank_gu",
            vec![experts * d_ff, d_in],
            DType::Quant(QuantFormat::GGUF_IQ2_XXS),
        );
        let bank_down = g.weight(
            "bank_down",
            vec![experts * d_in, d_ff],
            DType::Quant(QuantFormat::GGUF_Q4_K_M),
        );

        let logits = g.add(
            "logits",
            Op::Linear,
            &[x, w_router],
            vec![m, experts],
            DType::F32,
        );
        let ids = g.add(
            "ids",
            Op::TopK { k: top_k },
            &[logits],
            vec![m, top_k],
            DType::U32,
        );
        let picked = g.add(
            "picked",
            Op::GatherCols,
            &[logits, ids],
            vec![m, top_k],
            DType::F32,
        );
        let probs = g.add(
            "probs",
            Op::Softmax { axis: -1 },
            &[picked],
            vec![m, top_k],
            DType::F32,
        );
        let moe_gu = g.add(
            "moe_gu",
            Op::MoeLinear { experts, top_k },
            &[x, bank_gu, ids],
            vec![m * top_k, d_ff],
            DType::F32,
        );
        let moe_down = g.add(
            "moe_down",
            Op::MoeLinear { experts, top_k },
            &[moe_gu, bank_down, ids],
            vec![m * top_k, d_in],
            DType::F32,
        );
        g.output(probs);
        g.output(moe_down);

        let steps = plan_graph(&g).expect("CUDA MoE plan_graph should succeed");
        assert_eq!(steps.len(), 6);

        assert!(matches!(
            steps[0],
            CudaStep::Other {
                op: Op::Linear,
                out,
                ..
            } if out == logits
        ));
        assert!(matches!(
            steps[1],
            CudaStep::TopK {
                x: sx,
                out: sout,
                rows: 1,
                cols: 8,
                k: 2,
            } if sx == logits && sout == ids
        ));
        assert!(matches!(
            steps[2],
            CudaStep::GatherCols {
                x: sx,
                idx: sidx,
                out: sout,
                total: 2,
                cols: 8,
                k: 2,
                shared: true,
            } if sx == logits && sidx == ids && sout == picked
        ));
        assert!(matches!(
            steps[3],
            CudaStep::Other {
                op: Op::Softmax { axis: -1 },
                out,
                ..
            } if out == probs
        ));
        match &steps[4] {
            CudaStep::QuantMoeGemv {
                x: sx,
                bank: sbank,
                ids: sids,
                out: sout,
                format,
                rows,
                cols,
                slots,
                top_k: sk,
                experts: sexp,
                x_per_slot,
                params,
            } => {
                assert_eq!(*sx, x);
                assert_eq!(*sbank, bank_gu);
                assert_eq!(*sids, ids);
                assert_eq!(*sout, moe_gu);
                assert_eq!(*format, QuantFormat::GGUF_IQ2_XXS);
                assert_eq!(*rows, d_ff);
                assert_eq!(*cols, d_in);
                assert_eq!(*slots, m * top_k);
                assert_eq!(*sk, top_k);
                assert_eq!(*sexp, experts);
                assert!(!*x_per_slot);
                let src = quant_moe_gemv_kernel_source(*format, *params)
                    .expect("expected CUDA MoE kernel source for IQ2_XXS");
                assert!(src.contains("extern \"C\" __global__ void gemvgen_GGUF_IQ2_XXS_r2_ix"));
            }
            other => panic!("expected QuantMoeGemv at step 4, got {other:?}"),
        }
        match &steps[5] {
            CudaStep::QuantMoeGemv {
                x: sx,
                bank: sbank,
                ids: sids,
                out: sout,
                format,
                rows,
                cols,
                slots,
                top_k: sk,
                experts: sexp,
                x_per_slot,
                params,
            } => {
                assert_eq!(*sx, moe_gu);
                assert_eq!(*sbank, bank_down);
                assert_eq!(*sids, ids);
                assert_eq!(*sout, moe_down);
                assert_eq!(*format, QuantFormat::GGUF_Q4_K_M);
                assert_eq!(*rows, d_in);
                assert_eq!(*cols, d_ff);
                assert_eq!(*slots, m * top_k);
                assert_eq!(*sk, top_k);
                assert_eq!(*sexp, experts);
                assert!(*x_per_slot);
                let src = quant_moe_gemv_kernel_source(*format, *params)
                    .expect("expected CUDA MoE kernel source for Q4_K_M");
                assert!(src.contains("extern \"C\" __global__ void gemvgen_GGUF_Q4_K_M_r2_ix"));
            }
            other => panic!("expected QuantMoeGemv at step 5, got {other:?}"),
        }
    }
}
