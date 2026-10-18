//! ROCm / HIP runtime JIT executor.
//!
//! Uses `libloading` to probe and dynamically load `libamdhip64.so` (runtime)
//! and `libhiprtc.so` (HIPRTC compiler) at runtime. On machines without AMD
//! hardware or drivers (such as macOS Apple Silicon hosts), `is_available()` returns
//! `false` cleanly without link-time or compile-time dependencies.

use super::{DType, Graph, Op, Tensor, ValueId, ValueKind, Weights};
use libloading::{Library, Symbol};
use std::collections::HashMap;
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

/// Dynamically resolved function pointers from HIPRTC (`libhiprtc.so`).
#[allow(dead_code)]
struct HiprtcFunctions {
    _lib: Library,
    hiprtc_create_program: unsafe extern "C" fn(
        prog: *mut *mut c_void,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> HipResult,
    hiprtc_compile_program: unsafe extern "C" fn(
        prog: *mut c_void,
        num_options: i32,
        options: *const *const c_char,
    ) -> HipResult,
    hiprtc_get_code_size:
        unsafe extern "C" fn(prog: *mut c_void, code_size_ret: *mut usize) -> HipResult,
    hiprtc_get_code: unsafe extern "C" fn(prog: *mut c_void, code: *mut c_char) -> HipResult,
    hiprtc_destroy_program: unsafe extern "C" fn(prog: *mut *mut c_void) -> HipResult,
}

#[allow(dead_code)]
static HIPRTC: LazyLock<Option<HiprtcFunctions>> = LazyLock::new(|| {
    let lib_names = if cfg!(target_os = "windows") {
        &["hiprtc.dll"][..]
    } else {
        &["libhiprtc.so.6", "libhiprtc.so.5", "libhiprtc.so"][..]
    };

    for &name in lib_names {
        if let Ok(lib) = unsafe { Library::new(name) } {
            unsafe {
                let hiprtc_create_program = match lib.get(b"hiprtcCreateProgram\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hiprtc_compile_program = match lib.get(b"hiprtcCompileProgram\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hiprtc_get_code_size = match lib.get(b"hiprtcGetCodeSize\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hiprtc_get_code = match lib.get(b"hiprtcGetCode\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let hiprtc_destroy_program = match lib.get(b"hiprtcDestroyProgram\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                return Some(HiprtcFunctions {
                    hiprtc_create_program: *hiprtc_create_program,
                    hiprtc_compile_program: *hiprtc_compile_program,
                    hiprtc_get_code_size: *hiprtc_get_code_size,
                    hiprtc_get_code: *hiprtc_get_code,
                    hiprtc_destroy_program: *hiprtc_destroy_program,
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

/// Epilogue fused into a packed quantized ROCm/HIP GEMV dispatch (`PackedGemvParams.epi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmEpi {
    None,
    Gelu,
    Residual,
    Silu,
    SwiGlu,
    GeGlu,
    UpSwiGlu,
    UpGeGlu,
}

impl RocmEpi {
    pub fn code(self) -> u32 {
        match self {
            RocmEpi::None => 0,
            RocmEpi::Gelu => 1,
            RocmEpi::Residual => 2,
            RocmEpi::Silu => 3,
            RocmEpi::SwiGlu => 4,
            RocmEpi::GeGlu => 5,
            RocmEpi::UpSwiGlu => 6,
            RocmEpi::UpGeGlu => 7,
        }
    }
}

/// Planned execution step on the ROCm/HIP backend.
#[derive(Debug, Clone)]
pub enum RocmStep {
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
        epi: RocmEpi,
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

/// Lowers `g.nodes` into [`RocmStep`]s, fusing single-reader GELU, SiLU, residual Add,
/// SwiGLU, and GeGLU epilogues into preceding packed quantized `Op::Linear` steps.
pub fn plan_graph(g: &Graph) -> Result<Vec<RocmStep>, String> {
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
                        let (mut epi, mut res) = (RocmEpi::None, None);
                        if !g.outputs.contains(&out) {
                            if let Some(&[j]) = readers.get(&out).map(Vec::as_slice) {
                                if j > i && !fused[j] {
                                    let consumer = &g.nodes[j];
                                    match &consumer.op {
                                        Op::Gelu { tanh: false }
                                        | Op::Unary {
                                            kind: super::UnaryKind::GeluErf,
                                        } => {
                                            epi = RocmEpi::Gelu;
                                        }
                                        Op::Silu
                                        | Op::Unary {
                                            kind: super::UnaryKind::Silu,
                                        } => {
                                            epi = RocmEpi::Silu;
                                        }
                                        Op::SwiGlu if consumer.inputs.len() == 2 => {
                                            if consumer.inputs[0] == out {
                                                let up = consumer.inputs[1];
                                                if up != out
                                                    && exists_before(up, i)
                                                    && numel(up)? == batch * rows
                                                {
                                                    epi = RocmEpi::SwiGlu;
                                                    res = Some(up);
                                                }
                                            } else if consumer.inputs[1] == out {
                                                let gate = consumer.inputs[0];
                                                if gate != out
                                                    && exists_before(gate, i)
                                                    && numel(gate)? == batch * rows
                                                {
                                                    epi = RocmEpi::UpSwiGlu;
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
                                                    epi = RocmEpi::GeGlu;
                                                    res = Some(up);
                                                }
                                            } else if consumer.inputs[1] == out {
                                                let gate = consumer.inputs[0];
                                                if gate != out
                                                    && exists_before(gate, i)
                                                    && numel(gate)? == batch * rows
                                                {
                                                    epi = RocmEpi::UpGeGlu;
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
                                                epi = RocmEpi::Residual;
                                                res = Some(r);
                                            }
                                        }
                                        _ => {}
                                    }
                                    if epi != RocmEpi::None {
                                        fused[j] = true;
                                        out = *consumer.outputs.first().ok_or_else(|| {
                                            format!("node '{}' has no output", consumer.name)
                                        })?;
                                    }
                                }
                            }
                        }
                        steps.push(RocmStep::QuantGemv {
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
                steps.push(RocmStep::TopK {
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
                steps.push(RocmStep::GatherCols {
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
                            steps.push(RocmStep::QuantMoeGemv {
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

        steps.push(RocmStep::Other {
            op: node.op.clone(),
            inputs: node.inputs.clone(),
            out,
        });
    }

    Ok(steps)
}

/// Compiles a HIP C++ translation unit into HSACO code object bytes via runtime HIPRTC (`libhiprtc`).
pub fn compile_hsaco(src: &str, name: &str) -> Result<Vec<u8>, String> {
    let hiprtc = HIPRTC.as_ref().ok_or_else(|| {
        "HIPRTC library (libhiprtc.so / hiprtc.dll) is not available on this platform".to_string()
    })?;
    let c_src =
        std::ffi::CString::new(src).map_err(|e| format!("HIPRTC source contains NUL byte: {e}"))?;
    let c_name =
        std::ffi::CString::new(name).map_err(|e| format!("HIPRTC name contains NUL byte: {e}"))?;
    unsafe {
        let mut prog: *mut c_void = std::ptr::null_mut();
        let res = (hiprtc.hiprtc_create_program)(
            &mut prog,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if res != HIP_SUCCESS {
            return Err(format!("hiprtcCreateProgram failed with error code {res}"));
        }
        let compile_res = (hiprtc.hiprtc_compile_program)(prog, 0, std::ptr::null());
        if compile_res != HIP_SUCCESS {
            let _ = (hiprtc.hiprtc_destroy_program)(&mut prog);
            return Err(format!(
                "hiprtcCompileProgram({name}) failed with error code {compile_res}"
            ));
        }
        let mut code_size = 0usize;
        let size_res = (hiprtc.hiprtc_get_code_size)(prog, &mut code_size);
        if size_res != HIP_SUCCESS || code_size == 0 {
            let _ = (hiprtc.hiprtc_destroy_program)(&mut prog);
            return Err(format!(
                "hiprtcGetCodeSize failed with error code {size_res}"
            ));
        }
        let mut hsaco = vec![0u8; code_size];
        let get_res = (hiprtc.hiprtc_get_code)(prog, hsaco.as_mut_ptr() as *mut c_char);
        let _ = (hiprtc.hiprtc_destroy_program)(&mut prog);
        if get_res != HIP_SUCCESS {
            return Err(format!("hiprtcGetCode failed with error code {get_res}"));
        }
        Ok(hsaco)
    }
}

/// Synthesizes a self-contained HIP C++ quantized GEMV translation unit for `format` and `p`.
pub fn quant_gemv_kernel_source(
    format: crate::hardware::QuantFormat,
    p: crate::graph::codegen::GemvParams,
) -> Option<String> {
    let c = crate::graph::codec::codec(format)?;
    Some(crate::graph::codegen::c_dialect_translation_unit(
        c,
        p,
        crate::graph::codegen::Dialect::Hip,
        "",
        "",
    ))
}

/// Synthesizes a self-contained HIP C++ quantized expert-indexed GEMV translation unit for `format` and `p`.
pub fn quant_moe_gemv_kernel_source(
    format: crate::hardware::QuantFormat,
    p: crate::graph::codegen::GemvParams,
) -> Option<String> {
    let c = crate::graph::codec::codec(format)?;
    Some(crate::graph::codegen::c_dialect_indexed_translation_unit(
        crate::graph::codegen::Dialect::Hip,
        c,
        p,
    ))
}

/// JIT-compiled ROCm/HIP quantized GEMV module and kernel function handle.
pub struct RocmQuantGemv {
    pub module: *mut c_void,
    pub func: *mut c_void,
    pub kernel_name: String,
    pub hsaco: Vec<u8>,
}

unsafe impl Send for RocmQuantGemv {}
unsafe impl Sync for RocmQuantGemv {}

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

    pub fn run(&self, graph: &Graph, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        let steps = plan_graph(graph)?;
        let mut _compiled = Vec::new();
        for step in &steps {
            match step {
                RocmStep::QuantGemv { format, params, .. } => {
                    _compiled.push(self.compile_quant_gemv(*format, *params)?);
                }
                RocmStep::QuantMoeGemv { format, params, .. } => {
                    _compiled.push(self.compile_quant_moe_gemv(*format, *params)?);
                }
                _ => {}
            }
        }
        Err("ROCm execution graph compilation requires runtime HIPRTC toolchain".to_string())
    }

    /// JIT-compiles a packed `Codec` quantized GEMV kernel via `c_dialect_translation_unit` and HIPRTC.
    pub fn compile_quant_gemv(
        &self,
        fmt: crate::hardware::QuantFormat,
        p: crate::graph::codegen::GemvParams,
    ) -> Result<RocmQuantGemv, String> {
        let codec = crate::graph::codec::codec(fmt)
            .ok_or_else(|| format!("no packed GPU codec registered for {fmt:?}"))?;
        let src = crate::graph::codegen::c_dialect_translation_unit(
            codec,
            p,
            crate::graph::codegen::Dialect::Hip,
            "",
            "",
        );
        let format_str = format!("{:?}", codec.format).to_lowercase();
        let kernel_name = format!("gemv_{format_str}");
        let hsaco = compile_hsaco(&src, &kernel_name)?;
        let driver = HIP_DRIVER.as_ref().ok_or_else(|| {
            "ROCm driver (libamdhip64.so) is not available on this platform".to_string()
        })?;
        unsafe {
            let mut module: *mut c_void = std::ptr::null_mut();
            let res = (driver.hip_module_load_data)(&mut module, hsaco.as_ptr() as *const c_void);
            if res != HIP_SUCCESS {
                return Err(format!("hipModuleLoadData failed with error code {res}"));
            }
            let c_func = std::ffi::CString::new(kernel_name.as_str())
                .map_err(|e| format!("invalid kernel name: {e}"))?;
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = (driver.hip_module_get_function)(&mut func, module, c_func.as_ptr());
            if res != HIP_SUCCESS {
                return Err(format!(
                    "hipModuleGetFunction({kernel_name}) failed with error code {res}"
                ));
            }
            Ok(RocmQuantGemv {
                module,
                func,
                kernel_name,
                hsaco,
            })
        }
    }

    /// JIT-compiles a packed `Codec` quantized MoE indexed GEMV kernel via `c_dialect_indexed_translation_unit` and HIPRTC.
    pub fn compile_quant_moe_gemv(
        &self,
        fmt: crate::hardware::QuantFormat,
        p: crate::graph::codegen::GemvParams,
    ) -> Result<RocmQuantGemv, String> {
        let codec = crate::graph::codec::codec(fmt)
            .ok_or_else(|| format!("no packed GPU codec registered for {fmt:?}"))?;
        let src = crate::graph::codegen::c_dialect_indexed_translation_unit(
            crate::graph::codegen::Dialect::Hip,
            codec,
            p,
        );
        let kernel_name = crate::graph::codegen::gemv_indexed_name(codec, p);
        let hsaco = compile_hsaco(&src, &kernel_name)?;
        let driver = HIP_DRIVER.as_ref().ok_or_else(|| {
            "ROCm driver (libamdhip64.so) is not available on this platform".to_string()
        })?;
        unsafe {
            let mut module: *mut c_void = std::ptr::null_mut();
            let res = (driver.hip_module_load_data)(&mut module, hsaco.as_ptr() as *const c_void);
            if res != HIP_SUCCESS {
                return Err(format!("hipModuleLoadData failed with error code {res}"));
            }
            let c_func = std::ffi::CString::new(kernel_name.as_str())
                .map_err(|e| format!("invalid kernel name: {e}"))?;
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = (driver.hip_module_get_function)(&mut func, module, c_func.as_ptr());
            if res != HIP_SUCCESS {
                return Err(format!(
                    "hipModuleGetFunction({kernel_name}) failed with error code {res}"
                ));
            }
            Ok(RocmQuantGemv {
                module,
                func,
                kernel_name,
                hsaco,
            })
        }
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
        assert!(supports_op(&Op::Linear, DType::F32));
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

    #[test]
    fn test_rocm_plan_quant_gemv_and_epilogue_fusion() {
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
            (QuantFormat::GGUF_Q4_K, RocmEpi::Gelu, 1u32, None, act_gelu),
            (
                QuantFormat::GGUF_IQ4_XS,
                RocmEpi::Residual,
                2u32,
                Some(res_in),
                out_add,
            ),
            (
                QuantFormat::GGUF_Q4_K,
                RocmEpi::SwiGlu,
                4u32,
                Some(up_in),
                out_swiglu,
            ),
            (QuantFormat::GGUF_Q4_K, RocmEpi::None, 0u32, None, gate2),
            (
                QuantFormat::GGUF_IQ4_XS,
                RocmEpi::UpSwiGlu,
                6u32,
                Some(gate2),
                out_upswiglu,
            ),
        ];

        for (step, (exp_fmt, exp_epi, exp_code, exp_res, exp_out)) in
            steps.iter().zip(expected.iter())
        {
            match step {
                RocmStep::QuantGemv {
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
                        .expect("expected HIP kernel source");
                    assert!(src.contains("extern \"C\" __global__ void gemv_"));
                    assert!(src.contains("apply_gemv_epi"));
                }
                other => panic!("expected QuantGemv step, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_rocm_plan_moe_quant_gemv() {
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

        let steps = plan_graph(&g).expect("ROCm MoE plan_graph should succeed");
        assert_eq!(steps.len(), 6);

        assert!(matches!(
            steps[0],
            RocmStep::Other {
                op: Op::Linear,
                out,
                ..
            } if out == logits
        ));
        assert!(matches!(
            steps[1],
            RocmStep::TopK {
                x: sx,
                out: sout,
                rows: 1,
                cols: 8,
                k: 2,
            } if sx == logits && sout == ids
        ));
        assert!(matches!(
            steps[2],
            RocmStep::GatherCols {
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
            RocmStep::Other {
                op: Op::Softmax { axis: -1 },
                out,
                ..
            } if out == probs
        ));
        match &steps[4] {
            RocmStep::QuantMoeGemv {
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
                    .expect("expected HIP MoE kernel source for IQ2_XXS");
                assert!(src.contains("extern \"C\" __global__ void gemvgen_GGUF_IQ2_XXS_r2_ix"));
            }
            other => panic!("expected QuantMoeGemv at step 4, got {other:?}"),
        }
        match &steps[5] {
            RocmStep::QuantMoeGemv {
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
                    .expect("expected HIP MoE kernel source for Q4_K_M");
                assert!(src.contains("extern \"C\" __global__ void gemvgen_GGUF_Q4_K_M_r2_ix"));
            }
            other => panic!("expected QuantMoeGemv at step 5, got {other:?}"),
        }
    }
}
