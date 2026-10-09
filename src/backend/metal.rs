//! Apple Metal GPU backend (macOS): system-device discovery, runtime MSL compilation via
//! `newLibraryWithSource`, unified-memory (`StorageModeShared`) buffers, synchronous kernel
//! dispatch, and [`MetalGraphLm`]: the GraphLm decode step on the GPU.

/// Convert IEEE 754 binary16 (f16 stored in u16) to f32.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp32 = (127 - 15 - e + 1) as u32;
            let mant32 = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
        }
    } else {
        let exp32 = (exp + 127 - 15) as u32;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
    }
}

/// Convert f32 to IEEE 754 binary16 (f16 stored in u16).
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as u32;

    if exp == 255 {
        if mant == 0 {
            (sign << 15) | 0x7C00
        } else {
            (sign << 15) | 0x7E00
        }
    } else if exp - 127 > 15 {
        (sign << 15) | 0x7C00
    } else if exp - 127 < -14 {
        0
    } else {
        let new_exp = ((exp - 127 + 15) as u16) << 10;
        let new_mant = (mant >> 13) as u16;
        (sign << 15) | new_exp | new_mant
    }
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub(crate) use gpu::{commit_and_wait, new_shared_buffer, set_bytes, system_device, SHARED, TG};
#[cfg(target_os = "macos")]
pub use gpu::{
    system_device_name, system_has_unified_memory, MetalBackend, MetalGraphLm, MetalKvCache,
};
#[cfg(target_os = "macos")]
mod gpu {
    use crate::model::graph_lm::{DecodeScratchpad, GraphLm};
    use crate::model::layers::{QuantizedLinearLayer, QuantizedWeight};
    use metal::{
        Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
        ComputePipelineState, Device, LibraryRef, MTLCommandBufferStatus, MTLResourceOptions,
        MTLSize,
    };
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::time::{Duration, Instant};

    const KERNELS_MSL: &str = include_str!("metal_kernels.metal");
    pub(crate) const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
    /// Threads per threadgroup for GEMV (8 simdgroups = 8 rows), attention, and element-wise kernels.
    pub(crate) const TG: u64 = 256;
    const GEMV_ROWS_PER_TG: usize = 8;
    const ATTN_MAX_HEAD_DIM: usize = 256;

    extern "C" {
        fn malloc_size(ptr: *const c_void) -> usize;
        fn getpagesize() -> i32;
    }

    pub(crate) fn system_device() -> Result<Device, String> {
        Device::system_default()
            .ok_or_else(|| "no Metal GPU (MTLCreateSystemDefaultDevice returned nil)".to_string())
    }

    /// GPU name as reported by the Metal API (e.g. "Apple M5 Pro").
    pub fn system_device_name() -> Result<String, String> {
        Ok(system_device()?.name().to_string())
    }

    /// Whether the system GPU shares memory with the CPU, as reported by the Metal API.
    pub fn system_has_unified_memory() -> Result<bool, String> {
        Ok(system_device()?.has_unified_memory())
    }

    fn compile(device: &Device, source: &str) -> Result<metal::Library, String> {
        let opts = CompileOptions::new();
        // Precise exp/tanh/sqrt keep GPU results within float rounding of the CPU kernels.
        opts.set_fast_math_enabled(false);
        device
            .new_library_with_source(source, &opts)
            .map_err(|e| format!("Metal shader compilation failed: {e}"))
    }

    fn pipeline(
        device: &Device,
        lib: &LibraryRef,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("Metal function '{name}': {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("Metal pipeline '{name}': {e}"))
    }

    /// Commits `cb`, blocks until the GPU finishes, and surfaces execution failures.
    pub(crate) fn commit_and_wait(cb: &CommandBufferRef) -> Result<(), String> {
        cb.commit();
        cb.wait_until_completed();
        match cb.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            s => Err(format!("Metal command buffer finished with status {s:?}")),
        }
    }

    pub(crate) fn new_shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, String> {
        let buf = device.new_buffer(bytes.max(4) as u64, SHARED);
        if buf.contents().is_null() {
            return Err(format!(
                "Metal could not allocate a {bytes}-byte shared buffer"
            ));
        }
        Ok(buf)
    }

    pub(crate) fn set_bytes<T>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
        enc.set_bytes(
            index,
            std::mem::size_of::<T>() as u64,
            value as *const T as *const c_void,
        );
    }

    pub(crate) fn dispatch_1d(enc: &ComputeCommandEncoderRef, n: usize) {
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(TG), 1, 1),
            MTLSize::new(TG, 1, 1),
        );
    }

    // ---------------------------------------------------------------------------------------
    // Standalone kernels
    // ---------------------------------------------------------------------------------------

    /// Runtime MSL compilation and synchronous dispatch over shared-storage buffers allocated
    /// by this backend.
    pub struct MetalBackend {
        device: Device,
        queue: CommandQueue,
        pipelines: RwLock<HashMap<String, ComputePipelineState>>,
        // ponytail: buffers live until the backend drops (the trait has no free); add release when callers need it.
        buffers: RwLock<HashMap<usize, Buffer>>,
    }

    impl MetalBackend {
        pub fn new() -> Result<Self, String> {
            let device = system_device()?;
            let queue = device.new_command_queue();
            Ok(Self {
                device,
                queue,
                pipelines: RwLock::new(HashMap::new()),
                buffers: RwLock::new(HashMap::new()),
            })
        }

        pub fn device_name(&self) -> &str {
            self.device.name()
        }

        pub fn is_unified_memory(&self) -> bool {
            self.device.has_unified_memory()
        }

        /// Allocates a zeroed shared-storage buffer and returns its CPU-visible pointer.
        pub fn allocate_buffer(&self, size_bytes: usize) -> Result<*mut c_void, String> {
            let max = self.device.max_buffer_length();
            if size_bytes as u64 > max {
                return Err(format!(
                    "Metal buffer of {size_bytes} bytes exceeds the device limit of {max} bytes"
                ));
            }
            let buf = new_shared_buffer(&self.device, size_bytes)?;
            let ptr = buf.contents();
            self.buffers.write().insert(ptr as usize, buf);
            Ok(ptr)
        }

        /// Compiles `source_code` with `newLibraryWithSource` and builds a pipeline for the
        /// function named `kernel_name`; compiler diagnostics come back as `Err`.
        pub fn jit_compile(&self, kernel_name: &str, source_code: &str) -> Result<(), String> {
            if !self.pipelines.read().contains_key(kernel_name) {
                let lib = compile(&self.device, source_code)?;
                let pso = pipeline(&self.device, &lib, kernel_name)?;
                self.pipelines.write().insert(kernel_name.to_string(), pso);
            }
            Ok(())
        }

        /// Encodes one dispatch; `args[i]` must be a pointer returned by this backend's
        /// `allocate_buffer` and binds to `[[buffer(i)]]`. Blocks until the GPU completes.
        pub fn dispatch(
            &self,
            kernel: &str,
            grid: (u32, u32, u32),
            block: (u32, u32, u32),
            args: &[*mut c_void],
        ) -> Result<(), String> {
            let pipelines = self.pipelines.read();
            let pso = pipelines.get(kernel).ok_or_else(|| {
                format!("kernel '{kernel}' was not compiled by this Metal backend")
            })?;
            let threads = block.0 as u64 * block.1 as u64 * block.2 as u64;
            if threads == 0 || threads > pso.max_total_threads_per_threadgroup() {
                return Err(format!(
                    "threadgroup {block:?} is outside 1..={} threads for '{kernel}'",
                    pso.max_total_threads_per_threadgroup(),
                ));
            }
            let buffers = self.buffers.read();
            let bound = args
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    buffers.get(&(*p as usize)).ok_or_else(|| {
                        format!(
                            "argument {i} of '{kernel}' is not a buffer from this Metal backend"
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            metal::objc::rc::autoreleasepool(|| {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pso);
                for (i, &b) in bound.iter().enumerate() {
                    enc.set_buffer(i as u64, Some(b), 0);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                    MTLSize::new(block.0 as u64, block.1 as u64, block.2 as u64),
                );
                enc.end_encoding();
                commit_and_wait(cb)
            })
        }
    }

    // ---------------------------------------------------------------------------------------
    // GraphLm decode
    // ---------------------------------------------------------------------------------------

    #[repr(C)]
    struct EmbedParams {
        n: u32,
        tok_off: u32,
        pos_off: u32,
        has_pos: u32,
    }
    #[repr(C)]
    struct NormParams {
        n: u32,
        eps: f32,
        layernorm: u32,
        has_bias: u32,
        add_residual: u32,
    }
    #[repr(C)]
    struct GemvParams {
        rows: u32,
        cols4: u32,
        has_bias: u32,
    }
    #[repr(C)]
    struct RopeParams {
        n_heads: u32,
        head_dim: u32,
        pairs: u32,
        pair_stride: u32,
        pair_gap: u32,
        k_offset: u32,
    }
    #[repr(C)]
    struct KvParams {
        n: u32,
        src_k: u32,
        src_v: u32,
        dst: u32,
    }
    #[repr(C)]
    pub struct AttnParams {
        pub n_heads: u32,
        pub head_dim: u32,
        pub stride: u32,
        pub len: u32,
        pub group: u32,
        pub scale: f32,
        pub window: u32,
        pub softcap: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub enum GpuWeightFormat {
        F32,
        BF16,
        Q4_0,
        Q8_0,
    }

    /// Row-major f32, bf16, or packed Q4_0/Q8_0 weight `[rows, cols]` (+ optional bias) in a shared buffer.
    pub struct GpuLinear {
        pub w: Buffer,
        pub bias: Option<Buffer>,
        pub rows: usize,
        pub cols: usize,
        pub format: GpuWeightFormat,
    }

    impl GpuLinear {
        #[allow(dead_code)]
        pub fn is_bf16(&self) -> bool {
            self.format == GpuWeightFormat::BF16
        }
    }

    struct GpuLayer {
        norm1: Buffer,
        norm1_b: Option<Buffer>,
        qkv: GpuLinear,
        out: GpuLinear,
        post_attn: Option<Buffer>,
        norm2: Buffer,
        norm2_b: Option<Buffer>,
        gate_up: GpuLinear,
        down: GpuLinear,
        post_ffw: Option<Buffer>,
    }

    /// Activations for one token step; steps are serialized behind a mutex.
    struct Scratch {
        hidden: Buffer,
        norm_x: Buffer,
        qkv: Buffer,
        attn: Buffer,
        proj: Buffer,
        gate_up: Buffer,
        mlp: Buffer,
        down: Buffer,
        logits: Buffer,
    }

    #[derive(Default)]
    pub struct Uploaded {
        pub aliased: usize,
        pub copied: usize,
    }

    /// Device, queue, and the compiled decode pipelines.
    #[allow(dead_code)]
    pub struct Gpu {
        pub device: Device,
        pub queue: CommandQueue,
        pub embed: ComputePipelineState,
        pub embed_direct: ComputePipelineState,
        pub norm: ComputePipelineState,
        pub rms_norm_heads: ComputePipelineState,
        pub gemv: ComputePipelineState,
        pub gemv_bf16: ComputePipelineState,
        pub rope: ComputePipelineState,
        pub rope_single: ComputePipelineState,
        pub kv_append: ComputePipelineState,
        pub kv_append_heads: ComputePipelineState,
        pub attn: ComputePipelineState,
        pub add: ComputePipelineState,
        pub swiglu: ComputePipelineState,
        pub geglu: ComputePipelineState,
        pub gelu: ComputePipelineState,
        pub scale: ComputePipelineState,
        pub softcap: ComputePipelineState,
        pub norm_threads: u64,
        /// Bound to unused optional slots (absent bias / residual / position table).
        pub gemv_q4_0: Option<ComputePipelineState>,
        pub gemv_q8_0: Option<ComputePipelineState>,
        pub dummy: Buffer,
    }

    #[allow(dead_code)]
    impl Gpu {
        pub fn new() -> Result<Self, String> {
            let device = system_device()?;
            let lib = compile(&device, KERNELS_MSL)?;
            let p = |name: &str| pipeline(&device, &lib, name);
            let queue = device.new_command_queue();
            let gemv_base = p("gemv_f32")?;
            let gemv_u4 = p("gemv_unroll4").ok();
            let gemv_q4_0 = p("gemv_q4_0").ok();
            let gemv_q8_0 = p("gemv_q8_0").ok();
            let gemv = if let Some(u4) = gemv_u4 {
                let rows = 1024usize;
                let cols = 1024usize;
                let w_buf = new_shared_buffer(&device, rows * cols * 4)?;
                let x_buf = new_shared_buffer(&device, cols * 4)?;
                let y_buf = new_shared_buffer(&device, rows * 4)?;
                let bias_buf = new_shared_buffer(&device, rows * 4)?;
                let params = GemvParams {
                    rows: rows as u32,
                    cols4: (cols / 4) as u32,
                    has_bias: 0,
                };
                let bench = |pso: &ComputePipelineState| -> f64 {
                    let mut best = f64::INFINITY;
                    for _ in 0..3 {
                        let t0 = Instant::now();
                        let cb = queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(pso);
                        enc.set_buffer(0, Some(&w_buf), 0);
                        enc.set_buffer(1, Some(&x_buf), 0);
                        enc.set_buffer(2, Some(&y_buf), 0);
                        enc.set_buffer(3, Some(&bias_buf), 0);
                        set_bytes(enc, 4, &params);
                        enc.dispatch_thread_groups(
                            MTLSize::new(rows.div_ceil(GEMV_ROWS_PER_TG) as u64, 1, 1),
                            MTLSize::new(TG, 1, 1),
                        );
                        enc.end_encoding();
                        cb.commit();
                        cb.wait_until_completed();
                        let d = t0.elapsed().as_secs_f64();
                        if d < best {
                            best = d;
                        }
                    }
                    best
                };
                let t_base = bench(&gemv_base);
                let t_u4 = bench(&u4);
                if t_u4 < t_base {
                    u4
                } else {
                    gemv_base
                }
            } else {
                gemv_base
            };
            let gpu = Self {
                embed: p("embed_f32")?,
                embed_direct: p("embed_direct")?,
                norm: p("norm_f32")?,
                rms_norm_heads: p("rms_norm_heads")?,
                gemv,
                gemv_bf16: p("gemv_bf16")?,
                rope: p("rope_f32")?,
                rope_single: p("rope_single")?,
                kv_append: p("kv_append_f32")?,
                kv_append_heads: p("kv_append_heads")?,
                attn: p("attn_decode_f32")?,
                add: p("add_f32")?,
                swiglu: p("swiglu_f32")?,
                geglu: p("geglu_f32")?,
                gelu: p("gelu_f32")?,
                scale: p("scale_f32")?,
                softcap: p("softcap_f32")?,
                norm_threads: 0,
                gemv_q4_0,
                gemv_q8_0,
                dummy: new_shared_buffer(&device, 16)?,
                device,
                queue,
            };
            for (name, pso) in [
                ("embed_f32", &gpu.embed),
                ("embed_direct", &gpu.embed_direct),
                ("gemv_f32", &gpu.gemv),
                ("gemv_bf16", &gpu.gemv_bf16),
                ("rope_f32", &gpu.rope),
                ("rope_single", &gpu.rope_single),
                ("kv_append_f32", &gpu.kv_append),
                ("kv_append_heads", &gpu.kv_append_heads),
                ("attn_decode_f32", &gpu.attn),
                ("add_f32", &gpu.add),
                ("swiglu_f32", &gpu.swiglu),
                ("geglu_f32", &gpu.geglu),
                ("gelu_f32", &gpu.gelu),
                ("scale_f32", &gpu.scale),
                ("softcap_f32", &gpu.softcap),
                ("norm_f32", &gpu.norm),
            ] {
                if pso.thread_execution_width() != 32
                    || pso.max_total_threads_per_threadgroup() < TG
                {
                    return Err(format!(
                        "Metal pipeline '{name}' needs SIMD width 32 and >= {TG} threads per threadgroup (got {} / {})",
                        pso.thread_execution_width(),
                        pso.max_total_threads_per_threadgroup()
                    ));
                }
            }
            let norm_threads = gpu.norm.max_total_threads_per_threadgroup().min(1024) / 32 * 32;
            Ok(Self {
                norm_threads,
                ..gpu
            })
        }

        /// Shared-storage buffer holding `data`. Page-aligned malloc'd regions (large Vecs) are
        /// wrapped in place with `newBufferWithBytesNoCopy`; everything else is copied once.
        pub fn upload(&self, data: &[f32], up: &mut Uploaded) -> Result<Buffer, String> {
            let bytes = std::mem::size_of_val(data);
            if bytes == 0 {
                return new_shared_buffer(&self.device, 4);
            }
            let ptr = data.as_ptr() as *const c_void;
            let page = unsafe { getpagesize() } as usize;
            let len = bytes.div_ceil(page) * page;
            // SAFETY: malloc_size returns 0 for pointers malloc does not own.
            if ptr as usize % page == 0 && unsafe { malloc_size(ptr) } >= len {
                // ponytail: aliases the model's Vec; MetalGraphLm owns that GraphLm immutably and drops it after every buffer.
                let buf = self
                    .device
                    .new_buffer_with_bytes_no_copy(ptr, len as u64, SHARED, None);
                if !buf.contents().is_null() {
                    up.aliased += bytes;
                    return Ok(buf);
                }
            }
            let buf = self.device.new_buffer_with_data(ptr, bytes as u64, SHARED);
            if buf.contents().is_null() {
                return Err(format!(
                    "Metal could not allocate {bytes} bytes for weights"
                ));
            }
            up.copied += bytes;
            Ok(buf)
        }

        fn vector(
            &self,
            name: &str,
            data: &[f32],
            n: usize,
            up: &mut Uploaded,
        ) -> Result<Buffer, String> {
            if data.len() != n {
                return Err(format!("{name} has {} values, expected {n}", data.len()));
            }
            self.upload(data, up)
        }

        fn linear(
            &self,
            layer: &QuantizedLinearLayer,
            cols: usize,
            rows: usize,
            up: &mut Uploaded,
        ) -> Result<GpuLinear, String> {
            let QuantizedWeight::Uncompressed {
                data,
                in_features,
                out_features,
            } = &layer.weight
            else {
                return Err(format!(
                    "'{}' is stored as {:?}; the Metal backend runs f32 weights only (use --quant uncompressed)",
                    layer.name,
                    layer.quant_format()
                ));
            };
            if (*in_features, *out_features) != (cols, rows) || data.len() != rows * cols {
                return Err(format!(
                    "'{}' is [{out_features} x {in_features}] with {} values, expected [{rows} x {cols}]",
                    layer.name,
                    data.len()
                ));
            }
            if cols % 4 != 0 {
                return Err(format!(
                    "'{}' has {cols} input features; Metal GEMV needs a multiple of 4",
                    layer.name
                ));
            }
            let bias = match &layer.bias {
                Some(b) => Some(self.vector(&format!("'{}' bias", layer.name), b, rows, up)?),
                None => None,
            };
            Ok(GpuLinear {
                w: self.upload(data, up)?,
                bias,
                rows,
                cols,
                format: GpuWeightFormat::F32,
            })
        }

        pub fn linear_from_mat(
            &self,
            m: &crate::model::weights::Mat,
            up: &mut Uploaded,
        ) -> Result<GpuLinear, String> {
            let (rows, cols) = (m.rows, m.cols);
            if cols % 4 != 0 {
                return Err(format!("linear_from_mat: cols {cols} not a multiple of 4"));
            }
            if let Some(p) = m.packed() {
                if p.format == crate::hardware::QuantFormat::GGUF_Q4_0 && cols % 32 == 0 {
                    let len = p.format.tensor_bytes(rows * cols);
                    let slice = &p.map[p.off..p.off + len];
                    let buf = self.device.new_buffer_with_data(
                        slice.as_ptr() as *const c_void,
                        len as u64,
                        SHARED,
                    );
                    up.copied += len;
                    return Ok(GpuLinear {
                        w: buf,
                        bias: None,
                        rows,
                        cols,
                        format: GpuWeightFormat::Q4_0,
                    });
                }
                if p.format == crate::hardware::QuantFormat::GGUF_Q8_0 && cols % 32 == 0 {
                    let len = p.format.tensor_bytes(rows * cols);
                    let slice = &p.map[p.off..p.off + len];
                    let buf = self.device.new_buffer_with_data(
                        slice.as_ptr() as *const c_void,
                        len as u64,
                        SHARED,
                    );
                    up.copied += len;
                    return Ok(GpuLinear {
                        w: buf,
                        bias: None,
                        rows,
                        cols,
                        format: GpuWeightFormat::Q8_0,
                    });
                }
                let f = m.to_f32();
                let bytes = f.len() * 4;
                let buf = self.device.new_buffer_with_data(
                    f.as_ptr() as *const c_void,
                    bytes as u64,
                    SHARED,
                );
                up.copied += bytes;
                return Ok(GpuLinear {
                    w: buf,
                    bias: None,
                    rows,
                    cols,
                    format: GpuWeightFormat::F32,
                });
            }
            let data = m.data();
            let bytes = data.len() * 2;
            let ptr = data.as_ptr() as *const c_void;
            let buf = self.device.new_buffer_with_data(ptr, bytes as u64, SHARED);
            up.copied += bytes;
            Ok(GpuLinear {
                w: buf,
                bias: None,
                rows,
                cols,
                format: GpuWeightFormat::BF16,
            })
        }

        /// Encodes into one command buffer, commits it, and waits for the GPU.
        pub fn run(&self, encode: impl FnOnce(&ComputeCommandEncoderRef)) -> Result<(), String> {
            metal::objc::rc::autoreleasepool(|| {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                encode(enc);
                enc.end_encoding();
                commit_and_wait(cb)
            })
        }

        pub fn gemv(
            &self,
            enc: &ComputeCommandEncoderRef,
            lin: &GpuLinear,
            x: &Buffer,
            y: &Buffer,
        ) {
            #[repr(C)]
            struct PackedGemvParams {
                rows: u32,
                cols: u32,
                batch_size: u32,
                has_bias: u32,
                epi: u32,
            }

            match lin.format {
                GpuWeightFormat::Q4_0 => {
                    if let Some(pso) = &self.gemv_q4_0 {
                        enc.set_compute_pipeline_state(pso);
                        enc.set_buffer(0, Some(&lin.w), 0);
                        enc.set_buffer(1, Some(x), 0);
                        enc.set_buffer(2, Some(y), 0);
                        set_bytes(
                            enc,
                            3,
                            &PackedGemvParams {
                                rows: lin.rows as u32,
                                cols: lin.cols as u32,
                                batch_size: 1,
                                has_bias: lin.bias.is_some() as u32,
                                epi: 0,
                            },
                        );
                        enc.set_buffer(4, Some(lin.bias.as_ref().unwrap_or(&self.dummy)), 0);
                        enc.set_buffer(5, Some(&self.dummy), 0);
                        let num_tg = lin.rows.div_ceil(16) as u64;
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tg, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        return;
                    }
                }
                GpuWeightFormat::Q8_0 => {
                    if let Some(pso) = &self.gemv_q8_0 {
                        enc.set_compute_pipeline_state(pso);
                        enc.set_buffer(0, Some(&lin.w), 0);
                        enc.set_buffer(1, Some(x), 0);
                        enc.set_buffer(2, Some(y), 0);
                        set_bytes(
                            enc,
                            3,
                            &PackedGemvParams {
                                rows: lin.rows as u32,
                                cols: lin.cols as u32,
                                batch_size: 1,
                                has_bias: lin.bias.is_some() as u32,
                                epi: 0,
                            },
                        );
                        enc.set_buffer(4, Some(lin.bias.as_ref().unwrap_or(&self.dummy)), 0);
                        enc.set_buffer(5, Some(&self.dummy), 0);
                        let num_tg = lin.rows.div_ceil(8) as u64;
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tg, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        return;
                    }
                }
                GpuWeightFormat::BF16 => {
                    enc.set_compute_pipeline_state(&self.gemv_bf16);
                }
                GpuWeightFormat::F32 => {
                    enc.set_compute_pipeline_state(&self.gemv);
                }
            }
            enc.set_buffer(0, Some(&lin.w), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_buffer(3, Some(lin.bias.as_ref().unwrap_or(&self.dummy)), 0);
            set_bytes(
                enc,
                4,
                &GemvParams {
                    rows: lin.rows as u32,
                    cols4: (lin.cols / 4) as u32,
                    has_bias: lin.bias.is_some() as u32,
                },
            );
            enc.dispatch_thread_groups(
                MTLSize::new(lin.rows.div_ceil(GEMV_ROWS_PER_TG) as u64, 1, 1),
                MTLSize::new(TG, 1, 1),
            );
        }

        /// `x += res` when `res` is given, then RMSNorm (or LayerNorm) of `x` into `out`.
        #[allow(clippy::too_many_arguments)]
        pub fn norm(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            res: Option<&Buffer>,
            w: &Buffer,
            bias: Option<&Buffer>,
            out: &Buffer,
            layernorm: bool,
            n: usize,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.norm);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(res.unwrap_or(&self.dummy)), 0);
            enc.set_buffer(2, Some(w), 0);
            enc.set_buffer(3, Some(bias.unwrap_or(&self.dummy)), 0);
            enc.set_buffer(4, Some(out), 0);
            set_bytes(
                enc,
                5,
                &NormParams {
                    n: n as u32,
                    eps,
                    layernorm: layernorm as u32,
                    has_bias: bias.is_some() as u32,
                    add_residual: res.is_some() as u32,
                },
            );
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(self.norm_threads, 1, 1),
            );
        }

        pub fn add(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, y: &Buffer, n: usize) {
            enc.set_compute_pipeline_state(&self.add);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(y), 0);
            set_bytes(enc, 2, &(n as u32));
            dispatch_1d(enc, n);
        }
        pub fn device_name(&self) -> &str {
            self.device.name()
        }
        pub fn new_kv_cache(&self) -> Result<MetalKvCache, String> {
            Ok(MetalKvCache {
                k: new_shared_buffer(&self.device, 4)?,
                v: new_shared_buffer(&self.device, 4)?,
                cap: 0,
                len: 0,
            })
        }
        pub fn reserve_kv(
            &self,
            kv: &mut MetalKvCache,
            need: usize,
            layers: usize,
            width: usize,
            max_seq_len: usize,
        ) -> Result<(), String> {
            if need <= kv.cap {
                return Ok(());
            }
            let cap = need.next_power_of_two().max(64).min(max_seq_len);
            let k = new_shared_buffer(&self.device, layers * cap * width * 4)?;
            let v = new_shared_buffer(&self.device, layers * cap * width * 4)?;
            if kv.len > 0 {
                for (old, new) in [(&kv.k, &k), (&kv.v, &v)] {
                    let (src, dst) = (old.contents() as *const f32, new.contents() as *mut f32);
                    for l in 0..layers {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src.add(l * kv.cap * width),
                                dst.add(l * cap * width),
                                kv.len * width,
                            )
                        };
                    }
                }
            }
            *kv = MetalKvCache {
                k,
                v,
                cap,
                len: kv.len,
            };
            Ok(())
        }
    }

    /// GPU-resident KV cache of one sequence: `[layer][position][hidden]` f32 for K and V,
    /// grown on demand up to the model's context window.
    pub struct MetalKvCache {
        pub k: Buffer,
        pub v: Buffer,
        pub cap: usize,
        pub len: usize,
    }

    impl MetalKvCache {
        /// Positions already written (the next token's position).
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }
    }

    /// [`GraphLm`] decode on the Apple GPU. Every token step (embedding, norms, QKV/out/MLP/LM-head
    /// GEMVs, RoPE, KV append, attention, residuals) is encoded into one command buffer; only the
    /// logits are read back. Weights are fixed f32 (no runtime re-quantization).
    pub struct MetalGraphLm {
        gpu: Gpu,
        device_name: String,
        embedding: Buffer,
        pos_embedding: Option<Buffer>,
        layers: Vec<GpuLayer>,
        final_norm: Buffer,
        final_norm_b: Option<Buffer>,
        lm_head: GpuLinear,
        scratch: Mutex<Scratch>,
        rope_inv_freq: Vec<f32>,
        rope_half_dim: usize,
        /// Weight bytes wrapped in place (zero-copy) vs copied into Metal buffers.
        pub aliased_weight_bytes: usize,
        pub copied_weight_bytes: usize,
        /// Declared last: no-copy buffers above alias this model's weight Vecs.
        model: GraphLm,
    }

    impl MetalGraphLm {
        pub fn new(model: GraphLm) -> Result<Self, String> {
            let c = model.config.clone();
            let (h, v, inter, nh, hd) = (
                c.hidden_dim,
                c.vocab_size,
                c.intermediate_dim,
                c.num_heads,
                c.head_dim,
            );
            if h == 0 || v == 0 {
                return Err(format!("model has hidden_dim {h} and vocab_size {v}"));
            }
            // ponytail: no Metal kernels for BitLinear activation quant, relu² or sub-norms; port
            // them beside the f32 GEMV kernels if BitNet should decode on the GPU.
            let sub_norms = model
                .layers
                .iter()
                .any(|l| l.attn_sub_norm.is_some() || l.ffn_sub_norm.is_some());
            if c.act_quant_bits > 0 || c.use_relu2_mlp || sub_norms {
                return Err(
                    "BitNet b1.58 (BitLinear activation quant, relu², sub-norms) has no Metal kernels; use --backend cpu"
                        .into(),
                );
            }
            if nh == 0 || nh * hd != h {
                return Err(format!(
                    "Metal decode needs num_heads * head_dim == hidden_dim (got {nh} * {hd} vs {h})"
                ));
            }
            if hd > ATTN_MAX_HEAD_DIM {
                return Err(format!(
                    "head_dim {hd} exceeds the Metal attention limit of {ATTN_MAX_HEAD_DIM}"
                ));
            }
            if c.use_hadamard_attention && hd.is_power_of_two() {
                return Err("Hadamard attention has no Metal kernel; use --backend cpu".into());
            }
            if c.use_hadamard_mlp && inter.is_power_of_two() && inter <= h {
                return Err("Hadamard MLP has no Metal kernel; use --backend cpu".into());
            }
            if model.embedding.len() != v * h {
                return Err(format!(
                    "token embedding has {} values, expected vocab {v} x hidden {h}",
                    model.embedding.len()
                ));
            }
            let max_index = model
                .embedding
                .len()
                .max(model.pos_embedding.as_ref().map_or(0, Vec::len));
            if max_index > u32::MAX as usize || c.max_seq_len.saturating_mul(h) > u32::MAX as usize
            {
                return Err("embedding or KV rows exceed 32-bit GPU indexing".into());
            }

            let gpu = Gpu::new()?;
            let mut up = Uploaded::default();
            let embedding = gpu.upload(&model.embedding, &mut up)?;
            let pos_embedding = match &model.pos_embedding {
                Some(pe) if pe.len() % h != 0 => {
                    return Err(format!(
                        "position embedding has {} values, not a multiple of hidden {h}",
                        pe.len()
                    ))
                }
                Some(pe) => Some(gpu.upload(pe, &mut up)?),
                None => None,
            };
            let opt = |name: String,
                       data: &Option<Vec<f32>>,
                       up: &mut Uploaded|
             -> Result<Option<Buffer>, String> {
                data.as_ref()
                    .map(|d| gpu.vector(&name, d, h, up))
                    .transpose()
            };
            let gate_up_rows = if c.use_gelu_mlp { inter } else { 2 * inter };
            let mut layers = Vec::with_capacity(model.layers.len());
            for (l, t) in model.layers.iter().enumerate() {
                layers.push(GpuLayer {
                    norm1: gpu.vector(&format!("layer {l} norm1"), &t.norm1, h, &mut up)?,
                    norm1_b: opt(format!("layer {l} norm1 bias"), &t.norm1_bias, &mut up)?,
                    qkv: gpu.linear(&t.qkv_proj, h, 3 * h, &mut up)?,
                    out: gpu.linear(&t.out_proj, h, h, &mut up)?,
                    post_attn: opt(
                        format!("layer {l} post-attention norm"),
                        &t.post_attention_norm,
                        &mut up,
                    )?,
                    norm2: gpu.vector(&format!("layer {l} norm2"), &t.norm2, h, &mut up)?,
                    norm2_b: opt(format!("layer {l} norm2 bias"), &t.norm2_bias, &mut up)?,
                    gate_up: gpu.linear(&t.gate_up_proj, h, gate_up_rows, &mut up)?,
                    down: gpu.linear(&t.down_proj, inter, h, &mut up)?,
                    post_ffw: opt(
                        format!("layer {l} post-FFN norm"),
                        &t.post_ffw_norm,
                        &mut up,
                    )?,
                });
            }
            let final_norm = gpu.vector("final norm", &model.final_norm, h, &mut up)?;
            let final_norm_b = opt("final norm bias".into(), &model.final_norm_bias, &mut up)?;
            let lm_head = gpu.linear(&model.lm_head, h, v, &mut up)?;

            let f = |n: usize| new_shared_buffer(&gpu.device, n * 4);
            let scratch = Scratch {
                hidden: f(h)?,
                norm_x: f(h)?,
                qkv: f(3 * h)?,
                attn: f(h)?,
                proj: f(h)?,
                gate_up: f(gate_up_rows)?,
                mlp: f(inter)?,
                down: f(h)?,
                logits: f(v)?,
            };
            // Same frequencies (and the same 128-pair clamp) as the CPU decode step.
            let rotary_dim = ((hd as f32) * c.rotary_pct) as usize;
            let rope_half_dim = (rotary_dim / 2).min(hd / 2);
            let rope_inv_freq = DecodeScratchpad::from_config(&c).rope_inv_freq;

            Ok(Self {
                device_name: gpu.device.name().to_string(),
                gpu,
                embedding,
                pos_embedding,
                layers,
                final_norm,
                final_norm_b,
                lm_head,
                scratch: Mutex::new(scratch),
                rope_inv_freq,
                rope_half_dim,
                aliased_weight_bytes: up.aliased,
                copied_weight_bytes: up.copied,
                model,
            })
        }

        pub fn model(&self) -> &GraphLm {
            &self.model
        }

        /// GPU name from the Metal API.
        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        pub fn new_kv_cache(&self) -> Result<MetalKvCache, String> {
            Ok(MetalKvCache {
                k: new_shared_buffer(&self.gpu.device, 4)?,
                v: new_shared_buffer(&self.gpu.device, 4)?,
                cap: 0,
                len: 0,
            })
        }

        /// Runs `tokens` at positions `kv.len()..` in ONE command buffer (prefill loops the
        /// step kernels per token; the LM head runs only for the last) and returns its logits.
        pub fn forward_tokens(
            &self,
            tokens: &[u32],
            kv: &mut MetalKvCache,
        ) -> Result<Vec<f32>, String> {
            if tokens.is_empty() {
                return Err("forward_tokens needs at least one token".into());
            }
            let need = kv.len + tokens.len();
            let max = self.model.config.max_seq_len;
            if need > max {
                return Err(format!(
                    "{need} positions exceed the model context window of {max}"
                ));
            }
            self.reserve(kv, need)?;
            let s = self.scratch.lock();
            let kv_ref = &*kv;
            self.gpu.run(|enc| {
                for (i, &t) in tokens.iter().enumerate() {
                    self.encode_token(enc, &s, kv_ref, t, kv_ref.len + i, i + 1 == tokens.len());
                }
            })?;
            kv.len = need;
            let v = self.model.config.vocab_size;
            // SAFETY: shared buffer of `v` f32 logits, written by the completed command buffer.
            Ok(
                unsafe { std::slice::from_raw_parts(s.logits.contents() as *const f32, v) }
                    .to_vec(),
            )
        }

        /// Generates with the shared [`GraphLm::generate_with`] loop on the GPU.
        /// Returns (text, decode forward passes, decode time).
        pub fn generate_text_measured(
            &self,
            prompt: &str,
            max_new_tokens: usize,
            temperature: f32,
        ) -> Result<(String, usize, Duration), String> {
            let mut kv = self.new_kv_cache()?;
            self.model
                .generate_with(prompt, max_new_tokens, temperature, |t| {
                    self.forward_tokens(t, &mut kv)
                })
        }

        /// Grows `kv` to hold `need` positions, preserving the rows already written.
        fn reserve(&self, kv: &mut MetalKvCache, need: usize) -> Result<(), String> {
            let h = self.model.config.hidden_dim;
            let layers = self.layers.len();
            let max = self.model.config.max_seq_len;
            self.gpu.reserve_kv(kv, need, layers, h, max)
        }

        /// Encodes one decode step for `token` at `pos` (mirrors `forward_token_with_kv_scratchpad`).
        fn encode_token(
            &self,
            enc: &ComputeCommandEncoderRef,
            s: &Scratch,
            kv: &MetalKvCache,
            token: u32,
            pos: usize,
            want_logits: bool,
        ) {
            let g = &self.gpu;
            let c = &self.model.config;
            let (h, nh, hd, eps) = (c.hidden_dim, c.num_heads, c.head_dim, c.norm_eps);
            let ln = c.use_standard_layernorm;
            // LayerNorm biases only; RMSNorm has none.
            fn ln_bias(ln: bool, b: &Option<Buffer>) -> Option<&Buffer> {
                b.as_ref().filter(|_| ln)
            }

            let tok = (token as usize).min(c.vocab_size - 1);
            let pos_idx = pos.min(c.max_seq_len.saturating_sub(1));
            let has_pos =
                matches!(&self.model.pos_embedding, Some(pe) if (pos_idx + 1) * h <= pe.len());
            enc.set_compute_pipeline_state(&g.embed);
            enc.set_buffer(0, Some(&self.embedding), 0);
            enc.set_buffer(1, Some(self.pos_embedding.as_ref().unwrap_or(&g.dummy)), 0);
            enc.set_buffer(2, Some(&s.hidden), 0);
            set_bytes(
                enc,
                3,
                &EmbedParams {
                    n: h as u32,
                    tok_off: (tok * h) as u32,
                    pos_off: (pos_idx * h) as u32,
                    has_pos: has_pos as u32,
                },
            );
            dispatch_1d(enc, h);

            let pairs = self.rope_inv_freq.len();
            let mut cs = [0.0f32; 256];
            for d in 0..pairs {
                let theta = pos as f32 * self.rope_inv_freq[d];
                cs[d] = theta.cos();
                cs[pairs + d] = theta.sin();
            }
            let (pair_stride, pair_gap) = if c.rope_interleaved {
                (2, 1)
            } else {
                (1, self.rope_half_dim)
            };
            let kv_layer_bytes = (kv.cap * h * 4) as u64;

            for (l, layer) in self.layers.iter().enumerate() {
                g.norm(
                    enc,
                    &s.hidden,
                    None,
                    &layer.norm1,
                    ln_bias(ln, &layer.norm1_b),
                    &s.norm_x,
                    ln,
                    h,
                    eps,
                );
                g.gemv(enc, &layer.qkv, &s.norm_x, &s.qkv);

                if pairs > 0 {
                    enc.set_compute_pipeline_state(&g.rope);
                    enc.set_buffer(0, Some(&s.qkv), 0);
                    enc.set_bytes(1, (2 * pairs * 4) as u64, cs.as_ptr() as *const c_void);
                    set_bytes(
                        enc,
                        2,
                        &RopeParams {
                            n_heads: nh as u32,
                            head_dim: hd as u32,
                            pairs: pairs as u32,
                            pair_stride,
                            pair_gap: pair_gap as u32,
                            k_offset: h as u32,
                        },
                    );
                    dispatch_1d(enc, 2 * nh * pairs);
                }

                let off = l as u64 * kv_layer_bytes;
                enc.set_compute_pipeline_state(&g.kv_append);
                enc.set_buffer(0, Some(&s.qkv), 0);
                enc.set_buffer(1, Some(&kv.k), off);
                enc.set_buffer(2, Some(&kv.v), off);
                set_bytes(
                    enc,
                    3,
                    &KvParams {
                        n: h as u32,
                        src_k: h as u32,
                        src_v: (2 * h) as u32,
                        dst: (pos * h) as u32,
                    },
                );
                dispatch_1d(enc, h);

                // The CPU model stores K/V expanded to every query head, so the GQA group is 1.
                enc.set_compute_pipeline_state(&g.attn);
                enc.set_buffer(0, Some(&s.qkv), 0);
                enc.set_buffer(1, Some(&kv.k), off);
                enc.set_buffer(2, Some(&kv.v), off);
                enc.set_buffer(3, Some(&s.attn), 0);
                set_bytes(
                    enc,
                    4,
                    &AttnParams {
                        n_heads: nh as u32,
                        head_dim: hd as u32,
                        stride: h as u32,
                        len: (pos + 1) as u32,
                        group: 1,
                        scale: 1.0 / (hd as f32).sqrt(),
                        window: 0,
                        softcap: 0.0,
                    },
                );
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(TG, 1, 1));

                g.gemv(enc, &layer.out, &s.attn, &s.proj);
                if let Some(w) = &layer.post_attn {
                    g.norm(enc, &s.proj, None, w, None, &s.proj, false, h, eps);
                }
                let residual = (!c.use_parallel_residual).then_some(&s.proj);
                g.norm(
                    enc,
                    &s.hidden,
                    residual,
                    &layer.norm2,
                    ln_bias(ln, &layer.norm2_b),
                    &s.norm_x,
                    ln,
                    h,
                    eps,
                );

                let inter = layer.down.cols;
                if c.use_gelu_mlp {
                    g.gemv(enc, &layer.gate_up, &s.norm_x, &s.mlp);
                    enc.set_compute_pipeline_state(&g.gelu);
                    enc.set_buffer(0, Some(&s.mlp), 0);
                    set_bytes(enc, 1, &(inter as u32));
                } else {
                    g.gemv(enc, &layer.gate_up, &s.norm_x, &s.gate_up);
                    enc.set_compute_pipeline_state(&g.swiglu);
                    enc.set_buffer(0, Some(&s.gate_up), 0);
                    enc.set_buffer(1, Some(&s.mlp), 0);
                    set_bytes(enc, 2, &(inter as u32));
                }
                dispatch_1d(enc, inter);
                g.gemv(enc, &layer.down, &s.mlp, &s.down);
                if let Some(w) = &layer.post_ffw {
                    g.norm(enc, &s.down, None, w, None, &s.down, false, h, eps);
                }
                if c.use_parallel_residual {
                    g.add(enc, &s.hidden, &s.proj, h);
                }
                g.add(enc, &s.hidden, &s.down, h);
            }

            if want_logits {
                g.norm(
                    enc,
                    &s.hidden,
                    None,
                    &self.final_norm,
                    ln_bias(ln, &self.final_norm_b),
                    &s.norm_x,
                    ln,
                    h,
                    eps,
                );
                g.gemv(enc, &self.lm_head, &s.norm_x, &s.logits);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::hardware::QuantFormat;
        use crate::model::graph_lm::{GraphLmConfig, LayerKVCache};

        fn assert_close(gpu: &[f32], cpu: &[f32], what: &str) {
            assert_eq!(gpu.len(), cpu.len(), "{what}: length");
            let scale = cpu.iter().fold(1.0f32, |m, x| m.max(x.abs()));
            let worst = gpu
                .iter()
                .zip(cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1e-3 * scale,
                "{what}: max |gpu - cpu| = {worst} (scale {scale})"
            );
        }

        fn weights(n: usize, seed: usize) -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin() * 0.1)
                .collect()
        }

        #[test]
        fn gemv_matches_cpu() {
            let gpu = Gpu::new().expect("Metal device");
            let (rows, cols) = (37, 200);
            let layer = QuantizedLinearLayer::from_f32(
                "t",
                cols,
                rows,
                &weights(rows * cols, 1),
                QuantFormat::FP32,
            )
            .unwrap()
            .with_bias(weights(rows, 2))
            .unwrap();
            let x = weights(cols, 3);
            let mut cpu = vec![0.0f32; rows];
            layer.forward(&x, &mut cpu);

            let mut up = Uploaded::default();
            let lin = gpu.linear(&layer, cols, rows, &mut up).unwrap();
            let xb = gpu.upload(&x, &mut up).unwrap();
            let yb = new_shared_buffer(&gpu.device, rows * 4).unwrap();
            gpu.run(|enc| gpu.gemv(enc, &lin, &xb, &yb)).unwrap();
            let out = unsafe { std::slice::from_raw_parts(yb.contents() as *const f32, rows) };
            assert_close(out, &cpu, "gemv");
        }

        fn tiny_config() -> GraphLmConfig {
            GraphLmConfig {
                vocab_size: 300,
                hidden_dim: 64,
                intermediate_dim: 96,
                num_layers: 2,
                num_heads: 4,
                head_dim: 16,
                max_seq_len: 128,
                ..GraphLmConfig::smollm_1m()
            }
        }

        /// Decodes `tokens` one at a time on CPU and GPU, then re-runs them as one GPU prefill.
        fn check_decode(model: GraphLm, what: &str) {
            let tokens = [1u32, 7, 42, 299, 3, 150];
            let mut cpu_kv: Vec<LayerKVCache> = (0..model.layers.len())
                .map(|_| LayerKVCache::new(16, model.config.hidden_dim))
                .collect();
            let cpu: Vec<Vec<f32>> = tokens
                .iter()
                .enumerate()
                .map(|(p, &t)| model.forward_token_with_kv(t, p, &mut cpu_kv))
                .collect();

            let gpu = MetalGraphLm::new(model).expect("Metal model");
            let mut kv = gpu.new_kv_cache().unwrap();
            for (p, &t) in tokens.iter().enumerate() {
                let logits = gpu.forward_tokens(&[t], &mut kv).unwrap();
                assert_close(&logits, &cpu[p], &format!("{what} step {p}"));
            }
            let mut kv = gpu.new_kv_cache().unwrap();
            let logits = gpu.forward_tokens(&tokens, &mut kv).unwrap();
            assert_eq!(kv.len(), tokens.len());
            assert_close(&logits, cpu.last().unwrap(), &format!("{what} prefill"));
        }

        #[test]
        fn rope_variants_match_cpu() {
            for (interleaved, rotary_pct) in [(false, 1.0), (true, 1.0), (false, 0.5), (true, 0.5)]
            {
                let cfg = GraphLmConfig {
                    rope_interleaved: interleaved,
                    rotary_pct,
                    ..tiny_config()
                };
                check_decode(
                    GraphLm::new(cfg),
                    &format!("rope interleaved={interleaved} rotary_pct={rotary_pct}"),
                );
            }
        }

        #[test]
        fn decode_step_matches_cpu() {
            // RMSNorm + SwiGLU + sequential residual.
            check_decode(GraphLm::new(tiny_config()), "rmsnorm/swiglu");

            // LayerNorm (+bias) + GELU + parallel residual + post norms + QKV bias + position table.
            let cfg = GraphLmConfig {
                use_standard_layernorm: true,
                use_gelu_mlp: true,
                use_parallel_residual: true,
                rope_interleaved: true,
                rotary_pct: 0.5,
                ..tiny_config()
            };
            let (h, inter) = (cfg.hidden_dim, cfg.intermediate_dim);
            let mut model = GraphLm::new(cfg.clone());
            for (l, layer) in model.layers.iter_mut().enumerate() {
                layer.gate_up_proj = QuantizedLinearLayer::from_f32(
                    "gate_up",
                    h,
                    inter,
                    &weights(h * inter, l),
                    QuantFormat::FP32,
                )
                .unwrap();
                layer.norm1_bias = Some(weights(h, 10 + l));
                layer.norm2_bias = Some(weights(h, 20 + l));
                layer.post_attention_norm =
                    Some(weights(h, 30 + l).iter().map(|w| 1.0 + w).collect());
                layer.post_ffw_norm = Some(weights(h, 40 + l).iter().map(|w| 1.0 + w).collect());
                layer.qkv_proj = layer
                    .qkv_proj
                    .clone()
                    .with_bias(weights(3 * h, 50 + l))
                    .unwrap();
            }
            model.final_norm_bias = Some(weights(h, 60));
            model.pos_embedding = Some(weights(cfg.max_seq_len * h, 70));
            check_decode(model, "layernorm/gelu/parallel");
        }
    }
}
