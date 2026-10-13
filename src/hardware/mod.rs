use std::fmt;
use std::path::PathBuf;
pub mod cpu_compute;
pub mod profile;
pub use profile::HardwareProfile;

/// `$HOME/.cache/gibbs`: compiled kernels, host profiles, and per-model weight-format choices.
pub fn default_cache_dir() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set; cannot place the engine cache")?;
    Ok(PathBuf::from(home).join(".cache").join("gibbs"))
}

/// Exclusive advisory lock (`<cache>/gpu.lock`, flock) serializing benchmarks on the GPU.
/// Blocks (after saying so) while another benchmark holds it; released when the `File` drops.
pub fn gpu_lock() -> Result<std::fs::File, String> {
    let dir = default_cache_dir()?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("gpu.lock");
    let f = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    match f.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            if !crate::telemetry::TUI_OWNS_TERMINAL.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("gpu lock: waiting for another benchmark");
            }
            f.lock().map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(format!("{}: {e}", path.display())),
    }
    Ok(f)
}

/// Compute device for a pipeline or graph node, selected with `--backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Device {
    Cpu,
    /// Apple GPU through Metal.
    Metal,
    /// Apple Neural Engine through Core ML.
    Ane,
    Cuda,
    Rocm,
    LevelZero,
}

impl Device {
    /// The `--backend` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Ane => "ane",
            Device::Cuda => "cuda",
            Device::Rocm => "rocm",
            Device::LevelZero => "level-zero",
        }
    }

    /// The startup error for backends that exist only as stubs (CUDA, ROCm, Level Zero); `None` otherwise.
    pub fn stub_error(self) -> Option<String> {
        let api = match self {
            Device::Cuda => "CUDA",
            Device::Rocm => "ROCm",
            Device::LevelZero => "Level Zero",
            Device::Cpu | Device::Metal | Device::Ane => return None,
        };
        Some(format!(
            "--backend {self}: the {api} backend is a stub: it is not implemented because no {api} hardware is available to build and verify it"
        ))
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "metal" => Ok(Device::Metal),
            "ane" => Ok(Device::Ane),
            "cuda" => Ok(Device::Cuda),
            "rocm" => Ok(Device::Rocm),
            "level-zero" | "levelzero" => Ok(Device::LevelZero),
            other => Err(format!(
                "unknown --backend '{other}' (expected cpu, metal, ane, cuda, rocm, or level-zero)"
            )),
        }
    }
}

/// Model and graph placement mode, selected with `--backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PlacementMode {
    #[default]
    Auto,
    Pin(Device),
}

impl PlacementMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Pin(d) => d.name(),
        }
    }

    pub fn target_device(self) -> Option<Device> {
        match self {
            Self::Auto => None,
            Self::Pin(d) => Some(d),
        }
    }
}
impl From<Device> for PlacementMode {
    fn from(d: Device) -> Self {
        PlacementMode::Pin(d)
    }
}

impl fmt::Display for PlacementMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for PlacementMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("auto") {
            Ok(PlacementMode::Auto)
        } else {
            let d = Device::from_str(s)?;
            Ok(PlacementMode::Pin(d))
        }
    }
}

/// Returns hardware devices available at runtime.
pub fn available_devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    #[cfg(target_os = "macos")]
    {
        if metal::Device::system_default().is_some() {
            devices.push(Device::Metal);
        }
        devices.push(Device::Ane);
    }
    if crate::graph::cuda::is_available() {
        devices.push(Device::Cuda);
    }
    if crate::graph::rocm::is_available() {
        devices.push(Device::Rocm);
    }
    devices
}

/// The default high-performance compute device on this machine.
pub fn default_device() -> Device {
    #[cfg(target_os = "macos")]
    {
        if metal::Device::system_default().is_some() {
            return Device::Metal;
        }
    }
    Device::Cpu
}

/// Current resident memory (RSS) of this process in bytes.
pub fn current_resident_memory_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_out_cnt: *mut u32,
            ) -> i32;
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::uninit();
        let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
        unsafe {
            let kr = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            );
            if kr == 0 {
                return info.assume_init().resident_size as usize;
            }
        }
    }
    0
}

/// Performance profile of a compute device.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceProfile {
    pub mem_bandwidth_gbps: f64,
    pub compute_gflops: f64,
    pub bus_bandwidth_gbps: f64,
    pub dispatch_us: f64,
}

impl DeviceProfile {
    pub fn for_device(device: Device, host_bw: Option<f64>) -> Self {
        let hw = HardwareProfile::for_device(device);
        match device {
            Device::Cpu => {
                let num_cores = hw.compute_units as f64;
                let bw = host_bw.unwrap_or(hw.memory_bandwidth_gbps);
                Self {
                    mem_bandwidth_gbps: bw * 0.5,
                    compute_gflops: num_cores * 64.0,
                    bus_bandwidth_gbps: 1000.0,
                    dispatch_us: 1.0,
                }
            }
            Device::Metal => {
                let bw = host_bw.unwrap_or(hw.memory_bandwidth_gbps);
                Self {
                    mem_bandwidth_gbps: bw,
                    compute_gflops: (hw.compute_units as f64) * 300.0,
                    bus_bandwidth_gbps: bw,
                    dispatch_us: 5.0,
                }
            }
            Device::Ane => Self {
                mem_bandwidth_gbps: 100.0,
                compute_gflops: 15_000.0,
                bus_bandwidth_gbps: 64.0,
                dispatch_us: 20.0,
            },
            Device::Cuda | Device::Rocm => Self {
                mem_bandwidth_gbps: 500.0,
                compute_gflops: 20_000.0,
                bus_bandwidth_gbps: 32.0,
                dispatch_us: 10.0,
            },
            Device::LevelZero => Self {
                mem_bandwidth_gbps: 0.0,
                compute_gflops: 0.0,
                bus_bandwidth_gbps: 0.0,
                dispatch_us: 0.0,
            },
        }
    }

    #[cfg(target_os = "macos")]
    pub fn from_metal_host_profile(prof: &crate::graph::metal::HostProfile) -> Self {
        let measured_gflops = prof
            .gemm
            .iter()
            .filter_map(|t| {
                let min_us =
                    t.us.iter()
                        .map(|(_, us)| *us)
                        .filter(|&u| u > 0.0)
                        .fold(f64::INFINITY, f64::min);
                if min_us.is_finite() && min_us > 0.0 {
                    let ops = 2.0 * (t.m * t.n * t.k) as f64;
                    Some(ops / (min_us * 1000.0))
                } else {
                    None
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let compute_gflops = if measured_gflops.is_finite() && measured_gflops > 0.0 {
            measured_gflops
        } else {
            10_000.0
        };
        Self {
            mem_bandwidth_gbps: if prof.memory_bandwidth_gbps > 0.0 {
                prof.memory_bandwidth_gbps
            } else {
                250.0
            },
            compute_gflops,
            bus_bandwidth_gbps: if prof.memory_bandwidth_gbps > 0.0 {
                prof.memory_bandwidth_gbps
            } else {
                250.0
            },
            dispatch_us: if prof.dispatch_us > 0.0 {
                prof.dispatch_us
            } else {
                5.0
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    NVFP4,
    NVFP4_E2M1,
    MXFP4,
    MXFP6,
    EXL3_3Bit,
    EXL3_4Bit,
    EXL3_5Bit,
    AWQ_W4A16,
    GPTQ_W4A16,
    GGUF_Q4_0,
    GGUF_Q4_1,
    GGUF_Q4_K_M,
    GGUF_Q5_0,
    GGUF_Q5_1,
    GGUF_Q5_K_M,
    GGUF_Q6_K,
    GGUF_Q8_0,
    GGUF_IQ4_XS,
    GGUF_IQ4_NL,
    GGUF_Q2_K,
    GGUF_Q3_K,
    GGUF_IQ3_XXS,
    GGUF_IQ3_S,
    GGUF_IQ2_XXS,
    GGUF_IQ2_XS,
    GGUF_IQ2_S,
    GGUF_IQ2_M,
    GGUF_IQ1_S,
    GGUF_IQ1_M,
    GGUF_TQ1_0,
    GGUF_TQ2_0,
    BitNet_B1_58,
    KIVI_2Bit,
    TurboQuant,
    WeightlessJesseLUT,
    HadamardNeedle,
    /// bitsandbytes NF4/FP4 source tags; the GPU codecs are the blocksize-qualified `BNB_*_B*`.
    BNB_NF4,
    BNB_FP4,
    /// Repacked bitsandbytes 4-bit: per block of 64/128/256 elements, nibbles + f32 absmax.
    BNB_NF4_B64,
    BNB_NF4_B128,
    BNB_NF4_B256,
    BNB_FP4_B64,
    BNB_FP4_B128,
    BNB_FP4_B256,
    /// Repacked FP8: per block of 32 elements, 32 E4M3 bytes + f32 scale.
    FP8_E4M3_B32,
    /// Repacked MLX affine 4-bit: per group, nibbles + f16 scale + f16 bias.
    MLX_AFFINE4_G32,
    MLX_AFFINE4_G64,
    MLX_AFFINE4_G128,
    /// Repacked compressed-tensors int4: per group, nibbles + f16 scale + f16 zero point.
    INT4_G32,
    INT4_G64,
    INT4_G128,
    /// ggml type 40 `block_nvfp4`: per 64 elements, 4 UE4M3 scale bytes (one per 16) + 32 nibble bytes.
    /// Not the `NVFP4`/`NVFP4_E2M1` layout (`Nvfp4Tensor`: f32 scale per 16 elements).
    GGUF_NVFP4,
}
impl QuantFormat {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::WeightlessJesseLUT => 0.5,
            Self::BitNet_B1_58 => 1.58,
            Self::GGUF_IQ1_S => 1.5625,
            Self::GGUF_TQ1_0 => 1.6875,
            Self::GGUF_IQ1_M => 1.75,
            Self::KIVI_2Bit => 2.0,
            Self::GGUF_IQ2_XXS | Self::GGUF_TQ2_0 => 2.0625,
            Self::NVFP4 | Self::NVFP4_E2M1 => 2.25,
            Self::GGUF_IQ2_XS => 2.3125,
            Self::GGUF_IQ2_S | Self::GGUF_IQ2_M => 2.5625,
            Self::GGUF_Q2_K => 2.625,
            Self::GGUF_IQ3_XXS => 3.0625,
            Self::GGUF_Q3_K | Self::GGUF_IQ3_S => 3.4375,
            Self::MXFP4 | Self::GGUF_IQ4_XS => 4.25,
            Self::GGUF_NVFP4 => 4.5,
            Self::EXL3_3Bit => 3.0,
            Self::TurboQuant => 3.5,
            Self::EXL3_4Bit => 4.0,
            Self::HadamardNeedle => 4.0,
            Self::GGUF_Q4_0
            | Self::GGUF_Q4_K_M
            | Self::AWQ_W4A16
            | Self::GPTQ_W4A16
            | Self::GGUF_IQ4_NL => 4.5,
            Self::EXL3_5Bit | Self::GGUF_Q4_1 => 5.0,
            Self::GGUF_Q5_0 | Self::GGUF_Q5_K_M => 5.5,
            Self::MXFP6 | Self::GGUF_Q5_1 => 6.0,
            Self::GGUF_Q6_K => 6.5625,
            Self::FP8_E4M3 | Self::FP8_E5M2 => 8.0,
            Self::GGUF_Q8_0 => 8.5,
            Self::FP16 | Self::BF16 => 16.0,
            Self::FP32 => 32.0,
            Self::BNB_NF4 | Self::BNB_FP4 | Self::BNB_NF4_B64 | Self::BNB_FP4_B64 => 4.5,
            Self::BNB_NF4_B128 | Self::BNB_FP4_B128 => 4.25,
            Self::BNB_NF4_B256 | Self::BNB_FP4_B256 => 4.125,
            Self::FP8_E4M3_B32 => 9.0,
            Self::MLX_AFFINE4_G32 | Self::INT4_G32 => 5.0,
            Self::MLX_AFFINE4_G64 | Self::INT4_G64 => 4.5,
            Self::MLX_AFFINE4_G128 | Self::INT4_G128 => 4.25,
        }
    }

    pub fn is_smaller_than(&self, other: &QuantFormat) -> bool {
        self.bits_per_weight() < other.bits_per_weight()
    }
    pub fn tensor_bytes(&self, num_elements: usize) -> usize {
        match self {
            Self::GGUF_Q4_0 => (num_elements / 32) * 18,
            Self::GGUF_Q8_0 => (num_elements / 32) * 34,
            Self::GGUF_Q4_K_M => (num_elements / 256) * 144,
            Self::GGUF_IQ4_XS => (num_elements / 256) * 136,
            Self::GGUF_IQ3_S => (num_elements / 256) * 110,
            Self::GGUF_IQ2_S | Self::GGUF_IQ2_M => (num_elements / 256) * 82,
            Self::BF16 | Self::FP16 => num_elements * 2,
            Self::FP32 => num_elements * 4,
            // Every other GGUF block format: whole blocks of the exact size.
            f => match crate::model::layers::gguf_block_codec(*f) {
                Some((elems, bytes, _)) => num_elements / elems * bytes,
                None => (num_elements as f32 * self.bits_per_weight() / 8.0).ceil() as usize,
            },
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::FP32 => "FP32",
            Self::FP16 => "FP16",
            Self::BF16 => "BF16",
            Self::FP8_E4M3 => "FP8_E4M3",
            Self::FP8_E5M2 => "FP8_E5M2",
            Self::NVFP4 | Self::NVFP4_E2M1 => "NVFP4",
            Self::MXFP4 => "MXFP4",
            Self::MXFP6 => "MXFP6",
            Self::EXL3_3Bit => "EXL3_3Bit",
            Self::EXL3_4Bit => "EXL3_4Bit",
            Self::EXL3_5Bit => "EXL3_5Bit",
            Self::AWQ_W4A16 => "AWQ",
            Self::GPTQ_W4A16 => "GPTQ",
            Self::GGUF_Q4_0 => "Q4_0",
            Self::GGUF_Q4_1 => "Q4_1",
            Self::GGUF_Q4_K_M => "Q4_K_M",
            Self::GGUF_Q5_0 => "Q5_0",
            Self::GGUF_Q5_1 => "Q5_1",
            Self::GGUF_Q5_K_M => "Q5_K_M",
            Self::GGUF_Q6_K => "Q6_K",
            Self::GGUF_Q8_0 => "Q8_0",
            Self::GGUF_IQ4_XS => "IQ4_XS",
            Self::GGUF_IQ4_NL => "IQ4_NL",
            Self::GGUF_Q2_K => "Q2_K",
            Self::GGUF_Q3_K => "Q3_K",
            Self::GGUF_IQ3_XXS => "IQ3_XXS",
            Self::GGUF_IQ3_S => "IQ3_S",
            Self::GGUF_IQ2_XXS => "IQ2_XXS",
            Self::GGUF_IQ2_XS => "IQ2_XS",
            Self::GGUF_IQ2_S => "IQ2_S",
            Self::GGUF_IQ2_M => "IQ2_M",
            Self::GGUF_IQ1_S => "IQ1_S",
            Self::GGUF_IQ1_M => "IQ1_M",
            Self::GGUF_TQ1_0 => "TQ1_0",
            Self::GGUF_TQ2_0 => "TQ2_0",
            Self::BitNet_B1_58 => "BitNet_1.58",
            Self::KIVI_2Bit => "KIVI_2Bit",
            Self::TurboQuant => "TurboQuant",
            Self::WeightlessJesseLUT => "JesseLUT",
            Self::HadamardNeedle => "Needle",
            Self::BNB_NF4 => "BNB_NF4",
            Self::BNB_FP4 => "BNB_FP4",
            Self::BNB_NF4_B64 => "BNB_NF4_B64",
            Self::BNB_NF4_B128 => "BNB_NF4_B128",
            Self::BNB_NF4_B256 => "BNB_NF4_B256",
            Self::BNB_FP4_B64 => "BNB_FP4_B64",
            Self::BNB_FP4_B128 => "BNB_FP4_B128",
            Self::BNB_FP4_B256 => "BNB_FP4_B256",
            Self::FP8_E4M3_B32 => "FP8_E4M3_B32",
            Self::MLX_AFFINE4_G32 => "MLX_AFFINE4_G32",
            Self::MLX_AFFINE4_G64 => "MLX_AFFINE4_G64",
            Self::MLX_AFFINE4_G128 => "MLX_AFFINE4_G128",
            Self::INT4_G32 => "INT4_G32",
            Self::INT4_G64 => "INT4_G64",
            Self::INT4_G128 => "INT4_G128",
            Self::GGUF_NVFP4 => "GGUF_NVFP4",
        }
    }

    pub fn from_str_lenient(s: &str) -> Option<Self> {
        let clean = s.trim().to_lowercase().replace('-', "_");
        match clean.as_str() {
            "nvfp4" | "nvfp4_e2m1" | "fp4" => Some(Self::NVFP4_E2M1),
            "exl3" | "exl3_3bit" | "exl3_3" => Some(Self::EXL3_3Bit),
            "exl3_4bit" | "exl3_4" => Some(Self::EXL3_4Bit),
            "exl3_5bit" | "exl3_5" => Some(Self::EXL3_5Bit),
            "q4_k_m" | "q4_k" | "gguf_q4_k_m" => Some(Self::GGUF_Q4_K_M),
            "q4_0" | "gguf_q4_0" => Some(Self::GGUF_Q4_0),
            "q4_1" | "gguf_q4_1" => Some(Self::GGUF_Q4_1),
            "q5_0" | "gguf_q5_0" => Some(Self::GGUF_Q5_0),
            "q5_1" | "gguf_q5_1" => Some(Self::GGUF_Q5_1),
            "q5_k_m" | "q5_k" | "gguf_q5_k_m" => Some(Self::GGUF_Q5_K_M),
            "q6_k" | "gguf_q6_k" => Some(Self::GGUF_Q6_K),
            "q8_0" | "q8" | "gguf_q8_0" => Some(Self::GGUF_Q8_0),
            "bf16" | "bfloat16" => Some(Self::BF16),
            "fp16" | "f16" => Some(Self::FP16),
            "fp32" | "f32" | "uncompressed" => Some(Self::FP32),
            "fp8" | "fp8_e4m3" => Some(Self::FP8_E4M3),
            "fp8_e5m2" => Some(Self::FP8_E5M2),
            "iq4_xs" | "gguf_iq4_xs" => Some(Self::GGUF_IQ4_XS),
            "iq4_nl" | "gguf_iq4_nl" => Some(Self::GGUF_IQ4_NL),
            "q2_k" | "gguf_q2_k" => Some(Self::GGUF_Q2_K),
            "q3_k" | "gguf_q3_k" => Some(Self::GGUF_Q3_K),
            "iq3_xxs" | "gguf_iq3_xxs" => Some(Self::GGUF_IQ3_XXS),
            "iq3_s" | "gguf_iq3_s" => Some(Self::GGUF_IQ3_S),
            "iq2_xxs" | "gguf_iq2_xxs" => Some(Self::GGUF_IQ2_XXS),
            "iq2_xs" | "gguf_iq2_xs" => Some(Self::GGUF_IQ2_XS),
            "iq2_s" | "gguf_iq2_s" => Some(Self::GGUF_IQ2_S),
            "iq2_m" | "gguf_iq2_m" => Some(Self::GGUF_IQ2_M),
            "iq1_s" | "gguf_iq1_s" => Some(Self::GGUF_IQ1_S),
            "iq1_m" | "gguf_iq1_m" => Some(Self::GGUF_IQ1_M),
            "tq1_0" | "gguf_tq1_0" => Some(Self::GGUF_TQ1_0),
            "tq2_0" | "gguf_tq2_0" => Some(Self::GGUF_TQ2_0),
            "mxfp4" => Some(Self::MXFP4),
            "gguf_nvfp4" => Some(Self::GGUF_NVFP4),
            "kivi" | "kivi_2bit" => Some(Self::KIVI_2Bit),
            "turbo" | "turboquant" => Some(Self::TurboQuant),
            "jesse" | "weightless" | "lut" => Some(Self::WeightlessJesseLUT),
            "needle" | "hadamard" => Some(Self::HadamardNeedle),
            _ => None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantMode {
    Uncompressed,
    F16,
    BF16,
    Q8_0, // GGUF Q8_0: int8 values + f16 scale per 32
    NVFP4Ultra,
    EXL3Aggressive,  // 3-bit multi-rate
    EXL3Balanced,    // 4-bit multi-rate
    EXL3Quality,     // 5-bit multi-rate
    AdaptiveHybrid,  // Attention EXL3-4bit, FFN NVFP4, KV KIVI-2bit
    TurboSqueeze,    // NVFP4 + TurboQuant KV residual
    WeightlessJesse, // SolidSF Jesse Weightless Neural Network
}

impl QuantMode {
    pub fn primary_weight_format(&self) -> QuantFormat {
        match self {
            Self::Uncompressed => QuantFormat::FP32,
            Self::F16 => QuantFormat::FP16,
            Self::BF16 => QuantFormat::BF16,
            Self::Q8_0 => QuantFormat::GGUF_Q8_0,
            Self::NVFP4Ultra | Self::TurboSqueeze => QuantFormat::NVFP4_E2M1,
            Self::EXL3Aggressive => QuantFormat::EXL3_3Bit,
            Self::EXL3Balanced => QuantFormat::EXL3_4Bit,
            Self::EXL3Quality => QuantFormat::EXL3_5Bit,
            Self::AdaptiveHybrid => QuantFormat::NVFP4_E2M1,
            Self::WeightlessJesse => QuantFormat::WeightlessJesseLUT,
        }
    }

    pub fn effective_bits_per_weight(&self) -> f32 {
        match self {
            Self::Uncompressed => 32.0, // weights are held as f32
            Self::F16 | Self::BF16 => 16.0,
            Self::Q8_0 => 8.5,
            Self::NVFP4Ultra => 4.25, // 4-bit values + 16-elem block scale
            Self::EXL3Aggressive => 3.125,
            Self::EXL3Balanced => 4.125,
            Self::EXL3Quality => 5.125,
            Self::AdaptiveHybrid => 3.85,
            Self::TurboSqueeze => 3.5,
            Self::WeightlessJesse => 0.5,
        }
    }

    pub fn bits_per_weight(&self) -> f32 {
        self.effective_bits_per_weight()
    }

    pub fn is_smaller_than(&self, other: &QuantFormat) -> bool {
        self.bits_per_weight() < other.bits_per_weight()
    }
}

/// Hardware facts detected at runtime. Anything the platform cannot report is `None`/`false`, never guessed.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HardwareCapability {
    /// `machdep.cpu.brand_string` (macOS).
    pub cpu_name: Option<String>,
    /// `hw.memsize` (macOS).
    pub total_memory_bytes: Option<u64>,
    /// `hw.perflevel0.physicalcpu` (macOS, Apple Silicon).
    pub performance_cores: Option<u32>,
    /// `hw.perflevel1.physicalcpu` (macOS, Apple Silicon).
    pub efficiency_cores: Option<u32>,
    /// System GPU name from the Metal API (macOS).
    pub gpu_name: Option<String>,
    /// The Metal GPU shares memory with the CPU.
    pub has_unified_memory: bool,
    /// Advanced SIMD is architecturally mandatory on AArch64.
    pub cpu_has_arm_neon: bool,
    /// `hw.optional.arm.FEAT_*` runtime flags (macOS).
    pub cpu_has_arm_sme: bool,
    pub cpu_has_arm_sme2: bool,
    pub cpu_has_arm_i8mm: bool,
    pub cpu_has_arm_bf16: bool,
    pub cpu_has_arm_fp16: bool,
    /// `is_x86_feature_detected!` (x86_64).
    pub cpu_has_avx512_vnni: bool,
    pub cpu_has_amx: bool,
}

impl HardwareCapability {
    pub fn probe_system() -> Self {
        // Advanced SIMD is architecturally mandatory on AArch64.
        #[allow(unused_mut)] // nothing else is detected on non-macOS AArch64
        let mut hw = Self {
            cpu_has_arm_neon: cfg!(target_arch = "aarch64"),
            ..Self::default()
        };
        #[cfg(target_arch = "x86_64")]
        {
            hw.cpu_has_avx512_vnni = is_x86_feature_detected!("avx512vnni");
            hw.cpu_has_amx = is_x86_feature_detected!("amx-tile");
        }
        #[cfg(target_os = "macos")]
        {
            let feat = |name: &std::ffi::CStr| sysctl::int(name) == Some(1);
            hw.cpu_name = sysctl::string(c"machdep.cpu.brand_string");
            hw.total_memory_bytes = sysctl::int(c"hw.memsize");
            hw.performance_cores = sysctl::int(c"hw.perflevel0.physicalcpu").map(|n| n as u32);
            hw.efficiency_cores = sysctl::int(c"hw.perflevel1.physicalcpu").map(|n| n as u32);
            hw.cpu_has_arm_sme = feat(c"hw.optional.arm.FEAT_SME");
            hw.cpu_has_arm_sme2 = feat(c"hw.optional.arm.FEAT_SME2");
            hw.cpu_has_arm_i8mm = feat(c"hw.optional.arm.FEAT_I8MM");
            hw.cpu_has_arm_bf16 = feat(c"hw.optional.arm.FEAT_BF16");
            hw.cpu_has_arm_fp16 = feat(c"hw.optional.arm.FEAT_FP16");
            hw.gpu_name = crate::backend::metal::system_device_name().ok();
            hw.has_unified_memory =
                crate::backend::metal::system_has_unified_memory().unwrap_or(false);
        }
        hw
    }

    /// Detected values for the startup banner and TUI; undetected values say so.
    pub fn summary(&self) -> Vec<String> {
        const ND: &str = "not detected";
        let cores = match (self.performance_cores, self.efficiency_cores) {
            (Some(p), Some(e)) => format!(" ({p} performance + {e} efficiency cores)"),
            _ => String::new(),
        };
        let gpu = match &self.gpu_name {
            Some(n) if self.has_unified_memory => format!("{n} (unified memory)"),
            Some(n) => n.clone(),
            None => ND.into(),
        };
        let features: Vec<&str> = [
            (self.cpu_has_arm_neon, "NEON"),
            (self.cpu_has_arm_sme, "SME"),
            (self.cpu_has_arm_sme2, "SME2"),
            (self.cpu_has_arm_i8mm, "I8MM"),
            (self.cpu_has_arm_bf16, "BF16"),
            (self.cpu_has_arm_fp16, "FP16"),
            (self.cpu_has_avx512_vnni, "AVX512-VNNI"),
            (self.cpu_has_amx, "AMX"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect();
        vec![
            format!("CPU:    {}{cores}", self.cpu_name.as_deref().unwrap_or(ND)),
            format!(
                "Memory: {}",
                self.total_memory_bytes
                    .map_or(ND.into(), |b| format!("{:.1} GB", b as f64 / 1e9))
            ),
            format!("GPU:    {gpu}"),
            format!(
                "CPU features: {}",
                if features.is_empty() {
                    "none detected".into()
                } else {
                    features.join(" ")
                }
            ),
        ]
    }
}

/// `sysctlbyname(3)` reads (libSystem; no libc crate).
#[cfg(target_os = "macos")]
pub(crate) mod sysctl {
    use std::ffi::{c_char, c_int, c_void, CStr};

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Reads `name` into `buf`; `None` if the key does not exist (or does not fit).
    fn read(name: &CStr, buf: &mut [u8]) -> Option<usize> {
        let mut len = buf.len();
        // SAFETY: `buf` is writable for `len` bytes; sysctlbyname writes at most `len` and updates it.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(len)
    }

    /// A 4- or 8-byte integer key.
    pub fn int(name: &CStr) -> Option<u64> {
        let mut buf = [0u8; 8];
        match read(name, &mut buf)? {
            4 => Some(u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) as u64),
            8 => Some(u64::from_ne_bytes(buf)),
            _ => None,
        }
    }

    /// A NUL-terminated string key.
    pub fn string(name: &CStr) -> Option<String> {
        let mut buf = [0u8; 256];
        let len = read(name, &mut buf)?;
        let s = CStr::from_bytes_until_nul(&buf[..len])
            .ok()?
            .to_string_lossy();
        (!s.is_empty()).then(|| s.into_owned())
    }
}

/// Trait-based abstraction for quantization formats, enabling generic dequantization.
/// Each format defines its block structure, size, and dequantization logic.
pub trait QuantizationFormat {
    /// Associated type for per-block quantization parameters (scales, mins, etc).
    type BlockParams: Clone + std::fmt::Debug;

    /// Returns (block_width, block_height) for multi-dimensional blocking; typically (block_size, 1).
    fn block_size() -> (usize, usize);

    /// Bytes required to store one block's quantized data.
    fn bytes_per_block() -> usize;

    /// Dequantizes a single block of quantized bytes into f32 values.
    /// Returns a Vec of f32 with length equal to block_size().0 * block_size().1.
    fn dequantize_block(block: &[u8], params: &Self::BlockParams) -> Result<Vec<f32>, String>;

    /// Extract block parameters from the raw quantized bytes (e.g., scales, mins).
    /// Called once per block during initialization.
    fn extract_block_params(block_data: &[u8]) -> Result<Self::BlockParams, String>;
}

/// Marker types for quantization formats, implementing QuantizationFormat.
/// These are zero-cost abstractions enabling generic Tensor<Q> instantiation.

#[derive(Debug, Clone)]
pub struct Q4_0;

impl QuantizationFormat for Q4_0 {
    type BlockParams = (f32,); // single scale
    fn block_size() -> (usize, usize) {
        (32, 1)
    }
    fn bytes_per_block() -> usize {
        18
    } // 2 bytes scale + 16 bytes 4-bit values
    fn dequantize_block(block: &[u8], (scale,): &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 2 {
            return Err("Q4_0 block too small".to_string());
        }
        let mut out = vec![0.0f32; 32];
        let qh = &block[2..];
        for i in 0..16 {
            out[i] = ((qh[i] & 0xf) as f32 - 8.0) * scale;
            out[i + 16] = ((qh[i] >> 4) as f32 - 8.0) * scale;
        }
        Ok(out)
    }
    fn extract_block_params(block_data: &[u8]) -> Result<Self::BlockParams, String> {
        if block_data.len() < 2 {
            return Err("Q4_0 block too small".to_string());
        }
        let d_u16 = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = crate::model::f16_to_f32(d_u16);
        Ok((scale,))
    }
}

#[derive(Debug, Clone)]
pub struct Q4_1;

impl QuantizationFormat for Q4_1 {
    type BlockParams = (f32, f32); // scale, min
    fn block_size() -> (usize, usize) {
        (32, 1)
    }
    fn bytes_per_block() -> usize {
        20
    } // 2 scale + 2 min + 16 quantized
    fn dequantize_block(
        block: &[u8],
        (scale, min_val): &Self::BlockParams,
    ) -> Result<Vec<f32>, String> {
        if block.len() < 4 {
            return Err("Q4_1 block too small".to_string());
        }
        let mut out = vec![0.0f32; 32];
        let qh = &block[4..];
        for i in 0..16 {
            out[i] = (qh[i] & 0xf) as f32 * scale + min_val;
            out[i + 16] = (qh[i] >> 4) as f32 * scale + min_val;
        }
        Ok(out)
    }
    fn extract_block_params(block_data: &[u8]) -> Result<Self::BlockParams, String> {
        if block_data.len() < 4 {
            return Err("Q4_1 block too small".to_string());
        }
        let d_u16 = u16::from_le_bytes([block_data[0], block_data[1]]);
        let m_u16 = u16::from_le_bytes([block_data[2], block_data[3]]);
        let scale = crate::model::f16_to_f32(d_u16);
        let min_val = crate::model::f16_to_f32(m_u16);
        Ok((scale, min_val))
    }
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q4_K_M;

impl QuantizationFormat for Q4_K_M {
    type BlockParams = (f32, f32); // d (scale), dmin
    fn block_size() -> (usize, usize) {
        (256, 1)
    }
    fn bytes_per_block() -> usize {
        144
    } // 2 + 2 + 12 + 128
    fn dequantize_block(block: &[u8], (d, dmin): &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 144 {
            return Err("Q4_K block too small".to_string());
        }
        let mut out = vec![0.0f32; 256];
        let scales = &block[4..16];
        let qs = &block[16..144];
        // Simplified dequantization; full implementation follows llama.cpp k_scale_min
        for j in 0..8 {
            let scale_idx = j;
            let min_idx = 8 + j;
            let scale = (scales[scale_idx] as f32 + 1.0) / 256.0;
            let min = (scales[min_idx] as f32 - 128.0) / 256.0;
            for i in 0..32 {
                let q = qs[j * 32 + i] & 0xf;
                out[j * 32 + i] = d * scale * (q as f32) - dmin * min;
            }
        }
        Ok(out)
    }
    fn extract_block_params(block_data: &[u8]) -> Result<Self::BlockParams, String> {
        if block_data.len() < 4 {
            return Err("Q4_K block too small".to_string());
        }
        let d_u16 = u16::from_le_bytes([block_data[0], block_data[1]]);
        let dmin_u16 = u16::from_le_bytes([block_data[2], block_data[3]]);
        let d = crate::model::f16_to_f32(d_u16);
        let dmin = crate::model::f16_to_f32(dmin_u16);
        Ok((d, dmin))
    }
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Q5_K_M;

impl QuantizationFormat for Q5_K_M {
    type BlockParams = (f32, f32);
    fn block_size() -> (usize, usize) {
        (256, 1)
    }
    fn bytes_per_block() -> usize {
        176
    } // 2 + 2 + 12 + 160 for q5
    fn dequantize_block(block: &[u8], (d, dmin): &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 176 {
            return Err("Q5_K block too small".to_string());
        }
        let mut out = vec![0.0f32; 256];
        let scales = &block[4..16];
        let qs = &block[16..176];
        for j in 0..8 {
            let scale = (scales[j] as f32 + 1.0) / 256.0;
            let min = (scales[8 + j] as f32 - 128.0) / 256.0;
            for i in 0..32 {
                let low = qs[j * 40 + i] as u32;
                let high_idx = j * 40 + 32 + i / 4;
                let shift = (i % 4) * 2;
                let high = (qs[high_idx] >> shift) & 0x3;
                let q = low | ((high as u32) << 4);
                out[j * 32 + i] = d * scale * (q as f32 - 16.0) - dmin * min;
            }
        }
        Ok(out)
    }
    fn extract_block_params(block_data: &[u8]) -> Result<Self::BlockParams, String> {
        if block_data.len() < 4 {
            return Err("Q5_K block too small".to_string());
        }
        let d_u16 = u16::from_le_bytes([block_data[0], block_data[1]]);
        let dmin_u16 = u16::from_le_bytes([block_data[2], block_data[3]]);
        let d = crate::model::f16_to_f32(d_u16);
        let dmin = crate::model::f16_to_f32(dmin_u16);
        Ok((d, dmin))
    }
}

#[derive(Debug, Clone)]
pub struct Q8_0;

impl QuantizationFormat for Q8_0 {
    type BlockParams = (f32,);
    fn block_size() -> (usize, usize) {
        (32, 1)
    }
    fn bytes_per_block() -> usize {
        34
    } // 2 scale + 32 i8
    fn dequantize_block(block: &[u8], (scale,): &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 34 {
            return Err("Q8_0 block too small".to_string());
        }
        let mut out = vec![0.0f32; 32];
        for i in 0..32 {
            out[i] = (block[2 + i] as i8) as f32 * scale;
        }
        Ok(out)
    }
    fn extract_block_params(block_data: &[u8]) -> Result<Self::BlockParams, String> {
        if block_data.len() < 2 {
            return Err("Q8_0 block too small".to_string());
        }
        let d_u16 = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = crate::model::f16_to_f32(d_u16);
        Ok((scale,))
    }
}

#[derive(Debug, Clone)]
pub struct FP32;

impl QuantizationFormat for FP32 {
    type BlockParams = ();
    fn block_size() -> (usize, usize) {
        (1, 1)
    }
    fn bytes_per_block() -> usize {
        4
    }
    fn dequantize_block(block: &[u8], _: &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 4 {
            return Err("FP32 block too small".to_string());
        }
        Ok(vec![f32::from_le_bytes([
            block[0], block[1], block[2], block[3],
        ])])
    }
    fn extract_block_params(_: &[u8]) -> Result<Self::BlockParams, String> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FP16;

impl QuantizationFormat for FP16 {
    type BlockParams = ();
    fn block_size() -> (usize, usize) {
        (1, 1)
    }
    fn bytes_per_block() -> usize {
        2
    }
    fn dequantize_block(block: &[u8], _: &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 2 {
            return Err("FP16 block too small".to_string());
        }
        let u16_val = u16::from_le_bytes([block[0], block[1]]);
        Ok(vec![crate::model::f16_to_f32(u16_val)])
    }
    fn extract_block_params(_: &[u8]) -> Result<Self::BlockParams, String> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BF16;

impl QuantizationFormat for BF16 {
    type BlockParams = ();
    fn block_size() -> (usize, usize) {
        (1, 1)
    }
    fn bytes_per_block() -> usize {
        2
    }
    fn dequantize_block(block: &[u8], _: &Self::BlockParams) -> Result<Vec<f32>, String> {
        if block.len() < 2 {
            return Err("BF16 block too small".to_string());
        }
        let u16_val = u16::from_le_bytes([block[0], block[1]]);
        Ok(vec![crate::model::bf16_to_f32(u16_val)])
    }
    fn extract_block_params(_: &[u8]) -> Result<Self::BlockParams, String> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WeightlessJesseLUT;

impl QuantizationFormat for WeightlessJesseLUT {
    type BlockParams = ();
    fn block_size() -> (usize, usize) {
        (1, 1)
    }
    fn bytes_per_block() -> usize {
        1
    }
    fn dequantize_block(_: &[u8], _: &Self::BlockParams) -> Result<Vec<f32>, String> {
        Err("WeightlessJesseLUT requires lookup table context".to_string())
    }
    fn extract_block_params(_: &[u8]) -> Result<Self::BlockParams, String> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NVFP4;

impl QuantizationFormat for NVFP4 {
    type BlockParams = u8;
    fn block_size() -> (usize, usize) {
        (32, 1)
    }
    fn bytes_per_block() -> usize {
        16
    }
    fn dequantize_block(block: &[u8], _scale: &Self::BlockParams) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(32);
        for &byte in block {
            let hi = (byte >> 4) & 0xf;
            let lo = byte & 0xf;
            out.push((hi as f32) / 15.0);
            out.push((lo as f32) / 15.0);
        }
        Ok(out[..32].to_vec())
    }
    fn extract_block_params(block: &[u8]) -> Result<Self::BlockParams, String> {
        block
            .get(0)
            .copied()
            .ok_or_else(|| "Empty NVFP4 block".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct INT8;

impl QuantizationFormat for INT8 {
    type BlockParams = (f32, f32);
    fn block_size() -> (usize, usize) {
        (32, 1)
    }
    fn bytes_per_block() -> usize {
        33
    }
    fn dequantize_block(
        block: &[u8],
        (scale, zero_point): &Self::BlockParams,
    ) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(32);
        for &byte in &block[1..] {
            let val = (byte as i8 as f32 - zero_point) * scale;
            out.push(val);
        }
        Ok(out)
    }
    fn extract_block_params(block: &[u8]) -> Result<Self::BlockParams, String> {
        if block.len() < 1 {
            return Err("INT8 block too short".to_string());
        }
        Ok((1.0, block[0] as f32))
    }
}
