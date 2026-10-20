use crate::graph::codec::{planar, tile, wide, Codec, Tile};

#[derive(Debug, Clone, Copy)]
pub struct GemvParams {
    pub rows_per_sg: usize,
    pub k_unroll: usize,
    /// Simdgroups per threadgroup of [`gemv_packed`]: 8 (256 threads), or 2 (64 threads, ggml-metal's
    /// K-quant launch shape).
    pub simdgroups: usize,
}

impl Default for GemvParams {
    fn default() -> Self {
        Self {
            rows_per_sg: 2,
            k_unroll: 1,
            simdgroups: 8,
        }
    }
}

impl crate::hardware::QuantFormat {
    /// Alias for the GGUF Q4_K block format.
    pub const GGUF_Q4_K: Self = Self::GGUF_Q4_K_M;

    /// Every `QuantFormat` variant.
    pub const ALL: &'static [Self] = &[
        Self::FP32,
        Self::FP16,
        Self::BF16,
        Self::FP8_E4M3,
        Self::FP8_E5M2,
        Self::NVFP4,
        Self::NVFP4_E2M1,
        Self::MXFP4,
        Self::MXFP6,
        Self::EXL3_3Bit,
        Self::EXL3_4Bit,
        Self::EXL3_5Bit,
        Self::AWQ_W4A16,
        Self::GPTQ_W4A16,
        Self::GGUF_Q4_0,
        Self::GGUF_Q4_1,
        Self::GGUF_Q4_K_M,
        Self::GGUF_Q5_0,
        Self::GGUF_Q5_1,
        Self::GGUF_Q5_K_M,
        Self::GGUF_Q6_K,
        Self::GGUF_Q8_0,
        Self::GGUF_IQ4_XS,
        Self::GGUF_IQ4_NL,
        Self::GGUF_Q2_K,
        Self::GGUF_Q3_K,
        Self::GGUF_IQ3_XXS,
        Self::GGUF_IQ3_S,
        Self::GGUF_IQ2_XXS,
        Self::GGUF_IQ2_XS,
        Self::GGUF_IQ2_S,
        Self::GGUF_IQ2_M,
        Self::GGUF_IQ1_S,
        Self::GGUF_IQ1_M,
        Self::GGUF_TQ1_0,
        Self::GGUF_TQ2_0,
        Self::BitNet_B1_58,
        Self::KIVI_2Bit,
        Self::TurboQuant,
        Self::WeightlessJesseLUT,
        Self::HadamardNeedle,
        Self::BNB_NF4,
        Self::BNB_FP4,
        Self::BNB_NF4_B64,
        Self::BNB_NF4_B128,
        Self::BNB_NF4_B256,
        Self::BNB_FP4_B64,
        Self::BNB_FP4_B128,
        Self::BNB_FP4_B256,
        Self::FP8_E4M3_B32,
        Self::MLX_AFFINE4_G32,
        Self::MLX_AFFINE4_G64,
        Self::MLX_AFFINE4_G128,
        Self::INT4_G32,
        Self::INT4_G64,
        Self::INT4_G128,
        Self::GGUF_NVFP4,
    ];
}

/// Target GPU shading/compute language dialect for kernel synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Msl,
    Cuda,
    Hip,
}

impl Dialect {
    /// Selects the kernel dialect corresponding to a hardware profile's compute device.
    pub fn from_profile(profile: &crate::hardware::profile::HardwareProfile) -> Self {
        use crate::hardware::Device;
        match profile.device {
            Device::Cuda => Self::Cuda,
            Device::Rocm => Self::Hip,
            Device::Metal | Device::Ane | Device::Cpu | Device::LevelZero => Self::Msl,
        }
    }

    /// Returns the SIMD/warp/wavefront width for this dialect and optional hardware profile.
    pub fn simd_width(self, profile: Option<&crate::hardware::profile::HardwareProfile>) -> usize {
        profile.map(|p| p.simd_width).unwrap_or(match self {
            Self::Msl | Self::Cuda => 32,
            Self::Hip => 64,
        })
    }
}

fn translate_as_type(mut src: String) -> String {
    while let Some(pos) = src.find("as_type<") {
        let after_prefix = pos + "as_type<".len();
        let Some(gt_rel) = src[after_prefix..].find('>') else {
            break;
        };
        let gt_pos = after_prefix + gt_rel;
        let ty = src[after_prefix..gt_pos].trim().to_string();
        let open_paren = gt_pos + 1;
        if src.as_bytes().get(open_paren) != Some(&b'(') {
            break;
        }
        let mut depth = 1usize;
        let mut i = open_paren + 1;
        let bytes = src.as_bytes();
        while i < bytes.len() && depth > 0 {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        if depth != 0 {
            break;
        }
        let close_paren = i - 1;
        let inner = translate_as_type(src[open_paren + 1..close_paren].to_string());
        let trimmed = inner.trim();
        let is_ident = !trimmed.is_empty()
            && trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_');
        let replacement = match ty.as_str() {
            "float" => format!("__uint_as_float({inner})"),
            "uint" => format!("__float_as_uint({inner})"),
            "uchar4" if is_ident => format!("*reinterpret_cast<const uchar4*>(&{trimmed})"),
            "uchar4" => format!("gibbs_bitcast<uchar4>({inner})"),
            "char" => format!("static_cast<signed char>({inner})"),
            "half" => format!("gibbs_bitcast<__half>({inner})"),
            other => format!("gibbs_bitcast<{other}>({inner})"),
        };
        src.replace_range(pos..=close_paren, &replacement);
    }
    src
}

/// Translates an MSL codec or epilogue snippet into the target C++/GPU compute dialect.
pub fn translate_msl_to_c_dialect(msl: &str, dialect: Dialect) -> String {
    if dialect == Dialect::Msl {
        return msl.to_string();
    }
    let s = msl
        .replace("#pragma clang fp contract(off)\n", "")
        .replace("#pragma clang fp contract(off)", "")
        .replace(
            "float(*(device const half*)blk)",
            "__half2float(*(const __half*)blk)",
        )
        .replace(
            "float(*(device const half*)p)",
            "__half2float(*(const __half*)p)",
        )
        .replace(
            "float(*(device const half*)",
            "__half2float(*(const __half*)",
        )
        .replace(
            "half d = *(device const half*)blk;",
            "__half d = *(const __half*)blk;",
        )
        .replace("float(d)", "__half2float(d)")
        .replace(
            "float(as_type<half>(hb))",
            "__half2float(gibbs_bitcast<__half>(hb))",
        )
        .replace("device const uchar*", "const unsigned char*")
        .replace("device const ushort*", "const unsigned short*")
        .replace("device const char*", "const signed char*")
        .replace("device const half*", "const __half*")
        .replace("device const float*", "const float*")
        .replace("device const ", "const ")
        .replace("device float*", "float*")
        .replace("thread float*", "float*")
        .replace("thread uint&", "unsigned int&")
        .replace("constant const ", "__device__ const ")
        .replace("\nconstant char ", "\n__device__ const signed char ")
        .replace("\nconstant ", "\n__device__ const ")
        .replace(", constant ", ", const ")
        .replace("(constant ", "(const ")
        .replace("float4", "msl_float4")
        .replace("static inline ", "static __device__ inline ")
        .replace("inline ", "__device__ inline ")
        .replace(
            "static __device__ __device__ inline ",
            "static __device__ inline ",
        );
    translate_as_type(s)
}

/// Self-contained device prelude for CUDA and HIP translation units.
pub fn c_dialect_prelude(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Cuda => {
            r#"#ifndef GIBBS_C_DIALECT_PRELUDE
#define GIBBS_C_DIALECT_PRELUDE
#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#else
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
struct dim3 { unsigned int x, y, z; };
extern const __device__ dim3 blockIdx;
extern const __device__ dim3 blockDim;
extern const __device__ dim3 threadIdx;
extern const __device__ dim3 gridDim;
struct __half { unsigned short __x; __device__ inline operator float() const; };
typedef __half half;
__device__ inline float __uint_as_float(unsigned int u) { return __builtin_bit_cast(float, u); }
__device__ inline unsigned int __float_as_uint(float f) { return __builtin_bit_cast(unsigned int, f); }
__device__ inline float __half2float(__half h) {
    unsigned int s = (unsigned int)(h.__x >> 15) & 1u;
    unsigned int e = (unsigned int)(h.__x >> 10) & 0x1fu;
    unsigned int m = (unsigned int)h.__x & 0x3ffu;
    unsigned int bits = (e == 0u) ? (s << 31) : ((s << 31) | ((e + 112u) << 23) | (m << 13));
    return __uint_as_float(bits);
}
__device__ inline __half::operator float() const { return __half2float(*this); }
__device__ inline float __shfl_xor_sync(unsigned int, float v, int) { return v; }
__device__ inline float fabs(float x) { return __builtin_fabsf(x); }
__device__ inline float exp(float x) { return __builtin_expf(x); }
__device__ inline float tanh(float x) { return __builtin_tanhf(x); }
#endif
#ifndef M_SQRT1_2_F
#define M_SQRT1_2_F 0.7071067811865475244f
#endif
#define GIBBS_CAT2(a, b) a##b
#define GIBBS_CAT(a, b) GIBBS_CAT2(a, b)
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;
struct msl_uchar4 {
    unsigned char x, y, z, w;
    __device__ inline msl_uchar4() : x(0), y(0), z(0), w(0) {}
    __device__ inline explicit msl_uchar4(unsigned char s) : x(s), y(s), z(s), w(s) {}
    __device__ inline msl_uchar4(unsigned char x_, unsigned char y_, unsigned char z_, unsigned char w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ inline unsigned char operator[](unsigned int i) const { return (&x)[i]; }
};
struct msl_ushort2 {
    unsigned short x, y;
    __device__ inline msl_ushort2() : x(0), y(0) {}
    __device__ inline explicit msl_ushort2(unsigned short s) : x(s), y(s) {}
    __device__ inline msl_ushort2(unsigned short x_, unsigned short y_) : x(x_), y(y_) {}
    __device__ inline unsigned short operator[](unsigned int i) const { return (&x)[i]; }
};
struct msl_uint2 {
    unsigned int x, y;
    __device__ inline msl_uint2() : x(0), y(0) {}
    __device__ inline explicit msl_uint2(unsigned int s) : x(s), y(s) {}
    __device__ inline msl_uint2(unsigned int x_, unsigned int y_) : x(x_), y(y_) {}
    __device__ inline unsigned int operator[](unsigned int i) const { return (&x)[i]; }
};
struct msl_uint4 {
    unsigned int x, y, z, w;
    __device__ inline msl_uint4() : x(0), y(0), z(0), w(0) {}
    __device__ inline explicit msl_uint4(unsigned int s) : x(s), y(s), z(s), w(s) {}
    __device__ inline msl_uint4(unsigned int x_, unsigned int y_, unsigned int z_, unsigned int w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ inline unsigned int operator[](unsigned int i) const { return (&x)[i]; }
};
#undef uchar4
#define uchar4 msl_uchar4
#undef ushort2
#define ushort2 msl_ushort2
#undef uint2
#define uint2 msl_uint2
#undef uint4
#define uint4 msl_uint4
typedef msl_uint2 packed_uint2;
typedef msl_ushort2 packed_ushort2;
template <typename T, typename U> __device__ inline T gibbs_bitcast(U u) { return __builtin_bit_cast(T, u); }
struct msl_float4 {
    float x, y, z, w;
    __device__ inline msl_float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    __device__ inline explicit msl_float4(float s) : x(s), y(s), z(s), w(s) {}
    __device__ inline msl_float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ inline explicit msl_float4(msl_uchar4 u) : x(float(u.x)), y(float(u.y)), z(float(u.z)), w(float(u.w)) {}
    __device__ inline float operator[](unsigned int i) const { return (&x)[i]; }
    __device__ inline msl_float4 operator+(msl_float4 o) const { return msl_float4(x + o.x, y + o.y, z + o.z, w + o.w); }
    __device__ inline msl_float4& operator+=(msl_float4 o) { x += o.x; y += o.y; z += o.z; w += o.w; return *this; }
    __device__ inline msl_float4 operator-(float s) const { return msl_float4(x - s, y - s, z - s, w - s); }
    __device__ inline msl_float4 operator*(float s) const { return msl_float4(x * s, y * s, z * s, w * s); }
};
__device__ inline msl_float4 make_float4(float x, float y, float z, float w) { return msl_float4(x, y, z, w); }
__device__ inline float dot(msl_float4 a, msl_float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static __device__ inline float erf_approx(float x) {
    float z = fabs(x);
    float t = 1.0f / (1.0f + 0.5f * z);
    float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
        + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    float erfc_v = t * exp(poly);
    return x >= 0.0f ? 1.0f - erfc_v : erfc_v - 1.0f;
}
#endif"#
        }
        Dialect::Hip => {
            r#"#ifndef GIBBS_C_DIALECT_PRELUDE
#define GIBBS_C_DIALECT_PRELUDE
#if __has_include(<hip/hip_fp16.h>)
#include <hip/hip_fp16.h>
#else
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
struct dim3 { unsigned int x, y, z; };
extern const __device__ dim3 blockIdx;
extern const __device__ dim3 blockDim;
extern const __device__ dim3 threadIdx;
extern const __device__ dim3 gridDim;
struct __half { unsigned short __x; __device__ inline operator float() const; };
typedef __half half;
__device__ inline float __uint_as_float(unsigned int u) { return __builtin_bit_cast(float, u); }
__device__ inline unsigned int __float_as_uint(float f) { return __builtin_bit_cast(unsigned int, f); }
__device__ inline float __half2float(__half h) {
    unsigned int s = (unsigned int)(h.__x >> 15) & 1u;
    unsigned int e = (unsigned int)(h.__x >> 10) & 0x1fu;
    unsigned int m = (unsigned int)h.__x & 0x3ffu;
    unsigned int bits = (e == 0u) ? (s << 31) : ((s << 31) | ((e + 112u) << 23) | (m << 13));
    return __uint_as_float(bits);
}
__device__ inline __half::operator float() const { return __half2float(*this); }
__device__ inline float __shfl_xor(float v, int, int = 64) { return v; }
__device__ inline float fabs(float x) { return __builtin_fabsf(x); }
__device__ inline float exp(float x) { return __builtin_expf(x); }
__device__ inline float tanh(float x) { return __builtin_tanhf(x); }
#endif
#ifndef M_SQRT1_2_F
#define M_SQRT1_2_F 0.7071067811865475244f
#endif
#define GIBBS_CAT2(a, b) a##b
#define GIBBS_CAT(a, b) GIBBS_CAT2(a, b)
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;
struct msl_uchar4 {
    unsigned char x, y, z, w;
    __device__ inline msl_uchar4() : x(0), y(0), z(0), w(0) {}
    __device__ inline explicit msl_uchar4(unsigned char s) : x(s), y(s), z(s), w(s) {}
    __device__ inline msl_uchar4(unsigned char x_, unsigned char y_, unsigned char z_, unsigned char w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ inline unsigned char operator[](unsigned int i) const { return (&x)[i]; }
};
struct msl_ushort2 {
    unsigned short x, y;
    __device__ inline msl_ushort2() : x(0), y(0) {}
    __device__ inline explicit msl_ushort2(unsigned short s) : x(s), y(s) {}
    __device__ inline msl_ushort2(unsigned short x_, unsigned short y_) : x(x_), y(y_) {}
    __device__ inline unsigned short operator[](unsigned int i) const { return (&x)[i]; }
};
struct msl_uint2 {
    unsigned int x, y;
    __device__ inline msl_uint2() : x(0), y(0) {}
    __device__ inline explicit msl_uint2(unsigned int s) : x(s), y(s) {}
    __device__ inline msl_uint2(unsigned int x_, unsigned int y_) : x(x_), y(y_) {}
    __device__ inline unsigned int operator[](unsigned int i) const { return (&x)[i]; }
};
struct msl_uint4 {
    unsigned int x, y, z, w;
    __device__ inline msl_uint4() : x(0), y(0), z(0), w(0) {}
    __device__ inline explicit msl_uint4(unsigned int s) : x(s), y(s), z(s), w(s) {}
    __device__ inline msl_uint4(unsigned int x_, unsigned int y_, unsigned int z_, unsigned int w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ inline unsigned int operator[](unsigned int i) const { return (&x)[i]; }
};
#undef uchar4
#define uchar4 msl_uchar4
#undef ushort2
#define ushort2 msl_ushort2
#undef uint2
#define uint2 msl_uint2
#undef uint4
#define uint4 msl_uint4
typedef msl_uint2 packed_uint2;
typedef msl_ushort2 packed_ushort2;
template <typename T, typename U> __device__ inline T gibbs_bitcast(U u) { return __builtin_bit_cast(T, u); }
struct msl_float4 {
    float x, y, z, w;
    __device__ inline msl_float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    __device__ inline explicit msl_float4(float s) : x(s), y(s), z(s), w(s) {}
    __device__ inline msl_float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ inline explicit msl_float4(msl_uchar4 u) : x(float(u.x)), y(float(u.y)), z(float(u.z)), w(float(u.w)) {}
    __device__ inline float operator[](unsigned int i) const { return (&x)[i]; }
    __device__ inline msl_float4 operator+(msl_float4 o) const { return msl_float4(x + o.x, y + o.y, z + o.z, w + o.w); }
    __device__ inline msl_float4& operator+=(msl_float4 o) { x += o.x; y += o.y; z += o.z; w += o.w; return *this; }
    __device__ inline msl_float4 operator-(float s) const { return msl_float4(x - s, y - s, z - s, w - s); }
    __device__ inline msl_float4 operator*(float s) const { return msl_float4(x * s, y * s, z * s, w * s); }
};
__device__ inline msl_float4 make_float4(float x, float y, float z, float w) { return msl_float4(x, y, z, w); }
__device__ inline float dot(msl_float4 a, msl_float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static __device__ inline float erf_approx(float x) {
    float z = fabs(x);
    float t = 1.0f / (1.0f + 0.5f * z);
    float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
        + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    float erfc_v = t * exp(poly);
    return x >= 0.0f ? 1.0f - erfc_v : erfc_v - 1.0f;
}
#endif"#
        }
        Dialect::Msl => "",
    }
}

fn dialect_preamble(dialect: Dialect) -> &'static str {
    c_dialect_prelude(dialect)
}

fn gemv_common_for_dialect(dialect: Dialect) -> String {
    let common_msl = gemv_common(include_str!("metal_kernels.metal"));
    if dialect == Dialect::Msl {
        return common_msl.to_string();
    }
    format!(
        "#ifndef GIBBS_GEMV_COMMON\n#define GIBBS_GEMV_COMMON\n{}\n#endif\n",
        translate_msl_to_c_dialect(common_msl, dialect)
    )
}

fn warp_reduce_sum(var: &str, dialect: Dialect, simd_width: usize, indent: &str) -> String {
    let mut out = String::new();
    let mut offset = simd_width / 2;
    while offset >= 1 {
        match dialect {
            Dialect::Cuda => {
                out.push_str(&format!(
                    "{indent}{var} += __shfl_xor_sync(0xffffffffu, {var}, {offset});\n"
                ));
            }
            Dialect::Hip => {
                out.push_str(&format!(
                    "{indent}{var} += __shfl_xor({var}, {offset}, {simd_width});\n"
                ));
            }
            Dialect::Msl => unreachable!(),
        }
        offset /= 2;
    }
    out
}

/// Generates an MSL GEMV kernel function from a codec, parameters, prologue, and epilogue.
pub fn gemv(codec: &Codec, p: GemvParams, prologue: &str, epilogue: &str) -> String {
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let prologue_expr = if prologue.is_empty() {
        "x[col]"
    } else {
        prologue
    };
    let epilogue_stmt = if epilogue.is_empty() {
        "out[row] = sum;"
    } else {
        epilogue
    };

    format!(
        r#"
// Generated GEMV for {format_str}
{codec_msl}

struct GemvParams_{format_str} {{
    uint rows;
    uint cols;
    uint block_elems;
    uint block_bytes;
}};

kernel void gemv_{format_str}(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant GemvParams_{format_str}& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    uint row = tg * {rows_per_sg} + (sg % {rows_per_sg});
    if (row >= p.rows) return;

    float sum = 0.0f;
    uint num_sub = p.cols / 8;
    for (uint s = lane; s < num_sub; s += 32) {{
        float dec8[8];
        uint blk_idx = (s * 8) / p.block_elems;
        uint sub_in_blk = (s * 8) % p.block_elems / 8;
        device const uchar* blk = weights + (ulong)row * (p.cols / p.block_elems * p.block_bytes) + blk_idx * p.block_bytes;
        decode8(blk, sub_in_blk, dec8);

        for (uint i = 0; i < 8; i++) {{
            uint col = s * 8 + i;
            float x_val = {prologue_expr};
            sum += dec8[i] * x_val;
        }}
    }}

    sum = simd_sum(sum);
    if (lane == 0) {{
        {epilogue_stmt}
    }}
}}
"#,
        format_str = format_str,
        codec_msl = codec.msl,
        rows_per_sg = p.rows_per_sg,
        prologue_expr = prologue_expr,
        epilogue_stmt = epilogue_stmt,
    )
}

/// Generates an MSL Multi-GEMV kernel where N weights share the same input activation vector.
pub fn multi_gemv(codec: &Codec, n_weights: usize, prologue: &str, epilogue: &str) -> String {
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let prologue_expr = if prologue.is_empty() {
        "x[col]"
    } else {
        prologue
    };
    let epilogue_stmt = if epilogue.is_empty() {
        "for (uint w = 0; w < N; w++) out[w * p.rows + row] = sum[w];"
    } else {
        epilogue
    };

    format!(
        r#"
// Generated Multi-GEMV (N={n_weights}) for {format_str}
{codec_msl}

struct MultiGemvParams_{format_str} {{
    uint rows;
    uint cols;
    uint block_elems;
    uint block_bytes;
}};

kernel void multi_gemv_{format_str}_{n_weights}(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MultiGemvParams_{format_str}& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    uint row = tg * 2 + (sg % 2);
    if (row >= p.rows) return;

    float sum[{n_weights}];
    for (uint w = 0; w < {n_weights}; w++) sum[w] = 0.0f;

    uint num_sub = p.cols / 8;
    for (uint s = lane; s < num_sub; s += 32) {{
        for (uint i = 0; i < 8; i++) {{
            uint col = s * 8 + i;
            float x_val = {prologue_expr};
            for (uint w = 0; w < {n_weights}; w++) {{
                float dec8[8];
                uint blk_idx = (s * 8) / p.block_elems;
                uint sub_in_blk = (s * 8) % p.block_elems / 8;
                device const uchar* blk = weights + (ulong)(w * p.rows + row) * (p.cols / p.block_elems * p.block_bytes) + blk_idx * p.block_bytes;
                decode8(blk, sub_in_blk, dec8);
                sum[w] += dec8[i] * x_val;
            }}
        }}
    }}

    for (uint w = 0; w < {n_weights}; w++) {{
        sum[w] = simd_sum(sum[w]);
    }}

    if (lane == 0) {{
        {epilogue_stmt}
    }}
}}
"#,
        format_str = format_str,
        codec_msl = codec.msl,
        n_weights = n_weights,
        prologue_expr = prologue_expr,
        epilogue_stmt = epilogue_stmt,
    )
}

/// Alias for [`multi_gemv`].
pub fn gemv_multi(codec: &Codec, n_weights: usize, prologue: &str, epilogue: &str) -> String {
    multi_gemv(codec, n_weights, prologue, epilogue)
}

/// Generates a dialect-parameterized GEMV kernel function from a codec, parameters, dialect,
/// SIMD/warp width, prologue, and epilogue.
pub fn gemv_for_dialect(
    codec: &Codec,
    p: GemvParams,
    dialect: Dialect,
    simd_width: usize,
    prologue: &str,
    epilogue: &str,
) -> String {
    if dialect == Dialect::Msl {
        return gemv(codec, p, prologue, epilogue);
    }
    let simd_width = simd_width.max(1);
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let prologue_expr = if prologue.is_empty() {
        "x[col]".to_string()
    } else {
        translate_msl_to_c_dialect(prologue, dialect)
    };
    let epilogue_stmt = if epilogue.is_empty() {
        "out[row] = sum;".to_string()
    } else {
        translate_msl_to_c_dialect(epilogue, dialect)
    };
    let preamble = dialect_preamble(dialect);
    let common_c = gemv_common_for_dialect(dialect);
    let codec_c = translate_msl_to_c_dialect(codec.msl, dialect);
    let reduction = warp_reduce_sum("sum", dialect, simd_width, "    ");

    format!(
        r#"
// Generated GEMV ({dialect:?}) for {format_str}
{preamble}
{common_c}
#undef decode8
#define decode8 decode8_{format_str}
#ifndef GIBBS_DEC8_{format_str}
#define GIBBS_DEC8_{format_str}
{codec_c}
#endif

#ifdef GIBBS_KERN_gemv_{format_str}
#undef gemv_{format_str}
#define gemv_{format_str} GIBBS_CAT(gemv_{format_str}_v, __COUNTER__)
#endif
#define GIBBS_KERN_gemv_{format_str}
extern "C" __global__ void gemv_{format_str}(const unsigned char* __restrict__ weights, const float* __restrict__ x, float* __restrict__ out, unsigned int rows, unsigned int cols, unsigned int block_elems, unsigned int block_bytes) {{
    unsigned int tg = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int sg = tid / {simd_width};
    unsigned int lane = tid % {simd_width};
    unsigned int row = tg * {rows_per_sg} + (sg % {rows_per_sg});
    if (row >= rows) return;

    float sum = 0.0f;
    unsigned int num_sub = cols / 8;
    for (unsigned int s = lane; s < num_sub; s += {simd_width}) {{
        float dec8[8];
        unsigned int blk_idx = (s * 8) / block_elems;
        unsigned int sub_in_blk = (s * 8) % block_elems / 8;
        const unsigned char* blk = weights + (unsigned long long)row * (cols / block_elems * block_bytes) + blk_idx * block_bytes;
        decode8(blk, sub_in_blk, dec8);

        for (unsigned int i = 0; i < 8; i++) {{
            unsigned int col = s * 8 + i;
            float x_val = {prologue_expr};
            sum += dec8[i] * x_val;
        }}
    }}

{reduction}    if (lane == 0) {{
        const PackedGemvParams p = {{ rows, cols, 1u, 0u, 0u }};
        const float* res = out;
        (void)p;
        (void)res;
        {epilogue_stmt}
    }}
}}
"#,
        dialect = dialect,
        format_str = format_str,
        preamble = preamble,
        common_c = common_c,
        codec_c = codec_c,
        simd_width = simd_width,
        rows_per_sg = p.rows_per_sg,
        prologue_expr = prologue_expr,
        reduction = reduction,
        epilogue_stmt = epilogue_stmt,
    )
}

/// Generates a dialect-parameterized Multi-GEMV kernel where N weights share the same input activation vector.
pub fn multi_gemv_for_dialect(
    codec: &Codec,
    n_weights: usize,
    dialect: Dialect,
    simd_width: usize,
    prologue: &str,
    epilogue: &str,
) -> String {
    if dialect == Dialect::Msl {
        return multi_gemv(codec, n_weights, prologue, epilogue);
    }
    let simd_width = simd_width.max(1);
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let prologue_expr = if prologue.is_empty() {
        "x[col]".to_string()
    } else {
        translate_msl_to_c_dialect(prologue, dialect)
    };
    let epilogue_stmt = if epilogue.is_empty() {
        format!("for (unsigned int w = 0; w < {n_weights}; w++) out[w * rows + row] = sum[w];")
    } else {
        translate_msl_to_c_dialect(epilogue, dialect)
    };
    let preamble = dialect_preamble(dialect);
    let common_c = gemv_common_for_dialect(dialect);
    let codec_c = translate_msl_to_c_dialect(codec.msl, dialect);
    let reduction_w = warp_reduce_sum("sum[w]", dialect, simd_width, "        ");

    format!(
        r#"
// Generated Multi-GEMV ({dialect:?}, N={n_weights}) for {format_str}
{preamble}
{common_c}
#undef decode8
#define decode8 decode8_{format_str}
#ifndef GIBBS_DEC8_{format_str}
#define GIBBS_DEC8_{format_str}
{codec_c}
#endif

#ifdef GIBBS_KERN_multi_gemv_{format_str}_{n_weights}
#undef multi_gemv_{format_str}_{n_weights}
#define multi_gemv_{format_str}_{n_weights} GIBBS_CAT(multi_gemv_{format_str}_{n_weights}_v, __COUNTER__)
#endif
#define GIBBS_KERN_multi_gemv_{format_str}_{n_weights}
extern "C" __global__ void multi_gemv_{format_str}_{n_weights}(const unsigned char* __restrict__ weights, const float* __restrict__ x, float* __restrict__ out, unsigned int rows, unsigned int cols, unsigned int block_elems, unsigned int block_bytes) {{
    unsigned int tg = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int sg = tid / {simd_width};
    unsigned int lane = tid % {simd_width};
    unsigned int row = tg * 2 + (sg % 2);
    if (row >= rows) return;

    float sum[{n_weights}];
    for (unsigned int w = 0; w < {n_weights}; w++) sum[w] = 0.0f;

    unsigned int num_sub = cols / 8;
    for (unsigned int s = lane; s < num_sub; s += {simd_width}) {{
        for (unsigned int i = 0; i < 8; i++) {{
            unsigned int col = s * 8 + i;
            float x_val = {prologue_expr};
            for (unsigned int w = 0; w < {n_weights}; w++) {{
                float dec8[8];
                unsigned int blk_idx = (s * 8) / block_elems;
                unsigned int sub_in_blk = (s * 8) % block_elems / 8;
                const unsigned char* blk = weights + (unsigned long long)(w * rows + row) * (cols / block_elems * block_bytes) + blk_idx * block_bytes;
                decode8(blk, sub_in_blk, dec8);
                sum[w] += dec8[i] * x_val;
            }}
        }}
    }}

    for (unsigned int w = 0; w < {n_weights}; w++) {{
{reduction_w}    }}

    if (lane == 0) {{
        const unsigned int N = {n_weights};
        const PackedGemvParams p = {{ rows, cols, 1u, 0u, 0u }};
        const float* res = out;
        (void)N;
        (void)p;
        (void)res;
        {epilogue_stmt}
    }}
}}
"#,
        dialect = dialect,
        n_weights = n_weights,
        format_str = format_str,
        preamble = preamble,
        common_c = common_c,
        codec_c = codec_c,
        simd_width = simd_width,
        prologue_expr = prologue_expr,
        reduction_w = reduction_w,
        epilogue_stmt = epilogue_stmt,
    )
}

/// Alias for [`multi_gemv_for_dialect`].
pub fn gemv_multi_for_dialect(
    codec: &Codec,
    n_weights: usize,
    dialect: Dialect,
    simd_width: usize,
    prologue: &str,
    epilogue: &str,
) -> String {
    multi_gemv_for_dialect(codec, n_weights, dialect, simd_width, prologue, epilogue)
}

/// Emits a self-contained translation unit for `dialect` (prepending device prelude,
/// `GEMV_COMMON` epilogue helpers translated to `dialect`, and the GEMV kernel).
pub fn c_dialect_translation_unit(
    codec: &Codec,
    p: GemvParams,
    dialect: Dialect,
    prologue: &str,
    epilogue: &str,
) -> String {
    if dialect == Dialect::Msl {
        return format!(
            "{}\n{}",
            gemv_common(include_str!("metal_kernels.metal")),
            gemv(codec, p, prologue, epilogue)
        );
    }
    gemv_for_dialect(
        codec,
        p,
        dialect,
        dialect.simd_width(None),
        prologue,
        epilogue,
    )
}

/// Emits a self-contained expert-indexed GEMV translation unit (`Op::MoeLinear`) for `dialect`.
pub fn c_dialect_indexed_translation_unit(
    dialect: Dialect,
    codec: &Codec,
    p: GemvParams,
) -> String {
    if dialect == Dialect::Msl {
        return format!(
            "{}\n{}",
            gemv_common(include_str!("metal_kernels.metal")),
            gemv_indexed(codec, p)
        );
    }
    gemv_indexed_for_dialect(dialect, codec, p)
}

fn warp_reduce_topk(dialect: Dialect, simd_width: usize, indent: &str) -> String {
    let mut out = String::new();
    let mut offset = simd_width / 2;
    while offset >= 1 {
        let (shfl_v, shfl_i) = match dialect {
            Dialect::Cuda => (
                format!("__shfl_xor_sync(0xffffffffu, bv, {offset})"),
                format!("__shfl_xor_sync(0xffffffffu, bi_f, {offset})"),
            ),
            Dialect::Hip => (
                format!("__shfl_xor(bv, {offset}, {simd_width})"),
                format!("__shfl_xor(bi_f, {offset}, {simd_width})"),
            ),
            Dialect::Msl => unreachable!(),
        };
        out.push_str(&format!(
            "{indent}{{\n{indent}    float bi_f = (bi == 0xffffffffu) ? -1.0f : (float)bi;\n{indent}    float ov = {shfl_v};\n{indent}    float oi_f = {shfl_i};\n{indent}    unsigned int oi = (oi_f < 0.0f) ? 0xffffffffu : (unsigned int)oi_f;\n{indent}    if (oi != 0xffffffffu && (bi == 0xffffffffu || ov > bv || (ov == bv && oi < bi))) {{\n{indent}        bv = ov;\n{indent}        bi = oi;\n{indent}    }}\n{indent}}}\n"
        ));
        offset /= 2;
    }
    out
}

/// Emits a self-contained fused `gibbs_topk_softmax_f32` translation unit for `Dialect::Cuda` / `Dialect::Hip`.
pub fn topk_softmax_for_dialect(dialect: Dialect) -> String {
    if dialect == Dialect::Msl {
        return String::new();
    }
    let simd_width = dialect.simd_width(None).max(1);
    let preamble = dialect_preamble(dialect);
    let reduction = warp_reduce_topk(dialect, simd_width, "        ");
    format!(
        r#"
// Generated TopKSoftmax ({dialect:?})
{preamble}
#ifndef GIBBS_KERN_gibbs_topk_softmax_f32
#define GIBBS_KERN_gibbs_topk_softmax_f32
extern "C" __global__ void gibbs_topk_softmax_f32(
    const float* __restrict__ x,
    unsigned int* __restrict__ out_ids,
    float* __restrict__ out_weights,
    const float* __restrict__ scale,
    unsigned int p_rows,
    unsigned int p_cols,
    unsigned int p_k,
    unsigned int p_has_scale
) {{
    unsigned int row = blockIdx.x;
    unsigned int lane = threadIdx.x;
    if (row >= p_rows || lane >= {simd_width}) return;
    const float pos_inf = __uint_as_float(0x7f800000u);
    const float neg_inf = __uint_as_float(0xff800000u);
    const float nan_val = __uint_as_float(0x7fc00000u);
    const float* xr = x + (unsigned long long)row * p_cols;
    float picked_v[32];
    unsigned int picked_i[32];
    float pv = pos_inf;
    unsigned int pi = 0xffffffffu;
    unsigned int k_lim = p_k <= 32u ? p_k : 32u;
    for (unsigned int round = 0; round < k_lim; ++round) {{
        float bv = neg_inf;
        unsigned int bi = 0xffffffffu;
        for (unsigned int i = lane; i < p_cols; i += {simd_width}) {{
            float raw = xr[i];
            float v = (raw == raw) ? raw : neg_inf;
            if (pi == 0xffffffffu || v < pv || (v == pv && i > pi)) {{
                if (bi == 0xffffffffu || v > bv || (v == bv && i < bi)) {{
                    bv = v;
                    bi = i;
                }}
            }}
        }}
{reduction}        if (lane == 0) {{
            out_ids[(unsigned long long)row * p_k + round] = bi;
            picked_v[round] = bv;
            picked_i[round] = bi;
        }}
        pv = bv;
        pi = bi;
    }}
    if (lane == 0 && k_lim > 0u && out_weights != 0) {{
        float max_v = picked_v[0];
        for (unsigned int r = 1; r < k_lim; ++r) {{
            if (picked_v[r] > max_v) max_v = picked_v[r];
        }}
        float sum_exp = 0.0f;
        for (unsigned int r = 0; r < k_lim; ++r) {{
            picked_v[r] = exp(picked_v[r] - max_v);
            sum_exp += picked_v[r];
        }}
        for (unsigned int r = 0; r < k_lim; ++r) {{
            float w = picked_v[r] / sum_exp;
            if (p_has_scale != 0u && scale != 0) {{
                w *= (picked_i[r] < p_cols ? scale[picked_i[r]] : nan_val);
            }}
            out_weights[(unsigned long long)row * p_k + r] = w;
        }}
    }}
}}
#endif
"#,
        dialect = dialect,
        preamble = preamble,
        simd_width = simd_width,
        reduction = reduction,
    )
}

/// Emits a self-contained fused `gibbs_split_glu_f32` translation unit for `Dialect::Cuda` / `Dialect::Hip`.
pub fn split_glu_for_dialect(dialect: Dialect) -> String {
    if dialect == Dialect::Msl {
        return String::new();
    }
    let preamble = dialect_preamble(dialect);
    let common_c = gemv_common_for_dialect(dialect);
    format!(
        r#"
// Generated SplitGlu ({dialect:?})
{preamble}
{common_c}
#ifndef GIBBS_KERN_gibbs_split_glu_f32
#define GIBBS_KERN_gibbs_split_glu_f32
extern "C" __global__ void gibbs_split_glu_f32(
    const float* __restrict__ gu,
    float* __restrict__ out,
    unsigned int p_rows,
    unsigned int p_half_cols,
    unsigned int p_kind
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p_rows * p_half_cols) return;
    unsigned int r = idx / p_half_cols;
    unsigned int c = idx % p_half_cols;
    float g = gu[(unsigned long long)r * (2u * p_half_cols) + c];
    float u = gu[(unsigned long long)r * (2u * p_half_cols) + p_half_cols + c];
    float act = p_kind == 0u ? gelu_erf(g) : (p_kind == 1u ? gelu_tanh(g) : (g / (1.0f + exp(-g))));
    out[idx] = act * u;
}}
#endif
"#,
        dialect = dialect,
        preamble = preamble,
        common_c = common_c,
    )
}

/// Emits a self-contained fused `gibbs_moe_weighted_sum_f32` translation unit for `Dialect::Cuda` / `Dialect::Hip`.
pub fn moe_weighted_sum_for_dialect(dialect: Dialect) -> String {
    if dialect == Dialect::Msl {
        return String::new();
    }
    let preamble = dialect_preamble(dialect);
    format!(
        r#"
// Generated MoeWeightedSum ({dialect:?})
{preamble}
#ifndef GIBBS_KERN_gibbs_moe_weighted_sum_f32
#define GIBBS_KERN_gibbs_moe_weighted_sum_f32
extern "C" __global__ void gibbs_moe_weighted_sum_f32(
    const float* __restrict__ down,
    const float* __restrict__ weights,
    float* __restrict__ out,
    unsigned int p_n,
    unsigned int p_k,
    unsigned int p_h
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p_n * p_h) return;
    unsigned int r = idx / p_h;
    unsigned int c = idx % p_h;
    float sum = 0.0f;
    unsigned long long base = (unsigned long long)r * p_k;
    for (unsigned int s = 0; s < p_k; ++s) {{
        sum += down[(base + s) * p_h + c] * weights[base + s];
    }}
    out[idx] = sum;
}}
#endif
"#,
        dialect = dialect,
        preamble = preamble,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::codec::{codec, CODEC_Q4_0};
    use crate::hardware::{Device, HardwareProfile, QuantFormat};

    #[test]
    fn gemv_packed_64_thread_variant_covers_two_simdgroups() {
        let c = codec(QuantFormat::GGUF_Q4_K_M).unwrap();
        let p = GemvParams {
            simdgroups: 2,
            ..Default::default()
        };
        assert_eq!((gemv_rows_per_tg(p), gemv_threads(p)), (4, 64));
        assert_eq!(gemv_packed_name(c, p), "gemvgen_GGUF_Q4_K_M_r2_sg2");
        assert!(gemv_packed(c, p).contains("const uint row0 = tg.x * 4 + sg * 2;"));
        // The 256-thread default keeps its name and source (tuned shapes keep their cache keys).
        let d = GemvParams::default();
        assert_eq!(gemv_packed_name(c, d), "gemvgen_GGUF_Q4_K_M_r2");
        assert!(gemv_packed(c, d).contains("const uint row0 = tg.x * 16 + sg * 2;"));
    }

    #[test]
    fn gemv_packed_specializable_reads_shape_from_function_constants() {
        for f in [QuantFormat::GGUF_TQ2_0, QuantFormat::GGUF_Q4_1, QuantFormat::GGUF_Q6_K] {
            let (c, p) = (codec(f).unwrap(), GemvParams::default());
            let s = gemv_packed_specializable(c, p);
            let kernel = s.split_once("kernel void ").unwrap().1;
            assert!(!kernel.contains("p.rows ") && !kernel.contains("p.cols /"), "{f:?}");
            assert!(kernel.contains("const uint n_rows = fc_has_rows ? fc_rows : p.rows;"));
            assert!(s.contains("[[function_constant(1)]]"));
            assert!(!gemv_packed(c, p).contains("function_constant"));
        }
    }

    #[test]
    fn iq2_grid_rewrites_hit_every_anchor() {
        let c = codec(QuantFormat::GGUF_IQ2_XXS).unwrap();
        for r in [1, 2, 4] {
            let p = GemvParams {
                rows_per_sg: r,
                ..Default::default()
            };
            let src = gemv_indexed(c, p);
            assert_eq!(iq2_grid_msl(src.clone(), Iq2Grid::Program).unwrap(), src);
            let tg = iq2_grid_msl(src.clone(), Iq2Grid::Threadgroup).unwrap();
            assert!(tg.contains("threadgroup uint grid[512];"));
            assert_eq!(tg.matches("(b, g, xs, grid);").count(), r);
            let buf = iq2_grid_msl(src.clone(), Iq2Grid::Buffer).unwrap();
            assert!(buf.starts_with("struct Iq2GridTable { uint2 g[256]; };"));
            assert!(buf.contains("constant Iq2GridTable& grid [[buffer(5)]],"));
            assert!(buf.contains("uint2 gw = grid.g[((q4 >> (8 * k)) & 0xFFu)];"));
            for s in [&tg, &buf] {
                assert!(!s.contains("as_type<uint2>(IQ2XXS_GRID[(q4"));
            }
        }
        // Any other codec has no grid lookup to rewrite.
        let q4 = gemv_indexed(codec(QuantFormat::GGUF_Q4_0).unwrap(), GemvParams::default());
        assert!(iq2_grid_msl(q4, Iq2Grid::Buffer).is_err());
    }

    #[test]
    fn test_gemv_codegen() {
        let code = gemv(&CODEC_Q4_0, GemvParams::default(), "", "");
        assert!(code.contains("kernel void gemv_gguf_q4_0"));
        assert!(code.contains("decode8("));
        assert!(code.contains("out[row] = sum;"));
    }

    #[test]
    fn test_multi_gemv_codegen() {
        let code = multi_gemv(&CODEC_Q4_0, 2, "x[col]", "out[row] = sum[0] * sum[1];");
        assert!(code.contains("kernel void multi_gemv_gguf_q4_0_2"));
        assert!(code.contains("out[row] = sum[0] * sum[1];"));
    }

    #[test]
    fn test_dialect_from_profile_and_simd_width() {
        let cuda_prof = HardwareProfile::for_device(Device::Cuda);
        let rocm_prof = HardwareProfile::for_device(Device::Rocm);
        let metal_prof = HardwareProfile::for_device(Device::Metal);
        let cpu_prof = HardwareProfile::for_device(Device::Cpu);

        assert_eq!(Dialect::from_profile(&cuda_prof), Dialect::Cuda);
        assert_eq!(Dialect::from_profile(&rocm_prof), Dialect::Hip);
        assert_eq!(Dialect::from_profile(&metal_prof), Dialect::Msl);
        assert_eq!(Dialect::from_profile(&cpu_prof), Dialect::Msl);

        assert_eq!(Dialect::Msl.simd_width(None), 32);
        assert_eq!(Dialect::Cuda.simd_width(None), 32);
        assert_eq!(Dialect::Hip.simd_width(None), 64);
        assert_eq!(
            Dialect::Cuda.simd_width(Some(&cuda_prof)),
            cuda_prof.simd_width
        );
        assert_eq!(
            Dialect::Hip.simd_width(Some(&rocm_prof)),
            rocm_prof.simd_width
        );
    }

    #[test]
    fn test_dialect_msl_matches_gemv_byte_for_byte_across_all_formats() {
        assert!(crate::graph::codec::formats()
            .iter()
            .all(|f| QuantFormat::ALL.contains(f)));
        let p = GemvParams::default();
        let mut checked = 0usize;
        for &fmt in QuantFormat::ALL {
            if let Some(c) = codec(fmt) {
                assert_eq!(
                    gemv_for_dialect(c, p, Dialect::Msl, 32, "", ""),
                    gemv(c, p, "", ""),
                    "MSL GEMV mismatch for {:?}",
                    fmt
                );
                assert_eq!(
                    multi_gemv_for_dialect(c, 2, Dialect::Msl, 32, "", ""),
                    multi_gemv(c, 2, "", ""),
                    "MSL Multi-GEMV mismatch for {:?}",
                    fmt
                );
                assert_eq!(
                    gemv_indexed_for_dialect(Dialect::Msl, c, p),
                    gemv_indexed(c, p),
                    "MSL Indexed GEMV mismatch for {:?}",
                    fmt
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn test_dialect_cuda_and_hip_across_all_codecs() {
        let p = GemvParams::default();
        let forbidden = [
            "[[buffer",
            "[[threadgroup_position_in_grid]]",
            "device const",
            "thread float*",
            "simd_sum(",
            "as_type<",
        ];
        let mut checked = 0usize;
        for &fmt in QuantFormat::ALL {
            let Some(c) = codec(fmt) else { continue };
            checked += 1;

            let cuda_src = gemv_for_dialect(c, p, Dialect::Cuda, 32, "", "");
            assert!(!cuda_src.is_empty());
            assert!(cuda_src.contains("#include <cuda_fp16.h>"));
            assert!(cuda_src.contains("extern \"C\" __global__ void gemv_"));
            assert!(cuda_src.contains("decode8("));
            assert!(cuda_src.contains("unsigned int tg = blockIdx.x;"));
            assert!(cuda_src.contains("unsigned int tid = threadIdx.x;"));
            assert!(cuda_src.contains("unsigned int sg = tid / 32;"));
            assert!(cuda_src.contains("unsigned int lane = tid % 32;"));
            assert!(!cuda_src.contains("&a &"));
            for tok in forbidden {
                assert!(
                    !cuda_src.contains(tok),
                    "CUDA kernel for {:?} contains leftover MSL token {tok:?}",
                    fmt
                );
            }
            for off in [16, 8, 4, 2, 1] {
                let pat = format!("__shfl_xor_sync(0xffffffffu, sum, {off});");
                assert!(
                    cuda_src.contains(&pat),
                    "CUDA kernel for {:?} missing warp shuffle {pat}",
                    fmt
                );
            }

            let hip_src = gemv_for_dialect(c, p, Dialect::Hip, 64, "", "");
            assert!(!hip_src.is_empty());
            assert!(hip_src.contains("#include <hip/hip_fp16.h>"));
            assert!(hip_src.contains("extern \"C\" __global__ void gemv_"));
            assert!(hip_src.contains("decode8("));
            assert!(hip_src.contains("unsigned int tg = blockIdx.x;"));
            assert!(hip_src.contains("unsigned int tid = threadIdx.x;"));
            assert!(hip_src.contains("unsigned int sg = tid / 64;"));
            assert!(hip_src.contains("unsigned int lane = tid % 64;"));
            assert!(!hip_src.contains("&a &"));
            for tok in forbidden {
                assert!(
                    !hip_src.contains(tok),
                    "HIP kernel for {:?} contains leftover MSL token {tok:?}",
                    fmt
                );
            }
            for off in [32, 16, 8, 4, 2, 1] {
                let pat = format!("__shfl_xor(sum, {off}, 64);");
                assert!(
                    hip_src.contains(&pat),
                    "HIP kernel for {:?} missing wave shuffle {pat}",
                    fmt
                );
            }

            let cuda_multi = multi_gemv_for_dialect(c, 2, Dialect::Cuda, 32, "", "");
            assert!(cuda_multi.contains("extern \"C\" __global__ void multi_gemv_"));
            assert!(cuda_multi.contains("__shfl_xor_sync(0xffffffffu, sum[w], 16);"));

            let hip_multi = multi_gemv_for_dialect(c, 2, Dialect::Hip, 64, "", "");
            assert!(hip_multi.contains("extern \"C\" __global__ void multi_gemv_"));
            assert!(hip_multi.contains("__shfl_xor(sum[w], 32, 64);"));

            let cuda_ix = gemv_indexed_for_dialect(Dialect::Cuda, c, p);
            assert!(cuda_ix.contains(&format!(
                "extern \"C\" __global__ void {}",
                gemv_indexed_name(c, p)
            )));
            assert!(cuda_ix.contains("__shfl_xor_sync(0xffffffffu, sum, 16);"));
            for tok in forbidden {
                assert!(
                    !cuda_ix.contains(tok),
                    "CUDA indexed kernel for {:?} contains leftover MSL token {tok:?}",
                    fmt
                );
            }

            let hip_ix = gemv_indexed_for_dialect(Dialect::Hip, c, p);
            assert!(hip_ix.contains(&format!(
                "extern \"C\" __global__ void {}",
                gemv_indexed_name(c, p)
            )));
            assert!(hip_ix.contains("__shfl_xor(sum, 32, 64);"));
            for tok in forbidden {
                assert!(
                    !hip_ix.contains(tok),
                    "HIP indexed kernel for {:?} contains leftover MSL token {tok:?}",
                    fmt
                );
            }
        }
        assert!(checked >= 25);
        for dialect in [Dialect::Cuda, Dialect::Hip] {
            let topk_src = topk_softmax_for_dialect(dialect);
            assert!(topk_src.contains("extern \"C\" __global__ void gibbs_topk_softmax_f32"));
            let sglu_src = split_glu_for_dialect(dialect);
            assert!(sglu_src.contains("extern \"C\" __global__ void gibbs_split_glu_f32"));
            let wsum_src = moe_weighted_sum_for_dialect(dialect);
            assert!(wsum_src.contains("extern \"C\" __global__ void gibbs_moe_weighted_sum_f32"));
            for src in [&topk_src, &sglu_src, &wsum_src] {
                for tok in forbidden {
                    assert!(
                        !src.contains(tok),
                        "{dialect:?} fused MoE kernel contains leftover MSL token {tok:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn clang_compiles_cuda_and_hip_kernels_for_every_codec() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let clang_bin = if std::path::Path::new("/opt/homebrew/opt/llvm/bin/clang").exists() {
            "/opt/homebrew/opt/llvm/bin/clang"
        } else {
            let ok = Command::new("clang")
                .args([
                    "-x",
                    "cuda",
                    "--cuda-device-only",
                    "-nocudainc",
                    "-nocudalib",
                    "--cuda-gpu-arch=sm_80",
                    "-fsyntax-only",
                    "-",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                eprintln!("SKIPPED clang_compiles_cuda_and_hip_kernels_for_every_codec: clang without CUDA/HIP target support");
                return;
            }
            "clang"
        };

        let p = GemvParams::default();
        for (dialect, simd_width, args) in [
            (
                Dialect::Cuda,
                32usize,
                &[
                    "-x",
                    "cuda",
                    "--cuda-device-only",
                    "-nocudainc",
                    "-nocudalib",
                    "--cuda-gpu-arch=sm_80",
                    "-fsyntax-only",
                    "-",
                ][..],
            ),
            (
                Dialect::Hip,
                64usize,
                &[
                    "-x",
                    "hip",
                    "-nogpuinc",
                    "-nogpulib",
                    "--offload-arch=gfx90a",
                    "-fsyntax-only",
                    "-",
                ][..],
            ),
        ] {
            let mut tu = String::new();
            let mut compiled_codecs = 0usize;
            for &fmt in QuantFormat::ALL {
                if let Some(c) = codec(fmt) {
                    compiled_codecs += 1;
                    tu.push_str(&c_dialect_translation_unit(c, p, dialect, "", ""));
                    tu.push_str(&c_dialect_indexed_translation_unit(dialect, c, p));
                }
            }
            assert!(
                compiled_codecs >= 25,
                "expected at least 25 codecs compiled, got {compiled_codecs}"
            );
            tu.push_str(&gemv_multi_for_dialect(
                &CODEC_Q4_0,
                2,
                dialect,
                simd_width,
                "x[col]",
                "out[row] = gelu_erf(sum[0]) * gelu_tanh(sum[1]);",
            ));
            tu.push_str(&c_dialect_translation_unit(
                &CODEC_Q4_0,
                p,
                dialect,
                "x[col]",
                "for (uint epi = 0; epi <= 7; ++epi) { PackedGemvParams ep = { rows, cols, 1u, 0u, epi }; sum = apply_gemv_epi(sum, (ulong)row, ep, res); } out[row] = sum + erf_approx(sum);",
            ));
            tu.push_str(&topk_softmax_for_dialect(dialect));
            tu.push_str(&split_glu_for_dialect(dialect));
            tu.push_str(&moe_weighted_sum_for_dialect(dialect));

            let mut child = Command::new(clang_bin)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn clang");
            child
                .stdin
                .take()
                .expect("clang stdin")
                .write_all(tu.as_bytes())
                .expect("write translation unit to clang stdin");
            let out = child.wait_with_output().expect("wait clang");
            assert!(
                out.status.success(),
                "clang {:?} syntax/typecheck failed:\n{}",
                dialect,
                String::from_utf8_lossy(&out.stderr)
            );
            eprintln!(
                "clang_compiles_cuda_and_hip_kernels_for_every_codec: {clang_bin} verified {compiled_codecs} codecs for {dialect:?}"
            );
        }
    }
}

/// Whether the generated-kernel templates cover `op` at `dtype`. Nothing is covered until the
/// generated GEMV is wired into `GraphProgram::compile` (G3.2) and the movement/elementwise/reduction
/// templates exist (G4.1).
pub fn can_generate(_op: &crate::graph::Op, _dtype: crate::graph::DType) -> bool {
    false
}

/// Rows one threadgroup of [`gemv_packed`] covers: `simdgroups` × `rows_per_sg`.
pub const fn gemv_rows_per_tg(p: GemvParams) -> usize {
    p.simdgroups * p.rows_per_sg
}

/// Threads per threadgroup of [`gemv_packed`]: 32 per simdgroup.
pub const fn gemv_threads(p: GemvParams) -> usize {
    32 * p.simdgroups
}

/// The MSL a GPU consumer of the device weight buffer includes: [`Codec::msl`], or the planar-layout decoder
/// for a codec stored [`planar`].
pub fn device_msl(codec: &Codec) -> &'static str {
    planar(codec.format).map_or(codec.msl, |p| p.msl)
}

/// How MSL addresses block `idx` (a `ulong` expression) of weights `w`: a declaration of `b` (what `decode8`
/// takes as `blk`) and the prefixes to put before `sub` (resp. the wide group index): empty unless the codec is
/// stored [`planar`], where `b` is the block's group and the prefixes select the block within it.
pub struct BlockRef {
    pub decl: String,
    pub sub: String,
    pub group: String,
}

pub fn block_ref(codec: &Codec, w: &str, idx: &str) -> BlockRef {
    match planar(codec.format) {
        None => BlockRef {
            decl: format!(
                "device const uchar* b = {w} + ({idx}) * {};",
                codec.block_bytes
            ),
            sub: String::new(),
            group: String::new(),
        },
        Some(p) => {
            let groups = tile(codec.format)
                .map(|t| t.groups)
                .or_else(|| wide(codec.format).map(|w| w.groups))
                .unwrap_or(0);
            BlockRef {
                decl: format!(
                    "const ulong bi = {idx}; device const uchar* b = {w} + (bi / {}) * {};",
                    p.blocks,
                    p.blocks * codec.block_bytes
                ),
                sub: format!("uint(bi % {}) * {} + ", p.blocks, codec.block_elems / 8),
                group: format!("uint(bi % {}) * {groups} + ", p.blocks),
            }
        }
    }
}

/// MSL common to the hand kernels and generated GEMVs: GELU variants, `PackedGemvParams` and
/// `apply_gemv_epi`, taken from `metal_kernels.metal` between its `GEMV-COMMON` markers.
pub fn gemv_common(kernels_msl: &str) -> &str {
    let begin = kernels_msl
        .find("// GEMV-COMMON-BEGIN")
        .expect("GEMV-COMMON-BEGIN marker in metal_kernels.metal");
    let end = kernels_msl
        .find("// GEMV-COMMON-END")
        .expect("GEMV-COMMON-END marker in metal_kernels.metal");
    &kernels_msl[begin..end]
}

/// Name of the kernel [`gemv_packed`] emits for `codec` with `p` (`_sg2` suffix off the 8-simdgroup default).
pub fn gemv_packed_name(codec: &Codec, p: GemvParams) -> String {
    let sg = if p.simdgroups == 8 {
        String::new()
    } else {
        format!("_sg{}", p.simdgroups)
    };
    format!("gemvgen_{:?}_r{}{sg}", codec.format, p.rows_per_sg)
}

/// Name of the kernel [`dequant_packed`] emits for `codec`.
pub fn dequant_packed_name(codec: &Codec) -> String {
    format!("dequantgen_{:?}", codec.format)
}

/// Expands a packed `[rows, cols]` weight to f32: buffers `w, out`, bytes `(rows, cols)`, one thread
/// per 8-element sub-block (`ceil(rows * cols / 8)` threads). Prefill GEMMs read the result.
pub fn dequant_packed(codec: &Codec) -> String {
    let name = dequant_packed_name(codec);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let (be, subs) = (codec.block_elems, codec.block_elems / 8);
    let br = block_ref(
        codec,
        "w",
        &format!("(ulong)row * (dims.y / {be}) + s / {subs}"),
    );
    let (decl, sub) = (&br.decl, &br.sub);
    format!(
        r#"
{codec_msl}

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint per_row = dims.y / 8;
    if (gid >= dims.x * per_row) return;
    const uint row = gid / per_row, s = gid % per_row;
    {decl}
    float v[8];
    {dec}(b, {sub}s % {subs}, v);
    device float* o = out + (ulong)row * dims.y + s * 8;
    for (uint i = 0; i < 8; ++i) o[i] = v[i];
}}
"#
    )
}

/// A packed-weight GEMV with the `Step::GemvPacked` contract: buffers `w, x, y, PackedGemvParams,
/// bias, res`; grid `(ceil(rows / gemv_rows_per_tg), batch)` of `gemv_threads` threads. Lane `l` of a
/// simdgroup walks each row's 8-element sub-blocks `l, l+32, …`, decodes them with the codec's
/// `decode8`, and the simdgroup sums the row. Bias and the fused epilogue are applied by lane 0.
pub fn gemv_packed(codec: &Codec, p: GemvParams) -> String {
    let name = gemv_packed_name(codec, p);
    // Every codec defines `decode8`; one library holds several codecs, so each gets its own name.
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let r = p.rows_per_sg;
    let be = codec.block_elems;
    let loop_src = gemv_loop(codec, p, &dec, "(ulong)(row0", true);
    let store_row = |acc: &str, row: &str, ind: &str| {
        format!(
            "{ind}float a = {acc};\n{ind}if (p.has_bias != 0) a += bias[{row}];\n{ind}ulong o = (ulong)batch * p.rows + {row};\n{ind}y[o] = apply_gemv_epi(a, o, p, res);\n"
        )
    };
    // A tile codec splits the simdgroup's lanes across its `r` rows (`32 / r` lanes per row, summed by
    // xor-shuffles): its row has few 16-byte tiles, so letting every lane serve one row keeps all of them busy.
    let (acc_decl, sums, guard, store) = if tile(codec.format).is_some() {
        let lpr = 32 / r;
        let shuffles: String = std::iter::successors(Some(lpr / 2), |o| Some(o / 2))
            .take_while(|&o| o > 0)
            .map(|o| format!("    acc0 += simd_shuffle_xor(acc0, {o});\n"))
            .collect();
        (
            format!("const uint sl = lane % {lpr}, rl = lane / {lpr}, myrow = row0 + rl;\n    float acc0 = 0.0f;"),
            shuffles,
            format!("sl == 0 && myrow < p.rows"),
            store_row("acc0", "myrow", "        "),
        )
    } else {
        let store: String = (0..r)
            .map(|i| {
                format!(
                    "        if (row0 + {i} < p.rows) {{\n{}        }}\n",
                    store_row(
                        &format!("simd_sum_{i}"),
                        &format!("row0 + {i}"),
                        "            "
                    )
                )
            })
            .collect();
        (
            (0..r)
                .map(|i| format!("float acc{i} = 0.0f;"))
                .collect::<Vec<_>>()
                .join(" "),
            (0..r)
                .map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n"))
                .collect(),
            "lane == 0".to_string(),
            store,
        )
    };
    format!(
        r#"
{codec_msl}
{wide_msl}
kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant PackedGemvParams& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device const float* res [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    const uint blocks_per_row = p.cols / {be};
    device const float* xb = x + (ulong)batch * p.cols;
    {acc_decl}
{loop_src}{sums}    if ({guard}) {{
{store}    }}
}}
"#,
        rows_per_tg = gemv_rows_per_tg(p),
        wide_msl = wide_msl(codec, &dec, true),
    )
}

/// Function-constant indices a [`gemv_packed_specializable`] pipeline may be specialized to: `rows` and blocks
/// per row (`cols / block_elems`).
pub const FC_ROWS: u64 = 0;
pub const FC_BLOCKS_PER_ROW: u64 = 1;

/// [`gemv_packed`] reading `rows` and `cols` from function constants [`FC_ROWS`] / [`FC_BLOCKS_PER_ROW`] when
/// defined (else from `PackedGemvParams`), so loop trip counts and row guards fold at pipeline compile time.
/// Separate from [`gemv_packed`] so the default (unspecialized) source stays unchanged.
pub fn gemv_packed_specializable(codec: &Codec, p: GemvParams) -> String {
    let src = gemv_packed(codec, p);
    let (head, kernel) = src.split_once("kernel void ").expect("gemv_packed emits a kernel");
    let kernel = kernel.replace("p.rows", "n_rows").replace("p.cols", "n_cols");
    let (sig, body) = kernel.split_once(") {\n").expect("kernel body");
    format!(
        "{head}constant uint fc_rows [[function_constant({FC_ROWS})]];\nconstant uint fc_bpr [[function_constant({FC_BLOCKS_PER_ROW})]];\nconstant bool fc_has_rows = is_function_constant_defined(fc_rows);\nconstant bool fc_has_bpr = is_function_constant_defined(fc_bpr);\nkernel void {sig}) {{\n    const uint n_rows = fc_has_rows ? fc_rows : p.rows;\n    const uint n_cols = fc_has_bpr ? fc_bpr * {be} : p.cols;\n{body}",
        be = codec.block_elems
    )
}

fn wide_msl(codec: &Codec, dec: &str, allow_tile: bool) -> String {
    wide(codec.format)
        .map(|w| {
            w.msl
                .replace("decode_group", &dec.replace("decode8", "decode_group"))
                .replace("dot_group", &dec.replace("decode8", "dot_group"))
                .replace("group_seg", &dec.replace("decode8", "group_seg"))
        })
        .or_else(|| {
            allow_tile.then(|| tile(codec.format)).flatten().map(|t| {
                ["group_seg", "tile_load", "tile_scale", "tile_seg"]
                    .iter()
                    .fold(t.msl.to_string(), |m, n| {
                        m.replace(n, &dec.replace("decode8", n))
                    })
            })
        })
        .unwrap_or_default()
}

/// The lane loop of [`gemv_packed`] / [`gemv_indexed`]: one 8-element sub-block per iteration (`decode8`), or,
/// for a codec with a [`wide`] decode, one group of `segs` sub-blocks (its bytes are read once, not once per sub-block).
fn gemv_loop(codec: &Codec, p: GemvParams, dec: &str, row_base: &str, allow_tile: bool) -> String {
    let (be, r) = (codec.block_elems, p.rows_per_sg);
    // `decode` gets the block's prefixes (`BlockRef::sub` for decode8, `group` for decode_group).
    let row = |i: usize, decode: &dyn Fn(&BlockRef) -> String, dots: &str, v_len: u32| {
        let br = block_ref(
            codec,
            "w",
            &format!("{row_base} + {i}) * blocks_per_row + blk"),
        );
        format!(
            "        if (row0 + {i} < p.rows) {{\n            {}\n            float v[{v_len}];\n            {}\n{dots}        }}\n",
            br.decl,
            decode(&br)
        )
    };
    if allow_tile {
        if let Some(t) = tile(codec.format) {
            return tile_loop(codec, t, p, dec);
        }
    }
    let Some(wd) = wide(codec.format) else {
        let rows: String = (0..r)
            .map(|i| {
                row(
                    i,
                    &|br| format!("{dec}(b, {}sub, v);", br.sub),
                    &format!("            acc{i} += dot(float4(v[0], v[1], v[2], v[3]), x0) + dot(float4(v[4], v[5], v[6], v[7]), x1);\n"),
                    8,
                )
            })
            .collect();
        return format!(
            "    const uint subs = p.cols / 8;\n    for (uint s = lane; s < subs; s += 32) {{\n        float4 x0 = *(device const float4*)(xb + s * 8);\n        float4 x1 = *(device const float4*)(xb + s * 8 + 4);\n        const uint blk = (s * 8) / {be};\n        const uint sub = ((s * 8) % {be}) / 8;\n{rows}    }}\n"
        );
    };
    let (gdec, gseg) = (
        dec.replace("decode8", "decode_group"),
        dec.replace("decode8", "group_seg"),
    );
    let xs: String = (0..wd.segs)
        .map(|k| {
            format!(
                "        {{ const uint e = blk * {be} + {gseg}(g, {k}); xs[{}] = *(device const float4*)(xb + e); xs[{}] = *(device const float4*)(xb + e + 4); }}\n",
                2 * k,
                2 * k + 1
            )
        })
        .collect();
    let groups = wd.groups;
    if wd.fused_dot {
        let gdot = dec.replace("decode8", "dot_group");
        let rows: String = (0..r)
            .map(|i| {
                let br = block_ref(
                    codec,
                    "w",
                    &format!("{row_base} + {i}) * blocks_per_row + blk"),
                );
                format!(
                    "        if (row0 + {i} < p.rows) {{\n            {}\n            acc{i} += {gdot}(b, {}g, xs);\n        }}\n",
                    br.decl, br.group
                )
            })
            .collect();
        return format!(
            "    const uint groups = blocks_per_row * {groups};\n    for (uint s = lane; s < groups; s += 32) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        float4 xs[{}];\n{xs}{rows}    }}\n",
            2 * wd.segs
        );
    }
    let nseg = wd.segs as usize;
    let rows: String = (0..r)
        .map(|i| {
            let seg_dot = |k: usize| {
                let o = 8 * k;
                format!(
                    "dot(float4(v[{o}], v[{}], v[{}], v[{}]), xs[{}]) + dot(float4(v[{}], v[{}], v[{}], v[{}]), xs[{}])",
                    o + 1, o + 2, o + 3, 2 * k, o + 4, o + 5, o + 6, o + 7, 2 * k + 1
                )
            };
            let mut dots = String::new();
            if wd.uniform {
                let sum = (0..nseg).map(|k| format!("({})", seg_dot(k))).collect::<Vec<_>>().join(" + ");
                dots += &format!("            acc{i} += v[{}] * ({sum});\n", 8 * nseg);
            } else {
                for k in 0..nseg {
                    dots += &format!(
                        "            acc{i} += v[{}] * ({}) + v[{}] * sx{k};\n",
                        8 * nseg + k, seg_dot(k), 9 * nseg + k
                    );
                }
            }
            let v_len = if wd.uniform { 8 * wd.segs + 1 } else { 10 * wd.segs };
            row(i, &|br| format!("{gdec}(b, {}g, v);", br.group), &dots, v_len)
        })
        .collect();
    let sx: String = if wd.uniform || wd.fused_dot {
        String::new()
    } else {
        (0..nseg)
            .map(|k| {
                format!(
                    "        const float sx{k} = dot(xs[{}], float4(1.0f)) + dot(xs[{}], float4(1.0f));\n",
                    2 * k,
                    2 * k + 1
                )
            })
            .collect()
    };
    format!(
        "    const uint groups = blocks_per_row * {groups};\n    for (uint s = lane; s < groups; s += 32) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        float4 xs[{}];\n{xs}{sx}{rows}    }}\n",
        2 * wd.segs
    )
}

/// The lane loop of [`gemv_packed`] for a [`Tile`] codec: lane `l` serves row `myrow` (`32 / rows_per_sg`
/// lanes per row), walks that row's tiles `sl, sl + lpr, …`, loads the tile and its scale, then decodes one
/// 8-element segment at a time against `x` and accumulates `scale * (dot(codes, x) - sum(x))`.
fn tile_loop(codec: &Codec, t: &Tile, p: GemvParams, dec: &str) -> String {
    let (be, r, segs) = (codec.block_elems, p.rows_per_sg, t.segs);
    let lpr = 32 / r;
    assert!(
        segs as usize % r == 0,
        "{} rows per simdgroup do not divide {segs} segments",
        r
    );
    let name = |n: &str| dec.replace("decode8", n);
    let (seg, load, scale, tseg) = (
        name("group_seg"),
        name("tile_load"),
        name("tile_scale"),
        name("tile_seg"),
    );
    let br = block_ref(codec, "w", "(ulong)myrow * blocks_per_row + blk");
    let body: String = (0..segs)
        .map(|k| {
            format!(
                "        {{\n            const uint e = blk * {be} + {seg}(g, {k});\n            const float4 xa = *(device const float4*)(xb + e), xc = *(device const float4*)(xb + e + 4);\n            float v[8];\n            {tseg}(q, {k}, v);\n            t += dot(float4(v[0], v[1], v[2], v[3]), xa) + dot(float4(v[4], v[5], v[6], v[7]), xc);\n        }}\n"
            )
        })
        .collect();
    let groups = t.groups;
    if t.b == 0.0 {
        return format!(
            "    const uint tiles = blocks_per_row * {groups};\n    for (uint s = sl; s < tiles; s += {lpr}) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        uint4 q = uint4(0);\n        float d = 0.0f, t = 0.0f;\n        if (myrow < p.rows) {{\n            {}\n            q = {load}(b, {}g);\n            d = {scale}(b, {}g);\n        }}\n{body}        acc0 += d * ({a:?}f * t);\n    }}\n",
            br.decl, br.group, br.group, a = t.a
        );
    }
    // The lanes serving the same tile of different rows (`lane % lpr` equal) split sum(x) over the tile
    // between them (`segs / r` segments each) and add the parts with xor-shuffles.
    let spr = segs as usize / r;
    let sx_parts: String = (0..spr)
        .map(|m| {
            format!(
                "        {{\n            const uint e = blk * {be} + {seg}(g, rl * {spr} + {m});\n            sx += *(device const float4*)(xb + e) + *(device const float4*)(xb + e + 4);\n        }}\n"
            )
        })
        .collect();
    let sx_sum: String = std::iter::successors(Some(lpr), |o| Some(o * 2))
        .take_while(|&o| o < 32)
        .map(|o| format!("        sxs += simd_shuffle_xor(sxs, {o});\n"))
        .collect();
    format!(
        "    const uint tiles = blocks_per_row * {groups};\n    for (uint s = sl; s < tiles; s += {lpr}) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        uint4 q = uint4(0);\n        float d = 0.0f, t = 0.0f;\n        if (myrow < p.rows) {{\n            {}\n            q = {load}(b, {}g);\n            d = {scale}(b, {}g);\n        }}\n        float4 sx = float4(0.0f);\n{sx_parts}        float sxs = dot(sx, float4(1.0f));\n{sx_sum}{body}        acc0 += d * ({a:?}f * t - {b:?}f * sxs);\n    }}\n",
        br.decl, br.group, br.group, a = t.a, b = t.b
    )
}

/// Name of the kernel [`gemv_indexed`] emits for `codec` with `p`.
pub fn gemv_indexed_name(codec: &Codec, p: GemvParams) -> String {
    format!("gemvgen_{:?}_r{}_ix", codec.format, p.rows_per_sg)
}

/// [`gemv_packed`] over an expert bank: the weight is `experts` stacked `[rows, cols]` matrices and
/// slot `s` (grid y) multiplies row `s` of `x` (`x_per_slot != 0`) or row `s / top_k` by the matrix
/// `ids[s]`, writing row `s` of `y` (`[slots, rows]`). Buffers `w, x, y, MoeGemvParams {rows, cols,
/// slots, top_k, experts, x_per_slot}, ids`; grid `(ceil(rows / gemv_rows_per_tg), slots)` of 256
/// threads. A slot whose id is not below `experts` writes zeros.
pub fn gemv_indexed(codec: &Codec, p: GemvParams) -> String {
    gemv_indexed_for_dialect(Dialect::Msl, codec, p)
}

/// Where an IQ2_XXS [`gemv_indexed`] reads its 2 KB grid: the codec's program-scope `constant ulong`
/// array (default), a per-threadgroup `uint` copy staged once behind one barrier, or a `constant`
/// buffer bound at slot 5 (the host binds `IQ2XXS_GRID`'s bytes there).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Iq2Grid {
    Program,
    Threadgroup,
    Buffer,
}

/// Rewrites the IQ2_XXS [`gemv_indexed`] source so `dot_group` reads the grid through `grid`: a
/// threadgroup `uint[512]` copy the kernel's 256 threads stage (2 stores each, one barrier) before
/// any early return, or a reference to a sized `constant` table at slot 5 (statically sized, so the
/// driver may preload it, which an unbounded `constant` pointer forbids). `Err` when the source lacks an anchor.
pub fn iq2_grid_msl(src: String, mode: Iq2Grid) -> Result<String, String> {
    const IDX: &str = "((q4 >> (8 * k)) & 0xFFu)";
    let (decl, param, lookup, kparam, stage) = match mode {
        Iq2Grid::Program => return Ok(src),
        Iq2Grid::Threadgroup => (
            "",
            "threadgroup const uint* grid",
            format!("uint2(grid[2u * {IDX}], grid[2u * {IDX} + 1u])"),
            "",
            "    threadgroup uint grid[512];\n    {\n        const uint t = sg * 32 + lane;\n        const uint2 e = as_type<uint2>(IQ2XXS_GRID[t]);\n        grid[2 * t] = e.x;\n        grid[2 * t + 1] = e.y;\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        ),
        Iq2Grid::Buffer => (
            "struct Iq2GridTable { uint2 g[256]; };\n",
            "constant Iq2GridTable& grid",
            format!("grid.g[{IDX}]"),
            "    constant Iq2GridTable& grid [[buffer(5)]],\n",
            "",
        ),
    };
    let ids_param = "    device const uint* ids [[buffer(4)]],\n";
    let body = "    const uint row0 = tg.x * ";
    let edits = [
        ("as_type<uint2>(IQ2XXS_GRID[(q4 >> (8 * k)) & 0xFFu])".to_string(), lookup),
        ("thread const float4* xs) {".to_string(), format!("thread const float4* xs, {param}) {{")),
        ("(b, g, xs);".to_string(), "(b, g, xs, grid);".to_string()),
        (ids_param.to_string(), format!("{ids_param}{kparam}")),
        (body.to_string(), format!("{stage}{body}")),
    ];
    let src = edits.into_iter().try_fold(src, |s, (from, to)| {
        if s.contains(&from) {
            Ok(s.replace(&from, &to))
        } else {
            Err(format!("IQ2_XXS grid {mode:?}: generated GEMV has no `{from}`"))
        }
    })?;
    Ok(format!("{decl}{src}"))
}

/// Dialect-parameterized [`gemv_indexed`] kernel across `Dialect::Msl`, `Dialect::Cuda`, and `Dialect::Hip`.
pub fn gemv_indexed_for_dialect(dialect: Dialect, codec: &Codec, p: GemvParams) -> String {
    if dialect == Dialect::Msl {
        let name = gemv_indexed_name(codec, p);
        let dec = format!("decode8_{:?}", codec.format);
        let codec_msl = device_msl(codec).replace("decode8", &dec);
        let r = p.rows_per_sg;
        let be = codec.block_elems;
        let acc_decl = (0..r)
            .map(|i| format!("float acc{i} = 0.0f;"))
            .collect::<Vec<_>>()
            .join(" ");
        let loop_src = gemv_loop(codec, p, &dec, "(bank_row + row0", false);
        let sums: String = (0..r)
            .map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n"))
            .collect();
        let store: String = (0..r)
            .map(|i| {
                format!("        if (row0 + {i} < p.rows) y[(ulong)slot * p.rows + row0 + {i}] = simd_sum_{i};\n")
            })
            .collect();
        return format!(
            r#"
{codec_msl}
{wide_msl}

struct MoeGemvParams {{
    uint rows;
    uint cols;
    uint slots;
    uint top_k;
    uint experts;
    uint x_per_slot;
}};

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvParams& p [[buffer(3)]],
    device const uint* ids [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint slot = tg.y;
    if (row0 >= p.rows || slot >= p.slots) return;
    const uint expert = ids[slot];
    if (expert >= p.experts) {{
        if (lane == 0) {{
            for (uint i = 0; i < {r}; i++) if (row0 + i < p.rows) y[(ulong)slot * p.rows + row0 + i] = 0.0f;
        }}
        return;
    }}
    const ulong bank_row = (ulong)expert * p.rows;
    const uint blocks_per_row = p.cols / {be};
    device const float* xb = x + (ulong)(p.x_per_slot != 0 ? slot : slot / p.top_k) * p.cols;
    {acc_decl}
{loop_src}{sums}    if (lane == 0) {{
{store}    }}
}}
"#,
            rows_per_tg = gemv_rows_per_tg(p),
            wide_msl = wide_msl(codec, &dec, false),
        );
    }

    let simd_width = dialect.simd_width(None).max(1);
    let name = gemv_indexed_name(codec, p);
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let preamble = dialect_preamble(dialect);
    let common_c = gemv_common_for_dialect(dialect);
    let codec_c = translate_msl_to_c_dialect(codec.msl, dialect);
    let r = p.rows_per_sg.max(1);
    let rows_per_tg = gemv_rows_per_tg(p);
    let be = codec.block_elems;
    let bb = codec.block_bytes;
    let reduction = warp_reduce_sum("sum", dialect, simd_width, "        ");

    format!(
        r#"
// Generated Indexed GEMV ({dialect:?}) for {format_str}
{preamble}
{common_c}
#undef decode8
#define decode8 decode8_{format_str}
#ifndef GIBBS_DEC8_{format_str}
#define GIBBS_DEC8_{format_str}
{codec_c}
#endif

#ifdef GIBBS_KERN_{name}
#undef {name}
#define {name} GIBBS_CAT({name}_v, __COUNTER__)
#endif
#define GIBBS_KERN_{name}
extern "C" __global__ void {name}(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, unsigned int p_rows, unsigned int p_cols, unsigned int p_slots, unsigned int p_top_k, unsigned int p_experts, unsigned int p_x_per_slot, const unsigned int* __restrict__ ids) {{
    unsigned int tg = blockIdx.x;
    unsigned int slot = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int sg = tid / {simd_width};
    unsigned int lane = tid % {simd_width};
    unsigned int row0 = tg * {rows_per_tg} + sg * {r};
    if (row0 >= p_rows || slot >= p_slots) return;
    unsigned int expert = ids[slot];
    if (expert >= p_experts) {{
        if (lane == 0) {{
            for (unsigned int i = 0; i < {r}; i++) {{
                if (row0 + i < p_rows) y[(unsigned long long)slot * p_rows + row0 + i] = 0.0f;
            }}
        }}
        return;
    }}
    unsigned long long bank_row = (unsigned long long)expert * p_rows;
    unsigned int blocks_per_row = p_cols / {be};
    const float* xb = x + (unsigned long long)(p_x_per_slot != 0 ? slot : slot / p_top_k) * p_cols;
    float acc[{r}];
    for (unsigned int i = 0; i < {r}; i++) acc[i] = 0.0f;
    unsigned int num_sub = p_cols / 8;
    for (unsigned int s = lane; s < num_sub; s += {simd_width}) {{
        unsigned int blk_idx = (s * 8) / {be};
        unsigned int sub_in_blk = ((s * 8) % {be}) / 8;
        for (unsigned int r_i = 0; r_i < {r}; r_i++) {{
            if (row0 + r_i < p_rows) {{
                float dec8[8];
                const unsigned char* blk = w + ((bank_row + row0 + r_i) * blocks_per_row + blk_idx) * {bb};
                decode8(blk, sub_in_blk, dec8);
                for (unsigned int i = 0; i < 8; i++) {{
                    acc[r_i] += dec8[i] * xb[s * 8 + i];
                }}
            }}
        }}
    }}
    for (unsigned int r_i = 0; r_i < {r}; r_i++) {{
        float sum = acc[r_i];
{reduction}        if (lane == 0 && row0 + r_i < p_rows) {{
            y[(unsigned long long)slot * p_rows + row0 + r_i] = sum;
        }}
    }}
}}
"#,
        dialect = dialect,
        format_str = format_str,
        preamble = preamble,
        common_c = common_c,
        codec_c = codec_c,
        name = name,
        simd_width = simd_width,
        rows_per_tg = rows_per_tg,
        r = r,
        be = be,
        bb = bb,
        reduction = reduction,
    )
}
