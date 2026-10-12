use crate::hardware::QuantFormat;

pub struct Codec {
    pub format: QuantFormat,
    pub block_elems: u32,
    pub block_bytes: u32,
    /// MSL defining `inline void decode8(device const uchar* blk, uint sub, thread float* out8)`.
    pub msl: &'static str,
}

pub const CODEC_VERSION: u32 = 1;

pub fn codec(format: QuantFormat) -> Option<&'static Codec> {
    match format {
        QuantFormat::GGUF_Q4_0 => Some(&CODEC_Q4_0),
        QuantFormat::GGUF_Q8_0 => Some(&CODEC_Q8_0),
        QuantFormat::GGUF_Q4_K_M => Some(&CODEC_Q4_K),
        QuantFormat::GGUF_Q2_K => Some(&CODEC_Q2_K),
        QuantFormat::GGUF_IQ4_XS => Some(&CODEC_IQ4_XS),
        QuantFormat::GGUF_IQ2_M | QuantFormat::GGUF_IQ2_S => Some(&CODEC_IQ2_M),
        QuantFormat::GGUF_IQ3_S => Some(&CODEC_IQ3_S),
        QuantFormat::GGUF_TQ2_0 => Some(&CODEC_TQ2_0),
        _ => None,
    }
}

pub const CODEC_Q4_0: Codec = Codec {
    format: QuantFormat::GGUF_Q4_0,
    block_elems: 32,
    block_bytes: 18,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    device const uchar* q = blk + 2 + sub * 4;
    for (uint i = 0; i < 4; i++) {
        uchar v = q[i];
        out8[i] = float(d) * (float(v & 0x0F) - 8.0f);
        out8[i + 4] = float(d) * (float(v >> 4) - 8.0f);
    }
}
"#,
};

pub const CODEC_Q8_0: Codec = Codec {
    format: QuantFormat::GGUF_Q8_0,
    block_elems: 32,
    block_bytes: 34,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    device const char* q = (device const char*)(blk + 2) + sub * 8;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d) * float(q[i]);
    }
}
"#,
};

pub const CODEC_Q4_K: Codec = Codec {
    format: QuantFormat::GGUF_Q4_K_M,
    block_elems: 256,
    block_bytes: 144,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    half dmin = *(device const half*)(blk + 2);
    // Simplified decode8 for sub-block of 8 values
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d);
    }
}
"#,
};

pub const CODEC_Q2_K: Codec = Codec {
    format: QuantFormat::GGUF_Q2_K,
    block_elems: 256,
    block_bytes: 84,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d);
    }
}
"#,
};

pub const CODEC_IQ4_XS: Codec = Codec {
    format: QuantFormat::GGUF_IQ4_XS,
    block_elems: 256,
    block_bytes: 136,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d);
    }
}
"#,
};

pub const CODEC_IQ2_M: Codec = Codec {
    format: QuantFormat::GGUF_IQ2_M,
    block_elems: 256,
    block_bytes: 82,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d);
    }
}
"#,
};

pub const CODEC_IQ3_S: Codec = Codec {
    format: QuantFormat::GGUF_IQ3_S,
    block_elems: 256,
    block_bytes: 110,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d);
    }
}
"#,
};

pub const CODEC_TQ2_0: Codec = Codec {
    format: QuantFormat::GGUF_TQ2_0,
    block_elems: 256,
    block_bytes: 66,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d);
    }
}
"#,
};

pub fn msl_prelude() -> String {
    "// Codec MSL prelude\n".to_string()
}

use crate::model::weight_format::ReencodePolicy;
use std::collections::HashMap;

/// Result of codec micro-benchmarking on host GPU.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodecBenchProfile {
    pub device: String,
    pub src_hash: u64,
    pub eff: HashMap<String, f64>,
}

/// Predicts decode tokens per second given streamed tensor weights and re-encode policy.
pub fn predict_decode_tok_s(
    tensor_bytes: &[(QuantFormat, u64)],
    policy: ReencodePolicy,
    ceiling_gbps: f64,
    eff_map: Option<&HashMap<String, f64>>,
) -> f64 {
    if ceiling_gbps <= 0.0 || tensor_bytes.is_empty() {
        return 0.0;
    }

    let mut total_s = 0.0f64;
    for &(fmt, bytes) in tensor_bytes {
        let active_fmt = match policy {
            ReencodePolicy::MaxSpeed => {
                let bpw = fmt.bits_per_weight();
                if bpw > 4.5 {
                    QuantFormat::GGUF_Q4_K_M
                } else {
                    fmt
                }
            }
            ReencodePolicy::LowBitNative => fmt,
            ReencodePolicy::BitExact => fmt,
        };

        let fmt_key = format!("{active_fmt:?}");
        let eff = eff_map
            .and_then(|m| m.get(&fmt_key).copied())
            .unwrap_or(0.9);

        let eff_bw = (eff * ceiling_gbps * 1e9).max(1e6);
        total_s += bytes as f64 / eff_bw;
    }

    if total_s > 0.0 {
        1.0 / total_s
    } else {
        0.0
    }
}
