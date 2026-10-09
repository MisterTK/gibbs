use crate::hardware::QuantFormat;
use crate::model::{f16_to_f32, f32_to_bf16, f32_to_f16};
use crate::quant::exl3::{gemv_exl3_f32, quantize_exl3_tensor, Exl3BitRate, Exl3Tensor};
use crate::quant::jesse::WeightlessJesseLayer;
use crate::quant::nvfp4::{gemv_nvfp4_f32, quantize_nvfp4_tensor, Nvfp4Tensor};
use bytes::Bytes;

/// Quantized weight representation supporting uncompressed and accelerated compressed formats.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizedWeight {
    Uncompressed {
        data: Vec<f32>,
        in_features: usize,
        out_features: usize,
    },
    /// Zero-copy byte slice view into memory-mapped or shared model buffers.
    ZeroCopySlice {
        data: Bytes,
        in_features: usize,
        out_features: usize,
        format: QuantFormat,
    },
    /// Direct GGUF-quantized weight blocks (Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0).
    GgufQuantized {
        format: QuantFormat,
        raw_data: Bytes,
        in_features: usize,
        out_features: usize,
    },
    Nvfp4(Nvfp4Tensor),
    Exl3(Exl3Tensor),
    WeightlessJesse(WeightlessJesseLayer),
}

impl QuantizedWeight {
    /// Effective bits per weight element for this storage format.
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Uncompressed { .. } => 32.0, // Vec<f32>
            Self::ZeroCopySlice { format, .. } => format.bits_per_weight(),
            Self::GgufQuantized { format, .. } => format.bits_per_weight(),
            Self::Nvfp4(_) => 4.25,
            Self::Exl3(tensor) => match tensor.bit_rate {
                Exl3BitRate::Bits3 => 3.125,
                Exl3BitRate::Bits4 => 4.125,
                Exl3BitRate::Bits5 => 5.125,
            },
            Self::WeightlessJesse(layer) => layer.bits_per_weight(),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Uncompressed { in_features, .. }
            | Self::ZeroCopySlice { in_features, .. }
            | Self::GgufQuantized { in_features, .. } => *in_features,
            Self::Nvfp4(t) => t.cols,
            Self::Exl3(t) => t.cols,
            Self::WeightlessJesse(l) => l.in_features,
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Uncompressed { out_features, .. }
            | Self::ZeroCopySlice { out_features, .. }
            | Self::GgufQuantized { out_features, .. } => *out_features,
            Self::Nvfp4(t) => t.rows,
            Self::Exl3(t) => t.rows,
            Self::WeightlessJesse(l) => l.out_features,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.in_features() * self.out_features()
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::Uncompressed { data, .. } => data.clone(),
            Self::GgufQuantized {
                format, raw_data, ..
            } => dequantize_gguf_blocks(*format, raw_data).unwrap_or_default(),
            Self::ZeroCopySlice { format, data, .. } => {
                if let Some(v) = dequantize_gguf_blocks(*format, data) {
                    v
                } else if *format == QuantFormat::FP16 {
                    data.chunks_exact(2)
                        .map(|c| crate::model::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect()
                } else if *format == QuantFormat::BF16 {
                    data.chunks_exact(2)
                        .map(|c| crate::model::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect()
                } else {
                    vec![0.0f32; self.num_elements()]
                }
            }
            Self::Nvfp4(tensor) => {
                let mut out = vec![0.0f32; tensor.rows * tensor.cols];
                crate::quant::nvfp4::dequantize_nvfp4_tensor(tensor, &mut out);
                out
            }
            Self::Exl3(tensor) => {
                let mut out = vec![0.0f32; tensor.rows * tensor.cols];
                crate::quant::exl3::dequantize_exl3_tensor(tensor, &mut out);
                out
            }
            Self::WeightlessJesse(layer) => {
                let in_f = layer.in_features;
                let out_f = layer.out_features;
                let mut out = vec![0.0f32; in_f * out_f];
                let mut eye = vec![0.0f32; in_f];
                let mut col_out = vec![0.0f32; out_f];
                for c in 0..in_f {
                    eye[c] = 1.0;
                    layer.forward(&eye, &mut col_out);
                    for r in 0..out_f {
                        out[r * in_f + c] = col_out[r];
                    }
                    eye[c] = 0.0;
                }
                out
            }
        }
    }
}

/// Linear layer supporting zero-copy and compressed forward GEMV operations.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedLinearLayer {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub weight: QuantizedWeight,
    pub bias: Option<Vec<f32>>,
}

impl QuantizedLinearLayer {
    /// Creates a quantized linear layer from an uncompressed f32 weight buffer (row-major [out_features, in_features]).
    pub fn from_f32(
        name: &str,
        in_features: usize,
        out_features: usize,
        weights: &[f32],
        format: QuantFormat,
    ) -> Result<Self, String> {
        if weights.len() != in_features * out_features {
            return Err(format!(
                "Weight size mismatch for layer '{}': expected {} ({}x{}), got {}",
                name,
                in_features * out_features,
                out_features,
                in_features,
                weights.len()
            ));
        }

        let weight = match format {
            QuantFormat::NVFP4_E2M1 | QuantFormat::NVFP4 => {
                let tensor = quantize_nvfp4_tensor(weights, out_features, in_features)?;
                QuantizedWeight::Nvfp4(tensor)
            }
            QuantFormat::EXL3_3Bit => {
                let tensor =
                    quantize_exl3_tensor(weights, out_features, in_features, Exl3BitRate::Bits3)?;
                QuantizedWeight::Exl3(tensor)
            }
            QuantFormat::EXL3_4Bit => {
                let tensor =
                    quantize_exl3_tensor(weights, out_features, in_features, Exl3BitRate::Bits4)?;
                QuantizedWeight::Exl3(tensor)
            }
            QuantFormat::EXL3_5Bit => {
                let tensor =
                    quantize_exl3_tensor(weights, out_features, in_features, Exl3BitRate::Bits5)?;
                QuantizedWeight::Exl3(tensor)
            }
            QuantFormat::WeightlessJesseLUT => {
                let jesse = WeightlessJesseLayer::from_f32_weights(
                    name,
                    in_features,
                    out_features,
                    6,
                    weights,
                );
                QuantizedWeight::WeightlessJesse(jesse)
            }
            QuantFormat::GGUF_Q4_0 => {
                let raw = quantize_f32_to_q4_0(weights);
                QuantizedWeight::GgufQuantized {
                    format,
                    raw_data: Bytes::from(raw),
                    in_features,
                    out_features,
                }
            }
            QuantFormat::GGUF_Q8_0 => {
                let raw = quantize_f32_to_q8_0(weights);
                QuantizedWeight::GgufQuantized {
                    format,
                    raw_data: Bytes::from(raw),
                    in_features,
                    out_features,
                }
            }
            QuantFormat::FP16 | QuantFormat::BF16 => {
                let encode = if format == QuantFormat::FP16 {
                    f32_to_f16
                } else {
                    f32_to_bf16
                };
                let raw: Vec<u8> = weights
                    .iter()
                    .flat_map(|&w| encode(w).to_le_bytes())
                    .collect();
                QuantizedWeight::ZeroCopySlice {
                    data: Bytes::from(raw),
                    in_features,
                    out_features,
                    format,
                }
            }
            _ => QuantizedWeight::Uncompressed {
                data: weights.to_vec(),
                in_features,
                out_features,
            },
        };

        Ok(Self {
            name: name.to_string(),
            in_features,
            out_features,
            weight,
            bias: None,
        })
    }

    /// Attach an optional bias vector to the layer.
    pub fn with_bias(mut self, bias: Vec<f32>) -> Result<Self, String> {
        if bias.len() != self.out_features {
            return Err(format!(
                "Bias length {} does not match out_features {}",
                bias.len(),
                self.out_features
            ));
        }
        self.bias = Some(bias);
        Ok(self)
    }

    /// Creates a quantized linear layer backed by a SolidSF Jesse Weightless Neural Network.
    pub fn from_weightless_jesse(
        name: &str,
        in_features: usize,
        out_features: usize,
        address_bits: usize,
        weights: &[f32],
    ) -> Self {
        let layer = WeightlessJesseLayer::from_f32_weights(
            name,
            in_features,
            out_features,
            address_bits,
            weights,
        );
        Self {
            name: name.to_string(),
            in_features,
            out_features,
            weight: QuantizedWeight::WeightlessJesse(layer),
            bias: None,
        }
    }
    /// Creates a quantized linear layer directly from a zero-copy byte slice view.
    pub fn from_zero_copy_slice(
        name: &str,
        in_features: usize,
        out_features: usize,
        data: Bytes,
        format: QuantFormat,
    ) -> Self {
        Self {
            name: name.to_string(),
            in_features,
            out_features,
            weight: QuantizedWeight::ZeroCopySlice {
                data,
                in_features,
                out_features,
                format,
            },
            bias: None,
        }
    }

    /// Creates a quantized linear layer from an arbitrary QuantizedWeight variant.
    pub fn from_quantized_weight(
        name: &str,
        in_features: usize,
        out_features: usize,
        weight: QuantizedWeight,
    ) -> Self {
        Self {
            name: name.to_string(),
            in_features,
            out_features,
            weight,
            bias: None,
        }
    }

    /// Returns the underlying QuantFormat for the layer.
    pub fn quant_format(&self) -> QuantFormat {
        match &self.weight {
            QuantizedWeight::Uncompressed { .. } => QuantFormat::FP32,
            QuantizedWeight::ZeroCopySlice { format, .. } => *format,
            QuantizedWeight::GgufQuantized { format, .. } => *format,
            QuantizedWeight::Nvfp4(_) => QuantFormat::NVFP4_E2M1,
            QuantizedWeight::Exl3(t) => match t.bit_rate {
                Exl3BitRate::Bits3 => QuantFormat::EXL3_3Bit,
                Exl3BitRate::Bits4 => QuantFormat::EXL3_4Bit,
                Exl3BitRate::Bits5 => QuantFormat::EXL3_5Bit,
            },
            QuantizedWeight::WeightlessJesse(_) => QuantFormat::WeightlessJesseLUT,
        }
    }

    /// Evaluates forward GEMV: out = weight * x + bias.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            self.in_features,
            "Input length {} != in_features {} for layer '{}'",
            x.len(),
            self.in_features,
            self.name
        );
        assert_eq!(
            out.len(),
            self.out_features,
            "Output length {} != out_features {} for layer '{}'",
            out.len(),
            self.out_features,
            self.name
        );

        match &self.weight {
            QuantizedWeight::Uncompressed {
                data,
                in_features,
                out_features,
            } => {
                let inf = *in_features;
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .min(8);

                if *out_features >= 2048 && num_threads > 1 {
                    let raw_chunk = (*out_features + num_threads - 1) / num_threads;
                    let chunk_size = ((raw_chunk + 7) / 8) * 8;
                    std::thread::scope(|s| {
                        let bias_slice = self.bias.as_deref();
                        for (t_idx, out_chunk) in out.chunks_mut(chunk_size).enumerate() {
                            let start_row = t_idx * chunk_size;
                            let row_data =
                                &data[start_row * inf..(start_row + out_chunk.len()) * inf];
                            let b_chunk =
                                bias_slice.map(|b| &b[start_row..start_row + out_chunk.len()]);
                            s.spawn(move || {
                                gemv_rows_uncompressed(row_data, inf, x, out_chunk, b_chunk);
                            });
                        }
                    });
                } else {
                    gemv_rows_uncompressed(data, inf, x, out, self.bias.as_deref());
                }
            }
            QuantizedWeight::ZeroCopySlice {
                data,
                in_features,
                out_features,
                format,
            } => {
                dispatch_fused_gemv(
                    data,
                    *format,
                    *in_features,
                    *out_features,
                    x,
                    out,
                    self.bias.as_deref(),
                );
            }
            QuantizedWeight::GgufQuantized {
                format,
                raw_data,
                in_features,
                out_features,
            } => {
                dispatch_fused_gemv(
                    raw_data,
                    *format,
                    *in_features,
                    *out_features,
                    x,
                    out,
                    self.bias.as_deref(),
                );
            }
            QuantizedWeight::Nvfp4(tensor) => {
                gemv_nvfp4_f32(tensor, x, out);
                if let Some(bias) = &self.bias {
                    for i in 0..self.out_features {
                        out[i] += bias[i];
                    }
                }
            }
            QuantizedWeight::Exl3(tensor) => {
                gemv_exl3_f32(tensor, x, out);
                if let Some(bias) = &self.bias {
                    for i in 0..self.out_features {
                        out[i] += bias[i];
                    }
                }
            }
            QuantizedWeight::WeightlessJesse(layer) => {
                layer.forward(x, out);
                if let Some(bias) = &self.bias {
                    for i in 0..self.out_features {
                        out[i] += bias[i];
                    }
                }
            }
        }
    }

    /// Effective storage bits per weight element.
    #[inline]
    pub fn bits_per_weight(&self) -> f32 {
        self.weight.bits_per_weight()
    }
}
/// Dispatches fused dequantization and GEMV across any quantized or uncompressed weight format.
pub fn dispatch_fused_gemv(
    raw_bytes: &[u8],
    format: QuantFormat,
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    if let Some((n, bytes, dequant)) = gguf_block_codec(format) {
        return fused_gemv_gguf_blocks(
            raw_bytes,
            n,
            bytes,
            dequant,
            in_features,
            out_features,
            x,
            out,
            bias,
        );
    }
    match format {
        QuantFormat::FP32 => {
            fused_gemv_f32_slice(raw_bytes, in_features, out_features, x, out, bias);
        }
        QuantFormat::FP16 => {
            fused_gemv_f16(raw_bytes, in_features, out_features, x, out, bias);
        }
        QuantFormat::BF16 => {
            fused_gemv_bf16(raw_bytes, in_features, out_features, x, out, bias);
        }
        // Only codec formats, Q4_0/Q8_0 re-quantized weights, and F32/F16/BF16 slices are built
        // as raw-byte weights; anything else is a construction bug, not a format to guess at.
        other => unreachable!("no fused GEMV for {other:?} raw weights"),
    }
}

/// Fused GEMV for uncompressed or sliced f32 byte buffers.
pub fn fused_gemv_f32_slice(
    raw_bytes: &[u8],
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    if raw_bytes.len() < in_features * out_features * 4 {
        return;
    }
    if raw_bytes.as_ptr() as usize % 4 == 0 {
        let f32_slice = unsafe {
            std::slice::from_raw_parts(raw_bytes.as_ptr() as *const f32, in_features * out_features)
        };
        gemv_rows_uncompressed(f32_slice, in_features, x, out, bias);
    } else {
        for r in 0..out_features {
            let row = &raw_bytes[r * in_features * 4..(r + 1) * in_features * 4];
            let mut sum = bias.map(|b| b[r]).unwrap_or(0.0);
            for (c, chunk) in row.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                sum += v * x[c];
            }
            out[r] = sum;
        }
    }
}

/// Fused dequantization GEMV for IEEE 754 half-precision float16 bytes.
pub fn fused_gemv_f16(
    raw_bytes: &[u8],
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    let row_len = in_features * 2;
    if raw_bytes.len() < out_features * row_len {
        return;
    }
    for r in 0..out_features {
        let row = &raw_bytes[r * row_len..(r + 1) * row_len];
        let mut sum = bias.map(|b| b[r]).unwrap_or(0.0);
        let mut c = 0;
        while c + 4 <= in_features {
            let o = c * 2;
            let w0 = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
            let w1 = f16_to_f32(u16::from_le_bytes([row[o + 2], row[o + 3]]));
            let w2 = f16_to_f32(u16::from_le_bytes([row[o + 4], row[o + 5]]));
            let w3 = f16_to_f32(u16::from_le_bytes([row[o + 6], row[o + 7]]));
            sum += w0 * x[c] + w1 * x[c + 1] + w2 * x[c + 2] + w3 * x[c + 3];
            c += 4;
        }
        while c < in_features {
            let o = c * 2;
            let w = f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]));
            sum += w * x[c];
            c += 1;
        }
        out[r] = sum;
    }
}

/// Fused dequantization GEMV for bfloat16 bytes.
pub fn fused_gemv_bf16(
    raw_bytes: &[u8],
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    let row_len = in_features * 2;
    if raw_bytes.len() < out_features * row_len {
        return;
    }
    for r in 0..out_features {
        let row = &raw_bytes[r * row_len..(r + 1) * row_len];
        let mut sum = bias.map(|b| b[r]).unwrap_or(0.0);
        let mut c = 0;
        while c + 4 <= in_features {
            let o = c * 2;
            let w0 = f32::from_bits((u16::from_le_bytes([row[o], row[o + 1]]) as u32) << 16);
            let w1 = f32::from_bits((u16::from_le_bytes([row[o + 2], row[o + 3]]) as u32) << 16);
            let w2 = f32::from_bits((u16::from_le_bytes([row[o + 4], row[o + 5]]) as u32) << 16);
            let w3 = f32::from_bits((u16::from_le_bytes([row[o + 6], row[o + 7]]) as u32) << 16);
            sum += w0 * x[c] + w1 * x[c + 1] + w2 * x[c + 2] + w3 * x[c + 3];
            c += 4;
        }
        while c < in_features {
            let o = c * 2;
            let w = f32::from_bits((u16::from_le_bytes([row[o], row[o + 1]]) as u32) << 16);
            sum += w * x[c];
            c += 1;
        }
        out[r] = sum;
    }
}

/// Dequantizes one GGUF block into `y` in ggml `dequantize_row_*` element order.
pub type GgufBlockFn = fn(&[u8], &mut [f32]);

/// `(elements, bytes, dequantizer)` per block for every GGUF block format the engine decodes;
/// the f32 extraction path and the fused GEMV both go through these.
pub fn gguf_block_codec(format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    Some(match format {
        QuantFormat::GGUF_Q4_0 => (32, 18, dequantize_q4_0_block as GgufBlockFn),
        QuantFormat::GGUF_Q4_1 => (32, 20, dequantize_q4_1_block),
        QuantFormat::GGUF_Q5_0 => (32, 22, dequantize_q5_0_block),
        QuantFormat::GGUF_Q5_1 => (32, 24, dequantize_q5_1_block),
        QuantFormat::GGUF_Q8_0 => (32, 34, dequantize_q8_0_block),
        QuantFormat::GGUF_Q4_K_M => (256, 144, dequantize_q4_k_block),
        QuantFormat::GGUF_Q5_K_M => (256, 176, dequantize_q5_k_block),
        QuantFormat::GGUF_Q6_K => (256, 210, dequantize_q6_k_block),
        QuantFormat::GGUF_Q2_K => (256, 84, dequantize_q2_k_block),
        QuantFormat::GGUF_Q3_K => (256, 110, dequantize_q3_k_block),
        QuantFormat::GGUF_IQ4_NL => (32, 18, dequantize_iq4_nl_block),
        QuantFormat::GGUF_IQ4_XS => (256, 136, dequantize_iq4_xs_block),
        QuantFormat::GGUF_IQ3_XXS => (256, 98, dequantize_iq3_xxs_block),
        QuantFormat::GGUF_IQ3_S => (256, 110, dequantize_iq3_s_block),
        QuantFormat::GGUF_IQ2_XXS => (256, 66, dequantize_iq2_xxs_block),
        QuantFormat::GGUF_IQ2_XS => (256, 74, dequantize_iq2_xs_block),
        QuantFormat::GGUF_IQ2_S | QuantFormat::GGUF_IQ2_M => (256, 82, dequantize_iq2_s_block),
        QuantFormat::GGUF_IQ1_S => (256, 50, dequantize_iq1_s_block),
        QuantFormat::GGUF_IQ1_M => (256, 56, dequantize_iq1_m_block),
        QuantFormat::GGUF_TQ1_0 => (256, 54, dequantize_tq1_0_block),
        QuantFormat::GGUF_TQ2_0 => (256, 66, dequantize_tq2_0_block),
        QuantFormat::MXFP4 => (32, 17, dequantize_mxfp4_block),
        _ => return None,
    })
}

/// Dequantizes whole GGUF blocks of `format` to f32; `None` if `format` is not a block format.
pub fn dequantize_gguf_blocks(format: QuantFormat, raw: &[u8]) -> Option<Vec<f32>> {
    let (n, bytes, dequant) = gguf_block_codec(format)?;
    let mut out = vec![0.0f32; raw.len() / bytes * n];
    for (block, y) in raw.chunks_exact(bytes).zip(out.chunks_exact_mut(n)) {
        dequant(block, y);
    }
    Some(out)
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// Q4_0 (18 B / 32): `d`, then low nibbles = elements 0..16, high nibbles = 16..32.
pub fn dequantize_q4_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (i, &q) in b[2..18].iter().enumerate() {
        y[i] = ((q & 0x0F) as i32 - 8) as f32 * d;
        y[i + 16] = ((q >> 4) as i32 - 8) as f32 * d;
    }
}

/// Q4_1 (20 B / 32): `d`, `m`, then Q4_0 nibble order; value = q * d + m.
pub fn dequantize_q4_1_block(b: &[u8], y: &mut [f32]) {
    let (d, m) = (f16_at(b, 0), f16_at(b, 2));
    for (i, &q) in b[4..20].iter().enumerate() {
        y[i] = (q & 0x0F) as f32 * d + m;
        y[i + 16] = (q >> 4) as f32 * d + m;
    }
}

/// 5th bits of element `i` / `i + 16` from a Q5_0/Q5_1 `qh` word.
#[inline]
fn q5_high_bits(qh: u32, i: usize) -> (u8, u8) {
    (
        (((qh >> i) & 1) << 4) as u8,
        (((qh >> (i + 16)) & 1) << 4) as u8,
    )
}

/// Q5_0 (22 B / 32): `d`, 32-bit `qh`, Q4_0 nibble order; value = (q | h << 4) - 16.
pub fn dequantize_q5_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for (i, &q) in b[6..22].iter().enumerate() {
        let (h0, h1) = q5_high_bits(qh, i);
        y[i] = (((q & 0x0F) | h0) as i32 - 16) as f32 * d;
        y[i + 16] = (((q >> 4) | h1) as i32 - 16) as f32 * d;
    }
}

/// Q5_1 (24 B / 32): `d`, `m`, 32-bit `qh`, Q4_0 nibble order; value = (q | h << 4) * d + m.
pub fn dequantize_q5_1_block(b: &[u8], y: &mut [f32]) {
    let (d, m) = (f16_at(b, 0), f16_at(b, 2));
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for (i, &q) in b[8..24].iter().enumerate() {
        let (h0, h1) = q5_high_bits(qh, i);
        y[i] = ((q & 0x0F) | h0) as f32 * d + m;
        y[i + 16] = ((q >> 4) | h1) as f32 * d + m;
    }
}

/// Q8_0 (34 B / 32): `d`, then 32 signed bytes.
pub fn dequantize_q8_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (yi, &q) in y.iter_mut().zip(&b[2..34]) {
        *yi = q as i8 as f32 * d;
    }
}

/// ggml `get_scale_min_k4`: unpacks 6-bit (scale, min) pair `j` from a K-quant block's 12 scale bytes.
#[inline]
fn k4_scale_min(j: usize, s: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((s[j] & 63) as f32, (s[j + 4] & 63) as f32)
    } else {
        (
            ((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)) as f32,
            ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as f32,
        )
    }
}

/// Dequantizes one 144-byte GGUF Q4_K block into 256 values (ggml `dequantize_row_q4_K` order).
pub fn dequantize_q4_k_block(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    for g in 0..4 {
        let (sc0, m0) = k4_scale_min(2 * g, scales);
        let (sc1, m1) = k4_scale_min(2 * g + 1, scales);
        let (d0, min0, d1, min1) = (d * sc0, dmin * m0, d * sc1, dmin * m1);
        let q = &qs[g * 32..(g + 1) * 32];
        for l in 0..32 {
            y[g * 64 + l] = d0 * (q[l] & 0x0F) as f32 - min0;
            y[g * 64 + 32 + l] = d1 * (q[l] >> 4) as f32 - min1;
        }
    }
}

/// Dequantizes one 176-byte GGUF Q5_K block into 256 values (ggml `dequantize_row_q5_K` order).
pub fn dequantize_q5_k_block(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
    for g in 0..4 {
        let (sc0, m0) = k4_scale_min(2 * g, scales);
        let (sc1, m1) = k4_scale_min(2 * g + 1, scales);
        let (d0, min0, d1, min1) = (d * sc0, dmin * m0, d * sc1, dmin * m1);
        let q = &qs[g * 32..(g + 1) * 32];
        for l in 0..32 {
            let h0 = ((qh[l] >> (2 * g)) & 1) << 4;
            let h1 = ((qh[l] >> (2 * g + 1)) & 1) << 4;
            y[g * 64 + l] = d0 * ((q[l] & 0x0F) | h0) as f32 - min0;
            y[g * 64 + 32 + l] = d1 * ((q[l] >> 4) | h1) as f32 - min1;
        }
    }
}

/// Dequantizes one 210-byte GGUF Q6_K block into 256 values (ggml `dequantize_row_q6_K` order).
pub fn dequantize_q6_k_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 208);
    for n in 0..2 {
        let ql = &b[n * 64..n * 64 + 64];
        let qh = &b[128 + n * 32..128 + n * 32 + 32];
        let sc = &b[192 + n * 8..192 + n * 8 + 8];
        let y = &mut y[n * 128..n * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q = |lo: u8, shift: u32| ((lo | (((qh[l] >> shift) & 3) << 4)) as i32 - 32) as f32;
            y[l] = d * sc[is] as i8 as f32 * q(ql[l] & 0x0F, 0);
            y[l + 32] = d * sc[is + 2] as i8 as f32 * q(ql[l + 32] & 0x0F, 2);
            y[l + 64] = d * sc[is + 4] as i8 as f32 * q(ql[l] >> 4, 4);
            y[l + 96] = d * sc[is + 6] as i8 as f32 * q(ql[l + 32] >> 4, 6);
        }
    }
}

include!("gguf_tables.rs");

#[inline]
fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

#[inline]
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Byte `j` of a packed ggml grid entry (ggml reads the u64/u32 grids as little-endian byte arrays).
#[inline]
fn grid_byte(g: u64, j: usize) -> u8 {
    (g >> (8 * j)) as u8
}

#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & (1 << j) != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Q2_K (84 B / 256): scales[16], qs[64], d, dmin.
pub fn dequantize_q2_k_block(b: &[u8], y: &mut [f32]) {
    let (d, min) = (f16_at(b, 80), f16_at(b, 82));
    let (scales, qs) = (&b[0..16], &b[16..80]);
    let mut is = 0;
    let mut o = 0;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                for l in 0..16 {
                    y[o] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                    o += 1;
                }
            }
        }
    }
}

/// Q3_K (110 B / 256): hmask[32], qs[64], 6-bit scales[12], d.
pub fn dequantize_q3_k_block(b: &[u8], y: &mut [f32]) {
    let (hm, qs, d_all) = (&b[0..32], &b[32..96], f16_at(b, 108));
    let a = [u32_at(b, 96), u32_at(b, 100), u32_at(b, 104)];
    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let aux = [
        (a[0] & k2) | ((a[2] & k1) << 4),
        (a[1] & k2) | (((a[2] >> 2) & k1) << 4),
        ((a[0] >> 4) & k2) | (((a[2] >> 4) & k1) << 4),
        ((a[1] >> 4) & k2) | (((a[2] >> 6) & k1) << 4),
    ];
    let scale = |i: usize| (aux[i / 4] >> (8 * (i % 4))) as u8 as i8 as i32 - 32;
    let (mut is, mut o, mut m) = (0, 0, 1u8);
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let dl = d_all * scale(is) as f32;
                is += 1;
                for l in 0..16 {
                    let k = l + 16 * half;
                    let h = if hm[k] & m != 0 { 0 } else { 4 };
                    y[o] = dl * (((q[k] >> shift) & 3) as i32 - h) as f32;
                    o += 1;
                }
            }
            m <<= 1;
        }
    }
}

/// TQ1_0 (54 B / 256): base-3 packed qs[48], qh[4], d.
pub fn dequantize_tq1_0_block(b: &[u8], y: &mut [f32]) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let d = f16_at(b, 52);
    let trit = |q: u8, n: usize| ((q.wrapping_mul(POW3[n]) as u16 * 3) >> 8) as f32 - 1.0;
    let mut o = 0;
    for (start, width) in [(0, 32), (32, 16)] {
        for n in 0..5 {
            for m in 0..width {
                y[o] = trit(b[start + m], n) * d;
                o += 1;
            }
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            y[o] = trit(b[48 + j], n) * d;
            o += 1;
        }
    }
}

/// TQ2_0 (66 B / 256): 2-bit qs[64], d; value = (q - 1) * d.
pub fn dequantize_tq2_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 64);
    let mut o = 0;
    for j in [0, 32] {
        for l in 0..4 {
            for m in 0..32 {
                y[o] = (((b[j + m] >> (2 * l)) & 3) as f32 - 1.0) * d;
                o += 1;
            }
        }
    }
}

/// IQ2_XXS (66 B / 256): d, then per 32: 4 grid indices + 32-bit (4x7 sign bits, 4-bit scale).
pub fn dequantize_iq2_xxs_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let q = &b[2 + 8 * ib..10 + 8 * ib];
        let aux1 = u32_at(q, 4);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let g = IQ2XXS_GRID[q[l] as usize];
            let s = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db * grid_byte(g, j) as f32 * sign(s, j);
            }
        }
    }
}

/// IQ2_XS (74 B / 256): d, u16 qs[32] (9-bit grid index + 7-bit sign index), scales[8].
pub fn dequantize_iq2_xs_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let sc = b[66 + ib];
        let db = [
            d * (0.5 + (sc & 0xF) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = u16_at(b, 2 + 2 * (4 * ib + l)) as usize;
            let (g, s) = (IQ2XS_GRID[q & 511], KSIGNS_IQ2XS[q >> 9]);
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db[l / 2] * grid_byte(g, j) as f32 * sign(s, j);
            }
        }
    }
}

/// IQ2_S (82 B / 256): d, qs[32] + signs[32], qh[8], scales[8].
pub fn dequantize_iq2_s_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for ib in 0..8 {
        let sc = scales[ib];
        let db = [
            d * (0.5 + (sc & 0xF) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let idx = qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300);
            let (g, s) = (IQ2S_GRID[idx], signs[4 * ib + l]);
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db[l / 2] * grid_byte(g, j) as f32 * sign(s, j);
            }
        }
    }
}

/// IQ3_XXS (98 B / 256): d, qs[64] grid indices, then per 32 a u32 (4x7 sign bits, 4-bit scale).
pub fn dequantize_iq3_xxs_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qs = &b[2..66];
    for ib in 0..8 {
        let aux = u32_at(b, 66 + 4 * ib);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let s = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize] as u64;
            let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize] as u64;
            for j in 0..4 {
                y[ib * 32 + l * 8 + j] = db * grid_byte(g1, j) as f32 * sign(s, j);
                y[ib * 32 + l * 8 + j + 4] = db * grid_byte(g2, j) as f32 * sign(s, j + 4);
            }
        }
    }
}

/// IQ3_S (110 B / 256): d, qs[64], qh[8], signs[32], scales[4].
pub fn dequantize_iq3_s_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    for ib in 0..8 {
        let sc = scales[ib / 2] >> (4 * (ib % 2)) & 0xF;
        let db = d * (1 + 2 * sc as i32) as f32;
        let h = qh[ib] as usize;
        for l in 0..4 {
            let g1 = IQ3S_GRID[qs[8 * ib + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)] as u64;
            let g2 = IQ3S_GRID[qs[8 * ib + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)] as u64;
            let s = signs[4 * ib + l];
            for j in 0..4 {
                y[ib * 32 + l * 8 + j] = db * grid_byte(g1, j) as f32 * sign(s, j);
                y[ib * 32 + l * 8 + j + 4] = db * grid_byte(g2, j) as f32 * sign(s, j + 4);
            }
        }
    }
}

const IQ1S_DELTA: f32 = 0.125;

/// IQ1_S (50 B / 256): d, qs[32], u16 qh[8] (3-bit index highs, 3-bit scale, delta sign).
pub fn dequantize_iq1_s_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let qh = u16_at(b, 34 + 2 * ib) as usize;
        let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
        let delta = if qh & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let g = IQ1S_GRID[b[2 + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8)];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid_byte(g, j) as i8 as f32 + delta);
            }
        }
    }
}

/// IQ1_M (56 B / 256): qs[32], qh[16], scales[8] (3-bit sub-scales; f16 d spread over the top nibbles).
pub fn dequantize_iq1_m_block(b: &[u8], y: &mut [f32]) {
    let (qs, qh) = (&b[0..32], &b[32..48]);
    let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
    let d = f16_to_f32(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000),
    );
    for ib in 0..8 {
        let s = sc[ib / 2] as usize >> (6 * (ib % 2));
        let dl = [
            d * (2 * (s & 7) + 1) as f32,
            d * (2 * ((s >> 3) & 7) + 1) as f32,
        ];
        let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
        let idx = [
            (h0 << 8) & 0x700,
            (h0 << 4) & 0x700,
            (h1 << 8) & 0x700,
            (h1 << 4) & 0x700,
        ];
        let neg = [h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80];
        for l in 0..4 {
            let g = IQ1S_GRID[qs[4 * ib + l] as usize | idx[l]];
            let delta = if neg[l] != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl[l / 2] * (grid_byte(g, j) as i8 as f32 + delta);
            }
        }
    }
}

/// IQ4_NL (18 B / 32): d, then Q4_0 nibble order through the non-linear `kvalues_iq4nl` table.
pub fn dequantize_iq4_nl_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (i, &q) in b[2..18].iter().enumerate() {
        y[i] = d * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
        y[i + 16] = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
    }
}

/// IQ4_XS (136 B / 256): d, u16 scales_h, scales_l[4], qs[128]; 6-bit sub-scales, IQ4_NL values.
pub fn dequantize_iq4_xs_block(b: &[u8], y: &mut [f32]) {
    let (d, scales_h) = (f16_at(b, 0), u16_at(b, 2));
    for ib in 0..8 {
        let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        for (j, &q) in b[8 + 16 * ib..24 + 16 * ib].iter().enumerate() {
            y[ib * 32 + j] = dl * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
            y[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
        }
    }
}

/// MXFP4 (17 B / 32): E8M0 exponent, then Q4_0 nibble order of doubled E2M1 values; d = 2^(e-128).
pub fn dequantize_mxfp4_block(b: &[u8], y: &mut [f32]) {
    let e = b[0] as u32;
    let d = f32::from_bits(if e < 2 {
        0x0020_0000 << e
    } else {
        (e - 1) << 23
    });
    for (i, &q) in b[1..17].iter().enumerate() {
        y[i] = KVALUES_FP4[(q & 0xF) as usize] as f32 * d;
        y[i + 16] = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
    }
}

/// GEMV over rows of GGUF blocks, dequantizing each block once into a stack buffer.
#[allow(clippy::too_many_arguments)]
fn fused_gemv_gguf_blocks(
    raw_bytes: &[u8],
    block_elems: usize,
    block_bytes: usize,
    dequant: GgufBlockFn,
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    let row_len = in_features.div_ceil(block_elems) * block_bytes;
    if raw_bytes.len() < out_features * row_len {
        return;
    }
    let mut w = [0.0f32; 256];
    for r in 0..out_features {
        let row = &raw_bytes[r * row_len..(r + 1) * row_len];
        let mut sum = bias.map(|b| b[r]).unwrap_or(0.0);
        for (b, chunk) in row.chunks_exact(block_bytes).enumerate() {
            dequant(chunk, &mut w[..block_elems]);
            let base = b * block_elems;
            let n = block_elems.min(in_features - base);
            sum += w[..n]
                .iter()
                .zip(&x[base..base + n])
                .map(|(a, c)| a * c)
                .sum::<f32>();
        }
        out[r] = sum;
    }
}

/// Quantize an f32 weight slice into GGUF Q4_0 format (18 bytes per 32 elements).
pub fn quantize_f32_to_q4_0(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);
    for chunk in weights.chunks(32) {
        let mut max_abs = 0.0f32;
        for &w in chunk {
            let a = w.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let d = max_abs / 7.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = crate::model::f32_to_f16(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let w0 = if i < chunk.len() { chunk[i] } else { 0.0 };
            let w1 = if i + 16 < chunk.len() {
                chunk[i + 16]
            } else {
                0.0
            };
            let q0 = ((w0 * id).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((w1 * id).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            nibbles[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        out.extend_from_slice(&nibbles);
    }
    out
}

/// Quantize an f32 weight slice into GGUF Q8_0 format (34 bytes per 32 elements).
pub fn quantize_f32_to_q8_0(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks * 34);
    for chunk in weights.chunks(32) {
        let mut max_abs = 0.0f32;
        for &w in chunk {
            let a = w.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let d = max_abs / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = crate::model::f32_to_f16(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        for i in 0..32 {
            let w = if i < chunk.len() { chunk[i] } else { 0.0 };
            let q = (w * id).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}
/// Evaluates uncompressed GEMV rows with 8-way and 4-way unrolled SIMD accumulators.
pub fn gemv_rows_uncompressed(
    data: &[f32],
    inf: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    let out_features = out.len();
    let mut i = 0;
    while i + 8 <= out_features {
        let r0 = &data[i * inf..(i + 1) * inf];
        let r1 = &data[(i + 1) * inf..(i + 2) * inf];
        let r2 = &data[(i + 2) * inf..(i + 3) * inf];
        let r3 = &data[(i + 3) * inf..(i + 4) * inf];
        let r4 = &data[(i + 4) * inf..(i + 5) * inf];
        let r5 = &data[(i + 5) * inf..(i + 6) * inf];
        let r6 = &data[(i + 6) * inf..(i + 7) * inf];
        let r7 = &data[(i + 7) * inf..(i + 8) * inf];

        let (mut d0, mut d1, mut d2, mut d3, mut d4, mut d5, mut d6, mut d7) =
            dot_product_8rows_f32(x, r0, r1, r2, r3, r4, r5, r6, r7);

        if let Some(b) = bias {
            d0 += b[i];
            d1 += b[i + 1];
            d2 += b[i + 2];
            d3 += b[i + 3];
            d4 += b[i + 4];
            d5 += b[i + 5];
            d6 += b[i + 6];
            d7 += b[i + 7];
        }

        out[i] = d0;
        out[i + 1] = d1;
        out[i + 2] = d2;
        out[i + 3] = d3;
        out[i + 4] = d4;
        out[i + 5] = d5;
        out[i + 6] = d6;
        out[i + 7] = d7;

        i += 8;
    }

    while i + 4 <= out_features {
        let r0 = &data[i * inf..(i + 1) * inf];
        let r1 = &data[(i + 1) * inf..(i + 2) * inf];
        let r2 = &data[(i + 2) * inf..(i + 3) * inf];
        let r3 = &data[(i + 3) * inf..(i + 4) * inf];

        let (mut d0, mut d1, mut d2, mut d3) = dot_product_4rows_f32(x, r0, r1, r2, r3);

        if let Some(b) = bias {
            d0 += b[i];
            d1 += b[i + 1];
            d2 += b[i + 2];
            d3 += b[i + 3];
        }

        out[i] = d0;
        out[i + 1] = d1;
        out[i + 2] = d2;
        out[i + 3] = d3;

        i += 4;
    }

    while i < out_features {
        let row = &data[i * inf..(i + 1) * inf];
        let mut dot = dot_product_f32(x, row);
        if let Some(b) = bias {
            dot += b[i];
        }
        out[i] = dot;
        i += 1;
    }
}

/// Computes dot products of vector `x` with 8 row vectors simultaneously using ARM NEON intrinsics.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_product_8rows_f32(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    r4: &[f32],
    r5: &[f32],
    r6: &[f32],
    r7: &[f32],
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    use core::arch::aarch64::*;
    let len = x.len();
    let x_ptr = x.as_ptr();
    let r0_ptr = r0.as_ptr();
    let r1_ptr = r1.as_ptr();
    let r2_ptr = r2.as_ptr();
    let r3_ptr = r3.as_ptr();
    let r4_ptr = r4.as_ptr();
    let r5_ptr = r5.as_ptr();
    let r6_ptr = r6.as_ptr();
    let r7_ptr = r7.as_ptr();
    let mut i = 0;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);

        let mut acc0_b = vdupq_n_f32(0.0);
        let mut acc1_b = vdupq_n_f32(0.0);
        let mut acc2_b = vdupq_n_f32(0.0);
        let mut acc3_b = vdupq_n_f32(0.0);
        let mut acc4_b = vdupq_n_f32(0.0);
        let mut acc5_b = vdupq_n_f32(0.0);
        let mut acc6_b = vdupq_n_f32(0.0);
        let mut acc7_b = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let vx_0 = vld1q_f32(x_ptr.add(i));
            let vx_1 = vld1q_f32(x_ptr.add(i + 4));
            let vx_2 = vld1q_f32(x_ptr.add(i + 8));
            let vx_3 = vld1q_f32(x_ptr.add(i + 12));

            let vr0_0 = vld1q_f32(r0_ptr.add(i));
            let vr0_1 = vld1q_f32(r0_ptr.add(i + 4));
            let vr0_2 = vld1q_f32(r0_ptr.add(i + 8));
            let vr0_3 = vld1q_f32(r0_ptr.add(i + 12));
            acc0 = vfmaq_f32(acc0, vx_0, vr0_0);
            acc0_b = vfmaq_f32(acc0_b, vx_1, vr0_1);
            acc0 = vfmaq_f32(acc0, vx_2, vr0_2);
            acc0_b = vfmaq_f32(acc0_b, vx_3, vr0_3);

            let vr1_0 = vld1q_f32(r1_ptr.add(i));
            let vr1_1 = vld1q_f32(r1_ptr.add(i + 4));
            let vr1_2 = vld1q_f32(r1_ptr.add(i + 8));
            let vr1_3 = vld1q_f32(r1_ptr.add(i + 12));
            acc1 = vfmaq_f32(acc1, vx_0, vr1_0);
            acc1_b = vfmaq_f32(acc1_b, vx_1, vr1_1);
            acc1 = vfmaq_f32(acc1, vx_2, vr1_2);
            acc1_b = vfmaq_f32(acc1_b, vx_3, vr1_3);

            let vr2_0 = vld1q_f32(r2_ptr.add(i));
            let vr2_1 = vld1q_f32(r2_ptr.add(i + 4));
            let vr2_2 = vld1q_f32(r2_ptr.add(i + 8));
            let vr2_3 = vld1q_f32(r2_ptr.add(i + 12));
            acc2 = vfmaq_f32(acc2, vx_0, vr2_0);
            acc2_b = vfmaq_f32(acc2_b, vx_1, vr2_1);
            acc2 = vfmaq_f32(acc2, vx_2, vr2_2);
            acc2_b = vfmaq_f32(acc2_b, vx_3, vr2_3);

            let vr3_0 = vld1q_f32(r3_ptr.add(i));
            let vr3_1 = vld1q_f32(r3_ptr.add(i + 4));
            let vr3_2 = vld1q_f32(r3_ptr.add(i + 8));
            let vr3_3 = vld1q_f32(r3_ptr.add(i + 12));
            acc3 = vfmaq_f32(acc3, vx_0, vr3_0);
            acc3_b = vfmaq_f32(acc3_b, vx_1, vr3_1);
            acc3 = vfmaq_f32(acc3, vx_2, vr3_2);
            acc3_b = vfmaq_f32(acc3_b, vx_3, vr3_3);

            let vr4_0 = vld1q_f32(r4_ptr.add(i));
            let vr4_1 = vld1q_f32(r4_ptr.add(i + 4));
            let vr4_2 = vld1q_f32(r4_ptr.add(i + 8));
            let vr4_3 = vld1q_f32(r4_ptr.add(i + 12));
            acc4 = vfmaq_f32(acc4, vx_0, vr4_0);
            acc4_b = vfmaq_f32(acc4_b, vx_1, vr4_1);
            acc4 = vfmaq_f32(acc4, vx_2, vr4_2);
            acc4_b = vfmaq_f32(acc4_b, vx_3, vr4_3);

            let vr5_0 = vld1q_f32(r5_ptr.add(i));
            let vr5_1 = vld1q_f32(r5_ptr.add(i + 4));
            let vr5_2 = vld1q_f32(r5_ptr.add(i + 8));
            let vr5_3 = vld1q_f32(r5_ptr.add(i + 12));
            acc5 = vfmaq_f32(acc5, vx_0, vr5_0);
            acc5_b = vfmaq_f32(acc5_b, vx_1, vr5_1);
            acc5 = vfmaq_f32(acc5, vx_2, vr5_2);
            acc5_b = vfmaq_f32(acc5_b, vx_3, vr5_3);

            let vr6_0 = vld1q_f32(r6_ptr.add(i));
            let vr6_1 = vld1q_f32(r6_ptr.add(i + 4));
            let vr6_2 = vld1q_f32(r6_ptr.add(i + 8));
            let vr6_3 = vld1q_f32(r6_ptr.add(i + 12));
            acc6 = vfmaq_f32(acc6, vx_0, vr6_0);
            acc6_b = vfmaq_f32(acc6_b, vx_1, vr6_1);
            acc6 = vfmaq_f32(acc6, vx_2, vr6_2);
            acc6_b = vfmaq_f32(acc6_b, vx_3, vr6_3);

            let vr7_0 = vld1q_f32(r7_ptr.add(i));
            let vr7_1 = vld1q_f32(r7_ptr.add(i + 4));
            let vr7_2 = vld1q_f32(r7_ptr.add(i + 8));
            let vr7_3 = vld1q_f32(r7_ptr.add(i + 12));
            acc7 = vfmaq_f32(acc7, vx_0, vr7_0);
            acc7_b = vfmaq_f32(acc7_b, vx_1, vr7_1);
            acc7 = vfmaq_f32(acc7, vx_2, vr7_2);
            acc7_b = vfmaq_f32(acc7_b, vx_3, vr7_3);

            i += 16;
        }

        while i + 8 <= len {
            let vx_a = vld1q_f32(x_ptr.add(i));
            let vx_b = vld1q_f32(x_ptr.add(i + 4));

            let vr0_a = vld1q_f32(r0_ptr.add(i));
            let vr0_b = vld1q_f32(r0_ptr.add(i + 4));
            acc0 = vfmaq_f32(acc0, vx_a, vr0_a);
            acc0_b = vfmaq_f32(acc0_b, vx_b, vr0_b);

            let vr1_a = vld1q_f32(r1_ptr.add(i));
            let vr1_b = vld1q_f32(r1_ptr.add(i + 4));
            acc1 = vfmaq_f32(acc1, vx_a, vr1_a);
            acc1_b = vfmaq_f32(acc1_b, vx_b, vr1_b);

            let vr2_a = vld1q_f32(r2_ptr.add(i));
            let vr2_b = vld1q_f32(r2_ptr.add(i + 4));
            acc2 = vfmaq_f32(acc2, vx_a, vr2_a);
            acc2_b = vfmaq_f32(acc2_b, vx_b, vr2_b);

            let vr3_a = vld1q_f32(r3_ptr.add(i));
            let vr3_b = vld1q_f32(r3_ptr.add(i + 4));
            acc3 = vfmaq_f32(acc3, vx_a, vr3_a);
            acc3_b = vfmaq_f32(acc3_b, vx_b, vr3_b);

            let vr4_a = vld1q_f32(r4_ptr.add(i));
            let vr4_b = vld1q_f32(r4_ptr.add(i + 4));
            acc4 = vfmaq_f32(acc4, vx_a, vr4_a);
            acc4_b = vfmaq_f32(acc4_b, vx_b, vr4_b);

            let vr5_a = vld1q_f32(r5_ptr.add(i));
            let vr5_b = vld1q_f32(r5_ptr.add(i + 4));
            acc5 = vfmaq_f32(acc5, vx_a, vr5_a);
            acc5_b = vfmaq_f32(acc5_b, vx_b, vr5_b);

            let vr6_a = vld1q_f32(r6_ptr.add(i));
            let vr6_b = vld1q_f32(r6_ptr.add(i + 4));
            acc6 = vfmaq_f32(acc6, vx_a, vr6_a);
            acc6_b = vfmaq_f32(acc6_b, vx_b, vr6_b);

            let vr7_a = vld1q_f32(r7_ptr.add(i));
            let vr7_b = vld1q_f32(r7_ptr.add(i + 4));
            acc7 = vfmaq_f32(acc7, vx_a, vr7_a);
            acc7_b = vfmaq_f32(acc7_b, vx_b, vr7_b);

            i += 8;
        }

        while i + 4 <= len {
            let vx = vld1q_f32(x_ptr.add(i));
            acc0 = vfmaq_f32(acc0, vx, vld1q_f32(r0_ptr.add(i)));
            acc1 = vfmaq_f32(acc1, vx, vld1q_f32(r1_ptr.add(i)));
            acc2 = vfmaq_f32(acc2, vx, vld1q_f32(r2_ptr.add(i)));
            acc3 = vfmaq_f32(acc3, vx, vld1q_f32(r3_ptr.add(i)));
            acc4 = vfmaq_f32(acc4, vx, vld1q_f32(r4_ptr.add(i)));
            acc5 = vfmaq_f32(acc5, vx, vld1q_f32(r5_ptr.add(i)));
            acc6 = vfmaq_f32(acc6, vx, vld1q_f32(r6_ptr.add(i)));
            acc7 = vfmaq_f32(acc7, vx, vld1q_f32(r7_ptr.add(i)));
            i += 4;
        }

        sum0 += vaddvq_f32(vaddq_f32(acc0, acc0_b));
        sum1 += vaddvq_f32(vaddq_f32(acc1, acc1_b));
        sum2 += vaddvq_f32(vaddq_f32(acc2, acc2_b));
        sum3 += vaddvq_f32(vaddq_f32(acc3, acc3_b));
        sum4 += vaddvq_f32(vaddq_f32(acc4, acc4_b));
        sum5 += vaddvq_f32(vaddq_f32(acc5, acc5_b));
        sum6 += vaddvq_f32(vaddq_f32(acc6, acc6_b));
        sum7 += vaddvq_f32(vaddq_f32(acc7, acc7_b));
    }

    while i < len {
        let vx = x[i];
        sum0 += vx * r0[i];
        sum1 += vx * r1[i];
        sum2 += vx * r2[i];
        sum3 += vx * r3[i];
        sum4 += vx * r4[i];
        sum5 += vx * r5[i];
        sum6 += vx * r6[i];
        sum7 += vx * r7[i];
        i += 1;
    }

    (sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7)
}

/// Fallback for non-aarch64 targets
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn dot_product_8rows_f32(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    r4: &[f32],
    r5: &[f32],
    r6: &[f32],
    r7: &[f32],
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let (d0, d1, d2, d3) = dot_product_4rows_f32(x, r0, r1, r2, r3);
    let (d4, d5, d6, d7) = dot_product_4rows_f32(x, r4, r5, r6, r7);
    (d0, d1, d2, d3, d4, d5, d6, d7)
}

/// Vectorized dot-product of two f32 slices using SIMD/NEON intrinsics where available.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut sum = 0.0f32;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        while i + 16 <= len {
            let va0 = vld1q_f32(a_ptr.add(i));
            let vb0 = vld1q_f32(b_ptr.add(i));
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(a_ptr.add(i + 4));
            let vb1 = vld1q_f32(b_ptr.add(i + 4));
            acc1 = vfmaq_f32(acc1, va1, vb1);

            let va2 = vld1q_f32(a_ptr.add(i + 8));
            let vb2 = vld1q_f32(b_ptr.add(i + 8));
            acc2 = vfmaq_f32(acc2, va2, vb2);

            let va3 = vld1q_f32(a_ptr.add(i + 12));
            let vb3 = vld1q_f32(b_ptr.add(i + 12));
            acc3 = vfmaq_f32(acc3, va3, vb3);

            i += 16;
        }

        let acc01 = vaddq_f32(acc0, acc1);
        let acc23 = vaddq_f32(acc2, acc3);
        let acc_total = vaddq_f32(acc01, acc23);
        sum += vaddvq_f32(acc_total);
    }

    let mut rem0 = 0.0f32;
    let mut rem1 = 0.0f32;
    let mut rem2 = 0.0f32;
    let mut rem3 = 0.0f32;

    while i + 4 <= len {
        rem0 += a[i] * b[i];
        rem1 += a[i + 1] * b[i + 1];
        rem2 += a[i + 2] * b[i + 2];
        rem3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    sum += (rem0 + rem1) + (rem2 + rem3);

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// Computes dot products of vector `x` with 4 row vectors simultaneously using ARM NEON intrinsics.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_product_4rows_f32(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
) -> (f32, f32, f32, f32) {
    use core::arch::aarch64::*;
    assert_eq!(x.len(), r0.len());
    assert_eq!(x.len(), r1.len());
    assert_eq!(x.len(), r2.len());
    assert_eq!(x.len(), r3.len());

    let len = x.len();
    let x_ptr = x.as_ptr();
    let r0_ptr = r0.as_ptr();
    let r1_ptr = r1.as_ptr();
    let r2_ptr = r2.as_ptr();
    let r3_ptr = r3.as_ptr();
    let mut i = 0;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 4 <= len {
            let vx = vld1q_f32(x_ptr.add(i));
            let vr0 = vld1q_f32(r0_ptr.add(i));
            let vr1 = vld1q_f32(r1_ptr.add(i));
            let vr2 = vld1q_f32(r2_ptr.add(i));
            let vr3 = vld1q_f32(r3_ptr.add(i));

            acc0 = vfmaq_f32(acc0, vx, vr0);
            acc1 = vfmaq_f32(acc1, vx, vr1);
            acc2 = vfmaq_f32(acc2, vx, vr2);
            acc3 = vfmaq_f32(acc3, vx, vr3);

            i += 4;
        }

        sum0 += vaddvq_f32(acc0);
        sum1 += vaddvq_f32(acc1);
        sum2 += vaddvq_f32(acc2);
        sum3 += vaddvq_f32(acc3);
    }

    while i < len {
        let vx = x[i];
        sum0 += vx * r0[i];
        sum1 += vx * r1[i];
        sum2 += vx * r2[i];
        sum3 += vx * r3[i];
        i += 1;
    }

    (sum0, sum1, sum2, sum3)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        unsafe { dot_product_f32_avx_fma(a, b) }
    } else if is_x86_feature_detected!("avx") {
        unsafe { dot_product_f32_avx(a, b) }
    } else {
        dot_product_f32_fallback(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn dot_product_f32_avx_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a_ptr.add(i));
        let vb0 = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        i += 16;
    }

    if i + 8 <= len {
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut sum: f32 = buf.iter().sum();

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a_ptr.add(i));
        let vb0 = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));

        let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(i + 8));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));

        i += 16;
    }

    if i + 8 <= len {
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut sum: f32 = buf.iter().sum();

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_product_4rows_f32(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
) -> (f32, f32, f32, f32) {
    assert_eq!(x.len(), r0.len());
    assert_eq!(x.len(), r1.len());
    assert_eq!(x.len(), r2.len());
    assert_eq!(x.len(), r3.len());

    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        unsafe { dot_product_4rows_f32_avx_fma(x, r0, r1, r2, r3) }
    } else {
        dot_product_4rows_f32_fallback(x, r0, r1, r2, r3)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn dot_product_4rows_f32_avx_fma(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::*;
    let len = x.len();
    let x_ptr = x.as_ptr();
    let r0_ptr = r0.as_ptr();
    let r1_ptr = r1.as_ptr();
    let r2_ptr = r2.as_ptr();
    let r3_ptr = r3.as_ptr();
    let mut i = 0;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 8 <= len {
        let vx = _mm256_loadu_ps(x_ptr.add(i));
        let vr0 = _mm256_loadu_ps(r0_ptr.add(i));
        let vr1 = _mm256_loadu_ps(r1_ptr.add(i));
        let vr2 = _mm256_loadu_ps(r2_ptr.add(i));
        let vr3 = _mm256_loadu_ps(r3_ptr.add(i));

        acc0 = _mm256_fmadd_ps(vx, vr0, acc0);
        acc1 = _mm256_fmadd_ps(vx, vr1, acc1);
        acc2 = _mm256_fmadd_ps(vx, vr2, acc2);
        acc3 = _mm256_fmadd_ps(vx, vr3, acc3);

        i += 8;
    }

    let mut buf0 = [0.0f32; 8];
    let mut buf1 = [0.0f32; 8];
    let mut buf2 = [0.0f32; 8];
    let mut buf3 = [0.0f32; 8];

    _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
    _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
    _mm256_storeu_ps(buf2.as_mut_ptr(), acc2);
    _mm256_storeu_ps(buf3.as_mut_ptr(), acc3);

    let mut sum0: f32 = buf0.iter().sum();
    let mut sum1: f32 = buf1.iter().sum();
    let mut sum2: f32 = buf2.iter().sum();
    let mut sum3: f32 = buf3.iter().sum();

    while i < len {
        let vx = x[i];
        sum0 += vx * r0[i];
        sum1 += vx * r1[i];
        sum2 += vx * r2[i];
        sum3 += vx * r3[i];
        i += 1;
    }

    (sum0, sum1, sum2, sum3)
}

/// Fallback unrolled 8-wide loop with multiple accumulators.
#[inline]
pub fn dot_product_f32_fallback(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    while i + 8 <= len {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];

        acc0 += a[i + 4] * b[i + 4];
        acc1 += a[i + 5] * b[i + 5];
        acc2 += a[i + 6] * b[i + 6];
        acc3 += a[i + 7] * b[i + 7];

        i += 8;
    }

    while i < len {
        acc0 += a[i] * b[i];
        i += 1;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Fallback calling scalar/vector dot_product_f32 for 4 rows.
#[inline]
pub fn dot_product_4rows_f32_fallback(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
) -> (f32, f32, f32, f32) {
    (
        dot_product_f32(x, r0),
        dot_product_f32(x, r1),
        dot_product_f32(x, r2),
        dot_product_f32(x, r3),
    )
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_product_f32_fallback(a, b)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn dot_product_4rows_f32(
    x: &[f32],
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
) -> (f32, f32, f32, f32) {
    dot_product_4rows_f32_fallback(x, r0, r1, r2, r3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::QuantFormat;

    #[test]
    fn test_uncompressed_forward_with_bias() {
        // 2 out_features x 3 in_features
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let layer = QuantizedLinearLayer::from_f32("test_layer", 3, 2, &weights, QuantFormat::FP32)
            .unwrap()
            .with_bias(vec![0.5, -0.5])
            .unwrap();

        let x = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 2];
        layer.forward(&x, &mut out);

        // out[0] = 1*1 + 2*1 + 3*1 + 0.5 = 6.5
        // out[1] = 4*1 + 5*1 + 6*1 - 0.5 = 14.5
        assert!((out[0] - 6.5).abs() < 1e-6);
        assert!((out[1] - 14.5).abs() < 1e-6);
        assert_eq!(layer.bits_per_weight(), 32.0); // stored as f32
    }

    /// `from_f32` stores FP16/BF16 as 2-byte weights whose GEMV matches f32 to the format's precision.
    #[test]
    fn test_half_precision_from_f32_matches_f32() {
        let (inf, outf) = (48, 20);
        let weights: Vec<f32> = (0..inf * outf)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let x: Vec<f32> = (0..inf).map(|i| (i as f32 * 0.3).sin()).collect();
        let run = |format| {
            let layer = QuantizedLinearLayer::from_f32("l", inf, outf, &weights, format).unwrap();
            let mut out = vec![0.0; outf];
            layer.forward(&x, &mut out);
            (layer.bits_per_weight(), out)
        };
        let (_, want) = run(QuantFormat::FP32);
        for (format, rel) in [(QuantFormat::FP16, 2e-3), (QuantFormat::BF16, 1e-2)] {
            let (bits, got) = run(format);
            assert_eq!(bits, 16.0, "{format:?} must be stored in 2 bytes");
            let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            for (w, g) in want.iter().zip(&got) {
                assert!((w - g).abs() <= rel * scale, "{format:?}: {g} vs f32 {w}");
            }
        }
    }

    #[test]
    fn test_nvfp4_forward() {
        let in_features = 32;
        let out_features = 32;
        let weights: Vec<f32> = (0..(in_features * out_features))
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();

        let layer = QuantizedLinearLayer::from_f32(
            "nvfp4_layer",
            in_features,
            out_features,
            &weights,
            QuantFormat::NVFP4_E2M1,
        )
        .unwrap();

        assert_eq!(layer.bits_per_weight(), 4.25);

        let x = vec![0.5; in_features];
        let mut out = vec![0.0; out_features];
        layer.forward(&x, &mut out);

        // Verify output is non-zero and finite
        assert!(out.iter().all(|&v| v.is_finite()));
        assert!(out.iter().any(|&v| v.abs() > 1e-4));
    }

    #[test]
    fn test_exl3_forward_all_bitrates() {
        let in_features = 64;
        let out_features = 32;
        let weights: Vec<f32> = (0..(in_features * out_features))
            .map(|i| (i as f32 * 0.05).cos() * 0.5)
            .collect();

        for (format, expected_bpw) in [
            (QuantFormat::EXL3_3Bit, 3.125),
            (QuantFormat::EXL3_4Bit, 4.125),
            (QuantFormat::EXL3_5Bit, 5.125),
        ] {
            let layer = QuantizedLinearLayer::from_f32(
                "exl3_layer",
                in_features,
                out_features,
                &weights,
                format,
            )
            .unwrap();

            assert_eq!(layer.bits_per_weight(), expected_bpw);

            let x = vec![1.0; in_features];
            let mut out = vec![0.0; out_features];
            layer.forward(&x, &mut out);

            assert!(out.iter().all(|&v| v.is_finite()));
            assert!(out.iter().any(|&v| v.abs() > 1e-4));
        }
    }

    #[test]
    fn test_dot_product_f32_correctness() {
        for len in [
            0, 1, 3, 7, 8, 15, 16, 17, 31, 32, 47, 64, 128, 255, 256, 513,
        ] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();

            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let sim = dot_product_f32(&a, &b);
            let fallback = dot_product_f32_fallback(&a, &b);

            assert!(
                (sim - expected).abs() < 1e-4,
                "SIMD dot product mismatch for len {}: got {}, expected {}",
                len,
                sim,
                expected
            );
            assert!(
                (fallback - expected).abs() < 1e-4,
                "Fallback dot product mismatch for len {}: got {}, expected {}",
                len,
                fallback,
                expected
            );
        }
    }

    #[test]
    fn test_dot_product_4rows_f32_matches_individual() {
        for len in [
            0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 47, 64, 127, 128, 255, 256, 513,
        ] {
            let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
            let r0: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2 + 0.1).cos()).collect();
            let r1: Vec<f32> = (0..len).map(|i| (i as f32 * 0.3 + 0.2).sin()).collect();
            let r2: Vec<f32> = (0..len).map(|i| (i as f32 * 0.4 + 0.3).cos()).collect();
            let r3: Vec<f32> = (0..len).map(|i| (i as f32 * 0.5 + 0.4).sin()).collect();

            let (d0, d1, d2, d3) = dot_product_4rows_f32(&x, &r0, &r1, &r2, &r3);

            let e0 = dot_product_f32(&x, &r0);
            let e1 = dot_product_f32(&x, &r1);
            let e2 = dot_product_f32(&x, &r2);
            let e3 = dot_product_f32(&x, &r3);

            assert!(
                (d0 - e0).abs() < 1e-4,
                "Row 0 mismatch for len {}: 4rows={}, single={}",
                len,
                d0,
                e0
            );
            assert!(
                (d1 - e1).abs() < 1e-4,
                "Row 1 mismatch for len {}: 4rows={}, single={}",
                len,
                d1,
                e1
            );
            assert!(
                (d2 - e2).abs() < 1e-4,
                "Row 2 mismatch for len {}: 4rows={}, single={}",
                len,
                d2,
                e2
            );
            assert!(
                (d3 - e3).abs() < 1e-4,
                "Row 3 mismatch for len {}: 4rows={}, single={}",
                len,
                d3,
                e3
            );
        }
    }

    #[test]
    fn test_uncompressed_forward_4rows_unrolled() {
        let in_features = 4;
        let out_features = 5;
        let weights: Vec<f32> = (0..(in_features * out_features))
            .map(|i| i as f32 * 0.5)
            .collect();
        let bias: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.1).collect();

        let layer = QuantizedLinearLayer::from_f32(
            "unrolled_layer",
            in_features,
            out_features,
            &weights,
            QuantFormat::FP32,
        )
        .unwrap()
        .with_bias(bias.clone())
        .unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; out_features];
        layer.forward(&x, &mut out);

        for i in 0..out_features {
            let row = &weights[i * in_features..(i + 1) * in_features];
            let expected = dot_product_f32(&x, row) + bias[i];
            assert!((out[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_weightless_jesse_layer_forward_zero_multiplications() {
        let in_features = 32;
        let out_features = 16;
        let address_bits = 6;
        let weights: Vec<f32> = (0..(in_features * out_features))
            .map(|i| (i as f32 * 0.07).sin() * 2.0)
            .collect();
        let bias: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.1).collect();

        let layer = QuantizedLinearLayer::from_weightless_jesse(
            "jesse_layer",
            in_features,
            out_features,
            address_bits,
            &weights,
        )
        .with_bias(bias.clone())
        .unwrap();

        let layer_from_format = QuantizedLinearLayer::from_f32(
            "jesse_layer_fmt",
            in_features,
            out_features,
            &weights,
            QuantFormat::WeightlessJesseLUT,
        )
        .unwrap();
        assert_eq!(layer_from_format.bits_per_weight(), 0.5);
        assert!(matches!(
            layer_from_format.weight,
            QuantizedWeight::WeightlessJesse(_)
        ));
        assert_eq!(layer.bits_per_weight(), 0.5);

        let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.15).cos()).collect();
        let mut out = vec![0.0; out_features];

        layer.forward(&x, &mut out);

        // Verify valid activations produced
        assert!(
            out.iter().all(|&v| v.is_finite()),
            "Activations must be finite"
        );
        assert!(
            out.iter().any(|&v| v.abs() > 0.0),
            "Activations must produce non-zero signal"
        );
    }

    #[test]
    fn test_zero_copy_slice_f16_fused_gemv() {
        let in_features = 4;
        let out_features = 2;
        // Row 0: [1.0, 2.0, -1.0, 0.0], Row 1: [0.5, -0.5, 2.0, 1.0]
        let f16_w = [
            crate::model::f32_to_f16(1.0),
            crate::model::f32_to_f16(2.0),
            crate::model::f32_to_f16(-1.0),
            crate::model::f32_to_f16(0.0),
            crate::model::f32_to_f16(0.5),
            crate::model::f32_to_f16(-0.5),
            crate::model::f32_to_f16(2.0),
            crate::model::f32_to_f16(1.0),
        ];
        let mut bytes = Vec::new();
        for val in f16_w {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        let layer = QuantizedLinearLayer::from_zero_copy_slice(
            "test_f16_slice",
            in_features,
            out_features,
            Bytes::from(bytes),
            QuantFormat::FP16,
        );
        assert_eq!(layer.bits_per_weight(), 16.0);

        let x = vec![2.0, 1.0, 3.0, 4.0];
        let mut out = vec![0.0; out_features];
        layer.forward(&x, &mut out);

        // Row 0: 1.0*2 + 2.0*1 + -1.0*3 + 0*4 = 1.0
        // Row 1: 0.5*2 + -0.5*1 + 2.0*3 + 1.0*4 = 10.5
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - 10.5).abs() < 1e-4);
    }

    #[test]
    fn test_zero_copy_slice_bf16_fused_gemv() {
        let in_features = 4;
        let out_features = 2;
        let bf16_w = [
            crate::model::f32_to_bf16(1.0),
            crate::model::f32_to_bf16(2.0),
            crate::model::f32_to_bf16(-1.0),
            crate::model::f32_to_bf16(0.0),
            crate::model::f32_to_bf16(0.5),
            crate::model::f32_to_bf16(-0.5),
            crate::model::f32_to_bf16(2.0),
            crate::model::f32_to_bf16(1.0),
        ];
        let mut bytes = Vec::new();
        for val in bf16_w {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        let layer = QuantizedLinearLayer::from_zero_copy_slice(
            "test_bf16_slice",
            in_features,
            out_features,
            Bytes::from(bytes),
            QuantFormat::BF16,
        );

        let x = vec![2.0, 1.0, 3.0, 4.0];
        let mut out = vec![0.0; out_features];
        layer.forward(&x, &mut out);

        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - 10.5).abs() < 1e-4);
    }

    #[test]
    fn test_gguf_q4_0_and_q8_0_fused_gemv() {
        let in_features = 32;
        let out_features = 2;
        let weights: Vec<f32> = (0..(in_features * out_features))
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let layer_q4 = QuantizedLinearLayer::from_f32(
            "q4_0_layer",
            in_features,
            out_features,
            &weights,
            QuantFormat::GGUF_Q4_0,
        )
        .unwrap();
        assert_eq!(layer_q4.bits_per_weight(), 4.5);

        let layer_q8 = QuantizedLinearLayer::from_f32(
            "q8_0_layer",
            in_features,
            out_features,
            &weights,
            QuantFormat::GGUF_Q8_0,
        )
        .unwrap();
        assert_eq!(layer_q8.bits_per_weight(), 8.5);

        let x = vec![1.0; in_features];
        let mut out_q4 = vec![0.0; out_features];
        let mut out_q8 = vec![0.0; out_features];

        layer_q4.forward(&x, &mut out_q4);
        layer_q8.forward(&x, &mut out_q8);

        assert!(out_q4.iter().all(|&v| v.is_finite()));
        assert!(out_q8.iter().all(|&v| v.is_finite()));
    }
}
