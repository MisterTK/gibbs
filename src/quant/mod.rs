pub mod nvfp4;

pub use nvfp4::*;

pub struct QuantEngine;

impl QuantEngine {
    /// Fast Walsh-Hadamard Transform (FWHT) for SpinQuant / QuaRot online outlier elimination
    /// Transforms vector of length N (power of 2) in O(N log N) in-place.
    pub fn fast_walsh_hadamard_transform_f32(vec: &mut [f32]) {
        let n = vec.len();
        assert!(n.is_power_of_two(), "FWHT requires power-of-two length");

        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let u = vec[j];
                    let v = vec[j + h];
                    vec[j] = u + v;
                    vec[j + h] = u - v;
                }
            }
            h *= 2;
        }

        // Normalize by 1 / sqrt(n)
        let norm = 1.0 / (n as f32).sqrt();
        for x in vec.iter_mut() {
            *x *= norm;
        }
    }

    /// KIVI 2-Bit Asymmetric Quantization for Key/Value vectors
    pub fn quantize_kivi_2bit(data: &[f32], quantized_out: &mut [u8]) -> (f32, f32) {
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max_val > min_val { (max_val - min_val) / 3.0 } else { 1.0 };

        // Pack 4 x 2-bit numbers per byte
        for (byte_idx, chunk) in data.chunks(4).enumerate() {
            let mut byte_val = 0u8;
            for (sub_idx, &val) in chunk.iter().enumerate() {
                let q = (((val - min_val) / scale).round() as u8).min(3);
                byte_val |= q << (sub_idx * 2);
            }
            if byte_idx < quantized_out.len() {
                quantized_out[byte_idx] = byte_val;
            }
        }

        (min_val, scale)
    }

    /// BitNet b1.58 Ternary Packing {-1, 0, +1}
    pub fn pack_ternary_b1_58(weights: &[i8], packed_out: &mut [u8]) {
        // Pack 4 ternary values into 1 byte (2 bits each: 0b00 = -1, 0b01 = 0, 0b10 = +1)
        for (byte_idx, chunk) in weights.chunks(4).enumerate() {
            let mut byte_val = 0u8;
            for (sub_idx, &w) in chunk.iter().enumerate() {
                let code = match w {
                    -1 => 0b00,
                    0 => 0b01,
                    1 => 0b10,
                    _ => 0b01,
                };
                byte_val |= code << (sub_idx * 2);
            }
            if byte_idx < packed_out.len() {
                packed_out[byte_idx] = byte_val;
            }
        }
    }

    /// TurboQuant codebook residual compression
    pub fn compress_turboquant(
        data: &[f32],
        codebook: &[f32; 16],
        indices_out: &mut [u8],
    ) {
        // Pack 2 x 4-bit codebook indices per byte
        for (byte_idx, chunk) in data.chunks(2).enumerate() {
            let mut byte_val = 0u8;
            for (sub_idx, &val) in chunk.iter().enumerate() {
                let mut best_idx = 0;
                let mut best_dist = f32::INFINITY;
                for (c_idx, &c_val) in codebook.iter().enumerate() {
                    let dist = (val - c_val).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = c_idx as u8;
                    }
                }
                byte_val |= (best_idx & 0x0F) << (sub_idx * 4);
            }
            if byte_idx < indices_out.len() {
                indices_out[byte_idx] = byte_val;
            }
        }
    }
}
