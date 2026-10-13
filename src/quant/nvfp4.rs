//! NVFP4 (E2M1) microscaled quantization format for gibbs-engine.
//!
//! Provides 4-bit microscaled floating point (1 sign bit, 2 exponent bits, 1 mantissa bit)
//! block quantization, dequantization, and fused GEMV kernel.

/// Size of each microscaling block (16 elements per block).
pub const NVFP4_BLOCK_SIZE: usize = 16;

/// NVFP4 (E2M1) lookup table representing 16 values for nibbles 0x0..=0xF.
///
/// Positive: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (nibbles 0x0..=0x7)
/// Negative: -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 (nibbles 0x8..=0xF)
pub const NVFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// A microscaled block holding 16 elements packed into 8 bytes (4 bits per element)
/// alongside a 32-bit floating point scaling factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nvfp4Block {
    /// 16 4-bit values packed into 8 bytes (low nibble first, high nibble second).
    pub packed_nibbles: [u8; 8],
    /// Block scaling factor (max absolute value / 6.0).
    pub scale: f32,
}

/// A 2D weight matrix quantized in microscaled NVFP4 format.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Tensor {
    /// Contiguous array of microscaled blocks.
    pub blocks: Vec<Nvfp4Block>,
    /// Number of matrix rows.
    pub rows: usize,
    /// Number of matrix columns.
    pub cols: usize,
    /// The checkpoint's own block scales, when this tensor was read from one rather than quantized from
    /// f32: `(raw E4M3 bytes, one per block in block order, global scale)`. Each block's `scale` is
    /// then `e4m3(byte) * global`; the Metal upload binds the bytes and global scale verbatim.
    pub source_scales: Option<(Vec<u8>, f32)>,
}

/// Quantize a 16-element slice into a single microscaled NVFP4 block.
///
/// Computes scale = max_abs / 6.0, maps each float to the closest value in NVFP4_E2M1_LUT,
/// and packs two 4-bit nibbles per byte (low nibble first, high nibble second).
pub fn quantize_nvfp4_block(data: &[f32; NVFP4_BLOCK_SIZE]) -> Nvfp4Block {
    let mut max_abs = 0.0f32;
    for &val in data.iter() {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs == 0.0 || !max_abs.is_finite() {
        return Nvfp4Block {
            packed_nibbles: [0u8; 8],
            scale: 0.0,
        };
    }

    let scale = max_abs / 6.0;
    let inv_scale = 1.0 / scale;

    let mut packed_nibbles = [0u8; 8];
    for byte_idx in 0..8 {
        let val0 = data[byte_idx * 2] * inv_scale;
        let val1 = data[byte_idx * 2 + 1] * inv_scale;

        let q0 = find_best_nvfp4_nibble(val0);
        let q1 = find_best_nvfp4_nibble(val1);

        packed_nibbles[byte_idx] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }

    Nvfp4Block {
        packed_nibbles,
        scale,
    }
}

/// Find the closest 4-bit NVFP4 index for a normalized value.
#[inline(always)]
fn find_best_nvfp4_nibble(val: f32) -> u8 {
    let is_neg = val.is_sign_negative();
    let (start, end) = if is_neg { (8, 16) } else { (0, 8) };

    let mut best_idx = start as u8;
    let mut best_dist = f32::INFINITY;

    for idx in start..end {
        let dist = (val - NVFP4_E2M1_LUT[idx]).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx as u8;
        }
    }

    best_idx
}

/// Dequantize a microscaled NVFP4 block back into 16 f32 values.
///
/// Unpacks two nibbles per byte, maps each to NVFP4_E2M1_LUT, and multiplies by block.scale.
pub fn dequantize_nvfp4_block(block: &Nvfp4Block, out: &mut [f32; NVFP4_BLOCK_SIZE]) {
    let scale = block.scale;
    for (byte_idx, &byte) in block.packed_nibbles.iter().enumerate() {
        let low = (byte & 0x0F) as usize;
        let high = ((byte >> 4) & 0x0F) as usize;
        out[byte_idx * 2] = NVFP4_E2M1_LUT[low] * scale;
        out[byte_idx * 2 + 1] = NVFP4_E2M1_LUT[high] * scale;
    }
}

/// Quantize a slice of floats into a 2D NVFP4 tensor.
///
/// Dimensions `rows` and `cols` must both be divisible by 16.
pub fn quantize_nvfp4_tensor(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4Tensor, String> {
    if rows == 0 || cols == 0 {
        return Err("Dimensions rows and cols must be greater than zero".to_string());
    }
    if rows % NVFP4_BLOCK_SIZE != 0 || cols % NVFP4_BLOCK_SIZE != 0 {
        return Err(format!(
            "Rows ({}) and cols ({}) must be divisible by {}",
            rows, cols, NVFP4_BLOCK_SIZE
        ));
    }
    let expected_len = rows
        .checked_mul(cols)
        .ok_or_else(|| "Matrix dimensions overflow usize".to_string())?;
    if weights.len() != expected_len {
        return Err(format!(
            "Weights slice length {} does not match rows * cols = {}",
            weights.len(),
            expected_len
        ));
    }

    let num_blocks = expected_len / NVFP4_BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(num_blocks);

    for chunk in weights.chunks_exact(NVFP4_BLOCK_SIZE) {
        let block_data: &[f32; NVFP4_BLOCK_SIZE] = chunk.try_into().unwrap();
        blocks.push(quantize_nvfp4_block(block_data));
    }

    Ok(Nvfp4Tensor { blocks, rows, cols, source_scales: None })
}

/// Dequantize an entire NVFP4 tensor into a contiguous f32 slice.
///
/// A tensor with `source_scales` is expanded as `(lut * e4m3) * global` (the checkpoint dequantizer's
/// order), which can differ in the last bit from `lut * block.scale` when `e4m3 * global` rounds.
pub fn dequantize_nvfp4_tensor(tensor: &Nvfp4Tensor, out: &mut [f32]) {
    let total_elements = tensor.rows * tensor.cols;
    assert_eq!(
        out.len(),
        total_elements,
        "Output buffer length ({}) does not match tensor element count ({})",
        out.len(),
        total_elements
    );
    assert_eq!(
        tensor.blocks.len() * NVFP4_BLOCK_SIZE,
        total_elements,
        "Tensor block count does not match rows * cols"
    );

    for (b_idx, block) in tensor.blocks.iter().enumerate() {
        let offset = b_idx * NVFP4_BLOCK_SIZE;
        let target: &mut [f32; NVFP4_BLOCK_SIZE] = (&mut out[offset..offset + NVFP4_BLOCK_SIZE])
            .try_into()
            .unwrap();
        match &tensor.source_scales {
            None => dequantize_nvfp4_block(block, target),
            Some((bytes, global)) => {
                let s = crate::model::f8_e4m3_to_f32(bytes[b_idx]);
                for (byte_idx, &byte) in block.packed_nibbles.iter().enumerate() {
                    target[byte_idx * 2] = (NVFP4_E2M1_LUT[(byte & 0x0F) as usize] * s) * global;
                    target[byte_idx * 2 + 1] = (NVFP4_E2M1_LUT[(byte >> 4) as usize] * s) * global;
                }
            }
        }
    }
}

/// Fused matrix-vector multiplication (GEMV): out = tensor * x.
///
/// Multiplies `tensor` (rows x cols) with vector `x` (length cols) and writes to `out` (length rows).
/// Multi-accumulator inner dot product for a single microscaled NVFP4 block (16 elements).
#[inline(always)]
fn compute_nvfp4_block_dot(block: &Nvfp4Block, x_ptr: &[f32; NVFP4_BLOCK_SIZE]) -> f32 {
    let p = block.packed_nibbles;

    let s0 = NVFP4_E2M1_LUT[(p[0] & 0x0F) as usize] * x_ptr[0]
        + NVFP4_E2M1_LUT[(p[1] & 0x0F) as usize] * x_ptr[2]
        + NVFP4_E2M1_LUT[(p[2] & 0x0F) as usize] * x_ptr[4]
        + NVFP4_E2M1_LUT[(p[3] & 0x0F) as usize] * x_ptr[6];

    let s1 = NVFP4_E2M1_LUT[((p[0] >> 4) & 0x0F) as usize] * x_ptr[1]
        + NVFP4_E2M1_LUT[((p[1] >> 4) & 0x0F) as usize] * x_ptr[3]
        + NVFP4_E2M1_LUT[((p[2] >> 4) & 0x0F) as usize] * x_ptr[5]
        + NVFP4_E2M1_LUT[((p[3] >> 4) & 0x0F) as usize] * x_ptr[7];

    let s2 = NVFP4_E2M1_LUT[(p[4] & 0x0F) as usize] * x_ptr[8]
        + NVFP4_E2M1_LUT[(p[5] & 0x0F) as usize] * x_ptr[10]
        + NVFP4_E2M1_LUT[(p[6] & 0x0F) as usize] * x_ptr[12]
        + NVFP4_E2M1_LUT[(p[7] & 0x0F) as usize] * x_ptr[14];

    let s3 = NVFP4_E2M1_LUT[((p[4] >> 4) & 0x0F) as usize] * x_ptr[9]
        + NVFP4_E2M1_LUT[((p[5] >> 4) & 0x0F) as usize] * x_ptr[11]
        + NVFP4_E2M1_LUT[((p[6] >> 4) & 0x0F) as usize] * x_ptr[13]
        + NVFP4_E2M1_LUT[((p[7] >> 4) & 0x0F) as usize] * x_ptr[15];

    let block_acc = (s0 + s1) + (s2 + s3);
    block_acc * block.scale
}

/// Fused matrix-vector multiplication (GEMV): out = tensor * x.
///
/// Multiplies `tensor` (rows x cols) with vector `x` (length cols) and writes to `out` (length rows).
/// Maintains 4 separate accumulators (acc0, acc1, acc2, acc3) across blocks to break
/// dependency chains for superscalar IPC.
pub fn gemv_nvfp4_f32(tensor: &Nvfp4Tensor, x: &[f32], out: &mut [f32]) {
    assert_eq!(
        x.len(),
        tensor.cols,
        "Vector x length ({}) must match matrix cols ({})",
        x.len(),
        tensor.cols
    );
    assert_eq!(
        out.len(),
        tensor.rows,
        "Output vector length ({}) must match matrix rows ({})",
        out.len(),
        tensor.rows
    );

    if tensor.rows == 0 || tensor.cols == 0 {
        return;
    }

    let blocks_per_row = tensor.cols / NVFP4_BLOCK_SIZE;

    for r in 0..tensor.rows {
        let row_block_start = r * blocks_per_row;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        let mut b = 0;
        while b + 4 <= blocks_per_row {
            let block0 = &tensor.blocks[row_block_start + b];
            let col0 = b * NVFP4_BLOCK_SIZE;
            let x0: &[f32; NVFP4_BLOCK_SIZE] =
                (&x[col0..col0 + NVFP4_BLOCK_SIZE]).try_into().unwrap();
            acc0 += compute_nvfp4_block_dot(block0, x0);

            let block1 = &tensor.blocks[row_block_start + b + 1];
            let col1 = (b + 1) * NVFP4_BLOCK_SIZE;
            let x1: &[f32; NVFP4_BLOCK_SIZE] =
                (&x[col1..col1 + NVFP4_BLOCK_SIZE]).try_into().unwrap();
            acc1 += compute_nvfp4_block_dot(block1, x1);

            let block2 = &tensor.blocks[row_block_start + b + 2];
            let col2 = (b + 2) * NVFP4_BLOCK_SIZE;
            let x2: &[f32; NVFP4_BLOCK_SIZE] =
                (&x[col2..col2 + NVFP4_BLOCK_SIZE]).try_into().unwrap();
            acc2 += compute_nvfp4_block_dot(block2, x2);

            let block3 = &tensor.blocks[row_block_start + b + 3];
            let col3 = (b + 3) * NVFP4_BLOCK_SIZE;
            let x3: &[f32; NVFP4_BLOCK_SIZE] =
                (&x[col3..col3 + NVFP4_BLOCK_SIZE]).try_into().unwrap();
            acc3 += compute_nvfp4_block_dot(block3, x3);

            b += 4;
        }

        while b < blocks_per_row {
            let block = &tensor.blocks[row_block_start + b];
            let col = b * NVFP4_BLOCK_SIZE;
            let x_ptr: &[f32; NVFP4_BLOCK_SIZE] =
                (&x[col..col + NVFP4_BLOCK_SIZE]).try_into().unwrap();
            acc0 += compute_nvfp4_block_dot(block, x_ptr);
            b += 1;
        }

        out[r] = (acc0 + acc1) + (acc2 + acc3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvfp4_lut_values() {
        assert_eq!(NVFP4_E2M1_LUT.len(), 16);
        assert_eq!(NVFP4_E2M1_LUT[0], 0.0);
        assert_eq!(NVFP4_E2M1_LUT[1], 0.5);
        assert_eq!(NVFP4_E2M1_LUT[2], 1.0);
        assert_eq!(NVFP4_E2M1_LUT[3], 1.5);
        assert_eq!(NVFP4_E2M1_LUT[4], 2.0);
        assert_eq!(NVFP4_E2M1_LUT[5], 3.0);
        assert_eq!(NVFP4_E2M1_LUT[6], 4.0);
        assert_eq!(NVFP4_E2M1_LUT[7], 6.0);
        assert_eq!(NVFP4_E2M1_LUT[8], -0.0);
        assert!(NVFP4_E2M1_LUT[8].is_sign_negative());
        assert_eq!(NVFP4_E2M1_LUT[9], -0.5);
        assert_eq!(NVFP4_E2M1_LUT[10], -1.0);
        assert_eq!(NVFP4_E2M1_LUT[11], -1.5);
        assert_eq!(NVFP4_E2M1_LUT[12], -2.0);
        assert_eq!(NVFP4_E2M1_LUT[13], -3.0);
        assert_eq!(NVFP4_E2M1_LUT[14], -4.0);
        assert_eq!(NVFP4_E2M1_LUT[15], -6.0);
    }

    #[test]
    fn test_quantize_dequantize_exact_block() {
        let input: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let block = quantize_nvfp4_block(&input);
        assert_eq!(block.scale, 1.0);
        assert_eq!(block.packed_nibbles[0], 0x10);
        assert_eq!(block.packed_nibbles[1], 0x32);
        assert_eq!(block.packed_nibbles[2], 0x54);
        assert_eq!(block.packed_nibbles[3], 0x76);
        assert_eq!(block.packed_nibbles[4], 0x98);
        assert_eq!(block.packed_nibbles[5], 0xBA);
        assert_eq!(block.packed_nibbles[6], 0xDC);
        assert_eq!(block.packed_nibbles[7], 0xFE);

        let mut output = [0.0f32; 16];
        dequantize_nvfp4_block(&block, &mut output);
        for i in 0..16 {
            assert_eq!(input[i], output[i]);
        }
    }

    #[test]
    fn test_quantize_dequantize_scaled_block() {
        let scale_factor = 2.5f32;
        let input: [f32; 16] = [
            0.0 * scale_factor, 0.5 * scale_factor, 1.0 * scale_factor, 1.5 * scale_factor,
            2.0 * scale_factor, 3.0 * scale_factor, 4.0 * scale_factor, 6.0 * scale_factor,
            -0.0 * scale_factor, -0.5 * scale_factor, -1.0 * scale_factor, -1.5 * scale_factor,
            -2.0 * scale_factor, -3.0 * scale_factor, -4.0 * scale_factor, -6.0 * scale_factor,
        ];
        let block = quantize_nvfp4_block(&input);
        assert!((block.scale - scale_factor).abs() < 1e-6);

        let mut output = [0.0f32; 16];
        dequantize_nvfp4_block(&block, &mut output);
        for i in 0..16 {
            assert!((input[i] - output[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_quantize_zero_block() {
        let input = [0.0f32; 16];
        let block = quantize_nvfp4_block(&input);
        assert_eq!(block.scale, 0.0);
        assert_eq!(block.packed_nibbles, [0u8; 8]);

        let mut output = [1.0f32; 16];
        dequantize_nvfp4_block(&block, &mut output);
        assert_eq!(output, [0.0f32; 16]);
    }

    #[test]
    fn test_quantize_nvfp4_tensor_validation() {
        let data = vec![0.0f32; 16 * 16];
        assert!(quantize_nvfp4_tensor(&data, 0, 16).is_err());
        assert!(quantize_nvfp4_tensor(&data, 16, 0).is_err());
        assert!(quantize_nvfp4_tensor(&data, 15, 16).is_err());
        assert!(quantize_nvfp4_tensor(&data, 16, 17).is_err());
        assert!(quantize_nvfp4_tensor(&data[..100], 16, 16).is_err());
        assert!(quantize_nvfp4_tensor(&data, 16, 16).is_ok());
    }

    #[test]
    fn test_gemv_nvfp4_f32_correctness() {
        let rows = 32;
        let cols = 32;
        let mut weights = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = ((r as f32) * 0.1 - (c as f32) * 0.05).sin() * 6.0;
                weights.push(v);
            }
        }

        let tensor = quantize_nvfp4_tensor(&weights, rows, cols).expect("quantization failed");

        let mut dequantized = vec![0.0f32; rows * cols];
        dequantize_nvfp4_tensor(&tensor, &mut dequantized);

        let mut x = vec![0.0f32; cols];
        for i in 0..cols {
            x[i] = (i as f32 * 0.2).cos();
        }

        let mut gemv_out = vec![0.0f32; rows];
        gemv_nvfp4_f32(&tensor, &x, &mut gemv_out);

        // Compute reference output from dequantized matrix
        for r in 0..rows {
            let mut expected = 0.0f32;
            for c in 0..cols {
                expected += dequantized[r * cols + c] * x[c];
            }
            assert!(
                (gemv_out[r] - expected).abs() < 1e-4,
                "GEMV mismatch at row {}: gemv={} vs expected={}",
                r, gemv_out[r], expected
            );
        }
    }

    #[test]
    fn test_gemv_nvfp4_f32_multi_block_unrolling() {
        for &cols in &[16, 32, 64, 80, 128] {
            let rows = 16;
            let total = rows * cols;
            let weights: Vec<f32> = (0..total)
                .map(|i| ((i as f32) * 0.1).sin() * 5.0)
                .collect();

            let tensor = quantize_nvfp4_tensor(&weights, rows, cols).unwrap();
            let mut dequantized = vec![0.0f32; total];
            dequantize_nvfp4_tensor(&tensor, &mut dequantized);

            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.15).cos()).collect();
            let mut gemv_out = vec![0.0f32; rows];
            gemv_nvfp4_f32(&tensor, &x, &mut gemv_out);

            for r in 0..rows {
                let mut expected = 0.0f32;
                for c in 0..cols {
                    expected += dequantized[r * cols + c] * x[c];
                }
                assert!(
                    (gemv_out[r] - expected).abs() < 1e-4,
                    "Cols {}, row {} mismatch: gemv={} vs expected={}",
                    cols, r, gemv_out[r], expected
                );
            }
        }
    }
}
