//! EXL3 multi-rate grouped bitstream quantization (3-bit, 4-bit, and 5-bit).
//!
//! Provides asymmetric linear quantization grouped per 32 weights, with compact bitstream
//! packing and high-throughput fused GEMV matrix-vector multiplication.

pub const EXL3_GROUP_SIZE: usize = 32;

/// Supported multi-rate bit depths for EXL3 quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3BitRate {
    /// 3-bit quantization (96 bits = 12 bytes per 32-element group).
    Bits3,
    /// 4-bit quantization (128 bits = 16 bytes per 32-element group).
    Bits4,
    /// 5-bit quantization (160 bits = 20 bytes per 32-element group).
    Bits5,
}

impl Exl3BitRate {
    /// Number of bits per quantized weight element.
    #[inline]
    pub fn bits(&self) -> usize {
        match self {
            Self::Bits3 => 3,
            Self::Bits4 => 4,
            Self::Bits5 => 5,
        }
    }

    /// Number of packed bytes required for a group of 32 weights.
    #[inline]
    pub fn bytes_per_group(&self) -> usize {
        match self {
            Self::Bits3 => 12,
            Self::Bits4 => 16,
            Self::Bits5 => 20,
        }
    }

    /// Maximum integer level representable with this bit depth: (1 << bits) - 1.
    #[inline]
    pub fn max_level(&self) -> u8 {
        match self {
            Self::Bits3 => 7,
            Self::Bits4 => 15,
            Self::Bits5 => 31,
        }
    }
}

/// A quantized group of 32 weights with packed bitstream data and affine scales.
#[derive(Debug, Clone, PartialEq)]
pub struct Exl3Group {
    pub packed_bytes: Vec<u8>,
    pub scale: f32,
    pub zero_point: f32,
}

/// A 2D quantized matrix of shape (rows, cols) quantized in groups of 32 weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Exl3Tensor {
    pub groups: Vec<Exl3Group>,
    pub rows: usize,
    pub cols: usize,
    pub bit_rate: Exl3BitRate,
}

/// Packs 32 integer values in [0, max_level] into a compact byte vector according to bit rate.
#[inline]
fn pack_group_bits(q: &[u8; EXL3_GROUP_SIZE], bit_rate: Exl3BitRate) -> Vec<u8> {
    match bit_rate {
        Exl3BitRate::Bits3 => {
            // 32 elements * 3 bits = 96 bits = 12 bytes.
            // 4 blocks of 8 elements * 3 bits = 24 bits = 3 bytes per block.
            let mut bytes = vec![0u8; 12];
            for block in 0..4 {
                let v = &q[block * 8..(block + 1) * 8];
                let w: u32 = (v[0] as u32 & 0x07)
                    | ((v[1] as u32 & 0x07) << 3)
                    | ((v[2] as u32 & 0x07) << 6)
                    | ((v[3] as u32 & 0x07) << 9)
                    | ((v[4] as u32 & 0x07) << 12)
                    | ((v[5] as u32 & 0x07) << 15)
                    | ((v[6] as u32 & 0x07) << 18)
                    | ((v[7] as u32 & 0x07) << 21);
                let out_offset = block * 3;
                bytes[out_offset] = (w & 0xFF) as u8;
                bytes[out_offset + 1] = ((w >> 8) & 0xFF) as u8;
                bytes[out_offset + 2] = ((w >> 16) & 0xFF) as u8;
            }
            bytes
        }
        Exl3BitRate::Bits4 => {
            // 32 elements * 4 bits = 128 bits = 16 bytes (2 elements per byte).
            let mut bytes = vec![0u8; 16];
            for i in 0..16 {
                bytes[i] = (q[2 * i] & 0x0F) | ((q[2 * i + 1] & 0x0F) << 4);
            }
            bytes
        }
        Exl3BitRate::Bits5 => {
            // 32 elements * 5 bits = 160 bits = 20 bytes.
            // 4 blocks of 8 elements * 5 bits = 40 bits = 5 bytes per block.
            let mut bytes = vec![0u8; 20];
            for block in 0..4 {
                let v = &q[block * 8..(block + 1) * 8];
                let w: u64 = (v[0] as u64 & 0x1F)
                    | ((v[1] as u64 & 0x1F) << 5)
                    | ((v[2] as u64 & 0x1F) << 10)
                    | ((v[3] as u64 & 0x1F) << 15)
                    | ((v[4] as u64 & 0x1F) << 20)
                    | ((v[5] as u64 & 0x1F) << 25)
                    | ((v[6] as u64 & 0x1F) << 30)
                    | ((v[7] as u64 & 0x1F) << 35);
                let out_offset = block * 5;
                bytes[out_offset] = (w & 0xFF) as u8;
                bytes[out_offset + 1] = ((w >> 8) & 0xFF) as u8;
                bytes[out_offset + 2] = ((w >> 16) & 0xFF) as u8;
                bytes[out_offset + 3] = ((w >> 24) & 0xFF) as u8;
                bytes[out_offset + 4] = ((w >> 32) & 0xFF) as u8;
            }
            bytes
        }
    }
}

/// Unpacks a group bitstream into 32 quantized integer values.
#[inline]
fn unpack_group_bits(bytes: &[u8], bit_rate: Exl3BitRate, q: &mut [u8; EXL3_GROUP_SIZE]) {
    match bit_rate {
        Exl3BitRate::Bits3 => {
            assert!(bytes.len() >= 12, "Bits3 requires at least 12 packed bytes");
            for block in 0..4 {
                let in_offset = block * 3;
                let w: u32 = (bytes[in_offset] as u32)
                    | ((bytes[in_offset + 1] as u32) << 8)
                    | ((bytes[in_offset + 2] as u32) << 16);
                let out_offset = block * 8;
                q[out_offset] = (w & 0x07) as u8;
                q[out_offset + 1] = ((w >> 3) & 0x07) as u8;
                q[out_offset + 2] = ((w >> 6) & 0x07) as u8;
                q[out_offset + 3] = ((w >> 9) & 0x07) as u8;
                q[out_offset + 4] = ((w >> 12) & 0x07) as u8;
                q[out_offset + 5] = ((w >> 15) & 0x07) as u8;
                q[out_offset + 6] = ((w >> 18) & 0x07) as u8;
                q[out_offset + 7] = ((w >> 21) & 0x07) as u8;
            }
        }
        Exl3BitRate::Bits4 => {
            assert!(bytes.len() >= 16, "Bits4 requires at least 16 packed bytes");
            for i in 0..16 {
                let b = bytes[i];
                q[2 * i] = b & 0x0F;
                q[2 * i + 1] = (b >> 4) & 0x0F;
            }
        }
        Exl3BitRate::Bits5 => {
            assert!(bytes.len() >= 20, "Bits5 requires at least 20 packed bytes");
            for block in 0..4 {
                let in_offset = block * 5;
                let w: u64 = (bytes[in_offset] as u64)
                    | ((bytes[in_offset + 1] as u64) << 8)
                    | ((bytes[in_offset + 2] as u64) << 16)
                    | ((bytes[in_offset + 3] as u64) << 24)
                    | ((bytes[in_offset + 4] as u64) << 32);
                let out_offset = block * 8;
                q[out_offset] = (w & 0x1F) as u8;
                q[out_offset + 1] = ((w >> 5) & 0x1F) as u8;
                q[out_offset + 2] = ((w >> 10) & 0x1F) as u8;
                q[out_offset + 3] = ((w >> 15) & 0x1F) as u8;
                q[out_offset + 4] = ((w >> 20) & 0x1F) as u8;
                q[out_offset + 5] = ((w >> 25) & 0x1F) as u8;
                q[out_offset + 6] = ((w >> 30) & 0x1F) as u8;
                q[out_offset + 7] = ((w >> 35) & 0x1F) as u8;
            }
        }
    }
}

/// Quantizes a single group of 32 float weights into an [`Exl3Group`].
///
/// Finds the min and max values, sets `zero_point = min`, computes `scale = (max - min) / max_level`,
/// quantizes each element to `u8` in `[0, max_level]`, and bit-packs them.
pub fn quantize_exl3_group(data: &[f32; 32], bit_rate: Exl3BitRate) -> Exl3Group {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &val in data.iter() {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    if !min.is_finite() || !max.is_finite() {
        min = 0.0;
        max = 0.0;
    }

    let zero_point = min;
    let max_level = bit_rate.max_level() as f32;
    let scale = if max > min {
        (max - min) / max_level
    } else {
        0.0
    };

    let mut q_vals = [0u8; EXL3_GROUP_SIZE];
    if scale > 0.0 {
        let inv_scale = 1.0 / scale;
        for i in 0..EXL3_GROUP_SIZE {
            let q = ((data[i] - zero_point) * inv_scale).round();
            q_vals[i] = q.clamp(0.0, max_level) as u8;
        }
    }

    let packed_bytes = pack_group_bits(&q_vals, bit_rate);

    Exl3Group {
        packed_bytes,
        scale,
        zero_point,
    }
}

/// Dequantizes an [`Exl3Group`] back into 32 float values in `out`.
///
/// Unpacks the bitstream into integer values, then reconstructs `val * scale + zero_point`.
pub fn dequantize_exl3_group(group: &Exl3Group, bit_rate: Exl3BitRate, out: &mut [f32; 32]) {
    let mut q_vals = [0u8; EXL3_GROUP_SIZE];
    unpack_group_bits(&group.packed_bytes, bit_rate, &mut q_vals);

    let scale = group.scale;
    let zero_point = group.zero_point;
    for i in 0..EXL3_GROUP_SIZE {
        out[i] = (q_vals[i] as f32) * scale + zero_point;
    }
}

/// Quantizes a full weight matrix into an [`Exl3Tensor`].
///
/// Matrix dimensions must satisfy `weights.len() == rows * cols` and `cols % EXL3_GROUP_SIZE == 0`.
pub fn quantize_exl3_tensor(
    weights: &[f32],
    rows: usize,
    cols: usize,
    bit_rate: Exl3BitRate,
) -> Result<Exl3Tensor, String> {
    if weights.len() != rows * cols {
        return Err(format!(
            "Weights length {} does not match rows * cols ({} * {} = {})",
            weights.len(),
            rows,
            cols,
            rows * cols
        ));
    }
    if cols % EXL3_GROUP_SIZE != 0 {
        return Err(format!(
            "cols ({}) must be a multiple of EXL3_GROUP_SIZE ({})",
            cols, EXL3_GROUP_SIZE
        ));
    }

    let num_groups = (rows * cols) / EXL3_GROUP_SIZE;
    let mut groups = Vec::with_capacity(num_groups);

    for chunk in weights.chunks_exact(EXL3_GROUP_SIZE) {
        let chunk_arr: &[f32; EXL3_GROUP_SIZE] = chunk
            .try_into()
            .map_err(|_| "Failed to chunk weights into group size".to_string())?;
        groups.push(quantize_exl3_group(chunk_arr, bit_rate));
    }

    Ok(Exl3Tensor {
        groups,
        rows,
        cols,
        bit_rate,
    })
}

/// Dequantizes an entire [`Exl3Tensor`] into a destination float slice `out`.
///
/// The output slice length must equal `tensor.rows * tensor.cols`.
pub fn dequantize_exl3_tensor(tensor: &Exl3Tensor, out: &mut [f32]) {
    let expected_len = tensor.rows * tensor.cols;
    assert_eq!(
        out.len(),
        expected_len,
        "Output slice length {} does not match tensor dimensions ({} * {} = {})",
        out.len(),
        tensor.rows,
        tensor.cols,
        expected_len
    );

    let num_groups = expected_len / EXL3_GROUP_SIZE;
    assert_eq!(
        tensor.groups.len(),
        num_groups,
        "Tensor groups count {} does not match expected group count {}",
        tensor.groups.len(),
        num_groups
    );

    for (g_idx, group) in tensor.groups.iter().enumerate() {
        let offset = g_idx * EXL3_GROUP_SIZE;
        let out_group: &mut [f32; EXL3_GROUP_SIZE] = (&mut out[offset..offset + EXL3_GROUP_SIZE])
            .try_into()
            .expect("group slice conversion");
        dequantize_exl3_group(group, tensor.bit_rate, out_group);
    }
}

/// Multi-accumulator dot product for a single dequantized EXL3 group (32 elements).
#[inline(always)]
fn compute_exl3_group_dot(
    group: &Exl3Group,
    bit_rate: Exl3BitRate,
    x_sub: &[f32; EXL3_GROUP_SIZE],
    w_group: &mut [f32; EXL3_GROUP_SIZE],
) -> f32 {
    dequantize_exl3_group(group, bit_rate, w_group);

    // 8-way unrolled multi-accumulator dot product for 32 elements
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;

    for i in (0..EXL3_GROUP_SIZE).step_by(8) {
        s0 += w_group[i] * x_sub[i];
        s1 += w_group[i + 1] * x_sub[i + 1];
        s2 += w_group[i + 2] * x_sub[i + 2];
        s3 += w_group[i + 3] * x_sub[i + 3];
        s4 += w_group[i + 4] * x_sub[i + 4];
        s5 += w_group[i + 5] * x_sub[i + 5];
        s6 += w_group[i + 6] * x_sub[i + 6];
        s7 += w_group[i + 7] * x_sub[i + 7];
    }

    ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

/// Fused GEMV multiplying [`Exl3Tensor`] (matrix rows x cols) with vector `x` (length cols)
/// and writing the result into `out` (length rows).
///
/// Employs 4-way group unrolling with multiple accumulators (`acc0`, `acc1`, `acc2`, `acc3`)
/// and 8-way intra-group unrolling with zero heap allocations.
pub fn gemv_exl3_f32(tensor: &Exl3Tensor, x: &[f32], out: &mut [f32]) {
    assert_eq!(
        x.len(),
        tensor.cols,
        "Vector x length {} does not match tensor cols {}",
        x.len(),
        tensor.cols
    );
    assert_eq!(
        out.len(),
        tensor.rows,
        "Output slice length {} does not match tensor rows {}",
        out.len(),
        tensor.rows
    );

    let groups_per_row = tensor.cols / EXL3_GROUP_SIZE;
    assert_eq!(
        tensor.groups.len(),
        tensor.rows * groups_per_row,
        "Tensor groups count {} does not match rows * groups_per_row ({})",
        tensor.groups.len(),
        tensor.rows * groups_per_row
    );

    let mut w_group0 = [0.0f32; EXL3_GROUP_SIZE];
    let mut w_group1 = [0.0f32; EXL3_GROUP_SIZE];
    let mut w_group2 = [0.0f32; EXL3_GROUP_SIZE];
    let mut w_group3 = [0.0f32; EXL3_GROUP_SIZE];

    for r in 0..tensor.rows {
        let row_group_offset = r * groups_per_row;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        let mut g = 0;
        while g + 4 <= groups_per_row {
            let g0 = row_group_offset + g;
            let x0 = g * EXL3_GROUP_SIZE;
            let x_sub0: &[f32; EXL3_GROUP_SIZE] =
                (&x[x0..x0 + EXL3_GROUP_SIZE]).try_into().unwrap();
            acc0 += compute_exl3_group_dot(
                &tensor.groups[g0],
                tensor.bit_rate,
                x_sub0,
                &mut w_group0,
            );

            let g1 = row_group_offset + g + 1;
            let x1 = (g + 1) * EXL3_GROUP_SIZE;
            let x_sub1: &[f32; EXL3_GROUP_SIZE] =
                (&x[x1..x1 + EXL3_GROUP_SIZE]).try_into().unwrap();
            acc1 += compute_exl3_group_dot(
                &tensor.groups[g1],
                tensor.bit_rate,
                x_sub1,
                &mut w_group1,
            );

            let g2 = row_group_offset + g + 2;
            let x2 = (g + 2) * EXL3_GROUP_SIZE;
            let x_sub2: &[f32; EXL3_GROUP_SIZE] =
                (&x[x2..x2 + EXL3_GROUP_SIZE]).try_into().unwrap();
            acc2 += compute_exl3_group_dot(
                &tensor.groups[g2],
                tensor.bit_rate,
                x_sub2,
                &mut w_group2,
            );

            let g3 = row_group_offset + g + 3;
            let x3 = (g + 3) * EXL3_GROUP_SIZE;
            let x_sub3: &[f32; EXL3_GROUP_SIZE] =
                (&x[x3..x3 + EXL3_GROUP_SIZE]).try_into().unwrap();
            acc3 += compute_exl3_group_dot(
                &tensor.groups[g3],
                tensor.bit_rate,
                x_sub3,
                &mut w_group3,
            );

            g += 4;
        }

        while g < groups_per_row {
            let g_idx = row_group_offset + g;
            let x_off = g * EXL3_GROUP_SIZE;
            let x_sub: &[f32; EXL3_GROUP_SIZE] =
                (&x[x_off..x_off + EXL3_GROUP_SIZE]).try_into().unwrap();
            acc0 += compute_exl3_group_dot(
                &tensor.groups[g_idx],
                tensor.bit_rate,
                x_sub,
                &mut w_group0,
            );
            g += 1;
        }

        out[r] = (acc0 + acc1) + (acc2 + acc3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_packing_roundtrip_bits3() {
        let mut original = [0u8; EXL3_GROUP_SIZE];
        for i in 0..EXL3_GROUP_SIZE {
            original[i] = (i % 8) as u8;
        }

        let packed = pack_group_bits(&original, Exl3BitRate::Bits3);
        assert_eq!(packed.len(), 12);

        let mut unpacked = [0u8; EXL3_GROUP_SIZE];
        unpack_group_bits(&packed, Exl3BitRate::Bits3, &mut unpacked);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_bit_packing_roundtrip_bits4() {
        let mut original = [0u8; EXL3_GROUP_SIZE];
        for i in 0..EXL3_GROUP_SIZE {
            original[i] = (i % 16) as u8;
        }

        let packed = pack_group_bits(&original, Exl3BitRate::Bits4);
        assert_eq!(packed.len(), 16);

        let mut unpacked = [0u8; EXL3_GROUP_SIZE];
        unpack_group_bits(&packed, Exl3BitRate::Bits4, &mut unpacked);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_bit_packing_roundtrip_bits5() {
        let mut original = [0u8; EXL3_GROUP_SIZE];
        for i in 0..EXL3_GROUP_SIZE {
            original[i] = (i % 32) as u8;
        }

        let packed = pack_group_bits(&original, Exl3BitRate::Bits5);
        assert_eq!(packed.len(), 20);

        let mut unpacked = [0u8; EXL3_GROUP_SIZE];
        unpack_group_bits(&packed, Exl3BitRate::Bits5, &mut unpacked);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_quantize_dequantize_group() {
        let bit_rates = [Exl3BitRate::Bits3, Exl3BitRate::Bits4, Exl3BitRate::Bits5];
        for &rate in &bit_rates {
            let mut data = [0.0f32; EXL3_GROUP_SIZE];
            for i in 0..EXL3_GROUP_SIZE {
                data[i] = (i as f32) * 0.25 - 2.0;
            }

            let group = quantize_exl3_group(&data, rate);
            assert_eq!(group.packed_bytes.len(), rate.bytes_per_group());

            let mut dequant = [0.0f32; EXL3_GROUP_SIZE];
            dequantize_exl3_group(&group, rate, &mut dequant);

            let max_error = (data[31] - data[0]) / (rate.max_level() as f32);
            for i in 0..EXL3_GROUP_SIZE {
                let err = (data[i] - dequant[i]).abs();
                assert!(
                    err <= max_error + 1e-5,
                    "Rate {:?}: element {} error {} exceeds max_error {}",
                    rate,
                    i,
                    err,
                    max_error
                );
            }
        }
    }

    #[test]
    fn test_constant_data_quantize() {
        let data = [5.5f32; EXL3_GROUP_SIZE];
        let group = quantize_exl3_group(&data, Exl3BitRate::Bits4);
        assert_eq!(group.scale, 0.0);
        assert_eq!(group.zero_point, 5.5);

        let mut dequant = [0.0f32; EXL3_GROUP_SIZE];
        dequantize_exl3_group(&group, Exl3BitRate::Bits4, &mut dequant);
        for &val in dequant.iter() {
            assert_eq!(val, 5.5);
        }
    }

    #[test]
    fn test_tensor_quantize_gemv() {
        let rows = 4;
        let cols = 64;
        let total = rows * cols;
        let mut weights = vec![0.0f32; total];
        for i in 0..total {
            weights[i] = ((i as f32) * 0.1).sin();
        }

        let bit_rate = Exl3BitRate::Bits4;
        let tensor = quantize_exl3_tensor(&weights, rows, cols, bit_rate).unwrap();
        assert_eq!(tensor.groups.len(), (rows * cols) / EXL3_GROUP_SIZE);

        let mut dequant = vec![0.0f32; total];
        dequantize_exl3_tensor(&tensor, &mut dequant);

        let mut x = vec![0.0f32; cols];
        for j in 0..cols {
            x[j] = (j as f32) * 0.05 + 0.1;
        }

        let mut out_gemv = vec![0.0f32; rows];
        gemv_exl3_f32(&tensor, &x, &mut out_gemv);

        // Reference multiplication using dequantized matrix
        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += dequant[r * cols + c] * x[c];
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            let diff = (out_gemv[r] - expected[r]).abs();
            assert!(
                diff < 1e-4,
                "Row {} GEMV mismatch: got {}, expected {}",
                r,
                out_gemv[r],
                expected[r]
            );
        }
    }

    #[test]
    fn test_dimension_validation() {
        let weights = vec![1.0; 64];
        let err1 = quantize_exl3_tensor(&weights, 2, 30, Exl3BitRate::Bits3);
        assert!(err1.is_err());

        let err2 = quantize_exl3_tensor(&weights, 2, 48, Exl3BitRate::Bits3);
        assert!(err2.is_err());
    }

    #[test]
    fn test_tensor_quantize_gemv_multi_accumulator_all_rates() {
        for &bit_rate in &[Exl3BitRate::Bits3, Exl3BitRate::Bits4, Exl3BitRate::Bits5] {
            for &cols in &[64, 128, 160] {
                let rows = 4;
                let total = rows * cols;
                let weights: Vec<f32> = (0..total)
                    .map(|i| ((i as f32) * 0.07).sin() * 2.0)
                    .collect();

                let tensor = quantize_exl3_tensor(&weights, rows, cols, bit_rate).unwrap();
                let mut dequant = vec![0.0f32; total];
                dequantize_exl3_tensor(&tensor, &mut dequant);

                let x: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.03).cos()).collect();
                let mut out_gemv = vec![0.0f32; rows];
                gemv_exl3_f32(&tensor, &x, &mut out_gemv);

                for r in 0..rows {
                    let mut expected = 0.0f32;
                    for c in 0..cols {
                        expected += dequant[r * cols + c] * x[c];
                    }
                    let diff = (out_gemv[r] - expected).abs();
                    assert!(
                        diff < 1e-4,
                        "Rate {:?}, cols {}, row {} GEMV mismatch: got {}, expected {}",
                        bit_rate,
                        cols,
                        r,
                        out_gemv[r],
                        expected
                    );
                }
            }
        }
    }
}
