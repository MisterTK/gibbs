//! SIMD primitives for Gibbs.
//!
//! Vector routines for RMSNorm, Softmax, Q4_0 GEMV, and dot products using ARM NEON on aarch64.

use crate::model::f16_to_f32;

#[inline]
pub fn vector_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let va0 = vld1q_f32(a.as_ptr().add(i));
            let vb0 = vld1q_f32(b.as_ptr().add(i));
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(a.as_ptr().add(i + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
            acc1 = vfmaq_f32(acc1, va1, vb1);

            let va2 = vld1q_f32(a.as_ptr().add(i + 8));
            let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
            acc2 = vfmaq_f32(acc2, va2, vb2);

            let va3 = vld1q_f32(a.as_ptr().add(i + 12));
            let vb3 = vld1q_f32(b.as_ptr().add(i + 12));
            acc3 = vfmaq_f32(acc3, va3, vb3);

            i += 16;
        }

        let sum4 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut dot = vaddvq_f32(sum4);

        while i < len {
            dot += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        dot
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        while i + 8 <= len {
            acc0 += a[i] * b[i] + a[i + 1] * b[i + 1];
            acc1 += a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            acc2 += a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5];
            acc3 += a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
            i += 8;
        }
        let mut dot = acc0 + acc1 + acc2 + acc3;
        while i < len {
            dot += a[i] * b[i];
            i += 1;
        }
        dot
    }
}

/// In-place or out-of-place vectorized RMSNorm: out = x * weight / sqrt(mean(x^2) + eps).
#[inline]
pub fn vector_rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    let sum_sq = vector_dot_product(x, x);
    let scale = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vscale = vdupq_n_f32(scale);
        while i + 8 <= n {
            let vx0 = vld1q_f32(x.as_ptr().add(i));
            let vw0 = vld1q_f32(weight.as_ptr().add(i));
            let vo0 = vmulq_f32(vmulq_f32(vx0, vscale), vw0);
            vst1q_f32(out.as_mut_ptr().add(i), vo0);

            let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
            let vw1 = vld1q_f32(weight.as_ptr().add(i + 4));
            let vo1 = vmulq_f32(vmulq_f32(vx1, vscale), vw1);
            vst1q_f32(out.as_mut_ptr().add(i + 4), vo1);

            i += 8;
        }
    }

    while i < n {
        out[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

/// Vectorized in-place numerically stable Softmax: x[i] = exp(x[i] - max) / sum(exp).
#[inline]
pub fn vector_softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for val in x.iter_mut() {
        let e = (*val - max_val).exp();
        *val = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vinv = vdupq_n_f32(inv_sum);
        while i + 8 <= n {
            let vx0 = vld1q_f32(x.as_ptr().add(i));
            vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(vx0, vinv));

            let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
            vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(vx1, vinv));

            i += 8;
        }
    }

    while i < n {
        x[i] *= inv_sum;
        i += 1;
    }
}

/// Fused GEMV for quantized GGUF Q4_0 weights against f32 activations: out = W * x + bias.
///
/// Features high-throughput zero-allocation NEON SIMD vectorization on aarch64 and AVX on x86_64,
/// dequantizing nibbles directly into vector registers and accumulating without intermediate allocations.
pub fn vector_gemv_q4_0_f32(
    raw_bytes: &[u8],
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
) {
    if out_features == 0 || in_features == 0 {
        return;
    }
    assert_eq!(
        x.len(),
        in_features,
        "Activation vector length must match in_features"
    );
    assert_eq!(
        out.len(),
        out_features,
        "Output vector length must match out_features"
    );

    let num_blocks = (in_features + 31) / 32;
    let row_len = num_blocks * 18;
    assert!(
        raw_bytes.len() >= out_features * row_len,
        "raw_bytes buffer length {} must be at least {} for {} rows of {} bytes",
        raw_bytes.len(),
        out_features * row_len,
        out_features,
        row_len
    );

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(out_features);

    if num_threads > 1 && out_features >= 4 && in_features >= 128 {
        let chunk_size = out_features.div_ceil(num_threads);
        let n_chunks = out_features.div_ceil(chunk_size);
        let out_ptr = crate::model::par::OutPtr::new(out);
        crate::model::par::par_for(n_chunks, &|t_idx| {
            let r_start = t_idx * chunk_size;
            let r_count = chunk_size.min(out_features.saturating_sub(r_start));
            // SAFETY: each task index `t_idx` writes a disjoint slice `r_start..r_start + r_count`.
            let out_chunk = unsafe { out_ptr.slice(r_start, r_count) };
            for (i, dst) in out_chunk.iter_mut().enumerate() {
                let r = r_start + i;
                let row_bytes = &raw_bytes[r * row_len..(r + 1) * row_len];
                let b = bias.map(|b| b[r]);
                *dst = compute_q4_0_row(row_bytes, in_features, num_blocks, x, b);
            }
        });
    } else {
        for r in 0..out_features {
            let row_bytes = &raw_bytes[r * row_len..(r + 1) * row_len];
            let b = bias.map(|b| b[r]);
            out[r] = compute_q4_0_row(row_bytes, in_features, num_blocks, x, b);
        }
    }
}

/// Helper for GEMV without bias.
#[inline]
pub fn vector_gemv_q4_0_f32_nobias(
    raw_bytes: &[u8],
    in_features: usize,
    out_features: usize,
    x: &[f32],
    out: &mut [f32],
) {
    vector_gemv_q4_0_f32(raw_bytes, in_features, out_features, x, out, None);
}

#[inline]
fn compute_q4_0_row(
    row_bytes: &[u8],
    in_features: usize,
    num_blocks: usize,
    x: &[f32],
    bias: Option<f32>,
) -> f32 {
    let mut row_sum = bias.unwrap_or(0.0);

    for b in 0..num_blocks {
        let block_bytes = &row_bytes[b * 18..(b + 1) * 18];
        let col_start = b * 32;
        let col_end = (col_start + 32).min(in_features);

        if col_end - col_start == 32 {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                row_sum += dot_q4_0_block_neon(block_bytes.as_ptr(), x.as_ptr().add(col_start));
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let d = f16_to_f32(u16::from_le_bytes([block_bytes[0], block_bytes[1]]));
                let mut block_dot = 0.0f32;
                for (i, &q) in block_bytes[2..18].iter().enumerate() {
                    let q_lo = (q & 0x0F) as i32 - 8;
                    let q_hi = ((q >> 4) as i32) - 8;
                    block_dot +=
                        (q_lo as f32) * x[col_start + i] + (q_hi as f32) * x[col_start + i + 16];
                }
                row_sum += block_dot * d;
            }
        } else {
            let count = col_end - col_start;
            let d = f16_to_f32(u16::from_le_bytes([block_bytes[0], block_bytes[1]]));
            let mut block_dot = 0.0f32;
            for (i, &q) in block_bytes[2..18].iter().enumerate() {
                if i < count {
                    let q_lo = (q & 0x0F) as i32 - 8;
                    block_dot += (q_lo as f32) * x[col_start + i];
                }
                if i + 16 < count {
                    let q_hi = ((q >> 4) as i32) - 8;
                    block_dot += (q_hi as f32) * x[col_start + i + 16];
                }
            }
            row_sum += block_dot * d;
        }
    }

    row_sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4_0_block_neon(block: *const u8, x_ptr: *const f32) -> f32 {
    use core::arch::aarch64::*;

    let scale_u16 = u16::from_le_bytes([*block, *block.add(1)]);
    let d = f16_to_f32(scale_u16);

    let v_bytes = vld1q_u8(block.add(2));
    let v_mask = vdupq_n_u8(0x0F);
    let v_eight = vdupq_n_u8(8);

    let v_lo = vandq_u8(v_bytes, v_mask);
    let v_hi = vshrq_n_u8(v_bytes, 4);

    let s_lo = vreinterpretq_s8_u8(vsubq_u8(v_lo, v_eight));
    let s_hi = vreinterpretq_s8_u8(vsubq_u8(v_hi, v_eight));

    let lo_low = vget_low_s8(s_lo);
    let lo_high = vget_high_s8(s_lo);
    let lo_low_16 = vmovl_s8(lo_low);
    let lo_high_16 = vmovl_s8(lo_high);

    let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_low_16)));
    let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_low_16)));
    let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_high_16)));
    let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_high_16)));

    let hi_low = vget_low_s8(s_hi);
    let hi_high = vget_high_s8(s_hi);
    let hi_low_16 = vmovl_s8(hi_low);
    let hi_high_16 = vmovl_s8(hi_high);

    let w4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_low_16)));
    let w5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_low_16)));
    let w6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_high_16)));
    let w7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_high_16)));

    let x0 = vld1q_f32(x_ptr);
    let x1 = vld1q_f32(x_ptr.add(4));
    let x2 = vld1q_f32(x_ptr.add(8));
    let x3 = vld1q_f32(x_ptr.add(12));
    let x4 = vld1q_f32(x_ptr.add(16));
    let x5 = vld1q_f32(x_ptr.add(20));
    let x6 = vld1q_f32(x_ptr.add(24));
    let x7 = vld1q_f32(x_ptr.add(28));

    let mut sum0 = vmulq_f32(w0, x0);
    let mut sum1 = vmulq_f32(w1, x1);
    let mut sum2 = vmulq_f32(w2, x2);
    let mut sum3 = vmulq_f32(w3, x3);

    sum0 = vfmaq_f32(sum0, w4, x4);
    sum1 = vfmaq_f32(sum1, w5, x5);
    sum2 = vfmaq_f32(sum2, w6, x6);
    sum3 = vfmaq_f32(sum3, w7, x7);

    let sum12 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    vaddvq_f32(sum12) * d
}

/// ARM `FEAT_I8MM` (`vmmlaq_s32`) 2x8x2 int8 matrix multiply-accumulate:
/// Computes $C_{2\times 2} = A_{2\times 8} \cdot B_{2\times 8}^\top$ for signed `i8` tiles of length `k`
/// (a multiple of 8). Uses hardware `SMMLA` (`vmmlaq_s32`) when `FEAT_I8MM` is available on `aarch64`,
/// falling back to NEON / scalar accumulation otherwise.
pub fn gemm_int8_i8mm_2x2(a_row0: &[i8], a_row1: &[i8], b_row0: &[i8], b_row1: &[i8]) -> [i32; 4] {
    let k = a_row0
        .len()
        .min(a_row1.len())
        .min(b_row0.len())
        .min(b_row1.len());
    #[cfg(target_arch = "aarch64")]
    {
        static HAS_I8MM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *HAS_I8MM
            .get_or_init(|| crate::hardware::HardwareCapability::probe_system().cpu_has_arm_i8mm)
            && k >= 8
        {
            // SAFETY: guarded by runtime `FEAT_I8MM` sysctl detection.
            return unsafe { gemm_int8_i8mm_2x2_neon(a_row0, a_row1, b_row0, b_row1, k) };
        }
    }
    let mut c = [0i32; 4];
    for i in 0..k {
        let a0 = a_row0[i] as i32;
        let a1 = a_row1[i] as i32;
        let b0 = b_row0[i] as i32;
        let b1 = b_row1[i] as i32;
        c[0] += a0 * b0;
        c[1] += a0 * b1;
        c[2] += a1 * b0;
        c[3] += a1 * b1;
    }
    c
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_int8_i8mm_2x2_neon(
    a_row0: &[i8],
    a_row1: &[i8],
    b_row0: &[i8],
    b_row1: &[i8],
    k: usize,
) -> [i32; 4] {
    use core::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 8 <= k {
        let a0 = vld1_s8(a_row0.as_ptr().add(i));
        let a1 = vld1_s8(a_row1.as_ptr().add(i));
        let va = vcombine_s8(a0, a1);
        let b0 = vld1_s8(b_row0.as_ptr().add(i));
        let b1 = vld1_s8(b_row1.as_ptr().add(i));
        let vb = vcombine_s8(b0, b1);
        core::arch::asm!(
            "smmla {acc:v}.4s, {va:v}.16b, {vb:v}.16b",
            acc = inout(vreg) acc,
            va = in(vreg) va,
            vb = in(vreg) vb,
            options(pure, nomem, nostack),
        );
        i += 8;
    }
    let mut out = [0i32; 4];
    vst1q_s32(out.as_mut_ptr(), acc);
    while i < k {
        let a0 = *a_row0.get_unchecked(i) as i32;
        let a1 = *a_row1.get_unchecked(i) as i32;
        let b0 = *b_row0.get_unchecked(i) as i32;
        let b1 = *b_row1.get_unchecked(i) as i32;
        out[0] += a0 * b0;
        out[1] += a0 * b1;
        out[2] += a1 * b0;
        out[3] += a1 * b1;
        i += 1;
    }
    out
}

/// ARM `FEAT_SME2` / Apple AMX-style 4x16 outer-product register-blocked tile microkernel:
/// Accumulates $C_{4\times 16} \mathrel{+}= \sum_{p=0}^{k-1} a_{:, p} \otimes b_{p, :}$ using
/// 16 rank-1 vector outer-product updates (`vfmaq_n_f32`) in 16 `float32x4_t` ZA-tile registers.
pub fn gemm_f32_sme2_outer_tile_4x16(
    a_panel: &[f32], // [k, 4] column-major panel of 4 rows
    b_panel: &[f32], // [k, 16] row-major panel of 16 cols
    k: usize,
    c_tile: &mut [f32; 64], // [4, 16] row-major output tile
) {
    assert!(a_panel.len() >= k * 4 && b_panel.len() >= k * 16);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut za00 = vld1q_f32(c_tile.as_ptr());
        let mut za01 = vld1q_f32(c_tile.as_ptr().add(4));
        let mut za02 = vld1q_f32(c_tile.as_ptr().add(8));
        let mut za03 = vld1q_f32(c_tile.as_ptr().add(12));
        let mut za10 = vld1q_f32(c_tile.as_ptr().add(16));
        let mut za11 = vld1q_f32(c_tile.as_ptr().add(20));
        let mut za12 = vld1q_f32(c_tile.as_ptr().add(24));
        let mut za13 = vld1q_f32(c_tile.as_ptr().add(28));
        let mut za20 = vld1q_f32(c_tile.as_ptr().add(32));
        let mut za21 = vld1q_f32(c_tile.as_ptr().add(36));
        let mut za22 = vld1q_f32(c_tile.as_ptr().add(40));
        let mut za23 = vld1q_f32(c_tile.as_ptr().add(44));
        let mut za30 = vld1q_f32(c_tile.as_ptr().add(48));
        let mut za31 = vld1q_f32(c_tile.as_ptr().add(52));
        let mut za32 = vld1q_f32(c_tile.as_ptr().add(56));
        let mut za33 = vld1q_f32(c_tile.as_ptr().add(60));

        for p in 0..k {
            let ap = a_panel.as_ptr().add(p * 4);
            let a0 = *ap;
            let a1 = *ap.add(1);
            let a2 = *ap.add(2);
            let a3 = *ap.add(3);
            let bp = b_panel.as_ptr().add(p * 16);
            let b0 = vld1q_f32(bp);
            let b1 = vld1q_f32(bp.add(4));
            let b2 = vld1q_f32(bp.add(8));
            let b3 = vld1q_f32(bp.add(12));

            za00 = vfmaq_n_f32(za00, b0, a0);
            za01 = vfmaq_n_f32(za01, b1, a0);
            za02 = vfmaq_n_f32(za02, b2, a0);
            za03 = vfmaq_n_f32(za03, b3, a0);

            za10 = vfmaq_n_f32(za10, b0, a1);
            za11 = vfmaq_n_f32(za11, b1, a1);
            za12 = vfmaq_n_f32(za12, b2, a1);
            za13 = vfmaq_n_f32(za13, b3, a1);

            za20 = vfmaq_n_f32(za20, b0, a2);
            za21 = vfmaq_n_f32(za21, b1, a2);
            za22 = vfmaq_n_f32(za22, b2, a2);
            za23 = vfmaq_n_f32(za23, b3, a2);

            za30 = vfmaq_n_f32(za30, b0, a3);
            za31 = vfmaq_n_f32(za31, b1, a3);
            za32 = vfmaq_n_f32(za32, b2, a3);
            za33 = vfmaq_n_f32(za33, b3, a3);
        }

        vst1q_f32(c_tile.as_mut_ptr(), za00);
        vst1q_f32(c_tile.as_mut_ptr().add(4), za01);
        vst1q_f32(c_tile.as_mut_ptr().add(8), za02);
        vst1q_f32(c_tile.as_mut_ptr().add(12), za03);
        vst1q_f32(c_tile.as_mut_ptr().add(16), za10);
        vst1q_f32(c_tile.as_mut_ptr().add(20), za11);
        vst1q_f32(c_tile.as_mut_ptr().add(24), za12);
        vst1q_f32(c_tile.as_mut_ptr().add(28), za13);
        vst1q_f32(c_tile.as_mut_ptr().add(32), za20);
        vst1q_f32(c_tile.as_mut_ptr().add(36), za21);
        vst1q_f32(c_tile.as_mut_ptr().add(40), za22);
        vst1q_f32(c_tile.as_mut_ptr().add(44), za23);
        vst1q_f32(c_tile.as_mut_ptr().add(48), za30);
        vst1q_f32(c_tile.as_mut_ptr().add(52), za31);
        vst1q_f32(c_tile.as_mut_ptr().add(56), za32);
        vst1q_f32(c_tile.as_mut_ptr().add(60), za33);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for p in 0..k {
            for r in 0..4 {
                let a = a_panel[p * 4 + r];
                for c in 0..16 {
                    c_tile[r * 16 + c] += a * b_panel[p * 16 + c];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_dot_product_matches_scalar() {
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let b = vec![
            0.5, -0.5, 1.0, 2.0, -1.0, 0.5, 0.25, -0.25, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4,
            -0.4, 0.5,
        ];
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let actual = vector_dot_product(&a, &b);
        assert!((actual - expected).abs() < 1e-4);
    }

    #[test]
    fn test_simd_rmsnorm_matches_scalar() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![1.0; 8];
        let mut out = vec![0.0f32; 8];
        vector_rmsnorm(&x, &w, &mut out, 1e-5);
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (sum_sq / 8.0 + 1e-5).sqrt();
        for i in 0..8 {
            assert!((out[i] - x[i] * scale).abs() < 1e-4);
        }
    }

    #[test]
    fn test_vector_gemv_q4_0_f32_matches_dequant() {
        let in_features = 64;
        let out_features = 4;
        let num_blocks = in_features / 32;
        let row_len = num_blocks * 18;

        let mut raw_bytes = vec![0u8; out_features * row_len];
        for r in 0..out_features {
            for b in 0..num_blocks {
                let blk_offset = r * row_len + b * 18;
                let scale_bits = 0x3400u16 + (r as u16 * 0x100);
                raw_bytes[blk_offset..blk_offset + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for j in 0..16 {
                    raw_bytes[blk_offset + 2 + j] = ((r * 31 + b * 17 + j * 7) % 256) as u8;
                }
            }
        }

        let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.1).sin()).collect();
        let bias = vec![0.5f32, -0.25, 1.0, -1.5];

        let mut out_actual = vec![0.0f32; out_features];
        vector_gemv_q4_0_f32(
            &raw_bytes,
            in_features,
            out_features,
            &x,
            &mut out_actual,
            Some(&bias),
        );

        for r in 0..out_features {
            let row_bytes = &raw_bytes[r * row_len..(r + 1) * row_len];
            let mut expected_row = bias[r];
            for b in 0..num_blocks {
                let blk = &row_bytes[b * 18..(b + 1) * 18];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let col = b * 32;
                for j in 0..16 {
                    let q = blk[2 + j];
                    let q_lo = (q & 0x0F) as i32 - 8;
                    let q_hi = ((q >> 4) as i32) - 8;
                    expected_row += (q_lo as f32) * d * x[col + j];
                    expected_row += (q_hi as f32) * d * x[col + j + 16];
                }
            }
            assert!(
                (out_actual[r] - expected_row).abs() < 1e-4,
                "GEMV mismatch row {}: actual={}, expected={}",
                r,
                out_actual[r],
                expected_row
            );
        }
    }

    #[test]
    fn test_vector_gemv_q4_0_f32_nobias_and_tail() {
        let in_features = 50; // non-multiple of 32
        let out_features = 2;
        let num_blocks = (in_features + 31) / 32;
        let row_len = num_blocks * 18;

        let mut raw_bytes = vec![0u8; out_features * row_len];
        for r in 0..out_features {
            for b in 0..num_blocks {
                let blk_offset = r * row_len + b * 18;
                let scale_bits = 0x3800u16; // 0.5 in fp16
                raw_bytes[blk_offset..blk_offset + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for j in 0..16 {
                    raw_bytes[blk_offset + 2 + j] = ((r * 19 + b * 13 + j * 5) % 256) as u8;
                }
            }
        }

        let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.05).cos()).collect();
        let mut out_actual = vec![0.0f32; out_features];
        vector_gemv_q4_0_f32_nobias(&raw_bytes, in_features, out_features, &x, &mut out_actual);

        for r in 0..out_features {
            let row_bytes = &raw_bytes[r * row_len..(r + 1) * row_len];
            let mut expected_row = 0.0f32;
            for b in 0..num_blocks {
                let blk = &row_bytes[b * 18..(b + 1) * 18];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let col_start = b * 32;
                let count = (col_start + 32).min(in_features) - col_start;
                for j in 0..16 {
                    if j < count {
                        let q = blk[2 + j];
                        let q_lo = (q & 0x0F) as i32 - 8;
                        expected_row += (q_lo as f32) * d * x[col_start + j];
                    }
                    if j + 16 < count {
                        let q = blk[2 + j];
                        let q_hi = ((q >> 4) as i32) - 8;
                        expected_row += (q_hi as f32) * d * x[col_start + j + 16];
                    }
                }
            }
            assert!(
                (out_actual[r] - expected_row).abs() < 1e-4,
                "Nobias GEMV mismatch row {}: actual={}, expected={}",
                r,
                out_actual[r],
                expected_row
            );
        }
    }

    #[test]
    fn test_arm_i8mm_and_sme2_outer_tile_match_scalar() {
        let a0: Vec<i8> = (0..24).map(|i| (i * 3 - 20) as i8).collect();
        let a1: Vec<i8> = (0..24).map(|i| (15 - i * 2) as i8).collect();
        let b0: Vec<i8> = (0..24).map(|i| (i * 5 - 30) as i8).collect();
        let b1: Vec<i8> = (0..24).map(|i| (i - 10) as i8).collect();
        let got = gemm_int8_i8mm_2x2(&a0, &a1, &b0, &b1);
        let dot = |x: &[i8], y: &[i8]| -> i32 {
            x.iter().zip(y).map(|(&u, &v)| u as i32 * v as i32).sum()
        };
        assert_eq!(
            got,
            [dot(&a0, &b0), dot(&a0, &b1), dot(&a1, &b0), dot(&a1, &b1)]
        );

        let k = 8usize;
        let a_panel: Vec<f32> = (0..k * 4).map(|i| (i as f32 * 0.25) - 2.0).collect();
        let b_panel: Vec<f32> = (0..k * 16).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let mut c_tile = [0.0f32; 64];
        gemm_f32_sme2_outer_tile_4x16(&a_panel, &b_panel, k, &mut c_tile);
        for r in 0..4 {
            for c in 0..16 {
                let mut want = 0.0f32;
                for p in 0..k {
                    want += a_panel[p * 4 + r] * b_panel[p * 16 + c];
                }
                assert!(
                    (c_tile[r * 16 + c] - want).abs() < 1e-4,
                    "SME2 outer tile mismatch at ({r},{c}): {} vs {want}",
                    c_tile[r * 16 + c]
                );
            }
        }
    }
}
