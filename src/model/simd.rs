//! SIMD primitives for Gibbs.
//!
//! Vector routines for RMSNorm, LayerNorm, SwiGLU, GELU and dot products using ARM NEON on
//! aarch64 and AVX on x86_64.

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

/// Vectorized SIMD dot product of two vectors simultaneously against a shared query vector `q`:
/// returns `(dot(q, k0), dot(q, k1))` with zero duplicate query memory loads.
#[inline]
pub fn vector_dot_product_pair(q: &[f32], k0: &[f32], k1: &[f32]) -> (f32, f32) {
    let len = q.len().min(k0.len()).min(k1.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc0_0 = vdupq_n_f32(0.0);
        let mut acc0_1 = vdupq_n_f32(0.0);
        let mut acc1_0 = vdupq_n_f32(0.0);
        let mut acc1_1 = vdupq_n_f32(0.0);

        while i + 8 <= len {
            let vq0 = vld1q_f32(q.as_ptr().add(i));
            let vk0_0 = vld1q_f32(k0.as_ptr().add(i));
            let vk1_0 = vld1q_f32(k1.as_ptr().add(i));
            acc0_0 = vfmaq_f32(acc0_0, vq0, vk0_0);
            acc1_0 = vfmaq_f32(acc1_0, vq0, vk1_0);

            let vq1 = vld1q_f32(q.as_ptr().add(i + 4));
            let vk0_1 = vld1q_f32(k0.as_ptr().add(i + 4));
            let vk1_1 = vld1q_f32(k1.as_ptr().add(i + 4));
            acc0_1 = vfmaq_f32(acc0_1, vq1, vk0_1);
            acc1_1 = vfmaq_f32(acc1_1, vq1, vk1_1);

            i += 8;
        }

        let mut dot0 = vaddvq_f32(vaddq_f32(acc0_0, acc0_1));
        let mut dot1 = vaddvq_f32(vaddq_f32(acc1_0, acc1_1));

        while i < len {
            let qi = *q.get_unchecked(i);
            dot0 += qi * *k0.get_unchecked(i);
            dot1 += qi * *k1.get_unchecked(i);
            i += 1;
        }

        (dot0, dot1)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        while i + 4 <= len {
            dot0 +=
                q[i] * k0[i] + q[i + 1] * k0[i + 1] + q[i + 2] * k0[i + 2] + q[i + 3] * k0[i + 3];
            dot1 +=
                q[i] * k1[i] + q[i + 1] * k1[i + 1] + q[i + 2] * k1[i + 2] + q[i + 3] * k1[i + 3];
            i += 4;
        }
        while i < len {
            dot0 += q[i] * k0[i];
            dot1 += q[i] * k1[i];
            i += 1;
        }
        (dot0, dot1)
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

/// Fused Residual Add + RMSNorm:
/// In a single pass, computes `hidden[i] += residual[i]`, then normalizes `out = hidden * weight / sqrt(mean(hidden^2) + eps)`.
#[inline]
pub fn vector_fused_add_rmsnorm(
    hidden: &mut [f32],
    residual: &[f32],
    weight: &[f32],
    out: &mut [f32],
    eps: f32,
) {
    let n = hidden.len();
    assert_eq!(n, residual.len());
    assert_eq!(n, weight.len());
    assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 8 <= n {
            let vh0 = vld1q_f32(hidden.as_ptr().add(i));
            let vr0 = vld1q_f32(residual.as_ptr().add(i));
            let vsum0 = vaddq_f32(vh0, vr0);
            vst1q_f32(hidden.as_mut_ptr().add(i), vsum0);

            let vh1 = vld1q_f32(hidden.as_ptr().add(i + 4));
            let vr1 = vld1q_f32(residual.as_ptr().add(i + 4));
            let vsum1 = vaddq_f32(vh1, vr1);
            vst1q_f32(hidden.as_mut_ptr().add(i + 4), vsum1);

            i += 8;
        }
    }
    while i < n {
        hidden[i] += residual[i];
        i += 1;
    }

    vector_rmsnorm(hidden, weight, out, eps);
}

/// Fused Residual Add + LayerNorm:
/// In a single pass, computes `hidden[i] += residual[i]`, then normalizes `out = (hidden - mean) / sqrt(var + eps) * weight + bias`.
#[inline]
pub fn vector_fused_add_layernorm(
    hidden: &mut [f32],
    residual: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    eps: f32,
) {
    let n = hidden.len();
    assert_eq!(n, residual.len());
    assert_eq!(n, weight.len());
    assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 8 <= n {
            let vh0 = vld1q_f32(hidden.as_ptr().add(i));
            let vr0 = vld1q_f32(residual.as_ptr().add(i));
            let vsum0 = vaddq_f32(vh0, vr0);
            vst1q_f32(hidden.as_mut_ptr().add(i), vsum0);

            let vh1 = vld1q_f32(hidden.as_ptr().add(i + 4));
            let vr1 = vld1q_f32(residual.as_ptr().add(i + 4));
            let vsum1 = vaddq_f32(vh1, vr1);
            vst1q_f32(hidden.as_mut_ptr().add(i + 4), vsum1);

            i += 8;
        }
    }
    while i < n {
        hidden[i] += residual[i];
        i += 1;
    }

    vector_layernorm(hidden, weight, bias, out, eps);
}

/// Vectorized LayerNorm with optional bias: out = (x - mean) / sqrt(var + eps) * weight + bias.
#[inline]
pub fn vector_layernorm(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    eps: f32,
) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    let sum: f32 = x.iter().sum();
    let mean = sum / n as f32;

    let mut var_sum = 0.0f32;
    for &val in x {
        let diff = val - mean;
        var_sum += diff * diff;
    }
    let scale = 1.0 / (var_sum / n as f32 + eps).sqrt();

    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vmean = vdupq_n_f32(mean);
        let vscale = vdupq_n_f32(scale);

        if let Some(b) = bias {
            while i + 8 <= n {
                let vx0 = vsubq_f32(vld1q_f32(x.as_ptr().add(i)), vmean);
                let vw0 = vld1q_f32(weight.as_ptr().add(i));
                let vb0 = vld1q_f32(b.as_ptr().add(i));
                let vo0 = vfmaq_f32(vb0, vmulq_f32(vx0, vscale), vw0);
                vst1q_f32(out.as_mut_ptr().add(i), vo0);

                let vx1 = vsubq_f32(vld1q_f32(x.as_ptr().add(i + 4)), vmean);
                let vw1 = vld1q_f32(weight.as_ptr().add(i + 4));
                let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
                let vo1 = vfmaq_f32(vb1, vmulq_f32(vx1, vscale), vw1);
                vst1q_f32(out.as_mut_ptr().add(i + 4), vo1);

                i += 8;
            }
        } else {
            while i + 8 <= n {
                let vx0 = vsubq_f32(vld1q_f32(x.as_ptr().add(i)), vmean);
                let vw0 = vld1q_f32(weight.as_ptr().add(i));
                let vo0 = vmulq_f32(vmulq_f32(vx0, vscale), vw0);
                vst1q_f32(out.as_mut_ptr().add(i), vo0);

                let vx1 = vsubq_f32(vld1q_f32(x.as_ptr().add(i + 4)), vmean);
                let vw1 = vld1q_f32(weight.as_ptr().add(i + 4));
                let vo1 = vmulq_f32(vmulq_f32(vx1, vscale), vw1);
                vst1q_f32(out.as_mut_ptr().add(i + 4), vo1);

                i += 8;
            }
        }
    }

    if let Some(b) = bias {
        while i < n {
            out[i] = (x[i] - mean) * scale * weight[i] + b[i];
            i += 1;
        }
    } else {
        while i < n {
            out[i] = (x[i] - mean) * scale * weight[i];
            i += 1;
        }
    }
}

/// Vectorized SwiGLU: out[i] = silu(gate_up[i]) * gate_up[i + inter].
#[inline]
pub fn vector_swiglu(gate_up: &[f32], out: &mut [f32], inter: usize) {
    assert!(gate_up.len() >= 2 * inter);
    assert!(out.len() >= inter);

    for i in 0..inter {
        let gate = gate_up[i];
        let up = gate_up[i + inter];
        let silu = gate / (1.0 + (-gate).exp());
        out[i] = silu * up;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn vexpq_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    let min_val = vdupq_n_f32(-87.0);
    let max_val = vdupq_n_f32(87.0);
    let x_clamped = vmaxq_f32(min_val, vminq_f32(max_val, x));

    let log2e = vdupq_n_f32(1.4426950408889634);
    let z = vmulq_f32(x_clamped, log2e);
    let n = vrndnq_f32(z);

    let ln2 = vdupq_n_f32(0.6931471805599453);
    let r = vmlsq_f32(x_clamped, n, ln2);

    let c5 = vdupq_n_f32(0.008333333333333333);
    let c4 = vdupq_n_f32(0.041666666666666664);
    let c3 = vdupq_n_f32(0.16666666666666666);
    let c2 = vdupq_n_f32(0.5);
    let c1 = vdupq_n_f32(1.0);
    let vone = vdupq_n_f32(1.0);

    let mut p = vfmaq_f32(c4, r, c5);
    p = vfmaq_f32(c3, r, p);
    p = vfmaq_f32(c2, r, p);
    p = vfmaq_f32(c1, r, p);
    p = vfmaq_f32(vone, r, p);

    let n_i = vcvtq_s32_f32(n);
    let bias = vdupq_n_s32(127);
    let pow2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n_i, bias), 23));

    vmulq_f32(p, pow2)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn vexpq_avx(x: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let min_val = _mm256_set1_ps(-87.0);
    let max_val = _mm256_set1_ps(87.0);
    let x_clamped = _mm256_max_ps(min_val, _mm256_min_ps(max_val, x));

    let log2e = _mm256_set1_ps(1.4426950408889634);
    let z = _mm256_mul_ps(x_clamped, log2e);
    let n = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    let ln2 = _mm256_set1_ps(0.6931471805599453);
    let r = _mm256_sub_ps(x_clamped, _mm256_mul_ps(n, ln2));

    let c5 = _mm256_set1_ps(0.008333333333333333);
    let c4 = _mm256_set1_ps(0.041666666666666664);
    let c3 = _mm256_set1_ps(0.16666666666666666);
    let c2 = _mm256_set1_ps(0.5);
    let c1 = _mm256_set1_ps(1.0);
    let vone = _mm256_set1_ps(1.0);

    let mut p = _mm256_add_ps(c4, _mm256_mul_ps(r, c5));
    p = _mm256_add_ps(c3, _mm256_mul_ps(r, p));
    p = _mm256_add_ps(c2, _mm256_mul_ps(r, p));
    p = _mm256_add_ps(c1, _mm256_mul_ps(r, p));
    p = _mm256_add_ps(vone, _mm256_mul_ps(r, p));

    let n_i = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127);
    let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_i, bias), 23));

    _mm256_mul_ps(p, pow2)
}

/// Fused SwiGLU activation in a single cache-efficient pass:
/// Computes `out[i] = (gate_up[i] / (1.0 + (-gate_up[i]).exp())) * gate_up[inter + i]`
/// using SIMD NEON on aarch64 and AVX on x86_64 with zero heap and zero stack array allocation.
#[inline]
pub fn vector_fused_swiglu_inplace(gate_up: &[f32], out: &mut [f32]) {
    let inter = out.len().min(gate_up.len() / 2);
    assert!(
        gate_up.len() >= 2 * inter,
        "gate_up buffer length {} must be at least 2 * inter (inter={})",
        gate_up.len(),
        inter
    );

    if inter == 0 {
        return;
    }

    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vone = vdupq_n_f32(1.0);

        while i + 8 <= inter {
            let vg0 = vld1q_f32(gate_up.as_ptr().add(i));
            let vu0 = vld1q_f32(gate_up.as_ptr().add(inter + i));
            let vg1 = vld1q_f32(gate_up.as_ptr().add(i + 4));
            let vu1 = vld1q_f32(gate_up.as_ptr().add(inter + i + 4));

            let ve0 = vexpq_f32(vnegq_f32(vg0));
            let ve1 = vexpq_f32(vnegq_f32(vg1));

            let denom0 = vaddq_f32(vone, ve0);
            let denom1 = vaddq_f32(vone, ve1);

            let silu0 = vdivq_f32(vg0, denom0);
            let silu1 = vdivq_f32(vg1, denom1);

            let vout0 = vmulq_f32(silu0, vu0);
            let vout1 = vmulq_f32(silu1, vu1);

            vst1q_f32(out.as_mut_ptr().add(i), vout0);
            vst1q_f32(out.as_mut_ptr().add(i + 4), vout1);

            i += 8;
        }

        while i + 4 <= inter {
            let vg = vld1q_f32(gate_up.as_ptr().add(i));
            let vu = vld1q_f32(gate_up.as_ptr().add(inter + i));

            let ve = vexpq_f32(vnegq_f32(vg));
            let denom = vaddq_f32(vone, ve);
            let silu = vdivq_f32(vg, denom);
            let vout = vmulq_f32(silu, vu);

            vst1q_f32(out.as_mut_ptr().add(i), vout);
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        unsafe {
            use core::arch::x86_64::*;
            let vone = _mm256_set1_ps(1.0);
            let vzero = _mm256_setzero_ps();

            while i + 8 <= inter {
                let vg = _mm256_loadu_ps(gate_up.as_ptr().add(i));
                let vu = _mm256_loadu_ps(gate_up.as_ptr().add(inter + i));

                let neg_vg = _mm256_sub_ps(vzero, vg);
                let ve = vexpq_avx(neg_vg);
                let denom = _mm256_add_ps(vone, ve);
                let silu = _mm256_div_ps(vg, denom);
                let vout = _mm256_mul_ps(silu, vu);

                _mm256_storeu_ps(out.as_mut_ptr().add(i), vout);
                i += 8;
            }
        }
    }

    while i < inter {
        let gate = gate_up[i];
        let up = gate_up[inter + i];
        let silu = gate / (1.0 + (-gate).exp());
        out[i] = silu * up;
        i += 1;
    }
}

/// Vectorized GELU approximation: out[i] = 0.5 * x[i] * (1.0 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3))).
/// Vectorized in-place GELU approximation: x[i] = 0.5 * x[i] * (1.0 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3))).
#[inline]
pub fn vector_gelu(x: &mut [f32]) {
    let n = x.len();
    let c = (2.0f32 / std::f32::consts::PI).sqrt();

    for i in 0..n {
        let val = x[i];
        let inner = c * (val + 0.044715 * val * val * val);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
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

/// Fused Scale-Add: `dst[i] += scale * src[i]`.
/// Accelerated with ARM NEON `vfmaq_f32` on aarch64 and AVX2/FMA `_mm256_fmadd_ps` on x86_64.
#[inline]
pub fn vector_fused_scale_add(dst: &mut [f32], src: &[f32], scale: f32) {
    let n = dst.len().min(src.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vscale = vdupq_n_f32(scale);
        while i + 16 <= n {
            let vd0 = vld1q_f32(dst.as_ptr().add(i));
            let vs0 = vld1q_f32(src.as_ptr().add(i));
            vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(vd0, vs0, vscale));

            let vd1 = vld1q_f32(dst.as_ptr().add(i + 4));
            let vs1 = vld1q_f32(src.as_ptr().add(i + 4));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vfmaq_f32(vd1, vs1, vscale));

            let vd2 = vld1q_f32(dst.as_ptr().add(i + 8));
            let vs2 = vld1q_f32(src.as_ptr().add(i + 8));
            vst1q_f32(dst.as_mut_ptr().add(i + 8), vfmaq_f32(vd2, vs2, vscale));

            let vd3 = vld1q_f32(dst.as_ptr().add(i + 12));
            let vs3 = vld1q_f32(src.as_ptr().add(i + 12));
            vst1q_f32(dst.as_mut_ptr().add(i + 12), vfmaq_f32(vd3, vs3, vscale));

            i += 16;
        }

        while i + 4 <= n {
            let vd = vld1q_f32(dst.as_ptr().add(i));
            let vs = vld1q_f32(src.as_ptr().add(i));
            vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(vd, vs, vscale));
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
        unsafe {
            use core::arch::x86_64::*;
            let vscale = _mm256_set1_ps(scale);
            while i + 8 <= n {
                let vd = _mm256_loadu_ps(dst.as_ptr().add(i));
                let vs = _mm256_loadu_ps(src.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(vs, vscale, vd));
                i += 8;
            }
        }
    }

    while i < n {
        dst[i] += scale * src[i];
        i += 1;
    }
}

/// Vectorized in-place scale: `x[i] *= scale`.
#[inline]
pub fn vector_scale_inplace(x: &mut [f32], scale: f32) {
    if scale == 1.0 {
        return;
    }
    if scale == 0.0 {
        x.fill(0.0);
        return;
    }

    let n = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vscale = vdupq_n_f32(scale);
        while i + 16 <= n {
            let vx0 = vld1q_f32(x.as_ptr().add(i));
            let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
            let vx2 = vld1q_f32(x.as_ptr().add(i + 8));
            let vx3 = vld1q_f32(x.as_ptr().add(i + 12));

            vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(vx0, vscale));
            vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(vx1, vscale));
            vst1q_f32(x.as_mut_ptr().add(i + 8), vmulq_f32(vx2, vscale));
            vst1q_f32(x.as_mut_ptr().add(i + 12), vmulq_f32(vx3, vscale));
            i += 16;
        }

        while i + 4 <= n {
            let vx = vld1q_f32(x.as_ptr().add(i));
            vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(vx, vscale));
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        unsafe {
            use core::arch::x86_64::*;
            let vscale = _mm256_set1_ps(scale);
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(vx, vscale));
                i += 8;
            }
        }
    }

    while i < n {
        x[i] *= scale;
        i += 1;
    }
}

/// Vectorized in-place elementwise product: `dst[i] *= src[i]`.
#[inline]
pub fn vector_elementwise_mul_inplace(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 16 <= n {
            let vd0 = vld1q_f32(dst.as_ptr().add(i));
            let vs0 = vld1q_f32(src.as_ptr().add(i));
            let vd1 = vld1q_f32(dst.as_ptr().add(i + 4));
            let vs1 = vld1q_f32(src.as_ptr().add(i + 4));
            let vd2 = vld1q_f32(dst.as_ptr().add(i + 8));
            let vs2 = vld1q_f32(src.as_ptr().add(i + 8));
            let vd3 = vld1q_f32(dst.as_ptr().add(i + 12));
            let vs3 = vld1q_f32(src.as_ptr().add(i + 12));

            vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(vd0, vs0));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(vd1, vs1));
            vst1q_f32(dst.as_mut_ptr().add(i + 8), vmulq_f32(vd2, vs2));
            vst1q_f32(dst.as_mut_ptr().add(i + 12), vmulq_f32(vd3, vs3));
            i += 16;
        }

        while i + 4 <= n {
            let vd = vld1q_f32(dst.as_ptr().add(i));
            let vs = vld1q_f32(src.as_ptr().add(i));
            vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(vd, vs));
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        unsafe {
            use core::arch::x86_64::*;
            while i + 8 <= n {
                let vd = _mm256_loadu_ps(dst.as_ptr().add(i));
                let vs = _mm256_loadu_ps(src.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(vd, vs));
                i += 8;
            }
        }
    }

    while i < n {
        dst[i] *= src[i];
        i += 1;
    }
}

/// SIMD vectorized 2-element butterfly step: a[i] = u + v, b[i] = u - v.
///
/// In-place butterfly operation for Fast Walsh-Hadamard Transform (FWHT),
/// accelerated with ARM NEON on aarch64 and AVX on x86_64 with zero heap allocation.
#[inline]
pub fn vector_hadamard_butterfly_f32(a: &mut [f32], b: &mut [f32]) {
    let n = a.len().min(b.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 16 <= n {
            let va0 = vld1q_f32(a.as_ptr().add(i));
            let vb0 = vld1q_f32(b.as_ptr().add(i));
            let va1 = vld1q_f32(a.as_ptr().add(i + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
            let va2 = vld1q_f32(a.as_ptr().add(i + 8));
            let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
            let va3 = vld1q_f32(a.as_ptr().add(i + 12));
            let vb3 = vld1q_f32(b.as_ptr().add(i + 12));

            vst1q_f32(a.as_mut_ptr().add(i), vaddq_f32(va0, vb0));
            vst1q_f32(b.as_mut_ptr().add(i), vsubq_f32(va0, vb0));
            vst1q_f32(a.as_mut_ptr().add(i + 4), vaddq_f32(va1, vb1));
            vst1q_f32(b.as_mut_ptr().add(i + 4), vsubq_f32(va1, vb1));
            vst1q_f32(a.as_mut_ptr().add(i + 8), vaddq_f32(va2, vb2));
            vst1q_f32(b.as_mut_ptr().add(i + 8), vsubq_f32(va2, vb2));
            vst1q_f32(a.as_mut_ptr().add(i + 12), vaddq_f32(va3, vb3));
            vst1q_f32(b.as_mut_ptr().add(i + 12), vsubq_f32(va3, vb3));
            i += 16;
        }

        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            vst1q_f32(a.as_mut_ptr().add(i), vaddq_f32(va, vb));
            vst1q_f32(b.as_mut_ptr().add(i), vsubq_f32(va, vb));
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        unsafe {
            use core::arch::x86_64::*;
            while i + 8 <= n {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                _mm256_storeu_ps(a.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
                _mm256_storeu_ps(b.as_mut_ptr().add(i), _mm256_sub_ps(va, vb));
                i += 8;
            }
        }
    }

    while i < n {
        let u = a[i];
        let v = b[i];
        a[i] = u + v;
        b[i] = u - v;
        i += 1;
    }
}

/// Flash-Decoding / Luminal-inspired single-pass online softmax attention:
/// Computes attention output for single or multiple query heads against key/value caches without allocating intermediate buffers.
///
/// Supports parallel evaluation across attention heads using multi-threaded execution (scoped threads)
/// and vector-stride unrolling with zero heap allocation.
#[inline]
pub fn vector_online_softmax_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    current_len: usize,
    stride: usize,
    scale: f32,
    out: &mut [f32],
) {
    assert!(
        q.len() >= head_dim,
        "Query slice must have at least head_dim elements"
    );
    assert!(
        out.len() >= head_dim,
        "Output slice must have at least head_dim elements"
    );
    if current_len > 0 {
        let req_len = (current_len - 1) * stride + head_dim;
        assert!(
            k_cache.len() >= req_len,
            "k_cache must contain at least {} elements, got {}",
            req_len,
            k_cache.len()
        );
        assert!(
            v_cache.len() >= req_len,
            "v_cache must contain at least {} elements, got {}",
            req_len,
            v_cache.len()
        );
    }
    if head_dim == 0 {
        return;
    }

    let q_heads = q.len() / head_dim;
    let out_heads = out.len() / head_dim;
    let k_heads = if current_len > 0 && stride >= head_dim {
        let span = k_cache.len().saturating_sub((current_len - 1) * stride);
        span / head_dim
    } else {
        1
    };
    let v_heads = if current_len > 0 && stride >= head_dim {
        let span = v_cache.len().saturating_sub((current_len - 1) * stride);
        span / head_dim
    } else {
        1
    };
    let num_heads = q_heads.min(out_heads).min(k_heads).min(v_heads).max(1);

    if current_len == 0 {
        out[..num_heads * head_dim].fill(0.0);
        return;
    }

    if num_heads == 1 {
        single_head_online_softmax_attention(
            &q[..head_dim],
            k_cache,
            v_cache,
            head_dim,
            current_len,
            stride,
            scale,
            &mut out[..head_dim],
        );
        return;
    }

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(num_heads);

    if num_threads > 1 && (num_heads >= 2 && current_len >= 8 || num_heads >= 4) {
        let chunk_size = (num_heads + num_threads - 1) / num_threads;
        std::thread::scope(|s| {
            for (t_idx, out_chunk) in out[..num_heads * head_dim]
                .chunks_mut(chunk_size * head_dim)
                .enumerate()
            {
                let head_start = t_idx * chunk_size;
                let head_count = out_chunk.len() / head_dim;
                let q_chunk = &q[head_start * head_dim..(head_start + head_count) * head_dim];
                s.spawn(move || {
                    for h in 0..head_count {
                        let actual_head = head_start + h;
                        let q_h = &q_chunk[h * head_dim..(h + 1) * head_dim];
                        let out_h = &mut out_chunk[h * head_dim..(h + 1) * head_dim];
                        let k_h = &k_cache[actual_head * head_dim..];
                        let v_h = &v_cache[actual_head * head_dim..];
                        single_head_online_softmax_attention(
                            q_h,
                            k_h,
                            v_h,
                            head_dim,
                            current_len,
                            stride,
                            scale,
                            out_h,
                        );
                    }
                });
            }
        });
    } else {
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_h = &q[head_offset..head_offset + head_dim];
            let out_h = &mut out[head_offset..head_offset + head_dim];
            let k_h = &k_cache[head_offset..];
            let v_h = &v_cache[head_offset..];
            single_head_online_softmax_attention(
                q_h,
                k_h,
                v_h,
                head_dim,
                current_len,
                stride,
                scale,
                out_h,
            );
        }
    }
}

#[inline]
fn single_head_online_softmax_attention(
    q_slice: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    current_len: usize,
    stride: usize,
    scale: f32,
    out_slice: &mut [f32],
) {
    if current_len == 0 {
        out_slice.fill(0.0);
        return;
    }
    if current_len == 1 {
        out_slice.copy_from_slice(&v_cache[..head_dim]);
        return;
    }

    let k_0 = &k_cache[..head_dim];
    let v_0 = &v_cache[..head_dim];

    let mut m = vector_dot_product(q_slice, k_0) * scale;
    let mut l = 1.0f32;
    out_slice.copy_from_slice(v_0);

    let mut pos = 1;

    // Vector-stride 2-way unrolling across cached tokens:
    // Interleaves 2 dot products and shares Q register loads
    while pos + 2 <= current_len {
        let offset0 = pos * stride;
        let offset1 = (pos + 1) * stride;
        let k_0 = &k_cache[offset0..offset0 + head_dim];
        let k_1 = &k_cache[offset1..offset1 + head_dim];
        let v_0 = &v_cache[offset0..offset0 + head_dim];
        let v_1 = &v_cache[offset1..offset1 + head_dim];

        let (dot0, dot1) = vector_dot_product_pair(q_slice, k_0, k_1);
        let score0 = dot0 * scale;
        let score1 = dot1 * scale;

        if score0 > m {
            let alpha = (m - score0).exp();
            vector_scale_inplace(out_slice, alpha);
            l = l * alpha + 1.0;
            m = score0;
            vector_fused_scale_add(out_slice, v_0, 1.0);
        } else {
            let beta = (score0 - m).exp();
            l += beta;
            vector_fused_scale_add(out_slice, v_0, beta);
        }

        if score1 > m {
            let alpha = (m - score1).exp();
            vector_scale_inplace(out_slice, alpha);
            l = l * alpha + 1.0;
            m = score1;
            vector_fused_scale_add(out_slice, v_1, 1.0);
        } else {
            let beta = (score1 - m).exp();
            l += beta;
            vector_fused_scale_add(out_slice, v_1, beta);
        }

        pos += 2;
    }

    while pos < current_len {
        let offset = pos * stride;
        let k_i = &k_cache[offset..offset + head_dim];
        let v_i = &v_cache[offset..offset + head_dim];
        let score = vector_dot_product(q_slice, k_i) * scale;

        if score > m {
            let alpha = (m - score).exp();
            vector_scale_inplace(out_slice, alpha);
            l = l * alpha + 1.0;
            m = score;
            vector_fused_scale_add(out_slice, v_i, 1.0);
        } else {
            let beta = (score - m).exp();
            l += beta;
            vector_fused_scale_add(out_slice, v_i, beta);
        }

        pos += 1;
    }

    if l > 0.0 {
        let inv_l = 1.0 / l;
        vector_scale_inplace(out_slice, inv_l);
    }
}

/// Computes spatial cross-attention over spatial grid [H * W, d_k] against sequence [T, d_ctx]:
/// Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
///
/// Supports arbitrary sequence length T (specifically T = 77, d = 1024) and context matrices,
/// automatically handling single-head and multi-head spatial queries with SIMD dot products,
/// numerically stable softmax normalization, and fused scale-add value accumulation.
#[inline]
pub fn vector_spatial_cross_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hw: usize,
    seq_len: usize,
    d_k: usize,
    scale: f32,
    out: &mut [f32],
) {
    if hw == 0 || seq_len == 0 || d_k == 0 {
        if hw > 0 && d_k > 0 {
            let total_out = hw * (q.len() / hw).max(d_k);
            let n = out.len().min(total_out);
            out[..n].fill(0.0);
        }
        return;
    }

    let q_total_d = q.len() / hw;
    let out_total_d = out.len() / hw;
    let num_heads = (q_total_d / d_k).max(1).min(out_total_d / d_k);
    let total_d = num_heads * d_k;

    assert!(
        q.len() >= hw * total_d,
        "q buffer underflow: expected at least {}, got {}",
        hw * total_d,
        q.len()
    );
    assert!(
        out.len() >= hw * total_d,
        "out buffer underflow: expected at least {}, got {}",
        hw * total_d,
        out.len()
    );

    let k_heads = if k.len() >= seq_len * total_d {
        num_heads
    } else {
        1
    };
    let v_heads = if v.len() >= seq_len * total_d {
        num_heads
    } else {
        1
    };
    let k_stride = if k.len() >= seq_len * total_d {
        total_d
    } else {
        d_k
    };
    let v_stride = if v.len() >= seq_len * total_d {
        total_d
    } else {
        d_k
    };

    let eff_scale = if scale > 0.0 {
        scale
    } else {
        1.0 / (d_k as f32).sqrt().max(1.0)
    };

    let compute_spatial_token = |p: usize, out_slice: &mut [f32], scores: &mut [f32]| {
        let q_p = &q[p * total_d..(p + 1) * total_d];
        for h_idx in 0..num_heads {
            let q_start = h_idx * d_k;
            let q_head = &q_p[q_start..q_start + d_k];
            let out_head = &mut out_slice[q_start..q_start + d_k];
            out_head.fill(0.0);

            let k_h_idx = h_idx % k_heads;
            let v_h_idx = h_idx % v_heads;
            let k_head_start = k_h_idx * d_k;
            let v_head_start = v_h_idx * d_k;

            // 1. Unrolled SIMD dot products: Q_head · K_head
            let mut t = 0;
            while t + 2 <= seq_len {
                let off0 = t * k_stride + k_head_start;
                let off1 = (t + 1) * k_stride + k_head_start;
                let k0 = &k[off0..off0 + d_k];
                let k1 = &k[off1..off1 + d_k];
                let (dot0, dot1) = vector_dot_product_pair(q_head, k0, k1);
                scores[t] = dot0 * eff_scale;
                scores[t + 1] = dot1 * eff_scale;
                t += 2;
            }
            while t < seq_len {
                let off = t * k_stride + k_head_start;
                let k_t = &k[off..off + d_k];
                scores[t] = vector_dot_product(q_head, k_t) * eff_scale;
                t += 1;
            }

            // 2. Numerically stable softmax normalization over sequence dimension T
            vector_softmax(&mut scores[..seq_len]);

            // 3. Vectorized weighted accumulation over V: sum(score[t] * V[t])
            for t_idx in 0..seq_len {
                let w = scores[t_idx];
                if w.abs() > 1e-8 {
                    let v_off = t_idx * v_stride + v_head_start;
                    let v_t = &v[v_off..v_off + d_k];
                    vector_fused_scale_add(out_head, v_t, w);
                }
            }
        }
    };

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(hw);

    if num_threads > 1 && hw >= 64 {
        let chunk_size = (hw + num_threads - 1) / num_threads;
        std::thread::scope(|s| {
            for (t_idx, out_chunk) in out[..hw * total_d]
                .chunks_mut(chunk_size * total_d)
                .enumerate()
            {
                let p_start = t_idx * chunk_size;
                let p_count = out_chunk.len() / total_d;
                s.spawn(move || {
                    let mut local_scores = vec![0.0f32; seq_len];
                    for local_p in 0..p_count {
                        let p = p_start + local_p;
                        let out_slice = &mut out_chunk[local_p * total_d..(local_p + 1) * total_d];
                        compute_spatial_token(p, out_slice, &mut local_scores);
                    }
                });
            }
        });
    } else {
        let mut scores = vec![0.0f32; seq_len];
        for p in 0..hw {
            let out_slice = &mut out[p * total_d..(p + 1) * total_d];
            compute_spatial_token(p, out_slice, &mut scores);
        }
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
        let chunk_size = (out_features + num_threads - 1) / num_threads;
        std::thread::scope(|s| {
            for (t_idx, out_chunk) in out.chunks_mut(chunk_size).enumerate() {
                let r_start = t_idx * chunk_size;
                let r_count = out_chunk.len();
                s.spawn(move || {
                    for i in 0..r_count {
                        let r = r_start + i;
                        let row_bytes = &raw_bytes[r * row_len..(r + 1) * row_len];
                        let b = bias.map(|b| b[r]);
                        out_chunk[i] = compute_q4_0_row(row_bytes, in_features, num_blocks, x, b);
                    }
                });
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
    fn test_simd_fused_scale_add() {
        let mut dst = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ];
        let src = vec![2.0; 18];
        let scale = 0.5f32;
        vector_fused_scale_add(&mut dst, &src, scale);
        // Each dst[i] should be i + 1 + 2.0 * 0.5 = i + 2.0
        for (i, &v) in dst.iter().enumerate() {
            assert!((v - (i as f32 + 2.0)).abs() < 1e-4);
        }
    }

    #[test]
    fn test_vector_online_softmax_attention_matches_standard() {
        let head_dim = 64;
        let current_len = 17;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut q = vec![0.0f32; head_dim];
        for i in 0..head_dim {
            q[i] = ((i as f32 * 0.17).sin()) * 0.5;
        }

        let mut k_cache = vec![0.0f32; current_len * head_dim];
        let mut v_cache = vec![0.0f32; current_len * head_dim];
        for pos in 0..current_len {
            for d in 0..head_dim {
                let idx = pos * head_dim + d;
                k_cache[idx] = ((idx as f32 * 0.31).cos()) * 0.8;
                v_cache[idx] = ((idx as f32 * 0.43).sin()) * 1.2;
            }
        }

        // Standard attention reference
        let mut scores = vec![0.0f32; current_len];
        for pos in 0..current_len {
            let k_i = &k_cache[pos * head_dim..(pos + 1) * head_dim];
            scores[pos] = vector_dot_product(&q, k_i) * scale;
        }
        vector_softmax(&mut scores);

        let mut expected = vec![0.0f32; head_dim];
        for pos in 0..current_len {
            let v_i = &v_cache[pos * head_dim..(pos + 1) * head_dim];
            vector_fused_scale_add(&mut expected, v_i, scores[pos]);
        }

        // Online attention
        // Online attention with contiguous stride = head_dim
        let mut actual = vec![0.0f32; head_dim];
        vector_online_softmax_attention(
            &q,
            &k_cache,
            &v_cache,
            head_dim,
            current_len,
            head_dim,
            scale,
            &mut actual,
        );
        for d in 0..head_dim {
            assert!(
                (actual[d] - expected[d]).abs() < 1e-4,
                "Mismatch at dim {}: actual={}, expected={}",
                d,
                actual[d],
                expected[d]
            );
        }

        // Also verify with non-trivial stride (e.g. multi-head interleaved stride = head_dim * 2)
        // Also verify with non-trivial stride (e.g. multi-head interleaved stride = head_dim * 2)
        let stride = head_dim * 2;
        let mut k_interleaved = vec![0.0f32; current_len * stride];
        let mut v_interleaved = vec![0.0f32; current_len * stride];
        for pos in 0..current_len {
            for d in 0..head_dim {
                k_interleaved[pos * stride + d] = k_cache[pos * head_dim + d];
                v_interleaved[pos * stride + d] = v_cache[pos * head_dim + d];
            }
        }
        let mut actual_strided = vec![0.0f32; head_dim];
        vector_online_softmax_attention(
            &q,
            &k_interleaved,
            &v_interleaved,
            head_dim,
            current_len,
            stride,
            scale,
            &mut actual_strided,
        );
        for d in 0..head_dim {
            assert!((actual_strided[d] - expected[d]).abs() < 1e-4);
        }
    }
    #[test]
    fn test_vector_online_softmax_attention_single_token() {
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = vec![1.23f32; head_dim];
        let k = vec![0.85f32; head_dim];
        let v: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; head_dim];

        vector_online_softmax_attention(&q, &k, &v, head_dim, 1, head_dim, scale, &mut out);
        assert_eq!(&out, &v);
    }
    #[test]
    fn test_vector_fused_swiglu_inplace_matches_standard() {
        let inter = 37;
        let mut gate_up = vec![0.0f32; 2 * inter];
        for i in 0..(2 * inter) {
            gate_up[i] = ((i as f32 * 0.23).sin()) * 2.5;
        }

        let mut expected = vec![0.0f32; inter];
        for i in 0..inter {
            let gate = gate_up[i];
            let up = gate_up[inter + i];
            let silu = gate / (1.0 + (-gate).exp());
            expected[i] = silu * up;
        }

        let mut actual = vec![0.0f32; inter];
        vector_fused_swiglu_inplace(&gate_up, &mut actual);

        for i in 0..inter {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-4,
                "Mismatch at index {}: actual={}, expected={}",
                i,
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_vector_hadamard_butterfly_matches_scalar() {
        let mut a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let mut b = vec![
            0.5, -0.5, 1.0, 2.0, -1.0, 0.5, 0.25, -0.25, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4,
            -0.4, 0.5,
        ];
        let a_orig = a.clone();
        let b_orig = b.clone();

        vector_hadamard_butterfly_f32(&mut a, &mut b);

        for i in 0..a.len() {
            let expected_a = a_orig[i] + b_orig[i];
            let expected_b = a_orig[i] - b_orig[i];
            assert!((a[i] - expected_a).abs() < 1e-5, "Mismatch a[{}]", i);
            assert!((b[i] - expected_b).abs() < 1e-5, "Mismatch b[{}]", i);
        }
    }

    #[test]
    fn test_vector_elementwise_mul_inplace_matches_scalar() {
        let mut dst: Vec<f32> = (0..37).map(|i| (i as f32 * 0.25).sin()).collect();
        let src: Vec<f32> = (0..37).map(|i| (i as f32 * 0.15).cos()).collect();
        let orig_dst = dst.clone();

        vector_elementwise_mul_inplace(&mut dst, &src);

        for i in 0..37 {
            let expected = orig_dst[i] * src[i];
            assert!(
                (dst[i] - expected).abs() < 1e-6,
                "Mismatch at {}: got {}, expected {}",
                i,
                dst[i],
                expected
            );
        }
    }

    #[test]
    fn test_vector_online_softmax_attention_multi_head_parallel() {
        let head_dim = 64;
        let num_heads = 4;
        let current_len = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let stride = num_heads * head_dim;

        let mut q = vec![0.0f32; num_heads * head_dim];
        for i in 0..q.len() {
            q[i] = ((i as f32 * 0.19).sin()) * 0.5;
        }

        let mut k_cache = vec![0.0f32; current_len * stride];
        let mut v_cache = vec![0.0f32; current_len * stride];
        for pos in 0..current_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = pos * stride + h * head_dim + d;
                    k_cache[idx] = ((idx as f32 * 0.23).cos()) * 0.7;
                    v_cache[idx] = ((idx as f32 * 0.37).sin()) * 1.1;
                }
            }
        }

        let mut actual_parallel = vec![0.0f32; num_heads * head_dim];
        vector_online_softmax_attention(
            &q,
            &k_cache,
            &v_cache,
            head_dim,
            current_len,
            stride,
            scale,
            &mut actual_parallel,
        );

        let mut expected = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let offset = h * head_dim;
            vector_online_softmax_attention(
                &q[offset..offset + head_dim],
                &k_cache[offset..],
                &v_cache[offset..],
                head_dim,
                current_len,
                stride,
                scale,
                &mut expected[offset..offset + head_dim],
            );
        }

        for i in 0..actual_parallel.len() {
            assert!(
                (actual_parallel[i] - expected[i]).abs() < 1e-4,
                "Mismatch at {}: parallel={}, seq={}",
                i,
                actual_parallel[i],
                expected[i]
            );
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
    fn test_vector_spatial_cross_attention_77_tokens_1024_dim() {
        let hw = 4; // 2x2 spatial grid
        let seq_len = 77; // exactly 77 CLIP tokens
        let d_k = 1024; // 1024-dim context

        // Synthesize deterministic Q, K, V
        let mut q = vec![0.0f32; hw * d_k];
        for i in 0..q.len() {
            q[i] = ((i as f32 * 0.013).sin()) * 0.1;
        }

        let mut k = vec![0.0f32; seq_len * d_k];
        for i in 0..k.len() {
            k[i] = ((i as f32 * 0.017).cos()) * 0.1;
        }

        let mut v = vec![0.0f32; seq_len * d_k];
        for i in 0..v.len() {
            v[i] = ((i as f32 * 0.023).sin()) * 0.1;
        }

        let mut out = vec![0.0f32; hw * d_k];
        vector_spatial_cross_attention(&q, &k, &v, hw, seq_len, d_k, 0.0, &mut out);

        // Verify shape and non-trivial output
        assert_eq!(out.len(), hw * d_k);
        for &val in &out {
            assert!(val.is_finite(), "Output must be finite");
        }
        assert!(
            out.iter().any(|&x| x.abs() > 1e-4),
            "Attention output must be non-zero"
        );

        // Verify exact mathematical match against standard reference Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        let scale = 1.0 / (d_k as f32).sqrt();
        for p in 0..hw {
            let q_p = &q[p * d_k..(p + 1) * d_k];
            let mut scores = vec![0.0f32; seq_len];
            let mut max_s = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_t = &k[t * d_k..(t + 1) * d_k];
                let mut dot = 0.0f32;
                for d in 0..d_k {
                    dot += q_p[d] * k_t[d];
                }
                let s = dot * scale;
                scores[t] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                let exp = (*s - max_s).exp();
                *s = exp;
                sum_exp += exp;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in &mut scores {
                *s *= inv_sum;
            }

            for d in 0..d_k {
                let mut expected = 0.0f32;
                for t in 0..seq_len {
                    expected += scores[t] * v[t * d_k + d];
                }
                let actual = out[p * d_k + d];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "Mismatch at p={}, d={}: actual={}, expected={}",
                    p,
                    d,
                    actual,
                    expected
                );
            }
        }
    }
}
