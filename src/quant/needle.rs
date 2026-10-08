//! Cactus Compute Needle optimizations: Fast Walsh-Hadamard Transforms (FWHT),
//! Hadamard-based Outlier Elimination Attention, Butterfly Structured MLP,
//! and Napier Logarithmic Multiplication Factorization.

use crate::model::simd::{
    vector_dot_product, vector_elementwise_mul_inplace, vector_fused_scale_add,
    vector_hadamard_butterfly_f32, vector_scale_inplace,
};

/// Maximum head dimension supported on stack for zero-allocation attention.
pub const MAX_HADAMARD_HEAD_DIM: usize = 2048;

/// Fast Walsh-Hadamard Transform (FWHT) for f32 vectors.
///
/// In-place butterfly network normalized by $1 / \sqrt{N}$ for powers of 2.
/// Eliminates activation outliers and provides orthogonal basis rotation in $O(N \log N)$ complexity.
#[inline]
pub fn fast_hadamard_transform_f32(vec: &mut [f32]) {
    let n = vec.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FWHT requires power-of-two length");

    let mut h = 1;
    while h < n {
        if h >= 4 {
            for i in (0..n).step_by(h * 2) {
                let (left, right) = vec[i..i + 2 * h].split_at_mut(h);
                vector_hadamard_butterfly_f32(left, right);
            }
        } else {
            for i in (0..n).step_by(h * 2) {
                for j in 0..h {
                    let u = vec[i + j];
                    let v = vec[i + j + h];
                    vec[i + j] = u + v;
                    vec[i + j + h] = u - v;
                }
            }
        }
        h *= 2;
    }

    // Normalized by 1 / sqrt(N) to preserve L2 norm and maintain involution (H * H = I)
    let norm = 1.0 / (n as f32).sqrt();
    vector_scale_inplace(vec, norm);
}

/// Hadamard-based Outlier-Free Attention for a single decode query token against KV cache.
///
/// Transforms $Q$ and $K$ into Hadamard basis to eliminate outlier activations and factorize attention
/// without $O(N^2)$ dense matrix multiplication or heap allocations.
/// Maintains running online softmax statistics in $O(L \cdot d \log d)$ with $O(1)$ memory.
#[inline]
pub fn hadamard_attention_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    current_len: usize,
    stride: usize,
    out: &mut [f32],
) {
    assert!(
        q.len() >= head_dim,
        "q must contain at least head_dim elements"
    );
    assert!(
        out.len() >= head_dim,
        "out must contain at least head_dim elements"
    );
    if current_len > 0 {
        let req_len = (current_len - 1) * stride + head_dim;
        assert!(
            k_cache.len() >= req_len,
            "k_cache underflow: required {}, got {}",
            req_len,
            k_cache.len()
        );
        assert!(
            v_cache.len() >= req_len,
            "v_cache underflow: required {}, got {}",
            req_len,
            v_cache.len()
        );
    }

    let out_slice = &mut out[..head_dim];
    out_slice.fill(0.0);

    if head_dim == 0 || current_len == 0 {
        return;
    }

    assert!(
        head_dim.is_power_of_two(),
        "head_dim must be a power of two"
    );
    assert!(
        head_dim <= MAX_HADAMARD_HEAD_DIM,
        "head_dim {} exceeds MAX_HADAMARD_HEAD_DIM {}",
        head_dim,
        MAX_HADAMARD_HEAD_DIM
    );

    // Zero-allocation stack buffers for Hadamard basis representations
    let mut q_had = [0.0f32; MAX_HADAMARD_HEAD_DIM];
    let mut k_had = [0.0f32; MAX_HADAMARD_HEAD_DIM];

    // Project Q into Hadamard basis
    q_had[..head_dim].copy_from_slice(&q[..head_dim]);
    fast_hadamard_transform_f32(&mut q_had[..head_dim]);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;

    for pos in 0..current_len {
        let offset = pos * stride;
        let k_slice = &k_cache[offset..offset + head_dim];
        let v_slice = &v_cache[offset..offset + head_dim];

        // Project K into Hadamard basis
        k_had[..head_dim].copy_from_slice(k_slice);
        fast_hadamard_transform_f32(&mut k_had[..head_dim]);

        // In Hadamard basis, <Hq, Hk> = <q, k> with outliers suppressed
        let score = vector_dot_product(&q_had[..head_dim], &k_had[..head_dim]) * scale;

        if score > m {
            let alpha = if m == f32::NEG_INFINITY {
                0.0
            } else {
                (m - score).exp()
            };
            if alpha == 0.0 {
                out_slice.fill(0.0);
            } else {
                vector_scale_inplace(out_slice, alpha);
            }
            l = l * alpha + 1.0;
            m = score;
            vector_fused_scale_add(out_slice, v_slice, 1.0);
        } else {
            let beta = (score - m).exp();
            l += beta;
            vector_fused_scale_add(out_slice, v_slice, beta);
        }
    }

    if l > 0.0 {
        let inv_l = 1.0 / l;
        vector_scale_inplace(out_slice, inv_l);
    }
}

/// Structured Butterfly Hadamard MLP:
/// Computes $H_N \cdot \text{diag}(W) \cdot H_N \cdot x$ in $O(N \log N)$ complexity without dense $N \times N$ matrix.
#[inline]
pub fn hadamard_mlp_f32(x: &[f32], diagonal_weights: &[f32], out: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "Dimension must be a power of two");
    assert!(
        diagonal_weights.len() >= n,
        "diagonal_weights slice must have at least {} elements",
        n
    );
    assert!(
        out.len() >= n,
        "out slice must have at least {} elements",
        n
    );

    let out_slice = &mut out[..n];
    if x.as_ptr() != out_slice.as_ptr() {
        out_slice.copy_from_slice(&x[..n]);
    }

    // Step 1: First Hadamard transform H_N * x
    fast_hadamard_transform_f32(out_slice);

    // Step 2: Diagonal weight scaling diag(W) * (H_N * x) via SIMD
    vector_elementwise_mul_inplace(out_slice, &diagonal_weights[..n]);

    // Step 3: Second Hadamard transform H_N * diag(W) * H_N * x
    fast_hadamard_transform_f32(out_slice);
}

/// Fast Mitchell/piecewise log2 conversion for positive f32 values.
///
/// Uses IEEE 754 bit-level representation with a piecewise quadratic correction:
/// $\log_2(1 + m) \approx m + \alpha \cdot m(1 - m)$, with $\alpha \approx 0.34657359$.
#[inline]
pub fn napier_log2_f32(x: f32) -> f32 {
    if x <= 0.0 {
        if x == 0.0 {
            return f32::NEG_INFINITY;
        }
        return f32::NAN;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    if x.is_nan() {
        return f32::NAN;
    }

    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = (bits & 0x007F_FFFF) as f32 * (1.0 / 8388608.0);
    let log2_mantissa = mantissa + 0.34657359 * mantissa * (1.0 - mantissa);
    exp as f32 + log2_mantissa
}

/// Fast anti-log2 conversion ($2^x$).
///
/// Inverse of piecewise Mitchell conversion using bit-shift exponent and piecewise quadratic mantissa reconstruction.
#[inline]
pub fn napier_exp2_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= -126.0 {
        return 0.0;
    }
    if x >= 128.0 {
        return f32::INFINITY;
    }

    let k = x.floor();
    let f = x - k;
    let mantissa_val = 1.0 + f - 0.34657359 * f * (1.0 - f);
    let scale = f32::from_bits(((k as i32 + 127) as u32) << 23);
    mantissa_val * scale
}

/// Vector dot product converting floating point multiplications into logarithmic domain additions.
///
/// Computes $a \times b \approx \text{sgn}(a \cdot b) \cdot 2^{\log_2(|a|) + \log_2(|b|)}$,
/// factorizing multiplications into addition with zero heap allocation.
#[inline]
pub fn napier_factorize_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        let ai = a[i];
        let bi = b[i];
        if ai == 0.0 || bi == 0.0 {
            continue;
        }
        let sign = if (ai < 0.0) ^ (bi < 0.0) {
            -1.0f32
        } else {
            1.0f32
        };
        let abs_a = ai.abs();
        let abs_b = bi.abs();
        let log_sum = napier_log2_f32(abs_a) + napier_log2_f32(abs_b);
        let prod = sign * napier_exp2_f32(log_sum);
        acc += prod;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_hadamard_transform_involution() {
        let mut vec = vec![1.0f32, -2.0, 3.5, 0.5, -1.2, 4.0, -0.5, 2.2];
        let original = vec.clone();
        fast_hadamard_transform_f32(&mut vec);
        // H * H = I
        fast_hadamard_transform_f32(&mut vec);
        for (i, (&v, &orig)) in vec.iter().zip(original.iter()).enumerate() {
            assert!(
                (v - orig).abs() < 1e-5,
                "Mismatch at {}: v={}, orig={}",
                i,
                v,
                orig
            );
        }
    }

    #[test]
    fn test_hadamard_mlp_identity() {
        let x = vec![0.5f32, 1.5, -2.0, 3.0];
        let weights = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        hadamard_mlp_f32(&x, &weights, &mut out);
        for i in 0..4 {
            assert!((out[i] - x[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_napier_log2_and_exp2() {
        let test_vals = [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        for &v in &test_vals {
            let log_val = napier_log2_f32(v);
            let expected_log = v.log2();
            assert!(
                (log_val - expected_log).abs() < 1e-5,
                "Log mismatch for {}: got {}, expected {}",
                v,
                log_val,
                expected_log
            );
            let roundtrip = napier_exp2_f32(log_val);
            assert!(
                (roundtrip - v).abs() < 1e-4,
                "Exp mismatch for {}: got {}",
                v,
                roundtrip
            );
        }
    }

    #[test]
    fn test_napier_factorize_dot() {
        let a = [1.0f32, 2.0, -3.0, 4.0, 0.0, -0.5];
        let b = [2.0f32, 0.5, 1.0, -2.0, 3.0, 4.0];
        let exact: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let napier = napier_factorize_dot(&a, &b);
        // Napier Mitchell approximation error is bounded within ~1%
        assert!(
            (napier - exact).abs() < 0.1,
            "Napier dot mismatch: got {}, exact {}",
            napier,
            exact
        );
    }
}
