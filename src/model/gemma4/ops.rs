//! CPU kernels shared by the Gemma 4 text decoder and its vision/audio towers: bf16 x f32 linear
//! layers over weights borrowed from the memory-mapped checkpoint, and the small Gemma building
//! blocks (RMSNorm without the `1 +` offset, tanh GELU, half-split RoPE).

use crate::model::nn;
use crate::model::par::{dot_bf16, par_for, par_rows, threads, OutPtr};
use crate::model::weights::{Mat, Weights};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Rows at or above which a linear converts its weight to f32 once and runs the packed GEMM;
/// below, each row streams the bf16 weight (decode is memory-bound).
const GEMM_ROWS: usize = 8;

/// x `[rows, w.cols]` . w^T -> `[rows, w.rows]` (PyTorch `nn.Linear` layout, no bias).
pub fn linear(x: &[f32], rows: usize, w: &Mat) -> Vec<f32> {
    let (k, n) = (w.cols, w.rows);
    assert_eq!(x.len(), rows * k, "linear: input is not rows x {k}");
    let mut out = vec![0.0f32; rows * n];
    if rows == 1 && w.packed().is_some() {
        w.gemv_f32(x, &mut out);
        return out;
    }
    if w.packed().is_some() {
        nn::matmul_nt(x, &w.to_f32(), None, rows, k, n, &mut out);
        return out;
    }
    if rows >= GEMM_ROWS {
        nn::linear_bf16_direct(x, w.data(), rows, k, n, &mut out);
        return out;
    }
    let per = n.div_ceil(threads() * 4).max(16);
    let o = OutPtr::new(&mut out);
    par_for(n.div_ceil(per), &|t| {
        let (j0, j1) = (t * per, ((t + 1) * per).min(n));
        for r in 0..rows {
            let xr = &x[r * k..(r + 1) * k];
            // SAFETY: columns j0..j1 of row r belong to task t.
            let dst = unsafe { o.slice(r * n + j0, j1 - j0) };
            for (d, j) in dst.iter_mut().zip(j0..j1) {
                *d = dot_bf16(w.row(j), xr);
            }
        }
    });
    out
}

/// [`linear`] with `ClippableLinear` input/output clamps.
pub struct Clipped {
    pub w: Mat,
    pub bounds: [f32; 4],
}

impl Clipped {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self, String> {
        let mut bounds = [
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        for (b, s) in bounds
            .iter_mut()
            .zip(["input_min", "input_max", "output_min", "output_max"])
        {
            let name = format!("{prefix}.{s}");
            if w.has(&name) {
                *b = w.scalar(&name)?;
            }
        }
        Ok(Self {
            w: w.mat(&format!("{prefix}.linear.weight"))?,
            bounds,
        })
    }

    /// The layer as graph nodes: input clamp, matmul, output clamp (each only when the checkpoint
    /// set it). `x` is `[rows, cols]`.
    pub fn add_to(&self, b: &mut crate::model::vision_graph::Gb, name: &str, x: crate::graph::ValueId, rows: usize) -> crate::graph::ValueId {
        use crate::graph::UnaryKind;
        let clamp = |b: &mut crate::model::vision_graph::Gb, tag: &str, v: crate::graph::ValueId, lo: f32, hi: f32, width: usize| {
            if lo == f32::NEG_INFINITY && hi == f32::INFINITY {
                return v;
            }
            let kind = UnaryKind::Clamp(lo.max(f32::MIN), hi.min(f32::MAX));
            b.unary(&format!("{name}.{tag}"), kind, v, vec![rows, width])
        };
        let [imin, imax, omin, omax] = self.bounds;
        let x = clamp(b, "in_clamp", x, imin, imax, self.w.cols);
        let w = b.mat(&format!("{name}.weight"), &self.w);
        let y = b.linear(name, x, w, None, rows, self.w.rows);
        clamp(b, "out_clamp", y, omin, omax, self.w.rows)
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let [imin, imax, omin, omax] = self.bounds;
        let clamped: Vec<f32>;
        let x = if imin > f32::NEG_INFINITY || imax < f32::INFINITY {
            clamped = x.iter().map(|v| v.clamp(imin, imax)).collect();
            &clamped
        } else {
            x
        };
        let mut y = linear(x, rows, &self.w);
        if omin > f32::NEG_INFINITY || omax < f32::INFINITY {
            y.iter_mut().for_each(|v| *v = v.clamp(omin, omax));
        }
        y
    }
}

/// Gemma 4 RMSNorm over each `width` row: `x * (mean(x^2) + eps)^-1/2 * w` (`w` absent = 1).
pub fn rms_norm(x: &mut [f32], width: usize, w: Option<&[f32]>, eps: f32) {
    par_rows(x, width, &|_, r| {
        let ms = r.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inv = (ms + eps).powf(-0.5);
        match w {
            Some(w) => r.iter_mut().zip(w).for_each(|(v, g)| *v *= inv * g),
            None => r.iter_mut().for_each(|v| *v *= inv),
        }
    });
}

/// `gelu_pytorch_tanh`.
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// `gelu_tanh(a) * b`, elementwise into `a`.
pub fn gelu_mul(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "gelu_mul: length mismatch");
    const CH: usize = 1 << 12;
    let o = OutPtr::new(a);
    let len = b.len();
    par_for(len.div_ceil(CH), &|t| {
        let (s, e) = (t * CH, ((t + 1) * CH).min(len));
        // SAFETY: chunk t is this task's.
        let a = unsafe { o.slice(s, e - s) };
        a.iter_mut()
            .zip(&b[s..e])
            .for_each(|(a, &b)| *a = gelu_tanh(*a) * b);
    });
}

/// Half-split RoPE (`rotate_half`) on one head: pairs (i, i + half) rotate by angle i.
#[inline]
pub fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    let (a, b) = x[..2 * half].split_at_mut(half);
    for i in 0..half {
        let (x0, x1) = (a[i], b[i]);
        a[i] = x0 * cos[i] - x1 * sin[i];
        b[i] = x1 * cos[i] + x0 * sin[i];
    }
}

/// `cos`/`sin` of `pos * inv_freq`.
pub fn rope_table(pos: f32, inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    inv_freq
        .iter()
        .map(|f| (pos * f).sin_cos())
        .map(|(s, c)| (c, s))
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_bf16_matches_reference() {
        let in_f = 64;
        let out_f = 128;
        let w_vals: Vec<f32> = (0..in_f * out_f)
            .map(|i| (((i * 17) % 31) as f32 - 15.0) / 10.0)
            .collect();
        let weights = Weights::from_f32(vec![("layer.weight".into(), vec![out_f, in_f])], |_| {
            Ok(w_vals.clone())
        })
        .unwrap();
        let mat = weights.mat("layer.weight").unwrap();

        for rows in [1, 4, 7, 8, 12, 17] {
            let x: Vec<f32> = (0..rows * in_f)
                .map(|i| (((i * 13) % 29) as f32 - 14.0) / 10.0)
                .collect();
            let got = linear(&x, rows, &mat);
            assert_eq!(got.len(), rows * out_f);

            // Compute reference with exact dequantized float weights
            let mut want = vec![0.0f32; rows * out_f];
            nn::matmul_nt(&x, &mat.to_f32(), None, rows, in_f, out_f, &mut want);

            for r in 0..rows {
                for c in 0..out_f {
                    let idx = r * out_f + c;
                    let diff = (got[idx] - want[idx]).abs();
                    assert!(
                        diff < 1e-2,
                        "mismatch at rows={rows}, r={r}, c={c}: got={}, want={}",
                        got[idx],
                        want[idx]
                    );
                }
            }
        }
    }
}
