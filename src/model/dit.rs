//! Building blocks shared by the diffusion transformers (SD3 MMDiT, Z-Image S3-DiT): bf16 linear
//! layers over mapped checkpoints, row-wise norms and adaLN modulation, sinusoidal timestep
//! features, and diffusers component-directory weight lookup. Activations are f32 `[rows, d]`.

use crate::model::nn;
use crate::model::par::par_rows;
use crate::model::weights::{Mat, Weights};
use std::path::Path;

/// `nn.Linear` with a bf16 weight `[out, in]` (F32/F16 checkpoints are rounded to bf16 at load).
pub struct Lin {
    w: Mat,
    b: Option<Vec<f32>>,
}

impl Lin {
    /// `{p}.weight` and, when stored, `{p}.bias`.
    pub fn load(w: &Weights, p: &str) -> Result<Self, String> {
        let wt = w.mat(&format!("{p}.weight"))?;
        let bias = format!("{p}.bias");
        let b = if w.has(&bias) {
            Some(w.vec_n(&bias, wt.rows)?)
        } else {
            None
        };
        Ok(Self { w: wt, b })
    }

    pub fn out_dim(&self) -> usize {
        self.w.rows
    }

    pub fn in_dim(&self) -> usize {
        self.w.cols
    }

    /// `x [m, in] -> [m, out]` as a graph node, the weight bound by name in its stored encoding.
    pub fn emit(
        &self,
        b: &mut crate::model::dit_graph::Gb,
        name: &str,
        x: crate::graph::ValueId,
    ) -> crate::graph::ValueId {
        b.lin_mat(name, x, &self.w, self.b.as_deref())
    }

    /// x [rows, in] -> [rows, out].
    pub fn fwd(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * self.w.rows];
        nn::matmul_bf16(
            x,
            self.w.data(),
            self.b.as_deref(),
            rows,
            self.w.cols,
            self.w.rows,
            &mut out,
        );
        out
    }
}

/// RMSNorm over every `d`-wide chunk of `x` (per token, or per head for QK-norm): `x / rms * w`.
pub fn rms_norm(x: &mut [f32], d: usize, w: &[f32], eps: f32) {
    par_rows(x, d, &|_, r| {
        let inv = 1.0 / (r.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for (v, g) in r.iter_mut().zip(w) {
            *v *= inv * g;
        }
    });
}

/// Affine-free LayerNorm of x [rows, d], then `* (1 + scale) + shift` (both `[d]`; `shift` may be empty).
pub fn norm_modulate(x: &[f32], d: usize, eps: f32, scale: &[f32], shift: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    par_rows(&mut y, d, &|_, r| {
        let mean = r.iter().sum::<f32>() / d as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v = (*v - mean) * inv * (1.0 + scale[i]) + shift.get(i).copied().unwrap_or(0.0);
        }
    });
    y
}

/// h [rows, d] += gate[d] * y (gate `None`: plain residual add).
pub fn add_gated(h: &mut [f32], d: usize, gate: Option<&[f32]>, y: &[f32]) {
    par_rows(h, d, &|r, row| {
        let y = &y[r * d..(r + 1) * d];
        match gate {
            Some(g) => row
                .iter_mut()
                .zip(y)
                .zip(g)
                .for_each(|((h, y), g)| *h += g * y),
            None => row.iter_mut().zip(y).for_each(|(h, y)| *h += y),
        }
    });
}

/// `[cos(t f_i), sin(t f_i)]` with `f_i = 10000^(-i / (dim/2))`: diffusers `get_timestep_embedding`
/// with flip_sin_to_cos and shift 0 (SD3), and Z-Image's `TimestepEmbedder.timestep_embedding`.
pub fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let args: Vec<f32> = (0..half)
        .map(|i| t * (-(10000f32.ln()) * i as f32 / half as f32).exp())
        .collect();
    args.iter()
        .map(|a| a.cos())
        .chain(args.iter().map(|a| a.sin()))
        .collect()
}

/// Copy of `x` with SiLU applied.
pub fn silu_of(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    nn::silu(&mut y);
    y
}

/// The mapped weights of a diffusers component directory (e.g. `<snapshot>/transformer`), the file
/// [`crate::model::hub::component_weights`] picks.
pub fn component_weights(dir: &Path) -> Result<Weights, String> {
    Weights::open(&crate::model::hub::component_weights(dir)?)
}

/// `<dir>/config.json` parsed.
pub fn read_config(dir: &Path) -> Result<serde_json::Value, String> {
    let p = dir.join("config.json");
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

/// `<snapshot>/scheduler/scheduler_config.json` parsed. Python's bare `-Infinity`/`Infinity`/`NaN`
/// (e.g. Sana's `lambda_min_clipped`) are read as null.
pub fn read_scheduler(snapshot: &Path) -> Result<serde_json::Value, String> {
    let p = snapshot.join("scheduler").join("scheduler_config.json");
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let s = s
        .replace("-Infinity", "null")
        .replace("Infinity", "null")
        .replace("NaN", "null");
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

/// diffusers pipelines built on these blocks (bf16-resident weights, 2 bytes per parameter).
pub const PIPELINES: &[&str] = &["StableDiffusion3Pipeline", "ZImagePipeline", "SanaPipeline"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_features_are_cos_then_sin() {
        let f = timestep_features(2.0, 8);
        // frequencies 1, 10000^-0.25, 10000^-0.5, 10000^-0.75
        let fr = [1.0f32, 0.1, 0.01, 0.001];
        for i in 0..4 {
            assert!((f[i] - (2.0 * fr[i]).cos()).abs() < 1e-5);
            assert!((f[4 + i] - (2.0 * fr[i]).sin()).abs() < 1e-5);
        }
    }

    #[test]
    fn norm_modulate_matches_definition() {
        let y = norm_modulate(&[1.0, 3.0], 2, 0.0, &[1.0, 0.0], &[0.5, -0.5]);
        // normalised: [-1, 1]; * (1 + [1, 0]) + [0.5, -0.5]
        assert!(
            (y[0] + 1.5).abs() < 1e-6 && (y[1] - 0.5).abs() < 1e-6,
            "{y:?}"
        );
    }
}
