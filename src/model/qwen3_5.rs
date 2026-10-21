//! [`Qwen3Encoder`]: the dense Qwen3 / Qwen3-VL / Qwen2.5-VL text decoder as a host text encoder
//! for the diffusion pipelines (hidden states, no LM head). The Qwen language models themselves
//! run from their family specs (`arch/families/qwen*.json5`).
//!
//! Reference: transformers `models/qwen3/modeling_qwen3.py`, `models/qwen3_vl/modeling_qwen3_vl.py`.

mod text;

use crate::model::par::{dot_bf16, par_for, par_rows, OutPtr};
use crate::model::weights::{Mat, Weights};
use serde_json::Value;
use std::path::Path;
use text::{Arch, TextModel};

fn text_config(cfg: &Value) -> &Value {
    if cfg["text_config"].is_object() {
        &cfg["text_config"]
    } else {
        cfg
    }
}

/// A `qwen3` checkpoint directory (`config.json`, `model.safetensors[.index.json]`) used as a
/// text encoder (e.g. by diffusion pipelines).
pub struct Qwen3Encoder {
    text: TextModel,
}

impl Qwen3Encoder {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg_path = dir.join("config.json");
        let cfg: Value = std::fs::read(&cfg_path)
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", cfg_path.display()))?;
        let index = dir.join("model.safetensors.index.json");
        let weights = if index.exists() {
            index
        } else {
            dir.join("model.safetensors")
        };
        // A Qwen3-VL checkpoint (Qwen-Image 2.1's encoder) and a Qwen2.5-VL one (Qwen-Image 1's)
        // are read for their text decoders only; plain `qwen2` shares that decoder.
        let arch = match cfg["model_type"].as_str() {
            Some("qwen3_vl") => Arch::Qwen3Vl,
            Some("qwen2" | "qwen2_5_vl") => Arch::Qwen2,
            _ => Arch::Qwen3,
        };
        let text = TextModel::load(&Weights::open(&weights)?, text_config(&cfg), arch)?;
        Ok(Self { text })
    }

    /// The text tower of a Qwen3-VL checkpoint file or shard index (`model.language_model.*`; its
    /// vision tower stays unread), `cfg` being the model's `config.json` (`text_config` nested).
    pub fn from_weights(weights: &Path, cfg: &Value) -> Result<Self, String> {
        let text = TextModel::load(&Weights::open(weights)?, text_config(cfg), Arch::Qwen3Vl)?;
        Ok(Self { text })
    }

    /// `[ids.len(), hidden]` after the first `layers` decoder layers, before the final norm.
    pub fn hidden_states(&self, ids: &[u32], layers: usize) -> Result<Vec<f32>, String> {
        Ok(self.hidden_states_at(ids, ids.len(), &[layers])?.remove(0))
    }

    /// transformers' `hidden_states[-1]` of `ids`: `[ids.len(), hidden]` after the final norm.
    pub fn hidden_states_normed(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.text.normed_hidden_states(ids)
    }

    /// transformers' `hidden_states[k]` for each `k` in `layers` (ascending) with attention mask
    /// `[1] * valid + [0] * (len - valid)`, in one pass.
    pub fn hidden_states_at(
        &self,
        ids: &[u32],
        valid: usize,
        layers: &[usize],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.text.hidden_states_at(ids, valid, layers)
    }
}

// ---------------------------------------------------------------------------

/// `prefixes[i].weight` stacked along the output dim (one pass for several projections),
/// borrowed from the mapped checkpoint, with `.bias` when `bias`.
fn linear(w: &Weights, prefixes: &[&str], bias: bool) -> Result<Linear, String> {
    let (mut parts, mut b) = (Vec::with_capacity(prefixes.len()), Vec::new());
    for p in prefixes {
        let m = w.mat(&format!("{p}.weight"))?;
        if let Some(first) = parts.first().map(|f: &Mat| f.cols) {
            if m.cols != first {
                return Err(format!("{p}.weight: {} inputs, expected {first}", m.cols));
            }
        }
        if bias {
            b.extend(w.vec_n(&format!("{p}.bias"), m.rows)?);
        }
        parts.push(m);
    }
    Ok(Linear {
        rows: parts.iter().map(|m| m.rows).sum(),
        cols: parts.first().map_or(0, |m| m.cols),
        parts,
        bias: bias.then_some(b),
    })
}

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

/// `y = x W^T (+ b)` with `W` the bf16 `parts` stacked along the output dim, `[rows, cols]`.
pub(crate) struct Linear {
    pub(crate) parts: Vec<Mat>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bias: Option<Vec<f32>>,
}

impl Linear {
    /// `x` `[n, cols]` -> `[n, rows]`.
    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        if self.parts.iter().any(|m| m.packed().is_some()) {
            let mut out = vec![0.0f32; n * self.rows];
            if n == 1 && self.parts.len() == 1 {
                self.parts[0].gemv_f32(x, &mut out);
                if let Some(b) = &self.bias {
                    for r in 0..self.rows {
                        out[r] += b[r];
                    }
                }
                return out;
            }
            let mut row_offset = 0;
            for part in &self.parts {
                let p_rows = part.rows;
                let w_f32 = part.to_f32();
                for sample in 0..n {
                    let xs = &x[sample * self.cols..(sample + 1) * self.cols];
                    for r in 0..p_rows {
                        let wr = &w_f32[r * self.cols..(r + 1) * self.cols];
                        let mut dot = 0.0f32;
                        for c in 0..self.cols {
                            dot += xs[c] * wr[c];
                        }
                        out[sample * self.rows + row_offset + r] = dot;
                    }
                }
                row_offset += p_rows;
            }
            if let Some(b) = &self.bias {
                for sample in 0..n {
                    for r in 0..self.rows {
                        out[sample * self.rows + r] += b[r];
                    }
                }
            }
            out
        } else {
            let parts: Vec<&[u16]> = self.parts.iter().map(Mat::data).collect();
            linear_bf16(&parts, self.cols, self.bias.as_deref(), x, n)
        }
    }

}


/// Weight rows one task converts to f32 and multiplies against every input row.
const ROW_BLOCK: usize = 16;
/// Weight rows per task when there is a single input row (memory-bound GEMV).
const GEMV_BLOCK: usize = 64;

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `x` `[n, k]` times the row-stacked bf16 `parts` (each `[rows_p, k]`) transposed, plus `bias`
/// -> `[n, sum rows_p]`: NEON GEMV for up to [`EXACT_ROWS`] rows (each row's result is
/// bit-identical to the one-row GEMV, so speculative verify reproduces plain decode exactly),
/// 4x4 register tiles over f32-converted weight blocks for more.
fn linear_bf16(parts: &[&[u16]], k: usize, bias: Option<&[f32]>, x: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), n * k, "linear input is not n x cols");
    let rows: usize = parts.iter().map(|p| p.len() / k.max(1)).sum();
    let mut out = vec![0.0f32; n * rows];
    let o = OutPtr::new(&mut out);
    let gemv = n <= EXACT_ROWS;
    let block = if gemv { GEMV_BLOCK } else { ROW_BLOCK };
    // (part, first row within it, first output column, row count): blocks never span parts.
    let mut tasks = Vec::new();
    let mut g = 0;
    for (pi, p) in parts.iter().enumerate() {
        let pr = p.len() / k.max(1);
        for r0 in (0..pr).step_by(block) {
            tasks.push((pi, r0, g + r0, block.min(pr - r0)));
        }
        g += pr;
    }
    par_for(tasks.len(), &|t| {
        let (pi, r0, c0, rb) = tasks[t];
        let w = &parts[pi][r0 * k..(r0 + rb) * k];
        // SAFETY: task t alone writes columns c0..c0+rb of every output row.
        let put = |i: usize, j: usize, v: f32| unsafe { o.slice(i * rows + c0 + j, 1)[0] = v };
        if gemv {
            let xr = |i: usize| &x[i * k..][..k];
            for j in 0..rb {
                let wj = &w[j * k..][..k];
                let emit = |i: usize, r: &[f32]| {
                    for (a, &v) in r.iter().enumerate() {
                        put(i + a, j, v);
                    }
                };
                for i in (0..n).step_by(4) {
                    match n - i {
                        1 => put(i, j, dot_bf16(wj, xr(i))),
                        2 => emit(i, &dot_bf16_rows(wj, [xr(i), xr(i + 1)])),
                        3 => emit(i, &dot_bf16_rows(wj, [xr(i), xr(i + 1), xr(i + 2)])),
                        _ => emit(
                            i,
                            &dot_bf16_rows(wj, [xr(i), xr(i + 1), xr(i + 2), xr(i + 3)]),
                        ),
                    }
                }
            }
        } else {
            SCRATCH.with_borrow_mut(|wf| {
                wf.resize(rb * k, 0.0);
                for (d, &s) in wf.iter_mut().zip(w) {
                    *d = f32::from_bits((s as u32) << 16);
                }
                let wr = |j: usize| &wf[j * k..][..k];
                let xr = |i: usize| &x[i * k..][..k];
                let (n4, rb4) = (n / 4 * 4, rb / 4 * 4);
                for i in (0..n4).step_by(4) {
                    for j in (0..rb4).step_by(4) {
                        let tile = tile4x4(
                            [xr(i), xr(i + 1), xr(i + 2), xr(i + 3)],
                            [wr(j), wr(j + 1), wr(j + 2), wr(j + 3)],
                            k,
                        );
                        for (a, row) in tile.iter().enumerate() {
                            for (b, &v) in row.iter().enumerate() {
                                put(i + a, j + b, v);
                            }
                        }
                    }
                    for j in rb4..rb {
                        for a in 0..4 {
                            put(i + a, j, vector_dot(xr(i + a), wr(j)));
                        }
                    }
                }
                for i in n4..n {
                    for j in 0..rb {
                        put(i, j, vector_dot(xr(i), wr(j)));
                    }
                }
            });
        }
        if let Some(b) = bias {
            for i in 0..n {
                for j in 0..rb {
                    // SAFETY: as above.
                    unsafe { o.slice(i * rows + c0 + j, 1)[0] += b[c0 + j] };
                }
            }
        }
    });
    out
}

use crate::model::simd::vector_dot_product as vector_dot;

/// Input rows up to which [`linear_bf16`] runs the exact GEMV path (a verify pass of up to 15
/// draft tokens).
const EXACT_ROWS: usize = 16;

/// [`dot_bf16`] of one weight row against `R` input rows, reading the weights once; every result
/// is bit-identical to `dot_bf16(w, x[r])` (same lanes, FMA order, reduction tree and tail).
#[cfg(target_arch = "aarch64")]
fn dot_bf16_rows<const R: usize>(w: &[u16], x: [&[f32]; R]) -> [f32; R] {
    use core::arch::aarch64::*;
    let n = w.len();
    assert!(x.iter().all(|s| s.len() >= n));
    let mut out = [0.0f32; R];
    let mut i = 0;
    // SAFETY: every load is at i + 16 <= n, bounded by the assert above.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; R];
        while i + 16 <= n {
            let a = vld1q_u16(w.as_ptr().add(i));
            let b = vld1q_u16(w.as_ptr().add(i + 8));
            let wv = [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(a))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(a)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(b))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(b)),
            ];
            for (acc, xs) in acc.iter_mut().zip(&x) {
                for q in 0..4 {
                    acc[q] = vfmaq_f32(acc[q], wv[q], vld1q_f32(xs.as_ptr().add(i + 4 * q)));
                }
            }
            i += 16;
        }
        for (o, acc) in out.iter_mut().zip(&acc) {
            *o = vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ));
        }
    }
    for (o, xs) in out.iter_mut().zip(&x) {
        for j in i..n {
            *o += crate::model::bf16_to_f32(w[j]) * xs[j];
        }
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_bf16_rows<const R: usize>(w: &[u16], x: [&[f32]; R]) -> [f32; R] {
    x.map(|xs| dot_bf16(w, &xs[..w.len()]))
}

/// `out[a][b] = dot(x[a], w[b])` over the first `k` elements: 16 accumulators, 8 loads per 16 FMAs.
#[cfg(target_arch = "aarch64")]
fn tile4x4(x: [&[f32]; 4], w: [&[f32]; 4], k: usize) -> [[f32; 4]; 4] {
    use core::arch::aarch64::*;
    assert!(x.iter().chain(&w).all(|s| s.len() >= k));
    let mut out = [[0.0f32; 4]; 4];
    let mut p = 0;
    // SAFETY: every load is at p + 4 <= k, bounded by the assert above.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
        while p + 4 <= k {
            let xv = x.map(|s| vld1q_f32(s.as_ptr().add(p)));
            let wv = w.map(|s| vld1q_f32(s.as_ptr().add(p)));
            for a in 0..4 {
                for b in 0..4 {
                    acc[a][b] = vfmaq_f32(acc[a][b], xv[a], wv[b]);
                }
            }
            p += 4;
        }
        for a in 0..4 {
            for b in 0..4 {
                out[a][b] = vaddvq_f32(acc[a][b]);
            }
        }
    }
    for a in 0..4 {
        for b in 0..4 {
            for q in p..k {
                out[a][b] += x[a][q] * w[b][q];
            }
        }
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
fn tile4x4(x: [&[f32]; 4], w: [&[f32]; 4], k: usize) -> [[f32; 4]; 4] {
    x.map(|xa| w.map(|wb| vector_dot(&xa[..k], &wb[..k])))
}

/// RMSNorm of every row in place (`weight` already includes Qwen3.5's `1 +` offset).
fn rms_norm_rows(x: &mut [f32], weight: &[f32], eps: f32) {
    let d = weight.len();
    par_rows(x, d, &|_, row| rms_norm(row, weight, eps));
}

fn rms_norm(row: &mut [f32], weight: &[f32], eps: f32) {
    let inv = 1.0 / (vector_dot(row, row) / row.len() as f32 + eps).sqrt();
    for (v, w) in row.iter_mut().zip(weight) {
        *v *= inv * w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_reference_for_gemv_and_tiled_gemm() {
        let (rows, cols) = (37, 70);
        let w: Vec<u16> = (0..rows * cols)
            .map(|i| crate::model::f32_to_bf16(((i * 7919) % 13) as f32 / 7.0 - 0.9))
            .collect();
        let bias: Vec<f32> = (0..rows).map(|r| r as f32 * 0.1).collect();
        // Two stacked parts (20 + 17 rows): blocks restart at the part boundary.
        let parts = [&w[..20 * cols], &w[20 * cols..]];
        for n in [1, 2, 3, 9, 16, 21] {
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i * 31) % 17) as f32 / 9.0 - 1.0)
                .collect();
            let y = linear_bf16(&parts, cols, Some(&bias), &x, n);
            for i in 0..n {
                for r in 0..rows {
                    let want: f32 = bias[r]
                        + (0..cols)
                            .map(|c| {
                                f32::from_bits((w[r * cols + c] as u32) << 16) * x[i * cols + c]
                            })
                            .sum::<f32>();
                    let got = y[i * rows + r];
                    assert!(
                        (got - want).abs() < 1e-3,
                        "n={n} row {i} col {r}: {got} vs {want}"
                    );
                }
                // Up to EXACT_ROWS a row's output does not depend on its batch (speculative
                // verify must reproduce one-token decode bit for bit).
                if n <= EXACT_ROWS {
                    let alone = linear_bf16(&parts, cols, Some(&bias), &x[i * cols..][..cols], 1);
                    assert_eq!(alone, y[i * rows..][..rows], "n={n} row {i}");
                }
            }
        }
    }
}
