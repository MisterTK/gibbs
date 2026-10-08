//! The Qwen3.5 / Qwen3-VL vision tower and the `Qwen2VLImageProcessor` that feeds it.

use super::{linear, Linear};
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::model::weights::Weights;
use serde_json::Value;
use std::path::Path;

/// `preprocessor_config.json`: pixel budget and normalization.
pub(super) struct Preprocessor {
    min_pixels: usize,
    max_pixels: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl Preprocessor {
    pub(super) fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes, &path.display().to_string())
    }

    /// `preprocessor_config.json` contents; `what` names their source in errors.
    pub(super) fn parse(bytes: &[u8], what: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("{what}: {e}"))?;
        let px = |a: &str, b: &str| {
            v["size"][a]
                .as_u64()
                .or(v[b].as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("{what}: no size.{a} / {b}"))
        };
        let triple = |k: &str| -> Result<[f32; 3], String> {
            let a = v[k].as_array().filter(|a| a.len() == 3);
            let a = a.ok_or_else(|| format!("{what}: {k} is not 3 numbers"))?;
            Ok([0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
        };
        Ok(Self {
            min_pixels: px("shortest_edge", "min_pixels")?,
            max_pixels: px("longest_edge", "max_pixels")?,
            mean: triple("image_mean")?,
            std: triple("image_std")?,
        })
    }
}

/// `smart_resize`: (height, width) divisible by `factor`, aspect kept, pixel count in
/// `[min_pixels, max_pixels]`. Python's `round` is half-to-even.
fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min: usize,
    max: usize,
) -> Result<(usize, usize), String> {
    let (hf, wf, f) = (h as f64, w as f64, factor as f64);
    if h == 0 || w == 0 || hf.max(wf) / hf.min(wf) > 200.0 {
        return Err(format!("image {w}x{h}: aspect ratio must be below 200"));
    }
    let mut hb = (hf / f).round_ties_even() * f;
    let mut wb = (wf / f).round_ties_even() * f;
    if hb * wb > max as f64 {
        let beta = (hf * wf / max as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min as f64 {
        let beta = (min as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

struct Block {
    norm1: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

/// LayerNorm, 2x2 patch concat, MLP to the text width. The deepstack mergers normalize after
/// the concat (`use_postshuffle_norm`).
struct Merger {
    norm: (Vec<f32>, Vec<f32>),
    postshuffle: bool,
    fc1: Linear,
    fc2: Linear,
}

impl Merger {
    fn load(ck: &Weights, p: &str, postshuffle: bool) -> Result<Self, String> {
        Ok(Self {
            norm: (
                ck.vec(&format!("{p}.norm.weight"))?,
                ck.vec(&format!("{p}.norm.bias"))?,
            ),
            postshuffle,
            fc1: linear(ck, &[&format!("{p}.linear_fc1")], true)?,
            fc2: linear(ck, &[&format!("{p}.linear_fc2")], true)?,
        })
    }

    /// `x` `[patches, d]` in merge-block order -> `[patches / unit, out]`.
    fn forward(&self, x: &[f32], d: usize, unit: usize) -> Vec<f32> {
        let rows = x.len() / d;
        let (w, b) = &self.norm;
        let y = if self.postshuffle {
            nn::layer_norm(x, rows / unit, d * unit, w, b, 1e-6)
        } else {
            nn::layer_norm(x, rows, d, w, b, 1e-6)
        };
        let mut h = self.fc1.forward(&y, rows / unit);
        nn::gelu(&mut h);
        self.fc2.forward(&h, rows / unit)
    }
}

pub(super) struct Vision {
    hidden: usize,
    heads: usize,
    patch: usize,
    temporal: usize,
    merge: usize,
    /// Side of the learned square position grid.
    side: usize,
    patch_embed: Linear,
    pos_embed: Vec<f32>,
    /// Rotary inverse frequencies over a quarter of the head dim (rows, then cols).
    inv_freq: Vec<f32>,
    blocks: Vec<Block>,
    merger: Merger,
    /// (block index, merger) for Qwen3-VL deepstack features.
    deepstack: Vec<(usize, Merger)>,
}

/// One image through the tower.
pub(super) struct Encoded {
    /// Merged grid (tokens = rows * cols).
    pub rows: usize,
    pub cols: usize,
    /// `[rows * cols, text hidden]`.
    pub embeds: Vec<f32>,
    pub deepstack: Vec<Vec<f32>>,
}

impl Vision {
    pub(super) fn load(ck: &Weights, c: &Value) -> Result<Self, String> {
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        if c["hidden_act"] != "gelu_pytorch_tanh" {
            return Err(format!(
                "vision hidden_act {} is not gelu_pytorch_tanh",
                c["hidden_act"]
            ));
        }
        let (hidden, heads, depth) = (get("hidden_size")?, get("num_heads")?, get("depth")?);
        let positions = get("num_position_embeddings")?;
        let side = (positions as f64).sqrt() as usize;
        if side * side != positions {
            return Err(format!(
                "num_position_embeddings {positions} is not a square"
            ));
        }
        let p = "model.visual";
        let blocks = (0..depth)
            .map(|i| {
                let b = format!("{p}.blocks.{i}");
                let ln = |n: &str| -> Result<_, String> {
                    Ok((
                        ck.vec(&format!("{b}.{n}.weight"))?,
                        ck.vec(&format!("{b}.{n}.bias"))?,
                    ))
                };
                Ok(Block {
                    norm1: ln("norm1")?,
                    norm2: ln("norm2")?,
                    qkv: linear(ck, &[&format!("{b}.attn.qkv")], true)?,
                    proj: linear(ck, &[&format!("{b}.attn.proj")], true)?,
                    fc1: linear(ck, &[&format!("{b}.mlp.linear_fc1")], true)?,
                    fc2: linear(ck, &[&format!("{b}.mlp.linear_fc2")], true)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let deepstack = c["deepstack_visual_indexes"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(k, idx)| {
                let idx = idx.as_u64().unwrap_or(u64::MAX) as usize;
                if idx >= depth {
                    return Err(format!(
                        "deepstack index {idx} is past the {depth} vision blocks"
                    ));
                }
                Ok((
                    idx,
                    Merger::load(ck, &format!("{p}.deepstack_merger_list.{k}"), true)?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let head_dim = hidden / heads;
        let quarter = head_dim / 4;
        let v = Self {
            hidden,
            heads,
            patch: get("patch_size")?,
            temporal: get("temporal_patch_size")?,
            merge: get("spatial_merge_size")?,
            side,
            patch_embed: linear(ck, &[&format!("{p}.patch_embed.proj")], true)?,
            pos_embed: ck.vec(&format!("{p}.pos_embed.weight"))?,
            inv_freq: (0..quarter)
                .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / (head_dim / 2) as f32))
                .collect(),
            blocks,
            merger: Merger::load(ck, &format!("{p}.merger"), false)?,
            deepstack,
        };
        if v.patch_embed.cols != 3 * v.temporal * v.patch * v.patch
            || v.pos_embed.len() != positions * hidden
        {
            return Err("patch_embed / pos_embed shapes disagree with vision_config".into());
        }
        Ok(v)
    }

    pub(super) fn out_hidden(&self) -> usize {
        self.merger.fc2.rows
    }

    pub(super) fn deepstack_layers(&self) -> usize {
        self.deepstack.len()
    }

    pub(super) fn summary(&self) -> String {
        format!(
            "vision {} blocks, hidden {}",
            self.blocks.len(),
            self.hidden
        )
    }

    /// Merged token grid (rows, cols) `img` becomes: `rows * cols` placeholder tokens.
    pub(super) fn grid(
        &self,
        img: &ImageBuffer,
        pre: &Preprocessor,
    ) -> Result<(usize, usize), String> {
        let f = self.patch * self.merge;
        let (rh, rw) = smart_resize(img.height, img.width, f, pre.min_pixels, pre.max_pixels)?;
        Ok((rh / f, rw / f))
    }

    pub(super) fn encode(&self, img: &ImageBuffer, pre: &Preprocessor) -> Result<Encoded, String> {
        let (p, m, d) = (self.patch, self.merge, self.hidden);
        // ponytail: pixel budget comes from preprocessor_config.json (Qwen ships 16M px max), so a large photo costs O(patches^2) attention; cap max_pixels if latency matters.
        let (rh, rw) = smart_resize(img.height, img.width, p * m, pre.min_pixels, pre.max_pixels)?;
        let resized;
        let img = if (rh, rw) == (img.height, img.width) {
            img
        } else {
            resized = img.resize(rw, rh);
            &resized
        };
        let (gh, gw) = (rh / p, rw / p);
        let n = gh * gw;
        // Patches in merge-block order (block row, block col, row in block, col in block); each
        // is [channel, frame, y, x] with the still image repeated over `temporal` frames.
        let k = 3 * self.temporal * p * p;
        let mut patches = vec![0.0f32; n * k];
        let mut coords = Vec::with_capacity(n);
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for ir in 0..m {
                    for ic in 0..m {
                        coords.push((br * m + ir, bc * m + ic));
                    }
                }
            }
        }
        par_rows(&mut patches, k, &|i, row| {
            let (r, c) = coords[i];
            let mut o = 0;
            for ch in 0..3 {
                for _ in 0..self.temporal {
                    for y in 0..p {
                        for x in 0..p {
                            let px =
                                img.data[((r * p + y) * rw + c * p + x) * 3 + ch] as f32 / 255.0;
                            row[o] = (px - pre.mean[ch]) / pre.std[ch];
                            o += 1;
                        }
                    }
                }
            }
        });
        let mut x = self.patch_embed.forward(&patches, n);
        drop(patches);
        // Learned positions, bilinearly resampled (align_corners) from the side x side grid.
        let taps = |i: usize, size: usize| {
            let src = i as f32 * (self.side - 1) as f32 / size.saturating_sub(1).max(1) as f32;
            let f = src.floor();
            let lo = (f as usize).min(self.side - 1);
            (lo, (lo + 1).min(self.side - 1), src - f)
        };
        par_rows(&mut x, d, &|i, row| {
            let (r, c) = coords[i];
            let ((r0, r1, fr), (c0, c1, fc)) = (taps(r, gh), taps(c, gw));
            for (idx, wt) in [
                (r0 * self.side + c0, (1.0 - fr) * (1.0 - fc)),
                (r0 * self.side + c1, (1.0 - fr) * fc),
                (r1 * self.side + c0, fr * (1.0 - fc)),
                (r1 * self.side + c1, fr * fc),
            ] {
                for (v, &e) in row.iter_mut().zip(&self.pos_embed[idx * d..][..d]) {
                    *v += wt * e;
                }
            }
        });
        // 2D rotary: first quarter of each head's pairs follow the row, second quarter the column.
        let q4 = self.inv_freq.len();
        let (mut cos, mut sin) = (
            Vec::with_capacity(n * 2 * q4),
            Vec::with_capacity(n * 2 * q4),
        );
        for &(r, c) in &coords {
            for pos in [r, c] {
                for f in &self.inv_freq {
                    let a = pos as f32 * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        let hd = d / self.heads;
        let mut deepstack = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            let y = nn::layer_norm(&x, n, d, &b.norm1.0, &b.norm1.1, 1e-6);
            let qkv = b.qkv.forward(&y, n);
            let (mut q, mut kk, mut v) = (
                vec![0.0f32; n * d],
                vec![0.0f32; n * d],
                vec![0.0f32; n * d],
            );
            for i in 0..n {
                let row = &qkv[i * 3 * d..][..3 * d];
                q[i * d..][..d].copy_from_slice(&row[..d]);
                kk[i * d..][..d].copy_from_slice(&row[d..2 * d]);
                v[i * d..][..d].copy_from_slice(&row[2 * d..]);
            }
            for t in [&mut q, &mut kk] {
                par_rows(t, d, &|i, row| {
                    let (c, s) = (&cos[i * 2 * q4..][..2 * q4], &sin[i * 2 * q4..][..2 * q4]);
                    for head in row.chunks_exact_mut(hd) {
                        let half = hd / 2;
                        for j in 0..half {
                            let (a, bb) = (head[j], head[j + half]);
                            head[j] = a * c[j] - bb * s[j];
                            head[j + half] = bb * c[j] + a * s[j];
                        }
                    }
                });
            }
            let a = nn::attention(&q, &kk, &v, n, n, self.heads, hd, false);
            nn::add_in_place(&mut x, &b.proj.forward(&a, n));
            let y = nn::layer_norm(&x, n, d, &b.norm2.0, &b.norm2.1, 1e-6);
            let mut h = b.fc1.forward(&y, n);
            nn::gelu_tanh(&mut h);
            nn::add_in_place(&mut x, &b.fc2.forward(&h, n));
            for (_, mg) in self.deepstack.iter().filter(|(i, _)| *i == bi) {
                deepstack.push(mg.forward(&x, d, m * m));
            }
        }
        Ok(Encoded {
            rows: gh / m,
            cols: gw / m,
            embeds: self.merger.forward(&x, d, m * m),
            deepstack,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::smart_resize;

    #[test]
    fn smart_resize_rounds_half_even_and_respects_pixel_bounds() {
        // 48 / 32 = 1.5 rounds to 2 (even), 80 / 32 = 2.5 rounds to 2 (even); then min_pixels scales up.
        assert_eq!(smart_resize(48, 80, 32, 0, 1 << 24).unwrap(), (64, 64));
        // Too small: scaled up to at least min_pixels, multiples of the factor.
        let (h, w) = smart_resize(10, 20, 32, 65536, 1 << 24).unwrap();
        assert!(h * w >= 65536 && h % 32 == 0 && w % 32 == 0);
        // Too large: scaled down under max_pixels, aspect kept within a factor step.
        let (h, w) = smart_resize(3000, 4000, 32, 65536, 1 << 20).unwrap();
        assert!(h * w <= 1 << 20 && h % 32 == 0 && w % 32 == 0);
        assert!(((w as f64 / h as f64) - 4.0 / 3.0).abs() < 0.05);
        assert!(smart_resize(1, 300, 32, 0, 1 << 24).is_err());
    }
}
