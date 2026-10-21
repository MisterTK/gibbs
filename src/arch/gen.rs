//! Constant and input generators: spec `constants: {name: {gen, args}}` and runner-filled
//! `inputs: [{name, gen, args}]` call [`generate`] with evaluated args. Each generator is named by
//! algorithm/layout and reproduces its source bit for bit (same f32/f64 arithmetic order).
//!
//! Index tables (positions, gather indices) are U32; everything else F32. Position ids are the
//! flattened `[tokens * 3]` layout `MRope` reads unless stated otherwise.

use serde_json::Value;
use std::f64::consts::PI;

pub enum GenData {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

pub struct Gen {
    pub shape: Vec<usize>,
    pub data: GenData,
}

fn f32s(shape: Vec<usize>, v: Vec<f32>) -> Result<Gen, String> {
    Ok(Gen { shape, data: GenData::F32(v) })
}

fn u32s(shape: Vec<usize>, v: Vec<u32>) -> Result<Gen, String> {
    Ok(Gen { shape, data: GenData::U32(v) })
}

/// Typed access to the evaluated args object.
struct Args<'a>(&'a Value);

impl Args<'_> {
    fn get(&self, k: &str) -> Option<&Value> {
        self.0.get(k).filter(|v| !v.is_null())
    }
    fn f(&self, k: &str) -> Result<f64, String> {
        let v = self.get(k).ok_or_else(|| format!("missing arg `{k}`"))?;
        v.as_f64().ok_or_else(|| format!("arg `{k}` is not a number: {v}"))
    }
    fn opt_f(&self, k: &str) -> Result<Option<f64>, String> {
        self.get(k).map(|_| self.f(k)).transpose()
    }
    fn f_or(&self, k: &str, d: f64) -> Result<f64, String> {
        Ok(self.opt_f(k)?.unwrap_or(d))
    }
    fn uint(k: &str, v: &Value) -> Result<usize, String> {
        v.as_u64()
            .or_else(|| v.as_f64().filter(|x| x.fract() == 0.0 && *x >= 0.0).map(|x| x as u64))
            .map(|x| x as usize)
            .ok_or_else(|| format!("arg `{k}` is not a non-negative integer: {v}"))
    }
    fn u(&self, k: &str) -> Result<usize, String> {
        Self::uint(k, self.get(k).ok_or_else(|| format!("missing arg `{k}`"))?)
    }
    fn u_or(&self, k: &str, d: usize) -> Result<usize, String> {
        self.get(k).map_or(Ok(d), |_| self.u(k))
    }
    fn b_or(&self, k: &str, d: bool) -> Result<bool, String> {
        self.get(k).map_or(Ok(d), |v| v.as_bool().ok_or_else(|| format!("arg `{k}` is not a bool: {v}")))
    }
    fn s_or<'b>(&'b self, k: &str, d: &'b str) -> Result<&'b str, String> {
        self.get(k).map_or(Ok(d), |v| v.as_str().ok_or_else(|| format!("arg `{k}` is not a string: {v}")))
    }
    fn list(&self, k: &str) -> Result<&Vec<Value>, String> {
        self.get(k)
            .ok_or_else(|| format!("missing arg `{k}`"))?
            .as_array()
            .ok_or_else(|| format!("arg `{k}` is not a list"))
    }
    fn us(&self, k: &str) -> Result<Vec<usize>, String> {
        self.list(k)?.iter().map(|v| Self::uint(k, v)).collect()
    }
    fn us_or(&self, k: &str) -> Result<Vec<usize>, String> {
        self.get(k).map_or(Ok(Vec::new()), |_| self.us(k))
    }
    fn fs(&self, k: &str) -> Result<Vec<f64>, String> {
        self.list(k)?
            .iter()
            .map(|v| v.as_f64().ok_or_else(|| format!("arg `{k}` has a non-number: {v}")))
            .collect()
    }
}

fn bad(k: &str, v: impl std::fmt::Display) -> String {
    format!("arg `{k}`: unsupported value `{v}`")
}

/// Runs generator `name`. Unknown generator or bad/missing arg -> `Err` naming both.
pub fn generate(name: &str, args: &Value) -> Result<Gen, String> {
    let a = &Args(args);
    let r = match name {
        "rope_inv_freq" => rope_inv_freq(a),
        "yarn_attention_factor" => yarn_attention_factor(a),
        "axis_freqs" => axis_freqs(a),
        "sinusoid_table" => sinusoid_table(a),
        // `sinusoid_table{kind: tensor2tensor, rows: left+1, reverse: true}`; args `left`, `dim`
        // (`gemma4/audio.rs` relative positions `left..=0`).
        "rel_pos_sinusoid" => sinusoid_table(&Args(&serde_json::json!({
            "kind": "tensor2tensor", "rows": a.u("left").map_err(|e| format!("generator `{name}`: {e}"))? + 1,
            "dim": a.0.get("dim"), "reverse": true}))),
        "timestep_embedding" => timestep_embedding(a),
        "modulation_index_rows" => modulation_index_rows(a),
        "text_zero_grid3" => text_zero_grid3(a),
        // `text_zero_grid3` with `text_axis: false` (`krea2/graph_program.rs`); args `text_len`,
        // `grid_h`, `grid_w`.
        "text_zero_grid_rowcol" => {
            let mut o = args.as_object().cloned().unwrap_or_default();
            o.insert("text_axis".into(), Value::Bool(false));
            text_zero_grid3(&Args(&Value::Object(o)))
        }
        "caption_then_grid_padded" => caption_then_grid_padded(a),
        "centered_grid_text_after" => centered_grid_text_after(a),
        "text_then_centered_grid" => text_then_centered_grid(a),
        "video_rope_cos_sin" => video_rope_cos_sin(a),
        "centered_crop_rows" => centered_crop_rows(a),
        "grid_xy" => grid_xy(a),
        "bilinear_pos_taps" => bilinear_pos_taps(a),
        "vision_rope_positions" => vision_rope_positions(a),
        "dft_matrix" => dft_matrix(a),
        "mel_filters_htk" => mel_filters_htk(a),
        "local_window_bias" => local_window_bias(a),
        "rel_score_index" => rel_score_index(a),
        "validity_mask" => validity_mask(a),
        "log_bucket_rel_index" => log_bucket_rel_index(a),
        "causal_key_mask" => causal_key_mask(a),
        "channel_dup_index" => channel_dup_index(a),
        "ones" => fill_with(a, 1.0),
        "zeros" => fill_with(a, 0.0),
        "fill" => a.f("value").and_then(|v| fill_with(a, v)),
        "iota" => iota(a),
        "ngram_hash_rows" => ngram_hash_rows(a),
        _ => return Err(format!("unknown generator `{name}`")),
    };
    r.map_err(|e| format!("generator `{name}`: {e}"))
}

/// `rope_inv_freq`: RoPE inverse frequencies, F32 `[dim / 2]`.
/// Args: `dim` (rotary dim), `theta`, `kind` (default `default`), `precision` (`f64` default;
/// `f32` computes in f32 like ModernBERT / GraphLm, only `default` and `yarn`).
/// - `default`: `1 / theta^(2i/dim)`.
/// - `linear`: `1 / (factor * theta^(2i/dim))` (`factor`).
/// - `ntk_alpha`: `default` with `theta * alpha^(dim / (dim - 2))` (`alpha`; `dim` = head dim).
/// - `proportional`: the first `floor(partial_rotary_factor * dim / 2)` pairs get
///   `1 / theta^(2i/dim) / factor`, the rest 0 (`partial_rotary_factor`, `factor`, both default 1).
/// - `wavelen_band` (transformers `llama3` rope_type): `factor`, `low_freq_factor`,
///   `high_freq_factor`, `original_max_position_embeddings`
///   (each default 1).
/// - `yarn`: `factor`, `original_max_position_embeddings`, `beta_fast` (32), `beta_slow` (1),
///   `truncate` (true). Zero betas are NOT mapped to the defaults (Gemma3's loader does that; its
///   spec must). The cos/sin scale comes from `yarn_attention_factor`.
///
/// `tile` (default 1): the table repeated that many times, F32 `[tile * dim / 2]` (a 2D ViT
/// rotary rotates by grid row with one copy and by grid column with the next).
///
/// Sources: `gemma3_text.rs` `inv_freq`, `classifier/modernbert.rs` `rope_inv_freq`,
/// `gemma4/text.rs` `inv_freq`, `qwen3_5/text.rs` (default / HunYuan alpha),
/// `qwen3_5/draft.rs` (wavelen_band), `graph_lm.rs` (f32 default).
fn rope_inv_freq(a: &Args) -> Result<Gen, String> {
    let (dim, theta) = (a.u("dim")?, a.f("theta")?);
    let half = dim / 2;
    let kind = a.s_or("kind", "default")?;
    let precision = a.s_or("precision", "f64")?;
    let pow = |t: f64| (0..half).map(move |i| t.powf(2.0 * i as f64 / dim as f64));
    let inv: Vec<f32> = match (kind, precision) {
        ("default", "f64") => pow(theta).map(|p| (1.0 / p) as f32).collect(),
        ("default", "f32") => (0..half)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / dim as f32))
            .collect(),
        ("linear", "f64") => {
            let factor = a.f("factor")?;
            pow(theta).map(|p| (1.0 / (factor * p)) as f32).collect()
        }
        ("ntk_alpha", "f64") => {
            let t = theta * a.f("alpha")?.powf(dim as f64 / (dim as f64 - 2.0));
            pow(t).map(|p| (1.0 / p) as f32).collect()
        }
        ("proportional", "f64") => {
            let prf = a.f_or("partial_rotary_factor", 1.0)?;
            let factor = a.f_or("factor", 1.0)?;
            let rotated = (prf * dim as f64 / 2.0) as usize;
            pow(theta)
                .enumerate()
                .map(|(i, p)| if i < rotated { (1.0 / p / factor) as f32 } else { 0.0 })
                .collect()
        }
        ("wavelen_band", "f64") => {
            let (factor, lo, hi, orig) = (
                a.f_or("factor", 1.0)?,
                a.f_or("low_freq_factor", 1.0)?,
                a.f_or("high_freq_factor", 1.0)?,
                a.f_or("original_max_position_embeddings", 1.0)?,
            );
            pow(theta)
                .map(|p| {
                    let f = 1.0 / p;
                    let wavelen = 2.0 * PI / f;
                    (if wavelen < orig / hi {
                        f
                    } else if wavelen > orig / lo {
                        f / factor
                    } else {
                        let s = (orig / wavelen - lo) / (hi - lo);
                        (1.0 - s) * f / factor + s * f
                    }) as f32
                })
                .collect()
        }
        ("yarn", _) => {
            let factor = a.f("factor")?;
            let orig = a.f("original_max_position_embeddings")?;
            let (beta_fast, beta_slow) = (a.f_or("beta_fast", 32.0)?, a.f_or("beta_slow", 1.0)?);
            let corr = |rot: f64| dim as f64 * (orig / (rot * 2.0 * PI)).ln() / (2.0 * theta.ln());
            let (mut low, mut high) = (corr(beta_fast), corr(beta_slow));
            if a.b_or("truncate", true)? {
                (low, high) = (low.floor(), high.ceil());
            }
            match precision {
                "f64" => {
                    let (low, mut high) = (low.max(0.0), high.min(dim as f64 - 1.0));
                    if low == high {
                        high += 0.001;
                    }
                    pow(theta)
                        .enumerate()
                        .map(|(i, p)| {
                            let (extra, inter) = ((1.0 / (1.0 * p)) as f32, (1.0 / (factor * p)) as f32);
                            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0) as f32;
                            let ef = 1.0 - ramp;
                            inter * (1.0 - ef) + extra * ef
                        })
                        .collect()
                }
                "f32" => {
                    let (low, mut high) = (low.max(0.0) as f32, high.min(dim as f64 - 1.0) as f32);
                    if low == high {
                        high += 0.001;
                    }
                    (0..half)
                        .map(|i| {
                            let pf = (theta as f32).powf((2 * i) as f32 / dim as f32);
                            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                            let extra = 1.0 - ramp;
                            let inter = 1.0 / (factor as f32 * pf);
                            inter * (1.0 - extra) + (1.0 / pf) * extra
                        })
                        .collect()
                }
                p => return Err(bad("precision", p)),
            }
        }
        (_, "f64" | "f32") => return Err(bad("kind", format!("{kind} ({precision})"))),
        (_, p) => return Err(bad("precision", p)),
    };
    let tile = a.u_or("tile", 1)?;
    f32s(vec![tile * half], inv.repeat(tile))
}

/// `yarn_attention_factor`: YaRN cos/sin scale, F32 `[1]`. Args: `factor`, optional
/// `attention_factor` (wins), `mscale` + `mscale_all_dim` (used when both are non-zero).
/// Source: `gemma3_text.rs` `inv_freq` / `modernbert.rs` `rope_inv_freq` (second result).
fn yarn_attention_factor(a: &Args) -> Result<Gen, String> {
    let factor = a.f("factor")?;
    let ms = |m: f64| if factor <= 1.0 { 1.0 } else { 0.1 * m * factor.ln() + 1.0 };
    let nz = |k: &str| a.opt_f(k).map(|v| v.filter(|&m| m != 0.0));
    let attn = match (a.opt_f("attention_factor")?, nz("mscale")?, nz("mscale_all_dim")?) {
        (Some(x), _, _) => x,
        (None, Some(m), Some(d)) => ms(m) / ms(d),
        _ => ms(1.0),
    };
    f32s(vec![1], vec![attn as f32])
}

/// `axis_freqs`: per-pair frequencies of a multi-axis RoPE, F32 `[sum(dims) / 2]`: axis `a`
/// contributes `theta^(-2i/dims[a])`, or zeros (identity rotation) when `a` is in
/// `identity_axes` (default `[]`). Args: `dims` (each even), `theta`, `identity_axes`.
/// FLUX.2's 4-axis table is `dims: [t, h, w, l], identity_axes: [0]` with MRope sections
/// `[(t+h)/2, w/2, l/2]`. Sources: `dit_graph.rs` `axis_freqs`, `flux/graph_program.rs` `rope_tables`.
fn axis_freqs(a: &Args) -> Result<Gen, String> {
    let (dims, theta, id) = (a.us("dims")?, a.f("theta")?, a.us_or("identity_axes")?);
    let mut f = Vec::new();
    for (ax, &dim) in dims.iter().enumerate() {
        for i in 0..dim / 2 {
            f.push(if id.contains(&ax) { 0.0 } else { theta.powf(-((2 * i) as f64) / dim as f64) as f32 });
        }
    }
    f32s(vec![f.len()], f)
}

/// `sinusoid_table`: sinusoid position table, F32 `[rows, dim]`, `[sin | cos]` halves.
/// Args: `rows`, `dim`, `kind`, optional `zero_row` (left all-zero: a padding id), `reverse`
/// (row r holds position `rows-1-r`; default false), `start` (added to every row's position;
/// default 0: fairseq tables read from position `2` drop the offset rows).
/// - `standard`: angle `p / 10000^(2i/dim)` (`seq2seq.rs` `sinusoidal_marian`).
/// - `tensor2tensor`: angle `p * exp(-i ln(10000) / (dim/2 - 1))` (`seq2seq.rs`
///   `sinusoidal_fairseq` with `zero_row: 1`; `gemma4/audio.rs` relative positions
///   `left..=0` with `rows: left + 1, reverse: true`).
fn sinusoid_table(a: &Args) -> Result<Gen, String> {
    let (rows, d) = (a.u("rows")?, a.u("dim")?);
    let zero = a.get("zero_row").map(|_| a.u("zero_row")).transpose()?;
    let reverse = a.b_or("reverse", false)?;
    let start = a.get("start").map(|_| a.u("start")).transpose()?.unwrap_or(0);
    let half = d / 2;
    let kind = a.s_or("kind", "standard")?;
    let step = 10000f64.ln() / (half as f64 - 1.0);
    let angle: Box<dyn Fn(usize, usize) -> f64> = match kind {
        "standard" => Box::new(|p, i| p as f64 / 10000f64.powf(2.0 * i as f64 / d as f64)),
        "tensor2tensor" => Box::new(move |p, i| p as f64 * (-(i as f64) * step).exp()),
        k => return Err(bad("kind", k)),
    };
    let mut t = vec![0.0f32; rows * d];
    for r in 0..rows {
        if zero == Some(r) {
            continue;
        }
        let p = if reverse { rows - 1 - r } else { r } + start;
        for i in 0..half {
            let x = angle(p, i);
            t[r * d + i] = x.sin() as f32;
            t[r * d + half + i] = x.cos() as f32;
        }
    }
    f32s(vec![rows, d], t)
}

/// One timestep feature vector (diffusers `get_timestep_embedding`), `dim` long (odd: last 0).
/// Shaped like `dit.rs` / `flux.rs` (collect angles, then map cos / sin): codegen of the f32
/// `exp`/`cos`/`sin` calls depends on loop shape and can differ by an ulp from other shapes.
fn timestep_vec(t: f64, dim: usize, flip: bool, shift: f64, scale: f64, max_period: f64, f32p: bool) -> Vec<f32> {
    let half = dim / 2;
    let (cos, sin): (Vec<f32>, Vec<f32>) = if f32p {
        let (t, ln, sh, sc) = (t as f32, max_period.ln() as f32, shift as f32, scale as f32);
        let args: Vec<f32> = (0..half)
            .map(|i| t * (-ln * i as f32 / (half as f32 - sh)).exp() * sc)
            .collect();
        (args.iter().map(|a| a.cos()).collect(), args.iter().map(|a| a.sin()).collect())
    } else {
        let args: Vec<f64> = (0..half)
            .map(|i| t * (-(max_period.ln()) * i as f64 / (half as f64 - shift)).exp() * scale)
            .collect();
        (
            args.iter().map(|a| a.cos() as f32).collect(),
            args.iter().map(|a| a.sin() as f32).collect(),
        )
    };
    let mut out = if flip { [cos, sin].concat() } else { [sin, cos].concat() };
    out.resize(dim, 0.0);
    out
}

/// `timestep_embedding`: sinusoidal timestep features, F32 `[dim]`.
/// Args: `t`, `dim`, `flip_sin_to_cos` (true: cos half first), `downscale_freq_shift` (0),
/// `scale` (1), `max_period` (10000), `precision` (`f64` default; `f32` for sources computing in f32).
/// Reproduces `unet.rs` `timestep_embedding` and `dit.rs` `timestep_features` with
/// `precision: "f32"`, `flux.rs` `sinusoid` with the defaults.
fn timestep_embedding(a: &Args) -> Result<Gen, String> {
    let dim = a.u("dim")?;
    let f32p = match a.s_or("precision", "f64")? {
        "f64" => false,
        "f32" => true,
        p => return Err(bad("precision", p)),
    };
    let v = timestep_vec(
        a.f("t")?,
        dim,
        a.b_or("flip_sin_to_cos", true)?,
        a.f_or("downscale_freq_shift", 0.0)?,
        a.f_or("scale", 1.0)?,
        a.f_or("max_period", 10000.0)?,
        f32p,
    );
    f32s(vec![dim], v)
}

/// `modulation_index_rows`: input rows of a distilled modulation-table MLP, F32
/// `[rows, sum(dims)]`: row i = `[emb(t*scale, dims[0]), emb(0, dims[1]), emb(i*scale, dims[2])]`,
/// `emb` = f64 cos-first `timestep_embedding`. Args: `t`, `rows`, `dims` (3), `scale`.
/// Source: `flux/graph_program.rs` `Approx::input_rows` (`dims: [16, 16, 32], scale: 1000`).
fn modulation_index_rows(a: &Args) -> Result<Gen, String> {
    let (t, rows, scale, dims) = (a.f("t")?, a.u("rows")?, a.f("scale")?, a.us("dims")?);
    let [dt, dg, di] = dims[..] else { return Err(bad("dims", format!("{dims:?}"))) };
    let emb = |x: f64, d| timestep_vec(x, d, true, 0.0, 1.0, 10000.0, false);
    let (tp, gp) = (emb(t * scale, dt), emb(0.0, dg));
    let mut out = Vec::with_capacity(rows * (dt + dg + di));
    for i in 0..rows {
        out.extend(&tp);
        out.extend(&gp);
        out.extend(emb(i as f64 * scale, di));
    }
    f32s(vec![rows, dt + dg + di], out)
}

/// `text_zero_grid3`: U32 `[(text_len + grid_h*grid_w) * 3]`: text tokens then the row-major
/// grid. `text_axis: false` (default): text `(0,0,0)`, grid `(0,row,col)` (FLUX.1, Krea2);
/// `text_axis: true`: the FLUX.2 4-axis ids `(t,h,w,l)` read on axes `(h,w,l)`: text `(0,0,l)`,
/// grid `(row,col,0)`. Sources: `flux.rs` `position_ids` + `dit_graph::positions`,
/// `krea2/graph_program.rs` `forward_gpu`.
fn text_zero_grid3(a: &Args) -> Result<Gen, String> {
    let (nt, h, w, axis) = (a.u("text_len")?, a.u("grid_h")?, a.u("grid_w")?, a.b_or("text_axis", false)?);
    let mut v = Vec::with_capacity(3 * (nt + h * w));
    for l in 0..nt as u32 {
        v.extend(if axis { [0, 0, l] } else { [0; 3] });
    }
    for i in 0..h * w {
        let (r, c) = ((i / w) as u32, (i % w) as u32);
        v.extend(if axis { [r, c, 0] } else { [0, r, c] });
    }
    u32s(vec![v.len()], v)
}

/// `caption_then_grid_padded`: U32 `(frame,row,col)` ids of a padded image grid
/// (`grid_total` rows, the first `grid_len` real: `(text_total+1, t/grid_w, t%grid_w)`, pads 0)
/// and padded caption tokens (`text_total` rows `(1+t, 0, 0)`). `part`: `grid`, `text` or `all`
/// (default: grid then text). Source: `zimage.rs` `token_pos`.
fn caption_then_grid_padded(a: &Args) -> Result<Gen, String> {
    let (wp, n_img, img_total, cap_total) =
        (a.u("grid_w")?, a.u("grid_len")?, a.u("grid_total")?, a.u("text_total")?);
    let grid = || {
        (0..img_total).flat_map(move |t| {
            if t < n_img {
                [cap_total + 1, t / wp, t % wp].map(|x| x as u32)
            } else {
                [0; 3]
            }
        })
    };
    let text = || (0..cap_total).flat_map(|t| [1 + t as u32, 0, 0]);
    let v: Vec<u32> = match a.s_or("part", "all")? {
        "grid" => grid().collect(),
        "text" => text().collect(),
        "all" => grid().chain(text()).collect(),
        p => return Err(bad("part", p)),
    };
    u32s(vec![v.len()], v)
}

/// Each axis shifted so its minimum is `min(0, min)`-based 0 (`qwen_image.rs` `rope_positions`).
fn shift_nonneg(pos: &[[i32; 3]]) -> Vec<u32> {
    let lo = [0, 1, 2].map(|a| pos.iter().map(|p| p[a]).min().unwrap_or(0).min(0));
    pos.iter().flat_map(|p| [0, 1, 2].map(|a| (p[a] - lo[a]) as u32)).collect()
}

fn centered_grid(h: usize, w: usize, frame: i32) -> impl Iterator<Item = [i32; 3]> {
    let (oy, ox) = ((h - h / 2) as i32, (w - w / 2) as i32);
    (0..h as i32).flat_map(move |y| (0..w as i32).map(move |x| [frame, y - oy, x - ox]))
}

/// `centered_grid_text_after`: U32 `[(text_len + grid_h*grid_w) * 3]`: text `m+t` on all axes
/// (`m = max(grid_h/2, grid_w/2)`), then the grid at frame 0 centred on zero; each axis shifted
/// non-negative. Args: `text_len`, `grid_h`, `grid_w`. Source: `qwen_image_v1.rs` `positions` +
/// `qwen_image.rs` `rope_positions`.
fn centered_grid_text_after(a: &Args) -> Result<Gen, String> {
    let (nt, h, w) = (a.u("text_len")?, a.u("grid_h")?, a.u("grid_w")?);
    let m = (h / 2).max(w / 2) as i32;
    let pos: Vec<[i32; 3]> = (0..nt as i32).map(|t| [m + t; 3]).chain(centered_grid(h, w, 0)).collect();
    let v = shift_nonneg(&pos);
    u32s(vec![v.len()], v)
}

/// `text_then_centered_grid`: like `centered_grid_text_after` but text `t` on all axes and the
/// grid at frame `text_len`. Source: `qwen_image21.rs` `t2i_positions` + `rope_positions`.
fn text_then_centered_grid(a: &Args) -> Result<Gen, String> {
    let (nt, h, w) = (a.u("text_len")?, a.u("grid_h")?, a.u("grid_w")?);
    let pos: Vec<[i32; 3]> = (0..nt as i32).map(|p| [p; 3]).chain(centered_grid(h, w, nt as i32)).collect();
    let v = shift_nonneg(&pos);
    u32s(vec![v.len()], v)
}

/// `video_rope_cos_sin`: interleaved-pair video RoPE table, F32 `[frames*grid_h*grid_w, dim]`
/// (`output`: `cos` or `sin`). Per axis `dim/6` freqs `theta^(j/(n-1)) * pi/2` times
/// `2 * pos * axis_scale / base - 1`, axes interleaved, each angle twice, `dim % 6` leading
/// identity entries. Args: `frames`, `grid_h`, `grid_w`, `fps`, `dim`, `theta` (10000),
/// `base` (`[20, 2048, 2048]`), `temporal_scale` (8; divided by `fps`), `spatial_scale` (32).
/// Source: `ltx.rs` `Rope::new`.
fn video_rope_cos_sin(a: &Args) -> Result<Gen, String> {
    let (f, h, w, d) = (a.u("frames")?, a.u("grid_h")?, a.u("grid_w")?, a.u("dim")?);
    let fps = a.f("fps")?;
    let theta = a.f_or("theta", 10000.0)?;
    let base = match a.get("base") {
        Some(_) => a.fs("base")?,
        None => vec![20.0, 2048.0, 2048.0],
    };
    if base.len() != 3 {
        return Err(bad("base", format!("{base:?}")));
    }
    let (ts, ss) = (a.f_or("temporal_scale", 8.0)?, a.f_or("spatial_scale", 32.0)?);
    let cos = match a.s_or("output", "")? {
        "cos" => true,
        "sin" => false,
        o => return Err(bad("output", o)),
    };
    let n = d / 6;
    let freqs: Vec<f64> = (0..n).map(|j| theta.powf(j as f64 / (n - 1) as f64) * std::f64::consts::FRAC_PI_2).collect();
    let scale = [ts / fps, ss, ss];
    let mut v = Vec::with_capacity(f * h * w * d);
    for t in 0..f * h * w {
        let pos = [t / (h * w), (t / w) % h, t % w];
        v.extend(std::iter::repeat(if cos { 1.0 } else { 0.0 }).take(d % 6));
        for &fr in &freqs {
            for axis in 0..3 {
                let g = pos[axis] as f64 * scale[axis] / base[axis];
                let x = fr * (g * 2.0 - 1.0);
                v.extend([(if cos { x.cos() } else { x.sin() }) as f32; 2]);
            }
        }
    }
    f32s(vec![f * h * w, d], v)
}

/// `centered_crop_rows`: U32 `[grid_h*grid_w]` rows of a `[pos_max^2, d]` learned position table
/// for the centred `grid_h x grid_w` crop. Source: `sd3/graph_program.rs` `forward_gpu`.
fn centered_crop_rows(a: &Args) -> Result<Gen, String> {
    let (pm, hp, wp) = (a.u("pos_max")?, a.u("grid_h")?, a.u("grid_w")?);
    if hp > pm || wp > pm {
        return Err(bad("grid_h/grid_w", format!("{hp}x{wp} exceeds pos_max {pm}")));
    }
    let (top, left) = ((pm - hp) / 2, (pm - wp) / 2);
    let v: Vec<u32> = (0..hp).flat_map(|i| (0..wp).map(move |j| ((top + i) * pm + left + j) as u32)).collect();
    u32s(vec![v.len()], v)
}

/// `grid_xy`: U32 `[grid_h*grid_w*repeat]`: per row-major patch (each repeated `repeat` times,
/// default 1, e.g. once per head) its column (`axis: 0`) or row (`axis: 1`). `repeat: 1` gives
/// the gather rows of per-axis learned position tables. Source: `gemma4/vision.rs`
/// `pos_rows` (table rows `gx` / `gy`) and `forward_gpu` `rope_pos`.
fn grid_xy(a: &Args) -> Result<Gen, String> {
    let (h, w, axis, rep) = (a.u("grid_h")?, a.u("grid_w")?, a.u("axis")?, a.u_or("repeat", 1)?);
    if axis > 1 || rep == 0 {
        return Err(bad("axis/repeat", format!("{axis}/{rep}")));
    }
    let v: Vec<u32> = (0..h * w * rep)
        .map(|r| if axis == 0 { (r / rep % w) as u32 } else { (r / rep / w) as u32 })
        .collect();
    u32s(vec![v.len()], v)
}

/// Patch coordinates in merge-block order (`qwen3_5/vision.rs` `prepare`).
fn merge_coords(a: &Args) -> Result<(usize, usize, Vec<(usize, usize)>), String> {
    let (gh, gw, m) = (a.u("grid_h")?, a.u("grid_w")?, a.u("merge")?);
    if m == 0 || gh % m != 0 || gw % m != 0 {
        return Err(bad("merge", m));
    }
    let mut c = Vec::with_capacity(gh * gw);
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    c.push((br * m + ir, bc * m + ic));
                }
            }
        }
    }
    Ok((gh, gw, c))
}

/// `bilinear_pos_taps`: one corner of the align-corners bilinear resample of a `side x side`
/// learned position grid onto the `grid_h x grid_w` patch grid, patches in `merge`-block order:
/// `[n]` U32 table rows (`output: "index"`) or F32 weights (`output: "weight"`).
/// `corner`: 0 = (lo,lo), 1 = (lo,hi), 2 = (hi,lo), 3 = (hi,hi) as (row, col).
/// Source: `qwen3_5/vision.rs` `pos_taps` + `encode_gpu`.
fn bilinear_pos_taps(a: &Args) -> Result<Gen, String> {
    let (gh, gw, coords) = merge_coords(a)?;
    let (side, corner) = (a.u("side")?, a.u("corner")?);
    if side == 0 || corner > 3 {
        return Err(bad("side/corner", format!("{side}/{corner}")));
    }
    let taps = |i: usize, size: usize| {
        let src = i as f32 * (side - 1) as f32 / size.saturating_sub(1).max(1) as f32;
        let f = src.floor();
        let lo = (f as usize).min(side - 1);
        (lo, (lo + 1).min(side - 1), src - f)
    };
    let pick = |&(r, c): &(usize, usize)| {
        let ((r0, r1, fr), (c0, c1, fc)) = (taps(r, gh), taps(c, gw));
        [
            (r0 * side + c0, (1.0 - fr) * (1.0 - fc)),
            (r0 * side + c1, (1.0 - fr) * fc),
            (r1 * side + c0, fr * (1.0 - fc)),
            (r1 * side + c1, fr * fc),
        ][corner]
    };
    let n = coords.len();
    match a.s_or("output", "")? {
        "index" => u32s(vec![n], coords.iter().map(|c| pick(c).0 as u32).collect()),
        "weight" => f32s(vec![n], coords.iter().map(|c| pick(c).1).collect()),
        o => Err(bad("output", o)),
    }
}

/// `vision_rope_positions`: U32 `[grid_h*grid_w*3]` `(row, col, 0)` per patch in `merge`-block
/// order. Source: `qwen3_5/vision.rs` `encode_gpu` (`rpos`).
fn vision_rope_positions(a: &Args) -> Result<Gen, String> {
    let (_, _, coords) = merge_coords(a)?;
    let v: Vec<u32> = coords.iter().flat_map(|&(r, c)| [r as u32, c as u32, 0]).collect();
    u32s(vec![v.len()], v)
}

/// `dft_matrix`: periodic-Hann-windowed real DFT, F32 `[2*bins, frame_length]`
/// (`bins = fft_length/2 + 1`): cos rows then sin rows. Source: `gemma4/audio.rs` `Features::new`.
fn dft_matrix(a: &Args) -> Result<Gen, String> {
    let (frame, fft) = (a.u("frame_length")?, a.u("fft_length")?);
    let bins = fft / 2 + 1;
    let tau = 2.0 * PI;
    let mut dft = vec![0.0f32; 2 * bins * frame];
    for k in 0..bins {
        for n in 0..frame {
            let win = 0.5 - 0.5 * (tau * n as f64 / frame as f64).cos();
            let x = tau * (k * n) as f64 / fft as f64;
            dft[k * frame + n] = (win * x.cos()) as f32;
            dft[(bins + k) * frame + n] = (win * x.sin()) as f32;
        }
    }
    f32s(vec![2 * bins, frame], dft)
}

/// `mel_filters_htk`: unnormalised HTK triangular mel filters, F32 `[mel_bins, fft_length/2+1]`.
/// Args: `fft_length`, `mel_bins`, `sampling_rate` (integer Hz), `min_frequency` (0),
/// `max_frequency` (8000). Source: `gemma4/audio.rs` `Features::new`.
fn mel_filters_htk(a: &Args) -> Result<Gen, String> {
    let (fft, mel, rate) = (a.u("fft_length")?, a.u("mel_bins")?, a.u("sampling_rate")?);
    let (fmin, fmax) = (a.f_or("min_frequency", 0.0)?, a.f_or("max_frequency", 8000.0)?);
    let bins = fft / 2 + 1;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let (mmin, mmax) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let pts: Vec<f64> = (0..mel + 2)
        .map(|i| mmin + (mmax - mmin) * i as f64 / (mel + 1) as f64)
        .map(|m| 700.0 * (10f64.powf(m / 2595.0) - 1.0))
        .collect();
    let mut filters = vec![0.0f32; mel * bins];
    for j in 0..mel {
        for i in 0..bins {
            let f = (rate / 2) as f64 * i as f64 / (bins - 1) as f64;
            let down = (f - pts[j]) / (pts[j + 1] - pts[j]);
            let up = (pts[j + 2] - f) / (pts[j + 2] - pts[j + 1]);
            filters[j * bins + i] = down.min(up).max(0.0) as f32;
        }
    }
    f32s(vec![mel, bins], filters)
}

/// Local-attention pairs: query `i` sees keys `i-left < j <= i` with `j < valid`.
fn local_pairs(t: usize, left: usize, valid: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..t).flat_map(move |i| ((i + 1).saturating_sub(left)..=i).filter(move |&j| j < valid).map(move |j| (i, j)))
}

/// `local_window_bias`: F32 `[t, t]` additive bias, 0 on local pairs, else `masked` (-1e30).
/// Args: `t`, `left`, `valid` (valid frames form a prefix). Source: `gemma4/audio.rs` `encode_gpu`.
fn local_window_bias(a: &Args) -> Result<Gen, String> {
    let (t, left, valid) = (a.u("t")?, a.u("left")?, a.u("valid")?);
    let mut bias = vec![a.f_or("masked", -1e30)? as f32; t * t];
    for (i, j) in local_pairs(t, left, valid) {
        bias[i * t + j] = 0.0;
    }
    f32s(vec![t, t], bias)
}

/// `rel_score_index`: U32 `[heads, t, t]` gather index into the flattened `[heads, t, left+1]`
/// relative score table: local pair `(i, j)` of head h reads `(h*t + i)*(left+1) + left-(i-j)`,
/// other pairs 0. Args as `local_window_bias` plus `heads`. Source: `gemma4/audio.rs` `encode_gpu`.
fn rel_score_index(a: &Args) -> Result<Gen, String> {
    let (nh, t, left, valid) = (a.u("heads")?, a.u("t")?, a.u("left")?, a.u("valid")?);
    let mut idx = vec![0u32; nh * t * t];
    for (i, j) in local_pairs(t, left, valid) {
        for h in 0..nh {
            idx[(h * t + i) * t + j] = ((h * t + i) * (left + 1) + left - (i - j)) as u32;
        }
    }
    u32s(vec![nh, t, t], idx)
}

/// `validity_mask`: F32 `[rows, cols]`, 1 on the first `valid_rows` rows, else 0.
/// Source: `gemma4/audio.rs` `encode_gpu` (`mask_plane`; the frame mask is a prefix).
fn validity_mask(a: &Args) -> Result<Gen, String> {
    let (rows, cols, valid) = (a.u("rows")?, a.u("cols")?, a.u("valid_rows")?);
    let v = (0..rows * cols).map(|i| (i / cols.max(1) < valid) as u8 as f32).collect();
    f32s(vec![rows, cols], v)
}

/// `log_bucket_rel_index`: U32 `[rows, cols]` index into `2*span` relative embeddings for query
/// `i`, key `j` (transformers `make_log_bucket_position` in f32 when `buckets > 0`, then `+span`
/// clamped to `[0, 2*span)`). Args: `rows`, `cols` (default rows), `buckets`, `max_relative`,
/// `span`. Source: `classifier.rs` DeBERTa `rel_index`.
fn log_bucket_rel_index(a: &Args) -> Result<Gen, String> {
    let (rows, buckets, span) = (a.u("rows")?, a.u("buckets")?, a.u("span")? as i64);
    let cols = a.u_or("cols", rows)?;
    let max_rel = if buckets > 0 { a.u("max_relative")? } else { 0 };
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let rel = i as i64 - j as i64;
            let mid = (buckets / 2) as i64;
            let bucket = if buckets > 0 && rel.abs() > mid {
                let log_pos = ((rel.abs() as f32 / mid as f32).ln()
                    / ((max_rel - 1) as f32 / mid as f32).ln()
                    * (mid - 1) as f32)
                    .ceil()
                    + mid as f32;
                rel.signum() * log_pos as i64
            } else {
                rel
            };
            v.push((bucket + span).clamp(0, 2 * span - 1) as u32);
        }
    }
    u32s(vec![rows, cols], v)
}

/// `causal_key_mask`: F32 `[cap]`, 0 for keys `j <= pos`, else `masked` (default -inf).
/// Source: the former `encdec_gpu.rs` `step`.
fn causal_key_mask(a: &Args) -> Result<Gen, String> {
    let (cap, pos) = (a.u("cap")?, a.u("pos")?);
    let m = a.f_or("masked", f64::NEG_INFINITY)? as f32;
    f32s(vec![cap], (0..cap).map(|j| if j <= pos { 0.0 } else { m }).collect())
}

/// `channel_dup_index`: U32 `[channels*4]` gather index of a channel-duplicating 2x upsample:
/// output `co*4 + k` reads `((co*frames + frames-1)*4 + k) / repeats`. Args: `channels`,
/// `repeats`, `frames`. Source: `qwen_vae21.rs` `dup_idx`.
fn channel_dup_index(a: &Args) -> Result<Gen, String> {
    let (c, rep, ft) = (a.u("channels")?, a.u("repeats")?, a.u("frames")?);
    if rep == 0 || ft == 0 {
        return Err(bad("repeats/frames", format!("{rep}/{ft}")));
    }
    let v = (0..c * 4).map(|i| (((i / 4 * ft + ft - 1) * 4 + i % 4) / rep) as u32).collect();
    u32s(vec![c * 4], v)
}

/// `ones` / `zeros` / `fill`: F32 of `shape` filled with 1, 0 or `value`.
fn fill_with(a: &Args, v: f64) -> Result<Gen, String> {
    let shape = a.us("shape")?;
    let n = shape.iter().product();
    f32s(shape, vec![v as f32; n])
}

/// `iota`: `[n]` of `start + i*step` (defaults 0, 1); `dtype` `u32` (default) or `f32`.
fn iota(a: &Args) -> Result<Gen, String> {
    let n = a.u("n")?;
    match a.s_or("dtype", "u32")? {
        "u32" => {
            let (s, st) = (a.u_or("start", 0)? as u32, a.u_or("step", 1)? as u32);
            u32s(vec![n], (0..n as u32).map(|i| s + i * st).collect())
        }
        "f32" => {
            let (s, st) = (a.f_or("start", 0.0)? as f32, a.f_or("step", 1.0)? as f32);
            f32s(vec![n], (0..n).map(|i| s + i as f32 * st).collect())
        }
        d => Err(bad("dtype", d)),
    }
}

/// `ngram_hash_rows`: U32 `[rows * (ngram - 1) * per_gram]` table rows of hashed n-gram
/// embeddings. `ids` holds up to `ngram - 1` tokens of left context, then the `rows` tokens; missing
/// context (sequence start) is `eos`. Row `t`'s context `c[0..ngram]` is the token and its
/// predecessors, every one at or before an `eos` predecessor replaced by `eos`; n-gram `g` in
/// `2..=ngram` hashes `h = c[0]*m[0] ^ .. ^ c[g-1]*m[g-1]` (u64 wrapping), and its `per_gram` heads
/// `j = (g-2)*per_gram + k` read row `h % vocab_sizes[j] + offsets[j]`. Args: `ids`, `rows`,
/// `ngram`, `per_gram`, `multipliers`, `offsets`, `vocab_sizes`, `eos`.
fn ngram_hash_rows(a: &Args) -> Result<Gen, String> {
    let (rows, ngram, per, eos) = (a.u("rows")?, a.u("ngram")?, a.u("per_gram")?, a.u("eos")? as u64);
    let (mult, off, voc) = (a.us("multipliers")?, a.us("offsets")?, a.us("vocab_sizes")?);
    let ids = a.us("ids")?;
    let heads = ngram.saturating_sub(1) * per;
    if ngram < 2 || mult.len() < ngram || off.len() < heads || voc.len() < heads || voc[..heads].contains(&0) {
        return Err("hash constants do not fit ngram / per_gram".into());
    }
    if ids.len() < rows {
        return Err(format!("{} ids for {rows} rows", ids.len()));
    }
    let back = ngram - 1;
    let pad = back.saturating_sub(ids.len() - rows);
    let chain: Vec<u64> = std::iter::repeat_n(eos, pad)
        .chain(ids[ids.len() - rows - (back - pad)..].iter().map(|&x| x as u64))
        .collect();
    let mut out = Vec::with_capacity(rows * heads);
    let mut ctx = vec![0u64; ngram];
    for t in 0..rows {
        ctx[0] = chain[back + t];
        let mut cut = false;
        for s in 1..ngram {
            let p = chain[back + t - s];
            cut |= p == eos;
            ctx[s] = if cut { eos } else { p };
        }
        for g in 2..=ngram {
            let h = (1..g).fold(ctx[0].wrapping_mul(mult[0] as u64), |h, j| {
                h ^ ctx[j].wrapping_mul(mult[j] as u64)
            });
            for k in 0..per {
                let j = (g - 2) * per + k;
                out.push((h % voc[j] as u64 + off[j] as u64) as u32);
            }
        }
    }
    u32s(vec![rows * heads], out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn f(name: &str, args: Value) -> Vec<f32> {
        match generate(name, &args).unwrap().data {
            GenData::F32(v) => v,
            GenData::U32(_) => panic!("{name}: expected f32"),
        }
    }

    fn u(name: &str, args: Value) -> Vec<u32> {
        match generate(name, &args).unwrap().data {
            GenData::U32(v) => v,
            GenData::F32(_) => panic!("{name}: expected u32"),
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn errors_name_generator_and_arg() {
        let e = generate("nope", &json!({})).err().unwrap();
        assert!(e.contains("nope"), "{e}");
        let e = generate("rope_inv_freq", &json!({"dim": 4})).err().unwrap();
        assert!(e.contains("rope_inv_freq") && e.contains("theta"), "{e}");
        let e = generate("rope_inv_freq", &json!({"dim": 4, "theta": 1e4, "kind": "wavelen_band", "precision": "f32"})).err().unwrap();
        assert!(e.contains("kind"), "{e}");
    }

    // Pinned literals: survive deleting the originals.
    #[test]
    fn pinned_values() {
        let r = |k: Value| f("rope_inv_freq", k);
        assert_eq!(r(json!({"dim": 4, "theta": 10000.0})), [1.0, 0.01]);
        assert_eq!(r(json!({"dim": 4, "theta": 10000.0, "precision": "f32"})), [1.0, 0.01]);
        assert_eq!(r(json!({"dim": 4, "theta": 10000.0, "precision": "f32", "tile": 2})), [1.0, 0.01, 1.0, 0.01]);
        assert_eq!(r(json!({"dim": 4, "theta": 10000.0, "kind": "linear", "factor": 2.0})), [0.5, 0.005]);
        // HunYuan alpha: 10000 * 2^(4/2) = 40000.
        assert_eq!(r(json!({"dim": 4, "theta": 10000.0, "kind": "ntk_alpha", "alpha": 2.0})), [1.0, 0.005]);
        assert_eq!(r(json!({"dim": 8, "theta": 10000.0, "kind": "proportional", "partial_rotary_factor": 0.5})), [1.0, 0.1, 0.0, 0.0]);
        // wavelen_band: wavelen 2pi < 100/4 kept; 200pi > 100/1 slowed by 8.
        assert_eq!(
            r(json!({"dim": 4, "theta": 10000.0, "kind": "wavelen_band", "factor": 8.0, "low_freq_factor": 1.0,
                     "high_freq_factor": 4.0, "original_max_position_embeddings": 100.0})),
            [1.0, (0.01f64 / 8.0) as f32]
        );
        assert_eq!(
            u("text_zero_grid_rowcol", json!({"text_len": 1, "grid_h": 1, "grid_w": 2})),
            u("text_zero_grid3", json!({"text_len": 1, "grid_h": 1, "grid_w": 2}))
        );
        assert_eq!(
            f("rel_pos_sinusoid", json!({"left": 2, "dim": 4})),
            f("sinusoid_table", json!({"kind": "tensor2tensor", "rows": 3, "dim": 4, "reverse": true}))
        );
        assert_eq!(f("yarn_attention_factor", json!({"factor": 1.0})), [1.0]);
        assert_eq!(f("yarn_attention_factor", json!({"factor": 4.0, "attention_factor": 0.5})), [0.5]);
        assert_eq!(f("axis_freqs", json!({"dims": [2, 4], "theta": 10000.0, "identity_axes": [0]})), [0.0, 1.0, 0.01]);
        let s = f("sinusoid_table", json!({"kind": "standard", "rows": 2, "dim": 2}));
        assert_eq!(s, [0.0, 1.0, 1f64.sin() as f32, 1f64.cos() as f32]);
        // gemma4 audio relative positions left..=0 (left = 2): last row is position 0.
        let rel = f("sinusoid_table", json!({"kind": "tensor2tensor", "rows": 3, "dim": 4, "reverse": true}));
        assert_eq!(rel[8..], [0.0, 0.0, 1.0, 1.0]);
        let off = f("sinusoid_table", json!({"kind": "tensor2tensor", "rows": 3, "dim": 4, "start": 2}));
        let full = f("sinusoid_table", json!({"kind": "tensor2tensor", "rows": 5, "dim": 4}));
        assert_eq!(off, full[8..]);
        assert_eq!(rel[0], 2f32.sin());
        assert_eq!(rel[1], (2.0 * (-(10000f64.ln())).exp()).sin() as f32);
        assert_eq!(f("timestep_embedding", json!({"t": 0.0, "dim": 4})), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(f("timestep_embedding", json!({"t": 0.0, "dim": 4, "flip_sin_to_cos": false})), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            f("modulation_index_rows", json!({"t": 0.0, "rows": 2, "dims": [2, 2, 2], "scale": 1000.0})),
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1000f64.cos() as f32, 1000f64.sin() as f32]
        );
        // FLUX.1 / Krea2: text at the origin, grid (0, row, col).
        assert_eq!(
            u("text_zero_grid3", json!({"text_len": 1, "grid_h": 1, "grid_w": 2})),
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            u("text_zero_grid3", json!({"text_len": 2, "grid_h": 1, "grid_w": 2, "text_axis": true})),
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            u("caption_then_grid_padded", json!({"grid_w": 1, "grid_len": 1, "grid_total": 2, "text_total": 1})),
            [2, 0, 0, 0, 0, 0, 1, 0, 0]
        );
        // v1: text at max(1,1)=1 then grid rows -1..0, cols -1..0; all axes already >= -1.
        assert_eq!(
            u("centered_grid_text_after", json!({"text_len": 1, "grid_h": 2, "grid_w": 1})),
            [1, 2, 2, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            u("text_then_centered_grid", json!({"text_len": 1, "grid_h": 1, "grid_w": 2})),
            [0, 1, 1, 1, 0, 0, 1, 0, 1]
        );
        let c = f("video_rope_cos_sin", json!({"frames": 1, "grid_h": 1, "grid_w": 1, "fps": 25.0, "dim": 14, "output": "cos"}));
        assert_eq!(c[..4], [1.0, 1.0, (std::f64::consts::FRAC_PI_2 * -1.0).cos() as f32, (std::f64::consts::FRAC_PI_2 * -1.0).cos() as f32]);
        assert_eq!(u("centered_crop_rows", json!({"pos_max": 4, "grid_h": 2, "grid_w": 2})), [5, 6, 9, 10]);
        assert_eq!(u("grid_xy", json!({"grid_h": 2, "grid_w": 3, "axis": 0, "repeat": 2})), [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]);
        assert_eq!(u("grid_xy", json!({"grid_h": 2, "grid_w": 3, "axis": 1})), [0, 0, 0, 1, 1, 1]);
        let taps = json!({"side": 2, "grid_h": 3, "grid_w": 1, "merge": 1, "corner": 0, "output": "index"});
        assert_eq!(u("bilinear_pos_taps", taps.clone()), [0, 0, 2]);
        let mut w = taps.clone();
        w["output"] = json!("weight");
        assert_eq!(f("bilinear_pos_taps", w.clone()), [1.0, 0.5, 1.0]);
        w["corner"] = json!(2);
        assert_eq!(f("bilinear_pos_taps", w), [0.0, 0.5, 0.0]);
        assert_eq!(
            u("vision_rope_positions", json!({"grid_h": 2, "grid_w": 4, "merge": 2}))[..12],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
        );
        let dft = f("dft_matrix", json!({"frame_length": 2, "fft_length": 2}));
        assert_eq!(dft, [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, PI.sin() as f32]);
        let mel = f("mel_filters_htk", json!({"fft_length": 4, "mel_bins": 1, "sampling_rate": 16000}));
        assert_eq!(mel.len(), 3);
        assert!(mel[0] == 0.0 && mel[2].abs() < 1e-12 && mel[1] > 0.0);
        const M: f32 = -1e30;
        assert_eq!(f("local_window_bias", json!({"t": 3, "left": 2, "valid": 2})), [0.0, M, M, 0.0, 0.0, M, M, 0.0, M]);
        assert_eq!(u("rel_score_index", json!({"heads": 1, "t": 3, "left": 2, "valid": 2})), [2, 0, 0, 4, 5, 0, 0, 7, 0]);
        assert_eq!(f("validity_mask", json!({"rows": 2, "cols": 2, "valid_rows": 1})), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(u("log_bucket_rel_index", json!({"rows": 2, "buckets": 0, "span": 2})), [2, 1, 3, 2]);
        assert_eq!(
            u("log_bucket_rel_index", json!({"rows": 6, "cols": 1, "buckets": 4, "max_relative": 8, "span": 8})),
            [8, 9, 10, 11, 11, 11]
        );
        assert_eq!(f("causal_key_mask", json!({"cap": 3, "pos": 1})), [0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(u("channel_dup_index", json!({"channels": 2, "repeats": 2, "frames": 1})), [0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(f("ones", json!({"shape": [2]})), [1.0, 1.0]);
        assert_eq!(f("zeros", json!({"shape": [1, 2]})), [0.0, 0.0]);
        assert_eq!(f("fill", json!({"shape": [1], "value": 3.5})), [3.5]);
        assert_eq!(u("iota", json!({"n": 3, "start": 2, "step": 2})), [2, 4, 6]);
        assert_eq!(f("iota", json!({"n": 2, "dtype": "f32", "step": 0.5})), [0.0, 0.5]);
    }

    // parity with original (delete with the original)
    mod parity {
        use super::*;
        use crate::model;

        #[test]
        fn rope_inv_freq_matches_sources() {
            // gemma3: default, linear, yarn (truncate on/off, explicit factor).
            for (p, dim, truncate) in [
                (json!({"rope_theta": 1e6}), 256, true),
                (json!({"rope_theta": 1e4, "rope_type": "linear", "factor": 8.0}), 128, true),
                (json!({"rope_theta": 1e6, "rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192.0}), 128, true),
                (json!({"rope_theta": 5e5, "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096.0, "beta_fast": 16.0, "beta_slow": 2.0}), 64, false),
            ] {
                let (want, attn) = model::gemma3_text::inv_freq(p.as_object().unwrap(), dim, 0, truncate).unwrap();
                let kind = p.get("rope_type").cloned().unwrap_or(json!("default"));
                let mut args = json!({"dim": dim, "theta": p["rope_theta"], "kind": kind, "truncate": truncate});
                for k in ["factor", "original_max_position_embeddings", "beta_fast", "beta_slow"] {
                    if let Some(v) = p.get(k) {
                        args[k] = v.clone();
                    }
                }
                assert_eq!(bits(&f("rope_inv_freq", args.clone())), bits(&want), "{p}");
                if kind == "yarn" {
                    assert_eq!(f("yarn_attention_factor", args)[0].to_bits(), attn.to_bits());
                }
            }
            // ModernBERT (f32): default and yarn.
            for (cfg, d) in [
                (json!({"global_rope_theta": 160000.0}), 64),
                (json!({"max_position_embeddings": 8192, "rope_scaling": {"rope_type": "yarn", "rope_theta": 160000.0,
                        "original_max_position_embeddings": 1024.0, "mscale": 1.0, "mscale_all_dim": 0.5}}), 64),
            ] {
                let (want, attn) = model::classifier::modernbert::rope_inv_freq(&cfg, "full_attention", d).unwrap();
                let yarn = !cfg["rope_scaling"].is_null();
                let args = if yarn {
                    json!({"dim": d, "theta": 160000.0, "kind": "yarn", "precision": "f32", "factor": 8.0,
                           "original_max_position_embeddings": 1024.0})
                } else {
                    json!({"dim": d, "theta": 160000.0, "precision": "f32"})
                };
                assert_eq!(bits(&f("rope_inv_freq", args)), bits(&want), "{cfg}");
                if yarn {
                    let a = f("yarn_attention_factor", json!({"factor": 8.0, "mscale": 1.0, "mscale_all_dim": 0.5}));
                    assert_eq!(a[0].to_bits(), attn.to_bits());
                }
            }
            // gemma4: default (with factor) and proportional.
            for p in [
                json!({"rope_theta": 1e4, "factor": 2.0}),
                json!({"rope_theta": 1e6, "rope_type": "proportional", "partial_rotary_factor": 0.25}),
            ] {
                let want = model::gemma4::text::inv_freq(&p, 512).unwrap();
                let mut args = json!({"dim": 512, "theta": p["rope_theta"], "kind": "proportional"});
                for k in ["factor", "partial_rotary_factor"] {
                    if let Some(v) = p.get(k) {
                        args[k] = v.clone();
                    }
                }
                assert_eq!(bits(&f("rope_inv_freq", args)), bits(&want), "{p}");
            }
        }

        #[test]
        fn axis_freqs_match_sources() {
            for dims in [vec![16usize, 56, 56], vec![32, 48]] {
                let (want, _) = model::dit_graph::axis_freqs(&dims, 10000.0);
                assert_eq!(bits(&f("axis_freqs", json!({"dims": dims, "theta": 10000.0}))), bits(&want));
            }
            let (want, _, _) = model::flux::graph_program::rope_tables(&[32, 32, 32, 32], 2000.0);
            let got = f("axis_freqs", json!({"dims": [32, 32, 32, 32], "theta": 2000.0, "identity_axes": [0]}));
            assert_eq!(bits(&got), bits(&want));
        }

        #[test]
        fn sinusoid_tables_match_sources() {
            // The former `seq2seq.rs` tables (Marian; fairseq with padding row 1).
            fn table(rows: usize, d: usize, fairseq: bool) -> Vec<f32> {
                let mut t = vec![0.0f32; rows * d];
                let half = d / 2;
                let step = (10000f64).ln() / (half as f64 - 1.0);
                for p in 0..rows {
                    if fairseq && p == 1 {
                        continue;
                    }
                    for i in 0..half {
                        let a = match fairseq {
                            true => p as f64 * (-(i as f64) * step).exp(),
                            false => p as f64 / 10000f64.powf(2.0 * i as f64 / d as f64),
                        };
                        t[p * d + i] = a.sin() as f32;
                        t[p * d + half + i] = a.cos() as f32;
                    }
                }
                t
            }
            for (rows, d) in [(10, 8), (33, 64)] {
                let m = f("sinusoid_table", json!({"kind": "standard", "rows": rows, "dim": d}));
                assert_eq!(bits(&m), bits(&table(rows, d, false)));
                let fs = f("sinusoid_table", json!({"kind": "tensor2tensor", "rows": rows, "dim": d, "zero_row": 1}));
                assert_eq!(bits(&fs), bits(&table(rows, d, true)));
            }
        }

        #[test]
        fn timestep_embedding_matches_sources() {
            for (t, dim) in [(999.0f32, 320), (0.37, 256), (1.0, 7)] {
                let f32a = json!({"t": t, "dim": dim, "precision": "f32"});
                assert_eq!(bits(&f("timestep_embedding", f32a.clone())), bits(&model::unet::timestep_embedding(t, dim)));
                let want = model::dit::timestep_features(t, dim);
                assert_eq!(bits(&f("timestep_embedding", f32a)[..want.len()]), bits(&want));
                let want = model::flux::sinusoid(t as f64 * 1000.0, dim);
                let got = f("timestep_embedding", json!({"t": t as f64 * 1000.0, "dim": dim}));
                assert_eq!(bits(&got[..want.len()]), bits(&want));
            }
            // modulation_index_rows == Approx::input_rows composition.
            let t = 0.3f32;
            let s = model::flux::sinusoid;
            let want: Vec<f32> = (0..3)
                .flat_map(|i| [s(t as f64 * 1000.0, 16), s(0.0, 16), s(i as f64 * 1000.0, 32)].concat())
                .collect();
            let got = f("modulation_index_rows", json!({"t": t as f64, "rows": 3, "dims": [16, 16, 32], "scale": 1000.0}));
            assert_eq!(bits(&got), bits(&want));
        }

        #[test]
        fn positions_match_sources() {
            for (nt, h, w, ax) in [(3, 2, 3, false), (2, 4, 1, true)] {
                let ids: Vec<Vec<usize>> = model::flux::position_ids(nt, h, w, ax).iter().map(|i| i.to_vec()).collect();
                let want = model::dit_graph::positions(&ids, if ax { [1, 2, 3] } else { [0, 1, 2] });
                assert_eq!(u("text_zero_grid3", json!({"text_len": nt, "grid_h": h, "grid_w": w, "text_axis": ax})), want);
            }
            for (wp, ni, it, ct) in [(3, 6, 32, 32), (2, 3, 5, 4)] {
                let (img, cap) = model::zimage::token_pos(wp, ni, it, ct);
                let fl = |v: &[[usize; 3]]| v.iter().flatten().map(|&x| x as u32).collect::<Vec<_>>();
                let a = json!({"grid_w": wp, "grid_len": ni, "grid_total": it, "text_total": ct});
                assert_eq!(u("caption_then_grid_padded", a.clone()), fl(&[img.clone(), cap.clone()].concat()));
                let mut g = a.clone();
                g["part"] = json!("grid");
                assert_eq!(u("caption_then_grid_padded", g), fl(&img));
                let mut t = a;
                t["part"] = json!("text");
                assert_eq!(u("caption_then_grid_padded", t), fl(&cap));
            }
            for (nt, h, w) in [(2, 4, 6), (1, 3, 5)] {
                let args = json!({"text_len": nt, "grid_h": h, "grid_w": w});
                let want = model::qwen_image::rope_positions(&model::qwen_image_v1::positions(nt, h, w));
                assert_eq!(u("centered_grid_text_after", args.clone()), want);
                let want = model::qwen_image::rope_positions(&model::qwen_image21::t2i_positions(nt, h, w));
                assert_eq!(u("text_then_centered_grid", args), want);
            }
            for (fr, h, w, fps, d) in [(2, 3, 4, 25.0, 14), (1, 2, 2, 24.0, 12)] {
                let r = model::ltx::Rope::new(fr, h, w, fps, d);
                let a = |o: &str| json!({"frames": fr, "grid_h": h, "grid_w": w, "fps": fps, "dim": d, "output": o});
                assert_eq!(bits(&f("video_rope_cos_sin", a("cos"))), bits(&r.cos));
                assert_eq!(bits(&f("video_rope_cos_sin", a("sin"))), bits(&r.sin));
            }
        }

        #[test]
        fn audio_front_end_matches_sources() {
            for fe in [
                json!({}),
                json!({"frame_length": 8, "fft_length": 16, "feature_size": 4, "sampling_rate": 8000, "max_frequency": 4000.0, "min_frequency": 20.0}),
            ] {
                let src = model::gemma4::audio::Features::new(&fe).unwrap();
                let g = |k: &str, d: f64| fe.get(k).cloned().unwrap_or(json!(d));
                let dft = f("dft_matrix", json!({"frame_length": g("frame_length", 320.0), "fft_length": g("fft_length", 512.0)}));
                assert_eq!(bits(&dft), bits(&src.dft));
                let mel = f(
                    "mel_filters_htk",
                    json!({"fft_length": g("fft_length", 512.0), "mel_bins": g("feature_size", 128.0),
                           "sampling_rate": g("sampling_rate", 16000.0), "min_frequency": g("min_frequency", 0.0),
                           "max_frequency": g("max_frequency", 8000.0)}),
                );
                assert_eq!(bits(&mel), bits(&src.filters));
            }
        }

        #[test]
        fn indices_match_sources() {
            for (c, r, ft) in [(4, 2, 1), (3, 1, 2)] {
                assert_eq!(
                    u("channel_dup_index", json!({"channels": c, "repeats": r, "frames": ft})),
                    model::qwen_vae21::dup_idx(c, r, ft)
                );
            }
        }
    }

    /// Hand-computed: m = [3, 5, 7], eos 2 cuts the context, missing context is eos.
    #[test]
    fn ngram_hash_rows_cuts_at_eos_and_pads_the_start() {
        let hash = |ids: &[u32]| {
            u("ngram_hash_rows", json!({"ids": ids, "rows": 2, "ngram": 3, "per_gram": 1, "eos": 2,
                "multipliers": [3, 5, 7], "offsets": [0, 10], "vocab_sizes": [7, 11]}))
        };
        assert_eq!(hash(&[9, 4]), [3, 19, 5, 13]);
        assert_eq!(hash(&[7, 2, 9, 4]), [3, 19, 5, 13]);
        assert_eq!(hash(&[6, 9, 4]), [5, 10, 5, 10]);
    }
}
