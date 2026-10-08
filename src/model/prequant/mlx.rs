//! MLX-quantized safetensors (mlx-lm / mlx-vlm `config.json` `quantization`): `X.weight` is uint32
//! words beside `X.scales` (and `X.biases` in affine mode), decoded as mlx `dequantize`
//! (mlx/ops.cpp `affine_dequantize` / `fp_dequantize`). Per-module entries of the `quantization`
//! block (mixed-precision recipes) override the top-level `bits` / `group_size` / `mode`.

use crate::model::par::par_rows;
use crate::model::{decode_dtype, f32_to_bf16, f8_e4m3_to_f32, ModelManifest, VirtualTensor};
use crate::quant::nvfp4::NVFP4_E2M1_LUT;
use serde_json::Value;
use std::path::Path;

/// Manifest metadata key holding the `quantization` block (JSON) once [`attach`] has run.
const META: &str = "mlx.quantization";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Affine,
    Mxfp4,
    Mxfp8,
    Nvfp4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Spec {
    pub mode: Mode,
    pub bits: usize,
    pub group: usize,
}

/// The MLX `quantization` block of a config.json (`quantization`, or a `quantization_config`
/// without `quant_method` that names `bits` / `mode`).
pub(crate) fn block(cfg: &Value) -> Option<&Value> {
    cfg.get("quantization")
        .filter(|q| q.is_object())
        .or_else(|| {
            cfg.get("quantization_config").filter(|q| {
                q["quant_method"].is_null() && (q.get("bits").is_some() || q.get("mode").is_some())
            })
        })
}

/// Mode defaults (mlx `_defaults_for_mode`).
fn defaults(mode: &str) -> Option<Spec> {
    let (mode, bits, group) = match mode {
        "affine" => (Mode::Affine, 4, 64),
        "mxfp4" => (Mode::Mxfp4, 4, 32),
        "mxfp8" => (Mode::Mxfp8, 8, 32),
        "nvfp4" => (Mode::Nvfp4, 4, 16),
        _ => return None,
    };
    Some(Spec { mode, bits, group })
}

/// `to_quantized(**entry)`: an absent `mode` is affine, absent bits / group size take the mode's
/// defaults.
fn parse(e: &Value) -> Result<Spec, String> {
    let mode = e["mode"].as_str().unwrap_or("affine");
    let d = defaults(mode).ok_or_else(|| {
        format!("MLX quantization mode '{mode}' is not supported (affine, mxfp4, mxfp8, nvfp4 are)")
    })?;
    let num = |k: &str, d: usize| match &e[k] {
        Value::Null => Ok(d),
        v => v
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("MLX quantization {k} {v} is not an integer")),
    };
    let s = Spec {
        bits: num("bits", d.bits)?,
        group: num("group_size", d.group)?,
        ..d
    };
    match s.mode {
        Mode::Affine if ![2, 3, 4, 5, 6, 8].contains(&s.bits) || ![32, 64, 128].contains(&s.group) => {
            Err(format!(
                "MLX affine {}-bit with group size {} is not supported (mlx packs 2, 3, 4, 5, 6, 8 bits in groups of 32, 64, 128)",
                s.bits, s.group
            ))
        }
        Mode::Affine => Ok(s),
        _ if s == d => Ok(s),
        _ => Err(format!(
            "MLX {mode} needs {} bits with group size {}, config has {} / {}",
            d.bits, d.group, s.bits, s.group
        )),
    }
}

/// Why an MLX `quantization` block can't load: checks the default and every per-module entry.
pub(crate) fn blocker(q: &Value) -> Option<String> {
    let modules = q.as_object().into_iter().flatten();
    std::iter::once(q)
        .chain(modules.map(|(_, v)| v).filter(|v| v.is_object()))
        .find_map(|e| parse(e).err())
}

/// Spec and logical shape of `name` when it is an MLX-packed weight (U32 beside `X.scales`).
/// `get` looks tensors up by stored name.
pub(crate) fn packed<'a>(
    q: &Value,
    name: &str,
    get: impl Fn(&str) -> Option<&'a VirtualTensor>,
) -> Result<Option<(Spec, Vec<usize>)>, String> {
    let Some(stem) = name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let (Some(w), Some(sc)) = (get(name), get(&format!("{stem}.scales"))) else {
        return Ok(None);
    };
    if w.dtype != "U32" || w.shape.is_empty() {
        return Ok(None);
    }
    let rows: usize = w.shape[..w.shape.len() - 1].iter().product();
    let words = w.byte_len / 4 / rows.max(1);
    let groups = sc.shape.last().copied().unwrap_or(0);
    let spec = match q.get(stem) {
        Some(Value::Bool(false)) => {
            return Err(format!(
                "{name} is packed but config `quantization` marks {stem} unquantized"
            ))
        }
        Some(e) if e.is_object() => parse(e)?,
        // Unlisted modules: the default, or the one fp mode the shapes allow (mlx-lm
        // `infer_quant_config`; only affine keeps scales in the weight dtype).
        _ => {
            let d = parse(q)?;
            let fits = |s: &Spec| words * 32 == groups * s.group * s.bits;
            if fits(&d) || sc.dtype != "U8" {
                d
            } else {
                ["nvfp4", "mxfp4", "mxfp8"]
                    .into_iter()
                    .filter_map(defaults)
                    .find(|s| fits(s))
                    .unwrap_or(d)
            }
        }
    };
    if words * 32 != groups * spec.group * spec.bits {
        return Err(format!(
            "{name}: {words} packed words per row and {groups} scale groups don't fit {}-bit {:?} with group size {}",
            spec.bits, spec.mode, spec.group
        ));
    }
    let mut shape = w.shape.clone();
    *shape.last_mut().unwrap() = words * 32 / spec.bits;
    Ok(Some((spec, shape)))
}

/// One MLX-packed matrix: `rows` packed rows of `cols` logical values; `scales` / `biases` are
/// `(bytes, dtype)`.
#[derive(Clone, Copy)]
pub(crate) struct Packed<'a> {
    pub spec: Spec,
    pub rows: usize,
    pub cols: usize,
    pub w: &'a [u8],
    pub scales: (&'a [u8], &'a str),
    pub biases: Option<(&'a [u8], &'a str)>,
}

impl<'a> Packed<'a> {
    /// Row-major f32 `[rows, cols]`.
    pub(crate) fn decode(&self) -> Result<Vec<f32>, String> {
        let (s, rows, cols, w) = (self.spec, self.rows, self.cols, self.w);
        let row_bytes = cols * s.bits / 8;
        let groups = cols / s.group;
        if w.len() != rows * row_bytes {
            return Err(format!(
                "{} packed bytes for [{rows}, {cols}] at {} bits",
                w.len(),
                s.bits
            ));
        }
        if s.mode != Mode::Affine && self.scales.1 != "U8" {
            return Err(format!(
                "{:?} scales are {}, expected U8",
                s.mode, self.scales.1
            ));
        }
        let scale: Vec<f32> = match s.mode {
            Mode::Affine => decode_dtype(self.scales.0, self.scales.1)?,
            // e8m0: 2^(s - 127); mlx decodes 0 as the bf16 subnormal 2^-127.
            Mode::Mxfp4 | Mode::Mxfp8 => self
                .scales
                .0
                .iter()
                .map(|&e| f32::from_bits(if e == 0 { 1 << 22 } else { (e as u32) << 23 }))
                .collect(),
            Mode::Nvfp4 => self.scales.0.iter().map(|&e| f8_e4m3_to_f32(e)).collect(),
        };
        let bias = match (s.mode, self.biases) {
            (Mode::Affine, Some((b, dt))) => decode_dtype(b, dt)?,
            (Mode::Affine, None) => return Err("affine weight without `biases`".into()),
            _ => Vec::new(),
        };
        if scale.len() != rows * groups || (s.mode == Mode::Affine && bias.len() != scale.len()) {
            return Err(format!(
                "{} scales / {} biases for [{rows}, {groups}] groups",
                scale.len(),
                bias.len()
            ));
        }
        let mask = (1u16 << s.bits) - 1;
        let mut out = vec![0f32; rows * cols];
        par_rows(&mut out, cols, &|r, o| {
            let src = &w[r * row_bytes..][..row_bytes];
            let (sc, bi) = (&scale[r * groups..][..groups], &bias);
            for (c, v) in o.iter_mut().enumerate() {
                let g = c / s.group;
                *v = match s.mode {
                    // One little-endian bitstream per row: element c is bits [c·bits, (c+1)·bits),
                    // lowest first, crossing uint32 words for 3/5/6 bits (mlx `extract_bits`).
                    Mode::Affine => {
                        let bit = c * s.bits;
                        let lo =
                            src[bit / 8] as u16 | (*src.get(bit / 8 + 1).unwrap_or(&0) as u16) << 8;
                        ((lo >> (bit % 8)) & mask) as f32 * sc[g] + bi[r * groups + g]
                    }
                    // E2M1 nibbles, low nibble first.
                    Mode::Mxfp4 | Mode::Nvfp4 => {
                        NVFP4_E2M1_LUT[(src[c / 2] >> (4 * (c % 2)) & 15) as usize] * sc[g]
                    }
                    Mode::Mxfp8 => f8_e4m3_to_f32(src[c]) * sc[g],
                };
            }
        });
        Ok(out)
    }

    /// [`Self::decode`] rounded to bf16 (mlx's output dtype for bf16 scales), a block of rows at a
    /// time so no whole-matrix f32 copy exists (Gemma 4's per-layer embedding is 2.3G values).
    pub(crate) fn decode_bf16(&self) -> Result<Vec<u16>, String> {
        const BLOCK: usize = 1024;
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in (0..self.rows).step_by(BLOCK) {
            let n = BLOCK.min(self.rows - r);
            let cut = |b: &'a [u8]| -> &'a [u8] {
                let per = b.len() / self.rows;
                &b[r * per..(r + n) * per]
            };
            let block = Packed {
                rows: n,
                w: cut(self.w),
                scales: (cut(self.scales.0), self.scales.1),
                biases: self.biases.map(|(b, d)| (cut(b), d)),
                ..*self
            };
            out.extend(block.decode()?.into_iter().map(f32_to_bf16));
        }
        Ok(out)
    }
}

/// Reads the MLX `quantization` block from the config.json beside `weights` (`None`: not MLX).
pub(crate) fn config_beside(weights: &Path) -> Result<Option<Value>, String> {
    let p = weights.with_file_name("config.json");
    let Ok(b) = std::fs::read(&p) else {
        return Ok(None);
    };
    let cfg: Value = serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))?;
    let Some(q) = block(&cfg) else {
        return Ok(None);
    };
    if let Some(why) = blocker(q) {
        return Err(why);
    }
    Ok(Some(q.clone()))
}

/// For checkpoints read whole into a manifest: records the MLX block of the config.json beside
/// `weights` and gives every packed `X.weight` its logical shape; [`dequantize`] then expands it.
pub(crate) fn attach(m: &mut ModelManifest, weights: &Path) -> Result<(), String> {
    let Some(q) = config_beside(weights)? else {
        return Ok(());
    };
    let mut shapes = Vec::new();
    for name in m.tensors.keys() {
        if let Some((_, shape)) = packed(&q, name, |n| m.tensors.get(n))? {
            shapes.push((name.clone(), shape));
        }
    }
    for (name, shape) in shapes {
        m.tensors.get_mut(&name).unwrap().shape = shape;
    }
    m.metadata.insert(META.into(), q.to_string());
    Ok(())
}

/// Expands MLX-packed `name` to f32 `[.., in]`; `None` when the manifest isn't MLX or `name`
/// isn't packed.
pub(crate) fn dequantize(
    m: &ModelManifest,
    data: &[u8],
    name: &str,
) -> Result<Option<Vec<f32>>, String> {
    let Some(q) = m.metadata.get(META) else {
        return Ok(None);
    };
    let q: Value = serde_json::from_str(q).map_err(|e| e.to_string())?;
    let Some((spec, shape)) = packed(&q, name, |n| m.tensors.get(n))? else {
        return Ok(None);
    };
    let stem = &name[..name.len() - ".weight".len()];
    let bytes = |t: &VirtualTensor| {
        data.get(t.byte_offset..t.byte_offset + t.byte_len)
            .ok_or_else(|| format!("tensor '{}' lies outside the weight buffer", t.name))
    };
    let part = |s: &str| -> Result<Option<(&[u8], &str)>, String> {
        m.tensors
            .get(&format!("{stem}.{s}"))
            .map(|t| Ok((bytes(t)?, t.dtype.as_str())))
            .transpose()
    };
    let cols = *shape.last().unwrap();
    Packed {
        spec,
        rows: shape.iter().product::<usize>() / cols,
        cols,
        w: bytes(&m.tensors[name])?,
        scales: part("scales")?.unwrap(),
        biases: part("biases")?,
    }
    .decode()
    .map(Some)
    .map_err(|e| format!("{name}: {e}"))
}

/// mlx-vlm's name for HF tensor `name` (`model.language_model.*` -> `language_model.model.*`,
/// `lm_head.*` -> `language_model.lm_head.*`, `model.visual.*` -> `vision_tower.*`, other
/// `model.*` -> top level), tried when `name` itself is absent.
pub(crate) fn vlm_name(name: &str) -> Option<String> {
    if let Some(r) = name.strip_prefix("model.language_model.") {
        Some(format!("language_model.model.{r}"))
    } else if let Some(r) = name.strip_prefix("lm_head.") {
        Some(format!("language_model.lm_head.{r}"))
    } else if let Some(r) = name.strip_prefix("model.visual.") {
        Some(format!("vision_tower.{r}"))
    } else {
        name.strip_prefix("model.").map(String::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// [`Packed::decode`], checked against the blockwise bf16 path.
    fn decode(
        spec: Spec,
        rows: usize,
        cols: usize,
        w: &[u8],
        scales: (&[u8], &str),
        biases: Option<(&[u8], &str)>,
    ) -> Result<Vec<f32>, String> {
        let p = Packed {
            spec,
            rows,
            cols,
            w,
            scales,
            biases,
        };
        let out = p.decode()?;
        let bf: Vec<u16> = out.iter().map(|&v| f32_to_bf16(v)).collect();
        assert_eq!(p.decode_bf16()?, bf);
        Ok(out)
    }

    /// mlx `quantize` packing (backend/cpu/quantized.cpp): `el_per_int` codes OR-ed at `k * bits`,
    /// stored as one uint32 (power-of-2 bits) or `bytes_per_pack` little-endian bytes (3/5/6 bits).
    fn mlx_pack(q: &[u32], bits: usize) -> Vec<u8> {
        let (el_per_int, bytes_per_pack) = match bits {
            3 => (8, 3),
            5 => (8, 5),
            6 => (4, 3),
            b => (32 / b, 4),
        };
        q.chunks(el_per_int)
            .flat_map(|c| {
                let el = c
                    .iter()
                    .enumerate()
                    .fold(0u64, |a, (k, &v)| a | (v as u64) << (k * bits));
                el.to_le_bytes()[..bytes_per_pack].to_vec()
            })
            .collect()
    }

    #[test]
    fn affine_unpacks_every_mlx_bit_width() {
        for bits in [2, 3, 4, 5, 6, 8] {
            let (rows, cols, group) = (3, 128, 32);
            let q: Vec<u32> = (0..rows * cols)
                .map(|i| (i * 7 + i / 5) as u32 % (1 << bits))
                .collect();
            let w = mlx_pack(&q, bits);
            assert_eq!(w.len(), rows * cols * bits / 8, "{bits}-bit");
            let groups = rows * cols / group;
            let scales: Vec<f32> = (0..groups).map(|g| 0.5 + g as f32).collect();
            let biases: Vec<f32> = (0..groups).map(|g| -(g as f32)).collect();
            let spec = Spec {
                mode: Mode::Affine,
                bits,
                group,
            };
            let out = decode(
                spec,
                rows,
                cols,
                &w,
                (&f32s(&scales), "F32"),
                Some((&f32s(&biases), "F32")),
            )
            .unwrap();
            for (i, (&o, &qv)) in out.iter().zip(&q).enumerate() {
                let g = i / group;
                assert_eq!(
                    o,
                    qv as f32 * scales[g] + biases[g],
                    "{bits}-bit element {i}"
                );
            }
        }
    }

    #[test]
    fn three_bit_matches_mlx_extract_bits() {
        // mlx `extract_bits<3>`: w0 = b0 & 7, w2 = (b0 >> 6) + ((b1 & 1) << 2), w7 = b2 >> 5.
        let w = [0b1100_0101u8, 0b0000_0001, 0b1110_0000];
        let one = (&f32s(&[1.0])[..], "F32");
        let zero = f32s(&[0.0]);
        let spec = |bits| Spec {
            mode: Mode::Affine,
            bits,
            group: 32,
        };
        // Pad to one 32-element group (12 bytes).
        let mut buf = w.to_vec();
        buf.resize(12, 0);
        let out = decode(spec(3), 1, 32, &buf, one, Some((&zero, "F32"))).unwrap();
        assert_eq!(&out[..8], &[5.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    }

    #[test]
    fn fp_modes_decode_like_mlx() {
        // mxfp4: low nibble first, e8m0 scale 2^(s-127); 0x92 = +1.0 (lo 2), -0.5 (hi 9).
        let mut w = vec![0x92u8, 0x0F];
        w.resize(16, 0);
        let s = Spec {
            mode: Mode::Mxfp4,
            bits: 4,
            group: 32,
        };
        let out = decode(s, 1, 32, &w, (&[128u8], "U8"), None).unwrap();
        assert_eq!(&out[..4], &[2.0, -1.0, -12.0, 0.0]);
        // nvfp4: e4m3 scale 0x38 = 1.0, group 16.
        let s = Spec {
            mode: Mode::Nvfp4,
            bits: 4,
            group: 16,
        };
        let out = decode(s, 1, 16, &w[..8], (&[0x38u8], "U8"), None).unwrap();
        assert_eq!(&out[..4], &[1.0, -0.5, -6.0, 0.0]);
        // mxfp8: one e4m3 byte per element.
        let mut w8 = vec![0x38u8, 0xB8];
        w8.resize(32, 0);
        let s = Spec {
            mode: Mode::Mxfp8,
            bits: 8,
            group: 32,
        };
        let out = decode(s, 1, 32, &w8, (&[126u8], "U8"), None).unwrap();
        assert_eq!(&out[..2], &[0.5, -0.5]);
    }

    #[test]
    fn config_overrides_and_refusals() {
        let q = serde_json::json!({
            "group_size": 64, "bits": 4, "mode": "affine",
            "model.embed_tokens": {"group_size": 32, "bits": 6},
            "model.layers.0.mlp.down_proj": false
        });
        assert_eq!(blocker(&q), None);
        let t = |name: &str, dtype: &str, shape: Vec<usize>, bytes: usize| VirtualTensor {
            name: name.into(),
            shape,
            quant_format: crate::hardware::QuantFormat::FP32,
            byte_offset: 0,
            byte_len: bytes,
            dtype: dtype.into(),
            shard_index: 0,
        };
        // 6-bit / group 32 over 256 inputs: 48 words, 8 groups per row.
        let w = t(
            "model.embed_tokens.weight",
            "U32",
            vec![10, 48],
            10 * 48 * 4,
        );
        let s = t("model.embed_tokens.scales", "BF16", vec![10, 8], 160);
        let get = |n: &str| [&w, &s].into_iter().find(|t| t.name == n);
        let (spec, shape) = packed(&q, "model.embed_tokens.weight", get)
            .unwrap()
            .unwrap();
        assert_eq!((spec.bits, spec.group, shape), (6, 32, vec![10, 256]));
        // Unlisted modules with U8 scales: the one fp mode the shapes fit (mlx-lm inference).
        let w = t("a.weight", "U32", vec![2, 32], 256);
        let s = t("a.scales", "U8", vec![2, 16], 32);
        let get = |n: &str| [&w, &s].into_iter().find(|t| t.name == n);
        assert_eq!(
            packed(&q, "a.weight", get).unwrap().unwrap().0.mode,
            Mode::Nvfp4
        );
        assert!(blocker(&serde_json::json!({"bits": 4, "mode": "int4"}))
            .unwrap()
            .contains("mode 'int4' is not supported"));
        assert!(
            blocker(&serde_json::json!({"bits": 8, "group_size": 32, "mode": "mxfp4"}))
                .unwrap()
                .contains("mxfp4 needs 4 bits")
        );
        assert!(blocker(&serde_json::json!({"bits": 4, "x": {"bits": 7}})).is_some());
    }
}
