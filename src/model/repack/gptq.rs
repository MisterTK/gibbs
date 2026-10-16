//! AutoGPTQ / GPTQModel and AutoAWQ (GEMM) 4-bit checkpoints: `X.qweight`, `X.qzeros`, `X.scales` (and
//! GPTQ's `X.g_idx`) beside the layer's bias. Unpacked to the same `(nibble - zero) * scale` the
//! compressed-tensors int4 path uses, then repacked to its `INT4_G{32,64,128}` blocks (a zero point
//! is an integer, so it is exact in f16; MLX's `bias = -zero * scale` generally is not).
//!
//! * GPTQ: `qweight` `[in/8, out]` I32, eight nibbles along `in` (nibble k = row 8r+k); `qzeros`
//!   `[groups, out/8]`, eight along `out` (nibble k = column 8c+k) and stored as `zero - 1` unless the
//!   config's `checkpoint_format` is `gptq_v2`; `scales` `[groups, out]`; `g_idx` `[in]` names each
//!   input's group (absent = `i / group`).
//! * AWQ: `qweight` `[in, out/8]`, eight nibbles along `out` in the order [0, 2, 4, 6, 1, 3, 5, 7]
//!   (nibble k = column 8c + order[k]); `qzeros` packed the same, never offset.
//!
//! Which of the two a weight is follows from the shapes (`qweight` columns equal the scale columns
//! for GPTQ, an eighth of them for AWQ); only the zero-point convention needs the config
//! ([`annotate`]).

use super::{affine, Repacked};
use crate::hardware::QuantFormat;
use crate::model::prequant::bytes;
use crate::model::{decode_dtype, ModelManifest, VirtualTensor};
use serde_json::Value;
use std::path::Path;

/// AutoAWQ's `order_map`: nibble k of a packed word holds column `ORDER[k]` of its 8.
#[cfg(test)]
const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
/// Manifest metadata key [`annotate`] writes from `quantization_config.checkpoint_format`.
const FORMAT_KEY: &str = "gptq.checkpoint_format";

/// Why a `gptq` / `awq` `quantization_config` can't load. Per-tensor group sizes need no check: any
/// grid decodes, and only 32 / 64 / 128 repack.
pub fn blocker(method: &str, q: &Value) -> Option<String> {
    let bits = q["bits"].as_u64().unwrap_or(4);
    if bits != 4 {
        return Some(format!(
            "{method} {bits}-bit checkpoints are not supported (4-bit are)"
        ));
    }
    if method == "gptq" {
        if let Some(f) = q["checkpoint_format"]
            .as_str()
            .filter(|f| !matches!(*f, "gptq" | "gptq_v2"))
        {
            return Some(format!(
                "gptq checkpoint_format '{f}' is not supported (gptq, gptq_v2 are)"
            ));
        }
        if q["dynamic"].as_object().is_some_and(|d| !d.is_empty()) {
            return Some("gptq `dynamic` per-module overrides are not supported".into());
        }
    } else {
        if let Some(v) = q["version"]
            .as_str()
            .filter(|v| !v.eq_ignore_ascii_case("gemm"))
        {
            return Some(format!("awq version '{v}' is not supported (gemm is)"));
        }
        if q["zero_point"] == false {
            return Some("awq without zero points (`zero_point: false`) is not supported".into());
        }
    }
    None
}

/// Records the GPTQ zero-point convention from `config.json` (see [`FORMAT_KEY`]).
pub fn annotate(m: &mut ModelManifest, config_path: Option<&Path>) {
    let cfg = config_path
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let qc = cfg
        .as_ref()
        .map_or(&Value::Null, |c| &c["quantization_config"]);
    if let (true, Some(f)) = (
        qc["quant_method"] == "gptq",
        qc["checkpoint_format"].as_str(),
    ) {
        m.metadata.insert(FORMAT_KEY.into(), f.into());
    }
}

/// `(is_awq, out, in, groups)` of the int4 weight `{stem}.qweight`, from its shape and the scale grid.
fn layout(m: &ModelManifest, stem: &str) -> Option<(bool, usize, usize, usize)> {
    let get = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    let (qw, sc) = (get("qweight")?, get("scales")?);
    let (&[a, b], &[groups, out]) = (&qw.shape[..], &sc.shape[..]) else {
        return None;
    };
    let (awq, inn) = if b == out {
        (false, a * 8)
    } else if b * 8 == out {
        (true, a)
    } else {
        return None;
    };
    (qw.dtype == "I32" && groups > 0 && inn % groups == 0).then_some((awq, out, inn, groups))
}

/// Gives every GPTQ / AWQ `X.qweight` its logical `[out, in]` `X.weight` (same bytes) tagged
/// `GPTQ_W4A16` / `AWQ_W4A16`. Run once, from `prequant::alias_packed`.
pub fn alias(m: &mut ModelManifest) {
    let mut add = Vec::new();
    for (name, t) in &m.tensors {
        let Some(stem) = name.strip_suffix(".qweight") else {
            continue;
        };
        if let Some((awq, out, inn, _)) = layout(m, stem) {
            add.push(VirtualTensor {
                name: format!("{stem}.weight"),
                shape: vec![out, inn],
                quant_format: if awq {
                    QuantFormat::AWQ_W4A16
                } else {
                    QuantFormat::GPTQ_W4A16
                },
                ..t.clone()
            });
        }
    }
    for t in add {
        m.tensors.entry(t.name.clone()).or_insert(t);
    }
}

/// Quantizer state of a GPTQ / AWQ weight (`X.qweight`, `.qzeros`, `.scales`, `.g_idx`), not model
/// weights.
pub fn is_companion(m: &ModelManifest, name: &str) -> bool {
    [".qweight", ".qzeros", ".scales", ".g_idx"]
        .iter()
        .any(|s| {
            name.strip_suffix(s)
                .is_some_and(|stem| layout(m, stem).is_some())
        })
}

struct Int4 {
    rows: usize,
    cols: usize,
    group: usize,
    /// One nibble per byte, row-major `[rows, cols]`.
    q: Vec<u8>,
    /// `[groups, rows]`.
    scale: Vec<f32>,
    zero: Vec<f32>,
    g_idx: Option<Vec<usize>>,
}

fn parse(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Int4>, String> {
    let Some(stem) = weight.strip_suffix(".weight") else {
        return Ok(None);
    };
    let Some((awq, rows, cols, groups)) = layout(m, stem) else {
        return Ok(None);
    };
    let err = |e: String| format!("{weight}: {e}");
    let get = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    let words = |name: &str, n: usize| -> Result<Option<Vec<u32>>, String> {
        let Some(t) = get(name) else {
            return Ok(None);
        };
        let raw = bytes(data, t)?;
        if raw.len() != n * 4 {
            return Err(err(format!(
                "{} bytes in {name}, expected {n} words",
                raw.len()
            )));
        }
        Ok(Some(
            raw.chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect(),
        ))
    };
    // Nibble position of column `o` within its 8-wide word.
    let pos = |o: usize| if awq { (o % 2) * 4 + o % 8 / 2 } else { o % 8 };
    let qw = words("qweight", rows * cols / 8)?.unwrap();
    let q = (0..rows * cols)
        .map(|k| {
            let (o, i) = (k / cols, k % cols);
            let w = if awq {
                qw[i * (rows / 8) + o / 8]
            } else {
                qw[i / 8 * rows + o]
            };
            let shift = 4 * if awq { pos(o) } else { i % 8 };
            ((w >> shift) & 0x0F) as u8
        })
        .collect();
    // GPTQ v1 (the AutoGPTQ default) stores zero - 1.
    let offset = if awq || m.metadata.get(FORMAT_KEY).is_some_and(|f| f == "gptq_v2") {
        0.0
    } else {
        1.0
    };
    let zero = match words("qzeros", groups * rows / 8)? {
        Some(z) => (0..groups * rows)
            .map(|k| {
                let (g, o) = (k / rows, k % rows);
                ((z[g * (rows / 8) + o / 8] >> (4 * pos(o))) & 0x0F) as f32 + offset
            })
            .collect(),
        // Symmetric export without stored zeros.
        None => vec![8.0; groups * rows],
    };
    let st = get("scales").unwrap();
    let scale = decode_dtype(bytes(data, st)?, &st.dtype).map_err(err)?;
    if scale.len() != groups * rows {
        return Err(err(format!(
            "{} scales for [{groups}, {rows}]",
            scale.len()
        )));
    }
    let g_idx = get("g_idx")
        .map(|t| decode_dtype(bytes(data, t)?, &t.dtype).map_err(err))
        .transpose()?
        .map(|g| g.into_iter().map(|v| v as usize).collect::<Vec<_>>());
    if g_idx
        .as_ref()
        .is_some_and(|g| g.len() != cols || g.iter().any(|&v| v >= groups))
    {
        return Err(err("g_idx does not match the scale grid".into()));
    }
    Ok(Some(Int4 {
        rows,
        cols,
        group: cols / groups,
        q,
        scale,
        zero,
        g_idx,
    }))
}

/// Expands a GPTQ / AWQ weight to row-major f32 `[out, in]` (`(nibble - zero) * scale`, grouped by
/// `g_idx` when present); `None` when `name` is not one.
pub fn dequantize(m: &ModelManifest, data: &[u8], name: &str) -> Result<Option<Vec<f32>>, String> {
    Ok(parse(m, data, name)?.map(|p| {
        (0..p.rows * p.cols)
            .map(|k| {
                let (o, i) = (k / p.cols, k % p.cols);
                let g = p.g_idx.as_ref().map_or(i / p.group, |g| g[i]);
                (p.q[k] as f32 - p.zero[g * p.rows + o]) * p.scale[g * p.rows + o]
            })
            .collect()
    }))
}

/// `INT4_G{32,64,128}` blocks of a GPTQ / AWQ weight. `Ok(None)` (the f32 path applies) for any other
/// group size, a `g_idx` that is not `i / group` (activation order), or a scale that is not exactly
/// an f16.
pub fn repack(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Repacked>, String> {
    let Some(p) = parse(m, data, weight)? else {
        return Ok(None);
    };
    let format = match p.group {
        32 => QuantFormat::INT4_G32,
        64 => QuantFormat::INT4_G64,
        128 => QuantFormat::INT4_G128,
        _ => return Ok(None),
    };
    if p.g_idx
        .as_ref()
        .is_some_and(|g| g.iter().enumerate().any(|(i, &v)| v != i / p.group))
    {
        return Ok(None);
    }
    // `[groups, rows]` -> one block per (row, group), row-major.
    let groups = p.cols / p.group;
    let by_row = |v: &[f32]| -> Vec<f32> {
        (0..p.rows * groups)
            .map(|k| v[k % groups * p.rows + k / groups])
            .collect()
    };
    let (Some(s), Some(z)) = (
        affine::exact_f16(&by_row(&p.scale)),
        affine::exact_f16(&by_row(&p.zero)),
    ) else {
        return Ok(None);
    };
    let nibbles: Vec<u8> = p.q.chunks_exact(2).map(|c| c[0] | c[1] << 4).collect();
    Ok(Some(Repacked {
        format,
        rows: p.rows,
        cols: p.cols,
        bytes: affine::blocks(&nibbles, p.group, &s, &z),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::dequantize_gguf_blocks;
    use crate::model::{f32_to_f16, prequant};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        /// Values with 8 mantissa bits over a few binades: exact in f16.
        fn scale(&mut self) -> f32 {
            (1.0 + (self.next() % 256) as f32 / 256.0) * 2f32.powi(self.next() as i32 % 6 - 8)
        }
    }

    /// A synthetic checkpoint's ground truth: nibbles `[out][in]`, scales and zeros `[groups][out]`.
    struct Truth {
        out: usize,
        inn: usize,
        group: usize,
        q: Vec<u8>,
        scale: Vec<f32>,
        zero: Vec<u8>,
        /// Group of each input column.
        g_of: Vec<usize>,
    }

    impl Truth {
        fn new(out: usize, inn: usize, group: usize, seed: u64) -> Self {
            let mut r = Lcg(seed);
            let groups = inn / group;
            Truth {
                out,
                inn,
                group,
                q: (0..out * inn).map(|_| (r.next() % 16) as u8).collect(),
                scale: (0..groups * out).map(|_| r.scale()).collect(),
                // 1..=15: GPTQ v1 stores `zero - 1`, which cannot hold 0.
                zero: (0..groups * out)
                    .map(|_| 1 + (r.next() % 15) as u8)
                    .collect(),
                g_of: (0..inn).map(|i| i / group).collect(),
            }
        }

        fn reference(&self) -> Vec<f32> {
            (0..self.out * self.inn)
                .map(|k| {
                    let (o, i) = (k / self.inn, k % self.inn);
                    let s = self.g_of[i] * self.out + o;
                    (self.q[k] as f32 - self.zero[s] as f32) * self.scale[s]
                })
                .collect()
        }
    }

    fn words(codes: &[u32]) -> Vec<u8> {
        codes.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Writes `x.{qweight,qzeros,scales[,g_idx]}` the way AutoGPTQ (`awq = false`) / AutoAWQ pack them.
    fn checkpoint(
        t: &Truth,
        awq: bool,
        v2: bool,
        scale_dtype: &str,
        g_idx: bool,
    ) -> (ModelManifest, Vec<u8>) {
        let (out, inn, groups) = (t.out, t.inn, t.inn / t.group);
        let pack8 =
            |at: &dyn Fn(usize) -> u32| -> u32 { (0..8).fold(0, |w, k| w | at(k) << (4 * k)) };
        let (qw, qw_shape) = if awq {
            // [in, out/8]: nibble k holds column ORDER[k].
            let w: Vec<u32> = (0..inn * out / 8)
                .map(|n| {
                    let (i, c) = (n / (out / 8), n % (out / 8));
                    pack8(&|k| t.q[(c * 8 + AWQ_ORDER[k]) * inn + i] as u32)
                })
                .collect();
            (w, [inn, out / 8])
        } else {
            // [in/8, out]: nibble k holds input row 8r + k.
            let w: Vec<u32> = (0..inn / 8 * out)
                .map(|n| {
                    let (r, o) = (n / out, n % out);
                    pack8(&|k| t.q[o * inn + r * 8 + k] as u32)
                })
                .collect();
            (w, [inn / 8, out])
        };
        let offset = if awq || v2 { 0 } else { 1 };
        let qz: Vec<u32> = (0..groups * out / 8)
            .map(|n| {
                let (g, c) = (n / (out / 8), n % (out / 8));
                pack8(&|k| {
                    let col = if awq { c * 8 + AWQ_ORDER[k] } else { c * 8 + k };
                    (t.zero[g * out + col] - offset) as u32
                })
            })
            .collect();
        let mut m = ModelManifest::default();
        let mut data = Vec::new();
        let mut add = |name: &str, dtype: &str, shape: &[usize], b: &[u8]| {
            m.tensors.insert(
                name.into(),
                VirtualTensor {
                    name: name.into(),
                    shape: shape.to_vec(),
                    quant_format: QuantFormat::FP32,
                    byte_offset: data.len(),
                    byte_len: b.len(),
                    dtype: dtype.into(),
                    shard_index: 0,
                },
            );
            data.extend_from_slice(b);
        };
        add("x.qweight", "I32", &qw_shape, &words(&qw));
        add("x.qzeros", "I32", &[groups, out / 8], &words(&qz));
        let sb: Vec<u8> = t
            .scale
            .iter()
            .flat_map(|&s| {
                if scale_dtype == "F16" {
                    f32_to_f16(s).to_le_bytes().to_vec()
                } else {
                    s.to_le_bytes().to_vec()
                }
            })
            .collect();
        add("x.scales", scale_dtype, &[groups, out], &sb);
        if g_idx {
            let gi: Vec<u8> = t
                .g_of
                .iter()
                .flat_map(|&g| (g as i32).to_le_bytes())
                .collect();
            add("x.g_idx", "I32", &[inn], &gi);
        }
        if v2 {
            m.metadata.insert(FORMAT_KEY.into(), "gptq_v2".into());
        }
        prequant::alias_packed(&mut m, &data).unwrap();
        (m, data)
    }

    fn bit_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x} vs {y}");
        }
    }

    #[test]
    fn gptq_and_awq_repack_to_int4_blocks_equal_to_the_reference() {
        for (group, fmt) in [
            (32, QuantFormat::INT4_G32),
            (64, QuantFormat::INT4_G64),
            (128, QuantFormat::INT4_G128),
        ] {
            for (awq, v2, g_idx) in [
                (false, false, false),
                (false, false, true),
                (false, true, true),
                (true, false, false),
            ] {
                let t = Truth::new(16, 256, group, group as u64);
                let (m, data) = checkpoint(&t, awq, v2, "F16", g_idx);
                let tag = format!("group {group} awq={awq} v2={v2} g_idx={g_idx}");
                assert_eq!(
                    m.tensors["x.weight"].quant_format,
                    if awq {
                        QuantFormat::AWQ_W4A16
                    } else {
                        QuantFormat::GPTQ_W4A16
                    }
                );
                assert_eq!(m.tensors["x.weight"].shape, [16, 256]);
                let rp = repack(&m, &data, "x.weight").unwrap().expect(&tag);
                assert_eq!(
                    (rp.format, rp.rows, rp.cols, rp.bytes.len()),
                    (fmt, 16, 256, 16 * 256 / group * (group / 2 + 4))
                );
                let want = t.reference();
                // Exact, so well inside the 1e-3 relative bound.
                bit_eq(
                    &dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap(),
                    &want,
                    &format!("{tag} blocks"),
                );
                bit_eq(
                    &prequant::dequantize(&m, &data, "x.weight")
                        .unwrap()
                        .unwrap(),
                    &want,
                    &format!("{tag} f32 path"),
                );
            }
        }
    }

    #[test]
    fn gptq_v1_zero_offset_is_applied_unless_the_config_says_v2() {
        let t = Truth::new(8, 128, 64, 7);
        let (mut m, data) = checkpoint(&t, false, false, "F16", false);
        m.metadata.insert(FORMAT_KEY.into(), "gptq_v2".into());
        let wrong = prequant::dequantize(&m, &data, "x.weight")
            .unwrap()
            .unwrap();
        assert_ne!(
            wrong,
            t.reference(),
            "a v1 file read as v2 is off by one zero point"
        );
        // `annotate` writes the same key from config.json.
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"quantization_config":{"quant_method":"gptq","checkpoint_format":"gptq_v2"}}"#,
        )
        .unwrap();
        let mut m = ModelManifest::default();
        annotate(&mut m, Some(&cfg));
        assert_eq!(m.metadata[FORMAT_KEY], "gptq_v2");
    }

    #[test]
    fn activation_order_group_sizes_and_inexact_scales_fall_back_to_the_f32_path() {
        // desc_act: columns visit groups in a shuffled order.
        let mut t = Truth::new(8, 128, 32, 3);
        let mut r = Lcg(5);
        for i in (1..t.inn).rev() {
            t.g_of.swap(i, r.next() as usize % (i + 1));
        }
        let (m, data) = checkpoint(&t, false, false, "F16", true);
        assert!(repack(&m, &data, "x.weight").unwrap().is_none());
        bit_eq(
            &prequant::dequantize(&m, &data, "x.weight")
                .unwrap()
                .unwrap(),
            &t.reference(),
            "g_idx",
        );
        // Group sizes outside 32 / 64 / 128, including one scale per row.
        for (inn, group) in [(64, 16), (256, 256)] {
            let t = Truth::new(8, inn, group, 4);
            for awq in [false, true] {
                let (m, data) = checkpoint(&t, awq, false, "F16", false);
                assert!(
                    repack(&m, &data, "x.weight").unwrap().is_none(),
                    "group {group}"
                );
                bit_eq(
                    &prequant::dequantize(&m, &data, "x.weight")
                        .unwrap()
                        .unwrap(),
                    &t.reference(),
                    "group",
                );
            }
        }
        // An f32 scale that no f16 holds.
        let mut t = Truth::new(8, 128, 64, 6);
        t.scale[3] = 0.1;
        let (m, data) = checkpoint(&t, false, false, "F32", false);
        assert!(repack(&m, &data, "x.weight").unwrap().is_none());
        bit_eq(
            &prequant::dequantize(&m, &data, "x.weight")
                .unwrap()
                .unwrap(),
            &t.reference(),
            "f32 scale",
        );
    }

    #[test]
    fn companions_and_blockers() {
        let t = Truth::new(8, 128, 64, 9);
        let (m, _) = checkpoint(&t, false, false, "F16", true);
        for n in ["x.qweight", "x.qzeros", "x.scales", "x.g_idx"] {
            assert!(is_companion(&m, n), "{n}");
        }
        assert!(!is_companion(&m, "x.weight") && !is_companion(&m, "y.scales"));
        let j = |s: &str| serde_json::from_str::<Value>(s).unwrap();
        assert_eq!(
            blocker("gptq", &j(r#"{"bits":4,"group_size":128,"desc_act":true}"#)),
            None
        );
        assert_eq!(
            blocker("gptq", &j(r#"{"checkpoint_format":"gptq_v2"}"#)),
            None
        );
        assert_eq!(
            blocker(
                "awq",
                &j(r#"{"bits":4,"version":"gemm","zero_point":true}"#)
            ),
            None
        );
        for (m, q, why) in [
            ("gptq", r#"{"bits":8}"#, "gptq 8-bit"),
            (
                "gptq",
                r#"{"checkpoint_format":"marlin"}"#,
                "checkpoint_format 'marlin'",
            ),
            ("gptq", r#"{"dynamic":{"+:.*":{"bits":8}}}"#, "`dynamic`"),
            ("awq", r#"{"version":"gemv"}"#, "awq version 'gemv'"),
            ("awq", r#"{"zero_point":false}"#, "without zero points"),
        ] {
            assert!(
                blocker(m, &j(q)).is_some_and(|b| b.contains(why)),
                "{m} {q}"
            );
        }
    }
}
