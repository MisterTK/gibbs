//! FP8 checkpoints (ModelOpt, compressed-tensors `float-quantized`, vLLM `fp8`, Quark) repacked to
//! `FP8_E4M3_B32`: per 32 contiguous elements of a row, 32 E4M3FN bytes then one little-endian f32
//! scale, `value = e4m3(byte) * scale`. See [`super`] for the layout contract.
//!
//! E4M3FN: bias 7, no infinities, `S.1111.111` is NaN (the repack never emits it), max 448.
//! E5M2: bias 15, `S.11111.xx` is inf/NaN (a checkpoint containing either fails to repack).
//!
//! `scale` is the checkpoint's own per-tensor / per-channel / block (`weight_scale`,
//! `weight_scale_inv`) scale of that row segment, so an E4M3 source is copied byte for byte and is
//! bit-identical to [`crate::model::prequant::dequantize`]. An E5M2 source is re-expressed as
//! E4M3 x `scale * 2^k`, `2^k` chosen per block so the block max lands in [256, 448]. That is exact
//! (E5M2's 2 mantissa bits fit E4M3's 3, and scaling by a power of two is exact) for every element at
//! least 2^-6 of the way down E4M3's range, i.e. within 14 binades of the block max; smaller
//! elements round to E4M3 subnormals (multiples of 2^-9 · 2^k) or flush to zero, so the absolute
//! error per element is at most 2^-10 · 2^k <= block max · 2^-18, plus the two f32 product roundings
//! (2 · 2^-24 relative to the element, <= block max · 2^-23).
//!
//! The GPU flushes f32-subnormal products to zero while the CPU keeps them. The smallest nonzero
//! E4M3 magnitude is 2^-9, so a block whose scale is nonzero with `|s| < 2^-117` (or not finite)
//! is not emitted: `repack` returns `Ok(None)` and the weight takes the f32 path. Every emitted
//! block therefore decodes identically on CPU and GPU.

use super::Repacked;
use crate::graph::codec::Spec;
use crate::hardware::QuantFormat;
use crate::model::layers::GgufBlockFn;
use crate::model::prequant::{block, bytes};
use crate::model::{decode_dtype, f8_e4m3_to_f32, ModelManifest};

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = BLOCK + 4;

/// GPU codecs of this family (MSL `decode8`), merged into the registry by `graph::codec`.
pub const SPECS: &[Spec] = &[Spec {
    format: QuantFormat::FP8_E4M3_B32,
    block_elems: BLOCK as u32,
    block_bytes: BLOCK_BYTES as u32,
    tables: &[],
    body: r#"
#ifndef GIBBS_CD_E4M3
#define GIBBS_CD_E4M3
// E4M3FN by bit tricks: subnormals are m/512, normals are (8+m)·2^(e-10), 0x7F is NaN.
inline float cd_e4m3(uint x) {
    uint m = x & 0x7Fu;
    float v = (m < 8u) ? float(m) * (1.0f / 512.0f) : as_type<float>(((m >> 3) + 120u) << 23 | ((m & 7u) << 20));
    if (m == 0x7Fu) v = as_type<float>(0x7FC00000u);
    return (x & 0x80u) ? -v : v;
}
#endif
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float s = as_type<float>(cd_u32(blk + 32));
    for (uint i = 0; i < 8; i++) {
        out8[i] = cd_e4m3(blk[sub * 8 + i]) * s;
    }
}
"#,
}];

fn dequantize_block(b: &[u8], y: &mut [f32]) {
    let s = f32::from_le_bytes([b[32], b[33], b[34], b[35]]);
    for (y, &x) in y.iter_mut().zip(&b[..BLOCK]) {
        *y = f8_e4m3_to_f32(x) * s;
    }
}

pub fn block_codec(format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    (format == QuantFormat::FP8_E4M3_B32).then_some((BLOCK, BLOCK_BYTES, dequantize_block))
}

fn pow2(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

/// Nearest-even E4M3FN code of `x`, |x| <= 448 (so never the NaN code).
fn encode_e4m3(x: f32) -> u8 {
    let (sign, a) = (if x.is_sign_negative() { 0x80 } else { 0 }, x.abs());
    if a < pow2(-6) {
        return sign | (a * 512.0).round_ties_even() as u8; // subnormals; 8 is the smallest normal
    }
    let b = a.to_bits();
    let (m, rem) = ((b >> 20) & 7, b & 0xFFFFF);
    let up = rem > 0x80000 || (rem == 0x80000 && m & 1 == 1);
    // A mantissa carry propagates into the exponent field.
    sign | ((((b >> 23) - 120) << 3 | m) + up as u32) as u8
}

/// One block of E5M2 values as E4M3 bytes plus the f32 scale to apply (`scale * 2^k`).
fn e5m2_block(v: &[f32], scale: f32) -> Result<([u8; BLOCK], f32), String> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err("E5M2 inf/NaN".into());
    }
    let max = v.iter().fold(0f32, |a, x| a.max(x.abs()));
    let k = if max == 0.0 { 0 } else { (max.to_bits() >> 23) as i32 - 127 - 8 };
    let inv = pow2(-k);
    Ok((std::array::from_fn(|i| encode_e4m3(v[i] * inv)), scale * pow2(k)))
}

pub fn repack(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Repacked>, String> {
    let Some(stem) = weight.strip_suffix(".weight") else {
        return Ok(None);
    };
    let get = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    let (Some(w), Some(st)) = (
        m.tensors.get(weight),
        get("weight_scale").or_else(|| get("weight_scale_inv")),
    ) else {
        return Ok(None);
    };
    let &[rows, cols] = &w.shape[..] else {
        return Ok(None);
    };
    let e5m2 = w.dtype == "F8_E5M2";
    // Zero points, activation order and global scales are never stored beside fp8 weights.
    let other = ["weight_global_scale", "weight_scale_2", "weight_zero_point", "weight_g_idx"];
    if !(e5m2 || w.dtype == "F8_E4M3") || cols % BLOCK != 0 || other.iter().any(|s| get(s).is_some()) {
        return Ok(None);
    }
    let err = |e: String| format!("{weight}: {e}");
    let (sr, sc) = match st.shape[..] {
        [] => (1, 1),
        [n] => (n, 1),
        [a, b] => (a, b),
        _ => return Ok(None),
    };
    let scale = decode_dtype(bytes(data, st)?, &st.dtype).map_err(err)?;
    if scale.len() != sr * sc {
        return Err(err(format!("{} scales for grid [{sr}, {sc}]", scale.len())));
    }
    // Same grid inference as `prequant::dequantize`; a 32-block must not straddle two scale groups.
    let (Ok(rb), Ok(cb)) = (block(rows, sr), block(cols, sc)) else {
        return Ok(None);
    };
    if cb % BLOCK != 0 {
        return Ok(None);
    }
    let raw = bytes(data, w)?;
    if raw.len() != rows * cols {
        return Err(err(format!("{} bytes for fp8 [{rows}, {cols}]", raw.len())));
    }
    let vals = if e5m2 { decode_dtype(raw, &w.dtype).map_err(err)? } else { Vec::new() };
    let mut out = Vec::with_capacity(rows * cols / BLOCK * BLOCK_BYTES);
    for r in 0..rows {
        for c in (0..cols).step_by(BLOCK) {
            let s = scale[(r / rb) * sc + c / cb];
            if !s.is_finite() {
                return Err(err(format!("non-finite scale {s}")));
            }
            let i = r * cols + c;
            let (q, s) = if e5m2 {
                e5m2_block(&vals[i..i + BLOCK], s).map_err(err)?
            } else {
                let q: [u8; BLOCK] = raw[i..i + BLOCK].try_into().unwrap();
                if q.iter().any(|b| b & 0x7F == 0x7F) {
                    return Err(err("E4M3 NaN".into()));
                }
                (q, s)
            };
            // GPU flushes subnormal products; see the module docs.
            if s != 0.0 && !(s.is_finite() && s.abs() >= pow2(-117)) {
                return Ok(None);
            }
            out.extend_from_slice(&q);
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    Ok(Some(Repacked { format: QuantFormat::FP8_E4M3_B32, rows, cols, bytes: out }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::dequantize_gguf_blocks;
    use crate::model::{prequant, VirtualTensor};

    /// Independent E4M3FN decoder: sign · (1 + m/8) · 2^(e-7), subnormal m/8 · 2^-6 (`None` = NaN).
    fn e4m3_ref(x: u8) -> Option<f64> {
        let (e, m) = (((x >> 3) & 15) as i32, (x & 7) as f64);
        let mag = match (e, x & 7) {
            (15, 7) => return None,
            (0, _) => m / 8.0 * 2f64.powi(-6),
            _ => (1.0 + m / 8.0) * 2f64.powi(e - 7),
        };
        Some(if x & 0x80 != 0 { -mag } else { mag })
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    fn fixture(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> (ModelManifest, Vec<u8>) {
        let (mut m, mut data) = (ModelManifest::default(), Vec::new());
        for (name, dtype, shape, b) in tensors {
            m.tensors.insert(
                name.to_string(),
                VirtualTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    quant_format: QuantFormat::FP32,
                    byte_offset: data.len(),
                    byte_len: b.len(),
                    dtype: dtype.to_string(),
                    shard_index: 0,
                },
            );
            data.extend_from_slice(b);
        }
        (m, data)
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Random non-NaN E4M3 bytes, or E5M2 bytes with exponent field in `exp` (default: all finite).
    fn weights(dtype: &str, n: usize, exp: std::ops::RangeInclusive<u32>, r: &mut Lcg) -> Vec<u8> {
        (0..n)
            .map(|_| loop {
                let b = r.next() as u8;
                let ok = if dtype == "F8_E4M3" {
                    b & 0x7F != 0x7F
                } else {
                    exp.contains(&((b as u32 >> 2) & 31))
                };
                if ok {
                    return b;
                }
            })
            .collect()
    }

    /// Scale tensors of the three layouts for a `[rows, cols]` weight.
    fn scale_layouts(rows: usize, cols: usize, r: &mut Lcg) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let mut s = |n: usize| -> Vec<f32> { (0..n).map(|_| 0.001 + (r.next() % 1000) as f32 / 7.0).collect() };
        let (br, bc) = (rows.div_ceil(128), cols.div_ceil(128));
        vec![
            ("weight_scale", vec![], s(1)),
            ("weight_scale", vec![rows, 1], s(rows)),
            ("weight_scale_inv", vec![br, bc], s(br * bc)),
        ]
    }

    #[test]
    fn all_e4m3_bytes_decode_like_the_spec() {
        let (n, bytes, f) = block_codec(QuantFormat::FP8_E4M3_B32).unwrap();
        assert_eq!((n, bytes), (32, 36));
        for base in (0..256).step_by(32) {
            let mut blk: Vec<u8> = (base..base + 32).map(|b| b as u8).collect();
            blk.extend_from_slice(&1.0f32.to_le_bytes());
            let mut y = [0f32; 32];
            f(&blk, &mut y);
            for (i, v) in y.iter().enumerate() {
                match e4m3_ref((base + i) as u8) {
                    None => assert!(v.is_nan(), "{:#x}", base + i),
                    Some(want) => assert_eq!(*v as f64, want, "{:#x}", base + i),
                }
            }
        }
        // Largest finite 448 and the nearest-even encoder agree with the decoder on every code.
        for b in (0..=255u8).filter(|b| b & 0x7F != 0x7F) {
            assert_eq!(encode_e4m3(f8_e4m3_to_f32(b)), b);
        }
    }

    #[test]
    fn e4m3_repack_is_exact() {
        let mut r = Lcg(1);
        for (rows, cols) in [(4, 64), (200, 256)] {
            for (name, shape, scale) in scale_layouts(rows, cols, &mut r) {
                let w = weights("F8_E4M3", rows * cols, 0..=31, &mut r);
                let (m, data) = fixture(&[
                    ("a.weight", "F8_E4M3", vec![rows, cols], w),
                    (&format!("a.{name}"), "F32", shape.clone(), f32_bytes(&scale)),
                ]);
                let rp = repack(&m, &data, "a.weight").unwrap().unwrap();
                assert_eq!((rp.rows, rp.cols, rp.bytes.len()), (rows, cols, rows * cols / 32 * 36));
                let got = dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap();
                let want = prequant::dequantize(&m, &data, "a.weight").unwrap().unwrap();
                assert!(got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()), "{name} {shape:?}");
            }
        }
    }

    /// Max |repacked - f32 path| over every element, as a fraction of the block max.
    fn e5m2_error(exp: std::ops::RangeInclusive<u32>, seed: u64) -> f64 {
        let mut r = Lcg(seed);
        let mut worst = 0f64;
        for (rows, cols) in [(4, 64), (200, 256)] {
            for (name, shape, scale) in scale_layouts(rows, cols, &mut r) {
                let w = weights("F8_E5M2", rows * cols, exp.clone(), &mut r);
                let (m, data) = fixture(&[
                    ("a.weight", "F8_E5M2", vec![rows, cols], w),
                    (&format!("a.{name}"), "F32", shape.clone(), f32_bytes(&scale)),
                ]);
                let rp = repack(&m, &data, "a.weight").unwrap().unwrap();
                let got = dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap();
                let want = prequant::dequantize(&m, &data, "a.weight").unwrap().unwrap();
                for (g, w) in got.chunks(32).zip(want.chunks(32)) {
                    let max = w.iter().fold(0f64, |a, x| a.max(x.abs() as f64));
                    for (g, w) in g.iter().zip(w) {
                        worst = worst.max((*g as f64 - *w as f64).abs() / max.max(f64::MIN_POSITIVE));
                    }
                }
            }
        }
        worst
    }

    #[test]
    fn e5m2_repack_is_exact_within_14_binades_and_bounded_beyond() {
        // Exponent fields 10..=23: every block spans <= 14 binades below its max (+ zero/subnormal codes
        // excluded), so every element is exactly representable.
        assert_eq!(e5m2_error(10..=23, 7), 0.0);
        // Full range incl. subnormals, tiny elements flush: error stays under block max · 2^-18.
        let worst = e5m2_error(0..=30, 8);
        eprintln!("E5M2 -> E4M3 worst |err| / block max = {worst:e} (bound {:e})", 2f64.powi(-18));
        assert!(worst <= 2f64.powi(-18) + 2f64.powi(-23), "{worst}");
        assert!(worst > 0.0, "full-range data should exercise the flush path");
    }

    #[test]
    fn refusals() {
        let q = |dtype: &str, cols, w: Vec<u8>, extra: &[(&str, &str, Vec<usize>, Vec<u8>)]| {
            let mut t = vec![
                ("a.weight", dtype, vec![2, cols], w),
                ("a.weight_scale", "F32", vec![], f32_bytes(&[0.5])),
            ];
            t.extend_from_slice(extra);
            let (m, data) = fixture(&t);
            repack(&m, &data, "a.weight")
        };
        // cols not a multiple of 32, unscaled, unsupported dtype -> f32 path.
        assert!(q("F8_E4M3", 16, vec![0x38; 32], &[]).unwrap().is_none());
        assert!(q("F16", 32, vec![0; 128], &[]).unwrap().is_none());
        // Scales whose smallest product would be f32-subnormal stay on the f32 path.
        for dtype in ["F8_E4M3", "F8_E5M2"] {
            let tiny = |s: f32| {
                let (m, data) = fixture(&[
                    ("a.weight", dtype, vec![1, 32], vec![0x38; 32]),
                    ("a.weight_scale", "F32", vec![], f32_bytes(&[s])),
                ]);
                repack(&m, &data, "a.weight").unwrap().is_none()
            };
            assert!(tiny(2f32.powi(-118)) && tiny(1e-40) && !tiny(2f32.powi(-100)) && !tiny(0.0));
        }
        let (m, data) = fixture(&[("a.weight", "F8_E4M3", vec![2, 32], vec![0x38; 64])]);
        assert!(repack(&m, &data, "a.weight").unwrap().is_none());
        // A scale grid that splits a 32-block (per-16-column scales) can't be one scale per block.
        let (m, data) = fixture(&[
            ("a.weight", "F8_E4M3", vec![2, 64], vec![0x38; 128]),
            ("a.weight_scale", "F32", vec![2, 4], f32_bytes(&[1.0; 8])),
        ]);
        assert!(repack(&m, &data, "a.weight").unwrap().is_none());
        // NaN / inf are errors, never emitted.
        let mut nan = vec![0x38; 64];
        nan[5] = 0xFF;
        assert!(q("F8_E4M3", 32, nan, &[]).err().unwrap().contains("NaN"));
        let mut inf = vec![0x3C; 64];
        inf[9] = 0x7C;
        assert!(q("F8_E5M2", 32, inf, &[]).err().unwrap().contains("inf"));
    }

    /// `weight_scale_inv` multiplies per 128x128 block, ragged edges included (transformers
    /// finegrained FP8: `w = w_fp8.float() * scale_inv` broadcast over each block).
    #[test]
    fn block_scale_inv_dequantizes_per_block() {
        let (rows, cols) = (130, 129);
        let scale = [2.0f32, 3.0, 5.0, 7.0];
        let (m, data) = fixture(&[
            ("a.weight", "F8_E4M3", vec![rows, cols], vec![0x38; rows * cols]), // 1.0
            ("a.weight_scale_inv", "F32", vec![2, 2], f32_bytes(&scale)),
        ]);
        let got = prequant::dequantize(&m, &data, "a.weight").unwrap().unwrap();
        for i in 0..rows * cols {
            let (r, c) = (i / cols, i % cols);
            assert_eq!(got[i], scale[(r / 128) * 2 + c / 128], "[{r},{c}]");
        }
    }
}
