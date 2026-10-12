//! 4-bit affine group formats: MLX `MLX_AFFINE4_G{32,64,128}` (`nibble * scale + bias`) and
//! compressed-tensors `INT4_G{32,64,128}` (`(nibble - zero) * scale`).
//!
//! Block of group size `g`: `g/2` nibble bytes (element `2i` in the low nibble of byte `i`, the order
//! both checkpoints already store), then little-endian f16 scale and f16 bias / zero point.
//! Scales and biases are f16 on the GPU, so a tensor whose values do not all convert to f16 exactly
//! (bf16 MLX checkpoints routinely) is not repacked: [`repack`] returns `Ok(None)` and the exact f32
//! path applies. Never rounded. An unsupported MLX group size never reaches `repack`: `mlx::parse` makes it
//! a load-time `Err` shared with the f32 path.

use super::Repacked;
use crate::graph::codec::Spec;
use crate::hardware::QuantFormat;
use crate::model::layers::GgufBlockFn;
use crate::model::prequant::{self, mlx};
use crate::model::{f16_to_f32, f32_to_f16, ModelManifest};

/// `nibble * scale + bias` (MLX) or `(nibble - zero) * scale` (INT4): `$expr` over `n`, `s`, `z`.
macro_rules! spec {
    ($format:ident, $g:literal, $half:literal, $expr:literal) => {
        Spec {
            format: QuantFormat::$format,
            block_elems: $g,
            block_bytes: $g / 2 + 4,
            tables: &[],
            body: concat!(
                "\ninline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n",
                "    #pragma clang fp contract(off)\n",
                "    float s = cd_f16(blk + ", $half, "), z = cd_f16(blk + ", $half, " + 2);\n",
                "    device const uchar* q = blk + sub * 4;\n",
                "    for (uint i = 0; i < 4; i++) {\n",
                "        float n = float(q[i] & 0x0F);\n",
                "        out8[2 * i] = ", $expr, ";\n",
                "        n = float(q[i] >> 4);\n",
                "        out8[2 * i + 1] = ", $expr, ";\n",
                "    }\n}\n"
            ),
        }
    };
}

/// GPU codecs of this family (MSL `decode8`), merged into the registry by `graph::codec`.
pub const SPECS: &[Spec] = &[
    spec!(MLX_AFFINE4_G32, 32, "16", "n * s + z"),
    spec!(MLX_AFFINE4_G64, 64, "32", "n * s + z"),
    spec!(MLX_AFFINE4_G128, 128, "64", "n * s + z"),
    spec!(INT4_G32, 32, "16", "(n - z) * s"),
    spec!(INT4_G64, 64, "32", "(n - z) * s"),
    spec!(INT4_G128, 128, "64", "(n - z) * s"),
];

/// MLX block: `value = nibble * scale + bias`, separate multiply and add.
fn mlx_block<const G: usize>(b: &[u8], y: &mut [f32]) {
    let f16 = |i: usize| f16_to_f32(u16::from_le_bytes([b[G / 2 + i], b[G / 2 + i + 1]]));
    let (s, bias) = (f16(0), f16(2));
    for (i, &q) in b[..G / 2].iter().enumerate() {
        y[2 * i] = (q & 0x0F) as f32 * s + bias;
        y[2 * i + 1] = (q >> 4) as f32 * s + bias;
    }
}

/// INT4 block: `value = (nibble - zero) * scale`.
fn int4_block<const G: usize>(b: &[u8], y: &mut [f32]) {
    let f16 = |i: usize| f16_to_f32(u16::from_le_bytes([b[G / 2 + i], b[G / 2 + i + 1]]));
    let (s, z) = (f16(0), f16(2));
    for (i, &q) in b[..G / 2].iter().enumerate() {
        y[2 * i] = ((q & 0x0F) as f32 - z) * s;
        y[2 * i + 1] = ((q >> 4) as f32 - z) * s;
    }
}

pub fn block_codec(format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    use QuantFormat as Q;
    Some(match format {
        Q::MLX_AFFINE4_G32 => (32, 20, mlx_block::<32> as GgufBlockFn),
        Q::MLX_AFFINE4_G64 => (64, 36, mlx_block::<64>),
        Q::MLX_AFFINE4_G128 => (128, 68, mlx_block::<128>),
        Q::INT4_G32 => (32, 20, int4_block::<32>),
        Q::INT4_G64 => (64, 36, int4_block::<64>),
        Q::INT4_G128 => (128, 68, int4_block::<128>),
        _ => return None,
    })
}

/// Every value as f16 bits, `None` unless each converts back to the identical f32.
fn exact_f16(v: &[f32]) -> Option<Vec<u16>> {
    v.iter()
        .map(|&x| {
            let h = f32_to_f16(x);
            (x.is_finite() && f16_to_f32(h).to_bits() == x.to_bits()).then_some(h)
        })
        .collect()
}

/// Blocks of `w` (`rows * cols / 2` nibble bytes, so group `i` is the `i`-th run of `g / 2` bytes) with
/// the per-group f16 `scale` and `bias`.
fn blocks(w: &[u8], g: usize, scale: &[u16], bias: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(scale.len() * (g / 2 + 4));
    for ((q, s), z) in w.chunks_exact(g / 2).zip(scale).zip(bias) {
        out.extend_from_slice(q);
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&z.to_le_bytes());
    }
    out
}

/// `[rows, cols]` MLX affine 4-bit or compressed-tensors int4-group `weight` as one block per group.
/// `Ok(None)` (the f32 dequantize path applies) when:
/// * MLX: not affine mode or not 4 bits, no `biases`, or any scale / bias is not exactly an f16
///   (bf16 values that need more than 10 mantissa bits or leave the f16 range, non-finite);
/// * compressed-tensors: not an int4 `pack-quantized` I32 weight, per-channel / per-tensor scales
///   (a scale grid other than `[out, in / group]` with `group` of 32, 64 or 128), activation order
///   (`weight_g_idx`), a global scale (`weight_global_scale` / `weight_scale_2`), fractional zero points, or any scale / zero point not exactly an f16.
pub fn repack(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Repacked>, String> {
    use QuantFormat as Q;
    if let Some(p) = mlx::packed_tensor(m, data, weight)? {
        let (g, groups) = (p.spec.group, p.cols / p.spec.group);
        let format = match g {
            32 => Q::MLX_AFFINE4_G32,
            64 => Q::MLX_AFFINE4_G64,
            128 => Q::MLX_AFFINE4_G128,
            _ => return Ok(None),
        };
        let (sb, sd) = p.scales;
        let Some((bb, bd)) = p.biases else {
            return Ok(None);
        };
        if p.spec.mode != mlx::Mode::Affine || p.spec.bits != 4 || p.w.len() != p.rows * p.cols / 2 {
            return Ok(None);
        }
        let dec = |b, d| crate::model::decode_dtype(b, d).map(|v| exact_f16(&v));
        let (Some(s), Some(z)) = (dec(sb, sd)?, dec(bb, bd)?) else {
            return Ok(None);
        };
        if s.len() != p.rows * groups || z.len() != s.len() {
            return Ok(None);
        }
        return Ok(Some(Repacked { format, rows: p.rows, cols: p.cols, bytes: blocks(p.w, g, &s, &z) }));
    }
    let Some(i) = prequant::int4_groups(m, data, weight)? else {
        return Ok(None);
    };
    let format = match i.group {
        32 => Q::INT4_G32,
        64 => Q::INT4_G64,
        _ => Q::INT4_G128,
    };
    let (Some(s), Some(z)) = (exact_f16(&i.scale), exact_f16(&i.zero)) else {
        return Ok(None);
    };
    Ok(Some(Repacked { format, rows: i.rows, cols: i.cols, bytes: blocks(i.w, i.group, &s, &z) }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::QuantFormat as Q;
    use crate::model::layers::dequantize_gguf_blocks;
    use crate::model::{f32_to_bf16, VirtualTensor};

    struct Fx {
        m: ModelManifest,
        data: Vec<u8>,
    }

    impl Fx {
        fn new() -> Self {
            Fx { m: ModelManifest::default(), data: Vec::new() }
        }
        fn add(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) {
            let t = VirtualTensor {
                name: name.into(),
                shape: shape.to_vec(),
                quant_format: Q::FP32,
                byte_offset: self.data.len(),
                byte_len: bytes.len(),
                dtype: dtype.into(),
                shard_index: 0,
            };
            self.data.extend_from_slice(bytes);
            self.m.tensors.insert(name.into(), t);
        }
        /// Float tensor of `dtype` F16 / BF16 / F32 from f32 values (must be exact in `dtype`).
        fn add_f(&mut self, name: &str, dtype: &str, shape: &[usize], v: &[f32]) {
            let b: Vec<u8> = v
                .iter()
                .flat_map(|&x| match dtype {
                    "F16" => f32_to_f16(x).to_le_bytes().to_vec(),
                    "BF16" => f32_to_bf16(x).to_le_bytes().to_vec(),
                    _ => x.to_le_bytes().to_vec(),
                })
                .collect();
            self.add(name, dtype, shape, &b);
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn nibbles(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| (self.next() % 16) as u8).collect()
        }
        /// Values with 7 mantissa bits (exact in bf16 and f16) over a few binades, either sign.
        fn exact(&mut self) -> f32 {
            let m = 1.0 + (self.next() % 128) as f32 / 128.0;
            let e = 2f32.powi(self.next() as i32 % 8 - 6);
            if self.next() % 2 == 0 { m * e } else { -m * e }
        }
        fn vals(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.exact()).collect()
        }
    }

    /// 8 little-endian-ordered `bits`-wide unsigned codes per word, bytes of the u32/i32 words.
    fn pack_words(codes: &[u8]) -> Vec<u8> {
        codes
            .chunks(8)
            .flat_map(|c| c.iter().enumerate().fold(0u32, |a, (k, &v)| a | (v as u32) << (4 * k)).to_le_bytes())
            .collect()
    }

    const BLOCK: [(usize, Q, Q); 3] =
        [(32, Q::MLX_AFFINE4_G32, Q::INT4_G32), (64, Q::MLX_AFFINE4_G64, Q::INT4_G64), (128, Q::MLX_AFFINE4_G128, Q::INT4_G128)];

    fn bit_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x} vs {y}");
        }
    }

    /// An MLX checkpoint of one `[rows, cols]` 4-bit weight; returns it with its codes, scales, biases.
    fn mlx_fx(g: usize, bits: usize, dtype: &str, seed: u64) -> (Fx, Vec<u8>, Vec<f32>, Vec<f32>) {
        let (rows, cols) = (3, 256);
        let mut r = Lcg(seed);
        let q = r.nibbles(rows * cols);
        let (s, b) = (r.vals(rows * cols / g), r.vals(rows * cols / g));
        let mut fx = Fx::new();
        fx.m.metadata.insert(mlx::META.into(), format!(r#"{{"group_size":{g},"bits":{bits}}}"#));
        fx.add("a.weight", "U32", &[rows, cols], &pack_words(&q));
        fx.add_f("a.scales", dtype, &[rows, cols / g], &s);
        fx.add_f("a.biases", dtype, &[rows, cols / g], &b);
        (fx, q, s, b)
    }

    #[test]
    fn mlx_repack_equals_f32_path_and_reference() {
        for (g, mlx_f, _) in BLOCK {
            for dtype in ["F16", "BF16", "F32"] {
                let (fx, q, s, b) = mlx_fx(g, 4, dtype, g as u64);
                let rp = repack(&fx.m, &fx.data, "a.weight").unwrap().expect("repacks");
                assert_eq!((rp.format, rp.rows, rp.cols, rp.bytes.len()), (mlx_f, 3, 256, 3 * 256 / g * (g / 2 + 4)));
                let got = dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap();
                let f32_path = mlx::dequantize(&fx.m, &fx.data, "a.weight").unwrap().unwrap();
                let want: Vec<f32> = q.iter().enumerate().map(|(i, &n)| n as f32 * s[i / g] + b[i / g]).collect();
                bit_eq(&got, &f32_path, &format!("mlx g{g} {dtype} vs f32 path"));
                bit_eq(&got, &want, &format!("mlx g{g} {dtype} vs reference"));
            }
        }
    }

    #[test]
    fn mlx_repack_declines() {
        // bf16 values that do not fit f16: underflow (below 2^-24) and overflow (above 65504).
        for bad in [2f32.powi(-30), 1.0e30, 1.0 + 2f32.powi(-11)] {
            let (mut fx, ..) = mlx_fx(64, 4, "F32", 1);
            let t = fx.m.tensors["a.scales"].clone();
            let mut v = vec![1.0f32; t.byte_len / 4];
            v[2] = bad;
            let at = t.byte_offset;
            fx.data[at..at + t.byte_len].copy_from_slice(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
            assert!(repack(&fx.m, &fx.data, "a.weight").unwrap().is_none(), "scale {bad:e}");
            // The f32 path still decodes it.
            assert!(mlx::dequantize(&fx.m, &fx.data, "a.weight").unwrap().is_some());
        }
        // bf16 checkpoint whose bias is not an f16.
        let (mut fx, ..) = mlx_fx(32, 4, "BF16", 2);
        let t = fx.m.tensors["a.biases"].clone();
        fx.data[t.byte_offset..t.byte_offset + 2].copy_from_slice(&f32_to_bf16(1.0e30).to_le_bytes());
        assert!(repack(&fx.m, &fx.data, "a.weight").unwrap().is_none());
        // Not 4 bits: 8-bit words (cols/4 per row) with matching scales.
        let (rows, cols, g) = (2, 128, 64);
        let mut fx = Fx::new();
        fx.m.metadata.insert(mlx::META.into(), r#"{"group_size":64,"bits":8}"#.into());
        fx.add("a.weight", "U32", &[rows, cols], &vec![0x11u8; rows * cols]);
        fx.add_f("a.scales", "F16", &[rows, cols / g], &[1.0; 4]);
        fx.add_f("a.biases", "F16", &[rows, cols / g], &[0.0; 4]);
        assert!(repack(&fx.m, &fx.data, "a.weight").unwrap().is_none());
        // fp mode (nvfp4: 16-element groups, U8 scales).
        let mut fx = Fx::new();
        fx.m.metadata.insert(mlx::META.into(), r#"{"mode":"nvfp4"}"#.into());
        fx.add("a.weight", "U32", &[2, 32], &[0u8; 2 * 32 * 4]);
        fx.add("a.scales", "U8", &[2, 16], &[0x38u8; 32]);
        assert!(repack(&fx.m, &fx.data, "a.weight").unwrap().is_none());
        // Affine without biases, a name that is not an MLX weight, and a manifest that is not MLX.
        let (mut fx, ..) = mlx_fx(32, 4, "F16", 3);
        assert!(repack(&fx.m, &fx.data, "b.weight").unwrap().is_none());
        fx.m.tensors.remove("a.biases");
        assert!(repack(&fx.m, &fx.data, "a.weight").unwrap().is_none());
        fx.m.metadata.clear();
        assert!(repack(&fx.m, &fx.data, "a.weight").unwrap().is_none());
    }

    /// A compressed-tensors `pack-quantized` int4 checkpoint of `x.weight_packed` `[rows, cols]` with
    /// `[rows, cols / g]` scales; `zero` = unsigned asymmetric zero points (packed along `out`). Returns
    /// the manifest (`alias_packed` run), the codes and the signed zero points' unsigned form.
    fn ct_fx(rows: usize, cols: usize, g: usize, dtype: &str, zero: bool, seed: u64) -> (Fx, Vec<u8>, Vec<f32>, Vec<u8>) {
        let mut r = Lcg(seed);
        let q = r.nibbles(rows * cols);
        let s = r.vals(rows * cols / g);
        let mut fx = Fx::new();
        fx.add("x.weight_packed", "I32", &[rows, cols / 8], &pack_words(&q));
        fx.add("x.weight_shape", "I64", &[2], &[rows as i64, cols as i64].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        fx.add_f("x.weight_scale", dtype, &[rows, cols / g], &s);
        let mut z = vec![8u8; rows * cols / g];
        if zero {
            z = r.nibbles(z.len());
            // [ceil(rows/8), groups] words, 8 consecutive rows per word, one group column each.
            let groups = cols / g;
            let mut words = Vec::new();
            for wr in 0..rows.div_ceil(8) {
                for c in 0..groups {
                    let codes: Vec<u8> = (0..8).map(|k| z.get((wr * 8 + k) * groups + c).copied().filter(|_| wr * 8 + k < rows).unwrap_or(0)).collect();
                    words.extend(pack_words(&codes));
                }
            }
            fx.add("x.weight_zero_point", "I32", &[rows.div_ceil(8), groups], &words);
        }
        prequant::alias_packed(&mut fx.m, &fx.data).unwrap();
        (fx, q, s, z)
    }

    #[test]
    fn int4_repack_equals_f32_path_and_reference() {
        for (g, _, int4_f) in BLOCK {
            for dtype in ["F16", "BF16", "F32"] {
                for zero in [false, true] {
                    let (rows, cols) = (11, 256);
                    let (fx, q, s, z) = ct_fx(rows, cols, g, dtype, zero, g as u64 + zero as u64);
                    let rp = repack(&fx.m, &fx.data, "x.weight").unwrap().expect("repacks");
                    assert_eq!((rp.format, rp.rows, rp.cols, rp.bytes.len()), (int4_f, rows, cols, rows * cols / g * (g / 2 + 4)));
                    let got = dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap();
                    let f32_path = prequant::dequantize(&fx.m, &fx.data, "x.weight").unwrap().unwrap();
                    let want: Vec<f32> = q.iter().enumerate().map(|(i, &n)| {
                        let k = i / cols * (cols / g) + i % cols / g;
                        (n as f32 - z[k] as f32) * s[k]
                    }).collect();
                    let tag = format!("int4 g{g} {dtype} zero={zero}");
                    bit_eq(&got, &f32_path, &format!("{tag} vs f32 path"));
                    bit_eq(&got, &want, &format!("{tag} vs reference"));
                }
            }
        }
    }

    #[test]
    fn int4_repack_declines() {
        let none = |fx: &Fx| repack(&fx.m, &fx.data, "x.weight").unwrap().is_none();
        // Group sizes outside 32 / 64 / 128, per-channel (one scale per row) and per-tensor scales.
        for (cols, g) in [(64, 16), (256, 256), (512, 512)] {
            let (fx, ..) = ct_fx(4, cols, g, "F16", false, 1);
            assert!(none(&fx), "group {g}");
            assert!(prequant::dequantize(&fx.m, &fx.data, "x.weight").unwrap().is_some());
        }
        let (mut fx, ..) = ct_fx(4, 256, 64, "F16", false, 2);
        let t = fx.m.tensors.get_mut("x.weight_scale").unwrap();
        t.shape = vec![1];
        assert!(none(&fx), "per-tensor");
        // Activation order.
        let (mut fx, ..) = ct_fx(4, 128, 64, "F16", false, 3);
        fx.add("x.weight_g_idx", "I32", &[128], &vec![0u8; 128 * 4]);
        assert!(none(&fx), "g_idx");
        // 8-bit pack: i32 words hold four bytes.
        let mut fx = Fx::new();
        fx.add("x.weight_packed", "I32", &[4, 32], &vec![0u8; 4 * 128]);
        fx.add("x.weight_shape", "I64", &[2], &[4i64, 128].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        fx.add_f("x.weight_scale", "F16", &[4, 2], &[1.0; 8]);
        prequant::alias_packed(&mut fx.m, &fx.data).unwrap();
        assert!(none(&fx), "8-bit");
        // Scales that are not f16 (f32 0.1, bf16 below 2^-24, above 65504), zero points that are not whole.
        for bad in [0.1f32, 2f32.powi(-30), 1.0e30] {
            let (mut fx, ..) = ct_fx(4, 128, 64, "F32", false, 4);
            let t = fx.m.tensors["x.weight_scale"].clone();
            fx.data[t.byte_offset..t.byte_offset + 4].copy_from_slice(&bad.to_le_bytes());
            assert!(none(&fx), "scale {bad:e}");
        }
        let (mut fx, ..) = ct_fx(4, 128, 64, "F32", false, 5);
        fx.add_f("x.weight_zero_point", "F32", &[4, 2], &[0.5; 8]);
        assert!(none(&fx), "fractional zero");
        // A global scale the f32 path multiplies in.
        for g in ["x.weight_global_scale", "x.weight_scale_2"] {
            let (mut fx, ..) = ct_fx(4, 128, 64, "F16", false, 7);
            fx.add_f(g, "F32", &[1], &[2.0]);
            assert!(none(&fx), "{g}");
        }
        // Not a pack-quantized weight / unknown name.
        let (fx, ..) = ct_fx(4, 128, 64, "F16", false, 6);
        assert!(repack(&fx.m, &fx.data, "y.weight").unwrap().is_none());
        assert!(repack(&fx.m, &fx.data, "x.weight_scale").unwrap().is_none());
    }
}
