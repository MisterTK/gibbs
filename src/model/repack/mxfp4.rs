//! OCP MXFP4 safetensors (E2M1 values, one e8m0 scale per 32) repacked losslessly to ggml `MXFP4`
//! blocks: 1 scale byte + 16 nibble bytes, byte `i` holding element `i` (low nibble) and `i + 16`
//! (high). The checkpoint's own scale byte is kept; the nibbles are only reordered.
//!
//! Layouts read (both put the even element in the low nibble of each byte):
//! * compressed-tensors `mxfp4-pack-quantized` and Quark MX fp4: `X.weight_packed` / `X.weight`
//!   `[rows, cols / 2]` U8 beside `X.weight_scale` `[rows, cols / 32]` U8;
//! * gpt-oss `X_blocks` `[..., rows, cols / 32, 16]` + `X_scales` `[..., rows, cols / 32]`, exposed by
//!   [`alias`] as one `X` of `[prod(leading) * rows, cols]` (an expert bank is a row concatenation).
//!
//! Value = `E2M1[nibble] * 2^(scale - 127)`, which is what the registered MXFP4 codec decodes.

use super::Repacked;
use crate::hardware::QuantFormat;
use crate::model::prequant::bytes;
use crate::model::{ModelManifest, VirtualTensor};

/// Whether the U8 fp4 weight `{stem}.weight` of `[rows, cols]` is MX (e8m0 scale per 32, no global
/// scale) rather than NVFP4 (e4m3 scale per 16 under a global scale).
pub fn is_mx(m: &ModelManifest, stem: &str, rows: usize, cols: usize) -> bool {
    let get = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    get("weight_scale").is_some_and(|s| s.dtype == "U8" && s.byte_len * 32 == rows * cols)
        && get("weight_global_scale").is_none()
        && get("weight_scale_2").is_none()
}

/// Gives every gpt-oss `X_blocks` / `X_scales` pair a 2-D `X` tagged MXFP4 over the blocks' bytes.
pub fn alias(m: &mut ModelManifest) {
    let mut add = Vec::new();
    for (name, t) in &m.tensors {
        let Some(stem) = name.strip_suffix("_blocks") else {
            continue;
        };
        let (Some(s), true) = (m.tensors.get(&format!("{stem}_scales")), t.dtype == "U8") else {
            continue;
        };
        let n = t.shape.len();
        if n < 3 || t.shape[n - 1] != 16 || s.dtype != "U8" || s.shape[..] != t.shape[..n - 1] {
            continue;
        }
        add.push(VirtualTensor {
            name: stem.to_string(),
            shape: vec![t.shape[..n - 2].iter().product(), t.shape[n - 2] * 32],
            quant_format: QuantFormat::MXFP4,
            ..t.clone()
        });
    }
    for t in add {
        m.tensors.entry(t.name.clone()).or_insert(t);
    }
}

struct Mx<'a> {
    rows: usize,
    cols: usize,
    nibbles: &'a [u8],
    scales: &'a [u8],
}

/// The nibble and scale bytes of `weight` when it is an MXFP4 weight ([`is_mx`] / [`alias`] tagged).
fn parse<'a>(m: &ModelManifest, data: &'a [u8], weight: &str) -> Result<Option<Mx<'a>>, String> {
    let Some(w) = m
        .tensors
        .get(weight)
        .filter(|t| t.quant_format == QuantFormat::MXFP4 && t.dtype == "U8")
    else {
        return Ok(None);
    };
    let scale_name = match weight.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.weight_scale"),
        None => format!("{weight}_scales"),
    };
    let (&[rows, cols], Some(s)) = (&w.shape[..], m.tensors.get(&scale_name)) else {
        return Ok(None);
    };
    if cols % 32 != 0 || w.byte_len != rows * cols / 2 || s.byte_len != rows * cols / 32 {
        return Err(format!(
            "{weight}: {} nibble and {} scale bytes for MXFP4 [{rows}, {cols}]",
            w.byte_len, s.byte_len
        ));
    }
    Ok(Some(Mx { rows, cols, nibbles: bytes(data, w)?, scales: bytes(data, s)? }))
}

impl Mx<'_> {
    fn blocks(&self) -> Vec<u8> {
        let nib = |b: &[u8], k: usize| (b[k / 2] >> (4 * (k % 2))) & 0x0F;
        let mut out = Vec::with_capacity(self.scales.len() * 17);
        for (&e, b) in self.scales.iter().zip(self.nibbles.chunks_exact(16)) {
            out.push(e);
            out.extend((0..16).map(|i| nib(b, i) | nib(b, i + 16) << 4));
        }
        out
    }
}

/// Expands an MXFP4 weight to row-major f32 `[rows, cols]` through the MXFP4 codec's CPU decode, so
/// it is bit-identical to the repacked blocks; `None` when `name` is not one.
pub fn dequantize(m: &ModelManifest, data: &[u8], name: &str) -> Result<Option<Vec<f32>>, String> {
    Ok(parse(m, data, name)?.map(|x| {
        crate::model::layers::dequantize_gguf_blocks(QuantFormat::MXFP4, &x.blocks())
            .expect("MXFP4 has a block codec")
    }))
}

pub fn repack(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Repacked>, String> {
    Ok(parse(m, data, weight)?.map(|x| Repacked {
        format: QuantFormat::MXFP4,
        rows: x.rows,
        cols: x.cols,
        bytes: x.blocks(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::dequantize_gguf_blocks;
    use crate::model::prequant;
    use crate::quant::nvfp4::NVFP4_E2M1_LUT;

    #[derive(Default)]
    struct Fx {
        m: ModelManifest,
        data: Vec<u8>,
    }
    impl Fx {
        fn add(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) {
            self.m.tensors.insert(
                name.into(),
                VirtualTensor {
                    name: name.into(),
                    shape: shape.to_vec(),
                    quant_format: QuantFormat::FP32,
                    byte_offset: self.data.len(),
                    byte_len: bytes.len(),
                    dtype: dtype.into(),
                    shard_index: 0,
                },
            );
            self.data.extend_from_slice(bytes);
        }
    }

    /// Source nibbles (`rows * cols`, any 0..16) and e8m0 scales (`rows * cols / 32`, including the
    /// subnormal-scale bytes 0 and 1) from a fixed LCG.
    fn source(rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        };
        let nibbles = (0..rows * cols).map(|_| next() % 16).collect();
        let scales = (0..rows * cols / 32)
            .map(|i| if i < 3 { i as u8 } else { 100 + next() % 40 })
            .collect();
        (nibbles, scales)
    }

    /// `value = fp4_table[nibble] * 2^(scale - 127)`.
    fn reference(nibbles: &[u8], scales: &[u8]) -> Vec<f32> {
        nibbles
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let e = scales[i / 32] as i32 - 127;
                (NVFP4_E2M1_LUT[n as usize] as f64 * 2f64.powi(e)) as f32
            })
            .collect()
    }

    /// Adjacent elements per byte, even element in the low nibble.
    fn pack(nibbles: &[u8]) -> Vec<u8> {
        nibbles.chunks(2).map(|p| p[0] | p[1] << 4).collect()
    }

    fn eq(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        // `==`: the codec's integer table has no -0.
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g == w, "{what}[{i}]: {g} vs {w}");
        }
    }

    #[test]
    fn compressed_tensors_mxfp4_repacks_to_the_codec() {
        let (rows, cols) = (5, 96);
        let (nibbles, scales) = source(rows, cols);
        let mut fx = Fx::default();
        fx.add("x.weight_packed", "U8", &[rows, cols / 2], &pack(&nibbles));
        fx.add("x.weight_scale", "U8", &[rows, cols / 32], &scales);
        prequant::alias_packed(&mut fx.m, &fx.data).unwrap();
        assert_eq!(fx.m.tensors["x.weight"].quant_format, QuantFormat::MXFP4);
        assert_eq!(fx.m.tensors["x.weight"].shape, [rows, cols]);

        let rp = repack(&fx.m, &fx.data, "x.weight").unwrap().expect("repacks");
        assert_eq!((rp.format, rp.rows, rp.cols, rp.bytes.len()), (QuantFormat::MXFP4, rows, cols, rows * cols / 32 * 17));
        // The checkpoint's scale byte is carried verbatim.
        assert!(rp.bytes.chunks(17).zip(&scales).all(|(b, &e)| b[0] == e));
        let want = reference(&nibbles, &scales);
        eq(&dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap(), &want, "codec decode");
        // The loader's f32 path and the repack agree bit for bit.
        let f32_path = prequant::dequantize(&fx.m, &fx.data, "x.weight").unwrap().unwrap();
        assert_eq!(f32_path, dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap());
        eq(&f32_path, &want, "f32 path");
    }

    #[test]
    fn gpt_oss_blocks_and_scales_repack_as_an_expert_bank() {
        let (experts, out, groups) = (2, 3, 2);
        let cols = groups * 32;
        let (nibbles, scales) = source(experts * out, cols);
        let mut fx = Fx::default();
        fx.add("e.gate_up_proj_blocks", "U8", &[experts, out, groups, 16], &pack(&nibbles));
        fx.add("e.gate_up_proj_scales", "U8", &[experts, out, groups], &scales);
        prequant::alias_packed(&mut fx.m, &fx.data).unwrap();
        let t = &fx.m.tensors["e.gate_up_proj"];
        assert_eq!((t.quant_format, &t.shape[..]), (QuantFormat::MXFP4, &[experts * out, cols][..]));

        let rp = repack(&fx.m, &fx.data, "e.gate_up_proj").unwrap().expect("repacks");
        assert_eq!((rp.rows, rp.cols), (experts * out, cols));
        eq(&dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap(), &reference(&nibbles, &scales), "gpt-oss");
    }

    #[test]
    fn nvfp4_and_mismatched_sizes_are_not_mxfp4() {
        // modelopt NVFP4: e4m3 scale per 16 under a global scale.
        let mut fx = Fx::default();
        fx.add("x.weight", "U8", &[2, 16], &[0u8; 32]);
        fx.add("x.weight_scale", "U8", &[2, 32], &[0x38u8; 64]);
        fx.add("x.weight_scale_2", "F32", &[], &1f32.to_le_bytes());
        prequant::alias_packed(&mut fx.m, &fx.data).unwrap();
        assert_eq!(fx.m.tensors["x.weight"].quant_format, QuantFormat::NVFP4);
        assert!(repack(&fx.m, &fx.data, "x.weight").unwrap().is_none());
        // A scale grid that does not tile the nibbles is an error, not a silent mis-decode.
        let mut fx = Fx::default();
        fx.add("x.weight_packed", "U8", &[2, 32], &[0u8; 64]);
        fx.add("x.weight_scale", "U8", &[2, 1], &[127u8; 2]);
        prequant::alias_packed(&mut fx.m, &fx.data).unwrap();
        fx.m.tensors.get_mut("x.weight").unwrap().quant_format = QuantFormat::MXFP4;
        assert!(repack(&fx.m, &fx.data, "x.weight").is_err());
    }
}
