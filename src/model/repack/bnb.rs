//! bitsandbytes 4-bit (NF4 / FP4, optionally double-quantized absmax): reads the safetensors layout,
//! dequantizes to f32 ([`dequantize`]) and repacks to the `BNB_{NF4,FP4}_B{64,128,256}` codecs.
//!
//! Checkpoint layout per weight `W` (`X.weight`): `W` (U8, two nibbles per byte over the flattened
//! row-major weight, FIRST element in the HIGH nibble), `W.absmax` (f32 per block, or u8 when nested),
//! `W.nested_absmax` / `W.nested_quant_map` (nested only), `W.quant_map` (16 f32) and
//! `W.quant_state.bitsandbytes__{nf4|fp4}` (U8 JSON: `blocksize`, `shape`, `nested_blocksize`,
//! `nested_offset`). Value = `quant_map[nibble] * absmax[block]`; nested:
//! `absmax = nested_quant_map[u8] * nested_absmax[block / nested_blocksize] + nested_offset`.
//!
//! Codec block (`4 + blocksize/2` bytes): f32 absmax, then the checkpoint's nibble bytes unchanged
//! (byte `j` = elements `2j` high nibble, `2j+1` low nibble).

use super::Repacked;
use crate::graph::codec::Spec;
use crate::hardware::QuantFormat;
use crate::model::layers::GgufBlockFn;
use crate::model::loaders::onnx::{FP4_LUT, NF4_LUT};
use crate::model::{decode_dtype, ModelManifest, VirtualTensor};
use serde_json::Value;

/// MSL tables: the f32 values of [`NF4_LUT`] / [`FP4_LUT`] (`codec_parity_bnb` proves they agree).
macro_rules! nf4_table {
    () => {
        "#ifndef GIBBS_TBL_BNB_NF4\n#define GIBBS_TBL_BNB_NF4\nconstant float BNB_NF4[16] = {\
         -1.0f,-0.6961928f,-0.52507305f,-0.3949175f,-0.28444138f,-0.18477343f,-0.091050036f,0.0f,\
         0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.0f};\n#endif\n"
    };
}
macro_rules! fp4_table {
    () => {
        "#ifndef GIBBS_TBL_BNB_FP4\n#define GIBBS_TBL_BNB_FP4\nconstant float BNB_FP4[16] = {\
         0.0f,0.005208333f,0.6666667f,1.0f,0.33333334f,0.5f,0.16666667f,0.25f,\
         -0.0f,-0.005208333f,-0.6666667f,-1.0f,-0.33333334f,-0.5f,-0.16666667f,-0.25f};\n#endif\n"
    };
}

macro_rules! spec {
    ($format:ident, $bs:literal, $table:expr, $lut:literal) => {
        Spec {
            format: QuantFormat::$format,
            block_elems: $bs,
            block_bytes: 4 + $bs / 2,
            tables: &[],
            body: concat!(
                $table,
                "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n\
                 float a = as_type<float>(cd_u32(blk));\n\
                 device const uchar* q = blk + 4 + sub * 4;\n\
                 for (uint i = 0; i < 4; i++) {\n\
                     out8[2 * i] = ",
                $lut,
                "[q[i] >> 4] * a;\n\
                     out8[2 * i + 1] = ",
                $lut,
                "[q[i] & 0x0F] * a;\n\
                 }\n}\n"
            ),
        }
    };
}

/// GPU codecs of this family (MSL `decode8`), merged into the registry by `graph::codec`.
pub const SPECS: &[Spec] = &[
    spec!(BNB_NF4_B64, 64, nf4_table!(), "BNB_NF4"),
    spec!(BNB_NF4_B128, 128, nf4_table!(), "BNB_NF4"),
    spec!(BNB_NF4_B256, 256, nf4_table!(), "BNB_NF4"),
    spec!(BNB_FP4_B64, 64, fp4_table!(), "BNB_FP4"),
    spec!(BNB_FP4_B128, 128, fp4_table!(), "BNB_FP4"),
    spec!(BNB_FP4_B256, 256, fp4_table!(), "BNB_FP4"),
];

fn dec<const FP4: bool, const BS: usize>(b: &[u8], y: &mut [f32]) {
    let lut = if FP4 { &FP4_LUT } else { &NF4_LUT };
    let a = f32::from_le_bytes(b[..4].try_into().unwrap());
    for (j, &q) in b[4..4 + BS / 2].iter().enumerate() {
        y[2 * j] = lut[(q >> 4) as usize] * a;
        y[2 * j + 1] = lut[(q & 0x0F) as usize] * a;
    }
}

pub fn block_codec(format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    use QuantFormat as Q;
    Some(match format {
        Q::BNB_NF4_B64 => (64, 36, dec::<false, 64> as GgufBlockFn),
        Q::BNB_NF4_B128 => (128, 68, dec::<false, 128>),
        Q::BNB_NF4_B256 => (256, 132, dec::<false, 256>),
        Q::BNB_FP4_B64 => (64, 36, dec::<true, 64>),
        Q::BNB_FP4_B128 => (128, 68, dec::<true, 128>),
        Q::BNB_FP4_B256 => (256, 132, dec::<true, 256>),
        _ => return None,
    })
}

/// Why a `bitsandbytes` `quantization_config` can't load: only 4-bit NF4/FP4 is decoded.
pub fn blocker(q: &Value) -> Option<String> {
    if q["load_in_8bit"] == true {
        return Some("bitsandbytes 8-bit (LLM.int8) checkpoints are not supported (4-bit nf4/fp4 are)".into());
    }
    match q["bnb_4bit_quant_type"].as_str() {
        None | Some("nf4" | "fp4") => None,
        Some(t) => Some(format!("bitsandbytes 4-bit quant type '{t}' is not supported (nf4, fp4 are)")),
    }
}

/// The checkpoint's own tensors of one bnb weight.
struct Bnb<'a> {
    fp4: bool,
    blocksize: usize,
    rows: usize,
    cols: usize,
    /// Flattened nibble bytes, first element in the high nibble.
    nibbles: &'a [u8],
    /// One f32 per block (nested absmax already expanded).
    absmax: Vec<f32>,
    quant_map: [f32; 16],
}

fn raw<'a>(data: &'a [u8], t: &VirtualTensor) -> Result<&'a [u8], String> {
    data.get(t.byte_offset..t.byte_offset + t.byte_len)
        .ok_or_else(|| format!("tensor '{}' lies outside the weight buffer", t.name))
}

/// `(is_fp4, quant_state JSON)` when `weight` is a bnb 4-bit weight.
fn state(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<(bool, Value)>, String> {
    for (fp4, kind) in [(false, "nf4"), (true, "fp4")] {
        if let Some(t) = m.tensors.get(&format!("{weight}.quant_state.bitsandbytes__{kind}")) {
            let json = serde_json::from_slice(raw(data, t)?).map_err(|e| format!("{}: {e}", t.name))?;
            return Ok(Some((fp4, json)));
        }
    }
    Ok(None)
}

/// Gives every bnb weight its logical `[out, in]` shape and `BNB_NF4`/`BNB_FP4` tag (stored it is
/// `[numel/2, 1]` U8). Run once, from `prequant::alias_packed`.
pub fn alias(m: &mut ModelManifest, data: &[u8]) -> Result<(), String> {
    let mut fix = Vec::new();
    for name in m.tensors.keys().filter(|n| n.ends_with(".weight")) {
        if let Some((fp4, q)) = state(m, data, name)? {
            fix.push((name.clone(), fp4, shape(&q).map_err(|e| format!("{name}: {e}"))?));
        }
    }
    for (name, fp4, shape) in fix {
        let t = m.tensors.get_mut(&name).unwrap();
        t.shape = shape.to_vec();
        t.quant_format = if fp4 { QuantFormat::BNB_FP4 } else { QuantFormat::BNB_NF4 };
    }
    Ok(())
}

/// Companion tensors of a bnb weight (`X.weight.absmax`, `.quant_map`, `.nested_*`, `.quant_state.*`):
/// quantizer state, not model weights.
pub fn is_companion(name: &str) -> bool {
    name.contains(".weight.quant_state.")
        || [".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map"]
            .iter()
            .any(|s| name.strip_suffix(s).is_some_and(|w| w.ends_with(".weight")))
}

fn shape(q: &Value) -> Result<[usize; 2], String> {
    let dims: Vec<usize> = q["shape"]
        .as_array()
        .map(|s| s.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect())
        .unwrap_or_default();
    <[usize; 2]>::try_from(dims).map_err(|_| format!("bnb quant_state shape {} is not [out, in]", q["shape"]))
}

fn parse<'a>(m: &ModelManifest, data: &'a [u8], weight: &str) -> Result<Option<Bnb<'a>>, String> {
    let Some((fp4, q)) = state(m, data, weight)? else {
        return Ok(None);
    };
    let err = |e: String| format!("{weight}: {e}");
    let get = |s: &str| m.tensors.get(&format!("{weight}.{s}"));
    let f32s = |s: &str| -> Result<Option<Vec<f32>>, String> {
        get(s)
            .map(|t| decode_dtype(raw(data, t)?, &t.dtype).map_err(err))
            .transpose()
    };
    let [rows, cols] = shape(&q).map_err(err)?;
    let blocksize = q["blocksize"].as_u64().filter(|&b| b > 0).ok_or_else(|| err("bnb quant_state has no blocksize".into()))? as usize;
    let numel = rows * cols;
    let nibbles = raw(data, m.tensors.get(weight).ok_or_else(|| err("missing".into()))?)?;
    if nibbles.len() != numel.div_ceil(2) {
        return Err(err(format!("{} bytes for 4-bit [{rows}, {cols}]", nibbles.len())));
    }
    let blocks = numel.div_ceil(blocksize);
    let absmax_t = get("absmax").ok_or_else(|| err("bnb weight has no `absmax` companion".into()))?;
    let absmax = match f32s("nested_absmax")? {
        // Double quantization: u8 absmax codes through the 256-entry nested map.
        Some(nested) => {
            let nmap = f32s("nested_quant_map")?.ok_or_else(|| err("nested absmax without `nested_quant_map`".into()))?;
            let nb = q["nested_blocksize"].as_u64().filter(|&b| b > 0).unwrap_or(256) as usize;
            let off = q["nested_offset"].as_f64().unwrap_or(0.0) as f32;
            let codes = raw(data, absmax_t)?;
            if nmap.len() != 256 || codes.len() != blocks || nested.len() != blocks.div_ceil(nb) {
                return Err(err("nested absmax sizes do not match the weight".into()));
            }
            codes.iter().enumerate().map(|(i, &c)| nmap[c as usize] * nested[i / nb] + off).collect()
        }
        None => f32s("absmax")?.unwrap(),
    };
    if absmax.len() != blocks {
        return Err(err(format!("{} absmax values for {blocks} blocks", absmax.len())));
    }
    let quant_map = match f32s("quant_map")? {
        Some(v) => v.try_into().map_err(|_| err("quant_map is not 16 values".into()))?,
        None => if fp4 { FP4_LUT } else { NF4_LUT },
    };
    Ok(Some(Bnb { fp4, blocksize, rows, cols, nibbles, absmax, quant_map }))
}

/// Expands a bnb weight to row-major f32 `[out, in]`; `None` when `name` is not one.
pub fn dequantize(m: &ModelManifest, data: &[u8], name: &str) -> Result<Option<Vec<f32>>, String> {
    let Some(b) = parse(m, data, name)? else {
        return Ok(None);
    };
    Ok(Some(
        (0..b.rows * b.cols)
            .map(|i| {
                let byte = b.nibbles[i / 2];
                let nib = if i % 2 == 0 { byte >> 4 } else { byte & 0x0F };
                b.quant_map[nib as usize] * b.absmax[i / b.blocksize]
            })
            .collect(),
    ))
}

pub fn repack(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Repacked>, String> {
    let Some(b) = parse(m, data, weight)? else {
        return Ok(None);
    };
    let bs = b.blocksize;
    if ![64, 128, 256].contains(&bs) {
        return Err(format!("bnb blocksize {bs} unsupported"));
    }
    let table = if b.fp4 { &FP4_LUT } else { &NF4_LUT };
    // Rows must hold whole blocks, and the codec tables are the standard maps only.
    if b.cols % bs != 0 || b.quant_map.iter().zip(table).any(|(a, c)| a.to_bits() != c.to_bits()) {
        return Ok(None);
    }
    let format = match (b.fp4, bs) {
        (false, 64) => QuantFormat::BNB_NF4_B64,
        (false, 128) => QuantFormat::BNB_NF4_B128,
        (false, _) => QuantFormat::BNB_NF4_B256,
        (true, 64) => QuantFormat::BNB_FP4_B64,
        (true, 128) => QuantFormat::BNB_FP4_B128,
        (true, _) => QuantFormat::BNB_FP4_B256,
    };
    let mut bytes = Vec::with_capacity(b.absmax.len() * (4 + bs / 2));
    for (a, nib) in b.absmax.iter().zip(b.nibbles.chunks(bs / 2)) {
        bytes.extend_from_slice(&a.to_le_bytes());
        bytes.extend_from_slice(nib);
    }
    Ok(Some(Repacked { format, rows: b.rows, cols: b.cols, bytes }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::dequantize_gguf_blocks;
    use crate::model::UniversalModelLoader;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    #[derive(Default)]
    struct Fixture {
        m: ModelManifest,
        data: Vec<u8>,
    }
    impl Fixture {
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
        fn f32s(&mut self, name: &str, v: &[f32]) {
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.add(name, "F32", &[v.len()], &b);
        }
    }

    /// Builds a `[rows, cols]` bnb weight `w.weight` and returns the independent reference dequant
    /// computed straight from the original layout (separate from [`dequantize`]).
    fn build(fp4: bool, bs: usize, nested: bool, rows: usize, cols: usize, seed: u64) -> (Fixture, Vec<f32>) {
        let mut r = Lcg(seed);
        let (numel, blocks) = (rows * cols, rows * cols / bs);
        let nibbles: Vec<u8> = (0..numel / 2).map(|_| (r.next() >> 3) as u8).collect();
        let lut = if fp4 { FP4_LUT } else { NF4_LUT };
        let mut f = Fixture::default();
        f.add("w.weight", "U8", &[numel / 2, 1], &nibbles);
        f.f32s("w.weight.quant_map", &lut);
        let absmax: Vec<f32> = if nested {
            let codes: Vec<u8> = (0..blocks).map(|_| (r.next() >> 2) as u8).collect();
            let nmap: Vec<f32> = (0..256).map(|_| r.unit()).collect();
            let nabs: Vec<f32> = (0..blocks.div_ceil(256)).map(|_| r.unit().abs() + 0.1).collect();
            f.add("w.weight.absmax", "U8", &[blocks], &codes);
            f.f32s("w.weight.nested_absmax", &nabs);
            f.f32s("w.weight.nested_quant_map", &nmap);
            codes.iter().enumerate().map(|(i, &c)| nmap[c as usize] * nabs[i / 256] + 0.25).collect()
        } else {
            let a: Vec<f32> = (0..blocks).map(|_| r.unit() * 3.0).collect();
            f.f32s("w.weight.absmax", &a);
            a
        };
        let json = format!(
            r#"{{"quant_type":"{}","blocksize":{bs},"dtype":"bfloat16","shape":[{rows},{cols}],"nested_blocksize":{},"nested_offset":0.25}}"#,
            if fp4 { "fp4" } else { "nf4" },
            if nested { "256" } else { "null" },
        );
        f.add(&format!("w.weight.quant_state.bitsandbytes__{}", if fp4 { "fp4" } else { "nf4" }), "U8", &[json.len()], json.as_bytes());
        // Original layout: element i is the high (even i) / low (odd i) nibble of byte i/2.
        let want = (0..numel)
            .map(|i| {
                let n = if i % 2 == 0 { nibbles[i / 2] >> 4 } else { nibbles[i / 2] & 15 };
                lut[n as usize] * absmax[i / bs]
            })
            .collect();
        alias(&mut f.m, &f.data).unwrap();
        (f, want)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn repack_equals_reference_and_f32_path() {
        for fp4 in [false, true] {
            for bs in [64, 128, 256] {
                for nested in [false, true] {
                    let (f, want) = build(fp4, bs, nested, 3, 512, bs as u64 + nested as u64);
                    let t = &f.m.tensors["w.weight"];
                    assert_eq!(t.shape, [3, 512]);
                    assert_eq!(t.quant_format, if fp4 { QuantFormat::BNB_FP4 } else { QuantFormat::BNB_NF4 });
                    let ctx = format!("fp4={fp4} bs={bs} nested={nested}");
                    let f32_path =
                        UniversalModelLoader::extract_tensor_from_manifest(&f.m, &f.data, "w.weight").unwrap();
                    assert_eq!(bits(&f32_path), bits(&want), "f32 path {ctx}");
                    let rp = repack(&f.m, &f.data, "w.weight").unwrap().expect(&ctx);
                    assert_eq!((rp.rows, rp.cols), (3, 512));
                    assert_eq!(rp.bytes.len(), 3 * 512 / bs * (4 + bs / 2));
                    let got = dequantize_gguf_blocks(rp.format, &rp.bytes).unwrap();
                    assert_eq!(bits(&got), bits(&want), "repack {ctx}");
                }
            }
        }
    }

    #[test]
    fn blocksize_32_is_an_error() {
        let (f, _) = build(false, 32, false, 2, 64, 7);
        assert_eq!(repack(&f.m, &f.data, "w.weight").err().as_deref(), Some("bnb blocksize 32 unsupported"));
        // The f32 path still loads it.
        assert!(UniversalModelLoader::extract_tensor_from_manifest(&f.m, &f.data, "w.weight").is_ok());
    }

    #[test]
    fn non_standard_layouts_stay_on_the_f32_path() {
        // cols not a multiple of the blocksize (blocks straddle rows).
        let (f, _) = build(false, 64, false, 4, 96, 3);
        assert!(repack(&f.m, &f.data, "w.weight").unwrap().is_none());
        // Unfamiliar code book.
        let (mut f, _) = build(false, 64, false, 2, 128, 4);
        let t = f.m.tensors["w.weight.quant_map"].clone();
        f.data[t.byte_offset] ^= 1;
        assert!(repack(&f.m, &f.data, "w.weight").unwrap().is_none());
        // Ordinary tensors are not bnb.
        assert!(repack(&f.m, &f.data, "other.weight").unwrap().is_none());
    }

    #[test]
    fn blocker_and_companions() {
        let q = |v: Value| blocker(&v);
        assert_eq!(q(serde_json::json!({"quant_method": "bitsandbytes"})), None);
        assert_eq!(q(serde_json::json!({"load_in_4bit": true, "bnb_4bit_quant_type": "nf4"})), None);
        assert!(q(serde_json::json!({"load_in_8bit": true})).unwrap().contains("8-bit"));
        assert!(q(serde_json::json!({"bnb_4bit_quant_type": "x"})).is_some());
        for n in ["a.q_proj.weight.absmax", "a.q_proj.weight.quant_map", "a.q_proj.weight.nested_absmax", "a.q_proj.weight.nested_quant_map", "a.q_proj.weight.quant_state.bitsandbytes__nf4"] {
            assert!(is_companion(n), "{n}");
        }
        assert!(!is_companion("a.q_proj.weight") && !is_companion("model.norm.weight"));
    }
}
