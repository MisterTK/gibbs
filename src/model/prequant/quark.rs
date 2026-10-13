//! AMD Quark exports (`quant_method: "quark"`), laid out as github.com/amd/Quark (release 0.13)
//! `quark/torch/utils/pack.py` and `export/nn/modules/realquantizer.py` write them:
//! - fp8 and per-tensor / per-channel int8: a plain `[out, in]` weight beside `weight_scale` (and int8
//!   `weight_zero_point`), expanded by the shared path in [`super::dequantize`].
//! - per-group int2 / int4 / uint4 / int8: the weight is transposed to `[in, out]` before packing and
//!   scale / zero point are `[in/g, out]`. int4 / uint4 pack 8 nibbles per i32 along `out`, nibble k
//!   holding column `[0, 2, 4, 6, 1, 3, 5, 7][k]` of each 8 (`pack_method: reorder`) or column k
//!   (`order`); int2 and int8 keep one integer per element.
//! - per-group int3: not transposed; 8 codes per 3 bytes (little-endian 24 bits, lowest first) along `in`.
//! - OCP MX fp4 / fp6_e3m2 / fp6_e2m3: `[out, in·bits/8]` (fp4 low nibble first, fp6 four codes per
//!   little-endian 24 bits) with e8m0 scales `[out, in/32]`.
//!
//! Which of these a weight uses lives only in `config.json`, so [`annotate`] records it in the manifest
//! metadata (`quark:<module>`) for [`dequantize`]. Activation, output and KV-cache quantizers only
//! affect activations; weights are expanded to f32, so their scales are unused.

use super::{block, bytes};
use crate::model::{decode_dtype, ModelManifest, VirtualTensor};
use serde_json::Value;
use std::path::Path;

/// Weight dtypes and the qschemes of each that [`dequantize`] (or the shared path) expands.
const SUPPORTED: &[(&str, &[&str])] = &[
    ("fp8_e4m3", &["per_tensor", "per_channel", "per_block"]),
    ("fp8_e5m2", &["per_tensor", "per_channel", "per_block"]),
    ("int8", &["per_tensor", "per_channel", "per_group"]),
    ("int4", &["per_group"]),
    ("uint4", &["per_group"]),
    ("int3", &["per_group"]),
    ("int2", &["per_group"]),
    ("fp4", &["per_group"]),
    ("fp6_e3m2", &["per_group"]),
    ("fp6_e2m3", &["per_group"]),
];

/// Algorithms whose effect is folded into the exported weights (smoothing scales into the previous
/// op, offline rotations into neighbouring weights).
const FOLDED_ALGOS: &[&str] = &[
    "awq",
    "gptq",
    "qronos",
    "smooth",
    "smoothquant",
    "autosmoothquant",
    "rotation",
];

/// Why the quark `quantization_config` `qc` can't load.
pub fn blocker(qc: &Value) -> Option<String> {
    let pack = pack_method(qc);
    if !matches!(pack, "reorder" | "order") {
        return Some(format!(
            "quark pack_method '{pack}' is not supported (reorder, order are)"
        ));
    }
    let algos: Vec<&Value> = match &qc["algo_config"] {
        Value::Array(v) => v.iter().collect(),
        Value::Null => Vec::new(),
        a => vec![a],
    };
    for a in algos {
        let name = a["name"].as_str().unwrap_or("");
        if !FOLDED_ALGOS.contains(&name) {
            return Some(format!(
                "quark algorithm '{name}' is not supported (supported: {})",
                FOLDED_ALGOS.join(", ")
            ));
        }
        let online = a["online_config"]["online_rotation_layers"]
            .as_array()
            .is_some_and(|l| !l.is_empty());
        if name == "rotation"
            && (online || a["online_r1_rotation"] == true || a["r3"] == true || a["r4"] == true)
        {
            return Some(
                "quark rotation with online rotations (runtime Hadamard on activations) is not supported"
                    .into(),
            );
        }
    }
    let patterns = ["layer_quant_config", "layer_type_quant_config"]
        .iter()
        .flat_map(|k| qc[k].as_object().into_iter().flatten());
    if let Some((p, _)) = patterns.clone().find(|(p, _)| p.contains('[')) {
        // ponytail: glob supports `*` and `?`; add `[...]` classes when an export uses them.
        return Some(format!(
            "quark layer pattern '{p}' uses a character class (not supported)"
        ));
    }
    std::iter::once(&qc["global_quant_config"])
        .chain(patterns.map(|(_, c)| c))
        .find_map(layer_blocker)
}

fn layer_blocker(layer: &Value) -> Option<String> {
    if !layer["bias"].is_null() {
        return Some("quark quantized biases are not supported".into());
    }
    let w = &layer["weight"];
    if w.is_null() {
        return None;
    }
    if w.is_array() {
        return Some(
            "quark multi-stage weight quantization (e.g. NVFP4 [fp4, fp8], W4A8 [fp8, int4]) is not supported"
                .into(),
        );
    }
    let (dtype, qscheme) = (
        w["dtype"].as_str().unwrap_or(""),
        w["qscheme"].as_str().unwrap_or(""),
    );
    if !SUPPORTED
        .iter()
        .any(|(d, q)| *d == dtype && q.contains(&qscheme))
    {
        let list: Vec<String> = SUPPORTED
            .iter()
            .map(|(d, q)| format!("{d} {}", q.join("/")))
            .collect();
        return Some(format!(
            "quark weight {dtype} {qscheme} is not supported (supported: {})",
            list.join(", ")
        ));
    }
    if dtype.starts_with("fp4") || dtype.starts_with("fp6") {
        if w["scale_format"] != "e8m0" || w["group_size"] != 32 {
            return Some(format!(
                "quark {dtype} weights are supported with e8m0 scales in groups of 32 (OCP MX) only"
            ));
        }
    } else if qscheme == "per_group" {
        let axis = w.get("ch_axis").or_else(|| w.get("axis"));
        if !axis
            .and_then(Value::as_i64)
            .is_some_and(|a| a == 1 || a == -1)
        {
            return Some(format!(
                "quark per-group {dtype} needs groups along the input axis (ch_axis 1)"
            ));
        }
    }
    None
}

fn pack_method(qc: &Value) -> &str {
    qc["export"]["pack_method"]
        .as_str()
        .or(qc["pack_method"].as_str())
        .unwrap_or("reorder")
}

/// `fnmatch` with `*` and `?` (the patterns [`blocker`] lets through).
fn glob(p: &[u8], s: &[u8]) -> bool {
    match (p.first(), s.first()) {
        (None, None) => true,
        (Some(b'*'), _) => glob(&p[1..], s) || (!s.is_empty() && glob(p, &s[1..])),
        (Some(b'?'), Some(_)) => glob(&p[1..], &s[1..]),
        (Some(a), Some(b)) if a == b => glob(&p[1..], &s[1..]),
        _ => false,
    }
}

/// The weight quantizer quark applies to Linear `module`: `layer_quant_config` (fnmatch) first, then
/// `layer_type_quant_config.Linear`, then `global_quant_config`; `None` for an `exclude`d module.
fn weight_spec<'a>(qc: &'a Value, module: &str) -> Result<Option<&'a Value>, String> {
    let m = module.as_bytes();
    let excluded = qc["exclude"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|p| glob(p.as_bytes(), m));
    if excluded {
        return Ok(None);
    }
    // serde_json maps are sorted, not in file order: quark takes the first match, so overlapping
    // patterns must agree.
    let mut hits = qc["layer_quant_config"]
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(p, _)| glob(p.as_bytes(), m))
        .map(|(_, c)| &c["weight"]);
    if let Some(first) = hits.next() {
        if hits.any(|h| h != first) {
            return Err(format!(
                "{module}: overlapping quark layer_quant_config patterns name different weight schemes"
            ));
        }
        return Ok(Some(first));
    }
    Ok(Some(
        qc["layer_type_quant_config"]
            .get("Linear")
            .map_or(&qc["global_quant_config"]["weight"], |c| &c["weight"]),
    ))
}

/// (bits, mantissa bits, exponent bias) of an OCP MX element dtype.
fn mx_format(dtype: &str) -> Option<(u32, u32, u32)> {
    match dtype {
        "fp4" => Some((4, 1, 1)),
        "fp6_e3m2" => Some((6, 2, 3)),
        "fp6_e2m3" => Some((6, 3, 1)),
        _ => None,
    }
}

/// Logical `[out, in]` of a weight [`dequantize`] expands, from its stored extent and scale grid;
/// `None` for the plain layouts the shared path reads (fp8, per-tensor / per-channel int8).
fn logical_shape(
    w: &VirtualTensor,
    s: &VirtualTensor,
    dtype: &str,
    qscheme: &str,
) -> Result<Option<[usize; 2]>, String> {
    if dtype.starts_with("fp8") || (dtype == "int8" && qscheme != "per_group") {
        return Ok(None);
    }
    let [sr, sc] = s.shape[..] else {
        return Err(format!("scale shape {:?} is not 2-D", s.shape));
    };
    let n = w.byte_len;
    if let Some((bits, ..)) = mx_format(dtype) {
        let bits = bits as usize;
        let cols = (n * 8 / bits).checked_div(sr).unwrap_or(0);
        if s.dtype != "U8" || n * 8 != sr * cols * bits || cols.div_ceil(32) != sc {
            return Err(format!(
                "{n} bytes do not match e8m0 scales {:?} {}",
                s.shape, s.dtype
            ));
        }
        return Ok(Some([sr, cols]));
    }
    // Stored `[in, out]` (`[out, in]` for int3); packed rows round up to whole words.
    let rows = w.shape.first().copied().unwrap_or(0);
    let per_row = match dtype {
        "int4" | "uint4" => n / rows.max(1) * 2,
        "int2" => n / rows.max(1) / 4,
        "int3" => n / rows.max(1) / 3 * 8,
        _ => n / rows.max(1),
    };
    if dtype == "int3" {
        return match (sr == rows, sc == rows) {
            (true, false) | (false, true) => Ok(Some([rows, per_row])),
            _ => Err(format!(
                "scale grid {:?} does not match {rows} outputs",
                s.shape
            )),
        };
    }
    let fits = |x: usize| x <= per_row && per_row - x < 8;
    match (fits(sc), fits(sr)) {
        (true, false) => Ok(Some([sc, rows])),
        (false, true) => Ok(Some([sr, rows])),
        _ => Err(format!(
            "scale grid {:?} does not match {per_row} packed outputs",
            s.shape
        )),
    }
}

/// When `config_path` names a quark checkpoint: records each quantized module's weight dtype,
/// qscheme and pack order (`quark:<module>` metadata) and gives the weights [`dequantize`] expands
/// their logical `[out, in]` shape. Run once on a complete manifest.
pub fn annotate(m: &mut ModelManifest, config_path: Option<&Path>) -> Result<(), String> {
    let Some(p) = config_path else {
        return Ok(());
    };
    let cfg: Value =
        serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
            .map_err(|e| format!("{}: {e}", p.display()))?;
    let qc = &cfg["quantization_config"];
    if qc["quant_method"] != "quark" {
        return Ok(());
    }
    let pack = pack_method(qc);
    let (mut specs, mut shapes) = (Vec::new(), Vec::new());
    for (name, s) in &m.tensors {
        let Some(module) = name.strip_suffix(".weight_scale") else {
            continue;
        };
        let Some(spec) = weight_spec(qc, module)?.filter(|w| !w.is_null()) else {
            continue;
        };
        let (dtype, qscheme) = (
            spec["dtype"].as_str().unwrap_or(""),
            spec["qscheme"].as_str().unwrap_or(""),
        );
        let weight = format!("{module}.weight");
        let Some(w) = m.tensors.get(&weight) else {
            continue;
        };
        if matches!(w.dtype.as_str(), "F32" | "F16" | "BF16") {
            continue; // fake-quantized export: real values beside calibration scales
        }
        let shape = logical_shape(w, s, dtype, qscheme)
            .map_err(|e| format!("{weight}: quark {dtype} {qscheme}: {e}"))?;
        specs.push((
            format!("quark:{module}"),
            format!("{dtype} {qscheme} {pack}"),
        ));
        shapes.extend(shape.map(|s| (weight, s)));
    }
    m.metadata.extend(specs);
    for (name, shape) in shapes {
        let t = m.tensors.get_mut(&name).unwrap();
        t.shape = shape.to_vec();
        // quark expands its own fp4 (`dequantize`); `alias_packed` cannot tell it from OCP MXFP4
        // checkpoints, which repack to ggml blocks, so it keeps the tag it always had.
        if t.quant_format == crate::hardware::QuantFormat::MXFP4 {
            t.quant_format = crate::hardware::QuantFormat::NVFP4;
        }
    }
    Ok(())
}

/// Expands quark weight `name` to row-major f32 `[out, in]`; `None` when [`annotate`] recorded no
/// quark scheme for it or the shared path decodes it (float, fp8, per-tensor / per-channel int8).
pub fn dequantize(m: &ModelManifest, data: &[u8], name: &str) -> Result<Option<Vec<f32>>, String> {
    let Some(module) = name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let Some(spec) = m.metadata.get(&format!("quark:{module}")) else {
        return Ok(None);
    };
    let mut spec = spec.split(' ');
    let (dtype, qscheme, pack) = (
        spec.next().unwrap_or(""),
        spec.next().unwrap_or(""),
        spec.next().unwrap_or(""),
    );
    let get = |s: &str| m.tensors.get(&format!("{module}.{s}"));
    let (Some(w), Some(s)) = (m.tensors.get(name), get("weight_scale")) else {
        return Ok(None);
    };
    let int8_plain = dtype == "int8" && qscheme != "per_group";
    if matches!(w.dtype.as_str(), "F32" | "F16" | "BF16") || dtype.starts_with("fp8") || int8_plain
    {
        return Ok(None);
    }
    let err = |e: String| format!("{name}: quark {dtype} {qscheme}: {e}");
    let raw = bytes(data, w)?;
    let (&[out, inn], &[sr, sc]) = (&w.shape[..], &s.shape[..]) else {
        return Err(err(format!("shapes {:?} / {:?}", w.shape, s.shape)));
    };

    if let Some((bits, mant, bias)) = mx_format(dtype) {
        let scale = bytes(data, s)?;
        // quark `Pack_fp4` / `Pack_fp6` unpack: the code's sign and exponent|mantissa bits placed in
        // an f32, rescaled by 2^(127 - bias).
        let sign = 1u32 << (bits - 1);
        let lut: Vec<f32> = (0..1u32 << bits)
            .map(|c| {
                f32::from_bits(((c & (sign - 1)) << (23 - mant)) | ((c & sign) << (32 - bits)))
                    * f32::from_bits((254 - bias) << 23)
            })
            .collect();
        let row = inn * bits as usize / 8;
        return Ok(Some(
            (0..out * inn)
                .map(|i| {
                    let (r, c) = (i / inn, i % inn);
                    let b = &raw[r * row..];
                    let code = if bits == 4 {
                        (b[c / 2] >> (4 * (c % 2))) & 0xF
                    } else {
                        let b = &b[c / 4 * 3..];
                        let word = b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16;
                        ((word >> (6 * (c % 4))) & 0x3F) as u8
                    };
                    let e = scale[r * sc + c / 32];
                    // e8m0: 2^(e - 127); e = 0 is the subnormal 2^-127.
                    let e = f32::from_bits(if e == 0 { 1 << 22 } else { (e as u32) << 23 });
                    lut[code as usize] * e
                })
                .collect(),
        ));
    }

    // Per-group integers: the weight is stored `[in, out]` except int3 (`[out, in]`); scale and zero
    // point share one orientation, `[groups, out]` or `[out, groups]`.
    let reorder = pack == "reorder";
    let transposed = dtype != "int3";
    let (q, _, wc) = codes(
        raw,
        if transposed { inn } else { out },
        &w.dtype,
        dtype,
        reorder,
    )
    .map_err(|e| err(format!("weight: {e}")))?;
    let s_groups_first = sc == out && sr != out;
    let groups = if s_groups_first { sr } else { sc };
    let gs = block(inn, groups).map_err(err)?;
    let scale = decode_dtype(bytes(data, s)?, &s.dtype).map_err(err)?;
    let zero = get("weight_zero_point")
        .map(|z| {
            let rows = z.shape.first().copied().unwrap_or(1);
            codes(bytes(data, z)?, rows, &z.dtype, dtype, reorder)
                .map_err(|e| format!("weight_zero_point: {e}"))
        })
        .transpose()
        .map_err(err)?;
    if zero
        .as_ref()
        .is_some_and(|(_, zr, zc)| *zr != sr || *zc < sc)
    {
        return Err(err(format!(
            "zero points do not match scale grid {:?}",
            s.shape
        )));
    }
    Ok(Some(
        (0..out * inn)
            .map(|i| {
                let (o, k) = (i / inn, i % inn);
                let g = k / gs;
                let (r, c) = if s_groups_first { (g, o) } else { (o, g) };
                let z = zero.as_ref().map_or(0.0, |(z, _, zc)| z[r * zc + c]);
                let v = if transposed {
                    q[k * wc + o]
                } else {
                    q[o * wc + k]
                };
                (v - z) * scale[r * sc + c]
            })
            .collect(),
    ))
}

/// Integer codes of a stored quark tensor (weight or zero point; safetensors dtype `stored`, `rows`
/// rows) and its `[rows, cols]`: int4 / uint4 nibbles of i32 words, int3 triples of bytes; int2 /
/// int8 and float zero points as stored.
fn codes(
    raw: &[u8],
    rows: usize,
    stored: &str,
    dtype: &str,
    reorder: bool,
) -> Result<(Vec<f32>, usize, usize), String> {
    let rows = rows.max(1);
    if raw.len() % rows != 0 {
        return Err(format!("{} bytes for {rows} rows", raw.len()));
    }
    let row = raw.len() / rows;
    let packed = matches!(stored, "I32" | "U8") && matches!(dtype, "int4" | "uint4" | "int3");
    if !packed {
        let v = decode_dtype(raw, stored)?;
        let cols = v.len() / rows;
        return Ok((v, rows, cols));
    }
    // Sign-extends a `bits`-wide code unless the dtype is unsigned.
    let signed = dtype != "uint4";
    let ext = |v: u32, bits: u32| {
        if signed {
            ((v << (32 - bits)) as i32 >> (32 - bits)) as f32
        } else {
            v as f32
        }
    };
    let v = if dtype == "int3" {
        let cols = row / 3 * 8;
        (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let b = &raw[r * row + c / 8 * 3..];
                let word = b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16;
                ext((word >> (3 * (c % 8))) & 7, 3)
            })
            .collect::<Vec<f32>>()
    } else {
        // Nibble holding column j of each 8: inverse of pack's reorder [0, 2, 4, 6, 1, 3, 5, 7].
        const SLOT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
        let cols = row / 4 * 8;
        (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let at = r * row + c / 8 * 4;
                let word = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
                let slot = if reorder { SLOT[c % 8] } else { c % 8 };
                ext((word >> (4 * slot)) & 0xF, 4)
            })
            .collect()
    };
    let cols = v.len() / rows;
    Ok((v, rows, cols))
}

#[cfg(test)]
mod tests {
    use crate::hardware::QuantMode;
    use crate::model::hub::ModelArtifacts;
    use crate::model::UniversalModelLoader;
    use std::path::PathBuf;

    /// Serializes the real-weight checks: each holds a full model in memory.
    static HEAVY: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// `repo`'s range-fetched tensor subset (`disk/quark-subsets/<org>--<name>/`: config.json + the
    /// tested layers), else its full snapshot in disk/: (directory, weights).
    fn local(repo: &str) -> Option<(PathBuf, PathBuf)> {
        let disk = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("disk");
        let flat = repo.replace('/', "--");
        [
            disk.join(format!("quark-subsets/{flat}")),
            disk.join(format!("models--{flat}/snapshots/main")),
        ]
        .into_iter()
        .find_map(|d| {
            ["model.safetensors", "model.safetensors.index.json"]
                .iter()
                .map(|f| d.join(f))
                .find(|p| p.exists())
                .map(|w| (d, w))
        })
    }

    /// Dequantized `repo` vs unquantized `reference` (worst cosine over every quantized weight of
    /// `layers`), then, with `paris` and a full snapshot, a greedy completion through the normal loader.
    fn scheme(repo: &str, reference: &str, layers: &[usize], min_cos: f64, paris: bool) {
        let (Some((q, qw)), Some((_, rw))) = (local(repo), local(reference)) else {
            eprintln!("SKIP: {repo} or {reference} not in disk/");
            return;
        };
        let _guard = HEAVY.lock();
        let cfg = q.join("config.json");
        super::super::check(Some(&cfg), &q).unwrap();
        let (mut qm, qd) = UniversalModelLoader::read_safetensors(&qw).unwrap();
        super::annotate(&mut qm, Some(&cfg)).unwrap();
        let (rm, rd) = UniversalModelLoader::read_safetensors(&rw).unwrap();
        let mut names: Vec<String> = qm
            .tensors
            .keys()
            .filter_map(|n| n.strip_suffix(".weight_scale"))
            .filter(|n| layers.iter().any(|l| n.contains(&format!(".layers.{l}."))))
            .map(|n| format!("{n}.weight"))
            .collect();
        names.sort();
        assert!(
            !names.is_empty(),
            "{repo}: no quantized weights in layers {layers:?}"
        );
        let mut worst = 1.0f64;
        for name in names {
            let a = qm.extract_tensor_f32(&qd, &name).unwrap();
            let b = rm.extract_tensor_f32(&rd, &name).unwrap();
            assert_eq!(a.len(), b.len(), "{name}");
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(&b) {
                let (x, y) = (x as f64, y as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            println!("{repo} {name}: cos {cos:.5}");
            worst = worst.min(cos);
        }
        println!("{repo}: worst cos {worst:.5}");
        assert!(worst >= min_cos, "{repo}: cosine {worst} < {min_cos}");
        drop((qm, qd, rm, rd));
        if !paris || q.ends_with(format!("quark-subsets/{}", repo.replace('/', "--"))) {
            return;
        }
        let artifacts = ModelArtifacts {
            model_id: repo.into(),
            config_path: Some(cfg),
            tokenizer_path: Some(q.join("tokenizer.json")),
            weights_path: Some(qw),
            ..Default::default()
        };
        let prompt = "The capital of France is";
        // Native architectures (qwen3, ...) go through `load_native`, the rest through GraphLm.
        let out = match crate::model::native::load_native(&artifacts, crate::hardware::Device::Cpu)
        {
            Some(lm) => {
                let lm = lm.unwrap();
                let (ids, mut st) = lm.prepare(lm.tokenizer().encode(prompt), &[]).unwrap();
                let mut logits = lm.forward(&ids, 0, &mut st).unwrap();
                let mut out = Vec::new();
                for _ in 0..8 {
                    let t = (0..logits.len())
                        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                        .unwrap() as u32;
                    out.push(t);
                    logits = lm
                        .forward(&[t], ids.len() + out.len() - 1, &mut st)
                        .unwrap();
                }
                lm.tokenizer().decode(&out)
            }
            None => crate::model::hf_loader::load_causal_lm(&artifacts, QuantMode::Uncompressed)
                .unwrap()
                .generate_text(prompt, 8, 0.0),
        };
        println!("{repo}: {out:?}");
        assert!(out.contains("Paris"), "{repo}: {out:?}");
    }

    #[test]
    fn quark_mxfp4() {
        scheme(
            "matmelis/Llama_3.2_1B_w_mxfp4_a_mxfp4_gptq",
            "unsloth/Llama-3.2-1B",
            &[0, 15],
            0.98,
            true,
        );
    }

    /// Native bf16 decoder (qwen3) over dequantized weights ([`super::super::open_weights`]).
    #[test]
    fn quark_mxfp4_qwen3_native() {
        scheme(
            "amd-quark/Qwen3-1.7B-MXFP4",
            "Qwen/Qwen3-1.7B",
            &[0, 27],
            0.98,
            true,
        );
    }

    #[test]
    fn quark_uint4_per_group() {
        scheme(
            "matmelis/Llama_3.2_1B_w_uint4_gptq",
            "unsloth/Llama-3.2-1B",
            &[0, 15],
            0.98,
            true,
        );
    }

    #[test]
    fn quark_int4_signed_asym_per_group() {
        scheme(
            "matmelis/Llama_3.2_3B_w_uint4_gptq",
            "unsloth/Llama-3.2-3B",
            &[0, 27],
            0.98,
            false,
        );
    }

    #[test]
    fn quark_int8_per_tensor() {
        scheme(
            "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test",
            "unsloth/Llama-3.1-8B-Instruct",
            &[0, 31],
            0.99,
            false,
        );
    }

    #[test]
    fn quark_fp8_per_tensor() {
        scheme(
            "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test",
            "unsloth/Llama-3.1-8B-Instruct",
            &[0, 31],
            0.999,
            false,
        );
    }

    #[test]
    fn quark_mxfp6_e3m2() {
        scheme(
            "fxmarty/qwen1.5_moe_a2.7b_chat_w_fp6_e3m2_a_fp6_e3m2",
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            &[0, 23],
            0.998,
            false,
        );
    }

    #[test]
    fn quark_mxfp6_e2m3() {
        scheme(
            "fxmarty/qwen1.5_moe_a2.7b_chat_w_fp6_e2m3_a_fp6_e2m3",
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            &[0, 23],
            0.999,
            false,
        );
    }

    #[test]
    fn quark_int3_per_group() {
        scheme(
            "matmelis/Llama_3.2_1B_w_int3_gptq",
            "unsloth/Llama-3.2-1B",
            &[0, 15],
            0.93,
            false,
        );
    }

    #[test]
    fn quark_int2_per_group() {
        scheme(
            "matmelis/Llama_3.2_1B_w_int2_gptq",
            "unsloth/Llama-3.2-1B",
            &[0, 15],
            0.75,
            false,
        );
    }

    #[test]
    fn quark_fp8_per_channel() {
        scheme(
            "amd/Qwen2.5-1.5B-Instruct-ptpc-Quark-ts",
            "Qwen/Qwen2.5-1.5B-Instruct",
            &[0, 27],
            0.999,
            false,
        );
    }

    /// vLLM `fp8` serialization from AMD Quark (per-tensor F8 + scalar `weight_scale`, fp8 KV scales).
    #[test]
    fn fp8_method_per_tensor() {
        scheme(
            "amd/Llama-3.2-1B-Instruct-FP8-KV",
            "unsloth/Llama-3.2-1B-Instruct",
            &[0, 15],
            0.0,
            true,
        );
    }

    /// Sub-schemes we don't expand are refused from config.json alone, with the reason.
    #[test]
    fn quark_refusals() {
        let blocker = |qc: serde_json::Value| {
            super::super::scheme_blocker(&serde_json::json!({ "quantization_config": qc }), None)
        };
        let w = |w: serde_json::Value| {
            serde_json::json!({"quant_method": "quark", "export": {"pack_method": "reorder"},
                "global_quant_config": {"weight": w}, "layer_quant_config": {}})
        };
        let mxfp4 = serde_json::json!({"dtype": "fp4", "qscheme": "per_group", "group_size": 32,
            "ch_axis": -1, "scale_format": "e8m0"});
        let uint4 = serde_json::json!({"dtype": "uint4", "qscheme": "per_group", "group_size": 128,
            "ch_axis": 1, "symmetric": false});
        assert_eq!(blocker(w(mxfp4.clone())), None);
        assert_eq!(blocker(w(uint4.clone())), None);
        // The hub API trims quantization_config to quant_method: the picker lets quark rows through.
        assert_eq!(blocker(serde_json::json!({"quant_method": "quark"})), None);
        // amd-quark/Qwen3-30B-A3B-nvfp4-quark, amd/Kimi-K2.5-W4A8.
        assert_eq!(
            blocker(w(serde_json::json!([mxfp4, uint4]))).unwrap(),
            "quark multi-stage weight quantization (e.g. NVFP4 [fp4, fp8], W4A8 [fp8, int4]) is not supported"
        );
        let mut nvfp4 = w(
            serde_json::json!({"dtype": "fp4", "qscheme": "per_group", "group_size": 16,
            "scale_format": "float32"}),
        );
        assert_eq!(
            blocker(nvfp4.clone()).unwrap(),
            "quark fp4 weights are supported with e8m0 scales in groups of 32 (OCP MX) only"
        );
        nvfp4["global_quant_config"]["weight"]["dtype"] = "mx".into();
        assert!(blocker(nvfp4)
            .unwrap()
            .starts_with("quark weight mx per_group is not supported"));
        // amd-quark/internal-testing-qwen3_0.6b-int8-hadamard.
        let mut rotated = w(uint4.clone());
        rotated["algo_config"] = serde_json::json!([{"name": "rotation", "r1": true,
            "online_r1_rotation": true, "online_config": {"online_rotation_layers": ["model.layers.0.mlp.up_proj"]}}]);
        assert_eq!(
            blocker(rotated).unwrap(),
            "quark rotation with online rotations (runtime Hadamard on activations) is not supported"
        );
        let mut packed = w(uint4);
        packed["export"]["pack_method"] = "pack".into();
        assert_eq!(
            blocker(packed).unwrap(),
            "quark pack_method 'pack' is not supported (reorder, order are)"
        );
        assert_eq!(
            blocker(serde_json::json!({"quant_method": "fp8", "weight_block_size": [128, 128]})),
            None
        );
        assert_eq!(
            blocker(serde_json::json!({"quant_method": "fp8", "weight_block_size": [64, 64]}))
                .unwrap(),
            "fp8 weight_block_size [64,64] is not supported ([128, 128] is)"
        );
    }

    #[test]
    fn layer_patterns_resolve_like_quark() {
        let qc = serde_json::json!({"exclude": ["lm_head", "*.mlp.gate"],
            "global_quant_config": {"weight": {"dtype": "fp4"}},
            "layer_quant_config": {"*self_attn*": {"weight": {"dtype": "fp8_e4m3"}},
                "*k_proj": {"weight": {"dtype": "fp8_e4m3"}}, "*v_pro?": {"weight": {"dtype": "int8"}}},
            "layer_type_quant_config": {}});
        let dtype = |m: &str| super::weight_spec(&qc, m).map(|w| w.map(|w| w["dtype"].clone()));
        assert_eq!(dtype("model.layers.0.mlp.up_proj"), Ok(Some("fp4".into())));
        assert_eq!(
            dtype("model.layers.0.self_attn.k_proj"),
            Ok(Some("fp8_e4m3".into()))
        );
        assert_eq!(dtype("lm_head"), Ok(None));
        assert_eq!(dtype("model.layers.3.mlp.gate"), Ok(None));
        assert!(dtype("model.layers.0.self_attn.v_proj").is_err());
    }

    /// A quark MX fp4 weight still is not repacked (it decodes through [`dequantize`]) and keeps its
    /// pre-existing NVFP4 tag once the config has been applied.
    #[test]
    fn quark_fp4_stays_out_of_the_mxfp4_repack() {
        use crate::model::{ModelManifest, VirtualTensor};
        let (rows, cols) = (2usize, 64usize);
        let mut m = ModelManifest::default();
        let mut data = Vec::new();
        for (name, shape, len) in [
            ("l.weight", vec![rows, cols / 2], rows * cols / 2),
            ("l.weight_scale", vec![rows, cols / 32], rows * cols / 32),
        ] {
            m.tensors.insert(
                name.into(),
                VirtualTensor {
                    name: name.into(),
                    shape,
                    quant_format: crate::hardware::QuantFormat::FP32,
                    byte_offset: data.len(),
                    byte_len: len,
                    dtype: "U8".into(),
                    shard_index: 0,
                },
            );
            data.extend(std::iter::repeat(0x77u8).take(len));
        }
        super::super::alias_packed(&mut m, &data).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"quantization_config":{"quant_method":"quark","global_quant_config":{"weight":{"dtype":"fp4","qscheme":"per_group","group_size":32,"ch_axis":-1,"scale_format":"e8m0"}}}}"#,
        )
        .unwrap();
        super::annotate(&mut m, Some(&cfg)).unwrap();
        assert_eq!(m.tensors["l.weight"].quant_format, crate::hardware::QuantFormat::NVFP4);
        assert_eq!(m.tensors["l.weight"].shape, [rows, cols]);
        // `repack` reads only MXFP4-tagged weights, so quark's fp4 goes the way it always did.
        assert!(crate::model::repack::repack(&m, &data, "l.weight").unwrap().is_none());
        assert!(super::super::dequantize(&m, &data, "l.weight").unwrap().is_some());
    }
}
