//! Pre-quantized safetensors checkpoints: compressed-tensors (llm-compressor), NVIDIA modelopt,
//! transformers BitNet (`quant_method: "bitnet"`), AMD Quark and vLLM `fp8` ([`quark`]), MLX
//! ([`mlx`]), EXL3 ([`exl3`]) and bitsandbytes 4-bit ([`crate::model::repack::bnb`]).
//!
//! [`check`] rejects schemes we can't decode before any weight is read, [`alias_packed`] exposes packed
//! weights under the `X.weight` names the family specs read, and [`dequantize`] expands one such
//! weight to f32 when it is extracted (one tensor at a time, no full-model copy).

use crate::hardware::QuantFormat;
use crate::model::weights::Weights;
use crate::model::repack::mxfp4;
use crate::model::{decode_dtype, ModelManifest, VirtualTensor};
use crate::quant::nvfp4::NVFP4_E2M1_LUT;
use serde_json::Value;
use std::path::Path;

pub(crate) mod exl3;
pub mod glq;
pub(crate) mod mlx;
mod quark;

/// Config-dependent manifest fix-ups before weights are read: Quark weight specs
/// ([`quark::annotate`]) and GLQ's true shapes and shipped codebook ([`glq::prepare`]).
pub fn annotate(m: &mut ModelManifest, config_path: Option<&Path>) -> Result<(), String> {
    quark::annotate(m, config_path)?;
    crate::model::repack::gptq::annotate(m, config_path);
    glq::prepare(m, config_path)
}

/// compressed-tensors storage formats [`dequantize`] expands.
const CT_FORMATS: [&str; 7] = [
    "dense",
    "naive-quantized",
    "int-quantized",
    "float-quantized",
    "pack-quantized",
    "nvfp4-pack-quantized",
    "mxfp4-pack-quantized",
];

/// Errs when `config.json` (`quantization_config`) or `hf_quant_config.json` beside the weights names a
/// scheme [`dequantize`] can't expand. Unquantized checkpoints pass.
pub fn check(config_path: Option<&Path>, weights_dir: &Path) -> Result<(), String> {
    let read = |p: &Path| -> Result<Option<Value>, String> {
        match std::fs::read(p) {
            Ok(b) => serde_json::from_slice(&b)
                .map(Some)
                .map_err(|e| format!("{}: {e}", p.display())),
            Err(_) => Ok(None),
        }
    };
    let cfg = config_path
        .map(read)
        .transpose()?
        .flatten()
        .unwrap_or(Value::Null);
    let hf_quant = read(&weights_dir.join("hf_quant_config.json"))?;
    if let Some(why) = scheme_blocker(&cfg, hf_quant.as_ref()) {
        return Err(why);
    }
    // The hub API trims `quantization_config` to `quant_method`, so the picker lets a `bitnet` row
    // without `linear_class` through; the full config must name it (absent = divide by the scale).
    let qc = &cfg["quantization_config"];
    if qc["quant_method"] == "bitnet" && qc["linear_class"].is_null() {
        return Err(
            "bitnet linear_class 'bitlinear' (the default) is not supported (autobitlinear is)"
                .into(),
        );
    }
    Ok(())
}

/// Why a checkpoint's quantization scheme can't load, from its `config.json` value and, when read,
/// `hf_quant_config.json`. Shared by the loader ([`check`]) and the hub picker (config only), so a
/// row the picker shows as loadable never fails this check at load.
pub fn scheme_blocker(cfg: &Value, hf_quant: Option<&Value>) -> Option<String> {
    if let Some(q) = mlx::block(cfg) {
        return mlx::blocker(q);
    }
    let qc = cfg
        .get("quantization_config")
        .or_else(|| cfg.get("compression_config"));
    let method = qc
        .and_then(|q| q["quant_method"].as_str())
        .or(hf_quant.map(|_| "modelopt"));
    match method {
        None => None,
        Some("compressed-tensors") => {
            let q = qc.unwrap_or(&Value::Null);
            if let Some(f) = q["sparsity_config"]["format"].as_str().filter(|f| *f != "dense") {
                return Some(format!("compressed-tensors sparse format '{f}' is not supported"));
            }
            if q["transform_config"].as_object().is_some_and(|t| !t.is_empty()) {
                return Some(
                    "compressed-tensors `transform_config` (rotated weights) is not supported".into(),
                );
            }
            // `mixed-precision` (llm-compressor's per-group schemes): every group names its own
            // storage format, and [`dequantize`] reads each tensor by its dtype and companions.
            let groups = q["config_groups"].as_object().into_iter().flat_map(|g| g.values());
            std::iter::once(&q["format"])
                .filter(|f| *f != "mixed-precision")
                .chain(groups.map(|g| &g["format"]))
                .find_map(|f| f.as_str().filter(|f| !CT_FORMATS.contains(f)))
                .map(|f| {
                    format!(
                        "compressed-tensors format '{f}' is not supported (supported: {})",
                        CT_FORMATS.join(", ")
                    )
                })
        }
        Some("modelopt") => {
            let algo = qc
                .and_then(|q| q["quant_algo"].as_str())
                .or_else(|| hf_quant.and_then(|h| h["quantization"]["quant_algo"].as_str()));
            match algo {
                Some(a) if a.starts_with("FP8") || a == "NVFP4" => None,
                // The picker sees only config.json; the algorithm may sit in hf_quant_config.json.
                None if hf_quant.is_none() => None,
                a => Some(format!(
                    "modelopt quant_algo '{}' is not supported (FP8*, NVFP4 are)",
                    a.unwrap_or("")
                )),
            }
        }
        // ponytail: transformers' default `bitlinear` class divides by `weight_scale` (HF1BitLLM
        // Llama3-8B-1.58); only `autobitlinear` (multiplies) decodes. Add the divide in
        // `dequantize` once a loadable-size bitlinear checkpoint can verify it.
        Some("bitnet") => {
            let q = qc.unwrap_or(&Value::Null);
            let class = q["linear_class"]
                .as_str()
                .or((cfg["model_type"] == "bitnet").then_some("autobitlinear"));
            if class != Some("autobitlinear") {
                Some(format!(
                    "bitnet linear_class '{}' is not supported (autobitlinear is)",
                    class.unwrap_or("bitlinear")
                ))
            } else if q["use_rms_norm"] == true {
                Some("bitnet `use_rms_norm` (RMSNorm inside BitLinear) is not supported".into())
            } else if q["modules_to_not_convert"]
                .as_array()
                .is_some_and(|m| !m.is_empty())
            {
                Some("bitnet `modules_to_not_convert` is not supported".into())
            } else {
                None
            }
        }
        Some("quark") => quark::blocker(qc.unwrap_or(&Value::Null)),
        // vLLM `fp8` serialization (amd/Llama-3.2-1B-Instruct-FP8-KV): F8 weights, one `weight_scale` each.
        // `weight_block_size` (DeepSeek / Qwen3-FP8): 2-D `weight_scale_inv` grid, read by [`dequantize`],
        // whose grid inference ([`block`]) only knows 128-blocks.
        Some("fp8") => qc.and_then(|q| {
            let b = &q["weight_block_size"];
            (!b.is_null() && *b != serde_json::json!([128, 128])).then(|| {
                format!("fp8 weight_block_size {b} is not supported ([128, 128] is)")
            })
        }),
        // Every linear decodes at load (`exl3`); quantized embedding tables are refused there.
        Some("exl3") => None,
        Some("glq") => {
            let q = qc.unwrap_or(&Value::Null);
            match q["codebook"].as_str() {
                None | Some("e8_shell" | "e8p") => None,
                Some("trellis") if q["variant"] != "3inst" => Some(format!(
                    "GLQ trellis variant '{}' is not supported (3inst is)",
                    q["variant"].as_str().unwrap_or("hyb")
                )),
                // Pre-kernel natural-layout 3inst checkpoints are refused by the reference too.
                Some("trellis") if q["trellis_layout"] != "kernel" => Some(
                    "GLQ 3inst trellis in the natural layout (no `trellis_layout: kernel`) is not supported".into(),
                ),
                Some("trellis") => None,
                Some(c) => Some(format!(
                    "GLQ codebook '{c}' is not supported (e8_shell, e8p, trellis 3inst are)"
                )),
            }
        }
        Some("bitsandbytes") => crate::model::repack::bnb::blocker(qc.unwrap_or(&Value::Null)),
        Some(m @ ("gptq" | "awq")) => crate::model::repack::gptq::blocker(m, qc.unwrap_or(&Value::Null)),
        Some(m) => Some(format!(
            "quantization method '{m}' is not supported (compressed-tensors, modelopt FP8/NVFP4, gptq and awq 4-bit, bitnet, quark, fp8, exl3, glq and bitsandbytes 4-bit are)"
        )),
    }
}

/// A safetensors checkpoint as [`Weights`] for the bf16 native models: memory-mapped when
/// `config.json` names no `quantization_config` or a scheme [`Weights`] decodes per tensor itself
/// (MLX, EXL3), else every tensor expanded through [`dequantize`] and held as bf16. With `repack`,
/// weights [`crate::model::repack`] turns into a GPU codec layout stay packed
/// ([`Weights::packed_gguf`]) instead.
pub fn open_weights(
    weights: &Path,
    config_path: Option<&Path>,
    repack: bool,
) -> Result<Weights, String> {
    let cfg = config_path
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .unwrap_or_default();
    let qc = &cfg["quantization_config"];
    // MLX, EXL3 and GLQ: [`Weights::open`] decodes those tensors to bf16 one at a time. BitNet
    // `online` checkpoints hold plain master weights (ternarized by the model at load).
    let self_decoding = !cfg["quantization"].is_null()
        || qc["quant_method"] == "exl3"
        || qc["quant_method"] == "glq"
        || (qc["quant_method"] == "bitnet" && qc["quantization_mode"] == "online");
    if qc.is_null() || self_decoding {
        return Weights::open(weights);
    }
    let (mut m, data) = crate::model::UniversalModelLoader::read_safetensors(weights)?;
    annotate(&mut m, config_path)?;
    let mut packed = Vec::new();
    let mut specs = Vec::new();
    for (n, t) in &m.tensors {
        if crate::model::repack::bnb::is_companion(n)
            || crate::model::repack::gptq::is_companion(&m, n)
        {
            continue;
        }
        match repack
            .then(|| crate::model::repack::repack(&m, &data, n))
            .transpose()?
            .flatten()
        {
            Some(r) => packed.push((n.clone(), r)),
            None => specs.push((n.clone(), t.shape.clone())),
        }
    }
    let mut w = Weights::from_f32(specs, |n| m.extract_tensor_f32(&data, n))?;
    for (n, r) in packed {
        let p = crate::model::weights::Packed {
            map: std::sync::Arc::new(crate::model::loaders::gguf::GgufData::Buf(r.bytes)),
            off: 0,
            format: r.format,
            shape: vec![r.rows, r.cols],
        };
        w.insert_packed(n, p);
    }
    Ok(w)
}

/// Exposes compressed-tensors `X.weight_packed` as `X.weight` with its logical `[out, in]` shape, and
/// gives modelopt NVFP4 `X.weight` (U8 nibble pairs) and BitNet `X.weight` (U8 2-bit quads beside a
/// one-element float `weight_scale`) their unpacked shapes. Run once on a complete manifest.
pub fn alias_packed(m: &mut ModelManifest, data: &[u8]) -> Result<(), String> {
    glq::alias(m);
    crate::model::repack::bnb::alias(m, data)?;
    crate::model::repack::gptq::alias(m);
    crate::model::repack::mxfp4::alias(m);
    let mut add = Vec::new();
    let mut fp4 = Vec::new();
    let mut ternary = Vec::new();
    for (name, t) in &m.tensors {
        if let Some(stem) = name.strip_suffix(".weight_packed") {
            let shape = match m.tensors.get(&format!("{stem}.weight_shape")) {
                Some(s) => decode_dtype(bytes(data, s)?, &s.dtype)?
                    .iter()
                    .map(|&v| v as usize)
                    .collect(),
                None if t.dtype == "U8" && t.shape.len() == 2 => vec![t.shape[0], t.shape[1] * 2],
                None => return Err(format!("{name} has no `weight_shape` companion")),
            };
            // ponytail: int4/int8 packs share the GPTQ_W4A16 tag; QuantFormat has no plain INT8 variant.
            let quant_format = if t.dtype != "U8" {
                QuantFormat::GPTQ_W4A16
            } else if matches!(shape[..], [r, c] if mxfp4::is_mx(m, stem, r, c)) {
                QuantFormat::MXFP4
            } else {
                QuantFormat::NVFP4
            };
            add.push(VirtualTensor {
                name: format!("{stem}.weight"),
                shape,
                quant_format,
                ..t.clone()
            });
        } else if let Some(stem) = name.strip_suffix(".trellis") {
            // EXL3: trellis [in/16, out/16, 16·K] stands for the [out, in] weight.
            let [kt, nt, _] = t.shape[..] else {
                return Err(format!(
                    "{name}: EXL3 trellis-quantized embedding / n-gram tables are not supported"
                ));
            };
            add.push(VirtualTensor {
                name: format!("{stem}.weight"),
                shape: vec![nt * 16, kt * 16],
                ..t.clone()
            });
        } else if let Some((stem, scale)) = name
            .strip_suffix(".weight")
            .and_then(|s| Some((s, m.tensors.get(&format!("{s}.weight_scale"))?)))
            .filter(|_| t.dtype == "U8" && t.shape.len() == 2)
        {
            if scale.dtype != "U8" && scale.shape.iter().product::<usize>() == 1 {
                ternary.push(name.clone());
            } else {
                let mx = mxfp4::is_mx(m, stem, t.shape[0], t.shape[1] * 2);
                fp4.push((name.clone(), if mx { QuantFormat::MXFP4 } else { QuantFormat::NVFP4 }));
            }
        }
    }
    for t in add {
        m.tensors.insert(t.name.clone(), t);
    }
    for (name, quant_format) in fp4 {
        let t = m.tensors.get_mut(&name).unwrap();
        t.shape[1] *= 2;
        t.quant_format = quant_format;
    }
    for name in ternary {
        let t = m.tensors.get_mut(&name).unwrap();
        t.shape[0] *= 4;
        t.quant_format = QuantFormat::BitNet_B1_58;
    }
    Ok(())
}

/// Expands pre-quantized weight `name` to row-major f32 `[out, in]`; `None` for ordinary tensors.
pub fn dequantize(m: &ModelManifest, data: &[u8], name: &str) -> Result<Option<Vec<f32>>, String> {
    if let Some(w) = glq::dequantize(m, data, name)? {
        return Ok(Some(w));
    }
    if let Some(w) = mlx::dequantize(m, data, name)? {
        return Ok(Some(w));
    }
    if let Some(w) = crate::model::repack::bnb::dequantize(m, data, name)? {
        return Ok(Some(w));
    }
    if let Some(w) = crate::model::repack::gptq::dequantize(m, data, name)? {
        return Ok(Some(w));
    }
    let Some(stem) = name.strip_suffix(".weight") else {
        // gpt-oss `X_blocks` / `X_scales` are exposed as a bare `X`.
        return mxfp4::dequantize(m, data, name);
    };
    if let Some(w) = quark::dequantize(m, data, name)? {
        return Ok(Some(w));
    }
    if let Some(w) = mxfp4::dequantize(m, data, name)? {
        return Ok(Some(w));
    }
    if m.tensors.contains_key(&format!("{stem}.trellis"))
        && m.tensors.get(name).is_some_and(|w| w.dtype == "I16")
    {
        let get = |n: &str| {
            let t = m.tensors.get(n)?;
            Some((t, bytes(data, t).ok()?))
        };
        return exl3::dequantize(&get, stem).map(Some);
    }
    let get = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    // fp8 block scales (`weight_scale_inv`) multiply like `weight_scale`: w = w_fp8 · scale_inv per block.
    let (Some(w), Some(scale_t)) = (
        m.tensors.get(name),
        get("weight_scale").or_else(|| get("weight_scale_inv")),
    ) else {
        return Ok(None);
    };
    // Float weights beside calibration scales (frozen / fake-quant exports) are already real values.
    if !matches!(
        w.dtype.as_str(),
        "I8" | "I32" | "U8" | "F8_E4M3" | "F8_E5M2"
    ) || w.shape.len() != 2
    {
        return Ok(None);
    }
    let err = |e: String| format!("{name}: {e}");
    let f32s = |t: &VirtualTensor| decode_dtype(bytes(data, t)?, &t.dtype).map_err(err);
    let (rows, cols) = (w.shape[0], w.shape[1]);
    let raw = bytes(data, w)?;

    // BitNet b1.58 (transformers `pack_weights`): four 2-bit `q + 1` codes per byte along `out`; code
    // i of packed row r is row i·rows/4 + r. Offline `autobitlinear`: weight = q · weight_scale.
    if w.quant_format == QuantFormat::BitNet_B1_58 {
        let packed_rows = rows / 4;
        if raw.len() != packed_rows * cols {
            return Err(err(format!(
                "{} bytes for 2-bit [{rows}, {cols}]",
                raw.len()
            )));
        }
        let s = f32s(scale_t)?[0];
        return Ok(Some(
            (0..rows * cols)
                .map(|i| {
                    let (r, c) = (i / cols, i % cols);
                    let code = raw[(r % packed_rows) * cols + c] >> (2 * (r / packed_rows)) & 3;
                    (code as f32 - 1.0) * s
                })
                .collect(),
        ));
    }

    let mut bits = 0;
    let mut out = match w.dtype.as_str() {
        // compressed-tensors pack-quantized int4/int8 (I32 words along `in`).
        "I32" => {
            if !m.tensors.contains_key(&format!("{stem}.weight_packed")) {
                return Err(err(
                    "packed i32 weight without compressed-tensors `weight_packed` (a quark export read without its config.json?)".into(),
                ));
            }
            let packed_cols = raw.len() / 4 / rows.max(1);
            bits = [4, 8, 2]
                .into_iter()
                .find(|b| cols.div_ceil(32 / b) == packed_cols)
                .ok_or_else(|| {
                    err(format!(
                        "packed width {packed_cols} fits no int bit-width for {cols} columns"
                    ))
                })?;
            unpack_int(raw, rows, cols, bits, false).map_err(err)?
        }
        // NVFP4 E2M1 nibble pairs, low nibble = even column (compressed-tensors and modelopt).
        "U8" => {
            if raw.len() != rows * cols.div_ceil(2) {
                return Err(err(format!("{} bytes for fp4 [{rows}, {cols}]", raw.len())));
            }
            let stride = cols.div_ceil(2);
            (0..rows * cols)
                .map(|i| {
                    let (r, c) = (i / cols, i % cols);
                    let b = raw[r * stride + c / 2];
                    NVFP4_E2M1_LUT[(if c % 2 == 0 { b & 0x0F } else { b >> 4 }) as usize]
                })
                .collect()
        }
        // int-quantized / naive-quantized I8, float-quantized / modelopt FP8.
        _ => decode_dtype(raw, &w.dtype).map_err(err)?,
    };
    if out.len() != rows * cols {
        return Err(err(format!(
            "{} values for shape [{rows}, {cols}]",
            out.len()
        )));
    }

    // modelopt stores NVFP4 block scales as raw E4M3 bytes in a U8 tensor.
    let scale = if w.dtype == "U8" && scale_t.dtype == "U8" {
        decode_dtype(bytes(data, scale_t)?, "F8_E4M3")?
    } else {
        f32s(scale_t)?
    };
    let (sr, sc) = match scale_t.shape[..] {
        [] => (1, 1),
        [n] => (n, 1),
        [a, b] => (a, b),
        _ => return Err(err(format!("scale shape {:?}", scale_t.shape))),
    };
    if scale.len() != sr * sc {
        return Err(err(format!("{} scales for grid [{sr}, {sc}]", scale.len())));
    }
    let scalar = |t: &VirtualTensor| {
        f32s(t)?
            .first()
            .copied()
            .ok_or_else(|| err(format!("{} is empty", t.name)))
    };
    let global = match (get("weight_global_scale"), get("weight_scale_2")) {
        (Some(g), _) => 1.0 / scalar(g)?, // compressed-tensors nvfp4: fp4 * scale / global_scale
        (None, Some(g)) => scalar(g)?,    // modelopt nvfp4: fp4 * scale * scale_2
        (None, None) if w.dtype == "U8" => {
            return Err(err(
                "fp4 weight without a global scale or an MXFP4 scale grid is not supported".into(),
            ))
        }
        (None, None) => 1.0,
    };
    let zero = match get("weight_zero_point") {
        // pack-quantized asymmetric: zero points packed along `out` with the weight's bit-width.
        Some(z) if z.dtype == "I32" && bits > 0 => {
            Some(unpack_int(bytes(data, z)?, sr, sc, bits, true).map_err(err)?)
        }
        Some(z) => Some(f32s(z)?),
        None => None,
    };
    if zero.as_ref().is_some_and(|z| z.len() != sr * sc) {
        return Err(err(format!(
            "zero points do not match scale grid [{sr}, {sc}]"
        )));
    }
    // Activation-ordered groups map each input column to its scale column.
    let g_idx = get("weight_g_idx").map(f32s).transpose()?;
    if g_idx
        .as_ref()
        .is_some_and(|g| g.len() != cols || g.iter().any(|&v| v < 0.0 || v as usize >= sc))
    {
        return Err(err("weight_g_idx does not match the scale grid".into()));
    }
    let (rb, cb) = (block(rows, sr).map_err(err)?, block(cols, sc).map_err(err)?);
    for (i, v) in out.iter_mut().enumerate() {
        let (r, c) = (i / cols, i % cols);
        let s = (r / rb) * sc + g_idx.as_ref().map_or(c / cb, |g| g[c] as usize);
        *v = (*v - zero.as_ref().map_or(0.0, |z| z[s])) * scale[s] * global;
    }
    Ok(Some(out))
}

/// A compressed-tensors `pack-quantized` int4 weight with one scale per `group` input columns of a row,
/// as [`dequantize`] reads it: element = `(nibble - zero) * scale`. `w` is the packed words' bytes,
/// i.e. one unsigned nibble per element (`q + 8`), element `2i` in the low nibble of byte `i`;
/// `scale` / `zero` are `[rows, cols / group]` row-major, `zero` in nibble units (`8` = symmetric).
pub(crate) struct Int4Groups<'a> {
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub w: &'a [u8],
    pub scale: Vec<f32>,
    pub zero: Vec<f32>,
}

/// [`Int4Groups`] of `name`; `None` when it is not such a weight: not int4, no group grid (per-tensor
/// scale, a group size outside 32 / 64 / 128, a scale row count other than `out`), activation-ordered
/// (`weight_g_idx`), a global scale (`weight_global_scale` / `weight_scale_2`), or zero points that are not whole numbers.
pub(crate) fn int4_groups<'a>(
    m: &ModelManifest,
    data: &'a [u8],
    name: &str,
) -> Result<Option<Int4Groups<'a>>, String> {
    let Some(stem) = name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let get = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    let (Some(w), Some(st), Some(_)) = (m.tensors.get(name), get("weight_scale"), get("weight_packed"))
    else {
        return Ok(None);
    };
    let [rows, cols] = w.shape[..] else {
        return Ok(None);
    };
    let [sr, sc] = st.shape[..] else {
        return Ok(None);
    };
    // 4 bits: half a byte per element (8-bit packs are `rows * cols`, 2-bit a quarter).
    if w.dtype != "I32" || get("weight_g_idx").is_some()
        || get("weight_global_scale").is_some()
        || get("weight_scale_2").is_some()
        || w.byte_len * 2 != rows * cols {
        return Ok(None);
    }
    if sr != rows || sc == 0 || cols % sc != 0 || ![32, 64, 128].contains(&(cols / sc)) {
        return Ok(None);
    }
    let err = |e: String| format!("{name}: {e}");
    let scale = decode_dtype(bytes(data, st)?, &st.dtype).map_err(err)?;
    let zero = match get("weight_zero_point") {
        None => vec![8.0; sr * sc],
        // Packed along `out` like the weight; stored signed (`z + 8` unsigned).
        Some(z) if z.dtype == "I32" => unpack_int(bytes(data, z)?, sr, sc, 4, true)
            .map_err(err)?
            .into_iter()
            .map(|z| z + 8.0)
            .collect(),
        Some(z) => decode_dtype(bytes(data, z)?, &z.dtype)
            .map_err(err)?
            .into_iter()
            .map(|z| z + 8.0)
            .collect(),
    };
    // `(q - z) * s` and `(nibble - (z + 8)) * s` only agree while `z` is a whole number.
    if scale.len() != sr * sc || zero.len() != sr * sc || zero.iter().any(|z| z.fract() != 0.0) {
        return Ok(None);
    }
    Ok(Some(Int4Groups {
        rows,
        cols,
        group: cols / sc,
        w: bytes(data, w)?,
        scale,
        zero,
    }))
}

/// Elements of `n` covered by each of `s` scale entries.
// ponytail: inferred from the scale grid (128-blocks preferred, else exact division); ragged non-128
// blocks would need the config's `block_structure`.
pub(crate) fn block(n: usize, s: usize) -> Result<usize, String> {
    if s == n {
        Ok(1)
    } else if n.div_ceil(128) == s {
        Ok(128)
    } else if s > 0 && n % s == 0 {
        Ok(n / s)
    } else {
        Err(format!("scale grid of {s} does not tile {n} elements"))
    }
}

/// compressed-tensors `pack_to_int32`: `32/bits` unsigned values (`q + 2^(bits-1)`) per little-endian
/// i32, lowest bits first, packed along columns (weights) or rows (`along_rows`, zero points).
/// Returns the signed `q` values of the logical `[rows, cols]` matrix.
fn unpack_int(
    raw: &[u8],
    rows: usize,
    cols: usize,
    bits: usize,
    along_rows: bool,
) -> Result<Vec<f32>, String> {
    let per = 32 / bits;
    let (pr, pc) = if along_rows {
        (rows.div_ceil(per), cols)
    } else {
        (rows, cols.div_ceil(per))
    };
    if raw.len() != pr * pc * 4 {
        return Err(format!(
            "{} packed bytes, expected [{pr}, {pc}] i32",
            raw.len()
        ));
    }
    let word =
        |r: usize, c: usize| u32::from_le_bytes(raw[(r * pc + c) * 4..][..4].try_into().unwrap());
    let (mask, half) = ((1u32 << bits) - 1, 1i32 << (bits - 1));
    Ok((0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            let (v, k) = if along_rows {
                (word(r / per, c), r % per)
            } else {
                (word(r, c / per), c % per)
            };
            (((v >> (k * bits)) & mask) as i32 - half) as f32
        })
        .collect())
}

pub(crate) fn bytes<'a>(data: &'a [u8], t: &VirtualTensor) -> Result<&'a [u8], String> {
    data.get(t.byte_offset..t.byte_offset + t.byte_len)
        .ok_or_else(|| format!("tensor '{}' lies outside the weight buffer", t.name))
}

#[cfg(test)]
mod tests {
    use crate::model::hub::ModelArtifacts;
    use crate::model::loaders::{SafetensorsWeightReader, WeightReader};
    use std::path::PathBuf;

    /// BitNet `online` checkpoints are plain masters: mapped as stored, not dequantized to bf16.
    #[test]
    fn online_bitnet_masters_keep_their_dtype() {
        let d = tempfile::tempdir().unwrap();
        let v: Vec<u8> = [0.1f32, -0.2, 0.3, 0.4].iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut h = serde_json::to_vec(&serde_json::json!(
            {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}})).unwrap();
        h.resize(h.len().next_multiple_of(8), b' ');
        let mut st = (h.len() as u64).to_le_bytes().to_vec();
        st.extend(h);
        st.extend(v);
        let (p, c) = (d.path().join("model.safetensors"), d.path().join("config.json"));
        std::fs::write(&p, st).unwrap();
        std::fs::write(&c, serde_json::json!({"model_type": "bitnet", "quantization_config":
            {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}}).to_string()).unwrap();
        let w = super::open_weights(&p, Some(&c), false).unwrap();
        assert_eq!(w.stored_dtype("w"), Some("F32"));
    }

    /// Serializes the real-weight checks: each holds a full f32 model in memory.
    pub(super) static HEAVY: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    pub(super) fn snapshot(repo: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "disk/models--{}/snapshots/main",
            repo.replace('/', "--")
        ))
    }

    const PROJS: &[&str] = &[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ];

    /// Dequantized weights of `repo` vs the unquantized `reference` (min cosine over sampled projections),
    /// then a greedy completion through the normal loader.
    fn scheme(repo: &str, reference: &str, layers: [usize; 2], projs: &[&str], min_cos: f64) {
        let (q, r) = (snapshot(repo), snapshot(reference));
        if !q.join("model.safetensors").exists() || !r.join("model.safetensors").exists() {
            println!("SKIP: {repo} or {reference} not in disk/");
            return;
        }
        let _guard = HEAVY.lock();
        super::check(Some(&q.join("config.json")), &q).unwrap();
        let (qr, rr) = (
            SafetensorsWeightReader::new(q.join("model.safetensors")).unwrap(),
            SafetensorsWeightReader::new(r.join("model.safetensors")).unwrap(),
        );
        let mut worst = (1.0f64, 0.0f64);
        for l in layers {
            for p in projs {
                let name = format!("model.layers.{l}.{p}.weight");
                let (a, b) = (
                    qr.load_tensor(&name).unwrap(),
                    rr.load_tensor(&name).unwrap(),
                );
                assert_eq!(a.len(), b.len(), "{name}");
                let (mut dot, mut na, mut nb, mut err) = (0f64, 0f64, 0f64, 0f64);
                for (&x, &y) in a.iter().zip(&b) {
                    let (x, y) = (x as f64, y as f64);
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                    err += (x - y) * (x - y);
                }
                let (cos, rel) = (dot / (na.sqrt() * nb.sqrt()), (err / nb).sqrt());
                println!("{repo} {name}: cos {cos:.5} rel_err {rel:.4}");
                worst = (worst.0.min(cos), worst.1.max(rel));
            }
        }
        println!(
            "{repo}: worst cos {:.5} worst rel_err {:.4}",
            worst.0, worst.1
        );
        assert!(worst.0 >= min_cos, "{repo}: cosine {} < {min_cos}", worst.0);

        let artifacts = ModelArtifacts {
            model_id: repo.into(),
            config_path: Some(q.join("config.json")),
            tokenizer_path: Some(q.join("tokenizer.json")),
            weights_path: Some(q.join("model.safetensors")),
            ..Default::default()
        };
        let out = crate::arch::testing::greedy_text(&artifacts, "The capital of France is", 8);
        println!("{repo}: {out:?}");
        assert!(out.contains("Paris"), "{repo}: {out:?}");
    }

    #[test]
    fn mlx_affine_4bit() {
        scheme(
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.99,
        );
    }

    #[test]
    fn mlx_affine_8bit() {
        scheme(
            "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.999,
        );
    }

    /// meta-llama is gated; mlx-community's bf16 conversion is the unquantized reference.
    #[test]
    fn mlx_affine_4bit_llama() {
        scheme(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community/Llama-3.2-1B-Instruct-bf16",
            [0, 15],
            PROJS,
            0.99,
        );
    }

    #[test]
    fn ct_int_quantized_w8a8() {
        scheme(
            "RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.999,
        );
    }

    #[test]
    fn ct_float_quantized_fp8_channel() {
        scheme(
            "RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.999,
        );
    }

    #[test]
    fn ct_float_quantized_fp8_block() {
        scheme(
            "xihc-ucb/Qwen2.5-0.5B-Instruct-vLLM-FP8-Block",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.999,
        );
    }

    #[test]
    fn ct_pack_quantized_int4_group_actorder() {
        scheme(
            "RedHatAI/Qwen2.5-0.5B-quantized.w4a16",
            "Qwen/Qwen2.5-0.5B",
            [0, 23],
            PROJS,
            0.99,
        );
    }

    #[test]
    fn ct_pack_quantized_int8_channel() {
        scheme(
            "RedHatAI/Qwen2.5-0.5B-quantized.w8a16",
            "Qwen/Qwen2.5-0.5B",
            [0, 23],
            PROJS,
            0.999,
        );
    }

    #[test]
    fn ct_pack_quantized_int4_asymmetric() {
        // AWQ smoothing rescales the input columns of q/k/v/gate/up/down; o_proj is left as-is here.
        scheme(
            "Amr04/Qwen2.5-0.5B-Instruct-AWQ-W4A16-ASYM",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            &["self_attn.o_proj"],
            0.99,
        );
    }

    #[test]
    fn ct_nvfp4_pack_quantized() {
        scheme(
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            [0, 21],
            PROJS,
            0.99,
        );
    }

    /// llm-compressor `mixed-precision`: NVFP4 for q/k/v/o/gate/up, FP8 channel for down_proj.
    #[test]
    fn ct_mixed_precision_nvfp4_fp8() {
        scheme(
            "nm-testing/nvfp4_fp8_mixed-e2e",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            [0, 21],
            PROJS,
            0.99,
        );
    }

    /// `mixed-precision`: int4 group-128 pack-quantized attention, int8 channel MLP. The recipe's
    /// AWQ smoothing rescales the input columns of q/k/v (and folds the inverse into the norms),
    /// so only o_proj (int4) and the MLP (int8) compare against the unquantized weights.
    #[test]
    fn ct_mixed_precision_int4_int8() {
        scheme(
            "nm-testing/multiple_modifiers_gptq_awq-e2e",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            [0, 21],
            &["self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"],
            0.99,
        );
    }

    #[test]
    fn modelopt_fp8_per_channel() {
        scheme(
            "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pc-pt",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.999,
        );
    }

    #[test]
    fn modelopt_nvfp4() {
        scheme(
            "Azzam15/qwen2.5-0.5b-nvfp4",
            "Qwen/Qwen2.5-0.5B-Instruct",
            [0, 23],
            PROJS,
            0.99,
        );
    }

    /// Schemes we can't expand are refused from the config alone, before any weight is read.
    #[test]
    fn unsupported_quant_configs_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        for (qc, ok) in [
            (r#"{"quantization_config":{"quant_method":"gptq","bits":4}}"#, true),
            (r#"{"quantization_config":{"quant_method":"gptq","bits":8}}"#, false),
            (r#"{"quantization_config":{"quant_method":"awq"}}"#, true),
            (r#"{"quantization_config":{"quant_method":"awq","version":"gemv"}}"#, false),
            (r#"{"quantization_config":{"quant_method":"bitsandbytes","bnb_4bit_quant_type":"nf4"}}"#, true),
            (
                r#"{"quantization_config":{"quant_method":"compressed-tensors","format":"mxfp4-pack-quantized"}}"#,
                true,
            ),
            (r#"{"quantization":{"bits":4,"group_size":64}}"#, true),
            (
                r#"{"quantization_config":{"quant_method":"compressed-tensors","format":"marlin-24"}}"#,
                false,
            ),
            (
                r#"{"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"g0":{"format":"nvfp4-pack-quantized"},"g1":{"format":"float-quantized"}}}}"#,
                true,
            ),
            // A group in a format `dequantize` can't expand still refuses the whole checkpoint.
            (
                r#"{"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"g0":{"format":"mxfp4-pack-quantized"},"g1":{"format":"mxfp8-quantized"}}}}"#,
                false,
            ),
            (
                r#"{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","sparsity_config":{"format":"sparse-24-bitmask"}}}"#,
                false,
            ),
            (
                r#"{"quantization_config":{"quant_method":"modelopt","quant_algo":"W4A8_AWQ"}}"#,
                false,
            ),
            (
                r#"{"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","sparsity_config":{}}}"#,
                true,
            ),
            (
                r#"{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}"#,
                true,
            ),
            (r#"{"hidden_size":8}"#, true),
            (
                r#"{"quantization_config":{"quant_method":"glq","codebook":"e8_shell","bpw":4}}"#,
                true,
            ),
            (
                r#"{"quantization_config":{"quant_method":"glq","codebook":"e8p","bpw":3}}"#,
                true,
            ),
            (
                r#"{"quantization_config":{"quant_method":"glq","codebook":"trellis","variant":"3inst","trellis_layout":"kernel"}}"#,
                true,
            ),
            (
                r#"{"quantization_config":{"quant_method":"glq","codebook":"trellis","variant":"3inst"}}"#,
                false,
            ),
            (
                r#"{"quantization_config":{"quant_method":"glq","codebook":"trellis","variant":"hyb","trellis_layout":"kernel"}}"#,
                false,
            ),
            (
                r#"{"quantization_config":{"quant_method":"glq","codebook":"e8_relaxed"}}"#,
                false,
            ),
            (
                r#"{"model_type":"bitnet","quantization_config":{"quant_method":"bitnet","linear_class":"autobitlinear","quantization_mode":"offline"}}"#,
                true,
            ),
            // transformers' default class divides by the scale (HF1BitLLM).
            (
                r#"{"model_type":"llama","quantization_config":{"quant_method":"bitnet"}}"#,
                false,
            ),
            (
                r#"{"model_type":"bitnet","quantization_config":{"quant_method":"bitnet"}}"#,
                false,
            ),
            (
                r#"{"model_type":"bitnet","quantization_config":{"quant_method":"bitnet","linear_class":"autobitlinear","use_rms_norm":true}}"#,
                false,
            ),
        ] {
            std::fs::write(&cfg, qc).unwrap();
            assert_eq!(super::check(Some(&cfg), dir.path()).is_ok(), ok, "{qc}");
        }
        // The hub API trims quantization_config to quant_method: the picker keeps model_type bitnet
        // rows (microsoft/bitnet-b1.58-2B-4T) and blocks llama-typed bitnet ones.
        let hub = |t: &str| serde_json::json!({"model_type": t, "quantization_config": {"quant_method": "bitnet"}});
        assert_eq!(super::scheme_blocker(&hub("bitnet"), None), None);
        assert!(super::scheme_blocker(&hub("llama"), None).is_some());
    }
}
