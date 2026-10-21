//! # ONNX loader (rten-onnx backend)
//!
//! - `OnnxWeightReader`: initializers as tensors ([`WeightReader`]).
//! - [`load_causal_lm`]: decoder-only LLM exports → HF-named f32 weights for `GraphLm`, resolved
//!   through the graph (MatMul/Gemm, MatMulNBits, MatMulBnb4, MatMulInteger, DequantizeLinear).

use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path};

use rten_onnx::onnx::{AttributeProto, DataLocation, DataType, ModelProto, NodeProto, TensorProto};

use crate::model::loaders::{TensorMetadata, WeightReader};
use crate::model::graph_lm::{GraphLm, GraphLmConfig};
use crate::model::{bf16_to_f32, f16_to_f32, ModelManifest, SourceFormat};
use crate::model::{QuantFormat, VirtualTensor};

/// ONNX weight reader: maps tensor names to Gibbs Tensor objects
pub struct OnnxWeightReader {
    model: ModelProto,
    tensor_map: HashMap<String, VirtualTensor>,
}

impl OnnxWeightReader {
    /// Load ONNX model from binary file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Failed to read ONNX file: {}", e))?;
        Self::from_bytes(&bytes)
    }

    /// Load ONNX model from byte slice
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let model = ModelProto::parse_buf(bytes)
            .map_err(|e| format!("Failed to parse ONNX protobuf: {}", e))?;

        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| "No graph in ONNX model".to_string())?;

        let mut tensor_map = HashMap::new();
        let mut byte_offset = 0;

        // Extract initializers (weight tensors) from the graph
        for initializer in &graph.initializer {
            let name = initializer
                .name
                .as_ref()
                .cloned()
                .unwrap_or_else(|| "".to_string());
            let dims: Vec<usize> = initializer.dims.iter().map(|d| *d as usize).collect();

            let quant_format = Self::dtype_to_quant_format(initializer.data_type);
            let dtype = Self::dtype_to_string(initializer.data_type);

            let raw_len = initializer
                .raw_data
                .as_ref()
                .map(|cell| cell.borrow().len())
                .unwrap_or(0);

            let byte_len = if raw_len > 0 {
                raw_len
            } else if !initializer.float_data.is_empty() {
                initializer.float_data.len() * 4
            } else if !initializer.int32_data.is_empty() {
                initializer.int32_data.len() * 4
            } else if !initializer.int64_data.is_empty() {
                initializer.int64_data.len() * 8
            } else {
                0
            };

            tensor_map.insert(
                name.clone(),
                VirtualTensor {
                    name,
                    shape: dims,
                    quant_format,
                    byte_offset,
                    byte_len,
                    dtype,
                    shard_index: 0,
                },
            );

            byte_offset += byte_len;
        }

        Ok(OnnxWeightReader { model, tensor_map })
    }

    /// Convert ONNX dtype to quantization format
    fn dtype_to_quant_format(dtype: Option<DataType>) -> QuantFormat {
        match dtype {
            Some(DataType::FLOAT) => QuantFormat::FP32,
            Some(DataType::FLOAT16) => QuantFormat::FP16,
            Some(DataType::BFLOAT16) => QuantFormat::BF16,
            Some(DataType::INT8) | Some(DataType::UINT8) => QuantFormat::GGUF_Q8_0,
            _ => QuantFormat::FP32,
        }
    }

    /// Convert ONNX dtype code to readable string
    fn dtype_to_string(dtype: Option<DataType>) -> String {
        match dtype {
            Some(DataType::FLOAT) => "F32".to_string(),
            Some(DataType::FLOAT16) => "F16".to_string(),
            Some(DataType::DOUBLE) => "F64".to_string(),
            Some(DataType::BFLOAT16) => "BF16".to_string(),
            Some(DataType::INT8) => "I8".to_string(),
            Some(DataType::UINT8) => "U8".to_string(),
            Some(DataType::INT16) => "I16".to_string(),
            Some(DataType::INT32) => "I32".to_string(),
            Some(DataType::INT64) => "I64".to_string(),
            Some(other) => other.to_string(),
            None => "UNKNOWN".to_string(),
        }
    }
    /// Load initializer data into tensor (copies raw bytes)
    fn load_initializer_bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        let graph = self
            .model
            .graph
            .as_ref()
            .ok_or_else(|| "No graph in model".to_string())?;

        for init in &graph.initializer {
            if init.name.as_deref() == Some(name) {
                if let Some(cell) = &init.raw_data {
                    let borrowed = cell.borrow();
                    if !borrowed.is_empty() {
                        return Ok(borrowed.clone());
                    }
                }
                if !init.float_data.is_empty() {
                    let mut bytes = Vec::with_capacity(init.float_data.len() * 4);
                    for f in &init.float_data {
                        bytes.extend_from_slice(&f.to_le_bytes());
                    }
                    return Ok(bytes);
                } else if !init.int32_data.is_empty() {
                    let mut bytes = Vec::with_capacity(init.int32_data.len() * 4);
                    for i in &init.int32_data {
                        bytes.extend_from_slice(&i.to_le_bytes());
                    }
                    return Ok(bytes);
                } else if !init.int64_data.is_empty() {
                    let mut bytes = Vec::with_capacity(init.int64_data.len() * 8);
                    for i in &init.int64_data {
                        bytes.extend_from_slice(&i.to_le_bytes());
                    }
                    return Ok(bytes);
                }
            }
        }
        Err(format!("Initializer not found: {}", name))
    }
}

impl WeightReader for OnnxWeightReader {
    fn parse_header(&mut self) -> Result<HashMap<String, String>, String> {
        let mut meta = HashMap::new();
        meta.insert(
            "producer_name".to_string(),
            self.model.producer_name.clone().unwrap_or_default(),
        );
        meta.insert(
            "producer_version".to_string(),
            self.model.producer_version.clone().unwrap_or_default(),
        );
        meta.insert(
            "ir_version".to_string(),
            self.model
                .ir_version
                .map(|v| v.to_string())
                .unwrap_or_default(),
        );
        Ok(meta)
    }

    fn list_tensors(&self) -> Result<Vec<TensorMetadata>, String> {
        Ok(self
            .tensor_map
            .values()
            .map(|vt| TensorMetadata {
                name: vt.name.clone(),
                shape: vt.shape.clone(),
                dtype: vt.dtype.clone(),
                byte_offset: 0,
                byte_len: vt.byte_len,
            })
            .collect())
    }

    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let bytes = self.load_initializer_bytes(name)?;
        let vt = self
            .tensor_map
            .get(name)
            .ok_or_else(|| format!("Tensor not found: {}", name))?;
        if vt.dtype == "F32" {
            let floats = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(floats)
        } else if vt.dtype == "F16" {
            let floats = bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect();
            Ok(floats)
        } else {
            let floats = bytes.iter().map(|&b| b as f32).collect();
            Ok(floats)
        }
    }

    fn load_tensor_quantized(&self, name: &str) -> Result<Vec<u8>, String> {
        self.load_initializer_bytes(name)
    }
}

fn attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute
        .iter()
        .find(|a| a.name.as_deref() == Some(name))
}

fn int(node: &NodeProto, name: &str) -> Option<i64> {
    attr(node, name).and_then(|a| a.i)
}

/// Optional ONNX inputs are omitted or named "".
fn has_input(node: &NodeProto, i: usize) -> bool {
    node.input.get(i).is_some_and(|s| !s.is_empty())
}

fn op(node: &NodeProto) -> &str {
    node.op_type.as_deref().unwrap_or("")
}

/// Ops that carry weights in a layout we don't decode; refusing beats silently dropping a layer.
const UNSUPPORTED_WEIGHT_OPS: &[&str] = &[
    "QLinearMatMul",
    "MatMulFpQ4",
    "GatherBlockQuantized",
    "QGemm",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
    "QAttention",
    "Attention",
    "FusedMatMul",
    "QOrderedMatMul",
];

/// A constant folded out of the graph; integer dtypes keep their integer values (`int`).
struct Const {
    dims: Vec<usize>,
    vals: Vec<f32>,
    int: bool,
}

/// Loads a decoder-only LLM ONNX export (optimum / onnxruntime-genai / transformers.js) by pulling
/// its weights out of the graph into the HF tensor names [`GraphLm::from_manifest`] reads.
/// The graph is never executed; external data files are resolved beside `path`.
pub fn load_causal_lm(cfg: GraphLmConfig, path: &Path) -> Result<GraphLm, String> {
    let tensors = extract_hf_tensors(path, cfg.num_heads * cfg.head_dim)?;
    // ponytail: materializes every weight as f32 (4 B/param on top of the source file); stream
    // tensors into from_manifest if RAM for multi-B-param ONNX exports matters.
    let mut data = Vec::with_capacity(tensors.values().map(|t| t.1.len() * 4).sum());
    let mut manifest = ModelManifest {
        format: SourceFormat::Onnx,
        tensors: HashMap::new(),
        metadata: HashMap::new(),
        shard_count: 1,
        detected_architecture: String::new(),
    };
    for (name, (shape, vals)) in tensors {
        let byte_offset = data.len();
        data.extend(vals.iter().flat_map(|v| v.to_le_bytes()));
        let vt = VirtualTensor {
            name: name.clone(),
            shape,
            quant_format: QuantFormat::FP32,
            byte_offset,
            byte_len: vals.len() * 4,
            dtype: "F32".into(),
            shard_index: 0,
        };
        manifest.tensors.insert(name, vt);
    }
    GraphLm::from_manifest(cfg, &manifest, &data)
}

/// HF tensor name → (HF shape, f32 values). `q_dim` (= heads × head_dim) splits fused `qkv_proj`.
pub fn extract_hf_tensors(
    path: &Path,
    q_dim: usize,
) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let model = ModelProto::parse_buf(&bytes).map_err(|e| format!("ONNX protobuf: {e}"))?;
    drop(bytes);
    let graph = model.graph.as_ref().ok_or("ONNX model has no graph")?;
    let mut g = OnnxGraph {
        dir: path.parent().unwrap_or(Path::new(".")),
        inits: graph
            .initializer
            .iter()
            .map(|t| (t.name.as_deref().unwrap_or(""), t))
            .collect(),
        producer: HashMap::new(),
        consumers: HashMap::new(),
        ext: HashMap::new(),
        depth: 0,
    };
    for n in &graph.node {
        if n.output.is_empty() {
            let name = n.name.as_deref().unwrap_or("");
            return Err(format!("ONNX node '{name}' ({}) has no outputs", op(n)));
        }
        for o in &n.output {
            g.producer.insert(o, n);
        }
        for i in &n.input {
            g.consumers.entry(i.as_str()).or_default().push(n);
        }
    }

    let mut out = HashMap::new();
    for node in &graph.node {
        let name = node.name.as_deref().unwrap_or("");
        if UNSUPPORTED_WEIGHT_OPS.contains(&op(node))
            && node.input.iter().any(|i| g.inits.contains_key(i.as_str()))
        {
            return Err(format!(
                "unsupported ONNX weight op '{}' at node '{name}'",
                op(node)
            ));
        }
        let Some(module) = module_path(name) else {
            continue;
        };
        match op(node) {
            "Gather" => {
                let Some(mut w) = g.resolve_opt(node, 0)? else {
                    continue;
                };
                // Dynamic-quantized exports gather int8 rows and dequantize the gathered output;
                // other integer Gathers index constant shape tensors, not weights.
                if w.int {
                    match g.consumer(&node.output[0], "DequantizeLinear") {
                        Some(dq) => w = g.dequantize(dq, w)?,
                        None => continue,
                    }
                }
                if w.dims.len() == 2 {
                    out.insert(format!("{module}.weight"), (w.dims, w.vals));
                }
            }
            "SimplifiedLayerNormalization"
            | "LayerNormalization"
            | "SkipSimplifiedLayerNormalization"
            | "SkipLayerNormalization" => {
                // Simplified (RMS) norms have no beta; SkipSimplified input 3 is a pre-norm bias.
                let (w, beta) = match op(node) {
                    "LayerNormalization" => (1, Some(2)),
                    "SkipLayerNormalization" => (2, Some(3)),
                    "SkipSimplifiedLayerNormalization" => (2, None),
                    _ => (1, None),
                };
                for (i, suffix) in [(Some(w), "weight"), (beta, "bias")] {
                    let Some(i) = i else { continue };
                    if let Some(c) = g.resolve_opt(node, i)? {
                        out.insert(format!("{module}.{suffix}"), (c.dims, c.vals));
                    }
                }
                // Pre-norm `bias` (Skip* input 3 / 4) belongs to the preceding linear; dropping
                // a constant one would silently corrupt the layer.
                let pre_bias = match op(node) {
                    "SkipSimplifiedLayerNormalization" => Some(3),
                    "SkipLayerNormalization" => Some(4),
                    _ => None,
                };
                if let Some(i) = pre_bias {
                    if g.resolve_opt(node, i)?.is_some() {
                        return Err(format!(
                            "{name}: {} pre-norm bias input unsupported",
                            op(node)
                        ));
                    }
                }
            }
            // Decomposed RMSNorm (optimum): `weight * normed` with a 1-D constant weight.
            "Mul" if module.ends_with("norm") => {
                for i in 0..node.input.len() {
                    if let Some(c) = g.resolve_opt(node, i)? {
                        if c.dims.len() == 1 && c.vals.len() > 1 {
                            out.insert(format!("{module}.weight"), (c.dims, c.vals));
                        }
                    }
                }
            }
            _ => {
                let Some((w, n, k, bias)) = g.linear(node)? else {
                    continue;
                };
                let mut lin = vec![(module.clone(), w, n, bias)];
                if let Some(base) = module.strip_suffix("qkv_proj") {
                    let (_, w, _, bias) = lin.pop().unwrap();
                    let kv = n
                        .checked_sub(q_dim)
                        .map(|r| r / 2)
                        .filter(|kv| q_dim + 2 * kv == n);
                    let kv = kv.ok_or_else(|| {
                        format!("{name}: qkv rows {n} don't split as {q_dim} + 2×kv")
                    })?;
                    let mut row = 0;
                    for (p, rows) in [("q_proj", q_dim), ("k_proj", kv), ("v_proj", kv)] {
                        let b = bias.as_ref().map(|b| b[row..row + rows].to_vec());
                        lin.push((
                            format!("{base}{p}"),
                            w[row * k..(row + rows) * k].to_vec(),
                            rows,
                            b,
                        ));
                        row += rows;
                    }
                }
                for (m, w, n, bias) in lin {
                    out.insert(format!("{m}.weight"), (vec![n, k], w));
                    if let Some(b) = bias {
                        out.insert(format!("{m}.bias"), (vec![n], b));
                    }
                }
            }
        }
    }
    if out.is_empty() {
        return Err("ONNX graph has no weights reachable from named module nodes".into());
    }
    Ok(out)
}

/// `/model/layers.0/attn/q_proj/MatMul` → `model.layers.0.self_attn.q_proj` (genai builder names
/// attention `attn` and the final norm `layers.N/final_norm_layernorm`).
fn module_path(node_name: &str) -> Option<String> {
    let (dir, _op) = node_name.trim_start_matches('/').rsplit_once('/')?;
    if dir.ends_with("final_norm_layernorm") {
        return Some("model.norm".into());
    }
    Some(dir.replace('/', ".").replace(".attn.", ".self_attn."))
}

struct OnnxGraph<'a> {
    dir: &'a Path,
    inits: HashMap<&'a str, &'a TensorProto>,
    producer: HashMap<&'a str, &'a NodeProto>,
    consumers: HashMap<&'a str, Vec<&'a NodeProto>>,
    ext: HashMap<String, memmap2::Mmap>,
    /// Producer-chain depth of the current [`Self::resolve`]; bounds cyclic malformed graphs.
    depth: usize,
}

impl<'a> OnnxGraph<'a> {
    fn consumer(&self, value: &str, op_type: &str) -> Option<&'a NodeProto> {
        self.consumers
            .get(value)?
            .iter()
            .copied()
            .find(|n| op(n) == op_type)
    }

    fn resolve_opt(&mut self, node: &NodeProto, i: usize) -> Result<Option<Const>, String> {
        if !has_input(node, i) {
            return Ok(None);
        }
        if self.depth >= 64 {
            return Err(format!(
                "{}: constant chain too deep (cycle?)",
                node.input[i]
            ));
        }
        self.depth += 1;
        let r = self.resolve(&node.input[i]);
        self.depth -= 1;
        r
    }

    fn resolve_req(&mut self, node: &NodeProto, i: usize, what: &str) -> Result<Const, String> {
        self.resolve_opt(node, i)?.ok_or_else(|| {
            format!(
                "{}: {what} (input {i}) is not a constant",
                node.name.as_deref().unwrap_or("")
            )
        })
    }

    /// Folds `value` to a constant through weight-only chains; `None` for runtime activations.
    fn resolve(&mut self, value: &str) -> Result<Option<Const>, String> {
        if let Some(&t) = self.inits.get(value) {
            return self.decode(t).map(Some);
        }
        let Some(&node) = self.producer.get(value) else {
            return Ok(None);
        };
        match op(node) {
            "Identity" | "Cast" => self.resolve_opt(node, 0),
            "Constant" => match node.attribute.iter().find_map(|a| a.t.as_ref()) {
                Some(t) => self.decode(t).map(Some),
                None => Ok(None),
            },
            "Transpose" => {
                let Some(c) = self.resolve_opt(node, 0)? else {
                    return Ok(None);
                };
                let perm = attr(node, "perm")
                    .map(|a| a.ints.as_slice())
                    .unwrap_or(&[1, 0]);
                if c.dims.len() != 2 || perm != [1, 0] {
                    return Err(format!("Transpose of constant {value}: only 2-D supported"));
                }
                Ok(Some(Const {
                    dims: vec![c.dims[1], c.dims[0]],
                    vals: transpose(&c.vals, c.dims[0], c.dims[1]),
                    int: c.int,
                }))
            }
            "DequantizeLinear" => match self.resolve_opt(node, 0)? {
                Some(x) => self.dequantize(node, x).map(Some),
                None => Ok(None),
            },
            _ => Ok(None),
        }
    }

    /// `DequantizeLinear(x, scale, zero_point?)`: per-tensor or per-axis.
    fn dequantize(&mut self, node: &NodeProto, mut x: Const) -> Result<Const, String> {
        let name = node.name.as_deref().unwrap_or("");
        if int(node, "block_size").unwrap_or(0) != 0 {
            return Err(format!("{name}: blocked DequantizeLinear is unsupported"));
        }
        let s = self.resolve_req(node, 1, "scale")?;
        let zp = self.resolve_opt(node, 2)?.map(|z| z.vals);
        let rank = x.dims.len() as i64;
        let axis = int(node, "axis").unwrap_or(1).rem_euclid(rank.max(1)) as usize;
        let (stride, extent) = if s.vals.len() == 1 {
            (1, 1)
        } else if axis < x.dims.len() {
            (x.dims[axis + 1..].iter().product(), x.dims[axis])
        } else {
            return Err(format!("{name}: per-axis scale on a rank-{rank} tensor"));
        };
        if s.vals.len() != extent || zp.as_ref().is_some_and(|z| z.len() != extent) {
            return Err(format!(
                "{name}: scale/zero_point len {}/{:?} vs axis extent {extent}",
                s.vals.len(),
                zp.as_ref().map(Vec::len)
            ));
        }
        for (i, v) in x.vals.iter_mut().enumerate() {
            let j = (i / stride) % extent;
            *v = (*v - zp.as_ref().map_or(0.0, |z| z[j])) * s.vals[j];
        }
        x.int = false;
        Ok(x)
    }

    /// Linear layers → (HF weight `[n, k]` row-major, n, k, bias). `None` when the node isn't a
    /// linear op with constant weights (e.g. attention score MatMuls).
    #[allow(clippy::type_complexity)]
    fn linear(
        &mut self,
        node: &'a NodeProto,
    ) -> Result<Option<(Vec<f32>, usize, usize, Option<Vec<f32>>)>, String> {
        let name = node.name.as_deref().unwrap_or("");
        let attr_usize = |a: &str| {
            int(node, a)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("{name}: missing attribute {a}"))
        };
        let mut out = node.output[0].as_str();
        let mut bias = None;
        let (w, n, k) = match op(node) {
            "MatMul" | "Gemm" => {
                let Some(b) = self.resolve_opt(node, 1)? else {
                    return Ok(None);
                };
                if b.dims.len() != 2 {
                    return Ok(None);
                }
                if op(node) == "Gemm" {
                    let f = |a: &str| attr(node, a).and_then(|a| a.f).unwrap_or(1.0);
                    let beta = f("beta");
                    bias = self
                        .resolve_opt(node, 2)?
                        .map(|c| c.vals.iter().map(|v| v * beta).collect());
                    if int(node, "transA").unwrap_or(0) != 0 || f("alpha") != 1.0 {
                        return Err(format!(
                            "{name}: Gemm with transA or alpha != 1 is unsupported"
                        ));
                    }
                }
                if op(node) == "Gemm" && int(node, "transB").unwrap_or(0) != 0 {
                    (b.vals, b.dims[0], b.dims[1])
                } else {
                    (
                        transpose(&b.vals, b.dims[0], b.dims[1]),
                        b.dims[1],
                        b.dims[0],
                    )
                }
            }
            "MatMulNBits" => {
                let (k, n, bits, bs) = (
                    attr_usize("K")?,
                    attr_usize("N")?,
                    attr_usize("bits")?,
                    attr_usize("block_size")?,
                );
                // ORT contrib constraint; also keeps `div_ceil(bs)` and the blob math sound.
                if (bits != 4 && bits != 8) || bs < 16 || !bs.is_power_of_two() {
                    return Err(format!(
                        "{name}: MatMulNBits bits={bits} block_size={bs} unsupported \
                         (bits 4|8, block_size power of two >= 16)"
                    ));
                }
                if has_input(node, 4) {
                    return Err(format!("{name}: MatMulNBits g_idx (act-order) unsupported"));
                }
                let q = self.resolve_req(node, 1, "B")?.vals;
                let scales = self.resolve_req(node, 2, "scales")?.vals;
                let zp = self.resolve_opt(node, 3)?;
                bias = self.resolve_opt(node, 5)?.map(|c| c.vals);
                let nblk = k.div_ceil(bs);
                let blob = bs * bits / 8;
                let sizes = || format!("{name}: MatMulNBits sizes don't match K={k} N={n}");
                let n_blocks = n.checked_mul(nblk).ok_or_else(sizes)?;
                let numel = n.checked_mul(k).ok_or_else(sizes)?;
                let zp_len = match &zp {
                    Some(z) if z.int && bits == 4 => n.checked_mul(nblk.div_ceil(2)),
                    _ => Some(n_blocks),
                };
                if n_blocks.checked_mul(blob) != Some(q.len())
                    || scales.len() != n_blocks
                    || zp.as_ref().is_some_and(|z| Some(z.vals.len()) != zp_len)
                {
                    return Err(sizes());
                }
                let mut w = vec![0f32; numel];
                for r in 0..n {
                    for b in 0..nblk {
                        let blk = r * nblk + b;
                        let z = match &zp {
                            None => (1u32 << (bits - 1)) as f32,
                            Some(z) if z.int && bits == 4 => {
                                let byte = z.vals[r * nblk.div_ceil(2) + b / 2] as u8;
                                ((byte >> (4 * (b % 2))) & 0xF) as f32
                            }
                            Some(z) => z.vals[blk],
                        };
                        for j in 0..bs.min(k - b * bs) {
                            let v = if bits == 4 {
                                let byte = q[blk * blob + j / 2] as u8;
                                ((byte >> (4 * (j % 2))) & 0xF) as f32
                            } else {
                                q[blk * blob + j]
                            };
                            w[r * k + b * bs + j] = (v - z) * scales[blk];
                        }
                    }
                }
                (w, n, k)
            }
            "MatMulBnb4" => {
                let (k, n, bs) = (
                    attr_usize("K")?,
                    attr_usize("N")?,
                    attr_usize("block_size")?,
                );
                let lut = match int(node, "quant_type") {
                    Some(0) => &FP4_LUT,
                    Some(1) => &NF4_LUT,
                    t => return Err(format!("{name}: MatMulBnb4 quant_type {t:?} unsupported")),
                };
                let q = self.resolve_req(node, 1, "B")?.vals;
                let absmax = self.resolve_req(node, 2, "absmax")?.vals;
                let numel = n.checked_mul(k).filter(|&nk| {
                    bs > 0 && q.len() == nk.div_ceil(2) && absmax.len() == nk.div_ceil(bs)
                });
                let Some(numel) = numel else {
                    return Err(format!(
                        "{name}: MatMulBnb4 B/absmax sizes don't match K={k} N={n} block_size={bs}"
                    ));
                };
                // bitsandbytes order: transposed [n, k], first element in the high nibble.
                let w = (0..numel)
                    .map(|i| {
                        let byte = q[i / 2] as u8;
                        let nib = if i % 2 == 0 { byte >> 4 } else { byte & 0xF };
                        lut[nib as usize] * absmax[i / bs]
                    })
                    .collect();
                (w, n, k)
            }
            // Dynamic quantization (onnxruntime quantize_dynamic):
            // MatMulInteger(xq, Bq, x_zp, B_zp) → Cast → Mul(·, Mul(x_scale, B_scale)).
            "MatMulInteger" => {
                let Some(b) = self.resolve_opt(node, 1)? else {
                    return Ok(None);
                };
                let &[k, n] = b.dims.as_slice() else {
                    return Err(format!(
                        "{name}: MatMulInteger weight must be 2-D, got {:?}",
                        b.dims
                    ));
                };
                let zp = self.resolve_opt(node, 3)?.map_or(vec![0.0], |c| c.vals);
                let missing = || format!("{name}: MatMulInteger without Cast→Mul output scale");
                let cast = self.consumer(out, "Cast").ok_or_else(missing)?;
                let mul = self.consumer(&cast.output[0], "Mul").ok_or_else(missing)?;
                let scale_in = mul
                    .input
                    .iter()
                    .find(|i| **i != cast.output[0])
                    .ok_or_else(missing)?;
                let scale_mul = *self.producer.get(scale_in.as_str()).ok_or_else(missing)?;
                let mut scale = None;
                for i in 0..scale_mul.input.len() {
                    scale = scale.or(self.resolve_opt(scale_mul, i)?);
                }
                let s = scale.ok_or_else(missing)?.vals;
                out = mul.output[0].as_str();
                let pick = |v: &[f32], c: usize| if v.len() == 1 { v[0] } else { v[c] };
                if (s.len() != 1 && s.len() != n) || (zp.len() != 1 && zp.len() != n) {
                    return Err(format!(
                        "{name}: MatMulInteger scale/zero_point must be per-tensor or per-column"
                    ));
                }
                let mut w = vec![0f32; n * k];
                for kk in 0..k {
                    for c in 0..n {
                        w[c * k + kk] = (b.vals[kk * n + c] - pick(&zp, c)) * pick(&s, c);
                    }
                }
                (w, n, k)
            }
            _ => return Ok(None),
        };
        if bias.is_none() {
            if let Some(add) = self.consumer(out, "Add") {
                for i in 0..add.input.len() {
                    if let Some(c) = self.resolve_opt(add, i)? {
                        if c.vals.len() == n {
                            bias = Some(c.vals);
                        }
                    }
                }
            }
        }
        if bias.as_ref().is_some_and(|b| b.len() != n) {
            return Err(format!("{name}: bias length doesn't match N={n}"));
        }
        Ok(Some((w, n, k, bias)))
    }

    /// Initializer → f32 values (raw_data, typed fields or external data beside the model).
    fn decode(&mut self, t: &TensorProto) -> Result<Const, String> {
        let name = t.name.as_deref().unwrap_or("");
        let bad = || format!("{name}: invalid dims {:?}", t.dims);
        let dims = t
            .dims
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| bad()))
            .collect::<Result<Vec<usize>, _>>()?;
        let numel = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(bad)?;
        let dt = t
            .data_type
            .ok_or_else(|| format!("{name}: missing data_type"))?;
        let raw_cell = t.raw_data.as_ref().map(|c| c.borrow());
        let mut raw: &[u8] = raw_cell.as_deref().map_or(&[], |v| v.as_slice());
        if t.data_location == Some(DataLocation::EXTERNAL) {
            let kv = |k: &str| {
                t.external_data
                    .iter()
                    .find(|e| e.key.as_deref() == Some(k))
                    .and_then(|e| e.value.as_deref())
            };
            let num = |k: &str| {
                kv(k).map(|v| {
                    v.parse::<usize>()
                        .map_err(|e| format!("{name}: external {k}: {e}"))
                })
            };
            let loc =
                kv("location").ok_or_else(|| format!("{name}: external data without location"))?;
            // External data must stay beside the model: no absolute paths, `..`, or prefixes.
            if Path::new(loc)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(format!(
                    "{name}: external data location {loc:?} escapes the model dir"
                ));
            }
            if !self.ext.contains_key(loc) {
                let p = self.dir.join(loc);
                let f = fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                // SAFETY: read-only map of a model file nobody mutates while loading.
                let m = unsafe { memmap2::Mmap::map(&f) }
                    .map_err(|e| format!("{}: {e}", p.display()))?;
                self.ext.insert(loc.to_string(), m);
            }
            let m = &self.ext[loc];
            let off = num("offset").transpose()?.unwrap_or(0);
            let len = num("length")
                .transpose()?
                .unwrap_or(m.len().saturating_sub(off));
            raw = off
                .checked_add(len)
                .and_then(|end| m.get(off..end))
                .ok_or_else(|| format!("{name}: external range {off}+{len} past end of {loc}"))?;
        }
        let int = matches!(
            dt,
            DataType::INT8 | DataType::UINT8 | DataType::INT32 | DataType::INT64
        );
        let vals: Vec<f32> = if !raw.is_empty() {
            match dt {
                DataType::FLOAT => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                DataType::FLOAT16 => raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                DataType::BFLOAT16 => raw
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                DataType::INT8 => raw.iter().map(|&b| b as i8 as f32).collect(),
                DataType::UINT8 => raw.iter().map(|&b| b as f32).collect(),
                DataType::INT32 => raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                DataType::INT64 => raw
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                other => return Err(format!("{name}: unsupported ONNX dtype {other:?}")),
            }
        } else {
            match dt {
                DataType::FLOAT => t.float_data.clone(),
                DataType::FLOAT16 => t.int32_data.iter().map(|&b| f16_to_f32(b as u16)).collect(),
                DataType::BFLOAT16 => t
                    .int32_data
                    .iter()
                    .map(|&b| bf16_to_f32(b as u16))
                    .collect(),
                DataType::INT8 | DataType::UINT8 | DataType::INT32 => {
                    t.int32_data.iter().map(|&v| v as f32).collect()
                }
                DataType::INT64 => t.int64_data.iter().map(|&v| v as f32).collect(),
                DataType::DOUBLE => t.double_data.iter().map(|&v| v as f32).collect(),
                other => return Err(format!("{name}: unsupported ONNX dtype {other:?}")),
            }
        };
        if vals.len() != numel {
            return Err(format!("{name}: {} values for shape {dims:?}", vals.len()));
        }
        Ok(Const { dims, vals, int })
    }
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

/// bitsandbytes NF4 / FP4 code books (as in onnxruntime's MatMulBnb4).
pub(crate) const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];
pub(crate) const FP4_LUT: [f32; 16] = [
    0.0,
    0.005_208_333,
    0.666_666_7,
    1.0,
    0.333_333_34,
    0.5,
    0.166_666_67,
    0.25,
    -0.0,
    -0.005_208_333,
    -0.666_666_7,
    -1.0,
    -0.333_333_34,
    -0.5,
    -0.166_666_67,
    -0.25,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::hub::ModelArtifacts;
    use crate::model::UniversalModelLoader;
    use std::path::PathBuf;

    const SMOL: &str = "disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main";
    const QWEN: &str = "disk/models--onnx-community--Qwen2.5-Coder-0.5B-Instruct/snapshots/main";

    /// Crafted malformed inputs (path-escaping external data, bad MatMulNBits block_size,
    /// missing dtype, cyclic constant chain) are `Err`, never a panic or an out-of-dir read.
    #[test]
    fn malformed_onnx_inputs_are_errors() {
        use rten_onnx::onnx::StringStringEntryProto;
        let ext = |loc: &str| TensorProto {
            name: Some("w".into()),
            dims: vec![2],
            data_type: Some(DataType::FLOAT),
            data_location: Some(DataLocation::EXTERNAL),
            external_data: vec![StringStringEntryProto {
                key: Some("location".into()),
                value: Some(loc.into()),
            }],
            ..Default::default()
        };
        let f32_init = |name: &str, dims: Vec<i64>, vals: Vec<f32>| TensorProto {
            name: Some(name.into()),
            dims,
            data_type: Some(DataType::FLOAT),
            float_data: vals,
            ..Default::default()
        };
        let u8_init = |name: &str, n: usize| TensorProto {
            name: Some(name.into()),
            dims: vec![n as i64],
            data_type: Some(DataType::UINT8),
            raw_data: Some(std::cell::RefCell::new(vec![0x88; n])),
            ..Default::default()
        };
        let iattr = |name: &str, i: i64| AttributeProto {
            name: Some(name.into()),
            i: Some(i),
            ..Default::default()
        };
        let node =
            |op: &str, inputs: &[&str], out: &str, attribute: Vec<AttributeProto>| NodeProto {
                name: Some(format!("/model/layers.0/mlp/up_proj/{op}")),
                op_type: Some(op.into()),
                input: inputs.iter().map(|s| s.to_string()).collect(),
                output: vec![out.into()],
                attribute,
                ..Default::default()
            };
        let new_graph = || OnnxGraph {
            dir: Path::new("/tmp"),
            inits: HashMap::new(),
            producer: HashMap::new(),
            consumers: HashMap::new(),
            ext: HashMap::new(),
            depth: 0,
        };

        for loc in ["../x", "/etc/hosts", "a/../../b"] {
            let err = new_graph().decode(&ext(loc)).err().unwrap();
            assert!(err.contains("escapes"), "{loc}: {err}");
        }
        let mut no_dtype = f32_init("w", vec![1], vec![1.0]);
        no_dtype.data_type = None;
        assert!(new_graph().decode(&no_dtype).is_err());
        assert!(new_graph()
            .decode(&f32_init("w", vec![-1], vec![]))
            .is_err());

        // MatMulNBits K=N=16 with 4-bit B; valid block_size decodes, bad ones are Err.
        let (b, s) = (u8_init("B", 16 * 8), f32_init("S", vec![16], vec![1.0; 16]));
        for (bs, ok) in [(16, true), (0, false), (8, false), (24, false)] {
            let mm = node(
                "MatMulNBits",
                &["x", "B", "S"],
                "y",
                vec![
                    iattr("K", 16),
                    iattr("N", 16),
                    iattr("bits", 4),
                    iattr("block_size", bs),
                ],
            );
            let mut g = new_graph();
            g.inits = HashMap::from([("B", &b), ("S", &s)]);
            let r = g.linear(&mm);
            assert_eq!(r.is_ok(), ok, "block_size {bs}: {:?}", r.err());
        }

        // Identity cycle a <-> b: bounded depth, Err instead of stack overflow.
        let (ia, ib) = (
            node("Identity", &["b"], "a", vec![]),
            node("Identity", &["a"], "b", vec![]),
        );
        let mm = node("MatMul", &["x", "a"], "y", vec![]);
        let mut g = new_graph();
        g.producer = HashMap::from([("a", &ia), ("b", &ib)]);
        assert!(g.linear(&mm).is_err());
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += x as f64 * y as f64;
            aa += x as f64 * x as f64;
            bb += y as f64 * y as f64;
        }
        ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
    }

    /// Every ONNX variant decodes to its safetensors original: f32 exact (bf16 originals widen
    /// exactly), fp16 within half-precision rounding, quantized variants by per-tensor cosine.
    /// Key sets must match too (a tied `lm_head.weight` is compared against the embedding).
    #[test]
    fn onnx_variants_match_safetensors_original() {
        const QWEN_ORIG: &str = "disk/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/main";
        // (original repo, onnx repo, q_dim, [(variant, None = exact/fp16 rounding | Some(min cosine))])
        let pairs: [(&str, &str, usize, &[(&str, Option<f64>)]); 2] = [
            (
                SMOL,
                SMOL,
                576,
                &[
                    ("model", None),
                    ("model_fp16", None),
                    ("model_int8", Some(0.99)),
                    ("model_uint8", Some(0.99)),
                    ("model_quantized", Some(0.99)),
                    ("model_q4", Some(0.97)),
                    ("model_q4f16", Some(0.97)),
                    ("model_bnb4", Some(0.97)),
                ],
            ),
            (
                QWEN_ORIG,
                QWEN,
                896,
                &[("model", None), ("model_q4", Some(0.97))],
            ),
        ];
        for (orig, repo, q_dim, variants) in pairs {
            let Ok(st) = fs::read(Path::new(orig).join("model.safetensors")) else {
                println!("SKIP: {orig}/model.safetensors absent");
                continue;
            };
            let manifest = UniversalModelLoader::parse_safetensors(&st).unwrap();
            for &(variant, min_cos) in variants {
                let path = PathBuf::from(repo).join(format!("onnx/{variant}.onnx"));
                if !path.exists() {
                    println!("SKIP: {}", path.display());
                    continue;
                }
                let got = extract_hf_tensors(&path, q_dim).unwrap();
                for name in got.keys() {
                    let tied = name == "lm_head.weight";
                    assert!(
                        tied || manifest.tensors.contains_key(name),
                        "{variant}: extracted {name} not in original"
                    );
                }
                let mut names: Vec<&String> = manifest.tensors.keys().collect();
                if !manifest.tensors.contains_key("lm_head.weight")
                    && got.contains_key("lm_head.weight")
                {
                    names.push(got.get_key_value("lm_head.weight").unwrap().0);
                }
                let (mut worst, mut sum_cos) = ((f64::MAX, 0f32, String::new()), 0f64);
                for name in &names {
                    let src = if manifest.tensors.contains_key(*name) {
                        name.as_str()
                    } else {
                        "model.embed_tokens.weight"
                    };
                    let want = manifest.extract_tensor_f32(&st, src).unwrap();
                    let (shape, vals) = got
                        .get(*name)
                        .unwrap_or_else(|| panic!("{variant}: {name} missing"));
                    assert_eq!(shape, &manifest.tensors[src].shape, "{variant}: {name}");
                    let cos = cosine(vals, &want);
                    let mut max_diff = 0f32;
                    for (&a, &b) in vals.iter().zip(&want) {
                        let d = (a - b).abs();
                        max_diff = max_diff.max(d);
                        match (variant, min_cos) {
                            ("model", _) => assert!(d <= 1e-6, "{variant}: {name} {a} vs {b}"),
                            (_, None) => {
                                assert!(d <= b.abs() * 1e-3 + 6e-8, "{variant}: {name} {a} vs {b}")
                            }
                            _ => {}
                        }
                    }
                    if let Some(min) = min_cos {
                        assert!(cos >= min, "{variant}: {name} cosine {cos:.5} < {min}");
                    }
                    sum_cos += cos;
                    if cos < worst.0 {
                        worst = (cos, max_diff, name.to_string());
                    }
                }
                println!(
                    "{repo} {variant}: {} tensors, mean cosine {:.5}, worst {:.5} (max abs diff {:.3e}) at {}",
                    names.len(),
                    sum_cos / names.len() as f64,
                    worst.0,
                    worst.1,
                    worst.2
                );
            }
        }
    }

    /// `artifacts` through the family-spec loader (`runtime::lm::SpecLm`, detected family).
    fn spec_lm(mut artifacts: ModelArtifacts) -> crate::scheduler::ExecutableModel {
        artifacts.local_dir = artifacts.config_path.as_ref().and_then(|p| p.parent()).map(Path::to_path_buf);
        let reg = crate::arch::registry().unwrap();
        let mode = crate::hardware::PlacementMode::Auto;
        crate::scheduler::ExecutableModel::Native(crate::runtime::lm::load(&artifacts, reg, None, mode).unwrap())
    }

    /// The f32 ONNX export and its safetensors original produce the same greedy tokens through
    /// the spec loader (each repo with its own config.json / tokenizer.json).
    #[test]
    fn onnx_f32_greedy_matches_safetensors_original() {
        let pairs = [(
            "disk/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/main",
            QWEN,
        )];
        let load = |dir: &str, weights: &str| {
            let dir = PathBuf::from(dir);
            let artifacts = ModelArtifacts {
                model_id: dir.display().to_string(),
                config_path: Some(dir.join("config.json")),
                tokenizer_path: Some(dir.join("tokenizer.json")),
                weights_path: Some(dir.join(weights)),
                ..Default::default()
            };
            spec_lm(artifacts)
        };
        for (orig, repo) in pairs {
            let onnx = PathBuf::from(repo).join("onnx/model.onnx");
            if !Path::new(orig).join("model.safetensors").exists() || !onnx.exists() {
                println!("SKIP: {orig} / {}", onnx.display());
                continue;
            }
            let msg = [(
                "user",
                "What is the capital of France? Answer in one sentence.",
            )];
            let mut outs = Vec::new();
            for llm in [
                load(orig, "model.safetensors"),
                load(repo, "onnx/model.onnx"),
            ] {
                let tok = llm.tokenizer();
                let prompt = tok.apply_chat_template(&msg);
                let ids = tok.encode(&prompt);
                let answer = llm.generate_text_measured_plain(&prompt, 24, 0.0).unwrap().0;
                println!("{}: {} prompt tokens -> {answer:?}", tok.vocab_size(), ids.len());
                outs.push((ids, answer));
            }
            assert_eq!(outs[0].0, outs[1].0, "{repo}: prompt tokens differ");
            assert_eq!(outs[0].1, outs[1].1, "{repo}: greedy answers differ");
        }
    }

    /// Greedy chat through the spec loader (weights = .onnx).
    #[test]
    fn onnx_greedy_chat_is_coherent() {
        let cases = [
            (SMOL, "model", "Paris"),
            (SMOL, "model_q4", "Paris"),
            (SMOL, "model_int8", "Paris"),
            (SMOL, "model_bnb4", "Paris"),
            (QWEN, "model", "Paris"),
            (QWEN, "model_q4", "Paris"),
        ];
        for (repo, variant, want) in cases {
            let dir = PathBuf::from(repo);
            let weights = dir.join(format!("onnx/{variant}.onnx"));
            if !weights.exists() || !dir.join("tokenizer.json").exists() {
                println!("SKIP: {}", weights.display());
                continue;
            }
            let artifacts = ModelArtifacts {
                model_id: repo.into(),
                config_path: Some(dir.join("config.json")),
                tokenizer_path: Some(dir.join("tokenizer.json")),
                weights_path: Some(weights),
                ..Default::default()
            };
            let llm = spec_lm(artifacts);
            let prompt = llm
                .tokenizer()
                .apply_chat_template(&[("user", "What is the capital of France?")]);
            let answer = llm.generate_text_measured_plain(&prompt, 16, 0.0).unwrap().0;
            println!("{repo} {variant}: {answer:?}");
            assert!(answer.contains(want), "{variant}: {answer:?}");
        }
    }
}
