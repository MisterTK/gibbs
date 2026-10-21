//! Model format-specific loaders and unified configuration extractors.
//!
//! This module provides trait-based abstractions for loading models from various formats
//! (HuggingFace, GGUF, ONNX, PyTorch, Safetensors) without model-specific branches. Each format
//! implements independent weight loaders and config extractors that funnel into
//! the unified [`ModelConfig`] type.

pub mod config;

pub use config::{
    ConfigExtractor, GgufConfigExtractor, HuggingFaceConfigExtractor, ModelConfig,
    OnnxConfigExtractor,
};

use std::collections::HashMap;
use std::path::Path;

/// Model loader trait abstraction for format-agnostic weight reading.
///
/// Each format (GGUF, Safetensors, ONNX) implements this trait
/// to provide a unified interface for header parsing, tensor metadata extraction,
/// and lazy weight loading.
pub trait WeightReader {
    /// Parse the format header and extract model metadata (KV pairs).
    fn parse_header(&mut self) -> Result<HashMap<String, String>, String>;

    /// List all tensors with their metadata.
    fn list_tensors(&self) -> Result<Vec<TensorMetadata>, String>;

    /// Load a single tensor by name into f32 format (with dequantization).
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String>;

    /// Load a single tensor by name in its native quantization format.
    fn load_tensor_quantized(&self, name: &str) -> Result<Vec<u8>, String>;
}

/// Tensor metadata extracted from a model file header.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Canonical tensor name (e.g., `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Dimension extents of the tensor.
    pub shape: Vec<usize>,
    /// Internal data type string (e.g., `F16`, `BF16`, `F32`, `Q4_K_M`).
    pub dtype: String,
    /// Byte offset relative to the binary data base address.
    pub byte_offset: usize,
    /// Byte length of the raw tensor payload.
    pub byte_len: usize,
}

pub mod gguf;
pub use gguf::GgufWeightReader;

pub mod ggml;
pub use ggml::GgmlWeightReader;
pub mod safetensors;
pub use safetensors::SafetensorsWeightReader;

pub mod flax;
pub mod numpy;
pub mod onnx;
pub mod pytorch_bin;
pub use onnx::OnnxWeightReader;

/// Unified model loader interface supporting multiple formats.
///
/// Detects format from file extension or explicit format parameter and returns
/// a trait object implementing the WeightReader interface.
///
/// # Supported Formats
/// - `.safetensors`: Hugging Face zero-copy memory-mapped format.
/// - `.gguf`: GGUF quantized format.
///
/// # Example
/// ```ignore
/// let loader = load_with_format("model.safetensors", None)?;
/// let tensors = loader.list_tensors()?;
pub fn load_with_format(
    path: impl AsRef<Path>,
    format: Option<&str>,
) -> Result<Box<dyn WeightReader>, String> {
    let path = path.as_ref();

    // Sniff magic bytes before extension matching for unhinted files (e.g. *.bin legacy GGML).
    if format.is_none() {
        if let Ok(mut f) = std::fs::File::open(path) {
            use std::io::Read;
            let mut magic = [0u8; 8];
            if let Ok(n) = f.read(&mut magic) {
                if n >= 4 {
                    if magic.starts_with(b"GGUF") {
                        return Ok(Box::new(GgufWeightReader::open(path)?));
                    }
                    let m = u32::from_le_bytes(magic[0..4].try_into().unwrap());
                    if m == 0x67676d6c
                        || m == 0x67676d66
                        || m == 0x67676a74
                        || magic.starts_with(b"ggml")
                        || magic.starts_with(b"ggmf")
                        || magic.starts_with(b"ggjt")
                    {
                        return Ok(Box::new(GgmlWeightReader::open(path)?));
                    }
                }
            }
        }
    }

    // Detect format from file extension if not explicitly specified.
    let detected_format =
        format.unwrap_or_else(|| path.extension().and_then(|ext| ext.to_str()).unwrap_or(""));
    match detected_format {
        "safetensors" => {
            let loader = SafetensorsWeightReader::new(path)?;
            Ok(Box::new(loader))
        }
        "gguf" => Ok(Box::new(GgufWeightReader::open(path)?)),
        "ggml" | "ggmf" | "ggjt" => Ok(Box::new(GgmlWeightReader::open(path)?)),
        "onnx" => Ok(Box::new(OnnxWeightReader::from_file(path)?)),
        other => Err(format!(
            "Unsupported format: {}. Supported formats: safetensors, gguf, onnx",
            other
        )),
    }
}

/// Tensors gathered from a non-safetensors checkpoint into the single buffer + manifest
/// [`crate::arch::load`] reads (safetensors dtype names, native bytes kept).
#[derive(Default)]
pub struct Checkpoint {
    pub manifest: crate::model::ModelManifest,
    pub data: Vec<u8>,
}

impl Checkpoint {
    /// Appends one tensor; `bytes` are little-endian, row-major, `dtype` a safetensors dtype name.
    pub fn push(
        &mut self,
        name: String,
        shape: Vec<usize>,
        dtype: &str,
        bytes: &[u8],
    ) -> Result<(), String> {
        use crate::hardware::QuantFormat;
        if self.manifest.tensors.contains_key(&name) {
            return Err(format!("tensor {name} appears twice"));
        }
        let (quant_format, size) = match dtype {
            "F16" => (QuantFormat::FP16, 2),
            "BF16" => (QuantFormat::BF16, 2),
            "F64" | "I64" | "U64" => (QuantFormat::FP32, 8),
            "F32" | "I32" | "U32" => (QuantFormat::FP32, 4),
            "I16" | "U16" => (QuantFormat::FP32, 2),
            _ => (QuantFormat::FP32, 1),
        };
        // One shape/byte check for every format: later code never sees a mismatch.
        if shape.iter().try_fold(size, |a: usize, &d| a.checked_mul(d)) != Some(bytes.len()) {
            return Err(format!(
                "tensor {name}: {} bytes do not match shape {shape:?} × {dtype}",
                bytes.len()
            ));
        }
        let vt = crate::model::VirtualTensor {
            name: name.clone(),
            shape,
            quant_format,
            byte_offset: self.data.len(),
            byte_len: bytes.len(),
            dtype: dtype.to_string(),
            shard_index: 0,
        };
        self.data.extend_from_slice(bytes);
        self.manifest.tensors.insert(name, vt);
        Ok(())
    }
}

/// Reads any weight checkpoint `from_manifest` consumes into one buffer: `*.safetensors`,
/// PyTorch pickles (`.bin`/`.pt`/`.pth`/`.ckpt`), Flax `.msgpack`, NumPy `.npz`/`.npy`, or a
/// `<file>.index.json` weight map whose shards (same directory) are concatenated.
pub fn read_checkpoint(path: &Path) -> Result<(crate::model::ModelManifest, Vec<u8>), String> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let file = name.strip_suffix(".index.json").unwrap_or(name);
    let ext = Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let read_one: fn(&Path, &mut Checkpoint) -> Result<(), String> = match ext.as_str() {
        "safetensors" => return crate::model::UniversalModelLoader::read_safetensors(path),
        "bin" | "pt" | "pth" | "ckpt" => pytorch_bin::read,
        "msgpack" => flax::read,
        "npz" | "npy" => numpy::read,
        "ot" => {
            return Err("TorchScript/tch `.ot` files are not supported".into())
        }
        "h5" | "tflite" | "mlmodel" | "mlpackage" => {
            return Err(format!(
                "'.{ext}' (TensorFlow / Core ML) weights are not supported"
            ))
        }
        other => return Err(format!("unsupported weight format '.{other}'")),
    };
    let mut ck = Checkpoint::default();
    if file.len() == name.len() {
        read_one(path, &mut ck).map_err(|e| format!("{}: {e}", path.display()))?;
        return Ok((ck.manifest, ck.data));
    }
    let json = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let index = crate::model::UniversalModelLoader::parse_safetensors_index(&json)?;
    if index.is_empty() {
        return Err(format!("{}: empty or missing weight_map", path.display()));
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    let shards: std::collections::BTreeSet<&str> = index.values().map(String::as_str).collect();
    for s in &shards {
        read_one(&dir.join(s), &mut ck).map_err(|e| format!("shard {s}: {e}"))?;
    }
    ck.manifest.shard_count = shards.len();
    if let Some((t, s)) = index
        .iter()
        .find(|(t, _)| !ck.manifest.tensors.contains_key(*t))
    {
        return Err(format!(
            "tensor {t} listed in the index is absent from shard {s}"
        ));
    }
    Ok((ck.manifest, ck.data))
}
