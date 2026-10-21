//! Legacy GGML container format loader (`ggml`, `ggmf`, `ggjt`).
//!
//! Parses original llama.cpp binary formats with magic numbers `0x67676d6c` (`ggml`),
//! `0x67676d66` (`ggmf`), and `0x67676a74` (`ggjt` v1/v2/v3).

use crate::model::loaders::{TensorMetadata, WeightReader};
use crate::model::{ModelManifest, SourceFormat, UniversalModelLoader, VirtualTensor};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Magic numbers for legacy GGML container formats.
pub const MAGIC_GGML: u32 = 0x67676d6c; // b"ggml"
pub const MAGIC_GGMF: u32 = 0x67676d66; // b"ggmf"
pub const MAGIC_GGJT: u32 = 0x67676a74; // b"ggjt"

/// Parses a legacy GGML/GGMF/GGJT byte buffer into a ModelManifest.
pub fn parse_ggml(data: &[u8]) -> Result<ModelManifest, String> {
    if data.len() < 32 {
        return Err("GGML buffer too small for header".into());
    }

    let mut cursor = 0;
    let magic = read_u32_le(data, &mut cursor)?;
    let is_ggml = magic == MAGIC_GGML || &data[0..4] == b"ggml";
    let is_ggmf = magic == MAGIC_GGMF || &data[0..4] == b"ggmf";
    let is_ggjt = magic == MAGIC_GGJT || &data[0..4] == b"ggjt";
    if !is_ggml && !is_ggmf && !is_ggjt {
        return Err(format!(
            "Invalid GGML magic: 0x{:08x} (expected ggml, ggmf, or ggjt)",
            magic
        ));
    }

    let version = if is_ggmf || is_ggjt {
        read_u32_le(data, &mut cursor)?
    } else {
        1
    };

    // Hyperparameters (7 x 32-bit integers)
    let n_vocab = read_i32_le(data, &mut cursor)?;
    let n_embd = read_i32_le(data, &mut cursor)?;
    let n_mult = read_i32_le(data, &mut cursor)?;
    let n_head = read_i32_le(data, &mut cursor)?;
    let n_layer = read_i32_le(data, &mut cursor)?;
    let n_rot = read_i32_le(data, &mut cursor)?;
    let ftype = read_i32_le(data, &mut cursor)?;

    let mut metadata = HashMap::new();
    metadata.insert("general.architecture".into(), "llama".into());
    metadata.insert("format".into(), "ggml".into());
    metadata.insert("version".into(), version.to_string());
    metadata.insert("n_vocab".into(), n_vocab.to_string());
    metadata.insert("n_embd".into(), n_embd.to_string());
    metadata.insert("n_mult".into(), n_mult.to_string());
    metadata.insert("n_head".into(), n_head.to_string());
    metadata.insert("n_layer".into(), n_layer.to_string());
    metadata.insert("n_rot".into(), n_rot.to_string());
    metadata.insert("ftype".into(), ftype.to_string());

    // Map to the GGUF architecture keys the family specs read
    metadata.insert("llama.embedding_length".into(), n_embd.to_string());
    metadata.insert("llama.block_count".into(), n_layer.to_string());
    metadata.insert("llama.attention.head_count".into(), n_head.to_string());
    metadata.insert("llama.rope.dimension_count".into(), n_rot.to_string());

    if n_vocab < 0 || n_vocab > 1_000_000 {
        return Err(format!("Unreasonable vocab size: {n_vocab}"));
    }

    // Vocabulary tokens and scores
    let mut tokens = Vec::with_capacity(n_vocab as usize);
    let mut scores = Vec::with_capacity(n_vocab as usize);
    for _ in 0..n_vocab {
        let len = read_u32_le(data, &mut cursor)? as usize;
        if cursor + len > data.len() {
            return Err("Unexpected EOF reading GGML vocabulary token".into());
        }
        let tok_str = String::from_utf8_lossy(&data[cursor..cursor + len]).to_string();
        cursor += len;
        let score = if is_ggmf || is_ggjt {
            read_f32_le(data, &mut cursor)?
        } else {
            0.0
        };
        tokens.push(tok_str);
        scores.push(score);
    }
    metadata.insert(
        "tokenizer.ggml.tokens".into(),
        serde_json::to_string(&tokens).unwrap_or_default(),
    );
    metadata.insert(
        "tokenizer.ggml.scores".into(),
        serde_json::to_string(&scores).unwrap_or_default(),
    );

    let mut tensors = HashMap::new();

    // Tensor records
    while cursor < data.len() {
        if cursor + 12 > data.len() {
            break;
        }
        let n_dims = read_u32_le(data, &mut cursor)? as usize;
        let name_len = read_u32_le(data, &mut cursor)? as usize;
        let tensor_ftype = read_u32_le(data, &mut cursor)?;

        if n_dims == 0 || n_dims > 4 {
            return Err(format!("Invalid GGML tensor dims count: {n_dims}"));
        }
        if cursor + n_dims * 4 + name_len > data.len() {
            return Err("Unexpected EOF reading GGML tensor header".into());
        }

        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u32_le(data, &mut cursor)? as usize);
        }

        let name_bytes = &data[cursor..cursor + name_len];
        cursor += name_len;
        let name = String::from_utf8_lossy(name_bytes).to_string();

        let (quant_format, dtype, block_elems, block_bytes) =
            UniversalModelLoader::ggml_type_info(tensor_ftype)?;

        let num_elements: usize = shape.iter().product();
        let byte_len = num_elements.div_ceil(block_elems) * block_bytes;

        // GGJT aligns data to 32 bytes
        if is_ggjt {
            cursor = (cursor + 31) & !31;
        }

        let byte_offset = cursor;
        if byte_offset + byte_len > data.len() {
            return Err(format!(
                "Tensor '{name}' data out of bounds (offset {byte_offset} + {byte_len} > buffer {})",
                data.len()
            ));
        }
        cursor += byte_len;

        tensors.insert(
            name.clone(),
            VirtualTensor {
                name,
                shape,
                quant_format,
                byte_offset,
                byte_len,
                dtype: dtype.to_string(),
                shard_index: 0,
            },
        );
    }

    Ok(ModelManifest {
        format: SourceFormat::Ggml,
        tensors,
        metadata,
        shard_count: 1,
        detected_architecture: "llama".into(),
    })
}

/// Reads a GGML checkpoint from disk into memory.
pub fn open(path: &Path) -> Result<(ModelManifest, Vec<u8>), String> {
    let data = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let manifest = parse_ggml(&data)?;
    Ok((manifest, data))
}
/// Extracts a Tokenizer from GGML metadata tokens.
pub fn extract_ggml_tokenizer(
    manifest: &ModelManifest,
) -> Result<crate::model::tokenizer::Tokenizer, String> {
    let tokens_json = manifest
        .metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| "No tokenizer.ggml.tokens in manifest metadata".to_string())?;
    let tokens: Vec<String> = serde_json::from_str(tokens_json)
        .map_err(|e| format!("Failed to parse GGML tokens: {e}"))?;
    Ok(crate::model::tokenizer::Tokenizer::from_tokens(tokens))
}

/// Weight reader implementation for legacy GGML models.
pub struct GgmlWeightReader {
    pub manifest: ModelManifest,
    pub data: Vec<u8>,
}

impl GgmlWeightReader {
    pub fn new(data: Vec<u8>) -> Result<Self, String> {
        let manifest = parse_ggml(&data)?;
        Ok(Self { manifest, data })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let (manifest, data) = open(path.as_ref())?;
        Ok(Self { manifest, data })
    }

    pub fn get_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.manifest.tensors.get(name).map(|t| t.shape.as_slice())
    }

    pub fn get_tensor_byte_offset(&self, name: &str) -> Option<usize> {
        self.manifest.tensors.get(name).map(|t| t.byte_offset)
    }
}

impl WeightReader for GgmlWeightReader {
    fn parse_header(&mut self) -> Result<HashMap<String, String>, String> {
        Ok(self.manifest.metadata.clone())
    }
    fn list_tensors(&self) -> Result<Vec<TensorMetadata>, String> {
        Ok(self
            .manifest
            .tensors
            .values()
            .map(|t| TensorMetadata {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype.clone(),
                byte_offset: t.byte_offset,
                byte_len: t.byte_len,
            })
            .collect())
    }

    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        UniversalModelLoader::extract_tensor_from_manifest(&self.manifest, &self.data, name)
    }

    fn load_tensor_quantized(&self, name: &str) -> Result<Vec<u8>, String> {
        let tensor = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor not found: {name}"))?;
        let end = tensor.byte_offset + tensor.byte_len;
        if end > self.data.len() {
            return Err(format!("Tensor '{name}' slice out of bounds"));
        }
        Ok(self.data[tensor.byte_offset..end].to_vec())
    }
}

fn read_u32_le(data: &[u8], cursor: &mut usize) -> Result<u32, String> {
    if *cursor + 4 > data.len() {
        return Err("Unexpected EOF reading u32".into());
    }
    let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_i32_le(data: &[u8], cursor: &mut usize) -> Result<i32, String> {
    if *cursor + 4 > data.len() {
        return Err("Unexpected EOF reading i32".into());
    }
    let val = i32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_f32_le(data: &[u8], cursor: &mut usize) -> Result<f32, String> {
    if *cursor + 4 > data.len() {
        return Err("Unexpected EOF reading f32".into());
    }
    let val = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_mock_ggml_container(magic: u32, version: Option<u32>, aligned_32: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&magic.to_le_bytes());
        if let Some(v) = version {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // Hyperparameters:
        // n_vocab = 2, n_embd = 4, n_mult = 1, n_head = 2, n_layer = 1, n_rot = 2, ftype = 0 (FP32)
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&0i32.to_le_bytes());

        // Vocab: 2 tokens: "foo", "bar"
        for token in ["foo", "bar"] {
            buf.extend_from_slice(&(token.len() as u32).to_le_bytes());
            buf.extend_from_slice(token.as_bytes());
            // ggmf and ggjt have f32 score
            if version.is_some() {
                buf.extend_from_slice(&0.5f32.to_le_bytes());
            }
        }

        // Tensor 1: "weight_a", 1-D len 4, ftype 0 (FP32)
        let tensor_name = "weight_a";
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        buf.extend_from_slice(&(tensor_name.len() as u32).to_le_bytes()); // name_len
        buf.extend_from_slice(&0u32.to_le_bytes()); // ftype = 0 (FP32)
        buf.extend_from_slice(&4u32.to_le_bytes()); // ne[0] = 4
        buf.extend_from_slice(tensor_name.as_bytes());
        if aligned_32 {
            let pad = (32 - (buf.len() % 32)) % 32;
            buf.extend(std::iter::repeat(0u8).take(pad));
        }
        let vals_a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        for v in vals_a {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // Tensor 2: "weight_b", 2-D 2x2, ftype 0 (FP32)
        let tensor_b_name = "weight_b";
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        buf.extend_from_slice(&(tensor_b_name.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // ne[0] = 2
        buf.extend_from_slice(&2u32.to_le_bytes()); // ne[1] = 2
        buf.extend_from_slice(tensor_b_name.as_bytes());
        if aligned_32 {
            let pad = (32 - (buf.len() % 32)) % 32;
            buf.extend(std::iter::repeat(0u8).take(pad));
        }
        let vals_b: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        for v in vals_b {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_parse_ggml_ggmf_ggjt() {
        // 1. Unversioned GGML
        let ggml_buf = build_mock_ggml_container(MAGIC_GGML, None, false);
        let m_ggml = parse_ggml(&ggml_buf).unwrap();
        assert_eq!(m_ggml.format, SourceFormat::Ggml);
        assert_eq!(m_ggml.tensors.len(), 2);
        assert_eq!(m_ggml.metadata["n_vocab"], "2");
        assert_eq!(m_ggml.metadata["n_embd"], "4");
        let vals_a =
            UniversalModelLoader::extract_tensor_from_manifest(&m_ggml, &ggml_buf, "weight_a")
                .unwrap();
        assert_eq!(vals_a, vec![1.0, 2.0, 3.0, 4.0]);

        // 2. Versioned GGMF
        let ggmf_buf = build_mock_ggml_container(MAGIC_GGMF, Some(1), false);
        let m_ggmf = parse_ggml(&ggmf_buf).unwrap();
        assert_eq!(m_ggmf.format, SourceFormat::Ggml);
        assert_eq!(m_ggmf.tensors.len(), 2);
        let vals_b =
            UniversalModelLoader::extract_tensor_from_manifest(&m_ggmf, &ggmf_buf, "weight_b")
                .unwrap();
        assert_eq!(vals_b, vec![5.0, 6.0, 7.0, 8.0]);

        // 3. Versioned GGJT with 32-byte alignment
        let ggjt_buf = build_mock_ggml_container(MAGIC_GGJT, Some(3), true);
        let m_ggjt = parse_ggml(&ggjt_buf).unwrap();
        assert_eq!(m_ggjt.format, SourceFormat::Ggml);
        assert_eq!(m_ggjt.tensors.len(), 2);
        let tensor_a = &m_ggjt.tensors["weight_a"];
        assert_eq!(
            tensor_a.byte_offset % 32,
            0,
            "GGJT tensor data must be 32-byte aligned"
        );
        let vals_a_jt =
            UniversalModelLoader::extract_tensor_from_manifest(&m_ggjt, &ggjt_buf, "weight_a")
                .unwrap();
        assert_eq!(vals_a_jt, vec![1.0, 2.0, 3.0, 4.0]);

        // Auto-detect format test
        assert_eq!(
            UniversalModelLoader::detect_format(&ggml_buf).unwrap(),
            SourceFormat::Ggml
        );
        assert_eq!(
            UniversalModelLoader::detect_format(&ggmf_buf).unwrap(),
            SourceFormat::Ggml
        );
        assert_eq!(
            UniversalModelLoader::detect_format(&ggjt_buf).unwrap(),
            SourceFormat::Ggml
        );

        // WeightReader test
        let reader = GgmlWeightReader::new(ggjt_buf).unwrap();
        assert_eq!(
            reader.load_tensor("weight_a").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            reader.load_tensor("weight_b").unwrap(),
            vec![5.0, 6.0, 7.0, 8.0]
        );
        // Vocabulary extraction test
        let tok = extract_ggml_tokenizer(&m_ggml).unwrap();
        assert_eq!(tok.encode("foo"), vec![0]);
        assert_eq!(tok.encode("bar"), vec![1]);

        // Sniffing test: writing ggjt to a .bin file
        let dir = tempfile::tempdir().unwrap();
        let bin_path = dir.path().join("ggml-model-q4_0.bin");
        std::fs::write(
            &bin_path,
            &build_mock_ggml_container(MAGIC_GGJT, Some(3), true),
        )
        .unwrap();
        let bin_reader = crate::model::loaders::load_with_format(&bin_path, None).unwrap();
        assert_eq!(
            bin_reader.load_tensor("weight_a").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }
}
