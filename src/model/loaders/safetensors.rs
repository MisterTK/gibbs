//! Safetensors format loader with memory-mapped zero-copy tensor access.
//!
//! Parses the JSON header to extract tensor metadata without loading the entire file.
//! Tensor data is accessed lazily via memory mapping; dequantization to f32 is done on-demand.

use super::{TensorMetadata, WeightReader};
use crate::model::{ModelManifest, UniversalModelLoader};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Safetensors format loader over a memory-mapped file; decoding shares
/// [`UniversalModelLoader::extract_tensor_from_manifest`] (all dtypes, pre-quantized weights).
pub struct SafetensorsWeightReader {
    mmap: Mmap,
    manifest: ModelManifest,
}

impl SafetensorsWeightReader {
    /// Load Safetensors file from path. Parses header without loading weight data.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("Failed to mmap file: {}", e))? };
        let mut manifest = UniversalModelLoader::parse_safetensors(&mmap)?;
        crate::model::prequant::mlx::attach(&mut manifest, path.as_ref())?;
        Ok(Self { mmap, manifest })
    }
}

impl WeightReader for SafetensorsWeightReader {
    fn parse_header(&mut self) -> Result<HashMap<String, String>, String> {
        // Safetensors doesn't have a dedicated KV header like GGUF; return empty.
        Ok(HashMap::new())
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
        UniversalModelLoader::extract_tensor_from_manifest(&self.manifest, &self.mmap, name)
    }

    fn load_tensor_quantized(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor {} not found in Safetensors", name))?;
        Ok(self.mmap[t.byte_offset..t.byte_offset + t.byte_len].to_vec())
    }
}
