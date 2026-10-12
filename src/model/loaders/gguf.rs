use super::{TensorMetadata, WeightReader};
use crate::hardware::QuantFormat;
use crate::model::{ModelManifest, SourceFormat};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// GGUF format loader with lazy tensor dequantization.
///
/// Parses GGUF header (KV pairs) and tensor metadata without loading the entire file,
/// deferring weight dequantization until `load_tensor` is called.
#[derive(Debug)]
pub struct GgufWeightReader {
    /// Raw file data (memory-mapped or buffered).
    data: Vec<u8>,
    /// Parsed header KV metadata.
    metadata: HashMap<String, String>,
    /// Tensor list with offsets and metadata.
    tensors: HashMap<String, GgufTensorInfo>,
    #[allow(dead_code)]
    version: u32,
}

impl GgufWeightReader {
    /// Create a new GGUF reader from raw bytes.
    pub fn new(data: Vec<u8>) -> Result<Self, String> {
        if data.len() < 24 {
            return Err("GGUF buffer too small".into());
        }
        if &data[0..4] != b"GGUF" {
            return Err("Invalid GGUF magic bytes".into());
        }
        let h = parse_gguf_header(&data).map_err(|e| e.to_string())?;
        Ok(GgufWeightReader {
            data,
            metadata: h.metadata,
            tensors: h.tensors.into_iter().map(|t| (t.name.clone(), t)).collect(),
            version: h.version,
        })
    }

    /// Reads the GGUF at `path`, or the whole split set it belongs to (`<stem>-00001-of-00003.gguf`):
    /// shards are appended to one buffer and every tensor offset shifted past the shards before
    /// it; the first shard's KV metadata describes the model (see [`open`] for the validated form).
    pub fn open(path: &Path) -> Result<Self, String> {
        let read = |p: &Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
        let Some(paths) = shard_paths(path) else {
            return Self::new(read(path)?).map_err(|e| format!("{}: {e}", path.display()));
        };
        let mut merged: Option<Self> = None;
        for p in &paths {
            let shard = Self::new(read(p)?).map_err(|e| format!("{}: {e}", p.display()))?;
            let Some(m) = merged.as_mut() else {
                merged = Some(shard);
                continue;
            };
            let base = m.data.len();
            m.data.extend_from_slice(&shard.data);
            for (name, mut t) in shard.tensors {
                t.byte_offset += base;
                if m.tensors.insert(name.clone(), t).is_some() {
                    return Err(format!(
                        "{}: tensor '{name}' appears in two shards",
                        p.display()
                    ));
                }
            }
        }
        merged.ok_or_else(|| "split GGUF with no shards".into())
    }

    /// The file bytes (a split set's shards back to back) the tensor offsets index.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Retrieve shape for a specific tensor.
    pub fn get_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Retrieve byte offset for a specific tensor.
    pub fn get_tensor_byte_offset(&self, name: &str) -> Option<usize> {
        self.tensors.get(name).map(|t| t.byte_offset)
    }
    /// Returns an iterator over all tensor names in the GGUF file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// GGML type id → (format, type name, elements per block, bytes per block); shared with `parse_gguf`.
    fn get_quant_info(raw_type: u32) -> Result<(QuantFormat, &'static str, usize, usize), String> {
        crate::model::UniversalModelLoader::ggml_type_info(raw_type)
    }
}

impl WeightReader for GgufWeightReader {
    fn parse_header(&mut self) -> Result<std::collections::HashMap<String, String>, String> {
        Ok(self.metadata.clone())
    }

    fn list_tensors(&self) -> Result<Vec<TensorMetadata>, String> {
        let mut result = Vec::new();
        for (name, info) in &self.tensors {
            let num_elements: usize = info
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| format!("Tensor '{}' shape overflows usize", name))?;
            let (_quant_format, dtype, block_elems, block_bytes) =
                Self::get_quant_info(info.raw_type)?;
            let byte_len = num_elements.div_ceil(block_elems) * block_bytes;

            result.push(TensorMetadata {
                name: name.clone(),
                shape: info.shape.clone(),
                dtype: dtype.to_string(),
                byte_offset: info.byte_offset,
                byte_len,
            });
        }
        Ok(result)
    }

    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor not found: {}", name))?;

        let (quant_format, dtype, block_elems, block_bytes) = Self::get_quant_info(info.raw_type)?;

        let num_elements: usize = info
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("Tensor '{}' shape overflows usize", name))?;

        let byte_len = num_elements.div_ceil(block_elems) * block_bytes;
        let end = info
            .byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| format!("Tensor '{}' offset overflows usize", name))?;
        if end > self.data.len() {
            return Err(format!(
                "Tensor '{}' data out of bounds (offset {} + {} bytes > buffer {})",
                name,
                info.byte_offset,
                byte_len,
                self.data.len()
            ));
        }

        let raw = &self.data[info.byte_offset..end];
        let mut out: Vec<f32> = match quant_format {
            QuantFormat::FP32 => crate::model::decode_dtype(raw, dtype)?,
            QuantFormat::FP16 => raw
                .chunks_exact(2)
                .map(|c| crate::model::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            QuantFormat::BF16 => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            f => crate::model::layers::dequantize_gguf_blocks(f, raw)
                .ok_or_else(|| format!("GGML type {dtype} has no block decoder"))?,
        };
        out.truncate(num_elements);
        Ok(out)
    }

    fn load_tensor_quantized(&self, name: &str) -> Result<Vec<u8>, String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor not found: {}", name))?;

        let num_elements: usize = info
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("Tensor '{}' shape overflows usize", name))?;
        let (_quant_format, _dtype, block_elems, block_bytes) =
            Self::get_quant_info(info.raw_type)?;
        let byte_len = num_elements.div_ceil(block_elems) * block_bytes;

        let end = info
            .byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| format!("Tensor '{}' offset overflows usize", name))?;
        if end > self.data.len() {
            return Err("Tensor data out of bounds".into());
        }

        Ok(self.data[info.byte_offset..end].to_vec())
    }
}

/// `general.architecture` of the GGUF at `path` (first shard of a split set), `None` if unreadable.
pub fn architecture(path: &Path) -> Option<String> {
    // Only the first shard of a split set carries the KVs.
    let first = shard_paths(path).map_or_else(|| path.to_path_buf(), |s| s[0].clone());
    let file = std::fs::File::open(first).ok()?;
    // SAFETY: read-only mapping of a cache file nothing rewrites while the model is loaded.
    let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    let manifest = crate::model::UniversalModelLoader::parse_gguf(&map).ok()?;
    manifest.metadata.get("general.architecture").cloned()
}

/// A GGUF checkpoint's bytes: the mapped file, or a llama.cpp split set read into one buffer.
pub enum GgufData {
    Map(memmap2::Mmap),
    Buf(Vec<u8>),
}

impl std::ops::Deref for GgufData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Map(m) => m,
            Self::Buf(b) => b,
        }
    }
}

/// Every shard file of the split set `path` (`<prefix>-00001-of-00003.gguf`, any member) belongs to,
/// in order; `None` for a single-file GGUF.
pub fn shard_paths(path: &Path) -> Option<Vec<PathBuf>> {
    let name = path.file_name()?.to_str()?;
    let (prefix, _, total, w) = crate::model::hub::shard_of(name, ".gguf")?;
    let dir = path.parent().unwrap_or(Path::new("."));
    Some(
        (1..=total)
            .map(|n| dir.join(format!("{prefix}-{n:0w$}-of-{total:0w$}.gguf")))
            .collect(),
    )
}

/// Opens a GGUF checkpoint as one weight set. A single file is mapped; a llama.cpp split set
/// (`gguf-split`: first shard carries the KV metadata, tensors are spread over the shards, each
/// shard's `split.no` / `split.count` / `split.tensors.count` KVs say where it sits) is read into
/// one contiguous buffer with every tensor offset shifted past the shards before it, so callers
/// index `data` exactly as for a single file.
// ponytail: a split set costs its size in resident memory (one buffer); map each shard and index
// tensors per shard if a split model must load beyond RAM.
pub fn open(path: &Path) -> Result<(ModelManifest, GgufData), String> {
    use std::io::Read;
    let fail = |p: &Path, e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
    let Some(paths) = shard_paths(path) else {
        let file = std::fs::File::open(path).map_err(|e| fail(path, &e))?;
        // SAFETY: read-only mapping of a cache file nothing rewrites while the model is loaded.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| fail(path, &e))?;
        let manifest =
            crate::model::UniversalModelLoader::parse_gguf(&map).map_err(|e| fail(path, &e))?;
        return Ok((manifest, GgufData::Map(map)));
    };
    let mut size = 0usize;
    for p in &paths {
        let len = std::fs::metadata(p)
            .map_err(|e| format!("split GGUF shard {}: {e}", p.display()))?
            .len();
        size += len as usize;
    }
    let mut buf = Vec::with_capacity(size);
    let mut manifest = ModelManifest {
        format: SourceFormat::Gguf,
        shard_count: paths.len(),
        ..Default::default()
    };
    for (i, p) in paths.iter().enumerate() {
        let base = buf.len();
        std::fs::File::open(p)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .map_err(|e| fail(p, &e))?;
        let shard = crate::model::UniversalModelLoader::parse_gguf(&buf[base..])
            .map_err(|e| fail(p, &e))?;
        let kv = |k: &str| shard.metadata.get(k).and_then(|v| v.parse::<usize>().ok());
        if kv("split.no").is_some_and(|n| n != i)
            || kv("split.count").is_some_and(|c| c != paths.len())
        {
            return Err(format!(
                "{}: split.no {:?} / split.count {:?} disagree with its name (shard {} of {})",
                p.display(),
                kv("split.no"),
                kv("split.count"),
                i + 1,
                paths.len()
            ));
        }
        for (name, mut t) in shard.tensors {
            t.byte_offset += base;
            if manifest.tensors.insert(name.clone(), t).is_some() {
                return Err(format!(
                    "{}: tensor '{name}' appears in two shards",
                    p.display()
                ));
            }
        }
        // The first shard's KVs win; later shards only repeat the split ones.
        for (k, v) in shard.metadata {
            manifest.metadata.entry(k).or_insert(v);
        }
    }
    if let Some(n) = manifest
        .metadata
        .get("split.tensors.count")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n != manifest.tensors.len())
    {
        return Err(format!(
            "split GGUF {}: split.tensors.count is {n}, the {} shards hold {}",
            path.display(),
            paths.len(),
            manifest.tensors.len()
        ));
    }
    Ok((manifest, GgufData::Buf(buf)))
}

// ============================================================================
// HEADER PARSING (shared by the file reader and the remote range-request path)
// ============================================================================

/// One tensor-info entry; `byte_offset` is absolute within the file once the header is parsed.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// GGML order: fastest-varying dimension first.
    pub shape: Vec<usize>,
    pub raw_type: u32,
    pub byte_offset: usize,
}

/// The KV scalars/strings and tensor infos at the start of a GGUF file (array payloads are skipped).
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: HashMap<String, String>,
    pub tensors: Vec<GgufTensorInfo>,
    /// First tensor-data byte: the header end rounded up to `general.alignment`.
    pub data_start: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    /// The buffer ends inside the header; the header is at least this many bytes long.
    NeedMore(usize),
    Invalid(String),
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NeedMore(n) => write!(f, "GGUF header needs at least {n} bytes"),
            Self::Invalid(e) => f.write_str(e),
        }
    }
}

struct Cur<'a> {
    d: &'a [u8],
    o: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], HeaderError> {
        let end = self
            .o
            .checked_add(n)
            .ok_or_else(|| HeaderError::Invalid("GGUF length overflows".into()))?;
        let s = self.d.get(self.o..end).ok_or(HeaderError::NeedMore(end))?;
        self.o = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, HeaderError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, HeaderError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, HeaderError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// A count, length or offset: u32 in GGUF v1, u64 after.
    fn word(&mut self, version: u32) -> Result<usize, HeaderError> {
        let v = if version == 1 {
            u64::from(self.u32()?)
        } else {
            self.u64()?
        };
        usize::try_from(v).map_err(|_| HeaderError::Invalid("GGUF count overflows".into()))
    }
    fn string(&mut self, version: u32) -> Result<String, HeaderError> {
        let n = self.word(version)?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

/// Parses the header at the start of `bytes` (a prefix of the file is enough). `NeedMore(n)` says
/// the header is at least `n` bytes long, so the caller can fetch more and retry.
pub fn parse_gguf_header(bytes: &[u8]) -> Result<GgufHeader, HeaderError> {
    let mut c = Cur { d: bytes, o: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(HeaderError::Invalid("Invalid GGUF magic bytes".into()));
    }
    let version = c.u32()?;
    if !(1..=3).contains(&version) {
        return Err(HeaderError::Invalid(format!(
            "Unsupported GGUF version: {version}"
        )));
    }
    let (tensor_count, kv_count) = (c.word(version)?, c.word(version)?);
    let mut metadata = HashMap::new();
    for _ in 0..kv_count {
        let key = c.string(version)?;
        let ty = c.u32()?;
        match ty {
            0 | 1 | 7 => {
                metadata.insert(key, c.take(1)?[0].to_string());
            }
            2 | 3 => {
                metadata.insert(key, c.u16()?.to_string());
            }
            4 | 5 => {
                metadata.insert(key, c.u32()?.to_string());
            }
            6 => {
                metadata.insert(key, f32::from_bits(c.u32()?).to_string());
            }
            10 | 11 => {
                metadata.insert(key, c.u64()?.to_string());
            }
            12 => {
                c.take(8)?;
            }
            8 => {
                metadata.insert(key, c.string(version)?);
            }
            9 => {
                let elem = c.u32()?;
                let n = c.word(version)?;
                let width = match elem {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    8 => {
                        for _ in 0..n {
                            c.string(version)?;
                        }
                        continue;
                    }
                    t => {
                        return Err(HeaderError::Invalid(format!(
                            "GGUF array of type {t} in '{key}'"
                        )))
                    }
                };
                c.take(
                    n.checked_mul(width)
                        .ok_or_else(|| HeaderError::Invalid("GGUF array overflows".into()))?,
                )?;
            }
            t => {
                return Err(HeaderError::Invalid(format!(
                    "GGUF value type {t} in '{key}'"
                )))
            }
        }
    }
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = c.string(version)?;
        let n_dims = c.u32()? as usize;
        let mut shape = Vec::new();
        for _ in 0..n_dims {
            shape.push(c.word(version)?);
        }
        let raw_type = c.u32()?;
        let byte_offset = c.word(version)?;
        tensors.push(GgufTensorInfo {
            name,
            shape,
            raw_type,
            byte_offset,
        });
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&a| a > 0)
        .unwrap_or(32);
    let data_start = c.o.div_ceil(alignment) * alignment;
    for t in &mut tensors {
        t.byte_offset = data_start.saturating_add(t.byte_offset);
    }
    Ok(GgufHeader {
        version,
        metadata,
        tensors,
        data_start,
    })
}

/// A GGUF v3 file of F32 tensors (`name`, GGML dims fastest first, values) with the given KVs
/// (`name`, GGUF value type, payload); tensor data 32-byte aligned.
#[cfg(test)]
pub(crate) fn test_gguf_file(
    kv: &[(&str, u32, Vec<u8>)],
    tensors: &[(&str, &[usize], &[f32])],
) -> Vec<u8> {
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, t, payload) in kv {
        s(&mut b, k);
        b.extend_from_slice(&t.to_le_bytes());
        b.extend_from_slice(payload);
    }
    let mut off = 0u64;
    for (n, dims, v) in tensors {
        assert_eq!(dims.iter().product::<usize>(), v.len(), "{n}");
        s(&mut b, n);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            b.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&off.to_le_bytes());
        off += (v.len() as u64 * 4).div_ceil(32) * 32;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    for (_, _, v) in tensors {
        for x in *v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::f16_to_f32;

    fn deq(format: QuantFormat, blk: &[u8]) -> Vec<f32> {
        crate::model::layers::dequantize_gguf_blocks(format, blk).unwrap()
    }

    #[test]
    fn test_f16_conversion_accuracy() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x3555), 0.33325195);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_dequant_q4_0_block_accuracy() {
        let mut blk = vec![0u8; 18];
        for i in 0..18 {
            blk[i] = ((i * 73 + 29) % 256) as u8;
        }
        blk[0..2].copy_from_slice(&0x211Fu16.to_le_bytes());

        let deq = deq(QuantFormat::GGUF_Q4_0, &blk);
        assert_eq!(deq.len(), 32);
        assert!((deq[0] - 0.070015).abs() < 1e-4);
        assert!((deq[17] - 0.070015).abs() < 1e-4);
        assert!((deq[31] - 0.070015).abs() < 1e-4);
    }

    #[test]
    fn test_dequant_q4_1_block_accuracy() {
        let mut blk = vec![0u8; 20];
        for i in 0..20 {
            blk[i] = ((i * 73 + 29) % 256) as u8;
        }
        blk[0..2].copy_from_slice(&0x211Fu16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x9D1Fu16.to_le_bytes());

        let deq = deq(QuantFormat::GGUF_Q4_1, &blk);
        assert_eq!(deq.len(), 32);
        assert!((deq[0] - 0.00500107).abs() < 1e-4);
        assert!((deq[17] - 0.075016).abs() < 1e-4);
        assert!((deq[31] - 0.075016).abs() < 1e-4);
    }

    #[test]
    fn test_dequant_q5_0_block_accuracy() {
        let mut blk = vec![0u8; 22];
        for i in 0..22 {
            blk[i] = ((i * 73 + 29) % 256) as u8;
        }
        blk[0..2].copy_from_slice(&0x211Fu16.to_le_bytes());

        let deq = deq(QuantFormat::GGUF_Q5_0, &blk);
        assert_eq!(deq.len(), 32);
        assert!((deq[0] - 0.0300064).abs() < 1e-4);
        assert!((deq[17] - (-0.150032)).abs() < 1e-4);
        assert!((deq[31] - 0.0100021).abs() < 1e-4);
    }

    #[test]
    fn test_dequant_q5_1_block_accuracy() {
        let mut blk = vec![0u8; 24];
        for i in 0..24 {
            blk[i] = ((i * 73 + 29) % 256) as u8;
        }
        blk[0..2].copy_from_slice(&0x211Fu16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x9D1Fu16.to_le_bytes());

        let deq = deq(QuantFormat::GGUF_Q5_1, &blk);
        assert_eq!(deq.len(), 32);
        assert!((deq[0] - 0.205044).abs() < 1e-4);
        assert!((deq[17] - 0.255054).abs() < 1e-4);
        assert!((deq[31] - 0.0950203).abs() < 1e-4);
    }

    #[test]
    fn test_dequant_q8_0_block_accuracy() {
        let mut blk = vec![0u8; 34];
        for i in 0..34 {
            blk[i] = ((i * 73 + 29) % 256) as u8;
        }
        blk[0..2].copy_from_slice(&0x211Fu16.to_le_bytes());

        let deq = deq(QuantFormat::GGUF_Q8_0, &blk);
        assert_eq!(deq.len(), 32);
        assert!((deq[0] - (-0.810173)).abs() < 1e-4);
        assert!((deq[17] - (-1.20026)).abs() < 1e-4);
        assert!((deq[31] - (-1.22026)).abs() < 1e-4);
    }

    #[test]
    fn test_dequant_q4_k_m_block_accuracy() {
        let mut blk = vec![0u8; 144];
        for i in 0..144 {
            blk[i] = ((i * 73 + 29) % 256) as u8;
        }
        blk[0..2].copy_from_slice(&0x211Fu16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x1D1Fu16.to_le_bytes());

        let deq = deq(QuantFormat::GGUF_Q4_K_M, &blk);
        assert_eq!(deq.len(), 256);
        assert!((deq[0] - (-0.0550117)).abs() < 1e-4);
        assert!((deq[17] - (-0.125027)).abs() < 1e-4);
        assert!((deq[31] - (-0.145031)).abs() < 1e-4);
        assert!((deq[32] - 0.770164).abs() < 1e-4);
        assert!((deq[100] - 4.2009).abs() < 1e-4);
        assert!((deq[255] - 0.450096).abs() < 1e-4);
    }

    #[test]
    fn test_multidimensional_tensor_shape_and_offsets() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        let write_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        write_str(&mut buf, "general.alignment");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&64u32.to_le_bytes());

        write_str(&mut buf, "test_3d_q4_0");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let pad = (64 - (buf.len() % 64)) % 64;
        buf.extend(std::iter::repeat(0u8).take(pad));
        let expected_offset = buf.len();

        let mut q4_data = vec![0u8; 72];
        for b in 0..4 {
            q4_data[b * 18..b * 18 + 2].copy_from_slice(&0x211Fu16.to_le_bytes());
            for i in 0..16 {
                q4_data[b * 18 + 2 + i] = 0x88;
            }
        }
        buf.extend_from_slice(&q4_data);

        let reader = GgufWeightReader::new(buf).unwrap();
        assert_eq!(
            reader.get_tensor_shape("test_3d_q4_0"),
            Some(&[32, 2, 2][..])
        );
        assert_eq!(
            reader.get_tensor_byte_offset("test_3d_q4_0"),
            Some(expected_offset)
        );

        let t_list = reader.list_tensors().unwrap();
        assert_eq!(t_list.len(), 1);
        assert_eq!(t_list[0].shape, vec![32, 2, 2]);
        assert_eq!(t_list[0].byte_len, 72);
        assert_eq!(t_list[0].byte_offset, expected_offset);

        let vals = reader.load_tensor("test_3d_q4_0").unwrap();
        assert_eq!(vals.len(), 128);
        for v in vals {
            assert_eq!(v, 0.0);
        }
    }

    /// A GGUF v3 file of 1-D F32 tensors (`name`, values) with the given KVs.
    fn gguf_file(kv: &[(&str, u32, Vec<u8>)], tensors: &[(&str, &[f32])]) -> Vec<u8> {
        let dims: Vec<[usize; 1]> = tensors.iter().map(|(_, v)| [v.len()]).collect();
        let all: Vec<(&str, &[usize], &[f32])> = tensors
            .iter()
            .zip(&dims)
            .map(|((n, v), d)| (*n, &d[..], *v))
            .collect();
        test_gguf_file(kv, &all)
    }

    /// A llama.cpp `gguf-split` set (first shard carries the KVs, `split.no` is 0-based) reads as
    /// one weight set whose tensor bytes equal the unsplit file's, through both GGUF readers.
    #[test]
    fn split_set_reads_as_one_weight_set() {
        let a: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..13).map(|i| (i * i) as f32).collect();
        let c: Vec<f32> = vec![3.25, -7.5, 0.125];
        let all: [(&str, &[f32]); 3] = [("a", &a), ("b", &b), ("c", &c)];
        let arch = {
            let mut p = (5u64).to_le_bytes().to_vec();
            p.extend_from_slice(b"llama");
            p
        };
        let dir = tempfile::tempdir().unwrap();
        let whole = dir.path().join("whole.gguf");
        std::fs::write(
            &whole,
            gguf_file(&[("general.architecture", 8, arch.clone())], &all),
        )
        .unwrap();
        for (i, t) in all.iter().enumerate() {
            let mut kv = vec![
                ("split.no", 2, (i as u16).to_le_bytes().to_vec()),
                ("split.count", 2, 3u16.to_le_bytes().to_vec()),
                ("split.tensors.count", 5, 3i32.to_le_bytes().to_vec()),
            ];
            if i == 0 {
                kv.insert(0, ("general.architecture", 8, arch.clone()));
            }
            std::fs::write(
                dir.path().join(format!("m-{:05}-of-00003.gguf", i + 1)),
                gguf_file(&kv, &[*t]),
            )
            .unwrap();
        }
        let first = dir.path().join("m-00001-of-00003.gguf");
        let (whole_m, whole_d) = open(&whole).unwrap();
        for p in [&first, &dir.path().join("m-00002-of-00003.gguf")] {
            let (m, d) = open(p).unwrap();
            assert_eq!(m.shard_count, 3);
            assert_eq!(m.metadata["general.architecture"], "llama");
            for (name, v) in all {
                assert_eq!(m.extract_tensor_f32(&d, name).unwrap(), v, "{name}");
                let (t, w) = (&m.tensors[name], &whole_m.tensors[name]);
                assert_eq!(
                    &d[t.byte_offset..][..t.byte_len],
                    &whole_d[w.byte_offset..][..w.byte_len]
                );
            }
        }
        assert_eq!(architecture(&first).as_deref(), Some("llama"));
        let r = GgufWeightReader::open(&first).unwrap();
        for (name, v) in all {
            assert_eq!(r.load_tensor(name).unwrap(), v, "{name}");
        }
        // A missing shard, or a shard that is not where its name says, is an error naming it.
        std::fs::copy(
            dir.path().join("m-00003-of-00003.gguf"),
            dir.path().join("m-00002-of-00003.gguf"),
        )
        .unwrap();
        let err = open(&first).err().unwrap();
        assert!(
            err.contains("m-00002-of-00003.gguf") && err.contains("split.no"),
            "{err}"
        );
        std::fs::remove_file(dir.path().join("m-00002-of-00003.gguf")).unwrap();
        assert!(open(&first)
            .err()
            .unwrap()
            .contains("m-00002-of-00003.gguf"));
    }
    #[test]
    fn test_gguf_f64_and_i64_tensors() {
        let s = |b: &mut Vec<u8>, v: &str| {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        };
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // 0 KVs

        // Tensor 1: "f64_tensor", 1-D len 2, type 28 (F64), offset 0
        s(&mut b, "f64_tensor");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&28u32.to_le_bytes()); // type 28 = F64
        b.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // Tensor 2: "i64_tensor", 1-D len 2, type 27 (I64), offset 32
        s(&mut b, "i64_tensor");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&27u32.to_le_bytes()); // type 27 = I64
        b.extend_from_slice(&32u64.to_le_bytes()); // offset 32

        b.resize(b.len().div_ceil(32) * 32, 0);

        // Payload 1: 2 x f64: 1.25, -42.5
        b.extend_from_slice(&1.25f64.to_le_bytes());
        b.extend_from_slice(&(-42.5f64).to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);

        // Payload 2: 2 x i64: 123456789i64, -987654321i64
        b.extend_from_slice(&123456789i64.to_le_bytes());
        b.extend_from_slice(&(-987654321i64).to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);

        let loader = crate::model::UniversalModelLoader::from_bytes(&b).unwrap();
        let f64_vals = loader
            .manifest
            .extract_tensor_f32(&b, "f64_tensor")
            .unwrap();
        assert_eq!(f64_vals, vec![1.25, -42.5]);

        let i64_vals = loader
            .manifest
            .extract_tensor_f32(&b, "i64_tensor")
            .unwrap();
        assert_eq!(i64_vals, vec![123456789.0, -987654321.0]);

        let reader = GgufWeightReader::new(b).unwrap();
        assert_eq!(reader.load_tensor("f64_tensor").unwrap(), vec![1.25, -42.5]);
        assert_eq!(
            reader.load_tensor("i64_tensor").unwrap(),
            vec![123456789.0, -987654321.0]
        );
    }

    /// One-tensor GGUF v3 file `t` of ggml type `ty`, dims `ne` (fastest first), payload `raw`.
    fn typed_gguf(ty: u32, ne: &[u64], raw: &[u8]) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push(b't');
        b.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for d in ne {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.extend_from_slice(raw);
        b.resize(b.len().div_ceil(32) * 32, 0);
        b
    }

    /// Both GGUF readers decode `raw` of ggml type `ty` to `want`, and the f32 matrix path does not
    /// hand the bytes out as a zero-copy slice.
    fn assert_decodes(ty: u32, ne: &[u64], raw: &[u8], want: &[f32]) {
        let b = typed_gguf(ty, ne, raw);
        let loader = crate::model::UniversalModelLoader::from_bytes(&b).unwrap();
        assert_eq!(
            loader.manifest.extract_tensor_f32(&b, "t").unwrap(),
            want,
            "type {ty}"
        );
        assert_eq!(
            GgufWeightReader::new(b.clone())
                .unwrap()
                .load_tensor("t")
                .unwrap(),
            want
        );
        let q = loader
            .manifest
            .extract_tensor_as_quantized(&b, "t", QuantFormat::FP32)
            .unwrap();
        assert!(
            !matches!(
                q,
                crate::model::layers::QuantizedWeight::ZeroCopySlice { .. }
            ),
            "type {ty}"
        );
    }

    #[test]
    fn gguf_integer_types_decode_to_f32() {
        let le = |v: &[i64], n: usize| -> Vec<u8> {
            v.iter()
                .flat_map(|x| x.to_le_bytes()[..n].to_vec())
                .collect()
        };
        let i8s = [-128i64, -1, 0, 127];
        assert_decodes(24, &[4], &le(&i8s, 1), &i8s.map(|x| x as f32));
        let i16s = [-32768i64, -2, 300, 32767];
        assert_decodes(25, &[4], &le(&i16s, 2), &i16s.map(|x| x as f32));
        let i32s = [i32::MIN as i64, -7, 100_000, i32::MAX as i64];
        assert_decodes(26, &[2, 2], &le(&i32s, 4), &i32s.map(|x| x as f32));
        let i64s = [i64::MIN, -(1 << 40), 1 << 40, i64::MAX];
        assert_decodes(27, &[4], &le(&i64s, 8), &i64s.map(|x| x as f32));
        let f64s = [-1e300f64, -0.5, 1.0 / 3.0, 1e-300];
        let raw: Vec<u8> = f64s.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_decodes(28, &[4], &raw, &f64s.map(|x| x as f32));
    }

    /// ggml `dequantize_row_nvfp4`: 4 UE4M3 scale bytes (value * 0.5), nibbles low = 0..8, high = 8..16
    /// of each 16-element sub-block, doubled E2M1 table.
    #[test]
    fn gguf_nvfp4_block_matches_ggml_reference() {
        // Scales: 0x38 = 1.0 (d 0.5), 0x01 = 2^-9 subnormal (d 2^-10), 0x7F = NaN code (d 0), 0x7E = 448 (d 224).
        let mut blk = vec![0x38, 0x01, 0x7F, 0x7E];
        for _ in 0..4 {
            blk.extend((0..8u8).map(|j| j | ((15 - j) << 4)));
        }
        let fp4 = [
            0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.,
        ];
        let mut want = Vec::new();
        for d in [0.5f32, 2f32.powi(-10), 0.0, 224.0] {
            want.extend((0..8).map(|j| fp4[j] * d));
            want.extend((0..8).map(|j| fp4[15 - j] * d));
        }
        assert_eq!(deq(QuantFormat::GGUF_NVFP4, &blk), want);
        // Two blocks (one row of 128) through the container path.
        assert_decodes(
            40,
            &[128],
            &[blk.clone(), blk].concat(),
            &[want.clone(), want].concat(),
        );
    }

    #[test]
    fn header_prefix_asks_for_more_until_complete() {
        let s = |v: &str| {
            let mut b = (v.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(v.as_bytes());
            b
        };
        let mut arr = 8u32.to_le_bytes().to_vec();
        arr.extend_from_slice(&2u64.to_le_bytes());
        arr.extend(s("ab"));
        arr.extend(s("cde"));
        let file = test_gguf_file(
            &[
                ("general.architecture", 8, s("llama")),
                ("tokenizer.ggml.tokens", 9, arr),
            ],
            &[("w", &[4, 2], &[0.0; 8])],
        );
        let full = parse_gguf_header(&file).unwrap();
        assert_eq!(full.metadata["general.architecture"], "llama");
        assert_eq!(
            (full.tensors[0].shape.clone(), full.tensors[0].byte_offset),
            (vec![4, 2], full.data_start)
        );
        let mut complete = false;
        for cut in 0..full.data_start {
            match parse_gguf_header(&file[..cut]) {
                Err(HeaderError::NeedMore(n)) => {
                    assert!(
                        !complete && n > cut && n <= full.data_start,
                        "cut {cut}: needs {n}"
                    );
                }
                Ok(h) => {
                    complete = true;
                    assert_eq!(h.tensors.len(), 1);
                }
                Err(e) => panic!("cut {cut}: {e}"),
            }
        }
        assert!(complete, "the header ends before its alignment padding");
        assert!(matches!(
            parse_gguf_header(b"GGUX\0\0\0\0"),
            Err(HeaderError::Invalid(_))
        ));
    }
}
