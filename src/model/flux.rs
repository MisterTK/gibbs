//! The legacy DiT pipelines' (Z-Image, Qwen-Image, Krea 2) transformer weights: a checkpoint left
//! in its stored encoding (mmap'd bf16 safetensors or GGUF blocks, decoded per GEMM panel or bound
//! as packed graph weights, never expanded), linears with their LoRA updates, and LoRA key folding.

use crate::hardware::QuantFormat;
use crate::graph::ValueId;
use crate::model::dit_graph::Gb;
use crate::model::layers::gguf_block_codec;
use crate::model::loaders::gguf::GgufData;
use crate::model::weights::{Mat, Packed, Weights};
use crate::model::{bf16_to_f32, decode_dtype, f16_to_f32, nn, UniversalModelLoader, VirtualTensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Weight storage
// ---------------------------------------------------------------------------

/// A weight matrix `[rows, cols]` left in its stored encoding.
pub(crate) enum Wt {
    Bf16(Mat),
    Gguf {
        map: Arc<GgufData>,
        off: usize,
        rows: usize,
        cols: usize,
        fmt: QuantFormat,
        dtype: String,
    },
}

impl Wt {
    pub(crate) fn dims(&self) -> (usize, usize) {
        match self {
            Wt::Bf16(m) => (m.rows, m.cols),
            Wt::Gguf { rows, cols, .. } => (*rows, *cols),
        }
    }

    /// Row `r`, columns `k0..k0 + dst.len()`, as f32. `k0` is a multiple of the GEMM's k block
    /// (256), so GGUF blocks (32 or 256 wide) never straddle a call.
    pub(crate) fn decode(&self, r: usize, k0: usize, dst: &mut [f32]) {
        match self {
            Wt::Bf16(m) => {
                for (d, &s) in dst.iter_mut().zip(&m.row(r)[k0..]) {
                    *d = bf16_to_f32(s);
                }
            }
            Wt::Gguf {
                map,
                off,
                cols,
                fmt,
                dtype,
                ..
            } => {
                let at = |es: usize| &map[off + (r * cols + k0) * es..];
                match fmt {
                    QuantFormat::FP32 => {
                        if dtype == "F32" {
                            for (d, c) in dst.iter_mut().zip(at(4).chunks_exact(4)) {
                                *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                            }
                        } else {
                            let es = match dtype.as_str() {
                                "F64" | "I64" | "U64" => 8,
                                "I32" | "U32" => 4,
                                "I16" | "U16" => 2,
                                _ => 1,
                            };
                            if let Ok(vals) = decode_dtype(&at(es)[..dst.len() * es], dtype) {
                                dst.copy_from_slice(&vals);
                            }
                        }
                    }
                    QuantFormat::FP16 => {
                        for (d, c) in dst.iter_mut().zip(at(2).chunks_exact(2)) {
                            *d = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                        }
                    }
                    QuantFormat::BF16 => {
                        for (d, c) in dst.iter_mut().zip(at(2).chunks_exact(2)) {
                            *d = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                        }
                    }
                    f => {
                        let (bn, bb, dq) = gguf_block_codec(*f).expect("checked at load");
                        let row = &map[off + (r * cols / bn + k0 / bn) * bb..];
                        for (y, b) in dst.chunks_exact_mut(bn).zip(row.chunks_exact(bb)) {
                            dq(b, y);
                        }
                    }
                }
            }
        }
    }
}

/// The transformer checkpoint: mmap'd safetensors (one file or shards) or one GGUF file.
pub enum Store {
    St(Weights, String),
    Gguf(Arc<GgufData>, HashMap<String, VirtualTensor>),
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, String> {
        if path.extension().is_some_and(|e| e == "gguf") {
            let (manifest, map) = crate::model::loaders::gguf::open(path)?;
            let tensors = manifest
                .tensors
                .into_iter()
                .map(|(n, t)| {
                    (
                        n.strip_prefix("model.diffusion_model.")
                            .map(str::to_string)
                            .unwrap_or(n),
                        t,
                    )
                })
                .collect();
            return Ok(Store::Gguf(Arc::new(map), tensors));
        }
        let w = Weights::open(path)?;
        // ComfyUI-style single files nest the transformer under `model.diffusion_model.`.
        let prefix = if w.has("model.diffusion_model.img_in.weight")
            || w.has("model.diffusion_model.first.weight")
        {
            "model.diffusion_model."
        } else {
            ""
        };
        Ok(Store::St(w, prefix.into()))
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        match self {
            Store::St(w, p) => w.has(&format!("{p}{name}")),
            Store::Gguf(_, t) => t.contains_key(name),
        }
    }

    fn gguf(t: &HashMap<String, VirtualTensor>, name: &str) -> Result<VirtualTensor, String> {
        t.get(name)
            .cloned()
            .ok_or_else(|| format!("GGUF: tensor '{name}' missing"))
    }

    pub(crate) fn vec(&self, name: &str) -> Result<Vec<f32>, String> {
        match self {
            Store::St(w, p) => w.vec(&format!("{p}{name}")),
            Store::Gguf(map, t) => {
                let t = Self::gguf(t, name)?;
                let raw = &map[t.byte_offset..t.byte_offset + t.byte_len];
                let mut v = match t.quant_format {
                    QuantFormat::FP32 | QuantFormat::FP16 | QuantFormat::BF16 => {
                        decode_dtype(raw, &t.dtype)?
                    }
                    f => crate::model::layers::dequantize_gguf_blocks(f, raw).ok_or_else(|| {
                        format!("GGUF: '{name}' has unsupported GGML type {}", t.dtype)
                    })?,
                };
                v.truncate(t.shape.iter().product());
                Ok(v)
            }
        }
    }

    pub(crate) fn mat(&self, name: &str) -> Result<Wt, String> {
        match self {
            Store::St(w, p) => Ok(Wt::Bf16(w.mat(&format!("{p}{name}"))?)),
            Store::Gguf(map, t) => {
                let t = Self::gguf(t, name)?;
                // GGML order: ne0 (cols) first.
                let (cols, rows) = match t.shape[..] {
                    [c, r] => (c, r),
                    _ => {
                        return Err(format!(
                            "GGUF: '{name}' has shape {:?}, expected 2-D",
                            t.shape
                        ))
                    }
                };
                let fmt = t.quant_format;
                if !matches!(
                    fmt,
                    QuantFormat::FP32 | QuantFormat::FP16 | QuantFormat::BF16
                ) {
                    let (bn, _, _) = gguf_block_codec(fmt).ok_or_else(|| {
                        format!("GGUF: '{name}' has unsupported GGML type {}", t.dtype)
                    })?;
                    if cols % bn != 0 {
                        return Err(format!(
                            "GGUF: '{name}' row of {cols} is not whole {} blocks",
                            t.dtype
                        ));
                    }
                }
                Ok(Wt::Gguf {
                    map: map.clone(),
                    off: t.byte_offset,
                    rows,
                    cols,
                    fmt,
                    dtype: t.dtype.clone(),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Linear layers and LoRA
// ---------------------------------------------------------------------------

/// Low-rank update `out[:, off..off+len] += scale * up(down(x))`.
pub(crate) struct Lora {
    down: Vec<f32>,
    up: Vec<f32>,
    rank: usize,
    off: usize,
    len: usize,
    scale: f32,
}

/// One logical linear whose output rows may be stored as several matrices (diffusers keeps
/// q/k/v apart where BFL fuses them).
pub struct Lin {
    parts: Vec<(usize, Wt)>,
    bias: Option<Vec<f32>>,
    n: usize,
    k: usize,
    loras: Vec<Lora>,
}

impl Lin {
    /// One stored matrix with its bias and the LoRAs taken for it.
    pub(crate) fn new(w: Wt, bias: Option<Vec<f32>>, loras: Vec<Lora>) -> Self {
        let (n, k) = w.dims();
        Self {
            parts: vec![(0, w)],
            bias,
            n,
            k,
            loras,
        }
    }

    pub(crate) fn in_dim(&self) -> usize {
        self.k
    }

    pub(crate) fn out_dim(&self) -> usize {
        self.n
    }

    pub fn fwd(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * self.n];
        nn::matmul_rows(
            x,
            m,
            self.k,
            self.n,
            self.bias.as_deref(),
            &mut out,
            &|j, k0, dst| {
                let (s, w) = self
                    .parts
                    .iter()
                    .rev()
                    .find(|(s, _)| *s <= j)
                    .expect("part 0 starts at 0");
                w.decode(j - s, k0, dst);
            },
        );
        for l in &self.loras {
            let t = nn::linear(x, m, &l.down, None, self.k, l.rank);
            let u = nn::linear(&t, m, &l.up, None, l.rank, l.len);
            for (o, u) in out.chunks_exact_mut(self.n).zip(u.chunks_exact(l.len)) {
                for (o, u) in o[l.off..l.off + l.len].iter_mut().zip(u) {
                    *o += l.scale * u;
                }
            }
        }
        out
    }
}

/// LoRA updates keyed by module name with `.`/`_` folded to `_` and the trainer prefix stripped,
/// so diffusers/PEFT (`transformer.transformer_blocks.0.attn.to_q`) and kohya
/// (`lora_unet_double_blocks_0_img_attn_qkv`) keys meet the names the loader asks for.
#[derive(Default)]
pub struct Loras {
    pub(crate) by_module: HashMap<String, (Vec<f32>, Vec<f32>, usize, usize, f32)>,
    /// ComfyUI `<module>.diff_b` bias deltas (already scaled by the strength).
    pub(crate) bias: HashMap<String, Vec<f32>>,
}

impl Loras {
    fn key(module: &str) -> String {
        module.replace('.', "_")
    }

    /// A LoRA file's module name without its trainer prefix (diffusers/PEFT `transformer.`,
    /// kohya `lora_unet_`, ...), as [`Self::key`].
    fn lora_key(module: &str) -> String {
        let m = [
            "transformer.",
            "diffusion_model.",
            "base_model.model.",
            "lora_unet_",
            "lora_transformer_",
        ]
        .iter()
        .find_map(|p| module.strip_prefix(p))
        .unwrap_or(module);
        Self::key(m)
    }

    /// Adds one LoRA file at `strength`. Every tensor must pair into a transformer module update.
    pub(crate) fn add_file(&mut self, path: &Path, strength: f32) -> Result<(), String> {
        let (manifest, data) = crate::model::loaders::read_checkpoint(path)?;
        let get = |n: &str| -> Result<Vec<f32>, String> {
            let t = &manifest.tensors[n];
            decode_dtype(&data[t.byte_offset..t.byte_offset + t.byte_len], &t.dtype)
                .map_err(|e| format!("LoRA {}: '{n}': {e}", path.display()))
        };
        let mut modules: HashMap<String, [Option<String>; 3]> = HashMap::new();
        for name in manifest.tensors.keys() {
            if let Some(m) = name.strip_suffix(".diff_b") {
                let b = get(name)?.into_iter().map(|v| v * strength).collect();
                self.bias.insert(Self::lora_key(m), b);
                continue;
            }
            let (module, slot) = [
                (".lora_A.weight", 0),
                (".lora_down.weight", 0),
                (".lora.down.weight", 0),
                (".lora_B.weight", 1),
                (".lora_up.weight", 1),
                (".lora.up.weight", 1),
                (".alpha", 2),
            ]
            .iter()
            .find_map(|(s, i)| Some((name.strip_suffix(s)?, *i)))
            .ok_or_else(|| {
                format!(
                    "LoRA {}: tensor '{name}' is not a LoRA down/up/alpha",
                    path.display()
                )
            })?;
            modules.entry(module.to_string()).or_default()[slot] = Some(name.clone());
        }
        for (module, [down, up, alpha]) in modules {
            let (Some(down), Some(up)) = (down, up) else {
                return Err(format!(
                    "LoRA {}: module '{module}' lacks a down or up matrix",
                    path.display()
                ));
            };
            let shape = |n: &str| manifest.tensors[n].shape.clone();
            let (ds, us) = (shape(&down), shape(&up));
            let (rank, n) = (ds[0], us[0]);
            if us.iter().skip(1).product::<usize>() != rank {
                return Err(format!(
                    "LoRA {}: '{module}' up {us:?} does not match down {ds:?}",
                    path.display()
                ));
            }
            let scale = match alpha {
                Some(a) => get(&a)?[0] / rank as f32,
                None => 1.0,
            } * strength;
            self.by_module.insert(
                Self::lora_key(&module),
                (get(&down)?, get(&up)?, rank, n, scale),
            );
        }
        Ok(())
    }

    /// The bias delta of `module`, if the LoRA carries one.
    pub(crate) fn take_bias(&mut self, module: &str) -> Option<Vec<f32>> {
        self.bias.remove(&Self::key(module))
    }

    pub(crate) fn take(
        &mut self,
        module: &str,
        k: usize,
        off: usize,
        len: usize,
    ) -> Result<Option<Lora>, String> {
        let Some((down, up, rank, n, scale)) = self.by_module.remove(&Self::key(module)) else {
            return Ok(None);
        };
        if n != len || down.len() != rank * k {
            return Err(format!(
                "LoRA module '{module}' is {n}x{} but the model's is {len}x{k}",
                down.len() / rank
            ));
        }
        Ok(Some(Lora {
            down,
            up,
            rank,
            off,
            len,
            scale,
        }))
    }
}

fn wt_mat(w: &Wt) -> Mat {
    match w {
        Wt::Bf16(m) => m.clone(),
        Wt::Gguf {
            map,
            off,
            rows,
            cols,
            fmt,
            ..
        } => Mat::from_packed(
            Packed {
                map: map.clone(),
                off: *off,
                format: *fmt,
                shape: vec![*rows, *cols],
            },
            *rows,
            *cols,
        ),
    }
}

impl Lin {
    /// `x [m, k] -> [m, n]`: one linear per stored part (concatenated along the output), then the
    /// LoRAs `out[:, off..off+len] += scale * up(down(x))`.
    pub(crate) fn emit(&self, b: &mut Gb, name: &str, x: ValueId) -> ValueId {
        let outs: Vec<ValueId> = self
            .parts
            .iter()
            .enumerate()
            .map(|(i, (off, w))| {
                let m = wt_mat(w);
                let bias = self.bias.as_ref().map(|v| &v[*off..*off + m.rows]);
                b.lin_mat(&format!("{name}.p{i}"), x, &m, bias)
            })
            .collect();
        let mut y = if outs.len() == 1 {
            outs[0]
        } else {
            b.concat(name, &outs, 1)
        };
        for (i, l) in self.loras.iter().enumerate() {
            let p = format!("{name}.lora{i}");
            let down = b.konst(&format!("{p}.down"), &[l.rank, self.k], l.down.clone());
            let up = b.konst(&format!("{p}.up"), &[l.len, l.rank], l.up.clone());
            let t = b.linear(&format!("{p}.t"), x, down, None);
            let u = b.linear(&format!("{p}.u"), t, up, None);
            let mut u = b.scale(&format!("{p}.s"), u, l.scale);
            if l.len != self.n {
                u = b.pad_last(&format!("{p}.pad"), u, l.off, self.n - l.off - l.len);
            }
            y = b.add(&format!("{p}.add"), y, u);
        }
        y
    }
}

/// Tensor names of a GGUF or safetensors file (headers only); `None` for any other file.
pub(crate) fn tensor_names(path: &Path) -> Option<Vec<String>> {
    Some(match path.extension()?.to_str()? {
        "gguf" => {
            // Splits are large LMs unless the file says FLUX: probe the first shard's header
            // instead of reading the whole set.
            if crate::model::loaders::gguf::shard_paths(path).is_some()
                && crate::model::loaders::gguf::architecture(path).as_deref() != Some("flux")
            {
                return None;
            }
            crate::model::loaders::gguf::open(path)
        }
        .ok()?
        .0
        .tensors
        .into_keys()
        .collect(),
        "safetensors" => {
            let file = std::fs::File::open(path).ok()?;
            // SAFETY: read-only mapping of a cache file, dropped before returning.
            let map = unsafe { Mmap::map(&file) }.ok()?;
            UniversalModelLoader::parse_safetensors(&map)
                .ok()?
                .tensors
                .into_keys()
                .collect()
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_wt_decode_non_f32() {
        let f64_vals: Vec<f64> = vec![1.5, -2.5, 3.25, 4.75];
        let mut buf = Vec::new();
        for &v in &f64_vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let wt = Wt::Gguf {
            map: Arc::new(GgufData::Buf(buf)),
            off: 0,
            rows: 2,
            cols: 2,
            fmt: QuantFormat::FP32,
            dtype: "F64".to_string(),
        };
        let mut dst = [0f32; 2];
        wt.decode(0, 0, &mut dst);
        assert_eq!(dst, [1.5, -2.5]);
        wt.decode(1, 0, &mut dst);
        assert_eq!(dst, [3.25, 4.75]);
    }

    #[test]
    fn lora_names_meet_the_modules_the_loader_asks_for() {
        // (LoRA file module, module name the loader looks up): PEFT/diffusers keys on a diffusers
        // base, kohya keys on a BFL base (GGUF / single file).
        for (lora, module) in [
            (
                "transformer.single_transformer_blocks.0.attn.to_k",
                "single_transformer_blocks.0.attn.to_k",
            ),
            (
                "transformer.single_transformer_blocks.3.proj_mlp",
                "single_transformer_blocks.3.proj_mlp",
            ),
            (
                "transformer.transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.attn.to_q",
            ),
            (
                "transformer.transformer_blocks.2.ff_context.net.0.proj",
                "transformer_blocks.2.ff_context.net.0.proj",
            ),
            (
                "lora_unet_double_blocks_0_img_attn_qkv",
                "double_blocks.0.img_attn.qkv",
            ),
            (
                "lora_unet_single_blocks_0_linear1",
                "single_blocks.0.linear1",
            ),
        ] {
            assert_eq!(Loras::lora_key(lora), Loras::key(module), "{lora}");
        }
    }
}
