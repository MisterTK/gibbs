//! FLUX.1 text-to-image (schnell, dev, their finetunes and LoRAs): CLIP-L pooled + T5-XXL text
//! conditioning, the MMDiT (double-stream then single-stream blocks, 3-axis RoPE), flow-matching
//! Euler over 2x2-packed 16-channel latents, and the 16-channel AutoencoderKL.
//!
//! The transformer reads diffusers names (`transformer_blocks.*`) or the original BFL names
//! (`double_blocks.*`, single-file checkpoints and city96-style GGUFs). Weights stay where they
//! are stored (mmap'd bf16 or GGUF blocks) and are decoded per GEMM panel, never expanded.

use crate::hardware::QuantFormat;
use crate::model::clip::{ClipTextTransformer, Tensors};
use crate::model::flow;
use crate::model::hub::{component_weights, HuggingFaceHub, ModelArtifacts};
use crate::model::image::ImageBuffer;
use crate::model::layers::gguf_block_codec;
use crate::model::loaders::gguf::GgufData;
use crate::model::loaders::safetensors::SafetensorsWeightReader;
use crate::model::par::par_rows;
use crate::model::qwen3_5::Qwen3Encoder;
use crate::model::sd::Rng;
use crate::model::t5::T5Encoder;
use crate::model::text_to_image::{pipeline_class, ImageDefaults, ImageGenerator, ImageRequest};
use crate::model::tokenizer::Tokenizer;
use crate::model::vae::SdVae;
use crate::model::weights::{Mat, Weights};
use crate::model::{
    bf16_to_f32, decode_dtype, f16_to_f32, nn, UniversalModelLoader, VirtualTensor,
};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const HEAD_DIM: usize = 128;
/// FLUX.1/Chroma RoPE: (unused, row, col) axes; FLUX.2 uses `config.axes_dims_rope` (t, h, w, l).
const AXES: [usize; 3] = [16, 56, 56];
const THETA: f64 = 10000.0;
const EPS: f32 = 1e-6;

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
pub(crate) enum Store {
    St(Weights, String),
    Gguf(Arc<GgufData>, HashMap<String, VirtualTensor>),
}

impl Store {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
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

    /// `{name}.weight` (+ `{name}.bias` when stored) as one linear.
    pub(crate) fn lin(&self, name: &str) -> Result<Lin, String> {
        let w = self.mat(&format!("{name}.weight"))?;
        let (n, k) = w.dims();
        let b = format!("{name}.bias");
        Ok(Lin {
            bias: self.has(&b).then(|| self.vec(&b)).transpose()?,
            parts: vec![(0, w)],
            n,
            k,
            loras: Vec::new(),
        })
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
pub(crate) struct Lin {
    parts: Vec<(usize, Wt)>,
    bias: Option<Vec<f32>>,
    n: usize,
    k: usize,
    loras: Vec<Lora>,
}

impl Lin {
    /// `(out, in)` features.
    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

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

    pub(crate) fn fwd(&self, x: &[f32], m: usize) -> Vec<f32> {
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
pub(crate) struct Loras {
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

/// Resolves BFL names (or their diffusers equivalents) into layers, attaching LoRAs.
struct Ld<'a> {
    s: &'a Store,
    diffusers: bool,
    loras: &'a mut Loras,
}

impl Ld<'_> {
    /// `bfl` is the fused BFL module; `diff` its diffusers parts, stacked along the output.
    fn lin(&mut self, bfl: &str, diff: &[&str]) -> Result<Lin, String> {
        let names: Vec<&str> = if self.diffusers {
            diff.to_vec()
        } else {
            vec![bfl]
        };
        let (mut parts, mut biases, mut n, mut k) = (Vec::new(), Vec::new(), 0, None);
        for name in &names {
            let w = self.s.mat(&format!("{name}.weight"))?;
            let (r, c) = w.dims();
            if *k.get_or_insert(c) != c {
                return Err(format!(
                    "FLUX: '{name}' has {c} inputs, sibling parts {}",
                    k.unwrap()
                ));
            }
            let b = format!("{name}.bias");
            biases.push(if self.s.has(&b) {
                Some(self.s.vec(&b)?)
            } else {
                None
            });
            parts.push((n, w));
            n += r;
        }
        let k = k.expect("at least one part");
        let bias = biases.iter().any(Option::is_some).then(|| {
            biases
                .iter()
                .zip(&parts)
                .flat_map(|(b, (_, w))| b.clone().unwrap_or_else(|| vec![0.0; w.dims().0]))
                .collect()
        });
        let mut loras = Vec::new();
        loras.extend(self.loras.take(bfl, k, 0, n)?);
        let widths: Vec<usize> = match (self.diffusers, diff.len()) {
            (true, _) => parts.iter().map(|(_, w)| w.dims().0).collect(),
            (false, 1) => vec![n],
            (false, 3) => vec![n / 3; 3],
            // Single-block linear1 = q, k, v (hidden = k wide each), then the MLP.
            (false, c) => (0..c)
                .map(|i| if i + 1 < c { k } else { n - (c - 1) * k })
                .collect(),
        };
        let mut off = 0;
        for (name, w) in diff.iter().zip(widths) {
            loras.extend(self.loras.take(name, k, off, w)?);
            off += w;
        }
        Ok(Lin {
            parts,
            bias,
            n,
            k,
            loras,
        })
    }

    fn vec(&self, bfl: &str, diff: &str) -> Result<Vec<f32>, String> {
        self.s.vec(if self.diffusers { diff } else { bfl })
    }
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

/// `in_layer -> SiLU -> out_layer` (time, guidance and pooled-text embedders).
struct Embed {
    a: Lin,
    b: Lin,
}

impl Embed {
    fn load(l: &mut Ld, bfl: &str, diff: &str) -> Result<Self, String> {
        Ok(Self {
            a: l.lin(&format!("{bfl}.in_layer"), &[&format!("{diff}.linear_1")])?,
            b: l.lin(&format!("{bfl}.out_layer"), &[&format!("{diff}.linear_2")])?,
        })
    }

    fn fwd(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.a.fwd(x, 1);
        nn::silu(&mut h);
        self.b.fwd(&h, 1)
    }
}

/// One stream (image or text) of a double block.
struct Stream {
    /// FLUX.1's per-block modulation; `None` for Chroma (approximator rows).
    modl: Option<Lin>,
    qkv: Lin,
    qn: Vec<f32>,
    kn: Vec<f32>,
    proj: Lin,
    mlp0: Lin,
    mlp2: Lin,
}

impl Stream {
    /// `x += gate2 * mlp(LN(x) * (1 + scale2) + shift2)`, modulation `m` = 6 chunks of `d`.
    fn mlp(&self, x: &mut [f32], n: usize, m: &[f32], d: usize, swiglu: bool) {
        let mut h = self
            .mlp0
            .fwd(&ln_mod(x, d, &m[3 * d..4 * d], &m[4 * d..5 * d]), n);
        if swiglu {
            h = swiglu_rows(&h, self.mlp0.n);
        } else {
            nn::gelu_tanh(&mut h);
        }
        gated_add(x, &self.mlp2.fwd(&h, n), &m[5 * d..]);
    }
}

/// FLUX.2 SwiGLU on rows of width `w`: `silu(first half) * second half`.
fn swiglu_rows(h: &[f32], w: usize) -> Vec<f32> {
    let half = w / 2;
    let mut out = vec![0.0; h.len() / 2];
    par_rows(&mut out, half, &|r, o| {
        let row = &h[r * w..(r + 1) * w];
        for ((o, &g), &u) in o.iter_mut().zip(&row[..half]).zip(&row[half..]) {
            *o = g / (1.0 + (-g).exp()) * u;
        }
    });
    out
}

struct Double {
    img: Stream,
    txt: Stream,
}

struct Single {
    modl: Option<Lin>,
    linear1: Lin,
    linear2: Lin,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

/// Chroma's `distilled_guidance_layer`: an MLP mapping (timestep, modulation index) to every
/// block's shift/scale/gate row.
struct Approx {
    in_proj: Lin,
    layers: Vec<(Vec<f32>, Embed)>,
    out_proj: Lin,
}

impl Approx {
    fn load(l: &mut Ld) -> Result<Self, String> {
        let p = "distilled_guidance_layer";
        let mut layers = Vec::new();
        while l
            .s
            .has(&format!("{p}.layers.{}.linear_1.weight", layers.len()))
            || l.s
                .has(&format!("{p}.layers.{}.in_layer.weight", layers.len()))
        {
            let i = layers.len();
            let norm = l.vec(
                &format!("{p}.norms.{i}.scale"),
                &format!("{p}.norms.{i}.weight"),
            )?;
            let bfl = format!("{p}.layers.{i}");
            let mlp = Embed {
                a: l.lin(&format!("{bfl}.in_layer"), &[&format!("{bfl}.linear_1")])?,
                b: l.lin(&format!("{bfl}.out_layer"), &[&format!("{bfl}.linear_2")])?,
            };
            layers.push((norm, mlp));
        }
        Ok(Self {
            in_proj: l.lin(&format!("{p}.in_proj"), &[&format!("{p}.in_proj")])?,
            layers,
            out_proj: l.lin(&format!("{p}.out_proj"), &[&format!("{p}.out_proj")])?,
        })
    }

    /// `rows` modulation rows `[rows, d]` at time `t`: row i reads
    /// `[sin(t*1000) 16, sin(guidance 0) 16, sin(i*1000) 32]` (cos halves first).
    fn fwd(&self, t: f32, rows: usize) -> Vec<f32> {
        let (tp, gp) = (sinusoid(t as f64 * 1000.0, 16), sinusoid(0.0, 16));
        let input: Vec<f32> = (0..rows)
            .flat_map(|i| {
                tp.iter()
                    .chain(&gp)
                    .copied()
                    .chain(sinusoid(i as f64 * 1000.0, 32))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut x = self.in_proj.fwd(&input, rows);
        let w = self.in_proj.n;
        for (norm, mlp) in &self.layers {
            let mut h = x.clone();
            // torch.nn.RMSNorm(eps=None) uses finfo(dtype).eps; the reference runs bf16 (2^-7).
            par_rows(&mut h, w, &|_, r| {
                let inv =
                    1.0 / (r.iter().map(|v| v * v).sum::<f32>() / w as f32 + 0.0078125).sqrt();
                r.iter_mut().zip(norm).for_each(|(v, g)| *v *= inv * g);
            });
            let mut h = mlp.a.fwd(&h, rows);
            nn::silu(&mut h);
            add(&mut x, &mlp.b.fwd(&h, rows));
        }
        self.out_proj.fwd(&x, rows)
    }
}

/// Where block modulations come from.
enum Modulation {
    /// FLUX.1: per-block linears on SiLU(time + pooled text (+ guidance) embedding).
    Vector {
        time_in: Embed,
        vector_in: Embed,
        guidance_in: Option<Embed>,
    },
    /// Chroma: approximator rows, no pooled text or guidance input.
    Table(Approx),
    /// FLUX.2: one modulation per stream kind shared by every block, from the timestep alone.
    Shared {
        time_in: Embed,
        img: Lin,
        txt: Lin,
        single: Lin,
    },
}

pub struct FluxTransformer {
    img_in: Lin,
    txt_in: Lin,
    modulation: Modulation,
    double: Vec<Double>,
    single: Vec<Single>,
    final_mod: Option<Lin>,
    /// diffusers stores the final modulation as (scale, shift), BFL as (shift, scale).
    final_swap: bool,
    final_lin: Lin,
    d: usize,
    heads: usize,
    /// RoPE axis widths and base (FLUX.1/Chroma: [16, 56, 56], 10000).
    axes: Vec<usize>,
    theta: f64,
    /// FLUX.2 feed-forwards are SwiGLU; FLUX.1/Chroma use tanh-GELU.
    swiglu: bool,
    label: String,
}

impl FluxTransformer {
    fn load(s: &Store, loras: &mut Loras) -> Result<Self, String> {
        let diffusers = s.has("x_embedder.weight");
        if !diffusers && !s.has("img_in.weight") {
            return Err("not a FLUX transformer: no x_embedder.weight or img_in.weight".into());
        }
        let l = &mut Ld {
            s,
            diffusers,
            loras,
        };
        let img_in = l.lin("img_in", &["x_embedder"])?;
        let d = img_in.n;
        if d % HEAD_DIM != 0 {
            return Err(format!(
                "FLUX: hidden size {d} is not a multiple of head dim {HEAD_DIM}"
            ));
        }
        let txt_in = l.lin("txt_in", &["context_embedder"])?;
        let chroma = s.has("distilled_guidance_layer.in_proj.weight");
        let flux2 = s.has("double_stream_modulation_img.linear.weight");
        let modulation = if chroma {
            Modulation::Table(Approx::load(l)?)
        } else if flux2 {
            let m = |l: &mut Ld, n: &str| l.lin(n, &[n]);
            Modulation::Shared {
                time_in: Embed::load(l, "time_in", "time_guidance_embed.timestep_embedder")?,
                img: m(l, "double_stream_modulation_img.linear")?,
                txt: m(l, "double_stream_modulation_txt.linear")?,
                single: m(l, "single_stream_modulation.linear")?,
            }
        } else {
            Modulation::Vector {
                time_in: Embed::load(l, "time_in", "time_text_embed.timestep_embedder")?,
                vector_in: Embed::load(l, "vector_in", "time_text_embed.text_embedder")?,
                guidance_in: if s.has("guidance_in.in_layer.weight")
                    || s.has("time_text_embed.guidance_embedder.linear_1.weight")
                {
                    Some(Embed::load(
                        l,
                        "guidance_in",
                        "time_text_embed.guidance_embedder",
                    )?)
                } else {
                    None
                },
            }
        };
        let mut double = Vec::new();
        loop {
            let (b, t) = (
                format!("double_blocks.{}", double.len()),
                format!("transformer_blocks.{}", double.len()),
            );
            if !s.has(&format!("{b}.img_attn.qkv.weight"))
                && !s.has(&format!("{t}.attn.to_q.weight"))
            {
                break;
            }
            let mut stream = |bs: &str,
                              m: &str,
                              q: [&str; 3],
                              qn: [&str; 2],
                              proj: &str,
                              ff: &str|
             -> Result<Stream, String> {
                let a = format!("{t}.attn");
                let ff_in = if flux2 {
                    format!("{t}.{ff}.linear_in")
                } else {
                    format!("{t}.{ff}.net.0.proj")
                };
                let ff_out = if flux2 {
                    format!("{t}.{ff}.linear_out")
                } else {
                    format!("{t}.{ff}.net.2")
                };
                Ok(Stream {
                    modl: if chroma || flux2 {
                        None
                    } else {
                        Some(l.lin(&format!("{b}.{bs}_mod.lin"), &[&format!("{t}.{m}.linear")])?)
                    },
                    qkv: l.lin(
                        &format!("{b}.{bs}_attn.qkv"),
                        &q.map(|q| format!("{a}.{q}")).each_ref().map(String::as_str),
                    )?,
                    qn: l.vec(
                        &format!("{b}.{bs}_attn.norm.query_norm.scale"),
                        &format!("{a}.{}.weight", qn[0]),
                    )?,
                    kn: l.vec(
                        &format!("{b}.{bs}_attn.norm.key_norm.scale"),
                        &format!("{a}.{}.weight", qn[1]),
                    )?,
                    proj: l.lin(&format!("{b}.{bs}_attn.proj"), &[&format!("{a}.{proj}")])?,
                    mlp0: l.lin(&format!("{b}.{bs}_mlp.0"), &[&ff_in])?,
                    mlp2: l.lin(&format!("{b}.{bs}_mlp.2"), &[&ff_out])?,
                })
            };
            let img = stream(
                "img",
                "norm1",
                ["to_q", "to_k", "to_v"],
                ["norm_q", "norm_k"],
                "to_out.0",
                "ff",
            )?;
            let txt = stream(
                "txt",
                "norm1_context",
                ["add_q_proj", "add_k_proj", "add_v_proj"],
                ["norm_added_q", "norm_added_k"],
                "to_add_out",
                "ff_context",
            )?;
            double.push(Double { img, txt });
        }
        let mut single = Vec::new();
        loop {
            let (b, t) = (
                format!("single_blocks.{}", single.len()),
                format!("single_transformer_blocks.{}", single.len()),
            );
            if !s.has(&format!("{b}.linear1.weight")) && !s.has(&format!("{t}.attn.to_q.weight")) {
                break;
            }
            let qkvm =
                ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"].map(|p| format!("{t}.{p}"));
            let (fused_in, fused_out, proj_out) = (
                format!("{t}.attn.to_qkv_mlp_proj"),
                format!("{t}.attn.to_out"),
                format!("{t}.proj_out"),
            );
            single.push(Single {
                modl: if chroma || flux2 {
                    None
                } else {
                    Some(l.lin(
                        &format!("{b}.modulation.lin"),
                        &[&format!("{t}.norm.linear")],
                    )?)
                },
                linear1: if flux2 {
                    l.lin(&format!("{b}.linear1"), &[&fused_in])?
                } else {
                    l.lin(
                        &format!("{b}.linear1"),
                        &qkvm.each_ref().map(String::as_str),
                    )?
                },
                linear2: l.lin(
                    &format!("{b}.linear2"),
                    &[if flux2 { &fused_out } else { &proj_out }],
                )?,
                qn: l.vec(
                    &format!("{b}.norm.query_norm.scale"),
                    &format!("{t}.attn.norm_q.weight"),
                )?,
                kn: l.vec(
                    &format!("{b}.norm.key_norm.scale"),
                    &format!("{t}.attn.norm_k.weight"),
                )?,
            });
        }
        if double.is_empty() || single.is_empty() {
            return Err(format!(
                "FLUX: found {} double and {} single blocks",
                double.len(),
                single.len()
            ));
        }
        // ponytail: a LoRA on the final modulation is applied in the base's (scale, shift) order;
        // swap its halves too if a BFL-named LoRA ever targets a diffusers base.
        let final_mod = if chroma {
            None
        } else {
            Some(l.lin("final_layer.adaLN_modulation.1", &["norm_out.linear"])?)
        };
        let final_lin = l.lin("final_layer.linear", &["proj_out"])?;
        let label = match s {
            Store::St(..) => "bf16".to_string(),
            Store::Gguf(_, t) => {
                let mut types: Vec<&str> = t.values().map(|v| v.dtype.as_str()).collect();
                types.sort();
                types.dedup();
                format!("GGUF {}", types.join("/"))
            }
        };
        Ok(Self {
            img_in,
            txt_in,
            modulation,
            double,
            single,
            final_mod,
            final_swap: diffusers && !chroma,
            final_lin,
            heads: d / HEAD_DIM,
            d,
            axes: if flux2 { vec![32; 4] } else { AXES.to_vec() },
            theta: if flux2 { 2000.0 } else { THETA },
            swiglu: flux2,
            label,
        })
    }

    fn guidance_embeds(&self) -> bool {
        matches!(
            self.modulation,
            Modulation::Vector {
                guidance_in: Some(_),
                ..
            }
        )
    }

    /// Pooled-text embedding `vector_in(pooled)` (FLUX.1; Chroma has none).
    fn pooled(&self, pooled: &[f32]) -> Vec<f32> {
        match &self.modulation {
            Modulation::Vector { vector_in, .. } => vector_in.fwd(pooled),
            Modulation::Table(_) | Modulation::Shared { .. } => Vec::new(),
        }
    }

    /// Velocity for packed latents `img` `[ni, 64]` at time `t` (= sigma) given the projected
    /// text `txt` `[nt, d]` and the pooled-text embedding `y` from [`Self::pooled`].
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &[f32],
        ni: usize,
        txt: &[f32],
        nt: usize,
        y: &[f32],
        t: f32,
        guidance: Option<f32>,
        rope: &Rope,
    ) -> Vec<f32> {
        let d = self.d;
        let (nd, ns) = (self.double.len(), self.single.len());
        // Either SiLU(vec), which every modulation linear reads, or Chroma's table of rows.
        let src = match &self.modulation {
            Modulation::Vector {
                time_in,
                guidance_in,
                ..
            } => {
                let mut v = time_in.fwd(&sinusoid(t as f64 * 1000.0, 256));
                add(&mut v, y);
                if let (Some(g_in), Some(g)) = (guidance_in, guidance) {
                    add(&mut v, &g_in.fwd(&sinusoid(g as f64 * 1000.0, 256)));
                }
                nn::silu(&mut v);
                v
            }
            Modulation::Table(a) => a.fwd(t, 3 * ns + 12 * nd + 2),
            Modulation::Shared { time_in, .. } => {
                let mut v = time_in.fwd(&sinusoid(t as f64 * 1000.0, 256));
                nn::silu(&mut v);
                v
            }
        };
        let shared = match &self.modulation {
            Modulation::Shared {
                img, txt, single, ..
            } => Some((img.fwd(&src, 1), txt.fwd(&src, 1), single.fwd(&src, 1))),
            _ => None,
        };
        let modv = |l: &Option<Lin>, row: usize, rows: usize| match l {
            Some(l) => l.fwd(&src, 1),
            None => src[row * d..(row + rows) * d].to_vec(),
        };
        let mut img = self.img_in.fwd(img, ni);
        let mut txt = txt.to_vec();
        let s = nt + ni;
        for (i, b) in self.double.iter().enumerate() {
            let (im, tm) = match &shared {
                Some((im, tm, _)) => (im.clone(), tm.clone()),
                None => (
                    modv(&b.img.modl, 3 * ns + 6 * i, 6),
                    modv(&b.txt.modl, 3 * ns + 6 * nd + 6 * i, 6),
                ),
            };
            let iqkv = b.img.qkv.fwd(&ln_mod(&img, d, &im[..d], &im[d..2 * d]), ni);
            let tqkv = b.txt.qkv.fwd(&ln_mod(&txt, d, &tm[..d], &tm[d..2 * d]), nt);
            // Joint attention over [text; image].
            let (mut q, mut k, mut vv) = (vec![0.0; s * d], vec![0.0; s * d], vec![0.0; s * d]);
            for (src, row0) in [(&tqkv, 0), (&iqkv, nt)] {
                for (r, x) in src.chunks_exact(3 * d).enumerate() {
                    let o = (row0 + r) * d;
                    q[o..o + d].copy_from_slice(&x[..d]);
                    k[o..o + d].copy_from_slice(&x[d..2 * d]);
                    vv[o..o + d].copy_from_slice(&x[2 * d..]);
                }
            }
            rms_heads(&mut q[..nt * d], &b.txt.qn);
            rms_heads(&mut q[nt * d..], &b.img.qn);
            rms_heads(&mut k[..nt * d], &b.txt.kn);
            rms_heads(&mut k[nt * d..], &b.img.kn);
            rope.apply(&mut q);
            rope.apply(&mut k);
            let a = nn::attention(&q, &k, &vv, s, s, self.heads, HEAD_DIM, false);
            gated_add(
                &mut txt,
                &b.txt.proj.fwd(&a[..nt * d], nt),
                &tm[2 * d..3 * d],
            );
            gated_add(
                &mut img,
                &b.img.proj.fwd(&a[nt * d..], ni),
                &im[2 * d..3 * d],
            );
            b.img.mlp(&mut img, ni, &im, d, self.swiglu);
            b.txt.mlp(&mut txt, nt, &tm, d, self.swiglu);
        }
        let mut x = txt;
        x.extend_from_slice(&img);
        for (i, b) in self.single.iter().enumerate() {
            let m = match &shared {
                Some((_, _, sm)) => sm.clone(),
                None => modv(&b.modl, 3 * i, 3),
            };
            let h = b.linear1.fwd(&ln_mod(&x, d, &m[..d], &m[d..2 * d]), s);
            let w1 = b.linear1.n;
            // MLP input width; SwiGLU halves it.
            let mlp_in = w1 - 3 * d;
            let mlp = if self.swiglu { mlp_in / 2 } else { mlp_in };
            let (mut q, mut k, mut vv) = (vec![0.0; s * d], vec![0.0; s * d], vec![0.0; s * d]);
            let mut cat = vec![0.0; s * (d + mlp)];
            let act = if self.swiglu {
                swiglu_rows(&h, w1)
            } else {
                Vec::new()
            };
            for (r, row) in h.chunks_exact(w1).enumerate() {
                q[r * d..(r + 1) * d].copy_from_slice(&row[..d]);
                k[r * d..(r + 1) * d].copy_from_slice(&row[d..2 * d]);
                vv[r * d..(r + 1) * d].copy_from_slice(&row[2 * d..3 * d]);
                let src = if self.swiglu {
                    &act[r * mlp..(r + 1) * mlp]
                } else {
                    &row[3 * d..]
                };
                cat[r * (d + mlp) + d..(r + 1) * (d + mlp)].copy_from_slice(src);
            }
            rms_heads(&mut q, &b.qn);
            rms_heads(&mut k, &b.kn);
            rope.apply(&mut q);
            rope.apply(&mut k);
            let a = nn::attention(&q, &k, &vv, s, s, self.heads, HEAD_DIM, false);
            par_rows(&mut cat, d + mlp, &|r, row| {
                row[..d].copy_from_slice(&a[r * d..(r + 1) * d]);
                if self.swiglu {
                    return;
                }
                for v in &mut row[d..] {
                    let z = *v;
                    *v = 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044715 * z * z * z)).tanh());
                }
            });
            gated_add(&mut x, &b.linear2.fwd(&cat, s), &m[2 * d..3 * d]);
        }
        let m = modv(&self.final_mod, 3 * ns + 12 * nd, 2);
        let (mut shift, mut scale) = m.split_at(d);
        if self.final_swap {
            (shift, scale) = (scale, shift);
        }
        self.final_lin
            .fwd(&ln_mod(&x[nt * d..], d, shift, scale), ni)
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
}

/// `LayerNorm(x) * (1 + scale) + shift` per row (no affine, eps 1e-6).
fn ln_mod(x: &[f32], d: usize, shift: &[f32], scale: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    par_rows(&mut y, d, &|_, r| {
        let mean = r.iter().sum::<f32>() / d as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, sh), sc) in r.iter_mut().zip(shift).zip(scale) {
            *v = (*v - mean) * inv * (1.0 + sc) + sh;
        }
    });
    y
}

/// `x += gate * y` per row.
fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32]) {
    for (xr, yr) in x
        .chunks_exact_mut(gate.len())
        .zip(y.chunks_exact(gate.len()))
    {
        for ((x, y), g) in xr.iter_mut().zip(yr).zip(gate) {
            *x += g * y;
        }
    }
}

/// Per-head RMSNorm (q/k norm), eps 1e-6.
fn rms_heads(x: &mut [f32], w: &[f32]) {
    par_rows(x, HEAD_DIM, &|_, r| {
        let inv = 1.0 / (r.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32 + EPS).sqrt();
        for (v, w) in r.iter_mut().zip(w) {
            *v *= inv * w;
        }
    });
}

/// Sinusoidal embedding of `x`, `dim` wide, cos half first (BFL `timestep_embedding`,
/// diffusers `Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)`).
pub(crate) fn sinusoid(x: f64, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let args: Vec<f64> = (0..half)
        .map(|i| x * (-(10000f64.ln()) * i as f64 / half as f64).exp())
        .collect();
    args.iter()
        .map(|a| a.cos() as f32)
        .chain(args.iter().map(|a| a.sin() as f32))
        .collect()
}

/// Multi-axis rotary tables: per token, `HEAD_DIM / 2` angle pairs, axis 0 first.
pub(crate) struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// `axes` widths (summing to `HEAD_DIM`) read the leading coordinates of each id.
    fn new(ids: &[[usize; 4]], axes: &[usize], theta: f64) -> Self {
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for id in ids {
            for (pos, &dim) in id.iter().zip(axes) {
                for i in 0..dim / 2 {
                    let a = *pos as f64 * theta.powf(-((2 * i) as f64) / dim as f64);
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        Self { cos, sin }
    }

    /// Rotates interleaved pairs `(x[2i], x[2i+1])` of every head of `x` `[tokens, heads * HEAD_DIM]`.
    fn apply(&self, x: &mut [f32]) {
        let half = HEAD_DIM / 2;
        let tokens = self.cos.len() / half;
        par_rows(x, x.len() / tokens, &|t, row| {
            let (c, s) = (
                &self.cos[t * half..(t + 1) * half],
                &self.sin[t * half..(t + 1) * half],
            );
            for head in row.chunks_exact_mut(HEAD_DIM) {
                for ((p, c), s) in head.chunks_exact_mut(2).zip(c).zip(s) {
                    let (a, b) = (p[0], p[1]);
                    p[0] = c * a - s * b;
                    p[1] = s * a + c * b;
                }
            }
        });
    }
}

/// Position ids: `nt` text tokens, then the packed `hp x wp` latent grid as `(0, row, col, 0)`
/// row-major (diffusers `_prepare_latent_image_ids`). FLUX.1 text sits at the origin; FLUX.2
/// numbers it along the last axis (`text_axis`: `(0, 0, 0, l)`).
fn position_ids(nt: usize, hp: usize, wp: usize, text_axis: bool) -> Vec<[usize; 4]> {
    let mut ids: Vec<[usize; 4]> = (0..nt)
        .map(|l| [0, 0, 0, if text_axis { l } else { 0 }])
        .collect();
    ids.extend((0..hp * wp).map(|i| [0, i / wp, i % wp, 0]));
    ids
}

/// `[c, h, w]` latents -> `[(h/2)(w/2), 4c]` 2x2 patches, feature `ch*4 + dy*2 + dx`.
fn pack(lat: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let wp = w / 2;
    let mut out = vec![0.0; lat.len()];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[((y / 2) * wp + x / 2) * 4 * c + ch * 4 + (y % 2) * 2 + x % 2] =
                    lat[(ch * h + y) * w + x];
            }
        }
    }
    out
}

fn unpack(p: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let wp = w / 2;
    let mut out = vec![0.0; p.len()];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(ch * h + y) * w + x] =
                    p[((y / 2) * wp + x / 2) * 4 * c + ch * 4 + (y % 2) * 2 + x % 2];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Flow-matching schedule (diffusers FlowMatchEulerDiscreteScheduler as FluxPipeline drives it)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Sched {
    shift: f64,
    dynamic: bool,
    base_seq: f64,
    max_seq: f64,
    base_shift: f64,
    max_shift: f64,
    /// FLUX.2 pipelines pick `mu` with `compute_empirical_mu(image tokens, steps)`.
    empirical_mu: bool,
}

/// diffusers `compute_empirical_mu` (Flux2Pipeline / Flux2KleinPipeline).
fn empirical_mu(seq: usize, steps: usize) -> f64 {
    let (a1, b1, a2, b2) = (8.73809524e-05, 1.89833333, 0.00016927, 0.45666666);
    let seq = seq as f64;
    if seq > 4300.0 {
        return a2 * seq + b2;
    }
    let (m200, m10) = (a2 * seq + b2, a1 * seq + b1);
    let a = (m200 - m10) / 190.0;
    a * steps as f64 + m200 - 200.0 * a
}

impl Sched {
    fn from_config(v: &Value, empirical_mu: bool) -> Self {
        let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        Self {
            shift: f("shift", 1.0),
            dynamic: v["use_dynamic_shifting"].as_bool().unwrap_or(false),
            base_seq: f("base_image_seq_len", 256.0),
            max_seq: f("max_image_seq_len", 4096.0),
            base_shift: f("base_shift", 0.5),
            max_shift: f("max_shift", 1.15),
            empirical_mu,
        }
    }

    /// FluxPipeline: `sigmas = linspace(1, 1/steps, steps)`, `mu = calculate_shift(image tokens)`,
    /// shifted by the scheduler config, 0 appended.
    fn sigmas(&self, steps: usize, seq: usize) -> Vec<f32> {
        let base = flow::linspace_sigmas(steps);
        if !self.dynamic {
            return flow::shifted_sigmas(&base, self.shift);
        }
        let mu = if self.empirical_mu {
            empirical_mu(seq, steps)
        } else {
            flow::calculate_shift(
                seq,
                self.base_seq,
                self.max_seq,
                self.base_shift,
                self.max_shift,
            )
        };
        flow::dynamic_sigmas(&base, mu)
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// The text side of each pipeline.
enum TextEnc {
    /// FLUX.1: CLIP-L pooled + T5-XXL on ids padded to `max_seq`, every token attended.
    ClipT5 {
        clip: ClipTextTransformer,
        t5: T5Encoder,
        tok: Tokenizer,
        max_seq: usize,
    },
    /// Chroma: T5-XXL with all padding but the first masked.
    T5 { t5: T5Encoder, tok: Tokenizer },
    /// FLUX.2 klein: Qwen3 hidden states after layers 9, 18, 27, concatenated per token, of the
    /// non-thinking chat prompt padded to 512 (padding masked as keys, every row kept).
    Qwen3 {
        enc: Qwen3Encoder,
        tok: Tokenizer,
        pad: u32,
    },
}

const T5_MAX: usize = 512;
const QWEN_LAYERS: [usize; 3] = [9, 18, 27];

pub struct FluxPipeline {
    text: TextEnc,
    tr: FluxTransformer,
    vae: SdVae,
    sched: Sched,
    /// VAE latent channels (16; FLUX.2: 32, normalised by the VAE's BatchNorm stats `bn`
    /// (mean, std) over the 2x2-patchified channels).
    latent_ch: usize,
    bn: Option<(Vec<f32>, Vec<f32>)>,
    /// FLUX.2 klein step-distilled (`model_index.json` `is_distilled`): no CFG.
    distilled: bool,
    label: String,
}

pub(crate) fn read_json(path: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))
}

/// T5 tokenizer of a diffusers component dir: tokenizer.json, else SentencePiece spiece.model.
pub(crate) fn t5_tokenizer(dir: &Path) -> Result<Tokenizer, String> {
    let json = dir.join("tokenizer.json");
    if json.is_file() {
        Tokenizer::from_json_str(
            &std::fs::read_to_string(&json).map_err(|e| format!("{}: {e}", json.display()))?,
        )
    } else {
        crate::model::spm::load(&dir.join("spiece.model"))
    }
}

/// `</s>`-terminated T5 ids, truncated to `max` (EOS kept).
pub(crate) fn t5_ids(tok: &Tokenizer, prompt: &str, max: usize) -> Vec<u32> {
    let mut ids = tok.encode(prompt);
    if ids.last() != Some(&1) {
        ids.push(1);
    }
    if ids.len() > max {
        ids.truncate(max - 1);
        ids.push(1);
    }
    ids
}

pub(crate) fn t5_load(dir: &Path) -> Result<T5Encoder, String> {
    T5Encoder::load(
        &Weights::open(&component_weights(dir)?)?,
        &read_json(&dir.join("config.json"))?,
        "",
    )
}

impl FluxPipeline {
    /// A diffusers FluxPipeline, ChromaPipeline or Flux2KleinPipeline snapshot.
    pub fn load(dir: &Path, hub: &HuggingFaceHub) -> Result<Self, String> {
        Self::load_with(dir, hub, None, &[])
    }

    /// `base` supplies the text encoders, VAE, scheduler (and the transformer unless `transformer`
    /// names a GGUF or single-file checkpoint); `loras` are `(file, strength)` on the transformer.
    pub fn load_with(
        base: &Path,
        hub: &HuggingFaceHub,
        transformer: Option<&Path>,
        loras: &[(PathBuf, f32)],
    ) -> Result<Self, String> {
        let tr_path = match transformer {
            Some(p) => p.to_path_buf(),
            None => component_weights(&base.join("transformer"))?,
        };
        let mut lora_set = Loras::default();
        for (path, strength) in loras {
            lora_set.add_file(path, *strength)?;
        }
        let store = Store::open(&tr_path)?;
        let tr = FluxTransformer::load(&store, &mut lora_set)?;
        if !lora_set.by_module.is_empty() {
            let mut left: Vec<&String> = lora_set.by_module.keys().collect();
            left.sort();
            return Err(format!(
                "LoRA targets {} modules this FLUX transformer does not have (e.g. {}); text-encoder LoRAs are not supported",
                left.len(),
                left.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        let (name, text) = match &tr.modulation {
            Modulation::Table(_) => {
                let t5 = t5_load(&base.join("text_encoder"))?;
                (
                    "Chroma",
                    TextEnc::T5 {
                        t5,
                        tok: t5_tokenizer(&base.join("tokenizer"))?,
                    },
                )
            }
            Modulation::Shared { .. } => {
                let dir = base.join("tokenizer");
                let tok =
                    crate::model::hf_loader::load_json_tokenizer(&dir.join("tokenizer.json"))?;
                if tok.chat_template.is_none() {
                    return Err(format!("FLUX.2: {} has no chat template", dir.display()));
                }
                let pad = *tok
                    .vocab
                    .get("<|endoftext|>")
                    .ok_or("FLUX.2: tokenizer has no <|endoftext|> pad token")?;
                let enc = Qwen3Encoder::open(&base.join("text_encoder"))?;
                ("FLUX.2 klein", TextEnc::Qwen3 { enc, tok, pad })
            }
            Modulation::Vector { guidance_in, .. } => {
                let clip = ClipTextTransformer::load_diffusers(
                    hub,
                    &base.join("text_encoder"),
                    &base.join("tokenizer"),
                )?;
                let t5 = t5_load(&base.join("text_encoder_2"))?;
                let tok = t5_tokenizer(&base.join("tokenizer_2"))?;
                // 256 for schnell (its model card), 512 for guidance-distilled dev.
                let dev = guidance_in.is_some();
                let max_seq = if dev { 512 } else { 256 };
                (
                    if dev { "FLUX.1 dev" } else { "FLUX.1 schnell" },
                    TextEnc::ClipT5 {
                        clip,
                        t5,
                        tok,
                        max_seq,
                    },
                )
            }
        };
        let vae_dir = base.join("vae");
        let vae_cfg = read_json(&vae_dir.join("config.json"))?;
        let vae_reader = SafetensorsWeightReader::new(component_weights(&vae_dir)?)?;
        let vae_t = Tensors::safetensors(&vae_reader)?;
        let flux2 = matches!(tr.modulation, Modulation::Shared { .. });
        let (vae, bn) = if flux2 {
            let eps = vae_cfg["batch_norm_eps"].as_f64().unwrap_or(1e-4) as f32;
            let std: Vec<f32> = vae_t
                .load("bn.running_var")?
                .iter()
                .map(|v| (v + eps).sqrt())
                .collect();
            (
                SdVae::from_diffusers(&vae_t)?.with_scaling(1.0, 0.0),
                Some((vae_t.load("bn.running_mean")?, std)),
            )
        } else {
            let vae = SdVae::from_diffusers(&vae_t)?.with_scaling(
                vae_cfg["scaling_factor"].as_f64().unwrap_or(0.3611) as f32,
                vae_cfg["shift_factor"].as_f64().unwrap_or(0.1159) as f32,
            );
            (vae, None)
        };
        let latent_ch = vae.latent_channels();
        if tr.final_lin.n != 4 * latent_ch
            || bn
                .as_ref()
                .is_some_and(|(m, s)| m.len() != 4 * latent_ch || s.len() != m.len())
        {
            return Err(format!(
                "{name}: VAE has {latent_ch} latent channels but the transformer emits {} per 2x2 patch",
                tr.final_lin.n
            ));
        }
        let ctx_w = match &text {
            TextEnc::ClipT5 { t5, .. } | TextEnc::T5 { t5, .. } => t5.d_model(),
            TextEnc::Qwen3 { .. } => tr.txt_in.k,
        };
        if ctx_w != tr.txt_in.k {
            return Err(format!(
                "{name}: T5 width {ctx_w} != transformer context width {}",
                tr.txt_in.k
            ));
        }
        let index = read_json(&base.join("model_index.json")).unwrap_or(Value::Null);
        let sched = Sched::from_config(
            &read_json(&base.join("scheduler/scheduler_config.json"))?,
            flux2,
        );
        let label = format!(
            "{name} ({}{}), {} + VAE",
            tr.label,
            if loras.is_empty() {
                String::new()
            } else {
                format!(" + {} LoRA", loras.len())
            },
            match &text {
                TextEnc::ClipT5 { .. } => "CLIP-L + T5-XXL",
                TextEnc::T5 { .. } => "T5-XXL",
                TextEnc::Qwen3 { .. } => "Qwen3",
            },
        );
        let distilled = index["is_distilled"].as_bool().unwrap_or(false);
        Ok(Self {
            text,
            tr,
            vae,
            sched,
            latent_ch,
            bn,
            distilled,
            label,
        })
    }

    /// Projected text tokens `[nt, d]` and `nt`.
    fn text(&self, prompt: &str) -> Result<(Vec<f32>, usize), String> {
        let ctx = match &self.text {
            TextEnc::ClipT5 {
                t5, tok, max_seq, ..
            } => {
                let mut ids = t5_ids(tok, prompt, *max_seq);
                ids.resize(*max_seq, 0);
                t5.encode(&ids)
            }
            // Masked text never reaches another token or the output, so exactly the
            // `valid + 1` unmasked tokens are kept.
            TextEnc::T5 { t5, tok } => {
                let mut ids = t5_ids(tok, prompt, T5_MAX);
                let valid = ids.len();
                if valid < T5_MAX {
                    ids.push(0);
                }
                t5.encode_masked(&ids, valid)
            }
            TextEnc::Qwen3 { enc, tok, pad } => {
                let msg: crate::model::hf_loader::ChatMessage =
                    serde_json::from_value(serde_json::json!({"role": "user", "content": prompt}))
                        .map_err(|e| e.to_string())?;
                let text = tok
                    .chat_template
                    .as_ref()
                    .expect("checked at load")
                    .render(
                        &[msg],
                        None,
                        true,
                        &serde_json::json!({"enable_thinking": false}),
                    )
                    .map_err(|e| format!("FLUX.2 chat template: {e}"))?;
                let mut ids = tok.encode(&text);
                ids.truncate(T5_MAX);
                let valid = ids.len();
                ids.resize(T5_MAX, *pad);
                let hs = enc.hidden_states_at(&ids, valid, &QWEN_LAYERS)?;
                let w = hs[0].len() / ids.len();
                (0..ids.len())
                    .flat_map(|t| {
                        hs.iter()
                            .flat_map(move |h| h[t * w..(t + 1) * w].iter().copied())
                    })
                    .collect()
            }
        };
        let nt = ctx.len() / self.tr.txt_in.k;
        Ok((self.tr.txt_in.fwd(&ctx, nt), nt))
    }

    /// True CFG on a negative prompt: Chroma when guidance > 1, undistilled FLUX.2 klein likewise.
    fn uses_cfg(&self, guidance: f32) -> bool {
        match &self.text {
            TextEnc::ClipT5 { .. } => false,
            TextEnc::T5 { .. } => guidance > 1.0,
            TextEnc::Qwen3 { .. } => guidance > 1.0 && !self.distilled,
        }
    }
}

/// What a non-diffusers repo's weight file is to FLUX.
enum Derived {
    Transformer,
    Lora,
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

fn derived_kind(path: &Path) -> Option<Derived> {
    let ext = path.extension()?.to_str()?;
    let names = tensor_names(path)?;
    let has = |n: &str| {
        names
            .iter()
            .any(|x| x == n || x.strip_prefix("model.diffusion_model.") == Some(n))
    };
    if has("img_in.weight")
        || has("x_embedder.weight") && has("single_transformer_blocks.0.proj_out.weight")
    {
        return Some(Derived::Transformer);
    }
    let flux_module = |n: &str| {
        [
            "single_blocks",
            "single_transformer_blocks",
            "double_blocks",
        ]
        .iter()
        .any(|b| n.contains(b))
    };
    (ext == "safetensors" && names.iter().any(|n| n.contains("lora") && flux_module(n)))
        .then_some(Derived::Lora)
}

/// Repos that build on a FLUX.1 diffusers repo named by their card's `base_model`: a
/// transformer-only checkpoint (city96 GGUF, single-file finetune) or a LoRA. `None` when the
/// weights are neither or the base is not a FluxPipeline.
pub fn load_derived(
    a: &ModelArtifacts,
    hub: &HuggingFaceHub,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let weights = a.weights_path.as_ref()?;
    let kind = derived_kind(weights)?;
    // `--model org/repo/file.gguf` names one file of the repo.
    let repo = crate::model::hub::split_weight_file(&a.model_id).0;
    let card = a
        .local_dir
        .as_ref()
        .map(|d| d.join("README.md"))
        .filter(|p| p.is_file())
        .map_or_else(|| hub.fetch_file(repo, "README.md"), Ok)
        .and_then(|p| std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display())));
    let base = match card.map(|c| crate::model::lora::card_base_model(&c)) {
        Ok(Some(b)) => b,
        Ok(None) => {
            return Some(Err(format!(
                "{repo}: FLUX weights, but the model card names no base_model"
            )))
        }
        Err(e) => {
            return Some(Err(format!(
                "{repo}: FLUX weights, model card unreadable: {e}"
            )))
        }
    };
    let gated = |e: String| {
        format!(
            "FLUX base model {base} could not be fetched ({e}); if it is gated, accept its license at https://huggingface.co/{base} with the HF_TOKEN account"
        )
    };
    let class = match hub.fetch_file(&base, "model_index.json").map_err(gated) {
        Ok(p) => p.parent().and_then(pipeline_class),
        Err(e) => return Some(Err(e)),
    };
    if !matches!(class.as_deref(), Some("FluxPipeline" | "ChromaPipeline")) {
        return None;
    }
    let skip: &[&str] = match kind {
        Derived::Transformer => &["transformer"],
        Derived::Lora => &[],
    };
    Some((|| {
        let base_dir = hub.fetch_diffusers(&base, skip).map_err(gated)?;
        let p = match kind {
            Derived::Transformer => FluxPipeline::load_with(&base_dir, hub, Some(weights), &[])?,
            Derived::Lora => {
                FluxPipeline::load_with(&base_dir, hub, None, &[(weights.clone(), 1.0)])?
            }
        };
        Ok(Arc::new(p) as Arc<dyn ImageGenerator>)
    })())
}

impl ImageGenerator for FluxPipeline {
    fn summary(&self) -> String {
        self.label.clone()
    }

    /// Each pipeline's diffusers defaults (schnell: its model card's 4 steps; klein distilled: 4).
    fn defaults(&self) -> ImageDefaults {
        let (steps, guidance) = match &self.text {
            TextEnc::ClipT5 { max_seq: 512, .. } => (28, 3.5),
            TextEnc::ClipT5 { .. } => (4, 0.0),
            TextEnc::T5 { .. } => (35, 5.0),
            TextEnc::Qwen3 { .. } if self.distilled => (4, 1.0),
            TextEnc::Qwen3 { .. } => (50, 4.0),
        };
        ImageDefaults {
            width: 512,
            height: 512,
            steps,
            guidance,
            multiple: 16,
        }
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        let d = self.defaults();
        let (width, height, steps) = (
            req.width.unwrap_or(d.width),
            req.height.unwrap_or(d.height),
            req.steps.unwrap_or(d.steps),
        );
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(format!(
                "FLUX: width/height must be non-zero multiples of 16, got {width}x{height}"
            ));
        }
        if steps == 0 {
            return Err("FLUX: steps must be at least 1".into());
        }
        let guidance = req.guidance.unwrap_or(d.guidance);
        let t0 = Instant::now();
        let y = match &self.text {
            TextEnc::ClipT5 { clip, .. } => {
                self.tr.pooled(&clip.encode(&req.prompt, false)?.pooled)
            }
            _ => Vec::new(),
        };
        let (txt, nt) = self.text(&req.prompt)?;
        let neg = match self.uses_cfg(guidance) {
            true => Some(self.text(req.negative_prompt.as_deref().unwrap_or(""))?),
            false => None,
        };
        tracing::info!("FLUX text encoders: {:.1}s", t0.elapsed().as_secs_f32());

        let (h, w, c) = (height / 8, width / 8, self.latent_ch);
        let ni = (h / 2) * (w / 2);
        let mut rng = Rng::new(req.seed);
        let noise: Vec<f32> = (0..c * h * w).map(|_| rng.gaussian()).collect();
        let mut x = pack(&noise, c, h, w);
        let rope_for = |n: usize| {
            Rope::new(
                &position_ids(n, h / 2, w / 2, self.tr.swiglu),
                &self.tr.axes,
                self.tr.theta,
            )
        };
        let rope = rope_for(nt);
        let neg_rope = neg.as_ref().map(|(_, n)| rope_for(*n));
        let sigmas = self.sched.sigmas(steps, ni);
        let g_in = self.tr.guidance_embeds().then_some(guidance);
        for i in 0..steps {
            let t = Instant::now();
            let mut v = self
                .tr
                .forward(&x, ni, &txt, nt, &y, sigmas[i], g_in, &rope);
            if let (Some((ntxt, n)), Some(nr)) = (&neg, &neg_rope) {
                let vn = self.tr.forward(&x, ni, ntxt, *n, &y, sigmas[i], None, nr);
                v.iter_mut()
                    .zip(&vn)
                    .for_each(|(v, n)| *v = n + guidance * (*v - n));
            }
            flow::euler_step(&mut x, &v, sigmas[i], sigmas[i + 1]);
            tracing::info!(
                "FLUX step {}/{steps}: {:.1}s",
                i + 1,
                t.elapsed().as_secs_f32()
            );
        }
        if let Some((mean, std)) = &self.bn {
            for row in x.chunks_exact_mut(4 * c) {
                for ((v, m), s) in row.iter_mut().zip(mean).zip(std) {
                    *v = *v * s + m;
                }
            }
        }
        self.vae.decode(&unpack(&x, c, h, w), h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schnell_schedule_is_unshifted_linspace() {
        let cfg = serde_json::json!({"shift": 1.0, "use_dynamic_shifting": false});
        assert_eq!(
            Sched::from_config(&cfg, false).sigmas(4, 1024),
            vec![1.0, 0.75, 0.5, 0.25, 0.0]
        );
    }

    #[test]
    fn dev_schedule_matches_diffusers_dynamic_shift() {
        // FLUX.1-dev scheduler_config.json; 512x512 -> 32x32 packed tokens.
        let cfg = serde_json::json!({"base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096,
            "max_shift": 1.15, "shift": 3.0, "use_dynamic_shifting": true});
        // mu = 1024 * 0.65/3840 + 0.5 - 256 * 0.65/3840 = 0.63; sigma' = e^mu / (e^mu + 1/sigma - 1).
        let e = 0.63f64.exp();
        let want: Vec<f64> = [1.0, 0.75, 0.5, 0.25]
            .iter()
            .map(|s| e / (e + 1.0 / s - 1.0))
            .collect();
        let got = Sched::from_config(&cfg, false).sigmas(4, 1024);
        assert_eq!(got.len(), 5);
        assert_eq!(got[4], 0.0);
        for (g, w) in got.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-6, "{got:?} vs {want:?}");
        }
        assert!((got[2] - 0.652_490).abs() < 1e-5);
    }

    #[test]
    fn flux2_empirical_mu_matches_diffusers() {
        // compute_empirical_mu(1024, 4): m200 = 0.62999914, m10 = 1.98781143, a = (m200 - m10) / 190.
        assert!((empirical_mu(1024, 4) - 2.030_689_7).abs() < 1e-6);
        // Past 4300 tokens it is the m200 line alone.
        assert!((empirical_mu(5000, 4) - (0.00016927 * 5000.0 + 0.45666666)).abs() < 1e-12);
    }
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
    fn rope_positions_follow_packed_grid() {
        // 2 text tokens at the origin, then a 2x3 packed latent grid row-major.
        let ids = position_ids(2, 2, 3, false);
        let grid = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
        ];
        assert_eq!(ids[..2], [[0; 4]; 2]);
        assert_eq!(ids[2..], grid);
        // FLUX.2 numbers text along the last axis; the grid is unchanged.
        let ids2 = position_ids(2, 2, 3, true);
        assert_eq!(ids2[..2], [[0, 0, 0, 0], [0, 0, 0, 1]]);
        assert_eq!(ids2[2..], grid);
        let rope = Rope::new(&ids, &AXES, THETA);
        let half = HEAD_DIM / 2;
        let angle = |tok: usize, pair: usize| {
            rope.sin[tok * half + pair].atan2(rope.cos[tok * half + pair])
        };
        // Axis 0 (8 pairs) is always 0; axis 1 starts at pair 8 (theta^0 = 1 rad per row),
        // axis 2 at pair 8 + 28 = 36 (1 rad per column); later pairs use theta^(-2i/56).
        let tok = 2 + 5; // row 1, col 2
        for p in 0..8 {
            assert_eq!(angle(tok, p), 0.0);
        }
        assert!((angle(tok, 8) - 1.0).abs() < 1e-6);
        assert!((angle(tok, 36) - 2.0).abs() < 1e-6);
        let f = 10000f64.powf(-2.0 / 56.0) as f32;
        assert!((angle(tok, 9) - f).abs() < 1e-6);
        assert!((angle(tok, 37) - 2.0 * f).abs() < 1e-6);
        // Text tokens are unrotated.
        assert!(rope.sin[..2 * half].iter().all(|&s| s == 0.0));
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

    #[test]
    fn pack_is_2x2_patches_and_unpack_inverts_it() {
        let (c, h, w) = (2, 4, 6);
        let lat: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let p = pack(&lat, c, h, w);
        // Token (row 1, col 2) channel 1: pixels (2..4, 4..6) of channel 1.
        let tok = (1 * 3 + 2) * 4 * c;
        let at = |ch: usize, y: usize, x: usize| lat[(ch * h + y) * w + x];
        assert_eq!(
            &p[tok + 4..tok + 8],
            &[at(1, 2, 4), at(1, 2, 5), at(1, 3, 4), at(1, 3, 5)]
        );
        assert_eq!(unpack(&p, c, h, w), lat);
    }
}
