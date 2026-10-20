//! Idefics3 / SmolVLM (`model_type` `idefics3`; SmolVLM2's `smolvlm` is the same model for
//! images): SigLIP tower → pixel-shuffle connector → Llama text model run by [`GraphLm`].
//! Images are resized and split like transformers' `Idefics3ImageProcessor` (longest edge to
//! `size`, then `max_image_size` tiles plus a global view) and each `<image>` placeholder expands
//! like `Idefics3Processor` (`<fake_token_around_image>`, `<row_i_col_j>`, `<global-img>`,
//! `image_seq_len` `<image>` tokens per tile). On Metal the tower + connector is one graph program
//! per tile and the text model a decode program fed `inputs_embeds` (the `<image>` rows scattered
//! over the token embeddings by a GPU `Scatter`); `--backend cpu` runs the host forward.

use crate::graph::program::GraphProgram;
use crate::graph::{DType, Tensor};
use crate::model::vision_graph::{compile, is_gpu, scatter_rows, Gb};
use crate::hardware::QuantMode;
use crate::model::graph_lm::{GraphLm, GraphLmConfig, LayerKVCache};
use crate::model::hf_loader::{load_json_tokenizer, HfConfigJson};
use crate::model::hub::ModelArtifacts;
use crate::model::image::ImageBuffer;
use crate::model::native::{MediaInput, NativeLm, SeqState};
use crate::model::siglip::SiglipVision;
use crate::model::tokenizer::Tokenizer;
use crate::model::{nn, UniversalModelLoader};
use serde_json::Value;
use std::sync::Arc;

const FAKE: &str = "<fake_token_around_image>";
const GLOBAL: &str = "<global-img>";
const IMAGE: &str = "<image>";
/// `Idefics3ImageProcessor` MAX_IMAGE_SIZE: no resized side exceeds it.
const MAX_SIDE: usize = 4096;

/// What `preprocessor_config.json` says about turning one image into encoder tiles.
#[derive(Debug, Clone)]
pub struct Preprocess {
    /// `size.longest_edge`: the image's longest side is resized to this first.
    pub longest_edge: usize,
    /// `max_image_size.longest_edge`: tile side (= the vision encoder's input side).
    pub tile: usize,
    pub split: bool,
    /// PIL resample filter: true = LANCZOS (1), false = BICUBIC (3).
    pub lanczos: bool,
    pub rescale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Preprocess {
    pub fn from_json(p: &Value) -> Result<Self, String> {
        let edge = |k: &str| {
            p[k]["longest_edge"]
                .as_u64()
                .map(|v| v as usize)
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("preprocessor_config.json: {k}.longest_edge missing"))
        };
        let triple = |k: &str| -> Result<[f32; 3], String> {
            let v: Vec<f32> = p[k]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64())
                        .map(|x| x as f32)
                        .collect()
                })
                .unwrap_or_default();
            v.try_into()
                .map_err(|_| format!("preprocessor_config.json: {k} is not 3 numbers"))
        };
        let lanczos = match p["resample"].as_u64() {
            Some(1) | None => true,
            Some(3) => false,
            Some(r) => {
                return Err(format!(
                    "preprocessor_config.json: resample {r} unsupported (1 LANCZOS, 3 BICUBIC)"
                ))
            }
        };
        if p["do_resize"] == false || p["do_rescale"] == false || p["do_normalize"] == false {
            return Err(
                "preprocessor_config.json: do_resize/do_rescale/do_normalize off is unsupported"
                    .into(),
            );
        }
        Ok(Self {
            longest_edge: edge("size")?,
            tile: edge("max_image_size")?,
            split: p["do_image_splitting"].as_bool().unwrap_or(true),
            lanczos,
            rescale: p["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0) as f32,
            mean: triple("image_mean")?,
            std: triple("image_std")?,
        })
    }

    fn resize(&self, img: &ImageBuffer, w: usize, h: usize) -> ImageBuffer {
        if self.lanczos {
            img.resize_lanczos(w, h)
        } else {
            img.resize(w, h)
        }
    }

    /// Sizes a `w × h` image is resized to: longest edge to `longest_edge` (sides capped at
    /// [`MAX_SIDE`]), then, when splitting, sides up to multiples of the tile keeping the aspect
    /// ratio.
    fn sizes(&self, w: usize, h: usize) -> ((usize, usize), Option<(usize, usize)>) {
        let t = self.tile;
        // Python `int(a / b)` on the aspect ratio, as the HF processor computes it.
        let (mut w, mut h) = (w, h);
        let ar = w as f64 / h as f64;
        if w >= h {
            w = self.longest_edge;
            h = (w as f64 / ar) as usize;
            h += h % 2;
        } else {
            h = self.longest_edge;
            w = (h as f64 * ar) as usize;
            w += w % 2;
        }
        let ar = w.max(1) as f64 / h.max(1) as f64;
        if w >= h && w > MAX_SIDE {
            (w, h) = (MAX_SIDE, (MAX_SIDE as f64 / ar) as usize);
        } else if h > w && h > MAX_SIDE {
            (w, h) = ((MAX_SIDE as f64 * ar) as usize, MAX_SIDE);
        }
        let (w, h) = (w.max(1), h.max(1));
        if !self.split {
            return ((w, h), None);
        }
        let ar = w as f64 / h as f64;
        let split = if w >= h {
            let w = w.div_ceil(t) * t;
            (w, ((w as f64 / ar) as usize).div_ceil(t).max(1) * t)
        } else {
            let h = h.div_ceil(t) * t;
            (((h as f64 * ar) as usize).div_ceil(t).max(1) * t, h)
        };
        ((w, h), Some(split))
    }

    /// `(rows, cols)` of crops a `w × h` image splits into; `(0, 0)` when it is a single tile.
    pub fn grid(&self, w: usize, h: usize) -> (usize, usize) {
        match self.sizes(w, h).1 {
            Some((w, h)) if w > self.tile || h > self.tile => (h / self.tile, w / self.tile),
            _ => (0, 0),
        }
    }

    /// `(rows, cols, tiles)`: `rows × cols` row-major crops then the global view, or `(0, 0,
    /// [image])` when it fits one tile. Every tile is `tile × tile`.
    pub fn tiles(&self, img: &ImageBuffer) -> (usize, usize, Vec<ImageBuffer>) {
        let t = self.tile;
        let (rows, cols) = self.grid(img.width, img.height);
        let ((w, h), split) = self.sizes(img.width, img.height);
        let img = self.resize(img, w, h);
        let Some((w, h)) = split else {
            return (0, 0, vec![self.resize(&img, t, t)]);
        };
        let img = self.resize(&img, w, h);
        if rows == 0 {
            return (0, 0, vec![img]);
        }
        let mut tiles = Vec::with_capacity(rows * cols + 1);
        for r in 0..rows {
            for c in 0..cols {
                let data = (0..t)
                    .flat_map(|y| {
                        let start = ((r * t + y) * w + c * t) * 3;
                        img.data[start..start + t * 3].iter().copied()
                    })
                    .collect();
                tiles.push(ImageBuffer::from_rgb(t, t, data).expect("tile is t*t*3"));
            }
        }
        tiles.push(self.resize(&img, t, t));
        (rows, cols, tiles)
    }

    /// RGB8 tile → rescaled, normalized `[3, h, w]`.
    fn pixels(&self, img: &ImageBuffer) -> Vec<f32> {
        let n = img.width * img.height;
        let mut out = vec![0.0; 3 * n];
        for (i, px) in img.data.chunks_exact(3).enumerate() {
            for c in 0..3 {
                out[c * n + i] = (px[c] as f32 * self.rescale - self.mean[c]) / self.std[c];
            }
        }
        out
    }
}

/// The text `Idefics3Processor` puts in place of one `<image>` for a `rows × cols` split.
pub fn placeholder(rows: usize, cols: usize, image_seq_len: usize) -> String {
    let tokens = IMAGE.repeat(image_seq_len);
    let mut s = String::new();
    for r in 1..=rows {
        for c in 1..=cols {
            s += &format!("{FAKE}<row_{r}_col_{c}>{tokens}");
        }
        s.push('\n');
    }
    if rows > 0 {
        s.push('\n');
    }
    s + &format!("{FAKE}{GLOBAL}{tokens}{FAKE}")
}

pub struct Idefics3 {
    engine: Engine,
    tokenizer: Tokenizer,
    image_token: u32,
    image_seq_len: usize,
    pre: Preprocess,
    context_len: usize,
}

/// SigLIP + connector + Llama text, on the CPU (`--backend cpu`) or as two GPU programs.
struct Engine {
    cfg: GraphLmConfig,
    image_size: usize,
    image_token: u32,
    kind: Kind,
}

enum Kind {
    Cpu {
        vision: SiglipVision,
        /// Connector projection `[text_hidden, vision_hidden * scale²]`, no bias.
        proj: Vec<f32>,
        scale: usize,
        text: GraphLm,
    },
    Gpu {
        mode: crate::hardware::PlacementMode,
        /// `pixels [3, S, S]` → connector rows `[image_seq_len, text_hidden]`.
        vision: parking_lot::Mutex<GraphProgram>,
        /// `inputs_embeds [1, hidden]` + position → logits; KV cache in the sequence's state.
        text: parking_lot::Mutex<GraphProgram>,
        /// Token embedding table (host row lookup, like every decode loop's token feed).
        embedding: Vec<f32>,
    },
}

/// One sequence: its KV cache and the projected image rows, one per `<image>` position.
struct Seq {
    /// CPU KV cache (empty on the GPU).
    kv: Vec<LayerKVCache>,
    /// The text program's KV state (GPU).
    state: crate::graph::StateStore,
    /// Ascending prompt positions of the `<image>` tokens; row `i` of `rows` belongs to the `i`th.
    image_pos: Vec<usize>,
    rows: Vec<f32>,
}

impl Engine {
    fn new(
        vision: SiglipVision,
        proj: Vec<f32>,
        scale: usize,
        text: GraphLm,
        image_token: u32,
        mode: crate::hardware::PlacementMode,
    ) -> Result<Self, String> {
        let (cfg, image_size) = (text.config.clone(), vision.image_size);
        if !is_gpu(mode) {
            let kind = Kind::Cpu { vision, proj, scale, text };
            return Ok(Self { cfg, image_size, image_token, kind });
        }
        let (d, g, th) = (vision.hidden, vision.image_size / vision.patch, cfg.hidden_dim);
        let mut b = Gb::new();
        let px = b.input("pixels", vec![3, image_size, image_size], DType::F32);
        let feats = vision.add_to_graph(&mut b, px);
        // Pixel shuffle: [g·g, d] → (block row, block col, in-block row, in-block col, d) rows.
        let s = scale;
        let x = b.reshape("connector.split", feats, vec![g / s, s, g / s, s, d]);
        let x = b.transpose("connector.shuffle", x, &[g / s, s, g / s, s, d], &[0, 2, 1, 3, 4]);
        let rows = g * g / (s * s);
        let x = b.reshape("connector.rows", x, vec![rows, s * s * d]);
        let pw = b.f32w("connector.proj.weight", vec![th, s * s * d], proj);
        let out = b.linear("connector.proj", x, pw, None, rows, th);
        let (vg, vw) = b.finish(&[out]);
        let vision = compile(vg, vw, mode).map_err(|e| format!("Idefics3 vision: {e}"))?;
        let (tg, tw) = text.to_decode_graph_embeds();
        let text_prog = compile(tg, tw, mode).map_err(|e| format!("Idefics3 text: {e}"))?;
        let embedding = text.embedding;
        let kind = Kind::Gpu {
            mode,
            vision: parking_lot::Mutex::new(vision),
            text: parking_lot::Mutex::new(text_prog),
            embedding,
        };
        Ok(Self { cfg, image_size, image_token, kind })
    }

    /// One normalized `[3, S, S]` tile → `image_seq_len` rows in the text model's embedding space.
    fn embed_tile(&self, pixels: &[f32]) -> Result<Vec<f32>, String> {
        match &self.kind {
            Kind::Cpu { vision, proj, scale, .. } => {
                let feats = vision.forward(pixels);
                // Pixel shuffle: each scale×scale patch block becomes one row, features ordered
                // (block row, block col, channel) like `Idefics3Connector.pixel_shuffle`.
                let (d, s) = (vision.hidden, *scale);
                let g = vision.image_size / vision.patch;
                let mut shuffled = Vec::with_capacity(feats.len());
                for bi in 0..g / s {
                    for bj in 0..g / s {
                        for m in 0..s {
                            for k in 0..s {
                                let p = (bi * s + m) * g + bj * s + k;
                                shuffled.extend_from_slice(&feats[p * d..(p + 1) * d]);
                            }
                        }
                    }
                }
                let rows = g * g / (s * s);
                Ok(nn::linear(&shuffled, rows, proj, None, d * s * s, self.cfg.hidden_dim))
            }
            Kind::Gpu { vision, .. } => {
                let out = vision.lock().run(vec![Tensor::F32(pixels.to_vec())])?;
                out.into_iter().next().ok_or("vision program returned nothing")?.into_f32()
            }
        }
    }

    fn new_seq(&self, prompt_len: usize, image_pos: Vec<usize>, rows: Vec<f32>) -> Seq {
        let kv = match &self.kind {
            Kind::Cpu { text, .. } => (0..text.layers.len())
                .map(|_| LayerKVCache::new(prompt_len, self.cfg.hidden_dim))
                .collect(),
            Kind::Gpu { .. } => Vec::new(),
        };
        Seq { kv, state: crate::graph::StateStore::new(), image_pos, rows }
    }

    fn forward(&self, tokens: &[u32], start_pos: usize, seq: &mut Seq) -> Result<Vec<f32>, String> {
        let d = self.cfg.hidden_dim;
        match &self.kind {
            Kind::Cpu { text, .. } => {
                let mut logits = Vec::new();
                for (i, &t) in tokens.iter().enumerate() {
                    let pos = start_pos + i;
                    logits = match seq.image_pos.binary_search(&pos) {
                        Ok(r) if t == self.image_token => text.forward_embedding_with_kv(
                            &seq.rows[r * d..(r + 1) * d],
                            pos,
                            &mut seq.kv,
                        ),
                        _ => text.forward_token_with_kv(t, pos, &mut seq.kv),
                    };
                }
                Ok(logits)
            }
            Kind::Gpu { mode, text, embedding, .. } => {
                // The chunk's text rows, then the `<image>` rows scattered over theirs on the GPU.
                let (mut base, mut at, mut rows) = (Vec::with_capacity(tokens.len() * d), Vec::new(), Vec::new());
                for (i, &t) in tokens.iter().enumerate() {
                    let row = embedding
                        .get(t as usize * d..(t as usize + 1) * d)
                        .ok_or_else(|| format!("token id {t} outside the vocabulary"))?;
                    base.extend_from_slice(row);
                    if t == self.image_token {
                        if let Ok(r) = seq.image_pos.binary_search(&(start_pos + i)) {
                            at.push(i as u32);
                            rows.extend_from_slice(&seq.rows[r * d..(r + 1) * d]);
                        }
                    }
                }
                let x = scatter_rows(base, tokens.len(), d, &at, rows, *mode)?;
                let mut prog = text.lock();
                std::mem::swap(&mut prog.state, &mut seq.state);
                let mut logits = Ok(Vec::new());
                for (i, row) in x.chunks_exact(d).enumerate() {
                    let inputs = vec![Tensor::F32(row.to_vec()), Tensor::U32(vec![(start_pos + i) as u32])];
                    logits = prog
                        .run(inputs)
                        .and_then(|o| o.into_iter().next().ok_or("text program returned nothing".to_string())?.into_f32());
                    if logits.is_err() {
                        break;
                    }
                }
                std::mem::swap(&mut prog.state, &mut seq.state);
                logits
            }
        }
    }
}

pub fn load(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let mode = mode.into();
    crate::graph::avenues::family_gate(
        crate::graph::avenues::Family::Idefics3Siglip,
        crate::graph::avenues::resolved_backend(mode),
    )?;
    let weights = artifacts.weights_path.as_deref().ok_or("no weight file")?;
    if !matches!(
        weights.extension().and_then(|e| e.to_str()),
        Some("safetensors" | "json")
    ) {
        return Err(format!(
            "{}: Idefics3/SmolVLM loads safetensors weights only",
            weights.display()
        ));
    }
    let config_path = artifacts.config_path.as_deref().ok_or("no config.json")?;
    let pre_path = config_path.with_file_name("preprocessor_config.json");
    let pre: Value = std::fs::read(&pre_path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
        .map_err(|e| format!("{}: {e}", pre_path.display()))?;
    let pre = Preprocess::from_json(&pre)?;
    let tokenizer = load_json_tokenizer(
        artifacts
            .tokenizer_path
            .as_deref()
            .ok_or("Idefics3/SmolVLM needs tokenizer.json")?,
    )?;
    let image_token = cfg["image_token_id"]
        .as_u64()
        .ok_or("config.json: image_token_id missing")? as u32;
    for tok in [FAKE, GLOBAL, IMAGE] {
        if !tokenizer.vocab.contains_key(tok) {
            return Err(format!("tokenizer lacks the {tok} token"));
        }
    }
    if tokenizer.vocab[IMAGE] != image_token {
        return Err(format!(
            "config.json image_token_id {image_token} is not the tokenizer's {IMAGE}"
        ));
    }

    crate::model::prequant::check(Some(config_path), weights.parent().unwrap_or(config_path))?;
    let (manifest, data) = UniversalModelLoader::read_safetensors(weights)?;
    let get = |n: &str| manifest.extract_tensor_f32(&data, n);
    // config.json stores only fields that differ from `Idefics3VisionConfig`'s defaults.
    let mut vcfg = serde_json::json!({
        "hidden_size": 1152, "intermediate_size": 3072, "num_hidden_layers": 12,
        "num_attention_heads": 16, "image_size": 224, "patch_size": 32,
    });
    if let Some(set) = cfg["vision_config"].as_object() {
        for (k, v) in set {
            vcfg[k] = v.clone();
        }
    }
    let vision = SiglipVision::load(&vcfg, "model.vision_model", &get)?;
    if vision.image_size != pre.tile {
        return Err(format!(
            "max_image_size {} differs from the vision encoder's image_size {}",
            pre.tile, vision.image_size
        ));
    }
    let scale = cfg["scale_factor"].as_u64().unwrap_or(2) as usize;
    let grid = vision.image_size / vision.patch;
    if scale == 0 || grid % scale != 0 {
        return Err(format!(
            "scale_factor {scale} does not divide the {grid}-patch grid"
        ));
    }
    let proj = get("model.connector.modality_projection.proj.weight")?;

    let text_cfg = &cfg["text_config"];
    let hf: HfConfigJson = serde_json::from_value(text_cfg.clone())
        .map_err(|e| format!("config.json text_config: {e}"))?;
    // ponytail: f32 text weights (per-token decode measured 6x faster than this engine's bf16
    // GEMV); bf16 when a VLM's text model is multi-GB.
    let mut tcfg = GraphLmConfig::from_hf(&hf, QuantMode::Uncompressed);
    if let Some(n) = text_cfg["max_position_embeddings"].as_u64() {
        tcfg.max_seq_len = n as usize;
    }
    if proj.len() != tcfg.hidden_dim * vision.hidden * scale * scale {
        return Err("connector projection shape does not match the text/vision widths".into());
    }
    // The text model's tensors under the names GraphLm reads (`model.*`, `lm_head.weight`).
    let mut text_manifest = manifest.clone();
    text_manifest.tensors = manifest
        .tensors
        .into_iter()
        .filter_map(|(k, mut t)| {
            let k = match k.strip_prefix("model.text_model.") {
                Some(rest) => format!("model.{rest}"),
                None if k == "lm_head.weight" => k,
                None => return None,
            };
            t.name = k.clone();
            Some((k, t))
        })
        .collect();
    let text = GraphLm::from_manifest(tcfg, &text_manifest, &data)?;
    let (image_seq_len, context_len) = (grid * grid / (scale * scale), text.config.max_seq_len);
    let engine = Engine::new(vision, proj, scale, text, image_token, mode)?;
    Ok(Arc::new(Idefics3 {
        image_seq_len,
        context_len,
        engine,
        tokenizer,
        image_token,
        pre,
    }))
}

impl Idefics3 {

    /// The prompt's image bytes, each placeholder's expansion ids, and the expanded prompt length,
    /// from cheap decodes only (no SigLIP): a prompt that can't fit the context Errs here, not
    /// after minutes of tile encoding and a huge KV reservation.
    #[allow(clippy::type_complexity)]
    fn expand<'m>(
        &self,
        ids: &[u32],
        media: &'m [MediaInput],
    ) -> Result<(Vec<&'m [u8]>, Vec<Vec<u32>>, usize), String> {
        let images: Vec<&[u8]> = media
            .iter()
            .map(|m| match m {
                MediaInput::Image(bytes) => Ok(bytes.as_slice()),
                MediaInput::Audio(_) => Err("Idefics3/SmolVLM takes images, not audio"),
            })
            .collect::<Result<_, _>>()?;
        let placeholders = ids.iter().filter(|&&t| t == self.image_token).count();
        if placeholders != images.len() {
            return Err(format!(
                "prompt has {placeholders} {IMAGE} placeholder(s) for {} image(s)",
                images.len()
            ));
        }
        let hf = self
            .tokenizer
            .hf
            .as_ref()
            .ok_or("tokenizer.json required")?;
        let mut expansions = Vec::with_capacity(images.len());
        let mut total = ids.len() - images.len();
        for (i, bytes) in images.iter().enumerate() {
            let img = ImageBuffer::decode(bytes)?;
            let (r, c) = self.pre.grid(img.width, img.height);
            let text = placeholder(r, c, self.image_seq_len);
            let enc = hf.encode_fast(text, false).map_err(|e| e.to_string())?;
            total += enc.get_ids().len();
            if total >= self.context_len {
                return Err(format!(
                    "prompt with {} image(s) reaches {total} tokens by image {}, over the {}-token context",
                    images.len(),
                    i + 1,
                    self.context_len
                ));
            }
            expansions.push(enc.get_ids().to_vec());
        }
        Ok((images, expansions, total))
    }
}

impl NativeLm for Idefics3 {
    fn backend_label(&self) -> String {
        match &self.engine.kind {
            Kind::Cpu { .. } => crate::scheduler::cpu_backend_label(),
            Kind::Gpu { text, .. } => text.lock().backend_label(),
        }
    }

    fn device(&self) -> crate::hardware::Device {
        match &self.engine.kind {
            Kind::Cpu { .. } => crate::hardware::Device::Cpu,
            Kind::Gpu { text, .. } => text.lock().primary_device(),
        }
    }

    fn summary(&self) -> String {
        let c = &self.engine.cfg;
        format!(
            "Idefics3: SigLIP {}px tiles ({} image tokens each) + Llama {} layers, hidden {}, f32 weights",
            self.engine.image_size, self.image_seq_len, c.num_layers, c.hidden_dim
        )
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_len(&self) -> usize {
        self.context_len
    }

    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        let (images, expansions, total) = self.expand(&ids, media)?;
        let (mut images, mut expansions) = (images.into_iter(), expansions.into_iter());
        let (mut out, mut rows) = (Vec::with_capacity(total), Vec::new());
        for t in ids {
            if t != self.image_token {
                out.push(t);
                continue;
            }
            let img = ImageBuffer::decode(images.next().expect("one image per placeholder"))?;
            for tile in &self.pre.tiles(&img).2 {
                rows.extend(self.engine.embed_tile(&self.pre.pixels(tile))?);
            }
            out.extend(expansions.next().expect("one expansion per image"));
        }
        let image_pos: Vec<usize> = (0..out.len())
            .filter(|&i| out[i] == self.image_token)
            .collect();
        let d = self.engine.cfg.hidden_dim;
        if image_pos.len() * d != rows.len() {
            return Err(format!(
                "{} {IMAGE} tokens for {} image rows",
                image_pos.len(),
                rows.len() / d
            ));
        }
        let seq = self.engine.new_seq(out.len(), image_pos, rows);
        Ok((out, Box::new(seq)))
    }

    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        Ok(self.expand(&ids, media)?.2)
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        let seq = state
            .downcast_mut::<Seq>()
            .ok_or("sequence state is not Idefics3's")?;
        self.engine.forward(tokens, start_pos, seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::Device;

    fn pre(split: bool) -> Preprocess {
        Preprocess {
            longest_edge: 2048,
            tile: 512,
            split,
            lanczos: true,
            rescale: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }

    /// Tile grid and count follow `Idefics3ImageProcessor` (resize longest edge to 2048, sides up
    /// to multiples of 512), and the crops are the resized image's pixels in row-major order.
    #[test]
    fn tiles_follow_the_hf_split() {
        let mut img = ImageBuffer::new(300, 200);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        // 300x200 → 2048x1366 → 2048x1536: 3 rows × 4 cols + global.
        let (r, c, tiles) = pre(true).tiles(&img);
        assert_eq!((r, c, tiles.len()), (3, 4, 13));
        assert!(tiles.iter().all(|t| (t.width, t.height) == (512, 512)));
        let full = img.resize_lanczos(2048, 1366).resize_lanczos(2048, 1536);
        assert_eq!(tiles[5].get_pixel(3, 4), full.get_pixel(512 + 3, 512 + 4));
        // Portrait, and an image already within one tile after resizing (no split).
        let (r, c, tiles) = pre(true).tiles(&ImageBuffer::new(100, 700));
        assert_eq!((r, c, tiles.len()), (4, 1, 5));
        let (r, c, tiles) = pre(false).tiles(&img);
        assert_eq!((r, c, tiles.len()), (0, 0, 1));
        let small = Preprocess {
            longest_edge: 512,
            ..pre(true)
        };
        assert_eq!(small.tiles(&img).0, 0);
    }

    #[test]
    fn placeholder_matches_the_hf_processor() {
        assert_eq!(
            placeholder(0, 0, 2),
            "<fake_token_around_image><global-img><image><image><fake_token_around_image>"
        );
        assert_eq!(
            placeholder(1, 2, 1),
            "<fake_token_around_image><row_1_col_1><image><fake_token_around_image><row_1_col_2><image>\n\n<fake_token_around_image><global-img><image><fake_token_around_image>"
        );
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale
    }

    const IMG: u32 = 5;

    fn tiny_engine(mode: crate::hardware::PlacementMode) -> Engine {
        use crate::model::siglip::tiny;
        let text = GraphLm::new(GraphLmConfig::smollm_1m());
        let proj = tiny::values("connector", text.config.hidden_dim * tiny::HIDDEN * 4)
            .into_iter()
            .map(|v| v * 0.1)
            .collect();
        Engine::new(tiny::vision(), proj, 2, text, IMG, mode).expect("tiny Idefics3 engine")
    }

    /// The Metal tower + pixel-shuffle connector and the stateful text program (image rows scattered
    /// over the token embeddings, a prefill chunk then two decode steps) match the host forward.
    #[test]
    fn metal_programs_match_the_host_forward() {
        use crate::hardware::{available_devices, cpu_compute, PlacementMode};
        if !available_devices().contains(&Device::Metal) {
            println!("SKIP: no Metal device");
            return;
        }
        let _allow = cpu_compute::allow();
        let cpu = tiny_engine(PlacementMode::Pin(Device::Cpu));
        // The tower + connector and text graphs compiled here reproduce `graph::shape::infer`.
        let gpu = crate::graph::shape::assert_compiled_shapes(|| {
            tiny_engine(PlacementMode::Pin(Device::Metal))
        });
        let Kind::Gpu { vision, text, .. } = &gpu.kind else { panic!("not a GPU engine") };
        assert_eq!(vision.lock().cpu_node_count() + text.lock().cpu_node_count(), 0, "nodes run on the CPU");
        let pixels = crate::model::siglip::tiny::values("pixels", 3 * 16 * 16);
        let (rc, rg) = (cpu.embed_tile(&pixels).unwrap(), gpu.embed_tile(&pixels).unwrap());
        assert_eq!(rc.len(), 4 * 128);
        let e = rel_err(&rg, &rc);
        println!("idefics3 tile rows: rel err {e:e}");
        assert!(e < 1e-3, "tile rows differ: {e}");
        let abs = rg.iter().zip(&rc).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(abs < 1e-3, "tile rows max |d| {abs}");

        let image_pos = vec![2, 3, 4, 5];
        let (mut sc, mut sg) = (
            cpu.new_seq(9, image_pos.clone(), rc),
            gpu.new_seq(9, image_pos, rg),
        );
        for (tokens, start) in [(vec![10, 11, IMG, IMG, IMG, IMG, 12], 0), (vec![13], 7), (vec![14], 8)] {
            let lc = cpu.forward(&tokens, start, &mut sc).unwrap();
            let lg = gpu.forward(&tokens, start, &mut sg).unwrap();
            let e = rel_err(&lg, &lc);
            println!("idefics3 text at {start}: rel err {e:e}");
            assert!(e < 1e-3, "logits differ at {start}: {e}");
            // Top-1 up to fp noise: a random tiny model has near-tied logits.
            let best = lc.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            let gmax = lg.iter().fold(f32::MIN, |m, v| m.max(*v));
            assert!(gmax - lg[best] < 1e-3, "top-1 at {start}");
        }
    }

    /// Many images that can't fit the context Err from cheap decodes, before any SigLIP pass or
    /// KV reservation (each tiny image still expands to 17 tiles, ~1100 tokens).
    #[test]
    fn prompt_over_context_errs_before_encoding() {
        let dir = std::path::Path::new(
            "disk/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/main",
        );
        if cfg!(debug_assertions) || !dir.join("model.safetensors").exists() {
            println!("SKIP: needs --release and SmolVLM-256M-Instruct in disk/");
            return;
        }
        let artifacts = ModelArtifacts {
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            ..Default::default()
        };
        let cfg: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let m = load(&artifacts, &cfg, Device::Cpu).unwrap();
        let image = m.tokenizer().vocab[IMAGE];
        let png = ImageBuffer::new(4, 4).to_png();
        let started = std::time::Instant::now();
        let err = m
            .prepare(vec![image; 200], &vec![MediaInput::Image(png); 200])
            .err()
            .expect("200 images exceed the context");
        assert!(err.contains("context"), "{err}");
        assert!(
            started.elapsed().as_secs() < 5,
            "took {:?}",
            started.elapsed()
        );
    }
}
