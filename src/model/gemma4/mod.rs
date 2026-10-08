//! Gemma 4 any-to-any checkpoints (`model_type: "gemma4"`, `Gemma4ForConditionalGeneration`:
//! google/gemma-4-E2B-it, E4B-it; `gemma4_unified`, `Gemma4UnifiedForConditionalGeneration`:
//! google/gemma-4-12B-it) as a [`NativeLm`]: text, images and audio in, text out, on CPU with bf16
//! weights read straight from the memory-mapped safetensors file, or dequantized from a llama.cpp
//! `gemma4` GGUF and its `mmproj` ([`load_gguf`]).

mod audio;
mod gguf;
pub(crate) mod ops;
pub mod seqcls;
mod text;
mod unified;
mod vision;

#[cfg(target_os = "macos")]
pub mod metal;
pub use gguf::load_gguf;
#[cfg(target_os = "macos")]
pub use metal::MetalGemma4Decoder;

use crate::hardware::Device;
use crate::model::hub::ModelArtifacts;
use crate::model::native::{MediaInput, NativeLm, SeqState};
use crate::model::tokenizer::Tokenizer;
use crate::model::weights::Weights;
use audio::Audio;
use serde_json::Value;
use std::ops::Range;
use std::sync::Arc;
use text::{TextCache, TextModel};
use vision::Vision;

/// A media encoder: file bytes -> `[n, text_hidden]` soft tokens and `n`.
trait Encoder: Send + Sync {
    fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String>;
    /// The `n` [`Self::encode`] returns, from the file's decode alone (no model compute).
    fn count(&self, bytes: &[u8]) -> Result<usize, String>;
}

/// The encoder of a modality the checkpoint ships no projector for.
struct NoEncoder(String);

impl Encoder for NoEncoder {
    fn encode(&self, _: &[u8]) -> Result<(Vec<f32>, usize), String> {
        Err(self.0.clone())
    }
    fn count(&self, _: &[u8]) -> Result<usize, String> {
        Err(self.0.clone())
    }
}

impl Encoder for Vision {
    fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        Vision::encode(self, bytes)
    }
    fn count(&self, bytes: &[u8]) -> Result<usize, String> {
        Vision::count(self, bytes)
    }
}

impl Encoder for Audio {
    fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        Audio::encode(self, bytes)
    }
    fn count(&self, bytes: &[u8]) -> Result<usize, String> {
        Audio::count(self, bytes)
    }
}

pub struct Gemma4 {
    tokenizer: Tokenizer,
    text: Arc<TextModel>,
    #[cfg(target_os = "macos")]
    pub metal_decoder: Option<Arc<MetalGemma4Decoder>>,
    unified: bool,
    vision: Box<dyn Encoder>,
    audio: Box<dyn Encoder>,
    image_token: u32,
    audio_token: u32,
    /// Begin/end markers around one image's (audio clip's) soft tokens.
    boi: u32,
    eoi: u32,
    boa: u32,
    eoa: u32,
}
/// Per-sequence state: the decoder caches plus the media soft tokens, already projected to the
/// text width, for the placeholder positions `prepare` laid out.
struct State {
    cache: TextCache,
    #[cfg(target_os = "macos")]
    metal_kv: Option<crate::backend::metal::MetalKvCache>,
    /// Positions of media soft tokens, ascending; row `i` of `media` fills `media_pos[i]`.
    media_pos: Vec<usize>,
    media: Vec<f32>,
    /// Image soft-token position ranges attended bidirectionally (empty unless the text model is
    /// `bidir_vision`).
    blocks: Vec<Range<usize>>,
    /// End of the positions `forward` was asked for, and of those already run: a chunk ending
    /// inside an image block runs on to the block's end (its rows are all media), and `logits`
    /// holds the result for the chunks it covered.
    asked: usize,
    done: usize,
    logits: Vec<f32>,
}

pub fn load(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let weights_path = artifacts.weights_path.as_deref().ok_or("no weights file")?;
    if weights_path.extension().is_some_and(|e| e == "gguf") {
        // `load_native` tries `load_gguf` first: this GGUF is another llama.cpp architecture.
        return Err(format!(
            "Gemma 4 loads safetensors checkpoints or GGUFs of architecture 'gemma4'; {} is neither",
            weights_path.display()
        ));
    }
    let tok_path = artifacts
        .tokenizer_path
        .as_deref()
        .ok_or("no tokenizer.json")?;
    let tokenizer = crate::model::hf_loader::load_json_tokenizer(tok_path)?;
    let w = Weights::open(weights_path)?;
    let text = Arc::new(
        TextModel::load(&w, &cfg["text_config"]).map_err(|e| format!("gemma4 text: {e}"))?,
    );
    let unified = cfg["model_type"] == "gemma4_unified";
    let processor: Value = artifacts
        .config_path
        .as_deref()
        .and_then(|p| std::fs::read(p.with_file_name("processor_config.json")).ok())
        .map(|b| serde_json::from_slice(&b).map_err(|e| format!("processor_config.json: {e}")))
        .transpose()?
        .unwrap_or(Value::Null);
    let (vc, ac) = (&cfg["vision_config"], &cfg["audio_config"]);
    let (vision, audio): (Box<dyn Encoder>, Box<dyn Encoder>) = if unified {
        (
            Box::new(
                unified::VisionEmbedder::load(&w, vc, &processor)
                    .map_err(|e| format!("gemma4 vision: {e}"))?,
            ),
            Box::new(
                unified::AudioEmbedder::load(&w, ac, &processor)
                    .map_err(|e| format!("gemma4 audio: {e}"))?,
            ),
        )
    } else {
        (
            Box::new(Vision::load(&w, vc, &processor).map_err(|e| format!("gemma4 vision: {e}"))?),
            Box::new(Audio::load(&w, ac, &processor).map_err(|e| format!("gemma4 audio: {e}"))?),
        )
    };
    let id = |k: &str| {
        cfg[k]
            .as_u64()
            .map(|v| v as u32)
            .ok_or_else(|| format!("config.{k} missing"))
    };
    let mode = mode.into();
    let want_metal = match mode {
        crate::hardware::PlacementMode::Pin(Device::Metal) => true,
        crate::hardware::PlacementMode::Auto => {
            crate::hardware::available_devices().contains(&Device::Metal)
        }
        _ => false,
    };
    #[cfg(target_os = "macos")]
    let metal_decoder = if want_metal {
        match MetalGemma4Decoder::new(&text) {
            Ok(dec) => Some(Arc::new(dec)),
            Err(e) => {
                if matches!(mode, crate::hardware::PlacementMode::Pin(Device::Metal)) {
                    return Err(format!("Metal decoder initialization failed: {e}"));
                }
                tracing::warn!("Metal initialization failed, falling back to CPU: {e}");
                None
            }
        }
    } else {
        None
    };
    Ok(Arc::new(Gemma4 {
        tokenizer,
        text,
        #[cfg(target_os = "macos")]
        metal_decoder,
        unified,
        vision,
        audio,
        image_token: id("image_token_id")?,
        audio_token: id("audio_token_id")?,
        boi: id("boi_token_id")?,
        eoi: id("eoi_token_id")?,
        boa: id("boa_token_id")?,
        // gemma-4-12B's config spells it `eoa_token_index`.
        eoa: id("eoa_token_id").or_else(|_| id("eoa_token_index"))?,
    }))
}

impl Gemma4 {
    /// Each media item's soft-token count (from image dimensions / WAV sample counts, no
    /// encoder) and the expanded prompt length; Errs when the media do not match the prompt's
    /// placeholders or the prompt cannot fit the context.
    fn count(&self, ids: &[u32], media: &[MediaInput]) -> Result<(Vec<usize>, usize), String> {
        let slots: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&t| t == self.image_token || t == self.audio_token)
            .collect();
        if slots.len() != media.len() {
            return Err(format!(
                "prompt has {} image/audio placeholders but {} media items were attached",
                slots.len(),
                media.len()
            ));
        }
        let mut total = ids.len();
        let mut counts = Vec::with_capacity(media.len());
        for (&t, m) in slots.iter().zip(media) {
            let n = match m {
                MediaInput::Image(b) if t == self.image_token => {
                    self.vision.count(b).map_err(|e| format!("image: {e}"))?
                }
                MediaInput::Audio(b) if t == self.audio_token => {
                    self.audio.count(b).map_err(|e| format!("audio: {e}"))?
                }
                _ => {
                    return Err(
                        "media order does not match the prompt's image/audio placeholders".into(),
                    )
                }
            };
            total += n + 1; // placeholder -> begin marker + n soft tokens + end marker
            counts.push(n);
        }
        if total >= self.context_len() {
            return Err(format!(
                "prompt with its media is {total} tokens; the context holds {}",
                self.context_len()
            ));
        }
        Ok((counts, total))
    }
}

impl NativeLm for Gemma4 {
    fn backend_label(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(dec) = &self.metal_decoder {
            return format!("Metal ({})", dec.device_name());
        }
        format!("CPU (NEON bf16), {} threads", crate::model::par::threads())
    }

    fn device(&self) -> Device {
        #[cfg(target_os = "macos")]
        if self.metal_decoder.is_some() {
            return Device::Metal;
        }
        Device::Cpu
    }
    fn summary(&self) -> String {
        let t = &self.text;
        format!(
            "Gemma 4{}: bf16 weights, {} layers, hidden {}, vocab {}",
            if self.unified { " unified" } else { "" },
            t.layers_n,
            t.hidden,
            t.vocab
        )
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_len(&self) -> usize {
        self.text.max_pos
    }

    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        Ok(self.count(&ids, media)?.1)
    }

    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        let (counts, total) = self.count(&ids, media)?;
        let (mut out, mut media_pos, mut rows, mut blocks) = (
            Vec::with_capacity(total),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut items = media.iter().zip(counts);
        for t in ids {
            if t != self.image_token && t != self.audio_token {
                out.push(t);
                continue;
            }
            // `count` checked the placeholder/media pairing.
            let (item, want) = items.next().ok_or("media/placeholder mismatch")?;
            let (soft, n, open, close) = match item {
                MediaInput::Image(b) => {
                    let (soft, n) = self.vision.encode(b).map_err(|e| format!("image: {e}"))?;
                    (soft, n, self.boi, self.eoi)
                }
                MediaInput::Audio(b) => {
                    let (soft, n) = self.audio.encode(b).map_err(|e| format!("audio: {e}"))?;
                    (soft, n, self.boa, self.eoa)
                }
            };
            if n != want {
                return Err(format!(
                    "gemma4: media encoded to {n} soft tokens, {want} were counted"
                ));
            }
            out.push(open);
            media_pos.extend(out.len()..out.len() + n);
            if t == self.image_token && self.text.bidir_vision {
                blocks.push(out.len()..out.len() + n);
            }
            out.extend(std::iter::repeat_n(t, n));
            out.push(close);
            rows.extend(soft);
        }
        #[cfg(target_os = "macos")]
        let metal_kv = if let Some(dec) = &self.metal_decoder {
            dec.new_kv_cache().ok()
        } else {
            None
        };
        let state = State {
            cache: self.text.new_cache(),
            #[cfg(target_os = "macos")]
            metal_kv,
            media_pos,
            media: rows,
            blocks,
            asked: 0,
            done: 0,
            logits: Vec::new(),
        };
        Ok((out, Box::new(state)))
    }
    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        let st = state
            .downcast_mut::<State>()
            .ok_or("gemma4: foreign sequence state")?;
        if tokens.is_empty() {
            return Err("gemma4: forward with no tokens".into());
        }
        if start_pos != st.asked {
            return Err(format!(
                "gemma4: forward at {start_pos} but the sequence continues at {}",
                st.asked
            ));
        }
        let end = start_pos + tokens.len();
        st.asked = end;
        if end <= st.done {
            return Ok(st.logits.clone()); // run ahead with an earlier chunk's image block
        }
        #[cfg(target_os = "macos")]
        if let Some(dec) = &self.metal_decoder {
            if st.media_pos.is_empty() {
                if let Some(kv) = &mut st.metal_kv {
                    let logits = dec.forward_tokens(tokens, start_pos, kv)?;
                    st.asked = end;
                    st.done = end;
                    return Ok(logits);
                }
            }
        }
        // Bidirectional image blocks must be whole within one pass: finish a straddled block.
        let stop = match st.blocks.iter().find(|b| b.start < end && end < b.end) {
            Some(b) => b.end,
            None => end,
        };
        let from = st.done;
        let h = self.text.hidden;
        let mut x = vec![0.0f32; (stop - from) * h];
        let mut ple_ids = Vec::with_capacity(stop - from);
        for (r, pos) in (from..stop).enumerate() {
            let row = &mut x[r * h..(r + 1) * h];
            // Positions past `tokens` are the rest of an image block: media rows.
            let t = tokens
                .get(pos - start_pos)
                .copied()
                .unwrap_or(self.image_token);
            match st.media_pos.binary_search(&pos) {
                Ok(i) => {
                    row.copy_from_slice(&st.media[i * h..(i + 1) * h]);
                    ple_ids.push(u32::MAX); // media rows take the pad id's per-layer embedding
                }
                Err(_) if (t as usize) < self.text.vocab => {
                    self.text.embed_row(t, row);
                    ple_ids.push(t);
                }
                Err(_) => return Err(format!("gemma4: token id {t} outside the vocabulary")),
            }
        }
        let logits = self
            .text
            .forward(x, &ple_ids, from, &mut st.cache, &st.blocks);
        st.done = stop;
        if stop > end {
            st.logits = logits.clone();
        }
        Ok(logits)
    }
}
