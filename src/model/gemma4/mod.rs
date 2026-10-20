//! Gemma 4 any-to-any checkpoints (`model_type: "gemma4"`, `Gemma4ForConditionalGeneration`:
//! google/gemma-4-E2B-it, E4B-it; `gemma4_unified`, `Gemma4UnifiedForConditionalGeneration`:
//! google/gemma-4-12B-it) as a [`NativeLm`]: text, images and audio in, text out, on CPU with bf16
//! weights read straight from the memory-mapped safetensors file, or dequantized from a llama.cpp
//! `gemma4` GGUF and its `mmproj` ([`load_gguf`]).

mod audio;
mod gguf;
pub mod graph;
mod moe;
pub(crate) mod ops;
pub mod seqcls;
mod text;
pub use audio::tiny::{graph as tiny_audio_graph, wav_graph as tiny_audio_wav_graph};
pub use text::{tiny_decode_graph, tiny_moe_decode_graph};
pub use vision::tiny::graph as tiny_vision_graph;
mod unified;
mod vision;

pub use gguf::{gguf_decode_fixture, load_gguf};

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
    pub program: Option<Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>>,
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
    program_state: crate::graph::StateStore,
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
    let mode = mode.into();
    let gpu = crate::model::vision_graph::is_gpu(mode);
    let (vision, audio): (Box<dyn Encoder>, Box<dyn Encoder>) = if unified {
        let v = unified::VisionEmbedder::load(&w, vc, &processor)
            .map_err(|e| format!("gemma4 vision: {e}"))?;
        let a = unified::AudioEmbedder::load(&w, ac, &processor)
            .map_err(|e| format!("gemma4 audio: {e}"))?;
        if gpu {
            (Box::new(v.on_gpu(mode)), Box::new(a.on_gpu(mode)))
        } else {
            (Box::new(v), Box::new(a))
        }
    } else {
        let v = Vision::load(&w, vc, &processor).map_err(|e| format!("gemma4 vision: {e}"))?;
        let a = Audio::load(&w, ac, &processor).map_err(|e| format!("gemma4 audio: {e}"))?;
        if gpu {
            (Box::new(v.on_gpu(mode)), Box::new(a.on_gpu(mode)))
        } else {
            (Box::new(v), Box::new(a))
        }
    };
    let id = |k: &str| {
        cfg[k]
            .as_u64()
            .map(|v| v as u32)
            .ok_or_else(|| format!("config.{k} missing"))
    };
    let program = compile_gemma4_program(&text, mode)?;
    Ok(Arc::new(Gemma4 {
        tokenizer,
        text,
        program,
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
        if let Some(prog) = &self.program {
            return prog.lock().backend_label();
        }
        format!("CPU (NEON bf16), {} threads", crate::model::par::threads())
    }

    fn device(&self) -> Device {
        if let Some(prog) = &self.program {
            return prog.lock().primary_device();
        }
        Device::Cpu
    }

    fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        Some(self.program.as_ref()?.lock().placement_summary())
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
                    crate::graph::avenues::gate(crate::graph::avenues::Family::Gemma4Media)?;
                    let (soft, n) = self.vision.encode(b).map_err(|e| format!("image: {e}"))?;
                    (soft, n, self.boi, self.eoi)
                }
                MediaInput::Audio(b) => {
                    crate::graph::avenues::gate(crate::graph::avenues::Family::Audio)?;
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
        let state = State {
            cache: self.text.new_cache(),
            program_state: crate::graph::StateStore::new(),
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
        crate::model::native::logits_of(self.pass(tokens, start_pos, state, false)?)
    }
    fn forward_greedy(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<crate::graph::Tensor, String> {
        self.pass(tokens, start_pos, state, true)
    }
}

/// Rows per batched prefill pass on the Metal program, and the most a pass may grow to when an image
/// block (attended bidirectionally, so whole in one pass) straddles the boundary.
const PREFILL_ROWS: usize = 64;
const PREFILL_MAX_ROWS: usize = 512;

impl Gemma4 {
    /// [`NativeLm::forward`]; `greedy` lets the decode program pick the token on the device
    /// (`U32([token])`, see [`crate::graph::program::GraphProgram::run_greedy`]), else `F32(logits)`.
    fn pass(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
        greedy: bool,
    ) -> Result<crate::graph::Tensor, String> {
        use crate::graph::Tensor;
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
            return Ok(Tensor::F32(st.logits.clone())); // run ahead with an earlier chunk's image block
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
        let logits = match &self.program {
            // A straddled image block caches its logits for the run-ahead: no device pick then.
            Some(prog) => {
                self.run_program(prog, st, &x, &ple_ids, from, greedy && stop == end)?
            }
            None if self.text.has_moe() => {
                return Err("gemma4 MoE: host forward not implemented; the graph program runs on Metal and the CPU graph executor".into())
            }
            None => Tensor::F32(
                self.text
                    .forward(x, &ple_ids, from, &mut st.cache, &st.blocks),
            ),
        };
        st.done = stop;
        if stop > end {
            st.logits = crate::model::native::logits_of(logits.clone())?;
        }
        Ok(logits)
    }
    /// Runs rows `[from, from + ids.len())` (`x` = their embeddings, `ids` = per-layer-embedding ids,
    /// `u32::MAX` for media rows) through the graph program inside the sequence's state; the last row's
    /// logits, or its device-picked token when `greedy` ([`Self::pass`]).
    fn run_program(
        &self,
        prog: &parking_lot::Mutex<crate::graph::program::GraphProgram>,
        st: &mut State,
        x: &[f32],
        ids: &[u32],
        from: usize,
        greedy: bool,
    ) -> Result<crate::graph::Tensor, String> {
        let h = self.text.hidden;
        let mut p = prog.lock();
        std::mem::swap(&mut p.state, &mut st.program_state);
        let mut last = Ok(crate::graph::Tensor::F32(Vec::new()));
        let mut at = 0;
        while at < ids.len() && last.is_ok() {
            let mut end = (at + PREFILL_ROWS).min(ids.len());
            if let Some(b) = st
                .blocks
                .iter()
                .find(|b| b.start < from + end && from + end < b.end)
            {
                end = b.end - from;
            }
            last = if end - at > PREFILL_MAX_ROWS {
                Err(format!(
                    "image block of {} rows exceeds the {PREFILL_MAX_ROWS}-row prefill chunk",
                    end - at
                ))
            } else {
                self.run_chunk(
                    &mut p,
                    &st.blocks,
                    &x[at * h..end * h],
                    &ids[at..end],
                    from + at,
                    greedy && end == ids.len(),
                )
            };
            at = end;
        }
        std::mem::swap(&mut p.state, &mut st.program_state);
        last
    }

    /// One pass over `ids.len()` consecutive rows from position `pos0`: the single-row decode program for
    /// one row (or when the program is not one Metal segment), else the batched prefill graph. `greedy`:
    /// the last single-row step goes through [`crate::graph::program::GraphProgram::run_greedy`].
    fn run_chunk(
        &self,
        p: &mut crate::graph::program::GraphProgram,
        blocks: &[std::ops::Range<usize>],
        x: &[f32],
        ids: &[u32],
        pos0: usize,
        greedy: bool,
    ) -> Result<crate::graph::Tensor, String> {
        use crate::graph::Tensor;
        let (h, m) = (self.text.hidden, ids.len());
        let mut ple = Vec::new();
        if let Some(pl) = &self.text.ple {
            let (pd, nl) = (pl.dim, self.text.layers_n);
            for &id in ids {
                let mut e = vec![0.0f32; nl * pd];
                let id = if (id as usize) < pl.embed.rows {
                    id
                } else {
                    self.text.pad_id
                };
                pl.embed.get_row(id as usize, &mut e);
                let scale = (pd as f32).sqrt();
                e.iter_mut().for_each(|v| *v *= scale);
                ple.extend(e);
            }
        }
        let ple_row = ple.len() / m;
        let logits = |res: Vec<Tensor>| match res.into_iter().next() {
            Some(t @ (Tensor::F32(_) | Tensor::U32(_))) => Ok(t),
            _ => Err("gemma4: program did not return F32 logits".to_string()),
        };
        if m > 1 && p.is_single_metal() {
            let g = self.text.to_prefill_graph(m)?;
            let end = |pos: usize| match blocks.iter().find(|b| b.contains(&pos)) {
                Some(b) => b.end as u32,
                None => pos as u32 + 1,
            };
            let mut inputs = vec![
                Tensor::F32(x.to_vec()),
                Tensor::U32((pos0..pos0 + m).map(|q| q as u32).collect()),
            ];
            if self.text.ple.is_some() {
                inputs.push(Tensor::F32(ple));
            }
            inputs.push(Tensor::U32((pos0..pos0 + m).map(end).collect()));
            return logits(p.run_reshaped(&g, inputs)?);
        }
        if m > 1 && blocks.iter().any(|b| b.start < pos0 + m && pos0 < b.end) {
            return Err("gpu-only: image blocks are attended bidirectionally, which needs the Metal program; rerun with --backend cpu".into());
        }
        let mut last = Tensor::F32(Vec::new());
        for (i, emb) in x.chunks_exact(h).enumerate() {
            let mut inputs = vec![
                Tensor::F32(emb.to_vec()),
                Tensor::U32(vec![(pos0 + i) as u32]),
            ];
            if self.text.ple.is_some() {
                inputs.push(Tensor::F32(ple[i * ple_row..(i + 1) * ple_row].to_vec()));
            }
            last = logits(if greedy && i + 1 == m {
                p.run_greedy(inputs, self.tokenizer.vocab_size())?
            } else {
                p.run(inputs)?
            })?;
        }
        Ok(last)
    }
}

pub(crate) fn compile_gemma4_program(
    text: &Arc<TextModel>,
    mode: crate::hardware::PlacementMode,
) -> Result<Option<Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>>, String> {
    if matches!(mode, crate::hardware::PlacementMode::Pin(Device::Cpu)) {
        return Ok(None);
    }
    crate::graph::avenues::family_gate(
        crate::graph::avenues::Family::Gemma4,
        crate::graph::avenues::resolved_backend(mode),
    )?;
    let (graph, weights) = text
        .to_decode_graph()
        .map_err(|e| format!("Gemma 4: {e}"))?;
    crate::graph::program::GraphProgram::compile(graph, weights, mode)
        .map(|prog| Some(Arc::new(parking_lot::Mutex::new(prog))))
        .map_err(|e| format!("Gemma 4: {e}"))
}
