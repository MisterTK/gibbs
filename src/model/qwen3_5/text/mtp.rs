//! Qwen3.5 multi-token prediction head (`mtp.*`) as a speculative-decoding [`Drafter`].
//!
//! Reference: vLLM `model_executor/models/qwen3_5_mtp.py` (`Qwen3_5MultiTokenPredictor`). The
//! step at position `p` reads the target's final-normed hidden state `h_p` and the next token:
//! `fc(concat(norm_e(embed(t_{p+1})), norm_h(h_p)))`, one gated full-attention decoder layer with
//! its own KV cache, then `norm`; the shared LM head turns that into logits for `t_{p+2}`, and
//! the normed output stands in for `h_{p+1}` at the next draft step.

use super::{Attention, Ffn, Layer, LayerState, Mixer, TextModel};
use crate::model::par::dot_bf16;
use crate::model::qwen3_5::{linear, linear_bf16, rms_norm, rms_norm_rows, Linear};
use crate::model::weights::Weights;
use crate::scheduler::spec::{DraftState, Drafter};
use serde_json::Value;
use std::sync::Arc;

pub(in crate::model::qwen3_5) struct Mtp {
    pub(super) fc: Linear,
    pub(super) pre_emb: Vec<f32>,
    pub(super) pre_hid: Vec<f32>,
    pub(super) layers: Vec<Layer>,
    pub(super) norm: Vec<f32>,
}

impl Mtp {
    /// The head of a `qwen3_5` text config `c`, `None` when the checkpoint ships none.
    pub(super) fn load(ck: &Weights, c: &Value) -> Result<Option<Self>, String> {
        let layers = c["mtp_num_hidden_layers"].as_u64().unwrap_or(0);
        // ponytail: one MTP layer (every released Qwen3.5); vLLM cycles `step % layers` for more,
        // with per-layer caches this drafter would have to keep.
        if layers != 1 || c["mtp_use_dedicated_embeddings"] == true || !ck.has("mtp.fc.weight") {
            return Ok(None);
        }
        let norm = |name: &str| -> Result<Vec<f32>, String> {
            Ok(ck.vec(name)?.into_iter().map(|w| w + 1.0).collect())
        };
        let (l, bias) = ("mtp.layers.0", c["attention_bias"] == true);
        let a = format!("{l}.self_attn");
        let layer = Layer {
            in_norm: norm(&format!("{l}.input_layernorm.weight"))?,
            post_norm: norm(&format!("{l}.post_attention_layernorm.weight"))?,
            mixer: Mixer::Full(Attention {
                qkv: linear(
                    ck,
                    &[
                        &format!("{a}.q_proj"),
                        &format!("{a}.k_proj"),
                        &format!("{a}.v_proj"),
                    ],
                    bias,
                )?,
                q_norm: norm(&format!("{a}.q_norm.weight"))?,
                k_norm: norm(&format!("{a}.k_norm.weight"))?,
                o: linear(ck, &[&format!("{a}.o_proj")], bias)?,
                indexer: None,
            }),
            ffn: Ffn::Dense {
                gate_up: linear(
                    ck,
                    &[&format!("{l}.mlp.gate_proj"), &format!("{l}.mlp.up_proj")],
                    false,
                )?,
                down: linear(ck, &[&format!("{l}.mlp.down_proj")], false)?,
            },
            hc: None,
        };
        let fc = linear(ck, &["mtp.fc"], false)?;
        if fc.cols != 2 * fc.rows {
            return Err(format!(
                "mtp.fc.weight is {}x{}, expected [d, 2d]",
                fc.rows, fc.cols
            ));
        }
        Ok(Some(Self {
            fc,
            pre_emb: norm("mtp.pre_fc_norm_embedding.weight")?,
            pre_hid: norm("mtp.pre_fc_norm_hidden.weight")?,
            layers: vec![layer],
            norm: norm("mtp.norm.weight")?,
        }))
    }
}

impl TextModel {
    /// One MTP pass: `tokens[i]` with normed hidden row `hid[i]` at position `start + i`, against
    /// the head's cache `kv` (positions `..start`); returns the head's normed output rows.
    fn mtp_step(
        &self,
        m: &Mtp,
        kv: &mut LayerState,
        tokens: &[u32],
        hid: &[f32],
        start: usize,
    ) -> Result<Vec<f32>, String> {
        let (n, d) = (tokens.len(), self.hidden);
        let mut x = vec![0.0f32; n * 2 * d];
        for (i, (&t, row)) in tokens.iter().zip(x.chunks_exact_mut(2 * d)).enumerate() {
            let (e, h) = row.split_at_mut(d);
            self.embed(t, e)?;
            rms_norm(e, &m.pre_emb, self.eps);
            h.copy_from_slice(&hid[i * d..][..d]);
            rms_norm(h, &m.pre_hid, self.eps);
        }
        let mut h = m.fc.forward(&x, n);
        // ponytail: 1-D positions (all three M-RoPE axes = sequence index), exact for text
        // prompts; image prompts' M-RoPE offsets only lower acceptance (verify keeps outputs).
        let pos: Vec<[u32; 3]> = (start..start + n).map(|p| [p as u32; 3]).collect();
        self.layer_forward(
            &m.layers[0],
            kv,
            &mut h,
            n,
            &self.rope_table(&pos),
            start,
            usize::MAX,
            false,
        );
        rms_norm_rows(&mut h, &m.norm, self.eps);
        Ok(h)
    }

    /// Draft logits after normed row `x`: ids below [`DRAFT_VOCAB`] and those in `seen`; every
    /// other id stays -inf.
    fn draft_logits(&self, x: &[f32], seen: &[u32], out: &mut Vec<f32>) {
        let head = &self.lm_head.as_ref().unwrap_or(&self.embed).parts[0];
        let n = DRAFT_VOCAB.min(head.rows);
        out.clear();
        out.resize(head.rows, f32::NEG_INFINITY);
        out[..n].copy_from_slice(&linear_bf16(
            &[&head.data()[..n * head.cols]],
            head.cols,
            None,
            x,
            1,
        ));
        for &t in seen {
            out[t as usize] = dot_bf16(head.row(t as usize), x);
        }
    }
}

/// Draft logits cover token ids below this plus every id already in the sequence. Qwen's BPE
/// vocabulary is roughly frequency ordered, so this keeps the likely continuations while reading
/// an eighth of the 248k-row head, which otherwise makes one draft step cost a third of a target
/// forward. Verification keeps outputs exact either way; only acceptance can drop.
// ponytail: frequency prefix + the sequence's own tokens as draft vocabulary; a learned draft
// vocab (EAGLE-3 style d2t) if acceptance on scripts past the prefix matters.
const DRAFT_VOCAB: usize = 32768;

/// [`Drafter`] over a Qwen3.5 decoder's MTP head; taps the last decoder layer.
pub(in crate::model::qwen3_5) struct MtpDrafter(pub Arc<TextModel>);

/// The head's cache and the target positions it holds (all committed), the sequence's token ids
/// at or above [`DRAFT_VOCAB`] (sorted) and how many tokens they cover, and a logits buffer.
struct MtpState {
    kv: LayerState,
    len: usize,
    seen: Vec<u32>,
    scanned: usize,
    logits: Vec<f32>,
}

impl Drafter for MtpDrafter {
    fn label(&self) -> String {
        "MTP".into()
    }

    fn tap_layers(&self) -> Vec<usize> {
        vec![self.0.layers.len() - 1]
    }

    fn new_state(&self) -> DraftState {
        Box::new(MtpState {
            kv: LayerState::Full {
                k: Vec::new(),
                v: Vec::new(),
                idx: Default::default(),
            },
            len: 0,
            seen: Vec::new(),
            scanned: 0,
            logits: Vec::new(),
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<Vec<u32>, String> {
        let t = &*self.0;
        let m = t.mtp.as_ref().ok_or("this model has no MTP head")?;
        let s = st
            .downcast_mut::<MtpState>()
            .ok_or("draft state was not built by the MTP drafter")?;
        let d = t.hidden;
        let rows = hidden.len() / d;
        if rows == 0 || hidden.len() % d != 0 || s.len + rows + 1 != tokens.len() {
            return Err(format!(
                "MTP drafter holds {} positions and got {} hidden values for a {}-token sequence",
                s.len,
                hidden.len(),
                tokens.len()
            ));
        }
        // Row i is target position s.len + i; it pairs with the token after it.
        let mut hid = hidden.to_vec();
        rms_norm_rows(&mut hid, &t.norm, t.eps);
        let mut out = t.mtp_step(m, &mut s.kv, &tokens[s.len + 1..], &hid, s.len)?;
        s.len += rows;
        let mut last = out.split_off((rows - 1) * d);
        for &id in &tokens[s.scanned..] {
            if id as usize >= DRAFT_VOCAB && (id as usize) < t.vocab {
                if let Err(i) = s.seen.binary_search(&id) {
                    s.seen.insert(i, id);
                }
            }
        }
        s.scanned = tokens.len();
        let mut drafts = Vec::with_capacity(k);
        while drafts.len() < k {
            t.draft_logits(&last, &s.seen, &mut s.logits);
            let tok = pick(&s.logits);
            drafts.push(tok);
            if drafts.len() < k {
                let p = s.len + drafts.len() - 1;
                last = t.mtp_step(m, &mut s.kv, &[tok], &last, p)?;
            }
        }
        // Draft rows leave the cache: the next call reruns them on the target's hidden states.
        t.truncate(std::slice::from_mut(&mut s.kv), s.len)?;
        Ok(drafts)
    }
}

/// [`Drafter`] running the MTP head as a program (Metal): the steps of [`MtpDrafter`] with the
/// head's logits over the whole vocabulary (no frequency-prefix draft vocabulary: the head is a
/// GPU matvec). It reads the target program's tap, which is already final-normed.
pub(in crate::model::qwen3_5) struct MtpGpuDrafter {
    text: Arc<TextModel>,
    program: Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>,
}

impl MtpGpuDrafter {
    pub(in crate::model::qwen3_5) fn new(
        text: Arc<TextModel>,
        program: Arc<parking_lot::Mutex<crate::graph::program::GraphProgram>>,
    ) -> Self {
        Self { text, program }
    }
}

/// The head's KV/state buffers and the target positions it holds (all committed).
struct MtpGpuState {
    state: crate::graph::StateStore,
    len: usize,
}

impl Drafter for MtpGpuDrafter {
    fn label(&self) -> String {
        "MTP".into()
    }

    fn tap_layers(&self) -> Vec<usize> {
        vec![self.text.layers.len() - 1]
    }

    fn new_state(&self) -> DraftState {
        Box::new(MtpGpuState {
            state: crate::graph::StateStore::new(),
            len: 0,
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<Vec<u32>, String> {
        use crate::graph::Tensor;
        let t = &*self.text;
        let s = st
            .downcast_mut::<MtpGpuState>()
            .ok_or("draft state was not built by the MTP drafter")?;
        let d = t.hidden;
        let rows = hidden.len() / d;
        if rows == 0 || hidden.len() % d != 0 || s.len + rows + 1 != tokens.len() {
            return Err(format!(
                "MTP drafter holds {} positions and got {} hidden values for a {}-token sequence",
                s.len,
                hidden.len(),
                tokens.len()
            ));
        }
        if s.len + rows + k > t.decode_max_ctx {
            return Err(format!(
                "MTP position {} exceeds the compiled context limit {}",
                s.len + rows + k,
                t.decode_max_ctx
            ));
        }
        let mut p = self.program.lock();
        std::mem::swap(&mut p.state, &mut s.state);
        let out = (|| -> Result<Vec<u32>, String> {
            let mut step = |p: &mut crate::graph::program::GraphProgram,
                            tok: u32,
                            hid: &[f32],
                            pos: usize|
             -> Result<(Vec<f32>, Vec<f32>), String> {
                let mut emb = vec![0.0f32; d];
                t.embed(tok, &mut emb)?;
                let res = p.run(vec![
                    Tensor::F32(emb),
                    Tensor::F32(hid.to_vec()),
                    Tensor::U32(vec![pos as u32]),
                ])?;
                match <[Tensor; 2]>::try_from(res) {
                    Ok([Tensor::F32(logits), Tensor::F32(normed)]) => Ok((logits, normed)),
                    _ => Err("the MTP program returned unexpected outputs".into()),
                }
            };
            // Row i is target position s.len + i; it pairs with the token after it.
            let mut last = (Vec::new(), Vec::new());
            for i in 0..rows {
                last = step(&mut p, tokens[s.len + 1 + i], &hidden[i * d..][..d], s.len + i)?;
            }
            s.len += rows;
            let (mut logits, mut normed) = last;
            let mut drafts = Vec::with_capacity(k);
            while drafts.len() < k {
                let tok = pick(&logits);
                drafts.push(tok);
                if drafts.len() < k {
                    (logits, normed) = step(&mut p, tok, &normed, s.len + drafts.len() - 1)?;
                }
            }
            // Draft rows stay in the KV buffers; the next call overwrites them with the target's.
            Ok(drafts)
        })();
        std::mem::swap(&mut p.state, &mut s.state);
        out
    }
}
