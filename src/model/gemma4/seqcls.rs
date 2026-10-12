//! `Gemma4ForSequenceClassification` as the gevva cross-encoders define it (transformers has no
//! such class; github.com/davidburhans/gevva `gemma4_cross_encoder.py`): the Gemma 4 text
//! decoder's final-norm hidden state of the last token, normalized once more by the head's own
//! RMSNorm (`norm.weight`), then `score`. A (premise, hypothesis) pair is laid out as gevva's
//! `tokenize_nli_pair_safe` does.

use super::ops::rms_norm;
use super::text::TextModel;
use crate::model::classifier::DecoderBackbone;
use crate::model::hub::ModelArtifacts;
use crate::model::tokenizer::Tokenizer;
use crate::model::weights::Weights;
use serde_json::Value;

/// gevva `Gemma4CrossEncoder` default `max_length`.
const MAX_LEN: usize = 4096;

pub struct Gemma4Seq {
    tokenizer: Tokenizer,
    text: TextModel,
    /// The head's RMSNorm (Gemma 4 style: `x * w`).
    norm: Vec<f32>,
    eps: f32,
    bos: Option<u32>,
    /// The Metal program once `attach_score_gpu` ran.
    program: Option<parking_lot::Mutex<crate::graph::program::GraphProgram>>,
}

impl Gemma4Seq {
    pub fn load(artifacts: &ModelArtifacts, cfg: &Value) -> Result<Self, String> {
        let weights = artifacts.weights_path.as_deref().ok_or("no weights file")?;
        let tok = artifacts
            .tokenizer_path
            .as_deref()
            .ok_or("no tokenizer.json")?;
        let tokenizer = crate::model::hf_loader::load_json_tokenizer(tok)?;
        let w = Weights::open(weights)?;
        let c = &cfg["text_config"];
        let text = TextModel::load(&w, c).map_err(|e| format!("gemma4 text: {e}"))?;
        let norm = w.vec_n("norm.weight", text.hidden)?;
        Ok(Self {
            tokenizer,
            norm,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            bos: c["bos_token_id"].as_u64().map(|v| v as u32),
            program: None,
            text,
        })
    }
}

impl Gemma4Seq {
    /// Positions a request may hold: `encode` caps inputs at [`MAX_LEN`], which also sizes the
    /// Metal program's KV state (max_position_embeddings is 128k+).
    fn ctx(&self) -> usize {
        self.text.max_pos.min(MAX_LEN)
    }
}

impl DecoderBackbone for Gemma4Seq {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn context_len(&self) -> usize {
        self.ctx()
    }
    fn hidden(&self) -> usize {
        self.text.hidden
    }
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.text.hidden;
        let mut x = vec![0.0f32; ids.len() * h];
        for (row, &id) in x.chunks_exact_mut(h).zip(ids) {
            if id as usize >= self.text.vocab {
                return Err(format!("token id {id} is outside the vocabulary"));
            }
            self.text.embed_row(id, row);
        }
        let mut v = self
            .text
            .last_hidden(x, ids, 0, &mut self.text.new_cache(), &[]);
        rms_norm(&mut v, h, Some(&self.norm), self.eps);
        Ok(v)
    }
    /// Compiles the decode graph (one token per step, KV state) with the cross-encoder head for
    /// Metal, as `Gemma4` compiles its generation program.
    fn attach_score_gpu(&mut self, score: &[f32], labels: usize) -> Result<(), String> {
        if score.len() != labels * self.text.hidden {
            return Err(format!("score head is not [{labels}, {}]", self.text.hidden));
        }
        let gpu_only = |e: String| format!("gpu-only: Gemma 4 seq-cls: {e}");
        let (graph, weights) = self
            .text
            .to_seqcls_graph(self.ctx(), &self.norm, self.eps, score)
            .map_err(gpu_only)?;
        let mode = crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal);
        let program = crate::graph::program::GraphProgram::compile(graph, weights, mode)
            .map_err(gpu_only)?;
        self.program = Some(parking_lot::Mutex::new(program));
        Ok(())
    }

    // ponytail: n sequential decode steps over a fresh KV state; a prefill graph would cut launches.
    fn score_gpu(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        use crate::graph::Tensor;
        let prog = self.program.as_ref().ok_or("no GPU score program attached")?;
        let t = &self.text;
        if ids.is_empty() || ids.len() > self.ctx() {
            return Err(format!("{} tokens: the context holds 1..={}", ids.len(), self.ctx()));
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= t.vocab) {
            return Err(format!("token id {id} is outside the vocabulary"));
        }
        let mut p = prog.lock();
        let saved = std::mem::replace(&mut p.state, crate::graph::StateStore::new());
        let run = |p: &mut crate::graph::program::GraphProgram| -> Result<Vec<f32>, String> {
            let mut logits = Vec::new();
            for (pos, &id) in ids.iter().enumerate() {
                let mut emb = vec![0.0f32; t.hidden];
                t.embed_row(id, &mut emb);
                let mut inputs = vec![Tensor::F32(emb), Tensor::U32(vec![pos as u32])];
                if let Some(ple) = &t.ple {
                    // Per-layer embedding of the token, scaled as `per_layer_inputs` does.
                    let mut e = vec![0.0f32; t.layers_n * ple.dim];
                    let row = if (id as usize) < ple.embed.rows { id } else { t.pad_id };
                    ple.embed.get_row(row as usize, &mut e);
                    let scale = (ple.dim as f32).sqrt();
                    e.iter_mut().for_each(|v| *v *= scale);
                    inputs.push(Tensor::F32(e));
                }
                logits = p
                    .run(inputs)?
                    .into_iter()
                    .next()
                    .ok_or("seq-cls program produced no output")?
                    .into_f32()?;
            }
            Ok(logits)
        };
        let logits = run(&mut p);
        p.state = saved;
        logits
    }
    /// `<bos>Premise: {premise}\nHypothesis: {hypothesis}\nPrediction:`, each piece tokenized on
    /// its own; the hypothesis keeps at most half of [`MAX_LEN`], the premise is cut from its end.
    fn encode(&self, text: &str, pair: Option<&str>) -> Result<Vec<u32>, String> {
        let hyp = pair.ok_or("this NLI cross-encoder scores (premise || hypothesis) pairs")?;
        let hf = self
            .tokenizer
            .hf
            .as_ref()
            .ok_or("tokenizer.json is required")?;
        let enc = |s: &str| -> Result<Vec<u32>, String> {
            hf.encode_fast(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| format!("tokenize: {e}"))
        };
        // Media placeholders are stripped from user text (no images here).
        let clean = |s: &str| {
            [
                "<|image|>",
                "<image|>",
                "<|image_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            ]
            .iter()
            .fold(s.to_string(), |s, t| s.replace(t, ""))
        };
        let suffix = enc("\nPrediction:")?;
        let hyp_prefix = enc("\nHypothesis: ")?;
        let prem_prefix = enc("Premise: ")?;
        let mut hyp_body = enc(clean(hyp).trim())?;
        hyp_body.truncate(
            (MAX_LEN / 2)
                .saturating_sub(suffix.len() + hyp_prefix.len())
                .max(16),
        );
        let mut prem = enc(clean(text).trim())?;
        let fixed = self.bos.iter().len()
            + prem_prefix.len()
            + hyp_prefix.len()
            + hyp_body.len()
            + suffix.len();
        prem.truncate(MAX_LEN.saturating_sub(fixed));
        Ok(self
            .bos
            .into_iter()
            .chain(prem_prefix)
            .chain(prem)
            .chain(hyp_prefix)
            .chain(hyp_body)
            .chain(suffix)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::text::{Attn, Layer, Ple, PleTable};
    use super::*;
    use crate::model::classifier::decoder_gpu::{fill, tests::parity};

    /// Three layers (sliding window 3 with GQA 4/2 heads, global, then a sliding layer sharing
    /// layer 0's KV), per-layer embeddings, layer scalars: the cross-encoder head on Metal vs the host forward.
    #[test]
    fn seqcls_gpu_matches_host() {
        let (h, v, nh, kvh, hd, inter, pd) = (64, 96, 4, 2, 16, 96, 8);
        let mut specs: Vec<(String, Vec<usize>)> = vec![
            ("embed".into(), vec![v, h]),
            ("ple_embed".into(), vec![v, 3 * pd]),
            ("ple_proj".into(), vec![3 * pd, h]),
        ];
        for l in 0..3 {
            for (n, r, c) in [
                ("q", nh * hd, h),
                ("k", kvh * hd, h),
                ("v", kvh * hd, h),
                ("o", h, nh * hd),
                ("gate", inter, h),
                ("up", inter, h),
                ("down", h, inter),
                ("ple_gate", pd, h),
                ("ple_out", h, pd),
            ] {
                specs.push((format!("{n}{l}"), vec![r, c]));
            }
        }
        let weights = Weights::from_f32(specs.clone(), |name| {
            let len = specs.iter().find(|(n, _)| n == name).unwrap().1.iter().product();
            let seed = name.bytes().map(|b| b as usize).sum();
            Ok(fill(len, seed, 0.3))
        })
        .unwrap();
        let m = |n: &str| weights.mat(n).unwrap();
        let gain = |seed| -> Vec<f32> { fill(h, seed, 0.3).iter().map(|v| v + 1.0).collect() };
        let layers = (0..3)
            .map(|l| Layer {
                // Layer 2 shares layer 0's KV (both sliding), as E2B/E4B's trailing layers do.
                sliding: l != 1,
                head_dim: hd,
                kv_heads: kvh,
                kv_src: if l == 2 { 0 } else { l },
                attn: Attn {
                    q: m(&format!("q{l}")),
                    kv: (l < 2).then(|| (m(&format!("k{l}")), Some(m(&format!("v{l}"))), vec![1.0; hd])),
                    o: m(&format!("o{l}")),
                    q_norm: fill(hd, 3 + l, 0.3).iter().map(|v| v + 1.0).collect(),
                },
                gate: m(&format!("gate{l}")),
                up: m(&format!("up{l}")),
                down: m(&format!("down{l}")),
                input_ln: gain(10 + l),
                post_attn_ln: gain(20 + l),
                pre_ff_ln: gain(30 + l),
                post_ff_ln: gain(40 + l),
                ple: Some((m(&format!("ple_gate{l}")), m(&format!("ple_out{l}")), gain(50 + l))),
                scalar: 0.8 + 0.1 * l as f32,
            })
            .collect();
        let text = TextModel {
            hidden: h,
            vocab: v,
            layers_n: 3,
            max_pos: 64,
            heads: nh,
            eps: 1e-6,
            window: 3,
            softcap: Some(30.0),
            embed_scale: 8.0,
            embed: m("embed"),
            ple: Some(Ple {
                dim: pd,
                embed: PleTable::mat(m("ple_embed")),
                model_proj: m("ple_proj"),
                proj_norm: fill(pd, 7, 0.3).iter().map(|v| v + 1.0).collect(),
            }),
            layers,
            first_shared: 2,
            norm: gain(60),
            inv_freq_sliding: (0..hd / 2).map(|i| 1.0 / 1.7f32.powi(i as i32)).collect(),
            inv_freq_full: (0..hd / 2).map(|i| 0.5 / 1.4f32.powi(i as i32)).collect(),
            pad_id: 0,
            bidir_vision: false,
        };
        let mut llm = Gemma4Seq {
            tokenizer: Tokenizer::default_bpe(),
            text,
            norm: gain(61),
            eps: 1e-6,
            bos: None,
            program: None,
        };
        let seqs: [&[u32]; 3] = [&[3, 17, 5], &[9, 8, 7, 6, 5, 4, 3, 2, 1, 70], &[42]];
        let e = parity(&mut llm, &fill(3 * h, 62, 0.5), 3, &seqs);
        eprintln!("gemma4 seq-cls max rel err {e:?}");
    }
}
