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
            text,
        })
    }
}

impl DecoderBackbone for Gemma4Seq {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn context_len(&self) -> usize {
        self.text.max_pos
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
