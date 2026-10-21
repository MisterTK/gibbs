//! Decoder sequence classification over any `causal_lm` family spec: the backbone is a
//! [`SpecLm`]; the logits come from the spec's in-graph score head when it declares
//! `runtime.seqcls: {head}` (input `last_index` [1] u32, output `logits`), else the classifier
//! applies `score.weight` to the last row of the `hidden` head's output. `runtime.seqcls.pair`
//! lays out a (text, pair) prompt (see [`PairTemplate`]).

use super::lm::SpecLm;
use crate::graph::Tensor;
use crate::hardware::{Device, PlacementMode};
use crate::model::classifier::DecoderBackbone;
use crate::model::hub::ModelArtifacts;
use crate::model::tokenizer::Tokenizer;
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use crate::model::native::NativeLm;
use serde::Deserialize;

/// `runtime.seqcls.pair`: `[bos] parts..`, each part (a literal, `{text}` or `{pair}`) tokenized
/// on its own without special tokens, `strip` strings removed from both inputs (then trimmed).
/// `{pair}` keeps at most half the context less the literal parts after `{text}` (at least 16
/// tokens); `{text}` is cut from its end to fit the context. A single text is refused.
#[derive(Deserialize)]
struct PairTemplate {
    /// Expression for the BOS id (null: none).
    #[serde(default)]
    bos: Option<String>,
    parts: Vec<String>,
    #[serde(default)]
    strip: Vec<String>,
}

struct Pair {
    bos: Option<u32>,
    parts: Vec<String>,
    strip: Vec<String>,
    /// Indices of `{text}` and `{pair}` in `parts`.
    text: usize,
    pair: usize,
}

pub struct SpecBackbone {
    lm: SpecLm,
    hidden: usize,
    /// `runtime.seqcls.head`.
    head: Option<String>,
    /// `[labels, hidden]` score head for specs without one in-graph.
    score: Vec<f32>,
    pair: Option<Pair>,
}

impl SpecBackbone {
    /// `artifacts` under their detected family spec, on `dev`.
    pub fn load(artifacts: &ModelArtifacts, dev: Device) -> Result<Self, String> {
        let exact = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            ..ReencodeConfig::effective()
        };
        let m = crate::arch::load::load_with(artifacts, crate::arch::registry()?, None, exact)?;
        let hidden = m.env.var("hidden")?.usize()?;
        let head = m.spec.runtime["seqcls"]["head"].as_str().map(String::from);
        let pair = match m.spec.runtime["seqcls"].get("pair") {
            None => None,
            Some(j) => {
                let fam = &m.spec.family;
                let t: PairTemplate = serde_json::from_value(j.clone())
                    .map_err(|e| format!("{fam}: runtime.seqcls.pair: {e}"))?;
                let at = |k: &str| {
                    t.parts.iter().position(|p| p == k)
                        .ok_or_else(|| format!("{fam}: runtime.seqcls.pair has no `{k}` part"))
                };
                let (text, pair) = (at("{text}")?, at("{pair}")?);
                let bos = match &t.bos {
                    Some(e) => m.env.eval(&serde_json::Value::String(e.clone()))?.to_json().as_u64().map(|v| v as u32),
                    None => None,
                };
                Some(Pair { bos, parts: t.parts, strip: t.strip, text, pair })
            }
        };
        let lm = SpecLm::new(m, PlacementMode::Pin(dev))?;
        Ok(Self { lm, hidden, head, score: Vec::new(), pair })
    }
}

impl DecoderBackbone for SpecBackbone {
    fn tokenizer(&self) -> &Tokenizer {
        self.lm.tokenizer()
    }

    fn context_len(&self) -> usize {
        self.lm.context_len()
    }

    fn hidden(&self) -> usize {
        self.hidden
    }

    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.lm.run_head(ids, "hidden", "hidden", &[])?;
        Ok(h[h.len() - self.hidden..].to_vec())
    }

    fn attach_score_gpu(&mut self, score: &[f32], labels: usize) -> Result<(), String> {
        if score.len() != labels * self.hidden {
            return Err(format!("score head is not [{labels}, {}]", self.hidden));
        }
        self.score = score.to_vec();
        Ok(())
    }

    fn score_gpu(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let Some(head) = &self.head else {
            let h = self.last_hidden(ids)?;
            return Ok(self.score.chunks(self.hidden).map(|r| r.iter().zip(&h).map(|(a, b)| a * b).sum()).collect());
        };
        let last = (ids.len() - self.lm.last_pass_start(ids.len()) - 1) as u32;
        self.lm.run_head(ids, head, "logits", &[("last_index", Tensor::U32(vec![last]))])
    }

    fn encode(&self, text: &str, pair: Option<&str>) -> Result<Vec<u32>, String> {
        let tok = self.tokenizer();
        let Some(t) = &self.pair else {
            return match pair {
                None => Ok(tok.encode(text)),
                Some(answer) => crate::model::classifier::encode_pair(tok, text, answer),
            };
        };
        let pair = pair.ok_or("this cross-encoder scores (text || pair) pairs")?;
        let hf = tok.hf.as_ref().ok_or("tokenizer.json is required")?;
        let enc = |s: &str| -> Result<Vec<u32>, String> {
            hf.encode_fast(s, false).map(|e| e.get_ids().to_vec()).map_err(|e| format!("tokenize: {e}"))
        };
        let clean = |s: &str| t.strip.iter().fold(s.to_string(), |s, x| s.replace(x.as_str(), ""));
        let mut pieces = t
            .parts
            .iter()
            .enumerate()
            .map(|(i, p)| match i {
                _ if i == t.text => enc(clean(text).trim()),
                _ if i == t.pair => enc(clean(pair).trim()),
                _ => enc(p),
            })
            .collect::<Result<Vec<_>, _>>()?;
        fit(&mut pieces, t.text, t.pair, t.bos.is_some(), self.context_len());
        Ok(t.bos.into_iter().chain(pieces.into_iter().flatten()).collect())
    }
}

/// Cuts the tokenized template `pieces` to `ctx` with a `bos`: `pieces[pair]` to half the context
/// less the literal pieces after `pieces[text]` (at least 16), then `pieces[text]` to the rest.
fn fit(pieces: &mut [Vec<u32>], text: usize, pair: usize, bos: bool, ctx: usize) {
    let after: usize = (text + 1..pieces.len()).filter(|&i| i != pair).map(|i| pieces[i].len()).sum();
    pieces[pair].truncate((ctx / 2).saturating_sub(after).max(16));
    let rest = bos as usize + pieces.iter().enumerate().filter(|&(i, _)| i != text).map(|(_, p)| p.len()).sum::<usize>();
    pieces[text].truncate(ctx.saturating_sub(rest));
}

#[cfg(test)]
mod tests {
    /// gevva `tokenize_nli_pair_safe` at max_length 4096: the hypothesis keeps 2048 less the
    /// hypothesis prefix and suffix, the premise the rest; a short context keeps 16 hypothesis tokens.
    #[test]
    fn fit_matches_gevva_truncation() {
        let parts = |p: usize, h: usize| vec![vec![1; 2], vec![2; p], vec![3; 3], vec![4; h], vec![5; 2]];
        let mut v = parts(5000, 3000);
        super::fit(&mut v, 1, 3, true, 4096);
        assert_eq!(v.iter().map(Vec::len).collect::<Vec<_>>(), [2, 4096 - (1 + 2 + 3 + 2043 + 2), 3, 2048 - 5, 2]);
        let mut v = parts(50, 50);
        super::fit(&mut v, 1, 3, false, 40);
        assert_eq!(v.iter().map(Vec::len).collect::<Vec<_>>(), [2, 40 - 23, 3, 16, 2]);
        // Short inputs are kept whole.
        let mut v = parts(7, 9);
        super::fit(&mut v, 1, 3, true, 4096);
        assert_eq!(v, parts(7, 9));
    }
}
