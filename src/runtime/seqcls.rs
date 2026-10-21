//! Decoder sequence classification over any `causal_lm` family spec: the backbone is a
//! [`SpecLm`]; the logits come from the spec's in-graph score head when it declares
//! `runtime.seqcls: {head}` (input `last_index` [1] u32, output `logits`), else the classifier
//! applies `score.weight` to the last row of the `hidden` head's output.

use super::lm::SpecLm;
use crate::graph::Tensor;
use crate::hardware::{Device, PlacementMode};
use crate::model::classifier::DecoderBackbone;
use crate::model::hub::ModelArtifacts;
use crate::model::tokenizer::Tokenizer;
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use crate::model::native::NativeLm;

pub struct SpecBackbone {
    lm: SpecLm,
    hidden: usize,
    /// `runtime.seqcls.head`.
    head: Option<String>,
    /// `[labels, hidden]` score head for specs without one in-graph.
    score: Vec<f32>,
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
        let lm = SpecLm::new(m, PlacementMode::Pin(dev))?;
        Ok(Self { lm, hidden, head, score: Vec::new() })
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
}
