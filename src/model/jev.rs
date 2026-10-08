//! "System 1" decision engine: a real pretrained BERT sentence encoder
//! (default sentence-transformers/all-MiniLM-L6-v2) answering structured questions
//! zero-shot by cosine similarity between embeddings, in a single forward pass.

use crate::graph::{DType, Graph, Op, Tensor, Weights};
use crate::hardware::Device;
use crate::model::hub::HuggingFaceHub;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An option in a categorical choice question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub label: String,
    pub description: Option<String>,
}

impl ChoiceOption {
    pub fn new(label: impl Into<String>, description: Option<impl Into<String>>) -> Self {
        Self {
            label: label.into(),
            description: description.map(|d| d.into()),
        }
    }

    /// Text embedded for this option: "label: description", or the bare label.
    fn text(&self) -> String {
        match &self.description {
            Some(d) if !d.trim().is_empty() => format!("{}: {}", self.label, d),
            _ => self.label.clone(),
        }
    }
}

/// Structured questions accepted by the System 1 engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JevQuestion {
    Bool {
        id: String,
        instructions: String,
        criteria_true: Option<String>,
        criteria_false: Option<String>,
    },
    Choice {
        id: String,
        instructions: String,
        options: Vec<ChoiceOption>,
    },
    Score {
        id: String,
        instructions: String,
        criteria: Vec<String>,
    },
}

impl JevQuestion {
    pub fn id(&self) -> &str {
        match self {
            Self::Bool { id, .. } | Self::Choice { id, .. } | Self::Score { id, .. } => id,
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Bool { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }
}

/// Decisions produced by the zero-shot encoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JevDecision {
    /// P(true) and the resulting decision.
    Bool { bool_prob: f32, decision: bool },
    /// Top option with the full distribution over option labels.
    Choice {
        selected: String,
        probabilities: HashMap<String, f32>,
        confidence: f32,
    },
}

const BOOL_CRITERIA_ERR: &str =
    "bool needs criteria_true and criteria_false for the zero-shot encoder";
const SCORE_ERR: &str =
    "score questions need a trained head; not supported by the zero-shot encoder";

#[derive(Deserialize)]
struct BertConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    #[serde(default = "default_type_vocab")]
    type_vocab_size: usize,
    #[serde(default = "default_ln_eps")]
    layer_norm_eps: f32,
}

fn default_type_vocab() -> usize {
    2
}
fn default_ln_eps() -> f32 {
    1e-12
}

/// Dimensions of a post-LN BERT encoder (BERT/MiniLM, and RoBERTa/XLM-R/DistilBERT once their
/// tensors are mapped to BERT names).
#[derive(Debug, Clone, Copy)]
pub struct BertDims {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_positions: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

/// A post-LN BERT encoder graph on the CPU, Metal, or Core ML (ANE). System 1 mean-pools it into
/// sentence embeddings; the sequence classifiers read its last hidden states.
pub struct BertEncoder {
    pub dims: BertDims,
    /// Every encoder tensor under its BERT name (`embeddings.*`, `encoder.layer.N.*`).
    weights: Weights,
    executor: Executor,
}

/// Where the encoder graph runs.
enum Executor {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal(crate::graph::metal::Session),
    #[cfg(target_os = "macos")]
    CoreMl(std::sync::Mutex<crate::graph::coreml::Session>),
}

impl BertEncoder {
    /// Builds the encoder on `device` (CPU, or Metal/ANE on macOS; anything else is an error).
    /// `tensor(name, shape)` returns each tensor under its BERT name.
    pub fn load(
        dims: BertDims,
        device: Device,
        mut tensor: impl FnMut(&str, &[usize]) -> Result<Vec<f32>, String>,
    ) -> Result<Self, String> {
        if dims.num_heads == 0 || dims.hidden_size % dims.num_heads != 0 {
            return Err(format!(
                "config.json: hidden_size {} not divisible by num_attention_heads {}",
                dims.hidden_size, dims.num_heads
            ));
        }
        let (h, inter) = (dims.hidden_size, dims.intermediate_size);
        let mut shapes = vec![
            (
                "embeddings.word_embeddings.weight".to_string(),
                vec![dims.vocab_size, h],
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                vec![dims.max_positions, h],
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                vec![dims.type_vocab_size, h],
            ),
            ("embeddings.LayerNorm.weight".to_string(), vec![h]),
            ("embeddings.LayerNorm.bias".to_string(), vec![h]),
        ];
        for l in 0..dims.num_layers {
            let p = format!("encoder.layer.{l}");
            for (s, shape) in [
                ("attention.self.query.weight", vec![h, h]),
                ("attention.self.query.bias", vec![h]),
                ("attention.self.key.weight", vec![h, h]),
                ("attention.self.key.bias", vec![h]),
                ("attention.self.value.weight", vec![h, h]),
                ("attention.self.value.bias", vec![h]),
                ("attention.output.dense.weight", vec![h, h]),
                ("attention.output.dense.bias", vec![h]),
                ("attention.output.LayerNorm.weight", vec![h]),
                ("attention.output.LayerNorm.bias", vec![h]),
                ("intermediate.dense.weight", vec![inter, h]),
                ("intermediate.dense.bias", vec![inter]),
                ("output.dense.weight", vec![h, inter]),
                ("output.dense.bias", vec![h]),
                ("output.LayerNorm.weight", vec![h]),
                ("output.LayerNorm.bias", vec![h]),
            ] {
                shapes.push((format!("{p}.{s}"), shape));
            }
        }
        let mut weights = Weights::new();
        for (name, shape) in shapes {
            let v = tensor(&name, &shape)?;
            if v.len() != shape.iter().product::<usize>() {
                return Err(format!(
                    "tensor {name} has {} values, expected shape {shape:?}",
                    v.len()
                ));
            }
            weights.insert(name, Tensor::F32(v));
        }

        let executor = match device {
            Device::Cpu => Executor::Cpu,
            #[cfg(target_os = "macos")]
            Device::Metal => {
                let rt = crate::graph::metal::Runtime::new(&crate::hardware::default_cache_dir()?)?;
                Executor::Metal(crate::graph::metal::Session::new(rt, &weights)?)
            }
            #[cfg(target_os = "macos")]
            Device::Ane => {
                let cache_dir = crate::hardware::default_cache_dir()?;
                let session = crate::graph::coreml::Session::new(
                    &cache_dir,
                    &weights,
                    h,
                    dims.num_layers,
                    dims.num_heads,
                )?;
                Executor::CoreMl(std::sync::Mutex::new(session))
            }
            d => return Err(format!("the BERT encoder has no {d} executor")),
        };
        Ok(Self {
            dims,
            weights,
            executor,
        })
    }

    /// The device the encoder graph runs on.
    pub fn device(&self) -> Device {
        match self.executor {
            Executor::Cpu => Device::Cpu,
            #[cfg(target_os = "macos")]
            Executor::Metal(_) => Device::Metal,
            #[cfg(target_os = "macos")]
            Executor::CoreMl(_) => Device::Ane,
        }
    }

    /// Load summary for hardware runtimes (Metal, Core ML ANE); `None` on the CPU.
    pub fn executor_report(&self) -> Option<String> {
        match &self.executor {
            Executor::Cpu => None,
            #[cfg(target_os = "macos")]
            Executor::Metal(s) => Some(s.runtime().report.clone()),
            #[cfg(target_os = "macos")]
            Executor::CoreMl(s) => Some(s.lock().ok()?.report.clone()),
        }
    }

    /// Total weight count of the loaded encoder.
    pub fn parameter_count(&self) -> usize {
        self.weights.parameter_count()
    }

    /// The encoder graph for `n` tokens: embeddings, BERT layers, then mean pooling and L2
    /// normalisation when `pooled` (else the last hidden states `[n, h]`).
    fn graph(&self, n: usize, pooled: bool) -> Graph {
        let d = &self.dims;
        let (h, inter, eps) = (d.hidden_size, d.intermediate_size, d.layer_norm_eps);
        let mut g = Graph::new();
        let ids = g.input("input_ids", vec![n], DType::U32);
        let pos = g.input("position_ids", vec![n], DType::U32);
        let types = g.input("token_type_ids", vec![n], DType::U32);
        let mask = g.input("attention_mask", vec![n], DType::U32);
        let w = |g: &mut Graph, name: String, shape: Vec<usize>| g.weight(name, shape, DType::F32);
        let f = DType::F32;

        let wt = w(
            &mut g,
            "embeddings.word_embeddings.weight".into(),
            vec![d.vocab_size, h],
        );
        let pt = w(
            &mut g,
            "embeddings.position_embeddings.weight".into(),
            vec![d.max_positions, h],
        );
        let tt = w(
            &mut g,
            "embeddings.token_type_embeddings.weight".into(),
            vec![d.type_vocab_size, h],
        );
        let we = g.add("emb.word", Op::Embedding, &[wt, ids], vec![n, h], f);
        let pe = g.add("emb.position", Op::Embedding, &[pt, pos], vec![n, h], f);
        let te = g.add("emb.type", Op::Embedding, &[tt, types], vec![n, h], f);
        let wp = g.add("emb.add_position", Op::Add, &[we, pe], vec![n, h], f);
        let e = g.add("emb.add_type", Op::Add, &[wp, te], vec![n, h], f);
        let lg = w(&mut g, "embeddings.LayerNorm.weight".into(), vec![h]);
        let lb = w(&mut g, "embeddings.LayerNorm.bias".into(), vec![h]);
        let mut x = g.add("emb.ln", Op::LayerNorm { eps }, &[e, lg, lb], vec![n, h], f);

        for l in 0..d.num_layers {
            let p = format!("encoder.layer.{l}");
            let lin = |g: &mut Graph, x, s: &str, in_f: usize, out_f: usize| {
                let wv = w(g, format!("{p}.{s}.weight"), vec![out_f, in_f]);
                let bv = w(g, format!("{p}.{s}.bias"), vec![out_f]);
                g.add(
                    format!("{p}.{s}"),
                    Op::Linear,
                    &[x, wv, bv],
                    vec![n, out_f],
                    f,
                )
            };
            let q = lin(&mut g, x, "attention.self.query", h, h);
            let k = lin(&mut g, x, "attention.self.key", h, h);
            let v = lin(&mut g, x, "attention.self.value", h, h);
            let heads = d.num_heads;
            let attn = Op::Attention {
                heads,
                kv_heads: heads,
                causal: false,
                window: None,
                soft_cap: None,
            };
            let a = g.add(format!("{p}.attention"), attn, &[q, k, v], vec![n, h], f);
            let o = lin(&mut g, a, "attention.output.dense", h, h);
            let r = g.add(
                format!("{p}.attention.residual"),
                Op::Add,
                &[o, x],
                vec![n, h],
                f,
            );
            let lg = w(
                &mut g,
                format!("{p}.attention.output.LayerNorm.weight"),
                vec![h],
            );
            let lb = w(
                &mut g,
                format!("{p}.attention.output.LayerNorm.bias"),
                vec![h],
            );
            x = g.add(
                format!("{p}.attention.output.LayerNorm"),
                Op::LayerNorm { eps },
                &[r, lg, lb],
                vec![n, h],
                f,
            );

            let i = lin(&mut g, x, "intermediate.dense", h, inter);
            let i = g.add(
                format!("{p}.intermediate.gelu"),
                Op::Gelu { tanh: false },
                &[i],
                vec![n, inter],
                f,
            );
            let o2 = lin(&mut g, i, "output.dense", inter, h);
            let r2 = g.add(
                format!("{p}.output.residual"),
                Op::Add,
                &[o2, x],
                vec![n, h],
                f,
            );
            let lg = w(&mut g, format!("{p}.output.LayerNorm.weight"), vec![h]);
            let lb = w(&mut g, format!("{p}.output.LayerNorm.bias"), vec![h]);
            x = g.add(
                format!("{p}.output.LayerNorm"),
                Op::LayerNorm { eps },
                &[r2, lg, lb],
                vec![n, h],
                f,
            );
        }

        if pooled {
            let pooled = g.add("mean_pool", Op::MeanPool, &[x, mask], vec![h], f);
            x = g.add("l2_normalize", Op::L2Normalize, &[pooled], vec![h], f);
        }
        g.output(x);
        g
    }

    /// Runs one unpadded sequence: the mean-pooled, L2-normalised embedding `[h]` when `pooled`,
    /// else the last hidden states `[n, h]`.
    pub fn run(
        &self,
        ids: &[u32],
        positions: &[u32],
        types: &[u32],
        pooled: bool,
    ) -> Result<Vec<f32>, String> {
        let n = ids.len();
        if n == 0 {
            return Err("tokenizer produced no tokens".into());
        }
        if n > self.dims.max_positions {
            return Err(format!(
                "{n} tokens exceed {} positions",
                self.dims.max_positions
            ));
        }
        let inputs = vec![
            Tensor::U32(ids.to_vec()),
            Tensor::U32(positions.to_vec()),
            Tensor::U32(types.to_vec()),
            Tensor::U32(vec![1; n]),
        ];
        let graph = self.graph(n, pooled);
        match &self.executor {
            Executor::Cpu => {
                let mut out = crate::graph::cpu::run_stateless(&graph, &self.weights, inputs)?;
                out.pop()
                    .ok_or_else(|| "encoder graph produced no output".to_string())?
                    .into_f32()
            }
            #[cfg(target_os = "macos")]
            Executor::Metal(s) => {
                let mut out = s.run(&graph, inputs)?;
                out.pop()
                    .ok_or_else(|| "encoder graph produced no output".to_string())?
                    .into_f32()
            }
            #[cfg(target_os = "macos")]
            Executor::CoreMl(s) => {
                let emb = Self::initial_embeddings(
                    &self.weights,
                    ids,
                    positions,
                    types,
                    self.dims.layer_norm_eps,
                    self.dims.hidden_size,
                )?;
                let mut session = s.lock().map_err(|e| e.to_string())?;
                let out = session.run_encoder(&emb, n)?;
                if !pooled {
                    return Ok(out);
                }
                Ok(Self::mean_pool_and_normalize(
                    &out,
                    &vec![1.0; n],
                    self.dims.hidden_size,
                ))
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn initial_embeddings(
        weights: &Weights,
        ids: &[u32],
        pos: &[u32],
        types: &[u32],
        eps: f32,
        h: usize,
    ) -> Result<Vec<f32>, String> {
        let n = ids.len();
        let w_emb = weights
            .get("embeddings.word_embeddings.weight")
            .ok_or("missing word embeddings")?
            .as_f32()?;
        let p_emb = weights
            .get("embeddings.position_embeddings.weight")
            .ok_or("missing position embeddings")?
            .as_f32()?;
        let t_emb = weights
            .get("embeddings.token_type_embeddings.weight")
            .ok_or("missing token type embeddings")?
            .as_f32()?;
        let ln_w = weights
            .get("embeddings.LayerNorm.weight")
            .ok_or("missing embeddings.LayerNorm.weight")?
            .as_f32()?;
        let ln_b = weights
            .get("embeddings.LayerNorm.bias")
            .ok_or("missing embeddings.LayerNorm.bias")?
            .as_f32()?;

        let mut x = vec![0.0f32; n * h];
        for i in 0..n {
            let wid = ids[i] as usize;
            let pid = pos[i] as usize;
            let tid = types[i] as usize;
            for c in 0..h {
                x[i * h + c] = w_emb[wid * h + c] + p_emb[pid * h + c] + t_emb[tid * h + c];
            }
            let token = &mut x[i * h..(i + 1) * h];
            let mean = token.iter().sum::<f32>() / h as f32;
            let var = token.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..h {
                token[c] = (token[c] - mean) * inv * ln_w[c] + ln_b[c];
            }
        }
        Ok(x)
    }

    #[cfg(target_os = "macos")]
    fn mean_pool_and_normalize(out: &[f32], mask: &[f32], h: usize) -> Vec<f32> {
        let n = mask.len();
        let mut sum = vec![0.0f32; h];
        let mut count = 0.0f32;
        for i in 0..n {
            if mask[i] > 0.0 {
                count += 1.0;
                for c in 0..h {
                    sum[c] += out[i * h + c];
                }
            }
        }
        let inv_count = if count > 0.0 { 1.0 / count } else { 1.0 };
        for c in 0..h {
            sum[c] *= inv_count;
        }
        let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 1.0 };
        for c in 0..h {
            sum[c] *= inv_norm;
        }
        sum
    }
}

/// Pretrained BERT sentence encoder used zero-shot for System 1 decisions.
pub struct JevSystem1Model {
    pub repo_id: String,
    /// Softmax temperature over cosine similarities.
    // ponytail: zero-shot similarity, not a trained head; calibrate T or train heads if accuracy matters.
    pub temperature: f32,
    tokenizer: tokenizers::Tokenizer,
    pub encoder: BertEncoder,
}

impl JevSystem1Model {
    /// Fetches config.json, tokenizer.json and model.safetensors from `repo_id` and builds the
    /// encoder on `device` (CPU, or Metal/ANE on macOS; anything else is an error).
    pub fn load(hub: &HuggingFaceHub, repo_id: &str, device: Device) -> Result<Self, String> {
        let config_path = hub.fetch_file(repo_id, "config.json")?;
        let tok_path = hub.fetch_file(repo_id, "tokenizer.json")?;
        let st_path = hub.fetch_file(repo_id, "model.safetensors")?;

        let cfg: BertConfig = serde_json::from_slice(
            &std::fs::read(&config_path)
                .map_err(|e| format!("read {}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("parse {}: {e}", config_path.display()))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("load {}: {e}", tok_path.display()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: cfg.max_position_embeddings,
                ..Default::default()
            }))
            .map_err(|e| format!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);

        let data =
            std::fs::read(&st_path).map_err(|e| format!("read {}: {e}", st_path.display()))?;
        let manifest = crate::model::UniversalModelLoader::parse_safetensors(&data)?;
        let t = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            let prefixed = format!("bert.{name}");
            let key = if manifest.tensors.contains_key(name) {
                name
            } else if manifest.tensors.contains_key(&prefixed) {
                prefixed.as_str()
            } else {
                return Err(format!("{repo_id}: missing tensor {name}"));
            };
            let got = &manifest.tensors[key].shape;
            if got.as_slice() != shape {
                return Err(format!(
                    "{repo_id}: tensor {key} has shape {got:?}, expected {shape:?}"
                ));
            }
            manifest.extract_tensor_f32(&data, key)
        };
        let dims = BertDims {
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_hidden_layers,
            num_heads: cfg.num_attention_heads,
            intermediate_size: cfg.intermediate_size,
            max_positions: cfg.max_position_embeddings,
            vocab_size: cfg.vocab_size,
            type_vocab_size: cfg.type_vocab_size,
            layer_norm_eps: cfg.layer_norm_eps,
        };
        Ok(Self {
            repo_id: repo_id.to_string(),
            temperature: 0.05,
            tokenizer,
            encoder: BertEncoder::load(dims, device, t)?,
        })
    }

    /// Number of WordPiece tokens the encoder sees for `text` (incl. [CLS]/[SEP], after truncation).
    pub fn token_count(&self, text: &str) -> Result<usize, String> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        Ok(enc.get_ids().len())
    }

    /// Sentence embedding: BERT forward, mean pooling over tokens, L2 normalisation.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let positions: Vec<u32> = (0..enc.get_ids().len() as u32).collect();
        self.encoder
            .run(enc.get_ids(), &positions, enc.get_type_ids(), true)
    }

    /// Zero-shot choice: softmax(cos(state, option) / T).
    pub fn evaluate_choice_embedded(
        &self,
        state: &[f32],
        options: &[ChoiceOption],
    ) -> Result<JevDecision, String> {
        if options.is_empty() {
            return Err("choice question has no options".into());
        }
        let cos = options
            .iter()
            .map(|o| Ok(dot(state, &self.embed(&o.text())?)))
            .collect::<Result<Vec<f32>, String>>()?;
        let probs = softmax_scaled(&cos, self.temperature);
        let best = (0..probs.len()).fold(0, |b, i| if probs[i] > probs[b] { i } else { b });
        Ok(JevDecision::Choice {
            selected: options[best].label.clone(),
            probabilities: options
                .iter()
                .map(|o| o.label.clone())
                .zip(probs.iter().copied())
                .collect(),
            confidence: probs[best],
        })
    }

    /// Zero-shot bool: softmax over [cos(state, true), cos(state, false)] / T.
    pub fn evaluate_bool_embedded(
        &self,
        state: &[f32],
        criteria_true: Option<&str>,
        criteria_false: Option<&str>,
    ) -> Result<JevDecision, String> {
        let (Some(t), Some(f)) = (
            criteria_true.filter(|s| !s.trim().is_empty()),
            criteria_false.filter(|s| !s.trim().is_empty()),
        ) else {
            return Err(BOOL_CRITERIA_ERR.into());
        };
        let cos = [dot(state, &self.embed(t)?), dot(state, &self.embed(f)?)];
        let p = softmax_scaled(&cos, self.temperature)[0];
        Ok(JevDecision::Bool {
            bool_prob: p,
            decision: p >= 0.5,
        })
    }

    /// Answers one question against a precomputed state embedding.
    pub fn evaluate_embedded(
        &self,
        state: &[f32],
        question: &JevQuestion,
    ) -> Result<JevDecision, String> {
        check_supported(question)?;
        match question {
            JevQuestion::Bool {
                criteria_true,
                criteria_false,
                ..
            } => self.evaluate_bool_embedded(
                state,
                criteria_true.as_deref(),
                criteria_false.as_deref(),
            ),
            JevQuestion::Choice { options, .. } => self.evaluate_choice_embedded(state, options),
            JevQuestion::Score { .. } => Err(SCORE_ERR.into()),
        }
    }

    /// Answers one question about `state`.
    pub fn evaluate(&self, state: &str, question: &JevQuestion) -> Result<JevDecision, String> {
        self.evaluate_embedded(&self.embed(state)?, question)
    }

    /// Answers every question with one shared state embedding. The outer error is a
    /// state-encoding failure; each question carries its own result.
    pub fn batch_evaluate(
        &self,
        state: &str,
        questions: &[JevQuestion],
    ) -> Result<Vec<(String, Result<JevDecision, String>)>, String> {
        let s = self.embed(state)?;
        Ok(questions
            .iter()
            .map(|q| (q.id().to_string(), self.evaluate_embedded(&s, q)))
            .collect())
    }
}

/// Rejects questions the zero-shot encoder cannot answer, before any forward pass.
fn check_supported(q: &JevQuestion) -> Result<(), String> {
    let present = |s: &Option<String>| s.as_deref().is_some_and(|s| !s.trim().is_empty());
    match q {
        JevQuestion::Bool {
            criteria_true,
            criteria_false,
            ..
        } if !(present(criteria_true) && present(criteria_false)) => Err(BOOL_CRITERIA_ERR.into()),
        JevQuestion::Choice { options, .. } if options.is_empty() => {
            Err("choice question has no options".into())
        }
        JevQuestion::Score { .. } => Err(SCORE_ERR.into()),
        _ => Ok(()),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// softmax(x / t), max-subtracted.
fn softmax_scaled(x: &[f32], t: f32) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| ((v - m) / t).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_over_cosines_with_temperature() {
        let p = softmax_scaled(&[0.9, 0.8], 0.05);
        let expect = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((p[0] - expect).abs() < 1e-6);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let flat = softmax_scaled(&[0.9, 0.8], 10.0);
        assert!(flat[0] < 0.51);
    }

    #[test]
    fn unsupported_questions_error() {
        let bool_q = |t: Option<&str>, f: Option<&str>| JevQuestion::Bool {
            id: "b".into(),
            instructions: String::new(),
            criteria_true: t.map(Into::into),
            criteria_false: f.map(Into::into),
        };
        assert_eq!(
            check_supported(&bool_q(Some("yes"), None)),
            Err(BOOL_CRITERIA_ERR.to_string())
        );
        assert_eq!(
            check_supported(&bool_q(None, None)),
            Err(BOOL_CRITERIA_ERR.to_string())
        );
        assert_eq!(check_supported(&bool_q(Some("yes"), Some("no"))), Ok(()));
        let score = JevQuestion::Score {
            id: "s".into(),
            instructions: String::new(),
            criteria: vec!["1".into()],
        };
        assert_eq!(check_supported(&score), Err(SCORE_ERR.to_string()));
    }

    const MINILM: &str = "sentence-transformers/all-MiniLM-L6-v2";

    /// The real all-MiniLM-L6-v2 encoder on `device`, or `None` when its weights are not cached in `disk/`.
    fn cached_minilm(device: Device) -> Option<JevSystem1Model> {
        let dir = std::path::Path::new("disk");
        let cached = dir
            .join(
                "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/main/model.safetensors",
            )
            .exists();
        if !cached {
            eprintln!("SKIP: {MINILM} not cached in disk/");
            return None;
        }
        Some(
            JevSystem1Model::load(&HuggingFaceHub::with_cache_dir(dir.into()), MINILM, device)
                .unwrap(),
        )
    }

    /// n=2 (empty), one word, truncation at max positions, WordPiece [UNK]/non-ASCII, a sentence.
    fn edge_texts() -> [String; 5] {
        [
            String::new(),
            "hello".into(),
            "word ".repeat(600),
            "naïve café 東京 🚀🦀".into(),
            "Delete the user's account and all their personal data.".into(),
        ]
    }

    /// Pins the exact CPU output (f32 bits) of the real encoder, so any change to the CPU path that
    /// moves a bit fails here. Covers the edge texts and both decision kinds. Bits are
    /// host-specific (NEON kernels), hence aarch64 only.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cpu_output_bits_pinned() {
        let Some(m) = cached_minilm(Device::Cpu) else {
            return;
        };
        let [empty, word, long, unicode, sentence] = edge_texts();
        let texts = [empty.as_str(), &word, &long, &unicode, &sentence];
        assert_eq!(m.token_count("").unwrap(), 2);
        assert_eq!(m.token_count(&long).unwrap(), m.encoder.dims.max_positions);
        // FNV-1a over every output bit: stable across Rust releases, unlike DefaultHasher.
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: f32| {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        };
        let mut state = Vec::new();
        for t in texts {
            state = m.embed(t).unwrap();
            assert_eq!(state.len(), 384);
            state.iter().copied().for_each(&mut mix);
        }
        let JevDecision::Bool {
            bool_prob,
            decision,
        } = m
            .evaluate_bool_embedded(
                &state,
                Some("the request asks to delete data"),
                Some("the request keeps data"),
            )
            .unwrap()
        else {
            unreachable!()
        };
        mix(bool_prob);
        assert!(decision);
        let options: Vec<ChoiceOption> = ["privacy", "billing", "shipping"]
            .iter()
            .map(|l| ChoiceOption {
                label: l.to_string(),
                description: None,
            })
            .collect();
        let JevDecision::Choice {
            selected,
            probabilities,
            ..
        } = m.evaluate_choice_embedded(&state, &options).unwrap()
        else {
            unreachable!()
        };
        options.iter().for_each(|o| mix(probabilities[&o.label]));
        assert_eq!(selected, "privacy");
        assert_eq!(
            h, 7844549035239271037,
            "System 1 CPU output bits changed; re-pin only if the numeric change is intended"
        );
    }

    /// P1 gate: on Metal the real encoder matches the CPU (cosine >= 0.99999 on every edge text)
    /// and every decision on a fixed set picks the same answer.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_cpu_embeddings_and_decisions() {
        let (Some(cpu), Some(gpu)) = (cached_minilm(Device::Cpu), cached_minilm(Device::Metal))
        else {
            return;
        };
        assert_eq!(gpu.encoder.device(), Device::Metal);
        for t in edge_texts() {
            let (a, b) = (cpu.embed(&t).unwrap(), gpu.embed(&t).unwrap());
            let cos = dot(&a, &b) / (dot(&a, &a) * dot(&b, &b)).sqrt();
            assert!(cos >= 0.99999, "{t:?}: cosine {cos}");
        }
        let questions = [
            JevQuestion::Bool {
                id: "delete".into(),
                instructions: String::new(),
                criteria_true: Some("the user wants their account and data deleted".into()),
                criteria_false: Some("the user wants to keep their account and data".into()),
            },
            JevQuestion::Choice {
                id: "topic".into(),
                instructions: String::new(),
                options: ["billing", "privacy", "bug", "shipping"]
                    .map(|l| ChoiceOption::new(l, None::<&str>))
                    .to_vec(),
            },
        ];
        for state in [
            "Please remove my account and all my data permanently.",
            "I was charged twice for my subscription this month.",
            "The app crashes when I open the settings page.",
            "Where is my package? It has not arrived yet.",
            "",
        ] {
            let (a, b) = (
                cpu.batch_evaluate(state, &questions).unwrap(),
                gpu.batch_evaluate(state, &questions).unwrap(),
            );
            for ((id, x), (_, y)) in a.iter().zip(&b) {
                match (x.as_ref().unwrap(), y.as_ref().unwrap()) {
                    (
                        JevDecision::Bool {
                            bool_prob: p,
                            decision: d,
                        },
                        JevDecision::Bool {
                            bool_prob: q,
                            decision: e,
                        },
                    ) => {
                        assert_eq!(d, e, "{state:?} {id}");
                        assert!((p - q).abs() < 1e-3, "{state:?} {id}: {p} vs {q}");
                    }
                    (
                        JevDecision::Choice { selected: s, .. },
                        JevDecision::Choice { selected: t, .. },
                    ) => {
                        assert_eq!(s, t, "{state:?} {id}")
                    }
                    other => panic!("{state:?} {id}: decision kinds differ: {other:?}"),
                }
            }
        }
    }

    /// P2 gate: on Core ML / ANE the real encoder matches the CPU (cosine >= 0.999 on edge texts)
    /// and every decision on a fixed set picks the same answer.
    #[cfg(target_os = "macos")]
    #[test]
    fn coreml_matches_cpu_embeddings_and_decisions() {
        let (Some(cpu), Some(ane)) = (cached_minilm(Device::Cpu), cached_minilm(Device::Ane))
        else {
            return;
        };
        assert_eq!(ane.encoder.device(), Device::Ane);
        let report = ane.encoder.executor_report().unwrap();
        assert!(report.contains("Core ML ANE:"), "report: {report}");
        assert!(report.contains("Neural Engine"), "report: {report}");
        let mut min_cos = 1.0f32;
        for t in edge_texts() {
            let (a, b) = (cpu.embed(&t).unwrap(), ane.embed(&t).unwrap());
            let cos = dot(&a, &b) / (dot(&a, &a) * dot(&b, &b)).sqrt();
            assert!(cos >= 0.999, "{t:?}: cosine {cos}");
            if cos < min_cos {
                min_cos = cos;
            }
        }
        let questions = [
            JevQuestion::Bool {
                id: "delete".into(),
                instructions: String::new(),
                criteria_true: Some("the user wants their account and data deleted".into()),
                criteria_false: Some("the user wants to keep their account and data".into()),
            },
            JevQuestion::Choice {
                id: "topic".into(),
                instructions: String::new(),
                options: ["billing", "privacy", "bug", "shipping"]
                    .map(|l| ChoiceOption::new(l, None::<&str>))
                    .to_vec(),
            },
        ];
        let mut max_dp = 0.0f32;
        for state in [
            "Please remove my account and all my data permanently.",
            "I was charged twice for my subscription this month.",
            "The app crashes when I open the settings page.",
            "Where is my package? It has not arrived yet.",
            "",
        ] {
            let (a, b) = (
                cpu.batch_evaluate(state, &questions).unwrap(),
                ane.batch_evaluate(state, &questions).unwrap(),
            );
            for ((id, x), (_, y)) in a.iter().zip(&b) {
                match (x.as_ref().unwrap(), y.as_ref().unwrap()) {
                    (
                        JevDecision::Bool {
                            bool_prob: p,
                            decision: d,
                        },
                        JevDecision::Bool {
                            bool_prob: q,
                            decision: e,
                        },
                    ) => {
                        assert_eq!(d, e, "{state:?} {id}");
                        let dp = (p - q).abs();
                        assert!(dp < 1e-2, "{state:?} {id}: {p} vs {q}");
                        if dp > max_dp {
                            max_dp = dp;
                        }
                    }
                    (
                        JevDecision::Choice { selected: s, .. },
                        JevDecision::Choice { selected: t, .. },
                    ) => {
                        assert_eq!(s, t, "{state:?} {id}")
                    }
                    other => panic!("{state:?} {id}: decision kinds differ: {other:?}"),
                }
            }
        }
        let s = format!("min_cosine={min_cos:.6} max_bool_prob_diff={max_dp:.6}\n");
        let _ = std::fs::write("/tmp/gibbs-p2/accuracy.log", s);
    }
}
