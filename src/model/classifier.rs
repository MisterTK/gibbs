//! Hugging Face sequence classifiers (`text-classification` / `text-ranking` repos). Encoders
//! (BERT, RoBERTa, XLM-RoBERTa, CamemBERT, DistilBERT, ELECTRA, ALBERT, DeBERTa v1/v2/v3,
//! ModernBERT, llama.cpp BERT GGUFs, SetFit bodies) run their family spec through
//! `runtime::encoder`; Llama/Qwen2/GPT-NeoX (GraphLm body), Qwen3/Qwen3.5, Gemma 3/Gemma 4 and
//! GPT-2/OPT (`ln_decoder`) decoders are scored on their last non-pad token. One forward pass
//! per input; scores follow `config.json` (`problem_type`, `id2label`).

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
pub(crate) mod synth;
pub mod ln_decoder;
pub(crate) mod decoder_gpu;
pub use decoder_gpu::gpu_fixture;
mod spm;

use crate::hardware::{Device, QuantFormat, QuantMode};
use crate::model::hf_loader::ChatMessage;
use crate::model::hub::{is_synthetic_model_id, split_weight_file, HuggingFaceHub};
use crate::model::layers::QuantizedLinearLayer;
use crate::model::loaders::{read_checkpoint, SafetensorsWeightReader, WeightReader};
use crate::model::nn;
use crate::model::graph_lm::GraphLm;
use crate::model::{ModelManifest, UniversalModelLoader};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use crate::runtime::encoder::SpecEncoder;

mod setfit;

mod gguf;
pub use gguf::GgufClassifier;

/// `architectures[0]` values this module runs.
pub const ARCHITECTURES: &[&str] = &[
    "BertForSequenceClassification",
    "RobertaForSequenceClassification",
    "XLMRobertaForSequenceClassification",
    "CamembertForSequenceClassification",
    "DistilBertForSequenceClassification",
    "ElectraForSequenceClassification",
    "AlbertForSequenceClassification",
    "DebertaForSequenceClassification",
    "DebertaV2ForSequenceClassification",
    "LlamaForSequenceClassification",
    "Qwen2ForSequenceClassification",
    "ModernBertForSequenceClassification",
    "GPT2ForSequenceClassification",
    "OPTForSequenceClassification",
    "GPTNeoXForSequenceClassification",
    "Qwen3ForSequenceClassification",
    "Qwen3_5ForSequenceClassification",
    "Qwen3_5TextForSequenceClassification",
    "Gemma4ForSequenceClassification",
    "Gemma3TextForSequenceClassification",
];

/// The classifier the engine serves, shared by the HTTP server and the TUI. `Some` makes it the
/// active model: text generation answers 400 until an LLM is loaded again.
pub type ClassifierSlot = Arc<parking_lot::RwLock<Option<Arc<SequenceClassifier>>>>;

/// Separates (text, text_pair) in TUI prompts and `--prompt`: `query || passage`.
pub const PAIR_SEPARATOR: &str = "||";

/// `a || b` → (`a`, Some(`b`)), trimmed; text without the separator is a single input.
pub fn split_pair(s: &str) -> (&str, Option<&str>) {
    match s.split_once(PAIR_SEPARATOR) {
        Some((a, b)) => (a.trim(), Some(b.trim())),
        None => (s.trim(), None),
    }
}

/// How logits become scores (the HF `problem_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// single_label_classification: probabilities over the labels.
    Softmax,
    /// multi_label_classification: an independent probability per label.
    Sigmoid,
    /// regression, or one label without a problem_type (rerankers, reward models): raw logits.
    Identity,
}

/// One label and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

/// A loaded sequence classifier.
pub struct SequenceClassifier {
    pub repo_id: String,
    pub architecture: String,
    pub labels: Vec<String>,
    pub activation: Activation,
    body: Body,
}

enum Body {
    /// An encoder family spec (`runtime::encoder`): `classify` logits of one sequence.
    Spec {
        enc: SpecEncoder,
        tok: tokenizers::Tokenizer,
    },
    /// Causal LM scored by `score` on the final hidden state of its last non-pad token.
    Decoder {
        llm: Box<dyn DecoderBackbone>,
        score: Lin,
        pad: Option<u32>,
        /// sentence-transformers CrossEncoder: a pair is `tokenizer(text, pair)` (the two id
        /// sequences concatenated), not a chat.
        concat_pair: bool,
        /// Metal: `llm` scores on the GPU (`score_gpu`); `score` stays for the label count.
        gpu: bool,
    },
    /// SetFit: sentence embedding → sklearn LogisticRegression (`predict_proba` scores).
    SetFit(setfit::SetFit),
    /// llama.cpp BERT-family GGUF with a `cls.output` head.
    Gguf(GgufClassifier),
    /// fastText supervised model on the bag runtime: `predict` already returns probabilities.
    FastText(crate::runtime::bag::BagClassifier),
}

/// A decoder body for `*ForSequenceClassification`: HF scores the final-norm hidden state of the
/// last non-pad token with a bias-free `score` layer.
pub trait DecoderBackbone: Send + Sync {
    fn tokenizer(&self) -> &crate::model::tokenizer::Tokenizer;
    fn context_len(&self) -> usize;
    fn hidden(&self) -> usize;
    /// Final-norm hidden state `[hidden]` of the last of `ids` (non-empty; one uncached pass).
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String>;
    /// Metal: compile the score head into a GPU program (called once at load when the backend is
    /// Metal). `score` is `[labels, hidden]` row-major, bias-free.
    fn attach_score_gpu(&mut self, _score: &[f32], _labels: usize) -> Result<(), String> {
        Err(format!(
            "gpu-only: {} has no Metal seq-cls program; run with --backend cpu",
            std::any::type_name_of_val(self)
        ))
    }
    /// Logits `[labels]` of `ids` (non-empty, last non-pad token last) computed entirely on the
    /// GPU. Only valid after a successful `attach_score_gpu`.
    fn score_gpu(&self, _ids: &[u32]) -> Result<Vec<f32>, String> {
        Err("no GPU score program attached".into())
    }
    /// Token ids of `text` (with `pair`, the second segment): `Tokenizer::encode`, and
    /// [`encode_pair`] for a pair.
    fn encode(&self, text: &str, pair: Option<&str>) -> Result<Vec<u32>, String> {
        let tok = self.tokenizer();
        match pair {
            None => Ok(tok.encode(text)),
            Some(answer) => encode_pair(tok, text, answer),
        }
    }
}

/// A (text, answer) pair for a decoder: with no chat template, HF `tokenizer(text, pair)` (how
/// such reward models' cards score (prompt, response)); else the finished user/assistant chat
/// reward models score (HF `apply_chat_template(conv, tokenize=False)`: no generation prompt).
pub(crate) fn encode_pair(
    tok: &crate::model::tokenizer::Tokenizer,
    text: &str,
    answer: &str,
) -> Result<Vec<u32>, String> {
    let Some(spec) = &tok.chat_template else {
        let hf = tok
            .hf
            .as_ref()
            .ok_or("the model has no chat template or tokenizer.json")?;
        return Ok(hf
            .encode((text, answer), true)
            .map_err(|e| format!("tokenize: {e}"))?
            .get_ids()
            .to_vec());
    };
    let conv = [
        ChatMessage::new("user", text),
        ChatMessage::new("assistant", answer),
    ];
    let rendered = spec
        .render(&conv, None, false, &Value::Null)
        .map_err(|e| format!("chat template: {e}"))?;
    Ok(tok.encode(&rendered))
}

impl DecoderBackbone for GraphLm {
    fn tokenizer(&self) -> &crate::model::tokenizer::Tokenizer {
        &self.tokenizer
    }
    fn context_len(&self) -> usize {
        self.config.max_seq_len
    }
    fn hidden(&self) -> usize {
        self.config.hidden_dim
    }
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        crate::hardware::cpu_compute::enter("decoder::last_hidden");
        Ok(self.final_hidden(ids))
    }
}

/// `architecture` of a fastText hierarchical-softmax re-export (`config.json` + `weights.pt`).
const FASTTEXT_EXPORT: &str = "fasttext_hierarchical_softmax";

/// `architectures[0]` of `repo_id` when it is a `*ForSequenceClassification` checkpoint (its
/// config.json from the cache, else fetched), or a PEFT LoRA `SEQ_CLS` adapter; `None` for
/// anything else.
pub fn detect(hub: &HuggingFaceHub, repo_id: &str) -> Option<String> {
    if is_synthetic_model_id(repo_id)
        || split_weight_file(repo_id).1.is_some()
        || Path::new(repo_id).is_dir()
    {
        return None;
    }
    let config = hub
        .locate_model(repo_id)
        .config_path
        .or_else(|| hub.fetch_file(repo_id, "config.json").ok());
    let cfg = config.as_deref().and_then(|p| read_json(p).ok());
    let arch = cfg.as_ref().and_then(|c| c["architectures"][0].as_str());
    if let Some(a) = arch.filter(|a| a.ends_with("ForSequenceClassification")) {
        return Some(a.to_string());
    }
    // Generators never carry a classifier; skip the adapter probe for them.
    if arch.is_some_and(is_generator) {
        return None;
    }
    // No config.json: a llama.cpp BERT-family GGUF classifier/reranker export, or a fastText model.
    if config.is_none() && gguf::hub_gguf_arch(hub, repo_id).as_deref() == Some("bert") {
        return Some("GGUF bert classifier".into());
    }
    if cfg
        .as_ref()
        .is_some_and(|c| c["architecture"] == FASTTEXT_EXPORT)
        || (config.is_none() && crate::model::fasttext::hub_file(hub, repo_id).is_some())
    {
        return Some("fastText".into());
    }
    if hub.fetch_file(repo_id, "config_setfit.json").is_ok() {
        return Some("SetFit".into());
    }
    let adapter = read_json(&hub.fetch_file(repo_id, "adapter_config.json").ok()?).ok()?;
    (adapter["task_type"] == "SEQ_CLS").then(|| "PEFT LoRA (SEQ_CLS)".to_string())
}

/// Text generators: `*ForCausalLM`, `*ForConditionalGeneration`, `*LMHeadModel`.
pub fn is_generator(arch: &str) -> bool {
    ["ForCausalLM", "ForConditionalGeneration", "LMHeadModel"]
        .iter()
        .any(|s| arch.ends_with(s))
}

/// transformers `model_type`s whose base classes (`<Family>Model`, `...ForMaskedLM`) are plain
/// backbones; another class name over them is the repo's own code.
const TRANSFORMERS_TYPES: &[&str] = &[
    "bert",
    "roberta",
    "xlm-roberta",
    "distilbert",
    "deberta",
    "deberta-v2",
    "modernbert",
    "electra",
    "albert",
    "camembert",
    "mpnet",
    "t5",
    "llama",
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen3_5_text",
    "gemma",
    "gemma2",
    "gemma3_text",
    "gemma4",
    "eurobert",
];

/// Picker verdict for a Hub search result `m` (`/api/models` JSON with config, library_name,
/// tags, siblings, gguf) that a classifier or a decoder tagged for classification decides:
/// `Some(Ok)` loads, `Some(Err(reason))` is refused for that reason, `None` leaves the repo to
/// the generic rules (a text-generation repo, an unrelated task).
pub fn hub_verdict(m: &Value) -> Option<Result<(), String>> {
    let cfg = &m["config"];
    let tag = m["pipeline_tag"].as_str().unwrap_or("");
    let lib = m["library_name"].as_str().unwrap_or("");
    let files: Vec<&str> = m["siblings"]
        .as_array()
        .map(|a| a.iter().filter_map(|s| s["rfilename"].as_str()).collect())
        .unwrap_or_default();
    let tags: Vec<&str> = m["tags"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let classifier_task = matches!(
        tag,
        "text-classification" | "text-ranking" | "zero-shot-classification" | "sentence-similarity"
    );
    let model_type = cfg["model_type"].as_str();
    let arch = cfg["architectures"][0].as_str();
    let custom_file = files.iter().find(|f| {
        let name = f.rsplit('/').next().unwrap_or(f);
        (name.starts_with("modeling_") && name.ends_with(".py"))
            || [".pt", ".pth", ".pkl", ".safetensors", ".json"]
                .iter()
                .any(|ext| {
                    name.strip_suffix(ext).is_some_and(|stem| {
                        stem.ends_with("head") || stem.ends_with("heads") || stem == "readout"
                    })
                })
    });

    // PEFT adapters: the API inlines adapter_config.json as `config.peft`.
    let peft = &cfg["peft"];
    if peft.is_object() || files.contains(&"adapter_config.json") {
        let base = peft["base_model_name_or_path"].as_str().unwrap_or("?");
        return Some(match peft["task_type"].as_str() {
            _ if custom_file.is_some() => Err(format!(
                "PEFT adapter on {base} with a custom head ({})",
                custom_file.unwrap()
            )),
            _ if base.starts_with('/') || base.starts_with('.') => {
                Err(format!("PEFT adapter whose base is a local path ({base})"))
            }
            // SEQ_CLS classifies; CAUSAL_LM generates from the merged base (crate::model::peft).
            Some("SEQ_CLS" | "CAUSAL_LM") | None => Ok(()),
            Some(t) => Err(format!("PEFT {t} adapter on {base} (not a classifier)")),
        });
    }
    if let Some(a) = arch {
        if ARCHITECTURES.contains(&a) || files.contains(&"config_setfit.json") {
            return Some(Ok(()));
        }
        if is_generator(a) {
            // The LLM loaders: causal-LM family specs, plus the native models.
            let runs = model_type.is_some_and(|t| {
                crate::arch::registry().is_ok_and(|r| r.runs_model_type(t))
                    || crate::model::native::NATIVE_MODEL_TYPES.contains(&t)
            });
            return (runs && !cfg["auto_map"].is_object()).then_some(Ok(()));
        }
        if !classifier_task {
            return None;
        }
        if cfg["auto_map"].is_object() || custom_file.is_some_and(|f| f.ends_with(".py")) {
            return Some(Err(format!("custom architecture '{a}' (remote code)")));
        }
        if let Some(f) = custom_file {
            return Some(Err(format!(
                "custom architecture '{a}' + head '{f}' (remote code)"
            )));
        }
        // A transformers class is named after its model_type (`BertModel`, `ModernBertForMaskedLM`,
        // `BigBirdForSequenceClassification`); any other class comes from the repo's own code.
        let fold = |s: &str| s.to_lowercase().replace(['-', '_'], "");
        let seq_cls = a.ends_with("ForSequenceClassification");
        let named = model_type.is_some_and(|t| {
            fold(a).starts_with(&fold(t)) && (seq_cls || TRANSFORMERS_TYPES.contains(&t))
        });
        return Some(Err(if !named {
            format!("custom architecture '{a}' (remote code)")
        } else if !seq_cls {
            format!("'{a}' has no sequence-classification head")
        } else {
            format!("architecture '{a}' is not supported")
        }));
    }
    if !classifier_task && lib != "fasttext" {
        return None;
    }
    // No transformers config: decide from the library, the tags and the files.
    if tags.contains(&"pytorch_model_hub_mixin") {
        return Some(Err(
            "custom PyTorch model (PyTorchModelHubMixin, remote code)".into(),
        ));
    }
    let gguf: Vec<&&str> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    Some(if !gguf.is_empty() {
        if gguf
            .iter()
            .all(|f| f.starts_with("loras/") || f.contains("lora"))
        {
            Err("GGUF LoRA adapter (needs its base model)".into())
        } else if let Some(a) = m["gguf"]["architecture"]
            .as_str()
            .filter(|a| a.contains("bert") && *a != "bert")
        {
            // ponytail: GGUF classifiers load for llama.cpp's `bert` arch (BERT/RoBERTa/XLM-R);
            // ModernBERT/Nomic/Jina encoders need their own graphs.
            Err(format!(
                "GGUF encoder architecture '{a}' (only 'bert' classifiers load)"
            ))
        } else {
            // A decoder GGUF generates; a `bert` GGUF classifier loads in the classifier.
            Ok(())
        }
    } else if lib == "fasttext" || tags.contains(&"fasttext") {
        // A `.bin`/`.ftz` model, or the hierarchical-softmax re-export (config + vocab + weights.pt).
        let export = ["config.json", "vocab.json", "weights.pt"]
            .iter()
            .all(|f| files.contains(f));
        if export
            || files
                .iter()
                .any(|f| f.ends_with(".bin") || f.ends_with(".ftz"))
        {
            Ok(())
        } else {
            Err("fastText repo without a .bin/.ftz model file".into())
        }
    } else if lib.is_empty() && model_type.is_none() && files.iter().any(|f| f.ends_with(".py")) {
        Err("custom architecture (remote code, no config.json)".into())
    } else if !lib.is_empty() && lib != "transformers" {
        Err(format!("'{lib}' model (not a transformers checkpoint)"))
    } else if let Some(t) = model_type {
        Err(format!(
            "config.json names no architecture (model_type '{t}')"
        ))
    } else if tags.contains(&"onnx") || files.iter().any(|f| f.ends_with(".onnx")) {
        Err("ONNX export without config.json".into())
    } else {
        Err("no config.json (custom checkpoint)".into())
    })
}

/// `architectures[0]` of the config.json at `path` when it names a sequence classifier.
pub fn architecture_in(path: &Path) -> Option<String> {
    let cfg: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let arch = cfg["architectures"][0].as_str()?;
    arch.ends_with("ForSequenceClassification")
        .then(|| arch.to_string())
}

impl SequenceClassifier {
    /// Fetches config.json, the tokenizer files, and the weights of `repo_id`, and builds the
    /// classifier on `device`. BERT-family encoders run on the CPU or Metal; DeBERTa and decoder
    /// classifiers on the CPU only.
    pub fn load(hub: &HuggingFaceHub, repo_id: &str, device: Device) -> Result<Self, String> {
        let resolved = resolve_config(hub, repo_id);
        // SetFit: a sentence-transformers body (`BertModel`) + a scikit-learn head.
        if resolved.is_err() && hub.fetch_file(repo_id, "config_setfit.json").is_ok() {
            return Self::from_setfit(hub, repo_id, device);
        }
        if resolved.is_err() && gguf::hub_gguf_arch(hub, repo_id).as_deref() == Some("bert") {
            let token = std::env::var("HF_TOKEN").ok();
            let art = hub.auto_download_and_cache(repo_id, token.as_deref())?;
            let path = art
                .weights_path
                .ok_or_else(|| format!("{repo_id}: no GGUF file"))?;
            return Self::from_gguf(repo_id, &path, device);
        }
        if resolved.is_err() {
            let cfg = hub
                .fetch_file(repo_id, "config.json")
                .and_then(|p| read_json(&p));
            if cfg.is_ok_and(|c| c["architecture"] == FASTTEXT_EXPORT) {
                hub.fetch_file(repo_id, "vocab.json")?;
                let w = hub.fetch_file(repo_id, "weights.pt")?;
                return Self::from_fasttext(repo_id, w.parent().unwrap_or(Path::new(".")));
            }
            if let Some(file) = crate::model::fasttext::hub_file(hub, repo_id) {
                return Self::from_fasttext(repo_id, &hub.fetch_file(repo_id, &file)?);
            }
        }
        let (cfg, arch, peft) = resolved?;
        if arch.starts_with("Qwen3Moe") {
            return Err(format!(
                "{repo_id}: {arch} is not supported: mixture-of-experts decoder layers (routed expert MLPs) are not implemented"
            ));
        }
        if !ARCHITECTURES.contains(&arch.as_str()) {
            if !cfg["auto_map"].is_null() {
                return Err(format!(
                    "{repo_id}: custom architecture '{arch}' (remote code)"
                ));
            }
            return Err(format!(
                "{repo_id}: {arch} is not supported (supported: {})",
                ARCHITECTURES.join(", ")
            ));
        }
        // A PEFT adapter reads its base model's tokenizer unless it ships one.
        let tok_repo = match &peft {
            Some((base, _)) if hub.fetch_file(repo_id, "tokenizer.json").is_err() => base.as_str(),
            _ => repo_id,
        };
        let pad_id = cfg["pad_token_id"].as_u64().map(|v| v as u32);

        let body = if is_decoder(&arch) {
            // No `cpu_only` gate: Metal gets the GPU seq-cls program (`decoder_gpu`).
            if peft.is_some() {
                // ponytail: LoRA merges into the f32 encoder checkpoints only; decoder bodies load
                // through their own readers (merge there when a decoder SEQ_CLS adapter matters).
                return Err(format!(
                    "{repo_id}: PEFT LoRA adapters on decoder bases ({arch}) are not supported"
                ));
            }
            // Same token source as `fetch_file` (the encoder path).
            let token = std::env::var("HF_TOKEN").ok();
            let artifacts = hub.auto_download_and_cache(repo_id, token.as_deref())?;
            let weights = artifacts
                .weights_path
                .as_deref()
                .ok_or_else(|| format!("{repo_id}: no weight file"))?;
            let llm: Box<dyn DecoderBackbone> = match arch.as_str() {
                "GPT2ForSequenceClassification" | "OPTForSequenceClassification" => {
                    Box::new(ln_decoder::LnDecoder::load(hub, repo_id, &cfg)?)
                }
                "Gemma4ForSequenceClassification"
                | "Gemma3TextForSequenceClassification"
                | "Qwen3ForSequenceClassification"
                | "Qwen3_5ForSequenceClassification"
                | "Qwen3_5TextForSequenceClassification" => {
                    Box::new(crate::runtime::seqcls::SpecBackbone::load(&artifacts, device)?)
                }
                _ => {
                    let mut llm = crate::model::hf_loader::load_causal_lm(
                        &artifacts,
                        QuantMode::Uncompressed,
                    )?;
                    // The causal LM head is never used: free it.
                    llm.lm_head =
                        QuantizedLinearLayer::from_f32("unused", 1, 1, &[0.0], QuantFormat::FP32)?;
                    Box::new(decoder_gpu::GraphLmSeq::new(llm))
                }
            };
            let h = llm.hidden();
            let score = load_score(weights)?;
            if score.is_empty() || score.len() % h != 0 {
                return Err(format!(
                    "{repo_id}: score.weight has {} values, not [num_labels, {h}]",
                    score.len()
                ));
            }
            let score = Lin {
                out_f: score.len() / h,
                w: score,
                b: None,
                in_f: h,
            };
            // Metal: the score head and backbone become a GPU program; a backbone without one
            // fails the load (`gpu-only:`) instead of running on the CPU.
            let mut llm = llm;
            let gpu = device != Device::Cpu;
            if gpu {
                llm.attach_score_gpu(&score.w, score.out_f)?;
            }
            Body::Decoder {
                llm,
                score,
                pad: pad_id,
                concat_pair: cfg["sentence_transformers"].is_object(),
                gpu,
            }
        } else {
            let tok_cfg = hub
                .fetch_file(tok_repo, "tokenizer_config.json")
                .ok()
                .map(|p| read_json(&p))
                .transpose()?
                .unwrap_or(Value::Null);
            let enc = spec_encoder(hub, repo_id, &cfg, &arch, peft.is_some(), device)?;
            let tok = load_tokenizer(hub, tok_repo, &arch, &tok_cfg, enc.capacity()?)?;
            Body::Spec { enc, tok }
        };

        let n = match &body {
            Body::Spec { enc, .. } => enc.var("labels")?,
            Body::Decoder { score, .. } => score.out_f,
            Body::SetFit(_) | Body::Gguf(_) | Body::FastText(_) => {
                unreachable!("SetFit, GGUF and fastText repos load through their constructors")
            }
        };
        let labels = (0..n)
            .map(|i| {
                cfg["id2label"][i.to_string()]
                    .as_str()
                    .map_or_else(|| format!("LABEL_{i}"), str::to_string)
            })
            .collect();
        let activation = activation_of(&cfg, n);
        Ok(Self {
            repo_id: repo_id.to_string(),
            architecture: arch,
            labels,
            activation,
            body,
        })
    }

    /// A fastText `.bin`/`.ftz` classifier, or (when `path` is a directory) a fastText
    /// hierarchical-softmax re-export (`config.json` + `vocab.json` + `weights.pt`).
    pub fn from_fasttext(repo_id: &str, path: &Path) -> Result<Self, String> {
        // The graph runs on the CPU, or on Metal once the GPU-only tripwire is armed.
        let dev = if crate::hardware::cpu_compute::banned() {
            Device::Metal
        } else {
            Device::Cpu
        };
        let m = if path.is_dir() {
            crate::model::fasttext::load_export(path)?
        } else {
            crate::model::fasttext::load(path)?
        };
        let ft = crate::runtime::bag::BagClassifier::new("fasttext", m, dev)?;
        Ok(Self {
            repo_id: repo_id.to_string(),
            architecture: "fastText".into(),
            labels: ft.labels().to_vec(),
            activation: Activation::Identity,
            body: Body::FastText(ft),
        })
    }

    /// A llama.cpp BERT/XLM-R classifier or reranker GGUF at `path`: labels from
    /// `<arch>.classifier.output_labels`; one logit is a raw score, more are softmaxed.
    pub fn from_gguf(repo_id: &str, path: &Path, device: Device) -> Result<Self, String> {
        let g = GgufClassifier::load(path, device)?;
        Ok(Self {
            repo_id: repo_id.to_string(),
            architecture: "GGUF bert classifier".into(),
            labels: g.labels.clone(),
            activation: if g.labels.len() == 1 {
                Activation::Identity
            } else {
                Activation::Softmax
            },
            body: Body::Gguf(g),
        })
    }

    /// Where the body runs.
    pub fn device(&self) -> Device {
        match &self.body {
            Body::Spec { enc, .. } => enc.device(),
            Body::SetFit(s) => s.device(),
            Body::Gguf(g) => g.device(),
            Body::FastText(b) => b.device(),
            Body::Decoder { gpu: true, .. } => Device::Metal,
            _ => Device::Cpu,
        }
    }

    /// One line for logs and the TUI: architecture, labels, scoring, device.
    pub fn summary(&self) -> String {
        let scoring = match self.activation {
            Activation::Softmax => "softmax",
            Activation::Sigmoid => "sigmoid (multi-label)",
            Activation::Identity => "raw score",
        };
        format!(
            "{}, {} label{} ({scoring}), on {}",
            self.architecture,
            self.labels.len(),
            if self.labels.len() == 1 { "" } else { "s" },
            self.device()
        )
    }

    /// Scores `text` (with `pair` as the second segment: cross-encoders and rerankers take
    /// (query, passage); decoder classifiers render it as a user/assistant chat). Returns every
    /// label sorted by descending score, and the number of tokens the model read.
    pub fn classify(
        &self,
        text: &str,
        pair: Option<&str>,
    ) -> Result<(Vec<LabelScore>, usize), String> {
        let (logits, n) = match &self.body {
            Body::Spec { enc, tok } => {
                let e = encode(tok, text, pair)?;
                (enc.run(e.get_ids(), e.get_type_ids())?, e.get_ids().len())
            }
            Body::Gguf(g) => g.logits(text, pair)?,
            Body::FastText(_) if pair.is_some() => {
                return Err("fastText classifiers take a single text, not a pair".into())
            }
            Body::FastText(ft) => ft.predict(text)?,
            Body::Decoder {
                llm,
                score,
                pad,
                concat_pair,
                gpu,
            } => {
                let ids = match pair {
                    Some(b) if *concat_pair => {
                        let tok = llm.tokenizer();
                        [tok.encode(text), tok.encode(b)].concat()
                    }
                    _ => llm
                        .encode(text, pair)
                        .map_err(|e| format!("{}: {e}", self.repo_id))?,
                };
                if ids.len() > llm.context_len() {
                    return Err(format!(
                        "{} tokens exceed the {}-token context",
                        ids.len(),
                        llm.context_len()
                    ));
                }
                // HF scores the last token that is not the pad token.
                let last = ids
                    .iter()
                    .rposition(|&t| Some(t) != *pad)
                    .ok_or("the input has no non-pad token")?;
                let logits = if *gpu {
                    llm.score_gpu(&ids[..=last])?
                } else {
                    score.apply(&llm.last_hidden(&ids[..=last])?, 1)
                };
                (logits, ids.len())
            }
            Body::SetFit(_) if pair.is_some() => {
                return Err("SetFit classifiers take a single text, not a pair".into())
            }
            Body::SetFit(s) => s.proba(text)?,
        };
        let scores = match self.activation {
            Activation::Softmax => {
                let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
                let s: f32 = e.iter().sum();
                e.into_iter().map(|v| v / s).collect()
            }
            Activation::Sigmoid => logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
            Activation::Identity => logits,
        };
        let mut out: Vec<LabelScore> = self
            .labels
            .iter()
            .zip(scores)
            .map(|(l, score)| LabelScore {
                label: l.clone(),
                score,
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok((out, n))
    }
}

/// The encoder spec of classifier `repo_id` on `device`. A PEFT adapter is merged into its base
/// once (`peft::prepare_seq_cls`) and its resolved `cfg` (classification architecture, labels)
/// written beside the merged weights.
fn spec_encoder(
    hub: &HuggingFaceHub,
    repo_id: &str,
    cfg: &Value,
    arch: &str,
    peft: bool,
    device: Device,
) -> Result<SpecEncoder, String> {
    let token = std::env::var("HF_TOKEN").ok();
    if peft {
        crate::model::peft::prepare_seq_cls(hub, repo_id, token.as_deref())?;
    }
    let mut a = hub.auto_download_and_cache(repo_id, token.as_deref())?;
    if peft {
        let dir = a
            .weights_path
            .as_deref()
            .and_then(Path::parent)
            .ok_or_else(|| format!("{repo_id}: no merged snapshot"))?;
        let path = dir.join("config.json");
        let mut cfg = cfg.clone();
        // The base names its own class (`BertModel`, `...ForMaskedLM`); the spec matches the
        // classification architecture.
        cfg["architectures"] = serde_json::json!([arch]);
        std::fs::write(&path, cfg.to_string()).map_err(|e| format!("{}: {e}", path.display()))?;
        a.config_path = Some(path);
    }
    SpecEncoder::load(&a, None, device, "classify").map_err(|e| format!("{repo_id}: {e}"))
}

/// config.json and the `*ForSequenceClassification` architecture to build. A PEFT LoRA adapter
/// repo (`adapter_config.json`, task `SEQ_CLS`) resolves to its base model's config with the
/// adapter repo's labels, and returns (base repo, adapter config).
fn resolve_config(
    hub: &HuggingFaceHub,
    repo_id: &str,
) -> Result<(Value, String, Option<(String, Value)>), String> {
    let own = hub
        .fetch_file(repo_id, "config.json")
        .and_then(|p| read_json(&p));
    let own_arch = own
        .as_ref()
        .ok()
        .and_then(|c| c["architectures"][0].as_str())
        .map(str::to_string);
    if let (Ok(cfg), Some(a)) = (&own, &own_arch) {
        if a.ends_with("ForSequenceClassification") {
            return Ok((cfg.clone(), a.clone(), None));
        }
    }
    let Ok(ad) = hub
        .fetch_file(repo_id, "adapter_config.json")
        .and_then(|p| read_json(&p))
    else {
        return match own {
            Err(e) => Err(e),
            Ok(cfg) => Err(match (&own_arch, cfg["auto_map"].is_null()) {
                (Some(a), false) => format!("{repo_id}: custom architecture '{a}' (remote code)"),
                (Some(a), true) => format!("{repo_id}: {a} has no sequence-classification head"),
                (None, _) => format!("{repo_id}: config.json names no architecture"),
            }),
        };
    };
    let task = ad["task_type"].as_str().unwrap_or("none");
    let kind = ad["peft_type"].as_str().unwrap_or("none");
    if kind != "LORA" || task != "SEQ_CLS" {
        return Err(format!(
            "{repo_id}: PEFT {kind} adapter for task {task}; only LORA SEQ_CLS adapters classify"
        ));
    }
    if ad["use_dora"] == true
        || ad["fan_in_fan_out"] == true
        || ad["rank_pattern"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
        || ad["alpha_pattern"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
    {
        return Err(format!(
            "{repo_id}: LoRA variant not supported (DoRA, fan_in_fan_out or per-module rank/alpha)"
        ));
    }
    let base = ad["base_model_name_or_path"]
        .as_str()
        .filter(|b| b.split('/').count() == 2 && !b.starts_with('.'))
        .ok_or_else(|| {
            format!(
                "{repo_id}: adapter base_model_name_or_path {} is not a Hub repo",
                ad["base_model_name_or_path"]
            )
        })?
        .to_string();
    let mut cfg = read_json(&hub.fetch_file(&base, "config.json")?)?;
    let base_arch = cfg["architectures"][0]
        .as_str()
        .ok_or_else(|| format!("{base}: config.json names no architecture"))?;
    let family = [
        "ForSequenceClassification",
        "ForMaskedLM",
        "ForPreTraining",
        "ForCausalLM",
        "Model",
    ]
    .iter()
    .find_map(|s| base_arch.strip_suffix(s))
    .unwrap_or(base_arch);
    let arch = format!("{family}ForSequenceClassification");
    // Labels and scoring of the fine-tune live in the adapter repo's config.json, when it has one.
    if let Ok(own) = &own {
        for k in ["id2label", "label2id", "problem_type"] {
            if !own[k].is_null() {
                cfg[k] = own[k].clone();
            }
        }
    }
    Ok((cfg, arch, Some((base, ad))))
}

/// Decoder `*ForSequenceClassification` architectures (scored on the last non-pad token).
fn is_decoder(arch: &str) -> bool {
    [
        "Llama",
        "Qwen2",
        "Qwen3",
        "GPT2",
        "OPT",
        "GPTNeoX",
        "Gemma4",
        "Gemma3Text",
    ]
    .iter()
    .any(|p| arch.starts_with(p))
}

fn read_json(p: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", p.display()))
}

fn encode(
    tok: &tokenizers::Tokenizer,
    text: &str,
    pair: Option<&str>,
) -> Result<tokenizers::Encoding, String> {
    let e = match pair {
        Some(p) => tok.encode((text, p), true),
        None => tok.encode(text, true),
    }
    .map_err(|e| format!("tokenize: {e}"))?;
    if e.get_ids().is_empty() {
        return Err("tokenizer produced no tokens".into());
    }
    Ok(e)
}

/// `score.weight` of a decoder checkpoint, read while the f32 body is already resident, so only
/// that tensor is read: safetensors (one file, or the shard of `<file>.index.json` holding it) are
/// memory-mapped and touch just the header and the tensor's pages. Other formats (`.bin`, ...)
/// have no random access and are read whole.
fn load_score(weights: &Path) -> Result<Vec<f32>, String> {
    let name = weights.to_string_lossy();
    let shard = if name.ends_with(".safetensors.index.json") {
        let json = std::fs::read(weights).map_err(|e| format!("{}: {e}", weights.display()))?;
        let index = UniversalModelLoader::parse_safetensors_index(&json)?;
        let file = index
            .get("score.weight")
            .ok_or_else(|| format!("{}: no score.weight in weight_map", weights.display()))?;
        weights.parent().unwrap_or(Path::new(".")).join(file)
    } else if name.ends_with(".safetensors") {
        weights.to_path_buf()
    } else {
        let (manifest, data) = read_checkpoint(weights)?;
        return UniversalModelLoader::extract_tensor_from_manifest(
            &manifest,
            &data,
            "score.weight",
        );
    };
    SafetensorsWeightReader::new(&shard)
        .and_then(|r| r.load_tensor("score.weight"))
        .map_err(|e| format!("{}: {e}", shard.display()))
}

/// The repo's weights (`model.safetensors`, else `pytorch_model.bin`) read whole as f32: the
/// `ln_decoder` seq-cls bodies (legacy: until Phase 4).
struct Checkpoint {
    manifest: ModelManifest,
    data: Vec<u8>,
}

impl Checkpoint {
    fn open(hub: &HuggingFaceHub, repo_id: &str) -> Result<Self, String> {
        let path = hub.fetch_file(repo_id, "model.safetensors").or_else(|st| {
            hub.fetch_file(repo_id, "pytorch_model.bin").map_err(|bin| {
                format!("{repo_id}: no model.safetensors ({st}) or pytorch_model.bin ({bin})")
            })
        })?;
        let (manifest, data) = read_checkpoint(&path)?;
        Ok(Self { manifest, data })
    }

    fn shape(&self, name: &str) -> Option<&[usize]> {
        self.manifest.tensors.get(name).map(|t| t.shape.as_slice())
    }

    fn has(&self, name: &str) -> bool {
        self.manifest.tensors.contains_key(name)
    }

    /// Tensor `name`, which must have `shape`.
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let got = self.shape(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if got != shape {
            return Err(format!("tensor {name} has shape {got:?}, expected {shape:?}"));
        }
        UniversalModelLoader::extract_tensor_from_manifest(&self.manifest, &self.data, name)
    }

    /// `name.weight` `[out, in_f]` and `name.bias` (when present) as a linear layer.
    fn linear(&self, name: &str, in_f: usize) -> Result<Lin, String> {
        let w = format!("{name}.weight");
        let out_f = match self.shape(&w) {
            Some(&[o, i]) if i == in_f => o,
            other => return Err(format!("tensor {w} has shape {other:?}, expected [_, {in_f}]")),
        };
        let b = format!("{name}.bias");
        Ok(Lin {
            w: self.get(&w, &[out_f, in_f])?,
            b: if self.has(&b) { Some(self.get(&b, &[out_f])?) } else { None },
            in_f,
            out_f,
        })
    }

    /// `name.weight` and `name.bias` of a LayerNorm over `d`.
    fn layer_norm(&self, name: &str, d: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((
            self.get(&format!("{name}.weight"), &[d])?,
            self.get(&format!("{name}.bias"), &[d])?,
        ))
    }
}

/// tokenizer.json, else the slow-tokenizer files (BERT `vocab.txt`, RoBERTa / DeBERTa v1
/// `vocab.json` + `merges.txt`, ALBERT / CamemBERT SentencePiece models) rebuilt as the fast
/// tokenizer HF converts them to. Truncates to `tokenizer_config.json`'s `model_max_length`, at
/// most `capacity` tokens; never pads.
fn load_tokenizer(
    hub: &HuggingFaceHub,
    repo_id: &str,
    arch: &str,
    tok_cfg: &Value,
    capacity: usize,
) -> Result<tokenizers::Tokenizer, String> {
    use tokenizers::AddedToken;
    let mut tok = match hub.fetch_file(repo_id, "tokenizer.json") {
        Ok(p) => tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| format!("load {}: {e}", p.display()))?,
        Err(json_err) if arch.starts_with("Bert") || arch.starts_with("DistilBert") => {
            let vocab = hub.fetch_file(repo_id, "vocab.txt").map_err(|e| {
                format!("{repo_id}: no tokenizer.json ({json_err}) or vocab.txt ({e})")
            })?;
            let wp = tokenizers::models::wordpiece::WordPiece::from_file(&vocab.to_string_lossy())
                .unk_token("[UNK]".into())
                .build()
                .map_err(|e| format!("{}: {e}", vocab.display()))?;
            let mut t = tokenizers::Tokenizer::new(wp);
            let lower = tok_cfg["do_lower_case"].as_bool().unwrap_or(true);
            t.with_normalizer(Some(tokenizers::normalizers::BertNormalizer::new(
                true, true, None, lower,
            )));
            t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::bert::BertPreTokenizer));
            let id = |s: &str| {
                t.token_to_id(s)
                    .ok_or_else(|| format!("{repo_id}: vocab.txt has no {s}"))
            };
            let (cls, sep) = (id("[CLS]")?, id("[SEP]")?);
            t.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
                ("[SEP]".into(), sep),
                ("[CLS]".into(), cls),
            )));
            let specials: Vec<AddedToken> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect();
            t.add_special_tokens(&specials);
            t
        }
        Err(json_err) if arch.starts_with("Albert") || arch.starts_with("Camembert") => {
            let file = if arch.starts_with("Albert") {
                "spiece.model"
            } else {
                "sentencepiece.bpe.model"
            };
            let model = hub.fetch_file(repo_id, file).map_err(|e| {
                format!("{repo_id}: no tokenizer.json ({json_err}) or {file} ({e})")
            })?;
            spm::tokenizer(&model, arch, tok_cfg)?
        }
        Err(json_err) if arch.contains("Roberta") || arch.starts_with("DebertaFor") => {
            let deberta = arch.starts_with("DebertaFor");
            let missing = |f: &str, e: String| {
                format!("{repo_id}: no tokenizer.json ({json_err}) or {f} ({e})")
            };
            let vocab = hub
                .fetch_file(repo_id, "vocab.json")
                .map_err(|e| missing("vocab.json", e))?;
            let merges = hub
                .fetch_file(repo_id, "merges.txt")
                .map_err(|e| missing("merges.txt", e))?;
            let bpe = tokenizers::models::bpe::BPE::from_file(
                &vocab.to_string_lossy(),
                &merges.to_string_lossy(),
            )
            .build()
            .map_err(|e| format!("{}: {e}", vocab.display()))?;
            let mut t = tokenizers::Tokenizer::new(bpe);
            let prefix = tok_cfg["add_prefix_space"].as_bool().unwrap_or(false);
            t.with_pre_tokenizer(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::new(prefix, true, true),
            ));
            let id = |s: &str| {
                t.token_to_id(s)
                    .ok_or_else(|| format!("{repo_id}: vocab.json has no {s}"))
            };
            let (cls_s, sep_s, specials) = if deberta {
                (
                    "[CLS]",
                    "[SEP]",
                    ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"],
                )
            } else {
                ("<s>", "</s>", ["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
            };
            let (cls, sep) = (id(cls_s)?, id(sep_s)?);
            if deberta {
                // DebertaConverter: `[CLS] A [SEP] B [SEP]` with segment ids 0/1.
                t.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
                    (sep_s.into(), sep),
                    (cls_s.into(), cls),
                )));
            } else {
                t.with_post_processor(Some(
                    tokenizers::processors::roberta::RobertaProcessing::new(
                        (sep_s.into(), sep),
                        (cls_s.into(), cls),
                    )
                    .add_prefix_space(prefix),
                ));
            }
            let specials: Vec<AddedToken> = specials
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect();
            t.add_special_tokens(&specials);
            t
        }
        Err(e) => return Err(format!("{repo_id}: no tokenizer.json ({e})")),
    };
    let max_length = tok_cfg["model_max_length"]
        .as_f64()
        .map_or(capacity, |m| (m as usize).min(capacity));
    tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length,
        ..Default::default()
    }))
    .map_err(|e| format!("tokenizer truncation: {e}"))?;
    tok.with_padding(None);
    Ok(tok)
}

/// A linear layer in PyTorch layout `[out_f, in_f]`.
struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    in_f: usize,
    out_f: usize,
}

impl Lin {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        nn::linear(x, rows, &self.w, self.b.as_deref(), self.in_f, self.out_f)
    }
}

/// How `config.json` (`problem_type`, label count) turns logits into scores.
fn activation_of(cfg: &Value, n: usize) -> Activation {
    match cfg["problem_type"].as_str() {
        Some("multi_label_classification") => Activation::Sigmoid,
        Some("regression") => Activation::Identity,
        Some("single_label_classification") => Activation::Softmax,
        _ if n == 1 => Activation::Identity,
        _ => Activation::Softmax,
    }
}





