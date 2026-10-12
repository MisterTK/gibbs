//! Hugging Face sequence classifiers (`text-classification` / `text-ranking` repos): BERT,
//! RoBERTa, XLM-RoBERTa, CamemBERT and DistilBERT on the System 1 BERT encoder graph (CPU or
//! Metal), ELECTRA and ALBERT (factorized embeddings, CPU), DeBERTa v1/v2/v3 with disentangled
//! relative attention (CPU), and Llama/Qwen2/GPT-NeoX (GraphLm body), Qwen3/Qwen3.5 (native text
//! decoder), Gemma 3/Gemma 4 and GPT-2/OPT (`ln_decoder`) decoders scored on their last non-pad
//! token (CPU). One forward pass per input; scores follow `config.json` (`problem_type`,
//! `id2label`).

mod factorized;
pub mod ln_decoder;
pub(crate) mod decoder_gpu;
pub use decoder_gpu::gpu_fixture;
mod spm;

use crate::hardware::{Device, QuantFormat, QuantMode};
use crate::model::hf_loader::ChatMessage;
use crate::model::hub::{is_synthetic_model_id, split_weight_file, HuggingFaceHub};
use crate::model::jev::{BertDims, BertEncoder};
use crate::model::layers::QuantizedLinearLayer;
use crate::model::loaders::{read_checkpoint, SafetensorsWeightReader, WeightReader};
use crate::model::nn;
use crate::model::graph_lm::GraphLm;
use crate::model::{ModelManifest, UniversalModelLoader};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

mod setfit;

mod gguf;
pub use gguf::GgufClassifier;

mod modernbert;

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
    /// BERT-layout encoder: hidden state of token 0 → head.
    Encoder {
        enc: BertEncoder,
        tok: tokenizers::Tokenizer,
        /// RoBERTa/XLM-R: positions count from `pad_token_id + 1`, pad tokens sit at `pad`.
        roberta_pad: Option<u32>,
        head: Head,
    },
    Deberta {
        model: Deberta,
        tok: tokenizers::Tokenizer,
        head: Head,
    },
    ModernBert {
        model: modernbert::ModernBert,
        tok: tokenizers::Tokenizer,
    },
    /// ELECTRA / ALBERT: hidden state of token 0 → head.
    Factorized {
        model: factorized::Factorized,
        tok: tokenizers::Tokenizer,
        head: Head,
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
    /// fastText supervised model: `predict` already returns probabilities.
    FastText(crate::model::fasttext::FastText),
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

/// `model_type`s the LLM loaders run (GraphLm by tensor layout, plus the native models).
const LLM_MODEL_TYPES: &[&str] = &[
    "llama", "mistral", "qwen2", "qwen3", "gpt2", "gpt_neo", "gpt_neox",
];

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
            let runs = model_type.is_some_and(|t| {
                LLM_MODEL_TYPES.contains(&t)
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
            crate::graph::avenues::family_gate(crate::graph::avenues::Family::ClassifierBert, device)?;
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
        // A PEFT adapter reads the base model's weights (and its tokenizer unless it ships one).
        let (weights_repo, tok_repo) = match &peft {
            Some((base, _)) => {
                let own = hub.fetch_file(repo_id, "tokenizer.json").is_ok();
                (base.as_str(), if own { repo_id } else { base.as_str() })
            }
            None => (repo_id, repo_id),
        };
        let cpu_only = |family: crate::graph::avenues::Family| {
            crate::graph::avenues::family_gate(family, device)
        };
        let usize_of = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("{repo_id}: config.json has no {k}"))
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
                "Gemma4ForSequenceClassification" => Box::new(
                    crate::model::gemma4::seqcls::Gemma4Seq::load(&artifacts, &cfg)?,
                ),
                "Gemma3TextForSequenceClassification" => Box::new(
                    crate::model::gemma3_text::Gemma3Text::load(&artifacts, &cfg)?,
                ),
                "Qwen3ForSequenceClassification"
                | "Qwen3_5ForSequenceClassification"
                | "Qwen3_5TextForSequenceClassification" => {
                    crate::model::qwen3_5::backbone(&artifacts, &cfg)?
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
            let mut ckpt = Checkpoint::open(hub, weights_repo)?;
            if let Some((_, adapter)) = &peft {
                ckpt = ckpt.with_lora(hub, repo_id, adapter)?;
            }
            let max_positions = usize_of("max_position_embeddings")?;
            let roberta_pad = (arch.contains("Roberta") || arch.starts_with("Camembert"))
                .then(|| {
                    pad_id.ok_or_else(|| format!("{repo_id}: config.json has no pad_token_id"))
                })
                .transpose()?;
            // RoBERTa positions start at pad + 1, so pad + 1 table rows are never a token's.
            let capacity = max_positions - roberta_pad.map_or(0, |p| p as usize + 1);
            let tok = load_tokenizer(hub, tok_repo, &arch, &tok_cfg, capacity)?;
            let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32;
            if arch == "ModernBertForSequenceClassification" {
                cpu_only(crate::graph::avenues::Family::ClassifierModern)?;
                let model = modernbert::ModernBert::load(&cfg, &ckpt)?;
                Body::ModernBert { model, tok }
            } else if arch.starts_with("Electra") || arch.starts_with("Albert") {
                cpu_only(crate::graph::avenues::Family::ClassifierModern)?;
                let albert = arch.starts_with("Albert");
                let model = factorized::Factorized::load(&cfg, &ckpt, albert)
                    .map_err(|e| format!("{repo_id}: {e}"))?;
                let head = if albert {
                    Head::load(
                        &ckpt,
                        Some(("albert.pooler", Act::Tanh)),
                        "classifier",
                        model.hidden,
                    )?
                } else {
                    // ElectraClassificationHead: GELU where BERT has tanh.
                    Head::load(
                        &ckpt,
                        Some(("classifier.dense", Act::Gelu)),
                        "classifier.out_proj",
                        model.hidden,
                    )?
                };
                Body::Factorized { model, tok, head }
            } else if arch.starts_with("Deberta") {
                cpu_only(crate::graph::avenues::Family::ClassifierModern)?;
                let model = Deberta::load(&cfg, &ckpt, !arch.starts_with("DebertaV2"))?;
                let act = match cfg["pooler_hidden_act"].as_str().unwrap_or("gelu") {
                    "gelu" => Act::Gelu,
                    other => {
                        return Err(format!(
                            "{repo_id}: pooler_hidden_act {other} is not supported"
                        ))
                    }
                };
                let head = Head::load(
                    &ckpt,
                    Some(("pooler.dense", act)),
                    "classifier",
                    model.hidden,
                )?;
                Body::Deberta { model, tok, head }
            } else {
                let (prefix, head) = match arch.as_str() {
                    "BertForSequenceClassification" => (
                        "bert",
                        Head::load(
                            &ckpt,
                            Some(("bert.pooler.dense", Act::Tanh)),
                            "classifier",
                            usize_of("hidden_size")?,
                        )?,
                    ),
                    "DistilBertForSequenceClassification" => (
                        "distilbert",
                        Head::load(
                            &ckpt,
                            Some(("pre_classifier", Act::Relu)),
                            "classifier",
                            usize_of("dim")?,
                        )?,
                    ),
                    _ => (
                        "roberta",
                        Head::load(
                            &ckpt,
                            Some(("classifier.dense", Act::Tanh)),
                            "classifier.out_proj",
                            usize_of("hidden_size")?,
                        )?,
                    ),
                };
                let dims = if prefix == "distilbert" {
                    if cfg["sinusoidal_pos_embds"].as_bool() == Some(true)
                        || cfg["activation"].as_str().unwrap_or("gelu") != "gelu"
                    {
                        return Err(format!(
                            "{repo_id}: only DistilBERT with learned positions and GELU is supported"
                        ));
                    }
                    BertDims {
                        hidden_size: usize_of("dim")?,
                        num_layers: usize_of("n_layers")?,
                        num_heads: usize_of("n_heads")?,
                        intermediate_size: usize_of("hidden_dim")?,
                        max_positions,
                        vocab_size: usize_of("vocab_size")?,
                        type_vocab_size: 1,
                        layer_norm_eps: 1e-12,
                    }
                } else {
                    if cfg["hidden_act"].as_str().unwrap_or("gelu") != "gelu"
                        || cfg["position_embedding_type"]
                            .as_str()
                            .unwrap_or("absolute")
                            != "absolute"
                    {
                        return Err(format!(
                            "{repo_id}: only GELU encoders with absolute positions are supported"
                        ));
                    }
                    BertDims {
                        hidden_size: usize_of("hidden_size")?,
                        num_layers: usize_of("num_hidden_layers")?,
                        num_heads: usize_of("num_attention_heads")?,
                        intermediate_size: usize_of("intermediate_size")?,
                        max_positions,
                        vocab_size: usize_of("vocab_size")?,
                        type_vocab_size: cfg["type_vocab_size"].as_u64().unwrap_or(2) as usize,
                        layer_norm_eps: eps,
                    }
                };
                let h = dims.hidden_size;
                cpu_only(crate::graph::avenues::Family::ClassifierBert)?;
                let enc = BertEncoder::load(dims, device, |name, shape| {
                    // DistilBERT has no segment embeddings: one all-zero row keeps the BERT graph.
                    if prefix == "distilbert" && name == "embeddings.token_type_embeddings.weight" {
                        return Ok(vec![0.0; h]);
                    }
                    let src = if prefix == "distilbert" {
                        distilbert_name(name)
                    } else {
                        name.to_string()
                    };
                    ckpt.get(&format!("{prefix}.{src}"), shape)
                })?;
                Body::Encoder {
                    enc,
                    tok,
                    roberta_pad,
                    head,
                }
            }
        };

        let n = match &body {
            Body::Encoder { head, .. }
            | Body::Deberta { head, .. }
            | Body::Factorized { head, .. } => head.out.out_f,
            Body::Decoder { score, .. } => score.out_f,
            Body::ModernBert { model, .. } => model.classifier.out_f,
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
        let activation = match cfg["problem_type"].as_str() {
            Some("multi_label_classification") => Activation::Sigmoid,
            Some("regression") => Activation::Identity,
            Some("single_label_classification") => Activation::Softmax,
            _ if n == 1 => Activation::Identity,
            _ => Activation::Softmax,
        };
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
        crate::graph::avenues::gate(crate::graph::avenues::Family::ClassifierModern)?;
        let ft = if path.is_dir() {
            crate::model::fasttext::FastText::load_export(path)?
        } else {
            crate::model::fasttext::FastText::load(path)?
        };
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
        crate::graph::avenues::family_gate(crate::graph::avenues::Family::ClassifierBert, device)?;
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
            Body::Encoder { enc, .. } => enc.device(),
            Body::SetFit(s) => s.device(),
            Body::Gguf(g) => g.device(),
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
            Body::Encoder {
                enc,
                tok,
                roberta_pad,
                head,
            } => {
                let e = encode(tok, text, pair)?;
                let ids = e.get_ids();
                let positions: Vec<u32> = match roberta_pad {
                    Some(pad) => {
                        let mut seen = 0;
                        ids.iter()
                            .map(|&t| {
                                if t == *pad {
                                    *pad
                                } else {
                                    seen += 1;
                                    pad + seen
                                }
                            })
                            .collect()
                    }
                    None => (0..ids.len() as u32).collect(),
                };
                let types = if enc.dims.type_vocab_size == 1 {
                    vec![0; ids.len()]
                } else {
                    e.get_type_ids().to_vec()
                };
                let hidden = enc.run(ids, &positions, &types, false)?;
                (head.apply(&hidden[..enc.dims.hidden_size]), ids.len())
            }
            Body::Deberta { model, tok, head } => {
                let e = encode(tok, text, pair)?;
                let hidden = model.forward(e.get_ids())?;
                (head.apply(&hidden[..model.hidden]), e.get_ids().len())
            }
            Body::Gguf(g) => g.logits(text, pair)?,
            Body::FastText(_) if pair.is_some() => {
                return Err("fastText classifiers take a single text, not a pair".into())
            }
            Body::FastText(ft) => ft.predict(text)?,
            Body::ModernBert { model, tok } => {
                let e = encode(tok, text, pair)?;
                (model.forward(e.get_ids())?, e.get_ids().len())
            }
            Body::Factorized { model, tok, head } => {
                let e = encode(tok, text, pair)?;
                let hidden = model.forward(e.get_ids(), e.get_type_ids())?;
                (head.apply(&hidden[..model.hidden]), e.get_ids().len())
            }
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

/// The repo's weights: `model.safetensors`, else `pytorch_model.bin`; for a PEFT adapter, the
/// base model's weights with the LoRA deltas merged as tensors are read.
struct Checkpoint {
    manifest: ModelManifest,
    data: Vec<u8>,
    lora: Option<Lora>,
}

/// A PEFT LoRA adapter: `adapter_model.*` tensors under `base_model.model.<task model name>`
/// (`lora_A`/`lora_B` per target module, whole `modules_to_save` tensors such as the classifier).
struct Lora {
    manifest: ModelManifest,
    data: Vec<u8>,
    /// `lora_alpha / r` (`/ sqrt(r)` with rsLoRA).
    scale: f32,
}

impl Checkpoint {
    fn open(hub: &HuggingFaceHub, repo_id: &str) -> Result<Self, String> {
        let path = hub.fetch_file(repo_id, "model.safetensors").or_else(|st| {
            hub.fetch_file(repo_id, "pytorch_model.bin").map_err(|bin| {
                format!("{repo_id}: no model.safetensors ({st}) or pytorch_model.bin ({bin})")
            })
        })?;
        let (manifest, data) = read_checkpoint(&path)?;
        Ok(Self {
            manifest,
            data,
            lora: None,
        })
    }

    /// Merges the LoRA adapter of `repo_id` (its `adapter_config.json` is `cfg`) into these base
    /// weights at read time: `W + scale · B·A`.
    fn with_lora(
        mut self,
        hub: &HuggingFaceHub,
        repo_id: &str,
        cfg: &Value,
    ) -> Result<Self, String> {
        let path = hub
            .fetch_file(repo_id, "adapter_model.safetensors")
            .or_else(|_| hub.fetch_file(repo_id, "adapter_model.bin"))?;
        let (manifest, data) = read_checkpoint(&path)?;
        let r = cfg["r"].as_f64().ok_or("adapter_config.json has no r")?;
        let alpha = cfg["lora_alpha"].as_f64().unwrap_or(r);
        let scale = if cfg["use_rslora"] == true {
            alpha / r.sqrt()
        } else {
            alpha / r
        };
        self.lora = Some(Lora {
            manifest,
            data,
            scale: scale as f32,
        });
        Ok(self)
    }

    /// Where tensor `name` (its name in the task model) lives: the adapter's saved modules, then
    /// the weights; an adapter's base checkpoint may lack the task model prefix (`bert.`).
    fn find(&self, name: &str) -> Option<(&ModelManifest, &[u8], String)> {
        if let Some(l) = &self.lora {
            let saved = format!("base_model.model.{name}");
            if l.manifest.tensors.contains_key(&saved) {
                return Some((&l.manifest, &l.data, saved));
            }
        }
        if self.manifest.tensors.contains_key(name) {
            return Some((&self.manifest, &self.data, name.to_string()));
        }
        let short = name.split_once('.')?.1;
        (self.lora.is_some() && self.manifest.tensors.contains_key(short))
            .then(|| (&self.manifest, &self.data[..], short.to_string()))
    }

    fn shape(&self, name: &str) -> Option<&[usize]> {
        let (m, _, key) = self.find(name)?;
        m.tensors.get(&key).map(|t| t.shape.as_slice())
    }

    fn has(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// Tensor `name`, which must have `shape` (with its LoRA delta merged).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (m, data, key) = self
            .find(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let got = &m.tensors[&key].shape;
        if got != shape {
            return Err(format!(
                "tensor {name} has shape {got:?}, expected {shape:?}"
            ));
        }
        let mut w = UniversalModelLoader::extract_tensor_from_manifest(m, data, &key)?;
        let (Some(l), Some(module), &[out_f, in_f]) =
            (&self.lora, name.strip_suffix(".weight"), shape)
        else {
            return Ok(w);
        };
        let a_name = format!("base_model.model.{module}.lora_A.weight");
        let Some(a_shape) = l.manifest.tensors.get(&a_name).map(|t| t.shape.clone()) else {
            return Ok(w);
        };
        let b_name = format!("base_model.model.{module}.lora_B.weight");
        let rank = a_shape[0];
        if a_shape != [rank, in_f]
            || l.manifest.tensors.get(&b_name).map(|t| &t.shape[..]) != Some(&[out_f, rank][..])
        {
            return Err(format!(
                "LoRA factors of {module} do not match [{out_f}, {in_f}]"
            ));
        }
        let a = UniversalModelLoader::extract_tensor_from_manifest(&l.manifest, &l.data, &a_name)?;
        let b = UniversalModelLoader::extract_tensor_from_manifest(&l.manifest, &l.data, &b_name)?;
        let mut delta = vec![0.0; out_f * in_f];
        nn::matmul_nt(
            &b,
            &nn::transpose(&a, rank, in_f),
            None,
            out_f,
            rank,
            in_f,
            &mut delta,
        );
        w.iter_mut().zip(delta).for_each(|(w, d)| *w += l.scale * d);
        Ok(w)
    }

    /// `name.weight` `[out, in_f]` and `name.bias` (when present) as a linear layer.
    fn linear(&self, name: &str, in_f: usize) -> Result<Lin, String> {
        let w = format!("{name}.weight");
        let out_f = match self.shape(&w) {
            Some(&[o, i]) if i == in_f => o,
            other => {
                return Err(format!(
                    "tensor {w} has shape {other:?}, expected [_, {in_f}]"
                ))
            }
        };
        let b = format!("{name}.bias");
        Ok(Lin {
            w: self.get(&w, &[out_f, in_f])?,
            b: if self.has(&b) {
                Some(self.get(&b, &[out_f])?)
            } else {
                None
            },
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

/// DistilBERT's name for a BERT-layout encoder tensor (without the `distilbert.` prefix).
fn distilbert_name(bert: &str) -> String {
    const MAP: [(&str, &str); 8] = [
        ("attention.self.query", "attention.q_lin"),
        ("attention.self.key", "attention.k_lin"),
        ("attention.self.value", "attention.v_lin"),
        ("attention.output.dense", "attention.out_lin"),
        ("attention.output.LayerNorm", "sa_layer_norm"),
        ("intermediate.dense", "ffn.lin1"),
        ("output.dense", "ffn.lin2"),
        ("output.LayerNorm", "output_layer_norm"),
    ];
    let Some((layer, rest)) = bert
        .strip_prefix("encoder.layer.")
        .and_then(|r| r.split_once('.'))
    else {
        return bert.to_string();
    };
    let (module, param) = rest.rsplit_once('.').unwrap_or((rest, ""));
    let module = MAP
        .iter()
        .find(|(b, _)| *b == module)
        .map_or(module, |(_, d)| d);
    format!("transformer.layer.{layer}.{module}.{param}")
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

#[derive(Clone, Copy)]
enum Act {
    Tanh,
    Relu,
    Gelu,
}

/// Classification head on the first token's hidden state: optional dense + activation, then
/// the output projection.
struct Head {
    dense: Option<(Lin, Act)>,
    out: Lin,
}

impl Head {
    fn load(
        ckpt: &Checkpoint,
        dense: Option<(&str, Act)>,
        out: &str,
        h: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            dense: dense
                .map(|(name, act)| Ok::<_, String>((ckpt.linear(name, h)?, act)))
                .transpose()?,
            out: ckpt.linear(out, h)?,
        })
    }

    fn apply(&self, first: &[f32]) -> Vec<f32> {
        let x = match &self.dense {
            Some((lin, act)) => {
                let mut y = lin.apply(first, 1);
                match act {
                    Act::Tanh => y.iter_mut().for_each(|v| *v = v.tanh()),
                    Act::Relu => y.iter_mut().for_each(|v| *v = v.max(0.0)),
                    Act::Gelu => nn::gelu(&mut y),
                }
                y
            }
            None => first.to_vec(),
        };
        self.out.apply(&x, 1)
    }
}

struct DebertaLayer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln1: (Vec<f32>, Vec<f32>),
    inter: Lin,
    out: Lin,
    ln2: (Vec<f32>, Vec<f32>),
    /// Per head `[2 * span, head_dim]`: query/key projections of the relative embeddings
    /// (v2: the shared query/key layers on normalised embeddings; v1: `pos_q_proj`/`pos_proj`),
    /// fixed at load.
    pos_q: Vec<Vec<f32>>,
    pos_k: Vec<Vec<f32>>,
}

/// DeBERTa encoder (transformers `DebertaModel` v1 and `DebertaV2Model` v2/v3): word (+ absolute
/// position) embeddings + LayerNorm, post-LN layers whose attention adds content→position and
/// position→content terms over (log-bucketed for v2) relative positions.
struct Deberta {
    hidden: usize,
    heads: usize,
    eps: f32,
    /// Attention span: relative positions are clamped to `[-span, span)`.
    span: usize,
    /// `position_buckets` (> 0 enables log buckets) and `max_relative_positions`.
    buckets: i64,
    max_relative: i64,
    word: Vec<f32>,
    /// v1 `position_biased_input`: absolute position embeddings `[max_positions, hidden]`.
    pos: Option<Vec<f32>>,
    emb_ln: (Vec<f32>, Vec<f32>),
    layers: Vec<DebertaLayer>,
}

impl Deberta {
    /// `v1`: `DebertaModel` (`in_proj` with per-head interleaved q/k/v + `q_bias`/`v_bias`,
    /// separate `pos_proj`/`pos_q_proj`, `|`-separated `pos_att_type`).
    fn load(cfg: &Value, ckpt: &Checkpoint, v1: bool) -> Result<Self, String> {
        let num = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {k}"))
        };
        let (h, heads, inter, layers) = (
            num("hidden_size")?,
            num("num_attention_heads")?,
            num("intermediate_size")?,
            num("num_hidden_layers")?,
        );
        let pos_att: Vec<String> = match &cfg["pos_att_type"] {
            Value::String(s) => s.split('|').map(|t| t.trim().to_lowercase()).collect(),
            v => v
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_lowercase)
                        .collect()
                })
                .unwrap_or_default(),
        };
        let has = |t: &str| pos_att.iter().any(|p| p == t);
        let biased_input = cfg["position_biased_input"].as_bool().unwrap_or(true);
        let supported = cfg["relative_attention"].as_bool() == Some(true)
            && has("c2p")
            && has("p2c")
            // v1 scales by sqrt(head_dim * (1 + len(pos_att_type))).
            && (!v1 || pos_att.len() == 2)
            && (v1 || (cfg["share_att_key"].as_bool() == Some(true) && !biased_input))
            && cfg["talking_head"].as_bool() != Some(true)
            && cfg["z_steps"].as_u64().unwrap_or(0) <= 1
            && cfg["conv_kernel_size"].as_u64().unwrap_or(0) == 0
            && cfg["type_vocab_size"].as_u64().unwrap_or(0) == 0
            && cfg["embedding_size"]
                .as_u64()
                .is_none_or(|e| e as usize == h)
            && cfg["hidden_act"].as_str().unwrap_or("gelu") == "gelu";
        if !supported || heads == 0 || h % heads != 0 {
            return Err("only DeBERTa configs with relative attention over c2p+p2c are supported (v2/v3: shared keys and no absolute positions; no talking heads, z_steps, conv layer, token types or embedding projection; GELU)".into());
        }
        let mut max_relative = cfg["max_relative_positions"].as_i64().unwrap_or(-1);
        if max_relative < 1 {
            max_relative = num("max_position_embeddings")? as i64;
        }
        let buckets = cfg["position_buckets"].as_i64().unwrap_or(-1);
        let span = if buckets > 0 { buckets } else { max_relative } as usize;
        let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-7) as f32;
        let p = "deberta.encoder";
        let mut rel = ckpt.get(&format!("{p}.rel_embeddings.weight"), &[2 * span, h])?;
        let norm_rel = cfg["norm_rel_ebd"].as_str().unwrap_or("none");
        if norm_rel.split('|').any(|s| s.trim() == "layer_norm") {
            let (g, b) = ckpt.layer_norm(&format!("{p}.LayerNorm"), h)?;
            rel = nn::layer_norm(&rel, 2 * span, h, &g, &b, eps);
        }
        let d = h / heads;
        let split_heads = |x: Vec<f32>| -> Vec<Vec<f32>> {
            (0..heads)
                .map(|hd| {
                    x.chunks_exact(h)
                        .flat_map(|row| &row[hd * d..(hd + 1) * d])
                        .copied()
                        .collect()
                })
                .collect()
        };
        let layers = (0..layers)
            .map(|l| {
                let p = format!("{p}.layer.{l}");
                let lin = |s: &str, in_f| ckpt.linear(&format!("{p}.{s}"), in_f);
                let (q, k, v) = if v1 {
                    let w = ckpt.get(&format!("{p}.attention.self.in_proj.weight"), &[3 * h, h])?;
                    // transpose_for_scores(in_proj(x)).chunk(3): head hd's rows are q, k, v.
                    let mut qkv = [vec![0.0; h * h], vec![0.0; h * h], vec![0.0; h * h]];
                    for (r, row) in w.chunks_exact(h).enumerate() {
                        let o = (r / (3 * d) * d + r % d) * h;
                        qkv[r % (3 * d) / d][o..o + h].copy_from_slice(row);
                    }
                    let [wq, wk, wv] = qkv;
                    let bias = |s: &str| ckpt.get(&format!("{p}.attention.self.{s}"), &[h]);
                    let full = |w, b| Lin {
                        w,
                        b,
                        in_f: h,
                        out_f: h,
                    };
                    (
                        full(wq, Some(bias("q_bias")?)),
                        full(wk, None),
                        full(wv, Some(bias("v_bias")?)),
                    )
                } else {
                    (
                        lin("attention.self.query_proj", h)?,
                        lin("attention.self.key_proj", h)?,
                        lin("attention.self.value_proj", h)?,
                    )
                };
                let (pos_q, pos_k) = if v1 {
                    (
                        lin("attention.self.pos_q_proj", h)?.apply(&rel, 2 * span),
                        lin("attention.self.pos_proj", h)?.apply(&rel, 2 * span),
                    )
                } else {
                    (q.apply(&rel, 2 * span), k.apply(&rel, 2 * span))
                };
                let (pos_q, pos_k) = (split_heads(pos_q), split_heads(pos_k));
                Ok(DebertaLayer {
                    q,
                    k,
                    v,
                    o: lin("attention.output.dense", h)?,
                    ln1: ckpt.layer_norm(&format!("{p}.attention.output.LayerNorm"), h)?,
                    inter: lin("intermediate.dense", h)?,
                    out: lin("output.dense", inter)?,
                    ln2: ckpt.layer_norm(&format!("{p}.output.LayerNorm"), h)?,
                    pos_q,
                    pos_k,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            hidden: h,
            heads,
            eps,
            span,
            buckets,
            max_relative,
            word: ckpt.get(
                "deberta.embeddings.word_embeddings.weight",
                &[num("vocab_size")?, h],
            )?,
            pos: biased_input
                .then(|| {
                    ckpt.get(
                        "deberta.embeddings.position_embeddings.weight",
                        &[num("max_position_embeddings")?, h],
                    )
                })
                .transpose()?,
            emb_ln: ckpt.layer_norm("deberta.embeddings.LayerNorm", h)?,
            layers,
        })
    }

    /// Index into the `2 * span` relative embeddings for query `i`, key `j`
    /// (`make_log_bucket_position`, then shifted by `span` and clamped).
    fn rel_index(&self, i: usize, j: usize) -> usize {
        let rel = i as i64 - j as i64;
        let bucket = if self.buckets > 0 {
            let mid = self.buckets / 2;
            if rel.abs() <= mid {
                rel
            } else {
                // f32 like the reference's float tensors.
                let log_pos = ((rel.abs() as f32 / mid as f32).ln()
                    / ((self.max_relative - 1) as f32 / mid as f32).ln()
                    * (mid - 1) as f32)
                    .ceil()
                    + mid as f32;
                rel.signum() * log_pos as i64
            }
        } else {
            rel
        };
        (bucket + self.span as i64).clamp(0, 2 * self.span as i64 - 1) as usize
    }

    /// Last hidden states `[n, hidden]` of one unpadded sequence.
    fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let (n, h, heads) = (ids.len(), self.hidden, self.heads);
        let d = h / heads;
        let rows = self.word.len() / h;
        let mut x = Vec::with_capacity(n * h);
        for &id in ids {
            let id = id as usize;
            if id >= rows {
                return Err(format!(
                    "token id {id} is outside the {rows}-row vocabulary"
                ));
            }
            x.extend_from_slice(&self.word[id * h..(id + 1) * h]);
        }
        if let Some(pos) = &self.pos {
            if n * h > pos.len() {
                return Err(format!("{n} tokens exceed {} positions", pos.len() / h));
            }
            nn::add_in_place(&mut x, &pos[..n * h]);
        }
        x = nn::layer_norm(&x, n, h, &self.emb_ln.0, &self.emb_ln.1, self.eps);
        let idx: Vec<usize> = (0..n * n).map(|t| self.rel_index(t / n, t % n)).collect();
        let span2 = 2 * self.span;
        let scale = 1.0 / ((d * 3) as f32).sqrt();
        let head_of = |m: &[f32], hd: usize| -> Vec<f32> {
            m.chunks_exact(h)
                .flat_map(|row| &row[hd * d..(hd + 1) * d])
                .copied()
                .collect()
        };
        for layer in &self.layers {
            let (q, k, v) = (
                layer.q.apply(&x, n),
                layer.k.apply(&x, n),
                layer.v.apply(&x, n),
            );
            let mut ctx = vec![0.0; n * h];
            for hd in 0..heads {
                let (qh, kh) = (head_of(&q, hd), head_of(&k, hd));
                let vt = nn::transpose(&head_of(&v, hd), n, d);
                let (mut qk, mut c2p, mut p2c) =
                    (vec![0.0; n * n], vec![0.0; n * span2], vec![0.0; n * span2]);
                nn::matmul_nt(&qh, &kh, None, n, d, n, &mut qk);
                nn::matmul_nt(&qh, &layer.pos_k[hd], None, n, d, span2, &mut c2p);
                nn::matmul_nt(&kh, &layer.pos_q[hd], None, n, d, span2, &mut p2c);
                for i in 0..n {
                    let row = &mut qk[i * n..(i + 1) * n];
                    for (j, s) in row.iter_mut().enumerate() {
                        let r = idx[i * n + j];
                        *s = (*s + c2p[i * span2 + r] + p2c[j * span2 + r]) * scale;
                    }
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for s in row.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    row.iter_mut().for_each(|s| *s /= sum);
                }
                let mut o = vec![0.0; n * d];
                nn::matmul_nt(&qk, &vt, None, n, n, d, &mut o);
                for (i, row) in o.chunks_exact(d).enumerate() {
                    ctx[i * h + hd * d..i * h + (hd + 1) * d].copy_from_slice(row);
                }
            }
            let mut a = layer.o.apply(&ctx, n);
            nn::add_in_place(&mut a, &x);
            let a = nn::layer_norm(&a, n, h, &layer.ln1.0, &layer.ln1.1, self.eps);
            let mut i = layer.inter.apply(&a, n);
            nn::gelu(&mut i);
            let mut o = layer.out.apply(&i, n);
            nn::add_in_place(&mut o, &a);
            x = nn::layer_norm(&o, n, h, &layer.ln2.0, &layer.ln2.1, self.eps);
        }
        Ok(x)
    }
}
