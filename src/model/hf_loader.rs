use crate::model::nano::{NanoTransformerConfig, NanoTransformerModel};
use crate::model::graph_lm::{GraphLm, GraphLmConfig};
use crate::model::tokenizer::Tokenizer;
use crate::model::UniversalModelLoader;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct HfConfigJson {
    #[serde(alias = "hidden_size", alias = "d_model", default = "default_hidden")]
    pub hidden_dim: usize,
    #[serde(
        alias = "num_hidden_layers",
        alias = "n_layers",
        default = "default_layers"
    )]
    pub num_layers: usize,
    #[serde(
        alias = "num_attention_heads",
        alias = "n_heads",
        default = "default_heads"
    )]
    pub num_heads: usize,
    #[serde(alias = "intermediate_size", default)]
    pub intermediate_dim: Option<usize>,
    #[serde(alias = "vocab_size", default = "default_vocab")]
    pub vocab_size: usize,
    #[serde(default)]
    pub model_type: Option<String>,
    // GPT-NeoX configs may carry both spellings; separate fields so they still parse.
    #[serde(default)]
    pub rotary_pct: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub use_parallel_residual: Option<bool>,
    #[serde(alias = "num_key_value_heads", alias = "n_kv_heads", default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(alias = "hidden_act", alias = "activation_function", default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// transformers v5 nests `rope_theta` here (no top-level key).
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    // Separate fields (not aliases): a config carrying two spellings must not fail to parse.
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    #[serde(default)]
    pub layer_norm_eps: Option<f32>,
    #[serde(default)]
    pub layer_norm_epsilon: Option<f32>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    /// 1bitLLM BitNet b1.58 checkpoints (`model_type: llama`): `weight_bits: 1`, `input_bits: 8`.
    #[serde(default)]
    pub weight_bits: Option<u32>,
    #[serde(default)]
    pub input_bits: Option<u32>,
}

impl HfConfigJson {
    pub fn resolved_intermediate_dim(&self) -> usize {
        self.intermediate_dim.unwrap_or(self.hidden_dim * 4)
    }

    pub fn resolved_rope_theta(&self) -> f32 {
        self.rope_theta
            .or_else(|| Some(self.rope_parameters.as_ref()?["rope_theta"].as_f64()? as f32))
            .unwrap_or(10000.0)
    }

    /// RMSNorm / LayerNorm epsilon (`rms_norm_eps`, `layer_norm_eps`, `layer_norm_epsilon`).
    pub fn resolved_norm_eps(&self) -> f32 {
        self.rms_norm_eps
            .or(self.layer_norm_eps)
            .or(self.layer_norm_epsilon)
            .unwrap_or(1e-5)
    }

    /// BitNet b1.58 BitLinear projections as (activation bits, ternarize weights at load):
    /// transformers `quant_method: "bitnet"` (int8 activations; `quantization_mode: "online"` holds
    /// master weights), or 1bitLLM's `weight_bits: 1` master weights with `input_bits` activations.
    pub fn bitnet(&self) -> Option<(u32, bool)> {
        if let Some(q) = self
            .quantization_config
            .as_ref()
            .filter(|q| q["quant_method"] == "bitnet")
        {
            return Some((8, q["quantization_mode"] == "online"));
        }
        (self.weight_bits == Some(1)).then(|| (self.input_bits.unwrap_or(8), true))
    }
    pub fn is_llama_or_swiglu(&self) -> bool {
        if let Some(m_type) = &self.model_type {
            if m_type == "llama" || m_type == "qwen2" || m_type == "mistral" {
                return true;
            }
        }
        for arch in &self.architectures {
            if arch.contains("Llama") || arch.contains("Qwen") || arch.contains("Mistral") {
                return true;
            }
        }
        false
    }

    pub fn resolved_num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_heads)
    }

    pub fn resolved_rotary_pct(&self) -> f32 {
        if let Some(pct) = self.rotary_pct.or(self.partial_rotary_factor) {
            pct
        } else if let Some(m_type) = &self.model_type {
            if m_type == "gpt_neox" {
                0.25
            } else if m_type == "glm" {
                0.5 // HF GlmConfig default partial_rotary_factor
            } else if m_type == "gpt_neo" || m_type == "gpt2" || m_type == "bert" {
                0.0
            } else {
                1.0
            }
        } else if self.architectures.iter().any(|a| a.contains("GPTNeoX")) {
            0.25
        } else if self
            .architectures
            .iter()
            .any(|a| a.contains("GPTNeo") || a.contains("GPT2") || a.contains("Bert"))
        {
            0.0
        } else {
            1.0
        }
    }

    pub fn resolved_parallel_residual(&self) -> bool {
        if let Some(par) = self.use_parallel_residual {
            par
        } else if let Some(m_type) = &self.model_type {
            m_type == "gpt_neox"
        } else {
            self.architectures.iter().any(|a| a.contains("GPTNeoX"))
        }
    }

    pub fn resolved_use_standard_layernorm(&self) -> bool {
        if self.is_llama_or_swiglu() {
            return false;
        }
        if let Some(m_type) = &self.model_type {
            m_type == "gpt_neo" || m_type == "gpt2" || m_type == "gpt_neox" || m_type == "bert"
        } else {
            self.architectures.iter().any(|a| {
                a.contains("GPTNeo")
                    || a.contains("GPT2")
                    || a.contains("GPTNeoX")
                    || a.contains("Bert")
            })
        }
    }

    pub fn resolved_use_gelu_mlp(&self) -> bool {
        if self.is_llama_or_swiglu() {
            return false;
        }
        if let Some(act) = &self.hidden_act {
            if act == "silu" {
                return false;
            }
            if act.starts_with("gelu") {
                return true;
            }
        }
        if let Some(m_type) = &self.model_type {
            m_type == "gpt_neo" || m_type == "gpt2" || m_type == "gpt_neox" || m_type == "bert"
        } else {
            self.architectures.iter().any(|a| {
                a.contains("GPTNeo")
                    || a.contains("GPT2")
                    || a.contains("GPTNeoX")
                    || a.contains("Bert")
            })
        }
    }
}

/// Reads `tokenizer.json` with the chat template and bos/eos strings beside it
/// (`tokenizer_config.json`, `chat_template.jinja`; both optional). With no template in either,
/// the `chat_template` field of a processor's `chat_template.json` (Qwen3-VL, SmolVLM) is used.
pub fn load_json_tokenizer(p: &Path) -> Result<Tokenizer, String> {
    let json = fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    tokenizer_from_files(&json, &|n| fs::read(p.with_file_name(n)).ok())
        .map_err(|e| format!("{}: {e}", p.display()))
}

/// [`load_json_tokenizer`] for a `tokenizer.json` text whose sibling files `sibling` serves by
/// name (a directory, or the resources inside a NInfer artifact).
pub fn tokenizer_from_files(
    json: &str,
    sibling: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<Tokenizer, String> {
    let mut tok = Tokenizer::from_json_str(json)?;
    let text = |n: &str| sibling(n).and_then(|b| String::from_utf8(b).ok());
    let cfg = text("tokenizer_config.json");
    let mut jinja = text("chat_template.jinja");
    let cfg_template = cfg
        .as_deref()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok())
        .is_some_and(|v| !v["chat_template"].is_null());
    if jinja.is_none() && !cfg_template {
        jinja = sibling("chat_template.json")
            .and_then(|j| serde_json::from_slice::<serde_json::Value>(&j).ok())
            .and_then(|v| v["chat_template"].as_str().map(str::to_string));
    }
    if let Some(g) = sibling("generation_config.json") {
        apply_generation_json(&mut tok, &g);
    }
    if cfg.is_some() || jinja.is_some() {
        tok.set_tokenizer_config(cfg.as_deref().unwrap_or("{}"), jinja)
            .map_err(|e| format!("tokenizer_config.json: {e}"))?;
    }
    Ok(tok)
}

/// Applies `generation_config.json`: its `eos_token_id`(s) end generation, and requests that leave
/// temperature/top_p unset get its sampling settings: greedy for `do_sample: false`, else its
/// temperature/top_p when it sets either (as vLLM does). A config silent on both (SmolLM2) keeps
/// the engine's defaults; encoder-decoders set greedy themselves. A missing file leaves
/// everything as is.
pub fn apply_generation_config(tok: &mut Tokenizer, path: &Path) {
    if let Ok(j) = fs::read(path) {
        apply_generation_json(tok, &j);
    }
}

/// [`apply_generation_config`] for the file's bytes.
fn apply_generation_json(tok: &mut Tokenizer, json: &[u8]) {
    let Some(g) = serde_json::from_slice::<serde_json::Value>(json).ok() else {
        return;
    };
    tok.stop_ids = match &g["eos_token_id"] {
        serde_json::Value::Number(n) => n.as_u64().map(|n| vec![n as u32]),
        serde_json::Value::Array(a) => a.iter().map(|v| Some(v.as_u64()? as u32)).collect(),
        _ => None,
    };
    let f = |k: &str| g[k].as_f64().map(|v| v as f32);
    let (t, p) = (f("temperature"), f("top_p"));
    tok.sampling_defaults = if g["do_sample"] == false {
        Some((0.0, 1.0))
    } else {
        (t.is_some() || p.is_some()).then(|| (t.unwrap_or(1.0), p.unwrap_or(1.0)))
    };
}

/// A checkpoint opened once so several [`GraphLm`]s (the source format and re-encoded variants) can
/// be built from the same bytes.
pub struct LmSource {
    cfg: GraphLmConfig,
    tokenizer: Tokenizer,
    label: String,
    body: LmBody,
}

enum LmBody {
    Checkpoint {
        manifest: crate::model::ModelManifest,
        data: crate::model::loaders::gguf::GgufData,
        gguf: bool,
    },
    /// Synthetic and ONNX models arrive built; there are no checkpoint tensors to carry.
    Built(GraphLm),
}

impl LmSource {
    /// The checkpoint's tensor index, `None` for synthetic and ONNX models.
    pub fn manifest(&self) -> Option<&crate::model::ModelManifest> {
        match &self.body {
            LmBody::Checkpoint { manifest, .. } => Some(manifest),
            LmBody::Built(_) => None,
        }
    }

    /// How many projections (4 per layer plus the LM head) a build passes to its encoder.
    pub fn projection_count(&self) -> usize {
        let layers = match &self.body {
            LmBody::Checkpoint { manifest, gguf: true, .. } => {
                GraphLmConfig::from_gguf(manifest, self.cfg.clone()).num_layers
            }
            LmBody::Checkpoint { .. } => self.cfg.num_layers,
            LmBody::Built(m) => m.layers.len(),
        };
        4 * layers + 1
    }

    /// Builds a model, passing every projection through `enc` when given.
    pub fn build(
        &self,
        enc: Option<&dyn crate::model::graph_lm::ProjectionEncoder>,
    ) -> Result<GraphLm, String> {
        let mut llm = match &self.body {
            LmBody::Checkpoint { manifest, data, gguf } => {
                let cfg = if *gguf {
                    GraphLmConfig::from_gguf(manifest, self.cfg.clone())
                } else {
                    self.cfg.clone()
                };
                GraphLm::from_manifest_with(cfg, manifest, data, enc)
                    .map_err(|e| format!("Failed to load {}: {e}", self.label))?
            }
            LmBody::Built(m) => {
                let mut m = m.clone();
                if let Some(e) = enc {
                    m.encode_projections(e);
                }
                m
            }
        };
        llm.set_tokenizer(self.tokenizer.clone());
        // Constructors already encode the weights in `cfg.quant_mode`; re-encoding would compound error.
        if llm.current_quant_mode() != self.cfg.quant_mode {
            llm.quantize(self.cfg.quant_mode);
        }
        Ok(llm)
    }
}

/// Loads an autoregressive LM from resolved artifacts, with the tokenizer that belongs to its weights.
/// Real repositories must provide weights and a tokenizer; only synthetic ids
/// (see [`crate::model::hub::is_synthetic_model_id`]) may run random-initialized weights.
pub fn load_causal_lm(
    artifacts: &crate::model::hub::ModelArtifacts,
    quant_mode: crate::hardware::QuantMode,
) -> Result<GraphLm, String> {
    open_causal_lm(artifacts, quant_mode)?.build(None)
}

/// Opens the checkpoint of `artifacts` without building a model.
pub fn open_causal_lm(
    artifacts: &crate::model::hub::ModelArtifacts,
    quant_mode: crate::hardware::QuantMode,
) -> Result<LmSource, String> {
    let id = artifacts.model_id.as_str();
    let preset = ["gibbs/", "foundry/"]
        .iter()
        .find_map(|p| id.strip_prefix(p))
        .unwrap_or("");
    let mut cfg = match preset {
        "smollm-1m" => GraphLmConfig::smollm_1m(),
        "tiny-5m" => GraphLmConfig::tiny_5m(),
        "tiny-10m" => GraphLmConfig::tiny_10m(),
        // GGUF without config.json: `from_gguf_data` derives dims from GGUF metadata.
        _ => match artifacts.config_path.as_deref() {
            Some(p) => {
                let bytes = fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
                let hf: HfConfigJson =
                    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", p.display()))?;
                GraphLmConfig::from_hf(&hf, quant_mode)
            }
            None => GraphLmConfig::default(),
        },
    };
    cfg.quant_mode = quant_mode;

    let json_tokenizer = artifacts
        .tokenizer_path
        .as_deref()
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .map(load_json_tokenizer)
        .transpose()?;

    use crate::model::loaders::gguf::GgufData;
    let (body, tokenizer, label) = match artifacts.weights_path.as_deref() {
        Some(p) => {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");

            // Sniff magic bytes before extension matching (legacy GGML models are often *.bin).
            let mut magic = [0u8; 8];
            let is_ggml = if let Ok(mut f) = fs::File::open(p) {
                use std::io::Read;
                if let Ok(n) = f.read(&mut magic) {
                    if n >= 4 {
                        let m = u32::from_le_bytes(magic[0..4].try_into().unwrap());
                        m == 0x67676d6c
                            || m == 0x67676d66
                            || m == 0x67676a74
                            || magic.starts_with(b"ggml")
                            || magic.starts_with(b"ggmf")
                            || magic.starts_with(b"ggjt")
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            let is_gguf = magic.starts_with(b"GGUF");
            let fail = |e: String| format!("Failed to load {}: {e}", p.display());

            // (checkpoint body, bytes the GGUF tokenizer lives in)
            let (body, gguf_tok) = if is_gguf || ext == "gguf" {
                let (manifest, data) = crate::model::loaders::gguf::open(p).map_err(fail)?;
                (LmBody::Checkpoint { manifest, data, gguf: true }, true)
            } else if is_ggml || matches!(ext, "ggml" | "ggmf" | "ggjt") {
                let (manifest, data) = crate::model::loaders::ggml::open(p).map_err(fail)?;
                (LmBody::Checkpoint { manifest, data: GgufData::Buf(data), gguf: true }, true)
            } else {
                match ext {
                    // safetensors / PyTorch / Flax / NumPy file or a sharded `<file>.index.json`; quant scheme vetted first.
                    "safetensors" | "json" | "bin" | "pt" | "pth" | "ckpt" | "msgpack" | "npz"
                    | "npy" => {
                        crate::model::prequant::check(
                            artifacts.config_path.as_deref(),
                            p.parent().unwrap_or(Path::new(".")),
                        )
                        .map_err(fail)?;
                        let (mut manifest, data) =
                            crate::model::loaders::read_checkpoint(p).map_err(fail)?;
                        crate::model::prequant::annotate(
                            &mut manifest,
                            artifacts.config_path.as_deref(),
                        )
                        .map_err(fail)?;
                        (LmBody::Checkpoint { manifest, data: GgufData::Buf(data), gguf: false }, false)
                    }
                    "onnx" => (
                        LmBody::Built(
                            crate::model::loaders::onnx::load_causal_lm(cfg.clone(), p).map_err(fail)?,
                        ),
                        false,
                    ),
                    other => return Err(fail(format!("unsupported weight format '.{other}'"))),
                }
            };
            let tok = match (json_tokenizer, &body) {
                (Some(t), _) => t,
                (None, LmBody::Checkpoint { data, .. }) if gguf_tok => {
                    UniversalModelLoader::extract_gguf_tokenizer(data).map_err(|e| {
                        format!(
                            "{}: no tokenizer.json and GGUF tokenizer unreadable: {e}",
                            p.display()
                        )
                    })?
                }
                (None, _) => return Err(format!("Model '{id}': weights found but no tokenizer.json")),
            };
            (body, tok, p.display().to_string())
        }
        None if crate::model::hub::is_synthetic_model_id(id) => {
            let llm = GraphLm::new(cfg.clone());
            let tok = json_tokenizer.unwrap_or_else(|| llm.tokenizer.clone());
            (LmBody::Built(llm), tok, id.to_string())
        }
        None => {
            return Err(format!(
                "Model '{id}' has no weight file; refusing to run random-initialized weights"
            ))
        }
    };
    Ok(LmSource { cfg, tokenizer, label, body })
}

fn default_hidden() -> usize {
    128
}
fn default_layers() -> usize {
    2
}
fn default_heads() -> usize {
    4
}
fn default_vocab() -> usize {
    512
}

pub struct HuggingFaceModelDir;

impl HuggingFaceModelDir {
    /// Loads and initializes a model from a local HuggingFace checkpoint directory
    pub fn load_from_dir(dir_path: &Path) -> Result<Arc<NanoTransformerModel>, String> {
        Self::load_with_quant_mode(dir_path, crate::hardware::QuantMode::Uncompressed)
    }

    /// Loads and initializes a model from a local HuggingFace checkpoint directory with specified QuantMode
    pub fn load_with_quant_mode(
        dir_path: &Path,
        mode: crate::hardware::QuantMode,
    ) -> Result<Arc<NanoTransformerModel>, String> {
        let config_path = dir_path.join("config.json");
        let model_cfg = if config_path.exists() {
            let config_bytes = fs::read(&config_path).map_err(|e| e.to_string())?;
            let hf_cfg: HfConfigJson =
                serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
            NanoTransformerConfig {
                vocab_size: hf_cfg.vocab_size,
                hidden_dim: hf_cfg.hidden_dim,
                intermediate_dim: hf_cfg.resolved_intermediate_dim(),
                num_layers: hf_cfg.num_layers,
                num_heads: hf_cfg.num_heads,
                head_dim: hf_cfg.hidden_dim / hf_cfg.num_heads.max(1),
                eps: 1e-5,
            }
        } else {
            NanoTransformerConfig::default()
        };

        let mut model = NanoTransformerModel::new(model_cfg);

        // Check for model.safetensors in directory and load weights if present
        let safetensors_path = dir_path.join("model.safetensors");
        if safetensors_path.exists() {
            let st_bytes = fs::read(&safetensors_path).map_err(|e| e.to_string())?;
            if let Ok(manifest) = UniversalModelLoader::parse_safetensors(&st_bytes) {
                let extract_cand = |cands: &[&str]| -> Option<Vec<f32>> {
                    let name = find_candidate(&manifest, cands)?;
                    manifest
                        .extract_safetensors_tensor_f32(&st_bytes, name)
                        .ok()
                };

                let h = model.config.hidden_dim;
                let v = model.config.vocab_size;
                let inter = model.config.intermediate_dim;

                // 1. Embedding table
                if let Some(emb) = extract_cand(&[
                    "embedding_table",
                    "model.embed_tokens.weight",
                    "embed_tokens.weight",
                    "transformer.wte.weight",
                    "embeddings.word_embeddings.weight",
                    "tok_embeddings.weight",
                    "wte.weight",
                ]) {
                    if emb.len() == v * h {
                        model.embedding_table = emb;
                    } else if emb.len() >= v * h {
                        model.embedding_table = emb[..v * h].to_vec();
                    } else if !emb.is_empty() {
                        let copy_len = emb.len().min(model.embedding_table.len());
                        model.embedding_table[..copy_len].copy_from_slice(&emb[..copy_len]);
                    }
                }

                // 2. Transformer layer weights
                for l in 0..model.config.num_layers {
                    // QKV weights
                    let qkv_cands = [
                        format!("qkv_weights.{l}"),
                        format!("layer_{l}_qkv"),
                        format!("layers.{l}.qkv.weight"),
                        format!("model.layers.{l}.self_attn.qkv_proj.weight"),
                        format!("model.layers.{l}.self_attn.qkv.weight"),
                        format!("transformer.h.{l}.attn.c_attn.weight"),
                    ];
                    let qkv_str_cands: Vec<&str> = qkv_cands.iter().map(|s| s.as_str()).collect();

                    if let Some(qkv) = extract_cand(&qkv_str_cands) {
                        if qkv.len() == h * 3 * h {
                            model.qkv_weights[l] = qkv;
                        } else if qkv.len() >= h * 3 * h {
                            model.qkv_weights[l] = qkv[..h * 3 * h].to_vec();
                        }
                    } else {
                        let q_cands = [
                            format!("model.layers.{l}.self_attn.q_proj.weight"),
                            format!("layers.{l}.self_attn.q_proj.weight"),
                            format!("layer_{l}_q"),
                        ];
                        let k_cands = [
                            format!("model.layers.{l}.self_attn.k_proj.weight"),
                            format!("layers.{l}.self_attn.k_proj.weight"),
                            format!("layer_{l}_k"),
                        ];
                        let v_cands = [
                            format!("model.layers.{l}.self_attn.v_proj.weight"),
                            format!("layers.{l}.self_attn.v_proj.weight"),
                            format!("layer_{l}_v"),
                        ];
                        let q_str: Vec<&str> = q_cands.iter().map(|s| s.as_str()).collect();
                        let k_str: Vec<&str> = k_cands.iter().map(|s| s.as_str()).collect();
                        let v_str: Vec<&str> = v_cands.iter().map(|s| s.as_str()).collect();

                        if let (Some(q), Some(k), Some(v)) = (
                            extract_cand(&q_str),
                            extract_cand(&k_str),
                            extract_cand(&v_str),
                        ) {
                            if q.len() == h * h && k.len() == h * h && v.len() == h * h {
                                let mut fused = vec![0.0f32; h * 3 * h];
                                for j in 0..h {
                                    for i in 0..h {
                                        fused[j * (3 * h) + i] = q[i * h + j];
                                        fused[j * (3 * h) + h + i] = k[i * h + j];
                                        fused[j * (3 * h) + 2 * h + i] = v[i * h + j];
                                    }
                                }
                                model.qkv_weights[l] = fused;
                            }
                        }
                    }

                    // Out projection
                    let out_cands = [
                        format!("out_weights.{l}"),
                        format!("layer_{l}_out"),
                        format!("layers.{l}.out.weight"),
                        format!("model.layers.{l}.self_attn.o_proj.weight"),
                        format!("model.layers.{l}.self_attn.out_proj.weight"),
                        format!("transformer.h.{l}.attn.c_proj.weight"),
                    ];
                    let out_str_cands: Vec<&str> = out_cands.iter().map(|s| s.as_str()).collect();
                    if let Some(out) = extract_cand(&out_str_cands) {
                        if out.len() == h * h {
                            model.out_weights[l] = out;
                        } else if out.len() >= h * h {
                            model.out_weights[l] = out[..h * h].to_vec();
                        }
                    }

                    // MLP gate/up
                    let mlp_gate_cands = [
                        format!("mlp_gate_up.{l}"),
                        format!("layer_{l}_mlp_gate_up"),
                        format!("layers.{l}.mlp_gate_up.weight"),
                        format!("model.layers.{l}.mlp.gate_up_proj.weight"),
                    ];
                    let mlp_gate_str_cands: Vec<&str> =
                        mlp_gate_cands.iter().map(|s| s.as_str()).collect();
                    if let Some(gate_up) = extract_cand(&mlp_gate_str_cands) {
                        if gate_up.len() == h * 2 * inter {
                            model.mlp_gate_up[l] = gate_up;
                        } else if gate_up.len() >= h * 2 * inter {
                            model.mlp_gate_up[l] = gate_up[..h * 2 * inter].to_vec();
                        }
                    } else {
                        let gate_cand = [
                            format!("model.layers.{l}.mlp.gate_proj.weight"),
                            format!("layers.{l}.mlp.gate_proj.weight"),
                            format!("layer_{l}_gate"),
                        ];
                        let up_cand = [
                            format!("model.layers.{l}.mlp.up_proj.weight"),
                            format!("layers.{l}.mlp.up_proj.weight"),
                            format!("layer_{l}_up"),
                        ];
                        let gate_str: Vec<&str> = gate_cand.iter().map(|s| s.as_str()).collect();
                        let up_str: Vec<&str> = up_cand.iter().map(|s| s.as_str()).collect();

                        if let (Some(gate), Some(up)) =
                            (extract_cand(&gate_str), extract_cand(&up_str))
                        {
                            if gate.len() == inter * h && up.len() == inter * h {
                                let mut fused = vec![0.0f32; h * 2 * inter];
                                for j in 0..h {
                                    for i in 0..inter {
                                        fused[j * (2 * inter) + i] = gate[i * h + j];
                                        fused[j * (2 * inter) + inter + i] = up[i * h + j];
                                    }
                                }
                                model.mlp_gate_up[l] = fused;
                            }
                        }
                    }

                    // MLP down
                    let mlp_down_cands = [
                        format!("mlp_down.{l}"),
                        format!("layer_{l}_mlp_down"),
                        format!("layers.{l}.mlp_down.weight"),
                        format!("model.layers.{l}.mlp.down_proj.weight"),
                        format!("transformer.h.{l}.mlp.c_proj.weight"),
                    ];
                    let mlp_down_str_cands: Vec<&str> =
                        mlp_down_cands.iter().map(|s| s.as_str()).collect();
                    if let Some(down) = extract_cand(&mlp_down_str_cands) {
                        if down.len() == inter * h {
                            model.mlp_down[l] = down;
                        } else if down.len() >= inter * h {
                            model.mlp_down[l] = down[..inter * h].to_vec();
                        }
                    }

                    // Norm weights
                    let norm_cands = [
                        format!("norm_weights.{l}"),
                        format!("layer_{l}_norm"),
                        format!("layers.{l}.norm.weight"),
                        format!("model.layers.{l}.input_layernorm.weight"),
                        format!("model.layers.{l}.norm.weight"),
                        format!("transformer.h.{l}.ln_1.weight"),
                    ];
                    let norm_str_cands: Vec<&str> = norm_cands.iter().map(|s| s.as_str()).collect();
                    if let Some(norm) = extract_cand(&norm_str_cands) {
                        if norm.len() == h {
                            model.norm_weights[l] = norm;
                        } else if norm.len() >= h {
                            model.norm_weights[l] = norm[..h].to_vec();
                        }
                    }
                }

                // 3. Final norm
                if let Some(fnorm) = extract_cand(&[
                    "final_norm",
                    "final_norm.weight",
                    "model.norm.weight",
                    "transformer.ln_f.weight",
                ]) {
                    if fnorm.len() == h {
                        model.final_norm = fnorm;
                    } else if fnorm.len() >= h {
                        model.final_norm = fnorm[..h].to_vec();
                    }
                }

                // 4. LM Head
                if let Some(lm) =
                    extract_cand(&["lm_head", "lm_head.weight", "model.lm_head.weight"])
                {
                    if lm.len() == h * v {
                        model.lm_head = lm;
                    } else if lm.len() >= h * v {
                        model.lm_head = lm[..h * v].to_vec();
                    }
                }
            }
        }

        if mode != crate::hardware::QuantMode::Uncompressed || safetensors_path.exists() {
            model.quantize_model(mode);
        }
        Ok(Arc::new(model))
    }
}

fn find_candidate<'a>(
    manifest: &'a crate::model::ModelManifest,
    candidates: &[&str],
) -> Option<&'a str> {
    for &cand in candidates {
        if let Some((key, _)) = manifest.tensors.get_key_value(cand) {
            return Some(key.as_str());
        }
    }
    None
}

/// One chat message as HF chat templates read it (`apply_chat_template` message dicts).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// A string, or HF-style parts: `{"type":"text","text":..}`, `{"type":"image"}`,
    /// `{"type":"audio"}`. Media bytes travel beside the prompt
    /// ([`crate::model::native::MediaInput`]), in part order.
    #[serde(default)]
    pub content: serde_json::Value,
    /// Assistant tool calls, OpenAI shape: `[{"id","type":"function","function":{"name","arguments"}}]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// For `role: "tool"` results: the call they answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: serde_json::Value::String(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// The message text: the string content, or its text parts concatenated.
    pub fn text(&self) -> String {
        match &self.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => {
                parts.iter().filter_map(|p| p["text"].as_str()).collect()
            }
            _ => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_load_model_with_safetensors_weights() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path();

        let config_json = r#"{
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8
        }"#;
        fs::write(model_dir.join("config.json"), config_json).unwrap();

        // Create safetensors with custom embedding table and final_norm
        // embedding_table: 8 * 4 = 32 floats
        let emb_weights: Vec<f32> = (0..32).map(|i| (i as f32) + 0.5).collect();
        let mut emb_bytes = Vec::new();
        for &w in &emb_weights {
            emb_bytes.extend_from_slice(&w.to_le_bytes());
        }

        let fnorm_weights: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
        let mut fnorm_bytes = Vec::new();
        for &w in &fnorm_weights {
            fnorm_bytes.extend_from_slice(&w.to_le_bytes());
        }

        let off0 = 0;
        let off1 = off0 + emb_bytes.len();
        let off2 = off1 + fnorm_bytes.len();

        let header_json = serde_json::json!({
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [8, 4],
                "data_offsets": [off0, off1]
            },
            "model.norm.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [off1, off2]
            }
        })
        .to_string();

        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut st_data = Vec::new();
        st_data.extend_from_slice(&header_len.to_le_bytes());
        st_data.extend_from_slice(header_bytes);
        st_data.extend_from_slice(&emb_bytes);
        st_data.extend_from_slice(&fnorm_bytes);

        fs::write(model_dir.join("model.safetensors"), st_data).unwrap();

        let model = HuggingFaceModelDir::load_from_dir(model_dir).unwrap();
        assert_eq!(model.config.hidden_dim, 4);
        assert_eq!(model.config.vocab_size, 8);
        assert_eq!(model.embedding_table, emb_weights);
        assert_eq!(model.final_norm, fnorm_weights);
    }

    /// A sharded checkpoint (`model.safetensors.index.json` + 2 bf16 shards) loads and answers greedily.
    #[test]
    fn sharded_safetensors_index_loads_and_generates() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("disk/models--Qwen--Qwen2.5-3B-Instruct/snapshots/main");
        if !dir.join("model-00002-of-00002.safetensors").exists() {
            println!("SKIP: Qwen/Qwen2.5-3B-Instruct not in disk/");
            return;
        }
        let artifacts = crate::model::hub::ModelArtifacts {
            model_id: "Qwen/Qwen2.5-3B-Instruct".into(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors.index.json")),
            ..Default::default()
        };
        let llm = load_causal_lm(&artifacts, crate::hardware::QuantMode::Uncompressed).unwrap();
        let out = llm.generate_text("The capital of France is", 8, 0.0);
        println!("Qwen2.5-3B-Instruct (sharded): {out:?}");
        assert!(out.contains("Paris"), "{out:?}");
    }

    /// A real llama.cpp split GGUF (`gguf-split`, 3 shards, quant folder) loads as one weight set
    /// and, with GIBBS_TEST_MERGED_GGUF set to the shards merged into one file, equals its output.
    #[test]
    fn split_gguf_loads_like_its_merge() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("disk/models--owalsh--SmolLM2-135M-Instruct-GGUF-Split/snapshots/main/Q4_0");
        let first = dir.join("SmolLM2-135M-Instruct-Q4_0-00001-of-00003.gguf");
        if !dir
            .join("SmolLM2-135M-Instruct-Q4_0-00003-of-00003.gguf")
            .exists()
        {
            println!("SKIP: owalsh/SmolLM2-135M-Instruct-GGUF-Split not in disk/");
            return;
        }
        let load = |p: std::path::PathBuf| {
            let artifacts = crate::model::hub::ModelArtifacts {
                model_id: "split".into(),
                weights_path: Some(p),
                ..Default::default()
            };
            load_causal_lm(&artifacts, crate::hardware::QuantMode::Uncompressed).unwrap()
        };
        let split = load(first.clone()).generate_text("The capital of France is", 12, 0.0);
        println!("split: {split:?}");
        assert!(!split.trim().is_empty());
        let (m, _) = crate::model::loaders::gguf::open(&first).unwrap();
        assert_eq!((m.shard_count, m.tensors.len()), (3, 272));
        // GIBBS_TEST_MERGED_GGUF: the same shards merged into one file (`gguf-split --merge`).
        if let Some(merged) = std::env::var_os("GIBBS_TEST_MERGED_GGUF") {
            let merged = load(merged.into()).generate_text("The capital of France is", 12, 0.0);
            assert_eq!(split, merged);
        }
    }
}
