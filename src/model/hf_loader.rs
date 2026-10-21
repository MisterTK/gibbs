use crate::model::tokenizer::Tokenizer;
use std::fs;
use std::path::Path;

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

    /// Greedy text of `prompt` from `weights` (config/tokenizer beside it when present) through
    /// the family-spec loader.
    fn complete(id: &str, weights: std::path::PathBuf, prompt: &str, n: usize) -> String {
        let dir = weights.parent().unwrap().to_path_buf();
        let file = |n: &str| Some(dir.join(n)).filter(|p| p.exists());
        let artifacts = crate::model::hub::ModelArtifacts {
            model_id: id.into(),
            local_dir: Some(dir.clone()),
            config_path: file("config.json"),
            tokenizer_path: file("tokenizer.json"),
            weights_path: Some(weights),
            ..Default::default()
        };
        let reg = crate::arch::registry().unwrap();
        let lm = crate::runtime::lm::load(&artifacts, reg, None, crate::hardware::PlacementMode::Auto).unwrap();
        crate::scheduler::ExecutableModel::new(lm)
            .generate_text_measured_plain(prompt, n, 0.0)
            .unwrap()
            .0
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
        let w = dir.join("model.safetensors.index.json");
        let out = complete("Qwen/Qwen2.5-3B-Instruct", w, "The capital of France is", 8);
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
        let split = complete("split", first.clone(), "The capital of France is", 12);
        println!("split: {split:?}");
        assert!(!split.trim().is_empty());
        let (m, _) = crate::model::loaders::gguf::open(&first).unwrap();
        assert_eq!((m.shard_count, m.tensors.len()), (3, 272));
        // GIBBS_TEST_MERGED_GGUF: the same shards merged into one file (`gguf-split --merge`).
        if let Some(merged) = std::env::var_os("GIBBS_TEST_MERGED_GGUF") {
            let merged = complete("merged", merged.into(), "The capital of France is", 12);
            assert_eq!(split, merged);
        }
    }
}
