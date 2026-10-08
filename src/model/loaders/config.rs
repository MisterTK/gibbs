use std::collections::HashMap;
use serde_json::Value;

pub trait ConfigExtractor {
    fn extract_config(&self) -> Result<ModelConfig, String>;
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub rope_interleaved: bool,
    pub norm_eps: f32,
    pub attention_bias: bool,
    pub model_type: String,
    pub arch_features: HashMap<String, String>,
}

pub struct HuggingFaceConfigExtractor {
    pub config_json: Value,
}

impl ConfigExtractor for HuggingFaceConfigExtractor {
    fn extract_config(&self) -> Result<ModelConfig, String> {
        let obj = self.config_json.as_object()
            .ok_or_else(|| "config.json must be an object".to_string())?;

        let vocab_size = obj.get("vocab_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "vocab_size missing".to_string())? as usize;

        let hidden_dim = obj.get("hidden_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "hidden_size missing".to_string())? as usize;

        let num_layers = obj.get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "num_hidden_layers missing".to_string())? as usize;

        let num_heads = obj.get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "num_attention_heads missing".to_string())? as usize;

        let head_dim = hidden_dim / num_heads;

        let intermediate_dim = obj.get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or((hidden_dim * 4) as u64) as usize;

        let max_seq_len = obj.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        let rope_dim = obj.get("rope_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(head_dim as u64) as usize;

        let rope_base = obj.get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        let rope_interleaved = obj.get("rope_interleaved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let norm_eps = obj.get("rms_norm_eps")
            .or_else(|| obj.get("layer_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        let attention_bias = obj.get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let model_type = obj.get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ModelConfig {
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            head_dim,
            intermediate_dim,
            max_seq_len,
            rope_dim,
            rope_base,
            rope_interleaved,
            norm_eps,
            attention_bias,
            model_type,
            arch_features: HashMap::new(),
        })
    }
}

pub struct GgufConfigExtractor {
    pub kv_pairs: HashMap<String, String>,
}

impl ConfigExtractor for GgufConfigExtractor {
    fn extract_config(&self) -> Result<ModelConfig, String> {
        let arch = Self::infer_arch(&self.kv_pairs)?;

        let vocab_size = self.get_u64(&format!("{}.vocab_size", arch))
            .ok_or("vocab_size not found")?;

        let hidden_dim = self.get_u64(&format!("{}.embedding_length", arch))
            .ok_or("embedding_length not found")?;

        let num_layers = self.get_u64(&format!("{}.block_count", arch))
            .ok_or("block_count not found")?;

        let num_heads = self.get_u64(&format!("{}.attention.head_count", arch))
            .ok_or("head_count not found")?;

        let head_dim = hidden_dim / num_heads;

        let intermediate_dim = self.get_u64(&format!("{}.feed_forward_length", arch))
            .unwrap_or(hidden_dim * 4);

        let max_seq_len = self.get_u64(&format!("{}.context_length", arch))
            .unwrap_or(2048);

        let rope_dim = self.get_u64(&format!("{}.rope.dimension_count", arch))
            .unwrap_or(head_dim);

        let rope_base = self.get_f64(&format!("{}.rope.freq_base", arch))
            .unwrap_or(10000.0) as f32;

        let rope_interleaved = self.kv_pairs.get("tokenizer.ggml.rope_interleaved")
            .map(|v| v == "true")
            .unwrap_or(false);

        let norm_eps = self.get_f64(&format!("{}.attention.layer_norm_rms_epsilon", arch))
            .unwrap_or(1e-5) as f32;

        let attention_bias = false;

        Ok(ModelConfig {
            vocab_size: vocab_size as usize,
            hidden_dim: hidden_dim as usize,
            num_layers: num_layers as usize,
            num_heads: num_heads as usize,
            head_dim: head_dim as usize,
            intermediate_dim: intermediate_dim as usize,
            max_seq_len: max_seq_len as usize,
            rope_dim: rope_dim as usize,
            rope_base,
            rope_interleaved,
            norm_eps,
            attention_bias,
            model_type: arch.to_string(),
            arch_features: HashMap::new(),
        })
    }
}

impl GgufConfigExtractor {
    fn infer_arch(kv_pairs: &HashMap<String, String>) -> Result<String, String> {
        for key in kv_pairs.keys() {
            if let Some(pos) = key.find('.') {
                let arch = &key[..pos];
                if arch != "general" && arch != "tokenizer" {
                    return Ok(arch.to_string());
                }
            }
        }
        Err("Could not infer architecture from GGUF keys".to_string())
    }

    fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv_pairs.get(key)
            .and_then(|v| v.parse().ok())
    }

    fn get_f64(&self, key: &str) -> Option<f64> {
        self.kv_pairs.get(key)
            .and_then(|v| v.parse().ok())
    }
}

pub struct OnnxConfigExtractor {
    pub graph_inputs: Vec<(String, Vec<usize>)>,
    pub graph_outputs: Vec<(String, Vec<usize>)>,
    pub node_attrs: HashMap<String, HashMap<String, String>>,
}

impl ConfigExtractor for OnnxConfigExtractor {
    fn extract_config(&self) -> Result<ModelConfig, String> {
        let vocab_size = self.infer_vocab_size()?;
        let hidden_dim = self.infer_hidden_dim()?;
        let num_heads = self.infer_num_heads()?;

        let head_dim = hidden_dim / num_heads;

        Ok(ModelConfig {
            vocab_size,
            hidden_dim,
            num_layers: 32,
            num_heads,
            head_dim,
            intermediate_dim: hidden_dim * 4,
            max_seq_len: 2048,
            rope_dim: head_dim,
            rope_base: 10000.0,
            rope_interleaved: false,
            norm_eps: 1e-5,
            attention_bias: false,
            model_type: "onnx".to_string(),
            arch_features: HashMap::new(),
        })
    }
}

impl OnnxConfigExtractor {
    fn infer_vocab_size(&self) -> Result<usize, String> {
        for (_, shape) in &self.graph_inputs {
            if shape.len() >= 1 {
                return Ok(shape[0]);
            }
        }
        Err("Cannot infer vocab_size from ONNX graph".to_string())
    }

    fn infer_hidden_dim(&self) -> Result<usize, String> {
        for (_, shape) in &self.graph_outputs {
            if shape.len() >= 2 {
                return Ok(shape[shape.len() - 1]);
            }
        }
        Err("Cannot infer hidden_dim from ONNX graph".to_string())
    }

    fn infer_num_heads(&self) -> Result<usize, String> {
        for attrs in self.node_attrs.values() {
            if let Some(num_heads_str) = attrs.get("num_heads") {
                if let Ok(n) = num_heads_str.parse::<usize>() {
                    return Ok(n);
                }
            }
        }
        Ok(8)
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_dim < 64 || self.hidden_dim > 65536 {
            return Err(format!("hidden_dim {} out of reasonable range", self.hidden_dim));
        }
        if self.num_heads == 0 || self.hidden_dim % self.num_heads != 0 {
            return Err("num_heads must divide hidden_dim evenly".to_string());
        }
        if self.vocab_size == 0 || self.vocab_size > 1_000_000 {
            return Err(format!("vocab_size {} unreasonable", self.vocab_size));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_config_extraction() {
        let json_str = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "model_type": "llama"
        }"#;

        let json: Value = serde_json::from_str(json_str).unwrap();
        let extractor = HuggingFaceConfigExtractor { config_json: json };
        let config = extractor.extract_config().unwrap();

        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_dim, 11008);
        assert_eq!(config.model_type, "llama");
    }

    #[test]
    fn test_config_validation() {
        let config = ModelConfig {
            vocab_size: 32000,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            intermediate_dim: 11008,
            max_seq_len: 2048,
            rope_dim: 128,
            rope_base: 10000.0,
            rope_interleaved: false,
            norm_eps: 1e-6,
            attention_bias: false,
            model_type: "llama".to_string(),
            arch_features: HashMap::new(),
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = ModelConfig {
            vocab_size: 0,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            intermediate_dim: 11008,
            max_seq_len: 2048,
            rope_dim: 128,
            rope_base: 10000.0,
            rope_interleaved: false,
            norm_eps: 1e-6,
            attention_bias: false,
            model_type: "llama".to_string(),
            arch_features: HashMap::new(),
        };

        assert!(config.validate().is_err());
    }
}
