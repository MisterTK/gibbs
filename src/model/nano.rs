use crate::hardware::QuantMode;
use crate::model::layers::QuantizedLinearLayer;

/// Helper to transpose weight matrices from [in_features, out_features] to [out_features, in_features]
fn transpose_weights(weights: &[f32], in_features: usize, out_features: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; out_features * in_features];
    for j in 0..in_features {
        for i in 0..out_features {
            transposed[i * in_features + j] = weights[j * out_features + i];
        }
    }
    transposed
}

/// Configuration for a tiny ~1M parameter model designed for instant in-memory verification
#[derive(Debug, Clone)]
pub struct NanoTransformerConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
}

impl Default for NanoTransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            hidden_dim: 128,
            intermediate_dim: 256,
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            eps: 1e-5,
        }
    }
}

#[derive(Clone)]
pub struct NanoTransformerModel {
    pub config: NanoTransformerConfig,
    pub embedding_table: Vec<f32>,   // vocab_size x hidden_dim
    pub qkv_weights: Vec<Vec<f32>>,  // num_layers x (hidden_dim x 3*hidden_dim)
    pub out_weights: Vec<Vec<f32>>,  // num_layers x (hidden_dim x hidden_dim)
    pub mlp_gate_up: Vec<Vec<f32>>,  // num_layers x (hidden_dim x 2*intermediate_dim)
    pub mlp_down: Vec<Vec<f32>>,     // num_layers x (intermediate_dim x hidden_dim)
    pub norm_weights: Vec<Vec<f32>>, // num_layers x hidden_dim
    pub final_norm: Vec<f32>,        // hidden_dim
    pub lm_head: Vec<f32>,           // hidden_dim x vocab_size
    pub quant_mode: QuantMode,
    pub quantized_qkv: Vec<QuantizedLinearLayer>,
    pub quantized_out: Vec<QuantizedLinearLayer>,
    pub quantized_mlp_gate_up: Vec<QuantizedLinearLayer>,
    pub quantized_mlp_down: Vec<QuantizedLinearLayer>,
    pub quantized_lm_head: Option<QuantizedLinearLayer>,
}

impl NanoTransformerModel {
    pub fn new(config: NanoTransformerConfig) -> Self {
        let h = config.hidden_dim;
        let v = config.vocab_size;
        let inter = config.intermediate_dim;
        let layers = config.num_layers;

        // Initialize deterministic synthetic weights
        let embedding_table = (0..(v * h))
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let mut qkv_weights = Vec::new();
        let mut out_weights = Vec::new();
        let mut mlp_gate_up = Vec::new();
        let mut mlp_down = Vec::new();
        let mut norm_weights = Vec::new();

        for l in 0..layers {
            qkv_weights.push(
                (0..(h * 3 * h))
                    .map(|i| ((i + l) as f32 * 0.02).cos() * 0.05)
                    .collect(),
            );
            out_weights.push(
                (0..(h * h))
                    .map(|i| ((i + l * 2) as f32 * 0.03).sin() * 0.05)
                    .collect(),
            );
            mlp_gate_up.push(
                (0..(h * 2 * inter))
                    .map(|i| ((i + l * 3) as f32 * 0.01).cos() * 0.05)
                    .collect(),
            );
            mlp_down.push(
                (0..(inter * h))
                    .map(|i| ((i + l * 4) as f32 * 0.02).sin() * 0.05)
                    .collect(),
            );
            norm_weights.push(vec![1.0; h]);
        }

        let final_norm = vec![1.0; h];
        let lm_head = (0..(h * v))
            .map(|i| (i as f32 * 0.005).sin() * 0.1)
            .collect();

        let mut model = Self {
            config,
            embedding_table,
            qkv_weights,
            out_weights,
            mlp_gate_up,
            mlp_down,
            norm_weights,
            final_norm,
            lm_head,
            quant_mode: QuantMode::Uncompressed,
            quantized_qkv: Vec::new(),
            quantized_out: Vec::new(),
            quantized_mlp_gate_up: Vec::new(),
            quantized_mlp_down: Vec::new(),
            quantized_lm_head: None,
        };
        model.quantize_model(QuantMode::Uncompressed);
        model
    }

    /// Quantizes the model's internal linear projections according to the specified QuantMode
    pub fn quantize_model(&mut self, mode: QuantMode) {
        let h = self.config.hidden_dim;
        let v = self.config.vocab_size;
        let inter = self.config.intermediate_dim;
        let layers = self.config.num_layers;

        self.quant_mode = mode;

        let (attn_format, ffn_format, lm_format) = crate::model::graph_lm::projection_formats(mode);

        let mut qkv = Vec::with_capacity(layers);
        let mut out = Vec::with_capacity(layers);
        let mut mlp_gate = Vec::with_capacity(layers);
        let mut mlp_d = Vec::with_capacity(layers);

        for l in 0..layers {
            let qkv_t = transpose_weights(&self.qkv_weights[l], h, 3 * h);
            qkv.push(
                QuantizedLinearLayer::from_f32(
                    &format!("layer_{}_qkv", l),
                    h,
                    3 * h,
                    &qkv_t,
                    attn_format,
                )
                .expect("Failed to quantize QKV layer"),
            );

            let out_t = transpose_weights(&self.out_weights[l], h, h);
            out.push(
                QuantizedLinearLayer::from_f32(
                    &format!("layer_{}_out", l),
                    h,
                    h,
                    &out_t,
                    attn_format,
                )
                .expect("Failed to quantize Out-Proj layer"),
            );

            let gate_t = transpose_weights(&self.mlp_gate_up[l], h, 2 * inter);
            mlp_gate.push(
                QuantizedLinearLayer::from_f32(
                    &format!("layer_{}_mlp_gate_up", l),
                    h,
                    2 * inter,
                    &gate_t,
                    ffn_format,
                )
                .expect("Failed to quantize MLP Gate/Up layer"),
            );

            let down_t = transpose_weights(&self.mlp_down[l], inter, h);
            mlp_d.push(
                QuantizedLinearLayer::from_f32(
                    &format!("layer_{}_mlp_down", l),
                    inter,
                    h,
                    &down_t,
                    ffn_format,
                )
                .expect("Failed to quantize MLP Down layer"),
            );
        }

        let lm_t = transpose_weights(&self.lm_head, h, v);
        let lm = QuantizedLinearLayer::from_f32("lm_head", h, v, &lm_t, lm_format)
            .expect("Failed to quantize LM Head layer");

        self.quantized_qkv = qkv;
        self.quantized_out = out;
        self.quantized_mlp_gate_up = mlp_gate;
        self.quantized_mlp_down = mlp_d;
        self.quantized_lm_head = Some(lm);
    }

    /// Single-token forward decode pass executing fused RMSNorm, GEMV, SwiGLU, and Softmax
    pub fn forward_token(&self, token: u32, _position: usize) -> u32 {
        let h = self.config.hidden_dim;
        let v = self.config.vocab_size;
        let inter = self.config.intermediate_dim;

        // 1. Embedding Lookup
        let token_idx = (token as usize).min(v - 1);
        let mut hidden: Vec<f32> =
            self.embedding_table[token_idx * h..(token_idx + 1) * h].to_vec();

        // 2. Transformer Layers
        for l in 0..self.config.num_layers {
            // Fused RMSNorm
            let mut normalized = vec![0.0; h];
            let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
            let scale = 1.0 / (sum_sq / h as f32 + self.config.eps).sqrt();
            for i in 0..h {
                normalized[i] = hidden[i] * scale * self.norm_weights[l][i];
            }

            // QKV GEMV
            let mut qkv = vec![0.0; 3 * h];
            self.quantized_qkv[l].forward(&normalized, &mut qkv);

            // Out-Proj GEMV + Residual Add
            let mut attn_out = vec![0.0; h];
            self.quantized_out[l].forward(&qkv[..h], &mut attn_out);
            for i in 0..h {
                hidden[i] += attn_out[i];
            }

            // MLP SwiGLU: (x * silu(gate)) * up -> down
            let mut gate_up = vec![0.0; 2 * inter];
            self.quantized_mlp_gate_up[l].forward(&hidden, &mut gate_up);

            let mut mlp_inter = vec![0.0; inter];
            for i in 0..inter {
                let gate = gate_up[i];
                let up = gate_up[i + inter];
                let silu = gate / (1.0 + (-gate).exp());
                mlp_inter[i] = silu * up;
            }

            let mut down_out = vec![0.0; h];
            self.quantized_mlp_down[l].forward(&mlp_inter, &mut down_out);
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }

        // 3. Final RMSNorm
        let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
        let scale = 1.0 / (sum_sq / h as f32 + self.config.eps).sqrt();
        for i in 0..h {
            hidden[i] = hidden[i] * scale * self.final_norm[i];
        }

        // 4. LM Head Logits & Greedy Argmax Selection
        let mut best_token = 0;
        let mut best_logit = f32::NEG_INFINITY;
        if let Some(lm_head) = &self.quantized_lm_head {
            let mut logits = vec![0.0; v];
            lm_head.forward(&hidden, &mut logits);
            for (t, &logit) in logits.iter().enumerate() {
                if logit > best_logit {
                    best_logit = logit;
                    best_token = t as u32;
                }
            }
        } else {
            for t in 0..v {
                let mut logit = 0.0;
                for i in 0..h {
                    logit += hidden[i] * self.lm_head[i * v + t];
                }
                if logit > best_logit {
                    best_logit = logit;
                    best_token = t as u32;
                }
            }
        }

        best_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_transformer_quantization_modes() {
        let cfg = NanoTransformerConfig {
            vocab_size: 256,
            hidden_dim: 64,
            intermediate_dim: 128,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            eps: 1e-5,
        };

        let mut model = NanoTransformerModel::new(cfg);
        let t0 = model.forward_token(42, 0);
        assert!(t0 < 256);

        // Quantize to NVFP4
        model.quantize_model(QuantMode::NVFP4Ultra);
        assert_eq!(model.quant_mode, QuantMode::NVFP4Ultra);
        let t_nvfp4 = model.forward_token(42, 0);
        assert!(t_nvfp4 < 256);

        // Quantize to EXL3
        model.quantize_model(QuantMode::EXL3Balanced);
        assert_eq!(model.quant_mode, QuantMode::EXL3Balanced);
        let t_exl3 = model.forward_token(42, 0);
        assert!(t_exl3 < 256);

        // Quantize to AdaptiveHybrid
        model.quantize_model(QuantMode::AdaptiveHybrid);
        assert_eq!(model.quant_mode, QuantMode::AdaptiveHybrid);
        let t_hybrid = model.forward_token(42, 0);
        assert!(t_hybrid < 256);
    }
}
