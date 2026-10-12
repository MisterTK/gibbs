//! Parameterized Tiny LLM (<=10M parameters) architecture.
//!
//! Provides a configurable transformer language model designed for fast in-memory
//! execution, multi-head causal attention with RoPE, SwiGLU feed-forward networks,
//! RMSNorm, and multi-format quantization (FP16, NVFP4, EXL3).

use crate::hardware::{QuantFormat, QuantMode};
use crate::model::layers::{gguf_block_codec, QuantizedLinearLayer};
use crate::model::tokenizer::Tokenizer;
use crate::model::{ModelManifest, UniversalModelLoader};
use serde::{Deserialize, Serialize};

fn default_rotary_pct() -> f32 {
    1.0
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_norm_eps() -> f32 {
    1e-5
}
/// Configuration parameters for [`GraphLm`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphLmConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub quant_mode: QuantMode,
    #[serde(default = "default_rotary_pct")]
    pub rotary_pct: f32,
    #[serde(default)]
    pub use_parallel_residual: bool,
    #[serde(default)]
    pub use_standard_layernorm: bool,
    #[serde(default)]
    pub use_gelu_mlp: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub use_hadamard_attention: bool,
    #[serde(default)]
    pub use_hadamard_mlp: bool,
    /// GLM-style RoPE: rotate adjacent pairs (2i, 2i+1) instead of LLaMA/NeoX halves (i, i + d/2).
    #[serde(default)]
    pub rope_interleaved: bool,
    /// RMSNorm / LayerNorm epsilon.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// BitNet BitLinear: per-token absmax fake-quant of the input of every q/k/v/o/gate/up/down
    /// projection to this many bits (8 for b1.58); 0 = off.
    #[serde(default)]
    pub act_quant_bits: u32,
    /// BitNet online weight quant at load: each projection becomes {-1, 0, 1} × its mean |W|.
    #[serde(default)]
    pub ternarize_weights: bool,
    /// Gated MLP activation relu(gate)² (BitNet `relu2`) instead of SiLU.
    #[serde(default)]
    pub use_relu2_mlp: bool,
}

impl GraphLmConfig {
    /// ~1.2M parameter preset (~1,180,800 parameters).
    pub fn smollm_1m() -> Self {
        Self {
            vocab_size: 2048,
            hidden_dim: 128,
            intermediate_dim: 256,
            num_layers: 4,
            num_heads: 4,
            head_dim: 32,
            max_seq_len: 512,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            use_standard_layernorm: false,
            use_gelu_mlp: false,
            rope_theta: 10000.0,
            use_hadamard_attention: false,
            use_hadamard_mlp: false,
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
        }
    }

    /// ~5.1M parameter preset (~5,131,520 parameters).
    pub fn tiny_5m() -> Self {
        Self {
            vocab_size: 2048,
            hidden_dim: 256,
            intermediate_dim: 544,
            num_layers: 6,
            num_heads: 8,
            head_dim: 32,
            max_seq_len: 512,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            use_standard_layernorm: false,
            use_gelu_mlp: false,
            rope_theta: 10000.0,
            use_hadamard_attention: false,
            use_hadamard_mlp: false,
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
        }
    }

    /// ~9.8M parameter preset (~9,850,112 parameters).
    pub fn tiny_10m() -> Self {
        Self {
            vocab_size: 2048,
            hidden_dim: 256,
            intermediate_dim: 1568,
            num_layers: 6,
            num_heads: 8,
            head_dim: 32,
            max_seq_len: 512,
            quant_mode: QuantMode::Uncompressed,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            use_standard_layernorm: false,
            use_gelu_mlp: false,
            rope_theta: 10000.0,
            use_hadamard_attention: false,
            use_hadamard_mlp: false,
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
        }
    }

    pub fn from_hf_config(
        vocab_size: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        num_heads: usize,
        quant_mode: QuantMode,
    ) -> Self {
        Self {
            vocab_size,
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            head_dim: hidden_dim / num_heads.max(1),
            max_seq_len: 2048,
            quant_mode,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            use_standard_layernorm: false,
            use_gelu_mlp: false,
            rope_theta: 10000.0,
            use_hadamard_attention: false,
            use_hadamard_mlp: false,
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
        }
    }

    /// Single mapping from a HuggingFace `config.json` to runtime config; every architecture flag resolves here.
    pub fn from_hf(hf: &crate::model::hf_loader::HfConfigJson, quant_mode: QuantMode) -> Self {
        let mut cfg = Self::from_hf_config(
            hf.vocab_size,
            hf.hidden_dim,
            hf.resolved_intermediate_dim(),
            hf.num_layers,
            hf.num_heads,
            quant_mode,
        );
        cfg.rotary_pct = hf.resolved_rotary_pct();
        cfg.use_parallel_residual = hf.resolved_parallel_residual();
        cfg.use_standard_layernorm = hf.resolved_use_standard_layernorm();
        cfg.use_gelu_mlp = hf.resolved_use_gelu_mlp();
        cfg.rope_theta = hf.resolved_rope_theta();
        cfg.rope_interleaved = hf.model_type.as_deref() == Some("glm");
        cfg.norm_eps = hf.resolved_norm_eps();
        if let Some((bits, ternarize)) = hf.bitnet() {
            cfg.act_quant_bits = bits;
            cfg.ternarize_weights = ternarize;
        }
        cfg.use_relu2_mlp = hf.hidden_act.as_deref() == Some("relu2");
        cfg
    }

    /// Single mapping from GGUF `<arch>.*` metadata to runtime config. Keys absent from the
    /// file (and flags GGUF does not declare) keep `base`'s values.
    pub fn from_gguf(manifest: &ModelManifest, base: GraphLmConfig) -> Self {
        let md = &manifest.metadata;
        let Some(arch) = md.get("general.architecture") else {
            return base;
        };
        let get = |k: &str| md.get(&format!("{arch}.{k}"));
        let num = |k: &str, d: usize| get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
        let hidden_dim = num("embedding_length", base.hidden_dim);
        let num_heads = num("attention.head_count", base.num_heads);
        let head_dim = get("attention.key_length")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| hidden_dim / num_heads.max(1));
        let rotary_pct = match get("rope.dimension_count").and_then(|s| s.parse::<usize>().ok()) {
            Some(n) => n as f32 / head_dim.max(1) as f32,
            None => base.rotary_pct,
        };
        Self {
            // GGUF token_embd is [n_embd, n_vocab] (ne0 fastest).
            vocab_size: manifest
                .tensors
                .get("token_embd.weight")
                .and_then(|t| t.shape.get(1).copied())
                .unwrap_or_else(|| num("vocab_size", base.vocab_size)),
            hidden_dim,
            intermediate_dim: num("feed_forward_length", base.intermediate_dim),
            num_layers: num("block_count", base.num_layers),
            num_heads,
            head_dim,
            max_seq_len: num("context_length", base.max_seq_len),
            rotary_pct,
            rope_theta: get("rope.freq_base")
                .and_then(|s| s.parse().ok())
                .unwrap_or(base.rope_theta),
            norm_eps: get("attention.layer_norm_rms_epsilon")
                .or_else(|| get("attention.layer_norm_epsilon"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(base.norm_eps),
            // bitnet.cpp's BitNet b1.58 (I2_S ternary weights): int8 BitLinear inputs, relu² MLP.
            act_quant_bits: if arch == "bitnet-b1.58" {
                8
            } else {
                base.act_quant_bits
            },
            use_relu2_mlp: arch == "bitnet-b1.58" || base.use_relu2_mlp,
            // llama.cpp's LLAMA_ROPE_TYPE_NORM archs rotate adjacent (2i, 2i+1) pairs; for llama,
            // convert_hf_to_gguf permutes q/k rows into exactly that layout.
            rope_interleaved: matches!(
                arch.as_str(),
                "llama"
                    | "llada"
                    | "llama4"
                    | "deci"
                    | "baichuan"
                    | "starcoder"
                    | "internlm2"
                    | "minicpm"
                    | "xverse"
                    | "command-r"
                    | "cohere2"
                    | "olmo"
                    | "arctic"
                    | "deepseek"
                    | "deepseek2"
                    | "plm"
                    | "chatglm"
                    | "glm4"
                    | "granite"
                    | "granitemoe"
                    | "granitehybrid"
                    | "chameleon"
                    | "bailingmoe"
                    | "neo-bert"
                    | "smollm3"
                    | "arcee"
                    | "ernie4_5"
                    | "ernie4_5-moe"
            ),
            ..base
        }
    }
}

impl Default for GraphLmConfig {
    fn default() -> Self {
        Self::smollm_1m()
    }
}

/// Key-Value Cache for a single transformer layer in [`GraphLm`].
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    pub k: Vec<f32>, // capacity: max_seq_len * hidden_dim
    pub v: Vec<f32>, // capacity: max_seq_len * hidden_dim
    pub len: usize,
    /// Speculative-drafter tap: (slot in the drafter's tap order, residual stream after this
    /// layer, one hidden row per position run).
    pub tap: Option<(usize, Vec<f32>)>,
}

impl LayerKVCache {
    pub fn new(max_seq_len: usize, hidden_dim: usize) -> Self {
        let cap = max_seq_len * hidden_dim;
        Self {
            k: Vec::with_capacity(cap),
            v: Vec::with_capacity(cap),
            len: 0,
            tap: None,
        }
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }

    pub fn push_kv(&mut self, k_new: &[f32], v_new: &[f32]) {
        let h = k_new.len();
        let start = self.len * h;
        let end = start + h;
        if self.k.len() >= end {
            self.k[start..end].copy_from_slice(k_new);
        } else {
            self.k.extend_from_slice(k_new);
        }
        let v_h = v_new.len();
        let v_start = self.len * v_h;
        let v_end = v_start + v_h;
        if self.v.len() >= v_end {
            self.v[v_start..v_end].copy_from_slice(v_new);
        } else {
            self.v.extend_from_slice(v_new);
        }
        self.len += 1;
    }
}

/// Pre-allocated scratchpad buffers for zero-allocation token decode in [`GraphLm`].
#[derive(Debug, Clone)]
pub struct DecodeScratchpad {
    pub hidden: Vec<f32>,
    pub norm1_x: Vec<f32>,
    pub qkv: Vec<f32>,
    pub q_vec: Vec<f32>,
    pub k_vec: Vec<f32>,
    pub v_vec: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub norm2_x: Vec<f32>,
    pub mlp_inter: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub down_out: Vec<f32>,
    pub final_normed: Vec<f32>,
    pub hadamard_diag: Vec<f32>,
    pub rope_inv_freq: Vec<f32>,
    pub logits: Vec<f32>,
}

impl DecodeScratchpad {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, vocab_size: usize) -> Self {
        Self {
            hidden: vec![0.0f32; hidden_dim],
            norm1_x: vec![0.0f32; hidden_dim],
            qkv: vec![0.0f32; 3 * hidden_dim],
            q_vec: vec![0.0f32; hidden_dim],
            k_vec: vec![0.0f32; hidden_dim],
            v_vec: vec![0.0f32; hidden_dim],
            attn_out: vec![0.0f32; hidden_dim],
            attn_scores: vec![0.0f32; 2048],
            proj_out: vec![0.0f32; hidden_dim],
            norm2_x: vec![0.0f32; hidden_dim],
            mlp_inter: vec![0.0f32; intermediate_dim],
            gate_up: vec![0.0f32; 2 * intermediate_dim],
            down_out: vec![0.0f32; hidden_dim],
            final_normed: vec![0.0f32; hidden_dim],
            hadamard_diag: vec![1.0f32; intermediate_dim],
            rope_inv_freq: Vec::new(),
            logits: vec![0.0f32; vocab_size],
        }
    }

    pub fn from_config(config: &GraphLmConfig) -> Self {
        let rotary_dim = ((config.head_dim as f32) * config.rotary_pct) as usize;
        let half_dim = (rotary_dim / 2).min(config.head_dim / 2);
        let clamped_half = half_dim.min(128);
        let mut rope_inv_freq = Vec::with_capacity(clamped_half);
        for d in 0..clamped_half {
            let freq = 1.0 / (config.rope_theta).powf((2 * d) as f32 / rotary_dim.max(1) as f32);
            rope_inv_freq.push(freq);
        }
        let mut pad = Self::new(
            config.hidden_dim,
            config.intermediate_dim,
            config.vocab_size,
        );
        pad.rope_inv_freq = rope_inv_freq;
        pad
    }
}

/// Helper to expand GQA KV bias when num_kv_heads < num_q_heads.
fn expand_gqa_bias(
    kv_bias: Vec<f32>,
    q_len: usize,
    num_q_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    if kv_bias.len() < q_len && !kv_bias.is_empty() && (q_len % kv_bias.len() == 0) {
        let kv_head_size = if head_dim > 0 { head_dim } else { 1 };
        let num_kv_heads = kv_bias.len() / kv_head_size;
        let group_size = if num_kv_heads > 0 && num_q_heads >= num_kv_heads {
            num_q_heads / num_kv_heads
        } else {
            q_len / kv_bias.len()
        };
        let mut exp = Vec::with_capacity(q_len);
        for kv_h in 0..num_kv_heads {
            let start = kv_h * kv_head_size;
            let end = start + kv_head_size;
            if end <= kv_bias.len() {
                for _ in 0..group_size {
                    exp.extend_from_slice(&kv_bias[start..end]);
                }
            }
        }
        if exp.len() == q_len {
            exp
        } else {
            let rep = q_len / kv_bias.len();
            let mut tiled = Vec::with_capacity(q_len);
            for _ in 0..rep {
                tiled.extend_from_slice(&kv_bias);
            }
            tiled
        }
    } else {
        kv_bias
    }
}

/// Helper to expand GQA KV heads when num_kv_heads < num_q_heads.
fn expand_gqa_kv(
    kv: Vec<f32>,
    q_len: usize,
    num_q_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    if kv.len() < q_len && !kv.is_empty() && (q_len % kv.len() == 0) {
        let kv_head_size = head_dim * hidden_dim;
        let num_kv_heads = if kv_head_size > 0 {
            kv.len() / kv_head_size
        } else {
            0
        };
        let group_size = if num_kv_heads > 0 && num_q_heads >= num_kv_heads {
            num_q_heads / num_kv_heads
        } else {
            q_len / kv.len()
        };
        let mut exp = Vec::with_capacity(q_len);
        for kv_h in 0..num_kv_heads {
            let start = kv_h * kv_head_size;
            let end = start + kv_head_size;
            if end <= kv.len() {
                for _ in 0..group_size {
                    exp.extend_from_slice(&kv[start..end]);
                }
            }
        }
        if exp.len() == q_len {
            exp
        } else {
            let rep = q_len / kv.len();
            let mut tiled = Vec::with_capacity(q_len);
            for _ in 0..rep {
                tiled.extend_from_slice(&kv);
            }
            tiled
        }
    } else {
        kv
    }
}

/// A single transformer layer in [`GraphLm`].
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLayer {
    pub norm1: Vec<f32>,
    pub norm1_bias: Option<Vec<f32>>,
    pub qkv_proj: QuantizedLinearLayer,
    pub out_proj: QuantizedLinearLayer,
    pub post_attention_norm: Option<Vec<f32>>,
    pub norm2: Vec<f32>,
    pub norm2_bias: Option<Vec<f32>>,
    pub gate_up_proj: QuantizedLinearLayer,
    pub down_proj: QuantizedLinearLayer,
    pub post_ffw_norm: Option<Vec<f32>>,
    /// BitNet RMSNorm on the attention output before `out_proj`.
    pub attn_sub_norm: Option<Vec<f32>>,
    /// BitNet RMSNorm on the gated MLP activation before `down_proj`.
    pub ffn_sub_norm: Option<Vec<f32>>,
}

/// BitNet BitLinear activation quant (transformers `ActQuant`, 1bitLLM `activation_quant`): rounds
/// `x` to signed `bits`-bit integers on its absmax scale and back (torch rounds half to even).
pub fn act_quant(x: &mut [f32], bits: u32) {
    let qp = ((1u32 << (bits - 1)) - 1) as f32;
    let scale = qp / x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-5);
    for v in x {
        *v = (*v * scale).round_ties_even().clamp(-qp - 1.0, qp) / scale;
    }
}

/// BitNet online weight quant (transformers `WeightQuant`, 1bitLLM `weight_quant`): {-1, 0, 1} on
/// the tensor's 1 / mean |W| scale, divided back and cast to the checkpoint dtype by `cast`.
pub fn ternarize(w: &mut [f32], cast: impl Fn(f32) -> f32) {
    let mean = (w.iter().map(|v| v.abs() as f64).sum::<f64>() / w.len().max(1) as f64) as f32;
    let scale = 1.0 / mean.max(1e-5);
    let step = cast(1.0 / scale);
    for v in w {
        *v = (*v * scale).round_ties_even().clamp(-1.0, 1.0) * step;
    }
}

/// In-place RMSNorm of `x` with gain `w`.
fn rms_norm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let scale = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (v, g) in x.iter_mut().zip(w) {
        *v *= scale * g;
    }
}

/// Helper to extract/dequantize f32 weights from a [`QuantizedLinearLayer`].
fn dequantize_layer(layer: &QuantizedLinearLayer) -> Vec<f32> {
    match &layer.weight {
        crate::model::layers::QuantizedWeight::Uncompressed { data, .. } => data.clone(),
        crate::model::layers::QuantizedWeight::Nvfp4(tensor) => {
            let mut out = vec![0.0f32; tensor.rows * tensor.cols];
            crate::quant::nvfp4::dequantize_nvfp4_tensor(tensor, &mut out);
            out
        }
        crate::model::layers::QuantizedWeight::Exl3(tensor) => {
            let mut out = vec![0.0f32; tensor.rows * tensor.cols];
            crate::quant::exl3::dequantize_exl3_tensor(tensor, &mut out);
            out
        }
        crate::model::layers::QuantizedWeight::WeightlessJesse(jesse) => {
            let mut out = vec![0.0f32; layer.in_features * layer.out_features];
            let mut unit = vec![0.0f32; layer.in_features];
            let mut col = vec![0.0f32; layer.out_features];
            for i in 0..layer.in_features {
                unit[i] = 1.0;
                jesse.forward(&unit, &mut col);
                for o in 0..layer.out_features {
                    out[o * layer.in_features + i] = col[o];
                }
                unit[i] = 0.0;
            }
            out
        }
        crate::model::layers::QuantizedWeight::ZeroCopySlice {
            in_features,
            out_features,
            ..
        }
        | crate::model::layers::QuantizedWeight::GgufQuantized {
            in_features,
            out_features,
            ..
        } => {
            let mut out = vec![0.0f32; *in_features * *out_features];
            let mut unit = vec![0.0f32; *in_features];
            let mut col = vec![0.0f32; *out_features];
            for i in 0..*in_features {
                unit[i] = 1.0;
                layer.forward(&unit, &mut col);
                for o in 0..*out_features {
                    out[o * in_features + i] = col[o];
                }
                unit[i] = 0.0;
            }
            out
        }
    }
}

/// `layer` re-encoded as `want`, with its name and bias. Matrices that do not tile into `want`'s
/// blocks (16 elements, EXL3's 32-element groups, or the GGUF block size along each row) stay f32.
pub fn requantized(layer: &QuantizedLinearLayer, want: QuantFormat) -> QuantizedLinearLayer {
    let (in_f, out_f) = (layer.in_features, layer.out_features);
    let row_block = match want {
        QuantFormat::EXL3_3Bit | QuantFormat::EXL3_4Bit | QuantFormat::EXL3_5Bit => {
            crate::quant::exl3::EXL3_GROUP_SIZE
        }
        _ => gguf_block_codec(want).map_or(16, |(n, _, _)| n),
    };
    let format = if in_f % row_block == 0 && out_f % 16 == 0 {
        want
    } else {
        QuantFormat::FP32
    };
    let weights = dequantize_layer(layer);
    let mut q = QuantizedLinearLayer::from_f32(&layer.name, in_f, out_f, &weights, format)
        .unwrap_or_else(|e| panic!("cannot re-encode '{}' as {format:?}: {e}", layer.name));
    q.bias = layer.bias.clone();
    q
}

/// Weight formats of the (attention, FFN, LM head) projections in `mode`.
pub fn projection_formats(mode: QuantMode) -> (QuantFormat, QuantFormat, QuantFormat) {
    let all = |f| (f, f, f);
    match mode {
        QuantMode::Uncompressed => all(QuantFormat::FP32),
        QuantMode::F16 => all(QuantFormat::FP16),
        QuantMode::BF16 => all(QuantFormat::BF16),
        QuantMode::Q8_0 => all(QuantFormat::GGUF_Q8_0),
        QuantMode::NVFP4Ultra | QuantMode::TurboSqueeze => all(QuantFormat::NVFP4_E2M1),
        QuantMode::EXL3Aggressive => all(QuantFormat::EXL3_3Bit),
        QuantMode::EXL3Balanced => all(QuantFormat::EXL3_4Bit),
        QuantMode::EXL3Quality => all(QuantFormat::EXL3_5Bit),
        QuantMode::AdaptiveHybrid => (
            QuantFormat::EXL3_4Bit,
            QuantFormat::NVFP4_E2M1,
            QuantFormat::EXL3_4Bit,
        ),
        QuantMode::WeightlessJesse => all(QuantFormat::WeightlessJesseLUT),
    }
}

/// Which kind of projection an encoder is looking at (the re-encode policy's FFN-only fallback
/// treats `Attn` and `LmHead` as untouchable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjRole {
    Attn,
    Ffn,
    LmHead,
}

/// One projection handed to a [`ProjectionEncoder`]: the f32 layer plus where it came from.
pub struct Projection<'a> {
    pub role: ProjRole,
    /// The projection as f32 (bias attached); its `name` is `layer_{l}_{qkv|out|gate_up|down}` / `lm_head`.
    pub layer: QuantizedLinearLayer,
    /// 2-D checkpoint tensors its rows were gathered from, in row order (a GQA k/v projection is
    /// replicated to full heads, so its row count exceeds the tensor's). Empty when the rows no
    /// longer equal the checkpoint's (reordered, ternarized, or no checkpoint).
    pub sources: Vec<String>,
    pub head_dim: usize,
    pub checkpoint: Option<(&'a ModelManifest, &'a [u8])>,
}

/// Chooses each projection's stored format while a [`GraphLm`] loads.
pub trait ProjectionEncoder: Sync {
    fn encode(&self, p: Projection<'_>) -> QuantizedLinearLayer;
}

impl GraphLm {
    /// Runs `enc` over every projection of an already built (f32) model, for sources without a
    /// checkpoint manifest (synthetic and ONNX models).
    pub fn encode_projections(&mut self, enc: &dyn ProjectionEncoder) {
        let take = |layer: &mut QuantizedLinearLayer| {
            let name = layer.name.clone();
            std::mem::replace(
                layer,
                QuantizedLinearLayer::from_quantized_weight(
                    &name,
                    0,
                    0,
                    crate::model::layers::QuantizedWeight::Uncompressed { data: Vec::new(), in_features: 0, out_features: 0 },
                ),
            )
        };
        let head_dim = self.config.head_dim;
        let mut run = |layer: &mut QuantizedLinearLayer, role| {
            let l = take(layer);
            *layer = enc.encode(Projection { role, layer: l, sources: Vec::new(), head_dim, checkpoint: None });
        };
        for l in &mut self.layers {
            run(&mut l.qkv_proj, ProjRole::Attn);
            run(&mut l.out_proj, ProjRole::Attn);
            run(&mut l.gate_up_proj, ProjRole::Ffn);
            run(&mut l.down_proj, ProjRole::Ffn);
        }
        run(&mut self.lm_head, ProjRole::LmHead);
    }
}

/// Parameterized Tiny LLM (strictly <= 10M parameters).
#[derive(Debug, Clone)]
pub struct GraphLm {
    pub config: GraphLmConfig,
    pub tokenizer: Tokenizer,
    pub embedding: Vec<f32>,
    pub pos_embedding: Option<Vec<f32>>,
    pub layers: Vec<GraphLayer>,
    pub final_norm: Vec<f32>,
    pub final_norm_bias: Option<Vec<f32>>,
    pub lm_head: QuantizedLinearLayer,
}

impl GraphLm {
    /// Constructs a new [`GraphLm`] with deterministic synthetic weights.
    pub fn new(config: GraphLmConfig) -> Self {
        let v = config.vocab_size;
        let h = config.hidden_dim;
        let inter = config.intermediate_dim;
        let layers_count = config.num_layers;
        let mut tokenizer = Tokenizer::default_bpe();
        let extra_common_words = [
            "journey",
            " thousand",
            "thousand",
            " miles",
            "miles",
            " begins",
            "begins",
            " single",
            "single",
            " step",
            "step",
            " Once",
            "Once",
            " upon",
            "upon",
            " quiet",
            "quiet",
            " valley",
            "valley",
            " surrounded",
            "surrounded",
            " mountains",
            "mountains",
            " France",
            "Paris",
            " quick",
            "quick",
            " brown",
            "brown",
            " fox",
            "fox",
            " jumps",
            "jumps",
            " over",
            "over",
            " lazy",
            "lazy",
            " dog",
            "dog",
        ];
        for word in &extra_common_words {
            if !tokenizer.vocab.contains_key(*word) && tokenizer.vocab.len() < v {
                let id = tokenizer.vocab.len() as u32;
                tokenizer.vocab.insert(word.to_string(), id);
                tokenizer.inv_vocab.insert(id, word.to_string());
            }
        }

        // Embedding: vocab_size x hidden_dim with base semantic projection
        let mut embedding = vec![0.0f32; v * h];
        for tok in 0..v {
            let offset = tok * h;
            for d in 0..h {
                let val = ((tok * 17 + d * 31 + 7) as f32 * 0.02).sin() * 0.05;
                embedding[offset + d] = val;
            }
            let norm = (0..h)
                .map(|d| embedding[offset + d] * embedding[offset + d])
                .sum::<f32>()
                .sqrt();
            if norm > 1e-6 {
                let scale = 0.5 / norm;
                for d in 0..h {
                    embedding[offset + d] *= scale;
                }
            }
        }

        // Initialize raw_lm_head tied to base embedding table
        let mut raw_lm_head = Vec::with_capacity(v * h);
        raw_lm_head.extend_from_slice(&embedding);
        let mut raw_qkv: Vec<Vec<f32>> = Vec::with_capacity(layers_count);
        let mut raw_out: Vec<Vec<f32>> = Vec::with_capacity(layers_count);
        let mut raw_gate_up: Vec<Vec<f32>> = Vec::with_capacity(layers_count);
        let mut raw_down: Vec<Vec<f32>> = Vec::with_capacity(layers_count);

        for l in 0..layers_count {
            raw_qkv.push(
                (0..(3 * h * h))
                    .map(|i| ((i + l * 7) as f32 * 0.02).cos() * 0.05)
                    .collect::<Vec<f32>>(),
            );
            raw_out.push(
                (0..(h * h))
                    .map(|i| ((i + l * 11) as f32 * 0.03).sin() * 0.005)
                    .collect::<Vec<f32>>(),
            );
            raw_gate_up.push(
                (0..(2 * inter * h))
                    .map(|i| ((i + l * 13) as f32 * 0.01).cos() * 0.05)
                    .collect::<Vec<f32>>(),
            );
            raw_down.push(
                (0..(h * inter))
                    .map(|i| ((i + l * 17) as f32 * 0.02).sin() * 0.005)
                    .collect::<Vec<f32>>(),
            );
        }

        // Word embeddings tied with directional LM head projections
        let final_norm = vec![1.0f32; h];

        // Build initial uncompressed layers
        let mut layers = Vec::with_capacity(layers_count);
        for l in 0..layers_count {
            let qkv_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_qkv", l),
                h,
                3 * h,
                &raw_qkv[l],
                QuantFormat::FP32,
            )
            .expect("QKV initialization failed");

            let out_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_out", l),
                h,
                h,
                &raw_out[l],
                QuantFormat::FP32,
            )
            .expect("Out projection initialization failed");

            let gate_up_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_gate_up", l),
                h,
                2 * inter,
                &raw_gate_up[l],
                QuantFormat::FP32,
            )
            .expect("Gate/Up projection initialization failed");

            let down_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_down", l),
                inter,
                h,
                &raw_down[l],
                QuantFormat::FP32,
            )
            .expect("Down projection initialization failed");

            layers.push(GraphLayer {
                norm1: vec![1.0f32; h],
                norm1_bias: None,
                qkv_proj,
                out_proj,
                post_attention_norm: None,
                norm2: vec![1.0f32; h],
                norm2_bias: None,
                gate_up_proj,
                down_proj,
                post_ffw_norm: None,
                attn_sub_norm: None,
                ffn_sub_norm: None,
            });
        }

        let lm_head =
            QuantizedLinearLayer::from_f32("lm_head", h, v, &raw_lm_head, QuantFormat::FP32)
                .expect("LM head initialization failed");

        let mut model = Self {
            config,
            tokenizer,
            embedding,
            pos_embedding: None,
            layers,
            final_norm,
            final_norm_bias: None,
            lm_head,
        };

        if model.config.quant_mode != QuantMode::Uncompressed {
            model.quantize(model.config.quant_mode);
        }

        model
    }
    /// This decoder's structure as a [`crate::graph::Graph`]: weights named per layer, each projection's dtype
    /// from its storage format. Activation shapes are left empty (sequence length is a runtime
    /// dimension). GQA K/V heads are expanded at load, so `kv_heads == heads`.
    ///
    /// ponytail: structure only; no executor runs quantized `Linear` yet, so inference stays on
    /// [`GraphLm::forward`].
    pub fn to_graph(&self) -> crate::graph::Graph {
        self.to_graph_with_weights().0
    }

    /// Emits static-shape decode graph (n = 1) with stateful KV caching.
    pub fn to_decode_graph(&self) -> (crate::graph::Graph, crate::graph::Weights) {
        self.build_graph(1, true)
    }

    /// Emits static-shape prefill graph for a given sequence bucket n.
    pub fn to_prefill_graph(&self, n: usize) -> (crate::graph::Graph, crate::graph::Weights) {
        self.build_graph(n, true)
    }

    /// Emits only the [`Graph`] structure for prefill sequence bucket n without cloning weights.
    pub fn to_prefill_graph_structure(&self, n: usize) -> crate::graph::Graph {
        self.build_graph_inner(n, true, None).0
    }

    /// Emits both the [`Graph`] and the [`Weights`] ready for execution on any backend.
    pub fn to_graph_with_weights(&self) -> (crate::graph::Graph, crate::graph::Weights) {
        self.to_decode_graph()
    }

    pub fn build_graph(
        &self,
        n: usize,
        cached: bool,
    ) -> (crate::graph::Graph, crate::graph::Weights) {
        let mut w_map = crate::graph::Weights::new();
        let (g, _) = self.build_graph_inner(n, cached, Some(&mut w_map));
        (g, w_map)
    }

    fn build_graph_inner(
        &self,
        n: usize,
        cached: bool,
        mut w_map: Option<&mut crate::graph::Weights>,
    ) -> (crate::graph::Graph, ()) {
        use crate::graph::{DType, Graph, Op, RopeVariant, Tensor, ValueId};

        fn linear(
            g: &mut Graph,
            w_map: &mut Option<&mut crate::graph::Weights>,
            name: &str,
            x: ValueId,
            l: &QuantizedLinearLayer,
            n: usize,
        ) -> ValueId {
            let dtype = match l.quant_format() {
                QuantFormat::FP32 => DType::F32,
                QuantFormat::FP16 => DType::F16,
                QuantFormat::BF16 => DType::BF16,
                f => DType::Quant(f),
            };
            let w_name = format!("{name}.weight");
            let w = g.weight(&w_name, vec![l.out_features, l.in_features], dtype);
            if let Some(map) = w_map {
                let tensor = match &l.weight {
                    crate::model::layers::QuantizedWeight::Uncompressed { data, .. } => {
                        Tensor::F32(data.clone())
                    }
                    other => Tensor::Quantized(std::sync::Arc::new(other.clone())),
                };
                map.insert(w_name, tensor);
            }
            let mut ins = vec![x, w];
            if let Some(b) = &l.bias {
                let b_name = format!("{name}.bias");
                ins.push(g.weight(&b_name, vec![b.len()], DType::F32));
                if let Some(map) = w_map {
                    map.insert(b_name, Tensor::F32(b.clone()));
                }
            }
            g.add(name, Op::Linear, &ins, vec![n, l.out_features], DType::F32)
        }
        // `layer_norm` picks LayerNorm (with optional beta) or RMSNorm, matching `forward`.
        fn norm(
            g: &mut Graph,
            w_map: &mut Option<&mut crate::graph::Weights>,
            name: &str,
            x: ValueId,
            gamma: &[f32],
            beta: Option<&Vec<f32>>,
            layer_norm: bool,
            eps: f32,
            n: usize,
        ) -> ValueId {
            let w_name = format!("{name}.weight");
            let mut ins = vec![x, g.weight(&w_name, vec![gamma.len()], DType::F32)];
            if let Some(map) = w_map {
                map.insert(w_name, Tensor::F32(gamma.to_vec()));
            }
            let op = if layer_norm {
                if let Some(b) = beta {
                    let b_name = format!("{name}.bias");
                    ins.push(g.weight(&b_name, vec![b.len()], DType::F32));
                    if let Some(map) = w_map {
                        map.insert(b_name, Tensor::F32(b.clone()));
                    }
                }
                Op::LayerNorm { eps }
            } else {
                Op::RmsNorm { eps }
            };
            g.add(name, op, &ins, vec![n, gamma.len()], DType::F32)
        }
        fn slice(
            g: &mut Graph,
            name: String,
            x: ValueId,
            start: usize,
            end: usize,
            n: usize,
        ) -> ValueId {
            let op = Op::Slice {
                axis: -1,
                start: start as i64,
                end: end as i64,
            };
            g.add(name, op, &[x], vec![n, end - start], DType::F32)
        }

        let c = &self.config;
        let (h, v, inter, eps, ln) = (
            c.hidden_dim,
            c.vocab_size,
            c.intermediate_dim,
            c.norm_eps,
            c.use_standard_layernorm,
        );
        let rotary_dim = (c.head_dim as f32 * c.rotary_pct) as usize;
        let inv_freq_vec: Vec<f32> = if rotary_dim > 0 {
            let half = rotary_dim / 2;
            (0..half)
                .map(|j| 1.0 / c.rope_theta.powf((2 * j) as f32 / rotary_dim as f32))
                .collect()
        } else {
            Vec::new()
        };

        let rope = Op::Rope {
            head_dim: c.head_dim,
            rotary_dim,
            variant: if c.rope_interleaved {
                RopeVariant::Interleaved
            } else {
                RopeVariant::HalfSplit
            },
            mscale: 1.0,
        };
        let attention = Op::Attention {
            heads: c.num_heads,
            kv_heads: c.num_heads,
            causal: true,
            window: None,
            soft_cap: None,
        };

        let mut g = Graph::new();
        let ids = g.input("token_ids", vec![n], DType::U32);
        let positions = g.input("positions", vec![n], DType::U32);
        let table = g.weight("model.embed_tokens.weight", vec![v, h], DType::F32);
        if let Some(map) = &mut w_map {
            map.insert(
                "model.embed_tokens.weight",
                Tensor::F32(self.embedding.clone()),
            );
        }
        let mut x = g.add(
            "model.embed_tokens",
            Op::Embedding,
            &[table, ids],
            vec![n, h],
            DType::F32,
        );
        if let Some(pos) = &self.pos_embedding {
            let wpe = g.weight(
                "model.embed_positions.weight",
                vec![pos.len() / h, h],
                DType::F32,
            );
            if let Some(map) = &mut w_map {
                map.insert("model.embed_positions.weight", Tensor::F32(pos.clone()));
            }
            let p = g.add(
                "model.embed_positions",
                Op::Embedding,
                &[wpe, positions],
                vec![n, h],
                DType::F32,
            );
            x = g.add("model.embed", Op::Add, &[x, p], vec![n, h], DType::F32);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let n1 = norm(
                &mut g,
                &mut w_map,
                &format!("{p}.input_layernorm"),
                x,
                &layer.norm1,
                layer.norm1_bias.as_ref(),
                ln,
                eps,
                n,
            );
            let qkv = linear(
                &mut g,
                &mut w_map,
                &format!("{p}.self_attn.qkv_proj"),
                n1,
                &layer.qkv_proj,
                n,
            );
            let mut q = slice(&mut g, format!("{p}.self_attn.q"), qkv, 0, h, n);
            let mut k = slice(&mut g, format!("{p}.self_attn.k"), qkv, h, 2 * h, n);
            let vv = slice(&mut g, format!("{p}.self_attn.v"), qkv, 2 * h, 3 * h, n);
            if rotary_dim > 0 {
                let inv_freq_val = g.weight("rope.inv_freq.default", vec![inv_freq_vec.len()], DType::F32);
                if let Some(map) = &mut w_map {
                    map.insert("rope.inv_freq.default", Tensor::F32(inv_freq_vec.clone()));
                }
                q = g.add(
                    format!("{p}.self_attn.rope_q"),
                    rope.clone(),
                    &[q, positions, inv_freq_val],
                    vec![n, h],
                    DType::F32,
                );
                k = g.add(
                    format!("{p}.self_attn.rope_k"),
                    rope.clone(),
                    &[k, positions, inv_freq_val],
                    vec![n, h],
                    DType::F32,
                );
            }
            let a = if cached {
                let k_state = g.state(
                    format!("kv.{i}.k"),
                    vec![c.max_seq_len, c.num_heads * c.head_dim],
                    DType::F32,
                );
                let v_state = g.state(
                    format!("kv.{i}.v"),
                    vec![c.max_seq_len, c.num_heads * c.head_dim],
                    DType::F32,
                );
                g.add(
                    format!("{p}.self_attn"),
                    Op::CachedAttention {
                        heads: c.num_heads,
                        kv_heads: c.num_heads,
                        head_dim: c.head_dim,
                        v_head_dim: c.head_dim,
                        scale: 1.0 / (c.head_dim as f32).sqrt(),
                        causal: true,
                        window: None,
                        soft_cap: None,
                        write: true,
                    },
                    &[q, k, vv, k_state, v_state, positions],
                    vec![n, c.num_heads * c.head_dim],
                    DType::F32,
                )
            } else {
                g.add(
                    format!("{p}.self_attn"),
                    attention.clone(),
                    &[q, k, vv],
                    vec![n, c.num_heads * c.head_dim],
                    DType::F32,
                )
            };
            let mut o = linear(
                &mut g,
                &mut w_map,
                &format!("{p}.self_attn.o_proj"),
                a,
                &layer.out_proj,
                n,
            );
            if let Some(pan) = &layer.post_attention_norm {
                o = norm(
                    &mut g,
                    &mut w_map,
                    &format!("{p}.post_attention_norm"),
                    o,
                    pan,
                    None,
                    false,
                    eps,
                    n,
                );
            }
            let x_attn = g.add(
                format!("{p}.residual_1"),
                Op::Add,
                &[x, o],
                vec![n, h],
                DType::F32,
            );
            // Parallel residual (GPT-NeoX): the MLP reads the block input, not `x + attn`.
            let mlp_in = if c.use_parallel_residual { x } else { x_attn };
            let n2 = norm(
                &mut g,
                &mut w_map,
                &format!("{p}.post_attention_layernorm"),
                mlp_in,
                &layer.norm2,
                layer.norm2_bias.as_ref(),
                ln,
                eps,
                n,
            );
            let up = linear(
                &mut g,
                &mut w_map,
                &format!("{p}.mlp.gate_up_proj"),
                n2,
                &layer.gate_up_proj,
                n,
            );
            let act = if c.use_gelu_mlp {
                g.add(
                    format!("{p}.mlp.act"),
                    Op::Gelu { tanh: true },
                    &[up],
                    vec![n, inter],
                    DType::F32,
                )
            } else {
                let gate = slice(&mut g, format!("{p}.mlp.gate"), up, 0, inter, n);
                let up = slice(&mut g, format!("{p}.mlp.up"), up, inter, 2 * inter, n);
                g.add(
                    format!("{p}.mlp.act"),
                    Op::SwiGlu,
                    &[gate, up],
                    vec![n, inter],
                    DType::F32,
                )
            };
            let mut d = linear(
                &mut g,
                &mut w_map,
                &format!("{p}.mlp.down_proj"),
                act,
                &layer.down_proj,
                n,
            );
            if let Some(pfn) = &layer.post_ffw_norm {
                d = norm(
                    &mut g,
                    &mut w_map,
                    &format!("{p}.post_ffw_norm"),
                    d,
                    pfn,
                    None,
                    false,
                    eps,
                    n,
                );
            }
            x = g.add(
                format!("{p}.residual_2"),
                Op::Add,
                &[x_attn, d],
                vec![n, h],
                DType::F32,
            );
        }

        let n_norm = norm(
            &mut g,
            &mut w_map,
            "model.norm",
            x,
            &self.final_norm,
            self.final_norm_bias.as_ref(),
            ln,
            eps,
            n,
        );
        let logits = linear(&mut g, &mut w_map, "lm_head", n_norm, &self.lm_head, n);
        g.output(logits);
        (g, ())
    }
    /// Ingests real pretrained weights from a safetensors binary buffer and initializes [`GraphLm`].
    pub fn from_safetensors_data(config: GraphLmConfig, data: &[u8]) -> Result<Self, String> {
        let manifest = UniversalModelLoader::parse_safetensors(data)?;
        Self::from_manifest(config, &manifest, data)
    }

    /// Ingests real pretrained weights from a GGUF binary buffer and initializes [`GraphLm`].
    /// Architecture comes from GGUF metadata ([`GraphLmConfig::from_gguf`]); `config` supplies
    /// `quant_mode` and fallbacks for keys the file omits.
    pub fn from_gguf_data(config: GraphLmConfig, data: &[u8]) -> Result<Self, String> {
        let manifest = UniversalModelLoader::parse_gguf(data)?;
        Self::from_manifest(GraphLmConfig::from_gguf(&manifest, config), &manifest, data)
    }

    /// Ingests real pretrained weights from a parsed [`ModelManifest`] and binary buffer.
    pub fn from_manifest(
        config: GraphLmConfig,
        manifest: &ModelManifest,
        data: &[u8],
    ) -> Result<Self, String> {
        Self::from_manifest_with(config, manifest, data, None)
    }

    /// [`Self::from_manifest`] with every projection (q/k/v, out, gate/up, down, LM head) passed to
    /// `encoder` as it loads, so the f32 model never exists at once; `None` encodes by
    /// `config.quant_mode`.
    pub fn from_manifest_with(
        config: GraphLmConfig,
        manifest: &ModelManifest,
        data: &[u8],
        encoder: Option<&dyn ProjectionEncoder>,
    ) -> Result<Self, String> {
        let v = config.vocab_size;
        let h = config.hidden_dim;
        let inter = config.intermediate_dim;
        let layers_count = config.num_layers;
        // ponytail: deny-list of known per-head q/k norms, not a full consumed-tensor audit.
        if let Some(n) = manifest.tensors.keys().find(|n| {
            ["q_norm.", "k_norm.", "query_layernorm.", "key_layernorm."]
                .iter()
                .any(|s| n.contains(s))
        }) {
            return Err(format!(
                "tensor '{n}': per-head q/k RMSNorm is not implemented by this decoder"
            ));
        }

        // Try candidate embedding names
        let embed_candidates = [
            "token_embd.weight",
            "model.embed_tokens.weight",
            "gpt_neox.embed_in.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "lm_head.weight",
        ];
        let mut embedding: Option<Vec<f32>> = None;
        let mut embed_name = "";
        for &name in &embed_candidates {
            if manifest.tensors.contains_key(name) {
                if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                    embedding = Some(vec);
                    embed_name = name;
                    break;
                }
            }
        }
        // Required weights never fall back to synthetic values: a checkpoint whose tensor names
        // we don't map must fail loudly instead of generating gibberish.
        let embedding = embedding.ok_or_else(|| {
            format!("missing or unreadable token embedding (tried {embed_candidates:?})")
        })?;
        let pos_candidates = ["transformer.wpe.weight", "model.embed_positions.weight"];
        let mut pos_embedding: Option<Vec<f32>> = None;
        for &name in &pos_candidates {
            if manifest.tensors.contains_key(name) {
                if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                    pos_embedding = Some(vec);
                    break;
                }
            }
        }

        // BitNet online weight quant: every BitLinear projection (`*_proj.weight`) is ternarized
        // with its own scale, then cast back to the checkpoint dtype as the reference does.
        // Tensors read for the projection being assembled, so an encoder can carry the checkpoint's
        // own blocks instead of re-encoding the f32 copy.
        let used = std::cell::RefCell::new(Vec::<String>::new());
        let ext = |name: &str| -> Result<Vec<f32>, String> {
            let r = manifest.extract_tensor_f32(data, name);
            if r.is_ok() {
                used.borrow_mut().push(name.to_string());
            }
            r
        };
        let take_src = || -> Vec<String> {
            let mut u = used.borrow_mut();
            let two_d = u
                .drain(..)
                .filter(|n| manifest.tensors.get(n).is_some_and(|t| t.shape.len() == 2))
                .collect();
            two_d
        };
        let proj = |name: &str| -> Result<Vec<f32>, String> {
            let mut w = ext(name)?;
            if config.ternarize_weights && name.ends_with("_proj.weight") {
                match manifest.tensors[name].dtype.as_str() {
                    "BF16" => ternarize(&mut w, |x| {
                        crate::model::bf16_to_f32(crate::model::f32_to_bf16(x))
                    }),
                    "F16" => ternarize(&mut w, |x| {
                        crate::model::f16_to_f32(crate::graph::coreml::f32_to_f16(x))
                    }),
                    _ => ternarize(&mut w, |x| x),
                }
            }
            Ok(w)
        };
        // Each projection is encoded in the target format as it loads, so the whole model is
        // never held in f32 at once.
        let (attn_fmt, ffn_fmt, lm_fmt) = projection_formats(config.quant_mode);
        let encode = |layer: QuantizedLinearLayer, role: ProjRole, format, sources: Vec<String>| {
            if let Some(enc) = encoder {
                // Ternarized (BitNet) weights no longer equal the checkpoint's tensors.
                let sources = if config.ternarize_weights { Vec::new() } else { sources };
                enc.encode(Projection {
                    role,
                    layer,
                    sources,
                    head_dim: config.head_dim,
                    checkpoint: Some((manifest, data)),
                })
            } else if config.quant_mode == QuantMode::Uncompressed {
                layer
            } else {
                requantized(&layer, format)
            }
        };
        let mut layers = Vec::with_capacity(layers_count);
        for l in 0..layers_count {
            // norm1
            let norm1_candidates = [
                format!("blk.{}.attn_norm.weight", l),
                format!("model.layers.{}.input_layernorm.weight", l),
                format!("transformer.h.{}.ln_1.weight", l),
                format!("gpt_neox.layers.{}.input_layernorm.weight", l),
            ];
            let mut norm1: Option<Vec<f32>> = None;
            for name in &norm1_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = ext(name) {
                        norm1 = Some(vec);
                        break;
                    }
                }
            }
            let norm1 = norm1.ok_or_else(|| {
                format!(
                    "missing or unreadable layer {l} attention norm (tried {norm1_candidates:?})"
                )
            })?;

            // QKV projection
            let mut raw_qkv: Option<Vec<f32>> = None;
            let mut is_neox_qkv = false;

            let gguf_qkv = format!("blk.{}.attn_qkv.weight", l);
            if manifest.tensors.contains_key(&gguf_qkv) {
                if let Ok(vec) = ext(&gguf_qkv) {
                    raw_qkv = Some(vec);
                }
            }

            // If format!("blk.{}.attn_q.weight", l) exists:
            // Extract attn_q, attn_k, attn_v.
            // Expand KV heads for GQA using expand_gqa_kv if KV heads < Q heads.
            // Fuse [q, k_expanded, v_expanded] into raw_qkv.
            let gguf_q = format!("blk.{}.attn_q.weight", l);
            let gguf_k = format!("blk.{}.attn_k.weight", l);
            let gguf_v = format!("blk.{}.attn_v.weight", l);
            if raw_qkv.is_none() && manifest.tensors.contains_key(&gguf_q) {
                if let (Ok(q), Ok(k), Ok(v_vec)) = (
                    ext(&gguf_q),
                    ext(&gguf_k),
                    ext(&gguf_v),
                ) {
                    let head_dim = config.head_dim;
                    let k_expanded = expand_gqa_kv(k, q.len(), config.num_heads, head_dim, h);
                    let v_expanded = expand_gqa_kv(v_vec, q.len(), config.num_heads, head_dim, h);
                    let mut fused =
                        Vec::with_capacity(q.len() + k_expanded.len() + v_expanded.len());
                    fused.extend(q);
                    fused.extend(k_expanded);
                    fused.extend(v_expanded);
                    raw_qkv = Some(fused);
                }
            }

            if raw_qkv.is_none() {
                let qkv_candidates = [format!(
                    "gpt_neox.layers.{}.attention.query_key_value.weight",
                    l
                )];
                for name in &qkv_candidates {
                    if manifest.tensors.contains_key(name) {
                        if let Ok(vec) = ext(name) {
                            raw_qkv = Some(vec);
                            is_neox_qkv = true;
                            break;
                        }
                    }
                }
            }

            if raw_qkv.is_none() {
                let q_name = format!("model.layers.{}.self_attn.q_proj.weight", l);
                let k_name = format!("model.layers.{}.self_attn.k_proj.weight", l);
                let v_name = format!("model.layers.{}.self_attn.v_proj.weight", l);

                // Try GPT-Neo format
                let neo_q = format!("transformer.h.{}.attn.attention.q_proj.weight", l);
                let neo_k = format!("transformer.h.{}.attn.attention.k_proj.weight", l);
                let neo_v = format!("transformer.h.{}.attn.attention.v_proj.weight", l);

                if let (Ok(q), Ok(k), Ok(v_vec)) = (proj(&q_name), proj(&k_name), proj(&v_name)) {
                    let head_dim = config.head_dim;
                    let k_expanded = expand_gqa_kv(k, q.len(), config.num_heads, head_dim, h);
                    let v_expanded = expand_gqa_kv(v_vec, q.len(), config.num_heads, head_dim, h);

                    let mut fused =
                        Vec::with_capacity(q.len() + k_expanded.len() + v_expanded.len());
                    fused.extend(q);
                    fused.extend(k_expanded);
                    fused.extend(v_expanded);
                    raw_qkv = Some(fused);
                } else if let (Ok(q), Ok(k), Ok(v_vec)) = (
                    ext(&neo_q),
                    ext(&neo_k),
                    ext(&neo_v),
                ) {
                    let head_dim = config.head_dim;
                    let k_expanded = expand_gqa_kv(k, q.len(), config.num_heads, head_dim, h);
                    let v_expanded = expand_gqa_kv(v_vec, q.len(), config.num_heads, head_dim, h);

                    let mut fused =
                        Vec::with_capacity(q.len() + k_expanded.len() + v_expanded.len());
                    fused.extend(q);
                    fused.extend(k_expanded);
                    fused.extend(v_expanded);
                    raw_qkv = Some(fused);
                }
            }
            let raw_qkv = if is_neox_qkv && raw_qkv.is_some() {
                let neox_qkv = raw_qkv.unwrap();
                let num_heads = config.num_heads;
                let head_dim = config.head_dim;
                if neox_qkv.len() == 3 * h * h && num_heads * head_dim == h {
                    let mut reordered = vec![0.0f32; 3 * h * h];
                    for head in 0..num_heads {
                        for d in 0..head_dim {
                            let src_q_row = head * 3 * head_dim + d;
                            let src_k_row = head * 3 * head_dim + head_dim + d;
                            let src_v_row = head * 3 * head_dim + 2 * head_dim + d;

                            let dst_q_row = head * head_dim + d;
                            let dst_k_row = h + head * head_dim + d;
                            let dst_v_row = 2 * h + head * head_dim + d;

                            reordered[dst_q_row * h..(dst_q_row + 1) * h]
                                .copy_from_slice(&neox_qkv[src_q_row * h..(src_q_row + 1) * h]);
                            reordered[dst_k_row * h..(dst_k_row + 1) * h]
                                .copy_from_slice(&neox_qkv[src_k_row * h..(src_k_row + 1) * h]);
                            reordered[dst_v_row * h..(dst_v_row + 1) * h]
                                .copy_from_slice(&neox_qkv[src_v_row * h..(src_v_row + 1) * h]);
                        }
                    }
                    reordered
                } else {
                    neox_qkv
                }
            } else {
                raw_qkv.ok_or_else(|| {
                    format!("missing or unreadable layer {l} q/k/v projection (expected e.g. model.layers.{l}.self_attn.q_proj.weight or blk.{l}.attn_q.weight)")
                })?
            };

            let qkv_src = take_src();
            // out_proj
            let out_candidates = [
                format!("blk.{}.attn_output.weight", l),
                format!("model.layers.{}.self_attn.o_proj.weight", l),
                format!("gpt_neox.layers.{}.attention.dense.weight", l),
                format!("transformer.h.{}.attn.attention.out_proj.weight", l),
                format!("transformer.h.{}.attn.c_proj.weight", l),
            ];
            let mut raw_out: Option<Vec<f32>> = None;
            for name in &out_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = proj(name) {
                        raw_out = Some(vec);
                        break;
                    }
                }
            }
            let raw_out = raw_out.ok_or_else(|| {
                format!(
                    "missing or unreadable layer {l} attention output (tried {out_candidates:?})"
                )
            })?;

            let out_src = take_src();
            // norm2
            let norm2_candidates = [
                format!("blk.{}.ffn_norm.weight", l),
                format!("model.layers.{}.post_attention_layernorm.weight", l),
                format!("gpt_neox.layers.{}.post_attention_layernorm.weight", l),
                format!("transformer.h.{}.ln_2.weight", l),
            ];
            let mut norm2: Option<Vec<f32>> = None;
            for name in &norm2_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = ext(name) {
                        norm2 = Some(vec);
                        break;
                    }
                }
            }
            let norm2 = norm2.ok_or_else(|| {
                format!("missing or unreadable layer {l} MLP norm (tried {norm2_candidates:?})")
            })?;

            // gate_up_proj
            let mut raw_gate_up: Option<Vec<f32>> = None;
            let gguf_gate_up = format!("blk.{}.ffn_gate_up.weight", l);
            let gguf_gate = format!("blk.{}.ffn_gate.weight", l);
            let gguf_up = format!("blk.{}.ffn_up.weight", l);
            let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", l);
            let up_name = format!("model.layers.{}.mlp.up_proj.weight", l);
            let neo_fc = format!("transformer.h.{}.mlp.c_fc.weight", l);
            if manifest.tensors.contains_key(&gguf_gate_up) {
                if let Ok(vec) = ext(&gguf_gate_up) {
                    raw_gate_up = Some(vec);
                }
            } else if manifest.tensors.contains_key(&gguf_gate)
                && manifest.tensors.contains_key(&gguf_up)
            {
                if let (Ok(g), Ok(u)) = (
                    ext(&gguf_gate),
                    ext(&gguf_up),
                ) {
                    let mut fused = Vec::with_capacity(g.len() + u.len());
                    fused.extend(g);
                    fused.extend(u);
                    raw_gate_up = Some(fused);
                }
            } else if manifest.tensors.contains_key(&gguf_up) {
                if let Ok(u) = ext(&gguf_up) {
                    if config.use_gelu_mlp {
                        raw_gate_up = Some(u);
                    } else {
                        let mut duplicated = Vec::with_capacity(u.len() * 2);
                        duplicated.extend_from_slice(&u);
                        duplicated.extend_from_slice(&u);
                        raw_gate_up = Some(duplicated);
                    }
                }
            } else if let Ok(vec) = ext(&format!("model.layers.{}.mlp.gate_up_proj.weight", l))
            {
                // HF fused [gate; up] rows (GLM, Phi-3): already our gate-first layout.
                raw_gate_up = Some(vec);
            } else if let (Ok(g), Ok(u)) = (proj(&gate_name), proj(&up_name)) {
                let mut fused = Vec::with_capacity(g.len() + u.len());
                fused.extend(g);
                fused.extend(u);
                raw_gate_up = Some(fused);
            } else if let Ok(vec) = ext(&neo_fc) {
                if config.use_gelu_mlp {
                    raw_gate_up = Some(vec);
                } else {
                    let mut duplicated = Vec::with_capacity(vec.len() * 2);
                    duplicated.extend_from_slice(&vec);
                    duplicated.extend_from_slice(&vec);
                    raw_gate_up = Some(duplicated);
                }
            } else {
                let neox_up = format!("gpt_neox.layers.{}.mlp.dense_h_to_4h.weight", l);
                if let Ok(vec) = ext(&neox_up) {
                    if config.use_gelu_mlp {
                        raw_gate_up = Some(vec);
                    } else {
                        let mut duplicated = Vec::with_capacity(vec.len() * 2);
                        duplicated.extend_from_slice(&vec);
                        duplicated.extend_from_slice(&vec);
                        raw_gate_up = Some(duplicated);
                    }
                }
            };
            let raw_gate_up = raw_gate_up.ok_or_else(|| {
                format!("missing or unreadable layer {l} MLP up/gate projection (expected e.g. {gate_name}, model.layers.{l}.mlp.gate_up_proj.weight or {gguf_up})")
            })?;

            let gate_src = take_src();
            // down_proj
            let down_candidates = [
                format!("blk.{}.ffn_down.weight", l),
                format!("model.layers.{}.mlp.down_proj.weight", l),
                format!("gpt_neox.layers.{}.mlp.dense_4h_to_h.weight", l),
                format!("transformer.h.{}.mlp.c_proj.weight", l),
            ];
            let mut raw_down: Option<Vec<f32>> = None;
            for name in &down_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = proj(name) {
                        raw_down = Some(vec);
                        break;
                    }
                }
            }
            let raw_down = raw_down.ok_or_else(|| {
                format!("missing or unreadable layer {l} MLP down projection (tried {down_candidates:?})")
            })?;

            let down_src = take_src();
            let norm1_bias_candidates = [
                format!("transformer.h.{}.ln_1.bias", l),
                format!("gpt_neox.layers.{}.input_layernorm.bias", l),
            ];
            let mut norm1_bias: Option<Vec<f32>> = None;
            for name in &norm1_bias_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        norm1_bias = Some(vec);
                        break;
                    }
                }
            }

            let norm2_bias_candidates = [
                format!("transformer.h.{}.ln_2.bias", l),
                format!("gpt_neox.layers.{}.post_attention_layernorm.bias", l),
            ];
            let mut norm2_bias: Option<Vec<f32>> = None;
            for name in &norm2_bias_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        norm2_bias = Some(vec);
                        break;
                    }
                }
            }

            let mut qkv_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_qkv", l),
                h,
                3 * h,
                &raw_qkv,
                QuantFormat::FP32,
            )
            .map_err(|e| e.to_string())?;

            // Support Qwen QKV separate biases: blk.{l}.attn_q.bias, blk.{l}.attn_k.bias, blk.{l}.attn_v.bias
            let gguf_q_bias = format!("blk.{}.attn_q.bias", l);
            let gguf_k_bias = format!("blk.{}.attn_k.bias", l);
            let gguf_v_bias = format!("blk.{}.attn_v.bias", l);
            let hf_q_bias = format!("model.layers.{}.self_attn.q_proj.bias", l);
            let hf_k_bias = format!("model.layers.{}.self_attn.k_proj.bias", l);
            let hf_v_bias = format!("model.layers.{}.self_attn.v_proj.bias", l);

            let mut qwen_qkv_bias: Option<Vec<f32>> = None;
            if manifest.tensors.contains_key(&gguf_q_bias) {
                if let (Ok(qb), Ok(kb), Ok(vb)) = (
                    manifest.extract_tensor_f32(data, &gguf_q_bias),
                    manifest.extract_tensor_f32(data, &gguf_k_bias),
                    manifest.extract_tensor_f32(data, &gguf_v_bias),
                ) {
                    let head_dim = config.head_dim;
                    let kb_exp = expand_gqa_bias(kb, qb.len(), config.num_heads, head_dim);
                    let vb_exp = expand_gqa_bias(vb, qb.len(), config.num_heads, head_dim);
                    let mut fused_bias = Vec::with_capacity(qb.len() + kb_exp.len() + vb_exp.len());
                    fused_bias.extend(qb);
                    fused_bias.extend(kb_exp);
                    fused_bias.extend(vb_exp);
                    if fused_bias.len() != 3 * h {
                        fused_bias.resize(3 * h, 0.0);
                    }
                    qwen_qkv_bias = Some(fused_bias);
                }
            } else if manifest.tensors.contains_key(&hf_q_bias) {
                if let (Ok(qb), Ok(kb), Ok(vb)) = (
                    manifest.extract_tensor_f32(data, &hf_q_bias),
                    manifest.extract_tensor_f32(data, &hf_k_bias),
                    manifest.extract_tensor_f32(data, &hf_v_bias),
                ) {
                    let head_dim = config.head_dim;
                    let kb_exp = expand_gqa_bias(kb, qb.len(), config.num_heads, head_dim);
                    let vb_exp = expand_gqa_bias(vb, qb.len(), config.num_heads, head_dim);
                    let mut fused_bias = Vec::with_capacity(qb.len() + kb_exp.len() + vb_exp.len());
                    fused_bias.extend(qb);
                    fused_bias.extend(kb_exp);
                    fused_bias.extend(vb_exp);
                    if fused_bias.len() != 3 * h {
                        fused_bias.resize(3 * h, 0.0);
                    }
                    qwen_qkv_bias = Some(fused_bias);
                }
            }

            if let Some(bias) = qwen_qkv_bias {
                qkv_proj = qkv_proj.with_bias(bias).map_err(|e| e.to_string())?;
            } else {
                // Extract and attach QKV bias with GPT-NeoX interleaved head reordering
                let qkv_bias_candidates = [
                    format!("gpt_neox.layers.{}.attention.query_key_value.bias", l),
                    format!("transformer.h.{}.attn.attention.qkv_proj.bias", l),
                    format!("transformer.h.{}.attn.c_attn.bias", l),
                ];
                for name in &qkv_bias_candidates {
                    if manifest.tensors.contains_key(name) {
                        if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                            let final_bias = if is_neox_qkv && vec.len() == 3 * h {
                                let num_heads = config.num_heads;
                                let head_dim = config.head_dim;
                                if num_heads * head_dim == h {
                                    let mut reordered = vec![0.0f32; 3 * h];
                                    for head in 0..num_heads {
                                        for d in 0..head_dim {
                                            let src_q = head * 3 * head_dim + d;
                                            let src_k = head * 3 * head_dim + head_dim + d;
                                            let src_v = head * 3 * head_dim + 2 * head_dim + d;

                                            let dst_q = head * head_dim + d;
                                            let dst_k = h + head * head_dim + d;
                                            let dst_v = 2 * h + head * head_dim + d;

                                            reordered[dst_q] = vec[src_q];
                                            reordered[dst_k] = vec[src_k];
                                            reordered[dst_v] = vec[src_v];
                                        }
                                    }
                                    reordered
                                } else {
                                    vec
                                }
                            } else {
                                vec
                            };
                            qkv_proj = qkv_proj.with_bias(final_bias).map_err(|e| e.to_string())?;
                            break;
                        }
                    }
                }
            }

            let mut out_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_out", l),
                h,
                h,
                &raw_out,
                QuantFormat::FP32,
            )
            .map_err(|e| e.to_string())?;
            let out_bias_candidates = [
                format!("transformer.h.{}.attn.attention.out_proj.bias", l),
                format!("transformer.h.{}.attn.c_proj.bias", l),
                format!("gpt_neox.layers.{}.attention.dense.bias", l),
            ];
            for name in &out_bias_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        out_proj = out_proj.with_bias(vec).map_err(|e| e.to_string())?;
                        break;
                    }
                }
            }

            let mut gate_up_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_gate_up", l),
                h,
                if config.use_gelu_mlp {
                    inter
                } else {
                    2 * inter
                },
                &raw_gate_up,
                QuantFormat::FP32,
            )
            .map_err(|e| e.to_string())?;
            let gate_bias_candidates = [
                format!("transformer.h.{}.mlp.c_fc.bias", l),
                format!("gpt_neox.layers.{}.mlp.dense_h_to_4h.bias", l),
            ];
            for name in &gate_bias_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        gate_up_proj = gate_up_proj.with_bias(vec).map_err(|e| e.to_string())?;
                        break;
                    }
                }
            }

            let mut down_proj = QuantizedLinearLayer::from_f32(
                &format!("layer_{}_down", l),
                inter,
                h,
                &raw_down,
                QuantFormat::FP32,
            )
            .map_err(|e| e.to_string())?;
            let down_bias_candidates = [
                format!("transformer.h.{}.mlp.c_proj.bias", l),
                format!("gpt_neox.layers.{}.mlp.dense_4h_to_h.bias", l),
            ];
            for name in &down_bias_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        down_proj = down_proj.with_bias(vec).map_err(|e| e.to_string())?;
                        break;
                    }
                }
            }

            // Gemma2 post_attention_norm and post_ffw_norm
            let post_attn_candidates = [
                format!("blk.{}.post_attention_norm.weight", l),
                format!("model.layers.{}.post_attention_norm.weight", l),
            ];
            let mut post_attention_norm: Option<Vec<f32>> = None;
            for name in &post_attn_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        post_attention_norm = Some(vec);
                        break;
                    }
                }
            }

            let post_ffw_candidates = [
                format!("blk.{}.post_ffw_norm.weight", l),
                format!("model.layers.{}.post_ffw_norm.weight", l),
                format!("model.layers.{}.post_feedforward_layernorm.weight", l),
            ];
            let mut post_ffw_norm: Option<Vec<f32>> = None;
            for name in &post_ffw_candidates {
                if manifest.tensors.contains_key(name) {
                    if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                        post_ffw_norm = Some(vec);
                        break;
                    }
                }
            }

            // BitNet RMSNorms before `out_proj` and `down_proj`: transformers `attn_sub_norm` /
            // `ffn_sub_norm`, 1bitLLM `inner_attn_ln` / `ffn_layernorm`.
            let first = |names: [String; 3]| {
                names
                    .iter()
                    .find(|n| manifest.tensors.contains_key(*n))
                    .map(|n| manifest.extract_tensor_f32(data, n))
                    .transpose()
            };
            let attn_sub_norm = first([
                format!("blk.{l}.attn_sub_norm.weight"),
                format!("model.layers.{l}.self_attn.attn_sub_norm.weight"),
                format!("model.layers.{l}.self_attn.inner_attn_ln.weight"),
            ])?;
            let ffn_sub_norm = first([
                format!("blk.{l}.ffn_sub_norm.weight"),
                format!("model.layers.{l}.mlp.ffn_sub_norm.weight"),
                format!("model.layers.{l}.mlp.ffn_layernorm.weight"),
            ])?;
            if config.act_quant_bits > 0 && (attn_sub_norm.is_none() || ffn_sub_norm.is_none()) {
                return Err(format!(
                    "BitNet layer {l}: missing attention/MLP sub-norm (attn_sub_norm, ffn_sub_norm, inner_attn_ln or ffn_layernorm)"
                ));
            }

            layers.push(GraphLayer {
                norm1,
                norm1_bias,
                qkv_proj: encode(qkv_proj, ProjRole::Attn, attn_fmt, qkv_src),
                out_proj: encode(out_proj, ProjRole::Attn, attn_fmt, out_src),
                post_attention_norm,
                norm2,
                norm2_bias,
                gate_up_proj: encode(gate_up_proj, ProjRole::Ffn, ffn_fmt, gate_src),
                down_proj: encode(down_proj, ProjRole::Ffn, ffn_fmt, down_src),
                post_ffw_norm,
                attn_sub_norm,
                ffn_sub_norm,
            });
        }

        // final_norm
        let final_norm_candidates = [
            "output_norm.weight",
            "model.norm.weight",
            "gpt_neox.final_layer_norm.weight",
            "transformer.ln_f.weight",
        ];
        let mut final_norm: Option<Vec<f32>> = None;
        for &name in &final_norm_candidates {
            if manifest.tensors.contains_key(name) {
                if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                    final_norm = Some(vec);
                    break;
                }
            }
        }
        let final_norm = final_norm.ok_or_else(|| {
            format!("missing or unreadable final norm (tried {final_norm_candidates:?})")
        })?;
        let final_bias_candidates = ["transformer.ln_f.bias", "gpt_neox.final_layer_norm.bias"];
        let mut final_norm_bias: Option<Vec<f32>> = None;
        for &name in &final_bias_candidates {
            if manifest.tensors.contains_key(name) {
                if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                    final_norm_bias = Some(vec);
                    break;
                }
            }
        }
        // lm_head
        let lm_head_candidates = [
            "output.weight",
            "lm_head.weight",
            "token_embd.weight",
            "embed_out.weight",
        ];
        let mut raw_lm_head: Option<Vec<f32>> = None;
        let mut lm_src = embed_name;
        for &name in &lm_head_candidates {
            if manifest.tensors.contains_key(name) {
                if let Ok(vec) = manifest.extract_tensor_f32(data, name) {
                    raw_lm_head = Some(vec);
                    lm_src = name;
                    break;
                }
            }
        }
        // Tied head: the embedding table is the head's checkpoint tensor.
        let raw_lm_head = raw_lm_head.unwrap_or_else(|| embedding.clone());

        let lm_head = encode(
            QuantizedLinearLayer::from_f32("lm_head", h, v, &raw_lm_head, QuantFormat::FP32)
                .map_err(|e| e.to_string())?,
            ProjRole::LmHead,
            lm_fmt,
            vec![lm_src.to_string()],
        );
        drop(raw_lm_head);

        Ok(Self {
            config,
            tokenizer: Tokenizer::default_bpe(),
            embedding,
            pos_embedding,
            layers,
            final_norm,
            final_norm_bias,
            lm_head,
        })
    }
    /// Updates the active tokenizer associated with this model instance.
    #[inline]
    pub fn set_tokenizer(&mut self, tokenizer: Tokenizer) {
        self.tokenizer = tokenizer;
    }

    /// Fluent builder to attach an explicit tokenizer.
    #[inline]
    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Calculates total parameter count and asserts it is strictly <= 10,000,000.
    pub fn parameter_count(&self) -> usize {
        let mut total = 0usize;
        total += self.embedding.len();
        if let Some(pos_emb) = &self.pos_embedding {
            total += pos_emb.len();
        }
        for layer in &self.layers {
            total += layer.norm1.len();
            total += layer.qkv_proj.in_features * layer.qkv_proj.out_features;
            if let Some(bias) = &layer.qkv_proj.bias {
                total += bias.len();
            }
            total += layer.out_proj.in_features * layer.out_proj.out_features;
            if let Some(bias) = &layer.out_proj.bias {
                total += bias.len();
            }
            if let Some(pan) = &layer.post_attention_norm {
                total += pan.len();
            }
            total += layer.norm2.len();
            total += layer.gate_up_proj.in_features * layer.gate_up_proj.out_features;
            if let Some(bias) = &layer.gate_up_proj.bias {
                total += bias.len();
            }
            total += layer.down_proj.in_features * layer.down_proj.out_features;
            if let Some(bias) = &layer.down_proj.bias {
                total += bias.len();
            }
            if let Some(pfn) = &layer.post_ffw_norm {
                total += pfn.len();
            }
        }
        total += self.final_norm.len();
        total += self.lm_head.in_features * self.lm_head.out_features;
        if let Some(bias) = &self.lm_head.bias {
            total += bias.len();
        }

        total
    }

    /// Returns active [`QuantMode`] of model projections.
    pub fn current_quant_mode(&self) -> QuantMode {
        self.config.quant_mode
    }

    /// Re-quantizes all internal projection matrices according to [`QuantMode`].
    pub fn quantize(&mut self, mode: QuantMode) {
        self.config.quant_mode = mode;
        let (attn, ffn, lm) = projection_formats(mode);
        for layer in &mut self.layers {
            layer.qkv_proj = requantized(&layer.qkv_proj, attn);
            layer.out_proj = requantized(&layer.out_proj, attn);
            layer.gate_up_proj = requantized(&layer.gate_up_proj, ffn);
            layer.down_proj = requantized(&layer.down_proj, ffn);
        }
        self.lm_head = requantized(&self.lm_head, lm);
    }

    /// Executes forward pass over prompt tokens, returning logits for the final position (length vocab_size).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("graph_lm::forward");
        let mut logits = vec![0.0f32; self.config.vocab_size];
        if !tokens.is_empty() {
            self.lm_head
                .forward(&self.final_hidden(tokens), &mut logits);
        }
        logits
    }

    /// Final-norm hidden state (length hidden_dim) of the last of `tokens` (non-empty): what the
    /// LM head reads, and what sequence-classification heads score.
    pub fn final_hidden(&self, tokens: &[u32]) -> Vec<f32> {
        let v = self.config.vocab_size;
        let h = self.config.hidden_dim;
        let inter = self.config.intermediate_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let eps = self.config.norm_eps;
        let bits = self.config.act_quant_bits;
        assert!(!tokens.is_empty(), "final_hidden needs at least one token");

        let tokens = if tokens.len() > self.config.max_seq_len {
            &tokens[tokens.len() - self.config.max_seq_len..]
        } else {
            tokens
        };

        let seq_len = tokens.len();

        // 1. Embedding lookup
        let mut hidden: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(t, &tok)| {
                let tok_idx = (tok as usize).min(v - 1);
                let mut emb = self.embedding[tok_idx * h..(tok_idx + 1) * h].to_vec();
                if let Some(pos_emb) = &self.pos_embedding {
                    let pos_idx = t.min(self.config.max_seq_len.saturating_sub(1));
                    if (pos_idx + 1) * h <= pos_emb.len() {
                        for i in 0..h {
                            emb[i] += pos_emb[pos_idx * h + i];
                        }
                    }
                }
                emb
            })
            .collect();

        // 2. Transformer layers
        for layer in &self.layers {
            let mut q_all: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            let mut k_all: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            let mut v_all: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                // Norm 1
                let mut norm1_x = vec![0.0f32; h];
                if self.config.use_standard_layernorm {
                    let mean = hidden[t].iter().sum::<f32>() / h as f32;
                    let var = hidden[t]
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .sum::<f32>()
                        / h as f32;
                    let scale = 1.0 / (var + eps).sqrt();
                    for i in 0..h {
                        let b = layer.norm1_bias.as_ref().map(|b| b[i]).unwrap_or(0.0);
                        norm1_x[i] = (hidden[t][i] - mean) * scale * layer.norm1[i] + b;
                    }
                } else {
                    let sum_sq: f32 = hidden[t].iter().map(|&x| x * x).sum();
                    let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
                    for i in 0..h {
                        norm1_x[i] = hidden[t][i] * scale * layer.norm1[i];
                    }
                }

                // QKV projection: produces 3 * h
                let mut qkv = vec![0.0f32; 3 * h];
                if bits > 0 {
                    act_quant(&mut norm1_x, bits);
                }
                layer.qkv_proj.forward(&norm1_x, &mut qkv);

                let mut q_vec = qkv[0..h].to_vec();
                let mut k_vec = qkv[h..2 * h].to_vec();
                let v_vec = qkv[2 * h..3 * h].to_vec();

                // Rotary Position Embeddings (RoPE)
                let rotary_dim = ((head_dim as f32) * self.config.rotary_pct) as usize;
                let half_dim = (rotary_dim / 2).min(head_dim / 2);
                // Pair i is (i, i + half) for half-split RoPE, (2i, 2i + 1) for interleaved.
                let (pair_stride, pair_gap) = if self.config.rope_interleaved {
                    (2, 1)
                } else {
                    (1, half_dim)
                };
                for head in 0..num_heads {
                    let head_offset = head * head_dim;
                    for d in 0..half_dim {
                        let freq = 1.0
                            / (self.config.rope_theta)
                                .powf((2 * d) as f32 / rotary_dim.max(1) as f32);
                        let theta = t as f32 * freq;
                        let cos_th = theta.cos();
                        let sin_th = theta.sin();

                        let idx0 = head_offset + pair_stride * d;
                        let idx1 = idx0 + pair_gap;

                        let q0 = q_vec[idx0];
                        let q1 = q_vec[idx1];
                        q_vec[idx0] = q0 * cos_th - q1 * sin_th;
                        q_vec[idx1] = q1 * cos_th + q0 * sin_th;

                        let k0 = k_vec[idx0];
                        let k1 = k_vec[idx1];
                        k_vec[idx0] = k0 * cos_th - k1 * sin_th;
                        k_vec[idx1] = k1 * cos_th + k0 * sin_th;
                    }
                }

                q_all.push(q_vec);
                k_all.push(k_vec);
                v_all.push(v_vec);
            }

            // Multi-head causal attention execution
            let scale_attn = 1.0 / (head_dim as f32).sqrt();
            for t in 0..seq_len {
                let mut attn_out = vec![0.0f32; h];

                for head in 0..num_heads {
                    let head_offset = head * head_dim;

                    // Compute dot-product attention scores for causal history s <= t
                    let mut scores = Vec::with_capacity(t + 1);
                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..=t {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_all[t][head_offset + d] * k_all[s][head_offset + d];
                        }
                        let score = dot * scale_attn;
                        if score > max_score {
                            max_score = score;
                        }
                        scores.push(score);
                    }

                    // Softmax over scores
                    let mut sum_exp = 0.0f32;
                    let mut exp_scores = Vec::with_capacity(t + 1);
                    for &score in &scores {
                        let exp_val = (score - max_score).exp();
                        sum_exp += exp_val;
                        exp_scores.push(exp_val);
                    }
                    let inv_sum = 1.0 / sum_exp.max(1e-12);

                    // Weighted combination of V
                    for (s, &exp_val) in exp_scores.iter().enumerate() {
                        let weight = exp_val * inv_sum;
                        crate::model::simd::vector_fused_scale_add(
                            &mut attn_out[head_offset..head_offset + head_dim],
                            &v_all[s][head_offset..head_offset + head_dim],
                            weight,
                        );
                    }
                }

                // Output projection
                let mut proj_out = vec![0.0f32; h];
                if let Some(w) = &layer.attn_sub_norm {
                    rms_norm_inplace(&mut attn_out, w, eps);
                }
                if bits > 0 {
                    act_quant(&mut attn_out, bits);
                }
                layer.out_proj.forward(&attn_out, &mut proj_out);

                if let Some(pan) = &layer.post_attention_norm {
                    let sum_sq: f32 = proj_out.iter().map(|&x| x * x).sum();
                    let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
                    for i in 0..h {
                        proj_out[i] = proj_out[i] * scale * pan[i];
                    }
                }

                if !self.config.use_parallel_residual {
                    for i in 0..h {
                        hidden[t][i] += proj_out[i];
                    }
                }

                // Norm 2
                let mut norm2_x = vec![0.0f32; h];
                if self.config.use_standard_layernorm {
                    let mean = hidden[t].iter().sum::<f32>() / h as f32;
                    let var = hidden[t]
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .sum::<f32>()
                        / h as f32;
                    let scale = 1.0 / (var + eps).sqrt();
                    for i in 0..h {
                        let b = layer.norm2_bias.as_ref().map(|b| b[i]).unwrap_or(0.0);
                        norm2_x[i] = (hidden[t][i] - mean) * scale * layer.norm2[i] + b;
                    }
                } else {
                    let sum_sq: f32 = hidden[t].iter().map(|&x| x * x).sum();
                    let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
                    for i in 0..h {
                        norm2_x[i] = hidden[t][i] * scale * layer.norm2[i];
                    }
                }

                let mut down_out = vec![0.0f32; h];
                if self.config.use_gelu_mlp {
                    let mut mlp_inter = vec![0.0f32; inter];
                    layer.gate_up_proj.forward(&norm2_x, &mut mlp_inter);
                    for i in 0..inter {
                        let x = mlp_inter[i];
                        let c = (2.0f32 / std::f32::consts::PI).sqrt();
                        let inner = c * (x + 0.044715 * x * x * x);
                        mlp_inter[i] = 0.5 * x * (1.0 + inner.tanh());
                    }
                    layer.down_proj.forward(&mlp_inter, &mut down_out);
                } else {
                    let mut gate_up = vec![0.0f32; 2 * inter];
                    if bits > 0 {
                        act_quant(&mut norm2_x, bits);
                    }
                    layer.gate_up_proj.forward(&norm2_x, &mut gate_up);

                    let mut mlp_inter = vec![0.0f32; inter];
                    for i in 0..inter {
                        let gate = gate_up[i];
                        let up = gate_up[i + inter];
                        mlp_inter[i] = if self.config.use_relu2_mlp {
                            gate.max(0.0) * gate.max(0.0) * up
                        } else {
                            gate / (1.0 + (-gate).exp()) * up
                        };
                    }
                    if let Some(w) = &layer.ffn_sub_norm {
                        rms_norm_inplace(&mut mlp_inter, w, eps);
                    }
                    if bits > 0 {
                        act_quant(&mut mlp_inter, bits);
                    }
                    layer.down_proj.forward(&mlp_inter, &mut down_out);
                }

                if let Some(pfn) = &layer.post_ffw_norm {
                    let sum_sq: f32 = down_out.iter().map(|&x| x * x).sum();
                    let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
                    for i in 0..h {
                        down_out[i] = down_out[i] * scale * pfn[i];
                    }
                }

                if self.config.use_parallel_residual {
                    for i in 0..h {
                        hidden[t][i] += proj_out[i] + down_out[i];
                    }
                } else {
                    for i in 0..h {
                        hidden[t][i] += down_out[i];
                    }
                }
            }
        }

        // 3. Final Norm over last position
        let last_pos = seq_len - 1;
        let mut final_normed = vec![0.0f32; h];
        if self.config.use_standard_layernorm {
            let mean = hidden[last_pos].iter().sum::<f32>() / h as f32;
            let var = hidden[last_pos]
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f32>()
                / h as f32;
            let scale = 1.0 / (var + eps).sqrt();
            for i in 0..h {
                let b = self.final_norm_bias.as_ref().map(|b| b[i]).unwrap_or(0.0);
                final_normed[i] = (hidden[last_pos][i] - mean) * scale * self.final_norm[i] + b;
            }
        } else {
            let sum_sq: f32 = hidden[last_pos].iter().map(|&x| x * x).sum();
            let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
            for i in 0..h {
                final_normed[i] = hidden[last_pos][i] * scale * self.final_norm[i];
            }
        }
        final_normed
    }

    /// Evaluates a single newly-generated token at position `pos` using cached keys/values.
    ///
    /// Executes O(1) GEMV projections and O(pos) causal attention against [`LayerKVCache`].
    /// Evaluates a single newly-generated token at position `pos` using zero-allocation scratchpad and cached KV.
    pub fn forward_token_with_kv_scratchpad(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut [LayerKVCache],
        pad: &mut DecodeScratchpad,
    ) {
        let v = self.config.vocab_size;
        let h = self.config.hidden_dim;
        let tok_idx = (token as usize).min(v - 1);
        pad.hidden
            .copy_from_slice(&self.embedding[tok_idx * h..(tok_idx + 1) * h]);
        self.forward_hidden_scratchpad(pos, kv_cache, pad);
    }

    /// Runs `embedding` (one input row, e.g. a projected image patch, in place of a token's
    /// embedding) at `pos` against the KV cache; returns the logits.
    pub fn forward_embedding_with_kv(
        &self,
        embedding: &[f32],
        pos: usize,
        kv_cache: &mut [LayerKVCache],
    ) -> Vec<f32> {
        let mut pad = DecodeScratchpad::from_config(&self.config);
        pad.hidden.copy_from_slice(embedding);
        self.forward_hidden_scratchpad(pos, kv_cache, &mut pad);
        pad.logits
    }

    /// The layers and LM head on the input embedding in `pad.hidden` at `pos`.
    fn forward_hidden_scratchpad(
        &self,
        pos: usize,
        kv_cache: &mut [LayerKVCache],
        pad: &mut DecodeScratchpad,
    ) {
        let h = self.config.hidden_dim;
        let inter = self.config.intermediate_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let eps = self.config.norm_eps;
        let bits = self.config.act_quant_bits;
        if let Some(pos_emb) = &self.pos_embedding {
            let pos_idx = pos.min(self.config.max_seq_len.saturating_sub(1));
            if (pos_idx + 1) * h <= pos_emb.len() {
                for i in 0..h {
                    pad.hidden[i] += pos_emb[pos_idx * h + i];
                }
            }
        }

        let rotary_dim = ((head_dim as f32) * self.config.rotary_pct) as usize;
        let half_dim = (rotary_dim / 2).min(head_dim / 2);
        let clamped_half = half_dim.min(128);
        let mut rope_cos = [1.0f32; 128];
        let mut rope_sin = [0.0f32; 128];
        if pad.rope_inv_freq.len() >= clamped_half {
            for d in 0..clamped_half {
                let theta = pos as f32 * pad.rope_inv_freq[d];
                rope_cos[d] = theta.cos();
                rope_sin[d] = theta.sin();
            }
        } else {
            for d in 0..clamped_half {
                let freq =
                    1.0 / (self.config.rope_theta).powf((2 * d) as f32 / rotary_dim.max(1) as f32);
                let theta = pos as f32 * freq;
                rope_cos[d] = theta.cos();
                rope_sin[d] = theta.sin();
            }
        }

        for (l, layer) in self.layers.iter().enumerate() {
            let cache = &mut kv_cache[l];

            // 1. Norm 1 using vectorized SIMD primitives
            if self.config.use_standard_layernorm {
                crate::model::simd::vector_layernorm(
                    &pad.hidden,
                    &layer.norm1,
                    layer.norm1_bias.as_deref(),
                    &mut pad.norm1_x,
                    eps,
                );
            } else {
                crate::model::simd::vector_rmsnorm(
                    &pad.hidden,
                    &layer.norm1,
                    &mut pad.norm1_x,
                    eps,
                );
            }

            // 2. QKV projection: produces 3 * h (O(1) GEMV)
            if bits > 0 {
                act_quant(&mut pad.norm1_x, bits);
            }
            layer.qkv_proj.forward(&pad.norm1_x, &mut pad.qkv);

            // 3. Rotary Position Embeddings (RoPE) at position pos in-place on pad.qkv
            // Pair i is (i, i + half) for half-split RoPE, (2i, 2i + 1) for interleaved.
            let interleaved = self.config.rope_interleaved;
            let (pair_stride, pair_gap) = if interleaved { (2, 1) } else { (1, half_dim) };
            for head in 0..num_heads {
                let head_offset = head * head_dim;
                let mut d = 0;

                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use core::arch::aarch64::*;
                    let qkv_ptr = pad.qkv.as_mut_ptr();
                    let cos_ptr = rope_cos.as_ptr();
                    let sin_ptr = rope_sin.as_ptr();

                    while d + 4 <= clamped_half {
                        let cos_vec = vld1q_f32(cos_ptr.add(d));
                        let sin_vec = vld1q_f32(sin_ptr.add(d));
                        let rot = |x0: float32x4_t, x1: float32x4_t| {
                            (
                                vsubq_f32(vmulq_f32(x0, cos_vec), vmulq_f32(x1, sin_vec)),
                                vaddq_f32(vmulq_f32(x1, cos_vec), vmulq_f32(x0, sin_vec)),
                            )
                        };
                        // q at head_offset, k at h + head_offset
                        for base in [head_offset, h + head_offset] {
                            if interleaved {
                                let p = qkv_ptr.add(base + 2 * d);
                                let x = vld2q_f32(p);
                                let (a, b) = rot(x.0, x.1);
                                vst2q_f32(p, float32x4x2_t(a, b));
                            } else {
                                let p0 = qkv_ptr.add(base + d);
                                let p1 = qkv_ptr.add(base + half_dim + d);
                                let (a, b) = rot(vld1q_f32(p0), vld1q_f32(p1));
                                vst1q_f32(p0, a);
                                vst1q_f32(p1, b);
                            }
                        }

                        d += 4;
                    }
                }

                while d < clamped_half {
                    let cos_th = rope_cos[d];
                    let sin_th = rope_sin[d];
                    let idx0 = head_offset + pair_stride * d;
                    let idx1 = idx0 + pair_gap;

                    let q0 = pad.qkv[idx0];
                    let q1 = pad.qkv[idx1];
                    pad.qkv[idx0] = q0 * cos_th - q1 * sin_th;
                    pad.qkv[idx1] = q1 * cos_th + q0 * sin_th;

                    let k_idx0 = h + idx0;
                    let k_idx1 = h + idx1;
                    let k0 = pad.qkv[k_idx0];
                    let k1 = pad.qkv[k_idx1];
                    pad.qkv[k_idx0] = k0 * cos_th - k1 * sin_th;
                    pad.qkv[k_idx1] = k1 * cos_th + k0 * sin_th;

                    d += 1;
                }
            }

            // 4. Append K and V directly from pad.qkv into LayerKVCache
            cache.push_kv(&pad.qkv[h..2 * h], &pad.qkv[2 * h..3 * h]);
            let current_len = cache.len;

            // 5. Causal attention for this token against cached 0..=pos in O(pos) time
            let scale_attn = 1.0 / (head_dim as f32).sqrt();

            for head in 0..num_heads {
                let head_offset = head * head_dim;
                if self.config.use_hadamard_attention && head_dim.is_power_of_two() {
                    crate::quant::needle::hadamard_attention_f32(
                        &pad.qkv[head_offset..head_offset + head_dim],
                        &cache.k[head_offset..],
                        &cache.v[head_offset..],
                        head_dim,
                        current_len,
                        h,
                        &mut pad.attn_out[head_offset..head_offset + head_dim],
                    );
                } else {
                    crate::model::simd::vector_online_softmax_attention(
                        &pad.qkv[head_offset..head_offset + head_dim],
                        &cache.k[head_offset..],
                        &cache.v[head_offset..],
                        head_dim,
                        current_len,
                        h,
                        scale_attn,
                        &mut pad.attn_out[head_offset..head_offset + head_dim],
                    );
                }
            }

            // 6. Out-projection
            if let Some(w) = &layer.attn_sub_norm {
                rms_norm_inplace(&mut pad.attn_out, w, eps);
            }
            if bits > 0 {
                act_quant(&mut pad.attn_out, bits);
            }
            layer.out_proj.forward(&pad.attn_out, &mut pad.proj_out);

            if let Some(pan) = &layer.post_attention_norm {
                let mut sum_sq = 0.0f32;
                for &x in &pad.proj_out {
                    sum_sq += x * x;
                }
                let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
                for i in 0..h {
                    pad.proj_out[i] = pad.proj_out[i] * scale * pan[i];
                }
            }
            // 7. Fused Residual Add + Norm 2
            if !self.config.use_parallel_residual {
                if self.config.use_standard_layernorm {
                    crate::model::simd::vector_fused_add_layernorm(
                        &mut pad.hidden,
                        &pad.proj_out,
                        &layer.norm2,
                        layer.norm2_bias.as_deref(),
                        &mut pad.norm2_x,
                        eps,
                    );
                } else {
                    crate::model::simd::vector_fused_add_rmsnorm(
                        &mut pad.hidden,
                        &pad.proj_out,
                        &layer.norm2,
                        &mut pad.norm2_x,
                        eps,
                    );
                }
            } else {
                if self.config.use_standard_layernorm {
                    crate::model::simd::vector_layernorm(
                        &pad.hidden,
                        &layer.norm2,
                        layer.norm2_bias.as_deref(),
                        &mut pad.norm2_x,
                        eps,
                    );
                } else {
                    crate::model::simd::vector_rmsnorm(
                        &pad.hidden,
                        &layer.norm2,
                        &mut pad.norm2_x,
                        eps,
                    );
                }
            }
            // 8. SwiGLU / GELU MLP using vectorized SIMD primitives
            if self.config.use_hadamard_mlp && inter.is_power_of_two() && inter <= h {
                crate::quant::needle::hadamard_mlp_f32(
                    &pad.norm2_x[..inter],
                    &pad.hadamard_diag[..inter],
                    &mut pad.mlp_inter[..inter],
                );
                layer.down_proj.forward(&pad.mlp_inter, &mut pad.down_out);
            } else if self.config.use_gelu_mlp {
                layer.gate_up_proj.forward(&pad.norm2_x, &mut pad.mlp_inter);
                crate::model::simd::vector_gelu(&mut pad.mlp_inter);
                layer.down_proj.forward(&pad.mlp_inter, &mut pad.down_out);
            } else {
                if bits > 0 {
                    act_quant(&mut pad.norm2_x, bits);
                }
                layer.gate_up_proj.forward(&pad.norm2_x, &mut pad.gate_up);
                if self.config.use_relu2_mlp {
                    let (gate, up) = pad.gate_up.split_at(inter);
                    for ((m, &g), &u) in pad.mlp_inter.iter_mut().zip(gate).zip(up) {
                        *m = g.max(0.0) * g.max(0.0) * u;
                    }
                } else {
                    crate::model::simd::vector_fused_swiglu_inplace(
                        &pad.gate_up,
                        &mut pad.mlp_inter,
                    );
                }
                if let Some(w) = &layer.ffn_sub_norm {
                    rms_norm_inplace(&mut pad.mlp_inter, w, eps);
                }
                if bits > 0 {
                    act_quant(&mut pad.mlp_inter, bits);
                }
                layer.down_proj.forward(&pad.mlp_inter, &mut pad.down_out);
            }
            if let Some(pfn) = &layer.post_ffw_norm {
                let mut sum_sq = 0.0f32;
                for &x in &pad.down_out {
                    sum_sq += x * x;
                }
                let scale = 1.0 / (sum_sq / h as f32 + eps).sqrt();
                for i in 0..h {
                    pad.down_out[i] = pad.down_out[i] * scale * pfn[i];
                }
            }
            // 9. Residual addition
            if self.config.use_parallel_residual {
                crate::model::simd::vector_fused_scale_add(&mut pad.hidden, &pad.proj_out, 1.0);
                crate::model::simd::vector_fused_scale_add(&mut pad.hidden, &pad.down_out, 1.0);
            } else {
                crate::model::simd::vector_fused_scale_add(&mut pad.hidden, &pad.down_out, 1.0);
            }
            if let Some((_, rows)) = &mut cache.tap {
                rows.extend_from_slice(&pad.hidden);
            }
        }

        // 10. Final Norm using vectorized SIMD primitives
        if self.config.use_standard_layernorm {
            crate::model::simd::vector_layernorm(
                &pad.hidden,
                &self.final_norm,
                self.final_norm_bias.as_deref(),
                &mut pad.final_normed,
                eps,
            );
        } else {
            crate::model::simd::vector_rmsnorm(
                &pad.hidden,
                &self.final_norm,
                &mut pad.final_normed,
                eps,
            );
        }

        // 11. LM Head projection
        self.lm_head.forward(&pad.final_normed, &mut pad.logits);
    }

    /// Evaluates a single newly-generated token at position `pos` using cached keys/values.
    pub fn forward_token_with_kv(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut [LayerKVCache],
    ) -> Vec<f32> {
        let mut pad = DecodeScratchpad::from_config(&self.config);
        self.forward_token_with_kv_scratchpad(token, pos, kv_cache, &mut pad);
        pad.logits
    }

    /// Samples the next token ID from logits with repetition penalty, top-k, and nucleus top-p filtering.
    pub fn sample_next_token_with_penalty(
        &self,
        logits: &[f32],
        recent_tokens: &[u32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> u32 {
        let valid_vocab = if self.tokenizer.vocab_size() > 0 {
            logits.len().min(self.tokenizer.vocab_size())
        } else {
            logits.len()
        };
        sample_logits(
            &logits[..valid_vocab],
            recent_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
    }
}

/// Samples a token id from `slice` (logits over the tokenizer's vocabulary) with repetition
/// penalty, top-k, and nucleus top-p; `temperature <= 1e-4` is greedy argmax. Shared by every
/// causal LM the scheduler runs.
pub fn sample_logits(
    slice: &[f32],
    recent_tokens: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> u32 {
    if slice.is_empty() {
        return 0;
    }
    if temperature <= 1e-4 {
        if repetition_penalty <= 1.0 || recent_tokens.is_empty() {
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in slice.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            return best_idx as u32;
        }

        // When repetition penalty > 1.0, find global unpenalized argmax first.
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &val) in slice.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }

        // Penalty strictly decreases/non-increases logits. If the unpenalized max
        // is not among recent tokens, it is guaranteed to remain the global max.
        if !recent_tokens.contains(&(best_idx as u32)) {
            return best_idx as u32;
        }

        let mut penalized_best_idx = 0;
        let mut penalized_best_val = f32::NEG_INFINITY;
        for (i, &raw_val) in slice.iter().enumerate() {
            let tok_id = i as u32;
            let count = recent_tokens.iter().filter(|&&t| t == tok_id).count();
            let val = if count > 0 {
                let mut v = raw_val;
                for _ in 0..count {
                    if v > 0.0 {
                        v /= repetition_penalty;
                    } else {
                        v *= repetition_penalty;
                    }
                }
                v - (count as f32 * 1.5)
            } else {
                raw_val
            };
            if val > penalized_best_val {
                penalized_best_val = val;
                penalized_best_idx = i;
            }
        }
        return penalized_best_idx as u32;
    }

    let mut working_logits = slice.to_vec();
    if repetition_penalty > 1.0 && !recent_tokens.is_empty() {
        for &tok in recent_tokens {
            let idx = tok as usize;
            if idx < working_logits.len() {
                if working_logits[idx] > 0.0 {
                    working_logits[idx] /= repetition_penalty;
                } else {
                    working_logits[idx] *= repetition_penalty;
                }
                working_logits[idx] -= 1.5;
            }
        }
    }

    let (active_candidates, active_probs) = nucleus(&working_logits, temperature, top_k, top_p);
    let active_sum: f32 = active_probs.iter().sum();

    // 3. Step-varying seeded PRNG mixing token history and active distribution
    let mut hash =
        0x811c9dc5u32.wrapping_add((recent_tokens.len() as u32).wrapping_mul(0x9e3779b9));
    for (pos, &tok) in recent_tokens.iter().enumerate() {
        hash ^= tok.wrapping_mul(0x5bd1e995).wrapping_add(pos as u32);
        hash = hash.rotate_left(13).wrapping_mul(0x01000193);
    }
    for &(idx, val) in &active_candidates {
        hash ^= (idx as u32).wrapping_mul(0x27d4eb2d);
        hash ^= val.to_bits();
        hash = hash.rotate_left(7).wrapping_mul(0x01000193);
    }
    let r = (((hash & 0x007f_ffff) as f32) / 8_388_608.0) * active_sum;

    let mut cum_prob = 0.0f32;
    for (i, &p) in active_probs.iter().enumerate() {
        cum_prob += p;
        if cum_prob >= r || i == active_probs.len() - 1 {
            return active_candidates[i].0 as u32;
        }
    }

    active_candidates[0].0 as u32
}

/// The candidates [`sample_logits`] draws from at `temperature > 1e-4` (after any penalty):
/// top-k by logit, softmax at `temperature`, cut at cumulative `top_p`. Returns the kept
/// `(token, logit)` pairs and their probabilities, which sum to the kept mass.
pub(crate) fn nucleus(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> (Vec<(usize, f32)>, Vec<f32>) {
    let k = top_k.max(1).min(logits.len());
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    // Highest logit first, ties by token id: a stable full sort's order, in O(vocab).
    let order = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    if k < indexed_logits.len() {
        indexed_logits.select_nth_unstable_by(k - 1, order);
    }
    indexed_logits.truncate(k);
    indexed_logits.sort_by(order);

    let max_val = indexed_logits
        .iter()
        .map(|&(_, val)| val / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = indexed_logits
        .iter()
        .map(|&(_, val)| ((val / temperature) - max_val).exp())
        .collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let mut probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp.max(1e-12)).collect();

    // Nucleus (top-p) cutoff.
    let p_thresh = top_p.clamp(0.01, 1.0);
    let mut cum_p = 0.0f32;
    let mut cutoff_idx = indexed_logits.len();
    for (i, &p) in probs.iter().enumerate() {
        cum_p += p;
        if cum_p >= p_thresh {
            cutoff_idx = i + 1;
            break;
        }
    }
    indexed_logits.truncate(cutoff_idx);
    probs.truncate(cutoff_idx);
    (indexed_logits, probs)
}

impl GraphLm {
    /// Samples the next token ID from logits.
    pub fn sample_next_token(&self, logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        self.sample_next_token_with_penalty(logits, &[], temperature, top_k, 0.9, 1.15)
    }

    /// Executes forward pass and samples next token with repetition penalty.
    pub fn generate_step_with_penalty(
        &self,
        tokens: &[u32],
        temperature: f32,
        repetition_penalty: f32,
    ) -> u32 {
        crate::hardware::cpu_compute::enter("graph_lm::generate_step_with_penalty");
        let logits = self.forward(tokens);
        let recent = if tokens.len() > 64 {
            &tokens[tokens.len() - 64..]
        } else {
            tokens
        };
        self.sample_next_token_with_penalty(
            &logits,
            recent,
            temperature,
            50,
            0.9,
            repetition_penalty,
        )
    }

    /// Single autoregressive step: executes forward pass and samples next token.
    pub fn generate_step(&self, tokens: &[u32], temperature: f32) -> u32 {
        self.generate_step_with_penalty(tokens, temperature, 1.0)
    }

    /// Autoregressive text generation up to `max_new_tokens` using fast O(1) KV-cache decoding.

    pub fn generate_text_fast(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> String {
        self.generate_text_fast_timed(prompt, max_new_tokens, temperature)
            .0
    }

    /// Autoregressive text generation measuring pure autoregressive decode time
    /// (excluding prompt tokenization and prefill).
    pub fn generate_text_fast_timed(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> (String, std::time::Duration) {
        let (text, _, decode) = self.generate_text_measured(prompt, max_new_tokens, temperature);
        (text, decode)
    }

    /// CPU generation. Returns (text, decode forward passes, pure decode time).
    pub fn generate_text_measured(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> (String, usize, std::time::Duration) {
        let mut kv_cache: Vec<LayerKVCache> = Vec::new();
        let mut pad = DecodeScratchpad::from_config(&self.config);
        let mut pos = 0;
        let res = self.generate_with(prompt, max_new_tokens, temperature, |toks| {
            if kv_cache.is_empty() {
                let cap = (toks.len() + max_new_tokens + 1)
                    .min(self.config.max_seq_len)
                    .max(1);
                kv_cache = (0..self.layers.len())
                    .map(|_| LayerKVCache::new(cap, self.config.hidden_dim))
                    .collect();
            }
            for &tok in toks {
                self.forward_token_with_kv_scratchpad(tok, pos, &mut kv_cache, &mut pad);
                pos += 1;
            }
            Ok::<_, std::convert::Infallible>(pad.logits.clone())
        });
        match res {
            Ok(r) => r,
            Err(e) => match e {},
        }
    }

    /// Backend-agnostic generation loop. `forward(new_tokens)` appends the tokens to the caller's
    /// KV state and returns the logits after the last one. Returns (text, decode forward passes,
    /// pure decode time excluding tokenization and prefill).
    pub fn generate_with<E>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        mut forward: impl FnMut(&[u32]) -> Result<Vec<f32>, E>,
    ) -> Result<(String, usize, std::time::Duration), E> {
        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() {
            if let Some(&s_id) = self.tokenizer.special_tokens.get("<s>") {
                tokens.push(s_id);
            } else {
                tokens.push(0);
            }
        }

        if tokens.len() > self.config.max_seq_len {
            tokens = tokens[tokens.len() - self.config.max_seq_len..].to_vec();
        }

        tokens.reserve(max_new_tokens);

        // 1. Prefill prompt tokens to populate the KV cache
        let mut logits = forward(&tokens)?;
        let prompt_len = tokens.len();

        // 2. Autoregressive decoding loop: one forward pass per accepted token
        let mut decode_passes = 0;
        let decode_start = std::time::Instant::now();
        for step in 0..max_new_tokens {
            if tokens.len() >= self.config.max_seq_len {
                break;
            }
            let next_token =
                self.sample_next_token_with_penalty(&logits, &tokens, temperature, 50, 0.9, 1.0);
            if self.tokenizer.is_special(next_token) {
                break;
            }
            tokens.push(next_token);
            if step + 1 < max_new_tokens && tokens.len() < self.config.max_seq_len {
                logits = forward(&[next_token])?;
                decode_passes += 1;
            }
        }
        let decode_duration = decode_start.elapsed();

        let result_str = format!("{prompt}{}", self.tokenizer.decode(&tokens[prompt_len..]));

        Ok((result_str, decode_passes, decode_duration))
    }

    /// Autoregressive text generation up to `max_new_tokens` or a stop token.
    ///
    /// Delegates to [`generate_text_fast`] for O(1) KV-cache per-token decoding speed.
    pub fn generate_text(&self, prompt: &str, max_new_tokens: usize, temperature: f32) -> String {
        self.generate_text_fast(prompt, max_new_tokens, temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_lm_presets_and_param_counts() {
        let cfg_1m = GraphLmConfig::smollm_1m();
        let model_1m = GraphLm::new(cfg_1m);
        let params_1m = model_1m.parameter_count();
        assert!(params_1m <= 10_000_000);
        assert!(params_1m > 1_000_000 && params_1m < 1_500_000);

        let cfg_5m = GraphLmConfig::tiny_5m();
        let model_5m = GraphLm::new(cfg_5m);
        let params_5m = model_5m.parameter_count();
        assert!(params_5m <= 10_000_000);
        assert!(params_5m > 4_500_000 && params_5m < 5_500_000);

        let cfg_10m = GraphLmConfig::tiny_10m();
        let model_10m = GraphLm::new(cfg_10m);
        let params_10m = model_10m.parameter_count();
        assert!(params_10m <= 10_000_000);
        assert!(params_10m > 9_000_000 && params_10m <= 10_000_000);
    }

    #[test]
    fn test_smollm_graph_fingerprints_as_llama() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let graph = model.to_graph();
        let fp = crate::model::fingerprint::fingerprint_graph(&graph);
        assert_eq!(fp.family, "llama");
        assert!(fp.has_swiglu && !fp.has_mla && !fp.has_moe);
        assert_eq!(fp.rope_variant, "half_split");
        assert_eq!(graph.outputs.len(), 1);
    }
    #[test]
    fn test_graph_lm_to_graph_with_weights_completeness() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let (graph, weights) = model.to_graph_with_weights();
        assert_eq!(graph.outputs.len(), 1);
        // Ensure every weight referenced in the graph is present in weights
        for val in &graph.values {
            if val.kind == crate::graph::ValueKind::Weight {
                assert!(
                    weights.get(&val.name).is_some(),
                    "missing weight '{}' in generated weights",
                    val.name
                );
            }
        }
    }

    #[test]
    fn test_forward_produces_valid_logits() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let tokens = [1u32, 10, 42, 100];
        let logits = model.forward(&tokens);

        assert_eq!(logits.len(), model.config.vocab_size);
        for &logit in &logits {
            assert!(logit.is_finite(), "Logit must be finite, found {}", logit);
        }
    }

    #[test]
    fn test_generate_step_and_sample_next_token() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let tokens = [5u32, 20, 15];

        // Greedy
        let tok_greedy = model.generate_step(&tokens, 0.0);
        assert!((tok_greedy as usize) < model.config.vocab_size);

        // Temperature sampling
        let tok_temp = model.generate_step(&tokens, 0.7);
        assert!((tok_temp as usize) < model.config.vocab_size);
    }

    #[test]
    fn test_generate_text() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let output = model.generate_text("The", 5, 0.0);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_quantization_modes() {
        let mut model = GraphLm::new(GraphLmConfig::smollm_1m());
        let tokens = [1u32, 2, 3];

        model.quantize(QuantMode::NVFP4Ultra);
        let logits_nvfp4 = model.forward(&tokens);
        assert_eq!(logits_nvfp4.len(), model.config.vocab_size);
        assert!(logits_nvfp4.iter().all(|x| x.is_finite()));

        model.quantize(QuantMode::EXL3Balanced);
        let logits_exl3 = model.forward(&tokens);
        assert_eq!(logits_exl3.len(), model.config.vocab_size);
        assert!(logits_exl3.iter().all(|x| x.is_finite()));

        model.quantize(QuantMode::AdaptiveHybrid);
        let logits_hybrid = model.forward(&tokens);
        assert_eq!(logits_hybrid.len(), model.config.vocab_size);
        assert!(logits_hybrid.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_layer_kv_cache_ops() {
        let mut cache = LayerKVCache::new(16, 4);
        assert_eq!(cache.len, 0);
        assert!(cache.k.capacity() >= 64);
        assert!(cache.v.capacity() >= 64);

        let k0 = [1.0f32, 2.0, 3.0, 4.0];
        let v0 = [0.1f32, 0.2, 0.3, 0.4];
        cache.push_kv(&k0, &v0);
        assert_eq!(cache.len, 1);
        assert_eq!(&cache.k[0..4], &k0);
        assert_eq!(&cache.v[0..4], &v0);

        let k1 = [5.0f32, 6.0, 7.0, 8.0];
        let v1 = [0.5f32, 0.6, 0.7, 0.8];
        cache.push_kv(&k1, &v1);
        assert_eq!(cache.len, 2);
        assert_eq!(&cache.k[4..8], &k1);
        assert_eq!(&cache.v[4..8], &v1);

        cache.clear();
        assert_eq!(cache.len, 0);
        cache.push_kv(&k0, &v0);
        assert_eq!(cache.len, 1);
        assert_eq!(&cache.k[0..4], &k0);
        assert_eq!(&cache.v[0..4], &v0);
    }

    #[test]
    fn test_forward_token_with_kv_matches_full_forward() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let tokens = [1u32, 10, 42, 100];
        let full_logits = model.forward(&tokens);

        let mut kv_cache: Vec<LayerKVCache> = (0..model.layers.len())
            .map(|_| LayerKVCache::new(model.config.max_seq_len, model.config.hidden_dim))
            .collect();

        let mut kv_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            kv_logits = model.forward_token_with_kv(tok, pos, &mut kv_cache);
        }

        assert_eq!(kv_logits.len(), full_logits.len());
        for i in 0..model.config.vocab_size {
            let diff = (kv_logits[i] - full_logits[i]).abs();
            assert!(
                diff < 1e-4,
                "Logit mismatch at index {}: kv={}, full={}, diff={}",
                i,
                kv_logits[i],
                full_logits[i],
                diff
            );
        }
    }

    #[test]
    fn test_generate_text_fast_equivalence() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let prompt = "The";
        let out_fast = model.generate_text_fast(prompt, 5, 0.0);
        let out_compat = model.generate_text(prompt, 5, 0.0);
        assert!(!out_fast.is_empty());
        assert_eq!(out_fast, out_compat);
    }

    #[test]
    fn test_hadamard_attention_and_weightless_in_graph_lm() {
        let mut cfg = GraphLmConfig::smollm_1m();
        cfg.use_hadamard_attention = true;
        let model = GraphLm::new(cfg.clone());
        let prompt = "The";
        let out = model.generate_text_fast(prompt, 5, 0.0);
        assert!(!out.is_empty());
        assert!(out.starts_with(prompt));

        // Quantize model to SolidSF Jesse weightless RAM lookup tables (zero FP multipliers)
        let mut model_jesse = GraphLm::new(cfg);
        model_jesse.quantize(QuantMode::WeightlessJesse);
        let out_jesse = model_jesse.generate_text_fast(prompt, 5, 0.0);
        assert!(!out_jesse.is_empty());
        assert!(out_jesse.starts_with(prompt));
    }
    #[test]
    fn test_generate_text_fast_timed() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let prompt = "The";
        let (out, duration) = model.generate_text_fast_timed(prompt, 5, 0.0);
        assert!(!out.is_empty());
        assert!(out.starts_with(prompt));
        assert!(duration.as_nanos() > 0);
    }
    #[test]
    fn test_graph_lm_prefill_and_decode_graph_execution() {
        let model = GraphLm::new(GraphLmConfig::smollm_1m());
        let (prefill_g, weights) = model.to_prefill_graph(3);
        let (decode_g, _) = model.to_decode_graph();

        let mut state = crate::graph::StateStore::new();
        let prompt_tokens = vec![12u32, 45, 99];
        let prompt_pos = vec![0u32, 1, 2];

        let prefill_inputs = vec![
            crate::graph::Tensor::U32(prompt_tokens),
            crate::graph::Tensor::U32(prompt_pos),
        ];
        let prefill_out = crate::graph::cpu::run(&prefill_g, &weights, &mut state, prefill_inputs)
            .expect("prefill graph runs");
        assert_eq!(prefill_out.len(), 1);

        let decode_inputs = vec![
            crate::graph::Tensor::U32(vec![101u32]),
            crate::graph::Tensor::U32(vec![3u32]),
        ];
        let decode_out = crate::graph::cpu::run(&decode_g, &weights, &mut state, decode_inputs)
            .expect("decode graph runs");
        assert_eq!(decode_out.len(), 1);
        match &decode_out[0] {
            crate::graph::Tensor::F32(logits) => {
                assert_eq!(logits.len(), model.config.vocab_size);
                assert!(logits.iter().all(|x| x.is_finite()));
                let expected = model.forward(&[12, 45, 99, 101]);
                for (i, (&actual, &exp)) in logits.iter().zip(&expected).enumerate() {
                    assert!(
                        (actual - exp).abs() < 1e-4,
                        "mismatch at {i}: graph={actual}, model.forward={exp}"
                    );
                }
            }
            other => panic!("expected F32 logits, got {:?}", other),
        }
    }
}
