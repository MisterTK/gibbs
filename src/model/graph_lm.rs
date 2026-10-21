//! Parameterized Tiny LLM (<=10M parameters) architecture.
//!
//! Provides a configurable transformer language model designed for fast in-memory
//! execution, multi-head causal attention with RoPE, SwiGLU feed-forward networks,
//! RMSNorm, and multi-format quantization (FP16, NVFP4, EXL3).

use crate::hardware::{QuantFormat, QuantMode};
use crate::model::layers::{gguf_block_codec, QuantizedLinearLayer};
use crate::model::tokenizer::Tokenizer;
use crate::model::{ModelManifest, UniversalModelLoader};
use crate::runtime::lm::sample_logits;
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
    /// Attention logit scale; 0 means 1/sqrt(head_dim).
    #[serde(default)]
    pub attention_scale: f32,
    /// Granite `embedding_scale`: input embeddings are multiplied by it; 0 = unset.
    #[serde(default)]
    pub embedding_scale: f32,
    /// Granite `residual_scale`: each block's attention / MLP output is multiplied by it before the
    /// residual add; 0 = unset.
    #[serde(default)]
    pub residual_scale: f32,
    /// Granite `logit_scale`: the logits are divided by it; 0 = unset.
    #[serde(default)]
    pub logit_scale: f32,
}

impl GraphLmConfig {
    /// The attention score scale: the model's own, else 1/sqrt(head_dim).
    pub fn attn_scale(&self) -> f32 {
        if self.attention_scale > 0.0 {
            self.attention_scale
        } else {
            1.0 / (self.head_dim as f32).sqrt()
        }
    }

    /// Width of q (and of the expanded k, v and the attention output): `num_heads * head_dim`, which
    /// differs from `hidden_dim` for models with an explicit `head_dim` (Qwen3, ERNIE-4.5, Gemma).
    pub fn attn_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// `Some(s)` for a multiplier that is set and not the identity.
    fn mult(s: f32) -> Option<f32> {
        (s > 0.0 && s != 1.0).then_some(s)
    }

    /// Factor applied to the input embeddings, if any.
    pub fn embedding_mult(&self) -> Option<f32> {
        Self::mult(self.embedding_scale)
    }

    /// Factor applied to each block's branch output before the residual add, if any.
    pub fn residual_mult(&self) -> Option<f32> {
        Self::mult(self.residual_scale)
    }

    /// Factor applied to the logits (1 / `logit_scale`), if any.
    pub fn logit_mult(&self) -> Option<f32> {
        Self::mult(self.logit_scale).map(|s| 1.0 / s)
    }
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
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
            attention_scale: 0.0,
            embedding_scale: 0.0,
            residual_scale: 0.0,
            logit_scale: 0.0,
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
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
            attention_scale: 0.0,
            embedding_scale: 0.0,
            residual_scale: 0.0,
            logit_scale: 0.0,
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
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
            attention_scale: 0.0,
            embedding_scale: 0.0,
            residual_scale: 0.0,
            logit_scale: 0.0,
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
            rope_interleaved: false,
            norm_eps: 1e-5,
            act_quant_bits: 0,
            ternarize_weights: false,
            use_relu2_mlp: false,
            attention_scale: 0.0,
            embedding_scale: 0.0,
            residual_scale: 0.0,
            logit_scale: 0.0,
        }
    }

    /// Single mapping from a HuggingFace `config.json` to runtime config; every architecture flag resolves here.
    pub fn from_hf(hf: &crate::model::hf_loader::HfConfigJson, quant_mode: QuantMode) -> Self {
        let mut cfg = Self::from_hf_config(
            hf.vocab_size,
            hf.hidden_dim,
            hf.resolved_intermediate_dim(),
            hf.num_layers,
            hf.resolved_num_heads(),
            quant_mode,
        );
        cfg.head_dim = hf.head_dim.unwrap_or(cfg.head_dim);
        cfg.rotary_pct = if hf.position_embedding_type.as_deref() == Some("nope") {
            0.0
        } else {
            hf.resolved_rotary_pct()
        };
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
        cfg.attention_scale = hf.attention_multiplier.unwrap_or(0.0);
        cfg.embedding_scale = hf.embedding_multiplier.unwrap_or(0.0);
        cfg.residual_scale = hf.residual_multiplier.unwrap_or(0.0);
        cfg.logit_scale = hf.logits_scaling.unwrap_or(0.0);
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
            // Granite's `attention.scale` replaces 1/sqrt(head_dim) (llama.cpp `f_attention_scale`).
            attention_scale: get("attention.scale")
                .and_then(|s| s.parse().ok())
                .unwrap_or(base.attention_scale),
            // Granite's embedding / residual multipliers and logit divisor (llama.cpp `f_*_scale`).
            embedding_scale: get("embedding_scale")
                .and_then(|s| s.parse().ok())
                .unwrap_or(base.embedding_scale),
            residual_scale: get("residual_scale")
                .and_then(|s| s.parse().ok())
                .unwrap_or(base.residual_scale),
            logit_scale: get("logit_scale")
                .and_then(|s| s.parse().ok())
                .unwrap_or(base.logit_scale),
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
    pub rope_inv_freq: Vec<f32>,
    pub logits: Vec<f32>,
}

impl DecodeScratchpad {
    pub fn new(
        hidden_dim: usize,
        attn_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            hidden: vec![0.0f32; hidden_dim],
            norm1_x: vec![0.0f32; hidden_dim],
            qkv: vec![0.0f32; 3 * attn_dim],
            q_vec: vec![0.0f32; attn_dim],
            k_vec: vec![0.0f32; attn_dim],
            v_vec: vec![0.0f32; attn_dim],
            attn_out: vec![0.0f32; attn_dim],
            attn_scores: vec![0.0f32; 2048],
            proj_out: vec![0.0f32; hidden_dim],
            norm2_x: vec![0.0f32; hidden_dim],
            mlp_inter: vec![0.0f32; intermediate_dim],
            gate_up: vec![0.0f32; 2 * intermediate_dim],
            down_out: vec![0.0f32; hidden_dim],
            final_normed: vec![0.0f32; hidden_dim],
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
            config.attn_dim(),
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
    /// Per-head RMSNorm gains (`head_dim`) on q and k before RoPE (Qwen3, LFM2 attention).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// LFM2 gated short-conv layer: the depthwise causal conv taps `[hidden, kernel]`. Such a layer has no
    /// attention; `qkv_proj` is its `in_proj` (`[B; C; x]`, 3 * hidden rows) and `out_proj` its `out_proj`.
    pub conv: Option<Vec<f32>>,
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

/// LFM2 gated short conv at one position: `out[ch] = c[ch] * sum_tap taps[ch][tap] * bx[row tap][ch]`, where
/// `recent` are the last `kernel` (or fewer at the sequence start, zero before it) rows of `B * x`,
/// oldest first, and the newest tap multiplies the current row.
fn short_conv(taps: &[f32], c: &[f32], recent: &[&[f32]], out: &mut [f32]) {
    let kernel = taps.len() / out.len();
    let skipped = kernel - recent.len();
    for (ch, o) in out.iter_mut().enumerate() {
        let acc: f32 = recent
            .iter()
            .enumerate()
            .map(|(j, row)| taps[ch * kernel + skipped + j] * row[ch])
            .sum();
        *o = c[ch] * acc;
    }
}

/// In-place RMSNorm of `x` with gain `w`.
fn rms_norm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let scale = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (v, g) in x.iter_mut().zip(w) {
        *v *= scale * g;
    }
}

/// Per-head RMSNorm in place: `x` is a row of heads, `w` the `head_dim` gains shared by every head.
fn head_rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    x.chunks_exact_mut(w.len())
        .for_each(|head| rms_norm_inplace(head, w, eps));
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
/// blocks (16 elements, or the GGUF block size along each row) stay f32.
pub fn requantized(layer: &QuantizedLinearLayer, want: QuantFormat) -> QuantizedLinearLayer {
    let (in_f, out_f) = (layer.in_features, layer.out_features);
    let row_block = gguf_block_codec(want).map_or(16, |(n, _, _)| n);
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
        QuantMode::NVFP4Ultra => all(QuantFormat::NVFP4_E2M1),
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
                    crate::model::layers::QuantizedWeight::Uncompressed {
                        data: Vec::new(),
                        in_features: 0,
                        out_features: 0,
                    },
                ),
            )
        };
        let head_dim = self.config.head_dim;
        let run = |layer: &mut QuantizedLinearLayer, role| {
            let l = take(layer);
            *layer = enc.encode(Projection {
                role,
                layer: l,
                sources: Vec::new(),
                head_dim,
                checkpoint: None,
            });
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
                q_norm: None,
                k_norm: None,
                conv: None,
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
        self.build_graph_inner(n, true, None, None).0
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
        let (g, _) = self.build_graph_inner(n, cached, Some(&mut w_map), None);
        (g, w_map)
    }

    /// Uncached causal graph over `n` tokens ending in a bias-free `score` head `[labels, hidden]`
    /// (instead of the LM head) on the last row: output `[1, labels]`, the sequence-classification
    /// logits. Weights (named as in [`Self::build_graph`], plus `score.weight`) are cloned into
    /// `weights` when given.
    pub fn build_seqcls_graph(
        &self,
        n: usize,
        score: &[f32],
        weights: Option<&mut crate::graph::Weights>,
    ) -> crate::graph::Graph {
        self.build_graph_inner(n, false, weights, Some(score)).0
    }

    /// Decode graph (n = 1) whose first input is `inputs_embeds [1, hidden]` instead of a token id:
    /// the caller embeds each token (or substitutes a projected media row, as Idefics3 does).
    pub fn to_decode_graph_embeds(&self) -> (crate::graph::Graph, crate::graph::Weights) {
        let mut w_map = crate::graph::Weights::new();
        let (g, _) = self.build_graph_x(1, true, true, Some(&mut w_map), None);
        (g, w_map)
    }

    fn build_graph_inner(
        &self,
        n: usize,
        cached: bool,
        w_map: Option<&mut crate::graph::Weights>,
        score: Option<&[f32]>,
    ) -> (crate::graph::Graph, ()) {
        self.build_graph_x(n, cached, false, w_map, score)
    }

    fn build_graph_x(
        &self,
        n: usize,
        cached: bool,
        embeds: bool,
        mut w_map: Option<&mut crate::graph::Weights>,
        score: Option<&[f32]>,
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
        /// `x * s` as a graph node (Granite multipliers, a model attention scale folded into q).
        fn scale_by(g: &mut Graph, name: String, x: ValueId, s: f32, shape: Vec<usize>) -> ValueId {
            let op = Op::Unary {
                kind: crate::graph::UnaryKind::Scale(s),
            };
            g.add(name, op, &[x], shape, DType::F32)
        }
        /// Per-head RMSNorm: a `head_dim`-wide gain over each head of `x`.
        fn head_norm(
            g: &mut Graph,
            w_map: &mut Option<&mut crate::graph::Weights>,
            name: String,
            x: ValueId,
            gamma: &[f32],
            eps: f32,
            shape: Vec<usize>,
        ) -> ValueId {
            let w_name = format!("{name}.weight");
            let w = g.weight(&w_name, vec![gamma.len()], DType::F32);
            if let Some(map) = w_map {
                map.insert(w_name, Tensor::F32(gamma.to_vec()));
            }
            g.add(name, Op::RmsNorm { eps }, &[x, w], shape, DType::F32)
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
            scale: None,
        };

        let attn_w = c.attn_dim();
        let mut g = Graph::new();
        let (mut x, positions) = if embeds {
            let e = g.input("inputs_embeds", vec![n, h], DType::F32);
            (e, g.input("positions", vec![n], DType::U32))
        } else {
            let ids = g.input("token_ids", vec![n], DType::U32);
            let positions = g.input("positions", vec![n], DType::U32);
            let table = g.weight("model.embed_tokens.weight", vec![v, h], DType::F32);
            if let Some(map) = &mut w_map {
                map.insert(
                    "model.embed_tokens.weight",
                    Tensor::F32(self.embedding.clone()),
                );
            }
            let x = g.add(
                "model.embed_tokens",
                Op::Embedding,
                &[table, ids],
                vec![n, h],
                DType::F32,
            );
            (x, positions)
        };
        if let Some(s) = c.embedding_mult() {
            x = scale_by(&mut g, "model.embed_scale".into(), x, s, vec![n, h]);
        }
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
            x = g.add("model.embed", Op::Binary { kind: crate::graph::BinaryKind::Add }, &[x, p], vec![n, h], DType::F32);
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
                &format!(
                    "{p}.{}",
                    if layer.conv.is_some() {
                        "conv.in_proj"
                    } else {
                        "self_attn.qkv_proj"
                    }
                ),
                n1,
                &layer.qkv_proj,
                n,
            );
            let mut q = slice(&mut g, format!("{p}.self_attn.q"), qkv, 0, attn_w, n);
            let mut k = slice(
                &mut g,
                format!("{p}.self_attn.k"),
                qkv,
                attn_w,
                2 * attn_w,
                n,
            );
            let vv = slice(
                &mut g,
                format!("{p}.self_attn.v"),
                qkv,
                2 * attn_w,
                3 * attn_w,
                n,
            );
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                // `RmsNorm` with a head_dim-wide gain normalises each head of the row.
                q = head_norm(
                    &mut g,
                    &mut w_map,
                    format!("{p}.self_attn.q_norm"),
                    q,
                    qn,
                    eps,
                    vec![n, attn_w],
                );
                k = head_norm(
                    &mut g,
                    &mut w_map,
                    format!("{p}.self_attn.k_norm"),
                    k,
                    kn,
                    eps,
                    vec![n, attn_w],
                );
            }
            if rotary_dim > 0 && layer.conv.is_none() {
                let inv_freq_val = g.weight(
                    "rope.inv_freq.default",
                    vec![inv_freq_vec.len()],
                    DType::F32,
                );
                if let Some(map) = &mut w_map {
                    map.insert("rope.inv_freq.default", Tensor::F32(inv_freq_vec.clone()));
                }
                q = g.add(
                    format!("{p}.self_attn.rope_q"),
                    rope.clone(),
                    &[q, positions, inv_freq_val],
                    vec![n, attn_w],
                    DType::F32,
                );
                k = g.add(
                    format!("{p}.self_attn.rope_k"),
                    rope.clone(),
                    &[k, positions, inv_freq_val],
                    vec![n, attn_w],
                    DType::F32,
                );
            }
            let a = if let Some(taps) = &layer.conv {
                // LFM2 `C * conv(B * x)`: q/k/v above are the B, C, x chunks of `in_proj`.
                let kernel = taps.len() / h;
                let bx = g.add(
                    format!("{p}.conv.bx"),
                    Op::Binary { kind: crate::graph::BinaryKind::Mul },
                    &[q, vv],
                    vec![n, h],
                    DType::F32,
                );
                let w_name = format!("{p}.conv.conv.weight");
                let taps_w = g.weight(&w_name, vec![h, kernel], DType::F32);
                if let Some(map) = &mut w_map {
                    map.insert(w_name, Tensor::F32(taps.clone()));
                }
                let mut conv_in = vec![bx, taps_w];
                if cached {
                    // Recurrent: `StateStore::truncate` refuses to rewind it.
                    conv_in.push(g.state(
                        format!("layers.{i}.conv_state"),
                        vec![kernel - 1, h],
                        DType::F32,
                        true,
                    ));
                }
                let op = Op::CausalConv1dCached {
                    kernel,
                    dilation: 1,
                };
                let conv = g.add(
                    format!("{p}.conv.conv"),
                    op,
                    &conv_in,
                    vec![n, h],
                    DType::F32,
                );
                g.add(
                    format!("{p}.conv.gate"),
                    Op::Binary { kind: crate::graph::BinaryKind::Mul },
                    &[k, conv],
                    vec![n, h],
                    DType::F32,
                )
            } else if cached {
                let k_state = g.state(
                    format!("kv.{i}.k"),
                    vec![c.max_seq_len, c.num_heads * c.head_dim],
                    DType::F32,
                    false,
                );
                let v_state = g.state(
                    format!("kv.{i}.v"),
                    vec![c.max_seq_len, c.num_heads * c.head_dim],
                    DType::F32,
                    false,
                );
                g.add(
                    format!("{p}.self_attn"),
                    Op::CachedAttention {
                        heads: c.num_heads,
                        kv_heads: c.num_heads,
                        head_dim: c.head_dim,
                        v_head_dim: c.head_dim,
                        scale: c.attn_scale(),
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
                // `Op::Attention` has no scale parameter (it is 1/sqrt(head_dim)): fold the model's into q.
                let q = if c.attention_scale > 0.0 {
                    let s = c.attn_scale() * (c.head_dim as f32).sqrt();
                    scale_by(
                        &mut g,
                        format!("{p}.self_attn.q_scale"),
                        q,
                        s,
                        vec![n, attn_w],
                    )
                } else {
                    q
                };
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
                &format!(
                    "{p}.{}",
                    if layer.conv.is_some() {
                        "conv.out_proj"
                    } else {
                        "self_attn.o_proj"
                    }
                ),
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
            if let Some(s) = c.residual_mult() {
                o = scale_by(&mut g, format!("{p}.attn_residual_scale"), o, s, vec![n, h]);
            }
            let x_attn = g.add(
                format!("{p}.residual_1"),
                Op::Binary { kind: crate::graph::BinaryKind::Add },
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
                    Op::Unary { kind: crate::graph::UnaryKind::GeluTanh },
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
            if let Some(s) = c.residual_mult() {
                d = scale_by(&mut g, format!("{p}.mlp_residual_scale"), d, s, vec![n, h]);
            }
            x = g.add(
                format!("{p}.residual_2"),
                Op::Binary { kind: crate::graph::BinaryKind::Add },
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
        let logits = match score {
            None => {
                let l = linear(&mut g, &mut w_map, "lm_head", n_norm, &self.lm_head, n);
                match c.logit_mult() {
                    Some(s) => scale_by(
                        &mut g,
                        "logit_scale".into(),
                        l,
                        s,
                        vec![n, self.lm_head.out_features],
                    ),
                    None => l,
                }
            }
            Some(s) => {
                let labels = s.len() / h;
                let w = g.weight("score.weight", vec![labels, h], DType::F32);
                if let Some(map) = &mut w_map {
                    map.insert("score.weight", Tensor::F32(s.to_vec()));
                }
                let slice = Op::Slice {
                    axis: 0,
                    start: n as i64 - 1,
                    end: n as i64,
                };
                let last = g.add("last_row", slice, &[n_norm], vec![1, h], DType::F32);
                g.add("score", Op::Linear, &[last, w], vec![1, labels], DType::F32)
            }
        };
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

    /// Granite GGUFs: Mamba (`granitehybrid`) and expert (`granitemoe`) layers are not implemented on the
    /// dense path, so such a file is an error instead of a plausible-looking wrong model. The attention,
    /// embedding, residual and logit multipliers are honoured through [`GraphLmConfig`].
    fn check_granite(manifest: &ModelManifest) -> Result<(), String> {
        let md = &manifest.metadata;
        let Some(arch) = md
            .get("general.architecture")
            .filter(|a| a.starts_with("granite"))
        else {
            return Ok(());
        };
        if arch == "granitehybrid" || arch == "granitemoe" {
            return Err(format!("architecture '{arch}' (Mamba / expert layers) is not implemented on the dense path"));
        }
        Ok(())
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
        Self::check_granite(manifest)?;
        let v = config.vocab_size;
        let h = config.hidden_dim;
        let inter = config.intermediate_dim;
        let layers_count = config.num_layers;
        let a = config.attn_dim();
        // Per-head q/k RMSNorm (`head_dim` gains, before RoPE) is consumed below; any other tensor of
        // that family (hunyuan's after-RoPE `query_layernorm`, a vision tower's, full-width OLMo2) is not.
        let qk_names = |l: usize| {
            [
                format!("blk.{l}.attn_q_norm.weight"),
                format!("blk.{l}.attn_k_norm.weight"),
                format!("model.layers.{l}.self_attn.q_norm.weight"),
                format!("model.layers.{l}.self_attn.k_norm.weight"),
                format!("model.layers.{l}.self_attn.q_layernorm.weight"),
                format!("model.layers.{l}.self_attn.k_layernorm.weight"),
            ]
        };
        let consumed: std::collections::HashSet<String> =
            (0..layers_count).flat_map(qk_names).collect();
        if let Some(n) = manifest.tensors.keys().find(|n| {
            [
                "q_norm.",
                "k_norm.",
                "q_layernorm.",
                "k_layernorm.",
                "query_layernorm.",
                "key_layernorm.",
            ]
            .iter()
            .any(|s| n.contains(s))
                && !consumed.contains(*n)
        }) {
            return Err(format!(
                "tensor '{n}': per-head q/k RMSNorm is not implemented by this decoder"
            ));
        }
        if let Some(n) = manifest
            .tensors
            .keys()
            .find(|n| n.contains(".block_sparse_moe."))
        {
            return Err(format!(
                "tensor '{n}': expert (MoE) layers are not implemented on the dense path"
            ));
        }
        // Gemma 3 / EXAONE 4 carry q/k norms but also sliding-window layers, per-layer RoPE bases and
        // (Gemma) scaled embeddings, none of which this decoder has: refuse instead of running a
        // plausible-looking wrong model.
        let arch = manifest
            .metadata
            .get("general.architecture")
            .map(String::as_str);
        let sandwich_qk = manifest
            .tensors
            .contains_key("model.layers.0.pre_feedforward_layernorm.weight")
            && manifest
                .tensors
                .contains_key("model.layers.0.self_attn.q_norm.weight");
        if matches!(arch, Some("gemma3" | "exaone4")) || sandwich_qk {
            return Err(format!(
                "architecture '{}' (sliding-window layers, per-layer RoPE, scaled embeddings) is not implemented on the dense path",
                arch.unwrap_or("gemma3")
            ));
        }
        // `head_dim` gains of layer `l`'s q (or k) norm; both or neither must exist.
        let qk_norm = |l: usize, q: bool| -> Result<Option<Vec<f32>>, String> {
            let [gq, gk, hq, hk, lq, lk] = qk_names(l);
            let names = if q { [gq, hq, lq] } else { [gk, hk, lk] };
            let Some(n) = names.iter().find(|n| manifest.tensors.contains_key(*n)) else {
                return Ok(None);
            };
            let w = manifest.extract_tensor_f32(data, n)?;
            if w.len() != config.head_dim {
                return Err(format!(
                    "tensor '{n}': {} gains, expected head_dim {} (full-width q/k norm is not implemented)",
                    w.len(),
                    config.head_dim
                ));
            }
            Ok(Some(w))
        };

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
                        crate::model::f16_to_f32(crate::model::f32_to_f16(x))
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
                let sources = if config.ternarize_weights {
                    Vec::new()
                } else {
                    sources
                };
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
            // LFM2 gated short-conv layer: `in_proj` ([B; C; x]), the depthwise `conv` taps and `out_proj`.
            let conv_pick =
                |names: [String; 2]| names.into_iter().find(|n| manifest.tensors.contains_key(n));
            let conv = match (
                conv_pick([
                    format!("blk.{l}.shortconv.conv.weight"),
                    format!("model.layers.{l}.conv.conv.weight"),
                ]),
                conv_pick([
                    format!("blk.{l}.shortconv.in_proj.weight"),
                    format!("model.layers.{l}.conv.in_proj.weight"),
                ]),
            ) {
                (Some(taps_name), Some(in_proj)) => {
                    if a != h {
                        return Err(format!(
                            "layer {l}: short-conv layers need num_heads * head_dim == hidden_dim"
                        ));
                    }
                    if manifest
                        .tensors
                        .contains_key(&format!("{}bias", taps_name.trim_end_matches("weight")))
                    {
                        return Err(format!(
                            "layer {l}: conv bias is not implemented by this decoder"
                        ));
                    }
                    let taps = manifest.extract_tensor_f32(data, &taps_name)?;
                    let kernel = taps.len() / h;
                    let l_cache: Option<usize> = manifest
                        .metadata
                        .get("lfm2.shortconv.l_cache")
                        .and_then(|s| s.parse().ok());
                    if taps.len() % h != 0 || kernel < 2 || l_cache.is_some_and(|k| k != kernel) {
                        return Err(format!(
                            "tensor '{taps_name}': {} taps for hidden {h} (l_cache {l_cache:?})",
                            taps.len()
                        ));
                    }
                    Some((taps, in_proj))
                }
                _ => None,
            };
            // norm1
            let norm1_candidates = [
                format!("blk.{}.attn_norm.weight", l),
                format!("model.layers.{}.input_layernorm.weight", l),
                format!("model.layers.{}.operator_norm.weight", l),
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
            if let Some((_, in_proj)) = &conv {
                raw_qkv = Some(ext(in_proj)?);
            }

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
                if let (Ok(q), Ok(k), Ok(v_vec)) = (ext(&gguf_q), ext(&gguf_k), ext(&gguf_v)) {
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
                } else if let (Ok(q), Ok(k), Ok(v_vec)) = (ext(&neo_q), ext(&neo_k), ext(&neo_v)) {
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
                format!("blk.{}.shortconv.out_proj.weight", l),
                format!("model.layers.{}.conv.out_proj.weight", l),
                format!("model.layers.{}.self_attn.out_proj.weight", l),
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
                format!("model.layers.{}.ffn_norm.weight", l),
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
                if let (Ok(g), Ok(u)) = (ext(&gguf_gate), ext(&gguf_up)) {
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
                .or_else(|_| {
                    ext(&format!(
                        "model.layers.{}.shared_mlp.input_linear.weight",
                        l
                    ))
                })
            {
                // HF fused [gate; up] rows (GLM, Phi-3, Granite-4 `shared_mlp.input_linear`): already our gate-first layout.
                raw_gate_up = Some(vec);
            } else if let (Ok(g), Ok(u)) = (proj(&gate_name), proj(&up_name)) {
                let mut fused = Vec::with_capacity(g.len() + u.len());
                fused.extend(g);
                fused.extend(u);
                raw_gate_up = Some(fused);
            } else if let (Ok(g), Ok(u)) = (
                proj(&format!("model.layers.{l}.feed_forward.w1.weight")),
                proj(&format!("model.layers.{l}.feed_forward.w3.weight")),
            ) {
                // LFM2: w1 = gate, w3 = up.
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
                format!("model.layers.{}.feed_forward.w2.weight", l),
                format!("model.layers.{}.shared_mlp.output_linear.weight", l),
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
                3 * a,
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
                    if fused_bias.len() != 3 * a {
                        fused_bias.resize(3 * a, 0.0);
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
                    if fused_bias.len() != 3 * a {
                        fused_bias.resize(3 * a, 0.0);
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
                a,
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

            let (q_norm, k_norm) = (qk_norm(l, true)?, qk_norm(l, false)?);
            if q_norm.is_some() != k_norm.is_some() {
                return Err(format!("layer {l}: q and k norm gains must come as a pair"));
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
                q_norm,
                k_norm,
                conv: conv.map(|(taps, _)| taps),
            });
        }

        // final_norm
        // LFM2 names its final norm `embedding_norm` (HF) / `token_embd_norm` (GGUF).
        let mut final_norm_candidates = vec![
            "output_norm.weight",
            "model.norm.weight",
            "gpt_neox.final_layer_norm.weight",
            "transformer.ln_f.weight",
            "model.embedding_norm.weight",
        ];
        if arch == Some("lfm2") {
            final_norm_candidates.push("token_embd_norm.weight");
        }
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
            total += layer.q_norm.as_ref().map_or(0, Vec::len);
            total += layer.k_norm.as_ref().map_or(0, Vec::len);
            total += layer.conv.as_ref().map_or(0, Vec::len);
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
            if let Some(s) = self.config.logit_mult() {
                logits.iter_mut().for_each(|l| *l *= s);
            }
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
        let a = self.config.attn_dim();
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
                if let Some(s) = self.config.embedding_mult() {
                    emb.iter_mut().for_each(|e| *e *= s);
                }
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
            let heads_here = if layer.conv.is_some() { 0 } else { num_heads };
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

                // QKV projection: produces 3 * a (a = num_heads * head_dim)
                let mut qkv = vec![0.0f32; 3 * a];
                if bits > 0 {
                    act_quant(&mut norm1_x, bits);
                }
                layer.qkv_proj.forward(&norm1_x, &mut qkv);

                let mut q_vec = qkv[0..a].to_vec();
                let mut k_vec = qkv[a..2 * a].to_vec();
                let v_vec = qkv[2 * a..3 * a].to_vec();
                if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                    head_rms_norm(&mut q_vec, qn, eps);
                    head_rms_norm(&mut k_vec, kn, eps);
                }

                if layer.conv.is_some() {
                    // short conv: q_all = C, k_all = B * x (the conv input); no RoPE or attention
                    let (b, x) = (&qkv[..a], &qkv[2 * a..]);
                    k_vec = b.iter().zip(x).map(|(b, x)| b * x).collect();
                    q_vec = qkv[a..2 * a].to_vec();
                }
                // Rotary Position Embeddings (RoPE)
                let rotary_dim = ((head_dim as f32) * self.config.rotary_pct) as usize;
                let half_dim = (rotary_dim / 2).min(head_dim / 2);
                // Pair i is (i, i + half) for half-split RoPE, (2i, 2i + 1) for interleaved.
                let (pair_stride, pair_gap) = if self.config.rope_interleaved {
                    (2, 1)
                } else {
                    (1, half_dim)
                };
                for head in 0..heads_here {
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
            let scale_attn = self.config.attn_scale();
            for t in 0..seq_len {
                let mut attn_out = vec![0.0f32; a];

                for head in 0..heads_here {
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

                if let Some(taps) = &layer.conv {
                    let lo = (t + 1).saturating_sub(taps.len() / h);
                    let recent: Vec<&[f32]> = k_all[lo..=t].iter().map(Vec::as_slice).collect();
                    short_conv(taps, &q_all[t], &recent, &mut attn_out);
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
                if let Some(s) = self.config.residual_mult() {
                    proj_out.iter_mut().for_each(|p| *p *= s);
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
                if let Some(s) = self.config.residual_mult() {
                    down_out.iter_mut().for_each(|d| *d *= s);
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
        let a = self.config.attn_dim();
        let eps = self.config.norm_eps;
        let bits = self.config.act_quant_bits;
        if let Some(s) = self.config.embedding_mult() {
            pad.hidden.iter_mut().for_each(|e| *e *= s);
        }
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
            let heads_here = if layer.conv.is_some() { 0 } else { num_heads };

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
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                head_rms_norm(&mut pad.qkv[..a], qn, eps);
                head_rms_norm(&mut pad.qkv[a..2 * a], kn, eps);
            }

            // 3. Rotary Position Embeddings (RoPE) at position pos in-place on pad.qkv
            // Pair i is (i, i + half) for half-split RoPE, (2i, 2i + 1) for interleaved.
            let interleaved = self.config.rope_interleaved;
            let (pair_stride, pair_gap) = if interleaved { (2, 1) } else { (1, half_dim) };
            for head in 0..heads_here {
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
                        // q at head_offset, k at a + head_offset
                        for base in [head_offset, a + head_offset] {
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

                    let k_idx0 = a + idx0;
                    let k_idx1 = a + idx1;
                    let k0 = pad.qkv[k_idx0];
                    let k1 = pad.qkv[k_idx1];
                    pad.qkv[k_idx0] = k0 * cos_th - k1 * sin_th;
                    pad.qkv[k_idx1] = k1 * cos_th + k0 * sin_th;

                    d += 1;
                }
            }

            // 4. Append K and V directly from pad.qkv into LayerKVCache
            if let Some(taps) = &layer.conv {
                // ponytail: host reference path; the whole B * x history stays in `cache.k` and one row is allocated per token.
                let bx: Vec<f32> = pad.qkv[..a]
                    .iter()
                    .zip(&pad.qkv[2 * a..])
                    .map(|(b, x)| b * x)
                    .collect();
                cache.push_kv(&bx, &[]);
                let skip = cache.len.saturating_sub(taps.len() / h);
                // Only rows below `cache.len`: a spec/draft rewind lowers `len` without truncating `k`.
                let recent: Vec<&[f32]> = cache.k[..cache.len * h]
                    .chunks_exact(h)
                    .skip(skip)
                    .collect();
                short_conv(taps, &pad.qkv[a..2 * a], &recent, &mut pad.attn_out);
            } else {
                cache.push_kv(&pad.qkv[a..2 * a], &pad.qkv[2 * a..3 * a]);
            }
            let current_len = cache.len;

            // 5. Causal attention for this token against cached 0..=pos in O(pos) time
            let scale_attn = self.config.attn_scale();

            for head in 0..heads_here {
                let head_offset = head * head_dim;
                crate::model::simd::vector_online_softmax_attention(
                    &pad.qkv[head_offset..head_offset + head_dim],
                    &cache.k[head_offset..],
                    &cache.v[head_offset..],
                    head_dim,
                    current_len,
                    a,
                    scale_attn,
                    &mut pad.attn_out[head_offset..head_offset + head_dim],
                );
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
            if let Some(s) = self.config.residual_mult() {
                pad.proj_out.iter_mut().for_each(|p| *p *= s);
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
            if self.config.use_gelu_mlp {
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
            if let Some(s) = self.config.residual_mult() {
                pad.down_out.iter_mut().for_each(|d| *d *= s);
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
        if let Some(s) = self.config.logit_mult() {
            pad.logits.iter_mut().for_each(|l| *l *= s);
        }
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

    /// Granite's `attention.scale` (1/head_dim for granite-4.x) replaces 1/sqrt(head_dim); other archs keep it.
    #[test]
    fn gguf_attention_scale_overrides_the_default() {
        let manifest = |scale: Option<&str>| {
            let mut md = std::collections::HashMap::new();
            md.insert("general.architecture".to_string(), "granite".to_string());
            md.insert("granite.embedding_length".to_string(), "256".to_string());
            md.insert("granite.attention.head_count".to_string(), "2".to_string());
            if let Some(s) = scale {
                md.insert("granite.attention.scale".to_string(), s.to_string());
            }
            ModelManifest {
                format: crate::model::SourceFormat::Gguf,
                tensors: Default::default(),
                metadata: md,
                shard_count: 1,
                detected_architecture: String::new(),
            }
        };
        let base = GraphLmConfig::smollm_1m();
        let with = GraphLmConfig::from_gguf(&manifest(Some("0.0078125")), base.clone());
        assert_eq!(with.attn_scale(), 0.0078125);
        let without = GraphLmConfig::from_gguf(&manifest(None), base);
        assert_eq!(without.attn_scale(), 1.0 / (128f32).sqrt());
    }

    fn granite_manifest(arch: &str, extra: &[(&str, &str)]) -> ModelManifest {
        let mut md = std::collections::HashMap::new();
        md.insert("general.architecture".to_string(), arch.to_string());
        for (k, v) in extra {
            md.insert(format!("{arch}.{k}"), v.to_string());
        }
        ModelManifest {
            format: crate::model::SourceFormat::Gguf,
            tensors: Default::default(),
            metadata: md,
            shard_count: 1,
            detected_architecture: String::new(),
        }
    }

    /// Hybrid (Mamba) and expert Granites are rejected instead of loaded wrong; dense ones load with their
    /// multipliers, which no longer make a file an error.
    #[test]
    fn granite_gguf_accepts_scales_and_rejects_unsupported_layers() {
        for extra in [
            &[("residual_scale", "0.22")][..],
            &[("logit_scale", "16")],
            &[("embedding_scale", "12")],
        ] {
            assert!(GraphLm::check_granite(&granite_manifest("granite", extra)).is_ok());
        }
        assert!(GraphLm::check_granite(&granite_manifest("granitehybrid", &[])).is_err());
        assert!(GraphLm::check_granite(&granite_manifest("granitemoe", &[])).is_err());
        assert!(
            GraphLm::check_granite(&granite_manifest("llama", &[("residual_scale", "0.22")]))
                .is_ok()
        );
    }

    /// `granite.{embedding,residual,logit}_scale` land in the config; 1.0 and absent keys are the identity.
    #[test]
    fn gguf_granite_multipliers_reach_the_config() {
        let m = granite_manifest(
            "granite",
            &[
                ("embedding_scale", "12"),
                ("residual_scale", "0.22"),
                ("logit_scale", "8"),
            ],
        );
        let c = GraphLmConfig::from_gguf(&m, GraphLmConfig::smollm_1m());
        assert_eq!(c.embedding_mult(), Some(12.0));
        assert_eq!(c.residual_mult(), Some(0.22));
        assert_eq!(c.logit_mult(), Some(0.125));
        let ones = granite_manifest(
            "granite",
            &[("embedding_scale", "1.0"), ("logit_scale", "1.0")],
        );
        let c = GraphLmConfig::from_gguf(&ones, GraphLmConfig::smollm_1m());
        assert_eq!(
            (c.embedding_mult(), c.residual_mult(), c.logit_mult()),
            (None, None, None)
        );
    }

    /// The Granite config.json multipliers map onto the same fields.
    #[test]
    fn hf_granite_multipliers_reach_the_config() {
        let hf: crate::model::hf_loader::HfConfigJson = serde_json::from_str(
            r#"{"model_type":"granite","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":2,
                "attention_multiplier":0.015625,"embedding_multiplier":12.0,
                "residual_multiplier":0.22,"logits_scaling":8.0}"#,
        )
        .unwrap();
        let c = GraphLmConfig::from_hf(&hf, QuantMode::Uncompressed);
        assert_eq!(c.attn_scale(), 0.015625);
        assert_eq!(
            (c.embedding_mult(), c.residual_mult(), c.logit_mult()),
            (Some(12.0), Some(0.22), Some(0.125))
        );
    }

    /// With every Granite multiplier set, the prefill/decode graphs (cached attention) and the uncached
    /// graph (q pre-scaled) agree with the host forward, and differ from the unscaled model.
    #[test]
    fn granite_scales_graph_matches_host_forward() {
        let scaled = GraphLmConfig {
            attention_scale: 0.2,
            embedding_scale: 3.0,
            residual_scale: 0.5,
            logit_scale: 4.0,
            ..GraphLmConfig::smollm_1m()
        };
        let model = GraphLm::new(scaled);
        let plain = GraphLm::new(GraphLmConfig::smollm_1m());
        let tokens = [12u32, 45, 99, 101];
        let want = model.forward(&tokens);
        assert!(want
            .iter()
            .zip(plain.forward(&tokens))
            .any(|(a, b)| (a - b).abs() > 1e-3));

        let v = model.config.vocab_size;
        let run =
            |g: &crate::graph::Graph, w: &crate::graph::Weights, toks: &[u32], pos: &[u32]| {
                let mut state = crate::graph::StateStore::new();
                let inputs = vec![
                    crate::graph::Tensor::U32(toks.to_vec()),
                    crate::graph::Tensor::U32(pos.to_vec()),
                ];
                match crate::graph::cpu::run(g, w, &mut state, inputs)
                    .expect("graph runs")
                    .remove(0)
                {
                    crate::graph::Tensor::F32(l) => l,
                    other => panic!("expected F32 logits, got {other:?}"),
                }
            };
        let pos: Vec<u32> = (0..tokens.len() as u32).collect();
        for cached in [true, false] {
            let (g, w) = model.build_graph(tokens.len(), cached);
            let logits = run(&g, &w, &tokens, &pos);
            let last = &logits[logits.len() - v..];
            for (i, (&got, &exp)) in last.iter().zip(&want).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "cached={cached} logit {i}: graph={got}, host={exp}"
                );
            }
        }
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
    #[derive(Clone, Copy, PartialEq)]
    enum Naming {
        Llama,
        Lfm2Hf,
        Lfm2Gguf,
    }

    /// Synthetic decoder checkpoint (F32 safetensors, HF Llama / LFM2 or llama.cpp LFM2 tensor names)
    /// plus an independent reference of the architecture: no code shared with [`GraphLm`].
    struct Tiny {
        cfg: GraphLmConfig,
        kv_heads: usize,
        qk_norm: bool,
        naming: Naming,
        /// Per layer: LFM2 gated short conv instead of attention.
        conv: Vec<bool>,
        kernel: usize,
        t: std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    }

    impl Tiny {
        fn new(h: usize, heads: usize, kv_heads: usize, head_dim: usize, qk_norm: bool) -> Self {
            Self::build(
                h,
                heads,
                kv_heads,
                head_dim,
                qk_norm,
                Naming::Llama,
                &[false, false],
                0,
            )
        }

        /// LFM2-shaped: hidden 16, 4 heads (2 kv) of 4, q/k norms, a conv layer wherever `conv` is true.
        fn lfm2(naming: Naming, conv: &[bool], kernel: usize) -> Self {
            Self::build(16, 4, 2, 4, true, naming, conv, kernel)
        }

        #[allow(clippy::too_many_arguments)]
        fn build(
            h: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            qk_norm: bool,
            naming: Naming,
            conv: &[bool],
            kernel: usize,
        ) -> Self {
            let (v, inter, layers) = (24, 32, conv.len());
            let mut cfg =
                GraphLmConfig::from_hf_config(v, h, inter, layers, heads, QuantMode::Uncompressed);
            cfg.head_dim = head_dim;
            cfg.max_seq_len = 16;
            let mut tiny = Self {
                cfg,
                kv_heads,
                qk_norm,
                naming,
                conv: conv.to_vec(),
                kernel,
                t: Default::default(),
            };
            let mut seed = 0usize;
            let mut put =
                |tiny: &mut Self, name: String, shape: Vec<usize>, amp: f32, bias: f32| {
                    seed += 1;
                    let n: usize = shape.iter().product();
                    let vals = (0..n)
                        .map(|i| {
                            bias + amp
                                * ((i * 7 + seed * 131) as f32 * 0.37 + (i % 13) as f32).sin()
                        })
                        .collect();
                    tiny.t.insert(name, (shape, vals));
                };
            let n = |t: &Self, l: usize, what: &str| t.name(l, what);
            let name = n(&tiny, 0, "embed");
            put(&mut tiny, name, vec![v, h], 0.5, 0.0);
            let name = n(&tiny, 0, "final_norm");
            put(&mut tiny, name, vec![h], 0.2, 1.0);
            for (l, &is_conv) in conv.iter().enumerate() {
                let mut layer = vec![
                    ("attn_norm", vec![h], 0.2, 1.0),
                    ("ffn_norm", vec![h], 0.2, 1.0),
                    ("gate", vec![inter, h], 0.4, 0.0),
                    ("up", vec![inter, h], 0.4, 0.0),
                    ("down", vec![h, inter], 0.4, 0.0),
                ];
                if is_conv {
                    let conv_w = if naming == Naming::Lfm2Hf {
                        vec![h, 1, kernel]
                    } else {
                        vec![h, kernel]
                    };
                    layer.extend([
                        ("conv_in", vec![3 * h, h], 0.4, 0.0),
                        ("conv_w", conv_w, 0.6, 0.0),
                        ("conv_out", vec![h, h], 0.4, 0.0),
                    ]);
                } else {
                    let a = heads * head_dim;
                    layer.extend([
                        ("q", vec![a, h], 0.4, 0.0),
                        ("k", vec![kv_heads * head_dim, h], 0.4, 0.0),
                        ("v", vec![kv_heads * head_dim, h], 0.4, 0.0),
                        ("o", vec![h, a], 0.4, 0.0),
                    ]);
                    if qk_norm {
                        layer.extend([
                            ("q_norm", vec![head_dim], 0.5, 1.0),
                            ("k_norm", vec![head_dim], 0.5, 1.0),
                        ]);
                    }
                }
                for (what, shape, amp, bias) in layer {
                    let name = n(&tiny, l, what);
                    put(&mut tiny, name, shape, amp, bias);
                }
            }
            tiny
        }

        fn name(&self, l: usize, what: &str) -> String {
            let (hf, g) = (format!("model.layers.{l}"), format!("blk.{l}"));
            match (self.naming, what) {
                (Naming::Lfm2Gguf, "embed") => "token_embd.weight".into(),
                (Naming::Lfm2Gguf, "final_norm") => "token_embd_norm.weight".into(),
                (_, "embed") => "model.embed_tokens.weight".into(),
                (Naming::Llama, "final_norm") => "model.norm.weight".into(),
                (_, "final_norm") => "model.embedding_norm.weight".into(),
                (Naming::Llama, w) => match w {
                    "attn_norm" => format!("{hf}.input_layernorm.weight"),
                    "ffn_norm" => format!("{hf}.post_attention_layernorm.weight"),
                    "q" | "k" | "v" => format!("{hf}.self_attn.{w}_proj.weight"),
                    "o" => format!("{hf}.self_attn.o_proj.weight"),
                    "gate" | "up" | "down" => format!("{hf}.mlp.{w}_proj.weight"),
                    "q_norm" | "k_norm" => format!("{hf}.self_attn.{w}.weight"),
                    _ => unreachable!("{w}"),
                },
                (Naming::Lfm2Hf, w) => match w {
                    "attn_norm" => format!("{hf}.operator_norm.weight"),
                    "ffn_norm" => format!("{hf}.ffn_norm.weight"),
                    "q" | "k" | "v" => format!("{hf}.self_attn.{w}_proj.weight"),
                    "o" => format!("{hf}.self_attn.out_proj.weight"),
                    "gate" => format!("{hf}.feed_forward.w1.weight"),
                    "up" => format!("{hf}.feed_forward.w3.weight"),
                    "down" => format!("{hf}.feed_forward.w2.weight"),
                    "q_norm" => format!("{hf}.self_attn.q_layernorm.weight"),
                    "k_norm" => format!("{hf}.self_attn.k_layernorm.weight"),
                    "conv_in" => format!("{hf}.conv.in_proj.weight"),
                    "conv_w" => format!("{hf}.conv.conv.weight"),
                    "conv_out" => format!("{hf}.conv.out_proj.weight"),
                    _ => unreachable!("{w}"),
                },
                (Naming::Lfm2Gguf, w) => match w {
                    "attn_norm" | "ffn_norm" => format!("{g}.{w}.weight"),
                    "q" | "k" | "v" => format!("{g}.attn_{w}.weight"),
                    "o" => format!("{g}.attn_output.weight"),
                    "gate" | "up" | "down" => format!("{g}.ffn_{w}.weight"),
                    "q_norm" => format!("{g}.attn_q_norm.weight"),
                    "k_norm" => format!("{g}.attn_k_norm.weight"),
                    "conv_in" => format!("{g}.shortconv.in_proj.weight"),
                    "conv_w" => format!("{g}.shortconv.conv.weight"),
                    "conv_out" => format!("{g}.shortconv.out_proj.weight"),
                    _ => unreachable!("{w}"),
                },
            }
        }

        fn bytes(&self) -> Vec<u8> {
            let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
            if self.naming == Naming::Lfm2Gguf {
                header.insert(
                    "__metadata__".into(),
                    serde_json::json!({
                        "general.architecture": "lfm2",
                        "lfm2.shortconv.l_cache": self.kernel.to_string(),
                    }),
                );
            }
            for (name, (shape, vals)) in &self.t {
                let start = data.len();
                vals.iter()
                    .for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
                header.insert(
                    name.clone(),
                    serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
                );
            }
            let header = serde_json::to_vec(&header).unwrap();
            let mut out = (header.len() as u64).to_le_bytes().to_vec();
            out.extend(header);
            out.extend(data);
            out
        }

        fn load(&self) -> Result<GraphLm, String> {
            GraphLm::from_safetensors_data(self.cfg.clone(), &self.bytes())
        }

        /// Last-position logits of a pre-norm decoder (half-split RoPE, SwiGLU, GQA by repeating kv heads,
        /// optional per-head q/k RMSNorm before RoPE, optional LFM2 gated short-conv layers), tied head.
        fn reference(&self, tokens: &[u32]) -> Vec<f32> {
            let c = &self.cfg;
            let (h, nh, hd, kvh) = (c.hidden_dim, c.num_heads, c.head_dim, self.kv_heads);
            let w = |l: usize, what: &str| &self.t[&self.name(l, what)].1;
            let mv = |m: &[f32], x: &[f32]| -> Vec<f32> {
                m.chunks(x.len())
                    .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
                    .collect()
            };
            let rms = |x: &[f32], g: &[f32]| -> Vec<f32> {
                let s = 1.0
                    / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + c.norm_eps).sqrt();
                x.iter().zip(g).map(|(v, g)| v * s * g).collect()
            };
            let emb = w(0, "embed");
            let mut xs: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| emb[t as usize * h..][..h].to_vec())
                .collect();
            for l in 0..c.num_layers {
                // Per position: the mixer's output before the residual add.
                let mut mixed: Vec<Vec<f32>> = vec![];
                let n1: Vec<Vec<f32>> = xs.iter().map(|x| rms(x, w(l, "attn_norm"))).collect();
                if self.conv[l] {
                    let (mut cs, mut bx) = (vec![], vec![]);
                    for n in &n1 {
                        let bcx = mv(w(l, "conv_in"), n);
                        cs.push(bcx[h..2 * h].to_vec());
                        bx.push(
                            (0..h)
                                .map(|i| bcx[i] * bcx[2 * h + i])
                                .collect::<Vec<f32>>(),
                        );
                    }
                    let k = self.kernel;
                    for t in 0..xs.len() {
                        let y: Vec<f32> = (0..h)
                            .map(|ch| {
                                let conv: f32 = (0..k)
                                    .filter(|tap| t + tap + 1 >= k)
                                    .map(|tap| {
                                        w(l, "conv_w")[ch * k + tap] * bx[t + tap + 1 - k][ch]
                                    })
                                    .sum();
                                cs[t][ch] * conv
                            })
                            .collect();
                        mixed.push(mv(w(l, "conv_out"), &y));
                    }
                } else {
                    let (mut qs, mut ks, mut vs) = (vec![], vec![], vec![]);
                    for (pos, n) in n1.iter().enumerate() {
                        let mut q = mv(w(l, "q"), n);
                        let mut k = mv(w(l, "k"), n);
                        vs.push(mv(w(l, "v"), n));
                        for (x, heads, what) in [(&mut q, nh, "q_norm"), (&mut k, kvh, "k_norm")] {
                            for head in x.chunks_mut(hd) {
                                if self.qk_norm {
                                    head.copy_from_slice(&rms(head, w(l, what)));
                                }
                                for d in 0..hd / 2 {
                                    let th = pos as f32
                                        * c.rope_theta.powf(-((2 * d) as f32) / hd as f32);
                                    let (x0, x1) = (head[d], head[d + hd / 2]);
                                    head[d] = x0 * th.cos() - x1 * th.sin();
                                    head[d + hd / 2] = x1 * th.cos() + x0 * th.sin();
                                }
                            }
                            assert_eq!(x.len(), heads * hd);
                        }
                        qs.push(q);
                        ks.push(k);
                    }
                    for t in 0..xs.len() {
                        let mut att = vec![0.0f32; nh * hd];
                        for head in 0..nh {
                            let kv = head / (nh / kvh);
                            let sc: Vec<f32> = (0..=t)
                                .map(|s| {
                                    let d: f32 = (0..hd)
                                        .map(|d| qs[t][head * hd + d] * ks[s][kv * hd + d])
                                        .sum();
                                    d / (hd as f32).sqrt()
                                })
                                .collect();
                            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
                            let e: Vec<f32> = sc.iter().map(|v| (v - m).exp()).collect();
                            let z: f32 = e.iter().sum();
                            for (s, e) in e.iter().enumerate() {
                                for d in 0..hd {
                                    att[head * hd + d] += e / z * vs[s][kv * hd + d];
                                }
                            }
                        }
                        mixed.push(mv(w(l, "o"), &att));
                    }
                }
                for (x, o) in xs.iter_mut().zip(&mixed) {
                    x.iter_mut().zip(o).for_each(|(x, o)| *x += o);
                    let n2 = rms(x, w(l, "ffn_norm"));
                    let (g, u) = (mv(w(l, "gate"), &n2), mv(w(l, "up"), &n2));
                    let a: Vec<f32> = g
                        .iter()
                        .zip(&u)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect();
                    let d = mv(w(l, "down"), &a);
                    x.iter_mut().zip(&d).for_each(|(x, d)| *x += d);
                }
            }
            let last = rms(xs.last().unwrap(), w(0, "final_norm"));
            mv(emb, &last)
        }
    }

    /// Runs `model`'s prefill and cached-decode graphs on the CPU executor; last-row logits per path.
    fn graph_logits(model: &GraphLm, tokens: &[u32]) -> Vec<Vec<f32>> {
        let v = model.config.vocab_size;
        let pos: Vec<u32> = (0..tokens.len() as u32).collect();
        [true, false]
            .into_iter()
            .map(|cached| {
                let (g, w) = model.build_graph(tokens.len(), cached);
                let inputs = vec![
                    crate::graph::Tensor::U32(tokens.to_vec()),
                    crate::graph::Tensor::U32(pos.clone()),
                ];
                let mut state = crate::graph::StateStore::new();
                match crate::graph::cpu::run(&g, &w, &mut state, inputs)
                    .expect("graph runs")
                    .remove(0)
                {
                    crate::graph::Tensor::F32(l) => l[l.len() - v..].to_vec(),
                    other => panic!("expected F32 logits, got {other:?}"),
                }
            })
            .collect()
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < 1e-3 * (1.0 + w.abs()),
                "{what} logit {i}: got {g}, want {w}"
            );
        }
    }

    /// Qwen3-style per-head q/k RMSNorm: the graph (prefill and cached) and the host forward match an
    /// independent reference, and the norm visibly changes the logits.
    #[test]
    fn qk_norm_graph_and_host_match_reference() {
        let _g = crate::hardware::cpu_compute::allow();
        let tokens = [3u32, 17, 5, 9];
        let tiny = Tiny::new(16, 2, 2, 8, true);
        let want = tiny.reference(&tokens);
        let plain = Tiny {
            qk_norm: false,
            ..Tiny::new(16, 2, 2, 8, false)
        };
        assert!(want
            .iter()
            .zip(plain.reference(&tokens))
            .any(|(a, b)| (a - b).abs() > 1e-2));

        let model = tiny.load().expect("loads with q/k norms");
        assert!(model
            .layers
            .iter()
            .all(|l| l.q_norm.is_some() && l.k_norm.is_some()));
        assert_close(&model.forward(&tokens), &want, "host");
        for (cached, got) in [true, false].into_iter().zip(graph_logits(&model, &tokens)) {
            assert_close(&got, &want, &format!("graph cached={cached}"));
        }
    }

    /// Norm tensors the decoder does not consume still fail the load: full-width (OLMo2-style) gains,
    /// after-RoPE hunyuan names, and Gemma 3 (norms plus layers this decoder lacks).
    #[test]
    fn unconsumed_qk_norm_tensors_are_rejected() {
        let mut full = Tiny::new(16, 2, 2, 8, true);
        let wide = (vec![16], vec![1.0; 16]);
        full.t
            .insert("model.layers.0.self_attn.q_norm.weight".into(), wide);
        let e = full.load().expect_err("full-width q norm");
        assert!(e.contains("head_dim"), "{e}");

        let mut hunyuan = Tiny::new(16, 2, 2, 8, false);
        hunyuan.t.insert(
            "model.layers.0.self_attn.query_layernorm.weight".into(),
            (vec![8], vec![1.0; 8]),
        );
        assert!(hunyuan
            .load()
            .expect_err("after-RoPE norm")
            .contains("not implemented"));

        let mut gemma = Tiny::new(16, 2, 2, 8, true);
        gemma.t.insert(
            "model.layers.0.pre_feedforward_layernorm.weight".into(),
            (vec![16], vec![1.0; 16]),
        );
        assert!(gemma.load().expect_err("gemma3").contains("sliding-window"));
    }

    /// Attention wider than the hidden size (ERNIE-4.5, Qwen3: `head_dim` 128 with hidden 1024 and 16
    /// heads) with GQA: loads, and the graph and host forward match the reference.
    #[test]
    fn attention_wider_than_hidden_with_gqa_matches_reference() {
        let _g = crate::hardware::cpu_compute::allow();
        let tokens = [3u32, 17, 5, 9];
        for qk_norm in [false, true] {
            let tiny = Tiny::new(16, 4, 2, 8, qk_norm);
            let model = tiny.load().expect("loads");
            let a = model.config.attn_dim();
            assert_eq!(a, 32);
            assert!(model.layers.iter().all(|l| {
                (l.qkv_proj.in_features, l.qkv_proj.out_features) == (16, 3 * a)
                    && (l.out_proj.in_features, l.out_proj.out_features) == (a, 16)
            }));
            let want = tiny.reference(&tokens);
            assert_close(
                &model.forward(&tokens),
                &want,
                &format!("host qk_norm={qk_norm}"),
            );
            for (cached, got) in [true, false].into_iter().zip(graph_logits(&model, &tokens)) {
                assert_close(
                    &got,
                    &want,
                    &format!("graph cached={cached} qk_norm={qk_norm}"),
                );
            }
            // The per-token KV-cache path (what the CPU scheduler runs) agrees too.
            let mut kv: Vec<LayerKVCache> = (0..model.layers.len())
                .map(|_| LayerKVCache::new(model.config.max_seq_len, model.config.hidden_dim))
                .collect();
            let mut last = vec![];
            for (p, &t) in tokens.iter().enumerate() {
                last = model.forward_token_with_kv(t, p, &mut kv);
            }
            assert_close(&last, &want, &format!("host kv qk_norm={qk_norm}"));
        }
    }

    /// `head_dim` comes from HF `head_dim` and GGUF `attention.key_length`, else hidden / heads.
    #[test]
    fn explicit_head_dim_reaches_the_config() {
        let hf = |extra: &str| -> GraphLmConfig {
            let j = format!(
                r#"{{"hidden_size":1024,"num_hidden_layers":2,"num_attention_heads":16{extra}}}"#
            );
            GraphLmConfig::from_hf(&serde_json::from_str(&j).unwrap(), QuantMode::Uncompressed)
        };
        assert_eq!(hf("").head_dim, 64);
        let wide = hf(r#","head_dim":128"#);
        assert_eq!((wide.head_dim, wide.attn_dim()), (128, 2048));

        let base = GraphLmConfig::smollm_1m();
        let gguf = |extra: &[(&str, &str)]| {
            let mut e = vec![("embedding_length", "1024"), ("attention.head_count", "16")];
            e.extend_from_slice(extra);
            GraphLmConfig::from_gguf(&granite_manifest("ernie4_5", &e), base.clone())
        };
        assert_eq!(gguf(&[]).head_dim, 64);
        assert_eq!(gguf(&[("attention.key_length", "128")]).attn_dim(), 2048);
    }

    /// Token-at-a-time decode graph with persistent state (KV rows, conv windows): last logits.
    fn decode_logits(model: &GraphLm, tokens: &[u32]) -> Vec<f32> {
        let (g, w) = model.build_graph(1, true);
        let mut state = crate::graph::StateStore::new();
        let mut last = vec![];
        for (p, &t) in tokens.iter().enumerate() {
            let inputs = vec![
                crate::graph::Tensor::U32(vec![t]),
                crate::graph::Tensor::U32(vec![p as u32]),
            ];
            last = match crate::graph::cpu::run(&g, &w, &mut state, inputs)
                .expect("decode runs")
                .remove(0)
            {
                crate::graph::Tensor::F32(l) => l,
                other => panic!("expected F32 logits, got {other:?}"),
            };
        }
        last
    }

    /// LFM2 (HF and llama.cpp names): hybrid gated short-conv + attention layers with q/k norms and GQA.
    /// Prefill graph, cached graph, token-by-token decode graph, host forward and host KV path all match
    /// an independent reference; the conv taps reach back further than the first tokens.
    #[test]
    fn lfm2_short_conv_hybrid_matches_reference() {
        let _g = crate::hardware::cpu_compute::allow();
        let tokens = [3u32, 17, 5, 9, 21, 2];
        for naming in [Naming::Lfm2Hf, Naming::Lfm2Gguf] {
            let tiny = Tiny::lfm2(naming, &[true, false, true, true, false], 3);
            let model = tiny.load().expect("loads");
            assert_eq!(
                model
                    .layers
                    .iter()
                    .map(|l| l.conv.is_some())
                    .collect::<Vec<_>>(),
                [true, false, true, true, false]
            );
            let want = tiny.reference(&tokens);
            assert_close(&model.forward(&tokens), &want, "host");
            for (cached, got) in [true, false].into_iter().zip(graph_logits(&model, &tokens)) {
                assert_close(&got, &want, &format!("graph cached={cached}"));
            }
            assert_close(&decode_logits(&model, &tokens), &want, "decode graph");
            let mut kv: Vec<LayerKVCache> = (0..model.layers.len())
                .map(|_| LayerKVCache::new(model.config.max_seq_len, model.config.hidden_dim))
                .collect();
            let mut last = vec![];
            for (p, &t) in tokens.iter().enumerate() {
                last = model.forward_token_with_kv(t, p, &mut kv);
            }
            assert_close(&last, &want, "host kv");
        }
    }

    /// A conv bias or a taps count disagreeing with `lfm2.shortconv.l_cache` fails the load.
    #[test]
    fn lfm2_conv_bias_and_cache_mismatch_are_rejected() {
        let mut bias = Tiny::lfm2(Naming::Lfm2Hf, &[true, false], 3);
        bias.t.insert(
            "model.layers.0.conv.conv.bias".into(),
            (vec![16], vec![0.0; 16]),
        );
        assert!(bias.load().expect_err("bias").contains("conv bias"));
        let mut cache = Tiny::lfm2(Naming::Lfm2Gguf, &[true, false], 3);
        cache.kernel = 2;
        assert!(cache.load().expect_err("l_cache").contains("l_cache"));
    }

    /// LFM2 `config.json`: `intermediate_size` 6656 is the pre-adjust width (real MLP 4608), `norm_eps` is the epsilon.
    #[test]
    fn lfm2_hf_config_resolves_ff_dim_and_eps() {
        let hf: crate::model::hf_loader::HfConfigJson = serde_json::from_str(
            r#"{"model_type":"lfm2","hidden_size":1024,"num_hidden_layers":16,"num_attention_heads":16,
                "intermediate_size":6656,"block_auto_adjust_ff_dim":true,"block_ffn_dim_multiplier":1.0,
                "block_multiple_of":256,"norm_eps":2e-5,"num_heads":16,"num_key_value_heads":8}"#,
        )
        .unwrap();
        let cfg = GraphLmConfig::from_hf(&hf, QuantMode::Uncompressed);
        assert_eq!((cfg.intermediate_dim, cfg.norm_eps), (4608, 2e-5));
        // `num_attention_heads` and `num_heads` together parse (no duplicate field); each alone works too.
        assert_eq!((cfg.num_heads, hf.resolved_num_kv_heads()), (16, 8));
        for key in ["num_heads", "n_heads", "num_attention_heads"] {
            let hf: crate::model::hf_loader::HfConfigJson =
                serde_json::from_str(&format!(r#"{{"hidden_size":64,"{key}":8}}"#)).unwrap();
            assert_eq!(hf.resolved_num_heads(), 8, "{key}");
        }
    }

    /// Granite-4 HF (`GraniteMoeHybrid`, attention-only layers): the MLP lives in `shared_mlp.input_linear`
    /// (fused gate; up) and `output_linear`; it loads exactly like the Llama gate/up/down names. Expert
    /// layers (`block_sparse_moe`) fail instead of running without their experts.
    #[test]
    fn granite_hf_shared_mlp_names_load_like_gate_up_down() {
        let _g = crate::hardware::cpu_compute::allow();
        let tokens = [3u32, 17, 5, 9];
        let mut tiny = Tiny::new(16, 2, 2, 8, false);
        let want = tiny.reference(&tokens);
        for l in 0..tiny.cfg.num_layers {
            let p = format!("model.layers.{l}");
            let (gate, up) = (
                tiny.t.remove(&format!("{p}.mlp.gate_proj.weight")).unwrap(),
                tiny.t.remove(&format!("{p}.mlp.up_proj.weight")).unwrap(),
            );
            let fused = (
                [gate.0[0] + up.0[0], gate.0[1]].to_vec(),
                [gate.1, up.1].concat(),
            );
            tiny.t
                .insert(format!("{p}.shared_mlp.input_linear.weight"), fused);
            let down = tiny.t.remove(&format!("{p}.mlp.down_proj.weight")).unwrap();
            tiny.t
                .insert(format!("{p}.shared_mlp.output_linear.weight"), down);
        }
        let model = tiny.load().expect("loads");
        assert_close(&model.forward(&tokens), &want, "host");
        tiny.t.insert(
            "model.layers.0.block_sparse_moe.router.layer.weight".into(),
            (vec![2, 16], vec![0.0; 32]),
        );
        assert!(tiny.load().expect_err("moe").contains("expert"));
    }

    /// A speculative / draft rewind lowers `LayerKVCache::len` without truncating `k`: conv layers must
    /// ignore the stale rows past `len`, and the graph's conv state must refuse `StateStore::truncate`.
    #[test]
    fn lfm2_rewind_ignores_stale_rows_and_conv_state_cannot_truncate() {
        let _g = crate::hardware::cpu_compute::allow();
        let tokens = [3u32, 17, 5, 9, 21, 2];
        let tiny = Tiny::lfm2(Naming::Lfm2Hf, &[true, false, true], 3);
        let model = tiny.load().expect("loads");
        let mut kv: Vec<LayerKVCache> = (0..model.layers.len())
            .map(|_| LayerKVCache::new(model.config.max_seq_len, model.config.hidden_dim))
            .collect();
        for (p, &t) in tokens.iter().enumerate() {
            model.forward_token_with_kv(t, p, &mut kv);
        }
        // Reject the last three tokens, then replay them.
        kv.iter_mut().for_each(|c| c.len = 3);
        let mut last = vec![];
        for (p, &t) in tokens.iter().enumerate().skip(3) {
            last = model.forward_token_with_kv(t, p, &mut kv);
        }
        assert_close(&last, &tiny.reference(&tokens), "after rewind");

        let (g, w) = model.build_graph(1, true);
        let mut state = crate::graph::StateStore::new();
        let inputs = vec![
            crate::graph::Tensor::U32(vec![3]),
            crate::graph::Tensor::U32(vec![0]),
        ];
        crate::graph::cpu::run(&g, &w, &mut state, inputs).expect("decode runs");
        assert!(state
            .truncate(0, &g.recurrent_states())
            .unwrap_err()
            .contains("conv_state"));
    }

    /// `graph::shape::infer` reproduces every node of the GraphLm graphs (prefill, cached and
    /// uncached, decode, embeds decode, seq-cls) across the config variants and fixtures.
    #[test]
    fn graph_shapes_match_inference() {
        use crate::graph::shape::assert_shapes;
        let base = GraphLmConfig::smollm_1m;
        let mut models: Vec<GraphLm> = [
            base(),
            GraphLmConfig {
                attention_scale: 0.2,
                embedding_scale: 3.0,
                residual_scale: 0.5,
                logit_scale: 4.0,
                ..base()
            },
            GraphLmConfig {
                use_parallel_residual: true,
                use_standard_layernorm: true,
                // GELU MLPs need an up-only `gate_up_proj` (random init makes it gate+up): the
                // `decoder_gpu::tiny_graph_lm` variants cover them.
                rotary_pct: 0.5,
                ..base()
            },
            GraphLmConfig {
                rope_interleaved: true,
                ..base()
            },
            GraphLmConfig {
                act_quant_bits: 8,
                use_relu2_mlp: true,
                ..base()
            },
        ]
        .into_iter()
        .map(GraphLm::new)
        .collect();
        for tiny in [
            Tiny::new(16, 2, 2, 8, true),
            Tiny::new(16, 4, 2, 8, false),
            Tiny::lfm2(Naming::Lfm2Hf, &[true, false, true], 3),
            Tiny::lfm2(Naming::Lfm2Gguf, &[true, false], 3),
        ] {
            models.push(tiny.load().expect("loads"));
        }
        for m in &models {
            for cached in [true, false] {
                assert_shapes(&m.build_graph(5, cached).0);
            }
            assert_shapes(&m.to_decode_graph().0);
            assert_shapes(&m.to_decode_graph_embeds().0);
            let score = vec![0.1; 3 * m.config.hidden_dim];
            assert_shapes(&m.build_seqcls_graph(5, &score, None));
        }
    }
}
