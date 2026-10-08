//! Gemma 3 text decoder (`gemma3_text`, transformers `Gemma3TextModel`) for
//! `Gemma3TextForSequenceClassification`: sqrt(hidden)-scaled embeddings, `(1 + w)` RMSNorms
//! before and after attention and the GeGLU (`gelu_pytorch_tanh`) MLP, q/k RMSNorm,
//! `query_pre_attn_scalar^-0.5` scores, sliding-window and global layers with their own RoPE
//! (default / linear / yarn), final norm. One uncached causal pass over bf16 (memory-mapped) or
//! f32 weights; the classifier scores the last token's final-norm hidden state.

use crate::model::classifier::{encode_pair, DecoderBackbone};
use crate::model::gemma4::ops::{gelu_mul, linear, rms_norm, rope, rope_table};
use crate::model::hub::ModelArtifacts;
use crate::model::par::{par_for, OutPtr};
use crate::model::simd::{vector_dot_product, vector_fused_scale_add, vector_softmax};
use crate::model::tokenizer::Tokenizer;
use crate::model::weights::{Mat, Weights};
use crate::model::{bf16_to_f32, f32_to_bf16, nn};
use serde_json::{Map, Value};

/// A weight matrix `[rows, cols]`: bf16 borrowed from the mapped checkpoint, else decoded to f32.
enum W {
    Bf16(Mat),
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
}

impl W {
    fn load(w: &Weights, name: &str) -> Result<Self, String> {
        // `mat` would round f32/f16 to bf16 (ncii-guard's merged LoRA lives in f32).
        if w.is_bf16(name) {
            return Ok(Self::Bf16(w.mat(name)?));
        }
        let shape = w.shape(name)?;
        let (rows, cols) = (shape[0], shape[1..].iter().product());
        Ok(Self::F32 {
            w: w.vec(name)?,
            rows,
            cols,
        })
    }

    fn dims(&self) -> (usize, usize) {
        match self {
            Self::Bf16(m) => (m.rows, m.cols),
            Self::F32 { rows, cols, .. } => (*rows, *cols),
        }
    }

    /// x `[n, cols]` . w^T -> `[n, rows]`.
    fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        match self {
            Self::Bf16(m) => linear(x, n, m),
            Self::F32 { w, rows, cols } => {
                let mut out = vec![0.0f32; n * rows];
                nn::matmul_nt(x, w, None, n, *cols, *rows, &mut out);
                out
            }
        }
    }

    /// Row `r` times `scale` into `out`.
    fn row(&self, r: usize, scale: f32, out: &mut [f32]) {
        match self {
            Self::Bf16(m) => {
                for (o, &v) in out.iter_mut().zip(m.row(r)) {
                    *o = bf16_to_f32(v) * scale;
                }
            }
            Self::F32 { w, cols, .. } => {
                for (o, &v) in out.iter_mut().zip(&w[r * cols..(r + 1) * cols]) {
                    *o = v * scale;
                }
            }
        }
    }
}

struct Layer {
    sliding: bool,
    q: W,
    k: W,
    v: W,
    o: W,
    gate: W,
    up: W,
    down: W,
    /// RMSNorm gains, stored as `1 + w`.
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
}

pub struct Gemma3Text {
    tokenizer: Tokenizer,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    /// `query_pre_attn_scalar^-0.5`.
    scale: f32,
    attn_softcap: Option<f32>,
    window: usize,
    max_pos: usize,
    embed: W,
    vocab: usize,
    embed_scale: f32,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    /// (inverse frequencies, cos/sin scale) per layer type.
    rope_full: (Vec<f32>, f32),
    rope_sliding: (Vec<f32>, f32),
}

/// `rope_parameters[ty]` with transformers' legacy fill-ins: `rope_scaling` merges into the
/// global layers' entry; a missing `rope_theta` comes from `rope_theta` (global) or
/// `rope_local_base_freq` (sliding), else 1e6 / 1e4.
fn rope_params(c: &Value, ty: &str) -> Map<String, Value> {
    let mut p = c["rope_parameters"][ty]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let full = ty == "full_attention";
    if let (true, Some(s)) = (full, c["rope_scaling"].as_object()) {
        p.extend(s.clone());
    }
    if p.get("rope_theta").and_then(Value::as_f64).is_none() {
        let (key, default) = if full {
            ("rope_theta", 1e6)
        } else {
            ("rope_local_base_freq", 1e4)
        };
        p.insert(
            "rope_theta".into(),
            c[key].as_f64().unwrap_or(default).into(),
        );
    }
    p
}

/// Inverse frequencies and the cos/sin scale (`attention_factor`) of one RoPE entry, as
/// transformers' `default`, `linear` and `yarn` init functions compute them.
fn inv_freq(
    p: &Map<String, Value>,
    dim: usize,
    max_pos: usize,
    truncate: bool,
) -> Result<(Vec<f32>, f32), String> {
    let f = |k: &str| p.get(k).and_then(Value::as_f64);
    let base = f("rope_theta").ok_or("rope_theta missing")?;
    if f("partial_rotary_factor").is_some_and(|v| v != 1.0) {
        return Err("partial_rotary_factor is not supported".into());
    }
    let ty = ["rope_type", "type"]
        .iter()
        .find_map(|k| p.get(*k).and_then(Value::as_str))
        .unwrap_or("default");
    let pos_freq: Vec<f64> = (0..dim / 2)
        .map(|i| base.powf(2.0 * i as f64 / dim as f64))
        .collect();
    let inv = |scale: f64| -> Vec<f32> {
        pos_freq
            .iter()
            .map(|p| (1.0 / (scale * p)) as f32)
            .collect()
    };
    match ty {
        "default" => Ok((inv(1.0), 1.0)),
        "linear" => Ok((inv(f("factor").ok_or("linear rope: factor missing")?), 1.0)),
        "yarn" => {
            let orig = f("original_max_position_embeddings")
                .ok_or("yarn rope: original_max_position_embeddings missing")?;
            let factor = f("factor").unwrap_or(max_pos as f64 / orig);
            let mscale = |scale: f64, m: f64| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * m * scale.ln() + 1.0
                }
            };
            let attn = match (f("attention_factor"), f("mscale"), f("mscale_all_dim")) {
                (Some(a), _, _) => a,
                (None, Some(m), Some(d)) if m != 0.0 && d != 0.0 => {
                    mscale(factor, m) / mscale(factor, d)
                }
                _ => mscale(factor, 1.0),
            };
            let beta_fast = f("beta_fast").filter(|&v| v != 0.0).unwrap_or(32.0);
            let beta_slow = f("beta_slow").filter(|&v| v != 0.0).unwrap_or(1.0);
            let correction = |rot: f64| {
                dim as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
            };
            let (mut low, mut high) = (correction(beta_fast), correction(beta_slow));
            if truncate {
                (low, high) = (low.floor(), high.ceil());
            }
            let (low, mut high) = (low.max(0.0), high.min(dim as f64 - 1.0));
            if low == high {
                high += 0.001;
            }
            let (extra, inter) = (inv(1.0), inv(factor));
            let freqs = (0..dim / 2)
                .map(|i| {
                    let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0) as f32;
                    let ef = 1.0 - ramp;
                    inter[i] * (1.0 - ef) + extra[i] * ef
                })
                .collect();
            Ok((freqs, attn as f32))
        }
        other => Err(format!("rope_type '{other}' is not supported")),
    }
}

impl Gemma3Text {
    pub fn load(artifacts: &ModelArtifacts, c: &Value) -> Result<Self, String> {
        let weights = artifacts.weights_path.as_deref().ok_or("no weights file")?;
        let tok = artifacts
            .tokenizer_path
            .as_deref()
            .ok_or("no tokenizer.json")?;
        let u = |k: &str, default: Option<usize>| {
            c[k].as_u64()
                .map(|v| v as usize)
                .or(default)
                .ok_or_else(|| format!("config.json has no {k}"))
        };
        if c["use_bidirectional_attention"].as_bool() == Some(true) {
            return Err("Gemma 3 with use_bidirectional_attention is not supported".into());
        }
        if c["attention_bias"].as_bool() == Some(true) {
            return Err("Gemma 3 with attention_bias is not supported".into());
        }
        let act = c["hidden_activation"]
            .as_str()
            .unwrap_or("gelu_pytorch_tanh");
        if act != "gelu_pytorch_tanh" {
            return Err(format!("hidden_activation {act} is not supported"));
        }
        let hidden = u("hidden_size", None)?;
        let n = u("num_hidden_layers", None)?;
        let heads = u("num_attention_heads", None)?;
        let kv_heads = u("num_key_value_heads", None)?;
        let head_dim = u("head_dim", Some(256))?;
        let max_pos = u("max_position_embeddings", Some(131_072))?;
        let eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let sliding: Vec<bool> = match c["layer_types"].as_array() {
            Some(t) => t
                .iter()
                .map(|t| t.as_str() == Some("sliding_attention"))
                .collect(),
            None => {
                let pattern = u("sliding_window_pattern", Some(6))?.max(1);
                (0..n).map(|i| (i + 1) % pattern != 0).collect()
            }
        };
        if sliding.len() != n || heads % kv_heads.max(1) != 0 {
            return Err("config.json: layer_types / head counts disagree".into());
        }
        let w = Weights::open(weights)?;
        // `(1 + w)` RMSNorm gains.
        let gain = |name: &str, len: usize| -> Result<Vec<f32>, String> {
            Ok(w.vec_n(name, len)?.into_iter().map(|v| 1.0 + v).collect())
        };
        let mut layers = Vec::with_capacity(n);
        for (i, &sliding) in sliding.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let m = |s: &str| W::load(&w, &format!("{p}.{s}.weight"));
            let l = Layer {
                sliding,
                q: m("self_attn.q_proj")?,
                k: m("self_attn.k_proj")?,
                v: m("self_attn.v_proj")?,
                o: m("self_attn.o_proj")?,
                gate: m("mlp.gate_proj")?,
                up: m("mlp.up_proj")?,
                down: m("mlp.down_proj")?,
                q_norm: gain(&format!("{p}.self_attn.q_norm.weight"), head_dim)?,
                k_norm: gain(&format!("{p}.self_attn.k_norm.weight"), head_dim)?,
                input_ln: gain(&format!("{p}.input_layernorm.weight"), hidden)?,
                post_attn_ln: gain(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
                pre_ff_ln: gain(&format!("{p}.pre_feedforward_layernorm.weight"), hidden)?,
                post_ff_ln: gain(&format!("{p}.post_feedforward_layernorm.weight"), hidden)?,
            };
            if l.q.dims() != (heads * head_dim, hidden)
                || l.k.dims() != (kv_heads * head_dim, hidden)
                || l.v.dims() != (kv_heads * head_dim, hidden)
                || l.o.dims() != (hidden, heads * head_dim)
                || l.gate.dims().1 != hidden
                || l.down.dims() != (hidden, l.gate.dims().0)
            {
                return Err(format!(
                    "layer {i}: projection shapes disagree with config.json"
                ));
            }
            layers.push(l);
        }
        let embed = W::load(&w, "model.embed_tokens.weight")?;
        let (vocab, cols) = embed.dims();
        if cols != hidden {
            return Err("model.embed_tokens.weight disagrees with hidden_size".into());
        }
        // transformers casts sqrt(hidden) to the embedding dtype before scaling.
        let mut embed_scale = (hidden as f32).sqrt();
        if matches!(embed, W::Bf16(_)) {
            embed_scale = bf16_to_f32(f32_to_bf16(embed_scale));
        }
        let truncate = c["rope_parameters"]["truncate"].as_bool().unwrap_or(true);
        let rope = |ty| inv_freq(&rope_params(c, ty), head_dim, max_pos, truncate);
        Ok(Self {
            tokenizer: crate::model::hf_loader::load_json_tokenizer(tok)?,
            hidden,
            heads,
            kv_heads,
            head_dim,
            eps,
            scale: (c["query_pre_attn_scalar"].as_f64().unwrap_or(256.0) as f32).powf(-0.5),
            attn_softcap: c["attn_logit_softcapping"].as_f64().map(|v| v as f32),
            window: u("sliding_window", Some(4096))?,
            max_pos,
            embed,
            vocab,
            embed_scale,
            layers,
            norm: gain("model.norm.weight", hidden)?,
            rope_full: rope("full_attention")?,
            rope_sliding: rope("sliding_attention")?,
        })
    }

    /// Causal self-attention of `n` rows (positions `0..n`), sliding window on sliding layers.
    fn attention(
        &self,
        l: &Layer,
        a: &[f32],
        n: usize,
        ropes: &[(Vec<f32>, Vec<f32>)],
    ) -> Vec<f32> {
        let (hd, nh, kvh) = (self.head_dim, self.heads, self.kv_heads);
        let mut q = l.q.apply(a, n);
        let mut k = l.k.apply(a, n);
        let v = l.v.apply(a, n);
        rms_norm(&mut q, hd, Some(&l.q_norm), self.eps);
        rms_norm(&mut k, hd, Some(&l.k_norm), self.eps);
        for (r, (c, s)) in ropes.iter().enumerate() {
            q[r * nh * hd..(r + 1) * nh * hd]
                .chunks_exact_mut(hd)
                .for_each(|x| rope(x, c, s));
            k[r * kvh * hd..(r + 1) * kvh * hd]
                .chunks_exact_mut(hd)
                .for_each(|x| rope(x, c, s));
        }
        let group = nh / kvh;
        let window = if l.sliding { self.window } else { usize::MAX };
        let mut out = vec![0.0f32; n * nh * hd];
        let o = OutPtr::new(&mut out);
        par_for(n * nh, &|t| {
            let (r, head) = (t / nh, t % nh);
            let lo = (r + 1).saturating_sub(window);
            let kh = head / group;
            let qv = &q[(r * nh + head) * hd..][..hd];
            let mut s: Vec<f32> = (lo..=r)
                .map(|p| vector_dot_product(qv, &k[(p * kvh + kh) * hd..][..hd]) * self.scale)
                .collect();
            if let Some(cap) = self.attn_softcap {
                s.iter_mut().for_each(|v| *v = (*v / cap).tanh() * cap);
            }
            vector_softmax(&mut s);
            // SAFETY: (row, head) slot belongs to task t.
            let dst = unsafe { o.slice((r * nh + head) * hd, hd) };
            for (p, &w) in (lo..=r).zip(&s) {
                vector_fused_scale_add(dst, &v[(p * kvh + kh) * hd..][..hd], w);
            }
        });
        out
    }
}

impl DecoderBackbone for Gemma3Text {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn context_len(&self) -> usize {
        self.max_pos
    }
    fn hidden(&self) -> usize {
        self.hidden
    }
    /// HF `tokenizer(text)`: the post-processor's specials included (BOS, and EOS where it
    /// appends one); a pair goes through [`encode_pair`].
    fn encode(&self, text: &str, pair: Option<&str>) -> Result<Vec<u32>, String> {
        if let Some(answer) = pair {
            return encode_pair(&self.tokenizer, text, answer);
        }
        match &self.tokenizer.hf {
            Some(hf) => hf
                .encode_fast(text, true)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| format!("tokenize: {e}")),
            None => Ok(self.tokenizer.encode(text)),
        }
    }
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let (h, n) = (self.hidden, ids.len());
        let mut x = vec![0.0f32; n * h];
        for (row, &id) in x.chunks_exact_mut(h).zip(ids) {
            if id as usize >= self.vocab {
                return Err(format!("token id {id} is outside the vocabulary"));
            }
            self.embed.row(id as usize, self.embed_scale, row);
        }
        let tables = |(inv, scale): &(Vec<f32>, f32)| -> Vec<(Vec<f32>, Vec<f32>)> {
            (0..n)
                .map(|p| {
                    let (mut c, mut s) = rope_table(p as f32, inv);
                    c.iter_mut().chain(s.iter_mut()).for_each(|v| *v *= scale);
                    (c, s)
                })
                .collect()
        };
        let (rope_f, rope_s) = (tables(&self.rope_full), tables(&self.rope_sliding));
        for l in &self.layers {
            let mut a = x.clone();
            rms_norm(&mut a, h, Some(&l.input_ln), self.eps);
            let ropes = if l.sliding { &rope_s } else { &rope_f };
            let mut o = l.o.apply(&self.attention(l, &a, n, ropes), n);
            rms_norm(&mut o, h, Some(&l.post_attn_ln), self.eps);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);

            let mut a = x.clone();
            rms_norm(&mut a, h, Some(&l.pre_ff_ln), self.eps);
            let mut g = l.gate.apply(&a, n);
            gelu_mul(&mut g, &l.up.apply(&a, n));
            let mut d = l.down.apply(&g, n);
            rms_norm(&mut d, h, Some(&l.post_ff_ln), self.eps);
            x.iter_mut().zip(&d).for_each(|(x, d)| *x += d);
        }
        let mut last = x.split_off((n - 1) * h);
        rms_norm(&mut last, h, Some(&self.norm), self.eps);
        Ok(last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// transformers `_compute_yarn_parameters` for OTel-LLM-8.3B's global layers (factor 4,
    /// 8192 original positions, beta 64/1, base 1e4, dim 128): the scale is 0.1 ln 4 + 1; the
    /// ramp runs over pairs 20..=50 (floor 20.95 / ceil 49.84 of the correction dims), so pairs up
    /// to 20 keep the unscaled frequency and pairs from 50 on are divided by 4.
    #[test]
    fn yarn_matches_transformers() {
        let p: Value = serde_json::json!({"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
            "original_max_position_embeddings": 8192, "beta_fast": 64.0, "beta_slow": 1.0});
        let (f, s) = inv_freq(p.as_object().unwrap(), 128, 32768, true).unwrap();
        assert!((s - (0.1 * 4f32.ln() + 1.0)).abs() < 1e-6);
        let plain = |i: usize| 1.0 / 10000f64.powf(2.0 * i as f64 / 128.0);
        assert!((f[0] as f64 - plain(0)).abs() < 1e-7);
        assert!((f[20] as f64 - plain(20)).abs() / plain(20) < 1e-6);
        assert!((f[50] as f64 - plain(50) / 4.0).abs() / plain(50) < 1e-6);
        let mid = (f[33] as f64 - plain(33) / 4.0) / (plain(33) - plain(33) / 4.0);
        assert!(
            (mid - (1.0 - 13.0 / 30.0)).abs() < 1e-5,
            "ramp at 33: {mid}"
        );
    }
}
