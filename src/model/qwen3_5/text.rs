//! The dense Qwen3 decoder stack (plain attention, per-head q/k RMSNorm) of Qwen3, Qwen3-VL and
//! the Qwen2.5-VL text tower, run on the host as a text encoder (hidden states, no LM head).

use super::{linear, rms_norm, rms_norm_rows, vector_dot, Linear};
use crate::model::nn::add_in_place;
use crate::model::par::{par_for, par_rows, OutPtr};
use crate::model::simd::{vector_fused_scale_add, vector_softmax, vector_swiglu};
use crate::model::weights::Weights;
use serde_json::Value;

/// Which decoder [`TextModel::load`] reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Arch {
    Qwen3Vl,
    /// `qwen3`: q_norm/k_norm before RoPE.
    Qwen3,
    /// `qwen2_5_vl` text tower: qkv bias, no q_norm/k_norm, plain RoPE (text positions agree on
    /// every M-RoPE axis).
    Qwen2,
}

struct Attention {
    /// `q_proj | k_proj | v_proj`.
    qkv: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o: Linear,
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    attn: Attention,
    /// `gate_proj | up_proj`.
    gate_up: Linear,
    down: Linear,
}

pub(super) struct TextModel {
    hidden: usize,
    inter: usize,
    vocab: usize,
    eps: f32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Inverse frequency of each rotary pair (text positions: every M-RoPE axis agrees).
    inv_freq: Vec<f32>,
    embed: Linear,
    layers: Vec<Layer>,
    norm: Vec<f32>,
}

impl TextModel {
    /// Reads the decoder of config `c` (`text_config` for the multimodal archs).
    pub(super) fn load(ck: &Weights, c: &Value, arch: Arch) -> Result<Self, String> {
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("text_config.{k} missing"))
        };
        if c["num_experts"].as_u64().is_some() {
            return Err("MoE text encoders are not implemented".into());
        }
        let (hidden, inter, layers_n) = (
            get("hidden_size")?,
            get("intermediate_size")?,
            get("num_hidden_layers")?,
        );
        let heads = get("num_attention_heads")?;
        let kv_heads = get("num_key_value_heads")?;
        let head_dim = get("head_dim").unwrap_or(hidden / heads);
        let eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        // Qwen3-VL keeps `rope_theta` beside `rope_scaling`.
        let rope = if c["rope_parameters"].is_object() {
            &c["rope_parameters"]
        } else {
            &c["rope_scaling"]
        };
        let theta = rope["rope_theta"]
            .as_f64()
            .or(c["rope_theta"].as_f64())
            .ok_or("config has no rope_theta")?;
        let rot =
            (head_dim as f64 * rope["partial_rotary_factor"].as_f64().unwrap_or(1.0)) as usize;
        if arch != Arch::Qwen3Vl {
            match rope["rope_type"].as_str().or(rope["type"].as_str()) {
                None | Some("default") => {}
                // Raw Qwen2.5-VL configs say `mrope`; text-only positions make it plain RoPE.
                Some("mrope") if arch == Arch::Qwen2 => {}
                Some(k) => return Err(format!("rope type '{k}' is not implemented")),
            }
        }
        let inv_freq = (0..rot / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / rot as f64)) as f32)
            .collect();
        // Multimodal checkpoints nest the decoder under `language_model`.
        let p = if arch != Arch::Qwen3 && ck.has("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else {
            "model"
        };
        let qwen2 = arch == Arch::Qwen2;
        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let l = format!("{p}.layers.{i}");
            let a = format!("{l}.self_attn");
            // Qwen2 has no q/k norm (empty weights skip it in `norm_rope`).
            let qk_norm = |n: &str| match qwen2 {
                true => Ok(Vec::new()),
                false => ck.vec(&format!("{a}.{n}.weight")),
            };
            layers.push(Layer {
                in_norm: ck.vec(&format!("{l}.input_layernorm.weight"))?,
                post_norm: ck.vec(&format!("{l}.post_attention_layernorm.weight"))?,
                attn: Attention {
                    qkv: linear(
                        ck,
                        &[&format!("{a}.q_proj"), &format!("{a}.k_proj"), &format!("{a}.v_proj")],
                        c["attention_bias"] == true || qwen2,
                    )?,
                    q_norm: qk_norm("q_norm")?,
                    k_norm: qk_norm("k_norm")?,
                    o: linear(ck, &[&format!("{a}.o_proj")], c["attention_bias"] == true)?,
                },
                gate_up: linear(
                    ck,
                    &[&format!("{l}.mlp.gate_proj"), &format!("{l}.mlp.up_proj")],
                    false,
                )?,
                down: linear(ck, &[&format!("{l}.mlp.down_proj")], false)?,
            });
        }
        let embed = linear(ck, &[&format!("{p}.embed_tokens")], false)?;
        let m = Self {
            hidden,
            inter,
            vocab: embed.rows,
            eps,
            heads,
            kv_heads,
            head_dim,
            inv_freq,
            embed,
            layers,
            norm: ck.vec(&format!("{p}.norm.weight"))?,
        };
        m.check_shapes()?;
        Ok(m)
    }

    /// Catches config/checkpoint disagreements at load instead of as panics mid-forward.
    fn check_shapes(&self) -> Result<(), String> {
        let (d, hd) = (self.hidden, self.head_dim);
        for (i, l) in self.layers.iter().enumerate() {
            let a = &l.attn;
            if a.qkv.rows != (self.heads + 2 * self.kv_heads) * hd
                || a.qkv.cols != d
                || a.o.rows != d
                || a.o.cols != self.heads * hd
                || l.gate_up.rows != 2 * self.inter
                || l.down.cols != self.inter
            {
                return Err(format!(
                    "layer {i}: projection shapes disagree with config.json"
                ));
            }
        }
        if self.embed.cols != d {
            return Err("embed_tokens width != hidden_size".into());
        }
        Ok(())
    }

    fn embed(&self, token: u32, out: &mut [f32]) -> Result<(), String> {
        let t = token as usize;
        if t >= self.vocab {
            return Err(format!(
                "token id {token} is outside the vocabulary ({})",
                self.vocab
            ));
        }
        self.embed.parts[0].decode_row_f32(t, out);
        Ok(())
    }

    /// Hidden states `[n, hidden]` of `ids` (positions `0..n`) after each layer count in `taps`
    /// (ascending), before the final norm: transformers' `hidden_states[k]` under an attention
    /// mask of `valid` ones then zeros (every row, padding included, attends causally to keys
    /// `< valid`). One pass.
    pub(super) fn hidden_states_at(
        &self,
        ids: &[u32],
        valid: usize,
        taps: &[usize],
    ) -> Result<Vec<Vec<f32>>, String> {
        if ids.is_empty() || valid == 0 || valid > ids.len() || !taps.is_sorted() {
            return Err(format!(
                "bad encoder input: {} ids, {valid} valid",
                ids.len()
            ));
        }
        if let Some(&t) = taps.iter().find(|&&t| t > self.layers.len()) {
            return Err(format!("tap {t} past the {} layers", self.layers.len()));
        }
        crate::hardware::cpu_compute::enter("qwen3_5::text::hidden_states_at");
        let n = ids.len();
        let mut h = vec![0.0f32; n * self.hidden];
        for (&t, row) in ids.iter().zip(h.chunks_exact_mut(self.hidden)) {
            self.embed(t, row)?;
        }
        let rope = self.rope_table(n);
        let (mut done, mut out) = (0, Vec::with_capacity(taps.len()));
        for &t in taps {
            for layer in &self.layers[done..t] {
                self.layer_forward(layer, &mut h, n, &rope, valid);
            }
            done = t;
            out.push(h.clone());
        }
        Ok(out)
    }

    /// transformers' `hidden_states[-1]`: every layer's output after the final norm `[n, hidden]`.
    pub(super) fn normed_hidden_states(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut h = self
            .hidden_states_at(ids, ids.len(), &[self.layers.len()])?
            .remove(0);
        rms_norm_rows(&mut h, &self.norm, self.eps);
        Ok(h)
    }

    /// One decoder layer over rows `h` `[n, hidden]` in place.
    fn layer_forward(
        &self,
        layer: &Layer,
        h: &mut [f32],
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        key_cap: usize,
    ) {
        let mut x = h.to_vec();
        rms_norm_rows(&mut x, &layer.in_norm, self.eps);
        let y = self.attention(&layer.attn, &x, n, rope, key_cap);
        add_in_place(h, &y);
        let mut x = h.to_vec();
        rms_norm_rows(&mut x, &layer.post_norm, self.eps);
        let gu = layer.gate_up.forward(&x, n);
        let inter = self.inter;
        let mut act = vec![0.0f32; n * inter];
        par_rows(&mut act, inter, &|i, row| {
            vector_swiglu(&gu[i * 2 * inter..][..2 * inter], row, inter)
        });
        add_in_place(h, &layer.down.forward(&act, n));
    }

    /// `(cos, sin)` per position `0..n` and rotary pair.
    fn rope_table(&self, n: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.inv_freq.len();
        let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
        for p in 0..n {
            for &f in &self.inv_freq {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        (cos, sin)
    }

    /// Per-head q/k RMSNorm, then rotate-half RoPE on the first `2 * half` dims of one head.
    fn norm_rope(&self, x: &mut [f32], norm: &[f32], cos: &[f32], sin: &[f32]) {
        if !norm.is_empty() {
            rms_norm(x, norm, self.eps);
        }
        let half = cos.len();
        for i in 0..half {
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * cos[i] - b * sin[i];
            x[i + half] = b * cos[i] + a * sin[i];
        }
    }

    /// Causal attention of rows `x` `[n, hidden]` over keys `< key_cap`.
    fn attention(
        &self,
        a: &Attention,
        x: &[f32],
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        key_cap: usize,
    ) -> Vec<f32> {
        let (cos, sin) = rope;
        let (hd, heads, kvh) = (self.head_dim, self.heads, self.kv_heads);
        let half = self.inv_freq.len();
        let qkv = a.qkv.forward(x, n);
        let width = a.qkv.rows;
        let (qw, kvw) = (heads * hd, kvh * hd);
        let mut q = vec![0.0f32; n * qw];
        let (mut kc, mut vc) = (Vec::with_capacity(n * kvw), Vec::with_capacity(n * kvw));
        for i in 0..n {
            let row = &qkv[i * width..][..width];
            let (c, s) = (&cos[i * half..][..half], &sin[i * half..][..half]);
            for h in 0..heads {
                let qh = &mut q[i * qw + h * hd..][..hd];
                qh.copy_from_slice(&row[h * hd..][..hd]);
                self.norm_rope(qh, &a.q_norm, c, s);
            }
            let k0 = kc.len();
            kc.extend_from_slice(&row[qw..][..kvw]);
            for g in 0..kvh {
                self.norm_rope(&mut kc[k0 + g * hd..][..hd], &a.k_norm, c, s);
            }
            vc.extend_from_slice(&row[qw + kvw..][..kvw]);
        }
        let mut out = vec![0.0f32; n * qw];
        let o = OutPtr::new(&mut out);
        let scale = 1.0 / (hd as f32).sqrt();
        par_for(heads * n, &|t| {
            let (h, i) = (t / n, t % n);
            let g = h / (heads / kvh);
            let qh = &q[i * qw + h * hd..][..hd];
            let len = (i + 1).min(key_cap);
            let mut p: Vec<f32> = (0..len)
                .map(|m| vector_dot(qh, &kc[m * kvw + g * hd..][..hd]) * scale)
                .collect();
            vector_softmax(&mut p);
            // SAFETY: task (h, i) alone writes head h of row i.
            let dst = unsafe { o.slice(i * qw + h * hd, hd) };
            for (m, &pj) in p.iter().enumerate() {
                vector_fused_scale_add(dst, &vc[m * kvw + g * hd..][..hd], pj);
            }
        });
        a.o.forward(&out, n)
    }
}
