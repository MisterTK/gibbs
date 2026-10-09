//! The decoder: Qwen3.5's hybrid stack (Gated DeltaNet linear attention + output-gated full
//! attention, `1 + w` RMSNorms) or the dense Qwen3 stack (plain attention, `w` RMSNorms) of
//! Qwen3-VL (interleaved M-RoPE), Qwen3 and HunYuan dense v1 (1-D RoPE); with a sparse
//! mixture-of-experts feed-forward for Qwen3.5-MoE ([`moe`]) and Qwen4-Exp's hyper-connections,
//! QSA attention and PLE ([`qwen4`]).

mod moe;
mod mtp;
mod qwen4;

use super::{linear, rms_norm, rms_norm_rows, sigmoid, silu, vector_dot, Linear};
use crate::model::nn::add_in_place;
use crate::model::par::{par_for, par_rows, OutPtr};
use crate::model::simd::{vector_fused_scale_add, vector_softmax, vector_swiglu};
use crate::model::weights::Weights;
use moe::Moe;
pub(super) use mtp::MtpDrafter;
use qwen4::{Hc, IndexState, Indexer, LayerHc, Mix, Ple, PleState};
use serde_json::Value;

/// Which decoder [`TextModel::load`] reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Arch {
    Qwen35,
    Qwen3Vl,
    /// `qwen3`: q_norm/k_norm before RoPE.
    Qwen3,
    /// `qwen2_5_vl` text tower: qkv bias, no q_norm/k_norm, plain RoPE (text positions agree on
    /// every M-RoPE axis).
    Qwen2,
    /// `hunyuan_v1_dense`: query_layernorm/key_layernorm after RoPE, dynamic-NTK alpha RoPE.
    HunYuan,
    /// `qwen4_exp`: Qwen3.5-MoE + hyper-connections, QSA attention, PLE.
    Qwen4Exp,
}

/// Gated DeltaNet shape.
struct LinearCfg {
    k_heads: usize,
    v_heads: usize,
    k_dim: usize,
    v_dim: usize,
    kernel: usize,
    /// The gated norm's output gate is a sigmoid (Qwen4-Exp), not SiLU.
    sigmoid_gate: bool,
}

impl LinearCfg {
    fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
}

/// Gated DeltaNet token mixer.
pub(crate) struct DeltaNet {
    /// `in_proj_qkv | in_proj_z | in_proj_b | in_proj_a`.
    proj: Linear,
    /// Depthwise causal conv `[conv_dim, kernel]`.
    conv: Vec<f32>,
    /// `-exp(A_log)` per value head.
    neg_a: Vec<f32>,
    dt_bias: Vec<f32>,
    /// Gated RMSNorm weight `[v_dim]` (plain `w`, no offset).
    norm: Vec<f32>,
    out: Linear,
}

/// Full (softmax) attention token mixer.
pub(crate) struct Attention {
    /// `q_proj | k_proj | v_proj`; with an output gate q_proj yields `[q, gate]` per head.
    pub(crate) qkv: Linear,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) o: Linear,
    /// QSA indexer (Qwen4-Exp).
    indexer: Option<Indexer>,
}

pub(crate) enum Mixer {
    Delta(DeltaNet),
    Full(Attention),
}

/// Feed-forward block: dense SwiGLU MLP or sparse mixture of experts.
pub(crate) enum Ffn {
    Dense {
        /// `gate_proj | up_proj`.
        gate_up: Linear,
        down: Linear,
    },
    Moe(Moe),
}

pub(crate) struct Layer {
    pub(crate) in_norm: Vec<f32>,
    pub(crate) post_norm: Vec<f32>,
    pub(crate) mixer: Mixer,
    pub(crate) ffn: Ffn,
    /// Hyper-connection mixers and PLE (Qwen4-Exp), which replace the layer norms.
    hc: Option<LayerHc>,
}

/// Per-layer sequence state.
pub(super) enum LayerState {
    /// Last `kernel - 1` conv inputs `[kernel-1, conv_dim]` and the delta-rule state
    /// `[v_heads, k_dim, v_dim]`. A checkpointing forward of positions `from..` also keeps `s`
    /// after each of them (`ckpt`) and the conv inputs it ran over (`window`,
    /// `[kernel-1+n, conv_dim]`), so [`TextModel::truncate`] can rewind into that span.
    Delta {
        conv: Vec<f32>,
        s: Vec<f32>,
        ckpt: Vec<f32>,
        window: Vec<f32>,
        from: usize,
        /// PLE history, on the one PLE layer.
        ple: Option<PleState>,
    },
    /// Keys/values `[positions, kv_heads * head_dim]`, and the QSA indexer's keys.
    Full {
        k: Vec<f32>,
        v: Vec<f32>,
        idx: IndexState,
    },
}

pub(crate) struct TextModel {
    pub(crate) hidden: usize,
    pub(crate) inter: usize,
    pub(crate) vocab: usize,
    pub(crate) eps: f32,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) attn_gate: bool,
    /// HunYuan: the per-head q/k RMSNorm runs after RoPE (Qwen3: before).
    pub(crate) norm_after_rope: bool,
    /// Inverse frequency of each rotary pair (rotary dim / 2 of them) and the M-RoPE axis
    /// (0 = t, 1 = h, 2 = w) that drives it.
    pub(crate) inv_freq: Vec<f32>,
    pub(crate) rope_theta: f32,
    axis: Vec<usize>,
    lin: Option<LinearCfg>,
    pub(crate) max_pos: usize,
    pub(crate) decode_max_ctx: usize,
    /// Token embeddings `[vocab, hidden]`, also the LM head when tied.
    pub(crate) embed: Linear,
    pub(crate) lm_head: Option<Linear>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) norm: Vec<f32>,
    /// Hyper-connection head (Qwen4-Exp); `norm` is unused then.
    hc: Option<Hc>,
    /// Qwen3.5's multi-token-prediction head (`mtp.*`), when the checkpoint ships one.
    pub(super) mtp: Option<mtp::Mtp>,
}

impl TextModel {
    /// Reads the decoder of config `c` (`text_config` for the multimodal archs).
    pub(super) fn load(ck: &Weights, c: &Value, arch: Arch) -> Result<Self, String> {
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("text_config.{k} missing"))
        };
        let moe = c["num_experts"].as_u64().is_some();
        // MoE configs carry no dense `intermediate_size`.
        let (hidden, inter, layers_n) = (
            get("hidden_size")?,
            if moe { 0 } else { get("intermediate_size")? },
            get("num_hidden_layers")?,
        );
        if moe {
            if c["mlp_only_layers"]
                .as_array()
                .is_some_and(|a| !a.is_empty())
                || c["decoder_sparse_step"].as_u64().is_some_and(|s| s != 1)
            {
                return Err("MoE configs with dense layers (mlp_only_layers, decoder_sparse_step) are not implemented".into());
            }
        }
        let heads = get("num_attention_heads")?;
        let kv_heads = get("num_key_value_heads")?;
        let head_dim = get("head_dim").unwrap_or(hidden / heads);
        let eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        // Qwen3.5 nests theta in `rope_parameters`; Qwen3-VL keeps `rope_theta` beside `rope_scaling`.
        let rope = if c["rope_parameters"].is_object() {
            &c["rope_parameters"]
        } else {
            &c["rope_scaling"]
        };
        let mut theta = rope["rope_theta"]
            .as_f64()
            .or(c["rope_theta"].as_f64())
            .ok_or("config has no rope_theta")?;
        let rot =
            (head_dim as f64 * rope["partial_rotary_factor"].as_f64().unwrap_or(1.0)) as usize;
        let half = rot / 2;
        let axis = if matches!(arch, Arch::Qwen35 | Arch::Qwen3Vl | Arch::Qwen4Exp) {
            if rope["mrope_interleaved"] != true {
                return Err(
                    "only interleaved M-RoPE (`mrope_interleaved: true`) is implemented".into(),
                );
            }
            let section: Vec<usize> = rope["mrope_section"]
                .as_array()
                .ok_or("text_config has no mrope_section")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            if section.len() != 3 || section.iter().sum::<usize>() != half {
                return Err(format!(
                    "mrope_section {section:?} does not cover rotary dim {rot}"
                ));
            }
            // Interleaved M-RoPE: pair i takes h when i % 3 == 1 and i < 3 * section[1], w when
            // i % 3 == 2 and i < 3 * section[2], else t.
            (0..half)
                .map(|i| match i % 3 {
                    1 if i < 3 * section[1] => 1,
                    2 if i < 3 * section[2] => 2,
                    _ => 0,
                })
                .collect()
        } else {
            let kind = rope["rope_type"].as_str().or(rope["type"].as_str());
            match (kind, rope["alpha"].as_f64()) {
                (None | Some("default"), _) => {}
                // Raw Qwen2.5-VL configs say `mrope`; text-only positions make it plain RoPE.
                (Some("mrope"), _) if arch == Arch::Qwen2 => {}
                // HunYuanDenseV1RotaryEmbedding: base = theta * alpha^(d / (d - 2)).
                (Some("dynamic"), Some(alpha)) if arch == Arch::HunYuan => {
                    theta *= alpha.powf(head_dim as f64 / (head_dim as f64 - 2.0))
                }
                (Some(k), _) => return Err(format!("rope type '{k}' is not implemented")),
            }
            if arch == Arch::HunYuan {
                for (key, bad) in [
                    ("use_cla", true),
                    ("mlp_bias", true),
                    ("use_qk_norm", false),
                ] {
                    if c[key] == bad {
                        return Err(format!("HunYuan `{key}: {bad}` is not implemented"));
                    }
                }
            }
            vec![0; half]
        };
        let inv_freq = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / rot as f64)) as f32)
            .collect();
        // Qwen4-Exp's attention layers are `full_attention` or `qwen_sparse_attention`.
        let kinds: Vec<bool> = match c["layer_types"].as_array() {
            Some(t) if arch == Arch::Qwen4Exp => {
                t.iter().map(|k| k != "linear_attention").collect()
            }
            Some(t) => t.iter().map(|k| k == "full_attention").collect(),
            None => vec![true; layers_n],
        };
        if kinds.len() != layers_n {
            return Err("layer_types length != num_hidden_layers".into());
        }
        let lin = if kinds.contains(&false) {
            Some(LinearCfg {
                k_heads: get("linear_num_key_heads")?,
                v_heads: get("linear_num_value_heads")?,
                k_dim: get("linear_key_head_dim")?,
                v_dim: get("linear_value_head_dim")?,
                kernel: get("linear_conv_kernel_dim")?,
                sigmoid_gate: c["output_gate_type"] == "sigmoid",
            })
        } else {
            None
        };
        let attn_gate = c["attn_output_gate"] == true || arch == Arch::Qwen4Exp;
        let hc_count = (arch == Arch::Qwen4Exp)
            .then(|| get("hc_count"))
            .transpose()?;
        // mlx-vlm stores these `1 + w` norm weights already shifted (its qwen3_5 `sanitize`).
        let offset = if matches!(arch, Arch::Qwen35 | Arch::Qwen4Exp) && !ck.mlx_vlm() {
            1.0
        } else {
            0.0
        };
        let norm = |name: &str| -> Result<Vec<f32>, String> {
            Ok(ck.vec(name)?.into_iter().map(|w| w + offset).collect())
        };
        // Multimodal checkpoints nest the decoder under `language_model`; text-only exports
        // (`qwen3_5_text`) usually don't.
        let nested = ck.has("model.language_model.embed_tokens.weight");
        let (p, qn, kn) = match arch {
            Arch::Qwen35 | Arch::Qwen3Vl | Arch::Qwen4Exp | Arch::Qwen2 if nested => {
                ("model.language_model", "q_norm", "k_norm")
            }
            Arch::Qwen35 | Arch::Qwen3Vl | Arch::Qwen4Exp | Arch::Qwen3 | Arch::Qwen2 => {
                ("model", "q_norm", "k_norm")
            }
            Arch::HunYuan => ("model", "query_layernorm", "key_layernorm"),
        };
        let mut layers = Vec::with_capacity(layers_n);
        for (i, &full) in kinds.iter().enumerate() {
            let l = format!("{p}.layers.{i}");
            let mixer = if full {
                let a = format!("{l}.self_attn");
                Mixer::Full(Attention {
                    qkv: linear(
                        ck,
                        &[
                            &format!("{a}.q_proj"),
                            &format!("{a}.k_proj"),
                            &format!("{a}.v_proj"),
                        ],
                        c["attention_bias"] == true || arch == Arch::Qwen2,
                    )?,
                    // Qwen2 has no q/k norm (empty weights skip it in `norm_rope`).
                    q_norm: if arch == Arch::Qwen2 {
                        Vec::new()
                    } else {
                        norm(&format!("{a}.{qn}.weight"))?
                    },
                    k_norm: if arch == Arch::Qwen2 {
                        Vec::new()
                    } else {
                        norm(&format!("{a}.{kn}.weight"))?
                    },
                    o: linear(ck, &[&format!("{a}.o_proj")], c["attention_bias"] == true)?,
                    indexer: (arch == Arch::Qwen4Exp)
                        .then(|| Indexer::load(ck, &a, c, hidden))
                        .transpose()?,
                })
            } else {
                let a = format!("{l}.linear_attn");
                let names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                    .map(|n| format!("{a}.{n}"));
                Mixer::Delta(DeltaNet {
                    proj: linear(ck, &names.each_ref().map(String::as_str), false)?,
                    conv: ck.vec(&format!("{a}.conv1d.weight"))?,
                    // GGUF files store `-exp(A_log)` as is (random-weight ones may hold any sign).
                    neg_a: match ck.has(&format!("{a}.A_log")) {
                        true => ck
                            .vec(&format!("{a}.A_log"))?
                            .into_iter()
                            .map(|v| -v.exp())
                            .collect(),
                        false => ck.vec(&format!("{a}.neg_A"))?,
                    },
                    dt_bias: ck.vec(&format!("{a}.dt_bias"))?,
                    norm: ck.vec(&format!("{a}.norm.weight"))?,
                    out: linear(ck, &[&format!("{a}.out_proj")], false)?,
                })
            };
            let hc = match hc_count {
                Some(count) => {
                    let has_ple = c["ple_layer_ids"]
                        .as_array()
                        .is_some_and(|ids| ids.iter().any(|v| v.as_u64() == Some(i as u64 + 1)));
                    if has_ple && full {
                        return Err(format!(
                            "layer {i}: PLE on an attention layer is not implemented"
                        ));
                    }
                    let hash = c["ple_hash"].as_object().map(|h| {
                        let v = |k: &str| {
                            h[k].as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(Value::as_u64)
                                .collect()
                        };
                        (v("multipliers"), v("offsets"), v("vocab_sizes"))
                    });
                    Some(LayerHc {
                        attn: Mix::load(
                            ck,
                            &format!("{l}.attn_hyper_connection"),
                            count,
                            hidden,
                            true,
                        )?,
                        ffn: Mix::load(
                            ck,
                            &format!("{l}.mlp_hyper_connection"),
                            count,
                            hidden,
                            true,
                        )?,
                        ple: has_ple
                            .then(|| Ple::load(ck, &l, c, count, hidden, hash))
                            .transpose()?,
                    })
                }
                None => None,
            };
            let norms = |n: &str| {
                if hc.is_some() {
                    Ok(Vec::new())
                } else {
                    norm(&format!("{l}.{n}.weight"))
                }
            };
            layers.push(Layer {
                in_norm: norms("input_layernorm")?,
                post_norm: norms("post_attention_layernorm")?,
                hc,
                mixer,
                ffn: if moe {
                    Ffn::Moe(Moe::load(
                        ck,
                        &format!("{l}.mlp"),
                        hidden,
                        get("num_experts_per_tok")?,
                        c["norm_topk_prob"] != false,
                    )?)
                } else {
                    Ffn::Dense {
                        gate_up: linear(
                            ck,
                            &[&format!("{l}.mlp.gate_proj"), &format!("{l}.mlp.up_proj")],
                            false,
                        )?,
                        down: linear(ck, &[&format!("{l}.mlp.down_proj")], false)?,
                    }
                },
            });
        }
        let embed = linear(ck, &[&format!("{p}.embed_tokens")], false)?;
        // Tied embeddings win over a stored lm_head (HunYuan ships both; transformers ties them).
        let lm_head = if ck.has("lm_head.weight") && c["tie_word_embeddings"] != true {
            Some(linear(ck, &["lm_head"], false)?)
        } else {
            None
        };
        let m = Self {
            hidden,
            inter,
            vocab: embed.rows,
            eps,
            heads,
            kv_heads,
            head_dim,
            attn_gate,
            norm_after_rope: arch == Arch::HunYuan,
            inv_freq,
            rope_theta: theta as f32,
            axis,
            lin,
            max_pos: get("max_position_embeddings").unwrap_or(32768),
            decode_max_ctx: get("max_position_embeddings").unwrap_or(32768).min(4096),
            embed,
            lm_head,
            layers,
            hc: hc_count
                .map(|count| {
                    Ok::<_, String>(Hc {
                        count,
                        head: Mix::load(
                            ck,
                            &format!("{p}.hyper_connection_mixer"),
                            count,
                            hidden,
                            false,
                        )?,
                    })
                })
                .transpose()?,
            norm: if hc_count.is_some() {
                Vec::new()
            } else {
                norm(&format!("{p}.norm.weight"))?
            },
            // ponytail: the MoE checkpoints' MTP head (per-expert `mtp.layers.0.mlp.experts.N.*`)
            // is not read; they decode without a drafter.
            mtp: if arch == Arch::Qwen35 && !moe {
                mtp::Mtp::load(ck, c)?
            } else {
                None
            },
        };
        m.check_shapes()?;
        Ok(m)
    }

    /// Catches config/checkpoint disagreements at load instead of as panics mid-forward.
    fn check_shapes(&self) -> Result<(), String> {
        let (d, hd) = (self.hidden, self.head_dim);
        let q_width = self.heads * hd * if self.attn_gate { 2 } else { 1 };
        let mtp_layers = self.mtp.iter().flat_map(|m| &m.layers);
        for (i, l) in self.layers.iter().chain(mtp_layers).enumerate() {
            let (proj, want_rows, out, out_in) = match &l.mixer {
                Mixer::Full(a) => (
                    &a.qkv,
                    q_width + 2 * self.kv_heads * hd,
                    &a.o,
                    self.heads * hd,
                ),
                Mixer::Delta(n) => {
                    let c = self
                        .lin
                        .as_ref()
                        .expect("linear layers come with LinearCfg");
                    let vd = c.v_heads * c.v_dim;
                    if n.conv.len() != c.conv_dim() * c.kernel || n.norm.len() != c.v_dim {
                        return Err(format!("layer {i}: conv1d/norm shape mismatch"));
                    }
                    (&n.proj, c.conv_dim() + vd + 2 * c.v_heads, &n.out, vd)
                }
            };
            let dense_bad = match &l.ffn {
                Ffn::Dense { gate_up, down } => {
                    gate_up.rows != 2 * self.inter || down.cols != self.inter
                }
                Ffn::Moe(_) => false,
            };
            if proj.rows != want_rows
                || proj.cols != d
                || out.rows != d
                || out.cols != out_in
                || dense_bad
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

    pub(super) fn hidden(&self) -> usize {
        self.hidden
    }

    /// Width of a residual row: `hidden`, or `hc_count * hidden` with hyper-connections.
    pub(super) fn width(&self) -> usize {
        self.hidden * self.hc.as_ref().map_or(1, |h| h.count)
    }

    /// Hyper-connected models run the PLE hash on the tokens ([`Self::set_ple_tokens`]) and keep
    /// no per-position recurrent checkpoints, taps or rewinds.
    pub(super) fn is_hc(&self) -> bool {
        self.hc.is_some()
    }
    pub(super) fn context_len(&self) -> usize {
        self.max_pos
    }

    pub(super) fn summary(&self) -> String {
        let full = self
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::Full(_)))
            .count();
        let mix = if full == self.layers.len() {
            String::new()
        } else {
            format!(
                " ({} linear-attention + {full} full)",
                self.layers.len() - full
            )
        };
        let moe = match self.layers.first().map(|l| &l.ffn) {
            Some(Ffn::Moe(m)) => format!(", {}", m.summary()),
            _ => String::new(),
        };
        format!(
            "bf16 weights, {} layers{mix}{moe}, hidden {}, vocab {}",
            self.layers.len(),
            self.hidden,
            self.vocab
        )
    }

    pub(super) fn new_state(&self) -> Vec<LayerState> {
        self.layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Full(_) => LayerState::Full {
                    k: Vec::new(),
                    v: Vec::new(),
                    idx: IndexState::default(),
                },
                Mixer::Delta(_) => {
                    let c = self
                        .lin
                        .as_ref()
                        .expect("linear layers come with LinearCfg");
                    LayerState::Delta {
                        conv: vec![0.0; (c.kernel - 1) * c.conv_dim()],
                        s: vec![0.0; c.v_heads * c.k_dim * c.v_dim],
                        ckpt: Vec::new(),
                        window: Vec::new(),
                        from: 0,
                        ple: l
                            .hc
                            .as_ref()
                            .and_then(|h| h.ple.as_ref())
                            .map(|p| p.new_state(self.width())),
                    }
                }
            })
            .collect()
    }

    pub(super) fn embed(&self, token: u32, out: &mut [f32]) -> Result<(), String> {
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

    /// Number of decoder layers (tap ids run `0..layers()`).
    pub(super) fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The final-norm hidden state `[hidden]` of the last row of a [`Self::forward_rows`] pass
    /// over every layer (no LM head).
    pub(super) fn last_hidden(
        &self,
        h: Vec<f32>,
        pos: &[[u32; 3]],
        start: usize,
        state: &mut [LayerState],
        visual: &[(usize, usize)],
        deepstack: &[Vec<f32>],
    ) -> Vec<f32> {
        let (n, d) = (pos.len(), self.hidden);
        let h = self.forward_rows(
            h,
            pos,
            start,
            state,
            0,
            usize::MAX,
            visual,
            deepstack,
            false,
            &[],
            &mut Vec::new(),
        );
        let mut last = h[(n - 1) * d..].to_vec();
        rms_norm(&mut last, &self.norm, self.eps);
        last
    }

    /// Hidden states `[n, hidden]` of `ids` (positions `0..n`) after each layer count in `taps`
    /// (ascending), before the final norm: transformers' `hidden_states[k]` under an attention
    /// mask of `valid` ones then zeros (every row, padding included, attends causally to keys
    /// `< valid`). One pass; for text encoders.
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
        let mut h = vec![0.0f32; ids.len() * self.hidden];
        for (&t, row) in ids.iter().zip(h.chunks_exact_mut(self.hidden)) {
            self.embed(t, row)?;
        }
        let pos: Vec<[u32; 3]> = (0..ids.len() as u32).map(|p| [p; 3]).collect();
        let mut state = self.new_state();
        let (mut done, mut out) = (0, Vec::with_capacity(taps.len()));
        for &t in taps {
            let st = &mut state[done..t];
            h = self.forward_rows(
                h,
                &pos,
                0,
                st,
                done,
                valid,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            done = t;
            out.push(h.clone());
        }
        Ok(out)
    }

    /// Runs rows `h` `[n, hidden]` at rotary positions `pos` (sequence positions
    /// `start..start+n`) through decoder layers `first..first + state.len()` and returns the
    /// residual stream of every row (before the final norm; [`Self::logits`] finishes it).
    /// Attention sees keys `< key_cap` only. `visual` maps rows to image-feature rows for
    /// deepstack injection (`deepstack[k]` is added after layer k). `ckpt` keeps per-position
    /// recurrent checkpoints for [`Self::truncate`]; the output of each decoder layer in `taps` is
    /// appended to `tapped` as `[n, taps.len() * hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_rows(
        &self,
        mut h: Vec<f32>,
        pos: &[[u32; 3]],
        start: usize,
        state: &mut [LayerState],
        first: usize,
        key_cap: usize,
        visual: &[(usize, usize)],
        deepstack: &[Vec<f32>],
        ckpt: bool,
        taps: &[usize],
        tapped: &mut Vec<f32>,
    ) -> Vec<f32> {
        let (n, d) = (pos.len(), self.hidden);
        if let Some(hc) = &self.hc {
            assert_eq!(first, 0, "hyper-connected decoders run from the embeddings");
            h = self.expand_streams(&h, hc.count);
        }
        let rope = self.rope_table(pos);
        let base = tapped.len();
        tapped.resize(base + n * taps.len() * d, 0.0);
        let layers = self.layers[first..].iter().zip(state.iter_mut());
        for (li, (layer, st)) in (first..).zip(layers) {
            self.layer_forward(layer, st, &mut h, n, &rope, start, key_cap, ckpt);
            if let Some(ds) = deepstack.get(li) {
                for &(row, f) in visual {
                    add_in_place(&mut h[row * d..][..d], &ds[f * d..][..d]);
                }
            }
            for (t, _) in taps.iter().enumerate().filter(|&(_, &l)| l == li) {
                for i in 0..n {
                    tapped[base + (i * taps.len() + t) * d..][..d]
                        .copy_from_slice(&h[i * d..][..d]);
                }
            }
        }
        h
    }

    /// One decoder layer over rows `h` `[n, hidden]` in place.
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &Layer,
        st: &mut LayerState,
        h: &mut [f32],
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        start: usize,
        key_cap: usize,
        ckpt: bool,
    ) {
        if let Some(lhc) = &layer.hc {
            return self.layer_forward_hc(layer, lhc, st, h, n, rope, start, key_cap, ckpt);
        }
        let mut x = h.to_vec();
        rms_norm_rows(&mut x, &layer.in_norm, self.eps);
        let y = self.token_mixer(layer, st, &x, n, rope, start, key_cap, ckpt);
        add_in_place(h, &y);
        let mut x = h.to_vec();
        rms_norm_rows(&mut x, &layer.post_norm, self.eps);
        let y = self.ffn_forward(&layer.ffn, &x, n);
        add_in_place(h, &y);
    }

    /// The layer's attention or Gated DeltaNet over normed rows `x` `[n, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn token_mixer(
        &self,
        layer: &Layer,
        st: &mut LayerState,
        x: &[f32],
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        start: usize,
        key_cap: usize,
        ckpt: bool,
    ) -> Vec<f32> {
        match (&layer.mixer, st) {
            (Mixer::Full(a), LayerState::Full { k, v, idx }) => {
                self.attention(a, x, n, rope, start, key_cap, k, v, idx)
            }
            (
                Mixer::Delta(dn),
                LayerState::Delta {
                    conv,
                    s,
                    ckpt: ck,
                    window,
                    from,
                    ..
                },
            ) => {
                // Consecutive checkpointing forwards extend one span; anything else starts over.
                let held = ck.len() / s.len().max(1);
                if !ckpt || held == 0 || *from + held != start {
                    *from = start;
                    ck.clear();
                    window.clear();
                }
                let ck = ckpt.then_some((ck, window));
                self.delta_net(dn, x, n, conv, s, ck)
            }
            _ => unreachable!("state built by new_state matches the layer kinds"),
        }
    }

    /// The feed-forward block over normed rows `x` `[n, hidden]`.
    fn ffn_forward(&self, ffn: &Ffn, x: &[f32], n: usize) -> Vec<f32> {
        match ffn {
            Ffn::Dense { gate_up, down } => {
                let gu = gate_up.forward(x, n);
                let mut act = vec![0.0f32; n * self.inter];
                let inter = self.inter;
                par_rows(&mut act, inter, &|i, row| {
                    vector_swiglu(&gu[i * 2 * inter..][..2 * inter], row, inter)
                });
                down.forward(&act, n)
            }
            Ffn::Moe(m) => m.forward(x, n),
        }
    }

    /// Final norm (or hyper-connection head mixer) + LM head over residual rows `h`
    /// `[n, width]` -> `[n, vocab]`.
    pub(super) fn logits(&self, h: &[f32], n: usize) -> Vec<f32> {
        self.lm_head(&self.head_hidden(h, n), n)
    }

    /// The rows the LM head reads: `[n, hidden]` after the final norm / head mixer.
    pub(super) fn head_hidden(&self, h: &[f32], n: usize) -> Vec<f32> {
        if let Some(hc) = &self.hc {
            return hc.head.mix(h, n, self.eps).0;
        }
        let mut x = h.to_vec();
        rms_norm_rows(&mut x, &self.norm, self.eps);
        x
    }

    /// transformers' `hidden_states[-1]`: every layer's output after the final norm `[n, hidden]`.
    pub(super) fn normed_hidden_states(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut h = self
            .hidden_states_at(ids, ids.len(), &[self.layers.len()])?
            .remove(0);
        rms_norm_rows(&mut h, &self.norm, self.eps);
        Ok(h)
    }

    /// LM head over already-normed rows.
    fn lm_head(&self, x: &[f32], n: usize) -> Vec<f32> {
        self.lm_head.as_ref().unwrap_or(&self.embed).forward(x, n)
    }

    /// Drops positions `len..` from `state`. Attention layers rewind to any `len` they hold; a
    /// recurrent layer only into the span its checkpointing forwards ran since the last truncate
    /// (exactly: it restores the delta-rule state and conv inputs recorded after position
    /// `len - 1`). Checkpoints are dropped.
    pub(super) fn truncate(&self, state: &mut [LayerState], len: usize) -> Result<(), String> {
        if self.hc.is_some() {
            return Err(
                "hyper-connected decoders (PLE hash and QSA indexer state) cannot rewind".into(),
            );
        }
        let w = self.kv_heads * self.head_dim;
        for st in state.iter() {
            if let LayerState::Delta { s, ckpt, from, .. } = st {
                let n = if s.is_empty() {
                    0
                } else {
                    ckpt.len() / s.len()
                };
                if len <= *from || len > from + n {
                    return Err(format!(
                        "recurrent state rewinds only into its last verified span {}..={}, not to {len}",
                        from + 1,
                        from + n
                    ));
                }
            }
        }
        for st in state {
            match st {
                LayerState::Full { k, v, .. } => {
                    k.truncate(len * w);
                    v.truncate(len * w);
                }
                LayerState::Delta {
                    conv,
                    s,
                    ckpt,
                    window,
                    from,
                    ..
                } => {
                    let cd = self.lin.as_ref().map_or(0, LinearCfg::conv_dim);
                    let (keep, ss, cs) = (len - *from, s.len(), conv.len());
                    s.copy_from_slice(&ckpt[(keep - 1) * ss..][..ss]);
                    conv.copy_from_slice(&window[keep * cd..][..cs]);
                    ckpt.clear();
                    window.clear();
                }
            }
        }
        Ok(())
    }

    /// Forgets the recurrent checkpoints (the sequence keeps every position it ran).
    pub(super) fn drop_checkpoints(&self, state: &mut [LayerState]) {
        for st in state {
            if let LayerState::Delta { ckpt, window, .. } = st {
                ckpt.clear();
                window.clear();
            }
        }
    }

    /// `(cos, sin)` per row and rotary pair.
    fn rope_table(&self, pos: &[[u32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let half = self.inv_freq.len();
        let (mut cos, mut sin) = (
            Vec::with_capacity(pos.len() * half),
            Vec::with_capacity(pos.len() * half),
        );
        for p in pos {
            for (f, &ax) in self.inv_freq.iter().zip(&self.axis) {
                let a = p[ax] as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        (cos, sin)
    }

    /// Per-head q/k RMSNorm and rotate-half RoPE on the first `2 * half` dims of one head, in the
    /// architecture's order.
    fn norm_rope(&self, x: &mut [f32], norm: &[f32], cos: &[f32], sin: &[f32]) {
        let qk_norm = |x: &mut [f32]| {
            if !norm.is_empty() {
                rms_norm(x, norm, self.eps);
            }
        };
        if !self.norm_after_rope {
            qk_norm(x);
        }
        let half = cos.len();
        for i in 0..half {
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * cos[i] - b * sin[i];
            x[i + half] = b * cos[i] + a * sin[i];
        }
        if self.norm_after_rope {
            qk_norm(x);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        a: &Attention,
        x: &[f32],
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        start: usize,
        key_cap: usize,
        kc: &mut Vec<f32>,
        vc: &mut Vec<f32>,
        idx: &mut IndexState,
    ) -> Vec<f32> {
        let (cos, sin) = rope;
        let (hd, heads, kvh) = (self.head_dim, self.heads, self.kv_heads);
        let half = self.inv_freq.len();
        let qstride = if self.attn_gate { 2 * hd } else { hd };
        let qkv = a.qkv.forward(x, n);
        let width = a.qkv.rows;
        let (qw, kvw) = (heads * hd, kvh * hd);
        let mut q = vec![0.0f32; n * qw];
        let mut gate = vec![0.0f32; if self.attn_gate { n * qw } else { 0 }];
        for i in 0..n {
            let row = &qkv[i * width..][..width];
            let (c, s) = (&cos[i * half..][..half], &sin[i * half..][..half]);
            for h in 0..heads {
                let qh = &mut q[i * qw + h * hd..][..hd];
                qh.copy_from_slice(&row[h * qstride..][..hd]);
                self.norm_rope(qh, &a.q_norm, c, s);
                if self.attn_gate {
                    gate[i * qw + h * hd..][..hd].copy_from_slice(&row[h * qstride + hd..][..hd]);
                }
            }
            let k0 = kc.len();
            kc.extend_from_slice(&row[heads * qstride..][..kvw]);
            for g in 0..kvh {
                self.norm_rope(&mut kc[k0 + g * hd..][..hd], &a.k_norm, c, s);
            }
            vc.extend_from_slice(&row[heads * qstride + kvw..][..kvw]);
        }
        // QSA: per row, the keys the indexer lets it read (`None`: all visible keys).
        let sel = a.indexer.as_ref().map_or_else(Vec::new, |ix| {
            // ponytail: pooled keys rotate by their sequence index (text positions).
            let at = |p: usize| self.rope_table(&[[p as u32; 3]]);
            ix.select(x, n, start, rope, &at, idx, self.eps)
        });
        let mut out = vec![0.0f32; n * qw];
        let o = OutPtr::new(&mut out);
        let scale = 1.0 / (hd as f32).sqrt();
        let (kc, vc) = (&*kc, &*vc);
        par_for(heads * n, &|t| {
            let (h, i) = (t / n, t % n);
            let g = h / (heads / kvh);
            let qh = &q[i * qw + h * hd..][..hd];
            let rows = sel.get(i).and_then(|s| s.as_deref());
            let key = |m: usize| rows.map_or(m, |r| r[m] as usize);
            let len = rows.map_or((start + i + 1).min(key_cap), <[u32]>::len);
            let mut p: Vec<f32> = (0..len)
                .map(|m| vector_dot(qh, &kc[key(m) * kvw + g * hd..][..hd]) * scale)
                .collect();
            vector_softmax(&mut p);
            // SAFETY: task (h, i) alone writes head h of row i.
            let dst = unsafe { o.slice(i * qw + h * hd, hd) };
            for (m, &pj) in p.iter().enumerate() {
                vector_fused_scale_add(dst, &vc[key(m) * kvw + g * hd..][..hd], pj);
            }
            if self.attn_gate {
                for (v, &gt) in dst.iter_mut().zip(&gate[i * qw + h * hd..][..hd]) {
                    *v *= sigmoid(gt);
                }
            }
        });
        a.o.forward(&out, n)
    }

    /// `ck`: (per-position state checkpoints, conv window) to record, see [`LayerState::Delta`].
    fn delta_net(
        &self,
        dn: &DeltaNet,
        x: &[f32],
        n: usize,
        conv_state: &mut Vec<f32>,
        s: &mut [f32],
        ck: Option<(&mut Vec<f32>, &mut Vec<f32>)>,
    ) -> Vec<f32> {
        let c = self
            .lin
            .as_ref()
            .expect("linear layers come with LinearCfg");
        let (cd, kk, vd) = (c.conv_dim(), c.k_heads * c.k_dim, c.v_heads * c.v_dim);
        let width = dn.proj.rows;
        let p = dn.proj.forward(x, n);
        // Causal depthwise conv over [previous kernel-1 inputs ; this chunk], then SiLU.
        let hist = c.kernel - 1;
        let mut window = std::mem::take(conv_state);
        for i in 0..n {
            window.extend_from_slice(&p[i * width..][..cd]);
        }
        let mut mixed = vec![0.0f32; n * cd];
        par_rows(&mut mixed, cd, &|i, row| {
            for (ch, m) in row.iter_mut().enumerate() {
                let w = &dn.conv[ch * c.kernel..][..c.kernel];
                let acc: f32 = (0..c.kernel)
                    .map(|j| w[j] * window[(i + j) * cd + ch])
                    .sum();
                *m = silu(acc);
            }
        });
        *conv_state = window[n * cd..].to_vec();
        debug_assert_eq!(conv_state.len(), hist * cd);
        let ss = s.len();
        let mut ckpt = ck.map(|(ckpt, win)| {
            if win.is_empty() {
                *win = window;
            } else {
                win.extend_from_slice(&window[hist * cd..]);
            }
            ckpt.reserve(n * ss);
            (
                OutPtr::uninit(&mut ckpt.spare_capacity_mut()[..n * ss]),
                ckpt,
            )
        });
        let cp = ckpt.as_ref().map(|c| c.0);
        // Gated delta rule, one task per value head over the chunk's tokens in order.
        let rep = c.v_heads / c.k_heads;
        let (dk, dv) = (c.k_dim, c.v_dim);
        let mut core = vec![0.0f32; n * vd];
        let o = OutPtr::new(&mut core);
        let sp = OutPtr::new(s);
        par_for(c.v_heads, &|h| {
            // SAFETY: head h alone owns its state block and its columns of `core`.
            let st = unsafe { sp.slice(h * dk * dv, dk * dv) };
            let kh = h / rep;
            let (mut q, mut k, mut kv) = (vec![0.0f32; dk], vec![0.0f32; dk], vec![0.0f32; dv]);
            for i in 0..n {
                let m = &mixed[i * cd..][..cd];
                let pr = &p[i * width..][..width];
                q.copy_from_slice(&m[kh * dk..][..dk]);
                k.copy_from_slice(&m[kk + kh * dk..][..dk]);
                let v = &m[2 * kk + h * dv..][..dv];
                l2_normalize(&mut q, 1.0 / (dk as f32).sqrt());
                l2_normalize(&mut k, 1.0);
                let z = &pr[cd + h * dv..][..dv];
                let beta = sigmoid(pr[cd + vd + h]);
                let ga = pr[cd + vd + c.v_heads + h] + dn.dt_bias[h];
                let softplus = if ga > 20.0 { ga } else { ga.exp().ln_1p() };
                let decay = (dn.neg_a[h] * softplus).exp();
                // S <- decay S; delta = beta (v - S^T k); S <- S + k delta^T; o = S^T q.
                kv.fill(0.0);
                for (a, srow) in k.iter().zip(st.chunks_exact(dv)) {
                    vector_fused_scale_add(&mut kv, srow, *a);
                }
                for (dl, &vv) in kv.iter_mut().zip(v) {
                    *dl = (vv - decay * *dl) * beta;
                }
                // SAFETY: as above.
                let out = unsafe { o.slice(i * vd + h * dv, dv) };
                for ((srow, &kj), &qj) in st.chunks_exact_mut(dv).zip(&k).zip(&q) {
                    for (sv, &dl) in srow.iter_mut().zip(&kv) {
                        *sv = decay * *sv + kj * dl;
                    }
                    vector_fused_scale_add(out, srow, qj);
                }
                if let Some(cp) = &cp {
                    // SAFETY: head h alone writes its block of each position's checkpoint.
                    unsafe { cp.write(i * ss + h * dk * dv, st) };
                }
                rms_norm(out, &dn.norm, self.eps);
                for (ov, &zv) in out.iter_mut().zip(z) {
                    *ov *= if c.sigmoid_gate {
                        sigmoid(zv)
                    } else {
                        silu(zv)
                    };
                }
            }
        });
        if let Some((_, ckpt)) = ckpt.as_mut() {
            // SAFETY: the heads wrote every element of the `n` new positions (v_heads blocks of
            // k_dim * v_dim = `ss` each); `par_for` re-raises a task's panic before this point.
            unsafe { ckpt.set_len(ckpt.len() + n * ss) };
        }
        dn.out.forward(&core, n)
    }

    /// Emits static-shape decode graph (n = 1) with stateful KV and DeltaNet caching and zero-copy weight bindings.
    pub fn to_decode_graph(&self) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        self.to_decode_graph_with_max_ctx(self.decode_max_ctx)
    }

    /// Emits static-shape decode graph (n = 1) with stateful KV and DeltaNet caching sized to `max_ctx`.
    pub fn to_decode_graph_with_max_ctx(
        &self,
        max_ctx: usize,
    ) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        let max_ctx = max_ctx.min(self.max_pos).max(1);
        use crate::graph::{DType, Graph, Op, RopeVariant, Tensor, UnaryKind, ValueId};
        use crate::model::weights::Mat;

        if self.hc.is_some() {
            return Err("Hyper-connections not supported in decode graph".into());
        }

        fn mat_dtype(m: &Mat) -> DType {
            if let Some(p) = m.packed() {
                match p.format {
                    crate::hardware::QuantFormat::FP32 => DType::F32,
                    crate::hardware::QuantFormat::FP16 => DType::F16,
                    crate::hardware::QuantFormat::BF16 => DType::BF16,
                    f => DType::Quant(f),
                }
            } else {
                DType::BF16
            }
        }

        fn add_linear_part(
            g: &mut Graph,
            w: &mut crate::graph::Weights,
            name: &str,
            x: ValueId,
            part: &Mat,
            bias: Option<&[f32]>,
            n: usize,
        ) -> ValueId {
            let dtype = mat_dtype(part);
            let w_name = format!("{name}.weight");
            let w_node = g.weight(&w_name, vec![part.rows, part.cols], dtype);
            w.insert(&w_name, part.to_graph_tensor());
            let mut ins = vec![x, w_node];
            if let Some(b) = bias {
                let b_name = format!("{name}.bias");
                let b_node = g.weight(&b_name, vec![b.len()], DType::F32);
                w.insert(&b_name, Tensor::F32(b.to_vec()));
                ins.push(b_node);
            }
            g.add(name, Op::Linear, &ins, vec![n, part.rows], DType::F32)
        }

        fn add_linear_part_shaped(
            g: &mut Graph,
            w: &mut crate::graph::Weights,
            name: &str,
            x: ValueId,
            part: &Mat,
            bias: Option<&[f32]>,
            shape: Vec<usize>,
        ) -> ValueId {
            let dtype = mat_dtype(part);
            let w_name = format!("{name}.weight");
            let w_node = g.weight(&w_name, vec![part.rows, part.cols], dtype);
            w.insert(&w_name, part.to_graph_tensor());
            let mut ins = vec![x, w_node];
            if let Some(b) = bias {
                let b_name = format!("{name}.bias");
                let b_node = g.weight(&b_name, vec![b.len()], DType::F32);
                w.insert(&b_name, Tensor::F32(b.to_vec()));
                ins.push(b_node);
            }
            g.add(name, Op::Linear, &ins, shape, DType::F32)
        }

        let mut g = Graph::new();
        let mut w = crate::graph::Weights::new();

        let n = 1;
        let mut x = g.input("inputs_embeds", vec![n, self.hidden], DType::F32);
        let positions = g.input("positions", vec![n], DType::U32);

        // 2. Decoder layers
        for (li, l) in self.layers.iter().enumerate() {
            if l.hc.is_some() {
                return Err(format!(
                    "Layer {li}: Hyper-connections / PLE not supported in decode graph"
                ));
            }
            let p = format!("layers.{li}");

            // Input LayerNorm / RMSNorm
            let in_norm_w = g.weight(
                format!("{p}.input_layernorm.weight"),
                vec![self.hidden],
                DType::F32,
            );
            w.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::F32(l.in_norm.clone()),
            );
            let normed_x = g.add(
                format!("{p}.input_layernorm"),
                Op::RmsNorm { eps: self.eps },
                &[x, in_norm_w],
                vec![n, self.hidden],
                DType::F32,
            );

            // Mixer
            let mixer_out = match &l.mixer {
                Mixer::Delta(dn) => {
                    let c = self.lin.as_ref().expect("linear config for DeltaNet");
                    let (cd, kk, vd, vh, kh, kd, dv) = (
                        c.conv_dim(),
                        c.k_heads * c.k_dim,
                        c.v_heads * c.v_dim,
                        c.v_heads,
                        c.k_heads,
                        c.k_dim,
                        c.v_dim,
                    );

                    // Separate linear projections for conv_in (qkv), z, b, a
                    let conv_in = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mixer.proj_qkv"),
                        normed_x,
                        &dn.proj.parts[0],
                        None,
                        n,
                    );
                    let z = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mixer.proj_z"),
                        normed_x,
                        &dn.proj.parts[1],
                        None,
                        n,
                    );
                    let b = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mixer.proj_b"),
                        normed_x,
                        &dn.proj.parts[2],
                        None,
                        n,
                    );
                    let a = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mixer.proj_a"),
                        normed_x,
                        &dn.proj.parts[3],
                        None,
                        n,
                    );

                    // Conv1d
                    let conv_state = g.state(
                        format!("{p}.conv_state"),
                        vec![c.kernel - 1, cd],
                        DType::F32,
                    );
                    let conv_w = g.weight(
                        format!("{p}.mixer.conv.weight"),
                        vec![cd, c.kernel],
                        DType::F32,
                    );
                    w.insert(
                        format!("{p}.mixer.conv.weight"),
                        Tensor::F32(dn.conv.clone()),
                    );
                    let conv_out = g.add(
                        format!("{p}.mixer.conv"),
                        Op::CausalConv1dCached {
                            kernel: c.kernel,
                            dilation: 1,
                        },
                        &[conv_in, conv_w, conv_state],
                        vec![n, cd],
                        DType::F32,
                    );
                    let mixed = g.add(
                        format!("{p}.mixer.silu"),
                        Op::Silu,
                        &[conv_out],
                        vec![n, cd],
                        DType::F32,
                    );

                    // Slices for q, k, v
                    let q = g.add(
                        format!("{p}.mixer.q"),
                        Op::Slice {
                            axis: -1,
                            start: 0,
                            end: kk as i64,
                        },
                        &[mixed],
                        vec![n, kk],
                        DType::F32,
                    );
                    let k = g.add(
                        format!("{p}.mixer.k"),
                        Op::Slice {
                            axis: -1,
                            start: kk as i64,
                            end: (2 * kk) as i64,
                        },
                        &[mixed],
                        vec![n, kk],
                        DType::F32,
                    );
                    let v = g.add(
                        format!("{p}.mixer.v"),
                        Op::Slice {
                            axis: -1,
                            start: (2 * kk) as i64,
                            end: cd as i64,
                        },
                        &[mixed],
                        vec![n, vd],
                        DType::F32,
                    );

                    // Beta & log_decay
                    let beta = g.add(
                        format!("{p}.mixer.beta"),
                        Op::Unary {
                            kind: UnaryKind::Sigmoid,
                        },
                        &[b],
                        vec![n, vh],
                        DType::F32,
                    );
                    let dt_bias_w = g.weight(format!("{p}.mixer.dt_bias"), vec![vh], DType::F32);
                    w.insert(
                        format!("{p}.mixer.dt_bias"),
                        Tensor::F32(dn.dt_bias.clone()),
                    );
                    let a_biased = g.add(
                        format!("{p}.mixer.a_biased"),
                        Op::Add,
                        &[a, dt_bias_w],
                        vec![n, vh],
                        DType::F32,
                    );
                    let a_softplus = g.add(
                        format!("{p}.mixer.a_softplus"),
                        Op::Unary {
                            kind: UnaryKind::Softplus,
                        },
                        &[a_biased],
                        vec![n, vh],
                        DType::F32,
                    );
                    let neg_a_w = g.weight(format!("{p}.mixer.neg_a"), vec![vh], DType::F32);
                    w.insert(format!("{p}.mixer.neg_a"), Tensor::F32(dn.neg_a.clone()));
                    let log_decay = g.add(
                        format!("{p}.mixer.log_decay"),
                        Op::Mul,
                        &[neg_a_w, a_softplus],
                        vec![n, vh],
                        DType::F32,
                    );

                    // DeltaRule
                    let ssm_state =
                        g.state(format!("{p}.ssm_state"), vec![vh * kd * dv], DType::F32);
                    let core = g.add(
                        format!("{p}.mixer.delta"),
                        Op::DeltaRule {
                            key_heads: kh,
                            val_heads: vh,
                            key_dim: kd,
                            val_dim: dv,
                        },
                        &[q, k, v, beta, log_decay, ssm_state],
                        vec![n, vd],
                        DType::F32,
                    );

                    // Gated norm & Out
                    let norm_w = g.weight(format!("{p}.mixer.norm.weight"), vec![dv], DType::F32);
                    w.insert(
                        format!("{p}.mixer.norm.weight"),
                        Tensor::F32(dn.norm.clone()),
                    );
                    let normed_core = g.add(
                        format!("{p}.mixer.norm"),
                        Op::RmsNorm { eps: self.eps },
                        &[core, norm_w],
                        vec![n, vd],
                        DType::F32,
                    );
                    let z_act = if c.sigmoid_gate {
                        g.add(
                            format!("{p}.mixer.z_act"),
                            Op::Unary {
                                kind: UnaryKind::Sigmoid,
                            },
                            &[z],
                            vec![n, vd],
                            DType::F32,
                        )
                    } else {
                        g.add(
                            format!("{p}.mixer.z_act"),
                            Op::Silu,
                            &[z],
                            vec![n, vd],
                            DType::F32,
                        )
                    };
                    let gated = g.add(
                        format!("{p}.mixer.gated"),
                        Op::Mul,
                        &[normed_core, z_act],
                        vec![n, vd],
                        DType::F32,
                    );

                    add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mixer.out"),
                        gated,
                        &dn.out.parts[0],
                        dn.out.bias.as_deref(),
                        n,
                    )
                }
                Mixer::Full(attn) => {
                    if attn.indexer.is_some() {
                        return Err(format!(
                            "Layer {li}: QSA indexer not supported in decode graph"
                        ));
                    }
                    let (q_bias, k_bias, v_bias) = if let Some(b) = &attn.qkv.bias {
                        let qr = attn.qkv.parts[0].rows;
                        let kr = attn.qkv.parts[1].rows;
                        let vr = attn.qkv.parts[2].rows;
                        (
                            Some(&b[0..qr]),
                            Some(&b[qr..qr + kr]),
                            Some(&b[qr + kr..qr + kr + vr]),
                        )
                    } else {
                        (None, None, None)
                    };

                    let (q, gate) = if self.attn_gate {
                        let q_raw = add_linear_part_shaped(
                            &mut g,
                            &mut w,
                            &format!("{p}.self_attn.q"),
                            normed_x,
                            &attn.qkv.parts[0],
                            q_bias,
                            vec![n * self.heads, 2 * self.head_dim],
                        );
                        let q = g.add(
                            format!("{p}.self_attn.q_slice"),
                            Op::Slice {
                                axis: -1,
                                start: 0,
                                end: self.head_dim as i64,
                            },
                            &[q_raw],
                            vec![n * self.heads, self.head_dim],
                            DType::F32,
                        );
                        let gate = g.add(
                            format!("{p}.self_attn.gate_slice"),
                            Op::Slice {
                                axis: -1,
                                start: self.head_dim as i64,
                                end: (2 * self.head_dim) as i64,
                            },
                            &[q_raw],
                            vec![n * self.heads, self.head_dim],
                            DType::F32,
                        );
                        (q, Some(gate))
                    } else {
                        let q_raw = add_linear_part(
                            &mut g,
                            &mut w,
                            &format!("{p}.self_attn.q"),
                            normed_x,
                            &attn.qkv.parts[0],
                            q_bias,
                            n,
                        );
                        (q_raw, None)
                    };
                    let k_raw = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.self_attn.k"),
                        normed_x,
                        &attn.qkv.parts[1],
                        k_bias,
                        n,
                    );
                    let v = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.self_attn.v"),
                        normed_x,
                        &attn.qkv.parts[2],
                        v_bias,
                        n,
                    );

                    let (q_rope_in, k_rope_in) = if !self.norm_after_rope {
                        let q_n = if !attn.q_norm.is_empty() {
                            let q_norm_w = g.weight(
                                format!("{p}.self_attn.q_norm.weight"),
                                vec![self.head_dim],
                                DType::F32,
                            );
                            w.insert(
                                format!("{p}.self_attn.q_norm.weight"),
                                Tensor::F32(attn.q_norm.clone()),
                            );
                            g.add(
                                format!("{p}.self_attn.q_norm"),
                                Op::RmsNorm { eps: self.eps },
                                &[q, q_norm_w],
                                vec![n, self.heads * self.head_dim],
                                DType::F32,
                            )
                        } else {
                            q
                        };
                        let k_n = if !attn.k_norm.is_empty() {
                            let k_norm_w = g.weight(
                                format!("{p}.self_attn.k_norm.weight"),
                                vec![self.head_dim],
                                DType::F32,
                            );
                            w.insert(
                                format!("{p}.self_attn.k_norm.weight"),
                                Tensor::F32(attn.k_norm.clone()),
                            );
                            g.add(
                                format!("{p}.self_attn.k_norm"),
                                Op::RmsNorm { eps: self.eps },
                                &[k_raw, k_norm_w],
                                vec![n, self.kv_heads * self.head_dim],
                                DType::F32,
                            )
                        } else {
                            k_raw
                        };
                        (q_n, k_n)
                    } else {
                        (q, k_raw)
                    };

                    let mut q_rope = g.add(
                        format!("{p}.self_attn.q_rope"),
                        Op::Rope {
                            head_dim: self.head_dim,
                            rotary_dim: 2 * self.inv_freq.len(),
                            base: self.rope_theta,
                            variant: RopeVariant::HalfSplit,
                        },
                        &[q_rope_in, positions],
                        vec![n, self.heads * self.head_dim],
                        DType::F32,
                    );
                    let mut k_rope = g.add(
                        format!("{p}.self_attn.k_rope"),
                        Op::Rope {
                            head_dim: self.head_dim,
                            rotary_dim: 2 * self.inv_freq.len(),
                            base: self.rope_theta,
                            variant: RopeVariant::HalfSplit,
                        },
                        &[k_rope_in, positions],
                        vec![n, self.kv_heads * self.head_dim],
                        DType::F32,
                    );

                    if self.norm_after_rope {
                        if !attn.q_norm.is_empty() {
                            let q_norm_w = g.weight(
                                format!("{p}.self_attn.q_norm.weight"),
                                vec![self.head_dim],
                                DType::F32,
                            );
                            w.insert(
                                format!("{p}.self_attn.q_norm.weight"),
                                Tensor::F32(attn.q_norm.clone()),
                            );
                            q_rope = g.add(
                                format!("{p}.self_attn.q_norm_post"),
                                Op::RmsNorm { eps: self.eps },
                                &[q_rope, q_norm_w],
                                vec![n, self.heads * self.head_dim],
                                DType::F32,
                            );
                        }
                        if !attn.k_norm.is_empty() {
                            let k_norm_w = g.weight(
                                format!("{p}.self_attn.k_norm.weight"),
                                vec![self.head_dim],
                                DType::F32,
                            );
                            w.insert(
                                format!("{p}.self_attn.k_norm.weight"),
                                Tensor::F32(attn.k_norm.clone()),
                            );
                            k_rope = g.add(
                                format!("{p}.self_attn.k_norm_post"),
                                Op::RmsNorm { eps: self.eps },
                                &[k_rope, k_norm_w],
                                vec![n, self.kv_heads * self.head_dim],
                                DType::F32,
                            );
                        }
                    }

                    let k_cache = g.state(
                        format!("{p}.k_cache"),
                        vec![max_ctx, self.kv_heads * self.head_dim],
                        DType::F32,
                    );
                    let v_cache = g.state(
                        format!("{p}.v_cache"),
                        vec![max_ctx, self.kv_heads * self.head_dim],
                        DType::F32,
                    );
                    let mut attn_core = g.add(
                        format!("{p}.self_attn.core"),
                        Op::CachedAttention {
                            heads: self.heads,
                            kv_heads: self.kv_heads,
                            head_dim: self.head_dim,
                            v_head_dim: self.head_dim,
                            scale: 1.0 / (self.head_dim as f32).sqrt(),
                            causal: true,
                            window: None,
                            soft_cap: None,
                            write: true,
                        },
                        &[q_rope, k_rope, v, k_cache, v_cache, positions],
                        vec![n, self.heads * self.head_dim],
                        DType::F32,
                    );

                    if let Some(gate_val) = gate {
                        let gate_act = g.add(
                            format!("{p}.self_attn.gate_act"),
                            Op::Unary {
                                kind: UnaryKind::Sigmoid,
                            },
                            &[gate_val],
                            vec![n, self.heads * self.head_dim],
                            DType::F32,
                        );
                        attn_core = g.add(
                            format!("{p}.self_attn.gated_core"),
                            Op::Mul,
                            &[attn_core, gate_act],
                            vec![n, self.heads * self.head_dim],
                            DType::F32,
                        );
                    }

                    add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.self_attn.o"),
                        attn_core,
                        &attn.o.parts[0],
                        attn.o.bias.as_deref(),
                        n,
                    )
                }
            };

            x = g.add(
                format!("{p}.add_mixer"),
                Op::Add,
                &[x, mixer_out],
                vec![n, self.hidden],
                DType::F32,
            );

            // Post LayerNorm / RMSNorm
            let post_norm_w = g.weight(
                format!("{p}.post_attention_layernorm.weight"),
                vec![self.hidden],
                DType::F32,
            );
            w.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::F32(l.post_norm.clone()),
            );
            let normed_post = g.add(
                format!("{p}.post_attention_layernorm"),
                Op::RmsNorm { eps: self.eps },
                &[x, post_norm_w],
                vec![n, self.hidden],
                DType::F32,
            );

            // FFN
            let ffn_out = match &l.ffn {
                Ffn::Dense { gate_up, down } => {
                    let gate = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mlp.gate"),
                        normed_post,
                        &gate_up.parts[0],
                        gate_up.bias.as_deref(),
                        n,
                    );
                    let up = add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mlp.up"),
                        normed_post,
                        &gate_up.parts[1],
                        None,
                        n,
                    );
                    let swiglu = g.add(
                        format!("{p}.mlp.swiglu"),
                        Op::SwiGlu,
                        &[gate, up],
                        vec![n, gate_up.parts[0].rows],
                        DType::F32,
                    );
                    add_linear_part(
                        &mut g,
                        &mut w,
                        &format!("{p}.mlp.down"),
                        swiglu,
                        &down.parts[0],
                        down.bias.as_deref(),
                        n,
                    )
                }
                Ffn::Moe(moe) => {
                    let router_w = g.weight(
                        format!("{p}.mlp.router.weight"),
                        vec![moe.router.rows, moe.router.cols],
                        mat_dtype(&moe.router.parts[0]),
                    );
                    w.insert(
                        format!("{p}.mlp.router.weight"),
                        moe.router.parts[0].to_graph_tensor(),
                    );
                    g.add(
                        format!("{p}.mlp.moe"),
                        Op::Moe {
                            experts: moe.num_experts(),
                            top_k: moe.top_k(),
                        },
                        &[normed_post, router_w],
                        vec![n, self.hidden],
                        DType::F32,
                    )
                }
            };

            x = g.add(
                format!("{p}.add_ffn"),
                Op::Add,
                &[x, ffn_out],
                vec![n, self.hidden],
                DType::F32,
            );
        }

        // 3. Final Norm & LM Head
        let norm_w = g.weight("final_norm.weight", vec![self.hidden], DType::F32);
        w.insert("final_norm.weight", Tensor::F32(self.norm.clone()));
        let final_x = g.add(
            "final_norm",
            Op::RmsNorm { eps: self.eps },
            &[x, norm_w],
            vec![n, self.hidden],
            DType::F32,
        );

        let (lm_w, lm_dim) = if let Some(lm) = &self.lm_head {
            let w_node = g.weight(
                "lm_head.weight",
                vec![lm.rows, self.hidden],
                mat_dtype(&lm.parts[0]),
            );
            w.insert("lm_head.weight", lm.parts[0].to_graph_tensor());
            (w_node, lm.rows)
        } else {
            let w_node = g.weight(
                "lm_head.weight",
                vec![self.vocab, self.hidden],
                mat_dtype(&self.embed.parts[0]),
            );
            w.insert("lm_head.weight", self.embed.parts[0].to_graph_tensor());
            (w_node, self.vocab)
        };
        let logits = g.add(
            "logits",
            Op::Linear,
            &[final_x, lm_w],
            vec![n, lm_dim],
            DType::F32,
        );

        g.output(logits);
        Ok((g, w))
    }
}

/// `x * scale / sqrt(sum x^2 + 1e-6)` (the FLA-aligned l2norm the reference uses).
fn l2_normalize(x: &mut [f32], scale: f32) {
    let inv = scale / (vector_dot(x, x) + 1e-6).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_5_to_decode_graph() {
        let (hidden, vocab, layers, heads, kv_heads, hd) = (128, 128, 2, 4, 2, 32);
        let inter = 128;
        let c_kernel = 4;
        let (lin_kh, lin_vh, lin_kd, lin_vd) = (1, 2, 16, 16);

        let cfg = serde_json::json!({
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": hd,
            "intermediate_size": inter,
            "max_position_embeddings": 64,
            "attn_output_gate": true,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": lin_kh,
            "linear_num_value_heads": lin_vh,
            "linear_key_head_dim": lin_kd,
            "linear_value_head_dim": lin_vd,
            "linear_conv_kernel_dim": c_kernel,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 2]
            }
        });

        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
        specs.push(("model.embed_tokens.weight".into(), vec![vocab, hidden]));
        specs.push(("model.norm.weight".into(), vec![hidden]));
        specs.push(("lm_head.weight".into(), vec![vocab, hidden]));

        // Layer 0: DeltaNet
        let cd = 2 * lin_kh * lin_kd + lin_vh * lin_vd;
        specs.push(("model.layers.0.input_layernorm.weight".into(), vec![hidden]));
        specs.push((
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![hidden],
        ));
        specs.push((
            "model.layers.0.linear_attn.in_proj_qkv.weight".into(),
            vec![cd, hidden],
        ));
        specs.push((
            "model.layers.0.linear_attn.in_proj_z.weight".into(),
            vec![lin_vh * lin_vd, hidden],
        ));
        specs.push((
            "model.layers.0.linear_attn.in_proj_b.weight".into(),
            vec![lin_vh, hidden],
        ));
        specs.push((
            "model.layers.0.linear_attn.in_proj_a.weight".into(),
            vec![lin_vh, hidden],
        ));
        specs.push((
            "model.layers.0.linear_attn.conv1d.weight".into(),
            vec![cd, c_kernel],
        ));
        specs.push(("model.layers.0.linear_attn.neg_A".into(), vec![lin_vh]));
        specs.push(("model.layers.0.linear_attn.dt_bias".into(), vec![lin_vh]));
        specs.push((
            "model.layers.0.linear_attn.norm.weight".into(),
            vec![lin_vd],
        ));
        specs.push((
            "model.layers.0.linear_attn.out_proj.weight".into(),
            vec![hidden, lin_vh * lin_vd],
        ));
        specs.push((
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![inter, hidden],
        ));
        specs.push((
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![inter, hidden],
        ));
        specs.push((
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![hidden, inter],
        ));

        // Layer 1: Full Attention (with attn_output_gate)
        specs.push(("model.layers.1.input_layernorm.weight".into(), vec![hidden]));
        specs.push((
            "model.layers.1.post_attention_layernorm.weight".into(),
            vec![hidden],
        ));
        specs.push((
            "model.layers.1.self_attn.q_proj.weight".into(),
            vec![heads * 2 * hd, hidden],
        ));
        specs.push((
            "model.layers.1.self_attn.k_proj.weight".into(),
            vec![kv_heads * hd, hidden],
        ));
        specs.push((
            "model.layers.1.self_attn.v_proj.weight".into(),
            vec![kv_heads * hd, hidden],
        ));
        specs.push((
            "model.layers.1.self_attn.o_proj.weight".into(),
            vec![hidden, heads * hd],
        ));
        specs.push(("model.layers.1.self_attn.q_norm.weight".into(), vec![hd]));
        specs.push(("model.layers.1.self_attn.k_norm.weight".into(), vec![hd]));
        specs.push((
            "model.layers.1.mlp.gate_proj.weight".into(),
            vec![inter, hidden],
        ));
        specs.push((
            "model.layers.1.mlp.up_proj.weight".into(),
            vec![inter, hidden],
        ));
        specs.push((
            "model.layers.1.mlp.down_proj.weight".into(),
            vec![hidden, inter],
        ));

        let shapes: std::collections::HashMap<String, usize> = specs
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().product::<usize>()))
            .collect();

        let weights = crate::model::weights::Weights::from_f32(specs, |name| {
            let n: usize = name
                .bytes()
                .enumerate()
                .map(|(i, b)| (i + 1) * b as usize)
                .sum();
            let count = shapes[name];
            Ok((0..count)
                .map(|i| (((i * 19 + n * 13) % 41) as f32 - 20.0) / 25.0)
                .collect())
        })
        .unwrap();

        let text_model = TextModel::load(&weights, &cfg, Arch::Qwen35).unwrap();
        let (decode_g, w) = text_model.to_decode_graph().unwrap();

        // Validate SSA
        crate::graph::pattern::validate_ssa(&decode_g).expect("valid SSA graph");

        // Verify key operators exist
        assert!(decode_g
            .nodes
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::DeltaRule { .. })));
        assert!(decode_g
            .nodes
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::CausalConv1dCached { .. })));
        assert!(decode_g
            .nodes
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::CachedAttention { .. })));
        assert!(decode_g
            .nodes
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::SwiGlu)));
        assert!(decode_g
            .nodes
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::RmsNorm { .. })));

        // Run consecutive decode steps on CPU and compare against reference
        let mut cpu_st = text_model.new_state();
        let mut graph_st = crate::graph::StateStore::new();
        let tokens = [7u32, 23, 55, 89];
        for (pos, &tok) in tokens.iter().enumerate() {
            // Reference forward
            let mut h = vec![0.0f32; hidden];
            text_model.embed(tok, &mut h).unwrap();
            let pos_arr = [[pos as u32; 3]];
            let ref_hid = text_model.forward_rows(
                h,
                &pos_arr,
                pos,
                &mut cpu_st,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let ref_logits = text_model.logits(&ref_hid, 1);

            // Graph forward
            let mut emb = vec![0.0f32; hidden];
            text_model.embed(tok, &mut emb).unwrap();
            let inputs = vec![
                crate::graph::Tensor::F32(emb),
                crate::graph::Tensor::U32(vec![pos as u32]),
            ];
            let out = crate::graph::cpu::run(&decode_g, &w, &mut graph_st, inputs)
                .expect("decode graph execution succeeded");
            let graph_logits = out[0].as_f32().expect("f32 logits");

            assert_eq!(graph_logits.len(), ref_logits.len());
            let d = graph_logits
                .iter()
                .zip(&ref_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(d < 0.05, "step {pos}: max |Δlogit| = {d}");
        }

        #[cfg(target_os = "macos")]
        if metal::Device::system_default().is_some() {
            let (decode_g, w) = text_model.to_decode_graph().unwrap();
            let mut prog = crate::graph::program::GraphProgram::compile(
                decode_g,
                w,
                crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
            )
            .expect("compile Metal Pin GraphProgram");
            println!(
                "Metal Pin op counts: {:?}",
                prog.plan.report.device_op_counts
            );
            assert!(prog
                .plan
                .segments
                .iter()
                .any(|s| s.device == crate::hardware::Device::Metal));

            let mut cpu_st2 = text_model.new_state();
            for (pos, &tok) in tokens.iter().enumerate() {
                let mut h = vec![0.0f32; hidden];
                text_model.embed(tok, &mut h).unwrap();
                let pos_arr = [[pos as u32; 3]];
                let ref_hid = text_model.forward_rows(
                    h,
                    &pos_arr,
                    pos,
                    &mut cpu_st2,
                    0,
                    usize::MAX,
                    &[],
                    &[],
                    false,
                    &[],
                    &mut Vec::new(),
                );
                let ref_logits = text_model.logits(&ref_hid, 1);

                let mut emb = vec![0.0f32; hidden];
                text_model.embed(tok, &mut emb).unwrap();
                let inputs = vec![
                    crate::graph::Tensor::F32(emb),
                    crate::graph::Tensor::U32(vec![pos as u32]),
                ];
                let out = prog.run(inputs).expect("run Metal Auto GraphProgram");
                let prog_logits = out[0].as_f32().expect("f32 logits");

                assert_eq!(prog_logits.len(), ref_logits.len());
                let d = prog_logits
                    .iter()
                    .zip(&ref_logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(d < 0.05, "Metal Auto step {pos}: max |Δlogit| = {d}");
            }
        }
    }

    #[test]
    fn test_qwen3_to_decode_graph() {
        let hidden = 64;
        let vocab = 128;
        let layers = 2;
        let heads = 4;
        let hd = 16;
        let kv_heads = 2;
        let inter = 128;

        let cfg = serde_json::json!({
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": hd,
            "intermediate_size": inter,
            "max_position_embeddings": 64,
            "rope_scaling": {
                "rope_theta": 10000.0
            }
        });

        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
        specs.push(("model.embed_tokens.weight".into(), vec![vocab, hidden]));
        specs.push(("model.norm.weight".into(), vec![hidden]));
        specs.push(("lm_head.weight".into(), vec![vocab, hidden]));
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            specs.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
            specs.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
            specs.push((
                format!("{p}.self_attn.q_proj.weight"),
                vec![heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.o_proj.weight"),
                vec![hidden, heads * hd],
            ));
            specs.push((format!("{p}.self_attn.q_norm.weight"), vec![hd]));
            specs.push((format!("{p}.self_attn.k_norm.weight"), vec![hd]));
            specs.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]));
        }

        let shapes: std::collections::HashMap<String, usize> = specs
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().product::<usize>()))
            .collect();

        let weights = crate::model::weights::Weights::from_f32(specs, |name| {
            let n: usize = name
                .bytes()
                .enumerate()
                .map(|(i, b)| (i + 1) * b as usize)
                .sum();
            let count = shapes[name];
            Ok((0..count)
                .map(|i| (((i * 17 + n * 11) % 37) as f32 - 18.0) / 25.0)
                .collect())
        })
        .unwrap();

        let text_model = TextModel::load(&weights, &cfg, Arch::Qwen3).unwrap();
        let (decode_g, w) = text_model.to_decode_graph().unwrap();

        let mut cpu_st = text_model.new_state();
        let mut graph_st = crate::graph::StateStore::new();

        let tokens = vec![7u32, 23, 55];
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; hidden];
            text_model.embed(tok, &mut h).unwrap();
            let pos_arr = [[pos as u32; 3]];
            let cpu_hidden = text_model.forward_rows(
                h,
                &pos_arr,
                pos,
                &mut cpu_st,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let cpu_logits = text_model.logits(&cpu_hidden, 1);

            let mut emb = vec![0.0f32; hidden];
            text_model.embed(tok, &mut emb).unwrap();
            let inputs = vec![
                crate::graph::Tensor::F32(emb),
                crate::graph::Tensor::U32(vec![pos as u32]),
            ];
            let out = crate::graph::cpu::run(&decode_g, &w, &mut graph_st, inputs).unwrap();
            let graph_logits = out[0].as_f32().unwrap();

            assert_eq!(cpu_logits.len(), graph_logits.len());
            for (c, m) in cpu_logits.iter().zip(graph_logits) {
                assert!(
                    (c - m).abs() < 5e-2,
                    "Qwen3 CPU {c} vs Graph {m} diverged at pos {pos}"
                );
            }
        }
    }

    #[test]
    fn test_hunyuan_to_decode_graph() {
        let hidden = 64;
        let vocab = 128;
        let layers = 2;
        let heads = 4;
        let hd = 16;
        let kv_heads = 2;
        let inter = 128;

        let cfg = serde_json::json!({
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": hd,
            "intermediate_size": inter,
            "max_position_embeddings": 64,
            "model_type": "hunyuan_v1_dense",
            "rope_scaling": {
                "rope_theta": 10000.0
            }
        });

        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
        specs.push(("model.embed_tokens.weight".into(), vec![vocab, hidden]));
        specs.push(("model.norm.weight".into(), vec![hidden]));
        specs.push(("lm_head.weight".into(), vec![vocab, hidden]));
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            specs.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
            specs.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
            specs.push((
                format!("{p}.self_attn.q_proj.weight"),
                vec![heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_heads * hd, hidden],
            ));
            specs.push((
                format!("{p}.self_attn.o_proj.weight"),
                vec![hidden, heads * hd],
            ));
            specs.push((format!("{p}.self_attn.query_layernorm.weight"), vec![hd]));
            specs.push((format!("{p}.self_attn.key_layernorm.weight"), vec![hd]));
            specs.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]));
            specs.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]));
        }

        let shapes: std::collections::HashMap<String, usize> = specs
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().product::<usize>()))
            .collect();

        let weights = crate::model::weights::Weights::from_f32(specs, |name| {
            let n: usize = name
                .bytes()
                .enumerate()
                .map(|(i, b)| (i + 1) * b as usize)
                .sum();
            let count = shapes[name];
            Ok((0..count)
                .map(|i| (((i * 19 + n * 13) % 41) as f32 - 20.0) / 25.0)
                .collect())
        })
        .unwrap();

        let text_model = TextModel::load(&weights, &cfg, Arch::HunYuan).unwrap();
        let (decode_g, w) = text_model.to_decode_graph().unwrap();

        let mut cpu_st = text_model.new_state();
        let mut graph_st = crate::graph::StateStore::new();

        let tokens = vec![9u32, 31, 67];
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; hidden];
            text_model.embed(tok, &mut h).unwrap();
            let pos_arr = [[pos as u32; 3]];
            let cpu_hidden = text_model.forward_rows(
                h,
                &pos_arr,
                pos,
                &mut cpu_st,
                0,
                usize::MAX,
                &[],
                &[],
                false,
                &[],
                &mut Vec::new(),
            );
            let cpu_logits = text_model.logits(&cpu_hidden, 1);

            let mut emb = vec![0.0f32; hidden];
            text_model.embed(tok, &mut emb).unwrap();
            let inputs = vec![
                crate::graph::Tensor::F32(emb),
                crate::graph::Tensor::U32(vec![pos as u32]),
            ];
            let out = crate::graph::cpu::run(&decode_g, &w, &mut graph_st, inputs).unwrap();
            let graph_logits = out[0].as_f32().unwrap();

            assert_eq!(cpu_logits.len(), graph_logits.len());
            for (c, m) in cpu_logits.iter().zip(graph_logits) {
                assert!(
                    (c - m).abs() < 5e-2,
                    "HunYuan CPU {c} vs Graph {m} diverged at pos {pos}"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf smoke: cargo test --release --lib test_qwen35_metal_megakernel_smoke -- --ignored --test-threads=1"]
    fn test_qwen35_metal_megakernel_smoke() {
        use crate::hardware::Device;
        use crate::model::hub::HuggingFaceHub;
        use crate::model::native::load_native;

        let repo =
            "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF";
        let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
        if art.weights_path.is_none() {
            println!("SKIP: {repo} not found in disk/");
            return;
        }

        let lm = match load_native(&art, Device::Metal) {
            Some(Ok(m)) => m,
            Some(Err(e)) => panic!("Failed to load {repo}: {e}"),
            None => panic!("load_native returned None for {repo}"),
        };

        println!("Loaded Qwen3.5 9B model successfully!");
        println!("Backend label: {}", lm.backend_label());
        println!("Device: {:?}", lm.device());

        assert_eq!(lm.device(), Device::Metal, "Must select Metal device");
        assert!(
            lm.backend_label().contains("Metal"),
            "Backend label must be Metal"
        );
        let rss_bytes = crate::hardware::current_resident_memory_bytes();
        let rss_gb = rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "Resident memory (RSS): {:.2} GB ({rss_bytes} bytes)",
            rss_gb
        );
        assert!(
            rss_gb < 6.0,
            "Resident memory must drop below 6.0 GB for IQ2_M execution (was {rss_gb:.2} GB)"
        );

        // Run inference
        let tok = lm.tokenizer();
        let prompt = "The capital of France is";
        let ids = tok.encode(prompt);
        let (seq_ids, mut state) = lm.prepare(ids, &[]).expect("prepare sequence");

        let mut generated = Vec::new();
        let mut pos = 0;

        // Prefill
        let prefill_t0 = std::time::Instant::now();
        let logits = lm
            .forward(&seq_ids, pos, &mut state)
            .expect("forward prompt");
        let prefill_elapsed = prefill_t0.elapsed();
        println!("Prefill ({} tokens): {:?}", seq_ids.len(), prefill_elapsed);
        pos += seq_ids.len();

        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("Top 5 Metal prefill tokens:");
        for &(id, score) in &indexed[..5] {
            println!("  token {id} ('{}'): {score}", tok.decode(&[id as u32]));
        }
        let mut next_token = indexed[0].0 as u32;
        let decode_t0 = std::time::Instant::now();
        let max_new_tokens = 8;
        for step in 0..max_new_tokens {
            if tok.is_special(next_token) {
                break;
            }
            generated.push(next_token);
            let step_t0 = std::time::Instant::now();
            let next_logits = lm
                .forward(&[next_token], pos, &mut state)
                .expect("decode step");
            let step_elapsed = step_t0.elapsed();
            println!(
                "Decode step {step} (token {next_token}): {:?}",
                step_elapsed
            );
            pos += 1;
            let mut idx_step: Vec<(usize, f32)> = next_logits.iter().copied().enumerate().collect();
            idx_step.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "  step {step} top: token {} ('{}'): {}",
                idx_step[0].0,
                tok.decode(&[idx_step[0].0 as u32]),
                idx_step[0].1
            );
            next_token = idx_step[0].0 as u32;
        }
        let decode_elapsed = decode_t0.elapsed();

        let text = tok.decode(&generated);
        println!("Generated text: {:?}", text);
        let decode_tps = (generated.len() as f64) / decode_elapsed.as_secs_f64();
        println!(
            "Decode throughput: {:.2} tok/s (decode total: {:?})",
            decode_tps, decode_elapsed
        );
        let post_decode_rss = crate::hardware::current_resident_memory_bytes() as f64 / 1e9;
        println!(
            "Post-decode resident memory (RSS): {:.2} GB",
            post_decode_rss
        );
        assert!(
            post_decode_rss < 4.0,
            "Post-decode resident memory must remain below 4.0 GB (was {post_decode_rss:.2} GB)"
        );
        assert!(
            decode_tps > 25.0,
            "Decode throughput regression floor (was {decode_tps:.2} tok/s)"
        );
        assert!(!generated.is_empty(), "Must generate tokens");
        assert!(!text.is_empty(), "Generated text must not be empty");
    }
}
