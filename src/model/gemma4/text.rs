//! Gemma 4 text decoder (`gemma4_text`), as transformers `Gemma4TextModel` + the tied LM head with
//! final logit softcapping: per-layer input embeddings, sliding/full attention with per-type head
//! dims and RoPE (proportional partial rotary on full layers), KV sharing for the last
//! `num_kv_shared_layers`, double-wide MLPs on the shared layers, attention scale 1 (q/k RMSNorm).
//! Also the `gemma4_unified_text` backbone: no per-layer embeddings, `attention_k_eq_v` (full
//! layers use the key projection as values), and bidirectional attention inside image blocks on
//! sliding layers (`use_bidirectional_attention: "vision"`).

use super::moe::MoeBlock;
use super::ops::{gelu_mul, linear, rms_norm, rope, rope_table};
use crate::model::bf16_to_f32;
use crate::model::par::{par_for, OutPtr};
use crate::model::simd::{vector_dot_product, vector_fused_scale_add, vector_softmax};
use crate::model::weights::{Mat, Weights};
use serde_json::Value;

pub(crate) struct Attn {
    pub(crate) q: Mat,
    pub(crate) kv: Option<(Mat, Option<Mat>, Vec<f32>)>,
    pub(crate) o: Mat,
    pub(crate) q_norm: Vec<f32>,
}

pub(crate) struct Layer {
    pub(crate) sliding: bool,
    pub(crate) head_dim: usize,
    pub(crate) kv_heads: usize,
    pub(crate) kv_src: usize,
    pub(crate) attn: Attn,
    pub(crate) gate: Mat,
    pub(crate) up: Mat,
    pub(crate) down: Mat,
    pub(crate) input_ln: Vec<f32>,
    pub(crate) post_attn_ln: Vec<f32>,
    pub(crate) pre_ff_ln: Vec<f32>,
    pub(crate) post_ff_ln: Vec<f32>,
    pub(crate) ple: Option<(Mat, Mat, Vec<f32>)>,
    pub(crate) scalar: f32,
    /// The routed expert block of a sparse layer, beside the dense MLP above.
    pub(crate) moe: Option<MoeBlock>,
}

/// Per-layer embedding table rows (`[vocab, layers * dim]`): a mapped bf16 matrix, or rows decoded
/// on demand (a GGUF keeps the multi-GB table quantized and a request touches a handful of rows).
pub struct PleTable {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    row: Box<dyn Fn(usize, &mut [f32]) + Send + Sync>,
    /// The table as a device-decodable matrix (a GGUF table in BF16/FP16 or a codec format): the
    /// decode graph gathers its rows itself. `None`: rows come from the host only.
    pub(crate) mat: Option<Mat>,
}

impl PleTable {
    pub(crate) fn get_row(&self, id: usize, out: &mut [f32]) {
        (self.row)(id, out);
    }

    pub fn mat(m: Mat) -> Self {
        Self {
            rows: m.rows,
            cols: m.cols,
            row: Box::new(move |id, out| {
                for (o, &e) in out.iter_mut().zip(m.row(id)) {
                    *o = bf16_to_f32(e);
                }
            }),
            mat: None,
        }
    }

    pub fn lazy(
        rows: usize,
        cols: usize,
        row: impl Fn(usize, &mut [f32]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            rows,
            cols,
            row: Box::new(row),
            mat: None,
        }
    }
}

/// Per-layer embeddings (`hidden_size_per_layer_input > 0`): table, model projection, its norm.
pub(crate) struct Ple {
    pub(crate) dim: usize,
    pub(crate) embed: PleTable,
    pub(crate) model_proj: Mat,
    pub(crate) proj_norm: Vec<f32>,
}

pub struct TextModel {
    pub hidden: usize,
    pub vocab: usize,
    pub layers_n: usize,
    pub max_pos: usize,
    pub(crate) heads: usize,
    pub(crate) eps: f32,
    pub(crate) window: usize,
    pub(crate) softcap: Option<f32>,
    pub(crate) embed_scale: f32,
    pub(crate) embed: Mat,
    pub(crate) ple: Option<Ple>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) first_shared: usize,
    pub(crate) norm: Vec<f32>,
    pub(crate) inv_freq_sliding: Vec<f32>,
    pub(crate) inv_freq_full: Vec<f32>,
    pub(crate) pad_id: u32,
    pub bidir_vision: bool,
}

/// One layer's keys/values (post norm + RoPE) for positions `base..base + len`, rows of
/// `kv_heads * head_dim`.
#[derive(Default)]
pub struct Kv {
    base: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Per-sequence decoder state: one cache per KV-owning layer.
pub struct TextCache {
    kv: Vec<Kv>,
}

fn usize_of(c: &Value, key: &str) -> Result<usize, String> {
    c[key]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("text_config.{key} missing"))
}

/// RoPE inverse frequencies for one layer type (`rope_parameters[type]`): `default` rotates every
/// pair; `proportional` rotates the first `partial_rotary_factor` share, leaving the rest at
/// frequency 0 (identity) while still spanning `head_dim / 2` pairs.
fn inv_freq(p: &Value, head_dim: usize) -> Result<Vec<f32>, String> {
    let theta = p["rope_theta"]
        .as_f64()
        .ok_or("rope_parameters: rope_theta missing")?;
    let rotated = match p["rope_type"].as_str().unwrap_or("default") {
        "default" => head_dim / 2,
        "proportional" => {
            (p["partial_rotary_factor"].as_f64().unwrap_or(1.0) * head_dim as f64 / 2.0) as usize
        }
        other => return Err(format!("rope_type '{other}' is not supported")),
    };
    let factor = p["factor"].as_f64().unwrap_or(1.0);
    Ok((0..head_dim / 2)
        .map(|i| {
            if i < rotated {
                (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) / factor) as f32
            } else {
                0.0
            }
        })
        .collect())
}

impl TextModel {
    pub fn load(w: &Weights, c: &Value) -> Result<Self, String> {
        Self::load_with_ple(w, c, None)
    }

    /// [`Self::load`] with the per-layer embedding table supplied by the caller (`None`: the
    /// checkpoint's `embed_tokens_per_layer.weight`).
    pub fn load_with_ple(w: &Weights, c: &Value, ple: Option<PleTable>) -> Result<Self, String> {
        let k_eq_v = c["attention_k_eq_v"].as_bool().unwrap_or(false);
        let bidir_vision = match c["use_bidirectional_attention"].as_str() {
            None => false,
            Some("vision") => true,
            Some(other) => {
                return Err(format!(
                    "use_bidirectional_attention '{other}' is not supported"
                ))
            }
        };
        let hidden = usize_of(c, "hidden_size")?;
        let n = usize_of(c, "num_hidden_layers")?;
        let heads = usize_of(c, "num_attention_heads")?;
        let kv_heads = usize_of(c, "num_key_value_heads")?;
        let head_dim = usize_of(c, "head_dim")?;
        let global_head_dim = c["global_head_dim"]
            .as_u64()
            .map_or(head_dim, |v| v as usize);
        let global_kv_heads = c["num_global_key_value_heads"]
            .as_u64()
            .map_or(kv_heads, |v| v as usize);
        let ple_dim = c["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as usize;
        let shared = c["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize;
        let eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let types: Vec<bool> = c["layer_types"]
            .as_array()
            .ok_or("text_config.layer_types missing")?
            .iter()
            .map(|t| t.as_str() == Some("sliding_attention"))
            .collect();
        if types.len() != n {
            return Err("text_config: layer_types length disagrees with num_hidden_layers".into());
        }
        // transformers >= 5.17 writes per-layer overrides (`per_layer_config`:
        // `{"04": {"head_dim": 512}, ...}`) instead of global_head_dim / num_global_key_value_heads.
        let plc = c["per_layer_config"].as_object();
        let dims = |i: usize| -> (usize, usize) {
            let Some(m) = plc else {
                return if types[i] {
                    (head_dim, kv_heads)
                } else {
                    (global_head_dim, global_kv_heads)
                };
            };
            let o = m.iter().find(|(k, _)| k.parse::<usize>() == Ok(i));
            let get = |key: &str, d: usize| {
                o.and_then(|(_, o)| o[key].as_u64())
                    .map_or(d, |v| v as usize)
            };
            (
                get("head_dim", head_dim),
                get("num_key_value_heads", kv_heads),
            )
        };
        let first_shared = n - shared.min(n);
        let p = "model.language_model";
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let sliding = types[i];
            let (hd, kvh) = dims(i);
            let lp = format!("{p}.layers.{i}");
            let kv_src = if i >= first_shared {
                // The last non-shared layer of the same type.
                (0..first_shared)
                    .rev()
                    .find(|&j| types[j] == sliding)
                    .ok_or("no KV source layer")?
            } else {
                i
            };
            let kv = if kv_src == i {
                Some((
                    w.mat(&format!("{lp}.self_attn.k_proj.weight"))?,
                    if k_eq_v && !sliding {
                        None
                    } else {
                        Some(w.mat(&format!("{lp}.self_attn.v_proj.weight"))?)
                    },
                    w.vec_n(&format!("{lp}.self_attn.k_norm.weight"), hd)?,
                ))
            } else {
                None
            };
            let v = |s: &str| w.vec_n(&format!("{lp}.{s}.weight"), hidden);
            let layer = Layer {
                sliding,
                head_dim: hd,
                kv_heads: kvh,
                kv_src,
                attn: Attn {
                    q: w.mat(&format!("{lp}.self_attn.q_proj.weight"))?,
                    kv,
                    o: w.mat(&format!("{lp}.self_attn.o_proj.weight"))?,
                    q_norm: w.vec_n(&format!("{lp}.self_attn.q_norm.weight"), hd)?,
                },
                gate: w.mat(&format!("{lp}.mlp.gate_proj.weight"))?,
                up: w.mat(&format!("{lp}.mlp.up_proj.weight"))?,
                down: w.mat(&format!("{lp}.mlp.down_proj.weight"))?,
                input_ln: v("input_layernorm")?,
                post_attn_ln: v("post_attention_layernorm")?,
                pre_ff_ln: v("pre_feedforward_layernorm")?,
                post_ff_ln: v("post_feedforward_layernorm")?,
                ple: if ple_dim > 0 {
                    Some((
                        w.mat(&format!("{lp}.per_layer_input_gate.weight"))?,
                        w.mat(&format!("{lp}.per_layer_projection.weight"))?,
                        v("post_per_layer_input_norm")?,
                    ))
                } else {
                    None
                },
                scalar: w.scalar(&format!("{lp}.layer_scalar"))?,
                moe: MoeBlock::enabled(c)
                    .then(|| MoeBlock::load(w, &lp, c, hidden))
                    .transpose()
                    .map_err(|e| format!("layer {i}: {e}"))?,
            };
            if layer.attn.q.rows != heads * hd
                || layer.attn.o.cols != heads * hd
                || layer.gate.cols != hidden
            {
                return Err(format!("layer {i}: projection shapes disagree with config"));
            }
            if let Some((k, _, _)) = &layer.attn.kv {
                if k.rows != kvh * hd {
                    return Err(format!(
                        "layer {i}: k_proj is {}x{}, expected {}",
                        k.rows,
                        k.cols,
                        kvh * hd
                    ));
                }
            }
            layers.push(layer);
        }
        let embed = w.mat(&format!("{p}.embed_tokens.weight"))?;
        let ple = if ple_dim > 0 {
            let embed = match ple {
                Some(t) => t,
                None => PleTable::mat(w.mat(&format!("{p}.embed_tokens_per_layer.weight"))?),
            };
            if embed.cols != n * ple_dim {
                return Err("per-layer embedding shape disagrees with config".into());
            }
            Some(Ple {
                dim: ple_dim,
                embed,
                model_proj: w.mat(&format!("{p}.per_layer_model_projection.weight"))?,
                proj_norm: w.vec_n(&format!("{p}.per_layer_projection_norm.weight"), ple_dim)?,
            })
        } else {
            None
        };
        if embed.cols != hidden {
            return Err("embedding shape disagrees with config".into());
        }
        let rope = &c["rope_parameters"];
        // Gemma casts sqrt(hidden) to the bf16 weight dtype before scaling.
        let bf16 = |x: f32| bf16_to_f32(crate::model::f32_to_bf16(x));
        Ok(Self {
            hidden,
            vocab: embed.rows,
            layers_n: n,
            max_pos: usize_of(c, "max_position_embeddings")?,
            heads,
            eps,
            window: usize_of(c, "sliding_window")?,
            softcap: c["final_logit_softcapping"].as_f64().map(|v| v as f32),
            embed_scale: bf16((hidden as f32).sqrt()),
            embed,
            ple,
            layers,
            first_shared,
            norm: w.vec_n(&format!("{p}.norm.weight"), hidden)?,
            inv_freq_sliding: inv_freq(&rope["sliding_attention"], head_dim)?,
            inv_freq_full: inv_freq(
                &rope["full_attention"],
                types
                    .iter()
                    .position(|s| !s)
                    .map_or(global_head_dim, |i| dims(i).0),
            )?,
            pad_id: c["pad_token_id"].as_u64().unwrap_or(0) as u32,
            bidir_vision,
        })
    }

    /// Whether any layer has a routed expert block (decode graph only, no host forward).
    pub(crate) fn has_moe(&self) -> bool {
        self.layers.iter().any(|l| l.moe.is_some())
    }

    /// Whether the decode graph takes token ids and gathers the token and per-layer embeddings on
    /// the device (else the host feeds `inputs_embeds` and `per_layer_embed`).
    pub(crate) fn decode_takes_ids(&self) -> bool {
        self.embed.gathers_on_device()
            && self.ple.as_ref().is_none_or(|p| {
                p.embed.rows >= self.vocab
                    && p.embed.mat.as_ref().is_some_and(Mat::gathers_on_device)
            })
    }

    /// The inputs [`Self::to_decode_graph`] takes for token `tok` at position `pos`.
    pub fn decode_inputs(&self, tok: u32, pos: u32) -> Vec<crate::graph::Tensor> {
        use crate::graph::Tensor;
        if self.decode_takes_ids() {
            return vec![Tensor::U32(vec![tok]), Tensor::U32(vec![pos])];
        }
        let mut e = vec![0.0f32; self.hidden];
        self.embed_row(tok, &mut e);
        let mut v = vec![Tensor::F32(e), Tensor::U32(vec![pos])];
        if let Some(pl) = &self.ple {
            let mut r = vec![0.0f32; self.layers_n * pl.dim];
            let id = if (tok as usize) < pl.embed.rows {
                tok
            } else {
                self.pad_id
            };
            pl.embed.get_row(id as usize, &mut r);
            let scale = (pl.dim as f32).sqrt();
            r.iter_mut().for_each(|x| *x *= scale);
            v.push(Tensor::F32(r));
        }
        v
    }

    pub fn new_cache(&self) -> TextCache {
        TextCache {
            kv: (0..self.layers.len()).map(|_| Kv::default()).collect(),
        }
    }

    /// Scaled token embedding of `id`.
    pub fn embed_row(&self, id: u32, out: &mut [f32]) {
        self.embed.decode_row_f32(id as usize, out);
        if (self.embed_scale - 1.0).abs() > 1e-6 {
            for o in out.iter_mut() {
                *o *= self.embed_scale;
            }
        }
    }

    /// Runs rows `x` (`[n, hidden]` input embeddings) at positions `start..start + n`; `ple_ids` are
    /// the per-layer-embedding ids (the pad id for media soft tokens). `blocks` are the image
    /// soft-token position ranges attended bidirectionally on sliding layers (empty unless
    /// [`Self::bidir_vision`]); a block must not straddle the end of `x`. Returns the last row's
    /// logits.
    pub fn forward(
        &self,
        x: Vec<f32>,
        ple_ids: &[u32],
        start: usize,
        cache: &mut TextCache,
        blocks: &[std::ops::Range<usize>],
    ) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("gemma4::text::forward");
        let last = self.last_hidden(x, ple_ids, start, cache, blocks);
        let mut logits = linear(&last, 1, &self.embed);
        if let Some(cap) = self.softcap {
            logits.iter_mut().for_each(|v| *v = (*v / cap).tanh() * cap);
        }
        logits
    }

    pub(crate) fn per_layer_inputs(&self, x: &[f32], ple_ids: &[u32]) -> Vec<f32> {
        let (h, nl) = (self.hidden, self.layers_n);
        let pd = self.ple.as_ref().map_or(0, |p| p.dim);
        let n = ple_ids.len();
        if let Some(p) = &self.ple {
            let mut ple = linear(x, n, &p.model_proj);
            let proj_scale = (h as f32).powf(-0.5);
            ple.iter_mut().for_each(|v| *v *= proj_scale);
            rms_norm(&mut ple, pd, Some(&p.proj_norm), self.eps);
            let ident_scale = (pd as f32).sqrt();
            let half = std::f32::consts::FRAC_1_SQRT_2;
            let mut e = vec![0.0f32; nl * pd];
            for (r, &id) in ple_ids.iter().enumerate() {
                let row = &mut ple[r * nl * pd..(r + 1) * nl * pd];
                let id = if (id as usize) < p.embed.rows {
                    id
                } else {
                    self.pad_id
                };
                (p.embed.row)(id as usize, &mut e);
                for (v, &e) in row.iter_mut().zip(&e) {
                    *v = (*v + e * ident_scale) * half;
                }
            }
            ple
        } else {
            Vec::new()
        }
    }

    /// [`Self::forward`] up to the final norm: the last row's hidden state `[hidden]`.
    pub fn last_hidden(
        &self,
        mut x: Vec<f32>,
        ple_ids: &[u32],
        start: usize,
        cache: &mut TextCache,
        blocks: &[std::ops::Range<usize>],
    ) -> Vec<f32> {
        let (h, nl) = (self.hidden, self.layers_n);
        let pd = self.ple.as_ref().map_or(0, |p| p.dim);
        let mut n = ple_ids.len();
        // Per-layer inputs: (RMSNorm(proj(x) / sqrt(h)) + embed_per_layer(id) * sqrt(pd)) / sqrt(2).
        let mut ple = self.per_layer_inputs(&x, ple_ids);
        let tables = |inv: &[f32], n: usize, start: usize| -> Vec<(Vec<f32>, Vec<f32>)> {
            (0..n)
                .map(|r| rope_table((start + r) as f32, inv))
                .collect()
        };
        let mut rope_s = tables(&self.inv_freq_sliding, n, start);
        let mut rope_f = tables(&self.inv_freq_full, n, start);
        let mut row0 = 0; // first row still computed (KV-shared layers only need the last)
        for (li, l) in self.layers.iter().enumerate() {
            if li == self.first_shared && n > 1 {
                // Shared layers write no cache: only the last row reaches the logits.
                x.drain(..(n - 1) * h);
                ple.drain(..(n - 1) * nl * pd);
                rope_s.drain(..n - 1);
                rope_f.drain(..n - 1);
                row0 = n - 1;
                n = 1;
            }
            let ropes = if l.sliding { &rope_s } else { &rope_f };
            let mut a = x.clone();
            rms_norm(&mut a, h, Some(&l.input_ln), self.eps);
            let attn = self.attention(l, li, &a, n, start + row0, ropes, cache, blocks);
            let mut o = linear(&attn, n, &l.attn.o);
            rms_norm(&mut o, h, Some(&l.post_attn_ln), self.eps);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);

            let mut a = x.clone();
            rms_norm(&mut a, h, Some(&l.pre_ff_ln), self.eps);
            let mut g = linear(&a, n, &l.gate);
            gelu_mul(&mut g, &linear(&a, n, &l.up));
            let mut d = linear(&g, n, &l.down);
            rms_norm(&mut d, h, Some(&l.post_ff_ln), self.eps);
            x.iter_mut().zip(&d).for_each(|(x, d)| *x += d);

            if let Some((gate, proj, norm)) = &l.ple {
                let mut g = linear(&x, n, gate);
                let pli: Vec<f32> = (0..n)
                    .flat_map(|r| &ple[(r * nl + li) * pd..(r * nl + li + 1) * pd])
                    .copied()
                    .collect();
                gelu_mul(&mut g, &pli);
                let mut p = linear(&g, n, proj);
                rms_norm(&mut p, h, Some(norm), self.eps);
                x.iter_mut().zip(&p).for_each(|(x, p)| *x += p);
            }
            x.iter_mut().for_each(|x| *x *= l.scalar);
        }
        let mut last = x[(n - 1) * h..].to_vec();
        rms_norm(&mut last, h, Some(&self.norm), self.eps);
        last
    }

    /// Self-attention of `n` rows at positions `start..` (scale 1, causal, sliding window on
    /// sliding layers, where rows inside a `blocks` range also see the rest of their block);
    /// appends k/v to this layer's cache unless it is KV-shared.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        l: &Layer,
        li: usize,
        a: &[f32],
        n: usize,
        start: usize,
        ropes: &[(Vec<f32>, Vec<f32>)],
        cache: &mut TextCache,
        blocks: &[std::ops::Range<usize>],
    ) -> Vec<f32> {
        let (hd, nh, kvh) = (l.head_dim, self.heads, l.kv_heads);
        let mut q = linear(a, n, &l.attn.q);
        rms_norm(&mut q, hd, Some(&l.attn.q_norm), self.eps);
        for (r, (c, s)) in ropes.iter().enumerate() {
            q[r * nh * hd..(r + 1) * nh * hd]
                .chunks_exact_mut(hd)
                .for_each(|hq| rope(hq, c, s));
        }
        if let Some((kw, vw, k_norm)) = &l.attn.kv {
            let mut k = linear(a, n, kw);
            let mut v = match vw {
                Some(vw) => linear(a, n, vw),
                None => k.clone(),
            };
            rms_norm(&mut k, hd, Some(k_norm), self.eps);
            rms_norm(&mut v, hd, None, self.eps);
            for (r, (c, s)) in ropes.iter().enumerate() {
                k[r * kvh * hd..(r + 1) * kvh * hd]
                    .chunks_exact_mut(hd)
                    .for_each(|hk| rope(hk, c, s));
            }
            let kv = &mut cache.kv[li];
            assert_eq!(
                kv.base + kv.k.len() / (kvh * hd),
                start,
                "gemma4: KV cache out of sync with positions"
            );
            if l.sliding {
                // Keep the window the first new query needs; drop older rows in window-sized steps.
                let keep_from = start.saturating_sub(self.window - 1);
                if keep_from >= kv.base + self.window {
                    let drop = (keep_from - kv.base) * kvh * hd;
                    kv.k.drain(..drop);
                    kv.v.drain(..drop);
                    kv.base = keep_from;
                }
            }
            kv.k.extend_from_slice(&k);
            kv.v.extend_from_slice(&v);
        }
        let kv = &cache.kv[l.kv_src];
        let width = kvh * hd;
        let group = nh / kvh;
        let mut out = vec![0.0f32; n * nh * hd];
        let o = OutPtr::new(&mut out);
        let window = if l.sliding { self.window } else { usize::MAX };
        // Last key position each row sees: itself, or the end of its image block (sliding layers).
        let his: Vec<usize> = (start..start + n)
            .map(
                |pos| match blocks.iter().find(|b| l.sliding && b.contains(&pos)) {
                    Some(b) => b.end - 1,
                    None => pos,
                },
            )
            .collect();
        assert!(
            his.iter().all(|&hi| hi < kv.base + kv.k.len() / width),
            "gemma4: image block straddles the forward chunk"
        );
        par_for(n * nh, &|t| {
            let (r, head) = (t / nh, t % nh);
            let (pos, hi) = (start + r, his[r]);
            let lo = (pos + 1).saturating_sub(window).max(kv.base);
            let kh = head / group;
            let qv = &q[(r * nh + head) * hd..][..hd];
            let mut scores: Vec<f32> = (lo..=hi)
                .map(|p| vector_dot_product(qv, &kv.k[(p - kv.base) * width + kh * hd..][..hd]))
                .collect();
            vector_softmax(&mut scores);
            // SAFETY: (row, head) slot belongs to task t.
            let dst = unsafe { o.slice((r * nh + head) * hd, hd) };
            for (p, &s) in (lo..=hi).zip(&scores) {
                vector_fused_scale_add(dst, &kv.v[(p - kv.base) * width + kh * hd..][..hd], s);
            }
        });
        out
    }
    /// Emits static-shape decode graph (n = 1) with stateful KV caching, PLE, and weight bindings.
    /// Inputs `token_ids [1]`, `positions` when [`Self::decode_takes_ids`], else `inputs_embeds`,
    /// `positions`, `per_layer_embed` (with PLE).
    pub fn to_decode_graph(&self) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        self.to_decode_graph_unchecked()
    }

    pub(crate) fn to_decode_graph_unchecked(
        &self,
    ) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        self.decode_graph(1, false, self.max_pos, None, self.decode_takes_ids())
    }

    /// The same network over `rows` consecutive tokens (inputs `inputs_embeds [rows, h]`, `positions`,
    /// `per_layer_embed`, `visible_end [rows]`): output is the last row's logits `[1, vocab]`. Its weights
    /// are those of [`Self::to_decode_graph`], which is the program that owns them.
    pub(crate) fn to_prefill_graph(&self, rows: usize) -> Result<crate::graph::Graph, String> {
        self.decode_graph(rows, true, self.max_pos, None, false)
            .map(|(g, _)| g)
    }

    /// The decode graph with the gevva cross-encoder head (`head_norm` RMSNorm at `eps`, then the
    /// bias-free `score` `[labels, hidden]`) in place of the LM head: each step's output is the
    /// `[labels]` logits of the token it ran. KV state holds `max_ctx` positions.
    pub(crate) fn to_seqcls_graph(
        &self,
        max_ctx: usize,
        head_norm: &[f32],
        eps: f32,
        score: &[f32],
    ) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        self.decode_graph(1, false, max_ctx, Some((head_norm, eps, score)), false)
    }

    /// Builds the graph for `rows` token rows. `prefill` adds the `visible_end` input (sliding layers),
    /// slices the last row before the head and leaves the per-token decode shape (`rows == 1`) alone.
    /// `ids`: the graph takes token ids and gathers its embeddings ([`Self::decode_takes_ids`]).
    fn decode_graph(
        &self,
        rows: usize,
        prefill: bool,
        max_ctx: usize,
        head: Option<(&[f32], f32, &[f32])>,
        ids: bool,
    ) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        use crate::graph::{DType, Graph, Op, RopeVariant, Tensor};
        fn mat_dtype(m: &crate::model::weights::Mat) -> DType {
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

        let mut g = Graph::new();
        let mut w = crate::graph::Weights::new();

        let n = rows;
        let h = self.hidden;
        let token_ids = ids.then(|| g.input("token_ids", vec![n], DType::U32));
        let x_in = (!ids).then(|| g.input("inputs_embeds", vec![n, h], DType::F32));
        let positions = g.input("positions", vec![n], DType::U32);
        let pd = self.ple.as_ref().map_or(0, |p| p.dim);
        let nl = self.layers_n;
        // The tied LM head is also the token embedding table.
        let lm_w = ids.then(|| {
            let v = g.weight(
                "lm_head.weight",
                vec![self.vocab, h],
                mat_dtype(&self.embed),
            );
            w.insert("lm_head.weight", self.embed.to_graph_tensor());
            v
        });
        let mut x = match (x_in, token_ids, lm_w) {
            (Some(x), ..) => x,
            (None, Some(t), Some(table)) => {
                let e = g.add(
                    "model.embed",
                    Op::Embedding,
                    &[table, t],
                    vec![n, h],
                    DType::F32,
                );
                if (self.embed_scale - 1.0).abs() > 1e-6 {
                    let s = g.weight("model.embed_scale", vec![1], DType::F32);
                    w.insert("model.embed_scale", Tensor::F32(vec![self.embed_scale]));
                    g.add(
                        "model.embed_scaled",
                        Op::Mul,
                        &[e, s],
                        vec![n, h],
                        DType::F32,
                    )
                } else {
                    e
                }
            }
            _ => unreachable!("ids declare the token ids and the table"),
        };
        let ple_node = if let Some(p_model) = &self.ple {
            let ple_embed_input = match token_ids {
                None => g.input("per_layer_embed", vec![n, nl * pd], DType::F32),
                Some(t) => {
                    let mat = p_model.embed.mat.as_ref().ok_or("PLE table is host-only")?;
                    let tw = g.weight("model.ple.table", vec![mat.rows, mat.cols], mat_dtype(mat));
                    w.insert("model.ple.table", mat.to_graph_tensor());
                    let e = g.add(
                        "model.ple.embed",
                        Op::Embedding,
                        &[tw, t],
                        vec![n, nl * pd],
                        DType::F32,
                    );
                    let s = g.weight("model.ple.embed_scale", vec![1], DType::F32);
                    w.insert(
                        "model.ple.embed_scale",
                        Tensor::F32(vec![(pd as f32).sqrt()]),
                    );
                    g.add(
                        "model.ple.embed_scaled",
                        Op::Mul,
                        &[e, s],
                        vec![n, nl * pd],
                        DType::F32,
                    )
                }
            };
            let model_proj_w = g.weight(
                "model.ple.model_proj.weight",
                vec![p_model.model_proj.rows, p_model.model_proj.cols],
                mat_dtype(&p_model.model_proj),
            );
            w.insert(
                "model.ple.model_proj.weight",
                p_model.model_proj.to_graph_tensor(),
            );
            let ple_proj = g.add(
                "model.ple.model_proj",
                Op::Linear,
                &[x, model_proj_w],
                vec![n, nl * pd],
                DType::F32,
            );
            let proj_scale = (h as f32).powf(-0.5);
            let proj_scale_w = g.weight("model.ple.proj_scale", vec![1], DType::F32);
            w.insert("model.ple.proj_scale", Tensor::F32(vec![proj_scale]));
            let ple_scaled = g.add(
                "model.ple.proj_scaled",
                Op::Mul,
                &[ple_proj, proj_scale_w],
                vec![n, nl * pd],
                DType::F32,
            );
            let proj_norm_w = g.weight("model.ple.proj_norm.weight", vec![pd], DType::F32);
            w.insert(
                "model.ple.proj_norm.weight",
                Tensor::F32(p_model.proj_norm.clone()),
            );
            let ple_normed = g.add(
                "model.ple.proj_norm",
                Op::RmsNorm { eps: self.eps },
                &[ple_scaled, proj_norm_w],
                vec![n, nl * pd],
                DType::F32,
            );
            let ple_combined = g.add(
                "model.ple.add",
                Op::Add,
                &[ple_normed, ple_embed_input],
                vec![n, nl * pd],
                DType::F32,
            );
            let half = std::f32::consts::FRAC_1_SQRT_2;
            let half_w = g.weight("model.ple.half", vec![1], DType::F32);
            w.insert("model.ple.half", Tensor::F32(vec![half]));
            Some(g.add(
                "model.ple.scaled",
                Op::Mul,
                &[ple_combined, half_w],
                vec![n, nl * pd],
                DType::F32,
            ))
        } else {
            None
        };
        let visible_end = prefill.then(|| g.input("visible_end", vec![n], DType::U32));
        // 2. Decoder Layers
        for (li, l) in self.layers.iter().enumerate() {
            let p = format!("layers.{li}");

            // Input RMSNorm
            let in_ln_w = g.weight(format!("{p}.input_ln.weight"), vec![h], DType::F32);
            w.insert(
                format!("{p}.input_ln.weight"),
                Tensor::F32(l.input_ln.clone()),
            );
            let a = g.add(
                format!("{p}.input_ln"),
                Op::RmsNorm { eps: self.eps },
                &[x, in_ln_w],
                vec![n, h],
                DType::F32,
            );

            // Self-Attention Block
            let q_w = g.weight(
                format!("{p}.attn.q.weight"),
                vec![l.attn.q.rows, l.attn.q.cols],
                mat_dtype(&l.attn.q),
            );
            w.insert(format!("{p}.attn.q.weight"), l.attn.q.to_graph_tensor());
            let mut q = g.add(
                format!("{p}.attn.q"),
                Op::Linear,
                &[a, q_w],
                vec![n, l.attn.q.rows],
                DType::F32,
            );

            // Q per-head RMSNorm
            let q_norm_w = g.weight(
                format!("{p}.attn.q_norm.weight"),
                vec![l.head_dim],
                DType::F32,
            );
            w.insert(
                format!("{p}.attn.q_norm.weight"),
                Tensor::F32(l.attn.q_norm.clone()),
            );
            q = g.add(
                format!("{p}.attn.q_norm"),
                Op::RmsNorm { eps: self.eps },
                &[q, q_norm_w],
                vec![n, l.attn.q.rows],
                DType::F32,
            );

            // Q RoPE
            // Pairs are (i, i + head_dim/2) over the whole head; proportional (partial) rope is the
            // tail of `inv_freq` being zero, i.e. identity rotations, not a shorter rotary_dim.
            let (rotary_dim, inv_freq_slice, inv_freq_name) = if l.sliding {
                (l.head_dim, &self.inv_freq_sliding, "rope.inv_freq.sliding")
            } else {
                (l.head_dim, &self.inv_freq_full, "rope.inv_freq.full")
            };
            let inv_freq_val = g.weight(inv_freq_name, vec![rotary_dim / 2], DType::F32);
            w.insert(
                inv_freq_name.to_string(),
                Tensor::F32(inv_freq_slice[..rotary_dim / 2].to_vec()),
            );
            q = g.add(
                format!("{p}.attn.q_rope"),
                Op::Rope {
                    head_dim: l.head_dim,
                    rotary_dim,
                    variant: RopeVariant::HalfSplit,
                    mscale: 1.0,
                },
                &[q, positions, inv_freq_val],
                vec![n, l.attn.q.rows],
                DType::F32,
            );

            // K & V projections
            let (k_val, v_val) = if let Some((kw, vw, k_norm)) = &l.attn.kv {
                let k_w = g.weight(
                    format!("{p}.attn.k.weight"),
                    vec![kw.rows, kw.cols],
                    mat_dtype(kw),
                );
                w.insert(format!("{p}.attn.k.weight"), kw.to_graph_tensor());
                let mut k = g.add(
                    format!("{p}.attn.k"),
                    Op::Linear,
                    &[a, k_w],
                    vec![n, kw.rows],
                    DType::F32,
                );

                let k_raw = k;
                let k_norm_w = g.weight(
                    format!("{p}.attn.k_norm.weight"),
                    vec![l.head_dim],
                    DType::F32,
                );
                w.insert(
                    format!("{p}.attn.k_norm.weight"),
                    Tensor::F32(k_norm.clone()),
                );
                k = g.add(
                    format!("{p}.attn.k_norm"),
                    Op::RmsNorm { eps: self.eps },
                    &[k, k_norm_w],
                    vec![n, kw.rows],
                    DType::F32,
                );

                k = g.add(
                    format!("{p}.attn.k_rope"),
                    Op::Rope {
                        head_dim: l.head_dim,
                        rotary_dim,
                        variant: RopeVariant::HalfSplit,
                        mscale: 1.0,
                    },
                    &[k, positions, inv_freq_val],
                    vec![n, kw.rows],
                    DType::F32,
                );

                let v = if let Some(vw) = vw {
                    let v_w = g.weight(
                        format!("{p}.attn.v.weight"),
                        vec![vw.rows, vw.cols],
                        mat_dtype(vw),
                    );
                    w.insert(format!("{p}.attn.v.weight"), vw.to_graph_tensor());
                    g.add(
                        format!("{p}.attn.v"),
                        Op::Linear,
                        &[a, v_w],
                        vec![n, vw.rows],
                        DType::F32,
                    )
                } else {
                    // attention_k_eq_v: the value is the raw key projection, before k_norm and RoPE.
                    k_raw
                };
                // The value is RMS-normalised per head without a learned gain (all-ones gamma).
                let ones_name = format!("ones.{}", l.head_dim);
                let ones = g.weight(ones_name.clone(), vec![l.head_dim], DType::F32);
                w.insert(ones_name, Tensor::F32(vec![1.0; l.head_dim]));
                let v = g.add(
                    format!("{p}.attn.v_norm"),
                    Op::RmsNorm { eps: self.eps },
                    &[v, ones],
                    vec![n, kw.rows],
                    DType::F32,
                );
                (k, v)
            } else {
                (q, q)
            };

            // KV Cache States
            let kv_width = l.kv_heads * l.head_dim;
            let (k_state, v_state, write) = if li < self.first_shared {
                let ks = g.state(format!("kv.{li}.k"), vec![max_ctx, kv_width], DType::F32);
                let vs = g.state(format!("kv.{li}.v"), vec![max_ctx, kv_width], DType::F32);
                (ks, vs, true)
            } else {
                let ks = g.state(
                    format!("kv.{}.k", l.kv_src),
                    vec![max_ctx, kv_width],
                    DType::F32,
                );
                let vs = g.state(
                    format!("kv.{}.v", l.kv_src),
                    vec![max_ctx, kv_width],
                    DType::F32,
                );
                (ks, vs, false)
            };
            // Sliding layers also see the rest of their image block (`visible_end`) in a prefill pass.
            let mut attn_ins = vec![q, k_val, v_val, k_state, v_state, positions];
            if l.sliding {
                attn_ins.extend(visible_end);
            }
            let attn_out = g.add(
                format!("{p}.attn"),
                Op::CachedAttention {
                    heads: self.heads,
                    kv_heads: l.kv_heads,
                    head_dim: l.head_dim,
                    v_head_dim: l.head_dim,
                    scale: 1.0,
                    causal: true,
                    window: if l.sliding { Some(self.window) } else { None },
                    soft_cap: None,
                    write,
                },
                &attn_ins,
                vec![n, self.heads * l.head_dim],
                DType::F32,
            );

            // O projection
            let o_w = g.weight(
                format!("{p}.attn.o.weight"),
                vec![l.attn.o.rows, l.attn.o.cols],
                mat_dtype(&l.attn.o),
            );
            w.insert(format!("{p}.attn.o.weight"), l.attn.o.to_graph_tensor());
            let o = g.add(
                format!("{p}.attn.o"),
                Op::Linear,
                &[attn_out, o_w],
                vec![n, h],
                DType::F32,
            );

            // Post-attention RMSNorm
            let post_attn_w = g.weight(format!("{p}.post_attn_ln.weight"), vec![h], DType::F32);
            w.insert(
                format!("{p}.post_attn_ln.weight"),
                Tensor::F32(l.post_attn_ln.clone()),
            );
            let o_norm = g.add(
                format!("{p}.post_attn_ln"),
                Op::RmsNorm { eps: self.eps },
                &[o, post_attn_w],
                vec![n, h],
                DType::F32,
            );

            // Residual 1
            x = g.add(
                format!("{p}.residual_attn"),
                Op::Add,
                &[x, o_norm],
                vec![n, h],
                DType::F32,
            );

            // Feed-Forward Block
            let pre_ff_w = g.weight(format!("{p}.pre_ff_ln.weight"), vec![h], DType::F32);
            w.insert(
                format!("{p}.pre_ff_ln.weight"),
                Tensor::F32(l.pre_ff_ln.clone()),
            );
            let ff_in = g.add(
                format!("{p}.pre_ff_ln"),
                Op::RmsNorm { eps: self.eps },
                &[x, pre_ff_w],
                vec![n, h],
                DType::F32,
            );

            let gate_w = g.weight(
                format!("{p}.mlp.gate.weight"),
                vec![l.gate.rows, l.gate.cols],
                mat_dtype(&l.gate),
            );
            w.insert(format!("{p}.mlp.gate.weight"), l.gate.to_graph_tensor());
            let gate = g.add(
                format!("{p}.mlp.gate"),
                Op::Linear,
                &[ff_in, gate_w],
                vec![n, l.gate.rows],
                DType::F32,
            );

            let up_w = g.weight(
                format!("{p}.mlp.up.weight"),
                vec![l.up.rows, l.up.cols],
                mat_dtype(&l.up),
            );
            w.insert(format!("{p}.mlp.up.weight"), l.up.to_graph_tensor());
            let up = g.add(
                format!("{p}.mlp.up"),
                Op::Linear,
                &[ff_in, up_w],
                vec![n, l.up.rows],
                DType::F32,
            );

            let act = g.add(
                format!("{p}.mlp.act"),
                Op::GeGlu,
                &[gate, up],
                vec![n, l.gate.rows],
                DType::F32,
            );

            let down_w = g.weight(
                format!("{p}.mlp.down.weight"),
                vec![l.down.rows, l.down.cols],
                mat_dtype(&l.down),
            );
            w.insert(format!("{p}.mlp.down.weight"), l.down.to_graph_tensor());
            let down = g.add(
                format!("{p}.mlp.down"),
                Op::Linear,
                &[act, down_w],
                vec![n, h],
                DType::F32,
            );

            let down = match &l.moe {
                Some(m) => {
                    m.add_to_graph(&mut g, &mut w, &format!("{p}.moe"), x, down, self.eps, n)
                }
                None => down,
            };
            let post_ff_w = g.weight(format!("{p}.post_ff_ln.weight"), vec![h], DType::F32);
            w.insert(
                format!("{p}.post_ff_ln.weight"),
                Tensor::F32(l.post_ff_ln.clone()),
            );
            let down_norm = g.add(
                format!("{p}.post_ff_ln"),
                Op::RmsNorm { eps: self.eps },
                &[down, post_ff_w],
                vec![n, h],
                DType::F32,
            );

            // Residual 2
            x = g.add(
                format!("{p}.residual_ff"),
                Op::Add,
                &[x, down_norm],
                vec![n, h],
                DType::F32,
            );
            // PLE Layer Projection if present
            if let (Some((gate, proj, norm)), Some(ple_val)) = (&l.ple, ple_node) {
                let pli = g.add(
                    format!("{p}.ple.slice"),
                    Op::Slice {
                        axis: 1,
                        start: (li * pd) as i64,
                        end: ((li + 1) * pd) as i64,
                    },
                    &[ple_val],
                    vec![n, pd],
                    DType::F32,
                );
                let ple_gate_w = g.weight(
                    format!("{p}.ple.gate.weight"),
                    vec![gate.rows, gate.cols],
                    mat_dtype(gate),
                );
                w.insert(format!("{p}.ple.gate.weight"), gate.to_graph_tensor());
                let ple_gate = g.add(
                    format!("{p}.ple.gate"),
                    Op::Linear,
                    &[x, ple_gate_w],
                    vec![n, pd],
                    DType::F32,
                );
                let ple_act = g.add(
                    format!("{p}.ple.act"),
                    Op::GeGlu,
                    &[ple_gate, pli],
                    vec![n, pd],
                    DType::F32,
                );
                let ple_proj_w = g.weight(
                    format!("{p}.ple.proj.weight"),
                    vec![proj.rows, proj.cols],
                    mat_dtype(proj),
                );
                w.insert(format!("{p}.ple.proj.weight"), proj.to_graph_tensor());
                let ple_out = g.add(
                    format!("{p}.ple.proj"),
                    Op::Linear,
                    &[ple_act, ple_proj_w],
                    vec![n, h],
                    DType::F32,
                );
                let ple_norm_w = g.weight(format!("{p}.ple.norm.weight"), vec![h], DType::F32);
                w.insert(format!("{p}.ple.norm.weight"), Tensor::F32(norm.clone()));
                let ple_normed = g.add(
                    format!("{p}.ple.norm"),
                    Op::RmsNorm { eps: self.eps },
                    &[ple_out, ple_norm_w],
                    vec![n, h],
                    DType::F32,
                );
                x = g.add(
                    format!("{p}.ple.res"),
                    Op::Add,
                    &[x, ple_normed],
                    vec![n, h],
                    DType::F32,
                );
            }

            // Scalar scaling if present
            if (l.scalar - 1.0).abs() > 1e-6 {
                let scalar_w = g.weight(format!("{p}.scalar"), vec![1], DType::F32);
                w.insert(format!("{p}.scalar"), Tensor::F32(vec![l.scalar]));
                x = g.add(
                    format!("{p}.scaled"),
                    Op::Mul,
                    &[x, scalar_w],
                    vec![n, h],
                    DType::F32,
                );
            }
        }

        // 3. Final RMSNorm. A prefill pass only needs the last row's logits: slice before the head.
        let n = if prefill && n > 1 {
            x = g.add(
                "last_row",
                Op::Slice {
                    axis: 0,
                    start: n as i64 - 1,
                    end: n as i64,
                },
                &[x],
                vec![1, h],
                DType::F32,
            );
            1
        } else {
            n
        };
        let final_norm_w = g.weight("final_norm.weight", vec![h], DType::F32);
        w.insert("final_norm.weight", Tensor::F32(self.norm.clone()));
        let final_norm = g.add(
            "final_norm",
            Op::RmsNorm { eps: self.eps },
            &[x, final_norm_w],
            vec![n, h],
            DType::F32,
        );

        let logits = if let Some((head_norm, eps, score)) = head {
            // Sequence classification (gevva): the head's own RMSNorm, then the bias-free
            // `[labels, hidden]` score instead of the LM head.
            let norm_w = g.weight("seqcls.norm.weight", vec![h], DType::F32);
            w.insert("seqcls.norm.weight", Tensor::F32(head_norm.to_vec()));
            let normed = g.add(
                "seqcls.norm",
                Op::RmsNorm { eps },
                &[final_norm, norm_w],
                vec![n, h],
                DType::F32,
            );
            let labels = score.len() / h;
            let score_w = g.weight("score.weight", vec![labels, h], DType::F32);
            w.insert("score.weight", Tensor::F32(score.to_vec()));
            g.add(
                "score",
                Op::Linear,
                &[normed, score_w],
                vec![n, labels],
                DType::F32,
            )
        } else {
            // 4. LM Head (quantized native representation, avoiding dense F32 dequantization)
            let lm_w = lm_w.unwrap_or_else(|| {
                let v = g.weight(
                    "lm_head.weight",
                    vec![self.vocab, h],
                    mat_dtype(&self.embed),
                );
                w.insert("lm_head.weight", self.embed.to_graph_tensor());
                v
            });
            let logits = g.add(
                "lm_head",
                Op::Linear,
                &[final_norm, lm_w],
                vec![n, self.vocab],
                DType::F32,
            );
            if let Some(cap) = self.softcap {
                g.add(
                    "logits_softcap",
                    Op::Unary {
                        kind: crate::graph::UnaryKind::SoftCap(cap),
                    },
                    &[logits],
                    vec![n, self.vocab],
                    DType::F32,
                )
            } else {
                logits
            }
        };
        g.output(logits);

        Ok((g, w))
    }
}

/// Tiny synthetic Gemma 4 text model (one full-attention layer, constant weights). Fixture for
/// `tests/cpu_free_matrix.rs` and this module's tests.
pub fn tiny_model() -> TextModel {
    use crate::model::weights::Weights;
    let (h, v, nh, kvh, hd) = (64, 128, 2, 2, 32);
    let (inter, eps) = (128, 1e-5);

    let specs = vec![
        ("embed".into(), vec![v, h]),
        ("q".into(), vec![h, h]),
        ("k".into(), vec![kvh * hd, h]),
        ("v".into(), vec![kvh * hd, h]),
        ("o".into(), vec![h, h]),
        ("gate".into(), vec![inter, h]),
        ("up".into(), vec![inter, h]),
        ("down".into(), vec![h, inter]),
    ];
    let weights_store = Weights::from_f32(specs, |name| {
        let len = match name {
            "embed" => v * h,
            "q" | "o" => h * h,
            "k" | "v" => kvh * hd * h,
            "gate" | "up" | "down" => inter * h,
            _ => 0,
        };
        Ok(vec![0.05f32; len])
    })
    .unwrap();
    let embed_mat = weights_store.mat("embed").unwrap();
    let q_mat = weights_store.mat("q").unwrap();
    let k_mat = weights_store.mat("k").unwrap();
    let v_mat = weights_store.mat("v").unwrap();
    let o_mat = weights_store.mat("o").unwrap();
    let gate_mat = weights_store.mat("gate").unwrap();
    let up_mat = weights_store.mat("up").unwrap();
    let down_mat = weights_store.mat("down").unwrap();
    let layer = Layer {
        sliding: false,
        head_dim: hd,
        kv_heads: kvh,
        kv_src: 0,
        attn: Attn {
            q: q_mat,
            kv: Some((k_mat, Some(v_mat), vec![1.0; hd])),
            o: o_mat,
            q_norm: vec![1.0; hd],
        },
        gate: gate_mat,
        up: up_mat,
        down: down_mat,
        input_ln: vec![1.0; h],
        post_attn_ln: vec![1.0; h],
        pre_ff_ln: vec![1.0; h],
        post_ff_ln: vec![1.0; h],
        ple: None,
        scalar: 1.0,
        moe: None,
    };

    let model = TextModel {
        hidden: h,
        vocab: v,
        layers_n: 1,
        max_pos: 512,
        heads: nh,
        eps,
        window: 256,
        softcap: Some(30.0),
        embed_scale: 1.0,
        embed: embed_mat,
        ple: None,
        layers: vec![layer],
        first_shared: 1,
        norm: vec![1.0; h],
        inv_freq_sliding: vec![1.0; hd / 2],
        inv_freq_full: vec![1.0; hd / 2],
        pad_id: 0,
        bidir_vision: false,
    };
    model
}

/// Decode graph of [`tiny_model`].
pub fn tiny_decode_graph() -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
    tiny_model().to_decode_graph()
}

/// Tiny synthetic 26B-A4B-style model: a sliding and a full-attention layer, each with the dense MLP
/// beside 8 routed experts (top 2), a per-expert scale, and pseudo-random weights (distinct router
/// logits). It has a decode graph only: the sparse block has no host forward.
pub fn tiny_moe_model() -> TextModel {
    let (h, v, nh, hd, inter) = (64usize, 128usize, 2usize, 32usize, 48usize);
    let (experts, top_k, ff) = (8usize, 2usize, 32usize);
    let cfg = serde_json::json!({
        "hidden_size": h, "num_hidden_layers": 2, "num_attention_heads": nh, "num_key_value_heads": nh,
        "head_dim": hd, "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 8,
        "max_position_embeddings": 64, "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        "enable_moe_block": true, "num_experts": experts, "top_k_experts": top_k, "moe_intermediate_size": ff,
    });
    let p = "model.language_model";
    let mut specs: Vec<(String, Vec<usize>)> = vec![
        (format!("{p}.embed_tokens.weight"), vec![v, h]),
        (format!("{p}.norm.weight"), vec![h]),
    ];
    for l in 0..2 {
        let mut s = |n: &str, shape: Vec<usize>| specs.push((format!("{p}.layers.{l}.{n}"), shape));
        for n in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "router.scale",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
        ] {
            s(n, vec![h]);
        }
        s("self_attn.q_proj.weight", vec![nh * hd, h]);
        s("self_attn.k_proj.weight", vec![nh * hd, h]);
        s("self_attn.v_proj.weight", vec![nh * hd, h]);
        s("self_attn.o_proj.weight", vec![h, nh * hd]);
        s("self_attn.q_norm.weight", vec![hd]);
        s("self_attn.k_norm.weight", vec![hd]);
        s("mlp.gate_proj.weight", vec![inter, h]);
        s("mlp.up_proj.weight", vec![inter, h]);
        s("mlp.down_proj.weight", vec![h, inter]);
        s("layer_scalar", vec![1]);
        s("router.proj.weight", vec![experts, h]);
        s("router.per_expert_scale", vec![experts]);
        s("mlp.experts.gate_proj", vec![experts, ff, h]);
        s("mlp.experts.up_proj", vec![experts, ff, h]);
        s("mlp.experts.down_proj", vec![experts, h, ff]);
    }
    let sizes: std::collections::HashMap<String, usize> = specs
        .iter()
        .map(|(n, s)| (n.clone(), s.iter().product()))
        .collect();
    let weights = crate::model::weights::Weights::from_f32(specs, |name| {
        let n: u64 = name
            .bytes()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1) * b as u64)
            .sum();
        let one = name.ends_with("layer_scalar") || name.ends_with("_norm.weight");
        Ok((0..sizes[name])
            .map(|i| {
                let hash = ((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)
                    ^ n.wrapping_mul(0xD1B54A32D192ED03))
                .wrapping_mul(0x2545F4914F6CDD1D);
                let r = (hash >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                if one {
                    1.0 + 0.2 * r
                } else {
                    0.8 * r
                }
            })
            .collect())
    })
    .unwrap();
    TextModel::load(&weights, &cfg).unwrap()
}

/// Decode graph of [`tiny_moe_model`].
pub fn tiny_moe_decode_graph() -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
    tiny_moe_model().to_decode_graph()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_rope_rotates_only_its_share() {
        let p = serde_json::json!({"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6});
        let f = inv_freq(&p, 512).unwrap();
        assert_eq!(f.len(), 256);
        assert_eq!(f.iter().filter(|&&v| v > 0.0).count(), 64);
        // Exponent uses the full head_dim, not the rotated width.
        assert!((f[1] - 1e6f64.powf(-2.0 / 512.0) as f32).abs() < 1e-9);
    }
    /// The sparse block (router, top-k, banked experts, per-expert scale, both norms) on the Metal
    /// program against the CPU graph executor: the host forward has no sparse block.
    #[test]
    fn gemma4_moe_decode_graph_metal_matches_cpu_graph() {
        let model = tiny_moe_model();
        assert!(model.has_moe());
        let (g, w) = model.to_decode_graph().unwrap();
        assert!(g
            .nodes
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::MoeLinear { .. })));
        #[cfg(target_os = "macos")]
        let mut prog = crate::graph::program::GraphProgram::compile(
            g.clone(),
            w.clone(),
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Metal),
        )
        .expect("compile the Gemma 4 MoE decode graph on Metal");
        #[cfg(target_os = "macos")]
        assert_eq!(prog.cpu_node_count(), 0);
        let mut cpu_state = crate::graph::StateStore::new();
        let mut tok = 5u32;
        for pos in 0..6usize {
            let inputs = || model.decode_inputs(tok, pos as u32);
            let want = crate::graph::cpu::run(&g, &w, &mut cpu_state, inputs()).unwrap();
            let want = want[0].as_f32().unwrap();
            assert!(want.iter().all(|v| v.is_finite()));
            #[cfg(target_os = "macos")]
            {
                let got = prog.run(inputs()).unwrap();
                let got = got[0].as_f32().unwrap();
                let dot: f64 = want
                    .iter()
                    .zip(got)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                let n = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (n(want) * n(got));
                assert!(cos >= 0.999, "step {pos}: cosine {cos}");
            }
            tok = want
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
        }
    }

    /// Greedy decoding on Metal with f16 and with f32 KV caches over the pseudo-random tiny MoE model (a
    /// sliding layer with an 8-position window and a full layer): 48 steps give the same token stream and
    /// logits within f16 rounding of the cached K/V.
    #[cfg(target_os = "macos")]
    #[test]
    fn gemma4_moe_greedy_stream_f16_kv_matches_f32() {
        use crate::graph::metal::{Runtime, Session};
        use crate::graph::StateStore;
        let model = tiny_moe_model();
        let (g, w) = model.to_decode_graph().unwrap();
        let rt = Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir()).unwrap();
        let stream = |kv_f16: bool| {
            let mut session = Session::new(rt.clone(), &w).unwrap();
            session.set_kv_f16(kv_f16);
            let plan = session.prepare_plan(&g).unwrap();
            assert_eq!(!plan.f16_states.is_empty(), kv_f16);
            let mut state = StateStore::new();
            let (mut tok, mut toks, mut all_logits) = (5u32, vec![], vec![]);
            for pos in 0..48u32 {
                let inputs = model.decode_inputs(tok, pos);
                let out = session
                    .run_prepared(&g, &plan, Some(&mut state), inputs)
                    .unwrap();
                let logits = out[0].as_f32().unwrap().to_vec();
                tok = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                toks.push(tok);
                all_logits.push(logits);
            }
            (toks, all_logits)
        };
        let (want_toks, want) = stream(false);
        let (got_toks, got) = stream(true);
        assert_eq!(got_toks, want_toks, "greedy streams diverge");
        for (step, (w, g)) in want.iter().zip(&got).enumerate() {
            let diff = w
                .iter()
                .zip(g)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = w.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            assert!(
                diff / scale < 2e-3,
                "step {step}: max logit diff {diff} (scale {scale})"
            );
        }
    }

    #[test]
    fn test_gemma4_to_decode_graph_cpu_parity() {
        use crate::graph::cpu::run_stateless;

        let model = tiny_model();
        let h = model.hidden;

        // 1. Reference forward pass
        let tok = 5u32;
        let mut emb = vec![0.0f32; h];
        model.embed_row(tok, &mut emb);
        let mut cache = model.new_cache();
        let ref_logits = model.forward(emb, &[tok], 0, &mut cache, &[]);

        // 2. Graph forward pass
        let (decode_g, w) = model.to_decode_graph().unwrap();
        let inputs = model.decode_inputs(tok, 0);
        let graph_res = run_stateless(&decode_g, &w, inputs).expect("graph execution succeeds");
        let graph_logits = graph_res[0].as_f32().expect("f32 logits");

        assert_eq!(ref_logits.len(), graph_logits.len());
        for (i, (&r, &g)) in ref_logits.iter().zip(graph_logits).enumerate() {
            let diff = (r - g).abs();
            assert!(
                diff < 1e-4,
                "Logit mismatch at index {i}: ref={r}, graph={g}"
            );
        }
    }

    /// Fusing q/k/v keeps every other node in its original relative order and puts the fused
    /// projection and its slices where the first fused projection was, so order-sensitive planner
    /// fusions downstream (MoE router/top-k, norm batching) still see their patterns. Graph only:
    /// the builder's loader `Mat` weights are never fused.
    #[test]
    fn qkv_fusion_keeps_node_order_of_moe_decode_graph() {
        let (g, _) = tiny_moe_decode_graph().unwrap();
        let mut fg = g.clone();
        let qkv = crate::graph::egraph::Rule::Qkv;
        assert!(crate::graph::egraph::fuse_rule(&mut fg, None, qkv) > 0);
        let names = |g: &crate::graph::Graph| -> Vec<String> {
            g.nodes.iter().map(|n| n.name.clone()).collect()
        };
        let (before, after) = (names(&g), names(&fg));
        let fused: Vec<&String> = before.iter().filter(|n| !after.contains(n)).collect();
        assert!(!fused.is_empty());
        let kept: Vec<&String> = before.iter().filter(|n| after.contains(n)).collect();
        let kept_after: Vec<&String> = after.iter().filter(|n| before.contains(n)).collect();
        assert_eq!(kept, kept_after);
        // Each layer's fused Linear sits right after the last node that preceded its q.
        for (i, n) in after
            .iter()
            .enumerate()
            .filter(|(_, n)| n.ends_with(".fused_qkv"))
        {
            let q = n.trim_end_matches(".fused_qkv");
            let q_at = before.iter().position(|b| b == q).unwrap();
            assert_eq!(after[i - 1], before[q_at - 1], "{n}");
        }
    }
    #[test]
    fn test_gemma4_to_decode_graph_with_ple_cpu_parity() {
        use crate::graph::cpu::run_stateless;
        use crate::graph::Tensor;
        use crate::model::weights::Weights;

        let (h, v, nh, kvh, hd) = (64, 128, 2, 2, 32);
        let (inter, eps, pd) = (128, 1e-5, 16);

        let specs = vec![
            ("embed".into(), vec![v, h]),
            ("q".into(), vec![h, h]),
            ("k".into(), vec![kvh * hd, h]),
            ("v".into(), vec![kvh * hd, h]),
            ("o".into(), vec![h, h]),
            ("gate".into(), vec![inter, h]),
            ("up".into(), vec![inter, h]),
            ("down".into(), vec![h, inter]),
            ("model_proj".into(), vec![pd, h]),
            ("ple_gate".into(), vec![pd, h]),
            ("ple_proj".into(), vec![h, pd]),
        ];
        let weights_store = Weights::from_f32(specs, |name| {
            let len = match name {
                "embed" => v * h,
                "q" | "o" => h * h,
                "k" | "v" => kvh * hd * h,
                "gate" | "up" | "down" => inter * h,
                "model_proj" | "ple_gate" | "ple_proj" => pd * h,
                _ => 0,
            };
            Ok(vec![0.02f32; len])
        })
        .unwrap();

        let ple_gate_mat = weights_store.mat("ple_gate").unwrap();
        let ple_proj_mat = weights_store.mat("ple_proj").unwrap();

        let layer = Layer {
            sliding: false,
            head_dim: hd,
            kv_heads: kvh,
            kv_src: 0,
            attn: Attn {
                q: weights_store.mat("q").unwrap(),
                kv: Some((
                    weights_store.mat("k").unwrap(),
                    Some(weights_store.mat("v").unwrap()),
                    vec![1.0; hd],
                )),
                o: weights_store.mat("o").unwrap(),
                q_norm: vec![1.0; hd],
            },
            gate: weights_store.mat("gate").unwrap(),
            up: weights_store.mat("up").unwrap(),
            down: weights_store.mat("down").unwrap(),
            input_ln: vec![1.0; h],
            post_attn_ln: vec![1.0; h],
            pre_ff_ln: vec![1.0; h],
            post_ff_ln: vec![1.0; h],
            ple: Some((ple_gate_mat, ple_proj_mat, vec![1.0; h])),
            scalar: 1.0,
            moe: None,
        };

        let ple_model = Ple {
            dim: pd,
            embed: PleTable::lazy(
                v,
                pd,
                Box::new(|id: usize, out: &mut [f32]| {
                    for (i, o) in out.iter_mut().enumerate() {
                        *o = 0.01 * ((id + i) % 10 + 1) as f32;
                    }
                }),
            ),
            model_proj: weights_store.mat("model_proj").unwrap(),
            proj_norm: vec![1.0; pd],
        };

        let model = TextModel {
            hidden: h,
            vocab: v,
            layers_n: 1,
            max_pos: 512,
            heads: nh,
            eps,
            window: 256,
            softcap: Some(30.0),
            embed_scale: 1.0,
            embed: weights_store.mat("embed").unwrap(),
            ple: Some(ple_model),
            layers: vec![layer],
            first_shared: 1,
            norm: vec![1.0; h],
            inv_freq_sliding: vec![1.0; hd / 2],
            inv_freq_full: vec![1.0; hd / 2],
            pad_id: 0,
            bidir_vision: false,
        };

        // 1. Reference forward pass with PLE
        let tok = 5u32;
        let mut emb = vec![0.0f32; h];
        model.embed_row(tok, &mut emb);
        let mut cache = model.new_cache();
        let ref_logits = model.forward(emb, &[tok], 0, &mut cache, &[]);

        // 2. Graph forward pass with lowered PLE
        let (decode_g, w) = model
            .to_decode_graph_unchecked()
            .expect("to_decode_graph_unchecked with PLE succeeds");
        assert!(decode_g.nodes.iter().any(|n| n.name.contains("model.ple")));
        assert!(decode_g
            .nodes
            .iter()
            .any(|n| n.name.contains("layers.0.ple")));

        let mut ple_e = vec![0.0f32; pd];
        model
            .ple
            .as_ref()
            .unwrap()
            .embed
            .get_row(tok as usize, &mut ple_e);
        let ident_scale = (pd as f32).sqrt();
        for v in &mut ple_e {
            *v *= ident_scale;
        }
        let mut emb_in = vec![0.0f32; h];
        model.embed_row(tok, &mut emb_in);
        let inputs = vec![
            Tensor::F32(emb_in),
            Tensor::U32(vec![0]),
            Tensor::F32(ple_e),
        ];
        let graph_res = run_stateless(&decode_g, &w, inputs).expect("graph execution succeeds");
        let graph_logits = graph_res[0].as_f32().expect("f32 logits");

        assert_eq!(ref_logits.len(), graph_logits.len());
        for (i, (&r, &g)) in ref_logits.iter().zip(graph_logits).enumerate() {
            let diff = (r - g).abs();
            assert!(
                diff < 1e-4 || (r - g).abs() / r.abs().max(1.0) < 1e-3,
                "PLE logit mismatch at index {i}: ref={r}, graph={g}"
            );
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn test_gemma4_to_decode_graph_metal_matches_cpu() {
        use crate::graph::cpu::run_stateless;
        use crate::graph::metal::{Runtime, Session};
        use crate::model::weights::Weights;

        let (h, v, nh, kvh, hd) = (64, 128, 2, 2, 32);
        let (inter, eps) = (128, 1e-5);

        let specs = vec![
            ("embed".into(), vec![v, h]),
            ("q".into(), vec![h, h]),
            ("k".into(), vec![kvh * hd, h]),
            ("v".into(), vec![kvh * hd, h]),
            ("o".into(), vec![h, h]),
            ("gate".into(), vec![inter, h]),
            ("up".into(), vec![inter, h]),
            ("down".into(), vec![h, inter]),
        ];
        let weights_store = Weights::from_f32(specs, |name| {
            let len = match name {
                "embed" => v * h,
                "q" | "o" => h * h,
                "k" | "v" => kvh * hd * h,
                "gate" | "up" | "down" => inter * h,
                _ => 0,
            };
            Ok(vec![0.05f32; len])
        })
        .unwrap();

        let layer = Layer {
            sliding: false,
            head_dim: hd,
            kv_heads: kvh,
            kv_src: 0,
            attn: Attn {
                q: weights_store.mat("q").unwrap(),
                kv: Some((
                    weights_store.mat("k").unwrap(),
                    Some(weights_store.mat("v").unwrap()),
                    vec![1.0; hd],
                )),
                o: weights_store.mat("o").unwrap(),
                q_norm: vec![1.0; hd],
            },
            gate: weights_store.mat("gate").unwrap(),
            up: weights_store.mat("up").unwrap(),
            down: weights_store.mat("down").unwrap(),
            input_ln: vec![1.0; h],
            post_attn_ln: vec![1.0; h],
            pre_ff_ln: vec![1.0; h],
            post_ff_ln: vec![1.0; h],
            ple: None,
            scalar: 1.0,
            moe: None,
        };

        let model = TextModel {
            hidden: h,
            vocab: v,
            layers_n: 1,
            max_pos: 512,
            heads: nh,
            eps,
            window: 256,
            softcap: Some(30.0),
            embed_scale: 1.0,
            embed: weights_store.mat("embed").unwrap(),
            ple: None,
            layers: vec![layer],
            first_shared: 1,
            norm: vec![1.0; h],
            inv_freq_sliding: vec![1.0; hd / 2],
            inv_freq_full: vec![1.0; hd / 2],
            pad_id: 0,
            bidir_vision: false,
        };

        let (decode_g, w) = model.to_decode_graph().unwrap();
        let tok = 7u32;
        let inputs = model.decode_inputs(tok, 0);

        let cpu_res = run_stateless(&decode_g, &w, inputs.clone()).expect("cpu run succeeds");
        let cpu_logits = cpu_res[0].as_f32().expect("f32 cpu logits");

        let rt = Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir()).unwrap();
        let mut session = Session::new(rt, &w).expect("session succeeds");
        session.set_kv_f16(false);
        let metal_res = session.run(&decode_g, inputs).expect("metal run succeeds");
        let metal_logits = metal_res[0].as_f32().expect("f32 metal logits");

        assert_eq!(cpu_logits.len(), metal_logits.len());
        for (i, (&c, &m)) in cpu_logits.iter().zip(metal_logits).enumerate() {
            let diff = (c - m).abs();
            assert!(
                diff < 1e-4,
                "Metal mismatch at index {i}: cpu={c}, metal={m}"
            );
        }
    }

    /// The batched prefill graph (last-row logits; `visible_end` widening sliding layers over an image
    /// block) against the host forward, on random weights with one sliding and one full layer.
    #[cfg(target_os = "macos")]
    #[test]
    fn prefill_graph_matches_host_with_image_block() {
        use crate::graph::metal::{Runtime, Session};
        use crate::graph::{StateStore, Tensor};
        use crate::model::weights::Weights;

        let (h, v, nh, kvh, hd, inter) = (64, 128, 2, 2, 32, 128);
        let specs = vec![
            ("embed".into(), vec![v, h]),
            ("q".into(), vec![nh * hd, h]),
            ("k".into(), vec![kvh * hd, h]),
            ("v".into(), vec![kvh * hd, h]),
            ("o".into(), vec![h, nh * hd]),
            ("gate".into(), vec![inter, h]),
            ("up".into(), vec![inter, h]),
            ("down".into(), vec![h, inter]),
        ];
        let store = Weights::from_f32(specs, |name| {
            let len = match name {
                "embed" => v * h,
                "q" => nh * hd * h,
                "o" => h * nh * hd,
                "k" | "v" => kvh * hd * h,
                _ => inter * h,
            };
            let mut s = 0x9E37_79B9u32 ^ (name.len() as u32 * 7919);
            Ok((0..len)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((s >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 0.4
                })
                .collect())
        })
        .unwrap();
        let layer = |sliding: bool, li: usize| Layer {
            sliding,
            head_dim: hd,
            kv_heads: kvh,
            kv_src: li,
            attn: Attn {
                q: store.mat("q").unwrap(),
                kv: Some((
                    store.mat("k").unwrap(),
                    Some(store.mat("v").unwrap()),
                    vec![1.0; hd],
                )),
                o: store.mat("o").unwrap(),
                q_norm: vec![1.0; hd],
            },
            gate: store.mat("gate").unwrap(),
            up: store.mat("up").unwrap(),
            down: store.mat("down").unwrap(),
            input_ln: vec![1.0; h],
            post_attn_ln: vec![1.0; h],
            pre_ff_ln: vec![1.0; h],
            post_ff_ln: vec![1.0; h],
            ple: None,
            moe: None,
            scalar: 1.0,
        };
        let model = TextModel {
            hidden: h,
            vocab: v,
            layers_n: 2,
            max_pos: 64,
            heads: nh,
            eps: 1e-5,
            window: 4,
            softcap: Some(30.0),
            embed_scale: 1.0,
            embed: store.mat("embed").unwrap(),
            ple: None,
            layers: vec![layer(true, 0), layer(false, 1)],
            first_shared: 2,
            norm: vec![1.0; h],
            inv_freq_sliding: (0..hd / 2)
                .map(|i| 1.0 / 10f32.powi(i as i32 % 4))
                .collect(),
            inv_freq_full: (0..hd / 2)
                .map(|i| 1.0 / 100f32.powi(i as i32 % 3))
                .collect(),
            pad_id: 0,
            bidir_vision: true,
        };
        let ids: Vec<u32> = (0..10).map(|i| 3 + 5 * i).collect();
        let mut x = vec![0.0f32; ids.len() * h];
        for (row, &t) in x.chunks_exact_mut(h).zip(&ids) {
            model.embed_row(t, row);
        }
        let graph = model.to_prefill_graph(ids.len()).unwrap();
        let weights = model.to_decode_graph().unwrap().1;
        let mut session = Session::new(Runtime::shared().unwrap(), &weights).unwrap();
        session.set_kv_f16(false);
        for blocks in [vec![], vec![3..7]] {
            let want = model.forward(x.clone(), &ids, 0, &mut model.new_cache(), &blocks);
            let visible_end = (0..ids.len())
                .map(|p| {
                    blocks
                        .iter()
                        .find(|b| b.contains(&p))
                        .map_or(p + 1, |b| b.end) as u32
                })
                .collect();
            let inputs = vec![
                Tensor::F32(x.clone()),
                Tensor::U32((0..ids.len() as u32).collect()),
                Tensor::U32(visible_end),
            ];
            let got = session
                .run_with_state(&graph, Some(&mut StateStore::new()), inputs)
                .unwrap();
            let got = got[0].as_f32().unwrap();
            assert_eq!(got.len(), want.len());
            let worst = want
                .iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 2e-3,
                "blocks {blocks:?}: max |host - metal| = {worst}"
            );
        }
    }

    #[test]
    #[ignore = "real-checkpoint: E-series PLE gated to eager CPU"]
    fn test_gemma4_cpu_fallback_coherent() {
        use crate::hardware::Device;
        use crate::model::hub::HuggingFaceHub;
        use crate::model::native::load_native;

        let repo = "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive";
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

        println!("Loaded Gemma 4 model successfully!");
        println!("Backend label: {}", lm.backend_label());
        println!("Device: {:?}", lm.device());

        let rss_bytes = crate::hardware::current_resident_memory_bytes();
        let rss_gb = rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("Resident memory (RSS): {:.2} GB", rss_gb);

        let tok = lm.tokenizer();
        let prompt = "The capital of France is";
        let ids = tok.encode(prompt);
        let (seq_ids, mut state) = lm.prepare(ids, &[]).expect("prepare sequence");

        let logits = lm.forward(&seq_ids, 0, &mut state).expect("forward prompt");
        let next_token = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap() as u32;
        let mut pos = seq_ids.len();
        let mut generated = Vec::new();

        let decode_t0 = std::time::Instant::now();
        let mut cur_token = next_token;
        for _ in 0..8 {
            if tok.is_special(cur_token) {
                break;
            }
            generated.push(cur_token);
            let next_logits = lm
                .forward(&[cur_token], pos, &mut state)
                .expect("decode step");
            pos += 1;
            cur_token = (0..next_logits.len())
                .max_by(|&a, &b| next_logits[a].total_cmp(&next_logits[b]))
                .unwrap() as u32;
        }
        let decode_elapsed = decode_t0.elapsed();
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
        assert!(!generated.is_empty(), "Must generate tokens");
        let text = tok.decode(&generated);
        println!("Generated text: {:?}", text);
        // On this 2-bit aggressive finetune checkpoint, greedy decode without chat template repeats prompt tokens
        assert!(!text.is_empty(), "Generated text must not be empty");
        assert_eq!(
            lm.device(),
            Device::Cpu,
            "Gemma 4 E-series falls back to CPU due to PLE and Q2_K"
        );
    }
}
