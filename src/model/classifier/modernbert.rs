//! `ModernBertForSequenceClassification` (transformers `modeling_modernbert.py`) on the CPU in
//! f32: pre-LN encoder with RoPE, alternating global / sliding-window attention and a GeGLU MLP;
//! the head pools (CLS or mean), then dense → activation → LayerNorm → classifier.

use super::head::{self, Gb, HeadGraph, HeadLin, HeadPool, HeadPost};
use super::{Checkpoint, Lin};
use crate::graph::{DType, Op, RopeVariant, Tensor};
use crate::model::nn;
use crate::model::par::{par_for, OutPtr};
use serde_json::Value;

#[derive(Clone, Copy)]
enum Act {
    Gelu,
    GeluTanh,
    Silu,
    Relu,
}

impl Act {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "gelu" => Act::Gelu,
            "gelu_pytorch_tanh" | "gelu_new" => Act::GeluTanh,
            "silu" | "swish" => Act::Silu,
            "relu" => Act::Relu,
            other => return Err(format!("ModernBERT activation {other} is not supported")),
        })
    }

    fn apply(self, x: &mut [f32]) {
        match self {
            Act::Gelu => nn::gelu(x),
            Act::GeluTanh => nn::gelu_tanh(x),
            Act::Silu => nn::silu(x),
            Act::Relu => x.iter_mut().for_each(|v| *v = v.max(0.0)),
        }
    }

    fn graph(self) -> head::Act {
        match self {
            Act::Gelu => head::Act::Gelu,
            Act::GeluTanh => head::Act::GeluTanh,
            Act::Silu => head::Act::Silu,
            Act::Relu => head::Act::Relu,
        }
    }
}

/// LayerNorm weight and bias (zeros when the config has `norm_bias: false`).
type Norm = (Vec<f32>, Vec<f32>);

struct Layer {
    /// `None` for layer 0 (`nn.Identity`).
    attn_norm: Option<Norm>,
    wqkv: Lin,
    wo: Lin,
    mlp_norm: Norm,
    wi: Lin,
    mlp_wo: Lin,
    /// `Some(half window)` for sliding-window layers: keys with `|i - j| <= half`.
    window: Option<usize>,
    /// RoPE inverse frequencies `[head_dim / 2]` of this layer's type and the cos/sin scale.
    rope: (Vec<f32>, f32),
}

pub(super) struct ModernBert {
    hidden: usize,
    heads: usize,
    eps: f32,
    act: Act,
    tok_emb: Vec<f32>,
    emb_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    mean_pool: bool,
    head_dense: Lin,
    head_act: Act,
    head_norm: Norm,
    pub(super) classifier: Lin,
}

/// RoPE inverse frequencies `[d / 2]` and the cos/sin scale (YaRN `attention_factor`) of layer
/// type `ty`: `rope_parameters[ty]` (new configs), else the legacy `rope_scaling` applied to both
/// types with `global_rope_theta` / `local_rope_theta` (HF defaults 160k / 10k).
pub(crate) fn rope_inv_freq(cfg: &Value, ty: &str, d: usize) -> Result<(Vec<f32>, f32), String> {
    let p = match &cfg["rope_parameters"][ty] {
        Value::Null => &cfg["rope_scaling"],
        p => p,
    };
    let (legacy, default) = if ty == "full_attention" {
        ("global_rope_theta", 160_000.0)
    } else {
        ("local_rope_theta", 10_000.0)
    };
    let base = p["rope_theta"]
        .as_f64()
        .or_else(|| cfg[legacy].as_f64())
        .unwrap_or(default);
    if p["partial_rotary_factor"]
        .as_f64()
        .is_some_and(|f| f != 1.0)
    {
        return Err("ModernBERT partial_rotary_factor is not supported".into());
    }
    let half = d / 2;
    let pos_freq = |i: usize| (base as f32).powf((2 * i) as f32 / d as f32);
    let extrapolation: Vec<f32> = (0..half).map(|i| 1.0 / pos_freq(i)).collect();
    match p["rope_type"].as_str().unwrap_or("default") {
        "default" => Ok((extrapolation, 1.0)),
        // transformers `_compute_yarn_parameters`.
        "yarn" => {
            let orig = p["original_max_position_embeddings"]
                .as_f64()
                .ok_or("YaRN rope has no original_max_position_embeddings")?;
            let max_pos = cfg["max_position_embeddings"]
                .as_u64()
                .ok_or("config.json has no max_position_embeddings")?;
            let factor = p["factor"].as_f64().unwrap_or(max_pos as f64 / orig);
            let mscale = |m: f64| {
                if factor <= 1.0 {
                    1.0
                } else {
                    0.1 * m * factor.ln() + 1.0
                }
            };
            let attention = match (
                p["attention_factor"].as_f64(),
                p["mscale"].as_f64().filter(|&m| m != 0.0),
                p["mscale_all_dim"].as_f64().filter(|&m| m != 0.0),
            ) {
                (Some(a), _, _) => a,
                (None, Some(m), Some(all)) => mscale(m) / mscale(all),
                _ => mscale(1.0),
            };
            let beta_fast = p["beta_fast"].as_f64().unwrap_or(32.0);
            let beta_slow = p["beta_slow"].as_f64().unwrap_or(1.0);
            let corr = |rot: f64| {
                d as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
            };
            let (mut low, mut high) = (corr(beta_fast), corr(beta_slow));
            if p["truncate"].as_bool().unwrap_or(true) {
                (low, high) = (low.floor(), high.ceil());
            }
            let (low, mut high) = (low.max(0.0) as f32, high.min(d as f64 - 1.0) as f32);
            if low == high {
                high += 0.001;
            }
            let inv = (0..half)
                .map(|i| {
                    let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    let extra = 1.0 - ramp;
                    let inter = 1.0 / (factor as f32 * pos_freq(i));
                    inter * (1.0 - extra) + extrapolation[i] * extra
                })
                .collect();
            Ok((inv, attention as f32))
        }
        other => Err(format!("ModernBERT rope_type {other} is not supported")),
    }
}

impl ModernBert {
    pub(super) fn load(cfg: &Value, ckpt: &Checkpoint) -> Result<Self, String> {
        let num = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {k}"))
        };
        let (h, heads, inter, n_layers, vocab) = (
            num("hidden_size")?,
            num("num_attention_heads")?,
            num("intermediate_size")?,
            num("num_hidden_layers")?,
            num("vocab_size")?,
        );
        if heads == 0 || h % heads != 0 {
            return Err(format!(
                "hidden_size {h} is not a multiple of {heads} heads"
            ));
        }
        let d = h / heads;
        let eps = cfg["norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let norm_bias = cfg["norm_bias"].as_bool().unwrap_or(false);
        let norm = |name: &str| -> Result<Norm, String> {
            Ok((
                ckpt.get(&format!("{name}.weight"), &[h])?,
                if norm_bias {
                    ckpt.get(&format!("{name}.bias"), &[h])?
                } else {
                    vec![0.0; h]
                },
            ))
        };
        let window = num("local_attention").unwrap_or(128) / 2;
        // Newer configs list `layer_types`; older ones give `global_attn_every_n_layers`.
        let types: Vec<String> = match cfg["layer_types"].as_array() {
            Some(a) => a
                .iter()
                .map(|t| t.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or("config.json layer_types is not a list of strings")?,
            None => {
                let every = cfg["global_attn_every_n_layers"].as_u64().unwrap_or(3) as usize;
                (0..n_layers)
                    .map(|i| {
                        if i % every.max(1) != 0 {
                            "sliding_attention"
                        } else {
                            "full_attention"
                        }
                        .to_string()
                    })
                    .collect()
            }
        };
        if types.len() != n_layers {
            return Err(format!(
                "config.json has {} layer_types for {n_layers} layers",
                types.len()
            ));
        }
        let rope = |ty: &str| rope_inv_freq(cfg, ty, d);
        let lin = |name: &str, in_f| ckpt.linear(name, in_f);
        let layers = types
            .iter()
            .enumerate()
            .map(|(l, ty)| {
                let p = format!("model.layers.{l}");
                let window = match ty.as_str() {
                    "full_attention" => None,
                    "sliding_attention" => Some(window),
                    other => return Err(format!("ModernBERT layer type {other} is not supported")),
                };
                let layer = Layer {
                    attn_norm: (l > 0)
                        .then(|| norm(&format!("{p}.attn_norm")))
                        .transpose()?,
                    wqkv: lin(&format!("{p}.attn.Wqkv"), h)?,
                    wo: lin(&format!("{p}.attn.Wo"), h)?,
                    mlp_norm: norm(&format!("{p}.mlp_norm"))?,
                    wi: lin(&format!("{p}.mlp.Wi"), h)?,
                    mlp_wo: lin(&format!("{p}.mlp.Wo"), inter)?,
                    window,
                    rope: rope(ty)?,
                };
                if layer.wqkv.out_f != 3 * h || layer.wo.out_f != h || layer.wi.out_f != 2 * inter {
                    return Err(format!("layer {l}: unexpected Wqkv/Wo/Wi shapes"));
                }
                Ok(layer)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mean_pool = match cfg["classifier_pooling"].as_str().unwrap_or("cls") {
            "cls" => false,
            "mean" => true,
            other => return Err(format!("classifier_pooling {other} is not supported")),
        };
        let head_dense = lin("head.dense", h)?;
        if head_dense.out_f != h {
            return Err("head.dense is not [hidden, hidden]".into());
        }
        Ok(Self {
            hidden: h,
            heads,
            eps,
            act: Act::parse(cfg["hidden_activation"].as_str().unwrap_or("gelu"))?,
            tok_emb: ckpt.get("model.embeddings.tok_embeddings.weight", &[vocab, h])?,
            emb_norm: norm("model.embeddings.norm")?,
            layers,
            final_norm: norm("model.final_norm")?,
            mean_pool,
            head_dense,
            head_act: Act::parse(cfg["classifier_activation"].as_str().unwrap_or("gelu"))?,
            head_norm: norm("head.norm")?,
            classifier: lin("classifier", h)?,
        })
    }

    fn ln(&self, x: &[f32], n: usize, (g, b): &Norm) -> Vec<f32> {
        nn::layer_norm(x, n, self.hidden, g, b, self.eps)
    }

    /// Logits `[num_labels]` of one unpadded sequence.
    pub(super) fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        crate::hardware::cpu_compute::enter("classifier::modernbert::forward");
        let (n, h, heads) = (ids.len(), self.hidden, self.heads);
        let d = h / heads;
        let rows = self.tok_emb.len() / h;
        let mut x = Vec::with_capacity(n * h);
        for &id in ids {
            let id = id as usize;
            if id >= rows {
                return Err(format!(
                    "token id {id} is outside the {rows}-row vocabulary"
                ));
            }
            x.extend_from_slice(&self.tok_emb[id * h..(id + 1) * h]);
        }
        x = self.ln(&x, n, &self.emb_norm);
        let scale = 1.0 / (d as f32).sqrt();
        for layer in &self.layers {
            let a = match &layer.attn_norm {
                Some(norm) => self.ln(&x, n, norm),
                None => x.clone(),
            };
            let mut qkv = layer.wqkv.apply(&a, n);
            // RoPE on q and k (rotate_half: pairs (i, i + d/2)), positions 0..n.
            let half = d / 2;
            for (pos, row) in qkv.chunks_exact_mut(3 * h).enumerate() {
                for (i, f) in layer.rope.0.iter().enumerate() {
                    let (s, c) = (pos as f32 * f).sin_cos();
                    let (s, c) = (s * layer.rope.1, c * layer.rope.1);
                    for base in (0..2 * heads).map(|t| t * d) {
                        let (x1, x2) = (row[base + i], row[base + i + half]);
                        row[base + i] = x1 * c - x2 * s;
                        row[base + i + half] = x2 * c + x1 * s;
                    }
                }
            }
            let mut ctx = vec![0.0f32; n * h];
            let out = OutPtr::new(&mut ctx);
            let qkv = &qkv;
            par_for(heads, &|hd| {
                let mut scores = Vec::with_capacity(n);
                for i in 0..n {
                    let (lo, hi) = match layer.window {
                        Some(w) => (i.saturating_sub(w), (i + w + 1).min(n)),
                        None => (0, n),
                    };
                    let q = &qkv[i * 3 * h + hd * d..][..d];
                    scores.clear();
                    scores.extend((lo..hi).map(|j| {
                        let k = &qkv[j * 3 * h + h + hd * d..][..d];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                    }));
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    // SAFETY: row i, head hd's d columns belong to this task only.
                    let o = unsafe { out.slice(i * h + hd * d, d) };
                    for (j, p) in (lo..hi).zip(&scores) {
                        let v = &qkv[j * 3 * h + 2 * h + hd * d..][..d];
                        let p = p / sum;
                        o.iter_mut().zip(v).for_each(|(o, v)| *o += p * v);
                    }
                }
            });
            nn::add_in_place(&mut x, &layer.wo.apply(&ctx, n));
            let m = self.ln(&x, n, &layer.mlp_norm);
            let wi = layer.wi.apply(&m, n);
            let inter = layer.mlp_wo.in_f;
            let mut g: Vec<f32> = Vec::with_capacity(n * inter);
            for row in wi.chunks_exact(2 * inter) {
                g.extend_from_slice(&row[..inter]);
            }
            self.act.apply(&mut g);
            for (gr, row) in g.chunks_exact_mut(inter).zip(wi.chunks_exact(2 * inter)) {
                gr.iter_mut().zip(&row[inter..]).for_each(|(a, b)| *a *= b);
            }
            nn::add_in_place(&mut x, &layer.mlp_wo.apply(&g, n));
        }
        x = self.ln(&x, n, &self.final_norm);
        let pooled = if self.mean_pool {
            nn::mean_pool(&x, &vec![1; n], h)
        } else {
            x[..h].to_vec()
        };
        let mut y = self.head_dense.apply(&pooled, 1);
        self.head_act.apply(&mut y);
        let y = self.ln(&y, 1, &self.head_norm);
        Ok(self.classifier.apply(&y, 1))
    }

    /// The pool → dense → act → LayerNorm → classifier head as graph nodes.
    pub(super) fn head_graph(&self, post: HeadPost) -> HeadGraph {
        HeadGraph {
            pool: if self.mean_pool { HeadPool::Mean } else { HeadPool::First },
            normalize: false,
            dense: Some((HeadLin::from(&self.head_dense), self.head_act.graph())),
            norm: Some((self.head_norm.0.clone(), self.head_norm.1.clone(), self.eps)),
            out: HeadLin::from(&self.classifier),
            post,
        }
    }

    /// The Metal program for `ids`: embeddings, pre-LN layers (RoPE, global / windowed attention,
    /// GeGLU) and `head`, as graph nodes on `gb`; returns the graph inputs.
    pub(super) fn graph(&self, gb: &mut Gb, ids: &[u32], head: &HeadGraph) -> Result<Vec<Tensor>, String> {
        let (n, h, heads) = (ids.len(), self.hidden, self.heads);
        let d = h / heads;
        let id_v = gb.g.input("input_ids", vec![n], DType::U32);
        let pos_v = gb.g.input("position_ids", vec![n], DType::U32);
        let mask_v = gb.g.input("attention_mask", vec![n], DType::U32);
        let table = gb.weight("embeddings.tok", vec![self.tok_emb.len() / h, h], || self.tok_emb.clone());
        let e = gb.add("emb.tok", Op::Embedding, &[table, id_v], vec![n, h]);
        let mut x = gb.ln(e, "embeddings.norm", &self.emb_norm, self.eps);
        for (l, layer) in self.layers.iter().enumerate() {
            let p = format!("layer.{l}");
            let a = match &layer.attn_norm {
                Some(norm) => gb.ln(x, &format!("{p}.attn_norm"), norm, self.eps),
                None => x,
            };
            let qkv = gb.lin(a, &format!("{p}.Wqkv"), &layer.wqkv);
            let inv = gb.weight(&format!("{p}.inv_freq"), vec![d / 2], || layer.rope.0.clone());
            // rotate_half pairs (i, i + d/2) = HalfSplit; the YaRN attention factor scales cos/sin.
            let rope = Op::Rope {
                head_dim: d,
                rotary_dim: d,
                variant: RopeVariant::HalfSplit,
                mscale: layer.rope.1,
            };
            let q = gb.slice(&format!("{p}.q"), qkv, -1, 0, h);
            let q = gb.add(&format!("{p}.rope_q"), rope.clone(), &[q, pos_v, inv], vec![n, h]);
            let k = gb.slice(&format!("{p}.k"), qkv, -1, h, 2 * h);
            let k = gb.add(&format!("{p}.rope_k"), rope, &[k, pos_v, inv], vec![n, h]);
            let v = gb.slice(&format!("{p}.v"), qkv, -1, 2 * h, 3 * h);
            // Keys `|i - j| <= half` = the op's non-causal window `w / 2`.
            let attn = Op::Attention {
                heads,
                kv_heads: heads,
                causal: false,
                window: layer.window.map(|half| 2 * half),
                soft_cap: None,
                scale: None,
            };
            let c = gb.add(&format!("{p}.attn"), attn, &[q, k, v], vec![n, h]);
            let o = gb.lin(c, &format!("{p}.Wo"), &layer.wo);
            x = gb.add(&format!("{p}.res1"), Op::Binary { kind: crate::graph::BinaryKind::Add }, &[x, o], vec![n, h]);
            let m = gb.ln(x, &format!("{p}.mlp_norm"), &layer.mlp_norm, self.eps);
            let wi = gb.lin(m, &format!("{p}.Wi"), &layer.wi);
            let inter = layer.mlp_wo.in_f;
            let gate = gb.slice(&format!("{p}.gate"), wi, -1, 0, inter);
            let up = gb.slice(&format!("{p}.up"), wi, -1, inter, 2 * inter);
            let gate = gb.act(&format!("{p}.act"), self.act.graph(), gate);
            let gu = gb.add(&format!("{p}.glu"), Op::Binary { kind: crate::graph::BinaryKind::Mul }, &[gate, up], vec![n, inter]);
            let mo = gb.lin(gu, &format!("{p}.mlp_Wo"), &layer.mlp_wo);
            x = gb.add(&format!("{p}.res2"), Op::Binary { kind: crate::graph::BinaryKind::Add }, &[x, mo], vec![n, h]);
        }
        let x = gb.ln(x, "final_norm", &self.final_norm, self.eps);
        let out = head.nodes(gb, x, Some(mask_v))?;
        gb.g.output(out);
        Ok(vec![
            Tensor::U32(ids.to_vec()),
            Tensor::U32((0..n as u32).collect()),
            Tensor::U32(vec![1; n]),
        ])
    }
}
