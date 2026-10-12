//! Post-LN BERT encoders with factorized embeddings on the CPU: ELECTRA (`embedding_size`
//! embeddings, then `embeddings_project` to the hidden size when they differ) and ALBERT
//! (`embedding_hidden_mapping_in`, layer groups shared across `num_hidden_layers`).

use super::head::{Act, Gb, HeadGraph};
use super::{Checkpoint, Lin};
use crate::graph::{DType, Op, Tensor};
use crate::model::nn;
use serde_json::Value;

struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln1: (Vec<f32>, Vec<f32>),
    inter: Lin,
    out: Lin,
    ln2: (Vec<f32>, Vec<f32>),
}

pub(super) struct Factorized {
    pub(super) hidden: usize,
    heads: usize,
    eps: f32,
    /// `gelu_new` (tanh approximation) instead of exact GELU.
    gelu_tanh: bool,
    emb: usize,
    word: Vec<f32>,
    pos: Vec<f32>,
    types: Vec<f32>,
    emb_ln: (Vec<f32>, Vec<f32>),
    /// `embedding_size` → `hidden_size`.
    proj: Option<Lin>,
    layers: Vec<Layer>,
    /// The layer each of the `num_hidden_layers` steps runs (ALBERT shares them).
    schedule: Vec<usize>,
}

/// LayerNorm `name` over `d`: `weight`/`bias`, or the TF-era `gamma`/`beta` some ALBERT
/// checkpoints keep (transformers renames them on load).
fn layer_norm(ckpt: &Checkpoint, name: &str, d: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    if ckpt.has(&format!("{name}.gamma")) {
        Ok((
            ckpt.get(&format!("{name}.gamma"), &[d])?,
            ckpt.get(&format!("{name}.beta"), &[d])?,
        ))
    } else {
        ckpt.layer_norm(name, d)
    }
}

impl Factorized {
    /// `ElectraModel` (`albert == false`, tensors under `electra.`) or `AlbertModel` (`albert.`).
    pub(super) fn load(cfg: &Value, ckpt: &Checkpoint, albert: bool) -> Result<Self, String> {
        let num = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {k}"))
        };
        let (h, heads, inter, n_layers) = (
            num("hidden_size")?,
            num("num_attention_heads")?,
            num("intermediate_size")?,
            num("num_hidden_layers")?,
        );
        let e = cfg["embedding_size"].as_u64().map_or(h, |v| v as usize);
        let gelu_tanh = match cfg["hidden_act"].as_str().unwrap_or("gelu") {
            "gelu" => false,
            "gelu_new" | "gelu_pytorch_tanh" if albert => true,
            other => return Err(format!("hidden_act {other} is not supported")),
        };
        if heads == 0 || h % heads != 0 {
            return Err(format!(
                "hidden_size {h} is not divisible by num_attention_heads {heads}"
            ));
        }
        if cfg["position_embedding_type"]
            .as_str()
            .unwrap_or("absolute")
            != "absolute"
        {
            return Err("only absolute position embeddings are supported".into());
        }
        let p = if albert { "albert" } else { "electra" };
        let (groups, inner) = if albert {
            (num("num_hidden_groups")?, num("inner_group_num")?)
        } else {
            (n_layers, 1)
        };
        if groups == 0 || inner == 0 || n_layers % groups != 0 {
            return Err(format!(
                "{n_layers} layers do not split into {groups} groups of {inner}"
            ));
        }
        let lin = |name: String, in_f| ckpt.linear(&name, in_f);
        let mut layers = Vec::with_capacity(groups * inner);
        for g in 0..groups {
            for i in 0..inner {
                let l = if albert {
                    format!("{p}.encoder.albert_layer_groups.{g}.albert_layers.{i}")
                } else {
                    format!("{p}.encoder.layer.{g}")
                };
                let n = |a: &str, b: &str| format!("{l}.{}", if albert { a } else { b });
                layers.push(Layer {
                    q: lin(n("attention.query", "attention.self.query"), h)?,
                    k: lin(n("attention.key", "attention.self.key"), h)?,
                    v: lin(n("attention.value", "attention.self.value"), h)?,
                    o: lin(n("attention.dense", "attention.output.dense"), h)?,
                    ln1: layer_norm(
                        ckpt,
                        &n("attention.LayerNorm", "attention.output.LayerNorm"),
                        h,
                    )?,
                    inter: lin(n("ffn", "intermediate.dense"), h)?,
                    out: lin(n("ffn_output", "output.dense"), inter)?,
                    ln2: layer_norm(ckpt, &n("full_layer_layer_norm", "output.LayerNorm"), h)?,
                });
            }
        }
        // AlbertTransformer: step i runs group int(i / (layers / groups)), all its inner layers.
        let schedule = (0..n_layers)
            .flat_map(|i| {
                let g = i / (n_layers / groups);
                g * inner..(g + 1) * inner
            })
            .collect();
        let proj = if albert {
            Some(lin(format!("{p}.encoder.embedding_hidden_mapping_in"), e)?)
        } else if e != h {
            Some(lin(format!("{p}.embeddings_project"), e)?)
        } else {
            None
        };
        let emb =
            |s: &str, rows: usize| ckpt.get(&format!("{p}.embeddings.{s}.weight"), &[rows, e]);
        Ok(Self {
            hidden: h,
            heads,
            eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32,
            gelu_tanh,
            emb: e,
            word: emb("word_embeddings", num("vocab_size")?)?,
            pos: emb("position_embeddings", num("max_position_embeddings")?)?,
            types: emb("token_type_embeddings", num("type_vocab_size")?)?,
            emb_ln: layer_norm(ckpt, &format!("{p}.embeddings.LayerNorm"), e)?,
            proj,
            layers,
            schedule,
        })
    }

    /// Last hidden states `[n, hidden]` of one unpadded sequence.
    pub(super) fn forward(&self, ids: &[u32], types: &[u32]) -> Result<Vec<f32>, String> {
        crate::hardware::cpu_compute::enter("classifier::factorized::forward");
        let (n, e, h) = (ids.len(), self.emb, self.hidden);
        fn row<'a>(table: &'a [f32], e: usize, i: usize, what: &str) -> Result<&'a [f32], String> {
            table
                .get(i * e..(i + 1) * e)
                .ok_or_else(|| format!("{what} {i} is outside the {}-row table", table.len() / e))
        }
        let mut x = Vec::with_capacity(n * e);
        for (p, (&id, &t)) in ids.iter().zip(types).enumerate() {
            let (w, ps, ty) = (
                row(&self.word, e, id as usize, "token id")?,
                row(&self.pos, e, p, "position")?,
                row(&self.types, e, t as usize, "token type")?,
            );
            x.extend((0..e).map(|c| w[c] + ty[c] + ps[c]));
        }
        x = nn::layer_norm(&x, n, e, &self.emb_ln.0, &self.emb_ln.1, self.eps);
        if let Some(proj) = &self.proj {
            x = proj.apply(&x, n);
        }
        let d = h / self.heads;
        for &l in &self.schedule {
            let layer = &self.layers[l];
            let (q, k, v) = (
                layer.q.apply(&x, n),
                layer.k.apply(&x, n),
                layer.v.apply(&x, n),
            );
            let ctx = nn::attention(&q, &k, &v, n, n, self.heads, d, false);
            let mut a = layer.o.apply(&ctx, n);
            nn::add_in_place(&mut a, &x);
            let a = nn::layer_norm(&a, n, h, &layer.ln1.0, &layer.ln1.1, self.eps);
            let mut i = layer.inter.apply(&a, n);
            if self.gelu_tanh {
                nn::gelu_tanh(&mut i);
            } else {
                nn::gelu(&mut i);
            }
            let mut o = layer.out.apply(&i, n);
            nn::add_in_place(&mut o, &a);
            x = nn::layer_norm(&o, n, h, &layer.ln2.0, &layer.ln2.1, self.eps);
        }
        Ok(x)
    }

    /// The Metal program for `ids`: factorized embeddings, post-LN layers per `schedule` (shared
    /// ALBERT weights are declared once) and `head`, as graph nodes on `gb`; returns the inputs.
    pub(super) fn graph(
        &self,
        gb: &mut Gb,
        ids: &[u32],
        types: &[u32],
        head: &HeadGraph,
    ) -> Result<Vec<Tensor>, String> {
        let (n, e, h) = (ids.len(), self.emb, self.hidden);
        let id_v = gb.g.input("input_ids", vec![n], DType::U32);
        let pos_v = gb.g.input("position_ids", vec![n], DType::U32);
        let ty_v = gb.g.input("token_type_ids", vec![n], DType::U32);
        let mask_v = gb.g.input("attention_mask", vec![n], DType::U32);
        let table = |gb: &mut Gb, name: &str, t: &[f32], idx| {
            let w = gb.weight(&format!("emb.{name}"), vec![t.len() / e, e], || t.to_vec());
            gb.add(&format!("emb.{name}"), Op::Embedding, &[w, idx], vec![n, e])
        };
        let (w, p, t) = (
            table(gb, "word", &self.word, id_v),
            table(gb, "position", &self.pos, pos_v),
            table(gb, "type", &self.types, ty_v),
        );
        let wp = gb.add("emb.add_position", Op::Add, &[w, p], vec![n, e]);
        let wpt = gb.add("emb.add_type", Op::Add, &[wp, t], vec![n, e]);
        let mut x = gb.ln(wpt, "emb.ln", &self.emb_ln, self.eps);
        if let Some(proj) = &self.proj {
            x = gb.lin(x, "emb.proj", proj);
        }
        let attn = Op::Attention {
            heads: self.heads,
            kv_heads: self.heads,
            causal: false,
            window: None,
            soft_cap: None,
        };
        for (step, &l) in self.schedule.iter().enumerate() {
            gb.scope = format!("s{step}.");
            let (p, layer) = (format!("layer.{l}"), &self.layers[l]);
            let q = gb.lin(x, &format!("{p}.q"), &layer.q);
            let k = gb.lin(x, &format!("{p}.k"), &layer.k);
            let v = gb.lin(x, &format!("{p}.v"), &layer.v);
            let c = gb.add(&format!("{p}.attn"), attn.clone(), &[q, k, v], vec![n, h]);
            let o = gb.lin(c, &format!("{p}.o"), &layer.o);
            let r = gb.add(&format!("{p}.res1"), Op::Add, &[o, x], vec![n, h]);
            let a = gb.ln(r, &format!("{p}.ln1"), &layer.ln1, self.eps);
            let i = gb.lin(a, &format!("{p}.inter"), &layer.inter);
            let act = if self.gelu_tanh { Act::GeluTanh } else { Act::Gelu };
            let i = gb.act(&format!("{p}.act"), act, i);
            let o2 = gb.lin(i, &format!("{p}.out"), &layer.out);
            let r2 = gb.add(&format!("{p}.res2"), Op::Add, &[o2, a], vec![n, h]);
            x = gb.ln(r2, &format!("{p}.ln2"), &layer.ln2, self.eps);
        }
        gb.scope.clear();
        let out = head.nodes(gb, x, Some(mask_v))?;
        gb.g.output(out);
        Ok(vec![
            Tensor::U32(ids.to_vec()),
            Tensor::U32((0..n as u32).collect()),
            Tensor::U32(types.to_vec()),
            Tensor::U32(vec![1; n]),
        ])
    }
}
