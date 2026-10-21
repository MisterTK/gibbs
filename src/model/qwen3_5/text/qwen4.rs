//! Qwen4-Exp (`qwen4_exp`, llama.cpp `qwen4exp`): the Qwen3.5-MoE decoder with three additions.
//!
//! * Hyper-connections: `hc` parallel residual streams. Before each token mixer and each MoE a
//!   low-rank gate ([`Mix`]) collapses them to one normed input; the block's output is scattered
//!   back to every stream with its own weight. A final mixer replaces the output norm.
//! * QSA sparse attention: a block-pooled key indexer ([`Indexer`]) picks, per query, the
//!   `budget / ratio` best blocks of `ratio` keys plus the incomplete tail the attention layers see.
//! * PLE: one linear-attention layer adds a hashed n-gram embedding ([`Ple`]) to every stream.
//!
//! Reference: transformers `models/qwen4_exp/modeling_qwen4_exp.py`, llama.cpp
//! `src/models/qwen4exp.cpp`.

use super::moe::Bank;
use super::{add_in_place, Layer, LayerState, TextModel};
use crate::model::par::par_rows;
use crate::model::qwen3_5::{linear, rms_norm, sigmoid, silu, vector_dot, Linear};
use crate::model::simd::vector_fused_scale_add;
use crate::model::weights::Weights;
use serde_json::Value;

/// `x` (`[groups * d]`) RMS-normed per group of `d` with `w` (already `1 + w`).
fn grouped_norm(x: &mut [f32], w: &[f32], d: usize, eps: f32) {
    for (g, wg) in x.chunks_exact_mut(d).zip(w.chunks_exact(d)) {
        rms_norm(g, wg, eps);
    }
}

/// Neox-style RoPE of the first `2 * cos.len()` dims of `x`.
fn rotate(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

/// A hyper-connection mixer: grouped RMSNorm over the streams, a low-rank SiLU gate per channel,
/// and (when it feeds a block) the per-stream injection weights.
pub(super) struct Mix {
    count: usize,
    /// `1 + w`, `[count * hidden]`.
    norm: Vec<f32>,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
}

impl Mix {
    /// The mixer under `p` (`hc_norm`, `input_mix_weight_down/up`, `block_inject_weight`).
    pub(super) fn load(
        ck: &Weights,
        p: &str,
        count: usize,
        hidden: usize,
        inject: bool,
    ) -> Result<Self, String> {
        let w = count * hidden;
        let norm = ck
            .vec_n(&format!("{p}.hc_norm.weight"), w)?
            .into_iter()
            .map(|v| v + 1.0)
            .collect();
        let down = linear(ck, &[&format!("{p}.input_mix_weight_down")], false)?;
        let up = linear(ck, &[&format!("{p}.input_mix_weight_up")], false)?;
        let inject = inject
            .then(|| linear(ck, &[&format!("{p}.block_inject_weight")], false))
            .transpose()?;
        if down.cols != w
            || up.rows != w
            || up.cols != down.rows
            || inject
                .as_ref()
                .is_some_and(|i| i.rows != count || i.cols != w)
        {
            return Err(format!(
                "{p}: hyper-connection shapes disagree with config.json"
            ));
        }
        Ok(Self {
            count,
            norm,
            down,
            up,
            inject,
        })
    }

    /// Streams `x` `[n, count * hidden]` -> the mixed input `[n, hidden]` and, for block mixers,
    /// the injection weights `[n, count]`.
    pub(super) fn mix(&self, x: &[f32], n: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let (c, w) = (self.count, self.norm.len());
        let d = w / c;
        let mut normed = x.to_vec();
        par_rows(&mut normed, w, &|_, row| {
            grouped_norm(row, &self.norm, d, eps)
        });
        let mut gate = self.down.forward(&normed, n);
        gate.iter_mut().for_each(|v| *v = silu(*v / c as f32));
        let gate = self.up.forward(&gate, n);
        let mut mixed = vec![0.0f32; n * d];
        par_rows(&mut mixed, d, &|t, out| {
            for s in 0..c {
                let (g, x) = (&gate[t * w + s * d..][..d], &normed[t * w + s * d..][..d]);
                for ((o, &g), &x) in out.iter_mut().zip(g).zip(x) {
                    *o += sigmoid(g) * x;
                }
            }
            out.iter_mut().for_each(|o| *o /= c as f32);
        });
        let inj = self.inject.as_ref().map_or_else(Vec::new, |l| {
            let mut v = l.forward(&normed, n);
            v.iter_mut().for_each(|x| *x = 2.0 * sigmoid(*x / c as f32));
            v
        });
        (mixed, inj)
    }
}

/// `h[t][s] += y[t] * inj[t][s]` for every stream `s`.
fn combine(h: &mut [f32], y: &[f32], inj: &[f32], count: usize, d: usize) {
    par_rows(h, count * d, &|t, row| {
        for (s, stream) in row.chunks_exact_mut(d).enumerate() {
            vector_fused_scale_add(stream, &y[t * d..][..d], inj[t * count + s]);
        }
    });
}

/// Hyper-connection parts of one decoder layer.
pub(super) struct LayerHc {
    pub(super) attn: Mix,
    pub(super) ffn: Mix,
    pub(super) ple: Option<Ple>,
}

/// The hyper-connection stack of the whole model.
pub(super) struct Hc {
    pub(super) count: usize,
    /// Final mixer (the output norm).
    pub(super) head: Mix,
}

/// QSA indexer: scores blocks of `ratio` keys by their pooled indexer key.
pub(super) struct Indexer {
    /// `q (heads * dim) | k (dim)` projection of the layer input.
    qk: Linear,
    /// `1 + w`, `[dim]`.
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    heads: usize,
    dim: usize,
    ratio: usize,
    /// Blocks a query attends to: `budget / ratio`.
    top_blocks: usize,
}

/// Indexer keys of the sequence: raw per token, pooled (mean, normed, rotated to the block start)
/// per complete block.
#[derive(Default)]
pub(in crate::model::qwen3_5) struct IndexState {
    raw: Vec<f32>,
    pooled: Vec<f32>,
}

impl Indexer {
    /// The indexer of attention layer prefix `p`: `indexer.index_qk_proj` (or `index_q_proj` and
    /// `index_k_proj` as llama.cpp splits it), `q_layernorm`, `k_layernorm`.
    pub(super) fn load(ck: &Weights, p: &str, c: &Value, hidden: usize) -> Result<Self, String> {
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("text_config.{k} missing"))
        };
        let (heads, dim) = (get("indexer_n_heads")?, get("indexer_head_dim")?);
        let (ratio, budget) = (get("indexer_compress_ratio")?, get("indexer_budget")?);
        if c["indexer_kv_heads"].as_u64().is_some_and(|k| k != 1) {
            return Err("QSA indexers with several key heads are not implemented".into());
        }
        if ratio < 2 || budget < ratio || budget % ratio != 0 {
            return Err(format!("QSA needs a compress ratio above 1 that divides the budget, got {ratio} and {budget}"));
        }
        let ix = format!("{p}.indexer");
        let qk = if ck.has(&format!("{ix}.index_qk_proj.weight")) {
            linear(ck, &[&format!("{ix}.index_qk_proj")], false)?
        } else {
            linear(
                ck,
                &[&format!("{ix}.index_q_proj"), &format!("{ix}.index_k_proj")],
                false,
            )?
        };
        let norm = |n: &str| -> Result<Vec<f32>, String> {
            Ok(ck
                .vec_n(&format!("{ix}.{n}.weight"), dim)?
                .into_iter()
                .map(|v| v + 1.0)
                .collect())
        };
        if qk.cols != hidden || qk.rows != (heads + 1) * dim {
            return Err(format!("{ix}: projection shape disagrees with config.json"));
        }
        Ok(Self {
            qk,
            q_norm: norm("q_layernorm")?,
            k_norm: norm("k_layernorm")?,
            heads,
            dim,
            ratio,
            top_blocks: budget / ratio,
        })
    }

    /// Per row of `x` `[n, hidden]` (positions `start..`), the keys its attention may read:
    /// `None` while every visible key is selected anyway (at most `top_blocks` complete blocks),
    /// else the best blocks' keys plus the incomplete tail, ascending. Appends the rows' keys to
    /// `st`. `rope` is `(cos, sin)` per row, `rope_at(p)` that of position `p`.
    pub(super) fn select(
        &self,
        x: &[f32],
        n: usize,
        start: usize,
        (cos, sin): &(Vec<f32>, Vec<f32>),
        rope_at: &dyn Fn(usize) -> (Vec<f32>, Vec<f32>),
        st: &mut IndexState,
        eps: f32,
    ) -> Vec<Option<Vec<u32>>> {
        let (dim, ratio, half) = (self.dim, self.ratio, cos.len() / n);
        let width = self.qk.rows;
        let qk = self.qk.forward(x, n);
        let mut q = vec![0.0f32; n * self.heads * dim];
        for t in 0..n {
            let row = &qk[t * width..][..width];
            for h in 0..self.heads {
                let qh = &mut q[(t * self.heads + h) * dim..][..dim];
                qh.copy_from_slice(&row[h * dim..][..dim]);
                rms_norm(qh, &self.q_norm, eps);
                rotate(qh, &cos[t * half..][..half], &sin[t * half..][..half]);
            }
            st.raw.extend_from_slice(&row[self.heads * dim..]);
        }
        // Blocks this chunk completed: mean of their raw keys, normed, rotated to their first key.
        let done = st.pooled.len() / dim;
        for b in done..st.raw.len() / dim / ratio {
            let mut k = vec![0.0f32; dim];
            for key in st.raw[b * ratio * dim..][..ratio * dim].chunks_exact(dim) {
                k.iter_mut().zip(key).for_each(|(a, &v)| *a += v);
            }
            k.iter_mut().for_each(|a| *a /= ratio as f32);
            rms_norm(&mut k, &self.k_norm, eps);
            let (c, s) = rope_at(b * ratio);
            rotate(&mut k, &c, &s);
            st.pooled.extend_from_slice(&k);
        }
        let scale = 1.0 / (dim as f32).sqrt();
        (0..n)
            .map(|t| {
                let visible = start + t + 1;
                let blocks = visible / ratio;
                if blocks <= self.top_blocks {
                    return None;
                }
                let qt = &q[t * self.heads * dim..][..self.heads * dim];
                let score: Vec<(f32, u32)> = (0..blocks)
                    .map(|b| {
                        let kb = &st.pooled[b * dim..][..dim];
                        let s: f32 = qt
                            .chunks_exact(dim)
                            .map(|qh| vector_dot(qh, kb).max(0.0))
                            .sum();
                        (s * scale, b as u32)
                    })
                    .collect();
                Some(block_keys(score, visible, ratio, self.top_blocks))
            })
            .collect()
    }
}

/// The keys a row with `visible` keys reads, given `(score, block)` of its complete blocks (more
/// than `top_blocks` of them): the `top_blocks` best blocks (score descending, then block
/// ascending) in ascending order, each expanded to its `ratio` keys, then the incomplete tail.
fn block_keys(
    mut score: Vec<(f32, u32)>,
    visible: usize,
    ratio: usize,
    top_blocks: usize,
) -> Vec<u32> {
    let blocks = score.len();
    score.select_nth_unstable_by(top_blocks - 1, |a, b| {
        b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
    });
    score.truncate(top_blocks);
    score.sort_unstable_by_key(|s| s.1);
    let mut keys: Vec<u32> = score
        .iter()
        .flat_map(|&(_, b)| (b as usize * ratio..(b as usize + 1) * ratio).map(|j| j as u32))
        .collect();
    keys.extend((blocks * ratio..visible).map(|j| j as u32));
    keys
}

/// PLE: hashed n-gram embeddings of the token and its predecessors, gated per stream by the
/// streams' own state, plus a dilated depthwise conv of the gated value.
pub(super) struct Ple {
    /// Rows of the n-gram table (`[parts, rows, head_dim]`).
    table: Bank,
    mult: Vec<u64>,
    offsets: Vec<u64>,
    vocab: Vec<u64>,
    ngram: usize,
    per_gram: usize,
    eos: u32,
    kernel: usize,
    key: Linear,
    value: Linear,
    /// `1 + w`, `[count * hidden]`.
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    /// Depthwise taps `[count * hidden, kernel]`.
    conv: Vec<f32>,
}

/// PLE history of one sequence: the tokens of the next forward (set by the caller), the last
/// `ngram - 1` tokens and the conv inputs of the last `(kernel - 1) * ngram` positions.
pub(in crate::model::qwen3_5) struct PleState {
    pub(super) pending: Vec<u32>,
    prev: Vec<u32>,
    conv: Vec<f32>,
}

impl Ple {
    /// The PLE module of layer prefix `l`. `hash` is `(multipliers, offsets, vocab sizes)` when
    /// the file carries them as metadata (GGUF), else they are read from the checkpoint's buffers.
    pub(super) fn load(
        ck: &Weights,
        l: &str,
        c: &Value,
        count: usize,
        hidden: usize,
        hash: Option<(Vec<u64>, Vec<u64>, Vec<u64>)>,
    ) -> Result<Self, String> {
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("text_config.{k} missing"))
        };
        let (ngram, per_gram, kernel) = (
            get("ngram_size")?,
            get("heads_per_ngram")?,
            get("ple_conv_kernel_size")?,
        );
        let eos = match &c["eos_token_id"] {
            Value::Array(a) => a.first().and_then(Value::as_u64),
            v => v.as_u64(),
        }
        .ok_or("text_config.eos_token_id missing (PLE n-grams reset on it)")?
            as u32;
        let pe = format!("{l}.ple.ple_embedding");
        let (mult, offsets, vocab) = match hash {
            Some(h) => h,
            None => (
                ck.u64s(&format!("{pe}.layer_multipliers"))?,
                ck.u64s(&format!("{pe}.ngram_heads_offsets"))?,
                ck.u64s(&format!("{pe}.ngram_heads_vocab_sizes"))?,
            ),
        };
        let heads = (ngram - 1) * per_gram;
        if ngram < 2
            || mult.len() < ngram
            || offsets.len() < heads
            || vocab.len() < heads
            || vocab.contains(&0)
        {
            return Err(format!("{pe}: hash constants disagree with the config"));
        }
        let table = if ck.has(&format!("{pe}.table")) {
            Bank::load(ck, &format!("{pe}.table"))?
        } else {
            let parts = get("split_ngram_parts")?;
            Bank::load_mats(
                ck,
                (0..parts).map(|i| format!("{pe}.ngram_embedding.shard_{i}.weight")),
            )?
        };
        let p = format!("{l}.ple");
        let norm = |n: &str| -> Result<Vec<f32>, String> {
            Ok(ck
                .vec_n(&format!("{p}.{n}.weight"), count * hidden)?
                .into_iter()
                .map(|v| v + 1.0)
                .collect())
        };
        let me = Self {
            table,
            mult,
            offsets,
            vocab,
            ngram,
            per_gram,
            eos,
            kernel,
            key: linear(ck, &[&format!("{p}.key_proj")], false)?,
            value: linear(ck, &[&format!("{p}.value_proj")], false)?,
            norm_key: norm("norm_key")?,
            norm_query: norm("norm_query")?,
            norm_conv: norm("norm_conv")?,
            conv: ck.vec_n(&format!("{p}.conv1d.weight"), count * hidden * kernel)?,
        };
        let rows = me.table.experts * me.table.rows;
        let emb = heads * me.table.cols;
        if me.key.cols != emb
            || me.value.cols != emb
            || me.key.rows != count * hidden
            || me.value.rows != hidden
            || (0..heads).any(|h| me.offsets[h] + me.vocab[h] > rows as u64)
        {
            return Err(format!("{p}: shapes disagree with the config"));
        }
        Ok(me)
    }

    pub(super) fn new_state(&self, width: usize) -> PleState {
        PleState {
            pending: Vec::new(),
            prev: vec![self.eos; self.ngram - 1],
            conv: vec![0.0; (self.kernel - 1) * self.ngram * width],
        }
    }

    /// The addend `[n, count * hidden]` for streams `h` and `st.pending` (this chunk's tokens).
    fn forward(&self, h: &[f32], st: &mut PleState, n: usize, d: usize, eps: f32) -> Vec<f32> {
        let tokens = std::mem::take(&mut st.pending);
        assert_eq!(
            tokens.len(),
            n,
            "PLE tokens were not announced for this forward"
        );
        let w = self.norm_key.len();
        let (hd, heads) = (self.table.cols, (self.ngram - 1) * self.per_gram);
        // Each token with its predecessors; an EOS among them cuts everything at or before it.
        let chain: Vec<u32> = st.prev.iter().chain(&tokens).copied().collect();
        let back = self.ngram - 1;
        let mut emb = vec![0.0f32; n * heads * hd];
        for t in 0..n {
            let mut ctx = vec![chain[back + t]; self.ngram];
            let mut cut = false;
            for s in 1..self.ngram {
                let p = chain[back + t - s];
                cut |= p == self.eos;
                ctx[s] = if cut { self.eos } else { p };
            }
            for g in 2..=self.ngram {
                let mut mixed = ctx[0] as u64 * self.mult[0];
                for j in 1..g {
                    mixed ^= ctx[j] as u64 * self.mult[j];
                }
                for k in 0..self.per_gram {
                    let head = (g - 2) * self.per_gram + k;
                    let row = (mixed % self.vocab[head] + self.offsets[head]) as usize;
                    let dst = &mut emb[(t * heads + head) * hd..][..hd];
                    self.table
                        .row(row / self.table.rows, row % self.table.rows, dst);
                }
            }
        }
        st.prev = chain[chain.len() - back..].to_vec();
        let mut key = self.key.forward(&emb, n);
        let value = self.value.forward(&emb, n);
        let mut query = h.to_vec();
        par_rows(&mut key, w, &|_, r| grouped_norm(r, &self.norm_key, d, eps));
        par_rows(&mut query, w, &|_, r| {
            grouped_norm(r, &self.norm_query, d, eps)
        });
        // Per stream: sigmoid(signed sqrt of the scaled key . query) scales the shared value.
        let mut gated = vec![0.0f32; n * w];
        par_rows(&mut gated, w, &|t, row| {
            for (s, g) in row.chunks_exact_mut(d).enumerate() {
                let dot = vector_dot(&key[t * w + s * d..][..d], &query[t * w + s * d..][..d])
                    / (d as f32).sqrt();
                let m = dot.abs().max(1e-6).sqrt();
                let gate = sigmoid(if dot > 0.0 {
                    m
                } else if dot < 0.0 {
                    -m
                } else {
                    0.0
                });
                for (o, &v) in g.iter_mut().zip(&value[t * d..][..d]) {
                    *o = gate * v;
                }
            }
        });
        let mut normed = gated.clone();
        par_rows(&mut normed, w, &|_, r| {
            grouped_norm(r, &self.norm_conv, d, eps)
        });
        // Causal depthwise conv dilated by `ngram`: tap k reads (kernel - 1 - k) * ngram steps back.
        let hist = (self.kernel - 1) * self.ngram;
        let seq: Vec<f32> = st.conv.iter().chain(&normed).copied().collect();
        let mut out = gated;
        par_rows(&mut out, w, &|t, row| {
            for (c, o) in row.iter_mut().enumerate() {
                let acc: f32 = (0..self.kernel)
                    .map(|k| {
                        self.conv[c * self.kernel + k]
                            * seq[(hist + t - (self.kernel - 1 - k) * self.ngram) * w + c]
                    })
                    .sum();
                *o += silu(acc);
            }
        });
        st.conv = seq[seq.len() - hist * w..].to_vec();
        out
    }
}

impl TextModel {
    /// Tokens of the next forward, for the PLE layer's n-gram hash (a no-op without one).
    pub(in crate::model::qwen3_5) fn set_ple_tokens(
        &self,
        state: &mut [LayerState],
        tokens: &[u32],
    ) {
        for st in state {
            if let LayerState::Delta { ple: Some(p), .. } = st {
                p.pending = tokens.to_vec();
            }
        }
    }

    /// Embedding rows `[n, hidden]` -> the `hc` identical residual streams `[n, hc * hidden]`.
    pub(super) fn expand_streams(&self, h: &[f32], count: usize) -> Vec<f32> {
        let d = self.hidden;
        h.chunks_exact(d)
            .flat_map(|r| (0..count).flat_map(move |_| r.iter().copied()))
            .collect()
    }

    /// One hyper-connected decoder layer over streams `h` `[n, count * hidden]` in place.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_forward_hc(
        &self,
        layer: &Layer,
        lhc: &LayerHc,
        st: &mut LayerState,
        h: &mut [f32],
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        start: usize,
        key_cap: usize,
        ckpt: bool,
    ) {
        let (d, count) = (self.hidden, lhc.attn.count);
        if let (Some(ple), LayerState::Delta { ple: Some(ps), .. }) = (&lhc.ple, &mut *st) {
            let add = ple.forward(h, ps, n, d, self.eps);
            add_in_place(h, &add);
        }
        let (x, inj) = lhc.attn.mix(h, n, self.eps);
        let y = self.token_mixer(layer, st, &x, n, rope, start, key_cap, ckpt);
        combine(h, &y, &inj, count, d);
        let (x, inj) = lhc.ffn.mix(h, n, self.eps);
        let y = self.ffn_forward(&layer.ffn, &x, n);
        combine(h, &y, &inj, count, d);
    }
}

#[cfg(test)]
mod tests {
    use super::block_keys;
    use crate::graph::{cpu, DType, Graph, Op, Tensor, Weights};

    /// `Op::QsaSelect` / `Op::QsaBias` reproduce the indexer's host key choice (`select`: every visible
    /// key while at most `top_blocks` blocks are complete, else `block_keys`) on random scores with
    /// ties, on the CPU executor (Metal equals it bit for bit: `qsa_select_and_bias_match_cpu`).
    #[test]
    fn graph_qsa_matches_host_selection() {
        let mut s = 0x9e37_79b9u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let (n, b) = (11usize, 40usize);
        for (ratio, top) in [(1usize, 1usize), (4, 2), (3, 5), (16, 3), (2, 39)] {
            // Few levels so equal scores happen; positions keep every row within the B blocks.
            let score: Vec<f32> = (0..n * b).map(|_| (next() % 7) as f32 * 0.5 - 1.0).collect();
            let pos: Vec<u32> = (0..n).map(|_| next() % (b * ratio) as u32).collect();
            let k = top * ratio + ratio - 1;
            let (mut ids, mut bias) = (vec![0u32; n * k], vec![f32::NEG_INFINITY; n * k]);
            for t in 0..n {
                let visible = pos[t] as usize + 1;
                let blocks = visible / ratio;
                let keys: Vec<u32> = if blocks <= top {
                    (0..visible as u32).collect()
                } else {
                    let row = (0..blocks).map(|j| (score[t * b + j], j as u32)).collect();
                    block_keys(row, visible, ratio, top)
                };
                ids[t * k..][..keys.len()].copy_from_slice(&keys);
                bias[t * k..][..keys.len()].fill(0.0);
            }
            for select in [true, false] {
                let mut g = Graph::new();
                let sc = g.input("score", vec![n, b], DType::F32);
                let p = g.input("pos", vec![n], DType::U32);
                let op = if select {
                    Op::QsaSelect { ratio, top_blocks: top }
                } else {
                    Op::QsaBias { ratio, top_blocks: top }
                };
                let y = g.push("qsa", op, &[sc, p]).unwrap();
                g.output(y);
                let inputs = vec![Tensor::F32(score.clone()), Tensor::U32(pos.clone())];
                let got = cpu::run_stateless(&g, &Weights::new(), inputs).unwrap();
                let what = format!("ratio {ratio} top {top} select {select}");
                // Bits: ids as u32, the bias as f32 (0 / -inf compare exactly).
                let bits = |t: &Tensor| -> Vec<u32> {
                    match t {
                        Tensor::U32(v) => v.clone(),
                        Tensor::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
                        other => panic!("unexpected {other:?}"),
                    }
                };
                let want: Vec<u32> = if select {
                    ids.clone()
                } else {
                    bias.iter().map(|x| x.to_bits()).collect()
                };
                assert_eq!(bits(&got[0]), want, "CPU {what}");
            }
        }
    }
}
