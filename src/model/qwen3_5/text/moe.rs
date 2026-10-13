//! Sparse mixture-of-experts feed-forward of Qwen3.5-MoE (`qwen3_5_moe`, llama.cpp `qwen35moe`):
//! a softmax router picks the top-k of E SwiGLU experts per token (weights renormalized over the
//! picked ones), a shared expert gated by a sigmoid of its own logit is added.
//!
//! Expert banks stay as stored and are read per routed row, so resident memory tracks the experts
//! the prompts actually reach: bf16 safetensors tensors are borrowed from the mapping; GGUF blocks
//! are dequantized one row at a time out of the file mapping ([`Packed`]).
//!
//! Reference: transformers `models/qwen3_5_moe/modeling_qwen3_5_moe.py` (`Qwen3_5MoeSparseMoeBlock`),
//! llama.cpp `build_moe_ffn` (softmax gating, `norm_w`).

use super::{sigmoid, silu, vector_dot};
use crate::graph::{DType, Graph, Op, ReduceKind, Tensor, UnaryKind, ValueId};
use crate::hardware::QuantFormat;
use crate::model::layers::gguf_block_codec;
use crate::model::par::{dot_bf16, par_for, par_rows, OutPtr};
use crate::model::qwen3_5::{linear, Linear};
use crate::model::simd::{vector_fused_scale_add, vector_swiglu};
use crate::model::weights::{Mat, Packed, Weights};

/// Weight rows one task decodes and multiplies against every token routed to its expert.
const ROW_BLOCK: usize = 16;

enum Src {
    /// bf16 matrices borrowed from the mapped checkpoint, `per` experts of `stride * cols` each:
    /// one fused `[experts, rows, cols]` tensor (`per = experts`) or one `[rows, cols]` matrix per
    /// expert (`per = 1`, transformers' unfused `experts.N.gate_proj.weight` layout).
    Bf16(Vec<Mat>, usize),
    Packed(Packed),
}

/// `[experts, rows, cols]` expert matrices; a view selects `rows` of the `stride` stored rows of
/// every expert from `row0` (gate and up are the halves of a fused `gate_up` tensor).
pub(super) struct Bank {
    src: Src,
    stride: usize,
    row0: usize,
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
}

thread_local! {
    static ROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Bank {
    /// The 3-D tensor `name`: a bf16-convertible safetensors tensor or a packed GGUF bank.
    pub(super) fn load(ck: &Weights, name: &str) -> Result<Self, String> {
        let (src, shape) = match ck.packed_gguf(name) {
            Some(p) => {
                let [experts, rows, cols] = p.shape[..] else {
                    return Err(format!(
                        "{name}: expert bank is {:?}, expected [experts, rows, cols]",
                        p.shape
                    ));
                };
                let (n, bytes) = match p.format {
                    QuantFormat::BF16 | QuantFormat::FP16 => (1, 2),
                    QuantFormat::FP32 => (1, 4),
                    f => {
                        let (n, bytes, _) = gguf_block_codec(f).ok_or_else(|| {
                            format!("{name}: {f:?} expert weights are not supported")
                        })?;
                        (n, bytes)
                    }
                };
                if cols % n != 0 {
                    return Err(format!(
                        "{name}: {cols} columns are not whole {:?} blocks",
                        p.format
                    ));
                }
                if p.off + experts * rows * (cols / n) * bytes > p.map.len() {
                    return Err(format!(
                        "{name}: the file is truncated (expert bank runs past its end)"
                    ));
                }
                (Src::Packed(p.clone()), p.shape.clone())
            }
            None => {
                let shape = ck.shape(name)?;
                (
                    Src::Bf16(vec![ck.mat(name)?], shape.first().copied().unwrap_or(1)),
                    shape,
                )
            }
        };
        let [experts, rows, cols] = shape[..] else {
            return Err(format!(
                "{name}: expert bank is {shape:?}, expected [experts, rows, cols]"
            ));
        };
        Ok(Self {
            src,
            stride: rows,
            row0: 0,
            rows,
            cols,
            experts,
        })
    }

    /// The `experts` matrices `{prefix}.{e}.{part}.weight`.
    pub(super) fn load_each(
        ck: &Weights,
        prefix: &str,
        part: &str,
        experts: usize,
    ) -> Result<Self, String> {
        Self::load_mats(
            ck,
            (0..experts).map(|e| format!("{prefix}.{e}.{part}.weight")),
        )
    }

    /// One `[rows, cols]` bf16 matrix per name, as the "experts" of a bank.
    pub(super) fn load_mats(
        ck: &Weights,
        names: impl Iterator<Item = String>,
    ) -> Result<Self, String> {
        let mats = names.map(|n| ck.mat(&n)).collect::<Result<Vec<_>, _>>()?;
        let Some((rows, cols)) = mats.first().map(|m| (m.rows, m.cols)) else {
            return Err("expert bank without matrices".into());
        };
        if mats.iter().any(|m| (m.rows, m.cols) != (rows, cols)) {
            return Err("expert matrices differ in shape".into());
        }
        let experts = mats.len();
        Ok(Self {
            src: Src::Bf16(mats, 1),
            stride: rows,
            row0: 0,
            rows,
            cols,
            experts,
        })
    }

    /// Row `r` of expert `e` as f32 into `out` (`cols` long).
    pub(super) fn row(&self, e: usize, r: usize, out: &mut [f32]) {
        match &self.src {
            Src::Bf16(mats, per) => {
                let row = e % per * self.stride + self.row0 + r;
                for (o, &w) in out
                    .iter_mut()
                    .zip(&mats[e / per].data()[row * self.cols..][..self.cols])
                {
                    *o = crate::model::bf16_to_f32(w);
                }
            }
            Src::Packed(p) => decode_row(p, e * self.stride + self.row0 + r, out),
        }
    }

    /// The bank as the `[experts * rows, cols]` weight of an `Op::MoeLinear`. A whole GGUF tensor is
    /// shared with the file mapping; a view of one (the halves of a fused gate/up tensor) or a bf16
    /// checkpoint is gathered into an owned copy.
    // ponytail: bf16 and fused-view banks are copied once at graph build; only GGUF banks stay mapped.
    fn graph_weight(&self) -> (DType, Tensor) {
        let total = self.experts * self.rows;
        match &self.src {
            Src::Packed(p) => {
                let dtype = match p.format {
                    QuantFormat::FP32 => DType::F32,
                    QuantFormat::FP16 => DType::F16,
                    QuantFormat::BF16 => DType::BF16,
                    f => DType::Quant(f),
                };
                let whole = self.row0 == 0 && self.stride == self.rows;
                let p = if whole {
                    p.clone()
                } else {
                    let row_bytes = p.format.tensor_bytes(self.cols);
                    let mut buf = Vec::with_capacity(total * row_bytes);
                    for e in 0..self.experts {
                        let at = p.off + (e * self.stride + self.row0) * row_bytes;
                        buf.extend_from_slice(&p.map[at..at + self.rows * row_bytes]);
                    }
                    Packed {
                        map: std::sync::Arc::new(crate::model::loaders::gguf::GgufData::Buf(buf)),
                        off: 0,
                        format: p.format,
                        shape: vec![self.experts, self.rows, self.cols],
                    }
                };
                (dtype, Tensor::Mat(Mat::from_packed(p, total, self.cols)))
            }
            Src::Bf16(..) => {
                let mut v = vec![0u16; total * self.cols];
                let mut row = vec![0.0f32; self.cols];
                for e in 0..self.experts {
                    for r in 0..self.rows {
                        self.row(e, r, &mut row);
                        let at = (e * self.rows + r) * self.cols;
                        for (o, &x) in v[at..at + self.cols].iter_mut().zip(&row) {
                            *o = crate::model::f32_to_bf16(x);
                        }
                    }
                }
                (DType::BF16, Tensor::BF16(v))
            }
        }
    }

    /// Rows `row0..row0 + rows` of every expert.
    pub(super) fn view(&self, row0: usize, rows: usize) -> Self {
        let src = match &self.src {
            Src::Bf16(m, per) => Src::Bf16(m.clone(), *per),
            Src::Packed(p) => Src::Packed(p.clone()),
        };
        Self {
            src,
            row0: self.row0 + row0,
            rows,
            ..*self
        }
    }

    /// `out[i] = row r of expert e . xs[i]`.
    fn dots(&self, e: usize, r: usize, xs: &[&[f32]], out: &mut [f32]) {
        match &self.src {
            Src::Bf16(mats, per) => {
                let row = e % per * self.stride + self.row0 + r;
                let w = &mats[e / per].data()[row * self.cols..][..self.cols];
                for (o, x) in out.iter_mut().zip(xs) {
                    *o = dot_bf16(w, x);
                }
            }
            Src::Packed(p) => ROW.with_borrow_mut(|buf| {
                buf.resize(self.cols, 0.0);
                decode_row(p, e * self.stride + self.row0 + r, buf);
                for (o, x) in out.iter_mut().zip(xs) {
                    *o = vector_dot(buf, x);
                }
            }),
        }
    }
}

/// Row `row` (of `buf.len()` weights) of packed bank `p` as f32.
fn decode_row(p: &Packed, row: usize, buf: &mut [f32]) {
    let cols = buf.len();
    let (elem, bytes) = match gguf_block_codec(p.format) {
        Some((n, b, _)) => (n, b),
        None => match p.format {
            QuantFormat::FP32 => (1, 4),
            _ => (1, 2),
        },
    };
    let len = cols / elem * bytes;
    let raw = &p.map[p.off + row * len..][..len];
    match (gguf_block_codec(p.format), p.format) {
        (Some((n, b, f)), _) => {
            for (blk, y) in raw.chunks_exact(b).zip(buf.chunks_exact_mut(n)) {
                f(blk, y);
            }
        }
        (None, QuantFormat::FP32) => {
            for (y, v) in buf.iter_mut().zip(raw.chunks_exact(4)) {
                *y = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
            }
        }
        (None, QuantFormat::FP16) => {
            for (y, v) in buf.iter_mut().zip(raw.chunks_exact(2)) {
                *y = crate::model::f16_to_f32(u16::from_le_bytes([v[0], v[1]]));
            }
        }
        (None, _) => {
            for (y, v) in buf.iter_mut().zip(raw.chunks_exact(2)) {
                *y = crate::model::bf16_to_f32(u16::from_le_bytes([v[0], v[1]]));
            }
        }
    }
}

struct Shared {
    /// `gate_proj | up_proj`.
    gate_up: Linear,
    down: Linear,
    /// `[1, hidden]`: the sigmoid gate of the shared expert's output.
    gate: Linear,
}

pub(crate) struct Moe {
    pub(crate) router: Linear,
    gate: Bank,
    up: Bank,
    down: Bank,
    top_k: usize,
    /// Renormalize the picked experts' probabilities to sum to 1 (`norm_topk_prob`).
    norm: bool,
    shared: Option<Shared>,
}

impl Moe {
    #[allow(dead_code)]
    pub(super) fn num_experts(&self) -> usize {
        self.router.rows
    }

    #[allow(dead_code)]
    pub(super) fn top_k(&self) -> usize {
        self.top_k
    }
    /// The block under `p` (`...layers.N.mlp`): `gate.weight`, `experts.{gate_up_proj | gate_proj,
    /// up_proj}` / `experts.down_proj`, `shared_expert.*` and `shared_expert_gate.weight`.
    pub(super) fn load(
        ck: &Weights,
        p: &str,
        hidden: usize,
        top_k: usize,
        norm: bool,
    ) -> Result<Self, String> {
        let router = linear(ck, &[&format!("{p}.gate")], false)?;
        let (fused, ex) = (format!("{p}.experts.gate_up_proj"), format!("{p}.experts"));
        let (gate, up, down) = if ck.has(&fused) {
            let b = Bank::load(ck, &fused)?;
            let i = b.rows / 2;
            (
                b.view(0, i),
                b.view(i, i),
                Bank::load(ck, &format!("{ex}.down_proj"))?,
            )
        } else if ck.has(&format!("{ex}.gate_proj")) {
            (
                Bank::load(ck, &format!("{ex}.gate_proj"))?,
                Bank::load(ck, &format!("{ex}.up_proj"))?,
                Bank::load(ck, &format!("{ex}.down_proj"))?,
            )
        } else {
            let e = router.rows;
            (
                Bank::load_each(ck, &ex, "gate_proj", e)?,
                Bank::load_each(ck, &ex, "up_proj", e)?,
                Bank::load_each(ck, &ex, "down_proj", e)?,
            )
        };
        let sh = |n: &str| format!("{p}.shared_expert.{n}");
        let shared = if ck.has(&format!("{}.weight", sh("gate_proj"))) {
            Some(Shared {
                gate_up: linear(ck, &[&sh("gate_proj"), &sh("up_proj")], false)?,
                down: linear(ck, &[&sh("down_proj")], false)?,
                gate: linear(ck, &[&format!("{p}.shared_expert_gate")], false)?,
            })
        } else {
            None
        };
        let m = Self {
            router,
            gate,
            up,
            down,
            top_k,
            norm,
            shared,
        };
        let (e, i) = (m.router.rows, m.gate.rows);
        let bad = m.router.cols != hidden
            || top_k == 0
            || top_k > e
            || [&m.gate, &m.up, &m.down].iter().any(|b| b.experts != e)
            || m.up.rows != i
            || m.gate.cols != hidden
            || m.up.cols != hidden
            || m.down.rows != hidden
            || m.down.cols != i
            || m.shared.as_ref().is_some_and(|s| {
                s.gate_up.cols != hidden
                    || s.down.rows != hidden
                    || s.gate_up.rows != 2 * s.down.cols
                    || s.gate.rows != 1
                    || s.gate.cols != hidden
            });
        if bad {
            return Err(format!("{p}: expert shapes disagree with config.json"));
        }
        Ok(m)
    }

    /// Appends this block to the decode graph: `x` `[n, hidden]` (already normed) -> `[n, hidden]`.
    /// Router, top-k, expert GEMVs and the routing-weighted sum mirror [`Self::forward`]:
    /// softmax over the top-k logits when the picked weights are renormalized, else the picks of the
    /// softmax over all experts; the shared expert's output is scaled by a sigmoid of its own gate.
    pub(super) fn add_to_graph(
        &self,
        g: &mut Graph,
        w: &mut crate::graph::Weights,
        p: &str,
        x: ValueId,
        n: usize,
    ) -> Result<ValueId, String> {
        let f = DType::F32;
        let (e, k, ff, d) = (self.router.rows, self.top_k, self.gate.rows, self.router.cols);
        let part = |g: &mut Graph, w: &mut crate::graph::Weights, name: String, m: &Mat| -> ValueId {
            let dtype = match m.packed().map(|p| p.format) {
                None => DType::BF16,
                Some(QuantFormat::FP32) => DType::F32,
                Some(QuantFormat::FP16) => DType::F16,
                Some(QuantFormat::BF16) => DType::BF16,
                Some(f) => DType::Quant(f),
            };
            w.insert(&name, m.to_graph_tensor());
            g.weight(name, vec![m.rows, m.cols], dtype)
        };
        let router = part(g, w, format!("{p}.gate.weight"), &self.router.parts[0]);
        let logits = g.add(format!("{p}.router"), Op::Linear, &[x, router], vec![n, e], f);
        let ids = g.add(format!("{p}.topk"), Op::TopK { k }, &[logits], vec![n, k], DType::U32);
        let weights = if self.norm {
            let picked = g.add(format!("{p}.picked"), Op::GatherCols, &[logits, ids], vec![n, k], f);
            g.add(format!("{p}.route_w"), Op::Softmax { axis: -1 }, &[picked], vec![n, k], f)
        } else {
            let probs = g.add(format!("{p}.router_probs"), Op::Softmax { axis: -1 }, &[logits], vec![n, e], f);
            g.add(format!("{p}.route_w"), Op::GatherCols, &[probs, ids], vec![n, k], f)
        };
        let bank = |g: &mut Graph, w: &mut crate::graph::Weights, name: &str, b: &Bank| -> ValueId {
            let (dtype, t) = b.graph_weight();
            let name = format!("{p}.experts.{name}.weight");
            w.insert(&name, t);
            g.weight(name, vec![b.experts * b.rows, b.cols], dtype)
        };
        let (gate_w, up_w, down_w) = (
            bank(g, w, "gate_proj", &self.gate),
            bank(g, w, "up_proj", &self.up),
            bank(g, w, "down_proj", &self.down),
        );
        let moe = |g: &mut Graph, name: &str, input: ValueId, bank: ValueId, out: usize| {
            g.add(
                format!("{p}.{name}"),
                Op::MoeLinear { experts: e, top_k: k },
                &[input, bank, ids],
                vec![n * k, out],
                f,
            )
        };
        let gate = moe(g, "expert_gate", x, gate_w, ff);
        let up = moe(g, "expert_up", x, up_w, ff);
        let act = g.add(format!("{p}.expert_swiglu"), Op::SwiGlu, &[gate, up], vec![n * k, ff], f);
        let down = moe(g, "expert_down", act, down_w, d);
        let w_col = g.add(
            format!("{p}.route_w_col"),
            Op::Reshape { shape: Some(vec![n * k, 1]) },
            &[weights],
            vec![n * k, 1],
            f,
        );
        let weighted = g.add(format!("{p}.expert_weighted"), Op::Mul, &[down, w_col], vec![n * k, d], f);
        let grouped = g.add(
            format!("{p}.expert_slots"),
            Op::Reshape { shape: Some(vec![n, k, d]) },
            &[weighted],
            vec![n, k, d],
            f,
        );
        let routed = g.add(
            format!("{p}.routed"),
            Op::Reduce { kind: ReduceKind::Sum, axes: vec![1], keep_dims: false },
            &[grouped],
            vec![n, d],
            f,
        );
        let Some(sh) = &self.shared else {
            return Ok(routed);
        };
        let si = sh.down.cols;
        let gate_p = part(g, w, format!("{p}.shared_expert.gate_proj.weight"), &sh.gate_up.parts[0]);
        let up_p = part(g, w, format!("{p}.shared_expert.up_proj.weight"), &sh.gate_up.parts[1]);
        let down_p = part(g, w, format!("{p}.shared_expert.down_proj.weight"), &sh.down.parts[0]);
        let gate_g = part(g, w, format!("{p}.shared_expert_gate.weight"), &sh.gate.parts[0]);
        let lin = |g: &mut Graph, name: &str, a: ValueId, b: ValueId, out: usize| {
            g.add(format!("{p}.{name}"), Op::Linear, &[a, b], vec![n, out], f)
        };
        let (sg, su) = (lin(g, "shared_gate", x, gate_p, si), lin(g, "shared_up", x, up_p, si));
        let sact = g.add(format!("{p}.shared_swiglu"), Op::SwiGlu, &[sg, su], vec![n, si], f);
        let sdown = lin(g, "shared_down", sact, down_p, d);
        let gate_logit = lin(g, "shared_gate_logit", x, gate_g, 1);
        let gate_act = g.add(
            format!("{p}.shared_gate_act"),
            Op::Unary { kind: UnaryKind::Sigmoid },
            &[gate_logit],
            vec![n, 1],
            f,
        );
        let shared = g.add(format!("{p}.shared_out"), Op::Mul, &[sdown, gate_act], vec![n, d], f);
        Ok(g.add(format!("{p}.moe_out"), Op::Add, &[routed, shared], vec![n, d], f))
    }

    pub(super) fn summary(&self) -> String {
        format!("{} experts (top-{})", self.router.rows, self.top_k)
    }

    /// `x` `[n, hidden]` (already normed) -> `[n, hidden]`.
    pub(super) fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let (d, e, k, inter) = (
            self.router.cols,
            self.router.rows,
            self.top_k,
            self.gate.rows,
        );
        // Router: softmax over all experts, top-k by probability (lowest index on ties), weights
        // renormalized over the picked experts.
        let logits = self.router.forward(x, n);
        let mut picks = vec![(0u32, 0.0f32); n * k];
        for (i, row) in logits.chunks_exact(e).enumerate() {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut order: Vec<(f32, u32)> = row
                .iter()
                .enumerate()
                .map(|(j, &l)| ((l - max).exp(), j as u32))
                .collect();
            let all: f32 = order.iter().map(|o| o.0).sum();
            order.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            order.truncate(k);
            order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let sum = match self.norm {
                true => order.iter().map(|o| o.0).sum::<f32>().max(6.103_515_6e-5),
                false => all,
            };
            for (dst, (p, j)) in picks[i * k..][..k].iter_mut().zip(order) {
                *dst = (j, p / sum);
            }
        }
        // Assignments `a = token * k + slot`, grouped by expert so each weight row is decoded once
        // for all the tokens routed to it.
        let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); e];
        for (a, &(j, _)) in picks.iter().enumerate() {
            by_expert[j as usize].push(a);
        }
        let active: Vec<(usize, &[usize])> = by_expert
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(j, v)| (j, v.as_slice()))
            .collect();

        let mut act = vec![0.0f32; n * k * inter];
        let ap = OutPtr::new(&mut act);
        let blocks = inter.div_ceil(ROW_BLOCK);
        par_for(active.len() * blocks, &|t| {
            let (e, rows) = active[t / blocks];
            let r0 = t % blocks * ROW_BLOCK;
            let xs: Vec<&[f32]> = rows.iter().map(|&a| &x[a / k * d..][..d]).collect();
            let (mut g, mut u) = (vec![0.0f32; rows.len()], vec![0.0f32; rows.len()]);
            for r in r0..(r0 + ROW_BLOCK).min(inter) {
                self.gate.dots(e, r, &xs, &mut g);
                self.up.dots(e, r, &xs, &mut u);
                for ((&a, &g), &u) in rows.iter().zip(&g).zip(&u) {
                    // SAFETY: cell (a, r) belongs to this task alone: `a` routes to expert `e` once.
                    unsafe { ap.slice(a * inter + r, 1)[0] = silu(g) * u };
                }
            }
        });
        let mut y = vec![0.0f32; n * k * d];
        let yp = OutPtr::new(&mut y);
        let blocks = d.div_ceil(ROW_BLOCK);
        par_for(active.len() * blocks, &|t| {
            let (e, rows) = active[t / blocks];
            let r0 = t % blocks * ROW_BLOCK;
            let xs: Vec<&[f32]> = rows.iter().map(|&a| &act[a * inter..][..inter]).collect();
            let mut o = vec![0.0f32; rows.len()];
            for r in r0..(r0 + ROW_BLOCK).min(d) {
                self.down.dots(e, r, &xs, &mut o);
                for (&a, &v) in rows.iter().zip(&o) {
                    // SAFETY: as above, cell (a, r) is this task's.
                    unsafe { yp.slice(a * d + r, 1)[0] = v };
                }
            }
        });
        let mut out = vec![0.0f32; n * d];
        par_rows(&mut out, d, &|i, row| {
            for (s, &(_, w)) in picks[i * k..][..k].iter().enumerate() {
                vector_fused_scale_add(row, &y[(i * k + s) * d..][..d], w);
            }
        });
        if let Some(sh) = &self.shared {
            let gu = sh.gate_up.forward(x, n);
            let si = sh.down.cols;
            let mut act = vec![0.0f32; n * si];
            par_rows(&mut act, si, &|i, row| {
                vector_swiglu(&gu[i * 2 * si..][..2 * si], row, si)
            });
            let down = sh.down.forward(&act, n);
            let gate = sh.gate.forward(x, n);
            par_rows(&mut out, d, &|i, row| {
                vector_fused_scale_add(row, &down[i * d..][..d], sigmoid(gate[i]));
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::{dequantize_gguf_blocks, quantize_f32_to_q4_0};
    use crate::model::{bf16_to_f32, f32_to_bf16};
    use std::collections::HashMap;

    const D: usize = 64;
    const E: usize = 8;
    const I: usize = 32;
    const S: usize = 32;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.5
    }

    fn round(v: &[f32]) -> Vec<f32> {
        v.iter().map(|x| bf16_to_f32(f32_to_bf16(*x))).collect()
    }

    /// `m` `[rows, cols]` times `v`.
    fn mv(m: &[f32], cols: usize, v: &[f32]) -> Vec<f32> {
        m.chunks_exact(cols)
            .map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Dense-expanded reference: every expert is a plain f32 MLP, summed with the routing weights.
    fn reference(t: &HashMap<&str, Vec<f32>>, x: &[f32], k: usize) -> Vec<f32> {
        let logits = mv(&t["router"], D, x);
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let p: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let mut idx: Vec<usize> = (0..E).collect();
        idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        idx.truncate(k);
        let zk: f32 = idx.iter().map(|&j| p[j]).sum();
        let mut out = vec![0.0f32; D];
        for &j in &idx {
            let g = mv(&t["gate"][j * I * D..][..I * D], D, x);
            let u = mv(&t["up"][j * I * D..][..I * D], D, x);
            let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            for (o, y) in out
                .iter_mut()
                .zip(mv(&t["down"][j * D * I..][..D * I], I, &a))
            {
                *o += p[j] / zk * y;
            }
        }
        let a: Vec<f32> = mv(&t["sgate"], D, x)
            .iter()
            .zip(mv(&t["sup"], D, x))
            .map(|(g, u)| silu(*g) * u)
            .collect();
        let gate = sigmoid(mv(&t["sg"], D, x)[0]);
        for (o, y) in out.iter_mut().zip(mv(&t["sdown"], S, &a)) {
            *o += gate * y;
        }
        out
    }

    /// The routed experts of `q4` Q4_0 GGUF banks (separate gate/up, as llama.cpp stores them) or
    /// of bf16 safetensors tensors (HF's fused `gate_up_proj`) against dense f32 math on the same
    /// rounded weights, for one token and a batch.
    fn check(layout: &str) {
        let q4 = layout == "q4";
        let mut s = 7u64;
        let mut rnd = |n: usize| (0..n).map(|_| lcg(&mut s)).collect::<Vec<f32>>();
        let raw: Vec<(&str, Vec<f32>)> = vec![
            ("router", rnd(E * D)),
            ("gate", rnd(E * I * D)),
            ("up", rnd(E * I * D)),
            ("down", rnd(E * D * I)),
            ("sgate", rnd(S * D)),
            ("sup", rnd(S * D)),
            ("sdown", rnd(D * S)),
            ("sg", rnd(D)),
        ];
        let mut eff: HashMap<&str, Vec<f32>> = raw.iter().map(|(n, v)| (*n, round(v))).collect();
        let get = |n: &str| raw.iter().find(|r| r.0 == n).unwrap().1.clone();
        let mut data: HashMap<String, Vec<f32>> = HashMap::new();
        let mut specs = vec![
            ("m.gate.weight".to_string(), vec![E, D]),
            ("m.shared_expert.gate_proj.weight".into(), vec![S, D]),
            ("m.shared_expert.up_proj.weight".into(), vec![S, D]),
            ("m.shared_expert.down_proj.weight".into(), vec![D, S]),
            ("m.shared_expert_gate.weight".into(), vec![1, D]),
        ];
        for (name, key) in [
            ("m.gate.weight", "router"),
            ("m.shared_expert.gate_proj.weight", "sgate"),
            ("m.shared_expert.up_proj.weight", "sup"),
            ("m.shared_expert.down_proj.weight", "sdown"),
            ("m.shared_expert_gate.weight", "sg"),
        ] {
            data.insert(name.into(), get(key));
        }
        let mut packed = Vec::new();
        let mut bytes = Vec::new();
        if q4 {
            for (name, key, shape) in [
                ("m.experts.gate_proj", "gate", vec![E, I, D]),
                ("m.experts.up_proj", "up", vec![E, I, D]),
                ("m.experts.down_proj", "down", vec![E, D, I]),
            ] {
                let raw_q = quantize_f32_to_q4_0(&get(key));
                eff.insert(
                    key,
                    dequantize_gguf_blocks(QuantFormat::GGUF_Q4_0, &raw_q).unwrap(),
                );
                packed.push((name.to_string(), bytes.len(), QuantFormat::GGUF_Q4_0, shape));
                bytes.extend(raw_q);
            }
        } else if layout == "per-expert" {
            for (part, key, shape) in [
                ("gate_proj", "gate", [I, D]),
                ("up_proj", "up", [I, D]),
                ("down_proj", "down", [D, I]),
            ] {
                let w = get(key);
                for e in 0..E {
                    let name = format!("m.experts.{e}.{part}.weight");
                    specs.push((name.clone(), shape.to_vec()));
                    data.insert(
                        name,
                        w[e * shape[0] * shape[1]..][..shape[0] * shape[1]].to_vec(),
                    );
                }
            }
        } else {
            let fused: Vec<f32> = (0..E)
                .flat_map(|j| {
                    [
                        &get("gate")[j * I * D..][..I * D],
                        &get("up")[j * I * D..][..I * D],
                    ]
                    .concat()
                })
                .collect();
            specs.push(("m.experts.gate_up_proj".into(), vec![E, 2 * I, D]));
            specs.push(("m.experts.down_proj".into(), vec![E, D, I]));
            data.insert("m.experts.gate_up_proj".into(), fused);
            data.insert("m.experts.down_proj".into(), get("down"));
        }
        let mut map = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        map[..bytes.len()].copy_from_slice(&bytes);
        let map = std::sync::Arc::new(crate::model::loaders::gguf::GgufData::Map(
            map.make_read_only().unwrap(),
        ));
        let ck = Weights::from_f32(specs, |n| Ok(data[n].clone()))
            .unwrap()
            .with_gguf_packed(&map, packed);
        let k = 3;
        let m = Moe::load(&ck, "m", D, k, true).unwrap();
        let x: Vec<f32> = (0..5 * D).map(|j| (j as f32 * 0.37).sin()).collect();
        for n in [1, 5] {
            let got = m.forward(&x[..n * D], n);
            for t in 0..n {
                let want = reference(&eff, &x[t * D..][..D], k);
                let err = got[t * D..][..D]
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                let mag = want.iter().map(|v| v.abs()).fold(1.0, f32::max);
                assert!(
                    err <= 3e-3 * mag,
                    "{layout} n={n} token {t}: max err {err} (|out| <= {mag})"
                );
            }
        }
    }

    #[test]
    fn moe_matches_dense_expanded_math() {
        for layout in ["fused", "per-expert", "q4"] {
            check(layout);
        }
    }
}
