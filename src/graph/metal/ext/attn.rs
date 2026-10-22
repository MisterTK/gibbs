//! Metal lowering group `attn`: planning, encoding and kernels for its ops.
//! Kernels live in `attn.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Ops: non-cached `Attention` (grouped kv heads, causal / windowed, soft cap, additive bias
//! operand; one flash-style kernel per head-dim class), `RelPosBias`, `StateReorder`. All f32.

use super::super::{numel, set_bytes, u32_of, Planned, Session};
use super::{Enc, PlanOut};
use crate::graph::{DType, Graph, Node, Op, PairLayout, RopeVariant, ValueId};
use metal::MTLSize;
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("attn.metal");
/// Compute functions in `attn.metal`.
pub(super) const KERNELS: &[&str] = &[
    "attn_flash_nj2",
    "attn_flash_nj3",
    "attn_flash_nj4",
    "attn_flash_nj8",
    "attn_flash_nj16",
    "rel_pos_bias_f32",
    "state_reorder_f32",
    "rope_mrope_f32",
];

/// Query rows (= simdgroups) per threadgroup; `ATTNF_NSG` in `attn.metal`.
const QB: usize = 8;
/// 32 lanes x 16 dims per lane, the widest `attn_flash_nj*`.
const MAX_HEAD_DIM: usize = 512;

/// `AttnFlashParams` in `attn.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttnParams {
    nq: u32,
    nk: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    flags: u32,
    window: u32,
    bias_mode: u32,
    scale: f32,
    soft_cap: f32,
}

/// `RelPosParams` in `attn.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RelPosParams {
    n: u32,
    m: u32,
    heads: u32,
    buckets: u32,
    bidir: u32,
    scale: f32,
}

/// `ReorderParams` in `attn.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReorderParams {
    batch: u32,
    row: u32,
}

/// `MRopeParams` in `attn.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MRopeParams {
    n_seq: u32,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    adjacent: u32,
    interleaved: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    mscale: f32,
}

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    Attention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        bias: Option<ValueId>,
        kernel: &'static str,
        params: AttnParams,
    },
    RelPosBias {
        table: ValueId,
        q_pos: ValueId,
        k_pos: ValueId,
        params: RelPosParams,
    },
    StateReorder {
        state: ValueId,
        parent: ValueId,
        params: ReorderParams,
    },
    MRope {
        x: ValueId,
        pos: ValueId,
        inv_freq: ValueId,
        params: MRopeParams,
    },
}

/// Buckets per direction after the bidirectional split, if `RelPosBias` is well formed: at least
/// two (so the exact range is non-empty) and `max_distance` beyond it (so the log scale is positive).
fn rel_pos_buckets(buckets: usize, max_distance: usize, bidirectional: bool) -> Option<usize> {
    let nb = if bidirectional { buckets / 2 } else { buckets };
    (nb >= 2 && max_distance > nb / 2).then_some(nb)
}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    if dtype != DType::F32 {
        return false;
    }
    match *op {
        Op::Attention {
            heads,
            kv_heads,
            soft_cap,
            ..
        } => {
            heads > 0
                && kv_heads > 0
                && heads % kv_heads == 0
                && soft_cap.is_none_or(|c| c.is_finite() && c > 0.0)
        }
        Op::RelPosBias {
            buckets,
            max_distance,
            bidirectional,
            heads,
        } => heads > 0 && rel_pos_buckets(buckets, max_distance, bidirectional).is_some(),
        Op::StateReorder => true,
        Op::Rope {
            variant: RopeVariant::MRope { .. },
            ..
        } => true,
        _ => false,
    }
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    match node.op {
        Op::Attention { .. } => Some(plan_attention(g, node)),
        Op::RelPosBias { .. } => Some(plan_rel_pos_bias(g, node)),
        Op::StateReorder => Some(plan_state_reorder(g, node)),
        Op::Rope {
            variant: RopeVariant::MRope { .. },
            ..
        } => Some(plan_mrope(g, node)),
        _ => None,
    }
}

fn operand(node: &Node, k: usize) -> Result<ValueId, String> {
    node.inputs
        .get(k)
        .copied()
        .ok_or_else(|| format!("node '{}': {} needs operand {k}", node.name, node.op.name()))
}

fn plan_attention(g: &Graph, node: &Node) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    let Op::Attention {
        heads,
        kv_heads,
        causal,
        window,
        soft_cap,
        scale,
    } = node.op
    else {
        unreachable!("dispatched on Op::Attention")
    };
    if node.inputs.len() > 4 {
        return Err(format!(
            "node '{name}': Attention takes q, k, v and an optional bias, got {} operands",
            node.inputs.len()
        ));
    }
    if !supports(&node.op, DType::F32) {
        return Err(format!(
            "node '{name}': Attention needs heads divisible by kv_heads and a positive finite soft_cap \
             (heads {heads}, kv_heads {kv_heads}, soft_cap {soft_cap:?})"
        ));
    }
    let (q, k, v) = (operand(node, 0)?, operand(node, 1)?, operand(node, 2)?);
    let bias = node.inputs.get(3).copied();
    let width = g.value(q).shape.last().copied().unwrap_or(0);
    let (ql, kl) = (numel(g, q)?, numel(g, k)?);
    if width == 0 || width % heads != 0 {
        return Err(format!(
            "node '{name}': q/k/v shapes do not fit {heads} heads"
        ));
    }
    let head_dim = width / heads;
    if head_dim > MAX_HEAD_DIM {
        return Err(format!(
            "node '{name}': head_dim {head_dim} exceeds the Metal attention kernel's {MAX_HEAD_DIM}"
        ));
    }
    let kv_width = kv_heads * head_dim;
    if ql % width != 0 || kl == 0 || kl % kv_width != 0 || numel(g, v)? != kl {
        return Err(format!(
            "node '{name}': q/k/v shapes do not fit {heads} heads over {kv_heads} kv heads"
        ));
    }
    let (nq, nk) = (ql / width, kl / kv_width);
    // Same classification as the CPU executor: by element count (equal counts share a layout).
    let bias_mode = match bias.map(|b| numel(g, b)).transpose()? {
        None => 0,
        Some(n) if n == nk => 1,
        Some(n) if n == nq * nk => 2,
        Some(n) if n == heads * nq * nk => 3,
        Some(n) => {
            return Err(format!(
                "node '{name}': bias of {n} elements is not [{heads}, {nq}, {nk}], [{nq}, {nk}] or [{nk}]"
            ))
        }
    };
    // A window beyond every position is no window; clamping keeps the kernel's u32 math safe.
    let max_window = 2 * (nq + nk) + 2;
    u32_of(max_window)?;
    let window = window.map(|w| w.min(max_window));
    let params = AttnParams {
        nq: u32_of(nq)?,
        nk: u32_of(nk)?,
        heads: u32_of(heads)?,
        kv_heads: u32_of(kv_heads)?,
        head_dim: u32_of(head_dim)?,
        flags: causal as u32 | (window.is_some() as u32) << 1 | (soft_cap.is_some() as u32) << 2,
        window: window.map_or(Ok(0), u32_of)?,
        bias_mode,
        scale: scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
        soft_cap: soft_cap.unwrap_or(1.0),
    };
    let kernel = match head_dim.div_ceil(32) {
        0..=2 => "attn_flash_nj2",
        3 => "attn_flash_nj3",
        4 => "attn_flash_nj4",
        5..=8 => "attn_flash_nj8",
        _ => "attn_flash_nj16",
    };
    Ok(PlanOut::Step(
        Step::Attention {
            q,
            k,
            v,
            bias,
            kernel,
            params,
        },
        ql,
    ))
}

fn plan_rel_pos_bias(g: &Graph, node: &Node) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    let Op::RelPosBias {
        buckets,
        max_distance,
        bidirectional,
        heads,
    } = node.op
    else {
        unreachable!("dispatched on Op::RelPosBias")
    };
    let nb = rel_pos_buckets(buckets, max_distance, bidirectional)
        .filter(|_| heads > 0)
        .ok_or_else(|| {
            format!(
                "node '{name}': RelPosBias needs heads > 0, at least 2 buckets per direction and \
                 max_distance beyond the exact range (buckets {buckets}, max_distance {max_distance})"
            )
        })?;
    let (table, q_pos, k_pos) = (operand(node, 0)?, operand(node, 1)?, operand(node, 2)?);
    if g.value(q_pos).dtype != DType::U32 || g.value(k_pos).dtype != DType::U32 {
        return Err(format!("node '{name}': RelPosBias positions must be u32"));
    }
    if numel(g, table)? != buckets * heads {
        return Err(format!(
            "node '{name}': RelPosBias table is not [{buckets}, {heads}]"
        ));
    }
    let (n, m) = (numel(g, q_pos)?, numel(g, k_pos)?);
    u32_of(n * m)?;
    let exact = nb / 2;
    let params = RelPosParams {
        n: u32_of(n)?,
        m: u32_of(m)?,
        heads: u32_of(heads)?,
        buckets: u32_of(buckets)?,
        bidir: bidirectional as u32,
        scale: (max_distance as f64 / exact as f64).ln() as f32,
    };
    Ok(PlanOut::Step(
        Step::RelPosBias {
            table,
            q_pos,
            k_pos,
            params,
        },
        heads * n * m,
    ))
}

fn plan_state_reorder(g: &Graph, node: &Node) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    let (state, parent) = (operand(node, 0)?, operand(node, 1)?);
    if g.value(parent).dtype != DType::U32 {
        return Err(format!("node '{name}': StateReorder parents must be u32"));
    }
    let (len, batch) = (numel(g, state)?, numel(g, parent)?);
    if batch == 0 || len % batch != 0 {
        return Err(format!(
            "node '{name}': StateReorder state of {len} elements does not split into {batch} batch rows"
        ));
    }
    let params = ReorderParams {
        batch: u32_of(batch)?,
        row: u32_of(len / batch)?,
    };
    Ok(PlanOut::Step(
        Step::StateReorder {
            state,
            parent,
            params,
        },
        len,
    ))
}

/// `Rope` with `RopeVariant::MRope`: positions `[n_seq, 3]` (t, h, w), one `inv_freq` per pair.
fn plan_mrope(g: &Graph, node: &Node) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    let Op::Rope {
        head_dim,
        rotary_dim,
        variant:
            RopeVariant::MRope {
                sections,
                interleaved,
                pairs,
            },
        mscale,
    } = node.op
    else {
        unreachable!("plan_mrope is called for MRope only")
    };
    let (x, pos, inv_freq) = (operand(node, 0)?, operand(node, 1)?, operand(node, 2)?);
    let (xl, pl) = (numel(g, x)?, numel(g, pos)?);
    if g.value(pos).dtype != DType::U32 || pl == 0 || pl % 3 != 0 {
        return Err(format!(
            "node '{name}': MRope positions must be u32 [n, 3], got {pl} elements"
        ));
    }
    if head_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(format!(
            "node '{name}': MRope rotary_dim {rotary_dim} must be even and within head_dim {head_dim}"
        ));
    }
    let n_seq = pl / 3;
    if xl % (n_seq * head_dim) != 0 || numel(g, inv_freq)? != rotary_dim / 2 {
        return Err(format!(
            "node '{name}': MRope needs x of n*heads*{head_dim} elements and {} inv_freq entries",
            rotary_dim / 2
        ));
    }
    let params = MRopeParams {
        n_seq: u32_of(n_seq)?,
        n_heads: u32_of(xl / (n_seq * head_dim))?,
        head_dim: u32_of(head_dim)?,
        rotary_dim: u32_of(rotary_dim)?,
        adjacent: (pairs == PairLayout::Adjacent) as u32,
        interleaved: interleaved as u32,
        s0: sections[0],
        s1: sections[1],
        s2: sections[2],
        mscale,
    };
    Ok(PlanOut::Step(
        Step::MRope {
            x,
            pos,
            inv_freq,
            params,
        },
        xl,
    ))
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match *st {
        Step::Attention { q, k, v, bias, .. } => [Some(q), Some(k), Some(v), bias]
            .into_iter()
            .flatten()
            .collect(),
        Step::RelPosBias {
            table,
            q_pos,
            k_pos,
            ..
        } => vec![table, q_pos, k_pos],
        Step::StateReorder { state, parent, .. } => vec![state, parent],
        Step::MRope {
            x, pos, inv_freq, ..
        } => vec![x, pos, inv_freq],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, v: ValueId) -> bool {
    match *st {
        Step::Attention { .. } => false,
        Step::RelPosBias { q_pos, k_pos, .. } => v == q_pos || v == k_pos,
        Step::StateReorder { parent, .. } => v == parent,
        Step::MRope { pos, .. } => v == pos,
    }
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    let rt = &sess.rt;
    match *st {
        Step::Attention {
            q,
            k,
            v,
            bias,
            kernel,
            ref params,
        } => {
            let (k_name, qb, threads) = if rt.profile.mpp.is_ok()
                && params.nq >= 8
                && (params.nq >= 64 || params.head_dim >= 64)
                && params.head_dim % 8 == 0
                && params.head_dim <= 512
            {
                match params.head_dim {
                    0..=32 => ("flash_attn_mm_32", 32u64, 128u64),
                    33..=64 => ("flash_attn_mm_64", 32, 128),
                    65..=128 => ("flash_attn_mm_128", 32, 128),
                    129..=256 => ("flash_attn_mm_256", 32, 128),
                    _ => ("flash_attn_mm_512", 16, 128),
                }
            } else {
                (kernel, QB as u64, QB as u64 * 32)
            };
            e.enc.set_compute_pipeline_state(rt.pso(k_name)?);
            e.bind(0, q);
            e.bind(1, k);
            e.bind(2, v);
            // Never read without a bias; a buffer must still be bound.
            e.bind(3, bias.unwrap_or(q));
            e.bind(4, s.out);
            set_bytes(e.enc, 5, params);
            e.enc.dispatch_thread_groups(
                MTLSize::new((params.nq as u64).div_ceil(qb), params.heads as u64, 1),
                MTLSize::new(threads, 1, 1),
            );
        }
        Step::RelPosBias {
            table,
            q_pos,
            k_pos,
            ref params,
        } => {
            e.enc
                .set_compute_pipeline_state(rt.pso("rel_pos_bias_f32")?);
            e.bind(0, table);
            e.bind(1, q_pos);
            e.bind(2, k_pos);
            e.bind(3, s.out);
            set_bytes(e.enc, 4, params);
            e.grid1(params.n as usize * params.m as usize);
        }
        Step::StateReorder {
            state,
            parent,
            ref params,
        } => {
            e.enc
                .set_compute_pipeline_state(rt.pso("state_reorder_f32")?);
            e.bind(0, state);
            e.bind(1, parent);
            e.bind(2, s.out);
            set_bytes(e.enc, 3, params);
            e.grid1(params.batch as usize * params.row as usize);
        }
        Step::MRope {
            x,
            pos,
            inv_freq,
            ref params,
        } => {
            e.enc.set_compute_pipeline_state(rt.pso("rope_mrope_f32")?);
            e.bind(0, x);
            e.bind(1, pos);
            e.bind(2, inv_freq);
            e.bind(3, s.out);
            set_bytes(e.enc, 4, params);
            e.grid1(params.n_seq as usize * params.n_heads as usize * params.head_dim as usize);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::metal::testutil::{both, rand, rel_err, rt};
    use crate::graph::{cpu, Tensor, Weights};

    const F: DType = DType::F32;

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Bias {
        None,
        Keys,
        Rows,
        Heads,
    }

    #[derive(Clone, Debug)]
    struct Case {
        nq: usize,
        nk: usize,
        heads: usize,
        kv: usize,
        hd: usize,
        causal: bool,
        window: Option<usize>,
        cap: Option<f32>,
        scale: Option<f32>,
        bias: Bias,
        /// Multiplies q and k so scores are large enough for a soft cap to bite.
        gain: f32,
    }

    impl Case {
        fn new(nq: usize, nk: usize, heads: usize, kv: usize, hd: usize) -> Self {
            Case {
                nq,
                nk,
                heads,
                kv,
                hd,
                causal: false,
                window: None,
                cap: None,
                scale: None,
                bias: Bias::None,
                gain: 1.0,
            }
        }
        fn bias_len(&self) -> usize {
            match self.bias {
                Bias::None => 0,
                Bias::Keys => self.nk,
                Bias::Rows => self.nq * self.nk,
                Bias::Heads => self.heads * self.nq * self.nk,
            }
        }
    }

    fn rand_bias(c: &Case) -> Vec<f32> {
        rand(c.bias_len(), 77).iter().map(|x| x * 3.0).collect()
    }

    /// (CPU reference, Metal) outputs; `bias` overrides the random bias data.
    fn run(c: &Case, bias: Option<Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
        let (qw, kw) = (c.heads * c.hd, c.kv * c.hd);
        let mut g = Graph::new();
        let q = g.input("q", vec![c.nq, qw], F);
        let k = g.input("k", vec![c.nk, kw], F);
        let v = g.input("v", vec![c.nk, kw], F);
        let scaled = |n, seed| {
            rand(n, seed)
                .into_iter()
                .map(|x| x * c.gain)
                .collect::<Vec<_>>()
        };
        let mut ins = vec![q, k, v];
        let mut data = vec![
            Tensor::F32(scaled(c.nq * qw, 1)),
            Tensor::F32(scaled(c.nk * kw, 2)),
            Tensor::F32(rand(c.nk * kw, 3)),
        ];
        let shape = match c.bias {
            Bias::None => None,
            Bias::Keys => Some(vec![c.nk]),
            Bias::Rows => Some(vec![c.nq, c.nk]),
            Bias::Heads => Some(vec![c.heads, c.nq, c.nk]),
        };
        if let Some(shape) = shape {
            ins.push(g.input("bias", shape, F));
            data.push(Tensor::F32(bias.unwrap_or_else(|| rand_bias(c))));
        }
        let op = Op::Attention {
            heads: c.heads,
            kv_heads: c.kv,
            causal: c.causal,
            window: c.window,
            soft_cap: c.cap,
            scale: c.scale,
        };
        let a = g.add("attn", op, &ins, vec![c.nq, qw], F);
        g.output(a);
        both(&g, &Weights::new(), data)
    }

    fn check(c: &Case) {
        check_with(c, None);
    }

    fn check_with(c: &Case, bias: Option<Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
        let (want, got) = run(c, bias);
        assert!(
            want.iter().chain(&got).all(|v| v.is_finite()),
            "{c:?}: non-finite output"
        );
        let tol = if c.nq >= 8 && c.hd > 64 && c.hd % 8 == 0 && c.hd <= 512 {
            2e-3
        } else {
            1e-4
        };
        let e = rel_err(&want, &got);
        assert!(e <= tol, "{c:?}: rel err {e:e} (tol {tol:e})");
        (want, got)
    }

    #[test]
    fn mha_causal_and_cross_shapes() {
        // (300, 257) spans several K/V tiles at every head_dim below.
        let sizes = [
            (1, 1),
            (1, 37),
            (3, 3),
            (37, 37),
            (129, 129),
            (5, 129),
            (129, 5),
            (33, 100),
            (100, 33),
            (8, 8),
            (9, 9),
            (16, 16),
            (300, 257),
        ];
        for (nq, nk) in sizes {
            for causal in [false, true] {
                for (heads, hd) in [(3, 24), (2, 64), (1, 8)] {
                    check(&Case {
                        causal,
                        ..Case::new(nq, nk, heads, heads, hd)
                    });
                }
            }
        }
    }

    #[test]
    fn grouped_kv_heads_and_head_dims() {
        for (heads, kv) in [(4, 2), (8, 1), (6, 3), (2, 2), (12, 4)] {
            for hd in [8, 24, 32, 40, 64, 72, 96, 128, 160, 256, 512] {
                for causal in [false, true] {
                    check(&Case {
                        causal,
                        ..Case::new(21, 23, heads, kv, hd)
                    });
                }
            }
        }
    }

    #[test]
    fn windows() {
        for (nq, nk) in [(37, 37), (129, 129), (40, 7), (7, 40)] {
            for causal in [false, true] {
                for w in [0, 1, 2, 3, 4, 5, 16, 500] {
                    check(&Case {
                        causal,
                        window: Some(w),
                        ..Case::new(nq, nk, 4, 2, 32)
                    });
                }
            }
        }
        // A window wider than usize::MAX / 4 must not overflow the kernel's u32 math.
        check(&Case {
            causal: true,
            window: Some(usize::MAX),
            ..Case::new(9, 9, 2, 1, 16)
        });
        check(&Case {
            causal: false,
            window: Some(usize::MAX),
            ..Case::new(9, 9, 2, 1, 16)
        });
    }

    #[test]
    fn soft_cap() {
        for cap in [5.0, 30.0] {
            for causal in [false, true] {
                for (heads, kv, hd, window) in
                    [(4, 2, 32, None), (3, 3, 24, Some(7)), (8, 1, 64, Some(6))]
                {
                    let c = Case {
                        causal,
                        window,
                        cap: Some(cap),
                        gain: 6.0,
                        ..Case::new(37, 37, heads, kv, hd)
                    };
                    let (capped, _) = check_with(&c, None);
                    // The test only has power if the cap changes the result.
                    let (plain, _) = run(
                        &Case {
                            cap: None,
                            ..c.clone()
                        },
                        None,
                    );
                    let diff = capped
                        .iter()
                        .zip(&plain)
                        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                    assert!(diff > 1e-2, "cap {cap} left {c:?} unchanged ({diff:e})");
                }
            }
        }
    }

    /// An explicit `scale` replaces 1/sqrt(head_dim) on both executors (and changes the result).
    #[test]
    fn explicit_scale() {
        for scale in [1.0, 0.05] {
            for (causal, window, cap) in [(false, None, None), (true, Some(5), Some(8.0))] {
                let c = Case {
                    causal,
                    window,
                    cap,
                    scale: Some(scale),
                    gain: 2.0,
                    ..Case::new(19, 19, 4, 2, 32)
                };
                let (scaled, _) = check_with(&c, None);
                let (plain, _) = run(
                    &Case {
                        scale: None,
                        ..c.clone()
                    },
                    None,
                );
                let diff = scaled
                    .iter()
                    .zip(&plain)
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(diff > 1e-3, "scale {scale} left {c:?} unchanged ({diff:e})");
            }
        }
    }

    #[test]
    fn bias_operand_forms() {
        for bias in [Bias::Keys, Bias::Rows, Bias::Heads] {
            for causal in [false, true] {
                for (nq, nk) in [(1, 1), (1, 37), (37, 37), (129, 129), (5, 129), (129, 5)] {
                    check(&Case {
                        causal,
                        bias,
                        ..Case::new(nq, nk, 4, 2, 32)
                    });
                }
            }
        }
        // Everything at once.
        let all = Case {
            causal: true,
            window: Some(9),
            cap: Some(20.0),
            bias: Bias::Heads,
            gain: 4.0,
            ..Case::new(41, 41, 6, 3, 40)
        };
        check(&all);
    }

    #[test]
    fn masks_and_empty_rows_give_zeros() {
        // Padding mask on the last keys of a [nk] bias: -inf and a finite -1e9.
        for neg in [f32::NEG_INFINITY, -1e9] {
            let c = Case {
                bias: Bias::Keys,
                ..Case::new(37, 37, 4, 2, 32)
            };
            let mut b = vec![0.0; 37];
            b[30..].fill(neg);
            check_with(&c, Some(b));
        }
        // A [nq, nk] bias with row 2 entirely -inf: that row is zero on both executors.
        let c = Case {
            bias: Bias::Rows,
            causal: true,
            ..Case::new(9, 9, 2, 1, 16)
        };
        let mut b = rand(81, 5);
        b[2 * 9..3 * 9].fill(f32::NEG_INFINITY);
        let (want, got) = check_with(&c, Some(b));
        for out in [&want, &got] {
            assert!(
                out[2 * 32..3 * 32].iter().all(|&v| v == 0.0),
                "masked row is not zero"
            );
            assert!(out[3 * 32..4 * 32].iter().any(|&v| v != 0.0));
        }
        // One head fully masked in a [heads, nq, nk] bias.
        let c = Case {
            bias: Bias::Heads,
            ..Case::new(5, 11, 3, 3, 8)
        };
        let mut b = rand(3 * 5 * 11, 6);
        b[5 * 11..2 * 5 * 11].fill(f32::NEG_INFINITY);
        let (want, got) = check_with(&c, Some(b));
        for out in [&want, &got] {
            for row in 0..5 {
                assert!(out[row * 24 + 8..row * 24 + 16].iter().all(|&v| v == 0.0));
            }
        }
        // Causal window 0 sees no key; causal queries beyond a short key sequence's window too.
        let (want, got) = check_with(
            &Case {
                causal: true,
                window: Some(0),
                ..Case::new(6, 6, 2, 1, 16)
            },
            None,
        );
        assert!(want.iter().chain(&got).all(|&v| v == 0.0));
        // Row p sees keys p-2..=p; from row 9 on that is past the 7 keys.
        let (want, got) = check_with(
            &Case {
                causal: true,
                window: Some(3),
                ..Case::new(40, 7, 2, 2, 8)
            },
            None,
        );
        for out in [&want, &got] {
            assert!(
                out[9 * 16..].iter().all(|&v| v == 0.0),
                "rows past the window must be zero"
            );
            assert!(out[8 * 16..9 * 16].iter().any(|&v| v != 0.0));
        }
    }

    /// With `v[t] = e_t` the output row is the attention weight vector, so the mask is visible.
    #[test]
    fn window_semantics_match_s22() {
        let n = 15;
        let mut ident = vec![0.0f32; n * n];
        for t in 0..n {
            ident[t * n + t] = 1.0;
        }
        for (causal, w, row, keys) in [
            (true, 4, 10, 7..=10),
            (true, 1, 10, 10..=10),
            (false, 4, 10, 8..=12),
            (false, 5, 10, 8..=12),
            (false, 0, 10, 10..=10),
        ] {
            let mut g = Graph::new();
            let q = g.input("q", vec![n, n], F);
            let k = g.input("k", vec![n, n], F);
            let v = g.input("v", vec![n, n], F);
            let op = Op::Attention {
                heads: 1,
                kv_heads: 1,
                causal,
                window: Some(w),
                soft_cap: None,
                scale: None,
            };
            let a = g.add("attn", op, &[q, k, v], vec![n, n], F);
            g.output(a);
            let data = vec![
                Tensor::F32(rand(n * n, 1)),
                Tensor::F32(rand(n * n, 2)),
                Tensor::F32(ident.clone()),
            ];
            let (want, got) = both(&g, &Weights::new(), data);
            for (who, out) in [("cpu", &want), ("metal", &got)] {
                let weights = &out[row * n..(row + 1) * n];
                for (t, &p) in weights.iter().enumerate() {
                    assert_eq!(
                        p > 0.0,
                        keys.contains(&t),
                        "{who} causal {causal} w {w}: key {t} weight {p}"
                    );
                }
                assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
        }
    }

    /// The scalar reference against the independent `CachedAttention` implementation.
    #[test]
    fn cpu_general_attention_matches_cached_attention() {
        let (n, heads, kv, hd) = (21usize, 4usize, 2usize, 16usize);
        for window in [None, Some(1), Some(5), Some(100)] {
            for cap in [None, Some(7.0f32)] {
                let build = |op: Op| {
                    let mut g = Graph::new();
                    let q = g.input("q", vec![n, heads * hd], F);
                    let k = g.input("k", vec![n, kv * hd], F);
                    let v = g.input("v", vec![n, kv * hd], F);
                    let a = g.add("a", op, &[q, k, v], vec![n, heads * hd], F);
                    g.output(a);
                    g
                };
                let attn = Op::Attention {
                    heads,
                    kv_heads: kv,
                    causal: true,
                    window,
                    soft_cap: cap,
                    scale: None,
                };
                let cached = Op::CachedAttention {
                    heads,
                    kv_heads: kv,
                    head_dim: hd,
                    v_head_dim: hd,
                    scale: 1.0 / (hd as f32).sqrt(),
                    causal: true,
                    window,
                    soft_cap: cap,
                    write: true,
                };
                let data = || {
                    vec![
                        Tensor::F32(rand(n * heads * hd, 1).iter().map(|x| x * 3.0).collect()),
                        Tensor::F32(rand(n * kv * hd, 2).iter().map(|x| x * 3.0).collect()),
                        Tensor::F32(rand(n * kv * hd, 3)),
                    ]
                };
                let w = Weights::new();
                let a = cpu::run_stateless(&build(attn), &w, data())
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                let b = cpu::run_stateless(&build(cached), &w, data())
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                assert!(
                    rel_err(&b, &a) < 1e-5,
                    "window {window:?} cap {cap:?}: {:e}",
                    rel_err(&b, &a)
                );
            }
        }
    }

    /// Table, RelPosBias and Attention in one graph, the T5 layer shape.
    #[test]
    fn rel_pos_bias_feeds_attention() {
        for (bidirectional, causal, nq, nk, q_pos) in [
            (
                true,
                false,
                37usize,
                37usize,
                (0..37u32).collect::<Vec<_>>(),
            ),
            (false, true, 37, 37, (0..37).collect()),
            (false, false, 1, 37, vec![36]),
            (true, false, 5, 29, vec![3, 9, 0, 28, 14]),
        ] {
            let (heads, hd, buckets) = (4usize, 16usize, 32usize);
            let mut g = Graph::new();
            let table = g.weight("table", vec![buckets, heads], F);
            let q = g.input("q", vec![nq, heads * hd], F);
            let k = g.input("k", vec![nk, heads * hd], F);
            let v = g.input("v", vec![nk, heads * hd], F);
            let qp = g.input("q_pos", vec![nq], DType::U32);
            let kp = g.input("k_pos", vec![nk], DType::U32);
            let rpb = Op::RelPosBias {
                buckets,
                max_distance: 128,
                bidirectional,
                heads,
            };
            let bias = g.add("rpb", rpb, &[table, qp, kp], vec![heads, nq, nk], F);
            let op = Op::Attention {
                heads,
                kv_heads: heads,
                causal,
                window: None,
                soft_cap: None,
                scale: None,
            };
            let a = g.add("attn", op, &[q, k, v, bias], vec![nq, heads * hd], F);
            g.output(a);
            let mut w = Weights::new();
            w.insert(
                "table",
                Tensor::F32(rand(buckets * heads, 4).iter().map(|x| x * 4.0).collect()),
            );
            let data = vec![
                Tensor::F32(rand(nq * heads * hd, 1)),
                Tensor::F32(rand(nk * heads * hd, 2)),
                Tensor::F32(rand(nk * heads * hd, 3)),
                Tensor::U32(q_pos),
                Tensor::U32((0..nk as u32).collect()),
            ];
            let (want, got) = both(&g, &w, data);
            assert!(
                rel_err(&want, &got) < 1e-4,
                "bidir {bidirectional} causal {causal}: {:e}",
                rel_err(&want, &got)
            );
        }
    }

    /// Tiled MPP flash attention (`flash_attn_mm`) across query lengths {8, 33, 512}, head dims
    /// {64, 128, 256, 512}, GQA 8:1 and 1:1, causal, window 128, soft-cap 50, mask and bias.
    #[test]
    fn flash_attn_mm_matrix() {
        for nq in [8usize, 33, 512] {
            for hd in [64usize, 128, 256, 512] {
                for (heads, kv) in [(8usize, 1usize), (2, 2)] {
                    for (causal, window, cap, bias, masked) in [
                        (true, None, None, Bias::None, false),
                        (true, Some(128), None, Bias::None, false),
                        (false, Some(128), None, Bias::None, false),
                        (true, None, Some(50.0), Bias::None, false),
                        (false, None, None, Bias::Rows, false),
                        (false, None, None, Bias::Keys, true),
                    ] {
                        let c = Case {
                            causal,
                            window,
                            cap,
                            bias,
                            ..Case::new(nq, nq, heads, kv, hd)
                        };
                        let mask_vec = masked.then(|| {
                            let mut b = vec![0.0f32; nq];
                            b[nq / 2..].fill(f32::NEG_INFINITY);
                            b
                        });
                        let (want, got) = run(&c, mask_vec);
                        let e = rel_err(&want, &got);
                        assert!(
                            e <= 2e-3,
                            "nq {nq} hd {hd} {heads}:{kv} causal {causal} win {window:?} cap {cap:?} bias {bias:?} mask {masked}: {e:e}"
                        );
                    }
                }
            }
        }
    }

    /// `table[i] = i`, so the output is `bucket * heads + head`, compared exactly.
    fn rel_pos_case(
        buckets: usize,
        max_distance: usize,
        bidirectional: bool,
        heads: usize,
        q_pos: Vec<u32>,
        k_pos: Vec<u32>,
    ) -> (Vec<f32>, Vec<f32>) {
        let (n, m) = (q_pos.len(), k_pos.len());
        let mut g = Graph::new();
        let table = g.weight("table", vec![buckets, heads], F);
        let qp = g.input("q_pos", vec![n], DType::U32);
        let kp = g.input("k_pos", vec![m], DType::U32);
        let op = Op::RelPosBias {
            buckets,
            max_distance,
            bidirectional,
            heads,
        };
        let o = g.add("rpb", op, &[table, qp, kp], vec![heads, n, m], F);
        g.output(o);
        let mut w = Weights::new();
        w.insert(
            "table",
            Tensor::F32((0..buckets * heads).map(|i| i as f32).collect()),
        );
        both(&g, &w, vec![Tensor::U32(q_pos), Tensor::U32(k_pos)])
    }

    #[test]
    fn rel_pos_bias_matches_cpu() {
        let configs = [
            (32, 128),
            (8, 128),
            (16, 64),
            (64, 256),
            (128, 512),
            (12, 30),
            (5, 3),
            (4, 3),
            (32, 256),
            (32, 1000),
        ];
        for (buckets, max_distance) in configs {
            for bidirectional in [true, false] {
                for heads in [1, 3, 8] {
                    let (want, got) = rel_pos_case(
                        buckets,
                        max_distance,
                        bidirectional,
                        heads,
                        vec![0, 5, 250, 300],
                        (0..601).collect(),
                    );
                    assert_eq!(want, got, "buckets {buckets} max_distance {max_distance} bidir {bidirectional} heads {heads}");
                }
            }
        }
        // Odd extents and n = m = 1.
        let (want, got) = rel_pos_case(32, 128, true, 2, vec![7], vec![3]);
        assert_eq!(want, got);
        let (want, got) = rel_pos_case(32, 128, false, 5, (0..37).collect(), (0..129).collect());
        assert_eq!(want, got);
    }

    /// The CPU bucketing is transformers' `_relative_position_bucket` (same expectations as `t5::tests`).
    #[test]
    fn rel_pos_bias_buckets_match_hf() {
        let want_buckets = |bidir: bool, rels: &[i64], expect: &[f32]| {
            let k_pos: Vec<u32> = rels.iter().map(|r| (200 + r) as u32).collect();
            let (cpu_out, metal_out) = rel_pos_case(32, 128, bidir, 1, vec![200], k_pos);
            assert_eq!(cpu_out, expect, "cpu bidirectional {bidir}");
            assert_eq!(metal_out, expect, "metal bidirectional {bidir}");
        };
        want_buckets(
            true,
            &[-200, -20, -9, -8, -1, 0, 1, 7, 8, 16, 100],
            &[15., 10., 8., 8., 1., 0., 17., 23., 24., 26., 31.],
        );
        want_buckets(
            false,
            &[-200, -20, -16, -15, -1, 0, 3],
            &[31., 17., 16., 15., 1., 0., 0.],
        );
    }

    #[test]
    fn state_reorder_matches_cpu() {
        for (batch, row) in [(1usize, 1usize), (2, 3), (5, 37), (37, 129), (4, 1)] {
            let parents: [Vec<u32>; 4] = [
                (0..batch as u32).rev().collect(),
                (0..batch as u32).map(|b| (b + 1) % batch as u32).collect(),
                vec![0; batch],
                // The last row points past the batch: zeros, as on the CPU.
                (0..batch as u32)
                    .map(|b| {
                        if b + 1 == batch as u32 {
                            batch as u32 + 3
                        } else {
                            b
                        }
                    })
                    .collect(),
            ];
            for parent in parents {
                let mut g = Graph::new();
                let state = g.input("state", vec![batch, row], F);
                let p = g.input("parent", vec![batch], DType::U32);
                let o = g.add(
                    "reorder",
                    Op::StateReorder,
                    &[state, p],
                    vec![batch, row],
                    F,
                );
                g.output(o);
                let data = vec![
                    Tensor::F32(rand(batch * row, 9)),
                    Tensor::U32(parent.clone()),
                ];
                let (want, got) = both(&g, &Weights::new(), data);
                assert_eq!(want, got, "batch {batch} row {row} parent {parent:?}");
            }
        }
    }

    /// Every MRope form against the CPU arm: contiguous and interleaved sections, both pair
    /// layouts, partial rotary_dim, a head count above one, mscale.
    #[test]
    fn mrope_matches_cpu() {
        let cases = [
            // (head_dim, rotary_dim, sections, interleaved, pairs, mscale)
            (
                16usize,
                16usize,
                [3u32, 3, 2],
                false,
                PairLayout::Half,
                1.0f32,
            ),
            (16, 16, [3, 3, 2], false, PairLayout::Adjacent, 1.0),
            (16, 8, [2, 1, 1], false, PairLayout::Half, 1.0),
            (16, 8, [2, 1, 1], true, PairLayout::Half, 1.0),
            (16, 16, [3, 3, 2], true, PairLayout::Adjacent, 0.5),
            (128, 128, [8, 28, 28], false, PairLayout::Adjacent, 1.0),
        ];
        for (hd, rot, sections, interleaved, pairs, mscale) in cases {
            let (n, heads) = (5usize, 3usize);
            let mut g = Graph::new();
            let x = g.input("x", vec![n, heads * hd], F);
            let pos = g.input("pos", vec![n * 3], DType::U32);
            let f = g.input("inv_freq", vec![rot / 2], F);
            let y = g.add(
                "rope",
                Op::Rope {
                    head_dim: hd,
                    rotary_dim: rot,
                    variant: RopeVariant::MRope {
                        sections,
                        interleaved,
                        pairs,
                    },
                    mscale,
                },
                &[x, pos, f],
                vec![n, heads * hd],
                F,
            );
            g.output(y);
            let positions: Vec<u32> = (0..n as u32 * 3).map(|i| (i * 7 + i / 3) % 23).collect();
            let inv: Vec<f32> = (0..rot / 2)
                .map(|i| 100f32.powf(-(i as f32) / (rot / 2) as f32))
                .collect();
            let data = vec![
                Tensor::F32(rand(n * heads * hd, 4)),
                Tensor::U32(positions),
                Tensor::F32(inv),
            ];
            let (want, got) = both(&g, &Weights::new(), data);
            let e = rel_err(&want, &got);
            assert!(
                e < 1e-4,
                "hd {hd} rot {rot} {sections:?} il {interleaved} {pairs:?}: {e}"
            );
        }
    }

    fn run_metal(g: &Graph, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        Session::new(rt(), &Weights::new()).unwrap().run(g, inputs)
    }

    #[test]
    fn claims_and_rejections() {
        let attn = |heads, kv_heads, window, soft_cap| Op::Attention {
            heads,
            kv_heads,
            causal: true,
            window,
            soft_cap,
            scale: None,
        };
        assert!(supports(&attn(4, 2, Some(8), Some(30.0)), F));
        assert!(!supports(&attn(4, 2, None, None), DType::F16));
        assert!(!supports(&attn(4, 3, None, None), F));
        assert!(!supports(&attn(4, 2, None, Some(0.0)), F));
        let rpb = |buckets, max_distance| Op::RelPosBias {
            buckets,
            max_distance,
            bidirectional: true,
            heads: 2,
        };
        assert!(supports(&rpb(32, 128), F));
        assert!(!supports(&rpb(2, 128), F)); // one bucket per direction: empty exact range
        assert!(!supports(&rpb(32, 8), F)); // max_distance not beyond the 8 exact buckets
        assert!(supports(&Op::StateReorder, F));
        assert!(!supports(&Op::StateReorder, DType::BF16));

        // head_dim above the widest kernel is a plan error naming the node.
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 544], F);
        let k = g.input("k", vec![1, 544], F);
        let v = g.input("v", vec![1, 544], F);
        let a = g.add("wide", attn(1, 1, None, None), &[q, k, v], vec![1, 544], F);
        g.output(a);
        let inputs = vec![
            Tensor::F32(rand(544, 1)),
            Tensor::F32(rand(544, 2)),
            Tensor::F32(rand(544, 3)),
        ];
        let err = run_metal(&g, inputs).unwrap_err();
        assert!(
            err.contains("wide") && err.contains("head_dim 544"),
            "{err}"
        );

        // A bias that is none of [heads, nq, nk], [nq, nk], [nk].
        let mut g = Graph::new();
        let q = g.input("q", vec![3, 8], F);
        let k = g.input("k", vec![5, 8], F);
        let v = g.input("v", vec![5, 8], F);
        let b = g.input("b", vec![7], F);
        let a = g.add(
            "biased",
            attn(1, 1, None, None),
            &[q, k, v, b],
            vec![3, 8],
            F,
        );
        g.output(a);
        let inputs = vec![
            Tensor::F32(rand(24, 1)),
            Tensor::F32(rand(40, 2)),
            Tensor::F32(rand(40, 3)),
            Tensor::F32(rand(7, 4)),
        ];
        let err = run_metal(&g, inputs.clone()).unwrap_err();
        assert!(err.contains("biased") && err.contains("bias"), "{err}");
        assert!(cpu::run_stateless(&g, &Weights::new(), inputs)
            .unwrap_err()
            .contains("bias"));

        // heads not divisible by kv_heads.
        let mut g = Graph::new();
        let q = g.input("q", vec![2, 12], F);
        let k = g.input("k", vec![2, 8], F);
        let v = g.input("v", vec![2, 8], F);
        let a = g.add("ragged", attn(3, 2, None, None), &[q, k, v], vec![2, 12], F);
        g.output(a);
        let err = run_metal(
            &g,
            vec![
                Tensor::F32(rand(24, 1)),
                Tensor::F32(rand(16, 2)),
                Tensor::F32(rand(16, 3)),
            ],
        )
        .unwrap_err();
        assert!(err.contains("ragged") && err.contains("kv_heads"), "{err}");
    }
}
