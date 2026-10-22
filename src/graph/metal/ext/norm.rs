//! Metal lowering group `norm`: planning, encoding and kernels for its ops.
//! Kernels live in `norm.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Owned: `Reduce` (every kind, any axes, keep_dims either way), `Softmax` (any axis),
//! `GroupNorm`, `InstanceNorm` (`[N, C, ...]`, see `cpu::channel_layout`), `LayerNorm` without
//! beta and `RmsNorm` without gamma. `LayerNorm` with beta, `RmsNorm` with gamma (a `[group]`
//! gamma already normalizes x in chunks of its length) and `L2Normalize` (whole tensor, no axis
//! attribute) stay on the core arms.

use super::super::{set_bytes, u32_of, Planned, Session, TG};
use super::{Enc, PlanOut};
use crate::graph::cpu::channel_layout;
use crate::graph::{DType, Graph, Node, Op, ReduceKind, ValueId};
use metal::MTLSize;
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("norm.metal");
/// Compute functions in `norm.metal`.
pub(super) const KERNELS: &[&str] = &[
    "norm_reduce_f32",
    "norm_softmax_f32",
    "norm_layernorm_nobeta_f32",
    "norm_group_stats_f32",
    "norm_group_apply_f32",
    "norm_rmsnorm_nogamma_f32",
];

/// Mirrors `NormReduceParams` in `norm.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceParams {
    kind: u32,
    rcount: u32,
    nk: u32,
    nr: u32,
    n_out: u32,
    kdim: [u32; 4],
    kstr: [u32; 4],
    rdim: [u32; 4],
    rstr: [u32; 4],
}

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    Reduce {
        x: ValueId,
        p: ReduceParams,
        n_out: usize,
    },
    Softmax {
        x: ValueId,
        outer: usize,
        n: usize,
        inner: usize,
    },
    LayerNormNoBeta {
        x: ValueId,
        g: ValueId,
        rows: usize,
        d: usize,
        eps: f32,
    },
    RmsNormNoGamma {
        x: ValueId,
        rows: usize,
        d: usize,
        eps: f32,
    },
    Group {
        x: ValueId,
        g: ValueId,
        b: ValueId,
        tgs: usize,
        groups: usize,
        cg: usize,
        hw: usize,
        eps: f32,
        silu: bool,
    },
}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    dtype == DType::F32
        && matches!(
            op,
            Op::Reduce { .. }
                | Op::Softmax { .. }
                | Op::GroupNorm { .. }
                | Op::InstanceNorm { .. }
                | Op::LayerNorm { .. }
                | Op::RmsNorm { .. }
        )
}

fn norm_axis(axis: isize, rank: usize) -> Option<usize> {
    let ax = if axis < 0 { axis + rank as isize } else { axis };
    (0..rank as isize).contains(&ax).then_some(ax as usize)
}

/// Splits `shape` into kept and reduced runs `(dim, stride)`: size-1 axes dropped, adjacent
/// same-kind axes coalesced. Errors on a bad/duplicate axis or more than 4 runs of a kind.
fn reduce_params(
    kind: ReduceKind,
    shape: &[usize],
    axes: &[isize],
    name: &str,
) -> Result<(ReduceParams, usize), String> {
    let rank = shape.len();
    if axes.is_empty() {
        return Err(format!("node '{name}': Reduce with no axes is ambiguous"));
    }
    let mut red = vec![false; rank];
    for &a in axes {
        match norm_axis(a, rank) {
            Some(ax) if !red[ax] => red[ax] = true,
            _ => {
                return Err(format!(
                    "node '{name}': bad Reduce axis {a} for rank {rank}"
                ))
            }
        }
    }
    let mut stride = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    let (mut kept, mut reduced): (Vec<(usize, usize)>, Vec<(usize, usize)>) = (vec![], vec![]);
    for i in 0..rank {
        if shape[i] == 1 {
            continue;
        }
        let runs = if red[i] { &mut reduced } else { &mut kept };
        match runs.last_mut() {
            // The previous run ends exactly where this axis begins in memory.
            Some((d, s)) if *s == shape[i] * stride[i] => {
                *d *= shape[i];
                *s = stride[i];
            }
            _ => runs.push((shape[i], stride[i])),
        }
    }
    if kept.len() > 4 || reduced.len() > 4 {
        return Err(format!(
            "node '{name}': Reduce over {rank} axes needs more than 4 strided runs"
        ));
    }
    let mut p = ReduceParams {
        kind: match kind {
            ReduceKind::Sum => 0,
            ReduceKind::Mean => 1,
            ReduceKind::Max => 2,
            ReduceKind::Min => 3,
        },
        nk: kept.len() as u32,
        nr: reduced.len() as u32,
        ..Default::default()
    };
    let (mut n_out, mut rcount) = (1usize, 1usize);
    for (i, &(d, s)) in kept.iter().enumerate() {
        p.kdim[i] = u32_of(d)?;
        p.kstr[i] = u32_of(s)?;
        n_out *= d;
    }
    for (i, &(d, s)) in reduced.iter().enumerate() {
        p.rdim[i] = u32_of(d)?;
        p.rstr[i] = u32_of(s)?;
        rcount *= d;
    }
    p.rcount = u32_of(rcount)?;
    p.n_out = u32_of(n_out)?;
    Ok((p, n_out))
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    let name = &node.name;
    let arg = |k: usize| {
        node.inputs
            .get(k)
            .copied()
            .ok_or_else(|| format!("node '{name}': {} needs operand {k}", node.op.name()))
    };
    let n_of = |v: ValueId| -> Result<usize, String> {
        let n: usize = g.value(v).shape.iter().product();
        if n == 0 {
            Err(format!(
                "node '{name}': value '{}' has no static non-empty shape",
                g.value(v).name
            ))
        } else {
            Ok(n)
        }
    };
    let r: Result<(Step, usize), String> = (|| match &node.op {
        Op::Reduce { kind, axes, .. } => {
            let x = arg(0)?;
            n_of(x)?;
            let (p, n_out) = reduce_params(*kind, &g.value(x).shape, axes, name)?;
            Ok((Step::Reduce { x, p, n_out }, n_out))
        }
        Op::Softmax { axis } => {
            let x = arg(0)?;
            let xl = n_of(x)?;
            let shape = &g.value(x).shape;
            let ax = norm_axis(*axis, shape.len())
                .ok_or_else(|| format!("node '{name}': bad Softmax axis {axis}"))?;
            let (outer, n, inner) = (
                shape[..ax].iter().product::<usize>(),
                shape[ax],
                shape[ax + 1..].iter().product::<usize>(),
            );
            u32_of(n)?;
            u32_of(inner)?;
            u32_of(outer * inner)?;
            Ok((Step::Softmax { x, outer, n, inner }, xl))
        }
        // The beta form stays on the core arm.
        Op::LayerNorm { eps } if node.inputs.len() == 2 => {
            let (x, gm) = (arg(0)?, arg(1)?);
            let (xl, d) = (n_of(x)?, n_of(gm)?);
            if xl % d != 0 {
                return Err(format!("node '{name}': x/gamma lengths {xl}/{d} disagree"));
            }
            u32_of(d)?;
            Ok((
                Step::LayerNormNoBeta {
                    x,
                    g: gm,
                    rows: xl / d,
                    d,
                    eps: *eps,
                },
                xl,
            ))
        }
        // The gamma form stays on the core arm (gamma `[group]` already normalizes x in chunks).
        Op::RmsNorm { eps } if node.inputs.len() == 1 => {
            let x = arg(0)?;
            let xl = n_of(x)?;
            let d = g.value(x).shape.last().copied().unwrap_or(0);
            u32_of(d)?;
            Ok((
                Step::RmsNormNoGamma {
                    x,
                    rows: xl / d.max(1),
                    d,
                    eps: *eps,
                },
                xl,
            ))
        }
        Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
            let (x, gm, b) = (arg(0)?, arg(1)?, arg(2)?);
            let (xl, c) = (n_of(x)?, n_of(gm)?);
            let (groups, eps) = match node.op {
                Op::GroupNorm { groups, eps } => (groups, eps),
                Op::InstanceNorm { eps } => (c, eps),
                _ => unreachable!(),
            };
            if n_of(b)? != c || groups == 0 || c % groups != 0 {
                return Err(format!(
                    "node '{name}': gamma/beta lengths {c}/{} do not fit {groups} groups",
                    n_of(b)?
                ));
            }
            let (n, hw) =
                channel_layout(&g.value(x).shape, c).map_err(|e| format!("node '{name}': {e}"))?;
            if n * c * hw != xl {
                return Err(format!(
                    "node '{name}': x of {xl} elements is not {n}*{c}*{hw}"
                ));
            }
            u32_of(c / groups * hw)?;
            u32_of(n * groups)?;
            Ok((
                Step::Group {
                    x,
                    g: gm,
                    b,
                    tgs: n * groups,
                    groups,
                    cg: c / groups,
                    hw,
                    eps,
                    silu: false,
                },
                xl,
            ))
        }
        _ => Err(String::new()),
    })();
    match r {
        Err(e) if e.is_empty() => None,
        Err(e) => Some(Err(e)),
        Ok((st, n)) => Some(Ok(PlanOut::Step(st, n))),
    }
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match *st {
        Step::Reduce { x, .. } | Step::Softmax { x, .. } | Step::RmsNormNoGamma { x, .. } => {
            vec![x]
        }
        Step::LayerNormNoBeta { x, g, .. } => vec![x, g],
        Step::Group { x, g, b, .. } => vec![x, g, b],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(_st: &Step, _v: ValueId) -> bool {
    false
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    let tg = MTLSize::new(TG, 1, 1);
    match *st {
        Step::Reduce { x, ref p, n_out } => {
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("norm_reduce_f32")?);
            e.bind(0, x);
            e.bind(1, s.out);
            set_bytes(e.enc, 2, p);
            if p.rcount <= 32 {
                e.grid1(n_out);
            } else {
                e.enc.dispatch_thread_groups(
                    MTLSize::new(n_out as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
            }
        }
        Step::Softmax { x, outer, n, inner } => {
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("norm_softmax_f32")?);
            e.bind(0, x);
            e.bind(1, s.out);
            set_bytes(e.enc, 2, &[u32_of(n)?, u32_of(inner)?]);
            e.enc
                .dispatch_thread_groups(MTLSize::new((outer * inner) as u64, 1, 1), tg);
        }
        Step::LayerNormNoBeta { x, g, rows, d, eps } => {
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("norm_layernorm_nobeta_f32")?);
            e.bind(0, x);
            e.bind(1, g);
            e.bind(2, s.out);
            set_bytes(e.enc, 3, &[u32_of(d)?, eps.to_bits()]);
            e.enc
                .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), tg);
        }
        Step::RmsNormNoGamma { x, rows, d, eps } => {
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("norm_rmsnorm_nogamma_f32")?);
            e.bind(0, x);
            e.bind(1, s.out);
            set_bytes(e.enc, 2, &[u32_of(d)?, eps.to_bits()]);
            e.enc
                .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), tg);
        }
        Step::Group {
            x,
            g,
            b,
            tgs,
            groups,
            cg,
            hw,
            eps,
            silu,
        } => {
            let cnt = cg * hw;
            let splits = cnt.div_ceil(4096).max(1);
            let bytes = tgs * splits * 8;
            let mut guard = sess.argmax_scratch.lock();
            if guard
                .as_ref()
                .is_none_or(|buf| (buf.length() as usize) < bytes)
            {
                *guard = Some(super::super::new_shared_buffer(&sess.rt.device, bytes)?);
            }
            let stats = guard.as_ref().expect("allocated above");
            #[repr(C)]
            struct NormGroupParams {
                n_groups: u32,
                groups: u32,
                cg: u32,
                hw: u32,
                splits: u32,
                silu: u32,
                eps: f32,
            }
            let p = NormGroupParams {
                n_groups: u32_of(tgs)?,
                groups: u32_of(groups)?,
                cg: u32_of(cg)?,
                hw: u32_of(hw)?,
                splits: u32_of(splits)?,
                silu: silu as u32,
                eps,
            };
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("norm_group_stats_f32")?);
            e.bind(0, x);
            e.enc.set_buffer(1, Some(stats), 0);
            set_bytes(e.enc, 2, &p);
            e.enc.dispatch_thread_groups(
                MTLSize::new(splits as u64, tgs as u64, 1),
                tg,
            );
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("norm_group_apply_f32")?);
            e.bind(0, x);
            e.bind(1, g);
            e.bind(2, b);
            e.enc.set_buffer(3, Some(stats), 0);
            e.bind(4, s.out);
            set_bytes(e.enc, 5, &p);
            e.grid1((tgs * cnt).div_ceil(4));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{both, rand, rel_err};
    use crate::graph::{DType, Graph, Op, ReduceKind, Tensor, Weights};

    const F: DType = DType::F32;

    fn run1(g: &Graph, w: &Weights, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        both(g, w, vec![Tensor::F32(x.to_vec())])
    }

    #[test]
    fn reduce_matches_cpu() {
        let shapes: &[&[usize]] = &[
            &[1],
            &[37],
            &[3, 5],
            &[129, 3],
            &[2, 3, 4],
            &[4, 1, 7],
            &[2, 3, 4, 5],
            &[2, 3, 4, 5, 3],
            &[1, 1],
        ];
        let kinds = [
            ReduceKind::Sum,
            ReduceKind::Mean,
            ReduceKind::Max,
            ReduceKind::Min,
        ];
        let mut cases = 0;
        for shape in shapes {
            let rank = shape.len();
            // Every non-empty subset of axes (as negative indices for odd masks).
            for mask in 1u32..(1 << rank) {
                let axes: Vec<isize> = (0..rank)
                    .filter(|i| mask >> i & 1 == 1)
                    .map(|i| {
                        if mask % 2 == 1 {
                            i as isize
                        } else {
                            i as isize - rank as isize
                        }
                    })
                    .collect();
                for keep_dims in [true, false] {
                    let out_shape: Vec<usize> = (0..rank)
                        .filter_map(|i| {
                            let reduced = mask >> i & 1 == 1;
                            match (reduced, keep_dims) {
                                (true, true) => Some(1),
                                (true, false) => None,
                                _ => Some(shape[i]),
                            }
                        })
                        .collect();
                    // Reduce-all without keep_dims has a rank-0 output, which the executor cannot place.
                    if out_shape.is_empty() {
                        continue;
                    }
                    for kind in kinds {
                        let mut g = Graph::new();
                        let x = g.input("x", shape.to_vec(), F);
                        let y = g.add(
                            "r",
                            Op::Reduce {
                                kind,
                                axes: axes.clone(),
                                keep_dims,
                            },
                            &[x],
                            out_shape.clone(),
                            F,
                        );
                        g.output(y);
                        let n: usize = shape.iter().product();
                        let (want, got) = run1(&g, &Weights::new(), &rand(n, 7 + cases as u64));
                        let e = rel_err(&want, &got);
                        assert!(
                            e < 1e-4,
                            "{kind:?} {shape:?} axes {axes:?} keep {keep_dims}: {e}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases > 400, "{cases}");
    }

    #[test]
    fn reduce_large_axis_and_scalar_keepdims() {
        let mut g = Graph::new();
        let x = g.input("x", vec![5, 4099], F);
        let y = g.add(
            "r",
            Op::Reduce {
                kind: ReduceKind::Sum,
                axes: vec![1],
                keep_dims: false,
            },
            &[x],
            vec![5],
            F,
        );
        g.output(y);
        let (want, got) = run1(&g, &Weights::new(), &rand(5 * 4099, 3));
        assert!(rel_err(&want, &got) < 1e-4);
    }

    #[test]
    fn softmax_every_axis_matches_cpu() {
        for shape in [
            vec![1usize],
            vec![37],
            vec![3, 5],
            vec![129, 3],
            vec![2, 3, 4],
            vec![2, 37, 3, 2],
            vec![4, 300],
        ] {
            for axis in 0..shape.len() as isize {
                for ax in [axis, axis - shape.len() as isize] {
                    let mut g = Graph::new();
                    let x = g.input("x", shape.clone(), F);
                    let y = g.add("sm", Op::Softmax { axis: ax }, &[x], shape.clone(), F);
                    g.output(y);
                    let n: usize = shape.iter().product();
                    let xs: Vec<f32> = rand(n, 11).iter().map(|v| v * 8.0).collect();
                    let (want, got) = run1(&g, &Weights::new(), &xs);
                    let e = rel_err(&want, &got);
                    assert!(e < 1e-5, "{shape:?} axis {ax}: {e}");
                }
            }
        }
    }

    #[test]
    fn layernorm_without_beta_and_with_beta_match_cpu() {
        for (rows, d) in [(1usize, 1usize), (3, 37), (5, 129), (2, 1024), (7, 3)] {
            for with_beta in [false, true] {
                let mut g = Graph::new();
                let x = g.input("x", vec![rows, d], F);
                let gm = g.weight("g", vec![d], F);
                let mut ins = vec![x, gm];
                let mut w = Weights::new();
                w.insert(
                    "g",
                    Tensor::F32(rand(d, 5).iter().map(|v| v + 1.5).collect()),
                );
                if with_beta {
                    ins.push(g.weight("b", vec![d], F));
                    w.insert("b", Tensor::F32(rand(d, 6)));
                }
                let y = g.add("ln", Op::LayerNorm { eps: 1e-5 }, &ins, vec![rows, d], F);
                g.output(y);
                let (want, got) = run1(&g, &w, &rand(rows * d, 9));
                let e = rel_err(&want, &got);
                assert!(e < 1e-4, "{rows}x{d} beta {with_beta}: {e}");
            }
        }
    }

    #[test]
    fn group_and_instance_norm_match_cpu() {
        // (shape, c, groups); channels at axis 0 or axis 1 with batch.
        let cases: &[(&[usize], usize, usize)] = &[
            (&[4, 5], 4, 2),
            (&[8, 3, 3], 8, 4),
            (&[6, 37], 6, 1),
            (&[6, 1], 6, 3),
            (&[12, 129], 12, 12),
            (&[3, 8, 2, 2], 8, 2),
            (&[2, 6, 7], 6, 3),
            (&[5, 4, 3, 3, 2], 4, 4),
        ];
        for &(shape, c, groups) in cases {
            for instance in [false, true] {
                let mut g = Graph::new();
                let x = g.input("x", shape.to_vec(), F);
                let gm = g.weight("g", vec![c], F);
                let b = g.weight("b", vec![c], F);
                let op = if instance {
                    Op::InstanceNorm { eps: 1e-5 }
                } else {
                    Op::GroupNorm { groups, eps: 1e-5 }
                };
                let y = g.add("n", op, &[x, gm, b], shape.to_vec(), F);
                g.output(y);
                let mut w = Weights::new();
                w.insert(
                    "g",
                    Tensor::F32(rand(c, 1).iter().map(|v| v + 1.2).collect()),
                );
                w.insert("b", Tensor::F32(rand(c, 2)));
                let n: usize = shape.iter().product();
                let xs: Vec<f32> = rand(n, 4).iter().map(|v| v * 3.0 + 0.7).collect();
                let (want, got) = run1(&g, &w, &xs);
                let e = rel_err(&want, &got);
                assert!(
                    e < 1e-4,
                    "{shape:?} c {c} groups {groups} instance {instance}: {e}"
                );
            }
        }
    }

    /// `[N, C, H, W]` with N == C: expected values from a naive loop written here, not from the CPU executor.
    #[test]
    fn group_norm_n_equals_c_matches_naive_nchw() {
        let (n, c, hw) = (4usize, 4usize, 64usize);
        let xs: Vec<f32> = rand(n * c * hw, 21).iter().map(|v| v * 2.0 + 0.3).collect();
        let (ga, be) = (rand(c, 22), rand(c, 23));
        for groups in [1usize, 2, 4] {
            let cg = c / groups;
            let mut want = vec![0.0f32; xs.len()];
            for ni in 0..n {
                for gi in 0..groups {
                    let idx = |ci: usize, j: usize| ((ni * c) + gi * cg + ci) * hw + j;
                    let vals: Vec<f64> = (0..cg)
                        .flat_map(|ci| (0..hw).map(move |j| (ci, j)))
                        .map(|(ci, j)| xs[idx(ci, j)] as f64)
                        .collect();
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let var =
                        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                    for ci in 0..cg {
                        for j in 0..hw {
                            let ch = gi * cg + ci;
                            want[idx(ci, j)] =
                                ((xs[idx(ci, j)] as f64 - mean) / (var + 1e-5).sqrt()
                                    * ga[ch] as f64
                                    + be[ch] as f64) as f32;
                        }
                    }
                }
            }
            let mut g = Graph::new();
            let x = g.input("x", vec![n, c, 8, 8], F);
            let gm = g.weight("g", vec![c], F);
            let b = g.weight("b", vec![c], F);
            let y = g.add(
                "n",
                Op::GroupNorm { groups, eps: 1e-5 },
                &[x, gm, b],
                vec![n, c, 8, 8],
                F,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert("g", Tensor::F32(ga.clone()));
            w.insert("b", Tensor::F32(be.clone()));
            let (cpu, got) = run1(&g, &w, &xs);
            assert!(rel_err(&want, &got) < 1e-4, "metal groups {groups}");
            assert!(rel_err(&want, &cpu) < 1e-4, "cpu groups {groups}");
        }
    }

    #[test]
    fn group_norm_split_and_fused_silu_match_cpu() {
        for c in [128usize, 320, 512] {
            for s in [64usize, 128] {
                for silu in [false, true] {
                    let shape = vec![1, c, s, s];
                    let mut g = Graph::new();
                    let x = g.input("x", shape.clone(), F);
                    let gm = g.weight("g", vec![c], F);
                    let b = g.weight("b", vec![c], F);
                    let gn = g.add(
                        "gn",
                        Op::GroupNorm { groups: 32, eps: 1e-5 },
                        &[x, gm, b],
                        shape.clone(),
                        F,
                    );
                    let out = if silu {
                        g.add(
                            "silu",
                            Op::Unary {
                                kind: crate::graph::UnaryKind::Silu,
                            },
                            &[gn],
                            shape.clone(),
                            F,
                        )
                    } else {
                        gn
                    };
                    g.output(out);
                    let (steps, _) = super::super::super::plan(&g).unwrap();
                    assert_eq!(steps.len(), 1);
                    let mut w = Weights::new();
                    w.insert("g", Tensor::F32(rand(c, 1).iter().map(|v| v + 1.1).collect()));
                    w.insert("b", Tensor::F32(rand(c, 2)));
                    let xs = rand(c * s * s, 3);
                    let (want, got) = run1(&g, &w, &xs);
                    let e = rel_err(&want, &got);
                    assert!(e < 1e-4, "c={c} s={s} silu={silu}: {e}");
                }
            }
        }
        // Large DC offset with multi-threadgroup split (> 4096 elements per group) to verify shifted-sum stability.
        let (c, s) = (128usize, 64usize);
        let shape = vec![1, c, s, s];
        let mut g = Graph::new();
        let x = g.input("x", shape.clone(), F);
        let gm = g.weight("g", vec![c], F);
        let b = g.weight("b", vec![c], F);
        let y = g.add(
            "gn",
            Op::GroupNorm { groups: 32, eps: 1e-5 },
            &[x, gm, b],
            shape,
            F,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("g", Tensor::F32(rand(c, 1).iter().map(|v| v + 1.1).collect()));
        w.insert("b", Tensor::F32(rand(c, 2)));
        let xs: Vec<f32> = rand(c * s * s, 9).into_iter().map(|v| v + 1000.0).collect();
        let (want, got) = run1(&g, &w, &xs);
        assert!(rel_err(&want, &got) < 1e-4);
    }

    #[test]
    fn rmsnorm_without_gamma_and_group_gamma_match_cpu() {
        // (x shape, gamma len or 0 for none)
        for (shape, glen) in [
            (vec![3usize, 37], 0usize),
            (vec![5, 129], 0),
            (vec![1, 1], 0),
            (vec![2, 3, 8], 0),
            (vec![5, 4 * 16], 16),
            (vec![3, 6 * 7], 7),
            (vec![2, 37], 37),
        ] {
            let mut g = Graph::new();
            let x = g.input("x", shape.clone(), F);
            let mut ins = vec![x];
            let mut w = Weights::new();
            if glen > 0 {
                ins.push(g.weight("g", vec![glen], F));
                w.insert(
                    "g",
                    Tensor::F32(rand(glen, 8).iter().map(|v| v + 1.3).collect()),
                );
            }
            let y = g.add("rms", Op::RmsNorm { eps: 1e-6 }, &ins, shape.clone(), F);
            g.output(y);
            let n: usize = shape.iter().product();
            let (want, got) = run1(&g, &w, &rand(n, 12));
            let e = rel_err(&want, &got);
            assert!(e < 1e-4, "{shape:?} gamma {glen}: {e}");
        }
    }
}
