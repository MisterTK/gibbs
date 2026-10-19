//! Zero-copy views over planned steps (after `batch_horizontal`, before `allocate`).
//!
//! - Row-strided view: a single-source copy that gathers `len`-element rows `stride` apart (a
//!   `Slice` on a non-leading axis) becomes an alias `(root, offset)` when every reader takes a row
//!   stride for that operand ([`Rows`]); the readers get the stride and the copy step goes away.
//! - Destination passing: a copy whose sources are each one dense run of the output (a `Concat`
//!   whose operands are contiguous slots) aliases every source, when all are outputs of earlier
//!   steps, to its slot of the output, so the producers write in place and the copy step goes away.
//!
//! Weights and states never become views (their copies stay). `allocate` keeps an alias root alive
//! from its first writer to the last reader of any alias of it.

use super::ext::{elementwise, movement, ExtStep};
use super::{Planned, Step};
use crate::graph::{Graph, ValueId, ValueKind};
use std::collections::HashMap;

/// Row-strided layout of an operand: rows of `len` contiguous elements, `stride` elements apart.
/// Dense when `len == stride`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rows {
    pub len: usize,
    pub stride: usize,
}

impl Rows {
    pub fn dense(n: usize) -> Rows {
        Rows { len: n, stride: n }
    }
}

/// Removes the copy steps that can become views, recording them in `aliases`.
pub(super) fn elide(
    g: &Graph,
    steps: &mut Vec<Planned>,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) {
    let mut i = 0;
    while i < steps.len() {
        if row_view(g, steps, i, aliases) || dest_passing(g, steps, i, aliases) {
            steps.remove(i);
        } else {
            i += 1;
        }
    }
}

fn copy_parts(st: &Planned) -> Option<&[(ValueId, movement::Map)]> {
    match &st.op {
        Step::Ext(ExtStep::Movement(movement::Step::Copy { fill: None, parts }))
            if st.extra_outs.is_empty() =>
        {
            Some(parts)
        }
        _ => None,
    }
}

fn row_view(
    g: &Graph,
    steps: &mut [Planned],
    i: usize,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> bool {
    let Some(&[(x, ref m)]) = copy_parts(&steps[i]) else {
        return false;
    };
    let Some((len, stride, ibase)) = m.row_gather() else {
        return false;
    };
    let out = steps[i].out;
    let (root, base) = aliases.get(&x).copied().unwrap_or((x, 0));
    if !matches!(g.value(root).kind, ValueKind::Node(_) | ValueKind::Input)
        || g.outputs.contains(&out)
        || aliases.values().any(|a| a.0 == out)
    {
        return false;
    }
    let view = Rows { len, stride };
    let readers: Vec<usize> = (i + 1..steps.len())
        .filter(|&j| steps[j].reads.contains(&out))
        .collect();
    if !readers
        .iter()
        .all(|&j| take_view(&mut steps[j].op, out, view, false))
    {
        return false;
    }
    for j in readers {
        take_view(&mut steps[j].op, out, view, true);
    }
    aliases.insert(out, (root, base + ibase * 4));
    true
}

/// Whether `op` reads `v` only through operands that take a row stride; with `apply`, gives them
/// `view`. ponytail: RmsNormBatch, RopeBatch and Unary are the readers of strided slices in the
/// suite's plans; any other reader (plain RmsNorm, Rope, attention `v`, GLU) keeps the copy step.
fn take_view(op: &mut Step, v: ValueId, view: Rows, apply: bool) -> bool {
    match op {
        Step::RmsNormBatch { segs, rope } => {
            if rope.is_some_and(|r| r.pos == v || r.inv_freq == v)
                || segs
                    .iter()
                    .any(|s| s.g == v || (s.x == v && (s.d == 0 || view.len % s.d != 0)))
            {
                return false;
            }
            if apply {
                segs.iter_mut()
                    .filter(|s| s.x == v)
                    .for_each(|s| s.xv = view);
            }
            true
        }
        Step::RopeBatch {
            pos,
            inv_freq,
            head_dim,
            segs,
            ..
        } => {
            if *pos == v
                || *inv_freq == v
                || segs.iter().any(|s| s.0 == v && view.len != s.2 * *head_dim)
            {
                return false;
            }
            if apply {
                segs.iter_mut()
                    .filter(|s| s.0 == v)
                    .for_each(|s| s.3 = view);
            }
            true
        }
        Step::Ext(ExtStep::Elementwise(elementwise::Step::Unary { xv, .. })) => {
            if apply {
                *xv = view;
            }
            true
        }
        _ => false,
    }
}

fn dest_passing(
    g: &Graph,
    steps: &[Planned],
    i: usize,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> bool {
    let Some(parts) = copy_parts(&steps[i]) else {
        return false;
    };
    let out = steps[i].out;
    let mut slots = Vec::with_capacity(parts.len());
    for (k, (v, m)) in parts.iter().enumerate() {
        let Some((at, n)) = m.dense_run() else {
            return false;
        };
        let produced = steps[..i]
            .iter()
            .any(|s| s.out == *v || s.extra_outs.contains(v));
        if !produced
            || n != g.value(*v).shape.iter().product::<usize>()
            || aliases.contains_key(v)
            || parts[..k].iter().any(|p| p.0 == *v)
        {
            return false;
        }
        slots.push((*v, at * 4));
    }
    for (v, off) in slots {
        for a in aliases.values_mut().filter(|a| a.0 == v) {
            *a = (out, off + a.1);
        }
        aliases.insert(v, (out, off));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{rand, rel_err, rt};
    use super::super::{allocate, batch_horizontal, kernel_kind, plan, Session};
    use super::*;
    use crate::graph::{cpu, DType, Op, RopeVariant, Tensor, UnaryKind, Weights};

    const F: DType = DType::F32;

    fn slice(g: &mut Graph, x: ValueId, start: i64, end: i64, shape: Vec<usize>) -> ValueId {
        g.add(
            "sl",
            Op::Slice {
                axis: -1,
                start,
                end,
            },
            &[x],
            shape,
            F,
        )
    }

    fn planned(g: &Graph) -> (Vec<Planned>, HashMap<ValueId, (ValueId, usize)>) {
        let (steps, mut aliases) = plan(g).unwrap();
        let mut steps = batch_horizontal(steps, &aliases);
        elide(g, &mut steps, &mut aliases);
        (steps, aliases)
    }

    fn kinds(steps: &[Planned]) -> Vec<String> {
        steps
            .iter()
            .map(|s| kernel_kind(&format!("{:?}", s.op)).to_string())
            .collect()
    }

    /// Row-strided slices read by an RmsNormBatch (decode q/gate split), a Unary and a RopeBatch
    /// (prefill q/k split); a Binary-read slice keeps its copy. Returns the graph and its inputs.
    fn strided_graph(g: &mut Graph) -> (ValueId, ValueId) {
        let x = g.input("x", vec![4, 64], F);
        let k = g.input("k", vec![4, 32], F);
        let x2 = g.input("x2", vec![3, 48], F);
        let pos = g.input("pos", vec![3], DType::U32);
        let gq = g.weight("gq", vec![32], F);
        let inv = g.weight("inv", vec![4], F);
        let q = slice(g, x, 0, 32, vec![4, 32]);
        let gate = slice(g, x, 32, 64, vec![4, 32]);
        let q2 = slice(g, x2, 0, 16, vec![3, 16]);
        let k2 = slice(g, x2, 16, 32, vec![3, 16]);
        let v2 = slice(g, x2, 32, 48, vec![3, 16]);
        let qn = g.add("qn", Op::RmsNorm { eps: 1e-6 }, &[q, gq], vec![4, 32], F);
        let kn = g.add("kn", Op::RmsNorm { eps: 1e-6 }, &[k, gq], vec![4, 32], F);
        let sg = g.add(
            "sg",
            Op::Unary {
                kind: UnaryKind::Sigmoid,
            },
            &[gate],
            vec![4, 32],
            F,
        );
        let rope = Op::Rope {
            head_dim: 8,
            rotary_dim: 8,
            variant: RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        let rq = g.add("rq", rope.clone(), &[q2, pos, inv], vec![3, 16], F);
        let rk = g.add("rk", rope, &[k2, pos, inv], vec![3, 16], F);
        let ad = g.add("ad", Op::Add, &[v2, v2], vec![3, 16], F);
        for o in [qn, kn, sg, rq, rk, ad] {
            g.output(o);
        }
        (x, v2)
    }

    #[test]
    fn strided_slices_become_views_of_stride_taking_readers() {
        let mut g = Graph::new();
        let (x, v2) = strided_graph(&mut g);
        // A slice of a state never becomes a view.
        let st = g.state("s", vec![4, 64], F);
        let ss = slice(&mut g, st, 0, 32, vec![4, 32]);
        let so = g.add(
            "so",
            Op::Unary {
                kind: UnaryKind::Exp,
            },
            &[ss],
            vec![4, 32],
            F,
        );
        g.output(so);
        let (steps, aliases) = planned(&g);
        let copies: Vec<ValueId> = steps
            .iter()
            .filter(|s| kinds(std::slice::from_ref(s))[0] == "Copy")
            .map(|s| s.out)
            .collect();
        assert_eq!(copies, [v2, ss], "{:?}", kinds(&steps));
        let mut seen = 0;
        for s in &steps {
            match &s.op {
                Step::RmsNormBatch { segs, .. } => {
                    assert_eq!(
                        segs[0].xv,
                        Rows {
                            len: 32,
                            stride: 64
                        }
                    );
                    assert_eq!(aliases[&segs[0].x], (x, 0));
                    assert_eq!(segs[1].xv, Rows::dense(128));
                    seen |= 1;
                }
                Step::Ext(ExtStep::Elementwise(elementwise::Step::Unary { x: gx, xv, .. }))
                    if *gx != ss =>
                {
                    assert_eq!(
                        *xv,
                        Rows {
                            len: 32,
                            stride: 64
                        }
                    );
                    assert_eq!(aliases[gx], (x, 32 * 4));
                    seen |= 2;
                }
                Step::RopeBatch { segs, .. } => {
                    assert_eq!(
                        segs[0].3,
                        Rows {
                            len: 16,
                            stride: 48
                        }
                    );
                    assert_eq!(aliases[&segs[1].0].1, 16 * 4);
                    seen |= 4;
                }
                _ => {}
            }
        }
        assert_eq!(seen, 7, "{:?}", kinds(&steps));
    }

    fn concat_graph(g: &mut Graph) -> (ValueId, ValueId, ValueId) {
        let a = g.input("a", vec![1, 8], F);
        let b = g.input("b", vec![1, 8], F);
        let ea = g.add(
            "ea",
            Op::Unary {
                kind: UnaryKind::Exp,
            },
            &[a],
            vec![1, 8],
            F,
        );
        let eb = g.add(
            "eb",
            Op::Unary {
                kind: UnaryKind::Tanh,
            },
            &[b],
            vec![1, 8],
            F,
        );
        let cat = g.add("cat", Op::Concat { axis: -1 }, &[ea, eb], vec![1, 16], F);
        let o = g.add(
            "o",
            Op::Unary {
                kind: UnaryKind::Neg,
            },
            &[cat],
            vec![1, 16],
            F,
        );
        g.output(o);
        (ea, eb, cat)
    }

    #[test]
    fn concat_of_producers_writes_in_place() {
        let mut g = Graph::new();
        let (ea, eb, cat) = concat_graph(&mut g);
        // Operands that are graph inputs keep the copy.
        let c = g.input("c", vec![1, 8], F);
        let cat2 = g.add("cat2", Op::Concat { axis: -1 }, &[c, ea], vec![1, 16], F);
        g.output(cat2);
        let (steps, aliases) = planned(&g);
        assert_eq!(kinds(&steps), ["Unary", "Unary", "Unary", "Copy"]);
        assert_eq!(steps[3].out, cat2);
        assert_eq!((aliases[&ea], aliases[&eb]), ((cat, 0), (cat, 32)));
        let (offsets, _) = allocate(&g, &steps, &aliases).unwrap();
        assert_eq!(
            (offsets[&ea], offsets[&eb]),
            (offsets[&cat], offsets[&cat] + 32)
        );
    }

    /// An alias root lives from its first writer (destination passing) to the last reader of any
    /// alias of it, even with no reader of its own.
    #[test]
    fn allocate_keeps_alias_roots_alive() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 64], F);
        let un = |g: &mut Graph, kind, v| g.add("u", Op::Unary { kind }, &[v], vec![1, 64], F);
        // Read alias: `y` is only read through its contiguous slice `ys`, after `z` and `z2` exist.
        let y = un(&mut g, UnaryKind::Exp, x);
        let ys = g.add(
            "ys",
            Op::Slice {
                axis: -1,
                start: 0,
                end: 32,
            },
            &[y],
            vec![1, 32],
            F,
        );
        let z = un(&mut g, UnaryKind::Tanh, x);
        let z2 = un(&mut g, UnaryKind::Neg, z);
        let out = g.add("out", Op::Concat { axis: -1 }, &[ys, ys], vec![1, 64], F);
        g.output(out);
        g.output(z2);
        // Write alias: dead concat `cat` whose second part is written after `m` is allocated.
        let p0 = un(&mut g, UnaryKind::Exp, x);
        let m = un(&mut g, UnaryKind::Tanh, x);
        let p1 = un(&mut g, UnaryKind::Neg, m);
        let cat = g.add("cat", Op::Concat { axis: -1 }, &[p0, p1], vec![1, 128], F);
        let (steps, aliases) = planned(&g);
        assert_eq!(aliases[&ys], (y, 0));
        assert_eq!((aliases[&p0], aliases[&p1]), ((cat, 0), (cat, 256)));
        let (offsets, _) = allocate(&g, &steps, &aliases).unwrap();
        let size =
            |v: ValueId| (g.value(v).shape.iter().product::<usize>() * 4).div_ceil(256) * 256;
        let disjoint = |p: ValueId, q: ValueId| {
            offsets[&p] + size(p) <= offsets[&q] || offsets[&q] + size(q) <= offsets[&p]
        };
        for (p, q) in [(y, z), (y, z2), (cat, m)] {
            assert!(disjoint(p, q), "{p:?} and {q:?} overlap");
        }
    }

    /// Metal parity for every reader that takes a row-strided view, and for destination passing.
    #[test]
    fn strided_views_and_concat_dps_match_cpu() {
        let mut g = Graph::new();
        strided_graph(&mut g);
        let mut w = Weights::new();
        w.insert("gq", Tensor::F32(rand(32, 9)));
        w.insert("inv", Tensor::F32(vec![1.0, 0.3, 0.1, 0.03]));
        let inputs = vec![
            Tensor::F32(rand(256, 1)),
            Tensor::F32(rand(128, 2)),
            Tensor::F32(rand(144, 3)),
            Tensor::U32(vec![4, 5, 6]),
        ];
        let sess = Session::new(rt(), &w).unwrap();
        let copies = |g: &Graph| {
            let p = sess.prepare_plan(g).unwrap();
            p.steps
                .iter()
                .filter(|s| kernel_kind(&format!("{:?}", s.op)) == "Copy")
                .count()
        };
        assert_eq!(copies(&g), 1);
        let want = cpu::run_stateless(&g, &w, inputs.clone()).unwrap();
        let got = sess.run(&g, inputs).unwrap();
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            let e = rel_err(a.as_f32().unwrap(), b.as_f32().unwrap());
            assert!(e < 1e-5, "output {i}: {e}");
        }

        let mut g = Graph::new();
        concat_graph(&mut g);
        assert_eq!(copies(&g), 0);
        let inputs = vec![Tensor::F32(rand(8, 4)), Tensor::F32(rand(8, 5))];
        let want = cpu::run_stateless(&g, &w, inputs.clone()).unwrap();
        let got = sess.run(&g, inputs).unwrap();
        assert!(rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap()) < 1e-6);
    }
}
