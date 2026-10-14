//! Metal lowering group `elementwise`: planning, encoding and kernels for its ops.
//! Kernels live in `elementwise.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Owned ops, all at F32: every `Unary` kind (plus `Relu`, `Silu`, `Gelu` which are the same maps),
//! every `Binary` kind (plus `Add`, `Mul`) with numpy broadcasting.

use super::super::{numel, set_bytes, u32_of, Planned, Session};
use super::{Enc, PlanOut};
use crate::graph::{BinaryKind, DType, Graph, Node, Op, UnaryKind, ValueId};
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("elementwise.metal");
/// Compute functions in `elementwise.metal`.
pub(super) const KERNELS: &[&str] = &["unary_f32", "binary_f32"];

// ponytail: rank > 6 after coalescing is an error; add dims if a model needs it. F32 only: F16/BF16
// operands reach these kernels through the `Cast` of `passes::dtype_adapt` (no `Op::Cast` exists yet).
/// The kernel's coalesced output rank limit.
const MAX_RANK: usize = 6;

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    /// `out = f(x)`; `kind` is the `unary_f32` switch code, `a`/`b` its parameters.
    Unary {
        x: ValueId,
        n: usize,
        kind: u32,
        a: f32,
        b: f32,
    },
    /// `out = a ∘ b` over `dims` (outermost first); `sa`/`sb` are element strides, 0 when broadcast.
    /// `pre` maps one operand through a unary kind first (`Pre::NONE`: none): a single-reader unary
    /// producer fused into its consumer.
    Binary {
        a: ValueId,
        b: ValueId,
        n: usize,
        kind: u32,
        rank: usize,
        dims: [u32; MAX_RANK],
        sa: [u32; MAX_RANK],
        sb: [u32; MAX_RANK],
        pre: Pre,
    },
}

/// A unary map applied to a [`Step::Binary`] operand while it is read: `side` 0 = none, 1 = `a`, 2 = `b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pre {
    pub side: u32,
    pub kind: u32,
    pub a: f32,
    pub b: f32,
}

impl Pre {
    pub const NONE: Pre = Pre { side: 0, kind: 0, a: 0.0, b: 0.0 };
}

/// The `(x, kind, a, b)` of an unary step, `None` for any other.
pub(in crate::graph::metal) fn as_unary(st: &Step) -> Option<(ValueId, u32, f32, f32)> {
    match *st {
        Step::Unary { x, kind, a, b, .. } => Some((x, kind, a, b)),
        Step::Binary { .. } => None,
    }
}

/// Folds the unary step `(kind, a, b)` that produced operand `side` (1 = `a`, 2 = `b`) of binary
/// step `bin` into it, reading the unary's input `x` instead. `None` unless `bin` is a plain binary.
pub(in crate::graph::metal) fn with_pre(bin: &Step, side: u32, x: ValueId, kind: u32, a: f32, b: f32) -> Option<Step> {
    match *bin {
        Step::Binary { a: ba, b: bb, n, kind: k, rank, dims, sa, sb, pre } if pre == Pre::NONE => {
            let (ba, bb) = if side == 1 { (x, bb) } else { (ba, x) };
            Some(Step::Binary { a: ba, b: bb, n, kind: k, rank, dims, sa, sb, pre: Pre { side, kind, a, b } })
        }
        _ => None,
    }
}

/// `(kind code, a, b)` for ops that are an element-wise unary map. The match is exhaustive over
/// `UnaryKind`: a new kind breaks the build here until its kernel case exists.
fn unary_code(op: &Op) -> Option<(u32, f32, f32)> {
    Some(match op {
        Op::Relu => (8, 0.0, 0.0),
        Op::Silu => (10, 0.0, 0.0),
        Op::Gelu { tanh } => (if *tanh { 12 } else { 11 }, 0.0, 0.0),
        Op::Unary { kind } => match kind {
            UnaryKind::Exp => (0, 0.0, 0.0),
            UnaryKind::Sigmoid => (1, 0.0, 0.0),
            UnaryKind::Tanh => (2, 0.0, 0.0),
            UnaryKind::Softplus => (3, 0.0, 0.0),
            UnaryKind::Neg => (4, 0.0, 0.0),
            UnaryKind::Sqrt => (5, 0.0, 0.0),
            UnaryKind::Rsqrt => (6, 0.0, 0.0),
            UnaryKind::Abs => (7, 0.0, 0.0),
            UnaryKind::Relu => (8, 0.0, 0.0),
            UnaryKind::Relu2 => (9, 0.0, 0.0),
            UnaryKind::Silu => (10, 0.0, 0.0),
            UnaryKind::GeluErf => (11, 0.0, 0.0),
            UnaryKind::GeluTanh => (12, 0.0, 0.0),
            UnaryKind::Scale(s) => (13, *s, 0.0),
            UnaryKind::Clamp(lo, hi) => (14, *lo, *hi),
            UnaryKind::SoftCap(cap) => (15, *cap, 0.0),
        },
        _ => return None,
    })
}

/// `binary_f32` switch code for ops that are an element-wise binary map.
fn binary_code(op: &Op) -> Option<u32> {
    Some(match op {
        Op::Add => 0,
        Op::Mul => 2,
        Op::Binary { kind } => match kind {
            BinaryKind::Add => 0,
            BinaryKind::Sub => 1,
            BinaryKind::Mul => 2,
            BinaryKind::Div => 3,
            BinaryKind::Max => 4,
            BinaryKind::Min => 5,
        },
        _ => return None,
    })
}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
/// Activations flow as f32 buffers; F16/BF16 values reach these kernels through a `Cast`.
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    dtype == DType::F32 && (unary_code(op).is_some() || binary_code(op).is_some())
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    if let Some((kind, a, b)) = unary_code(&node.op) {
        return Some(plan_unary(g, node, kind, a, b));
    }
    binary_code(&node.op).map(|kind| plan_binary(g, node, kind))
}

fn plan_unary(g: &Graph, node: &Node, kind: u32, a: f32, b: f32) -> Result<PlanOut<Step>, String> {
    let x = *node
        .inputs
        .first()
        .ok_or_else(|| format!("node '{}': {} needs operand 0", node.name, node.op.name()))?;
    let n = numel(g, x)?;
    u32_of(n)?;
    Ok(PlanOut::Step(Step::Unary { x, n, kind, a, b }, n))
}

/// Output dims and each operand's element strides (0 where broadcast), outermost first.
type View = (Vec<usize>, Vec<usize>, Vec<usize>);

/// How `a ∘ b` indexes its operands, with the CPU executor's rules (`cpu.rs`, Binary/Add/Mul):
/// 1. numpy broadcasting of the declared shapes, when it yields `out_n` elements;
/// 2. equal element counts: flat;
/// 3. `Mul` only: a numpy-incompatible `[m, ..] * [m, 1]` scales row `r` of `a` by `b[r]` (either order);
/// 4. else the shorter operand tiles over the longer one by flat length.
fn broadcast_view(
    sa: &[usize],
    sb: &[usize],
    out_n: usize,
    row_scale: bool,
    name: &str,
) -> Result<View, String> {
    let (al, bl) = (sa.iter().product::<usize>(), sb.iter().product::<usize>());
    let r = sa.len().max(sb.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() >= r {
            s[i + s.len() - r]
        } else {
            1
        }
    };
    let np: Option<Vec<usize>> = (0..r)
        .map(|i| {
            let (x, y) = (dim(sa, i), dim(sb, i));
            (x == y || x == 1 || y == 1).then_some(if x == 1 { y } else { x })
        })
        .collect();
    let strides = |s: &[usize]| {
        let mut st = vec![0; r];
        let mut acc = 1;
        for i in (0..r).rev() {
            let d = dim(s, i);
            st[i] = if d == 1 { 0 } else { acc };
            acc *= d;
        }
        st
    };
    let view = match np {
        Some(dims) if dims.iter().product::<usize>() == out_n => (dims, strides(sa), strides(sb)),
        _ if al == bl => (vec![al], vec![1], vec![1]),
        _ if row_scale
            && sa.len() >= 2
            && sb.len() >= 2
            && sb.last() == Some(&1)
            && sa[0] == sb[0]
            && bl > 0 =>
        {
            (vec![bl, al / bl], vec![al / bl, 1], vec![1, 0])
        }
        _ if row_scale
            && sa.len() >= 2
            && sb.len() >= 2
            && sa.last() == Some(&1)
            && sa[0] == sb[0]
            && al > 0 =>
        {
            (vec![al, bl / al], vec![1, 0], vec![bl / al, 1])
        }
        _ if bl > 0 && al % bl == 0 => (vec![al / bl, bl], vec![bl, 1], vec![0, 1]),
        _ if al > 0 && bl % al == 0 => (vec![bl / al, al], vec![0, 1], vec![al, 1]),
        _ => {
            return Err(format!(
                "node '{name}': cannot broadcast operands of shape {sa:?} and {sb:?}"
            ))
        }
    };
    if view.0.iter().product::<usize>() != out_n {
        return Err(format!(
            "node '{name}': operands {sa:?} and {sb:?} broadcast to {} elements, output has {out_n}",
            view.0.iter().product::<usize>()
        ));
    }
    Ok(view)
}

/// Drops size-1 dims and merges neighbours that are contiguous for both operands, so equal-shape
/// and scalar/row broadcasts run as rank 1 or 2.
fn coalesce((dims, sa, sb): View) -> View {
    let mut out: View = (Vec::new(), Vec::new(), Vec::new());
    for ((d, a), b) in dims.into_iter().zip(sa).zip(sb) {
        if d == 1 {
            continue;
        }
        if let (Some(pd), Some(pa), Some(pb)) =
            (out.0.last_mut(), out.1.last_mut(), out.2.last_mut())
        {
            if *pa == a * d && *pb == b * d {
                *pd *= d;
                *pa = a;
                *pb = b;
                continue;
            }
        }
        out.0.push(d);
        out.1.push(a);
        out.2.push(b);
    }
    if out.0.is_empty() {
        out = (vec![1], vec![0], vec![0]);
    }
    out
}

fn plan_binary(g: &Graph, node: &Node, kind: u32) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    let (Some(&a), Some(&b)) = (node.inputs.first(), node.inputs.get(1)) else {
        return Err(format!(
            "node '{name}': {} needs 2 operands",
            node.op.name()
        ));
    };
    let out = *node
        .outputs
        .first()
        .ok_or_else(|| format!("node '{name}' has no output"))?;
    let out_n = numel(g, out)?;
    numel(g, a)?;
    numel(g, b)?;
    let row_scale = matches!(node.op, Op::Mul);
    let (dims, sa, sb) = coalesce(broadcast_view(
        &g.value(a).shape,
        &g.value(b).shape,
        out_n,
        row_scale,
        name,
    )?);
    if dims.len() > MAX_RANK {
        return Err(format!(
            "node '{name}': broadcast needs rank {} after coalescing; the Metal kernel handles {MAX_RANK}",
            dims.len()
        ));
    }
    u32_of(out_n)?;
    let pack = |v: &[usize]| -> Result<[u32; MAX_RANK], String> {
        let mut p = [0u32; MAX_RANK];
        for (slot, &x) in p.iter_mut().zip(v) {
            *slot = u32_of(x)?;
        }
        Ok(p)
    };
    Ok(PlanOut::Step(
        Step::Binary {
            a,
            b,
            n: out_n,
            kind,
            rank: dims.len(),
            dims: pack(&dims)?,
            sa: pack(&sa)?,
            sb: pack(&sb)?,
            pre: Pre::NONE,
        },
        out_n,
    ))
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match *st {
        Step::Unary { x, .. } => vec![x],
        Step::Binary { a, b, .. } => vec![a, b],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, _v: ValueId) -> bool {
    match *st {
        Step::Unary { .. } | Step::Binary { .. } => false,
    }
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    match *st {
        Step::Unary { x, n, kind, a, b } => {
            if n == 0 {
                return Ok(());
            }
            #[repr(C)]
            struct Params(u32, u32, f32, f32);
            e.enc.set_compute_pipeline_state(sess.rt.pso("unary_f32")?);
            e.bind(0, x);
            e.bind(1, s.out);
            set_bytes(e.enc, 2, &Params(u32_of(n)?, kind, a, b));
            e.grid1(n);
        }
        Step::Binary {
            a,
            b,
            n,
            kind,
            rank,
            dims,
            sa,
            sb,
            pre,
        } => {
            if n == 0 {
                return Ok(());
            }
            #[repr(C)]
            struct Params {
                n: u32,
                kind: u32,
                rank: u32,
                pre_side: u32,
                pre_kind: u32,
                pre_a: f32,
                pre_b: f32,
                dims: [u32; MAX_RANK],
                sa: [u32; MAX_RANK],
                sb: [u32; MAX_RANK],
            }
            e.enc.set_compute_pipeline_state(sess.rt.pso("binary_f32")?);
            e.bind(0, a);
            e.bind(1, b);
            e.bind(2, s.out);
            let p = Params {
                n: u32_of(n)?,
                kind,
                rank: rank as u32,
                pre_side: pre.side,
                pre_kind: pre.kind,
                pre_a: pre.a,
                pre_b: pre.b,
                dims,
                sa,
                sb,
            };
            set_bytes(e.enc, 3, &p);
            e.grid1(n);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{both, rand, rel_err, rt};
    use super::*;
    use crate::graph::{Graph, Tensor, Weights};

    /// NaN/inf agree exactly; finite values agree within `tol` relative to max(1, max |want|).
    fn close(want: &[f32], got: &[f32], tol: f32, what: &str) {
        assert_eq!(want.len(), got.len(), "{what}: length");
        let (mut w, mut g) = (Vec::new(), Vec::new());
        for (i, (&a, &b)) in want.iter().zip(got).enumerate() {
            if a.is_finite() && b.is_finite() {
                w.push(a);
                g.push(b);
            } else {
                assert!(
                    (a.is_nan() && b.is_nan()) || a == b,
                    "{what}[{i}]: want {a}, got {b}"
                );
            }
        }
        let e = rel_err(&w, &g);
        assert!(e <= tol, "{what}: rel err {e} > {tol}");
    }

    fn run1(op: Op, x: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let mut g = Graph::new();
        let i = g.input("x", vec![x.len()], DType::F32);
        let y = g.add("y", op, &[i], vec![x.len()], DType::F32);
        g.output(y);
        both(&g, &Weights::new(), vec![Tensor::F32(x)])
    }

    fn run2(
        op: Op,
        (ash, bsh, osh): (&[usize], &[usize], &[usize]),
        (a, b): (Vec<f32>, Vec<f32>),
    ) -> (Vec<f32>, Vec<f32>) {
        let f = DType::F32;
        let mut g = Graph::new();
        let x = g.input("a", ash.to_vec(), f);
        let y = g.input("b", bsh.to_vec(), f);
        let z = g.add("z", op, &[x, y], osh.to_vec(), f);
        g.output(z);
        both(&g, &Weights::new(), vec![Tensor::F32(a), Tensor::F32(b)])
    }

    fn all_unary() -> Vec<(Op, f32, bool)> {
        use UnaryKind::*;
        let mut v: Vec<(Op, f32, bool)> = [
            (Exp, 10.0, false),
            (Sigmoid, 8.0, false),
            (Tanh, 4.0, false),
            (Softplus, 30.0, false),
            (Neg, 3.0, false),
            (Sqrt, 10.0, true),
            (Rsqrt, 10.0, true),
            (Abs, 3.0, false),
            (Relu, 3.0, false),
            (Relu2, 3.0, false),
            (Silu, 8.0, false),
            (GeluErf, 6.0, false),
            (GeluTanh, 6.0, false),
            (Scale(0.5), 3.0, false),
            (Scale(-2.5), 3.0, false),
            (Clamp(-0.3, 0.4), 1.0, false),
            (Clamp(0.0, 6.0), 9.0, false),
            (SoftCap(30.0), 90.0, false),
            (SoftCap(2.0), 9.0, false),
        ]
        .into_iter()
        .map(|(k, s, p)| (Op::Unary { kind: k }, s, p))
        .collect();
        v.push((Op::Relu, 3.0, false));
        v.push((Op::Silu, 8.0, false));
        v.push((Op::Gelu { tanh: false }, 6.0, false));
        v.push((Op::Gelu { tanh: true }, 6.0, false));
        v
    }

    #[test]
    fn every_unary_matches_cpu_across_sizes() {
        for (op, scale, positive) in all_unary() {
            for n in [1usize, 3, 37, 129, 1601, 70001] {
                let x: Vec<f32> = rand(n, n as u64 + 7)
                    .into_iter()
                    .map(|v| {
                        if positive {
                            (v + 1.05) * scale
                        } else {
                            v * scale
                        }
                    })
                    .collect();
                let (want, got) = run1(op.clone(), x);
                close(&want, &got, 1e-5, &format!("{op:?} n={n}"));
            }
        }
    }

    /// Saturation: large |x| must give the CPU's limits (0, 1, x), never NaN.
    #[test]
    fn unary_saturates_like_cpu() {
        use UnaryKind::*;
        let x: Vec<f32> = [
            -1e4f32, -1000.0, -100.0, -30.0, -20.0, -1.0, 0.0, 1.0, 20.0, 21.0, 30.0, 100.0,
            1000.0, 1e4,
        ]
        .into_iter()
        .collect();
        for op in [
            Op::Unary { kind: Sigmoid },
            Op::Unary { kind: Tanh },
            Op::Unary { kind: Softplus },
            Op::Unary { kind: Silu },
            Op::Unary { kind: GeluErf },
            Op::Unary { kind: GeluTanh },
            Op::Unary {
                kind: SoftCap(30.0),
            },
            Op::Unary { kind: Relu2 },
            Op::Unary {
                kind: Clamp(-5.0, 5.0),
            },
            Op::Gelu { tanh: true },
        ] {
            let (want, got) = run1(op.clone(), x.clone());
            close(&want, &got, 1e-5, &format!("{op:?} saturation"));
        }
        // Exp overflows to inf past ~88.7 on both.
        let x = vec![-200.0, -88.0, -1.0, 0.0, 1.0, 80.0, 88.0, 89.0, 200.0];
        let (want, got) = run1(Op::Unary { kind: Exp }, x);
        close(&want, &got, 1e-5, "exp overflow");
        // Rsqrt/Sqrt at zero and tiny values.
        let x = vec![0.0, 1e-30, 1e-10, 1.0, 1e10];
        for kind in [Sqrt, Rsqrt] {
            let (want, got) = run1(Op::Unary { kind: kind.clone() }, x.clone());
            close(&want, &got, 1e-5, &format!("{kind:?} edge"));
        }
    }

    fn prod(s: &[usize]) -> usize {
        s.iter().product()
    }

    /// Per-element oracle: unravel the output index, index each operand right-aligned (numpy).
    fn oracle(
        a: &[f32],
        ash: &[usize],
        b: &[f32],
        bsh: &[usize],
        osh: &[usize],
        f: fn(f32, f32) -> f32,
    ) -> Vec<f32> {
        (0..prod(osh))
            .map(|mut o| {
                let mut idx = vec![0; osh.len()];
                for d in (0..osh.len()).rev() {
                    idx[d] = o % osh[d];
                    o /= osh[d];
                }
                let at = |sh: &[usize], data: &[f32]| {
                    let off = osh.len() - sh.len();
                    let mut flat = 0;
                    for (k, &dim) in sh.iter().enumerate() {
                        flat = flat * dim + if dim == 1 { 0 } else { idx[off + k] };
                    }
                    data[flat]
                };
                f(at(ash, a), at(bsh, b))
            })
            .collect()
    }

    const KINDS: [(BinaryKind, fn(f32, f32) -> f32); 6] = [
        (BinaryKind::Add, |x, y| x + y),
        (BinaryKind::Sub, |x, y| x - y),
        (BinaryKind::Mul, |x, y| x * y),
        (BinaryKind::Div, |x, y| x / y),
        (BinaryKind::Max, f32::max),
        (BinaryKind::Min, f32::min),
    ];

    /// numpy-compatible shape pairs `(a, b, out)` with the output declared at the numpy size.
    fn np_cases() -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let c = |a: &[usize], b: &[usize], o: &[usize]| (a.to_vec(), b.to_vec(), o.to_vec());
        vec![
            c(&[1], &[1], &[1]),
            c(&[37], &[37], &[37]),
            c(&[3, 43], &[3, 43], &[3, 43]),
            c(&[129], &[1], &[129]),
            c(&[1], &[129], &[129]),
            c(&[5, 129], &[129], &[5, 129]),
            c(&[129], &[5, 129], &[5, 129]),
            c(&[129, 5], &[129, 1], &[129, 5]),
            c(&[129, 1], &[129, 5], &[129, 5]),
            c(&[6, 1], &[1, 7], &[6, 7]),
            c(&[1, 7], &[6, 1], &[6, 7]),
            // equal element counts that still broadcast to something larger
            c(&[4, 1], &[1, 4], &[4, 4]),
            c(&[1, 6], &[6, 1], &[6, 6]),
            c(&[2, 1, 3], &[1, 4, 1], &[2, 4, 3]),
            c(&[4, 3], &[4, 1], &[4, 3]),
            c(&[2, 3, 4, 5], &[1, 3, 1, 1], &[2, 3, 4, 5]),
            c(&[2, 1, 4, 5], &[3, 1, 1], &[2, 3, 4, 5]),
            c(&[1, 3, 1, 1], &[2, 3, 4, 5], &[2, 3, 4, 5]),
            c(&[4, 1, 37], &[1, 3, 1], &[4, 3, 37]),
            c(
                &[2, 1, 3, 1, 4, 1],
                &[1, 5, 1, 2, 1, 3],
                &[2, 5, 3, 2, 4, 3],
            ),
            c(
                &[1, 2, 1, 3, 1, 4, 1, 2],
                &[1, 2, 1, 3, 1, 4, 1, 2],
                &[1, 2, 1, 3, 1, 4, 1, 2],
            ),
            c(&[1, 2, 1, 3, 1, 4, 1, 2], &[2], &[1, 2, 1, 3, 1, 4, 1, 2]),
            c(&[2, 129], &[1, 129], &[2, 129]),
            c(&[3, 1, 1], &[1], &[3, 1, 1]),
        ]
    }

    #[test]
    fn binary_broadcasting_matches_numpy_oracle_and_cpu() {
        for (ash, bsh, osh) in np_cases() {
            let (a, b) = (rand(prod(&ash), 3), rand(prod(&bsh), 5));
            for (kind, f) in KINDS {
                let want = oracle(&a, &ash, &b, &bsh, &osh, f);
                for op in [
                    Op::Binary { kind },
                    if kind == BinaryKind::Add {
                        Op::Add
                    } else if kind == BinaryKind::Mul {
                        Op::Mul
                    } else {
                        Op::Binary { kind }
                    },
                ] {
                    let (cpu, metal) = run2(op.clone(), (&ash, &bsh, &osh), (a.clone(), b.clone()));
                    let what = format!("{op:?} {ash:?} {bsh:?}");
                    close(&want, &cpu, 1e-6, &format!("cpu {what}"));
                    close(&want, &metal, 1e-5, &format!("metal {what}"));
                }
            }
        }
    }

    #[test]
    fn binary_division_by_zero_and_extremes() {
        let a = vec![1.0, -1.0, 0.0, 5.0, -5.0, 1e30, 3.0, 2.0];
        let b = vec![0.0, 0.0, 0.0, 1e-30, 2.0, 1e-30, -0.0, 1e30];
        for (kind, _) in KINDS {
            let (want, got) = run2(
                Op::Binary { kind },
                (&[8], &[8], &[8]),
                (a.clone(), b.clone()),
            );
            close(&want, &got, 1e-5, &format!("{kind:?} extremes"));
        }
    }

    /// Shapes numpy cannot broadcast to the declared output fall back to the CPU's flat-length rules.
    #[test]
    fn binary_flat_length_fallbacks_match_cpu() {
        let c = |a: &[usize], b: &[usize], o: &[usize]| (a.to_vec(), b.to_vec(), o.to_vec());
        for (ash, bsh, osh) in [
            c(&[2, 6], &[3], &[2, 6]),    // b tiles over a
            c(&[3], &[2, 6], &[2, 6]),    // a tiles over b
            c(&[2, 3], &[3, 2], &[2, 3]), // equal length, different shape
            c(&[6, 1], &[1, 6], &[6]),    // equal length, declared as flat
            c(&[2, 3, 4], &[4, 3, 2], &[24]),
        ] {
            let (a, b) = (rand(prod(&ash), 11), rand(prod(&bsh), 13));
            for op in [
                Op::Add,
                Op::Mul,
                Op::Binary {
                    kind: BinaryKind::Sub,
                },
                Op::Binary {
                    kind: BinaryKind::Max,
                },
            ] {
                let (want, got) = run2(op.clone(), (&ash, &bsh, &osh), (a.clone(), b.clone()));
                close(&want, &got, 1e-5, &format!("{op:?} {ash:?} {bsh:?} flat"));
            }
        }
    }

    /// `Mul`'s legacy row scaling `[m, ..] * [m, 1]` for shapes numpy rejects.
    #[test]
    fn mul_row_scale_matches_cpu() {
        for (ash, bsh, osh) in [
            (vec![2usize, 3, 4], vec![2usize, 1], vec![2usize, 3, 4]),
            (vec![2, 1], vec![2, 3, 4], vec![2, 3, 4]),
            (vec![5, 37], vec![5, 1], vec![5, 37]),
        ] {
            let (a, b) = (rand(prod(&ash), 17), rand(prod(&bsh), 19));
            let (want, got) = run2(Op::Mul, (&ash, &bsh, &osh), (a, b));
            close(&want, &got, 1e-5, &format!("Mul {ash:?} {bsh:?}"));
        }
        // Row scaling by hand: row r of a is multiplied by b[r].
        let (a, b) = (rand(24, 1), rand(2, 2));
        let (cpu, metal) = run2(
            Op::Mul,
            (&[2, 3, 4], &[2, 1], &[2, 3, 4]),
            (a.clone(), b.clone()),
        );
        let hand: Vec<f32> = (0..24).map(|i| a[i] * b[i / 12]).collect();
        close(&hand, &cpu, 1e-6, "cpu row scale");
        close(&hand, &metal, 1e-5, "metal row scale");
    }

    #[test]
    fn large_binary_spans_many_threadgroups() {
        let n = 1_000_003;
        let (a, b) = (rand(n, 23), rand(n, 29));
        let (want, got) = run2(
            Op::Binary {
                kind: BinaryKind::Div,
            },
            (&[n], &[n], &[n]),
            (a, b),
        );
        close(&want, &got, 1e-5, "large div");
        let (a, b) = (rand(1000 * 1001, 31), rand(1001, 37));
        let (want, got) = run2(Op::Add, (&[1000, 1001], &[1001], &[1000, 1001]), (a, b));
        close(&want, &got, 1e-6, "large row add");
    }

    #[test]
    fn coalescing_reduces_rank() {
        let v = |a: &[usize], b: &[usize], o: usize| {
            coalesce(broadcast_view(a, b, o, false, "n").unwrap())
        };
        assert_eq!(v(&[4, 3, 5], &[4, 3, 5], 60), (vec![60], vec![1], vec![1]));
        assert_eq!(v(&[4, 3, 5], &[1], 60), (vec![60], vec![1], vec![0]));
        assert_eq!(v(&[4, 3], &[3], 12), (vec![4, 3], vec![3, 1], vec![0, 1]));
        assert_eq!(
            v(&[1, 2, 1, 3, 1, 4, 1, 2], &[1, 2, 1, 3, 1, 4, 1, 2], 48),
            (vec![48], vec![1], vec![1])
        );
        assert_eq!(
            v(&[2, 3, 4, 5], &[1, 3, 1, 1], 120),
            (vec![2, 3, 20], vec![60, 20, 1], vec![0, 1, 0])
        );
    }

    #[test]
    fn unbroadcastable_and_too_high_rank_report_the_node() {
        let s = |g: &Graph, inputs: Vec<Tensor>| {
            Session::new(rt(), &Weights::new())
                .unwrap()
                .run(g, inputs)
                .unwrap_err()
        };
        let f = DType::F32;
        let mut g = Graph::new();
        let (a, b) = (g.input("a", vec![5], f), g.input("b", vec![3], f));
        let z = g.add("bad_add", Op::Add, &[a, b], vec![5], f);
        g.output(z);
        let e = s(
            &g,
            vec![Tensor::F32(vec![0.0; 5]), Tensor::F32(vec![0.0; 3])],
        );
        assert!(e.contains("bad_add") && e.contains("broadcast"), "{e}");

        // Seven non-mergeable dims exceed the kernel's rank.
        let mut g = Graph::new();
        let (a, b) = (
            g.input("a", vec![2, 1, 2, 1, 2, 1, 2], f),
            g.input("b", vec![1, 2, 1, 2, 1, 2, 1], f),
        );
        let z = g.add("rank7", Op::Add, &[a, b], vec![2; 7], f);
        g.output(z);
        let e = s(
            &g,
            vec![Tensor::F32(vec![0.0; 16]), Tensor::F32(vec![0.0; 8])],
        );
        assert!(e.contains("rank7") && e.contains("rank"), "{e}");
    }

    #[test]
    fn claims_f32_only_and_every_kind() {
        use UnaryKind::*;
        let unary = [
            Exp,
            Sigmoid,
            Tanh,
            Softplus,
            Neg,
            Sqrt,
            Rsqrt,
            Abs,
            Relu,
            Relu2,
            Silu,
            GeluErf,
            GeluTanh,
            Scale(2.0),
            Clamp(0.0, 1.0),
            SoftCap(3.0),
        ];
        for k in unary {
            assert!(
                supports(&Op::Unary { kind: k.clone() }, DType::F32),
                "{k:?}"
            );
            assert!(
                !supports(&Op::Unary { kind: k.clone() }, DType::F16),
                "{k:?}"
            );
        }
        for (k, _) in KINDS {
            assert!(supports(&Op::Binary { kind: k }, DType::F32));
            assert!(!supports(&Op::Binary { kind: k }, DType::BF16));
            assert!(!supports(&Op::Binary { kind: k }, DType::U32));
        }
        for op in [
            Op::Relu,
            Op::Silu,
            Op::Gelu { tanh: true },
            Op::Gelu { tanh: false },
            Op::Add,
            Op::Mul,
        ] {
            assert!(supports(&op, DType::F32), "{op:?}");
            assert!(!supports(&op, DType::F16), "{op:?}");
        }
        assert!(!supports(&Op::Softmax { axis: -1 }, DType::F32));
        assert!(!supports(&Op::SwiGlu, DType::F32));
    }

    /// A unary whose only reader is a binary becomes that binary's `pre` map (either operand side,
    /// equal-shape and broadcast operands, widths off the threadgroup size); a second reader keeps it a
    /// separate step. Every case matches the CPU executor.
    #[test]
    fn unary_feeding_binary_fuses_and_matches_cpu() {
        use super::super::super::{fuse_chains, plan, Step as CoreStep};
        let f = DType::F32;
        let shapes: [(&[usize], &[usize]); 3] =
            [(&[7, 33], &[7, 33]), (&[5, 300], &[300]), (&[257], &[1])];
        let fused_steps = |g: &Graph| {
            let (steps, aliases) = plan(g).unwrap();
            let steps = fuse_chains(g, steps, &aliases);
            let pre = steps
                .iter()
                .filter(|s| {
                    matches!(&s.op, CoreStep::Ext(super::super::ExtStep::Elementwise(Step::Binary { pre, .. })) if pre.side != 0)
                })
                .count();
            (steps.len(), pre)
        };
        for (op, scale, positive) in all_unary() {
            for (si, (ash, bsh)) in shapes.into_iter().enumerate() {
                for (bin, unary_side, shared) in [
                    (Op::Mul, 1, false),
                    (Op::Binary { kind: BinaryKind::Sub }, 1, false),
                    (Op::Binary { kind: BinaryKind::Sub }, 2, false),
                    (Op::Add, 1, true),
                ] {
                    let ushape: Vec<usize> = if unary_side == 1 { ash } else { bsh }.to_vec();
                    let other_shape: Vec<usize> = if unary_side == 1 { bsh } else { ash }.to_vec();
                    let mut g = Graph::new();
                    let x = g.input("x", ushape.clone(), f);
                    let o = g.input("o", other_shape.clone(), f);
                    let u = g.add("u", op.clone(), &[x], ushape.clone(), f);
                    let ins = if unary_side == 1 { [u, o] } else { [o, u] };
                    let z = g.add("z", bin, &ins, ash.to_vec(), f);
                    // `shared`: u is also read by the final Add, so it must stay materialised.
                    let shared = shared && ushape == ash;
                    let out = if shared {
                        g.add("t", Op::Add, &[z, u], ash.to_vec(), f)
                    } else {
                        z
                    };
                    g.output(out);
                    let case = format!("{op:?} shape {si} side {unary_side} shared {shared}");
                    assert_eq!(fused_steps(&g), if shared { (3, 0) } else { (1, 1) }, "{case}");
                    let xs: Vec<f32> = rand(prod(&ushape), 11 + si as u64)
                        .into_iter()
                        .map(|v| if positive { (v + 1.05) * scale } else { v * scale })
                        .collect();
                    let inputs = vec![Tensor::F32(xs), Tensor::F32(rand(prod(&other_shape), 99))];
                    let (want, got) = both(&g, &Weights::new(), inputs);
                    close(&want, &got, 1e-5, &case);
                }
            }
        }
    }
}
