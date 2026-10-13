//! Metal lowering group `matmul`: planning, encoding and kernels for its ops.
//! Kernels live in `matmul.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! `MatMul { transpose_a, transpose_b }` at F32: `[..batch, m, k] x [..batch, k, n] -> [..batch, m, n]`
//! (transposes swap the last two dims of that operand). An operand's leading dims must equal the
//! output's or be all 1 (stride-0 broadcast).

use super::super::Counted;
use super::super::{u32_of, Planned, Runtime, Session};
use super::{Enc, PlanOut};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use metal::MTLSize;
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("matmul.metal");
/// Compute functions in `matmul.metal`.
pub(super) const KERNELS: &[&str] = &[
    "matmul_sg32",
    "matmul_sg64",
    "matmul_small_n",
    "matmul_small_k",
];

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    /// `p` = `MatmulParams` (m, n, k, a_row, a_col, b_row, b_col, a_bs, b_bs); `batch` on grid z.
    MatMul {
        a: ValueId,
        b: ValueId,
        p: [u32; 9],
        batch: u32,
    },
}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    matches!(op, Op::MatMul { .. }) && dtype == DType::F32
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    let Op::MatMul {
        transpose_a: ta,
        transpose_b: tb,
    } = node.op
    else {
        return None;
    };
    Some(plan_matmul(g, node, ta, tb))
}

fn plan_matmul(g: &Graph, node: &Node, ta: bool, tb: bool) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    let &[a, b] = &node.inputs[..] else {
        return Err(format!("node '{name}': MatMul needs operands [a, b]"));
    };
    let out = *node
        .outputs
        .first()
        .ok_or_else(|| format!("node '{name}' has no output"))?;
    if [a, b, out].iter().any(|&v| g.value(v).dtype != DType::F32) {
        return Err(format!(
            "node '{name}': Metal MatMul needs f32 operands and output"
        ));
    }
    let (sa, sb) = (&g.value(a).shape, &g.value(b).shape);
    if sa.len() < 2 || sb.len() < 2 {
        return Err(format!(
            "node '{name}': MatMul operands {sa:?} x {sb:?} must be rank >= 2"
        ));
    }
    let (ba, bb) = (&sa[..sa.len() - 2], &sb[..sb.len() - 2]);
    let (ar, ac, br, bc) = (
        sa[sa.len() - 2],
        sa[sa.len() - 1],
        sb[sb.len() - 2],
        sb[sb.len() - 1],
    );
    let (m, k) = if ta { (ac, ar) } else { (ar, ac) };
    let (k2, n) = if tb { (bc, br) } else { (br, bc) };
    if k != k2 {
        return Err(format!(
            "node '{name}': MatMul inner dims {k} != {k2} ({sa:?} x {sb:?})"
        ));
    }
    // Output batch = the larger operand batch; the other must match it (modulo leading 1s) or be all 1s.
    let prod = |d: &[usize]| d.iter().product::<usize>();
    let trim = |d: &[usize]| {
        d.iter()
            .copied()
            .skip_while(|&x| x == 1)
            .collect::<Vec<_>>()
    };
    let batch = prod(ba).max(prod(bb));
    let full = trim(if prod(ba) >= prod(bb) { ba } else { bb });
    let stride = |d: &[usize], size: usize| -> Result<u32, String> {
        if prod(d) == 1 {
            Ok(0)
        } else if trim(d) == full {
            u32_of(size)
        } else {
            Err(format!(
                "node '{name}': MatMul batch dims {ba:?} and {bb:?} are neither equal nor broadcast from 1"
            ))
        }
    };
    if m == 0 || n == 0 || batch == 0 {
        return Err(format!("node '{name}': empty MatMul output"));
    }
    let (a_bs, b_bs) = (stride(ba, ar * ac)?, stride(bb, br * bc)?);
    let (ac32, bc32) = (u32_of(ac)?, u32_of(bc)?);
    let (a_row, a_col) = if ta { (1, ac32) } else { (ac32, 1) };
    let (b_row, b_col) = if tb { (1, bc32) } else { (bc32, 1) };
    let p = [
        u32_of(m)?,
        u32_of(n)?,
        u32_of(k)?,
        a_row,
        a_col,
        b_row,
        b_col,
        a_bs,
        b_bs,
    ];
    let step = Step::MatMul {
        a,
        b,
        p,
        batch: u32_of(batch)?,
    };
    Ok(PlanOut::Step(step, batch * m * n))
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match *st {
        Step::MatMul { a, b, .. } => vec![a, b],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, _v: ValueId) -> bool {
    match *st {
        Step::MatMul { .. } => false,
    }
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    match *st {
        Step::MatMul { a, b, p, batch } => {
            e.bind(0, a);
            e.bind(1, b);
            e.bind(2, s.out);
            let mut bytes = [0u8; 36];
            for (i, v) in p.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
            }
            e.bind_param(3, &bytes);
            launch(&sess.rt, e.enc, p, batch)
        }
    }
}

/// Picks the kernel for the shape, sets the pipeline and dispatches (buffers 0..=3 already bound).
fn launch(rt: &Runtime, enc: &Counted, p: [u32; 9], batch: u32) -> Result<(), String> {
    let [m, n, _, _, _, _, b_col, _, _] = p;
    let (kernel, grid, tg) = if m <= 8 {
        if b_col == 1 {
            ("matmul_small_n", (n.div_ceil(32), 1), 256)
        } else {
            ("matmul_small_k", (n.div_ceil(8), 1), 256)
        }
    } else {
        // ponytail: tile choice by threadgroup count only; tune per shape if a model needs it.
        let big = (n.div_ceil(64) as u64) * (m.div_ceil(64) as u64) * batch as u64 >= 32;
        let (name, t) = if big {
            ("matmul_sg64", 64)
        } else {
            ("matmul_sg32", 32)
        };
        (name, (n.div_ceil(t), m.div_ceil(t)), 128)
    };
    enc.set_compute_pipeline_state(rt.pso(kernel)?);
    enc.dispatch_thread_groups(
        MTLSize::new(grid.0 as u64, grid.1 as u64, batch as u64),
        MTLSize::new(tg, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{both, rand, rel_err, rt};
    use crate::graph::{DType, Graph, Op, Tensor, Weights};

    /// `[..ba, ar, ac] x [..bb, br, bc]` through Metal vs the CPU executor.
    fn check(
        ba: &[usize],
        bb: &[usize],
        m: usize,
        n: usize,
        k: usize,
        ta: bool,
        tb: bool,
        seed: u64,
    ) {
        let dims = |batch: &[usize], r: usize, c: usize, t: bool| {
            let mut s = batch.to_vec();
            s.extend(if t { [c, r] } else { [r, c] });
            s
        };
        let (sa, sb) = (dims(ba, m, k, ta), dims(bb, k, n, tb));
        let full = if ba.iter().product::<usize>() >= bb.iter().product::<usize>() {
            ba
        } else {
            bb
        };
        let mut so = full.to_vec();
        so.extend([m, n]);
        let mut g = Graph::new();
        let a = g.input("a", sa.clone(), DType::F32);
        let b = g.input("b", sb.clone(), DType::F32);
        let o = g.add(
            "mm",
            Op::MatMul {
                transpose_a: ta,
                transpose_b: tb,
            },
            &[a, b],
            so,
            DType::F32,
        );
        g.output(o);
        let inputs = vec![
            Tensor::F32(rand(sa.iter().product(), seed)),
            Tensor::F32(rand(sb.iter().product(), seed + 1)),
        ];
        let (want, got) = both(&g, &Weights::new(), inputs);
        let e = rel_err(&want, &got);
        assert!(
            e < 1e-4,
            "{ba:?}x{bb:?} m{m} n{n} k{k} ta={ta} tb={tb}: rel err {e:e}"
        );
    }

    #[test]
    fn matmul_all_transposes_and_edges() {
        let mut n_cases = 0;
        // m <= 8 hits the small kernels, m > 8 the simdgroup ones; odd sizes straddle tile edges.
        for &(m, n, k) in &[
            (1, 1, 1),
            (1, 3, 5),
            (1, 37, 129),
            (1, 130, 64),
            (3, 33, 7),
            (8, 64, 40),
            (9, 9, 9),
            (7, 31, 8),
            (16, 16, 16),
            (33, 65, 17),
            (37, 37, 37),
            (64, 64, 64),
            (65, 63, 129),
            (129, 129, 129),
            (100, 1, 50),
            (1, 200, 1),
            (130, 70, 3),
        ] {
            for ta in [false, true] {
                for tb in [false, true] {
                    check(&[], &[], m, n, k, ta, tb, (m * 31 + n * 7 + k) as u64);
                    n_cases += 1;
                }
            }
        }
        assert_eq!(n_cases, 68);
    }

    #[test]
    fn matmul_batched_and_broadcast() {
        for ta in [false, true] {
            for tb in [false, true] {
                // equal batch dims, several ranks
                check(&[3], &[3], 17, 21, 19, ta, tb, 1);
                check(&[2, 4], &[2, 4], 40, 9, 33, ta, tb, 2);
                check(&[5], &[5], 1, 70, 64, ta, tb, 3);
                check(&[2, 3], &[2, 3], 5, 6, 7, ta, tb, 4);
                // broadcast: one operand without (or with all-1) batch dims
                check(&[4], &[], 12, 20, 10, ta, tb, 5);
                check(&[], &[4], 12, 20, 10, ta, tb, 6);
                check(&[3], &[1], 1, 33, 40, ta, tb, 7);
                check(&[1, 1], &[2, 3], 9, 9, 9, ta, tb, 8);
                check(&[1, 3], &[3], 70, 70, 70, ta, tb, 9);
                check(&[6], &[], 1, 129, 37, ta, tb, 10);
            }
        }
    }

    #[test]
    fn matmul_rejects_bad_shapes() {
        for (sa, sb, want) in [
            (vec![2, 3, 4], vec![3, 4, 5], "neither equal"),
            (vec![3, 4], vec![5, 6], "inner dims"),
            (vec![4], vec![4, 5], "rank >= 2"),
        ] {
            let mut g = Graph::new();
            let a = g.input("a", sa, DType::F32);
            let b = g.input("b", sb, DType::F32);
            let o = g.add(
                "mm",
                Op::MatMul {
                    transpose_a: false,
                    transpose_b: false,
                },
                &[a, b],
                vec![1],
                DType::F32,
            );
            g.output(o);
            let Err(e) = super::plan_matmul(&g, &g.nodes[0], false, false) else {
                panic!("accepted")
            };
            assert!(e.contains(want) && e.contains("'mm'"), "{e}");
        }
    }

    /// The F32 `Linear` forms (bias, Gelu epilogue, residual) the core fuses must match the CPU.
    #[test]
    fn linear_f32_epilogue_forms() {
        for m in [1, 5, 37] {
            let (k, n) = (21, 19);
            for (bias, gelu, resid) in [
                (false, false, false),
                (true, false, false),
                (false, true, false),
                (true, true, false),
                (true, false, true),
                (false, false, true),
            ] {
                let mut g = Graph::new();
                let x = g.input("x", vec![m, k], DType::F32);
                let w = g.weight("w", vec![n, k], DType::F32);
                let mut ins = vec![x, w];
                if bias {
                    ins.push(g.weight("b", vec![n], DType::F32));
                }
                let mut y = g.add("lin", Op::Linear, &ins, vec![m, n], DType::F32);
                let mut inputs = vec![Tensor::F32(rand(m * k, 1))];
                if gelu {
                    y = g.add(
                        "gelu",
                        Op::Gelu { tanh: false },
                        &[y],
                        vec![m, n],
                        DType::F32,
                    );
                }
                if resid {
                    let r = g.input("r", vec![m, n], DType::F32);
                    y = g.add("res", Op::Add, &[y, r], vec![m, n], DType::F32);
                    inputs.push(Tensor::F32(rand(m * n, 4)));
                }
                g.output(y);
                let mut ws = Weights::new();
                ws.insert("w", Tensor::F32(rand(n * k, 2)));
                ws.insert("b", Tensor::F32(rand(n, 3)));
                let (want, got) = both(&g, &ws, inputs);
                let e = rel_err(&want, &got);
                assert!(e < 1e-4, "m{m} bias={bias} gelu={gelu} res={resid}: {e:e}");
            }
        }
    }

    #[test]
    fn matmul_in_graph_with_weight() {
        // a weight operand and a chained consumer, through the planner's liveness.
        let (m, k, n) = (13, 40, 24);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w = g.weight("w", vec![n, k], DType::F32);
        let y = g.add(
            "y",
            Op::MatMul {
                transpose_a: false,
                transpose_b: true,
            },
            &[x, w],
            vec![m, n],
            DType::F32,
        );
        let z = g.add(
            "z",
            Op::MatMul {
                transpose_a: true,
                transpose_b: false,
            },
            &[y, y],
            vec![n, n],
            DType::F32,
        );
        g.output(z);
        let mut ws = Weights::new();
        ws.insert("w", Tensor::F32(rand(n * k, 5)));
        let (want, got) = both(&g, &ws, vec![Tensor::F32(rand(m * k, 6))]);
        assert!(rel_err(&want, &got) < 1e-4);
    }

    /// GPU time (µs) of one `[m, k] x [k, n]` dispatch: `Runtime::timed`, best of 3 command buffers.
    fn gpu_us(m: usize, n: usize, k: usize, tb: bool) -> f64 {
        let rt = rt();
        let buf = |len: usize| {
            rt.device.new_buffer_with_data(
                rand(len, len as u64).as_ptr().cast(),
                (len * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        let (a, b, c) = (
            buf(m * k),
            buf(k * n),
            rt.device.new_buffer(
                (m * n * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ),
        );
        let p: [u32; 9] = if tb {
            [m as u32, n as u32, k as u32, k as u32, 1, 1, k as u32, 0, 0]
        } else {
            [m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0]
        };
        rt.timed(20, |enc| {
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&b), 0);
            enc.set_buffer(2, Some(&c), 0);
            super::super::super::set_bytes(enc, 3, &p);
            super::launch(&rt, enc, p, 1)
        })
        .unwrap()
    }

    #[test]
    #[ignore = "perf report: cargo test --release --lib matmul_perf -- --ignored --nocapture"]
    fn matmul_perf() {
        for (m, n, k, tb) in [
            (512, 512, 512, false),
            (512, 512, 512, true),
            (2048, 2048, 2048, false),
            (1, 4096, 4096, false),
            (1, 4096, 4096, true),
        ] {
            let us = gpu_us(m, n, k, tb);
            let gb = ((k * n + m * k + m * n) * 4) as f64 / us / 1e3;
            println!(
                "matmul {m}x{n}x{k} tb={tb}: {:.0} GFLOP/s, {gb:.0} GB/s, {us:.0} us",
                2.0 * (m * n * k) as f64 / us / 1e3
            );
        }
    }
}
