//! Metal lowering group `conv`: planning, encoding and kernels for its ops.
//! Kernels live in `conv.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Lowers `Conv1d` / `Conv2d` / `Conv3d` (stride, padding, dilation, groups, depthwise, non-square
//! kernels, optional bias; leading dims of `x` are batch) and `AvgPool2d` (leading dims are planes).
//! `Conv1d` runs on the 2-d kernel with H = 1. `Conv3d` padding is `[front, top, left, back,
//! bottom, right]` (begin dims, then end dims, like `Conv2d`'s `[top, left, bottom, right]`).

use super::{Enc, PlanOut};
use super::super::{Planned, Session};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use metal::MTLSize;
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("conv.metal");
/// Compute functions in `conv.metal`.
pub(super) const KERNELS: &[&str] =
    &["conv2d_gen_f32", "conv3d_gen_f32", "avgpool2d_f32", "conv_igemm_q2", "conv_igemm_q4"];

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    /// Direct convolution; `p` is the kernel's `ConvNdParams` / `Conv3dNdParams` as u32 words.
    Conv {
        x: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        kernel: &'static str,
        p: Vec<u32>,
        total: usize,
    },
    AvgPool {
        x: ValueId,
        p: [u32; 9],
        total: usize,
    },
    /// Implicit-GEMM convolution (groups == 1): `p` is `ConvGemmParams`, `tiles` the (pixel, channel,
    /// batch) threadgroup grid.
    ConvGemm {
        x: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        kernel: &'static str,
        p: [u32; 24],
        tiles: [u64; 3],
    },
}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    dtype == DType::F32
        && match op {
            Op::Conv2d { stride, dilation, groups, .. } => {
                *groups > 0 && !stride.contains(&0) && !dilation.contains(&0)
            }
            Op::Conv1d { stride, dilation, groups, .. } => *groups > 0 && *stride > 0 && *dilation > 0,
            Op::Conv3d { stride, dilation, groups, .. } => {
                *groups > 0 && !stride.contains(&0) && !dilation.contains(&0)
            }
            Op::AvgPool2d { kernel, stride } => !kernel.contains(&0) && !stride.contains(&0),
            _ => false,
        }
}

fn shape_of(g: &Graph, v: ValueId, what: &str, name: &str) -> Result<Vec<usize>, String> {
    let s = &g.value(v).shape;
    if s.is_empty() {
        return Err(format!("node '{name}': {what} has no static shape"));
    }
    Ok(s.clone())
}

fn out_dim(name: &str, inp: usize, pb: usize, pe: usize, k: usize, s: usize, d: usize) -> Result<usize, String> {
    let span = d * (k - 1) + 1;
    if k == 0 || inp + pb + pe < span {
        return Err(format!(
            "node '{name}': kernel {k} (dilation {d}) is larger than padded input {}",
            inp + pb + pe
        ));
    }
    Ok((inp + pb + pe - span) / s + 1)
}

/// Plans an `nd`-spatial-dim convolution; per-dim arrays are right-aligned into 3 (unused dims
/// neutral: stride 1, pad 0, dilation 1).
#[allow(clippy::type_complexity)]
fn plan_conv(
    g: &Graph,
    node: &Node,
    nd: usize,
    (stride, pb, pe, dil, groups): ([usize; 3], [usize; 3], [usize; 3], [usize; 3], usize),
) -> Result<(Step, usize), String> {
    let name = &node.name;
    let arg = |k: usize| {
        node.inputs
            .get(k)
            .copied()
            .ok_or_else(|| format!("node '{name}': {} needs operand {k}", node.op.name()))
    };
    let (x, w) = (arg(0)?, arg(1)?);
    let bias = node.inputs.get(2).copied();
    let (xs, ws) = (shape_of(g, x, "x", name)?, shape_of(g, w, "weight", name)?);
    if xs.len() < nd + 1 || ws.len() != nd + 2 {
        return Err(format!("node '{name}': x {xs:?} / weight {ws:?} do not fit a {nd}-d convolution"));
    }
    let (mut inp, mut k) = ([1usize; 3], [1usize; 3]);
    inp[3 - nd..].copy_from_slice(&xs[xs.len() - nd..]);
    k[3 - nd..].copy_from_slice(&ws[2..]);
    let c_in = xs[xs.len() - nd - 1];
    let n: usize = xs[..xs.len() - nd - 1].iter().product();
    let (c_out, cpg) = (ws[0], ws[1]);
    if groups == 0 || c_in != cpg * groups || c_out % groups != 0 {
        return Err(format!("node '{name}': c_in {c_in}, weight {ws:?}, groups {groups} are inconsistent"));
    }
    if let Some(b) = bias {
        if g.value(b).shape.iter().product::<usize>() != c_out {
            return Err(format!("node '{name}': bias is not [{c_out}]"));
        }
    }
    let mut out = [1usize; 3];
    for a in 3 - nd..3 {
        out[a] = out_dim(name, inp[a], pb[a], pe[a], k[a], stride[a], dil[a])?;
    }
    let total = n * c_out * out.iter().product::<usize>();
    let in_total = n * c_in * inp.iter().product::<usize>();
    if total.max(in_total) > u32::MAX as usize {
        return Err(format!("node '{name}': tensor too large for 32-bit indexing"));
    }
    let u = |v: usize| v as u32;
    let has_bias = u(bias.is_some() as usize);
    if groups == 1 {
        let (kk, ss) = (c_in * k.iter().product::<usize>(), out.iter().product::<usize>());
        let p = [
            u(n), u(c_in), u(inp[0]), u(inp[1]), u(inp[2]), u(c_out), u(k[0]), u(k[1]), u(k[2]),
            u(stride[0]), u(stride[1]), u(stride[2]), u(pb[0]), u(pb[1]), u(pb[2]), u(dil[0]), u(dil[1]),
            u(dil[2]), u(out[0]), u(out[1]), u(out[2]), has_bias, u(kk), u(ss),
        ];
        // Same tile choice as the MatMul kernels: small tiles when the big ones would leave the GPU idle.
        let tiles = |t: usize| [ss.div_ceil(t) as u64, c_out.div_ceil(t) as u64, n as u64];
        let big = tiles(64).iter().product::<u64>() >= 32;
        let (kernel, grid) = if big { ("conv_igemm_q4", tiles(64)) } else { ("conv_igemm_q2", tiles(32)) };
        return Ok((Step::ConvGemm { x, w, bias, kernel, p, tiles: grid }, total));
    }
    let (kernel, p) = if nd == 3 {
        (
            "conv3d_gen_f32",
            vec![
                u(n), u(c_in), u(inp[0]), u(inp[1]), u(inp[2]), u(c_out), u(k[0]), u(k[1]), u(k[2]),
                u(stride[0]), u(stride[1]), u(stride[2]), u(pb[0]), u(pb[1]), u(pb[2]), u(dil[0]),
                u(dil[1]), u(dil[2]), u(groups), u(out[0]), u(out[1]), u(out[2]), has_bias,
            ],
        )
    } else {
        (
            "conv2d_gen_f32",
            vec![
                u(n), u(c_in), u(inp[1]), u(inp[2]), u(c_out), u(k[1]), u(k[2]), u(stride[1]),
                u(stride[2]), u(pb[1]), u(pb[2]), u(dil[1]), u(dil[2]), u(groups), u(out[1]),
                u(out[2]), has_bias,
            ],
        )
    };
    Ok((Step::Conv { x, w, bias, kernel, p, total }, total))
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    let r = match &node.op {
        Op::Conv2d { stride, padding: p, dilation, groups } => plan_conv(
            g,
            node,
            2,
            ([1, stride[0], stride[1]], [0, p[0], p[1]], [0, p[2], p[3]], [1, dilation[0], dilation[1]], *groups),
        ),
        Op::Conv1d { stride, padding: p, dilation, groups } => plan_conv(
            g,
            node,
            1,
            ([1, 1, *stride], [0, 0, p[0]], [0, 0, p[1]], [1, 1, *dilation], *groups),
        ),
        Op::Conv3d { stride, padding: p, dilation, groups } => plan_conv(
            g,
            node,
            3,
            (*stride, [p[0], p[1], p[2]], [p[3], p[4], p[5]], *dilation, *groups),
        ),
        Op::AvgPool2d { kernel, stride } => plan_pool(g, node, *kernel, *stride),
        _ => return None,
    };
    Some(r.map(|(s, n)| PlanOut::Step(s, n)))
}

fn plan_pool(g: &Graph, node: &Node, kernel: [usize; 2], stride: [usize; 2]) -> Result<(Step, usize), String> {
    let name = &node.name;
    let x = *node.inputs.first().ok_or_else(|| format!("node '{name}': AvgPool2d needs operand 0"))?;
    let xs = shape_of(g, x, "x", name)?;
    if xs.len() < 3 {
        return Err(format!("node '{name}': AvgPool2d needs [.., c, h, w], got {xs:?}"));
    }
    let (h, w) = (xs[xs.len() - 2], xs[xs.len() - 1]);
    let planes: usize = xs[..xs.len() - 2].iter().product();
    let (oh, ow) = (out_dim(name, h, 0, 0, kernel[0], stride[0], 1)?, out_dim(name, w, 0, 0, kernel[1], stride[1], 1)?);
    let total = planes * oh * ow;
    if planes * h * w > u32::MAX as usize {
        return Err(format!("node '{name}': tensor too large for 32-bit indexing"));
    }
    let u = |v: usize| v as u32;
    let p = [u(planes), u(h), u(w), u(kernel[0]), u(kernel[1]), u(stride[0]), u(stride[1]), u(oh), u(ow)];
    Ok((Step::AvgPool { x, p, total }, total))
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match st {
        Step::Conv { x, w, bias, .. } => [Some(*x), Some(*w), *bias].into_iter().flatten().collect(),
        Step::AvgPool { x, .. } => vec![*x],
        Step::ConvGemm { x, w, bias, .. } => [Some(*x), Some(*w), *bias].into_iter().flatten().collect(),
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(_st: &Step, _v: ValueId) -> bool {
    false
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    match st {
        Step::Conv { x, w, bias, kernel, p, total } => {
            e.enc.set_compute_pipeline_state(sess.rt.pso(kernel)?);
            e.bind(0, *x);
            e.bind(1, *w);
            e.bind(2, bias.unwrap_or(*x));
            e.bind(3, s.out);
            e.enc.set_bytes(4, (p.len() * 4) as u64, p.as_ptr() as *const std::ffi::c_void);
            e.grid1(*total);
        }
        Step::ConvGemm { x, w, bias, kernel, p, tiles } => {
            e.enc.set_compute_pipeline_state(sess.rt.pso(kernel)?);
            e.bind(0, *x);
            e.bind(1, *w);
            e.bind(2, bias.unwrap_or(*x));
            e.bind(3, s.out);
            super::super::set_bytes(e.enc, 4, p);
            e.enc.dispatch_thread_groups(MTLSize::new(tiles[0], tiles[1], tiles[2]), MTLSize::new(128, 1, 1));
        }
        Step::AvgPool { x, p, total } => {
            e.enc.set_compute_pipeline_state(sess.rt.pso("avgpool2d_f32")?);
            e.bind(0, *x);
            e.bind(1, s.out);
            super::super::set_bytes(e.enc, 2, p);
            e.grid1(*total);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{both, rand, rel_err, rt};
    use super::super::super::Session;
    use crate::graph::{DType, Graph, Op, Tensor, Weights};

    /// Output extent of one conv dim, derived independently of `out_dim`.
    fn od(i: usize, pb: usize, pe: usize, k: usize, s: usize, d: usize) -> usize {
        (i + pb + pe - d * (k - 1) - 1) / s + 1
    }

    /// Runs `op` on random data through the CPU executor and Metal and compares them.
    fn check(label: &str, op: Op, xs: &[usize], ws: &[usize], bias: bool, out: &[usize]) {
        let mut g = Graph::new();
        let x = g.input("x", xs.to_vec(), DType::F32);
        let w = g.weight("w", ws.to_vec(), DType::F32);
        let mut ins = vec![x, w];
        let mut wt = Weights::new();
        wt.insert("w", Tensor::F32(rand(ws.iter().product(), 7)));
        if bias {
            ins.push(g.weight("b", vec![ws[0]], DType::F32));
            wt.insert("b", Tensor::F32(rand(ws[0], 11)));
        }
        let y = g.add("conv", op, &ins, out.to_vec(), DType::F32);
        g.output(y);
        let inp = vec![Tensor::F32(rand(xs.iter().product(), 3))];
        let (want, got) = both(&g, &wt, inp);
        let e = rel_err(&want, &got);
        assert!(e < 1e-4, "{label}: rel err {e}");
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        label: &str,
        xs: [usize; 3],
        oc: usize,
        k: [usize; 2],
        stride: [usize; 2],
        pad: [usize; 4],
        dil: [usize; 2],
        groups: usize,
        bias: bool,
    ) {
        let [c, h, w] = xs;
        let out = [
            oc,
            od(h, pad[0], pad[2], k[0], stride[0], dil[0]),
            od(w, pad[1], pad[3], k[1], stride[1], dil[1]),
        ];
        let op = Op::Conv2d { stride, padding: pad, dilation: dil, groups };
        check(label, op, &xs, &[oc, c / groups, k[0], k[1]], bias, &out);
    }

    #[test]
    fn conv2d_matches_cpu() {
        conv2d("3x3 p1 bias", [3, 7, 9], 5, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1, true);
        conv2d("odd 37 k5 s2", [2, 37, 37], 4, [5, 5], [2, 2], [2, 2, 2, 2], [1, 1], 1, false);
        conv2d("dilation 2", [4, 11, 13], 6, [3, 3], [1, 1], [2, 2, 2, 2], [2, 2], 1, true);
        conv2d("dilation 2 stride 2", [3, 16, 15], 2, [3, 3], [2, 2], [0, 0, 0, 0], [2, 2], 1, true);
        conv2d("groups 2", [4, 9, 9], 6, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 2, true);
        conv2d("groups 3 stride 2", [6, 10, 7], 9, [3, 3], [2, 2], [1, 1, 1, 1], [1, 1], 3, false);
        conv2d("depthwise", [8, 12, 11], 8, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 8, true);
        conv2d("depthwise s2 asym pad", [8, 13, 9], 8, [3, 3], [2, 2], [0, 1, 2, 0], [1, 1], 8, false);
        conv2d("depthwise mult 2", [4, 8, 8], 8, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 4, true);
        conv2d("1x5", [3, 6, 21], 4, [1, 5], [1, 1], [0, 2, 0, 2], [1, 1], 1, true);
        conv2d("3x1 s(2,1)", [3, 9, 6], 4, [3, 1], [2, 1], [1, 0, 1, 0], [1, 1], 1, false);
        conv2d("2x3 stride (1,2) dil (1,2)", [2, 8, 17], 3, [2, 3], [1, 2], [1, 2, 0, 2], [1, 2], 1, true);
        conv2d("1x1", [5, 7, 3], 6, [1, 1], [1, 1], [0, 0, 0, 0], [1, 1], 1, true);
        conv2d("wide 129", [1, 3, 129], 2, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1, true);
        conv2d("single pixel", [3, 1, 1], 2, [1, 1], [1, 1], [0, 0, 0, 0], [1, 1], 1, true);
        conv2d("kernel = padded input", [2, 3, 3], 2, [5, 5], [1, 1], [1, 1, 1, 1], [1, 1], 1, true);
        conv2d("big channels", [37, 5, 5], 33, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1, true);
        // Enough tiles for the 64-wide GEMM kernel, with partial edge tiles in both dims.
        conv2d("igemm q4", [8, 40, 41], 70, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1], 1, true);
        conv2d("igemm q4 stride 2 asym pad", [5, 67, 66], 65, [3, 3], [2, 2], [0, 1, 1, 0], [1, 1], 1, false);
        conv2d("igemm k tail (K = 3*25 + 1 channel)", [3, 33, 33], 130, [5, 5], [1, 1], [2, 2, 2, 2], [1, 1], 1, true);
    }

    #[test]
    fn conv2d_batched_matches_cpu() {
        let (xs, ws) = ([2, 4, 7, 7], [6, 2, 3, 3]);
        let op = Op::Conv2d { stride: [2, 2], padding: [1, 1, 1, 1], dilation: [1, 1], groups: 2 };
        check("batch 2 groups 2", op, &xs, &ws, true, &[2, 6, 4, 4]);
    }

    /// groups = 1 through the implicit-GEMM kernel: batch on the grid z axis, several K steps with a tail
    /// (c_in 37 x 9 = 333), full 64-wide tiles on both axes, and a UNet-like 160-channel layer.
    #[test]
    fn conv2d_igemm_batched_and_wide_matches_cpu() {
        let op = Op::Conv2d { stride: [1, 1], padding: [1; 4], dilation: [1, 1], groups: 1 };
        check("batch 3 k tail", op.clone(), &[3, 37, 20, 22], &[70, 37, 3, 3], true, &[3, 70, 20, 22]);
        check("unet-like 160ch", op.clone(), &[160, 24, 24], &[160, 160, 3, 3], true, &[160, 24, 24]);
        check("1x1 batch 2 no bias", Op::Conv2d { stride: [1, 1], padding: [0; 4], dilation: [1, 1], groups: 1 },
            &[2, 96, 17, 19], &[130, 96, 1, 1], false, &[2, 130, 17, 19]);
    }

    /// The old CPU conv (`nn::conv2d`) agrees with the generalized reference on its own domain.
    #[test]
    fn cpu_reference_matches_nn_conv2d() {
        let (c, h, w, oc, k, s, p) = (3, 9, 11, 4, 3, 2, 1);
        let (x, wt, b) = (rand(c * h * w, 1), rand(oc * c * k * k, 2), rand(oc, 3));
        let (want, oh, ow) = crate::model::nn::conv2d(&x, c, h, w, &wt, Some(&b), oc, k, s, p);
        let mut g = Graph::new();
        let xi = g.input("x", vec![c, h, w], DType::F32);
        let wi = g.weight("w", vec![oc, c, k, k], DType::F32);
        let bi = g.weight("b", vec![oc], DType::F32);
        let y = g.add(
            "conv",
            Op::Conv2d { stride: [s, s], padding: [p; 4], dilation: [1, 1], groups: 1 },
            &[xi, wi, bi],
            vec![oc, oh, ow],
            DType::F32,
        );
        g.output(y);
        let mut ws = Weights::new();
        ws.insert("w", Tensor::F32(wt));
        ws.insert("b", Tensor::F32(b));
        let got = crate::graph::cpu::run_stateless(&g, &ws, vec![Tensor::F32(x)])
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        assert!(rel_err(&want, &got) < 1e-5);
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        label: &str,
        xs: &[usize],
        oc: usize,
        k: usize,
        stride: usize,
        pad: [usize; 2],
        dil: usize,
        groups: usize,
        bias: bool,
    ) {
        let (c, l) = (xs[xs.len() - 2], xs[xs.len() - 1]);
        let mut out = xs[..xs.len() - 2].to_vec();
        out.extend([oc, od(l, pad[0], pad[1], k, stride, dil)]);
        let op = Op::Conv1d { stride, padding: pad, dilation: dil, groups };
        check(label, op, xs, &[oc, c / groups, k], bias, &out);
    }

    #[test]
    fn conv1d_matches_cpu() {
        conv1d("k3 p1", &[3, 37], 5, 3, 1, [1, 1], 1, 1, true);
        conv1d("l=1 k=1", &[4, 1], 3, 1, 1, [0, 0], 1, 1, true);
        conv1d("asym pad", &[2, 20], 3, 5, 1, [4, 0], 1, 1, false);
        conv1d("stride 2 dil 2", &[4, 129], 6, 3, 2, [2, 2], 2, 1, true);
        conv1d("groups 2", &[4, 33], 6, 3, 1, [1, 1], 1, 2, true);
        conv1d("depthwise k7", &[8, 129], 8, 7, 1, [3, 3], 1, 8, true);
        conv1d("depthwise k31 s2", &[6, 100], 6, 31, 2, [15, 15], 1, 6, false);
        conv1d("batch 2", &[2, 4, 17], 5, 3, 1, [1, 1], 1, 1, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn conv3d(
        label: &str,
        xs: &[usize],
        oc: usize,
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 6],
        dil: [usize; 3],
        groups: usize,
        bias: bool,
    ) {
        let r = xs.len();
        let c = xs[r - 4];
        let mut out = xs[..r - 4].to_vec();
        out.push(oc);
        for a in 0..3 {
            out.push(od(xs[r - 3 + a], pad[a], pad[3 + a], k[a], stride[a], dil[a]));
        }
        let op = Op::Conv3d { stride, padding: pad, dilation: dil, groups };
        check(label, op, xs, &[oc, c / groups, k[0], k[1], k[2]], bias, &out);
    }

    #[test]
    fn conv3d_matches_cpu() {
        conv3d("3x3x3 p1", &[2, 4, 5, 6], 3, [3, 3, 3], [1; 3], [1; 6], [1; 3], 1, true);
        conv3d("causal front pad", &[2, 5, 4, 4], 3, [3, 3, 3], [1; 3], [2, 1, 1, 0, 1, 1], [1; 3], 1, true);
        conv3d("temporal (3,1,1)", &[4, 6, 3, 3], 4, [3, 1, 1], [1; 3], [1, 0, 0, 1, 0, 0], [1; 3], 1, true);
        conv3d("stride 2", &[2, 7, 9, 5], 3, [3, 3, 3], [2; 3], [1; 6], [1; 3], 1, false);
        conv3d("dilation 2", &[2, 7, 8, 9], 2, [3, 3, 3], [1; 3], [2; 6], [2; 3], 1, true);
        conv3d("groups 2", &[4, 4, 5, 5], 6, [2, 3, 3], [1; 3], [0, 1, 1, 0, 1, 1], [1; 3], 2, true);
        conv3d("depthwise", &[6, 5, 6, 7], 6, [3, 3, 3], [1, 2, 2], [1; 6], [1; 3], 6, true);
        conv3d("1x1x1x1", &[3, 1, 1, 1], 2, [1, 1, 1], [1; 3], [0; 6], [1; 3], 1, true);
        conv3d("batch 2", &[2, 3, 4, 5, 5], 4, [3, 3, 3], [1; 3], [1; 6], [1; 3], 1, true);
        conv3d("wide 129", &[1, 2, 2, 129], 2, [1, 1, 3], [1; 3], [0, 0, 1, 0, 0, 1], [1; 3], 1, false);
        conv3d("igemm q4", &[6, 5, 20, 20], 70, [3, 3, 3], [1; 3], [2, 1, 1, 0, 1, 1], [1; 3], 1, true);
    }

    /// Direct (groups = 2) vs implicit-GEMM (groups = 1) throughput on a 128-channel 3x3 conv at
    /// 256x256: `cargo test --release --lib conv_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn conv_throughput() {
        for (groups, c, k) in [(2usize, 128usize, 3usize), (1, 128, 3), (1, 320, 3), (1, 128, 1)] {
            let (h, w) = (256usize, 256usize);
            let ws = [c, c / groups, k, k];
            let mut g = Graph::new();
            let x = g.input("x", vec![c, h, w], DType::F32);
            let wi = g.weight("w", ws.to_vec(), DType::F32);
            let y = g.add(
                "conv",
                Op::Conv2d { stride: [1, 1], padding: [k / 2; 4], dilation: [1, 1], groups },
                &[x, wi],
                vec![c, h, w],
                DType::F32,
            );
            g.output(y);
            let mut wt = Weights::new();
            wt.insert("w", Tensor::F32(rand(ws.iter().product(), 7)));
            let sess = Session::new(rt(), &wt).unwrap();
            let plan = sess.prepare_plan(&g).unwrap();
            let data = rand(c * h * w, 3);
            let inp = || vec![Tensor::F32(data.clone())];
            sess.run_prepared(&g, &plan, None, inp()).unwrap();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                sess.run_prepared(&g, &plan, None, inp()).unwrap();
            }
            let dt = t.elapsed().as_secs_f64() / 5.0;
            let flops = 2.0 * (c * c / groups * k * k * h * w) as f64;
            eprintln!("conv c={c} k={k} groups={groups}: {:.1} ms, {:.0} GFLOP/s", dt * 1e3, flops / dt / 1e9);
        }
    }

    fn pool(label: &str, xs: &[usize], kernel: [usize; 2], stride: [usize; 2]) {
        let r = xs.len();
        let mut out = xs[..r - 2].to_vec();
        out.extend([od(xs[r - 2], 0, 0, kernel[0], stride[0], 1), od(xs[r - 1], 0, 0, kernel[1], stride[1], 1)]);
        let mut g = Graph::new();
        let x = g.input("x", xs.to_vec(), DType::F32);
        let y = g.add("pool", Op::AvgPool2d { kernel, stride }, &[x], out, DType::F32);
        g.output(y);
        let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(rand(xs.iter().product(), 5))]);
        let e = rel_err(&want, &got);
        assert!(e < 1e-5, "{label}: rel err {e}");
    }

    #[test]
    fn avgpool2d_matches_cpu() {
        pool("2x2 s2", &[3, 8, 8], [2, 2], [2, 2]);
        pool("odd 3x3 s2", &[3, 37, 41], [3, 3], [2, 2]);
        pool("2x3 s(1,2)", &[2, 9, 17], [2, 3], [1, 2]);
        pool("global 1x129", &[1, 1, 129], [1, 129], [1, 1]);
        pool("k1", &[4, 5, 5], [1, 1], [1, 1]);
        pool("batch 2", &[2, 3, 9, 9], [3, 3], [3, 3]);
    }

    #[test]
    fn rejects_inconsistent_shapes_and_unsupported_dtypes() {
        let op = Op::Conv2d { stride: [1, 1], padding: [0; 4], dilation: [1, 1], groups: 1 };
        assert!(super::supports(&op, DType::F32));
        assert!(!super::supports(&op, DType::F16));
        assert!(!super::supports(&Op::Conv2d { stride: [0, 1], padding: [0; 4], dilation: [1, 1], groups: 1 }, DType::F32));
        // weight in-channels (2) do not match x channels (3)
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 4, 4], DType::F32);
        let w = g.weight("w", vec![2, 2, 3, 3], DType::F32);
        let y = g.add("bad", op, &[x, w], vec![2, 2, 2], DType::F32);
        g.output(y);
        let mut wt = Weights::new();
        wt.insert("w", Tensor::F32(vec![0.0; 36]));
        let r = Session::new(rt(), &wt).unwrap().run(&g, vec![Tensor::F32(vec![0.0; 48])]);
        assert!(r.is_err_and(|e| e.contains("bad")));
    }
}
