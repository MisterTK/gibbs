//! Output shape and dtype of an [`Op`] from its operands' shapes and dtypes. The CPU executor
//! (`cpu.rs`) is the semantic reference; operand orders are documented on [`Op`].

use super::{DType, MegakernelKind, Op};

type Operand<'a> = (&'a [usize], DType);

/// Output `(shape, dtype)` of `op` applied to `inputs` (in operand order). Errors name the op and
/// the operand shapes. Activations are F32 (weights of any storage, quantized included, yield F32);
/// `TopK` yields U32 and data-movement ops keep a U32 operand's dtype.
pub fn infer(op: &Op, inputs: &[Operand]) -> Result<(Vec<usize>, DType), String> {
    let fail = |msg: String| {
        let shapes: Vec<_> = inputs.iter().map(|(s, d)| format!("{s:?} {d:?}")).collect();
        format!("{}: {msg} (operands: [{}])", op.name(), shapes.join(", "))
    };
    let need = |n: usize| -> Result<(), String> {
        if inputs.len() < n {
            return Err(fail(format!("needs {n} operands, got {}", inputs.len())));
        }
        Ok(())
    };
    need(1)?;
    if let Some(k) = inputs.iter().position(|(s, _)| s.is_empty()) {
        return Err(fail(format!("operand {k} has an unknown (empty) shape")));
    }
    let s = |k: usize| inputs[k].0;
    let numel = |k: usize| s(k).iter().product::<usize>();
    let last = |k: usize| *s(k).last().unwrap();
    let lead = |k: usize| s(k)[..s(k).len() - 1].to_vec();
    let with_last = |k: usize, d: usize| [lead(k), vec![d]].concat();
    // Data movement keeps U32 (ids, positions); everything else computes in F32.
    let moved = |k: usize| match inputs[k].1 {
        DType::U32 => DType::U32,
        _ => DType::F32,
    };
    let f32 = |shape: Vec<usize>| Ok((shape, DType::F32));
    let axis = |a: isize, rank: usize| -> Result<usize, String> {
        let ax = if a < 0 { a + rank as isize } else { a };
        if !(0..rank as isize).contains(&ax) {
            return Err(fail(format!("axis {a} out of range for rank {rank}")));
        }
        Ok(ax as usize)
    };

    match op {
        Op::Embedding => {
            need(2)?;
            f32(vec![numel(1), last(0)])
        }
        Op::Linear => {
            need(2)?;
            let [out, inn] = s(1)[..] else {
                return Err(fail("weight is not [out, in]".into()));
            };
            if last(0) != inn {
                return Err(fail(format!("x width {} != weight in {inn}", last(0))));
            }
            if inputs.len() > 2 && numel(2) != out {
                return Err(fail(format!("bias is not [{out}]")));
            }
            f32(with_last(0, out))
        }
        Op::MatMul {
            transpose_a,
            transpose_b,
        } => {
            need(2)?;
            let (a, b) = (s(0), s(1));
            if a.len() < 2 || b.len() < 2 {
                return Err(fail("operands must be rank >= 2".into()));
            }
            let (ar, ac, br, bc) = (
                a[a.len() - 2],
                a[a.len() - 1],
                b[b.len() - 2],
                b[b.len() - 1],
            );
            let (m, k) = if *transpose_a { (ac, ar) } else { (ar, ac) };
            let (k2, n) = if *transpose_b { (bc, br) } else { (br, bc) };
            if k != k2 {
                return Err(fail(format!("inner dims {k} != {k2}")));
            }
            let mut out = broadcast(&a[..a.len() - 2], &b[..b.len() - 2])
                .ok_or_else(|| fail("batch dims do not broadcast".into()))?;
            out.extend([m, n]);
            f32(out)
        }
        Op::RmsNorm { .. } => {
            let d = if inputs.len() > 1 { numel(1) } else { last(0) };
            if d == 0 || numel(0) % d != 0 {
                return Err(fail("gamma does not tile x".into()));
            }
            f32(s(0).to_vec())
        }
        Op::LayerNorm { .. } => {
            need(2)?;
            let d = numel(1);
            if d == 0 || numel(0) % d != 0 || (inputs.len() > 2 && numel(2) != d) {
                return Err(fail("gamma/beta do not tile x".into()));
            }
            f32(s(0).to_vec())
        }
        Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
            need(3)?;
            let c = numel(1);
            let groups = match op {
                Op::GroupNorm { groups, .. } => *groups,
                _ => c,
            };
            if c == 0 || numel(2) != c || groups == 0 || c % groups != 0 {
                return Err(fail(format!("gamma/beta do not fit {groups} groups")));
            }
            super::cpu::channel_layout(s(0), c).map_err(fail)?;
            f32(s(0).to_vec())
        }
        Op::Rope {
            head_dim,
            rotary_dim,
            ..
        } => {
            need(2)?;
            let freqs_ok = inputs.len() < 3 || numel(2) == rotary_dim / 2;
            if !freqs_ok || *head_dim == 0 || last(0) % head_dim != 0 {
                return Err(fail(format!(
                    "x is not [n, heads*{head_dim}] or inv_freq is not [{}]",
                    rotary_dim / 2
                )));
            }
            f32(s(0).to_vec())
        }
        Op::Attention {
            heads, kv_heads, ..
        } => {
            need(3)?;
            let hd = last(0);
            if *heads == 0 || *kv_heads == 0 || heads % kv_heads != 0 || hd % heads != 0 {
                return Err(fail(format!(
                    "q width {hd} does not fit {heads}/{kv_heads} heads"
                )));
            }
            let kvw = kv_heads * (hd / heads);
            if last(1) != kvw || numel(1) != numel(2) {
                return Err(fail(format!("k/v are not [nk, {kvw}]")));
            }
            f32(s(0).to_vec())
        }
        Op::SwiGlu | Op::GeGlu => {
            need(2)?;
            if numel(0) != numel(1) {
                return Err(fail("gate and up differ in size".into()));
            }
            f32(s(0).to_vec())
        }
        Op::Unary { .. } | Op::Softmax { .. } | Op::L2Normalize | Op::CausalConv1dCached { .. } => {
            f32(s(0).to_vec())
        }
        Op::Binary { .. } => {
            need(2)?;
            let out =
                broadcast(s(0), s(1)).ok_or_else(|| fail("operands do not broadcast".into()))?;
            f32(out)
        }
        Op::CachedAttention {
            heads,
            head_dim,
            v_head_dim,
            ..
        }
        | Op::PagedCachedAttention {
            heads,
            head_dim,
            v_head_dim,
            ..
        } => {
            need(3)?;
            if last(0) != heads * head_dim {
                return Err(fail(format!("q is not [n, {}]", heads * head_dim)));
            }
            f32(with_last(0, heads * v_head_dim))
        }
        Op::DeltaRule {
            key_heads,
            val_heads,
            key_dim,
            val_dim,
        } => {
            need(5)?;
            if last(0) != key_heads * key_dim {
                return Err(fail(format!("q is not [n, {}]", key_heads * key_dim)));
            }
            f32(with_last(0, val_heads * val_dim))
        }
        Op::Gather { axis: a } => {
            need(2)?;
            let ax = axis(*a, s(0).len())?;
            Ok(([&s(0)[..ax], s(1), &s(0)[ax + 1..]].concat(), moved(0)))
        }
        Op::Scatter { .. } => {
            need(3)?;
            Ok((s(0).to_vec(), moved(0)))
        }
        Op::Pad { padding, .. } => {
            let (r, k) = (s(0).len(), padding.len() / 2);
            if padding.len() % 2 != 0 || k > r {
                return Err(fail(format!("padding {padding:?} does not fit rank {r}")));
            }
            let mut out = s(0).to_vec();
            for j in 0..k {
                out[r - k + j] += padding[2 * j] + padding[2 * j + 1];
            }
            Ok((out, moved(0)))
        }
        Op::AvgPool2d { kernel, stride } => {
            let r = s(0).len();
            let (h, w) = (s(0)[r.saturating_sub(2)], last(0));
            if r < 3 || h < kernel[0] || w < kernel[1] || stride.contains(&0) {
                return Err(fail(format!("{kernel:?}/{stride:?} pool does not fit")));
            }
            let mut out = s(0)[..r - 2].to_vec();
            out.extend([
                (h - kernel[0]) / stride[0] + 1,
                (w - kernel[1]) / stride[1] + 1,
            ]);
            f32(out)
        }
        // The op does not say how many trailing axes it scales (the CPU executor reads it from the
        // declared output, `cpu::upsample_nsp`); this infers the 2-D (last two axes) form.
        Op::Upsample { scale, .. } => {
            let r = s(0).len();
            if r < 2 || *scale == 0 {
                return Err(fail(format!("cannot upsample by {scale}")));
            }
            let mut out = s(0).to_vec();
            out[r - 2..].iter_mut().for_each(|d| *d *= scale);
            f32(out)
        }
        Op::Conv1d {
            stride,
            padding: p,
            dilation,
            groups,
        } => conv(
            inputs,
            1,
            (
                [1, 1, *stride],
                [0, 0, p[0]],
                [0, 0, p[1]],
                [1, 1, *dilation],
                *groups,
            ),
        )
        .map_err(fail),
        Op::Conv2d {
            stride,
            padding: p,
            dilation,
            groups,
        } => conv(
            inputs,
            2,
            (
                [1, stride[0], stride[1]],
                [0, p[0], p[1]],
                [0, p[2], p[3]],
                [1, dilation[0], dilation[1]],
                *groups,
            ),
        )
        .map_err(fail),
        Op::Conv3d {
            stride,
            padding: p,
            dilation,
            groups,
        } => conv(
            inputs,
            3,
            (
                *stride,
                [p[0], p[1], p[2]],
                [p[3], p[4], p[5]],
                *dilation,
                *groups,
            ),
        )
        .map_err(fail),
        Op::TopK { k } => {
            if *k == 0 || *k > last(0) {
                return Err(fail(format!("k={k} not in 1..={}", last(0))));
            }
            Ok((with_last(0, *k), DType::U32))
        }
        Op::GatherCols => {
            need(2)?;
            let (rows, cols) = (numel(0) / last(0).max(1), last(0));
            if cols == 0 || (rows != 1 && rows != numel(1) / last(1).max(1)) {
                return Err(fail("x is neither one row nor as many rows as idx".into()));
            }
            f32(s(1).to_vec())
        }
        Op::MoeLinear { experts, top_k } => {
            need(3)?;
            let [rows, inn] = s(1)[..] else {
                return Err(fail("bank is not [experts * out, in]".into()));
            };
            let slots = numel(2);
            let xr = numel(0) / inn.max(1);
            if *experts == 0
                || *top_k == 0
                || rows % experts != 0
                || last(0) != inn
                || (xr != slots && xr * top_k != slots)
            {
                return Err(fail(format!(
                    "does not fit {experts} experts, top_k {top_k}"
                )));
            }
            f32(vec![slots, rows / experts])
        }
        Op::RelPosBias { heads, .. } => {
            need(3)?;
            f32(vec![*heads, numel(1), numel(2)])
        }
        Op::StateReorder => {
            need(2)?;
            f32(s(0).to_vec())
        }
        Op::Reshape { shape } => match shape {
            Some(t) if t.iter().product::<usize>() == numel(0) => Ok((t.clone(), moved(0))),
            Some(t) => Err(fail(format!("target {t:?} changes the element count"))),
            None => Err(fail("the target shape is a runtime tensor".into())),
        },
        Op::Transpose { perm } => {
            let r = s(0).len();
            let mut seen = vec![false; r];
            if perm.len() != r
                || perm
                    .iter()
                    .any(|&p| p >= r || std::mem::replace(&mut seen[p], true))
            {
                return Err(fail(format!("{perm:?} is not a permutation of rank {r}")));
            }
            Ok((perm.iter().map(|&p| s(0)[p]).collect(), moved(0)))
        }
        Op::Concat { axis: a } => {
            let r = s(0).len();
            let ax = axis(*a, r)?;
            let mut out = s(0).to_vec();
            for k in 1..inputs.len() {
                if s(k).len() != r || (0..r).any(|d| d != ax && s(k)[d] != out[d]) {
                    return Err(fail(format!("operands disagree off axis {ax}")));
                }
                out[ax] += s(k)[ax];
            }
            Ok((out, moved(0)))
        }
        Op::Slice {
            axis: a,
            start,
            end,
        } => {
            let ax = axis(*a, s(0).len())?;
            let n = s(0)[ax] as i64;
            let clamp = |i: i64| if i < 0 { (n + i).max(0) } else { i.min(n) };
            let mut out = s(0).to_vec();
            out[ax] = (clamp(*end) - clamp(*start)).max(0) as usize;
            Ok((out, moved(0)))
        }
        Op::Squeeze { axes } => {
            let r = s(0).len();
            let drop: Vec<usize> = if axes.is_empty() {
                (0..r).filter(|&d| s(0)[d] == 1).collect()
            } else {
                axes.iter().map(|&a| axis(a, r)).collect::<Result<_, _>>()?
            };
            if drop.iter().any(|&d| s(0)[d] != 1) {
                return Err(fail(format!("axes {axes:?} are not all size 1")));
            }
            let out = (0..r)
                .filter(|d| !drop.contains(d))
                .map(|d| s(0)[d])
                .collect();
            Ok((out, moved(0)))
        }
        Op::Unsqueeze { axes } => {
            let r = s(0).len() + axes.len();
            let ins: Vec<usize> = axes.iter().map(|&a| axis(a, r)).collect::<Result<_, _>>()?;
            let mut rest = s(0).iter();
            let out = (0..r)
                .map(|d| {
                    if ins.contains(&d) {
                        1
                    } else {
                        *rest.next().unwrap_or(&1)
                    }
                })
                .collect();
            Ok((out, moved(0)))
        }
        Op::Tile { reps } => {
            let r = s(0).len().max(reps.len());
            let pad = |v: &[usize]| [vec![1; r - v.len()], v.to_vec()].concat();
            let out = pad(s(0))
                .iter()
                .zip(pad(reps))
                .map(|(a, b)| a * b)
                .collect();
            Ok((out, moved(0)))
        }
        Op::Reduce {
            axes, keep_dims, ..
        } => {
            let r = s(0).len();
            if axes.is_empty() {
                return Err(fail("no axes".into()));
            }
            let red: Vec<usize> = axes.iter().map(|&a| axis(a, r)).collect::<Result<_, _>>()?;
            let out: Vec<usize> = (0..r)
                .filter_map(|d| match (red.contains(&d), keep_dims) {
                    (false, _) => Some(s(0)[d]),
                    (true, true) => Some(1),
                    (true, false) => None,
                })
                .collect();
            f32(if out.is_empty() { vec![1] } else { out })
        }
        Op::MeanPool => {
            need(2)?;
            if numel(0) != numel(1) * last(0) {
                return Err(fail("x is not [mask rows, d]".into()));
            }
            f32(vec![last(0)])
        }
        // A residual block: `x + f(x)`, shaped like `x`.
        Op::Megakernel(desc) => {
            let d = match desc.kind {
                MegakernelKind::Mlp { hidden_dim, .. }
                | MegakernelKind::Attention { hidden_dim, .. } => hidden_dim,
            };
            if last(0) != d {
                return Err(fail(format!("x width is not hidden_dim {d}")));
            }
            f32(s(0).to_vec())
        }

        // ---- Legacy forms of `Unary`/`Binary` (being deleted; drop with the variants). ----
        Op::Silu | Op::Relu | Op::Gelu { .. } => f32(s(0).to_vec()),
        Op::Add | Op::Mul => {
            need(2)?;
            let out =
                broadcast(s(0), s(1)).ok_or_else(|| fail("operands do not broadcast".into()))?;
            f32(out)
        }
    }
}

/// numpy right-aligned broadcast of two shapes.
fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() >= r {
            s[i + s.len() - r]
        } else {
            1
        }
    };
    (0..r)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y || y == 1 => Some(x),
            (1, y) => Some(y),
            _ => None,
        })
        .collect()
}

/// `conv_nd` geometry: `x [batch.., c_in, spatial(nd)]`, `w [c_out, c_in/groups, k(nd)]`;
/// stride/pad/dilation right-aligned into 3 spatial dims.
#[allow(clippy::type_complexity)]
fn conv(
    inputs: &[Operand],
    nd: usize,
    (stride, pb, pe, dil, groups): ([usize; 3], [usize; 3], [usize; 3], [usize; 3], usize),
) -> Result<(Vec<usize>, DType), String> {
    if inputs.len() < 2 {
        return Err("needs x and w".into());
    }
    let (xs, ws) = (inputs[0].0, inputs[1].0);
    if xs.len() < nd + 1 || ws.len() != nd + 2 {
        return Err(format!("x/w ranks do not fit a {nd}-d conv"));
    }
    let c_in = xs[xs.len() - nd - 1];
    let (c_out, cpg) = (ws[0], ws[1]);
    if groups == 0
        || c_in != cpg * groups
        || c_out % groups != 0
        || stride.contains(&0)
        || dil.contains(&0)
    {
        return Err(format!(
            "c_in {c_in} does not fit weight with {groups} groups"
        ));
    }
    if inputs.len() > 2 && inputs[2].0.iter().product::<usize>() != c_out {
        return Err(format!("bias is not [{c_out}]"));
    }
    let mut out = xs[..xs.len() - nd - 1].to_vec();
    out.push(c_out);
    for j in 0..nd {
        let a = 3 - nd + j;
        let (inp, k) = (xs[xs.len() - nd + j], ws[2 + j]);
        let span = dil[a] * (k - 1) + 1;
        if inp + pb[a] + pe[a] < span {
            return Err("kernel larger than padded input".into());
        }
        out.push((inp + pb[a] + pe[a] - span) / stride[a] + 1);
    }
    Ok((out, DType::F32))
}

#[cfg(test)]
thread_local! {
    static CAPTURED: std::cell::RefCell<Option<Vec<super::Graph>>> = const { std::cell::RefCell::new(None) };
}

/// Test hook on every graph a program compiles or runs reshaped and a Metal session runs: kept
/// while a [`CheckCompiled`] guard is live on this thread.
#[cfg(test)]
pub(crate) fn note(g: &super::Graph) {
    CAPTURED.with(|c| {
        if let Some(v) = c.borrow_mut().as_mut() {
            v.push(g.clone());
        }
    });
}

/// Nodes of `g` whose declared output shape/dtype differs from [`infer`] on their operands. Nodes
/// with an unknown (empty) shape are skipped. An `Upsample` also passes when the CPU executor
/// accepts its declared output as the 3-D form (`cpu::upsample_nsp`), which the op cannot express.
#[cfg(test)]
pub(crate) fn shape_mismatches(g: &super::Graph) -> Vec<String> {
    let mut bad = Vec::new();
    for n in &g.nodes {
        let out = g.value(n.outputs[0]);
        let ops: Vec<Operand> = n
            .inputs
            .iter()
            .map(|&v| (g.value(v).shape.as_slice(), g.value(v).dtype))
            .collect();
        if out.shape.is_empty() || ops.iter().any(|(s, _)| s.is_empty()) {
            continue;
        }
        if let (Op::Upsample { scale, .. }, [(inp, _)]) = (&n.op, ops.as_slice()) {
            if super::cpu::upsample_nsp(inp, &out.shape, *scale).is_ok() && out.dtype == DType::F32
            {
                continue;
            }
        }
        match infer(&n.op, &ops) {
            Ok((s, d)) if s == out.shape && d == out.dtype => {}
            Ok((s, d)) => bad.push(format!(
                "node '{}': declared {:?} {:?}, inferred {s:?} {d:?}",
                n.name, out.shape, out.dtype
            )),
            Err(e) => bad.push(format!(
                "node '{}': declared {:?} {:?}, {e}",
                n.name, out.shape, out.dtype
            )),
        }
    }
    bad
}

/// Asserts [`infer`] reproduces every node's declared shape and dtype in `g`.
#[cfg(test)]
#[track_caller]
pub(crate) fn assert_shapes(g: &super::Graph) {
    assert!(!g.nodes.is_empty(), "empty graph");
    let bad = shape_mismatches(g);
    assert!(
        bad.is_empty(),
        "{} shape mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// Test guard: when dropped (outside a panic), [`assert_shapes`] on every graph [`note`]d on this
/// thread since [`CheckCompiled::new`]; at least one must have been.
#[cfg(test)]
pub(crate) struct CheckCompiled(Option<Vec<super::Graph>>);

#[cfg(test)]
impl CheckCompiled {
    pub(crate) fn new() -> Self {
        Self(CAPTURED.with(|c| c.borrow_mut().replace(Vec::new())))
    }
}

#[cfg(test)]
impl Drop for CheckCompiled {
    fn drop(&mut self) {
        let gs = CAPTURED
            .with(|c| std::mem::replace(&mut *c.borrow_mut(), self.0.take()))
            .unwrap_or_default();
        if std::thread::panicking() {
            return;
        }
        assert!(!gs.is_empty(), "no graph was compiled");
        gs.iter().for_each(assert_shapes);
    }
}

/// [`CheckCompiled`] around `f`.
#[cfg(test)]
pub(crate) fn assert_compiled_shapes<R>(f: impl FnOnce() -> R) -> R {
    let _check = CheckCompiled::new();
    f()
}

#[cfg(test)]
mod tests {
    use super::super::{DType, Graph, Op};

    /// `Graph::push` appends the inferred shape/dtype and refuses inconsistent operands by name.
    #[test]
    fn push_infers_or_names_the_mismatch() {
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 8], DType::F32);
        let w = g.weight("w", vec![5, 8], DType::BF16);
        let y = g.push("lin", Op::Linear, &[x, w]).unwrap();
        assert_eq!(
            (&g.value(y).shape[..], g.value(y).dtype),
            (&[3, 5][..], DType::F32)
        );
        let ids = g.push("top", Op::TopK { k: 2 }, &[y]).unwrap();
        assert_eq!(
            (&g.value(ids).shape[..], g.value(ids).dtype),
            (&[3, 2][..], DType::U32)
        );
        let e = g.push("bad", Op::Linear, &[y, w]).unwrap_err();
        assert!(
            e.contains("node 'bad': Linear") && e.contains("[5, 8] BF16"),
            "{e}"
        );
        assert_eq!(g.nodes.len(), 2);
    }
}
