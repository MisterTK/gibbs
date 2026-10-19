//! Metal lowering group `deltanet`: one decode token of a gated delta-rule recurrent layer in one
//! dispatch. Kernels live in `deltanet.metal`.
//!
//! No op is planned here; [`fuse`] runs after node lowering and replaces the post-projection chain
//! `CausalConv1dCached -> SiLU -> last-axis Slice q|k|v -> DeltaRule(beta, log_decay) -> RmsNorm(·, g[val_dim])
//! -> Mul(·, gate)` with `beta = Sigmoid(b)`, `log_decay = w * Softplus([a + bias])` and
//! `gate = SiLU|Sigmoid(z)` by one [`DeltaNetStep`]. Gate pieces another step already absorbed (a GEMV
//! epilogue) stay inputs. Exact up to fp reassociation, so on by default; `GIBBS_DELTANET_STATE=f16|bf16`
//! stores the recurrent state in half / bfloat (f32 arithmetic).

use super::super::{fusion_policy, numel, set_bytes, FusionPolicy, Planned, Session};
use super::{Enc, ExtStep};
use crate::graph::{BinaryKind, Graph, Node, Op, UnaryKind, ValueId, ValueKind};
use metal::MTLSize;
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("deltanet.metal");
/// Compute functions in `deltanet.metal`.
pub(super) const KERNELS: &[&str] = &[
    "deltanet_step_f32",
    "deltanet_step_f16",
    "deltanet_step_bf16",
];

/// Threads per threadgroup and the kernel's fixed-size limits (`DN_*` in `deltanet.metal`).
const THREADS: usize = 1024;
const MAX_KD: usize = 256;
const MAX_VD: usize = 256;
const MAX_RPT: usize = 16;

/// Storage type of the recurrent state; arithmetic is always f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateDtype {
    F32,
    F16,
    BF16,
}

impl StateDtype {
    fn kernel(self) -> &'static str {
        match self {
            StateDtype::F32 => "deltanet_step_f32",
            StateDtype::F16 => "deltanet_step_f16",
            StateDtype::BF16 => "deltanet_step_bf16",
        }
    }
}

// ponytail: a half/bfloat state is opaque to the host: `StateStore::snapshot`/`restore` round-trip its
// bytes, but `sync_metal_to_cpu` or a CPU-seeded state read it as f32. Hence opt-in only; an unfused
// stateful `DeltaRule` (which reads f32) fails the plan instead (`check_state_dtype`).
/// `GIBBS_DELTANET_STATE=f16|bf16` (read once); default f32.
static STATE_DTYPE: std::sync::LazyLock<StateDtype> =
    std::sync::LazyLock::new(|| match std::env::var("GIBBS_DELTANET_STATE").as_deref() {
        Ok("f16" | "half") => StateDtype::F16,
        Ok("bf16") => StateDtype::BF16,
        _ => StateDtype::F32,
    });

/// The fused decode step (`deltanet_step_*`); writes `conv_state` and `state` in place.
#[derive(Debug, Clone)]
pub struct DeltaNetStep {
    pub x: ValueId,
    pub conv_w: ValueId,
    pub conv_state: ValueId,
    pub z: ValueId,
    pub b: ValueId,
    pub a: ValueId,
    pub dt_bias: Option<ValueId>,
    pub neg_a: ValueId,
    pub state: ValueId,
    pub norm_w: ValueId,
    pub kernel: usize,
    pub dilation: usize,
    pub key_heads: usize,
    pub val_heads: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    pub eps: f32,
    /// 0: `z` is the gate, 1: SiLU(z), 2: sigmoid(z).
    pub z_act: u32,
    pub beta_sigmoid: bool,
    pub state_dtype: StateDtype,
}

/// `(cw, rgroups, rpt)` thread layout of the state, `None` when the kernel's limits are exceeded.
fn layout(kd: usize, vd: usize) -> Option<(usize, usize, usize)> {
    if kd == 0 || vd == 0 || kd > MAX_KD || vd > MAX_VD {
        return None;
    }
    let cw = vd.next_multiple_of(32);
    let rgroups = THREADS / cw;
    let rpt = kd.div_ceil(rgroups);
    (rpt <= MAX_RPT).then_some((cw, rgroups, rpt))
}

fn is_silu(op: &Op) -> bool {
    matches!(
        op,
        Op::Silu
            | Op::Unary {
                kind: UnaryKind::Silu
            }
    )
}
fn is_sigmoid(op: &Op) -> bool {
    matches!(
        op,
        Op::Unary {
            kind: UnaryKind::Sigmoid
        }
    )
}
fn is_mul(op: &Op) -> bool {
    matches!(
        op,
        Op::Mul
            | Op::Binary {
                kind: BinaryKind::Mul
            }
    )
}
fn is_add(op: &Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Binary {
                kind: BinaryKind::Add
            }
    )
}

/// Replaces every matched decode (`n == 1`) chain by one fused step at the chain's last step.
pub(in super::super) fn fuse(g: &Graph, steps: &mut Vec<Planned>) -> Result<(), String> {
    if fusion_policy() != FusionPolicy::Full {
        return check_state_dtype(steps, *STATE_DTYPE);
    }
    let mut readers: HashMap<ValueId, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for &v in &n.inputs {
            readers.entry(v).or_default().push(i);
        }
    }
    let step_of: HashMap<ValueId, usize> =
        steps.iter().enumerate().map(|(i, s)| (s.out, i)).collect();
    let mut found = Vec::new();
    for di in 0..steps.len() {
        if let Some(m) = match_chain(g, steps, di, &readers, &step_of, *STATE_DTYPE) {
            found.push(m);
        }
    }
    let mut drop = Vec::new();
    for (at, rm, st) in found {
        // `rm[0]` is the DeltaRule step: profiles and plan dumps show the fused step under its name.
        steps[at] = Planned {
            name: steps[rm[0]].name.clone(),
            out: steps[at].out,
            extra_outs: Vec::new(),
            reads: reads(&st),
            op: super::super::Step::Ext(ExtStep::DeltaNet(st)),
        };
        drop.extend(rm);
    }
    drop.sort_unstable();
    for i in drop.into_iter().rev() {
        steps.remove(i);
    }
    check_state_dtype(steps, *STATE_DTYPE)
}

/// A non-f32 state is only written by the fused kernel: any stateful `DeltaRule` left unfused would
/// read it as f32.
fn check_state_dtype(steps: &[Planned], dtype: StateDtype) -> Result<(), String> {
    match steps.iter().find(|s| {
        matches!(s.op, super::super::Step::DeltaRule { state: Some(_), .. })
    }) {
        Some(s) if dtype != StateDtype::F32 => Err(format!(
            "node '{}': GIBBS_DELTANET_STATE={dtype:?} needs every stateful DeltaRule fused, and this one is not",
            s.name
        )),
        _ => Ok(()),
    }
}

/// Matches the chain around the `DeltaRule` step `di`: `(index of its last step, other step indices
/// starting with `di`, step)`.
fn match_chain(
    g: &Graph,
    steps: &[Planned],
    di: usize,
    readers: &HashMap<ValueId, Vec<usize>>,
    step_of: &HashMap<ValueId, usize>,
    state_dtype: StateDtype,
) -> Option<(usize, Vec<usize>, DeltaNetStep)> {
    let super::super::Step::DeltaRule {
        q,
        k,
        v,
        beta,
        log_decay,
        state: Some(state),
        n: 1,
        key_heads: kh,
        val_heads: vh,
        key_dim: kd,
        val_dim: vd,
    } = steps[di].op
    else {
        return None;
    };
    let is_state = |v: ValueId| matches!(g.value(v).kind, ValueKind::State(_));
    let len = |v: ValueId| numel(g, v).ok();
    let node_of = |v: ValueId| match g.value(v).kind {
        ValueKind::Node(n) => Some(&g.nodes[n.0 as usize]),
        _ => None,
    };
    let only_reader = |v: ValueId| -> Option<&Node> {
        match readers.get(&v).map(Vec::as_slice) {
            Some(&[r]) if !g.outputs.contains(&v) => Some(&g.nodes[r]),
            _ => None,
        }
    };
    let single = |v: ValueId| only_reader(v).is_some();
    // The step lowering exactly node `n`: same output, same name (not a producer that absorbed it).
    let own = |n: &Node| {
        step_of
            .get(n.outputs.first()?)
            .copied()
            .filter(|&i| steps[i].name == n.name)
    };
    // `f(input)` as its own step with a single reader: `Some((input, step))`.
    let absorbable = |v: ValueId, f: fn(&Op) -> bool| -> Option<(ValueId, usize)> {
        let n = node_of(v).filter(|n| f(&n.op) && n.inputs.len() == 1 && single(v))?;
        Some((n.inputs[0], own(n)?))
    };

    if !is_state(state) || kh == 0 || vh % kh != 0 {
        return None;
    }
    layout(kd, vd)?;
    let kk = kh * kd;
    let cd = 2 * kk + vh * vd;
    let mut rm = vec![di];

    // q | k | v: last-axis slices of one SiLU(conv) value, views or single-reader copies.
    let slice_of = |x: ValueId, lo: usize, hi: usize| -> Option<ValueId> {
        let n = node_of(x)?;
        let Op::Slice { axis, start, end } = n.op else {
            return None;
        };
        let src = *n.inputs.first()?;
        let rank = g.value(src).shape.len() as isize;
        let last = axis == -1 || axis == rank - 1;
        (single(x) && last && start == lo as i64 && end == hi as i64 && len(src)? == cd)
            .then_some(src)
    };
    let m = slice_of(q, 0, kk)?;
    if slice_of(k, kk, 2 * kk)? != m || slice_of(v, 2 * kk, cd)? != m {
        return None;
    }
    if readers.get(&m).map_or(0, Vec::len) != 3 || g.outputs.contains(&m) {
        return None;
    }
    for x in [q, k, v] {
        rm.extend(node_of(x).and_then(own));
    }
    let silu = node_of(m).filter(|n| is_silu(&n.op) && n.inputs.len() == 1)?;
    rm.push(own(silu)?);
    let c = silu.inputs[0];
    let conv = node_of(c).filter(|_| single(c))?;
    let Op::CausalConv1dCached { kernel, dilation } = conv.op else {
        return None;
    };
    let &[x, conv_w, conv_state] = conv.inputs.as_slice() else {
        return None;
    };
    if kernel == 0
        || dilation == 0
        || !is_state(conv_state)
        || len(x)? != cd
        || len(conv_w)? != cd * kernel
        || len(conv_state)? != (kernel - 1) * dilation * cd
    {
        return None;
    }
    rm.push(own(conv)?);

    // beta = sigmoid(b), or any [val_heads] value.
    let (b, beta_sigmoid) = match absorbable(beta, is_sigmoid) {
        Some((b, i)) => {
            rm.push(i);
            (b, true)
        }
        None => (beta, false),
    };
    // log_decay = w * softplus(u), u = a + bias or any [val_heads] value.
    let ldn =
        node_of(log_decay).filter(|n| is_mul(&n.op) && n.inputs.len() == 2 && single(log_decay))?;
    rm.push(own(ldn)?);
    let is_softplus = |op: &Op| {
        matches!(
            op,
            Op::Unary {
                kind: UnaryKind::Softplus
            }
        )
    };
    let (neg_a, (u, sp_i)) = [
        (ldn.inputs[0], ldn.inputs[1]),
        (ldn.inputs[1], ldn.inputs[0]),
    ]
    .into_iter()
    .find_map(|(w, s)| Some((w, absorbable(s, is_softplus)?)))?;
    rm.push(sp_i);
    let (a, dt_bias) = match node_of(u)
        .filter(|n| is_add(&n.op) && n.inputs.len() == 2 && single(u))
        .and_then(|n| Some((n, own(n)?)))
    {
        Some((n, i)) if len(n.inputs[0])? == vh && len(n.inputs[1])? == vh => {
            rm.push(i);
            (n.inputs[0], Some(n.inputs[1]))
        }
        _ => (u, None),
    };
    if [b, neg_a, a].into_iter().any(|v| len(v) != Some(vh)) {
        return None;
    }

    // out = RmsNorm(core, g[val_dim]) * gate, gate = act(z) or any [val_heads * val_dim] value.
    let core = steps[di].out;
    let rn = only_reader(core)?;
    let Op::RmsNorm { eps } = rn.op else {
        return None;
    };
    let &[rx, norm_w] = rn.inputs.as_slice() else {
        return None;
    };
    let rn_i = own(rn)?;
    if rx != core || len(core)? != vh * vd || len(norm_w)? != vd {
        return None;
    }
    if !matches!(
        steps[rn_i].op,
        super::super::Step::RmsNorm {
            res: None,
            scale: None,
            chain: None,
            rope: None,
            ..
        }
    ) {
        return None;
    }
    rm.push(rn_i);
    let normed = rn.outputs[0];
    let gm = only_reader(normed).filter(|n| is_mul(&n.op) && n.inputs.len() == 2)?;
    let za = if gm.inputs[0] == normed {
        gm.inputs[1]
    } else {
        gm.inputs[0]
    };
    if za == normed || len(za)? != vh * vd {
        return None;
    }
    let at = own(gm)?;
    let (z, z_act) = if let Some((z, i)) = absorbable(za, is_silu) {
        rm.push(i);
        (z, 1)
    } else if let Some((z, i)) = absorbable(za, is_sigmoid) {
        rm.push(i);
        (z, 2)
    } else {
        (za, 0)
    };
    if len(z)? != vh * vd || rm.iter().any(|&i| i >= at) {
        return None;
    }
    let st = DeltaNetStep {
        x,
        conv_w,
        conv_state,
        z,
        b,
        a,
        dt_bias,
        neg_a,
        state,
        norm_w,
        kernel,
        dilation,
        key_heads: kh,
        val_heads: vh,
        key_dim: kd,
        val_dim: vd,
        eps,
        z_act,
        beta_sigmoid,
        state_dtype,
    };
    Some((at, rm, st))
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &DeltaNetStep) -> Vec<ValueId> {
    [
        Some(st.x),
        Some(st.conv_w),
        Some(st.conv_state),
        Some(st.z),
        Some(st.b),
        Some(st.a),
        st.dt_bias,
        Some(st.neg_a),
        Some(st.state),
        Some(st.norm_w),
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub(super) fn allows_non_f32(_st: &DeltaNetStep, _v: ValueId) -> bool {
    false
}

pub(super) fn encode(
    sess: &Session,
    e: &Enc,
    s: &Planned,
    st: &DeltaNetStep,
) -> Result<(), String> {
    let (cw, rgroups, rpt) = layout(st.key_dim, st.val_dim).ok_or_else(|| {
        format!(
            "node '{}': delta-rule head dims exceed the fused kernel",
            s.name
        )
    })?;
    let pso = sess.rt.pso(st.state_dtype.kernel())?;
    if (pso.max_total_threads_per_threadgroup() as usize) < THREADS {
        return Err(format!(
            "node '{}': {} allows {} threads per threadgroup, needs {THREADS}",
            s.name,
            st.state_dtype.kernel(),
            pso.max_total_threads_per_threadgroup()
        ));
    }
    e.enc.set_compute_pipeline_state(pso);
    for (slot, v) in [
        st.x,
        st.conv_w,
        st.conv_state,
        st.z,
        st.b,
        st.a,
        st.dt_bias.unwrap_or(st.a),
        st.neg_a,
        st.state,
        st.norm_w,
        s.out,
    ]
    .into_iter()
    .enumerate()
    {
        e.bind(slot as u64, v);
    }
    #[repr(C)]
    struct DeltaNetParams {
        channels: u32,
        kernel: u32,
        dilation: u32,
        key_heads: u32,
        key_dim: u32,
        val_dim: u32,
        rep: u32,
        cw: u32,
        rgroups: u32,
        rpt: u32,
        z_act: u32,
        beta_sigmoid: u32,
        has_dt_bias: u32,
        eps: f32,
    }
    let u = |v: usize| v as u32;
    let p = DeltaNetParams {
        channels: u(2 * st.key_heads * st.key_dim + st.val_heads * st.val_dim),
        kernel: u(st.kernel),
        dilation: u(st.dilation),
        key_heads: u(st.key_heads),
        key_dim: u(st.key_dim),
        val_dim: u(st.val_dim),
        rep: u(st.val_heads / st.key_heads),
        cw: u(cw),
        rgroups: u(rgroups),
        rpt: u(rpt),
        z_act: st.z_act,
        beta_sigmoid: st.beta_sigmoid as u32,
        has_dt_bias: st.dt_bias.is_some() as u32,
        eps: st.eps,
    };
    set_bytes(e.enc, 11, &p);
    e.enc.dispatch_thread_groups(
        MTLSize::new(st.key_heads as u64, 1, 1),
        MTLSize::new(THREADS as u64, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{rand, rel_err, rt};
    use super::super::super::{kernel_kind, plan, Session, Step as Core};
    use super::super::ExtStep;
    use super::StateDtype;
    use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, UnaryKind, ValueId, Weights};

    /// `(key_heads, val_heads, key_dim, val_dim, conv taps)`.
    type Dims = (usize, usize, usize, usize, usize);

    /// One decode-token recurrent layer between its projections, laid out as the hybrid-LM graph
    /// builders emit it. Inputs `x`, `z`, `b`, `a`; with `projected`, `z` (bf16 GEMV, so its SiLU
    /// becomes the GEMV epilogue) and `a` (f32 Linear, so `+ dt_bias` becomes its residual) are
    /// projections of an input `h` instead. Returns the graph, weights and the `log_decay` value.
    fn layer(
        (kh, vh, kd, vd, kw): Dims,
        sigmoid_gate: bool,
        projected: bool,
    ) -> (Graph, Weights, ValueId) {
        let (kk, cd, f) = (kh * kd, 2 * kh * kd + vh * vd, DType::F32);
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![1, cd], f);
        let (z, a) = if projected {
            let hd = 64;
            let h = g.input("h", vec![1, hd], f);
            let bf16 = rand(vh * vd * hd, 9)
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect();
            w.insert("w_z", Tensor::BF16(bf16));
            w.insert("w_a", Tensor::F32(rand(vh * hd, 10)));
            let (wz, wa) = (
                g.weight("w_z", vec![vh * vd, hd], DType::BF16),
                g.weight("w_a", vec![vh, hd], f),
            );
            (
                g.add("proj_z", Op::Linear, &[h, wz], vec![1, vh * vd], f),
                g.add("proj_a", Op::Linear, &[h, wa], vec![1, vh], f),
            )
        } else {
            (
                g.input("z", vec![1, vh * vd], f),
                g.input("a", vec![1, vh], f),
            )
        };
        let b = g.input("b", vec![1, vh], f);
        let mut weight = |g: &mut Graph, name: &str, shape: Vec<usize>, vals: Vec<f32>| {
            w.insert(name, Tensor::F32(vals));
            g.weight(name, shape, f)
        };
        let conv_w = weight(&mut g, "conv_w", vec![cd, kw], rand(cd * kw, 5));
        let dt = weight(&mut g, "dt_bias", vec![vh], rand(vh, 6));
        let neg_a = weight(
            &mut g,
            "neg_a",
            vec![vh],
            rand(vh, 7).iter().map(|v| -0.5 - v.abs()).collect(),
        );
        let norm_w = weight(
            &mut g,
            "norm_w",
            vec![vd],
            rand(vd, 8).iter().map(|v| 1.0 + 0.2 * v).collect(),
        );
        let conv_state = g.state("conv_state", vec![kw - 1, cd], f);
        let ssm_state = g.state("ssm_state", vec![vh * kd * vd], f);
        let conv = g.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: kw,
                dilation: 1,
            },
            &[x, conv_w, conv_state],
            vec![1, cd],
            f,
        );
        let mixed = g.add("silu", Op::Silu, &[conv], vec![1, cd], f);
        let slice = |g: &mut Graph, name: &str, s: usize, e: usize| {
            g.add(
                name,
                Op::Slice {
                    axis: -1,
                    start: s as i64,
                    end: e as i64,
                },
                &[mixed],
                vec![1, e - s],
                f,
            )
        };
        let q = slice(&mut g, "q", 0, kk);
        let k = slice(&mut g, "k", kk, 2 * kk);
        let v = slice(&mut g, "v", 2 * kk, cd);
        let un = |kind| Op::Unary { kind };
        let beta = g.add("beta", un(UnaryKind::Sigmoid), &[b], vec![1, vh], f);
        let ab = g.add("a_biased", Op::Add, &[a, dt], vec![1, vh], f);
        let sp = g.add("a_softplus", un(UnaryKind::Softplus), &[ab], vec![1, vh], f);
        let ld = g.add("log_decay", Op::Mul, &[neg_a, sp], vec![1, vh], f);
        let delta = Op::DeltaRule {
            key_heads: kh,
            val_heads: vh,
            key_dim: kd,
            val_dim: vd,
        };
        let core = g.add(
            "delta",
            delta,
            &[q, k, v, beta, ld, ssm_state],
            vec![1, vh * vd],
            f,
        );
        let normed = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-6 },
            &[core, norm_w],
            vec![1, vh * vd],
            f,
        );
        let gate = if sigmoid_gate {
            un(UnaryKind::Sigmoid)
        } else {
            Op::Silu
        };
        let za = g.add("z_act", gate, &[z], vec![1, vh * vd], f);
        let out = g.add("gated", Op::Mul, &[normed, za], vec![1, vh * vd], f);
        g.output(out);
        (g, w, ld)
    }

    fn listing(g: &Graph) -> Vec<String> {
        let (steps, _) = plan(g).unwrap();
        steps
            .iter()
            .map(|s| format!("{} {}", s.name, kernel_kind(&format!("{:?}", s.op))))
            .collect()
    }

    /// `(dims, sigmoid gate, projected)` cases: padded columns with rep 2, the 128x128 head with 16 rows
    /// per thread, a 2-tap conv, and gate pieces absorbed into the projections.
    const CASES: [(Dims, bool, bool); 4] = [
        ((2, 4, 16, 48, 4), false, false),
        ((2, 4, 128, 128, 4), true, false),
        ((1, 1, 32, 32, 2), false, false),
        ((2, 4, 16, 48, 4), false, true),
    ];

    #[test]
    fn decode_chain_plans_one_step() {
        for (dims, sigmoid_gate, projected) in CASES {
            let (g, _, _) = layer(dims, sigmoid_gate, projected);
            let (steps, _) = plan(&g).unwrap();
            println!("{dims:?} projected={projected}: {:?}", listing(&g));
            let last = steps.last().expect("steps");
            let Core::Ext(ExtStep::DeltaNet(st)) = &last.op else {
                panic!("not fused: {:?}", listing(&g))
            };
            assert_eq!(steps.len(), if projected { 3 } else { 1 });
            assert_eq!(last.out, g.outputs[0]);
            assert!(st.beta_sigmoid);
            assert_eq!(st.state_dtype, StateDtype::F32);
            // Projected: SiLU(z) is the z GEMV's epilogue and `+ dt_bias` the `a` Linear's residual.
            let z_act = match (projected, sigmoid_gate) {
                (true, _) => 0,
                (false, true) => 2,
                (false, false) => 1,
            };
            assert_eq!(st.z_act, z_act);
            assert_eq!(st.dt_bias.is_some(), !projected);
        }
    }

    #[test]
    fn shared_intermediate_keeps_the_chain_unfused() {
        let (mut g, _, ld) = layer((2, 4, 16, 48, 4), false, false);
        g.output(ld);
        let names = listing(&g);
        assert!(names.iter().any(|n| n.ends_with(" DeltaRule")), "{names:?}");
        assert!(
            !names.iter().any(|n| n.ends_with(" DeltaNetStep")),
            "{names:?}"
        );
    }

    #[test]
    fn oversized_heads_stay_unfused_and_refuse_reduced_state() {
        // key_dim 128 over val_dim 256 needs 32 state rows per thread (> MAX_RPT).
        let (g, _, _) = layer((1, 1, 128, 256, 4), false, false);
        let names = listing(&g);
        assert!(names.iter().any(|n| n.ends_with(" DeltaRule")), "{names:?}");
        let (steps, _) = plan(&g).unwrap();
        assert!(super::check_state_dtype(&steps, StateDtype::F32).is_ok());
        assert!(super::check_state_dtype(&steps, StateDtype::F16).is_err());
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let n = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (n(a) * n(b)).max(1e-30)
    }

    /// Runs `tokens` decode steps on the CPU executor (f32 reference) and on Metal with the fused step
    /// storing its state as `dtype`; returns per-token (rel err, cosine) and the final state rel err.
    fn run(
        (dims, sigmoid_gate, projected): (Dims, bool, bool),
        dtype: StateDtype,
        tokens: usize,
    ) -> (Vec<(f32, f64)>, f32) {
        let (g, w, _) = layer(dims, sigmoid_gate, projected);
        let s = Session::new(rt(), &w).unwrap();
        let mut prep = s.prepare_plan(&g).unwrap();
        let fused: Vec<_> = prep
            .steps
            .iter_mut()
            .filter_map(|st| match &mut st.op {
                Core::Ext(ExtStep::DeltaNet(f)) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(fused.len(), 1, "chain not fused");
        fused.into_iter().for_each(|f| f.state_dtype = dtype);
        let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
        let mut errs = Vec::new();
        for t in 0..tokens as u64 {
            let inputs: Vec<Tensor> = g
                .inputs
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let n = g.value(v).shape.iter().product();
                    Tensor::F32(rand(n, 100 * (i as u64 + 1) + t))
                })
                .collect();
            let want = cpu::run(&g, &w, &mut cs, inputs.clone())
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            let got = s
                .run_prepared(&g, &prep, Some(&mut ms), inputs)
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            assert_eq!(want.len(), got.len());
            errs.push((rel_err(&want, &got), cosine(&want, &got)));
        }
        let state_err = ["conv_state", "ssm_state"]
            .iter()
            .map(|&n| {
                let want = cs.get(n).unwrap().as_f32().unwrap().to_vec();
                let got = ms.snapshot(&[n]).remove(n).unwrap().into_f32().unwrap();
                rel_err(&want, &got[..want.len()])
            })
            .fold(0.0f32, f32::max);
        (errs, state_err)
    }

    #[test]
    fn fused_chain_matches_cpu_over_decode_steps() {
        for case in CASES {
            let (errs, state_err) = run(case, StateDtype::F32, 20);
            for (t, &(e, _)) in errs.iter().enumerate() {
                assert!(e < 1e-4, "{case:?} token {t}: rel err {e}");
            }
            assert!(state_err < 1e-4, "{case:?}: state rel err {state_err}");
        }
    }

    #[test]
    fn reduced_precision_state_tracks_f32_reference() {
        // ponytail: bf16 bound is a margin guess (8-bit mantissa, round-to-nearest-even), not measured.
        for (dtype, min_cos) in [(StateDtype::F16, 0.9999), (StateDtype::BF16, 0.999)] {
            for case in [CASES[0], CASES[1]] {
                let (errs, _) = run(case, dtype, 32);
                let worst = errs.iter().map(|e| e.1).fold(1.0f64, f64::min);
                assert!(worst >= min_cos, "{dtype:?} {case:?}: min cosine {worst}");
            }
        }
    }
}
