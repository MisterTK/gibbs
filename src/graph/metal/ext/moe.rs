//! Metal lowering group `moe`: planning, encoding and kernels for its ops.
//! Kernels live in `moe.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Owns: `Moe` (router → top-k softmax → expert GEMVs → weighted sum, experts any Linear weight
//! format, unselected experts skipped by indirect dispatch), `TopKSoftmax` for any `k`, `Embedding`
//! over F16/BF16/codec-quantized tables, and shape guards for `CausalConv1dCached` / `DeltaRule`
//! forms the core kernels would read out of bounds.

use super::super::Counted;
use super::super::{numel, set_bytes, u32_of, GemmKernel, Planned, Session, WeightFormat, TG};
use super::{Enc, PlanOut};
use crate::graph::codec::{codec, Codec};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use crate::hardware::QuantFormat;
use metal::{Buffer, CompileOptions, ComputePipelineState, MTLSize};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;

pub(super) const MSL: &str = include_str!("moe.metal");
/// Compute functions in `moe.metal`.
pub(super) const KERNELS: &[&str] = &[
    "moe_topk_softmax_f32",
    "moe_zero_f32",
    "moe_args",
    "moe_accum_f32",
    "moe_embed_f16",
    "moe_embed_bf16",
];

/// Source encoding of an embedding table.
#[derive(Debug, Clone, Copy)]
pub enum Table {
    F16,
    BF16,
    Quant(QuantFormat),
}

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    Moe {
        x: ValueId,
        router: ValueId,
        experts: Vec<ValueId>,
        router_fmt: WeightFormat,
        expert_fmts: Vec<WeightFormat>,
        m: usize,
        d_in: usize,
        d_out: usize,
        n_experts: usize,
        top_k: usize,
    },
    TopK {
        x: ValueId,
        rows: usize,
        cols: usize,
        k: usize,
    },
    Embed {
        table: ValueId,
        ids: ValueId,
        n: usize,
        d: usize,
        rows: usize,
        src: Table,
    },
}

/// Whether the Metal executor lowers `op` for `dtype`: the output dtype for `Moe`/`TopKSoftmax`
/// (activations are f32), the table dtype for `Embedding`.
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    match op {
        Op::Moe { .. } | Op::TopKSoftmax { .. } => dtype == DType::F32,
        Op::Embedding => match dtype {
            DType::F16 | DType::BF16 => true,
            DType::Quant(f) => codec(f).is_some(),
            _ => false,
        },
        _ => false,
    }
}

/// How the executor reads a Linear-style weight of `dtype` with `k` columns; mirrors `Op::Linear`'s
/// planner (NVFP4/Jesse need per-weight aux buffers and are not claimed).
fn weight_format(dtype: DType, k: usize) -> Option<WeightFormat> {
    use QuantFormat as Q;
    let (fmt, align) = match dtype {
        DType::F32 => return Some(WeightFormat::F32),
        DType::BF16 => return Some(WeightFormat::BF16),
        DType::F16 => return Some(WeightFormat::F16),
        DType::Quant(Q::GGUF_Q8_0) => (WeightFormat::Q8_0, 32),
        DType::Quant(Q::GGUF_Q4_0) => (WeightFormat::Q4_0, 32),
        DType::Quant(Q::GGUF_IQ4_XS) => (WeightFormat::IQ4_XS, 256),
        DType::Quant(Q::GGUF_IQ2_M | Q::GGUF_IQ2_S) => (WeightFormat::IQ2_M, 256),
        DType::Quant(Q::GGUF_Q4_K_M) => (WeightFormat::Q4_K, 256),
        DType::Quant(Q::GGUF_Q2_K) => (WeightFormat::Q2_K, 256),
        // `gemv_iq3_s` disagrees with the CPU dequantizer (rel err ~0.5 on random blocks, plain Linear too): not claimed.
        DType::Quant(Q::GGUF_IQ3_S) => return None,
        DType::Quant(f) => {
            let c = codec(f)?;
            (WeightFormat::Gen(f), c.block_elems as usize)
        }
        DType::U32 => return None,
    };
    (k % align == 0).then_some(fmt)
}

fn shape2(g: &Graph, v: ValueId) -> Option<[usize; 2]> {
    match g.value(v).shape[..] {
        [a, b] => Some([a, b]),
        _ => None,
    }
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
    match &node.op {
        Op::Moe { experts, top_k } => Some(plan_moe(g, node, *experts, *top_k)),
        Op::TopKSoftmax { k, .. } => Some((|| {
            let x = arg(0)?;
            let cols = g.value(x).shape.last().copied().unwrap_or(0);
            if *k == 0 || cols == 0 {
                return Err(format!("node '{name}': TopKSoftmax needs k >= 1 and a static row width"));
            }
            let n = numel(g, x)?;
            Ok(PlanOut::Step(Step::TopK { x, rows: n / cols, cols, k: *k }, n))
        })()),
        Op::Embedding => {
            let table = node.inputs.first().copied()?;
            let src = match g.value(table).dtype {
                DType::F16 => Table::F16,
                DType::BF16 => Table::BF16,
                DType::Quant(f) if codec(f).is_some() => Table::Quant(f),
                _ => return None,
            };
            Some((|| {
                let ids = arg(1)?;
                let Some([rows, d]) = shape2(g, table) else {
                    return Err(format!("node '{name}': embedding table is not [rows, d]"));
                };
                if g.value(ids).dtype != DType::U32 || !g.inputs.contains(&ids) {
                    return Err(format!(
                        "node '{name}': Metal embedding ids must be a u32 graph input (range-checked on the host)"
                    ));
                }
                if let Table::Quant(f) = src {
                    let be = codec(f).map_or(1, |c| c.block_elems as usize);
                    if d % be != 0 {
                        return Err(format!("node '{name}': width {d} is not a multiple of the {f:?} block ({be})"));
                    }
                }
                let n = numel(g, ids)?;
                Ok(PlanOut::Step(Step::Embed { table, ids, n, d, rows, src }, n * d))
            })())
        }
        // The core kernels index state and per-head inputs unchecked; the CPU executor resets a
        // mis-sized state to zeros and defaults short gates. Reject those forms instead of reading OOB.
        Op::CausalConv1dCached { kernel, dilation } => (|| {
            let (k, d) = (*kernel, *dilation);
            let w = arg(1)?;
            if k == 0 {
                return Err(format!("node '{name}': kernel must be >= 1"));
            }
            let channels = numel(g, w)? / k;
            match node.inputs.get(2) {
                Some(&s) if numel(g, s)? != (k - 1) * d * channels => Err(format!(
                    "node '{name}': conv state must hold (kernel-1)*dilation = {} rows of {channels} channels",
                    (k - 1) * d
                )),
                _ => Ok(()),
            }
        })()
        .err()
        .map(Err),
        Op::DeltaRule { key_heads: kh, val_heads: vh, key_dim: kd, val_dim: vd } => (|| {
            let (q, beta, ld) = (arg(0)?, arg(3)?, arg(4)?);
            if *kh == 0 || *vh % *kh != 0 || *kd == 0 || *vd == 0 {
                return Err(format!("node '{name}': val_heads must be a positive multiple of key_heads"));
            }
            let n = numel(g, q)? / (kh * kd);
            if numel(g, arg(1)?)? != n * kh * kd
                || numel(g, arg(2)?)? != n * vh * vd
                || numel(g, beta)? < n * vh
                || numel(g, ld)? < n * vh
            {
                return Err(format!("node '{name}': q/k/v/beta/log_decay sizes disagree for {n} tokens"));
            }
            match node.inputs.get(5) {
                Some(&s) if numel(g, s)? != vh * kd * vd => {
                    Err(format!("node '{name}': ssm state must hold {} values", vh * kd * vd))
                }
                _ => Ok(()),
            }
        })()
        .err()
        .map(Err),
        _ => None,
    }
}

fn plan_moe(g: &Graph, node: &Node, experts: usize, top_k: usize) -> Result<PlanOut<Step>, String> {
    let name = &node.name;
    if node.inputs.len() < 3 {
        return Err(format!(
            "node '{name}': Moe needs [x, router, expert...], got {} operands",
            node.inputs.len()
        ));
    }
    if experts == 0 || top_k == 0 {
        return Err(format!(
            "node '{name}': Moe needs experts >= 1 and top_k >= 1"
        ));
    }
    let out = node.outputs[0];
    let Some([m, d_out]) = shape2(g, out) else {
        return Err(format!("node '{name}': Moe output must be [m, d_out]"));
    };
    let (x, router) = (node.inputs[0], node.inputs[1]);
    let expert_ids: Vec<ValueId> = node.inputs[2..].iter().copied().take(experts).collect();
    let Some([r_rows, d_in]) = shape2(g, router) else {
        return Err(format!(
            "node '{name}': router weight is not [experts, d_in]"
        ));
    };
    if r_rows != experts || numel(g, x)? != m * d_in {
        return Err(format!(
            "node '{name}': x of {} elements and router [{r_rows}, {d_in}] do not match {m} rows x {experts} experts",
            numel(g, x)?
        ));
    }
    let fmt = |w: ValueId, rows: usize| -> Result<WeightFormat, String> {
        let val = g.value(w);
        if val.shape[..] != [rows, d_in] {
            return Err(format!(
                "node '{name}': weight '{}' is {:?}, expected [{rows}, {d_in}]",
                val.name, val.shape
            ));
        }
        weight_format(val.dtype, d_in).ok_or_else(|| {
            format!(
                "node '{name}': weight '{}' ({:?}) has no packed GEMV for {d_in} columns",
                val.name, val.dtype
            )
        })
    };
    let router_fmt = fmt(router, experts)?;
    let expert_fmts = expert_ids
        .iter()
        .map(|&w| fmt(w, d_out))
        .collect::<Result<Vec<_>, _>>()?;
    // One indirect-argument geometry serves every expert: F32 uses the GEMM tiles, the rest the packed GEMV grid.
    if expert_fmts
        .iter()
        .any(|f| (*f == WeightFormat::F32) != (expert_fmts[0] == WeightFormat::F32))
    {
        return Err(format!(
            "node '{name}': experts mix f32 and packed weight formats"
        ));
    }
    Ok(PlanOut::Step(
        Step::Moe {
            x,
            router,
            experts: expert_ids,
            router_fmt,
            expert_fmts,
            m,
            d_in,
            d_out,
            n_experts: experts,
            top_k,
        },
        m * d_out,
    ))
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match st {
        Step::Moe {
            x, router, experts, ..
        } => [*x, *router]
            .into_iter()
            .chain(experts.iter().copied())
            .collect(),
        Step::TopK { x, .. } => vec![*x],
        Step::Embed { table, ids, .. } => vec![*table, *ids],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, v: ValueId) -> bool {
    match st {
        Step::Moe { x, .. } => v != *x,
        Step::TopK { .. } => false,
        Step::Embed { .. } => true,
    }
}

/// Threadgroups and size for one `[m, rows] = x[m, cols] · wᵀ` dispatch plus the F32 GEMM kernel it uses.
struct Geometry {
    groups: MTLSize,
    threads: u64,
    kind: Option<GemmKernel>,
}

fn geometry(sess: &Session, fmt: WeightFormat, m: usize, rows: usize, cols: usize) -> Geometry {
    if fmt == WeightFormat::F32 {
        let kind = sess
            .gemm
            .unwrap_or_else(|| sess.rt.profile.gemm_for([m, rows, cols]));
        let (_, groups, threads) = kind.spec(m, rows);
        Geometry {
            groups,
            threads,
            kind: Some(kind),
        }
    } else {
        Geometry {
            groups: MTLSize::new((rows as u64).div_ceil(16), m as u64, 1),
            threads: 256,
            kind: None,
        }
    }
}

/// `out[m, rows] = x[m, cols] · w[rows, cols]ᵀ` through the existing GEMM / packed-GEMV kernels;
/// `indirect` takes the threadgroup counts from a GPU-written argument triple instead.
#[allow(clippy::too_many_arguments)]
fn matvec(
    sess: &Session,
    enc: &Counted,
    geo: &Geometry,
    fmt: WeightFormat,
    x: (&Buffer, usize),
    w: (&Buffer, usize),
    out: (&Buffer, usize),
    [m, rows, cols]: [usize; 3],
    indirect: Option<(&Buffer, usize)>,
) -> Result<(), String> {
    if let Some(kind) = geo.kind {
        let (prefix, ..) = kind.spec(m, rows);
        enc.set_compute_pipeline_state(sess.rt.pso(&format!("{prefix}_none"))?);
        for (i, (b, off)) in [x, w, out, out, out].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), off as u64);
        }
        set_bytes(enc, 5, &[u32_of(m)?, u32_of(rows)?, u32_of(cols)?, 0u32]);
    } else {
        enc.set_compute_pipeline_state(sess.gemv_pso(fmt)?);
        for (i, (b, off)) in [w, x, out].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), off as u64);
        }
        set_bytes(
            enc,
            3,
            &[u32_of(rows)?, u32_of(cols)?, u32_of(m)?, 0u32, 0u32],
        );
        enc.set_buffer(4, Some(out.0), out.1 as u64);
        enc.set_buffer(5, Some(out.0), out.1 as u64);
    }
    let tg = MTLSize::new(geo.threads, 1, 1);
    match indirect {
        None => enc.dispatch_thread_groups(geo.groups, tg),
        Some((b, off)) => enc.dispatch_thread_groups_indirect(b, off as u64, tg),
    }
    Ok(())
}

fn align(n: usize) -> usize {
    n.div_ceil(256) * 256
}

/// Generated gather for one codec: one thread per 8-element sub-block of one token's row.
fn quant_embed_src(c: &Codec) -> String {
    let (be, bb) = (c.block_elems, c.block_bytes);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{msl}
kernel void embed_quant(device const uchar* table [[buffer(0)]],
                        device const uint* ids [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint2& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {{
    uint n = p.x, d = p.y;
    if (gid.x * 8 >= d || gid.y >= n) return;
    uint blk = (gid.x * 8) / {be}u;
    uint sub = ((gid.x * 8) % {be}u) / 8;
    device const uchar* b = table + ((ulong)ids[gid.y] * (d / {be}u) + blk) * {bb}ul;
    float v[8];
    decode8(b, sub, v);
    for (uint i = 0; i < 8; ++i) out[(ulong)gid.y * d + gid.x * 8 + i] = v[i];
}}
"#,
        msl = c.msl
    )
}

/// Per (device, codec) pipelines of [`quant_embed_src`], compiled on first use.
static QUANT_EMBED: LazyLock<Mutex<HashMap<(u64, QuantFormat), ComputePipelineState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn quant_embed_pso(sess: &Session, f: QuantFormat) -> Result<ComputePipelineState, String> {
    let device = &sess.rt.device;
    let key = (device.registry_id(), f);
    if let Some(p) = QUANT_EMBED.lock().get(&key) {
        return Ok(p.clone());
    }
    let c = codec(f).ok_or_else(|| format!("no codec for {f:?}"))?;
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let lib = device
        .new_library_with_source(&quant_embed_src(c), &opts)
        .map_err(|e| format!("generated {f:?} embedding gather failed to compile: {e}"))?;
    let func = lib.get_function("embed_quant", None)?;
    let pso = device.new_compute_pipeline_state_with_function(&func)?;
    QUANT_EMBED.lock().insert(key, pso.clone());
    Ok(pso)
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    let (enc, rt) = (e.enc, &sess.rt);
    match st {
        Step::TopK { x, rows, cols, k } => {
            enc.set_compute_pipeline_state(rt.pso("moe_topk_softmax_f32")?);
            e.bind(0, *x);
            e.bind(1, s.out);
            set_bytes(enc, 2, &[u32_of(*rows)?, u32_of(*cols)?, u32_of(*k)?]);
            e.grid1(*rows);
        }
        Step::Embed {
            table,
            ids,
            n,
            d,
            rows,
            src,
        } => {
            let (ib, io) = (e.loc)(*ids);
            // SAFETY: `plan` requires `ids` to be a graph input, which the host copies into this shared
            // buffer (n u32) before any step is encoded.
            let idv = unsafe {
                std::slice::from_raw_parts((ib.contents() as *const u8).add(io) as *const u32, *n)
            };
            if let Some((r, &id)) = idv
                .iter()
                .enumerate()
                .find(|&(_, &id)| id as usize >= *rows)
            {
                return Err(format!(
                    "node '{}': row {r} id {id} out of range for table of {rows} rows",
                    s.name
                ));
            }
            let (pso, cols_per_thread) = match src {
                Table::F16 => (rt.pso("moe_embed_f16")?.clone(), 1),
                Table::BF16 => (rt.pso("moe_embed_bf16")?.clone(), 1),
                Table::Quant(f) => (quant_embed_pso(sess, *f)?, 8),
            };
            enc.set_compute_pipeline_state(&pso);
            e.bind(0, *table);
            e.bind(1, *ids);
            e.bind(2, s.out);
            set_bytes(enc, 3, &[u32_of(*n)?, u32_of(*d)?]);
            enc.dispatch_thread_groups(
                MTLSize::new(((*d / cols_per_thread) as u64).div_ceil(TG), *n as u64, 1),
                MTLSize::new(TG, 1, 1),
            );
        }
        Step::Moe {
            x,
            router,
            experts,
            router_fmt,
            expert_fmts,
            m,
            d_in,
            d_out,
            n_experts,
            top_k,
        } => {
            let (m, d_in, d_out, ne) = (*m, *d_in, *d_out, *n_experts);
            let actual = experts.len();
            let (off_logits, off_probs) = (0, align(m * ne * 4));
            let off_tmp = off_probs * 2;
            let off_args = off_tmp + align(m * d_out * 4);
            let total = off_args + align(actual * 24);
            // ponytail: shares the megakernel's scratch buffer; steps run in order on one serial encoder.
            let mut guard = sess.mlp_scratch.lock();
            if guard.as_ref().is_none_or(|b| (b.length() as usize) < total) {
                *guard = Some(super::super::new_shared_buffer(&rt.device, total.max(256))?);
            }
            let scratch = guard.as_ref().ok_or("moe scratch missing")?;
            let (xl, out) = ((e.loc)(*x), (e.loc)(s.out));

            // 1. router logits, 2. top-k softmax -> probs
            let geo_r = geometry(sess, *router_fmt, m, ne, d_in);
            matvec(
                sess,
                enc,
                &geo_r,
                *router_fmt,
                xl,
                (e.loc)(*router),
                (scratch, off_logits),
                [m, ne, d_in],
                None,
            )?;
            enc.set_compute_pipeline_state(rt.pso("moe_topk_softmax_f32")?);
            enc.set_buffer(0, Some(scratch), off_logits as u64);
            enc.set_buffer(1, Some(scratch), off_probs as u64);
            set_bytes(enc, 2, &[u32_of(m)?, u32_of(ne)?, u32_of(*top_k)?]);
            e.grid1(m);
            // 3. out = 0
            enc.set_compute_pipeline_state(rt.pso("moe_zero_f32")?);
            enc.set_buffer(0, Some(out.0), out.1 as u64);
            set_bytes(enc, 1, &u32_of(m * d_out)?);
            e.grid1(m * d_out);
            // 4. per-expert indirect arguments
            let geo = geometry(sess, expert_fmts[0], m, d_out, d_in);
            let acc_x = (d_out as u64).div_ceil(TG) as u32;
            enc.set_compute_pipeline_state(rt.pso("moe_args")?);
            enc.set_buffer(0, Some(scratch), off_probs as u64);
            enc.set_buffer(1, Some(scratch), off_args as u64);
            set_bytes(
                enc,
                2,
                &[
                    u32_of(m)?,
                    u32_of(ne)?,
                    u32_of(actual)?,
                    geo.groups.width as u32,
                    geo.groups.height as u32,
                    geo.groups.depth as u32,
                    acc_x,
                    u32_of(m)?,
                    1,
                ],
            );
            e.grid1(actual);
            // 5. selected experts: tmp = x · Wᵀ, out += p · tmp
            // ponytail: prefill (m > 1) reads each selected expert once per row; a batched expert GEMM is the upgrade.
            for (ei, (&w, &fmt)) in experts.iter().zip(expert_fmts).enumerate() {
                let args = off_args + ei * 24;
                matvec(
                    sess,
                    enc,
                    &geo,
                    fmt,
                    xl,
                    (e.loc)(w),
                    (scratch, off_tmp),
                    [m, d_out, d_in],
                    Some((scratch, args)),
                )?;
                enc.set_compute_pipeline_state(rt.pso("moe_accum_f32")?);
                enc.set_buffer(0, Some(out.0), out.1 as u64);
                enc.set_buffer(1, Some(scratch), off_tmp as u64);
                enc.set_buffer(2, Some(scratch), off_probs as u64);
                set_bytes(
                    enc,
                    3,
                    &[u32_of(m)?, u32_of(d_out)?, u32_of(ne)?, u32_of(ei)?],
                );
                enc.dispatch_thread_groups_indirect(
                    scratch,
                    (args + 12) as u64,
                    MTLSize::new(TG, 1, 1),
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{both, rand, rel_err, rt};
    use super::super::super::Session;
    use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
    use crate::hardware::QuantFormat as Q;
    use crate::model::layers::QuantizedWeight;
    use crate::model::{f32_to_bf16, f32_to_f16};

    /// Storage format of a test weight.
    #[derive(Clone, Copy, Debug)]
    enum Fmt {
        F32,
        F16,
        BF16,
        Q(Q),
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Byte offsets of the f16 fields a block dequantizer reads (mirrors `tests/codec_parity.rs`).
    fn f16_offsets(f: Q) -> &'static [usize] {
        match f {
            Q::GGUF_Q4_1 | Q::GGUF_Q5_1 | Q::GGUF_Q4_K_M | Q::GGUF_Q5_K_M => &[0, 2],
            Q::GGUF_Q6_K => &[208],
            Q::GGUF_Q2_K => &[80, 82],
            Q::GGUF_Q3_K => &[108],
            Q::GGUF_TQ1_0 => &[52],
            Q::GGUF_TQ2_0 => &[64],
            Q::GGUF_IQ1_M | Q::MXFP4 => &[],
            _ => &[0],
        }
    }

    /// Random finite blocks. `small` keeps scales in 2^-5..2^1 so GEMV sums stay well conditioned.
    fn random_blocks(f: Q, n_blocks: usize, small: bool, seed: u64) -> Vec<u8> {
        let bb = crate::graph::codec::codec(f).expect("codec").block_bytes as usize;
        let mut r = Lcg(seed);
        let mut f16 = |r: &mut Lcg| -> u16 {
            let sign = (r.next() & 1) as u16;
            let exp = if small {
                10 + (r.next() % 6) as u16
            } else if r.next() % 8 == 0 {
                0
            } else {
                1 + (r.next() % 25) as u16
            };
            (sign << 15) | (exp << 10) | (r.next() & 0x3FF) as u16
        };
        let mut raw: Vec<u8> = (0..n_blocks * bb).map(|_| (r.next() >> 5) as u8).collect();
        for b in raw.chunks_exact_mut(bb) {
            for &o in f16_offsets(f) {
                b[o..o + 2].copy_from_slice(&f16(&mut r).to_le_bytes());
            }
            match f {
                Q::GGUF_IQ1_M => {
                    let h = f16(&mut r);
                    for k in 0..4 {
                        let at = 48 + 2 * k;
                        let w = u16::from_le_bytes([b[at], b[at + 1]]);
                        let w = (w & 0x0FFF) | (((h >> (4 * k)) & 0xF) << 12);
                        b[at..at + 2].copy_from_slice(&w.to_le_bytes());
                    }
                }
                Q::MXFP4 => b[0] = 100 + (r.next() % 50) as u8,
                _ => {}
            }
        }
        raw
    }

    /// A `[rows, cols]` weight in `fmt` with its graph dtype.
    fn weight(fmt: Fmt, rows: usize, cols: usize, seed: u64) -> (DType, Tensor) {
        let vals = rand(rows * cols, seed);
        match fmt {
            Fmt::F32 => (DType::F32, Tensor::F32(vals)),
            Fmt::F16 => (
                DType::F16,
                Tensor::F16(vals.iter().map(|&v| f32_to_f16(v)).collect()),
            ),
            Fmt::BF16 => (
                DType::BF16,
                Tensor::BF16(vals.iter().map(|&v| f32_to_bf16(v)).collect()),
            ),
            Fmt::Q(f) => {
                let be = crate::graph::codec::codec(f).expect("codec").block_elems as usize;
                assert_eq!(cols % be, 0);
                let raw = random_blocks(f, rows * cols / be, true, seed);
                let q = QuantizedWeight::GgufQuantized {
                    format: f,
                    raw_data: raw.into(),
                    in_features: cols,
                    out_features: rows,
                };
                (DType::Quant(f), Tensor::Quantized(std::sync::Arc::new(q)))
            }
        }
    }

    // ---- TopKSoftmax ----

    fn topk_graph(rows: usize, cols: usize, k: usize) -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", vec![rows, cols], DType::F32);
        let y = g.add(
            "tk",
            Op::TopKSoftmax { axis: -1, k },
            &[x],
            vec![rows, cols],
            DType::F32,
        );
        g.output(y);
        g
    }

    #[test]
    fn topk_softmax_matches_cpu_for_any_k() {
        // k beyond 32 is where the core kernel's fixed top-32 buffer truncates; k >= cols is a plain softmax.
        for (rows, cols, k) in [
            (1, 1, 1),
            (1, 2, 1),
            (3, 4, 2),
            (5, 37, 8),
            (4, 129, 33),
            (2, 64, 40),
            (3, 10, 10),
            (3, 10, 17),
            (7, 300, 64),
            (1, 256, 1),
        ] {
            let x: Vec<f32> = rand(rows * cols, (rows * 31 + cols) as u64)
                .iter()
                .map(|v| v * 4.0)
                .collect();
            let (want, got) = both(
                &topk_graph(rows, cols, k),
                &Weights::new(),
                vec![Tensor::F32(x)],
            );
            let nz = want
                .chunks(cols)
                .all(|r| r.iter().filter(|&&v| v != 0.0).count() == k.min(cols));
            assert!(nz, "reference keeps k entries");
            let err = rel_err(&want, &got);
            assert!(err < 1e-5, "topk {rows}x{cols} k={k}: {err}");
        }
    }

    #[test]
    fn topk_softmax_rejects_zero_k() {
        let e = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(&topk_graph(2, 4, 0), vec![Tensor::F32(vec![0.0; 8])])
            .unwrap_err();
        assert!(e.contains("tk"), "{e}");
    }

    // ---- Embedding ----

    fn embed_run(
        fmt: Fmt,
        rows: usize,
        d: usize,
        ids: Vec<u32>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (dt, t) = weight(fmt, rows, d, (rows * 7 + d) as u64);
        let mut g = Graph::new();
        let table = g.weight("t", vec![rows, d], dt);
        let idv = g.input("ids", vec![ids.len()], DType::U32);
        let y = g.add(
            "emb",
            Op::Embedding,
            &[table, idv],
            vec![ids.len(), d],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("t", t);
        let inputs = vec![Tensor::U32(ids)];
        let want = cpu::run_stateless(&g, &w, inputs.clone())?
            .remove(0)
            .into_f32()?;
        let got = Session::new(rt(), &w)?
            .run(&g, inputs)?
            .remove(0)
            .into_f32()?;
        Ok((want, got))
    }

    #[test]
    fn embedding_half_tables_match_cpu() {
        for fmt in [Fmt::F16, Fmt::BF16] {
            for (rows, d) in [(1, 1), (3, 3), (37, 37), (129, 64), (5, 129)] {
                for n in [1usize, 5, 37] {
                    let ids: Vec<u32> = (0..n)
                        .map(|i| ((i * 7 + 3) % rows) as u32)
                        .chain([0, rows as u32 - 1])
                        .collect();
                    let (want, got) = embed_run(fmt, rows, d, ids).unwrap();
                    assert!(rel_err(&want, &got) < 1e-6, "{fmt:?} {rows}x{d} n={n}");
                }
            }
        }
    }

    #[test]
    fn embedding_quantized_tables_match_cpu() {
        let formats = [
            Q::GGUF_Q4_0,
            Q::GGUF_Q4_1,
            Q::GGUF_Q5_0,
            Q::GGUF_Q5_1,
            Q::GGUF_Q8_0,
            Q::GGUF_Q2_K,
            Q::GGUF_Q3_K,
            Q::GGUF_Q4_K_M,
            Q::GGUF_Q5_K_M,
            Q::GGUF_Q6_K,
            Q::GGUF_IQ4_NL,
            Q::GGUF_IQ4_XS,
            Q::GGUF_IQ3_XXS,
            Q::GGUF_IQ3_S,
            Q::GGUF_IQ2_XXS,
            Q::GGUF_IQ2_XS,
            Q::GGUF_IQ2_S,
            Q::GGUF_IQ2_M,
            Q::GGUF_IQ1_S,
            Q::GGUF_IQ1_M,
            Q::GGUF_TQ1_0,
            Q::GGUF_TQ2_0,
            Q::MXFP4,
        ];
        let mut checked = 0;
        for f in formats {
            let Some(c) = crate::graph::codec::codec(f) else {
                continue;
            };
            let be = c.block_elems as usize;
            for (rows, blocks) in [(1usize, 1usize), (7, 3), (37, 2)] {
                let ids: Vec<u32> = vec![0, rows as u32 - 1, (rows / 2) as u32, 0];
                let (want, got) = embed_run(Fmt::Q(f), rows, be * blocks, ids).unwrap();
                let bad = want
                    .iter()
                    .zip(&got)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    bad.is_none(),
                    "{f:?} {rows} rows x {} cols: element {bad:?} differs",
                    be * blocks
                );
            }
            checked += 1;
        }
        assert!(checked >= 10, "only {checked} codecs registered");
    }

    #[test]
    fn embedding_rejects_bad_ids_and_widths() {
        let e = embed_run(Fmt::F16, 4, 8, vec![0, 4]).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let e = embed_run(Fmt::BF16, 4, 8, vec![9]).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let (dt, t) = weight(Fmt::Q(Q::GGUF_Q4_0), 4, 32, 1);
        let mut g = Graph::new();
        let table = g.weight("t", vec![4, 16], dt); // width 16 is not a Q4_0 block multiple
        let ids = g.input("ids", vec![1], DType::U32);
        let y = g.add("emb", Op::Embedding, &[table, ids], vec![1, 16], DType::F32);
        g.output(y);
        let mut w = Weights::new();
        w.insert("t", t);
        let e = Session::new(rt(), &w)
            .unwrap()
            .run(&g, vec![Tensor::U32(vec![0])])
            .unwrap_err();
        assert!(e.contains("block"), "{e}");
    }

    // ---- Moe ----

    struct MoeCase {
        m: usize,
        d_in: usize,
        d_out: usize,
        experts: usize,
        top_k: usize,
        given: usize,
        router: Fmt,
        fmts: Vec<Fmt>,
    }

    fn moe_graph(c: &MoeCase, seed: u64) -> (Graph, Weights, Vec<Tensor>) {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![c.m, c.d_in], DType::F32);
        let (dt, t) = weight(c.router, c.experts, c.d_in, seed);
        let r = g.weight("router", vec![c.experts, c.d_in], dt);
        w.insert("router", t);
        let mut ins = vec![x, r];
        for e in 0..c.given {
            let (dt, t) = weight(
                c.fmts[e % c.fmts.len()],
                c.d_out,
                c.d_in,
                seed + 100 + e as u64,
            );
            ins.push(g.weight(format!("e{e}"), vec![c.d_out, c.d_in], dt));
            w.insert(format!("e{e}"), t);
        }
        let y = g.add(
            "moe",
            Op::Moe {
                experts: c.experts,
                top_k: c.top_k,
            },
            &ins,
            vec![c.m, c.d_out],
            DType::F32,
        );
        g.output(y);
        (g, w, vec![Tensor::F32(rand(c.m * c.d_in, seed + 7))])
    }

    /// (CPU executor on the lowered graph, Metal on the `Moe` node itself).
    fn moe_both(c: &MoeCase, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let (g, w, inputs) = moe_graph(c, seed);
        let mut lowered = g.clone();
        crate::graph::lower_moe_pass(&mut lowered).unwrap();
        let want = cpu::run_stateless(&lowered, &w, inputs.clone())
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        let got = Session::new(rt(), &w)
            .unwrap()
            .run(&g, inputs)
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        assert_eq!(got.len(), want.len());
        (want, got)
    }

    fn check(c: &MoeCase, seed: u64) {
        let (want, got) = moe_both(c, seed);
        let err = rel_err(&want, &got);
        assert!(
            err < 1e-4,
            "moe m={} {}->{} E={} k={} router {:?} experts {:?}: {err}",
            c.m,
            c.d_in,
            c.d_out,
            c.experts,
            c.top_k,
            c.router,
            c.fmts
        );
    }

    #[test]
    fn moe_f32_shapes_match_cpu() {
        for (i, (m, d_in, d_out, experts, top_k, given)) in [
            (1, 16, 8, 4, 2, 4),
            (3, 37, 29, 8, 3, 8),
            (17, 64, 33, 6, 6, 6), // top_k == experts: every expert runs for every row
            (1, 129, 1, 5, 1, 5),
            (4, 32, 16, 8, 20, 8),   // top_k > experts clamps
            (2, 24, 40, 4, 2, 3),    // fewer expert weights than router columns
            (37, 8, 129, 12, 4, 12), // prefill-sized, d_out not a tile multiple
        ]
        .into_iter()
        .enumerate()
        {
            check(
                &MoeCase {
                    m,
                    d_in,
                    d_out,
                    experts,
                    top_k,
                    given,
                    router: Fmt::F32,
                    fmts: vec![Fmt::F32],
                },
                i as u64 + 1,
            );
        }
    }

    #[test]
    fn moe_half_and_mixed_packed_experts_match_cpu() {
        for (i, fmts) in [
            vec![Fmt::F16],
            vec![Fmt::BF16],
            vec![
                Fmt::F16,
                Fmt::BF16,
                Fmt::Q(Q::GGUF_Q4_0),
                Fmt::Q(Q::GGUF_Q8_0),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            for m in [1, 3, 20] {
                check(
                    &MoeCase {
                        m,
                        d_in: 64,
                        d_out: 19,
                        experts: 6,
                        top_k: 2,
                        given: 6,
                        router: Fmt::F32,
                        fmts: fmts.clone(),
                    },
                    50 + i as u64,
                );
            }
        }
        check(
            &MoeCase {
                m: 2,
                d_in: 32,
                d_out: 8,
                experts: 4,
                top_k: 2,
                given: 4,
                router: Fmt::F16,
                fmts: vec![Fmt::BF16],
            },
            60,
        );
    }

    #[test]
    fn moe_quantized_experts_match_cpu() {
        for (i, f) in [
            Q::GGUF_Q4_0,
            Q::GGUF_Q8_0,
            Q::GGUF_Q4_K_M,
            Q::GGUF_Q2_K,
            Q::GGUF_IQ4_XS,
            Q::GGUF_IQ2_M,
            Q::GGUF_TQ2_0,
            Q::GGUF_Q6_K,
            Q::GGUF_Q5_0,
            Q::GGUF_IQ4_NL,
        ]
        .into_iter()
        .enumerate()
        {
            let d_in = if matches!(
                f,
                Q::GGUF_Q4_0 | Q::GGUF_Q8_0 | Q::GGUF_Q5_0 | Q::GGUF_IQ4_NL
            ) {
                96
            } else {
                512
            };
            for m in [1, 3] {
                check(
                    &MoeCase {
                        m,
                        d_in,
                        d_out: 19,
                        experts: 5,
                        top_k: 2,
                        given: 5,
                        router: Fmt::F32,
                        fmts: vec![Fmt::Q(f)],
                    },
                    80 + i as u64,
                );
            }
        }
        // quantized router too
        check(
            &MoeCase {
                m: 2,
                d_in: 256,
                d_out: 8,
                experts: 4,
                top_k: 2,
                given: 4,
                router: Fmt::Q(Q::GGUF_Q4_K_M),
                fmts: vec![Fmt::Q(Q::GGUF_Q4_K_M)],
            },
            99,
        );
    }

    /// An expert no row routes to must not run: NaN weights in it leave the output untouched.
    #[test]
    fn moe_skips_unselected_experts() {
        let c = MoeCase {
            m: 1,
            d_in: 32,
            d_out: 16,
            experts: 4,
            top_k: 2,
            given: 4,
            router: Fmt::F32,
            fmts: vec![Fmt::F32],
        };
        let (g, mut w, _) = moe_graph(&c, 5);
        // x > 0 and an all -10 router row 3 put expert 3's logit far below the others.
        let inputs = vec![Tensor::F32(rand(32, 1).iter().map(|v| v + 1.5).collect())];
        let mut router = rand(4 * 32, 9);
        router[3 * 32..].iter_mut().for_each(|v| *v = -10.0);
        w.insert("router", Tensor::F32(router));
        let run = |w: &Weights| {
            Session::new(rt(), w)
                .unwrap()
                .run(&g, inputs.clone())
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap()
        };
        let clean = run(&w);
        w.insert("e3", Tensor::F32(vec![f32::NAN; 16 * 32]));
        let poisoned = run(&w);
        assert!(clean.iter().all(|v| v.is_finite()));
        assert_eq!(clean, poisoned);
    }

    #[test]
    fn moe_rejects_malformed_nodes() {
        let run = |c: &MoeCase| {
            let (g, w, inputs) = moe_graph(c, 3);
            Session::new(rt(), &w).unwrap().run(&g, inputs).map(|_| ())
        };
        let base = || MoeCase {
            m: 1,
            d_in: 16,
            d_out: 8,
            experts: 4,
            top_k: 2,
            given: 4,
            router: Fmt::F32,
            fmts: vec![Fmt::F32],
        };
        run(&base()).unwrap();
        let e = run(&MoeCase { top_k: 0, ..base() }).unwrap_err();
        assert!(e.contains("moe") && e.contains("top_k"), "{e}");
        let e = run(&MoeCase { given: 0, ..base() }).unwrap_err();
        assert!(e.contains("moe"), "{e}");
        let e = run(&MoeCase {
            fmts: vec![Fmt::F32, Fmt::F16],
            ..base()
        })
        .unwrap_err();
        assert!(e.contains("mix"), "{e}");
        let e = run(&MoeCase {
            d_in: 256,
            fmts: vec![Fmt::Q(Q::GGUF_IQ3_S)],
            ..base()
        })
        .unwrap_err();
        assert!(e.contains("no packed GEMV"), "{e}");
    }

    // ---- guards on core kernels ----

    /// Two chained steps through persistent state, CPU vs Metal; returns the second step's outputs.
    fn two_steps(g: &Graph, w: &Weights, step: [Vec<Tensor>; 2]) -> (Vec<f32>, Vec<f32>) {
        let s = Session::new(rt(), w).unwrap();
        let (mut sc, mut sm) = (StateStore::new(), StateStore::new());
        let mut last = (vec![], vec![]);
        for inputs in step {
            let want = cpu::run(g, w, &mut sc, inputs.clone())
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            let got = s
                .run_with_state(g, Some(&mut sm), inputs)
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            last = (want, got);
        }
        last
    }

    #[test]
    fn causal_conv_dilated_state_matches_cpu() {
        for (k, d, ch, n) in [
            (2, 1, 1, 1),
            (4, 1, 5, 3),
            (3, 2, 5, 7),
            (4, 3, 37, 2),
            (2, 4, 3, 1),
        ] {
            let mut g = Graph::new();
            let x = g.input("x", vec![n, ch], DType::F32);
            let wv = g.weight("w", vec![ch, k], DType::F32);
            let st = g.state("st", vec![(k - 1) * d, ch], DType::F32);
            let y = g.add(
                "conv",
                Op::CausalConv1dCached {
                    kernel: k,
                    dilation: d,
                },
                &[x, wv, st],
                vec![n, ch],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert("w", Tensor::F32(rand(ch * k, 1)));
            let steps = [
                vec![Tensor::F32(rand(n * ch, 2))],
                vec![Tensor::F32(rand(n * ch, 3))],
            ];
            let (want, got) = two_steps(&g, &w, steps);
            assert!(
                rel_err(&want, &got) < 1e-5,
                "conv k={k} d={d} ch={ch} n={n}"
            );
        }
    }

    #[test]
    fn causal_conv_rejects_mis_sized_state() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let wv = g.weight("w", vec![4, 3], DType::F32);
        let st = g.state("st", vec![2, 4], DType::F32); // dilation 2 needs 4 rows
        let y = g.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 3,
                dilation: 2,
            },
            &[x, wv, st],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("w", Tensor::F32(rand(12, 1)));
        let e = Session::new(rt(), &w)
            .unwrap()
            .run(&g, vec![Tensor::F32(rand(8, 2))])
            .unwrap_err();
        assert!(e.contains("conv") && e.contains("state"), "{e}");
    }

    fn delta_graph(kh: usize, vh: usize, kd: usize, vd: usize, n: usize, beta_len: usize) -> Graph {
        let mut g = Graph::new();
        let q = g.input("q", vec![n, kh * kd], DType::F32);
        let k = g.input("k", vec![n, kh * kd], DType::F32);
        let v = g.input("v", vec![n, vh * vd], DType::F32);
        let beta = g.input("beta", vec![beta_len], DType::F32);
        let ld = g.input("ld", vec![n, vh], DType::F32);
        let st = g.state("ssm", vec![vh * kd * vd], DType::F32);
        let y = g.add(
            "delta",
            Op::DeltaRule {
                key_heads: kh,
                val_heads: vh,
                key_dim: kd,
                val_dim: vd,
            },
            &[q, k, v, beta, ld, st],
            vec![n, vh * vd],
            DType::F32,
        );
        g.output(y);
        g
    }

    #[test]
    fn delta_rule_state_matches_cpu() {
        for (kh, vh, kd, vd, n) in [
            (1, 1, 2, 2, 1),
            (2, 4, 16, 16, 3),
            (2, 2, 8, 37, 5),
            (1, 3, 5, 33, 2),
        ] {
            let g = delta_graph(kh, vh, kd, vd, n, n * vh);
            let step = |s: u64| {
                vec![
                    Tensor::F32(rand(n * kh * kd, s)),
                    Tensor::F32(rand(n * kh * kd, s + 1)),
                    Tensor::F32(rand(n * vh * vd, s + 2)),
                    Tensor::F32(rand(n * vh, s + 3).iter().map(|v| v * 0.5 + 0.5).collect()),
                    Tensor::F32(rand(n * vh, s + 4).iter().map(|v| v * 0.3 - 0.3).collect()),
                ]
            };
            let (want, got) = two_steps(&g, &Weights::new(), [step(10), step(20)]);
            assert!(
                rel_err(&want, &got) < 1e-4,
                "delta {kh}/{vh} {kd}x{vd} n={n}"
            );
        }
    }

    #[test]
    fn delta_rule_rejects_short_gates_and_ragged_heads() {
        let run = |g: &Graph, n_in: [usize; 5]| {
            let inputs = n_in
                .iter()
                .enumerate()
                .map(|(i, &n)| Tensor::F32(rand(n, i as u64)))
                .collect();
            Session::new(rt(), &Weights::new())
                .unwrap()
                .run(g, inputs)
                .unwrap_err()
        };
        let e = run(&delta_graph(1, 1, 2, 2, 3, 1), [6, 6, 6, 1, 3]);
        assert!(e.contains("delta"), "{e}");
        let e = run(&delta_graph(2, 3, 2, 2, 1, 3), [4, 4, 6, 3, 3]);
        assert!(e.contains("multiple"), "{e}");
    }
}
