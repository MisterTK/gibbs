//! Metal lowering group `moe`: planning, encoding and kernels for its ops.
//! Kernels live in `moe.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Owns: `TopK` (row top-k indices), `GatherCols`, `MoeLinear` (expert-indexed GEMV over a stacked
//! weight bank, generated per codec by `codegen::gemv_indexed`, f32/f16/bf16 banks through
//! pseudo-codecs), `Embedding` over F16/BF16/codec-quantized tables, and shape guards for
//! `CausalConv1dCached` / `DeltaRule` forms the core kernels would read out of bounds.

use super::super::{numel, set_bytes, u32_of, Planned, Session, TG};
use super::{Enc, PlanOut};
use crate::graph::codec::{codec, Codec};
use crate::graph::codegen::{
    block_ref, device_msl, gemv_indexed, gemv_indexed_name, gemv_rows_per_tg, GemvParams,
};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use crate::hardware::QuantFormat;
use metal::{CompileOptions, ComputePipelineState, MTLSize};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;

pub(super) const MSL: &str = include_str!("moe.metal");
/// Compute functions in `moe.metal`.
pub(super) const KERNELS: &[&str] = &[
    "moe_topk_f32",
    "moe_topk_softmax_f32",
    "moe_gather_cols_f32",
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
    TopK {
        x: ValueId,
        rows: usize,
        cols: usize,
        k: usize,
        weights_out: Option<ValueId>,
        scale: Option<ValueId>,
    },
    GatherCols {
        x: ValueId,
        idx: ValueId,
        total: usize,
        cols: usize,
        k: usize,
        /// `x` is one row shared by every row of `idx`.
        shared: bool,
    },
    ExpertLinear {
        x: ValueId,
        bank: ValueId,
        ids: ValueId,
        /// Storage dtype of the bank (`bank_codec` reads it).
        dtype: DType,
        /// Rows of one expert's `[rows, cols]` matrix.
        rows: usize,
        cols: usize,
        slots: usize,
        top_k: usize,
        experts: usize,
        x_per_slot: bool,
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

/// Float banks read through the same `decode8` contract as the quantized codecs: one 8-element block.
static CODEC_F32: Codec = Codec {
    format: QuantFormat::FP32,
    block_elems: 8,
    block_bytes: 32,
    msl: "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n    device const float* f = (device const float*)blk;\n    for (uint i = 0; i < 8; ++i) out8[i] = f[i];\n}\n",
};
static CODEC_F16: Codec = Codec {
    format: QuantFormat::FP16,
    block_elems: 8,
    block_bytes: 16,
    msl: "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n    device const half* h = (device const half*)blk;\n    for (uint i = 0; i < 8; ++i) out8[i] = float(h[i]);\n}\n",
};
static CODEC_BF16: Codec = Codec {
    format: QuantFormat::BF16,
    block_elems: 8,
    block_bytes: 16,
    msl: "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n    device const ushort* h = (device const ushort*)blk;\n    for (uint i = 0; i < 8; ++i) out8[i] = as_type<float>(uint(h[i]) << 16);\n}\n",
};

/// How the expert GEMV reads a bank stored as `dtype`: every codec format, plus f32/f16/bf16.
fn bank_codec(dtype: DType) -> Option<&'static Codec> {
    match dtype {
        DType::F32 => Some(&CODEC_F32),
        DType::F16 => Some(&CODEC_F16),
        DType::BF16 => Some(&CODEC_BF16),
        DType::Quant(f) => codec(f),
        DType::U32 => None,
    }
}

/// Whether the Metal executor lowers `op` for `dtype`: the activation dtype for `TopK`/`GatherCols`,
/// the bank dtype for `MoeLinear`, the table dtype for `Embedding`.
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    match op {
        Op::TopK { .. } | Op::GatherCols => dtype == DType::F32,
        Op::MoeLinear { .. } => bank_codec(dtype).is_some(),
        Op::Embedding => match dtype {
            DType::F16 | DType::BF16 => true,
            DType::Quant(f) => codec(f).is_some(),
            _ => false,
        },
        _ => false,
    }
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
        Op::TopK { k } => Some((|| {
            let x = arg(0)?;
            let cols = g.value(x).shape.last().copied().unwrap_or(0);
            if *k == 0 || *k > cols {
                return Err(format!("node '{name}': TopK needs 1 <= k <= the row width {cols}, got k={k}"));
            }
            let n = numel(g, x)?;
            u32_of(*k)?;
            Ok(PlanOut::Step(
                Step::TopK {
                    x,
                    rows: n / cols,
                    cols,
                    k: *k,
                    weights_out: None,
                    scale: None,
                },
                n / cols * k,
            ))
        })()),
        Op::GatherCols => Some((|| {
            let (x, idx) = (arg(0)?, arg(1)?);
            let (cols, k) = (
                g.value(x).shape.last().copied().unwrap_or(0),
                g.value(idx).shape.last().copied().unwrap_or(0),
            );
            let total = numel(g, idx)?;
            let x_rows = numel(g, x)?.checked_div(cols).unwrap_or(0);
            if cols == 0 || k == 0 || (x_rows != total / k && x_rows != 1) || g.value(idx).dtype != DType::U32 {
                return Err(format!(
                    "node '{name}': GatherCols needs x [rows, n] (or one shared row) and u32 idx [rows, k]"
                ));
            }
            u32_of(total)?;
            Ok(PlanOut::Step(Step::GatherCols { x, idx, total, cols, k, shared: x_rows == 1 }, total))
        })()),
        Op::MoeLinear { experts, top_k } => Some((|| {
            if node.inputs.len() != 3 {
                return Err(format!("node '{name}': MoeLinear needs [x, bank, ids], got {} operands", node.inputs.len()));
            }
            let (x, bank, ids) = (arg(0)?, arg(1)?, arg(2)?);
            if *experts == 0 || *top_k == 0 {
                return Err(format!("node '{name}': MoeLinear needs experts >= 1 and top_k >= 1"));
            }
            let (Some([x_rows, cols]), Some([bank_rows, bank_cols])) = (shape2(g, x), shape2(g, bank)) else {
                return Err(format!("node '{name}': MoeLinear needs x [m, in] and a bank [experts * out, in]"));
            };
            if bank_cols != cols || bank_rows % experts != 0 || bank_rows == 0 {
                return Err(format!(
                    "node '{name}': bank [{bank_rows}, {bank_cols}] is not {experts} stacked [out, {cols}] matrices"
                ));
            }
            let slots = numel(g, ids)?;
            if g.value(ids).dtype != DType::U32 || slots % top_k != 0 {
                return Err(format!("node '{name}': ids must be u32 [m, {top_k}]"));
            }
            if x_rows != slots && x_rows * top_k != slots {
                return Err(format!(
                    "node '{name}': x has {x_rows} rows, neither m nor m * {top_k} for {slots} slots"
                ));
            }
            let rows = bank_rows / experts;
            if numel(g, node.outputs[0])? != slots * rows {
                return Err(format!("node '{name}': output must be [{slots}, {rows}]"));
            }
            let dtype = g.value(bank).dtype;
            let c = bank_codec(dtype).ok_or_else(|| {
                format!("node '{name}': bank '{}' ({dtype:?}) has no GEMV codec", g.value(bank).name)
            })?;
            if cols % c.block_elems as usize != 0 {
                return Err(format!(
                    "node '{name}': {cols} columns are not a multiple of the {:?} block ({})",
                    c.format, c.block_elems
                ));
            }
            for n in [rows, cols, slots, *top_k, *experts] {
                u32_of(n)?;
            }
            Ok(PlanOut::Step(
                Step::ExpertLinear {
                    x,
                    bank,
                    ids,
                    dtype,
                    rows,
                    cols,
                    slots,
                    top_k: *top_k,
                    experts: *experts,
                    x_per_slot: x_rows == slots,
                },
                slots * rows,
            ))
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

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match st {
        Step::TopK { x, scale, .. } => {
            let mut v = vec![*x];
            if let Some(sc) = *scale {
                v.push(sc);
            }
            v
        }
        Step::GatherCols { x, idx, .. } => vec![*x, *idx],
        Step::ExpertLinear { x, bank, ids, .. } => vec![*x, *bank, *ids],
        Step::Embed { table, ids, .. } => vec![*table, *ids],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, v: ValueId) -> bool {
    match st {
        Step::TopK { .. } => false,
        Step::GatherCols { idx, .. } => v == *idx,
        Step::ExpertLinear { x, .. } => v != *x,
        Step::Embed { .. } => true,
    }
}

/// Generated gather for one codec: one thread per 8-element sub-block of one token's row.
fn quant_embed_src(c: &Codec) -> String {
    let be = c.block_elems;
    let br = block_ref(
        c,
        "table",
        &format!("(ulong)ids[gid.y] * (d / {be}u) + blk"),
    );
    let (decl, sub) = (&br.decl, &br.sub);
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
    {decl}
    float v[8];
    decode8(b, {sub}sub, v);
    for (uint i = 0; i < 8; ++i) out[(ulong)gid.y * d + gid.x * 8 + i] = v[i];
}}
"#,
        msl = device_msl(c)
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

/// Rows per simdgroup of the expert GEMV.
const ROWS_PER_SG: usize = 2;

/// Per (device, codec) pipelines of [`gemv_indexed`], compiled on first use.
static MOE_GEMV: LazyLock<Mutex<HashMap<(u64, QuantFormat), ComputePipelineState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn moe_gemv_pso(sess: &Session, c: &Codec) -> Result<ComputePipelineState, String> {
    let device = &sess.rt.device;
    let key = (device.registry_id(), c.format);
    if let Some(p) = MOE_GEMV.lock().get(&key) {
        return Ok(p.clone());
    }
    let params = GemvParams {
        rows_per_sg: ROWS_PER_SG,
        ..Default::default()
    };
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}",
        gemv_indexed(c, params)
    );
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).map_err(|e| {
        format!(
            "generated {:?} expert GEMV failed to compile: {e}",
            c.format
        )
    })?;
    let func = lib.get_function(&gemv_indexed_name(c, params), None)?;
    let pso = device.new_compute_pipeline_state_with_function(&func)?;
    MOE_GEMV.lock().insert(key, pso.clone());
    Ok(pso)
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    let (enc, rt) = (e.enc, &sess.rt);
    match st {
        Step::TopK {
            x,
            rows,
            cols,
            k,
            weights_out,
            scale,
        } => {
            if let Some(w_out) = *weights_out {
                enc.set_compute_pipeline_state(rt.pso("moe_topk_softmax_f32")?);
                e.bind(0, *x);
                e.bind(1, s.out);
                e.bind(2, w_out);
                e.bind(3, scale.unwrap_or(*x));
                set_bytes(
                    enc,
                    4,
                    &[
                        u32_of(*rows)?,
                        u32_of(*cols)?,
                        u32_of(*k)?,
                        scale.is_some() as u32,
                    ],
                );
                enc.dispatch_thread_groups(MTLSize::new(*rows as u64, 1, 1), MTLSize::new(256, 1, 1));
            } else {
                enc.set_compute_pipeline_state(rt.pso("moe_topk_f32")?);
                e.bind(0, *x);
                e.bind(1, s.out);
                set_bytes(enc, 2, &[u32_of(*rows)?, u32_of(*cols)?, u32_of(*k)?]);
                enc.dispatch_thread_groups(MTLSize::new(*rows as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
        }
        Step::GatherCols {
            x,
            idx,
            total,
            cols,
            k,
            shared,
        } => {
            enc.set_compute_pipeline_state(rt.pso("moe_gather_cols_f32")?);
            e.bind(0, *x);
            e.bind(1, *idx);
            e.bind(2, s.out);
            set_bytes(
                enc,
                3,
                &[u32_of(*total)?, u32_of(*cols)?, u32_of(*k)?, *shared as u32],
            );
            e.grid1(*total);
        }
        Step::ExpertLinear {
            x,
            bank,
            ids,
            dtype,
            rows,
            cols,
            slots,
            top_k,
            experts,
            x_per_slot,
        } => {
            let c = bank_codec(*dtype)
                .ok_or_else(|| format!("node '{}': no GEMV codec for {dtype:?}", s.name))?;
            let pso = moe_gemv_pso(sess, c)?;
            enc.set_compute_pipeline_state(&pso);
            e.bind(0, *bank);
            e.bind(1, *x);
            e.bind(2, s.out);
            set_bytes(
                enc,
                3,
                &[
                    u32_of(*rows)?,
                    u32_of(*cols)?,
                    u32_of(*slots)?,
                    u32_of(*top_k)?,
                    u32_of(*experts)?,
                    *x_per_slot as u32,
                ],
            );
            e.bind(4, *ids);
            let rows_per_tg = gemv_rows_per_tg(GemvParams {
                rows_per_sg: ROWS_PER_SG,
                ..Default::default()
            });
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (*rows as u64).div_ceil(rows_per_tg as u64),
                    *slots as u64,
                    1,
                ),
                MTLSize::new(256, 1, 1),
            );
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
        let f16 = |r: &mut Lcg| -> u16 {
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

    /// Random blocks of `f` that decode to finite, moderate values on the CPU (formats with unknown
    /// scale fields, like the repacked ones, get bad blocks redrawn).
    fn sane_blocks(f: Q, n_blocks: usize, seed: u64) -> Vec<u8> {
        let (_, _, dequant) = crate::model::layers::gguf_block_codec(f).expect("CPU dequantizer");
        let be = crate::graph::codec::codec(f).expect("codec").block_elems as usize;
        let mut out = Vec::new();
        for b in 0..n_blocks {
            for attempt in 0..1000u64 {
                let blk = random_blocks(f, 1, true, seed * 1_000_003 + b as u64 * 1009 + attempt);
                let mut y = vec![0.0f32; be];
                dequant(&blk, &mut y);
                if y.iter().all(|v| v.is_finite() && v.abs() < 4096.0) {
                    out.extend(blk);
                    break;
                }
                assert!(attempt < 999, "{f:?}: no sane random block found");
            }
        }
        out
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
                let raw = sane_blocks(f, rows * cols / be, seed);
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

    // ---- TopK / GatherCols ----

    /// (CPU reference, Metal) indices; the Metal output carries u32 bits in its f32 slots.
    fn topk_both(x: Vec<f32>, rows: usize, cols: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
        let mut g = Graph::new();
        let xv = g.input("x", vec![rows, cols], DType::F32);
        let y = g.add("tk", Op::TopK { k }, &[xv], vec![rows, k], DType::U32);
        g.output(y);
        let inputs = vec![Tensor::F32(x)];
        let want = match cpu::run_stateless(&g, &Weights::new(), inputs.clone())
            .unwrap()
            .remove(0)
        {
            Tensor::U32(v) => v,
            t => panic!("TopK must produce u32 indices, got {t:?}"),
        };
        let got = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(&g, inputs)
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        (want, got)
    }

    #[test]
    fn topk_orders_by_value_then_lower_index() {
        let (want, got) = topk_both(vec![1.0, 3.0, 3.0, 2.0, -0.0, 0.0, f32::NAN, 5.0], 2, 4, 3);
        assert_eq!(
            want,
            vec![1, 2, 3, 3, 0, 1],
            "ties and signed zeros keep the lower index first"
        );
        assert_eq!(got, want);
        // NaN ranks below every number: it is picked last even when k covers the whole row.
        let (want, got) = topk_both(vec![f32::NAN, 1.0, f32::NAN, 2.0], 1, 4, 4);
        assert_eq!(want, vec![3, 1, 0, 2]);
        assert_eq!(got, want);
    }

    #[test]
    fn topk_matches_cpu_across_shapes_and_ties() {
        for (rows, cols, k) in [
            (1, 1, 1),
            (1, 2, 2),
            (3, 8, 3),
            (4, 128, 8),
            (2, 129, 129),
            (1, 1000, 64),
            (5, 300, 1),
            (2, 2048, 8),
            (7, 37, 37),
        ] {
            let seed = (rows * 31 + cols) as u64;
            let distinct = rand(rows * cols, seed);
            // A handful of levels: most values tie.
            let tied: Vec<f32> = distinct.iter().map(|v| (v * 3.0).round()).collect();
            let constant = vec![0.5f32; rows * cols];
            for (name, x) in [
                ("distinct", distinct),
                ("tied", tied),
                ("constant", constant),
            ] {
                let (want, got) = topk_both(x, rows, cols, k);
                assert_eq!(got, want, "topk {name} {rows}x{cols} k={k}");
            }
        }
    }

    #[test]
    fn topk_rejects_bad_k() {
        for k in [0, 5] {
            let mut g = Graph::new();
            let x = g.input("x", vec![2, 4], DType::F32);
            let y = g.add("tk", Op::TopK { k }, &[x], vec![2, k.max(1)], DType::U32);
            g.output(y);
            let inputs = vec![Tensor::F32(vec![0.0; 8])];
            let e = Session::new(rt(), &Weights::new())
                .unwrap()
                .run(&g, inputs.clone())
                .unwrap_err();
            assert!(e.contains("tk"), "{e}");
            assert!(cpu::run_stateless(&g, &Weights::new(), inputs)
                .unwrap_err()
                .contains("tk"));
        }
    }

    #[test]
    fn gather_cols_follows_topk_indices() {
        for (rows, cols, k) in [(1, 8, 3), (4, 100, 8), (3, 37, 37)] {
            let mut g = Graph::new();
            let x = g.input("x", vec![rows, cols], DType::F32);
            let ids = g.add("ids", Op::TopK { k }, &[x], vec![rows, k], DType::U32);
            let y = g.add("top", Op::GatherCols, &[x, ids], vec![rows, k], DType::F32);
            g.output(y);
            let xs = rand(rows * cols, 99 + cols as u64);
            let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(xs.clone())]);
            assert_eq!(want, got);
            for (r, row) in want.chunks(k).enumerate() {
                let mut sorted = xs[r * cols..(r + 1) * cols].to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a));
                assert_eq!(
                    row,
                    &sorted[..k],
                    "row {r} holds the k largest values, descending"
                );
            }
        }
    }

    #[test]
    fn gather_cols_shares_a_single_row() {
        // One row of per-expert scales indexed by the ids of three tokens.
        let (n, e, k) = (3, 8, 2);
        let mut g = Graph::new();
        let table = g.input("table", vec![1, e], DType::F32);
        let ids = g.input("ids", vec![n, k], DType::U32);
        let y = g.add(
            "scale",
            Op::GatherCols,
            &[table, ids],
            vec![n, k],
            DType::F32,
        );
        g.output(y);
        let table = rand(e, 5);
        let ids = vec![0u32, 7, 3, 3, 6, 1];
        let inputs = vec![Tensor::F32(table.clone()), Tensor::U32(ids.clone())];
        let (want, got) = both(&g, &Weights::new(), inputs);
        assert_eq!(
            want,
            ids.iter().map(|&i| table[i as usize]).collect::<Vec<_>>()
        );
        assert_eq!(got, want);
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

    // ---- MoeLinear ----

    /// `(CPU executor, Metal)` outputs of one `MoeLinear` over a random `[experts * rows, cols]` bank;
    /// `x_rows` is `ids.len() / top_k` (broadcast) or `ids.len()` (one row per slot).
    #[allow(clippy::too_many_arguments)]
    fn moe_both(
        fmt: Fmt,
        x_rows: usize,
        cols: usize,
        rows: usize,
        experts: usize,
        top_k: usize,
        ids: Vec<u32>,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let slots = ids.len();
        let (dt, bank) = weight(fmt, experts * rows, cols, seed);
        let mut g = Graph::new();
        let x = g.input("x", vec![x_rows, cols], DType::F32);
        let b = g.weight("bank", vec![experts * rows, cols], dt);
        let idv = g.input("ids", vec![slots / top_k, top_k], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear { experts, top_k },
            &[x, b, idv],
            vec![slots, rows],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("bank", bank);
        both(
            &g,
            &w,
            vec![Tensor::F32(rand(x_rows * cols, seed + 7)), Tensor::U32(ids)],
        )
    }

    #[test]
    fn moe_linear_matches_cpu_for_every_format() {
        let mut fmts = vec![Fmt::F32, Fmt::F16, Fmt::BF16];
        fmts.extend(crate::graph::codec::formats().into_iter().map(Fmt::Q));
        let (experts, top_k, rows) = (6, 3, 19);
        let ids = vec![4, 0, 5, 2, 2, 1];
        for (i, fmt) in fmts.iter().enumerate() {
            let cols = match fmt {
                Fmt::Q(f) => {
                    let be = crate::graph::codec::codec(*f).expect("codec").block_elems as usize;
                    (512 / be).max(1) * be
                }
                _ => 64,
            };
            // Broadcast x (2 rows x 3 slots) and one x row per slot.
            for x_rows in [2, 6] {
                let (want, got) = moe_both(
                    *fmt,
                    x_rows,
                    cols,
                    rows,
                    experts,
                    top_k,
                    ids.clone(),
                    10 + i as u64,
                );
                let err = rel_err(&want, &got);
                assert!(err < 1e-4, "{fmt:?} x_rows={x_rows}: {err}");
            }
        }
    }

    #[test]
    fn moe_linear_shapes_and_routing_match_cpu() {
        let mut s = 7u64;
        let mut route = |n: usize, experts: usize| -> Vec<u32> {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) % experts as u64) as u32
                })
                .collect()
        };
        for (i, (m, top_k, experts, rows, cols)) in [
            (1, 1, 1, 8, 8),
            (1, 8, 128, 17, 64), // a decode token: 8 of 128 experts
            (37, 2, 12, 33, 40), // prefill-sized, rows not a tile multiple
            (3, 4, 4, 129, 24),  // every expert routed several times
            (2, 1, 5, 1, 16),    // one output row per expert
        ]
        .into_iter()
        .enumerate()
        {
            let ids = route(m * top_k, experts);
            for x_rows in [m, m * top_k] {
                let (want, got) = moe_both(
                    Fmt::F32,
                    x_rows,
                    cols,
                    rows,
                    experts,
                    top_k,
                    ids.clone(),
                    40 + i as u64,
                );
                assert!(
                    rel_err(&want, &got) < 1e-4,
                    "m={m} top_k={top_k} E={experts} {rows}x{cols} x_rows={x_rows}"
                );
            }
        }
    }

    /// An expert no slot routes to is never read: NaN weights in it leave the output untouched.
    #[test]
    fn moe_linear_reads_only_routed_experts() {
        let (experts, rows, cols) = (4, 16, 32);
        let mut g = Graph::new();
        let x = g.input("x", vec![1, cols], DType::F32);
        let b = g.weight("bank", vec![experts * rows, cols], DType::F32);
        let ids = g.input("ids", vec![1, 2], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear { experts, top_k: 2 },
            &[x, b, ids],
            vec![2, rows],
            DType::F32,
        );
        g.output(y);
        let mut bank = rand(experts * rows * cols, 5);
        let run = |bank: &[f32]| {
            let mut w = Weights::new();
            w.insert("bank", Tensor::F32(bank.to_vec()));
            Session::new(rt(), &w)
                .unwrap()
                .run(
                    &g,
                    vec![Tensor::F32(rand(cols, 1)), Tensor::U32(vec![2, 0])],
                )
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap()
        };
        let clean = run(&bank);
        bank[3 * rows * cols..]
            .iter_mut()
            .for_each(|v| *v = f32::NAN);
        let poisoned = run(&bank);
        assert!(clean.iter().all(|v| v.is_finite()));
        assert_eq!(clean, poisoned);
    }

    #[test]
    fn moe_linear_rejects_malformed_nodes() {
        // (experts, top_k, bank rows, cols, x rows, ids, bank format) -> error must name the node.
        let run = |experts: usize,
                   top_k: usize,
                   bank_rows: usize,
                   cols: usize,
                   x_rows: usize,
                   fmt: Fmt| {
            let (dt, bank) = weight(fmt, bank_rows, cols, 3);
            let mut g = Graph::new();
            let x = g.input("x", vec![x_rows, cols], DType::F32);
            let b = g.weight("bank", vec![bank_rows, cols], dt);
            let ids = g.input("ids", vec![1, 2], DType::U32);
            let y = g.add(
                "moe",
                Op::MoeLinear { experts, top_k },
                &[x, b, ids],
                vec![2, 4],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert("bank", bank);
            Session::new(rt(), &w)
                .unwrap()
                .run(
                    &g,
                    vec![
                        Tensor::F32(vec![0.5; x_rows * cols]),
                        Tensor::U32(vec![0, 1]),
                    ],
                )
                .map(|_| ())
        };
        run(4, 2, 16, 8, 1, Fmt::F32).unwrap();
        let e = run(4, 0, 16, 8, 1, Fmt::F32).unwrap_err();
        assert!(e.contains("moe") && e.contains("top_k"), "{e}");
        let e = run(5, 2, 16, 8, 1, Fmt::F32).unwrap_err();
        assert!(e.contains("moe") && e.contains("stacked"), "{e}");
        let e = run(4, 2, 16, 8, 3, Fmt::F32).unwrap_err();
        assert!(e.contains("moe") && e.contains("rows"), "{e}");
        // 16 columns are not a whole Q4_K block; planning fails before any weight is read.
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 16], DType::F32);
        let b = g.weight("bank", vec![16, 16], DType::Quant(Q::GGUF_Q4_K_M));
        let ids = g.input("ids", vec![1, 2], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear {
                experts: 4,
                top_k: 2,
            },
            &[x, b, ids],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let e = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(
                &g,
                vec![Tensor::F32(vec![0.5; 16]), Tensor::U32(vec![0, 1])],
            )
            .unwrap_err();
        assert!(e.contains("moe") && e.contains("block"), "{e}");
    }

    #[test]
    fn moe_linear_cpu_reference_rejects_out_of_range_experts() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 8], DType::F32);
        let b = g.weight("bank", vec![8, 8], DType::F32);
        let ids = g.input("ids", vec![1, 1], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear {
                experts: 2,
                top_k: 1,
            },
            &[x, b, ids],
            vec![1, 4],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("bank", Tensor::F32(rand(64, 1)));
        let e = cpu::run_stateless(&g, &w, vec![Tensor::F32(rand(8, 2)), Tensor::U32(vec![2])])
            .unwrap_err();
        assert!(e.contains("routes to expert 2"), "{e}");
    }

    /// A routed id past the bank writes zeros on Metal, never stale arena bytes.
    #[test]
    fn moe_linear_metal_zeroes_out_of_range_slots() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 8], DType::F32);
        let b = g.weight("bank", vec![8, 8], DType::F32);
        let ids = g.input("ids", vec![1, 2], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear {
                experts: 2,
                top_k: 2,
            },
            &[x, b, ids],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("bank", Tensor::F32(rand(64, 1)));
        let got = Session::new(rt(), &w)
            .unwrap()
            .run(&g, vec![Tensor::F32(rand(8, 2)), Tensor::U32(vec![1, 7])])
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        assert!(got[..4].iter().any(|v| *v != 0.0));
        assert_eq!(&got[4..], &[0.0; 4]);
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

    #[test]
    #[ignore = "bandwidth measurement"]
    fn moe_gemv_bandwidth() {
        let (experts, top_k) = (16usize, 8usize);
        let ids = Tensor::U32(vec![0, 2, 5, 7, 9, 11, 13, 15]);
        for (f, rows, cols, x_rows) in [
            (Q::GGUF_IQ2_XXS, 1408usize, 2816usize, 1usize),
            (Q::GGUF_IQ2_M, 1408, 2816, 1),
            (Q::GGUF_IQ4_NL, 2816, 704, 8),
            (Q::GGUF_Q4_0, 2816, 704, 8),
        ] {
            let c = crate::graph::codec::codec(f).unwrap();
            let (dt, bank) = weight(Fmt::Q(f), experts * rows, cols, 1);
            let routed_bytes =
                top_k * rows * (cols / c.block_elems as usize) * c.block_bytes as usize;
            let mut g = Graph::new();
            let x = g.input("x", vec![x_rows, cols], DType::F32);
            let b = g.weight("bank", vec![experts * rows, cols], dt);
            let id_in = g.input("ids", vec![1, top_k], DType::U32);
            for step in 0..8 {
                let y = g.add(
                    format!("moe_{step}"),
                    Op::MoeLinear { experts, top_k },
                    &[x, b, id_in],
                    vec![top_k, rows],
                    DType::F32,
                );
                g.output(y);
            }
            let mut w = Weights::new();
            w.insert("bank", bank);
            let s = Session::new(rt(), &w).unwrap();
            let plan = s.prepare_plan(&g).unwrap();
            let xs = Tensor::F32(vec![0.25; x_rows * cols]);
            for _ in 0..10 {
                s.run_prepared(&g, &plan, None, vec![xs.clone(), ids.clone()])
                    .unwrap();
            }
            let mut dt_s = 0.0;
            for _ in 0..20 {
                s.run_prepared(&g, &plan, None, vec![xs.clone(), ids.clone()])
                    .unwrap();
                dt_s += s.last_gpu_seconds() / (20.0 * 8.0);
            }
            println!(
                "moe_gemv {f:?} {rows}x{cols} (top_k={top_k}, x_rows={x_rows}): {:.1} us, {:.1} GB/s",
                dt_s * 1e6,
                routed_bytes as f64 / dt_s / 1e9
            );
        }
    }
}
