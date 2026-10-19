//! Metal lowering group `ple`: an m = 1 `GemvPacked` with the `GeGlu` epilogue (`gelu(W_a x) * side`)
//! whose only reader is a plain m = 1 `GemvPacked` (`W_b act`), as one dispatch (opt-in
//! `GIBBS_PLE_FUSED=one`). Every threadgroup recomputes the `mid`-wide (`<= MAX_MID`) `act` vector into
//! threadgroup memory, then produces its grid-strided rows of `W_b act`. Kernels are generated per
//! (codec a, codec b) pair from the codecs' `decode8` and compiled on first use; no static MSL.
//!
//! ponytail: Phase 5 measurement candidate (sprint 3, 1.1). Every threadgroup re-streams `W_a`
//! (`ceil(rows / PROJ_ROWS_PER_TG)` times per call, from cache); default off until an ABBA shows
//! the saved dispatch beats that redundant traffic.

use super::super::{set_bytes, u32_of, Epi, Planned, Session, Step as Core, KERNELS_MSL};
use super::{Enc, ExtStep};
use crate::graph::codec::{codec, Codec};
use crate::graph::codegen::{block_ref, device_msl, gemv_common};
use crate::graph::{Graph, ValueId};
use crate::hardware::QuantFormat;
use metal::{ComputePipelineState, MTLSize};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Largest `act` width the kernel keeps in threadgroup memory (2 KB).
const MAX_MID: usize = 512;
/// Rows of `W_b` per threadgroup (8 simdgroups, grid-strided); sets the threadgroup count.
// ponytail: fixed at 64 (24 threadgroups for 1536 rows); sweep it when Phase 5 measures this kernel.
const PROJ_ROWS_PER_TG: usize = 64;

#[derive(Debug, Clone)]
pub enum Step {
    /// `y = W_b (gelu_tanh(W_a x) * side)`: `x [cols]`, `W_a [mid, cols]`, `side [mid]`, `W_b [rows, mid]`.
    GluGemvChain {
        x: ValueId,
        w_a: ValueId,
        side: ValueId,
        w_b: ValueId,
        cols: usize,
        mid: usize,
        rows: usize,
        fmt_a: QuantFormat,
        fmt_b: QuantFormat,
    },
}

#[cfg(test)]
thread_local! {
    pub(crate) static FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `GIBBS_PLE_FUSED=one`, read once (tests force it per thread with [`FORCE`]).
pub(in super::super) fn enabled() -> bool {
    #[cfg(test)]
    if FORCE.get() {
        return true;
    }
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var("GIBBS_PLE_FUSED").is_ok_and(|v| v == "one"));
    *ON
}

/// Codec of an m = 1 `GemvPacked` whose `cols` are whole blocks of it.
fn gemv_codec(st: &Core, cols: usize) -> Option<&'static Codec> {
    let Core::GemvPacked { format, .. } = st else {
        return None;
    };
    codec(format.codec_format()?).filter(|c| cols % c.block_elems as usize == 0)
}

/// Rewrites each `GemvPacked{GeGlu, side}` → `GemvPacked` pair (see the module doc) into one
/// [`Step::GluGemvChain`] at the first step's position. `act` must have no other reader, view, or
/// graph-output use.
pub(in super::super) fn fuse(
    g: &Graph,
    steps: &mut Vec<Planned>,
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) {
    let root = |v: ValueId| aliases.get(&v).map_or(v, |a| a.0);
    let mut uses: HashMap<ValueId, usize> = HashMap::new();
    for v in steps.iter().flat_map(|s| s.reads.iter()).chain(&g.outputs) {
        *uses.entry(root(*v)).or_default() += 1;
    }
    let mut i = 0;
    while i < steps.len() {
        let a = &steps[i];
        let Core::GemvPacked {
            x,
            w: w_a,
            bias: None,
            res: Some(side),
            epi: Epi::GeGlu,
            m: 1,
            rows: mid,
            cols,
            ..
        } = a.op
        else {
            i += 1;
            continue;
        };
        let act = a.out;
        let found = (mid <= MAX_MID && a.extra_outs.is_empty() && uses.get(&act) == Some(&1))
            .then(|| gemv_codec(&a.op, cols))
            .flatten()
            .and_then(|ca| {
                let j = i + 1 + steps[i + 1..].iter().position(|s| s.reads.contains(&act))?;
                let b = &steps[j];
                match b.op {
                    Core::GemvPacked {
                        x: bx,
                        w: w_b,
                        bias: None,
                        res: None,
                        epi: Epi::None,
                        m: 1,
                        rows,
                        cols: bc,
                        ..
                    } if bx == act && bc == mid && b.extra_outs.is_empty() => {
                        let cb = gemv_codec(&b.op, mid)?;
                        Some((j, w_b, rows, ca.format, cb.format))
                    }
                    _ => None,
                }
            });
        let Some((j, w_b, rows, fmt_a, fmt_b)) = found else {
            i += 1;
            continue;
        };
        let b = steps.remove(j);
        let a = &mut steps[i];
        a.name = format!("{}+{}", a.name, b.name);
        a.op = Core::Ext(ExtStep::Ple(Step::GluGemvChain {
            x,
            w_a,
            side,
            w_b,
            cols,
            mid,
            rows,
            fmt_a,
            fmt_b,
        }));
        a.out = b.out;
        a.reads = vec![x, w_a, side, w_b];
        i += 1;
    }
}

pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    let Step::GluGemvChain {
        x, w_a, side, w_b, ..
    } = *st;
    vec![x, w_a, side, w_b]
}

/// The packed weights.
pub(super) fn allows_non_f32(st: &Step, v: ValueId) -> bool {
    let Step::GluGemvChain { w_a, w_b, .. } = *st;
    v == w_a || v == w_b
}

fn kernel_name(a: &Codec, b: &Codec) -> String {
    format!("glu_gemv_chain_{:?}_{:?}", a.format, b.format)
}

/// Buffers `w_a, x, y, uint3(cols, mid, rows), side, w_b`; 256-thread groups.
fn kernel_src(a: &Codec, b: &Codec) -> String {
    let (da, db) = (
        format!("decode8_{:?}", a.format),
        format!("decode8_{:?}", b.format),
    );
    let mut codecs = device_msl(a).replace("decode8", &da);
    if b.format != a.format {
        codecs += &device_msl(b).replace("decode8", &db);
    }
    let (ba, bb) = (a.block_elems, b.block_elems);
    let ra = block_ref(a, "w_a", &format!("(ulong)r * bpr_a + (s * 8) / {ba}"));
    let rb = block_ref(b, "w_b", &format!("(ulong)row * bpr_b + (s * 8) / {bb}"));
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{common}
{codecs}
kernel void {name}(
    device const uchar* w_a [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint3& p [[buffer(3)]],
    device const float* side [[buffer(4)]],
    device const uchar* w_b [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    threadgroup float act[{MAX_MID}];
    const uint cols = p.x, mid = p.y, rows = p.z;
    const uint bpr_a = cols / {ba}, subs_a = cols / 8;
    for (uint r = sg; r < mid; r += 8) {{
        float acc = 0.0f;
        for (uint s = lane; s < subs_a; s += 32) {{
            {decl_a}
            float v[8];
            {da}(b, {sub_a}((s * 8) % {ba}) / 8, v);
            float4 x0 = *(device const float4*)(x + s * 8);
            float4 x1 = *(device const float4*)(x + s * 8 + 4);
            acc += dot(float4(v[0], v[1], v[2], v[3]), x0) + dot(float4(v[4], v[5], v[6], v[7]), x1);
        }}
        acc = simd_sum(acc);
        if (lane == 0) act[r] = gelu_tanh(acc) * side[r];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bpr_b = mid / {bb}, subs_b = mid / 8;
    for (uint row = tg * 8 + sg; row < rows; row += ntg * 8) {{
        float acc = 0.0f;
        for (uint s = lane; s < subs_b; s += 32) {{
            {decl_b}
            float v[8];
            {db}(b, {sub_b}((s * 8) % {bb}) / 8, v);
            threadgroup const float* av = act + s * 8;
            acc += dot(float4(v[0], v[1], v[2], v[3]), float4(av[0], av[1], av[2], av[3]))
                 + dot(float4(v[4], v[5], v[6], v[7]), float4(av[4], av[5], av[6], av[7]));
        }}
        acc = simd_sum(acc);
        if (lane == 0) y[row] = acc;
    }}
}}
"#,
        common = gemv_common(&KERNELS_MSL),
        name = kernel_name(a, b),
        decl_a = ra.decl,
        sub_a = ra.sub,
        decl_b = rb.decl,
        sub_b = rb.sub,
    )
}

/// Per (device, codec a, codec b) pipelines of [`kernel_src`], compiled on first use.
static PSOS: LazyLock<Mutex<HashMap<(u64, QuantFormat, QuantFormat), ComputePipelineState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn pso(sess: &Session, fa: QuantFormat, fb: QuantFormat) -> Result<ComputePipelineState, String> {
    let key = (sess.rt.device.registry_id(), fa, fb);
    if let Some(p) = PSOS.lock().get(&key) {
        return Ok(p.clone());
    }
    let (a, b) = (
        codec(fa).ok_or_else(|| format!("no codec for {fa:?}"))?,
        codec(fb).ok_or_else(|| format!("no codec for {fb:?}"))?,
    );
    let pso = sess
        .rt
        .compile_generated(&kernel_src(a, b), &kernel_name(a, b))?;
    PSOS.lock().insert(key, pso.clone());
    Ok(pso)
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    let Step::GluGemvChain {
        x,
        w_a,
        side,
        w_b,
        cols,
        mid,
        rows,
        fmt_a,
        fmt_b,
    } = *st;
    let pso = pso(sess, fmt_a, fmt_b)?;
    e.enc.set_compute_pipeline_state(&pso);
    e.bind(0, w_a);
    e.bind(1, x);
    e.bind(2, s.out);
    set_bytes(
        e.enc,
        3,
        &[u32_of(cols)?, u32_of(mid)?, u32_of(rows)?, 0u32],
    );
    e.bind(4, side);
    e.bind(5, w_b);
    e.enc.dispatch_thread_groups(
        MTLSize::new(rows.div_ceil(PROJ_ROWS_PER_TG) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{rand, rel_err, rt};
    use super::super::super::{batch_horizontal, plan, Session, Step as Core};
    use super::super::ExtStep;
    use super::{fuse, Step, FORCE};
    use crate::graph::{cpu, DType, Graph, Op, Tensor, Weights};
    use crate::hardware::QuantFormat;

    const F: DType = DType::F32;

    /// Decode-shaped side-input GLU chain, `layers` times: `side = Slice(scaled per-token input)`,
    /// `x += RmsNorm(W_b (gelu(W_a x) * side))`, then a final RmsNorm (chains into the last norm).
    fn chain_graph(h: usize, mid: usize, layers: usize, fa: QuantFormat, fb: QuantFormat) -> Graph {
        let mut g = Graph::new();
        let mut x = g.input("x", vec![1, h], F);
        let per = g.input("per", vec![1, layers * mid], F);
        let half = g.weight("half", vec![1], F);
        let side_all = g.add("side_all", Op::Mul, &[per, half], vec![1, layers * mid], F);
        for l in 0..layers {
            let (s, e) = ((l * mid) as i64, ((l + 1) * mid) as i64);
            let side = g.add(
                format!("l{l}.slice"),
                Op::Slice {
                    axis: 1,
                    start: s,
                    end: e,
                },
                &[side_all],
                vec![1, mid],
                F,
            );
            let wa = g.weight(format!("l{l}.wa"), vec![mid, h], DType::Quant(fa));
            let gate = g.add(format!("l{l}.gate"), Op::Linear, &[x, wa], vec![1, mid], F);
            let act = g.add(
                format!("l{l}.act"),
                Op::GeGlu,
                &[gate, side],
                vec![1, mid],
                F,
            );
            let wb = g.weight(format!("l{l}.wb"), vec![h, mid], DType::Quant(fb));
            let proj = g.add(format!("l{l}.proj"), Op::Linear, &[act, wb], vec![1, h], F);
            let nw = g.weight(format!("l{l}.nw"), vec![h], F);
            let norm = g.add(
                format!("l{l}.norm"),
                Op::RmsNorm { eps: 1e-6 },
                &[proj, nw],
                vec![1, h],
                F,
            );
            x = g.add(format!("l{l}.res"), Op::Add, &[x, norm], vec![1, h], F);
        }
        let fw = g.weight("final.nw", vec![h], F);
        let y = g.add("final", Op::RmsNorm { eps: 1e-6 }, &[x, fw], vec![1, h], F);
        g.output(y);
        g
    }

    fn kinds(steps: &[super::Planned]) -> Vec<String> {
        steps
            .iter()
            .map(|s| match &s.op {
                Core::GemvPacked { epi, res, .. } => {
                    format!("GemvPacked({epi:?},res={})", res.is_some())
                }
                Core::RmsNorm { res, chain, .. } => {
                    format!("RmsNorm(res={},chain={})", res.is_some(), chain.is_some())
                }
                Core::Ext(ExtStep::Ple(Step::GluGemvChain { .. })) => "GluGemvChain".into(),
                o => super::super::super::kernel_kind(&format!("{o:?}")).to_string(),
            })
            .collect()
    }

    /// Device-free: the chain plans to GeGlu-epilogue GEMV, GEMV, RmsNorm+residual(+chain) per layer,
    /// and `fuse` turns the two GEMVs into one `GluGemvChain`.
    #[test]
    fn glu_gemv_chain_plans_to_one_step() {
        let g = chain_graph(
            512,
            256,
            2,
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_K_M,
        );
        let (steps, aliases) = plan(&g).unwrap();
        let base = kinds(&batch_horizontal(steps.clone(), &aliases));
        eprintln!("default: {base:?}");
        let layer = [
            "GemvPacked(GeGlu,res=true)",
            "GemvPacked(None,res=false)",
            "RmsNorm(res=true,chain=false)",
        ];
        assert_eq!(base[1..4], layer);
        assert_eq!(base[4..6], layer[..2]);
        assert_eq!(base[6], "RmsNorm(res=true,chain=true)");
        assert_eq!(
            base.len(),
            7,
            "Mul + 2 x 3 layer steps (final norm chained)"
        );

        let mut fused = steps;
        fuse(&g, &mut fused, &aliases);
        let fused = kinds(&batch_horizontal(fused, &aliases));
        eprintln!("fused: {fused:?}");
        assert_eq!(
            fused[1..],
            [
                "GluGemvChain",
                "RmsNorm(res=true,chain=false)",
                "GluGemvChain",
                "RmsNorm(res=true,chain=true)",
            ]
        );
    }

    /// Device-free: a second reader of `act`, or `mid > 512`, keeps the two GEMVs.
    #[test]
    fn glu_gemv_chain_guards() {
        let mut g = chain_graph(512, 256, 1, QuantFormat::GGUF_Q4_0, QuantFormat::GGUF_Q4_0);
        let act = g.nodes.iter().find(|n| n.name == "l0.act").unwrap().outputs[0];
        g.output(act);
        let (mut steps, aliases) = plan(&g).unwrap();
        let n = steps.len();
        fuse(&g, &mut steps, &aliases);
        assert_eq!(steps.len(), n);

        let g = chain_graph(
            1024,
            1024,
            1,
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q4_0,
        );
        let (mut steps, aliases) = plan(&g).unwrap();
        let n = steps.len();
        fuse(&g, &mut steps, &aliases);
        assert_eq!(steps.len(), n);
    }

    /// GGUF blocks with random codes and small finite f16 scales at the codec's scale offsets.
    fn blocks(f: QuantFormat, n_blocks: usize, seed: u64) -> Vec<u8> {
        let (_, bb, _) = crate::model::layers::gguf_block_codec(f).unwrap();
        let scale_at: &[usize] = match f {
            QuantFormat::GGUF_Q4_K_M => &[0, 2],
            _ => &[0],
        };
        let r = rand(n_blocks * bb, seed);
        let mut raw: Vec<u8> = r.iter().map(|v| ((v + 1.0) * 127.9) as u8).collect();
        for (i, blk) in raw.chunks_exact_mut(bb).enumerate() {
            for &o in scale_at {
                let d = crate::model::f32_to_f16(0.004 * (1 + i % 4) as f32);
                blk[o..o + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        raw
    }

    fn quant(f: QuantFormat, rows: usize, cols: usize, seed: u64) -> Tensor {
        let (be, _, _) = crate::model::layers::gguf_block_codec(f).unwrap();
        Tensor::Quantized(std::sync::Arc::new(
            crate::model::layers::QuantizedWeight::GgufQuantized {
                format: f,
                raw_data: blocks(f, rows * cols / be, seed).into(),
                in_features: cols,
                out_features: rows,
            },
        ))
    }

    /// Metal vs CPU for the default (GeGlu-epilogue GEMV + GEMV) and the forced one-dispatch form,
    /// on Q4_0, Q4_K and a mixed pair.
    #[test]
    fn glu_gemv_chain_matches_cpu() {
        use QuantFormat::{GGUF_Q4_0 as Q4, GGUF_Q4_K_M as QK};
        let (h, mid, layers) = (512usize, 256usize, 2usize);
        for (fa, fb) in [(Q4, Q4), (QK, QK), (Q4, QK)] {
            let g = chain_graph(h, mid, layers, fa, fb);
            let mut w = Weights::new();
            w.insert("half", Tensor::F32(vec![std::f32::consts::FRAC_1_SQRT_2]));
            w.insert("final.nw", Tensor::F32(rand(h, 9)));
            for l in 0..layers {
                let s = 10 * l as u64;
                w.insert(format!("l{l}.wa"), quant(fa, mid, h, s + 1));
                w.insert(format!("l{l}.wb"), quant(fb, h, mid, s + 2));
                w.insert(format!("l{l}.nw"), Tensor::F32(rand(h, s + 3)));
            }
            let inputs = vec![
                Tensor::F32(rand(h, 100)),
                Tensor::F32(rand(layers * mid, 101)),
            ];
            let want = cpu::run_stateless(&g, &w, inputs.clone())
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            for force in [false, true] {
                FORCE.set(force);
                let s = Session::new(rt(), &w).unwrap();
                let n_fused = s
                    .prepare_plan(&g)
                    .unwrap()
                    .steps
                    .iter()
                    .filter(|st| matches!(st.op, Core::Ext(ExtStep::Ple(_))))
                    .count();
                let got = s
                    .run(&g, inputs.clone())
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                FORCE.set(false);
                assert_eq!(n_fused, if force { layers } else { 0 }, "{fa:?}/{fb:?}");
                let err = rel_err(&want, &got);
                assert!(err < 1e-4, "{fa:?}/{fb:?} force={force}: rel err {err}");
            }
        }
    }
}
