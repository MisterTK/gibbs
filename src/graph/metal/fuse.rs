//! Planner fusions as e-graph rules. [`decide`] encodes the lowered graph's nodes, the operand
//! facts the Metal kernels need (element counts, producer order, op parameters, weight formats),
//! saturates the rules below, and returns the fusions `plan` lowers: a Linear absorbing its single
//! reader as a GEMV epilogue, an RmsNorm absorbing a residual Add, a scalar Mul and/or a RoPE, and a
//! cached attention absorbing its q/k RoPEs.
//!
//! Every rule claims a node read by exactly one other node, and the operands it captures must exist
//! before the claiming node, so at most one rule claims any node: extraction takes every match.

use super::{
    fusion_policy, linear_format, numel, Epi, FusionClass, Planned, Step, WeightFormat,
    ATTN_MAX_HEAD_DIM,
};
use crate::graph::{DType, Graph, Op, RopeVariant, ValueId, ValueKind};
use egglog::{EGraph, Write as _};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

const PROGRAM: &str = r#"
(datatype* (E (V i64) (Node i64 L)) (L (Nil) (Cons E L)))
(relation P (i64 String))
(function Uses (E) i64 :no-merge)
(function Pos (E) i64 :no-merge)
(function Numel (E) i64 :no-merge)
(function Vi (E) i64 :no-merge)
(relation NotOut (E))
(relation NotNorm (E))
(relation F32 (E))
(relation LinMN (i64 i64))
(relation PackedEpi (i64))
(relation NormP (i64 i64 i64))
(relation RopeP (i64 i64 i64 i64 i64))
(relation AttnP (i64 i64 i64 i64))
(relation Epi (i64 i64 i64 i64))
(relation NormRes (i64 i64 i64))
(relation NormScale (i64 i64 i64))
(relation NormRope (i64 i64))
(relation AttnRope (i64 i64 i64))
(ruleset plan)

; Linear `li` (output `l`, one reader `j`) → GEMV epilogue (Epi li j kind side-value|-1).
(rule ((P li "LinEpi") (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons l r))) (P j "GeluErf"))
      ((Epi li j 1 -1)) :ruleset plan)
(rule ((P li "LinEpi") (PackedEpi li) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons l r))) (P j "Silu"))
      ((Epi li j 2 -1)) :ruleset plan)
(rule ((P li "LinEpi") (PackedEpi li) (LinMN li mn) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons l (Cons o (Nil))))) (P j "SwiGlu")
       (= po (Pos o)) (< po li) (= (Numel o) mn) (= vo (Vi o)))
      ((Epi li j 3 vo)) :ruleset plan)
(rule ((P li "LinEpi") (PackedEpi li) (LinMN li mn) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons o (Cons l (Nil))))) (P j "SwiGlu")
       (= po (Pos o)) (< po li) (= (Numel o) mn) (= vo (Vi o)))
      ((Epi li j 4 vo)) :ruleset plan)
(rule ((P li "LinEpi") (PackedEpi li) (LinMN li mn) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons l (Cons o (Nil))))) (P j "GeGlu")
       (= po (Pos o)) (< po li) (= (Numel o) mn) (= vo (Vi o)))
      ((Epi li j 5 vo)) :ruleset plan)
(rule ((P li "LinEpi") (PackedEpi li) (LinMN li mn) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons o (Cons l (Nil))))) (P j "GeGlu")
       (= po (Pos o)) (< po li) (= (Numel o) mn) (= vo (Vi o)))
      ((Epi li j 6 vo)) :ruleset plan)
(rule ((P li "LinEpi") (LinMN li mn) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons l (Cons o (Nil))))) (P j "Add")
       (= po (Pos o)) (< po li) (= (Numel o) mn) (= vo (Vi o)))
      ((Epi li j 7 vo)) :ruleset plan)
(rule ((P li "LinEpi") (LinMN li mn) (= l (Node li la)) (= (Uses l) 1) (NotOut l)
       (= c (Node j (Cons o (Cons l (Nil))))) (P j "Add")
       (= po (Pos o)) (< po li) (= (Numel o) mn) (= vo (Vi o)))
      ((Epi li j 7 vo)) :ruleset plan)

; RmsNorm `ni` (gamma width d, x of xl elements): `+ r`, then `* s` (scalar f32), or a RoPE.
(rule ((NormP ni d xl) (= n (Node ni na)) (= (Uses n) 1) (NotOut n)
       (= c (Node j (Cons n (Cons o (Nil))))) (P j "Add")
       (= po (Pos o)) (< po ni) (= (Numel o) xl) (= vo (Vi o)))
      ((NormRes ni j vo)) :ruleset plan)
(rule ((NormP ni d xl) (= n (Node ni na)) (= (Uses n) 1) (NotOut n)
       (= c (Node j (Cons o (Cons n (Nil))))) (P j "Add")
       (= po (Pos o)) (< po ni) (= (Numel o) xl) (= vo (Vi o)))
      ((NormRes ni j vo)) :ruleset plan)
(rule ((NormP ni d xl) (= n (Node ni na)) (= (Uses n) 1) (NotOut n)
       (= c (Node k (Cons n (Cons o (Nil))))) (P k "Mul")
       (= po (Pos o)) (< po ni) (= (Numel o) 1) (F32 o) (= vo (Vi o)))
      ((NormScale ni k vo)) :ruleset plan)
(rule ((NormP ni d xl) (= n (Node ni na)) (= (Uses n) 1) (NotOut n)
       (= c (Node k (Cons o (Cons n (Nil))))) (P k "Mul")
       (= po (Pos o)) (< po ni) (= (Numel o) 1) (F32 o) (= vo (Vi o)))
      ((NormScale ni k vo)) :ruleset plan)
(rule ((NormRes ni j vr) (= n (Node j na)) (= (Uses n) 1) (NotOut n)
       (= c (Node k (Cons n (Cons o (Nil))))) (P k "Mul")
       (= po (Pos o)) (< po ni) (= (Numel o) 1) (F32 o) (= vo (Vi o)))
      ((NormScale ni k vo)) :ruleset plan)
(rule ((NormRes ni j vr) (= n (Node j na)) (= (Uses n) 1) (NotOut n)
       (= c (Node k (Cons o (Cons n (Nil))))) (P k "Mul")
       (= po (Pos o)) (< po ni) (= (Numel o) 1) (F32 o) (= vo (Vi o)))
      ((NormScale ni k vo)) :ruleset plan)
(rule ((NormP ni d xl) (= n (Node ni na)) (= (Uses n) 1) (NotOut n)
       (= c (Node j (Cons n (Cons pos (Cons inv r))))) (RopeP j d rot var ms) (<= rot d)
       (= pp (Pos pos)) (< pp ni) (= pv (Pos inv)) (< pv ni)
       (= ns (Numel pos)) (> ns 0) (= 0 (% (/ xl d) ns)))
      ((NormRope ni j)) :ruleset plan)

; Cached attention `ai` writing its KV state: q and k each the single-reader RoPE of a non-norm.
(rule ((AttnP ai h kvh hd)
       (= a (Node ai (Cons q (Cons k (Cons v (Cons ks (Cons vs (Cons pos (Nil)))))))))
       (= (Uses q) 1) (= (Uses k) 1) (NotOut q) (NotOut k)
       (= q (Node qi (Cons qr (Cons pos (Cons inv qx)))))
       (= k (Node ki (Cons kr (Cons pos (Cons inv kx)))))
       (RopeP qi hd rot var ms) (RopeP ki hd rot var ms) (> rot 0) (<= rot hd)
       (NotNorm qr) (NotNorm kr)
       (= nq (Numel qr)) (= nq (* h hd)) (= nk (Numel kr)) (= nk (* kvh hd))
       (= (Numel pos) 1) (= ni (Numel inv)) (>= ni (/ rot 2)))
      ((AttnRope ai qi ki)) :ruleset plan)

(run-schedule (saturate (run plan)))
"#;

/// What an RmsNorm absorbs: the residual `Add` node and its other operand, the scalar `Mul` node
/// and its scale, the RoPE node.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct NormFusion {
    pub res: Option<(usize, ValueId)>,
    pub scale: Option<(usize, ValueId)>,
    pub rope: Option<usize>,
}

/// The fusions `plan` lowers, keyed by the absorbing node.
#[derive(Debug, Default)]
pub(super) struct Decisions {
    /// Linear → (absorbed reader, epilogue, its second operand).
    pub epi: HashMap<usize, (usize, Epi, Option<ValueId>)>,
    pub norm: HashMap<usize, NormFusion>,
    /// CachedAttention → its q and k RoPE nodes.
    pub attn_rope: HashMap<usize, (usize, usize)>,
}

impl Decisions {
    /// Nodes lowered inside another node's step.
    pub fn absorbed(&self) -> impl Iterator<Item = usize> + '_ {
        let epi = self.epi.values().map(|e| e.0);
        let norm = self.norm.values().flat_map(|n| {
            [n.res.map(|r| r.0), n.scale.map(|s| s.0), n.rope]
                .into_iter()
                .flatten()
        });
        let attn = self.attn_rope.values().flat_map(|&(q, k)| [q, k]);
        epi.chain(norm).chain(attn)
    }
}

pub(super) fn err(e: egglog::Error) -> String {
    format!("planner e-graph: {e}")
}

/// The rows of the all-`i64` relation `name`.
pub(super) fn relation(eg: &EGraph, name: &str) -> Result<Vec<Vec<i64>>, String> {
    let mut out = Vec::new();
    eg.constructor_enodes(name, |en| {
        out.push(
            en.children
                .iter()
                .map(|&v| eg.value_to_base::<i64>(v))
                .collect(),
        )
    })
    .map_err(err)?;
    Ok(out)
}

/// Saturates the planner rules over `g` (see the module doc).
pub(super) fn decide(g: &Graph) -> Result<Decisions, String> {
    let mut uses = vec![0i64; g.values.len()];
    for n in &g.nodes {
        for v in &n.inputs {
            uses[v.0 as usize] += 1;
        }
    }
    let epilogues = fusion_policy().allows(FusionClass::DotEpilogue);
    let reduce_input = fusion_policy().allows(FusionClass::ReduceInput);
    let produced_by_norm = |v: ValueId| {
        matches!(g.value(v).kind,
            ValueKind::Node(n) if matches!(g.nodes[n.0 as usize].op, Op::RmsNorm { .. }))
    };
    let mut eg = EGraph::default();
    let (schema, run) = PROGRAM.split_at(PROGRAM.find("(run-schedule").expect("schedule"));
    eg.parse_and_run_program(None, schema).map_err(err)?;
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        let mut seen = vec![false; g.values.len()];
        let mut class =
            |fs: &mut egglog::FullState<'_, '_>, v: ValueId| -> Result<_, egglog::Error> {
                let c = fs.add("V", v.0 as i64)?;
                if !std::mem::replace(&mut seen[v.0 as usize], true) {
                    let pos = match g.value(v).kind {
                        ValueKind::Node(n) => n.0 as i64,
                        _ => -1,
                    };
                    fs.set("Pos", c, pos)?;
                    fs.set("Vi", c, v.0 as i64)?;
                    if let Ok(n) = numel(g, v) {
                        fs.set("Numel", c, n as i64)?;
                    }
                    if g.value(v).dtype == DType::F32 {
                        fs.add("F32", c)?;
                    }
                    if !produced_by_norm(v) {
                        fs.add("NotNorm", c)?;
                    }
                    if matches!(g.value(v).kind, ValueKind::Node(_)) && !g.outputs.contains(&v) {
                        fs.add("NotOut", c)?;
                    }
                }
                Ok(c)
            };
        for (i, n) in g.nodes.iter().enumerate() {
            let id = i as i64;
            let mut args = Vec::with_capacity(n.inputs.len());
            for &v in &n.inputs {
                args.push(class(&mut fs, v)?);
            }
            let mut list = fs.add("Nil", egglog::api::RawValues(Vec::new()))?;
            for &c in args.iter().rev() {
                list = fs.add("Cons", (c, list))?;
            }
            let Some(&out) = n.outputs.first() else {
                continue;
            };
            let o = class(&mut fs, out)?;
            let node = fs.add("Node", (id, list))?;
            fs.union(o, node)?;
            fs.set("Uses", o, uses[out.0 as usize])?;
            let tag = match &n.op {
                Op::Gelu { tanh: false } => Some("GeluErf"),
                Op::Silu => Some("Silu"),
                Op::SwiGlu => Some("SwiGlu"),
                Op::GeGlu => Some("GeGlu"),
                Op::Add => Some("Add"),
                Op::Mul => Some("Mul"),
                _ => None,
            };
            if let Some(t) = tag {
                fs.add("P", (id, t))?;
            }
            match &n.op {
                Op::Linear if epilogues => {
                    let (Some(&x), Some(&w)) = (n.inputs.first(), n.inputs.get(1)) else {
                        continue;
                    };
                    let (&[rows, k], Ok(xl)) = (&g.value(w).shape[..], numel(g, x)) else {
                        continue;
                    };
                    let format = linear_format(g, w);
                    if k == 0 || format == Some(WeightFormat::JesseLUT) {
                        continue;
                    }
                    let m = xl / k;
                    fs.add("P", (id, "LinEpi"))?;
                    fs.add("LinMN", (id, (m * rows) as i64))?;
                    // The batched GEMM kernels (m >= 16) only carry Gelu/Residual epilogues.
                    if format.is_some_and(|f| !(m >= 16 && f.batched_gemm())) {
                        fs.add("PackedEpi", id)?;
                    }
                }
                Op::RmsNorm { .. } if reduce_input && n.inputs.len() >= 2 => {
                    let d = g.value(n.inputs[1]).shape.last().copied().unwrap_or(0);
                    if let Ok(xl) = numel(g, n.inputs[0]) {
                        if d > 0 && xl % d == 0 {
                            fs.add("NormP", (id, d as i64, xl as i64))?;
                        }
                    }
                }
                Op::Rope {
                    head_dim,
                    rotary_dim,
                    variant,
                    mscale,
                } if rotary_dim % 2 == 0 => {
                    let var = match variant {
                        RopeVariant::Interleaved => 2,
                        RopeVariant::HalfSplit => 1,
                        RopeVariant::MRope { .. } => continue,
                    };
                    let p = (id, *head_dim as i64, *rotary_dim as i64, var as i64);
                    fs.add("RopeP", (p.0, p.1, p.2, p.3, mscale.to_bits() as i64))?;
                }
                Op::CachedAttention {
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    write: true,
                    ..
                } if reduce_input && head_dim == v_head_dim && *head_dim <= ATTN_MAX_HEAD_DIM => {
                    let p = (*heads as i64, *kv_heads as i64, *head_dim as i64);
                    fs.add("AttnP", (id, p.0, p.1, p.2))?;
                }
                _ => {}
            }
        }
        Ok(())
    })
    .map_err(err)?;
    eg.parse_and_run_program(None, run).map_err(err)?;

    let rows = |eg: &EGraph, name: &str| relation(eg, name);
    let val = |v: i64| (v >= 0).then(|| ValueId(v as u32));
    let mut d = Decisions::default();
    for r in rows(&eg, "Epi")? {
        let epi = match r[2] {
            1 => Epi::Gelu,
            2 => Epi::Silu,
            3 => Epi::SwiGlu,
            4 => Epi::UpSwiGlu,
            5 => Epi::GeGlu,
            6 => Epi::UpGeGlu,
            _ => Epi::Residual,
        };
        d.epi.insert(r[0] as usize, (r[1] as usize, epi, val(r[3])));
    }
    for r in rows(&eg, "NormRes")? {
        d.norm.entry(r[0] as usize).or_default().res = Some((r[1] as usize, ValueId(r[2] as u32)));
    }
    for r in rows(&eg, "NormScale")? {
        d.norm.entry(r[0] as usize).or_default().scale =
            Some((r[1] as usize, ValueId(r[2] as u32)));
    }
    for r in rows(&eg, "NormRope")? {
        d.norm.entry(r[0] as usize).or_default().rope = Some(r[1] as usize);
    }
    for r in rows(&eg, "AttnRope")? {
        d.attn_rope
            .insert(r[0] as usize, (r[1] as usize, r[2] as usize));
    }
    Ok(d)
}

/// Horizontal batching over the step list, as [`batches`] extracts it.
#[derive(Debug, Default)]
pub(super) struct Batches {
    /// First member → the members (ascending, 2..=4) of one batched dispatch. Steps between members
    /// are independent GEMVs that run ahead of the batch.
    pub runs: HashMap<usize, Vec<usize>>,
    /// An RmsNorm with a residual or scale whose next step, a plain RmsNorm of its output, chains in.
    pub chain: HashSet<usize>,
    /// A one-row RmsNorm with a residual or scale that the next step, a MegakernelMlp of its output,
    /// absorbs.
    pub absorb: HashSet<usize>,
}

/// Step-order facts: `BKind p k` (1: RmsNorm without residual/scale/chain; 1000+: a Rope geometry),
/// `RopeKey p rk` (the RoPE fused into that norm, 0: none), `Indep a b` (step `b` reads no output of
/// `a`, `b - a < 8`). `R{L}` is a run of `L` members from `s` whose next candidate is at `c` and
/// whose RoPE is `rk`; it grows by the next step when that is of the run's kind, independent of
/// every member and RoPE-compatible, or — for norm runs — by the step after an independent GEMV.
const BATCH_PROGRAM: &str = r#"
(relation BKind (i64 i64))
(relation RopeKey (i64 i64))
(relation Solo (i64))
(relation Gemv (i64))
(relation Indep (i64 i64))
(relation NormRS (i64 i64 i64 i64))
(relation NormPlain0 (i64 i64 i64 i64))
(relation NormRS1 (i64 i64 i64))
(relation MlpIn (i64 i64 i64))
(relation R1 (i64 i64 i64))
(relation R2 (i64 i64 i64 i64))
(relation R3 (i64 i64 i64 i64 i64))
(relation R4 (i64 i64 i64 i64 i64 i64))
(relation Chain (i64))
(relation Absorb (i64))
(ruleset batch)
(rule ((BKind s k) (RopeKey s rk)) ((R1 s (+ s 1) rk)) :ruleset batch)
(rule ((NormRS s rows d o) (= n (+ s 1)) (NormPlain0 n rows d o)) ((Chain s)) :ruleset batch)
(rule ((NormRS1 s d o) (= n (+ s 1)) (MlpIn n o d)) ((Absorb s)) :ruleset batch)
"#;

/// The run-growing rules of [`BATCH_PROGRAM`] for runs of 1..=3 members.
fn batch_rules() -> String {
    let mut p = String::new();
    for l in 1..=3 {
        let ms: Vec<String> = (2..=l).map(|i| format!("m{i}")).collect();
        let members: Vec<&str> = std::iter::once("s")
            .chain(ms.iter().map(String::as_str))
            .collect();
        let indep = |at: &str| -> String {
            members
                .iter()
                .map(|m| format!("(Indep {m} {at}) "))
                .collect()
        };
        let state = format!("(R{l} s {} c rk)", ms.join(" "));
        let grow = [
            ("(BKind s k)".to_string(), "c"),
            (
                format!("(BKind s k) (= k 1) (Gemv c) {}(= at (+ c 1))", indep("c")),
                "at",
            ),
        ];
        for (pre, at) in &grow {
            for (compat, rk) in [
                (format!("(RopeKey {at} 0)"), "rk"),
                (format!("(RopeKey {at} nk) (> nk 0) (= rk 0)"), "nk"),
                (format!("(RopeKey {at} rk) (> rk 0)"), "rk"),
            ] {
                let _ = writeln!(
                    p,
                    "(rule ({state} {pre} (BKind {at} k) (Solo {at}) {}{compat}) \
                     ((R{} s {} {at} (+ {at} 1) {rk})) :ruleset batch)",
                    indep(at),
                    l + 1,
                    ms.join(" "),
                );
            }
        }
    }
    p.push_str("(run-schedule (saturate (run batch)))\n");
    p
}

fn intern<K: std::hash::Hash + Eq>(m: &mut HashMap<K, i64>, k: K) -> i64 {
    let n = m.len() as i64 + 1;
    *m.entry(k).or_insert(n)
}

/// Saturates the batching rules over `steps` and extracts what the greedy in-order walk takes:
/// at each step, the longest run starting there, else its chain, else its absorption.
pub(super) fn batches(
    steps: &[Planned],
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<Batches, String> {
    let root = |v: ValueId| aliases.get(&v).map_or(v, |a| a.0);
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, BATCH_PROGRAM).map_err(err)?;
    let mut ropes = HashMap::new();
    let mut geoms = HashMap::new();
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        for (p, s) in steps.iter().enumerate() {
            let pi = p as i64;
            let solo = s.extra_outs.is_empty();
            if solo {
                fs.add("Solo", pi)?;
            }
            let kind = match &s.op {
                Step::RmsNorm {
                    x,
                    res: None,
                    scale: None,
                    chain: None,
                    rope,
                    rows,
                    d,
                    ..
                } => {
                    let rk = rope.map_or(0, |r| {
                        let key = (r.pos, r.inv_freq, r.n_seq, r.rotary_dim, r.pair_stride);
                        intern(&mut ropes, (key, r.mscale.to_bits()))
                    });
                    fs.add("RopeKey", (pi, rk))?;
                    if rope.is_none() && solo {
                        let at = (pi, *rows as i64, *d as i64, root(*x).0 as i64);
                        fs.add("NormPlain0", at)?;
                    }
                    Some(1)
                }
                Step::Rope {
                    pos,
                    inv_freq,
                    n_seq,
                    head_dim,
                    rotary_dim,
                    pair_stride,
                    mscale,
                    ..
                } => {
                    fs.add("RopeKey", (pi, 0i64))?;
                    let key = (
                        *pos,
                        *inv_freq,
                        *n_seq,
                        *head_dim,
                        *rotary_dim,
                        *pair_stride,
                    );
                    Some(1000 + intern(&mut geoms, (key, mscale.to_bits())))
                }
                Step::RmsNorm {
                    chain: None,
                    rope: None,
                    rows,
                    d,
                    ..
                } => {
                    let (rows, d) = (*rows as i64, *d as i64);
                    fs.add("NormRS", (pi, rows, d, root(s.out).0 as i64))?;
                    if rows == 1 && solo && !aliases.contains_key(&s.out) {
                        fs.add("NormRS1", (pi, d, s.out.0 as i64))?;
                    }
                    None
                }
                Step::GemvPacked { .. } if solo => {
                    fs.add("Gemv", pi)?;
                    None
                }
                Step::MegakernelMlp {
                    x,
                    pre_res: None,
                    hidden,
                    ..
                } => {
                    fs.add("MlpIn", (pi, x.0 as i64, *hidden as i64))?;
                    None
                }
                _ => None,
            };
            if let Some(k) = kind {
                fs.add("BKind", (pi, k))?;
                let out = root(s.out);
                for (b, t) in steps.iter().enumerate().take(p + 8).skip(p + 1) {
                    if !t.reads.iter().any(|&r| root(r) == out) {
                        fs.add("Indep", (pi, b as i64))?;
                    }
                }
            }
        }
        Ok(())
    })
    .map_err(err)?;
    eg.parse_and_run_program(None, &batch_rules())
        .map_err(err)?;

    let mut b = Batches::default();
    for l in 2..=4usize {
        for r in relation(&eg, &format!("R{l}"))? {
            let members: Vec<usize> = r[..l].iter().map(|&m| m as usize).collect();
            // Runs grow one member at a time: the longest one from `s` is the extracted one.
            b.runs.insert(members[0], members);
        }
    }
    b.chain = relation(&eg, "Chain")?
        .iter()
        .map(|r| r[0] as usize)
        .collect();
    b.absorb = relation(&eg, "Absorb")?
        .iter()
        .map(|r| r[0] as usize)
        .collect();
    Ok(b)
}

/// One MoE subgraph fusion [`moe`] extracts, at its first step.
#[derive(Debug, Clone, Copy)]
pub(super) enum MoeFusion {
    /// TopK + GatherCols(logits) + Softmax (3 steps): the softmax output becomes TopK's weights.
    TopKSoftmax { weights: ValueId },
    /// ... + GatherCols(scale) + Mul (5 steps): scaled weights.
    TopKSoftmaxScaled { scale: ValueId, weights: ValueId },
    /// Copy(lower half) + Copy(upper half) + GeGlu/SwiGlu (3 steps) of one `[rows, 2 * half]` value.
    SplitGlu {
        x: ValueId,
        rows: usize,
        half_cols: usize,
        kind: u32,
    },
    /// Mul([slots, h] by [slots, 1]) + Reduce(Sum) over `k` (2 steps).
    WeightedSum {
        down: ValueId,
        weights: ValueId,
        n: usize,
        k: usize,
        h: usize,
    },
}

impl MoeFusion {
    /// Steps the fusion replaces, starting at its position.
    pub fn len(&self) -> usize {
        match self {
            MoeFusion::TopKSoftmax { .. } | MoeFusion::SplitGlu { .. } => 3,
            MoeFusion::TopKSoftmaxScaled { .. } => 5,
            MoeFusion::WeightedSum { .. } => 2,
        }
    }
}

/// Step-window facts of [`moe`]: `Single v` (one reader across steps and graph outputs, by alias
/// root), `Solo p` (no extra outputs), and per-kind step fields.
const MOE_PROGRAM: &str = r#"
(relation Single (i64))
(relation Solo (i64))
(relation TopK (i64 i64 i64 i64 i64 i64))
(relation Gat (i64 i64 i64 i64 i64 i64 i64 i64))
(relation Smax (i64 i64 i64 i64 i64))
(relation Bin (i64 i64 i64 i64 i64 i64 i64))
(relation Copy1 (i64 i64))
(relation Glu (i64 i64 i64 i64 i64 i64))
(relation Half (i64 i64 i64 i64 i64))
(relation RedSum (i64 i64 i64 i64 i64 i64 i64 i64))
(relation Shape2 (i64 i64 i64))
(relation MoeTopK (i64 i64))
(relation MoeScaled (i64 i64 i64))
(relation SplitG (i64 i64 i64 i64 i64))
(relation WSum (i64 i64 i64 i64 i64 i64 i64))
(ruleset moe)
; TopK(logits) -> ids; GatherCols(logits, ids) single-use; Softmax over k.
(rule ((TopK p l rows cols k ids) (= p1 (+ p 1)) (Gat p1 l ids tot cols k 0 go) (= tot (* rows k))
       (Single go) (= p2 (+ p 2)) (Smax p2 go rows k so))
      ((MoeTopK p so)) :ruleset moe)
(rule ((TopK p l 1 cols k ids) (= p1 (+ p 1)) (Gat p1 l ids k cols k 1 go)
       (Single go) (= p2 (+ p 2)) (Smax p2 go 1 k so))
      ((MoeTopK p so)) :ruleset moe)
; ... * GatherCols(scale table, ids) (shared), both single-use.
(rule ((MoeTopK p so) (TopK p l rows cols k ids) (Single so) (= p3 (+ p 3))
       (Gat p3 sct ids tot cols k 1 sco) (= tot (* rows k)) (Single sco)
       (= p4 (+ p 4)) (Bin p4 so sco tot 2 1 wo))
      ((MoeScaled p sct wo)) :ruleset moe)
(rule ((MoeTopK p so) (TopK p l rows cols k ids) (Single so) (= p3 (+ p 3))
       (Gat p3 sct ids tot cols k 1 sco) (= tot (* rows k)) (Single sco)
       (= p4 (+ p 4)) (Bin p4 sco so tot 2 1 wo))
      ((MoeScaled p sct wo)) :ruleset moe)
; Copy, Copy, Glu(gate, up): gate the lower and up the upper last-axis half of one value.
(rule ((Copy1 p o0) (Solo p) (= p1 (+ p 1)) (Copy1 p1 o1) (Solo p1) (= p2 (+ p 2))
       (Glu p2 o0 o1 n kind go) (Single o0) (Single o1)
       (Half o0 src r hc 0) (Half o1 src r hc 1) (= n (* r hc)))
      ((SplitG p src r hc kind)) :ruleset moe)
(rule ((Copy1 p o0) (Solo p) (= p1 (+ p 1)) (Copy1 p1 o1) (Solo p1) (= p2 (+ p 2))
       (Glu p2 o1 o0 n kind go) (Single o0) (Single o1)
       (Half o1 src r hc 0) (Half o0 src r hc 1) (= n (* r hc)))
      ((SplitG p src r hc kind)) :ruleset moe)
; Mul(down [slots, h], w [slots, 1]) single-use, then its Sum over k: WSum p down w n k h.
(rule ((Bin p a b mn 2 rk mo) (Solo p) (Single mo) (= p1 (+ p 1)) (RedSum p1 mo 0 nout ro n k h)
       (Solo p1) (= slots (* n k)) (Shape2 mo slots h) (= mn (* slots h)) (= nout (* n h))
       (Shape2 a slots h) (Shape2 b slots 1))
      ((WSum p 0 a b n k h)) :ruleset moe)
(rule ((Bin p a b mn 2 rk mo) (Solo p) (Single mo) (= p1 (+ p 1)) (RedSum p1 mo 0 nout ro n k h)
       (Solo p1) (= slots (* n k)) (Shape2 mo slots h) (= mn (* slots h)) (= nout (* n h))
       (Shape2 b slots h) (Shape2 a slots 1))
      ((WSum p 1 b a n k h)) :ruleset moe)
(run-schedule (saturate (run moe)))
"#;

/// The last-axis half `v` copies, when `v` is `Slice(src)` of exactly the lower or upper half of an
/// even last axis: `(src, rows, half, upper)`.
fn slice_half(g: &Graph, v: ValueId) -> Option<(ValueId, usize, usize, bool)> {
    let ValueKind::Node(nid) = g.value(v).kind else {
        return None;
    };
    let node = &g.nodes[nid.0 as usize];
    let Op::Slice { axis, start, end } = node.op else {
        return None;
    };
    let src = *node.inputs.first()?;
    let sh = &g.value(src).shape;
    let rank = sh.len();
    let ax = if axis < 0 { axis + rank as isize } else { axis };
    if rank < 2 || ax != rank as isize - 1 {
        return None;
    }
    let cols = sh[rank - 1];
    if cols == 0 || cols % 2 != 0 {
        return None;
    }
    let half = cols / 2;
    let rows: usize = sh[..rank - 1].iter().product();
    let dim = cols as i64;
    let clamp = |x: i64| {
        if x < 0 {
            (dim + x).max(0) as usize
        } else {
            x.min(dim) as usize
        }
    };
    match (clamp(start), clamp(end)) {
        (0, e) if e == half => Some((src, rows, half, false)),
        (s, e) if s == half && e == cols => Some((src, rows, half, true)),
        _ => None,
    }
}

/// Saturates the MoE window rules over `steps` and extracts, in step order, the fusion each
/// window start takes (scaled TopK before plain TopK; for a weighted sum, `down` is the first
/// operand when both operand orders fit).
pub(super) fn moe(
    g: &Graph,
    steps: &[Planned],
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<HashMap<usize, MoeFusion>, String> {
    use super::ext::{elementwise, movement, norm, ExtStep};
    let root = |v: ValueId| aliases.get(&v).map_or(v, |a| a.0);
    let mut use_count: HashMap<ValueId, usize> = HashMap::new();
    for v in steps.iter().flat_map(|s| &s.reads).chain(&g.outputs) {
        *use_count.entry(root(*v)).or_default() += 1;
    }
    let vi = |v: ValueId| v.0 as i64;
    let mut eg = EGraph::default();
    let (schema, run) = MOE_PROGRAM.split_at(MOE_PROGRAM.find("(run-schedule").expect("schedule"));
    eg.parse_and_run_program(None, schema).map_err(err)?;
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        for (v, _) in use_count.iter().filter(|(_, &n)| n == 1) {
            fs.add("Single", vi(*v))?;
        }
        let shape2 = |fs: &mut egglog::FullState<'_, '_>, v: ValueId| match g.value(v).shape[..] {
            [a, b] => fs.add("Shape2", (vi(v), a as i64, b as i64)).map(|_| ()),
            _ => Ok(()),
        };
        for (p, s) in steps.iter().enumerate() {
            let pi = p as i64;
            if s.extra_outs.is_empty() {
                fs.add("Solo", pi)?;
            }
            let o = vi(s.out);
            match &s.op {
                Step::Ext(ExtStep::Experts(super::ext::moe::Step::TopK {
                    x,
                    rows,
                    cols,
                    k,
                    weights_out: None,
                    scale: None,
                })) if *k <= 32 => {
                    let f = (pi, vi(*x), *rows as i64, *cols as i64, *k as i64, o);
                    fs.add("TopK", f)?;
                }
                Step::Ext(ExtStep::Experts(super::ext::moe::Step::GatherCols {
                    x,
                    idx,
                    total,
                    cols,
                    k,
                    shared,
                })) => {
                    let f = (pi, vi(*x), vi(*idx), *total as i64, *cols as i64, *k as i64);
                    fs.add("Gat", (f.0, f.1, f.2, f.3, f.4, f.5, *shared as i64, o))?;
                }
                Step::Ext(ExtStep::Norm(norm::Step::Softmax {
                    x,
                    outer,
                    n,
                    inner: 1,
                })) => {
                    fs.add("Smax", (pi, vi(*x), *outer as i64, *n as i64, o))?;
                }
                Step::Ext(ExtStep::Elementwise(elementwise::Step::Binary {
                    a,
                    b,
                    n,
                    kind,
                    rank,
                    ..
                })) => {
                    let f = (pi, vi(*a), vi(*b), *n as i64, *kind as i64, *rank as i64, o);
                    fs.add("Bin", f)?;
                    for v in [*a, *b, s.out] {
                        shape2(&mut fs, v)?;
                    }
                }
                Step::Ext(ExtStep::Movement(movement::Step::Copy { fill: None, parts }))
                    if parts.len() == 1 =>
                {
                    fs.add("Copy1", (pi, o))?;
                }
                Step::GeGlu { gate, up, n } | Step::SwiGlu { gate, up, n } => {
                    let kind = if matches!(s.op, Step::GeGlu { .. }) {
                        1
                    } else {
                        2
                    };
                    fs.add("Glu", (pi, vi(*gate), vi(*up), *n as i64, kind, o))?;
                    for v in [*gate, *up] {
                        if let Some((src, rows, half, upper)) = slice_half(g, v) {
                            let f = (vi(v), vi(src), rows as i64, half as i64, upper as i64);
                            fs.add("Half", f)?;
                        }
                    }
                }
                Step::Ext(ExtStep::Norm(norm::Step::Reduce { x, n_out, .. })) => {
                    let ValueKind::Node(nid) = g.value(s.out).kind else {
                        continue;
                    };
                    let Op::Reduce {
                        kind: crate::graph::ReduceKind::Sum,
                        axes,
                        ..
                    } = &g.nodes[nid.0 as usize].op
                    else {
                        continue;
                    };
                    let rsh = &g.value(*x).shape;
                    let ax = match axes[..] {
                        [a] if a < 0 => a + rsh.len() as isize,
                        [a] => a,
                        _ => continue,
                    };
                    if ax < 0 || ax as usize >= rsh.len() {
                        continue;
                    }
                    let ax = ax as usize;
                    let n: usize = rsh[..ax].iter().product();
                    let h: usize = rsh[ax + 1..].iter().product();
                    let (rx, off) = aliases.get(x).copied().unwrap_or((*x, 0));
                    let f = (pi, vi(rx), off as i64, *n_out as i64, o, n as i64);
                    fs.add(
                        "RedSum",
                        (f.0, f.1, f.2, f.3, f.4, f.5, rsh[ax] as i64, h as i64),
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    })
    .map_err(err)?;
    eg.parse_and_run_program(None, run).map_err(err)?;

    let v = |x: i64| ValueId(x as u32);
    let mut d: HashMap<usize, MoeFusion> = HashMap::new();
    // Lowest priority first: later inserts at the same start win.
    let mut wsum = relation(&eg, "WSum")?;
    wsum.sort_by_key(|r| std::cmp::Reverse(r[1]));
    for r in wsum {
        let (n, k, h) = (r[4] as usize, r[5] as usize, r[6] as usize);
        let f = MoeFusion::WeightedSum {
            down: v(r[2]),
            weights: v(r[3]),
            n,
            k,
            h,
        };
        d.insert(r[0] as usize, f);
    }
    for r in relation(&eg, "SplitG")? {
        let f = MoeFusion::SplitGlu {
            x: v(r[1]),
            rows: r[2] as usize,
            half_cols: r[3] as usize,
            kind: r[4] as u32,
        };
        d.insert(r[0] as usize, f);
    }
    for r in relation(&eg, "MoeTopK")? {
        d.insert(r[0] as usize, MoeFusion::TopKSoftmax { weights: v(r[1]) });
    }
    for r in relation(&eg, "MoeScaled")? {
        let f = MoeFusion::TopKSoftmaxScaled {
            scale: v(r[1]),
            weights: v(r[2]),
        };
        d.insert(r[0] as usize, f);
    }
    Ok(d)
}
