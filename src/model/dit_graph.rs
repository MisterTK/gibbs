//! Graph-IR builder helpers shared by the diffusion/video transformer Metal programs (FLUX, SD3,
//! Qwen-Image, Krea, Z-Image, Sana, LTX). Activations are f32 `[rows, d]`; weights stay in their
//! stored encoding (`Mat`: mapped bf16 or packed GGUF blocks) and bind by name. Sampling-schedule
//! arithmetic and timestep sinusoids stay on the host; every network forward is one program built
//! per (token counts) and cached.

use crate::graph::program::GraphProgram;
use crate::graph::{
    BinaryKind, DType, Graph, Op, PairLayout, RopeVariant, Tensor, UnaryKind, ValueId, Weights,
};
use crate::hardware::{Device, PlacementMode};
use crate::model::weights::Mat;
use std::collections::HashMap;

/// Whether the process runs GPU-only (`main` armed the tripwire for a Metal backend): pipelines
/// run their transformer through a graph program instead of the host forward.
pub fn use_metal() -> bool {
    crate::hardware::cpu_compute::banned()
}

/// A graph and its weights under construction.
pub struct Gb {
    pub g: Graph,
    pub w: Weights,
    consts: HashMap<String, ValueId>,
}

impl Default for Gb {
    fn default() -> Self {
        Self::new()
    }
}

impl Gb {
    pub fn new() -> Self {
        Self {
            g: Graph::new(),
            w: Weights::new(),
            consts: HashMap::new(),
        }
    }

    pub fn shape(&self, v: ValueId) -> Vec<usize> {
        self.g.value(v).shape.clone()
    }

    pub fn input(&mut self, name: &str, shape: &[usize]) -> ValueId {
        self.g.input(name, shape.to_vec(), DType::F32)
    }

    pub fn input_u32(&mut self, name: &str, shape: &[usize]) -> ValueId {
        self.g.input(name, shape.to_vec(), DType::U32)
    }

    /// An f32 weight (norm gains, biases, LoRA factors, frequency tables), bound once per name.
    pub fn konst(&mut self, name: &str, shape: &[usize], data: Vec<f32>) -> ValueId {
        if let Some(&v) = self.consts.get(name) {
            return v;
        }
        assert_eq!(data.len(), shape.iter().product::<usize>(), "const {name}");
        let v = self.g.weight(name, shape.to_vec(), DType::F32);
        self.w.insert(name, Tensor::F32(data));
        self.consts.insert(name.to_string(), v);
        v
    }

    /// `[1]` holding `v`.
    pub fn scalar(&mut self, v: f32) -> ValueId {
        self.konst(&format!("const.{:08x}", v.to_bits()), &[1], vec![v])
    }

    /// A matrix weight `[rows, cols]` in its stored encoding (bf16 or packed GGUF blocks).
    pub fn mat(&mut self, name: &str, m: &Mat) -> ValueId {
        if let Some(&v) = self.consts.get(name) {
            return v;
        }
        let dtype = match m.packed() {
            Some(p) => match p.format {
                crate::hardware::QuantFormat::FP32 => DType::F32,
                crate::hardware::QuantFormat::FP16 => DType::F16,
                crate::hardware::QuantFormat::BF16 => DType::BF16,
                f => DType::Quant(f),
            },
            None => DType::BF16,
        };
        let v = self.g.weight(name, vec![m.rows, m.cols], dtype);
        self.w.insert(name, m.to_graph_tensor());
        self.consts.insert(name.to_string(), v);
        v
    }

    /// `x [m, k] @ w [n, k]^T (+ bias [n])` → `[m, n]`.
    pub fn linear(&mut self, name: &str, x: ValueId, w: ValueId, bias: Option<ValueId>) -> ValueId {
        let (xs, ws) = (self.shape(x), self.shape(w));
        let m = xs.iter().product::<usize>() / ws[1];
        let mut ins = vec![x, w];
        ins.extend(bias);
        self.g.add(name, Op::Linear, &ins, vec![m, ws[0]], DType::F32)
    }

    /// A stored matrix applied to `x`, with an optional f32 bias.
    pub fn lin_mat(&mut self, name: &str, x: ValueId, m: &Mat, bias: Option<&[f32]>) -> ValueId {
        let w = self.mat(&format!("{name}.weight"), m);
        let b = bias.map(|b| self.konst(&format!("{name}.bias"), &[b.len()], b.to_vec()));
        self.linear(name, x, w, b)
    }

    fn bin(&mut self, name: &str, kind: BinaryKind, a: ValueId, b: ValueId) -> ValueId {
        // numpy broadcasting: the larger extent per aligned dim.
        let (sa, sb) = (self.shape(a), self.shape(b));
        let r = sa.len().max(sb.len());
        let out: Vec<usize> = (0..r)
            .map(|i| {
                let da = (i + sa.len()).checked_sub(r).map_or(1, |j| sa[j]);
                let db = (i + sb.len()).checked_sub(r).map_or(1, |j| sb[j]);
                da.max(db)
            })
            .collect();
        self.g
            .add(name, Op::Binary { kind }, &[a, b], out, DType::F32)
    }

    pub fn add(&mut self, name: &str, a: ValueId, b: ValueId) -> ValueId {
        self.bin(name, BinaryKind::Add, a, b)
    }

    pub fn mul(&mut self, name: &str, a: ValueId, b: ValueId) -> ValueId {
        self.bin(name, BinaryKind::Mul, a, b)
    }

    pub fn unary(&mut self, name: &str, kind: UnaryKind, x: ValueId) -> ValueId {
        let s = self.shape(x);
        self.g.add(name, Op::Unary { kind }, &[x], s, DType::F32)
    }

    pub fn scale(&mut self, name: &str, x: ValueId, f: f32) -> ValueId {
        self.unary(name, UnaryKind::Scale(f), x)
    }

    pub fn silu(&mut self, name: &str, x: ValueId) -> ValueId {
        let s = self.shape(x);
        self.g.add(name, Op::Unary { kind: UnaryKind::Silu }, &[x], s, DType::F32)
    }

    pub fn gelu_tanh(&mut self, name: &str, x: ValueId) -> ValueId {
        let s = self.shape(x);
        self.g
            .add(name, Op::Unary { kind: UnaryKind::GeluTanh }, &[x], s, DType::F32)
    }

    /// `silu(gate) * up`.
    pub fn swiglu(&mut self, name: &str, gate: ValueId, up: ValueId) -> ValueId {
        let s = self.shape(gate);
        self.g.add(name, Op::SwiGlu, &[gate, up], s, DType::F32)
    }

    pub fn reshape(&mut self, name: &str, x: ValueId, shape: &[usize]) -> ValueId {
        self.g.add(
            name,
            Op::Reshape {
                shape: Some(shape.to_vec()),
            },
            &[x],
            shape.to_vec(),
            DType::F32,
        )
    }

    pub fn transpose(&mut self, name: &str, x: ValueId, perm: &[usize]) -> ValueId {
        let s = self.shape(x);
        let out = perm.iter().map(|&p| s[p]).collect();
        self.g.add(
            name,
            Op::Transpose {
                perm: perm.to_vec(),
            },
            &[x],
            out,
            DType::F32,
        )
    }

    /// `x[.., start..end, ..]` along `axis`.
    pub fn slice(&mut self, name: &str, x: ValueId, axis: usize, start: usize, end: usize) -> ValueId {
        let mut s = self.shape(x);
        s[axis] = end - start;
        self.g.add(
            name,
            Op::Slice {
                axis: axis as isize,
                start: start as i64,
                end: end as i64,
            },
            &[x],
            s,
            DType::F32,
        )
    }

    pub fn concat(&mut self, name: &str, xs: &[ValueId], axis: usize) -> ValueId {
        let mut s = self.shape(xs[0]);
        s[axis] = xs.iter().map(|&x| self.shape(x)[axis]).sum();
        self.g.add(
            name,
            Op::Concat { axis: axis as isize },
            xs,
            s,
            DType::F32,
        )
    }

    /// Zero-pads the last dim by `(before, after)`.
    pub fn pad_last(&mut self, name: &str, x: ValueId, before: usize, after: usize) -> ValueId {
        let mut s = self.shape(x);
        *s.last_mut().expect("rank >= 1") += before + after;
        self.g.add(
            name,
            Op::Pad {
                padding: vec![before, after],
                value: 0.0,
            },
            &[x],
            s,
            DType::F32,
        )
    }

    /// Affine-free LayerNorm over the last dim (`x [n, d]`).
    pub fn layer_norm(&mut self, name: &str, x: ValueId, eps: f32) -> ValueId {
        let s = self.shape(x);
        let d = *s.last().expect("rank >= 1");
        let gamma = self.konst(&format!("ones.{d}"), &[d], vec![1.0; d]);
        self.g
            .add(name, Op::LayerNorm { eps }, &[x, gamma], s, DType::F32)
    }

    pub fn rms_norm(&mut self, name: &str, x: ValueId, gamma: ValueId, eps: f32) -> ValueId {
        let s = self.shape(x);
        self.g
            .add(name, Op::RmsNorm { eps }, &[x, gamma], s, DType::F32)
    }

    /// RMSNorm of every `hd`-wide head of `x [n, heads * hd]` (QK-norm).
    pub fn heads_rms(&mut self, name: &str, x: ValueId, gamma: ValueId, hd: usize, eps: f32) -> ValueId {
        let s = self.shape(x);
        let total: usize = s.iter().product();
        let r = self.reshape(&format!("{name}.split"), x, &[total / hd, hd]);
        let n = self.rms_norm(name, r, gamma, eps);
        self.reshape(&format!("{name}.join"), n, &s)
    }

    /// `LayerNorm(x) * (1 + scale) + shift`, `x [n, d]`, `shift`/`scale` `[1, d]`.
    pub fn ln_mod(&mut self, name: &str, x: ValueId, shift: ValueId, scale: ValueId, eps: f32) -> ValueId {
        let n = self.layer_norm(&format!("{name}.ln"), x, eps);
        self.modulate(name, n, shift, scale)
    }

    /// `x * (1 + scale) + shift`.
    pub fn modulate(&mut self, name: &str, x: ValueId, shift: ValueId, scale: ValueId) -> ValueId {
        let one = self.scalar(1.0);
        let s1 = self.add(&format!("{name}.1p"), scale, one);
        let m = self.mul(&format!("{name}.mul"), x, s1);
        self.add(&format!("{name}.shift"), m, shift)
    }

    /// `h + gate * y`, `gate` `[1, d]`.
    pub fn gated_add(&mut self, name: &str, h: ValueId, y: ValueId, gate: ValueId) -> ValueId {
        let gy = self.mul(&format!("{name}.gate"), y, gate);
        self.add(name, h, gy)
    }

    /// Rotary embedding with per-pair frequencies and up to three position axes: `pos` is U32
    /// `[n * 3]` (t, h, w per token); pair `j` reads the axis its section covers.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &mut self,
        name: &str,
        x: ValueId,
        pos: ValueId,
        inv_freq: ValueId,
        head_dim: usize,
        sections: [u32; 3],
        pairs: PairLayout,
    ) -> ValueId {
        let s = self.shape(x);
        self.g.add(
            name,
            Op::Rope {
                head_dim,
                rotary_dim: head_dim,
                variant: RopeVariant::MRope {
                    sections,
                    interleaved: false,
                    pairs,
                },
                mscale: 1.0,
            },
            &[x, pos, inv_freq],
            s,
            DType::F32,
        )
    }

    /// Bidirectional multi-head attention, `q [nq, heads*hd]`, `k`/`v [nk, heads*hd]`,
    /// scores scaled by `1/sqrt(hd)`.
    pub fn attention(&mut self, name: &str, q: ValueId, k: ValueId, v: ValueId, heads: usize) -> ValueId {
        let s = self.shape(q);
        self.g.add(
            name,
            Op::Attention {
                heads,
                kv_heads: heads,
                causal: false,
                window: None,
                soft_cap: None,
                scale: None,
            },
            &[q, k, v],
            s,
            DType::F32,
        )
    }
}

/// Per-pair inverse frequencies and the section sizes of a multi-axis rotary embedding whose
/// axes (`dims`, each even) rotate adjacent pairs with `theta^(-2i/dim)`; `dims.len() <= 3`.
/// Axes beyond what `MRope` reads positions for are folded by the caller (see [`fold_axes`]).
pub fn axis_freqs(dims: &[usize], theta: f64) -> (Vec<f32>, [u32; 3]) {
    assert!(dims.len() <= 3 && dims.iter().all(|d| d % 2 == 0));
    let mut f = Vec::new();
    let mut sec = [0u32; 3];
    for (a, &dim) in dims.iter().enumerate() {
        sec[a] = (dim / 2) as u32;
        for i in 0..dim / 2 {
            f.push(theta.powf(-((2 * i) as f64) / dim as f64) as f32);
        }
    }
    (f, sec)
}

/// A host-side id table `[tokens][axes]` flattened to the `[tokens * 3]` U32 positions `MRope`
/// reads, taking axes `use_axes` (missing axes read position 0).
pub fn positions(ids: &[Vec<usize>], use_axes: [usize; 3]) -> Vec<u32> {
    ids.iter()
        .flat_map(|id| use_axes.map(|a| id.get(a).copied().unwrap_or(0) as u32))
        .collect()
}

/// A compiled Metal program with its input binding.
pub struct Program(GraphProgram);

impl Program {
    pub fn compile(b: Gb, outputs: &[ValueId]) -> Result<Self, String> {
        let Gb { mut g, w, .. } = b;
        for &o in outputs {
            g.output(o);
        }
        GraphProgram::compile(g, w, PlacementMode::Pin(Device::Metal)).map(Self)
    }

    pub fn run(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Vec<f32>>, String> {
        self.0.run(inputs)?.into_iter().map(Tensor::into_f32).collect()
    }
}

/// Programs keyed by their static shapes (token counts, resolution); one is built per key.
pub struct ProgramCache<K> {
    map: parking_lot::Mutex<HashMap<K, Program>>,
}

impl<K: std::hash::Hash + Eq + Clone> Default for ProgramCache<K> {
    fn default() -> Self {
        Self {
            map: parking_lot::Mutex::new(HashMap::new()),
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone> ProgramCache<K> {
    /// Runs the program for `key`, building it with `build` (graph + output values) on first use.
    /// ponytail: at most 8 resident programs; the whole cache is dropped when a 9th shape arrives.
    pub fn run(
        &self,
        key: K,
        build: impl FnOnce() -> Result<(Gb, Vec<ValueId>), String>,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut map = self.map.lock();
        if !map.contains_key(&key) {
            let (gb, outs) = build()?;
            if map.len() >= 8 {
                map.clear();
            }
            map.insert(key.clone(), Program::compile(gb, &outs)?);
        }
        map.get_mut(&key).expect("just inserted").run(inputs)
    }
}

/// Random-weight fixtures and error metrics for parity tests and the GPU-only matrix.
pub mod fixture {
    use crate::model::weights::{Mat, Weights};

    /// Deterministic values in `[-scale, scale)`.
    pub fn rand(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// A bf16 `[rows, cols]` matrix of small random values.
    pub fn mat(rows: usize, cols: usize, seed: u64) -> Mat {
        let scale = 1.0 / (cols as f32).sqrt();
        let data = rand(rows * cols, seed, scale);
        Weights::from_f32(vec![("w".into(), vec![rows, cols])], |_| Ok(data.clone()))
            .and_then(|w| w.mat("w"))
            .expect("anonymous map")
    }

    /// max |a - b| relative to max(1, max |a|).
    pub fn rel_err(want: &[f32], got: &[f32]) -> f32 {
        assert_eq!(want.len(), got.len());
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        want.iter()
            .zip(got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
            / scale
    }

    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += (x * y) as f64;
            aa += (x * x) as f64;
            bb += (y * y) as f64;
        }
        (ab / (aa.sqrt() * bb.sqrt()).max(1e-30)) as f32
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::fixture::{mat, rand, rel_err};
    use super::*;

    /// Linear -> {Silu, SwiGlu, GeGlu} on bf16 weights at row counts either side of the batched
    /// GEMM threshold (16): the epilogue must never be dropped.
    #[test]
    fn linear_activation_epilogues_match_cpu() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        for m in [1usize, 15, 16, 64] {
            for act in ["silu", "swiglu", "geglu"] {
                let (k, n) = (64usize, 96usize);
                let mut b = Gb::new();
                let x = b.input("x", &[m, k]);
                let w = b.mat("w", &mat(n, k, 3));
                let y = b.linear("lin", x, w, None);
                let out = match act {
                    "silu" => b.silu("act", y),
                    _ => {
                        let w2 = b.mat("w2", &mat(n, k, 4));
                        let up = b.linear("up", x, w2, None);
                        match act {
                            "swiglu" => b.swiglu("act", y, up),
                            _ => b.g.add("act", Op::GeGlu, &[y, up], vec![m, n], DType::F32),
                        }
                    }
                };
                b.g.output(out);
                let inputs = vec![Tensor::F32(rand(m * k, 5, 1.0))];
                let want = {
                    let _cpu = crate::hardware::cpu_compute::allow();
                    crate::graph::cpu::run_stateless(&b.g, &b.w, inputs.clone())
                        .unwrap()
                        .remove(0)
                        .into_f32()
                        .unwrap()
                };
                let got = Program::compile(b, &[])
                    .unwrap()
                    .run(inputs)
                    .unwrap()
                    .remove(0);
                let e = rel_err(&want, &got);
                assert!(e < 1e-4, "{act} m={m}: rel {e}");
            }
        }
    }
}

pub use fixture as testutil;
