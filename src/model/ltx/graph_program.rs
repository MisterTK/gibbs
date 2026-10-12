//! The LTX-Video transformer forward as one Metal graph program per `(tokens, text tokens, clean
//! tokens)`. The host supplies only scalar arithmetic: the timestep sinusoids and the 3-axis RoPE
//! angle tables. The T5 caption projection, patch embed, adaLN-single, attention, FFN and final
//! layer all run in the program; weights stay in their stored encoding.

use super::*;
use crate::graph::{Tensor, ValueId};
use crate::model::dit_graph::{Gb, ProgramCache};

/// `(latent tokens, text tokens, leading clean tokens)`.
pub(super) type Key = (usize, usize, usize);

pub(super) type Cache = ProgramCache<Key>;

/// Contiguous token ranges and the timestep they run at: `clean` image-conditioning tokens at
/// timestep 0, the rest at `t` (empty ranges dropped).
fn spans(n: usize, clean: usize) -> Vec<(usize, usize, bool)> {
    [(0, clean, false), (clean, n, true)]
        .into_iter()
        .filter(|s| s.0 < s.1)
        .collect()
}

/// Applies `f(builder, rows of x, span index)` to each span's rows and re-joins them.
fn per_span(
    b: &mut Gb,
    name: &str,
    x: ValueId,
    sp: &[(usize, usize, bool)],
    mut f: impl FnMut(&mut Gb, ValueId, usize) -> ValueId,
) -> ValueId {
    if sp.len() == 1 {
        return f(b, x, 0);
    }
    let parts: Vec<ValueId> = sp
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let xs = b.slice(&format!("{name}.rows{i}"), x, 0, s.0, s.1);
            f(b, xs, i)
        })
        .collect();
    b.concat(name, &parts, 0)
}

/// Row `i` of a `[spans, w]` table (the table itself when there is one span).
fn row(b: &mut Gb, name: &str, t: ValueId, i: usize, spans: usize) -> ValueId {
    if spans == 1 {
        t
    } else {
        b.slice(name, t, 0, i, i + 1)
    }
}

/// `x * cos + pair_swap(x) * sin_signed` on `[n, d]`: the interleaved rotation of
/// [`Rope::apply`], with `sin_signed = [-sin, +sin]` per pair.
fn rope(b: &mut Gb, name: &str, x: ValueId, cos: ValueId, sin: ValueId) -> ValueId {
    let s = b.shape(x);
    let pairs = b.reshape(&format!("{name}.pairs"), x, &[s[0] * s[1] / 2, 2]);
    let even = b.slice(&format!("{name}.even"), pairs, 1, 0, 1);
    let odd = b.slice(&format!("{name}.odd"), pairs, 1, 1, 2);
    let sw = b.concat(&format!("{name}.swap"), &[odd, even], 1);
    let sw = b.reshape(&format!("{name}.swapped"), sw, &s);
    let a = b.mul(&format!("{name}.cos"), x, cos);
    let c = b.mul(&format!("{name}.sin"), sw, sin);
    b.add(name, a, c)
}

impl LtxTransformer {
    /// The velocity for latent tokens `x [n, 128]` given the *raw* T5 tokens `ctx [nt, 4096]`
    /// (the caption projection runs in the program), model timestep `t` and the first `clean`
    /// tokens held at timestep 0, on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_gpu(
        &self,
        x: &[f32],
        n: usize,
        ctx: &[f32],
        nt: usize,
        t: f32,
        clean: usize,
        rope: &Rope,
    ) -> Result<Vec<f32>, String> {
        let sin: Vec<f32> = rope
            .sin
            .chunks_exact(2)
            .flat_map(|p| [-p[0], p[1]])
            .collect();
        let mut inputs = vec![
            Tensor::F32(x.to_vec()),
            Tensor::F32(ctx.to_vec()),
            Tensor::F32(rope.cos.clone()),
            Tensor::F32(sin),
        ];
        let feats: Vec<f32> = spans(n, clean.min(n))
            .iter()
            .flat_map(|s| crate::model::flux::sinusoid(if s.2 { t as f64 } else { 0.0 }, TIME_DIM))
            .collect();
        inputs.push(Tensor::F32(feats));
        let mut out = self.gpu.run(
            (n, nt, clean.min(n)),
            || self.build(n, nt, clean.min(n)),
            inputs,
        )?;
        Ok(out.remove(0))
    }

    /// Emits the whole forward; inputs in order `x, ctx, rope_cos, rope_sin, t_feat`.
    fn build(&self, n: usize, nt: usize, clean: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let d = self.d;
        let sp = spans(n, clean);
        let ns = sp.len();
        let mut b = Gb::new();
        let x_in = b.input("x", &[n, self.proj_in.in_dim()]);
        let ctx_in = b.input("ctx", &[nt, self.c1.in_dim()]);
        let cos = b.input("rope.cos", &[n, d]);
        let sin = b.input("rope.sin", &[n, d]);
        let tf = b.input("t_feat", &[ns, TIME_DIM]);
        let ones = b.konst(&format!("ones.{d}"), &[d], vec![1.0; d]);

        // PixArtAlphaTextProjection: linear, tanh-GELU, linear.
        let h = self.c1.emit(&mut b, "caption.1", ctx_in);
        let h = b.gelu_tanh("caption.act", h);
        let ctx = self.c2.emit(&mut b, "caption.2", h);

        // AdaLayerNormSingle: `emb [spans, d]`, `temb [spans, 6d]`.
        let h = self.t1.emit(&mut b, "time.1", tf);
        let h = b.silu("time.act", h);
        let emb = self.t2.emit(&mut b, "time.2", h);
        let a = b.silu("time.silu", emb);
        let temb = self.ada.emit(&mut b, "time.ada", a);

        // `rms(x) * (1 + scale) + shift` per span with chunks `(i, j)` of the `[spans, 6d]` table `m`.
        let norm_mod = |b: &mut Gb, p: &str, x: ValueId, m: ValueId, (i, j): (usize, usize), layer: bool| {
            let nrm = if layer {
                b.layer_norm(&format!("{p}.ln"), x, 1e-6)
            } else {
                b.rms_norm(&format!("{p}.rms"), x, ones, 1e-6)
            };
            let sh = b.slice(&format!("{p}.sh"), m, 1, i * d, (i + 1) * d);
            let sc = b.slice(&format!("{p}.sc"), m, 1, j * d, (j + 1) * d);
            per_span(b, p, nrm, &sp, |b, xs, s| {
                let sh = row(b, &format!("{p}.sh{s}"), sh, s, ns);
                let sc = row(b, &format!("{p}.sc{s}"), sc, s, ns);
                b.modulate(&format!("{p}.m{s}"), xs, sh, sc)
            })
        };
        // `x + gate * y` per span, `gate` chunk `i` of `m`.
        let gate = |b: &mut Gb, p: &str, x: ValueId, y: ValueId, m: ValueId, i: usize| {
            let g = b.slice(&format!("{p}.g"), m, 1, i * d, (i + 1) * d);
            let gy = per_span(b, &format!("{p}.gy"), y, &sp, |b, ys, s| {
                let g = row(b, &format!("{p}.g{s}"), g, s, ns);
                b.mul(&format!("{p}.gm{s}"), ys, g)
            });
            b.add(p, x, gy)
        };
        let attn = |b: &mut Gb, p: &str, a: &Attn, h: ValueId, kv: ValueId, rot: bool| {
            let q = a.q.emit(b, &format!("{p}.q"), h);
            let k = a.k.emit(b, &format!("{p}.k"), kv);
            let v = a.v.emit(b, &format!("{p}.v"), kv);
            let (qn, kn) = (
                b.konst(&format!("{p}.qn"), &[d], a.qn.clone()),
                b.konst(&format!("{p}.kn"), &[d], a.kn.clone()),
            );
            let mut q = b.rms_norm(&format!("{p}.qnorm"), q, qn, 1e-5);
            let mut k = b.rms_norm(&format!("{p}.knorm"), k, kn, 1e-5);
            if rot {
                q = rope(b, &format!("{p}.rq"), q, cos, sin);
                k = rope(b, &format!("{p}.rk"), k, cos, sin);
            }
            let o = b.attention(&format!("{p}.attn"), q, k, v, HEADS);
            a.o.emit(b, &format!("{p}.o"), o)
        };

        let mut x = self.proj_in.emit(&mut b, "proj_in", x_in);
        for (i, blk) in self.blocks.iter().enumerate() {
            let p = format!("blk{i}");
            let sst = b.konst(&format!("{p}.sst"), &[1, 6 * d], blk.sst.clone());
            let m = b.add(&format!("{p}.m"), sst, temb);
            let h = norm_mod(&mut b, &format!("{p}.n1"), x, m, (0, 1), false);
            let a = attn(&mut b, &format!("{p}.a1"), &blk.a1, h, h, true);
            x = gate(&mut b, &format!("{p}.r1"), x, a, m, 2);
            // Cross-attention reads the raw residual stream and is added ungated.
            let y = attn(&mut b, &format!("{p}.a2"), &blk.a2, x, ctx, false);
            x = b.add(&format!("{p}.r2"), x, y);
            let h = norm_mod(&mut b, &format!("{p}.n2"), x, m, (3, 4), false);
            let f = blk.ff1.emit(&mut b, &format!("{p}.ff1"), h);
            let f = b.gelu_tanh(&format!("{p}.gelu"), f);
            let f = blk.ff2.emit(&mut b, &format!("{p}.ff2"), f);
            x = gate(&mut b, &format!("{p}.r3"), x, f, m, 5);
        }

        // Output shift/scale: `scale_shift_table[0|1] + emb`, `[spans, d]` each.
        let sst = b.konst("out.sst", &[2, d], self.sst.clone());
        let om = {
            let r0 = b.slice("out.sst0", sst, 0, 0, 1);
            let r1 = b.slice("out.sst1", sst, 0, 1, 2);
            let sh = b.add("out.shift", r0, emb);
            let sc = b.add("out.scale", r1, emb);
            b.concat("out.m", &[sh, sc], 1)
        };
        let h = norm_mod(&mut b, "out", x, om, (0, 1), true);
        let y = self.proj_out.emit(&mut b, "proj_out", h);
        Ok((b, vec![y]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, mat, rand, rel_err};
    use crate::model::flux::Wt;

    struct Seed(u64);
    impl Seed {
        fn next(&mut self) -> u64 {
            self.0 += 1;
            self.0
        }
    }

    fn lin(s: &mut Seed, n: usize, k: usize) -> Lin {
        let bias = rand(n, s.next(), 0.1);
        Lin::new(Wt::Bf16(mat(n, k, s.next())), Some(bias), vec![])
    }

    fn gain(s: &mut Seed, n: usize) -> Vec<f32> {
        rand(n, s.next(), 1.0).iter().map(|v| 1.0 + 0.3 * v).collect()
    }

    const D: usize = 512;
    const TEXT_IN: usize = 96;

    fn tiny(seed: u64) -> LtxTransformer {
        let s = &mut Seed(seed * 1000);
        let mut attn = |s: &mut Seed| Attn {
            q: lin(s, D, D),
            k: lin(s, D, D),
            v: lin(s, D, D),
            o: lin(s, D, D),
            qn: gain(s, D),
            kn: gain(s, D),
        };
        let blocks = (0..2)
            .map(|_| Block {
                a1: attn(s),
                a2: attn(s),
                ff1: lin(s, 4 * D, D),
                ff2: lin(s, D, 4 * D),
                sst: rand(6 * D, s.next(), 0.5),
            })
            .collect();
        LtxTransformer {
            proj_in: lin(s, D, LATENT),
            t1: lin(s, D, TIME_DIM),
            t2: lin(s, D, D),
            ada: lin(s, 6 * D, D),
            c1: lin(s, D, TEXT_IN),
            c2: lin(s, D, D),
            blocks,
            sst: rand(2 * D, s.next(), 0.5),
            proj_out: lin(s, LATENT, D),
            d: D,
            gpu: Default::default(),
        }
    }

    /// Metal program == native CPU forward: plain, a partial image-conditioned split and the
    /// all-clean split, over a 3x3x4 latent grid (36 tokens).
    #[test]
    fn ltx_program_matches_cpu_forward() {
        let tr = tiny(1);
        let (f, h, w, nt) = (3usize, 3usize, 4usize, 24usize);
        let n = f * h * w;
        let x = rand(n * LATENT, 11, 1.0);
        let raw = rand(nt * TEXT_IN, 12, 1.0);
        let rope = Rope::new(f, h, w, FPS as f64, D);
        for (t, clean) in [(700.0f32, 0usize), (350.0, 5), (700.0, n)] {
            let want = {
                let _cpu = crate::hardware::cpu_compute::allow();
                let ctx = tr.project_text(&raw, nt);
                tr.forward(&x, n, &ctx, nt, t, clean, &rope)
            };
            let got = tr
                .forward_gpu(&x, n, &raw, nt, t, clean, &rope)
                .expect("metal program");
            let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("t={t} clean={clean}: rel {e:.2e} cos {c:.8}");
            assert!(e < 1e-3 && c > 0.99999, "t={t} clean={clean}: rel {e} cos {c}");
        }
    }
}
