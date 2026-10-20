//! The FLUX transformer forward as one Metal graph program per `(image tokens, text tokens,
//! guided)`. The host supplies only scalar arithmetic: timestep/guidance sinusoids, Chroma's
//! approximator input rows and the RoPE position ids. Every matmul, norm, modulation, RoPE,
//! attention and activation runs in the program; weights stay in their stored encoding.

use super::*;
use crate::graph::{PairLayout, Tensor, ValueId};
use crate::model::dit_graph::{self as dg, Gb, ProgramCache};
use crate::model::weights::{Mat, Packed};

/// `(ni, nt, guidance input present)`.
pub(super) type Key = (usize, usize, bool);

pub(super) type Cache = ProgramCache<Key>;

fn wt_mat(w: &Wt) -> Mat {
    match w {
        Wt::Bf16(m) => m.clone(),
        Wt::Gguf {
            map,
            off,
            rows,
            cols,
            fmt,
            ..
        } => Mat::from_packed(
            Packed {
                map: map.clone(),
                off: *off,
                format: *fmt,
                shape: vec![*rows, *cols],
            },
            *rows,
            *cols,
        ),
    }
}

impl Lin {
    /// `x [m, k] -> [m, n]`: one linear per stored part (concatenated along the output), then the
    /// LoRAs `out[:, off..off+len] += scale * up(down(x))`.
    pub(crate) fn emit(&self, b: &mut Gb, name: &str, x: ValueId) -> ValueId {
        let outs: Vec<ValueId> = self
            .parts
            .iter()
            .enumerate()
            .map(|(i, (off, w))| {
                let m = wt_mat(w);
                let bias = self.bias.as_ref().map(|v| &v[*off..*off + m.rows]);
                b.lin_mat(&format!("{name}.p{i}"), x, &m, bias)
            })
            .collect();
        let mut y = if outs.len() == 1 {
            outs[0]
        } else {
            b.concat(name, &outs, 1)
        };
        for (i, l) in self.loras.iter().enumerate() {
            let p = format!("{name}.lora{i}");
            let down = b.konst(&format!("{p}.down"), &[l.rank, self.k], l.down.clone());
            let up = b.konst(&format!("{p}.up"), &[l.len, l.rank], l.up.clone());
            let t = b.linear(&format!("{p}.t"), x, down, None);
            let u = b.linear(&format!("{p}.u"), t, up, None);
            let mut u = b.scale(&format!("{p}.s"), u, l.scale);
            if l.len != self.n {
                u = b.pad_last(&format!("{p}.pad"), u, l.off, self.n - l.off - l.len);
            }
            y = b.add(&format!("{p}.add"), y, u);
        }
        y
    }
}

impl Embed {
    /// `in_layer -> SiLU -> out_layer`.
    fn emit(&self, b: &mut Gb, name: &str, x: ValueId) -> ValueId {
        let h = self.a.emit(b, &format!("{name}.a"), x);
        let h = b.silu(&format!("{name}.act"), h);
        self.b.emit(b, &format!("{name}.b"), h)
    }
}

/// RoPE tables for `MRope`: per-pair frequencies, section sizes and which id axes feed the three
/// position slots. FLUX.2's four axes `(t, h, w, l)` have `t == 0` for every token, so its pairs
/// share the first section with `h` at zero frequency (an identity rotation).
pub(crate) fn rope_tables(axes: &[usize], theta: f64) -> (Vec<f32>, [u32; 3], [usize; 3]) {
    if axes.len() == 4 {
        let (f, sec) = dg::axis_freqs(&[axes[0] + axes[1], axes[2], axes[3]], theta);
        let mut f = f;
        // The merged section's first `axes[0] / 2` pairs belong to the t axis: no rotation.
        let (h, _) = dg::axis_freqs(&[axes[1]], theta);
        f.splice(
            0..(axes[0] + axes[1]) / 2,
            std::iter::repeat_n(0.0, axes[0] / 2).chain(h),
        );
        (f, sec, [1, 2, 3])
    } else {
        let (f, sec) = dg::axis_freqs(axes, theta);
        (f, sec, [0, 1, 2])
    }
}

impl FluxTransformer {
    /// The velocity for packed latents `img [ni, c]` given the unprojected text `ctx [nt, k]`,
    /// the raw pooled CLIP vector (FLUX.1; empty otherwise), the time `t` (= sigma) and the
    /// guidance scale, on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        cache: &Cache,
        img: &[f32],
        ni: usize,
        ctx: &[f32],
        nt: usize,
        pooled: &[f32],
        t: f32,
        guidance: Option<f32>,
        ids: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        let guided = guidance.is_some() && self.guidance_embeds();
        let (_, _, use_axes) = rope_tables(&self.axes, self.theta);
        let pos = dg::positions(
            &ids.iter().map(|i| i.to_vec()).collect::<Vec<_>>(),
            use_axes,
        );
        let mut inputs = vec![
            Tensor::F32(img.to_vec()),
            Tensor::F32(ctx.to_vec()),
            Tensor::U32(pos),
        ];
        match &self.modulation {
            Modulation::Vector { .. } => {
                inputs.push(Tensor::F32(sinusoid(t as f64 * 1000.0, 256)));
                inputs.push(Tensor::F32(pooled.to_vec()));
                if guided {
                    inputs.push(Tensor::F32(sinusoid(
                        guidance.unwrap_or(0.0) as f64 * 1000.0,
                        256,
                    )));
                }
            }
            Modulation::Table(a) => inputs.push(Tensor::F32(a.input_rows(t, self.table_rows()))),
            Modulation::Shared { .. } => inputs.push(Tensor::F32(sinusoid(t as f64 * 1000.0, 256))),
        }
        let mut out = cache.run((ni, nt, guided), || self.build(ni, nt, guided), inputs)?;
        Ok(out.remove(0))
    }

    fn table_rows(&self) -> usize {
        3 * self.single.len() + 12 * self.double.len() + 2
    }

    /// Emits the whole forward; inputs in order `img, txt, pos, <modulation inputs>`.
    fn build(&self, ni: usize, nt: usize, guided: bool) -> Result<(Gb, Vec<ValueId>), String> {
        let d = self.d;
        let (nd, ns) = (self.double.len(), self.single.len());
        let s = nt + ni;
        let mut b = Gb::new();
        let img_x = b.input("img", &[ni, self.img_in.k]);
        let ctx = b.input("txt", &[nt, self.txt_in.k]);
        let pos = b.input_u32("pos", &[s * 3]);
        let (freq, sections, _) = rope_tables(&self.axes, self.theta);
        let inv_freq = b.konst("rope.inv_freq", &[freq.len()], freq);
        let swiglu = self.swiglu;

        // Modulation source: [1, d] SiLU(vec), or Chroma's [rows, d] table.
        let table = matches!(self.modulation, Modulation::Table(_));
        let src = match &self.modulation {
            Modulation::Vector {
                time_in,
                vector_in,
                guidance_in,
            } => {
                let tf = b.input("t_feat", &[1, 256]);
                let pooled = b.input("pooled", &[1, vector_in.a.k]);
                let mut v = time_in.emit(&mut b, "time_in", tf);
                let y = vector_in.emit(&mut b, "vector_in", pooled);
                v = b.add("vec.y", v, y);
                if let (true, Some(g_in)) = (guided, guidance_in) {
                    let gf = b.input("g_feat", &[1, 256]);
                    let g = g_in.emit(&mut b, "guidance_in", gf);
                    v = b.add("vec.g", v, g);
                }
                b.silu("vec.silu", v)
            }
            Modulation::Table(a) => {
                let rows = self.table_rows();
                let input = b.input("approx_in", &[rows, a.in_proj.k]);
                a.emit(&mut b, input)
            }
            Modulation::Shared { time_in, .. } => {
                let tf = b.input("t_feat", &[1, 256]);
                let v = time_in.emit(&mut b, "time_in", tf);
                b.silu("vec.silu", v)
            }
        };
        let shared = match &self.modulation {
            Modulation::Shared {
                img, txt, single, ..
            } => Some((
                img.emit(&mut b, "mod.img", src),
                txt.emit(&mut b, "mod.txt", src),
                single.emit(&mut b, "mod.single", src),
            )),
            _ => None,
        };
        // `rows` modulation rows starting at table row `row`, as [1, rows * d].
        let modv = |b: &mut Gb, name: &str, l: &Option<Lin>, row: usize, rows: usize| match l {
            Some(l) => l.emit(b, name, src),
            None => {
                debug_assert!(table);
                let sl = b.slice(&format!("{name}.rows"), src, 0, row, row + rows);
                b.reshape(name, sl, &[1, rows * d])
            }
        };
        let chunk = |b: &mut Gb, name: &str, m: ValueId, i: usize| {
            b.slice(&format!("{name}.c{i}"), m, 1, i * d, (i + 1) * d)
        };

        let mut img = self.img_in.emit(&mut b, "img_in", img_x);
        let mut txt = self.txt_in.emit(&mut b, "txt_in", ctx);

        for (i, blk) in self.double.iter().enumerate() {
            let p = format!("double.{i}");
            let (im, tm) = match shared {
                Some((im, tm, _)) => (im, tm),
                None => (
                    modv(
                        &mut b,
                        &format!("{p}.img.mod"),
                        &blk.img.modl,
                        3 * ns + 6 * i,
                        6,
                    ),
                    modv(
                        &mut b,
                        &format!("{p}.txt.mod"),
                        &blk.txt.modl,
                        3 * ns + 6 * nd + 6 * i,
                        6,
                    ),
                ),
            };
            let qkv = |b: &mut Gb, st: &Stream, tag: &str, m: ValueId, x: ValueId, n: usize| {
                let name = format!("{p}.{tag}");
                let (sh, sc) = (
                    chunk(b, &format!("{name}.m"), m, 0),
                    chunk(b, &format!("{name}.m"), m, 1),
                );
                let h = b.ln_mod(&format!("{name}.in"), x, sh, sc, EPS);
                let f = st.qkv.emit(b, &format!("{name}.qkv"), h);
                let q = b.slice(&format!("{name}.q"), f, 1, 0, d);
                let k = b.slice(&format!("{name}.k"), f, 1, d, 2 * d);
                let v = b.slice(&format!("{name}.v"), f, 1, 2 * d, 3 * d);
                let (qn, kn) = (
                    b.konst(&format!("{name}.qn"), &[HEAD_DIM], st.qn.clone()),
                    b.konst(&format!("{name}.kn"), &[HEAD_DIM], st.kn.clone()),
                );
                let q = b.heads_rms(&format!("{name}.qnorm"), q, qn, HEAD_DIM, EPS);
                let k = b.heads_rms(&format!("{name}.knorm"), k, kn, HEAD_DIM, EPS);
                debug_assert_eq!(b.shape(q), vec![n, d]);
                (q, k, v)
            };
            let (iq, ik, iv) = qkv(&mut b, &blk.img, "img", im, img, ni);
            let (tq, tk, tv) = qkv(&mut b, &blk.txt, "txt", tm, txt, nt);
            // Joint attention over [text; image].
            let q = b.concat(&format!("{p}.q"), &[tq, iq], 0);
            let k = b.concat(&format!("{p}.k"), &[tk, ik], 0);
            let v = b.concat(&format!("{p}.v"), &[tv, iv], 0);
            let q = b.rope(
                &format!("{p}.rq"),
                q,
                pos,
                inv_freq,
                HEAD_DIM,
                sections,
                PairLayout::Adjacent,
            );
            let k = b.rope(
                &format!("{p}.rk"),
                k,
                pos,
                inv_freq,
                HEAD_DIM,
                sections,
                PairLayout::Adjacent,
            );
            let a = b.attention(&format!("{p}.attn"), q, k, v, self.heads);
            let ta = b.slice(&format!("{p}.ta"), a, 0, 0, nt);
            let ia = b.slice(&format!("{p}.ia"), a, 0, nt, s);
            for (tag, st, m, x, ao) in [
                ("img", &blk.img, im, &mut img, ia),
                ("txt", &blk.txt, tm, &mut txt, ta),
            ] {
                let name = format!("{p}.{tag}");
                let o = st.proj.emit(&mut b, &format!("{name}.proj"), ao);
                let g1 = chunk(&mut b, &format!("{name}.g1"), m, 2);
                *x = b.gated_add(&format!("{name}.res1"), *x, o, g1);
                let (sh, sc, g2) = (
                    chunk(&mut b, &format!("{name}.m2"), m, 3),
                    chunk(&mut b, &format!("{name}.m2"), m, 4),
                    chunk(&mut b, &format!("{name}.m2"), m, 5),
                );
                let h = b.ln_mod(&format!("{name}.mlpin"), *x, sh, sc, EPS);
                let h = st.mlp0.emit(&mut b, &format!("{name}.mlp0"), h);
                let h = if swiglu {
                    let w = st.mlp0.n / 2;
                    let g = b.slice(&format!("{name}.swg"), h, 1, 0, w);
                    let u = b.slice(&format!("{name}.swu"), h, 1, w, 2 * w);
                    b.swiglu(&format!("{name}.swiglu"), g, u)
                } else {
                    b.gelu_tanh(&format!("{name}.gelu"), h)
                };
                let o = st.mlp2.emit(&mut b, &format!("{name}.mlp2"), h);
                *x = b.gated_add(&format!("{name}.res2"), *x, o, g2);
            }
        }

        let mut x = b.concat("stream.cat", &[txt, img], 0);
        for (i, blk) in self.single.iter().enumerate() {
            let p = format!("single.{i}");
            let m = match shared {
                Some((_, _, sm)) => sm,
                None => modv(&mut b, &format!("{p}.mod"), &blk.modl, 3 * i, 3),
            };
            let (sh, sc, g) = (
                chunk(&mut b, &format!("{p}.m"), m, 0),
                chunk(&mut b, &format!("{p}.m"), m, 1),
                chunk(&mut b, &format!("{p}.m"), m, 2),
            );
            let h = b.ln_mod(&format!("{p}.in"), x, sh, sc, EPS);
            let h = blk.linear1.emit(&mut b, &format!("{p}.linear1"), h);
            let w1 = blk.linear1.n;
            let q = b.slice(&format!("{p}.q"), h, 1, 0, d);
            let k = b.slice(&format!("{p}.k"), h, 1, d, 2 * d);
            let v = b.slice(&format!("{p}.v"), h, 1, 2 * d, 3 * d);
            let mlp_in = b.slice(&format!("{p}.mlp"), h, 1, 3 * d, w1);
            let act = if swiglu {
                let w = (w1 - 3 * d) / 2;
                let gt = b.slice(&format!("{p}.swg"), mlp_in, 1, 0, w);
                let up = b.slice(&format!("{p}.swu"), mlp_in, 1, w, 2 * w);
                b.swiglu(&format!("{p}.swiglu"), gt, up)
            } else {
                b.gelu_tanh(&format!("{p}.gelu"), mlp_in)
            };
            let (qn, kn) = (
                b.konst(&format!("{p}.qn"), &[HEAD_DIM], blk.qn.clone()),
                b.konst(&format!("{p}.kn"), &[HEAD_DIM], blk.kn.clone()),
            );
            let q = b.heads_rms(&format!("{p}.qnorm"), q, qn, HEAD_DIM, EPS);
            let k = b.heads_rms(&format!("{p}.knorm"), k, kn, HEAD_DIM, EPS);
            let q = b.rope(
                &format!("{p}.rq"),
                q,
                pos,
                inv_freq,
                HEAD_DIM,
                sections,
                PairLayout::Adjacent,
            );
            let k = b.rope(
                &format!("{p}.rk"),
                k,
                pos,
                inv_freq,
                HEAD_DIM,
                sections,
                PairLayout::Adjacent,
            );
            let a = b.attention(&format!("{p}.attn"), q, k, v, self.heads);
            let cat = b.concat(&format!("{p}.cat"), &[a, act], 1);
            let o = blk.linear2.emit(&mut b, &format!("{p}.linear2"), cat);
            x = b.gated_add(&format!("{p}.res"), x, o, g);
        }

        let m = modv(&mut b, "final.mod", &self.final_mod, 3 * ns + 12 * nd, 2);
        let (mut shift, mut scale) = (
            chunk(&mut b, "final.m", m, 0),
            chunk(&mut b, "final.m", m, 1),
        );
        if self.final_swap {
            (shift, scale) = (scale, shift);
        }
        let xi = b.slice("final.img", x, 0, nt, s);
        let h = b.ln_mod("final.in", xi, shift, scale, EPS);
        let out = self.final_lin.emit(&mut b, "final_lin", h);
        Ok((b, vec![out]))
    }
}

impl Approx {
    /// The `[rows, in_proj.k]` input of [`Self::fwd`]: per row `[sin(t*1000) 16, sin(0) 16, sin(i*1000) 32]`.
    pub(super) fn input_rows(&self, t: f32, rows: usize) -> Vec<f32> {
        let (tp, gp) = (sinusoid(t as f64 * 1000.0, 16), sinusoid(0.0, 16));
        (0..rows)
            .flat_map(|i| {
                tp.iter()
                    .chain(&gp)
                    .copied()
                    .chain(sinusoid(i as f64 * 1000.0, 32))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The modulation table `[rows, d]` from its input rows.
    fn emit(&self, b: &mut Gb, input: ValueId) -> ValueId {
        let mut x = self.in_proj.emit(b, "approx.in_proj", input);
        let w = self.in_proj.n;
        for (i, (norm, mlp)) in self.layers.iter().enumerate() {
            let g = b.konst(&format!("approx.{i}.norm"), &[w], norm.clone());
            // torch.nn.RMSNorm(eps=None) in bf16 uses 2^-7.
            let h = b.rms_norm(&format!("approx.{i}.rms"), x, g, 0.0078125);
            let h = mlp.emit(b, &format!("approx.{i}.mlp"), h);
            x = b.add(&format!("approx.{i}.res"), x, h);
        }
        self.out_proj.emit(b, "approx.out_proj", x)
    }
}

/// Tiny random-weight transformers: the parity tests' and the GPU-only matrix's fixtures.
pub(super) mod fixture {
    use super::super::*;
    use crate::model::dit_graph::fixture::{mat, rand};

    /// The tiny FLUX.1 forward (36 image + 28 text tokens, guided) as a graph.
    pub(in crate::model::flux) fn tiny_graph(
    ) -> Result<(crate::graph::Graph, crate::graph::Weights), String> {
        let (mut b, outs) = tiny(Kind::Flux1, 1).build(36, 28, true)?;
        b.g.output(outs[0]);
        Ok((b.g, b.w))
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy, Debug)]
    pub(in crate::model::flux) enum Kind {
        Flux1,
        Chroma,
        Flux2,
    }

    pub(in crate::model::flux) struct Rng(u64);
    impl Rng {
        fn seed(&mut self) -> u64 {
            self.0 += 1;
            self.0
        }
    }

    fn lin(r: &mut Rng, n: usize, k: usize) -> Lin {
        let bias = rand(n, r.seed(), 0.1);
        Lin::new(Wt::Bf16(mat(n, k, r.seed())), Some(bias), vec![])
    }

    /// A linear stored as `parts` stacked row blocks (diffusers q/k/v), with an optional LoRA.
    fn lin_parts(r: &mut Rng, widths: &[usize], k: usize, lora: bool) -> Lin {
        let n: usize = widths.iter().sum();
        let (mut parts, mut off) = (Vec::new(), 0);
        for &w in widths {
            parts.push((off, Wt::Bf16(mat(w, k, r.seed()))));
            off += w;
        }
        let loras = if lora {
            vec![Lora {
                down: rand(2 * k, r.seed(), 0.2),
                up: rand(widths[0] * 2, r.seed(), 0.2),
                rank: 2,
                off: 0,
                len: widths[0],
                scale: 0.7,
            }]
        } else {
            vec![]
        };
        Lin {
            parts,
            bias: Some(rand(n, r.seed(), 0.1)),
            n,
            k,
            loras,
        }
    }

    fn embed(r: &mut Rng, d_in: usize, d: usize) -> Embed {
        Embed {
            a: lin(r, d, d_in),
            b: lin(r, d, d),
        }
    }

    pub(in crate::model::flux) const D: usize = 256;
    pub(in crate::model::flux) const POOLED: usize = 32;

    pub(in crate::model::flux) fn tiny(kind: Kind, seed: u64) -> FluxTransformer {
        let r = &mut Rng(seed * 1000);
        let flux2 = matches!(kind, Kind::Flux2);
        let chroma = matches!(kind, Kind::Chroma);
        let mlp = if flux2 { 2 * 384 } else { 4 * D };
        let mlp_out_in = if flux2 { 384 } else { 4 * D };
        let (nd, ns) = (2usize, 2usize);
        let stream = |r: &mut Rng, lora: bool| Stream {
            modl: (!chroma && !flux2).then(|| lin(r, 6 * D, D)),
            qkv: lin_parts(r, &[D, D, D], D, lora),
            qn: rand(HEAD_DIM, r.seed(), 1.0)
                .iter()
                .map(|v| 1.0 + 0.3 * v)
                .collect(),
            kn: rand(HEAD_DIM, r.seed(), 1.0)
                .iter()
                .map(|v| 1.0 + 0.3 * v)
                .collect(),
            proj: lin(r, D, D),
            mlp0: lin(r, mlp, D),
            mlp2: lin(r, D, mlp_out_in),
        };
        let double = (0..nd)
            .map(|i| Double {
                img: stream(r, i == 0),
                txt: stream(r, false),
            })
            .collect();
        let single = (0..ns)
            .map(|_| {
                let mlp_in = if flux2 { 2 * 320 } else { 4 * D };
                let mlp_w = if flux2 { 320 } else { 4 * D };
                Single {
                    modl: (!chroma && !flux2).then(|| lin(r, 3 * D, D)),
                    linear1: lin_parts(r, &[D, D, D, mlp_in], D, false),
                    linear2: lin(r, D, D + mlp_w),
                    qn: rand(HEAD_DIM, r.seed(), 1.0)
                        .iter()
                        .map(|v| 1.0 + 0.3 * v)
                        .collect(),
                    kn: rand(HEAD_DIM, r.seed(), 1.0)
                        .iter()
                        .map(|v| 1.0 + 0.3 * v)
                        .collect(),
                }
            })
            .collect();
        let modulation = match kind {
            Kind::Flux1 => Modulation::Vector {
                time_in: embed(r, 256, D),
                vector_in: embed(r, POOLED, D),
                guidance_in: Some(embed(r, 256, D)),
            },
            Kind::Chroma => Modulation::Table(Approx {
                in_proj: lin(r, D, 64),
                layers: (0..2)
                    .map(|_| {
                        (
                            rand(D, r.seed(), 1.0)
                                .iter()
                                .map(|v| 1.0 + 0.2 * v)
                                .collect(),
                            embed(r, D, D),
                        )
                    })
                    .collect(),
                out_proj: lin(r, D, D),
            }),
            Kind::Flux2 => Modulation::Shared {
                time_in: embed(r, 256, D),
                img: lin(r, 6 * D, D),
                txt: lin(r, 6 * D, D),
                single: lin(r, 3 * D, D),
            },
        };
        FluxTransformer {
            img_in: lin(r, D, 16),
            txt_in: lin(r, D, 24),
            modulation,
            double,
            single,
            final_mod: (!chroma).then(|| lin(r, 2 * D, D)),
            final_swap: !chroma,
            final_lin: lin(r, 16, D),
            d: D,
            heads: D / HEAD_DIM,
            axes: if flux2 { vec![32; 4] } else { AXES.to_vec() },
            theta: if flux2 { 2000.0 } else { THETA },
            swiglu: flux2,
            label: "test".into(),
            gpu: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;
    use crate::model::dit_graph::fixture::{cosine, rand, rel_err};

    /// Metal program == native CPU forward, per FLUX variant, at f32 (bf16 weights shared).
    #[test]
    fn flux_program_matches_cpu_forward() {
        for kind in [Kind::Flux1, Kind::Chroma, Kind::Flux2] {
            let tr = tiny(kind, 1 + kind as u64);
            let (hp, wp, nt) = (6usize, 6usize, 28usize);
            let ni = hp * wp;
            let img = rand(ni * 16, 11, 1.0);
            let ctx = rand(nt * 24, 12, 1.0);
            let pooled = rand(POOLED, 13, 1.0);
            let ids = position_ids(nt, hp, wp, tr.swiglu);
            let rope = Rope::new(&ids, &tr.axes, tr.theta);
            let y = if matches!(kind, Kind::Flux1) {
                tr.pooled(&pooled)
            } else {
                Vec::new()
            };
            for (t, guidance) in [(0.7f32, Some(3.5f32)), (0.2, None)] {
                let (want, got) = {
                    let txt = tr.txt_in.fwd(&ctx, nt);
                    let want = {
                        let _cpu = crate::hardware::cpu_compute::allow();
                        tr.forward(&img, ni, &txt, nt, &y, t, guidance, &rope)
                    };
                    let pooled_in: &[f32] = if matches!(kind, Kind::Flux1) {
                        &pooled
                    } else {
                        &[]
                    };
                    let got = tr
                        .forward_gpu(&tr.gpu, &img, ni, &ctx, nt, pooled_in, t, guidance, &ids)
                        .expect("metal program");
                    (want, got)
                };
                let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
                eprintln!("{kind:?} t={t} guidance={guidance:?}: rel {e:.2e} cos {c:.7}");
                assert!(e < 2e-3 && c > 0.99999, "{kind:?}: rel {e} cos {c}");
            }
        }
    }
}

#[cfg(test)]
mod approx_tests {
    use super::*;
    use crate::model::dit_graph::fixture::{mat, rand, rel_err};

    #[test]
    fn approx_table_matches_cpu() {
        let lin = |n: usize, k: usize, s: u64| {
            Lin::new(Wt::Bf16(mat(n, k, s)), Some(rand(n, s + 99, 0.1)), vec![])
        };
        let emb = |s: u64| Embed {
            a: lin(256, 256, s),
            b: lin(256, 256, s + 1),
        };
        let a = Approx {
            in_proj: lin(256, 64, 1),
            layers: (0..2)
                .map(|i| {
                    (
                        rand(256, 7 + i, 1.0)
                            .iter()
                            .map(|v| 1.0 + 0.2 * v)
                            .collect(),
                        emb(10 + 3 * i),
                    )
                })
                .collect(),
            out_proj: lin(256, 256, 40),
        };
        let rows = 32;
        let want = {
            let _g = crate::hardware::cpu_compute::allow();
            a.fwd(0.6, rows)
        };
        let mut b = Gb::new();
        let input = b.input("approx_in", &[rows, 64]);
        let out = a.emit(&mut b, input);
        let mut p = crate::model::dit_graph::Program::compile(b, &[out]).unwrap();
        let got = p
            .run(vec![Tensor::F32(a.input_rows(0.6, rows))])
            .unwrap()
            .remove(0);
        eprintln!("approx rel {:.2e}", rel_err(&want, &got));
        assert!(rel_err(&want, &got) < 1e-3);
    }
}
