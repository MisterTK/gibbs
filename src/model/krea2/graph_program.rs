//! The Krea 2 transformer on Metal: one program fusing the twelve encoder taps into text features
//! (per token count) and one program for the DiT forward (per `(image tokens, text tokens)`). The
//! host supplies only the timestep sinusoid, RoPE position ids and the patchify shuffles.

use super::*;
use crate::graph::{DType, Op, PairLayout, Tensor, UnaryKind, ValueId};
use crate::model::dit_graph::{self as dg, Gb, ProgramCache};

pub(super) type FuseCache = ProgramCache<usize>;
/// `(image tokens, text tokens)`.
pub(super) type FwdCache = ProgramCache<(usize, usize)>;

/// Position ids, per-pair frequencies and sections of the shared RoPE.
type Rope = (ValueId, ValueId, [u32; 3]);

impl Attn {
    /// Attention of `x [rows, d]`: inside the `rows / group` consecutive groups when `group` is
    /// `Some`, else over all rows. Sigmoid-gated, GQA, per-head QK-norm, optional RoPE.
    fn emit(&self, b: &mut Gb, p: &str, x: ValueId, rows: usize, group: Option<usize>, rope: Option<Rope>) -> ValueId {
        let (heads, kv, hd) = (self.heads, self.kv, self.hd);
        let q = self.q.emit(b, &format!("{p}.wq"), x);
        let k = self.k.emit(b, &format!("{p}.wk"), x);
        let v = self.v.emit(b, &format!("{p}.wv"), x);
        let gate = self.gate.emit(b, &format!("{p}.gate"), x);
        let qn = b.konst(&format!("{p}.qn"), &[hd], self.qn.clone());
        let kn = b.konst(&format!("{p}.kn"), &[hd], self.kn.clone());
        let mut q = b.heads_rms(&format!("{p}.qnorm"), q, qn, hd, EPS);
        let mut k = b.heads_rms(&format!("{p}.knorm"), k, kn, hd, EPS);
        if let Some((pos, freq, sec)) = rope {
            q = b.rope(&format!("{p}.rq"), q, pos, freq, hd, sec, PairLayout::Adjacent);
            k = b.rope(&format!("{p}.rk"), k, pos, freq, hd, sec, PairLayout::Adjacent);
        }
        let w = heads * hd;
        let a = match group {
            None => b.g.add(
                &format!("{p}.attn"),
                Op::Attention {
                    heads,
                    kv_heads: kv,
                    causal: false,
                    window: None,
                    soft_cap: None,
                },
                &[q, k, v],
                vec![rows, w],
                DType::F32,
            ),
            Some(g) => {
                // Batched over groups: query rows of one kv head are `(pos, rep)` pairs, which all
                // see the same g keys (no mask), so the order inside the group is free.
                let (n, rep) = (rows / g, heads / kv);
                let to_heads = |b: &mut Gb, name: &str, x: ValueId, per: usize| {
                    let r = b.reshape(&format!("{name}.r"), x, &[n, g, kv, per * hd]);
                    let t = b.transpose(&format!("{name}.t"), r, &[0, 2, 1, 3]);
                    b.reshape(name, t, &[n, kv, g * per, hd])
                };
                let q = to_heads(b, &format!("{p}.gq"), q, rep);
                let k = to_heads(b, &format!("{p}.gk"), k, 1);
                let v = to_heads(b, &format!("{p}.gv"), v, 1);
                let q = b.scale(&format!("{p}.gqs"), q, 1.0 / (hd as f32).sqrt());
                let s = b.g.add(
                    &format!("{p}.gs"),
                    Op::MatMul {
                        transpose_a: false,
                        transpose_b: true,
                    },
                    &[q, k],
                    vec![n, kv, g * rep, g],
                    DType::F32,
                );
                let s = b.g.add(
                    &format!("{p}.gp"),
                    Op::Softmax { axis: -1 },
                    &[s],
                    vec![n, kv, g * rep, g],
                    DType::F32,
                );
                let o = b.g.add(
                    &format!("{p}.go"),
                    Op::MatMul {
                        transpose_a: false,
                        transpose_b: false,
                    },
                    &[s, v],
                    vec![n, kv, g * rep, hd],
                    DType::F32,
                );
                let o = b.reshape(&format!("{p}.go.r"), o, &[n, kv, g, rep * hd]);
                let o = b.transpose(&format!("{p}.go.t"), o, &[0, 2, 1, 3]);
                b.reshape(&format!("{p}.go.j"), o, &[rows, w])
            }
        };
        let sg = b.unary(&format!("{p}.sig"), UnaryKind::Sigmoid, gate);
        let a = b.mul(&format!("{p}.gated"), a, sg);
        self.o.emit(b, &format!("{p}.wo"), a)
    }
}

impl Block {
    /// `x [rows, d]` -> same; `tvec` (`[1, 6d]`, the shared `tproj` output) selects the modulated form.
    #[allow(clippy::too_many_arguments)]
    fn emit(&self, b: &mut Gb, p: &str, x: ValueId, rows: usize, group: Option<usize>, rope: Option<Rope>, tvec: Option<ValueId>) -> ValueId {
        let d = self.pre.len();
        let m = self.modulation.as_ref().zip(tvec).map(|(l, t)| {
            let k = b.konst(&format!("{p}.mod"), &[1, 6 * d], l.clone());
            b.add(&format!("{p}.mod.sum"), k, t)
        });
        let ch = |b: &mut Gb, i: usize| m.map(|m| b.slice(&format!("{p}.m{i}"), m, 1, i * d, (i + 1) * d));
        let finish = |b: &mut Gb, name: &str, x: ValueId, y: ValueId, g: Option<ValueId>| match g {
            Some(g) => b.gated_add(name, x, y, g),
            None => b.add(name, x, y),
        };
        let pre = b.konst(&format!("{p}.pre"), &[d], self.pre.clone());
        let mut h = b.rms_norm(&format!("{p}.prenorm"), x, pre, EPS);
        if let (Some(sc), Some(sh)) = (ch(b, 0), ch(b, 1)) {
            h = b.modulate(&format!("{p}.premod"), h, sh, sc);
        }
        let a = self.attn.emit(b, &format!("{p}.attn"), h, rows, group, rope);
        let g = ch(b, 2);
        let x = finish(b, &format!("{p}.res1"), x, a, g);
        let post = b.konst(&format!("{p}.post"), &[d], self.post.clone());
        let mut h = b.rms_norm(&format!("{p}.postnorm"), x, post, EPS);
        if let (Some(sc), Some(sh)) = (ch(b, 3), ch(b, 4)) {
            h = b.modulate(&format!("{p}.postmod"), h, sh, sc);
        }
        let g = self.gate.emit(b, &format!("{p}.mlp.gate"), h);
        let u = self.up.emit(b, &format!("{p}.mlp.up"), h);
        let a = b.swiglu(&format!("{p}.mlp.act"), g, u);
        let o = self.down.emit(b, &format!("{p}.mlp.down"), a);
        let g = ch(b, 5);
        finish(b, &format!("{p}.res2"), x, o, g)
    }
}

impl Dit {
    /// `taps` (`[12][n, txt_dim]` flat, layer-major) -> text features `[n, dim]`, on the GPU.
    pub(super) fn fuse_gpu(&self, cache: &FuseCache, taps: &[Vec<f32>], n: usize) -> Result<Vec<f32>, String> {
        let flat: Vec<f32> = taps.concat();
        let mut out = cache.run(n, || self.build_fuse(taps.len(), n), vec![Tensor::F32(flat)])?;
        Ok(out.remove(0))
    }

    fn build_fuse(&self, nl: usize, n: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let td = self.txt_norm.len();
        let mut b = Gb::new();
        let taps = b.input("taps", &[nl * n, td]);
        // Layer-major -> token-major rows (t * nl + layer).
        let r = b.reshape("taps.r", taps, &[nl, n, td]);
        let t = b.transpose("taps.t", r, &[1, 0, 2]);
        let mut x = b.reshape("taps.x", t, &[n * nl, td]);
        for (i, blk) in self.layerwise.iter().enumerate() {
            x = blk.emit(&mut b, &format!("layerwise.{i}"), x, n * nl, Some(nl), None, None);
        }
        // Linear(12 -> 1) over the layer axis: one row of 12 per (token, channel).
        let r = b.reshape("proj.r", x, &[n, nl, td]);
        let t = b.transpose("proj.t", r, &[0, 2, 1]);
        let cols = b.reshape("proj.cols", t, &[n * td, nl]);
        let y = self.projector.emit(&mut b, "projector", cols);
        let mut y = b.reshape("proj.y", y, &[n, td]);
        for (i, blk) in self.refiner.iter().enumerate() {
            y = blk.emit(&mut b, &format!("refiner.{i}"), y, n, None, None, None);
        }
        let g = b.konst("txt_norm", &[td], self.txt_norm.clone());
        let y = b.rms_norm("txt_norm.out", y, g, EPS);
        let h = self.txt1.emit(&mut b, "txt1", y);
        let h = b.gelu_tanh("txt1.gelu", h);
        let out = self.txt3.emit(&mut b, "txt3", h);
        Ok((b, vec![out]))
    }

    /// Velocity for latents `[C, h, w]` at noise level `sigma`; `ctx` is the fused `[n_txt, dim]`.
    pub(super) fn forward_gpu(&self, cache: &FwdCache, lat: &[f32], h: usize, w: usize, sigma: f32, ctx: &[f32]) -> Result<Vec<f32>, String> {
        let (gw, n_img) = (w / PATCH, (h / PATCH) * (w / PATCH));
        let n_txt = ctx.len() / self.dim;
        let ids: Vec<Vec<usize>> = (0..n_txt + n_img)
            .map(|r| match r.checked_sub(n_txt) {
                Some(i) => vec![0, i / gw, i % gw],
                None => vec![0; 3],
            })
            .collect();
        let inputs = vec![
            Tensor::F32(self.patchify(lat, h, w)),
            Tensor::F32(ctx.to_vec()),
            Tensor::U32(dg::positions(&ids, [0, 1, 2])),
            Tensor::F32(timestep_features(flow::timestep(sigma), self.tdim)),
        ];
        let mut out = cache.run((n_img, n_txt), || self.build(n_img, n_txt), inputs)?;
        Ok(self.unpatchify(&out.remove(0), h, w))
    }

    fn build(&self, ni: usize, nt: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let (d, n) = (self.dim, ni + nt);
        let mut b = Gb::new();
        let patches = b.input("patches", &[ni, self.first.in_dim()]);
        let ctx = b.input("ctx", &[nt, d]);
        let pos = b.input_u32("pos", &[n * 3]);
        let tf = b.input("t_feat", &[1, self.tdim]);
        let (freq, sections) = dg::axis_freqs(&self.axes, self.theta);
        let freq = b.konst("rope.inv_freq", &[freq.len()], freq);
        let rope = Some((pos, freq, sections));

        let t = self.tmlp0.emit(&mut b, "tmlp0", tf);
        let t = b.gelu_tanh("tmlp0.gelu", t);
        let t = self.tmlp2.emit(&mut b, "tmlp2", t);
        let tg = b.gelu_tanh("tproj.gelu", t);
        let tvec = self.tproj.emit(&mut b, "tproj", tg);

        let img = self.first.emit(&mut b, "first", patches);
        let mut x = b.concat("stream.cat", &[ctx, img], 0);
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.emit(&mut b, &format!("blocks.{i}"), x, n, None, rope, Some(tvec));
        }
        let xi = b.slice("final.img", x, 0, nt, n);
        let g = b.konst("last_norm", &[d], self.last_norm.clone());
        let y = b.rms_norm("final.norm", xi, g, EPS);
        let sc = b.konst("last_mod.scale", &[1, d], self.last_mod[..d].to_vec());
        let sh = b.konst("last_mod.shift", &[1, d], self.last_mod[d..].to_vec());
        let scale = b.add("final.scale", t, sc);
        let shift = b.add("final.shift", t, sh);
        let y = b.modulate("final.mod", y, shift, scale);
        let out = self.last.emit(&mut b, "last", y);
        Ok((b, vec![out]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, rand, rel_err};

    /// Metal programs == native CPU forward on the tiny 2-block transformer (4 query heads over
    /// 2 kv heads, text layer groups of 12, 24 text + 36 image tokens).
    #[test]
    fn krea2_program_matches_cpu_forward() {
        let path = std::env::temp_dir().join("krea2_graph_tiny.safetensors");
        super::super::tests::write_st(&path, &super::super::tests::tiny_dit());
        let dit = Dit::load(&path, &mut Loras::default(), &Value::Null).expect("tiny load");
        let (n, td) = (24usize, dit.txt_norm.len());
        let taps: Vec<Vec<f32>> = (0..12).map(|l| rand(n * td, 100 + l, 1.0)).collect();
        let want = {
            let _cpu = crate::hardware::cpu_compute::allow();
            dit.fuse(&taps, n)
        };
        let got = dit.fuse_gpu(&dit.gpu_fuse, &taps, n).expect("fuse program");
        let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
        eprintln!("fuse: rel {e:.2e} cos {c:.7}");
        assert!(e < 1e-3 && c > 0.99999, "fuse: rel {e} cos {c}");

        let ctx = want;
        let (h, w) = (12usize, 12usize);
        let lat = rand(dit.chan * h * w, 7, 1.0);
        for sigma in [0.8f32, 0.3] {
            let want = {
                let _cpu = crate::hardware::cpu_compute::allow();
                dit.forward(&lat, h, w, sigma, &ctx)
            };
            let got = dit
                .forward_gpu(&dit.gpu, &lat, h, w, sigma, &ctx)
                .expect("forward program");
            let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("forward sigma={sigma}: rel {e:.2e} cos {c:.7}");
            assert!(e < 1e-3 && c > 0.99999, "forward: rel {e} cos {c}");
        }
    }
}
