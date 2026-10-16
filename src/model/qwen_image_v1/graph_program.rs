//! The Qwen-Image 1 transformer forward as one Metal graph program per `(text tokens, ph, pw)`.
//! The host supplies only the timestep sinusoid and the rotary position ids; `txt_norm`/`txt_in`,
//! the timestep MLP, both streams' modulation, QK-norm, RoPE, joint attention, MLPs and the
//! output head all run in the program, weights in their stored encoding.

use super::*;
use crate::graph::{PairLayout, Tensor, ValueId};
use crate::model::dit_graph::{self as dg, Gb, ProgramCache};
use crate::model::qwen_image::rope_positions;

pub(super) type Key = (usize, usize, usize);

pub(super) type Cache = ProgramCache<Key>;

impl BLin {
    fn emit(&self, b: &mut Gb, name: &str, x: ValueId) -> ValueId {
        self.w.emit(b, name, x, Some(&self.b))
    }
}

impl Dit {
    /// The velocity for packed latents `lat [ph * pw, in_ch]` given the raw text-encoder
    /// features `txt [n_text, text_width]` at `sigma`, on the GPU.
    pub(super) fn forward_gpu(
        &self,
        cache: &Cache,
        lat: &[f32],
        txt: &[f32],
        ph: usize,
        pw: usize,
        sigma: f32,
    ) -> Result<Vec<f32>, String> {
        let nt = txt.len() / self.text_width();
        let ids = rope_positions(&positions(nt, ph, pw));
        let inputs = vec![
            Tensor::F32(lat.to_vec()),
            Tensor::F32(txt.to_vec()),
            Tensor::U32(ids),
            Tensor::F32(timestep_features(1000.0 * sigma, 256)),
        ];
        let mut out = cache.run((nt, ph, pw), || self.build(nt, ph, pw), inputs)?;
        Ok(out.remove(0))
    }

    /// Inputs in order `lat, txt, pos, t_feat`.
    fn build(&self, nt: usize, ph: usize, pw: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let (d, hd, heads) = (self.dim, self.head_dim, self.heads);
        let (ni, n) = (ph * pw, nt + ph * pw);
        let mut b = Gb::new();
        let lat = b.input("lat", &[ni, self.in_ch]);
        let txt = b.input("txt", &[nt, self.text_width()]);
        let pos = b.input_u32("pos", &[n * 3]);
        let tf = b.input("t_feat", &[1, 256]);
        let (freq, sections) = dg::axis_freqs(&self.axes, THETA);
        let inv_freq = b.konst("rope.inv_freq", &[freq.len()], freq);

        let g = b.konst("txt_norm", &[self.text_width()], self.txt_norm.clone());
        let t = b.rms_norm("txt_norm.rms", txt, g, 1e-6);
        let mut xt = self.txt_in.emit(&mut b, "txt_in", t);
        let mut xi = self.img_in.emit(&mut b, "img_in", lat);

        let h = self.t_lin1.emit(&mut b, "t_lin1", tf);
        let h = b.silu("temb.act", h);
        let temb = self.t_lin2.emit(&mut b, "t_lin2", h);
        let cond = b.silu("cond", temb);

        let chunk = |b: &mut Gb, name: &str, m: ValueId, i: usize| {
            b.slice(&format!("{name}.c{i}"), m, 1, i * d, (i + 1) * d)
        };

        for (i, blk) in self.blocks.iter().enumerate() {
            let p = format!("blk{i}");
            let mi = blk
                .img
                .modulation
                .emit(&mut b, &format!("{p}.img.mod"), cond);
            let mt = blk
                .txt
                .modulation
                .emit(&mut b, &format!("{p}.txt.mod"), cond);
            let qkv = |b: &mut Gb, st: &Stream, tag: &str, m: ValueId, x: ValueId| {
                let name = format!("{p}.{tag}");
                let (sh, sc) = (
                    chunk(b, &format!("{name}.m"), m, SHIFT1),
                    chunk(b, &format!("{name}.m"), m, SCALE1),
                );
                let h = b.ln_mod(&format!("{name}.in"), x, sh, sc, 1e-6);
                let q = st.q.emit(b, &format!("{name}.q"), h);
                let k = st.k.emit(b, &format!("{name}.k"), h);
                let v = st.v.emit(b, &format!("{name}.v"), h);
                let (nq, nk) = (
                    b.konst(&format!("{name}.nq"), &[hd], st.norm_q.clone()),
                    b.konst(&format!("{name}.nk"), &[hd], st.norm_k.clone()),
                );
                let q = b.heads_rms(&format!("{name}.qn"), q, nq, hd, 1e-6);
                let k = b.heads_rms(&format!("{name}.kn"), k, nk, hd, 1e-6);
                (q, k, v)
            };
            let (iq, ik, iv) = qkv(&mut b, &blk.img, "img", mi, xi);
            let (tq, tk, tv) = qkv(&mut b, &blk.txt, "txt", mt, xt);
            // Joint attention over [text; image].
            let q = b.concat(&format!("{p}.q"), &[tq, iq], 0);
            let k = b.concat(&format!("{p}.k"), &[tk, ik], 0);
            let v = b.concat(&format!("{p}.v"), &[tv, iv], 0);
            let q = b.rope(
                &format!("{p}.rq"),
                q,
                pos,
                inv_freq,
                hd,
                sections,
                PairLayout::Adjacent,
            );
            let k = b.rope(
                &format!("{p}.rk"),
                k,
                pos,
                inv_freq,
                hd,
                sections,
                PairLayout::Adjacent,
            );
            let a = b.attention(&format!("{p}.attn"), q, k, v, heads);
            let ta = b.slice(&format!("{p}.ta"), a, 0, 0, nt);
            let ia = b.slice(&format!("{p}.ia"), a, 0, nt, n);
            for (tag, st, m, x, ao) in [
                ("img", &blk.img, mi, &mut xi, ia),
                ("txt", &blk.txt, mt, &mut xt, ta),
            ] {
                let name = format!("{p}.{tag}");
                let o = st.o.emit(&mut b, &format!("{name}.o"), ao);
                let g1 = chunk(&mut b, &format!("{name}.g1"), m, GATE1);
                *x = b.gated_add(&format!("{name}.res1"), *x, o, g1);
                let (sh, sc, g2) = (
                    chunk(&mut b, &format!("{name}.m2"), m, SHIFT2),
                    chunk(&mut b, &format!("{name}.m2"), m, SCALE2),
                    chunk(&mut b, &format!("{name}.m2"), m, GATE2),
                );
                let h = b.ln_mod(&format!("{name}.mlpin"), *x, sh, sc, 1e-6);
                let h = st.mlp_in.emit(&mut b, &format!("{name}.mlp_in"), h);
                let h = b.gelu_tanh(&format!("{name}.gelu"), h);
                let o = st.mlp_out.emit(&mut b, &format!("{name}.mlp_out"), h);
                *x = b.gated_add(&format!("{name}.res2"), *x, o, g2);
            }
        }

        // AdaLayerNormContinuous: `emb = [scale | shift]`.
        let emb = self.norm_out.emit(&mut b, "norm_out", cond);
        let (sc, sh) = (
            chunk(&mut b, "final.m", emb, 0),
            chunk(&mut b, "final.m", emb, 1),
        );
        let y = b.ln_mod("final.in", xi, sh, sc, 1e-6);
        let out = self.proj_out.emit(&mut b, "proj_out", y);
        Ok((b, vec![out]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, mat, rand, rel_err};

    const D: usize = 256;
    const HD: usize = 64;
    const FF: usize = 384;
    const TW: usize = 48;
    const IN: usize = 24;

    struct Seed(u64);
    impl Seed {
        fn lin(&mut self, n: usize, k: usize) -> BLin {
            self.0 += 1;
            let w = QLin::from_mat(mat(n, k, self.0));
            BLin {
                w,
                b: rand(n, self.0 + 7777, 0.1),
            }
        }
        fn gain(&mut self, n: usize) -> Vec<f32> {
            self.0 += 1;
            rand(n, self.0, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect()
        }
    }

    fn tiny() -> Dit {
        let s = &mut Seed(200);
        let blocks = (0..2)
            .map(|_| {
                let stream = |s: &mut Seed| Stream {
                    modulation: s.lin(6 * D, D),
                    q: s.lin(D, D),
                    k: s.lin(D, D),
                    v: s.lin(D, D),
                    norm_q: s.gain(HD),
                    norm_k: s.gain(HD),
                    o: s.lin(D, D),
                    mlp_in: s.lin(FF, D),
                    mlp_out: s.lin(D, FF),
                };
                Block {
                    img: stream(s),
                    txt: stream(s),
                }
            })
            .collect();
        Dit {
            dim: D,
            heads: D / HD,
            head_dim: HD,
            in_ch: IN,
            axes: [8, 28, 28],
            img_in: s.lin(D, IN),
            txt_norm: s.gain(TW),
            txt_in: s.lin(D, TW),
            t_lin1: s.lin(D, 256),
            t_lin2: s.lin(D, D),
            blocks,
            norm_out: s.lin(2 * D, D),
            proj_out: s.lin(IN, D),
            gpu: Default::default(),
        }
    }

    /// Metal program == native CPU forward on an odd patch grid (asymmetric centred positions).
    #[test]
    fn qwen1_program_matches_cpu_forward() {
        let dit = tiny();
        let (ph, pw, nt) = (5usize, 7usize, 21usize);
        let ni = ph * pw;
        let lat = rand(ni * IN, 11, 1.0);
        let txt = rand(nt * TW, 12, 1.0);
        let rope = Rope::new(&positions(nt, ph, pw), &dit.axes);
        for sigma in [0.8f32, 0.15] {
            let want = {
                let _cpu = crate::hardware::cpu_compute::allow();
                let ctx = dit.project_text(&txt, nt);
                dit.forward(&lat, &ctx, &rope, sigma)
            };
            let got = dit
                .forward_gpu(&dit.gpu, &lat, &txt, ph, pw, sigma)
                .expect("metal program");
            let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("sigma={sigma}: rel {e:.2e} cos {c:.7}");
            assert!(e < 1e-3 && c > 0.99999, "rel {e} cos {c}");
        }
    }
}
