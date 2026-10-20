//! The Qwen-Image 2.1 transformer forward as one Metal graph program per `(text tokens, h, w)`.
//! The host supplies only the timestep sinusoid and the rotary position ids; text-feature
//! projection, the timestep MLP, modulation, QK-norm, RoPE, block-causal attention, the SwiGLU
//! MLP and the output head all run in the program, weights in their stored encoding.

use super::*;
use crate::graph::{DType, Op, PairLayout, Tensor, UnaryKind, ValueId};
use crate::model::dit_graph::{self as dg, Gb, ProgramCache};
use crate::model::qwen_image::rope_positions;

pub(super) type Key = (usize, usize, usize);

pub(super) type Cache = ProgramCache<Key>;

/// `[scale1 | gate1 | scale2 | gate2]` slices of a shared modulation `[1, 4d]`; gates `tanh`ed.
struct M {
    s1: ValueId,
    g1: ValueId,
    s2: ValueId,
    g2: ValueId,
}

fn mods(b: &mut Gb, name: &str, m: ValueId, d: usize) -> M {
    let mut part = |i: usize, gate: bool| {
        let s = b.slice(&format!("{name}.c{i}"), m, 1, i * d, (i + 1) * d);
        if gate {
            b.unary(&format!("{name}.tanh{i}"), UnaryKind::Tanh, s)
        } else {
            s
        }
    };
    M {
        s1: part(0, false),
        g1: part(1, true),
        s2: part(2, false),
        g2: part(3, true),
    }
}

/// Affine-free `LayerNorm(x) * (1 + scale)`, `scale [1, d]`.
fn ln_scale(b: &mut Gb, name: &str, x: ValueId, scale: ValueId) -> ValueId {
    let ln = b.layer_norm(&format!("{name}.ln"), x, 1e-6);
    let one = b.scalar(1.0);
    let s = b.add(&format!("{name}.1p"), scale, one);
    b.mul(name, ln, s)
}

/// `LayerNorm` of a text and an image stream, each modulated by its own scale, concatenated.
fn joint_ln(b: &mut Gb, name: &str, xt: ValueId, st: ValueId, xi: ValueId, si: ValueId) -> ValueId {
    let t = ln_scale(b, &format!("{name}.t"), xt, st);
    let i = ln_scale(b, &format!("{name}.i"), xi, si);
    b.concat(name, &[t, i], 0)
}

impl Dit21 {
    /// The velocity for packed latents `lat [h * w, in_ch]` given the raw text-encoder features
    /// `txt [n_text, text_width]` at `sigma`, on the GPU.
    pub fn forward_gpu(
        &self,
        cache: &Cache,
        lat: &[f32],
        txt: &[f32],
        h: usize,
        w: usize,
        sigma: f32,
    ) -> Result<Vec<f32>, String> {
        let nt = txt.len() / self.text_width();
        let ids = rope_positions(&t2i_positions(nt, h, w));
        let inputs = vec![
            Tensor::F32(lat.to_vec()),
            Tensor::F32(txt.to_vec()),
            Tensor::U32(ids),
            Tensor::F32(timestep_features(1000.0 * sigma, 256)),
        ];
        let mut out = cache.run((nt, h, w), || self.build(nt, h, w), inputs)?;
        Ok(out.remove(0))
    }

    /// Inputs in order `lat, txt, pos, t_feat`.
    fn build(&self, nt: usize, h: usize, w: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let (d, hd, heads) = (self.dim, self.head_dim, self.heads);
        let (ni, n) = (h * w, nt + h * w);
        let mut b = Gb::new();
        let lat = b.input("lat", &[ni, self.in_ch]);
        let txt = b.input("txt", &[nt, self.text_width()]);
        let pos = b.input_u32("pos", &[n * 3]);
        let tf = b.input("t_feat", &[1, 256]);
        let (freq, sections) = dg::axis_freqs(&self.axes, THETA);
        let inv_freq = b.konst("rope.inv_freq", &[freq.len()], freq);

        // txt_in: zero-centred RMSNorm (stored `+ 1`), linear, tanh-GELU, linear.
        let g = b.konst("txt_norm", &[self.text_width()], self.text_norm.clone());
        let t = b.rms_norm("txt_norm.rms", txt, g, 1e-6);
        let t = self.txt_in.emit(&mut b, "txt_in", t, None);
        let t = b.gelu_tanh("txt_in.gelu", t);
        let mut xt = self.txt_out.emit(&mut b, "txt_out", t, None);
        let mut xi = self.img_in.emit(&mut b, "img_in", lat, None);

        // Timestep embedder at t = 0 (text) and t (image), one shared modulation.
        let (w1, w2, wm) = (
            self.t_lin1.bind(&mut b, "t_lin1"),
            self.t_lin2.bind(&mut b, "t_lin2"),
            self.modulation.bind(&mut b, "modulation"),
        );
        let t0 = b.konst("t0_feat", &[1, 256], timestep_features(0.0, 256));
        let silu_temb = |b: &mut Gb, tag: &str, feat: ValueId| {
            let h = b.linear(&format!("{tag}.l1"), feat, w1, None);
            let h = b.silu(&format!("{tag}.act"), h);
            let e = b.linear(&format!("{tag}.l2"), h, w2, None);
            b.silu(&format!("{tag}.silu"), e)
        };
        let (s0, st) = (silu_temb(&mut b, "temb0", t0), silu_temb(&mut b, "temb", tf));
        let (m0, mt) = (
            b.linear("mod0", s0, wm, None),
            b.linear("mod", st, wm, None),
        );
        let (m0, mt) = (mods(&mut b, "mod0", m0, d), mods(&mut b, "mod", mt, d));

        for (i, blk) in self.blocks.iter().enumerate() {
            let p = format!("blk{i}");
            let h1 = joint_ln(&mut b, &format!("{p}.n1"), xt, m0.s1, xi, mt.s1);
            let q = blk.q.emit(&mut b, &format!("{p}.q"), h1, None);
            let k = blk.k.emit(&mut b, &format!("{p}.k"), h1, None);
            let v = blk.v.emit(&mut b, &format!("{p}.v"), h1, None);
            let (nq, nk) = (
                b.konst(&format!("{p}.nq"), &[hd], blk.norm_q.clone()),
                b.konst(&format!("{p}.nk"), &[hd], blk.norm_k.clone()),
            );
            let q = b.heads_rms(&format!("{p}.qn"), q, nq, hd, 1e-6);
            let k = b.heads_rms(&format!("{p}.kn"), k, nk, hd, 1e-6);
            let q = b.rope(&format!("{p}.rq"), q, pos, inv_freq, hd, sections, PairLayout::Adjacent);
            let k = b.rope(&format!("{p}.rk"), k, pos, inv_freq, hd, sections, PairLayout::Adjacent);
            // Block-causal: text rows are causal over the text, image rows see everything.
            let qt = b.slice(&format!("{p}.qt"), q, 0, 0, nt);
            let kt = b.slice(&format!("{p}.kt"), k, 0, 0, nt);
            let vt = b.slice(&format!("{p}.vt"), v, 0, 0, nt);
            let qi = b.slice(&format!("{p}.qi"), q, 0, nt, n);
            let at = b.g.add(
                &format!("{p}.at"),
                Op::Attention {
                    heads,
                    kv_heads: heads,
                    causal: true,
                    window: None,
                    soft_cap: None,
                },
                &[qt, kt, vt],
                vec![nt, d],
                DType::F32,
            );
            let ai = b.attention(&format!("{p}.ai"), qi, k, v, heads);
            let a = b.concat(&format!("{p}.a"), &[at, ai], 0);
            let o = blk.o.emit(&mut b, &format!("{p}.o"), a, None);
            let ot = b.slice(&format!("{p}.ot"), o, 0, 0, nt);
            let oi = b.slice(&format!("{p}.oi"), o, 0, nt, n);
            xt = b.gated_add(&format!("{p}.rt1"), xt, ot, m0.g1);
            xi = b.gated_add(&format!("{p}.ri1"), xi, oi, mt.g1);

            let h2 = joint_ln(&mut b, &format!("{p}.n2"), xt, m0.s2, xi, mt.s2);
            let (gate, up) = blk.gate.emit_pair(&blk.up, &mut b, &format!("{p}.gu"), h2);
            let act = b.swiglu(&format!("{p}.swiglu"), gate, up);
            let y = blk.down.emit(&mut b, &format!("{p}.down"), act, None);
            let yt = b.slice(&format!("{p}.yt"), y, 0, 0, nt);
            let yi = b.slice(&format!("{p}.yi"), y, 0, nt, n);
            xt = b.gated_add(&format!("{p}.rt2"), xt, yt, m0.g2);
            xi = b.gated_add(&format!("{p}.ri2"), xi, yi, mt.g2);
        }

        let scale = self.norm_out.emit(&mut b, "norm_out", st, None);
        let y = ln_scale(&mut b, "final", xi, scale);
        let out = self.proj_out.emit(&mut b, "proj_out", y, None);
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
        fn lin(&mut self, n: usize, k: usize) -> QLin {
            self.0 += 1;
            QLin::from_mat(mat(n, k, self.0))
        }
        fn gain(&mut self, n: usize) -> Vec<f32> {
            self.0 += 1;
            rand(n, self.0, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect()
        }
    }

    fn tiny(fused: bool) -> Dit21 {
        let s = &mut Seed(100);
        let blocks = (0..2)
            .map(|_| {
                let (gate, up) = if fused {
                    let f = s.lin(2 * FF, D);
                    (f.rows(0, FF).unwrap(), f.rows(FF, FF).unwrap())
                } else {
                    (s.lin(FF, D), s.lin(FF, D))
                };
                Block {
                    norm_q: s.gain(HD),
                    norm_k: s.gain(HD),
                    q: s.lin(D, D),
                    k: s.lin(D, D),
                    v: s.lin(D, D),
                    o: s.lin(D, D),
                    gate,
                    up,
                    down: s.lin(D, FF),
                }
            })
            .collect();
        Dit21 {
            dim: D,
            heads: D / HD,
            head_dim: HD,
            in_ch: IN,
            axes: [8, 28, 28],
            img_in: s.lin(D, IN),
            text_norm: s.gain(TW),
            txt_in: s.lin(D, TW),
            txt_out: s.lin(D, D),
            t_lin1: s.lin(D, 256),
            t_lin2: s.lin(D, D),
            modulation: s.lin(4 * D, D),
            blocks,
            norm_out: s.lin(D, D),
            proj_out: s.lin(IN, D),
            gpu: Default::default(),
        }
    }

    /// Metal program == native CPU forward (fused and split gate/up, two sigmas), on an odd grid
    /// so the centred (negative) height/width positions are asymmetric.
    #[test]
    fn qwen21_program_matches_cpu_forward() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        for fused in [false, true] {
            let dit = tiny(fused);
            let (h, w, nt) = (5usize, 7usize, 21usize);
            let ni = h * w;
            let lat = rand(ni * IN, 11, 1.0);
            let txt = rand(nt * TW, 12, 1.0);
            let rope = Rope::new(&t2i_positions(nt, h, w), &dit.axes);
            for sigma in [0.8f32, 0.15] {
                let want = {
                    let _cpu = crate::hardware::cpu_compute::allow();
                    let ctx = dit.project_text(&txt, nt);
                    dit.forward(&lat, &ctx, &rope, sigma)
                };
                let got = dit
                    .forward_gpu(&dit.gpu, &lat, &txt, h, w, sigma)
                    .expect("metal program");
                let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
                eprintln!("fused={fused} sigma={sigma}: rel {e:.2e} cos {c:.7}");
                assert!(e < 1e-3 && c > 0.99999, "fused={fused}: rel {e} cos {c}");
            }
        }
    }
}

#[cfg(test)]
mod real_tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, rand, rel_err};

    /// One real denoising forward (a 4x4 token grid, 12 text tokens, sigma 0.7) of the GGUF at
    /// `GIBBS_QWEN21_TRANSFORMER` (default: the Q4_0 finetune under `disk/`), Metal program vs the
    /// native CPU forward.
    #[test]
    #[ignore = "loads a ~4 GB GGUF; the CPU reference is slow"]
    fn real_gguf_step_matches_cpu() {
        let path = std::env::var("GIBBS_QWEN21_TRANSFORMER").unwrap_or_else(|_| {
            "/Users/keas/code/rust-inference-server/disk/models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF/snapshots/main/qwen-image-2.1-UC-Q4_0.gguf".into()
        });
        let dit = Dit21::load(&Store::open(Path::new(&path)).unwrap(), None).unwrap();
        let (h, w, nt) = (4usize, 4usize, 12usize);
        let lat = rand(h * w * dit.in_ch, 21, 1.0);
        let txt = rand(nt * dit.text_width(), 22, 1.0);
        let t = Instant::now();
        let got = dit
            .forward_gpu(&dit.gpu, &lat, &txt, h, w, 0.7)
            .expect("metal program");
        eprintln!("metal (compile + run): {:.2}s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let want = {
            let _cpu = crate::hardware::cpu_compute::allow();
            let ctx = dit.project_text(&txt, nt);
            let rope = Rope::new(&t2i_positions(nt, h, w), &dit.axes);
            dit.forward(&lat, &ctx, &rope, 0.7)
        };
        eprintln!("cpu: {:.2}s", t.elapsed().as_secs_f64());
        let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
        eprintln!(
            "{} blocks, dim {}: rel {e:.3e} cos {c:.7}",
            dit.blocks.len(),
            dit.dim
        );
        assert!(e < 1e-3 && c > 0.99999, "rel {e} cos {c}");
    }
}
