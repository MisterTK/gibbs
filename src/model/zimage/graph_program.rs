//! The Z-Image S3-DiT forward as one Metal graph program per `(image tokens, caption tokens)`.
//! The host supplies only scalar arithmetic: the timestep sinusoid, patchify/unpatchify index
//! shuffles and the three-axis RoPE position ids. Every matmul, norm, modulation, RoPE, attention
//! and activation (embedders, refiners, joint blocks, final layer) runs in the program; weights
//! stay in their stored encoding.

use super::*;
use crate::graph::{PairLayout, Tensor, UnaryKind, ValueId};
use crate::model::dit_graph::{self as dg, Gb, ProgramCache};

/// `(patch tokens, caption tokens)` before padding.
pub(super) type Key = (usize, usize);

pub(super) type Cache = ProgramCache<Key>;

/// RoPE frequency table and section sizes shared by every block.
struct Rp {
    inv_freq: ValueId,
    sections: [u32; 3],
}

fn flat(pos: &[[usize; 3]]) -> Vec<u32> {
    pos.iter().flat_map(|p| p.map(|v| v as u32)).collect()
}

impl ZDit {
    /// The model output (as [`Self::forward`]) for latents [C, h, w] at noise level `sigma`
    /// given the raw caption features `cap [n_cap, cap_feat_dim]`, on the GPU.
    pub(super) fn forward_gpu(
        &self,
        lat: &[f32],
        h: usize,
        w: usize,
        sigma: f32,
        cap: &[f32],
        n_cap: usize,
    ) -> Result<Vec<f32>, String> {
        let (c, wp) = (self.in_ch, w / PATCH);
        let n_img = (h / PATCH) * wp;
        let (img_total, cap_total) = (pad_to(n_img), pad_to(n_cap));
        let (img_pos, cap_pos) = token_pos(wp, n_img, img_total, cap_total);
        let inputs = vec![
            Tensor::F32(patchify(lat, c, h, w)),
            Tensor::F32(cap.to_vec()),
            Tensor::F32(timestep_features((1.0 - sigma) * self.t_scale, 256)),
            Tensor::U32(flat(&img_pos)),
            Tensor::U32(flat(&cap_pos)),
            Tensor::U32(flat(&[img_pos, cap_pos].concat())),
        ];
        let mut out = self
            .gpu
            .run((n_img, n_cap), || self.build(n_img, n_cap), inputs)?;
        Ok(unpatchify(&out.remove(0), c, h, w))
    }

    /// One S3-DiT block over `x [n, dim]`; `adaln [1, a]` feeds the tanh-gated modulation of
    /// modulated blocks, the context refiner has none.
    #[allow(clippy::too_many_arguments)]
    fn emit_block(
        &self,
        b: &mut Gb,
        blk: &Block,
        p: &str,
        x: ValueId,
        pos: ValueId,
        rp: &Rp,
        adaln: ValueId,
    ) -> ValueId {
        let (d, hd, eps) = (self.dim, self.head_dim, self.eps);
        let one = b.scalar(1.0);
        // (1 + scale_msa, tanh gate_msa, 1 + scale_mlp, tanh gate_mlp).
        let m = blk.ada.as_ref().map(|l| {
            let m = l.emit(b, &format!("{p}.ada"), adaln);
            let mut chunk = |i: usize| {
                let s = b.slice(&format!("{p}.m{i}"), m, 1, i * d, (i + 1) * d);
                if i % 2 == 1 {
                    b.unary(&format!("{p}.m{i}.tanh"), UnaryKind::Tanh, s)
                } else {
                    b.add(&format!("{p}.m{i}.1p"), s, one)
                }
            };
            [chunk(0), chunk(1), chunk(2), chunk(3)]
        });
        let gamma = |b: &mut Gb, n: &str, v: &[f32]| b.konst(&format!("{p}.{n}"), &[v.len()], v.to_vec());
        let scaled = |b: &mut Gb, n: &str, h: ValueId, s: Option<ValueId>| match s {
            Some(s) => b.mul(&format!("{p}.{n}"), h, s),
            None => h,
        };
        let resid = |b: &mut Gb, n: &str, x: ValueId, y: ValueId, g: Option<ValueId>| match g {
            Some(g) => b.gated_add(&format!("{p}.{n}"), x, y, g),
            None => b.add(&format!("{p}.{n}"), x, y),
        };

        let g = gamma(b, "attn_norm1", &blk.attn_norm1);
        let h = b.rms_norm(&format!("{p}.attn_in"), x, g, eps);
        let h = scaled(b, "attn_mod", h, m.map(|m| m[0]));
        let q = blk.q.emit(b, &format!("{p}.q"), h);
        let k = blk.k.emit(b, &format!("{p}.k"), h);
        let v = blk.v.emit(b, &format!("{p}.v"), h);
        let (qn, kn) = (gamma(b, "norm_q", &blk.norm_q), gamma(b, "norm_k", &blk.norm_k));
        let q = b.heads_rms(&format!("{p}.qnorm"), q, qn, hd, eps);
        let k = b.heads_rms(&format!("{p}.knorm"), k, kn, hd, eps);
        let q = b.rope(&format!("{p}.rq"), q, pos, rp.inv_freq, hd, rp.sections, PairLayout::Adjacent);
        let k = b.rope(&format!("{p}.rk"), k, pos, rp.inv_freq, hd, rp.sections, PairLayout::Adjacent);
        let a = b.attention(&format!("{p}.attn"), q, k, v, self.heads);
        let a = blk.o.emit(b, &format!("{p}.o"), a);
        let g = gamma(b, "attn_norm2", &blk.attn_norm2);
        let a = b.rms_norm(&format!("{p}.attn_out"), a, g, eps);
        let x = resid(b, "res1", x, a, m.map(|m| m[1]));

        let g = gamma(b, "ffn_norm1", &blk.ffn_norm1);
        let h = b.rms_norm(&format!("{p}.ffn_in"), x, g, eps);
        let h = scaled(b, "ffn_mod", h, m.map(|m| m[2]));
        let gt = blk.w1.emit(b, &format!("{p}.w1"), h);
        let up = blk.w3.emit(b, &format!("{p}.w3"), h);
        let act = b.swiglu(&format!("{p}.swiglu"), gt, up);
        let f = blk.w2.emit(b, &format!("{p}.w2"), act);
        let g = gamma(b, "ffn_norm2", &blk.ffn_norm2);
        let f = b.rms_norm(&format!("{p}.ffn_out"), f, g, eps);
        resid(b, "res2", x, f, m.map(|m| m[3]))
    }

    /// `x` with `total - n` copies of the learned pad token row appended.
    fn emit_pad(b: &mut Gb, name: &str, x: ValueId, pad: &[f32], n: usize, total: usize) -> ValueId {
        if total == n {
            return x;
        }
        let rows = pad.repeat(total - n);
        let pad = b.konst(&format!("{name}.rows"), &[total - n, pad.len()], rows);
        b.concat(name, &[x, pad], 0)
    }

    /// Emits the whole forward; inputs in order `patches, cap, t_feat, pos_img, pos_cap, pos_all`.
    fn build(&self, n_img: usize, n_cap: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let (img_total, cap_total) = (pad_to(n_img), pad_to(n_cap));
        let n = img_total + cap_total;
        let mut b = Gb::new();
        let patches = b.input("patches", &[n_img, PATCH * PATCH * self.in_ch]);
        let cap = b.input("cap", &[n_cap, self.cap_norm.len()]);
        let tf = b.input("t_feat", &[1, 256]);
        let pos_img = b.input_u32("pos_img", &[img_total * 3]);
        let pos_cap = b.input_u32("pos_cap", &[cap_total * 3]);
        let pos_all = b.input_u32("pos_all", &[n * 3]);
        let (freq, sections) = dg::axis_freqs(&self.axes, self.theta);
        let inv_freq = b.konst("rope.inv_freq", &[freq.len()], freq);
        let rp = Rp { inv_freq, sections };

        // Timestep conditioning [1, adaln]: Linear -> SiLU -> Linear.
        let t = self.t_mlp0.emit(&mut b, "t_mlp0", tf);
        let t = b.silu("t_silu", t);
        let adaln = self.t_mlp2.emit(&mut b, "t_mlp2", t);

        let x = self.x_embedder.emit(&mut b, "x_embedder", patches);
        let mut x = Self::emit_pad(&mut b, "x_pad", x, &self.x_pad, n_img, img_total);
        for (i, blk) in self.noise_refiner.iter().enumerate() {
            x = self.emit_block(&mut b, blk, &format!("nr.{i}"), x, pos_img, &rp, adaln);
        }

        let cn_gamma = b.konst("cap_norm", &[self.cap_norm.len()], self.cap_norm.clone());
        let cn = b.rms_norm("cap_normed", cap, cn_gamma, self.eps);
        let cx = self.cap_lin.emit(&mut b, "cap_lin", cn);
        let mut cx = Self::emit_pad(&mut b, "cap_pad", cx, &self.cap_pad, n_cap, cap_total);
        for (i, blk) in self.context_refiner.iter().enumerate() {
            cx = self.emit_block(&mut b, blk, &format!("cr.{i}"), cx, pos_cap, &rp, adaln);
        }

        // Unified sequence: image tokens first, then caption tokens.
        let mut u = b.concat("unified", &[x, cx], 0);
        for (i, blk) in self.layers.iter().enumerate() {
            u = self.emit_block(&mut b, blk, &format!("layer.{i}"), u, pos_all, &rp, adaln);
        }

        let sa = b.silu("final.silu", adaln);
        let scale = self.final_ada.emit(&mut b, "final_ada", sa);
        let one = b.scalar(1.0);
        let s1 = b.add("final.1p", scale, one);
        let xi = b.slice("final.img", u, 0, 0, n_img);
        let h = b.layer_norm("final.ln", xi, 1e-6);
        let h = b.mul("final.mod", h, s1);
        let out = self.final_lin.emit(&mut b, "final_lin", h);
        Ok((b, vec![out]))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, rand, rel_err};
    use crate::model::weights::Weights;
    use std::collections::HashMap;

    const DIM: usize = 128;
    const HEADS: usize = 4;
    const HD: usize = DIM / HEADS;
    const AXES: [usize; 3] = [8, 12, 12];
    const CAP: usize = 48;
    const IN_CH: usize = 4;
    const ADA: usize = 64;
    const FF: usize = 192;

    /// A random-weight Z-Image (2 noise refiners, 2 context refiners, 2 joint layers) built
    /// through the real `Lin`/`Block` loaders over an anonymous bf16 weight map.
    fn tiny(seed: u64) -> ZDit {
        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
        let mut lin = |p: &str, out: usize, inp: usize, bias: bool| {
            specs.push((format!("{p}.weight"), vec![out, inp]));
            if bias {
                specs.push((format!("{p}.bias"), vec![out]));
            }
        };
        let fams = [("noise_refiner", true), ("context_refiner", false), ("layers", true)];
        for (fam, modulated) in fams {
            for i in 0..2 {
                let p = format!("{fam}.{i}");
                if modulated {
                    lin(&format!("{p}.adaLN_modulation.0"), 4 * DIM, ADA, true);
                }
                for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                    lin(&format!("{p}.attention.{n}"), DIM, DIM, false);
                }
                lin(&format!("{p}.feed_forward.w1"), FF, DIM, false);
                lin(&format!("{p}.feed_forward.w3"), FF, DIM, false);
                lin(&format!("{p}.feed_forward.w2"), DIM, FF, false);
            }
        }
        lin("t_embedder.mlp.0", 96, 256, true);
        lin("t_embedder.mlp.2", ADA, 96, true);
        lin("all_x_embedder.2-1", DIM, PATCH * PATCH * IN_CH, true);
        lin("cap_embedder.1", DIM, CAP, true);
        lin("all_final_layer.2-1.adaLN_modulation.1", DIM, ADA, true);
        lin("all_final_layer.2-1.linear", PATCH * PATCH * IN_CH, DIM, true);
        for (n, len) in [("x_pad_token", DIM), ("cap_pad_token", DIM), ("cap_embedder.0.weight", CAP)] {
            specs.push((n.into(), vec![len]));
        }
        for (fam, _) in fams {
            for i in 0..2 {
                for (n, len) in [
                    ("attention.norm_q", HD),
                    ("attention.norm_k", HD),
                    ("attention_norm1", DIM),
                    ("attention_norm2", DIM),
                    ("ffn_norm1", DIM),
                    ("ffn_norm2", DIM),
                ] {
                    specs.push((format!("{fam}.{i}.{n}.weight"), vec![len]));
                }
            }
        }
        let shapes: HashMap<String, Vec<usize>> = specs.iter().cloned().collect();
        let w = Weights::from_f32(specs, |name| {
            let s = &shapes[name];
            let h = name.bytes().fold(seed, |h, c| h.wrapping_mul(31).wrapping_add(c as u64));
            let n = s.iter().product();
            Ok(if name.ends_with(".bias") {
                rand(n, h, 0.1)
            } else if s.len() == 2 {
                rand(n, h, 1.0 / (s[1] as f32).sqrt())
            } else if name.contains("pad") {
                rand(n, h, 1.0)
            } else {
                rand(n, h, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect()
            })
        })
        .expect("tiny weights");
        let blocks = |fam: &str, m: bool| {
            (0..2)
                .map(|i| Block::load(&w, &format!("{fam}.{i}"), DIM, HD, m).expect("block"))
                .collect::<Vec<_>>()
        };
        let l = |p: &str| Lin::load(&w, p).expect("lin");
        ZDit {
            dim: DIM,
            heads: HEADS,
            head_dim: HD,
            in_ch: IN_CH,
            eps: 1e-5,
            t_scale: 1000.0,
            theta: 256.0,
            axes: AXES.to_vec(),
            t_mlp0: l("t_embedder.mlp.0"),
            t_mlp2: l("t_embedder.mlp.2"),
            x_embedder: l("all_x_embedder.2-1"),
            x_pad: w.vec_n("x_pad_token", DIM).unwrap(),
            cap_norm: w.vec_n("cap_embedder.0.weight", CAP).unwrap(),
            cap_lin: l("cap_embedder.1"),
            cap_pad: w.vec_n("cap_pad_token", DIM).unwrap(),
            noise_refiner: blocks("noise_refiner", true),
            context_refiner: blocks("context_refiner", false),
            layers: blocks("layers", true),
            final_ada: l("all_final_layer.2-1.adaLN_modulation.1"),
            final_lin: l("all_final_layer.2-1.linear"),
            gpu: Default::default(),
        }
    }

    /// Metal program == native CPU forward: with caption/image padding (21 caption tokens -> 32,
    /// 6x6 = 36 patches -> 64) and without (32 caption tokens, 8x8 = 64 patches), 96+ unified tokens.
    #[test]
    fn zimage_program_matches_cpu_forward() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        let tr = tiny(7);
        for (h, w, n_cap, sigma) in [(12usize, 12usize, 21usize, 0.7f32), (16, 16, 32, 0.25)] {
            let lat = rand(IN_CH * h * w, 11, 1.0);
            let cap = rand(n_cap * CAP, 12, 1.0);
            let want = {
                let _cpu = crate::hardware::cpu_compute::allow();
                tr.forward(&lat, h, w, sigma, &cap, n_cap)
            };
            let got = tr.forward_gpu(&lat, h, w, sigma, &cap, n_cap).expect("metal program");
            let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("zimage {h}x{w} cap {n_cap} sigma {sigma}: rel {e:.2e} cos {c:.7}");
            assert!(e < 1e-3 && c > 0.99999, "rel {e} cos {c}");
        }
    }
}
