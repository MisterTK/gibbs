//! The Sana transformer forward as one Metal graph program per `(h, w, caption tokens)`. The host
//! supplies only the timestep sinusoid; the caption projection, patch embed, adaLN-single,
//! ReLU linear attention, cross-attention, GLUMBConv and final layer all run in the program.

use super::*;
use crate::graph::{BinaryKind, DType, Op, ReduceKind, Tensor, UnaryKind, ValueId};
use crate::model::dit_graph::{Gb, ProgramCache};

/// `(h, w, caption tokens)`.
pub(super) type Key = (usize, usize, usize);

pub(super) type Cache = ProgramCache<Key>;

fn div(b: &mut Gb, name: &str, x: ValueId, y: ValueId) -> ValueId {
    let (sx, sy) = (b.shape(x), b.shape(y));
    let out = sx.iter().zip(&sy).map(|(&a, &c)| a.max(c)).collect();
    b.g.add(name, Op::Binary { kind: BinaryKind::Div }, &[x, y], out, DType::F32)
}

/// `op(a) · op(b)` over the last two dims, equal batch dims.
fn matmul(b: &mut Gb, name: &str, x: ValueId, y: ValueId, ta: bool, tb: bool) -> ValueId {
    let (sx, sy) = (b.shape(x), b.shape(y));
    let r = sx.len();
    let m = sx[r - if ta { 1 } else { 2 }];
    let n = sy[r - if tb { 2 } else { 1 }];
    let mut out = sx[..r - 2].to_vec();
    out.extend([m, n]);
    b.g.add(
        name,
        Op::MatMul {
            transpose_a: ta,
            transpose_b: tb,
        },
        &[x, y],
        out,
        DType::F32,
    )
}

fn sum_keep(b: &mut Gb, name: &str, x: ValueId, axis: usize) -> ValueId {
    let mut s = b.shape(x);
    s[axis] = 1;
    b.g.add(
        name,
        Op::Reduce {
            kind: ReduceKind::Sum,
            axes: vec![axis as isize],
            keep_dims: true,
        },
        &[x],
        s,
        DType::F32,
    )
}

/// `[n, heads * hd] -> [heads, n, hd]`.
fn to_heads(b: &mut Gb, name: &str, x: ValueId, n: usize, heads: usize, hd: usize) -> ValueId {
    let r = b.reshape(&format!("{name}.split"), x, &[n, heads, hd]);
    b.transpose(name, r, &[1, 0, 2])
}

/// ReLU linear attention (see [`linear_attention`]): `out_i = (sum_j v_j relu(k_j)^T) relu(q_i) /
/// (sum_j relu(k_j) . relu(q_i) + 1e-15)` per head.
fn linear_attn(b: &mut Gb, p: &str, q: ValueId, k: ValueId, v: ValueId, n: usize, heads: usize, hd: usize) -> ValueId {
    let q = b.unary(&format!("{p}.qr"), UnaryKind::Relu, q);
    let k = b.unary(&format!("{p}.kr"), UnaryKind::Relu, k);
    let qh = to_heads(b, &format!("{p}.qh"), q, n, heads, hd);
    let kh = to_heads(b, &format!("{p}.kh"), k, n, heads, hd);
    let vh = to_heads(b, &format!("{p}.vh"), v, n, heads, hd);
    // kv[h, c, d] = sum_j v[j, c] k[j, d].
    let kv = matmul(b, &format!("{p}.kv"), vh, kh, true, false);
    // num[h, i, c] = sum_d q[i, d] kv[c, d].
    let num = matmul(b, &format!("{p}.num"), qh, kv, false, true);
    let ks = sum_keep(b, &format!("{p}.ks"), kh, 1);
    let qk = b.mul(&format!("{p}.qk"), qh, ks);
    let den = sum_keep(b, &format!("{p}.den0"), qk, 2);
    let eps = b.scalar(1e-15);
    let den = b.add(&format!("{p}.den"), den, eps);
    let o = div(b, &format!("{p}.out"), num, den);
    let o = b.transpose(&format!("{p}.merge"), o, &[1, 0, 2]);
    b.reshape(&format!("{p}.flat"), o, &[n, heads * hd])
}

/// Depthwise 3x3 (pad 1) over token rows `x [h*w, c]` (weight `[c, 9]`) as a Conv2d on `[c, h, w]`.
fn depthwise(b: &mut Gb, p: &str, x: ValueId, h: usize, w: usize, wt: &[f32], bias: &[f32]) -> ValueId {
    let c = b.shape(x)[1];
    let img = b.transpose(&format!("{p}.dw.in"), x, &[1, 0]);
    let img = b.reshape(&format!("{p}.dw.img"), img, &[c, h, w]);
    let dw_w = b.konst(&format!("{p}.dw.weight"), &[c, 1, 3, 3], wt.to_vec());
    let dw_b = b.konst(&format!("{p}.dw.bias"), &[c], bias.to_vec());
    let img = b.g.add(
        &format!("{p}.dw"),
        Op::Conv2d {
            stride: [1, 1],
            padding: [1, 1, 1, 1],
            dilation: [1, 1],
            groups: c,
        },
        &[img, dw_w, dw_b],
        vec![c, h, w],
        DType::F32,
    );
    let flat = b.reshape(&format!("{p}.dw.flat"), img, &[c, h * w]);
    b.transpose(&format!("{p}.dw.out"), flat, &[1, 0])
}

impl SanaDit {
    /// Flow prediction for latents `lat [C, h, w]` at integer `timestep` given the raw encoder
    /// states `enc [n_cap, caption_channels]` (the program applies the caption projection).
    pub fn forward_gpu(
        &self,
        cache: &Cache,
        lat: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        enc: &[f32],
        n_cap: usize,
    ) -> Result<Vec<f32>, String> {
        let inputs = vec![
            Tensor::F32(lat.to_vec()),
            Tensor::F32(enc.to_vec()),
            Tensor::F32(timestep_features(timestep * self.timestep_scale, 256)),
        ];
        let mut out = cache.run((h, w, n_cap), || self.build(h, w, n_cap), inputs)?;
        Ok(out.remove(0))
    }

    fn build(&self, h: usize, w: usize, n_cap: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let (d, c, n) = (self.d, self.in_ch, h * w);
        let mut b = Gb::new();
        let lat = b.input("lat", &[c, n]);
        let enc = b.input("enc", &[n_cap, self.cap1.in_dim()]);
        let tfeat = b.input("tfeat", &[1, 256]);

        // Caption: linear_1 -> GELU(tanh) -> linear_2 -> RMSNorm.
        let cap = self.cap1.emit(&mut b, "cap1", enc);
        let cap = b.gelu_tanh("cap1.act", cap);
        let cap = self.cap2.emit(&mut b, "cap2", cap);
        let cap_norm = b.konst("cap_norm", &[d], self.cap_norm.clone());
        let cap = b.rms_norm("cap_norm.out", cap, cap_norm, 1e-5);

        // Patch embed (patch size 1) and adaLN-single embedding.
        let xt = b.transpose("lat.t", lat, &[1, 0]);
        let mut x = self.patch_proj.emit(&mut b, "patch", xt);
        let e = self.t_lin1.emit(&mut b, "t1", tfeat);
        let e = b.silu("t1.act", e);
        let emb = self.t_lin2.emit(&mut b, "t2", e);
        let e = b.silu("t2.act", emb);
        let tm = self.t_mod.emit(&mut b, "tmod", e);

        let (heads, hd) = (self.heads, self.head_dim);
        for (i, blk) in self.blocks.iter().enumerate() {
            let p = format!("b{i}");
            let sst = b.konst(&format!("{p}.sst"), &[1, 6 * d], blk.sst.clone());
            let m = b.add(&format!("{p}.mod"), sst, tm);
            let ch: Vec<ValueId> = (0..6)
                .map(|k| b.slice(&format!("{p}.mod{k}"), m, 1, k * d, (k + 1) * d))
                .collect();
            // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            let nx = b.ln_mod(&format!("{p}.n1"), x, ch[0], ch[1], 1e-6);
            let q = blk.q.emit(&mut b, &format!("{p}.q"), nx);
            let k = blk.k.emit(&mut b, &format!("{p}.k"), nx);
            let v = blk.v.emit(&mut b, &format!("{p}.v"), nx);
            let a = linear_attn(&mut b, &format!("{p}.la"), q, k, v, n, heads, hd);
            let a = blk.o.emit(&mut b, &format!("{p}.o"), a);
            x = b.gated_add(&format!("{p}.x1"), x, a, ch[2]);

            let cq = blk.cq.emit(&mut b, &format!("{p}.cq"), x);
            let ck = blk.ck.emit(&mut b, &format!("{p}.ck"), cap);
            let cv = blk.cv.emit(&mut b, &format!("{p}.cv"), cap);
            let a = b.attention(&format!("{p}.xattn"), cq, ck, cv, self.cross_heads);
            let a = blk.co.emit(&mut b, &format!("{p}.co"), a);
            x = b.add(&format!("{p}.x2"), x, a);

            // GLUMBConv: 1x1 -> SiLU -> depthwise 3x3 -> (a * silu(gate)) -> 1x1.
            let nx = b.ln_mod(&format!("{p}.n2"), x, ch[3], ch[4], 1e-6);
            let t = blk.inverted.emit(&mut b, &format!("{p}.inv"), nx);
            let t = b.silu(&format!("{p}.inv.act"), t);
            let c2 = blk.inverted.out_dim();
            let t = depthwise(&mut b, &p, t, h, w, &blk.depth_w, &blk.depth_b);
            let half = c2 / 2;
            let a = b.slice(&format!("{p}.glu.a"), t, 1, 0, half);
            let gate = b.slice(&format!("{p}.glu.g"), t, 1, half, c2);
            let g = b.swiglu(&format!("{p}.glu"), gate, a);
            let y = blk.point.emit(&mut b, &format!("{p}.point"), g);
            x = b.gated_add(&format!("{p}.x3"), x, y, ch[5]);
        }

        let sst = b.konst("sst_out", &[2, d], self.sst_out.clone());
        let shift_t = b.slice("sst_out.shift", sst, 0, 0, 1);
        let scale_t = b.slice("sst_out.scale", sst, 0, 1, 2);
        let shift = b.add("final.shift", shift_t, emb);
        let scale = b.add("final.scale", scale_t, emb);
        let y = b.ln_mod("final.n", x, shift, scale, 1e-6);
        let y = self.proj_out.emit(&mut b, "proj_out", y);
        let out = b.transpose("out", y, &[1, 0]);
        Ok((b, vec![out]))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, rand, rel_err};

    const HEADS: usize = 2;
    const HD: usize = 16;
    const D: usize = HEADS * HD;
    const C: usize = 8;
    const CAP_IN: usize = 24;

    fn lin(n: usize, k: usize, seed: u64, bias: bool) -> Lin {
        let mut specs = vec![("p.weight".to_string(), vec![n, k])];
        if bias {
            specs.push(("p.bias".to_string(), vec![n]));
        }
        let wd = rand(n * k, seed, 1.0 / (k as f32).sqrt());
        let bd = rand(n, seed + 77, 0.1);
        let w = Weights::from_f32(specs, |name| {
            Ok(if name == "p.weight" { wd.clone() } else { bd.clone() })
        })
        .expect("anonymous map");
        Lin::load(&w, "p").expect("tiny lin")
    }

    fn near_one(n: usize, seed: u64) -> Vec<f32> {
        rand(n, seed, 0.2).iter().map(|v| 1.0 + v).collect()
    }

    fn tiny() -> SanaDit {
        let c2 = 2 * D;
        let blocks = (0..2u64)
            .map(|i| {
                let s = 100 * (i + 1);
                Block {
                    sst: rand(6 * D, s, 0.5),
                    q: lin(D, D, s + 1, true),
                    k: lin(D, D, s + 2, true),
                    v: lin(D, D, s + 3, true),
                    o: lin(D, D, s + 4, true),
                    cq: lin(D, D, s + 5, true),
                    ck: lin(D, D, s + 6, true),
                    cv: lin(D, D, s + 7, true),
                    co: lin(D, D, s + 8, true),
                    inverted: lin(c2, D, s + 9, true),
                    depth_w: rand(c2 * 9, s + 10, 0.4),
                    depth_b: rand(c2, s + 11, 0.1),
                    point: lin(D, D, s + 12, true),
                }
            })
            .collect();
        SanaDit {
            d: D,
            heads: HEADS,
            head_dim: HD,
            cross_heads: 4,
            in_ch: C,
            patch_proj: lin(D, C, 1, true),
            t_lin1: lin(D, 256, 2, true),
            t_lin2: lin(D, D, 3, true),
            t_mod: lin(6 * D, D, 4, true),
            cap1: lin(D, CAP_IN, 5, true),
            cap2: lin(D, D, 6, true),
            cap_norm: near_one(D, 7),
            blocks,
            sst_out: rand(2 * D, 8, 0.5),
            proj_out: lin(C, D, 9, true),
            timestep_scale: 1.0,
            sample_size: 8,
            gpu: Default::default(),
        }
    }

    fn run1(b: Gb, out: ValueId, inputs: Vec<Tensor>) -> Vec<f32> {
        crate::model::dit_graph::Program::compile(b, &[out])
            .expect("compile")
            .run(inputs)
            .expect("run")
            .remove(0)
    }

    /// Linear attention at Sana-600M size (36 heads x 32, 256 tokens) against the host
    /// `linear_attention`: head 0's first queries are all <= 0 (denominator exactly the 1e-15 eps,
    /// output must be 0), head 1's q/k are ~1e-9 so the eps shifts the denominator.
    #[test]
    fn linear_attention_matches_cpu_at_real_size_and_eps() {
        let (heads, hd, n) = (36usize, 32usize, 256usize);
        let d = heads * hd;
        let (mut q, mut k, v) = (rand(n * d, 1, 1.0), rand(n * d, 2, 1.0), rand(n * d, 3, 1.0));
        for i in 0..n {
            for j in 0..hd {
                if i < 4 {
                    q[i * d + j] = -q[i * d + j].abs();
                }
                q[i * d + hd + j] *= 1e-9;
                k[i * d + hd + j] *= 1e-9;
            }
        }
        let want = {
            let _cpu = crate::hardware::cpu_compute::allow();
            linear_attention(&q, &k, &v, n, heads, hd)
        };
        let mut b = Gb::new();
        let (qi, ki, vi) = (b.input("q", &[n, d]), b.input("k", &[n, d]), b.input("v", &[n, d]));
        let o = linear_attn(&mut b, "la", qi, ki, vi, n, heads, hd);
        let got = run1(b, o, vec![Tensor::F32(q), Tensor::F32(k), Tensor::F32(v)]);
        assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
        assert!(got[..hd].iter().all(|&v| v == 0.0), "eps rows must be exactly 0");
        let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
        eprintln!("linear attention 36x32 n=256: rel {e:.2e} cos {c:.7}");
        assert!(e < 1e-3 && c > 0.99999, "rel {e} cos {c}");
    }

    /// Cross-attention at Sana-600M size (16 heads x 72, 300 caption keys) against `nn::attention`.
    #[test]
    fn cross_attention_matches_cpu_at_real_size() {
        let (heads, hd) = (16usize, 72usize);
        let d = heads * hd;
        for (nq, nk) in [(256usize, 300usize), (1024, 300), (64, 2100)] {
            let (q, k, v) = (rand(nq * d, 7, 1.0), rand(nk * d, 8, 1.0), rand(nk * d, 9, 1.0));
            let want = {
                let _cpu = crate::hardware::cpu_compute::allow();
                nn::attention(&q, &k, &v, nq, nk, heads, hd, false)
            };
            let mut b = Gb::new();
            let (qi, ki, vi) = (b.input("q", &[nq, d]), b.input("k", &[nk, d]), b.input("v", &[nk, d]));
            let o = b.attention("xa", qi, ki, vi, heads);
            let got = run1(b, o, vec![Tensor::F32(q), Tensor::F32(k), Tensor::F32(v)]);
            let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("cross attention 16x72 nq={nq} nk={nk}: rel {e:.2e} cos {c:.7}");
            assert!(e < 1e-3 && c > 0.99999, "nq={nq} nk={nk}: rel {e} cos {c}");
        }
    }

    /// Depthwise 3x3 Conv2d at Sana-600M width (5760 channels) on 16x16 and an odd 5x7 grid.
    #[test]
    fn depthwise_matches_cpu_at_real_width() {
        let c = 5760usize;
        for (h, w) in [(16usize, 16usize), (5, 7)] {
            let n = h * w;
            let (x, wt, bias) = (rand(n * c, 4, 1.0), rand(c * 9, 5, 0.4), rand(c, 6, 0.1));
            let want = {
                let _cpu = crate::hardware::cpu_compute::allow();
                depthwise_tokens(&x, c, h, w, &wt, &bias)
            };
            let mut b = Gb::new();
            let xi = b.input("x", &[n, c]);
            let o = depthwise(&mut b, "t", xi, h, w, &wt, &bias);
            let got = run1(b, o, vec![Tensor::F32(x)]);
            let (e, cs) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("depthwise c={c} {h}x{w}: rel {e:.2e} cos {cs:.7}");
            assert!(e < 1e-4 && cs > 0.99999, "{h}x{w}: rel {e} cos {cs}");
        }
    }

    /// Metal program == native CPU forward, at two grids (24 and 35 tokens: non-square, odd).
    #[test]
    fn sana_program_matches_cpu_forward() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        let tr = tiny();
        for (h, w, n_cap, t) in [(4usize, 6usize, 7usize, 731.0f32), (5, 7, 21, 40.0)] {
            let lat = rand(C * h * w, 11, 1.0);
            let enc = rand(n_cap * CAP_IN, 12, 1.0);
            let (want, got) = {
                let want = {
                    let _cpu = crate::hardware::cpu_compute::allow();
                    let cap = tr.caption(&enc, n_cap);
                    tr.forward(&lat, h, w, t, &cap, n_cap)
                };
                let got = tr
                    .forward_gpu(&tr.gpu, &lat, h, w, t, &enc, n_cap)
                    .expect("metal program");
                (want, got)
            };
            let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
            eprintln!("sana {h}x{w} cap {n_cap}: rel {e:.2e} cos {c:.7}");
            assert!(e < 1e-3 && c > 0.99999, "{h}x{w}: rel {e} cos {c}");
        }
    }
}
