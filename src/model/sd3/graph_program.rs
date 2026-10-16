//! The SD3 / SD3.5 MMDiT forward as one Metal graph program per `(hp, wp, n_ctx)`. The host supplies
//! only data movement and scalars: patchify/unpatchify, the cropped positional-embedding row ids and
//! the timestep sinusoid. The patch projection, the positional-embedding add, both embedder MLPs,
//! `context_embedder`, every block (adaLN-zero, joint attention, SD3.5 dual attention, QK RMSNorm,
//! the context-pre-only last block) and the final adaLN + `proj_out` run in the program.

use super::*;
use crate::graph::{DType, Op, Tensor, ValueId};
use crate::model::dit_graph::{Gb, ProgramCache};

/// `(hp, wp, n_ctx)`.
pub(super) type Key = (usize, usize, usize);

pub(super) type Cache = ProgramCache<Key>;

impl Lin {
    fn emit(&self, b: &mut Gb, name: &str, x: ValueId) -> ValueId {
        b.lin_mat(name, x, &self.w, self.b.as_deref())
    }
}

impl Proj {
    /// `x -> (q, k, v)` with the optional per-head QK RMSNorm on q and k.
    fn emit_qkv(&self, b: &mut Gb, p: &str, x: ValueId) -> (ValueId, ValueId, ValueId) {
        let qk = |b: &mut Gb, tag: &str, l: &Lin, g: &Option<Vec<f32>>| {
            let y = l.emit(b, &format!("{p}.{tag}"), x);
            match g {
                Some(g) => {
                    let gv = b.konst(&format!("{p}.{tag}n"), &[HEAD_DIM], g.clone());
                    b.heads_rms(&format!("{p}.{tag}norm"), y, gv, HEAD_DIM, EPS)
                }
                None => y,
            }
        };
        let q = qk(b, "q", &self.q, &self.norm_q);
        let k = qk(b, "k", &self.k, &self.norm_k);
        (q, k, self.v.emit(b, &format!("{p}.v"), x))
    }

    /// The output projection, or the identity when the stream has none.
    fn emit_out(&self, b: &mut Gb, p: &str, a: ValueId) -> ValueId {
        match &self.out {
            Some(l) => l.emit(b, &format!("{p}.out"), a),
            None => a,
        }
    }
}

impl Ff {
    fn emit(&self, b: &mut Gb, p: &str, x: ValueId) -> ValueId {
        let h = self.up.emit(b, &format!("{p}.up"), x);
        let h = b.gelu_tanh(&format!("{p}.gelu"), h);
        self.down.emit(b, &format!("{p}.down"), h)
    }
}

impl Mmdit {
    /// The velocity for latents `[C, h, w]` (see [`Mmdit::forward`]) on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_gpu(
        &self,
        lat: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        ctx: &[f32],
        n_ctx: usize,
        pooled: &[f32],
    ) -> Result<Vec<f32>, String> {
        let (p, c) = (self.patch, self.in_ch);
        let (hp, wp) = (h / p, w / p);
        // Row of the `[pos_max^2, d]` table feeding each patch: the centred crop.
        let (top, left) = ((self.pos_max - hp) / 2, (self.pos_max - wp) / 2);
        let rows: Vec<u32> = (0..hp)
            .flat_map(|i| (0..wp).map(move |j| ((top + i) * self.pos_max + left + j) as u32))
            .collect();
        let inputs = vec![
            Tensor::F32(patchify(lat, c, h, w, p)),
            Tensor::U32(rows),
            Tensor::F32(timestep_features(timestep, 256)),
            Tensor::F32(pooled.to_vec()),
            Tensor::F32(ctx.to_vec()),
        ];
        let mut out = self
            .gpu
            .run((hp, wp, n_ctx), || self.build(hp, wp, n_ctx), inputs)?;
        Ok(unpatchify(&out.remove(0), c, h, w, p))
    }

    /// Emits the whole forward; inputs in order `patches, pos_rows, t_feat, pooled, ctx`.
    fn build(&self, hp: usize, wp: usize, n_ctx: usize) -> Result<(Gb, Vec<ValueId>), String> {
        let (d, n) = (self.d, hp * wp);
        let mut b = Gb::new();
        let patches = b.input("patches", &[n, self.patch_proj.in_dim()]);
        let rows = b.input_u32("pos_rows", &[n]);
        let t_feat = b.input("t_feat", &[1, 256]);
        let pooled = b.input("pooled", &[1, self.text_lin1.in_dim()]);
        let ctx = b.input("ctx", &[n_ctx, self.context_embedder.in_dim()]);

        // Patch projection + cropped positional embedding.
        let x = self.patch_proj.emit(&mut b, "pos_embed.proj", patches);
        let table = b.mat("pos_embed.table", &self.pos_embed);
        let pe = b.g.add(
            "pos_embed.rows",
            Op::Embedding,
            &[table, rows],
            vec![n, d],
            DType::F32,
        );
        let mut x = b.add("pos_embed.add", x, pe);

        // Conditioning: SiLU(timestep MLP + pooled-text MLP).
        let t = self.t_lin1.emit(&mut b, "time.l1", t_feat);
        let t = b.silu("time.silu", t);
        let t = self.t_lin2.emit(&mut b, "time.l2", t);
        let y = self.text_lin1.emit(&mut b, "text.l1", pooled);
        let y = b.silu("text.silu", y);
        let y = self.text_lin2.emit(&mut b, "text.l2", y);
        let temb = b.add("temb", t, y);
        let st = b.silu("temb.silu", temb);
        let mut cx = self.context_embedder.emit(&mut b, "context_embedder", ctx);

        for (i, blk) in self.blocks.iter().enumerate() {
            (x, cx) = blk.emit(
                &mut b,
                &format!("blk{i}"),
                st,
                x,
                cx,
                (n, n_ctx),
                (d, self.heads),
            );
        }

        // AdaLayerNormContinuous: chunks (scale, shift).
        let e = self.norm_out.emit(&mut b, "norm_out", st);
        let scale = b.slice("norm_out.scale", e, 1, 0, d);
        let shift = b.slice("norm_out.shift", e, 1, d, 2 * d);
        let ln = b.layer_norm("norm_out.ln", x, EPS);
        let xn = b.modulate("norm_out.mod", ln, shift, scale);
        let out = self.proj_out.emit(&mut b, "proj_out", xn);
        Ok((b, vec![out]))
    }
}

impl Block {
    /// One MMDiT block on `(x [n, d], cx [n_ctx, d])` with conditioning `st [1, d]`.
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &self,
        b: &mut Gb,
        p: &str,
        st: ValueId,
        x: ValueId,
        cx: ValueId,
        (n, n_ctx): (usize, usize),
        (d, heads): (usize, usize),
    ) -> (ValueId, ValueId) {
        let chunks = |b: &mut Gb, tag: &str, m: ValueId, k: usize| -> Vec<ValueId> {
            (0..k)
                .map(|i| b.slice(&format!("{p}.{tag}.c{i}"), m, 1, i * d, (i + 1) * d))
                .collect()
        };
        // Image stream chunks: shift, scale, gate (msa); shift, scale, gate (mlp); [shift, scale, gate (msa2)].
        let e = self.norm1.emit(b, &format!("{p}.norm1"), st);
        let ch = chunks(b, "e", e, self.norm1.out_dim() / d);
        let ln = b.layer_norm(&format!("{p}.ln1"), x, EPS);
        let nx = b.modulate(&format!("{p}.nx"), ln, ch[0], ch[1]);
        // SD3.5 MMDiT-X: the second self-attention reads the pre-attention hidden states.
        let nx2 = self
            .attn2
            .as_ref()
            .map(|_| b.modulate(&format!("{p}.nx2"), ln, ch[6], ch[7]));
        let ce = self.norm1_ctx.emit(b, &format!("{p}.norm1_ctx"), st);
        let cc = chunks(b, "ce", ce, self.norm1_ctx.out_dim() / d);
        let cln = b.layer_norm(&format!("{p}.ctx_ln1"), cx, EPS);
        let nc = match self.ff_ctx {
            // AdaLayerNormContinuous in the last block: chunks (scale, shift).
            None => b.modulate(&format!("{p}.nc"), cln, cc[1], cc[0]),
            Some(_) => b.modulate(&format!("{p}.nc"), cln, cc[0], cc[1]),
        };

        // Joint attention over [image; text].
        let (iq, ik, iv) = self.img.emit_qkv(b, &format!("{p}.img"), nx);
        let (cq, ck, cv) = self.ctx.emit_qkv(b, &format!("{p}.ctx"), nc);
        let q = b.concat(&format!("{p}.q"), &[iq, cq], 0);
        let k = b.concat(&format!("{p}.k"), &[ik, ck], 0);
        let v = b.concat(&format!("{p}.v"), &[iv, cv], 0);
        let a = b.attention(&format!("{p}.attn"), q, k, v, heads);
        let ia = b.slice(&format!("{p}.ia"), a, 0, 0, n);
        let ca = b.slice(&format!("{p}.ca"), a, 0, n, n + n_ctx);

        let o = self.img.emit_out(b, &format!("{p}.img"), ia);
        let mut x = b.gated_add(&format!("{p}.res1"), x, o, ch[2]);
        if let (Some(a2), Some(nx2)) = (&self.attn2, nx2) {
            let p2 = format!("{p}.attn2");
            let (q, k, v) = a2.emit_qkv(b, &p2, nx2);
            let o2 = b.attention(&format!("{p2}.attn"), q, k, v, heads);
            let o2 = a2.emit_out(b, &p2, o2);
            x = b.gated_add(&format!("{p}.res_a2"), x, o2, ch[8]);
        }
        let ln = b.layer_norm(&format!("{p}.ln2"), x, EPS);
        let nx = b.modulate(&format!("{p}.mlpin"), ln, ch[3], ch[4]);
        let f = self.ff.emit(b, &format!("{p}.ff"), nx);
        let x = b.gated_add(&format!("{p}.res2"), x, f, ch[5]);

        // The last block's context output is discarded.
        let cx = match &self.ff_ctx {
            None => cx,
            Some(ff) => {
                let o = self.ctx.emit_out(b, &format!("{p}.ctx"), ca);
                let cx = b.gated_add(&format!("{p}.cres1"), cx, o, cc[2]);
                let ln = b.layer_norm(&format!("{p}.ctx_ln2"), cx, EPS);
                let nc = b.modulate(&format!("{p}.cmlpin"), ln, cc[3], cc[4]);
                let f = ff.emit(b, &format!("{p}.ctx_ff"), nc);
                b.gated_add(&format!("{p}.cres2"), cx, f, cc[5])
            }
        };
        (x, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dit_graph::testutil::{cosine, mat, rand, rel_err};

    struct Rng(u64);
    impl Rng {
        fn seed(&mut self) -> u64 {
            self.0 += 1;
            self.0
        }
    }

    fn lin(r: &mut Rng, n: usize, k: usize) -> Lin {
        let b = rand(n, r.seed(), 0.1);
        Lin {
            w: mat(n, k, r.seed()),
            b: Some(b),
        }
    }

    fn gain(r: &mut Rng) -> Vec<f32> {
        rand(HEAD_DIM, r.seed(), 1.0)
            .iter()
            .map(|v| 1.0 + 0.3 * v)
            .collect()
    }

    const D: usize = 128;
    const JOINT: usize = 48;
    const POOLED: usize = 40;
    const POS_MAX: usize = 8;
    const LAYERS: usize = 3;

    /// Per-stream attention projections; `out: false` is the context-pre-only last block.
    fn proj(r: &mut Rng, qk_norm: bool, out: bool) -> Proj {
        Proj {
            q: lin(r, D, D),
            k: lin(r, D, D),
            v: lin(r, D, D),
            out: out.then(|| lin(r, D, D)),
            norm_q: qk_norm.then(|| gain(r)),
            norm_k: qk_norm.then(|| gain(r)),
        }
    }

    fn ff(r: &mut Rng) -> Ff {
        Ff {
            up: lin(r, 4 * D, D),
            down: lin(r, D, 4 * D),
        }
    }

    /// `dual`: SD3.5 dual attention (norm1 9 chunks, attn2) on block 0.
    fn tiny(dual: bool, qk_norm: bool, seed: u64) -> Mmdit {
        let r = &mut Rng(seed * 1000);
        let blocks = (0..LAYERS)
            .map(|i| {
                let last = i + 1 == LAYERS;
                let a2 = dual && i == 0;
                Block {
                    norm1: lin(r, if a2 { 9 * D } else { 6 * D }, D),
                    norm1_ctx: lin(r, if last { 2 * D } else { 6 * D }, D),
                    img: proj(r, qk_norm, true),
                    ctx: proj(r, qk_norm, !last),
                    attn2: a2.then(|| proj(r, qk_norm, true)),
                    ff: ff(r),
                    ff_ctx: (!last).then(|| ff(r)),
                }
            })
            .collect();
        Mmdit {
            d: D,
            heads: D / HEAD_DIM,
            patch: 2,
            in_ch: 4,
            pos_max: POS_MAX,
            patch_proj: lin(r, D, 4 * 2 * 2),
            pos_embed: mat(POS_MAX * POS_MAX, D, r.seed()),
            t_lin1: lin(r, D, 256),
            t_lin2: lin(r, D, D),
            text_lin1: lin(r, D, POOLED),
            text_lin2: lin(r, D, D),
            context_embedder: lin(r, D, JOINT),
            blocks,
            norm_out: lin(r, 2 * D, D),
            proj_out: lin(r, 4 * 2 * 2, D),
            gpu: Default::default(),
        }
    }

    /// Metal program == native CPU forward, per SD3 / SD3.5 variant, at f32 (bf16 weights shared).
    #[test]
    fn sd3_program_matches_cpu_forward() {
        for (dual, qk_norm) in [(false, false), (true, true), (true, false), (false, true)] {
            let m = tiny(dual, qk_norm, 1 + dual as u64 * 2 + qk_norm as u64);
            // 12x10 latents -> 6x5 patches = 30 image tokens; 28 text tokens; cropped from an 8x8 table.
            let (h, w, n_ctx) = (12usize, 10usize, 28usize);
            let lat = rand(4 * h * w, 11, 1.0);
            let ctx = rand(n_ctx * JOINT, 12, 1.0);
            let pooled = rand(POOLED, 13, 1.0);
            for t in [700.0f32, 120.0] {
                let want = {
                    let _cpu = crate::hardware::cpu_compute::allow();
                    m.forward(&lat, h, w, t, &ctx, n_ctx, &pooled)
                };
                let got = m
                    .forward_gpu(&lat, h, w, t, &ctx, n_ctx, &pooled)
                    .expect("metal program");
                let (e, c) = (rel_err(&want, &got), cosine(&want, &got));
                eprintln!("dual={dual} qk_norm={qk_norm} t={t}: rel {e:.2e} cos {c:.7}");
                assert!(
                    e < 1e-3 && c > 0.99999,
                    "dual={dual} qk={qk_norm}: rel {e} cos {c}"
                );
            }
        }
    }
}
