//! Gemma 4 vision tower (`gemma4_vision`) and its image processor: aspect-preserving resize to a
//! soft-token budget, 16 px patches with learned 2-D position embeddings, a bidirectional ViT with
//! axial 2-D RoPE, 3x3 average pooling, then `embed_vision` into the text width.

use super::ops::{gelu_mul, linear, rms_norm, rope, rope_table, Clipped};
use crate::graph::{DType, Graph, Op, RopeVariant, Tensor, UnaryKind, ValueId, Weights as GWeights};
use crate::model::image::ImageBuffer;
use crate::model::vision_graph::{Gb, ProgramCache};
use crate::model::weights::{Mat, Weights};
use crate::model::{bf16_to_f32, nn};
use serde_json::Value;

struct Layer {
    q: Clipped,
    k: Clipped,
    v: Clipped,
    o: Clipped,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Clipped,
    up: Clipped,
    down: Clipped,
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
}

pub struct Vision {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    patch: usize,
    pool: usize,
    /// Soft tokens per image at most (`image_processor.max_soft_tokens`).
    max_soft: usize,
    eps: f32,
    input_proj: Mat,
    /// `[2, position_embedding_size * hidden]`: x table, y table.
    pos_table: Mat,
    layers: Vec<Layer>,
    inv_freq: Vec<f32>,
    /// `embed_vision.embedding_projection` (vision hidden -> text hidden).
    proj: Mat,
    /// Metal programs (set by [`Self::on_gpu`]); `None` = the host forward.
    gpu: Option<ProgramCache>,
}

/// Largest `(h, w)` keeping the aspect ratio, both multiples of `pool * patch`, within
/// `max_patches` patches (transformers `get_aspect_ratio_preserving_size`).
pub fn target_size(
    height: usize,
    width: usize,
    patch: usize,
    max_patches: usize,
    pool: usize,
) -> Result<(usize, usize), String> {
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = pool * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    let max_side = (max_patches / (pool * pool)) * side;
    match (th, tw) {
        (0, 0) => return Err(format!("image {width}x{height} resizes to 0x0")),
        (0, _) => {
            th = side;
            tw = ((width / height) * side).min(max_side);
        }
        (_, 0) => {
            tw = side;
            th = ((height / width) * side).min(max_side);
        }
        _ => {}
    }
    if (th * tw) as f64 > target_px {
        return Err(format!(
            "image {width}x{height} exceeds {max_patches} patches"
        ));
    }
    Ok((th, tw))
}

/// Soft tokens image file `bytes` becomes: its [`target_size`] patch grid pooled `pool x pool`.
pub fn soft_tokens(
    bytes: &[u8],
    patch: usize,
    max_soft: usize,
    pool: usize,
) -> Result<usize, String> {
    let img = ImageBuffer::decode(bytes)?;
    let (th, tw) = target_size(img.height, img.width, patch, max_soft * pool * pool, pool)?;
    Ok((th / (patch * pool)) * (tw / (patch * pool)))
}

impl Vision {
    pub fn count(&self, bytes: &[u8]) -> Result<usize, String> {
        soft_tokens(bytes, self.patch, self.max_soft, self.pool)
    }

    pub fn load(w: &Weights, vc: &Value, processor: &Value) -> Result<Self, String> {
        let u = |k: &str| {
            vc[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let head_dim = u("head_dim")?;
        if vc["standardize"].as_bool().unwrap_or(false) {
            return Err("vision_config.standardize is not supported".into());
        }
        if u("num_key_value_heads")? != heads {
            return Err("vision tower with grouped KV heads is not supported".into());
        }
        let ip = &processor["image_processor"];
        let max_soft = ip["max_soft_tokens"].as_u64().unwrap_or(280) as usize;
        let p = "model.vision_tower";
        let n = u("num_hidden_layers")?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let lp = format!("{p}.encoder.layers.{i}");
            let c = |s: &str| Clipped::load(w, &format!("{lp}.{s}"));
            let v = |s: &str, d: usize| w.vec_n(&format!("{lp}.{s}.weight"), d);
            layers.push(Layer {
                q: c("self_attn.q_proj")?,
                k: c("self_attn.k_proj")?,
                v: c("self_attn.v_proj")?,
                o: c("self_attn.o_proj")?,
                q_norm: v("self_attn.q_norm", head_dim)?,
                k_norm: v("self_attn.k_norm", head_dim)?,
                gate: c("mlp.gate_proj")?,
                up: c("mlp.up_proj")?,
                down: c("mlp.down_proj")?,
                input_ln: v("input_layernorm", hidden)?,
                post_attn_ln: v("post_attention_layernorm", hidden)?,
                pre_ff_ln: v("pre_feedforward_layernorm", hidden)?,
                post_ff_ln: v("post_feedforward_layernorm", hidden)?,
            });
        }
        let theta = vc["rope_parameters"]["rope_theta"]
            .as_f64()
            .unwrap_or(100.0);
        // Axial RoPE: each of x and y rotates half the head with head_dim/4 frequencies.
        let spatial = head_dim / 2;
        let inv_freq = (0..spatial / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / spatial as f64)) as f32)
            .collect();
        let patch = u("patch_size")?;
        Ok(Self {
            hidden,
            heads,
            head_dim,
            patch,
            pool: u("pooling_kernel_size")?,
            max_soft,
            eps: vc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            input_proj: w.mat(&format!("{p}.patch_embedder.input_proj.weight"))?,
            pos_table: w.mat(&format!("{p}.patch_embedder.position_embedding_table"))?,
            layers,
            inv_freq,
            proj: w.mat("model.embed_vision.embedding_projection.weight")?,
            gpu: None,
        })
    }

    /// Runs the tower as Metal graph programs on `mode` (the default is the host forward).
    pub fn on_gpu(mut self, mode: crate::hardware::PlacementMode) -> Self {
        self.gpu = Some(ProgramCache::new(mode));
        self
    }

    /// Encodes one image (file bytes) to `[n, text_hidden]` soft tokens; returns `(rows, n)`.
    pub fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let (pix, ph, pw) = self.preprocess(bytes)?;
        let rows = match &self.gpu {
            Some(g) => self.forward_gpu(g, &pix, ph, pw)?,
            None => self.forward_cpu(&pix, ph, pw),
        };
        let k = self.pool;
        Ok((rows, (ph / k) * (pw / k)))
    }

    /// Decodes and resizes the image to its [`target_size`], then cuts `[np, p·p·3]` patches
    /// (`(py, px, c)` inside each patch, rescaled to `[-1, 1]`); returns them with the patch grid.
    fn preprocess(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
        let img = ImageBuffer::decode(bytes)?;
        let (p, k) = (self.patch, self.pool);
        let (th, tw) = target_size(img.height, img.width, p, self.max_soft * k * k, k)?;
        let img = if (th, tw) == (img.height, img.width) {
            img
        } else {
            img.resize(tw, th)
        };
        let (ph, pw) = (th / p, tw / p);
        let mut pix = vec![0.0f32; ph * pw * p * p * 3];
        for (i, patch) in pix.chunks_exact_mut(p * p * 3).enumerate() {
            let (gy, gx) = (i / pw, i % pw);
            for y in 0..p {
                for x in 0..p {
                    let src = ((gy * p + y) * tw + gx * p + x) * 3;
                    for c in 0..3 {
                        patch[(y * p + x) * 3 + c] = 2.0 * (img.data[src + c] as f32 / 255.0 - 0.5);
                    }
                }
            }
        }
        Ok((pix, ph, pw))
    }

    /// Row `i` of the x / y learned position tables as f32 (`[np, hidden]` each).
    fn pos_rows(&self, ph: usize, pw: usize) -> (Vec<f32>, Vec<f32>) {
        let (d, np) = (self.hidden, ph * pw);
        let table = self.pos_table.data();
        let tlen = self.pos_table.cols;
        let (mut xs, mut ys) = (Vec::with_capacity(np * d), Vec::with_capacity(np * d));
        for i in 0..np {
            let (gx, gy) = (i % pw, i / pw);
            xs.extend(table[gx * d..][..d].iter().map(|&v| bf16_to_f32(v)));
            ys.extend(table[tlen + gy * d..][..d].iter().map(|&v| bf16_to_f32(v)));
        }
        (xs, ys)
    }

    /// The host forward (`--backend cpu`): `[np, p·p·3]` patches → `[n, text_hidden]` soft tokens.
    pub(crate) fn forward_cpu(&self, pix: &[f32], ph: usize, pw: usize) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("gemma4::vision::forward");
        let k = self.pool;
        let np = ph * pw;
        let d = self.hidden;
        let mut h = linear(pix, np, &self.input_proj);
        let table = self.pos_table.data();
        let tlen = self.pos_table.cols;
        for (i, row) in h.chunks_exact_mut(d).enumerate() {
            let (gx, gy) = (i % pw, i / pw);
            let xe = &table[gx * d..][..d];
            let ye = &table[tlen + gy * d..][..d];
            for ((v, &a), &b) in row.iter_mut().zip(xe).zip(ye) {
                *v += bf16_to_f32(a) + bf16_to_f32(b);
            }
        }
        let ropes: Vec<_> = (0..np)
            .map(|i| {
                (
                    rope_table((i % pw) as f32, &self.inv_freq),
                    rope_table((i / pw) as f32, &self.inv_freq),
                )
            })
            .collect();
        let (hd, nh) = (self.head_dim, self.heads);
        let half = hd / 2;
        for l in &self.layers {
            let mut a = h.clone();
            rms_norm(&mut a, d, Some(&l.input_ln), self.eps);
            let mut q = l.q.forward(&a, np);
            let mut kk = l.k.forward(&a, np);
            let mut v = l.v.forward(&a, np);
            rms_norm(&mut q, hd, Some(&l.q_norm), self.eps);
            rms_norm(&mut kk, hd, Some(&l.k_norm), self.eps);
            rms_norm(&mut v, hd, None, self.eps);
            for (i, ((cx, sx), (cy, sy))) in ropes.iter().enumerate() {
                for t in [&mut q, &mut kk] {
                    for head in t[i * nh * hd..(i + 1) * nh * hd].chunks_exact_mut(hd) {
                        rope(&mut head[..half], cx, sx);
                        rope(&mut head[half..], cy, sy);
                    }
                }
            }
            // Scale 1: undo `nn::attention`'s 1/sqrt(head_dim).
            let s = (hd as f32).sqrt();
            q.iter_mut().for_each(|v| *v *= s);
            let att = nn::attention(&q, &kk, &v, np, np, nh, hd, false);
            let mut o = l.o.forward(&att, np);
            rms_norm(&mut o, d, Some(&l.post_attn_ln), self.eps);
            h.iter_mut().zip(&o).for_each(|(h, o)| *h += o);
            let mut a = h.clone();
            rms_norm(&mut a, d, Some(&l.pre_ff_ln), self.eps);
            let mut g = l.gate.forward(&a, np);
            gelu_mul(&mut g, &l.up.forward(&a, np));
            let mut m = l.down.forward(&g, np);
            rms_norm(&mut m, d, Some(&l.post_ff_ln), self.eps);
            h.iter_mut().zip(&m).for_each(|(h, m)| *h += m);
        }
        // k x k average pooling over the patch grid (row-major pooled grid), times sqrt(hidden).
        let (oh, ow) = (ph / k, pw / k);
        let n = oh * ow;
        let mut pooled = vec![0.0f32; n * d];
        let scale = (d as f32).sqrt() / (k * k) as f32;
        for (i, row) in h.chunks_exact(d).enumerate() {
            let (gy, gx) = (i / pw, i % pw);
            let dst = &mut pooled[((gy / k) * ow + gx / k) * d..][..d];
            dst.iter_mut().zip(row).for_each(|(o, v)| *o += v * scale);
        }
        rms_norm(&mut pooled, d, None, self.eps);
        linear(&pooled, n, &self.proj)
    }

    /// Runs the tower as a Metal graph program (compiled once per patch-grid shape).
    pub(crate) fn forward_gpu(&self, gpu: &ProgramCache, pix: &[f32], ph: usize, pw: usize) -> Result<Vec<f32>, String> {
        let (np, nh) = (ph * pw, self.heads);
        let prog = gpu
            .get(&[ph, pw], || self.build_graph(ph, pw))
            .map_err(|e| format!("vision: {e}"))?;
        let (xs, ys) = self.pos_rows(ph, pw);
        // Per-head row `i * nh + h` takes patch `i`'s column (x) / row (y) as its RoPE position.
        let rope_pos = |axis: usize| -> Vec<u32> {
            (0..np * nh)
                .map(|r| if axis == 0 { (r / nh % pw) as u32 } else { (r / nh / pw) as u32 })
                .collect()
        };
        let inputs = vec![
            Tensor::F32(pix.to_vec()),
            Tensor::F32(xs),
            Tensor::F32(ys),
            Tensor::U32(rope_pos(0)),
            Tensor::U32(rope_pos(1)),
        ];
        let out = prog.lock().run(inputs)?;
        out.into_iter().next().ok_or("vision program returned nothing")?.into_f32()
    }

    /// Axial 2-D RoPE: per head, the first half rotates by the patch column, the second by its row.
    #[allow(clippy::too_many_arguments)]
    fn axial_rope(&self, b: &mut Gb, name: &str, x: ValueId, np: usize, pos: (ValueId, ValueId), inv: ValueId) -> ValueId {
        let (nh, hd) = (self.heads, self.head_dim);
        let half = hd / 2;
        let rows = np * nh;
        let x = b.reshape(&format!("{name}.heads"), x, vec![rows, hd]);
        let rot = |b: &mut Gb, tag: &str, start: usize, p: ValueId| {
            let part = b.op(
                &format!("{name}.{tag}"),
                Op::Slice { axis: -1, start: start as i64, end: (start + half) as i64 },
                &[x],
                vec![rows, half],
            );
            let op = Op::Rope { head_dim: half, rotary_dim: half, variant: RopeVariant::HalfSplit, mscale: 1.0 };
            b.op(&format!("{name}.{tag}.rope"), op, &[part, p, inv], vec![rows, half])
        };
        let (lo, hi) = (rot(b, "x", 0, pos.0), rot(b, "y", half, pos.1));
        let cat = b.op(&format!("{name}.cat"), Op::Concat { axis: -1 }, &[lo, hi], vec![rows, hd]);
        b.reshape(&format!("{name}.flat"), cat, vec![np, nh * hd])
    }

    /// The tower as a graph for a `ph × pw` patch grid. Inputs: `patches [np, p·p·3]`, the x / y
    /// position-table rows `[np, hidden]` (checkpoint row lookups), and the per-head RoPE
    /// positions `[np·heads]` (U32, x then y). Output: `[n, text_hidden]` soft tokens.
    pub(crate) fn build_graph(&self, ph: usize, pw: usize) -> Result<(Graph, GWeights), String> {
        let (d, nh, hd, k) = (self.hidden, self.heads, self.head_dim, self.pool);
        if ph % k != 0 || pw % k != 0 {
            return Err(format!("{ph}x{pw} patch grid is not a multiple of the {k}-pooling window"));
        }
        let (np, eps) = (ph * pw, self.eps);
        let mut b = Gb::new();
        let pix = b.input("patches", vec![np, self.input_proj.cols], DType::F32);
        let xe = b.input("pos_x", vec![np, d], DType::F32);
        let ye = b.input("pos_y", vec![np, d], DType::F32);
        let px = b.input("rope_x", vec![np * nh], DType::U32);
        let py = b.input("rope_y", vec![np * nh], DType::U32);
        let inv = b.f32w("rope.inv_freq", vec![self.inv_freq.len()], self.inv_freq.clone());
        let w = b.mat("vision.input_proj.weight", &self.input_proj);
        let h0 = b.linear("vision.input_proj", pix, w, None, np, d);
        let h1 = b.add("vision.pos_x", h0, xe, vec![np, d]);
        let mut h = b.add("vision.pos_y", h1, ye, vec![np, d]);
        let sq = (hd as f32).sqrt();
        let attn = Op::Attention { heads: nh, kv_heads: nh, causal: false, window: None, soft_cap: None };
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("vision.layers.{i}");
            let a = b.rms(&format!("{p}.input_ln"), h, l.input_ln.clone(), eps, vec![np, d]);
            let q = l.q.add_to(&mut b, &format!("{p}.q"), a, np);
            let kk = l.k.add_to(&mut b, &format!("{p}.k"), a, np);
            let v = l.v.add_to(&mut b, &format!("{p}.v"), a, np);
            let width = nh * hd;
            // Attention scales by 1/sqrt(hd); the checkpoint's scale is 1, so q's gain carries sqrt(hd).
            let qn = l.q_norm.iter().map(|g| g * sq).collect();
            let q = b.rms(&format!("{p}.q_norm"), q, qn, eps, vec![np, width]);
            let kk = b.rms(&format!("{p}.k_norm"), kk, l.k_norm.clone(), eps, vec![np, width]);
            let v = b.rms(&format!("{p}.v_norm"), v, vec![1.0; hd], eps, vec![np, width]);
            let q = self.axial_rope(&mut b, &format!("{p}.q_rope"), q, np, (px, py), inv);
            let kk = self.axial_rope(&mut b, &format!("{p}.k_rope"), kk, np, (px, py), inv);
            let att = b.op(&format!("{p}.attn"), attn.clone(), &[q, kk, v], vec![np, width]);
            let o = l.o.add_to(&mut b, &format!("{p}.o"), att, np);
            let o = b.rms(&format!("{p}.post_attn_ln"), o, l.post_attn_ln.clone(), eps, vec![np, d]);
            h = b.add(&format!("{p}.res1"), h, o, vec![np, d]);
            let a = b.rms(&format!("{p}.pre_ff_ln"), h, l.pre_ff_ln.clone(), eps, vec![np, d]);
            let g = l.gate.add_to(&mut b, &format!("{p}.gate"), a, np);
            let u = l.up.add_to(&mut b, &format!("{p}.up"), a, np);
            let inter = l.gate.w.rows;
            // gelu(gate) * up as Unary + Mul: a literal `GeGlu` here lets the megakernel MLP fusion
            // claim the block, and that kernel is single-row only (this graph has `np` rows).
            let g = b.unary(&format!("{p}.gelu"), UnaryKind::GeluTanh, g, vec![np, inter]);
            let m = b.op(&format!("{p}.glu"), Op::Mul, &[g, u], vec![np, inter]);
            let m = l.down.add_to(&mut b, &format!("{p}.down"), m, np);
            let m = b.rms(&format!("{p}.post_ff_ln"), m, l.post_ff_ln.clone(), eps, vec![np, d]);
            h = b.add(&format!("{p}.res2"), h, m, vec![np, d]);
        }
        // k×k average pooling over the patch grid, times sqrt(hidden), then the unit-gain norm.
        let (oh, ow) = (ph / k, pw / k);
        let n = oh * ow;
        let ht = b.transpose("vision.pool.t", h, &[np, d], &[1, 0]);
        let grid = b.reshape("vision.pool.grid", ht, vec![d, ph, pw]);
        let pool = Op::AvgPool2d { kernel: [k, k], stride: [k, k] };
        let pooled = b.op("vision.pool", pool, &[grid], vec![d, oh, ow]);
        let pooled = b.unary("vision.pool.scale", UnaryKind::Scale((d as f32).sqrt()), pooled, vec![d, oh, ow]);
        let flat = b.reshape("vision.pool.flat", pooled, vec![d, n]);
        let rows = b.transpose("vision.pool.rows", flat, &[d, n], &[1, 0]);
        let rows = b.rms("vision.pool.norm", rows, vec![1.0; d], eps, vec![n, d]);
        let pw_ = b.mat("vision.embed.weight", &self.proj);
        let out = b.linear("vision.embed", rows, pw_, None, n, self.proj.rows);
        Ok(b.finish(&[out]))
    }
}

/// Deterministic synthetic tiny tower (hidden 16, 2 layers, 4 px patches, 2x2 pooling, text width
/// 24): the fixture of `tests/cpu_free_matrix.rs` and the CPU-vs-Metal parity tests.
pub mod tiny {
    use super::*;
    use crate::model::siglip::tiny::values;

    pub const TEXT: usize = 24;
    const D: usize = 16;
    const INTER: usize = 32;
    const PATCH: usize = 4;
    const POS: usize = 8;

    pub fn weights() -> Weights {
        let (hd, mut specs) = (8, Vec::new());
        let mut add = |n: String, shape: &[usize]| specs.push((n, shape.to_vec()));
        let p = "model.vision_tower";
        add(format!("{p}.patch_embedder.input_proj.weight"), &[D, PATCH * PATCH * 3]);
        add(format!("{p}.patch_embedder.position_embedding_table"), &[2, POS * D]);
        add("model.embed_vision.embedding_projection.weight".into(), &[TEXT, D]);
        for i in 0..2 {
            let l = format!("{p}.encoder.layers.{i}");
            for (n, rows, cols) in [
                ("self_attn.q_proj", D, D),
                ("self_attn.k_proj", D, D),
                ("self_attn.v_proj", D, D),
                ("self_attn.o_proj", D, D),
                ("mlp.gate_proj", INTER, D),
                ("mlp.up_proj", INTER, D),
                ("mlp.down_proj", D, INTER),
            ] {
                add(format!("{l}.{n}.linear.weight"), &[rows, cols]);
            }
            // `ClippableLinear` bounds on the query and output projections only.
            for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                for b in ["input_min", "input_max", "output_min", "output_max"] {
                    add(format!("{l}.self_attn.{n}.{b}"), &[1]);
                }
            }
            for (n, len) in [
                ("self_attn.q_norm", hd),
                ("self_attn.k_norm", hd),
                ("input_layernorm", D),
                ("post_attention_layernorm", D),
                ("pre_feedforward_layernorm", D),
                ("post_feedforward_layernorm", D),
            ] {
                add(format!("{l}.{n}.weight"), &[len]);
            }
        }
        Weights::from_f32(specs, |name| Ok(tensor(name))).expect("tiny gemma4 vision weights")
    }

    fn tensor(name: &str) -> Vec<f32> {
        let bound = |n: &str| name.ends_with(n);
        if bound("input_min") {
            return vec![-1.5];
        } else if bound("input_max") {
            return vec![1.5];
        } else if bound("output_min") {
            return vec![-2.5];
        } else if bound("output_max") {
            return vec![2.5];
        }
        let (n, scale, offset) = match name {
            n if n.ends_with("norm.weight") || n.ends_with("layernorm.weight") => (D, 0.1, 1.0),
            n if n.ends_with("position_embedding_table") => (2 * POS * D, 0.3, 0.0),
            n if n.ends_with("input_proj.weight") => (D * PATCH * PATCH * 3, 0.2, 0.0),
            n if n.ends_with("embedding_projection.weight") => (TEXT * D, 0.2, 0.0),
            n if n.contains("gate_proj") || n.contains("up_proj") => (INTER * D, 0.2, 0.0),
            n if n.contains("down_proj") => (D * INTER, 0.2, 0.0),
            _ => (D * D, 0.2, 0.0),
        };
        // q_norm / k_norm gains are `head_dim` (8) long.
        let n = if name.contains("q_norm") || name.contains("k_norm") { 8 } else { n };
        values(name, n).into_iter().map(|v| offset + v * scale).collect()
    }

    pub fn vision() -> Vision {
        let vc = serde_json::json!({
            "hidden_size": D, "num_attention_heads": 2, "head_dim": 8, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "patch_size": PATCH, "pooling_kernel_size": 2,
            "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 100.0},
        });
        let processor = serde_json::json!({"image_processor": {"max_soft_tokens": 4}});
        Vision::load(&weights(), &vc, &processor).expect("tiny gemma4 vision loads")
    }

    /// `[np, p·p·3]` pseudo-random patches for a `ph × pw` grid.
    #[cfg(test)]
    pub fn patches(ph: usize, pw: usize) -> Vec<f32> {
        values("patches", ph * pw * PATCH * PATCH * 3)
    }

    /// The tower graph for a 4x4 patch grid (4 soft tokens).
    pub fn graph() -> (Graph, GWeights) {
        vision().build_graph(4, 4).expect("tiny gemma4 vision graph")
    }
}

#[cfg(test)]
mod tests {
    use super::target_size;

    #[test]
    fn target_size_fills_the_budget_on_pool_multiples() {
        // 2520 patches = 280 soft tokens of 3x3 16-px patches.
        let (h, w) = target_size(480, 640, 16, 2520, 3).unwrap();
        assert_eq!((h % 48, w % 48), (0, 0));
        assert!(h * w <= 2520 * 256 && (h + 48) * (w + 48) > 2520 * 256);
        assert_eq!(target_size(768, 768, 16, 2520, 3).unwrap(), (768, 768));
        // Extreme aspect: one side clamps to a single pooled cell.
        assert_eq!(target_size(10, 4000, 16, 2520, 3).unwrap().0, 48);
    }

    /// The Metal tower (clamps, QK/V norms, axial RoPE, bidirectional attention, pooling) matches the
    /// host forward on a non-square patch grid.
    #[test]
    fn metal_tower_matches_the_host_forward() {
        use crate::hardware::{available_devices, cpu_compute, Device, PlacementMode};
        if !available_devices().contains(&Device::Metal) {
            println!("SKIP: no Metal device");
            return;
        }
        let _allow = cpu_compute::allow();
        let cpu = super::tiny::vision();
        let gpu = super::tiny::vision().on_gpu(PlacementMode::Pin(Device::Metal));
        for (ph, pw) in [(4, 4), (2, 6)] {
            let pix = super::tiny::patches(ph, pw);
            let want = cpu.forward_cpu(&pix, ph, pw);
            let cache = gpu.gpu.as_ref().unwrap();
            let got = gpu.forward_gpu(cache, &pix, ph, pw).unwrap();
            let prog = cache.get(&[ph, pw], || unreachable!("compiled by forward_gpu")).unwrap();
            assert_eq!(prog.lock().cpu_node_count(), 0, "nodes run on the CPU");
            let abs = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(abs < 1e-3, "max |d| {abs}");
            assert_eq!(got.len(), (ph / 2) * (pw / 2) * super::tiny::TEXT);
            let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let err = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            println!("gemma4 vision {ph}x{pw}: rel err {err:e}");
            assert!(err < 1e-3, "{ph}x{pw}: {err}");
        }
    }
}
