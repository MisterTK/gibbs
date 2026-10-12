//! Krea 2 text-to-image (`krea/Krea-2-Raw`, `krea/Krea-2-Turbo`, ComfyUI's `Comfy-Org/Krea-2` split
//! files, their finetunes, LoRAs and city96-style GGUFs): twelve Qwen3-VL-4B hidden states per token
//! fused by a small text transformer, a 13B single-stream DiT (GQA, per-head QK-norm, sigmoid-gated
//! attention, SwiGLU, 3-axis RoPE, shared AdaLN-single timestep modulation) over 2x2-packed
//! 16-channel latents, flow-matching Euler, and the Qwen-Image VAE.
//!
//! The transformer reads the original/ComfyUI names (`blocks.N.attn.wq`, single files and GGUFs) or
//! diffusers' (`transformer_blocks.N.attn.to_q`). Weights stay where they are stored (mmap'd bf16 or
//! GGUF blocks) and are decoded per GEMM panel, as for FLUX.

use crate::model::dit::{add_gated, read_config, rms_norm, timestep_features};
use crate::model::flow;
use crate::model::flux::{Lin, Loras, Store};
use crate::model::hub::{component_weights, HuggingFaceHub, ModelArtifacts};
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::model::qwen3_5::Qwen3Encoder;
use crate::model::qwen_vae::QwenVae;
use crate::model::sd::Rng;
use crate::model::text_to_image::{ImageDefaults, ImageGenerator, ImageRequest};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const PATCH: usize = 2;
const EPS: f32 = 1e-5;
/// Qwen3-VL decoder outputs fed to the DiT: transformers' `hidden_states[k]`.
const TAPS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
/// Tokens of the chat-template prefix the DiT never sees, and of its closing suffix.
const PREFIX_TOKENS: usize = 34;
const MAX_TEXT: usize = 512;
const PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// The ungated tokenizer of the Qwen3-VL-4B text encoder (ComfyUI ships none beside its weights).
const TOKENIZER_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
/// ComfyUI's own distribution of the Krea 2 weights (split files; README: "Repackaged model files").
pub const COMFY_REPO: &str = "Comfy-Org/Krea-2";

/// The diffusers name of an original/ComfyUI tensor (diffusers' `convert_krea2_transformer_checkpoint`).
fn to_diffusers(key: &str) -> String {
    const PREFIX: [(&str, &str); 12] = [
        ("first.", "img_in."),
        ("tmlp.0.", "time_embed.linear_1."),
        ("tmlp.2.", "time_embed.linear_2."),
        ("tproj.1.", "time_mod_proj."),
        ("txtmlp.0.scale", "txt_in.norm.weight"),
        ("txtmlp.1.", "txt_in.linear_1."),
        ("txtmlp.3.", "txt_in.linear_2."),
        ("txtfusion.", "text_fusion."),
        ("blocks.", "transformer_blocks."),
        ("last.linear.", "final_layer.linear."),
        ("last.norm.scale", "final_layer.norm.weight"),
        ("last.modulation.lin", "final_layer.scale_shift_table"),
    ];
    const BLOCK: [(&str, &str); 11] = [
        (".attn.wq.", ".attn.to_q."),
        (".attn.wk.", ".attn.to_k."),
        (".attn.wv.", ".attn.to_v."),
        (".attn.wo.", ".attn.to_out.0."),
        (".attn.gate.", ".attn.to_gate."),
        (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
        (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
        (".mlp.", ".ff."),
        (".prenorm.scale", ".norm1.weight"),
        (".postnorm.scale", ".norm2.weight"),
        (".mod.lin", ".scale_shift_table"),
    ];
    let mut k = PREFIX
        .iter()
        .find_map(|(a, b)| key.strip_prefix(a).map(|r| format!("{b}{r}")))
        .unwrap_or_else(|| key.to_string());
    for (a, b) in BLOCK {
        k = k.replace(a, b);
    }
    k
}

/// Weights reader with the original names resolved to whichever layout the file has.
struct Src<'a> {
    s: &'a Store,
    diffusers: bool,
    loras: &'a mut Loras,
}

impl Src<'_> {
    fn nm(&self, orig: &str) -> String {
        if self.diffusers {
            to_diffusers(orig)
        } else {
            orig.to_string()
        }
    }

    fn vec(&self, orig: &str, len: usize) -> Result<Vec<f32>, String> {
        let v = self.s.vec(&self.nm(orig))?;
        if v.len() != len {
            return Err(format!(
                "Krea 2: '{orig}' has {} values, expected {len}",
                v.len()
            ));
        }
        Ok(v)
    }

    /// A zero-centred RMSNorm scale as the multiplier `1 + scale`.
    fn norm(&self, orig: &str, len: usize) -> Result<Vec<f32>, String> {
        Ok(self.vec(orig, len)?.into_iter().map(|v| v + 1.0).collect())
    }

    /// `{module}.weight` (+ `.bias`) with the LoRAs aimed at `module` (either naming) attached.
    fn lin(&mut self, module: &str) -> Result<Lin, String> {
        let w = self.s.mat(&self.nm(&format!("{module}.weight")))?;
        let (n, k) = w.dims();
        let bn = self.nm(&format!("{module}.bias"));
        let mut bias = self.s.has(&bn).then(|| self.s.vec(&bn)).transpose()?;
        let dname = to_diffusers(&format!("{module}.weight"));
        let dname = dname.strip_suffix(".weight").unwrap_or(&dname);
        let mut loras = Vec::new();
        for m in [module, dname] {
            loras.extend(self.loras.take(m, k, 0, n)?);
            if let Some(d) = self.loras.take_bias(m) {
                match &mut bias {
                    Some(b) if b.len() == d.len() => {
                        b.iter_mut().zip(&d).for_each(|(b, d)| *b += d)
                    }
                    _ => {
                        return Err(format!(
                            "LoRA bias delta for '{module}' has no matching bias"
                        ))
                    }
                }
            }
        }
        Ok(Lin::new(w, bias, loras))
    }
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    gate: Lin,
    o: Lin,
    qn: Vec<f32>,
    kn: Vec<f32>,
    heads: usize,
    kv: usize,
    hd: usize,
}

impl Attn {
    fn load(s: &mut Src, p: &str, hd: usize) -> Result<Self, String> {
        let (q, k) = (
            s.lin(&format!("{p}.attn.wq"))?,
            s.lin(&format!("{p}.attn.wk"))?,
        );
        let (heads, kv) = (q.out_dim() / hd, k.out_dim() / hd);
        if heads * hd != q.out_dim() || kv == 0 || kv * hd != k.out_dim() || heads % kv != 0 {
            return Err(format!(
                "Krea 2: {p}.attn has {heads} query / {kv} key heads of {hd}"
            ));
        }
        Ok(Self {
            v: s.lin(&format!("{p}.attn.wv"))?,
            gate: s.lin(&format!("{p}.attn.gate"))?,
            o: s.lin(&format!("{p}.attn.wo"))?,
            qn: s.norm(&format!("{p}.attn.qknorm.qnorm.scale"), hd)?,
            kn: s.norm(&format!("{p}.attn.qknorm.knorm.scale"), hd)?,
            q,
            k,
            heads,
            kv,
            hd,
        })
    }

    /// x [rows, dim]; attention runs inside each of the `rows / group` consecutive groups.
    /// `rope` is cos/sin `[rows, hd/2]` (shared by q and k) when the block rotates.
    fn fwd(
        &self,
        x: &[f32],
        rows: usize,
        group: usize,
        rope: Option<&(Vec<f32>, Vec<f32>)>,
    ) -> Vec<f32> {
        let (mut q, mut k) = (self.q.fwd(x, rows), self.k.fwd(x, rows));
        let (v, mut gate) = (self.v.fwd(x, rows), self.gate.fwd(x, rows));
        rms_norm(&mut q, self.hd, &self.qn, EPS);
        rms_norm(&mut k, self.hd, &self.kn, EPS);
        if let Some((cos, sin)) = rope {
            rotate(&mut q, self.heads, self.hd, cos, sin);
            rotate(&mut k, self.kv, self.hd, cos, sin);
        }
        let (k, v) = (self.expand(&k, rows), self.expand(&v, rows));
        let w = self.heads * self.hd;
        let mut a = vec![0.0; rows * w];
        for ((a, q), (k, v)) in a
            .chunks_mut(group * w)
            .zip(q.chunks(group * w))
            .zip(k.chunks(group * w).zip(v.chunks(group * w)))
        {
            a.copy_from_slice(&nn::attention(
                q, k, v, group, group, self.heads, self.hd, false,
            ));
        }
        gate.iter_mut().for_each(|g| *g = 1.0 / (1.0 + (-*g).exp()));
        a.iter_mut().zip(&gate).for_each(|(a, g)| *a *= g);
        self.o.fwd(&a, rows)
    }

    /// `repeat_interleave` of the key/value heads up to the query head count.
    fn expand(&self, x: &[f32], rows: usize) -> Vec<f32> {
        if self.kv == self.heads {
            return x.to_vec();
        }
        let (rep, hd) = (self.heads / self.kv, self.hd);
        let mut out = Vec::with_capacity(rows * self.heads * hd);
        for row in x.chunks_exact(self.kv * hd) {
            for h in 0..self.heads {
                out.extend_from_slice(&row[(h / rep) * hd..][..hd]);
            }
        }
        out
    }
}

/// Rotates interleaved pairs of every head of x [rows, heads*hd] by cos/sin [rows, hd/2].
fn rotate(x: &mut [f32], heads: usize, hd: usize, cos: &[f32], sin: &[f32]) {
    let half = hd / 2;
    par_rows(x, heads * hd, &|r, row| {
        let (c, s) = (&cos[r * half..][..half], &sin[r * half..][..half]);
        for head in row.chunks_exact_mut(hd) {
            for i in 0..half {
                let (a, b) = (head[2 * i], head[2 * i + 1]);
                head[2 * i] = a * c[i] - b * s[i];
                head[2 * i + 1] = a * s[i] + b * c[i];
            }
        }
    });
}

/// `h[row] = h[row] * (1 + scale) + shift` for every row of `[_, d]`.
fn modulate(h: &mut [f32], d: usize, scale: &[f32], shift: &[f32]) {
    par_rows(h, d, &|_, r| {
        for (i, v) in r.iter_mut().enumerate() {
            *v = *v * (1.0 + scale[i]) + shift[i];
        }
    });
}

struct Block {
    /// Six AdaLN offsets `[prescale, preshift, pregate, postscale, postshift, postgate]` (DiT blocks).
    modulation: Option<Vec<f32>>,
    pre: Vec<f32>,
    post: Vec<f32>,
    attn: Attn,
    gate: Lin,
    up: Lin,
    down: Lin,
}

impl Block {
    fn load(s: &mut Src, p: &str, dim: usize, hd: usize, modulated: bool) -> Result<Self, String> {
        Ok(Self {
            modulation: modulated
                .then(|| s.vec(&format!("{p}.mod.lin"), 6 * dim))
                .transpose()?,
            pre: s.norm(&format!("{p}.prenorm.scale"), dim)?,
            post: s.norm(&format!("{p}.postnorm.scale"), dim)?,
            attn: Attn::load(s, p, hd)?,
            gate: s.lin(&format!("{p}.mlp.gate"))?,
            up: s.lin(&format!("{p}.mlp.up"))?,
            down: s.lin(&format!("{p}.mlp.down"))?,
        })
    }

    fn mlp(&self, h: &[f32], rows: usize) -> Vec<f32> {
        let mut g = self.gate.fwd(h, rows);
        nn::silu(&mut g);
        let u = self.up.fwd(h, rows);
        g.iter_mut().zip(&u).for_each(|(g, u)| *g *= u);
        self.down.fwd(&g, rows)
    }

    /// x [rows, d] in place. `tvec` (the shared `tproj` output) selects the modulated DiT form.
    fn fwd(
        &self,
        x: &mut [f32],
        rows: usize,
        group: usize,
        rope: Option<&(Vec<f32>, Vec<f32>)>,
        tvec: Option<&[f32]>,
    ) {
        let d = self.pre.len();
        let m: Option<Vec<f32>> = self
            .modulation
            .as_ref()
            .zip(tvec)
            .map(|(l, t)| l.iter().zip(t).map(|(a, b)| a + b).collect());
        let ch = |i: usize| m.as_ref().map(|m| &m[i * d..(i + 1) * d]);
        let mut h = x.to_vec();
        rms_norm(&mut h, d, &self.pre, EPS);
        if let (Some(sc), Some(sh)) = (ch(0), ch(1)) {
            modulate(&mut h, d, sc, sh);
        }
        add_gated(x, d, ch(2), &self.attn.fwd(&h, rows, group, rope));
        let mut h = x.to_vec();
        rms_norm(&mut h, d, &self.post, EPS);
        if let (Some(sc), Some(sh)) = (ch(3), ch(4)) {
            modulate(&mut h, d, sc, sh);
        }
        add_gated(x, d, ch(5), &self.mlp(&h, rows));
    }
}

struct Dit {
    dim: usize,
    tdim: usize,
    chan: usize,
    theta: f64,
    axes: [usize; 3],
    hd: usize,
    first: Lin,
    tmlp0: Lin,
    tmlp2: Lin,
    tproj: Lin,
    layerwise: Vec<Block>,
    projector: Lin,
    refiner: Vec<Block>,
    txt_norm: Vec<f32>,
    txt1: Lin,
    txt3: Lin,
    blocks: Vec<Block>,
    last_norm: Vec<f32>,
    last_mod: Vec<f32>,
    last: Lin,
}

impl Dit {
    /// `cfg`: the diffusers `transformer/config.json` when there is one (RoPE axes/theta; the sizes
    /// come from the tensors either way).
    fn load(path: &Path, loras: &mut Loras, cfg: &Value) -> Result<Self, String> {
        let store = Store::open(path)?;
        let diffusers = !store.has("first.weight") && store.has("img_in.weight");
        let mut s = Src {
            s: &store,
            diffusers,
            loras,
        };
        let first = s.lin("first")?;
        let dim = first.out_dim();
        let hd =
            s.s.vec(&s.nm("blocks.0.attn.qknorm.qnorm.scale"))
                .map_err(|e| format!("{}: not a Krea 2 transformer ({e})", path.display()))?
                .len();
        let n_blocks = (0..)
            .take_while(|i| s.s.has(&s.nm(&format!("blocks.{i}.mod.lin"))))
            .count();
        let tmlp0 = s.lin("tmlp.0")?;
        let tdim = tmlp0.in_dim();
        let tdim_txt =
            s.s.vec(&s.nm("txtfusion.layerwise_blocks.0.prenorm.scale"))?
                .len();
        let projector = s.lin("txtfusion.projector")?;
        let n_txt_layers = projector.in_dim();
        let theta = cfg["rope_theta"].as_f64().unwrap_or(1000.0);
        // ComfyUI: axes = [hd - 12 * (hd / 16), 6 * (hd / 16), 6 * (hd / 16)]; diffusers config lists them.
        let axes = match cfg["axes_dims_rope"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
        {
            Some(a) if a.len() == 3 => [a[0] as usize, a[1] as usize, a[2] as usize],
            _ => [hd - 12 * (hd / 16), 6 * (hd / 16), 6 * (hd / 16)],
        };
        if axes.iter().sum::<usize>() != hd || axes.iter().any(|a| a % 2 != 0) {
            return Err(format!(
                "Krea 2: RoPE axes {axes:?} do not split head_dim {hd}"
            ));
        }
        let blocks = (0..n_blocks)
            .map(|i| Block::load(&mut s, &format!("blocks.{i}"), dim, hd, true))
            .collect::<Result<Vec<_>, _>>()?;
        let txt_blocks = |s: &mut Src, kind: &str| {
            (0..2)
                .map(|i| Block::load(s, &format!("txtfusion.{kind}.{i}"), tdim_txt, hd, false))
                .collect::<Result<Vec<_>, _>>()
        };
        let dit = Self {
            dim,
            tdim,
            chan: first.in_dim() / (PATCH * PATCH),
            theta,
            axes,
            hd,
            tmlp0,
            tmlp2: s.lin("tmlp.2")?,
            tproj: s.lin("tproj.1")?,
            layerwise: txt_blocks(&mut s, "layerwise_blocks")?,
            projector,
            refiner: txt_blocks(&mut s, "refiner_blocks")?,
            txt_norm: s.norm("txtmlp.0.scale", tdim_txt)?,
            txt1: s.lin("txtmlp.1")?,
            txt3: s.lin("txtmlp.3")?,
            blocks,
            last_norm: s.norm("last.norm.scale", dim)?,
            last_mod: s.vec("last.modulation.lin", 2 * dim)?,
            last: s.lin("last.linear")?,
            first,
        };
        if n_blocks == 0
            || n_txt_layers != TAPS.len()
            || dit.projector.out_dim() != 1
            || dit.tproj.out_dim() != 6 * dim
            || dit.tmlp2.out_dim() != dim
            || dit.tmlp0.out_dim() != dim
            || dit.txt1.in_dim() != tdim_txt
            || dit.txt1.out_dim() != dim
            || dit.first.in_dim() % (PATCH * PATCH) != 0
            || dit.last.out_dim() != dit.first.in_dim()
        {
            return Err(format!(
                "{}: tensor shapes do not form a Krea 2 transformer",
                path.display()
            ));
        }
        Ok(dit)
    }

    /// The 12 tapped encoder states (`[12][n, txt_dim]`) -> text features `[n, dim]` for the DiT.
    fn fuse(&self, taps: &[Vec<f32>], n: usize) -> Vec<f32> {
        let (td, nl) = (self.txt_norm.len(), taps.len());
        // Token-major rows: row t * nl + layer.
        let mut x = vec![0.0f32; n * nl * td];
        for (l, tap) in taps.iter().enumerate() {
            for t in 0..n {
                x[(t * nl + l) * td..][..td].copy_from_slice(&tap[t * td..][..td]);
            }
        }
        for b in &self.layerwise {
            b.fwd(&mut x, n * nl, nl, None, None);
        }
        // Linear(12 -> 1) over the layer axis: one row of 12 per (token, channel).
        let mut cols = vec![0.0f32; n * td * nl];
        for t in 0..n {
            for l in 0..nl {
                for (c, v) in x[(t * nl + l) * td..][..td].iter().enumerate() {
                    cols[(t * td + c) * nl + l] = *v;
                }
            }
        }
        let mut y = self.projector.fwd(&cols, n * td);
        for b in &self.refiner {
            b.fwd(&mut y, n, n, None, None);
        }
        rms_norm(&mut y, td, &self.txt_norm, EPS);
        let mut h = self.txt1.fwd(&y, n);
        nn::gelu_tanh(&mut h);
        self.txt3.fwd(&h, n)
    }

    /// cos/sin tables `[n, hd/2]` for `(t, row, col)` positions.
    fn rope(&self, pos: &[[usize; 3]]) -> (Vec<f32>, Vec<f32>) {
        let half = self.hd / 2;
        let freqs: Vec<(usize, f64)> = self
            .axes
            .iter()
            .enumerate()
            .flat_map(|(a, &d)| {
                (0..d / 2).map(move |i| (a, 1.0 / self.theta.powf(2.0 * i as f64 / d as f64)))
            })
            .collect();
        let (mut cos, mut sin) = (
            vec![0.0f32; pos.len() * half],
            vec![0.0f32; pos.len() * half],
        );
        for (r, p) in pos.iter().enumerate() {
            for (k, &(a, f)) in freqs.iter().enumerate() {
                let ang = (p[a] as f64 * f) as f32;
                cos[r * half + k] = ang.cos();
                sin[r * half + k] = ang.sin();
            }
        }
        (cos, sin)
    }

    /// Velocity for latents `[C, h, w]` at noise level `sigma`; `ctx` is [`Self::fuse`]'s `[n_txt, dim]`.
    fn forward(&self, lat: &[f32], h: usize, w: usize, sigma: f32, ctx: &[f32]) -> Vec<f32> {
        let (d, c, p) = (self.dim, self.chan, PATCH);
        let (gh, gw) = (h / p, w / p);
        let n_img = gh * gw;
        let n_txt = ctx.len() / d;
        // Patch features in (c, py, px) order.
        let mut patches = vec![0.0f32; n_img * c * p * p];
        for i in 0..gh {
            for j in 0..gw {
                let row = &mut patches[(i * gw + j) * c * p * p..][..c * p * p];
                for ch in 0..c {
                    for py in 0..p {
                        for px in 0..p {
                            row[(ch * p + py) * p + px] =
                                lat[(ch * h + i * p + py) * w + j * p + px];
                        }
                    }
                }
            }
        }
        let mut t = self
            .tmlp0
            .fwd(&timestep_features(flow::timestep(sigma), self.tdim), 1);
        nn::gelu_tanh(&mut t);
        let t = self.tmlp2.fwd(&t, 1);
        let mut tg = t.clone();
        nn::gelu_tanh(&mut tg);
        let tvec = self.tproj.fwd(&tg, 1);

        let n = n_txt + n_img;
        let mut x = Vec::with_capacity(n * d);
        x.extend_from_slice(ctx);
        x.extend(self.first.fwd(&patches, n_img));
        let pos: Vec<[usize; 3]> = (0..n)
            .map(|r| match r.checked_sub(n_txt) {
                Some(i) => [0, i / gw, i % gw],
                None => [0; 3],
            })
            .collect();
        let rope = self.rope(&pos);
        for b in &self.blocks {
            b.fwd(&mut x, n, n, Some(&rope), Some(&tvec));
        }
        let mut y = x[n_txt * d..].to_vec();
        rms_norm(&mut y, d, &self.last_norm, EPS);
        let scale: Vec<f32> = t
            .iter()
            .zip(&self.last_mod[..d])
            .map(|(a, b)| a + b)
            .collect();
        let shift: Vec<f32> = t
            .iter()
            .zip(&self.last_mod[d..])
            .map(|(a, b)| a + b)
            .collect();
        modulate(&mut y, d, &scale, &shift);
        let y = self.last.fwd(&y, n_img);
        let mut out = vec![0.0f32; c * h * w];
        for i in 0..gh {
            for j in 0..gw {
                let row = &y[(i * gw + j) * c * p * p..][..c * p * p];
                for ch in 0..c {
                    for py in 0..p {
                        for px in 0..p {
                            out[(ch * h + i * p + py) * w + j * p + px] =
                                row[(ch * p + py) * p + px];
                        }
                    }
                }
            }
        }
        out
    }
}

/// Where each component of one pipeline comes from.
struct Parts {
    /// The transformer: safetensors file/index or GGUF.
    dit: PathBuf,
    /// `transformer/config.json` of a diffusers snapshot (`Null` for single files).
    dit_cfg: Value,
    loras: Vec<(PathBuf, f32)>,
    /// Qwen3-VL weights and their `config.json`.
    te: PathBuf,
    te_cfg: Value,
    tokenizer: PathBuf,
    vae_dir: PathBuf,
    distilled: bool,
}

/// transformers' `Qwen3VLConfig` of the 4B encoder, for ComfyUI's config-less `qwen3vl_4b_*.safetensors`.
fn qwen3vl_4b_config() -> Value {
    json!({"text_config": {
        "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 5000000,
        "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "tie_word_embeddings": true,
    }})
}

pub struct Krea2Pipeline {
    dit: Dit,
    enc: Qwen3Encoder,
    tok: tokenizers::Tokenizer,
    vae: QwenVae,
    distilled: bool,
    label: String,
}

impl Krea2Pipeline {
    /// A diffusers `Krea2Pipeline` snapshot (`krea/Krea-2-Raw`, `krea/Krea-2-Turbo`, ...).
    pub fn load(dir: &Path) -> Result<Self, String> {
        crate::graph::avenues::gate(crate::graph::avenues::Family::DiffusionDit)?;
        Self::from_diffusers(dir, None, Vec::new())
    }

    /// `dir`'s text encoder, VAE and scheduler around another transformer / extra LoRAs.
    fn from_diffusers(
        dir: &Path,
        dit: Option<&Path>,
        loras: Vec<(PathBuf, f32)>,
    ) -> Result<Self, String> {
        let mi: Value = std::fs::read_to_string(dir.join("model_index.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", dir.join("model_index.json").display()))?;
        let te_dir = dir.join("text_encoder");
        let tr_dir = dir.join("transformer");
        Self::assemble(Parts {
            dit: match dit {
                Some(p) => p.to_path_buf(),
                None => component_weights(&tr_dir)?,
            },
            dit_cfg: read_config(&tr_dir).unwrap_or(Value::Null),
            loras,
            te: component_weights(&te_dir)?,
            te_cfg: read_config(&te_dir)?,
            tokenizer: dir.join("tokenizer").join("tokenizer.json"),
            vae_dir: dir.join("vae"),
            distilled: mi["is_distilled"].as_bool().unwrap_or(false),
        })
    }

    fn assemble(p: Parts) -> Result<Self, String> {
        let mut loras = Loras::default();
        for (path, strength) in &p.loras {
            loras.add_file(path, *strength)?;
        }
        let dit = Dit::load(&p.dit, &mut loras, &p.dit_cfg)?;
        if !loras.by_module.is_empty() || !loras.bias.is_empty() {
            let mut left: Vec<&String> = loras.by_module.keys().chain(loras.bias.keys()).collect();
            left.sort();
            return Err(format!(
                "LoRA targets {} modules this Krea 2 transformer does not have (e.g. {}); text-encoder LoRAs are not supported",
                left.len(),
                left.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        let tok = tokenizers::Tokenizer::from_file(&p.tokenizer)
            .map_err(|e| format!("{}: {e}", p.tokenizer.display()))?;
        let vae = QwenVae::load(&p.vae_dir)?;
        let enc = Qwen3Encoder::from_weights(&p.te, &p.te_cfg)?;
        let quant = match &Store::open(&p.dit)? {
            Store::Gguf(..) => "GGUF",
            Store::St(..) => "bf16",
        };
        let label =
            format!(
            "Krea 2 {} ({} layers, dim {}, {quant}{}), Qwen3-VL-4B text encoder + Qwen-Image VAE",
            if p.distilled { "Turbo" } else { "Raw" },
            dit.blocks.len(),
            dit.dim,
            if p.loras.is_empty() { String::new() } else { format!(" + {} LoRA", p.loras.len()) },
        );
        Ok(Self {
            dit,
            enc,
            tok,
            vae,
            distilled: p.distilled,
            label,
        })
    }

    /// Prompt -> fused text features `[n, dim]` and `n`.
    fn encode(&self, prompt: &str) -> Result<(Vec<f32>, usize), String> {
        let ids = |s: &str| -> Result<Vec<u32>, String> {
            Ok(self
                .tok
                .encode(s, false)
                .map_err(|e| format!("Qwen tokenizer: {e}"))?
                .get_ids()
                .to_vec())
        };
        let mut all = ids(&format!("{PREFIX}{prompt}"))?;
        all.truncate(MAX_TEXT + PREFIX_TOKENS - 5);
        all.extend(ids(SUFFIX)?);
        if all.len() <= PREFIX_TOKENS {
            return Err("Krea 2: empty prompt encoding".into());
        }
        let hs = self.enc.hidden_states_at(&all, all.len(), &TAPS)?;
        let n = all.len() - PREFIX_TOKENS;
        let d = hs[0].len() / all.len();
        let taps: Vec<Vec<f32>> = hs.iter().map(|h| h[PREFIX_TOKENS * d..].to_vec()).collect();
        Ok((self.dit.fuse(&taps, n), n))
    }
}

/// `is_distilled` for a single file or derived repo: ComfyUI ships `krea2_turbo_*` and `krea2_raw_*`.
pub fn looks_distilled(name: &str) -> bool {
    name.to_lowercase().contains("turbo")
}

impl ImageGenerator for Krea2Pipeline {
    fn summary(&self) -> String {
        self.label.clone()
    }

    /// The diffusers examples: Raw 28 steps with guidance 4.5, Turbo 8 steps without guidance.
    /// ponytail: CPU-sized 512 default, 1024 is native.
    fn defaults(&self) -> ImageDefaults {
        let (steps, guidance) = if self.distilled { (8, 0.0) } else { (28, 4.5) };
        ImageDefaults {
            width: 512,
            height: 512,
            steps,
            guidance,
            multiple: 16,
        }
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        let d = self.defaults();
        let (width, height) = (req.width.unwrap_or(d.width), req.height.unwrap_or(d.height));
        let (steps, guidance) = (
            req.steps.unwrap_or(d.steps),
            req.guidance.unwrap_or(d.guidance),
        );
        if width == 0
            || height == 0
            || width % 16 != 0
            || height % 16 != 0
            || width > 8192
            || height > 8192
        {
            return Err(format!(
                "Krea 2: width/height must be multiples of 16 up to 8192, got {width}x{height}"
            ));
        }
        if !(1..=200).contains(&steps) {
            return Err(format!("Krea 2: steps must be in 1..=200, got {steps}"));
        }
        let (h, w) = (height / 8, width / 8);
        let t0 = Instant::now();
        let (cond, _) = self.encode(&req.prompt)?;
        // Guidance (> 0) runs the unconditional pass too: pred = cond + g (cond - uncond).
        let uncond = if guidance > 0.0 {
            Some(self.encode(req.negative_prompt.as_deref().unwrap_or(""))?.0)
        } else {
            None
        };
        tracing::info!("Krea 2 text encoder: {:.1}s", t0.elapsed().as_secs_f32());
        // Turbo: mu = 1.15; Raw: mu from the image token count (scheduler: 256/6400 tokens, 0.5/1.15).
        let mu = if self.distilled {
            1.15
        } else {
            flow::calculate_shift((h / 2) * (w / 2), 256.0, 6400.0, 0.5, 1.15)
        };
        let sigmas = flow::dynamic_sigmas(&flow::linspace_sigmas(steps), mu);
        let mut rng = Rng::new(req.seed);
        let mut x: Vec<f32> = (0..self.dit.chan * h * w).map(|_| rng.gaussian()).collect();
        for i in 0..steps {
            let t = Instant::now();
            let mut v = self.dit.forward(&x, h, w, sigmas[i], &cond);
            if let Some(u) = &uncond {
                let vu = self.dit.forward(&x, h, w, sigmas[i], u);
                v.iter_mut()
                    .zip(&vu)
                    .for_each(|(v, u)| *v += guidance * (*v - u));
            }
            flow::euler_step(&mut x, &v, sigmas[i], sigmas[i + 1]);
            tracing::info!(
                "Krea 2 step {}/{steps} (sigma {:.4}): {:.1}s",
                i + 1,
                sigmas[i],
                t.elapsed().as_secs_f32()
            );
        }
        self.vae.decode(&x, h, w)
    }
}

// ---------------------------------------------------------------------------
// Non-diffusers repos
// ---------------------------------------------------------------------------

/// What a weight file of a Krea 2 repo is.
enum Kind {
    Transformer,
    Lora,
}

/// Tensor names of a safetensors / GGUF file (header only).
fn tensor_names(path: &Path) -> Option<Vec<String>> {
    let file = std::fs::File::open(path).ok()?;
    // SAFETY: read-only mapping of a cache file, dropped before returning.
    let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    let m = match path.extension()?.to_str()? {
        "gguf" => crate::model::UniversalModelLoader::parse_gguf(&map).ok()?,
        "safetensors" => crate::model::UniversalModelLoader::parse_safetensors(&map).ok()?,
        _ => return None,
    };
    Some(m.tensors.into_keys().collect())
}

/// Krea 2 weights by their tensor names. A diffusers-named LoRA on attention projections alone
/// cannot be told from FLUX's by name, so `card_says_krea` (the model card's base) decides there.
fn kind(names: &[String], card_says_krea: bool) -> Option<Kind> {
    let has = |s: &str| names.iter().any(|n| n.contains(s));
    let transformer = ["txtfusion.projector.weight", "text_fusion.projector.weight"]
        .iter()
        .any(|s| has(s));
    if names.iter().any(|n| n.contains("lora")) {
        let own = has("txtfusion")
            || has("text_fusion")
            || has(".attn.wq.")
            || has(".attn.gate.")
            || has(".attn.to_gate.");
        return (own || card_says_krea).then_some(Kind::Lora);
    }
    transformer.then_some(Kind::Transformer)
}

fn card_says_krea(base: &str) -> bool {
    let b = base.to_lowercase();
    b.contains("krea-2") || b.contains("krea2")
}

/// The one file of `dir` (a ComfyUI `text_encoders/` / `vae/` folder) the bf16 pipeline uses.
fn pick_bf16(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    let bf16 = |p: &PathBuf| {
        p.file_name()
            .is_some_and(|n| n.to_string_lossy().contains("bf16"))
    };
    files.iter().find(|p| bf16(p)).or(files.first()).cloned()
}

/// ComfyUI's Krea 2 text encoder, VAE and (Qwen3-VL's) tokenizer for a transformer file, from the
/// split-file snapshot `root` when it has them, else fetched from [`COMFY_REPO`].
fn comfy_parts(
    hub: &HuggingFaceHub,
    root: Option<&Path>,
    dit: PathBuf,
    loras: Vec<(PathBuf, f32)>,
    distilled: bool,
) -> Result<Parts, String> {
    let local = |sub: &str| root.and_then(|r| pick_bf16(&r.join(sub)));
    let te = match local("text_encoders") {
        Some(p) => p,
        None => hub.fetch_file(COMFY_REPO, "text_encoders/qwen3vl_4b_bf16.safetensors")?,
    };
    let vae = match local("vae") {
        Some(p) => p,
        None => hub.fetch_file(COMFY_REPO, "vae/qwen_image_vae.safetensors")?,
    };
    Ok(Parts {
        dit,
        dit_cfg: Value::Null,
        loras,
        te,
        te_cfg: qwen3vl_4b_config(),
        tokenizer: hub.fetch_file(TOKENIZER_REPO, "tokenizer.json")?,
        vae_dir: vae.parent().expect("file has a parent").to_path_buf(),
        distilled,
    })
}

/// Krea 2 repos without a diffusers pipeline: ComfyUI's split files (`Comfy-Org/Krea-2`), a
/// transformer-only single file or city96-style GGUF (finetune, quantization), or a LoRA - the
/// latter two on the base their card names (`krea/Krea-2-Raw|Turbo`; a base that cannot be
/// fetched, e.g. gated for this token, is an error naming its license page, never a mirror).
/// `None` when the weights are not Krea 2's.
pub fn load_derived(
    a: &ModelArtifacts,
    hub: &HuggingFaceHub,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let weights = a.weights_path.as_ref()?;
    let repo = crate::model::hub::split_weight_file(&a.model_id).0;
    let card = a
        .local_dir
        .as_ref()
        .map(|d| d.join("README.md"))
        .filter(|p| p.is_file())
        .map_or_else(|| hub.fetch_file(repo, "README.md"), Ok)
        .and_then(|p| std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display())));
    let base = card
        .ok()
        .and_then(|c| crate::model::lora::card_base_model(&c));
    let kind = kind(
        &tensor_names(weights)?,
        base.as_deref().is_some_and(card_says_krea),
    )?;
    let file = weights.file_name()?.to_string_lossy().to_string();
    let distilled = [file.as_str(), repo, base.as_deref().unwrap_or("")]
        .iter()
        .any(|s| looks_distilled(s));
    Some((|| {
        let root = weights
            .parent()
            .filter(|p| p.file_name().is_some_and(|n| n == "diffusion_models"))
            .and_then(Path::parent);
        // ComfyUI's own split snapshot carries everything beside the transformer.
        let comfy = |loras, dit: PathBuf, root| comfy_parts(hub, root, dit, loras, distilled);
        let parts = match (&kind, root) {
            (Kind::Transformer, Some(root)) => comfy(Vec::new(), weights.clone(), Some(root))?,
            _ => {
                let base = base.ok_or_else(|| {
                    format!("{repo}: Krea 2 weights, but the model card names no base_model")
                })?;
                let skip: &[&str] = match kind {
                    Kind::Transformer => &["transformer"],
                    Kind::Lora => &[],
                };
                let (lora, dit) = match kind {
                    Kind::Transformer => (Vec::new(), Some(weights.as_path())),
                    Kind::Lora => (vec![(weights.clone(), 1.0)], None),
                };
                let snap = if base.eq_ignore_ascii_case(COMFY_REPO) {
                    Err(format!("{COMFY_REPO} has no diffusers pipeline"))
                } else {
                    hub.fetch_diffusers(&base, skip)
                };
                match snap {
                    Ok(dir) => {
                        return Krea2Pipeline::from_diffusers(&dir, dit, lora)
                            .map(|p| Arc::new(p) as Arc<dyn ImageGenerator>)
                    }
                    Err(e) => {
                        // No mirror fallback for a gated base: the user accepts its license.
                        return Err(format!(
                            "Krea 2 base model {base} could not be fetched ({e}); if it is gated, accept its license at https://huggingface.co/{base} with the HF_TOKEN account"
                        ));
                    }
                }
            }
        };
        Krea2Pipeline::assemble(parts).map(|p| Arc::new(p) as Arc<dyn ImageGenerator>)
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `tensors` as an F32 safetensors file.
    fn write_st(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
        for (name, shape, v) in tensors {
            let start = data.len();
            v.iter().for_each(|x| data.extend(x.to_le_bytes()));
            header.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        std::fs::write(path, out).unwrap();
    }

    /// A 2-block Krea 2 transformer with tiny dims and deterministic weights, original names.
    fn tiny_dit() -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut seed = 12345u64;
        let mut t = |name: &str, shape: &[usize]| {
            let n: usize = shape.iter().product();
            let v = (0..n)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.3
                })
                .collect();
            (name.to_string(), shape.to_vec(), v)
        };
        let (dim, hd, kv, mlp, td, tmlp) = (64, 16, 32, 96, 32, 48);
        let mut out = vec![
            t("first.weight", &[dim, 64]),
            t("first.bias", &[dim]),
            t("tmlp.0.weight", &[dim, 16]),
            t("tmlp.0.bias", &[dim]),
            t("tmlp.2.weight", &[dim, dim]),
            t("tmlp.2.bias", &[dim]),
            t("tproj.1.weight", &[6 * dim, dim]),
            t("tproj.1.bias", &[6 * dim]),
            t("txtfusion.projector.weight", &[1, 12]),
            t("txtmlp.0.scale", &[td]),
            t("txtmlp.1.weight", &[dim, td]),
            t("txtmlp.1.bias", &[dim]),
            t("txtmlp.3.weight", &[dim, dim]),
            t("txtmlp.3.bias", &[dim]),
            t("last.linear.weight", &[64, dim]),
            t("last.linear.bias", &[64]),
            t("last.norm.scale", &[dim]),
            t("last.modulation.lin", &[2, dim]),
        ];
        let mut block = |p: String, d: usize, a: usize, kvw: usize, m: usize, modulated: bool| {
            if modulated {
                out.push(t(&format!("{p}.mod.lin"), &[6 * d]));
            }
            for (n, s) in [
                ("attn.wq.weight", vec![a, d]),
                ("attn.wk.weight", vec![kvw, d]),
                ("attn.wv.weight", vec![kvw, d]),
                ("attn.gate.weight", vec![d, d]),
                ("attn.wo.weight", vec![d, a]),
                ("attn.qknorm.qnorm.scale", vec![hd]),
                ("attn.qknorm.knorm.scale", vec![hd]),
                ("prenorm.scale", vec![d]),
                ("postnorm.scale", vec![d]),
                ("mlp.gate.weight", vec![m, d]),
                ("mlp.up.weight", vec![m, d]),
                ("mlp.down.weight", vec![d, m]),
            ] {
                out.push(t(&format!("{p}.{n}"), &s));
            }
        };
        for i in 0..2 {
            block(format!("blocks.{i}"), dim, dim, kv, mlp, true);
            for kind in ["layerwise_blocks", "refiner_blocks"] {
                block(format!("txtfusion.{kind}.{i}"), td, td, td, tmlp, false);
            }
        }
        out
    }

    #[test]
    fn diffusers_names_are_the_module_tree_paths() {
        for (orig, want) in [
            ("first.bias", "img_in.bias"),
            ("tmlp.0.weight", "time_embed.linear_1.weight"),
            ("tproj.1.bias", "time_mod_proj.bias"),
            (
                "blocks.3.attn.wo.weight",
                "transformer_blocks.3.attn.to_out.0.weight",
            ),
            (
                "blocks.3.attn.gate.weight",
                "transformer_blocks.3.attn.to_gate.weight",
            ),
            (
                "blocks.3.attn.qknorm.qnorm.scale",
                "transformer_blocks.3.attn.norm_q.weight",
            ),
            (
                "blocks.3.mlp.down.weight",
                "transformer_blocks.3.ff.down.weight",
            ),
            (
                "blocks.3.prenorm.scale",
                "transformer_blocks.3.norm1.weight",
            ),
            ("blocks.3.mod.lin", "transformer_blocks.3.scale_shift_table"),
            (
                "txtfusion.layerwise_blocks.1.postnorm.scale",
                "text_fusion.layerwise_blocks.1.norm2.weight",
            ),
            ("txtfusion.projector.weight", "text_fusion.projector.weight"),
            ("txtmlp.0.scale", "txt_in.norm.weight"),
            ("txtmlp.3.bias", "txt_in.linear_2.bias"),
            ("last.norm.scale", "final_layer.norm.weight"),
            ("last.modulation.lin", "final_layer.scale_shift_table"),
            ("last.linear.weight", "final_layer.linear.weight"),
        ] {
            assert_eq!(to_diffusers(orig), want);
        }
    }

    /// The same weights under ComfyUI's names and under diffusers' run identically, a LoRA on both
    /// namings attaches, and a LoRA aimed at a module the transformer lacks is refused.
    #[test]
    fn tiny_transformer_loads_under_both_namings() {
        let dir = tempfile::tempdir().unwrap();
        let orig = tiny_dit();
        let (a, b) = (
            dir.path().join("comfy.safetensors"),
            dir.path().join("diffusers.safetensors"),
        );
        write_st(&a, &orig);
        let renamed: Vec<_> = orig
            .iter()
            .map(|(n, s, v)| (to_diffusers(n), s.clone(), v.clone()))
            .collect();
        write_st(&b, &renamed);
        let mut seed = 7u32;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect()
        };
        let taps: Vec<Vec<f32>> = (0..12).map(|_| rnd(5 * 32)).collect();
        let lat = rnd(16 * 4 * 6);
        let run = |p: &Path, loras: &mut Loras| {
            let dit = Dit::load(p, loras, &Value::Null).unwrap();
            let ctx = dit.fuse(&taps, 5);
            assert_eq!(ctx.len(), 5 * 64);
            dit.forward(&lat, 4, 6, 0.7, &ctx)
        };
        let (ya, yb) = (
            run(&a, &mut Loras::default()),
            run(&b, &mut Loras::default()),
        );
        assert_eq!(ya.len(), 16 * 4 * 6);
        assert!(ya.iter().all(|v| v.is_finite()) && ya.iter().any(|v| v.abs() > 1e-4));
        for (x, y) in ya.iter().zip(&yb) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
        // A LoRA with a bias delta on one module changes the output; either file reads it.
        let lora = dir.path().join("lora.safetensors");
        write_st(
            &lora,
            &[
                (
                    "diffusion_model.blocks.1.attn.wq.lora_down.weight".into(),
                    vec![2, 64],
                    rnd(128),
                ),
                (
                    "diffusion_model.blocks.1.attn.wq.lora_up.weight".into(),
                    vec![64, 2],
                    rnd(128),
                ),
                ("diffusion_model.first.diff_b".into(), vec![64], rnd(64)),
            ],
        );
        for p in [&a, &b] {
            let mut l = Loras::default();
            l.add_file(&lora, 1.0).unwrap();
            let y = run(p, &mut l);
            assert!(l.by_module.is_empty() && l.bias.is_empty());
            assert!(y.iter().zip(&ya).any(|(x, y)| (x - y).abs() > 1e-4));
        }
        let bad = dir.path().join("bad.safetensors");
        write_st(
            &bad,
            &[
                (
                    "diffusion_model.blocks.9.attn.wq.lora_down.weight".into(),
                    vec![2, 64],
                    rnd(128),
                ),
                (
                    "diffusion_model.blocks.9.attn.wq.lora_up.weight".into(),
                    vec![64, 2],
                    rnd(128),
                ),
            ],
        );
        let mut l = Loras::default();
        l.add_file(&bad, 1.0).unwrap();
        Dit::load(&a, &mut l, &Value::Null).unwrap();
        assert_eq!(
            l.by_module.len(),
            1,
            "an unmatched LoRA module stays behind for assemble() to refuse"
        );
    }

    #[test]
    fn comfy_split_repo_selection() {
        let ls = |n: &[&str]| -> Vec<crate::model::hub::RepoFile> {
            n.iter()
                .map(|p| crate::model::hub::RepoFile {
                    path: p.to_string(),
                    size: None,
                })
                .collect()
        };
        let repo = ls(&[
            "diffusion_models/krea2_raw_bf16.safetensors",
            "diffusion_models/krea2_turbo_bf16.safetensors",
            "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
            "loras/krea2_darkbrush.safetensors",
            "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
            "text_encoders/qwen3vl_4b_bf16.safetensors",
            "vae/qwen_image_vae.safetensors",
        ]);
        let sel = |q| crate::model::hub::select_weight_files(&repo, q);
        assert_eq!(
            sel(None).unwrap(),
            [
                "diffusion_models/krea2_turbo_bf16.safetensors",
                "text_encoders/qwen3vl_4b_bf16.safetensors",
                "vae/qwen_image_vae.safetensors"
            ]
        );
        assert_eq!(
            sel(Some("raw")).unwrap()[0],
            "diffusion_models/krea2_raw_bf16.safetensors"
        );
        let quant_only = ls(&[
            "diffusion_models/krea2_turbo_nvfp4.safetensors",
            "text_encoders/qwen3vl_4b_bf16.safetensors",
            "vae/qwen_image_vae.safetensors",
        ]);
        assert!(crate::model::hub::select_weight_files(&quant_only, None)
            .unwrap_err()
            .contains("only the bf16"));
    }

    /// Real Krea 2 weights: ComfyUI's split files cached by `Comfy-Org/Krea-2` (or
    /// `GIBBS_TEST_KREA2` = a diffusers snapshot): one 512x512 image of the Turbo recipe.
    #[test]
    fn real_weights_generate() {
        let hub = HuggingFaceHub::new();
        let id = std::env::var("GIBBS_TEST_KREA2").unwrap_or_else(|_| COMFY_REPO.into());
        let a = hub.locate_model(&id);
        let (Some(dir), Some(_)) = (a.local_dir.as_ref(), a.weights_path.as_ref()) else {
            println!("SKIP: {id} is not cached");
            return;
        };
        let p = if dir.join("model_index.json").is_file() {
            Krea2Pipeline::load(dir)
        } else {
            match load_derived(&a, &hub) {
                Some(Ok(g)) => {
                    let img = g
                        .text_to_image(&ImageRequest {
                            prompt: "a red fox in the snow".into(),
                            seed: 1,
                            ..Default::default()
                        })
                        .unwrap();
                    assert_eq!((img.width, img.height), (512, 512));
                    return;
                }
                r => panic!("{:?}", r.map(|r| r.err())),
            }
        };
        let img = p
            .unwrap()
            .text_to_image(&ImageRequest {
                prompt: "a red fox in the snow".into(),
                seed: 1,
                ..Default::default()
            })
            .unwrap();
        assert_eq!((img.width, img.height), (512, 512));
    }

    /// The real transformer's tensor names and shapes (`GIBBS_TEST_KREA2_DIT`: a ComfyUI
    /// `krea2_*_bf16.safetensors`, or a sparse stand-in carrying its real header) load with every
    /// LoRA in `GIBBS_TEST_KREA2_LORAS` (colon-separated) attached to a module, and one block runs.
    #[test]
    fn real_transformer_structure() {
        let Some(dit) = std::env::var_os("GIBBS_TEST_KREA2_DIT").map(PathBuf::from) else {
            println!("SKIP: set GIBBS_TEST_KREA2_DIT to a krea2 transformer file");
            return;
        };
        let mut loras = Loras::default();
        for l in std::env::var("GIBBS_TEST_KREA2_LORAS")
            .unwrap_or_default()
            .split(':')
            .filter(|s| !s.is_empty())
        {
            loras.add_file(Path::new(l), 1.0).unwrap();
        }
        let n_lora = loras.by_module.len() + loras.bias.len();
        let t = Instant::now();
        let d = Dit::load(&dit, &mut loras, &Value::Null).unwrap();
        println!(
            "loaded in {:.2}s: {} blocks, dim {}, head_dim {}, heads {}/{} kv, mlp {}, {} LoRA modules pending before / {} after",
            t.elapsed().as_secs_f32(),
            d.blocks.len(),
            d.dim,
            d.hd,
            d.blocks[0].attn.heads,
            d.blocks[0].attn.kv,
            d.blocks[0].gate.out_dim(),
            n_lora,
            loras.by_module.len() + loras.bias.len()
        );
        assert!(
            loras.by_module.is_empty() && loras.bias.is_empty(),
            "unattached LoRA modules: {:?}",
            loras.by_module.keys().take(5).collect::<Vec<_>>()
        );
        assert_eq!((d.blocks.len(), d.dim, d.hd, d.chan), (28, 6144, 128, 16));
        let n = 1536;
        let mut x = vec![0.01f32; n * d.dim];
        let tvec = vec![0.0f32; 6 * d.dim];
        let rope = d.rope(&(0..n).map(|i| [0, i / 32, i % 32]).collect::<Vec<_>>());
        let t = Instant::now();
        d.blocks[0].fwd(&mut x, n, n, Some(&rope), Some(&tvec));
        println!(
            "one block over {n} tokens: {:.2}s",
            t.elapsed().as_secs_f32()
        );
    }

    /// The chat-template prefix is the 34 tokens the DiT never sees and the suffix its 5 closing
    /// ones (diffusers `prompt_template_encode_start_idx` / `_num_suffix_tokens`), with the real
    /// Qwen3-VL tokenizer.
    #[test]
    fn template_token_counts() {
        let p = HuggingFaceHub::new()
            .cache_root
            .join(format!("models--{}", TOKENIZER_REPO.replace('/', "--")))
            .join("snapshots/main/tokenizer.json");
        let Ok(tok) = tokenizers::Tokenizer::from_file(&p) else {
            println!("SKIP: {} is not cached", p.display());
            return;
        };
        let n = |s: &str| tok.encode(s, false).unwrap().get_ids().len();
        assert_eq!((n(PREFIX), n(SUFFIX)), (PREFIX_TOKENS, 5));
        let both = tok.encode(format!("{PREFIX}a fox"), false).unwrap();
        assert_eq!(
            &both.get_ids()[..PREFIX_TOKENS],
            tok.encode(PREFIX, false).unwrap().get_ids()
        );
    }
}
