//! Z-Image (diffusers `ZImagePipeline`, e.g. `Tongyi-MAI/Z-Image-Turbo`) on the CPU: a Qwen3 text
//! encoder (hidden_states[-2] of the chat-templated prompt), the single-stream S3-DiT
//! `ZImageTransformer2DModel` (noise/context refiners, 3-axis RoPE, sandwich RMSNorm, tanh-gated
//! adaLN) on bf16 weights (its F32 checkpoint is rounded to bf16 at load, as the reference runs it),
//! FlowMatchEulerDiscrete with a static shift over `linspace(1, 1/N, N)`, and the FLUX 16-channel VAE.

use crate::model::clip::Tensors;
use crate::model::dit::{
    add_gated, component_weights, norm_modulate, read_config, rms_norm, silu_of, timestep_features,
    Lin,
};
use crate::model::flow;
use crate::model::image::ImageBuffer;
use crate::model::loaders::safetensors::SafetensorsWeightReader;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::model::qwen3_5::Qwen3Encoder;
use crate::model::sd::Rng;
use crate::model::text_to_image::{ImageDefaults, ImageGenerator, ImageRequest};
use crate::model::vae::SdVae;
use crate::model::weights::Weights;
use std::path::Path;
use std::time::Instant;

/// Sequences are padded to a multiple of this with learned pad tokens (`SEQ_MULTI_OF`).
const SEQ_MULTI: usize = 32;
/// adaLN conditioning width (`ADALN_EMBED_DIM`).
const ADALN: usize = 256;
/// Prompt tokens kept (pipeline `max_sequence_length`).
const MAX_TOKENS: usize = 512;
const PATCH: usize = 2;

mod graph_program;

/// `n` rounded up to a multiple of [`SEQ_MULTI`].
fn pad_to(n: usize) -> usize {
    n + (SEQ_MULTI - n % SEQ_MULTI) % SEQ_MULTI
}

/// Latents [c, h, w] -> `[hp*wp, PATCH*PATCH*c]` patch rows in (py, px, c) order.
fn patchify(lat: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (p, hp, wp) = (PATCH, h / PATCH, w / PATCH);
    let pc = p * p * c;
    let mut out = vec![0.0f32; hp * wp * pc];
    for i in 0..hp {
        for j in 0..wp {
            let row = &mut out[(i * wp + j) * pc..][..pc];
            for py in 0..p {
                for px in 0..p {
                    for ch in 0..c {
                        row[(py * p + px) * c + ch] = lat[(ch * h + i * p + py) * w + j * p + px];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`]: patch rows -> latents [c, h, w].
fn unpatchify(y: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (p, hp, wp) = (PATCH, h / PATCH, w / PATCH);
    let pc = p * p * c;
    let mut out = vec![0.0f32; c * h * w];
    for i in 0..hp {
        for j in 0..wp {
            let row = &y[(i * wp + j) * pc..][..pc];
            for py in 0..p {
                for px in 0..p {
                    for ch in 0..c {
                        out[(ch * h + i * p + py) * w + j * p + px] = row[(py * p + px) * c + ch];
                    }
                }
            }
        }
    }
    out
}

/// (frame, row, col) RoPE positions of the padded image tokens (pads at the origin) and of the
/// padded caption tokens (1, 2, ...).
fn token_pos(wp: usize, n_img: usize, img_total: usize, cap_total: usize) -> (Vec<[usize; 3]>, Vec<[usize; 3]>) {
    let img = (0..img_total)
        .map(|t| if t < n_img { [cap_total + 1, t / wp, t % wp] } else { [0, 0, 0] })
        .collect();
    (img, (0..cap_total).map(|t| [1 + t, 0, 0]).collect())
}

struct Block {
    /// `None` in the context refiner.
    ada: Option<Lin>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    attn_norm1: Vec<f32>,
    attn_norm2: Vec<f32>,
    ffn_norm1: Vec<f32>,
    ffn_norm2: Vec<f32>,
    w1: Lin,
    w2: Lin,
    w3: Lin,
}

impl Block {
    fn load(
        w: &Weights,
        p: &str,
        dim: usize,
        head_dim: usize,
        modulated: bool,
    ) -> Result<Self, String> {
        let lin = |n: &str| Lin::load(w, &format!("{p}.{n}"));
        let vec = |n: &str, len: usize| w.vec_n(&format!("{p}.{n}.weight"), len);
        Ok(Self {
            ada: if modulated {
                Some(lin("adaLN_modulation.0")?)
            } else {
                None
            },
            q: lin("attention.to_q")?,
            k: lin("attention.to_k")?,
            v: lin("attention.to_v")?,
            o: lin("attention.to_out.0")?,
            norm_q: vec("attention.norm_q", head_dim)?,
            norm_k: vec("attention.norm_k", head_dim)?,
            attn_norm1: vec("attention_norm1", dim)?,
            attn_norm2: vec("attention_norm2", dim)?,
            ffn_norm1: vec("ffn_norm1", dim)?,
            ffn_norm2: vec("ffn_norm2", dim)?,
            w1: lin("feed_forward.w1")?,
            w2: lin("feed_forward.w2")?,
            w3: lin("feed_forward.w3")?,
        })
    }
}

struct ZDit {
    dim: usize,
    heads: usize,
    head_dim: usize,
    in_ch: usize,
    eps: f32,
    t_scale: f32,
    theta: f64,
    axes: Vec<usize>,
    t_mlp0: Lin,
    t_mlp2: Lin,
    x_embedder: Lin,
    x_pad: Vec<f32>,
    cap_norm: Vec<f32>,
    cap_lin: Lin,
    cap_pad: Vec<f32>,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    layers: Vec<Block>,
    final_ada: Lin,
    final_lin: Lin,
    gpu: graph_program::Cache,
}

impl ZDit {
    fn load(dir: &Path) -> Result<Self, String> {
        let cfg = read_config(dir)?;
        let get = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("transformer config: missing {k}"))
        };
        let (dim, heads, in_ch) = (get("dim")?, get("n_heads")?, get("in_channels")?);
        if get("n_kv_heads")? != heads {
            return Err("Z-Image: n_kv_heads != n_heads is not supported".into());
        }
        if cfg["all_patch_size"] != serde_json::json!([PATCH])
            || cfg["all_f_patch_size"] != serde_json::json!([1])
        {
            return Err(format!(
                "Z-Image: only patch size [2] / f_patch [1] is supported, config has {} / {}",
                cfg["all_patch_size"], cfg["all_f_patch_size"]
            ));
        }
        if cfg["qk_norm"].as_bool() != Some(true) || !cfg["siglip_feat_dim"].is_null() {
            return Err(
                "Z-Image: only qk_norm=true without SigLIP (Omni) inputs is supported".into(),
            );
        }
        let head_dim = dim / heads;
        let axes: Vec<usize> = cfg["axes_dims"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        if axes.len() != 3
            || axes.iter().sum::<usize>() != head_dim
            || axes.iter().any(|a| a % 2 != 0)
        {
            return Err(format!(
                "Z-Image: axes_dims {axes:?} must be 3 even dims summing to head_dim {head_dim}"
            ));
        }
        let w = component_weights(dir)?;
        let blocks = |p: &str, n: usize, m: bool| {
            (0..n)
                .map(|i| Block::load(&w, &format!("{p}.{i}"), dim, head_dim, m))
                .collect::<Result<Vec<_>, String>>()
        };
        let cap_dim = get("cap_feat_dim")?;
        let refiners = get("n_refiner_layers")?;
        let dit = Self {
            dim,
            heads,
            head_dim,
            in_ch,
            eps: cfg["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            t_scale: cfg["t_scale"].as_f64().unwrap_or(1000.0) as f32,
            theta: cfg["rope_theta"].as_f64().unwrap_or(256.0),
            axes,
            t_mlp0: Lin::load(&w, "t_embedder.mlp.0")?,
            t_mlp2: Lin::load(&w, "t_embedder.mlp.2")?,
            x_embedder: Lin::load(&w, "all_x_embedder.2-1")?,
            x_pad: w.vec_n("x_pad_token", dim)?,
            cap_norm: w.vec_n("cap_embedder.0.weight", cap_dim)?,
            cap_lin: Lin::load(&w, "cap_embedder.1")?,
            cap_pad: w.vec_n("cap_pad_token", dim)?,
            noise_refiner: blocks("noise_refiner", refiners, true)?,
            context_refiner: blocks("context_refiner", refiners, false)?,
            layers: blocks("layers", get("n_layers")?, true)?,
            final_ada: Lin::load(&w, "all_final_layer.2-1.adaLN_modulation.1")?,
            final_lin: Lin::load(&w, "all_final_layer.2-1.linear")?,
            gpu: Default::default(),
        };
        if dit.x_embedder.in_dim() != PATCH * PATCH * in_ch
            || dit.t_mlp2.out_dim() != ADALN.min(dim)
        {
            return Err("Z-Image: embedder shapes do not match the config".into());
        }
        Ok(dit)
    }

    /// cos/sin tables [n, head_dim/2] for 3-axis positions.
    fn rope(&self, pos: &[[usize; 3]]) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_dim / 2;
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
                // The reference casts the f64 angle to f32 before polar().
                let ang = (p[a] as f64 * f) as f32;
                cos[r * half + k] = ang.cos();
                sin[r * half + k] = ang.sin();
            }
        }
        (cos, sin)
    }

    /// Rotates interleaved pairs of every head of x [n, heads*head_dim].
    fn apply_rope(&self, x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let (hd, half) = (self.head_dim, self.head_dim / 2);
        par_rows(x, self.dim, &|r, row| {
            let (c, s) = (&cos[r * half..][..half], &sin[r * half..][..half]);
            for head in row.chunks_exact_mut(hd) {
                for k in 0..half {
                    let (a, b) = (head[2 * k], head[2 * k + 1]);
                    head[2 * k] = a * c[k] - b * s[k];
                    head[2 * k + 1] = a * s[k] + b * c[k];
                }
            }
        });
    }

    fn block(
        &self,
        b: &Block,
        x: &mut Vec<f32>,
        n: usize,
        rope: &(Vec<f32>, Vec<f32>),
        adaln: &[f32],
    ) {
        let (d, eps) = (self.dim, self.eps);
        let m = b.ada.as_ref().map(|l| l.fwd(adaln, 1));
        let chunk = |k: usize| m.as_ref().map(|m| m[k * d..(k + 1) * d].to_vec());
        // Chunks: scale_msa, gate_msa, scale_mlp, gate_mlp; gates are tanh'd, scales are 1 + s.
        let tanh =
            |g: Option<Vec<f32>>| g.map(|g| g.into_iter().map(f32::tanh).collect::<Vec<f32>>());
        let (s_msa, g_msa, s_mlp, g_mlp) = (chunk(0), tanh(chunk(1)), chunk(2), tanh(chunk(3)));
        let scaled = |v: &mut Vec<f32>, s: &Option<Vec<f32>>| {
            if let Some(s) = s {
                par_rows(v, d, &|_, r| {
                    r.iter_mut().zip(s).for_each(|(v, s)| *v *= 1.0 + s)
                });
            }
        };

        let mut h = x.clone();
        rms_norm(&mut h, d, &b.attn_norm1, eps);
        scaled(&mut h, &s_msa);
        let (mut q, mut k) = (b.q.fwd(&h, n), b.k.fwd(&h, n));
        let v = b.v.fwd(&h, n);
        rms_norm(&mut q, self.head_dim, &b.norm_q, eps);
        rms_norm(&mut k, self.head_dim, &b.norm_k, eps);
        self.apply_rope(&mut q, &rope.0, &rope.1);
        self.apply_rope(&mut k, &rope.0, &rope.1);
        let a = nn::attention(&q, &k, &v, n, n, self.heads, self.head_dim, false);
        let mut a = b.o.fwd(&a, n);
        rms_norm(&mut a, d, &b.attn_norm2, eps);
        add_gated(x, d, g_msa.as_deref(), &a);

        let mut h = x.clone();
        rms_norm(&mut h, d, &b.ffn_norm1, eps);
        scaled(&mut h, &s_mlp);
        let mut g = b.w1.fwd(&h, n);
        nn::silu(&mut g);
        let up = b.w3.fwd(&h, n);
        g.iter_mut().zip(&up).for_each(|(g, u)| *g *= u);
        let mut f = b.w2.fwd(&g, n);
        rms_norm(&mut f, d, &b.ffn_norm2, eps);
        add_gated(x, d, g_mlp.as_deref(), &f);
    }

    /// Raw model output for latents [C, h, w] at noise level `sigma`; `cap` is [n_cap, cap_feat_dim].
    fn forward(
        &self,
        lat: &[f32],
        h: usize,
        w: usize,
        sigma: f32,
        cap: &[f32],
        n_cap: usize,
    ) -> Vec<f32> {
        let (d, c, p) = (self.dim, self.in_ch, PATCH);
        let adaln = self.t_mlp2.fwd(
            &silu_of(
                &self
                    .t_mlp0
                    .fwd(&timestep_features((1.0 - sigma) * self.t_scale, 256), 1),
            ),
            1,
        );

        let (cap_total, cap_pad) = (pad_to(n_cap), self.cap_pad.as_slice());
        let (hp, wp) = (h / p, w / p);
        let n_img = hp * wp;
        let img_total = pad_to(n_img);

        // Image tokens: patch features in (py, px, c) order; pads are the learned x pad token at position 0.
        let mut x = self.x_embedder.fwd(&patchify(lat, c, h, w), n_img);
        for _ in n_img..img_total {
            x.extend_from_slice(&self.x_pad);
        }
        let (img_pos, cap_pos) = token_pos(wp, n_img, img_total, cap_total);
        let img_rope = self.rope(&img_pos);
        for b in &self.noise_refiner {
            self.block(b, &mut x, img_total, &img_rope, &adaln);
        }

        let mut cn = cap.to_vec();
        rms_norm(&mut cn, self.cap_norm.len(), &self.cap_norm, self.eps);
        let mut cx = self.cap_lin.fwd(&cn, n_cap);
        for _ in n_cap..cap_total {
            cx.extend_from_slice(cap_pad);
        }
        let cap_rope = self.rope(&cap_pos);
        for b in &self.context_refiner {
            self.block(b, &mut cx, cap_total, &cap_rope, &adaln);
        }

        // Unified sequence: image tokens first, then caption tokens.
        let n = img_total + cap_total;
        let mut u = x;
        u.extend_from_slice(&cx);
        let rope = self.rope(&[img_pos, cap_pos].concat());
        for b in &self.layers {
            self.block(b, &mut u, n, &rope, &adaln);
        }
        let scale = self.final_ada.fwd(&silu_of(&adaln), 1);
        let y = norm_modulate(&u[..n_img * d], d, 1e-6, &scale, &[]);
        let y = self.final_lin.fwd(&y, n_img);
        unpatchify(&y, c, h, w)
    }
}

pub struct ZImagePipeline {
    encoder: Qwen3Encoder,
    tokenizer: tokenizers::Tokenizer,
    /// hidden_states[-2]: the output of all but the last decoder layer.
    enc_layers: usize,
    dit: ZDit,
    vae: SdVae,
    shift: f64,
}

impl ZImagePipeline {
    /// Loads a diffusers `ZImagePipeline` snapshot directory.
    pub fn load(dir: &Path) -> Result<Self, String> {
        crate::graph::avenues::gate(crate::graph::avenues::Family::DiffusionDit)?;
        let scfg = crate::model::dit::read_scheduler(dir)?;
        if scfg["use_dynamic_shifting"].as_bool() == Some(true) {
            return Err("Z-Image scheduler with use_dynamic_shifting is not supported".into());
        }
        let te = dir.join("text_encoder");
        let tcfg = read_config(&te)?;
        let enc_layers = tcfg["num_hidden_layers"]
            .as_u64()
            .ok_or("text_encoder config: missing num_hidden_layers")?
            as usize
            - 1;
        let tok_path = dir.join("tokenizer").join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("{}: {e}", tok_path.display()))?;
        let vae_dir = dir.join("vae");
        let vcfg = read_config(&vae_dir)?;
        let reader = SafetensorsWeightReader::new(crate::model::hub::component_weights(&vae_dir)?)?;
        let vae = SdVae::from_diffusers(&Tensors::safetensors(&reader)?)?.with_scaling(
            vcfg["scaling_factor"]
                .as_f64()
                .ok_or("vae config: missing scaling_factor")? as f32,
            vcfg["shift_factor"].as_f64().unwrap_or(0.0) as f32,
        );
        let dit = ZDit::load(&dir.join("transformer"))?;
        if vae.latent_channels() != dit.in_ch
            || tcfg["hidden_size"].as_u64() != Some(dit.cap_norm.len() as u64)
        {
            return Err(
                "Z-Image: VAE channels / text encoder width do not match the transformer".into(),
            );
        }
        Ok(Self {
            encoder: Qwen3Encoder::open(&te)?,
            tokenizer,
            enc_layers,
            dit,
            vae,
            shift: scfg["shift"].as_f64().unwrap_or(3.0),
        })
    }

    /// Prompt -> [n, hidden] features of its Qwen3 chat-template rendering (`enable_thinking=True`).
    fn encode(&self, prompt: &str) -> Result<(Vec<f32>, usize), String> {
        let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("Qwen tokenizer: {e}"))?;
        let ids = &enc.get_ids()[..enc.get_ids().len().min(MAX_TOKENS)];
        let hs = self.encoder.hidden_states(ids, self.enc_layers)?;
        Ok((hs, ids.len()))
    }
}

impl ImageGenerator for ZImagePipeline {
    fn summary(&self) -> String {
        format!(
            "Z-Image S3-DiT ({} layers, dim {}), Qwen3 text encoder + 16-ch VAE, bf16",
            self.dit.layers.len(),
            self.dit.dim
        )
    }

    fn defaults(&self) -> ImageDefaults {
        // Turbo settings (8 steps, no CFG); ponytail: CPU-sized 512 default, 1024 is native.
        ImageDefaults {
            width: 512,
            height: 512,
            steps: 8,
            guidance: 0.0,
            multiple: 16,
        }
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        let dflt = self.defaults();
        let (width, height) = (
            req.width.unwrap_or(dflt.width),
            req.height.unwrap_or(dflt.height),
        );
        let steps = req.steps.unwrap_or(dflt.steps);
        let guidance = req.guidance.unwrap_or(dflt.guidance);
        // RoPE tables cover 512 tokens per spatial axis.
        if width == 0
            || height == 0
            || width % 16 != 0
            || height % 16 != 0
            || width > 8192
            || height > 8192
        {
            return Err(format!(
                "Z-Image: width/height must be multiples of 16 up to 8192, got {width}x{height}"
            ));
        }
        if !(1..=200).contains(&steps) {
            return Err(format!("Z-Image: steps must be in 1..=200, got {steps}"));
        }
        let (h, w) = (height / 8, width / 8);
        let start = Instant::now();
        let (cap, n_cap) = self.encode(&req.prompt)?;
        // The pipeline runs CFG whenever guidance_scale > 0: pred = pos + g (pos - neg).
        let neg = if guidance > 0.0 {
            Some(self.encode(req.negative_prompt.as_deref().unwrap_or(""))?)
        } else {
            None
        };
        println!(
            "  text encoder: {n_cap} tokens, {:.2}s",
            start.elapsed().as_secs_f64()
        );
        let sigmas = flow::shifted_sigmas(&flow::linspace_sigmas(steps), self.shift);
        let mut rng = Rng::new(req.seed);
        let mut x: Vec<f32> = (0..self.dit.in_ch * h * w)
            .map(|_| rng.gaussian())
            .collect();
        let gpu = crate::model::dit_graph::use_metal();
        for (i, s) in sigmas.windows(2).enumerate() {
            let start = Instant::now();
            let fwd = |cap: &[f32], n: usize| {
                if gpu {
                    self.dit.forward_gpu(&x, h, w, s[0], cap, n)
                } else {
                    Ok(self.dit.forward(&x, h, w, s[0], cap, n))
                }
            };
            let mut v = fwd(&cap, n_cap)?;
            if let Some((nc, nn)) = &neg {
                let vn = fwd(nc, *nn)?;
                v.iter_mut()
                    .zip(&vn)
                    .for_each(|(p, n)| *p += guidance * (*p - n));
            }
            // The model predicts -velocity.
            v.iter_mut().for_each(|v| *v = -*v);
            flow::euler_step(&mut x, &v, s[0], s[1]);
            println!(
                "  step {}/{steps} (sigma {:.4}): {:.2}s",
                i + 1,
                s[0],
                start.elapsed().as_secs_f64()
            );
        }
        self.vae.decode(&x, h, w)
    }
}
