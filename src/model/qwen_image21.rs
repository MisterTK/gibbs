//! Qwen-Image 2.1 (diffusers `QwenImage21Pipeline`, e.g. `Qwen/Qwen-Image-2.1`, their GGUFs and
//! finetunes) on the CPU: the Qwen3-VL text tower (last decoder layer's output, before its final
//! norm, of the system-prompt template without the system tokens), the single-stream
//! `QwenImage21Transformer2DModel` (one shared modulation, text modulated from t = 0,
//! block-causal attention: text causal, every image block bidirectional), FlowMatchEuler with
//! an exponential dynamic shift, and the 64-channel 16x VAE ([`crate::model::qwen_vae21`]).
//! Text-to-image only: condition images and the KV prefix cache are not implemented.

mod graph_program;

use crate::model::dit::{
    add_gated, norm_modulate, read_config, read_scheduler, rms_norm, silu_of, timestep_features,
};
use crate::model::flow;
use crate::model::flux::Store;
use crate::model::image::ImageBuffer;
use crate::model::nn;
use crate::model::par::par_rows;
use crate::model::qwen3_5::Qwen3Encoder;
use crate::model::qwen_image::{QLin, Sched};
use crate::model::qwen_vae21::QwenVae21;
use crate::model::sd::Rng;
use crate::model::text_to_image::{ImageDefaults, ImageGenerator, ImageRequest};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

/// The pipeline's system prompt; its tokens lead every encoded prompt and are dropped.
const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";
const THETA: f64 = 10000.0;
/// Pixels per latent token side (16x VAE, unpatched latents).
const TOKEN_PX: usize = 16;

struct Block {
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    gate: QLin,
    up: QLin,
    down: QLin,
}

/// One shared modulation `[scale1 | gate1 | scale2 | gate2]` (gates already `tanh`ed).
struct Mods {
    s1: Vec<f32>,
    g1: Vec<f32>,
    s2: Vec<f32>,
    g2: Vec<f32>,
}

impl Mods {
    fn new(m: &[f32]) -> Self {
        let d = m.len() / 4;
        let tanh = |g: &[f32]| g.iter().map(|v| v.tanh()).collect();
        Self {
            s1: m[..d].to_vec(),
            g1: tanh(&m[d..2 * d]),
            s2: m[2 * d..3 * d].to_vec(),
            g2: tanh(&m[3 * d..]),
        }
    }
}

/// Rotary tables `[n, head_dim / 2]` over the joint sequence (text, then the target image).
pub struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// `QwenImage21Rope` positions `(frame, height, width)` of a text prompt of `n_text` tokens
/// followed by one `h x w` image: text advances all three axes together; the image sits at the
/// frame after the text, on a grid centred on zero.
pub fn t2i_positions(n_text: usize, h: usize, w: usize) -> Vec<[i32; 3]> {
    let mut pos: Vec<[i32; 3]> = (0..n_text as i32).map(|p| [p; 3]).collect();
    let (oy, ox) = ((h - h / 2) as i32, (w - w / 2) as i32);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            pos.push([n_text as i32, y - oy, x - ox]);
        }
    }
    pos
}

impl Rope {
    pub fn new(pos: &[[i32; 3]], axes: &[usize; 3]) -> Self {
        let half: usize = axes.iter().map(|a| a / 2).sum();
        let freqs: Vec<(usize, f32)> = axes
            .iter()
            .enumerate()
            .flat_map(|(a, &d)| {
                (0..d / 2).map(move |i| (a, (1.0 / THETA.powf(2.0 * i as f64 / d as f64)) as f32))
            })
            .collect();
        let (mut cos, mut sin) = (
            vec![0.0f32; pos.len() * half],
            vec![0.0f32; pos.len() * half],
        );
        for (r, p) in pos.iter().enumerate() {
            for (k, &(a, f)) in freqs.iter().enumerate() {
                let ang = p[a] as f32 * f;
                cos[r * half + k] = ang.cos();
                sin[r * half + k] = ang.sin();
            }
        }
        Self { cos, sin }
    }

    /// Rotates interleaved pairs of every head of x `[n, heads * head_dim]` (`apply_rotary_emb_qwen`).
    fn apply(&self, x: &mut [f32], width: usize, head_dim: usize) {
        let half = head_dim / 2;
        par_rows(x, width, &|r, row| {
            let (c, s) = (&self.cos[r * half..][..half], &self.sin[r * half..][..half]);
            for head in row.chunks_exact_mut(head_dim) {
                for k in 0..half {
                    let (a, b) = (head[2 * k], head[2 * k + 1]);
                    head[2 * k] = a * c[k] - b * s[k];
                    head[2 * k + 1] = a * s[k] + b * c[k];
                }
            }
        });
    }
}

/// Block-causal attention of `pre` text rows followed by one image block over `n` rows of
/// `heads * hd`: text rows are causal over the text, image rows see everything.
fn joint_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pre: usize,
    n: usize,
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let d = heads * hd;
    let mut out = nn::attention(
        &q[..pre * d],
        &k[..pre * d],
        &v[..pre * d],
        pre,
        pre,
        heads,
        hd,
        true,
    );
    out.extend(nn::attention(
        &q[pre * d..],
        k,
        v,
        n - pre,
        n,
        heads,
        hd,
        false,
    ));
    out
}

/// Affine-free LayerNorm of the prefix rows (modulated from t = 0) and the target rows (from t).
fn modulated_ln(x: &[f32], d: usize, pre: usize, s0: &[f32], st: &[f32]) -> Vec<f32> {
    let mut h = norm_modulate(&x[..pre * d], d, 1e-6, s0, &[]);
    h.extend(norm_modulate(&x[pre * d..], d, 1e-6, st, &[]));
    h
}

fn gated_add(x: &mut [f32], d: usize, pre: usize, g0: &[f32], gt: &[f32], y: &[f32]) {
    let (a, b) = x.split_at_mut(pre * d);
    add_gated(a, d, Some(g0), &y[..pre * d]);
    add_gated(b, d, Some(gt), &y[pre * d..]);
}

pub struct Dit21 {
    dim: usize,
    heads: usize,
    head_dim: usize,
    in_ch: usize,
    pub axes: [usize; 3],
    img_in: QLin,
    /// `text_norm.weight + 1` (the checkpoint stores the zero-centred scale).
    text_norm: Vec<f32>,
    txt_in: QLin,
    txt_out: QLin,
    t_lin1: QLin,
    t_lin2: QLin,
    modulation: QLin,
    blocks: Vec<Block>,
    norm_out: QLin,
    proj_out: QLin,
    /// Metal programs per (text tokens, h, w) (GPU-only runs).
    pub gpu: graph_program::Cache,
}

impl Dit21 {
    /// Reads the transformer; sizes come from the tensors. `cfg` (a diffusers `config.json`, absent
    /// for GGUFs) only supplies what tensors cannot: the rotary axes and the unsupported options.
    pub fn load(s: &Store, cfg: Option<&Value>) -> Result<Self, String> {
        let null = Value::Null;
        let cfg = cfg.unwrap_or(&null);
        if cfg["causal_condition"] == false || cfg["patch_size"].as_u64().is_some_and(|p| p != 1) {
            return Err(
                "Qwen-Image 2.1: only causal_condition=true with patch_size 1 is supported".into(),
            );
        }
        let layers = (0..)
            .take_while(|i| s.has(&format!("transformer_blocks.{i}.attn.to_q.weight")))
            .count();
        if layers == 0 {
            return Err("Qwen-Image 2.1 transformer: no transformer_blocks".into());
        }
        let img_in = QLin::load(s, "img_in.weight")?;
        let dim = img_in.n;
        let norm_q0 = s.vec("transformer_blocks.0.attn.norm_q.weight")?;
        let head_dim = norm_q0.len();
        if head_dim == 0 || dim % head_dim != 0 {
            return Err(format!(
                "Qwen-Image 2.1: dim {dim} is not a multiple of head_dim {head_dim}"
            ));
        }
        let axes: Vec<usize> = cfg["axes_dims_rope"]
            .as_array()
            .map_or(vec![16, 56, 56], |a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            });
        let axes: [usize; 3] = axes
            .try_into()
            .map_err(|_| "axes_dims_rope must have 3 entries")?;
        if axes.iter().sum::<usize>() != head_dim || axes.iter().any(|a| a % 2 != 0) {
            return Err(format!("Qwen-Image 2.1: axes_dims_rope {axes:?} must be 3 even sizes summing to head_dim {head_dim}"));
        }
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("transformer_blocks.{i}");
            let lin = |n: &str| QLin::load(s, &format!("{p}.{n}.weight"));
            // Some GGUFs fuse the SwiGLU's `gate_layer` and `proj` into one `gate_up` (gate first).
            let (gate, up) = if s.has(&format!("{p}.img_mlp.gate_up.weight")) {
                let f = lin("img_mlp.gate_up")?;
                (f.rows(0, f.n / 2)?, f.rows(f.n / 2, f.n / 2)?)
            } else {
                (lin("img_mlp.gate_layer")?, lin("img_mlp.proj")?)
            };
            blocks.push(Block {
                norm_q: s.vec(&format!("{p}.attn.norm_q.weight"))?,
                norm_k: s.vec(&format!("{p}.attn.norm_k.weight"))?,
                q: lin("attn.to_q")?,
                k: lin("attn.to_k")?,
                v: lin("attn.to_v")?,
                o: lin("attn.to_out.0")?,
                gate,
                up,
                down: lin("img_mlp.out")?,
            });
        }
        let dit = Self {
            dim,
            heads: dim / head_dim,
            head_dim,
            in_ch: img_in.k,
            axes,
            img_in,
            text_norm: s
                .vec("txt_in.text_norm.weight")?
                .into_iter()
                .map(|w| w + 1.0)
                .collect(),
            txt_in: QLin::load(s, "txt_in.in_layer.weight")?,
            txt_out: QLin::load(s, "txt_in.out_layer.weight")?,
            t_lin1: QLin::load(s, "time_text_embed.timestep_embedder.linear_1.weight")?,
            t_lin2: QLin::load(s, "time_text_embed.timestep_embedder.linear_2.weight")?,
            modulation: QLin::load(s, "modulation.1.weight")?,
            blocks,
            norm_out: QLin::load(s, "norm_out.linear.weight")?,
            proj_out: QLin::load(s, "proj_out.weight")?,
            gpu: Default::default(),
        };
        if dit.modulation.n != 4 * dim
            || dit.t_lin2.n != dim
            || dit.proj_out.k != dim
            || dit.txt_out.n != dim
        {
            return Err(
                "Qwen-Image 2.1: embedder shapes do not match the transformer width".into(),
            );
        }
        Ok(dit)
    }

    fn text_width(&self) -> usize {
        self.text_norm.len()
    }

    /// `txt_in` (zero-centred RMSNorm, linear, tanh-GELU, linear) of text features `[n, text_width]`.
    pub fn project_text(&self, txt: &[f32], n: usize) -> Vec<f32> {
        let mut t = txt.to_vec();
        rms_norm(&mut t, self.text_width(), &self.text_norm, 1e-6);
        let mut h = self.txt_in.fwd(&t, n);
        nn::gelu_tanh(&mut h);
        self.txt_out.fwd(&h, n)
    }

    /// Timestep embedding of `sigma` (the model sees `sigma * 1000`, `TemporalTimesteps` scales by
    /// 1000 again on `timestep / 1000`).
    fn temb(&self, sigma: f32) -> Vec<f32> {
        let mut h = self.t_lin1.fwd(&timestep_features(1000.0 * sigma, 256), 1);
        nn::silu(&mut h);
        self.t_lin2.fwd(&h, 1)
    }

    fn block(&self, b: &Block, x: &mut [f32], pre: usize, rope: &Rope, m0: &Mods, mt: &Mods) {
        let (d, hd) = (self.dim, self.head_dim);
        let n = x.len() / d;
        let h = modulated_ln(x, d, pre, &m0.s1, &mt.s1);
        let (mut q, mut k, v) = (b.q.fwd(&h, n), b.k.fwd(&h, n), b.v.fwd(&h, n));
        rms_norm(&mut q, hd, &b.norm_q, 1e-6);
        rms_norm(&mut k, hd, &b.norm_k, 1e-6);
        rope.apply(&mut q, d, hd);
        rope.apply(&mut k, d, hd);
        let a =
            b.o.fwd(&joint_attention(&q, &k, &v, pre, n, self.heads, hd), n);
        gated_add(x, d, pre, &m0.g1, &mt.g1, &a);

        let h = modulated_ln(x, d, pre, &m0.s2, &mt.s2);
        let mut g = b.gate.fwd(&h, n);
        nn::silu(&mut g);
        let u = b.up.fwd(&h, n);
        g.iter_mut().zip(&u).for_each(|(g, u)| *g *= u);
        gated_add(x, d, pre, &m0.g2, &mt.g2, &b.down.fwd(&g, n));
    }

    /// Velocity of the packed latents `lat [n_img, in_ch]` at `sigma`, given the projected text
    /// `ctx [n_text, dim]`.
    pub fn forward(&self, lat: &[f32], ctx: &[f32], rope: &Rope, sigma: f32) -> Vec<f32> {
        let d = self.dim;
        let (pre, n_img) = (ctx.len() / d, lat.len() / self.in_ch);
        let mut x = ctx.to_vec();
        x.extend(self.img_in.fwd(lat, n_img));
        // causal_condition: text modulates from t = 0, the target image from t.
        let temb = self.temb(sigma);
        let (m0, mt) = (
            Mods::new(&self.modulation.fwd(&silu_of(&self.temb(0.0)), 1)),
            Mods::new(&self.modulation.fwd(&silu_of(&temb), 1)),
        );
        for b in &self.blocks {
            self.block(b, &mut x, pre, rope, &m0, &mt);
        }
        let scale = self.norm_out.fwd(&silu_of(&temb), 1);
        let y = norm_modulate(&x[pre * d..], d, 1e-6, &scale, &[]);
        self.proj_out.fwd(&y, n_img)
    }
}

/// Everything after the text encoder: transformer, VAE and sampling schedule.
pub(crate) struct Core {
    dit: Dit21,
    vae: QwenVae21,
    sched: Sched,
    /// Raw sampling sigmas a distilled finetune's card prescribes; `None`: `linspace(1, 1/steps)`.
    sigmas: Option<Vec<f64>>,
}

/// A prompt's text-encoder features `[n, text_width]` and `n`.
type Features = (Vec<f32>, usize);

impl Core {
    /// Checks the transformer, VAE and (text width) `text_width` against each other.
    fn new(
        dit: Dit21,
        vae: QwenVae21,
        sched: Sched,
        sigmas: Option<Vec<f64>>,
        text_width: usize,
    ) -> Result<Self, String> {
        if vae.latent_channels() != dit.in_ch || dit.proj_out.n != dit.in_ch {
            return Err(format!(
                "Qwen-Image 2.1: VAE has {} latent channels, the transformer takes {} and emits {}",
                vae.latent_channels(),
                dit.in_ch,
                dit.proj_out.n
            ));
        }
        if text_width != dit.text_width() {
            return Err(
                "Qwen-Image 2.1: text encoder width does not match the transformer's context width"
                    .into(),
            );
        }
        Ok(Self {
            dit,
            vae,
            sched,
            sigmas,
        })
    }

    fn defaults(&self) -> ImageDefaults {
        // The model card's 40 steps without guidance (a distilled finetune's card lists its own
        // sigmas); ponytail: CPU-sized 512 default, 1024 is the card's `output_resolution`.
        ImageDefaults {
            width: 512,
            height: 512,
            steps: self.sigmas.as_ref().map_or(40, Vec::len),
            guidance: 1.0,
            multiple: 2 * TOKEN_PX,
        }
    }

    /// Denoises from noise and decodes. `encode` turns a prompt into text-encoder features; the
    /// negative prompt (" " when none is given) is encoded only for true CFG (`guidance > 1`).
    fn generate(
        &self,
        req: &ImageRequest,
        encode: &dyn Fn(&str) -> Result<Features, String>,
    ) -> Result<ImageBuffer, String> {
        let d = self.defaults();
        let (width, height) = (req.width.unwrap_or(d.width), req.height.unwrap_or(d.height));
        let steps = req.steps.unwrap_or(d.steps);
        let guidance = req.guidance.unwrap_or(d.guidance);
        if width == 0 || height == 0 || width % d.multiple != 0 || height % d.multiple != 0 {
            return Err(format!("Qwen-Image 2.1: width/height must be non-zero multiples of {}, got {width}x{height}", d.multiple));
        }
        if steps == 0 {
            return Err("Qwen-Image 2.1: steps must be at least 1".into());
        }
        let start = Instant::now();
        let (h, w) = (height / TOKEN_PX, width / TOKEN_PX);
        let (n_img, ch) = (h * w, self.dit.in_ch);
        // Text features -> (projected context, rotary tables of the joint sequence); on the GPU
        // the raw features go to the program, which projects them itself.
        let gpu = crate::model::dit_graph::use_metal();
        let prepare = |(txt, n): Features| {
            if gpu {
                (txt, None)
            } else {
                (
                    self.dit.project_text(&txt, n),
                    Some(Rope::new(&t2i_positions(n, h, w), &self.dit.axes)),
                )
            }
        };
        let velocity = |(ctx, rope): &(Vec<f32>, Option<Rope>), x: &[f32], sigma: f32| match rope {
            Some(rope) => Ok(self.dit.forward(x, ctx, rope, sigma)),
            None => self.dit.forward_gpu(&self.dit.gpu, x, ctx, h, w, sigma),
        };
        let pos = prepare(encode(&req.prompt)?);
        let neg = (guidance > 1.0)
            .then(|| encode(req.negative_prompt.as_deref().unwrap_or(" ")).map(prepare))
            .transpose()?;
        println!("  text encoder: {:.2}s", start.elapsed().as_secs_f64());

        let base: Vec<f64> = match &self.sigmas {
            Some(s) if req.steps.is_none_or(|n| n == s.len()) => s.clone(),
            _ => flow::linspace_sigmas(steps),
        };
        let sigmas = self.sched.sigmas(&base, n_img);
        let mut rng = Rng::new(req.seed);
        let noise: Vec<f32> = (0..ch * n_img).map(|_| rng.gaussian()).collect();
        // Channel-first noise `[ch, h*w]` -> tokens `[h*w, ch]` (`_pack_latents`).
        let mut x = nn::transpose(&noise, ch, n_img);
        for (i, s) in sigmas.windows(2).enumerate() {
            let t = Instant::now();
            let mut v = velocity(&pos, &x, s[0])?;
            if let Some(neg) = &neg {
                let vn = velocity(neg, &x, s[0])?;
                v.iter_mut()
                    .zip(&vn)
                    .for_each(|(v, n)| *v = n + guidance * (*v - n));
            }
            flow::euler_step(&mut x, &v, s[0], s[1]);
            println!(
                "  step {}/{} (sigma {:.4}): {:.2}s",
                i + 1,
                sigmas.len() - 1,
                s[0],
                t.elapsed().as_secs_f64()
            );
        }
        self.vae.decode(&nn::transpose(&x, n_img, ch), h, w)
    }
}

pub struct QwenImage21Pipeline {
    enc: Qwen3Encoder,
    tok: tokenizers::Tokenizer,
    enc_layers: usize,
    /// Tokens of the system message that lead every prompt (dropped from the hidden states).
    drop: usize,
    core: Core,
    label: String,
}

impl QwenImage21Pipeline {
    /// Loads a diffusers `QwenImage21Pipeline` snapshot directory.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_with(dir, None, None, None)
    }

    /// `dir` is the base repo snapshot (text encoder, processor, VAE, scheduler, and the transformer
    /// unless `transformer` names a replacement file: a GGUF or single-file finetune). `scheduler`
    /// replaces the base's scheduler config, `sigmas` is the finetune's default raw schedule.
    pub fn load_with(
        dir: &Path,
        transformer: Option<&Path>,
        scheduler: Option<Value>,
        sigmas: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        let te = dir.join("text_encoder");
        let tcfg = read_config(&te)?;
        if tcfg["model_type"] != "qwen3_vl" {
            return Err(format!(
                "Qwen-Image 2.1: text encoder is {}, expected qwen3_vl",
                tcfg["model_type"]
            ));
        }
        let text = &tcfg["text_config"];
        let enc_layers = text["num_hidden_layers"]
            .as_u64()
            .ok_or("text_encoder config: missing text_config.num_hidden_layers")?
            as usize;
        let tok_path = dir.join("processor").join("tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("{}: {e}", tok_path.display()))?;
        let (path, cfg) = match transformer {
            Some(p) => (p.to_path_buf(), None),
            None => {
                let t = dir.join("transformer");
                (
                    crate::model::hub::component_weights(&t)?,
                    Some(read_config(&t)?),
                )
            }
        };
        let dit = Dit21::load(&Store::open(&path)?, cfg.as_ref())?;
        let label = format!(
            "Qwen-Image 2.1 ({} single-stream blocks, dim {}; {}), Qwen3-VL text encoder + 64-ch VAE",
            dit.blocks.len(),
            dit.dim,
            path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())
        );
        let sched = Sched::from_config(&scheduler.map_or_else(|| read_scheduler(dir), Ok)?);
        let width = text["hidden_size"]
            .as_u64()
            .ok_or("text_encoder config: missing text_config.hidden_size")?
            as usize;
        let core = Core::new(
            dit,
            QwenVae21::load(&dir.join("vae"))?,
            sched,
            sigmas,
            width,
        )?;
        // The system message's tokens lead every prompt and are dropped from the hidden states.
        let sys = format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n");
        let drop = tok
            .encode(sys, false)
            .map_err(|e| format!("Qwen tokenizer: {e}"))?
            .get_ids()
            .len();
        Ok(Self {
            enc: Qwen3Encoder::open(&te)?,
            tok,
            enc_layers,
            drop,
            core,
            label,
        })
    }
}

/// Prompt -> the text tower's last decoder layer output (before its final norm) for the
/// system-prompt template, minus the system tokens.
fn encode(
    enc: &Qwen3Encoder,
    tok: &tokenizers::Tokenizer,
    layers: usize,
    drop: usize,
    prompt: &str,
) -> Result<Features, String> {
    // The pipeline replaces an empty prompt by a space: Qwen has no BOS to read otherwise.
    let p = if prompt.is_empty() { " " } else { prompt };
    let text = format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n");
    let ids = tok
        .encode(text, false)
        .map_err(|e| format!("Qwen tokenizer: {e}"))?;
    let ids = ids.get_ids();
    let hs = enc.hidden_states(ids, layers)?;
    let w = hs.len() / ids.len();
    Ok((hs[drop * w..].to_vec(), ids.len() - drop))
}

impl ImageGenerator for QwenImage21Pipeline {
    fn summary(&self) -> String {
        self.label.clone()
    }

    fn defaults(&self) -> ImageDefaults {
        self.core.defaults()
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        self.core.generate(req, &|p| {
            encode(&self.enc, &self.tok, self.enc_layers, self.drop, p)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_tokens_are_centred_after_the_text() {
        // 3 text tokens, then a 2x4 image: text (p, p, p); image frame 3, rows -1..1, columns -2..2.
        let pos = t2i_positions(3, 2, 4);
        assert_eq!(&pos[..3], &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(pos.len(), 3 + 8);
        assert_eq!(pos[3], [3, -1, -2]);
        assert_eq!(pos[6], [3, -1, 1]);
        assert_eq!(pos[10], [3, 0, 1]);
    }

    /// `block-causal` split attention against one masked softmax over the whole sequence.
    #[test]
    fn text_causal_image_full_attention_matches_the_mask() {
        let (heads, hd, pre, n) = (2, 4, 3, 7);
        let d = heads * hd;
        let r = |s: u32| -> Vec<f32> {
            (0..n * d)
                .map(|i| {
                    (((i as u32).wrapping_mul(2654435761u32) ^ s).wrapping_mul(40503) % 1000) as f32
                        / 500.0
                        - 1.0
                })
                .collect()
        };
        let (q, k, v) = (r(1), r(2), r(3));
        let got = joint_attention(&q, &k, &v, pre, n, heads, hd);
        // Reference: row i attends to j <= i, or to every j when i is an image row.
        let mut want = vec![0.0f32; n * d];
        for i in 0..n {
            for h in 0..heads {
                let allowed = |j: usize| j <= i || i >= pre;
                let sc: Vec<f32> = (0..n)
                    .map(|j| {
                        if allowed(j) {
                            (0..hd)
                                .map(|c| q[i * d + h * hd + c] * k[j * d + h * hd + c])
                                .sum::<f32>()
                                / (hd as f32).sqrt()
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for c in 0..hd {
                    want[i * d + h * hd + c] =
                        (0..n).map(|j| e[j] / z * v[j * d + h * hd + c]).sum();
                }
            }
        }
        assert!(
            got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5),
            "{got:?} vs {want:?}"
        );
    }

    #[test]
    fn rope_rotates_pairs_and_is_identity_at_position_zero() {
        let rope = Rope::new(&[[0, 0, 0], [1, 2, 3]], &[2, 2, 4]);
        let mut x: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect(); // 2 rows, 1 head of 8
        let orig = x.clone();
        rope.apply(&mut x, 8, 8);
        assert_eq!(&x[..8], &orig[..8]);
        // Row 1, pair 0 (frame axis, inv_freq 1): rotated by 1 rad.
        let (a, b) = (orig[8], orig[9]);
        assert!((x[8] - (a * 1f32.cos() - b * 1f32.sin())).abs() < 1e-5);
        assert!((x[9] - (a * 1f32.sin() + b * 1f32.cos())).abs() < 1e-5);
    }

    fn env_path(k: &str) -> Option<std::path::PathBuf> {
        std::env::var_os(k).map(Into::into)
    }

    /// Stage 1 of the real-weight run, for hosts that cannot hold the 17.5 GB text encoder and the
    /// transformer on disk together: `GIBBS_QWEN21_ENCODER_GGUF` (a `qwen3vl` GGUF, the same
    /// weights as the pipeline's `text_encoder/`), `GIBBS_QWEN21_TOKENIZER` (`processor/tokenizer.json`)
    /// and `GIBBS_QWEN21_PROMPT` -> text-tower features in `GIBBS_QWEN21_FEATURES` (f32 LE).
    #[test]
    fn real_text_tower_features() {
        let (Some(gguf), Some(tok), Some(out), Ok(prompt)) = (
            env_path("GIBBS_QWEN21_ENCODER_GGUF"),
            env_path("GIBBS_QWEN21_TOKENIZER"),
            env_path("GIBBS_QWEN21_FEATURES"),
            std::env::var("GIBBS_QWEN21_PROMPT"),
        ) else {
            eprintln!("SKIP: GIBBS_QWEN21_{{ENCODER_GGUF,TOKENIZER,FEATURES,PROMPT}} not set");
            return;
        };
        let enc = Qwen3Encoder::from_gguf(&gguf).unwrap();
        let tok = tokenizers::Tokenizer::from_file(&tok).unwrap();
        let sys = format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n");
        let drop = tok.encode(sys, false).unwrap().get_ids().len();
        let (feats, n) = encode(&enc, &tok, 36, drop, &prompt).unwrap();
        assert!(n > 0 && feats.iter().all(|v| v.is_finite()));
        println!(
            "{n} prompt tokens after dropping {drop} system tokens; |feat| mean {:.3}",
            feats.iter().map(|v| v.abs()).sum::<f32>() / feats.len() as f32
        );
        std::fs::write(
            &out,
            feats
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
    }

    /// Stage 2: `GIBBS_QWEN21_TRANSFORMER` (a Qwen-Image 2.1 GGUF or safetensors), `GIBBS_QWEN21_VAE`
    /// (diffusers `vae/` dir), the stage-1 features, and optionally `GIBBS_QWEN21_SCHEDULER` (a
    /// scheduler_config.json), `GIBBS_QWEN21_SIGMAS` (comma-separated), `GIBBS_QWEN21_SIZE`
    /// (pixels, default 512) -> a PNG at `GIBBS_QWEN21_PNG`.
    #[test]
    fn real_transformer_and_vae_generate() {
        let (Some(tr), Some(vae), Some(feats), Some(png)) = (
            env_path("GIBBS_QWEN21_TRANSFORMER"),
            env_path("GIBBS_QWEN21_VAE"),
            env_path("GIBBS_QWEN21_FEATURES"),
            env_path("GIBBS_QWEN21_PNG"),
        ) else {
            eprintln!("SKIP: GIBBS_QWEN21_{{TRANSFORMER,VAE,FEATURES,PNG}} not set");
            return;
        };
        let raw = std::fs::read(feats).unwrap();
        let f: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = f.len() / 4096;
        let sched = env_path("GIBBS_QWEN21_SCHEDULER").map_or_else(
            || serde_json::json!({"use_dynamic_shifting": true, "max_image_seq_len": 8192, "max_shift": 0.9, "shift_terminal": 0.02}),
            |p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap(),
        );
        let sigmas = std::env::var("GIBBS_QWEN21_SIGMAS")
            .ok()
            .map(|s| s.split(',').map(|v| v.trim().parse().unwrap()).collect());
        let dit = Dit21::load(&Store::open(&tr).unwrap(), None).unwrap();
        let core = Core::new(
            dit,
            QwenVae21::load(&vae).unwrap(),
            Sched::from_config(&sched),
            sigmas,
            4096,
        )
        .unwrap();
        let size = std::env::var("GIBBS_QWEN21_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let req = ImageRequest {
            prompt: "x".into(),
            width: Some(size),
            height: Some(size),
            seed: 7,
            ..Default::default()
        };
        let img = core.generate(&req, &|_| Ok((f.clone(), n))).unwrap();
        std::fs::write(&png, img.to_png()).unwrap();
        println!("wrote {} ({}x{})", png.display(), img.width, img.height);
    }
}
