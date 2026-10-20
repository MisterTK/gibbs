//! Stable Diffusion pipelines on the CPU: SD 1.x / 2.x, SDXL (CLIP-L + OpenCLIP-G, pooled text
//! embeddings and `add_time_ids`), LCM (guidance embedding) and turbo models, from a diffusers repo
//! (`model_index.json` + components), an LDM single-file checkpoint, or the SD 2.1 turbo GGUF; any
//! of them optionally with a LoRA merged in. Sampling uses the repo's declared scheduler
//! ([`crate::model::sd_sched`]) with classifier-free guidance against the negative prompt.

use crate::model::clip::{Act, ClipTextTransformer, Tensors};
use crate::model::hub::{component_weights, HuggingFaceHub};
use crate::model::image::ImageBuffer;
use crate::model::loaders::gguf::GgufWeightReader;
use crate::model::loaders::SafetensorsWeightReader;
use crate::model::lora::{Lora, Target};
use crate::model::sd_sched::{Kind, SchedulerConfig};
use crate::model::text_to_image::{ImageDefaults, ImageRequest};
use crate::model::unet::{alias_diffusers, timestep_embedding, Heads, SdUnet};
use crate::model::vae::SdVae;
use serde_json::{json, Value};
use std::path::Path;

pub(crate) const TRAIN_STEPS: usize = 1000;
const CONTEXT_LEN: usize = 77;
/// SD 1.x / 2.x VAE `scaling_factor` (SDXL's config says 0.13025).
const VAE_SCALE: f32 = 0.18215;
const MAX_STEPS: usize = 100;
const CLIP_EOS: u32 = 49407;
const CLIP_PAD: u32 = 0;

pub struct SdPipeline {
    /// `[clip]` (SD 1.x / 2.x) or `[clip_l, clip_g]` (SDXL).
    encoders: Vec<ClipTextTransformer>,
    unet: SdUnet,
    vae: SdVae,
    sched: SchedulerConfig,
    /// SDXL `force_zeros_for_empty_prompt`: an empty negative prompt is all-zero conditioning.
    zeros_for_empty: bool,
    defaults: ImageDefaults,
    summary: String,
    /// Repo id it was loaded as (`""` for a bare GGUF path).
    id: String,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))
}

/// Pad id of a CLIP tokenizer dir (`special_tokens_map.json`).
fn pad_id(tokenizer_json: &Value) -> u32 {
    let p = &tokenizer_json["pad_token"];
    match p.as_str().or_else(|| p["content"].as_str()) {
        Some("!") => CLIP_PAD,
        _ => CLIP_EOS,
    }
}

/// Defaults from the UNet's `sample_size` and what kind of model this is.
// ponytail: turbo/lightning are recognised by repo id (their configs look like base models); a card field if one appears.
fn defaults(id: &str, sample: usize, xl: bool, lcm: bool, multiple: usize) -> ImageDefaults {
    let l = id.to_lowercase();
    let turbo = ["turbo", "lightning", "hyper-sd"]
        .iter()
        .any(|k| l.contains(k));
    let (steps, guidance) = match (turbo, lcm, xl) {
        (true, _, _) => (1, 0.0),
        (_, true, _) => (4, 8.5),
        (_, _, true) => (25, 5.0),
        _ => (25, 7.5),
    };
    ImageDefaults {
        width: sample,
        height: sample,
        steps,
        guidance,
        multiple,
    }
}

impl SdPipeline {
    pub fn is_sd_checkpoint(reader: &GgufWeightReader) -> bool {
        let mut unet = false;
        let mut vae = false;
        for n in reader.tensor_names() {
            unet |= n.starts_with("model.diffusion_model.");
            vae |= n.starts_with("first_stage_model.decoder.");
        }
        unet && vae
    }

    /// SD 2.1 turbo GGUF (gpustack/stable-diffusion-v2-1-turbo-GGUF): Euler trailing, no CFG.
    pub fn load(gguf_path: &Path, hub: &HuggingFaceHub) -> Result<Self, String> {
        let reader = GgufWeightReader::open(gguf_path)?;
        if !Self::is_sd_checkpoint(&reader) {
            return Err(format!(
                "{} is not a Stable Diffusion checkpoint (needs model.diffusion_model.* and first_stage_model.decoder.*)",
                gguf_path.display()
            ));
        }
        let t = Tensors::gguf(&reader);
        let unet = SdUnet::from_gguf_reader(&reader)?;
        let multiple = 8 * unet.downscale();
        Ok(Self {
            encoders: vec![ClipTextTransformer::load(
                &t,
                "cond_stage_model.transformer.text_model",
                hub,
            )?],
            unet,
            vae: SdVae::from_ldm(&t)?,
            sched: SchedulerConfig::from_json(&json!({
                "_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear",
                "beta_start": 0.00085, "beta_end": 0.012, "timestep_spacing": "trailing"
            }))?,
            zeros_for_empty: false,
            // 4 steps: 1-step Euler multiplies eps error by sigma_max (~14.6), which turns Q4_0 UNet
            // weight error (~10% eps error) into noise.
            defaults: ImageDefaults {
                width: 512,
                height: 512,
                steps: 4,
                guidance: 0.0,
                multiple,
            },
            summary: "Stable Diffusion 2.x GGUF (CLIP + UNet + VAE), EulerDiscreteScheduler".into(),
            id: String::new(),
        })
    }

    /// A diffusers StableDiffusion / StableDiffusionXL / LatentConsistencyModel pipeline directory.
    pub fn load_diffusers(
        dir: &Path,
        hub: &HuggingFaceHub,
        id: &str,
        lora: Option<&Lora>,
    ) -> Result<Self, String> {
        let index = read_json(&dir.join("model_index.json"))?;
        let class = index["_class_name"].as_str().unwrap_or("").to_string();
        let xl = class == "StableDiffusionXLPipeline";
        let te_names: &[&str] = if xl {
            &["text_encoder", "text_encoder_2"]
        } else {
            &["text_encoder"]
        };
        let mut encoders = Vec::new();
        for (i, te) in te_names.iter().enumerate() {
            let tok = dir.join(if i == 0 {
                "tokenizer".to_string()
            } else {
                format!("tokenizer_{}", i + 1)
            });
            let target = if i == 0 {
                Target::TextEncoder
            } else {
                Target::TextEncoder2
            };
            encoders.push(match lora.filter(|l| l.targets(target) > 0) {
                None => ClipTextTransformer::load_diffusers(hub, &dir.join(te), &tok)?,
                Some(l) => {
                    let cfg = read_json(&dir.join(te).join("config.json"))?;
                    let act = if cfg["hidden_act"] == "quick_gelu" {
                        Act::QuickGelu
                    } else {
                        Act::Gelu
                    };
                    let pad = pad_id(&read_json(&tok.join("special_tokens_map.json"))?);
                    let path = component_weights(&dir.join(te))?;
                    let r = SafetensorsWeightReader::new(&path)?;
                    let mut t = Tensors::safetensors(&r)?;
                    l.apply(target, &mut t, "")?;
                    let proj = t
                        .shape("text_projection.weight")
                        .map(|_| "text_projection.weight");
                    ClipTextTransformer::load_with(&t, "text_model", hub, act, pad, proj)?
                }
            });
        }

        let ucfg = read_json(&dir.join("unet/config.json"))?;
        let upath = component_weights(&dir.join("unet"))?;
        if upath.extension().is_some_and(|e| e == "json") {
            return Err(format!(
                "{}: sharded UNet weights are not supported",
                upath.display()
            ));
        }
        let ur = SafetensorsWeightReader::new(&upath)
            .map_err(|e| format!("{}: {e}", upath.display()))?;
        let mut ut = Tensors::safetensors(&ur)?;
        let lpb = ucfg["layers_per_block"].as_u64().unwrap_or(2) as usize;
        alias_diffusers(&mut ut, lpb)?;
        if let Some(l) = lora {
            l.apply(Target::Unet, &mut ut, "")?;
        }
        let unet = SdUnet::load(&ut, "", Heads::from_diffusers(&ucfg)?)?;
        drop(ut);
        drop(ur);

        let vcfg = read_json(&dir.join("vae/config.json"))?;
        if !vcfg["shift_factor"].is_null()
            || vcfg["latent_channels"].as_u64().is_some_and(|c| c != 4)
        {
            return Err("VAE with shift_factor / non-4 latent channels is not an SD VAE".into());
        }
        let vpath = component_weights(&dir.join("vae"))?;
        let vr = SafetensorsWeightReader::new(&vpath)
            .map_err(|e| format!("{}: {e}", vpath.display()))?;
        let vae = SdVae::from_diffusers(&Tensors::safetensors(&vr)?)?;
        let vae_scale = vcfg["scaling_factor"].as_f64().unwrap_or(VAE_SCALE as f64) as f32;

        let sched =
            SchedulerConfig::from_json(&read_json(&dir.join("scheduler/scheduler_config.json"))?)?;
        let lcm = unet.cond_dim().is_some() || sched.kind == Kind::Lcm;
        let sample = ucfg["sample_size"].as_u64().unwrap_or(64) as usize * 8;
        let multiple = 8 * unet.downscale();
        let summary = format!(
            "{class} ({}{}, UNet + VAE{}), {}",
            if xl { "CLIP-L + OpenCLIP-G" } else { "CLIP" },
            if lora.is_some() { " + LoRA" } else { "" },
            if upath.to_string_lossy().contains(".fp16.") {
                ", fp16"
            } else {
                ""
            },
            sched.class
        );
        Self::new(
            encoders,
            unet,
            vae,
            vae_scale,
            sched,
            xl,
            index["force_zeros_for_empty_prompt"]
                .as_bool()
                .unwrap_or(xl),
            defaults(id, sample, xl, lcm, multiple),
            summary,
            id,
        )
    }

    /// An LDM single-file checkpoint (`*.safetensors` holding `model.diffusion_model.*`,
    /// `first_stage_model.*` and the text encoder(s)). Scheduler, VAE scale and default size come
    /// from the reference diffusers repo of its family, as diffusers' `from_single_file` does.
    pub fn load_single_file(
        path: &Path,
        hub: &HuggingFaceHub,
        id: &str,
        lora: Option<&Lora>,
    ) -> Result<Self, String> {
        let r =
            SafetensorsWeightReader::new(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut t = Tensors::safetensors(&r)?;
        let (family, reference) = if t
            .shape("conditioner.embedders.1.model.ln_final.weight")
            .is_some()
        {
            ("SDXL", "stabilityai/stable-diffusion-xl-base-1.0")
        } else if t.shape("cond_stage_model.model.ln_final.weight").is_some() {
            ("SD 2.x", "stabilityai/stable-diffusion-2-1")
        } else if t
            .shape("cond_stage_model.transformer.text_model.final_layer_norm.weight")
            .is_some()
        {
            ("SD 1.x", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        } else {
            return Err(format!(
                "{}: not an SD 1.x / 2.x / XL single-file checkpoint",
                path.display()
            ));
        };
        if t.shape("model.diffusion_model.input_blocks.0.0.weight")
            .is_none()
        {
            return Err(format!(
                "{}: no model.diffusion_model.* UNet",
                path.display()
            ));
        }
        let fetch = |f: &str| hub.fetch_file(reference, f).and_then(|p| read_json(&p));
        let sched = SchedulerConfig::from_json(&fetch("scheduler/scheduler_config.json")?)?;
        let ucfg = fetch("unet/config.json")?;
        let vae_scale = fetch("vae/config.json")?["scaling_factor"]
            .as_f64()
            .unwrap_or(VAE_SCALE as f64) as f32;
        let xl = family == "SDXL";
        let (te1, te2) = match family {
            "SDXL" => {
                alias_openclip(&mut t, "conditioner.embedders.1.model.", "te2.", false)?;
                ("conditioner.embedders.0.transformer.", Some("te2."))
            }
            "SD 2.x" => {
                // diffusers' SD 2.x text encoder is the OpenCLIP tower minus its last block.
                alias_openclip(&mut t, "cond_stage_model.model.", "te.", true)?;
                ("te.", None)
            }
            _ => ("cond_stage_model.transformer.", None),
        };
        if let Some(l) = lora {
            l.apply(Target::Unet, &mut t, "model.diffusion_model.")?;
            l.apply(Target::TextEncoder, &mut t, te1)?;
            if let Some(p) = te2 {
                l.apply(Target::TextEncoder2, &mut t, p)?;
            }
        }
        let (act1, pad1) = if family == "SD 2.x" {
            (Act::Gelu, CLIP_PAD)
        } else {
            (Act::QuickGelu, CLIP_EOS)
        };
        let mut encoders = vec![ClipTextTransformer::load_with(
            &t,
            &format!("{te1}text_model"),
            hub,
            act1,
            pad1,
            None,
        )?];
        if let Some(p) = te2 {
            let proj = format!("{p}text_projection.weight");
            encoders.push(ClipTextTransformer::load_with(
                &t,
                &format!("{p}text_model"),
                hub,
                Act::Gelu,
                CLIP_PAD,
                Some(&proj),
            )?);
        }
        let heads = if family == "SD 1.x" {
            Heads::Count(8)
        } else {
            Heads::Dim(64)
        };
        let unet = SdUnet::load(&t, "model.diffusion_model.", heads)?;
        let vae = SdVae::from_ldm(&t)?;
        let lcm = unet.cond_dim().is_some();
        let sample = ucfg["sample_size"].as_u64().unwrap_or(64) as usize * 8;
        let multiple = 8 * unet.downscale();
        let summary = format!(
            "{family} single-file checkpoint ({}{}), {} (from {reference})",
            path.file_name()
                .map_or(String::new(), |f| f.to_string_lossy().into_owned()),
            if lora.is_some() { " + LoRA" } else { "" },
            sched.class
        );
        Self::new(
            encoders,
            unet,
            vae,
            vae_scale,
            sched,
            xl,
            xl,
            defaults(id, sample, xl, lcm, multiple),
            summary,
            id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        encoders: Vec<ClipTextTransformer>,
        unet: SdUnet,
        vae: SdVae,
        vae_scale: f32,
        sched: SchedulerConfig,
        xl: bool,
        zeros_for_empty: bool,
        defaults: ImageDefaults,
        summary: String,
        id: &str,
    ) -> Result<Self, String> {
        let ctx: usize = encoders.iter().map(|e| e.d_model()).sum();
        let ctx = if xl { ctx } else { encoders[0].d_model() };
        if ctx != unet.context_dim() {
            return Err(format!(
                "text encoders give {ctx}-wide context, UNet cross-attends {}",
                unet.context_dim()
            ));
        }
        if xl {
            let want = encoders[1].pooled_dim() + 6 * 256;
            if unet.add_embed_dim() != Some(want) {
                return Err(format!(
                    "SDXL UNet add_embedding takes {:?}, expected {want}",
                    unet.add_embed_dim()
                ));
            }
        } else if unet.add_embed_dim().is_some() {
            return Err("UNet has an SDXL add_embedding but the pipeline is not SDXL".into());
        }
        Ok(Self {
            encoders,
            unet,
            vae: vae.with_scaling(vae_scale, 0.0),
            sched,
            zeros_for_empty,
            defaults,
            summary,
            id: id.into(),
        })
    }

    pub fn summary(&self) -> String {
        self.summary.clone()
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn defaults(&self) -> ImageDefaults {
        self.defaults
    }

    fn xl(&self) -> bool {
        self.encoders.len() == 2
    }

    /// Prompt -> (context `[77, ctx_dim]`, SDXL pooled embedding).
    fn encode(&self, prompt: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
        if !self.xl() {
            return Ok((self.encoders[0].encode(prompt, false)?.hidden, Vec::new()));
        }
        let a = self.encoders[0].encode(prompt, true)?;
        let b = self.encoders[1].encode(prompt, true)?;
        let (da, db) = (self.encoders[0].d_model(), self.encoders[1].d_model());
        let mut ctx = Vec::with_capacity(CONTEXT_LEN * (da + db));
        for (ra, rb) in a.hidden.chunks_exact(da).zip(b.hidden.chunks_exact(db)) {
            ctx.extend_from_slice(ra);
            ctx.extend_from_slice(rb);
        }
        Ok((ctx, b.pooled))
    }

    /// `[pooled, emb(h), emb(w), emb(0), emb(0), emb(h), emb(w)]` (original size, crop, target size).
    fn add_embeds(&self, pooled: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
        if !self.xl() {
            return None;
        }
        let mut a = pooled.to_vec();
        for v in [h, w, 0, 0, h, w] {
            a.extend(timestep_embedding(v as f32, 256));
        }
        Some(a)
    }

    /// One generation; `init` = (image, strength) for image-to-image.
    pub fn generate(
        &self,
        req: &ImageRequest,
        init: Option<(&ImageBuffer, f32)>,
    ) -> Result<ImageBuffer, String> {
        let d = self.defaults;
        let (mut width, mut height) =
            (req.width.unwrap_or(d.width), req.height.unwrap_or(d.height));
        if let (Some((img, _)), None, None) = (init, req.width, req.height) {
            // The input's own size, scaled down to the default long side when it is over twice that
            // (phone photos): SD at 4x its trained area duplicates subjects and VAE memory explodes.
            let long = img.width.max(img.height) as f32;
            let fit = d.width.max(d.height) as f32;
            let s = if long > 2.0 * fit { fit / long } else { 1.0 };
            let side = |v: usize| ((v as f32 * s) as usize) / d.multiple * d.multiple;
            (width, height) = (side(img.width), side(img.height));
        }
        if width == 0
            || height == 0
            || width % d.multiple != 0
            || height % d.multiple != 0
            || width.max(height) > 2048
        {
            return Err(format!(
                "width/height must be non-zero multiples of {} up to 2048, got {width}x{height}",
                d.multiple
            ));
        }
        let steps = req.steps.unwrap_or(d.steps);
        if !(1..=MAX_STEPS).contains(&steps) {
            return Err(format!("steps must be in 1..={MAX_STEPS}, got {steps}"));
        }
        let guidance = req.guidance.unwrap_or(d.guidance);
        let embedded = self.unet.cond_dim();
        let cfg = guidance > 1.0 && embedded.is_none();
        let t_cond: Option<Vec<f32>> = embedded.map(|dim| guidance_embedding(guidance - 1.0, dim));
        let (h, w) = (height / 8, width / 8);

        let t0 = std::time::Instant::now();
        let (ctx, pooled) = self.encode(&req.prompt)?;
        let add = self.add_embeds(&pooled, height, width);
        let uncond = if cfg {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            let (c, p) = if neg.is_empty() && self.zeros_for_empty {
                (vec![0.0; ctx.len()], vec![0.0; pooled.len()])
            } else {
                self.encode(neg)?
            };
            let a = self.add_embeds(&p, height, width);
            Some((c, a))
        } else {
            None
        };

        let mut sched = self.sched.build(steps)?;
        let mut rng = Rng::new(req.seed);
        let (mut x, start) = match init {
            None => {
                let s = sched.init_sigma() as f32;
                (
                    (0..4 * h * w)
                        .map(|_| rng.gaussian() * s)
                        .collect::<Vec<f32>>(),
                    0,
                )
            }
            Some((img, strength)) => {
                let start = sched.start_index(strength)?;
                let img = if (img.width, img.height) == (width, height) {
                    img.clone()
                } else {
                    img.resize_lanczos(width, height)
                };
                let (mut x0, _, _) = self.vae.encode(&img)?;
                sched.add_noise(&mut x0, start, &mut rng);
                (x0, start)
            }
        };
        let ts = sched.timesteps.clone();
        let t1 = std::time::Instant::now();
        for (i, &t) in ts.iter().enumerate().skip(start) {
            let k = sched.input_scale(i);
            let xin: Vec<f32> = x.iter().map(|v| v * k).collect();
            let run = |c: &[f32], a: Option<&Vec<f32>>| {
                self.unet.forward(
                    &xin,
                    h,
                    w,
                    t as f32,
                    c,
                    CONTEXT_LEN,
                    a.map(Vec::as_slice),
                    t_cond.as_deref(),
                )
            };
            let mut out = run(&ctx, add.as_ref())?;
            if let Some((uc, ua)) = &uncond {
                let u = run(uc, ua.as_ref())?;
                let g = guidance;
                out.iter_mut()
                    .zip(&u)
                    .for_each(|(c, u)| *c = u + g * (*c - u));
            }
            if out.len() != x.len() {
                return Err(format!(
                    "unet returned {} values, expected {}",
                    out.len(),
                    x.len()
                ));
            }
            sched.step(i, &out, &mut x, &mut rng);
        }
        let t2 = std::time::Instant::now();
        let img = self.vae.decode(&x, h, w)?;
        let n = ts.len() - start;
        eprintln!(
            "sd: {width}x{height} encode {:.2}s, denoise {n} steps x {} UNet evals {:.2}s ({:.2} s/step), vae decode {:.2}s",
            (t1 - t0).as_secs_f64(),
            1 + uncond.is_some() as usize,
            (t2 - t1).as_secs_f64(),
            (t2 - t1).as_secs_f64() / n.max(1) as f64,
            t2.elapsed().as_secs_f64()
        );
        Ok(img)
    }
}

/// LCM `get_guidance_scale_embedding(w)`: `[sin(1000 w f), cos(1000 w f)]`, f = 10000^(-i/(half-1)).
fn guidance_embedding(w: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let w = w * 1000.0;
    let f = |i: usize| (-(10000f32.ln()) * i as f32 / (half as f32 - 1.0)).exp();
    let mut e: Vec<f32> = (0..half).map(|i| (w * f(i)).sin()).collect();
    e.extend((0..half).map(|i| (w * f(i)).cos()));
    e.resize(dim, 0.0);
    e
}

/// OpenCLIP text tower under `src` (`token_embedding`, `positional_embedding`, `transformer.resblocks.N`
/// with fused `in_proj`, `ln_final`, `text_projection`) aliased as an HF `CLIPTextModel` under
/// `dst` + `text_model.` (+ `dst` + `text_projection.weight`). `drop_last` leaves out the final block.
pub fn alias_openclip(
    t: &mut Tensors,
    src: &str,
    dst: &str,
    drop_last: bool,
) -> Result<(), String> {
    let p = |n: &str| format!("{src}{n}");
    let q = |n: &str| format!("{dst}text_model.{n}");
    let layers = t
        .names()
        .filter_map(|n| {
            n.strip_prefix(&p("transformer.resblocks."))?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .max()
        .map(|m| m + 1)
        .ok_or_else(|| format!("no OpenCLIP blocks under {src}transformer.resblocks"))?;
    let d = t
        .shape(&p("token_embedding.weight"))
        .ok_or_else(|| format!("missing {src}token_embedding.weight"))?[1];
    t.alias(
        q("embeddings.token_embedding.weight"),
        &p("token_embedding.weight"),
        None,
        false,
    )?;
    t.alias(
        q("embeddings.position_embedding.weight"),
        &p("positional_embedding"),
        None,
        false,
    )?;
    for (a, b) in [("weight", "weight"), ("bias", "bias")] {
        t.alias(
            q(&format!("final_layer_norm.{a}")),
            &p(&format!("ln_final.{b}")),
            None,
            false,
        )?;
    }
    if t.shape(&p("text_projection")).is_some() {
        t.alias(
            format!("{dst}text_projection.weight"),
            &p("text_projection"),
            None,
            true,
        )?;
    }
    for i in 0..layers - usize::from(drop_last) {
        let s = |n: &str| p(&format!("transformer.resblocks.{i}.{n}"));
        let o = |n: &str| q(&format!("encoder.layers.{i}.{n}"));
        for (from, to) in [
            ("ln_1", "layer_norm1"),
            ("ln_2", "layer_norm2"),
            ("attn.out_proj", "self_attn.out_proj"),
            ("mlp.c_fc", "mlp.fc1"),
            ("mlp.c_proj", "mlp.fc2"),
        ] {
            for k in ["weight", "bias"] {
                t.alias(
                    o(&format!("{to}.{k}")),
                    &s(&format!("{from}.{k}")),
                    None,
                    false,
                )?;
            }
        }
        for (j, name) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            t.alias(
                o(&format!("self_attn.{name}.weight")),
                &s("attn.in_proj_weight"),
                Some((j * d, (j + 1) * d)),
                false,
            )?;
            t.alias(
                o(&format!("self_attn.{name}.bias")),
                &s("attn.in_proj_bias"),
                Some((j * d, (j + 1) * d)),
                false,
            )?;
        }
    }
    Ok(())
}

/// splitmix64 + Box-Muller.
pub(crate) struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    pub(crate) fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * self.uniform();
        self.spare = Some((r * th.sin()) as f32);
        (r * th.cos()) as f32
    }
}

/// scaled_linear betas 0.00085..0.012 over 1000 steps (SD 2.x and ModelScope share it).
pub(crate) fn alphas_cumprod() -> Vec<f64> {
    let (a, b) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let mut acc = 1.0;
    (0..TRAIN_STEPS)
        .map(|i| {
            let beta = (a + (b - a) * i as f64 / (TRAIN_STEPS - 1) as f64).powi(2);
            acc *= 1.0 - beta;
            acc
        })
        .collect()
}

/// A tiny SD 1.x-style pipeline (CLIP + 2-level UNet + 2-level VAE, synthetic weights, Euler, CFG) for
/// the parity test and the GPU-only matrix fixture.
pub fn tiny_pipeline() -> Result<SdPipeline, String> {
    use crate::model::diff_fixtures::{tiny_clip, unet_specs, vae_specs};
    use crate::model::diff_graph::testutil::synth;
    let unet_r = synth(&unet_specs(128, 0, 0), 11);
    let unet = SdUnet::load(&Tensors::from_reader(&unet_r)?, "", Heads::Dim(16))?;
    let vae_r = synth(&vae_specs(4), 7);
    let vae = SdVae::from_ldm(&Tensors::from_reader(&vae_r)?)?;
    let sched = SchedulerConfig::from_json(&json!({
        "_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear",
        "beta_start": 0.00085, "beta_end": 0.012, "timestep_spacing": "trailing"
    }))?;
    let d = ImageDefaults { width: 32, height: 32, steps: 3, guidance: 5.0, multiple: 16 };
    SdPipeline::new(
        vec![tiny_clip(Act::QuickGelu, false)],
        unet,
        vae,
        VAE_SCALE,
        sched,
        false,
        false,
        d,
        "tiny".into(),
        "",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_moments() {
        let mut r = Rng::new(42);
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| r.gaussian() as f64).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 0.02 && (var - 1.0).abs() < 0.02,
            "mean {mean} var {var}"
        );
    }

    /// diffusers LatentConsistencyModelPipeline.get_guidance_scale_embedding.
    #[test]
    fn lcm_guidance_embedding_formula() {
        let e = guidance_embedding(7.5, 256);
        let f = |i: f32| (-(10000f32.ln()) * i / 127.0).exp();
        assert!((e[0] - 7500f32.sin()).abs() < 1e-3 && (e[128] - 7500f32.cos()).abs() < 1e-3);
        assert!((e[127] - (7500.0 * f(127.0)).sin()).abs() < 1e-5);
    }

    /// The whole pipeline (CLIP, CFG UNet x3 Euler steps, VAE decode) on Metal vs the native CPU run.
    /// u8 pixels: allow a 2-level difference from f32 rounding.
    #[test]
    fn generate_gpu_matches_cpu() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        use crate::model::diff_graph::with_gpu;
        let pipe = tiny_pipeline().unwrap();
        let req = ImageRequest { prompt: "a lighthouse".into(), seed: 5, ..Default::default() };
        let cpu = pipe.generate(&req, None).unwrap();
        let gpu = with_gpu(|| pipe.generate(&req, None)).unwrap();
        assert_eq!((cpu.width, cpu.height), (gpu.width, gpu.height));
        let mut worst = 0i32;
        for y in 0..cpu.height {
            for x in 0..cpu.width {
                let (a, b) = (cpu.get_pixel(x, y), gpu.get_pixel(x, y));
                worst = worst.max((a.0 as i32 - b.0 as i32).abs()).max((a.1 as i32 - b.1 as i32).abs()).max((a.2 as i32 - b.2 as i32).abs());
            }
        }
        eprintln!("sd pipeline gpu-vs-cpu max pixel diff {worst}");
        assert!(worst <= 2, "max pixel diff {worst}");
    }
}
