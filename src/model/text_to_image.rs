//! Text-to-image pipelines behind one interface, so the server, the TUI and the CLI drive every
//! architecture (SD GGUF, diffusers SD/SDXL, FLUX, ...) the same way.

use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::model::image::ImageBuffer;
use crate::model::loaders::WeightReader;
use crate::model::lora::{self, Lora};
use crate::model::sd::SdPipeline;
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;

/// One generation. `None` fields take the pipeline's [`ImageDefaults`].
#[derive(Debug, Clone, Default)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    /// Classifier-free guidance scale; distilled/turbo models ignore it.
    pub guidance: Option<f32>,
    pub seed: u64,
}

/// What a pipeline runs when the request leaves a field unset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageDefaults {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    /// Pixel sides must be multiples of this.
    pub multiple: usize,
}

pub trait ImageGenerator: Send + Sync {
    /// Architecture and weights, for logs and the TUI (e.g. `SD 2.1 GGUF Q4_0, CLIP + UNet + VAE`).
    fn summary(&self) -> String;
    /// Repo id it was loaded as, when known (listed by `/v1/models` and the TUI after a hot swap).
    fn model_id(&self) -> Option<String> {
        None
    }
    fn defaults(&self) -> ImageDefaults;
    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String>;
    /// Denoises from `init` noised to `strength` (0..=1).
    fn image_to_image(
        &self,
        _req: &ImageRequest,
        _init: &ImageBuffer,
        _strength: f32,
    ) -> Result<ImageBuffer, String> {
        Err(format!(
            "{} does not support image-to-image",
            self.summary()
        ))
    }
}

/// The active image model, shared by the HTTP server and the TUI (hot-swapped by the picker).
pub type ImageSlot = Arc<RwLock<Option<Arc<dyn ImageGenerator>>>>;

impl ImageGenerator for SdPipeline {
    fn summary(&self) -> String {
        SdPipeline::summary(self)
    }

    fn model_id(&self) -> Option<String> {
        Some(self.id().to_string()).filter(|s| !s.is_empty())
    }

    fn defaults(&self) -> ImageDefaults {
        SdPipeline::defaults(self)
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        self.generate(req, None)
    }

    fn image_to_image(
        &self,
        req: &ImageRequest,
        init: &ImageBuffer,
        strength: f32,
    ) -> Result<ImageBuffer, String> {
        self.generate(req, Some((init, strength)))
    }
}

/// diffusers `_class_name`s the Stable Diffusion loader runs.
pub const SD_PIPELINES: &[&str] = &[
    "StableDiffusionPipeline",
    "StableDiffusionXLPipeline",
    "LatentConsistencyModelPipeline",
];

/// The repo's diffusers pipeline class, from `<dir>/model_index.json`.
pub fn pipeline_class(dir: &Path) -> Option<String> {
    let s = std::fs::read_to_string(dir.join("model_index.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    v["_class_name"].as_str().map(str::to_string)
}

fn arc<G: ImageGenerator + 'static>(
    g: Result<G, String>,
) -> Result<Arc<dyn ImageGenerator>, String> {
    g.map(|g| Arc::new(g) as Arc<dyn ImageGenerator>)
}

/// Loads `artifacts` as an image pipeline. `None`: not an image repo (the caller tries other model
/// kinds). Diffusers repos dispatch on `model_index.json`'s `_class_name`; a root safetensors is an
/// SD single-file checkpoint or an SD LoRA (its card's `base_model` is fetched and loaded under it);
/// an SD GGUF is the SD 2.x turbo pipeline.
pub fn load_image_model(
    artifacts: &ModelArtifacts,
    hub: &HuggingFaceHub,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let dir = artifacts.local_dir.as_deref()?;
    let id = artifacts.model_id.as_str();
    if let Some(class) = pipeline_class(dir) {
        return Some(match class.as_str() {
            c if SD_PIPELINES.contains(&c) => arc(SdPipeline::load_diffusers(dir, hub, id, None)),
            "StableDiffusion3Pipeline" => arc(crate::model::sd3::Sd3Pipeline::load(dir, hub)),
            "ZImagePipeline" => arc(crate::model::zimage::ZImagePipeline::load(dir)),
            "SanaPipeline" => arc(crate::model::sana::SanaPipeline::load(dir)),
            "Krea2Pipeline" => arc(crate::model::krea2::Krea2Pipeline::load(dir)),
            "QwenImage21Pipeline" => arc(crate::model::qwen_image21::QwenImage21Pipeline::load(dir)),
            "QwenImagePipeline" => arc(crate::model::qwen_image_v1::QwenImagePipeline::load(dir)),
            "FluxPipeline" | "ChromaPipeline" | "Flux2KleinPipeline" => arc(crate::model::flux::FluxPipeline::load(dir, hub)),
            c if c.contains("Inpaint") => Err(format!(
                "{c} is an inpainting pipeline: it needs a mask, which no endpoint takes (use a text-to-image repo)"
            )),
            c if c.contains("Img2Img") || c.contains("Refiner") => Err(format!(
                "{c} is an image-to-image-only pipeline; load its text-to-image base repo and use /v1/images/edits"
            )),
            "TextToVideoSDPipeline" => return None,
            c => Err(format!("diffusers pipeline {c} is not supported")),
        });
    }
    if let Some(r) = crate::model::krea2::load_derived(artifacts, hub) {
        return Some(r);
    }
    if let Some(r) = crate::model::qwen_image::load_derived(artifacts, hub) {
        return Some(r);
    }
    if let Some(r) = crate::model::flux::load_derived(artifacts, hub) {
        return Some(r);
    }
    let weights = artifacts.weights_path.as_deref()?;
    let sd = match weights.extension().and_then(|e| e.to_str()) {
        Some("gguf") => {
            // Probe tensor names in the header (SD GGUFs have little metadata) instead of reading
            // a multi-GB LLM GGUF into memory just to learn it is not SD.
            let mut head = Vec::new();
            let f = std::fs::File::open(weights).ok()?;
            std::io::Read::read_to_end(&mut std::io::Read::take(f, 16 << 20), &mut head).ok()?;
            let has = |s: &[u8]| head.windows(s.len()).any(|w| w == s);
            (has(b"model.diffusion_model.") && has(b"first_stage_model.decoder."))
                .then(|| arc(SdPipeline::load(weights, hub)))
        }
        Some("safetensors") => {
            let names: Vec<String> = crate::model::loaders::SafetensorsWeightReader::new(weights)
                .and_then(|r| r.list_tensors())
                .ok()?
                .into_iter()
                .map(|m| m.name)
                .collect();
            if lora::is_lora(names.iter().map(String::as_str)) {
                load_lora(artifacts, weights, hub)
            } else {
                let single = names
                    .iter()
                    .any(|n| n.starts_with("model.diffusion_model."))
                    && names.iter().any(|n| n.starts_with("first_stage_model."));
                single.then(|| arc(SdPipeline::load_single_file(weights, hub, id, None)))
            }
        }
        _ => None,
    };
    // Derived (non-diffusers) image repos of other families go here.
    sd
}

/// diffusers' official mirror of the SD 1.5 repo that LoRA cards still name.
fn base_alias(id: &str) -> &str {
    match id {
        "runwayml/stable-diffusion-v1-5" => "stable-diffusion-v1-5/stable-diffusion-v1-5",
        other => other,
    }
}

/// A LoRA repo whose card names a Stable Diffusion base (a diffusers SD pipeline, or a repo without
/// `model_index.json`, i.e. a single-file checkpoint); `None` for other bases. The base is fetched.
fn load_lora(
    artifacts: &ModelArtifacts,
    weights: &Path,
    hub: &HuggingFaceHub,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let repo = crate::model::hub::split_weight_file(&artifacts.model_id).0;
    let readme = hub.fetch_file(repo, "README.md").ok()?;
    let base = lora::card_base_model(&std::fs::read_to_string(readme).ok()?)?;
    let base = base_alias(&base).to_string();
    let base_class = hub
        .fetch_file(&base, "model_index.json")
        .ok()
        .and_then(|p| pipeline_class(p.parent()?));
    if base_class
        .as_deref()
        .is_some_and(|c| !SD_PIPELINES.contains(&c))
    {
        return None;
    }
    Some((|| {
        let lora = Lora::open(weights)?;
        let token = std::env::var("HF_TOKEN").ok();
        let b = hub
            .auto_download_and_cache(&base, token.as_deref())
            .map_err(|e| format!("LoRA base model {base}: {e}"))?;
        let dir = b
            .local_dir
            .as_deref()
            .ok_or_else(|| format!("LoRA base model {base} has no snapshot"))?;
        let id = format!("{} on {base}", artifacts.model_id);
        match pipeline_class(dir) {
            Some(c) if SD_PIPELINES.contains(&c.as_str()) => {
                arc(SdPipeline::load_diffusers(dir, hub, &id, Some(&lora)))
            }
            Some(c) => Err(format!(
                "LoRA base {base} is a {c}, not a Stable Diffusion pipeline"
            )),
            None => match b.weights_path.as_deref() {
                Some(w) => arc(SdPipeline::load_single_file(w, hub, &id, Some(&lora))),
                None => Err(format!("LoRA base {base} has no weights")),
            },
        }
    })())
}
