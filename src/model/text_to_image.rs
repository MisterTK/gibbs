//! Legacy image loaders of the families no diffusion spec runs yet (Z-Image, Qwen-Image, Krea 2,
//! FLUX.2 klein); [`crate::runtime::diffusion::load_image`] runs first.

use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::runtime::diffusion::{pipeline_class, ImageGenerator};
use std::sync::Arc;

fn arc<G: ImageGenerator + 'static>(g: Result<G, String>) -> Result<Arc<dyn ImageGenerator>, String> {
    g.map(|g| Arc::new(g) as Arc<dyn ImageGenerator>)
}

/// Loads `artifacts` as an image pipeline of a legacy family. `None`: not one (the caller tries
/// other model kinds). Diffusers repos dispatch on `model_index.json`'s `_class_name`; other repos
/// are the families' derived (LoRA / GGUF / single-file) repos.
pub fn load_image_model(
    artifacts: &ModelArtifacts,
    hub: &HuggingFaceHub,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let dir = artifacts.local_dir.as_deref()?;
    if let Some(class) = pipeline_class(dir) {
        return Some(match class.as_str() {
            "ZImagePipeline" => arc(crate::model::zimage::ZImagePipeline::load(dir)),
            "Krea2Pipeline" => arc(crate::model::krea2::Krea2Pipeline::load(dir)),
            "QwenImage21Pipeline" => arc(crate::model::qwen_image21::QwenImage21Pipeline::load(dir)),
            "QwenImagePipeline" => arc(crate::model::qwen_image_v1::QwenImagePipeline::load(dir)),
            "Flux2KleinPipeline" => arc(crate::model::flux::FluxPipeline::load(dir, hub)),
            "TextToVideoSDPipeline" => return None,
            c => Err(format!("diffusers pipeline {c} is not supported")),
        });
    }
    crate::model::krea2::load_derived(artifacts, hub)
        .or_else(|| crate::model::qwen_image::load_derived(artifacts, hub))
        .or_else(|| crate::model::flux::load_derived(artifacts, hub))
}
