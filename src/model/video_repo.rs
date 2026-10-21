//! Where the parts of a text-to-video generation come from: a diffusers video pipeline repo
//! (`model_index.json` class of a `pipeline: video` family spec), or an LTX-Video GGUF repo (hub
//! arch `ltxv`) whose card names such a base, its transformer (and VAE, when the repo carries one)
//! replacing the base's, with a `t5encoder` GGUF text encoder by default.

use crate::hardware::PlacementMode;
use crate::model::hub::{
    hub_json, is_synthetic_model_id, list_repo_files, select_weight_files, split_weight_file, HuggingFaceHub,
    ModelArtifacts, RepoFile,
};
use crate::model::image::ImageBuffer;
use crate::runtime::diffusion::{Extras, SpecDiffusion, VideoRequest};
use std::path::Path;

/// T5-XXL `t5encoder` GGUF used with GGUF transformers unless the caller names another text encoder.
// ponytail: Q4_K_M is a fixed quality/size pick; `text_encoder` overrides it with another T5 GGUF quant.
const DEFAULT_T5_GGUF: &str = "city96/t5-v1_1-xxl-encoder-gguf/t5-v1_1-xxl-encoder-Q4_K_M.gguf";

/// Whether a repo file name belongs to the LTX-Video v0.9.0 generation (`ltx-video-2b-v0.9-*.gguf`,
/// `ltxv_0.9_vae_*`): later versions (`v0.9.1`, `0.9.6`, `13b-0.9.7`, ...) pair with other VAE decoders
/// (timestep conditioning, noise injection, wider blocks) that are not implemented. Transformer GGUFs
/// (`marker`) must carry the version; the original VAE files (`LTX-Video-VAE-*`) have none.
fn is_v090(name: &str, marker: bool) -> bool {
    let n = name.to_lowercase();
    match n.find("0.9") {
        Some(i) => !(n[i + 3..].starts_with('.') && n[i + 4..].starts_with(|c: char| c.is_ascii_digit())),
        None => !marker && !n.contains("13b"),
    }
}

/// Files of an LTX-Video GGUF repo: the transformer GGUF (the hub's usual quant choice among the v0.9.0
/// ones) and, when the repo carries its own, the matching VAE. `Err` carries the reason a repo is refused.
pub struct RepoPick {
    pub transformer: String,
    pub vae: Option<String>,
}

pub fn pick_gguf(files: &[RepoFile], quant: Option<&str>) -> Result<RepoPick, String> {
    let base = |p: &str| p.rsplit('/').next().unwrap_or(p).to_lowercase();
    let ggufs: Vec<&RepoFile> = files
        .iter()
        .filter(|f| {
            let b = base(&f.path);
            b.ends_with(".gguf") && !b.contains("vae") && !b.contains("t5")
        })
        .collect();
    if ggufs.is_empty() {
        return Err("no LTX-Video transformer GGUF in the repo".into());
    }
    let ok: Vec<RepoFile> = ggufs.iter().filter(|f| is_v090(&base(&f.path), true)).map(|f| (*f).clone()).collect();
    if ok.is_empty() {
        let names: Vec<String> = ggufs.iter().take(3).map(|f| f.path.clone()).collect();
        return Err(format!(
            "LTX-Video GGUF without a v0.9.0 transformer (found {}): v0.9.1+ need their own VAE decoder \
             (timestep conditioning, noise injection), which is not implemented; only v0.9.0 loads",
            names.join(", ")
        ));
    }
    let transformer = select_weight_files(&ok, quant)?.swap_remove(0);
    let mut vaes: Vec<&RepoFile> = files
        .iter()
        .filter(|f| {
            let b = base(&f.path);
            b.ends_with(".safetensors") && b.contains("vae") && !b.contains("fp8") && is_v090(&b, false)
        })
        .collect();
    vaes.sort_by(|a, b| (a.size, &a.path).cmp(&(b.size, &b.path)));
    Ok(RepoPick { transformer, vae: vaes.first().map(|f| f.path.clone()) })
}

/// One video generation's sources: the diffusers base repo, its family, and (component, repo, file)
/// weights replacing the base's.
#[derive(Clone)]
pub struct Plan {
    pub family: String,
    base: String,
    weights: Vec<(String, String, String)>,
    label: String,
}

impl Plan {
    /// `repo/file.gguf` or `repo (diffusers)`.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// The `pipeline: video` family of diffusers pipeline class `class`.
fn video_family(class: &str) -> Option<String> {
    let reg = crate::arch::registry().ok()?;
    let sig = crate::arch::Signals { diffusers_class: Some(class.to_string()), ..Default::default() };
    let f = reg.detect(&sig)?;
    (reg.spec(&f, &sig).ok()?.pipeline == crate::arch::Pipeline::Video).then_some(f)
}

/// The `model_index.json` class of `repo`: the cached snapshot's, else fetched (`None` on 404/offline).
fn class_of(hub: &HuggingFaceHub, repo: &str) -> Option<String> {
    let snapshot = hub.cache_root.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots").join("main");
    let index = match snapshot.join("model_index.json") {
        p if p.is_file() => p,
        _ => match hub.fetch_file(repo, "model_index.json") {
            Ok(p) => p,
            Err(_) => {
                // Drop the empty dirs the failed fetch created so the id never looks cached.
                for dir in snapshot.ancestors().take(3) {
                    let _ = std::fs::remove_dir(dir);
                }
                return None;
            }
        },
    };
    crate::runtime::diffusion::pipeline_class(index.parent()?)
}

/// `Some` when `model_id` names a video repo; `quant` picks an LTX-Video GGUF.
pub fn detect(hub: &HuggingFaceHub, model_id: &str, quant: Option<&str>) -> Option<Result<Plan, String>> {
    let (repo, file) = split_weight_file(model_id);
    if repo.split('/').count() != 2 || is_synthetic_model_id(repo) || Path::new(model_id).exists() {
        return None;
    }
    if file.is_none() {
        if let Some(family) = class_of(hub, repo).as_deref().and_then(video_family) {
            return Some(Ok(Plan { family, base: repo.into(), weights: Vec::new(), label: format!("{repo} (diffusers)") }));
        }
    }
    let token = std::env::var("HF_TOKEN").ok();
    let info = hub_json(&format!("https://huggingface.co/api/models/{repo}"), token.as_deref()).ok()?;
    if info["gguf"]["architecture"] != "ltxv" {
        return None;
    }
    Some((|| {
        let files = list_repo_files(repo, token.as_deref())?;
        let mut pick = pick_gguf(&files, quant)?;
        if let Some(f) = file {
            if !files.iter().any(|x| x.path == f) {
                return Err(format!("{repo} has no file {f}"));
            }
            if !is_v090(f, true) {
                return Err(format!("{f}: only v0.9.0 LTX-Video GGUFs load (later versions need other VAE decoders)"));
            }
            pick.transformer = f.to_string();
        }
        let card = hub.fetch_file(repo, "README.md")?;
        let card = std::fs::read_to_string(&card).map_err(|e| format!("{}: {e}", card.display()))?;
        let base = crate::model::lora::card_base_model(&card)
            .ok_or_else(|| format!("{repo}: LTX-Video GGUF whose model card names no base_model"))?;
        let family = class_of(hub, &base)
            .as_deref()
            .and_then(video_family)
            .ok_or_else(|| format!("{repo}: base model {base} is not a video pipeline repo"))?;
        let mut weights = vec![("transformer".to_string(), repo.to_string(), pick.transformer.clone())];
        if let Some(v) = pick.vae {
            weights.push(("vae".into(), repo.into(), v));
        }
        // A quantized transformer defaults to the quantized T5 (2.9 GB) rather than the base's 19 GB fp32 shards.
        let (t5_repo, t5_file) = split_weight_file(DEFAULT_T5_GGUF);
        weights.push(("text_encoder".into(), t5_repo.into(), t5_file.unwrap_or_default().into()));
        Ok(Plan { family, label: format!("{repo}/{}", pick.transformer), base, weights })
    })())
}

/// Fetches and loads `plan`; `text_encoder` (a weights repo or `repo/file`) replaces the text
/// encoder's weights.
pub fn load(hub: &HuggingFaceHub, plan: &Plan, mode: PlacementMode, text_encoder: Option<&str>) -> Result<SpecDiffusion, String> {
    let mut weights = plan.weights.clone();
    if let Some(id) = text_encoder {
        let (repo, file) = split_weight_file(id);
        let file = match file {
            Some(f) => f.to_string(),
            None => select_weight_files(&list_repo_files(repo, std::env::var("HF_TOKEN").ok().as_deref())?, None)?.swap_remove(0),
        };
        weights.retain(|w| w.0 != "text_encoder");
        weights.push(("text_encoder".into(), repo.into(), file));
    }
    let skip: Vec<&str> = weights.iter().map(|w| w.0.as_str()).collect();
    let dir = hub.fetch_diffusers(&plan.base, &skip)?;
    let mut files = Vec::new();
    for (comp, repo, file) in &weights {
        // The replaced component keeps the base's config.
        let _ = hub.fetch_file(&plan.base, &format!("{comp}/config.json"));
        files.push((comp.clone(), hub.fetch_file(repo, file).map_err(|e| format!("{repo}/{file}: {e}"))?));
    }
    let a = ModelArtifacts { model_id: plan.label.clone(), local_dir: Some(dir), is_cached: true, ..Default::default() };
    SpecDiffusion::load(&a, hub, mode, Some(&plan.family), &Extras { weights: files, ..Default::default() })
}

/// The loaded video plan the TUI's `/video` and `/i2v` generate with.
pub type VideoSlot = std::sync::Arc<parking_lot::RwLock<Option<std::sync::Arc<(HuggingFaceHub, Plan)>>>>;

/// Loads `plan` and generates one clip: `req`'s unset fields take the pipeline defaults (the
/// resolved request is returned with the frames and their frame rate).
pub fn generate(
    hub: &HuggingFaceHub,
    plan: &Plan,
    mode: PlacementMode,
    text_encoder: Option<&str>,
    req: &VideoRequest,
    image: Option<&ImageBuffer>,
    log: &dyn Fn(&str),
) -> Result<(Vec<ImageBuffer>, u32, VideoRequest), String> {
    let t = std::time::Instant::now();
    let p = load(hub, plan, mode, text_encoder)?;
    log(&format!("{} loaded in {:.2}s", plan.label, t.elapsed().as_secs_f64()));
    let d = p.video_defaults().ok_or_else(|| format!("{}: not a video pipeline", plan.label))??;
    let req = VideoRequest {
        frames: Some(req.frames.unwrap_or(d.frames)),
        width: Some(req.width.unwrap_or(d.width)),
        height: Some(req.height.unwrap_or(d.height)),
        steps: Some(req.steps.unwrap_or(d.steps)),
        guidance: Some(req.guidance.unwrap_or(d.guidance)),
        ..req.clone()
    };
    let frames = p.generate_video(&req, image, log)?;
    Ok((frames, d.fps, req))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> Vec<RepoFile> {
        names.iter().map(|n| RepoFile { path: n.to_string(), size: None }).collect()
    }

    #[test]
    fn only_v090_files_are_picked() {
        let p = pick_gguf(
            &files(&[
                "ltx-video-2b-v0.9-Q8_0.gguf",
                "ltx-video-2b-v0.9-Q4_0.gguf",
                "ltx-video-2b-v0.9.1-q4_0.gguf",
                "ltxv-13b-0.9.7-dev-q4_0.gguf",
                "LTX-Video-VAE-BF16.safetensors",
                "ltxv_0.9.5_vae_fp16.safetensors",
            ]),
            None,
        )
        .unwrap();
        assert_eq!(p.transformer, "ltx-video-2b-v0.9-Q4_0.gguf");
        assert_eq!(p.vae.as_deref(), Some("LTX-Video-VAE-BF16.safetensors"));
        let e = pick_gguf(&files(&["ltxv-13b-0.9.7-dev-q4_0.gguf", "ltxv-2b-0.9.6-dev-q4_0.gguf"]), None)
            .err()
            .unwrap();
        assert!(e.contains("13b-0.9.7") && e.contains("v0.9.0"), "{e}");
    }

    #[test]
    fn video_classes_map_to_video_families() {
        assert_eq!(video_family("LTXPipeline").as_deref(), Some("ltx"));
        assert_eq!(video_family("LTXImageToVideoPipeline").as_deref(), Some("ltx"));
        assert_eq!(video_family("TextToVideoSDPipeline").as_deref(), Some("modelscope_t2v"));
        assert_eq!(video_family("StableDiffusionPipeline"), None);
    }

    /// A tiny `LTXPipeline` snapshot in a hub cache (T5 encoder, the tiny LTX transformer and VAE
    /// of `diff_fixtures`, a flow-match scheduler config).
    fn tiny_ltx_repo(root: &Path) -> HuggingFaceHub {
        use crate::model::diff_fixtures::{values, write_f32_safetensors, write_tiny_ltx, write_tiny_ltx_vae};
        use serde_json::json;
        let dir = root.join("hub/models--test--tiny-ltx/snapshots/main");
        let put = |p: &str, v: serde_json::Value| {
            let p = dir.join(p);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, v.to_string()).unwrap();
        };
        put("model_index.json", json!({"_class_name": "LTXPipeline"}));
        put("scheduler/scheduler_config.json", json!({"_class_name": "FlowMatchEulerDiscreteScheduler",
            "use_dynamic_shifting": true, "base_image_seq_len": 1024, "max_image_seq_len": 4096,
            "base_shift": 0.95, "max_shift": 2.05, "shift_terminal": 0.1, "num_train_timesteps": 1000}));
        let (vocab, d, ff) = (40, 32, 48);
        put("text_encoder/config.json", json!({"architectures": ["T5EncoderModel"], "model_type": "t5",
            "vocab_size": vocab, "d_model": d, "d_kv": 8, "num_heads": 4, "d_ff": ff, "num_layers": 2,
            "relative_attention_num_buckets": 8, "relative_attention_max_distance": 16,
            "layer_norm_epsilon": 1e-6, "feed_forward_proj": "gated-gelu"}));
        let mut t5: Vec<(String, Vec<usize>)> = vec![("shared.weight".into(), vec![vocab, d]),
            ("encoder.final_layer_norm.weight".into(), vec![d]),
            ("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(), vec![8, 4])];
        for i in 0..2 {
            let b = format!("encoder.block.{i}.layer");
            for n in ["q", "k", "v", "o"] {
                t5.push((format!("{b}.0.SelfAttention.{n}.weight"), vec![d, d]));
            }
            t5.push((format!("{b}.0.layer_norm.weight"), vec![d]));
            t5.push((format!("{b}.1.layer_norm.weight"), vec![d]));
            t5.push((format!("{b}.1.DenseReluDense.wi_0.weight"), vec![ff, d]));
            t5.push((format!("{b}.1.DenseReluDense.wi_1.weight"), vec![ff, d]));
            t5.push((format!("{b}.1.DenseReluDense.wo.weight"), vec![d, ff]));
        }
        write_f32_safetensors(&dir.join("text_encoder/model.safetensors"), &values(&t5, 21)).unwrap();
        let vocab_map: serde_json::Map<String, serde_json::Value> = (0..vocab).map(|i| (format!("t{i}"), json!(i))).collect();
        put("tokenizer/tokenizer.json", json!({"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab_map, "unk_token": "t0"}}));
        write_tiny_ltx(&dir.join("transformer"), d).unwrap();
        put("transformer/config.json", json!({"_class_name": "LTXVideoTransformer3DModel"}));
        write_tiny_ltx_vae(&dir.join("vae"), 128).unwrap();
        HuggingFaceHub::with_cache_dir(root.join("hub"))
    }

    /// Text- and image-to-video through detection + `generate_video` on the tiny LTX snapshot: 5
    /// frames of 12x8 (3 latent frames of 2x3 tokens), 2 flow steps, CFG. The image-conditioned
    /// first latent frame is bit-identical across seeds and step counts (never stepped).
    #[test]
    fn tiny_ltx_text_and_image_to_video() {
        use crate::hardware::Device;
        let _cpu = crate::hardware::cpu_compute::allow();
        let tmp = tempfile::tempdir().unwrap();
        let hub = tiny_ltx_repo(tmp.path());
        let plan = detect(&hub, "test/tiny-ltx", None).expect("a video repo").unwrap();
        assert_eq!(plan.family, "ltx");
        let req = VideoRequest {
            prompt: "t3 t5 t7".into(),
            frames: Some(5),
            width: Some(12),
            height: Some(8),
            steps: Some(2),
            guidance: Some(3.0),
            seed: 1,
            ..Default::default()
        };
        let mut img = ImageBuffer::new(12, 8);
        img.data.iter_mut().enumerate().for_each(|(i, v)| *v = (i * 37 % 251) as u8);
        for dev in crate::hardware::available_devices().into_iter().filter(|d| matches!(d, Device::Cpu | Device::Metal)) {
            let p = load(&hub, &plan, PlacementMode::Pin(dev), None).unwrap();
            let frames = p.generate_video(&req, None, &|_| {}).unwrap();
            assert_eq!(frames.len(), 5, "{dev:?}");
            assert!(frames.iter().all(|f| (f.width, f.height) == (12, 8)));
            let a = p.video_latents(&req, Some(&img), &|_| {}).unwrap();
            let b = p.video_latents(&VideoRequest { seed: 9, steps: Some(3), ..req.clone() }, Some(&img), &|_| {}).unwrap();
            assert_eq!((a.frames, a.h, a.w, a.held), (3, 2, 3, 2 * 3 * 128));
            assert_eq!(a.x[..a.held], b.x[..b.held], "{dev:?}: the clean first frame moved");
            assert!(a.x[a.held..].iter().zip(&b.x[b.held..]).any(|(x, y)| x != y));
            assert!(a.x.iter().all(|v| v.is_finite()));
            let e = p.generate_video(&VideoRequest { frames: Some(6), ..req.clone() }, None, &|_| {}).unwrap_err();
            assert!(e.contains("k+1"), "{e}");
        }
    }
}
