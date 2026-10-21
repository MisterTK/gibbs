//! ModelScope text-to-video (`ali-vilab/text-to-video-ms-1.7b`, diffusers TextToVideoSDPipeline): CLIP text
//! encoder + UNet3D + AutoencoderKL (Metal programs when GPU-only) from the repo's fp16 safetensors, DDIM ("leading" spacing,
//! steps_offset 1, eta 0, set_alpha_to_one false) with classifier-free guidance against the "" prompt.

use crate::model::clip::{ClipTextTransformer, Tensors};
use crate::model::hub::{is_synthetic_model_id, HuggingFaceHub};
use crate::model::image::ImageBuffer;
use crate::model::loaders::safetensors::SafetensorsWeightReader;
use crate::model::sd::{alphas_cumprod, Rng, TRAIN_STEPS};
use crate::model::unet3d::Unet3d;
use crate::model::vae::SdVae;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const CONTEXT_LEN: usize = 77;
/// Accepted clip lengths. ModelScope is trained on 16-frame clips.
// ponytail: bounds are measured, not derived: 8 frames gives garbage (temporal layers out of distribution);
// 16 gives coherent motion (fine prompt details are often missed); 24 gives coherent motion but drifted
// off-prompt in our sample; 12 and anything above 24 are untested.
const FRAMES: std::ops::RangeInclusive<usize> = 16..=24;
const STEPS_OFFSET: usize = 1;
const TEXT_ENCODER: &str = "text_encoder/model.fp16.safetensors";
const UNET: &str = "unet/diffusion_pytorch_model.fp16.safetensors";
const VAE: &str = "vae/diffusion_pytorch_model.fp16.safetensors";

/// A text-to-video repo, as `arch::load_model` routes it (legacy: until Phase 10 video specs).
pub enum Video {
    /// An LTX-Video generation plan (diffusers repo or `ltxv` GGUF).
    Ltx(crate::model::ltx::Plan),
    /// A ModelScope `TextToVideoSDPipeline` repo ([`TextToVideoPipeline`]).
    ModelScope,
}

/// The video route of `repo_id`, if it is a text-to-video repo; `quant` picks an LTX GGUF.
pub fn detect(
    hub: &HuggingFaceHub,
    repo_id: &str,
    quant: Option<&str>,
) -> Option<Result<Video, String>> {
    if let Some(plan) = crate::model::ltx::resolve(hub, repo_id, quant) {
        return Some(plan.map(Video::Ltx));
    }
    TextToVideoPipeline::is_t2v_repo(hub, repo_id).then_some(Ok(Video::ModelScope))
}

pub struct TextToVideoPipeline {
    clip: ClipTextTransformer,
    unet: Unet3d,
    vae: SdVae,
}

impl TextToVideoPipeline {
    /// True iff the repo's `model_index.json` declares `TextToVideoSDPipeline`. A cached snapshot dir is
    /// answered locally; otherwise `model_index.json` is fetched, and a 404 or offline error means false.
    pub fn is_t2v_repo(hub: &HuggingFaceHub, repo_id: &str) -> bool {
        // Only plain `<org>/<repo>` hub ids can be diffusers repos (not local paths, weight files, synthetic ids).
        if repo_id.split('/').count() != 2
            || is_synthetic_model_id(repo_id)
            || Path::new(repo_id).exists()
        {
            return false;
        }
        let snapshot = hub
            .cache_root
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("snapshots")
            .join("main");
        let index = if snapshot.is_dir() {
            snapshot.join("model_index.json")
        } else {
            match hub.fetch_file(repo_id, "model_index.json") {
                Ok(p) => p,
                Err(_) => {
                    // Drop the empty dirs the failed fetch created so the id never looks cached.
                    for dir in snapshot.ancestors().take(3) {
                        let _ = std::fs::remove_dir(dir);
                    }
                    return false;
                }
            }
        };
        std::fs::read_to_string(index)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .is_some_and(|v| v["_class_name"] == "TextToVideoSDPipeline")
    }

    pub fn load(hub: &HuggingFaceHub, repo_id: &str) -> Result<Self, String> {
        let fetch = |file: &str| -> Result<PathBuf, String> {
            println!("Fetching {repo_id}/{file} (reused if cached)");
            let path = hub
                .fetch_file(repo_id, file)
                .map_err(|e| format!("{repo_id}/{file}: {e}"))?;
            let bytes = std::fs::metadata(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .len();
            println!(
                "  {file}: {:.1} MB at {}",
                bytes as f64 / 1e6,
                path.display()
            );
            Ok(path)
        };
        let open = |path: &Path| {
            SafetensorsWeightReader::new(path).map_err(|e| format!("{}: {e}", path.display()))
        };

        let te = open(&fetch(TEXT_ENCODER)?)?;
        let clip = ClipTextTransformer::load(&Tensors::safetensors(&te)?, "text_model", hub)?;
        drop(te);
        let vae_reader = open(&fetch(VAE)?)?;
        let vae = SdVae::from_diffusers(&Tensors::safetensors(&vae_reader)?)?;
        drop(vae_reader);
        let unet = Unet3d::load(&fetch(UNET)?)?;
        Ok(Self { clip, unet, vae })
    }

    /// `frames` RGB frames of `width`x`height`. Prints the time of each denoising step.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        frames: usize,
        width: usize,
        height: usize,
        steps: usize,
        guidance: f32,
        seed: u64,
    ) -> Result<Vec<ImageBuffer>, String> {
        if frames < *FRAMES.start() {
            return Err(format!(
                "--frames {frames} is too short: ModelScope is trained on 16-frame clips and 8 frames measured as garbage. \
                 Use {}..={} frames",
                FRAMES.start(),
                FRAMES.end()
            ));
        }
        if frames > *FRAMES.end() {
            return Err(format!(
                "--frames {frames} is untested: only 16 and 24 frames were tried. Use {}..={} frames",
                FRAMES.start(),
                FRAMES.end()
            ));
        }
        if width == 0
            || height == 0
            || width % 64 != 0
            || height % 64 != 0
            || width > 512
            || height > 512
        {
            return Err(format!(
                "width/height must be multiples of 64 up to 512, got {width}x{height}"
            ));
        }
        if !(1..=100).contains(&steps) {
            return Err(format!("steps must be in 1..=100, got {steps}"));
        }
        let (h, w) = (height / 8, width / 8);
        let frame_len = 4 * h * w;
        let cond = self.clip.encode_prompt(prompt)?;
        // diffusers only runs CFG when guidance_scale > 1.
        let uncond = if guidance > 1.0 {
            Some(self.clip.encode_prompt("")?)
        } else {
            None
        };

        let ac = alphas_cumprod();
        let ts = ddim_timesteps(steps);
        let mut rng = Rng::new(seed);
        let mut x: Vec<f32> = (0..frames * frame_len).map(|_| rng.gaussian()).collect();
        for (i, &t) in ts.iter().enumerate() {
            let start = Instant::now();
            let mut eps = self
                .unet
                .forward(&x, frames, h, w, t as f32, &cond, CONTEXT_LEN)?;
            if let Some(u) = &uncond {
                let eps_u = self
                    .unet
                    .forward(&x, frames, h, w, t as f32, u, CONTEXT_LEN)?;
                for (c, u) in eps.iter_mut().zip(&eps_u) {
                    *c = u + guidance * (*c - u);
                }
            }
            if eps.len() != x.len() {
                return Err(format!(
                    "unet3d returned {} values, expected {}",
                    eps.len(),
                    x.len()
                ));
            }
            ddim_step(&mut x, &eps, &ac, t, steps);
            println!(
                "  step {}/{steps} (t={t}): {:.2}s",
                i + 1,
                start.elapsed().as_secs_f64()
            );
        }
        x.chunks_exact(frame_len)
            .map(|z| self.vae.decode(z, h, w))
            .collect()
    }
}

/// diffusers DDIMScheduler "leading" timesteps with steps_offset 1, descending.
fn ddim_timesteps(steps: usize) -> Vec<usize> {
    let ratio = TRAIN_STEPS / steps;
    (0..steps).rev().map(|i| i * ratio + STEPS_OFFSET).collect()
}

/// One DDIM step, eta 0, epsilon prediction, no clipping; past the first train step uses alphas_cumprod[0].
fn ddim_step(x: &mut [f32], eps: &[f32], ac: &[f64], t: usize, steps: usize) {
    let a_t = ac[t];
    let a_prev = t.checked_sub(TRAIN_STEPS / steps).map_or(ac[0], |p| ac[p]);
    let (sa, sb) = (a_t.sqrt(), (1.0 - a_t).sqrt());
    let (pa, pb) = (a_prev.sqrt(), (1.0 - a_prev).sqrt());
    for (v, &e) in x.iter_mut().zip(eps) {
        let e = e as f64;
        let x0 = (*v as f64 - sb * e) / sa;
        *v = (pa * x0 + pb * e) as f32;
    }
}

/// Encodes `frames` (same size, even dims) to H.264 `path` by piping rgb24 into the `ffmpeg` binary.
pub fn write_mp4(frames: &[ImageBuffer], fps: u32, path: &Path) -> Result<(), String> {
    let first = frames.first().ok_or("write_mp4: no frames")?;
    let (w, h) = (first.width, first.height);
    if frames
        .iter()
        .any(|f| f.width != w || f.height != h || f.data.len() != w * h * 3)
    {
        return Err(format!("write_mp4: frames must all be RGB {w}x{h}"));
    }
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
        ])
        .arg(format!("{w}x{h}"))
        .arg("-r")
        .arg(fps.to_string())
        .args(["-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                "ffmpeg not found on PATH; install ffmpeg (with libx264) to write .mp4 video"
                    .to_string()
            }
            _ => format!("failed to start ffmpeg: {e}"),
        })?;
    let mut stdin = child.stdin.take().ok_or("ffmpeg stdin unavailable")?;
    // A write error means ffmpeg exited early; its stderr below says why.
    let write_err = frames
        .iter()
        .try_for_each(|f| stdin.write_all(&f.data))
        .err();
    drop(stdin);
    let out = child
        .wait_with_output()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !out.status.success() || write_err.is_some() {
        return Err(format!(
            "ffmpeg failed ({}): {}{}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim(),
            write_err
                .map(|e| format!(" (pipe: {e})"))
                .unwrap_or_default()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_timesteps_with_offset() {
        let ts = ddim_timesteps(25);
        assert_eq!(ts.len(), 25);
        assert_eq!(&ts[..3], &[961, 921, 881]);
        assert_eq!(*ts.last().unwrap(), 1);
        assert_eq!(ddim_timesteps(1), vec![1]);
    }

    #[test]
    fn ddim_step_zero_eps_rescales_by_alpha_ratio() {
        let ac = alphas_cumprod();
        // t=41 -> prev 1; t=1 -> prev -39 -> final alpha ac[0].
        for (t, prev) in [(41, ac[1]), (1, ac[0])] {
            let mut x = vec![1.0f32];
            ddim_step(&mut x, &[0.0], &ac, t, 25);
            assert!(
                (x[0] as f64 - (prev / ac[t]).sqrt()).abs() < 1e-6,
                "t={t}: {}",
                x[0]
            );
        }
    }
}
