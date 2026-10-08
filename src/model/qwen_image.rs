//! Qwen-Image text-to-image, both generations, behind one derived-repo loader: Qwen-Image 2.1
//! ([`crate::model::qwen_image21`]: single-stream block-causal DiT, Qwen3-VL encoder, 64-channel
//! VAE) and Qwen-Image 1 ([`crate::model::qwen_image_v1`]: 60-block MMDiT, Qwen2.5-VL encoder).
//! This file holds what both share: weights left in their stored encoding, the flow-match
//! schedule with `shift_terminal`, and the repos that build on a diffusers base (transformer-only
//! GGUFs and single-file finetunes, as FLUX's `load_derived`).

use crate::model::flow;
use crate::model::flux::{Store, Wt};
use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::model::nn;
use crate::model::text_to_image::{pipeline_class, ImageGenerator};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

/// diffusers repos whose `transformer` the derived GGUFs / finetunes replace.
pub const BASES: &[&str] = &["Qwen/Qwen-Image", "Qwen/Qwen-Image-2.1"];

/// llama.cpp-style `general.architecture` of those GGUFs (a 2.1 GGUF may say either).
pub const GGUF_ARCHS: &[&str] = &["qwen_image", "qwen_image21"];

/// A linear layer's weights `[n, k]` (a row range of a stored matrix, so a fused `gate_up` serves
/// as two) decoded per GEMM panel, never expanded.
pub(crate) struct QLin {
    w: Arc<Wt>,
    off: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl QLin {
    pub(crate) fn load(s: &Store, name: &str) -> Result<Self, String> {
        let w = s.mat(name)?;
        let (n, k) = w.dims();
        Ok(Self {
            w: Arc::new(w),
            off: 0,
            n,
            k,
        })
    }

    /// Rows `off..off + n` of the same stored matrix.
    pub(crate) fn rows(&self, off: usize, n: usize) -> Result<Self, String> {
        if off + n > self.n {
            return Err(format!(
                "rows {off}..{} of a {}-row weight",
                off + n,
                self.n
            ));
        }
        Ok(Self {
            w: self.w.clone(),
            off: self.off + off,
            n,
            k: self.k,
        })
    }

    /// x `[m, k]` -> `[m, n]`.
    pub(crate) fn fwd(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * self.n];
        nn::matmul_rows(x, m, self.k, self.n, None, &mut out, &|j, k0, dst| {
            self.w.decode(self.off + j, k0, dst)
        });
        out
    }
}

/// diffusers `FlowMatchEulerDiscreteScheduler` as Qwen-Image configures it: resolution-dependent
/// exponential shift of the requested sigmas, optionally stretched to end at `shift_terminal`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sched {
    dynamic: bool,
    shift: f64,
    base_seq: f64,
    max_seq: f64,
    base_shift: f64,
    max_shift: f64,
    terminal: Option<f64>,
}

impl Sched {
    pub(crate) fn from_config(v: &Value) -> Self {
        let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        Self {
            dynamic: v["use_dynamic_shifting"].as_bool().unwrap_or(false),
            shift: f("shift", 1.0),
            base_seq: f("base_image_seq_len", 256.0),
            max_seq: f("max_image_seq_len", 4096.0),
            base_shift: f("base_shift", 0.5),
            max_shift: f("max_shift", 1.15),
            // diffusers tests `if self.config.shift_terminal`: null and 0 both mean off.
            terminal: v["shift_terminal"].as_f64().filter(|t| *t != 0.0),
        }
    }

    /// `set_timesteps(sigmas = base, mu)` for an image of `tokens` latent tokens, 0 appended.
    pub(crate) fn sigmas(&self, base: &[f64], tokens: usize) -> Vec<f32> {
        let mu = flow::calculate_shift(
            tokens,
            self.base_seq,
            self.max_seq,
            self.base_shift,
            self.max_shift,
        );
        let mut s: Vec<f64> = base
            .iter()
            .map(|&t| {
                if self.dynamic {
                    flow::time_shift(mu, t)
                } else {
                    flow::shift_sigma(t, self.shift)
                }
            })
            .collect();
        if let Some(t) = self.terminal {
            flow::stretch_to_terminal(&mut s, t);
        }
        s.into_iter().map(|v| v as f32).chain([0.0]).collect()
    }
}

/// `STEPS, SIGMAS = 6, [1.0, 0.9375, ...]` of a distilled finetune's model card: the raw sigmas
/// its authors sample with (the card's first `SIGMAS` list of plain numbers).
pub(crate) fn card_sigmas(card: &str) -> Option<Vec<f64>> {
    let rest = &card[card.find("SIGMAS")?..];
    let list = &rest[rest.find('[')? + 1..];
    let list = &list[..list.find(']')?];
    let v: Option<Vec<f64>> = list.split(',').map(|s| s.trim().parse().ok()).collect();
    v.filter(|v| !v.is_empty())
}

/// Tensor names of a GGUF / safetensors file (headers only).
fn tensor_names(path: &Path) -> Option<Vec<String>> {
    crate::model::flux::tensor_names(path)
}

/// Which Qwen-Image transformer a weight file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gen {
    /// Qwen-Image 1: `img_mod`/`txt_mod`, separate text and image streams.
    V1,
    /// Qwen-Image 2.1: one shared `modulation`, `img_mlp.{gate_layer,proj,out}` or fused `gate_up`.
    V21,
}

/// `Gen` of a transformer-only checkpoint, by its tensor names (diffusers names, optionally under
/// ComfyUI's `model.diffusion_model.`).
pub(crate) fn gen_of(names: &[String]) -> Option<Gen> {
    let has = |n: &str| {
        names
            .iter()
            .any(|x| x == n || x.strip_prefix("model.diffusion_model.") == Some(n))
    };
    if has("modulation.1.weight") && has("txt_in.in_layer.weight") {
        Some(Gen::V21)
    } else if has("transformer_blocks.0.img_mod.1.weight") && has("txt_norm.weight") {
        Some(Gen::V1)
    } else {
        None
    }
}

/// Repos that build on a Qwen-Image diffusers repo named by their card's `base_model`: a
/// transformer-only checkpoint (GGUF, single-file finetune). `None` when the weights are not
/// one or the base is not a Qwen-Image pipeline.
pub fn load_derived(
    a: &ModelArtifacts,
    hub: &HuggingFaceHub,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let weights = a.weights_path.as_ref()?;
    let gen = gen_of(&tensor_names(weights)?)?;
    // `--model org/repo/file.gguf` names one file of the repo.
    let repo = crate::model::hub::split_weight_file(&a.model_id).0;
    let card = a
        .local_dir
        .as_ref()
        .map(|d| d.join("README.md"))
        .filter(|p| p.is_file())
        .map_or_else(|| hub.fetch_file(repo, "README.md"), Ok)
        .and_then(|p| std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display())));
    let card = match card {
        Ok(c) => c,
        Err(e) => {
            return Some(Err(format!(
                "{repo}: Qwen-Image weights, model card unreadable: {e}"
            )))
        }
    };
    let Some(base) = crate::model::lora::card_base_model(&card) else {
        return Some(Err(format!(
            "{repo}: Qwen-Image weights, but the model card names no base_model"
        )));
    };
    let class = hub
        .fetch_file(&base, "model_index.json")
        .map_err(|e| format!("Qwen-Image base model {base} could not be fetched ({e})"));
    let class = match class {
        Ok(p) => p.parent().and_then(pipeline_class),
        Err(e) => return Some(Err(e)),
    };
    let want = match gen {
        Gen::V21 => "QwenImage21Pipeline",
        Gen::V1 => "QwenImagePipeline",
    };
    if class.as_deref() != Some(want) {
        return Some(Err(format!(
            "{repo}: these are {gen:?} transformer weights, but its base_model {base} is {}",
            class.as_deref().unwrap_or("not a diffusers pipeline")
        )));
    }
    Some((|| {
        let dir = hub
            .fetch_diffusers(&base, &["transformer"])
            .map_err(|e| format!("Qwen-Image base model {base} could not be fetched ({e})"))?;
        // A distilled finetune ships its own schedule (its card's sigmas, a scheduler/ without the
        // base's terminal stretch); the repo's `scheduler/` wins over the base's when present.
        let sched = hub
            .fetch_file(repo, "scheduler/scheduler_config.json")
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<Value>(&s).ok());
        let p: Arc<dyn ImageGenerator> = match gen {
            Gen::V21 => Arc::new(crate::model::qwen_image21::QwenImage21Pipeline::load_with(
                &dir,
                Some(weights),
                sched,
                card_sigmas(&card),
            )?),
            Gen::V1 => Arc::new(crate::model::qwen_image_v1::QwenImagePipeline::load_with(
                &dir,
                Some(weights),
                sched,
            )?),
        };
        Ok(p)
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_stretch_lands_on_shift_terminal_and_keeps_start() {
        let s = Sched::from_config(&serde_json::json!({
            "use_dynamic_shifting": true, "base_image_seq_len": 256, "max_image_seq_len": 8192,
            "base_shift": 0.5, "max_shift": 0.9, "shift_terminal": 0.02,
        }));
        let sig = s.sigmas(&flow::linspace_sigmas(4), 4096);
        assert_eq!(sig.len(), 5);
        assert!((sig[0] - 1.0).abs() < 1e-6, "{sig:?}");
        assert!((sig[3] - 0.02).abs() < 1e-6, "{sig:?}");
        assert_eq!(sig[4], 0.0);
        assert!(sig.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn no_terminal_is_plain_exponential_shift() {
        // Viggle turbo: raw nodes, mu = 0.5 + 0.4 * (4096 - 256) / (8192 - 256), no stretch.
        let s = Sched::from_config(&serde_json::json!({
            "use_dynamic_shifting": true, "max_image_seq_len": 8192, "max_shift": 0.9, "shift_terminal": null,
        }));
        let sig = s.sigmas(&[1.0, 0.5], 4096);
        let mu: f64 = 0.5 + 0.4 * (4096.0 - 256.0) / (8192.0 - 256.0);
        let want = mu.exp() / (mu.exp() + 1.0);
        assert!(
            (sig[1] as f64 - want).abs() < 1e-6 && sig[0] == 1.0,
            "{sig:?}"
        );
    }

    #[test]
    fn card_sigmas_reads_the_first_list() {
        let card = "x\nSTEPS, SIGMAS = 6, [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]   # pass both\nSIGMAS_9 = [1.0, 1 / 6]";
        assert_eq!(
            card_sigmas(card),
            Some(vec![1.0, 0.9375, 0.875, 0.75, 0.5, 0.25])
        );
        assert_eq!(card_sigmas("no sigmas here"), None);
        assert_eq!(card_sigmas("SIGMAS = [1 / 6]"), None);
    }

    #[test]
    fn generation_follows_tensor_names() {
        let n = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let v21 = n(&[
            "img_in.weight",
            "modulation.1.weight",
            "txt_in.in_layer.weight",
        ]);
        assert_eq!(gen_of(&v21), Some(Gen::V21));
        let v1 = n(&["transformer_blocks.0.img_mod.1.weight", "txt_norm.weight"]);
        assert_eq!(gen_of(&v1), Some(Gen::V1));
        assert_eq!(gen_of(&n(&["double_blocks.0.img_attn.qkv.weight"])), None);
    }
}
