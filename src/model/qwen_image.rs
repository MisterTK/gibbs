//! Qwen-Image text-to-image, both generations, behind one derived-repo loader: Qwen-Image 2.1
//! ([`crate::model::qwen_image21`]: single-stream block-causal DiT, Qwen3-VL encoder, 64-channel
//! VAE) and Qwen-Image 1 ([`crate::model::qwen_image_v1`]: 60-block MMDiT, Qwen2.5-VL encoder).
//! This file holds what both share: weights left in their stored encoding, the flow-match
//! schedule with `shift_terminal`, and the repos that build on a diffusers base (transformer-only
//! GGUFs and single-file finetunes, as FLUX's `load_derived`).

use crate::graph::ValueId;
use crate::model::dit_graph::Gb;
use crate::model::weights::{Mat, Packed};
use crate::runtime::sched;
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

    /// The whole stored matrix as a graph weight, still in its stored encoding (bf16 or packed
    /// GGUF blocks). ponytail: same conversion as `flux::graph_program::wt_mat`; dedupe when a
    /// third family needs it.
    fn stored(&self) -> Mat {
        match &*self.w {
            Wt::Bf16(m) => m.clone(),
            Wt::Gguf {
                map,
                off,
                rows,
                cols,
                fmt,
                ..
            } => Mat::from_packed(
                Packed {
                    map: map.clone(),
                    off: *off,
                    format: *fmt,
                    shape: vec![*rows, *cols],
                },
                *rows,
                *cols,
            ),
        }
    }

    /// Binds this (whole-matrix) weight as `{name}.weight`; calls with one name share it.
    pub(crate) fn bind(&self, b: &mut Gb, name: &str) -> ValueId {
        let m = self.stored();
        assert!(
            self.off == 0 && self.n == m.rows,
            "{name}: bind of a row range"
        );
        b.mat(&format!("{name}.weight"), &m)
    }

    /// Graph form of [`Self::fwd`]: `x [m, k] -> [m, n]` (+ `bias`), named `name`.
    pub(crate) fn emit(&self, b: &mut Gb, name: &str, x: ValueId, bias: Option<&[f32]>) -> ValueId {
        let w = self.bind(b, name);
        let bias = bias.map(|v| b.konst(&format!("{name}.bias"), &[v.len()], v.to_vec()));
        b.linear(name, x, w, bias)
    }

    /// `self` and `other` (row ranges of one stored matrix, as a fused `gate_up` split in two)
    /// from a single GEMM over the whole matrix, then column slices; separate GEMMs otherwise.
    pub(crate) fn emit_pair(
        &self,
        other: &Self,
        b: &mut Gb,
        name: &str,
        x: ValueId,
    ) -> (ValueId, ValueId) {
        if !Arc::ptr_eq(&self.w, &other.w) {
            return (
                self.emit(b, &format!("{name}.a"), x, None),
                other.emit(b, &format!("{name}.b"), x, None),
            );
        }
        let full = b.lin_mat(&format!("{name}.fused"), x, &self.stored(), None);
        let mut part = |tag: &str, l: &Self| b.slice(&format!("{name}.{tag}"), full, 1, l.off, l.off + l.n);
        (part("a", self), part("b", other))
    }

    #[cfg(test)]
    pub(crate) fn from_mat(m: Mat) -> Self {
        let (n, k) = (m.rows, m.cols);
        Self {
            w: Arc::new(Wt::Bf16(m)),
            off: 0,
            n,
            k,
        }
    }
}

/// `MRope` positions (U32 `[n * 3]`) of per-token `(frame, height, width)` ids that may be
/// negative (the image grid is centred on zero). Each axis is shifted by a constant so its
/// minimum is 0: attention logits depend only on position differences, so a shift common to all
/// tokens (q and k alike) leaves them unchanged.
pub(crate) fn rope_positions(pos: &[[i32; 3]]) -> Vec<u32> {
    let lo = [0, 1, 2].map(|a| pos.iter().map(|p| p[a]).min().unwrap_or(0).min(0));
    pos.iter()
        .flat_map(|p| [0, 1, 2].map(|a| (p[a] - lo[a]) as u32))
        .collect()
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
                sched::card_sigmas(&card),
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
