//! Stable Diffusion LoRA adapters merged into the base weights at load: kohya / sd-scripts names
//! (`lora_unet_*`, `lora_te_*`, `lora_te1_*`, `lora_te2_*` with `.alpha`, diffusers- or LDM-style
//! module paths) and diffusers / PEFT names (`unet.<module>.lora.down.weight`, `lora_A` / `lora_B`,
//! `processor.to_q_lora`). Each module adds `scale * up . down` (scale = alpha / rank, else 1) to its
//! target weight; a key that matches no base weight, or a format beyond plain low-rank pairs
//! (LoHa, LoKr, DoRA), is an Err.

use crate::model::clip::{Delta, Tensors};
use crate::model::loaders::{SafetensorsWeightReader, WeightReader};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Which base component a LoRA module patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Unet,
    TextEncoder,
    TextEncoder2,
}

#[derive(Default)]
struct Module {
    up: Option<String>,
    down: Option<String>,
    alpha: Option<String>,
}

pub struct Lora {
    reader: SafetensorsWeightReader,
    shapes: HashMap<String, Vec<usize>>,
    /// (target, module key with `.` and `_` normalised to `_`) -> tensor names.
    modules: BTreeMap<(Target, String), Module>,
    /// PEFT `lora_alpha` from `adapter_config.json`, when present.
    peft_alpha: Option<f32>,
}

/// Whether the tensor names of a safetensors file are a Stable Diffusion LoRA: low-rank pairs on SD
/// UNet blocks (diffusers or LDM names) or on the CLIP text encoders. LLM (`layers`) and FLUX
/// (`double_blocks`, `transformer_blocks`) adapters are not.
pub fn is_lora<'a>(mut names: impl Iterator<Item = &'a str>) -> bool {
    const SD: [&str; 9] = [
        "down_blocks",
        "up_blocks",
        "mid_block",
        "input_blocks",
        "output_blocks",
        "middle_block",
        "lora_te",
        "text_encoder",
        "text_model",
    ];
    names.any(|n| {
        (n.ends_with(".lora_down.weight")
            || n.ends_with(".lora.down.weight")
            || n.ends_with(".lora_A.weight"))
            && SD.iter().any(|k| n.contains(k))
    })
}

impl Lora {
    pub fn open(path: &Path) -> Result<Self, String> {
        let reader = SafetensorsWeightReader::new(path)
            .map_err(|e| format!("LoRA {}: {e}", path.display()))?;
        let mut modules: BTreeMap<(Target, String), Module> = BTreeMap::new();
        let mut shapes = HashMap::new();
        for meta in reader.list_tensors()? {
            shapes.insert(meta.name.clone(), meta.shape.clone());
            let n = meta.name.as_str();
            let (module, part) = [
                (".lora_down.weight", 0),
                (".lora.down.weight", 0),
                ("_lora.down.weight", 0),
                (".lora_A.weight", 0),
                (".lora_up.weight", 1),
                (".lora.up.weight", 1),
                ("_lora.up.weight", 1),
                (".lora_B.weight", 1),
                (".alpha", 2),
            ]
            .iter()
            .find_map(|(s, p)| Some((n.strip_suffix(s)?, *p)))
            .ok_or_else(|| format!("LoRA tensor '{n}' is not a low-rank up/down/alpha weight (LoHa/LoKr/DoRA are not supported)"))?;
            let (target, path) = [
                ("lora_unet_", Target::Unet),
                ("lora_te1_", Target::TextEncoder),
                ("lora_te2_", Target::TextEncoder2),
                ("lora_te_", Target::TextEncoder),
                ("unet.", Target::Unet),
                ("text_encoder_2.", Target::TextEncoder2),
                ("text_encoder.", Target::TextEncoder),
            ]
            .iter()
            .find_map(|(p, t)| Some((*t, module.strip_prefix(p)?)))
            .unwrap_or((Target::Unet, module));
            let mut key = path.replace(".processor.", ".").replace('.', "_");
            // Old diffusers attention processors name `to_out.0` as `to_out_lora`.
            if key.ends_with("_to_out") {
                key.push_str("_0");
            }
            let m = modules.entry((target, key)).or_default();
            let slot = match part {
                0 => &mut m.down,
                1 => &mut m.up,
                _ => &mut m.alpha,
            };
            *slot = Some(meta.name.clone());
        }
        if let Some(((_, k), _)) = modules
            .iter()
            .find(|(_, m)| m.up.is_none() || m.down.is_none())
        {
            return Err(format!("LoRA module '{k}' lacks its up or down weight"));
        }
        if modules.is_empty() {
            return Err(format!("{} has no LoRA tensors", path.display()));
        }
        let peft_alpha = path
            .parent()
            .and_then(|d| std::fs::read_to_string(d.join("adapter_config.json")).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|c| c["lora_alpha"].as_f64())
            .map(|a| a as f32);
        Ok(Self {
            reader,
            shapes,
            modules,
            peft_alpha,
        })
    }

    pub fn targets(&self, t: Target) -> usize {
        self.modules.keys().filter(|(k, _)| *k == t).count()
    }

    /// Registers every `target` module as a delta on `base`, whose weights are indexed by their names
    /// past `strip` (e.g. `model.diffusion_model.`) and past any `text_model.` prefix, with `.` as `_`.
    /// Returns the number of patched weights.
    pub fn apply(&self, target: Target, base: &mut Tensors, strip: &str) -> Result<usize, String> {
        let mut index: HashMap<String, String> = HashMap::new();
        for name in base.names() {
            let Some(stem) = name
                .strip_prefix(strip)
                .and_then(|n| n.strip_suffix(".weight"))
            else {
                continue;
            };
            index.insert(stem.replace('.', "_"), name.clone());
            if let Some(i) = stem.find("text_model.") {
                index.insert(stem[i..].replace('.', "_"), name.clone());
            }
        }
        let mut done = 0;
        for ((t, key), m) in &self.modules {
            if *t != target {
                continue;
            }
            let name = index
                .get(key)
                .ok_or_else(|| format!("LoRA module '{key}' ({target:?}) matches no base weight"))?
                .clone();
            let (up_n, down_n) = (
                m.up.as_deref().unwrap_or_default(),
                m.down.as_deref().unwrap_or_default(),
            );
            let shape = |n: &str| {
                self.shapes
                    .get(n)
                    .cloned()
                    .ok_or_else(|| format!("LoRA tensor {n} missing"))
            };
            let rank = shape(down_n)?[0];
            let up = self.reader.load_tensor(up_n)?;
            let down = self.reader.load_tensor(down_n)?;
            let alpha = match &m.alpha {
                Some(a) => Some(
                    self.reader
                        .load_tensor(a)?
                        .first()
                        .copied()
                        .ok_or("empty LoRA alpha")?,
                ),
                None => self.peft_alpha,
            };
            let scale = alpha.map_or(1.0, |a| a / rank as f32);
            base.add_delta(
                &name,
                Delta {
                    up,
                    down,
                    rank,
                    scale,
                },
            )
            .map_err(|e| format!("{e} (module '{key}')"))?;
            done += 1;
        }
        Ok(done)
    }
}

/// `base_model` of a model card (`README.md` YAML front matter): `base_model: x` or a list.
pub fn card_base_model(readme: &str) -> Option<String> {
    let front = readme.strip_prefix("---")?.split("\n---").next()?;
    let mut lines = front.lines();
    while let Some(l) = lines.next() {
        let Some(v) = l.strip_prefix("base_model:") else {
            continue;
        };
        let v = v.trim().trim_matches(['"', '\'']);
        if !v.is_empty() {
            return Some(v.to_string());
        }
        let item = lines.next()?.trim().strip_prefix("- ")?;
        return Some(item.trim().trim_matches(['"', '\'']).to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SD LoRAs (kohya LDM, kohya diffusers, PEFT) are recognised; LLM and FLUX adapters are not,
    /// so loading them never fetches an SD base.
    #[test]
    fn only_sd_adapters_are_sd_loras() {
        for sd in [
            "lora_unet_input_blocks_4_1_proj_in.lora_down.weight",
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight",
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
        ] {
            assert!(is_lora([sd].into_iter()), "{sd}");
        }
        for other in [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
            "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
            "model.diffusion_model.input_blocks.0.0.weight",
        ] {
            assert!(!is_lora([other].into_iter()), "{other}");
        }
    }

    #[test]
    fn card_base_model_forms() {
        let a = "---\nlicense: mit\nbase_model: stabilityai/stable-diffusion-xl-base-1.0\ntags:\n- lora\n---\n# x";
        assert_eq!(
            card_base_model(a).as_deref(),
            Some("stabilityai/stable-diffusion-xl-base-1.0")
        );
        let b = "---\nbase_model:\n- 'runwayml/stable-diffusion-v1-5'\n---\n";
        assert_eq!(
            card_base_model(b).as_deref(),
            Some("runwayml/stable-diffusion-v1-5")
        );
        assert_eq!(card_base_model("# no front matter"), None);
    }
}
