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

/// Which network of a pipeline a LoRA module patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    Denoiser,
    /// The n-th text encoder.
    TextEncoder(usize),
}

/// One low-rank module of a LoRA file.
#[derive(Debug, Clone, Default)]
struct Pair {
    down: Option<String>,
    up: Option<String>,
    alpha: Option<String>,
    bias: Option<String>,
}

/// A LoRA file (safetensors / any `read_checkpoint` format): kohya / sd-scripts (`lora_unet_*`,
/// `lora_te{,1,2}_*`, `lora_transformer_*`, `.alpha`), diffusers / PEFT (`unet.` / `transformer.` /
/// `text_encoder{,_2}.` + `lora.down` / `lora_A`, `processor.to_q_lora`) and ComfyUI `.diff_b` bias
/// deltas. Each module adds `strength · alpha/rank · up·down` (scale 1 without alpha) to the
/// weight its key names, through `arch::load::WeightStore::add_lora`; LoHa / LoKr / DoRA are Errs.
pub struct LoraFile {
    path: std::path::PathBuf,
    manifest: crate::model::ModelManifest,
    data: Vec<u8>,
    /// (part, module key: trainer prefix stripped, `.` as `_`) -> tensors.
    modules: BTreeMap<(Part, String), Pair>,
    /// PEFT `lora_alpha` from `adapter_config.json`, when present.
    peft_alpha: Option<f32>,
}

impl LoraFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let (manifest, data) = crate::model::loaders::read_checkpoint(path)
            .map_err(|e| format!("LoRA {}: {e}", path.display()))?;
        let mut modules: BTreeMap<(Part, String), Pair> = BTreeMap::new();
        for n in manifest.tensors.keys() {
            let (module, slot) = [
                (".lora_down.weight", 0),
                (".lora.down.weight", 0),
                ("_lora.down.weight", 0),
                (".lora_A.weight", 0),
                (".lora_up.weight", 1),
                (".lora.up.weight", 1),
                ("_lora.up.weight", 1),
                (".lora_B.weight", 1),
                (".alpha", 2),
                (".diff_b", 3),
            ]
            .iter()
            .find_map(|(s, p)| Some((n.strip_suffix(s)?, *p)))
            .ok_or_else(|| {
                format!("LoRA tensor '{n}' is not a low-rank up/down/alpha weight (LoHa/LoKr/DoRA are not supported)")
            })?;
            let (part, path) = [
                ("lora_unet_", Part::Denoiser),
                ("lora_transformer_", Part::Denoiser),
                ("lora_te1_", Part::TextEncoder(0)),
                ("lora_te2_", Part::TextEncoder(1)),
                ("lora_te_", Part::TextEncoder(0)),
                ("unet.", Part::Denoiser),
                ("transformer.", Part::Denoiser),
                ("diffusion_model.", Part::Denoiser),
                ("base_model.model.", Part::Denoiser),
                ("text_encoder_2.", Part::TextEncoder(1)),
                ("text_encoder.", Part::TextEncoder(0)),
            ]
            .iter()
            .find_map(|(p, t)| Some((*t, module.strip_prefix(p)?)))
            .unwrap_or((Part::Denoiser, module));
            let mut key = path.replace(".processor.", ".").replace('.', "_");
            // Old diffusers attention processors name `to_out.0` as `to_out_lora`.
            if key.ends_with("_to_out") {
                key.push_str("_0");
            }
            let m = modules.entry((part, key)).or_default();
            let slot = match slot {
                0 => &mut m.down,
                1 => &mut m.up,
                2 => &mut m.alpha,
                _ => &mut m.bias,
            };
            *slot = Some(n.clone());
        }
        if let Some(((_, k), _)) = modules
            .iter()
            .find(|(_, m)| m.up.is_none() != m.down.is_none() || (m.up.is_none() && m.bias.is_none()))
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
        Ok(Self { path: path.to_path_buf(), manifest, data, modules, peft_alpha })
    }

    /// Every module.
    pub fn modules(&self) -> usize {
        self.modules.len()
    }

    /// Modules patching `part`.
    pub fn targets(&self, part: Part) -> usize {
        self.modules.keys().filter(|(p, _)| *p == part).count()
    }

    fn tensor(&self, n: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let t = &self.manifest.tensors[n];
        let v = crate::model::decode_dtype(&self.data[t.byte_offset..t.byte_offset + t.byte_len], &t.dtype)
            .map_err(|e| format!("LoRA {}: '{n}': {e}", self.path.display()))?;
        Ok((t.shape.clone(), v))
    }

    /// Registers every `part` module on `model` at `strength`. A module key names a weight by its
    /// canonical name, its checkpoint name, or a name the family's rename rules map to a canonical
    /// one (`.` as `_`, e.g. kohya LDM keys on a diffusers checkpoint). Returns the modules applied.
    pub fn apply(
        &self,
        part: Part,
        model: &mut crate::arch::load::Model,
        reg: &crate::arch::Registry,
        strength: f32,
    ) -> Result<usize, String> {
        let us = |s: &str| s.replace('.', "_");
        let names = model.store.names();
        let mut index: HashMap<String, String> = HashMap::new();
        for c in &names {
            let stem = c.strip_suffix(".weight").unwrap_or(c);
            index.insert(us(stem), c.clone());
            if let Some(i) = stem.find("text_model.") {
                index.entry(us(&stem[i + "text_model.".len()..])).or_insert_with(|| c.clone());
                index.entry(us(&stem[i..])).or_insert_with(|| c.clone());
            }
        }
        for (raw, c) in &model.renames {
            let stem = raw.strip_suffix(".weight").unwrap_or(raw);
            index.entry(us(stem)).or_insert_with(|| c.clone());
        }
        // The rename rules over `_`-joined names.
        let mut pairs = model.spec.names.rules.clone();
        for set in &model.spec.names.use_sets {
            pairs.extend(reg.name_set(set)?);
        }
        let ren = crate::arch::names::Renamer::from_pairs(
            &pairs.iter().map(|(a, b)| (us(a), us(b))).collect::<Vec<_>>(),
        )?
        .with_strip(model.spec.names.strip.iter().map(|p| us(p)).collect());
        let canon: HashMap<String, String> = names
            .iter()
            .map(|c| (us(c), c.clone()))
            .collect();
        let resolve = |key: &str| -> Option<String> {
            if let Some(c) = index.get(key) {
                return Some(c.clone());
            }
            ["", "model_diffusion_model_"].iter().find_map(|p| {
                let r = ren.canonical(&format!("{p}{key}_weight"));
                canon.get(&r).cloned()
            })
        };
        let mut done = 0;
        for ((p, key), m) in &self.modules {
            if *p != part {
                continue;
            }
            let target = resolve(key).ok_or_else(|| {
                format!("LoRA module '{key}' ({part:?}) matches no weight of {}", model.spec.family)
            })?;
            let bias = match &m.bias {
                Some(b) => Some(self.tensor(b)?.1.into_iter().map(|v| v * strength).collect()),
                None => None,
            };
            let (Some(down), Some(up)) = (&m.down, &m.up) else {
                return Err(format!("LoRA module '{key}': a bias delta without its low-rank pair"));
            };
            let (a, b) = (self.tensor(down)?, self.tensor(up)?);
            let rank = a.0[0];
            let alpha = match &m.alpha {
                Some(n) => Some(*self.tensor(n)?.1.first().ok_or("empty LoRA alpha")?),
                None => self.peft_alpha,
            };
            let scale = alpha.map_or(1.0, |al| al / rank as f32) * strength;
            model
                .store
                .add_lora(&target, a, b, scale, bias)
                .map_err(|e| format!("{e} (module '{key}')"))?;
            done += 1;
        }
        Ok(done)
    }
}

/// Whether a LoRA file's tensor names patch a diffusion denoiser (SD UNet blocks, FLUX / DiT
/// blocks) or CLIP text encoders, rather than an LLM.
pub fn is_diffusion_lora<'a>(mut names: impl Iterator<Item = &'a str>) -> bool {
    const BLOCKS: [&str; 13] = [
        "down_blocks",
        "up_blocks",
        "mid_block",
        "input_blocks",
        "output_blocks",
        "middle_block",
        "lora_te",
        "text_encoder",
        "text_model",
        "double_blocks",
        "single_blocks",
        "single_transformer_blocks",
        "transformer_blocks",
    ];
    names.any(|n| {
        (n.ends_with(".lora_down.weight") || n.ends_with(".lora.down.weight") || n.ends_with(".lora_A.weight"))
            && BLOCKS.iter().any(|k| n.contains(k))
            && !n.contains("model.layers.")
    })
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

    /// A matrix target gets the runtime branch (scale alpha/rank · strength); a 1x1 conv kernel has
    /// `scale · up·down` merged in; kohya keys resolve through `.`-as-`_` names.
    #[test]
    fn lora_file_patches_matrices_and_convs() {
        use crate::arch::testing::{artifacts, Weights};
        use crate::arch::{load, Registry, Source};
        let dir = std::env::temp_dir().join(format!("gibbs-lora-{}", std::process::id()));
        let mut w = Weights::new(3);
        w.w("blk.lin.weight", &[4, 3]).w("blk.conv.weight", &[4, 3, 1, 1]);
        w.write_checkpoint(&dir.join("base"), &serde_json::json!({"model_type": "toy"}), 4);
        let mut l = Weights::new(4);
        l.w("lora_unet_blk_lin.lora_down.weight", &[2, 3])
            .w("lora_unet_blk_lin.lora_up.weight", &[4, 2])
            .put("lora_unet_blk_lin.alpha", &[1], vec![1.0])
            .w("lora_unet_blk_conv.lora_down.weight", &[2, 3, 1, 1])
            .w("lora_unet_blk_conv.lora_up.weight", &[4, 2, 1, 1]);
        l.write_checkpoint(&dir.join("lora"), &serde_json::json!({}), 4);
        let spec = r#"{family: "toy", pipeline: "diffusion", match: {hf_model_type: ["toy"]}, components: {}}"#;
        let reg = Registry::from_files(&[("families/toy.json5".into(), spec.into())], &[]).unwrap();
        let mut m = load::load(&artifacts(&dir.join("base")), &reg, None).unwrap();
        let base = m.store.materialize(&["blk.conv.weight".into()]).unwrap();
        let base = base.get("blk.conv.weight").unwrap().as_f32().unwrap().to_vec();
        let f = LoraFile::open(&dir.join("lora/model.safetensors")).unwrap();
        assert_eq!(f.targets(Part::Denoiser), 2);
        assert_eq!(f.apply(Part::Denoiser, &mut m, &reg, 2.0).unwrap(), 2);
        let br = m.store.lora("blk.lin.weight").expect("runtime branch");
        assert_eq!(br.scale, 2.0 * 1.0 / 2.0);
        let merged = m.store.materialize(&["blk.conv.weight".into()]).unwrap();
        let merged = merged.get("blk.conv.weight").unwrap().as_f32().unwrap().to_vec();
        let (down, up) = (&l.tensors[3].2, &l.tensors[4].2);
        for o in 0..4 {
            for i in 0..3 {
                let d: f32 = (0..2).map(|r| up[o * 2 + r] * down[r * 3 + i]).sum::<f32>() * 2.0;
                assert!((merged[o * 3 + i] - base[o * 3 + i] - d).abs() < 1e-5);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

}
