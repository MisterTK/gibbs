//! Diffusion LoRA adapters ([`LoraFile`]) registered on a loaded family spec's weights, and the
//! model-card `base_model` they name.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

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

/// Whether a LoRA file's tensor names patch a diffusion denoiser (UNet blocks, DiT
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

    /// FLUX LoRA key forms (ported from the deleted `flux.rs` test): PEFT/diffusers keys on
    /// diffusers names, kohya keys on a BFL-named base (resolved through the `bfl` renames).
    #[test]
    fn flux_lora_keys_resolve() {
        use crate::arch::testing::{artifacts, Weights};
        use crate::arch::{load, registry, Registry};
        let dir = std::env::temp_dir().join(format!("gibbs-lora-flux-{}", std::process::id()));
        let mut w = Weights::new(5);
        w.w("double_blocks.0.img_attn.qkv.weight", &[6, 2])
            .w("single_blocks.0.linear1.weight", &[6, 2])
            .w("single_transformer_blocks.3.proj_mlp.weight", &[4, 2])
            .w("transformer_blocks.2.ff_context.net.0.proj.weight", &[4, 2]);
        w.write_checkpoint(&dir.join("base"), &serde_json::json!({"model_type": "toy"}), 4);
        let mut l = Weights::new(6);
        for (m, out) in [
            ("lora_unet_double_blocks_0_img_attn_qkv", 6),
            ("lora_unet_single_blocks_0_linear1", 6),
            ("transformer.single_transformer_blocks.3.proj_mlp", 4),
            ("transformer.transformer_blocks.2.ff_context.net.0.proj", 4),
        ] {
            l.w(&format!("{m}.lora_down.weight"), &[1, 2]).w(&format!("{m}.lora_up.weight"), &[out, 1]);
        }
        l.write_checkpoint(&dir.join("lora"), &serde_json::json!({}), 4);
        let bfl = registry().unwrap().name_set("bfl").unwrap();
        let spec = format!(
            r#"{{family: "toy", pipeline: "diffusion", match: {{hf_model_type: ["toy"]}}, names: {{rules: {}}}, components: {{}}}}"#,
            serde_json::to_string(&bfl).unwrap()
        );
        let reg = Registry::from_files(&[("families/toy.json5".into(), spec)], &[]).unwrap();
        let mut m = load::load(&artifacts(&dir.join("base")), &reg, None).unwrap();
        let f = LoraFile::open(&dir.join("lora/model.safetensors")).unwrap();
        // `apply` errors on any module that matches no weight: 4 = every key form resolved.
        assert_eq!(f.apply(Part::Denoiser, &mut m, &reg, 1.0).unwrap(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

}
