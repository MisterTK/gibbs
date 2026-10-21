//! The one model-load dispatch: a `--model` id to a loaded model of whatever kind. Families with a
//! spec load through [`crate::arch`]; the others keep their loaders here, one arm each (marked
//! `legacy: until <phase>`), until their phase ports them.

use super::expr::Val;
use super::Pipeline;
use crate::hardware::{Device, HardwareCapability, PlacementMode, QuantMode};
use crate::model::classifier::{self, SequenceClassifier};
use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::model::jev::JevSystem1Model;
use crate::model::text_to_image::{load_image_model, ImageGenerator};
use crate::model::weight_format::{
    load_causal_lm_with, reencode_override, set_reencode_override, Choice, ReencodeConfig,
    ReencodePolicy,
};
use crate::scheduler::ExecutableModel;
use serde_json::{json, Map, Value as J};
use std::collections::HashSet;
use std::sync::Arc;

pub enum Loaded {
    /// A causal LM; `choice` is the GraphLm weight-format decision (legacy: until Phase 4).
    Lm {
        model: ExecutableModel,
        choice: Option<Choice>,
        artifacts: ModelArtifacts,
    },
    Classifier(SequenceClassifier),
    System1(JevSystem1Model),
    Image(Arc<dyn ImageGenerator>),
    /// A text-to-video plan, loaded per generation ([`crate::model::video_repo::load`]).
    Video(crate::model::video_repo::Plan),
}

pub struct LoadOpts<'a> {
    pub placement: PlacementMode,
    pub token: Option<&'a str>,
    /// `--quant` as the GraphLm loader reads it (legacy: until Phase 4).
    pub quant: Option<QuantMode>,
    /// `--quant` verbatim: the GGUF pick of a video repo.
    pub quant_name: Option<&'a str>,
    pub hw: &'a HardwareCapability,
    /// The re-encode configuration every loader runs under.
    pub reencode: ReencodeConfig,
    /// `--arch`: load this family spec, skipping detection.
    pub family: Option<&'a str>,
    /// Receives the stage the load is in, for progress text and failure records.
    pub stage: &'a dyn Fn(&'static str),
    /// Runs before any download of a repo no hub-level route took, with whether it is a PEFT
    /// adapter (merged into a snapshot of its base): an error refuses the load.
    pub admit: &'a dyn Fn(bool) -> Result<(), String>,
    /// The CLI's legacy order (legacy: until Phase 7): System 1 encoders are detected before
    /// classifiers and run on Metal wherever it exists; the TUI tries classifiers first and keeps
    /// System 1 on its own backend.
    pub system1_first: bool,
}

/// `GIBBS_SPEC_LM=1`: causal LMs a family spec claims load through [`crate::runtime::lm`]
/// instead of GraphLm.
pub fn spec_lm_enabled() -> bool {
    std::env::var("GIBBS_SPEC_LM").is_ok_and(|v| v == "1")
}

fn device(p: PlacementMode) -> Device {
    match p {
        PlacementMode::Pin(d) => d,
        PlacementMode::Auto => crate::hardware::default_device(),
    }
}

/// Resolves, downloads and loads `id`.
pub fn load_model(hub: &HuggingFaceHub, id: &str, o: &LoadOpts) -> Result<Loaded, String> {
    (o.stage)("resolving");
    if o.family.is_none() {
        // legacy: until Phase 7 (encoder specs)
        let system1 = || {
            let dev = match device(o.placement) {
                // The CLI runs the System 1 encoder on Metal wherever it exists.
                Device::Cpu
                    if o.system1_first
                        && crate::hardware::available_devices().contains(&Device::Metal) =>
                {
                    Device::Metal
                }
                d => d,
            };
            JevSystem1Model::load(hub, id, dev).map(Loaded::System1)
        };
        if o.system1_first && crate::model::jev::detect(hub, id) {
            return system1();
        }
        if classifier::detect(hub, id).is_some() {
            return SequenceClassifier::load(hub, id, device(o.placement)).map(Loaded::Classifier);
        }
        if !o.system1_first && crate::model::jev::detect(hub, id) {
            return system1();
        }
        if let Some(v) = crate::model::video_repo::detect(hub, id, o.quant_name) {
            return v.map(Loaded::Video);
        }
    }
    (o.stage)(if hub.locate_model(id).weights_path.is_some() {
        "reading cached files"
    } else {
        "downloading"
    });
    // A PEFT LoRA CAUSAL_LM adapter becomes a merged snapshot of its base first.
    let adapter = crate::model::peft::prepare_causal_lm(hub, id, o.token)?;
    (o.admit)(adapter)?;
    let artifacts = hub.auto_download_and_cache(id, o.token)?;
    (o.stage)("reading weights, choosing weight format");
    // legacy: until Phase 9 (diffusion specs). Image repos are recognised by their files.
    if o.family.is_none() {
        if let Some(img) = load_image_model(&artifacts, hub) {
            return img.map(Loaded::Image);
        }
    }
    let (model, choice) = load_lm(&artifacts, o)?;
    Ok(Loaded::Lm {
        model,
        choice,
        artifacts,
    })
}

/// Loads `artifacts` as a causal LM under `o.reencode`.
pub fn load_lm(
    a: &ModelArtifacts,
    o: &LoadOpts,
) -> Result<(ExecutableModel, Option<Choice>), String> {
    let previous = reencode_override();
    set_reencode_override(Some(o.reencode.clone()));
    let r = lm(a, o);
    set_reencode_override(previous);
    r
}

fn lm(a: &ModelArtifacts, o: &LoadOpts) -> Result<(ExecutableModel, Option<Choice>), String> {
    let spec = |family: Option<&str>| -> Result<(ExecutableModel, Option<Choice>), String> {
        let m = crate::runtime::lm::load(a, super::registry()?, family, o.placement)?;
        Ok((ExecutableModel::Native(m), None))
    };
    // Checkpoints the arch reader cannot open, and registry errors, leave the GraphLm path as
    // it was (a spec problem fails only spec loads).
    let ck = super::load::open(a).ok();
    let family = ck.as_ref().and_then(|ck| {
        let reg = super::registry().map_err(|e| tracing::warn!("{e}")).ok()?;
        let f = o.family.map(String::from).or_else(|| reg.detect(&ck.sig))?;
        let p = reg.spec(&f, &ck.sig).map_err(|e| tracing::warn!("{e}")).ok()?.pipeline;
        Some((f, p))
    });
    if let Some((f, Pipeline::Encdec)) = &family {
        let m = crate::runtime::encdec::load(a, super::registry()?, Some(f), o.placement)?;
        return Ok((ExecutableModel::Native(m), None));
    }
    if o.family.is_some() {
        return spec(o.family);
    }
    // legacy: until Phases 5 and 6 (the native loaders' families)
    if let Some(m) = crate::model::native::load_native(a, o.placement) {
        return Ok((ExecutableModel::Native(m?), None));
    }
    match family {
        Some((f, Pipeline::Unsupported)) => return spec(Some(&f)),
        Some((f, Pipeline::CausalLm)) if spec_lm_enabled() => return spec(Some(&f)),
        _ => {}
    }
    // legacy: until Phase 4 (GraphLm)
    let (lm, choice) = load_causal_lm_with(a, device(o.placement), o.quant, o.hw, &o.reencode)?;
    (o.stage)("compiling + tuning kernels");
    Ok((
        ExecutableModel::load_graph_lm(lm, o.placement)?,
        Some(choice),
    ))
}

/// `--arch-dump`: the merged spec, the resolved config vars, every raw tensor's canonical name
/// and the raw tensors no component instance reads, as JSON.
pub fn dump(artifacts: &ModelArtifacts, family: Option<&str>) -> Result<String, String> {
    let reg = super::registry()?;
    let exact = ReencodeConfig {
        policy: ReencodePolicy::BitExact,
        ..ReencodeConfig::effective()
    };
    let m = super::load::load_with(artifacts, reg, family, exact)?;
    let config: Map<String, J> = m
        .spec
        .config
        .keys()
        .map(|k| {
            let v = m
                .env
                .var(k)
                .map_or_else(|e| J::String(format!("error: {e}")), |v| v.to_json());
            (k.clone(), v)
        })
        .collect();
    let (mut used, mut unbuilt) = (HashSet::new(), Map::new());
    let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
    for name in m.spec.components.keys() {
        match super::build::build(&m.spec, name, &m.env, &m.store, &vars) {
            Ok(i) => used.extend(i.weights),
            Err(e) => {
                unbuilt.insert(name.clone(), J::String(e));
            }
        }
    }
    let out = json!({
        "family": m.spec.family,
        "spec": reg.merged(&m.spec.family, &m.ckpt.sig)?,
        "config": config,
        "tensors": m.renames,
        "unconsumed": m.store.unconsumed(&used, &m.spec.ignore),
        "unbuilt_components": unbuilt,
    });
    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
}
