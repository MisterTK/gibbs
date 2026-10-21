//! Model architectures as data. Each family is a JSON5 spec under `arch/families/` (embedded at
//! build time, see [`EMBEDDED`]); per-checkpoint patches live under `arch/overrides/`, and users
//! add or replace both under `<cache>/arch/{families,overrides}/`. [`load`] turns a matched spec
//! and a checkpoint into weights, [`build`] turns a spec component into a `graph::Graph`.
//!
//! Merge order: family <- its `extends` chain (parent first) <- matching embedded overrides (file
//! name order) <- user family file of the same name (replaces) <- matching user overrides. Merging is
//! RFC 7386 merge patch: objects merge, everything else (arrays included) replaces, `null` deletes.

pub mod build;
pub mod dispatch;
pub mod expr;
pub mod gen;
pub mod load;
pub mod names;
pub mod templates;
#[cfg(test)]
pub mod testing;
// Parity with the pre-spec implementations (deleted with them).
#[cfg(test)]
mod parity_gemma4_text;
#[cfg(test)]
mod parity_gemma4_vision;
#[cfg(test)]
mod parity_gemma4_audio;
#[cfg(test)]
mod parity_gemma3;

pub use dispatch::{load_model, LoadOpts, Loaded};

use serde::Deserialize;
use serde_json::{Map, Value as J};
use std::path::Path;

/// Every embedded spec file: (path under `arch/`, contents).
pub const EMBEDDED: &[(&str, &str)] = &[
    ("lib/transformer.json5", include_str!("../../arch/lib/transformer.json5")),
    ("lib/qwen.json5", include_str!("../../arch/lib/qwen.json5")),
    ("lib/deltanet.json5", include_str!("../../arch/lib/deltanet.json5")),
    ("lib/moe.json5", include_str!("../../arch/lib/moe.json5")),
    ("names/qwen35.json5", include_str!("../../arch/names/qwen35.json5")),
    ("lib/qwen_vit.json5", include_str!("../../arch/lib/qwen_vit.json5")),
    ("names/llamacpp.json5", include_str!("../../arch/names/llamacpp.json5")),
    ("families/gemma3.json5", include_str!("../../arch/families/gemma3.json5")),
    ("families/llama.json5", include_str!("../../arch/families/llama.json5")),
    ("families/glm.json5", include_str!("../../arch/families/glm.json5")),
    ("families/granite.json5", include_str!("../../arch/families/granite.json5")),
    ("families/bitnet.json5", include_str!("../../arch/families/bitnet.json5")),
    ("families/gpt_neox.json5", include_str!("../../arch/families/gpt_neox.json5")),
    ("families/gpt2.json5", include_str!("../../arch/families/gpt2.json5")),
    ("families/opt.json5", include_str!("../../arch/families/opt.json5")),
    ("families/lfm2.json5", include_str!("../../arch/families/lfm2.json5")),
    ("families/qwen3_5.json5", include_str!("../../arch/families/qwen3_5.json5")),
    ("families/qwen3.json5", include_str!("../../arch/families/qwen3.json5")),
    ("families/hunyuan.json5", include_str!("../../arch/families/hunyuan.json5")),
    ("families/qwen3_vl.json5", include_str!("../../arch/families/qwen3_vl.json5")),
    ("families/qwen4_exp.json5", include_str!("../../arch/families/qwen4_exp.json5")),
    ("families/dflash.json5", include_str!("../../arch/families/dflash.json5")),
    ("families/gemma4.json5", include_str!("../../arch/families/gemma4.json5")),
    ("families/gemma4_unified.json5", include_str!("../../arch/families/gemma4_unified.json5")),
    ("families/unsupported.json5", include_str!("../../arch/families/unsupported.json5")),
];

/// Tensor facts the interpreter needs: canonical weights and generated constants.
pub trait Source {
    /// Logical shape and graph dtype of a weight/constant.
    fn tensor(&self, name: &str) -> Option<(Vec<usize>, crate::graph::DType)>;
    /// The low-rank adapter registered for a `Linear` weight.
    fn lora(&self, _weight: &str) -> Option<Lora> {
        None
    }
}

/// A LoRA pair: `y += (x · Aᵀ · Bᵀ) · scale (+ bias_delta)`.
#[derive(Debug, Clone)]
pub struct Lora {
    pub a: String,
    pub b: String,
    pub scale: f32,
    pub bias_delta: Option<String>,
}

/// Which checkpoints a family claims.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Match {
    #[serde(default)]
    pub hf_model_type: Vec<String>,
    #[serde(default)]
    pub hf_architectures: Vec<String>,
    #[serde(default)]
    pub gguf_arch: Vec<String>,
    #[serde(default)]
    pub diffusers_class: Vec<String>,
    #[serde(default)]
    pub file_ext: Vec<String>,
    /// Claims causal LMs no other family matches; tried after every non-fallback family.
    #[serde(default)]
    pub fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pipeline {
    CausalLm,
    Encoder,
    Encdec,
    Diffusion,
    Video,
    Drafter,
    BagClassifier,
    Unsupported,
}

/// A load refusal: `expr` true means the checkpoint is refused with `error` (interpolated).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub expr: String,
    pub error: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Names {
    /// Shared rule sets from `arch/names/<set>.json5`, applied after `rules`.
    #[serde(default, rename = "use")]
    pub use_sets: Vec<String>,
    #[serde(default)]
    pub rules: Vec<(String, String)>,
}

/// A load-time value transform on a canonical tensor (`tensor` may hold `{i}` captures).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub tensor: String,
    pub op: String,
    #[serde(default)]
    pub args: J,
    #[serde(default)]
    pub when: Option<String>,
}

/// A generated weight tensor.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constant {
    pub gen: String,
    #[serde(default)]
    pub args: J,
    #[serde(default)]
    pub when: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Macro {
    #[serde(default)]
    pub params: Vec<String>,
    pub body: Vec<J>,
    /// Scope name (or list of names) the macro's `out` binds to; default `out`.
    #[serde(default)]
    pub returns: J,
}

/// A graph input of a network.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDecl {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<J>,
    #[serde(default)]
    pub when: Option<String>,
    /// Generator the runner fills this input with (see `gen`); absent = caller supplies it.
    #[serde(default)]
    pub gen: Option<String>,
    #[serde(default)]
    pub args: J,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Network {
    #[serde(default)]
    pub vars: Vec<String>,
    #[serde(default)]
    pub inputs: Vec<InputDecl>,
    pub body: Vec<J>,
    /// Names, or `{name, value?, when?}` objects (`value`: the scope value output as `name`).
    #[serde(default)]
    pub outputs: Vec<J>,
}

/// A component: an inline network, or another family's spec run on a checkpoint subfolder.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Component {
    Ref {
        family: String,
        #[serde(default)]
        dir: Option<String>,
    },
    Net(Network),
}

/// A fully merged family spec.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub family: String,
    #[serde(default)]
    pub extends: Option<String>,
    #[serde(default, rename = "match")]
    pub matcher: Match,
    pub pipeline: Pipeline,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub use_lib: Vec<String>,
    /// var -> expression; evaluated lazily on first use (order free, cycles are errors).
    #[serde(default)]
    pub config: Map<String, J>,
    #[serde(default)]
    pub checks: Vec<Check>,
    #[serde(default)]
    pub names: Names,
    #[serde(default)]
    pub transforms: Vec<Transform>,
    #[serde(default)]
    pub constants: Map<String, J>,
    #[serde(default)]
    pub ignore: Vec<String>,
    #[serde(default)]
    pub macros: Map<String, J>,
    #[serde(default)]
    pub components: Map<String, J>,
    #[serde(default)]
    pub runtime: J,
    /// Random-init weights for synthetic model ids: name pattern -> `{shape: [expr..], over?:
    /// {var: list-expr}}`; `over` instantiates `{var}` in the pattern for every list item.
    #[serde(default)]
    pub synthetic: Map<String, J>,
}

impl Spec {
    pub fn component(&self, name: &str) -> Result<Component, String> {
        let c = self
            .components
            .get(name)
            .ok_or_else(|| format!("{}: no component `{name}`", self.family))?;
        serde_json::from_value(c.clone())
            .map_err(|e| format!("{}: component `{name}`: {e}", self.family))
    }

    pub fn macro_def(&self, name: &str) -> Result<Macro, String> {
        let m = self
            .macros
            .get(name)
            .ok_or_else(|| format!("{}: no macro `{name}`", self.family))?;
        serde_json::from_value(m.clone())
            .map_err(|e| format!("{}: macro `{name}`: {e}", self.family))
    }

    /// `runtime` deserialized as a pipeline's own type.
    pub fn runtime_as<T: serde::de::DeserializeOwned>(&self) -> Result<T, String> {
        let rt = if self.runtime.is_null() {
            J::Object(Map::new())
        } else {
            self.runtime.clone()
        };
        serde_json::from_value(rt).map_err(|e| format!("{}: runtime: {e}", self.family))
    }

    /// An `unsupported` family's reason for `name`: the first `runtime.reasons` `[glob, reason]`
    /// pair matching it, else `reason`.
    pub fn unsupported_reason(&self, name: &str) -> String {
        self.runtime["reasons"]
            .as_array()
            .into_iter()
            .flatten()
            .find_map(|r| match (r[0].as_str(), r[1].as_str()) {
                (Some(g), Some(why)) if glob(g, name) => Some(why.to_string()),
                _ => None,
            })
            .or_else(|| self.reason.clone())
            .unwrap_or_else(|| format!("family `{}` is not supported", self.family))
    }
}

/// An override file: a partial spec applied to checkpoints it names.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Override {
    pub applies_to: AppliesTo,
    pub patch: J,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppliesTo {
    #[serde(default)]
    pub hub_id: Vec<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub gguf_arch: Option<String>,
    #[serde(default)]
    pub file_name: Vec<String>,
}

/// What a checkpoint says about itself, for matching.
#[derive(Debug, Clone, Default)]
pub struct Signals {
    pub hub_id: String,
    pub file_name: String,
    pub hf_model_type: Option<String>,
    pub hf_architectures: Vec<String>,
    pub gguf_arch: Option<String>,
    pub diffusers_class: Option<String>,
    pub causal_lm: bool,
}

impl Signals {
    fn ext(&self) -> &str {
        self.file_name.rsplit_once('.').map_or("", |(_, e)| e)
    }
}

/// RFC 7386 merge patch of `patch` into `target`.
pub fn merge_patch(target: &mut J, patch: &J) {
    let J::Object(p) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = J::Object(Map::new());
    }
    let t = target.as_object_mut().unwrap();
    for (k, v) in p {
        if v.is_null() {
            t.remove(k);
        } else {
            merge_patch(t.entry(k.clone()).or_insert(J::Null), v);
        }
    }
}

/// Parses JSON5 `text` (named `what` in errors).
pub fn parse_json5(what: &str, text: &str) -> Result<J, String> {
    json5::from_str::<J>(text).map_err(|e| format!("{what}: {e}"))
}

/// `*` and `?` glob match (no path semantics).
pub fn glob(pat: &str, s: &str) -> bool {
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.split_first() {
            None => s.is_empty(),
            Some((b'*', rest)) => (0..=s.len()).any(|i| go(rest, &s[i..])),
            Some((b'?', rest)) => !s.is_empty() && go(rest, &s[1..]),
            Some((c, rest)) => s.first() == Some(c) && go(rest, &s[1..]),
        }
    }
    go(pat.as_bytes(), s.as_bytes())
}

/// Embedded and user spec files.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    /// family name -> raw spec (unmerged), in registry order.
    families: Vec<(String, J)>,
    /// (file name, override), embedded first (file-name order), then user files (file-name order).
    overrides: Vec<(String, Override)>,
    lib: Vec<(String, J)>,
    names: Vec<(String, J)>,
}

impl Registry {
    /// The embedded specs plus the user's `<dir>/{families,overrides}/*.json5` (when `dir` exists).
    pub fn load(user_dir: Option<&Path>) -> Result<Self, String> {
        let mut files: Vec<(String, String)> = EMBEDDED
            .iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect();
        let mut user: Vec<(String, String)> = Vec::new();
        if let Some(dir) = user_dir {
            for sub in ["families", "overrides"] {
                let Ok(rd) = std::fs::read_dir(dir.join(sub)) else {
                    continue;
                };
                let mut paths: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
                paths.sort();
                for p in paths
                    .into_iter()
                    .filter(|p| p.extension().is_some_and(|e| e == "json5"))
                {
                    let text =
                        std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                    user.push((
                        format!("{sub}/{}", p.file_name().unwrap().to_string_lossy()),
                        text,
                    ));
                }
            }
        }
        Self::from_files(&files.drain(..).collect::<Vec<_>>(), &user)
    }

    /// Registry from (path, text) lists: `embedded` then `user` (user families replace embedded
    /// ones of the same `family`; user overrides apply after embedded ones).
    pub fn from_files(
        embedded: &[(String, String)],
        user: &[(String, String)],
    ) -> Result<Self, String> {
        let mut r = Registry::default();
        for (is_user, list) in [(false, embedded), (true, user)] {
            let mut ovs: Vec<(String, Override)> = Vec::new();
            for (path, text) in list {
                let j = parse_json5(path, text)?;
                let (dir, file) = path.rsplit_once('/').unwrap_or(("", path));
                match dir {
                    "families" => {
                        let name = j["family"]
                            .as_str()
                            .ok_or_else(|| format!("{path}: missing `family`"))?
                            .to_string();
                        if let Some(slot) = r.families.iter_mut().find(|(n, _)| *n == name) {
                            if !is_user {
                                return Err(format!("{path}: family `{name}` defined twice"));
                            }
                            slot.1 = j;
                        } else {
                            r.families.push((name, j));
                        }
                    }
                    "overrides" => ovs.push((
                        file.to_string(),
                        serde_json::from_value(j).map_err(|e| format!("{path}: {e}"))?,
                    )),
                    "lib" => r.lib.push((file.trim_end_matches(".json5").to_string(), j)),
                    "names" => r
                        .names
                        .push((file.trim_end_matches(".json5").to_string(), j)),
                    _ => {}
                }
            }
            ovs.sort_by(|a, b| a.0.cmp(&b.0));
            r.overrides.extend(ovs);
        }
        Ok(r)
    }

    pub fn family_names(&self) -> impl Iterator<Item = &str> {
        self.families.iter().map(|(n, _)| n.as_str())
    }

    fn raw(&self, name: &str) -> Result<&J, String> {
        self.families
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, j)| j)
            .ok_or_else(|| format!("unknown family `{name}`"))
    }

    /// `name` with its `extends` chain merged (parent first), without overrides.
    pub fn base(&self, name: &str) -> Result<J, String> {
        let mut chain = vec![self.raw(name)?.clone()];
        while let Some(parent) = chain.last().unwrap()["extends"].as_str() {
            if chain.len() > 16 {
                return Err(format!("{name}: `extends` cycle"));
            }
            chain.push(self.raw(parent)?.clone());
        }
        let mut merged = J::Object(Map::new());
        for layer in chain.iter().rev() {
            merge_patch(&mut merged, layer);
        }
        // The leaf's own name and match win; a parent's `match` must not claim the child's checkpoints.
        merged["family"] = J::String(name.to_string());
        merged["match"] = chain[0]
            .get("match")
            .cloned()
            .unwrap_or(J::Object(Map::new()));
        Ok(merged)
    }

    /// The merged spec of `family` for checkpoint `sig`: base + every applicable override.
    pub fn spec(&self, family: &str, sig: &Signals) -> Result<Spec, String> {
        serde_json::from_value(self.merged(family, sig)?)
            .map_err(|e| format!("family `{family}`: {e}"))
    }

    /// [`Self::spec`] as JSON (what `--arch-dump` prints).
    pub fn merged(&self, family: &str, sig: &Signals) -> Result<J, String> {
        let mut j = self.base(family)?;
        for (_, ov) in &self.overrides {
            if applies(&ov.applies_to, family, sig) {
                merge_patch(&mut j, &ov.patch);
            }
        }
        // Library macros named in `use_lib` (spec macros of the same name win).
        let mut macros = Map::new();
        for lib in j["use_lib"].as_array().cloned().unwrap_or_default() {
            let lib = lib.as_str().ok_or("use_lib: expected names")?;
            let l = self
                .lib
                .iter()
                .find(|(n, _)| n == lib)
                .ok_or_else(|| format!("{family}: unknown lib `{lib}`"))?;
            if let Some(m) = l.1["macros"].as_object() {
                macros.extend(m.clone());
            }
        }
        if let Some(m) = j["macros"].as_object() {
            macros.extend(m.clone());
        }
        j["macros"] = J::Object(macros);
        Ok(j)
    }

    /// Rename rules of a shared set.
    pub fn name_set(&self, set: &str) -> Result<Vec<(String, String)>, String> {
        let s = self
            .names
            .iter()
            .find(|(n, _)| n == set)
            .ok_or_else(|| format!("unknown name set `{set}`"))?;
        serde_json::from_value(s.1["rules"].clone()).map_err(|e| format!("names/{set}: {e}"))
    }

    /// The family claiming `sig`: first non-fallback match in registry order, then fallbacks.
    pub fn detect(&self, sig: &Signals) -> Option<String> {
        let matchers: Vec<(&str, Match)> = self
            .families
            .iter()
            .filter_map(|(n, j)| {
                Some((
                    n.as_str(),
                    serde_json::from_value(j.get("match")?.clone()).ok()?,
                ))
            })
            .collect();
        let any = |pats: &[String], s: &str| pats.iter().any(|p| glob(p, s));
        let hit = |m: &Match| {
            sig.hf_model_type
                .as_ref()
                .is_some_and(|t| any(&m.hf_model_type, t))
                || sig
                    .hf_architectures
                    .iter()
                    .any(|a| any(&m.hf_architectures, a))
                || sig.gguf_arch.as_ref().is_some_and(|a| any(&m.gguf_arch, a))
                || sig
                    .diffusers_class
                    .as_ref()
                    .is_some_and(|c| any(&m.diffusers_class, c))
                || (!sig.ext().is_empty() && m.file_ext.iter().any(|e| e == sig.ext()))
        };
        matchers
            .iter()
            .find(|(_, m)| !m.fallback && hit(m))
            .or_else(|| {
                matchers
                    .iter()
                    .find(|(_, m)| m.fallback && (hit(m) || sig.causal_lm))
            })
            .map(|(n, _)| n.to_string())
    }

    /// Every family's matcher, for listings (hub picker, README).
    pub fn matchers(&self) -> Vec<(String, Match, Pipeline)> {
        self.families
            .iter()
            .filter_map(|(n, j)| {
                let m = serde_json::from_value(
                    j.get("match").cloned().unwrap_or(J::Object(Map::new())),
                )
                .ok()?;
                let p = serde_json::from_value(self.base(n).ok()?["pipeline"].clone()).ok()?;
                Some((n.clone(), m, p))
            })
            .collect()
    }

    /// Why `name` (a `model_type` or GGUF architecture) is refused: the reason of the first
    /// `unsupported` family whose `hf_model_type` / `gguf_arch` globs match it.
    pub fn unsupported(&self, name: &str) -> Option<String> {
        self.families.iter().find_map(|(f, j)| {
            let m: Match = serde_json::from_value(j.get("match")?.clone()).ok()?;
            let hit = m
                .hf_model_type
                .iter()
                .chain(&m.gguf_arch)
                .any(|p| glob(p, name));
            let spec = self.spec(f, &Signals::default()).ok()?;
            (hit && spec.pipeline == Pipeline::Unsupported).then(|| spec.unsupported_reason(name))
        })
    }

    /// Whether a `causal_lm` family lists `model_type` (glob) in its `hf_model_type`.
    pub fn runs_model_type(&self, model_type: &str) -> bool {
        self.matchers().iter().any(|(_, m, p)| {
            *p == Pipeline::CausalLm && m.hf_model_type.iter().any(|g| glob(g, model_type))
        })
    }

    /// The GGUF architectures `causal_lm` families claim.
    pub fn causal_lm_gguf_archs(&self) -> Vec<String> {
        self.matchers()
            .into_iter()
            .filter(|(_, _, p)| *p == Pipeline::CausalLm)
            .flat_map(|(_, m, _)| m.gguf_arch)
            .collect()
    }
}

fn applies(a: &AppliesTo, family: &str, sig: &Signals) -> bool {
    (a.hub_id.is_empty() || a.hub_id.iter().any(|g| glob(g, &sig.hub_id)))
        && a.family.as_deref().is_none_or(|f| f == family)
        && a.model_type
            .as_deref()
            .is_none_or(|t| sig.hf_model_type.as_deref() == Some(t))
        && a.gguf_arch
            .as_deref()
            .is_none_or(|t| sig.gguf_arch.as_deref() == Some(t))
        && (a.file_name.is_empty() || a.file_name.iter().any(|g| glob(g, &sig.file_name)))
}

/// The embedded specs plus the user's `<cache>/arch/{families,overrides}`, read once per process.
pub fn registry() -> Result<&'static Registry, String> {
    static REG: std::sync::LazyLock<Result<Registry, String>> = std::sync::LazyLock::new(|| {
        let user = crate::hardware::default_cache_dir()
            .ok()
            .map(|d| d.join("arch"));
        Registry::load(user.as_deref())
    });
    REG.as_ref().map_err(Clone::clone)
}

/// [`Registry::unsupported`] over [`registry`].
pub fn unsupported(name: &str) -> Option<String> {
    registry().ok()?.unsupported(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    const BASE: &str = r#"{family: "base", match: {hf_model_type: ["base"]}, pipeline: "causal_lm",
        config: {a: "1", b: "2"}, runtime: {prefill_rows: 64, list: [1, 2]}}"#;
    const CHILD: &str = r#"{family: "child", extends: "base", match: {hf_model_type: ["child"]},
        config: {b: "3", c: "4"}, runtime: {list: [9]}}"#;
    const FALLBACK: &str = r#"{family: "fb", match: {fallback: true}, pipeline: "causal_lm"}"#;

    #[test]
    fn extends_overrides_and_user_dir_merge_in_order() {
        let embedded = files(&[
            ("families/base.json5", BASE),
            ("families/child.json5", CHILD),
            ("families/fb.json5", FALLBACK),
            (
                "overrides/b.json5",
                r#"{applies_to: {family: "child"}, patch: {runtime: {prefill_rows: 32}, config: {c: "5"}}}"#,
            ),
            (
                "overrides/a.json5",
                r#"{applies_to: {family: "child"}, patch: {runtime: {prefill_rows: 16}}}"#,
            ),
            (
                "overrides/x.json5",
                r#"{applies_to: {hub_id: ["other/*"]}, patch: {config: {a: "99"}}}"#,
            ),
        ]);
        let user = files(&[(
            "overrides/u.json5",
            r#"{applies_to: {hub_id: ["org/*"]}, patch: {config: {a: null, d: "7"}}}"#,
        )]);
        let r = Registry::from_files(&embedded, &user).unwrap();
        let sig = Signals {
            hub_id: "org/m".into(),
            hf_model_type: Some("child".into()),
            ..Default::default()
        };
        assert_eq!(r.detect(&sig).as_deref(), Some("child"));
        let s = r.spec("child", &sig).unwrap();
        assert_eq!(s.pipeline, Pipeline::CausalLm, "inherited");
        assert_eq!(s.config.get("a"), None, "user override deletes");
        assert_eq!(s.config["b"], "3", "child wins over parent");
        assert_eq!(s.config["c"], "5", "override wins over family");
        assert_eq!(s.config["d"], "7");
        // Overrides apply in file-name order: a (16) then b (32).
        assert_eq!(s.runtime["prefill_rows"], 32);
        assert_eq!(s.runtime["list"], serde_json::json!([9]), "arrays replace");
        assert_eq!(s.matcher.hf_model_type, vec!["child".to_string()]);
    }

    #[test]
    fn user_family_replaces_embedded_and_fallback_is_last() {
        let embedded = files(&[
            ("families/base.json5", BASE),
            ("families/fb.json5", FALLBACK),
        ]);
        let user = files(&[(
            "families/base.json5",
            r#"{family: "base", match: {hf_model_type: ["base"]}, pipeline: "encoder"}"#,
        )]);
        let r = Registry::from_files(&embedded, &user).unwrap();
        let sig = Signals {
            hf_model_type: Some("base".into()),
            ..Default::default()
        };
        assert_eq!(r.spec("base", &sig).unwrap().pipeline, Pipeline::Encoder);
        let lm = Signals {
            hf_model_type: Some("unknown".into()),
            causal_lm: true,
            ..Default::default()
        };
        assert_eq!(r.detect(&lm).as_deref(), Some("fb"));
        let other = Signals {
            hf_model_type: Some("unknown".into()),
            ..Default::default()
        };
        assert_eq!(r.detect(&other), None);
    }

    #[test]
    fn globs() {
        assert!(glob("org/*-GGUF", "org/model-GGUF"));
        assert!(glob("*.gguf", "a.Q4_0.gguf"));
        assert!(!glob("org/?", "org/ab"));
    }

    // ---- LM family specs vs GraphLm / LnDecoder (fixtures as in `tests/golden/lm.rs`, `enc.rs`) ----

    use super::testing::{compare, Runner, Weights};
    use crate::hardware::Device;
    use serde_json::{json, Value as J};
    use std::path::Path;

    const V: usize = 256;
    const PROMPT: [u32; 6] = [5, 17, 42, 99, 133, 200];
    const DECODE: [u32; 3] = [7, 64, 250];

    struct Dims {
        h: usize,
        heads: usize,
        kv: usize,
        hd: usize,
        inter: usize,
        layers: usize,
    }

    fn lm_config(model_type: &str, arch: &str, d: &Dims, extra: J) -> J {
        let mut c = json!({
            "model_type": model_type, "architectures": [arch], "vocab_size": V,
            "hidden_size": d.h, "num_hidden_layers": d.layers, "num_attention_heads": d.heads,
            "num_key_value_heads": d.kv, "head_dim": d.hd, "intermediate_size": d.inter,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu",
            "max_position_embeddings": 128, "bos_token_id": 1, "eos_token_id": 2,
            "tie_word_embeddings": true,
        });
        for (k, v) in extra.as_object().unwrap() {
            c[k] = v.clone();
        }
        c
    }

    /// `tests/golden/lm.rs` `llama_weights` (HF llama names).
    fn llama_weights(seed: u64, d: &Dims, qk_norm: bool, shared_mlp: bool, bitnet: bool, untied: bool) -> Weights {
        let mut w = Weights::new(seed);
        w.s("model.embed_tokens.weight", &[V, d.h], 0.05);
        w.norm("model.norm.weight", d.h, 1.0);
        if untied {
            w.w("lm_head.weight", &[V, d.h]);
        }
        let (a, kvd) = (d.heads * d.hd, d.kv * d.hd);
        let proj = |w: &mut Weights, name: String, shape: [usize; 2]| {
            if !bitnet {
                w.w(name, &shape);
                return;
            }
            let s = 1.0 / (shape[1] as f32).sqrt();
            let v = (0..shape[0] * shape[1]).map(|_| s * ((w.rng.next_u64() % 3) as f32 - 1.0)).collect();
            w.put(name, &shape, v);
        };
        for l in 0..d.layers {
            let p = format!("model.layers.{l}");
            w.norm(format!("{p}.input_layernorm.weight"), d.h, 1.0);
            w.norm(format!("{p}.post_attention_layernorm.weight"), d.h, 1.0);
            for (n, shape) in [("q_proj", [a, d.h]), ("k_proj", [kvd, d.h]), ("v_proj", [kvd, d.h]), ("o_proj", [d.h, a])] {
                proj(&mut w, format!("{p}.self_attn.{n}.weight"), shape);
            }
            if qk_norm {
                w.norm(format!("{p}.self_attn.q_norm.weight"), d.hd, 1.0);
                w.norm(format!("{p}.self_attn.k_norm.weight"), d.hd, 1.0);
            }
            if shared_mlp {
                w.w(format!("{p}.shared_mlp.input_linear.weight"), &[2 * d.inter, d.h]);
                w.w(format!("{p}.shared_mlp.output_linear.weight"), &[d.h, d.inter]);
            } else {
                for (n, shape) in [("gate_proj", [d.inter, d.h]), ("up_proj", [d.inter, d.h]), ("down_proj", [d.h, d.inter])] {
                    proj(&mut w, format!("{p}.mlp.{n}.weight"), shape);
                }
            }
            if bitnet {
                w.norm(format!("{p}.self_attn.attn_sub_norm.weight"), a, 1.0);
                w.norm(format!("{p}.mlp.ffn_sub_norm.weight"), d.inter, 1.0);
            }
        }
        w
    }

    fn lfm2_fixture() -> (J, Weights) {
        let d = Dims { h: 32, heads: 4, kv: 2, hd: 8, inter: 64, layers: 3 };
        let (conv, k) = ([true, false, true], 3);
        let cfg = lm_config("lfm2", "Lfm2ForCausalLM", &d, json!({
            "norm_eps": 1e-5, "conv_L_cache": k, "conv_bias": false,
            "block_auto_adjust_ff_dim": false, "rope_theta": 1000000.0,
            "layer_types": ["conv", "full_attention", "conv"]}));
        let mut w = Weights::new(13);
        w.s("model.embed_tokens.weight", &[V, d.h], 0.05);
        w.norm("model.embedding_norm.weight", d.h, 1.0);
        for (l, &is_conv) in conv.iter().enumerate() {
            let p = format!("model.layers.{l}");
            w.norm(format!("{p}.operator_norm.weight"), d.h, 1.0);
            w.norm(format!("{p}.ffn_norm.weight"), d.h, 1.0);
            if is_conv {
                w.w(format!("{p}.conv.in_proj.weight"), &[3 * d.h, d.h]);
                w.w(format!("{p}.conv.conv.weight"), &[d.h, 1, k]);
                w.w(format!("{p}.conv.out_proj.weight"), &[d.h, d.h]);
            } else {
                let (a, kvd) = (d.heads * d.hd, d.kv * d.hd);
                w.w(format!("{p}.self_attn.q_proj.weight"), &[a, d.h]);
                w.w(format!("{p}.self_attn.k_proj.weight"), &[kvd, d.h]);
                w.w(format!("{p}.self_attn.v_proj.weight"), &[kvd, d.h]);
                w.w(format!("{p}.self_attn.out_proj.weight"), &[d.h, a]);
                w.norm(format!("{p}.self_attn.q_layernorm.weight"), d.hd, 1.0);
                w.norm(format!("{p}.self_attn.k_layernorm.weight"), d.hd, 1.0);
            }
            w.w(format!("{p}.feed_forward.w1.weight"), &[d.inter, d.h]);
            w.w(format!("{p}.feed_forward.w3.weight"), &[d.inter, d.h]);
            w.w(format!("{p}.feed_forward.w2.weight"), &[d.h, d.inter]);
        }
        (cfg, w)
    }

    /// GPT-NeoX: fused per-head query_key_value, partial RoPE (rotary_pct 0.25), parallel residual.
    fn neox_fixture() -> (J, Weights) {
        let (h, heads, inter, layers) = (32, 2, 64, 2);
        let cfg = json!({"model_type": "gpt_neox", "architectures": ["GPTNeoXForCausalLM"], "vocab_size": V,
            "hidden_size": h, "num_hidden_layers": layers, "num_attention_heads": heads, "intermediate_size": inter,
            "rotary_pct": 0.25, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "hidden_act": "gelu_new",
            "use_parallel_residual": true, "max_position_embeddings": 128, "tie_word_embeddings": false});
        let mut w = Weights::new(17);
        w.s("gpt_neox.embed_in.weight", &[V, h], 0.05);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        let lin = |w: &mut Weights, n: String, out: usize, inp: usize| {
            w.w(format!("{n}.weight"), &[out, inp]);
            w.s(format!("{n}.bias"), &[out], 0.1);
        };
        for l in 0..layers {
            let p = format!("gpt_neox.layers.{l}");
            ln(&mut w, format!("{p}.input_layernorm"));
            lin(&mut w, format!("{p}.attention.query_key_value"), 3 * h, h);
            lin(&mut w, format!("{p}.attention.dense"), h, h);
            ln(&mut w, format!("{p}.post_attention_layernorm"));
            lin(&mut w, format!("{p}.mlp.dense_h_to_4h"), inter, h);
            lin(&mut w, format!("{p}.mlp.dense_4h_to_h"), h, inter);
        }
        ln(&mut w, "gpt_neox.final_layer_norm".into());
        w.w("embed_out.weight", &[V, h]);
        (cfg, w)
    }

    /// GPT-Neo (`transformer.h.{l}.attn.attention.*`, Linear MLP, learned positions), which GraphLm
    /// loads. GraphLm scales scores by 1/sqrt(head_dim); `attention_multiplier: 1.0` makes it use
    /// GPT-Neo's unscaled scores, and the local layer's window (256) covers the 9 tokens.
    fn neo_fixture() -> (J, Weights) {
        let (h, inter) = (32, 64);
        let cfg = json!({"model_type": "gpt_neo", "architectures": ["GPTNeoForCausalLM"], "vocab_size": V,
            "hidden_size": h, "num_layers": 2, "num_heads": 2, "intermediate_size": inter,
            "max_position_embeddings": 32, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
            "attention_layers": ["global", "local"], "window_size": 256, "attention_multiplier": 1.0});
        let mut w = Weights::new(18);
        w.s("transformer.wte.weight", &[V, h], 0.05);
        w.s("transformer.wpe.weight", &[32, h], 0.3);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        for l in 0..2 {
            let p = format!("transformer.h.{l}");
            ln(&mut w, format!("{p}.ln_1"));
            for s in ["q_proj", "k_proj", "v_proj"] {
                w.w(format!("{p}.attn.attention.{s}.weight"), &[h, h]);
            }
            w.w(format!("{p}.attn.attention.out_proj.weight"), &[h, h]);
            w.s(format!("{p}.attn.attention.out_proj.bias"), &[h], 0.1);
            ln(&mut w, format!("{p}.ln_2"));
            w.w(format!("{p}.mlp.c_fc.weight"), &[inter, h]);
            w.s(format!("{p}.mlp.c_fc.bias"), &[inter], 0.1);
            w.w(format!("{p}.mlp.c_proj.weight"), &[h, inter]);
            w.s(format!("{p}.mlp.c_proj.bias"), &[h], 0.1);
        }
        ln(&mut w, "transformer.ln_f".into());
        (cfg, w)
    }

    const DEC_VOCAB: usize = 64;

    /// `tests/golden/enc.rs` `gpt2`: pre-LN, tanh GELU, Conv1D weights, `transformer.` prefix.
    fn gpt2_fixture() -> (J, Weights) {
        let (h, ffn, ctx) = (32, 64, 32);
        let mut w = Weights::new(0xEB);
        w.s("transformer.wte.weight", &[DEC_VOCAB, h], 0.5);
        w.s("transformer.wpe.weight", &[ctx, h], 0.3);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        let conv1d = |w: &mut Weights, n: String, inp: usize, out: usize| {
            w.s(format!("{n}.weight"), &[inp, out], 1.0 / (inp as f32).sqrt());
            w.s(format!("{n}.bias"), &[out], 0.1);
        };
        for l in 0..2 {
            let p = format!("transformer.h.{l}");
            ln(&mut w, format!("{p}.ln_1"));
            conv1d(&mut w, format!("{p}.attn.c_attn"), h, 3 * h);
            conv1d(&mut w, format!("{p}.attn.c_proj"), h, h);
            ln(&mut w, format!("{p}.ln_2"));
            conv1d(&mut w, format!("{p}.mlp.c_fc"), h, ffn);
            conv1d(&mut w, format!("{p}.mlp.c_proj"), ffn, h);
        }
        ln(&mut w, "transformer.ln_f".into());
        w.s("score.weight", &[3, h], 0.3);
        let cfg = json!({"architectures": ["GPT2ForSequenceClassification"], "model_type": "gpt2",
            "n_layer": 2, "n_head": 2, "n_embd": h, "n_positions": ctx, "n_ctx": ctx, "vocab_size": DEC_VOCAB,
            "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5, "pad_token_id": 0});
        (cfg, w)
    }

    /// `tests/golden/enc.rs` `opt`: project_in/out (embeddings 16, hidden 32), positions offset 2.
    fn opt_fixture(pre_ln: bool, act: &str, seed: u64) -> (J, Weights) {
        let (h, e, ffn, ctx) = (32, 16, 64, 32);
        let p = "model.decoder";
        let mut w = Weights::new(seed);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        let lin = |w: &mut Weights, n: String, out: usize, inp: usize| {
            w.w(format!("{n}.weight"), &[out, inp]);
            w.s(format!("{n}.bias"), &[out], 0.1);
        };
        w.s(format!("{p}.embed_tokens.weight"), &[DEC_VOCAB, e], 0.5);
        w.s(format!("{p}.embed_positions.weight"), &[ctx + 2, h], 0.3);
        w.w(format!("{p}.project_in.weight"), &[h, e]);
        w.w(format!("{p}.project_out.weight"), &[e, h]);
        for l in 0..2 {
            let b = format!("{p}.layers.{l}");
            ln(&mut w, format!("{b}.self_attn_layer_norm"));
            for s in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                lin(&mut w, format!("{b}.self_attn.{s}"), h, h);
            }
            ln(&mut w, format!("{b}.final_layer_norm"));
            lin(&mut w, format!("{b}.fc1"), ffn, h);
            lin(&mut w, format!("{b}.fc2"), h, ffn);
        }
        if pre_ln {
            ln(&mut w, format!("{p}.final_layer_norm"));
        }
        w.s("score.weight", &[3, e], 1.0);
        let cfg = json!({"architectures": ["OPTForSequenceClassification"], "model_type": "opt",
            "hidden_size": h, "ffn_dim": ffn, "num_hidden_layers": 2, "num_attention_heads": 2,
            "word_embed_proj_dim": e, "max_position_embeddings": ctx, "vocab_size": DEC_VOCAB,
            "do_layer_norm_before": pre_ln, "activation_function": act, "enable_bias": true,
            "layer_norm_elementwise_affine": true, "pad_token_id": 0});
        (cfg, w)
    }

    fn devices() -> Vec<Device> {
        let mut d = vec![Device::Cpu];
        if crate::hardware::available_devices().contains(&Device::Metal) {
            d.push(Device::Metal);
        }
        d
    }

    fn close(bad: &mut Vec<String>, what: &str, got: &[f32], want: &[f32]) {
        let (rel, cos) = compare(got, want);
        eprintln!("{what}: rel {rel:.3e} cos {cos:.6}");
        if !(rel < 1e-3 && cos > 0.9999) {
            bad.push(format!("{what}: rel {rel:.3e} cos {cos:.6}"));
        }
    }

    /// The spec's `out` after a 6-row prefill of `prompt` (head `head`), then after each of 3
    /// one-row decode steps with the KV state carried, and the runner.
    fn spec_steps(dir: &Path, family: &str, dev: Device, prompt: &[u32], decode: &[u32], head: &str, out: &str) -> (Vec<Vec<f32>>, Runner) {
        let mut r = Runner::load(dir, None, dev, 64).unwrap_or_else(|e| panic!("{family}: {e}"));
        assert_eq!(r.model.spec.family, family, "detected family");
        let mut o = vec![r.lm(prompt, 0, head, out, vec![])];
        for (i, &t) in decode.iter().enumerate() {
            o.push(r.lm(&[t], prompt.len() + i, head, out, vec![]));
        }
        (o, r)
    }

    /// Every checkpoint tensor is read by some instance `r` built.
    fn all_read(r: &Runner, case: &str) {
        assert_eq!(r.unconsumed(), Vec::<String>::new(), "{case}: unconsumed checkpoint tensors");
    }

    /// Each GraphLm fixture family against GraphLm (`hf_loader::load_causal_lm` +
    /// `ExecutableModel::load_graph_lm`) on the same device; GPT-2 / OPT (which GraphLm does not
    /// load) against their `LnDecoder` host forward; BitNet against `tests/golden/lm_bitnet.json`
    /// (GraphLm's BitNet graph is wrong). Prefill 6 rows, then 3 cached decode steps.
    #[test]
    fn llama_family_matches_graph_lm() {
        use crate::hardware::QuantMode;
        let tmp = tempfile::tempdir().unwrap();
        let mut bad = Vec::new();
        let seq: Vec<u32> = PROMPT.iter().chain(&DECODE).copied().collect();
        let d = |h, heads, kv, hd, inter, layers| Dims { h, heads, kv, hd, inter, layers };
        let smol = d(64, 4, 2, 16, 128, 2);
        let granite_cfg = lm_config("granitemoehybrid", "GraniteMoeHybridForCausalLM", &smol, json!({
            "attention_multiplier": 0.1, "embedding_multiplier": 3.0, "residual_multiplier": 0.5,
            "logits_scaling": 4.0, "shared_intermediate_size": 128, "num_local_experts": 0,
            "layer_types": ["attention", "attention"], "position_embedding_type": "rope",
            "tie_word_embeddings": false}));
        let (qk, wide) = (d(32, 2, 2, 16, 64, 2), d(32, 4, 2, 16, 64, 2));
        let graph_lm: Vec<(&str, &str, J, Weights)> = vec![
            ("smollm", "llama", lm_config("llama", "LlamaForCausalLM", &smol, json!({})),
             llama_weights(11, &smol, false, false, false, false)),
            ("granite", "granite", granite_cfg, llama_weights(12, &smol, false, true, false, true)),
            ("lfm2", "lfm2", lfm2_fixture().0, lfm2_fixture().1),
            ("qk_norm", "llama", lm_config("llama", "LlamaForCausalLM", &qk, json!({})),
             llama_weights(14, &qk, true, false, false, false)),
            ("gqa_wide", "llama", lm_config("llama", "LlamaForCausalLM", &wide, json!({})),
             llama_weights(15, &wide, false, false, false, false)),
            ("gpt_neox", "gpt_neox", neox_fixture().0, neox_fixture().1),
            ("gpt_neo", "gpt2", neo_fixture().0, neo_fixture().1),
        ];
        for (case, family, cfg, w) in &graph_lm {
            let dir = tmp.path().join(case);
            w.write_checkpoint(&dir, cfg, V);
            for dev in devices() {
                let llm = crate::model::hf_loader::load_causal_lm(&super::testing::artifacts(&dir), QuantMode::Uncompressed)
                    .unwrap_or_else(|e| panic!("{case}: GraphLm: {e}"));
                let m = crate::scheduler::ExecutableModel::load_graph_lm(llm, dev).unwrap();
                let (got, r) = spec_steps(&dir, family, dev, &PROMPT, &DECODE, "last", "logits");
                all_read(&r, case);
                for (i, g) in got.iter().enumerate() {
                    let want = m.prompt_logits(seq[..PROMPT.len() + i].to_vec()).unwrap();
                    close(&mut bad, &format!("{case} {dev:?} step {i}"), g, &want);
                }
            }
        }

        // GPT-2 / OPT: last-row final hidden (after ln_f / project_out) and the score head.
        use crate::model::classifier::DecoderBackbone;
        let dec_seq: Vec<u32> = seq.iter().map(|t| t % DEC_VOCAB as u32).collect();
        let (dp, dd) = dec_seq.split_at(PROMPT.len());
        let decoders: Vec<(&str, &str, (J, Weights))> = vec![
            ("gpt2", "gpt2", gpt2_fixture()),
            ("opt_post_ln", "opt", opt_fixture(false, "relu", 0xEC)),
            ("opt_pre_ln", "opt", opt_fixture(true, "gelu", 0xED)),
        ];
        for (case, family, (cfg, w)) in &decoders {
            let root = tmp.path().join(case);
            let dir = root.join(format!("models--gibbs-test--{case}/snapshots/main"));
            w.write_checkpoint(&dir, cfg, DEC_VOCAB);
            let hub = crate::model::hub::HuggingFaceHub::with_cache_dir(root.clone());
            let host = crate::model::classifier::ln_decoder::LnDecoder::load(&hub, &format!("gibbs-test/{case}"), cfg)
                .unwrap_or_else(|e| panic!("{case}: LnDecoder: {e}"));
            let score = &w.tensors.iter().find(|t| t.0 == "score.weight").unwrap().2;
            for dev in devices() {
                let (got, mut r) = spec_steps(&dir, family, dev, dp, dd, "hidden", "hidden");
                for (i, g) in got.iter().enumerate() {
                    let want = host.last_hidden(&dec_seq[..PROMPT.len() + i]).unwrap();
                    close(&mut bad, &format!("{case} {dev:?} step {i}"), &g[g.len() - want.len()..], &want);
                }
                // Seq-cls: row `last_index` of the 6-row prompt (rewrites the same KV rows).
                let last = crate::graph::Tensor::U32(vec![PROMPT.len() as u32 - 1]);
                let logits = r.lm(dp, 0, "score", "logits", vec![last]);
                let h = host.last_hidden(dp).unwrap();
                let want: Vec<f32> = score.chunks(h.len()).map(|row| row.iter().zip(&h).map(|(a, b)| a * b).sum()).collect();
                close(&mut bad, &format!("{case} {dev:?} score"), &logits, &want);
                all_read(&r, case);
            }
        }

        // BitNet: online-ternarized masters, int8 activations, sub-norms, relu².
        let bit = d(64, 4, 2, 16, 128, 2);
        let cfg = lm_config("bitnet", "BitNetForCausalLM", &bit, json!({"hidden_act": "relu2",
            "quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}}));
        let dir = tmp.path().join("bitnet");
        llama_weights(16, &bit, false, false, true, false).write_checkpoint(&dir, &cfg, V);
        let golden: J = serde_json::from_slice(
            &std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/lm_bitnet.json")).unwrap(),
        )
        .unwrap();
        for dev in devices() {
            let (got, r) = spec_steps(&dir, "bitnet", dev, &PROMPT, &DECODE, "last", "logits");
            all_read(&r, "bitnet");
            for (i, g) in got.iter().enumerate() {
                let key = if i == 0 { "prefill6".to_string() } else { format!("decode{}", i - 1) };
                let want: Vec<f32> = golden["f32"][&key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
                close(&mut bad, &format!("bitnet {dev:?} {key}"), g, &want);
            }
        }
        assert!(bad.is_empty(), "mismatches:\n  {}", bad.join("\n  "));
    }

    /// A synthetic `gibbs/*` id (hub.rs writes a `LlamaForCausalLM` config, no weights) random-inits
    /// every weight the llama text instances read.
    #[test]
    fn llama_synthetic_id_runs() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = json!({"architectures": ["LlamaForCausalLM"], "model_type": "qwen2", "hidden_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 256,
            "vocab_size": 512, "max_position_embeddings": 2048, "rms_norm_eps": 1e-6});
        let p = tmp.path().join("config.json");
        std::fs::write(&p, cfg.to_string()).unwrap();
        let a = crate::model::hub::ModelArtifacts {
            model_id: "gibbs/tiny-1m".into(),
            local_dir: Some(tmp.path().to_path_buf()),
            config_path: Some(p),
            ..Default::default()
        };
        let mut r = super::testing::Runner::from_artifacts(&a, None, Device::Cpu, 16).unwrap();
        assert_eq!(r.model.spec.family, "llama");
        for (ids, pos) in [(&[1u32, 2, 3][..], 0), (&[4][..], 3)] {
            let l = r.lm(ids, pos, "last", "logits", vec![]);
            assert_eq!(l.len(), 512);
            assert!(l.iter().all(|x| x.is_finite()) && l.iter().any(|&x| x != 0.0));
        }
    }
}
