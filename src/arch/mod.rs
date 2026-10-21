//! Model architectures as data. Each family is a JSON5 spec under `arch/families/` (embedded at
//! build time, see [`EMBEDDED`]); per-checkpoint patches live under `arch/overrides/`, and users
//! add or replace both under `<cache>/arch/{families,overrides}/`. [`load`] turns a matched spec
//! and a checkpoint into weights, [`build`] turns a spec component into a `graph::Graph`.
//!
//! Merge order: family <- its `extends` chain (parent first) <- matching embedded overrides (file
//! name order) <- user family file of the same name (replaces) <- matching user overrides. Merging is
//! RFC 7386 merge patch: objects merge, everything else (arrays included) replaces, `null` deletes.

pub mod expr;
pub mod gen;
pub mod load;
pub mod names;

use serde::Deserialize;
use serde_json::{Map, Value as J};
use std::path::Path;

/// Every embedded spec file: (path under `arch/`, contents).
pub const EMBEDDED: &[(&str, &str)] = &[];

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
    /// Names, or `{name, when}` objects.
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
        serde_json::from_value(j).map_err(|e| format!("family `{family}`: {e}"))
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
        let hit = |m: &Match| {
            sig.hf_model_type
                .as_ref()
                .is_some_and(|t| m.hf_model_type.contains(t))
                || sig
                    .hf_architectures
                    .iter()
                    .any(|a| m.hf_architectures.contains(a))
                || sig
                    .gguf_arch
                    .as_ref()
                    .is_some_and(|a| m.gguf_arch.contains(a))
                || sig
                    .diffusers_class
                    .as_ref()
                    .is_some_and(|c| m.diffusers_class.contains(c))
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
}
