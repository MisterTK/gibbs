//! Test harness: tiny on-disk checkpoints, and a [`Runner`] that loads one through
//! [`load::load_with`], builds component instances, compiles one `GraphProgram` per instance
//! shape on a device and runs them over one [`StateStore`] (prefill, then decode steps).

use super::build::{build, Instance};
use super::expr::Val;
use super::load::{self, Model};
use super::Registry;
use crate::graph::program::GraphProgram;
use crate::graph::{StateStore, Tensor};
use crate::hardware::{Device, PlacementMode};
use crate::model::hub::ModelArtifacts;
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use serde_json::{json, Map, Value as J};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// SplitMix64 (the sequence `tests/golden/util.rs` uses, so fixtures match its goldens).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Named f32 tensors with deterministic values (`tests/golden/util.rs` `Weights`).
pub struct Weights {
    pub rng: Rng,
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl Weights {
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng(seed), tensors: Vec::new() }
    }
    /// Uniform in [-scale, scale).
    pub fn s(&mut self, name: impl Into<String>, shape: &[usize], scale: f32) -> &mut Self {
        let v = (0..shape.iter().product()).map(|_| self.rng.unit() * scale).collect();
        self.put(name, shape, v)
    }
    /// Scale 1/sqrt(last dim) (0.5 for rank 1).
    pub fn w(&mut self, name: impl Into<String>, shape: &[usize]) -> &mut Self {
        let scale = match shape {
            [.., last] if shape.len() >= 2 => 1.0 / (*last as f32).sqrt(),
            _ => 0.5,
        };
        self.s(name, shape, scale)
    }
    /// Norm gain: center ± 0.1.
    pub fn norm(&mut self, name: impl Into<String>, n: usize, center: f32) -> &mut Self {
        let v = (0..n).map(|_| center + self.rng.unit() * 0.1).collect();
        self.put(name, &[n], v)
    }
    pub fn put(&mut self, name: impl Into<String>, shape: &[usize], v: Vec<f32>) -> &mut Self {
        assert_eq!(v.len(), shape.iter().product::<usize>());
        self.tensors.push((name.into(), shape.to_vec(), v));
        self
    }

    /// `dir/model.safetensors` (F32), `dir/config.json` and a WordLevel `dir/tokenizer.json` over
    /// `vocab` ids.
    pub fn write_checkpoint(&self, dir: &Path, config: &J, vocab: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let (mut header, mut data) = (Map::new(), Vec::<u8>::new());
        for (name, shape, v) in &self.tensors {
            let start = data.len();
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            header.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let mut h = serde_json::to_vec(&J::Object(header)).unwrap();
        h.resize(h.len().next_multiple_of(8), b' ');
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let vocab_map: Map<String, J> = (0..vocab).map(|i| (format!("t{i}"), json!(i))).collect();
        let tok = json!({"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab_map, "unk_token": "t0"}});
        std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    }
}

/// `ModelArtifacts` of a checkpoint directory written by [`Weights::write_checkpoint`].
pub fn artifacts(dir: &Path) -> ModelArtifacts {
    ModelArtifacts {
        model_id: "test/arch".into(),
        local_dir: Some(dir.to_path_buf()),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        is_cached: true,
        ..Default::default()
    }
}

/// A loaded model run instance by instance on one device, all instances sharing one state.
pub struct Runner {
    pub model: Model,
    dev: Device,
    progs: HashMap<String, GraphProgram>,
    state: StateStore,
    used: HashSet<String>,
}

impl Runner {
    /// Loads `dir` (the embedded registry, `family` or detected; bit-exact weights) and checks the
    /// spec's refusals; KV states hold `ctx` positions.
    pub fn load(dir: &Path, family: Option<&str>, dev: Device, ctx: usize) -> Result<Self, String> {
        Self::from_artifacts(&artifacts(dir), family, dev, ctx)
    }

    /// [`Self::load`] of any artifacts (synthetic ids included).
    pub fn from_artifacts(a: &ModelArtifacts, family: Option<&str>, dev: Device, ctx: usize) -> Result<Self, String> {
        let reencode = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            cache_dir: None,
            ..Default::default()
        };
        let model = load::load_with(a, &Registry::load(None)?, family, reencode)?;
        load::check(&model)?;
        model.env.set_ctx(json!(ctx));
        Ok(Self {
            model,
            dev,
            progs: HashMap::new(),
            state: StateStore::new(),
            used: HashSet::new(),
        })
    }

    /// Builds `component` with `vars`.
    pub fn instance(&mut self, component: &str, vars: &[(&str, Val)]) -> Result<Instance, String> {
        let m = &self.model;
        let inst = build(&m.spec, component, &m.env, &m.store, vars)?;
        self.used.extend(inst.weights.iter().cloned());
        Ok(inst)
    }

    /// Runs `component` with `vars`: `inputs` are the caller-supplied inputs in declaration order;
    /// generated inputs are filled here. Returns the outputs by name.
    pub fn run(
        &mut self,
        component: &str,
        vars: &[(&str, Val)],
        inputs: Vec<Tensor>,
    ) -> Result<HashMap<String, Tensor>, String> {
        let key = format!("{component}:{}", super::build::ShapeCache::key(vars));
        let inst = self.instance(component, vars)?;
        let mut given = inputs.into_iter();
        let mut ins = Vec::new();
        for p in &inst.inputs {
            ins.push(match &p.gen {
                Some((g, args)) => match super::gen::generate(g, args)?.data {
                    super::gen::GenData::F32(v) => Tensor::F32(v),
                    super::gen::GenData::U32(v) => Tensor::U32(v),
                },
                None => given
                    .next()
                    .ok_or_else(|| format!("input `{}` not given", p.name))?,
            });
        }
        if !self.progs.contains_key(&key) {
            let w = self.model.store.materialize(&inst.weights)?;
            let p = GraphProgram::compile(inst.graph.clone(), w, PlacementMode::Pin(self.dev))?;
            if p.primary_device() != self.dev || (self.dev == Device::Metal && p.cpu_node_count() > 0) {
                return Err(format!(
                    "compiled for {:?} with {} CPU nodes, wanted all on {:?}",
                    p.primary_device(),
                    p.cpu_node_count(),
                    self.dev
                ));
            }
            self.progs.insert(key.clone(), p);
        }
        let p = self.progs.get_mut(&key).unwrap();
        std::mem::swap(&mut p.state, &mut self.state);
        let out = p.run(ins);
        std::mem::swap(&mut p.state, &mut self.state);
        Ok(inst.outputs.iter().cloned().zip(out?).collect())
    }

    /// The `text` component on `ids` at positions `pos0..`: output `out` (f32) of instance
    /// `head`. Extra caller inputs (after token_ids, positions) in `extra`.
    pub fn lm(&mut self, ids: &[u32], pos0: usize, head: &str, out: &str, extra: Vec<Tensor>) -> Vec<f32> {
        let vars = [("rows", Val::Num(ids.len() as f64)), ("head", Val::Str(head.into()))];
        let positions = (pos0..pos0 + ids.len()).map(|p| p as u32).collect();
        let mut ins = vec![Tensor::U32(ids.to_vec()), Tensor::U32(positions)];
        ins.extend(extra);
        let mut o = self.run("text", &vars, ins).unwrap_or_else(|e| panic!("{e}"));
        match o.remove(out) {
            Some(Tensor::F32(v)) => v,
            other => panic!("output `{out}`: {other:?}"),
        }
    }

    /// Checkpoint tensors no instance built so far reads (and `ignore` does not list).
    pub fn unconsumed(&self) -> Vec<String> {
        self.model.store.unconsumed(&self.used, &self.model.ignore)
    }
}

/// (relative L2 error, cosine) of `got` against `want`.
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "lengths");
    let (mut d2, mut w2, mut g2, mut dot) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        d2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
    }
    ((d2 / w2.max(1e-30)).sqrt(), dot / (w2.sqrt() * g2.sqrt()).max(1e-30))
}
