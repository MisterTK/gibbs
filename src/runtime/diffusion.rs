//! The diffusion pipeline. [`Net`] runs one network of a loaded family spec at any instance shape
//! (one compiled program per network; other shapes run reshaped on its weights).

use crate::arch::build::{build, Instance, ShapeCache};
use crate::arch::expr::{eval_str, walk, Env, Val};
use crate::arch::gen::{generate, GenData};
use crate::arch::load::Model;
use crate::graph::program::GraphProgram;
use crate::graph::Tensor;
use crate::hardware::PlacementMode;
use crate::model::hub::ModelArtifacts;
use parking_lot::Mutex;
use serde_json::Value as J;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

/// Artifacts of a component directory (its `config.json` + weights).
pub fn dir_artifacts(dir: &Path) -> ModelArtifacts {
    let file = |f: &str| Some(dir.join(f)).filter(|p| p.is_file());
    ModelArtifacts {
        model_id: dir.display().to_string(),
        local_dir: Some(dir.to_path_buf()),
        config_path: file("config.json"),
        tokenizer_path: file("tokenizer.json"),
        weights_path: crate::model::hub::component_weights(dir).ok(),
        is_cached: true,
        ..Default::default()
    }
}

/// Step values a `gen` input's `"step:<expr>"` args read.
struct StepEnv<'a>(&'a [(&'a str, f64)]);

impl Env for StepEnv<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        let v = self.0.iter().find(|(k, _)| *k == root)?.1;
        walk(Val::Num(v), path)
    }
}

/// `args` with every `"step:<expr>"` string evaluated over `step`.
fn step_args(args: &J, step: &[(&str, f64)]) -> Result<J, String> {
    Ok(match args {
        J::String(s) => match s.strip_prefix("step:") {
            Some(e) => eval_str(e, &StepEnv(step))
                .map_err(|e| format!("`{s}`: {e}"))?
                .to_json(),
            None => args.clone(),
        },
        J::Array(a) => J::Array(a.iter().map(|x| step_args(x, step)).collect::<Result<_, _>>()?),
        J::Object(o) => J::Object(
            o.iter()
                .map(|(k, v)| Ok((k.clone(), step_args(v, step)?)))
                .collect::<Result<_, String>>()?,
        ),
        other => other.clone(),
    })
}

fn gen_tensor(name: &str, args: &J) -> Result<Tensor, String> {
    Ok(match generate(name, args)?.data {
        GenData::F32(v) => Tensor::F32(v),
        GenData::U32(v) => Tensor::U32(v),
    })
}

fn tensor_len(t: &Tensor) -> usize {
    match t {
        Tensor::F32(v) => v.len(),
        Tensor::U32(v) => v.len(),
        Tensor::F16(v) | Tensor::BF16(v) => v.len(),
        _ => 0,
    }
}

/// One network of a loaded model, run at any instance shape.
pub struct Net {
    pub model: Arc<Model>,
    pub net: String,
    mode: PlacementMode,
    shapes: ShapeCache,
    /// Compiled from the first instance run; every other instance runs reshaped on it.
    program: Mutex<Option<(String, GraphProgram)>>,
    used: Mutex<HashSet<String>>,
}

impl Net {
    pub fn new(model: Arc<Model>, net: &str, mode: PlacementMode) -> Self {
        Self {
            model,
            net: net.to_string(),
            mode,
            shapes: ShapeCache::new(net),
            program: Mutex::new(None),
            used: Mutex::new(HashSet::new()),
        }
    }

    /// Loads component directory `dir` as `family` and runs its network `net` on `mode`.
    pub fn open(dir: &Path, family: &str, net: &str, mode: PlacementMode) -> Result<Self, String> {
        let m = crate::arch::load::load(&dir_artifacts(dir), crate::arch::registry()?, Some(family))?;
        crate::arch::load::check(&m)?;
        Ok(Self::new(Arc::new(m), net, mode))
    }

    /// The instance of `vars` (built once per var tuple).
    pub fn instance(&self, vars: &[(&str, Val)]) -> Result<Arc<Instance>, String> {
        let m = &self.model;
        let inst = self
            .shapes
            .get(vars, || build(&m.spec, &self.net, &m.env, &m.store, vars))?;
        self.used.lock().extend(inst.weights.iter().cloned());
        Ok(inst)
    }

    /// Runs the instance of `vars`: inputs without a generator come from `given` by name, `gen`
    /// inputs are generated (their `"step:<expr>"` args evaluated over `step`). Outputs by name, in
    /// graph output order.
    pub fn run(
        &self,
        vars: &[(&str, Val)],
        given: &[(&str, Tensor)],
        step: &[(&str, f64)],
    ) -> Result<Vec<(String, Tensor)>, String> {
        let inst = self.instance(vars)?;
        let mut ins = Vec::with_capacity(inst.inputs.len());
        for p in &inst.inputs {
            let t = match &p.gen {
                Some((g, args)) => gen_tensor(g, &step_args(args, step)?),
                None => given
                    .iter()
                    .find(|(n, _)| *n == p.name)
                    .map(|(_, t)| t.clone())
                    .ok_or_else(|| format!("{}: input `{}` not given", self.net, p.name)),
            }
            .map_err(|e| format!("{}: input `{}`: {e}", self.net, p.name))?;
            let want: usize = p.shape.iter().product();
            if tensor_len(&t) != want {
                return Err(format!(
                    "{}: input `{}` has {} values, shape {:?} wants {want}",
                    self.net,
                    p.name,
                    tensor_len(&t),
                    p.shape
                ));
            }
            ins.push(t);
        }
        let key = ShapeCache::key(vars);
        let mut slot = self.program.lock();
        let out = match slot.as_mut() {
            Some((k, p)) if *k == key => p.run(ins)?,
            Some((_, p)) => p.run_reshaped(&inst.graph, ins)?,
            None => {
                let w = self.model.store.materialize(&inst.weights)?;
                let mut p = GraphProgram::compile_reshapable(inst.graph.clone(), w, self.mode)?;
                let out = p.run(ins)?;
                *slot = Some((key, p));
                out
            }
        };
        Ok(inst.outputs.iter().cloned().zip(out).collect())
    }

    /// Output `name` (f32) of [`Self::run`].
    pub fn run_one(
        &self,
        vars: &[(&str, Val)],
        given: &[(&str, Tensor)],
        step: &[(&str, f64)],
        name: Option<&str>,
    ) -> Result<Vec<f32>, String> {
        let mut out = self.run(vars, given, step)?;
        let i = match name {
            None => 0,
            Some(n) => out
                .iter()
                .position(|(o, _)| o == n)
                .ok_or_else(|| format!("{}: no output `{n}`", self.net))?,
        };
        match out.swap_remove(i).1 {
            Tensor::F32(v) => Ok(v),
            t => Ok(t.as_f32()?.to_vec()),
        }
    }

    /// Weights every instance built so far reads.
    pub fn used(&self) -> HashSet<String> {
        self.used.lock().clone()
    }
}

/// Checkpoint tensors of `model` that none of `nets` (built at least once) reads and its spec
/// does not `ignore`, plus `extra` names read outside any graph.
pub fn unconsumed(model: &Model, nets: &[&Net], extra: &[&str]) -> Vec<String> {
    let mut used: HashSet<String> = nets.iter().flat_map(|n| n.used()).collect();
    used.extend(extra.iter().map(|s| s.to_string()));
    model.store.unconsumed(&used, &model.spec.ignore)
}
