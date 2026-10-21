//! Encoder pipeline: a family spec's `encoder` component run once per (unpadded) sequence. The
//! first instance (one row) compiles one `GraphProgram`; every sequence length runs through
//! `GraphProgram::run_reshaped` on its weights. One `head` per encoder (`classify` logits, a pooled
//! `embedding`, or the `hidden` states), fixed at load.
//!
//! Inputs the runtime supplies by name: `token_ids`, `positions` (per the spec's
//! `runtime.position_ids` expression: `absolute` 0.., or `pad_offset` from the config var
//! `pad_id` + 1), `token_types` (segment ids) and `attention_mask` (all ones); any other input is
//! a spec generator.

use crate::arch::build::{build, InputPlan, Instance, ShapeCache};
use crate::arch::expr::Val;
use crate::arch::gen::{generate, GenData};
use crate::arch::load::{self, Model};
use crate::graph::program::GraphProgram;
use crate::graph::Tensor;
use crate::hardware::{Device, PlacementMode};
use crate::model::hub::ModelArtifacts;
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use parking_lot::Mutex;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::Arc;

const COMPONENT: &str = "encoder";

#[derive(Debug, Default, Deserialize)]
struct Runtime {
    /// Expression evaluating to `absolute` | `pad_offset`.
    #[serde(default)]
    position_ids: Option<String>,
}

/// A loaded encoder spec compiled on one device.
pub struct SpecEncoder {
    pub model: Model,
    head: String,
    /// `Some(pad)`: positions count from `pad + 1`; pad tokens sit at `pad`.
    pad_offset: Option<u32>,
    shapes: ShapeCache,
    program: Mutex<GraphProgram>,
    device: Device,
    params: usize,
}

impl SpecEncoder {
    /// Loads `artifacts` as `family` (detected when `None`) with its weights as stored.
    pub fn load(
        artifacts: &ModelArtifacts,
        family: Option<&str>,
        device: Device,
        head: &str,
    ) -> Result<Self, String> {
        let reencode = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            cache_dir: None,
            ..Default::default()
        };
        let model = load::load_with(artifacts, crate::arch::registry()?, family, reencode)?;
        Self::new(model, device, head)
    }

    /// Compiles `model`'s `head` instance on `device`; refuses checkpoint tensors it never reads.
    pub fn new(model: Model, device: Device, head: &str) -> Result<Self, String> {
        load::check(&model)?;
        let fam = model.spec.family.clone();
        let rt: Runtime = model.spec.runtime_as()?;
        let pad_offset = match &rt.position_ids {
            None => None,
            Some(e) => match model.env.eval(&e.as_str().into())? {
                Val::Str(s) if s == "absolute" => None,
                Val::Str(s) if s == "pad_offset" => Some(model.env.var("pad_id")?.usize()? as u32),
                v => return Err(format!("{fam}: runtime.position_ids is {v}, not absolute|pad_offset")),
            },
        };
        let shapes = ShapeCache::new(COMPONENT);
        let v = vars(1, head);
        let first = shapes.get(&v, || build(&model.spec, COMPONENT, &model.env, &model.store, &v))?;
        let used: HashSet<String> = first.weights.iter().cloned().collect();
        let unused = model.store.unconsumed(&used, &model.ignore);
        if !unused.is_empty() {
            return Err(format!(
                "{fam}: checkpoint tensors the `{head}` encoder never reads (list deliberate ones in `ignore`): {}",
                unused.join(", ")
            ));
        }
        let weights = model.store.materialize(&first.weights)?;
        let params = weights.parameter_count();
        let program = GraphProgram::compile_reshapable(first.graph.clone(), weights, PlacementMode::Pin(device))?;
        if program.primary_device() != device || (device != Device::Cpu && program.cpu_node_count() > 0) {
            return Err(format!(
                "gpu-only: {fam} compiled for {} with {} CPU nodes, wanted every node on {device}",
                program.primary_device(),
                program.cpu_node_count()
            ));
        }
        Ok(Self {
            model,
            head: head.to_string(),
            pad_offset,
            shapes,
            program: Mutex::new(program),
            device,
            params,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Weight count of the compiled instance.
    pub fn parameter_count(&self) -> usize {
        self.params
    }

    /// A config var as a number (`hidden`, `layers`, `heads`, `max_position_embeddings`...).
    pub fn var(&self, name: &str) -> Result<usize, String> {
        self.model.env.var(name)?.usize()
    }

    /// The most tokens a sequence may hold (RoBERTa-style positions lose `pad + 1` table rows).
    pub fn capacity(&self) -> Result<usize, String> {
        let max = self.var("max_position_embeddings")?;
        Ok(max - self.pad_offset.map_or(0, |p| p as usize + 1))
    }

    /// The encoder head's output for one sequence (`types`: segment ids, one per id).
    pub fn run(&self, ids: &[u32], types: &[u32]) -> Result<Vec<f32>, String> {
        let n = ids.len();
        if n == 0 {
            return Err("tokenizer produced no tokens".into());
        }
        let cap = self.capacity()?;
        if n > cap {
            return Err(format!("{n} tokens exceed {cap} positions"));
        }
        let inst = self.instance(n)?;
        let inputs = inst
            .inputs
            .iter()
            .map(|d| self.input(d, ids, types))
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = self.program.lock().run_reshaped(&inst.graph, inputs)?;
        if out.is_empty() {
            return Err("encoder graph produced no output".into());
        }
        out.swap_remove(0).into_f32()
    }

    fn instance(&self, rows: usize) -> Result<Arc<Instance>, String> {
        let v = vars(rows, &self.head);
        let m = &self.model;
        self.shapes
            .get(&v, || build(&m.spec, COMPONENT, &m.env, &m.store, &v))
    }

    fn input(&self, d: &InputPlan, ids: &[u32], types: &[u32]) -> Result<Tensor, String> {
        let n = ids.len();
        Ok(match (&d.gen, d.name.as_str()) {
            (Some((g, args)), _) => match generate(g, args)
                .map_err(|e| format!("input `{}`: {e}", d.name))?
                .data
            {
                GenData::F32(v) => Tensor::F32(v),
                GenData::U32(v) => Tensor::U32(v),
            },
            (None, "token_ids") => Tensor::U32(ids.to_vec()),
            (None, "positions") => Tensor::U32(positions(ids, self.pad_offset)),
            (None, "token_types") if types.len() == n => Tensor::U32(types.to_vec()),
            (None, "token_types") => Tensor::U32(vec![0; n]),
            (None, "attention_mask") => Tensor::U32(vec![1; n]),
            (None, other) => {
                return Err(format!(
                    "input `{other}` has no generator and is none the encoder runtime supplies (token_ids, positions, token_types, attention_mask)"
                ))
            }
        })
    }
}

fn vars(rows: usize, head: &str) -> Vec<(&'static str, Val)> {
    vec![
        ("rows", Val::Num(rows as f64)),
        ("head", Val::Str(head.to_string())),
    ]
}

/// Position ids: 0.. , or (`pad`) non-pad tokens from `pad + 1` with pad tokens at `pad`.
fn positions(ids: &[u32], pad: Option<u32>) -> Vec<u32> {
    match pad {
        None => (0..ids.len() as u32).collect(),
        Some(pad) => {
            let mut seen = 0;
            ids.iter()
                .map(|&t| {
                    if t == pad {
                        pad
                    } else {
                        seen += 1;
                        pad + seen
                    }
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_offset_positions_skip_pads() {
        assert_eq!(positions(&[5, 6, 7], None), [0, 1, 2]);
        assert_eq!(positions(&[0, 9, 1, 8], Some(1)), [2, 3, 1, 4]);
    }
}
