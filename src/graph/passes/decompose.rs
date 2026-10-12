//! Rewrite rules that express an op the GPU lacks as ops it has.
//!
//! The rule table is empty; G4.1 adds rules (e.g. `Conv1d` → `Unsqueeze` + `Conv2d` + `Squeeze`).

use crate::graph::{DType, Graph, Op};

/// Rewrites `g` in place so every node with a rule is replaced by its decomposition.
pub fn run(_g: &mut Graph) {}

/// The rule that decomposes `op` at `dtype`, if one exists.
pub fn rule_for(_op: &Op, _dtype: DType) -> Option<&'static str> {
    None
}
