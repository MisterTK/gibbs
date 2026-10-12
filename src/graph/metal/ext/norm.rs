//! Metal lowering group `norm`: planning, encoding and kernels for its ops.
//! Kernels live in `norm.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.

use super::{Enc, PlanOut};
use super::super::{Planned, Session};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("norm.metal");
/// Compute functions in `norm.metal`.
pub(super) const KERNELS: &[&str] = &[];

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
pub(super) fn supports(_op: &Op, _dtype: DType) -> bool {
    false
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    _g: &Graph,
    _node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    None
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match *st {}
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, _v: ValueId) -> bool {
    match *st {}
}

pub(super) fn encode(_sess: &Session, _e: &Enc, _s: &Planned, st: &Step) -> Result<(), String> {
    match *st {}
}
