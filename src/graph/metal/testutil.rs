//! Shared helpers for lowering-group tests: one `Runtime`, a CPU-vs-Metal runner, error metrics.

use super::{Runtime, Session};
use crate::graph::{cpu, Graph, Tensor, Weights};
use std::sync::{Arc, LazyLock};

static RT: LazyLock<Arc<Runtime>> =
    LazyLock::new(|| Runtime::new(tempfile::tempdir().unwrap().path()).unwrap());

pub(crate) fn rt() -> Arc<Runtime> {
    RT.clone()
}

/// Deterministic values in [-1, 1).
pub(crate) fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// (CPU reference, Metal) outputs of the graph's first output.
pub(crate) fn both(g: &Graph, w: &Weights, inputs: Vec<Tensor>) -> (Vec<f32>, Vec<f32>) {
    let want = cpu::run_stateless(g, w, inputs.clone())
        .unwrap()
        .remove(0)
        .into_f32()
        .unwrap();
    let got = Session::new(rt(), w)
        .unwrap()
        .run(g, inputs)
        .unwrap()
        .remove(0)
        .into_f32()
        .unwrap();
    assert_eq!(got.len(), want.len());
    (want, got)
}

/// max |a - b| relative to max(1, max |a|).
pub(crate) fn rel_err(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    want.iter().zip(got).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}
