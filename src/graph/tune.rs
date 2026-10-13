//! On-device GEMV variant tuning: which kernel (hand-written or generated) and how many rows per
//! simdgroup read a given packed weight fastest on this GPU. The Metal session measures every
//! variant of a `(format, rows, cols)` the first time it plans such a step; winners persist in
//! `<cache>/host/tune-<device>.json`, keyed by a hash of the exact kernel sources, so editing a
//! kernel invalidates only its own entries.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GemvKernel {
    /// The hand-written kernel of `metal_kernels.metal` (formats that have one).
    Hand,
    /// The kernel generated from the format's codec (`codegen::gemv_packed`).
    Gen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GemvVariant {
    pub kernel: GemvKernel,
    /// Rows one simdgroup computes (generated kernels); `0` for the hand kernels, whose row
    /// mapping is fixed.
    pub rows_per_sg: u8,
}

/// Rows per simdgroup the generated kernel is tried with.
pub const GEN_ROWS: [u8; 3] = [1, 2, 4];

impl GemvVariant {
    pub const HAND: Self = Self {
        kernel: GemvKernel::Hand,
        rows_per_sg: 0,
    };
    pub const fn generated(rows_per_sg: u8) -> Self {
        Self {
            kernel: GemvKernel::Gen,
            rows_per_sg,
        }
    }
    /// `hand` or `gen/r4`.
    pub fn label(&self) -> String {
        match self.kernel {
            GemvKernel::Hand => "hand".into(),
            GemvKernel::Gen => format!("gen/r{}", self.rows_per_sg),
        }
    }
}

/// Timings within this fraction of the best are ties (measurement noise): the simpler variant wins.
const TIE: f64 = 0.01;

/// The fastest variant of `timings` (`GB/s`, higher is better). Variants within 1% of the best tie;
/// ties go to fewer rows per simdgroup, then to the hand kernel.
pub fn choose(timings: &[(GemvVariant, f64)]) -> Option<GemvVariant> {
    let best = timings
        .iter()
        .map(|t| t.1)
        .fold(f64::NEG_INFINITY, f64::max);
    if !best.is_finite() {
        return None;
    }
    timings
        .iter()
        .filter(|t| t.1 >= best * (1.0 - TIE))
        .map(|t| t.0)
        .min_by_key(|v| (v.rows_per_sg, v.kernel == GemvKernel::Gen))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Tuned {
    pub variant: GemvVariant,
    pub us: f64,
    pub gbps: f64,
}

/// Persisted winners: `"<format>|<rows>|<cols>|<kernel hash>"` -> result.
#[derive(Default, Serialize, Deserialize)]
pub struct TuneCache(BTreeMap<String, Tuned>);

impl TuneCache {
    pub fn load(path: &Path) -> Self {
        std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) {
        let write = || -> std::io::Result<()> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            // Written whole through a rename: concurrent processes never read a torn file.
            let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
            std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
            std::fs::rename(&tmp, path)
        };
        if let Err(e) = write() {
            tracing::warn!("tune cache: saving {}: {e}", path.display());
        }
    }

    pub fn key(format: &str, rows: usize, cols: usize, kernel_hash: u64) -> String {
        format!("{format}|{rows}|{cols}|{kernel_hash:016x}")
    }

    pub fn get(&self, key: &str) -> Option<Tuned> {
        self.0.get(key).copied()
    }

    pub fn insert(&mut self, key: String, t: Tuned) {
        self.0.insert(key, t);
    }
}

/// `GIBBS_TUNE=off` keeps each format's default variant (no measurement, no cache).
pub fn enabled() -> bool {
    !std::env::var("GIBBS_TUNE").is_ok_and(|v| matches!(v.as_str(), "off" | "0" | "false"))
}

/// Source text of the kernel function `name` in `msl` (from `kernel void name(` to its closing
/// brace), for hashing exactly what a hand variant runs.
pub fn kernel_source<'a>(msl: &'a str, name: &str) -> &'a str {
    let Some(start) = msl.find(&format!("kernel void {name}(")) else {
        return "";
    };
    let body = &msl[start..];
    let Some(open) = body.find('{') else {
        return body;
    };
    let mut depth = 0usize;
    for (i, c) in body[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &body[..open + i + 1];
                }
            }
            _ => {}
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_takes_the_fastest_and_breaks_ties_toward_simpler() {
        let (hand, r1, r2, r4) = (
            GemvVariant::HAND,
            GemvVariant::generated(1),
            GemvVariant::generated(2),
            GemvVariant::generated(4),
        );
        assert_eq!(
            choose(&[(hand, 200.0), (r1, 180.0), (r2, 250.0), (r4, 240.0)]),
            Some(r2)
        );
        // Within 1%: fewer rows per simdgroup, then Hand.
        assert_eq!(choose(&[(r4, 250.0), (r2, 249.0), (r1, 100.0)]), Some(r2));
        assert_eq!(
            choose(&[(r2, 250.0), (hand, 249.9), (r4, 100.0)]),
            Some(hand)
        );
        assert_eq!(choose(&[(r2, 250.0), (r1, 250.0)]), Some(r1));
        assert_eq!(choose(&[]), None);
        assert_eq!(choose(&[(r1, f64::NAN)]), None);
    }

    #[test]
    fn kernel_source_extracts_one_function() {
        let msl = "kernel void a(int x) { if (x) { x++; } }\nkernel void b(int y) { y--; }";
        assert_eq!(
            kernel_source(msl, "a"),
            "kernel void a(int x) { if (x) { x++; } }"
        );
        assert_eq!(kernel_source(msl, "b"), "kernel void b(int y) { y--; }");
        assert_eq!(kernel_source(msl, "c"), "");
    }

    #[test]
    fn cache_round_trips_and_keys_by_kernel_hash() {
        let dir = std::env::temp_dir().join(format!("gibbs-tune-test-{}", std::process::id()));
        let path = dir.join("tune.json");
        let mut c = TuneCache::default();
        let k = TuneCache::key("GGUF_Q4_0", 4096, 4096, 7);
        c.insert(
            k.clone(),
            Tuned {
                variant: GemvVariant::generated(4),
                us: 12.5,
                gbps: 231.0,
            },
        );
        c.save(&path);
        let back = TuneCache::load(&path);
        assert_eq!(
            back.get(&k).map(|t| t.variant),
            Some(GemvVariant::generated(4))
        );
        assert!(back
            .get(&TuneCache::key("GGUF_Q4_0", 4096, 4096, 8))
            .is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
