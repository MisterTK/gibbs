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

/// Winners measured while another process kept the GPU busier than this are used in-process but
/// never persisted: contended timings would poison later runs' variant choices. `GIBBS_TUNE_BUSY_PCT`
/// overrides it (set it to the verify gate's `QUIET_MAX_PCT` on a machine whose idle floor is higher).
pub const BUSY_PCT: u32 = 20;

/// Whether foreign load kept the GPU busy: sampled ONCE per process, on first use, before this process
/// has run any tuning kernel (a later sample would include its own earlier benchmarks, and the
/// utilization counter is a trailing average, so one reading per process is the only clean one).
pub fn foreign_busy() -> bool {
    static BUSY: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let limit = std::env::var("GIBBS_TUNE_BUSY_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(BUSY_PCT);
        if !gpu_utilization().is_some_and(|u| u > limit) {
            return false;
        }
        // IOAccelerator utilization decays within ~320 ms after a peer benchmark process exits:
        // wait 400 ms once before classifying the GPU as foreign-busy.
        std::thread::sleep(std::time::Duration::from_millis(400));
        gpu_utilization().is_some_and(|u| u > limit)
    });
    *BUSY
}

/// The `Device Utilization %` field of an `ioreg -r -d 1 -c IOAccelerator` dump.
pub fn parse_utilization(ioreg: &str) -> Option<u32> {
    let rest =
        &ioreg[ioreg.find("\"Device Utilization %\"=")? + "\"Device Utilization %\"=".len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Current GPU utilization (percent) from the IOAccelerator registry; `None` when unreadable.
/// Sample it before this process loads the GPU, so only foreign load is seen.
pub fn gpu_utilization() -> Option<u32> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    parse_utilization(&String::from_utf8_lossy(&out.stdout))
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

/// Under foreign GPU load an alternative must beat the default's throughput by this fraction in every
/// measurement round: contended timings differ from a quiet GPU's by more than the 1% tie band.
pub const BUSY_MARGIN: f64 = 0.08;

/// The variant to use from `rounds` of timings (`GB/s`, each round measuring the variants in another
/// order). A single quiet round takes [`choose`] over that round; two or more quiet rounds require
/// every round to agree on [`choose`], otherwise `default` wins. A `busy` GPU keeps `default` (the
/// proven variant) unless another variant is at least [`BUSY_MARGIN`] faster than it in EVERY round;
/// among those the one with the best worst-round ratio wins. `None`: no usable timing.
pub fn decide(
    rounds: &[Vec<(GemvVariant, f64)>],
    default: GemvVariant,
    busy: bool,
) -> Option<GemvVariant> {
    let first = rounds.first()?;
    if !busy {
        if rounds.len() == 1 {
            return choose(first);
        }
        let w0 = choose(first)?;
        return Some(if rounds[1..].iter().all(|r| choose(r) == Some(w0)) {
            w0
        } else {
            default
        });
    }
    let rate = |round: &[(GemvVariant, f64)], v: GemvVariant| {
        round
            .iter()
            .find(|t| t.0 == v)
            .map(|t| t.1)
            .filter(|g| g.is_finite() && *g > 0.0)
    };
    let worst_ratio = |v: GemvVariant| {
        rounds
            .iter()
            .map(|r| Some(rate(r, v)? / rate(r, default)?))
            .try_fold(f64::INFINITY, |m, ratio| Some(m.min(ratio?)))
    };
    Some(
        first
            .iter()
            .map(|t| t.0)
            .filter(|&v| v != default)
            .filter_map(|v| Some((v, worst_ratio(v)?)))
            .filter(|c| c.1 >= 1.0 + BUSY_MARGIN)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(default, |c| c.0),
    )
}

/// Maximum relative spread `(hi - lo) / hi` across rounds for a winner to be persisted to disk.
pub const MAX_ROUND_SPREAD: f64 = 0.05;

/// Whether `winner` won [`choose`] in every round of `rounds` and its throughput stayed within
/// [`MAX_ROUND_SPREAD`] (5%) across rounds. Single-round measurements return `true`.
pub fn rounds_consistent(rounds: &[Vec<(GemvVariant, f64)>], winner: GemvVariant) -> bool {
    if rounds.len() < 2 {
        return true;
    }
    if rounds.iter().any(|r| choose(r) != Some(winner)) {
        return false;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for r in rounds {
        let Some((_, g)) = r.iter().find(|t| t.0 == winner) else {
            return false;
        };
        if !g.is_finite() {
            return false;
        }
        lo = lo.min(*g);
        hi = hi.max(*g);
    }
    lo > 0.0 && hi.is_finite() && (hi - lo) / hi <= MAX_ROUND_SPREAD
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Tuned {
    pub variant: GemvVariant,
    pub us: f64,
    pub gbps: f64,
}

/// Persisted winners: `"<format>|<rows>|<cols>|<kernel hash>"` -> result. Winners timed under foreign
/// GPU load (`insert_contended`) answer `get` for this process only and are never written by `save`.
#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TuneCache {
    entries: BTreeMap<String, Tuned>,
    #[serde(skip)]
    contended: BTreeMap<String, Tuned>,
}

impl TuneCache {
    pub fn load(path: &Path) -> Self {
        let raw: BTreeMap<String, serde_json::Value> = std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let entries = raw
            .into_iter()
            .filter_map(|(k, v)| Some((k, serde_json::from_value(v).ok()?)))
            .collect();
        Self {
            entries,
            contended: BTreeMap::new(),
        }
    }

    /// Writes the persisted winners, preserving any entries on disk written by other builds.
    pub fn save(&self, path: &Path) {
        let write = || -> std::io::Result<()> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            let mut raw: BTreeMap<String, serde_json::Value> = std::fs::read(path)
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or_default();
            for (k, v) in &self.entries {
                if let Ok(val) = serde_json::to_value(v) {
                    raw.insert(k.clone(), val);
                }
            }
            // Written whole through a rename: concurrent processes never read a torn file.
            let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
            std::fs::write(&tmp, serde_json::to_vec_pretty(&raw)?)?;
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
        self.entries
            .get(key)
            .or_else(|| self.contended.get(key))
            .copied()
    }

    pub fn insert(&mut self, key: String, t: Tuned) {
        self.contended.remove(&key);
        self.entries.insert(key, t);
    }

    /// A winner measured while the GPU was busy: kept for this process, not persisted.
    pub fn insert_contended(&mut self, key: String, t: Tuned) {
        self.contended.insert(key, t);
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

    #[test]
    fn contended_winners_serve_this_process_but_never_reach_disk() {
        let dir = std::env::temp_dir().join(format!("gibbs-tune-busy-{}", std::process::id()));
        let path = dir.join("tune.json");
        let t = |g| Tuned {
            variant: GemvVariant::generated(2),
            us: 1.0,
            gbps: g,
        };
        let mut c = TuneCache::default();
        c.insert_contended("busy".into(), t(100.0));
        c.insert("quiet".into(), t(200.0));
        assert_eq!(c.get("busy").map(|x| x.gbps), Some(100.0));
        c.save(&path);
        let back = TuneCache::load(&path);
        assert!(back.get("busy").is_none());
        assert_eq!(back.get("quiet").map(|x| x.gbps), Some(200.0));
        // A later quiet measurement replaces the contended one.
        c.insert("busy".into(), t(250.0));
        c.save(&path);
        assert_eq!(
            TuneCache::load(&path).get("busy").map(|x| x.gbps),
            Some(250.0)
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn busy_gpu_keeps_the_default_unless_an_alternative_wins_every_round_by_the_margin() {
        let (def, hand, r1, r4) = (
            GemvVariant::generated(2),
            GemvVariant::HAND,
            GemvVariant::generated(1),
            GemvVariant::generated(4),
        );
        let round = |d: f64, h: f64, a: f64, b: f64| vec![(hand, h), (r1, a), (def, d), (r4, b)];
        assert_eq!(
            decide(&[round(100.0, 120.0, 90.0, 80.0)], def, false),
            Some(hand)
        );
        assert_eq!(
            decide(
                &[
                    round(100.0, 120.0, 90.0, 80.0),
                    round(100.0, 105.0, 90.0, 80.0)
                ],
                def,
                true
            ),
            Some(def)
        );
        assert_eq!(
            decide(&[round(100.0, 107.9, 90.0, 80.0)], def, true),
            Some(def)
        );
        assert_eq!(
            decide(
                &[
                    round(100.0, 108.5, 90.0, 80.0),
                    round(100.0, 130.0, 90.0, 80.0)
                ],
                def,
                true
            ),
            Some(hand)
        );
        assert_eq!(
            decide(
                &[
                    round(100.0, 112.0, 90.0, 140.0),
                    round(100.0, 150.0, 90.0, 115.0)
                ],
                def,
                true
            ),
            Some(r4)
        );
        assert_eq!(decide(&[], def, true), None);
        assert_eq!(decide(&[], def, false), None);
    }

    #[test]
    fn quiet_two_round_agreement_and_consistency_gate() {
        let (def, r1, r2) = (
            GemvVariant::HAND,
            GemvVariant::generated(1),
            GemvVariant::generated(2),
        );
        // Two quiet rounds agree on Gen(1) within 5% spread (201.0 vs 198.0).
        let agree = [
            vec![(def, 180.0), (r1, 201.0), (r2, 190.0)],
            vec![(r2, 189.0), (r1, 198.0), (def, 179.0)],
        ];
        assert_eq!(decide(&agree, def, false), Some(r1));
        assert!(rounds_consistent(&agree, r1));

        // Two quiet rounds disagree (Gen(2) in round 0, Gen(1) in round 1) -> returns default.
        let disagree = [
            vec![(def, 180.0), (r1, 195.0), (r2, 210.0)],
            vec![(r2, 190.0), (r1, 205.0), (def, 180.0)],
        ];
        assert_eq!(decide(&disagree, def, false), Some(def));
        assert!(!rounds_consistent(&disagree, def));
        assert!(!rounds_consistent(&disagree, r1));
        assert!(!rounds_consistent(&disagree, r2));

        // Both rounds pick Gen(1), but rates differ by > 5% (200.0 vs 170.0 -> 15% spread).
        let noisy = [
            vec![(def, 150.0), (r1, 200.0), (r2, 160.0)],
            vec![(r2, 155.0), (r1, 170.0), (def, 145.0)],
        ];
        assert_eq!(decide(&noisy, def, false), Some(r1));
        assert!(!rounds_consistent(&noisy, r1));
    }

    #[test]
    fn load_and_save_preserve_entries_with_unknown_variants_from_other_builds() {
        let dir = std::env::temp_dir().join(format!("gibbs-tune-compat-{}", std::process::id()));
        let path = dir.join("tune.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            r#"{"future|16|16|1": {"variant": {"kernel": "FutureKernel", "rows_per_sg": 0}, "us": 1.0, "gbps": 250.0}}"#,
        )
        .unwrap();
        let mut c = TuneCache::load(&path);
        c.insert(
            "known|16|16|1".into(),
            Tuned {
                variant: GemvVariant::HAND,
                us: 1.0,
                gbps: 200.0,
            },
        );
        c.save(&path);
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(raw.contains_key("future|16|16|1"));
        assert!(raw.contains_key("known|16|16|1"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn utilization_is_parsed_from_the_ioreg_dump() {
        let dump = r#""PerformanceStatistics" = {"Alloc system memory"=1,"Device Utilization %"=37,"Renderer Utilization %"=4}"#;
        assert_eq!(parse_utilization(dump), Some(37));
        assert_eq!(parse_utilization("nothing here"), None);
    }
}
