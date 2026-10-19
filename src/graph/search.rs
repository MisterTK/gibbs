//! On-device plan search (research sprint 3, phase 4; luminal L18 `luminal_metal/src/search.rs`),
//! opt-in via `GIBBS_PLAN_SEARCH=1`.
//!
//! * **Unit.** A pattern: the e-graph fusion choices that share one structural key (a repeated
//!   layer subgraph). A candidate sets every instance of its patterns to the opposite of the
//!   cost model's pick, so each candidate is one model-level plan. GEMV kernel variants stay with
//!   the per-shape tuner ([`crate::graph::tune`]).
//! * **Candidates.** Every single-pattern flip (least predicted loss first), then combinations
//!   of the singles that won.
//! * **Timing.** The caller's `time(flips)` returns the best GPU µs of one decode step of that
//!   plan. Each phase times the incumbent and its candidates in two opposite-order rounds;
//!   [`tune::beats`] keeps the incumbent unless a candidate beats it by [`tune::QUIET_MARGIN`]
//!   in BOTH rounds ([`tune::BUSY_MARGIN`] under foreign load). The F8 fusion search (ledger
//!   row 33) chose noise under load because it lacked exactly this gate.
//! * **Budget.** Each phase fits its candidate count into what is left of [`BUDGET`] after
//!   timing the incumbent once; a timeout or an incumbent error keeps the cost-model plan.
//! * **Cache.** The winning pattern states persist in the tune file (one per device) under
//!   `plan|<patterns hash>|<kernel-source hash>`, never when measured under foreign load.

use crate::graph::egraph::Choice;
use crate::graph::tune::{self, TuneCache};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

/// `GIBBS_PLAN_SEARCH=1` searches fusion choices on the device at load (default off).
pub fn enabled() -> bool {
    std::env::var("GIBBS_PLAN_SEARCH").is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true"))
}

/// First-load search time per model.
pub const BUDGET: Duration = Duration::from_secs(2);

// ponytail: at most 12 singles, then combinations of the best 4 winners (11 + the best single);
// no luminal genome/generations loop. Add one when models carry more patterns than this covers.
pub const MAX_CANDIDATES: usize = 12;
pub const MAX_COMBINED: usize = 4;

/// The choices of one structural key, with the cost model's state of its first instance.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub key: u64,
    pub on: bool,
    pub ids: Vec<usize>,
    /// Predicted µs lost by flipping every instance (Σ |delta_us|).
    pub loss_us: f64,
}

/// `choices` grouped by pattern key, least predicted loss first (ties: key).
pub fn patterns(choices: &[Choice]) -> Vec<Pattern> {
    let mut out: Vec<Pattern> = Vec::new();
    for c in choices {
        match out.iter_mut().find(|p| p.key == c.pattern) {
            Some(p) => {
                p.ids.push(c.id);
                p.loss_us += c.delta_us.abs();
            }
            None => out.push(Pattern {
                key: c.pattern,
                on: c.on,
                ids: vec![c.id],
                loss_us: c.delta_us.abs(),
            }),
        }
    }
    out.sort_by(|a, b| a.loss_us.total_cmp(&b.loss_us).then(a.key.cmp(&b.key)));
    out
}

/// Ids of the choices whose cost-model state differs from the state `set` asks for their key.
pub fn flips(choices: &[Choice], set: &[(u64, bool)]) -> Vec<usize> {
    choices
        .iter()
        .filter(|c| set.iter().any(|&(k, on)| k == c.pattern && on != c.on))
        .map(|c| c.id)
        .collect()
}

/// A persisted search result: the pattern states found faster than the cost model's
/// (empty: the cost-model plan won), with the chosen and incumbent best µs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pick {
    pub set: Vec<(u64, bool)>,
    pub us: f64,
    pub base_us: f64,
}

/// Tune-file key of a model's patterns (keys and instance counts) under `kernels_hash`.
pub fn cache_key(pats: &[Pattern], kernels_hash: u64) -> String {
    let mut ks: Vec<(u64, u64)> = pats.iter().map(|p| (p.key, p.ids.len() as u64)).collect();
    ks.sort_unstable();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in ks
        .iter()
        .flat_map(|&(k, n)| k.to_le_bytes().into_iter().chain(n.to_le_bytes()))
    {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    format!("plan|{h:016x}|{kernels_hash:016x}")
}

/// Choice ids to flip from the cost-model extraction: the cached pick, else a timed search
/// (`time(&[])` is the incumbent). The result persists to `path` unless `busy`; a timeout or
/// an incumbent error logs one line and returns no flips (nothing cached).
pub fn run(
    choices: &[Choice],
    kernels_hash: u64,
    cache: &mut TuneCache<Pick>,
    path: &Path,
    busy: bool,
    budget: Duration,
    mut time: impl FnMut(&[usize]) -> Result<f64, String>,
) -> Vec<usize> {
    let pats = patterns(choices);
    if pats.is_empty() {
        return Vec::new();
    }
    let key = cache_key(&pats, kernels_hash);
    if let Some(p) = cache.get(&key) {
        return flips(choices, &p.set);
    }
    let t0 = Instant::now();
    let set_of = |cand: &[usize]| -> Vec<(u64, bool)> {
        cand.iter().map(|&i| (pats[i].key, !pats[i].on)).collect()
    };
    let margin = if busy {
        tune::BUSY_MARGIN
    } else {
        tune::QUIET_MARGIN
    };
    let mut timed = |cand: &[usize]| time(&flips(choices, &set_of(cand)));
    match search(pats.len(), t0 + budget, margin, &mut timed) {
        Ok((best, us, base_us, n)) => {
            let pick = Pick {
                set: set_of(&best),
                us,
                base_us,
            };
            tracing::info!(
                "plan search: {} patterns, {n} candidates timed, {} flipped, {base_us:.0} -> {us:.0} us, {} ms{}",
                pats.len(),
                best.len(),
                t0.elapsed().as_millis(),
                if busy { " (GPU busy: not persisted)" } else { "" }
            );
            let out = flips(choices, &pick.set);
            if busy {
                cache.insert_contended(key, pick);
            } else {
                cache.insert(key, pick);
                cache.save(path);
            }
            out
        }
        Err(e) => {
            tracing::warn!("plan search: {e}; cost-model plan kept");
            Vec::new()
        }
    }
}

/// The best candidate (pattern indices; empty = incumbent), its µs, the incumbent's µs and the
/// number of candidates timed.
fn search(
    n: usize,
    deadline: Instant,
    margin: f64,
    time: &mut impl FnMut(&[usize]) -> Result<f64, String>,
) -> Result<(Vec<usize>, f64, f64, usize), String> {
    let singles: Vec<Vec<usize>> = (0..n.min(MAX_CANDIDATES)).map(|i| vec![i]).collect();
    let a = measure(&singles, deadline, time)?;
    let winners = ranked(&a.rounds, margin);
    let Some(&w0) = winners.first() else {
        return Ok((Vec::new(), a.us[0], a.us[0], a.n));
    };
    let top: Vec<usize> = winners
        .iter()
        .take(MAX_COMBINED)
        .map(|&w| singles[w - 1][0])
        .collect();
    let single = (singles[w0 - 1].clone(), a.us[w0], a.us[0], a.n);
    if top.len() < 2 {
        return Ok(single);
    }
    let cands = combos(&top);
    // A phase-B timeout still leaves phase A's measured winner.
    let Ok(b) = measure(&cands, deadline, time) else {
        return Ok(single);
    };
    let w = tune::beats(&b.rounds, 0, margin);
    Ok(match w {
        0 => (Vec::new(), b.us[0], b.us[0], a.n + b.n),
        w => (cands[w - 1].clone(), b.us[w], b.us[0], a.n + b.n),
    })
}

/// Candidate indices (1-based, 0 = incumbent) that beat the incumbent in every round, best first.
fn ranked(rounds: &[Vec<(usize, f64)>], margin: f64) -> Vec<usize> {
    let mut pool = rounds.to_vec();
    let mut out = Vec::new();
    loop {
        match tune::beats(&pool, 0, margin) {
            0 => return out,
            w => {
                out.push(w);
                pool.iter_mut().for_each(|r| r.retain(|t| t.0 != w));
            }
        }
    }
}

/// The best single `top[0]`, then every combination of ≥ 2 of `top` (best-ranked members first),
/// capped at [`MAX_CANDIDATES`]; each as sorted pattern indices.
fn combos(top: &[usize]) -> Vec<Vec<usize>> {
    let mut masks: Vec<u32> = (1u32..1 << top.len())
        .filter(|m| m.count_ones() >= 2)
        .collect();
    let rank = |m: u32| (0..top.len()).filter(|j| m & (1 << j) != 0).sum::<usize>();
    masks.sort_by_key(|&m| (rank(m), m.count_ones()));
    let pick = |m: u32| {
        let mut c: Vec<usize> = (0..top.len())
            .filter(|j| m & (1 << j) != 0)
            .map(|j| top[j])
            .collect();
        c.sort_unstable();
        c
    };
    std::iter::once(vec![top[0]])
        .chain(masks.into_iter().map(pick))
        .take(MAX_CANDIDATES)
        .collect()
}

struct Timed {
    /// Per round `(index, 1 / µs)`, index 0 = incumbent, `i` = candidate `i - 1`.
    rounds: Vec<Vec<(usize, f64)>>,
    /// Best µs per index (∞: every timing of it failed).
    us: Vec<f64>,
    /// Candidates timed.
    n: usize,
}

/// Times the incumbent, keeps as many of `cands` as fit before `deadline`, then times them in
/// two opposite-order rounds (round 1 ends with the incumbent again). A failing candidate is
/// dropped from the decision; a failing incumbent or a passed deadline is an error.
fn measure(
    cands: &[Vec<usize>],
    deadline: Instant,
    time: &mut impl FnMut(&[usize]) -> Result<f64, String>,
) -> Result<Timed, String> {
    let t = Instant::now();
    let base = time(&[])?;
    let per = t.elapsed().as_secs_f64().max(1e-9);
    let left = deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f64();
    // Still to time: every kept candidate twice, the incumbent once more.
    let fit = ((left / per - 1.0) / 2.0).floor().max(0.0) as usize;
    let cands = &cands[..fit.min(cands.len())];
    if cands.is_empty() {
        return Err(format!("budget too short (one plan: {:.0} ms)", per * 1e3));
    }
    let mut us = vec![f64::INFINITY; cands.len() + 1];
    us[0] = base;
    let mut rounds = Vec::new();
    for r in 0..2 {
        let mut order: Vec<usize> = (0..=cands.len()).collect();
        if r == 1 {
            order.reverse();
        }
        let mut round = Vec::with_capacity(order.len());
        for i in order {
            let t = if r == 0 && i == 0 {
                base
            } else if Instant::now() > deadline {
                return Err("budget exceeded".into());
            } else if i == 0 {
                time(&[])?
            } else {
                time(&cands[i - 1]).unwrap_or(f64::NAN)
            };
            us[i] = us[i].min(t);
            round.push((i, 1.0 / t));
        }
        rounds.push(round);
    }
    Ok(Timed {
        rounds,
        us,
        n: cands.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(id: usize, pattern: u64, on: bool, delta_us: f64) -> Choice {
        Choice {
            id,
            kind: "test",
            pattern,
            on,
            delta_us,
        }
    }

    /// Two layers of three patterns: A (key 10, cheapest to flip), B (20), C (30).
    fn model() -> Vec<Choice> {
        vec![
            ch(0, 10, true, 0.1),
            ch(1, 20, true, 1.0),
            ch(2, 30, true, 5.0),
            ch(3, 10, true, 0.1),
            ch(4, 20, true, 1.0),
            ch(5, 30, true, 5.0),
        ]
    }

    /// Fake GPU timer: 1000 µs times the factor of every pattern with a flipped instance.
    fn timer<'a>(
        choices: &'a [Choice],
        factors: &'a [(u64, f64)],
    ) -> impl FnMut(&[usize]) -> Result<f64, String> + 'a {
        move |flips: &[usize]| {
            let mut keys: Vec<u64> = flips.iter().map(|&i| choices[i].pattern).collect();
            keys.sort_unstable();
            keys.dedup();
            Ok(keys.iter().fold(1000.0, |us, k| {
                us * factors.iter().find(|f| f.0 == *k).map_or(1.0, |f| f.1)
            }))
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("gibbs-search-{name}-{}", std::process::id()))
            .join("tune.json")
    }

    fn quiet(choices: &[Choice], factors: &[(u64, f64)]) -> Vec<usize> {
        let mut cache = TuneCache::default();
        let path = tmp("quiet");
        let out = run(
            choices,
            1,
            &mut cache,
            &path,
            false,
            BUDGET,
            timer(choices, factors),
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        out
    }

    #[test]
    fn patterns_group_instances_by_key_least_loss_first() {
        let p = patterns(&model());
        assert_eq!(
            p.iter().map(|p| (p.key, p.ids.clone())).collect::<Vec<_>>(),
            [(10, vec![0, 3]), (20, vec![1, 4]), (30, vec![2, 5])]
        );
        assert!(p.iter().all(|p| p.on));
        assert!((p[2].loss_us - 10.0).abs() < 1e-9);
    }

    #[test]
    fn flips_move_only_instances_off_the_requested_state() {
        let mut c = model();
        c[3].on = false;
        assert_eq!(flips(&c, &[(10, false)]), [0]);
        assert_eq!(flips(&c, &[(10, true)]), [3]);
        assert_eq!(flips(&c, &[(10, false), (30, false)]), [0, 2, 5]);
        assert!(flips(&c, &[]).is_empty());
    }

    #[test]
    fn combos_rank_best_members_first_and_cap() {
        assert_eq!(combos(&[4, 2]), [vec![4], vec![2, 4]]);
        let c = combos(&[7, 3, 5, 9]);
        assert_eq!(c.len(), MAX_CANDIDATES);
        assert_eq!(c[0], [7]);
        assert_eq!(c[1], [3, 7]);
        assert!(c[1..].iter().all(|x| x.len() >= 2));
        assert!(c.contains(&vec![3, 5, 7, 9]));
        let mut u = c.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), c.len());
    }

    #[test]
    fn singles_are_capped_and_each_timed_in_two_rounds() {
        let choices: Vec<Choice> = (0..20).map(|i| ch(i, i as u64, true, 1.0)).collect();
        let mut calls: Vec<Vec<usize>> = Vec::new();
        let mut cache = TuneCache::default();
        let path = tmp("cap");
        let out = run(
            &choices,
            1,
            &mut cache,
            &path,
            false,
            BUDGET,
            |f: &[usize]| {
                calls.push(f.to_vec());
                Ok(1000.0)
            },
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(out.is_empty());
        // Incumbent, 12 singles, 12 singles reversed, incumbent.
        assert_eq!(calls.len(), 2 * MAX_CANDIDATES + 2);
        assert!(calls[0].is_empty() && calls.last().unwrap().is_empty());
        assert_eq!(calls[1], calls[2 * MAX_CANDIDATES]);
    }

    #[test]
    fn ties_and_wins_inside_the_quiet_margin_keep_the_incumbent() {
        let c = model();
        assert!(quiet(&c, &[]).is_empty());
        assert!(quiet(&c, &[(10, 0.99)]).is_empty());
        assert!(quiet(&c, &[(20, 1.0 / (1.0 + tune::QUIET_MARGIN) + 1e-6)]).is_empty());
    }

    #[test]
    fn clear_winners_flip_and_combine() {
        let c = model();
        assert_eq!(quiet(&c, &[(20, 0.95)]), [1, 4]);
        assert_eq!(
            quiet(&c, &[(10, 0.95), (20, 0.97), (30, 1.1)]),
            [0, 1, 3, 4]
        );
        // A combination slower than its best member keeps the single.
        let mut cache = TuneCache::default();
        let path = tmp("combo");
        let mut t = timer(&c, &[(10, 0.95), (20, 0.97)]);
        let out = run(&c, 1, &mut cache, &path, false, BUDGET, |f: &[usize]| {
            let us = t(f)?;
            Ok(if f.len() == 4 { us * 1.2 } else { us })
        });
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert_eq!(out, [0, 3]);
    }

    #[test]
    fn a_win_in_only_one_round_keeps_the_incumbent() {
        let c = model();
        let mut seen = 0;
        let mut cache = TuneCache::default();
        let path = tmp("order");
        let out = run(&c, 1, &mut cache, &path, false, BUDGET, |f: &[usize]| {
            if f.first() == Some(&0) {
                seen += 1;
                return Ok(if seen == 1 { 900.0 } else { 1000.0 });
            }
            Ok(1000.0)
        });
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(out.is_empty());
        assert_eq!(seen, 2);
    }

    #[test]
    fn foreign_load_needs_the_busy_margin_and_is_never_persisted() {
        let c = model();
        let path = tmp("busy");
        let key = cache_key(&patterns(&c), 1);
        let mut cache = TuneCache::default();
        let out = run(
            &c,
            1,
            &mut cache,
            &path,
            true,
            BUDGET,
            timer(&c, &[(10, 0.95)]),
        );
        assert!(out.is_empty());
        let mut cache = TuneCache::default();
        let out = run(
            &c,
            1,
            &mut cache,
            &path,
            true,
            BUDGET,
            timer(&c, &[(10, 0.9)]),
        );
        assert_eq!(out, [0, 3]);
        assert_eq!(cache.get(&key).map(|p| p.set), Some(vec![(10, false)]));
        assert!(!path.exists());
        assert!(TuneCache::<Pick>::load(&path).get(&key).is_none());
    }

    #[test]
    fn cache_round_trips_shares_the_tune_file_and_keys_by_kernels() {
        let c = model();
        let path = tmp("cache");
        let mut tuned = TuneCache::default();
        let tk = TuneCache::key("GGUF_Q4_0", 64, 64, 7);
        tuned.insert(
            tk.clone(),
            tune::Tuned {
                variant: tune::GemvVariant::HAND,
                us: 1.0,
                gbps: 100.0,
            },
        );
        tuned.save(&path);
        let mut cache = TuneCache::default();
        let out = run(
            &c,
            7,
            &mut cache,
            &path,
            false,
            BUDGET,
            timer(&c, &[(30, 0.8)]),
        );
        assert_eq!(out, [2, 5]);
        let key = cache_key(&patterns(&c), 7);
        assert_ne!(key, cache_key(&patterns(&c), 8));
        assert_ne!(key, cache_key(&patterns(&c[..3]), 7));
        let mut back = TuneCache::<Pick>::load(&path);
        let pick = back.get(&key).unwrap();
        assert_eq!(pick.set, [(30, false)]);
        assert!(pick.us < pick.base_us);
        assert!(back.get(&tk).is_none());
        let tback: TuneCache = TuneCache::load(&path);
        assert!(tback.get(&tk).is_some() && tback.get(&key).is_none());
        // A later load reuses the pick without timing.
        let out = run(&c, 7, &mut back, &path, false, BUDGET, |_: &[usize]| {
            panic!("cached pick must not be re-timed")
        });
        assert_eq!(out, [2, 5]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn timeout_and_incumbent_errors_keep_the_cost_model_plan_uncached() {
        let c = model();
        let path = tmp("timeout");
        let key = cache_key(&patterns(&c), 1);
        let mut cache = TuneCache::default();
        let out = run(
            &c,
            1,
            &mut cache,
            &path,
            false,
            Duration::ZERO,
            timer(&c, &[(10, 0.5)]),
        );
        assert!(out.is_empty() && cache.get(&key).is_none());
        let out = run(&c, 1, &mut cache, &path, false, BUDGET, |f: &[usize]| {
            if f.is_empty() {
                Err("no device".into())
            } else {
                Ok(1.0)
            }
        });
        assert!(out.is_empty() && cache.get(&key).is_none());
        // A failing candidate is only dropped.
        let out = run(
            &c,
            1,
            &mut cache,
            &path,
            false,
            BUDGET,
            |f: &[usize]| match f.first() {
                Some(0) => Err("plan".into()),
                Some(1) => Ok(900.0),
                _ => Ok(1000.0),
            },
        );
        assert_eq!(out, [1, 4]);
        assert!(!path.exists() || cache.get(&key).is_some());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
