# Roofline loop notes

Goal: maximize `geomean_roofline_pct` of `bench/loop.json` (Metal only; decode tok/s vs the memory-bandwidth
roofline of streamed bytes at the measured ceiling). 90% per model is the gate; E2B QAT must reach >= 150 tok/s.

## Keep rule (frozen)
Keep a commit only if ALL hold:
- `delta_pp >= max(0.30, noise_pp)` from `scripts/loop_verify.sh` (ABBA, base vs HEAD, one shared ceiling);
- `verify.json: ratios_ok` (every model's mean candidate tok/s >= (1 - max(3%, 2 x that model's base1/base2 spread)) x its mean
  base tok/s; separate processes of one binary differ by up to ~10% on small models, so a fixed 3% guard rejects A/A runs);
- `scripts/loop_guard.sh` passes.

On keep: `cp target/release/gibbs target/loop-base/gibbs && git rev-parse HEAD > target/loop-base/COMMIT`.
Otherwise: `git revert HEAD --no-edit`.
Phase-step rows (Phase 1-2 commits touching the decode path) are logged with status `infra` and kept if
`delta_pp >= -max(0.30, noise_pp)`.

## Frozen
`bench/loop.json`, `bench/prompts/*`, `scripts/loop_*`, `scripts/with_gpu_lock.py`, the metric code in `src/bench.rs`.
Editable: `src/graph/**`, `src/hardware/**`, `src/scheduler/mod.rs`; `src/model/**` only for graph-input plumbing
a generic capability requires. `scripts/generic_guard.sh` forbids model-specific identifiers.

## Per-iteration template
1. Review: `git log --oneline -20`, last 10 TSV rows, this file. Never repeat a discarded hypothesis (incl.
   `sprint2.tsv` rows 5-7: concurrent encoder with byte-range barriers, bare rows_per_sg sweep, fused decode+dot for TQ2_0).
2. Breakdown: per-model wall/gpu/run/host/weights ms + dispatches from `target/loop/verify.json`;
   `GIBBS_PROFILE_KERNELS=1 target/release/gibbs --bench bench/profile.json` per-kind table.
3. Roofline arithmetic: targeted ms term, expected ms saved per model, `expected_pp ~ geomean x mean_i(dpct_i/pct_i)`.
4. References: `autoresearch/luminal-catalog.md` row -> TSV `luminal_ref` (or why none applies).
5. Design note: generic op-level pattern, applicability, invariants, fallback, test.
6. Implement one change + test; `cargo build --release`; `git commit -m "experiment: <hypothesis>"`.
7. Verify then Guard; apply keep rule; promote base or revert.
8. Log TSV row (expected vs actual pp; breakdown before/after). actual < 50% expected -> reason here.
9. Checkpoint every 13 iterations: matrix summary, matrix_gen, graphify update, trend + kept/discarded ratio, plateau check.

## Stop rules
(A) every loop.json model >= 90%, E2B QAT >= 150 tok/s, every matrix item's latest row ok, >= 90%, parity ok,
oracle != fail. (B) three consecutive checkpoints each improve the kept geomean by < 0.5 pp after E1-E4 were tried.

## Iterations

Verify is per-model ABBA (base, cand, cand, base for each suite entry); the first whole-suite ABBA drifted 25% over four
minutes (thermal), giving noise_pp 5.6; per-model interleaving gives noise_pp ~0.2 (rows 0n2, 0n3 in the TSV).
