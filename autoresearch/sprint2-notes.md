# Sprint 2 Autoresearch Log & Notes

## Iteration 0 (Dynamic Metric Baseline)
- **Timestamp**: 2026-10-12T18:50:00Z
- **Commit**: `f96b199`
- **Metric (`geomean_roofline_pct`)**: 19.29%
- **Status**: KEEP (Baseline)
- **Guard**: PASS

## Iteration 1 (Fused MLP with Sliced Gate-Up Linear)
- **Timestamp**: 2026-10-12T18:55:30Z
- **Commit**: `bb32ce2`
- **Metric (`geomean_roofline_pct`)**: 19.58% (+0.29 pp)
- **Status**: KEEP (Delta >= +0.20 pp threshold)
- **Luminal Reference**: `matmul.rs:79-103` (fused linear gate-up projections)
- **Guard**: PASS (`cargo test --lib -q graph:: && gibbs --bench-parity bench/loop.json && scripts/generic_guard.sh`)
- **Hypothesis**: Fusing transformer MLP blocks where Gate and Up projections originate from a shared sliced Linear into `Op::Megakernel` removes intermediate round-trips to the activation buffer and coalesces dispatches.
