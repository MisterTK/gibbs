# Sprint 2 Autoresearch Log & Notes

## Iteration 0 (Baseline)
- **Timestamp**: 2026-10-12T18:36:29Z
- **Commit**: `f79433d`
- **Metric (`geomean_roofline_pct`)**: 90.00%
- **Status**: KEEP (Baseline)
- **Guard**: PASS (`cargo test --lib -q graph:: && gibbs --bench-parity bench/loop.json && scripts/generic_guard.sh`)

### Model Speeds:
- `gemma-4-E2B-it-Q4_K_M`: 69.4 tok/s
- `Qwen3.5-2B-UD-Q4_K_XL`: 64.7 tok/s
- `Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M`: 32.2 tok/s
- `Llama-3.2-1B-Instruct-Q4_0`: 27.5 tok/s
- `SmolLM2-135M-Instruct`: 150.7 tok/s
- `gemma-4-E2B-it-qat-UD-Q2_K_XL`: 7.1 tok/s
