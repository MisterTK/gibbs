# Sprint 2 Autoresearch Log & Notes

## Iteration 0 (Dynamic Metric Baseline)
- **Timestamp**: 2026-10-12T18:50:00Z
- **Commit**: `f96b199`
- **Metric (`geomean_roofline_pct`)**: 19.29%
- **Status**: KEEP (Baseline)
- **Guard**: PASS (`cargo test --lib -q graph:: && gibbs --bench-parity bench/loop.json && scripts/generic_guard.sh`)

### Model Performance & Roofline Saturation:
- `gemma-4-E2B-it-Q4_K_M`: 60.3 tok/s (roofline: 38.3%)
- `Qwen3.5-2B-UD-Q4_K_XL`: 56.3 tok/s (roofline: 42.1%)
- `DavidAU/Qwen3.5-9B-The-Defiant-Fable`: 25.7 tok/s (roofline: 49.3%)
- `HuggingFaceTB/SmolLM2-135M`: 151.0 tok/s (roofline: 15.7%)
- `unsloth/Llama-3.2-1B-Instruct-Q4_0`: 22.1 tok/s (roofline: 6.6%)
- `gemma-4-E2B-it-qat-UD-Q2_K_XL`: 5.6 tok/s (roofline: 4.7%)
