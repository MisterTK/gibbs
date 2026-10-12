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

## G1 finding: Gemma 4 on Metal was never measured (2026-10-12)
Removing the silent `Pin(Metal)` → CPU fallback (`compile_gemma4_program` used to `warn!` and return `None`) showed that every Gemma 4 "Metal" number in this log, including iteration 0–2 and `bench/loop.json`'s Q2_K_XL row (6 tok/s, `CPU (NEON bf16)` in the CLI), was the CPU executor. Real Metal:
- E2B QAT (Q2_K_XL): `--bench-parity` step 0 `metal 563 != cpu 14485 (cos 0.8556)`; wrong text.
- E4B QAT and HauhauCS E2B (Q2_K_P): Metal text wrong, CPU text right ("Paris").
- Fusions are not the cause: `NO_MK`/`NO_GU`/`NO_QKV` experiments left the text wrong.
- The tiny synthetic fixture (`gemma4::tiny_decode_graph`) matches the CPU, so the divergence lives in real-checkpoint features: sliding window ring, KV sharing, PLE, proportional RoPE, 12288-wide MLPs, f32-materialized TQ2_0 weights.
- Qwen3.5-2B Q4_K_XL, SmolLM2-135M pass parity on Metal.
`family_status(Gemma4)` is therefore `Gated("numerics: …")`; `--backend cpu` runs it. The loop's Gemma floors (150 tok/s) cannot be met until Gemma 4 is correct on Metal (G3/S4.3).
