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

## S6.2 / G3.4 decode ICB: not adopted (2026-10-12)
`GIBBS_TRACE_ENCODE=1` on this tree (Apple M5 Pro): Gemma 4 E2B QAT direct encode 0.20–0.27 ms CPU vs 8.8–9.1 ms GPU per token (2.3–3.0%); SmolLM2-135M 0.09–0.10 ms vs 4.8 ms (2.1%). The ceiling for any ICB saving is therefore the encode time itself (≤ 3% per token, before ICB's own `use_resources` + `execute_commands_in_buffer` cost), which cannot reliably clear S6.2's "≥ 2% lower" bar, while ICB needs every `encode` arm rewritten for `IndirectComputeCommand`. Decision rule applied: keep direct encode. The old `create_decode_icb` only allocated a descriptor and hard-coded `dispatch_icb = true`; it and `tests/decode_icb_test.rs` (which asserted that constant) are deleted. `bench.rs` `encode_us` is a hard-coded `10.0`; G3.6 replaces it with the measured value.
Per-token wall time at the CLI (E2B 12 ms vs 8.8 ms GPU) shows the larger lever is host work outside the command buffer (256K-vocab logits readback and sampling), not encode.
