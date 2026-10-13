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

## Finding: hand IQ3_S kernel table was corrupt (2026-10-12)
`gemv_matches_cpu_for_every_packed_format` (every hand and generated GEMV vs the CPU executor on random blocks) failed only for IQ3_S: relative error 0.59. Cause: `iq3s_grid` in `metal_kernels.metal` differed from `IQ3S_GRID` (`src/model/gguf_tables.rs`) from entry 57 on (455 of 512 words). The kernel table is regenerated from the Rust table; all 21 formats now agree to ≤ 7e-7. Any IQ3_S tensor (also inside the megakernel MLP, same table) was silently wrong on Metal before this fix.

## Generated GEMV tuning: rows_per_sg sweep (2026-10-12)
`generated_gemv_bandwidth` (GPU-timed wall of `Session::run`, includes ≈0.1 ms host overhead; weights all-zero) with the grid fixed to `8 × rows_per_sg` rows per threadgroup and `gemv_matches_cpu_for_every_packed_format` green at each value. Within the ±15% run-to-run noise on this machine, `rows_per_sg` 1, 2 and 4 are the same (TQ2_0 262144×1536 70–90 GB/s, Q6_K 100–140, Q5_K_M 14336×4096 94–102): no change kept, default stays 2. An earlier sweep that appeared to give 1.4–2× was invalid: the host dispatched `rows/16` threadgroups regardless of `rows_per_sg`, so `rows_per_sg = 1` computed half the rows (decode printed garbage). Lesson recorded: a bandwidth test that does not check its output cannot be trusted; the host grid for generated GEMV now comes from `codegen::gemv_rows_per_tg`.
The decode-time profile (`GIBBS_PROFILE_KERNELS=1`, per-kind summary) puts 232 packed GEMVs at ≈ 90 GB/s (35% of the ceiling) and 212 RmsNorm + 128 elementwise + 50 Rope steps (each a separate dispatch, < 3 GB/s) at ≈ 25% of step time: the remaining decode gap is dispatch count and small-kernel latency (S5 horizontal/prologue fusion), not codec throughput.
