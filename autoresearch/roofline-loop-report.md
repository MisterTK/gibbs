# Roofline loop: terminal report (ended by user instruction, not by a stop rule)

The user ended the loop after the iteration-27 verify. Stop rule A (every model >= 90%, E2B QAT >= 150 tok/s, every matrix item ok) did not fire; rule B did not fire either (two kept iterations in the last window improved the geomean by 0.62 and 0.65 pp, so its counter reset). This is the state at HEAD with base `33187e7`.

## Final verify (iteration 26, quiet GPU: `target/loop/verify27.json`)
Geomean roofline 47.34% (base 46.69%). Candidate per model (roofline %, tok/s):

| model | roofline % | tok/s |
|---|---|---|
| Gemma 4 E2B QAT Q2_K_XL | 37.3 | 132.7 |
| Gemma 4 E2B Q4_K_M | 52.5 | 106.0 |
| Qwen3.5-2B UD-Q4_K_XL | 47.1 | 110.9 |
| Qwen 9B Heretic IQ2_M | 58.0 | 38.4 |
| Llama-3.2-1B Q4_0 | 64.4 | 250.7 |
| SmolLM2-135M | 38.6 | 427.6 |
| Llama-3.2-1B bnb-4bit | 56.6 | 220.2 |
| Gemma 4 26B-A4B IQ2_XXS | 46.7 | 63.7 |
| E2B QAT, 4k context | 33.9 | 120.8 |

Every model is below 90%; E2B QAT is 132.7 tok/s against the 150 target. The main gaps: E2B QAT (TQ2_0 GEMVs are ALU-bound at about 200 GB/s after the planar repack, host gap about 0.3 ms per token), SmolLM2 and Llama-1B (dispatch and host bound), E2B at 4k context (attention).

## Iterations this session (TSV rows 17-27)
Kept: 17 greedy argmax (+0.84), 18 F1 horizontal norm/rope batching (+0.33), 21 chunked command buffers (+0.62), 26 planar TQ2_0 layout (+0.65, E2B QAT +3.4% tok/s). Total kept across the loop: 14 keep, 12 discard rows.
Discarded: 19 K/V prefetch, 20 bundle (inconclusive under contention), 22 F6 fused attention merge (-0.63), 23 F6 append-only (+0.12), 24 F5 GLU epilogue (+0.04), 25 E4 stack of F6-append and F5 (+0.17).
Unverified (infra): 27 lane-split Q4_0 GEMV (branch `experiment/q4_0-split`; microbench 164 -> 206 GB/s on 2048x1536, E2B A/B about 0.5% inside noise; the verify never ran, GPU never quiet for 3 h).
Parked branches: `experiment/bundle-21` (F6, F7 cost model, F8 on-device fusion search, GLU epilogue, pooled logits), `experiment/f6-append`, `experiment/e4-stack`, `experiment/q4_0-split`.

## Phase 4 (fusion vs XLA)
F1 kept. F5 and F6 discarded (rows 22-25). F7 and F8 only exist on `experiment/bundle-21`, with the bundle's row 20 as their verdict. F2, F3, F4 were not tried, so stop rule B's "every F item has a TSV row" is not met either.
4.9 (measured with `scripts/fusion_vs_xla.py`): gibbs Full 14 kernels vs 16 for the in-repo XlaClass policy (0.875x, needs <= 0.6x); host wall 482.5 us vs 700 us for jax-metal (0.69x, but this compares a Rust wall to a Python per-call wall and Full vs XlaClass differ by 0.3-0.5%, i.e. noise); device-side basis: XLA 286.9 us/call pipelined vs gibbs GPU 317 us (gibbs 1.11x slower; the XLA calls were independent, so this may be a throughput figure). Verdict: NOT ACHIEVED. XLA's kernel count is unavailable (jax-metal `as_text()` is None).

## Escalations
E4 tried (row 25, discard). E1 (persistent per-layer kernel), E2 (ICB replay) and E3 (execution-order weight layout) were not tried.

## Matrix
3303 items; 14 measured, 7 ok, 0 at >= 90% (p50 18.3%, p90 28.5% in those early rows, taken at an older commit). The runner was stopped during the verifies and not restarted. Error classes: LFM2.5 per-head q/k RMSNorm not implemented (2 items, in scope), 2 `exit -15` and 2 curl failures (infrastructure; reset), 1 granite item compiled for CPU (placement gap). The matrix cannot finish at the observed download speed. Modified third-party variants (e.g. Kev-0.8B `cls.output.weight`) are classified unsupported-variant (out of scope) by provenance.

## Measurement notes
Ambient GPU load (browser and video clients) repeatedly invalidated verifies; `scripts/verify_when_quiet.sh` gates on utilization and gave valid runs for iterations 21-26. A 3 h wait for iteration 27 ended with "never quiet" (utilization 99%) and the wrapper's python step printed the previous verify.json; that stale output was not used. Plan deviations: HorizontalBatch fusion class added; E2B dispatch baseline 672 (not 707); XLA kernel count null; matrix runner paused; MegakernelMlp still uses default GEMV variants.

## RSI grade
Level 0: no fixed-budget hand-tuned baseline exists; held-out generalization is the matrix (14 of 3303 items measured). Candidates tried this session: 11 experiments plus the bundle's parts (multiple-comparisons disclosure: the keep rule uses a noise-floor gate, but 4 of 8 gains are under 0.7 pp).
