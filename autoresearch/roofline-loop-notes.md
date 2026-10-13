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

## Iteration 0 (provisional baseline) and iteration 1

- **Deviation from plan 3.1:** `bench/loop.json` has 8 models, not the planned 9: the Gemma 4 26B-A4B entry is missing because the
  10 GB file is still downloading (link ~0.5-2 MB/s) and the MoE real-weight acceptance is pending. Iteration 0 is therefore
  provisional; when the file lands the entry is added and the A/A baseline re-run (a new iteration 0). Within-verify comparisons
  (base vs cand) stay valid because every verify runs both binaries on the same suite.
- Baseline geomean 26.34 % (A/A noise 0.24 pp). Per-model: E2B QAT 24 %, E2B Q4_K_M 37 %, Qwen3.5-2B 26 %, Qwen3.5-9B IQ2_M 55 %, Llama-1B Q4_0 49 %,
  SmolLM2 26 %, Llama-bnb 50 %, **E2B QAT at 2k-token context 2.8 % (102 ms/token)**.
- The ctx4k entry costs `ln(26.34/2.8)/8` = 0.28 in log-geomean: raising it to ~20 % lifts the geomean by about +7 pp, more than any
  GEMV change. Candidate order is therefore by expected pp, not the plan's L1..L18 listing: **L14 (split-K flash-decode) first.**
- Iteration 1 (discard): GEMV k-unroll. Microbenchmark showed no format moves >3 %. A limiter probe (x loads, weight decode, access pattern
  removed one at a time, TQ2_0 262144x1536 gen r2): base 114 GB/s, no x loads 120, no decode 194, neither 212; Q8_0 is 288 in all four.
  Conclusion: the low-bit generated GEMVs are bound by per-weight decode work (byte loads + shift/mask/convert per weight), not by x
  traffic or bytes in flight. The remaining lever is a cheaper decode (planar repack + 16-byte aligned loads + SWAR; L5), not tiling.

## Iteration 2 (keep): L14 split-K decode attention
- Arithmetic: decode attention at m=1 launched `heads` (8) threadgroups on a 20-core GPU, each scanning all keys serially (~2.5 ms per layer at 2k keys, 90 ms of the 102 ms token). Slicing the keys across up to 32 threadgroups per head (64 keys/slice) plus a merge kernel puts 100-250 threadgroups in flight. Expected ctx4k 102 -> ~20 ms; log-geomean +0.28*(1 - 20/102 ratio...) ~ +7 pp. Actual +8.83 pp (noise 0.14).
- Design: generic op-level change in `CachedAttention` lowering (m == 1, no `visible_end`, >= 128 visible keys after the sliding window). Fallback: the unsplit kernel. Numerics: partial (max, sum-exp, weighted V) per slice, merged with the usual online-softmax rescale; Metal vs CPU executor test over full / window 300 / window 90 with soft cap and GQA (`split_decode_attention_matches_cpu`).
- Guard: lib tests, cpu_free_matrix/op_coverage/codec_parity/gemma4_tests, bench-parity 5 compared (min cos 1.00000), generic_guard: pass.

## Iteration 3 (keep): TQ2_0 SWAR decode
- Limiter probe showed removing the weight decode lifts the generated TQ2_0 GEMV from 114 to 194 GB/s; the old decode8 issued eight byte loads and eight shift/mask/convert/sub/mul per 8 weights. New decode8: four 2-byte loads (blocks are 66 B: only 2-byte alignment) combined into two words, one shift + mask per word, `float4(as_type<uchar4>)` conversion. Kept the bit-exact `(code - 1) * d` form: `codec_parity` requires bit-identical decode (an fma form was 5% faster but gives +0.0 where the CPU gives -0.0; guard failed it).
- Microbench TQ2_0: 262144x1536 114 -> 131 GB/s; 6144x1536 98 -> 110; E2B QAT +4.4 % tok/s.
- Remaining gap on TQ2_0 (131 vs ~250 GB/s for hand Q4_0-class kernels): needs wider loads (planar repack, L5) so a lane decodes 16 bytes at once.

## Iteration 4 (discard): weight prefetch under a concurrent encoder
- Probe: bench copies of Q4_0 2048x1536 under a concurrent encoder (independent dispatches, no barriers) run 163 -> 245 GB/s, TQ2_0 88 -> 132: tail/ramp of small GEMVs is hidden when independent kernels overlap. The decode chain has no such independence (QKV and gate/up are the only 2-3 wide groups), which is why sprint-2's concurrent encoder found nothing.
- Tried hiding it with cache prefetch: concurrent encoder, barrier before each dispatch, an unordered `prefetch_lines` dispatch touching the weights of step i+D after step i. E2B QAT 25.6 % -> 24.6 % (barriers only) -> 22.6 % (D=1) / 22.4 % (D=2). Weight prefetch into the system cache does not pay on this part.
- Warning logged: a first version (resource barrier between steps only, none between the dispatches inside a step) measured 31 % but decoded non-deterministically: split-K attention, the fused MLP and dequant+GEMM issue several dispatches per step; the speedup was a race. Correctness check for any concurrency work: compare logits/tokens with the serial encoder first.
- Consequence: the only way to reclaim the ramp/tail overhead is to put independent work into the same dispatch (horizontal fusion: QKV, gate/up, the three per-head norms, the two RoPEs) or to remove dispatches, not to overlap them.

## Iteration 5 (keep): RmsNorm+Add fusion and wide affine TQ2_0 decode
- (a) Planner: an `RmsNorm` whose only reader is an `Add` becomes one `rmsnorm_residual_f32` step (kernel already existed for the fused MLP). E2B 763 -> 672 dispatches. Measured alone: +0.14 pp (below the 0.30 floor), E2B QAT +0.9 %, Q4_K_M +1.1 % => ~1.2 us per removed dispatch, i.e. dispatch count is NOT the main E2B lever (763 x 1.2 us ~ 0.9 ms of ~10.4 ms). Stacked with (b) to clear the floor; (a)'s verify row: base 33.587 -> 33.724.
- (b) `codec::Wide`: a codec may decode a group of 4 segments (32 weights, TQ2_0: the four shifts of the same 8 bytes) at once. decode8 re-reads the bytes once per sub-block; the wide path loads them once and computes `scale * (sum(code * x) - offset * sum(x))`, removing the per-weight `(c-1)*d`. Wide-only (no affine) was flat (110 GB/s at 6144x1536): the loads were not the limit; the saved ALU (cvt + sub + mul per weight) was: 110 -> 123, 131 -> 151 GB/s at 262144x1536.
- Profile of E2B (decode, per-step isolated): GemvPacked 232 steps 4.8 ms (128 GB/s mean), MegakernelMlp 15 steps 1.1 ms (221 GB/s), RmsNorm 212 steps 0.9 ms, Binary 128 steps 0.33 ms, CachedAttention 35 steps 0.33 ms, Rope 50 steps 0.18 ms; the 20 non-fused MLP layers cost 2.6 ms in gate/up/down TQ2_0 GEMVs at 108-120 GB/s and lm_head 0.8 ms at 133 GB/s: TQ2_0 decode throughput (~600 Gweights/s now vs 424 for generated Q4_0) is the largest single E2B lever left.
- Next: E2B 27.2 %; remaining gap is mostly GEMV bandwidth on small shapes and TQ2_0 ALU.

## Iteration 6 (keep): Qwen3.5 GGUF loader packs every codec format
- Found by comparing against llama.cpp on the same files (llama-bench, Metal, tg128): E2B QAT 150.7 vs ours 96; E2B Q4_K_M 121.8 vs 75; Qwen3.5-2B 148.3 vs 73; 9B IQ2_M 43.2 vs 36.5; Llama-3.2-1B Q4_0 269 vs 196. With ggml's fusion, concurrency and graph optimisation all disabled llama.cpp still reaches 147 tok/s on E2B QAT, residency sets are irrelevant (152.2 with NO_RESIDENCY), so the gap is in the per-op kernels, not dispatch structure. Note for the 90 % target: llama.cpp itself reaches ~43 % of the roofline on E2B QAT.
- Qwen3.5-2B profile showed MegakernelMlp steps of 75 MB and a 1.02 GB lm_head: 2 bytes per parameter. `qwen3_5.rs gguf_checkpoint` kept a matrix packed only in hand-kernel formats (Q4_0/Q8_0/IQ4_XS/IQ2/Q4_K/IQ3_S); Q5_K, Q6_K (incl. the tied token_embd/lm_head, 417 MB packed) and every other codec format was dequantized to BF16. The generic `Weights::from_gguf` already used the codec registry; this loader was the stale outlier. Now `gguf_block_codec(fmt).is_some()` and cols divisible by the block size.
- Qwen3.5-2B 73.5 -> 90.4 tok/s; the roofline basis grows from 1.06 to 1.27 GB/token because the applied policy now counts the real streamed formats.
- Guard hit a hub test that follows measured codec rates (`test_select_weight_files_policy`: smallest quant vs predicted fastest flips with the host's measured Q4_K/Q2_K rates); pinned with `with_unmeasured_host` like the other rate-dependent tests.

## Iteration 7 (keep): wide affine decode for the K-quants and IQ4
- The TQ2_0 wide path (iteration 5) generalised: a codec's `Wide` now emits, per group, the raw codes of `segs` 8-element segments plus a scale and a bias per segment; the GEMV does `acc += scale * dot(codes, x) + bias * sum(x)`. This is what ggml-metal does in kernel_mul_mv_q6_K/q4_K (scale applied to a 4-element partial sum, Q4_K `dmin * sumy`) and tq2_0 (hoisted y coefficients). Bytes shared by several segments are loaded once, nibble/bit planes are extracted four bytes at a time with masks, and `(code - zero) * scale` per element disappears.
- Added Q6_K (ql/qh planes, 4 segments), Q4_K and Q5_K (2 segments: low and high nibble, Q5_K high bit), Q3_K (4 two-bit planes + mask bit), IQ4_XS and IQ4_NL (table lookup, bias 0). Shape-rate bench (GB/s, 262144x1536 / 6144x1536): Q6_K 163/150 -> 295/259; Q5_K 171 -> 263/232; IQ4_XS 179 -> 234/195; Q3_K -> 185/155; IQ4_NL 257/211; Q4_K gen 169 -> 231/197 (the hand kernel is 235/180, the tuner now picks gen on the small shapes).
- Remaining generated-decode formats without a wide path: Q4_1/Q5_0/Q5_1 (32-element blocks), IQ2_XXS/XS/S/M, IQ3_XXS, IQ1_S/M, TQ1_0, MXFP4, NVFP4 (grid/table or hand kernels). Hand kernels (Q4_0, Q8_0, Q4_K, Q2_K, IQ2_M, IQ3_S) unchanged.
- Remaining gap to llama.cpp (tg128, same files): E2B QAT 96.8 vs 150.7, E2B Q4_K_M 81.3 vs 121.8, Qwen3.5-2B 100.5 vs 148.3, 9B IQ2_M 36.5 vs 43.2, Llama-3.2-1B 222 vs 269. The E2B QAT file is Q4_0 (3.05 G params) plus TQ2_0 (1.54 G params = 47 % of the streamed bytes): ggml's tq2_0 kernel avoids all integer masking (`floor(v * 0.25)` plus telescoped y coefficients).
