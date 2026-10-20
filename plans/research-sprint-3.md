# Research Sprint 3: Luminal compiler techniques and the Wave 12 decode gaps

## Goal
Implement every Luminal technique Gibbs does not use yet, and every TODO from the Wave 12 deep dive. The five "highest-ROI" items come first (Phase 1). Phase 0 gives them their go/no-go numbers; it is a prerequisite, not new scope. Every change must be generic (graph-IR patterns and codec descriptors, no model names), must pass `scripts/loop_guard.sh` plus a multi-commit `generic_guard` over the promoted range, and is kept only through an unspliced 4-pass ABBA (`scripts/loop_verify.sh`). Every GPU job runs through `python3 ~/.cache/gibbs/central_scheduler.py run [--quiet] --label <name> -- <cmd>`.

## Status on branch `s3/final` (`sprint3` minus the removed items; GPU go-ahead run in progress)

All code is on `sprint3`. Until the GPU go-ahead, the user's GPU was in use for training, so nothing below the go-ahead section was measured. What ran before it, CPU-only:
- 106 tests that never touch a device (list: `target/sprint3/cpu-tests.txt` in the integration worktree);
- `graph::metal::tests::every_kernel_source_compiles`. It creates a device and builds pipelines without dispatching:
  - the hand library (MSL 4.0 + MPP), every library kernel's pipeline (SIMD 32, at least the dispatched threads, 1024 for the delta-rule step);
  - every generated GEMV/dequant/expert/PLE-chain variant.
  It caught a reserved MSL field name (`kernel`) in the new DeltaNet params that would have broken every Metal model at startup;
- CPU plan dumps (P0.2 tables in `autoresearch/sprint3-probes.md`);
- `generic_guard` over `main..sprint3`.

GPU contact that did happen, against the instruction:
- **Agent test runs.** Two agents ran test-name prefix filters that included Metal tests: S3Qkv once with `graph::passes::` (the two `megakernel` fusion tests), and S3Egglog once with `graph::passes::` and `model::gemma4` (about 10 s of small sessions).
- **Contaminated host profile.** Those runs wrote `~/.cache/gibbs/host/Apple-M5-Pro-25G241-4ed0d8e9b05eb2d9.json` (10:40, 156.0 GB/s measured under training load, against the 286.9 GB/s baseline). Its kernel hash matches `sprint3`, so the go-ahead bench would have reused it as the roofline ceiling. It is renamed `*.contaminated`, so the first quiet run re-measures. The tune file was last written the previous evening and is unaffected.
- **Device on exit.** `plan_dump` loads the AGX driver on exit for M4 (no dispatch).

| Item | State | Default | Per-token change in the CPU plan (dispatches) |
| :--- | :--- | :--- | :--- |
| P0.2 | done (`examples/plan_dump.rs`, `--rates`) | — | — |
| P0.1 / P0.3 / P0.4 / P0.5 | run on the go-ahead (`autoresearch/sprint3-probes.md`, GPU probes) | — | — |
| 1.1 PLE | the default path was already 3 dispatches per layer (slice view, GeGlu GEMV epilogue, RmsNorm + residual); added the 2-dispatch chain (quant codecs and BF16/F16 via the expert pseudo-codecs) | opt-in `GIBBS_PLE_FUSED=one` | M1 dump: 486 → 451 (36 `GluGemvChain`) when on |
| 1.2 q/k/v | fused over loader `Tensor::Mat` weights, same-format pairs when formats differ; batched prefill replayed the program's fusions (`8cbbd9e`) | **removed** (`af6dcb7`): ABBA neutral (−0.03 pp), up to 378 MiB on M7; the order-stable `GraphRewriter::replace` (`15da3aa`) stays | — |
| 1.3 DeltaNet | one `DeltaNetStep` per layer (conv → gates → delta rule → gated norm) | on (`GIBBS_DISABLE=deltanet`; `GIBBS_FUSION=none` also turns off epilogues, MoE and batching) | M2 −126 (18 × 7); M3 −168 inferred from layer count, not dumped |
| M2 BF16 fix (found by P0.2) | DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj` stay packed when the Qwen3.5 transform is an identity | on (`GIBBS_DISABLE=qwen_packed`) | about −467 MB/token streamed on M2; +18 dispatches from FP16 a/b no longer pairing |
| 1.4 f16 KV | port of `v2-40`; split-K, flash and kv-append kernels store half, accumulate f32 | opt-in `GIBBS_KV_DTYPE=f16` since `3929384`: its ABBA lost 3.82 pp (E2B ctx4k −24%) | 0 dispatches; KV bytes halved |
| Phase 2 views | row-strided slice views for stride-taking readers (`RmsNormBatch`, `RopeBatch`, `Unary`), destination-passing concat, alias-root liveness | **removed** (`e3ce231`): ABBA neutral (+0.01 pp) | — |
| Phase 3 e-graph | egglog 3.0; graph fusions and every planner fusion are rules; greedy matchers, destructive passes and the `Pattern` matcher deleted; `pair_small_hand_gemvs` stays a post-pass (it needs tuned variants) | on (the only planner) | 0 by construction: plan dumps byte-identical before/after on rows 0/1/2/4/7 and 12 synthetic graphs |
| Phase 4 search | per-layer-pattern flips of e-graph choices, timed with the tuner's gating, cached in the tune file, 2 s budget | opt-in `GIBBS_PLAN_SEARCH=1` | — |
| Phase 5 | the PLE chain (above); the DeltaNet step runs one threadgroup per key head (the scoped persistent form) | as above | — |
| 6.1 | `gen64` 64-thread K-quant GEMV as a tuner candidate (Q4_K/Q5_K/Q6_K) | tuner decides; `shape_rates`: no shape gains > 1.5% | — |
| 6.2 | `kquant_fused_dot_ablation` and the TQ2_0 ablation rotate ≥ 64 MB; re-measured: Q4_K fused +1.9–2.5% (5/5), Q6_K not | Q4_K fused form on `s3/q4k-fused-dot`, ABBA pending | — |
| 6.3 | IQ2_XXS expert GEMV grid in threadgroup memory or a bound constant buffer | opt-in `GIBBS_IQ2_GRID=tg\|buf` | — |
| 6.4 | DeltaNet state in f16/bf16 | opt-in `GIBBS_DELTANET_STATE=f16\|bf16` | — |
| 6.5 | two-pass GPU argmax, one-u32 readback for greedy decode at vocab ≥ 150k | opt-in `GIBBS_GPU_ARGMAX=1` | — |
| 6.6 | L13 / E3 verification is ABBA only | — | — |
| 6.7 | `MTLResidencySet` not pursued (llama.cpp `NO_RESIDENCY` showed no effect) | — | — |
| Phase 7 | catalog status/sprint columns, this table, ledger rows `s3-*` | — | — |

**Premises this plan got wrong, found while implementing:**
- "M1 already fuses QKV": no suite model fused q/k/v. `fuse_qkv_pass` rejected every `Tensor::Mat`, which every model builder supplies.
- "M7 mixes q/k/v formats": every M7 layer shares one format. M1, M2 and M3 are the mixed ones.
- PLE was already fused to 3 dispatches per layer; only the opt-in 2-dispatch chain is new.
- M2's real gap included 467 MB/token of needlessly dequantized DeltaNet projections: about 1.80 GB streamed against the 1.33 GB the roofline counts.

**Costs introduced:**
- **Planning time.** About +6 ms per uncached `prepare_plan` (egglog saturation floor).
  - Greedy decode plans once and never re-plans.
  - Gemma prefill's fixed 64-row chunks hit a 4-entry plan cache.
  - Scheduler prefill now also goes through that cache, but its chunk sizes follow running averages and rarely repeat, so most chunks still pay it. The prefill timer includes planning, which shrinks the next budget. Rounding budgets to fixed sizes would fix this; not done, because it changes scheduler behavior.
- **Load time.** Graph-level e-graph compile is 6–38 ms per model. The post-extraction weight rewrite adds 38–44 ms on M0/M1/M2 and 365 ms on M7.
- **Memory.** Up to 378 MiB more resident on M7 for the concatenated q/k/v copies (worst case, if the original file pages stay resident).
- **DeltaNet parallelism (unmeasured risk).** The fused step dispatches one 1024-thread group per key head: 16 groups, against 64 (M2) / 128 (M3) for the unfused delta rule. On M3 each group walks two value heads in sequence. The state traffic may lose more than the saved dispatches; `GIBBS_DISABLE=deltanet` isolates it.
- **Plan-search memory.** `GIBBS_PLAN_SEARCH=1` clones the graph and weights, and re-concatenates q/k/v per candidate. That is cheap for mmap-backed weights but costly where the loader dequantized (M3). Do not enable it there before candidates share one weight set.
- **Tuner margin.** `tune::beats` now needs a win strictly above the margin. That also applies to GEMV variant choice (ties keep the incumbent).

**GPU go-ahead (2026-10-19), done:**
1. The host profile was re-measured quietly (292.7 GB/s); tune entries keyed by the current kernel hashes were moved aside to re-tune quietly. Test runs restore the tune file afterwards (`cargo test` persists the profile without the foreign-load check and its parallel tests tune under each other's load).
2. M0 failed to load (`missing weight 'layers.0.attn.q.weight'`): batched prefill planned an unfused graph against the q/k/v-fused weights. Fixed in `8cbbd9e` (`fusion::replay_qkv`, CPU regression test).
3. `cargo test --lib`: 872 passed, 0 failed (also after the f32 KV default); the guard's integration tests pass; `--bench-parity` 5/5.
4. P0.1, P0.3, P0.4 (Wave 11 neutral, +0.03 pp), P0.5, 6.1 and 6.2 measured.
5. ABBA, clean: `main` vs `sprint3` −2.13 pp, all of it f16 KV (−3.82 pp alone, now opt-in); views +0.01 and q/k/v over `Mat` −0.03 (neutral, both removed on the user's call: branch `s3/final`).
6. Opt-in candidates prepared for ABBA: 6.2 Q4_K `fused_dot` (`s3/q4k-fused-dot`), 6.6 L13 as `GIBBS_SPECIALIZE=on` (`s3/l13-specialize`, a port of `aeb50b6`); E3 is the existing `GIBBS_EXEC_ORDER_WEIGHTS=1`.

**Still pending (each needs a quiet GPU; a foreign graphics workload at ~45% contended the `deltanet` and `qwen_packed` runs):**
1. Rerun the `deltanet` and `qwen_packed` ABBAs against `3929384`.
2. Each opt-in knob alone (`GIBBS_PLE_FUSED=one`, `GIBBS_DELTANET_STATE=f16|bf16`, `GIBBS_IQ2_GRID=tg|buf`, `GIBBS_GPU_ARGMAX=1`, `GIBBS_PLAN_SEARCH=1`, `GIBBS_EXEC_ORDER_WEIGHTS=1`), then `GIBBS_SPECIALIZE=on` and the 6.2 Q4_K candidate (their branches need `every_kernel_source_compiles` and their GPU parity tests first).
3. `s3/final`: `every_kernel_source_compiles`, full `cargo test --lib`, `--bench-parity` (the removals changed MSL and the planner).
4. 6.1: the tuner's `gen64` picks on M1/M2's Q4_K/Q5_K/Q6_K shapes, read from the tune file after a quiet re-tune (only `shape_rates` ran).
5. Real-weights parity for M1, M3 and M7, which `--bench-parity` does not cover (its suite marks 5 reference rows).
6. `main` vs `s3/final`; merge to `main` only through the keep gate.

## Baseline (`main` @ `5de3057`, README rows = `bg_232` cand1 pass)

| Model | tok/s | % roofline | dispatches | GPU ms | weights ms at ceiling | GPU ms not streaming weights | llama.cpp tok/s (2026-10-13) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 Gemma 4 E2B QAT | 150.1 | 43.8% | 506 | 6.36 | 2.92 | 3.44 | 150.7 |
| M1 Gemma 4 E2B Q4_K_M | 107.4 | 55.2% | 509 | 8.92 | 5.14 | 3.78 | 121.8 |
| M2 Qwen3.5 2B | 113.7 | 52.7% | 412 | 8.51 | 4.63 | 3.88 | 148.3 |
| M3 Qwen3.5 9B IQ2_M | 41.3 | 64.7% | 570 | 23.81 | 15.66 | 8.15 | 43.2 |
| M4 Llama 3.2 1B Q4_0 | 254.0 | 67.7% | 147 | 3.65 | 2.67 | 0.98 | 269.0 |
| M5 SmolLM2 135M | 468.1 | 43.9% | 273 | 1.90 | 0.94 | 0.96 | — |
| M6 Llama 3.2 1B bnb | 224.8 | 60.0% | 147 | 4.13 | 2.67 | 1.46 | — |
| M7 Gemma 4 26B-A4B | 72.2 | 54.9% | 658 | 13.48 | 7.60 | 5.88 | — |
| M8 E2B QAT 4k ctx | 125.1 | 36.5% | 506 | 7.70 | 2.92 | 4.78 | — |

"Weights ms at ceiling" = `weights_ms` (bytes per token ÷ 286.9 GB/s); the next column is GPU time spent on anything else (slow GEMVs, attention, norms, dispatch gaps). That column is the target.

**Arithmetic that bounds every dispatch-count idea.** Iteration 5 removed 91 dispatches for ~1.2 µs each; row 14 measured a ~1 µs gap between dispatches. Removing 35–105 dispatches per token is therefore worth roughly 40–130 µs (0.6–2% on M0). Larger gains must come from in-kernel time, which Phase 0 measures per step kind.

## What the deep dive overstated (corrections from the ledger)
- **E1 persistent megakernel was tried** (`roofline-loop.tsv` row 14, branch `experiment/e1-persistent-chain` `e4bc460`): 124.4 → 116.5 tok/s. Grid barriers cost ~9%; the barrier-free ceiling was +1.8%.
- **M1 already fuses Q/K/V** (`fuse_qkv_pass`; its 509 dispatches did not change under W11a). Mixed q/k/v formats are confirmed only on M7. A cross-codec q/k/v GEMV (F3, row 29) was already discarded.
- **Logits readback costs 11.4 µs** (`bg_230`). Host argmax is vectorized (row 17). GPU top-K measured 52–104 µs GPU vs ~120 µs CPU (`v2-33-blk`).
- **f16 KV was implemented once** (`v2-40`) and never measured (cherry-pick conflict).
- **Several PLE-adjacent changes were already discarded:**
  - quarter-block GEMV for cols < 512 (`rsi-21` discard, `rsi-23` inconclusive, `fd58c71` never landed);
  - `HandSplit` (64–68 GB/s on 256-col shapes);
  - GLU epilogue F5 (row 24);
  - RmsNorm prologue F2 (row 28).
- **Qwen3.5 partial fusions were discarded** (`rsi-15`, `rsi-18`: DeltaRule beta/log-decay, Conv1d+SiLU). The delta-rule kernel itself is iteration 11 (kept).
- **On-device fusion search F8** (row 33) was discarded without ever being verified on its own. Per-weight tuned GEMV variants live in `tune.rs` (kept in the `rsi-12` stack).
- **Untried:** egglog equality saturation, composed (strided) views, destination-passing concat, layer-pattern plan search.

## Luminal catalog status (`autoresearch/luminal-catalog.md` + ledger)

| Cand | Idea | Status | Ledger |
| :--- | :--- | :--- | :--- |
| L1 | GPU TopK, host sampling over candidates | measured per block, unverified | `v2-33-blk`; host argmax kept (17) |
| L2 | static (buffer, offset) binding | kept | `PlanValueLoc` (`rsi-43`) |
| L3 | x staged in threadgroup memory | discarded | `v2-34`, `rsi-3`, `rsi-36` |
| L4 | rows per simdgroup | kept as tuner variants `Gen(1,2,4)` | 1, `rsi-37`, `rsi-41` |
| L5 | planar repack | kept for TQ2_0, discarded for Q3_K/Q5_K | 26, `v2-41-blk` |
| L6 | RmsNorm prologue into GEMV | discarded | 28 |
| L7 | elementwise chains, Add→RmsNorm | kept (F4 discarded) | 5, `rsi-13`, `rsi-14`; 30 |
| L8 | norm→RoPE, RoPE into kv_append | kept (F6 discarded) | `rsi-13`; 22, 23 |
| L9 | multi-weight sibling GEMV | pair kept, multi discarded | `rsi-38`, `rsi-43`; 29 |
| L10 | paired-row GLU epilogue | discarded | 24, `rsi-17` |
| L11 | GPU embedding / PLE gather | not logged | — |
| L12 | decode-ahead command buffer | discarded; chunked CBs kept | `v2-35-blk`; 21, `rsi-43` |
| L13 | function-constant specialization | measured per block, unverified | `v2-38-blk` |
| L14 | split-K flash decode | kept | 2, 8, 9 |
| L15 | f16 KV cache | implemented, never measured | `v2-40` |
| L16 | DFS order + liveness arena | in use (`allocate`, `src/graph/metal.rs`) | — |
| L17 | `MTLResidencySet` | not tried; llama.cpp `NO_RESIDENCY` showed no effect | notes, iteration 6 |
| L18 | search with on-device timing | per-GEMV tuner kept; fusion search F8 discarded; e-graph not tried | `rsi-12`; 33 |
| E1 | persistent per-layer kernel | discarded | 14 |
| E2 | ICB replay | discarded | `v2-39-blk` |
| E3 | weights in execution order | measured per block, unverified (`GIBBS_EXEC_ORDER_WEIGHTS`) | `v2-36-blk` |
| E4 | combine near-misses | discarded | 25 |

## Phase 0 — Probes (no code change; queued for a free GPU)
| Probe | Command | Answers |
| :--- | :--- | :--- |
| P0.1 kernel profile | `GIBBS_PROFILE_KERNELS=1 GIBBS_PROFILE_ROWS=40 target/release/gibbs --bench <one-model suite>` for M0, M1, M2, M7, M8 | µs per step kind and the 40 slowest steps: PLE GEMVs (M0/M1/M8), DeltaNet chain (M2), attention (M8), K-quant GEMVs (M1) |
| P0.2 plan dump | `GIBBS_DUMP_PLAN=1` on M1, M2, M7 (and M0 for PLE) | separate q/k/v GEMVs per layer; PLE and DeltaNet step counts; materialized Slice/Concat steps |
| P0.3 llama-bench refresh | via `central_scheduler.py run --quiet`: `llama-bench -m <same GGUF> -p 0 -n 128 -r 3` for M0–M4 and M7; M8 adds `-d 5777` (the ctx4k prompt is 5777 tokens; confirm `-d/--n-depth` in `llama-bench --help`). M5 (safetensors) and M6 (bnb) stay n/a: no same-file llama.cpp run exists | refresh `autoresearch/bench-reference.json` `same_machine` (date, llama.cpp and ggml versions), regenerate the README |
| P0.4 Wave 11 ABBA | `scripts/loop_verify.sh`, `5de3057` vs `4ff5e60`, quiet GPU | closes `rsi-44-w11` (currently merged, ABBA inconclusive) |
| P0.5 same-model reference | new suite `bench/reference-7b.json` (Llama 2 7B Q4_0 and Q8_0, `ggml-org/Llama-2-7B-GGUF`); `bench/loop.json` stays frozen | puts Gibbs rows next to the published llama.cpp rows in README panel B |

Output: `autoresearch/sprint3-probes.md` with the raw tables. Each Phase 1 item records its go/no-go there before code is written.

## Phase 1 — Wave 12 priorities (in order)

### 1.1 Fused PLE subgraph (M0, M1, M8; M7 if P0.2 finds the pattern)
- **Pattern (graph IR, generic; confirm the exact op chain in the builder and in P0.2 before coding):** `Linear([1,h]→[1,p])` → GELU → `Mul` with a contiguous `Slice` of a per-token side input `[1, L·p]` → `Linear([1,p]→[1,h])` → `RmsNorm` → `Add(residual)`, with `p ≤ 512`. Builder: `src/model/gemma4/text.rs` (the `ple.*` nodes around lines 1037–1050).
- **Change:**
  1. Gate GEMV with a `GeluMulSide` epilogue that reads the side-input slice by offset (the slice becomes a Phase 2 view). The epilogue is a separate MSL helper outside the `GEMV-COMMON` markers, so `variants_hash` and the warm `TuneCache` keep their keys.
  2. Proj GEMV feeding the existing `rmsnorm_residual_f32` fusion.
  3. The target is 2–3 dispatches per layer instead of today's count, measured by P0.2.
  4. A one-dispatch form needs either a grid barrier (the E1 failure mode) or one threadgroup streaming both PLE matrices per layer (check: if they are Q4_0 256×1536 as the `256|1536` / `1536|256` tune entries suggest, ~442 KB) on a single core of the 16. It goes to Phase 5 only if P0.1 shows the 256-column GEMVs are the cost.
- **Read first:**
  - `git show fd58c71` (`gemv_q4_0_quarter`);
  - the `rsi-21` and `rsi-23` rows;
  - the `HandSplit` tune entries for `256|1536` / `1536|256`;
  - row 24 (F5).
- **Hypothesis:** −35 dispatches per token on E2B (≈ 40 µs) plus whatever P0.1 attributes to the 256-column GEMVs.
- **Tests:**
  - fixture parity vs the CPU graph (`gemma4` decode fixture);
  - plan test that the pattern lowers to the fused steps;
  - ABBA on the full suite.

### 1.2 Mixed-format Q/K/V projections (scope set by P0.2)
- **Go condition:** P0.2 finds ≥ 30 separable q/k/v GEMV dispatches per token on some suite model. Known so far: M7 (mixed formats). M1 is already fused. M2 is unknown; P0.2 also records how many of its layers are attention vs DeltaNet.
- **Change (option a):** in `fuse_qkv_pass` (`src/graph/passes/fusion.rs`), fuse the same-format pair (q,k) when v differs.
- **Change (option b, only if a is insufficient):** a `GemvPackedMulti` with `x` read once into threadgroup memory. This is the piece F3 (row 29) lacked; note that L3 x-staging lost on single GEMVs.
- **Hypothesis:** ≤ 36 µs/token on M7 (0.3%). Drop the item if P0.1 shows the separate GEMVs at the same GB/s as fused ones.

### 1.3 DeltaNet per-layer recurrent chain (M2, M3)
- **Go condition:** P0.1 on M2 attributes ≥ 1 ms/token to the post-projection DeltaNet steps. Kind totals: `CausalConv1dCached`, `DeltaRule`, gating, norms.
- **Change:** one generated kernel per DeltaNet layer for conv1d step + SiLU → q/k L2 norm → β/α gates → delta-rule state update → gated RmsNorm, one threadgroup per head.
  - The recurrent state stays in device memory. Check its size per head from the model config (at 128×128 f32 it would be 64 KB, far over the 8 KB threadgroup-memory occupancy limit).
  - The fusion removes intermediate vectors and dispatches only. It differs from `rsi-15`/`rsi-18`, which fused 1–2 ops each.
- **Reference:**
  - ggml-metal `gated_delta_net` (`/Users/keas/code/references/llama.cpp`), with llama.cpp at 148.3 tok/s on M2;
  - iteration 11 (kept kernel).
- **Tests:** `qwen3_5` fixture parity (state after N steps, cosine ≥ 0.99999); ABBA on M2/M3 with controls.

### 1.4 f16 KV cache (L15)
- **Start from:** the `v2-40` commit (`git log --all --grep "f16 attention KV"`).
- **Change:**
  - a graph-level KV dtype option set by the loader policy (not per model);
  - `g.state(.., DType::F16)` for `kv.*.k`/`kv.*.v` (today `F32`, e.g. `src/model/gemma4/text.rs:869-881`);
  - attention and kv-append kernels read and write half and accumulate in f32.
- **Metric note:** `bench.rs` `bytes_per_token` counts weights only, so the gain shows as tok/s; the roofline denominator does not change.
- **Go condition:** P0.1 on M8 attributes the M8−M0 GPU gap (7.70 − 6.36 = 1.34 ms) mostly to attention.
- **Tests:**
  - `split_decode_attention_matches_cpu` with f16 state;
  - `--bench-parity` (min cosine ≥ 0.9999) on the 5 parity models;
  - ABBA with M8 as the target and the others as controls.

### 1.5 Luminal views and search
Phases 2–4 below; ordered after 1.1–1.4 because each depends on Phase 0 data and on the Phase 2 IR change.

## Phase 2 — Zero-copy views (Luminal composed layouts)
- **Today:** `aliases` in `src/graph/metal.rs` give offset aliasing, and `slice_half` (`metal.rs:6031-6071`) only recognizes last-axis halves feeding GLU. Other `Slice` steps are planned as steps (`Op::Slice` in the Metal op list, `metal.rs:340`); P0.2 counts how many `Slice`, `Concat` and `Transpose` steps each suite plan actually materializes.
- **2.1** A 2-D affine view `{base, offset, row_stride}` in `PlanValueLoc` for `Slice` on any axis and contiguous `Reshape`. Activation readers (GEMV `x`, elementwise, norm, attention q) take a stride parameter. Luminal: `crates/luminal_metal/src/kernels.rs:515-545` (`layout_read_index`), `layouts.rs`.
- **2.2** Destination-passing `Concat`: producers write into their slot of the concat output. Luminal: `ToDps` / `alias_info` `Sharing::Must`, e.g. `ops/mul_reduce_sum/mod.rs:72-80`.
- **2.3** Liveness in `allocate` follows the view root.
- **Tests:** plan tests that a Slice→GEMV and a Concat of producers emit no copy step; parity on all fixtures; ABBA. P0.2's count of materialized steps sets the expected effect.

## Phase 3 — Equality saturation (egglog) over the step IR
- **Why:** fusion today is a fixed order of destructive passes (`src/graph/passes/mod.rs:17-29`) plus greedy matches in `metal.rs` (`plan`, `batch_horizontal`, `fuse_moe_subgraphs`). W11a had to add an explicit commutation step in `batch_horizontal` before interleaved q/k/v norms could batch; an e-graph keeps such orderings as alternatives instead.
- **3.1** Add the `egglog` crate, pinned to the version luminal main uses. This is the sprint's only new dependency.
- **3.2** Encode the post-Phase-2 step IR as egglog datatypes (GemvPacked, RmsNorm, Rope, Add, Mul, Gelu/SiLU, View, CachedAttention, MoE ops, DeltaRule).
- **3.3** Port every current fusion as a non-destructive rewrite:
  - MegakernelMlp, gate/up, QKV;
  - RmsNorm+Add, RmsNormBatch/RopeBatch with commutation of independent steps;
  - GemvPackedPair;
  - MoE TopKSoftmax/SplitGlu/WeightedSum;
  - Rope→attention;
  - plus the Phase 1 fusions.
- **3.4** Extraction cost = bytes ÷ ceiling + measured per-dispatch µs + per-kind rates (TuneCache / `shape_rates`). Validate it against the P0.1 profiles (per-kind error ≤ 15%) before trusting it. F7 (row 32) was never validated on its own.
- **3.5** Cutover:
  1. A plan-equality test: on all 9 suite models, the e-graph extraction reproduces today's plans.
  2. Then delete the destructive passes and the greedy matchers.
  3. Then enable the new rules.
- **Budget:** a bounded match budget like luminal's `AlgebraBudget` (`crates/luminal_metal/src/saturation.rs:19-46`), with ≤ 200 ms compile time per model graph, measured.

## Phase 4 — On-device plan search (L18)
- **Luminal:** `crates/luminal_metal/src/search.rs` (genome sampling, `profile_candidate_at`, 8 generations × 8, 3 trials, 4 finalists).
- **4.1** Search unit: one repeated layer pattern (structural hash of a decoder-layer subgraph). Candidates are e-graph extractions that differ in fusion choices and GEMV variants.
- **4.2** Timing reuses `tune::decide`: two opposite-order rounds, `QUIET_MARGIN`, `foreign_busy`. One candidate layer runs N times in one command buffer with GPU timestamps.
- **4.3** Cache by (pattern hash, kernel-source hash, device) in the tune file; never persist under foreign load.
- **4.4** Budget: ≤ 2 s of first-load search per model, with cost-model extraction as the fallback.
- **Acceptance:** per model, the searched plan is not slower than the cost-model plan in ABBA.

## Phase 5 — Scoped persistent kernels (E1, revisited)
- **Changed since row 14:** E2B dispatches 707 → 506; 2-chunk command buffers. Unchanged: barriers cost ~9% and the dispatch gap is ~1 µs.
- **Scope:** only chains that need no grid barrier and whose intermediates fit one threadgroup, such as the DeltaNet per-head chain (1.3) and a one-dispatch PLE (1.1), when P0.1 shows their GEMVs are not bandwidth-limited at one threadgroup.
- **Stop rule:** if the fused forms in 1.1/1.3 do not beat their multi-dispatch forms in ABBA, record E1 as closed.

## Phase 6 — Remaining deep-dive TODOs
- **6.1** K-quant GEMV with 64-thread threadgroups (2 simdgroups × 4 rows, as in luminal 0.2 `quantized.rs:55-58`; check ggml-metal's `kernel_mul_mv_q4_K_f32` launch shape in `/Users/keas/code/references/llama.cpp` before porting) as a tuner variant for Q4_K/Q5_K/Q6_K (M1, M2). Confirm the current generated GEMV threadgroup size in `src/graph/codegen.rs` first. The TQ2_0 64-thread port was discarded (row 15); K-quants were not tried. Measure with `shape_rates` rotating ≥ 64 MB of weights.
- **6.2** Re-measure `WIDE_Q4_K` `fused_dot` with ≥ 64 MB rotation; the `bg_231` legs were SLC-resident (Q6_K read 414.5 GB/s > ceiling). Keep Q6_K unfused.
- **6.3** IQ2_XXS grid lookup: `uint` threadgroup staging or a constant-buffer binding (W11c), with the 7-round `moe_gemv_bandwidth` harness.
- **6.4** DeltaNet state in bf16/f16 after 1.3; a parity study first.
- **6.5** GPU argmax/top-K (L1), last: only for vocabularies ≥ 150k, and only if P0.1 shows host argmax > 50 µs/token.
- **6.6** Finish ABBA verification of function-constant specialization (L13, `v2-38-blk`) and execution-order weights (E3, `v2-36-blk`).
- **6.7** `MTLResidencySet` (L17): record as not pursued (llama.cpp `NO_RESIDENCY` showed no effect).

## Phase 7 — Bookkeeping
- **7.1** Add the status column above to `autoresearch/luminal-catalog.md`.
- **7.2** One `roofline-loop.tsv` row per probe and per ABBA.
- **7.3** README numbers come only from `gibbs --bench` runs written to `autoresearch/bench-readme.jsonl` + `.meta.json` and from `autoresearch/bench-reference.json`, regenerated with `cargo run --release --example gibbs_art`.

## Verification per item
1. Fixture parity test for the new step (CPU graph vs Metal).
2. `scripts/loop_guard.sh` rc = 0 and `parity: 5 compared`.
3. `generic_guard` regex over `git diff <base>..HEAD -U0 -- src/graph src/hardware src/scheduler`.
4. Unspliced 4-pass ABBA: `delta_pp ≥ max(0.30, noise_pp)`, `ratios_ok`, `max base_spread ≤ 0.04`.
5. A ledger row, then promote and update `target/loop-base`.
