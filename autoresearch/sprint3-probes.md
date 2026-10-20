# Sprint 3 probes (Phase 0)

Raw outputs of the Phase 0 probes in `plans/research-sprint-3.md`. The GPU probes (P0.1, P0.3, P0.4, P0.5) run through `scripts/sprint3_probes.sh p01|p03|p04|p05`; their results are in [GPU probes](#gpu-probes-go-ahead-run-2026-10-19) below.

## P0.2 plan dumps (CPU, no GPU)

**Tool.** `examples/plan_dump.rs` loads a suite row the way `gibbs --bench` does:
- `bench::load_for_bench`
- the suite's re-encode policy
- the codec rates of the newest `~/.cache/gibbs/host` profile measured with the current codec sources

With `GIBBS_PLAN_DUMP_CPU=<ctx>`, `GraphProgram::compile` then runs the passes, placement and dtype adaptation. It prints `graph::metal::plan_dump` and stops before any device creation, weight upload or tuning. `plan_dump` runs, in order: `plan` (MoE subgraph fusion included), `batch_horizontal`, `allocate`, `pair_small_hand_gemvs`.

Dispatches per step follow the encoder:
- `CachedAttention` is 2 when its keys split (kv_append is fused at decode).
- `MegakernelMlp` is 6, or 5 without a post-norm.
- Every other step is 1.

Tuner-chosen GEMV variants print `variant=unassigned`.

**Verified device-free:** each run had `DYLD_PRINT_LIBRARIES=1`, and the AGX driver bundle never loaded (`grep -ci AGX` = 0).

- **Commit:** `ec7a562` (worktree `s3/probe` on `sprint3` = `9045f1c`).
- **Command:** `DYLD_PRINT_LIBRARIES=1 target/release/examples/plan_dump bench/loop.json <row> --ctx <n>` (equivalently `cargo run --release --example plan_dump -- bench/loop.json <row> --ctx <n>`).
- **ctx:** set to the median decode position of the bench row. That is prompt tokens (the bench's `Prefill: N tokens` lines in the 2026-10-18 `target/loop/*.out`) plus half of `decode_tokens`.
- **Full dumps:** `target/sprint3/plan-dump-m{0,1,2,7,8}.txt` (not committed).

### Totals vs the bench

| row | ctx | steps | dispatches (dump) | README dispatches | step → dispatch gap |
|---|---|---|---|---|---|
| M0 E2B QAT Q2_K_XL | 84 | 396 | 506 | 506 | +35 split attention (35 × 2), +75 MegakernelMlp (15 × 6) |
| M1 E2B Q4_K_M | 84 | 384 | 509 | 509 | +35 split attention, +90 MegakernelMlp (18 × 6) |
| M2 Qwen3.5-2B Q4_K_XL | 89 | 374 | 412 | 412 | +6 split attention, +32 MegakernelMlp (8 × 5, no post-norm) |
| M7 26B-A4B IQ2_XXS | 88 | 628 | 658 | 658 | +30 split attention (run because `memory_pressure` showed 69 % free right before loading) |
| M8 (= M0, ctx4k) | 5809 | 396 | 506 | 506 | same plan as M0; attention splits at every ctx ≥ 48 keys (≥ 32 at head_dim 256) |

Splitting starts at 48 keys (32 at head_dim ≥ 256). Every attention step therefore costs 2 dispatches from ctx 48 on: this is why M8 has the same count as M0. `MegakernelMlp` is one step but 5–6 dispatches: norm, gate, up, activation, down, and an optional post-norm.

**Re-encode under `max-speed`:**
- M0: 70 of 275 projection tensors are re-encoded in the candidate the bench gates by KL. The dump's weight bytes (838.9 MB, estimated as numel × bpw) match the bench's 837.7 MB/token.
- M1, M2, M7: 0 tensors re-encoded.

### Step-kind histograms

- **M0:** GemvPacked 232, RmsNorm 97, CachedAttention 35, MegakernelMlp 15, RmsNormBatch 15, Binary 1, Unary 1.
- **M1:** GemvPacked 223, RmsNorm 91, CachedAttention 35, MegakernelMlp 18, RmsNormBatch 15, Binary 1, Unary 1.
- **M2:** GemvPacked 127, Unary 60, RmsNorm 59, Binary 42, CausalConv1dCached 18, DeltaRule 18, GemvPackedPair 18, Copy 12, MegakernelMlp 8, CachedAttention 6, RmsNormBatch 6.
- **M7:** GemvPacked 236, RmsNorm 91, ExpertLinear 60, RmsNormBatch 60, Binary 30, CachedAttention 30, MoeWeightedSum 30, RmsNormNoGamma 30, SplitGlu 30, TopK 30, Unary 1.

**Steps per layer** (layer count in parentheses):
- **M0:** 10 (14 layers), 11 (layer 0), 12 (20 layers).
- **M1:** 8 (10), 10 (8), 12 (10), 14 (6), 15 (1).
- **M2:** 12 (1), 14 (7), 15 (5), 17 (11).
- **M7:** 20 (5), 21 (24), 22 (1).

**Non-layer steps:**
- **M0/M1:** `model.ple.model_proj` GEMV, `ple.proj_scaled` Binary, `ple.proj_norm` RmsNorm, `lm_head` GEMV, `logits_softcap` Unary.
- **M2:** `final_norm`, `logits`.
- **M7:** `lm_head`, `logits_softcap`.

### Separate q/k/v GEMVs (go/no-go input for 1.2)

None of the four models fuses Q/K/V. Separate q/k/v `GemvPacked` steps per token:

| row | q | k | v | total | weight dtypes in the plan (GGUF source in brackets) |
|---|---|---|---|---|---|
| M0 | 35 | 15 | 15 | 65 | all Q4_0 [Q4_0]: **same dtype, still unfused**. Layers 15–34 share KV and have only q |
| M1 | 35 | 15 | 15 | 65 | q/k Q4_K; v Q6_K in 7 and Q4_K in 8 of the 15 k/v layers [17 of 35 GGUF layers have Q6_K v]. Mixed in 7 layers |
| M2 | 6 | 6 | 6 | 18 | mixed in every attention layer: q/k Q5_K (3), Q4_K (2), IQ4_XS (1); v Q6_K (5), Q5_K (1) |
| M7 | 30 | 30 | 25 | 85 | **same dtype within every layer**: Q5_K × 3 (25 layers), Q5_K q/k with no v (4), Q6_K q/k with no v (1) |

- **M7 per-layer dtypes:** read from the dump and confirmed from the GGUF header (CPU tensor-info parse). q/k/v share a dtype in every layer. This contradicts the plan's "M7 (mixed formats)". The 5 layers without a v GEMV use k as v.
- **Fusion gap:** `fuse_qkv_pass` (`src/graph/passes/fusion.rs:203`) cannot fuse any of these triples, whatever their dtypes. The Gemma 4 builder inserts q/k/v as `Mat::to_graph_tensor()` (`src/model/gemma4/text.rs:732, 792, 838`), which returns `Tensor::Mat` (`src/model/weights.rs:46-47`). The pass returns `None` whenever q, k or v is a `Tensor::Mat` (`fusion.rs:280-285`). The Qwen builder (M2) does the same (`part.to_graph_tensor()`, `src/model/qwen3_5/text.rs:1264, 1287`), so this blocks every suite GGUF model.
- **Plan correction:** the plan's "M1 is already fused" is not what the dump shows.
- **Go condition (≥ 30 separable q/k/v dispatches on some model):** met on M0, M1 and M7.

### PLE (M0, M1, M8)

PLE accounts for 3 global steps plus 3 steps in each of the 35 layers (108 dispatches/token):
- **Global:**
  - `model.ple.model_proj`: GemvPacked `[1, 8960]`, w BF16 `[8960, 1536]`.
  - `model.ple.proj_scaled`: Binary.
  - `model.ple.proj_norm`: RmsNorm over `[1, 8960]`.
- **Per layer:**
  - `ple.gate`: GemvPacked `[1, 256]`, w `[256, 1536]`.
  - `ple.proj`: GemvPacked `[1, 1536]`, w `[1536, 256]`.
  - `ple.norm`: RmsNorm, chained with the next layer's `input_ln` (`+1 outs`). The last layer chains it with `final_norm`.
- **No separate GELU, side-input `Slice` or `Mul` step:** GELU and the side Mul are the `ple.gate` epilogue (`Epi::GeGlu`, `gelu_tanh(acc) * side[row]`, S3Ple from the code), and the slice is aliased.
- **Gate/proj dtypes:** M0 Q5_K (re-encoded from the GGUF's Q8_0 by the max-speed candidate). M1 BF16 (the GGUF stores F32).
- **1.1 target:** fusing gate → proj leaves 35 dispatches to remove; the norm already shares a step.

### DeltaNet (M2)

M2 has **18 DeltaNet layers and 6 attention layers** (layers 3, 7, 11, 15, 19, 23).

**DeltaNet layer (13 steps):**
1. `input_layernorm` RmsNorm.
2. `mixer.proj_qkv` GEMV `[1, 6144]`.
3. `mixer.proj_z` GEMV `[1, 2048]`.
4. `proj_b+proj_a` GemvPackedPair `[1, 16]` × 2.
5. `mixer.conv` CausalConv1dCached `[1, 6144]`.
6. `beta` Unary.
7. `silu` Unary.
8. `a_softplus` Unary.
9. `log_decay` Binary.
10. `delta` DeltaRule `[1, 2048]`.
11. `mixer.norm` RmsNorm (`w [128]`).
12. `gated` Binary.
13. `mixer.out` GEMV.

Then a post-norm and an MLP. The 8 post-projection steps (conv … gated) are 144 dispatches/token.

**Attention layer:**
- q GEMV `[8, 512]`, k GEMV, v GEMV.
- two `Copy` steps (q/gate slices).
- RmsNormBatch (q_norm + k_norm).
- `gate_act` Unary.
- CachedAttention (2 dispatches).
- `gated_core` Binary.
- o GEMV.

**DeltaNet projections stream as BF16:** `proj_qkv`, `proj_z` and `out` are BF16 in the plan, although the GGUF stores them as Q5_K / Q5_K / Q8_0.
- **Cause:** `src/model/qwen3_5.rs:708-712` (`needs_transform`) dequantizes every `.linear_attn.*` tensor when the key and value head counts are equal.
- **Cost:** about 755 MB/token in BF16 vs about 288 MB in the stored formats. The dump's weight-byte estimate is 1.80 GB/token, while the bench's roofline (header formats) counts 1.33 GB. M2's roofline percentage therefore understates the bytes actually streamed by about 35 %.

### Materialized Slice / Concat / Transpose / Copy steps

| row | steps | detail |
|---|---|---|
| M0, M1, M7, M8 | 0 | every movement op is aliased |
| M2 | 12 | `self_attn.q_slice` and `self_attn.gate_slice` in each of the 6 attention layers: `Copy → [8, 256]`, slice axis 1 `[0..256]` / `[256..512]` of `[8, 512]`, **not contiguous** (8 heads, q and gate interleaved per head) |

## P0.2 after the Wave A merges (CPU, no GPU)

- **Commit:** `sprint3` @ `b0f86ee`.
- **Command:** `target/release/examples/plan_dump bench/loop.json <row> --ctx <n> --rates ~/.cache/gibbs/host/Apple-M5-Pro-25G241-92719a27da496e4e.json`.
- **Why `--rates`:** gen64 and the new kernels change `codec::source_hash()`, so without it the dump falls back to the static re-encode rule. On M7 that rule re-encodes 174 of 205 tensors, which changes the plans for reasons unrelated to the sprint. `--rates` reuses the profile the baseline above was dumped with (0 re-encoded on M1/M7, 70 of 275 on M0).
- **The dump now runs `views::elide`,** as `prepare_plan` does. The baseline above did not, so its 12 M2 `Copy` steps were never planned on the device after Phase 2.

| row | baseline steps / dispatches | sprint3 steps / dispatches | Δ dispatches | from |
|---|---|---|---|---|
| M0 (ctx 84) | 396 / 506 | 366 / 476 | −30 | q/k/v fused in the 15 layers that compute their own k/v (loader `Mat` concat) |
| M1 (ctx 84) | 384 / 509 | 361 / 486 | −23 | 8 full q/k/v triples + 7 same-format pairs |
| M2 (ctx 89) | 374 / 412 | 248 / 286 | −126 | DeltaNet chain 8 → 1 step × 18 layers (−126), q/gate copies elided (−12), q/k pairs (−6), lost BF16 `proj_b+proj_a` pairing (+18; see below) |
| M4 (ctx 80) | — / 147 | 131 / 147 | 0 | no sprint fusion applies |
| M7 (ctx 88) | 628 / 658 | 578 / 608 | −50 | q/k/v fused in the 25 layers with a v projection |

- **M2 bytes:** `in_proj_qkv`, `in_proj_z` and `out_proj` now stay in their GGUF formats (Q5_K / Q5_K / Q8_0), since `qwen35_from_gguf` is an identity at 16/16 heads. That is about 757 → 290 MB/token, so streamed bytes drop to about the 1.33 GB the roofline already counts. CPU parity on the real GGUF: same top-1 on 4 steps, min cosine 0.9999985.
- **M2 pairing loss:** `in_proj_a`/`in_proj_b` stay FP16, and `pair_small_hand_gemvs` pairs only BF16, so 18 pairs split. About 22 µs/token at the measured ~1.2 µs per dispatch. A `gemv_f16_pair` would recover it; add it if the M2 ABBA shows the gap matters.
- **Regression caught on the way:** fusing q/k/v over `Mat` re-sorted the graph (`GraphRewriter::replace`), which split M7's MoE subgraph and norm pairs (658 → 818 dispatches). Fixed at the root in `src/graph/pattern.rs` (`15da3aa`), with a CPU order test.
- **M5 / M6 (dumped after the e-graph cutover, ctx 80):** SmolLM2 243 steps / 273 dispatches and Llama-1B bnb 131 / 147. Both equal the README dispatch counts, so the order-stable `GraphRewriter::replace` changed nothing there (M4: 131 / 147, also equal).
- **Not dumped:** M3 (9B: the loader's f32/bf16 conversions need more free memory than the shared machine had) and M8 (= M0's plan at 4k ctx). M3's −8 (q/k/v pairs) and −168 (DeltaNet) are inferred from GGUF headers and layer counts.
- **E-graph cutover:** dumps of rows 0/1/2/4/7 after the cutover (`61b27c6`) are byte-identical to the pre-cutover dumps above (step lines, kind histogram, totals).
- **`GIBBS_PLE_FUSED=one` on M1:** 486 → 451 dispatches with 36 `GluGemvChain` steps (the BF16 PLE weights go through the expert pseudo-codecs).
- **Device check:** rows 0/1/2/7 never loaded the AGX driver. Row 4 loaded `AGXMetalG17X` and `IOGPU` as the last two images, after the dump printed, so something on the Llama path creates a device on exit. Nothing is dispatched, but the tool is not strictly device-free there.
- **Host profile quarantined:** `~/.cache/gibbs/host/Apple-M5-Pro-25G241-4ed0d8e9b05eb2d9.json` was written at 10:40 by Metal tests that agents ran by mistake, measuring 156.0 GB/s under training load (baseline 286.9). It is renamed `*.contaminated`. A second contaminated copy (11:45, 95.7 GB/s, 16 µs dispatch) is renamed `*.contaminated2`; the go-ahead run re-measured it (below).

## GPU probes (go-ahead run, 2026-10-19)

Every job ran through `central_scheduler.py run --quiet` on an otherwise idle GPU; raw logs are in `target/sprint3/` of the `s3-int` worktree.

**Host profile and tune file.** `cargo test --lib` writes the host profile without the foreign-load check (`cfg!(not(test))` in `Runtime` setup), and its parallel tests persist tune winners measured under each other's load. Before the probes:
- The `sprint3` profile was re-measured by one quiet single-model `gibbs --bench` (M0): 292.7 GB/s, 0.63 µs dispatch, codec rates included.
- The 107 tune entries keyed by the current kernel hashes (main and sprint3 share them for 33 of 36 formats; only Q4_K/Q5_K/Q6_K differ, through `gen64`) moved to `~/.cache/gibbs/host.pre-s3libtests/tune-current-hash-entries.json`, so every current shape re-tunes on a quiet GPU. Later test runs restore the tune file afterwards.

**Load bug found on the way.** M0 failed to load on `sprint3` (`missing weight 'layers.0.attn.q.weight'`): the q/k/v `Mat` fusion (1.2) removed the parts from the decode program's weights, but batched prefill plans a fresh, unfused build of the network through `GraphProgram::run_reshaped`. Fixed in `8cbbd9e`: the e-graph report names each applied q/k/v trio and `run_reshaped` replays them (`fusion::replay_qkv`), with a CPU regression test. After it: `cargo test --lib` 872 passed / 0 failed, the guard's integration tests pass, `--bench-parity` 5/5 (min cosine 0.99999–1.00000).

### P0.1 kernel profile (`GIBBS_PROFILE_KERNELS=1`, sprint3 `8cbbd9e`)

Last decode step's table per model (µs per token; CachedAttention's bytes count the whole KV allocation, so its GB/s is not a rate):

| row | GemvPacked | MegakernelMlp | CachedAttention | RmsNorm (+Batch) | other |
|---|---|---|---|---|---|
| M0 | 202 steps, 3494 µs, 173 GB/s (59%) | 15, 1035 µs, 79% | 35, 1315 µs | 112, 733 µs | 15 µs |
| M1 | 200, 4447 µs, 218 GB/s (74%) | 18, 2422 µs, 63% | 35, 1317 µs | 106, 701 µs | 14 µs |
| M2 | 157, 4830 µs, 241 GB/s (82%) | 8, 895 µs, 65% | 6, 196 µs | 47, 255 µs | DeltaNetStep 18 steps 285 µs (15.8 µs each, 80 GB/s); 32 µs |
| M7 | 186, 7027 µs, 235 GB/s (80%) | — | 30, 999 µs | 181, 1517 µs | ExpertLinear 60 steps 3180 µs; TopK 288 µs; MoeWeightedSum 116 µs |
| M8 | 202, 3504 µs, 59% | 15, 1034 µs, 79% | 35, 3800 µs | 112, 746 µs | 14 µs |

M0 and M1 (finished 17:21 and 17:24) ran while an 11 GB model download was still writing (done 17:25), so their tables may read slow; nothing was decided from them. Profiling serializes command buffers, so the decode tok/s these runs print (5–12) is not a bench number.

### P0.3 llama-bench refresh (`llama-bench -p 0 -n 128 -r 3`, Homebrew llama.cpp 0.6.0_1, ggml 0.26.0)

| row | llama.cpp tg128 now | 2026-10-13 value in `bench-reference.json` |
|---|---|---|
| M0 | 145.11 ± 0.89 | 150.7 |
| M1 | 115.69 ± 1.20 | 121.8 |
| M2 | 143.78 ± 2.15 | 148.3 |
| M3 | 42.60 ± 0.08 | 43.2 |
| M4 | 260.23 ± 1.63 | 269.0 |
| M7 | 80.86 ± 0.14 | — |
| M8 (`-d 5777`) | 122.88 ± 9.32 | — |

Every row reads 1.4–5% below the 2026-10-13 value on the same build, so this machine state is slower than that day; compare sprint3 with main only inside one ABBA session, not against the README numbers. `bench-reference.json` is not updated by this probe.

### P0.4 Wave 11 ABBA (`5de3057` cand vs `4ff5e60` base, `scripts/loop_verify.sh`)

delta_pp +0.03, noise_pp 0.04, min ratio 0.990, ratios_ok, E2B QAT 149.1 tok/s. Below the 0.30 pp keep gate: Wave 11 is neutral on the suite, which closes `rsi-44-w11` as no measurable effect.

### P0.5 same-model reference (`bench/reference-7b.json`, sprint3 `8cbbd9e`)

| model | Gibbs decode tok/s | roofline % (292.7 GB/s ceiling, streamed bytes) |
|---|---|---|
| Llama 2 7B Q4_0 | 62.4 | 79.2 |
| Llama 2 7B Q8_0 | 35.6 | 85.3 |

### 6.2 K-quant `fused_dot` with ≥ 64 MB of weights rotating

`kquant_fused_dot_ablation` (5 ABBA rounds, median):
- Q4_K 12288×1536 r2: unfused 210.6 GB/s vs fused 214.5 GB/s, fused wins 5/5 (1.019×).
- Q4_K 4096×1536 r1: 187.1 vs 191.8 GB/s, 5/5 (1.025×).
- Q6_K 1536×12288 r1: 261.1 vs 258.9 GB/s, 2/5 (0.992×); a second run 260.2 vs 261.0, 4/5 (1.003×). Q6_K stays unfused.

The Q4_K fused form is the candidate on branch `s3/q4k-fused-dot` (`3cc1ea3`), pending its own ABBA.

### 6.1 `gen64` (64-thread K-quant GEMV) against the 256-thread variants (`shape_rates`, DRAM rotation)

Best GB/s per shape, `gen64` vs the best other variant:
- Q4_K: `gen64/r2` 196 vs 195 at 6144×1536 (tie); behind at 1536×6144 (184 vs 198), 2048×1536, 1536×2048, 1536×2816; tie at the lm_head shapes.
- Q5_K: `gen64/r1` ahead by 1–3 GB/s at 6144×1536, 2048×1536, 2816×768; tie or behind elsewhere.
- Q6_K: `gen64` never ahead (`gen/r1` best at every projection shape).

No shape gains more than 1.5%. The tuner keeps the incumbent on ties (`tune::beats`), so `gen64` is picked only where it clears the margin; see the tune-file check after the suite ABBA.


## Phase 4 ABBA (`scripts/loop_verify.sh`, full suite, base binary first)

Clean runs started at 0% GPU utilization. From about 23:58Z a foreign graphics workload (Renderer ~45%, Tiler ~10%, no CPU-heavy process) held the GPU, so the `deltanet` and `qwen_packed` runs below are contended and need a rerun.

| run | base | cand | delta_pp | noise_pp | ratios_ok | per-model (cand / base tok/s) |
|---|---|---|---|---|---|---|
| main vs sprint3 | main `5de3057` | sprint3 `8cbbd9e` | −2.13 | 1.78 | no | E2B QAT 0.91, E2B Q4_K_M 0.94, Qwen3.5-2B **1.30**, 9B 0.91, Llama Q4_0 0.97, SmolLM2 0.91, bnb 1.02, 26B-A4B 1.01, E2B ctx4k 0.77 |
| f16 KV | sprint3, `GIBBS_KV_DTYPE=f32` | sprint3 (f16 default) | −3.82 | 0.06 | no | E2B QAT 0.91, E2B Q4_K_M 0.91, Qwen3.5-2B 0.99, 9B 1.00, Llama Q4_0 0.96, SmolLM2 0.92, bnb 0.97, 26B-A4B 0.96, E2B ctx4k **0.76** |
| views | sprint3, `GIBBS_DISABLE=views` | sprint3 | +0.01 | 0.10 | yes | all within 0.99–1.01 |
| q/k/v over `Mat` | sprint3, `GIBBS_DISABLE=qkv_mat` | sprint3 | −0.03 | 0.46 | yes | E2B QAT 1.02, E2B Q4_K_M 1.01, 26B-A4B 1.01, others 0.97–1.01 |
| DeltaNet step (contended) | sprint3, `GIBBS_DISABLE=deltanet` | sprint3 | +1.35 | 1.20 | no | Qwen3.5-2B 1.02 (spread 5.5%), 9B 0.84 (spread 43%); unrelated rows moved ±30% |
| Qwen3.5 packed (contended) | sprint3, `GIBBS_DISABLE=qwen_packed` | sprint3 | +0.51 | 0.17 | no (E2B QAT 0.96, unaffected by the switch) | Qwen3.5-2B **1.23** (105.1 → 129.2) |

- **f16 KV** is the whole main-vs-sprint3 regression: with f32 KV, sprint3 decodes E2B QAT at 152.8 tok/s against main's 149.8. Dispatch counts are equal; the GPU time grows in attention (E2B ctx4k 7.61 → 10.06 ms/token). f32 is the default again (`3929384`), f16 is opt-in.
- **views** and **q/k/v over `Mat`** are neutral: below the 0.30 pp keep gate, no ratio regression.
- The Qwen3.5-2B gain (1.30× vs main) comes from the packed `linear_attn` projections (1.23× alone, contended) and the DeltaNet step.
