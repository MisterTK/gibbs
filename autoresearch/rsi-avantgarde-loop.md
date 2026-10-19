# Autonomous RSI Loop v4: Avant-Garde Subgraph Megakernel JITting, MoE Fusion, Zero-Tax Specialization & Cross-Hardware Lowering

Run an unbounded (`Iterations: unlimited`) autonomous optimization loop on `/Users/keas/code/rust-inference-server` (`main`, currently at `662800a` / `a52ebc1` with `rsi-19` Infra + `Track D2` & `D2-Planner` promoted, and `rsi-23` `fd58c71` staged on `experiment/rsi-23-zero-tax-vec4-qtr-c2`) to push `geomean_roofline_pct` across `bench/loop.json` (and held-out `bench/sprint2.json`) toward the `>=90%` roofline target and lock in `>=150 tok/s` on Gemma 4 E2B QAT **across all clean 2-run ABBA passes** via **generic compiler, subgraph JIT codegen, kernel micro-architecture, and hardware-profile-driven scheduling changes only**.

---

## 1. Ground-Truth Baseline & Empirical Ledger Laws (`rsi-11` → `rsi-23`)

| # | Model (`bench/loop.json`) | Key Bottleneck / Format | Clean ABBA (`a52ebc1` base / `fd58c71` cand) | % of Roofline | Dispatches / tok | Primary Remaining Gap |
|---:|:---|:---|---:|---:|---:|:---|
| 0 | `Gemma 4 E2B QAT · UD-Q2_K_XL` | `TQ2_0` shared FFNs (`12288x1536`, `1536x12288`) + `Q4_0` PLE (`1536->256->1536` x35) + 506 dispatches | **148.3 → 151.2 tok/s** (`+1.95%` in `bg_168`: `151.0` & `151.5`) | 42.7 – 43.6% | 506 | `gemv_q4_0_quarter` (`cols=256` PLE) proven in `rsi-23`; remaining: 15x `k+v` pair GEMV & 70x `post_norm` |
| 1 | `Gemma 4 E2B · Q4_K_M` | `Q4_K` / `Q5_K` projections + 509 dispatches | **112.9 → 113.9 tok/s** (`+0.88%` in `bg_168`) | 57.3 – 57.8% | 509 | 15x `k+v` pair GEMV, 35x PLE bottleneck, 70x `post_norm` |
| 2 | `Qwen3.5 2B · UD-Q4_K_XL` | 24 `DeltaRule` + `CausalConv1dCached` layers + 8 `CachedAttention` gate layers + `IQ4_XS` | **114.6 → 116.7 tok/s** (`+1.81%` in `bg_168`) | 52.4 – 53.4% | 430 (`328` in `rsi-23`) | `-102` dispatches proven; remaining: 24x `in_proj_b + in_proj_a` 1-TG pair GEMV (`-24` disp) |
| 3 | `Qwen3.5 9B (DavidAU MTP) · IQ2_M` | `IQ2_S`/`IQ3_S`/`Q4_K` + `BF16` LM head + 594 dispatches | **39.2 – 40.0 tok/s** (`+0.64%` in `bg_154`) | 60.2% | 594 (`458` in `rsi-23`) | `-136` dispatches proven; remaining: 32x `in_proj_b + in_proj_a` pair GEMV (`-32` disp) |
| 4 | `Llama 3.2 1B · Q4_0` | `Q4_0` (`256–267 GB/s`) + 147 dispatches (`host_ms` `0.265 -> 0.201 ms` in `C2`) | **257.2 → 259.5 tok/s** (`+0.91%`, `4/4` in `bg_165` & `bg_168`) | **68.3%** | 147 | `Track C2` 2-CB `(16, usize::MAX)` pipeline proven (`-55 µs` `host_ms`); remaining: 16x `k+v` pair GEMV |
| 5 | `SmolLM2 135M · safetensors` | `Q4_K` re-encoded (`0.27 GB/tok`, `gpu_ms` 1.89 ms, 273 dispatches) | **482.7 → 483.0 tok/s** (`+0.21%` `4/4` in `bg_165`) | 43.3% | 273 | 30x `k+v` (`Q4_K` `[192, 576]`, 1 TG each!) pair GEMV (`-30` disp) |
| 6 | `Llama 3.2 1B · bnb 4-bit` | `BNB_NF4` -> `Q4_K` (`max-speed`) + 147 dispatches | **229.9 → 233.0 tok/s** (`+1.26%`, `4/4` in `bg_165`, `cand1` `234.4`) | 59.8% | 147 | `Track C2` proven (`-66 µs` `host_ms`); remaining: 16x `k+v` (`Q4_K` `[512, 2048]`) pair GEMV |
| 7 | `Gemma 4 26B-A4B · UD-IQ2_XXS` | MoE routed experts (`IQ2_XXS` `MoeExpertLinear` `cols=2816, 768`) + 923 dispatches | **69.9 → 72.5 tok/s** (`+3.65%` in `bg_178` `cand1`/`cand2` vs `base1`; `gpu_ms` `13.92 -> 13.54 ms`) | 52.5 – 54.4% | 923 (`713` in `rsi-24`) | `Track M1` (`-210` disp, `-380 µs` `gpu_ms`) staged; remaining: `IQ2_XXS` `MoeExpertLinear` (`7.33 ms/tok`) & `MoeWeightedSum+RmsNorm` fusion |
| 8 | `Gemma 4 E2B QAT · 4k context` | Same weights as #0 at `pos = 4031` (`gpu_ms` `7.71 -> 7.625 ms` in `bg_168`) | **125.3 → 126.8 tok/s** (`+1.21%` in `bg_168`) | 36.1 – 36.6% | 506 | `Track B3-v2` pure-vec4 `attn_cached_gqa_split_vec4_t<4,8,16>` proven (`-90 µs` `gpu_ms`) |

### Empirical Invariants & Proven Laws (`iters 1..41`, `rsi-11` → `rsi-25`)
1. **Previously Refuted Ideas (Single-Variable Quiet-GPU Ledger — Do Not Repeat)**:
   - Bare `k_unroll` (`iter 1`); concurrent encoder + prefetch (`iter 4`); barrier-heavy persistent chain over all `m=1` GEMVs (`iter 14`); removing 15–56 small dispatches via F2/F4/F5/F6 without reducing DRAM bytes or in-kernel time (`iters 22–25, 28, 30`); heterogeneous `GemvPackedMulti` without shared threadgroup `x` (`iter 29`); unchunked decode-ahead CB (`iter 35`); ICB replay (`iter 39`); planar `Q3_K`/`Q5_K` (`iter 41`); paired-row `gemv_q4_k_paired_glu` (`rsi-17`); 2-dispatch `MegakernelMlp` `Epi::UpGeGlu` (`rsi-21` `bg_153`, regressed `Model 1` `0.9925`); 16-group `TILE_TQ2_0` (`rsi-22` `bg_164`, broke `gen/r4` 16B coalescing `202 -> 151 GB/s`) & 16-group `WIDE_IQ2_XXS` (`bg_167`, regressed `1536x6144` `130 -> 108 GB/s`); 16 KB `shared_x[4096]` TG-staged `RmsNorm` prologue (`rsi-22` `bg_166`, halved Apple GPU TG occupancy `4 -> 2` TGs/core and regressed `Model 4` `0.9535`); refactoring shared hand kernels into `inline void gemv_<fmt>_body` helpers (`rsi-25` `bg_176`, regressed `Model 4` `0.9896` & `Model 5` `0.9881` even where pairing did not fire).
2. **Deterministic Invariant — `HostProfile` Ceiling & Codec Rate Stability (`rsi-19` `ed069d4` / `a52ebc1` + Wave 4 fix)**: When `KERNELS_MSL` changes while `codec::source_hash()` is unchanged, `Runtime::new` carries over `(memory_bandwidth_gbps, stream_gbps, ceiling_gbps, codec)` from the matching `HostProfile` on disk so `base` and `cand` share identical `ceiling_gbps` and `bytes_per_token`. Note also that `codec::source_hash()` (`src/graph/codec.rs:1067`) hashes `c.msl`, `wide(f)`, `tile(f)`, and `codegen::gemv_packed(c, p)` — it does **not** hash `codegen::gemv_indexed(c, p)` or `src/graph/metal/ext/moe.rs`.
3. **Apple GPU Threadgroup SRAM Occupancy Law (`rsi-22` `bg_166`)**: Apple M5 Pro GPU cores have `32 KB` threadgroup memory and support 4 resident threadgroups of 256 threads (`1024` threads/core). Any kernel allocating `> 8 KB` of `threadgroup` memory drops max occupancy to `<= 3` or `2` TGs/core. Never allocate `> 8 KB` of `threadgroup` memory in bandwidth-bound GEMV kernels.
4. **Staged Promotion Candidate (`rsi-24` `57dcdf2`, branch `experiment/rsi-24-full-stack`)**: Combines `rsi-23` (`2b7b6c8`) + `Track M1` (`30f6633`: fused MoE `moe_topk_softmax_f32` `-120` disp, `split_glu_f32` `-60` disp, `moe_weighted_sum_f32` `-30` disp & `-5.4 MB/tok` DRAM). Passed `scripts/loop_guard.sh` (`bg_180`, `rc=0`, `parity: 5 compared`) and achieved **`150.59 tok/s` (`bg_178`)** and **`150.78 tok/s` (`bg_179`)** on Gemma 4 E2B QAT (`72.50 tok/s` on `Model 7` `26B-A4B`, `923 -> 713` dispatches). Requires: (a) a real-weight greedy token-ID parity check on `Model 7` (`26B-A4B`, since `--bench-parity`'s 5 reference models are `<= 4B` non-MoE), and (b) one uncontended 4-pass ABBA (`base_spread <= 0.04`).
## 2. Hard Rules & Centralized Single-Resource Scheduler Protocol

### 2.1 Scope & Frozen Contracts
- **Editable**: `src/graph/**`, `src/hardware/**`, `src/scheduler/mod.rs`, `src/model/reencode.rs` (only when updating generic codec efficiency lookup), `autoresearch/**`.
- **Frozen evaluation contract**: NEVER edit `bench/loop.json`, `bench/sprint2.json`, `bench/prompts/*`, `scripts/loop_verify.sh`, `scripts/loop_score.py`, `scripts/loop_guard.sh`, `scripts/generic_guard.sh`, or `src/bench.rs`.
- **Generic compiler law**: Zero model-family names (`gemma`, `qwen`, `llama`, `mistral`, `smollm`, `phi`, `lfm`, `flux`, `ltx`, `bert`, `sana`, `t5`) in `src/graph/**`, `src/hardware/**`, or `src/scheduler/**` outside `#[cfg(test)]`. Every optimization is triggered by **IR topology, tensor shapes, `QuantFormat`, or `HardwareProfile`**.
- **Parity & coverage**: Every kept commit MUST pass `scripts/loop_guard.sh` (`rc=0`, full release lib suite, `codec_parity`, `op_coverage`, `cpu_free_matrix`, `gemma4_tests`, and `--bench-parity bench/loop.json` with `parity: 5 compared` and `min cos >= 0.999`).

### 2.2 Multi-Expert Worktree & Central Scheduler Discipline
1. **Strictly ONE Heavy Job at a Time (`autoresearch/central_scheduler.py`)**:
   - Every `cargo build`, `cargo test`, `shape_rates`, `ab_screen.py`, `loop_guard.sh`, `verify_when_quiet.sh`, or `gibbs --bench` command MUST run through `python3 autoresearch/central_scheduler.py run --label <tag> [--quiet] -- <cmd>`.
   - `central_scheduler.py` holds the machine-wide exclusive flock `~/.cache/gibbs/sched.lock` (never `~/.cache/gibbs/gpu.lock`, which `gibbs` acquires internally) and logs every job to `~/.cache/gibbs/sched.log`.
2. **Parallel Expert Subagents (`task`)**:
   - When parallelizing across independent tracks, create one isolated git worktree per expert (`../gibbs-wt/<track>`, branched from current `main`).
   - Experts ONLY read/edit code in their assigned worktree and commit `experiment: <track>` SHAs — **subagents NEVER invoke `cargo`, tests, or benchmarks**.
   - The orchestrator cherry-picks or tests each expert commit one at a time through `central_scheduler.py`, keeping attribution 100% isolated.

### 2.3 Three-Stage Gate (`Screen -> Guard -> ABBA Verify`)
1. **Stage 1 — Fast Quiet Screen (<3 min via `central_scheduler.py run --quiet`)**:
   - **Codec / Kernel changes**: `SHAPE_RATES_FMT=<FMT> cargo test --release --lib shape_rates -- --ignored --nocapture` must show `>= +5%` GB/s on target shapes with zero regressions.
   - **Graph / Fusion / Executor changes**: `python3 scripts/ab_screen.py target/loop-base/gibbs target/release/gibbs --rounds 4 --models "<targeted_and_control_indices>"`. Always include at least one untouched control model (`0`, `1`, `4`, or `7`). Require `median_ratio >= 1.012` on targeted model(s) and `>= 0.997` on untouched control model(s) when shared kernels are touched (`>= 0.993` otherwise).
   - If Stage 1 fails on a quiet GPU: `git revert HEAD --no-edit`, rebuild `cargo build --release -q`, log `discard (screen)` in `autoresearch/roofline-loop.tsv`, and move on immediately.
2. **Stage 2 — Full Guard (`scripts/loop_guard.sh`)**:
   - Must exit `0` with `parity: 5 compared`.
3. **Stage 3 — Quiet 2-Run ABBA Verify (`scripts/verify_when_quiet.sh`)**:
   - **Infra commit rule** (`infra:` commits that do not alter steady-state kernel execution, such as `HostProfile.codec` cache carry-over): keep iff `delta_pp >= -max(0.30, noise_pp)` and `ratios_ok == true`.
   - **Experiment commit rule**:
     - If `verify_when_quiet.sh` exits `2`/`4` or an untouched control model has `base_spread > 0.04` or `ratio < 0.985` / `> 1.020`, classify as **`inconclusive (contended)`** (save branch `experiment/rsi-<N>-<slug>`, revert `main` to `loop-base`).
     - **KEEP iff**:
       - (`delta_pp >= max(0.30, noise_pp)` OR targeted models improve by `>= +1.2%` with all untouched controls in `0.995..1.008` and `delta_pp >= +0.15 pp`),
       - `ratios_ok == true`, and
       - `scripts/loop_guard.sh` passed (`rc=0`).
     - **On KEEP**: update `target/loop-base/gibbs` and `target/loop-base/COMMIT`, log `keep` in `autoresearch/roofline-loop.tsv` + `autoresearch/roofline-loop-notes.md`. On milestone keeps (`+0.50+ pp` cumulative or E2B QAT crossing `>=150.0 tok/s` across both ABBA passes), refresh `autoresearch/bench-readme.jsonl`, `.meta.json`, and run `cargo run --release --example gibbs_art -q`.

---

## 3. Wave 10 Avant-Garde Priority Queue (Promoting `experiment/rsi-40-w8-hardened` `695b491` on Quiet ABBA & Attacking `IQ2_XXS` `7.5 us` LUT + Register-Residency Cliff)

### 1. Immediate First Action — Promote `experiment/rsi-40-w8-hardened` (`695b491`) on a Clean Unspliced ABBA (or Track-by-Track via Targeted-Model Gate)
- **What Wave 9 proved & fixed on `experiment/rsi-40-w8-hardened` (`695b491`)**:
  - **`variants_hash` attribution (`bg_218` + `RUST_LOG=info`)**: `variants_hash(WeightFormat::Q4_0)` (`src/graph/metal.rs:1148-1159`) hashes `gemv_common(KERNELS_MSL)` + `fmt.hand_kernel()` (`"gemv_q4_0_half"`), **not** `fmt.codec().msl`. `experiment/rsi-40-w8-hardened` uses `de19cee9ffcfa698` (all 16 shapes populated in `tune-Apple-M5-Pro-25G241.json`, 100% cache hit on `Model 0` and `Model 4`), whereas `fc235aa90db71dcd` was an older pre-`W1` `GEMV-COMMON` hash.
  - **Fixed `pair_small_hand_gemvs` (`1574839`) and `tune::decide` fallthrough (`695b491`)**:
    1. Restricted `pair_small_hand_gemvs` (`src/graph/metal.rs`) to `fmt0 == WeightFormat::BF16` (since `prepare_plan` runs before `assign_variants` when all steps still have `variant == GemvVariant::HAND`), restoring `506` dispatches on `M0`/`M8` and `147` on `M4`.
    2. In `tune::decide` (`src/graph/tune.rs`, `695b491`), when 2 quiet rounds disagree on `choose(r)` (`Gen(1)` vs `Gen(2)` within `<1%` after `Model 8`'s 27-second 5777-token prefill), it now falls through to the `worst_ratio >= 1.0 + BUSY_MARGIN` (`8%`) check across all rounds instead of returning `GemvVariant::HAND` (`2.7x` slower on `512x2048`).
  - **Unspliced 4-Pass ABBA `bg_224` on `695b491`**:
    - **`ratios_ok = true` across all 9 models (`M0..M8`)**, with strong gains on **`M0` (`+2.98%`, `spread = 0.0082`, `139.02 -> 143.16 tok/s`)**, **`M1` (`+1.45%`)**, **`M3` (`+0.62%`, `594 -> 570 dispatches`)**, **`M4` (`+1.83%`, `spread = 0.0041`, `229.00 -> 233.19 tok/s`)**, and **`M7` (`+2.35%`, `spread = 0.0123`, `63.12 -> 64.61 tok/s`, `923 -> 713 dispatches`)**, and `M8` `cand` running cleanly at `gpu_ms = 8.097 ms` (no `HAND` fallback).
    - However, one `base` pass on `M8` spiked by `9.9%` (`base_spread = 0.0993 > 0.04`), inflating `noise_pp` to `1.203 pp` (`delta_pp = +0.092 pp`), and `M6` control was `0.9759` under `38–44%` compositor load. (An earlier diagnostic splice of `bg_221` rows 0..7 + `bg_222` `M8` 4-pass ABBA `1.0223` scored `+0.72 pp`, logged as a diagnostic only, not a verdict.)
  - **Action**: Run a single unspliced `scripts/verify_when_quiet.sh` on `experiment/rsi-40-w8-hardened` (`695b491`) when compositor load is quiet, or promote `M4 + M1 + M5c + W8a/W9 tune::decide` (`de19cee9ffcfa698` / `695b491`) via the targeted-model gate (`>= +1.2%` on targeted models with controls in `0.995..1.008` and `delta_pp >= +0.15 pp`).

### 2. Next Avant-Garde Track: `IQ2_XXS` `7.5 us` Table-Lookup Elimination & `WIDE_IQ2_XXS` / `WIDE_IQ2_M` Register-Residency Cliff (`bg_220`)
- **7-Round `moe_gemv_bandwidth` & 5-Round Live-`q4` ALU Stub Evidence (`bg_220`)**:
  - Across 7 interleaved `[2, 4, 4, 2]` ABBA rounds (`bg_220`), `moe_rows_per_sg = 4` did not clear `6/7` wins with `worst_ratio >= 1.0` (`Q4_0` won `5/7` with `med_ratio = 1.0815` but `worst_ratio = 0.9122`; `IQ4_NL` won `4/7` with `med_ratio = 1.0220`; `IQ2_M` `4/7` `0.9988`; `IQ2_XXS` `3/7` `0.9043`), so `moe_rows_per_sg` stays `2` for all formats.
  - Across 5 interleaved ABBA rounds on `GGUF_IQ2_XXS 1408x2816 (top_k=8, x_rows=1)`, the live-`q4` ALU stub (`as_type<ulong>(uint2(0x08080808u ^ ((q4 >> (8 * k)) & 0xFFu), 0x08080808u))`) measured **`real_med_us = 57.8 us` (`141.4 GB/s`) vs `live_q4_no_lut_med_us = 50.3 us` (`162.6 GB/s`)**, isolating **`lut_cost_us = 7.5 us` (`13.0%` of `57.8 us`)** to the `IQ2XXS_GRID[256]` lookup itself (with the remaining `9.0 us` of Wave 8's `16.5 us` literal-stub delta attributable to `q4` load/shift DCE).
  - Furthermore, `WIDE_IQ2_XXS` (`segs = 4`) and `WIDE_IQ2_M` (`segs = 4`) allocate `float4 xs[8]` (`32` regs) + `float v[33]` (`33` regs) = **`65+` registers/thread**, crossing Apple Silicon's 64-register/thread 100%-occupancy cliff (`64 KB` register file / `1,024` threads = `64` floats/thread). Refactoring `WIDE_IQ2_XXS` and `WIDE_IQ2_M` to `groups: 8, segs: 2` (`16` elements = `2` grid lookups per group, matching `WIDE_IQ4_NL` / `WIDE_Q4_0`'s `16 + 17 = 33` registers/thread) or binding `IQ2XXS_GRID` in constant buffer space can be benchmarked directly with the 7-round `moe_gemv_bandwidth` harness.
