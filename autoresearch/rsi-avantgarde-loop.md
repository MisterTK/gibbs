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

## 3. Wave 10 Promoted Milestone (`+0.88 pp`, `51.45% -> 52.33%`, `bg_228` on `838c19d`) & Wave 11 Avant-Garde Priority Queue

### 1. Wave 10 Promoted Milestone (`rsi-43-w10-keep`, `838c19d` on `main`)
- **What Wave 10 proved & promoted to `main` (`838c19d`) and `target/loop-base`**:
  - **Track W10a (`2b5de70`) — Pre-Indexed `PlanValueLoc`, `(32, usize::MAX)` 2-Chunk CB Submission, and `QUIET_MARGIN = 0.02` in `tune::decide`**:
    - Replaced per-token `steps` scans, `state_map: HashMap<String, Arc<Buffer>>` construction (`70+` `String` clones/tok), and `~2,500` `SipHash-1-3` `HashMap` lookups/tok in `run_prepared` with O(1) slice indexing via `PlanValueLoc`.
    - Changed default `cb_chunking()` from `(24, 160)` (`4–5` command buffers/tok) to `(32, usize::MAX)` (`2` command buffers/tok for plans `>= 220` steps), eliminating 2–3 mid-token command-buffer boundaries and stabilizing `host_ms` at `0.23..0.31 ms` across all 9 models.
  - **Track W10b (`838c19d`) — `Wide::fused_dot` On-the-Fly `dot_group` on `WIDE_IQ2_XXS`, `WIDE_IQ2_M`, and `WIDE_IQ3_S`**:
    - Eliminated the `33`-float `float v[33]` array per thread in `WIDE_IQ2_XXS`, `WIDE_IQ2_M`, and `WIDE_IQ3_S` and vectorized `WIDE_IQ2_M` / `WIDE_IQ3_S` with `packed_ushort2`/`packed_ushort4` loads and `select`, improving `GGUF_IQ2_XXS 1408x2816` in `iq2xxs_ablation` from `57.8 us` (`141.4 GB/s`) to **`55.0 us` (`148.8 GB/s`, `-2.8 us` / `+5.2%`)** (`bg_226`), `Model 7` (`26B-A4B MoE`) by **`+3.69%` (`69.69 -> 72.27 tok/s`, `gpu_ms = 13.537 ms`, `923 -> 713 dispatches`)**, and `Model 3` (`Qwen3.5-9B IQ2_M`) by **`+5.39%` (`39.35 -> 41.48 tok/s`, `cand_pct = 64.86%`, `gpu_ms = 23.820 ms`, `594 -> 570 dispatches`)** (`bg_228`).
  - **Single Unspliced 4-Pass ABBA `bg_228` (`scripts/loop_verify.sh` + `scripts/loop_guard.sh` `rc = 0`, `parity: 5 compared`)**:
    - **`base_pct = 51.45% -> cand_pct = 52.33%` (`delta_pp = +0.880 pp`, `noise_pp = 0.036 pp`, `ratios_ok = true`, `min_ratio = 0.9992`, `max base_spread = 0.0049`)**:
      - `M0` (`E2B QAT`): `146.85 -> 148.92 tok/s` (`+1.41%`, `spread = 0.0028`, `43.42%` roofline)
      - `M1` (`E2B Q4_K_M`): `110.43 -> 110.34 tok/s` (`0.9992`, `spread = 0.0038`, `56.64%` roofline)
      - `M2` (`Qwen3.5-2B`): `111.57 -> 112.85 tok/s` (`+1.15%`, `spread = 0.0013`, `52.17%` roofline, `430 -> 412` disp)
      - `M3` (`Qwen3.5-9B IQ2_M`): `39.35 -> 41.48 tok/s` (`+5.39%`, `spread = 0.0004`, `64.86%` roofline, `594 -> 570` disp)
      - `M4` (`Llama 1B Q4_0`): `249.43 -> 251.75 tok/s` (`+0.93%`, `spread = 0.0027`, `67.02%` roofline)
      - `M5` (`SmolLM2 135M`): `463.44 -> 468.22 tok/s` (`+1.03%`, `spread = 0.0025`, `43.83%` roofline)
      - `M6` (`Llama 1B bnb`): `226.14 -> 226.53 tok/s` (`+0.17%`, `spread = 0.0009`, `60.36%` roofline)
      - `M7` (`26B-A4B MoE`): `69.69 -> 72.27 tok/s` (`+3.69%`, `spread = 0.0006`, `54.82%` roofline, `923 -> 713` disp)
      - `M8` (`E2B QAT ctx4k`): `122.62 -> 124.84 tok/s` (`+1.81%`, `spread = 0.0049`, `36.40%` roofline)

### 2. Wave 11 Avant-Garde Priority Queue (Pushing `M0` `E2B QAT` Past `150.0 tok/s` from `148.92 tok/s`, K-Quant `Wide::fused_dot`, & Zero-Copy Decode Output View)
1. **Track W11a — `Wide::fused_dot` (`dot_group`) for Non-Uniform K-Quants (`WIDE_Q4_K`, `WIDE_Q5_K`, `WIDE_Q6_K`, `WIDE_Q3_K`)**:
   - `WIDE_Q4_K`, `WIDE_Q5_K`, `WIDE_Q6_K`, and `WIDE_Q3_K` (`src/graph/codec.rs:197-330`) currently have `fused_dot: false` (`decode_group` materializing `float v[20..40]` and scalar `for (uint i = 0; i < 4; i++)` loops), which execute on `M0` (`E2B QAT`), `M1` (`E2B Q4_K_M`), `M2` (`Qwen3.5-2B`), `M3` (`Qwen3.5-9B`), `M5` (`SmolLM2`), `M6` (`Llama 1B bnb`), and `M8` (`E2B QAT ctx4k`).
   - Extending `dot_group` to `WIDE_Q4_K`, `WIDE_Q5_K`, `WIDE_Q6_K`, and `WIDE_Q3_K` with `packed_ushort4` loads eliminates `float v[20..40]` and the scalar nibble-unpack loops; pre-warm the new `GGUF_Q4_K_M` / `GGUF_Q5_K_M` / `GGUF_Q6_K` hashes on a quiet GPU before ABBA verification.
2. **Track W11b — `Gemma 4 E2B` PLE & Per-Token Host Overhead (`148.92 -> >= 150.5 tok/s`, `-60..100 us/tok`)**:
   - `Model 0` (`E2B QAT`) is now at **`148.92 tok/s` (`6.725 ms/tok`, `gpu_ms = 6.410 ms`, `host_ms = 0.316 ms`)** — only **`0.72%` (`48 us/tok`)** shy of the `150.0 tok/s` (`6.667 ms/tok`) floor across a 4-pass ABBA!
   - Eliminating the per-token `1.05 MB` (`262,144` `f32`) `s.to_vec()` page-faulting allocation on the decode path (`plans/research-sprint-2.md` §S6.2 / §S8.3 #8) or fusing the 35-layer PLE `inp_gate` (`256x1536` `Q4_0`) + `proj` (`1536x256` `Q4_0`) + `post_norm` saves `~60–120 us/tok`, pushing `M0` cleanly past `150.5+ tok/s`.
3. **Track W11c — `IQ2XXS_GRID` `15.9 us` Lookup Reduction (`55.0 us -> 39.0 us` Ceiling in `bg_226`)**:
   - With `dot_group`, `live_q4_no_lut_med_us` dropped to **`39.0 us` (`209.6 GB/s`)** while `real_med_us` is **`55.0 us` (`148.8 GB/s`)** (`bg_226`, `lut_cost_us = 15.9 us`). Evaluate `uint` (`32`-bit) table staging in `threadgroup uint tg_grid[512]` (`2 KB` TG SRAM, bank-conflict-free 32-bit loads vs Wave 7's 64-bit `ulong` bank conflicts) or constant-buffer `[[buffer(6)]]` binding via the 7-round `moe_gemv_bandwidth` harness.
