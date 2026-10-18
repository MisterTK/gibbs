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

## 3. Wave 6 Avant-Garde Priority Queue (`W1 + M1 + W2` without `M3`, Race-Free Multi-TG `MoeWeightedSumNorm`, & Indexed GEMV `shared_x` Bandwidth)

### 1. Track W2-v2 (`exp-w2v2-gate-up-glu-nom3`, branched from `74c00eb` `W1 + M1` without `M3`) — Isolate `Step::ExpertGateUpGlu` with Fast-Math Preserved
- **Context**: In Wave 5, `Step::ExpertGateUpGlu` (`3a7ba6a`, `-270` dispatches/tok on `Model 7`, `923 -> 653`, verified for 31/31 real-weight greedy tokens in `bg_191`) was only ever measured on top of `Track M3` (`bg_192`), which `bg_193` showed cancels `M1`'s gain, and with `set_fast_math_enabled(false)` disabling fast-math FMA across the entire GEMV loop.
- **Design**:
  - Port `Step::ExpertGateUpGlu` (`gemv_indexed_glu`) onto `74c00eb` (`W1 + M1` alone, **without** `M3`), keeping **fast-math enabled** (`CompileOptions::new()`) on both `moe_gemv_pso` and `moe_gemv_glu_pso`.
  - Prevent `tanh` overflow under fast-math by clamping the `tanh` argument inside `gelu_tanh` (`tanh(clamp(0.79788456f * x * (1.0f + 0.044715f * x * x), -10.0f, 10.0f))`) or calling `metal::precise::tanh`, and add ` IQ2_XXS + Q4_0` (`m = 1, 24`) to `moe_subgraph_fusions_match_cpu`.

### 2. Track M3-v2 (`exp-m3v2-multitg-wsum-norm`, branched from `74c00eb` `W1 + M1`) — Race-Free Multi-TG `MoeWeightedSumNorm`
- **Root Cause of Track M3 v1 (`beafc9b`) Regression (`bg_193` vs `bg_194`)**:
  - Track M3 v1 launched a **single threadgroup (`1` of `16` GPU cores)** to read all `90 KB` (`8 * 2816 * 4` B) of `down [slots, h]`, serializing `2.7 MB/token` of reads onto 1 core (`+275 us/tok`).
- **Race-Free Design**:
  - Launch `num_tgs = ceil(h / 256)` threadgroups (`11` TGs across `11` GPU cores for `h = 2816`, grid `(num_tgs, n, 1)`): each TG `tg_x` computes `256` columns of `routed[r, c] = sum_s down[(r*k + s)*h + c] * weights[r*k + s]`, writes `routed[r, c]` to a device scratch buffer, and writes its 256-column partial sum-of-squares `sum_sq_tg` to `scratch_partials[r * num_tgs + tg_x]`.
  - Lane 0 of each TG executes `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` (NOT `threadgroup_barrier`, which only orders within one TG!) and then `uint done = atomic_fetch_add_explicit(&scratch_counters[r], 1u, memory_order_acq_rel);` + `simd_broadcast_first(done)` / `threadgroup_barrier(mem_flags::mem_threadgroup)`.
  - Only the **last finishing threadgroup** (`done == num_tgs - 1`) executes `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)`, resets `scratch_counters[r] = 0u` (on lane 0), sums `scratch_partials[r * num_tgs .. + num_tgs]`, computes `inv_rms = rsqrt(total_sq / float(h) + eps)`, and normalizes `routed[r, 0..h]` -> `out[r, 0..h]` (`+res`, `*scale`).

### 3. Track M4 (`exp-m4-moe-indexed-bw`, branched from `main`) — `ExpertLinear` `gemv_indexed` Bandwidth Optimization (`moe_gemv_bandwidth`)
- **Context**: `moe_gemv_bandwidth` (`moe-gemv-bw-r2`) measured `GGUF_IQ2_XXS 1408x2816 (top_k=8, x_rows=1)` at `143.7 GB/s` (`56.9 us`) and `GGUF_IQ4_NL 2816x704 (top_k=8, x_rows=8)` at `158.3 GB/s` (`56.3 us`) against a `~290 GB/s` ceiling (`~3.4–4.5 ms/token` across 60 `ExpertLinear` steps on `Model 7`).
- **Design**:
  - Add `GGUF_Q4_0 2816x704 (top_k=8, x_rows=8)` to `moe_gemv_bandwidth` in `src/graph/metal/ext/moe.rs`.
  - In `gemv_indexed` (`src/graph/codegen.rs`), cooperately stage `xb[0..p.cols]` into `threadgroup float shared_x[2048]` (`8 KB` max, preserving `1024` max threads/TG = 100% occupancy) when `p.cols <= 2048` (or tile `xb` in `2048`-float chunks) and vectorize `WIDE_IQ2_XXS` sign selection (`select(g0, -g0, ...)`), gated directly by `moe_gemv_bandwidth`.

### 4. Staged Stack Ready for Quiet ABBA Promotion
- **`experiment/rsi-31-w1-m1` (`74c00eb`)** and **`experiment/rsi-31-w1-m1-rsi23` (`799202e`)**: Staged for `scripts/verify_when_quiet.sh` once the host GPU is idle (`base_spread <= 0.04`).
