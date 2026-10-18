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
| 7 | `Gemma 4 26B-A4B · UD-IQ2_XXS` | MoE routed experts (`IQ2_XXS` `MoeExpertLinear` `cols=2816, 768`) + 923 dispatches | **71.1 → 73.1 tok/s** (`cand1` `71.4` in `bg_168`, `73.1` in `bg_169`) | 51.8 – 53.3% | 923 (`863` in `rsi-23`) | **210 excess MoE dispatches/tok** (`TopK+Gather+Softmax` 150, `Slice+Slice+GeGlu` 60, `Mul+Reduce` 30) + `IQ2_XXS` 25% idle SIMD lanes on `cols=768, 2816` |
| 8 | `Gemma 4 E2B QAT · 4k context` | Same weights as #0 at `pos = 4031` (`gpu_ms` `7.71 -> 7.625 ms` in `bg_168`) | **125.3 → 126.8 tok/s** (`+1.21%` in `bg_168`) | 36.1 – 36.6% | 506 | `Track B3-v2` pure-vec4 `attn_cached_gqa_split_vec4_t<4,8,16>` proven (`-90 µs` `gpu_ms`) |

### Empirical Invariants & Proven Laws (`iters 1..41`, `rsi-11` → `rsi-23`)
1. **Previously Refuted Ideas (Single-Variable Quiet-GPU Ledger — Do Not Repeat)**:
   - Bare `k_unroll` (`iter 1`); concurrent encoder + prefetch (`iter 4`); barrier-heavy persistent chain over all `m=1` GEMVs (`iter 14`); removing 15–56 small dispatches via F2/F4/F5/F6 without reducing DRAM bytes or in-kernel time (`iters 22–25, 28, 30`); heterogeneous `GemvPackedMulti` without shared threadgroup `x` (`iter 29`); unchunked decode-ahead CB (`iter 35`); ICB replay (`iter 39`); planar `Q3_K`/`Q5_K` (`iter 41`); paired-row `gemv_q4_k_paired_glu` (`rsi-17`); 2-dispatch `MegakernelMlp` `Epi::UpGeGlu` (`rsi-21` `bg_153`, regressed `Model 1` `0.9925`); 16-group `TILE_TQ2_0` (`rsi-22` `bg_164`, broke `gen/r4` 16B coalescing `202 -> 151 GB/s`) & 16-group `WIDE_IQ2_XXS` (`bg_167`, regressed `1536x6144` `130 -> 108 GB/s`); 16 KB `shared_x[4096]` TG-staged `RmsNorm` prologue (`rsi-22` `bg_166`, halved Apple GPU TG occupancy `4 -> 2` TGs/core and regressed `Model 4` `0.9535`).
2. **Deterministic Invariant — `HostProfile.codec` Rate Stability (`rsi-19` `ed069d4` / `a52ebc1`)**: Landed on `main`. Note also that `codec::source_hash()` (`src/graph/codec.rs:1067`) hashes `c.msl`, `wide(f)`, `tile(f)`, and `codegen::gemv_packed(c, p)` — it does **not** hash `codegen::gemv_indexed(c, p)` or `src/graph/metal/ext/moe.rs`, so MoE kernel/planner changes preserve `codec::source_hash()` 100%.
3. **Apple GPU Threadgroup SRAM Occupancy Law (`rsi-22` `bg_166`)**: Apple M5 Pro GPU cores have `32 KB` threadgroup memory and support 4 resident threadgroups of 256 threads (`1024` threads/core). Any kernel allocating `> 8 KB` of `threadgroup` memory drops max occupancy to `<= 3` or `2` TGs/core. Never allocate `> 8 KB` of `threadgroup` memory in bandwidth-bound GEMV kernels.
4. **Staged Promotion Candidate (`rsi-23` `fd58c71`, branch `experiment/rsi-23-zero-tax-vec4-qtr-c2`)**: Combines `rsi-20` (`09a5148`) + `Track C2` (`42ff08b`). Won 8/9 models in uncontended Pass 1 (`bg_168`) and reached **`151.23 tok/s`** (`+1.95%`, `151.0` & `151.5`) on Gemma 4 E2B QAT before mid-run foreign desktop spikes hit `cand2`.
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

## 3. Wave 3 Avant-Garde Priority Queue (MoE Subgraph Fusion, 100%-SIMD Indexed GEMV, Pair GEMV & Cross-Hardware MoE)

### Track M1 (`exp-m1-moe-fusion`, `ExpertM1MoeSubgraphFusion`) — Fused MoE Router `TopKSoftmax`, Zero-Copy `SplitGlu` & `MoeWeightedSum` (`-210` dispatches/tok on Model 7)
- **Files**: `src/graph/metal/ext/moe.metal`, `src/graph/metal/ext/moe.rs`, `src/graph/metal_kernels.metal`, `src/graph/metal.rs`
- **Problem**: `Model 7` (`Gemma 4 26B-A4B · UD-IQ2_XXS`, 30 MoE layers) executes **923 dispatches/token**, of which **270 dispatches/token** are tiny non-GEMV MoE glue kernels:
  1. **Router (`5` dispatches/layer = `150` dispatches/tok)**: `ids = TopK(logits)` -> `picked = GatherCols(logits, ids)` -> `route_w = Softmax(picked)` -> `per_slot = GatherCols(per_expert_scale, ids)` -> `route_w_scaled = Mul(route_w, per_slot)`. Yet `moe_topk_f32` (`src/graph/metal/ext/moe.metal:47-54`) ALREADY computes both the winning index `fi` and winning logit `fv` in thread `t == 0`!
  2. **Fused Gate/Up Split (`3` dispatches/layer = `90` dispatches/tok)**: `gu = MoeLinear(expert_in, gate_up_proj)` (`[slots, 2 * ff]`) is followed by two non-contiguous axis-1 `Slice` copy dispatches (`expert_gate = Slice(gu, 0..ff)` and `expert_up = Slice(gu, ff..2*ff)`) before `GeGlu(expert_gate, expert_up)`.
  3. **Weighted Expert Reduction (`2` dispatches/layer = `60` dispatches/tok + `5.4 MB/tok` DRAM round-trip)**: `weighted = Mul(down [slots, h], w_col [slots, 1])` writes `[8, 2816]` floats to device memory, and `routed = Reduce(Sum, axes=[1])(Reshape(weighted, [n, k, h]))` immediately reads those `[8, 2816]` floats back to sum over `k = 8`.
- **Design**:
  1. **Fused `moe_topk_softmax_f32` (`-120` dispatches/tok)**: Extend `moe::Step::TopK` with `weights_out: Option<(ValueId, Option<ValueId>)>` (using `Planned.extra_outs: vec![weights_val]`). When `plan()` sees `TopK -> GatherCols(logits, ids) -> Softmax(axis=-1)` (optionally followed by `GatherCols(scale_table, ids, shared=true) -> Mul`), fuse all 3–5 steps into `TopK` (`moe_topk_softmax_f32` for `k <= 32`), where thread `t == 0` records `picked_v[round] = fv; picked_i[round] = fi;`, computes the softmax over `0..k`, optionally multiplies by `scale_table[picked_i[r]]`, and writes `weights_out` in the same dispatch!
  2. **Fused `Step::SplitGlu` (`split_glu_f32`, `-60` dispatches/tok)**: In `src/graph/metal.rs` `plan()`, when `Op::GeGlu` or `Op::SwiGlu` reads `Slice(gu, axis=1, 0..ff)` and `Slice(gu, axis=1, ff..2*ff)` (single-reader slices of the same `[slots, 2*ff]` tensor `gu`), replace the 2 `Copy2d` steps + `Unary` step with 1 `Step::SplitGlu { x: gu, rows: slots, half_cols: ff, is_swiglu }` dispatch reading `gu[r * 2 * ff + c]` and `gu[r * 2 * ff + ff + c]` directly!
  3. **Fused `Step::MoeWeightedSum` (`moe_weighted_sum_f32`, `-30` dispatches/tok)**: In `src/graph/metal.rs` `plan()`, when `Reduce(Sum, axes=[1])` reads `Reshape(Mul(down [slots, h], w_col [slots, 1]), [n, k, h])` (`slots == n * k`), replace `BroadcastBinary(Mul) + Reduce(Sum)` with 1 `Step::MoeWeightedSum { down, weights: w_col, n, k, h }` dispatch!

### Track M2 (`exp-m2-moe-idx`, `ExpertM2MoeIndexedGemv`) — 100%-SIMD-Utilization Low-Register `gemv_indexed` for `IQ2_XXS` (`Model 7`)
- **Files**: `src/graph/codegen.rs` (`1500–1570`), `src/graph/metal/ext/moe.rs` (`380–488`)
- **Problem**:
  1. In `Model 7` (`Gemma 4 26B-A4B · UD-IQ2_XXS`), `expert_down` has `cols = 768` (`3` blocks of 256) and `expert_gate_up` has `cols = 2816` (`11` blocks of 256). Because `WIDE_IQ2_XXS` uses `groups = 8` per 256-element block, `cols = 768` has only `24` groups (leaving **lanes 24..31 = 25% of every 32-lane simdgroup 100% idle**!) and `cols = 2816` has `88` groups (`32 + 32 + 24`, leaving 8 lanes idle on the 3rd pass).
  2. Worse, `wide_loop` allocates `float4 xs[8]` (`32` floats) + `float v[33]` (`33` floats) = **65 float registers per thread** before doing a single dot product!
  3. Note that `codec::source_hash()` (`src/graph/codec.rs:1067`) hashes `gemv_packed` only — it does **not** hash `gemv_indexed`, so optimizing `gemv_indexed` never invalidates `HostProfile.codec`!
- **Design**:
  1. In `src/graph/codegen.rs` `gemv_indexed`, when `codec.format == QuantFormat::GGUF_IQ2_XXS` (or when `wide(codec.format)` has `groups = 8` and `((p.cols / be) & 3) != 0`), emit a **32-sub-blocks-per-block (`subs = p.cols / 8`) vectorized `decode8` loop** where all 32 SIMD lanes process 1 sub-block (`8` elements) per 256-element block (**100% SIMD lane utilization on `cols = 768`, `1536`, `2816`, and every multiple of 256!**).
  2. In that vectorized `IQ2_XXS` sub-block loop (`sub = s & 31`, `blk = s >> 5`, `ib = sub >> 2`, `l = sub & 3`), 4 adjacent lanes (`l = 0..3`) share the same 8-byte group `q = blk_ptr + 2 + 8 * ib` and `aux1 = cd_u32(q + 4)` (1-cycle L1 broadcast across the 4 lanes), unpack `uint2 gw = as_type<uint2>(IQ2XXS_GRID[q[l]])` into `float4 v0 = float4(as_type<uchar4>(gw.x)), v1 = float4(as_type<uchar4>(gw.y))`, apply sign bits `KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 127u]`, and dot directly with `x0, x1` using **only 16 float registers instead of 65**!

### Track A5 (`exp-a5-multigemv`, `ExpertA5MultiGemv`) — Threadgroup-Partitioned Twin-Projection `GemvPackedPair` (`k_proj + v_proj`, `in_proj_b + in_proj_a`)
- **Files**: `src/graph/metal.rs`, `src/graph/metal_kernels.metal`
- **Problem**: Across every model in `bench/loop.json`, sibling projections with identical `x` and small row counts (`k_proj + v_proj` on Gemma 4 `[512, 1536]`, Llama 3.2 `[512, 2048]`, SmolLM2 `[192, 576]`, Qwen3.5 `[512, 2048]`; and `in_proj_b + in_proj_a` `[16, 2048]` on Qwen3.5 DeltaNet layers) launch 2 separate GEMV dispatches. For `rows = 16..512` (`1..16` threadgroups), each dispatch under-fills the 16-core Apple GPU and pays an extra dispatch boundary.
- **Design**:
  - Add `Step::GemvPackedPair` in `src/graph/metal.rs` and `gemv_<fmt>_pair` wrapper entry points in `src/graph/metal_kernels.metal` for `gemv_q4_0`, `gemv_q4_0_half`, `gemv_q4_k`, `gemv_q8_0`, `gemv_bf16`, `gemv_f16`:
  - When two adjacent `Step::GemvPacked` steps (`s0`, `s1`) in `plan()` (after `assign_variants`) share the same input `x`, `m == 1`, and use the same hand kernel (`pso0 == pso1`), merge them into `Step::GemvPackedPair` with grid `(tgs0 + tgs1, 1, 1)`.
  - Inside `gemv_<fmt>_pair`, `if (tg.x < p.tgs0)` calls the inline body of `gemv_<fmt>` on `(w0, x, y0, p0, bias0, res0)` at `tg.x`, else calls it on `(w1, x, y1, p1, bias1, res1)` at `tg.x - p.tgs0`. Each threadgroup executes **only one** matrix's loop, so per-thread register count is 100% unchanged while eliminating 15–56 dispatches/token and doubling GPU core utilization on small projections!

### Track D3 (`exp-d3-xla-cuda`, `ExpertD3CrossHwFusion`) — Cross-Hardware `MoeLinear` (`gemv_indexed_for_dialect`) + Clang CUDA/HIP Gate & Planner
- **Files**: `src/graph/codegen.rs`, `src/graph/cuda.rs`, `src/graph/rocm.rs`
- **Design**:
  1. Add `pub fn gemv_indexed_for_dialect(dialect: Dialect, codec: &Codec, p: GemvParams) -> String` in `src/graph/codegen.rs`, lowering expert-indexed `Op::MoeLinear` across `Dialect::Msl`, `Dialect::Cuda`, and `Dialect::Hip` for all 37 `QuantFormat::ALL` codecs.
  2. Extend `clang_compiles_cuda_and_hip_kernels_for_every_codec` in `src/graph/codegen.rs` to compile `gemv_indexed_for_dialect` for all 37 codecs under `/opt/homebrew/opt/llvm/bin/clang -x cuda --cuda-gpu-arch=sm_80 -fsyntax-only` and `-x hip --offload-arch=gfx90a -fsyntax-only`.
  3. Extend `src/graph/cuda.rs` and `src/graph/rocm.rs` (`supports_op`, `CudaStep::MoeLinear` / `RocmStep::MoeLinear`, `TopK`, `GatherCols`, `plan_graph`, and `CudaSession::run` / `RocmSession::run`) so full MoE graphs (`Router -> TopK -> GatherCols -> Softmax -> MoeLinear -> SwiGlu/GeGlu -> MoeLinear`) plan and compile cleanly on CUDA and ROCm, verified by unit tests in `cuda::tests` and `rocm::tests`.
