# Autonomous RSI Loop v3: Avant-Garde Subgraph Megakernel JITting, Zero-Tax Template Specialization & Cross-Hardware Lowering

Run an unbounded (`Iterations: unlimited`) autonomous optimization loop on `/Users/keas/code/rust-inference-server` (`main`, currently at `1dbeee8` with promoted Metal baseline `96ab660` + Track D1 `1f9d552`) to push `geomean_roofline_pct` across `bench/loop.json` (and held-out `bench/sprint2.json`) toward the `>=90%` roofline target and lock in `>=150 tok/s` on Gemma 4 E2B QAT **across all clean 2-run ABBA passes** (current clean ABBA range: `144.3–150.2 tok/s`, suite geomean `50.08–51.16%`) via **generic compiler, subgraph JIT codegen, kernel micro-architecture, and hardware-profile-driven scheduling changes only**.

---

## 1. Ground-Truth Baseline & Empirical Ledger Laws (`rsi-11` → `rsi-18`)

| # | Model (`bench/loop.json`) | Key Bottleneck / Format | Clean ABBA (`96ab660` / `bg_138`–`bg_146`) | % of Roofline | Dispatches / tok | Primary Remaining Gap |
|---:|:---|:---|---:|---:|---:|:---|
| 0 | `Gemma 4 E2B QAT · UD-Q2_K_XL` | `TQ2_0` shared FFNs (`12288x1536`, `1536x12288`) + `Q4_0` PLE (`1536->256->1536` x35) + 506 dispatches | **144.3 – 150.2 tok/s** (1-pass `151.1`) | 42.2 – 43.9% | 506 | `TQ2_0` FFN GEMV (`165–194 GB/s`), 35x PLE (`140` disp), 70x `post_attn_ln`/`post_ff_ln` |
| 1 | `Gemma 4 E2B · Q4_K_M` | `Q4_K` / `Q5_K` projections + 509 dispatches | 109.4 – 109.9 tok/s | 53.3 – 55.4% | 509 | 70x `post_norm` dispatches, 35x PLE bottleneck, `Q4_K` prologue norms |
| 2 | `Qwen3.5 2B · UD-Q4_K_XL` | 24 `DeltaRule` + `CausalConv1dCached` layers + 8 `CachedAttention` gate layers + `IQ4_XS` | 111.4 – 113.8 tok/s (`+1.29%` in `rsi-18`) | 50.7 – 52.6% | 430 (`328` in `rsi-18`) | `-102` dispatches proven in `rsi-18`; blocked only by `~0.5%` shared-kernel branch tax |
| 3 | `Qwen3.5 9B (DavidAU MTP) · IQ2_M` | `IQ2_S`/`IQ3_S`/`Q4_K` + `BF16` LM head + 594 dispatches | 39.2 – 40.4 tok/s (`+0.81%` in `rsi-18`) | 60.2% | 594 (`458` in `rsi-18`) | `-136` dispatches proven in `rsi-18`; `WIDE_IQ2_M`/`WIDE_IQ3_S` + `DeltaRule` post-norm |
| 4 | `Llama 3.2 1B · Q4_0` | `Q4_0` (`256–267 GB/s`) + 147 dispatches (`host_ms` ~0.30 ms) | 256.8 – 257.3 tok/s | **65.0%** | 147 | Single-CB encode gap (`147 < 220` chunk threshold) + standalone `qkv` pre-norm |
| 5 | `SmolLM2 135M · safetensors` | `Q4_K` re-encoded (`0.27 GB/tok`, `gpu_ms` 1.88 ms, 273 dispatches) | 468.6 – 471.2 tok/s | 41.7% | 273 | Dispatch-bound (`273` dispatches over `1.88 ms` = `6.9 µs`/dispatch average) |
| 6 | `Llama 3.2 1B · bnb 4-bit` | `BNB_NF4` -> `Q4_K` (`max-speed`) + 147 dispatches | 229.8 – 230.4 tok/s | 58.2% | 147 | Same 147-step encode gap + `qkv` prologue norm as Model 4 |
| 7 | `Gemma 4 26B-A4B · UD-IQ2_XXS` | MoE routed experts (`IQ2_XXS` `MoeExpertLinear`) + 923 dispatches | 68.5 – 71.1 tok/s | 51.2 – 52.2% | 923 | `MoeExpertLinear` `IQ2_XXS` kernel throughput + 60x weightless `RmsNorm` + `post_norm` |
| 8 | `Gemma 4 E2B QAT · 4k context` | Same weights as #0 at `pos = 4031` (`gpu_ms` 7.75 ms vs 6.49 ms at short ctx) | 122.3 – 124.7 tok/s | 34.5 – 35.8% | 506 | `attn_cached_gqa_split_t` scalar KV loop (`+1.26 ms/tok` at 4k ctx) + 35 merge dispatches |

### Proven Laws from `rsi-11` → `rsi-18` (NEVER violate)
1. **Shared-Kernel Zero-Tax Law (`rsi-18` `bg_146`)**: Adding runtime `has_gate` buffers/branches to `attn_cached_gqa_split_t` / `flash_attn_cached_t` / `attn_cached_merge_f32` or runtime `vec_scale` branches to `rmsnorm_fused_f32` imposes a **`~0.5–0.7%` tax across all 7 non-Qwen models**, wiping out Qwen3.5's `+1.29%` (`-102` dispatches) and `+0.81%` (`-136` dispatches) wins. Every optional fusion feature on a shared kernel **MUST** use compile-time C++ template specialization (`template <uint N_SPLITS, bool HAS_GATE = false>`, `rmsnorm_fused_vecscale_f32`) so untouched models execute the exact `96ab660` instruction stream.
2. **Paired-Row Register Ceiling Law (`rsi-17` `bg_141`)**: Decoding two `Q4_K` rows (`gate` + `up`) simultaneously inside one simdgroup loop (`gemv_q4_k_paired_glu`) doubles live register pressure and regresses `Gemma 4 E2B Q4_K_M` (`0.985`, `0/4`). Multi-matrix horizontal fusion (`MultiGemv`, `S5.2 P1`) MUST partition different matrices across **threadgroups** (`tg_y` ranges map to matrix `0..M-1`), NEVER by doubling per-lane inner-loop accumulators on complex K-quants.
3. **Small-`cols` Specialization Law (`rsi-17`)**: `gemv_q4_0_half` quarter-block lane splitting (`8` elements/lane instead of `16`) helps when `cols < 512` (`cols = 256` PLE where 16 of 32 lanes are otherwise idle), and MUST be gated on `cols < 512 && (cols & 255) == 0`, never `(cols & 511) != 0`.
4. **Attention Loop Invariance Law (`rsi-16` `bg_129`)**: Splitting `attn_cached_gqa_split_t`'s hot loop into `< hi_hist` and a post-loop single-step `start_pos` branch regresses E2B QAT by `~2.5–4.4%` (`0.975`/`0.956`). Keep a single unified loop over `t` and optimize via `float4`/`half4` vectorized loads and in-kernel last-TG merge.
5. **HostProfile Codec Rate Stability Law (`rsi-18`)**: Editing `src/graph/metal_kernels.metal` changes `KERNELS_MSL`'s hash (`src/graph/metal.rs:656`). Without `HostProfile.codec` carry-over when `codec::source_hash()` is unchanged, candidate binaries re-run `run_codec_bench()` and flip marginal `ReencodePolicy::MaxSpeed` decisions (`IQ4_XS` vs `Q4_0`), confounding `bytes_per_token`.

---

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

## 3. Avant-Garde Priority Queue (Ordered by Expected Roofline Gain & Empirical Proof)

### Phase A — Immediate High-Certainty Unlocks (`rsi-19` & `rsi-20`)

#### 1. `rsi-19` (`Infra` + Scheduler Quiet Gate): HostProfile Codec Carry-Over & `--quiet` Screen Gate
- **Files**: `src/graph/metal.rs` (`700–745`), `autoresearch/central_scheduler.py`
- **Change**:
  1. Re-land the `HostProfile.codec` cache carry-over from `3557243` (`src/graph/metal.rs:708–730`) as its own standalone `infra:` commit: when `KERNELS_MSL` changes so `profile_path` misses, scan `~/.cache/gibbs/host/{tag}-*.json` for any profile with matching `os_build` and `codec.codec_hash == codec::source_hash()` and reuse its `codec` table without re-running `run_codec_bench()`. Also skip persisting a freshly measured `HostProfile` if `tune::foreign_busy()` is true.
  2. Add `--quiet` to `autoresearch/central_scheduler.py run` (waits while holding `sched.lock` until 5 consecutive `IOAccelerator` samples are `<= recommended_QUIET_MAX_PCT`).
- **Verification**: Verify `bytes_per_token` matches `target/loop-base/gibbs` across all 9 models (`837697536, 1474940928, 1328365568, 4493328384, 764923904, 268959744, 765624320, 2179676928, 837697536`), run `loop_guard.sh`, and advance `target/loop-base/gibbs`.

#### 2. `rsi-20` (`Track A0 + B2`): Zero-Tax Compile-Time Template Specialization of `rsi-18` (`experiment/rsi-18-b2-qwen-fusions`)
- **Files**: `src/graph/metal.rs`, `src/graph/metal_kernels.metal`, `src/graph/codec.rs`
- **Problem**: `experiment/rsi-18-b2-qwen-fusions` (`3557243`) already proved `-102` dispatches/tok and **`+1.29%`** on `Qwen3.5 2B` (`430 -> 328` dispatches) plus `-136` dispatches/tok and **`+0.81%`** on `Qwen3.5 9B` (`594 -> 458` dispatches) and `+51%..+89%` on `IQ4_XS` shapes, but lost `~0.5–0.7%` on non-Qwen models because `has_gate` and `vec_scale` were added as runtime branches/buffers to shared `CachedAttention` and `RmsNorm` kernels.
- **Design**:
  1. Start from `experiment/rsi-18-b2-qwen-fusions` (`3557243`).
  2. In `src/graph/metal_kernels.metal`:
     - Keep `attn_cached_gqa_split_t<N_SPLITS>`, `flash_attn_cached_t<NJ>`, and `attn_cached_merge_f32` **100% identical to `96ab660`** (no `gate` buffer parameter, no `has_gate` field in `AttnCachedParams`).
     - Instantiate separate `_gated` template entry points (`attn_cached_gqa_split_gated_t<N_SPLITS, true>`, `flash_attn_cached_gated_t<NJ, true>`, `attn_cached_merge_gated_f32`) that bind `gate [[buffer(14)]]` ONLY when `Step::CachedAttention.gate.is_some()`.
     - Keep `rmsnorm_fused_f32` **100% identical to `96ab660`** (scalar `scale_buf[0]` only), and emit a separate `rmsnorm_fused_vecscale_f32` entry point used ONLY when `Step::RmsNorm.scale` has `is_vec == true`.
- **Screen**: `ab_screen.py --rounds 4 --models "0,1,2,3,4,7"` via `central_scheduler.py run --quiet`. Non-Qwen controls (`0, 1, 4, 7`) must sit at `~1.000` (`>= 0.997`) while `2, 3` show `+1.3%..+2.5%`.

---

### Phase B — Avant-Garde Subgraph Megakernel JITting (`plans/research-sprint-2.md` Phase 5 & Track A)

#### 3. `Track A3-v2`: True In-Kernel Last-TG Atomic `post_norm + residual + scale` Epilogue (`-70` dispatches/tok on Gemma 4)
- **Files**: `src/graph/codegen.rs`, `src/graph/metal_kernels.metal`, `src/graph/metal.rs`
- **Problem**: On Gemma 4 (`Models 0, 1, 7, 8`), every layer has `post_attention_norm` (after `o_proj`) and `post_feedforward_layernorm` (after `down_proj`) before the residual `Add(x, ...)` and optional `layer_output_scale`. Because `RmsNorm` requires the global sum-of-squares across all `rows = 1536` outputs of `o_proj` / `down_proj`, `Step::MegakernelMlp` and `o_proj` currently finish their GEMV and launch a separate `rmsnorm_fused_f32` dispatch (`70` dispatches/tok on E2B, `60` on 26B).
- **Design**:
  - For any `rows <= 4096` projection (`down_proj` inside `Step::MegakernelMlp` and standalone `o_proj`) followed by single-reader `RmsNorm(y, gamma) -> Add(res, .) -> Optional(Mul(., scale))`:
  - Allocate a 2-`uint` zero-initialized device scratch slot `(sum_sq_bits, tg_count)` in `Session`.
  - As each threadgroup (`n_tgs = ceil(rows / rows_per_tg)`, e.g. `48` TGs for `rows = 1536`) finishes its rows, it writes the raw unnormalized row outputs `y[r]` to `out[r]`, reduces `tg_sum_sq = sum(y[r]^2)` in threadgroup memory, atomically adds `tg_sum_sq` to `scratch[0]` (CAS loop on `atomic_uint`), issues `threadgroup_barrier(mem_flags::mem_device)`, and increments `done = atomic_fetch_add_explicit(&scratch[1], 1u, memory_order_acq_rel)`.
  - Only the **last arriving threadgroup** (`done == n_tgs - 1`) computes `inv_rms = rsqrt(as_type<float>(scratch[0]) / float(rows) + eps)`, resets `scratch[0] = 0; scratch[1] = 0`, and loops `for (uint i = tid; i < rows; i += 256)` (`6` floats/thread for `rows = 1536`, already hot in L2/SLC!) to write `out[i] = (out[i] * inv_rms * gamma[i] + res[i]) * post_scale`.
- **Expected gain**: Eliminates **70 dispatches/tok** (`506 -> 436` dispatches on E2B QAT & ctx4k; `509 -> 439` on E2B Q4_K_M; `923 -> 863` on 26B-A4B) and 70 device memory round-trips (`+0.8..+1.4 pp` suite-wide, `+4..+7 tok/s` on E2B QAT).

#### 4. `Track A2`: Threadgroup-Staged `RmsNorm` Prologue Inside Generated & Hand GEMVs (`plans/research-sprint-2.md` S5.2 P2)
- **Files**: `src/graph/codegen.rs`, `src/graph/metal_kernels.metal`, `src/graph/metal.rs`
- **Problem**: `Step::MegakernelMlp` (when `pre_res.is_none()`) and `qkv` / `o_proj` still launch a separate `rmsnorm_fused_f32` dispatch to write `x_norm` to the arena before the GEMV reads it back. Furthermore, inside the GEMV, 8 simdgroups per TG each read `x_norm` from device/L1 memory across their rows.
- **Design**:
  - For `cols <= 4096` (`<= 16 KB` of the `32 KB` Apple GPU threadgroup memory) when `pre_res.is_none()` (a pure `RmsNorm(x, gamma)` feeding 1 projection or `Step::MegakernelMlp` `gate_w` when `gate`+`up` are fused or single), pass `x` and `gamma` directly to the GEMV kernel:
  - The 256 threads in each threadgroup cooperatively compute `sum_sq` over `x[0..cols]`, reduce via `simd_sum` + 8-float shared scratch, barrier once, and populate `threadgroup float shared_x[cols] = x[i] * inv_rms * gamma[i]`.
  - The GEMV inner loop then reads `shared_x` from terabyte/s threadgroup SRAM instead of device memory, and the separate `rmsnorm_fused_f32` dispatch is deleted.
- **Expected gain**: Eliminates `35–70` dispatches/tok across Llama 3.2 1B (`147 -> 115` dispatches), SmolLM2 135M (`273 -> 213` dispatches), Qwen3.5, and Gemma 4 (`+0.6..+1.1 pp`).

#### 5. `Track A4-v2`: Fused Low-Rank Bottleneck Subgraph (`d -> r -> d`, `r <= 256`) + Safe `cols < 512` Quarter-Block `gemv_q4_0_half`
- **Files**: `src/graph/metal.rs`, `src/graph/metal_kernels.metal`
- **Problem**: On Gemma 4 (`E2B QAT`, `E2B Q4_K_M`, `ctx4k`), the 35 Per-Layer Embedding (PLE) blocks execute `Linear(x [1536], inp_gate [256, 1536]) -> Gelu -> Mul(ple_l [256]) -> Linear(proj [1536, 256]) -> RmsNorm(post_norm) -> Add(x) -> Mul(layer_output_scale)`:
  1. `proj` has `cols = 256` (`n_blocks = 8` of `Q4_0`), where `gemv_q4_0_half` (2 lanes per block = 16 active lanes) leaves **50% of every 32-lane simdgroup idle**.
  2. The 4-step sequence launches 3–4 dispatches per layer (`105–140` dispatches/tok) for a 256-float (`1 KB`) intermediate vector.
- **Design**:
  1. Add a 4-lanes-per-block (`8` elements/lane, `packed_ushort2`) branch in `gemv_q4_0_half` gated strictly on **`p.cols < 512u && (p.cols & 255u) == 0u`** so `cols = 256` achieves 100% SIMD lane utilization without touching `cols >= 512`.
  2. Fuse `Epi::GeluMul` (`gelu(y[r]) * aux[r]`) into `inp_gate` GEMV (`-35` dispatches/tok on E2B QAT & ctx4k) and combine `proj` with the `Track A3-v2` last-TG `post_norm + residual + layer_output_scale` epilogue (`-35` more dispatches/tok).
- **Expected gain**: `-70` dispatches/tok and 2x SIMD utilization on all 35 `proj` GEMVs (`+0.5..+0.9 pp`, `+3..+6 tok/s` on E2B QAT & ctx4k).

#### 6. `Track A5`: Threadgroup-Partitioned Heterogeneous `MultiGemv` (`plans/research-sprint-2.md` S5.2 P1)
- **Files**: `src/graph/metal.rs`, `src/graph/metal_kernels.metal`, `src/graph/codegen.rs`
- **Problem**: Unlike `rsi-17` (which doubled per-lane registers by running two matrices in the same simdgroup), independent projections reading the same `x` (`q_proj + k_proj + v_proj`, or `gate + up` before `down`, or Qwen3.5 `in_proj_qkv + in_proj_z + in_proj_b + in_proj_a`) can be launched as **one dispatch** where threadgroup `tg_y < T0` computes matrix 0 (`w0`) and `tg_y >= T0` computes matrix 1 (`w1`). Each threadgroup runs only **one** matrix's loop, so register count per thread is **unchanged**, while dispatch count and encoder overhead drop by 2–4x!
- **Design**:
  - Add a 2-matrix threadgroup-partitioned `gemv_tg_pair_<fmt>` (or dispatch wrapper) for same-codec `(gate, up)` where `tg_y < n_tgs_gate` computes `gate[row]` and `tg_y >= n_tgs_gate` computes `up[row]`, OR for Qwen3.5 `in_proj_b` (`16` rows) + `in_proj_a` (`16` rows) fused with `Sigmoid` / `Softplus(dt_bias) * neg_a` epilogues in 1 dispatch (`-24` dispatches/tok on Qwen3.5 2B, `-32` on Qwen3.5 9B).

---

### Phase C — Inner-Loop Codec Vectorization & Long-Context Attention (`Track B`)

#### 7. `Track B3-v2`: Vectorized `float4`/`half4` GQA Split-K & In-Kernel Last-TG Merge (`Model 8 ctx4k`)
- **Files**: `src/graph/metal_kernels.metal` (`2080–2260`), `src/graph/metal.rs`
- **Problem**: On `Model 8` (`Gemma 4 E2B QAT · 4k context`, `35.8%` roofline), `CachedAttention` takes `~1.3 ms/tok` more than short context because `attn_cached_gqa_split_t` loads `q` (`float`) and `k_cache`/`v_cache` (`half`) with scalar `for (uint d = lane; d < hd; d += 32)` loops (`16` scalar load iterations for `head_dim = 512`, `8` for `head_dim = 256`), and then launches 35 separate `attn_cached_merge_f32` dispatches.
- **Design** (preserving the single unified `t` loop per Law 4!):
  1. Since `head_dim` is always a multiple of `128` on Gemma 4 (`256` sliding, `512` full) and `64`/`128` on Qwen/Llama (`(head_dim & 127) == 0`), cast row pointers to `device const float4*` / `device const half4*` and load 4 contiguous elements per lane per iteration (`d4 = lane; d4 < hd/4; d4 += 32`), using `dot(q4, float4(k4))` and `float4` register accumulators `acc4[d4]`. This cuts load and address-generation instructions by **4x** without splitting the time loop.
  2. Add an atomic arrival counter per `(head_group)` so the last split-K threadgroup merges the `N_SPLITS` partial `(max, sum, out)` tiles in-kernel, eliminating all 35 `attn_cached_merge_f32` dispatches on `ctx4k`.
- **Screen**: `ab_screen.py --rounds 4 --models "0,8,4"` via `central_scheduler.py run --quiet`.

#### 8. `Track B1`: Vectorized `TQ2_0` (`WIDE_TQ2_0` / `TILE_TQ2_0`) & `IQ2_XXS` MoE Expert GEMV (`Models 0, 7, 8`)
- **Files**: `src/graph/codec.rs`, `src/graph/metal/ext/moe.metal`, `src/graph/metal_kernels.metal`
- **Problem**:
  - `Model 0` & `Model 8` (`Gemma 4 E2B QAT`) stream `395.8 MB` of `TQ2_0` shared-layer FFNs (`[12288, 1536]` and `[1536, 12288]`) at `165–194 GB/s` (`57–67%` of ceiling).
  - `Model 7` (`Gemma 4 26B-A4B UD-IQ2_XXS`) streams `IQ2_XXS` MoE expert banks (`MoeExpertLinear` in `src/graph/metal/ext/moe.metal`) at `51.2–52.2%` of roofline.
- **Design**:
  1. Profile `MoeExpertLinear` on `Model 7` with `GIBBS_PROFILE_KERNELS=1` and vectorize its `IQ2_XXS` inner loop (using 32-bit grid lookup tables `kvalues_iq2xxs` + `packed_uint2` loads and shared-memory `x` staging across active `top_k` experts).
  2. In `src/graph/codec.rs`, benchmark a `char4` / `float4` `dot()` formulation of `TILE_TQ2_0` / `WIDE_TQ2_0` via `SHAPE_RATES_FMT=TQ2_0 cargo test --release --lib shape_rates -- --ignored --nocapture` on `(12288, 1536)` and `(1536, 12288)`.

---

### Phase D — Pipelined Command Encoding, GPU Argmax & Cross-Hardware Compiler Verification (`Track C` & `Track D`)

#### 9. `Track C2`: Two-CB First-Micro-Chunk Pipeline (`first = 16, chunk = usize::MAX`) for Models with `96 <= steps < 220`
- **Files**: `src/graph/metal.rs` (`2645–2720`)
- **Problem**: `CB_CHUNK_MIN_STEPS = 220` leaves 147-step models (`Llama 3.2 1B Q4_0` and `bnb 4-bit`, Models 4 & 6) in a single command buffer, so the GPU waits `~220 µs` (`6%` of the `3.59 ms` token time!) while the CPU encodes all 147 dispatches before `commit()`.
- **Design**: When `96 <= steps.len() < 220` and `GIBBS_CB_CHUNK` is unset, split once after step `16` (`first = 16, chunk = usize::MAX` -> exactly **2 command buffers** total). The GPU starts executing steps `0..16` within `~20 µs` while the CPU encodes steps `16..147` in parallel, paying only **one** CB boundary (`~12 µs`) instead of `220 µs` of initial GPU idle time.
- **Screen**: `ab_screen.py --rounds 4 --models "4,5,6,0"` via `central_scheduler.py run --quiet`.

#### 10. `Track D2`: Cross-Hardware Fused-Subgraph Codegen & Clang CUDA/HIP Device Syntax Gate (`src/graph/{codegen,cuda,rocm}.rs`)
- **Files**: `src/graph/codegen.rs`, `src/graph/cuda.rs`, `src/graph/rocm.rs`, `src/model/reencode.rs`
- **Design**:
  1. Extend `gemv_for_dialect` in `src/graph/codegen.rs` so prologue/epilogue-fused kernels (`Epi::UpGeGlu`, `Epi::UpSwiGlu`, `Epi::Gelu`, `Epi::Residual`, threadgroup-staged `RmsNorm` prologue) emit complete, valid `Dialect::Cuda` and `Dialect::Hip` device code alongside `Dialect::Msl`.
  2. Add a unit test in `src/graph/codegen.rs` that invokes `/opt/homebrew/opt/llvm/bin/clang` (when present on the host) with `-x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -fsyntax-only` and `-x hip -nogpuinc -nogpulib --offload-arch=gfx90a -fsyntax-only` (against a minimal 20-line built-in device prelude defining `half`, ` ushort`, `__shfl_xor_sync`, `__shfl_xor`) over **every `QuantFormat::ALL` codec** to prove zero syntax or type errors in CUDA and HIP codegen!
  3. Wire `src/graph/cuda.rs` and `src/graph/rocm.rs` `Step` planners to dispatch the generated `gemv_for_dialect` kernels for all packed quantized weights (`codec(fmt).is_some()`).

---

## 4. Copy-Pasteable Kickoff Prompt for the Next Unbounded RSI Session

```markdown
Run the unbounded Autonomous RSI Loop v3 (`Iterations: unlimited`) on `/Users/keas/code/rust-inference-server` (`main`) following `autoresearch/rsi-avantgarde-loop.md` and `plans/research-sprint-2.md` to maximize `geomean_roofline_pct` on `bench/loop.json` toward >=90% of roofline and lock in >=150.0 tok/s on Gemma 4 E2B QAT across clean ABBA passes.

Strict operating rules:
1. Central Scheduler Only: Every build, test, `shape_rates`, `ab_screen.py`, `loop_guard.sh`, `verify_when_quiet.sh`, or `gibbs --bench` command MUST execute exclusively through `python3 autoresearch/central_scheduler.py run --label <tag> [--quiet] -- <cmd>`. Subagents (`task`) work in isolated git worktrees (`../gibbs-wt/<track>`) and ONLY read/edit/commit code — they NEVER run `cargo` or benchmarks.
2. Start at `rsi-19` (`infra:` commit): Re-land the `HostProfile.codec` cache carry-over (`src/graph/metal.rs:708-730` from `3557243`, plus `tune::foreign_busy()` save guard) and add `--quiet` support to `autoresearch/central_scheduler.py run` so all A/B screens and verifies compare identical `bytes_per_token` on a verified quiet GPU. Verify with `scripts/loop_guard.sh` and update `target/loop-base/gibbs`.
3. Execute `rsi-20` (`Track A0 + B2`): Cherry-pick `experiment/rsi-18-b2-qwen-fusions` (`3557243`) and eliminate its ~0.5-0.7% shared-kernel tax on non-Qwen models by making `HAS_GATE` a compile-time C++ template parameter on `attn_cached_gqa_split_t` / `flash_attn_cached_t` / `attn_cached_merge_f32` and splitting `rmsnorm_fused_vecscale_f32` into its own kernel entry point so gate-free / scalar-scale PSOs compile identically to `96ab660`. Screen on models `0,1,2,3,4,7`, run `loop_guard.sh`, and run `verify_when_quiet.sh`.
4. Continue through the Priority Queue in `autoresearch/rsi-avantgarde-loop.md` §3:
   - `Track A3-v2`: In-kernel last-TG atomic `post_norm + residual + scale` epilogue (`-70` dispatches/tok on Gemma 4).
   - `Track A2`: Threadgroup-staged `RmsNorm` prologue (`shared_x[cols]` for `cols <= 4096`) inside GEMVs.
   - `Track A4-v2`: Safe `cols < 512` quarter-block `gemv_q4_0_half` + fused PLE `inp_gate -> Gelu * ple -> proj` bottleneck (`-70` dispatches/tok on Gemma 4).
   - `Track B3-v2`: Vectorized `float4`/`half4` KV loads + in-kernel last-TG split-K merge in `attn_cached_gqa_split_t` (`ctx4k`).
   - `Track B1`: Vectorized `TQ2_0` (`WIDE_TQ2_0` / `TILE_TQ2_0`) & `IQ2_XXS` `MoeExpertLinear`.
   - `Track C2`: Two-CB micro-first-chunk pipelining (`first = 16, chunk = usize::MAX`) for `96 <= steps < 220` (Models 4 & 6).
   - `Track D2`: Cross-hardware `Dialect::{Cuda, Hip}` fused prologue/epilogue codegen + `/opt/homebrew/opt/llvm/bin/clang -x cuda / -x hip -fsyntax-only` verification test across `QuantFormat::ALL`.
5. Keep `autoresearch/roofline-loop.tsv` and `autoresearch/roofline-loop-notes.md` updated after every iteration, never push during the loop, and refresh `README.md` / `assets/*.svg` via `gibbs_art` on milestone keeps.
```
