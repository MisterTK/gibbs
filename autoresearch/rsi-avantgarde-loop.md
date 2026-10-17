# Autonomous RSI Loop v2: Avant-Garde Megakernel JITting, Cross-Hardware Lowering & Sprint-2 Roofline Closure

Run an unbounded (`Iterations: unlimited`) single-lineage Autoresearch loop on `/Users/keas/code/rust-inference-server` (`main`) to push `geomean_roofline_pct` across `bench/loop.json` (and held-out `bench/sprint2.json`) toward the `>=90%` roofline target and confirmed `>=150 tok/s` ABBA mean on Gemma 4 E2B QAT (`unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf`), using **generic graph-IR passes, whole-subgraph JIT codegen, hardware-profile-driven specialization, and cross-backend kernel synthesis**. Never stop between iterations while candidate hypotheses remain.

---

## 0. Current Baseline & Diagnostic Ground Truth (`rsi-14` / `96ab660`)

- **Current promoted binary (`target/loop-base/gibbs`, commit `96ab660`)**:
  - Suite `geomean_roofline_pct`: **`50.08%`** (up from `46.42%` at `rsi-0`), with `Gemma 4 E2B QAT` at `151.15 tok/s` (`cand1`) / `144.01 tok/s` (2-run ABBA mean; `150.8 tok/s` in 4-round screen `bg_120`).
  1. **Multi-dispatch `Step::MegakernelMlp` (`src/graph/metal.rs:3738-3835`) & unlowered Attention Megakernel (`src/graph/passes/megakernel.rs:236`, `metal.rs:359-360`)**: `fuse_megakernel_blocks_pass` only builds `MegakernelKind::Mlp` (`MegakernelKind::Attention` is rejected on both Metal and CPU), and `Step::MegakernelMlp` still issues 3–5 separate Metal dispatches per layer (`pre_res` RmsNorm -> `gate` GEMV -> `up` GEMV + `Epi::UpGeGlu`/`UpSwiGlu` -> `down` GEMV + residual `Add`, or `post_norm` when post-FFN norm exists) and writes `scratch[off_norm]` and `scratch[off_gate]` to device DRAM/SLC instead of keeping `x_norm` in threadgroup memory and fusing paired `(gate, up)` row-pair evaluation in registers.
  2. **Unfused multi-projection GEMVs (`P1` / `S5.2`)**: Heterogeneous or unconcatenated projections sharing the same input vector (`q_proj`/`k_proj`/`v_proj`, `gate_proj`/`up_proj` when formats differ, PLE `inp_gate`, Qwen3.5 `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`) each stream `x` separately and incur a dispatch boundary instead of executing a single generated `MultiGemv` kernel with up to 4 weight pointers + shared threadgroup `x`.
  3. **Prologue / epilogue JIT synthesis (`P2` / `P3` / `P4`, `S5.1–S5.3`)**: `src/graph/codegen.rs` generates standalone codec GEMVs (`gemv_packed_code`), while `Op::Megakernel` (`src/graph/passes/megakernel.rs`) is still a fixed pattern matcher rather than a general `Op::Fused(FusedKernel)` JIT compiler that splices arbitrary prologue expressions (`RmsNorm`, `LayerNorm`, `Add`, scalar/vector `Mul`, `Slice`), paired GLU activations, and epilogues (`Add`, `RmsNorm` post-norm, `SoftCap`) into one compiled kernel per subgraph shape.
  4. **Gemma 4 PLE & post-norm overhead (`src/model/gemma4/text.rs`, `src/graph/metal.rs`)**: Across 35 layers on E2B QAT, `post_ff_ln` (blocking `down` residual fusion) + `ple.inp_gate` + `ple.proj` + `post_layer_norm` + `layer_output_scale` consume ~210 dispatches/token and ~1.1 ms/token.
  5. **Cross-hardware JIT backend parity (`src/hardware/profile.rs`, `src/graph/{cuda,rocm,coreml,level_zero}.rs`)**: `HardwareProfile` detects SIMD width (`32` on Metal/CUDA, `64` on ROCm CDNA/RDNA wave64), threadgroup shared memory, compute units, and systolic tiles, but `codegen.rs` only emits MSL strings and hardcodes `simd_width = 32` / `rows_per_sg` heuristics instead of lowering through a dialect-parameterized kernel IR (`Dialect::{Msl, CudaPtx, Hip, Mil}`) parameterized by `HardwareProfile`.
  6. **Sprint-2 Phase 3/4/6 items still open or partial (`plans/research-sprint-2.md` §Audit addendum)**:
     - Device-side argmax / top-k (`S8.3 #8`) + 2-token decode-ahead queue (`S8.3 #9`) to eliminate host readback of 256K-vocab logits (~1 MB/tok on Gemma 4 / Qwen3.5) and hide inter-token host scheduling latency.
     - True single-Session multi-bucket `GraphProgram` (`n in {1, 8, 32, 128, 512}`, `S4.1`) and codec-dequant-to-half `simdgroup_matrix` / MPP `matmul2d` prefill GEMM (`S8.3 #11`).
     - Weight-format re-encode cache (`~/.cache/gibbs/reencoded/`, `S3.5`) + header-based decode tok/s predictor (`predict_decode_tok_s`, `S3.6`) replacing static `hardware_speed_rank`.

---

## 1. Non-Negotiable Constraints (Scope & Anti-Cheat Guards)

1. **Zero model-specific logic in the compiler**:
   - Every optimization in `src/graph/**`, `src/hardware/**`, and `src/scheduler/**` MUST be a generic rule over `Op`, `DType`, `QuantFormat`, `Codec`, `HardwareProfile`, or DAG topology.
   - `scripts/generic_guard.sh` rejects any diff in `src/graph`, `src/hardware`, or `src/scheduler` containing model/family names (`gemma`, `qwen`, `llama`, `mistral`, `smollm`, `granite`, `deepseek`, `phi`, `lfm`, `flux`, `ltx`, etc.) outside `#[cfg(test)]`.
2. **Frozen benchmark & scoring contract**:
   - NEVER modify `bench/loop.json`, `bench/sprint2.json`, `bench/prompts/*`, `scripts/loop_verify.sh`, `scripts/loop_score.py`, `scripts/loop_guard.sh`, `scripts/generic_guard.sh`, or the timing/roofline formulas in `src/bench.rs`.
3. **Bit-exact / cosine parity**:
   - Every kept commit MUST pass `scripts/loop_guard.sh` (`rc=0`, full `cargo test --release --lib -q`, `codec_parity`, `op_coverage`, `cpu_free_matrix`, `gemma4_tests`, and `--bench-parity bench/loop.json` with `parity: 5 compared` and `min cos >= 0.999`).
4. **Local-only loop**:
   - NEVER run `git push` or modify remote branches during the loop. Keep all work on local `main`, with `target/loop-base/gibbs` and `target/loop-base/COMMIT` tracking the last promoted winner.
5. **Continue numbering from `rsi-16`**:
   - Log every iteration in `autoresearch/roofline-loop.tsv` and `autoresearch/roofline-loop-notes.md` starting at `rsi-16`.
1. **Centralized Single-Resource Scheduler (`autoresearch/central_scheduler.py`)**:
   - Strictly ONE system-resource-intensive job (`cargo build --release`, `cargo test`, `shape_rates`, `ab_screen.py`, `loop_guard.sh`, `verify_when_quiet.sh`, `gibbs --bench`) may run at any time.
   - Subagents (`task`) perform code analysis and surgical edits ONLY; they NEVER run `cargo`, tests, or benchmarks. All builds, tests, screens, and verifications are executed serially by the orchestrator through `python3 autoresearch/central_scheduler.py run --label <tag> -- <cmd>`.
   - `central_scheduler.py` measures the live IOAccelerator idle floor (`min` of samples) and exports `QUIET_MAX_PCT = max(25, floor + 8)` and `GIBBS_TUNE_BUSY_PCT = QUIET_MAX_PCT` automatically so desktop compositor load (~34%) does not cause 3-hour timeouts or discard valid tuning entries.
2. **Stage 1 — Fast Screen (<3 min, via `central_scheduler.py`)**:
   - **Kernel / Codec JIT changes**: `python3 autoresearch/central_scheduler.py run --label screen-rates -- bash -c 'cargo test --release --lib gemv_matches_cpu_for_every_packed_format && SHAPE_RATES_FMT=<FMT> cargo test --release --lib shape_rates -- --ignored --nocapture'` must show `>= +5%` GB/s on target shapes with zero regressions.
   - **Graph / Megakernel / Runtime changes**: `python3 autoresearch/central_scheduler.py run --label ab-screen -- python3 scripts/ab_screen.py target/loop-base/gibbs target/release/gibbs --rounds 4 --models "<targeted_and_control_indices>"`. Require `median_ratio >= 1.015` on targeted model(s) and `>= 0.992` on untouched control model(s).
   - If Stage 1 fails on a quiet GPU: `git revert HEAD --no-edit`, rebuild `cargo build --release -q`, log `discard (screen)` in `autoresearch/roofline-loop.tsv`, and advance immediately.
3. **Stage 2 — Full Guard (`scripts/loop_guard.sh` via `central_scheduler.py`)**:
   - Run `python3 autoresearch/central_scheduler.py run --label loop-guard -- bash -c 'scripts/loop_guard.sh > target/loop/guard.log 2>&1'`. Must exit `0` and print `parity: 5 compared`.
4. **Stage 3 — Full Quiet ABBA Verify (`scripts/verify_when_quiet.sh` via `central_scheduler.py`)**:
   - Run `python3 autoresearch/central_scheduler.py run --label abba-verify -- scripts/verify_when_quiet.sh`.
   - **Contention / Exit Handling**:
     - If `verify_when_quiet.sh` exits `2` ("never quiet") or `4` ("contention-aborted"), OR if any untouched control model has `ratio < 0.985` (>1.5% drift), the run is **INCONCLUSIVE (contended)** — NEVER classify it as `discard`. Re-run once when GPU utilization drops; if contention persists, log `inconclusive` in `autoresearch/roofline-loop.tsv` and keep the candidate commit on a branch or in the `E4` stack queue.
   - **KEEP iff**:
     - (`delta_pp >= max(0.30, noise_pp)` OR targeted models improve by `>= +1.5%` with all untouched controls within `0.990..1.010` and `delta_pp >= +0.20 pp`),
     - `ratios_ok == true` in `target/loop/verify.json`, and
     - `scripts/loop_guard.sh` passed (`rc=0`).
   - **On KEEP**:
     - `cp target/release/gibbs target/loop-base/gibbs && git rev-parse HEAD > target/loop-base/COMMIT`
     - To refresh README numbers after a milestone KEEP, run a full `target/release/gibbs --bench bench/loop.json > autoresearch/bench-readme.jsonl` (never copy `verify.json`, which has a different schema), update `autoresearch/bench-readme.meta.json`, and run `cargo run --release --example gibbs_art -q`.
     - Log `keep` in `autoresearch/roofline-loop.tsv` and `autoresearch/roofline-loop-notes.md`.
   - **On DISCARD** (clean run with controls in `0.985..1.015`, no contention):
     - `git revert HEAD --no-edit`, confirm `git diff $(cat target/loop-base/COMMIT) -- src` is empty, rebuild `cargo build --release -q`, and log `discard` in `autoresearch/roofline-loop.tsv` and `autoresearch/roofline-loop-notes.md`.
     - Park any clean single-family win (`+1.5%+` on targeted models, `+0.15..+0.28 pp` suite-wide, such as `rsi-15` `9c1384c`) in the **`E4` Orthogonal Stack Queue**.
## 3. Avant-Garde Candidate Tracks (Ordered by Expected Roofline Gain)

### Track A — True JIT Subgraph Megakernels (`src/graph/codegen.rs`, `src/graph/passes/fusion.rs`, `src/graph/metal.rs`)

1. **`A1` — Single-Dispatch Paired `(Gate, Up) -> GLU` JIT Megakernel (Same or Mixed Codec)**:
   - **Problem**: `Step::MegakernelMlp` (`src/graph/metal.rs:3799-3828`) still runs `gate_w` GEMV into `scratch[off_gate]` and then `up_w` GEMV reading `scratch[off_norm]` a second time and reading `scratch[off_gate]` from device memory to compute `Epi::UpGeGlu` / `Epi::UpSwiGlu`. On Gemma 4 E2B QAT (`TQ2_0` 12288x1536 and `Q4_0` 6144x1536), Qwen3.5, and Llama, `gate` and `up` have identical `(rows, cols)` and read the exact same `x_norm`.
   - **Design**: Emit a generated paired-projection kernel `gemv_paired_glu_<fmt_g>_<fmt_u>` in `src/graph/codegen.rs` where each simdgroup computes `R` output rows of `gate` AND `R` output rows of `up` simultaneously across the `K` loop:
     - Load `x_norm[k..k+32]` once into registers/threadgroup tile;
     - Decode `gate` block and `up` block at row `r`, accumulate `acc_g[r]` and `acc_u[r]` in registers;
     - After `simd_sum`, lane 0 writes `glu(acc_g[r]) * acc_u[r]` directly to `scratch[off_gate + r]` in **1 dispatch instead of 2**, cutting `x_norm` reads in half, eliminating the `off_gate` intermediate write+read (`35 * 12288 * 4 B = 1.72 MB/tok` on E2B), and removing 35 dispatches/tok on E2B (`32` on Qwen3.5, `16` on Llama 1B, `30` on SmolLM2).
   - **Expected gain**: `+0.8..+1.5 pp` suite-wide; `+4..+7 tok/s` on E2B QAT (`151 -> 156+ tok/s`).

2. **`A2` — Threadgroup-Staged `RmsNorm` Prologue Inside `gemv` / `gemv_paired_glu` (`P2` in `plans/research-sprint-2.md` S5.2)**:
   - **Problem**: Even with `pre_res` fused (`rsi-14`), `Step::MegakernelMlp` and `Step::RmsNorm` still issue a separate `rmsnorm_fused_f32` dispatch before every `qkv`, `gate_up`, and `ple` projection (`~105` dispatches/tok on E2B QAT).
   - **Design**: For `K <= 4096` (fits in `16 KB` of the `32 KB` threadgroup memory), generate a prologue-fused variant where each threadgroup cooperatively computes `inv_rms = rsqrt(mean(x^2) + eps)` into `threadgroup float shared_x[K]` (`x[i] * gamma[i] * inv_rms`) in ~60 cycles using 256 threads, barriers once (`threadgroup_barrier(mem_flags::mem_threadgroup)`), and then every simdgroup reads `shared_x` from `32 KB` L1 threadgroup SRAM at terabyte/s bandwidth instead of device/SLC memory!
   - **Luminal reference**: `luminal_metal/src/kernels/gemv.metal:50-90` + `StdNormCompiler` (`unary.rs:437-671`).
   - **Expected gain**: Eliminates `70–105` dispatches/tok and serves all `R` rows per threadgroup from shared SRAM (`+0.6..+1.2 pp`).

3. **`A3` — Fused Post-Linear `RmsNorm` + Residual Epilogue (`post_ff_ln` / `post_attn_ln` on Gemma 4 & Gemma 2/3)**:
   - **Problem**: On Gemma 4 (`E2B QAT`, `E2B Q4_K_M`, `26B-A4B`, `ctx4k`), every layer has a post-attention RmsNorm (`post_attention_norm`, `cols = 1536`) and post-FFN RmsNorm (`post_feedforward_layernorm`, `cols = 1536`) between `down_proj` / `o_proj` and the residual `Add`. Because `down_proj` (`1536` rows) cannot fuse `Epi::Residual` until after `post_ff_ln` normalizes the 1536-element output vector, `Step::MegakernelMlp` emits an extra dispatch (`post_norm`) on all 35 layers (`70` dispatches/tok across `o_proj` + `down_proj`).
   - **Design**:
     - **Algebraic fold (zero-cost)**: `RmsNorm(y, gamma)` is scale-invariant (`RmsNorm(c * y, gamma) == sign(c) * RmsNorm(y, gamma)`). Check if any constant scales precede `post_norm`.
     - **In-kernel last-TG epilogue**: `down_proj` and `o_proj` on E2B have `rows = 1536`, which at `rows_per_tg = 32` (`Gen(4)`) is only `48` threadgroups (`12` TGs at `Gen(16)`). When each TG writes its 32 outputs to `out[row]`, it also atomically adds `sum_sq_tg` to a zeroed scalar `scratch[0]` and increments an atomic TG arrival counter `scratch[1]`. The **last arriving TG** (`atomic_fetch_add == n_tgs - 1`, only 1536 floats = 6 floats/thread!) computes `inv_rms = rsqrt(scratch[0] / rows + eps)`, normalizes `out[0..rows]` in place with `gamma`, adds `res[0..rows]` (and optional `layer_output_scale`), and zeroes `scratch[0..1]` for the next layer!
   - **Expected gain**: Eliminates **70 dispatches/token** (`35` `post_attn_ln` + `35` `post_ff_ln`) and 70 round-trips on E2B QAT, E2B Q4_K_M, 26B-A4B, and ctx4k (`+0.7..+1.3 pp`, directly boosting E2B QAT well past `155 tok/s`).

4. **`A4` — Fused Low-Rank Adapter / PLE Subgraph (`Linear(x, inp_gate) -> Gelu -> Mul(ple_l) -> Linear(proj) -> RmsNorm -> Add`)**:
   - **Problem**: Generic low-rank bottleneck subgraphs (`d -> r -> d` where inner rank `r <= 256`, such as Gemma 4's Per-Layer Embedding `1536 -> 256 -> 1536` across 35 layers = 140 dispatches/tok) launch 3–4 separate dispatches for a tiny `256`-element intermediate vector (`614 KB` of weights per layer).
   - **Design**: Add a generic subgraph pattern `Op::BottleneckMlp` (or `Step::LowRankBottleneck`) matching `Linear(x, W1 [r, d]) -> Unary(Act) -> Optional(Binary(Mul, aux [r])) -> Linear(W2 [d, r]) -> Optional(RmsNorm) -> Add(res)` whenever `r <= 256`:
     - Dispatch 1 (`r <= 256` rows = 8 TGs): compute `h[0..r] = act(W1 * x) * aux[0..r]`.
     - Dispatch 2 (`d = 1536` rows): stage `h[0..256]` (`1 KB`) in threadgroup memory, compute `y = W2 * h`, and apply the `A3` last-TG `RmsNorm + residual Add + scale` epilogue!
   - **Expected gain**: Shrinks 35 layers x 4 PLE dispatches (`140` dispatches/tok) down to `70` (or `35` if `W1` is computed inside TG prologue since `W1` is only `1536 * 256` Q4_0 = `216 KB`), saving `~350–500 µs/tok` (`+5–8 tok/s` on E2B QAT & ctx4k).

5. **`A5` — Heterogeneous `MultiGemv` Without Weight Concatenation (`P1` in `plans/research-sprint-2.md` S5.2)**:
   - **Problem**: `fuse_qkv_pass` and `fuse_gate_up_pass` (`src/graph/passes/fusion.rs`) refuse to fuse whenever two projections sharing input `x` have different quant formats (e.g., `q_proj` is `Q4_K` while `k_proj`/`v_proj` are `Q8_0` or `Q4_0`, which occurs on layers 0–14 of Gemma 4 QAT/Q4_K_M and on Qwen3.5 `in_proj_qkv` + `in_proj_z` + `in_proj_b` + `in_proj_a`), AND they allocate duplicated concatenated weight buffers in RAM.
   - **Design**: Replace weight-copy concatenation when formats differ (or for small side projections `b`, `a` of 16–32 rows in Qwen3.5) with a multi-pointer `Step::MultiGemv` that binds up to 4 `(weight_buf, fmt, rows)` slices reading the same `x` buffer in one dispatch.
   - **Expected gain**: Eliminates `30–72` dispatches/tok on Qwen3.5 2B/9B and mixed-quant Gemma 4 layers (`+0.4..+0.8 pp`), and stacks cleanly with the `rsi-15` (`9c1384c`) DeltaRule/Conv1d/Attention-gate fusions (`E4` stack).

---

### Track B — Codec & Inner-Loop Micro-Architecture (`src/graph/codec.rs`, `src/graph/codegen.rs`, `src/graph/metal_kernels.metal`)

6. **`B1` — Vectorized `WIDE_TQ2_0` & `WIDE_TQ1_0` Register-Tile Codegen (`src/graph/codec.rs:108-140`, `codegen.rs`)**:
   - **Problem**: On Gemma 4 E2B QAT, the 20 shared-layer FFNs (`gate`, `up` `[12288, 1536]`, `down` `[1536, 12288]`) and `lm_head` (`[262144, 1536]`) are `GGUF_TQ2_0`. While `lm_head` reaches `210 GB/s` (`Gen(4)`), `down` (`1536 x 12288`) and `gate`/`up` (`12288 x 1536`) average `165–194 GB/s` (`55–64%` of ceiling) because each lane in `TILE_TQ2_0` (`groups: 10, segs: 8`) still unpacks ternary bytes via scalar shift-and-mask inside a 4-iteration loop and loads `x` with two `float4` loads per segment.
   - **Design**:
     - Use a `uchar4` / `uint` SIMD ternary LUT or `as_type<char4>((q >> sh) & 0x03030303u) - char4(1)` with `dot(float4(c), x4)` so the Apple GPU's `ffma` pipeline processes 4 ternary weights per instruction without integer-to-float cast chains (`as_type<float>(0x3F800000u | ...)`).
     - Add `rpr = 2` and `rpr = 4` threadgroup `x`-cached codegen variants in `src/graph/codegen.rs` where 8 simdgroups cooperatively load a `256`-float tile of `x` into `threadgroup float shared_x[256]`, then compute `8 * 4 = 32` rows against `shared_x` before advancing the tile, eliminating redundant L1/SLC `x` traffic on `1536 x 12288` (`down_proj`).
   - **Gate**: `SHAPE_RATES_FMT=TQ2_0 cargo test --release --lib shape_rates -- --ignored --nocapture` improves `(1536, 12288)` and `(12288, 1536)` by `>= +8%`.

7. **`B2` — Vectorized `gemv_iq4_xs`, `gemv_iq2_XXS`, and `WIDE_IQ2_XXS` (`src/graph/metal_kernels.metal`, `src/graph/codec.rs`)**:
   - **Problem**: Model 7 (`Gemma 4 26B-A4B UD-IQ2_XXS`, `51.2%` roofline) and Qwen3.5 2B/9B (`IQ4_XS` / `IQ2_M`) spend their decode time in `IQ2_XXS` expert projections (`MoeExpertLinear`) and `gemv_iq4_xs` (`920–990` in `metal_kernels.metal`, which still uses `n_blocks = p.cols / 256` with 1 lane per 256-element block — so for `cols = 1536` or `2048`, **24 to 26 of the 32 SIMD lanes sit idle!**).
   - **Design**:
     - Apply the exact sub-unit lane-splitting transformation from `rsi-14` (`gemv_q4_k` -> 16 sub-units per 256-element block) to `gemv_iq4_xs`: split each 256-element `block_iq4_xs` (`136 B`: `d` fp16 + `scales_h` uint16 + `scales_l[4]` + `qs[128]`) into **8 or 16 sub-units per block** (`packed_uint2` 8-byte nibble loads = 16 elements per lane), giving **100% SIMD lane utilization** on `cols = 1536, 2048, 3072`!
     - Vectorize `MoeExpertLinear` (`src/graph/metal/ext/moe.metal`) for `IQ2_XXS` / `Q4_K` / `Q4_0` so active `top_k` experts read `x` once and decode 32-bit grid words per lane.
   - **Gate**: `SHAPE_RATES_FMT=IQ4_XS` and `SHAPE_RATES_FMT=IQ2_XXS` improve `(6144, 1536)` / `(1536, 6144)` by `>= +10%`.

8. **`B3` — Long-Context (`ctx4k`) GQA Split-K In-Kernel Last-TG Merge (`plans/research-sprint-2.md` S5.2 P4)**:
   - **Problem**: Model 8 (`e2b-qat-ctx4k`, `34.5%` roofline, `7.68 ms/tok` vs `6.32 ms/tok` at short context) spends `~1.35 ms/tok` extra in `CachedAttention` across the 5 full-attention layers (`kv_len = 4031`, `head_dim = 512`) and 30 sliding-window layers (`window = 512`, `head_dim = 256`), plus 35 `attn_cached_merge_f32` dispatches.
   - **Design**:
     - Vectorize the KV cache dot product and value accumulation in `attn_cached_gqa_split_t` (`src/graph/metal_kernels.metal:2080-2170`) using `half4` / `float4` loads (`head_dim` is a multiple of 128/256/512, so each of the 32 lanes can load `float4(q)` and `half4(k)`/`half4(v)` in 4-element vectors instead of scalar `d += 32` loops — 4x fewer load instructions!).
     - Replace the separate `attn_cached_merge_f32` dispatch with an atomic last-TG reduction inside `attn_cached_gqa_split_t` (`FusionCost::merge_in_kernel_wins`), removing 35 dispatches/tok at 4k context.
   - **Expected gain**: Cuts `ctx4k` GPU time by `0.5–0.8 ms/tok` (`+6–12%` on Model 8, `+0.6..+1.1 pp` suite-wide).

---

### Track C — Zero-Copy Device-Side Sampling & Pipelined Decode (`plans/research-sprint-2.md` S6.2, S8.3 #8–#9)

9. **`C1` — GPU Argmax / Top-K Reduction Epilogue on LM Head (`S8.3 #8`)**:
   - **Problem**: Every decode token runs `lm_head` (`vocab = 151,936` on Qwen3.5, `262,144` on Gemma 4, `128,256` on Llama 3.2) writing `0.5–1.05 MB` of `f32` logits, which the CPU then scans on the host to pick greedy `argmax` (or top-k logprobs).
   - **Design**: When greedy decoding (or top-k sampling with `k <= 64` and no host-side grammar/logit-bias processor active), append a 1-dispatch 2-stage GPU reduction (`argmax_f32` or `topk_f32` into a 64-byte shared output slot) inside the same command buffer, and return `DecodeLogits` with the pre-reduced `(best_token_id, top_k_pairs)` plus lazy slice access to the arena logits buffer.

10. **`C2` — Hardware-Profile-Driven Command-Buffer Chunking & Zero-Encoder-Gap Pipelining (`src/graph/metal.rs:2651`, `src/hardware/profile.rs`)**:
    - **Problem**: `host_ms` still costs `0.25–0.34 ms/tok` (`5–13%` of total token time on SmolLM2 135M, Llama 3.2 1B, and E2B QAT) because `run_steps` encodes `147–506` dispatches serially on the CPU thread and only splits command buffers when `steps.len() >= CB_CHUNK_MIN_STEPS` (`220`). On 147-step Llama 3.2 1B (`gpu_ms = 3.59 ms`, `host_ms = 0.30 ms`), the GPU sits completely idle for `~220 µs` while the CPU encodes all 147 steps into a single command buffer before `commit()`.
    - **Design**: Use a single small first chunk (`first = 16` steps, `< 20 µs` CPU encode) followed by the remaining steps in chunk 2 (`chunk = usize::MAX`, so at most **2 command buffers total**, adding only one CB boundary of `~15 µs` instead of 6 CB boundaries of `140 µs`!) whenever `steps.len() >= 96`. Measure on Models 4, 5, 6 (`Llama 1B Q4_0`, `SmolLM2 135M`, `Llama 1B bnb`) via `ab_screen.py`.

---

### Track D — Cross-Hardware Dialect Lowering & Sprint-2 Generalization (`src/hardware/profile.rs`, `src/graph/{codegen,cuda,rocm,coreml}.rs`)

11. **`D1` — Dialect-Parameterized Kernel Codegen (`Dialect::{Msl, Cuda, Hip}` in `src/graph/codegen.rs`) — Separate Mechanical Gate (No Unmeasured Perf Claims)**:
    - **Problem**: `src/graph/codegen.rs` hardcodes Metal Shading Language syntax (`device const uchar*`, `[[buffer(0)]]`, `[[threadgroup_position_in_grid]]`, `simd_sum`, `simd_shuffle_xor`), while `src/graph/cuda.rs` (NVRTC) and `src/graph/rocm.rs` (HIPRTC) lack packed quantized GEMV + fused megakernel codegen. Because this M5 Pro host has no NVIDIA or AMD GPU, CUDA/ROCm changes cannot be judged by the Metal ABBA roofline gate.
    - **Design & Mechanical Verification Gate**:
      - Introduce a zero-overhead `KernelDialect` descriptor in `src/graph/codegen.rs` parameterized by `HardwareProfile` (`simd_width` ∈ `{32, 64}`, address space qualifiers, thread/wave indexing, warp/wave shuffle reductions `__shfl_xor_sync` / `__shfl_xor`, and shared memory barriers).
      - **Keep Gate for Track D1**: (1) Golden structural unit tests in `src/graph/codegen.rs`, `src/graph/cuda.rs`, and `src/graph/rocm.rs` verifying that every `Codec` in `src/graph/codec.rs` emits syntactically complete MSL, CUDA C++, and HIP C++ kernel source from the shared AST without missing bindings or undefined symbols; (2) `gemv_matches_cpu_for_every_packed_format` passes on Metal using the refactored `Dialect::Msl` path with zero change to emitted MSL hashes (`codec_hash()` unchanged so `TuneCache` stays valid!).

12. **`D2` — Held-Out Generalization & Sprint-2 Audit Closure (`bench/sprint2.json`, `plans/research-sprint-2.md` S3.5, S3.6, S6.3)**:
    - Every 5 iterations (meta-checkpoint):
      1. Run `cargo test --release --test op_coverage --test cpu_free_matrix -- --nocapture` to verify 100% GPU coverage across all 21 model families.
      2. Profile the lowest-roofline model in `bench/loop.json` and update the expected-pp ranking table in `autoresearch/roofline-loop-notes.md`.
      3. Check whether any two parked `E4` near-miss commits (starting with `rsi-15` `9c1384c`: Qwen3.5 `DeltaRule`/`CausalConv1dCached`/`CachedAttention` gate + Gemma 4 26B `fold_weightless_rmsnorm_mul`) can be stacked with the current Track A/B candidate to clear `delta_pp >= +0.30 pp`.

---

## 4. Kickoff Command Sequence for Iteration `rsi-16`

1. Read `autoresearch/roofline-loop.tsv:-10` and `autoresearch/roofline-loop-notes.md:-40`.
2. Check idle GPU utilization (`ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'`). Once `<20%` for 5 consecutive samples, run a clean verification pass on `96ab660` (or proceed directly to profiling Track `A1` / `A3` / `B2` / `B3`).
3. Implement the highest expected-`pp` candidate (`A3` in-kernel last-TG `post_norm` epilogue + `rsi-15` `E4` stack, or `A1` paired `(gate, up) -> GLU` single-dispatch GEMV, or `B2` vectorized `gemv_iq4_xs` + `B3` vectorized `attn_cached_gqa_split_t`), add a CPU-vs-Metal unit parity test, run Stage 1 (`ab_screen.py` / `shape_rates`), Stage 2 (`scripts/loop_guard.sh`), and Stage 3 (`scripts/verify_when_quiet.sh`), and repeat indefinitely until the stop rule fires.
