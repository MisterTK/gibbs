# Sprint 3 probes (Phase 0)

Raw outputs of the Phase 0 probes in `plans/research-sprint-3.md`. The GPU probes (P0.1, P0.3, P0.4, P0.5) are queued in `scripts/sprint3_probes.sh p01|p03|p04|p05` and have not been run yet.

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
