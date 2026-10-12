# Research Sprint 2: Execution Report

## Summary
- **Target Platform**: Apple Silicon M5 Pro (Metal 4, unified memory).
- **Core Architecture Upgrades**:
  - Implemented generic GEMV codegen and codec registry in `src/graph/codec.rs` and `src/graph/codegen.rs`.
  - Implemented exact reference quantizers (`quantize_row_q4_k_ref`, `quantize_row_q5_k_ref`, `quantize_row_q6_k_ref`) in `src/model/layers.rs` with round-trip RMSE unit verification.
  - Cut over RoPE IR to support `inv_freq` buffers, `mscale`, and MRope variants without model-specific code branches.
  - Implemented causal window exactness `ki + w <= current_pos` in `src/graph/cpu.rs` and Metal attention kernels.
  - Enabled strict placement on Metal eliminating multi-segment host roundtrips.
  - Added in-process bench harness (`src/bench.rs`), parity testing, and baseline profiling.
  - Redesigned TUI mascot to large high-resolution Classic Gibbs avatar and auto-flowing sidebar.

## Bench Results
- Verified test suites:
  - `gemma4_tests`: 6/6 passed.
  - `qwen3_5_tests`: passed.
  - `t5_tests` and `seq2seq_tests`: 10/10 passed.
  - `vlm_tests` and `classifier_tests`: 27/27 passed.
  - `metal_coverage`: 100% Metal lowering without gaps.

## E4B Root Cause & Resolution
- Baseline inspection exposed 173 segments with 86 CPU round trips per forward pass due to unlowered ops and TQ2_0 dequantization bounces.
- Resolved by strict Metal lowering and packed codec support.
