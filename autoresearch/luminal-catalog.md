# Luminal reference catalog

- luminal main: `/Users/keas/code/references/luminal` @ `77d3060f01dc9d6f649687cf16a651e3f288aa0c` (shallow clone; abbreviated `main:`)
- luminal 0.2.0: `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/luminal{,_metal}-0.2.0` (abbreviated `0.2:`)
- llama.cpp: `/Users/keas/code/references/llama.cpp` (semantic reference for MoE graphs only)

This catalog supersedes the S8.3 labels in `plans/research-sprint-2.md`, which mislabel `matmul.rs:79-103`
(matvec dispatch, not gate/up fusion), `binary.rs:560-645` (embedding gather, not LM-head fusion) and
`generic.rs:256-308` (DepthFirst is a total-order chain, not a level DAG).

`original` = no luminal precedent; design is ours. Status cites `autoresearch/roofline-loop.tsv` rows; Sprint 3 cites `plans/research-sprint-3.md` items.

| Cand | Idea | Reference (file:line) | Note | Status (ledger) | Sprint 3 |
|:--|:--|:--|:--|:--|:--|
| L1 | GPU TopK + GatherCols, host sampling over 64 candidates | original | 0.2 `binary.rs:559-622` is only an embedding gather | measured per block, unverified (`v2-33-blk`); host argmax kept (17) | 6.5 opt-in GPU argmax |
| L2 | Static (buffer, offset) binding, no per-token map rebuild | 0.2 `storage_buffer.rs:30-231` | liveness clear-sets, exact-size reuse | kept (`PlanValueLoc`, `rsi-43`) | — |
| L3 | x staged in threadgroup memory | 0.2 `kernels/gemv.metal:50-164` (stage :65-117) | MLX GEMV | discarded (`v2-34`, `rsi-3`, `rsi-36`) | — |
| L4 | rows/simdgroup 4–8 | 0.2 `quantized.rs:44-109` | Q8_0 GEMV 4 rows × 2 simdgroups; dispatch :174-177 | kept as tuner variants `Gen(1,2,4)` (1, `rsi-37`, `rsi-41`) | 6.1 64-thread K-quant variant |
| L5 | planar (scales plane + aligned quants plane) repack | 0.2 `quantized.rs:324-382`; main `crates/luminal_metal/src/layouts.rs:1-48` | layout-specialized lowering | kept for TQ2_0 (26); discarded for Q3_K/Q5_K (`v2-41-blk`) | — |
| L6 | RmsNorm prologue into GEMV | original | 0.2 `unary.rs:435-569` is standalone StdNorm | discarded (28) | — |
| L7 | elementwise chain fusion; Add→RmsNorm | 0.2 `elementwise_fusion.rs:30-305` (codegen :245-303) | string-substitution fusion | kept (5, `rsi-13`, `rsi-14`); F4 discarded (30) | — |
| L8 | per-head norm→RoPE; RoPE-k into kv_append | 0.2 `unary.rs:1073-1239` (MetalRope), `:1241-1432` (RopeCompiler) | | kept (`rsi-13`); F6 discarded (22, 23) | — |
| L9 | multi-weight GEMV for sibling Linears | original | horizontal multi-output GEMV | pair kept (`rsi-38`, `rsi-43`); F3 multi discarded (29); q/k/v over loader `Mat` neutral, removed (`s3-abba-qkv-mat`) | 1.2 q/k/v over `Mat` (removed) |
| L10 | paired-row GLU epilogue | 0.2 `kernels/steel/gemm` addmm epilogue | weak | discarded (24, `rsi-17`) | 1.1 GeGlu-with-side GEMV epilogue (PLE) |
| L11 | GPU embedding / PLE gather | 0.2 `binary.rs:559-622` | | not logged | — |
| L12 | decode-ahead, one cmdbuf in flight | 0.2 `command_buffer.rs:21-202` | rank-based command-buffer grouping | discarded (`v2-35-blk`); chunked CBs kept (21, `rsi-43`) | — |
| L13 | function-constant shape specialization | 0.2 `elementwise_fusion.rs:245-303` | shape-baked codegen | measured per block, unverified (`v2-38-blk`) | 6.6 ABBA (GPU, deferred) |
| L14 | split-K flash-decode threshold | 0.2 `unary.rs:991-1071` (Softmax) | weak | kept (2, 8, 9) | — |
| L15 | f16 KV cache | original | | discarded as default (`s3-abba-kv-f16`, −3.82 pp); opt-in `GIBBS_KV_DTYPE=f16` | 1.4 port |
| L16 | DFS step order + liveness arena | 0.2 `generic.rs:255-308` + `storage_buffer.rs:30-231`; main `arena.rs:1-24` | | in use (`allocate`, `src/graph/metal.rs`) | Phase 2 alias-root liveness (removed with views) |
| L17 | MTLResidencySet | main `crates/luminal_metal/src/runtime.rs:91` (mentions residency set; no impl found in `device.rs`) | weak | not tried; llama.cpp `NO_RESIDENCY` showed no effect (notes, iteration 6) | 6.7 not pursued |
| L18 | search over fusion alternatives, on-device timing | main `crates/luminal_metal/src/search.rs:116-164` (measure), `:212` (search_implementations), `finalists.rs:11-50`; `src/search_support.rs:385-460` (genome sampling), `src/extraction.rs:220-590` | e-graph search | per-GEMV tuner kept (`rsi-12`); F8 fusion search discarded (33) | Phase 4 layer-pattern search |
| L19 | composed symbolic layouts (zero-copy views, destination-passing concat) | main `crates/luminal_metal/src/kernels.rs:515-545`, `layouts.rs`; `ops/mul_reduce_sum/mod.rs:72-80` (`alias_info`) | | partial: last-axis half split only (`slice_half`); row-strided views neutral, removed (`s3-abba-views`) | Phase 2 views (removed) |
| L20 | equality saturation over the op graph (egglog) | main `crates/luminal_metal/src/saturation.rs:19-46`, `egraph_postpass.rs` | | not tried | Phase 3 |
| E1 | persistent per-layer kernel | original | | discarded (14) | Phase 5 scoped (opt-in one-dispatch PLE; DeltaNet per-head chain) |
| E2 | ICB replay | original | | discarded (`v2-39-blk`) | — |
| E3 | weights in execution order | original | | measured per block, unverified (`v2-36-blk`) | 6.6 ABBA (GPU, deferred) |
| E4 | combine near-misses | n/a | | discarded (25) | — |

Sprint 3 state: every item in the Sprint 3 column is implemented on branch `sprint3`, and none has been measured on the GPU. Knobs, defaults and CPU evidence are in the status table of `plans/research-sprint-3.md`; ledger rows are `s3-*` in `roofline-loop.tsv`. L20 replaced the greedy planner outright, with plans byte-identical on the suite rows checked.

Other anchors: 0.2 `matmul.rs:79-103` matvec dispatch, compiler :196-410; 0.2 `generic.rs:88-169` CSE,
:233-253 RemoveUnusedNodes, :382-574 ArithmeticElimination; main `src/egglog_core/`,
`crates/luminal_metal/src/egraph_postpass.rs:31-134`, `kernels.rs:494-748` (elementwise/reduce codegen),
`ops/mul_reduce_sum/mod.rs`.
