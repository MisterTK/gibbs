# Luminal reference catalog

- luminal main: `/Users/keas/code/references/luminal` @ `77d3060f01dc9d6f649687cf16a651e3f288aa0c` (shallow clone; abbreviated `main:`)
- luminal 0.2.0: `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/luminal{,_metal}-0.2.0` (abbreviated `0.2:`)
- llama.cpp: `/Users/keas/code/references/llama.cpp` (semantic reference for MoE graphs only)

This catalog supersedes the S8.3 labels in `plans/research-sprint-2.md`, which mislabel `matmul.rs:79-103`
(matvec dispatch, not gate/up fusion), `binary.rs:560-645` (embedding gather, not LM-head fusion) and
`generic.rs:256-308` (DepthFirst is a total-order chain, not a level DAG).

`original` = no luminal precedent; design is ours.

| Cand | Idea | Reference (file:line) | Note |
|:--|:--|:--|:--|
| L1 | GPU TopK + GatherCols, host sampling over 64 candidates | original | 0.2 `binary.rs:559-622` is only an embedding gather |
| L2 | Static (buffer, offset) binding, no per-token map rebuild | 0.2 `storage_buffer.rs:30-231` | liveness clear-sets, exact-size reuse |
| L3 | x staged in threadgroup memory | 0.2 `kernels/gemv.metal:50-164` (stage :65-117) | MLX GEMV |
| L4 | rows/simdgroup 4–8 | 0.2 `quantized.rs:44-109` | Q8_0 GEMV 4 rows × 2 simdgroups; dispatch :174-177 |
| L5 | planar (scales plane + aligned quants plane) repack | 0.2 `quantized.rs:324-382`; main `crates/luminal_metal/src/layouts.rs:1-48` | layout-specialized lowering |
| L6 | RmsNorm prologue into GEMV | original | 0.2 `unary.rs:435-569` is standalone StdNorm |
| L7 | elementwise chain fusion; Add→RmsNorm | 0.2 `elementwise_fusion.rs:30-305` (codegen :245-303) | string-substitution fusion |
| L8 | per-head norm→RoPE; RoPE-k into kv_append | 0.2 `unary.rs:1073-1239` (MetalRope), `:1241-1432` (RopeCompiler) | |
| L9 | multi-weight GEMV for sibling Linears | original | horizontal multi-output GEMV |
| L10 | paired-row GLU epilogue | 0.2 `kernels/steel/gemm` addmm epilogue | weak |
| L11 | GPU embedding / PLE gather | 0.2 `binary.rs:559-622` | |
| L12 | decode-ahead, one cmdbuf in flight | 0.2 `command_buffer.rs:21-202` | rank-based command-buffer grouping |
| L13 | function-constant shape specialization | 0.2 `elementwise_fusion.rs:245-303` | shape-baked codegen |
| L14 | split-K flash-decode threshold | 0.2 `unary.rs:991-1071` (Softmax) | weak |
| L15 | f16 KV cache | original | |
| L16 | DFS step order + liveness arena | 0.2 `generic.rs:255-308` + `storage_buffer.rs:30-231`; main `arena.rs:1-24` | |
| L17 | MTLResidencySet | main `crates/luminal_metal/src/runtime.rs:91` (mentions residency set; no impl found in `device.rs`) | weak |
| L18 | search over fusion alternatives, on-device timing | main `crates/luminal_metal/src/search.rs:116-164` (measure), `:212` (search_implementations), `finalists.rs:11-50`; `src/search_support.rs:385-460` (genome sampling), `src/extraction.rs:220-590` | e-graph search |
| E1 | persistent per-layer kernel | original | |
| E2 | ICB replay | original | |
| E3 | weights in execution order | original | |
| E4 | combine near-misses | n/a | |

Other anchors: 0.2 `matmul.rs:79-103` matvec dispatch, compiler :196-410; 0.2 `generic.rs:88-169` CSE,
:233-253 RemoveUnusedNodes, :382-574 ArithmeticElimination; main `src/egglog_core/`,
`crates/luminal_metal/src/egraph_postpass.rs:31-134`, `kernels.rs:494-748` (elementwise/reduce codegen),
`ops/mul_reduce_sum/mod.rs`.
