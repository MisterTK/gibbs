# CPU-only gaps after Sprint 2 (G4.3 exhaustion record)

A family may stay `Gated` only with an entry here. One family does.

## DraftersBlock (`avenues::Family::DraftersBlock`, gate reason prefix `builder:`)
- What: DFlash / DFlash2 / DSpark block drafters (`src/model/draft*`, `load_drafter`). On Metal `load_drafter` returns `gpu-only: DraftersBlock …`; `--backend cpu` runs them.
- `required_ops` gaps: none. `explore_node` lowers every op the drafter graph needs at F32 (MetalKernel avenue, `Lowered("Metal kernel")`); the six-avenue ladder is not the blocker.
- Remaining work (smallest): (1) a block-drafter graph program: per-layer K/V projection of tapped target states, grouped dynamic conv, Markov rescoring; (2) arbitrary-layer taps in the Metal target decode program (`DecodeOpts::tap` currently returns only the final-normed row, `src/model/qwen3_5/text.rs`); (3) a tiny-config parity test vs the native forward and a `gpu_fixture` arm in `tests/cpu_free_matrix.rs`.
- Measured numbers that justified no avenue change: none needed (no op is unlowered).

## Sub-paths inside GPU families that still refuse Metal (`gpu-only:` errors, `--backend cpu` works)
| Path | Why |
|:---|:---|
| fastText classifiers (`classifier::from_fasttext`) | host hashing + averaging model, no graph builder |
| `gemma4_unified` embedders, Gemma 4 checkpoints with bidirectional vision attention | W3 builder covers the GGUF/mmproj path and standard vision layers only |
| MTP draft vocabulary | the Metal MTP head scores the full vocabulary (superset of the native 32768-token prefix), so acceptance rates differ slightly |

## Matmul primitive / MPS decision (G4.1-5)
Not measured to the plan's definition: `gemm_peak` (S4.9) was not implemented. Observed: ext `MatMul` 590 / 599 GFLOP/s (512³), 578 (2048³); implicit-GEMM Conv2d 540–590 GFLOP/s. No `MPSMatrixMultiplication` binding was added, so `avenues::matmul_primitive` stays `Rejected`; `MatMul` lowers through `MetalKernel` first.
