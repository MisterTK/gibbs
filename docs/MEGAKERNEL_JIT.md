> **Status: design notes, not implemented.** The JIT and megakernel code this document described
> was simulated and has been removed. The measured, implemented-or-planned design (megakernel tiers
> T0–T3, the host profile pass, and the shared `gibbs_engine::graph::Graph` IR that replaced
> `OperationGraph`) is in [PLAN.md](../PLAN.md).

# Luminal-Inspired Megakernel JIT Compilation

The Gibbs Engine megakernel JIT compiler synthesizes fused, single-invocation compute kernels designed to eliminate memory bandwidth round-trips to DRAM. Inspired by the compilation paradigms of Luminal, Gibbs collapses entire multi-operation transformer layers into single, persistent threadgroup kernels where activations live exclusively in fast GPU register files and threadgroup scratchpad memory.

---

## 1. Persistent Layer Kernel Fusion Architecture

In conventional inference engines, an autoregressive decoder step executes 6 to 10 distinct compute kernels per transformer layer:
1. `RMSNorm` (write back normalized tokens to DRAM)
2. `QKV GEMM` (read normalized tokens, write Q, K, V activations to DRAM)
3. `RoPE` (read Q, K, rotate in-place, write back)
4. `FlashAttention / FlashDecode` (read Q, read KV cache, compute online softmax, write attention output to DRAM)
5. `Attention Output Projection` (read attn output, write residual sum to DRAM)
6. `FFN Norm` (read residual, write normalized vector)
7. `Gate + Up Projections` (read normalized vector, write gate & up tensors)
8. `SwiGLU Activation` (read gate & up, write activated tensor)
9. `Down Projection + Residual Add` (read activated tensor, write final hidden state)

For memory-bound LLM decoding ($M=1$), every round-trip to DRAM wastes up to 80% of peak memory bandwidth on intermediate traffic.

### The Monolithic Fused Pipeline
Gibbs JIT synthesizes a persistent layer kernel (`monolithic_layer_megakernel`) executing the entire forward pass in one launch:

$$\mathbf{x}_{\text{norm}} = \text{RMSNorm}(\mathbf{x}_{\text{in}}, \mathbf{w}_{\text{norm}})$$
$$\mathbf{q}, \mathbf{k}, \mathbf{v} = \mathbf{x}_{\text{norm}} \mathbf{W}_{QKV}$$
$$\mathbf{q}_{\text{rot}}, \mathbf{k}_{\text{rot}} = \text{RoPE}(\mathbf{q}, \mathbf{k}, \text{pos})$$
$$\mathbf{a}_{\text{out}} = \text{FlashDecode}\left(\mathbf{q}_{\text{rot}}, \mathbf{k}_{\text{rot}}, \mathbf{v}, \text{KV-Cache}\right) \mathbf{W}_O + \mathbf{x}_{\text{in}}$$
$$\mathbf{x}_{\text{ffn}} = \text{RMSNorm}(\mathbf{a}_{\text{out}}, \mathbf{w}_{\text{ffn\_norm}})$$
$$\mathbf{x}_{\text{out}} = \mathbf{a}_{\text{out}} + \left(\text{SiLU}(\mathbf{x}_{\text{ffn}} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x}_{\text{ffn}} \mathbf{W}_{\text{up}})\right) \mathbf{W}_{\text{down}}$$

All intermediate vectors remain staged in:
- **Threadgroup Memory**: `threadgroup float act[4096]` (Apple Metal) or `__shared__ float smem_act[4096]` (CUDA / HIP).
- **SIMDgroup Registers**: Registers maintaining accumulated sums during inline dequantization and GEMM tiles.

---

## 2. Hardware Specialization & Target Architectures

Gibbs targets heterogeneous computing targets through dedicated code generators:

| Hardware Target | Backend / Language | Acceleration Primitives | Key Optimization Strategy |
|---|---|---|---|
| **Apple Silicon (M-Series)** | Metal Shading Language (MSL) | `simdgroup_matrix`, `simdgroup_barrier`, `threadgroup` memory | 32 KB threadgroup memory partitioning, unified memory zero-copy dispatch |
| **Apple Neural Engine (ANE)** | Specialized 5D Tiled Engine | FP16/INT8 hardware matrix multiplication engines | Static 5D tensor shape compilation with DMA descriptor queues |
| **Nvidia GPUs** | CUDA C++ / PTX | Tensor Core WMMA / MMA instructions, `cp.async`, TMA | Asynchronous multi-stage double buffering, register-pressure-conscious scheduling |
| **AMD GPUs** | ROCm HIP C++ | CDNA / RDNA MFMA / WMMA matrix intrinsics | LDS (Local Data Share) layout optimizations with wave32/wave64 dispatch |
| **Intel Arc / Xe / Xeon** | LevelZero / AMX / AVX-512 | Intel Subgroup Matrix / AVX-512 VNNI intrinsics | SLM (Shared Local Memory) staging with AMX tile configuration |
| **ARM64 (Grace / Graviton)** | C++ with NEON Intrinsics | `float32x4_t`, `vld1q_f32`, `vmulq_f32`, `vmlaq_f32` | 128-bit vector register loops with dual-issue fused multiply-accumulate |

### Metal Shading Language (MSL) Emission
The MSL code generator emits native SIMD matrix instructions and handles mixed quantization unpack routines (e.g., NVFP4, FP8_E4M3, GGUF_Q4_K_M, EXL3):

```rust
use foundry_engine::jit::ultra::{emit_msl_monolithic_layer, UltraKernelKey};
use foundry_engine::hardware::QuantFormat;

let key = UltraKernelKey {
    quant_format: QuantFormat::NVFP4,
    hidden_dim: 4096,
    tile_m: 1,      // Single-token decoding
    tile_n: 64,
    tile_k: 32,
    stages: 2,
};

let msl_source = emit_msl_monolithic_layer(&key);
assert!(msl_source.contains("#include <metal_stdlib>"));
assert!(msl_source.contains("threadgroup float act[4096]"));
```

### ARM NEON Intrinsics Emission
For edge and server ARM CPUs, Gibbs synthesizes vector intrinsics with minimal register spilling:

```rust
use foundry_engine::jit::ultra::emit_neon_monolithic_layer;

let neon_source = emit_neon_monolithic_layer(&key);
assert!(neon_source.contains("#include <arm_neon.h>"));
assert!(neon_source.contains("vmulq_f32"));
```

---

## 3. Tile Autotuning Engine

Choosing matrix tile dimensions $(T_m, T_n, T_k)$ requires balancing memory hierarchy limits:
- **Threadgroup / Shared Memory Budget**: Apple Silicon enforces a strict 32 KB threadgroup memory limit per threadgroup.
- **Register Pressure**: Exceeding 64–96 registers per thread introduces register spilling to local memory, causing catastrophic latency spikes.
- **Arithmetic Intensity**: Measured as:
  $$I_{\text{arithmetic}} = \frac{2 \cdot T_m \cdot T_n \cdot T_k}{\text{Bytes Transferred}(T_m, T_n, T_k)}$$

The `autotune_m_series_tiles` and `autotune_persistent_megakernel` functions evaluate the candidate space:

```rust
use foundry_engine::jit::ultra::{autotune_m_series_tiles, autotune_persistent_megakernel};
use foundry_engine::hardware::QuantFormat;

// Autotune for single-token decoding (batch_size = 1)
let (m, n, k) = autotune_m_series_tiles(QuantFormat::NVFP4, 4096, 1);
assert_eq!(m, 1);      // Decoding constrained to M=1
assert!(n >= 32);     // High output tile width to maximize register reuse
assert!(k >= 16);     // Outer accumulation step

// Autotune for batched prefill (batch_size = 32)
let (m_batch, n_batch, k_batch) = autotune_m_series_tiles(QuantFormat::NVFP4, 4096, 32);
assert!(m_batch > 1);  // Prefill uses 2D tiles to maximize GEMM tensor core utilization
```

---

## 4. Persistent Disk Cache (`PersistentDiskCache`)

`PersistentDiskCache` stores generated shader **source** and dispatch descriptors (JSON) on disk; it does not store compiled binaries:

- **Location**: Default path `~/.cache/gibbs/kernels/` (configurable via `GIBBS_KERNEL_CACHE_DIR`).
- **Cache Key**: 64-bit FxHasher digest of the generated shader source code (`hash_shader_source`).
- **Artifacts**: `{hash}.metal` (MSL source) and `{hash}.dispatch.json`.

The executing GPU path is separate: `--backend metal` compiles `src/backend/metal_kernels.metal` at startup with `newLibraryWithSource` (`MetalGraphLm`) and reports compiler errors as load failures.

```rust
use foundry_engine::jit::ultra::{PersistentDiskCache, hash_shader_source, emit_msl_monolithic_layer, UltraKernelKey};
use foundry_engine::hardware::QuantFormat;

let cache = PersistentDiskCache::new();
let key = UltraKernelKey {
    quant_format: QuantFormat::FP8_E4M3,
    hidden_dim: 4096,
    tile_m: 1,
    tile_n: 64,
    tile_k: 32,
    stages: 2,
};

// 1. Generate shader source
let source = emit_msl_monolithic_layer(&key);
let hash = hash_shader_source(&source);

// 2. Query cache
if cache.load_source(&hash).is_none() {
    cache.store_source(&hash, &source).unwrap();
}
```

---

## 5. Usage within the Engine Architecture

The Megakernel JIT sits between the high-level `OperationGraph` IR and physical device execution:

```
[Model Checkpoint] (Safetensors / GGUF / ONNX / Bin)
       │
       ▼ (WeightReader & ConfigExtractor)
[Unified ModelConfig & Weights]
       │
       ▼ (Dynamic Architecture Fingerprinting)
[OperationGraph IR] (MLA / MoE / RoPE / Soft-Capping identified)
       │
       ▼ (Megakernel Compiler & Autotuner)
[MegakernelConfig: (Tm, Tn, Tk, Stages)]
       │
       ├── Cache Hit? ──► Load ~/.cache/gibbs/kernels/{hash}.metal + .dispatch.json
       └── Cache Miss ──► Synthesize MSL/NEON ──► Store source
       │
       ▼
[Persistent Monolithic Execution] (RMSNorm + QKV + RoPE + FlashDecode + SwiGLU in 1 Launch)
```
