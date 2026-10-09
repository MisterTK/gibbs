# GIBBS (formerly Foundry) Inference Engine — Architecture Specification

## 1. System Vision & Core Principles

GIBBS (named after Josiah Willard Gibbs) is an autonomous, hot-reloadable universal inference engine and JIT megakernel runtime engineered in strict Rust 2021. It delivers zero-copy unified memory execution and sustained decode throughput exceeding 1000 tok/s across modern heterogeneous hardware targets, fronted by a standard OpenAI-compatible API surface for chat, completions, images, and decisions.

### 1.1 Luminal-Inspired Architecture Principles

FOUNDRY draws its execution model directly from Luminal compilation principles, treating inference as a functional graph compilation problem rather than model-specific procedural dispatch:

1. **Universal Megakernels over Model Hacks**: Optimization logic is never special-cased per model architecture or layer name. Instead, all neural network workloads decompose into universal compiler primitives: fused online softmax attention, single-pass SwiGLU activation, and unrolled GEMV register accumulators.
2. **Fused Online Softmax**: Attention computation eliminates intermediate $O(S^2)$ attention matrix materialization in memory. By streaming key/value vectors through register tiles while concurrently tracking running maximums ($m$) and normalization denominators ($d$), softmax scaling and reduction are computed in a single pass without extra memory allocations or global synchronization.
3. **Single-Pass SwiGLU**: Feed-forward network (FFN) layers fuse the linear gate projection, up projection, SiLU non-linearity ($x \cdot \sigma(x)$), and elementwise multiplication into a unified register/threadgroup pass. Evaluates `out[i] = silu(gate[i]) * up[i]` without round-tripping intermediate activation tensors through heap buffers or high-latency global memory.
4. **Unrolled Register Accumulators**: GEMV and GEMM execution engines leverage aggressive register blocking and compile-time static loop unrolling. Accumulator registers (e.g., `acc00`..`acc33`, SIMD vector registers) retain partial sums across dimension steps, saturating hardware execution units and maximizing arithmetic intensity while amortizing memory loads.
5. **Zero-Allocation Tensor Passes**: Zero heap allocation (`Vec::new`, `Box`, `malloc`) during forward decode iterations. All activations, working buffers, KV cache slots, and intermediate layer representations reside in persistent, pre-allocated scratchpads (`Scratchpad`, `DeviceBuffer`) or static unified memory arenas. Activations flow directly between functional transforms via registers and unified memory scratchpads.
6. **Hardware-Native Megakernel Emission**: The JIT compiler dynamically synthesizes target-native source code (Apple Metal MSL, Apple Neural Engine MIL/CoreML, Nvidia CUDA C++/PTX, AMD ROCm/HIP, Intel LevelZero/AVX-512, and ARM NEON) on the fly, compiles it into native pipelines, and caches the binary artifact in an RCU-synchronized pipeline cache.
7. **Adaptive Quantization & Weightless Execution**: Support for sub-byte and weightless computing: SolidSF Jesse Weightless RAM-LUT neural computing (zero floating-point multiplications), Cactus Compute Needle Fast Walsh-Hadamard attention/MLP ($O(N \log N)$), and Napier logarithmic arithmetic factorization.

---

## 2. Codebase Organization & Module Layout

The FOUNDRY codebase is partitioned into three distinct architectural planes: the **Data Plane** (high-throughput execution and compilation), the **Control Plane** (server streaming, orchestration, and TUI inspection), and the **Storage Layer** (remote model hub ingestion and multi-cloud persistence).

### 2.1 Architectural Plane Hierarchy

- **Data Plane (`src/model/`, `src/jit/`, `src/hardware/`)**:
  - `src/model/`: Concrete model graphs (`graph_lm.rs`, `jev.rs`, `image.rs`, `vae.rs`, `nano.rs`), quantized computational layers (`layers.rs`), SIMD vector primitives (`simd.rs`), byte-pair encoding (`tokenizer.rs`), and zero-copy manifest parsers (`mod.rs`). Implements zero-allocation forward decode execution.
  - `src/jit/`: Universal JIT Megakernel synthesis & compilation (`mod.rs`, `ultra.rs`). Drives schedule autotuning, dynamic cross-backend kernel generation (Metal MSL, CUDA PTX, HIP, LevelZero, NEON), and RCU-synchronized compiled pipeline management.
  - `src/hardware/`: Hardware detection, topology inspection, and quantization configuration (`QuantFormat`, `QuantMode`, `TargetArch`, `HardwareDetector`). Coordinates execution formats across backends.

- **Control Plane (`src/tui/`, `src/server/`)**:
  - `src/tui/`: Real-time interactive terminal UI dashboard (Ratatui/Crossterm). Provides live telemetry, token throughput metrics, memory utilization graphs, KV cache saturation tracking, and RCU quantization mode switching.
  - `src/server/`: High-performance asynchronous HTTP and SSE streaming server (Axum REST, OpenAI-compatible completions and chat endpoints). Coordinates asynchronous inference requests, stream scheduling, and connection lifecycles.

- **Storage Layer (`src/model/hub.rs`, `src/storage/`)**:
  - `src/model/hub.rs`: Hugging Face Hub model resolution, artifact downloading, parameter budget validation (`DEFAULT_MAX_PARAM_BUDGET`), and multi-cloud caching backends (`LocalDisk`, `S3`, `Gcs`).
  - `src/storage/`: Multi-cloud and local unified storage engine (`UnifiedStorage`, S3, GCS, DISK, and memory-mapped IO). Enables zero-copy virtual tensor paging directly from disk into compute memory.

### 2.2 Directory Tree Layout

```
src/
├── backend/            # Hardware-specific execution backends
│   ├── mod.rs          # ComputeBackend trait, DeviceBuffer, Unified Memory
│   ├── metal.rs        # Apple Silicon Metal 3 compute pipeline emitter
│   ├── ane.rs          # Apple Neural Engine (ANE) acceleration
│   ├── amd.rs          # AMD ROCm/HIP wavefront emitter
│   └── intel.rs        # Intel LevelZero / XMX / AVX-512 engine
├── hardware/           # [DATA PLANE] Hardware topology, detection & capabilities
│   └── mod.rs          # QuantFormat, QuantMode, TargetArch, HardwareDetector
├── jit/                # [DATA PLANE] Universal JIT Megakernel synthesis & compilation
│   ├── mod.rs          # JIT pipeline manager, memory-mapped compiler cache
│   └── ultra.rs        # Luminal-inspired UltraKernelEngine (MSL, CUDA, HIP, LevelZero, NEON)
├── model/              # [DATA PLANE] Model architectures, inference graphs & execution
│   ├── mod.rs          # UniversalModelLoader (GGUF, Safetensors, GGML), ModelManifest
│   ├── layers.rs       # QuantizedLinearLayer, QuantizedWeight, WeightlessJesse
│   ├── graph_lm.rs         # GraphLm (autoregressive decoder), GraphLmConfig, KV cache
│   ├── jev.rs          # JevSystem1Model (non-autoregressive, sub-10ms decision engine)
│   ├── image.rs        # QwenImageModel (Flow Matching DiT, 64-channel latent space)
│   ├── vae.rs          # VaeDecoderGraph (337M-param 5-stage convolutional decoder)
│   ├── tokenizer.rs    # High-throughput Byte-Pair Encoding (BPE) tokenizer
│   ├── simd.rs         # Vectorized NEON/AVX primitives (RMSNorm, SwiGLU, Conv2D, FWHT)
│   ├── hub.rs          # [STORAGE LAYER] HuggingFace Hub resolution & budget validation
│   └── nano.rs         # Reference transformer models
├── quant/              # Advanced quantization & mathematical transforms
│   ├── mod.rs          # QuantEngine, quantize_tensor dispatch
│   ├── nvfp4.rs        # Nvidia NVFP4 & NVFP4_E2M1 micro-scaling formats
│   ├── exl3.rs         # EXL3 3-bit, 4-bit, 5-bit variable-rate packing
│   ├── jesse.rs        # SolidSF Jesse Weightless LUT neural network layer
│   └── needle.rs       # Cactus Compute Needle Walsh-Hadamard Transform & Napier log2
├── storage/            # [STORAGE LAYER] Multi-cloud & local unified storage engine
│   └── mod.rs          # UnifiedStorage, S3, GCS, DISK, and memory-mapped IO
├── server/             # [CONTROL PLANE] Async HTTP / SSE streaming server
│   └── mod.rs          # Axum REST & OpenAI-compatible completions endpoints
└── tui/                # [CONTROL PLANE] Real-time interactive terminal UI
    └── mod.rs          # Ratatui/Crossterm dashboard, live metrics, RCU mode switching
```

---

## 3. Structural Analysis & Communities (Graphify-RS)

According to `graphify-rs` AST graph analysis (1,051 nodes, 1,555 edges across 79 communities):

### 3.1 Architectural God Nodes
1. **`.::mod` (`src/model/mod.rs` & `src/server/mod.rs`)** — Central coordinator across data plane manifests, execution interfaces, and streaming server endpoints (Degree: 40).
2. **`real_inference_and_system1_tests` (`tests/real_inference_and_system1_tests.rs`)** — High-degree integration node exercising cross-plane execution, verified output legibility, and streaming throughput (Degree: 38).
3. **`ultra` (`src/jit/ultra.rs`)** — Universal JIT Megakernel synthesis & pipeline generator covering MSL, CUDA, HIP, LevelZero, and ARM NEON (Degree: 37).
4. **`simd` (`src/model/simd.rs`)** — Vectorized execution primitives (RMSNorm, LayerNorm, single-pass SwiGLU, Fast Walsh-Hadamard Transforms, Conv2D) (Degree: 31).
5. **`tiny` (`src/model/graph_lm.rs`)** — Autoregressive transformer decoder with fused RoPE, GQA, and persistent zero-allocation scratchpads (Degree: 27).
6. **`hub` (`src/model/hub.rs`)** — Storage layer model resolver, cache layout manager, and parameter budget safeguard (Degree: 26).
7. **`layers` (`src/model/layers.rs`)** — Universal QuantizedLinearLayer abstracting FP16, NVFP4, EXL3, and Jesse LUT weight representations behind a zero-copy forward interface (Degree: 23).
8. **`.::main` (`src/main.rs`)** — Application entrypoint coordinating CLI arguments, TUI mission control, server binding, and hardware discovery (Degree: 21).
9. **`exl3` (`src/quant/exl3.rs`)** — Variable-rate sub-byte quantization (3.125 to 5.125 bpw) and fused GEMV kernels (Degree: 20).

### 3.2 Primary Communities & Data Flow
- **Community 0 (System 1 Decision Engine & Server Routing)**: `JevSystem1Model`, boolean/choice/score/plan primitives, non-autoregressive decision graphs, Axum REST & SSE streaming routers.
- **Community 1 & 18 (End-to-End Inference Verification)**: Verification suites covering Pythia, TinyStories, SmolLM2, Ant-5M, Bert-Tiny, and MiniLM models.
- **Community 4 (SIMD Vector Execution)**: Vectorized math (`vector_dot_product`, `vector_rmsnorm_f32`, `vector_swiglu`, `vector_fused_swiglu_inplace`, `vector_hadamard_butterfly_f32`, `spatial_upsample_2x_f32`).
- **Community 7 & 14 (Universal JIT Megakernel Compiler)**: `UltraKernelKey`, `UltraKernelEngine`, Luminal online softmax schedules, unrolled register GEMV emission, Napier log2 factorization.
- **Community 3 & 10 (Diffusion & Multimodal Pipelines)**: `VaeDecoderGraph`, `VaeUpsampleStage`, `QwenImageModel`, 64-channel latent space Flow Matching Euler ODE.
- **Community 9, 19, 28, 30 (Hardware Execution Backends)**: `MetalBackend` (Apple Silicon), `AppleNeuralEngineBackend` (ANE 5D systolic), `AmdBackend` (ROCm/HIP wavefront 64), `IntelBackend` (LevelZero / XMX / AVX-512).
- **Community 13 & 49 (Multi-Cloud Storage & Ingestion)**: `ModelArtifacts`, `StorageBackend`, `HuggingFaceHub`, `UnifiedStorage` (S3, GCS, DISK, and memory-mapped IO).
- **Community 65 & 72 (Control Plane & Telemetry)**: `TuiApp` Ratatui/Crossterm mission control dashboard, live rolling TPS history sparklines, RCU mode hot-swapping.
