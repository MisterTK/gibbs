# FOUNDRY Inference Engine — Engine Specification

## 1. Performance Goals & Throughput Targets

FOUNDRY is designed to maintain extreme low latency and high sustained decode throughput across all operational workloads:

| Workload Family | Latency / Throughput Target | Typical Hardware (Apple M-Series / RTX 4090) | Architecture & Acceleration Mode |
|---|---|---|---|
| **Autoregressive LLM (System 2)** | **> 1,000 tok/s** decode (unrolled GEMV / KV scratchpad) | Sustained decode across TinyStories, Ant-5M, SmolLM2, Qwen 3.8 | 16-element unrolled NEON & CUDA warp MMA GEMV; zero heap allocations |
| **Non-Autoregressive System 1** | **< 80 µs** decision latency SLA | `JevSystem1Model` evaluating boolean, choice, score | Single-pass non-autoregressive classification head; zero Python overhead |
| **Text-to-Image Diffusion (DiT)** | **< 2.5 s / step** (256x256 to 1024x1024) | 64-channel Flow Matching Euler ODE with 5-stage VAE | Spatial cache-tiled 2D convolution (`TiledVaeConv`) with fused attention |
| **Text+Image-to-Image (I2I Edit)** | **< 2.5 s** total edit time | Denoising strength-controlled Euler trajectory | Shared latent representation with progressive multi-stage denoising |
| **Embedding & Bidirectional State** | **< 2.0 ms** per sequence | `bert-tiny`, `all-MiniLM-L6-v2` | Bidirectional Transformer encoder with fused layer norm and pooling |

### 1.1 Autoregressive Decode SLA (> 1,000 tok/s)
- **Token Generation Throughput**: Sustained decode throughput exceeding **1,000 tokens/second** for single-stream generation and scaled batch decode on commodity silicon (Apple M-Series, NVIDIA Ada/Blackwell, AMD Instinct).
- **Inter-Token Latency (ITL)**: Sub-millisecond ITL ($< 1.0\text{ ms/tok}$) to guarantee real-time human interaction and autonomous tool-use loops without perceptible delay.
- **Time-to-First-Token (TTFT)**: Bound by chunked prefill chunk size ($C = 512\text{--}4096$) evaluated in single-pass fused attention kernels, achieving $< 20\text{ ms}$ TTFT on common context prefixes.
- **Zero-Allocation Hot Loop Guarantee**: Once initialized, the decode execution path performs zero dynamic heap allocations (`0 bytes`). All KV state, activation vectors, and intermediate results reuse persistent pre-allocated `PagedKVPool` and threadgroup scratchpads.

### 1.2 System 1 Latency SLA (< 80 µs Decisions)
- **Single-Pass Non-Autoregressive Evaluation**: Replaces multi-step autoregressive generation with direct, constant-time forward evaluations via `JevSystem1Model`.
- **Zero Python/IPC Contention**: Eliminates the $150\text{--}500\mu\text{s}$ operating system scheduler, Python GIL contention, and CUDA graph synchronization overheads inherent in traditional frameworks.
- **Deterministic Sub-80µs Invariants**: Guarantees sub-80µs response latency for safety filtering, schema validation, intent classification, and autonomous agent branching decisions.

---

## 2. Multi-Modal Streaming Pipelines

FOUNDRY implements three unified multi-modal streaming pipelines sharing the common JIT megakernel infrastructure:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-MODAL PIPELINES                                  │
├─────────────────────────┬────────────────────────────┬───────────────────────────┤
│ 1. Text-to-Text         │ 2. Text+Image-to-Text      │ 3. Text+Image-to-Image    │
│    (Streaming LLM)      │    (VLM Understanding)     │    (I2I Generation & Edit)│
├─────────────────────────┼────────────────────────────┼───────────────────────────┤
│ • Chunked Prefill       │ • Patch Embedding Flatten  │ • Cross-Attention Flow    │
│ • Radix Prefix Cache    │ • Vision Tokens -> KV Pool │   Matching Euler ODE      │
│ • Lock-Free ART-Tree    │ • Single-Pass Fused Cross- │ • 64-Channel Latent Space │
│ • SSE / REST Stream     │   Attention Decode         │ • Spatial Tiled VAE Conv  │
│ • >1000 tok/s Hot Loop  │ • Streaming Text Response  │ • Progressive PNG Chunks  │
└─────────────────────────┴────────────────────────────┴───────────────────────────┘
```

### 2.1 Text-to-Text Pipeline (Streaming LLM)
1. **Adaptive Chunked Prefill**: Long prompt sequences are segmented into dynamic chunks ($128\text{--}4096$ tokens) to prevent compute stalls and ensure bounded latency for concurrent decode passes.
2. **Lock-Free Radix Prefix Cache**: The ART-tree prefix cache indexes previously computed KV blocks, reusing physical memory pages and yielding $>70\%$ cache hit rates on repetitive system prompts.
3. **SSE Streaming**: Emits Server-Sent Events (SSE) immediately upon token generation with zero intermediate serialization overhead.

### 2.2 Text+Image-to-Text Pipeline (Multimodal Understanding)
1. **Vision Patch Ingestion**: Ingests image inputs via 2D patch projection, mapping spatial patches into dense token embeddings aligned with the text vocabulary dimension.
2. **Unified Prefix KV Insertion**: Visual patch tokens are inserted directly into the `PagedKVPool` prefix cache, enabling multimodal cross-attention without duplicating activations.
3. **Streaming Text Generation**: Subsequent autoregressive decode executes through the standard >1000 tok/s pipeline, conditioning directly on the cached image-text prefix.

### 2.3 Text+Image-to-Image Pipeline (Multimodal Generation & Editing)
1. **64-Channel Flow Matching DiT**: Operates on a compact 64-channel latent space using continuous-time Flow Matching Euler ODE integration trajectories.
2. **Denoising Trajectory Control**: Text conditioning is injected via cross-attention while image guidance modulates the initial latent point via strength-controlled ODE trajectories ($t_{\text{start}} \in [0.0, 1.0]$).
3. **Spatial Cache-Tiled VAE Decoder (`TiledVaeConv`)**: The 337M-parameter 5-stage convolutional VAE decodes latent blocks using spatial tiling, eliminating out-of-memory errors on high-resolution targets (up to $1024 \times 1024$).
4. **Progressive Preview Streaming**: Emits intermediate step previews and chunked PNG/WebP representations over HTTP chunked streaming during the ODE integration pass.

---

## 3. Multi-Cloud & Local Storage Engine

The storage engine dynamically resolves weights from multiple source protocols:
- **`s3://bucket/path`**: Amazon S3 endpoint with streaming range-get and local disk write-through caching.
- **`gs://bucket/path`**: Google Cloud Storage endpoint with parallel multipart chunk downloads.
- **`disk://path` / Local paths**: Direct `mmap` zero-copy memory mapping or instant file descriptor binding.
- **`hf://org/repo` / HuggingFace Hub**: Automated branch snapshot discovery, metadata parsing, and budget-enforced downloading (`validate_budget`).

### 3.1 Unified Memory Ingestion
All model formats (`.safetensors`, `.gguf`, `.bin`, `.pt`) pass through `UniversalModelLoader`:
1. **Zero-Copy Parsing**: The file header (JSON for Safetensors, binary KV for GGUF) is read without copying tensor data payloads.
2. **Virtual Tensor Directory**: Every weight tensor is cataloged in `ModelManifest` with its `byte_offset`, `byte_len`, `shape`, and `QuantFormat`.
3. **On-Demand Streaming & Dequantization**: Tensors can be read directly into unified memory or dequantized on-the-fly layer-by-layer during inference forward passes.

---

## 4. Megakernel JIT Compilation Specification

Derived from Luminal principles, the JIT synthesis pipeline creates single-pass unified kernels:
- **Fused Online Softmax Attention**: Evaluates query-key dot product, running maximum tracking, softmax normalization, and value reduction in a single register-resident pass without intermediate attention matrix materialization ($O(1)$ memory).
- **Single-Pass SwiGLU / GELU MLP**: Fuses gate linear projection, up linear projection, activation function, and elementwise multiplication into a unified accumulator kernel.
- **Unrolled Register GEMV Accumulators**: Dequantizes sub-byte weight tiles directly in SIMD registers and accumulates partial dot products across warp/threadgroup lanes.
- **Cactus Compute Needle Walsh-Hadamard Transform**: Employs $O(N \log N)$ in-place Fast Walsh-Hadamard Transforms (FWHT) to eliminate activation outliers and diagonalize attention/MLP layers without parameter expansion.
- **SolidSF Jesse Weightless RAM-LUT**: Replaces floating-point matrix multiplications with bitwise hyperplane hashing into compact `i8` lookup tables, achieving ~0.5 bpw zero-multiplier inference.
- **Zero-Allocation Scratchpads**: All execution passes use pre-allocated, persistent `KVPool` and `Scratchpad` memory buffers. No heap allocations occur during decode passes.

---

## 5. TUI Mode & Interactive Operations

When launched with `--tui`:
- **60 FPS Lock-Free Dashboard**: Terminal interface rendered via Ratatui and Crossterm, running on a dedicated thread decoupled from the async inference execution engine.
- **Adaptive On-the-Fly Kernel Inspection**:
  - Live introspection of the active JIT megakernel key (`QuantFormat`, `tile_m`, `tile_n`, `stages`).
  - Real-time reporting of hardware target architecture (`AppleMetal`, `NvidiaCuda`, `AmdRocm`, `IntelLevelZero`, `ArmNeon`, `GoogleTpuXla`).
  - Active dequantization schedule inspection (`DoubleBufferedTMA`, `HadamardButterfly`, `WeightlessTableLookup`, `SharedMemoryStaged`, `InlineRegister`).
  - Online softmax attention schedule tracking (`StreamingRegister`, `SharedMemoryTiled`, `SplitKParallel`).
  - Live arithmetic intensity calculation ($\text{FLOP/byte}$) and dynamic memory bandwidth bus saturation ($0\text{--}100\%$).
  - Full interactive disassembler toggle (`[I]` / `[K]` / `[Tab]`) revealing the synthesized target-native source code on the fly.
- **Dynamic Quantization Hot-Swapping via Lock-Free RCU**:
  - Runtime mutation of weight quantization (`QuantMode`) across NVFP4 Ultra, EXL3 (Aggressive/Balanced/Quality), SolidSF Jesse LUT, Adaptive Hybrid, and TurboSqueeze.
  - Runtime mutation of transform/KV quantization (`QuantFormat`) across Cactus Compute Needle (Hadamard Transform), SolidSF Jesse RAM-LUT, KIVI 2-bit, TurboQuant, and NVFP4.
  - Dedicated instant hot-swap shortcuts: `[H]` for Hadamard Needle, `[J]` for Jesse LUT, `[N]` for NVFP4 Ultra, and `[E]` for EXL3 Balanced.
  - Atomic configuration swaps via `ArcSwap` with zero engine pauses or dropped request connections.
- **Lock-Free Hardware Telemetry**:
  - Rolling throughput meters (TPS, sparklines, TTFT, ITL).
  - Radix prefix cache hit/miss statistics.
  - PagedKV VRAM slab memory utilization and eviction watermark monitoring.
