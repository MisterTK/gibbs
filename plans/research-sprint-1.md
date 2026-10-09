# Research Sprint 1: Universal Model Loader & Hyper-Advanced JIT Megakernel Compilation

**Vision**: Transform Gibbs from a model-specific inference engine into a **universal neural architecture compiler**. No model-specific hacks. All workloads decompose into Luminal-inspired universal compiler primitives. Load any Safetensors/ONNX/GGUF model once; the JIT synthesizes optimized megakernels unseen at compile time.

---

## Phase 0: Architecture & Abstraction Design

### Goal
Define the core abstraction layers that allow Gibbs to:
1. Parse any model format (Safetensors, GGUF, ONNX, `.bin`, PyTorch, TFLITE).
2. Convert to a unified internal representation (IR).
3. Dispatch to JIT megakernel compilation with zero model-specific branches.

### Deliverables
- **`ModelManifest` Redesign** (`src/model/mod.rs`):
  - Generic `Tensor<T>` abstraction over quantization format (not format-specific `QuantizedWeight<Q4K>`).
  - Architecture-agnostic node graph with universal operation primitives (GEMV, GEMM, SoftmaxAttention, SwiGLU, RMSNorm, etc.).
  - Trait-based weight resolver: `WeightReader { fn load_tensor(&self, name: &str) -> Tensor; }`.

- **`UniversalModelLoader`** (`src/model/universal_loader.rs` — **NEW**):
  - Single entry point: `UniversalModelLoader::load(path: &str, quant: QuantMode) -> Result<ModelGraph, String>`.
  - Format detection and dispatch:
    - **Safetensors**: Streaming JSON header parse; zero-copy tensor views via memory map.
    - **GGUF**: Streaming KV pair walk; lazy weight dequantization on-demand.
    - **ONNX**: Graph proto walk; op-level lowering to Luminal IR.
    - **`.bin` (PyTorch)**: Pickle parser; unpickled tensors loaded as Safetensors-like views.
  - Explicit error on unsupported architectures (no silent fallbacks or synthetic weights except `gibbs/*` prefixes).

- **`QuantizationFormat` Trait** (`src/hardware/mod.rs` extension):
  - Generic block dequantizer: `fn dequantize_block(&self, block_bytes: &[u8], block_shape: (usize, usize)) -> Vec<f32>;`.
  - Implemented for all formats: `Q4_0`, `Q4_1`, `Q4_K_M`, `Q5_K_M`, `Q8_0`, `NVFP4`, `MXFP8`, `INT8`, `BF16`, `FP16`, `FP32`, `WeightlessJesseLUT`.

- **`OperationGraph` IR** (`src/model/graph.rs` — **NEW**):
  - DAG representing all operations: `{Embedding, MatMul, GEMV, Attention, SwiGLU, RMSNorm, ...}`.
  - Quantization metadata embedded per edge: `{ op: MatMul, weight_format: Q4K, act_format: FP32, quant_axis: PerChannel }`.
  - No model-specific node types. Architecture-agnostic.

---

## Phase 1: Universal Model Loading & Format Abstraction (Parallel)

### Task Group 1A: Safetensors & PyTorch `.bin` Parsing

**Goal**: Load Safetensors and `.bin` weights without model-specific branches.

**Implementation**:
- Create `SafetensorsLoader` and `BinPytorchLoader` structs implementing `WeightReader` trait.
- Parse JSON header; extract all tensors into `ModelManifest` with lazy dequantization hooks.
- Test on 5 models: SmolLM2, GLM-edge, Qwen2-VL, Llama 3, Phi-3.

---

### Task Group 1B: GGUF Format Standardization

**Goal**: Ensure GGUF loading works identically across all quantization formats.

**Implementation**:
- Refactor `parse_gguf` to extract:
  - Tensor metadata: `name`, `shape`, `dtype` (GGML type).
  - Quantization parameters: block scales, minimums, merges/scores for tokenizers.
- Implement `GgufLoader` struct implementing `WeightReader`.
- Test on: stories15M, SmolLM2 GGUF, Qwen, LLaMA variants.

---

### Task Group 1C: ONNX Graph Lowering to Luminal IR

**Goal**: Parse ONNX GraphProto; lower to universal operation graph.

**Implementation**:
- Create `OnnxGraphLowerer` converting ONNX ops to Luminal IR:
  - `Conv`, `MatMul`, `Gemm`, `Add`, `Mul`, `Relu`, `LayerNormalization` → universal ops.
  - Extract initializers (weights) as `Tensor` with detected quantization.
- Test on ONNX Llama, Qwen, Stable Diffusion U-Net.

---

### Task Group 1D: Configuration File Unification

**Goal**: Extract model config from all sources (HF `config.json`, GGUF metadata, ONNX attrs, TFLITE schema).

**Implementation**:
- Create `ModelConfigExtractor` trait:
  - Unified struct `ModelConfig { vocab_size, hidden_dim, num_layers, attn_head_dim, ... }`.
  - Implement extractors for: HF JSON, GGUF KV, ONNX node attrs, TFLITE schema.
- Validate inferred dimensions against tensor shapes; error on mismatch.

---

## Phase 2: Operation Graph IR & Architecture Discovery (Parallel)

### Task Group 2A: Universal Operation Node Graph

**Goal**: Represent all transformer/diffusion operations in a single, model-agnostic DAG.

**Implementation**:
- Define `enum Op`:
  - `Embedding { vocab_size, embedding_dim }`.
  - `GEMV { in_dim, out_dim, quant_format: QuantFormat }`.
  - `Attention { num_heads, head_dim, rope_dim, has_bias, window_size }`.
  - `SwiGLU { hidden_dim, gate_proj_quant, up_proj_quant, down_proj_quant }`.
  - `RMSNorm { eps, layer_idx }`.
  - `FlashDecode { cache_v: i32, cache_k: i32 }`.
  - ... (15+ universal ops).
- Edges carry quantization metadata: `(src_node_id, dst_node_id, quant_format, activation_precision)`.
- Validate DAG acyclicity and that all tensor shapes align.

---

### Task Group 2B: Architecture Detection from Operations

**Goal**: Infer model family (LLaMA, Qwen, GLM, Gemma, etc.) from operation patterns without hardcoded matching.

**Implementation**:
- Implement architecture fingerprinting:
  - Detect RoPE variants (standard vs. interleaved) from `Attention` node patterns.
  - Detect SwiGLU vs. GELU vs. ReLU from activation graphs.
  - Detect Multi-Head Latent Attention (MLA) from compressed KV patterns.
  - Detect DeepSeekMoE from Expert routing ops.
- Store detected architecture in `ModelConfig::detected_arch: String`.
- If mismatch with declared architecture, log warning; use detected.

---

### Task Group 2C: Per-Layer Quantization Discovery

**Goal**: Map quantization format to each operation without re-serializing.

**Implementation**:
- Walk `ModelManifest` tensors; extract (name, format) pairs.
- For each `Op` node, match tensor names to quantization format:
  - `model.layers.0.self_attn.q_proj.weight` → detect Q4_K_M; set op's `quant_format`.
- Build per-layer quantization matrix: `[layer_id][op_type] → QuantFormat`.

---

## Phase 3: Luminal-Inspired Megakernel JIT Engine (Parallel)

### Task Group 3A: Universal Kernel Emission & Shape Specialization

**Goal**: Generate shape-specific kernels dynamically at load time, caching compiled binaries.

**Implementation**:
- Extend `UltraKernelEngine` with operation-level code generation:
  - Input: `Op`, quantization format, target hardware.
  - Output: Native source code (MSL, CUDA, HIP, LevelZero, NEON, MLIR).
  - Specialize tile dimensions `(tile_m, tile_n, tile_k)` based on hardware roofline and operation shape.
- Implement kernel specialization for:
  - GEMV (1 × d_out × d_in): Unroll output rows; stream weight blocks.
  - GEMM (batch_size × d_out × d_in): Tile-matrix multiply with shared memory buffer.
  - Online Softmax Attention: Single-pass streaming without O(N²) matrix.
  - SwiGLU: Fused gate activation and element-wise multiply.
  - RMSNorm + Residual: Single kernel pass.

---

### Task Group 3B: Hardware Abstraction & Multi-Backend Dispatch

**Goal**: Seamless kernel execution across 7 hardware targets with zero source duplication.

**Implementation**:
- Create `KernelTarget` enum: `AppleMetal`, `AppleANE`, `NvidiaCuda`, `AmdRocm`, `IntelLevelZero`, `ArmNeon`, `GoogleTpuXla`.
- Implement `KernelCompiler` trait with per-backend specialization:
  - Metal: MSL compute shader with `threadgroup_matrix`.
  - CUDA: WMMA Tensor Cores + TMA async copy.
  - ROCm: MFMA instructions + LDS staging.
  - ANE: Static tiling with systolic array alignment.
  - NEON: Inline intrinsics with register dequantization.
- Shared kernel cache: `~/.cache/gibbs/kernels/{arch}_{quant}_{tile_m}_{tile_n}_{tile_k}.{ext}` (cubin, metallib, etc.).

---

### Task Group 3C: Operation Fusion & Memory Layout Optimization

**Goal**: Fuse multi-operation sequences into single persistent kernels; eliminate intermediate tensor writes.

**Implementation**:
- Analyze `OperationGraph` for fusion patterns:
  - `RMSNorm + MatMul + RoPE + Attention → Single kernel`.
  - `SwiGLU (gate + up + activation + multiply) → Single pass`.
- For each fused sequence, emit a unified kernel maintaining activations in register/threadgroup memory.
- Benchmark: Measure speedup from fusion on 135M, 1.3B, 8B models.

---

### Task Group 3D: Dequantization Specialization & Block Codec

**Goal**: Inline quantization-format-specific dequantization directly into kernel source.

**Implementation**:
- For each `QuantFormat`, generate dequantization functions callable from kernel code:
  - Q4_K: Extract nibbles, apply block scale + offset.
  - NVFP4: E2M1 exponent/mantissa reconstruction.
  - INT8: Scale by quantization constant.
- Emit dequant functions as inline helpers in generated kernel source.
- Zero runtime dispatch overhead (format baked into kernel at compile time).

---

## Phase 4: Verification & Correctness (Parallel with Phase 3)

### Task Group 4A: Numerical Parity Testing

**Goal**: Ensure Gibbs output matches HuggingFace Transformers exactly for all models & quantization formats.

**Implementation**:
- Create `OracleVerifier` comparing Gibbs forward pass against HF transformers:
  - Test 10+ prompts × 5 models × 6 quantization formats = 300 test cases.
  - Assert logit cosine similarity > 0.9999 (or exact match for FP32).
  - Measure max absolute error and log distribution.
- Test models: SmolLM2, GLM-edge, Qwen2, Llama 3.1, Phi-3, Mistral.

---

### Task Group 4B: Edge Case & Robustness Audits

**Goal**: Ensure loader handles corrupted files, missing tensors, unsupported ops gracefully.

**Implementation**:
- Craft adversarial test files:
  - Truncated Safetensors (missing data payload).
  - Mismatched tensor shapes.
  - Unknown GGML quantization type.
  - ONNX with unsupported OpSet.
- Assert errors are caught with descriptive messages (no panics, no silent fallbacks).

---

### Task Group 4C: Performance Regression Benchmarks

**Goal**: Measure throughput before/after megakernel fusion on real hardware.

**Implementation**:
- Benchmark suite: 5 models × 3 batch sizes × 2 quant formats.
- Metrics: tokens/sec, memory peak, compilation time.
- Target: No regression; 1.5×+ speedup from fusion on decode.

---

## Phase 5: Model-Specific Validation & Documentation (Parallel)

### Task Group 5A: Architecture-Specific Feature Tests

**Goal**: Verify all model-specific features work (RoPE, GQA, MoE, etc.).

**Implementation**:
- Qwen2-VL: Patch projection ViT + 2D RoPE.
- DeepSeek V3: MLA + MoE.
- Gemma: Logit soft-capping, sliding-window attention.
- GLM-edge: Interleaved RoPE.
- Llama 3.1: 1M context, long RoPE.

---

### Task Group 5B: Integration Tests & CI Pipeline

**Goal**: Ensure all loaders pass with Gibbs HTTP and TUI interfaces.

**Implementation**:
- Add to CI: `cargo test --release model_loading` covering all formats and models.
- TUI: Verify model picker dynamically discovers loaders without explicit registration.

---

## Phase 6: Documentation & User Facing APIs

### Goal
- User can run `gibbs --model <any-hf-repo> --quant <format>` and get optimal inference.
- No model-specific config files or manual setup.

### Deliverables
- **API Documentation**: Trait `WeightReader`, `UniversalModelLoader`, `OperationGraph`.
- **User Guide**: "Loading Custom Models", "Adding Quantization Formats", "Extending to New Hardware".
- **README Update**: Clarify that Gibbs works on any architecture via generic JIT.

---

## Success Criteria

| Criterion | Target | Measurement |
|---|---|---|
| **Format Coverage** | Safetensors, GGUF, ONNX, `.bin`, TFLITE | All 5 load and infer correctly. |
| **Model Coverage** | 15+ architectures | SmolLM2, GLM-edge, Qwen, Llama, Phi, Mistral, Gemma, Deepseek, MoE variants tested. |
| **Numerical Parity** | Logit cosine sim > 0.9999 | Vs. HF Transformers on 300 test cases. |
| **Megakernel Fusion** | 1.5×–2.0× decode speedup | Measured on M-series & RTX 4090. |
| **Load-Time Compilation** | < 5 seconds | From disk to first token generation. |
| **Zero Model-Specific Hacks** | 100% generic dispatch | grep for model-name strings in jit/ should be 0. |
| **Error Handling** | All corrupted inputs caught | No panics, descriptive errors. |

---

## Estimated Timeline
- **Phase 0–1**: 2–3 weeks (format parsing, abstraction design).
- **Phase 2–3**: 3–4 weeks (IR design, kernel emission, multi-backend dispatch).
- **Phase 4–5**: 2–3 weeks (verification, benchmarking, integration).
- **Phase 6**: 1 week (documentation, user-facing cleanup).

**Total**: ~9–11 weeks of focused development across a small expert team.
