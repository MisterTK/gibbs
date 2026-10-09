# Universal Model Loader & Ingestion Architecture

Gibbs Engine implements an architecture-agnostic model ingestion pipeline designed to decouple model checkpoint formats from execution graph generation. Rather than hardcoding bespoke loading logic per model name or vendor repo, Gibbs ingests weights from diverse checkpoint formats into a unified `WeightReader` interface and parses hyper-parameters via `ConfigExtractor`. Model graphs are described in the shared IR (`gibbs_engine::graph::Graph`, see PLAN.md), and architecture fingerprinting runs as an analysis pass over that IR.

---

## 1. Supported Formats & Parsing Internals

Gibbs natively ingests models from four primary container ecosystems without external Python runtime dependencies:

| Format | Extension | Container Structure | Data Layout & Alignment |
|---|---|---|---|
| **Safetensors** | `.safetensors` | 8-byte little-endian header length $L$, followed by $L$ bytes of UTF-8 JSON metadata, followed by binary tensor payloads | 8-byte aligned zero-copy memory mapped slices |
| **GGUF** | `.gguf` | 4-byte magic (`GGUF`), version (`u32`), tensor count (`u64`), KV count (`u64`), typed metadata KV array, tensor info headers, followed by aligned weight buffers | Configurable padding (default 32 bytes) with quantized block encodings (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0) |
| **ONNX** | `.onnx` | Binary Protocol Buffers (`ModelProto` / `GraphProto`) | Graph node DAG with initializer `TensorProto` arrays (raw byte arrays, float arrays, or quantized pairs) |
| **PyTorch Checkpoint** | `.bin` | Uncompressed ZIP or raw PyTorch Pickle v2/v4 serialization stream | Reconstruction of nested dictionary keys mapping to float/half tensor blobs |

### Format Dispatch (`load_with_format`)
Format resolution is handled by `load_with_format(path, format_hint)`:

```rust
use std::path::Path;
use foundry_engine::model::loaders::{
    load_with_format, WeightReader, TensorMetadata,
    GgufWeightReader, SafetensorsWeightReader, PyTorchBinWeightReader, OnnxWeightReader,
};

// Auto-detects format from extension (.safetensors, .gguf, .onnx, .bin)
let reader: Box<dyn WeightReader> = load_with_format("checkpoints/model.safetensors", None)?;

// Or specify format explicitly:
let gguf_reader = load_with_format("weights.data", Some("gguf"))?;
```

---

## 2. Dynamic Architecture Fingerprinting

Rather than matching against model IDs (e.g. `meta-llama/Llama-3-8B`), Gibbs analyzes graph topology, connectivity, tensor naming semantics, and mathematical operation patterns to discover architectural primitives dynamically.

### Fingerprinted Primitives & Signatures

1. **Multi-Head Latent Attention (MLA)**:
   - *Signature*: Latent low-rank KV compression projections where query/key vectors are decoupled into compressed latent representations:
     $$\mathbf{c}_t^{KV} = \mathbf{x}_t \mathbf{W}_{DKV}, \quad \mathbf{k}_t^C = \mathbf{c}_t^{KV} \mathbf{W}_{UK}, \quad \mathbf{v}_t^C = \mathbf{c}_t^{KV} \mathbf{W}_{UV}$$
   - *Detection*: Discovering projection dimensions where $d_{\text{latent}} \ll d_{\text{hidden}}$ alongside decoupled RoPE key projections $\mathbf{k}_t^R$. Identified in models like DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1.

2. **Mixture of Experts (MoE)**:
   - *Signature*: Presence of router gating networks scoring top-$k$ experts:
     $$G(\mathbf{x}) = \text{Softmax}\left(\text{TopK}\left(\mathbf{x} \mathbf{W}_g, k\right)\right)$$
   - *Detection*: an `Op::Moe` node with more than one expert. Discovered in Qwen1.5-MoE, DeepSeek-V3, Mixtral, and Gemma MoE.

3. **RoPE Layouts (Rotary Position Embeddings)**:
   - *Interleaved (GGUF / NeoX style)*: Adjacent pairs rotated together:
     $$R_{\Theta, m} \mathbf{x} = \begin{bmatrix} x_0 \cos m\theta_0 - x_1 \sin m\theta_0 \\ x_0 \sin m\theta_0 + x_1 \cos m\theta_0 \\ \vdots \end{bmatrix}$$
   - *Half-Split (HuggingFace / LLaMA style)*: Vector split into two halves:
     $$R_{\Theta, m} \mathbf{x} = \begin{bmatrix} \mathbf{x}_{0:d/2} \odot \cos(m\boldsymbol{\theta}) - \mathbf{x}_{d/2:d} \odot \sin(m\boldsymbol{\theta}) \\ \mathbf{x}_{0:d/2} \odot \sin(m\boldsymbol{\theta}) + \mathbf{x}_{d/2:d} \odot \cos(m\boldsymbol{\theta}) \end{bmatrix}$$
   - *Decoupled RoPE*: RoPE applied exclusively to separate positional sub-vectors (e.g. MLA decoupled keys $\mathbf{k}_t^R$).

4. **Activation Functions & Soft-Capping**:
   - SwiGLU: $\text{SwiGLU}(\mathbf{x}) = (\mathbf{x} \mathbf{W}_{\text{gate}} \odot \sigma(\mathbf{x} \mathbf{W}_{\text{gate}})) \cdot (\mathbf{x} \mathbf{W}_{\text{up}})$
   - Attention Logit Soft-Capping (Gemma 2):
     $$S_{ij} = \text{cap} \cdot \tanh\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k} \cdot \text{cap}}\right)$$
   - Sliding Window Attention: Attention spans bounded by sliding window size $W$ (e.g., Qwen-VL, Mistral, Gemma).

### The `ArchitectureFingerprint` Struct
Extracted via `fingerprint_graph(&graph)` from a `gibbs_engine::graph::Graph`:

```rust
pub struct ArchitectureFingerprint {
    pub family: String,                     // "llama", "deepseek_v3", "qwen2", "glm", "gemma", "custom"
    pub has_mla: bool,                      // Multi-Head Latent Attention
    pub has_moe: bool,                      // Dynamic Mixture of Experts routing
    pub expert_count: usize,                // Total routed experts (0 if dense)
    pub rope_variant: String,               // "interleaved", "half_split", "none"
    pub has_swiglu: bool,                   // Gated SwiGLU FFN
    pub sliding_window: Option<usize>,       // Sliding window context span
    pub attention_soft_capping: Option<f32>,// Logit soft-capping scale (e.g. 50.0)
}
```

---

## 3. Step-by-Step Ingestion & Extraction Examples

### Ingesting Weights with `WeightReader`
The `WeightReader` trait exposes both dequantized FP32 views (for CPU verification / graph construction) and native quantized byte slices (for zero-copy GPU compilation):

```rust
use foundry_engine::model::loaders::{load_with_format, WeightReader, TensorMetadata};

fn inspect_and_extract_weights(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut loader = load_with_format(path, None).map_err(|e| e)?;

    // 1. Inspect header metadata
    let metadata = loader.parse_header().map_err(|e| e)?;
    for (key, val) in metadata.iter().take(5) {
        println!("Metadata: {} = {}", key, val);
    }

    // 2. Enumerate tensors
    let tensors: Vec<TensorMetadata> = loader.list_tensors().map_err(|e| e)?;
    for tensor in &tensors {
        println!(
            "Tensor: {:<40} Shape: {:?} DType: {:<8} ({} bytes)",
            tensor.name, tensor.shape, tensor.dtype, tensor.byte_len
        );
    }

    // 3. Load dequantized FP32 vector
    if let Some(target) = tensors.first() {
        let fp32_weights: Vec<f32> = loader.load_tensor(&target.name).map_err(|e| e)?;
        println!("Loaded {} elements for {}", fp32_weights.len(), target.name);

        // 4. Load raw quantized payload
        let raw_bytes: Vec<u8> = loader.load_tensor_quantized(&target.name).map_err(|e| e)?;
        println!("Loaded {} raw quantized bytes for {}", raw_bytes.len(), target.name);
    }

    Ok(())
}
```

### Parsing Model Hyperparameters with `ModelConfig`
`ModelConfig` normalizes disparate naming schemas (such as `n_embd`, `hidden_size`, `dim`, or `intermediate_size` vs `n_inner`) into a canonical configuration:

```rust
use foundry_engine::model::loaders::config::{
    ModelConfig, ConfigExtractor, HuggingFaceConfigExtractor, GgufConfigExtractor, OnnxConfigExtractor,
};
use serde_json::json;

// 1. Extracting from HuggingFace config.json
let hf_json = json!({
    "vocab_size": 151936,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 22016,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "model_type": "qwen2"
});
let extractor = HuggingFaceConfigExtractor { config_json: hf_json };
let config: ModelConfig = extractor.extract_config().unwrap();

assert_eq!(config.hidden_dim, 4096);
assert_eq!(config.num_layers, 32);
assert_eq!(config.num_heads, 32);
assert_eq!(config.head_dim, 128); // 4096 / 32
assert_eq!(config.intermediate_dim, 22016);
```

### End-to-End: Loaded Decoder to Fingerprint

```rust
use foundry_engine::model::fingerprint::{fingerprint_graph, ArchitectureFingerprint};
use foundry_engine::model::GraphLm;

fn fingerprint_decoder(model: &GraphLm) -> ArchitectureFingerprint {
    // GraphLm::to_graph describes the loaded decoder (norms, fused QKV, RoPE, attention, MLP)
    // in the shared IR; fingerprinting inspects its ops and SSA connectivity.
    let fingerprint = fingerprint_graph(&model.to_graph());
    println!("Detected Architecture Family: {}", fingerprint.family);
    println!("MLA: {}, MoE: {} (experts: {}), RoPE: {}",
        fingerprint.has_mla,
        fingerprint.has_moe,
        fingerprint.expert_count,
        fingerprint.rope_variant
    );
    fingerprint
}
```
