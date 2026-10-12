//! Metal programs for decoder `*ForSequenceClassification` bodies: the uncached causal pass over
//! the sequence and the bias-free `score` head on the last row run as one graph on the GPU
//! (graphs are static-shape, so one is built per sequence length; weights upload once).

use super::DecoderBackbone;
use crate::graph::{Graph, Tensor, Weights};
use crate::model::graph_lm::{GraphLm, GraphLmConfig};
use crate::model::tokenizer::Tokenizer;

/// A Metal session holding one backbone's weights and score head.
pub(crate) struct GpuScore {
    #[cfg(target_os = "macos")]
    session: crate::graph::metal::Session,
}

impl GpuScore {
    pub(crate) fn new(weights: &Weights) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            let rt = crate::graph::metal::Runtime::new(&crate::hardware::default_cache_dir()?)?;
            Ok(Self {
                session: crate::graph::metal::Session::new(rt, weights)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = weights;
            Err("gpu-only: no Metal runtime on this platform; run with --backend cpu".into())
        }
    }

    /// Runs `graph` (inputs `token_ids`, `positions`; one `[1, labels]` output) and returns the logits.
    pub(crate) fn run(
        &self,
        graph: &Graph,
        ids: &[u32],
        positions: Vec<u32>,
    ) -> Result<Vec<f32>, String> {
        #[cfg(target_os = "macos")]
        {
            let inputs = vec![Tensor::U32(ids.to_vec()), Tensor::U32(positions)];
            self.session
                .run(graph, inputs)?
                .pop()
                .ok_or_else(|| "seq-cls graph produced no output".to_string())?
                .into_f32()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (graph, ids, positions, Tensor::U32(Vec::new()));
            Err("gpu-only: no Metal runtime on this platform".into())
        }
    }
}

/// [`GraphLm`] (Llama, Qwen2, GPT-NeoX, ...) as a seq-cls backbone: host forward until
/// `attach_score_gpu`, then the Metal program.
pub struct GraphLmSeq {
    lm: GraphLm,
    /// The Metal session and the `[labels, hidden]` score head.
    gpu: Option<(GpuScore, Vec<f32>)>,
}

impl GraphLmSeq {
    pub fn new(lm: GraphLm) -> Self {
        Self { lm, gpu: None }
    }
}

impl DecoderBackbone for GraphLmSeq {
    fn tokenizer(&self) -> &Tokenizer {
        &self.lm.tokenizer
    }
    fn context_len(&self) -> usize {
        DecoderBackbone::context_len(&self.lm)
    }
    fn hidden(&self) -> usize {
        DecoderBackbone::hidden(&self.lm)
    }
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        DecoderBackbone::last_hidden(&self.lm, ids)
    }
    fn attach_score_gpu(&mut self, score: &[f32], labels: usize) -> Result<(), String> {
        let c = &self.lm.config;
        if score.len() != labels * c.hidden_dim {
            return Err(format!("score head is not [{labels}, {}]", c.hidden_dim));
        }
        // Features the graph builder does not emit (BitNet): refuse instead of computing wrong logits.
        if c.act_quant_bits > 0
            || c.use_relu2_mlp
            || self
                .lm
                .layers
                .iter()
                .any(|l| l.attn_sub_norm.is_some() || l.ffn_sub_norm.is_some())
        {
            return Err(
                "gpu-only: BitNet activation quant / sub-norms / relu2 have no Metal seq-cls program; run with --backend cpu"
                    .into(),
            );
        }
        // ponytail: the session holds its own copy of every weight (2x resident with the host copy).
        let mut weights = Weights::new();
        self.lm.build_seqcls_graph(1, score, Some(&mut weights));
        self.gpu = Some((GpuScore::new(&weights)?, score.to_vec()));
        Ok(())
    }
    fn score_gpu(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let (gpu, score) = self.gpu.as_ref().ok_or("no GPU score program attached")?;
        let c = &self.lm.config;
        let n = ids.len();
        if n == 0 || n > c.max_seq_len {
            return Err(format!("{n} tokens: the context holds 1..={}", c.max_seq_len));
        }
        // The host forward clamps out-of-vocabulary ids to the last row.
        let ids: Vec<u32> = ids.iter().map(|&t| t.min(c.vocab_size as u32 - 1)).collect();
        let graph = self.lm.build_seqcls_graph(n, score, None);
        gpu.run(&graph, &ids, (0..n as u32).collect())
    }
}

/// Deterministic pseudo-random values in `[-amp, amp]`.
pub(crate) fn fill(len: usize, seed: usize, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7 + seed * 131) as f32) * 0.37 + (i % 13) as f32).sin() * amp)
        .collect()
}

/// A tiny (hidden 32, 2 layers, 2 heads) [`GraphLm`] with weights large enough that attention and
/// the MLP matter. `variant` 0: Llama-style (RMSNorm, full RoPE, SwiGLU); 1: GPT-NeoX-style
/// (LayerNorm with biases, GELU, parallel residual, 25% rotary, no learned positions); 2:
/// GPT-Neo-style (LayerNorm, GELU, learned positions, no RoPE). `fmt` stores the projections.
pub(crate) fn tiny_graph_lm(variant: u8, fmt: crate::hardware::QuantFormat) -> GraphLm {
    use crate::model::layers::QuantizedLinearLayer;
    let (h, inter, ctx) = (32, 64, 48);
    let ln = variant > 0;
    let cfg = GraphLmConfig {
        vocab_size: 64,
        hidden_dim: h,
        intermediate_dim: inter,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        max_seq_len: ctx,
        use_standard_layernorm: ln,
        use_gelu_mlp: ln,
        use_parallel_residual: variant == 1,
        rotary_pct: match variant {
            0 => 1.0,
            1 => 0.25,
            _ => 0.0,
        },
        ..GraphLmConfig::smollm_1m()
    };
    let mut lm = GraphLm::new(cfg);
    let lin = |name: &str, i: usize, o: usize, seed: usize, amp: f32| {
        QuantizedLinearLayer::from_f32(name, i, o, &fill(i * o, seed, amp), fmt).unwrap()
    };
    for (l, layer) in lm.layers.iter_mut().enumerate() {
        layer.qkv_proj = lin("qkv", h, 3 * h, 1 + l, 0.3);
        layer.out_proj = lin("out", h, h, 2 + l, 0.3);
        layer.gate_up_proj = lin("gate_up", h, if ln { inter } else { 2 * inter }, 3 + l, 0.3);
        layer.down_proj = lin("down", inter, h, 4 + l, 0.3);
        layer.norm1 = fill(h, 5 + l, 0.5).iter().map(|v| v + 1.0).collect();
        layer.norm2 = fill(h, 6 + l, 0.5).iter().map(|v| v + 1.0).collect();
        if ln {
            layer.norm1_bias = Some(fill(h, 7 + l, 0.2));
            layer.norm2_bias = Some(fill(h, 8 + l, 0.2));
        }
    }
    lm.embedding = fill(64 * h, 9, 0.5);
    if ln {
        lm.final_norm_bias = Some(fill(h, 11, 0.2));
    }
    if variant == 2 {
        lm.pos_embedding = Some(fill(ctx * h, 10, 0.3));
    }
    lm.final_norm = fill(h, 12, 0.5).iter().map(|v| v + 1.0).collect();
    lm
}

/// The matrix fixture: the tiny Llama-style backbone scored on Metal for three inputs, no CPU
/// compute (the host forward is never run).
pub fn gpu_fixture() -> Result<(), String> {
    if !crate::hardware::available_devices().contains(&crate::hardware::Device::Metal) {
        return Err("no Metal device".into());
    }
    let (labels, h) = (3, 32);
    let mut llm = GraphLmSeq::new(tiny_graph_lm(0, crate::hardware::QuantFormat::FP32));
    llm.attach_score_gpu(&fill(labels * h, 20, 0.5), labels)?;
    for ids in [&[1u32, 5, 9][..], &[3, 3, 7, 11, 2, 40], &[63]] {
        let logits = llm.score_gpu(ids)?;
        if logits.len() != labels || logits.iter().any(|v| !v.is_finite()) {
            return Err(format!("seq-cls logits {logits:?} for {ids:?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::hardware::{available_devices, Device};

    /// Compares `score · last_hidden` (host forward, the parity reference) with `score_gpu` on
    /// each of `seqs`; returns the worst absolute logit difference (asserted <= 1e-3, the
    /// contract's bar: same f32 math on both sides, only GPU transcendentals and summation order
    /// differ), or `None` (loudly) without a Metal device. Reference logits must be non-trivial.
    pub(crate) fn parity(
        llm: &mut dyn DecoderBackbone,
        score: &[f32],
        labels: usize,
        seqs: &[&[u32]],
    ) -> Option<f32> {
        if !available_devices().contains(&Device::Metal) {
            eprintln!("SKIPPED: no Metal device, seq-cls GPU parity not run");
            return None;
        }
        let h = llm.hidden();
        llm.attach_score_gpu(score, labels).unwrap();
        let mut worst = 0.0f32;
        for ids in seqs {
            let want: Vec<f32> = {
                let _g = crate::hardware::cpu_compute::allow();
                let hid = llm.last_hidden(ids).unwrap();
                (0..labels)
                    .map(|l| score[l * h..(l + 1) * h].iter().zip(&hid).map(|(a, b)| a * b).sum())
                    .collect()
            };
            let got = llm.score_gpu(ids).unwrap();
            assert_eq!(got.len(), labels);
            assert!(want.iter().any(|v| v.abs() > 1e-2), "{ids:?}: trivial reference {want:?}");
            let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(err <= 1e-3, "{ids:?}: want {want:?} got {got:?} (abs {err})");
            worst = worst.max(err);
        }
        Some(worst)
    }

    /// 40 and 45 tokens cross the GEMM tile boundary (and m >= 16 batched-GEMM selection).
    fn long(n: u32) -> Vec<u32> {
        (0..n).map(|i| (i * 7 + 3) % 64).collect()
    }

    fn run(variant: u8, fmt: crate::hardware::QuantFormat, labels: usize, name: &str) {
        let score = fill(labels * 32, 20 + variant as usize, 0.5);
        let mut llm = GraphLmSeq::new(tiny_graph_lm(variant, fmt));
        let (l40, l45) = (long(40), long(45));
        let seqs: [&[u32]; 5] = [&[1, 5, 9], &[2], &[63, 0, 63], &l40, &l45];
        if let Some(e) = parity(&mut llm, &score, labels, &seqs) {
            eprintln!("graph_lm {name} max abs err {e:e}");
        }
    }

    #[test]
    fn graph_lm_llama_style_matches_host() {
        run(0, crate::hardware::QuantFormat::FP32, 3, "llama-style");
    }

    #[test]
    fn graph_lm_llama_style_bf16_matches_host() {
        run(0, crate::hardware::QuantFormat::BF16, 3, "llama-style bf16");
    }

    #[test]
    fn graph_lm_neox_style_matches_host() {
        run(1, crate::hardware::QuantFormat::FP32, 2, "neox-style");
    }

    #[test]
    fn graph_lm_learned_positions_match_host() {
        run(2, crate::hardware::QuantFormat::FP32, 2, "learned-positions");
    }

    #[test]
    fn fixture_runs_on_metal() {
        if available_devices().contains(&Device::Metal) {
            gpu_fixture().unwrap();
        }
    }

    #[test]
    fn backbone_without_program_refuses_cleanly() {
        struct NoGpu(Tokenizer);
        impl DecoderBackbone for NoGpu {
            fn tokenizer(&self) -> &Tokenizer {
                &self.0
            }
            fn context_len(&self) -> usize {
                1
            }
            fn hidden(&self) -> usize {
                1
            }
            fn last_hidden(&self, _: &[u32]) -> Result<Vec<f32>, String> {
                Ok(vec![0.0])
            }
        }
        let mut b = NoGpu(Tokenizer::default_bpe());
        let e = b.attach_score_gpu(&[0.0], 1).unwrap_err();
        assert!(e.starts_with("gpu-only:"), "{e}");
    }
}
