//! Gemma 4 sparse layers (26B-A4B; llama.cpp `gemma4` with `ffn_gate_inp`): a dense GeGLU MLP and
//! a routed expert block run side by side on the attention output, each with its own output norm,
//! and their sum goes through the layer's usual `post_feedforward_layernorm`.
//!
//! Router: `rms_norm(x) / sqrt(hidden) * router_scale` → `[experts]` logits; the top-k experts get
//! the softmax over their logits (llama.cpp: softmax over all experts, picked weights renormalized
//! to sum 1), optionally scaled per expert (`ffn_down_exps.scale`). Experts are GeGLU MLPs read
//! from banks that stay in the mapped GGUF. Only the decode graph implements this block; the host
//! forward does not.
//!
//! Reference: llama.cpp `src/models/gemma4.cpp` and `build_moe_ffn`.

use crate::graph::{DType, Graph, Op, ReduceKind, Tensor, ValueId};
use crate::hardware::QuantFormat;
use crate::model::weights::{Mat, Packed, Weights};
use serde_json::Value;

/// `[experts, rows, cols]` expert matrices: blocks mapped from a GGUF, or a bf16 tensor.
enum Bank {
    Packed(Packed),
    Bf16(Mat),
}

impl Bank {
    fn load(w: &Weights, name: &str, shape: [usize; 3]) -> Result<Self, String> {
        let [e, r, c] = shape;
        if let Some(p) = w.packed_gguf(name) {
            if p.shape != shape {
                return Err(format!("{name}: expert bank is {:?}, expected {shape:?}", p.shape));
            }
            return Ok(Self::Packed(p.clone()));
        }
        let m = w.mat(name)?;
        if m.rows * m.cols != e * r * c {
            return Err(format!("{name}: {}x{} values, expected {shape:?}", m.rows, m.cols));
        }
        Ok(Self::Bf16(m))
    }

    /// The `[experts * rows, cols]` weight of an `Op::MoeLinear`; a mapped GGUF bank is shared, not copied.
    fn graph_weight(&self, rows: usize, cols: usize, experts: usize) -> (DType, Tensor) {
        match self {
            Self::Packed(p) => {
                let dtype = match p.format {
                    QuantFormat::FP32 => DType::F32,
                    QuantFormat::FP16 => DType::F16,
                    QuantFormat::BF16 => DType::BF16,
                    f => DType::Quant(f),
                };
                (dtype, Tensor::Mat(Mat::from_packed(p.clone(), experts * rows, cols)))
            }
            Self::Bf16(m) => (DType::BF16, Tensor::BF16(m.data().to_vec())),
        }
    }
}

/// Gate and up of every expert: two banks, or one `[experts, 2 * ff, hidden]` bank (gate first).
enum GateUp {
    Split(Bank, Bank),
    Fused(Bank),
}

pub(crate) struct MoeBlock {
    router: Mat,
    /// `ffn_gate_inp.scale` `[hidden]`.
    router_scale: Vec<f32>,
    pre_norm_2: Vec<f32>,
    post_norm_1: Vec<f32>,
    post_norm_2: Vec<f32>,
    gate_up: GateUp,
    down: Bank,
    /// Per-expert multiplier of the expert output (`ffn_down_exps.scale`).
    down_scale: Option<Vec<f32>>,
    experts: usize,
    top_k: usize,
    ff: usize,
}

impl MoeBlock {
    /// Whether config `c` makes every layer sparse.
    pub(crate) fn enabled(c: &Value) -> bool {
        c["enable_moe_block"].as_bool().unwrap_or(false)
    }

    /// The block of layer `lp` (`model.language_model.layers.N`).
    pub(crate) fn load(w: &Weights, lp: &str, c: &Value, hidden: usize) -> Result<Self, String> {
        let get = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("text_config.{k} missing"))
        };
        let (experts, top_k, ff) = (get("num_experts")?, get("top_k_experts")?, get("moe_intermediate_size")?);
        if top_k == 0 || top_k > experts {
            return Err(format!("top_k_experts {top_k} is not within 1..={experts}"));
        }
        let router = w.mat(&format!("{lp}.router.proj.weight"))?;
        if (router.rows, router.cols) != (experts, hidden) {
            return Err(format!(
                "{lp}.router.proj.weight is {}x{}, expected {experts}x{hidden}",
                router.rows, router.cols
            ));
        }
        let ex = |part: &str| format!("{lp}.mlp.experts.{part}");
        let gate_up = if w.has(&ex("gate_up_proj")) {
            GateUp::Fused(Bank::load(w, &ex("gate_up_proj"), [experts, 2 * ff, hidden])?)
        } else {
            GateUp::Split(
                Bank::load(w, &ex("gate_proj"), [experts, ff, hidden])?,
                Bank::load(w, &ex("up_proj"), [experts, ff, hidden])?,
            )
        };
        let sc = format!("{lp}.router.per_expert_scale");
        Ok(Self {
            router,
            router_scale: w.vec_n(&format!("{lp}.router.scale"), hidden)?,
            pre_norm_2: w.vec_n(&format!("{lp}.pre_feedforward_layernorm_2.weight"), hidden)?,
            post_norm_1: w.vec_n(&format!("{lp}.post_feedforward_layernorm_1.weight"), hidden)?,
            post_norm_2: w.vec_n(&format!("{lp}.post_feedforward_layernorm_2.weight"), hidden)?,
            gate_up,
            down: Bank::load(w, &ex("down_proj"), [experts, hidden, ff])?,
            down_scale: w.has(&sc).then(|| w.vec_n(&sc, experts)).transpose()?,
            experts,
            top_k,
            ff,
        })
    }

    /// Appends the FFN of a sparse layer to the decode graph: `x` is the residual stream after
    /// attention `[n, hidden]`, `mlp` the dense MLP's down projection of `pre_feedforward_layernorm(x)`.
    /// Returns the sum of both branches, ready for the layer's post-feedforward norm.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_to_graph(
        &self,
        g: &mut Graph,
        w: &mut crate::graph::Weights,
        p: &str,
        x: ValueId,
        mlp: ValueId,
        eps: f32,
        n: usize,
    ) -> ValueId {
        let f = DType::F32;
        let (e, k, ff, h) = (self.experts, self.top_k, self.ff, self.router.cols);
        let vec_weight = |g: &mut Graph, w: &mut crate::graph::Weights, name: String, v: Vec<f32>, shape: Vec<usize>| {
            w.insert(&name, Tensor::F32(v));
            g.weight(name, shape, DType::F32)
        };
        let norm = |g: &mut Graph, w: &mut crate::graph::Weights, name: &str, input: ValueId, gamma: &[f32]| {
            let gw = vec_weight(g, w, format!("{p}.{name}.weight"), gamma.to_vec(), vec![h]);
            g.add(format!("{p}.{name}"), Op::RmsNorm { eps }, &[input, gw], vec![n, h], f)
        };

        let dense = norm(g, w, "post_ff_ln_1", mlp, &self.post_norm_1);
        let expert_in = norm(g, w, "pre_ff_ln_2", x, &self.pre_norm_2);

        // Router: the weightless RMS norm of the residual stream, scaled by `scale / sqrt(hidden)`.
        let unit = g.add(format!("{p}.router_norm"), Op::RmsNorm { eps }, &[x], vec![n, h], f);
        let rs = (h as f32).sqrt().recip();
        let scale_w = vec_weight(
            g,
            w,
            format!("{p}.router.scale"),
            self.router_scale.iter().map(|s| s * rs).collect(),
            vec![h],
        );
        let router_in = g.add(format!("{p}.router_in"), Op::Mul, &[unit, scale_w], vec![n, h], f);
        let router_dtype = match self.router.packed().map(|p| p.format) {
            None | Some(QuantFormat::BF16) => DType::BF16,
            Some(QuantFormat::FP32) => DType::F32,
            Some(QuantFormat::FP16) => DType::F16,
            Some(q) => DType::Quant(q),
        };
        let router_w = g.weight(format!("{p}.router.weight"), vec![e, h], router_dtype);
        w.insert(format!("{p}.router.weight"), self.router.to_graph_tensor());
        let logits = g.add(format!("{p}.router"), Op::Linear, &[router_in, router_w], vec![n, e], f);
        let ids = g.add(format!("{p}.topk"), Op::TopK { k }, &[logits], vec![n, k], DType::U32);
        let picked = g.add(format!("{p}.picked"), Op::GatherCols, &[logits, ids], vec![n, k], f);
        let mut weights = g.add(format!("{p}.route_w"), Op::Softmax { axis: -1 }, &[picked], vec![n, k], f);
        if let Some(s) = &self.down_scale {
            let table = vec_weight(g, w, format!("{p}.router.per_expert_scale"), s.clone(), vec![1, e]);
            let per_slot = g.add(format!("{p}.expert_scale"), Op::GatherCols, &[table, ids], vec![n, k], f);
            weights = g.add(format!("{p}.route_w_scaled"), Op::Mul, &[weights, per_slot], vec![n, k], f);
        }

        let mut bank = |g: &mut Graph, w: &mut crate::graph::Weights, name: &str, b: &Bank, rows: usize, cols: usize| {
            let (dtype, t) = b.graph_weight(rows, cols, e);
            let name = format!("{p}.experts.{name}.weight");
            w.insert(&name, t);
            g.weight(name, vec![e * rows, cols], dtype)
        };
        let moe = |g: &mut Graph, name: &str, input: ValueId, b: ValueId, out: usize| {
            g.add(
                format!("{p}.{name}"),
                Op::MoeLinear { experts: e, top_k: k },
                &[input, b, ids],
                vec![n * k, out],
                f,
            )
        };
        let (gate, up) = match &self.gate_up {
            GateUp::Split(gb, ub) => {
                let (gw, uw) = (bank(g, w, "gate_proj", gb, ff, h), bank(g, w, "up_proj", ub, ff, h));
                (moe(g, "expert_gate", expert_in, gw, ff), moe(g, "expert_up", expert_in, uw, ff))
            }
            GateUp::Fused(b) => {
                let gu_w = bank(g, w, "gate_up_proj", b, 2 * ff, h);
                let gu = moe(g, "expert_gate_up", expert_in, gu_w, 2 * ff);
                let half = |g: &mut Graph, name: &str, start: usize| {
                    g.add(
                        format!("{p}.{name}"),
                        Op::Slice { axis: 1, start: start as i64, end: (start + ff) as i64 },
                        &[gu],
                        vec![n * k, ff],
                        f,
                    )
                };
                (half(g, "expert_gate", 0), half(g, "expert_up", ff))
            }
        };
        let act = g.add(format!("{p}.expert_act"), Op::GeGlu, &[gate, up], vec![n * k, ff], f);
        let down_w = bank(g, w, "down_proj", &self.down, h, ff);
        let down = moe(g, "expert_down", act, down_w, h);
        let w_col = g.add(
            format!("{p}.route_w_col"),
            Op::Reshape { shape: Some(vec![n * k, 1]) },
            &[weights],
            vec![n * k, 1],
            f,
        );
        let weighted = g.add(format!("{p}.expert_weighted"), Op::Mul, &[down, w_col], vec![n * k, h], f);
        let slots = g.add(
            format!("{p}.expert_slots"),
            Op::Reshape { shape: Some(vec![n, k, h]) },
            &[weighted],
            vec![n, k, h],
            f,
        );
        let routed = g.add(
            format!("{p}.routed"),
            Op::Reduce { kind: ReduceKind::Sum, axes: vec![1], keep_dims: false },
            &[slots],
            vec![n, h],
            f,
        );
        let sparse = norm(g, w, "post_ff_ln_2", routed, &self.post_norm_2);
        g.add(format!("{p}.ffn_sum"), Op::Add, &[dense, sparse], vec![n, h], f)
    }
}
