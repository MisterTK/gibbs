//! Subgraph fusion optimization passes for Graph IR.

use crate::graph::pattern::{GraphRewriter, MatchContext};
use crate::graph::{DType, Graph, Op, Tensor, Weights};

/// Concatenates 2D weight tensors along axis 0 (rows).
/// Supports F32, BF16, F16, GGUF-quantized, and Mat.
pub fn concat_weight_tensors(tensors: &[&Tensor], in_features: usize) -> Option<Tensor> {
    if tensors.is_empty() {
        return None;
    }
    // Normalize any Tensor::Mat: if packed GGUF, convert to Tensor::Quantized(ZeroCopySlice);
    // if uncompressed safetensors, extract as Tensor::BF16.
    let normalized: Vec<Tensor> = tensors
        .iter()
        .map(|&t| match t {
            Tensor::Mat(m) => {
                if m.packed().is_some() {
                    m.to_graph_tensor()
                } else {
                    Tensor::BF16(m.data().to_vec())
                }
            }
            _ => t.clone(),
        })
        .collect();

    let norm_refs: Vec<&Tensor> = normalized.iter().collect();
    concat_weight_tensors_inner(&norm_refs, in_features)
}

fn concat_weight_tensors_inner(tensors: &[&Tensor], in_features: usize) -> Option<Tensor> {
    if tensors.is_empty() {
        return None;
    }
    match tensors[0] {
        Tensor::F32(_) => {
            let mut total_len = 0;
            for t in tensors {
                if let Tensor::F32(v) = t {
                    total_len += v.len();
                } else {
                    return None;
                }
            }
            let mut data = Vec::with_capacity(total_len);
            for t in tensors {
                if let Tensor::F32(v) = t {
                    data.extend_from_slice(v);
                }
            }
            Some(Tensor::F32(data))
        }
        Tensor::BF16(_) => {
            let mut total_len = 0;
            for t in tensors {
                if let Tensor::BF16(v) = t {
                    total_len += v.len();
                } else {
                    return None;
                }
            }
            let mut data = Vec::with_capacity(total_len);
            for t in tensors {
                if let Tensor::BF16(v) = t {
                    data.extend_from_slice(v);
                }
            }
            Some(Tensor::BF16(data))
        }
        Tensor::F16(_) => {
            let mut total_len = 0;
            for t in tensors {
                if let Tensor::F16(v) = t {
                    total_len += v.len();
                } else {
                    return None;
                }
            }
            let mut data = Vec::with_capacity(total_len);
            for t in tensors {
                if let Tensor::F16(v) = t {
                    data.extend_from_slice(v);
                }
            }
            Some(Tensor::F16(data))
        }
        Tensor::Quantized(first_q) => match first_q.as_ref() {
            crate::model::layers::QuantizedWeight::GgufQuantized { format, .. } => {
                let mut total_bytes = 0;
                let mut total_out = 0;
                for t in tensors {
                    if let Tensor::Quantized(q) = t {
                        if let crate::model::layers::QuantizedWeight::GgufQuantized {
                            format: fmt,
                            raw_data,
                            in_features: inf,
                            out_features: outf,
                        } = q.as_ref()
                        {
                            if fmt != format || *inf != in_features {
                                return None;
                            }
                            total_bytes += raw_data.len();
                            total_out += outf;
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                let mut raw = Vec::with_capacity(total_bytes);
                for t in tensors {
                    if let Tensor::Quantized(q) = t {
                        if let crate::model::layers::QuantizedWeight::GgufQuantized {
                            raw_data,
                            ..
                        } = q.as_ref()
                        {
                            raw.extend_from_slice(raw_data);
                        }
                    }
                }
                Some(Tensor::Quantized(std::sync::Arc::new(
                    crate::model::layers::QuantizedWeight::GgufQuantized {
                        format: *format,
                        raw_data: bytes::Bytes::from(raw),
                        in_features,
                        out_features: total_out,
                    },
                )))
            }
            crate::model::layers::QuantizedWeight::ZeroCopySlice { format, .. } => {
                let mut total_bytes = 0;
                let mut total_out = 0;
                for t in tensors {
                    if let Tensor::Quantized(q) = t {
                        if let crate::model::layers::QuantizedWeight::ZeroCopySlice {
                            format: fmt,
                            data,
                            in_features: inf,
                            out_features: outf,
                        } = q.as_ref()
                        {
                            if fmt != format || *inf != in_features {
                                return None;
                            }
                            total_bytes += data.len();
                            total_out += outf;
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                let mut raw = Vec::with_capacity(total_bytes);
                for t in tensors {
                    if let Tensor::Quantized(q) = t {
                        if let crate::model::layers::QuantizedWeight::ZeroCopySlice {
                            data, ..
                        } = q.as_ref()
                        {
                            raw.extend_from_slice(data);
                        }
                    }
                }
                Some(Tensor::Quantized(std::sync::Arc::new(
                    crate::model::layers::QuantizedWeight::ZeroCopySlice {
                        format: *format,
                        data: bytes::Bytes::from(raw),
                        in_features,
                        out_features: total_out,
                    },
                )))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Validates that `tensor` matches the declared graph `dtype`.
pub(crate) fn matches_dtype(dtype: DType, tensor: &Tensor) -> bool {
    match (dtype, tensor) {
        (DType::F32, Tensor::F32(_)) => true,
        (DType::BF16, Tensor::BF16(_)) => true,
        (DType::F16, Tensor::F16(_)) => true,
        (DType::Quant(fmt), Tensor::Quantized(q)) => match q.as_ref() {
            crate::model::layers::QuantizedWeight::GgufQuantized { format, .. } => *format == fmt,
            crate::model::layers::QuantizedWeight::ZeroCopySlice { format, .. } => *format == fmt,
            _ => false,
        },
        _ => false,
    }
}

/// Fuses parallel Q, K, V linear projections of one input into one projection and slices (streams
/// the activations once instead of 3 times): materializes the e-graph's chosen `matches` (captures
/// `q`, `k`, `v`; `q` first in the fused weight).
pub(crate) fn rewrite_qkv(
    graph: &mut Graph,
    matches: &[MatchContext],
    mut weights: Option<&mut Weights>,
) -> usize {
    if matches.is_empty() {
        return 0;
    }
    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();

    let rewrites = GraphRewriter::replace(graph, matches, |g, m| {
        let q_id = m.get("q")?;
        let k_id = m.get("k")?;
        let v_id = m.get("v")?;

        // Bias not supported for fusion yet: require exactly 2 inputs (x, w)
        if g.node(q_id).inputs.len() != 2
            || g.node(k_id).inputs.len() != 2
            || g.node(v_id).inputs.len() != 2
        {
            return None;
        }

        let in_val = g.node(q_id).inputs.first().copied()?;
        if g.node(k_id).inputs.first().copied()? != in_val
            || g.node(v_id).inputs.first().copied()? != in_val
        {
            return None;
        }

        let q_w_id = g.node(q_id).inputs.get(1).copied()?;
        let k_w_id = g.node(k_id).inputs.get(1).copied()?;
        let v_w_id = g.node(v_id).inputs.get(1).copied()?;

        // Verify identical dtypes
        let dtype = g.value(q_w_id).dtype;
        if g.value(k_w_id).dtype != dtype || g.value(v_w_id).dtype != dtype {
            return None;
        }

        let q_w_shape = g.value(q_w_id).shape.clone();
        let k_w_shape = g.value(k_w_id).shape.clone();
        let v_w_shape = g.value(v_w_id).shape.clone();

        if q_w_shape.len() != 2 || k_w_shape.len() != 2 || v_w_shape.len() != 2 {
            return None;
        }
        if q_w_shape[1] != k_w_shape[1] || q_w_shape[1] != v_w_shape[1] {
            return None;
        }

        let in_features = q_w_shape[1];
        let q_out = q_w_shape[0];
        let k_out = k_w_shape[0];
        let v_out = v_w_shape[0];
        let total_out = q_out + k_out + v_out;

        let q_name = g.node(q_id).name.clone();
        let k_name = g.node(k_id).name.clone();
        let v_name = g.node(v_id).name.clone();

        let q_out_shape = g.value(g.node(q_id).outputs[0]).shape.clone();
        let k_out_shape = g.value(g.node(k_id).outputs[0]).shape.clone();
        let v_out_shape = g.value(g.node(v_id).outputs[0]).shape.clone();

        let fused_w_name = format!("{}_fused_qkv_w", q_name);

        // If weights are provided, we MUST be able to fuse the weight tensors
        if let Some(ref w_store) = weights {
            let q_t = w_store.get(&g.value(q_w_id).name)?;
            let k_t = w_store.get(&g.value(k_w_id).name)?;
            let v_t = w_store.get(&g.value(v_w_id).name)?;
            if matches!(q_t, Tensor::Mat(_))
                || matches!(k_t, Tensor::Mat(_))
                || matches!(v_t, Tensor::Mat(_))
            {
                return None;
            }
            let fused_tensor = concat_weight_tensors(&[q_t, k_t, v_t], in_features)?;
            if !matches_dtype(dtype, &fused_tensor) {
                return None;
            }
            let old_names = vec![
                g.value(q_w_id).name.clone(),
                g.value(k_w_id).name.clone(),
                g.value(v_w_id).name.clone(),
            ];
            weight_updates.push((fused_w_name.clone(), fused_tensor, old_names));
        }

        let fused_w = g.weight(fused_w_name, vec![total_out, in_features], dtype);
        let in_shape = g.value(in_val).shape.clone();
        let mut fused_out_shape = in_shape;
        if let Some(last) = fused_out_shape.last_mut() {
            *last = total_out;
        }

        let fused_lin = g.add(
            format!("{}.fused_qkv", q_name),
            Op::Linear,
            &[in_val, fused_w],
            fused_out_shape,
            DType::F32,
        );

        let q_slice = g.add(
            format!("{}.q_slice", q_name),
            Op::Slice {
                axis: -1,
                start: 0,
                end: q_out as i64,
            },
            &[fused_lin],
            q_out_shape,
            DType::F32,
        );

        let k_slice = g.add(
            format!("{}.k_slice", k_name),
            Op::Slice {
                axis: -1,
                start: q_out as i64,
                end: (q_out + k_out) as i64,
            },
            &[fused_lin],
            k_out_shape,
            DType::F32,
        );

        let v_slice = g.add(
            format!("{}.v_slice", v_name),
            Op::Slice {
                axis: -1,
                start: (q_out + k_out) as i64,
                end: total_out as i64,
            },
            &[fused_lin],
            v_out_shape,
            DType::F32,
        );

        Some((vec![q_id, k_id, v_id], vec![q_slice, k_slice, v_slice]))
    });

    if let Some(ref mut w_store) = weights {
        for (name, tensor, old_names) in weight_updates {
            w_store.insert(name, tensor);
            for old in old_names {
                w_store.remove(&old);
            }
        }
    }

    rewrites
}

/// Fuses the gate and up projections of a SwiGLU / GeGLU into one projection (streams the
/// activations once instead of twice): materializes the e-graph's chosen `matches` (captures
/// `gate`, `up`; roles come from the GLU).
pub(crate) fn rewrite_gate_up(
    graph: &mut Graph,
    matches: &[MatchContext],
    mut weights: Option<&mut Weights>,
) -> usize {
    if matches.is_empty() {
        return 0;
    }
    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();

    let rewrites = GraphRewriter::replace(graph, matches, |g, m| {
        let a_id = m.get("gate")?;
        let b_id = m.get("up")?;

        // Bias not supported for fusion yet: require exactly 2 inputs (x, w)
        if g.node(a_id).inputs.len() != 2 || g.node(b_id).inputs.len() != 2 {
            return None;
        }

        let a_out = g.node(a_id).outputs[0];
        let b_out = g.node(b_id).outputs[0];

        // Consumers of a and b must feed into SwiGlu or GeGlu
        let glu_node = g.nodes.iter().find(|n| {
            matches!(n.op, Op::SwiGlu | Op::GeGlu)
                && (n.inputs.contains(&a_out) && n.inputs.contains(&b_out))
        })?;

        // Determine gate and up strictly by GLU inputs order: inputs[0] is gate, inputs[1] is up
        let (gate_id, up_id) = if glu_node.inputs[0] == a_out && glu_node.inputs[1] == b_out {
            (a_id, b_id)
        } else if glu_node.inputs[0] == b_out && glu_node.inputs[1] == a_out {
            (b_id, a_id)
        } else {
            return None;
        };

        let in_val = g.node(gate_id).inputs.first().copied()?;
        if g.node(up_id).inputs.first().copied()? != in_val {
            return None;
        }

        let gate_w_id = g.node(gate_id).inputs.get(1).copied()?;
        let up_w_id = g.node(up_id).inputs.get(1).copied()?;

        let dtype = g.value(gate_w_id).dtype;
        if g.value(up_w_id).dtype != dtype {
            return None;
        }

        let gate_w_shape = g.value(gate_w_id).shape.clone();
        let up_w_shape = g.value(up_w_id).shape.clone();

        if gate_w_shape.len() != 2 || up_w_shape.len() != 2 || gate_w_shape[1] != up_w_shape[1] {
            return None;
        }

        let in_features = gate_w_shape[1];
        let gate_out_dim = gate_w_shape[0];
        let up_out_dim = up_w_shape[0];
        let total_out = gate_out_dim + up_out_dim;

        let gate_name = g.node(gate_id).name.clone();
        let up_name = g.node(up_id).name.clone();
        let gate_out_shape = g.value(g.node(gate_id).outputs[0]).shape.clone();
        let up_out_shape = g.value(g.node(up_id).outputs[0]).shape.clone();

        let fused_w_name = format!("{}_fused_gate_up_w", gate_name);

        // If weights are provided, we MUST be able to fuse the weight tensors
        if let Some(ref w_store) = weights {
            let gate_t = w_store.get(&g.value(gate_w_id).name)?;
            let up_t = w_store.get(&g.value(up_w_id).name)?;
            // Model-loader Mats stay unfused: the Metal planner folds the GLU into the up GEMV's
            // epilogue (`Epi::UpGeGlu`/`UpSwiGlu`) when gate and up are separate, while the fused
            // form plans as one GEMV plus a standalone GLU step: same dispatch count, one more
            // activation round trip.
            if matches!(gate_t, Tensor::Mat(_)) || matches!(up_t, Tensor::Mat(_)) {
                return None;
            }
            let fused_tensor = concat_weight_tensors(&[gate_t, up_t], in_features)?;
            if !matches_dtype(dtype, &fused_tensor) {
                return None;
            }
            let old_names = vec![
                g.value(gate_w_id).name.clone(),
                g.value(up_w_id).name.clone(),
            ];
            weight_updates.push((fused_w_name.clone(), fused_tensor, old_names));
        }

        let fused_w = g.weight(fused_w_name, vec![total_out, in_features], dtype);
        let in_shape = g.value(in_val).shape.clone();
        let mut fused_out_shape = in_shape;
        if let Some(last) = fused_out_shape.last_mut() {
            *last = total_out;
        }

        let fused_lin = g.add(
            format!("{}.fused_gate_up", gate_name),
            Op::Linear,
            &[in_val, fused_w],
            fused_out_shape,
            DType::F32,
        );

        let gate_slice = g.add(
            format!("{}.gate_slice", gate_name),
            Op::Slice {
                axis: -1,
                start: 0,
                end: gate_out_dim as i64,
            },
            &[fused_lin],
            gate_out_shape,
            DType::F32,
        );

        let up_slice = g.add(
            format!("{}.up_slice", up_name),
            Op::Slice {
                axis: -1,
                start: gate_out_dim as i64,
                end: total_out as i64,
            },
            &[fused_lin],
            up_out_shape,
            DType::F32,
        );

        Some((vec![gate_id, up_id], vec![gate_slice, up_slice]))
    });

    if let Some(ref mut w_store) = weights {
        for (name, tensor, old_names) in weight_updates {
            w_store.insert(name, tensor);
            for old in old_names {
                w_store.remove(&old);
            }
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::egraph::{fuse_rule, Rule};
    use crate::graph::pattern::validate_ssa;

    #[test]
    fn test_fuse_qkv_pass() {
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 64], DType::F32);
        let gamma = g.weight("gamma", vec![64], DType::F32);
        let norm = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[in_x, gamma],
            vec![1, 64],
            DType::F32,
        );

        let w_q = g.weight("w_q", vec![64, 64], DType::F32);
        let w_k = g.weight("w_k", vec![32, 64], DType::F32);
        let w_v = g.weight("w_v", vec![32, 64], DType::F32);

        let q = g.add("q", Op::Linear, &[norm, w_q], vec![1, 64], DType::F32);
        let k = g.add("k", Op::Linear, &[norm, w_k], vec![1, 32], DType::F32);
        let v = g.add("v", Op::Linear, &[norm, w_v], vec![1, 32], DType::F32);

        let out = g.add("out_add", Op::Binary { kind: crate::graph::BinaryKind::Add }, &[k, v], vec![1, 32], DType::F32);
        g.output(q);
        g.output(out);

        let mut weights = Weights::new();
        weights.insert("w_q", Tensor::F32(vec![1.0; 64 * 64]));
        weights.insert("w_k", Tensor::F32(vec![2.0; 32 * 64]));
        weights.insert("w_v", Tensor::F32(vec![3.0; 32 * 64]));

        let rewrites = fuse_rule(&mut g, Some(&mut weights), Rule::Qkv);
        assert_eq!(rewrites, 1);
        assert!(validate_ssa(&g).is_ok());

        assert!(weights.get("q_fused_qkv_w").is_some());
        let fused_w = weights.get("q_fused_qkv_w").unwrap().as_f32().unwrap();
        assert_eq!(fused_w.len(), (64 + 32 + 32) * 64);
        assert_eq!(fused_w[0], 1.0);
        assert_eq!(fused_w[64 * 64], 2.0);
        assert_eq!(fused_w[(64 + 32) * 64], 3.0);
    }

    #[test]
    fn test_fuse_gate_up_pass() {
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 64], DType::F32);
        let w_g = g.weight("w_g", vec![128, 64], DType::F32);
        let w_u = g.weight("w_u", vec![128, 64], DType::F32);

        let gate = g.add("gate", Op::Linear, &[in_x, w_g], vec![1, 128], DType::F32);
        let up = g.add("up", Op::Linear, &[in_x, w_u], vec![1, 128], DType::F32);
        let mlp = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![1, 128], DType::F32);
        g.output(mlp);

        let mut weights = Weights::new();
        weights.insert("w_g", Tensor::F32(vec![0.5; 128 * 64]));
        weights.insert("w_u", Tensor::F32(vec![0.75; 128 * 64]));

        let rewrites = fuse_rule(&mut g, Some(&mut weights), Rule::GateUp);
        assert_eq!(rewrites, 1);
        assert!(validate_ssa(&g).is_ok());
        assert!(weights.get("gate_fused_gate_up_w").is_some());
    }

    #[test]
    fn test_cpu_parity_before_and_after_fusion() {
        use crate::graph::cpu::run_stateless;

        // Build a graph with distinct weights
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 4], DType::F32);
        let gamma = g.weight("gamma", vec![4], DType::F32);
        let norm = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[in_x, gamma],
            vec![1, 4],
            DType::F32,
        );

        let w_q = g.weight("w_q", vec![4, 4], DType::F32);
        let w_k = g.weight("w_k", vec![2, 4], DType::F32);
        let w_v = g.weight("w_v", vec![2, 4], DType::F32);
        let q = g.add("q", Op::Linear, &[norm, w_q], vec![1, 4], DType::F32);
        let k = g.add("k", Op::Linear, &[norm, w_k], vec![1, 2], DType::F32);
        let v = g.add("v", Op::Linear, &[norm, w_v], vec![1, 2], DType::F32);

        let w_g = g.weight("w_g", vec![8, 4], DType::F32);
        let w_u = g.weight("w_u", vec![8, 4], DType::F32);
        let gate = g.add("gate", Op::Linear, &[norm, w_g], vec![1, 8], DType::F32);
        let up = g.add("up", Op::Linear, &[norm, w_u], vec![1, 8], DType::F32);
        let mlp = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![1, 8], DType::F32);

        g.output(q);
        g.output(k);
        g.output(v);
        g.output(mlp);

        let mut weights = Weights::new();
        weights.insert("gamma", Tensor::F32(vec![1.1, 0.9, 1.05, 0.95]));
        weights.insert(
            "w_q",
            Tensor::F32((0..16).map(|i| (i as f32 * 0.1) - 0.7).collect()),
        );
        weights.insert(
            "w_k",
            Tensor::F32((0..8).map(|i| (i as f32 * 0.15) - 0.4).collect()),
        );
        weights.insert(
            "w_v",
            Tensor::F32((0..8).map(|i| (i as f32 * 0.2) - 0.5).collect()),
        );
        weights.insert(
            "w_g",
            Tensor::F32((0..32).map(|i| (i as f32 * 0.05) - 0.3).collect()),
        );
        weights.insert(
            "w_u",
            Tensor::F32((0..32).map(|i| (i as f32 * 0.07) - 0.6).collect()),
        );

        let input_tensor = vec![Tensor::F32(vec![0.5, -1.2, 2.3, -0.4])];

        // 1. Run baseline on unoptimized graph
        let baseline_res = run_stateless(&g, &weights, input_tensor.clone()).unwrap();

        // 2. Run passes
        let mut opt_g = g.clone();
        let mut opt_w = weights.clone();
        let qkv_rewrites = fuse_rule(&mut opt_g, Some(&mut opt_w), Rule::Qkv);
        let glu_rewrites = fuse_rule(&mut opt_g, Some(&mut opt_w), Rule::GateUp);
        assert_eq!(qkv_rewrites, 1);
        assert_eq!(glu_rewrites, 1);
        assert!(validate_ssa(&opt_g).is_ok());

        // 3. Run optimized graph
        let opt_res = run_stateless(&opt_g, &opt_w, input_tensor).unwrap();

        // 4. Assert exact numerical equality across all 4 outputs
        assert_eq!(baseline_res.len(), opt_res.len());
        for (out_idx, (b_t, o_t)) in baseline_res.iter().zip(&opt_res).enumerate() {
            let b_slice = b_t.as_f32().unwrap();
            let o_slice = o_t.as_f32().unwrap();
            assert_eq!(b_slice.len(), o_slice.len());
            for (i, (&b_val, &o_val)) in b_slice.iter().zip(o_slice).enumerate() {
                let diff = (b_val - o_val).abs();
                assert!(
                    diff < 1e-5,
                    "Parity failed at output {out_idx} element {i}: baseline={b_val}, opt={o_val}, diff={diff}"
                );
            }
        }
    }

    /// x -> norm -> {q,k,v} Linear -> Attention with per-projection weight dtypes. F16 weights
    /// stand in for a second plain codec; `Quant(f)` weights are packed GGUF `Mat`s in an
    /// anonymous mapping, as the GGUF loaders produce them. Returns the graph, its weights and the
    /// input.
    fn qkv_graph(dts: [DType; 3]) -> (Graph, Weights, Vec<Tensor>) {
        let (h, rows) = (256usize, [8usize, 4, 4]);
        let mut g = Graph::new();
        let x = g.input("x", vec![1, h], DType::F32);
        let gamma = g.weight("gamma", vec![h], DType::F32);
        let norm = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, gamma],
            vec![1, h],
            DType::F32,
        );
        let mut weights = Weights::new();
        weights.insert("gamma", Tensor::F32(vec![1.0; h]));
        let mut outs = Vec::new();
        for (i, name) in ["q", "k", "v"].into_iter().enumerate() {
            let n = rows[i] * h;
            let wn = format!("w_{name}");
            let w = g.weight(&wn, vec![rows[i], h], dts[i]);
            // Exact halves of the form ±(j%5)/4, so F16 and F32 weights both decode exactly.
            let vals = (0..n).map(|j| ((j * (i + 3)) % 5) as f32 / 4.0 - 0.5);
            weights.insert(
                wn,
                match dts[i] {
                    DType::F16 => Tensor::F16(
                        vals.map(|f| match (f * 4.0) as i32 {
                            -2 => 0xB800,
                            -1 => 0xB400,
                            0 => 0,
                            1 => 0x3400,
                            _ => 0x3800,
                        })
                        .collect(),
                    ),
                    DType::Quant(fmt) => packed_mat(fmt, rows[i], h, i),
                    _ => Tensor::F32(vals.collect()),
                },
            );
            let out = g.add(name, Op::Linear, &[norm, w], vec![1, rows[i]], DType::F32);
            g.output(out);
            outs.push(out);
        }
        let op = Op::Attention {
            heads: 2,
            kv_heads: 1,
            causal: true,
            window: None,
            soft_cap: None,
            scale: None,
        };
        let a = g.add("attn", op, &outs, vec![1, rows[0]], DType::F32);
        g.output(a);
        let input = vec![Tensor::F32((0..h).map(|i| i as f32 * 0.3 - 1.0).collect())];
        (g, weights, input)
    }

    /// A `[rows, cols]` packed GGUF `Mat` of patterned blocks whose f16 scales (Q4_0: `d`;
    /// Q4_K: `d`, `dmin`) are a finite 0.0625.
    fn packed_mat(
        fmt: crate::hardware::QuantFormat,
        rows: usize,
        cols: usize,
        seed: usize,
    ) -> Tensor {
        use crate::model::weights::{Mat, Packed};
        let (block, header) = match fmt {
            crate::hardware::QuantFormat::GGUF_Q4_0 => (18, 2),
            crate::hardware::QuantFormat::GGUF_Q4_K_M => (144, 4),
            f => panic!("no fixture for {f:?}"),
        };
        let n = rows * fmt.tensor_bytes(cols);
        let mut map = memmap2::MmapMut::map_anon(n).unwrap();
        for (j, b) in map.iter_mut().enumerate() {
            *b = (j * 31 + seed * 7) as u8;
        }
        for blk in map.chunks_exact_mut(block) {
            for c in blk[..header].chunks_exact_mut(2) {
                c.copy_from_slice(&0x2C00u16.to_le_bytes());
            }
        }
        let p = Packed {
            map: std::sync::Arc::new(crate::model::loaders::gguf::GgufData::Map(
                map.make_read_only().unwrap(),
            )),
            off: 0,
            format: fmt,
            shape: vec![rows, cols],
        };
        Tensor::Mat(Mat::from_packed(p, rows, cols))
    }

    /// Runs the QKV rule, checks CPU parity with the unfused graph, and returns the names of
    /// the remaining Linear nodes.
    fn fuse_and_check(dts: [DType; 3]) -> Vec<String> {
        use crate::graph::cpu::run_stateless;
        let (g, w, input) = qkv_graph(dts);
        let base = run_stateless(&g, &w, input.clone()).unwrap();
        let (mut fg, mut fw) = (g.clone(), w.clone());
        assert_eq!(fuse_rule(&mut fg, Some(&mut fw), Rule::Qkv), 1);
        assert!(validate_ssa(&fg).is_ok());
        let fused = run_stateless(&fg, &fw, input).unwrap();
        assert_eq!(base.len(), fused.len());
        for (b, f) in base.iter().zip(&fused) {
            assert_eq!(b.as_f32().unwrap(), f.as_f32().unwrap());
        }
        let mut names: Vec<String> = fg
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Linear))
            .map(|n| n.name.clone())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn fuse_qkv_pass_leaves_mixed_formats_and_mats_unfused() {
        use crate::hardware::QuantFormat as Q;
        let (h, f, q4) = (DType::F16, DType::F32, DType::Quant(Q::GGUF_Q4_0));
        for dts in [[h, h, f], [f, h, h], [q4, q4, q4]] {
            let (mut g, mut w, _) = qkv_graph(dts);
            assert_eq!(fuse_rule(&mut g, Some(&mut w), Rule::Qkv), 0, "{dts:?}");
        }
    }

    #[test]
    fn fuse_qkv_pass_same_format_still_fuses_all_three() {
        let f = DType::F32;
        assert_eq!(fuse_and_check([f, f, f]), ["q.fused_qkv"]);
    }
}

/// Fuses QKV slices and RoPE directly into CachedAttention, eliminating intermediate Slice and RoPE
/// nodes: materializes the e-graph's choice for the `only` attention nodes.
pub fn fuse_qkv_slices_and_attn_pass(
    graph: &mut Graph,
    only: &std::collections::HashSet<crate::graph::NodeId>,
) -> usize {
    let mut rewrites = 0;
    let mut eliminated_nodes: std::collections::HashSet<crate::graph::NodeId> =
        std::collections::HashSet::new();

    let mut val_producers: std::collections::HashMap<crate::graph::ValueId, crate::graph::NodeId> =
        std::collections::HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            val_producers.insert(out, crate::graph::NodeId(idx as u32));
        }
    }

    let n_nodes = graph.nodes.len();
    for i in 0..n_nodes {
        let nid = crate::graph::NodeId(i as u32);
        let node = graph.node(nid);
        if !only.contains(&nid) {
            continue;
        }
        if let crate::graph::Op::CachedAttention { .. } = &node.op {
            if node.inputs.len() >= 3 {
                let q_val = node.inputs[0];
                let k_val = node.inputs[1];
                let v_val = node.inputs[2];

                let q_rope_nid = val_producers.get(&q_val).copied();
                let k_rope_nid = val_producers.get(&k_val).copied();

                if let (Some(q_rnid), Some(k_rnid)) = (q_rope_nid, k_rope_nid) {
                    let q_rnode = graph.node(q_rnid);
                    let k_rnode = graph.node(k_rnid);
                    if matches!(q_rnode.op, crate::graph::Op::Rope { .. })
                        && matches!(k_rnode.op, crate::graph::Op::Rope { .. })
                    {
                        let q_unrot = q_rnode.inputs[0];
                        let k_unrot = k_rnode.inputs[0];

                        let q_slice_nid = val_producers.get(&q_unrot).copied();
                        let k_slice_nid = val_producers.get(&k_unrot).copied();
                        let v_slice_nid = val_producers.get(&v_val).copied();

                        if let (Some(q_snid), Some(k_snid), Some(v_snid)) =
                            (q_slice_nid, k_slice_nid, v_slice_nid)
                        {
                            let q_snode = graph.node(q_snid);
                            let k_snode = graph.node(k_snid);
                            let v_snode = graph.node(v_snid);

                            if matches!(q_snode.op, crate::graph::Op::Slice { .. })
                                && matches!(k_snode.op, crate::graph::Op::Slice { .. })
                                && matches!(v_snode.op, crate::graph::Op::Slice { .. })
                                && q_snode.inputs[0] == k_snode.inputs[0]
                                && k_snode.inputs[0] == v_snode.inputs[0]
                            {
                                let fused_qkv_val = q_snode.inputs[0];

                                let att_node = &mut graph.nodes[i];
                                att_node.inputs[0] = fused_qkv_val;
                                att_node.inputs[1] = fused_qkv_val;
                                att_node.inputs[2] = fused_qkv_val;

                                eliminated_nodes.insert(q_rnid);
                                eliminated_nodes.insert(k_rnid);
                                eliminated_nodes.insert(q_snid);
                                eliminated_nodes.insert(k_snid);
                                eliminated_nodes.insert(v_snid);
                                rewrites += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    if rewrites > 0 {
        crate::graph::pattern::compact_graph_public(graph, &eliminated_nodes);
    }
    rewrites
}
