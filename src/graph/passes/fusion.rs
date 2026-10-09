//! Subgraph fusion optimization passes for Graph IR.

use crate::graph::pattern::{any, capture, op_type, GraphRewriter, Pattern};
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
fn matches_dtype(dtype: DType, tensor: &Tensor) -> bool {
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

/// Fuses parallel Q, K, V linear projections branching from the same input into a single
/// fused linear projection, followed by slices.
///
/// Reduces memory bandwidth by streaming input activations once instead of 3 times.
pub fn fuse_qkv_pass(graph: &mut Graph, mut weights: Option<&mut Weights>) -> usize {
    let mut pat = Pattern::new();
    let root = pat.add_node(any());
    let q = pat.add_node(capture("q", op_type("Linear")));
    let k = pat.add_node(capture("k", op_type("Linear")));
    let v = pat.add_node(capture("v", op_type("Linear")));
    pat.add_branch(root, &[q, k, v]);

    let matches = pat.search(graph);
    if matches.is_empty() {
        return 0;
    }

    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();

    let rewrites = GraphRewriter::replace(graph, &matches, |g, m| {
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

/// Fuses Gate and Up linear projections branching into SwiGLU / GeGLU into a single
/// fused linear projection.
///
/// Reduces memory bandwidth by streaming input activations once instead of twice.
pub fn fuse_gate_up_pass(graph: &mut Graph, mut weights: Option<&mut Weights>) -> usize {
    let mut pat = Pattern::new();
    let gate = pat.add_node(capture("gate", op_type("Linear")));
    let up = pat.add_node(capture("up", op_type("Linear")));
    pat.add_shared_input(gate, up);

    let matches = pat.search(graph);
    if matches.is_empty() {
        return 0;
    }

    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();

    let rewrites = GraphRewriter::replace(graph, &matches, |g, m| {
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

        let out = g.add("out_add", Op::Add, &[k, v], vec![1, 32], DType::F32);
        g.output(q);
        g.output(out);

        let mut weights = Weights::new();
        weights.insert("w_q", Tensor::F32(vec![1.0; 64 * 64]));
        weights.insert("w_k", Tensor::F32(vec![2.0; 32 * 64]));
        weights.insert("w_v", Tensor::F32(vec![3.0; 32 * 64]));

        let rewrites = fuse_qkv_pass(&mut g, Some(&mut weights));
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

        let rewrites = fuse_gate_up_pass(&mut g, Some(&mut weights));
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
        let qkv_rewrites = fuse_qkv_pass(&mut opt_g, Some(&mut opt_w));
        let glu_rewrites = fuse_gate_up_pass(&mut opt_g, Some(&mut opt_w));
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
}
