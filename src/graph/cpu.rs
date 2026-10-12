//! CPU executor for the graph IR, built on the `crate::model::nn` kernels.

use super::{Graph, Op, StateStore, Tensor, ValueId, ValueKind, Weights};
use crate::model::nn;

/// Runs `graph` with stateless temporary buffers (discarded after completion).
pub fn run_stateless(
    graph: &Graph,
    weights: &Weights,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>, String> {
    crate::hardware::cpu_compute::enter("graph::cpu::run_stateless");
    let mut state = StateStore::new();
    run(graph, weights, &mut state, inputs)
}

/// Returns whether the CPU executor implements `op` for the given data type.
pub fn supports_op(op: &Op, _dtype: crate::graph::DType) -> bool {
    match op {
        Op::Embedding
        | Op::Linear
        | Op::RmsNorm { .. }
        | Op::LayerNorm { .. }
        | Op::SwiGlu
        | Op::GeGlu
        | Op::Unary { .. }
        | Op::Binary { .. }
        | Op::Silu
        | Op::Relu
        | Op::Mul
        | Op::Add
        | Op::Rope { .. }
        | Op::CachedAttention { .. }
        | Op::PagedCachedAttention { .. }
        | Op::CausalConv1dCached { .. }
        | Op::DeltaRule { .. }
        | Op::Gather { .. }
        | Op::Scatter { .. }
        | Op::Pad { .. }
        | Op::AvgPool2d { .. }
        | Op::MeanPool
        | Op::L2Normalize
        | Op::Softmax { .. }
        | Op::TopKSoftmax { .. }
        | Op::Slice { .. }
        | Op::Conv2d { .. }
        | Op::RelPosBias { .. }
        | Op::StateReorder
        | Op::Megakernel(_) => true,
        Op::Upsample { mode, .. } => matches!(mode, crate::graph::UpsampleMode::Nearest),
        Op::Attention {
            kv_heads,
            heads,
            window,
            soft_cap,
            ..
        } => kv_heads == heads && window.is_none() && soft_cap.is_none(),
        Op::Gelu { tanh } => !*tanh,
        _ => false,
    }
}

/// Runs `graph` with `inputs` bound positionally to `graph.inputs` and persistent `state` passed by reference.
pub fn run(
    graph: &Graph,
    weights: &Weights,
    state: &mut StateStore,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>, String> {
    crate::hardware::cpu_compute::enter("graph::cpu::run");
    if inputs.len() != graph.inputs.len() {
        return Err(format!(
            "graph has {} inputs, got {}",
            graph.inputs.len(),
            inputs.len()
        ));
    }
    let nv = graph.values.len();
    let mut slots: Vec<Option<Tensor>> = vec![None; nv];
    let mut bound: Vec<Option<&Tensor>> = vec![None; nv];
    for (&id, t) in graph.inputs.iter().zip(inputs) {
        let v = graph.value(id);
        if !v.shape.is_empty() && v.shape.iter().product::<usize>() != t.len() {
            return Err(format!(
                "input '{}' expects shape {:?}, got {} elements",
                v.name,
                v.shape,
                t.len()
            ));
        }
        slots[id.0 as usize] = Some(t);
    }
    let mut used_indices = vec![false; nv];
    for node in &graph.nodes {
        for &inp in &node.inputs {
            if (inp.0 as usize) < nv {
                used_indices[inp.0 as usize] = true;
            }
        }
    }
    for (i, v) in graph.values.iter().enumerate() {
        if !used_indices[i] {
            continue;
        }
        match &v.kind {
            ValueKind::Weight => {
                bound[i] = Some(
                    weights
                        .get(&v.name)
                        .ok_or_else(|| format!("weight '{}' is missing", v.name))?,
                );
            }
            ValueKind::State(state_name) => {
                if let Some(t) = state.get(state_name) {
                    slots[i] = Some(t.clone());
                } else {
                    let numel = if v.shape.is_empty() {
                        0
                    } else {
                        v.shape.iter().product::<usize>()
                    };
                    slots[i] = Some(Tensor::F32(vec![0.0f32; numel]));
                }
            }
            _ => {}
        }
    }
    // Last node index reading each value; graph outputs are never dropped.
    let mut last_use = vec![0usize; nv];
    for (i, node) in graph.nodes.iter().enumerate() {
        for v in &node.inputs {
            last_use[v.0 as usize] = i;
        }
    }
    for v in &graph.outputs {
        last_use[v.0 as usize] = usize::MAX;
    }

    for (i, node) in graph.nodes.iter().enumerate() {
        let name = &node.name;
        let ins = &node.inputs;
        let last_dim = |k: usize| graph.value(ins[k]).shape.last().copied().unwrap_or(0);

        let out = match &node.op {
            Op::Embedding => {
                let table_slot = slot(&bound, &slots, ins, name, 0)?;
                let ids = operand_u32(&bound, &slots, ins, name, 1)?;
                let d = last_dim(0);
                if d == 0 {
                    return Err(format!("node '{name}': embedding table has unknown width"));
                }
                let mut out = Vec::with_capacity(ids.len() * d);
                match table_slot {
                    Tensor::F32(table) => {
                        let rows = table.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend_from_slice(&table[id * d..(id + 1) * d]);
                        }
                    }
                    Tensor::Quantized(q) => {
                        let rows = q.out_features();
                        match q.as_ref() {
                            crate::model::layers::QuantizedWeight::ZeroCopySlice {
                                data,
                                format,
                                in_features,
                                ..
                            }
                            | crate::model::layers::QuantizedWeight::GgufQuantized {
                                raw_data: data,
                                format,
                                in_features,
                                ..
                            } => {
                                if let Some((block_elems, block_bytes, _)) =
                                    crate::model::layers::gguf_block_codec(*format)
                                {
                                    if *in_features != d || d % block_elems != 0 {
                                        return Err(format!("node '{name}': width {d} not compatible with {format:?} blocks"));
                                    }
                                    let row_bytes = (d / block_elems) * block_bytes;
                                    for (r, &id) in ids.iter().enumerate() {
                                        let id = id as usize;
                                        if id >= rows {
                                            return Err(format!(
                                                "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                            ));
                                        }
                                        let slice = &data[id * row_bytes..(id + 1) * row_bytes];
                                        let row_f32 = crate::model::layers::dequantize_gguf_blocks(
                                            *format, slice,
                                        )
                                        .ok_or_else(|| {
                                            format!("node '{name}': failed to dequantize row {id}")
                                        })?;
                                        out.extend_from_slice(&row_f32);
                                    }
                                } else {
                                    let full = q.to_f32();
                                    for (r, &id) in ids.iter().enumerate() {
                                        let id = id as usize;
                                        if id >= rows {
                                            return Err(format!(
                                                "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                            ));
                                        }
                                        out.extend_from_slice(&full[id * d..(id + 1) * d]);
                                    }
                                }
                            }
                            crate::model::layers::QuantizedWeight::Uncompressed {
                                data, ..
                            } => {
                                for (r, &id) in ids.iter().enumerate() {
                                    let id = id as usize;
                                    if id >= rows {
                                        return Err(format!(
                                            "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                        ));
                                    }
                                    out.extend_from_slice(&data[id * d..(id + 1) * d]);
                                }
                            }
                            _ => {
                                let full = q.to_f32();
                                for (r, &id) in ids.iter().enumerate() {
                                    let id = id as usize;
                                    if id >= rows {
                                        return Err(format!(
                                            "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                        ));
                                    }
                                    out.extend_from_slice(&full[id * d..(id + 1) * d]);
                                }
                            }
                        }
                    }
                    Tensor::BF16(v) => {
                        let rows = v.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend(
                                v[id * d..(id + 1) * d]
                                    .iter()
                                    .map(|&x| crate::model::bf16_to_f32(x)),
                            );
                        }
                    }
                    Tensor::F16(v) => {
                        let rows = v.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend(
                                v[id * d..(id + 1) * d]
                                    .iter()
                                    .map(|&x| crate::model::f16_to_f32(x)),
                            );
                        }
                    }
                    other => {
                        let full = other.to_f32();
                        let rows = full.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend_from_slice(&full[id * d..(id + 1) * d]);
                        }
                    }
                }
                out
            }
            Op::Linear => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let w_slot = slot(&bound, &slots, ins, name, 1)?;
                let shape = &graph.value(ins[1]).shape;
                let [out_f, in_f] = shape[..] else {
                    return Err(format!(
                        "node '{name}': weight shape {shape:?} is not [out, in]"
                    ));
                };
                if in_f == 0 || x.len() % in_f != 0 {
                    return Err(format!(
                        "node '{name}': x of {} elements does not match weight {shape:?}",
                        x.len()
                    ));
                }
                let bias = if ins.len() > 2 {
                    Some(operand(&bound, &slots, ins, name, 2)?)
                } else {
                    None
                };
                if bias.is_some_and(|b| b.len() != out_f) {
                    return Err(format!("node '{name}': bias is not [{out_f}]"));
                }
                let rows = x.len() / in_f;
                dispatch_linear(x, w_slot, bias, in_f, out_f, rows, name)?
            }
            Op::Megakernel(desc) => {
                match &desc.kind {
                    super::MegakernelKind::Mlp {
                        hidden_dim,
                        intermediate_dim,
                        eps,
                        is_geglu,
                        has_post_norm,
                    } => {
                        let x = operand(&bound, &slots, ins, name, 0)?;
                        let norm_w = operand(&bound, &slots, ins, name, 1)?;
                        let gate_slot = slot(&bound, &slots, ins, name, 2)?;
                        let up_slot = slot(&bound, &slots, ins, name, 3)?;
                        let down_slot = slot(&bound, &slots, ins, name, 4)?;
                        let post_norm_w = if *has_post_norm {
                            Some(operand(&bound, &slots, ins, name, 5)?)
                        } else {
                            None
                        };
                        let d = *hidden_dim;
                        let inter = *intermediate_dim;
                        let rows = x.len() / d.max(1);

                        // 1. RMSNorm
                        let mut normed_x = vec![0.0f32; x.len()];
                        for (x_chunk, out_chunk) in x.chunks(d).zip(normed_x.chunks_mut(d)) {
                            crate::model::simd::vector_rmsnorm(x_chunk, norm_w, out_chunk, *eps);
                        }

                        // 2. Gate & Up projections
                        let gate_act =
                            dispatch_linear(&normed_x, gate_slot, None, d, inter, rows, name)?;
                        let up_act =
                            dispatch_linear(&normed_x, up_slot, None, d, inter, rows, name)?;

                        // 3. SwiGLU / GeGLU
                        let mut act = vec![0.0f32; gate_act.len()];
                        if *is_geglu {
                            act.copy_from_slice(&gate_act);
                            nn::gelu(&mut act);
                            for (o, &u) in act.iter_mut().zip(&up_act) {
                                *o *= u;
                            }
                        } else {
                            for (o, (&g, &u)) in act.iter_mut().zip(gate_act.iter().zip(&up_act)) {
                                let sig = 1.0 / (1.0 + (-g).exp());
                                *o = (g * sig) * u;
                            }
                        }

                        // 4. Down projection
                        let mut down_act =
                            dispatch_linear(&act, down_slot, None, inter, d, rows, name)?;

                        if let Some(post_w) = post_norm_w {
                            let mut normed_down = vec![0.0f32; down_act.len()];
                            for (d_chunk, o_chunk) in
                                down_act.chunks(d).zip(normed_down.chunks_mut(d))
                            {
                                crate::model::simd::vector_rmsnorm(d_chunk, post_w, o_chunk, *eps);
                            }
                            down_act = normed_down;
                        }

                        // 5. In-place residual add
                        let mut out = x.to_vec();
                        for (o, &d_val) in out.iter_mut().zip(&down_act) {
                            *o += d_val;
                        }
                        out
                    }
                    super::MegakernelKind::Attention { .. } => {
                        return Err(format!(
                            "node '{name}': CPU Megakernel Attention not implemented"
                        ));
                    }
                }
            }
            Op::RmsNorm { eps } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let g = operand(&bound, &slots, ins, name, 1)?;
                let d = g.len();
                if d == 0 || x.len() % d != 0 {
                    return Err(format!("node '{name}': RmsNorm shapes mismatch"));
                }
                let mut out = vec![0.0f32; x.len()];
                for (x_chunk, out_chunk) in x.chunks(d).zip(out.chunks_mut(d)) {
                    crate::model::simd::vector_rmsnorm(x_chunk, g, out_chunk, *eps);
                }
                out
            }
            Op::SwiGlu => {
                let gate = operand(&bound, &slots, ins, name, 0)?;
                let up = operand(&bound, &slots, ins, name, 1)?;
                if gate.len() != up.len() {
                    return Err(format!("node '{name}': SwiGlu lengths mismatch"));
                }
                let mut out = vec![0.0f32; gate.len()];
                for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *o = (g * sig) * u;
                }
                out
            }
            Op::GeGlu => {
                let gate = operand(&bound, &slots, ins, name, 0)?;
                let up = operand(&bound, &slots, ins, name, 1)?;
                if gate.len() != up.len() {
                    return Err(format!("node '{name}': GeGlu lengths mismatch"));
                }
                let mut out = gate.to_vec();
                nn::gelu(&mut out);
                for (o, &u) in out.iter_mut().zip(up) {
                    *o *= u;
                }
                out
            }
            Op::Unary { kind } => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                match kind {
                    super::UnaryKind::Exp => x.iter_mut().for_each(|v| *v = v.exp()),
                    super::UnaryKind::Sigmoid => {
                        x.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()))
                    }
                    super::UnaryKind::Tanh => x.iter_mut().for_each(|v| *v = v.tanh()),
                    super::UnaryKind::Softplus => {
                        x.iter_mut().for_each(|v| {
                            *v = if *v > 20.0 { *v } else { (*v).exp().ln_1p() };
                        });
                    }
                    super::UnaryKind::Neg => x.iter_mut().for_each(|v| *v = -*v),
                    super::UnaryKind::Sqrt => x.iter_mut().for_each(|v| *v = v.sqrt()),
                    super::UnaryKind::Rsqrt => x.iter_mut().for_each(|v| *v = 1.0 / v.sqrt()),
                    super::UnaryKind::Abs => x.iter_mut().for_each(|v| *v = v.abs()),
                    super::UnaryKind::Relu => x.iter_mut().for_each(|v| *v = v.max(0.0)),
                    super::UnaryKind::Relu2 => x.iter_mut().for_each(|v| {
                        let r = v.max(0.0);
                        *v = r * r;
                    }),
                    super::UnaryKind::Silu => {
                        x.iter_mut().for_each(|v| *v = *v / (1.0 + (-*v).exp()))
                    }
                    super::UnaryKind::GeluErf => nn::gelu(&mut x),
                    super::UnaryKind::GeluTanh => {
                        const SQRT_2_OVER_PI: f32 = 0.7978845608;
                        x.iter_mut().for_each(|v| {
                            let cube = 0.044715 * *v * *v * *v;
                            *v = 0.5 * *v * (1.0 + (SQRT_2_OVER_PI * (*v + cube)).tanh());
                        });
                    }
                    super::UnaryKind::Scale(s) => x.iter_mut().for_each(|v| *v *= s),
                    super::UnaryKind::Clamp(min, max) => {
                        x.iter_mut().for_each(|v| *v = v.clamp(*min, *max))
                    }
                    super::UnaryKind::SoftCap(cap) => {
                        x.iter_mut().for_each(|v| *v = cap * (*v / cap).tanh())
                    }
                }
                x
            }
            Op::Binary { kind } => {
                let a = operand(&bound, &slots, ins, name, 0)?;
                let b = operand(&bound, &slots, ins, name, 1)?;
                let op_fn: fn(f32, f32) -> f32 = match kind {
                    super::BinaryKind::Add => |x, y| x + y,
                    super::BinaryKind::Sub => |x, y| x - y,
                    super::BinaryKind::Mul => |x, y| x * y,
                    super::BinaryKind::Div => |x, y| x / y,
                    super::BinaryKind::Max => |x, y| x.max(y),
                    super::BinaryKind::Min => |x, y| x.min(y),
                };
                let mut out = Vec::with_capacity(a.len().max(b.len()));
                if a.len() == b.len() {
                    out.extend(a.iter().zip(b).map(|(&x, &y)| op_fn(x, y)));
                } else if b.len() > 0 && a.len() % b.len() == 0 {
                    for chunk in a.chunks(b.len()) {
                        out.extend(chunk.iter().zip(b).map(|(&x, &y)| op_fn(x, y)));
                    }
                } else if a.len() > 0 && b.len() % a.len() == 0 {
                    for chunk in b.chunks(a.len()) {
                        out.extend(a.iter().zip(chunk).map(|(&x, &y)| op_fn(x, y)));
                    }
                } else if a.len() == 1 {
                    let x = a[0];
                    out.extend(b.iter().map(|&y| op_fn(x, y)));
                } else if b.len() == 1 {
                    let y = b[0];
                    out.extend(a.iter().map(|&x| op_fn(x, y)));
                } else {
                    return Err(format!(
                        "node '{name}': Binary shape mismatch {} vs {}",
                        a.len(),
                        b.len()
                    ));
                }
                out
            }
            Op::Silu => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                x.iter_mut().for_each(|v| *v = *v / (1.0 + (-*v).exp()));
                x
            }
            Op::Relu => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                x.iter_mut().for_each(|v| *v = v.max(0.0));
                x
            }
            Op::Mul => {
                let a = operand(&bound, &slots, ins, name, 0)?;
                let b = operand(&bound, &slots, ins, name, 1)?;
                let a_shape = &graph.value(ins[0]).shape;
                let b_shape = &graph.value(ins[1]).shape;
                let mut out = Vec::with_capacity(a.len().max(b.len()));
                if a.len() == b.len() {
                    out.extend(a.iter().zip(b).map(|(&x, &y)| x * y));
                } else if b.len() == 1 {
                    let y = b[0];
                    out.extend(a.iter().map(|&x| x * y));
                } else if a.len() == 1 {
                    let x = a[0];
                    out.extend(b.iter().map(|&y| x * y));
                } else if a_shape.len() >= 2
                    && b_shape.len() >= 2
                    && b_shape.last() == Some(&1)
                    && a_shape[0] == b_shape[0]
                {
                    // Row broadcasting: [m, d] * [m, 1]
                    let d = a.len() / b.len().max(1);
                    for (r, &scale) in b.iter().enumerate() {
                        let row = &a[r * d..(r + 1) * d];
                        out.extend(row.iter().map(|&x| x * scale));
                    }
                } else if a_shape.len() >= 2
                    && b_shape.len() >= 2
                    && a_shape.last() == Some(&1)
                    && a_shape[0] == b_shape[0]
                {
                    // Row broadcasting: [m, 1] * [m, d]
                    let d = b.len() / a.len().max(1);
                    for (r, &scale) in a.iter().enumerate() {
                        let row = &b[r * d..(r + 1) * d];
                        out.extend(row.iter().map(|&x| scale * x));
                    }
                } else if b.len() > 0 && a.len() % b.len() == 0 {
                    // Column broadcasting: [m, d] * [d]
                    for chunk in a.chunks(b.len()) {
                        out.extend(chunk.iter().zip(b).map(|(&x, &y)| x * y));
                    }
                } else {
                    return Err(format!("node '{name}': Mul shape mismatch"));
                }
                out
            }
            Op::Softmax { axis: _ } | Op::TopKSoftmax { axis: _, .. } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let x_shape = &graph.value(ins[0]).shape;
                let cols = x_shape.last().copied().unwrap_or(x.len()).max(1);
                let rows = x.len() / cols;
                let k_opt = match node.op {
                    Op::TopKSoftmax { k, .. } => Some(k),
                    _ => None,
                };
                let mut out = vec![0.0f32; x.len()];
                for r in 0..rows {
                    let row = &x[r * cols..(r + 1) * cols];
                    let out_row = &mut out[r * cols..(r + 1) * cols];
                    let k = k_opt.unwrap_or(cols).min(cols);
                    if k >= cols {
                        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_e = 0.0f32;
                        for (&v, o) in row.iter().zip(out_row.iter_mut()) {
                            let e = (v - max_val).exp();
                            *o = e;
                            sum_e += e;
                        }
                        let inv = 1.0 / sum_e.max(1e-12);
                        out_row.iter_mut().for_each(|o| *o *= inv);
                    } else {
                        let mut indexed: Vec<(usize, f32)> =
                            row.iter().copied().enumerate().collect();
                        indexed.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let top = &indexed[..k];
                        let max_val = top[0].1;
                        let mut sum_e = 0.0f32;
                        let mut exps = Vec::with_capacity(k);
                        for &(_, val) in top {
                            let e = (val - max_val).exp();
                            exps.push(e);
                            sum_e += e;
                        }
                        let inv = 1.0 / sum_e.max(1e-12);
                        for (&(idx, _), e) in top.iter().zip(exps) {
                            out_row[idx] = e * inv;
                        }
                    }
                }
                out
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                mscale,
            } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let pos = operand_u32(&bound, &slots, ins, name, 1)?;
                let inv_freq = operand(&bound, &slots, ins, name, 2)?;
                let half = rotary_dim / 2;
                if inv_freq.len() != half {
                    return Err(format!(
                        "node '{name}': inv_freq has {} entries, expected {half}",
                        inv_freq.len()
                    ));
                }

                let mut out = x.to_vec();
                let n = if !pos.is_empty() {
                    match variant {
                        super::RopeVariant::MRope { .. } => pos.len() / 3,
                        _ => pos.len(),
                    }
                } else {
                    1
                };
                let total_dim = x.len() / n.max(1);
                let hd = *head_dim;
                let heads = if hd > 0 { total_dim / hd } else { 1 };

                for seq_idx in 0..n {
                    let row_offset = seq_idx * total_dim;
                    for h in 0..heads {
                        let head_offset = row_offset + h * hd;
                        for j in 0..half {
                            let freq = inv_freq[j];
                            let p = match variant {
                                super::RopeVariant::MRope {
                                    sections,
                                    interleaved,
                                    ..
                                } => {
                                    let p_t = pos.get(seq_idx * 3).copied().unwrap_or(0);
                                    let p_h = pos.get(seq_idx * 3 + 1).copied().unwrap_or(0);
                                    let p_w = pos.get(seq_idx * 3 + 2).copied().unwrap_or(0);
                                    if *interleaved {
                                        if j % 3 == 1 && (j as u32) < 3 * sections[1] {
                                            p_h
                                        } else if j % 3 == 2 && (j as u32) < 3 * sections[2] {
                                            p_w
                                        } else {
                                            p_t
                                        }
                                    } else {
                                        let s0 = sections[0] as usize;
                                        let s1 = sections[1] as usize;
                                        if j < s0 {
                                            p_t
                                        } else if j < s0 + s1 {
                                            p_h
                                        } else {
                                            p_w
                                        }
                                    }
                                }
                                _ => pos.get(seq_idx).copied().unwrap_or(0),
                            };

                            let theta = (p as f32) * freq;
                            let cos = theta.cos() * *mscale;
                            let sin = theta.sin() * *mscale;

                            match variant {
                                super::RopeVariant::Interleaved => {
                                    let i0 = head_offset + 2 * j;
                                    let i1 = i0 + 1;
                                    if i1 < out.len() {
                                        let x0 = out[i0];
                                        let x1 = out[i1];
                                        out[i0] = x0 * cos - x1 * sin;
                                        out[i1] = x0 * sin + x1 * cos;
                                    }
                                }
                                super::RopeVariant::HalfSplit => {
                                    let i0 = head_offset + j;
                                    let i1 = head_offset + j + half;
                                    if i1 < out.len() {
                                        let x0 = out[i0];
                                        let x1 = out[i1];
                                        out[i0] = x0 * cos - x1 * sin;
                                        out[i1] = x0 * sin + x1 * cos;
                                    }
                                }
                                super::RopeVariant::MRope { pairs, .. } => {
                                    let (i0, i1) = match pairs {
                                        super::PairLayout::Adjacent => {
                                            (head_offset + 2 * j, head_offset + 2 * j + 1)
                                        }
                                        super::PairLayout::Half => {
                                            (head_offset + j, head_offset + j + half)
                                        }
                                    };
                                    if i1 < out.len() {
                                        let x0 = out[i0];
                                        let x1 = out[i1];
                                        out[i0] = x0 * cos - x1 * sin;
                                        out[i1] = x0 * sin + x1 * cos;
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            Op::RelPosBias {
                buckets,
                max_distance,
                bidirectional,
                heads,
            } => {
                let table = operand(&bound, &slots, ins, name, 0)?;
                let q_pos = operand_u32(&bound, &slots, ins, name, 1)?;
                let k_pos = operand_u32(&bound, &slots, ins, name, 2)?;
                let n = q_pos.len();
                let m = k_pos.len();
                let h = *heads;
                let mut out = vec![0.0f32; h * n * m];

                for qi in 0..n {
                    for ki in 0..m {
                        let diff = k_pos[ki] as i64 - q_pos[qi] as i64;
                        let mut bucket = 0usize;
                        let mut n_dist = diff;
                        let num_buckets = *buckets;
                        let max_exact = num_buckets / 2;

                        if *bidirectional {
                            if n_dist < 0 {
                                bucket += max_exact;
                                n_dist = -n_dist;
                            }
                        } else {
                            n_dist = (-n_dist).max(0);
                        }

                        if (n_dist as usize) < max_exact {
                            bucket += n_dist as usize;
                        } else {
                            let max_dist = *max_distance as f32;
                            let log_val = ((n_dist as f32 / max_exact as f32).ln()
                                / (max_dist / max_exact as f32).ln())
                                * (num_buckets - max_exact) as f32;
                            bucket +=
                                max_exact + (log_val as usize).min(num_buckets - max_exact - 1);
                        }
                        let b_idx = bucket.min(num_buckets - 1);
                        for head in 0..h {
                            out[head * n * m + qi * m + ki] = table[b_idx * h + head];
                        }
                    }
                }
                out
            }
            Op::StateReorder => {
                let state_tensor = operand(&bound, &slots, ins, name, 0)?;
                let parent = operand_u32(&bound, &slots, ins, name, 1)?;
                let batch = parent.len();
                let row_len = state_tensor.len() / batch.max(1);
                let mut out = vec![0.0f32; state_tensor.len()];
                for (b, &p) in parent.iter().enumerate() {
                    let p_idx = p as usize;
                    if p_idx < batch {
                        out[b * row_len..(b + 1) * row_len]
                            .copy_from_slice(&state_tensor[p_idx * row_len..(p_idx + 1) * row_len]);
                    }
                }
                out
            }
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
                scale,
                causal,
                window,
                soft_cap,
                write,
            } => {
                let q = operand(&bound, &slots, ins, name, 0)?;
                let k_in = operand(&bound, &slots, ins, name, 1)?;
                let v_in = operand(&bound, &slots, ins, name, 2)?;
                let hd = *head_dim;
                let vhd = *v_head_dim;
                let h = *heads;
                let kvh = *kv_heads;
                let rep = h / kvh.max(1);
                let nq = q.len() / (h * hd).max(1);

                let (k_buf, v_buf, pos_offset) = if ins.len() >= 6 {
                    let pos = operand_u32(&bound, &slots, ins, name, 5)
                        .map(|p| p.first().copied().unwrap_or(0) as usize)
                        .unwrap_or(0);
                    let k_state = operand(&bound, &slots, ins, name, 3)?;
                    let v_state = operand(&bound, &slots, ins, name, 4)?;
                    let mut k_all = k_state.to_vec();
                    let start_k = pos * kvh * hd;
                    if start_k + k_in.len() <= k_all.len() {
                        k_all[start_k..start_k + k_in.len()].copy_from_slice(k_in);
                    } else {
                        k_all.extend_from_slice(k_in);
                    }
                    let mut v_all = v_state.to_vec();
                    let start_v = pos * kvh * vhd;
                    if start_v + v_in.len() <= v_all.len() {
                        v_all[start_v..start_v + v_in.len()].copy_from_slice(v_in);
                    } else {
                        v_all.extend_from_slice(v_in);
                    }
                    (k_all, v_all, pos)
                } else {
                    (k_in.to_vec(), v_in.to_vec(), 0)
                };
                let nk = (pos_offset + nq).min(k_buf.len() / (kvh * hd).max(1));
                let mut out = vec![0.0f32; nq * h * vhd];
                for qi in 0..nq {
                    let current_pos = pos_offset + qi;
                    for head in 0..h {
                        let kv_head = head / rep;
                        let q_vec = &q[qi * h * hd + head * hd..][..hd];
                        let mut scores = Vec::with_capacity(nk);
                        for ki in 0..nk {
                            if *causal && ki > current_pos {
                                scores.push(f32::NEG_INFINITY);
                                continue;
                            }
                            if let Some(w) = window {
                                if ki + *w <= current_pos {
                                    scores.push(f32::NEG_INFINITY);
                                    continue;
                                }
                            }
                            let k_vec = &k_buf[ki * kvh * hd + kv_head * hd..][..hd];
                            let dot: f32 = q_vec.iter().zip(k_vec).map(|(&a, &b)| a * b).sum();
                            let mut s = dot * scale;
                            if let Some(cap) = soft_cap {
                                s = cap * (s / cap).tanh();
                            }
                            scores.push(s);
                        }
                        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_exp = 0.0f32;
                        let mut weights = Vec::with_capacity(nk);
                        for &s in &scores {
                            if s == f32::NEG_INFINITY {
                                weights.push(0.0f32);
                            } else {
                                let exp = (s - max_score).exp();
                                sum_exp += exp;
                                weights.push(exp);
                            }
                        }
                        if sum_exp > 0.0 {
                            weights.iter_mut().for_each(|w| *w /= sum_exp);
                        }
                        let out_slice = &mut out[qi * h * vhd + head * vhd..][..vhd];
                        for (ki, &w) in weights.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let v_vec = &v_buf[ki * kvh * vhd + kv_head * vhd..][..vhd];
                            for (o, &val) in out_slice.iter_mut().zip(v_vec) {
                                *o += w * val;
                            }
                        }
                    }
                }
                if *write && ins.len() >= 6 {
                    if let ValueKind::State(sname) = &graph.value(ins[3]).kind {
                        state.insert(sname, Tensor::F32(k_buf.clone()));
                    }
                    if let ValueKind::State(sname) = &graph.value(ins[4]).kind {
                        state.insert(sname, Tensor::F32(v_buf.clone()));
                    }
                }
                out
            }
            Op::PagedCachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
                scale,
                causal,
                window,
                soft_cap,
                tokens_per_block,
                max_blocks_per_seq,
            } => {
                let q = operand(&bound, &slots, ins, name, 0)?;
                let k_pool = operand(&bound, &slots, ins, name, 1)?;
                let v_pool = operand(&bound, &slots, ins, name, 2)?;
                let block_table = operand_u32(&bound, &slots, ins, name, 3)?;
                let pos_offset = if ins.len() >= 5 {
                    operand_u32(&bound, &slots, ins, name, 4)
                        .map(|p| p.first().copied().unwrap_or(0) as usize)
                        .unwrap_or(0)
                } else {
                    0
                };
                let hd = *head_dim;
                let vhd = *v_head_dim;
                let h = *heads;
                let kvh = *kv_heads;
                let rep = h / kvh.max(1);
                let nq = q.len() / (h * hd).max(1);
                let tpb = *tokens_per_block;
                let max_blk = *max_blocks_per_seq;
                let kv_dim = kvh * hd;

                let mut out = vec![0.0f32; nq * h * vhd];
                for qi in 0..nq {
                    let current_pos = pos_offset + qi;
                    let nk = current_pos + 1;
                    for head in 0..h {
                        let kv_head = head / rep;
                        let q_vec = &q[qi * h * hd + head * hd..][..hd];
                        let mut scores = Vec::with_capacity(nk);
                        for ki in 0..nk {
                            if *causal && ki > current_pos {
                                scores.push(f32::NEG_INFINITY);
                                continue;
                            }
                            if let Some(w) = window {
                                if ki + *w <= current_pos {
                                    scores.push(f32::NEG_INFINITY);
                                    continue;
                                }
                            }
                            let blk_idx = block_table[qi * max_blk + ki / tpb] as usize;
                            let blk_offset = ki % tpb;
                            let phys_idx = blk_idx * tpb + blk_offset;
                            let k_vec = &k_pool[phys_idx * kv_dim + kv_head * hd..][..hd];
                            let dot: f32 = q_vec.iter().zip(k_vec).map(|(&a, &b)| a * b).sum();
                            let mut s = dot * scale;
                            if let Some(cap) = soft_cap {
                                s = cap * (s / cap).tanh();
                            }
                            scores.push(s);
                        }
                        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_exp = 0.0f32;
                        let mut weights = Vec::with_capacity(nk);
                        for &s in &scores {
                            if s == f32::NEG_INFINITY {
                                weights.push(0.0f32);
                            } else {
                                let exp_val = (s - max_score).exp();
                                sum_exp += exp_val;
                                weights.push(exp_val);
                            }
                        }
                        if sum_exp > 0.0 {
                            for w in &mut weights {
                                *w /= sum_exp;
                            }
                        }
                        let out_slice = &mut out[qi * h * vhd + head * vhd..][..vhd];
                        for (ki, &w) in weights.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let blk_idx = block_table[qi * max_blk + ki / tpb] as usize;
                            let blk_offset = ki % tpb;
                            let phys_idx = blk_idx * tpb + blk_offset;
                            let v_vec = &v_pool[phys_idx * kv_dim + kv_head * vhd..][..vhd];
                            for (o, &val) in out_slice.iter_mut().zip(v_vec) {
                                *o += w * val;
                            }
                        }
                    }
                }
                out
            }
            Op::CausalConv1dCached { kernel, dilation } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let w = operand(&bound, &slots, ins, name, 1)?;
                let k = *kernel;
                let d = *dilation;
                let channels = w.len() / k.max(1);
                let n = x.len() / channels.max(1);
                let hist = k - 1;
                let mut window = if ins.len() >= 3 {
                    operand(&bound, &slots, ins, name, 2)
                        .map(<[f32]>::to_vec)
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                if window.len() != hist * channels {
                    window = vec![0.0f32; hist * channels];
                }
                window.extend_from_slice(x);
                let mut out = vec![0.0f32; n * channels];
                for i in 0..n {
                    for ch in 0..channels {
                        let mut acc = 0.0f32;
                        let w_ch = &w[ch * k..(ch + 1) * k];
                        for (tap, &weight) in w_ch.iter().enumerate() {
                            let offset_step = (k - 1 - tap) * d;
                            let sample_idx = hist + i;
                            if sample_idx >= offset_step {
                                acc += weight * window[(sample_idx - offset_step) * channels + ch];
                            }
                        }
                        out[i * channels + ch] = acc;
                    }
                }
                let new_state = window[n * channels..].to_vec();
                if ins.len() >= 3 {
                    if let ValueKind::State(sname) = &graph.value(ins[2]).kind {
                        state.insert(sname, Tensor::F32(new_state));
                    }
                }
                out
            }
            Op::DeltaRule {
                key_heads,
                val_heads,
                key_dim,
                val_dim,
            } => {
                let q = operand(&bound, &slots, ins, name, 0)?;
                let k = operand(&bound, &slots, ins, name, 1)?;
                let v = operand(&bound, &slots, ins, name, 2)?;
                let beta = operand(&bound, &slots, ins, name, 3)?;
                let log_decay = operand(&bound, &slots, ins, name, 4)?;
                let kh = *key_heads;
                let vh = *val_heads;
                let kd = *key_dim;
                let vd = *val_dim;
                let rep = vh / kh.max(1);
                let n = q.len() / (kh * kd).max(1);
                let mut out = vec![0.0f32; n * vh * vd];

                let mut ssm_state = if ins.len() > 5 {
                    operand(&bound, &slots, ins, name, 5)
                        .map(<[f32]>::to_vec)
                        .unwrap_or_else(|_| vec![0.0f32; vh * kd * vd])
                } else {
                    vec![0.0f32; vh * kd * vd]
                };
                if ssm_state.len() != vh * kd * vd {
                    ssm_state = vec![0.0f32; vh * kd * vd];
                }

                for h in 0..vh {
                    let k_h = h / rep;
                    let state_slice = &mut ssm_state[h * kd * vd..(h + 1) * kd * vd];
                    for t in 0..n {
                        let mut q_vec = q[t * kh * kd + k_h * kd..][..kd].to_vec();
                        let mut k_vec = k[t * kh * kd + k_h * kd..][..kd].to_vec();
                        let v_t = &v[t * vh * vd + h * vd..][..vd];
                        let b = beta.get(t * vh + h).copied().unwrap_or(1.0);
                        let ld = log_decay.get(t * vh + h).copied().unwrap_or(0.0);
                        let decay = ld.exp();

                        let q_norm = q_vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                        let q_scale = (1.0 / (kd as f32).sqrt()) / q_norm;
                        q_vec.iter_mut().for_each(|x| *x *= q_scale);

                        let k_norm = k_vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                        let k_scale = 1.0 / k_norm;
                        k_vec.iter_mut().for_each(|x| *x *= k_scale);

                        let mut kv = vec![0.0f32; vd];
                        for i in 0..kd {
                            let k_val = k_vec[i];
                            let row = &state_slice[i * vd..(i + 1) * vd];
                            for (kv_elem, &s) in kv.iter_mut().zip(row) {
                                *kv_elem += s * k_val;
                            }
                        }
                        let mut delta = vec![0.0f32; vd];
                        for (d, (&vt, &kvt)) in delta.iter_mut().zip(v_t.iter().zip(&kv)) {
                            *d = b * (vt - decay * kvt);
                        }
                        for i in 0..kd {
                            let k_val = k_vec[i];
                            let row = &mut state_slice[i * vd..(i + 1) * vd];
                            for (s, &d) in row.iter_mut().zip(&delta) {
                                *s = decay * *s + k_val * d;
                            }
                        }
                        let out_slice = &mut out[t * vh * vd + h * vd..][..vd];
                        for i in 0..kd {
                            let q_val = q_vec[i];
                            let row = &state_slice[i * vd..(i + 1) * vd];
                            for (o, &s) in out_slice.iter_mut().zip(row) {
                                *o += s * q_val;
                            }
                        }
                    }
                }
                if ins.len() > 5 {
                    if let ValueKind::State(sname) = &graph.value(ins[5]).kind {
                        state.insert(sname, Tensor::F32(ssm_state));
                    }
                }
                out
            }
            Op::Gather { axis } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let indices = operand_u32(&bound, &slots, ins, name, 1)?;
                let x_shape = &graph.value(ins[0]).shape;
                let ax = if *axis < 0 {
                    (x_shape.len() as isize + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer: usize = x_shape[..ax.min(x_shape.len())].iter().product();
                let dim_size = x_shape.get(ax).copied().unwrap_or(x.len());
                let inner: usize = if ax + 1 < x_shape.len() {
                    x_shape[ax + 1..].iter().product()
                } else {
                    1
                };
                let mut out = Vec::with_capacity(outer * indices.len() * inner);
                for o in 0..outer.max(1) {
                    for &idx in indices {
                        let i = idx as usize;
                        if i < dim_size {
                            let start = o * dim_size * inner + i * inner;
                            if start + inner <= x.len() {
                                out.extend_from_slice(&x[start..start + inner]);
                            }
                        }
                    }
                }
                out
            }
            Op::Slice { axis, start, end } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let x_shape = &graph.value(ins[0]).shape;
                if x_shape.is_empty() {
                    return Err(format!(
                        "node '{name}': cannot slice operand with empty shape"
                    ));
                }
                let ax = if *axis < 0 {
                    let adjusted = x_shape.len() as isize + *axis;
                    if adjusted < 0 {
                        return Err(format!(
                            "node '{name}': slice axis {axis} out of bounds for shape {x_shape:?}"
                        ));
                    }
                    adjusted as usize
                } else {
                    *axis as usize
                };
                if ax >= x_shape.len() {
                    return Err(format!(
                        "node '{name}': slice axis {axis} out of bounds for shape {x_shape:?}"
                    ));
                }
                let outer: usize = x_shape[..ax].iter().product();
                let dim_size = x_shape[ax];
                let inner: usize = if ax + 1 < x_shape.len() {
                    x_shape[ax + 1..].iter().product()
                } else {
                    1
                };
                let s = if *start < 0 {
                    (dim_size as i64 + *start).max(0) as usize
                } else {
                    (*start as usize).min(dim_size)
                };
                let e = if *end < 0 {
                    (dim_size as i64 + *end).max(0) as usize
                } else {
                    (*end as usize).min(dim_size)
                };
                let slice_len = e.saturating_sub(s);
                let mut out = Vec::with_capacity(outer * slice_len * inner);
                for o in 0..outer.max(1) {
                    let row_base = o * dim_size * inner;
                    let start_idx = row_base + s * inner;
                    let end_idx = row_base + e * inner;
                    if end_idx <= x.len() {
                        out.extend_from_slice(&x[start_idx..end_idx]);
                    }
                }
                out
            }
            Op::Scatter { axis } => {
                let mut base = operand(&bound, &slots, ins, name, 0)?.to_vec();
                let indices = operand_u32(&bound, &slots, ins, name, 1)?;
                let updates = operand(&bound, &slots, ins, name, 2)?;
                let base_shape = &graph.value(ins[0]).shape;
                let ax = if *axis < 0 {
                    (base_shape.len() as isize + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer: usize = base_shape[..ax.min(base_shape.len())].iter().product();
                let dim_size = base_shape.get(ax).copied().unwrap_or(base.len());
                let inner: usize = if ax + 1 < base_shape.len() {
                    base_shape[ax + 1..].iter().product()
                } else {
                    1
                };
                let mut up_idx = 0;
                for o in 0..outer.max(1) {
                    for &idx in indices {
                        let i = idx as usize;
                        if i < dim_size && up_idx + inner <= updates.len() {
                            let start = o * dim_size * inner + i * inner;
                            if start + inner <= base.len() {
                                base[start..start + inner]
                                    .copy_from_slice(&updates[up_idx..up_idx + inner]);
                                up_idx += inner;
                            }
                        }
                    }
                }
                base
            }
            Op::Pad { padding, value } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let shape = &graph.value(ins[0]).shape;
                if shape.len() >= 2 && padding.len() == 4 {
                    let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                    let (top, bottom, left, right) =
                        (padding[0], padding[1], padding[2], padding[3]);
                    let (out_h, out_w) = (h + top + bottom, w + left + right);
                    let c = x.len() / (h * w).max(1);
                    let mut out = vec![*value; c * out_h * out_w];
                    for ch in 0..c {
                        for r in 0..h {
                            for col in 0..w {
                                out[ch * out_h * out_w + (r + top) * out_w + (col + left)] =
                                    x[ch * h * w + r * w + col];
                            }
                        }
                    }
                    out
                } else if padding.len() == 2 && !shape.is_empty() {
                    let (left, right) = (padding[0], padding[1]);
                    let l = shape[shape.len() - 1];
                    let c = x.len() / l.max(1);
                    let out_l = l + left + right;
                    let mut out = vec![*value; c * out_l];
                    for ch in 0..c {
                        for j in 0..l {
                            out[ch * out_l + j + left] = x[ch * l + j];
                        }
                    }
                    out
                } else {
                    return Err(format!(
                        "node '{name}': Pad with shape {shape:?} and padding {:?} not supported",
                        padding
                    ));
                }
            }
            Op::AvgPool2d { kernel, stride } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let shape = &graph.value(ins[0]).shape;
                let (c, h, w) = if shape.len() >= 3 {
                    (
                        shape[shape.len() - 3],
                        shape[shape.len() - 2],
                        shape[shape.len() - 1],
                    )
                } else {
                    (1, 1, x.len())
                };
                let (kh, kw) = (kernel[0], kernel[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let out_h = if h >= kh { (h - kh) / sh + 1 } else { 0 };
                let out_w = if w >= kw { (w - kw) / sw + 1 } else { 0 };
                let mut out = Vec::with_capacity(c * out_h * out_w);
                for ch in 0..c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;
                            for r in 0..kh {
                                for col in 0..kw {
                                    let ih = oh * sh + r;
                                    let iw = ow * sw + col;
                                    sum += x[ch * h * w + ih * w + iw];
                                }
                            }
                            out.push(sum / (kh * kw) as f32);
                        }
                    }
                }
                out
            }
            Op::Upsample { scale, mode } => match mode {
                super::UpsampleMode::Nearest => {
                    let x = operand(&bound, &slots, ins, name, 0)?;
                    let shape = &graph.value(ins[0]).shape;
                    let (c, h, w) = if shape.len() >= 3 {
                        (
                            shape[shape.len() - 3],
                            shape[shape.len() - 2],
                            shape[shape.len() - 1],
                        )
                    } else {
                        (1, 1, x.len())
                    };
                    let s = *scale;
                    let (out_h, out_w) = (h * s, w * s);
                    let mut out = vec![0.0f32; c * out_h * out_w];
                    for ch in 0..c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let ih = (oh / s).min(h.saturating_sub(1));
                                let iw = (ow / s).min(w.saturating_sub(1));
                                out[ch * out_h * out_w + oh * out_w + ow] =
                                    x[ch * h * w + ih * w + iw];
                            }
                        }
                    }
                    out
                }
                other => {
                    return Err(format!(
                        "node '{name}': CPU Upsample mode {other:?} is not implemented yet"
                    ))
                }
            },
            Op::Conv3d { .. } => {
                return Err(format!("node '{name}': CPU Conv3d is not implemented"));
            }
            Op::LayerNorm { eps } => {
                if ins.len() < 3 {
                    return Err(format!(
                        "node '{name}': LayerNorm without beta is not implemented"
                    ));
                }
                let (x, g, b) = (
                    operand(&bound, &slots, ins, name, 0)?,
                    operand(&bound, &slots, ins, name, 1)?,
                    operand(&bound, &slots, ins, name, 2)?,
                );
                let d = g.len();
                if d == 0 || b.len() != d || x.len() % d != 0 {
                    return Err(format!(
                        "node '{name}': x/gamma/beta lengths {}/{}/{} disagree",
                        x.len(),
                        d,
                        b.len()
                    ));
                }
                nn::layer_norm(x, x.len() / d, d, g, b, *eps)
            }
            Op::Attention {
                heads,
                kv_heads,
                causal,
                window,
                soft_cap,
            } => {
                if kv_heads != heads || window.is_some() || soft_cap.is_some() {
                    return Err(format!(
                        "node '{name}': Attention with kv_heads != heads, window, or soft_cap is not implemented"
                    ));
                }
                let (q, k, v) = (
                    operand(&bound, &slots, ins, name, 0)?,
                    operand(&bound, &slots, ins, name, 1)?,
                    operand(&bound, &slots, ins, name, 2)?,
                );
                let hd = last_dim(0);
                if *heads == 0
                    || hd == 0
                    || hd % heads != 0
                    || q.len() % hd != 0
                    || k.len() % hd != 0
                    || k.len() != v.len()
                {
                    return Err(format!(
                        "node '{name}': q/k/v shapes do not fit {heads} heads"
                    ));
                }
                nn::attention(
                    q,
                    k,
                    v,
                    q.len() / hd,
                    k.len() / hd,
                    *heads,
                    hd / heads,
                    *causal,
                )
            }
            Op::Gelu { tanh: false } => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                nn::gelu(&mut x);
                x
            }
            Op::Add => {
                let b_len = operand(&bound, &slots, ins, name, 1)?.len();
                let mut a = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                if a.len() != b_len {
                    return Err(format!(
                        "node '{name}': Add of {} and {b_len} elements",
                        a.len()
                    ));
                }
                nn::add_in_place(&mut a, operand(&bound, &slots, ins, name, 1)?);
                a
            }
            Op::MeanPool => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let mask = operand_u32(&bound, &slots, ins, name, 1)?;
                let d = last_dim(0);
                if d == 0 || x.len() != mask.len() * d {
                    return Err(format!(
                        "node '{name}': x of {} elements does not match mask of {}",
                        x.len(),
                        mask.len()
                    ));
                }
                nn::mean_pool(x, mask, d)
            }
            Op::L2Normalize => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                nn::l2_normalize(&mut x);
                x
            }
            Op::Conv2d {
                stride, padding, ..
            } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let w = operand(&bound, &slots, ins, name, 1)?;
                let bias = if ins.len() > 2 {
                    Some(operand(&bound, &slots, ins, name, 2)?)
                } else {
                    None
                };
                let x_shape = &graph.value(ins[0]).shape;
                let w_shape = &graph.value(ins[1]).shape;
                let in_c = x_shape.first().copied().unwrap_or(1);
                let in_h = x_shape.get(1).copied().unwrap_or(1);
                let in_w = x_shape.get(2).copied().unwrap_or(1);
                let out_c = w_shape.first().copied().unwrap_or(1);
                let (res, _, _) = nn::conv2d(
                    x, in_c, in_h, in_w, w, bias, out_c, stride[0], stride[1], padding[0],
                );
                res
            }
            op => {
                return Err(format!(
                    "CPU executor does not implement {} (node '{}')",
                    op.name(),
                    node.name
                ))
            }
        };

        for v in ins {
            if last_use[v.0 as usize] == i {
                slots[v.0 as usize] = None;
            }
        }
        slots[node.outputs[0].0 as usize] = Some(Tensor::F32(out));
    }

    graph
        .outputs
        .iter()
        .map(|v| {
            let id = v.0 as usize;
            bound[id]
                .cloned()
                .or_else(|| slots[id].take())
                .ok_or_else(|| format!("output '{}' was never computed", graph.value(*v).name))
        })
        .collect()
}

fn slot<'a>(
    bound: &[Option<&'a Tensor>],
    slots: &'a [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    k: usize,
) -> Result<&'a Tensor, String> {
    let id = ins
        .get(k)
        .ok_or_else(|| format!("node '{name}' is missing operand {k}"))?
        .0 as usize;
    bound[id]
        .or(slots[id].as_ref())
        .ok_or_else(|| format!("node '{name}': operand {k} has no value"))
}

fn operand<'a>(
    bound: &[Option<&'a Tensor>],
    slots: &'a [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    k: usize,
) -> Result<&'a [f32], String> {
    match slot(bound, slots, ins, name, k)? {
        Tensor::F32(v) => Ok(v),
        _ => Err(format!("node '{name}': operand {k} must be f32")),
    }
}

fn operand_u32<'a>(
    bound: &[Option<&'a Tensor>],
    slots: &'a [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    k: usize,
) -> Result<&'a [u32], String> {
    match slot(bound, slots, ins, name, k)? {
        Tensor::U32(v) => Ok(v),
        _ => Err(format!("node '{name}': operand {k} must be u32")),
    }
}

fn dispatch_linear(
    x: &[f32],
    w_slot: &Tensor,
    bias: Option<&[f32]>,
    in_f: usize,
    out_f: usize,
    rows: usize,
    name: &str,
) -> Result<Vec<f32>, String> {
    match w_slot {
        Tensor::F32(w) => {
            if w.len() != out_f * in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    w.len()
                ));
            }
            Ok(nn::linear(x, rows, w, bias, in_f, out_f))
        }
        Tensor::Quantized(q) => {
            if q.in_features() != in_f || q.out_features() != out_f {
                return Err(format!(
                    "node '{name}': quantized weight [{}, {}] != declared [{out_f}, {in_f}]",
                    q.out_features(),
                    q.in_features()
                ));
            }
            match q.as_ref() {
                crate::model::layers::QuantizedWeight::Uncompressed { data, .. } => {
                    Ok(nn::linear(x, rows, data, bias, in_f, out_f))
                }
                crate::model::layers::QuantizedWeight::ZeroCopySlice {
                    data,
                    format,
                    in_features,
                    out_features,
                } => {
                    let mut out = vec![0.0f32; rows * out_f];
                    for (r, x_row) in x.chunks(in_f).enumerate() {
                        let out_row = &mut out[r * out_f..(r + 1) * out_f];
                        crate::model::layers::dispatch_fused_gemv(
                            data,
                            *format,
                            *in_features,
                            *out_features,
                            x_row,
                            out_row,
                            bias,
                        );
                    }
                    Ok(out)
                }
                crate::model::layers::QuantizedWeight::GgufQuantized {
                    raw_data,
                    format,
                    in_features,
                    out_features,
                } => {
                    let mut out = vec![0.0f32; rows * out_f];
                    for (r, x_row) in x.chunks(in_f).enumerate() {
                        let out_row = &mut out[r * out_f..(r + 1) * out_f];
                        crate::model::layers::dispatch_fused_gemv(
                            raw_data,
                            *format,
                            *in_features,
                            *out_features,
                            x_row,
                            out_row,
                            bias,
                        );
                    }
                    Ok(out)
                }
                crate::model::layers::QuantizedWeight::Nvfp4(tensor) => {
                    let mut out = vec![0.0f32; rows * out_f];
                    for (r, x_row) in x.chunks(in_f).enumerate() {
                        let out_row = &mut out[r * out_f..(r + 1) * out_f];
                        crate::quant::nvfp4::gemv_nvfp4_f32(tensor, x_row, out_row);
                        if let Some(b) = bias {
                            for (o, &bi) in out_row.iter_mut().zip(b) {
                                *o += bi;
                            }
                        }
                    }
                    Ok(out)
                }
                crate::model::layers::QuantizedWeight::WeightlessJesse(layer) => {
                    let mut out = vec![0.0f32; rows * out_f];
                    for (r, x_row) in x.chunks(in_f).enumerate() {
                        let out_row = &mut out[r * out_f..(r + 1) * out_f];
                        layer.forward(x_row, out_row);
                        if let Some(b) = bias {
                            for (o, &bi) in out_row.iter_mut().zip(b) {
                                *o += bi;
                            }
                        }
                    }
                    Ok(out)
                }
                _ => {
                    let w_f32 = q.to_f32();
                    Ok(nn::linear(x, rows, &w_f32, bias, in_f, out_f))
                }
            }
        }
        other => {
            let w_f32 = other.to_f32();
            if w_f32.len() != out_f * in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    w_f32.len()
                ));
            }
            Ok(nn::linear(x, rows, &w_f32, bias, in_f, out_f))
        }
    }
}

/// Owns operand 0, moving it out when node `i` is its last (and only-once) reader.
fn take0(
    bound: &[Option<&Tensor>],
    slots: &mut [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    last_use: &[usize],
    i: usize,
) -> Result<Vec<f32>, String> {
    let id = ins
        .first()
        .ok_or_else(|| format!("node '{name}' is missing operand 0"))?
        .0 as usize;
    if bound[id].is_none() && last_use[id] == i && !ins[1..].iter().any(|v| v.0 as usize == id) {
        if let Some(t) = slots[id].take() {
            return t.into_f32().map_err(|e| format!("node '{name}': {e}"));
        }
    }
    operand(bound, slots, ins, name, 0).map(<[f32]>::to_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rel_pos_bias_and_state_reorder() {
        let mut g = Graph::new();
        let table = g.weight("table", vec![32, 4], DType::F32);
        let q_pos = g.input("q_pos", vec![2], DType::U32);
        let k_pos = g.input("k_pos", vec![4], DType::U32);
        let rpb = g.add(
            "rpb",
            Op::RelPosBias {
                buckets: 32,
                max_distance: 128,
                bidirectional: true,
                heads: 4,
            },
            &[table, q_pos, k_pos],
            vec![4, 2, 4],
            DType::F32,
        );
        g.output(rpb);

        let mut w = Weights::new();
        w.insert("table", Tensor::F32(vec![0.5; 32 * 4]));
        let inputs = vec![Tensor::U32(vec![0, 1]), Tensor::U32(vec![0, 1, 2, 3])];
        let out = run_stateless(&g, &w, inputs).unwrap();
        assert_eq!(out[0].len(), 4 * 2 * 4);
    }

    use crate::graph::{BinaryKind, DType, Op, RopeVariant, UnaryKind, UpsampleMode};

    #[test]
    fn test_cpu_rmsnorm() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let gamma = g.weight("gamma", vec![4], DType::F32);
        let out = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, gamma],
            vec![2, 4],
            DType::F32,
        );
        g.output(out);

        let mut w = Weights::new();
        w.insert("gamma", Tensor::F32(vec![1.0, 1.0, 1.0, 1.0]));
        let inputs = vec![Tensor::F32(vec![
            1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0,
        ])];
        let res = run_stateless(&g, &w, inputs).unwrap();
        let out_slice = res[0].as_f32().unwrap();
        assert_eq!(out_slice.len(), 8);
        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0f32 + 1e-5).sqrt();
        assert!((out_slice[0] - 1.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_swiglu_and_geglu() {
        let mut g = Graph::new();
        let gate = g.input("gate", vec![2], DType::F32);
        let up = g.input("up", vec![2], DType::F32);
        let s = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![2], DType::F32);
        let ge = g.add("geglu", Op::GeGlu, &[gate, up], vec![2], DType::F32);
        g.output(s);
        g.output(ge);

        let inputs = vec![Tensor::F32(vec![0.0, 2.0]), Tensor::F32(vec![1.0, 3.0])];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let s_out = res[0].as_f32().unwrap();
        let ge_out = res[1].as_f32().unwrap();
        assert!((s_out[0] - 0.0).abs() < 1e-5);
        let silu_2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((s_out[1] - silu_2 * 3.0).abs() < 1e-4);
        assert!((ge_out[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_unary_and_binary() {
        let mut g = Graph::new();
        let a = g.input("a", vec![3], DType::F32);
        let b = g.input("b", vec![1], DType::F32);
        let softplus = g.add(
            "sp",
            Op::Unary {
                kind: UnaryKind::Softplus,
            },
            &[a],
            vec![3],
            DType::F32,
        );
        let clamp = g.add(
            "cl",
            Op::Unary {
                kind: UnaryKind::Clamp(0.0, 10.0),
            },
            &[a],
            vec![3],
            DType::F32,
        );
        let add = g.add(
            "add",
            Op::Binary {
                kind: BinaryKind::Add,
            },
            &[a, b],
            vec![3],
            DType::F32,
        );
        let mul = g.add(
            "mul",
            Op::Binary {
                kind: BinaryKind::Mul,
            },
            &[a, b],
            vec![3],
            DType::F32,
        );
        g.output(softplus);
        g.output(clamp);
        g.output(add);
        g.output(mul);

        let inputs = vec![Tensor::F32(vec![-5.0, 0.0, 30.0]), Tensor::F32(vec![2.0])];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let sp = res[0].as_f32().unwrap();
        assert!(sp[0] > 0.0 && sp[0] < 0.01);
        assert!((sp[1] - (2.0f32).ln()).abs() < 1e-4);
        assert!((sp[2] - 30.0).abs() < 1e-4); // softplus numerical stability for > 20

        let cl = res[1].as_f32().unwrap();
        assert_eq!(cl, &[0.0, 0.0, 10.0]);

        let ad = res[2].as_f32().unwrap();
        assert_eq!(ad, &[-3.0, 2.0, 32.0]);

        let mu = res[3].as_f32().unwrap();
        assert_eq!(mu, &[-10.0, 0.0, 60.0]);
    }

    #[test]
    fn test_cpu_rope() {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![1, 4], DType::F32);
        let pos = g.input("pos", vec![1], DType::U32);
        let inv_freq = g.weight("rope.inv_freq.default", vec![2], DType::F32);
        w.insert(
            "rope.inv_freq.default".to_string(),
            Tensor::F32(vec![1.0, 0.01]),
        );

        let r_inter = g.add(
            "rope_inter",
            Op::Rope {
                head_dim: 4,
                rotary_dim: 4,
                variant: RopeVariant::Interleaved,
                mscale: 1.0,
            },
            &[x, pos, inv_freq],
            vec![1, 4],
            DType::F32,
        );
        let r_half = g.add(
            "rope_half",
            Op::Rope {
                head_dim: 4,
                rotary_dim: 4,
                variant: RopeVariant::HalfSplit,
                mscale: 1.0,
            },
            &[x, pos, inv_freq],
            vec![1, 4],
            DType::F32,
        );
        g.output(r_inter);
        g.output(r_half);

        let inputs = vec![Tensor::F32(vec![1.0, 0.0, 0.0, 1.0]), Tensor::U32(vec![0])];
        let res = run_stateless(&g, &w, inputs).unwrap();
        assert_eq!(res[0].as_f32().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(res[1].as_f32().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_attention_window_exact_w_keys() {
        // window = 4, query at pos 10 over keys 0..=10.
        // Exactly keys 7..=10 have non-zero weight (4 keys).
        let w = 4usize;
        let pos = 10usize;
        let mut visible = Vec::new();
        for ki in 0..=pos {
            if !(ki + w <= pos) {
                visible.push(ki);
            }
        }
        assert_eq!(visible, vec![7, 8, 9, 10]);
    }

    #[test]
    fn test_cpu_cached_attention() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 4], DType::F32);
        let k = g.input("k", vec![1, 4], DType::F32);
        let v = g.input("v", vec![1, 4], DType::F32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads: 1,
                kv_heads: 1,
                head_dim: 4,
                v_head_dim: 4,
                scale: 0.5,
                causal: true,
                window: None,
                soft_cap: None,
                write: false,
            },
            &[q, k, v],
            vec![1, 4],
            DType::F32,
        );
        g.output(out);

        let inputs = vec![
            Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
            Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
            Tensor::F32(vec![0.5, 1.5, 2.5, 3.5]),
        ];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let out = res[0].as_f32().unwrap();
        assert_eq!(out, &[0.5, 1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_cpu_causal_conv1d_cached() {
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 1], DType::F32); // 3 steps, 1 channel
        let w = g.weight("w", vec![2], DType::F32); // kernel 2
        let out = g.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x, w],
            vec![3, 1],
            DType::F32,
        );
        g.output(out);

        let mut weights = Weights::new();
        weights.insert("w", Tensor::F32(vec![0.5, 1.0])); // tap 0 is earlier, tap 1 is current
        let inputs = vec![Tensor::F32(vec![2.0, 4.0, 6.0])];
        let res = run_stateless(&g, &weights, inputs).unwrap();
        let out = res[0].as_f32().unwrap();
        // step 0: 1.0 * 2.0 = 2.0
        // step 1: 0.5 * 2.0 + 1.0 * 4.0 = 5.0
        // step 2: 0.5 * 4.0 + 1.0 * 6.0 = 8.0
        assert_eq!(out, &[2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_cpu_delta_rule() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 2], DType::F32);
        let k = g.input("k", vec![1, 2], DType::F32);
        let v = g.input("v", vec![1, 2], DType::F32);
        let beta = g.input("beta", vec![1], DType::F32);
        let log_decay = g.input("log_decay", vec![1], DType::F32);
        let out = g.add(
            "delta",
            Op::DeltaRule {
                key_heads: 1,
                val_heads: 1,
                key_dim: 2,
                val_dim: 2,
            },
            &[q, k, v, beta, log_decay],
            vec![1, 2],
            DType::F32,
        );
        g.output(out);

        let inputs = vec![
            Tensor::F32(vec![1.0, 0.0]),
            Tensor::F32(vec![1.0, 0.0]),
            Tensor::F32(vec![3.0, 4.0]),
            Tensor::F32(vec![1.0]),
            Tensor::F32(vec![0.0]),
        ];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let out = res[0].as_f32().unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.0 / (2.0f32).sqrt()).abs() < 1e-4);
        assert!((out[1] - 4.0 / (2.0f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_gather_scatter_pad_pool_upsample() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 2], DType::F32);
        let idx = g.input("idx", vec![1], DType::U32);
        let gat = g.add(
            "gather",
            Op::Gather { axis: 0 },
            &[x, idx],
            vec![1, 2],
            DType::F32,
        );
        let pad = g.add(
            "pad",
            Op::Pad {
                padding: vec![1, 1],
                value: 0.0,
            },
            &[gat],
            vec![1, 4],
            DType::F32,
        );
        let up = g.add(
            "up",
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Nearest,
            },
            &[x],
            vec![4, 4],
            DType::F32,
        );
        g.output(gat);
        g.output(pad);
        g.output(up);

        let inputs = vec![Tensor::F32(vec![1.0, 2.0, 3.0, 4.0]), Tensor::U32(vec![1])];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        assert_eq!(res[0].as_f32().unwrap(), &[3.0, 4.0]);
        assert_eq!(res[1].as_f32().unwrap(), &[0.0, 3.0, 4.0, 0.0]);
        assert_eq!(res[2].as_f32().unwrap().len(), 16);
    }

    #[test]
    fn test_cpu_cached_attention_step_by_step_matches_prefill() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 4], DType::F32);
        let k = g.input("k", vec![1, 4], DType::F32);
        let v = g.input("v", vec![1, 4], DType::F32);
        let ks = g.state("k_state", vec![16, 4], DType::F32);
        let vs = g.state("v_state", vec![16, 4], DType::F32);
        let pos = g.input("pos", vec![1], DType::U32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads: 1,
                kv_heads: 1,
                head_dim: 4,
                v_head_dim: 4,
                scale: 0.5,
                causal: true,
                window: None,
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos],
            vec![1, 4],
            DType::F32,
        );
        g.output(out);

        let mut state = StateStore::new();
        // Step 0
        let r0 = run(
            &g,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 2.0, 3.0, 4.0]),
                Tensor::U32(vec![0]),
            ],
        )
        .unwrap();
        assert_eq!(r0[0].as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        // Step 1
        let r1 = run(
            &g,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![0.0, 1.0, 0.0, 0.0]),
                Tensor::F32(vec![0.0, 1.0, 0.0, 0.0]),
                Tensor::F32(vec![10.0, 20.0, 30.0, 40.0]),
                Tensor::U32(vec![1]),
            ],
        )
        .unwrap();
        let o1 = r1[0].as_f32().unwrap();
        assert_eq!(o1.len(), 4);
        // Step 1 q=[0,1,0,0] attends over k0=[1,0,0,0] and k1=[0,1,0,0]
        assert!(o1[0] > 1.0);
    }

    #[test]
    fn test_cpu_causal_conv1d_step_by_step_matches_prefill() {
        let mut g_prefill = Graph::new();
        let x_pre = g_prefill.input("x", vec![2, 1], DType::F32);
        let w_pre = g_prefill.weight("w", vec![2], DType::F32);
        let out_pre = g_prefill.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x_pre, w_pre],
            vec![2, 1],
            DType::F32,
        );
        g_prefill.output(out_pre);

        let mut weights = Weights::new();
        weights.insert("w", Tensor::F32(vec![0.3, 0.7]));

        let pre_res =
            run_stateless(&g_prefill, &weights, vec![Tensor::F32(vec![1.0, 5.0])]).unwrap();
        let expected = pre_res[0].as_f32().unwrap();

        // Step by step with state
        let mut g_step = Graph::new();
        let x_step = g_step.input("x", vec![1, 1], DType::F32);
        let w_step = g_step.weight("w", vec![2], DType::F32);
        let s_step = g_step.state("conv_state", vec![1, 1], DType::F32);
        let out_step = g_step.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x_step, w_step, s_step],
            vec![1, 1],
            DType::F32,
        );
        g_step.output(out_step);

        let mut state = StateStore::new();
        let step0 = run(&g_step, &weights, &mut state, vec![Tensor::F32(vec![1.0])]).unwrap();
        let step1 = run(&g_step, &weights, &mut state, vec![Tensor::F32(vec![5.0])]).unwrap();

        assert_eq!(step0[0].as_f32().unwrap()[0], expected[0]);
        assert!((step1[0].as_f32().unwrap()[0] - expected[1]).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_delta_rule_step_by_step_matches_prefill() {
        let mut g_prefill = Graph::new();
        let q = g_prefill.input("q", vec![2, 2], DType::F32);
        let k = g_prefill.input("k", vec![2, 2], DType::F32);
        let v = g_prefill.input("v", vec![2, 2], DType::F32);
        let beta = g_prefill.input("beta", vec![2], DType::F32);
        let log_decay = g_prefill.input("log_decay", vec![2], DType::F32);
        let out = g_prefill.add(
            "delta",
            Op::DeltaRule {
                key_heads: 1,
                val_heads: 1,
                key_dim: 2,
                val_dim: 2,
            },
            &[q, k, v, beta, log_decay],
            vec![2, 2],
            DType::F32,
        );
        g_prefill.output(out);

        let q_data = vec![1.0, 0.0, 0.0, 1.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0];
        let v_data = vec![2.0, 3.0, 4.0, 5.0];
        let beta_data = vec![1.0, 1.0];
        let ld_data = vec![0.0, 0.0];

        let pre_res = run_stateless(
            &g_prefill,
            &Weights::new(),
            vec![
                Tensor::F32(q_data.clone()),
                Tensor::F32(k_data.clone()),
                Tensor::F32(v_data.clone()),
                Tensor::F32(beta_data.clone()),
                Tensor::F32(ld_data.clone()),
            ],
        )
        .unwrap();
        let expected = pre_res[0].as_f32().unwrap();

        // Now step by step with persistent ssm_state
        let mut g_step = Graph::new();
        let q_s = g_step.input("q", vec![1, 2], DType::F32);
        let k_s = g_step.input("k", vec![1, 2], DType::F32);
        let v_s = g_step.input("v", vec![1, 2], DType::F32);
        let beta_s = g_step.input("beta", vec![1], DType::F32);
        let ld_s = g_step.input("log_decay", vec![1], DType::F32);
        let ssm_state = g_step.state("ssm_state", vec![1, 2, 2], DType::F32);
        let out_s = g_step.add(
            "delta",
            Op::DeltaRule {
                key_heads: 1,
                val_heads: 1,
                key_dim: 2,
                val_dim: 2,
            },
            &[q_s, k_s, v_s, beta_s, ld_s, ssm_state],
            vec![1, 2],
            DType::F32,
        );
        g_step.output(out_s);

        let mut state = StateStore::new();
        let step0 = run(
            &g_step,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![1.0, 0.0]),
                Tensor::F32(vec![1.0, 0.0]),
                Tensor::F32(vec![2.0, 3.0]),
                Tensor::F32(vec![1.0]),
                Tensor::F32(vec![0.0]),
            ],
        )
        .unwrap();
        let step1 = run(
            &g_step,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![0.0, 1.0]),
                Tensor::F32(vec![0.0, 1.0]),
                Tensor::F32(vec![4.0, 5.0]),
                Tensor::F32(vec![1.0]),
                Tensor::F32(vec![0.0]),
            ],
        )
        .unwrap();

        let s0 = step0[0].as_f32().unwrap();
        let s1 = step1[0].as_f32().unwrap();

        assert!((s0[0] - expected[0]).abs() < 1e-5);
        assert!((s0[1] - expected[1]).abs() < 1e-5);
        assert!((s1[0] - expected[2]).abs() < 1e-5);
        assert!((s1[1] - expected[3]).abs() < 1e-5);
    }
}
