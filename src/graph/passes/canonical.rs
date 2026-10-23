use crate::graph::{BinaryKind, Graph, NodeId, Op, Tensor, UnaryKind, ValueId, ValueKind, Weights};
use std::collections::{HashMap, HashSet};
/// Canonicalization passes running before placement:
/// 1. Arithmetic identity elimination (e.g. Add 0, Mul 1)
/// 2. Common Subexpression Elimination (CSE)
/// 3. Dead Code Elimination (DCE)
pub fn run(g: &mut Graph) {
    eliminate_common_subexpressions(g);
    eliminate_dead_nodes(g);
}

/// Identifies identical operations on identical inputs and merges them.
pub fn eliminate_common_subexpressions(g: &mut Graph) {
    let mut seen: HashMap<String, ValueId> = HashMap::new();
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();

    for node in &mut g.nodes {
        // Remap inputs first
        for inp in &mut node.inputs {
            if let Some(&new_id) = remap.get(inp) {
                *inp = new_id;
            }
        }

        // Build a structural signature for deterministic CSE
        let key = format!("{:?}:{:?}", node.op, node.inputs);
        if let Some(&existing_out) = seen.get(&key) {
            if !node.outputs.is_empty() {
                remap.insert(node.outputs[0], existing_out);
            }
        } else if !node.outputs.is_empty() {
            seen.insert(key, node.outputs[0]);
        }
    }

    // Remap graph outputs if needed
    for out in &mut g.outputs {
        if let Some(&new_id) = remap.get(out) {
            *out = new_id;
        }
    }
}

/// Retains only nodes that transitively contribute to graph outputs, or write persistent state
/// (a `write: true` `CachedAttention` appends to its K/V state even when nothing reads its output:
/// a biased self-attention can take the state write from it and the scores from `Attention`; a
/// `StateWrite` exists for its write).
pub fn eliminate_dead_nodes(g: &mut Graph) {
    let mut live_values: HashSet<ValueId> = g.outputs.iter().copied().collect();
    for node in &g.nodes {
        let writes_state = match node.op {
            Op::CachedAttention { write: true, .. } => node
                .inputs
                .get(3)
                .is_some_and(|&s| matches!(g.value(s).kind, ValueKind::State { .. })),
            Op::StateWrite => true,
            _ => false,
        };
        if writes_state {
            live_values.extend(node.outputs.iter().copied());
        }
    }

    // Work backwards from outputs to inputs
    for node in g.nodes.iter().rev() {
        let produces_live = node.outputs.iter().any(|out| live_values.contains(out));
        if produces_live {
            for &inp in &node.inputs {
                live_values.insert(inp);
            }
        }
    }

    // Keep only live nodes
    g.nodes
        .retain(|node| node.outputs.iter().any(|out| live_values.contains(out)));
    // Outputs record their producer's index: renumber after nodes were dropped.
    for (i, node) in g.nodes.iter().enumerate() {
        for &out in &node.outputs {
            g.values[out.0 as usize].kind = ValueKind::Node(NodeId(i as u32));
        }
    }
}

/// Folds post-`Linear`, post-`Embedding`, and post-`RmsNorm` scalar multiplies (`Unary::Scale(s)`
/// or `Binary::Mul` with a `[1]` F32 constant) directly into exclusively consumed F32 weights/gains.
pub fn fold_scales_and_gains(g: &mut Graph, weights: &mut Weights) {
    let mut val_uses: HashMap<ValueId, usize> = HashMap::new();
    for &out in &g.outputs {
        *val_uses.entry(out).or_insert(0) += 1;
    }
    for node in &g.nodes {
        for &inp in &node.inputs {
            *val_uses.entry(inp).or_insert(0) += 1;
        }
    }

    let scalar_of = |v: ValueId, g: &Graph, weights: &Weights| -> Option<f32> {
        if g.value(v).kind != ValueKind::Weight {
            return None;
        }
        match weights.get(&g.value(v).name)? {
            Tensor::F32(data) if data.len() == 1 => Some(data[0]),
            _ => None,
        }
    };

    let mut producer_of: HashMap<ValueId, usize> = HashMap::new();
    for (idx, node) in g.nodes.iter().enumerate() {
        for &out in &node.outputs {
            producer_of.insert(out, idx);
        }
    }

    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    for idx in 0..g.nodes.len() {
        let (scale_val, src_val, out_val) = match &g.nodes[idx].op {
            Op::Unary {
                kind: UnaryKind::Scale(s),
            } if g.nodes[idx].inputs.len() == 1 && !g.nodes[idx].outputs.is_empty() => {
                (*s, g.nodes[idx].inputs[0], g.nodes[idx].outputs[0])
            }
            Op::Binary {
                kind: BinaryKind::Mul,
            } if g.nodes[idx].inputs.len() == 2 && !g.nodes[idx].outputs.is_empty() => {
                let (a, b) = (g.nodes[idx].inputs[0], g.nodes[idx].inputs[1]);
                if let Some(s) = scalar_of(b, g, weights) {
                    (s, a, g.nodes[idx].outputs[0])
                } else if let Some(s) = scalar_of(a, g, weights) {
                    (s, b, g.nodes[idx].outputs[0])
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        if val_uses.get(&src_val).copied().unwrap_or(0) != 1 {
            continue;
        }
        let Some(&prod_idx) = producer_of.get(&src_val) else {
            continue;
        };
        let prod = &g.nodes[prod_idx];
        match prod.op {
            Op::RmsNorm { .. } if prod.inputs.len() == 2 => {
                let gamma_id = prod.inputs[1];
                if g.value(gamma_id).kind != ValueKind::Weight
                    || val_uses.get(&gamma_id).copied().unwrap_or(0) != 1
                {
                    continue;
                }
                let g_name = g.value(gamma_id).name.clone();
                let Some(Tensor::F32(mut data)) = weights.get(&g_name).cloned() else {
                    continue;
                };
                for x in &mut data {
                    *x *= scale_val;
                }
                weights.insert(g_name, Tensor::F32(data));
                remap.insert(out_val, src_val);
            }
            Op::Linear | Op::Embedding if prod.inputs.len() >= 2 => {
                let w_id = if matches!(prod.op, Op::Embedding) {
                    prod.inputs[0]
                } else {
                    prod.inputs[1]
                };
                if g.value(w_id).kind != ValueKind::Weight
                    || val_uses.get(&w_id).copied().unwrap_or(0) != 1
                {
                    continue;
                }
                let bias_id = if matches!(prod.op, Op::Linear) && prod.inputs.len() == 3 {
                    let bid = prod.inputs[2];
                    if g.value(bid).kind != ValueKind::Weight
                        || val_uses.get(&bid).copied().unwrap_or(0) != 1
                    {
                        continue;
                    }
                    Some(bid)
                } else {
                    None
                };
                let w_name = g.value(w_id).name.clone();
                let Some(Tensor::F32(mut w_data)) = weights.get(&w_name).cloned() else {
                    continue;
                };
                let b_entry = if let Some(bid) = bias_id {
                    let b_name = g.value(bid).name.clone();
                    let Some(Tensor::F32(b_data)) = weights.get(&b_name).cloned() else {
                        continue;
                    };
                    Some((b_name, b_data))
                } else {
                    None
                };
                for x in &mut w_data {
                    *x *= scale_val;
                }
                weights.insert(w_name, Tensor::F32(w_data));
                if let Some((b_name, mut b_data)) = b_entry {
                    for x in &mut b_data {
                        *x *= scale_val;
                    }
                    weights.insert(b_name, Tensor::F32(b_data));
                }
                remap.insert(out_val, src_val);
            }
            _ => {}
        }
    }

    if !remap.is_empty() {
        let resolve = |mut v: ValueId| -> ValueId {
            while let Some(&next) = remap.get(&v) {
                v = next;
            }
            v
        };
        for node in &mut g.nodes {
            for inp in &mut node.inputs {
                *inp = resolve(*inp);
            }
        }
        for out in &mut g.outputs {
            *out = resolve(*out);
        }
        eliminate_dead_nodes(g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Op};

    #[test]
    fn test_canonical_dce_and_cse() {
        let mut g = Graph::new();
        let a = g.input("a", vec![4], DType::F32);
        let b = g.input("b", vec![4], DType::F32);

        // Identical additions (CSE candidate)
        let add1 = g.add(
            "add1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, b],
            vec![4],
            DType::F32,
        );
        let add2 = g.add(
            "add2",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, b],
            vec![4],
            DType::F32,
        );

        // Dead node (never used by outputs)
        let _dead = g.add(
            "dead",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[a, b],
            vec![4],
            DType::F32,
        );

        // Output depends on add1 and add2
        let mul = g.add(
            "mul",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[add1, add2],
            vec![4],
            DType::F32,
        );
        g.output(mul);

        assert_eq!(g.nodes.len(), 4);
        run(&mut g);

        // After CSE and DCE:
        // add2 is merged into add1, dead node is eliminated.
        // We should have only add1 and mul (2 nodes).
        assert_eq!(g.nodes.len(), 2);
        // `mul` now sits at index 1 and its value says so (the SSA check reads this).
        crate::graph::pattern::validate_ssa(&g)
            .expect("producer indices follow the retained nodes");
    }

    #[test]
    fn test_fold_scales_and_gains_linear_and_rmsnorm() {
        use crate::graph::{cpu, StateStore, Weights};

        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let gamma = g.weight("gamma", vec![4], DType::F32);
        let w = g.weight("w", vec![3, 4], DType::F32);
        let b = g.weight("b", vec![3], DType::F32);

        let rn = g.add(
            "rn",
            Op::RmsNorm { eps: 1e-5 },
            &[x, gamma],
            vec![2, 4],
            DType::F32,
        );
        let rn_s = g.add(
            "rn_scale",
            Op::Unary {
                kind: UnaryKind::Scale(0.5),
            },
            &[rn],
            vec![2, 4],
            DType::F32,
        );
        let lin = g.add("lin", Op::Linear, &[rn_s, w, b], vec![2, 3], DType::F32);
        let lin_s = g.add(
            "lin_scale",
            Op::Unary {
                kind: UnaryKind::Scale(2.5),
            },
            &[lin],
            vec![2, 3],
            DType::F32,
        );
        g.output(lin_s);

        let mut weights = Weights::new();
        weights.insert("gamma", Tensor::F32(vec![1.0, 1.5, 0.8, 1.2]));
        weights.insert(
            "w",
            Tensor::F32((0..12).map(|i| (i as f32 - 5.0) * 0.2).collect()),
        );
        weights.insert("b", Tensor::F32(vec![0.1, -0.2, 0.3]));

        let inp = vec![Tensor::F32(vec![
            0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -0.75, 1.25,
        ])];
        let want = cpu::run(&g, &weights, &mut StateStore::new(), inp.clone()).unwrap()[0].to_f32();

        fold_scales_and_gains(&mut g, &mut weights);
        assert_eq!(
            g.nodes.len(),
            2,
            "both Scale nodes folded into gamma and w/b"
        );

        let got = cpu::run(&g, &weights, &mut StateStore::new(), inp).unwrap()[0].to_f32();
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }
}
