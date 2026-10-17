use crate::graph::{Graph, NodeId, Op, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

/// Canonicalization passes running before placement:
/// 1. Arithmetic identity elimination (e.g. Add 0, Mul 1)
/// 2. Common Subexpression Elimination (CSE)
/// 3. Dead Code Elimination (DCE)
pub fn run(g: &mut Graph) {
    fold_weightless_rmsnorm_mul(g);
    eliminate_common_subexpressions(g);
    eliminate_dead_nodes(g);
}

/// Folds `unit = RmsNorm(x)` (1 input, no gamma) followed by `Mul(unit, scale)` (where `scale` has
/// shape `[d]` matching the last dimension of `x` and `unit` has no other readers) into a single
/// 2-input `RmsNorm(x, scale)`.
fn fold_weightless_rmsnorm_mul(g: &mut Graph) {
    let mut use_count: HashMap<ValueId, usize> = HashMap::new();
    for n in &g.nodes {
        for &inp in &n.inputs {
            *use_count.entry(inp).or_insert(0) += 1;
        }
    }
    for &out in &g.outputs {
        *use_count.entry(out).or_insert(0) += 1;
    }
    let mut unit_norms: HashMap<ValueId, (ValueId, f32, usize)> = HashMap::new();
    for n in &g.nodes {
        if let Op::RmsNorm { eps } = n.op {
            if n.inputs.len() == 1 && n.outputs.len() == 1 {
                let out = n.outputs[0];
                if use_count.get(&out).copied() == Some(1) {
                    if let Some(&d) = g.value(out).shape.last() {
                        if d > 0 {
                            unit_norms.insert(out, (n.inputs[0], eps, d));
                        }
                    }
                }
            }
        }
    }
    if unit_norms.is_empty() {
        return;
    }
    for i in 0..g.nodes.len() {
        if g.nodes[i].op != Op::Mul || g.nodes[i].inputs.len() != 2 {
            continue;
        }
        let (a, b) = (g.nodes[i].inputs[0], g.nodes[i].inputs[1]);
        for (u, s) in [(a, b), (b, a)] {
            if let Some(&(x, eps, d)) = unit_norms.get(&u) {
                if g.value(s).shape.as_slice() == [d]
                    && g.value(s).dtype == crate::graph::DType::F32
                {
                    g.nodes[i].op = Op::RmsNorm { eps };
                    g.nodes[i].inputs = vec![x, s];
                    break;
                }
            }
        }
    }
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
/// T5's biased self-attention takes the state write from it and the scores from `Attention`).
pub fn eliminate_dead_nodes(g: &mut Graph) {
    let mut live_values: HashSet<ValueId> = g.outputs.iter().copied().collect();
    for node in &g.nodes {
        let writes_state = matches!(node.op, Op::CachedAttention { write: true, .. })
            && node
                .inputs
                .get(3)
                .is_some_and(|&s| matches!(g.value(s).kind, ValueKind::State(_)));
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
        let add1 = g.add("add1", Op::Add, &[a, b], vec![4], DType::F32);
        let add2 = g.add("add2", Op::Add, &[a, b], vec![4], DType::F32);

        // Dead node (never used by outputs)
        let _dead = g.add("dead", Op::Mul, &[a, b], vec![4], DType::F32);

        // Output depends on add1 and add2
        let mul = g.add("mul", Op::Mul, &[add1, add2], vec![4], DType::F32);
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
}
