//! Declarative Subgraph Pattern Combinators for Graph IR.
//!
//! Provides a composable, backtracking subgraph isomorphism matcher and
//! rewrite engine for structural graph fusions.

use crate::graph::{Graph, Node, NodeId, ValueId, ValueKind};
use std::collections::{HashMap, HashSet};

/// Match context storing captured nodes during pattern matching.
#[derive(Debug, Clone, Default)]
pub struct MatchContext {
    pub captures: HashMap<String, NodeId>,
}

impl MatchContext {
    pub fn get(&self, name: &str) -> Option<NodeId> {
        self.captures.get(name).copied()
    }

    pub fn insert(&mut self, name: impl Into<String>, id: NodeId) {
        self.captures.insert(name.into(), id);
    }
}

/// Pattern element representing a node matcher.
pub enum PatternNode {
    /// Matches by operation name or type predicate.
    OpType(String),
    /// Matches node with custom predicate.
    Predicate(Box<dyn Fn(&Node, &Graph) -> bool + Send + Sync>),
    /// Matches any node.
    Any,
    /// Captures matched node under a name.
    Capture(String, Box<PatternNode>),
}

/// Structural constraint between pattern nodes.
pub enum PatternEdge {
    /// Producer output flows to consumer input: `(producer_idx, consumer_idx)`.
    DataFlow(usize, usize),
    /// Producer flows to multiple consumers: `(root_idx, vec![consumer_idx, ...])`.
    Branch(usize, Vec<usize>),
    /// Consumers share the same primary input: `(consumer_a, consumer_b)`.
    SharedInput(usize, usize),
}

/// Composable pattern specification over subgraphs.
pub struct Pattern {
    pub nodes: Vec<PatternNode>,
    pub edges: Vec<PatternEdge>,
}

impl Pattern {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Appends a node matcher to the pattern and returns its index.
    pub fn add_node(&mut self, node: PatternNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Adds a dataflow edge: node `from` produces a value read by node `to`.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push(PatternEdge::DataFlow(from, to));
    }

    /// Adds a branch constraint: node `root` produces a value read by all `consumers`.
    pub fn add_branch(&mut self, root: usize, consumers: &[usize]) {
        self.edges
            .push(PatternEdge::Branch(root, consumers.to_vec()));
    }
    /// Adds a constraint that both consumers read the same primary input operand.
    pub fn add_shared_input(&mut self, a: usize, b: usize) {
        self.edges.push(PatternEdge::SharedInput(a, b));
    }

    /// Searches `graph` for all matching subgraphs via backtracking isomorphism.
    pub fn search(&self, graph: &Graph) -> Vec<MatchContext> {
        let mut results = Vec::new();
        let mut assignment: Vec<Option<NodeId>> = vec![None; self.nodes.len()];
        let mut used = HashSet::new();

        self.backtrack(graph, 0, &mut assignment, &mut used, &mut results);

        // Deduplicate symmetric matches that cover identical sets of nodes
        let mut unique_results = Vec::new();
        let mut seen_node_sets: HashSet<Vec<NodeId>> = HashSet::new();
        for ctx in results {
            let mut node_set: Vec<NodeId> = ctx.captures.values().copied().collect();
            node_set.sort_by_key(|n| n.0);
            if seen_node_sets.insert(node_set) {
                unique_results.push(ctx);
            }
        }
        unique_results
    }

    fn backtrack(
        &self,
        graph: &Graph,
        pat_idx: usize,
        assignment: &mut [Option<NodeId>],
        used: &mut HashSet<NodeId>,
        results: &mut Vec<MatchContext>,
    ) {
        if pat_idx == self.nodes.len() {
            // All nodes assigned, verify edges
            if self.check_edges(graph, assignment) {
                let mut ctx = MatchContext::default();
                for (idx, p_node) in self.nodes.iter().enumerate() {
                    if let Some(nid) = assignment[idx] {
                        extract_captures(p_node, nid, &mut ctx);
                    }
                }
                results.push(ctx);
            }
            return;
        }

        let p_node = &self.nodes[pat_idx];
        for (i, node) in graph.nodes.iter().enumerate() {
            let nid = NodeId(i as u32);
            if used.contains(&nid) {
                continue;
            }
            if matches_node(p_node, node, graph) {
                assignment[pat_idx] = Some(nid);
                used.insert(nid);

                // Early edge constraint pruning
                if self.check_partial_edges(graph, assignment, pat_idx) {
                    self.backtrack(graph, pat_idx + 1, assignment, used, results);
                }

                assignment[pat_idx] = None;
                used.remove(&nid);
            }
        }
    }

    fn check_edges(&self, graph: &Graph, assignment: &[Option<NodeId>]) -> bool {
        for edge in &self.edges {
            match edge {
                PatternEdge::DataFlow(from, to) => {
                    let from_nid = assignment[*from].expect("assigned");
                    let to_nid = assignment[*to].expect("assigned");
                    if !node_flows_to(graph, from_nid, to_nid) {
                        return false;
                    }
                }
                PatternEdge::Branch(root, consumers) => {
                    let root_nid = assignment[*root].expect("assigned");
                    for &c_idx in consumers {
                        let c_nid = assignment[c_idx].expect("assigned");
                        if !node_flows_to(graph, root_nid, c_nid) {
                            return false;
                        }
                    }
                }
                PatternEdge::SharedInput(a, b) => {
                    let a_nid = assignment[*a].expect("assigned");
                    let b_nid = assignment[*b].expect("assigned");
                    let a_in = graph.node(a_nid).inputs.first();
                    let b_in = graph.node(b_nid).inputs.first();
                    if a_in != b_in || a_in.is_none() {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn check_partial_edges(
        &self,
        graph: &Graph,
        assignment: &[Option<NodeId>],
        current_idx: usize,
    ) -> bool {
        for edge in &self.edges {
            match edge {
                PatternEdge::DataFlow(from, to) => {
                    if *from <= current_idx && *to <= current_idx {
                        let f = assignment[*from].unwrap();
                        let t = assignment[*to].unwrap();
                        if !node_flows_to(graph, f, t) {
                            return false;
                        }
                    }
                }
                PatternEdge::Branch(root, consumers) => {
                    if *root <= current_idx {
                        let r = assignment[*root].unwrap();
                        for &c in consumers {
                            if c <= current_idx {
                                let c_nid = assignment[c].unwrap();
                                if !node_flows_to(graph, r, c_nid) {
                                    return false;
                                }
                            }
                        }
                    }
                }
                PatternEdge::SharedInput(a, b) => {
                    if *a <= current_idx && *b <= current_idx {
                        let a_nid = assignment[*a].unwrap();
                        let b_nid = assignment[*b].unwrap();
                        let a_in = graph.node(a_nid).inputs.first();
                        let b_in = graph.node(b_nid).inputs.first();
                        if a_in != b_in || a_in.is_none() {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_node(pat: &PatternNode, node: &Node, graph: &Graph) -> bool {
    match pat {
        PatternNode::OpType(op_name) => node.op.name() == op_name,
        PatternNode::Predicate(pred) => pred(node, graph),
        PatternNode::Any => true,
        PatternNode::Capture(_, inner) => matches_node(inner, node, graph),
    }
}

fn extract_captures(pat: &PatternNode, nid: NodeId, ctx: &mut MatchContext) {
    if let PatternNode::Capture(name, inner) = pat {
        ctx.insert(name.clone(), nid);
        extract_captures(inner, nid, ctx);
    }
}

fn node_flows_to(graph: &Graph, from: NodeId, to: NodeId) -> bool {
    let from_node = graph.node(from);
    let to_node = graph.node(to);
    for out in &from_node.outputs {
        if to_node.inputs.contains(out) {
            return true;
        }
    }
    false
}

// Combinator builder helpers

/// Matches node with given operation name.
pub fn op_type(name: &str) -> PatternNode {
    PatternNode::OpType(name.to_string())
}

/// Matches node satisfying custom predicate.
pub fn pred<F: Fn(&Node, &Graph) -> bool + Send + Sync + 'static>(f: F) -> PatternNode {
    PatternNode::Predicate(Box::new(f))
}

/// Matches any node.
pub fn any() -> PatternNode {
    PatternNode::Any
}

/// Captures matched node under `name`.
pub fn capture(name: &str, node: PatternNode) -> PatternNode {
    PatternNode::Capture(name.to_string(), Box::new(node))
}

/// Graph rewriting engine.
pub struct GraphRewriter;

impl GraphRewriter {
    /// Replaces matched subgraphs with a new node generated by `builder_fn`.
    /// Reroutes consumers of replaced nodes to the replacement node's outputs,
    /// and removes eliminated nodes in topological order.
    pub fn replace<F>(graph: &mut Graph, matches: &[MatchContext], mut builder_fn: F) -> usize
    where
        F: FnMut(&mut Graph, &MatchContext) -> Option<(Vec<NodeId>, Vec<ValueId>)>,
    {
        let mut total_rewrites = 0;
        let mut all_eliminated: HashSet<NodeId> = HashSet::new();

        for m in matches {
            // Skip match if any of its nodes are already eliminated
            if m.captures.values().any(|nid| all_eliminated.contains(nid)) {
                continue;
            }

            if let Some((elim_nodes, new_outputs)) = builder_fn(graph, m) {
                // Determine original output values from eliminated nodes
                let mut old_outputs = Vec::new();
                for &nid in &elim_nodes {
                    for &out_val in &graph.node(nid).outputs {
                        old_outputs.push(out_val);
                    }
                    all_eliminated.insert(nid);
                }

                // Reroute consumers of old_outputs to corresponding new_outputs
                for (old_val, new_val) in old_outputs.into_iter().zip(new_outputs.into_iter()) {
                    for node in &mut graph.nodes {
                        for input in &mut node.inputs {
                            if *input == old_val {
                                *input = new_val;
                            }
                        }
                    }
                    // Reroute graph outputs if affected
                    for out in &mut graph.outputs {
                        if *out == old_val {
                            *out = new_val;
                        }
                    }
                }
                total_rewrites += 1;
            }
        }

        if total_rewrites > 0 {
            // Prune eliminated nodes and reconstruct graph values in topological order
            compact_graph(graph, &all_eliminated);
        }

        total_rewrites
    }
}

/// Validates SSA form: every node input must refer to an input, weight, state,
/// or a node strictly preceding it in topological order.
pub fn validate_ssa(graph: &Graph) -> Result<(), String> {
    for (i, node) in graph.nodes.iter().enumerate() {
        for &inp in &node.inputs {
            if (inp.0 as usize) >= graph.values.len() {
                return Err(format!(
                    "Node {i} ({}) references invalid ValueId {}",
                    node.name, inp.0
                ));
            }
            let val = graph.value(inp);
            match &val.kind {
                ValueKind::Input | ValueKind::Weight | ValueKind::State(_) => {}
                ValueKind::Node(prod_nid) => {
                    if prod_nid.0 as usize >= i {
                        return Err(format!(
                            "SSA violation: Node {i} ({}) references value {} from node {} ({}) which does not precede it",
                            node.name, inp.0, prod_nid.0, graph.node(*prod_nid).name
                        ));
                    }
                }
            }
        }
    }
    for &out in &graph.outputs {
        if (out.0 as usize) >= graph.values.len() {
            return Err(format!("Graph output ValueId {} out of range", out.0));
        }
    }
    Ok(())
}

/// Eliminates removed nodes and re-indexes the Graph to preserve strict topological order and SSA.
fn compact_graph(graph: &mut Graph, eliminated: &HashSet<NodeId>) {
    // Filter out eliminated nodes
    let remaining: Vec<Node> = graph
        .nodes
        .drain(..)
        .enumerate()
        .filter(|(idx, _)| !eliminated.contains(&NodeId(*idx as u32)))
        .map(|(_, node)| node)
        .collect();

    let n = remaining.len();
    // Map each output ValueId to the index of the node producing it
    let mut val_to_node_idx: HashMap<ValueId, usize> = HashMap::new();
    for (idx, node) in remaining.iter().enumerate() {
        for &out in &node.outputs {
            val_to_node_idx.insert(out, idx);
        }
    }

    // In-degree and adjacency list
    let mut in_degree = vec![0usize; n];
    let mut adj = vec![Vec::new(); n];
    for (consumer_idx, node) in remaining.iter().enumerate() {
        for inp in &node.inputs {
            if let Some(&producer_idx) = val_to_node_idx.get(inp) {
                if producer_idx != consumer_idx {
                    adj[producer_idx].push(consumer_idx);
                    in_degree[consumer_idx] += 1;
                }
            }
        }
    }

    // Kahn's algorithm
    let mut queue = std::collections::VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut sorted_indices = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        sorted_indices.push(u);
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    let ordered_nodes = if sorted_indices.len() == n {
        let rem = remaining;
        let mut ordered = Vec::with_capacity(n);
        for idx in sorted_indices {
            ordered.push(rem[idx].clone());
        }
        ordered
    } else {
        remaining
    };

    // Assign new NodeIds and update ValueKind::Node
    for (new_idx, node) in ordered_nodes.iter().enumerate() {
        let new_nid = NodeId(new_idx as u32);
        for &out in &node.outputs {
            if let Some(val) = graph.values.get_mut(out.0 as usize) {
                val.kind = ValueKind::Node(new_nid);
            }
        }
    }

    graph.nodes = ordered_nodes;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Op};

    #[test]
    fn test_graph_pattern_combinators_branch() {
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
        let w_k = g.weight("w_k", vec![64, 64], DType::F32);
        let w_v = g.weight("w_v", vec![64, 64], DType::F32);
        let _q = g.add("q", Op::Linear, &[norm, w_q], vec![1, 64], DType::F32);
        let _k = g.add("k", Op::Linear, &[norm, w_k], vec![1, 64], DType::F32);
        let _v = g.add("v", Op::Linear, &[norm, w_v], vec![1, 64], DType::F32);

        let mut pat = Pattern::new();
        let r = pat.add_node(capture("norm", op_type("RmsNorm")));
        let q_node = pat.add_node(capture("q", op_type("Linear")));
        let k_node = pat.add_node(capture("k", op_type("Linear")));
        let v_node = pat.add_node(capture("v", op_type("Linear")));
        pat.add_branch(r, &[q_node, k_node, v_node]);

        let matches = pat.search(&g);
        assert_eq!(
            matches.len(),
            1,
            "Pattern should find exactly one QKV branch"
        );
        let m = &matches[0];
        assert_eq!(m.get("norm"), Some(NodeId(0)));
        assert!(m.get("q").is_some());
        assert!(m.get("k").is_some());
        assert!(m.get("v").is_some());
    }

    #[test]
    fn test_graph_pattern_combinators_replace() {
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
        let w_k = g.weight("w_k", vec![64, 64], DType::F32);
        let w_v = g.weight("w_v", vec![64, 64], DType::F32);
        let q = g.add("q", Op::Linear, &[norm, w_q], vec![1, 64], DType::F32);
        let k = g.add("k", Op::Linear, &[norm, w_k], vec![1, 64], DType::F32);
        let v = g.add("v", Op::Linear, &[norm, w_v], vec![1, 64], DType::F32);

        // Attention consuming q, k, v
        let attn = g.add(
            "attn",
            Op::Attention {
                heads: 2,
                kv_heads: 2,
                causal: true,
                window: None,
                soft_cap: None,
            },
            &[q, k, v],
            vec![1, 64],
            DType::F32,
        );
        let final_out = g.add("final_add", Op::Add, &[attn, in_x], vec![1, 64], DType::F32);
        g.output(final_out);

        // Validate initial SSA
        assert!(validate_ssa(&g).is_ok());

        // Pattern to find Q, K, V
        let mut pat = Pattern::new();
        let r = pat.add_node(capture("norm", op_type("RmsNorm")));
        let q_node = pat.add_node(capture("q", op_type("Linear")));
        let k_node = pat.add_node(capture("k", op_type("Linear")));
        let v_node = pat.add_node(capture("v", op_type("Linear")));
        pat.add_branch(r, &[q_node, k_node, v_node]);

        let matches = pat.search(&g);
        assert_eq!(matches.len(), 1);

        // Replace Q, K, V with a single fused QKV projection
        let rewrites = GraphRewriter::replace(&mut g, &matches, |graph, m| {
            let norm_id = m.get("norm")?;
            let q_id = m.get("q")?;
            let k_id = m.get("k")?;
            let v_id = m.get("v")?;

            let norm_out = graph.node(norm_id).outputs[0];
            let w_qkv = graph.weight("w_fused_qkv", vec![192, 64], DType::F32);
            // Add fused node
            let fused_out = graph.add(
                "fused_qkv",
                Op::Linear,
                &[norm_out, w_qkv],
                vec![1, 192],
                DType::F32,
            );

            // In real lowering, slices would follow. For testing SSA rerouting:
            // We route all 3 consumers to fused_out (simulating replacement)
            Some((
                vec![q_id, k_id, v_id],
                vec![fused_out, fused_out, fused_out],
            ))
        });

        assert_eq!(rewrites, 1);
        // Validate SSA form after rewriting
        assert!(
            validate_ssa(&g).is_ok(),
            "Rewritten graph must have valid SSA: {:?}",
            validate_ssa(&g)
        );
        // Assert graph output still resolves
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.outputs[0], final_out);

        // Verify the eliminated nodes are gone and fused_qkv precedes attn
        let node_names: Vec<&str> = g.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(!node_names.contains(&"q"));
        assert!(!node_names.contains(&"k"));
        assert!(!node_names.contains(&"v"));
        assert!(node_names.contains(&"fused_qkv"));
        let fused_idx = node_names.iter().position(|&n| n == "fused_qkv").unwrap();
        let attn_idx = node_names.iter().position(|&n| n == "attn").unwrap();
        assert!(
            fused_idx < attn_idx,
            "fused_qkv must precede attn in topological order"
        );
    }
}
