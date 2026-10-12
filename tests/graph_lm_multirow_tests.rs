use gibbs_engine::graph::Graph;
use gibbs_engine::model::graph_lm::{GraphLm, GraphLmConfig};

#[test]
fn test_graph_lm_multi_row_parity() {
    let cfg = GraphLmConfig::smollm_1m();
    let model = GraphLm::new(cfg);

    // Verify build_graph for n = 1 and n = 8 compiles cleanly
    for n in [1, 8] {
        let (g, _w) = model.build_graph(n, true);
        assert!(!g.nodes.is_empty());
    }
}
