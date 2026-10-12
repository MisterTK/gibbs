//! No family computes on the CPU unless it is gated and the user opted in with `--backend cpu`.
//!
//! Runs in its own process because it arms the global CPU tripwire.

use gibbs_engine::graph::avenues::{explore_node, family_gate, family_status, required_ops, Family, Status};
use gibbs_engine::graph::program::GraphProgram;
use gibbs_engine::graph::{DType, Graph, Tensor, Weights};
use gibbs_engine::hardware::{available_devices, cpu_compute, Device, PlacementMode};
use gibbs_engine::model::graph_lm::{GraphLm, GraphLmConfig};
use std::sync::Once;

static INIT: Once = Once::new();

#[allow(dead_code)]
#[path = "../src/model/classifier/synth.rs"]
mod synth;

/// Loads a tiny synthetic classifier repo on Metal through the real loader and classifies a text and
/// a pair twice each; the tripwire is armed, so any host forward would panic.
fn classifier_on_gpu(name: &str, (cfg, ts): (serde_json::Value, synth::Ts)) {
    use gibbs_engine::model::classifier::SequenceClassifier;
    use gibbs_engine::model::hub::HuggingFaceHub;
    let hub = HuggingFaceHub::with_cache_dir(synth::write_repo(name, cfg, &ts));
    let c = SequenceClassifier::load(&hub, &format!("t/{name}"), Device::Metal)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(c.device(), Device::Metal, "{name}");
    for pair in [None, Some(synth::PAIR)] {
        for _ in 0..2 {
            let (scores, _) = c.classify(synth::TEXT, pair).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(scores.iter().all(|s| s.score.is_finite()), "{name}: non-finite scores");
            let sum: f32 = scores.iter().map(|s| s.score).sum();
            assert!((sum - 1.0).abs() < 1e-4, "{name}: softmax scores sum to {sum}");
        }
    }
}

fn metal_present() -> bool {
    available_devices().contains(&Device::Metal)
}

/// Compiles on `Auto`, asserts nothing runs on the CPU, then runs one prefill token and two decode steps.
fn run_on_gpu(name: &str, graph: Graph, weights: Weights) {
    let inputs: Vec<(String, DType, usize)> = graph
        .inputs
        .iter()
        .map(|&id| {
            let v = graph.value(id);
            (v.name.clone(), v.dtype, v.shape.iter().product::<usize>().max(1))
        })
        .collect();
    let mut prog = GraphProgram::compile(graph, weights, PlacementMode::Auto)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(prog.cpu_node_count(), 0, "{name}: nodes run on the CPU");
    assert_eq!(prog.primary_device(), Device::Metal, "{name}");
    for pos in 0..3u32 {
        let feed = inputs
            .iter()
            .map(|(n, dt, len)| match dt {
                DType::U32 if n.contains("pos") => Tensor::U32(vec![pos; *len]),
                DType::U32 => Tensor::U32(vec![1; *len]),
                _ => Tensor::F32(vec![0.01; *len]),
            })
            .collect();
        let out = prog.run(feed).unwrap_or_else(|e| panic!("{name} step {pos}: {e}"));
        let logits = out[0].as_f32().expect("f32 logits");
        assert!(logits.iter().all(|x| x.is_finite()), "{name} step {pos}: non-finite logits");
    }
}

fn gpu_fixture(f: Family) {
    match f {
        Family::GraphLm => {
            let (g, w) = GraphLm::new(GraphLmConfig::smollm_1m()).to_decode_graph();
            run_on_gpu("GraphLm", g, w);
        }
        Family::Gemma4 => {
            let (g, w) = gibbs_engine::model::gemma4::tiny_decode_graph().expect("tiny gemma4 graph");
            run_on_gpu("Gemma4", g, w);
        }
        Family::Qwen35 => {
            let (g, w) = gibbs_engine::model::qwen3_5::tiny_decode_graph().expect("tiny qwen graph");
            run_on_gpu("Qwen35", g, w);
        }
        Family::Qwen35Vision => {
            let (g, w) = gibbs_engine::model::qwen3_5::tiny_vision_tower_graph().expect("tiny vision tower graph");
            run_on_gpu("Qwen35Vision", g, w);
        }
        Family::DraftersMtp => {
            let (g, w) = gibbs_engine::model::qwen3_5::tiny_mtp_graph().expect("tiny MTP head graph");
            run_on_gpu("DraftersMtp", g, w);
        }
        Family::Ninfer => {
            let (g, w) = gibbs_engine::model::qwen3_5::tiny_ninfer_decode_graph().expect("tiny .ninfer decode graph");
            run_on_gpu("Ninfer", g, w);
        }
        Family::System1 => {
            use gibbs_engine::model::jev::{BertDims, BertEncoder};
            let dims = BertDims {
                hidden_size: 16,
                num_layers: 1,
                num_heads: 2,
                intermediate_size: 32,
                max_positions: 32,
                vocab_size: 64,
                type_vocab_size: 2,
                layer_norm_eps: 1e-5,
            };
            let enc = BertEncoder::load(dims, Device::Metal, |_, shape| {
                Ok(vec![0.05f32; shape.iter().product()])
            })
            .expect("System 1 loads on Metal");
            assert_eq!(enc.device(), Device::Metal);
            for _ in 0..3 {
                let out = enc.run(&[1, 10, 25, 4], &[0, 1, 2, 3], &[0, 0, 0, 0], true).expect("System 1 runs");
                assert_eq!(out.len(), 16);
            }
        }
        Family::T5 => gibbs_engine::model::t5::run_tiny_gpu_fixture(),
        Family::Seq2Seq => gibbs_engine::model::seq2seq::run_tiny_gpu_fixture(),
        Family::ClassifierBert => classifier_on_gpu("bert", synth::bert(None)),
        Family::ClassifierModern => {
            classifier_on_gpu("modernbert", synth::modernbert("mean", true));
            classifier_on_gpu("electra", synth::electra());
            classifier_on_gpu("albert", synth::albert());
            classifier_on_gpu("deberta-v2", synth::deberta_v2());
            classifier_on_gpu("deberta-v1", synth::deberta_v1());
        }
        Family::ClassifierDecoder => {
            gibbs_engine::model::classifier::gpu_fixture().expect("decoder seq-cls fixture")
        }
        other => panic!("{other:?} is marked Gpu but has no fixture in this test"),
    }
}

#[test]
fn cpu_free_matrix() {
    INIT.call_once(|| cpu_compute::ban(true));
    let mut rows = Vec::new();
    for f in Family::ALL {
        let gaps: Vec<String> = required_ops(f)
            .iter()
            .filter_map(|(op, dt)| {
                explore_node("required", op, *dt)
                    .lowered_by()
                    .is_none()
                    .then(|| format!("{} {dt:?}", op.name()))
            })
            .collect();
        let status = family_status(f);
        // (a) a family is `Gpu` exactly when none of its required ops lacks a lowering; a gate whose
        // reason starts with `numerics:` (parity unresolved) or `builder:` (every op lowers but the
        // Metal program is not written yet) may hold a family back although every op lowers.
        let numerics_gate = matches!(status, Status::Gated(r) if r.starts_with("numerics:"));
        let builder_gate = matches!(status, Status::Gated(r) if r.starts_with("builder:"));
        assert!(
            gaps.is_empty() == matches!(status, Status::Gpu)
                || (gaps.is_empty() && (numerics_gate || builder_gate)),
            "{f:?}: status {status:?} disagrees with unlowered ops {gaps:?}"
        );
        match status {
            Status::Gpu => {
                assert!(family_gate(f, Device::Metal).is_ok(), "{f:?}");
                if metal_present() {
                    gpu_fixture(f); // (b)
                }
            }
            Status::Gated(reason) => {
                if numerics_gate && metal_present() {
                    gpu_fixture(f); // the tiny fixture lowers and runs; real checkpoints are what diverge
                }
                // (c)
                let err = family_gate(f, Device::Metal).expect_err("gated family must refuse Metal");
                assert!(err.starts_with("gpu-only:") && err.contains(reason), "{err}");
                assert!(family_gate(f, Device::Cpu).is_ok(), "{f:?}: --backend cpu must stay open");
            }
        }
        let status_cell = match status {
            Status::Gpu => "GPU".to_string(),
            Status::Gated(_) => "CPU only: --backend cpu".to_string(),
        };
        let mut uniq = gaps.clone();
        uniq.sort();
        uniq.dedup();
        rows.push(format!(
            "| {f:?} | {status_cell} | {} |",
            if uniq.is_empty() { "—".to_string() } else { uniq.join(", ") }
        ));
    }
    // (d) generated matrix
    println!("\n| family | status | unlowered ops |\n|:---|:---|:---|");
    for r in rows {
        println!("{r}");
    }
}
