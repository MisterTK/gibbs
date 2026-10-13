//! Strict device placement and the single-segment partition for the graph IR.
//!
//! On a host with Metal, `Auto` and `Pin(Metal)` put every node on Metal or fail with the
//! [`GpuReport`] of avenues each unlowered node was tried against. The CPU executor runs only for an
//! explicit `Pin(Cpu)` (or when no GPU exists).

use crate::graph::avenues::{explore_node, GpuReport};
use crate::graph::{DType, Graph, NodeId, Op};
use crate::hardware::{Device, DeviceProfile, PlacementMode};
use std::collections::HashMap;

/// A contiguous sequence of graph nodes assigned to run on a single compute device.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub device: Device,
    pub nodes: Vec<NodeId>,
}

/// Placement telemetry: how many nodes run on each device.
#[derive(Debug, Clone, Default)]
pub struct PlacementReport {
    pub device_op_counts: HashMap<Device, usize>,
}

impl PlacementReport {
    pub fn summary(&self) -> String {
        self.device_op_counts
            .iter()
            .map(|(dev, count)| format!("{dev}: {count} ops"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn primary_device(&self) -> Device {
        self.device_op_counts
            .iter()
            .max_by_key(|a| a.1)
            .map(|(&dev, _)| dev)
            .unwrap_or(Device::Cpu)
    }
}

/// The execution plan: at most one segment, since every node shares one device.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    pub segments: Vec<Segment>,
    pub report: PlacementReport,
}

impl PlacementPlan {
    pub fn primary_device(&self) -> Device {
        self.report.primary_device()
    }
}

/// Returns whether `device` can execute `op` with element type `dtype`.
pub fn device_supports_op(device: Device, op: &Op, dtype: DType) -> bool {
    match device {
        Device::Cpu => crate::graph::cpu::supports_op(op, dtype),
        Device::Metal => {
            #[cfg(target_os = "macos")]
            {
                crate::graph::metal::supports_op(op, dtype)
            }
            #[cfg(not(target_os = "macos"))]
            false
        }
        Device::Cuda => crate::graph::cuda::supports_op(op, dtype),
        Device::Rocm => crate::graph::rocm::supports_op(op, dtype),
        Device::LevelZero => crate::graph::level_zero::supports_op(op, dtype),
        Device::Ane => false,
    }
}

/// The element type that decides whether a node's op is supported.
fn node_dtype(graph: &Graph, node: &crate::graph::Node) -> DType {
    let operand = |i: usize| node.inputs.get(i).map(|&v| graph.value(v).dtype);
    match &node.op {
        Op::Linear | Op::MoeLinear { .. } => operand(1),
        Op::TopK { .. } | Op::GatherCols => operand(0),
        Op::Embedding => operand(0),
        Op::Megakernel(_) => operand(2),
        _ => node.outputs.first().map(|&v| graph.value(v).dtype),
    }
    .unwrap_or(DType::F32)
}

/// Places every node of `graph` on one device.
///
/// `supports` answers whether a device runs `(op, dtype)`; production passes
/// [`device_supports_op`]. A node the target rejects still places when another avenue lowers it
/// (e.g. load-time dequantization); otherwise the call fails with one report per such node.
pub fn place_graph(
    graph: &mut Graph,
    mode: PlacementMode,
    profiles: &[(Device, DeviceProfile)],
    supports: &dyn Fn(Device, &Op, DType) -> bool,
) -> Result<PlacementPlan, GpuReport> {
    let available: Vec<Device> = profiles.iter().map(|(d, _)| *d).collect();
    let target = match mode {
        PlacementMode::Pin(d) if available.contains(&d) => d,
        PlacementMode::Auto if available.contains(&Device::Metal) => Device::Metal,
        _ => Device::Cpu,
    };

    if target != Device::Cpu {
        let mut report = GpuReport { total_nodes: graph.nodes.len(), unlowered: Vec::new() };
        for node in &graph.nodes {
            let dtype = node_dtype(graph, node);
            if supports(target, &node.op, dtype) {
                continue;
            }
            let explored = explore_node(&node.name, &node.op, dtype);
            if explored.lowered_by().is_none() {
                report.unlowered.push(explored);
            }
        }
        if !report.unlowered.is_empty() {
            return Err(report);
        }
    }

    for node in &mut graph.nodes {
        node.device = target;
    }
    let all: Vec<NodeId> = (0..graph.nodes.len()).map(|i| NodeId(i as u32)).collect();
    let mut device_op_counts = HashMap::new();
    let segments = if all.is_empty() {
        Vec::new()
    } else {
        device_op_counts.insert(target, all.len());
        vec![Segment { device: target, nodes: all }]
    };
    Ok(PlacementPlan { segments, report: PlacementReport { device_op_counts } })
}

/// Convenience helper constructing standard device profiles from available devices and measured bandwidth.
pub fn place_graph_default(
    graph: &mut Graph,
    mode: PlacementMode,
    available: &[Device],
    host_bandwidth_gbps: Option<f64>,
    supports: &dyn Fn(Device, &Op, DType) -> bool,
) -> Result<PlacementPlan, GpuReport> {
    let profiles: Vec<(Device, DeviceProfile)> = available
        .iter()
        .map(|&d| (d, DeviceProfile::for_device(d, host_bandwidth_gbps)))
        .collect();
    place_graph(graph, mode, &profiles, supports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph, Op};

    #[test]
    fn test_placement_pin_cpu() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let w = g.weight("w", vec![4, 4], DType::F32);
        let out = g.add("lin", Op::Linear, &[x, w], vec![2, 4], DType::F32);
        g.output(out);

        let plan = place_graph_default(
            &mut g,
            PlacementMode::Pin(Device::Cpu),
            &[Device::Cpu],
            None,
            &device_supports_op,
        )
        .unwrap();
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].device, Device::Cpu);
        assert_eq!(plan.report.device_op_counts[&Device::Cpu], 1);
    }

    fn relu_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let w = g.weight("w", vec![4, 4], DType::F32);
        let lin = g.add("lin", Op::Linear, &[x, w], vec![2, 4], DType::F32);
        // U32 output: Relu has no lowering at that dtype on any avenue.
        let out = g.add("act", Op::Relu, &[lin], vec![2, 4], DType::U32);
        g.output(out);
        g
    }

    /// Metal that supports everything except `Relu`.
    fn no_relu(d: Device, op: &Op, dt: DType) -> bool {
        d == Device::Metal && !matches!(op, Op::Relu) && device_supports_op(Device::Cpu, op, dt)
    }

    #[test]
    fn unlowered_node_fails_with_every_avenue() {
        let mut g = relu_graph();
        let err = place_graph_default(&mut g, PlacementMode::Auto, &[Device::Cpu, Device::Metal], None, &no_relu)
            .unwrap_err();
        let text = err.error();
        assert!(text.starts_with("gpu-only: 1 of 2 graph nodes"), "{text}");
        assert!(text.contains("act"), "{text}");
        for avenue in crate::graph::avenues::Avenue::ALL {
            assert!(text.contains(avenue.name()), "missing {} in {text}", avenue.name());
        }
    }

    #[test]
    fn pin_cpu_accepts_unlowered_node() {
        let mut g = relu_graph();
        let plan = place_graph_default(&mut g, PlacementMode::Pin(Device::Cpu), &[Device::Cpu, Device::Metal], None, &no_relu)
            .unwrap();
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].device, Device::Cpu);
    }

    #[test]
    fn supported_graph_is_all_metal() {
        let mut g = relu_graph();
        let plan = place_graph_default(&mut g, PlacementMode::Auto, &[Device::Cpu, Device::Metal], None, &|_, _, _| true)
            .unwrap();
        assert_eq!(plan.segments.len(), 1);
        assert!(g.nodes.iter().all(|n| n.device == Device::Metal));
    }

    #[test]
    fn moe_ops_place_on_metal_and_match_reference() {
        use crate::graph::{ReduceKind, Tensor};

        let (m, d_in, d_out, experts, top_k) = (4, 8, 8, 4, 2);
        let mut seed = 42u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let (x_data, router_data, bank_data) =
            (rnd(m * d_in), rnd(experts * d_in), rnd(experts * d_out * d_in));

        // Scalar reference: softmax over the top-k router logits weighs the picked experts' outputs.
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&a, &b)| a * b).sum() };
        let mut expected = vec![0.0f32; m * d_out];
        for r in 0..m {
            let xr = &x_data[r * d_in..(r + 1) * d_in];
            let mut logits: Vec<(usize, f32)> = (0..experts)
                .map(|e| (e, dot(xr, &router_data[e * d_in..(e + 1) * d_in])))
                .collect();
            logits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let top = &logits[..top_k];
            let exps: Vec<f32> = top.iter().map(|t| (t.1 - top[0].1).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (&(e, _), p) in top.iter().zip(&exps) {
                for o in 0..d_out {
                    let w = &bank_data[(e * d_out + o) * d_in..][..d_in];
                    expected[r * d_out + o] += p / sum * dot(xr, w);
                }
            }
        }

        let mut g = Graph::new();
        let f = DType::F32;
        let x = g.input("x", vec![m, d_in], f);
        let rw = g.weight("router", vec![experts, d_in], f);
        let bank = g.weight("bank", vec![experts * d_out, d_in], f);
        let logits = g.add("logits", Op::Linear, &[x, rw], vec![m, experts], f);
        let ids = g.add("ids", Op::TopK { k: top_k }, &[logits], vec![m, top_k], DType::U32);
        let picked = g.add("picked", Op::GatherCols, &[logits, ids], vec![m, top_k], f);
        let probs = g.add("probs", Op::Softmax { axis: -1 }, &[picked], vec![m, top_k], f);
        let y = g.add("y", Op::MoeLinear { experts, top_k }, &[x, bank, ids], vec![m * top_k, d_out], f);
        let p_col = g.add("p_col", Op::Reshape { shape: Some(vec![m * top_k, 1]) }, &[probs], vec![m * top_k, 1], f);
        let weighted = g.add("weighted", Op::Mul, &[y, p_col], vec![m * top_k, d_out], f);
        let grouped = g.add("grouped", Op::Reshape { shape: Some(vec![m, top_k, d_out]) }, &[weighted], vec![m, top_k, d_out], f);
        let out = g.add(
            "out",
            Op::Reduce { kind: ReduceKind::Sum, axes: vec![1], keep_dims: false },
            &[grouped],
            vec![m, d_out],
            f,
        );
        g.output(out);

        let mut weights = crate::graph::Weights::new();
        weights.insert("router", Tensor::F32(router_data));
        weights.insert("bank", Tensor::F32(bank_data));
        let inputs = vec![Tensor::F32(x_data)];

        let mut cpu_g = g.clone();
        place_graph_default(&mut cpu_g, PlacementMode::Pin(Device::Cpu), &[Device::Cpu], None, &device_supports_op)
            .unwrap();
        let mut state = crate::graph::StateStore::new();
        let cpu_out = crate::graph::cpu::run(&cpu_g, &weights, &mut state, inputs.clone()).unwrap();
        for (i, (got, want)) in cpu_out[0].as_f32().unwrap().iter().zip(&expected).enumerate() {
            assert!((got - want).abs() < 1e-4, "CPU MoE mismatch at {i}: {got} vs {want}");
        }

        #[cfg(target_os = "macos")]
        {
            let plan = place_graph_default(
                &mut g,
                PlacementMode::Auto,
                &[Device::Cpu, Device::Metal],
                None,
                &device_supports_op,
            )
            .unwrap();
            assert!(plan.segments.iter().all(|s| s.device == Device::Metal), "every MoE node lowers to Metal");
            let rt = crate::graph::metal::Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir()).unwrap();
            let metal_out = crate::graph::metal::Session::new(rt, &weights).unwrap().run(&g, inputs).unwrap();
            for (i, (got, want)) in metal_out[0].as_f32().unwrap().iter().zip(&expected).enumerate() {
                assert!((got - want).abs() < 1e-4, "Metal MoE mismatch at {i}: {got} vs {want}");
            }
        }
    }
}
