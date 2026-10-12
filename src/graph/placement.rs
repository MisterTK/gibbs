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
        Op::Linear => operand(1),
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
    let _ = crate::graph::lower_moe_pass(graph);
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
    fn test_moe_lowering_and_topk_parity() {
        use crate::graph::Tensor;

        let (m, d_in, d_out, experts, top_k) = (4, 8, 8, 4, 2);

        let mut seed = 42u64;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        let mut x_data = Vec::with_capacity(m * d_in);
        for _ in 0..m * d_in {
            x_data.push(next_f32());
        }

        let mut router_data = Vec::with_capacity(experts * d_in);
        for _ in 0..experts * d_in {
            router_data.push(next_f32());
        }

        let mut exp_data = Vec::with_capacity(experts);
        for _ in 0..experts {
            let mut ed = Vec::with_capacity(d_out * d_in);
            for _ in 0..d_out * d_in {
                ed.push(next_f32());
            }
            exp_data.push(ed);
        }

        // 1. Scalar Top-K reference calculation
        let mut expected = vec![0.0f32; m * d_out];
        for r in 0..m {
            let xr = &x_data[r * d_in..(r + 1) * d_in];
            let mut logits = Vec::with_capacity(experts);
            for e in 0..experts {
                let rw = &router_data[e * d_in..(e + 1) * d_in];
                let dot: f32 = xr.iter().zip(rw).map(|(&a, &b)| a * b).sum();
                logits.push(dot);
            }

            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top = &indexed[..top_k];
            let max_val = top[0].1;
            let mut sum_e = 0.0f32;
            let mut exps = Vec::with_capacity(top_k);
            for &(_, val) in top {
                let e = (val - max_val).exp();
                exps.push(e);
                sum_e += e;
            }
            let inv = 1.0 / sum_e.max(1e-12);
            let mut probs = vec![0.0f32; experts];
            for (&(idx, _), e) in top.iter().zip(exps) {
                probs[idx] = e * inv;
            }

            for e in 0..experts {
                let p = probs[e];
                if p == 0.0 {
                    continue;
                }
                let ew = &exp_data[e];
                for o in 0..d_out {
                    let w_row = &ew[o * d_in..(o + 1) * d_in];
                    let dot: f32 = xr.iter().zip(w_row).map(|(&a, &b)| a * b).sum();
                    expected[r * d_out + o] += p * dot;
                }
            }
        }

        // 2. Build graph with Op::Moe
        let mut g = Graph::new();
        let x = g.input("x", vec![m, d_in], DType::F32);
        let rw = g.weight("router", vec![experts, d_in], DType::F32);
        let mut ins = vec![x, rw];
        for e in 0..experts {
            ins.push(g.weight(format!("exp_{e}"), vec![d_out, d_in], DType::F32));
        }
        let out = g.add(
            "moe",
            Op::Moe { experts, top_k },
            &ins,
            vec![m, d_out],
            DType::F32,
        );
        g.output(out);

        let mut weights = crate::graph::Weights::new();
        weights.insert("router", Tensor::F32(router_data));
        for e in 0..experts {
            weights.insert(format!("exp_{e}"), Tensor::F32(exp_data[e].clone()));
        }
        let inputs = vec![Tensor::F32(x_data)];

        // Run on CPU
        let plan = place_graph_default(
            &mut g,
            PlacementMode::Pin(Device::Cpu),
            &[Device::Cpu],
            None,
            &device_supports_op,
        )
        .unwrap();
        assert_eq!(plan.segments[0].device, Device::Cpu);
        let mut state = crate::graph::StateStore::new();
        let cpu_out = crate::graph::cpu::run(&g, &weights, &mut state, inputs.clone()).unwrap();
        let cpu_out_f32 = cpu_out[0].as_f32().unwrap();

        for i in 0..m * d_out {
            assert!(
                (cpu_out_f32[i] - expected[i]).abs() < 1e-4,
                "CPU MoE mismatch at {i}: got {} vs expected {}",
                cpu_out_f32[i],
                expected[i]
            );
        }

        // Run on Metal if available
        #[cfg(target_os = "macos")]
        {
            let rt =
                crate::graph::metal::Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir())
                    .unwrap();
            let session = crate::graph::metal::Session::new(rt, &weights).unwrap();
            let metal_out = session.run(&g, inputs).unwrap();
            let metal_out_f32 = metal_out[0].as_f32().unwrap();
            for i in 0..m * d_out {
                assert!(
                    (metal_out_f32[i] - expected[i]).abs() < 1e-4,
                    "Metal MoE mismatch at {i}: got {} vs expected {}",
                    metal_out_f32[i],
                    expected[i]
                );
            }
        }

        // 3. Error on zero experts
        let mut bad_g = Graph::new();
        let bad_x = bad_g.input("x", vec![2, 4], DType::F32);
        let bad_rw = bad_g.weight("rw", vec![4, 4], DType::F32);
        let bad_out = bad_g.add(
            "bad_moe",
            Op::Moe {
                experts: 0,
                top_k: 1,
            },
            &[bad_x, bad_rw],
            vec![2, 4],
            DType::F32,
        );
        bad_g.output(bad_out);
        assert!(crate::graph::lower_moe_pass(&mut bad_g).is_err());
    }
}
