//! Cost-modeled device placement pass and segment partitioner for the graph IR.

use crate::graph::{DType, Graph, NodeId, Op, ValueId};
use crate::hardware::{Device, DeviceProfile, PlacementMode};
use std::collections::HashMap;

/// A contiguous sequence of graph nodes assigned to run on a single compute device.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub device: Device,
    pub nodes: Vec<NodeId>,
}

/// Placement telemetry detailing operation distribution across devices.
#[derive(Debug, Clone, Default)]
pub struct PlacementReport {
    pub device_op_counts: HashMap<Device, usize>,
    pub device_cost_shares: HashMap<Device, f64>,
    pub fallbacks: Vec<String>,
}

impl PlacementReport {
    pub fn summary(&self) -> String {
        let total_ops: usize = self.device_op_counts.values().sum();
        let mut parts = Vec::new();
        for (&dev, &count) in &self.device_op_counts {
            let pct = if total_ops > 0 {
                (count as f64 / total_ops as f64) * 100.0
            } else {
                0.0
            };
            let cost_pct = self.device_cost_shares.get(&dev).copied().unwrap_or(pct);
            parts.push(format!("{dev}: {count} ops ({cost_pct:.1}% cost share)"));
        }
        parts.join(", ")
    }

    pub fn primary_device(&self) -> Device {
        self.device_cost_shares
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&dev, _)| dev)
            .or_else(|| {
                self.device_op_counts
                    .iter()
                    .max_by_key(|a| a.1)
                    .map(|(&dev, _)| dev)
            })
            .unwrap_or(Device::Cpu)
    }
}

/// The partitioned graph execution plan.
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

/// Estimates computational complexity (FLOPs) and memory movement (bytes) for a node.
fn estimate_node_work(graph: &Graph, node_id: NodeId) -> (f64, f64) {
    let node = graph.node(node_id);
    let out_elements = node
        .outputs
        .first()
        .map(|&v| {
            let shape = &graph.value(v).shape;
            if shape.is_empty() {
                1
            } else {
                shape.iter().product::<usize>()
            }
        })
        .unwrap_or(1) as f64;

    match &node.op {
        Op::Linear | Op::MatMul { .. } => {
            let in_dim = node
                .inputs
                .get(1)
                .map(|&v| graph.value(v).shape.last().copied().unwrap_or(1))
                .unwrap_or(1) as f64;
            let out_dim = node
                .inputs
                .get(1)
                .and_then(|&v| {
                    let s = &graph.value(v).shape;
                    if s.len() >= 2 {
                        Some(s[s.len() - 2])
                    } else {
                        None
                    }
                })
                .unwrap_or(out_elements as usize) as f64;
            let rows = (out_elements / out_dim.max(1.0)).max(1.0);
            let flops = 2.0 * rows * in_dim * out_dim;
            let bytes = (rows * in_dim + out_dim * in_dim + out_elements) * 4.0;
            (flops, bytes)
        }
        Op::Attention {
            heads, kv_heads: _, ..
        }
        | Op::CachedAttention { heads, .. } => {
            let flops = out_elements * (*heads as f64) * 4.0;
            let bytes = out_elements * 8.0;
            (flops, bytes)
        }
        Op::Megakernel(desc) => match &desc.kind {
            crate::graph::MegakernelKind::Mlp {
                hidden_dim,
                intermediate_dim,
                ..
            } => {
                let flops = 2.0 * (*hidden_dim as f64) * (*intermediate_dim as f64) * 3.0;
                let bytes = (*hidden_dim as f64) * (*intermediate_dim as f64) * 3.0 * 2.0;
                (flops, bytes)
            }
            crate::graph::MegakernelKind::Attention {
                hidden_dim,
                heads,
                head_dim,
                ..
            } => {
                let flops = 2.0 * (*hidden_dim as f64) * ((*heads * *head_dim) as f64) * 2.0;
                let bytes = (*hidden_dim as f64) * ((*heads * *head_dim) as f64) * 2.0 * 2.0;
                (flops, bytes)
            }
        },
        _ => {
            let flops = out_elements;
            let bytes = out_elements * 8.0;
            (flops, bytes)
        }
    }
}

/// Computes transfer latency in seconds for passing a tensor between two devices.
fn transfer_cost(bytes: f64, src: Device, dst: Device, profile_dst: &DeviceProfile) -> f64 {
    if src == dst {
        0.0
    } else {
        let bus_speed = (profile_dst.bus_bandwidth_gbps * 1e9).max(1e8);
        (bytes / bus_speed) + (profile_dst.dispatch_us * 1e-6)
    }
}

/// Places each node in `graph` onto an available device using measured device profiles.
pub fn place_graph(
    graph: &mut Graph,
    mode: PlacementMode,
    profiles: &[(Device, DeviceProfile)],
) -> PlacementPlan {
    let _ = crate::graph::lower_moe_pass(graph);
    let mut node_devices: Vec<Device> = Vec::with_capacity(graph.nodes.len());
    let mut fallbacks = Vec::new();
    let mut node_costs = Vec::with_capacity(graph.nodes.len());
    let profile_map: HashMap<Device, DeviceProfile> = profiles.iter().copied().collect();
    let available: Vec<Device> = profiles.iter().map(|(d, _)| *d).collect();

    // Map each value to the device producing it
    let mut value_device: HashMap<ValueId, Device> = HashMap::new();
    for &inp in &graph.inputs {
        value_device.insert(inp, Device::Cpu);
    }

    for (i, node) in graph.nodes.iter().enumerate() {
        let node_id = NodeId(i as u32);
        let dtype = match &node.op {
            Op::Linear => node
                .inputs
                .get(1)
                .map(|&v| graph.value(v).dtype)
                .unwrap_or(DType::F32),
            Op::Embedding => node
                .inputs
                .get(0)
                .map(|&v| graph.value(v).dtype)
                .unwrap_or(DType::F32),
            Op::Megakernel(_) => node
                .inputs
                .get(2)
                .map(|&v| graph.value(v).dtype)
                .unwrap_or(DType::F32),
            _ => node
                .outputs
                .first()
                .map(|&v| graph.value(v).dtype)
                .unwrap_or(DType::F32),
        };
        let (flops, bytes) = estimate_node_work(graph, node_id);

        let chosen_device = match mode {
            PlacementMode::Pin(target) => {
                if available.contains(&target) && device_supports_op(target, &node.op, dtype) {
                    target
                } else {
                    if available.contains(&target) {
                        fallbacks.push(format!(
                            "Node '{}' (op {}) lacks kernel on {target}; falling back to CPU",
                            node.name,
                            node.op.name()
                        ));
                    }
                    Device::Cpu
                }
            }
            PlacementMode::Auto => {
                let mut best_dev = Device::Cpu;
                let mut best_cost = f64::INFINITY;

                for (&dev, prof) in &profile_map {
                    if !device_supports_op(dev, &node.op, dtype) {
                        continue;
                    }
                    let mem_bw = prof.mem_bandwidth_gbps * 1e9;
                    let comp_rate = prof.compute_gflops * 1e9;

                    let compute_time = (bytes / mem_bw).max(flops / comp_rate);

                    // Charge boundary transfer costs across edges
                    let mut boundary_time = 0.0;
                    for &inp in &node.inputs {
                        if let Some(&src_dev) = value_device.get(&inp) {
                            let inp_elements = {
                                let shape = &graph.value(inp).shape;
                                if shape.is_empty() {
                                    1
                                } else {
                                    shape.iter().product::<usize>()
                                }
                            } as f64;
                            let inp_bytes = inp_elements * 4.0;
                            boundary_time += transfer_cost(inp_bytes, src_dev, dev, prof);
                        }
                    }

                    let total_time = compute_time + boundary_time;
                    if total_time < best_cost {
                        best_cost = total_time;
                        best_dev = dev;
                    }
                }
                best_dev
            }
        };

        node_devices.push(chosen_device);
        for &out_val in &node.outputs {
            value_device.insert(out_val, chosen_device);
        }
        node_costs.push((chosen_device, flops.max(bytes)));
    }

    // Assign node devices into graph
    for (node, &dev) in graph.nodes.iter_mut().zip(&node_devices) {
        node.device = dev;
    }

    // Partition into contiguous segments
    let mut segments: Vec<Segment> = Vec::new();
    for (i, &dev) in node_devices.iter().enumerate() {
        let nid = NodeId(i as u32);
        if let Some(seg) = segments.last_mut() {
            if seg.device == dev {
                seg.nodes.push(nid);
                continue;
            }
        }
        segments.push(Segment {
            device: dev,
            nodes: vec![nid],
        });
    }

    // Calculate report statistics
    let mut device_op_counts = HashMap::new();
    let mut device_cost_totals: HashMap<Device, f64> = HashMap::new();
    let mut total_cost = 0.0;
    for &(dev, cost) in &node_costs {
        *device_op_counts.entry(dev).or_insert(0) += 1;
        *device_cost_totals.entry(dev).or_insert(0.0) += cost;
        total_cost += cost;
    }

    let mut device_cost_shares = HashMap::new();
    for (dev, cost) in device_cost_totals {
        let share = if total_cost > 0.0 {
            (cost / total_cost) * 100.0
        } else {
            0.0
        };
        device_cost_shares.insert(dev, share);
    }

    PlacementPlan {
        segments,
        report: PlacementReport {
            device_op_counts,
            device_cost_shares,
            fallbacks,
        },
    }
}

/// Convenience helper constructing standard device profiles from available devices and measured bandwidth.
pub fn place_graph_default(
    graph: &mut Graph,
    mode: PlacementMode,
    available: &[Device],
    host_bandwidth_gbps: Option<f64>,
) -> PlacementPlan {
    let profiles: Vec<(Device, DeviceProfile)> = available
        .iter()
        .map(|&d| (d, DeviceProfile::for_device(d, host_bandwidth_gbps)))
        .collect();
    place_graph(graph, mode, &profiles)
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
        );
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].device, Device::Cpu);
        assert_eq!(plan.report.device_op_counts[&Device::Cpu], 1);
    }

    #[test]
    fn test_placement_fallback_unsupported_op() {
        let mut g = Graph::new();
        let x = g.input("x", vec![4, 4], DType::F32);
        let idx = g.input("idx", vec![2], DType::U32);
        // Gather is supported on CPU, but NOT on Metal or Ane
        let out = g.add(
            "gather",
            Op::Gather { axis: 0 },
            &[x, idx],
            vec![2, 4],
            DType::F32,
        );
        g.output(out);

        let plan = place_graph_default(
            &mut g,
            PlacementMode::Pin(Device::Metal),
            &[Device::Cpu, Device::Metal],
            None,
        );
        assert_eq!(plan.segments[0].device, Device::Cpu);
        assert!(!plan.report.fallbacks.is_empty());
    }

    #[test]
    fn test_placement_auto_large_linear_vs_tiny_node() {
        let mut g = Graph::new();
        // CPU-only node: Gather
        let x = g.input("x", vec![2, 2048], DType::F32);
        let idx = g.input("idx", vec![2], DType::U32);
        let gat = g.add(
            "gather",
            Op::Gather { axis: 0 },
            &[x, idx],
            vec![2, 2048],
            DType::F32,
        );

        // A tiny op reading gather: on Auto, moving to Metal would cost transfer overhead > compute savings
        let small_out = g.add("add_tiny", Op::Add, &[gat, gat], vec![2, 2048], DType::F32);

        // A large linear projection with high FLOP count (2048 x 2048)
        let w_large = g.weight("w_large", vec![2048, 2048], DType::F32);
        let lin_large = g.add(
            "lin_large",
            Op::Linear,
            &[small_out, w_large],
            vec![2, 2048],
            DType::F32,
        );
        g.output(lin_large);

        #[cfg(target_os = "macos")]
        {
            let plan = place_graph_default(
                &mut g,
                PlacementMode::Auto,
                &[Device::Cpu, Device::Metal],
                Some(250.0),
            );
            // Gather must be on CPU (unsupported on Metal)
            assert_eq!(g.node(NodeId(0)).device, Device::Cpu);
            // Large Linear should be placed on Metal because compute savings outweigh transfer cost
            assert_eq!(g.node(NodeId(2)).device, Device::Metal);
            assert!(plan.segments.len() >= 2);
        }
    }
    #[test]
    fn test_placement_operand_dtype_routing_q6k_vs_q4_0() {
        use crate::hardware::QuantFormat;

        let mut g = Graph::new();
        let x = g.input("x", vec![1, 64], DType::F32);
        // Linear with Q4_0 should be supported on Metal
        let w_q4 = g.weight("w_q4", vec![32, 64], DType::Quant(QuantFormat::GGUF_Q4_0));
        let lin_q4 = g.add("lin_q4", Op::Linear, &[x, w_q4], vec![1, 32], DType::F32);

        // Linear with Q6_K should NOT be supported on Metal and must fallback to CPU
        let w_q6k = g.weight("w_q6k", vec![32, 32], DType::Quant(QuantFormat::GGUF_Q6_K));
        let lin_q6k = g.add(
            "lin_q6k",
            Op::Linear,
            &[lin_q4, w_q6k],
            vec![1, 32],
            DType::F32,
        );
        g.output(lin_q6k);

        #[cfg(target_os = "macos")]
        {
            let plan = place_graph_default(
                &mut g,
                PlacementMode::Pin(Device::Metal),
                &[Device::Cpu, Device::Metal],
                None,
            );
            // lin_q4 should be on Metal, lin_q6k should fall back to CPU
            assert_eq!(plan.segments.len(), 2);
            assert_eq!(plan.segments[0].device, Device::Metal);
            assert_eq!(plan.segments[1].device, Device::Cpu);
            assert!(plan.report.fallbacks.iter().any(|f| f.contains("lin_q6k")));
        }
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
        );
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
