//! Universal compiler optimization passes for Graph IR.

pub mod canonical;
pub mod decompose;
pub mod dtype_adapt;
pub mod fusion;
pub mod megakernel;

use crate::graph::pattern::validate_ssa;
use crate::graph::{lower_moe_pass, Graph, Weights};

/// Runs all canonical graph optimization passes on `graph` and its `weights`.
pub fn run_optimization_passes(
    graph: &mut Graph,
    mut weights: Option<&mut Weights>,
) -> Result<(), String> {
    canonical::run(graph);
    decompose::run(graph);
    dtype_adapt::run(graph);
    // 1. Lower MoE nodes if present
    lower_moe_pass(graph)?;

    // 2. Megakernel Fusion (fuses entire transformer blocks: RmsNorm -> Gate/Up -> SwiGlu -> Down -> Add)
    megakernel::fuse_megakernel_blocks_pass(graph);

    // 3. Gate-Up MLP Fusion (for any unfused MLP blocks)
    fusion::fuse_gate_up_pass(graph, weights.as_deref_mut());

    // 4. QKV Projection Fusion
    fusion::fuse_qkv_pass(graph, weights.as_deref_mut());
    // 5. Verify graph SSA integrity
    validate_ssa(graph)?;

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Op, Tensor};

    #[test]
    fn test_run_optimization_passes_end_to_end() {
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 64], DType::F32);
        let g_attn = g.weight("g_attn", vec![64], DType::F32);
        let norm1 = g.add(
            "norm1",
            Op::RmsNorm { eps: 1e-5 },
            &[in_x, g_attn],
            vec![1, 64],
            DType::F32,
        );

        let w_q = g.weight("w_q", vec![64, 64], DType::F32);
        let w_k = g.weight("w_k", vec![64, 64], DType::F32);
        let w_v = g.weight("w_v", vec![64, 64], DType::F32);
        let q = g.add("q", Op::Linear, &[norm1, w_q], vec![1, 64], DType::F32);
        let k = g.add("k", Op::Linear, &[norm1, w_k], vec![1, 64], DType::F32);
        let v = g.add("v", Op::Linear, &[norm1, w_v], vec![1, 64], DType::F32);

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

        let w_out = g.weight("w_out", vec![64, 64], DType::F32);
        let out_proj = g.add(
            "out_proj",
            Op::Linear,
            &[attn, w_out],
            vec![1, 64],
            DType::F32,
        );
        let res1 = g.add("res1", Op::Add, &[out_proj, in_x], vec![1, 64], DType::F32);

        let g_ff = g.weight("g_ff", vec![64], DType::F32);
        let norm2 = g.add(
            "norm2",
            Op::RmsNorm { eps: 1e-5 },
            &[res1, g_ff],
            vec![1, 64],
            DType::F32,
        );

        let w_gate = g.weight("w_gate", vec![128, 64], DType::F32);
        let w_up = g.weight("w_up", vec![128, 64], DType::F32);
        let gate = g.add(
            "gate",
            Op::Linear,
            &[norm2, w_gate],
            vec![1, 128],
            DType::F32,
        );
        let up = g.add("up", Op::Linear, &[norm2, w_up], vec![1, 128], DType::F32);
        let mlp = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![1, 128], DType::F32);

        let w_down = g.weight("w_down", vec![64, 128], DType::F32);
        let down = g.add("down", Op::Linear, &[mlp, w_down], vec![1, 64], DType::F32);
        let res2 = g.add("res2", Op::Add, &[down, res1], vec![1, 64], DType::F32);
        g.output(res2);

        let mut weights = Weights::new();
        weights.insert("g_attn", Tensor::F32(vec![1.0; 64]));
        weights.insert("w_q", Tensor::F32(vec![0.1; 64 * 64]));
        weights.insert("w_k", Tensor::F32(vec![0.2; 64 * 64]));
        weights.insert("w_v", Tensor::F32(vec![0.3; 64 * 64]));
        weights.insert("w_out", Tensor::F32(vec![0.05; 64 * 64]));
        weights.insert("g_ff", Tensor::F32(vec![1.0; 64]));
        weights.insert("w_gate", Tensor::F32(vec![0.4; 128 * 64]));
        weights.insert("w_up", Tensor::F32(vec![0.5; 128 * 64]));
        weights.insert("w_down", Tensor::F32(vec![0.02; 64 * 128]));

        let inp = vec![Tensor::F32(vec![0.5; 64])];
        let baseline = crate::graph::cpu::run_stateless(&g, &weights, inp.clone()).unwrap();

        let mut opt_g = g.clone();
        let mut opt_w = weights.clone();
        let res = run_optimization_passes(&mut opt_g, Some(&mut opt_w));
        assert!(res.is_ok(), "Optimization passes must succeed: {:?}", res);

        let opt_res = crate::graph::cpu::run_stateless(&opt_g, &opt_w, inp).unwrap();
        assert_eq!(baseline.len(), opt_res.len());
        let b_slice = baseline[0].as_f32().unwrap();
        let o_slice = opt_res[0].as_f32().unwrap();
        assert_eq!(b_slice.len(), o_slice.len());
        for (i, (&b_v, &o_v)) in b_slice.iter().zip(o_slice).enumerate() {
            let diff = (b_v - o_v).abs();
            assert!(
                diff < 1e-4,
                "End-to-end parity failed at {i}: {b_v} vs {o_v}"
            );
        }

        let node_names: Vec<&str> = opt_g.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(!node_names.contains(&"q"));
        assert!(!node_names.contains(&"k"));
        assert!(!node_names.contains(&"v"));
        assert!(!node_names.contains(&"gate"));
        assert!(!node_names.contains(&"up"));
        assert!(node_names.iter().any(|n| n.ends_with(".fused_qkv")));
        assert!(node_names.iter().any(|n| n.ends_with(".fused_mlp")));
    }
}
