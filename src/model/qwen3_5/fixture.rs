//! Tiny synthetic Qwen3.5 checkpoints (DeltaNet, DeltaNet, gated attention; optionally a vision
//! tower and an MTP head) and the graphs built from them. Fixtures of `gpu_tests` and of
//! `tests/cpu_free_matrix.rs`.

use super::text::{Arch, TextModel};
use super::vision::Vision;
use crate::graph::{Graph, Weights as GraphWeights};
use crate::model::weights::Weights;
use serde_json::Value;

pub(super) const H: usize = 64;
pub(super) const V: usize = 96;
const HD: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const INTER: usize = 96;
/// DeltaNet: 1 key head, 2 value heads, 16-wide, conv kernel 4 -> conv dim 64.
const CD: usize = 64;
pub(super) const IMAGE_TOKEN: u32 = 90;

fn values(name: &str, n: usize) -> Vec<f32> {
    let seed: usize = name.bytes().enumerate().map(|(i, b)| (i + 1) * b as usize).sum();
    let norm = name.ends_with("norm.weight") || name.contains("norm1.weight") || name.contains("norm2.weight");
    let visual_norm = name.contains("visual") && norm;
    (0..n)
        .map(|i| {
            let r = (((i * 19 + seed * 13) % 41) as f32 - 20.0) / 25.0;
            match (norm, visual_norm) {
                (_, true) => 1.0 + r * 0.2,
                (true, false) => r * 0.3,
                _ => r * if name.ends_with("neg_A") { 0.2 } else { 1.0 },
            }
        })
        .collect()
}

fn attn_specs(p: &str, specs: &mut Vec<(String, Vec<usize>)>) {
    let mut s = |n: &str, shape: Vec<usize>| specs.push((format!("{p}.{n}"), shape));
    s("input_layernorm.weight", vec![H]);
    s("post_attention_layernorm.weight", vec![H]);
    s("self_attn.q_proj.weight", vec![HEADS * 2 * HD, H]);
    s("self_attn.k_proj.weight", vec![KV_HEADS * HD, H]);
    s("self_attn.v_proj.weight", vec![KV_HEADS * HD, H]);
    s("self_attn.o_proj.weight", vec![H, HEADS * HD]);
    s("self_attn.q_norm.weight", vec![HD]);
    s("self_attn.k_norm.weight", vec![HD]);
    s("mlp.gate_proj.weight", vec![INTER, H]);
    s("mlp.up_proj.weight", vec![INTER, H]);
    s("mlp.down_proj.weight", vec![H, INTER]);
}

fn delta_specs(p: &str, specs: &mut Vec<(String, Vec<usize>)>) {
    let mut s = |n: &str, shape: Vec<usize>| specs.push((format!("{p}.{n}"), shape));
    s("input_layernorm.weight", vec![H]);
    s("post_attention_layernorm.weight", vec![H]);
    s("linear_attn.in_proj_qkv.weight", vec![CD, H]);
    s("linear_attn.in_proj_z.weight", vec![32, H]);
    s("linear_attn.in_proj_b.weight", vec![2, H]);
    s("linear_attn.in_proj_a.weight", vec![2, H]);
    s("linear_attn.conv1d.weight", vec![CD, 4]);
    s("linear_attn.neg_A", vec![2]);
    s("linear_attn.dt_bias", vec![2]);
    s("linear_attn.norm.weight", vec![16]);
    s("linear_attn.out_proj.weight", vec![H, 32]);
    s("mlp.gate_proj.weight", vec![INTER, H]);
    s("mlp.up_proj.weight", vec![INTER, H]);
    s("mlp.down_proj.weight", vec![H, INTER]);
}

/// Vision tower: hidden 32, 2 heads (head dim 16), 2 blocks, 4x4 position grid, patch 2,
/// temporal 2, merge 2, one deepstack merger after block 0.
fn vision_specs(specs: &mut Vec<(String, Vec<usize>)>) {
    let (d, k) = (32, 3 * 2 * 2 * 2);
    let mut s = |n: &str, shape: Vec<usize>| specs.push((format!("model.visual.{n}"), shape));
    s("patch_embed.proj.weight", vec![d, k]);
    s("patch_embed.proj.bias", vec![d]);
    s("pos_embed.weight", vec![16, d]);
    for b in 0..2 {
        for n in ["norm1", "norm2"] {
            s(&format!("blocks.{b}.{n}.weight"), vec![d]);
            s(&format!("blocks.{b}.{n}.bias"), vec![d]);
        }
        for (n, r, c) in [
            ("attn.qkv", 3 * d, d),
            ("attn.proj", d, d),
            ("mlp.linear_fc1", 64, d),
            ("mlp.linear_fc2", d, 64),
        ] {
            s(&format!("blocks.{b}.{n}.weight"), vec![r, c]);
            s(&format!("blocks.{b}.{n}.bias"), vec![r]);
        }
    }
    for (m, wide) in [("merger", d), ("deepstack_merger_list.0", 4 * d)] {
        s(&format!("{m}.norm.weight"), vec![wide]);
        s(&format!("{m}.norm.bias"), vec![wide]);
        s(&format!("{m}.linear_fc1.weight"), vec![4 * d, 4 * d]);
        s(&format!("{m}.linear_fc1.bias"), vec![4 * d]);
        s(&format!("{m}.linear_fc2.weight"), vec![H, 4 * d]);
        s(&format!("{m}.linear_fc2.bias"), vec![H]);
    }
}

fn mtp_specs(specs: &mut Vec<(String, Vec<usize>)>) {
    specs.push(("mtp.fc.weight".into(), vec![H, 2 * H]));
    specs.push(("mtp.pre_fc_norm_embedding.weight".into(), vec![H]));
    specs.push(("mtp.pre_fc_norm_hidden.weight".into(), vec![H]));
    specs.push(("mtp.norm.weight".into(), vec![H]));
    let mut a = Vec::new();
    attn_specs("mtp.layers.0", &mut a);
    specs.extend(a);
}

/// Layers: DeltaNet, DeltaNet, gated attention (so deepstack layer 0 is recurrent); optional
/// vision tower and MTP head. `cfg` is the whole `config.json`.
pub(super) fn checkpoint(vision: bool, mtp: bool) -> (Weights, Value) {
    let mut specs: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![V, H]),
        ("model.norm.weight".into(), vec![H]),
        ("lm_head.weight".into(), vec![V, H]),
    ];
    delta_specs("model.layers.0", &mut specs);
    delta_specs("model.layers.1", &mut specs);
    attn_specs("model.layers.2", &mut specs);
    if vision {
        vision_specs(&mut specs);
    }
    if mtp {
        mtp_specs(&mut specs);
    }
    let sizes: std::collections::HashMap<String, usize> = specs
        .iter()
        .map(|(n, s)| (n.clone(), s.iter().product()))
        .collect();
    let ck = Weights::from_f32(specs, |n| Ok(values(n, sizes[n]))).unwrap();
    let mut cfg = serde_json::json!({
        "hidden_size": H,
        "vocab_size": V,
        "num_hidden_layers": 3,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HD,
        "intermediate_size": INTER,
        "max_position_embeddings": 128,
        "attn_output_gate": true,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": u32::from(mtp),
        "rope_parameters": {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [2, 1, 1]
        }
    });
    if vision {
        cfg = serde_json::json!({
            "text_config": cfg,
            "image_token_id": IMAGE_TOKEN,
            "vision_config": {
                "hidden_size": 32, "num_heads": 2, "depth": 2, "num_position_embeddings": 16,
                "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [0]
            }
        });
    }
    (ck, cfg)
}


/// Metal-fixture graph of the vision tower on an 8x12 patch grid.
pub fn tiny_vision_tower_graph() -> Result<(Graph, GraphWeights), String> {
    let (ck, cfg) = checkpoint(true, false);
    Ok(Vision::load(&ck, &cfg["vision_config"])?.tower_graph(8, 12))
}

/// Metal-fixture graph of the MTP head step.
pub fn tiny_mtp_graph() -> Result<(Graph, GraphWeights), String> {
    let (ck, cfg) = checkpoint(false, true);
    TextModel::load(&ck, &cfg, Arch::Qwen35)?.mtp_decode_graph(64)
}

/// Decode graph of the model a tiny `.ninfer` artifact loads as (checkpoint decode and dequantize
/// to bf16 at load, then the same program as any Qwen3.5).
pub fn tiny_ninfer_decode_graph() -> Result<(Graph, GraphWeights), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let path = dir.path().join("tiny.ninfer");
    crate::model::ninfer::write_tiny_artifact(&path)?;
    let (ck, cfg) = crate::model::ninfer::Artifact::open(&path)?.checkpoint()?;
    TextModel::load(&ck, &cfg, Arch::Qwen35)?.to_decode_graph()
}
