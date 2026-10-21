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

/// Media soft tokens placed over prompt rows by the GPU `Scatter`.
fn scatter_fixture(name: &str) {
    let base: Vec<f32> = (0..4 * 8).map(|i| i as f32).collect();
    let out = gibbs_engine::model::vision_graph::scatter_rows(base.clone(), 4, 8, &[1, 3], vec![-1.0; 16], PlacementMode::Auto)
        .unwrap_or_else(|e| panic!("{name} scatter: {e}"));
    assert_eq!(&out[..8], &base[..8], "{name}");
    assert!(out[8..16].iter().chain(&out[24..]).all(|&v| v == -1.0), "{name}");
}

/// Deterministic f32 tensors written as a safetensors checkpoint.
struct Ckpt(u64, Vec<(String, Vec<usize>, Vec<f32>)>);
impl Ckpt {
    fn new(seed: u64) -> Self {
        Self(seed, Vec::new())
    }
    fn s(&mut self, n: impl Into<String>, shape: &[usize], scale: f32) -> &mut Self {
        let v = (0..shape.iter().product::<usize>())
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect();
        self.1.push((n.into(), shape.to_vec(), v));
        self
    }
    fn w(&mut self, n: impl Into<String>, shape: &[usize]) -> &mut Self {
        self.s(n, shape, 1.0 / (shape[1] as f32).sqrt())
    }
    fn norm(&mut self, n: impl Into<String>, d: usize, _c: f32) -> &mut Self {
        self.1.push((n.into(), vec![d], vec![1.0; d]));
        self
    }
    fn write_checkpoint(&self, dir: &std::path::Path, cfg: &serde_json::Value, vocab: usize) {
        use serde_json::json;
        std::fs::create_dir_all(dir).unwrap();
        let (mut h, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
        for (n, shape, v) in &self.1 {
            let a = data.len();
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            h.insert(n.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [a, data.len()]}));
        }
        let mut hb = serde_json::to_vec(&h).unwrap();
        hb.resize(hb.len().next_multiple_of(8), b' ');
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend(hb);
        out.extend(data);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let vm: serde_json::Map<String, serde_json::Value> = (0..vocab).map(|i| (format!("t{i}"), json!(i))).collect();
        let tok = json!({"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vm, "unk_token": "t0"}});
        std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    }
}

fn artifacts(dir: &std::path::Path) -> gibbs_engine::model::hub::ModelArtifacts {
    gibbs_engine::model::hub::ModelArtifacts {
        model_id: "test/cpu-free".into(),
        local_dir: Some(dir.to_path_buf()),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        is_cached: true,
        ..Default::default()
    }
}

/// A tiny random T5 (gated) or BART (3-beam search, forking beam state) through
/// `runtime::encdec` on Metal: encoder, search and decoder steps all on the GPU.
fn encdec_on_gpu(name: &str, bart: bool) {
    use serde_json::json;
    use gibbs_engine::runtime::encdec::SpecEncDec;
    use gibbs_engine::model::native::NativeLm;
    let (v, d) = (40, 32);
    let dir = std::env::temp_dir().join(format!("gibbs-matrix-encdec-{name}"));
    let mut w = Ckpt::new(7);
    let cfg = if bart {
        w.s("model.shared.weight", &[v, d], 0.3);
        for side in ["encoder", "decoder"] {
            let p = format!("model.{side}");
            w.s(format!("{p}.embed_positions.weight"), &[66, d], 0.3);
            w.norm(format!("{p}.layernorm_embedding.weight"), d, 1.0).s(format!("{p}.layernorm_embedding.bias"), &[d], 0.1);
            let mut attns = vec!["self_attn"];
            if side == "decoder" {
                attns.push("encoder_attn");
            }
            for a in attns {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    w.w(format!("{p}.layers.0.{a}.{proj}.weight"), &[d, d]).s(format!("{p}.layers.0.{a}.{proj}.bias"), &[d], 0.1);
                }
                w.norm(format!("{p}.layers.0.{a}_layer_norm.weight"), d, 1.0).s(format!("{p}.layers.0.{a}_layer_norm.bias"), &[d], 0.1);
            }
            w.w(format!("{p}.layers.0.fc1.weight"), &[64, d]).s(format!("{p}.layers.0.fc1.bias"), &[64], 0.1);
            w.w(format!("{p}.layers.0.fc2.weight"), &[d, 64]).s(format!("{p}.layers.0.fc2.bias"), &[d], 0.1);
            w.norm(format!("{p}.layers.0.final_layer_norm.weight"), d, 1.0).s(format!("{p}.layers.0.final_layer_norm.bias"), &[d], 0.1);
        }
        json!({"architectures": ["BartForConditionalGeneration"], "model_type": "bart", "vocab_size": v,
            "d_model": d, "max_position_embeddings": 64, "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 4, "decoder_attention_heads": 4, "activation_function": "gelu",
            "bos_token_id": 0, "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2,
            "num_beams": 3, "max_length": 8})
    } else {
        w.s("shared.weight", &[v, d], 1.0);
        for stack in ["encoder", "decoder"] {
            let b = format!("{stack}.block.0.layer");
            let mut subs = vec![(0, "SelfAttention")];
            if stack == "decoder" {
                subs.push((1, "EncDecAttention"));
            }
            for (l, m) in subs {
                w.norm(format!("{b}.{l}.layer_norm.weight"), d, 1.0);
                for p in ["q", "k", "v", "o"] {
                    w.w(format!("{b}.{l}.{m}.{p}.weight"), &[d, d]);
                }
            }
            w.s(format!("{b}.0.SelfAttention.relative_attention_bias.weight"), &[8, 4], 1.0);
            let f = if stack == "decoder" { 2 } else { 1 };
            w.norm(format!("{b}.{f}.layer_norm.weight"), d, 1.0);
            w.w(format!("{b}.{f}.DenseReluDense.wi_0.weight"), &[48, d]).w(format!("{b}.{f}.DenseReluDense.wi_1.weight"), &[48, d]);
            w.w(format!("{b}.{f}.DenseReluDense.wo.weight"), &[d, 48]);
            w.norm(format!("{stack}.final_layer_norm.weight"), d, 1.0);
        }
        json!({"architectures": ["T5ForConditionalGeneration"], "model_type": "t5", "vocab_size": v,
            "d_model": d, "d_kv": 8, "num_heads": 4, "d_ff": 48, "num_layers": 1, "relative_attention_num_buckets": 8,
            "relative_attention_max_distance": 16, "feed_forward_proj": "gated-gelu", "n_positions": 64})
    };
    w.write_checkpoint(&dir, &cfg, v);
    let reg = gibbs_engine::arch::registry().unwrap();
    let m = gibbs_engine::arch::load::load(&artifacts(&dir), reg, None).unwrap_or_else(|e| panic!("{name}: {e}"));
    let lm = SpecEncDec::new(m, PlacementMode::Pin(Device::Metal)).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(lm.device(), Device::Metal, "{name}");
    let src = if bart { b"t3 t5 t7".iter().map(|&b| b as u32).collect() } else { vec![3u32, 5, 7] };
    let (prompt, mut st) = lm.prepare(src, &[]).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut l = lm.forward(&prompt, 0, &mut st).unwrap_or_else(|e| panic!("{name}: {e}"));
    for pos in prompt.len()..prompt.len() + 2 {
        let t = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
        l = lm.forward(&[t], pos, &mut st).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    assert_eq!(lm.cpu_node_count(), 0, "{name}: nodes run on the CPU");
}

/// A tiny Qwen3.5 checkpoint (DeltaNet, DeltaNet, gated attention; `moe`: 8 routed experts + a
/// shared one, else dense with an MTP head; `vl`: nested decoder + ViT) through the `qwen3_5`
/// spec on Metal: prefill, the shipped drafter's proposal, and two decode steps.
fn qwen35_on_gpu(name: &str, moe: bool, vl: bool) {
    use gibbs_engine::model::native::{MediaInput, NativeLm};
    use serde_json::json;
    const H: usize = 64;
    const IMAGE: u32 = 90;
    fn mlp(w: &mut Ckpt, p: &str, moe: bool) {
        if moe {
            w.s(format!("{p}.mlp.gate.weight"), &[8, H], 1.0)
                .w(format!("{p}.mlp.experts.gate_up_proj"), &[8, 64, H])
                .w(format!("{p}.mlp.experts.down_proj"), &[8, H, 32])
                .w(format!("{p}.mlp.shared_expert.gate_proj.weight"), &[32, H])
                .w(format!("{p}.mlp.shared_expert.up_proj.weight"), &[32, H])
                .w(format!("{p}.mlp.shared_expert.down_proj.weight"), &[H, 32])
                .w(format!("{p}.mlp.shared_expert_gate.weight"), &[1, H]);
        } else {
            w.w(format!("{p}.mlp.gate_proj.weight"), &[96, H])
                .w(format!("{p}.mlp.up_proj.weight"), &[96, H])
                .w(format!("{p}.mlp.down_proj.weight"), &[H, 96]);
        }
    }
    fn attn(w: &mut Ckpt, a: &str) {
        w.w(format!("{a}.q_proj.weight"), &[2 * 2 * 32, H])
            .w(format!("{a}.k_proj.weight"), &[32, H])
            .w(format!("{a}.v_proj.weight"), &[32, H])
            .w(format!("{a}.o_proj.weight"), &[H, 2 * 32])
            .norm(format!("{a}.q_norm.weight"), 32, 0.0)
            .norm(format!("{a}.k_norm.weight"), 32, 0.0);
    }
    let mut w = Ckpt::new(0x5157_454E_00F0);
    let p = if vl { "model.language_model" } else { "model" };
    w.w(format!("{p}.embed_tokens.weight"), &[96, H]);
    for l in 0..3 {
        let lp = format!("{p}.layers.{l}");
        w.norm(format!("{lp}.input_layernorm.weight"), H, 0.0)
            .norm(format!("{lp}.post_attention_layernorm.weight"), H, 0.0);
        if l < 2 {
            let a = format!("{lp}.linear_attn");
            w.w(format!("{a}.in_proj_qkv.weight"), &[64, H])
                .w(format!("{a}.in_proj_z.weight"), &[32, H])
                .w(format!("{a}.in_proj_b.weight"), &[2, H])
                .w(format!("{a}.in_proj_a.weight"), &[2, H])
                .w(format!("{a}.conv1d.weight"), &[64, 1, 4])
                .s(format!("{a}.A_log"), &[2], 0.5)
                .s(format!("{a}.dt_bias"), &[2], 0.5)
                .norm(format!("{a}.norm.weight"), 16, 1.0)
                .w(format!("{a}.out_proj.weight"), &[H, 32]);
        } else {
            attn(&mut w, &format!("{lp}.self_attn"));
        }
        mlp(&mut w, &lp, moe);
    }
    w.norm(format!("{p}.norm.weight"), H, 0.0).w("lm_head.weight", &[96, H]);
    let mtp = !moe && !vl;
    if mtp {
        w.w("mtp.fc.weight", &[H, 2 * H]);
        for n in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm", "layers.0.input_layernorm", "layers.0.post_attention_layernorm"] {
            w.norm(format!("mtp.{n}.weight"), H, 0.0);
        }
        attn(&mut w, "mtp.layers.0.self_attn");
        mlp(&mut w, "mtp.layers.0", false);
    }
    let mut text = json!({
        "model_type": if moe { "qwen3_5_moe" } else { "qwen3_5" },
        "architectures": [if moe { "Qwen3_5MoeForCausalLM" } else { "Qwen3_5ForCausalLM" }],
        "hidden_size": H, "vocab_size": 96, "num_hidden_layers": 3,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "intermediate_size": 96, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "attn_output_gate": true, "tie_word_embeddings": false, "eos_token_id": 2,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 1, "linear_num_value_heads": 2,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": mtp as u32,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "mrope_interleaved": true, "mrope_section": [2, 1, 1]}
    });
    if moe {
        let c = text.as_object_mut().unwrap();
        c.remove("intermediate_size");
        for (k, v) in [("num_experts", 8), ("num_experts_per_tok", 2), ("moe_intermediate_size", 32), ("shared_expert_intermediate_size", 32)] {
            c.insert(k.into(), json!(v));
        }
    }
    let cfg = if vl {
        // ViT: hidden 32, 2 heads, 2 blocks, 4x4 position grid, patch 2, temporal 2, merge 2, one
        // deepstack merger.
        let (d, v) = (32, "model.visual");
        w.w(format!("{v}.patch_embed.proj.weight"), &[d, 3, 2, 2, 2])
            .s(format!("{v}.patch_embed.proj.bias"), &[d], 0.1)
            .s(format!("{v}.pos_embed.weight"), &[16, d], 0.5);
        for b in 0..2 {
            let bp = format!("{v}.blocks.{b}");
            for n in ["norm1", "norm2"] {
                w.norm(format!("{bp}.{n}.weight"), d, 1.0).s(format!("{bp}.{n}.bias"), &[d], 0.1);
            }
            for (n, r, c) in [("attn.qkv", 3 * d, d), ("attn.proj", d, d), ("mlp.linear_fc1", 64, d), ("mlp.linear_fc2", d, 64)] {
                w.w(format!("{bp}.{n}.weight"), &[r, c]).s(format!("{bp}.{n}.bias"), &[r], 0.1);
            }
        }
        for (m, wide) in [("merger", d), ("deepstack_merger_list.0", 4 * d)] {
            let mp = format!("{v}.{m}");
            w.norm(format!("{mp}.norm.weight"), wide, 1.0)
                .s(format!("{mp}.norm.bias"), &[wide], 0.1)
                .w(format!("{mp}.linear_fc1.weight"), &[4 * d, 4 * d])
                .s(format!("{mp}.linear_fc1.bias"), &[4 * d], 0.1)
                .w(format!("{mp}.linear_fc2.weight"), &[H, 4 * d])
                .s(format!("{mp}.linear_fc2.bias"), &[H], 0.1);
        }
        text.as_object_mut().unwrap().remove("architectures");
        text["model_type"] = "qwen3_5_text".into();
        json!({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": IMAGE, "text_config": text,
            "vision_config": {"hidden_size": 32, "num_heads": 2, "depth": 2, "num_position_embeddings": 16,
                "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
                "intermediate_size": 64, "out_hidden_size": H, "hidden_act": "gelu_pytorch_tanh",
                "deepstack_visual_indexes": [0]}})
    } else {
        text
    };
    let dir = std::env::temp_dir().join(format!("gibbs_cpu_free_{name}_{}", std::process::id()));
    w.write_checkpoint(&dir, &cfg, 96);
    std::fs::write(
        dir.join("preprocessor_config.json"),
        r#"{"size":{"shortest_edge":64,"longest_edge":100000},"image_mean":[0.5,0.5,0.5],
            "image_std":[0.5,0.5,0.5],"patch_size":2,"temporal_patch_size":2,"merge_size":2}"#,
    )
    .unwrap();
    let m = gibbs_engine::arch::load::load(&artifacts(&dir), gibbs_engine::arch::registry().unwrap(), None)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(m.spec.family, "qwen3_5", "{name}");
    let lm = gibbs_engine::runtime::lm::SpecLm::new(m, PlacementMode::Pin(Device::Metal))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(lm.device(), Device::Metal, "{name}");
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
    let media = match vl {
        true => {
            let mut img = gibbs_engine::model::image::ImageBuffer::new(16, 24);
            img.data.iter_mut().enumerate().for_each(|(i, px)| *px = (i * 37 % 251) as u8);
            vec![MediaInput::Image(img.to_png())]
        }
        false => Vec::new(),
    };
    let prompt = if vl { vec![3, 7, IMAGE, 11, 5] } else { vec![4, 8, 15, 16, 23] };
    let (ids, mut st) = lm.prepare(prompt, &media).unwrap_or_else(|e| panic!("{name}: {e}"));
    let n = ids.len();
    let drafter = lm.drafter();
    assert_eq!(drafter.is_some(), mtp, "{name}: shipped MTP drafter");
    let mut l = match &drafter {
        Some(d) => {
            lm.set_taps(&mut st, &d.tap_layers()).unwrap();
            lm.forward(&ids[..n - 1], 0, &mut st).unwrap_or_else(|e| panic!("{name}: {e}"));
            let h = lm.take_taps(&mut st);
            let drafts = d
                .propose(&mut d.new_state(), &ids, &h, 3, &mut |l: &[f32]| argmax(l))
                .unwrap_or_else(|e| panic!("{name} drafter: {e}"));
            assert!(!drafts.is_empty(), "{name}: no drafts");
            lm.forward(&ids[n - 1..], n - 1, &mut st)
        }
        None => lm.forward(&ids, 0, &mut st),
    }
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    for pos in n..n + 2 {
        assert!(l.iter().all(|x| x.is_finite()), "{name}: non-finite logits");
        l = lm.forward(&[argmax(&l)], pos, &mut st).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    let _ = std::fs::remove_dir_all(&dir);
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
            // Sparse layers: router, TopK, GatherCols and the banked expert GEMVs, all on Metal.
            let (g, w) = gibbs_engine::model::gemma4::tiny_moe_decode_graph().expect("tiny gemma4 MoE graph");
            run_on_gpu("Gemma4 MoE", g, w);
        }
        Family::Hybrid => {
            // Hybrid decoder + MTP drafter, routed experts, image prompt (vision tower), all via
            // the `qwen3_5` spec (`.ninfer` artifacts load the same spec).
            qwen35_on_gpu("qwen3_5", false, false);
            qwen35_on_gpu("qwen3_5_moe", true, false);
            qwen35_on_gpu("qwen3_5_vl", false, true);
        }
        Family::System1 => {
            use gibbs_engine::model::hub::HuggingFaceHub;
            use gibbs_engine::runtime::encoder::system1::JevSystem1Model;
            let mut ts = synth::Ts::new(3);
            ts.w("embeddings.word_embeddings.weight", &[synth::VOCAB, 16]);
            ts.w("embeddings.position_embeddings.weight", &[32, 16]);
            ts.w("embeddings.token_type_embeddings.weight", &[2, 16]);
            ts.ln("embeddings.LayerNorm", 16);
            let p = "encoder.layer.0";
            for s in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"] {
                ts.lin(&format!("{p}.{s}"), 16, 16);
            }
            ts.ln(&format!("{p}.attention.output.LayerNorm"), 16);
            ts.lin(&format!("{p}.intermediate.dense"), 32, 16);
            ts.lin(&format!("{p}.output.dense"), 16, 32);
            ts.ln(&format!("{p}.output.LayerNorm"), 16);
            let cfg = serde_json::json!({"architectures": ["BertModel"], "model_type": "bert", "hidden_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 32,
                "max_position_embeddings": 32, "type_vocab_size": 2, "layer_norm_eps": 1e-5});
            let hub = HuggingFaceHub::with_cache_dir(synth::write_repo("system1", cfg, &ts));
            let m = JevSystem1Model::load(&hub, "t/system1", Device::Metal).expect("System 1 loads on Metal");
            assert_eq!(m.encoder().map(|e| e.device()), Some(Device::Metal));
            for _ in 0..3 {
                assert_eq!(m.embed(synth::TEXT).expect("System 1 runs").len(), 16);
            }
        }
        Family::T5 => encdec_on_gpu("t5", false),
        Family::Seq2Seq => encdec_on_gpu("bart", true),
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
        Family::Idefics3Siglip => {
            let (g, w) = gibbs_engine::model::siglip::tiny::graph();
            run_on_gpu("Idefics3Siglip tower", g, w);
            let (g, w) = GraphLm::new(GraphLmConfig::smollm_1m()).to_decode_graph_embeds();
            run_on_gpu("Idefics3Siglip text", g, w);
            scatter_fixture("Idefics3Siglip");
        }
        Family::Gemma4Media => {
            let (g, w) = gibbs_engine::model::gemma4::tiny_vision_graph();
            run_on_gpu("Gemma4Media vision", g, w);
            scatter_fixture("Gemma4Media");
        }
        Family::Audio => {
            let (g, w) = gibbs_engine::model::gemma4::tiny_audio_wav_graph();
            run_on_gpu("Audio conformer", g, w);
            scatter_fixture("Audio");
        }
        Family::DiffusionDit => {
            let (g, w) = gibbs_engine::model::flux::tiny_dit_graph().expect("tiny FLUX DiT graph");
            run_on_gpu("DiffusionDit", g, w);
        }
        Family::VideoLtx => {
            // The `ltx` and `modelscope_t2v` spec denoisers on tiny checkpoints, Metal programs.
            gibbs_engine::model::diff_fixtures::tiny_video_on_gpu().expect("tiny video denoisers run on Metal");
        }
        // The spec pipeline (CLIP -> UNet2D x2 CFG steps -> VAE): golden `diffa::tiny_sd` asserts its
        // Metal leg enters no CPU compute.
        Family::DiffusionUnet => {}
        Family::VideoModelScope => {
            gibbs_engine::model::diff_fixtures::tiny_video_on_gpu().expect("tiny video denoisers run on Metal");
        }
        Family::Vae => {
            // SD and Qwen-Image (x2) VAEs decode as Metal programs.
            gibbs_engine::model::diff_fixtures::tiny_vae_on_gpu().expect("tiny VAEs run on Metal");
        }
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
