//! The Qwen3.5 Metal programs (decoder with image positions and deepstack, vision tower, MTP head,
//! verify/rewind) against the host forward, on tiny synthetic checkpoints.
//!
//! Tolerance: both sides read the same bf16 weights; the host accumulates f32 in NEON tiles, the
//! GPU in f32 simdgroups, so logits agree to a few 1e-5 relative to their scale: bound 1e-3.

use super::*;
use crate::hardware::{cpu_compute, PlacementMode};
use crate::model::native::MediaInput;

const H: usize = 64;
const V: usize = 96;
const HD: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const INTER: usize = 96;
/// DeltaNet: 1 key head, 2 value heads, 16-wide, conv kernel 4 -> conv dim 64.
const CD: usize = 64;
const IMAGE_TOKEN: u32 = 90;

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

const PRE: &str = r#"{"size":{"shortest_edge":64,"longest_edge":100000},
    "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#;

fn tokenizer() -> Tokenizer {
    Tokenizer::from_tokens((0..V).map(|i| format!("t{i}")).collect())
}

/// Host and Metal models over one checkpoint.
fn pair(vision: bool, mtp: bool) -> (Arc<dyn NativeLm>, Arc<dyn NativeLm>) {
    let (ck, cfg) = checkpoint(vision, mtp);
    let build = |mode: Device| {
        let text = Arc::new(TextModel::load(&ck, text_config(&cfg), Arch::Qwen35).unwrap());
        let vis = load_vision(&ck, &cfg, &text, || Preprocessor::parse(PRE.as_bytes(), "test")).unwrap();
        assemble(tokenizer(), text, vis, PlacementMode::Pin(mode)).unwrap()
    };
    (build(Device::Cpu), build(Device::Metal))
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn png(w: usize, h: usize) -> Vec<u8> {
    let mut img = ImageBuffer::new(w, h);
    for (i, px) in img.data.iter_mut().enumerate() {
        *px = ((i * 37 + (i / 7) * 11) % 251) as u8;
    }
    img.to_png()
}

/// One image, text before and after; six prompt-side forwards then four decode steps.
#[test]
fn image_prompt_program_matches_host() {
    let _g = cpu_compute::allow();
    let (cpu, gpu) = pair(true, false);
    let ids: Vec<u32> = vec![3, 7, IMAGE_TOKEN, 11, 5, 9];
    let media = [MediaInput::Image(png(16, 24))];
    let (ids_c, mut sc) = cpu.prepare(ids.clone(), &media).unwrap();
    let (ids_g, mut sg) = gpu.prepare(ids, &media).unwrap();
    assert_eq!(ids_c, ids_g);
    assert!(ids_c.len() > 20, "image expanded to its 4x6 merged grid");
    let (lc, lg) = (cpu.forward(&ids_c, 0, &mut sc).unwrap(), gpu.forward(&ids_g, 0, &mut sg).unwrap());
    let e = rel(&lc, &lg);
    println!("prefill (image) rel err {e:e}");
    assert!(e < 1e-3, "{e}");
    assert_eq!(argmax(&lc), argmax(&lg));
    let (mut tok, mut pos) = (argmax(&lc), ids_c.len());
    for step in 0..4 {
        let (a, b) = (cpu.forward(&[tok], pos, &mut sc).unwrap(), gpu.forward(&[tok], pos, &mut sg).unwrap());
        let e = rel(&a, &b);
        println!("decode {step} rel err {e:e}");
        assert!(e < 1e-3, "decode {step}: {e}");
        tok = argmax(&a);
        pos += 1;
    }
}

/// The vision tower program alone (merged features and the deepstack feature).
#[test]
fn vision_tower_program_matches_host() {
    let _g = cpu_compute::allow();
    let (ck, cfg) = checkpoint(true, false);
    let pre = Preprocessor::parse(PRE.as_bytes(), "test").unwrap();
    let host = Vision::load(&ck, &cfg["vision_config"]).unwrap();
    let dev = Vision::load(&ck, &cfg["vision_config"])
        .unwrap()
        .on_gpu(PlacementMode::Pin(Device::Metal))
        .unwrap();
    for (w, h) in [(16, 24), (24, 8)] {
        let img = ImageBuffer::decode(&png(w, h)).unwrap();
        let (a, b) = (host.encode(&img, &pre).unwrap(), dev.encode(&img, &pre).unwrap());
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        assert_eq!(a.deepstack.len(), 1);
        let (e, ed) = (rel(&a.embeds, &b.embeds), rel(&a.deepstack[0], &b.deepstack[0]));
        println!("{w}x{h}: embeds rel err {e:e}, deepstack {ed:e}");
        assert!(e < 1e-3 && ed < 1e-3, "{w}x{h}: {e} {ed}");
    }
}

/// Verify four positions, rewind to two, continue: bit-identical to running them one at a time
/// (the Metal-resident recurrent state and conv window come back from the snapshot); rewinding
/// outside the verified span errs; a plain forward drops the checkpoints.
#[test]
fn verify_rewind_is_exact_on_metal() {
    let (_, gpu) = pair(false, false);
    let prompt = vec![4u32, 8, 15, 16, 23];
    let n = prompt.len();
    let (_, mut a) = gpu.prepare(prompt.clone(), &[]).unwrap();
    gpu.forward(&prompt, 0, &mut a).unwrap();
    gpu.forward_all(&[30, 31, 50, 51], n, &mut a).unwrap();
    gpu.truncate(&mut a, n + 2).unwrap();
    let got = gpu.forward_all(&[32, 33], n + 2, &mut a).unwrap();

    let (_, mut b) = gpu.prepare(prompt.clone(), &[]).unwrap();
    gpu.forward(&prompt, 0, &mut b).unwrap();
    gpu.forward(&[30], n, &mut b).unwrap();
    gpu.forward(&[31], n + 1, &mut b).unwrap();
    let want = gpu.forward_all(&[32, 33], n + 2, &mut b).unwrap();
    assert!(got == want, "rewound state diverges from the plain run");

    assert!(gpu.truncate(&mut a, n + 1).is_err(), "before the verified span");
    assert!(gpu.truncate(&mut a, n + 5).is_err(), "past the sequence");
    gpu.forward(&[34], n + 4, &mut a).unwrap();
    assert!(gpu.truncate(&mut a, n + 4).is_err(), "checkpoints dropped");
}

/// MTP head: the Metal program drafts the same tokens from the same logits as the host head,
/// across a catch-up call after a partial accept.
#[test]
fn mtp_drafter_program_matches_host() {
    let _g = cpu_compute::allow();
    let (cpu, gpu) = pair(false, true);
    let (dc, dg) = (cpu.drafter().expect("host MTP"), gpu.drafter().expect("Metal MTP"));
    assert_eq!(dc.tap_layers(), dg.tap_layers());
    let prompt = vec![4u32, 8, 15, 16, 23, 42];
    let (_, mut sc) = cpu.prepare(prompt.clone(), &[]).unwrap();
    let (_, mut sg) = gpu.prepare(prompt.clone(), &[]).unwrap();
    cpu.set_taps(&mut sc, &dc.tap_layers()).unwrap();
    gpu.set_taps(&mut sg, &dg.tap_layers()).unwrap();
    let (mut stc, mut stg) = (dc.new_state(), dg.new_state());
    let n = prompt.len();
    // The target has run all but the last token.
    cpu.forward(&prompt[..n - 1], 0, &mut sc).unwrap();
    gpu.forward(&prompt[..n - 1], 0, &mut sg).unwrap();
    let (hc, hg) = (cpu.take_taps(&mut sc), gpu.take_taps(&mut sg));
    let mut errs = Vec::new();
    let mut draw = |d: &Arc<dyn Drafter>, st: &mut crate::scheduler::spec::DraftState, toks: &[u32], h: &[f32]| {
        let mut seen = Vec::new();
        let out = d
            .propose(st, toks, h, 3, &mut |l: &[f32]| {
                seen.push(l.to_vec());
                argmax(l)
            })
            .unwrap();
        (out, seen)
    };
    let (oc, lc) = draw(&dc, &mut stc, &prompt, &hc);
    let (og, lg) = draw(&dg, &mut stg, &prompt, &hg);
    errs.extend(lc.iter().zip(&lg).map(|(a, b)| rel(a, b)));
    assert_eq!(oc, og);
    // Verify [last, d0, d1, d2], accept two, then ask again: one catch-up row per position run.
    let ver = [prompt[n - 1], oc[0], oc[1], oc[2]];
    cpu.forward_all(&ver, n - 1, &mut sc).unwrap();
    gpu.forward_all(&ver, n - 1, &mut sg).unwrap();
    cpu.truncate(&mut sc, n + 2).unwrap();
    gpu.truncate(&mut sg, n + 2).unwrap();
    let (hc, hg) = (cpu.take_taps(&mut sc), gpu.take_taps(&mut sg));
    // Taps hold all four verified rows; the drafter only wants the accepted three.
    let d = H;
    let mut toks = prompt.clone();
    toks.extend([oc[0], oc[1], 77]);
    let (oc2, lc2) = draw(&dc, &mut stc, &toks, &hc[..3 * d]);
    let (og2, lg2) = draw(&dg, &mut stg, &toks, &hg[..3 * d]);
    errs.extend(lc2.iter().zip(&lg2).map(|(a, b)| rel(a, b)));
    println!("MTP draft logits rel err: {errs:?}");
    assert!(errs.iter().all(|&e| e < 1e-3), "{errs:?}");
    assert_eq!(oc2, og2);
}
