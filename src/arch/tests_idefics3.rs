//! `idefics3.json5` (SigLIP + pixel-shuffle connector + llama text) against the old Idefics3
//! engine (`model::idefics3`, SigLIP host forward / GraphLm text) on the `enc_smolvlm_*` golden
//! fixture shape, CPU and Metal: the media pipeline (`runtime::media` measure / expand /
//! preprocess), the `vision` tower and the `text` instance with `media_rows` scattered in, then
//! three decode steps.

use super::expr::Val;
use super::testing::{compare, Runner, Weights};
use crate::graph::Tensor;
use crate::hardware::Device;
use crate::model::image::ImageBuffer;
use crate::model::native::MediaInput;
use serde_json::json;
use std::path::Path;

const VOCAB: usize = 128;
const IMG: u32 = 100;
const SPECIALS: [&str; 7] = [
    "<image>",
    "<fake_token_around_image>",
    "<global-img>",
    "<row_1_col_1>",
    "<row_1_col_2>",
    "<row_2_col_1>",
    "<row_2_col_2>",
];

fn ln(w: &mut Weights, p: &str, d: usize) {
    w.norm(format!("{p}.weight"), d, 1.0);
    w.s(format!("{p}.bias"), &[d], 0.05);
}

fn lin(w: &mut Weights, p: &str, out: usize, inp: usize) {
    w.w(format!("{p}.weight"), &[out, inp]);
    w.s(format!("{p}.bias"), &[out], 0.05);
}

/// The `enc_smolvlm_*` checkpoint: SigLIP (hidden 16, 2 layers, 16 px tiles, 4 px patches),
/// scale 2 (4 rows per tile), llama (hidden 64, 2 layers, 4 heads, 2 KV heads).
fn checkpoint(dir: &Path) {
    let (vd, vi, patch, th, ti, kv) = (16, 32, 4, 64, 128, 32);
    let cfg = json!({"architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3",
        "image_token_id": IMG, "scale_factor": 2,
        "vision_config": {"hidden_size": vd, "intermediate_size": vi, "num_hidden_layers": 2,
            "num_attention_heads": 2, "image_size": 16, "patch_size": patch,
            "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6},
        "text_config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": th, "intermediate_size": ti, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": VOCAB,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
            "hidden_act": "silu", "tie_word_embeddings": false}});
    let mut w = Weights::new(0xF0);
    let v = "model.vision_model";
    w.s(format!("{v}.embeddings.patch_embedding.weight"), &[vd, 3, patch, patch], 1.0 / ((3 * patch * patch) as f32).sqrt());
    w.s(format!("{v}.embeddings.patch_embedding.bias"), &[vd], 0.05);
    w.s(format!("{v}.embeddings.position_embedding.weight"), &[16, vd], 0.2);
    for l in 0..2 {
        let p = format!("{v}.encoder.layers.{l}");
        ln(&mut w, &format!("{p}.layer_norm1"), vd);
        for s in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(&mut w, &format!("{p}.self_attn.{s}"), vd, vd);
        }
        ln(&mut w, &format!("{p}.layer_norm2"), vd);
        lin(&mut w, &format!("{p}.mlp.fc1"), vi, vd);
        lin(&mut w, &format!("{p}.mlp.fc2"), vd, vi);
    }
    ln(&mut w, &format!("{v}.post_layernorm"), vd);
    w.w("model.connector.modality_projection.proj.weight", &[th, vd * 4]);
    let t = "model.text_model";
    w.s(format!("{t}.embed_tokens.weight"), &[VOCAB, th], 1.0);
    for l in 0..2 {
        let p = format!("{t}.layers.{l}");
        w.norm(format!("{p}.input_layernorm.weight"), th, 1.0);
        w.w(format!("{p}.self_attn.q_proj.weight"), &[th, th]);
        w.w(format!("{p}.self_attn.k_proj.weight"), &[kv, th]);
        w.w(format!("{p}.self_attn.v_proj.weight"), &[kv, th]);
        w.w(format!("{p}.self_attn.o_proj.weight"), &[th, th]);
        w.norm(format!("{p}.post_attention_layernorm.weight"), th, 1.0);
        w.w(format!("{p}.mlp.gate_proj.weight"), &[ti, th]);
        w.w(format!("{p}.mlp.up_proj.weight"), &[ti, th]);
        w.w(format!("{p}.mlp.down_proj.weight"), &[th, ti]);
    }
    w.norm(format!("{t}.norm.weight"), th, 1.0);
    w.w("lm_head.weight", &[VOCAB, th]);
    w.write_checkpoint(dir, &cfg, IMG as usize);
    // The image specials as single added tokens.
    let p = dir.join("tokenizer.json");
    let mut tok: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    for (i, s) in SPECIALS.iter().enumerate() {
        let id = IMG as usize + i;
        tok["model"]["vocab"][s] = json!(id);
        tok["added_tokens"].as_array_mut().unwrap().push(json!({"id": id, "content": s,
            "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
    }
    std::fs::write(&p, tok.to_string()).unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        json!({"do_resize": true, "size": {"longest_edge": 32}, "max_image_size": {"longest_edge": 16},
            "do_image_splitting": true, "resample": 1, "do_rescale": true, "rescale_factor": 1.0 / 255.0,
            "do_normalize": true, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]})
        .to_string(),
    )
    .unwrap();
}

/// Expanded prompt ids, prefill logits and 3 decode-step logits of the spec on `dev`.
fn spec_run(dir: &Path, dev: Device, prompt: &[u32], png: &[u8]) -> (Vec<u32>, Vec<Vec<f32>>, Runner) {
    use crate::runtime::media;
    let mut r = Runner::load(dir, None, dev, 64).unwrap_or_else(|e| panic!("idefics3: {e}"));
    assert_eq!(r.model.spec.family, "idefics3");
    let m = media::resolve(&r.model.spec.runtime["media"], &r.model.env).unwrap();
    let image = &m["image"];
    let grid = media::measure(&image.preprocess, png).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let tokenize = |s: &str| Ok(tok.encode(s, false).map_err(|e| e.to_string())?.get_ids().to_vec());
    let e = media::expand(prompt, &m, &[("image", &grid)], &tokenize).unwrap();
    let item = media::preprocess(&image.preprocess, png).unwrap();
    let vars: Vec<(&str, Val)> = item.grid.vars.iter().map(|&(k, v)| (k, Val::Num(v as f64))).collect();
    let rows = match r.run(&image.component, &vars, vec![Tensor::F32(item.data)]).unwrap().remove("rows") {
        Some(t @ Tensor::F32(_)) => t,
        o => panic!("vision rows: {o:?}"),
    };
    let n = e.ids.len();
    let vars = [("rows", Val::Num(n as f64)), ("head", Val::Str("last".into())), ("media", Val::Num(e.media_index.len() as f64))];
    let ins = vec![
        Tensor::U32(e.ids.clone()),
        Tensor::U32((0..n as u32).collect()),
        rows,
        Tensor::U32(e.media_index.clone()),
    ];
    let mut out = vec![match r.run("text", &vars, ins).unwrap().remove("logits") {
        Some(Tensor::F32(v)) => v,
        o => panic!("logits: {o:?}"),
    }];
    for (i, t) in [20u32, 21, 22].into_iter().enumerate() {
        out.push(r.lm(&[t], n + i, "last", "logits", vec![]));
    }
    (e.ids, out, r)
}

#[test]
fn idefics3_spec_matches_old_engine() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    checkpoint(dir);
    let a = super::testing::artifacts(dir);
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    crate::model::weight_format::set_reencode_override(Some(crate::model::weight_format::ReencodeConfig {
        policy: crate::model::weight_format::ReencodePolicy::BitExact,
        cache_dir: None,
        ..Default::default()
    }));
    let mut devs = vec![Device::Cpu];
    if crate::hardware::available_devices().contains(&Device::Metal) {
        devs.push(Device::Metal);
    }
    let prompt = [1u32, 10, 11, IMG, 12, 13];
    let mut bad = Vec::new();
    // 24x24 -> 2x2 tiles + global; 30x10 -> 1x2 tiles + global.
    for (wd, ht) in [(24usize, 24usize), (30, 10)] {
        let data = (0..wd * ht * 3).map(|i| (i * 37 % 251) as u8).collect();
        let png = ImageBuffer::from_rgb(wd, ht, data).unwrap().to_png();
        for &dev in &devs {
            let old = crate::model::idefics3::load(&a, &cfg, dev).unwrap();
            let (ids, mut st) = old.prepare(prompt.to_vec(), &[MediaInput::Image(png.clone())]).unwrap();
            let mut want = vec![old.forward(&ids, 0, &mut st).unwrap()];
            for (i, t) in [20u32, 21, 22].into_iter().enumerate() {
                want.push(old.forward(&[t], ids.len() + i, &mut st).unwrap());
            }
            let (got_ids, got, r) = spec_run(dir, dev, &prompt, &png);
            assert_eq!(got_ids, ids, "{wd}x{ht} {dev:?}: expanded prompt");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let (rel, cos) = compare(g, w);
                eprintln!("idefics3 {wd}x{ht} {dev:?} step {i}: rel {rel:.2e} cos {cos:.8}");
                if !(rel < 1e-3 && cos > 0.9999) {
                    bad.push(format!("{wd}x{ht} {dev:?} step {i}: rel {rel:.2e} cos {cos:.8}"));
                }
            }
            assert!(r.unconsumed().is_empty(), "unconsumed {:?}", r.unconsumed());
        }
    }
    assert!(bad.is_empty(), "mismatches:\n  {}", bad.join("\n  "));
}

