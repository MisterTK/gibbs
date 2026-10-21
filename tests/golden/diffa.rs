//! Diffusion goldens, group A, through the family specs (`arch/families`) and `runtime::diffusion`:
//! CLIP text tower, SD UNet, SD3 MMDiT, FLUX transformer, Sana (Gemma 2 encoder + linear DiT), SD
//! VAE, DC-AE nets, and the SD / SD3 / Sana / FLUX.1 / Chroma pipelines through the public loader.
//! Every fixture is a diffusers-layout component directory on disk.

use crate::util::*;
use gibbs_engine::hardware::Device;
use gibbs_engine::model::hub::HuggingFaceHub;
use serde_json::{json, Map, Value};
use std::path::Path;

type Specs = Vec<(String, Vec<usize>)>;

fn lin(s: &mut Specs, p: &str, i: usize, o: usize, bias: bool) {
    s.push((format!("{p}.weight"), vec![o, i]));
    if bias {
        s.push((format!("{p}.bias"), vec![o]));
    }
}
fn conv(s: &mut Specs, p: &str, i: usize, o: usize, k: usize) {
    s.push((format!("{p}.weight"), vec![o, i, k, k]));
    s.push((format!("{p}.bias"), vec![o]));
}
fn nrm(s: &mut Specs, p: &str, c: usize) {
    s.push((format!("{p}.weight"), vec![c]));
    s.push((format!("{p}.bias"), vec![c]));
}

/// Writes `specs` with the in-crate `synth` init rule: 1-D `*weight`/`*scale` = 1 ± 0.1 (norm gains), biases
/// and other 1-D tensors ±0.1, everything else uniform ±sqrt(3 / fan_in) (fan_in = all dims but
/// the first, so conv kernels keep unit variance).
fn write_specs(path: &Path, seed: u64, specs: &[(String, Vec<usize>)], dtype: StDtype) {
    let mut w = Weights::new(seed);
    for (n, s) in specs {
        if s.len() == 1 && (n.ends_with("weight") || n.ends_with("scale") || n.ends_with("running_var")) {
            w.norm(n.clone(), s[0], 1.0);
        } else if s.len() == 1 || n.ends_with("bias") {
            w.s(n.clone(), s, 0.1);
        } else {
            let fan: usize = s[1..].iter().product::<usize>().max(1);
            w.s(n.clone(), s, (3.0 / fan as f32).sqrt());
        }
    }
    write_st_as(path, &w.tensors, dtype);
}

/// Runs `f` on `dev` (the spec networks are pinned to it). The Metal leg must not enter any CPU
/// compute entry point (a silent fallback would match by construction).
pub(crate) fn on<T>(dev: Device, f: impl FnOnce() -> T) -> T {
    use gibbs_engine::hardware::cpu_compute::entered;
    let n = entered();
    let out = f();
    if dev == Device::Metal {
        assert_eq!(entered(), n, "Metal leg ran CPU compute");
    }
    out
}

/// Every k-th value so at most 4096 remain.
pub(crate) fn stride(v: &[f32]) -> Vec<f32> {
    let k = v.len().div_ceil(4096).max(1);
    v.iter().step_by(k).copied().collect()
}

fn rand(n: usize, seed: u64) -> Vec<f32> {
    Rng::new(seed).vec(n, 1.0)
}

fn tol(rel: f64, cos: f64) -> Tol {
    Tol { rel, cos }
}

/// `Pin(dev)` placement of a spec network.
pub(crate) fn pin(dev: Device) -> gibbs_engine::hardware::PlacementMode {
    gibbs_engine::hardware::PlacementMode::Pin(dev)
}

// ---------------------------------------------------------------------------------------------
// CLIP
// ---------------------------------------------------------------------------------------------

const CLIP_D: usize = 64;
const CLIP_VOCAB: usize = 49408;

/// A WordLevel stand-in for the CLIP BPE: `!` = 0, `tK` = K (1..256), `<|startoftext|>` = 49406,
/// `<|endoftext|>` = 49407 (the ids the CLIP tower frames prompts with).
fn clip_tokenizer_json() -> Value {
    let mut vocab = Map::new();
    vocab.insert("!".into(), json!(0));
    for i in 1..256 {
        vocab.insert(format!("t{i}"), json!(i));
    }
    let specials = [("<|startoftext|>", 49406), ("<|endoftext|>", 49407)];
    for (s, id) in specials {
        vocab.insert(s.into(), json!(id));
    }
    let added: Vec<Value> = specials
        .iter()
        .map(|(s, id)| {
            json!({"id": id, "content": s, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>"}
    })
}

/// The hub cache the CLIP loader fetches `openai/clip-vit-large-patch14/tokenizer.json` from.
pub(crate) fn clip_hub(dir: &Path) -> HuggingFaceHub {
    let root = dir.join("hub");
    write_json(
        &root.join("models--openai--clip-vit-large-patch14/snapshots/main/tokenizer.json"),
        &clip_tokenizer_json(),
    );
    HuggingFaceHub::with_cache_dir(root)
}

/// diffusers `text_encoder/` (CLIPTextModel[WithProjection], 2 layers, d 64) + `tokenizer/`.
pub(crate) fn write_clip(dir: &Path, te: &str, tok: &str, quick: bool, proj: bool, seed: u64) {
    write_clip_proj(dir, te, tok, quick, if proj { 48 } else { 0 }, seed);
}

/// [`write_clip`] with a `proj`-wide `text_projection` (0: none).
fn write_clip_proj(dir: &Path, te: &str, tok: &str, quick: bool, proj: usize, seed: u64) {
    let d = CLIP_D;
    let mut s: Specs = vec![
        ("text_model.embeddings.token_embedding.weight".into(), vec![CLIP_VOCAB, d]),
        ("text_model.embeddings.position_embedding.weight".into(), vec![77, d]),
    ];
    for l in 0..2 {
        let p = format!("text_model.encoder.layers.{l}");
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(&mut s, &format!("{p}.self_attn.{n}"), d, d, true);
        }
        lin(&mut s, &format!("{p}.mlp.fc1"), d, 2 * d, true);
        lin(&mut s, &format!("{p}.mlp.fc2"), 2 * d, d, true);
        nrm(&mut s, &format!("{p}.layer_norm1"), d);
        nrm(&mut s, &format!("{p}.layer_norm2"), d);
    }
    nrm(&mut s, "text_model.final_layer_norm", d);
    if proj > 0 {
        s.push(("text_projection.weight".into(), vec![proj, d]));
    }
    write_specs(&dir.join(te).join("model.safetensors"), seed, &s, StDtype::F32);
    write_json(
        &dir.join(te).join("config.json"),
        &json!({
            "architectures": [if proj > 0 { "CLIPTextModelWithProjection" } else { "CLIPTextModel" }],
            "model_type": "clip_text_model", "hidden_act": if quick { "quick_gelu" } else { "gelu" },
            "hidden_size": d, "intermediate_size": 2 * d, "num_attention_heads": 1,
            "num_hidden_layers": 2, "max_position_embeddings": 77, "vocab_size": CLIP_VOCAB,
            "layer_norm_eps": 1e-5, "projection_dim": if proj > 0 { proj } else { 48 }
        }),
    );
    let pad = if quick { "<|endoftext|>" } else { "!" };
    let tdir = dir.join(tok);
    write_json(&tdir.join("tokenizer.json"), &clip_tokenizer_json());
    write_json(&tdir.join("special_tokens_map.json"), &json!({"pad_token": pad}));
    write_json(
        &tdir.join("tokenizer_config.json"),
        &json!({"pad_token": pad, "model_max_length": 77}),
    );
}

fn clip_case(case: &str, quick: bool, proj: bool) {
    let dir = fixture_dir(case);
    write_clip(&dir, "text_encoder", "tokenizer", quick, proj, 21);
    // runner: gibbs_engine::runtime::diffusion::Net (`clip_text` spec, network `text`) on
    // `text_encoder/`, prompts framed as `clip.rs` does: [BOS, ids <= 75, EOS] padded to 77.
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net};
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).unwrap();
    let map: Value =
        serde_json::from_slice(&std::fs::read(dir.join("tokenizer/special_tokens_map.json")).unwrap()).unwrap();
    let pad = if map["pad_token"] == "!" { 0 } else { 49407 };
    golden(case, TOL, |dev| {
        let net = Net::open(&dir.join("text_encoder"), "clip_text", "text", pin(dev)).unwrap();
        let mut o = Out::default();
        for (name, prompt) in [("a", "t5 t17 t200 t99"), ("empty", "")] {
            let ids = tok.encode(prompt, false).unwrap().get_ids().to_vec();
            let n = ids.len().min(75);
            let mut framed = vec![pad; 77];
            framed[0] = 49406;
            framed[1..=n].copy_from_slice(&ids[..n]);
            framed[n + 1] = 49407;
            let given = [
                ("token_ids", Tensor::U32(framed)),
                ("positions", Tensor::U32((0..77).collect())),
                ("last_index", Tensor::U32(vec![n as u32 + 1])),
            ];
            let out = on(dev, || net.run(&[("rows", Val::Num(77.0))], &given, &[])).unwrap();
            let get = |k: &str| out.iter().find(|(n, _)| n == k).unwrap().1.as_f32().unwrap().to_vec();
            o.f(format!("{name}.penult"), &stride(&get("penult")))
                .f(format!("{name}.last"), &stride(&get("hidden")))
                .f(format!("{name}.pooled"), &get("pooled"));
        }
        assert_eq!(unconsumed(&net.model, &[&net], &[]), Vec::<String>::new());
        o
    });
}

#[test]
fn clip_quick_gelu() {
    clip_case("diffa_clip_quick_gelu", true, false);
}

#[test]
fn clip_gelu_proj() {
    clip_case("diffa_clip_gelu_proj", false, true);
}

// ---------------------------------------------------------------------------------------------
// SD UNet (diffusers UNet2DConditionModel: 2 cross-attn levels, 1 ResBlock per level)
// ---------------------------------------------------------------------------------------------

fn res(s: &mut Specs, p: &str, i: usize, o: usize, emb: usize) {
    nrm(s, &format!("{p}.norm1"), i);
    conv(s, &format!("{p}.conv1"), i, o, 3);
    lin(s, &format!("{p}.time_emb_proj"), emb, o, true);
    nrm(s, &format!("{p}.norm2"), o);
    conv(s, &format!("{p}.conv2"), o, o, 3);
    if i != o {
        conv(s, &format!("{p}.conv_shortcut"), i, o, 1);
    }
}

fn trans(s: &mut Specs, p: &str, c: usize, ctx: usize) {
    nrm(s, &format!("{p}.norm"), c);
    lin(s, &format!("{p}.proj_in"), c, c, true);
    let b = format!("{p}.transformer_blocks.0");
    for n in ["norm1", "norm2", "norm3"] {
        nrm(s, &format!("{b}.{n}"), c);
    }
    for (a, kv) in [("attn1", c), ("attn2", ctx)] {
        lin(s, &format!("{b}.{a}.to_q"), c, c, false);
        lin(s, &format!("{b}.{a}.to_k"), kv, c, false);
        lin(s, &format!("{b}.{a}.to_v"), kv, c, false);
        lin(s, &format!("{b}.{a}.to_out.0"), c, c, true);
    }
    lin(s, &format!("{b}.ff.net.0.proj"), c, 4 * c, true);
    lin(s, &format!("{b}.ff.net.2"), 2 * c, c, true);
    lin(s, &format!("{p}.proj_out"), c, c, true);
}

/// `unet/` of block_out_channels [32, 64], layers_per_block 1, head dim 16, cross-attention width
/// `ctx`; `add` > 0 adds the SDXL text_time `add_embedding`, `cond` > 0 the LCM `cond_proj`.
fn write_unet(dir: &Path, ctx: usize, add: usize, cond: usize) {
    let (mc, emb) = (32, 128);
    let mut s = Specs::new();
    conv(&mut s, "conv_in", 4, mc, 3);
    lin(&mut s, "time_embedding.linear_1", mc, emb, true);
    lin(&mut s, "time_embedding.linear_2", emb, emb, true);
    if cond > 0 {
        lin(&mut s, "time_embedding.cond_proj", cond, mc, false);
    }
    if add > 0 {
        lin(&mut s, "add_embedding.linear_1", add, emb, true);
        lin(&mut s, "add_embedding.linear_2", emb, emb, true);
    }
    res(&mut s, "down_blocks.0.resnets.0", 32, 32, emb);
    trans(&mut s, "down_blocks.0.attentions.0", 32, ctx);
    s.extend([
        ("down_blocks.0.downsamplers.0.conv.weight".to_string(), vec![32, 32, 3, 3]),
        ("down_blocks.0.downsamplers.0.conv.bias".to_string(), vec![32]),
    ]);
    res(&mut s, "down_blocks.1.resnets.0", 32, 64, emb);
    trans(&mut s, "down_blocks.1.attentions.0", 64, ctx);
    res(&mut s, "mid_block.resnets.0", 64, 64, emb);
    trans(&mut s, "mid_block.attentions.0", 64, ctx);
    res(&mut s, "mid_block.resnets.1", 64, 64, emb);
    // (block, resnet, in, out): skip concatenation widths of the standard diffusers up path.
    for (b, r, i, o) in [(0, 0, 128, 64), (0, 1, 96, 64), (1, 0, 96, 32), (1, 1, 64, 32)] {
        res(&mut s, &format!("up_blocks.{b}.resnets.{r}"), i, o, emb);
        trans(&mut s, &format!("up_blocks.{b}.attentions.{r}"), o, ctx);
    }
    conv(&mut s, "up_blocks.0.upsamplers.0.conv", 64, 64, 3);
    nrm(&mut s, "conv_norm_out", 32);
    conv(&mut s, "conv_out", 32, 4, 3);
    let u = dir.join("unet");
    write_specs(&u.join("diffusion_pytorch_model.safetensors"), 11, &s, StDtype::F32);
    let mut cfg = json!({
        "_class_name": "UNet2DConditionModel", "sample_size": 4, "in_channels": 4,
        "out_channels": 4, "block_out_channels": [32, 64], "layers_per_block": 1,
        "attention_head_dim": [2, 4], "cross_attention_dim": ctx, "norm_num_groups": 32,
        "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        "mid_block_type": "UNetMidBlock2DCrossAttn"
    });
    if add > 0 {
        cfg["addition_embed_type"] = json!("text_time");
        cfg["projection_class_embeddings_input_dim"] = json!(add);
    }
    if cond > 0 {
        cfg["time_cond_proj_dim"] = json!(cond);
    }
    write_json(&u.join("config.json"), &cfg);
}

fn unet_case(case: &str, add: usize, cond: usize) {
    let (ctx, h, w, ctx_len) = (48, 8, 12, 5);
    let dir = fixture_dir(case);
    write_unet(&dir, ctx, add, cond);
    let u = dir.join("unet");
    let (x, c) = (rand(4 * h * w, 1), rand(ctx_len * ctx, 2));
    let (a, tc) = (rand(add, 3), rand(cond, 4));
    // runner: gibbs_engine::runtime::diffusion::Net (`sd` spec, network `denoiser`) on `unet/`;
    // the raw add vector goes in as `pooled` (no addition_time_embed_dim), the LCM `t_cond`
    // replaces its generated guidance embedding.
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net};
    let vars = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64)), ("nt", Val::Num(ctx_len as f64))];
    let mut given = vec![("x", Tensor::F32(x.clone())), ("context", Tensor::F32(c.clone()))];
    if add > 0 {
        given.push(("pooled", Tensor::F32(a.clone())));
    }
    if cond > 0 {
        given.push(("t_cond", Tensor::F32(tc.clone())));
    }
    golden(case, TOL, |dev| {
        let net = Net::open(&u, "sd", "denoiser", pin(dev)).unwrap();
        let mut o = Out::default();
        for t in [301.0f32, 5.0] {
            let y = on(dev, || net.run_one(&vars, &given, &[("timestep", t as f64)], None)).unwrap();
            o.f(format!("t{t}"), &y);
        }
        assert_eq!(unconsumed(&net.model, &[&net], &[]), Vec::<String>::new());
        o
    });
}

#[test]
fn unet_sd() {
    unet_case("diffa_unet_sd", 0, 0);
}

#[test]
fn unet_xl_lcm() {
    unet_case("diffa_unet_xl_lcm", 24, 16);
}

// ---------------------------------------------------------------------------------------------
// SD3 MMDiT (SD3Transformer2DModel)
// ---------------------------------------------------------------------------------------------

/// diffusers `transformer/` (SD3Transformer2DModel: d 64, 1 head, 3 blocks, the last
/// context-pre-only; `dual`: block 0 has SD3.5 `attn2`; `qk_norm`: RMSNorm q/k gains).
fn write_sd3(dir: &Path, joint: usize, pooled_dim: usize, dual: bool, qk_norm: bool, seed: u64) {
    let (d, pos_max, layers) = (64usize, 8usize, 3usize);
    let mut s = Specs::new();
    conv(&mut s, "pos_embed.proj", 4, d, 2);
    s.push(("pos_embed.pos_embed".into(), vec![1, pos_max * pos_max, d]));
    lin(&mut s, "time_text_embed.timestep_embedder.linear_1", 256, d, true);
    lin(&mut s, "time_text_embed.timestep_embedder.linear_2", d, d, true);
    lin(&mut s, "time_text_embed.text_embedder.linear_1", pooled_dim, d, true);
    lin(&mut s, "time_text_embed.text_embedder.linear_2", d, d, true);
    lin(&mut s, "context_embedder", joint, d, true);
    for i in 0..layers {
        let p = format!("transformer_blocks.{i}");
        let last = i + 1 == layers;
        let a2 = dual && i == 0;
        lin(&mut s, &format!("{p}.norm1.linear"), d, if a2 { 9 * d } else { 6 * d }, true);
        lin(&mut s, &format!("{p}.norm1_context.linear"), d, if last { 2 * d } else { 6 * d }, true);
        let mut proj = |a: &str, names: [&str; 6], out: bool| {
            for n in &names[..3] {
                lin(&mut s, &format!("{p}.{a}.{n}"), d, d, true);
            }
            if out {
                lin(&mut s, &format!("{p}.{a}.{}", names[3]), d, d, true);
            }
            if qk_norm {
                for n in &names[4..] {
                    s.push((format!("{p}.{a}.{n}.weight"), vec![64]));
                }
            }
        };
        proj("attn", ["to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k"], true);
        let ctx = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "norm_added_q", "norm_added_k"];
        proj("attn", ctx, !last);
        if a2 {
            proj("attn2", ["to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k"], true);
        }
        let ffs: &[&str] = if last { &["ff"] } else { &["ff", "ff_context"] };
        for f in ffs {
            lin(&mut s, &format!("{p}.{f}.net.0.proj"), d, 4 * d, true);
            lin(&mut s, &format!("{p}.{f}.net.2"), 4 * d, d, true);
        }
    }
    lin(&mut s, "norm_out.linear", d, 2 * d, true);
    lin(&mut s, "proj_out", d, 4 * 2 * 2, true);
    let dir = dir.join("transformer");
    write_specs(&dir.join("diffusion_pytorch_model.safetensors"), seed, &s, StDtype::BF16);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "SD3Transformer2DModel", "sample_size": 4, "patch_size": 2,
            "in_channels": 4, "out_channels": 4, "num_layers": layers, "attention_head_dim": 64,
            "num_attention_heads": 1, "joint_attention_dim": joint, "caption_projection_dim": d,
            "pooled_projection_dim": pooled_dim, "pos_embed_max_size": pos_max,
            "dual_attention_layers": if dual { json!([0]) } else { json!([]) },
            "qk_norm": if qk_norm { json!("rms_norm") } else { Value::Null }
        }),
    );
}

fn sd3_case(case: &str, dual: bool, qk_norm: bool, seed: u64) {
    let (joint, pooled_dim) = (48usize, 40usize);
    let root = fixture_dir(case);
    write_sd3(&root, joint, pooled_dim, dual, qk_norm, seed);
    let dir = root.join("transformer");
    // 12x10 latents -> 6x5 patches cropped from the 8x8 table; 12 text tokens.
    let (h, w, n_ctx) = (12usize, 10usize, 12usize);
    let (lat, ctx, pooled) = (rand(4 * h * w, 11), rand(n_ctx * joint, 12), rand(pooled_dim, 13));
    // runner: gibbs_engine::runtime::diffusion::Net (`sd3` spec, network `denoiser`)
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net, Runtime};
    // The root pipeline's `runtime` block parses (the integrator runs it end to end).
    let sig = gibbs_engine::arch::Signals::default();
    gibbs_engine::arch::registry().unwrap().spec("sd3", &sig).unwrap().runtime_as::<Runtime>().unwrap();
    let vars = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64)), ("nt", Val::Num(n_ctx as f64))];
    let given = [
        ("x", Tensor::F32(lat.clone())),
        ("context", Tensor::F32(ctx.clone())),
        ("pooled", Tensor::F32(pooled.clone())),
    ];
    golden(case, TOL, |dev| {
        let net = Net::open(&dir, "sd3", "denoiser", pin(dev)).unwrap();
        let mut o = Out::default();
        for t in [700.0f32, 120.0] {
            let y = on(dev, || net.run_one(&vars, &given, &[("timestep", t as f64)], None)).unwrap();
            o.f(format!("t{t}"), &y);
        }
        assert_eq!(unconsumed(&net.model, &[&net], &[]), Vec::<String>::new());
        o
    });
}

#[test]
fn sd3_plain() {
    sd3_case("diffa_sd3_plain", false, false, 1);
}

#[test]
fn sd3_dual_qk() {
    sd3_case("diffa_sd3_dual_qk", true, true, 2);
}

#[test]
fn sd3_dual() {
    sd3_case("diffa_sd3_dual", true, false, 3);
}

#[test]
fn sd3_qk() {
    sd3_case("diffa_sd3_qk", false, true, 4);
}

/// The whole `StableDiffusion3Pipeline`: CLIP-L (quick_gelu, EOS pad) + CLIP-G (gelu, `!` pad),
/// both with d_model-wide projections (pooled = d_L + d_G, as SD3's checks require), the
/// dual/qk-norm MMDiT, the 4-channel 8x VAE, 2 flow-Euler steps with CFG; `t5`: a 128-wide T5
/// encoder as `text_encoder_3` (else its context block is zeros).
fn sd3_pipeline_case(case: &str, t5: bool) {
    use gibbs_engine::runtime::diffusion::{load_image, ImageRequest};
    let dir = fixture_dir(case);
    write_json(
        &dir.join("model_index.json"),
        &json!({
            "_class_name": "StableDiffusion3Pipeline",
            "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "CLIPTokenizer"],
            "transformer": ["diffusers", "SD3Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]
        }),
    );
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0}),
    );
    write_clip_proj(&dir, "text_encoder", "tokenizer", true, CLIP_D, 21);
    write_clip_proj(&dir, "text_encoder_2", "tokenizer_2", false, CLIP_D, 22);
    if t5 {
        write_t5(&dir, "text_encoder_3", "tokenizer_3", 2 * CLIP_D, 23);
    }
    write_sd3(&dir, 2 * CLIP_D, 2 * CLIP_D, true, true, 5);
    write_sd_vae(&dir, &[32, 32, 32, 32]);
    let hub = clip_hub(&dir);
    let art = artifacts(&dir, "golden/tiny-sd3", "none");
    // 32x48 -> 4x6 latents -> 2x3 patches cropped from the 8x8 table.
    let req = ImageRequest {
        prompt: "t5 t17 t200".into(),
        width: Some(32),
        height: Some(48),
        steps: Some(2),
        guidance: Some(5.0),
        seed: 5,
        ..Default::default()
    };
    golden(case, TOL, |dev| {
        // runner: runtime::diffusion::load_image (the public loader), pinned to `dev`
        let p = load_image(&art, &hub, pin(dev)).expect("image repo").unwrap();
        let img = on(dev, || p.text_to_image(&req)).unwrap();
        assert_eq!((img.width, img.height, img.channels), (32, 48, 3));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}

#[test]
fn sd3_pipeline() {
    sd3_pipeline_case("diffa_sd3_pipeline", false);
}

#[test]
fn sd3_t5_pipeline() {
    sd3_pipeline_case("diffa_sd3_t5_pipeline", true);
}

/// diffusers `T5EncoderModel` at `d` (2 layers, gated-gelu, 2 heads of 16, d_ff 64, 8 buckets,
/// bf16) as `te/`, with the [`write_tokenizer`] WordLevel vocabulary (64 ids) as `tok/`.
fn write_t5(dir: &Path, te: &str, tok: &str, d: usize, seed: u64) {
    let (inner, ff, vocab) = (32usize, 64usize, 64usize);
    let mut s: Specs = vec![("shared.weight".into(), vec![vocab, d])];
    for i in 0..2 {
        let b = format!("encoder.block.{i}.layer");
        s.push((format!("{b}.0.layer_norm.weight"), vec![d]));
        for p in ["q", "k", "v"] {
            s.push((format!("{b}.0.SelfAttention.{p}.weight"), vec![inner, d]));
        }
        s.push((format!("{b}.0.SelfAttention.o.weight"), vec![d, inner]));
        if i == 0 {
            s.push((format!("{b}.0.SelfAttention.relative_attention_bias.weight"), vec![8, 2]));
        }
        s.push((format!("{b}.1.layer_norm.weight"), vec![d]));
        s.push((format!("{b}.1.DenseReluDense.wi_0.weight"), vec![ff, d]));
        s.push((format!("{b}.1.DenseReluDense.wi_1.weight"), vec![ff, d]));
        s.push((format!("{b}.1.DenseReluDense.wo.weight"), vec![d, ff]));
    }
    s.push(("encoder.final_layer_norm.weight".into(), vec![d]));
    let te = dir.join(te);
    write_specs(&te.join("model.safetensors"), seed, &s, StDtype::BF16);
    write_json(
        &te.join("config.json"),
        &json!({
            "architectures": ["T5EncoderModel"], "model_type": "t5", "vocab_size": vocab, "d_model": d,
            "d_kv": 16, "num_heads": 2, "d_ff": ff, "num_layers": 2, "relative_attention_num_buckets": 8,
            "relative_attention_max_distance": 16, "layer_norm_epsilon": 1e-6,
            "feed_forward_proj": "gated-gelu", "eos_token_id": 1, "pad_token_id": 0
        }),
    );
    write_tokenizer(&dir.join(tok), vocab);
}

// ---------------------------------------------------------------------------------------------
// FLUX transformer (diffusers names): FLUX.1 (guidance-distilled), Chroma, FLUX.2
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Flux {
    One,
    Chroma,
    Two,
}

/// FLUX.1 / Chroma transformers (diffusers names) at d 128 (one 128-wide head: the engine's fixed
/// FLUX head dim), 2 double and 2 single blocks, 16-wide packed latents, 24-wide text,
/// `pooled`-wide pooled text.
fn write_flux(dir: &Path, kind: Flux, seed: u64, pooled: usize) {
    if kind == Flux::Two {
        return write_flux2(dir, seed, 24);
    }
    let d = 128;
    let chroma = kind == Flux::Chroma;
    let mut s = Specs::new();
    lin(&mut s, "x_embedder", 16, d, true);
    lin(&mut s, "context_embedder", 24, d, true);
    let embed = |s: &mut Specs, p: &str, i: usize| {
        lin(s, &format!("{p}.linear_1"), i, d, true);
        lin(s, &format!("{p}.linear_2"), d, d, true);
    };
    if kind == Flux::One {
        embed(&mut s, "time_text_embed.timestep_embedder", 256);
        embed(&mut s, "time_text_embed.text_embedder", pooled);
        embed(&mut s, "time_text_embed.guidance_embedder", 256);
    } else {
        let p = "distilled_guidance_layer";
        lin(&mut s, &format!("{p}.in_proj"), 64, d, true);
        for i in 0..2 {
            s.push((format!("{p}.norms.{i}.weight"), vec![d]));
            embed(&mut s, &format!("{p}.layers.{i}"), d);
        }
        lin(&mut s, &format!("{p}.out_proj"), d, d, true);
    }
    let gain = |s: &mut Specs, n: String| s.push((n, vec![128]));
    for i in 0..2 {
        let t = format!("transformer_blocks.{i}");
        if kind == Flux::One {
            lin(&mut s, &format!("{t}.norm1.linear"), d, 6 * d, true);
            lin(&mut s, &format!("{t}.norm1_context.linear"), d, 6 * d, true);
        }
        for n in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out"] {
            lin(&mut s, &format!("{t}.attn.{n}"), d, d, true);
        }
        for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
            gain(&mut s, format!("{t}.attn.{n}.weight"));
        }
        for f in ["ff", "ff_context"] {
            lin(&mut s, &format!("{t}.{f}.net.0.proj"), d, 4 * d, true);
            lin(&mut s, &format!("{t}.{f}.net.2"), 4 * d, d, true);
        }
    }
    for i in 0..2 {
        let t = format!("single_transformer_blocks.{i}");
        if kind == Flux::One {
            lin(&mut s, &format!("{t}.norm.linear"), d, 3 * d, true);
        }
        for n in ["attn.to_q", "attn.to_k", "attn.to_v"] {
            lin(&mut s, &format!("{t}.{n}"), d, d, true);
        }
        lin(&mut s, &format!("{t}.proj_mlp"), d, 4 * d, true);
        lin(&mut s, &format!("{t}.proj_out"), 5 * d, d, true);
        gain(&mut s, format!("{t}.attn.norm_q.weight"));
        gain(&mut s, format!("{t}.attn.norm_k.weight"));
    }
    if !chroma {
        lin(&mut s, "norm_out.linear", d, 2 * d, true);
    }
    lin(&mut s, "proj_out", d, 16, true);
    let tr = dir.join("transformer");
    write_specs(&tr.join("diffusion_pytorch_model.safetensors"), seed, &s, StDtype::BF16);
    let class = if chroma { "ChromaTransformer2DModel" } else { "FluxTransformer2DModel" };
    write_json(
        &tr.join("config.json"),
        &json!({
            "_class_name": class, "patch_size": 1, "in_channels": 16, "num_layers": 2,
            "num_single_layers": 2, "attention_head_dim": 128, "num_attention_heads": 1,
            "joint_attention_dim": 24, "pooled_projection_dim": pooled, "guidance_embeds": !chroma,
            "axes_dims_rope": [16, 56, 56]
        }),
    );
}

/// FLUX.2 in BFL (single-file) names: the current loader detects diffusers FLUX.2 single blocks
/// only via `attn.to_q`, which the fused diffusers `to_qkv_mlp_proj` layout lacks. Shared
/// modulations, SwiGLU feed-forwards (double 192, single 160 hidden), fused qkv/linear1,
/// `joint`-wide text.
fn write_flux2(dir: &Path, seed: u64, joint: usize) {
    let d = 128;
    let mut s = Specs::new();
    lin(&mut s, "img_in", 16, d, true);
    lin(&mut s, "txt_in", joint, d, true);
    lin(&mut s, "time_in.in_layer", 256, d, true);
    lin(&mut s, "time_in.out_layer", d, d, true);
    lin(&mut s, "double_stream_modulation_img.linear", d, 6 * d, true);
    lin(&mut s, "double_stream_modulation_txt.linear", d, 6 * d, true);
    lin(&mut s, "single_stream_modulation.linear", d, 3 * d, true);
    for i in 0..2 {
        for st in ["img", "txt"] {
            let b = format!("double_blocks.{i}.{st}");
            lin(&mut s, &format!("{b}_attn.qkv"), d, 3 * d, true);
            s.push((format!("{b}_attn.norm.query_norm.scale"), vec![128]));
            s.push((format!("{b}_attn.norm.key_norm.scale"), vec![128]));
            lin(&mut s, &format!("{b}_attn.proj"), d, d, true);
            lin(&mut s, &format!("{b}_mlp.0"), d, 2 * 192, true);
            lin(&mut s, &format!("{b}_mlp.2"), 192, d, true);
        }
    }
    for i in 0..2 {
        let b = format!("single_blocks.{i}");
        lin(&mut s, &format!("{b}.linear1"), d, 3 * d + 2 * 160, true);
        lin(&mut s, &format!("{b}.linear2"), d + 160, d, true);
        s.push((format!("{b}.norm.query_norm.scale"), vec![128]));
        s.push((format!("{b}.norm.key_norm.scale"), vec![128]));
    }
    lin(&mut s, "final_layer.adaLN_modulation.1", d, 2 * d, true);
    lin(&mut s, "final_layer.linear", d, 16, true);
    let tr = dir.join("transformer");
    write_specs(&tr.join("diffusion_pytorch_model.safetensors"), seed, &s, StDtype::BF16);
    write_json(
        &tr.join("config.json"),
        &json!({
            "_class_name": "Flux2Transformer2DModel", "patch_size": 1, "in_channels": 16,
            "num_layers": 2, "num_single_layers": 2, "attention_head_dim": 128,
            "num_attention_heads": 1, "joint_attention_dim": joint, "axes_dims_rope": [32, 32, 32, 32]
        }),
    );
}

fn flux_case(case: &str, kind: Flux, seed: u64) {
    let dir = fixture_dir(case);
    write_flux(&dir, kind, seed, 32);
    // runner: gibbs_engine::runtime::diffusion::Net (`flux` spec, network `denoiser`) on the
    // packed 4x4 grid of an 8x8 latent, 8 text rows; `guided: false` drops the guidance embedding.
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net};
    let (img, ctx, pooled) = (rand(16 * 16, 11), rand(8 * 24, 12), rand(32, 13));
    let mut given = vec![("x", Tensor::F32(img)), ("context", Tensor::F32(ctx))];
    if kind == Flux::One {
        given.push(("pooled", Tensor::F32(pooled)));
    }
    golden(case, tol(2e-3, 0.99999), |dev| {
        let net = Net::open(&dir.join("transformer"), "flux", "denoiser", pin(dev)).unwrap();
        let mut o = Out::default();
        for (t, g) in [(0.7f32, Some(3.5f32)), (0.2, None)] {
            let vars = [
                ("h", Val::Num(8.0)),
                ("w", Val::Num(8.0)),
                ("nt", Val::Num(8.0)),
                ("guided", Val::Bool(g.is_some())),
            ];
            let step = [("sigma", t as f64), ("guidance", g.unwrap_or(0.0) as f64)];
            let y = on(dev, || net.run_one(&vars, &given, &step, None)).unwrap();
            o.f(format!("t{t}"), &y);
        }
        assert_eq!(unconsumed(&net.model, &[&net], &[]), Vec::<String>::new());
        o
    });
}

#[test]
fn flux1() {
    flux_case("diffa_flux1", Flux::One, 1);
}

#[test]
fn flux_chroma() {
    flux_case("diffa_flux_chroma", Flux::Chroma, 2);
}

#[test]
fn flux2() {
    flux_case("diffa_flux2", Flux::Two, 3);
}

/// The whole `FluxPipeline` (FLUX.1 dev: CLIP-L pooled + 24-wide T5 padded to 512, guidance
/// embedded) or `ChromaPipeline` (T5 only, valid + 1 rows, true CFG): the [`write_flux`]
/// transformer, the 4-channel 8x VAE, 2 dynamic-shift flow-Euler steps.
fn flux_pipeline_case(case: &str, kind: Flux) {
    use gibbs_engine::runtime::diffusion::{load_image, ImageRequest};
    let dir = fixture_dir(case);
    let class = if kind == Flux::Chroma { "ChromaPipeline" } else { "FluxPipeline" };
    write_json(&dir.join("model_index.json"), &json!({"_class_name": class}));
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000,
                "shift": 3.0, "use_dynamic_shifting": true, "base_shift": 0.5, "max_shift": 1.15,
                "base_image_seq_len": 256, "max_image_seq_len": 4096}),
    );
    if kind == Flux::Chroma {
        write_t5(&dir, "text_encoder", "tokenizer", 24, 31);
    } else {
        write_clip(&dir, "text_encoder", "tokenizer", true, false, 21);
        write_t5(&dir, "text_encoder_2", "tokenizer_2", 24, 31);
    }
    write_flux(&dir, kind, 4, CLIP_D);
    write_sd_vae(&dir, &[32, 32, 32, 32]);
    let hub = clip_hub(&dir);
    let art = artifacts(&dir, "golden/tiny-flux", "none");
    // 32x48 -> 4x6 latents -> 2x3 packed tokens.
    let req = ImageRequest {
        prompt: "t5 t17 t30".into(),
        width: Some(32),
        height: Some(48),
        steps: Some(2),
        guidance: Some(if kind == Flux::Chroma { 4.0 } else { 3.5 }),
        seed: 7,
        ..Default::default()
    };
    golden(case, tol(2e-3, 0.99999), |dev| {
        // runner: runtime::diffusion::load_image (the public loader), pinned to `dev`
        let p = load_image(&art, &hub, pin(dev)).expect("image repo").unwrap();
        let img = on(dev, || p.text_to_image(&req)).unwrap();
        assert_eq!((img.width, img.height, img.channels), (32, 48, 3));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}

#[test]
fn flux1_pipeline() {
    flux_pipeline_case("diffa_flux1_pipeline", Flux::One);
}

#[test]
fn chroma_pipeline() {
    flux_pipeline_case("diffa_chroma_pipeline", Flux::Chroma);
}

/// The whole `Flux2KleinPipeline` (undistilled: true CFG on the empty negative prompt): a 28-layer
/// Qwen3 text encoder (64 wide; layers 8 / 17 / 26 concatenated = the transformer's 192-wide
/// text) on the chat-templated prompt padded with `<|endoftext|>` to 512, the [`write_flux2`]
/// transformer, the 4-channel 8x VAE with BatchNorm stats, 2 empirical-mu flow-Euler steps.
#[test]
fn klein_pipeline() {
    use crate::text_enc::{text_config, weights, H, V};
    use gibbs_engine::runtime::diffusion::{load_image, ImageRequest};
    let dir = fixture_dir("diffa_klein_pipeline");
    write_json(&dir.join("model_index.json"), &json!({"_class_name": "Flux2KleinPipeline", "is_distilled": false}));
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000,
                "shift": 3.0, "use_dynamic_shifting": true, "base_shift": 0.5, "max_shift": 1.15,
                "base_image_seq_len": 256, "max_image_seq_len": 4096}),
    );
    let te = dir.join("text_encoder");
    let mut cfg = text_config("qwen3", 28);
    cfg["architectures"] = json!(["Qwen3ForCausalLM"]);
    cfg["max_position_embeddings"] = json!(1024);
    write_json(&te.join("config.json"), &cfg);
    write_st_as(&te.join("model.safetensors"), &weights(41, "model", true, 28).tensors, StDtype::BF16);
    // WordLevel `t4 .. t{V-4}` + the Qwen chat specials; `<|endoftext|>` pads.
    let specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];
    let mut vocab = Map::new();
    vocab.insert("<unk>".into(), json!(0));
    for i in 4..V - 3 {
        vocab.insert(format!("t{i}"), json!(i));
    }
    let added: Vec<Value> = specials
        .iter()
        .enumerate()
        .map(|(i, s)| {
            vocab.insert(s.to_string(), json!(V - 3 + i));
            json!({"id": V - 3 + i, "content": s, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    let tok = dir.join("tokenizer");
    write_json(
        &tok.join("tokenizer.json"),
        &json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
                "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
                "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}}),
    );
    write_json(
        &tok.join("tokenizer_config.json"),
        &json!({"chat_template": "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"}),
    );
    write_flux2(&dir, 5, 3 * H);
    write_vae(&dir, &[32, 32, 32, 32], true);
    let hub = clip_hub(&dir);
    let art = artifacts(&dir, "golden/tiny-klein", "none");
    let req = ImageRequest {
        prompt: "t5 t17 t30".into(),
        width: Some(32),
        height: Some(48),
        steps: Some(2),
        guidance: Some(4.0),
        seed: 7,
        ..Default::default()
    };
    golden("diffa_klein_pipeline", tol(2e-3, 0.99999), |dev| {
        // runner: runtime::diffusion::load_image (the public loader), pinned to `dev`
        let p = load_image(&art, &hub, pin(dev)).expect("image repo").unwrap();
        let img = on(dev, || p.text_to_image(&req)).unwrap();
        assert_eq!((img.width, img.height, img.channels), (32, 48, 3));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}

// ---------------------------------------------------------------------------------------------
// Sana: Gemma 2 text encoder + linear DiT
// ---------------------------------------------------------------------------------------------

fn write_gemma2(dir: &Path) {
    let (d, heads, kv, hd, inter, vocab) = (64usize, 2usize, 1usize, 32usize, 128usize, 256usize);
    let mut w = Weights::new(5);
    w.s("model.embed_tokens.weight", &[vocab, d], (3.0 / d as f32).sqrt());
    for i in 0..2 {
        let p = format!("model.layers.{i}");
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            // Stored as `w` in `x * (1 + w)`.
            w.s(format!("{p}.{n}.weight"), &[d], 0.1);
        }
        for (n, o, i) in [
            ("self_attn.q_proj", heads * hd, d),
            ("self_attn.k_proj", kv * hd, d),
            ("self_attn.v_proj", kv * hd, d),
            ("self_attn.o_proj", d, heads * hd),
            ("mlp.gate_proj", inter, d),
            ("mlp.up_proj", inter, d),
            ("mlp.down_proj", d, inter),
        ] {
            w.s(format!("{p}.{n}.weight"), &[o, i], (3.0 / i as f32).sqrt());
        }
    }
    w.s("model.norm.weight", &[d], 0.1);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    write_json(
        &dir.join("config.json"),
        &json!({
            "architectures": ["Gemma2Model"], "model_type": "gemma2", "hidden_size": d,
            "intermediate_size": inter, "num_attention_heads": heads, "num_key_value_heads": kv,
            "head_dim": hd, "num_hidden_layers": 2, "vocab_size": vocab, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": hd, "hidden_activation": "gelu_pytorch_tanh"
        }),
    );
}

const SANA_D: usize = 32;
const SANA_C: usize = 8;
const SANA_CAP: usize = 64;

fn write_sana_dit(dir: &Path) {
    let (d, c) = (SANA_D, SANA_C);
    let mut s = Specs::new();
    conv(&mut s, "patch_embed.proj", c, d, 1);
    lin(&mut s, "time_embed.emb.timestep_embedder.linear_1", 256, d, true);
    lin(&mut s, "time_embed.emb.timestep_embedder.linear_2", d, d, true);
    lin(&mut s, "time_embed.linear", d, 6 * d, true);
    lin(&mut s, "caption_projection.linear_1", SANA_CAP, d, true);
    lin(&mut s, "caption_projection.linear_2", d, d, true);
    s.push(("caption_norm.weight".into(), vec![d]));
    for i in 0..2 {
        let p = format!("transformer_blocks.{i}");
        s.push((format!("{p}.scale_shift_table"), vec![6, d]));
        for a in ["attn1", "attn2"] {
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                lin(&mut s, &format!("{p}.{a}.{n}"), d, d, true);
            }
        }
        conv(&mut s, &format!("{p}.ff.conv_inverted"), d, 2 * d, 1);
        s.push((format!("{p}.ff.conv_depth.weight"), vec![2 * d, 1, 3, 3]));
        s.push((format!("{p}.ff.conv_depth.bias"), vec![2 * d]));
        s.push((format!("{p}.ff.conv_point.weight"), vec![d, d, 1, 1]));
    }
    s.push(("scale_shift_table".into(), vec![2, d]));
    lin(&mut s, "proj_out", d, c, true);
    write_specs(&dir.join("diffusion_pytorch_model.safetensors"), 9, &s, StDtype::BF16);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "SanaTransformer2DModel", "patch_size": 1, "in_channels": c,
            "out_channels": c, "num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 16,
            "num_cross_attention_heads": 4, "cross_attention_head_dim": 8,
            "caption_channels": SANA_CAP, "mlp_ratio": 2.0, "sample_size": 8,
            "guidance_embeds": false, "qk_norm": null, "interpolation_scale": null
        }),
    );
}

#[test]
fn sana() {
    let case = "diffa_sana";
    let dir = fixture_dir(case);
    write_gemma2(&dir.join("text_encoder"));
    write_sana_dit(&dir.join("transformer"));
    let ids = [2u32, 17, 33, 200, 5, 9, 255];
    // runner: gibbs_engine::runtime::diffusion::Net (`gemma2` spec network `text` on
    // `text_encoder/`, `sana` spec network `denoiser` on `transformer/`)
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net};
    golden(case, TOL, |dev| {
        let te = Net::open(&dir.join("text_encoder"), "gemma2", "text", pin(dev)).unwrap();
        let dit = Net::open(&dir.join("transformer"), "sana", "denoiser", pin(dev)).unwrap();
        let mut o = Out::default();
        let given = [
            ("token_ids", Tensor::U32(ids.to_vec())),
            ("positions", Tensor::U32((0..ids.len() as u32).collect())),
        ];
        let rows = [("rows", Val::Num(ids.len() as f64))];
        o.f("gemma2.hidden", &on(dev, || te.run_one(&rows, &given, &[], Some("hidden"))).unwrap());
        for (h, w, n_cap, t) in [(4usize, 6usize, 7usize, 731.0f32), (5, 7, 21, 40.0)] {
            let vars = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64)), ("nt", Val::Num(n_cap as f64))];
            let given = [
                ("x", Tensor::F32(rand(SANA_C * h * w, 11))),
                ("context", Tensor::F32(rand(n_cap * SANA_CAP, 12))),
            ];
            let y = on(dev, || dit.run_one(&vars, &given, &[("timestep", t as f64)], None)).unwrap();
            o.f(format!("dit.{h}x{w}.t{t}"), &y);
        }
        assert_eq!(unconsumed(&te.model, &[&te], &[]), Vec::<String>::new());
        assert_eq!(unconsumed(&dit.model, &[&dit], &[]), Vec::<String>::new());
        o
    });
}

// ---------------------------------------------------------------------------------------------
// VAEs
// ---------------------------------------------------------------------------------------------

fn resnet(s: &mut Specs, p: &str, i: usize, o: usize) {
    nrm(s, &format!("{p}.norm1"), i);
    conv(s, &format!("{p}.conv1"), i, o, 3);
    nrm(s, &format!("{p}.norm2"), o);
    conv(s, &format!("{p}.conv2"), o, o, 3);
    if i != o {
        conv(s, &format!("{p}.conv_shortcut"), i, o, 1);
    }
}

fn vae_mid(s: &mut Specs, p: &str, c: usize) {
    resnet(s, &format!("{p}.mid_block.resnets.0"), c, c);
    resnet(s, &format!("{p}.mid_block.resnets.1"), c, c);
    nrm(s, &format!("{p}.mid_block.attentions.0.group_norm"), c);
    for n in ["to_q", "to_k", "to_v", "to_out.0"] {
        lin(s, &format!("{p}.mid_block.attentions.0.{n}"), c, c, true);
    }
}

/// diffusers AutoencoderKL `vae/` with `chans` block_out_channels, 4 latent channels, 1 encoder /
/// 2 decoder ResBlocks per level, a 2x down/upsample between levels.
pub(crate) fn write_sd_vae(dir: &Path, chans: &[usize]) {
    write_vae(dir, chans, false)
}

/// [`write_sd_vae`]; `bn`: plus the FLUX.2 VAE's BatchNorm stats over the 2x2-packed latent
/// channels (`bn.running_mean` / `bn.running_var`, 16 wide).
fn write_vae(dir: &Path, chans: &[usize], bn: bool) {
    let (n, c0, top) = (chans.len(), chans[0], chans[chans.len() - 1]);
    let mut s = Specs::new();
    conv(&mut s, "encoder.conv_in", 3, c0, 3);
    let mut prev = c0;
    for (i, &c) in chans.iter().enumerate() {
        resnet(&mut s, &format!("encoder.down_blocks.{i}.resnets.0"), prev, c);
        if i + 1 < n {
            conv(&mut s, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"), c, c, 3);
        }
        prev = c;
    }
    vae_mid(&mut s, "encoder", top);
    nrm(&mut s, "encoder.conv_norm_out", top);
    conv(&mut s, "encoder.conv_out", top, 8, 3);
    conv(&mut s, "quant_conv", 8, 8, 1);
    conv(&mut s, "post_quant_conv", 4, 4, 1);
    conv(&mut s, "decoder.conv_in", 4, top, 3);
    vae_mid(&mut s, "decoder", top);
    let mut prev = top;
    for (j, &c) in chans.iter().rev().enumerate() {
        resnet(&mut s, &format!("decoder.up_blocks.{j}.resnets.0"), prev, c);
        resnet(&mut s, &format!("decoder.up_blocks.{j}.resnets.1"), c, c);
        if j + 1 < n {
            conv(&mut s, &format!("decoder.up_blocks.{j}.upsamplers.0.conv"), c, c, 3);
        }
        prev = c;
    }
    nrm(&mut s, "decoder.conv_norm_out", c0);
    conv(&mut s, "decoder.conv_out", c0, 3, 3);
    if bn {
        s.push(("bn.running_mean".into(), vec![16]));
        s.push(("bn.running_var".into(), vec![16]));
    }
    let v = dir.join("vae");
    write_specs(&v.join("diffusion_pytorch_model.safetensors"), 7, &s, StDtype::F32);
    write_json(
        &v.join("config.json"),
        &json!({
            "_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3,
            "latent_channels": 4, "block_out_channels": chans, "layers_per_block": 1,
            "norm_num_groups": 32, "scaling_factor": 0.18215, "sample_size": 16, "batch_norm_eps": 1e-4,
            "down_block_types": vec!["DownEncoderBlock2D"; n],
            "up_block_types": vec!["UpDecoderBlock2D"; n]
        }),
    );
}

#[test]
fn sd_vae() {
    // runner: gibbs_engine::runtime::diffusion::Net (`sd_vae` spec, networks `decoder` / `encoder`):
    // affine-free networks; the host applies the latent affine around them.
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net};
    let case = "diffa_sd_vae";
    let dir = fixture_dir(case);
    write_sd_vae(&dir, &[32, 64]);
    let (h, w) = (6, 5);
    let z = rand(4 * h * w, 3);
    let img = rand(3 * 16 * 12, 4);
    golden(case, TOL, |dev| {
        let de = Net::open(&dir.join("vae"), "sd_vae", "decoder", pin(dev)).unwrap();
        let en = Net::new(de.model.clone(), "encoder", pin(dev));
        let var = |n: &str| de.model.env.var(n).unwrap().num().unwrap() as f32;
        let (scale, shift, f) = (var("scaling"), var("shift"), var("factor") as usize);
        let hw = |h: usize, w: usize| [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
        let zs: Vec<f32> = z.iter().map(|v| v / scale + shift).collect();
        let x = on(dev, || de.run_one(&hw(h, w), &[("z", Tensor::F32(zs))], &[], None)).unwrap();
        let (eh, ew) = (16 / f, 12 / f);
        let m = on(dev, || en.run_one(&hw(eh, ew), &[("image", Tensor::F32(img.clone()))], &[], None)).unwrap();
        let m: Vec<f32> = m.iter().map(|v| (v - shift) * scale).collect();
        assert_eq!(unconsumed(&de.model, &[&de, &en], &[]), Vec::<String>::new());
        assert_eq!((h * f, w * f, eh, ew), (2 * h, 2 * w, 8, 6));
        let mut o = Out::default();
        o.f("decode", &x).f("encode", &m);
        o
    });
}

/// diffusers AutoencoderDC `vae/` decoder (f4c8-style: ResBlock + 2 EfficientViT levels).
fn write_dcae(dir: &Path) {
    let (chans, hd, latent, ms_k) = ([8usize, 16, 32], 4usize, 8usize, 5usize);
    let mut s = Specs::new();
    conv(&mut s, "decoder.conv_in", latent, chans[2], 3);
    for i in (0..3).rev() {
        let c = chans[i];
        let j = if i + 1 < 3 {
            conv(&mut s, &format!("decoder.up_blocks.{i}.0.conv"), chans[i + 1], c, 3);
            1
        } else {
            0
        };
        let p = format!("decoder.up_blocks.{i}.{j}");
        if i == 0 {
            conv(&mut s, &format!("{p}.conv1"), c, c, 3);
            s.push((format!("{p}.conv2.weight"), vec![c, c, 3, 3]));
            nrm(&mut s, &format!("{p}.norm"), c);
        } else {
            let (a, inner, c2) = (format!("{p}.attn"), c, 4 * c);
            for n in ["to_q", "to_k", "to_v"] {
                s.push((format!("{a}.{n}.weight"), vec![inner, c]));
            }
            s.push((format!("{a}.to_qkv_multiscale.0.proj_in.weight"), vec![3 * inner, 1, ms_k, ms_k]));
            s.push((format!("{a}.to_qkv_multiscale.0.proj_out.weight"), vec![3 * inner, hd, 1, 1]));
            s.push((format!("{a}.to_out.weight"), vec![c, 2 * inner]));
            nrm(&mut s, &format!("{a}.norm_out"), c);
            let g = format!("{p}.conv_out");
            conv(&mut s, &format!("{g}.conv_inverted"), c, c2, 1);
            s.push((format!("{g}.conv_depth.weight"), vec![c2, 1, 3, 3]));
            s.push((format!("{g}.conv_depth.bias"), vec![c2]));
            s.push((format!("{g}.conv_point.weight"), vec![c, c2 / 2]));
            nrm(&mut s, &format!("{g}.norm"), c);
        }
    }
    nrm(&mut s, "decoder.norm_out", chans[0]);
    conv(&mut s, "decoder.conv_out", chans[0], 3, 3);
    write_specs(&dir.join("diffusion_pytorch_model.safetensors"), 41, &s, StDtype::F32);
    write_json(
        &dir.join("config.json"),
        &json!({
            "_class_name": "AutoencoderDC", "in_channels": 3, "latent_channels": latent,
            "attention_head_dim": hd, "decoder_block_out_channels": chans,
            "decoder_layers_per_block": [1, 1, 1],
            "decoder_block_types": ["ResBlock", "EfficientViTBlock", "EfficientViTBlock"],
            "decoder_qkv_multiscales": [[], [ms_k], [ms_k]],
            "upsample_block_type": "interpolate", "decoder_norm_types": "rms_norm",
            "decoder_act_fns": "silu", "scaling_factor": 0.5
        }),
    );
}

#[test]
fn dcae() {
    // runner: gibbs_engine::runtime::diffusion::Net (`dcae` spec, network `decoder`); `z` is
    // already latents / scaling_factor.
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::{unconsumed, Net};
    let case = "diffa_dcae";
    let dir = fixture_dir(case).join("vae");
    write_dcae(&dir);
    let (h, w) = (2, 3);
    let z = rand(8 * h * w, 3);
    golden(case, TOL, |dev| {
        let de = Net::open(&dir, "dcae", "decoder", pin(dev)).unwrap();
        let vars = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
        let y = on(dev, || de.run_one(&vars, &[("z", Tensor::F32(z.clone()))], &[], None)).unwrap();
        assert_eq!(unconsumed(&de.model, &[&de], &[]), Vec::<String>::new());
        let mut o = Out::default();
        o.f("decode", &y);
        o
    });
}

// ---------------------------------------------------------------------------------------------
// Tiny SD 1.x pipeline through the public image-model loader
// ---------------------------------------------------------------------------------------------

#[test]
fn tiny_sd() {
    use gibbs_engine::runtime::diffusion::{load_image, ImageRequest};
    let case = "diffa_tiny_sd";
    let dir = fixture_dir(case);
    write_json(
        &dir.join("model_index.json"),
        &json!({
            "_class_name": "StableDiffusionPipeline",
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "EulerDiscreteScheduler"]
        }),
    );
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({
            "_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear",
            "beta_start": 0.00085, "beta_end": 0.012, "num_train_timesteps": 1000,
            "prediction_type": "epsilon", "timestep_spacing": "trailing"
        }),
    );
    write_clip(&dir, "text_encoder", "tokenizer", true, false, 21);
    write_unet(&dir, CLIP_D, 0, 0);
    // Real SD geometry: 4 levels, 8x VAE factor.
    write_sd_vae(&dir, &[32, 32, 32, 32]);
    let hub = clip_hub(&dir);
    let art = artifacts(&dir, "golden/tiny-sd", "none");
    let req = ImageRequest {
        prompt: "t5 t17 t200".into(),
        width: Some(32),
        height: Some(32),
        steps: Some(2),
        guidance: Some(5.0),
        seed: 5,
        ..Default::default()
    };
    golden(case, TOL, |dev| {
        // runner: runtime::diffusion::load_image (the public loader), pinned to `dev`
        let p = load_image(&art, &hub, pin(dev)).expect("image repo").unwrap();
        let img = on(dev, || p.text_to_image(&req)).unwrap();
        assert_eq!((img.width, img.height, img.channels), (32, 32, 3));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}

/// An SD 1.x LDM single-file checkpoint: the CLIP tower of `write_clip` under
/// `cond_stage_model.transformer.`, a 2-level LDM UNet (layers_per_block 2, as every LDM file)
/// under `model.diffusion_model.`, a 4-level LDM VAE under `first_stage_model.`; the reference
/// repo's configs and tokenizer dir sit in the hub cache.
#[test]
fn sd_single_file() {
    use gibbs_engine::model::diff_fixtures::{ldm_unet_specs, ldm_vae_specs};
    use gibbs_engine::runtime::diffusion::{load_image, ImageRequest};
    let case = "diffa_sd_single_file";
    let dir = fixture_dir(case);
    let d = CLIP_D;
    let te = "cond_stage_model.transformer.text_model";
    let mut s: Specs = vec![
        (format!("{te}.embeddings.token_embedding.weight"), vec![CLIP_VOCAB, d]),
        (format!("{te}.embeddings.position_embedding.weight"), vec![77, d]),
    ];
    for l in 0..2 {
        let p = format!("{te}.encoder.layers.{l}");
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(&mut s, &format!("{p}.self_attn.{n}"), d, d, true);
        }
        lin(&mut s, &format!("{p}.mlp.fc1"), d, 2 * d, true);
        lin(&mut s, &format!("{p}.mlp.fc2"), 2 * d, d, true);
        nrm(&mut s, &format!("{p}.layer_norm1"), d);
        nrm(&mut s, &format!("{p}.layer_norm2"), d);
    }
    nrm(&mut s, &format!("{te}.final_layer_norm"), d);
    let unet = ldm_unet_specs(32, &[1, 2], 2, &[true, true], d, 4);
    s.extend(unet.into_iter().map(|(n, sh)| (format!("model.diffusion_model.{n}"), sh)));
    s.extend(ldm_vae_specs(32, &[1, 1, 1, 1]));
    write_specs(&dir.join("sd15.safetensors"), 31, &s, StDtype::F32);

    let hub = clip_hub(&dir);
    let r = dir.join("hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/main");
    write_json(
        &r.join("model_index.json"),
        &json!({
            "_class_name": "StableDiffusionPipeline",
            "text_encoder": ["transformers", "CLIPTextModel"], "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"], "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "EulerDiscreteScheduler"]
        }),
    );
    write_json(
        &r.join("scheduler/scheduler_config.json"),
        &json!({
            "_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear",
            "beta_start": 0.00085, "beta_end": 0.012, "num_train_timesteps": 1000,
            "prediction_type": "epsilon", "timestep_spacing": "trailing"
        }),
    );
    write_json(
        &r.join("text_encoder/config.json"),
        &json!({
            "architectures": ["CLIPTextModel"], "model_type": "clip_text_model", "hidden_act": "quick_gelu",
            "hidden_size": d, "intermediate_size": 2 * d, "num_attention_heads": 1,
            "num_hidden_layers": 2, "max_position_embeddings": 77, "vocab_size": CLIP_VOCAB,
            "layer_norm_eps": 1e-5
        }),
    );
    write_json(&r.join("tokenizer/tokenizer.json"), &clip_tokenizer_json());
    let pad = json!({"pad_token": "<|endoftext|>"});
    write_json(&r.join("tokenizer/special_tokens_map.json"), &pad);
    write_json(&r.join("tokenizer/tokenizer_config.json"), &pad);
    write_json(
        &r.join("unet/config.json"),
        &json!({
            "_class_name": "UNet2DConditionModel", "sample_size": 4, "in_channels": 4,
            "out_channels": 4, "block_out_channels": [32, 64], "layers_per_block": 2,
            "attention_head_dim": 8, "cross_attention_dim": d, "norm_num_groups": 32,
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
            "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            "mid_block_type": "UNetMidBlock2DCrossAttn"
        }),
    );
    write_json(
        &r.join("vae/config.json"),
        &json!({
            "_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3,
            "latent_channels": 4, "block_out_channels": [32, 32, 32, 32], "layers_per_block": 2,
            "norm_num_groups": 32, "scaling_factor": 0.18215, "sample_size": 32,
            "down_block_types": vec!["DownEncoderBlock2D"; 4],
            "up_block_types": vec!["UpDecoderBlock2D"; 4]
        }),
    );
    let art = artifacts(&dir, "golden/sd-single-file", "sd15.safetensors");
    let req = ImageRequest {
        prompt: "t5 t17 t200".into(),
        width: Some(32),
        height: Some(32),
        steps: Some(2),
        guidance: Some(5.0),
        seed: 5,
        ..Default::default()
    };
    golden(case, TOL, |dev| {
        // runner: runtime::diffusion::load_image (the public loader), pinned to `dev`
        let p = load_image(&art, &hub, pin(dev)).expect("image repo").unwrap();
        let img = on(dev, || p.text_to_image(&req)).unwrap();
        assert_eq!((img.width, img.height, img.channels), (32, 32, 3));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}

/// A WordLevel stand-in for the Gemma tokenizer: `<pad>` 0, `<eos>` 1, `<bos>` 2 (prepended by
/// the post-processor), `<unk>` 3 (every instruction word), `tK` = K (4..256).
fn gemma_tokenizer_json() -> Value {
    let mut vocab = Map::new();
    let specials = [("<pad>", 0), ("<eos>", 1), ("<bos>", 2), ("<unk>", 3)];
    for (s, id) in specials {
        vocab.insert(s.into(), json!(id));
    }
    for i in 4..256 {
        vocab.insert(format!("t{i}"), json!(i));
    }
    let added: Vec<Value> = specials
        .iter()
        .map(|(s, id)| {
            json!({"id": id, "content": s, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 0}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}}
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
}

// ---------------------------------------------------------------------------------------------
// Tiny Sana pipeline (Gemma 2 + linear DiT + DC-AE f4c8) through the public image-model loader
// ---------------------------------------------------------------------------------------------

#[test]
fn sana_pipeline() {
    use gibbs_engine::runtime::diffusion::{load_image, ImageRequest};
    let case = "diffa_sana_pipeline";
    let dir = fixture_dir(case);
    write_json(
        &dir.join("model_index.json"),
        &json!({
            "_class_name": "SanaPipeline",
            "text_encoder": ["transformers", "Gemma2Model"],
            "tokenizer": ["transformers", "GemmaTokenizerFast"],
            "transformer": ["diffusers", "SanaTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderDC"],
            "scheduler": ["diffusers", "DPMSolverMultistepScheduler"]
        }),
    );
    write_json(
        &dir.join("scheduler/scheduler_config.json"),
        &json!({
            "_class_name": "DPMSolverMultistepScheduler", "algorithm_type": "dpmsolver++",
            "prediction_type": "flow_prediction", "use_flow_sigmas": true, "solver_order": 2,
            "solver_type": "midpoint", "final_sigmas_type": "zero", "flow_shift": 3.0,
            "num_train_timesteps": 1000
        }),
    );
    write_json(&dir.join("tokenizer/tokenizer.json"), &gemma_tokenizer_json());
    write_gemma2(&dir.join("text_encoder"));
    write_sana_dit(&dir.join("transformer"));
    // DC-AE f4c8: latent channels == SANA_C.
    write_dcae(&dir.join("vae"));
    let hub = HuggingFaceHub::with_cache_dir(dir.join("hub"));
    let art = artifacts(&dir, "golden/tiny-sana", "none");
    // 32x32 = 8x8 latents (the old pipeline's minimum); CFG runs (guidance > 1).
    let req = ImageRequest {
        prompt: " T5 t17 t200 ".into(),
        width: Some(32),
        height: Some(32),
        steps: Some(2),
        guidance: Some(4.5),
        seed: 5,
        ..Default::default()
    };
    golden(case, TOL, |dev| {
        // runner: runtime::diffusion::load_image (the public loader), pinned to `dev`
        let p = load_image(&art, &hub, pin(dev)).expect("image repo").unwrap();
        let img = on(dev, || p.text_to_image(&req)).unwrap();
        assert_eq!((img.width, img.height, img.channels), (32, 32, 3));
        let px: Vec<f32> = img.data.iter().map(|&v| v as f32).collect();
        let mut o = Out::default();
        o.f("pixels", &stride(&px));
        o
    });
}
