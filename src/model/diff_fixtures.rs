//! Tiny synthetic checkpoints and GPU-only fixtures for the diffusion networks (SD UNet2D, VAE, CLIP):
//! shared by the lib parity tests and `tests/cpu_free_matrix.rs`.

use crate::model::clip::{Act, ClipTextTransformer};
use crate::model::clip::Tensors;
use crate::model::diff_graph::testutil::{synth, Lcg};
use crate::model::unet::{Heads, SdUnet};
use crate::model::vae::SdVae;

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
pub fn res(s: &mut Specs, p: &str, i: usize, o: usize, emb: usize) {
    nrm(s, &format!("{p}.in_layers.0"), i);
    conv(s, &format!("{p}.in_layers.2"), i, o, 3);
    lin(s, &format!("{p}.emb_layers.1"), emb, o, true);
    nrm(s, &format!("{p}.out_layers.0"), o);
    conv(s, &format!("{p}.out_layers.3"), o, o, 3);
    if i != o {
        conv(s, &format!("{p}.skip_connection"), i, o, 1);
    }
}
pub fn trans(s: &mut Specs, p: &str, c: usize, ctx: usize) {
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
    lin(s, &format!("{b}.ff.net.0.proj"), c, 2 * 2 * c, true);
    lin(s, &format!("{b}.ff.net.2"), 2 * c, c, true);
    lin(s, &format!("{p}.proj_out"), c, c, true);
}

/// A 2-level LDM UNet (SD layout, 1 ResBlock per level); `add` > 0 adds the SDXL `label_emb`, `cond` > 0 the
/// LCM `cond_proj`.
pub fn unet_specs(ctx: usize, add: usize, cond: usize) -> Specs {
    let (mc, emb) = (32, 128);
    let mut s = Specs::new();
    conv(&mut s, "input_blocks.0.0", 4, mc, 3);
    res(&mut s, "input_blocks.1.0", mc, mc, emb);
    trans(&mut s, "input_blocks.1.1", mc, ctx);
    conv(&mut s, "input_blocks.2.0.op", mc, mc, 3);
    res(&mut s, "input_blocks.3.0", mc, 64, emb);
    trans(&mut s, "input_blocks.3.1", 64, ctx);
    res(&mut s, "middle_block.0", 64, 64, emb);
    trans(&mut s, "middle_block.1", 64, ctx);
    res(&mut s, "middle_block.2", 64, 64, emb);
    // (skip channels popped, res in, res out, transformer width, upsample)
    let outs = [(64, 128, 64, 64, false), (32, 96, 64, 64, true), (32, 96, 32, 32, false), (32, 64, 32, 32, false)];
    for (i, (_, cin, cout, tc, up)) in outs.into_iter().enumerate() {
        res(&mut s, &format!("output_blocks.{i}.0"), cin, cout, emb);
        trans(&mut s, &format!("output_blocks.{i}.1"), tc, ctx);
        if up {
            conv(&mut s, &format!("output_blocks.{i}.2.conv"), cout, cout, 3);
        }
    }
    nrm(&mut s, "out.0", 32);
    conv(&mut s, "out.2", 32, 4, 3);
    lin(&mut s, "time_embed.0", mc, emb, true);
    lin(&mut s, "time_embed.2", emb, emb, true);
    if cond > 0 {
        lin(&mut s, "time_embed.cond_proj", cond, mc, false);
    }
    if add > 0 {
        lin(&mut s, "label_emb.0.0", add, emb, true);
        lin(&mut s, "label_emb.0.2", emb, emb, true);
    }
    s
}


pub fn vae_specs(latent: usize) -> Vec<(String, Vec<usize>)> {
    let mut s: Vec<(String, Vec<usize>)> = Vec::new();
    let mut conv = |s: &mut Vec<_>, p: &str, i: usize, o: usize, k: usize| {
        s.push((format!("first_stage_model.{p}.weight"), vec![o, i, k, k]));
        s.push((format!("first_stage_model.{p}.bias"), vec![o]));
    };
    let norm = |s: &mut Vec<_>, p: &str, c: usize| {
        s.push((format!("first_stage_model.{p}.weight"), vec![c]));
        s.push((format!("first_stage_model.{p}.bias"), vec![c]));
    };
    fn resnet(s: &mut Vec<(String, Vec<usize>)>, p: &str, i: usize, o: usize) {
        for (n, c) in [("norm1", i), ("norm2", o)] {
            s.push((format!("first_stage_model.{p}.{n}.weight"), vec![c]));
            s.push((format!("first_stage_model.{p}.{n}.bias"), vec![c]));
        }
        for (n, ci, co, k) in [("conv1", i, o, 3), ("conv2", o, o, 3)] {
            s.push((format!("first_stage_model.{p}.{n}.weight"), vec![co, ci, k, k]));
            s.push((format!("first_stage_model.{p}.{n}.bias"), vec![co]));
        }
        if i != o {
            s.push((format!("first_stage_model.{p}.nin_shortcut.weight"), vec![o, i, 1, 1]));
            s.push((format!("first_stage_model.{p}.nin_shortcut.bias"), vec![o]));
        }
    }
    let mid = |s: &mut Vec<(String, Vec<usize>)>, p: &str, c: usize| {
        resnet(s, &format!("{p}.mid.block_1"), c, c);
        resnet(s, &format!("{p}.mid.block_2"), c, c);
        s.push((format!("first_stage_model.{p}.mid.attn_1.norm.weight"), vec![c]));
        s.push((format!("first_stage_model.{p}.mid.attn_1.norm.bias"), vec![c]));
        for n in ["q", "k", "v", "proj_out"] {
            s.push((format!("first_stage_model.{p}.mid.attn_1.{n}.weight"), vec![c, c, 1, 1]));
            s.push((format!("first_stage_model.{p}.mid.attn_1.{n}.bias"), vec![c]));
        }
    };
    // encoder: 32 -> (down) 64; decoder mirrors it with 2 resnets per level.
    conv(&mut s, "encoder.conv_in", 3, 32, 3);
    resnet(&mut s, "encoder.down.0.block.0", 32, 32);
    conv(&mut s, "encoder.down.0.downsample.conv", 32, 32, 3);
    resnet(&mut s, "encoder.down.1.block.0", 32, 64);
    mid(&mut s, "encoder", 64);
    norm(&mut s, "encoder.norm_out", 64);
    conv(&mut s, "encoder.conv_out", 64, 2 * latent, 3);
    conv(&mut s, "quant_conv", 2 * latent, 2 * latent, 1);
    conv(&mut s, "post_quant_conv", latent, latent, 1);
    conv(&mut s, "decoder.conv_in", latent, 64, 3);
    mid(&mut s, "decoder", 64);
    resnet(&mut s, "decoder.up.1.block.0", 64, 64);
    resnet(&mut s, "decoder.up.1.block.1", 64, 64);
    conv(&mut s, "decoder.up.1.upsample.conv", 64, 64, 3);
    resnet(&mut s, "decoder.up.0.block.0", 64, 32);
    resnet(&mut s, "decoder.up.0.block.1", 32, 32);
    norm(&mut s, "decoder.norm_out", 32);
    conv(&mut s, "decoder.conv_out", 32, 3, 3);
    s
}


pub fn tiny_clip(act: Act, proj: bool) -> ClipTextTransformer {
    use crate::model::diff_graph::testutil::synth;
    let (d, inter, vocab) = (128, 256, 49408);
    let mut s: Vec<(String, Vec<usize>)> = vec![
        ("text_model.embeddings.token_embedding.weight".into(), vec![vocab, d]),
        ("text_model.embeddings.position_embedding.weight".into(), vec![77, d]),
    ];
    let mut lin = |p: String, i: usize, o: usize| {
        s.push((format!("{p}.weight"), vec![o, i]));
        s.push((format!("{p}.bias"), vec![o]));
    };
    for l in 0..2 {
        let p = format!("text_model.encoder.layers.{l}");
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(format!("{p}.self_attn.{n}"), d, d);
        }
        lin(format!("{p}.mlp.fc1"), d, inter);
        lin(format!("{p}.mlp.fc2"), inter, d);
    }
    for n in ["text_model.encoder.layers.0.layer_norm1", "text_model.encoder.layers.0.layer_norm2",
        "text_model.encoder.layers.1.layer_norm1", "text_model.encoder.layers.1.layer_norm2", "text_model.final_layer_norm"] {
        s.push((format!("{n}.weight"), vec![d]));
        s.push((format!("{n}.bias"), vec![d]));
    }
    if proj {
        s.push(("text_projection.weight".into(), vec![96, d]));
    }
    let r = synth(&s, 21);
    let t = Tensors::from_reader(&r).unwrap();
    let tk = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    ClipTextTransformer::from_parts(&t, "text_model", act, 0, proj.then_some("text_projection.weight"), tk).unwrap()
}



/// A tiny SD pipeline (CLIP -> UNet -> VAE) run end to end as Metal programs; the caller arms the CPU
/// tripwire, so any CPU compute panics. Returns the decoded image size.
pub fn tiny_sd_on_gpu() -> Result<(usize, usize), String> {
    let clip = tiny_clip(Act::QuickGelu, false);
    let ctx_dim = 128;
    let r = synth(&unet_specs(ctx_dim, 24, 16), 11);
    let unet = SdUnet::load(&Tensors::from_reader(&r)?, "", Heads::Dim(16))?;
    let r = synth(&vae_specs(4), 7);
    let vae = SdVae::from_ldm(&Tensors::from_reader(&r)?)?;
    let ctx = clip.encode("a lighthouse", false)?.hidden;
    let (h, w) = (4, 4);
    let (add, tc) = (Lcg(1).vec(24, 1.0), Lcg(2).vec(16, 1.0));
    let mut x = Lcg(3).vec(4 * h * w, 1.0);
    for t in [500.0, 250.0] {
        let eps = unet.forward(&x, h, w, t, &ctx, 77, Some(&add), Some(&tc))?;
        x.iter_mut().zip(&eps).for_each(|(x, e)| *x -= 0.1 * e);
    }
    let img = vae.decode(&x, h, w)?;
    Ok((img.width, img.height))
}
