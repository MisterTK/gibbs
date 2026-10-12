//! Tiny synthetic checkpoints and GPU-only fixtures for the diffusion networks (SD UNet2D, VAE, CLIP):
//! shared by the lib parity tests and `tests/cpu_free_matrix.rs`.

use crate::model::clip::{Act, ClipTextTransformer};
use crate::model::clip::Tensors;
use crate::model::diff_graph::testutil::{synth, Lcg};
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
    let conv = |s: &mut Vec<_>, p: &str, i: usize, o: usize, k: usize| {
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



/// The tiny SD pipeline (`sd::tiny_pipeline`) generating an image: CLIP, CFG UNet and VAE as Metal
/// programs; the caller arms the CPU tripwire, so any CPU compute panics. Returns the image size.
pub fn tiny_sd_on_gpu() -> Result<(usize, usize), String> {
    let pipe = crate::model::sd::tiny_pipeline()?;
    let req = crate::model::text_to_image::ImageRequest { prompt: "a lighthouse".into(), seed: 5, ..Default::default() };
    let img = pipe.generate(&req, None)?;
    Ok((img.width, img.height))
}

/// Every non-SD VAE (Qwen-Image, Qwen-Image 2.1, LTX-Video, DC-AE) plus the SD VAE decoded as Metal
/// programs from tiny synthetic checkpoints; the caller arms the CPU tripwire, so any CPU compute panics.
pub fn tiny_vae_on_gpu() -> Result<(), String> {
    crate::model::qwen_vae::fixture::tiny_on_gpu()?;
    crate::model::qwen_vae21::fixture::tiny_on_gpu()?;
    crate::model::ltx_vae::fixture::tiny_on_gpu()?;
    crate::model::dcae::fixture::tiny_on_gpu()?;
    let r = synth(&vae_specs(4), 7);
    let vae = SdVae::from_ldm(&Tensors::from_reader(&r)?)?;
    let (h, w) = (6, 5);
    let img = vae.decode(&Lcg(3).vec(4 * h * w, 1.0), h, w)?;
    if img.width != 2 * w || img.height != 2 * h {
        return Err(format!("sd vae: {}x{} image", img.width, img.height));
    }
    Ok(())
}

/// LDM UNet specs from a config: `mc` model channels, per-level multipliers, `nres` ResBlocks per level,
/// attention on the levels flagged in `attn`, cross-attention width `ctx`, feed-forward `ff_mult * c`.
/// (SD 1.5: `mc` 320, mult [1,2,4,4], nres 2, attn [t,t,t,f], ctx 768, ff_mult 4.)
pub fn ldm_unet_specs(mc: usize, mult: &[usize], nres: usize, attn: &[bool], ctx: usize, ff_mult: usize) -> Specs {
    let emb = 4 * mc;
    let mut s = Specs::new();
    let trans = |s: &mut Specs, p: &str, c: usize| {
        trans(s, p, c, ctx);
        // `trans` builds a 2c inner width; widen to ff_mult * c.
        let b = format!("{p}.transformer_blocks.0.ff.net");
        s.retain(|(n, _)| !n.starts_with(&b));
        lin(s, &format!("{b}.0.proj"), c, 2 * ff_mult * c, true);
        lin(s, &format!("{b}.2"), ff_mult * c, c, true);
    };
    conv(&mut s, "input_blocks.0.0", 4, mc, 3);
    let (mut ib, mut ch, mut skips) = (1, mc, vec![mc]);
    for (l, &m) in mult.iter().enumerate() {
        for _ in 0..nres {
            res(&mut s, &format!("input_blocks.{ib}.0"), ch, mc * m, emb);
            ch = mc * m;
            if attn[l] {
                trans(&mut s, &format!("input_blocks.{ib}.1"), ch);
            }
            skips.push(ch);
            ib += 1;
        }
        if l + 1 < mult.len() {
            conv(&mut s, &format!("input_blocks.{ib}.0.op"), ch, ch, 3);
            skips.push(ch);
            ib += 1;
        }
    }
    res(&mut s, "middle_block.0", ch, ch, emb);
    trans(&mut s, "middle_block.1", ch);
    res(&mut s, "middle_block.2", ch, ch, emb);
    let mut ob = 0;
    for (l, &m) in mult.iter().enumerate().rev() {
        for i in 0..=nres {
            let skip = skips.pop().unwrap();
            res(&mut s, &format!("output_blocks.{ob}.0"), ch + skip, mc * m, emb);
            ch = mc * m;
            let mut j = 1;
            if attn[l] {
                trans(&mut s, &format!("output_blocks.{ob}.1"), ch);
                j = 2;
            }
            if l > 0 && i == nres {
                conv(&mut s, &format!("output_blocks.{ob}.{j}.conv"), ch, ch, 3);
            }
            ob += 1;
        }
    }
    nrm(&mut s, "out.0", mc);
    conv(&mut s, "out.2", mc, 4, 3);
    lin(&mut s, "time_embed.0", mc, emb, true);
    lin(&mut s, "time_embed.2", emb, emb, true);
    s
}

/// SD AutoencoderKL (LDM names): `ch` base channels, per-level multipliers, 2 encoder / 3 decoder
/// ResBlocks per level, 4 latent channels.
pub fn ldm_vae_specs(ch: usize, mult: &[usize]) -> Specs {
    let p = |n: &str| format!("first_stage_model.{n}");
    let mut s = Specs::new();
    let mut resnet = |s: &mut Specs, n: &str, i: usize, o: usize| {
        for (k, c) in [("norm1", i), ("norm2", o)] {
            nrm(s, &p(&format!("{n}.{k}")), c);
        }
        conv(s, &p(&format!("{n}.conv1")), i, o, 3);
        conv(s, &p(&format!("{n}.conv2")), o, o, 3);
        if i != o {
            conv(s, &p(&format!("{n}.nin_shortcut")), i, o, 1);
        }
    };
    let top = ch * mult[mult.len() - 1];
    let mid = |s: &mut Specs, resnet: &mut dyn FnMut(&mut Specs, &str, usize, usize), side: &str| {
        resnet(s, &format!("{side}.mid.block_1"), top, top);
        resnet(s, &format!("{side}.mid.block_2"), top, top);
        nrm(s, &p(&format!("{side}.mid.attn_1.norm")), top);
        for n in ["q", "k", "v", "proj_out"] {
            conv(s, &p(&format!("{side}.mid.attn_1.{n}")), top, top, 1);
        }
    };
    conv(&mut s, &p("encoder.conv_in"), 3, ch, 3);
    let mut c = ch;
    for (l, &m) in mult.iter().enumerate() {
        for b in 0..2 {
            resnet(&mut s, &format!("encoder.down.{l}.block.{b}"), c, ch * m);
            c = ch * m;
        }
        if l + 1 < mult.len() {
            conv(&mut s, &p(&format!("encoder.down.{l}.downsample.conv")), c, c, 3);
        }
    }
    mid(&mut s, &mut resnet, "encoder");
    nrm(&mut s, &p("encoder.norm_out"), top);
    conv(&mut s, &p("encoder.conv_out"), top, 8, 3);
    conv(&mut s, &p("quant_conv"), 8, 8, 1);
    conv(&mut s, &p("post_quant_conv"), 4, 4, 1);
    conv(&mut s, &p("decoder.conv_in"), 4, top, 3);
    mid(&mut s, &mut resnet, "decoder");
    let mut c = top;
    for (l, &m) in mult.iter().enumerate().rev() {
        for b in 0..3 {
            resnet(&mut s, &format!("decoder.up.{l}.block.{b}"), c, ch * m);
            c = ch * m;
        }
        if l > 0 {
            conv(&mut s, &p(&format!("decoder.up.{l}.upsample.conv")), c, c, 3);
        }
    }
    nrm(&mut s, &p("decoder.norm_out"), c);
    conv(&mut s, &p("decoder.conv_out"), c, 3, 3);
    s
}

#[cfg(test)]
mod scale_tests {
    use super::*;
    use crate::model::diff_graph::testutil::rel_err;

    /// SD 1.5-sized UNet + VAE with random weights: CPU-vs-Metal parity at a small latent, then Metal
    /// timings at 512x512 (64x64 latent). `cargo test --release --lib sd15_scale -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sd15_scale_gpu_vs_cpu_and_speed() {
        use std::time::Instant;
        let ctx = 768;
        let ur = synth(&ldm_unet_specs(320, &[1, 2, 4, 4], 2, &[true, true, true, false], ctx, 4), 5);
        let unet = SdUnet::load(&Tensors::from_reader(&ur).unwrap(), "", Heads::Count(8)).unwrap();
        drop(ur);
        let vr = synth(&ldm_vae_specs(128, &[1, 2, 4, 4]), 6);
        let vae = SdVae::from_ldm(&Tensors::from_reader(&vr).unwrap()).unwrap();
        drop(vr);
        let mut rng = Lcg(9);
        let c = rng.vec(77 * ctx, 1.0);
        let (h, w) = (16, 16);
        let x = rng.vec(4 * h * w, 1.0);
        let t0 = Instant::now();
        let cpu = unet.forward_cpu(&x, h, w, 500.0, &c, 77, None, None).unwrap();
        let t_cpu = t0.elapsed();
        let gpu = unet.forward_gpu(&x, h, w, 500.0, &c, 77, None, None).unwrap();
        eprintln!("UNet 16x16: cpu {t_cpu:?}, rel err {:.2e}", rel_err(&gpu, &cpu));
        assert!(rel_err(&gpu, &cpu) < 1e-3);
        let z = rng.vec(4 * 16 * 16, 1.0);
        let (cpu, ..) = vae.decode_cpu(&z, 16, 16).unwrap();
        let (gpu, ..) = vae.decode_gpu(&z, 16, 16).unwrap();
        eprintln!("VAE 16x16 -> 128x128: rel err {:.2e}", rel_err(&gpu, &cpu));
        assert!(rel_err(&gpu, &cpu) < 1e-3);

        let (h, w) = (64, 64);
        let x = rng.vec(4 * h * w, 1.0);
        let t0 = Instant::now();
        unet.forward_gpu(&x, h, w, 500.0, &c, 77, None, None).unwrap();
        eprintln!("UNet 64x64 first call (compile + run) {:?}", t0.elapsed());
        let t0 = Instant::now();
        for _ in 0..3 {
            unet.forward_gpu(&x, h, w, 500.0, &c, 77, None, None).unwrap();
        }
        eprintln!("UNet 64x64 steady: {:?} / eval", t0.elapsed() / 3);
        let z = rng.vec(4 * 64 * 64, 1.0);
        let t0 = Instant::now();
        vae.decode_gpu(&z, 64, 64).unwrap();
        eprintln!("VAE decode 512x512 first call {:?}", t0.elapsed());
        let t0 = Instant::now();
        vae.decode_gpu(&z, 64, 64).unwrap();
        eprintln!("VAE decode 512x512 steady {:?}", t0.elapsed());
    }
}

#[cfg(test)]
mod vae_fixture_tests {
    /// The VAE fixtures run (CPU decode here: the tripwire is not armed in the lib tests; the Metal output
    /// of each is still checked for finiteness). The armed run lives in `tests/cpu_free_matrix.rs`.
    #[test]
    fn tiny_vae_fixtures_run() {
        super::tiny_vae_on_gpu().unwrap();
    }
}
