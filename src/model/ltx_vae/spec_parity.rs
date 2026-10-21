//! `arch/families/ltx_vae.json5` vs the hand-written LTX VAE (host forward and Metal program) on the
//! `diffb` golden geometry (widths [4, 8, 16, 16], layers [2, 1, 1, 1, 2], patch 2, 8 latents),
//! decoder and causal encoder, diffusers names and the original running-index names.

use super::*;
use crate::arch::expr::Val;
use crate::arch::testing::{compare, Runner, Weights as W};
use crate::graph::Tensor;
use crate::hardware::Device;

const CFG: Cfg = Cfg { ch: [4, 8, 16, 16], layers: [2, 1, 1, 1, 2], up: [true, true, true, false], patch: 2, latent: 8 };

/// Diffusers-named tensors and their original names.
fn tensors() -> Vec<(String, String, Vec<usize>)> {
    let mut t = Vec::new();
    let mut conv = |t: &mut Vec<_>, d: &str, o: &str, ci: usize, co: usize, k: usize| {
        t.push((format!("{d}.weight"), format!("{o}.weight"), vec![co, ci, k, k, k]));
        t.push((format!("{d}.bias"), format!("{o}.bias"), vec![co]));
    };
    let res = |t: &mut Vec<_>, d: &str, o: &str, ci: usize, co: usize| {
        conv(t, &format!("{d}.conv1.conv"), &format!("{o}.conv1.conv"), ci, co, 3);
        conv(t, &format!("{d}.conv2.conv"), &format!("{o}.conv2.conv"), co, co, 3);
        if ci != co {
            for p in ["weight", "bias"] {
                t.push((format!("{d}.norm3.{p}"), format!("{o}.norm3.norm.{p}"), vec![ci]));
            }
            conv(t, &format!("{d}.conv_shortcut.conv"), &format!("{o}.conv_shortcut"), ci, co, 1);
        }
    };
    let (ch, l) = (CFG.ch, CFG.layers);
    let rev: Vec<usize> = ch.iter().rev().copied().collect();
    let rl: Vec<usize> = l.iter().rev().copied().collect();
    conv(&mut t, "decoder.conv_in.conv", "decoder.conv_in.conv", CFG.latent, rev[0], 3);
    for j in 0..rl[0] {
        res(&mut t, &format!("decoder.mid_block.resnets.{j}"), &format!("decoder.up_blocks.0.res_blocks.{j}"), rev[0], rev[0]);
    }
    let (mut cin, mut idx) = (rev[0], 0);
    for i in 0..4 {
        let cout = rev[i];
        let b = format!("decoder.up_blocks.{i}");
        if cin != cout {
            idx += 1;
            res(&mut t, &format!("{b}.conv_in"), &format!("decoder.up_blocks.{idx}"), cin, cout);
        }
        if CFG.up[3 - i] {
            idx += 1;
            conv(&mut t, &format!("{b}.upsamplers.0.conv.conv"), &format!("decoder.up_blocks.{idx}.conv.conv"), cout, cout * 8, 3);
        }
        idx += 1;
        for j in 0..rl[i + 1] {
            res(&mut t, &format!("{b}.resnets.{j}"), &format!("decoder.up_blocks.{idx}.res_blocks.{j}"), cout, cout);
        }
        cin = cout;
    }
    conv(&mut t, "decoder.conv_out.conv", "decoder.conv_out.conv", ch[0], 3 * CFG.patch * CFG.patch, 3);
    // Encoder.
    conv(&mut t, "encoder.conv_in.conv", "encoder.conv_in.conv", 3 * CFG.patch * CFG.patch, ch[0], 3);
    let (mut cin, mut idx) = (ch[0], 0);
    for i in 0..4 {
        let cout = ch[(i + 1).min(3)];
        let b = format!("encoder.down_blocks.{i}");
        for j in 0..l[i] {
            res(&mut t, &format!("{b}.resnets.{j}"), &format!("encoder.down_blocks.{idx}.res_blocks.{j}"), cin, cin);
        }
        idx += 1;
        if CFG.up[i] {
            conv(&mut t, &format!("{b}.downsamplers.0.conv"), &format!("encoder.down_blocks.{idx}.conv"), cin, cin, 3);
            idx += 1;
        }
        if cin != cout {
            res(&mut t, &format!("{b}.conv_out"), &format!("encoder.down_blocks.{idx}"), cin, cout);
            idx += 1;
        }
        cin = cout;
    }
    for j in 0..l[4] {
        res(&mut t, &format!("encoder.mid_block.resnets.{j}"), &format!("encoder.down_blocks.{idx}.res_blocks.{j}"), cin, cin);
    }
    conv(&mut t, "encoder.conv_out.conv", "encoder.conv_out.conv", ch[3], CFG.latent + 1, 3);
    t.push(("latents_mean".into(), "per_channel_statistics.mean-of-means".into(), vec![CFG.latent]));
    t.push(("latents_std".into(), "per_channel_statistics.std-of-means".into(), vec![CFG.latent]));
    t
}

#[test]
fn ltx_vae_spec_matches_old_decoder_and_encoder() {
    let tmp = std::env::temp_dir().join(format!("gibbs-ltxvae-spec-{}", std::process::id()));
    let mut w = W::new(8);
    let names = tensors();
    for (d, _, s) in &names {
        if d.ends_with("weight") && s.len() == 1 {
            w.norm(d.clone(), s[0], 1.0);
        } else {
            w.w(d.clone(), s);
        }
    }
    let cfg = serde_json::json!({
        "_class_name": "AutoencoderKLLTXVideo", "block_out_channels": CFG.ch,
        "layers_per_block": CFG.layers, "spatio_temporal_scaling": CFG.up,
        "patch_size": CFG.patch, "latent_channels": CFG.latent,
    });
    let (dd, od) = (tmp.join("diffusers"), tmp.join("orig"));
    w.write_checkpoint(&dd, &cfg, 4);
    let orig = W {
        rng: crate::arch::testing::Rng(0),
        tensors: w.tensors.iter().zip(&names).map(|((_, s, v), (_, o, _))| (o.clone(), s.clone(), v.clone())).collect(),
    };
    orig.write_checkpoint(&od, &cfg, 4);
    let old = LtxVae::open(&Weights::open(&dd.join("model.safetensors")).unwrap(), Names { diffusers: true }, true, CFG).unwrap();
    let _cpu = crate::hardware::cpu_compute::allow();
    let mut r = crate::arch::testing::Rng(15);
    let (f, h, wd) = (2usize, 2usize, 3usize);
    let z: Vec<f32> = (0..f * CFG.latent * h * wd).map(|_| r.unit()).collect();
    let want = old.dec.fwd(Vid { d: z.clone(), c: CFG.latent, f, h, w: wd });
    // Encoder input: one patchified 16x24 frame.
    let (eh, ew) = (8usize, 12usize);
    let x: Vec<f32> = (0..12 * eh * ew).map(|_| r.unit()).collect();
    let want_e = old.enc.as_ref().unwrap().fwd(Vid { d: x.clone(), c: 12, f: 1, h: eh, w: ew });
    let num = |v: usize| Val::Num(v as f64);
    for dev in [Device::Cpu, Device::Metal] {
        if dev == Device::Metal && !crate::hardware::available_devices().contains(&Device::Metal) {
            continue;
        }
        for dir in [&dd, &od] {
            let mut run = Runner::load(dir, Some("ltx_vae"), dev, 0).unwrap();
            let got = run.run("decoder", &[("frames", num(f)), ("h", num(h)), ("w", num(wd))], vec![Tensor::F32(z.clone())]).unwrap();
            let got = got["y"].clone().into_f32().unwrap();
            let (e, c) = compare(&got, &want.d);
            eprintln!("{dev:?} {} decoder: rel {e:.2e} cos {c:.8}", dir.display());
            assert!(e < 1e-3 && c > 0.9999, "decoder rel {e} cos {c}");
            let got = run.run("encoder", &[("frames", num(1)), ("h", num(eh)), ("w", num(ew))], vec![Tensor::F32(x.clone())]).unwrap();
            let got = got["z"].clone().into_f32().unwrap();
            let (e, c) = compare(&got, &want_e.d);
            eprintln!("{dev:?} {} encoder: rel {e:.2e} cos {c:.8}", dir.display());
            assert!(e < 1e-3 && c > 0.9999, "encoder rel {e} cos {c}");
            assert!(run.unconsumed().is_empty(), "{:?}", run.unconsumed());
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
