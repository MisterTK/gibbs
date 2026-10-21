//! Real-weight checks for the diffusion-transformer pipelines' shared parts. Each test prints
//! `SKIP: <reason>` and passes when its weights are not cached in `disk/`.

use gibbs_engine::arch::expr::Val;
use gibbs_engine::graph::Tensor;
use gibbs_engine::hardware::{Device, PlacementMode};
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::runtime::diffusion::Net;
use std::path::Path;

/// The SD3 16-channel VAE (`sd_vae` spec: no quant convs, `to_q/to_k/to_v/to_out.0` mid attention,
/// scale 1.5305 and shift 0.0609 from its config) reconstructs a photo: encode -> decode stays
/// close to the input.
#[test]
fn sd3_vae_round_trip_reconstructs_photo() {
    let dir =
        Path::new("disk/models--adamo1139--stable-diffusion-3.5-medium-ungated/snapshots/main/vae");
    let img = Path::new("/tmp/gibbs-testimg/cat.png");
    if !dir.join("diffusion_pytorch_model.safetensors").exists() || !img.exists() {
        eprintln!("SKIP: {} or {} missing", dir.display(), img.display());
        return;
    }
    let mode = PlacementMode::Pin(Device::Metal);
    let de = Net::open(dir, "sd_vae", "decoder", mode).unwrap();
    let en = Net::new(de.model.clone(), "encoder", mode);
    let var = |n: &str| de.model.env.var(n).unwrap().num().unwrap() as f32;
    let (scale, shift) = (var("scaling"), var("shift"));
    assert_eq!(var("latent_channels"), 16.0);
    let input = ImageBuffer::decode(&std::fs::read(img).unwrap()).unwrap().resize(256, 256);
    let (h, w) = (256usize, 256usize);
    // HWC u8 -> CHW [-1, 1].
    let chw: Vec<f32> = (0..3)
        .flat_map(|c| (0..h * w).map(move |p| (c, p)))
        .map(|(c, p)| input.data[p * 3 + c] as f32 / 127.5 - 1.0)
        .collect();
    let hw = |h: usize, w: usize| [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
    let mean = en.run_one(&hw(h, w), &[("image", Tensor::F32(chw.clone()))], &[], None).unwrap();
    let z: Vec<f32> = mean.iter().map(|v| (v - shift) * scale).collect();
    assert_eq!(z.len(), 16 * 32 * 32);
    // Scaled latents are roughly unit-variance.
    let std = (z.iter().map(|v| v * v).sum::<f32>() / z.len() as f32).sqrt();
    assert!((0.5..2.0).contains(&std), "latent rms {std}");
    let zin = z.iter().map(|v| v / scale + shift).collect();
    let out = de.run_one(&hw(32, 32), &[("z", Tensor::F32(zin))], &[], None).unwrap();
    let px = |v: f32| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round();
    let mse = chw.iter().zip(&out).map(|(&a, &b)| (px(a) - px(b)).powi(2) as f64).sum::<f64>() / chw.len() as f64;
    let p = 10.0 * (255.0f64 * 255.0 / mse).log10();
    eprintln!("SD3 VAE round trip 256x256: PSNR {p:.2} dB, latent rms {std:.3}");
    assert!(p > 26.0, "PSNR {p:.2} dB");
}
