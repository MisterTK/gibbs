//! Real-weight checks for the diffusion-transformer pipelines' shared parts. Each test prints
//! `SKIP: <reason>` and passes when its weights are not cached in `disk/`.

use gibbs_engine::model::clip::Tensors;
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::loaders::safetensors::SafetensorsWeightReader;
use gibbs_engine::model::vae::SdVae;
use std::path::Path;

fn psnr(a: &ImageBuffer, b: &ImageBuffer) -> f64 {
    let mse = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.data.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// The SD3 16-channel VAE (no quant convs, `to_q/to_k/to_v/to_out.0` mid attention, scale 1.5305 and
/// shift 0.0609) reconstructs a photo: encode -> decode stays close to the input.
#[test]
fn sd3_vae_round_trip_reconstructs_photo() {
    let dir =
        Path::new("disk/models--adamo1139--stable-diffusion-3.5-medium-ungated/snapshots/main/vae");
    let file = dir.join("diffusion_pytorch_model.safetensors");
    let img = Path::new("/tmp/gibbs-testimg/cat.png");
    if !file.exists() || !img.exists() {
        eprintln!("SKIP: {} or {} missing", file.display(), img.display());
        return;
    }
    let reader = SafetensorsWeightReader::new(&file).unwrap();
    let vae = SdVae::from_diffusers(&Tensors::safetensors(&reader).unwrap())
        .unwrap()
        .with_scaling(1.5305, 0.0609);
    assert_eq!(vae.latent_channels(), 16);
    let input = ImageBuffer::decode(&std::fs::read(img).unwrap())
        .unwrap()
        .resize(256, 256);
    let (z, h, w) = vae.encode(&input).unwrap();
    assert_eq!((z.len(), h, w), (16 * 32 * 32, 32, 32));
    // Scaled latents are roughly unit-variance.
    let std = (z.iter().map(|v| v * v).sum::<f32>() / z.len() as f32).sqrt();
    assert!((0.5..2.0).contains(&std), "latent rms {std}");
    let out = vae.decode(&z, h, w).unwrap();
    let p = psnr(&input, &out);
    eprintln!("SD3 VAE round trip 256x256: PSNR {p:.2} dB, latent rms {std:.3}");
    assert!(p > 26.0, "PSNR {p:.2} dB");
}
