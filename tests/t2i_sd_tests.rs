//! Stable Diffusion on real weights (stabilityai/sd-turbo diffusers fp16 in disk/): prompts steer the
//! image, image-to-image keeps the input's layout. Prints SKIP when the repo is not downloaded.

use gibbs_engine::model::hub::HuggingFaceHub;
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::text_to_image::{load_image_model, ImageRequest};
use std::path::PathBuf;

const REPO: &str = "stabilityai/sd-turbo";

/// Mean of each RGB channel.
fn channel_means(img: &ImageBuffer) -> [f64; 3] {
    let mut s = [0f64; 3];
    for y in 0..img.height {
        for x in 0..img.width {
            let (r, g, b) = img.get_pixel(x, y);
            s[0] += r as f64;
            s[1] += g as f64;
            s[2] += b as f64;
        }
    }
    s.map(|v| v / (img.width * img.height) as f64)
}

#[test]
fn sd_turbo_follows_prompt_and_img2img_strength() {
    let hub =
        HuggingFaceHub::with_cache_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("disk"));
    let art = hub.locate_model(REPO);
    if art.weights_path.is_none() {
        eprintln!("SKIP: {REPO} not downloaded");
        return;
    }
    let sd = load_image_model(&art, &hub)
        .expect("sd-turbo is an image repo")
        .unwrap();
    let gen = |prompt: &str| {
        let req = ImageRequest {
            prompt: prompt.into(),
            steps: Some(1),
            seed: 7,
            ..Default::default()
        };
        sd.text_to_image(&req).unwrap()
    };
    let red = gen("a flat solid bright red background, plain red color");
    let blue = gen("a flat solid bright blue background, plain blue color");
    assert_eq!((red.width, red.height), (512, 512));
    let (r, b) = (channel_means(&red), channel_means(&blue));
    assert!(r[0] > r[2] + 60.0, "red prompt gave RGB means {r:?}");
    assert!(b[2] > b[0] + 60.0, "blue prompt gave RGB means {b:?}");

    // img2img of the red image toward "blue": low strength stays red, more strength moves toward blue
    // (non-zero-SNR schedules keep some input color even at strength 1, as in diffusers).
    let req = ImageRequest {
        prompt: "a flat solid bright blue background".into(),
        steps: Some(4),
        seed: 3,
        ..Default::default()
    };
    let red = red.resize(256, 256);
    let kept = channel_means(&sd.image_to_image(&req, &red, 0.25).unwrap());
    assert!(
        kept[0] > kept[2],
        "strength 0.25 should keep the red input, got {kept:?}"
    );
    let redrawn = channel_means(&sd.image_to_image(&req, &red, 1.0).unwrap());
    assert!(
        redrawn[2] - redrawn[0] > kept[2] - kept[0] + 20.0,
        "strength 1.0 should move further toward blue than 0.25: {redrawn:?} vs {kept:?}"
    );

    assert!(sd
        .text_to_image(&ImageRequest {
            prompt: "x".into(),
            width: Some(500),
            ..Default::default()
        })
        .is_err());
}
