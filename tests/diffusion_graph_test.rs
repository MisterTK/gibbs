use gibbs_engine::graph::cpu::run_stateless;
use gibbs_engine::graph::{DType, Graph, Op, Tensor, Weights};

#[test]
fn test_diffusion_denoising_and_vae_graph_programs() {
    let channels = 4usize;
    let side = 8usize;
    let latent_elems = channels * side * side; // 256
    let cond_dim = 16usize;

    // 1. Denoising Step Graph: [latent, text_cond, sigma] -> noise_pred
    let mut denoise_g = Graph::new();
    let mut denoise_w = Weights::new();

    let latent_in = denoise_g.input("latent", vec![latent_elems], DType::F32);
    let text_cond = denoise_g.input("cond", vec![cond_dim], DType::F32);
    let _sigma = denoise_g.input("sigma", vec![1], DType::F32);

    let proj_w = denoise_g.weight("cond_proj.weight", vec![latent_elems, cond_dim], DType::F32);
    denoise_w.insert(
        "cond_proj.weight",
        Tensor::F32(vec![0.02; latent_elems * cond_dim]),
    );
    let cond_proj = denoise_g.add(
        "cond_proj",
        Op::Linear,
        &[text_cond, proj_w],
        vec![latent_elems],
        DType::F32,
    );

    let combined = denoise_g.add(
        "add_cond",
        Op::Add,
        &[latent_in, cond_proj],
        vec![latent_elems],
        DType::F32,
    );
    let act = denoise_g.add("act", Op::Silu, &[combined], vec![latent_elems], DType::F32);

    let out_w = denoise_g.weight(
        "out_proj.weight",
        vec![latent_elems, latent_elems],
        DType::F32,
    );
    denoise_w.insert(
        "out_proj.weight",
        Tensor::F32(vec![0.01; latent_elems * latent_elems]),
    );
    let noise_pred = denoise_g.add(
        "out_proj",
        Op::Linear,
        &[act, out_w],
        vec![latent_elems],
        DType::F32,
    );

    denoise_g.output(noise_pred);

    // 2. VAE Decode Graph: [latent [4, 8, 8]] -> [3, 8, 8]
    let mut vae_g = Graph::new();
    let mut vae_w = Weights::new();

    let vae_in = vae_g.input("latent", vec![channels, side, side], DType::F32);
    let conv_w = vae_g.weight("vae.conv.weight", vec![3, channels, 1, 1], DType::F32);
    let conv_b = vae_g.weight("vae.conv.bias", vec![3], DType::F32);
    vae_w.insert(
        "vae.conv.weight",
        Tensor::F32(vec![0.1; 3 * channels * 1 * 1]),
    );
    vae_w.insert("vae.conv.bias", Tensor::F32(vec![0.0; 3]));

    let img = vae_g.add(
        "vae_conv",
        Op::Conv2d {
            stride: [1, 1],
            padding: [0, 0, 0, 0],
            dilation: [1, 1],
            groups: 1,
        },
        &[vae_in, conv_w, conv_b],
        vec![3, side, side],
        DType::F32,
    );
    vae_g.output(img);

    // 3. Run Euler diffusion loop for 4 steps on host
    let mut current_latent = vec![0.5f32; latent_elems];
    let cond = vec![0.1f32; cond_dim];

    let sigmas = [1.0f32, 0.75f32, 0.5f32, 0.25f32, 0.0f32];
    for s in 0..4 {
        let sigma = sigmas[s];
        let next_sigma = sigmas[s + 1];
        let dt = next_sigma - sigma;

        let inputs = vec![
            Tensor::F32(current_latent.clone()),
            Tensor::F32(cond.clone()),
            Tensor::F32(vec![sigma]),
        ];
        let res = run_stateless(&denoise_g, &denoise_w, inputs).expect("denoise step");
        let noise = res[0].as_f32().unwrap();

        // Euler update: x_{t+1} = x_t + dt * d
        for i in 0..latent_elems {
            current_latent[i] += dt * noise[i];
        }
    }

    // 4. Decode final image with VAE graph
    let vae_res =
        run_stateless(&vae_g, &vae_w, vec![Tensor::F32(current_latent)]).expect("vae decode");
    assert_eq!(vae_res.len(), 1);
    assert_eq!(vae_res[0].len(), 3 * side * side);
    println!("Diffusion denoising loop + VAE decode completed successfully!");
}
