//! SigLIP vision transformer, the image tower of Idefics3/SmolVLM (and PaliGemma-style VLMs): a
//! stride-`patch` conv embeds the patches, learned positions are added, pre-LN encoder layers with
//! bidirectional attention and a tanh-GELU MLP follow, then a final LayerNorm. CPU f32 on the
//! [`nn`] primitives.

use crate::model::nn;
use serde_json::Value;

/// Tensor name → f32 values (dequantized/upcast); a missing tensor is an Err naming it.
pub type Loader<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>, String>;

fn load_sized(load: Loader, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let t = load(name)?;
    if t.len() != len {
        return Err(format!("{name}: {} values, expected {len}", t.len()));
    }
    Ok(t)
}

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    in_f: usize,
    out_f: usize,
}

impl Linear {
    fn load(load: Loader, name: &str, in_f: usize, out_f: usize) -> Result<Self, String> {
        Ok(Self {
            w: load_sized(load, &format!("{name}.weight"), in_f * out_f)?,
            b: load_sized(load, &format!("{name}.bias"), out_f)?,
            in_f,
            out_f,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        nn::linear(x, rows, &self.w, Some(&self.b), self.in_f, self.out_f)
    }
}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl Norm {
    fn load(load: Loader, name: &str, d: usize) -> Result<Self, String> {
        Ok(Self {
            gamma: load_sized(load, &format!("{name}.weight"), d)?,
            beta: load_sized(load, &format!("{name}.bias"), d)?,
        })
    }
}

struct Layer {
    ln1: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: Norm,
    fc1: Linear,
    fc2: Linear,
}

pub struct SiglipVision {
    pub hidden: usize,
    /// Square input side in pixels.
    pub image_size: usize,
    pub patch: usize,
    heads: usize,
    eps: f32,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    pos: Vec<f32>,
    layers: Vec<Layer>,
    post: Norm,
}

impl SiglipVision {
    /// `cfg` is the HF `vision_config`; tensors live under `prefix` (e.g. `model.vision_model`).
    pub fn load(cfg: &Value, prefix: &str, load: Loader) -> Result<Self, String> {
        crate::graph::avenues::gate(crate::graph::avenues::Family::Idefics3Siglip)?;
        let num = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        let (d, inter, heads, depth) = (
            num("hidden_size")?,
            num("intermediate_size")?,
            num("num_attention_heads")?,
            num("num_hidden_layers")?,
        );
        let (patch, image_size) = (num("patch_size")?, num("image_size")?);
        let act = cfg["hidden_act"].as_str().unwrap_or("gelu_pytorch_tanh");
        if act != "gelu_pytorch_tanh" || d % heads != 0 || image_size % patch != 0 {
            return Err(format!(
                "unsupported SigLIP config (hidden_act {act}, hidden {d}, heads {heads}, image {image_size}, patch {patch})"
            ));
        }
        let grid = image_size / patch;
        let e = format!("{prefix}.embeddings");
        let layers = (0..depth)
            .map(|i| {
                let p = format!("{prefix}.encoder.layers.{i}");
                Ok(Layer {
                    ln1: Norm::load(load, &format!("{p}.layer_norm1"), d)?,
                    q: Linear::load(load, &format!("{p}.self_attn.q_proj"), d, d)?,
                    k: Linear::load(load, &format!("{p}.self_attn.k_proj"), d, d)?,
                    v: Linear::load(load, &format!("{p}.self_attn.v_proj"), d, d)?,
                    o: Linear::load(load, &format!("{p}.self_attn.out_proj"), d, d)?,
                    ln2: Norm::load(load, &format!("{p}.layer_norm2"), d)?,
                    fc1: Linear::load(load, &format!("{p}.mlp.fc1"), d, inter)?,
                    fc2: Linear::load(load, &format!("{p}.mlp.fc2"), inter, d)?,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            hidden: d,
            image_size,
            patch,
            heads,
            eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            patch_w: load_sized(
                load,
                &format!("{e}.patch_embedding.weight"),
                d * 3 * patch * patch,
            )?,
            patch_b: load_sized(load, &format!("{e}.patch_embedding.bias"), d)?,
            pos: load_sized(
                load,
                &format!("{e}.position_embedding.weight"),
                grid * grid * d,
            )?,
            layers,
            post: Norm::load(load, &format!("{prefix}.post_layernorm"), d)?,
        })
    }

    /// One normalized `[3, image_size, image_size]` image → `[(image_size / patch)², hidden]`
    /// patch features, row-major over the patch grid.
    // ponytail: full fixed-size images only (all Idefics3 tiles are); NaViT-style partial-patch
    // masks and bucketed positions when a VLM feeds variable-size images.
    pub fn forward(&self, pixels: &[f32]) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("siglip::forward");
        let (s, d) = (self.image_size, self.hidden);
        assert_eq!(pixels.len(), 3 * s * s, "SigLIP input is not [3, {s}, {s}]");
        let (emb, gh, gw) = nn::conv2d(
            pixels,
            3,
            s,
            s,
            &self.patch_w,
            Some(&self.patch_b),
            d,
            self.patch,
            self.patch,
            0,
        );
        let n = gh * gw;
        let mut x = nn::transpose(&emb, d, n);
        nn::add_in_place(&mut x, &self.pos);
        let norm = |x: &[f32], nm: &Norm| nn::layer_norm(x, n, d, &nm.gamma, &nm.beta, self.eps);
        for l in &self.layers {
            let h = norm(&x, &l.ln1);
            let (q, k, v) = (l.q.forward(&h, n), l.k.forward(&h, n), l.v.forward(&h, n));
            let a = nn::attention(&q, &k, &v, n, n, self.heads, d / self.heads, false);
            nn::add_in_place(&mut x, &l.o.forward(&a, n));
            let mut m = l.fc1.forward(&norm(&x, &l.ln2), n);
            nn::gelu_tanh(&mut m);
            nn::add_in_place(&mut x, &l.fc2.forward(&m, n));
        }
        norm(&x, &self.post)
    }
}
