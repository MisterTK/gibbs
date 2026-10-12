//! SigLIP vision transformer, the image tower of Idefics3/SmolVLM (and PaliGemma-style VLMs): a
//! stride-`patch` conv embeds the patches, learned positions are added, pre-LN encoder layers with
//! bidirectional attention and a tanh-GELU MLP follow, then a final LayerNorm. CPU f32 on the
//! [`nn`] primitives.

use crate::graph::{DType, Graph, Op, ValueId, Weights};
use crate::model::nn;
use crate::model::vision_graph::Gb;
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

    /// Adds the tower to `b` on `pixels [3, S, S]` and returns the `[n, hidden]` features; consumes
    /// the weights (moved into the graph, not copied).
    pub fn add_to_graph(self, b: &mut Gb, pixels: ValueId) -> ValueId {
        let (s, d) = (self.image_size, self.hidden);
        let (g, n) = (s / self.patch, (s / self.patch).pow(2));
        let pw = b.f32w("vision.patch.weight", vec![d, 3, self.patch, self.patch], self.patch_w);
        let pb = b.f32w("vision.patch.bias", vec![d], self.patch_b);
        let conv = Op::Conv2d {
            stride: [self.patch; 2],
            padding: [0; 4],
            dilation: [1; 2],
            groups: 1,
        };
        let emb = b.op("vision.patch", conv, &[pixels, pw, pb], vec![d, g, g]);
        let emb = b.reshape("vision.patch.flat", emb, vec![d, n]);
        let x = b.transpose("vision.patch.t", emb, &[d, n], &[1, 0]);
        let pos = b.f32w("vision.pos", vec![n, d], self.pos);
        let mut x = b.add("vision.embed", x, pos, vec![n, d]);
        let eps = self.eps;
        let norm = |b: &mut Gb, name: &str, x: ValueId, nm: Norm| {
            let gm = b.f32w(&format!("{name}.weight"), vec![d], nm.gamma);
            let bt = b.f32w(&format!("{name}.bias"), vec![d], nm.beta);
            b.op(name, Op::LayerNorm { eps }, &[x, gm, bt], vec![n, d])
        };
        let lin = |b: &mut Gb, name: &str, x: ValueId, l: Linear| {
            let w = b.f32w(&format!("{name}.weight"), vec![l.out_f, l.in_f], l.w);
            let bias = b.f32w(&format!("{name}.bias"), vec![l.out_f], l.b);
            b.linear(name, x, w, Some(bias), n, l.out_f)
        };
        let attn = Op::Attention {
            heads: self.heads,
            kv_heads: self.heads,
            causal: false,
            window: None,
            soft_cap: None,
        };
        for (i, l) in self.layers.into_iter().enumerate() {
            let p = format!("vision.layers.{i}");
            let h = norm(b, &format!("{p}.ln1"), x, l.ln1);
            let q = lin(b, &format!("{p}.q"), h, l.q);
            let k = lin(b, &format!("{p}.k"), h, l.k);
            let v = lin(b, &format!("{p}.v"), h, l.v);
            let a = b.op(&format!("{p}.attn"), attn.clone(), &[q, k, v], vec![n, d]);
            let o = lin(b, &format!("{p}.o"), a, l.o);
            x = b.add(&format!("{p}.res1"), x, o, vec![n, d]);
            let h2 = norm(b, &format!("{p}.ln2"), x, l.ln2);
            let inter = l.fc1.out_f;
            let m = lin(b, &format!("{p}.fc1"), h2, l.fc1);
            let m = b.op(&format!("{p}.act"), Op::Gelu { tanh: true }, &[m], vec![n, inter]);
            let m = lin(b, &format!("{p}.fc2"), m, l.fc2);
            x = b.add(&format!("{p}.res2"), x, m, vec![n, d]);
        }
        norm(b, "vision.post", x, self.post)
    }

    /// The tower alone as a graph: `pixels [3, S, S]` → `[n, hidden]`.
    pub fn into_graph(self) -> (Graph, Weights) {
        let mut b = Gb::new();
        let s = self.image_size;
        let px = b.input("pixels", vec![3, s, s], DType::F32);
        let out = self.add_to_graph(&mut b, px);
        b.finish(&[out])
    }
}

/// Deterministic synthetic tiny tower (hidden 16, 2 layers, 16 px image, 4 px patches): the
/// fixture of `tests/cpu_free_matrix.rs` and the CPU-vs-Metal parity tests.
pub mod tiny {
    use super::*;

    pub const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const PATCH: usize = 4;
    const IMAGE: usize = 16;

    pub fn config() -> Value {
        serde_json::json!({
            "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 2,
            "num_attention_heads": 2, "image_size": IMAGE, "patch_size": PATCH,
        })
    }

    /// Value stream seeded by `name` (LCG over its bytes), in `[-1, 1)`.
    pub fn values(name: &str, len: usize) -> Vec<f32> {
        let mut s = name.bytes().fold(7u64, |a, b| a.wrapping_mul(131).wrapping_add(b as u64));
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// The tensor `name` of a tiny tower: gains near 1, biases small, weights `±0.2`.
    pub fn tensor(name: &str) -> Result<Vec<f32>, String> {
        let grid2 = (IMAGE / PATCH).pow(2);
        let (len, scale, offset) = match name {
            n if n.ends_with("patch_embedding.weight") => (HIDDEN * 3 * PATCH * PATCH, 0.2, 0.0),
            n if n.ends_with("position_embedding.weight") => (grid2 * HIDDEN, 0.2, 0.0),
            n if n.ends_with("layer_norm1.weight")
                || n.ends_with("layer_norm2.weight")
                || n.ends_with("post_layernorm.weight") => (HIDDEN, 0.1, 1.0),
            n if n.ends_with("fc1.weight") => (INTER * HIDDEN, 0.2, 0.0),
            n if n.ends_with("fc2.weight") => (HIDDEN * INTER, 0.2, 0.0),
            n if n.ends_with("fc1.bias") => (INTER, 0.05, 0.0),
            n if n.ends_with("proj.weight") => (HIDDEN * HIDDEN, 0.2, 0.0),
            n if n.ends_with(".bias") => (HIDDEN, 0.05, 0.0),
            n => return Err(format!("tiny SigLIP has no tensor {n}")),
        };
        Ok(values(name, len).into_iter().map(|v| offset + v * scale).collect())
    }

    pub fn vision() -> SiglipVision {
        SiglipVision::load(&config(), "model.vision_model", &tensor).expect("tiny SigLIP loads")
    }

    /// The tower alone as a graph (`pixels [3, 16, 16]` → `[16, 16]`).
    pub fn graph() -> (Graph, Weights) {
        vision().into_graph()
    }
}
