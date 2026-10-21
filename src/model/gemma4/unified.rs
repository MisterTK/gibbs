//! Encoder-free media embedders of `gemma4_unified` (google/gemma-4-12B-it): images become 48 px
//! patches (3x3 merged 16 px patches) through LayerNorm -> Dense -> LayerNorm, plus factorized 2-D
//! position embeddings and a LayerNorm; audio becomes raw 40 ms frames of 16 kHz samples. Both end
//! in the shared `RMSNorm (no scale) -> Linear` projection into the text width.

use crate::model::audio::{read_wav, resample, resampled_len};
use super::ops::{linear, rms_norm};
use super::vision::{soft_tokens, target_size};
use super::Encoder;
use crate::model::image::ImageBuffer;
use crate::model::nn::layer_norm;
use crate::model::weights::{Mat, Weights};
use serde_json::Value;
use crate::graph::{DType, Graph, Op, Tensor, ValueId, Weights as GWeights};
use crate::hardware::PlacementMode;
use crate::model::bf16_to_f32;
use crate::model::vision_graph::{Gb, ProgramCache};

/// `nn.LayerNorm` default eps.
const LN_EPS: f32 = 1e-5;

/// `Gemma4UnifiedMultimodalEmbedder`: RMSNorm without scale, then a bias-free projection.
fn project(mut x: Vec<f32>, rows: usize, proj: &Mat, eps: f32) -> Vec<f32> {
    rms_norm(&mut x, proj.cols, None, eps);
    linear(&x, rows, proj)
}

/// `project` as nodes under `tag`: the unit-gain RMSNorm over `proj.cols`, then the projection.
fn project_node(b: &mut Gb, tag: &str, x: ValueId, rows: usize, proj: &Mat, eps: f32) -> ValueId {
    let x = b.rms(&format!("{tag}.embed_norm"), x, vec![1.0; proj.cols], eps, vec![rows, proj.cols]);
    let w = b.mat(&format!("{tag}.embed_proj.weight"), proj);
    b.linear(&format!("{tag}.embed_proj"), x, w, None, rows, proj.rows)
}

/// `nn.LayerNorm` (default eps) over rows of `x [n, width]` with the checkpoint's `(gamma, beta)`.
fn layer_norm_node(b: &mut Gb, name: &str, x: ValueId, ln: &(Vec<f32>, Vec<f32>), n: usize) -> ValueId {
    let (gamma, beta) = ln;
    let w = b.f32w(&format!("{name}.weight"), vec![gamma.len()], gamma.clone());
    let bias = b.f32w(&format!("{name}.bias"), vec![beta.len()], beta.clone());
    b.op(name, Op::LayerNorm { eps: LN_EPS }, &[x, w, bias], vec![n, gamma.len()])
}

pub struct VisionEmbedder {
    /// Teacher patch (px) and merge kernel: model patches are `patch * pool` px squares.
    patch: usize,
    pool: usize,
    max_soft: usize,
    eps: f32,
    ln1: (Vec<f32>, Vec<f32>),
    dense: Mat,
    dense_b: Vec<f32>,
    ln2: (Vec<f32>, Vec<f32>),
    /// `[posemb_size, 2, d]`: x table interleaved with y table per index.
    pos: Mat,
    pos_ln: (Vec<f32>, Vec<f32>),
    proj: Mat,
    /// Metal programs (set by [`Self::on_gpu`]), one per patch count; `None` = the host forward.
    gpu: Option<ProgramCache>,
}

impl VisionEmbedder {
    pub fn load(w: &Weights, vc: &Value, processor: &Value) -> Result<Self, String> {
        let u = |k: &str| {
            vc[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        let (patch, pool) = (u("patch_size")?, u("pooling_kernel_size")?);
        if u("model_patch_size")? != patch * pool {
            return Err(
                "vision_config: model_patch_size != patch_size * pooling_kernel_size".into(),
            );
        }
        let p = "model.vision_embedder";
        let ln = |s: &str, d: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((
                w.vec_n(&format!("{p}.{s}.weight"), d)?,
                w.vec_n(&format!("{p}.{s}.bias"), d)?,
            ))
        };
        let dense = w.mat(&format!("{p}.patch_dense.weight"))?;
        let (pd, d) = (dense.cols, dense.rows);
        if pd != (patch * pool).pow(2) * 3 {
            return Err(format!(
                "patch_dense takes {pd} inputs, not one RGB model patch"
            ));
        }
        let pos = w.mat(&format!("{p}.pos_embedding"))?;
        if pos.cols != 2 * d {
            return Err("pos_embedding is not [size, 2, mm_embed_dim]".into());
        }
        let proj = w.mat("model.embed_vision.embedding_projection.weight")?;
        if proj.cols != d {
            return Err("embed_vision projection width disagrees with patch_dense".into());
        }
        Ok(Self {
            patch,
            pool,
            max_soft: processor["image_processor"]["max_soft_tokens"]
                .as_u64()
                .or_else(|| vc["num_soft_tokens"].as_u64())
                .unwrap_or(280) as usize,
            eps: vc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            ln1: ln("patch_ln1", pd)?,
            dense_b: w.vec_n(&format!("{p}.patch_dense.bias"), d)?,
            dense,
            ln2: ln("patch_ln2", d)?,
            pos,
            pos_ln: ln("pos_norm", d)?,
            proj,
            gpu: None,
        })
    }
}

impl VisionEmbedder {
    /// Runs the embedder as Metal graph programs on `mode` (the default is the host forward).
    pub fn on_gpu(mut self, mode: PlacementMode) -> Self {
        self.gpu = Some(ProgramCache::new(mode));
        self
    }

    /// The x / y position-table rows of the `n` patches of a grid `gw` wide, as f32 `[n, d]` each
    /// (checkpoint row lookups).
    fn pos_rows(&self, n: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
        let d = self.dense.rows;
        let (mut xs, mut ys) = (Vec::with_capacity(n * d), Vec::with_capacity(n * d));
        for i in 0..n {
            xs.extend(self.pos.row(i % gw)[..d].iter().map(|&v| bf16_to_f32(v)));
            ys.extend(self.pos.row(i / gw)[d..].iter().map(|&v| bf16_to_f32(v)));
        }
        (xs, ys)
    }

    /// The host forward (`--backend cpu`): `[n, patch px]` patches in a grid `gw` wide → soft tokens.
    fn forward_cpu(&self, pix: &[f32], n: usize, gw: usize) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("gemma4::unified::vision");
        let (pd, d) = (pix.len() / n, self.dense.rows);
        let x = layer_norm(pix, n, pd, &self.ln1.0, &self.ln1.1, LN_EPS);
        let mut x = linear(&x, n, &self.dense);
        for row in x.chunks_exact_mut(d) {
            row.iter_mut().zip(&self.dense_b).for_each(|(v, b)| *v += b);
        }
        let mut x = layer_norm(&x, n, d, &self.ln2.0, &self.ln2.1, LN_EPS);
        let (xs, ys) = self.pos_rows(n, gw);
        x.iter_mut().zip(xs.iter().zip(&ys)).for_each(|(v, (a, b))| *v += a + b);
        let x = layer_norm(&x, n, d, &self.pos_ln.0, &self.pos_ln.1, LN_EPS);
        project(x, n, &self.proj, self.eps)
    }

    /// The embedder as a graph for `n` patches. Inputs: `patches [n, patch px]` and the x / y
    /// position rows `[n, d]`. Output: `[n, text_hidden]` soft tokens.
    fn build_graph(&self, n: usize) -> (Graph, GWeights) {
        let (pd, d) = (self.ln1.0.len(), self.dense.rows);
        let mut b = Gb::new();
        let pix = b.input("patches", vec![n, pd], DType::F32);
        let xe = b.input("pos_x", vec![n, d], DType::F32);
        let ye = b.input("pos_y", vec![n, d], DType::F32);
        let x = layer_norm_node(&mut b, "vision.patch_ln1", pix, &self.ln1, n);
        let w = b.mat("vision.patch_dense.weight", &self.dense);
        let bias = b.f32w("vision.patch_dense.bias", vec![d], self.dense_b.clone());
        let x = b.linear("vision.patch_dense", x, w, Some(bias), n, d);
        let x = layer_norm_node(&mut b, "vision.patch_ln2", x, &self.ln2, n);
        let x = b.add("vision.pos_x", x, xe, vec![n, d]);
        let x = b.add("vision.pos_y", x, ye, vec![n, d]);
        let x = layer_norm_node(&mut b, "vision.pos_norm", x, &self.pos_ln, n);
        let out = project_node(&mut b, "vision", x, n, &self.proj, self.eps);
        b.finish(&[out])
    }

    /// Runs the embedder as a Metal program (compiled once per patch count).
    fn forward_gpu(&self, gpu: &ProgramCache, pix: &[f32], n: usize, gw: usize) -> Result<Vec<f32>, String> {
        let prog = gpu
            .get(&[n], || Ok(self.build_graph(n)))
            .map_err(|e| format!("vision: {e}"))?;
        let (xs, ys) = self.pos_rows(n, gw);
        let out = prog
            .lock()
            .run(vec![Tensor::F32(pix.to_vec()), Tensor::F32(xs), Tensor::F32(ys)])?;
        out.into_iter().next().ok_or("vision program returned nothing")?.into_f32()
    }

    /// Decodes and resizes the image to its [`target_size`], then cuts model patches in raster
    /// order, each its (row, col, rgb) pixels rescaled to [0, 1]: `[n, patch px]`, `n`, grid width.
    pub(crate) fn patches(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
        let img = ImageBuffer::decode(bytes)?;
        let (k, mp) = (self.pool, self.patch * self.pool);
        let (th, tw) = target_size(img.height, img.width, self.patch, self.max_soft * k * k, k)?;
        let img = if (th, tw) == (img.height, img.width) {
            img
        } else {
            img.resize(tw, th)
        };
        let (gh, gw) = (th / mp, tw / mp);
        let n = gh * gw;
        if gw > self.pos.rows || gh > self.pos.rows {
            return Err(format!(
                "image grid {gw}x{gh} exceeds the {} position embeddings",
                self.pos.rows
            ));
        }
        let pd = mp * mp * 3;
        let mut pix = vec![0.0f32; n * pd];
        for (i, patch) in pix.chunks_exact_mut(pd).enumerate() {
            let (gy, gx) = (i / gw, i % gw);
            for y in 0..mp {
                let src = ((gy * mp + y) * tw + gx * mp) * 3;
                for (d, &s) in patch[y * mp * 3..(y + 1) * mp * 3]
                    .iter_mut()
                    .zip(&img.data[src..src + mp * 3])
                {
                    *d = s as f32 / 255.0;
                }
            }
        }
        Ok((pix, n, gw))
    }
}

impl Encoder for VisionEmbedder {
    fn count(&self, bytes: &[u8]) -> Result<usize, String> {
        soft_tokens(bytes, self.patch, self.max_soft, self.pool)
    }

    fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let (pix, n, gw) = self.patches(bytes)?;
        let rows = match &self.gpu {
            Some(gpu) => self.forward_gpu(gpu, &pix, n, gw)?,
            None => self.forward_cpu(&pix, n, gw),
        };
        Ok((rows, n))
    }
}

pub struct AudioEmbedder {
    rate: u32,
    frame: usize,
    max_tokens: usize,
    eps: f32,
    proj: Mat,
    /// Metal programs (set by [`Self::on_gpu`]), one per token count; `None` = the host forward.
    gpu: Option<ProgramCache>,
}

impl AudioEmbedder {
    pub fn load(w: &Weights, ac: &Value, processor: &Value) -> Result<Self, String> {
        let fe = &processor["feature_extractor"];
        let frame = ac["audio_samples_per_token"]
            .as_u64()
            .or_else(|| fe["audio_samples_per_token"].as_u64())
            .unwrap_or(640) as usize;
        let proj = w.mat("model.embed_audio.embedding_projection.weight")?;
        if proj.cols != frame {
            return Err(format!(
                "embed_audio takes {} samples per token, config says {frame}",
                proj.cols
            ));
        }
        Ok(Self {
            rate: fe["sampling_rate"].as_u64().unwrap_or(16_000) as u32,
            frame,
            max_tokens: processor["audio_seq_length"].as_u64().unwrap_or(750) as usize,
            eps: ac["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            proj,
            gpu: None,
        })
    }
}

impl AudioEmbedder {
    /// Runs the embedder as Metal graph programs on `mode` (the default is the host forward).
    pub fn on_gpu(mut self, mode: PlacementMode) -> Self {
        self.gpu = Some(ProgramCache::new(mode));
        self
    }

    /// `n` frames of `frame` samples (`n * frame` of `x`) to soft tokens.
    fn forward(&self, x: Vec<f32>, n: usize) -> Result<Vec<f32>, String> {
        let Some(gpu) = &self.gpu else {
            crate::hardware::cpu_compute::enter("gemma4::unified::audio");
            return Ok(project(x, n, &self.proj, self.eps));
        };
        let prog = gpu
            .get(&[n], || {
                let mut b = Gb::new();
                let frames = b.input("frames", vec![n, self.frame], DType::F32);
                let out = project_node(&mut b, "audio", frames, n, &self.proj, self.eps);
                Ok(b.finish(&[out]))
            })
            .map_err(|e| format!("audio: {e}"))?;
        let out = prog.lock().run(vec![Tensor::F32(x)])?;
        out.into_iter().next().ok_or("audio program returned nothing")?.into_f32()
    }
}

impl AudioEmbedder {
    /// Decoded clip and its token count (one per started `frame` of 16 kHz samples), checked
    /// against the processor's `audio_seq_length` before resampling.
    fn tokens(&self, bytes: &[u8]) -> Result<(Vec<f32>, u32, usize), String> {
        let (wav, rate) = read_wav(bytes)?;
        let n = resampled_len(wav.len(), rate, self.rate).div_ceil(self.frame);
        if n == 0 {
            return Err("audio clip is empty".into());
        }
        if n > self.max_tokens {
            return Err(format!(
                "audio clip is {:.1} s; the model takes at most {:.0} s",
                wav.len() as f64 / rate as f64,
                (self.max_tokens * self.frame) as f64 / self.rate as f64
            ));
        }
        Ok((wav, rate, n))
    }
}

impl Encoder for AudioEmbedder {
    fn count(&self, bytes: &[u8]) -> Result<usize, String> {
        Ok(self.tokens(bytes)?.2)
    }

    fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let (wav, rate, n) = self.tokens(bytes)?;
        let mut x = resample(&wav, rate, self.rate);
        x.resize(n * self.frame, 0.0); // zero-pad the last frame
        Ok((self.forward(x, n)?, n))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::text::TextModel;
    use super::super::{Gemma4, State};
    use super::*;
    use crate::model::native::{MediaInput, NativeLm, SeqState};

    const H: usize = 32;

    /// A 2-layer `gemma4_unified` (sliding + full attention with `attention_k_eq_v`, image blocks
    /// bidirectional) with pseudo-random weights: the real checkpoint's layout and code paths at
    /// toy size.
    pub(crate) fn tiny_weights() -> Weights {
        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
        let mut t = |n: &str, s: &[usize]| specs.push((n.to_string(), s.to_vec()));
        let p = "model.language_model";
        t(&format!("{p}.embed_tokens.weight"), &[64, H]);
        t(&format!("{p}.norm.weight"), &[H]);
        for (i, (hd, kv, v)) in [(16, 16, true), (32, 32, false)].into_iter().enumerate() {
            let l = format!("{p}.layers.{i}");
            for n in [
                "input",
                "post_attention",
                "pre_feedforward",
                "post_feedforward",
            ] {
                t(&format!("{l}.{n}_layernorm.weight"), &[H]);
            }
            t(&format!("{l}.layer_scalar"), &[1]);
            t(&format!("{l}.mlp.gate_proj.weight"), &[64, H]);
            t(&format!("{l}.mlp.up_proj.weight"), &[64, H]);
            t(&format!("{l}.mlp.down_proj.weight"), &[H, 64]);
            t(&format!("{l}.self_attn.q_proj.weight"), &[2 * hd, H]);
            t(&format!("{l}.self_attn.o_proj.weight"), &[H, 2 * hd]);
            t(&format!("{l}.self_attn.q_norm.weight"), &[hd]);
            t(&format!("{l}.self_attn.k_norm.weight"), &[hd]);
            t(&format!("{l}.self_attn.k_proj.weight"), &[kv, H]);
            if v {
                t(&format!("{l}.self_attn.v_proj.weight"), &[kv, H]);
            }
        }
        let v = "model.vision_embedder";
        t(&format!("{v}.patch_dense.weight"), &[H, 6912]);
        t(&format!("{v}.patch_dense.bias"), &[H]);
        for (n, d) in [("patch_ln1", 6912), ("patch_ln2", H), ("pos_norm", H)] {
            t(&format!("{v}.{n}.weight"), &[d]);
            t(&format!("{v}.{n}.bias"), &[d]);
        }
        t(&format!("{v}.pos_embedding"), &[1120, 2, H]);
        t("model.embed_vision.embedding_projection.weight", &[H, H]);
        t("model.embed_audio.embedding_projection.weight", &[H, 640]);
        Weights::from_f32(specs.clone(), |name| {
            let (_, shape) = specs.iter().find(|(n, _)| n == name).unwrap();
            let mut s = name
                .bytes()
                .fold(0x9E37_79B9u32, |h, b| h.rotate_left(5) ^ b as u32);
            let norm =
                name.ends_with("norm.weight") || name.contains("_ln") || name.ends_with("scalar");
            Ok((0..shape.iter().product::<usize>())
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let r = (s >> 8) as f32 / (1 << 24) as f32 - 0.5;
                    if norm {
                        1.0 + 0.2 * r
                    } else {
                        0.5 * r
                    }
                })
                .collect())
        })
        .unwrap()
    }

    pub(crate) fn tiny() -> Gemma4 {
        let w = tiny_weights();
        let tc = serde_json::json!({
            "hidden_size": H, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
            "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "use_bidirectional_attention": "vision", "hidden_size_per_layer_input": 0,
            "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 4,
            "max_position_embeddings": 4096, "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6}
            }
        });
        let vc =
            serde_json::json!({"patch_size": 16, "pooling_kernel_size": 3, "model_patch_size": 48});
        let processor = serde_json::json!({"image_processor": {"max_soft_tokens": 280}});
        Gemma4 {
            tokenizer: Default::default(),
            text: std::sync::Arc::new(TextModel::load(&w, &tc).unwrap()),
            program: None,
            unified: true,
            vision: Box::new(VisionEmbedder::load(&w, &vc, &processor).unwrap()),
            audio: Box::new(AudioEmbedder::load(&w, &Value::Null, &processor).unwrap()),
            image_token: 60,
            audio_token: 61,
            boi: 62,
            eoi: 63,
            boa: 58,
            eoa: 59,
        }
    }

    /// 16-bit mono PCM WAV of a 440 Hz tone.
    pub(crate) fn wav(rate: u32, samples: usize) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend(16u32.to_le_bytes());
        b.extend([1u16, 1].iter().flat_map(|v| v.to_le_bytes()));
        b.extend(rate.to_le_bytes());
        b.extend((rate * 2).to_le_bytes());
        b.extend([2u16, 16].iter().flat_map(|v| v.to_le_bytes()));
        b.extend(b"data");
        b.extend((samples as u32 * 2).to_le_bytes());
        for i in 0..samples {
            let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin();
            b.extend(((v * 12000.0) as i16).to_le_bytes());
        }
        b
    }

    pub(crate) fn photo() -> Vec<u8> {
        std::fs::read("/tmp/gibbs-testimg/cat.jpg").unwrap_or_else(|_| {
            let mut img = ImageBuffer::new(200, 150);
            (0..200 * 150).for_each(|i| img.set_pixel(i % 200, i / 200, i as u8, 90, 200));
            img.to_png()
        })
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// `graph::shape::infer` reproduces every node of the unified model's graphs: text decode and
    /// prefill (k_eq_v, per-type head dims, proportional rope), the vision embedder and, on Metal,
    /// the audio embedder program.
    #[test]
    fn graph_shapes_match_inference() {
        use crate::graph::shape::{assert_compiled_shapes, assert_shapes};
        let m = tiny();
        assert_shapes(&m.text.to_decode_graph().unwrap().0);
        assert_shapes(&m.text.to_prefill_graph(9).unwrap());
        let w = tiny_weights();
        let vc =
            serde_json::json!({"patch_size": 16, "pooling_kernel_size": 3, "model_patch_size": 48});
        let processor = serde_json::json!({"image_processor": {"max_soft_tokens": 280}});
        let v = VisionEmbedder::load(&w, &vc, &processor).unwrap();
        assert_shapes(&v.build_graph(9).0);
        use crate::hardware::{available_devices, Device};
        if available_devices().contains(&Device::Metal) {
            let a = AudioEmbedder::load(&w, &Value::Null, &processor)
                .unwrap()
                .on_gpu(PlacementMode::Pin(Device::Metal));
            assert_compiled_shapes(|| a.forward(vec![0.1; 3 * a.frame], 3).unwrap());
        }
    }

    /// The cheap counts (`prompt_len`, used for context checks and count_tokens) equal what the
    /// embedders produce and `prepare` lays out, for a photo and a resampled clip.
    #[test]
    fn counts_match_encoders_and_prepare() {
        let m = tiny();
        let (img, clip) = (photo(), wav(22_050, 30_000));
        assert_eq!(
            m.vision.count(&img).unwrap(),
            m.vision.encode(&img).unwrap().1
        );
        // 30000 samples at 22.05 kHz -> 21769 at 16 kHz -> 35 frames of 640.
        assert_eq!(m.audio.count(&clip).unwrap(), 35);
        assert_eq!(m.audio.encode(&clip).unwrap().1, 35);
        let media = [MediaInput::Image(img), MediaInput::Audio(clip)];
        let ids = vec![2, 60, 5, 61, 7];
        let n = m.prompt_len(ids.clone(), &media).unwrap();
        assert_eq!(n, m.prepare(ids, &media).unwrap().0.len());
        let long = wav(16_000, 640 * 751); // past audio_seq_length
        assert!(m.audio.count(&long).unwrap_err().contains("at most"));
    }

    /// Prefill split anywhere (inside the bidirectional image block too) gives the one-shot
    /// logits, and the block really is attended both ways.
    #[test]
    fn image_block_is_bidirectional_and_chunking_invariant() {
        let m = tiny();
        let media = [MediaInput::Image(photo())];
        let ids = vec![2, 5, 60, 6, 7, 8, 9, 10];
        let run = |chunk: usize, bidir: bool| {
            let (ids, mut st): (Vec<u32>, SeqState) = m.prepare(ids.clone(), &media).unwrap();
            if !bidir {
                st.downcast_mut::<State>().unwrap().blocks.clear();
            }
            let mut last = Vec::new();
            for (i, c) in ids.chunks(chunk).enumerate() {
                last = m.forward(c, i * chunk, &mut st).unwrap();
            }
            last
        };
        let one = run(usize::MAX, true);
        for chunk in [1, 7, 64] {
            let d = max_diff(&one, &run(chunk, true));
            assert!(d < 1e-3, "chunk {chunk}: max logit diff {d}");
        }
        let causal = run(usize::MAX, false);
        assert!(
            max_diff(&one, &causal) > 1e-3,
            "image block mask had no effect"
        );
    }

    /// The Metal embedders (LayerNorm / dense / position adds, unit-gain RMSNorm, projection) match the
    /// host forwards for a photo and a resampled clip.
    #[test]
    fn metal_embedders_match_the_host_forward() {
        use crate::hardware::{available_devices, cpu_compute, Device};
        if !available_devices().contains(&Device::Metal) {
            println!("SKIP: no Metal device");
            return;
        }
        let _allow = cpu_compute::allow();
        let w = tiny_weights();
        let mode = PlacementMode::Pin(Device::Metal);
        let vc =
            serde_json::json!({"patch_size": 16, "pooling_kernel_size": 3, "model_patch_size": 48});
        let processor = serde_json::json!({"image_processor": {"max_soft_tokens": 280}});
        let load_v = || VisionEmbedder::load(&w, &vc, &processor).unwrap();
        let load_a = || AudioEmbedder::load(&w, &Value::Null, &processor).unwrap();
        let (cpu_v, gpu_v) = (load_v(), load_v().on_gpu(mode));
        let (cpu_a, gpu_a) = (load_a(), load_a().on_gpu(mode));
        let rel = |want: &[f32], got: &[f32]| {
            assert_eq!(want.len(), got.len());
            let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            max_diff(want, got) / scale
        };
        let img = photo();
        let ((want, n), (got, m)) = (cpu_v.encode(&img).unwrap(), gpu_v.encode(&img).unwrap());
        assert_eq!(n, m);
        assert!(rel(&want, &got) < 1e-3, "vision rel err {}", rel(&want, &got));
        let clip = wav(22_050, 30_000);
        let ((want, n), (got, m)) = (cpu_a.encode(&clip).unwrap(), gpu_a.encode(&clip).unwrap());
        assert_eq!((n, m), (35, 35));
        assert!(rel(&want, &got) < 1e-3, "audio rel err {}", rel(&want, &got));
    }
}
