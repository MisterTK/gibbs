//! CLIP text encoder (OpenCLIP ViT-H text tower, penultimate-layer export) used for Stable Diffusion 2.x
//! (GGUF prefix `cond_stage_model.transformer.text_model`) and ModelScope text-to-video (safetensors
//! prefix `text_model`). Every dimension comes from the stored tensor shapes; the BPE tokenizer comes
//! from `openai/clip-vit-large-patch14`.

use crate::model::hub::HuggingFaceHub;
use crate::model::loaders::gguf::GgufWeightReader;
use crate::model::loaders::safetensors::SafetensorsWeightReader;
use crate::model::loaders::WeightReader;
use crate::graph::{Op, Tensor, UnaryKind};
use crate::model::diff_graph::{gpu_mode, ByShape, Gb, GpuNet, V};
use crate::model::nn;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

/// Tensor source with PyTorch-order shapes over a GGUF file (which stores GGML-order, i.e. reversed,
/// dims) or a safetensors file (PyTorch order). Data is the same row-major PyTorch layout in both.
/// Aliases expose a tensor under another name (diffusers -> LDM, OpenCLIP -> HF), optionally as a row
/// range and/or transposed; LoRA deltas registered by [`Tensors::add_delta`] are merged on load.
pub struct Tensors<'a> {
    reader: &'a dyn WeightReader,
    shapes: HashMap<String, Vec<usize>>,
    /// safetensors dtype per stored name (`F16`, `BF16`, `F32`, ...); empty for GGUF.
    dtypes: HashMap<String, String>,
    alias: HashMap<String, Alias>,
    deltas: HashMap<String, Vec<Delta>>,
}

struct Alias {
    src: String,
    rows: Option<(usize, usize)>,
    transpose: bool,
}

/// `scale * up [out, r] . down [r, in*k*k]`, added to a weight on load.
pub struct Delta {
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub rank: usize,
    pub scale: f32,
}

impl<'a> Tensors<'a> {
    pub fn gguf(r: &'a GgufWeightReader) -> Self {
        let shapes = r
            .tensor_names()
            .filter_map(|n| Some((n.clone(), r.get_tensor_shape(n)?.iter().rev().copied().collect())))
            .collect();
        Self { reader: r, shapes, dtypes: HashMap::new(), alias: HashMap::new(), deltas: HashMap::new() }
    }

    pub fn safetensors(r: &'a SafetensorsWeightReader) -> Result<Self, String> {
        Self::from_reader(r)
    }

    /// Any reader whose `list_tensors` reports PyTorch-order shapes and dtypes.
    pub fn from_reader(r: &'a dyn WeightReader) -> Result<Self, String> {
        let (mut shapes, mut dtypes) = (HashMap::new(), HashMap::new());
        for m in r.list_tensors()? {
            dtypes.insert(m.name.clone(), m.dtype);
            shapes.insert(m.name, m.shape);
        }
        Ok(Self { reader: r, shapes, dtypes, alias: HashMap::new(), deltas: HashMap::new() })
    }

    /// PyTorch-order shape of `name`, `None` if absent.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(Vec::as_slice)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.shapes.keys()
    }

    /// Exposes stored tensor `src` as `name`: rows `r0..r1` of its first dim, then transposed (2-D).
    pub fn alias(&mut self, name: String, src: &str, rows: Option<(usize, usize)>, transpose: bool) -> Result<(), String> {
        let mut shape = self.shapes.get(src).ok_or_else(|| format!("tensor '{src}' missing"))?.clone();
        if let Some((r0, r1)) = rows {
            if shape.is_empty() || r1 > shape[0] || r0 >= r1 {
                return Err(format!("tensor '{src}' {shape:?} has no rows {r0}..{r1}"));
            }
            shape[0] = r1 - r0;
        }
        if transpose {
            let [r, c] = shape[..] else { return Err(format!("tensor '{src}' {shape:?} is not 2-D")) };
            shape = vec![c, r];
        }
        self.shapes.insert(name.clone(), shape);
        self.alias.insert(name, Alias { src: src.to_string(), rows, transpose });
        Ok(())
    }

    /// Adds a LoRA delta to weight `name` (checked against its shape).
    pub fn add_delta(&mut self, name: &str, d: Delta) -> Result<(), String> {
        let shape = self.shape(name).ok_or_else(|| format!("LoRA targets missing tensor '{name}'"))?;
        let (out, inner) = (shape[0], shape[1..].iter().product::<usize>());
        if d.up.len() != out * d.rank || d.down.len() != d.rank * inner {
            return Err(format!(
                "LoRA for '{name}' {shape:?}: up {} / down {} values do not fit rank {}",
                d.up.len(),
                d.down.len(),
                d.rank
            ));
        }
        self.deltas.entry(name.to_string()).or_default().push(d);
        Ok(())
    }

    /// `name` as f32 (dequantized/upcast, aliases resolved, LoRA merged); a missing tensor is an Err naming it.
    pub fn load(&self, name: &str) -> Result<Vec<f32>, String> {
        let shape = self.shape(name).ok_or_else(|| format!("tensor '{name}' missing"))?.to_vec();
        let mut v = match self.alias.get(name) {
            None => self.reader.load_tensor(name).map_err(|e| format!("tensor '{name}': {e}"))?,
            Some(a) => {
                let mut v = self.reader.load_tensor(&a.src).map_err(|e| format!("tensor '{}': {e}", a.src))?;
                let row = self.shapes[&a.src][1..].iter().product::<usize>();
                if let Some((r0, r1)) = a.rows {
                    v = v[r0 * row..r1 * row].to_vec();
                }
                if a.transpose {
                    v = nn::transpose(&v, shape[1], shape[0]);
                }
                v
            }
        };
        if let Some(ds) = self.deltas_for(name) {
            let inner = shape[1..].iter().product::<usize>();
            for d in ds {
                let down_t = nn::transpose(&d.down, d.rank, inner);
                let delta = nn::linear(&d.up, shape[0], &down_t, None, d.rank, inner);
                v.iter_mut().zip(&delta).for_each(|(w, x)| *w += d.scale * x);
            }
        }
        Ok(v)
    }

    /// LoRA deltas of `name`, also when registered under the stored name of a plain alias.
    fn deltas_for(&self, name: &str) -> Option<&Vec<Delta>> {
        self.deltas.get(name).or_else(|| {
            let a = self.alias.get(name).filter(|a| a.rows.is_none() && !a.transpose)?;
            self.deltas.get(&a.src)
        })
    }

    /// `name` in its stored precision: f16/bf16 safetensors tensors stay 16-bit (merged LoRA deltas
    /// are re-encoded); everything else is f32.
    pub fn load_wt(&self, name: &str) -> Result<nn::Wt, String> {
        let alias = self.alias.get(name);
        let stored = alias.map_or(name, |a| a.src.as_str());
        let half = match self.dtypes.get(stored).map(String::as_str) {
            Some("F16") => nn::Wt::F16(Vec::new()),
            Some("BF16") => nn::Wt::BF16(Vec::new()),
            _ => return self.load(name).map(nn::Wt::F32),
        };
        let plain = alias.is_none_or(|a| a.rows.is_none() && !a.transpose);
        if !plain || self.deltas_for(name).is_some() {
            return Ok(half.like(self.load(name)?));
        }
        let raw = self.reader.load_tensor_quantized(stored).map_err(|e| format!("tensor '{stored}': {e}"))?;
        let bits = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        Ok(match half {
            nn::Wt::F16(_) => nn::Wt::F16(bits),
            _ => nn::Wt::BF16(bits),
        })
    }
}

const TOKENIZER_REPO: &str = "openai/clip-vit-large-patch14";
const SEQ_LEN: usize = 77;
const BOS_ID: u32 = 49406;
const EOS_ID: u32 = 49407;
/// SD 2.x / OpenCLIP towers pad with "!" (id 0), not EOS.
const PAD_ID: u32 = 0;
const HEAD_DIM: usize = 64;
const LN_EPS: f32 = 1e-5;

/// MLP activation: OpenAI CLIP (`quick_gelu`) or OpenCLIP (`gelu`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Act {
    Gelu,
    QuickGelu,
}

struct Linear {
    weight: nn::Wt,
    bias: Option<Vec<f32>>,
    in_f: usize,
    out_f: usize,
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.weight.linear(x, rows, self.bias.as_deref(), self.in_f, self.out_f)
    }
}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl Norm {
    fn forward(&self, x: &[f32], rows: usize, d: usize) -> Vec<f32> {
        nn::layer_norm(x, rows, d, &self.gamma, &self.beta, LN_EPS)
    }
}

struct EncoderLayer {
    layer_norm1: Norm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    layer_norm2: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// One prompt through the tower.
pub struct ClipOut {
    /// `[77 * d_model]`: the penultimate layer's raw output, or the final-layer-normed last layer.
    pub hidden: Vec<f32>,
    /// Final-layer-normed last layer at the first EOS (HF `pooler_output`), then `text_projection`
    /// when the checkpoint has one (`CLIPTextModelWithProjection.text_embeds`).
    pub pooled: Vec<f32>,
}

/// CLIP text transformer with its tokenizer: prompt -> `[77, d_model]` conditioning.
pub struct ClipTextTransformer {
    tokenizer: Tokenizer,
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    vocab: usize,
    d_model: usize,
    layers: Vec<EncoderLayer>,
    final_layer_norm: Norm,
    act: Act,
    pad_id: u32,
    text_projection: Option<Linear>,
    /// The Metal program (one shape: always 77 tokens).
    gpu: ByShape,
}

impl ClipTextTransformer {
    /// Loads the text tower under `prefix` (e.g. `text_model`) from `t` and the CLIP BPE tokenizer via `hub`
    /// (OpenCLIP-H as in SD 2.x: gelu, "!" padding, no projection).
    pub fn load(t: &Tensors, prefix: &str, hub: &HuggingFaceHub) -> Result<Self, String> {
        Self::load_with(t, prefix, hub, Act::Gelu, PAD_ID, None)
    }

    /// `load` with an explicit activation, pad token id and optional `text_projection` tensor name.
    pub fn load_with(
        t: &Tensors,
        prefix: &str,
        hub: &HuggingFaceHub,
        act: Act,
        pad_id: u32,
        projection: Option<&str>,
    ) -> Result<Self, String> {
        Self::from_parts(t, prefix, act, pad_id, projection, Self::fetch_tokenizer(hub)?)
    }

    fn fetch_tokenizer(hub: &HuggingFaceHub) -> Result<Tokenizer, String> {
        let tok_path = hub.fetch_file(TOKENIZER_REPO, "tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("Failed to load CLIP tokenizer {}: {e}", tok_path.display()))?;
        // Framing (BOS/EOS/pad to 77) is done by `frame_ids`; the file's own post-processor
        // (RobertaProcessing adding BOS/EOS) is bypassed via add_special_tokens=false.
        tokenizer
            .with_truncation(None)
            .map_err(|e| format!("CLIP tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        for (token, id) in [("<|startoftext|>", BOS_ID), ("<|endoftext|>", EOS_ID)] {
            if tokenizer.token_to_id(token) != Some(id) {
                return Err(format!(
                    "CLIP tokenizer {} maps '{token}' to {:?}, expected {id}",
                    tok_path.display(),
                    tokenizer.token_to_id(token)
                ));
            }
        }
        Ok(tokenizer)
    }

    pub(crate) fn from_parts(
        t: &Tensors,
        prefix: &str,
        act: Act,
        pad_id: u32,
        projection: Option<&str>,
        tokenizer: Tokenizer,
    ) -> Result<Self, String> {
        let tok_name = format!("{prefix}.embeddings.token_embedding.weight");
        let (vocab, d_model) = pt_dims2(t, &tok_name)?;
        if d_model % HEAD_DIM != 0 {
            return Err(format!(
                "CLIP d_model {d_model} (from '{tok_name}') is not a multiple of head_dim {HEAD_DIM}"
            ));
        }
        let token_embedding = tensor(t, &tok_name, &[vocab, d_model])?;
        let position_embedding = tensor(
            t,
            &format!("{prefix}.embeddings.position_embedding.weight"),
            &[SEQ_LEN, d_model],
        )?;

        let layer_prefix = format!("{prefix}.encoder.layers.");
        let num_layers = t
            .names()
            .filter_map(|n| n.strip_prefix(&layer_prefix)?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map(|i| i + 1)
            .ok_or_else(|| format!("No CLIP encoder layers found under '{layer_prefix}*'"))?;
        let (inter, _) = pt_dims2(t, &format!("{layer_prefix}0.mlp.fc1.weight"))?;

        let layers = (0..num_layers)
            .map(|i| -> Result<EncoderLayer, String> {
                let p = format!("{layer_prefix}{i}");
                Ok(EncoderLayer {
                    layer_norm1: norm(t, &format!("{p}.layer_norm1"), d_model)?,
                    q_proj: linear(t, &format!("{p}.self_attn.q_proj"), d_model, d_model)?,
                    k_proj: linear(t, &format!("{p}.self_attn.k_proj"), d_model, d_model)?,
                    v_proj: linear(t, &format!("{p}.self_attn.v_proj"), d_model, d_model)?,
                    out_proj: linear(t, &format!("{p}.self_attn.out_proj"), d_model, d_model)?,
                    layer_norm2: norm(t, &format!("{p}.layer_norm2"), d_model)?,
                    fc1: linear(t, &format!("{p}.mlp.fc1"), d_model, inter)?,
                    fc2: linear(t, &format!("{p}.mlp.fc2"), inter, d_model)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let final_layer_norm = norm(t, &format!("{prefix}.final_layer_norm"), d_model)?;
        let text_projection = match projection {
            Some(name) => {
                let (out_f, in_f) = pt_dims2(t, name)?;
                if in_f != d_model {
                    return Err(format!("CLIP '{name}' is [{out_f}, {in_f}], expected input {d_model}"));
                }
                Some(Linear { weight: t.load_wt(name)?, bias: None, in_f, out_f })
            }
            None => None,
        };

        let tok_vocab = tokenizer.get_vocab_size(true);
        if tok_vocab > vocab {
            return Err(format!(
                "CLIP tokenizer vocab {tok_vocab} exceeds token_embedding rows {vocab}"
            ));
        }

        Ok(Self {
            tokenizer,
            token_embedding,
            position_embedding,
            vocab,
            d_model,
            layers,
            final_layer_norm,
            act,
            pad_id,
            text_projection,
            gpu: ByShape::default(),
        })
    }

    /// A diffusers `text_encoder*/` (`CLIPTextModel` / `CLIPTextModelWithProjection`, safetensors)
    /// with its `tokenizer*/` (pad token from `special_tokens_map.json` / `tokenizer_config.json`;
    /// the BPE itself is the shared CLIP vocabulary).
    pub fn load_diffusers(hub: &HuggingFaceHub, text_encoder_dir: &Path, tokenizer_dir: &Path) -> Result<Self, String> {
        let cfg = read_json(&text_encoder_dir.join("config.json"))?;
        let act = match cfg["hidden_act"].as_str() {
            Some("quick_gelu") => Act::QuickGelu,
            Some("gelu") => Act::Gelu,
            other => return Err(format!("{}: unsupported CLIP hidden_act {other:?}", text_encoder_dir.display())),
        };
        if cfg["num_attention_heads"].as_u64().zip(cfg["hidden_size"].as_u64()).is_some_and(|(h, d)| d != h * 64) {
            return Err(format!("{}: CLIP head dim is not {HEAD_DIM}", text_encoder_dir.display()));
        }
        let pad = ["special_tokens_map.json", "tokenizer_config.json"]
            .iter()
            .filter_map(|f| read_json(&tokenizer_dir.join(f)).ok())
            .find_map(|j| {
                let p = &j["pad_token"];
                p.as_str().or_else(|| p["content"].as_str()).map(str::to_string)
            })
            .ok_or_else(|| format!("{}: no pad_token", tokenizer_dir.display()))?;
        let pad_id = match pad.as_str() {
            "<|endoftext|>" => EOS_ID,
            "!" => PAD_ID,
            other => return Err(format!("{}: unsupported CLIP pad_token {other:?}", tokenizer_dir.display())),
        };
        let path = crate::model::hub::component_weights(text_encoder_dir)?;
        let reader = SafetensorsWeightReader::new(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let t = Tensors::safetensors(&reader)?;
        let projection = t.shape("text_projection.weight").map(|_| "text_projection.weight");
        Self::load_with(&t, "text_model", hub, act, pad_id, projection)
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Width of [`ClipOut::pooled`].
    pub fn pooled_dim(&self) -> usize {
        self.text_projection.as_ref().map_or(self.d_model, |p| p.out_f)
    }

    /// Tokenizes and encodes `prompt`, returning the final-layer-normed hidden states `[77 * d_model]`.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>, String> {
        self.encode(prompt, false).map(|o| o.hidden)
    }

    /// Tokenizes and encodes `prompt` (on Metal when the process is GPU-only, else the native CPU tower).
    pub fn encode(&self, prompt: &str, penultimate: bool) -> Result<ClipOut, String> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("CLIP tokenization failed: {e}"))?;
        let ids = frame_ids(encoding.get_ids(), self.pad_id);
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(format!("CLIP token id {id} out of vocab {}", self.vocab));
        }
        if gpu_mode() {
            self.encode_ids_gpu(&ids, penultimate)
        } else {
            self.encode_ids_cpu(&ids, penultimate)
        }
    }

    /// Native CPU tower over framed token ids.
    pub fn encode_ids_cpu(&self, ids: &[u32; SEQ_LEN], penultimate: bool) -> Result<ClipOut, String> {
        crate::hardware::cpu_compute::enter("clip::encode");
        let d = self.d_model;

        let mut h = vec![0.0f32; SEQ_LEN * d];
        for (pos, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let tok = &self.token_embedding[id * d..(id + 1) * d];
            let pe = &self.position_embedding[pos * d..(pos + 1) * d];
            for ((o, t), p) in h[pos * d..(pos + 1) * d].iter_mut().zip(tok).zip(pe) {
                *o = t + p;
            }
        }

        let heads = d / HEAD_DIM;
        let mut penult = None;
        for (i, l) in self.layers.iter().enumerate() {
            if penultimate && i + 1 == self.layers.len() {
                penult = Some(h.clone());
            }
            let x = l.layer_norm1.forward(&h, SEQ_LEN, d);
            let q = l.q_proj.forward(&x, SEQ_LEN);
            let k = l.k_proj.forward(&x, SEQ_LEN);
            let v = l.v_proj.forward(&x, SEQ_LEN);
            let attn = nn::attention(&q, &k, &v, SEQ_LEN, SEQ_LEN, heads, HEAD_DIM, true);
            add_assign(&mut h, &l.out_proj.forward(&attn, SEQ_LEN));

            let x = l.layer_norm2.forward(&h, SEQ_LEN, d);
            let mut m = l.fc1.forward(&x, SEQ_LEN);
            match self.act {
                Act::Gelu => nn::gelu(&mut m),
                Act::QuickGelu => m.iter_mut().for_each(|v| *v /= 1.0 + (-1.702 * *v).exp()),
            }
            add_assign(&mut h, &l.fc2.forward(&m, SEQ_LEN));
        }
        let last = self.final_layer_norm.forward(&h, SEQ_LEN, d);
        // frame_ids always places an EOS.
        let eos = ids.iter().position(|&t| t == EOS_ID).unwrap_or(SEQ_LEN - 1);
        let pooled = last[eos * d..(eos + 1) * d].to_vec();
        let pooled = match &self.text_projection {
            Some(p) => p.forward(&pooled, 1),
            None => pooled,
        };
        Ok(ClipOut { hidden: penult.unwrap_or(last), pooled })
    }

    /// The tower as a Metal program: inputs `ids [77]` and `eos [1]` (U32); outputs the penultimate
    /// hidden state, the final-normed hidden state and the pooled (projected) EOS row.
    fn build_gpu(&self) -> Result<GpuNet, String> {
        let d = self.d_model;
        let mut b = Gb::new();
        let ids = b.input_u32("ids", &[SEQ_LEN]);
        let eos = b.input_u32("eos", &[1]);
        let table = b.wf32("tok_emb", &[self.vocab, d], self.token_embedding.clone());
        let pos = b.wf32("pos_emb", &[SEQ_LEN, d], self.position_embedding.clone());
        let tok = b.op("embed", Op::Embedding, &[table, ids], &[SEQ_LEN, d]);
        let mut h = b.add(tok, pos);
        let mut penult = h;
        let lin = |b: &mut Gb, x: V, l: &Linear| b.linear(x, &l.weight, l.bias.as_deref(), l.in_f, l.out_f);
        for (i, l) in self.layers.iter().enumerate() {
            if i + 1 == self.layers.len() {
                penult = h;
            }
            let x = b.layer_norm(h, &l.layer_norm1.gamma, &l.layer_norm1.beta, LN_EPS);
            let (q, k, v) = (lin(&mut b, x, &l.q_proj), lin(&mut b, x, &l.k_proj), lin(&mut b, x, &l.v_proj));
            let a = b.op(
                "attn",
                Op::Attention { heads: d / HEAD_DIM, kv_heads: d / HEAD_DIM, causal: true, window: None, soft_cap: None, scale: None },
                &[q, k, v],
                &[SEQ_LEN, d],
            );
            let o = lin(&mut b, a, &l.out_proj);
            h = b.add(h, o);

            let x = b.layer_norm(h, &l.layer_norm2.gamma, &l.layer_norm2.beta, LN_EPS);
            let m = lin(&mut b, x, &l.fc1);
            let m = match self.act {
                Act::Gelu => b.gelu(m, false),
                Act::QuickGelu => {
                    // x * sigmoid(1.702 x) == x / (1 + exp(-1.702 x))
                    let s = b.shape(m).to_vec();
                    let z = b.op("qg.scale", Op::Unary { kind: UnaryKind::Scale(1.702) }, &[m], &s);
                    let z = b.op("qg.sigmoid", Op::Unary { kind: UnaryKind::Sigmoid }, &[z], &s);
                    b.mul(m, z)
                }
            };
            let f = lin(&mut b, m, &l.fc2);
            h = b.add(h, f);
        }
        let last = b.layer_norm(h, &self.final_layer_norm.gamma, &self.final_layer_norm.beta, LN_EPS);
        let row = b.op("eos_row", Op::Gather { axis: 0 }, &[last, eos], &[1, d]);
        let pooled = match &self.text_projection {
            Some(p) => lin(&mut b, row, p),
            None => row,
        };
        for v in [penult, last, pooled] {
            b.output(v);
        }
        b.compile()
    }

    /// [`Self::encode_ids_cpu`] on Metal.
    pub fn encode_ids_gpu(&self, ids: &[u32; SEQ_LEN], penultimate: bool) -> Result<ClipOut, String> {
        let net = self.gpu.get(&[0], || self.build_gpu())?;
        let eos = ids.iter().position(|&t| t == EOS_ID).unwrap_or(SEQ_LEN - 1) as u32;
        let mut out = net.run_t(vec![Tensor::U32(ids.to_vec()), Tensor::U32(vec![eos])])?.into_iter();
        let (penult, last, pooled) = (out.next().unwrap(), out.next().unwrap(), out.next().unwrap());
        Ok(ClipOut { hidden: if penultimate { penult } else { last }, pooled })
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))
}

/// `[BOS, ids.., EOS]`, truncated to 77 with EOS kept last, padded with `pad`.
fn frame_ids(ids: &[u32], pad: u32) -> [u32; SEQ_LEN] {
    let n = ids.len().min(SEQ_LEN - 2);
    let mut out = [pad; SEQ_LEN];
    out[0] = BOS_ID;
    out[1..=n].copy_from_slice(&ids[..n]);
    out[n + 1] = EOS_ID;
    out
}

fn add_assign(h: &mut [f32], delta: &[f32]) {
    for (a, b) in h.iter_mut().zip(delta) {
        *a += b;
    }
}

/// PyTorch `[rows, cols]` of a 2-D tensor.
fn pt_dims2(t: &Tensors, name: &str) -> Result<(usize, usize), String> {
    match t.shape(name) {
        Some(&[rows, cols]) => Ok((rows, cols)),
        Some(s) => Err(format!("CLIP tensor '{name}' has shape {s:?}, expected 2-D")),
        None => Err(format!("CLIP tensor '{name}' missing")),
    }
}

fn check_shape(t: &Tensors, name: &str, dims: &[usize]) -> Result<(), String> {
    let shape = t.shape(name).ok_or_else(|| format!("CLIP tensor '{name}' missing"))?;
    if shape != dims {
        return Err(format!("CLIP tensor '{name}' has shape {shape:?}, expected {dims:?}"));
    }
    Ok(())
}

/// Loads `name` after checking its shape equals PyTorch `dims`.
fn tensor(t: &Tensors, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
    check_shape(t, name, dims)?;
    t.load(name).map_err(|e| format!("CLIP {e}"))
}

fn linear(t: &Tensors, p: &str, in_f: usize, out_f: usize) -> Result<Linear, String> {
    let w = format!("{p}.weight");
    check_shape(t, &w, &[out_f, in_f])?;
    Ok(Linear {
        weight: t.load_wt(&w).map_err(|e| format!("CLIP {e}"))?,
        bias: Some(tensor(t, &format!("{p}.bias"), &[out_f])?),
        in_f,
        out_f,
    })
}

fn norm(t: &Tensors, p: &str, d: usize) -> Result<Norm, String> {
    Ok(Norm {
        gamma: tensor(t, &format!("{p}.weight"), &[d])?,
        beta: tensor(t, &format!("{p}.bias"), &[d])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_prompt_is_bos_ids_eos_then_pad() {
        let f = frame_ids(&[10, 20, 30], PAD_ID);
        assert_eq!(&f[..5], &[BOS_ID, 10, 20, 30, EOS_ID]);
        assert!(f[5..].iter().all(|&t| t == PAD_ID));
    }

    #[test]
    fn empty_prompt_is_bos_eos_then_pad() {
        let f = frame_ids(&[], PAD_ID);
        assert_eq!(&f[..2], &[BOS_ID, EOS_ID]);
        assert!(f[2..].iter().all(|&t| t == PAD_ID));
    }

    #[test]
    fn exactly_75_tokens_fill_without_padding() {
        let ids: Vec<u32> = (1..=75).collect();
        let f = frame_ids(&ids, PAD_ID);
        assert_eq!(f[0], BOS_ID);
        assert_eq!(&f[1..76], ids.as_slice());
        assert_eq!(f[76], EOS_ID);
    }

    #[test]
    fn long_prompt_truncates_keeping_eos_last() {
        let ids: Vec<u32> = (100..200).collect();
        let f = frame_ids(&ids, EOS_ID);
        assert_eq!(f[0], BOS_ID);
        assert_eq!(&f[1..76], &ids[..75]);
        assert_eq!(f[76], EOS_ID);
    }

    /// CPU tower vs Metal program, both activations, with and without a projection (f32: 1e-3).
    #[test]
    fn gpu_matches_cpu() {
        // Every graph compiled here reproduces `graph::shape::infer`.
        let _shapes = crate::graph::shape::CheckCompiled::new();
        use crate::model::diff_graph::testutil::rel_err;
        for (act, proj) in [(Act::QuickGelu, false), (Act::Gelu, true)] {
            let clip = crate::model::diff_fixtures::tiny_clip(act, proj);
            let ids = frame_ids(&[5, 17, 300, 999, 20000], PAD_ID);
            for penult in [true, false] {
                let (c, g) = (clip.encode_ids_cpu(&ids, penult).unwrap(), clip.encode_ids_gpu(&ids, penult).unwrap());
                let (eh, ep) = (rel_err(&g.hidden, &c.hidden), rel_err(&g.pooled, &c.pooled));
                eprintln!("clip {act:?} proj={proj} penult={penult}: hidden {eh:.2e} pooled {ep:.2e}");
                assert!(eh < 1e-3 && ep < 1e-3, "hidden {eh} pooled {ep}");
                assert_eq!(g.pooled.len(), clip.pooled_dim());
            }
        }
    }
}
