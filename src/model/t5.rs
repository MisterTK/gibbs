//! T5 encoder-decoders (`t5` v1.0 and v1.1 / FLAN / MADLAD, `mt5`, `umt5`) as a [`NativeLm`]:
//! [`NativeLm::prepare`] runs the encoder over the user's text (task prefixes like
//! `translate English to German: ` are part of it) and [`NativeLm::forward`] the decoder, whose
//! prompt is the single `decoder_start_token_id`. [`T5Encoder`] is the encoder alone, for
//! diffusers text encoders (T5-XXL `text_encoder_2`).
//!
//! CPU. bf16 safetensors stay mapped ([`Mat`]); other dtypes and containers (f32/f16
//! safetensors, PyTorch `.bin`, Flax, candle GGUF) are expanded to f32.
//!
//! Reference: transformers `models/t5/modeling_t5.py` and `models/umt5/modeling_umt5.py`.

use crate::graph::{DType, Graph, Op, Tensor, ValueId};
use crate::model::encdec_gpu::{self as eg, Bag, Dims, EncDec, Sig};
use crate::model::flux::{Store, Wt};
use crate::model::hub::ModelArtifacts;
use crate::model::native::{MediaInput, NativeLm, SeqState};
use crate::model::par::{par_for, par_rows, threads, OutPtr};
use crate::model::simd::{vector_dot_product as dot, vector_fused_scale_add, vector_softmax};
use crate::model::tokenizer::{ChatTemplateSpec, Tokenizer};
use crate::model::weights::{Mat, Weights};
use crate::model::{nn, ModelManifest, UniversalModelLoader};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// A `[rows, cols]` weight: bf16 borrowed from the mapped checkpoint, or f32.
enum Lin {
    Bf16(Mat),
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    /// A llama.cpp / ComfyUI `t5encoder` GGUF matrix, decoded per GEMM panel.
    Gguf(Wt),
}

/// Input rows at or above which an f32 linear runs the packed GEMM instead of row dot products.
const GEMM_ROWS: usize = 8;

impl Lin {
    /// x `[n, cols]` -> `[n, rows]`.
    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let (w, rows, cols) = match self {
            Lin::Bf16(m) => return crate::model::gemma4::ops::linear(x, n, m),
            Lin::Gguf(w) => {
                let (rows, cols) = w.dims();
                let mut out = vec![0.0f32; n * rows];
                nn::matmul_rows(x, n, cols, rows, None, &mut out, &|j, k0, d| {
                    w.decode(j, k0, d)
                });
                return out;
            }
            Lin::F32 { w, rows, cols } => (w, *rows, *cols),
        };
        if n >= GEMM_ROWS {
            return nn::linear(x, n, w, None, cols, rows);
        }
        let mut out = vec![0.0f32; n * rows];
        let per = rows.div_ceil(threads() * 4).max(16);
        let o = OutPtr::new(&mut out);
        par_for(rows.div_ceil(per), &|t| {
            let (j0, j1) = (t * per, ((t + 1) * per).min(rows));
            for r in 0..n {
                let xr = &x[r * cols..(r + 1) * cols];
                // SAFETY: columns j0..j1 of row r belong to task t.
                let dst = unsafe { o.slice(r * rows + j0, j1 - j0) };
                for (d, j) in dst.iter_mut().zip(j0..j1) {
                    *d = dot(&w[j * cols..(j + 1) * cols], xr);
                }
            }
        });
        out
    }

    fn row(&self, r: usize) -> Vec<f32> {
        match self {
            Lin::Bf16(m) => m
                .row(r)
                .iter()
                .map(|&b| crate::model::bf16_to_f32(b))
                .collect(),
            Lin::Gguf(w) => {
                let mut row = vec![0.0; w.dims().1];
                w.decode(r, 0, &mut row);
                row
            }
            Lin::F32 { w, cols, .. } => w[r * cols..(r + 1) * cols].to_vec(),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Lin::Bf16(m) => m.rows,
            Lin::Gguf(w) => w.dims().0,
            Lin::F32 { rows, .. } => *rows,
        }
    }

    fn is_bf16(&self) -> bool {
        matches!(self, Lin::Bf16(_))
    }
}

/// Where tensors come from: a mapped safetensors checkpoint (`true`: every matrix as bf16, f32/f16
/// rounded into an anonymous map; `false`: only bf16 ones, the rest f32), or a checkpoint read
/// into memory (PyTorch `.bin`, Flax, GGUF; decoded per tensor), or a llama.cpp `t5encoder` GGUF
/// left mapped (names translated by [`ggml_name`]).
enum Src<'a> {
    Map(&'a Weights, bool),
    Mem(&'a ModelManifest, &'a [u8]),
    Gguf(&'a Store),
}

/// The llama.cpp / ComfyUI `t5encoder` name of an HF encoder tensor (`enc.blk.N.attn_q.weight`).
fn ggml_name(n: &str) -> String {
    let block = |rest: &str| -> Option<String> {
        let (i, tail) = rest.split_once(".layer.")?;
        let (l, t) = tail.split_once('.')?;
        let t = match (l, t) {
            ("0", "layer_norm.weight") => "attn_norm.weight",
            ("0", "SelfAttention.q.weight") => "attn_q.weight",
            ("0", "SelfAttention.k.weight") => "attn_k.weight",
            ("0", "SelfAttention.v.weight") => "attn_v.weight",
            ("0", "SelfAttention.o.weight") => "attn_o.weight",
            ("0", "SelfAttention.relative_attention_bias.weight") => "attn_rel_b.weight",
            ("1", "layer_norm.weight") => "ffn_norm.weight",
            ("1", "DenseReluDense.wi_0.weight") => "ffn_gate.weight",
            ("1", "DenseReluDense.wi_1.weight") => "ffn_up.weight",
            ("1", "DenseReluDense.wo.weight") => "ffn_down.weight",
            _ => return None,
        };
        Some(format!("enc.blk.{i}.{t}"))
    };
    match n {
        "shared.weight" | "encoder.embed_tokens.weight" => "token_embd.weight".into(),
        "encoder.final_layer_norm.weight" => "enc.output_norm.weight".into(),
        _ => n
            .strip_prefix("encoder.block.")
            .and_then(block)
            .unwrap_or_else(|| n.into()),
    }
}

impl Src<'_> {
    fn has(&self, name: &str) -> bool {
        match self {
            Src::Map(w, _) => w.has(name),
            Src::Mem(m, _) => m.tensors.contains_key(name),
            Src::Gguf(s) => s.has(&ggml_name(name)),
        }
    }

    fn vec_n(&self, name: &str, n: usize) -> Result<Vec<f32>, String> {
        let v = match self {
            Src::Map(w, _) => w.vec(name)?,
            Src::Mem(m, d) => UniversalModelLoader::extract_tensor_from_manifest(m, d, name)?,
            Src::Gguf(s) => s.vec(&ggml_name(name))?,
        };
        if v.len() != n {
            return Err(format!(
                "tensor '{name}' has {} values, expected {n}",
                v.len()
            ));
        }
        Ok(v)
    }

    /// `name` as a `[rows, cols]` weight; mapped bf16 stays bf16.
    fn lin(&self, name: &str, rows: usize, cols: usize) -> Result<Lin, String> {
        if let Src::Gguf(s) = self {
            let w = s.mat(&ggml_name(name))?;
            if w.dims() != (rows, cols) {
                return Err(format!(
                    "tensor '{name}' is {:?}, expected [{rows}, {cols}]",
                    w.dims()
                ));
            }
            return Ok(Lin::Gguf(w));
        }
        if let Src::Map(w, to_bf16) = self {
            // `mat` rounds f16/f32 to bf16; without `to_bf16` only stored bf16 is taken as is.
            if let Some(m) = (*to_bf16 || w.is_bf16(name))
                .then(|| w.mat(name))
                .and_then(Result::ok)
            {
                if (m.rows, m.cols) != (rows, cols) {
                    return Err(format!(
                        "tensor '{name}' is [{}, {}], expected [{rows}, {cols}]",
                        m.rows, m.cols
                    ));
                }
                return Ok(Lin::Bf16(m));
            }
        }
        Ok(Lin::F32 {
            w: self.vec_n(name, rows * cols)?,
            rows,
            cols,
        })
    }

    /// The token embedding `[vocab, d_model]`: `shared`, or one of its tied copies (candle's
    /// MADLAD GGUFs keep only `decoder.embed_tokens`).
    fn embedding(&self, c: &Cfg, prefix: &str) -> Result<Lin, String> {
        let names = ["shared", "encoder.embed_tokens", "decoder.embed_tokens"]
            .map(|n| format!("{prefix}{n}.weight"));
        let name = names.iter().find(|n| self.has(n)).unwrap_or(&names[0]);
        self.lin(name, c.vocab, c.d_model)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Act {
    Relu,
    Gelu,
    GeluTanh,
    Silu,
}
#[derive(Clone, Copy)]
struct Cfg {
    vocab: usize,
    d_model: usize,
    d_kv: usize,
    heads: usize,
    d_ff: usize,
    layers: usize,
    dec_layers: usize,
    buckets: usize,
    max_distance: usize,
    eps: f32,
    gated: bool,
    act: Act,
}

impl Cfg {
    fn parse(c: &Value) -> Result<Self, String> {
        let num = |k: &str| -> Result<usize, String> {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json has no {k}"))
        };
        let opt = |k: &str, d: usize| c[k].as_u64().map_or(d, |v| v as usize);
        // HF T5Config: `gated-<act>` is a gated FF; "gated-gelu" means gelu_new (tanh).
        let proj = c["feed_forward_proj"].as_str().unwrap_or("relu");
        let (gated, act) = match proj.strip_prefix("gated-") {
            Some(a) => (true, a),
            None => (false, proj),
        };
        let act = match (act, gated) {
            ("relu", _) => Act::Relu,
            ("gelu", true) | ("gelu_new" | "gelu_pytorch_tanh", _) => Act::GeluTanh,
            ("gelu", false) => Act::Gelu,
            ("silu" | "swish", _) => Act::Silu,
            _ => return Err(format!("feed_forward_proj '{proj}' is not supported")),
        };
        let layers = num("num_layers")?;
        Ok(Self {
            vocab: num("vocab_size")?,
            d_model: num("d_model")?,
            d_kv: num("d_kv")?,
            heads: num("num_heads")?,
            d_ff: num("d_ff")?,
            layers,
            dec_layers: opt("num_decoder_layers", layers),
            buckets: opt("relative_attention_num_buckets", 32),
            max_distance: opt("relative_attention_max_distance", 128),
            eps: c["layer_norm_epsilon"].as_f64().unwrap_or(1e-6) as f32,
            gated,
            act,
        })
    }

    fn inner(&self) -> usize {
        self.heads * self.d_kv
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// An attention sublayer with the T5LayerNorm in front of it.
struct Attn {
    norm: Vec<f32>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
}

impl Attn {
    /// `layer.{l}.layer_norm` + `layer.{l}.{module}.{q,k,v,o}` under `b` (`...block.{i}.layer`).
    fn load(src: &Src, c: &Cfg, b: &str, l: usize, module: &str) -> Result<Self, String> {
        let (d, inner) = (c.d_model, c.inner());
        let p = format!("{b}.{l}.{module}");
        Ok(Self {
            norm: src.vec_n(&format!("{b}.{l}.layer_norm.weight"), d)?,
            q: src.lin(&format!("{p}.q.weight"), inner, d)?,
            k: src.lin(&format!("{p}.k.weight"), inner, d)?,
            v: src.lin(&format!("{p}.v.weight"), inner, d)?,
            o: src.lin(&format!("{p}.o.weight"), d, inner)?,
        })
    }
}

struct Block {
    attn: Attn,
    /// Relative-position bias `[buckets, heads]`: T5 has it on block 0 only (shared by every
    /// block), UMT5 on every block.
    rel: Option<Vec<f32>>,
    /// Decoder only: attention over the encoder output.
    cross: Option<Attn>,
    ff_norm: Vec<f32>,
    /// `wi` (`wi_0` when gated), `wi_1` when gated, `wo`.
    wi: Lin,
    wi1: Option<Lin>,
    wo: Lin,
}

/// One encoder or decoder stack.
struct Stack {
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
}

impl Stack {
    fn load(src: &Src, c: &Cfg, prefix: &str, decoder: bool) -> Result<Self, String> {
        let d = c.d_model;
        let (n, stack) = match decoder {
            true => (c.dec_layers, "decoder"),
            false => (c.layers, "encoder"),
        };
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let b = format!("{prefix}{stack}.block.{i}.layer");
            let rel_name = format!("{b}.0.SelfAttention.relative_attention_bias.weight");
            let rel = if src.has(&rel_name) {
                Some(src.vec_n(&rel_name, c.buckets * c.heads)?)
            } else if i == 0 {
                return Err(format!("tensor '{rel_name}' missing"));
            } else {
                None
            };
            let cross = match decoder {
                true => Some(Attn::load(src, c, &b, 1, "EncDecAttention")?),
                false => None,
            };
            let f = format!("{b}.{}", if decoder { 2 } else { 1 });
            let ff = |s: &str, rows, cols| {
                src.lin(&format!("{f}.DenseReluDense.{s}.weight"), rows, cols)
            };
            let (wi, wi1) = match c.gated {
                true => (ff("wi_0", c.d_ff, d)?, Some(ff("wi_1", c.d_ff, d)?)),
                false => (ff("wi", c.d_ff, d)?, None),
            };
            blocks.push(Block {
                attn: Attn::load(src, c, &b, 0, "SelfAttention")?,
                rel,
                cross,
                ff_norm: src.vec_n(&format!("{f}.layer_norm.weight"), d)?,
                wi,
                wi1,
                wo: ff("wo", d, c.d_ff)?,
            });
        }
        Ok(Self {
            blocks,
            final_norm: src.vec_n(&format!("{prefix}{stack}.final_layer_norm.weight"), d)?,
        })
    }

    /// Block `i`'s relative-position bias table.
    fn rel(&self, i: usize) -> &[f32] {
        let own = self.blocks[i].rel.as_deref();
        own.or(self.blocks[0].rel.as_deref()).unwrap_or_default()
    }
}

/// T5LayerNorm (RMS, no mean, no bias) of every `w.len()`-wide row.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    par_rows(&mut y, w.len(), &|_, r| {
        let inv = 1.0 / (dot(r, r) / r.len() as f32 + eps).sqrt();
        r.iter_mut().zip(w).for_each(|(v, g)| *v *= inv * g);
    });
    y
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// HF `_relative_position_bucket` for `rel = key_pos - query_pos` (float32 log, truncated).
fn bucket(rel: i64, bidirectional: bool, buckets: usize, max_distance: usize) -> usize {
    let (mut nb, mut ret) = (buckets as i64, 0);
    let n = if bidirectional {
        nb /= 2;
        if rel > 0 {
            ret += nb;
        }
        rel.abs()
    } else {
        (-rel).max(0)
    };
    let exact = nb / 2;
    if n < exact {
        return (ret + n) as usize;
    }
    let scale = ((max_distance as f64 / exact as f64).ln()) as f32;
    let large = exact + ((n as f32 / exact as f32).ln() / scale * (nb - exact) as f32) as i64;
    (ret + large.min(nb - 1)) as usize
}

/// softmax(q k^T + bias) v per head, unscaled (T5 folds 1/sqrt(d_kv) into its weights).
/// `q` `[nq, inner]` is at positions `q0..`, `k`/`v` `[nk, inner]` at `0..`; `rel` (bias table,
/// bidirectional) adds the bucketed relative-position bias; `causal` hides keys past each query.
#[allow(clippy::too_many_arguments)]
fn attend(
    c: &Cfg,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    nq: usize,
    nk: usize,
    q0: usize,
    rel: Option<(&[f32], bool)>,
    causal: bool,
) -> Vec<f32> {
    let (heads, dk, inner) = (c.heads, c.d_kv, c.inner());
    // bias[h][r], r = key - query + off >= 0.
    let off = q0 + nq - 1;
    let span = off + nk;
    let mut bias = Vec::new();
    if let Some((table, bidir)) = rel {
        bias = vec![0.0f32; heads * span];
        for r in 0..span {
            let b = bucket(r as i64 - off as i64, bidir, c.buckets, c.max_distance);
            for h in 0..heads {
                bias[h * span + r] = table[b * heads + h];
            }
        }
    }
    let mut out = vec![0.0f32; nq * inner];
    let o = OutPtr::new(&mut out);
    par_for(heads * nq, &|t| {
        let (h, i) = (t / nq, t % nq);
        let qi = &q[i * inner + h * dk..][..dk];
        let kn = if causal { (q0 + i + 1).min(nk) } else { nk };
        let mut s: Vec<f32> = (0..kn)
            .map(|j| dot(qi, &k[j * inner + h * dk..][..dk]))
            .collect();
        if !bias.is_empty() {
            let b = &bias[h * span + off - (q0 + i)..][..kn];
            add(&mut s, b);
        }
        vector_softmax(&mut s);
        // SAFETY: task t alone writes head h of query row i.
        let dst = unsafe { o.slice(i * inner + h * dk, dk) };
        for (j, &p) in s.iter().enumerate() {
            vector_fused_scale_add(dst, &v[j * inner + h * dk..][..dk], p);
        }
    });
    out
}

impl Block {
    /// The feed-forward sublayer, in place on `x` `[n, d_model]`.
    fn ff(&self, c: &Cfg, x: &mut [f32], n: usize) {
        let h = rms_norm(x, &self.ff_norm, c.eps);
        let mut a = self.wi.forward(&h, n);
        match c.act {
            Act::Relu => a.iter_mut().for_each(|v| *v = v.max(0.0)),
            Act::Gelu => nn::gelu(&mut a),
            Act::GeluTanh => nn::gelu_tanh(&mut a),
            Act::Silu => nn::silu(&mut a),
        }
        if let Some(wi1) = &self.wi1 {
            let g = wi1.forward(&h, n);
            a.iter_mut().zip(&g).for_each(|(a, g)| *a *= g);
        }
        add(x, &self.wo.forward(&a, n));
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// The T5 encoder stack: token ids -> final-normed hidden states.
pub struct T5Encoder {
    c: Cfg,
    embed: Lin,
    stack: Stack,
}

impl T5Encoder {
    /// Loads the encoder from `w` (`{prefix}shared.weight` or `{prefix}encoder.embed_tokens.weight`,
    /// `{prefix}encoder.block.N...`); `cfg` is the parsed `config.json`. bf16 matrices stay
    /// mapped; f32/f16 ones are rounded to bf16 (half their f32 size, resident).
    pub fn load(w: &Weights, cfg: &Value, prefix: &str) -> Result<Self, String> {
        Self::from_src(&Src::Map(w, true), Cfg::parse(cfg)?, prefix)
    }

    /// Loads the encoder of a llama.cpp / ComfyUI `t5encoder` GGUF (`enc.blk.N.*`); `cfg` is the
    /// model's HF `config.json`. Quantized matrices stay mapped and are decoded per GEMM panel.
    pub(crate) fn load_gguf(store: &Store, cfg: &Value) -> Result<Self, String> {
        Self::from_src(&Src::Gguf(store), Cfg::parse(cfg)?, "")
    }

    fn from_src(src: &Src, c: Cfg, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            embed: src.embedding(&c, prefix)?,
            stack: Stack::load(src, &c, prefix, false)?,
            c,
        })
    }

    pub fn d_model(&self) -> usize {
        self.c.d_model
    }

    /// Final-layer-normed hidden states `[ids.len(), d_model]`; every token attends to every
    /// other (no padding mask). Panics on ids past the vocabulary.
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        self.encode_masked(ids, ids.len())
    }

    /// [`Self::encode`] with HF `attention_mask = [1] * valid + [0] * (len - valid)`: keys at
    /// positions `>= valid` are hidden from every query (padding rows still get outputs).
    pub fn encode_masked(&self, ids: &[u32], valid: usize) -> Vec<f32> {
        let (c, n) = (&self.c, ids.len());
        assert!(valid > 0 && valid <= n, "valid {valid} of {n} ids");
        let mut x: Vec<f32> = ids
            .iter()
            .flat_map(|&t| self.embed.row(t as usize))
            .collect();
        for (i, b) in self.stack.blocks.iter().enumerate() {
            let a = &b.attn;
            let h = rms_norm(&x, &a.norm, c.eps);
            let (q, k, v) = (a.q.forward(&h, n), a.k.forward(&h, n), a.v.forward(&h, n));
            let rel = Some((self.stack.rel(i), true));
            // Keys 0..valid only: the masked tail is never read.
            let y = attend(c, &q, &k, &v, n, valid, 0, rel, false);
            add(&mut x, &a.o.forward(&y, n));
            b.ff(c, &mut x, n);
        }
        rms_norm(&x, &self.stack.final_norm, c.eps)
    }
}

// ---------------------------------------------------------------------------
// Encoder-decoder
// ---------------------------------------------------------------------------

/// Renders the latest message alone: the encoder input is the user's text as is.
const LAST_MESSAGE_TEMPLATE: &str =
    "{% for m in messages %}{% if loop.last %}{{ m['content'] }}{% endif %}{% endfor %}";

struct T5 {
    tok: Tokenizer,
    enc: T5Encoder,
    dec: Stack,
    /// Untied `lm_head`; `None` = the shared embedding, with the `d_model^-0.5` rescale.
    head: Option<Lin>,
    start: u32,
    eos: u32,
    ctx: usize,
    summary: String,
    /// The Metal encoder/decoder programs; `Some` on a GPU backend, whose weights moved into it
    /// (the CPU fields above are then hollow). `None` runs the native CPU forward.
    gpu: Option<EncDec>,
}

/// One CPU sequence: the encoder output projected to every decoder layer's cross K/V, and the
/// decoder self-attention cache.
struct Seq {
    n_enc: usize,
    cross: Vec<(Vec<f32>, Vec<f32>)>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    len: usize,
}

/// Loads a `t5` / `mt5` / `umt5` checkpoint (`config.json` already parsed as `cfg`): the Metal
/// programs unless the backend is the explicit CPU opt-in.
pub fn load(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let backend = crate::graph::avenues::resolved_backend(mode.into());
    if !matches!(backend, crate::hardware::Device::Cpu | crate::hardware::Device::Metal) {
        return Err(format!("T5 has no {backend} program; use metal or --backend cpu"));
    }
    let path = artifacts
        .weights_path
        .as_deref()
        .ok_or("no weights in the model directory")?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.to_string_lossy();
    let c = Cfg::parse(cfg)?;
    let mut m = if name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json") {
        crate::model::prequant::check(artifacts.config_path.as_deref(), dir)?;
        let w = Weights::open(path)?;
        T5::from_src(&Src::Map(&w, false), c, cfg, dir)?
    } else {
        let (m, data): (_, Box<dyn std::ops::Deref<Target = [u8]>>) = if name.ends_with(".gguf") {
            let (m, data) = crate::model::loaders::gguf::open(path)?;
            (m, Box::new(data))
        } else {
            let (m, data) = crate::model::loaders::read_checkpoint(path)?;
            (m, Box::new(data))
        };
        T5::from_src(&Src::Mem(&m, &data[..]), c, cfg, dir)?
    };
    if backend == crate::hardware::Device::Metal {
        m.gpu = Some(m.build_gpu().map_err(|e| format!("T5: {e}"))?);
    }
    Ok(Arc::new(m))
}

/// tokenizer.json (Unigram) when present, else the SentencePiece `spiece.model`. Generation stops
/// at generation_config.json's `eos_token_id`, else config.json's (`eos`), and is greedy unless
/// generation_config.json sets `do_sample` (transformers' seq2seq default).
fn load_tokenizer(dir: &Path, eos: u32) -> Result<Tokenizer, String> {
    let json = dir.join("tokenizer.json");
    let mut tok = if json.is_file() {
        crate::model::hf_loader::load_json_tokenizer(&json)?
    } else {
        let mut tok = crate::model::spm::load(&dir.join("spiece.model"))?;
        let generation = dir.join("generation_config.json");
        crate::model::hf_loader::apply_generation_config(&mut tok, &generation);
        tok
    };
    tok.stop_ids.get_or_insert_with(|| vec![eos]);
    tok.sampling_defaults.get_or_insert((0.0, 1.0));
    if tok.chat_template.is_none() {
        tok.chat_template = Some(ChatTemplateSpec {
            template: LAST_MESSAGE_TEMPLATE.into(),
            bos_token: String::new(),
            eos_token: "</s>".into(),
        });
    }
    Ok(tok)
}

impl T5 {
    fn from_src(src: &Src, c: Cfg, cfg: &Value, dir: &Path) -> Result<Self, String> {
        let id = |k: &str, d: u64| cfg[k].as_u64().unwrap_or(d) as u32;
        let (start, eos) = (id("decoder_start_token_id", 0), id("eos_token_id", 1));
        let tied = cfg["tie_word_embeddings"].as_bool().unwrap_or(true);
        let head = match tied {
            true => None,
            false => Some(src.lin("lm_head.weight", c.vocab, c.d_model)?),
        };
        let dec = Stack::load(src, &c, "", true)?;
        let enc = T5Encoder::from_src(src, c, "")?;
        let c = &enc.c;
        let summary = format!(
            "{} {} weights, {}+{} layers, d_model {}, {} heads, {} FF",
            cfg["model_type"].as_str().unwrap_or("t5"),
            if enc.embed.is_bf16() { "bf16" } else { "f32" },
            c.layers,
            c.dec_layers,
            c.d_model,
            c.heads,
            if c.gated { "gated" } else { "relu" },
        );
        Ok(Self {
            tok: load_tokenizer(dir, eos)?,
            dec,
            head,
            start,
            eos,
            // Relative positions have no hard limit; n_positions is the trained length.
            ctx: cfg["n_positions"].as_u64().unwrap_or(512) as usize,
            summary,
            enc,
            gpu: None,
        })
    }

    /// Moves the weights into the Metal encoder/decoder programs (the CPU fields are left
    /// hollow: the native forward must not run afterwards).
    fn build_gpu(&mut self) -> Result<EncDec, String> {
        let c = self.enc.c;
        let tied = self.head.is_none();
        let mut bag = Bag::default();
        // Embedding takes an f32 table; a bf16 one is widened here, and a tied bf16 head keeps the
        // bf16 matrix as its own weight.
        let head = match (&mut self.head, &mut self.enc.embed) {
            (Some(h), _) => {
                put_lin(&mut bag, "lm_head", h)?;
                "lm_head"
            }
            (None, Lin::F32 { .. }) => "embed",
            (None, Lin::Bf16(m)) => {
                bag.add("lm_head", vec![m.rows, m.cols], DType::BF16, Tensor::Mat(m.clone()));
                "lm_head"
            }
            (None, Lin::Gguf(_)) => return Err("GGUF matrices have no Metal program".into()),
        };
        match &mut self.enc.embed {
            Lin::F32 { w, rows, cols } => {
                bag.add("embed", vec![*rows, *cols], DType::F32, Tensor::F32(std::mem::take(w)))
            }
            Lin::Bf16(m) => bag.add("embed", vec![m.rows, m.cols], DType::F32, Tensor::F32(m.to_f32())),
            Lin::Gguf(_) => return Err("GGUF matrices have no Metal program".into()),
        }
        for (i, b) in self.enc.stack.blocks.iter_mut().enumerate() {
            put_block(&mut bag, &format!("enc.{i}"), b)?;
        }
        put_vec(&mut bag, "enc.final_ln", &mut self.enc.stack.final_norm);
        for (i, b) in self.dec.blocks.iter_mut().enumerate() {
            put_block(&mut bag, &format!("dec.{i}"), b)?;
        }
        put_vec(&mut bag, "dec.final_ln", &mut self.dec.final_norm);
        let dims = Dims {
            layers: c.dec_layers,
            inner: c.inner(),
            enc_cap: self.ctx,
            self_cap: self.ctx,
            row_offset: None,
            self_mask: true,
        };
        bag.add("kpos", vec![dims.self_cap], DType::U32, Tensor::U32((0..dims.self_cap as u32).collect()));
        let sig = bag.sig();
        let enc_rel: Vec<bool> = self.enc.stack.blocks.iter().map(|b| b.rel.is_some()).collect();
        let dec_rel: Vec<bool> = self.dec.blocks.iter().map(|b| b.rel.is_some()).collect();
        let dec = decoder_graph(&c, &sig, &dec_rel, &dims, head, tied);
        let build_enc = Box::new(move |n| encoder_graph(&c, &sig, &enc_rel, n));
        EncDec::new(dims, bag, build_enc, dec)
    }
}

fn put_lin(bag: &mut Bag, name: &str, l: &mut Lin) -> Result<(), String> {
    match l {
        Lin::Bf16(m) => bag.add(name, vec![m.rows, m.cols], DType::BF16, Tensor::Mat(m.clone())),
        Lin::F32 { w, rows, cols } => {
            bag.add(name, vec![*rows, *cols], DType::F32, Tensor::F32(std::mem::take(w)))
        }
        Lin::Gguf(_) => return Err("GGUF matrices have no Metal program".into()),
    }
    Ok(())
}

fn put_vec(bag: &mut Bag, name: &str, v: &mut Vec<f32>) {
    let n = v.len();
    bag.add(name, vec![n], DType::F32, Tensor::F32(std::mem::take(v)));
}

/// `{p}.ln1/q/k/v/o`, `.rel` (own bias table), `.xln/xq/xk/xv/xo` (decoder), `.ln2/wi/wi1/wo`.
fn put_block(bag: &mut Bag, p: &str, b: &mut Block) -> Result<(), String> {
    let attn = |bag: &mut Bag, a: &mut Attn, ln: &str, q: &str, k: &str, v: &str, o: &str| {
        put_vec(bag, &format!("{p}.{ln}"), &mut a.norm);
        for (n, l) in [(q, &mut a.q), (k, &mut a.k), (v, &mut a.v), (o, &mut a.o)] {
            put_lin(bag, &format!("{p}.{n}"), l)?;
        }
        Ok::<(), String>(())
    };
    attn(bag, &mut b.attn, "ln1", "q", "k", "v", "o")?;
    if let Some(x) = &mut b.cross {
        attn(bag, x, "xln", "xq", "xk", "xv", "xo")?;
    }
    if let Some(rel) = &mut b.rel {
        let n = rel.len();
        bag.add(&format!("{p}.rel"), vec![n], DType::F32, Tensor::F32(std::mem::take(rel)));
    }
    put_vec(bag, &format!("{p}.ln2"), &mut b.ff_norm);
    put_lin(bag, &format!("{p}.wi"), &mut b.wi)?;
    if let Some(w) = &mut b.wi1 {
        put_lin(bag, &format!("{p}.wi1"), w)?;
    }
    put_lin(bag, &format!("{p}.wo"), &mut b.wo)
}

fn rms(g: &mut Graph, sig: &Sig, c: &Cfg, name: String, x: ValueId, ln: &str, n: usize) -> ValueId {
    let w = sig.w(g, ln);
    eg::node(g, name, Op::RmsNorm { eps: c.eps }, &[x, w], &[n, c.d_model])
}

fn lin(g: &mut Graph, sig: &Sig, name: &str, x: ValueId, n: usize) -> ValueId {
    let w = sig.w(g, name);
    let rows = g.value(w).shape[0];
    eg::node(g, name.to_string(), Op::Linear, &[x, w], &[n, rows])
}

/// `x + wo(act(wi(norm(x))) [* wi1(norm(x))])` over `n` rows.
fn ffn(g: &mut Graph, sig: &Sig, c: &Cfg, p: &str, x: ValueId, n: usize) -> ValueId {
    let h = rms(g, sig, c, format!("{p}.ffn_norm"), x, &format!("{p}.ln2"), n);
    let a = lin(g, sig, &format!("{p}.wi"), h, n);
    let shape = [n, c.d_ff];
    let act = |g: &mut Graph, a| match c.act {
        Act::Relu => eg::node(g, format!("{p}.act"), Op::Relu, &[a], &shape),
        Act::Gelu => eg::node(g, format!("{p}.act"), Op::Gelu { tanh: false }, &[a], &shape),
        Act::GeluTanh => eg::node(g, format!("{p}.act"), Op::Gelu { tanh: true }, &[a], &shape),
        Act::Silu => eg::node(g, format!("{p}.act"), Op::Silu, &[a], &shape),
    };
    let a = if c.gated {
        let gate = lin(g, sig, &format!("{p}.wi1"), h, n);
        match c.act {
            Act::GeluTanh => eg::node(g, format!("{p}.glu"), Op::GeGlu, &[a, gate], &shape),
            Act::Silu => eg::node(g, format!("{p}.glu"), Op::SwiGlu, &[a, gate], &shape),
            Act::Relu | Act::Gelu => {
                let a = act(g, a);
                eg::node(g, format!("{p}.glu"), Op::Mul, &[a, gate], &shape)
            }
        }
    } else {
        act(g, a)
    };
    let o = lin(g, sig, &format!("{p}.wo"), a, n);
    eg::node(g, format!("{p}.ffn_res"), Op::Add, &[x, o], &[n, c.d_model])
}

/// Unscaled attention (T5 folds 1/sqrt(d_kv) into its weights) as `Attention` (which divides by
/// sqrt(head_dim)) over `q` pre-multiplied by sqrt(head_dim).
#[allow(clippy::too_many_arguments)]
fn attention(
    g: &mut Graph,
    c: &Cfg,
    p: &str,
    q: ValueId,
    k: ValueId,
    v: ValueId,
    bias: ValueId,
    nq: usize,
) -> ValueId {
    let shape = [nq, c.inner()];
    let qs = eg::scale(g, format!("{p}.qs"), q, (c.d_kv as f32).sqrt(), &shape);
    let op = Op::Attention {
        heads: c.heads,
        kv_heads: c.heads,
        causal: false,
        window: None,
        soft_cap: None,
    };
    eg::node(g, format!("{p}.attn"), op, &[qs, k, v, bias], &shape)
}

/// `[heads, nq, nk]` relative-position bias of block table `table`, plus the `[nk]` key mask.
#[allow(clippy::too_many_arguments)]
fn rel_bias(
    g: &mut Graph,
    sig: &Sig,
    c: &Cfg,
    p: &str,
    bidirectional: bool,
    q_pos: ValueId,
    k_pos: ValueId,
    mask: ValueId,
    (nq, nk): (usize, usize),
) -> ValueId {
    let table = sig.w(g, &format!("{p}.rel"));
    let op = Op::RelPosBias {
        buckets: c.buckets,
        max_distance: c.max_distance,
        bidirectional,
        heads: c.heads,
    };
    let shape = [c.heads, nq, nk];
    let b = eg::node(g, format!("{p}.relbias"), op, &[table, q_pos, k_pos], &shape);
    eg::node(g, format!("{p}.relmask"), Op::Add, &[b, mask], &shape)
}

/// Encoder over `n` (padded) source rows: ids, positions `0..n` and a `[n]` key mask in; the
/// cross K/V projection of every decoder layer (`xk`, `xv`, ...) out.
fn encoder_graph(c: &Cfg, sig: &Sig, rel: &[bool], n: usize) -> Graph {
    let d = c.d_model;
    let mut g = Graph::new();
    let ids = g.input("ids", vec![n], DType::U32);
    let pos = g.input("pos", vec![n], DType::U32);
    let mask = g.input("mask", vec![n], DType::F32);
    let table = sig.w(&mut g, "embed");
    let mut x = eg::node(&mut g, "embed", Op::Embedding, &[table, ids], &[n, d]);
    let mut bias = None;
    for i in 0..c.layers {
        let p = format!("enc.{i}");
        let h = rms(&mut g, sig, c, format!("{p}.attn_norm"), x, &format!("{p}.ln1"), n);
        let (q, k, v) = (
            lin(&mut g, sig, &format!("{p}.q"), h, n),
            lin(&mut g, sig, &format!("{p}.k"), h, n),
            lin(&mut g, sig, &format!("{p}.v"), h, n),
        );
        // Block 0's table serves every block that has none of its own.
        if rel[i] {
            bias = Some(rel_bias(&mut g, sig, c, &p, true, pos, pos, mask, (n, n)));
        }
        let a = attention(&mut g, c, &p, q, k, v, bias.expect("set above"), n);
        let o = lin(&mut g, sig, &format!("{p}.o"), a, n);
        x = eg::node(&mut g, format!("{p}.attn_res"), Op::Add, &[x, o], &[n, d]);
        x = ffn(&mut g, sig, c, &p, x, n);
    }
    let h = rms(&mut g, sig, c, "enc.final_norm".into(), x, "enc.final_ln", n);
    for l in 0..c.dec_layers {
        for kv in ["xk", "xv"] {
            let out = lin(&mut g, sig, &format!("dec.{l}.{kv}"), h, n);
            g.output(out);
        }
    }
    g
}

/// One decoder position: ids `[1]`, position `[1]`, the `[self_cap]` self mask and `[enc_cap]`
/// cross mask in, logits `[1, vocab]` out. Self-attention writes its K/V into `sk/sv.{l}` with a
/// `CachedAttention` and scores them (with the relative-position bias, which `CachedAttention`
/// has no operand for) with an `Attention` over the whole state: the cached op's own output is
/// dead weight kept alive for its state write.
// ponytail: the second attention runs over all `self_cap` rows; compile per cap bucket if that
// bandwidth ever rivals the weights'.
fn decoder_graph(c: &Cfg, sig: &Sig, rel: &[bool], dims: &Dims, head: &str, tied: bool) -> Graph {
    let (d, inner, cap) = (c.d_model, c.inner(), dims.self_cap);
    let mut g = Graph::new();
    let ids = g.input("ids", vec![1], DType::U32);
    let pos = g.input("pos", vec![1], DType::U32);
    let self_mask = g.input("self_mask", vec![cap], DType::F32);
    let cross_mask = g.input("cross_mask", vec![dims.enc_cap], DType::F32);
    let kpos = sig.w(&mut g, "kpos");
    let table = sig.w(&mut g, "embed");
    let mut x = eg::node(&mut g, "embed", Op::Embedding, &[table, ids], &[1, d]);
    let mut bias = None;
    for i in 0..c.dec_layers {
        let p = format!("dec.{i}");
        let h = rms(&mut g, sig, c, format!("{p}.attn_norm"), x, &format!("{p}.ln1"), 1);
        let (q, k, v) = (
            lin(&mut g, sig, &format!("{p}.q"), h, 1),
            lin(&mut g, sig, &format!("{p}.k"), h, 1),
            lin(&mut g, sig, &format!("{p}.v"), h, 1),
        );
        let (sk, sv) = eg::self_states(&mut g, i, dims);
        let write = Op::CachedAttention {
            heads: c.heads,
            kv_heads: c.heads,
            head_dim: c.d_kv,
            v_head_dim: c.d_kv,
            scale: 1.0,
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        };
        eg::node(&mut g, format!("{p}.kv_write"), write, &[q, k, v, sk, sv, pos], &[1, inner]);
        if rel[i] {
            bias = Some(rel_bias(&mut g, sig, c, &p, false, pos, kpos, self_mask, (1, cap)));
        }
        let a = attention(&mut g, c, &format!("{p}.self"), q, sk, sv, bias.expect("set above"), 1);
        let o = lin(&mut g, sig, &format!("{p}.o"), a, 1);
        x = eg::node(&mut g, format!("{p}.attn_res"), Op::Add, &[x, o], &[1, d]);

        let h = rms(&mut g, sig, c, format!("{p}.cross_norm"), x, &format!("{p}.xln"), 1);
        let q = lin(&mut g, sig, &format!("{p}.xq"), h, 1);
        let (xk, xv) = eg::cross_states(&mut g, i, dims);
        let a = attention(&mut g, c, &format!("{p}.cross"), q, xk, xv, cross_mask, 1);
        let o = lin(&mut g, sig, &format!("{p}.xo"), a, 1);
        x = eg::node(&mut g, format!("{p}.cross_res"), Op::Add, &[x, o], &[1, d]);
        x = ffn(&mut g, sig, c, &p, x, 1);
    }
    let mut h = rms(&mut g, sig, c, "dec.final_norm".into(), x, "dec.final_ln", 1);
    if tied {
        h = eg::scale(&mut g, "tied_scale", h, (d as f32).powf(-0.5), &[1, d]);
    }
    let head = lin(&mut g, sig, head, h, 1);
    g.output(head);
    g
}

impl NativeLm for T5 {
    fn backend_label(&self) -> String {
        match &self.gpu {
            Some(gpu) => gpu.backend_label(),
            None => crate::scheduler::cpu_backend_label(),
        }
    }

    fn device(&self) -> crate::hardware::Device {
        match self.gpu {
            Some(_) => crate::hardware::Device::Metal,
            None => crate::hardware::Device::Cpu,
        }
    }

    fn summary(&self) -> String {
        self.summary.clone()
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn context_len(&self) -> usize {
        self.ctx
    }

    fn prepare(
        &self,
        mut ids: Vec<u32>,
        media: &[MediaInput],
    ) -> Result<(Vec<u32>, SeqState), String> {
        if !media.is_empty() {
            return Err("T5 takes text only (no image or audio input)".into());
        }
        // HF's T5 tokenizer ends the encoder input with </s>.
        if ids.last() != Some(&self.eos) {
            ids.push(self.eos);
        }
        let vocab = self.enc.embed.rows();
        if let Some(t) = ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!(
                "token id {t} is outside the {vocab}-entry vocabulary"
            ));
        }
        if let Some(gpu) = &self.gpu {
            return Ok((vec![self.start], Box::new(gpu.start(&ids)?)));
        }
        crate::hardware::cpu_compute::enter("t5::prepare");
        let h = self.enc.encode(&ids);
        let n = ids.len();
        let cross = self
            .dec
            .blocks
            .iter()
            .map(|b| {
                let x = b
                    .cross
                    .as_ref()
                    .expect("decoder blocks have cross-attention");
                (x.k.forward(&h, n), x.v.forward(&h, n))
            })
            .collect();
        let seq = Seq {
            n_enc: n,
            cross,
            kv: vec![(Vec::new(), Vec::new()); self.dec.blocks.len()],
            len: 0,
        };
        Ok((vec![self.start], Box::new(seq)))
    }

    fn prompt_len(&self, _ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        match media.is_empty() {
            true => Ok(1),
            false => Err("T5 takes text only (no image or audio input)".into()),
        }
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        if let Some(gpu) = &self.gpu {
            let s = state
                .downcast_mut::<eg::Seq>()
                .ok_or("sequence state is not a T5 state")?;
            if start_pos > s.len {
                return Err(format!(
                    "decoder position {start_pos} is past the {} cached",
                    s.len
                ));
            }
            // Rewinding is a length reset: later rows are masked and overwritten.
            s.len = start_pos;
            let vocab = self.enc.embed.rows();
            if let Some(t) = tokens.iter().find(|&&t| t as usize >= vocab) {
                return Err(format!("token id {t} is outside the vocabulary"));
            }
            let mut logits = Vec::new();
            for &t in tokens {
                logits = gpu.step(s, t)?;
            }
            return Ok(logits);
        }
        crate::hardware::cpu_compute::enter("t5::forward");
        let s = state
            .downcast_mut::<Seq>()
            .ok_or("sequence state is not a T5 state")?;
        if start_pos > s.len {
            return Err(format!(
                "decoder position {start_pos} is past the {} cached",
                s.len
            ));
        }
        let c = &self.enc.c;
        let (m, inner, embed) = (tokens.len(), c.inner(), &self.enc.embed);
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= embed.rows()) {
            return Err(format!("token id {t} is outside the vocabulary"));
        }
        let mut x: Vec<f32> = tokens.iter().flat_map(|&t| embed.row(t as usize)).collect();
        for (i, b) in self.dec.blocks.iter().enumerate() {
            let a = &b.attn;
            let h = rms_norm(&x, &a.norm, c.eps);
            let (ck, cv) = &mut s.kv[i];
            ck.truncate(start_pos * inner);
            cv.truncate(start_pos * inner);
            ck.extend(a.k.forward(&h, m));
            cv.extend(a.v.forward(&h, m));
            let q = a.q.forward(&h, m);
            let rel = Some((self.dec.rel(i), false));
            let y = attend(c, &q, ck, cv, m, start_pos + m, start_pos, rel, true);
            add(&mut x, &a.o.forward(&y, m));

            let xa = b
                .cross
                .as_ref()
                .expect("decoder blocks have cross-attention");
            let h = rms_norm(&x, &xa.norm, c.eps);
            let q = xa.q.forward(&h, m);
            let (ek, ev) = &s.cross[i];
            let y = attend(c, &q, ek, ev, m, s.n_enc, 0, None, false);
            add(&mut x, &xa.o.forward(&y, m));
            b.ff(c, &mut x, m);
        }
        s.len = start_pos + m;
        let last = &x[(m - 1) * c.d_model..];
        let mut h = rms_norm(last, &self.dec.final_norm, c.eps);
        let head = match &self.head {
            Some(head) => head,
            None => {
                let scale = (c.d_model as f32).powf(-0.5);
                h.iter_mut().for_each(|v| *v *= scale);
                embed
            }
        };
        Ok(head.forward(&h, 1))
    }
}

/// A random T5 (2+2 layers, d_model 32, 4 heads of 8, 8 buckets, 80 positions) for tests and the
/// GPU fixture; `per_block_rel` gives every block its own bias table (UMT5), `tied` the shared
/// embedding as head.
fn tiny(seed: u64, gated: bool, per_block_rel: bool, tied: bool) -> T5 {
    let c = Cfg {
        vocab: 40,
        d_model: 32,
        d_kv: 8,
        heads: 4,
        d_ff: 48,
        layers: 2,
        dec_layers: 2,
        buckets: 8,
        max_distance: 16,
        eps: 1e-6,
        gated,
        act: if gated { Act::GeluTanh } else { Act::Relu },
    };
    let mut r = Rng(seed);
    let (d, inner) = (c.d_model, c.inner());
    let attn = |r: &mut Rng| Attn {
        norm: r.norm(d),
        q: r.mat(inner, d),
        k: r.mat(inner, d),
        v: r.mat(inner, d),
        o: r.mat(d, inner),
    };
    let stack = |r: &mut Rng, decoder: bool, n: usize| {
        let mut blocks = Vec::new();
        for i in 0..n {
            blocks.push(Block {
                attn: attn(r),
                rel: (i == 0 || per_block_rel).then(|| r.vec(c.buckets * c.heads, 1.0)),
                cross: decoder.then(|| attn(r)),
                ff_norm: r.norm(d),
                wi: r.mat(c.d_ff, d),
                wi1: gated.then(|| r.mat(c.d_ff, d)),
                wo: r.mat(d, c.d_ff),
            });
        }
        Stack { blocks, final_norm: r.norm(d) }
    };
    let enc_stack = stack(&mut r, false, c.layers);
    let dec = stack(&mut r, true, c.dec_layers);
    let embed = r.mat(c.vocab, d);
    let head = (!tied).then(|| r.mat(c.vocab, d));
    T5 {
        tok: Tokenizer::from_tokens((0..c.vocab).map(|i| format!("t{i}")).collect()),
        enc: T5Encoder { c, embed, stack: enc_stack },
        dec,
        head,
        start: 0,
        eos: 1,
        ctx: 80,
        summary: "tiny t5".into(),
        gpu: None,
    }
}

/// The fixed LCG of the other tiny fixtures.
struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }
    fn mat(&mut self, rows: usize, cols: usize) -> Lin {
        Lin::F32 { w: self.vec(rows * cols, 0.4), rows, cols }
    }
    fn norm(&mut self, n: usize) -> Vec<f32> {
        self.vec(n, 0.3).iter().map(|x| 1.0 + x).collect()
    }
}

/// Tiny random T5 on Metal: encoder + prefill + two decode steps; panics unless every node of
/// every program ran on the GPU (`cpu_node_count() == 0`). The matrix test's `T5` fixture.
pub fn run_tiny_gpu_fixture() {
    let mut m = tiny(3, true, false, true);
    m.gpu = Some(m.build_gpu().expect("tiny T5 compiles on Metal"));
    let (ids, mut state) = m.prepare(vec![5, 9, 12, 7], &[]).expect("encoder runs on Metal");
    let mut tok = ids[0];
    for step in 0..3 {
        let logits = m.forward(&[tok], step, &mut state).expect("decoder runs on Metal");
        assert_eq!(logits.len(), 40);
        assert!(logits.iter().all(|v| v.is_finite()));
        tok = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
    }
    assert_eq!(m.gpu.as_ref().expect("built above").cpu_node_count(), 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_buckets_match_hf() {
        // transformers `_relative_position_bucket(rel, bidirectional, 32, 128)`.
        let enc: Vec<usize> = [-200, -20, -9, -8, -1, 0, 1, 7, 8, 16, 100]
            .iter()
            .map(|&r| bucket(r, true, 32, 128))
            .collect();
        assert_eq!(enc, [15, 10, 8, 8, 1, 0, 17, 23, 24, 26, 31]);
        let dec: Vec<usize> = [-200, -20, -16, -15, -1, 0, 3]
            .iter()
            .map(|&r| bucket(r, false, 32, 128))
            .collect();
        assert_eq!(dec, [31, 17, 16, 15, 1, 0, 0]);
    }

    /// [`T5Encoder::load`] rounds f32 matrices to bf16 (the diffusers text-encoder path); its
    /// output must track the f32 encoder the full model runs.
    #[test]
    fn bf16_encoder_tracks_f32_encoder() {
        let dir = Path::new("disk/models--google--flan-t5-small/snapshots/main");
        let Ok(w) = Weights::open(&dir.join("model.safetensors")) else {
            eprintln!("SKIP: no flan-t5-small checkpoint in {}", dir.display());
            return;
        };
        let cfg: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let bf = T5Encoder::load(&w, &cfg, "").unwrap();
        let f32 = T5Encoder::from_src(&Src::Map(&w, false), Cfg::parse(&cfg).unwrap(), "").unwrap();
        assert!(bf.embed.is_bf16() && !f32.embed.is_bf16());
        let ids: Vec<u32> = (0..40).map(|i| 100 + i * 37).chain([1]).collect();
        let (a, b) = (bf.encode(&ids), f32.encode(&ids));
        for (ra, rb) in a
            .chunks_exact(bf.d_model())
            .zip(b.chunks_exact(bf.d_model()))
        {
            let cos = dot(ra, rb) / (dot(ra, ra) * dot(rb, rb)).sqrt();
            assert!(cos > 0.99, "row cosine {cos}");
        }
    }

    fn argmax(l: &[f32]) -> u32 {
        (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum();
        (num / den).sqrt()
    }

    /// Native CPU forward vs the Metal programs on a tiny random T5, teacher-forced on the CPU's
    /// greedy tokens: logits agree to f32 rounding at every step, including a rewind.
    fn parity(gated: bool, per_block_rel: bool, tied: bool, src_len: usize) {
        parity_with(gated, per_block_rel, tied, src_len, false);
    }

    /// Rounds every matrix of `m` to bf16 (`Lin::Bf16`), as a bf16 safetensors checkpoint loads.
    fn to_bf16(m: &mut T5) {
        fn cv(l: &mut Lin) {
            if let Lin::F32 { w, rows, cols } = l {
                let spec = vec![("w".to_string(), vec![*rows, *cols])];
                let ws = Weights::from_f32(spec, |_| Ok(w.clone())).unwrap();
                *l = Lin::Bf16(ws.mat("w").unwrap());
            }
        }
        cv(&mut m.enc.embed);
        if let Some(h) = &mut m.head {
            cv(h);
        }
        let blocks = m.enc.stack.blocks.iter_mut().chain(m.dec.blocks.iter_mut());
        for b in blocks {
            for l in [&mut b.attn.q, &mut b.attn.k, &mut b.attn.v, &mut b.attn.o, &mut b.wi, &mut b.wo] {
                cv(l);
            }
            if let Some(x) = &mut b.cross {
                for l in [&mut x.q, &mut x.k, &mut x.v, &mut x.o] {
                    cv(l);
                }
            }
            if let Some(l) = &mut b.wi1 {
                cv(l);
            }
        }
    }

    fn parity_with(gated: bool, per_block_rel: bool, tied: bool, src_len: usize, bf16: bool) {
        let mut cpu = tiny(7, gated, per_block_rel, tied);
        let mut gpu = tiny(7, gated, per_block_rel, tied);
        if bf16 {
            to_bf16(&mut cpu);
            to_bf16(&mut gpu);
        }
        gpu.gpu = Some(gpu.build_gpu().expect("tiny T5 compiles on Metal"));
        let src: Vec<u32> = (0..src_len as u32).map(|i| 2 + (i * 7) % 30).collect();
        let _cpu_ok = crate::hardware::cpu_compute::allow();
        let (_, mut sc) = cpu.prepare(src.clone(), &[]).unwrap();
        let (_, mut sg) = gpu.prepare(src, &[]).unwrap();
        let mut tok = cpu.start;
        let mut toks = vec![tok];
        for step in 0..12 {
            let a = cpu.forward(&[tok], step, &mut sc).unwrap();
            let b = gpu.forward(&[tok], step, &mut sg).unwrap();
            let e = rel_err(&a, &b);
            assert!(e < 1e-3, "step {step}: rel err {e}");
            eprintln!("t5 tiny (gated {gated}, src {src_len}) step {step}: rel err {e:e}");
            tok = argmax(&a);
            toks.push(tok);
        }
        // Rewind to position 5 and replay two tokens.
        let a = cpu.forward(&toks[5..7], 5, &mut sc).unwrap();
        let b = gpu.forward(&toks[5..7], 5, &mut sg).unwrap();
        assert!(rel_err(&a, &b) < 1e-3, "rewind: {}", rel_err(&a, &b));
        assert_eq!(gpu.gpu.as_ref().unwrap().cpu_node_count(), 0);
    }

    #[test]
    fn metal_matches_cpu_relu_tied() {
        parity(false, false, true, 10);
    }

    #[test]
    fn metal_matches_cpu_gated_untied_per_block_bias() {
        parity(true, true, false, 10);
    }

    /// A source past the 64-row bucket runs the encoder at the model's full length.
    #[test]
    fn metal_matches_cpu_long_source() {
        parity(true, false, true, 70);
    }

    /// bf16 checkpoints (flan-t5, mT5, MADLAD) run bf16 Linear -> GeGlu and a tied bf16 head; the
    /// f32 cases above cannot reach those kernels.
    #[test]
    fn metal_matches_cpu_bf16_gated() {
        parity_with(true, false, true, 20, true);
        parity_with(true, true, false, 20, true);
    }

    #[test]
    fn tiny_fixture_runs_on_metal() {
        run_tiny_gpu_fixture();
    }
}
