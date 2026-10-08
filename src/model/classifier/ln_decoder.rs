//! GPT-2 and OPT decoder bodies for `*ForSequenceClassification` (transformers
//! `modeling_gpt2.py` / `modeling_opt.py`): learned absolute positions and biased LayerNorm
//! blocks. GPT-2 is pre-LN with Conv1D (`[in, out]`) weights; OPT is pre- or post-LN
//! (`do_layer_norm_before`) with optional `project_in`/`project_out` around the blocks and
//! positions offset by 2. f32 on the CPU, one uncached causal pass per input.

use super::{read_json, Checkpoint, DecoderBackbone, Lin};
use crate::model::hf_loader::load_json_tokenizer;
use crate::model::hub::HuggingFaceHub;
use crate::model::nn;
use crate::model::tokenizer::Tokenizer;
use serde_json::Value;

#[derive(Clone, Copy)]
enum Act {
    Gelu,
    GeluTanh,
    Relu,
}

type Norm = (Vec<f32>, Vec<f32>);

struct Block {
    ln1: Norm,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    ln2: Norm,
    fc1: Lin,
    fc2: Lin,
}

pub struct LnDecoder {
    tok: Tokenizer,
    /// `[vocab, e]`; `e` is OPT's `word_embed_proj_dim`, else the hidden size.
    embed: Vec<f32>,
    e: usize,
    /// `[positions, h]`, row `pos + pos_offset` for position `pos`.
    pos: Vec<f32>,
    pos_offset: usize,
    ctx: usize,
    project_in: Option<Lin>,
    project_out: Option<Lin>,
    blocks: Vec<Block>,
    final_ln: Option<Norm>,
    pre_ln: bool,
    act: Act,
    heads: usize,
    h: usize,
    eps: f32,
}

impl LnDecoder {
    /// The body of `repo_id` (`config.json` = `cfg`, `model_type` gpt2 or opt) and its tokenizer.
    pub fn load(hub: &HuggingFaceHub, repo_id: &str, cfg: &Value) -> Result<Self, String> {
        let ckpt = Checkpoint::open(hub, repo_id)?;
        let tok = tokenizer(hub, repo_id)?;
        match cfg["model_type"].as_str() {
            Some("gpt2") => gpt2(&ckpt, cfg, tok),
            Some("opt") => opt(&ckpt, cfg, tok),
            other => Err(format!(
                "{repo_id}: model_type {other:?} is not GPT-2 or OPT"
            )),
        }
        .map_err(|e| format!("{repo_id}: {e}"))
    }

    /// Hidden states `[n, h]` after the last block (before the final norm / `project_out`).
    fn blocks(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let (n, h, e) = (ids.len(), self.h, self.e);
        if n == 0 || n > self.ctx {
            return Err(format!("{n} tokens: the context holds 1..={}", self.ctx));
        }
        let vocab = self.embed.len() / e;
        let mut x = Vec::with_capacity(n * e);
        for &t in ids {
            let t = t as usize;
            if t >= vocab {
                return Err(format!("token id {t} is outside the {vocab}-row embedding"));
            }
            x.extend_from_slice(&self.embed[t * e..(t + 1) * e]);
        }
        if let Some(p) = &self.project_in {
            x = p.apply(&x, n);
        }
        for (i, row) in x.chunks_exact_mut(h).enumerate() {
            let p = i + self.pos_offset;
            nn::add_in_place(row, &self.pos[p * h..(p + 1) * h]);
        }
        let ln = |x: &[f32], (g, b): &Norm| nn::layer_norm(x, n, h, g, b, self.eps);
        for b in &self.blocks {
            let normed;
            let a_in = if self.pre_ln {
                normed = ln(&x, &b.ln1);
                &normed
            } else {
                &x
            };
            let att = nn::attention(
                &b.q.apply(a_in, n),
                &b.k.apply(a_in, n),
                &b.v.apply(a_in, n),
                n,
                n,
                self.heads,
                h / self.heads,
                true,
            );
            let mut y = b.out.apply(&att, n);
            nn::add_in_place(&mut y, &x);
            x = if self.pre_ln { y } else { ln(&y, &b.ln1) };
            let normed;
            let m_in = if self.pre_ln {
                normed = ln(&x, &b.ln2);
                &normed
            } else {
                &x
            };
            let mut f = b.fc1.apply(m_in, n);
            match self.act {
                Act::Gelu => nn::gelu(&mut f),
                Act::GeluTanh => nn::gelu_tanh(&mut f),
                Act::Relu => f.iter_mut().for_each(|v| *v = v.max(0.0)),
            }
            let mut y = b.fc2.apply(&f, n);
            nn::add_in_place(&mut y, &x);
            x = if self.pre_ln { y } else { ln(&y, &b.ln2) };
        }
        Ok(x)
    }
}

impl DecoderBackbone for LnDecoder {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    fn context_len(&self) -> usize {
        self.ctx
    }
    fn hidden(&self) -> usize {
        self.project_out.as_ref().map_or(self.h, |p| p.out_f)
    }
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let x = self.blocks(ids)?;
        let mut last = x[x.len() - self.h..].to_vec();
        if let Some((g, b)) = &self.final_ln {
            last = nn::layer_norm(&last, 1, self.h, g, b, self.eps);
        }
        Ok(match &self.project_out {
            Some(p) => p.apply(&last, 1),
            None => last,
        })
    }
}

fn act(cfg: &Value, default: &str) -> Result<Act, String> {
    match cfg["activation_function"].as_str().unwrap_or(default) {
        "gelu" => Ok(Act::Gelu),
        // Same tanh approximation, factored differently.
        "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => Ok(Act::GeluTanh),
        "relu" => Ok(Act::Relu),
        other => Err(format!("activation_function {other} is not supported")),
    }
}

fn usize_of(cfg: &Value, k: &str) -> Result<usize, String> {
    cfg[k]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("config.json has no {k}"))
}

/// `[rows, cols]` of tensor `name`.
fn dims(ckpt: &Checkpoint, name: &str) -> Result<(usize, usize), String> {
    match ckpt.shape(name) {
        Some(&[r, c]) => Ok((r, c)),
        other => Err(format!("tensor {name} has shape {other:?}, expected 2-D")),
    }
}

/// `w` `[3h, h]` and `b` `[3h]` split into three `[h, h]` layers.
fn split3(w: Vec<f32>, b: Vec<f32>, h: usize) -> [Lin; 3] {
    [0, 1, 2].map(|i| Lin {
        w: w[i * h * h..(i + 1) * h * h].to_vec(),
        b: Some(b[i * h..(i + 1) * h].to_vec()),
        in_f: h,
        out_f: h,
    })
}

fn gpt2(ckpt: &Checkpoint, cfg: &Value, tok: Tokenizer) -> Result<LnDecoder, String> {
    if cfg["scale_attn_by_inverse_layer_idx"].as_bool() == Some(true)
        || cfg["scale_attn_weights"].as_bool() == Some(false)
    {
        return Err("only 1/sqrt(head_dim) attention scaling is supported".into());
    }
    let p = if ckpt.has("transformer.wte.weight") {
        "transformer."
    } else {
        ""
    };
    let (vocab, h) = dims(ckpt, &format!("{p}wte.weight"))?;
    let (positions, _) = dims(ckpt, &format!("{p}wpe.weight"))?;
    // Conv1D `[in, out]` (x W + b) as a PyTorch-layout linear layer.
    let conv = |name: &str| -> Result<Lin, String> {
        let (in_f, out_f) = dims(ckpt, &format!("{name}.weight"))?;
        let w = ckpt.get(&format!("{name}.weight"), &[in_f, out_f])?;
        Ok(Lin {
            w: nn::transpose(&w, in_f, out_f),
            b: Some(ckpt.get(&format!("{name}.bias"), &[out_f])?),
            in_f,
            out_f,
        })
    };
    let blocks = (0..usize_of(cfg, "n_layer")?)
        .map(|l| {
            let at = |s: &str| format!("{p}h.{l}.{s}");
            let qkv = conv(&at("attn.c_attn"))?;
            if qkv.in_f != h || qkv.out_f != 3 * h {
                return Err(format!("{} is not [{h}, {}]", at("attn.c_attn"), 3 * h));
            }
            let [q, k, v] = split3(qkv.w, qkv.b.unwrap_or_default(), h);
            Ok(Block {
                ln1: ckpt.layer_norm(&at("ln_1"), h)?,
                q,
                k,
                v,
                out: conv(&at("attn.c_proj"))?,
                ln2: ckpt.layer_norm(&at("ln_2"), h)?,
                fc1: conv(&at("mlp.c_fc"))?,
                fc2: conv(&at("mlp.c_proj"))?,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(LnDecoder {
        tok,
        embed: ckpt.get(&format!("{p}wte.weight"), &[vocab, h])?,
        e: h,
        pos: ckpt.get(&format!("{p}wpe.weight"), &[positions, h])?,
        pos_offset: 0,
        ctx: positions,
        project_in: None,
        project_out: None,
        blocks,
        final_ln: Some(ckpt.layer_norm(&format!("{p}ln_f"), h)?),
        pre_ln: true,
        act: act(cfg, "gelu_new")?,
        heads: usize_of(cfg, "n_head")?,
        h,
        eps: cfg["layer_norm_epsilon"].as_f64().unwrap_or(1e-5) as f32,
    })
}

fn opt(ckpt: &Checkpoint, cfg: &Value, tok: Tokenizer) -> Result<LnDecoder, String> {
    if cfg["enable_bias"].as_bool() == Some(false)
        || cfg["layer_norm_elementwise_affine"].as_bool() == Some(false)
    {
        return Err("OPT without biases or LayerNorm affine weights is not supported".into());
    }
    let p = ["model.decoder.", "decoder."]
        .into_iter()
        .find(|p| ckpt.has(&format!("{p}embed_tokens.weight")))
        .ok_or("missing tensor model.decoder.embed_tokens.weight")?;
    let h = usize_of(cfg, "hidden_size")?;
    let (vocab, e) = dims(ckpt, &format!("{p}embed_tokens.weight"))?;
    let (positions, _) = dims(ckpt, &format!("{p}embed_positions.weight"))?;
    // OPTLearnedPositionalEmbedding has 2 rows before position 0.
    let ctx = positions
        .checked_sub(2)
        .ok_or("embed_positions has fewer than 2 rows")?;
    let proj = |name: &str, in_f| {
        let name = format!("{p}{name}");
        ckpt.has(&format!("{name}.weight"))
            .then(|| ckpt.linear(&name, in_f))
            .transpose()
    };
    let pre_ln = cfg["do_layer_norm_before"].as_bool().unwrap_or(true);
    let blocks = (0..usize_of(cfg, "num_hidden_layers")?)
        .map(|l| {
            let at = |s: &str| format!("{p}layers.{l}.{s}");
            Ok(Block {
                ln1: ckpt.layer_norm(&at("self_attn_layer_norm"), h)?,
                q: ckpt.linear(&at("self_attn.q_proj"), h)?,
                k: ckpt.linear(&at("self_attn.k_proj"), h)?,
                v: ckpt.linear(&at("self_attn.v_proj"), h)?,
                out: ckpt.linear(&at("self_attn.out_proj"), h)?,
                ln2: ckpt.layer_norm(&at("final_layer_norm"), h)?,
                fc1: ckpt.linear(&at("fc1"), h)?,
                fc2: ckpt.linear(&at("fc2"), usize_of(cfg, "ffn_dim")?)?,
            })
        })
        .collect::<Result<_, String>>()?;
    let project_in = proj("project_in", e)?;
    if project_in.is_none() && e != h {
        return Err(format!(
            "no project_in for embeddings of {e} into hidden {h}"
        ));
    }
    // Only pre-LN OPT keeps a final norm (`_remove_final_layer_norm` drops it for old ports).
    let final_ln = (pre_ln && cfg["_remove_final_layer_norm"].as_bool() != Some(true))
        .then(|| ckpt.layer_norm(&format!("{p}final_layer_norm"), h))
        .transpose()?;
    Ok(LnDecoder {
        tok,
        embed: ckpt.get(&format!("{p}embed_tokens.weight"), &[vocab, e])?,
        e,
        pos: ckpt.get(&format!("{p}embed_positions.weight"), &[positions, h])?,
        pos_offset: 2,
        ctx,
        project_in,
        project_out: proj("project_out", h)?,
        blocks,
        final_ln,
        pre_ln,
        act: act(cfg, "relu")?,
        heads: usize_of(cfg, "num_attention_heads")?,
        h,
        eps: 1e-5,
    })
}

/// tokenizer.json, else GPT2Tokenizer's `vocab.json` + `merges.txt` rebuilt as the byte-level BPE
/// its fast tokenizer is, with `tokenizer_config.json`'s special tokens and `add_bos_token`.
fn tokenizer(hub: &HuggingFaceHub, repo_id: &str) -> Result<Tokenizer, String> {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let json_err = match hub.fetch_file(repo_id, "tokenizer.json") {
        Ok(p) => return load_json_tokenizer(&p),
        Err(e) => e,
    };
    let file = |f: &str| {
        hub.fetch_file(repo_id, f)
            .map_err(|e| format!("{repo_id}: no tokenizer.json ({json_err}) or {f} ({e})"))
    };
    let (vocab, merges) = (file("vocab.json")?, file("merges.txt")?);
    let cfg = match file("tokenizer_config.json") {
        Ok(p) => read_json(&p)?,
        Err(_) => Value::Null,
    };
    let bpe = tokenizers::models::bpe::BPE::from_file(
        &vocab.to_string_lossy(),
        &merges.to_string_lossy(),
    )
    .build()
    .map_err(|e| format!("{}: {e}", vocab.display()))?;
    let mut t = tokenizers::Tokenizer::new(bpe);
    let prefix = cfg["add_prefix_space"].as_bool().unwrap_or(false);
    t.with_pre_tokenizer(Some(ByteLevel::new(prefix, true, true)));
    t.with_decoder(Some(ByteLevel::default()));
    // Special tokens are plain strings or AddedToken objects `{"content": ...}`.
    let content = |v: &Value| {
        v.as_str()
            .or_else(|| v["content"].as_str())
            .map(String::from)
    };
    let specials: Vec<tokenizers::AddedToken> = cfg["added_tokens_decoder"]
        .as_object()
        .into_iter()
        .flat_map(|m| m.values())
        .filter_map(content)
        .map(|s| tokenizers::AddedToken::from(s, true))
        .collect();
    t.add_special_tokens(&specials);
    let mut tok = Tokenizer::from_json_str(&t.to_string(false).map_err(|e| e.to_string())?)?;
    if cfg["add_bos_token"].as_bool() == Some(true) {
        let bos = content(&cfg["bos_token"]).ok_or("add_bos_token without a bos_token")?;
        tok.bos = Some(
            *tok.vocab
                .get(&bos)
                .ok_or_else(|| format!("{repo_id}: bos_token {bos} is not in vocab.json"))?,
        );
    }
    Ok(tok)
}
