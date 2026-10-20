//! BART-family encoder-decoder translation and summarization models (Marian / opus-mt,
//! M2M100 + NLLB, MBart-50, FSMT / wmt19, BART / distilbart) as a [`NativeLm`].
//!
//! Prompt convention: the text is the source. Language-coded models take a leading
//! `>>tgt<<` tag naming the target language (`>>fra<<` for Marian multi-target models, which
//! already use it; `>>fra_Latn<<` for NLLB, `>>fr<<` for M2M100, `>>fr_XX<<` for MBart-50) and an
//! optional `<<src>>` tag naming the source language (default: tokenizer_config `src_lang`, else
//! English). A multi-target model without a target tag is an error saying how to give one.
//!
//! [`NativeLm::prepare`] runs the encoder and the model's own search (generation_config
//! `num_beams`, `max_length`, `min_length`, `length_penalty`, `no_repeat_ngram_size`,
//! `bad_words_ids`, forced BOS/EOS), as transformers' `generate` does; `forward` then replays the
//! chosen hypothesis as one-hot logits, so every sampler setting yields the model's translation.

use crate::graph::{DType, Graph, Op, Tensor, ValueId};
use crate::model::encdec_gpu::{self as eg, Bag, Dims, EncDec, Sig};
use crate::model::hub::ModelArtifacts;
use crate::model::native::{MediaInput, NativeLm, PrepareCtx, SeqState};
use crate::model::nn;
use crate::model::par::{par_for, OutPtr};
use crate::model::spm::SpmModel;
use crate::model::tokenizer::{byte_to_gpt2_unicode, ChatTemplateSpec, Tokenizer};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::{byte_level::ByteLevel, whitespace::WhitespaceSplit};

/// `architectures[0]` values [`load`] runs.
pub const ARCHITECTURES: &[&str] = &[
    "MarianMTModel",
    "M2M100ForConditionalGeneration",
    "MBartForConditionalGeneration",
    "FSMTForConditionalGeneration",
    "BartForConditionalGeneration",
];

/// Renders the last user message: the source text.
const LAST_USER_TEMPLATE: &str = "{%- set ns = namespace(t='') -%}{%- for m in messages -%}\
{%- if m.role == 'user' -%}{%- if m.content is string -%}{%- set ns.t = m.content -%}\
{%- else -%}{%- set ns.t = m.content | selectattr('type', 'equalto', 'text') | map(attribute='text') | join('\n') -%}\
{%- endif -%}{%- endif -%}{%- endfor -%}{{ ns.t }}";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    Marian,
    M2m100,
    MBart,
    Fsmt,
    Bart,
}

/// How target pieces print: SentencePiece `▁`, GPT-2 byte-level `Ġ`, or subword-nmt `</w>`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pieces {
    Spm,
    Gpt2,
    Fsmt,
}

struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Lin {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        linear(x, rows, &self.w, Some(&self.b), self.b.len())
    }
}

/// `x [rows, in] · w[out, in]ᵀ + bias`. Decode steps (a few beams) stream the weight rows once
/// on the shared pool instead of packing the whole matrix into GEMM panels every step.
fn linear(x: &[f32], rows: usize, w: &[f32], bias: Option<&[f32]>, out: usize) -> Vec<f32> {
    let inp = w.len() / out;
    if rows > 8 {
        return nn::linear(x, rows, w, bias, inp, out);
    }
    let mut y = vec![0.0f32; rows * out];
    let o = OutPtr::new(&mut y);
    const CH: usize = 256;
    par_for(out.div_ceil(CH), &|t| {
        let (j0, j1) = (t * CH, ((t + 1) * CH).min(out));
        // SAFETY: task t owns columns j0..j1 of every row.
        let mut dst: Vec<&mut [f32]> = (0..rows)
            .map(|r| unsafe { o.slice(r * out + j0, j1 - j0) })
            .collect();
        let tail = inp - inp % 8;
        for j in j0..j1 {
            let wj = &w[j * inp..(j + 1) * inp];
            for (r, d) in dst.iter_mut().enumerate() {
                let xr = &x[r * inp..(r + 1) * inp];
                let mut acc = [0.0f32; 8];
                for (a, b) in wj.chunks_exact(8).zip(xr.chunks_exact(8)) {
                    for k in 0..8 {
                        acc[k] += a[k] * b[k];
                    }
                }
                let s: f32 = acc.iter().sum::<f32>()
                    + wj[tail..]
                        .iter()
                        .zip(&xr[tail..])
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                d[j - j0] = s + bias.map_or(0.0, |b| b[j]);
            }
        }
    });
    y
}

struct Ln {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Ln {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let d = self.g.len();
        nn::layer_norm(x, x.len() / d, d, &self.g, &self.b, 1e-5)
    }
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln: Ln,
}

struct Layer {
    sa: Attn,
    /// Decoder layers only.
    ca: Option<Attn>,
    fc1: Lin,
    fc2: Lin,
    ff_ln: Ln,
}

struct Stack {
    embed: Arc<Vec<f32>>,
    /// Position rows; position `p` is row `p + pos_offset`.
    pos: Vec<f32>,
    pos_offset: usize,
    ln_emb: Option<Ln>,
    layers: Vec<Layer>,
    final_ln: Option<Ln>,
    heads: usize,
}

/// generation_config.json (falling back to config.json).
struct Gen {
    beams: usize,
    max_len: usize,
    min_len: usize,
    length_penalty: f32,
    no_repeat: usize,
    early_stopping: bool,
    renormalize: bool,
    forced_eos: Option<u32>,
    bad: Vec<u32>,
}

enum Source {
    /// tokenizer.json, SentencePiece segmenters, byte-level BPE: ids are model ids.
    Hf(tokenizers::Tokenizer),
    /// FSMT: Moses tokenization, then subword-nmt BPE (`</w>` word ends).
    Moses {
        bpe: tokenizers::Tokenizer,
        lang: String,
        /// Ids at or above `known` are merge intermediates, not vocab entries: `unk`.
        known: u32,
        unk: u32,
    },
}

pub struct Seq2Seq {
    family: Family,
    d: usize,
    enc: Stack,
    dec: Stack,
    lm_head: Arc<Vec<f32>>,
    logits_bias: Option<Vec<f32>>,
    vocab: usize,
    act: fn(&mut [f32]),
    pre_ln: bool,
    embed_scale: f32,
    max_pos: usize,
    gen: Gen,
    eos: u32,
    bos: u32,
    decoder_start: u32,
    forced_bos: Option<u32>,
    source: Source,
    /// Decoding side: target pieces, plus display aliases at `alias + id`.
    tok: Tokenizer,
    alias: u32,
    pieces: Pieces,
    /// Raw target piece strings by id (FSMT spacing reads `</w>`).
    raw: HashMap<u32, String>,
    /// Language tokens (NLLB / M2M100 / MBart-50): tag code -> token id.
    langs: Vec<(String, u32)>,
    src_default: Option<u32>,
    tgt_default: Option<u32>,
    /// MBart (en-ro style) puts `</s> src` after the source and starts decoding at the target.
    legacy_mbart: bool,
    /// Marian multi-target models: their `>>xx<<` target tokens.
    marian_targets: Vec<String>,
    tgt_lang: String,
    summary: String,
    act_kind: ActKind,
    /// The Metal encoder/decoder programs; `Some` on a GPU backend, whose weights moved into it
    /// (`enc`, `dec`, `lm_head` are then hollow). `None` runs the native CPU forward.
    gpu: Option<EncDec>,
}

/// The FFN activation, by name for the Metal graphs (`Seq2Seq::act` is its CPU function).
#[derive(Clone, Copy)]
enum ActKind {
    Relu,
    Gelu,
    GeluTanh,
    Silu,
}

/// Beam search (`num_beams` > 1): the chosen output, replayed by `forward`.
struct Replay {
    prompt: usize,
    out: Vec<u32>,
}

/// The decoder side of one request: the encoder output as cross K/V plus one self-attention
/// cache per beam, on the CPU or in Metal state buffers.
enum Run {
    Cpu {
        cross: Vec<(Vec<f32>, Vec<f32>)>,
        n_src: usize,
        caches: Vec<Cache>,
    },
    Gpu(Vec<eg::Seq>),
}

/// Greedy configs: the decoder runs step by step in `forward`, so sampling, streaming and
/// cancellation work as for any model.
struct Live {
    run: Run,
    /// Decoder ids so far (prompt included).
    seq: Vec<u32>,
    prompt: usize,
}

fn read_json(dir: &Path, name: &str) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(dir.join(name)).ok()?).ok()
}

fn json_vocab(dir: &Path, name: &str) -> Result<HashMap<String, u32>, String> {
    let v = read_json(dir, name).ok_or_else(|| format!("{name} missing or invalid"))?;
    let map = v
        .as_object()
        .ok_or_else(|| format!("{name} is not an object"))?;
    Ok(map
        .iter()
        .filter_map(|(k, v)| Some((k.clone(), v.as_u64()? as u32)))
        .collect())
}

fn merges(dir: &Path) -> Result<Vec<(String, String)>, String> {
    let text =
        std::fs::read_to_string(dir.join("merges.txt")).map_err(|e| format!("merges.txt: {e}"))?;
    Ok(text
        .lines()
        .filter(|l| !l.starts_with("#version") && !l.trim().is_empty())
        .filter_map(|l| {
            let (a, b) = l.split_once(' ')?;
            Some((a.to_string(), b.to_string()))
        })
        .collect())
}

fn bpe_vocab(v: &HashMap<String, u32>) -> Vocab {
    v.iter().map(|(k, &v)| (k.clone(), v)).collect()
}

/// A tokenizer whose ids are the text's bytes: `prepare` gets the raw text back and runs the
/// model's own source tokenization (language tags, Moses, SentencePiece) itself.
fn byte_passthrough() -> Result<tokenizers::Tokenizer, String> {
    let vocab: HashMap<String, u32> = (0..=255u8)
        .map(|b| (byte_to_gpt2_unicode(b).to_string(), b as u32))
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(bpe_vocab(&vocab), Vec::new())
        .build()
        .map_err(|e| e.to_string())?;
    let mut t = tokenizers::Tokenizer::new(bpe);
    t.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    Ok(t)
}

/// Moses entity escapes (sacremoses `escape_xml`), reversed for display.
fn unescape(s: &str) -> String {
    s.replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&#124;", "|")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&amp;", "&")
}

/// sacremoses `MosesTokenizer.tokenize(escape=True, aggressive_dash_splits=True)` after FSMT's
/// punctuation normalisation, joined by spaces.
// ponytail: the common rules only (unicode punctuation, dashes, commas, apostrophes, final
// periods with an English abbreviation list); port the per-language nonbreaking-prefix files if
// abbreviations split wrongly.
fn moses_tokenize(text: &str, lang: &str) -> String {
    let mut t = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '“' | '”' | '„' | '«' | '»' | '《' | '》' | '「' | '」' => t.push('"'),
            '‘' | '’' | '‚' | '´' => t.push('\''),
            '–' | '━' => t.push('-'),
            '—' => t.push_str(" - "),
            '…' => t.push_str("..."),
            '，' | '、' => t.push(','),
            '。' => t.push_str(". "),
            '：' | '∶' => t.push(':'),
            '？' => t.push('?'),
            '！' => t.push('!'),
            '；' => t.push(';'),
            '（' => t.push('('),
            '）' => t.push(')'),
            '％' => t.push('%'),
            c if c.is_whitespace() => t.push(' '),
            c if c.is_control() => {}
            c => t.push(c),
        }
    }
    let cs: Vec<char> = t
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .collect();
    let mut o = String::with_capacity(cs.len() * 2);
    for (i, &c) in cs.iter().enumerate() {
        let p = i.checked_sub(1).map(|j| cs[j]);
        let n = cs.get(i + 1).copied();
        let is = |x: Option<char>, f: fn(char) -> bool| x.is_some_and(f);
        match c {
            '-' if is(p, char::is_alphanumeric) && is(n, char::is_alphanumeric) => {
                o.push_str(" @-@ ")
            }
            ',' if is(p, char::is_numeric) && is(n, char::is_numeric) => o.push(','),
            ',' => o.push_str(" , "),
            '.' if n == Some('.') && p != Some('.') => o.push_str(" ."),
            '.' if p == Some('.') && n != Some('.') => o.push_str(". "),
            '\'' if lang == "en" && p.is_some() && n.is_some() => {
                let (p, n) = (p.unwrap(), n.unwrap());
                if p.is_alphabetic() && n.is_alphabetic() || p.is_numeric() && n == 's' {
                    o.push_str(" '");
                } else if !p.is_alphabetic() && !n.is_alphabetic()
                    || !p.is_alphanumeric() && n.is_alphabetic()
                    || p.is_alphabetic()
                {
                    o.push_str(" ' ");
                } else {
                    o.push('\'');
                }
            }
            '\'' if lang != "en" => o.push_str(" ' "),
            c if c.is_alphanumeric() || " .'`-".contains(c) => o.push(c),
            c => {
                o.push(' ');
                o.push(c);
                o.push(' ');
            }
        }
    }
    const NONBREAKING: &[&str] = &[
        "Adj", "Adm", "Adv", "Asst", "Bldg", "Brig", "Bros", "Capt", "Cmdr", "Col", "Corp", "Cpl",
        "Dr", "Drs", "Gen", "Gov", "Hon", "Hr", "Hosp", "Insp", "Lt", "Maj", "Messrs", "Mlle",
        "Mme", "Mr", "Mrs", "Ms", "Msgr", "Prof", "Rep", "Rev", "Sen", "Sgt", "Sr", "St", "Supt",
        "v", "vs", "i.e", "e.g", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
        "Oct", "Nov", "Dec", "Nr", "No", "Art", "pp", "z.B", "bzw", "usw", "ca", "evtl",
    ];
    let toks: Vec<&str> = o.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(toks.len() + 4);
    for (i, tok) in toks.iter().enumerate() {
        let prefix = tok
            .strip_suffix('.')
            .filter(|p| !p.is_empty() && !p.chars().all(|c| c == '.'));
        let keep = prefix.is_none_or(|p| {
            p.contains('.') && p.chars().any(char::is_alphabetic)
                || NONBREAKING.contains(&p)
                || p.chars().count() == 1 && p.chars().all(char::is_uppercase)
                || toks
                    .get(i + 1)
                    .and_then(|n| n.chars().next())
                    .is_some_and(char::is_lowercase)
        });
        match prefix {
            Some(p) if !keep => {
                out.push(p.to_string());
                out.push(".".into());
            }
            _ => out.push(tok.to_string()),
        }
    }
    out.iter()
        .map(|t| {
            t.replace('&', "&amp;")
                .replace('|', "&#124;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('\'', "&apos;")
                .replace('"', "&quot;")
                .replace('[', "&#91;")
                .replace(']', "&#93;")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Marian: 10000^(-2i/d) angles, sin in the first half, cos in the second.
fn sinusoidal_marian(rows: usize, d: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * d];
    let half = d / 2;
    for p in 0..rows {
        for i in 0..half {
            let a = p as f64 / 10000f64.powf(2.0 * i as f64 / d as f64);
            t[p * d + i] = a.sin() as f32;
            t[p * d + half + i] = a.cos() as f32;
        }
    }
    t
}

/// fairseq (M2M100, FSMT): exp(-i ln(10000)/(half-1)) angles, sin | cos halves, padding row 1 zero.
fn sinusoidal_fairseq(rows: usize, d: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * d];
    let half = d / 2;
    let step = (10000f64).ln() / (half as f64 - 1.0);
    for p in 0..rows {
        if p == 1 {
            continue;
        }
        for i in 0..half {
            let a = p as f64 * (-(i as f64) * step).exp();
            t[p * d + i] = a.sin() as f32;
            t[p * d + half + i] = a.cos() as f32;
        }
    }
    t
}

fn log_softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return;
    }
    let s = x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln() + m;
    for v in x {
        *v -= s;
    }
}

/// One-query attention over `len` cached rows, all heads.
fn attend(q: &[f32], k: &[f32], v: &[f32], len: usize, heads: usize) -> Vec<f32> {
    let d = q.len();
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; d];
    let mut s = vec![0.0f32; len];
    for h in 0..heads {
        let qh = &q[h * hd..(h + 1) * hd];
        for (j, sj) in s.iter_mut().enumerate() {
            let kj = &k[j * d + h * hd..j * d + (h + 1) * hd];
            *sj = qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.0;
        for x in s.iter_mut() {
            *x = (*x - m).exp();
            z += *x;
        }
        let oh = &mut out[h * hd..(h + 1) * hd];
        for (j, &p) in s.iter().enumerate() {
            let vj = &v[j * d + h * hd..j * d + (h + 1) * hd];
            for (o, &x) in oh.iter_mut().zip(vj) {
                *o += p / z * x;
            }
        }
    }
    out
}

/// Per-beam self-attention cache: keys and values of every decoded position, per layer.
#[derive(Clone)]
struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Seq2Seq {
    fn embed(&self, s: &Stack, ids: &[u32], start: usize) -> Vec<f32> {
        let d = self.d;
        let mut x = vec![0.0f32; ids.len() * d];
        for (i, &id) in ids.iter().enumerate() {
            let e = &s.embed[id as usize * d..(id as usize + 1) * d];
            let p = &s.pos[(start + i + s.pos_offset) * d..][..d];
            for ((o, &e), &p) in x[i * d..(i + 1) * d].iter_mut().zip(e).zip(p) {
                *o = e * self.embed_scale + p;
            }
        }
        match &s.ln_emb {
            Some(ln) => ln.apply(&x),
            None => x,
        }
    }

    /// `x += f(norm(x))` (pre-LN) or `x = norm(x + f(x))` (post-LN).
    fn residual(&self, x: &mut Vec<f32>, ln: &Ln, f: impl FnOnce(&[f32]) -> Vec<f32>) {
        let h = if self.pre_ln { f(&ln.apply(x)) } else { f(x) };
        for (a, b) in x.iter_mut().zip(&h) {
            *a += b;
        }
        if !self.pre_ln {
            *x = ln.apply(x);
        }
    }

    fn ffn(&self, l: &Layer, h: &[f32], rows: usize) -> Vec<f32> {
        let mut a = l.fc1.apply(h, rows);
        (self.act)(&mut a);
        l.fc2.apply(&a, rows)
    }

    fn encode(&self, ids: &[u32]) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("seq2seq::encode");
        let n = ids.len();
        let s = &self.enc;
        let hd = self.d / s.heads;
        let mut x = self.embed(s, ids, 0);
        for l in &s.layers {
            self.residual(&mut x, &l.sa.ln, |h| {
                let (q, k, v) = (l.sa.q.apply(h, n), l.sa.k.apply(h, n), l.sa.v.apply(h, n));
                l.sa.o
                    .apply(&nn::attention(&q, &k, &v, n, n, s.heads, hd, false), n)
            });
            self.residual(&mut x, &l.ff_ln, |h| self.ffn(l, h, n));
        }
        match &s.final_ln {
            Some(ln) => ln.apply(&x),
            None => x,
        }
    }

    /// One decoder position for every beam: `toks[b]` at `pos`; returns `[beams, vocab]` logits.
    fn decode_step(
        &self,
        toks: &[u32],
        pos: usize,
        caches: &mut [Cache],
        cross: &[(Vec<f32>, Vec<f32>)],
        n_src: usize,
    ) -> Vec<f32> {
        crate::hardware::cpu_compute::enter("seq2seq::decode_step");
        let (b, d, s) = (toks.len(), self.d, &self.dec);
        let mut x: Vec<f32> = toks
            .iter()
            .flat_map(|&t| self.embed(s, &[t], pos))
            .collect();
        for (li, l) in s.layers.iter().enumerate() {
            self.residual(&mut x, &l.sa.ln, |h| {
                let (q, k, v) = (l.sa.q.apply(h, b), l.sa.k.apply(h, b), l.sa.v.apply(h, b));
                let mut a = Vec::with_capacity(b * d);
                for (bi, c) in caches.iter_mut().enumerate() {
                    c.k[li].extend_from_slice(&k[bi * d..(bi + 1) * d]);
                    c.v[li].extend_from_slice(&v[bi * d..(bi + 1) * d]);
                    a.extend(attend(
                        &q[bi * d..(bi + 1) * d],
                        &c.k[li],
                        &c.v[li],
                        pos + 1,
                        s.heads,
                    ));
                }
                l.sa.o.apply(&a, b)
            });
            let ca = l.ca.as_ref().expect("decoder layer has cross-attention");
            self.residual(&mut x, &ca.ln, |h| {
                let q = ca.q.apply(h, b);
                let (ck, cv) = &cross[li];
                let a: Vec<f32> = (0..b)
                    .flat_map(|bi| attend(&q[bi * d..(bi + 1) * d], ck, cv, n_src, s.heads))
                    .collect();
                ca.o.apply(&a, b)
            });
            self.residual(&mut x, &l.ff_ln, |h| self.ffn(l, h, b));
        }
        if let Some(ln) = &s.final_ln {
            x = ln.apply(&x);
        }
        linear(
            &x,
            b,
            &self.lm_head,
            self.logits_bias.as_deref(),
            self.vocab,
        )
    }

    /// transformers' logits processors on one beam's log-probabilities.
    fn process(&self, lp: &mut [f32], seq: &[u32]) {
        let g = &self.gen;
        let cur = seq.len();
        for &t in &g.bad {
            if let Some(v) = lp.get_mut(t as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
        if cur < g.min_len {
            lp[self.eos as usize] = f32::NEG_INFINITY;
        }
        let n = g.no_repeat;
        if n > 0 && cur + 1 >= n {
            let tail = &seq[cur + 1 - n..];
            for w in seq.windows(n) {
                if w[..n - 1] == *tail {
                    lp[w[n - 1] as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(e) = g.forced_eos.filter(|_| cur + 1 == g.max_len) {
            lp.fill(f32::NEG_INFINITY);
            lp[e as usize] = 0.0;
        }
        if g.renormalize {
            log_softmax(lp);
        }
    }

    /// Encoder output projected to every decoder layer's cross-attention keys/values.
    fn cross(&self, src: &[u32]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let n = src.len();
        let enc = self.encode(src);
        self.dec
            .layers
            .iter()
            .map(|l| {
                let ca = l.ca.as_ref().expect("decoder layer has cross-attention");
                (ca.k.apply(&enc, n), ca.v.apply(&enc, n))
            })
            .collect()
    }

    fn empty_cache(&self) -> Cache {
        let layers = self.dec.layers.len();
        Cache {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
        }
    }

    /// Encodes `src` into a request's decoder state with one empty beam.
    fn begin(&self, src: &[u32]) -> Result<Run, String> {
        match &self.gpu {
            Some(gpu) => Ok(Run::Gpu(vec![gpu.start(src)?])),
            None => Ok(Run::Cpu {
                cross: self.cross(src),
                n_src: src.len(),
                caches: vec![self.empty_cache()],
            }),
        }
    }

    /// One decoder position for every beam (`toks[b]` at `pos`): `[beams, vocab]` logits.
    fn advance(&self, run: &mut Run, toks: &[u32], pos: usize) -> Result<Vec<f32>, String> {
        match (run, &self.gpu) {
            (Run::Cpu { cross, n_src, caches }, _) => {
                Ok(self.decode_step(toks, pos, caches, cross, *n_src))
            }
            (Run::Gpu(beams), Some(gpu)) => {
                // One program run per beam: each beam owns its self-attention state.
                let mut logits = Vec::with_capacity(toks.len() * self.vocab);
                for (seq, &t) in beams.iter_mut().zip(toks) {
                    debug_assert_eq!(seq.len, pos);
                    logits.extend(gpu.step(seq, t)?);
                }
                Ok(logits)
            }
            (Run::Gpu(_), None) => Err("GPU decoder state without a GPU program".into()),
        }
    }

    /// Beam `i` continues from beam `parents[i]` (transformers' `reorder_cache`).
    fn reorder(&self, run: &mut Run, parents: Vec<usize>) -> Result<(), String> {
        match (run, &self.gpu) {
            (Run::Cpu { caches, .. }, _) => {
                *caches = parents.iter().map(|&bi| caches[bi].clone()).collect();
            }
            (Run::Gpu(beams), Some(gpu)) => {
                let next = parents.iter().map(|&bi| gpu.fork(&beams[bi])).collect::<Result<Vec<_>, _>>()?;
                *beams = next;
            }
            (Run::Gpu(_), None) => return Err("GPU decoder state without a GPU program".into()),
        }
        Ok(())
    }
    /// transformers' beam search: the best hypothesis after `prompt`, ending in EOS unless
    /// `max_length` or the request's `max_new_tokens` cut it; Err once `cancel` is set.
    fn search(&self, src: &[u32], prompt: &[u32], ctx: &PrepareCtx) -> Result<Vec<u32>, String> {
        let mut run = self.begin(src)?;
        let mut logits = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            logits = self.advance(&mut run, &[t], p)?;
        }
        let (k, v, g) = (self.gen.beams.max(1), self.vocab, &self.gen);
        let cap = g
            .max_len
            .min(prompt.len().saturating_add(ctx.max_new_tokens.max(1)));
        let norm = |s: f32, len: usize| s / (len as f32).powf(g.length_penalty);
        let mut beams: Vec<(Vec<u32>, f32)> = vec![(prompt.to_vec(), 0.0)];
        // (normalized score, sequence, ended with EOS)
        let mut done: Vec<(f32, Vec<u32>, bool)> = Vec::new();
        let add = |done: &mut Vec<(f32, Vec<u32>, bool)>, h: (f32, Vec<u32>, bool)| {
            let worst = done.iter().map(|d| d.0).fold(f32::INFINITY, f32::min);
            if done.len() < k || h.0 > worst {
                done.push(h);
                if done.len() > k {
                    let i = (0..done.len())
                        .min_by(|&a, &b| done[a].0.total_cmp(&done[b].0))
                        .unwrap();
                    done.remove(i);
                }
            }
        };
        loop {
            if ctx
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
            {
                // The engine installs no tracing subscriber; stderr reaches the server log.
                eprintln!(
                    "[seq2seq] beam search cancelled at decoder length {}: client disconnected",
                    beams[0].0.len()
                );
                return Err("cancelled: the client disconnected".into());
            }
            let cur = beams[0].0.len();
            let mut top: Vec<(f32, usize, u32)> = Vec::with_capacity(2 * k + 1);
            for (bi, (seq, score)) in beams.iter().enumerate() {
                let lp = &mut logits[bi * v..(bi + 1) * v];
                log_softmax(lp);
                self.process(lp, seq);
                for (t, &x) in lp.iter().enumerate() {
                    let c = score + x;
                    if c > f32::NEG_INFINITY && (top.len() < 2 * k || c > top[top.len() - 1].0) {
                        let at = top.partition_point(|e| e.0 >= c);
                        top.insert(at, (c, bi, t as u32));
                        top.truncate(2 * k);
                    }
                }
            }
            let mut next: Vec<(usize, u32, f32)> = Vec::with_capacity(k);
            for (rank, &(c, bi, t)) in top.iter().enumerate() {
                if t == self.eos {
                    if rank < k {
                        add(&mut done, (norm(c, cur), beams[bi].0.clone(), true));
                    }
                } else {
                    next.push((bi, t, c));
                }
                if next.len() == k {
                    break;
                }
            }
            let finished = done.len() >= k
                && (g.early_stopping
                    || top.first().is_none_or(|b| {
                        done.iter().map(|d| d.0).fold(f32::INFINITY, f32::min) >= norm(b.0, cur)
                    }));
            if finished || next.is_empty() {
                break;
            }
            self.reorder(&mut run, next.iter().map(|&(bi, _, _)| bi).collect())?;
            beams = next
                .iter()
                .map(|&(bi, t, c)| {
                    let mut s = beams[bi].0.clone();
                    s.push(t);
                    (s, c)
                })
                .collect();
            if cur + 1 >= cap {
                for (s, c) in beams {
                    let len = s.len() - 1;
                    add(&mut done, (norm(c, len), s, false));
                }
                break;
            }
            let toks: Vec<u32> = beams.iter().map(|b| *b.0.last().unwrap()).collect();
            logits = self.advance(&mut run, &toks, cur)?;
        }
        let (_, seq, eos) = done
            .into_iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap_or((0.0, prompt.to_vec(), true));
        let mut out = seq[prompt.len()..].to_vec();
        if eos {
            out.push(self.eos);
        }
        Ok(out)
    }

    /// Output ids as displayed: aliases (`alias + id`) carry the alternate piece text — the
    /// first piece without its leading space (SentencePiece / byte-level, as transformers'
    /// decode strips it), and FSMT word starts with a space (Moses detokenization).
    fn display(&self, out: &[u32]) -> Vec<u32> {
        let alias = |id: u32| id + self.alias;
        let n = out.iter().position(|&t| t == self.eos).unwrap_or(out.len());
        let mut ids = out.to_vec();
        if self.pieces != Pieces::Fsmt {
            if n > 0 {
                ids[0] = alias(ids[0]);
            }
            return ids;
        }
        // Words: runs of pieces up to one ending in `</w>`.
        let mut words: Vec<(usize, String)> = Vec::new();
        let mut start = 0;
        for i in 0..n {
            let raw = self.raw.get(&out[i]).map_or("", String::as_str);
            if raw.ends_with("</w>") || i + 1 == n {
                let w: String = out[start..=i]
                    .iter()
                    .map(|t| {
                        self.raw
                            .get(t)
                            .map_or("", String::as_str)
                            .trim_end_matches("</w>")
                    })
                    .collect();
                words.push((start, unescape(&w)));
                start = i + 1;
            }
        }
        let mut quotes = 0;
        let mut prev_attaches = true;
        for (at, w) in &words {
            let opening_quote = w == "\"" && {
                quotes += 1;
                quotes % 2 == 1
            };
            let attaches_left = matches!(
                w.as_str(),
                "," | "." | "!" | "?" | ":" | ";" | ")" | "]" | "}" | "%" | "..." | "“" | "@-@"
            ) || w == "\"" && !opening_quote
                || self.tgt_lang == "en" && w.starts_with('\'') && w.len() > 1;
            if !prev_attaches && !attaches_left {
                ids[*at] = alias(ids[*at]);
            }
            prev_attaches = matches!(w.as_str(), "(" | "[" | "{" | "„" | "@-@") || opening_quote;
        }
        ids
    }

    /// Splits leading `>>tgt<<` / `<<src>>` tags off `text`.
    fn tags<'a>(
        &self,
        mut text: &'a str,
    ) -> Result<(Option<&'a str>, Option<&'a str>, &'a str), String> {
        let (mut tgt, mut src) = (None, None);
        loop {
            text = text.trim_start();
            if let Some(r) = text.strip_prefix(">>") {
                let (code, rest) = r.split_once("<<").ok_or("unclosed >>target<< tag")?;
                tgt = Some(code.trim());
                text = rest;
            } else if let Some(r) = text.strip_prefix("<<") {
                let (code, rest) = r.split_once(">>").ok_or("unclosed <<source>> tag")?;
                src = Some(code.trim());
                text = rest;
            } else {
                return Ok((tgt, src, text.trim()));
            }
        }
    }

    fn lang(&self, code: &str, role: &str) -> Result<u32, String> {
        let wrapped = format!("__{code}__");
        self.langs
            .iter()
            .find(|(c, _)| c == code || *c == wrapped)
            .map(|l| l.1)
            .ok_or_else(|| {
                let some: Vec<String> = self
                    .langs
                    .iter()
                    .take(8)
                    .map(|(c, _)| c.trim_matches('_').to_string())
                    .collect();
                format!(
                    "unknown {role} language '{code}'; this model's codes are like {} ({} in all)",
                    some.join(", "),
                    self.langs.len()
                )
            })
    }

    /// A whole token's id in the source vocabulary (Marian `>>xx<<` target tokens).
    fn source_id(&self, token: &str) -> Option<u32> {
        match &self.source {
            Source::Hf(t) => t.token_to_id(token),
            Source::Moses { bpe, .. } => bpe.token_to_id(token),
        }
    }

    fn segment(&self, text: &str) -> Result<Vec<u32>, String> {
        let (t, s, known, unk) = match &self.source {
            Source::Hf(t) => (t, text.to_string(), u32::MAX, 0),
            Source::Moses {
                bpe,
                lang,
                known,
                unk,
            } => (bpe, moses_tokenize(text, lang), *known, *unk),
        };
        Ok(t.encode_fast(s, false)
            .map_err(|e| format!("source tokenization: {e}"))?
            .get_ids()
            .iter()
            .map(|&i| if i >= known { unk } else { i })
            .collect())
    }

    /// Source text -> (encoder ids, decoder prompt).
    fn source(&self, ids: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
        let bytes: Vec<u8> = ids
            .iter()
            .map(|&i| u8::try_from(i).map_err(|_| format!("token {i} is not a source byte")))
            .collect::<Result<_, _>>()?;
        let text = String::from_utf8_lossy(&bytes);
        let text: String = text
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect();
        let (tgt, src, mut body) = self.tags(&text)?;
        if body.is_empty() {
            return Err("nothing to translate: the text is empty".into());
        }
        // Byte-level BPE sees spaces as tokens (`ĠNew` vs `New`): BART reads the text as given.
        if self.family == Family::Bart && tgt.is_none() && src.is_none() {
            body = &text;
        }
        let mut enc = Vec::new();
        let mut prompt = vec![self.decoder_start];
        if !self.langs.is_empty() {
            let tgt = match tgt {
                Some(c) => self.lang(c, "target")?,
                None => self.tgt_default.ok_or_else(|| {
                    format!(
                        "this model translates into {} languages: start the text with the target \
                         language as >>code<<, e.g. '>>{}<< {body}' (optional source: <<code>>, default {})",
                        self.langs.len(),
                        self.langs.iter().find(|l| ["fra_Latn", "fr_XX", "__fr__"].contains(&l.0.as_str())).unwrap_or(&self.langs[0]).0.trim_matches('_'),
                        self.src_default.and_then(|s| self.langs.iter().find(|l| l.1 == s)).map_or("English", |l| l.0.as_str()).trim_matches('_'),
                    )
                })?,
            };
            let src = match src {
                Some(c) => self.lang(c, "source")?,
                None => self
                    .src_default
                    .ok_or("give the source language as <<code>> before the text")?,
            };
            let seg = self.segment(body)?;
            if self.legacy_mbart {
                enc.extend(seg);
                enc.extend([self.eos, src]);
                prompt = vec![tgt];
            } else {
                enc.push(src);
                enc.extend(seg);
                enc.push(self.eos);
                prompt.push(tgt);
            }
        } else {
            if src.is_some() {
                return Err(
                    "this model has no source-language codes; drop the <<source>> tag".into(),
                );
            }
            let example = || {
                let t = &self.marian_targets;
                t.iter()
                    .find(|t| t.as_str() == ">>fra<<" || t.as_str() == ">>fr<<")
                    .unwrap_or(&t[0])
                    .clone()
            };
            if let Some(c) = tgt {
                let tag = format!(">>{c}<<");
                if self.marian_targets.is_empty() {
                    return Err(format!(
                        "this model translates into one language only; drop the {tag} tag"
                    ));
                }
                let Some(id) = self
                    .marian_targets
                    .contains(&tag)
                    .then(|| self.source_id(&tag))
                    .flatten()
                else {
                    return Err(format!(
                        "unknown target {tag}; this model's {} targets include {}",
                        self.marian_targets.len(),
                        self.marian_targets
                            .iter()
                            .take(12)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                };
                enc.push(id);
            } else if !self.marian_targets.is_empty() {
                return Err(format!(
                    "this model translates into {} languages: start the text with the target \
                     token, e.g. '{} {body}'",
                    self.marian_targets.len(),
                    example()
                ));
            }
            if self.family == Family::Bart {
                enc.push(self.bos);
            }
            enc.extend(self.segment(body)?);
            enc.push(self.eos);
            prompt.extend(self.forced_bos);
        }
        if enc.len() > self.max_pos {
            return Err(format!(
                "the source is {} tokens; this model reads at most {}",
                enc.len(),
                self.max_pos
            ));
        }
        Ok((enc, prompt))
    }
}

impl NativeLm for Seq2Seq {
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
        self.gen.max_len.max(2)
    }
    fn default_max_new_tokens(&self) -> Option<usize> {
        Some(self.gen.max_len)
    }
    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        let ctx = PrepareCtx {
            max_new_tokens: usize::MAX,
            cancel: None,
        };
        self.prepare_ctx(ids, media, &ctx)
    }
    fn prepare_ctx(
        &self,
        ids: Vec<u32>,
        media: &[MediaInput],
        ctx: &PrepareCtx,
    ) -> Result<(Vec<u32>, SeqState), String> {
        if !media.is_empty() {
            return Err("this translation/summarization model takes text only".into());
        }
        let (src, prompt) = self.source(&ids)?;
        let n = prompt.len();
        // FSMT's Moses spacing needs the whole output, so it always takes the replay path.
        if self.gen.beams <= 1 && self.pieces != Pieces::Fsmt {
            let live = Live {
                run: self.begin(&src)?,
                seq: Vec::new(),
                prompt: n,
            };
            return Ok((prompt, Box::new(live)));
        }
        let out = self.display(&self.search(&src, &prompt, ctx)?);
        Ok((prompt, Box::new(Replay { prompt: n, out })))
    }
    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        if !media.is_empty() {
            return Err("this translation/summarization model takes text only".into());
        }
        Ok(self.source(&ids)?.1.len())
    }
    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        let mut logits = vec![-1e9f32; self.tok.vocab_size()];
        if let Some(r) = state.downcast_ref::<Replay>() {
            let end = start_pos + tokens.len();
            if end >= r.prompt {
                let id = r.out.get(end - r.prompt).copied().unwrap_or(self.eos);
                logits[id as usize] = 0.0;
            }
            return Ok(logits);
        }
        let live = state
            .downcast_mut::<Live>()
            .ok_or("sequence state is not a seq2seq decoder state")?;
        let mut lp = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            if start_pos + i != live.seq.len() {
                return Err(format!(
                    "decoder position {} but {} tokens decoded",
                    start_pos + i,
                    live.seq.len()
                ));
            }
            // The first output token is sampled as its display alias.
            let t = if t >= self.alias { t - self.alias } else { t };
            let pos = live.seq.len();
            lp = self.advance(&mut live.run, &[t], pos)?;
            live.seq.push(t);
        }
        if live.seq.len() < live.prompt {
            return Ok(logits);
        }
        log_softmax(&mut lp);
        self.process(&mut lp, &live.seq);
        let off = if live.seq.len() == live.prompt {
            self.alias as usize
        } else {
            0
        };
        let n = lp.len().min(self.alias as usize);
        for (o, &x) in logits[off..off + n].iter_mut().zip(&lp[..n]) {
            *o = if x.is_finite() { x } else { -1e9 };
        }
        Ok(logits)
    }
}

/// Every tensor of the checkpoint decoded to f32 on demand; embedding tables (often tied) are
/// shared through [`Ck::get`].
struct Ck {
    manifest: crate::model::ModelManifest,
    data: Vec<u8>,
    loaded: HashMap<String, Arc<Vec<f32>>>,
}

impl Ck {
    fn has(&self, name: &str) -> bool {
        self.manifest.tensors.contains_key(name)
    }
    fn get(&mut self, name: &str) -> Result<Arc<Vec<f32>>, String> {
        if let Some(t) = self.loaded.get(name) {
            return Ok(t.clone());
        }
        let v = Arc::new(self.vec(name)?);
        self.loaded.insert(name.to_string(), v.clone());
        Ok(v)
    }
    fn vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self
            .manifest
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' missing"))?;
        crate::model::decode_dtype(
            &self.data[t.byte_offset..t.byte_offset + t.byte_len],
            &t.dtype,
        )
        .map_err(|e| format!("tensor '{name}': {e}"))
    }
    fn lin(&mut self, p: &str) -> Result<Lin, String> {
        let (w, b) = (
            self.vec(&format!("{p}.weight"))?,
            self.vec(&format!("{p}.bias"))?,
        );
        if b.is_empty() || w.len() % b.len() != 0 {
            return Err(format!("{p}: weight/bias shapes disagree"));
        }
        Ok(Lin { w, b })
    }
    fn ln(&mut self, p: &str) -> Result<Ln, String> {
        Ok(Ln {
            g: self.vec(&format!("{p}.weight"))?,
            b: self.vec(&format!("{p}.bias"))?,
        })
    }
    fn attn(&mut self, p: &str, ln: &str) -> Result<Attn, String> {
        Ok(Attn {
            q: self.lin(&format!("{p}.q_proj"))?,
            k: self.lin(&format!("{p}.k_proj"))?,
            v: self.lin(&format!("{p}.v_proj"))?,
            o: self.lin(&format!("{p}.out_proj"))?,
            ln: self.ln(ln)?,
        })
    }
}

/// Builds the decoding [`Tokenizer`]: target pieces `vocab` (+ `specials`), aliases at
/// `alias + id`, byte-passthrough encoding and the last-user-message template.
fn decode_tokenizer(
    vocab: &HashMap<String, u32>,
    specials: &[&str],
    pieces: Pieces,
    unk: Option<&str>,
) -> Result<(Tokenizer, u32), String> {
    let alias = vocab.values().max().map_or(0, |&m| m + 1);
    let mut v = HashMap::with_capacity(2 * alias as usize);
    let mut inv = HashMap::with_capacity(2 * alias as usize);
    for (piece, &id) in vocab {
        v.insert(piece.clone(), id);
        let (main, alt) = if Some(piece.as_str()) == unk {
            (String::new(), String::new())
        } else {
            match pieces {
                Pieces::Spm => (
                    piece.clone(),
                    piece.trim_start_matches('\u{2581}').to_string(),
                ),
                Pieces::Gpt2 => (piece.clone(), piece.trim_start_matches('Ġ').to_string()),
                Pieces::Fsmt => {
                    let s = unescape(piece.trim_end_matches("</w>"));
                    let s = if s == "@-@" { "-".to_string() } else { s };
                    (s.clone(), format!(" {s}"))
                }
            }
        };
        if inv.insert(id, main).is_none() {
            v.insert(format!("\u{1}alias{id}"), alias + id);
            inv.insert(alias + id, alt);
        }
    }
    // The sampler reads `vocab_size()` logits: cover every alias id.
    let mut gap = 0;
    while v.len() < 2 * alias as usize {
        v.insert(format!("\u{1}gap{gap}"), 0);
        gap += 1;
    }
    let hf = byte_passthrough()?;
    Ok((
        Tokenizer {
            special_tokens: specials
                .iter()
                .filter_map(|s| Some((s.to_string(), *vocab.get(*s)?)))
                .collect(),
            vocab: v,
            inv_vocab: inv,
            // Display strings are UTF-8 text with `▁` as space; GPT-2 pieces are byte-mapped.
            spm: pieces != Pieces::Gpt2,
            add_space_prefix: false,
            hf: Some(Arc::new(hf)),
            chat_template: Some(ChatTemplateSpec {
                template: LAST_USER_TEMPLATE.into(),
                bos_token: String::new(),
                eos_token: String::new(),
            }),
            // Encoder-decoders decode greedily unless a request sets temperature/top_p (as T5 does).
            sampling_defaults: Some((0.0, 1.0)),
            ..Tokenizer::from_tokens(Vec::new())
        },
        alias,
    ))
}

/// tokenizer.json `model.vocab` (BPE map or Unigram list) plus `added_tokens`.
fn tokenizer_json_vocab(path: &Path) -> Result<HashMap<String, u32>, String> {
    let j: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("tokenizer.json: {e}"))?;
    let mut v: HashMap<String, u32> = match &j["model"]["vocab"] {
        Value::Object(m) => m
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_u64()? as u32)))
            .collect(),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .filter_map(|(i, p)| Some((p[0].as_str()?.to_string(), i as u32)))
            .collect(),
        _ => return Err("tokenizer.json has no model.vocab".into()),
    };
    for t in j["added_tokens"].as_array().into_iter().flatten() {
        if let (Some(c), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) {
            v.insert(c.to_string(), id as u32);
        }
    }
    Ok(v)
}

pub fn load(
    artifacts: &ModelArtifacts,
    cfg: &Value,
    mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    let backend = crate::graph::avenues::resolved_backend(mode.into());
    if !matches!(backend, crate::hardware::Device::Cpu | crate::hardware::Device::Metal) {
        return Err(format!("seq2seq has no {backend} program; use metal or --backend cpu"));
    }
    let arch = cfg["architectures"][0].as_str().unwrap_or("");
    let family = match arch {
        "MarianMTModel" => Family::Marian,
        "M2M100ForConditionalGeneration" => Family::M2m100,
        "MBartForConditionalGeneration" => Family::MBart,
        "FSMTForConditionalGeneration" => Family::Fsmt,
        "BartForConditionalGeneration" => Family::Bart,
        a => {
            return Err(format!(
                "architecture '{a}' is not a supported encoder-decoder"
            ))
        }
    };
    let dir = artifacts
        .config_path
        .as_deref()
        .and_then(Path::parent)
        .ok_or("no config.json")?;
    let gen_cfg = read_json(dir, "generation_config.json").unwrap_or(Value::Null);
    let g = |k: &str| -> Value {
        match &gen_cfg[k] {
            Value::Null => cfg[k].clone(),
            v => v.clone(),
        }
    };
    let gu = |k: &str| g(k).as_u64().map(|v| v as u32);
    let tok_cfg = read_json(dir, "tokenizer_config.json").unwrap_or(Value::Null);
    let num = |k: &str| -> Result<usize, String> {
        cfg[k]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("config.json: {k} missing"))
    };
    let d = num("d_model")?;
    let max_pos = num("max_position_embeddings")?;
    let eos = gu("eos_token_id").unwrap_or(2);
    let bos = gu("bos_token_id").unwrap_or(0);
    let pad = gu("pad_token_id").unwrap_or(1);

    // ---- Tokenizers: source segmentation + target pieces.
    let has = |f: &str| dir.join(f).is_file();
    let spm_file = ["source.spm", "sentencepiece.bpe.model", "spiece.model"]
        .into_iter()
        .find(|f| has(f));
    let (source, mut target, pieces) = if family == Family::Fsmt {
        let lang = cfg["langs"][0]
            .as_str()
            .ok_or("config.json: langs missing")?;
        // FSMT merges through symbols its vocab lacks; final tokens outside it become <unk>.
        let src_vocab = json_vocab(dir, "vocab-src.json")?;
        let known = src_vocab.values().max().map_or(0, |&m| m + 1);
        let unk = *src_vocab
            .get("<unk>")
            .ok_or("vocab-src.json has no <unk>")?;
        let merges = merges(dir)?;
        let mut ext = src_vocab;
        let mut next = known;
        for (a, b) in &merges {
            for s in [a.clone(), b.clone(), format!("{a}{b}")] {
                ext.entry(s).or_insert_with(|| {
                    next += 1;
                    next - 1
                });
            }
        }
        let bpe = BPE::builder()
            .vocab_and_merges(bpe_vocab(&ext), merges)
            .end_of_word_suffix("</w>".into())
            .unk_token("<unk>".into())
            .build()
            .map_err(|e| format!("FSMT BPE: {e}"))?;
        let mut t = tokenizers::Tokenizer::new(bpe);
        t.with_pre_tokenizer(Some(WhitespaceSplit));
        (
            Source::Moses {
                bpe: t,
                lang: lang.to_string(),
                known,
                unk,
            },
            json_vocab(dir, "vocab-tgt.json")?,
            Pieces::Fsmt,
        )
    } else if has("tokenizer.json") {
        let t = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let pieces = if family == Family::Bart {
            Pieces::Gpt2
        } else {
            Pieces::Spm
        };
        (
            Source::Hf(t),
            tokenizer_json_vocab(&dir.join("tokenizer.json"))?,
            pieces,
        )
    } else if family == Family::Bart && has("vocab.json") && has("merges.txt") {
        let v = json_vocab(dir, "vocab.json")?;
        let bpe = BPE::builder()
            .vocab_and_merges(bpe_vocab(&v), merges(dir)?)
            .build()
            .map_err(|e| format!("byte-level BPE: {e}"))?;
        let mut t = tokenizers::Tokenizer::new(bpe);
        t.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        (Source::Hf(t), v, Pieces::Gpt2)
    } else if let Some(f) = spm_file {
        let spm = SpmModel::open(&dir.join(f))?;
        let vocab = if has("vocab.json") {
            // Marian / M2M100: spm pieces, ids from vocab.json.
            json_vocab(dir, "vocab.json")?
        } else {
            // MBart-50 / NLLB slow layout: fairseq specials, then spm ids shifted by one.
            let mut v: HashMap<String, u32> = [("<s>", 0), ("<pad>", 1), ("</s>", 2), ("<unk>", 3)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            for (i, p) in spm.pieces.iter().enumerate().skip(3) {
                v.entry(p.text.clone()).or_insert(i as u32 + 1);
            }
            v
        };
        let seg = spm.segmenter(&vocab)?;
        // Marian `target_vocab.json`: separate target ids.
        let target = if has("target_vocab.json") {
            json_vocab(dir, "target_vocab.json")?
        } else {
            vocab
        };
        (Source::Hf(seg), target, Pieces::Spm)
    } else {
        return Err(format!(
            "{}: no tokenizer (tokenizer.json, *.spm/sentencepiece.bpe.model + vocab.json, or vocab.json + merges.txt)",
            dir.display()
        ));
    };

    // Language codes: special_tokens_map / tokenizer_config additional_special_tokens; ones the
    // vocab lacks follow it (M2M100, MBart-50 layout).
    let mut langs: Vec<(String, u32)> = Vec::new();
    if matches!(family, Family::M2m100 | Family::MBart) {
        let extra = read_json(dir, "special_tokens_map.json")
            .map(|j| j["additional_special_tokens"].clone())
            .filter(Value::is_array)
            .unwrap_or_else(|| tok_cfg["additional_special_tokens"].clone());
        let mut next = target.values().max().map_or(0, |&m| m + 1);
        for t in extra.as_array().into_iter().flatten() {
            let Some(code) = t.as_str().or_else(|| t["content"].as_str()) else {
                continue;
            };
            let id = *target.entry(code.to_string()).or_insert_with(|| {
                next += 1;
                next - 1
            });
            langs.push((code.to_string(), id));
        }
        if langs.is_empty() && family == Family::M2m100 {
            return Err("special_tokens_map.json lists no language codes".into());
        }
    }
    let lang_named = |c: &str| {
        langs
            .iter()
            .find(|l| l.0 == c || l.0 == format!("__{c}__"))
            .map(|l| l.1)
    };
    let src_default = tok_cfg["src_lang"]
        .as_str()
        .and_then(lang_named)
        .or_else(|| ["eng_Latn", "en_XX", "en"].into_iter().find_map(lang_named));
    let tgt_default = tok_cfg["tgt_lang"]
        .as_str()
        .and_then(lang_named)
        .or_else(|| gu("forced_bos_token_id").filter(|id| langs.iter().any(|l| l.1 == *id)));
    let legacy_mbart = family == Family::MBart
        && [&tok_cfg["tokenizer_class"], &cfg["tokenizer_class"]]
            .iter()
            .any(|c| c.as_str() == Some("MBartTokenizer"));
    let marian_targets: Vec<String> = if family == Family::Marian {
        let mut t: Vec<String> = target
            .keys()
            .filter(|k| k.starts_with(">>") && k.ends_with("<<") && k.len() > 4)
            .cloned()
            .collect();
        t.sort();
        t
    } else {
        Vec::new()
    };
    let mut specials: Vec<&str> = vec!["<s>", "</s>", "<pad>", "<mask>"];
    specials.extend(langs.iter().map(|l| l.0.as_str()));
    specials.extend(marian_targets.iter().map(String::as_str));
    let raw: HashMap<u32, String> = if pieces == Pieces::Fsmt {
        target.iter().map(|(k, &v)| (v, k.clone())).collect()
    } else {
        HashMap::new()
    };
    let (tok, alias) = decode_tokenizer(&target, &specials, pieces, Some("<unk>"))?;
    target.clear();

    // ---- Weights.
    let wpath = artifacts.weights_path.as_deref().ok_or("no weight file")?;
    let (manifest, data) = crate::model::loaders::read_checkpoint(wpath)?;
    let mut ck = Ck {
        manifest,
        data,
        loaded: HashMap::new(),
    };
    let (act, act_kind): (fn(&mut [f32]), ActKind) =
        match cfg["activation_function"].as_str().unwrap_or("gelu") {
            "relu" => (|x: &mut [f32]| x.iter_mut().for_each(|v| *v = v.max(0.0)), ActKind::Relu),
            "gelu" => (nn::gelu, ActKind::Gelu),
            "gelu_new" | "gelu_pytorch_tanh" => (nn::gelu_tanh, ActKind::GeluTanh),
            "swish" | "silu" => (nn::silu, ActKind::Silu),
            a => return Err(format!("activation '{a}' is not supported")),
        };
    // Checkpoints drop tied copies (wmt19-en-de keeps only the encoder table, bart-large-cnn's
    // safetensors only the decoder's): take the first table present.
    let (shared, enc, dec) = (
        "model.shared.weight",
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
    );
    let first = |ck: &Ck, names: &[&'static str]| names.iter().copied().find(|n| ck.has(n));
    let enc_name = first(&ck, &[shared, enc, dec]).ok_or("no token embedding tensor")?;
    let dec_name = if cfg["share_encoder_decoder_embeddings"].as_bool() == Some(false) {
        first(&ck, &[dec])
    } else {
        first(&ck, &[shared, dec, enc])
    }
    .ok_or("no decoder token embedding tensor")?;
    let enc_embed = ck.get(enc_name)?;
    let dec_embed = ck.get(dec_name)?;
    let vocab = dec_embed.len() / d;
    if cfg["tgt_vocab_size"]
        .as_u64()
        .is_some_and(|t| t as usize != vocab)
    {
        return Err(format!(
            "decoder embedding has {vocab} rows; config tgt_vocab_size disagrees"
        ));
    }
    let tied = cfg["tie_word_embeddings"].as_bool() != Some(false);
    let lm_head = match (tied, family) {
        (true, _) => dec_embed.clone(),
        (false, Family::Fsmt) => ck.get("model.decoder.output_projection.weight")?,
        (false, _) => ck.get("lm_head.weight")?,
    };
    if lm_head.len() != vocab * d || enc_embed.len() % d != 0 {
        return Err("embedding / output projection shapes disagree with d_model".into());
    }
    let logits_bias = if ck.has("final_logits_bias") {
        let b = ck.vec("final_logits_bias")?;
        (b.len() == vocab)
            .then_some(b)
            .ok_or("final_logits_bias length != vocab")?
            .into()
    } else {
        None
    };
    let pre_ln = matches!(family, Family::M2m100 | Family::MBart);
    let stack = |side: &str, embed: Arc<Vec<f32>>, ck: &mut Ck| -> Result<Stack, String> {
        let n = num(&format!("{side}_layers"))?;
        let heads = num(&format!("{side}_attention_heads"))?;
        let p = format!("model.{side}");
        let (pos, pos_offset) = match family {
            Family::Marian => (sinusoidal_marian(max_pos, d), 0),
            Family::M2m100 | Family::Fsmt => (sinusoidal_fairseq(max_pos + 2, d), 2),
            Family::MBart | Family::Bart => {
                let t = ck.vec(&format!("{p}.embed_positions.weight"))?;
                if t.len() < (max_pos + 2) * d {
                    return Err(format!(
                        "{p}.embed_positions is smaller than max_position_embeddings + 2"
                    ));
                }
                (t, 2)
            }
        };
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let l = format!("{p}.layers.{i}");
            layers.push(Layer {
                sa: ck.attn(
                    &format!("{l}.self_attn"),
                    &format!("{l}.self_attn_layer_norm"),
                )?,
                ca: if side == "decoder" {
                    Some(ck.attn(
                        &format!("{l}.encoder_attn"),
                        &format!("{l}.encoder_attn_layer_norm"),
                    )?)
                } else {
                    None
                },
                fc1: ck.lin(&format!("{l}.fc1"))?,
                fc2: ck.lin(&format!("{l}.fc2"))?,
                ff_ln: ck.ln(&format!("{l}.final_layer_norm"))?,
            });
        }
        Ok(Stack {
            embed,
            pos,
            pos_offset,
            ln_emb: if matches!(family, Family::MBart | Family::Bart) {
                Some(ck.ln(&format!("{p}.layernorm_embedding"))?)
            } else {
                None
            },
            layers,
            final_ln: if pre_ln {
                Some(ck.ln(&format!("{p}.layer_norm"))?)
            } else {
                None
            },
            heads,
        })
    };
    let enc = stack("encoder", enc_embed, &mut ck)?;
    let dec = stack("decoder", dec_embed, &mut ck)?;
    drop(ck);

    let max_len = gu("max_length")
        .map_or(max_pos, |m| m as usize)
        .min(max_pos);
    let gen = Gen {
        beams: gu("num_beams").unwrap_or(1).max(1) as usize,
        max_len,
        min_len: gu("min_length").unwrap_or(0) as usize,
        length_penalty: g("length_penalty").as_f64().unwrap_or(1.0) as f32,
        no_repeat: gu("no_repeat_ngram_size").unwrap_or(0) as usize,
        early_stopping: g("early_stopping").as_bool() == Some(true),
        renormalize: g("renormalize_logits").as_bool() == Some(true),
        forced_eos: gu("forced_eos_token_id"),
        bad: g("bad_words_ids")
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|w| match w.as_array()?.as_slice() {
                [t] => Some(t.as_u64()? as u32),
                _ => None,
            })
            .collect(),
    };
    let decoder_start = gu("decoder_start_token_id").unwrap_or(eos);
    let forced_bos = gu("forced_bos_token_id").filter(|_| langs.is_empty());
    if [eos, bos, pad, decoder_start]
        .iter()
        .any(|&t| t as usize >= vocab)
    {
        return Err("special token ids exceed the vocabulary".into());
    }
    let summary = format!(
        "{arch}: {}+{} layers, d_model {d}, vocab {vocab}, f32 weights, {} {}",
        enc.layers.len(),
        dec.layers.len(),
        if gen.beams > 1 {
            format!("beam search ×{}", gen.beams)
        } else {
            "greedy".into()
        },
        if langs.is_empty() && marian_targets.is_empty() {
            String::new()
        } else {
            format!(
                "({} target languages)",
                langs.len().max(marian_targets.len())
            )
        },
    );
    let mut model = Seq2Seq {
        family,
        d,
        enc,
        dec,
        lm_head,
        logits_bias,
        vocab,
        act,
        pre_ln,
        embed_scale: if cfg["scale_embedding"].as_bool() == Some(true) {
            (d as f32).sqrt()
        } else {
            1.0
        },
        max_pos,
        gen,
        eos,
        bos,
        decoder_start,
        forced_bos,
        source,
        tok,
        alias,
        pieces,
        raw,
        langs,
        src_default,
        tgt_default,
        legacy_mbart,
        marian_targets,
        tgt_lang: cfg["langs"][1].as_str().unwrap_or("").to_string(),
        summary: summary.trim_end().to_string(),
        act_kind,
        gpu: None,
    };
    if backend == crate::hardware::Device::Metal {
        model.gpu = Some(model.build_gpu().map_err(|e| format!("seq2seq: {e}"))?);
    }
    Ok(Arc::new(model))
}

// ---------------------------------------------------------------------------
// Metal programs (see `model::encdec_gpu`)
// ---------------------------------------------------------------------------

/// What the Metal graphs need besides the weights.
#[derive(Clone)]
struct Plan {
    d: usize,
    enc_heads: usize,
    dec_heads: usize,
    pre_ln: bool,
    ln_emb: bool,
    embed_scale: f32,
    act: ActKind,
    enc_layers: usize,
    dec_layers: usize,
    enc_embed: String,
    dec_embed: String,
    head: String,
    logits_bias: bool,
}

impl Seq2Seq {
    /// Moves the weights into the Metal encoder/decoder programs, leaving the CPU copies hollow.
    fn build_gpu(&mut self) -> Result<EncDec, String> {
        let d = self.d;
        let mut bag = Bag::default();
        let table = |bag: &mut Bag, name: &str, t: &Arc<Vec<f32>>| {
            bag.add(name, vec![t.len() / d, d], DType::F32, Tensor::F32((**t).clone()))
        };
        let (ee, de, lh) = (self.enc.embed.clone(), self.dec.embed.clone(), self.lm_head.clone());
        let enc_embed = if Arc::ptr_eq(&ee, &de) { "embed" } else { "enc.embed" };
        let dec_embed = if Arc::ptr_eq(&ee, &de) { "embed" } else { "dec.embed" };
        table(&mut bag, enc_embed, &ee);
        if dec_embed != enc_embed {
            table(&mut bag, dec_embed, &de);
        }
        let head = if Arc::ptr_eq(&lh, &de) {
            dec_embed
        } else if Arc::ptr_eq(&lh, &ee) {
            enc_embed
        } else {
            table(&mut bag, "lm_head", &lh);
            "lm_head"
        };
        let plan = Plan {
            d,
            enc_heads: self.enc.heads,
            dec_heads: self.dec.heads,
            pre_ln: self.pre_ln,
            ln_emb: self.enc.ln_emb.is_some(),
            embed_scale: self.embed_scale,
            act: self.act_kind,
            enc_layers: self.enc.layers.len(),
            dec_layers: self.dec.layers.len(),
            enc_embed: enc_embed.into(),
            dec_embed: dec_embed.into(),
            head: head.into(),
            logits_bias: self.logits_bias.is_some(),
        };
        drop((ee, de, lh));
        let empty = || Arc::new(Vec::new());
        (self.enc.embed, self.dec.embed, self.lm_head) = (empty(), empty(), empty());
        put_stack(&mut bag, "enc", &mut self.enc, d);
        put_stack(&mut bag, "dec", &mut self.dec, d);
        if let Some(b) = &mut self.logits_bias {
            let n = b.len();
            bag.add("logits_bias", vec![n], DType::F32, Tensor::F32(std::mem::take(b)));
        }
        let dims = Dims {
            layers: plan.dec_layers,
            inner: d,
            enc_cap: self.max_pos,
            self_cap: self.gen.max_len.max(2),
            row_offset: Some(self.dec.pos_offset),
            self_mask: false,
        };
        let sig = bag.sig();
        let dec = decoder_graph(&plan, &sig, &dims);
        let build_enc = Box::new(move |n| encoder_graph(&plan, &sig, n));
        EncDec::new(dims, bag, build_enc, dec)
    }
}

fn put_lin(bag: &mut Bag, name: &str, l: &mut Lin) {
    let out = l.b.len();
    let inp = l.w.len() / out.max(1);
    bag.add(&format!("{name}.w"), vec![out, inp], DType::F32, Tensor::F32(std::mem::take(&mut l.w)));
    bag.add(&format!("{name}.b"), vec![out], DType::F32, Tensor::F32(std::mem::take(&mut l.b)));
}

fn put_ln(bag: &mut Bag, name: &str, l: &mut Ln) {
    let n = l.g.len();
    bag.add(&format!("{name}.g"), vec![n], DType::F32, Tensor::F32(std::mem::take(&mut l.g)));
    bag.add(&format!("{name}.b"), vec![n], DType::F32, Tensor::F32(std::mem::take(&mut l.b)));
}

fn put_attn(bag: &mut Bag, p: &str, a: &mut Attn) {
    for (n, l) in [("q", &mut a.q), ("k", &mut a.k), ("v", &mut a.v), ("o", &mut a.o)] {
        put_lin(bag, &format!("{p}.{n}"), l);
    }
    put_ln(bag, &format!("{p}.ln"), &mut a.ln);
}

/// `{side}.pos`, `.lnemb`, `.final_ln`, and per layer `.{i}.sa/ca.{q,k,v,o,ln}`, `.fc1`, `.fc2`, `.ffln`.
fn put_stack(bag: &mut Bag, side: &str, s: &mut Stack, d: usize) {
    let pos = std::mem::take(&mut s.pos);
    bag.add(&format!("{side}.pos"), vec![pos.len() / d, d], DType::F32, Tensor::F32(pos));
    if let Some(l) = &mut s.ln_emb {
        put_ln(bag, &format!("{side}.lnemb"), l);
    }
    if let Some(l) = &mut s.final_ln {
        put_ln(bag, &format!("{side}.final_ln"), l);
    }
    for (i, l) in s.layers.iter_mut().enumerate() {
        let p = format!("{side}.{i}");
        put_attn(bag, &format!("{p}.sa"), &mut l.sa);
        if let Some(ca) = &mut l.ca {
            put_attn(bag, &format!("{p}.ca"), ca);
        }
        put_lin(bag, &format!("{p}.fc1"), &mut l.fc1);
        put_lin(bag, &format!("{p}.fc2"), &mut l.fc2);
        put_ln(bag, &format!("{p}.ffln"), &mut l.ff_ln);
    }
}

fn g_lin(g: &mut Graph, sig: &Sig, name: &str, x: ValueId, n: usize) -> ValueId {
    let (w, b) = (sig.w(g, &format!("{name}.w")), sig.w(g, &format!("{name}.b")));
    let rows = g.value(w).shape[0];
    eg::node(g, name.to_string(), Op::Linear, &[x, w, b], &[n, rows])
}

fn g_ln(g: &mut Graph, sig: &Sig, name: &str, x: ValueId, shape: &[usize]) -> ValueId {
    let (gamma, beta) = (sig.w(g, &format!("{name}.g")), sig.w(g, &format!("{name}.b")));
    eg::node(g, format!("{name}.out"), Op::LayerNorm { eps: 1e-5 }, &[x, gamma, beta], shape)
}

/// `x += f(norm(x))` (pre-LN) or `x = norm(x + f(x))` (post-LN).
fn g_residual(
    g: &mut Graph,
    sig: &Sig,
    p: &Plan,
    n: usize,
    ln: &str,
    x: ValueId,
    f: impl FnOnce(&mut Graph, ValueId) -> ValueId,
) -> ValueId {
    let shape = [n, p.d];
    let h = if p.pre_ln { g_ln(g, sig, ln, x, &shape) } else { x };
    let y = f(g, h);
    let s = eg::node(g, format!("{ln}.res"), Op::Binary { kind: crate::graph::BinaryKind::Add }, &[x, y], &shape);
    if p.pre_ln {
        s
    } else {
        g_ln(g, sig, ln, s, &shape)
    }
}

fn g_ffn(g: &mut Graph, sig: &Sig, p: &Plan, pre: &str, h: ValueId, n: usize) -> ValueId {
    let a = g_lin(g, sig, &format!("{pre}.fc1"), h, n);
    let shape = g.value(a).shape.clone();
    let op = match p.act {
        ActKind::Relu => Op::Unary { kind: crate::graph::UnaryKind::Relu },
        ActKind::Gelu => Op::Unary { kind: crate::graph::UnaryKind::GeluErf },
        ActKind::GeluTanh => Op::Unary { kind: crate::graph::UnaryKind::GeluTanh },
        ActKind::Silu => Op::Unary { kind: crate::graph::UnaryKind::Silu },
    };
    let a = eg::node(g, format!("{pre}.act"), op, &[a], &shape);
    g_lin(g, sig, &format!("{pre}.fc2"), a, n)
}

/// Scaled token embedding + position row (+ `layernorm_embedding`) of `n` ids.
#[allow(clippy::too_many_arguments)]
fn g_embed(
    g: &mut Graph,
    sig: &Sig,
    p: &Plan,
    side: &str,
    table: &str,
    ids: ValueId,
    rows: ValueId,
    n: usize,
) -> ValueId {
    let shape = [n, p.d];
    let t = sig.w(g, table);
    let mut e = eg::node(g, format!("{side}.tok"), Op::Embedding, &[t, ids], &shape);
    if p.embed_scale != 1.0 {
        e = eg::scale(g, format!("{side}.tok_scaled"), e, p.embed_scale, &shape);
    }
    let pt = sig.w(g, &format!("{side}.pos"));
    let pe = eg::node(g, format!("{side}.posrow"), Op::Embedding, &[pt, rows], &shape);
    let x = eg::node(g, format!("{side}.embed"), Op::Binary { kind: crate::graph::BinaryKind::Add }, &[e, pe], &shape);
    if p.ln_emb {
        g_ln(g, sig, &format!("{side}.lnemb"), x, &shape)
    } else {
        x
    }
}

fn mha(heads: usize) -> Op {
    Op::Attention { heads, kv_heads: heads, causal: false, window: None, soft_cap: None }
}

/// Encoder over `n` (padded) source rows: ids, position rows and a `[n]` key mask in; every
/// decoder layer's cross K/V projection (`xk`, `xv`, ...) out.
fn encoder_graph(p: &Plan, sig: &Sig, n: usize) -> Graph {
    let mut g = Graph::new();
    let ids = g.input("ids", vec![n], DType::U32);
    let rows = g.input("pos", vec![n], DType::U32);
    let mask = g.input("mask", vec![n], DType::F32);
    let mut x = g_embed(&mut g, sig, p, "enc", &p.enc_embed, ids, rows, n);
    for i in 0..p.enc_layers {
        let pre = format!("enc.{i}");
        x = g_residual(&mut g, sig, p, n, &format!("{pre}.sa.ln"), x, |g, h| {
            let q = g_lin(g, sig, &format!("{pre}.sa.q"), h, n);
            let k = g_lin(g, sig, &format!("{pre}.sa.k"), h, n);
            let v = g_lin(g, sig, &format!("{pre}.sa.v"), h, n);
            let a = eg::node(g, format!("{pre}.sa.attn"), mha(p.enc_heads), &[q, k, v, mask], &[n, p.d]);
            g_lin(g, sig, &format!("{pre}.sa.o"), a, n)
        });
        x = g_residual(&mut g, sig, p, n, &format!("{pre}.ffln"), x, |g, h| g_ffn(g, sig, p, &pre, h, n));
    }
    if p.pre_ln {
        x = g_ln(&mut g, sig, "enc.final_ln", x, &[n, p.d]);
    }
    for l in 0..p.dec_layers {
        for kv in ["k", "v"] {
            let out = g_lin(&mut g, sig, &format!("dec.{l}.ca.{kv}"), x, n);
            g.output(out);
        }
    }
    g
}

/// One decoder position: ids `[1]`, position `[1]`, position row `[1]` and the `[enc_cap]` cross
/// mask in, logits `[1, vocab]` out. Self-attention is a `CachedAttention` over `sk/sv.{l}`,
/// cross-attention an `Attention` over the `xk/xv.{l}` state.
fn decoder_graph(p: &Plan, sig: &Sig, dims: &Dims) -> Graph {
    let mut g = Graph::new();
    let ids = g.input("ids", vec![1], DType::U32);
    let pos = g.input("pos", vec![1], DType::U32);
    let rows = g.input("row", vec![1], DType::U32);
    let cross_mask = g.input("cross_mask", vec![dims.enc_cap], DType::F32);
    let mut x = g_embed(&mut g, sig, p, "dec", &p.dec_embed, ids, rows, 1);
    let hd = p.d / p.dec_heads;
    for i in 0..p.dec_layers {
        let pre = format!("dec.{i}");
        let (sk, sv) = eg::self_states(&mut g, i, dims);
        x = g_residual(&mut g, sig, p, 1, &format!("{pre}.sa.ln"), x, |g, h| {
            let q = g_lin(g, sig, &format!("{pre}.sa.q"), h, 1);
            let k = g_lin(g, sig, &format!("{pre}.sa.k"), h, 1);
            let v = g_lin(g, sig, &format!("{pre}.sa.v"), h, 1);
            let op = Op::CachedAttention {
                heads: p.dec_heads,
                kv_heads: p.dec_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
                window: None,
                soft_cap: None,
                write: true,
            };
            let a = eg::node(g, format!("{pre}.sa.attn"), op, &[q, k, v, sk, sv, pos], &[1, p.d]);
            g_lin(g, sig, &format!("{pre}.sa.o"), a, 1)
        });
        let (xk, xv) = eg::cross_states(&mut g, i, dims);
        x = g_residual(&mut g, sig, p, 1, &format!("{pre}.ca.ln"), x, |g, h| {
            let q = g_lin(g, sig, &format!("{pre}.ca.q"), h, 1);
            let a = eg::node(g, format!("{pre}.ca.attn"), mha(p.dec_heads), &[q, xk, xv, cross_mask], &[1, p.d]);
            g_lin(g, sig, &format!("{pre}.ca.o"), a, 1)
        });
        x = g_residual(&mut g, sig, p, 1, &format!("{pre}.ffln"), x, |g, h| g_ffn(g, sig, p, &pre, h, 1));
    }
    if p.pre_ln {
        x = g_ln(&mut g, sig, "dec.final_ln", x, &[1, p.d]);
    }
    let head = sig.w(&mut g, &p.head);
    let vocab = g.value(head).shape[0];
    let mut ins = vec![x, head];
    if p.logits_bias {
        ins.push(sig.w(&mut g, "logits_bias"));
    }
    let logits = eg::node(&mut g, "logits", Op::Linear, &ins, &[1, vocab]);
    g.output(logits);
    g
}

/// The fixed LCG of the other tiny fixtures.
struct Lcg(u64);

impl Lcg {
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }
    fn lin(&mut self, rows: usize, cols: usize) -> Lin {
        Lin { w: self.vec(rows * cols, 0.3), b: self.vec(rows, 0.1) }
    }
    fn ln(&mut self, n: usize) -> Ln {
        Ln { g: self.vec(n, 0.2).iter().map(|x| 1.0 + x).collect(), b: self.vec(n, 0.1) }
    }
}

/// A random encoder-decoder (2+2 layers, d_model 32, 4 heads, vocab 40, 80 positions) of
/// `family`'s layout for tests and the GPU fixture.
fn tiny(seed: u64, family: Family, beams: usize) -> Seq2Seq {
    let (d, vocab, max_pos, ffn) = (32usize, 40usize, 80usize, 64usize);
    let mut r = Lcg(seed);
    let pre_ln = matches!(family, Family::M2m100 | Family::MBart);
    let has_ln_emb = matches!(family, Family::MBart | Family::Bart);
    let embed = Arc::new(r.vec(vocab * d, 0.5));
    let stack = |r: &mut Lcg, decoder: bool| {
        let (pos, pos_offset) = match family {
            Family::Marian => (sinusoidal_marian(max_pos, d), 0),
            Family::M2m100 | Family::Fsmt => (sinusoidal_fairseq(max_pos + 2, d), 2),
            Family::MBart | Family::Bart => (r.vec((max_pos + 2) * d, 0.5), 2),
        };
        let attn = |r: &mut Lcg| Attn {
            q: r.lin(d, d),
            k: r.lin(d, d),
            v: r.lin(d, d),
            o: r.lin(d, d),
            ln: r.ln(d),
        };
        let layers = (0..2)
            .map(|_| Layer {
                sa: attn(r),
                ca: decoder.then(|| attn(r)),
                fc1: r.lin(ffn, d),
                fc2: r.lin(d, ffn),
                ff_ln: r.ln(d),
            })
            .collect();
        Stack {
            embed: embed.clone(),
            pos,
            pos_offset,
            ln_emb: has_ln_emb.then(|| r.ln(d)),
            layers,
            final_ln: pre_ln.then(|| r.ln(d)),
            heads: 4,
        }
    };
    let (enc, dec) = (stack(&mut r, false), stack(&mut r, true));
    let tied = family != Family::M2m100;
    let lm_head = if tied { embed.clone() } else { Arc::new(r.vec(vocab * d, 0.5)) };
    let logits_bias = matches!(family, Family::Marian | Family::Bart).then(|| r.vec(vocab, 0.2));
    let (act, act_kind): (fn(&mut [f32]), ActKind) = match family {
        Family::Marian => (|x: &mut [f32]| x.iter_mut().for_each(|v| *v = v.max(0.0)), ActKind::Relu),
        Family::M2m100 => (nn::silu, ActKind::Silu),
        _ => (nn::gelu, ActKind::Gelu),
    };
    Seq2Seq {
        family,
        d,
        enc,
        dec,
        lm_head,
        logits_bias,
        vocab,
        act,
        pre_ln,
        embed_scale: if matches!(family, Family::Marian | Family::M2m100) { (d as f32).sqrt() } else { 1.0 },
        max_pos,
        gen: Gen {
            beams,
            max_len: 30,
            min_len: 0,
            length_penalty: 1.0,
            no_repeat: 0,
            early_stopping: false,
            renormalize: false,
            forced_eos: None,
            bad: Vec::new(),
        },
        eos: 2,
        bos: 0,
        decoder_start: 2,
        forced_bos: None,
        source: Source::Hf(byte_passthrough().expect("byte tokenizer")),
        tok: Tokenizer::from_tokens((0..2 * vocab).map(|i| format!("t{i}")).collect()),
        alias: vocab as u32,
        pieces: Pieces::Spm,
        raw: HashMap::new(),
        langs: Vec::new(),
        src_default: None,
        tgt_default: None,
        legacy_mbart: false,
        marian_targets: Vec::new(),
        tgt_lang: String::new(),
        summary: "tiny seq2seq".into(),
        act_kind,
        gpu: None,
    }
}

/// Tiny random BART on Metal: encoder, a 3-beam search (forks beam state on the GPU) and greedy
/// steps; panics unless every node of every program ran on the GPU. The matrix test's fixture.
pub fn run_tiny_gpu_fixture() {
    let mut m = tiny(3, Family::Bart, 3);
    m.gpu = Some(m.build_gpu().expect("tiny seq2seq compiles on Metal"));
    let src = [5u32, 9, 12, 7, 20];
    let ctx = PrepareCtx { max_new_tokens: 8, cancel: None };
    let out = m.search(&src, &[m.decoder_start], &ctx).expect("beam search runs on Metal");
    assert!(!out.is_empty());
    let mut run = m.begin(&src).expect("encoder runs on Metal");
    let mut tok = m.decoder_start;
    for pos in 0..3 {
        let logits = m.advance(&mut run, &[tok], pos).expect("decoder runs on Metal");
        assert!(logits.iter().all(|v| v.is_finite()));
        tok = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
    }
    assert_eq!(m.gpu.as_ref().expect("built above").cpu_node_count(), 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moses_matches_sacremoses() {
        // sacremoses MosesTokenizer('en').tokenize(..., escape=True, aggressive_dash_splits=True)
        assert_eq!(
            moses_tokenize("Machine learning is great, isn't it?", "en"),
            "Machine learning is great , isn &apos;t it ?"
        );
        assert_eq!(
            moses_tokenize("Mr. Smith's e-mail costs $5,300.", "en"),
            "Mr. Smith &apos;s e @-@ mail costs $ 5,300 ."
        );
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum();
        (num / den).sqrt()
    }

    /// Per-step logits of the CPU forward vs the Metal programs, teacher-forced on the CPU's
    /// greedy tokens, for sources of a 64-bucket and a full-length-bucket size.
    fn step_parity(family: Family) {
        for src_len in [9usize, 70] {
            let cpu = tiny(5, family, 1);
            let mut gpu = tiny(5, family, 1);
            gpu.gpu = Some(gpu.build_gpu().expect("tiny seq2seq compiles on Metal"));
            let src: Vec<u32> = (0..src_len as u32).map(|i| 3 + (i * 7) % 30).collect();
            let _cpu_ok = crate::hardware::cpu_compute::allow();
            let (mut rc, mut rg) = (cpu.begin(&src).unwrap(), gpu.begin(&src).unwrap());
            let mut tok = cpu.decoder_start;
            let mut worst = 0.0f32;
            for pos in 0..12 {
                let a = cpu.advance(&mut rc, &[tok], pos).unwrap();
                let b = gpu.advance(&mut rg, &[tok], pos).unwrap();
                let e = rel_err(&a, &b);
                worst = worst.max(e);
                assert!(e < 1e-3, "{src_len}-token source, step {pos}: rel err {e}");
                tok = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap() as u32;
            }
            eprintln!("seq2seq {src_len}-token source: worst step rel err {worst:e}");
            assert_eq!(gpu.gpu.as_ref().unwrap().cpu_node_count(), 0);
        }
    }

    #[test]
    fn metal_matches_cpu_marian() {
        step_parity(Family::Marian);
    }

    #[test]
    fn metal_matches_cpu_bart() {
        step_parity(Family::Bart);
    }

    #[test]
    fn metal_matches_cpu_m2m100_prenorm() {
        step_parity(Family::M2m100);
    }

    #[test]
    fn metal_matches_cpu_mbart_prenorm_layernorm_embedding() {
        step_parity(Family::MBart);
    }

    /// Beam search (4 beams, host bookkeeping over GPU logits, beam state forked on the GPU) picks
    /// the same hypothesis as the CPU search, and the greedy `forward` path agrees.
    #[test]
    fn beam_search_matches_cpu() {
        for family in [Family::Marian, Family::Bart, Family::M2m100] {
            let cpu = tiny(11, family, 4);
            let mut gpu = tiny(11, family, 4);
            gpu.gpu = Some(gpu.build_gpu().expect("tiny seq2seq compiles on Metal"));
            let _cpu_ok = crate::hardware::cpu_compute::allow();
            let ctx = PrepareCtx { max_new_tokens: 12, cancel: None };
            let src: Vec<u32> = (0..11u32).map(|i| 3 + (i * 5) % 30).collect();
            let prompt = [cpu.decoder_start];
            let want = cpu.search(&src, &prompt, &ctx).unwrap();
            let got = gpu.search(&src, &prompt, &ctx).unwrap();
            assert_eq!(want, got, "{family:?}: beam hypotheses differ");
            assert!(!want.is_empty());
            assert_eq!(gpu.gpu.as_ref().unwrap().cpu_node_count(), 0);
        }
    }

    #[test]
    fn tiny_fixture_runs_on_metal() {
        run_tiny_gpu_fixture();
    }
}
