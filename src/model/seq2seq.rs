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
}

/// Beam search (`num_beams` > 1): the chosen output, replayed by `forward`.
struct Replay {
    prompt: usize,
    out: Vec<u32>,
}

/// Greedy configs: the decoder runs step by step in `forward`, so sampling, streaming and
/// cancellation work as for any model.
struct Live {
    cross: Vec<(Vec<f32>, Vec<f32>)>,
    n_src: usize,
    cache: Cache,
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

    /// transformers' beam search: the best hypothesis after `prompt`, ending in EOS unless
    /// `max_length` or the request's `max_new_tokens` cut it; Err once `cancel` is set.
    fn search(&self, src: &[u32], prompt: &[u32], ctx: &PrepareCtx) -> Result<Vec<u32>, String> {
        let n = src.len();
        let cross = self.cross(src);
        let mut caches = vec![self.empty_cache()];
        let mut logits = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            logits = self.decode_step(&[t], p, &mut caches, &cross, n);
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
            caches = next.iter().map(|&(bi, _, _)| caches[bi].clone()).collect();
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
            logits = self.decode_step(&toks, cur, &mut caches, &cross, n);
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
        crate::scheduler::cpu_backend_label()
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
                cross: self.cross(&src),
                n_src: src.len(),
                cache: self.empty_cache(),
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
            lp = self.decode_step(
                &[t],
                pos,
                std::slice::from_mut(&mut live.cache),
                &live.cross,
                live.n_src,
            );
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
    _mode: impl Into<crate::hardware::PlacementMode>,
) -> Result<Arc<dyn NativeLm>, String> {
    crate::graph::avenues::gate(crate::graph::avenues::Family::Seq2Seq)?;
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
    let act: fn(&mut [f32]) = match cfg["activation_function"].as_str().unwrap_or("gelu") {
        "relu" => |x: &mut [f32]| x.iter_mut().for_each(|v| *v = v.max(0.0)),
        "gelu" => nn::gelu,
        "gelu_new" | "gelu_pytorch_tanh" => nn::gelu_tanh,
        "swish" | "silu" => nn::silu,
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
    Ok(Arc::new(Seq2Seq {
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
    }))
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
}
