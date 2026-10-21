//! Source and target text of the encoder-decoder translation / summarization models (Marian /
//! opus-mt, M2M100 + NLLB, MBart-50, FSMT / wmt19, BART / distilbart) and of prompt-in /
//! text-out ones (T5): [`EncDecText`] turns what the scheduler tokenized into encoder ids and the
//! decoder prompt, and shapes the output ids for display.
//!
//! Translation prompt convention: the text is the source. Language-coded models take a leading
//! `>>tgt<<` tag naming the target language (`>>fra<<` for Marian multi-target models, which
//! already use it; `>>fra_Latn<<` for NLLB, `>>fr<<` for M2M100, `>>fr_XX<<` for MBart-50) and an
//! optional `<<src>>` tag naming the source language (default: tokenizer_config `src_lang`, else
//! English). A multi-target model without a target tag is an error saying how to give one.

use crate::model::spm::SpmModel;
use crate::model::tokenizer::{byte_to_gpt2_unicode, ChatTemplateSpec, Tokenizer};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::{byte_level::ByteLevel, whitespace::WhitespaceSplit};

/// Renders the latest message alone: the encoder input is the user's text as is.
const LAST_MESSAGE_TEMPLATE: &str =
    "{% for m in messages %}{% if loop.last %}{{ m['content'] }}{% endif %}{% endfor %}";

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

/// The text side of a translation model (see the module docs).
pub struct Translation {
    family: Family,
    pub eos: u32,
    bos: u32,
    pub decoder_start: u32,
    forced_bos: Option<u32>,
    source: Source,
    /// Decoding side: target pieces, plus display aliases at `alias + id`.
    pub tok: Tokenizer,
    pub alias: u32,
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
    max_pos: usize,
}

/// What [`EncDecText::load`] builds: a translation text side, or a prompt one (the scheduler's
/// tokens + EOS are the encoder input; the decoder starts at `decoder_start`).
pub enum EncDecText {
    Prompt { tok: Tokenizer, eos: u32, decoder_start: u32 },
    Translation(Box<Translation>),
}

impl EncDecText {
    /// `kind` `prompt_eos` (T5) or `translation` over the checkpoint directory `dir`;
    /// `vocab` is the decoder's vocabulary (special ids must lie inside it).
    pub fn load(kind: &str, dir: &Path, cfg: &Value, vocab: usize) -> Result<Self, String> {
        match kind {
            "prompt_eos" => {
                let id = |k: &str, d: u64| cfg[k].as_u64().unwrap_or(d) as u32;
                let (decoder_start, eos) = (id("decoder_start_token_id", 0), id("eos_token_id", 1));
                Ok(Self::Prompt { tok: prompt_tokenizer(dir, eos)?, eos, decoder_start })
            }
            "translation" => Ok(Self::Translation(Box::new(Translation::load(dir, cfg, vocab)?))),
            k => Err(format!("unknown encoder-decoder text kind `{k}`")),
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Prompt { tok, .. } => tok,
            Self::Translation(t) => &t.tok,
        }
    }

    pub fn eos(&self) -> u32 {
        match self {
            Self::Prompt { eos, .. } => *eos,
            Self::Translation(t) => t.eos,
        }
    }

    /// Scheduler tokens -> (encoder ids, decoder prompt).
    pub fn source(&self, ids: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
        match self {
            Self::Prompt { eos, decoder_start, .. } => {
                // HF's T5 tokenizer ends the encoder input with </s>.
                let mut ids = ids.to_vec();
                if ids.last() != Some(eos) {
                    ids.push(*eos);
                }
                Ok((ids, vec![*decoder_start]))
            }
            Self::Translation(t) => t.source(ids),
        }
    }
}

/// tokenizer.json (Unigram) when present, else the SentencePiece `spiece.model`. Generation stops
/// at generation_config.json's `eos_token_id`, else config.json's (`eos`), and is greedy unless
/// generation_config.json sets `do_sample` (transformers' seq2seq default).
fn prompt_tokenizer(dir: &Path, eos: u32) -> Result<Tokenizer, String> {
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

impl Translation {
    /// Whether the output needs the whole hypothesis before display (FSMT's Moses spacing).
    pub fn needs_replay(&self) -> bool {
        self.pieces == Pieces::Fsmt
    }

    /// `(target languages)` note for the model summary, if any.
    pub fn languages(&self) -> usize {
        self.langs.len().max(self.marian_targets.len())
    }

    pub fn load(dir: &Path, cfg: &Value, vocab: usize) -> Result<Self, String> {
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
        let gen_cfg = read_json(dir, "generation_config.json").unwrap_or(Value::Null);
        let g = |k: &str| -> Value {
            match &gen_cfg[k] {
                Value::Null => cfg[k].clone(),
                v => v.clone(),
            }
        };
        let gu = |k: &str| g(k).as_u64().map(|v| v as u32);
        let tok_cfg = read_json(dir, "tokenizer_config.json").unwrap_or(Value::Null);
        let max_pos = cfg["max_position_embeddings"]
            .as_u64()
            .ok_or("config.json: max_position_embeddings missing")? as usize;
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
        let decoder_start = gu("decoder_start_token_id").unwrap_or(eos);
        let forced_bos = gu("forced_bos_token_id").filter(|_| langs.is_empty());
        if [eos, bos, pad, decoder_start]
            .iter()
            .any(|&t| t as usize >= vocab)
        {
            return Err("special token ids exceed the vocabulary".into());
        }
        Ok(Self {
            family,
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
            max_pos,
        })
    }

    /// Output ids as displayed: aliases (`alias + id`) carry the alternate piece text — the
    /// first piece without its leading space (SentencePiece / byte-level, as transformers'
    /// decode strips it), and FSMT word starts with a space (Moses detokenization).
    pub fn display(&self, out: &[u32]) -> Vec<u32> {
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
