//! GGUF key-value metadata read with arrays (the shared header reader skips them): the
//! architecture, string lists (a classifier's `<arch>.classifier.output_labels`) and the HF
//! tokenizer rebuilt from `tokenizer.ggml.*`.

use crate::model::hub::HuggingFaceHub;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

/// A GGUF metadata value.
enum Kv {
    Num(f64),
    Str(String),
    Bytes(Vec<u8>),
    Arr(Vec<Kv>),
}

impl Kv {
    fn num(&self) -> Option<f64> {
        match self {
            Kv::Num(v) => Some(*v),
            _ => None,
        }
    }
    fn str(&self) -> Option<&str> {
        match self {
            Kv::Str(s) => Some(s),
            _ => None,
        }
    }
    fn arr(&self) -> Option<&[Kv]> {
        match self {
            Kv::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// Parses the GGUF (v2/v3) key-value header, arrays included (the shared reader skips them).
fn read_kv(d: &[u8]) -> Result<HashMap<String, Kv>, String> {
    let mut o = 0usize;
    fn take<'a>(d: &'a [u8], o: &mut usize, n: usize) -> Result<&'a [u8], String> {
        let s = d
            .get(*o..*o + n)
            .ok_or("GGUF header is truncated".to_string())?;
        *o += n;
        Ok(s)
    }
    fn u32_(d: &[u8], o: &mut usize) -> Result<u32, String> {
        Ok(u32::from_le_bytes(take(d, o, 4)?.try_into().unwrap()))
    }
    fn u64_(d: &[u8], o: &mut usize) -> Result<u64, String> {
        Ok(u64::from_le_bytes(take(d, o, 8)?.try_into().unwrap()))
    }
    fn string(d: &[u8], o: &mut usize) -> Result<String, String> {
        let n = u64_(d, o)? as usize;
        Ok(String::from_utf8_lossy(take(d, o, n)?).into_owned())
    }
    fn value(d: &[u8], o: &mut usize, t: u32) -> Result<Kv, String> {
        Ok(match t {
            0 | 7 => Kv::Num(take(d, o, 1)?[0] as f64),
            1 => Kv::Num(take(d, o, 1)?[0] as i8 as f64),
            2 => Kv::Num(u16::from_le_bytes(take(d, o, 2)?.try_into().unwrap()) as f64),
            3 => Kv::Num(i16::from_le_bytes(take(d, o, 2)?.try_into().unwrap()) as f64),
            4 => Kv::Num(u32_(d, o)? as f64),
            5 => Kv::Num(u32_(d, o)? as i32 as f64),
            6 => Kv::Num(f32::from_bits(u32_(d, o)?) as f64),
            8 => Kv::Str(string(d, o)?),
            10 => Kv::Num(u64_(d, o)? as f64),
            11 => Kv::Num(u64_(d, o)? as i64 as f64),
            12 => Kv::Num(f64::from_bits(u64_(d, o)?)),
            9 => {
                let et = u32_(d, o)?;
                let n = u64_(d, o)? as usize;
                if et == 0 || et == 1 {
                    Kv::Bytes(take(d, o, n)?.to_vec())
                } else {
                    Kv::Arr((0..n).map(|_| value(d, o, et)).collect::<Result<_, _>>()?)
                }
            }
            t => return Err(format!("GGUF metadata type {t} is unknown")),
        })
    }
    if take(d, &mut o, 4)? != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    if u32_(d, &mut o)? < 2 {
        return Err("GGUF v1 is not supported".into());
    }
    let _tensors = u64_(d, &mut o)?;
    let n = u64_(d, &mut o)?;
    let mut kv = HashMap::new();
    for _ in 0..n {
        let k = string(d, &mut o)?;
        let t = u32_(d, &mut o)?;
        kv.insert(k, value(d, &mut o, t)?);
    }
    Ok(kv)
}

/// A GGUF file's metadata.
pub struct GgufMeta(HashMap<String, Kv>);

impl GgufMeta {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: read-only mapping of a cache file; only the header is read.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
        read_kv(&map).map(Self)
    }

    /// `general.architecture`.
    pub fn arch(&self) -> Option<String> {
        self.0.get("general.architecture").and_then(Kv::str).map(str::to_string)
    }

    /// The string array under `key`.
    pub fn strings(&self, key: &str) -> Option<Vec<String>> {
        Some(
            self.0
                .get(key)
                .and_then(Kv::arr)?
                .iter()
                .map(|v| v.str().unwrap_or("").to_string())
                .collect(),
        )
    }

    /// The HF tokenizer of `tokenizer.ggml.*`, truncating at `max_len`.
    pub fn tokenizer(&self, max_len: usize) -> Result<tokenizers::Tokenizer, String> {
        tokenizer(&self.0, max_len)
    }
}

/// Rebuilds the HF tokenizer from `tokenizer.ggml.*`: `bert` → WordPiece (llama.cpp's WPM
/// lowercases and splits like BertNormalizer/BertPreTokenizer), `t5` → SentencePiece Unigram with
/// the precompiled charsmap (XLM-R `tokenizer.json`: Precompiled + collapse spaces, Metaspace).
fn tokenizer(kv: &HashMap<String, Kv>, max_len: usize) -> Result<tokenizers::Tokenizer, String> {
    let model = kv
        .get("tokenizer.ggml.model")
        .and_then(Kv::str)
        .unwrap_or("");
    let tokens: Vec<&str> = kv
        .get("tokenizer.ggml.tokens")
        .and_then(Kv::arr)
        .ok_or("GGUF has no tokenizer.ggml.tokens")?
        .iter()
        .map(|t| t.str().unwrap_or(""))
        .collect();
    let types: Vec<i64> = match kv.get("tokenizer.ggml.token_type").and_then(Kv::arr) {
        Some(a) => a.iter().map(|v| v.num().unwrap_or(1.0) as i64).collect(),
        None => vec![1; tokens.len()],
    };
    let id = |s: &str| {
        tokens
            .iter()
            .position(|t| *t == s)
            .ok_or_else(|| format!("GGUF vocab has no {s}"))
    };
    // SentencePiece token types: 2 unknown, 3 control, 4 user-defined.
    let added: Vec<Value> = tokens
        .iter()
        .zip(&types)
        .enumerate()
        .filter(|(_, (_, &t))| (2..=4).contains(&t))
        .map(|(i, (s, &t))| {
            json!({"id": i, "content": s, "single_word": false, "lstrip": false, "rstrip": false,
                   "normalized": t == 4, "special": t != 4})
        })
        .collect();
    let spec = match model {
        "t5" => {
            let scores = kv
                .get("tokenizer.ggml.scores")
                .and_then(Kv::arr)
                .ok_or("GGUF has no tokenizer.ggml.scores")?;
            let vocab: Vec<Value> = tokens
                .iter()
                .zip(scores)
                .map(|(t, s)| json!([t, s.num().unwrap_or(0.0)]))
                .collect();
            let unk = types.iter().position(|&t| t == 2).unwrap_or(id("<unk>")?);
            let flag = |k: &str| kv.get(k).and_then(Kv::num).map_or(true, |v| v != 0.0);
            let mut norm = vec![];
            if let Some(Kv::Bytes(map)) = kv.get("tokenizer.ggml.precompiled_charsmap") {
                norm.push(json!({"type": "Precompiled",
                    "precompiled_charsmap": crate::model::image::base64_encode(map)}));
            }
            if flag("tokenizer.ggml.remove_extra_whitespaces") {
                norm.push(
                    json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}),
                );
            }
            let prepend = if flag("tokenizer.ggml.add_space_prefix") {
                "always"
            } else {
                "never"
            };
            let (cls, sep) = (id("<s>")?, id("</s>")?);
            json!({
                "model": {"type": "Unigram", "unk_id": unk, "vocab": vocab, "byte_fallback": false},
                "normalizer": {"type": "Sequence", "normalizers": norm},
                "pre_tokenizer": {"type": "Metaspace", "replacement": "\u{2581}",
                                  "prepend_scheme": prepend, "split": true},
                "post_processor": {"type": "RobertaProcessing", "sep": ["</s>", sep],
                    "cls": ["<s>", cls], "trim_offsets": true, "add_prefix_space": true},
                "decoder": null,
            })
        }
        "bert" => {
            // The converter writes word starts as "▁w", continuations as "w", controls verbatim.
            let vocab: serde_json::Map<String, Value> = tokens
                .iter()
                .zip(&types)
                .enumerate()
                .map(|(i, (t, &ty))| {
                    let w = if ty == 3 {
                        t.to_string()
                    } else if let Some(w) = t.strip_prefix('\u{2581}') {
                        w.to_string()
                    } else {
                        format!("##{t}")
                    };
                    (w, json!(i))
                })
                .collect();
            let (cls, sep) = (id("[CLS]")?, id("[SEP]")?);
            json!({
                "model": {"type": "WordPiece", "unk_token": "[UNK]",
                          "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                          "vocab": vocab},
                "normalizer": {"type": "BertNormalizer", "clean_text": true,
                    "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
                "pre_tokenizer": {"type": "BertPreTokenizer"},
                "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", sep],
                                   "cls": ["[CLS]", cls]},
                "decoder": null,
            })
        }
        m => {
            return Err(format!(
                "GGUF tokenizer model {m:?} is not supported (bert, t5)"
            ))
        }
    };
    let mut spec = spec;
    spec["version"] = json!("1.0");
    spec["truncation"] = Value::Null;
    spec["padding"] = Value::Null;
    spec["added_tokens"] = Value::Array(added);
    let mut tok: tokenizers::Tokenizer = spec
        .to_string()
        .parse()
        .map_err(|e| format!("GGUF tokenizer: {e}"))?;
    tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length: max_len,
        ..Default::default()
    }))
    .map_err(|e| format!("tokenizer truncation: {e}"))?;
    Ok(tok)
}

/// `general.architecture` of `repo_id`'s GGUF weights: the cached file's header, else the Hub's
/// `gguf` metadata. `None` when the repo has no GGUF.
pub fn hub_arch(hub: &HuggingFaceHub, repo_id: &str) -> Option<String> {
    if let Some(p) = hub
        .locate_model(repo_id)
        .weights_path
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
    {
        let file = std::fs::File::open(&p).ok()?;
        // SAFETY: read-only mapping of a cache file; only the header is read.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        return read_kv(&map)
            .ok()?
            .get("general.architecture")
            .and_then(Kv::str)
            .map(str::to_string);
    }
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-fsSL", "-m", "15"]);
    if let Ok(t) = std::env::var("HF_TOKEN") {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let out = cmd
        .arg(format!(
            "https://huggingface.co/api/models/{repo_id}?expand%5B%5D=gguf"
        ))
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    let v: Value = serde_json::from_slice(&out.stdout).ok()?;
    v["gguf"]["architecture"].as_str().map(str::to_string)
}
