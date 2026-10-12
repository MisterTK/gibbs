//! BERT-family sequence classifiers and rerankers exported to GGUF by llama.cpp
//! (`convert_hf_to_gguf.py` BertModel / XLMRobertaModel): the encoder under llama.cpp tensor
//! names, the head as `cls.*` / `cls.output.*`, and the tokenizer as `tokenizer.ggml.*`.

use super::head::{HeadPool, HeadPost};
use super::{encode, Act, Head, Lin};
use crate::hardware::Device;
use crate::model::hub::HuggingFaceHub;
use crate::model::jev::{BertDims, BertEncoder};
use crate::model::loaders::{GgufWeightReader, WeightReader};
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

/// A llama.cpp BERT-architecture GGUF with a classification head.
pub struct GgufClassifier {
    pub labels: Vec<String>,
    enc: BertEncoder,
    tok: tokenizers::Tokenizer,
    head: Head,
}

impl GgufClassifier {
    pub fn load(path: &Path, device: Device) -> Result<Self, String> {
        let reader = GgufWeightReader::open(path)?;
        let kv = read_kv(reader.bytes())?;
        let arch = kv
            .get("general.architecture")
            .and_then(Kv::str)
            .unwrap_or("")
            .to_string();
        if !reader.tensor_names().any(|n| n == "cls.output.weight") {
            return Err("GGUF has no classification head".into());
        }
        if arch != "bert" {
            return Err(format!(
                "GGUF classifier architecture {arch} is not supported (only bert)"
            ));
        }
        let meta = |k: &str| {
            kv.get(&format!("{arch}.{k}"))
                .and_then(Kv::num)
                .ok_or_else(|| format!("GGUF has no {arch}.{k}"))
        };
        let h = meta("embedding_length")? as usize;
        // GGUF drops trailing unit dims ([h] for a 1-row table), so count rows from elements.
        let rows = |n: &str| {
            reader
                .get_tensor_shape(n)
                .map(|s| s.iter().product::<usize>() / h)
                .ok_or_else(|| format!("GGUF has no tensor {n}"))
        };
        let dims = BertDims {
            hidden_size: h,
            num_layers: meta("block_count")? as usize,
            num_heads: meta("attention.head_count")? as usize,
            intermediate_size: meta("feed_forward_length")? as usize,
            max_positions: rows("position_embd.weight")?,
            vocab_size: rows("token_embd.weight")?,
            type_vocab_size: rows("token_types.weight")?,
            layer_norm_eps: meta("attention.layer_norm_epsilon")? as f32,
        };
        let get = |n: &str| reader.load_tensor(n);
        let lin = |n: &str, in_f: usize| -> Result<Lin, String> {
            let w = get(&format!("{n}.weight"))?;
            Ok(Lin {
                out_f: w.len() / in_f,
                w,
                b: reader
                    .tensor_names()
                    .any(|t| *t == format!("{n}.bias"))
                    .then(|| get(&format!("{n}.bias")))
                    .transpose()?,
                in_f,
            })
        };
        // llama.cpp LLAMA_POOLING_TYPE_RANK: token 0 → [cls + tanh] → cls.output. BERT exports
        // drop the pooler, so their head is cls.output alone.
        let dense = reader
            .tensor_names()
            .any(|n| n == "cls.weight")
            .then(|| Ok::<_, String>((lin("cls", h)?, Act::Tanh)))
            .transpose()?;
        let in_out = dense.as_ref().map_or(h, |(l, _)| l.out_f);
        let head = Head {
            dense,
            out: lin("cls.output", in_out)?,
        };
        let n = head.out.out_f;
        // Metal runs the head (and the softmax over more than one logit) inside the encoder graph.
        let graph_head = (device != Device::Cpu)
            .then(|| head.graph(HeadPool::First, if n == 1 { HeadPost::Logits } else { HeadPost::Softmax }));
        let enc = BertEncoder::load_with_head(dims, device, |name, _| get(&gguf_name(name)?), graph_head)?;
        let labels = match kv
            .get(&format!("{arch}.classifier.output_labels"))
            .and_then(Kv::arr)
        {
            Some(l) if l.len() == n => l
                .iter()
                .map(|v| v.str().unwrap_or("").to_string())
                .collect(),
            _ => (0..n).map(|i| format!("LABEL_{i}")).collect(),
        };
        let tok = tokenizer(&kv, dims.max_positions)?;
        Ok(Self {
            labels,
            enc,
            tok,
            head,
        })
    }

    /// Head logits for `text` (and `pair` as the second segment), plus the token count; on Metal
    /// they are already softmaxed (a single logit stays raw).
    pub fn logits(&self, text: &str, pair: Option<&str>) -> Result<(Vec<f32>, usize), String> {
        let e = encode(&self.tok, text, pair)?;
        let ids = e.get_ids();
        // The converter drops RoBERTa's first pad + 1 position rows, so positions count from 0.
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let types = if self.enc.dims.type_vocab_size == 1 {
            vec![0; ids.len()]
        } else {
            e.get_type_ids().to_vec()
        };
        if self.enc.device() != Device::Cpu {
            return Ok((self.enc.run_head(ids, &positions, &types)?, ids.len()));
        }
        let hidden = self.enc.run(ids, &positions, &types, false)?;
        Ok((
            self.head.apply(&hidden[..self.enc.dims.hidden_size]),
            ids.len(),
        ))
    }

    pub fn device(&self) -> Device {
        self.enc.device()
    }
}

/// llama.cpp tensor name of a BERT encoder tensor.
fn gguf_name(bert: &str) -> Result<String, String> {
    Ok(match bert {
        "embeddings.word_embeddings.weight" => "token_embd.weight".into(),
        "embeddings.position_embeddings.weight" => "position_embd.weight".into(),
        "embeddings.token_type_embeddings.weight" => "token_types.weight".into(),
        "embeddings.LayerNorm.weight" => "token_embd_norm.weight".into(),
        "embeddings.LayerNorm.bias" => "token_embd_norm.bias".into(),
        _ => {
            let rest = bert
                .strip_prefix("encoder.layer.")
                .ok_or_else(|| format!("no GGUF name for {bert}"))?;
            let (l, t) = rest.split_once('.').unwrap_or((rest, ""));
            let (m, p) = t.rsplit_once('.').unwrap_or((t, ""));
            let g = match m {
                "attention.self.query" => "attn_q",
                "attention.self.key" => "attn_k",
                "attention.self.value" => "attn_v",
                "attention.output.dense" => "attn_output",
                "attention.output.LayerNorm" => "attn_output_norm",
                "intermediate.dense" => "ffn_up",
                "output.dense" => "ffn_down",
                "output.LayerNorm" => "layer_output_norm",
                _ => return Err(format!("no GGUF name for {bert}")),
            };
            format!("blk.{l}.{g}.{p}")
        }
    })
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
pub(super) fn hub_gguf_arch(hub: &HuggingFaceHub, repo_id: &str) -> Option<String> {
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
