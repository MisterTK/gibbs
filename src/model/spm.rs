//! SentencePiece models (`spiece.model`, Marian `source.spm`, `sentencepiece.bpe.model`): the
//! `ModelProto` protobuf is read by hand; segmentation runs in HF `tokenizers` (Unigram Viterbi
//! or BPE merges) behind spm's own precompiled normalizer and a `▁` Metaspace pre-tokenizer,
//! like HF's slow→fast `SpmConverter`.

use crate::model::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::models::{bpe::BPE, unigram::Unigram};
use tokenizers::normalizers::{
    replace::ReplacePattern, NormalizerWrapper, Precompiled, Replace, Sequence, Strip,
};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

/// `SentencePiece.Type`: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6.
pub const UNKNOWN: u64 = 2;
pub const CONTROL: u64 = 3;

pub struct Piece {
    pub text: String,
    pub score: f32,
    pub kind: u64,
}

pub struct SpmModel {
    /// In spm id order.
    pub pieces: Vec<Piece>,
    /// `TrainerSpec.model_type == BPE` (else Unigram; WORD/CHAR models are refused).
    pub bpe: bool,
    pub add_dummy_prefix: bool,
    byte_fallback: bool,
    remove_extra_whitespaces: bool,
    pub charsmap: Vec<u8>,
}

/// Protobuf wire reader: `(field number, varint / fixed bits, length-delimited bytes)`.
fn fields(mut b: &[u8]) -> impl Iterator<Item = Result<(u64, u64, &[u8]), String>> {
    fn varint(b: &mut &[u8]) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = b.split_first().ok_or("truncated varint")?;
            *b = rest;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("varint too long".into())
    }
    fn take<'a>(b: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
        if b.len() < n {
            return Err("truncated field".into());
        }
        let (h, t) = b.split_at(n);
        *b = t;
        Ok(h)
    }
    fn field<'a>(b: &mut &'a [u8]) -> Result<(u64, u64, &'a [u8]), String> {
        let key = varint(b)?;
        Ok(match key & 7 {
            0 => (key >> 3, varint(b)?, &[][..]),
            1 => (
                key >> 3,
                u64::from_le_bytes(take(b, 8)?.try_into().unwrap()),
                &[][..],
            ),
            2 => {
                let n = varint(b)? as usize;
                (key >> 3, 0, take(b, n)?)
            }
            5 => (
                key >> 3,
                u32::from_le_bytes(take(b, 4)?.try_into().unwrap()) as u64,
                &[][..],
            ),
            w => return Err(format!("unsupported protobuf wire type {w}")),
        })
    }
    std::iter::from_fn(move || (!b.is_empty()).then(|| field(&mut b)))
}

impl SpmModel {
    pub fn open(path: &Path) -> Result<Self, String> {
        let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&buf).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(buf: &[u8]) -> Result<Self, String> {
        let mut m = SpmModel {
            pieces: Vec::new(),
            bpe: false,
            add_dummy_prefix: true,
            byte_fallback: false,
            remove_extra_whitespaces: true,
            charsmap: Vec::new(),
        };
        for f in fields(buf) {
            match f? {
                (1, _, piece) => {
                    let mut p = Piece {
                        text: String::new(),
                        score: 0.0,
                        kind: 1,
                    };
                    for f in fields(piece) {
                        match f? {
                            (1, _, s) => {
                                p.text = String::from_utf8(s.to_vec())
                                    .map_err(|_| "piece is not UTF-8")?
                            }
                            (2, bits, _) => p.score = f32::from_bits(bits as u32),
                            (3, kind, _) => p.kind = kind,
                            _ => {}
                        }
                    }
                    m.pieces.push(p);
                }
                (2, _, trainer) => {
                    for f in fields(trainer) {
                        match f? {
                            (3, 1, _) => m.bpe = false,
                            (3, 2, _) => m.bpe = true,
                            (3, t, _) => {
                                return Err(format!(
                                    "SentencePiece model_type {t} (WORD/CHAR) is not supported"
                                ))
                            }
                            (35, v, _) => m.byte_fallback = v != 0,
                            _ => {}
                        }
                    }
                }
                (3, _, normalizer) => {
                    for f in fields(normalizer) {
                        match f? {
                            (2, _, map) => m.charsmap = map.to_vec(),
                            (3, v, _) => m.add_dummy_prefix = v != 0,
                            (4, v, _) => m.remove_extra_whitespaces = v != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if m.pieces.is_empty() {
            return Err("SentencePiece model has no pieces".into());
        }
        Ok(m)
    }

    /// piece -> spm id.
    pub fn piece_ids(&self) -> HashMap<String, u32> {
        let mut ids = HashMap::with_capacity(self.pieces.len());
        for (i, p) in self.pieces.iter().enumerate() {
            ids.entry(p.text.clone()).or_insert(i as u32);
        }
        ids
    }

    /// The segmentation pipeline with token ids from `vocab` (piece -> id; spm's own
    /// [`Self::piece_ids`] or a repo's `vocab.json`). `vocab` entries spm lacks are never chosen
    /// over spm's pieces; spm pieces `vocab` lacks are not produced.
    pub fn segmenter(&self, vocab: &HashMap<String, u32>) -> Result<tokenizers::Tokenizer, String> {
        let spm: HashMap<&str, (usize, &Piece)> = self
            .pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.text.as_str(), (i, p)))
            .collect();
        let unk = self
            .pieces
            .iter()
            .find(|p| p.kind == UNKNOWN)
            .and_then(|p| vocab.get(&p.text))
            .or_else(|| vocab.get("<unk>"))
            .copied();
        let mut hf = if self.bpe {
            // spm BPE merges the adjacent pair whose union scores highest: HF BPE ranks each
            // split of a piece by the piece's score (SentencePieceExtractor).
            let mut merges: Vec<(f32, usize, usize, &str, &str)> = Vec::new();
            for p in self
                .pieces
                .iter()
                .filter(|p| p.kind == 1 && vocab.contains_key(&p.text))
            {
                for (cut, _) in p.text.char_indices().skip(1) {
                    let (l, r) = p.text.split_at(cut);
                    if let (Some(&(li, _)), Some(&(ri, _))) = (spm.get(l), spm.get(r)) {
                        if vocab.contains_key(l) && vocab.contains_key(r) {
                            merges.push((p.score, li, ri, l, r));
                        }
                    }
                }
            }
            merges.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
            let mut b = BPE::builder()
                .vocab_and_merges(
                    vocab
                        .iter()
                        .map(|(k, &v)| (k.clone(), v))
                        .collect::<tokenizers::models::bpe::Vocab>(),
                    merges
                        .into_iter()
                        .map(|m| (m.3.to_string(), m.4.to_string()))
                        .collect(),
                )
                .fuse_unk(true)
                .byte_fallback(self.byte_fallback);
            if let Some(u) = unk {
                let name = vocab.iter().find(|(_, &v)| v == u).map(|(k, _)| k.clone());
                b = b.unk_token(name.unwrap_or_default());
            }
            tokenizers::Tokenizer::new(b.build().map_err(|e| e.to_string())?)
        } else {
            // Unigram: ids index the vocab, so it is laid out in `vocab` id order.
            const UNREACHABLE: f64 = -1e6;
            let n = vocab.values().max().map_or(0, |&m| m as usize + 1);
            let mut table: Vec<(String, f64)> = (0..n)
                .map(|i| (format!("<unused-{i}>"), UNREACHABLE))
                .collect();
            for (text, &id) in vocab {
                let score = match spm.get(text.as_str()) {
                    Some((_, p)) if p.kind != CONTROL && p.kind != UNKNOWN => p.score as f64,
                    _ => UNREACHABLE,
                };
                table[id as usize] = (text.clone(), score);
            }
            let model = Unigram::from(table, unk.map(|u| u as usize), self.byte_fallback)
                .map_err(|e| e.to_string())?;
            tokenizers::Tokenizer::new(model)
        };
        let mut norm: Vec<NormalizerWrapper> = Vec::new();
        if !self.charsmap.is_empty() {
            norm.push(
                Precompiled::from(&self.charsmap)
                    .map_err(|e| format!("precompiled_charsmap: {e}"))?
                    .into(),
            );
        }
        if self.remove_extra_whitespaces {
            norm.push(Strip::new(true, true).into());
            norm.push(
                Replace::new(ReplacePattern::Regex(" {2,}".into()), " ")
                    .map_err(|e| e.to_string())?
                    .into(),
            );
        }
        hf.with_normalizer(Some(Sequence::new(norm)));
        let scheme = if self.add_dummy_prefix {
            PrependScheme::Always
        } else {
            PrependScheme::Never
        };
        hf.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', scheme, true)));
        Ok(hf)
    }
}

/// A SentencePiece model as a [`Tokenizer`] whose ids are spm's own (e.g. T5's `spiece.model`).
/// `encode` adds no BOS/EOS; control and unknown pieces count as special.
pub fn load(path: &Path) -> Result<Tokenizer, String> {
    let m = SpmModel::open(path)?;
    let vocab = m.piece_ids();
    let hf = m.segmenter(&vocab)?;
    Ok(Tokenizer {
        inv_vocab: vocab.iter().map(|(k, &v)| (v, k.clone())).collect(),
        special_tokens: m
            .pieces
            .iter()
            .filter(|p| p.kind == CONTROL || p.kind == UNKNOWN)
            .map(|p| (p.text.clone(), vocab[&p.text]))
            .collect(),
        vocab,
        spm: true,
        add_space_prefix: m.add_dummy_prefix,
        hf: Some(Arc::new(hf)),
        // `..` keeps this literal valid as `Tokenizer` gains fields (e.g. stop ids).
        ..Tokenizer::from_tokens(Vec::new())
    })
}
