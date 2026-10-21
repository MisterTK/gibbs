//! Slow-tokenizer files rebuilt as the fast tokenizer transformers converts them to, for repos
//! without tokenizer.json. Each family spec lists its conversions in `runtime.tokenizer` (tried in
//! order, the first whose `when` holds); the kinds are the converter algorithms:
//! - `wordpiece`: a `vocab.txt` WordPiece with the clean-text/CJK normalizer, its word splitter
//!   and the `cls A sep (B sep)` post-processor (segment 1 on the pair);
//! - `byte_bpe`: a byte-level BPE (`vocab.json` + `merges.txt`), post-processed `roberta`
//!   (`cls A sep sep B sep`, prefix space per `add_prefix_space`) or `cls_sep` (as `wordpiece`);
//! - `unigram`: a SentencePiece `.model` (Unigram only) with its pieces as stored (`spm`) or behind
//!   fairseq's four leading ids (`fairseq`), optional quote/accent/case normalization and a pair
//!   template.

use crate::model::spm::{SpmModel, CONTROL, UNKNOWN};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::{
    BertNormalizer, Lowercase, NormalizerWrapper, Precompiled, Replace, Sequence, StripAccents, NFKD,
};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer};

/// One conversion of `runtime.tokenizer`.
#[derive(Debug, Clone, Deserialize)]
pub struct Converter {
    /// Expression over the family's config; absent = always.
    #[serde(default)]
    pub when: Option<String>,
    #[serde(flatten)]
    pub kind: Kind,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Kind {
    Wordpiece {
        vocab: String,
        unk: String,
        cls: String,
        sep: String,
        specials: Vec<String>,
    },
    ByteBpe {
        vocab: String,
        merges: String,
        cls: String,
        sep: String,
        /// `roberta` | `cls_sep`.
        post: String,
        specials: Vec<String>,
    },
    Unigram {
        model: String,
        /// `spm` (pieces as stored) | `fairseq` (fairseq's leading ids, `<mask>` last).
        #[serde(default = "spm")]
        vocab: String,
        /// Pieces ending in "<digit>," lose 100 from their score.
        #[serde(default)]
        demote_number_comma: bool,
        /// `` and '' become ".
        #[serde(default)]
        replace_quotes: bool,
        /// Accents stripped unless tokenizer_config `keep_accents`.
        #[serde(default)]
        strip_accents: bool,
        /// Lowercased unless tokenizer_config `do_lower_case` is false.
        #[serde(default)]
        lowercase: bool,
        cls: String,
        sep: String,
        /// Pair template over `{cls}`, `{sep}`, `$A`, `$B` (`:1` segment ids).
        pair: String,
    },
}

fn spm() -> String {
    "spm".into()
}

impl Kind {
    /// The files this conversion reads.
    pub fn files(&self) -> Vec<&str> {
        match self {
            Kind::Wordpiece { vocab, .. } => vec![vocab],
            Kind::ByteBpe { vocab, merges, .. } => vec![vocab, merges],
            Kind::Unigram { model, .. } => vec![model],
        }
    }

    /// Builds the tokenizer from `files` (in [`Self::files`] order), without truncation.
    pub fn build(&self, files: &[&Path], tok_cfg: &Value) -> Result<Tokenizer, String> {
        match self {
            Kind::Wordpiece { unk, cls, sep, specials, .. } => {
                let wp = WordPiece::from_file(&files[0].to_string_lossy())
                    .unk_token(unk.clone())
                    .build()
                    .map_err(|e| format!("{}: {e}", files[0].display()))?;
                let mut t = Tokenizer::new(wp);
                let lower = tok_cfg["do_lower_case"].as_bool().unwrap_or(true);
                t.with_normalizer(Some(BertNormalizer::new(true, true, None, lower)));
                t.with_pre_tokenizer(Some(wrapper::<PreTokenizerWrapper>(json!({"type": "BertPreTokenizer"}))?));
                cls_sep(&mut t, cls, sep, files[0])?;
                add_specials(&mut t, specials);
                Ok(t)
            }
            Kind::ByteBpe { cls, sep, post, specials, .. } => {
                let bpe = BPE::from_file(&files[0].to_string_lossy(), &files[1].to_string_lossy())
                    .build()
                    .map_err(|e| format!("{}: {e}", files[0].display()))?;
                let mut t = Tokenizer::new(bpe);
                let prefix = tok_cfg["add_prefix_space"].as_bool().unwrap_or(false);
                t.with_pre_tokenizer(Some(ByteLevel::new(prefix, true, true)));
                match post.as_str() {
                    "roberta" => {
                        let (c, s) = (id(&t, cls, files[0])?, id(&t, sep, files[0])?);
                        t.with_post_processor(Some(
                            RobertaProcessing::new((sep.clone(), s), (cls.clone(), c)).add_prefix_space(prefix),
                        ));
                    }
                    "cls_sep" => cls_sep(&mut t, cls, sep, files[0])?,
                    p => return Err(format!("byte_bpe post `{p}` is not roberta|cls_sep")),
                }
                add_specials(&mut t, specials);
                Ok(t)
            }
            Kind::Unigram {
                vocab,
                demote_number_comma,
                replace_quotes,
                strip_accents,
                lowercase,
                cls,
                sep,
                pair,
                ..
            } => {
                let path = files[0];
                let m = SpmModel::open(path)?;
                if m.bpe {
                    return Err(format!(
                        "{}: SentencePiece BPE model is not Unigram; only Unigram models are rebuilt",
                        path.display()
                    ));
                }
                let err = |e: tokenizers::Error| e.to_string();
                let mut norm: Vec<NormalizerWrapper> = Vec::new();
                let mut specials: Vec<AddedToken> = Vec::new();
                let (pieces, unk) = match vocab.as_str() {
                    "fairseq" => {
                        // fairseq's four leading ids, spm pieces after its <unk>, <mask> last.
                        let mut v: Vec<(String, f64)> = [
                            ("<s>NOTUSED", 0.0),
                            ("<pad>", 0.0),
                            ("</s>NOTUSED", 0.0),
                            ("<unk>", 0.0),
                            ("<unk>NOTUSED", -100.0),
                        ]
                        .iter()
                        .map(|&(t, s)| (t.to_string(), s))
                        .collect();
                        v.extend(m.pieces[1..].iter().map(|p| (p.text.clone(), p.score as f64)));
                        v.push(("<mask>".into(), 0.0));
                        specials.extend(
                            ["<s>NOTUSED", "<pad>", "</s>NOTUSED", "<unk>"]
                                .iter()
                                .map(|s| AddedToken::from(*s, true)),
                        );
                        specials.push(AddedToken::from("<mask>", true).lstrip(true));
                        (v, 3)
                    }
                    "spm" => {
                        let v = m
                            .pieces
                            .iter()
                            .map(|p| {
                                let c: Vec<char> = p.text.chars().collect();
                                let number_comma = *demote_number_comma
                                    && c.len() >= 2
                                    && c[c.len() - 1] == ','
                                    && c[c.len() - 2].is_numeric();
                                (p.text.clone(), p.score as f64 - if number_comma { 100.0 } else { 0.0 })
                            })
                            .collect();
                        let unk = m
                            .pieces
                            .iter()
                            .position(|p| p.kind == UNKNOWN)
                            .ok_or_else(|| format!("{}: no unknown piece", path.display()))?;
                        (v, unk)
                    }
                    v => return Err(format!("unigram vocab `{v}` is not spm|fairseq")),
                };
                if *replace_quotes {
                    norm.push(Replace::new("``", "\"").map_err(err)?.into());
                    norm.push(Replace::new("''", "\"").map_err(err)?.into());
                }
                if *strip_accents && !tok_cfg["keep_accents"].as_bool().unwrap_or(false) {
                    norm.push(NFKD.into());
                    norm.push(StripAccents.into());
                }
                if *lowercase && tok_cfg["do_lower_case"].as_bool().unwrap_or(true) {
                    norm.push(Lowercase.into());
                }
                // Control and unknown pieces are special tokens.
                specials.extend(
                    m.pieces
                        .iter()
                        .filter(|p| p.kind == CONTROL || p.kind == UNKNOWN)
                        .map(|p| AddedToken::from(p.text.clone(), true)),
                );
                if !m.charsmap.is_empty() {
                    norm.push(
                        Precompiled::from(&m.charsmap)
                            .map_err(|e| format!("{}: precompiled_charsmap: {e}", path.display()))?
                            .into(),
                    );
                }
                let mut t = Tokenizer::new(Unigram::from(pieces, Some(unk), false).map_err(err)?);
                t.with_normalizer(Some(Sequence::new(norm)));
                t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::sequence::Sequence::new(vec![
                    WhitespaceSplit.into(),
                    Metaspace::new('▁', PrependScheme::Always, true).into(),
                ])));
                t.add_special_tokens(&specials);
                let (cls_id, sep_id) = (id(&t, cls, path)?, id(&t, sep, path)?);
                let pair = pair.replace("{cls}", cls).replace("{sep}", sep);
                let post = TemplateProcessing::builder()
                    .try_single(format!("{cls} $A {sep}"))
                    .and_then(|b| b.try_pair(pair))
                    .map_err(|e| e.to_string())?
                    .special_tokens(vec![(cls.as_str(), cls_id), (sep.as_str(), sep_id)])
                    .build()
                    .map_err(|e| e.to_string())?;
                t.with_post_processor(Some(post));
                Ok(t)
            }
        }
    }
}

fn id(t: &Tokenizer, s: &str, file: &Path) -> Result<u32, String> {
    t.token_to_id(s)
        .ok_or_else(|| format!("{}: no {s} token", file.display()))
}

/// `cls A sep` / `cls A sep B sep` (segment 1 on B and its sep).
fn cls_sep(t: &mut Tokenizer, cls: &str, sep: &str, file: &Path) -> Result<(), String> {
    let (c, s) = (id(t, cls, file)?, id(t, sep, file)?);
    t.with_post_processor(Some(wrapper::<PostProcessorWrapper>(
        json!({"type": "BertProcessing", "sep": [sep, s], "cls": [cls, c]}),
    )?));
    Ok(())
}

fn add_specials(t: &mut Tokenizer, specials: &[String]) {
    let s: Vec<AddedToken> = specials.iter().map(|s| AddedToken::from(s.clone(), true)).collect();
    t.add_special_tokens(&s);
}

/// A tokenizer component from its tokenizer.json form.
fn wrapper<T: serde::de::DeserializeOwned>(j: Value) -> Result<T, String> {
    serde_json::from_value(j).map_err(|e| format!("tokenizer component: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rebuilt SentencePiece tokenizers match the tokenizer.json transformers converted from
    /// the same files (albert-base-v2 shares tals/albert-xlarge-vitaminc-mnli's spiece.model,
    /// camembert-base cmarkea/distilcamembert-base-sentiment's sentencepiece.bpe.model), with the
    /// conversions the albert and bert specs declare.
    #[test]
    fn rebuilt_spm_matches_converted_tokenizer_json() {
        let reg = crate::arch::registry().unwrap();
        let conv = |family: &str, arch: &str| -> Kind {
            let sig = crate::arch::Signals { hf_architectures: vec![arch.into()], ..Default::default() };
            let spec = reg.spec(family, &sig).unwrap();
            let rt: Value = spec.runtime["tokenizer"].clone();
            let list: Vec<Converter> = serde_json::from_value(rt).unwrap();
            list.into_iter()
                .find(|c| matches!(c.kind, Kind::Unigram { .. }))
                .unwrap()
                .kind
        };
        let snap = |r: &str| format!("disk/models--{}/snapshots/main", r.replace('/', "--"));
        let texts = [
            "J'aime me promener en forêt même si ça me donne mal aux pieds.",
            "``Héllo''  WORLD, in 2010, 1,000 cafés naïve — ok?",
            "  The Eiffel Tower is located in Paris.   ",
        ];
        for (kind, model, reference) in [
            (
                conv("albert", "AlbertForSequenceClassification"),
                "tals/albert-xlarge-vitaminc-mnli/spiece.model",
                "albert/albert-base-v2",
            ),
            (
                conv("bert", "CamembertForSequenceClassification"),
                "cmarkea/distilcamembert-base-sentiment/sentencepiece.bpe.model",
                "almanach/camembert-base",
            ),
        ] {
            let (repo, file) = model.rsplit_once('/').unwrap();
            let model = std::path::PathBuf::from(snap(repo)).join(file);
            let reference = std::path::PathBuf::from(snap(reference)).join("tokenizer.json");
            if !model.exists() || !reference.exists() {
                eprintln!("SKIP: {} or {} not cached", model.display(), reference.display());
                continue;
            }
            let ours = kind.build(&[model.as_path()], &Value::Null).unwrap();
            let theirs = Tokenizer::from_file(&reference).unwrap();
            for t in texts {
                let (a, b) = (ours.encode(t, true).unwrap(), theirs.encode(t, true).unwrap());
                assert_eq!(a.get_ids(), b.get_ids(), "{t:?} {:?}", a.get_tokens());
                let (a, b) = (
                    ours.encode((t, "Second segment."), true).unwrap(),
                    theirs.encode((t, "Second segment."), true).unwrap(),
                );
                assert_eq!(a.get_ids(), b.get_ids());
                assert_eq!(a.get_type_ids(), b.get_type_ids());
            }
        }
    }

    /// `wordpiece` from a vocab.txt: `cls A sep B sep` with segment 1 on the pair.
    #[test]
    fn wordpiece_conversion_frames_pairs() {
        let dir = std::env::temp_dir().join(format!("gibbs-wp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = dir.join("vocab.txt");
        std::fs::write(&vocab, "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n##s\n").unwrap();
        let k: Kind = serde_json::from_value(json!({"kind": "wordpiece", "vocab": "vocab.txt", "unk": "[UNK]",
            "cls": "[CLS]", "sep": "[SEP]", "specials": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]}))
        .unwrap();
        let t = k.build(&[vocab.as_path()], &Value::Null).unwrap();
        let e = t.encode(("Hello worlds", "world"), true).unwrap();
        assert_eq!(e.get_ids(), [2, 5, 6, 7, 3, 6, 3]);
        assert_eq!(e.get_type_ids(), [0, 0, 0, 0, 0, 1, 1]);
    }
}
