//! SentencePiece `.model` files (ALBERT `spiece.model`, CamemBERT `sentencepiece.bpe.model`)
//! rebuilt as the fast tokenizers transformers converts them to, for repos without
//! tokenizer.json. The `ModelProto` is read by [`crate::model::spm`]; only Unigram models are
//! built.

use crate::model::spm::{SpmModel, CONTROL, UNKNOWN};
use serde_json::Value;
use std::path::Path;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::{
    Lowercase, NormalizerWrapper, Precompiled, Replace, Sequence, StripAccents, NFKD,
};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::AddedToken;

/// The fast tokenizer transformers builds from the SentencePiece model at `path` for `arch`
/// (`AlbertTokenizer` / `CamembertTokenizer`), without truncation.
pub(super) fn tokenizer(
    path: &Path,
    arch: &str,
    tok_cfg: &Value,
) -> Result<tokenizers::Tokenizer, String> {
    let m = SpmModel::open(path)?;
    if m.bpe {
        return Err(format!(
            "{}: SentencePiece BPE model is not Unigram; only Unigram models are rebuilt",
            path.display()
        ));
    }
    let err = |e: tokenizers::Error| e.to_string();
    let camembert = arch.starts_with("Camembert");
    let mut norm: Vec<NormalizerWrapper> = Vec::new();
    let mut specials: Vec<AddedToken> = Vec::new();
    let (vocab, unk, cls, sep) = if camembert {
        // CamembertConverter: fairseq's four leading ids, spm pieces after its <unk>, <mask> last.
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
        v.extend(
            m.pieces[1..]
                .iter()
                .map(|p| (p.text.clone(), p.score as f64)),
        );
        v.push(("<mask>".into(), 0.0));
        specials.extend(
            ["<s>NOTUSED", "<pad>", "</s>NOTUSED", "<unk>"]
                .iter()
                .map(|s| AddedToken::from(*s, true)),
        );
        specials.push(AddedToken::from("<mask>", true).lstrip(true));
        (v, 3, "<s>", "</s>")
    } else {
        // AlbertConverter: pieces ending in "<digit>," are demoted by 100.
        let v = m
            .pieces
            .iter()
            .map(|p| {
                let c: Vec<char> = p.text.chars().collect();
                let number_comma =
                    c.len() >= 2 && c[c.len() - 1] == ',' && c[c.len() - 2].is_numeric();
                (
                    p.text.clone(),
                    p.score as f64 - if number_comma { 100.0 } else { 0.0 },
                )
            })
            .collect();
        norm.push(Replace::new("``", "\"").map_err(err)?.into());
        norm.push(Replace::new("''", "\"").map_err(err)?.into());
        if !tok_cfg["keep_accents"].as_bool().unwrap_or(false) {
            norm.push(NFKD.into());
            norm.push(StripAccents.into());
        }
        if tok_cfg["do_lower_case"].as_bool().unwrap_or(true) {
            norm.push(Lowercase.into());
        }
        let unk = m
            .pieces
            .iter()
            .position(|p| p.kind == UNKNOWN)
            .ok_or_else(|| format!("{}: no unknown piece", path.display()))?;
        (v, unk, "[CLS]", "[SEP]")
    };
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
    let mut t = tokenizers::Tokenizer::new(Unigram::from(vocab, Some(unk), false).map_err(err)?);
    t.with_normalizer(Some(Sequence::new(norm)));
    t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::sequence::Sequence::new(
        vec![
            WhitespaceSplit.into(),
            Metaspace::new('▁', PrependScheme::Always, true).into(),
        ],
    )));
    t.add_special_tokens(&specials);
    let id = |s: &str| {
        t.token_to_id(s)
            .ok_or_else(|| format!("{}: no {s} piece", path.display()))
    };
    let (cls_id, sep_id) = (id(cls)?, id(sep)?);
    let pair = if camembert {
        format!("{cls} $A {sep} {sep} $B {sep}")
    } else {
        format!("{cls} $A {sep} $B:1 {sep}:1")
    };
    let post = TemplateProcessing::builder()
        .try_single(format!("{cls} $A {sep}"))
        .and_then(|b| b.try_pair(pair))
        .map_err(|e| e.to_string())?
        .special_tokens(vec![(cls, cls_id), (sep, sep_id)])
        .build()
        .map_err(|e| e.to_string())?;
    t.with_post_processor(Some(post));
    Ok(t)
}

#[cfg(test)]
mod tests {
    /// The rebuilt tokenizers match the tokenizer.json transformers converted from the same
    /// SentencePiece files (albert-base-v2 shares tals/albert-xlarge-vitaminc-mnli's spiece.model,
    /// camembert-base cmarkea/distilcamembert-base-sentiment's sentencepiece.bpe.model).
    #[test]
    fn rebuilt_spm_matches_converted_tokenizer_json() {
        let snap = |r: &str| format!("disk/models--{}/snapshots/main", r.replace('/', "--"));
        let texts = [
            "J'aime me promener en forêt même si ça me donne mal aux pieds.",
            "``Héllo''  WORLD, in 2010, 1,000 cafés naïve — ok?",
            "  The Eiffel Tower is located in Paris.   ",
        ];
        for (arch, model, reference) in [
            (
                "AlbertForSequenceClassification",
                "tals/albert-xlarge-vitaminc-mnli/spiece.model",
                "albert/albert-base-v2",
            ),
            (
                "CamembertForSequenceClassification",
                "cmarkea/distilcamembert-base-sentiment/sentencepiece.bpe.model",
                "almanach/camembert-base",
            ),
        ] {
            let (repo, file) = model.rsplit_once('/').unwrap();
            let model = std::path::PathBuf::from(snap(repo)).join(file);
            let reference = std::path::PathBuf::from(snap(reference)).join("tokenizer.json");
            if !model.exists() || !reference.exists() {
                eprintln!(
                    "SKIP: {} or {} not cached",
                    model.display(),
                    reference.display()
                );
                continue;
            }
            let ours = super::tokenizer(&model, arch, &serde_json::Value::Null).unwrap();
            let theirs = tokenizers::Tokenizer::from_file(&reference).unwrap();
            for t in texts {
                let (a, b) = (
                    ours.encode(t, true).unwrap(),
                    theirs.encode(t, true).unwrap(),
                );
                assert_eq!(
                    a.get_ids(),
                    b.get_ids(),
                    "{arch}: {t:?} {:?}",
                    a.get_tokens()
                );
                let (a, b) = (
                    ours.encode((t, "Second segment."), true).unwrap(),
                    theirs.encode((t, "Second segment."), true).unwrap(),
                );
                assert_eq!(a.get_ids(), b.get_ids());
                assert_eq!(a.get_type_ids(), b.get_type_ids());
            }
        }
    }
}
