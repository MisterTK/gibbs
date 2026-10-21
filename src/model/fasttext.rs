//! fastText supervised models (`.bin` / quantized `.ftz`), read from the binary format of
//! facebookresearch/fastText (fasttext.cc `loadModel`) into a [`BagModel`]; scoring (hashing,
//! averaged input rows, softmax / hierarchical softmax / one-vs-all sigmoid) is
//! [`crate::runtime::bag`] over the `fasttext` family spec.

use std::collections::HashMap;
use std::path::Path;

use crate::runtime::bag::{BagModel, Head, Pq, QuantRows, Rows, Vocab};
use memmap2::Mmap;

const MAGIC: i32 = 793712314;
const VERSION: i32 = 12;
const LABEL: &[u8] = b"__label__";

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self
            .b
            .get(self.p..self.p + n)
            .ok_or("truncated fastText model")?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        Ok(f32s(self.take(n * 4)?))
    }

    /// `ProductQuantizer::load`.
    fn pq(&mut self) -> Result<Pq, String> {
        let dim = self.i32()? as usize;
        let (nsubq, dsub, lastdsub) = (
            self.i32()? as usize,
            self.i32()? as usize,
            self.i32()? as usize,
        );
        Ok(Pq {
            nsubq,
            dsub,
            lastdsub,
            centroids: self.f32s(dim * 256)?,
        })
    }

    /// A `DenseMatrix` / `QuantMatrix` of width `dim`.
    fn matrix(&mut self, quant: bool, dim: usize) -> Result<Rows, String> {
        if quant {
            let qnorm = self.u8()? != 0;
            let (rows, n) = (self.i64()? as usize, self.i64()? as usize);
            let codesize = self.i32()? as usize;
            let codes = self.take(codesize)?.to_vec();
            let pq = self.pq()?;
            let norm = if qnorm {
                let c = self.take(rows)?.to_vec();
                Some((c, self.pq()?))
            } else {
                None
            };
            if n != dim || codesize != rows * pq.nsubq {
                return Err(format!(
                    "fastText: bad quantized matrix {rows}x{n}, {codesize} codes"
                ));
            }
            Ok(Rows::Quant(QuantRows { codes, pq, norm }))
        } else {
            let (rows, n) = (self.i64()? as usize, self.i64()? as usize);
            if n != dim {
                return Err(format!("fastText: matrix width {n} != dim {dim}"));
            }
            Ok(Rows::Dense(self.f32s(rows * n)?))
        }
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn rows(m: &Rows, dim: usize) -> usize {
    match m {
        Rows::Dense(v) => v.len() / dim,
        Rows::Quant(q) => q.rows(),
    }
}

/// Reads a `.bin`/`.ftz` supervised model.
pub fn load(path: &Path) -> Result<BagModel, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: read-only map of a model file nobody rewrites while it is loaded.
    let map = unsafe { Mmap::map(&f) }.map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = Reader { b: &map, p: 0 };
    if r.i32()? != MAGIC {
        return Err(format!("{} is not a fastText model", path.display()));
    }
    let version = r.i32()?;
    if version > VERSION {
        return Err(format!("fastText model version {version} > {VERSION}"));
    }
    // Args::load
    let dim = r.i32()? as usize;
    let (_ws, _epoch, _min_count, _neg) = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
    let word_ngrams = r.i32()?.max(1) as usize;
    // 1 hs, 2 ns, 3 softmax, 4 ova.
    let loss = r.i32()?;
    if !(1..=4).contains(&loss) {
        return Err(format!("fastText: unknown loss {loss}"));
    }
    let model = r.i32()?;
    let bucket = r.i32()?;
    let (minn, mut maxn) = (r.i32()?, r.i32()?);
    let (_lr_update, _t) = (r.i32()?, r.i64()?);
    if model != 3 {
        let name = match model {
            1 => "cbow",
            2 => "skipgram",
            _ => "unknown",
        };
        return Err(format!(
            "fastText model is not a classifier (model: {name})"
        ));
    }
    if version == 11 {
        maxn = 0; // old supervised models do not use char n-grams
    }
    // Dictionary::load
    let size = r.i32()? as usize;
    let nwords = r.i32()? as usize;
    let nlabels = r.i32()? as usize;
    let _ntokens = r.i64()?;
    let prune_size = r.i64()?;
    let mut words = Vec::with_capacity(size);
    let mut counts = Vec::with_capacity(size);
    let mut is_label = Vec::with_capacity(size);
    let mut word2int = HashMap::with_capacity(size);
    for i in 0..size {
        let len = r.b[r.p..]
            .iter()
            .position(|&c| c == 0)
            .ok_or("truncated fastText dictionary")?;
        let w = r.take(len + 1)?[..len].to_vec();
        counts.push(r.i64()?);
        is_label.push(r.u8()? == 1);
        word2int.entry(w.clone()).or_insert(i as i32);
        words.push(w);
    }
    if nwords + nlabels != size || nlabels == 0 {
        return Err(format!(
            "fastText dictionary: {nwords} words + {nlabels} labels != {size}"
        ));
    }
    let prune = if prune_size < 0 {
        None
    } else {
        let mut m = HashMap::with_capacity(prune_size as usize);
        for _ in 0..prune_size {
            let (a, b) = (r.i32()?, r.i32()?);
            m.insert(a, b);
        }
        Some(m)
    };
    let quant = r.u8()? != 0;
    let input = r.matrix(quant, dim)?;
    if !quant && prune.is_some() {
        return Err("fastText: pruned dictionary with a dense input matrix".into());
    }
    let qout = r.u8()? != 0;
    let output = r.matrix(quant && qout, dim)?;
    let osz = rows(&output, dim);
    // Every loss allocates `nlabels` output rows (hierarchical softmax uses the first n-1).
    if osz != nlabels || (!quant && rows(&input, dim) != nwords + bucket as usize) {
        return Err(format!(
            "fastText: matrix shapes (input {}, output {osz}) do not match {nwords} words, {nlabels} labels, {bucket} buckets",
            rows(&input, dim)
        ));
    }
    let output = match output {
        Rows::Dense(v) => v,
        Rows::Quant(q) => {
            let mut v = vec![0f32; osz * dim];
            for (i, row) in v.chunks_exact_mut(dim).enumerate() {
                q.decode(i, row);
            }
            v
        }
    };
    let head = match loss {
        1 => Head::Tree(tree_paths(&counts[nwords..])),
        3 => Head::Softmax,
        _ => Head::Sigmoid,
    };
    let labels = words[nwords..]
        .iter()
        .map(|w| String::from_utf8_lossy(w.strip_prefix(LABEL).unwrap_or(w)).into_owned())
        .collect();
    Ok(BagModel {
        vocab: Vocab {
            nwords,
            bucket: bucket as u32,
            minn: minn as usize,
            maxn: maxn.max(0) as usize,
            word_ngrams,
            label: LABEL.to_vec(),
            word2int,
            words,
            is_label,
            prune,
        },
        dim,
        input,
        output,
        head,
        labels,
    })
}

/// `HierarchicalSoftmaxLoss::buildTree` on the label counts, then each leaf's root-first path.
fn tree_paths(counts: &[i64]) -> Vec<Vec<(usize, bool)>> {
    let n = counts.len();
    let mut count: Vec<i64> = vec![1_000_000_000_000_000; 2 * n - 1];
    count[..n].copy_from_slice(counts);
    let mut parent = vec![usize::MAX; 2 * n - 1];
    let mut binary = vec![false; 2 * n - 1];
    let (mut leaf, mut node) = (n as i64 - 1, n);
    for i in n..2 * n - 1 {
        let mut mini = [0usize; 2];
        for m in &mut mini {
            if leaf >= 0 && count[leaf as usize] < count[node] {
                *m = leaf as usize;
                leaf -= 1;
            } else {
                *m = node;
                node += 1;
            }
        }
        count[i] = count[mini[0]] + count[mini[1]];
        parent[mini[0]] = i;
        parent[mini[1]] = i;
        binary[mini[1]] = true;
    }
    (0..n)
        .map(|mut node| {
            let mut p = Vec::new();
            while parent[node] != usize::MAX {
                p.push((parent[node] - n, binary[node]));
                node = parent[node];
            }
            p.reverse();
            p
        })
        .collect()
}

/// A hierarchical-softmax model re-exported as `config.json` (args, labels, per-label
/// leaf-to-root `paths`/`codes`) + `vocab.json` (words in dictionary order) + `weights.pt`
/// (`input` `[nwords + bucket, dim]`, `output` rows): the script-langid leaf-surgery repos.
pub fn load_export(dir: &Path) -> Result<BagModel, String> {
    let json = |f: &str| -> Result<serde_json::Value, String> {
        let p = dir.join(f);
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))
    };
    let cfg = json("config.json")?;
    if cfg["architecture"] != "fasttext_hierarchical_softmax" {
        return Err(format!(
            "unsupported fastText export architecture {}",
            cfg["architecture"]
        ));
    }
    let a = &cfg["args"];
    let int = |k: &str| {
        a[k].as_u64()
            .map(|v| v as usize)
            .ok_or(format!("config.json: args.{k} missing"))
    };
    let (dim, minn, maxn, bucket, word_ngrams) = (
        int("dim")?,
        int("minn")?,
        int("maxn")?,
        int("bucket")?,
        int("wordNgrams")?.max(1),
    );
    let strs = |v: &serde_json::Value, what: &str| -> Result<Vec<String>, String> {
        v.as_array()
            .ok_or(format!("{what} missing"))?
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_string)
                    .ok_or(format!("{what}: not a string"))
            })
            .collect()
    };
    let labels = strs(&cfg["labels"], "config.json labels")?;
    let words: Vec<Vec<u8>> = strs(&json("vocab.json")?, "vocab.json")?
        .into_iter()
        .map(String::into_bytes)
        .collect();
    let ints = |v: &serde_json::Value| -> Option<Vec<u64>> {
        v.as_array()?.iter().map(|x| x.as_u64()).collect()
    };
    let (pa, co) = (cfg["paths"].as_array(), cfg["codes"].as_array());
    let paths: Vec<Vec<(usize, bool)>> = match (pa, co) {
        (Some(pa), Some(co)) if pa.len() == labels.len() && co.len() == labels.len() => pa
            .iter()
            .zip(co)
            .map(|(p, c)| {
                let (p, c) = (ints(p)?, ints(c)?);
                // Stored leaf to root (fastText `paths_`); scored root first.
                (p.len() == c.len()).then(|| {
                    p.iter()
                        .zip(&c)
                        .rev()
                        .map(|(&r, &c)| (r as usize, c == 1))
                        .collect()
                })
            })
            .collect::<Option<_>>()
            .ok_or("config.json: malformed paths/codes")?,
        _ => return Err("config.json: paths/codes must list one path per label".into()),
    };
    let (manifest, data) = crate::model::loaders::read_checkpoint(&dir.join("weights.pt"))?;
    let tensor = |name: &str, rows: usize| -> Result<Vec<f32>, String> {
        let t = manifest
            .tensors
            .get(name)
            .ok_or(format!("weights.pt: no `{name}` tensor"))?;
        if t.dtype != "F32" || t.shape.len() != 2 || t.shape[1] != dim || t.shape[0] < rows {
            return Err(format!(
                "weights.pt: `{name}` is {} {:?}, expected F32 [{rows}, {dim}]",
                t.dtype, t.shape
            ));
        }
        Ok(f32s(&data[t.byte_offset..][..t.shape[0] * dim * 4]))
    };
    let out_rows = paths
        .iter()
        .flatten()
        .map(|&(r, _)| r + 1)
        .max()
        .unwrap_or(0);
    let input = tensor("input", words.len() + bucket)?;
    let output = tensor("output", out_rows)?;
    let label = a["label"]
        .as_str()
        .unwrap_or("__label__")
        .as_bytes()
        .to_vec();
    let word2int = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i as i32))
        .collect();
    Ok(BagModel {
        vocab: Vocab {
            nwords: words.len(),
            bucket: bucket as u32,
            minn,
            maxn,
            word_ngrams,
            label,
            word2int,
            is_label: vec![false; words.len()],
            words,
            prune: None,
        },
        dim,
        input: Rows::Dense(input),
        output,
        head: Head::Tree(paths),
        labels,
    })
}

/// The fastText model file of `repo_id` (a repo without config.json): a cached `.bin`/`.ftz`
/// with the fastText magic, else, for a `library_name: fasttext` Hub repo, its `model.bin` /
/// `model.ftz` or its only root `.bin`/`.ftz`.
pub fn hub_file(hub: &crate::model::hub::HuggingFaceHub, repo_id: &str) -> Option<String> {
    let is_model = |n: &str| n.ends_with(".bin") || n.ends_with(".ftz");
    if let Some(dir) = hub.locate_model(repo_id).local_dir {
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let mut magic = [0u8; 4];
            let ok = is_model(&name)
                && std::fs::File::open(e.path())
                    .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
                    .is_ok();
            if ok && i32::from_le_bytes(magic) == MAGIC {
                return Some(name);
            }
        }
    }
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-fsSL", "-m", "15"]);
    if let Ok(t) = std::env::var("HF_TOKEN") {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let out = cmd
        .arg(format!(
            "https://huggingface.co/api/models/{repo_id}?expand%5B%5D=library_name&expand%5B%5D=siblings"
        ))
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    if v["library_name"] != "fasttext" {
        return None;
    }
    let files: Vec<&str> = v["siblings"]
        .as_array()?
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .filter(|f| !f.contains('/') && is_model(f))
        .collect();
    ["model.bin", "model.ftz"]
        .into_iter()
        .find(|f| files.contains(f))
        .or_else(|| (files.len() == 1).then(|| files[0]))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use crate::hardware::{cpu_compute, Device};
    use crate::runtime::bag::BagClassifier;

    /// A version-12 supervised `.bin`: 5 words, 4 labels (counts 50/30/15/5 shape the tree), word
    /// bigrams, char 2-3-grams, 64 buckets, dim 8; `quant` writes product-quantized input (with
    /// norms) and output matrices (3 sub-quantizers of 3+3+2 dims).
    fn write_bin(path: &std::path::Path, loss: i32, quant: bool) {
        let (dim, bucket) = (8usize, 64i32);
        let words = ["</s>", "the", "cat", "sat", "dog"];
        let labels = ["__label__a", "__label__b", "__label__c", "__label__d"];
        let counts = [20i64, 19, 18, 17, 16, 50, 30, 15, 5];
        let mut b = Vec::new();
        let i32s = |b: &mut Vec<u8>, v: &[i32]| v.iter().for_each(|x| b.extend(x.to_le_bytes()));
        i32s(&mut b, &[super::MAGIC, 12]);
        // dim ws epoch minCount neg wordNgrams loss model(sup) bucket minn maxn lrUpdateRate
        i32s(&mut b, &[dim as i32, 5, 5, 1, 5, 2, loss, 3, bucket, 2, 3, 100]);
        b.extend(1e-4f64.to_le_bytes());
        let size = (words.len() + labels.len()) as i32;
        i32s(&mut b, &[size, words.len() as i32, labels.len() as i32]);
        b.extend(100i64.to_le_bytes()); // ntokens
        b.extend((-1i64).to_le_bytes()); // pruneidx_size: not pruned
        for (i, w) in words.iter().chain(&labels).enumerate() {
            b.extend_from_slice(w.as_bytes());
            b.push(0);
            b.extend(counts[i].to_le_bytes());
            b.push((i >= words.len()) as u8);
        }
        let mut rng = crate::arch::testing::Rng(0xEE ^ loss as u64);
        let mut f32s = |b: &mut Vec<u8>, n: usize, scale: f32| {
            (0..n).for_each(|_| b.extend((rng.unit() * scale).to_le_bytes()))
        };
        let mut bytes = crate::arch::testing::Rng(0xC0DE);
        let mut matrix = |b: &mut Vec<u8>, rows: usize, scale: f32, qnorm: bool| {
            b.push(quant as u8);
            if !quant {
                b.extend((rows as i64).to_le_bytes());
                b.extend((dim as i64).to_le_bytes());
                f32s(b, rows * dim, scale);
                return;
            }
            b.push(qnorm as u8);
            b.extend((rows as i64).to_le_bytes());
            b.extend((dim as i64).to_le_bytes());
            i32s(b, &[rows as i32 * 3]);
            (0..rows * 3).for_each(|_| b.push(bytes.next_u64() as u8));
            i32s(b, &[dim as i32, 3, 3, 2]);
            f32s(b, dim * 256, scale);
            if qnorm {
                (0..rows).for_each(|_| b.push(bytes.next_u64() as u8));
                i32s(b, &[1, 1, 1, 1]);
                f32s(b, 256, 1.0);
            }
        };
        matrix(&mut b, words.len() + bucket as usize, 1.0, true);
        matrix(&mut b, labels.len(), 2.0, false);
        std::fs::write(path, b).unwrap();
    }

    /// (loss, quantized) cases: hierarchical softmax dense / quantized, one-vs-all quantized,
    /// negative sampling dense.
    const CASES: [(i32, bool); 4] = [(1, false), (1, true), (4, true), (2, false)];
    const TEXTS: [&str; 2] = ["the cat sat", "a dog sat on the mat"];

    /// `FastText::predict` of the pre-spec host implementation (commit 251be4c) on [`write_bin`]
    /// fixtures, per case then text: (token count, probabilities).
    const OLD: [[(usize, [f32; 4]); 2]; 4] = [
        [(4, [0.42800388, 0.16871147, 0.18828341, 0.2150408]), (7, [0.40358174, 0.20235285, 0.20580043, 0.18830487])],
        [(4, [0.5223296, 0.26141247, 0.08643455, 0.12985727]), (7, [0.48707005, 0.25290027, 0.1278607, 0.1322044])],
        [(4, [0.46248654, 0.4656299, 0.52239984, 0.48894987]), (7, [0.46962076, 0.49344978, 0.46248892, 0.49561393])],
        [(4, [0.6089103, 0.44071913, 0.3434737, 0.61539763]), (7, [0.4875182, 0.5094922, 0.41907105, 0.69172555])],
    ];

    /// Hierarchical softmax (dense, quantized), one-vs-all (quantized input and output) and
    /// negative sampling read from `.bin` files and run on the bag runtime match the old host
    /// scores on every device; the Metal leg enters no CPU compute.
    #[test]
    fn bag_runtime_matches_old_host_scores() {
        let dir = std::env::temp_dir().join(format!("gibbs_ft_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut devs = vec![Device::Cpu];
        if crate::hardware::available_devices().contains(&Device::Metal) {
            devs.push(Device::Metal);
        }
        for dev in devs {
            for ((loss, quant), old) in CASES.into_iter().zip(OLD) {
                let p = dir.join(format!("m{loss}{quant}.bin"));
                write_bin(&p, loss, quant);
                let c = BagClassifier::new("fasttext", super::load(&p).unwrap(), dev).unwrap();
                assert_eq!(c.labels(), ["a", "b", "c", "d"]);
                for (t, (n_old, p_old)) in TEXTS.into_iter().zip(old) {
                    let entered = cpu_compute::entered();
                    let (p, n) = c.predict(t).unwrap();
                    if dev == Device::Metal {
                        assert_eq!(cpu_compute::entered(), entered, "Metal leg ran CPU compute");
                    }
                    assert_eq!(n, n_old);
                    for (g, w) in p.iter().zip(p_old) {
                        assert!(
                            (g - w).abs() <= 1e-4 * w.abs() + 1e-6,
                            "{dev:?} loss {loss} quant {quant} {t:?}: {p:?} vs {p_old:?}"
                        );
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
