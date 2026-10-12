//! fastText supervised models (`.bin` / quantized `.ftz`), read from the binary format of
//! facebookresearch/fastText (fasttext.cc `loadModel`) and scored exactly like
//! `model.predict(text, k=-1)`: `getLine` tokens + `</s>`, word/char n-gram hashing, averaged
//! input rows, then softmax / hierarchical softmax / one-vs-all sigmoid.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

const MAGIC: i32 = 793712314;
const VERSION: i32 = 12;
const EOS: &[u8] = b"</s>";
const LABEL: &[u8] = b"__label__";
/// Bytes `Dictionary::readWord` splits on.
const SPACES: &[u8] = b" \n\r\t\x0b\x0c\0";

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
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `ProductQuantizer`: 256 centroids per sub-quantizer; the last one spans `lastdsub` dims.
struct Pq {
    nsubq: usize,
    dsub: usize,
    lastdsub: usize,
    centroids: Vec<f32>,
}

impl Pq {
    fn load(r: &mut Reader) -> Result<Self, String> {
        let dim = r.i32()? as usize;
        let (nsubq, dsub, lastdsub) = (r.i32()? as usize, r.i32()? as usize, r.i32()? as usize);
        Ok(Self {
            nsubq,
            dsub,
            lastdsub,
            centroids: r.f32s(dim * 256)?,
        })
    }
    fn centroid(&self, m: usize, i: u8) -> &[f32] {
        let i = i as usize;
        if m == self.nsubq - 1 {
            &self.centroids[m * 256 * self.dsub + i * self.lastdsub..][..self.lastdsub]
        } else {
            &self.centroids[(m * 256 + i) * self.dsub..][..self.dsub]
        }
    }
}

enum Matrix {
    /// Row-major `[rows, dim]` f32 at a byte offset into the mapped file.
    Dense { off: usize, rows: usize },
    Quant {
        rows: usize,
        /// `nsubq` codes per row, at a byte offset into the mapped file.
        codes: usize,
        pq: Pq,
        /// `qnorm`: one norm code per row and its 1-d quantizer.
        norm: Option<(Vec<u8>, Pq)>,
    },
}

impl Matrix {
    fn load(r: &mut Reader, quant: bool, dim: usize) -> Result<Self, String> {
        if quant {
            let qnorm = r.u8()? != 0;
            let (rows, n) = (r.i64()? as usize, r.i64()? as usize);
            let codesize = r.i32()? as usize;
            let codes = r.p;
            r.take(codesize)?;
            let pq = Pq::load(r)?;
            let norm = if qnorm {
                let c = r.take(rows)?.to_vec();
                Some((c, Pq::load(r)?))
            } else {
                None
            };
            if n != dim || codesize != rows * pq.nsubq {
                return Err(format!(
                    "fastText: bad quantized matrix {rows}x{n}, {codesize} codes"
                ));
            }
            Ok(Self::Quant {
                rows,
                codes,
                pq,
                norm,
            })
        } else {
            let (rows, n) = (r.i64()? as usize, r.i64()? as usize);
            if n != dim {
                return Err(format!("fastText: matrix width {n} != dim {dim}"));
            }
            let off = r.p;
            r.take(rows * n * 4)?;
            Ok(Self::Dense { off, rows })
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Dense { rows, .. } | Self::Quant { rows, .. } => *rows,
        }
    }

    /// Visits row `i` as (column offset, values, scale) segments.
    fn row(&self, map: &[u8], dim: usize, i: usize, mut f: impl FnMut(usize, &[f32], f32)) {
        match self {
            Self::Dense { off, .. } => f(0, &f32s(&map[off + i * dim * 4..][..dim * 4]), 1.0),
            Self::Quant {
                codes, pq, norm, ..
            } => {
                let scale = norm
                    .as_ref()
                    .map_or(1.0, |(c, npq)| npq.centroid(0, c[i])[0]);
                let code = &map[codes + i * pq.nsubq..][..pq.nsubq];
                for (m, &c) in code.iter().enumerate() {
                    f(m * pq.dsub, pq.centroid(m, c), scale);
                }
            }
        }
    }

    /// `x += row(i)` (`addRowToVector`).
    fn add_row(&self, map: &[u8], x: &mut [f32], i: usize) {
        self.row(map, x.len(), i, |o, v, a| {
            for (x, v) in x[o..].iter_mut().zip(v) {
                *x += a * v;
            }
        });
    }

    /// `row(i) · x` (`dotRow` / `mulcode`).
    fn dot(&self, map: &[u8], x: &[f32], i: usize) -> f32 {
        let mut d = 0f32;
        let mut scale = 1.0;
        self.row(map, x.len(), i, |o, v, a| {
            d += x[o..].iter().zip(v).map(|(x, v)| x * v).sum::<f32>();
            scale = a;
        });
        d * scale
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Loss {
    Hs,
    Ns,
    Softmax,
    Ova,
}

/// Model bytes: the mapped `.bin`/`.ftz`, or the tensors read from a PyTorch re-export.
enum Bytes {
    Map(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Map(m) => m,
            Self::Owned(v) => v,
        }
    }
}

pub struct FastText {
    map: Bytes,
    dim: usize,
    bucket: u32,
    minn: usize,
    maxn: usize,
    word_ngrams: usize,
    loss: Loss,
    /// `args.label` (always the default `__label__` for `.bin`: it is not saved).
    label: Vec<u8>,
    nwords: usize,
    word2int: HashMap<Vec<u8>, i32>,
    words: Vec<Vec<u8>>,
    is_label: Vec<bool>,
    /// `pruneidx` (`None` = not pruned: `pruneidx_size == -1`).
    prune: Option<HashMap<i32, i32>>,
    labels: Vec<String>,
    input: Matrix,
    output: Matrix,
    /// Hierarchical softmax: per label, root-first (output row, code) pairs; code 1 takes
    /// `sigmoid(row · h)`, code 0 its complement.
    paths: Vec<Vec<(usize, bool)>>,
    quant: bool,
}

/// FNV-1a with the int8 sign extension of `Dictionary::hash`.
fn hash(s: &[u8]) -> u32 {
    s.iter().fold(2166136261u32, |h, &c| {
        (h ^ (c as i8 as i32 as u32)).wrapping_mul(16777619)
    })
}

impl FastText {
    pub fn load(path: &Path) -> Result<Self, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: read-only map of a model file nobody rewrites while it is loaded.
        let map =
            Bytes::Map(unsafe { Mmap::map(&f) }.map_err(|e| format!("{}: {e}", path.display()))?);
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
        let loss = match r.i32()? {
            1 => Loss::Hs,
            2 => Loss::Ns,
            3 => Loss::Softmax,
            4 => Loss::Ova,
            l => return Err(format!("fastText: unknown loss {l}")),
        };
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
        let input = Matrix::load(&mut r, quant, dim)?;
        if !quant && prune.is_some() {
            return Err("fastText: pruned dictionary with a dense input matrix".into());
        }
        let qout = r.u8()? != 0;
        let output = Matrix::load(&mut r, quant && qout, dim)?;
        let osz = output.rows();
        // Every loss allocates `nlabels` output rows (hierarchical softmax uses the first n-1).
        if osz != nlabels || (!quant && input.rows() != nwords + bucket as usize) {
            return Err(format!(
                "fastText: matrix shapes (input {}, output {osz}) do not match {nwords} words, {nlabels} labels, {bucket} buckets",
                input.rows()
            ));
        }
        // HierarchicalSoftmaxLoss::buildTree on the label counts, then each leaf's path.
        let mut paths = Vec::new();
        if loss == Loss::Hs {
            let n = nlabels;
            let mut count: Vec<i64> = vec![1_000_000_000_000_000; 2 * n - 1];
            count[..n].copy_from_slice(&counts[nwords..]);
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
            paths = (0..n)
                .map(|mut node| {
                    let mut p = Vec::new();
                    while parent[node] != usize::MAX {
                        p.push((parent[node] - n, binary[node]));
                        node = parent[node];
                    }
                    p.reverse();
                    p
                })
                .collect();
        }
        let labels = words[nwords..]
            .iter()
            .map(|w| String::from_utf8_lossy(w.strip_prefix(LABEL).unwrap_or(w)).into_owned())
            .collect();
        drop(r);
        Ok(Self {
            dim,
            bucket: bucket as u32,
            minn: minn as usize,
            maxn: maxn.max(0) as usize,
            word_ngrams,
            loss,
            label: LABEL.to_vec(),
            nwords,
            word2int,
            words,
            is_label,
            prune,
            labels,
            input,
            output,
            paths,
            quant,
            map,
        })
    }

    /// A hierarchical-softmax model re-exported as `config.json` (args, labels, per-label
    /// leaf-to-root `paths`/`codes`) + `vocab.json` (words in dictionary order) + `weights.pt`
    /// (`input` `[nwords + bucket, dim]`, `output` rows): the script-langid leaf-surgery repos.
    pub fn load_export(dir: &Path) -> Result<Self, String> {
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
        let tensor = |name: &str, rows: usize| -> Result<Matrix, String> {
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
            Ok(Matrix::Dense {
                off: t.byte_offset,
                rows: t.shape[0],
            })
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
        Ok(Self {
            map: Bytes::Owned(data),
            dim,
            bucket: bucket as u32,
            minn,
            maxn,
            word_ngrams,
            loss: Loss::Hs,
            label,
            nwords: words.len(),
            word2int,
            is_label: vec![false; words.len()],
            words,
            prune: None,
            labels,
            input,
            output,
            paths,
            quant: false,
        })
    }

    /// Label names without the `__label__` prefix, in output order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn summary(&self) -> String {
        format!(
            "fastText, {} labels, dim {}, loss {:?}{}",
            self.labels.len(),
            self.dim,
            self.loss,
            if self.quant { ", quantized" } else { "" }
        )
    }

    /// `Dictionary::pushHash`.
    fn push_hash(&self, line: &mut Vec<usize>, id: i32) {
        let id = match &self.prune {
            None => id,
            Some(p) => match p.get(&id) {
                Some(&j) => j,
                None => return,
            },
        };
        if id >= 0 {
            line.push(self.nwords + id as usize);
        }
    }

    /// `Dictionary::computeSubwords` over `<word>` (UTF-8 aware char n-grams).
    fn subwords(&self, token: &[u8], line: &mut Vec<usize>) {
        let w = [b"<", token, b">"].concat();
        for i in 0..w.len() {
            if w[i] & 0xC0 == 0x80 {
                continue;
            }
            let (mut j, mut n) = (i, 1);
            while j < w.len() && n <= self.maxn {
                j += 1;
                while j < w.len() && w[j] & 0xC0 == 0x80 {
                    j += 1;
                }
                if n >= self.minn && !(n == 1 && (i == 0 || j == w.len())) {
                    self.push_hash(line, (hash(&w[i..j]) % self.bucket) as i32);
                }
                n += 1;
            }
        }
    }

    /// Probability of every label (in `labels()` order) and the number of tokens read, as
    /// `fasttext.predict(text, k=-1)` reports them: `p + 1e-5` (its `exp(log(p + 1e-5))`); labels
    /// below its implicit threshold (hierarchical-softmax paths under 1e-5) are 0.
    pub fn predict(&self, text: &str) -> Result<(Vec<f32>, usize), String> {
        crate::hardware::cpu_compute::enter("fasttext::predict");
        if text.contains('\n') {
            return Err("fastText predicts one line at a time (remove '\\n')".into());
        }
        // getLine: whitespace tokens, then the `</s>` the trailing newline produces.
        let tokens = text
            .as_bytes()
            .split(|c| SPACES.contains(c))
            .filter(|t| !t.is_empty())
            .chain([EOS]);
        let (mut line, mut hashes, mut ntokens) = (Vec::new(), Vec::new(), 0);
        for t in tokens {
            ntokens += 1;
            let h = hash(t);
            let wid = self.word2int.get(t).copied();
            let label = wid.map_or(t.starts_with(&self.label), |w| self.is_label[w as usize]);
            if label {
                continue;
            }
            match wid {
                Some(w) => {
                    line.push(w as usize);
                    if self.maxn > 0 && t != EOS {
                        self.subwords(&self.words[w as usize], &mut line);
                    }
                }
                None if t != EOS => self.subwords(t, &mut line),
                None => {}
            }
            hashes.push(h as i32);
        }
        // addWordNgrams: int32 hashes widen (sign-extended) to uint64.
        for i in 0..hashes.len() {
            let mut h = hashes[i] as i64 as u64;
            for &hj in hashes.iter().take(i + self.word_ngrams).skip(i + 1) {
                h = h.wrapping_mul(116049371).wrapping_add(hj as i64 as u64);
                self.push_hash(&mut line, (h % self.bucket as u64) as i32);
            }
        }
        if line.is_empty() {
            return Err("fastText: the input has no known word or n-gram".into());
        }
        let mut hidden = vec![0f32; self.dim];
        for &i in &line {
            self.input.add_row(&self.map, &mut hidden, i);
        }
        let inv = (1.0 / line.len() as f64) as f32;
        hidden.iter_mut().for_each(|v| *v *= inv);

        let probs = match self.loss {
            Loss::Hs => {
                // HierarchicalSoftmaxLoss::dfs from the root with threshold 0: a branch whose
                // log score falls under log(1e-5) is pruned (label absent → 0).
                let floor = 1e-5f32.ln();
                let mut sig = vec![None; self.output.rows()];
                self.paths
                    .iter()
                    .map(|path| {
                        let mut score = 0f32;
                        for &(row, code) in path {
                            if score < floor {
                                return 0.0;
                            }
                            let f = *sig[row].get_or_insert_with(|| {
                                1.0 / (1.0 + (-self.output.dot(&self.map, &hidden, row)).exp())
                            });
                            score += (if code { f } else { 1.0 - f } + 1e-5).ln();
                        }
                        if score < floor {
                            0.0
                        } else {
                            score.exp()
                        }
                    })
                    .collect()
            }
            loss => {
                let mut out: Vec<f32> = (0..self.labels.len())
                    .map(|i| self.output.dot(&self.map, &hidden, i))
                    .collect();
                if loss == Loss::Softmax {
                    let m = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    out.iter_mut().for_each(|v| *v = (*v - m).exp());
                    let z: f32 = out.iter().sum();
                    out.iter_mut().for_each(|v| *v /= z);
                } else {
                    out.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()));
                }
                out.into_iter().map(|p| (p + 1e-5).ln().exp()).collect()
            }
        };
        Ok((probs, ntokens))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fnv_sign_extension() {
        // ASCII is plain FNV-1a; bytes >= 0x80 xor in a sign-extended int8.
        assert_eq!(super::hash(b"a"), 0xe40c292c);
        assert_eq!(
            super::hash(&[0xc3]),
            (2166136261u32 ^ 0xffff_ffc3).wrapping_mul(16777619)
        );
    }
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
