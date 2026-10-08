//! SetFit repos (`config_setfit.json`): a sentence-transformers body (BERT encoder → pooling per
//! `1_Pooling/config.json` → optional `Normalize`) feeding the scikit-learn `LogisticRegression`
//! head pickled by joblib in `model_head.pkl`. Scores are sklearn `predict_proba`.

use super::{load_tokenizer, read_json, Activation, Body, Checkpoint, SequenceClassifier};
use crate::hardware::Device;
use crate::model::hub::HuggingFaceHub;
use crate::model::jev::{BertDims, BertEncoder};
use serde_json::Value;

const LOGISTIC: &str = "sklearn.linear_model._logistic.LogisticRegression";

pub(super) struct SetFit {
    enc: BertEncoder,
    tok: tokenizers::Tokenizer,
    pool: Pool,
    normalize: bool,
    head: Logistic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pool {
    Mean,
    Cls,
    Max,
}

/// sklearn `LogisticRegression` as `predict_proba` evaluates it.
struct Logistic {
    /// `coef_` `[rows, features]` row-major; rows = 1 for a binary model.
    coef: Vec<f64>,
    intercept: Vec<f64>,
    features: usize,
    /// OvR (sigmoid per class, normalised) instead of multinomial softmax.
    ovr: bool,
}

impl Logistic {
    fn proba(&self, x: &[f32]) -> Vec<f32> {
        let d: Vec<f64> = self
            .intercept
            .iter()
            .enumerate()
            .map(|(r, b)| {
                let w = &self.coef[r * self.features..(r + 1) * self.features];
                b + w.iter().zip(x).map(|(w, x)| w * *x as f64).sum::<f64>()
            })
            .collect();
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let p: Vec<f64> = match (d.len(), self.ovr) {
            // `_predict_proba_lr`: [1 - expit(d), expit(d)].
            (1, true) => vec![1.0 - sig(d[0]), sig(d[0])],
            // Multinomial binary: softmax([-d, d]).
            (1, false) => vec![sig(-2.0 * d[0]), sig(2.0 * d[0])],
            (_, true) => {
                let e: Vec<f64> = d.iter().map(|&v| sig(v)).collect();
                let s: f64 = e.iter().sum();
                if s == 0.0 {
                    vec![1.0 / e.len() as f64; e.len()]
                } else {
                    e.iter().map(|v| v / s).collect()
                }
            }
            (_, false) => {
                let m = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = d.iter().map(|v| (v - m).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|v| v / s).collect()
            }
        };
        p.into_iter().map(|v| v as f32).collect()
    }
}

impl SequenceClassifier {
    /// Loads a SetFit repo (BERT-family body + sklearn LogisticRegression head) on `device`.
    pub fn from_setfit(
        hub: &HuggingFaceHub,
        repo_id: &str,
        device: Device,
    ) -> Result<Self, String> {
        let err = |e: String| format!("{repo_id}: {e}");
        let setfit = read_json(&hub.fetch_file(repo_id, "config_setfit.json")?)?;
        let cfg = read_json(&hub.fetch_file(repo_id, "config.json")?)?;
        let arch = cfg["architectures"][0].as_str().unwrap_or("").to_string();
        if arch != "BertModel" {
            return Err(err(format!(
                "SetFit body '{arch}' is not supported (supported: BertModel)"
            )));
        }

        // modules.json: Transformer, Pooling, optional Normalize; nothing else.
        let modules = read_json(&hub.fetch_file(repo_id, "modules.json")?)?;
        let mut pool_dir = None;
        let mut normalize = setfit["normalize_embeddings"].as_bool().unwrap_or(false);
        for (i, m) in modules
            .as_array()
            .ok_or("modules.json is not a list")?
            .iter()
            .enumerate()
        {
            let ty = m["type"].as_str().unwrap_or("");
            match (i, ty.rsplit('.').next().unwrap_or("")) {
                (0, "Transformer") if m["path"].as_str().unwrap_or("").is_empty() => {}
                (1, "Pooling") => pool_dir = m["path"].as_str().map(str::to_string),
                (2, "Normalize") => normalize = true,
                _ => {
                    return Err(err(format!(
                        "sentence-transformers module {i} '{ty}' is not supported"
                    )))
                }
            }
        }
        let pool_dir = pool_dir.ok_or_else(|| err("modules.json has no Pooling module".into()))?;
        let pc = read_json(&hub.fetch_file(repo_id, &format!("{pool_dir}/config.json"))?)?;
        let pool = match pc["pooling_mode"].as_str() {
            Some("mean") => Pool::Mean,
            Some("cls") => Pool::Cls,
            Some("max") => Pool::Max,
            Some(other) => return Err(err(format!("pooling_mode '{other}' is not supported"))),
            None => {
                let on = |k: &str| pc[k].as_bool() == Some(true);
                let modes = [
                    "pooling_mode_cls_token",
                    "pooling_mode_mean_tokens",
                    "pooling_mode_max_tokens",
                    "pooling_mode_mean_sqrt_len_tokens",
                    "pooling_mode_weightedmean_tokens",
                    "pooling_mode_lasttoken",
                ];
                match modes.map(on) {
                    [true, false, false, false, false, false] => Pool::Cls,
                    [false, true, false, false, false, false] => Pool::Mean,
                    [false, false, true, false, false, false] => Pool::Max,
                    _ => return Err(err(format!("pooling config {pc} is not supported"))),
                }
            }
        };

        let usize_of = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| err(format!("config.json has no {k}")))
        };
        if cfg["hidden_act"].as_str().unwrap_or("gelu") != "gelu"
            || cfg["position_embedding_type"]
                .as_str()
                .unwrap_or("absolute")
                != "absolute"
        {
            return Err(err(
                "only GELU encoders with absolute positions are supported".into(),
            ));
        }
        let dims = BertDims {
            hidden_size: usize_of("hidden_size")?,
            num_layers: usize_of("num_hidden_layers")?,
            num_heads: usize_of("num_attention_heads")?,
            intermediate_size: usize_of("intermediate_size")?,
            max_positions: usize_of("max_position_embeddings")?,
            vocab_size: usize_of("vocab_size")?,
            type_vocab_size: cfg["type_vocab_size"].as_u64().unwrap_or(2) as usize,
            layer_norm_eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32,
        };
        let sbert = hub
            .fetch_file(repo_id, "sentence_bert_config.json")
            .ok()
            .map(|p| read_json(&p))
            .transpose()?
            .unwrap_or(Value::Null);
        let capacity = sbert["max_seq_length"]
            .as_u64()
            .map_or(dims.max_positions, |m| (m as usize).min(dims.max_positions));
        let tok_cfg = hub
            .fetch_file(repo_id, "tokenizer_config.json")
            .ok()
            .map(|p| read_json(&p))
            .transpose()?
            .unwrap_or(Value::Null);
        let tok = load_tokenizer(hub, repo_id, &arch, &tok_cfg, capacity)?;

        let pkl = std::fs::read(hub.fetch_file(repo_id, "model_head.pkl")?)
            .map_err(|e| err(format!("model_head.pkl: {e}")))?;
        let (head, classes) = read_head(&pkl, dims.hidden_size).map_err(err)?;
        let names: Vec<String> = setfit["labels"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let labels = classes
            .into_iter()
            .map(|c| match c {
                Class::Int(i) if !names.is_empty() => names
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| err(format!("class {i} has no entry in config_setfit labels"))),
                Class::Int(i) => Ok(i.to_string()),
                Class::Str(s) => Ok(s),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ckpt = Checkpoint::open(hub, repo_id)?;
        let enc = BertEncoder::load(dims, device, |name, shape| {
            let bare = ckpt.get(name, shape);
            if bare.is_ok() || ckpt.has(name) {
                return bare;
            }
            ckpt.get(&format!("bert.{name}"), shape)
        })?;
        Ok(Self {
            repo_id: repo_id.to_string(),
            architecture: format!("SetFit ({arch} + LogisticRegression)"),
            labels,
            activation: Activation::Identity,
            body: Body::SetFit(SetFit {
                enc,
                tok,
                pool,
                normalize,
                head,
            }),
        })
    }
}

impl SetFit {
    pub(super) fn device(&self) -> Device {
        self.enc.device()
    }

    /// sklearn `predict_proba` of the sentence embedding of `text`, in `classes_` order.
    pub(super) fn proba(&self, text: &str) -> Result<(Vec<f32>, usize), String> {
        let e = super::encode(&self.tok, text, None)?;
        let ids = e.get_ids();
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let hidden = self.enc.run(ids, &positions, e.get_type_ids(), false)?;
        let emb = pool(
            &hidden,
            self.enc.dims.hidden_size,
            self.pool,
            self.normalize,
        );
        Ok((self.head.proba(&emb), ids.len()))
    }
}

/// Pools last hidden states `[n, h]` (no padding: every token is attended) into `[h]`.
fn pool(hidden: &[f32], h: usize, mode: Pool, normalize: bool) -> Vec<f32> {
    let rows = hidden.chunks_exact(h);
    let n = rows.len() as f32;
    let mut v = match mode {
        Pool::Cls => hidden[..h].to_vec(),
        Pool::Mean => rows.fold(vec![0.0; h], |mut a, r| {
            a.iter_mut().zip(r).for_each(|(a, r)| *a += r / n);
            a
        }),
        Pool::Max => rows.fold(vec![f32::NEG_INFINITY; h], |mut a, r| {
            a.iter_mut().zip(r).for_each(|(a, r)| *a = a.max(*r));
            a
        }),
    };
    if normalize {
        // torch F.normalize: x / max(||x||, 1e-12).
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

enum Class {
    Int(i64),
    Str(String),
}

/// The LogisticRegression in a joblib `model_head.pkl` and its `classes_`.
fn read_head(pkl: &[u8], features: usize) -> Result<(Logistic, Vec<Class>), String> {
    let mut p = Pickle::new(pkl);
    let loaded = p.load();
    // The first global a pickle names is the class of the pickled head.
    if let Some(g) = p.globals.first().filter(|g| *g != LOGISTIC) {
        return Err(format!("SetFit head '{g}' is not supported"));
    }
    let (class, state) = match loaded? {
        V::Obj { class, state, .. } => (class, state),
        other => return Err(format!("model_head.pkl holds {other:?}, not a SetFit head")),
    };
    if class != LOGISTIC {
        return Err(format!("SetFit head '{class}' is not supported"));
    }
    let attrs = match state.as_deref() {
        Some(V::Dict(d)) => d,
        _ => return Err("LogisticRegression has no state dict".into()),
    };
    let get = |k: &str| {
        attrs
            .iter()
            .find(|(n, _)| matches!(n, V::Str(s) if s == k))
            .map(|(_, v)| v)
    };
    let arr = |k: &str| match get(k) {
        Some(V::Arr(a)) => Ok(a),
        _ => Err(format!("LogisticRegression has no array {k}")),
    };
    let floats = |a: &Arr| match &a.data {
        Data::F(v) => Ok(v.clone()),
        _ => Err("LogisticRegression coef_/intercept_ are not floats".to_string()),
    };
    let classes: Vec<Class> = match &arr("classes_")?.data {
        Data::I(v) => v.iter().map(|&i| Class::Int(i)).collect(),
        Data::S(v) => v.iter().cloned().map(Class::Str).collect(),
        Data::F(_) => return Err("float classes_ are not supported".into()),
    };
    let (coef_a, intercept) = (arr("coef_")?, floats(arr("intercept_")?)?);
    let coef = floats(coef_a)?;
    let rows = intercept.len();
    let expect_rows = if classes.len() == 2 { 1 } else { classes.len() };
    if coef_a.shape != [rows, features] || rows != expect_rows {
        return Err(format!(
            "coef_ {:?} / intercept_ [{rows}] do not fit {} classes of {features} features",
            coef_a.shape,
            classes.len()
        ));
    }
    let s = |k: &str| match get(k) {
        Some(V::Str(s)) => Some(s.as_str()),
        _ => None,
    };
    // sklearn < 1.8 `predict_proba`: OvR for multi_class "ovr"/"warn", or "auto"/"deprecated"
    // with <= 2 classes or liblinear; else multinomial. sklearn >= 1.8 drops multi_class:
    // binary is OvR (sigmoid), more classes softmax.
    let liblinear = s("solver") == Some("liblinear");
    let ovr = match s("multi_class") {
        Some("ovr" | "warn") => true,
        None | Some("auto" | "deprecated") => classes.len() <= 2 || liblinear,
        Some("multinomial") => false,
        Some(m) => return Err(format!("multi_class '{m}' is not supported")),
    };
    Ok((
        Logistic {
            coef,
            intercept,
            features,
            ovr,
        },
        classes,
    ))
}

// ---- minimal pickle reader (protocols 2-5, the opcodes joblib/sklearn emit) ----

#[derive(Clone, Debug)]
enum V {
    Mark,
    None,
    /// A bool or float: hyperparameters the head does not need.
    Scalar,
    Int(i64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    Obj {
        class: String,
        args: Vec<V>,
        state: Option<Box<V>>,
    },
    Arr(Arr),
}

#[derive(Clone, Debug)]
struct Arr {
    shape: Vec<usize>,
    data: Data,
}

#[derive(Clone, Debug)]
enum Data {
    F(Vec<f64>),
    I(Vec<i64>),
    S(Vec<String>),
}

struct Pickle<'a> {
    b: &'a [u8],
    pos: usize,
    stack: Vec<V>,
    memo: Vec<V>,
    globals: Vec<String>,
}

impl<'a> Pickle<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self {
            b,
            pos: 0,
            stack: Vec::new(),
            globals: Vec::new(),
            memo: Vec::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self
            .b
            .get(self.pos..self.pos + n)
            .ok_or("model_head.pkl is truncated")?;
        self.pos += n;
        Ok(s)
    }

    fn uint(&mut self, n: usize) -> Result<u64, String> {
        let mut v = [0u8; 8];
        v[..n].copy_from_slice(self.take(n)?);
        Ok(u64::from_le_bytes(v))
    }

    fn pop(&mut self) -> Result<V, String> {
        self.stack
            .pop()
            .ok_or_else(|| "pickle stack underflow".into())
    }

    fn pop_mark(&mut self) -> Result<Vec<V>, String> {
        let m = self
            .stack
            .iter()
            .rposition(|v| matches!(v, V::Mark))
            .ok_or("pickle MARK missing")?;
        let items = self.stack.split_off(m + 1);
        self.stack.pop();
        Ok(items)
    }

    fn string(&mut self, n: usize) -> Result<V, String> {
        let s = self.take(n)?;
        Ok(V::Str(
            std::str::from_utf8(s)
                .map_err(|e| format!("pickle string: {e}"))?
                .into(),
        ))
    }

    fn put(&mut self, idx: usize) -> Result<(), String> {
        let top = self
            .stack
            .last()
            .ok_or("pickle PUT on empty stack")?
            .clone();
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, V::None);
        }
        self.memo[idx] = top;
        Ok(())
    }

    fn get(&mut self, idx: usize) -> Result<(), String> {
        let v = self
            .memo
            .get(idx)
            .ok_or("pickle GET of unknown memo")?
            .clone();
        self.stack.push(v);
        Ok(())
    }

    fn load(&mut self) -> Result<V, String> {
        loop {
            let op = self.take(1)?[0];
            match op {
                0x80 => {
                    let p = self.take(1)?[0];
                    if !(2..=5).contains(&p) {
                        return Err(format!("pickle protocol {p} is not supported"));
                    }
                }
                0x95 => {
                    self.take(8)?; // FRAME length: a hint only.
                }
                b'.' => return self.pop(),
                b'(' => self.stack.push(V::Mark),
                b'N' => self.stack.push(V::None),
                0x88 | 0x89 => self.stack.push(V::Scalar),
                b'K' => {
                    let v = self.uint(1)? as i64;
                    self.stack.push(V::Int(v))
                }
                b'M' => {
                    let v = self.uint(2)? as i64;
                    self.stack.push(V::Int(v))
                }
                b'J' => {
                    let v = self.uint(4)? as u32 as i32 as i64;
                    self.stack.push(V::Int(v))
                }
                0x8a => {
                    let n = self.take(1)?[0] as usize;
                    if n > 8 {
                        return Err("pickle LONG1 wider than 64 bits".into());
                    }
                    let b = self.take(n)?;
                    let mut v = [if b.last().is_some_and(|x| x & 0x80 != 0) {
                        0xff
                    } else {
                        0
                    }; 8];
                    v[..n].copy_from_slice(b);
                    self.stack.push(V::Int(i64::from_le_bytes(v)))
                }
                b'G' => {
                    self.take(8)?;
                    self.stack.push(V::Scalar)
                }
                0x8c => {
                    let n = self.uint(1)? as usize;
                    let s = self.string(n)?;
                    self.stack.push(s)
                }
                b'X' => {
                    let n = self.uint(4)? as usize;
                    let s = self.string(n)?;
                    self.stack.push(s)
                }
                0x8d => {
                    let n = self.uint(8)? as usize;
                    let s = self.string(n)?;
                    self.stack.push(s)
                }
                0x94 => {
                    let i = self.memo.len();
                    self.put(i)?
                }
                b'q' => {
                    let i = self.uint(1)? as usize;
                    self.put(i)?
                }
                b'r' => {
                    let i = self.uint(4)? as usize;
                    self.put(i)?
                }
                b'h' => {
                    let i = self.uint(1)? as usize;
                    self.get(i)?
                }
                b'j' => {
                    let i = self.uint(4)? as usize;
                    self.get(i)?
                }
                b')' => self.stack.push(V::Tuple(vec![])),
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    let at = self
                        .stack
                        .len()
                        .checked_sub(n)
                        .ok_or("pickle stack underflow")?;
                    let t = self.stack.split_off(at);
                    self.stack.push(V::Tuple(t))
                }
                b't' => {
                    let t = self.pop_mark()?;
                    self.stack.push(V::Tuple(t))
                }
                b']' => self.stack.push(V::List(vec![])),
                b'}' => self.stack.push(V::Dict(vec![])),
                b'a' | b'e' => {
                    let items = if op == b'a' {
                        vec![self.pop()?]
                    } else {
                        self.pop_mark()?
                    };
                    match self.stack.last_mut() {
                        Some(V::List(l)) => l.extend(items),
                        _ => return Err("pickle APPEND to a non-list".into()),
                    }
                }
                b's' | b'u' => {
                    let items = if op == b's' {
                        let v = self.pop()?;
                        vec![self.pop()?, v]
                    } else {
                        self.pop_mark()?
                    };
                    if items.len() % 2 != 0 {
                        return Err("pickle SETITEMS with odd item count".into());
                    }
                    let mut it = items.into_iter();
                    match self.stack.last_mut() {
                        Some(V::Dict(d)) => {
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                d.push((k, v))
                            }
                        }
                        _ => return Err("pickle SETITEM on a non-dict".into()),
                    }
                }
                0x93 => {
                    let (name, module) = (self.pop()?, self.pop()?);
                    match (module, name) {
                        (V::Str(m), V::Str(n)) => {
                            self.globals.push(format!("{m}.{n}"));
                            self.stack.push(V::Global(format!("{m}.{n}")))
                        }
                        _ => return Err("pickle STACK_GLOBAL needs two strings".into()),
                    }
                }
                0x81 | b'R' => {
                    let args = match self.pop()? {
                        V::Tuple(a) => a,
                        _ => return Err("pickle NEWOBJ/REDUCE args are not a tuple".into()),
                    };
                    let class = match self.pop()? {
                        V::Global(g) => g,
                        _ => return Err("pickle NEWOBJ/REDUCE on a non-class".into()),
                    };
                    if op == b'R' && class != "numpy.dtype" {
                        return Err(format!("pickled callable '{class}' is not supported"));
                    }
                    self.stack.push(V::Obj {
                        class,
                        args,
                        state: None,
                    })
                }
                b'b' => {
                    let st = self.pop()?;
                    let wrapper = match self.stack.last_mut() {
                        Some(V::Obj { class, state, .. }) => {
                            *state = Some(Box::new(st));
                            class == "joblib.numpy_pickle.NumpyArrayWrapper"
                        }
                        _ => return Err("pickle BUILD on a non-object".into()),
                    };
                    if wrapper {
                        let w = self.pop()?;
                        let a = self.joblib_array(&w)?;
                        self.stack.push(V::Arr(a))
                    }
                }
                _ => return Err(format!("pickle opcode 0x{op:02x} is not supported")),
            }
        }
    }

    /// joblib writes a `NumpyArrayWrapper`'s raw bytes right after its BUILD (after a length-
    /// prefixed alignment padding when `numpy_array_alignment_bytes` is set).
    fn joblib_array(&mut self, w: &V) -> Result<Arr, String> {
        let V::Obj {
            state: Some(st), ..
        } = w
        else {
            return Err("NumpyArrayWrapper without state".into());
        };
        let V::Dict(d) = &**st else {
            return Err("NumpyArrayWrapper state is not a dict".into());
        };
        let get = |k: &str| {
            d.iter()
                .find(|(n, _)| matches!(n, V::Str(s) if s == k))
                .map(|(_, v)| v)
        };
        let shape: Vec<usize> = match get("shape") {
            Some(V::Tuple(t)) => t
                .iter()
                .map(|v| match v {
                    V::Int(i) if *i >= 0 => Ok(*i as usize),
                    _ => Err("bad array shape".to_string()),
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("NumpyArrayWrapper has no shape".into()),
        };
        if !matches!(get("subclass"), Some(V::Global(g)) if g == "numpy.ndarray") {
            return Err("only numpy.ndarray arrays are supported".into());
        }
        let fortran = matches!(get("order"), Some(V::Str(o)) if o == "F");
        let (code, endian) = match get("dtype") {
            Some(V::Obj {
                class, args, state, ..
            }) if class == "numpy.dtype" => {
                let code = match args.first() {
                    Some(V::Str(s)) => s.clone(),
                    _ => return Err("numpy.dtype without a type code".into()),
                };
                let endian = match state.as_deref() {
                    Some(V::Tuple(t)) => match t.get(1) {
                        Some(V::Str(e)) => e.clone(),
                        _ => "|".into(),
                    },
                    _ => "|".into(),
                };
                (code, endian)
            }
            _ => return Err("NumpyArrayWrapper has no dtype".into()),
        };
        if endian == ">" {
            return Err(format!("big-endian array dtype {code} is not supported"));
        }
        if get("numpy_array_alignment_bytes").is_some_and(|v| !matches!(v, V::None)) {
            let pad = self.take(1)?[0] as usize;
            self.take(pad)?;
        }
        let count: usize = shape.iter().product();
        let (kind, size) = code.split_at(1);
        let size: usize = size
            .parse()
            .map_err(|_| format!("dtype {code} is not supported"))?;
        let raw = self.take(count * if kind == "U" { size * 4 } else { size })?;
        let mut data = match (kind, size) {
            ("f", 8) => Data::F(
                raw.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            ("f", 4) => Data::F(
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect(),
            ),
            ("i", 8) => Data::I(
                raw.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            ("i", 4) => Data::I(
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            ),
            ("U", _) => Data::S(
                raw.chunks_exact(size * 4)
                    .map(|c| {
                        c.chunks_exact(4)
                            .map(|u| u32::from_le_bytes(u.try_into().unwrap()))
                            .take_while(|&u| u != 0)
                            .map(|u| char::from_u32(u).ok_or("bad UTF-32 in array"))
                            .collect::<Result<String, _>>()
                    })
                    .collect::<Result<_, _>>()?,
            ),
            _ => return Err(format!("array dtype {code} is not supported")),
        };
        if fortran && shape.len() == 2 {
            // Column-major on disk: transpose to row-major.
            let (r, c) = (shape[0], shape[1]);
            if let Data::F(v) = &mut data {
                let t: Vec<f64> = (0..r * c).map(|i| v[(i % c) * r + i / c]).collect();
                *v = t;
            } else {
                return Err("Fortran-ordered non-float arrays are not supported".into());
            }
        } else if fortran && shape.len() > 2 {
            return Err("Fortran-ordered arrays above 2-D are not supported".into());
        }
        Ok(Arr { shape, data })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pool_modes() {
        use super::{pool, Pool};
        let h = [1.0, -2.0, 3.0, 4.0];
        assert_eq!(pool(&h, 2, Pool::Mean, false), vec![2.0, 1.0]);
        assert_eq!(pool(&h, 2, Pool::Cls, false), vec![1.0, -2.0]);
        assert_eq!(pool(&h, 2, Pool::Max, false), vec![3.0, 4.0]);
        assert_eq!(pool(&[3.0, 4.0], 2, Pool::Mean, true), vec![0.6, 0.8]);
    }
}
