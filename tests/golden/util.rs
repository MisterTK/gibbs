//! Shared golden harness: deterministic tiny on-disk checkpoints and output comparison.
//!
//! Every golden case writes its fixture as a real checkpoint directory (config.json, safetensors,
//! tokenizer.json, ...) from a fixed seed, loads it through a public loader entry point and stores
//! the outputs in `tests/golden/<case>.json`. The fixture survives any rewrite of the model code,
//! so the JSON stays the oracle for the replacement implementation.
//!
//! `GIBBS_GOLDEN_WRITE=1` (re)writes the JSON from the CPU run; otherwise every device in
//! [`devices`] is compared against it.

#![allow(dead_code)]

use gibbs_engine::hardware::Device;
use gibbs_engine::model::hub::ModelArtifacts;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// SplitMix64: same sequence on every platform and build.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// `n` values uniform in [-scale, scale).
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
    /// `n` values uniform in [center - spread, center + spread) (norm weights, scales).
    pub fn around(&mut self, n: usize, center: f32, spread: f32) -> Vec<f32> {
        (0..n).map(|_| center + self.unit() * spread).collect()
    }
}

/// One tensor to write: name, shape, row-major f32 values.
pub type T = (String, Vec<usize>, Vec<f32>);

/// Builds tensors with deterministic random values; `w(name, shape)` scales by 1/sqrt(fan_in).
pub struct Weights {
    pub rng: Rng,
    pub tensors: Vec<T>,
}

impl Weights {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            tensors: Vec::new(),
        }
    }
    /// Random matrix/vector, scale 1/sqrt(last dim) (0.5 for rank 1).
    pub fn w(&mut self, name: impl Into<String>, shape: &[usize]) -> &mut Self {
        let n: usize = shape.iter().product();
        let scale = if shape.len() >= 2 {
            1.0 / (*shape.last().unwrap() as f32).sqrt()
        } else {
            0.5
        };
        let v = self.rng.vec(n, scale);
        self.tensors.push((name.into(), shape.to_vec(), v));
        self
    }
    /// Random values in [-scale, scale).
    pub fn s(&mut self, name: impl Into<String>, shape: &[usize], scale: f32) -> &mut Self {
        let v = self.rng.vec(shape.iter().product(), scale);
        self.tensors.push((name.into(), shape.to_vec(), v));
        self
    }
    /// Norm-style weight: center ± 0.1.
    pub fn norm(&mut self, name: impl Into<String>, n: usize, center: f32) -> &mut Self {
        let v = self.rng.around(n, center, 0.1);
        self.tensors.push((name.into(), vec![n], v));
        self
    }
    /// Explicit values.
    pub fn put(&mut self, name: impl Into<String>, shape: &[usize], v: Vec<f32>) -> &mut Self {
        assert_eq!(v.len(), shape.iter().product::<usize>());
        self.tensors.push((name.into(), shape.to_vec(), v));
        self
    }
    pub fn write(&self, path: &Path) {
        write_st(path, &self.tensors);
    }
}

/// Safetensors dtype tag for [`write_st_as`].
#[derive(Clone, Copy)]
pub enum StDtype {
    F32,
    F16,
    BF16,
}

/// Writes `tensors` as an F32 safetensors file.
pub fn write_st(path: &Path, tensors: &[T]) {
    write_st_as(path, tensors, StDtype::F32)
}

/// Writes `tensors` as a safetensors file in `dtype` (round to nearest even).
pub fn write_st_as(path: &Path, tensors: &[T], dtype: StDtype) {
    let (mut header, mut data) = (Map::new(), Vec::<u8>::new());
    for (name, shape, v) in tensors {
        let start = data.len();
        for &x in v {
            match dtype {
                StDtype::F32 => data.extend_from_slice(&x.to_le_bytes()),
                StDtype::F16 => data.extend_from_slice(&f32_to_f16(x).to_le_bytes()),
                StDtype::BF16 => data.extend_from_slice(&f32_to_bf16(x).to_le_bytes()),
            }
        }
        let tag = match dtype {
            StDtype::F32 => "F32",
            StDtype::F16 => "F16",
            StDtype::BF16 => "BF16",
        };
        header.insert(
            name.clone(),
            json!({"dtype": tag, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    std::fs::write(path, out).unwrap();
}

pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = b & 0x7F_FFFF;
    if exp >= 31 {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - exp);
        let r = (m + 0xFFF + ((m >> 13) & 1)) >> 13;
        return sign | r as u16;
    }
    let r = ((exp as u32) << 10 | (mant >> 13))
        + (((mant & 0x1FFF) + ((mant >> 13) & 1) + 0xFFF) >> 13);
    sign | r as u16
}

pub fn write_json(path: &Path, v: &Value) {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    std::fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}

/// Fresh fixture directory `target/golden-fixtures/<case>`.
pub fn fixture_dir(case: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/golden-fixtures")
        .join(case);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A WordLevel `tokenizer.json` over `<pad>`, `<s>`, `</s>`, `<unk>`, then `t4 .. t{n-1}`, split on
/// whitespace. Token `tK` encodes to id K.
pub fn write_tokenizer(dir: &Path, n_vocab: usize) {
    let mut vocab = Map::new();
    for (i, s) in ["<pad>", "<s>", "</s>", "<unk>"].iter().enumerate() {
        vocab.insert(s.to_string(), json!(i));
    }
    for i in 4..n_vocab {
        vocab.insert(format!("t{i}"), json!(i));
    }
    let added: Vec<Value> = ["<pad>", "<s>", "</s>", "<unk>"]
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({"id": i, "content": s, "single_word": false, "lstrip": false,
                              "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    write_json(
        &dir.join("tokenizer.json"),
        &json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
        }),
    );
}

/// `ModelArtifacts` for a fixture directory: config.json, tokenizer.json and `weights` (file name
/// inside `dir`), when present.
pub fn artifacts(dir: &Path, model_id: &str, weights: &str) -> ModelArtifacts {
    let opt = |p: PathBuf| p.exists().then_some(p);
    ModelArtifacts {
        model_id: model_id.into(),
        local_dir: Some(dir.to_path_buf()),
        config_path: opt(dir.join("config.json")),
        tokenizer_path: opt(dir.join("tokenizer.json")),
        weights_path: opt(dir.join(weights)),
        is_cached: true,
        ..Default::default()
    }
}

pub fn writing() -> bool {
    std::env::var("GIBBS_GOLDEN_WRITE").is_ok_and(|v| v == "1")
}

/// Devices a case runs on: CPU when writing (the reference), CPU and Metal (when present) when
/// checking.
pub fn devices() -> Vec<Device> {
    if writing() || !metal_present() {
        vec![Device::Cpu]
    } else {
        vec![Device::Cpu, Device::Metal]
    }
}

pub fn metal_present() -> bool {
    cfg!(target_os = "macos") && std::env::var("GIBBS_GOLDEN_CPU_ONLY").is_err()
}

/// Tolerances: relative L2 error and cosine similarity per float output.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    pub rel: f64,
    pub cos: f64,
}

pub const TOL: Tol = Tol {
    rel: 1e-3,
    cos: 0.9999,
};

/// Collected outputs of one case run.
#[derive(Default)]
pub struct Out {
    pub floats: Vec<(String, Vec<f32>)>,
    pub ids: Vec<(String, Vec<u32>)>,
}

impl Out {
    pub fn f(&mut self, name: impl Into<String>, v: &[f32]) -> &mut Self {
        self.floats.push((name.into(), v.to_vec()));
        self
    }
    pub fn ids(&mut self, name: impl Into<String>, v: &[u32]) -> &mut Self {
        self.ids.push((name.into(), v.to_vec()));
        self
    }
}

fn golden_path(case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{case}.json"))
}

/// Writes (`GIBBS_GOLDEN_WRITE=1`) or checks the outputs of `case` run on `device`. Panics with a
/// per-output report on mismatch. Float outputs: rel L2 error < `tol.rel` and cosine > `tol.cos`;
/// id outputs: exact.
pub fn check(case: &str, device: Device, out: &Out, tol: Tol) {
    let path = golden_path(case);
    if writing() {
        let mut f = Map::new();
        for (k, v) in &out.floats {
            assert!(
                v.iter().all(|x| x.is_finite()),
                "{case}/{k}: non-finite output"
            );
            f.insert(k.clone(), json!(v));
        }
        let mut i = Map::new();
        for (k, v) in &out.ids {
            i.insert(k.clone(), json!(v));
        }
        write_json(&path, &json!({"case": case, "f32": f, "ids": i}));
        eprintln!("golden: wrote {}", path.display());
        return;
    }
    let g: Value = serde_json::from_slice(
        &std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (GIBBS_GOLDEN_WRITE=1)", path.display())),
    )
    .unwrap();
    let mut errs = Vec::new();
    let gf = g["f32"].as_object().cloned().unwrap_or_default();
    let gi = g["ids"].as_object().cloned().unwrap_or_default();
    assert_eq!(gf.len(), out.floats.len(), "{case}: float output count");
    assert_eq!(gi.len(), out.ids.len(), "{case}: id output count");
    for (k, v) in &out.floats {
        let Some(want) = gf.get(k) else {
            errs.push(format!("{k}: missing in golden"));
            continue;
        };
        let want: Vec<f64> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        if want.len() != v.len() {
            errs.push(format!("{k}: len {} vs golden {}", v.len(), want.len()));
            continue;
        }
        let (mut d2, mut w2, mut dot, mut v2) = (0f64, 0f64, 0f64, 0f64);
        for (&a, &b) in v.iter().zip(&want) {
            let a = a as f64;
            d2 += (a - b) * (a - b);
            w2 += b * b;
            v2 += a * a;
            dot += a * b;
        }
        let rel = (d2 / w2.max(1e-30)).sqrt();
        let cos = if w2 == 0.0 && v2 == 0.0 {
            1.0
        } else {
            dot / (w2.sqrt() * v2.sqrt()).max(1e-30)
        };
        if !(rel < tol.rel && cos > tol.cos) {
            errs.push(format!("{k}: rel {rel:.3e} cos {cos:.6}"));
        }
    }
    for (k, v) in &out.ids {
        let want: Vec<u64> = gi
            .get(k)
            .and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_u64().unwrap()).collect())
            .unwrap_or_default();
        if want.iter().map(|&x| x as u32).collect::<Vec<_>>() != *v {
            errs.push(format!("{k}: ids {v:?} vs golden {want:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "{case} on {device:?}:\n  {}",
        errs.join("\n  ")
    );
}

/// Runs `run(device)` on every device of [`devices`] and checks each result.
pub fn golden(case: &str, tol: Tol, run: impl Fn(Device) -> Out) {
    for d in devices() {
        let out = run(d);
        check(case, d, &out, tol);
    }
}
