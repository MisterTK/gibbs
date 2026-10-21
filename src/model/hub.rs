use crate::hardware::QuantFormat;
use crate::model::variant::hub_rank;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Default maximum parameter budget ceiling (10M parameters).
pub const DEFAULT_MAX_PARAM_BUDGET: usize = 10_000_000;

/// Artifact locations and metadata for a resolved or inspected model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifacts {
    pub model_id: String,
    pub local_dir: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub weights_path: Option<PathBuf>,
    pub estimated_params: usize,
    pub is_cached: bool,
    pub quant_format: Option<QuantFormat>,
}

impl Default for ModelArtifacts {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            local_dir: None,
            config_path: None,
            tokenizer_path: None,
            weights_path: None,
            estimated_params: 0,
            is_cached: false,
            quant_format: None,
        }
    }
}

/// Quant tag of a `.gguf` filename (`-Q4_K_M.gguf`, `.q8_0.gguf`, ...), as the file's dominant GGML
/// type (files mix types; decoding uses each tensor's own type). `None` when the tag names a GGML
/// type the loader cannot decode (Q8_K, repacked Q4_0_x_y, ...); untagged files report Q4_K_M.
fn gguf_file_quant(stem: &str) -> Option<QuantFormat> {
    for raw_tag in stem.split(['-', '.']) {
        let tag = raw_tag.replace('_', "");
        // candle names its files `model-q4k.gguf` (google/madlad400-*-mt): `q4k` = `q4_k`.
        let format = match tag.as_str() {
            t if t.starts_with("q4k") => QuantFormat::GGUF_Q4_K_M,
            "q40" => QuantFormat::GGUF_Q4_0,
            "q41" => QuantFormat::GGUF_Q4_1,
            t if t.starts_with("q5k") => QuantFormat::GGUF_Q5_K_M,
            "q50" => QuantFormat::GGUF_Q5_0,
            "q51" => QuantFormat::GGUF_Q5_1,
            t if t.starts_with("q6k") => QuantFormat::GGUF_Q6_K,
            "q80" => QuantFormat::GGUF_Q8_0,
            "bf16" => QuantFormat::BF16,
            "f16" | "fp16" => QuantFormat::FP16,
            "f32" | "fp32" => QuantFormat::FP32,
            t if t.starts_with("q2k") => QuantFormat::GGUF_Q2_K,
            t if t.starts_with("q3k") => QuantFormat::GGUF_Q3_K,
            "iq4xs" => QuantFormat::GGUF_IQ4_XS,
            "iq4nl" => QuantFormat::GGUF_IQ4_NL,
            "iq3xxs" => QuantFormat::GGUF_IQ3_XXS,
            "iq3xs" | "iq3s" | "iq3m" => QuantFormat::GGUF_IQ3_S,
            "iq2xxs" => QuantFormat::GGUF_IQ2_XXS,
            "iq2xs" => QuantFormat::GGUF_IQ2_XS,
            "iq2s" => QuantFormat::GGUF_IQ2_S,
            "iq2m" => QuantFormat::GGUF_IQ2_M,
            "iq1s" => QuantFormat::GGUF_IQ1_S,
            "iq1m" => QuantFormat::GGUF_IQ1_M,
            "tq10" => QuantFormat::GGUF_TQ1_0,
            "tq20" => QuantFormat::GGUF_TQ2_0,
            "mxfp4" | "mxfp4moe" => QuantFormat::MXFP4,
            t if ["q40", "q8k", "iq", "tq", "mxfp"]
                .iter()
                .any(|p| t.starts_with(p)) =>
            {
                return None
            }
            _ => continue,
        };
        return Some(format);
    }
    Some(QuantFormat::GGUF_Q4_K_M)
}

/// Detects quantization format and corresponding bits-per-weight from a filename or path.
pub fn detect_quant_format(path: &Path) -> Option<QuantFormat> {
    let filename = path.file_name().and_then(|s| s.to_str())?.to_lowercase();
    if let Some(stem) = filename.strip_suffix(".gguf") {
        // Split sets are often foldered by quant (`IQ2_S/M-00001-of-00003.gguf`); the file's own tag wins.
        let dir = path
            .parent()
            .and_then(|d| d.file_name())
            .and_then(|d| d.to_str())
            .unwrap_or("")
            .to_lowercase();
        return gguf_file_quant(&format!("{stem}-{dir}"));
    }
    if filename.contains("nvfp4") || filename.ends_with(".nvfp4") {
        Some(QuantFormat::NVFP4_E2M1)
    } else if filename.contains("kivi") {
        Some(QuantFormat::KIVI_2Bit)
    } else if filename.contains("turbo") {
        Some(QuantFormat::TurboQuant)
    } else if filename.contains("bf16") || filename.contains("bfloat16") {
        Some(QuantFormat::BF16)
    } else if filename.contains("fp32") || filename.contains("f32") {
        Some(QuantFormat::FP32)
    } else if filename.contains("fp16") || filename.contains("f16") {
        Some(QuantFormat::FP16)
    } else if filename.ends_with(".safetensors")
        || filename.ends_with(".bin")
        || filename.ends_with(".pt")
    {
        Some(QuantFormat::FP16)
    } else {
        None
    }
}

/// Checks whether a weight file satisfies an explicit quant request. Specific tags (`Q4_K_M`, `Q4_0`,
/// `FP16`/`f16`, `BF16`, ...) match exactly one format; only bare family names (`q4`, `q5`,
/// `fp4`/`nvfp4`, `uncompressed`) span several. Unknown tags match a whole `-`/`.`-separated filename token.
pub fn quant_matches(expected_quant: &str, format: QuantFormat, filename: &str) -> bool {
    let exp = expected_quant.trim().to_lowercase().replace('-', "_");
    match exp.as_str() {
        "q4" => matches!(
            format,
            QuantFormat::GGUF_Q4_K_M | QuantFormat::GGUF_Q4_0 | QuantFormat::GGUF_Q4_1
        ),
        "q5" => matches!(
            format,
            QuantFormat::GGUF_Q5_K_M | QuantFormat::GGUF_Q5_0 | QuantFormat::GGUF_Q5_1
        ),
        "q6" => matches!(format, QuantFormat::GGUF_Q6_K),
        "nvfp4" | "fp4" => matches!(
            format,
            QuantFormat::NVFP4 | QuantFormat::NVFP4_E2M1 | QuantFormat::MXFP4
        ),
        "uncompressed" => matches!(
            format,
            QuantFormat::FP32 | QuantFormat::FP16 | QuantFormat::BF16
        ),
        _ => {
            let token = || {
                filename
                    .to_lowercase()
                    .split(['-', '.', '/'])
                    .any(|t| t == exp)
            };
            // GGUF size variants share a tensor format (Q3_K_S/M/L, Q4_K_S/M, IQ3_XS/S/M), so a
            // GGUF tag must also prefix a filename token: `Q3_K_M` never yields `-Q3_K_L`, `q4_k` spans both.
            let gguf_tag = ["iq", "tq", "mxfp"].iter().any(|p| exp.starts_with(p))
                || (exp.starts_with('q') && exp.as_bytes().get(1).is_some_and(u8::is_ascii_digit));
            let tagged = || {
                filename
                    .to_lowercase()
                    .split(['-', '.', '/'])
                    .any(|t| t.starts_with(&exp))
            };
            match QuantFormat::from_str_lenient(&exp) {
                Some(parsed) => parsed == format && (!gguf_tag || tagged()),
                None => token(),
            }
        }
    }
}

/// Splits `<org>/<repo>/<file>.(gguf|safetensors|bin|pt|pth|ckpt|msgpack|npz|npy)` into (`<org>/<repo>`, `Some(<file>)`);
/// any other id passes through as (`model_id`, `None`).
pub fn split_weight_file(model_id: &str) -> (&str, Option<&str>) {
    let is_weight_file = Path::new(model_id)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            [
                "gguf",
                "safetensors",
                "bin",
                "pt",
                "pth",
                "ckpt",
                "msgpack",
                "npz",
                "npy",
            ]
            .iter()
            .any(|w| e.eq_ignore_ascii_case(w))
        });
    match model_id.rsplit_once('/') {
        Some((repo, file)) if is_weight_file => (repo, Some(file)),
        _ => (model_id, None),
    }
}

/// `<org>/<repo>@<revision>` → (`<org>/<repo>`, `<revision>`: branch, tag or commit); without `@`
/// the revision is `main`. A revision is cached in `snapshots/<revision>`.
pub fn split_revision(repo_id: &str) -> (&str, &str) {
    repo_id.split_once('@').unwrap_or((repo_id, "main"))
}

/// A revision as one URL path segment / snapshot directory name.
pub(crate) fn rev_segment(rev: &str) -> String {
    rev.replace('/', "%2F")
}
fn rev_dir(rev: &str) -> String {
    rev.replace('/', "--")
}

/// Storage backend options for caching and persisting model artifacts across local disk and cloud storage.
///
/// Supports multi-cloud streaming and local caching protocols:
/// - **LocalDisk**: Direct filesystem path caching. Avoids network latency and enables
///   memory-mapped reads (`mmap`) for instant zero-copy tensor access.
/// - **S3**: Amazon S3 / MinIO object storage with optional custom endpoint. Streamingly
///   syncs model repositories or single-file quantized weights (`.Q4_K_M.gguf`, `.q8_0.gguf`).
/// - **Gcs**: Google Cloud Storage (`gs://` or `gcs://`) for cloud-native model deployment.
///
/// Across all backends, pre-quantized single-file GGUF models (`.Q4_K_M.gguf`, `.q8_0.gguf`)
/// are prioritized over uncompressed `.safetensors` artifacts. This minimizes storage transfer
/// times, cuts cloud egress and local cache footprint by 60-80%, and drastically reduces DRAM
/// memory bandwidth pressure during autoregressive KV decode passes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// Local filesystem directory cache (supports zero-copy mmap tensor reads).
    LocalDisk { cache_dir: PathBuf },
    /// Amazon S3 or S3-compatible (MinIO/Ceph) object storage.
    S3 {
        bucket: String,
        prefix: String,
        endpoint: Option<String>,
    },
    /// Google Cloud Storage (GCS) object storage.
    Gcs { bucket: String, prefix: String },
}

impl Default for StorageBackend {
    fn default() -> Self {
        let cache_root = if let Ok(val) = std::env::var("HF_HUB_CACHE") {
            PathBuf::from(val)
        } else if let Ok(val) = std::env::var("HF_HOME") {
            PathBuf::from(val).join("hub")
        } else if let Ok(val) = std::env::var("HOME") {
            PathBuf::from(val)
                .join(".cache")
                .join("huggingface")
                .join("hub")
        } else {
            PathBuf::from(".cache/huggingface/hub")
        };
        StorageBackend::LocalDisk {
            cache_dir: cache_root,
        }
    }
}

impl StorageBackend {
    /// Parses a storage URI string into a StorageBackend.
    ///
    /// - `s3://bucket/prefix` -> S3
    /// - `gs://bucket/prefix` or `gcs://bucket/prefix` -> Gcs
    /// - Otherwise -> LocalDisk with path.
    pub fn from_uri(uri: &str) -> Result<Self, String> {
        if let Some(rest) = uri.strip_prefix("s3://") {
            let (bucket, prefix) = match rest.split_once('/') {
                Some((b, p)) => (b.to_string(), p.trim_matches('/').to_string()),
                None => (rest.to_string(), String::new()),
            };
            if bucket.is_empty() {
                return Err("Invalid S3 URI: missing bucket name".to_string());
            }
            Ok(StorageBackend::S3 {
                bucket,
                prefix,
                endpoint: None,
            })
        } else if let Some(rest) = uri
            .strip_prefix("gs://")
            .or_else(|| uri.strip_prefix("gcs://"))
        {
            let (bucket, prefix) = match rest.split_once('/') {
                Some((b, p)) => (b.to_string(), p.trim_matches('/').to_string()),
                None => (rest.to_string(), String::new()),
            };
            if bucket.is_empty() {
                return Err("Invalid GCS URI: missing bucket name".to_string());
            }
            Ok(StorageBackend::Gcs { bucket, prefix })
        } else {
            let path_str = uri.strip_prefix("file://").unwrap_or(uri);
            Ok(StorageBackend::LocalDisk {
                cache_dir: PathBuf::from(path_str),
            })
        }
    }

    /// Syncs model artifacts from remote storage to local destination if configured.
    pub fn sync_from_remote(&self, model_id: &str, local_target: &Path) -> Result<(), String> {
        match self {
            StorageBackend::LocalDisk { .. } => Ok(()),
            StorageBackend::S3 {
                bucket,
                prefix,
                endpoint,
            } => {
                let remote_uri = if prefix.is_empty() {
                    format!("s3://{}/{}", bucket, model_id)
                } else {
                    format!("s3://{}/{}/{}", bucket, prefix.trim_matches('/'), model_id)
                };
                let mut cmd = std::process::Command::new("aws");
                cmd.arg("s3").arg("sync").arg(&remote_uri).arg(local_target);
                if let Some(ep) = endpoint {
                    cmd.arg("--endpoint-url").arg(ep);
                }
                let output = cmd
                    .output()
                    .map_err(|e| format!("Failed to run aws s3 sync: {}", e))?;
                if !output.status.success() {
                    return Err(format!(
                        "aws s3 sync failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                Ok(())
            }
            StorageBackend::Gcs { bucket, prefix } => {
                let remote_uri = if prefix.is_empty() {
                    format!("gs://{}/{}", bucket, model_id)
                } else {
                    format!("gs://{}/{}/{}", bucket, prefix.trim_matches('/'), model_id)
                };
                let mut cmd = std::process::Command::new("gcloud");
                cmd.arg("storage")
                    .arg("rsync")
                    .arg(&remote_uri)
                    .arg(local_target);
                let output = cmd
                    .output()
                    .map_err(|e| format!("Failed to run gcloud storage rsync: {}", e))?;
                if !output.status.success() {
                    return Err(format!(
                        "gcloud storage rsync failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                Ok(())
            }
        }
    }

    /// Syncs local model artifacts to remote storage if configured.
    pub fn sync_to_remote(&self, local_source: &Path, model_id: &str) -> Result<(), String> {
        match self {
            StorageBackend::LocalDisk { .. } => Ok(()),
            StorageBackend::S3 {
                bucket,
                prefix,
                endpoint,
            } => {
                let remote_uri = if prefix.is_empty() {
                    format!("s3://{}/{}", bucket, model_id)
                } else {
                    format!("s3://{}/{}/{}", bucket, prefix.trim_matches('/'), model_id)
                };
                let mut cmd = std::process::Command::new("aws");
                cmd.arg("s3").arg("sync").arg(local_source).arg(&remote_uri);
                if let Some(ep) = endpoint {
                    cmd.arg("--endpoint-url").arg(ep);
                }
                let output = cmd
                    .output()
                    .map_err(|e| format!("Failed to run aws s3 sync: {}", e))?;
                if !output.status.success() {
                    return Err(format!(
                        "aws s3 sync failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                Ok(())
            }
            StorageBackend::Gcs { bucket, prefix } => {
                let remote_uri = if prefix.is_empty() {
                    format!("gs://{}/{}", bucket, model_id)
                } else {
                    format!("gs://{}/{}/{}", bucket, prefix.trim_matches('/'), model_id)
                };
                let mut cmd = std::process::Command::new("gcloud");
                cmd.arg("storage")
                    .arg("rsync")
                    .arg(local_source)
                    .arg(&remote_uri);
                let output = cmd
                    .output()
                    .map_err(|e| format!("Failed to run gcloud storage rsync: {}", e))?;
                if !output.status.success() {
                    return Err(format!(
                        "gcloud storage rsync failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                Ok(())
            }
        }
    }
}

/// One file of a Hub repo listing (or of a local snapshot).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub path: String,
    pub size: Option<u64>,
}

/// Byte counts of the running download, shared with whoever displays it.
/// `total`: bytes still to fetch when the download started; `done`: bytes fetched since.
#[derive(Debug, Default)]
pub struct DownloadProgress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub file: parking_lot::Mutex<String>,
}

/// Non-weight files fetched with a model when the repo lists them.
const METADATA_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    // GLQ e8_shell lattice codebook (its row order isn't reproducible from the enumeration).
    "e8_codebook.pt",
    // Encoder-decoder tokenizers (SentencePiece, Unigram, BPE).
    "source.spm",
    "target.spm",
    "vocab.json",
    "target_vocab.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "merges.txt",
    "vocab-src.json",
    "vocab-tgt.json",
];

/// Whether `path` is a model metadata file the hub downloads alongside the weights.
pub fn is_metadata_file(path: &str) -> bool {
    METADATA_FILES.contains(&path)
}

/// ONNX decoder exports the loader reads, best (smallest) first when sizes are unknown.
const ONNX_VARIANTS: &[&str] = &[
    "q4f16",
    "q4",
    "bnb4",
    "int8",
    "uint8",
    "quantized",
    "fp16",
    "",
];

/// Recursive file listing with sizes (HF tree API), following `Link: <…>; rel="next"` pages.
pub fn list_repo_files(repo_id: &str, token: Option<&str>) -> Result<Vec<RepoFile>, String> {
    let (repo, rev) = split_revision(repo_id);
    let mut url = format!(
        "https://huggingface.co/api/models/{repo}/tree/{}?recursive=true",
        rev_segment(rev)
    );
    let mut files = Vec::new();
    loop {
        let mut cmd = std::process::Command::new("curl");
        cmd.args([
            "-fsSL",
            "--connect-timeout",
            "15",
            "-m",
            "60",
            "-w",
            "\n%header{link}",
        ]);
        if let Some(t) = token.filter(|t| !t.is_empty()) {
            cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
        }
        let out = cmd
            .arg(&url)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("Failed to execute curl: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "cannot list files of {repo_id}: {}",
                stderr.lines().next().unwrap_or("curl failed").trim()
            ));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let (body, link) = text.rsplit_once('\n').unwrap_or((&text, ""));
        let page: Vec<serde_json::Value> = serde_json::from_str(body)
            .map_err(|e| format!("unexpected file listing for {repo_id}: {e}"))?;
        files.extend(page.iter().filter(|e| e["type"] == "file").filter_map(|e| {
            Some(RepoFile {
                path: e["path"].as_str()?.to_string(),
                size: e["size"].as_u64(),
            })
        }));
        let next = link
            .split(',')
            .find(|l| l.contains("rel=\"next\""))
            .and_then(|l| Some(l[l.find('<')? + 1..l.find('>')?].to_string()));
        match next {
            Some(n) => url = n,
            None => return Ok(files),
        }
    }
}

/// One Hub API JSON document (curl GET).
pub(crate) fn hub_json(url: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-fsSL", "--connect-timeout", "15", "-m", "60"]);
    if let Some(t) = token.filter(|t| !t.is_empty()) {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let out = cmd
        .arg(url)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("Failed to execute curl: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "{url}: {}",
            stderr.lines().next().unwrap_or("curl failed").trim()
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{url}: {e}"))
}

/// Branches of `repo` other than `main` (HF refs API).
pub fn list_branches(repo: &str, token: Option<&str>) -> Result<Vec<String>, String> {
    let refs = hub_json(
        &format!("https://huggingface.co/api/models/{repo}/refs"),
        token,
    )?;
    Ok(refs["branches"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|b| b["name"].as_str())
        .filter(|n| *n != "main")
        .map(String::from)
        .collect())
}

/// `/api/models`-shaped entries (config, siblings, …) of every non-main branch of `repo`, ids
/// rewritten to `<repo>@<branch>`, as one JSON array: repos that keep one quantization per branch
/// (e.g. EXL3 bitrates) and only a README on main.
pub fn branch_models(repo: &str, token: Option<&str>) -> Result<String, String> {
    let expand: Vec<String> = [
        "config",
        "siblings",
        "safetensors",
        "pipeline_tag",
        "library_name",
        "gated",
        "lastModified",
    ]
    .iter()
    .map(|e| format!("expand%5B%5D={e}"))
    .collect();
    let branches = list_branches(repo, token)?;
    let models: Vec<serde_json::Value> = std::thread::scope(|s| {
        let handles: Vec<_> = branches
            .iter()
            .map(|b| {
                let url = format!(
                    "https://huggingface.co/api/models/{repo}/revision/{}?{}",
                    rev_segment(b),
                    expand.join("&")
                );
                s.spawn(move || {
                    hub_json(&url, token).map(|mut m| {
                        m["id"] = format!("{repo}@{b}").into();
                        m
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("branch lookup panicked".into()))
            })
            .collect::<Result<_, String>>()
    })?;
    Ok(serde_json::Value::from(models).to_string())
}

/// Every shard of the split GGUF whose first shard is `first`, in order. Err names the missing one.
fn gguf_split_set<'a>(files: &'a [RepoFile], first: &'a str) -> Result<Vec<&'a str>, String> {
    let (prefix, _, total, w) = shard_of(first, ".gguf").ok_or("not a split GGUF")?;
    (1..=total)
        .map(|n| {
            let want = format!("{prefix}-{n:0w$}-of-{total:0w$}.gguf");
            files
                .iter()
                .find(|f| f.path == want)
                .map(|f| f.path.as_str())
                .ok_or(format!("split GGUF {first} is missing shard {want}"))
        })
        .collect()
}

/// `<prefix>-NNNNN-of-MMMMM.<ext>` → (`prefix`, N, M, digits of M).
pub(crate) fn shard_of<'a>(path: &'a str, ext: &str) -> Option<(&'a str, u32, u32, usize)> {
    let (head, total) = path.strip_suffix(ext)?.rsplit_once("-of-")?;
    let (prefix, idx) = head.rsplit_once('-')?;
    Some((prefix, idx.parse().ok()?, total.parse().ok()?, total.len()))
}

/// PyTorch pickle / Flax / NumPy weight extension of a file the loader converts (trainer state and
/// LoRA adapters excluded). `.bin` is ambiguous (OpenVINO IR `openvino_model.bin`, CTranslate2
/// `model.bin`, GGML `ggml-*.bin` are not pickles), so only transformers' own
/// `pytorch_model*.bin` names are picked; any other `.bin` loads as an explicit file id.
fn fallback_weight_ext(path: &str) -> Option<&'static str> {
    const NOT_WEIGHTS: [&str; 8] = [
        "e8_codebook", // GLQ codebook (fetched as metadata)
        "adapter_",
        "optimizer",
        "scheduler",
        "rng_state",
        "training_args",
        "trainer_state",
        "mp_rank_",
    ];
    let ext = [".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".npz"]
        .into_iter()
        .find(|e| path.ends_with(e))?;
    if ext == ".bin" && !path.starts_with("pytorch_model") {
        return None;
    }
    (!NOT_WEIGHTS.iter().any(|t| path.starts_with(t))).then_some(ext)
}

/// `onnx/model_<variant>.onnx` → variant (`""` for `onnx/model.onnx`), if one the loader reads.
fn onnx_variant(path: &str) -> Option<&str> {
    let stem = path
        .strip_prefix("onnx/")?
        .strip_suffix(".onnx")?
        .strip_prefix("model")?;
    let v = stem.strip_prefix('_').unwrap_or(stem);
    (stem.is_empty() == v.is_empty() && ONNX_VARIANTS.contains(&v)).then_some(v)
}

/// A decodable weight set: primary file first, then companions.
struct WeightSet<'a> {
    files: Vec<&'a str>,
    /// 0: safetensors / GGUF, 1: ONNX, 2: PyTorch / Flax / NumPy (only for repos without the others).
    tier: u8,
    /// Ascending preference within the tier: bpw (tier 0); bytes or variant order (ONNX).
    rank: f64,
    /// Matches an explicit quant request.
    matches: Box<dyn Fn(&str) -> bool + 'a>,
}

/// The set to fetch: only sets matching an explicit quant count; otherwise smaller bpw / size
/// formats rank first (smallest decodable quant is the default), then tier, then rank.
fn best_set<'a>(
    mut sets: Vec<WeightSet<'a>>,
    explicit_quant: Option<&str>,
) -> Option<Vec<&'a str>> {
    if let Some(eq) = explicit_quant {
        sets.retain(|s| (s.matches)(eq));
    }
    sets.sort_by(|a, b| {
        a.tier
            .cmp(&b.tier)
            .then(a.rank.partial_cmp(&b.rank).unwrap_or(Ordering::Equal))
            .then(a.files[0].cmp(b.files[0]))
    });
    sets.into_iter().next().map(|s| s.files)
}

/// ComfyUI's split-file Krea 2 repos (`diffusion_models/krea2_{raw,turbo}_<quant>.safetensors`
/// beside `text_encoders/` and `vae/`): the transformer (turbo unless `quant` is a substring of
/// another file's name, e.g. `raw`), the text encoder and the VAE. Only the bf16 files load.
// ponytail: keyed on the `krea2_` file prefix; other ComfyUI split layouts still fall through.
fn comfy_krea2_files(files: &[RepoFile], quant: Option<&str>) -> Result<Vec<String>, String> {
    let dir = |d: &str| -> Vec<&str> {
        let in_dir = |p: &&str| {
            p.strip_prefix(d)
                .is_some_and(|n| !n.contains('/') && n.ends_with(".safetensors"))
        };
        files
            .iter()
            .map(|f| f.path.as_str())
            .filter(in_dir)
            .collect()
    };
    let (dits, tes, vaes) = (dir("diffusion_models/"), dir("text_encoders/"), dir("vae/"));
    fn bf16<'a>(c: &[&'a str], want: &str) -> Option<&'a str> {
        c.iter()
            .copied()
            .find(|p| p.contains("bf16") && p.contains(want))
    }
    let want = quant.map(str::to_lowercase);
    let dit = match &want {
        Some(q) => bf16(&dits, q),
        None => bf16(&dits, "turbo").or_else(|| bf16(&dits, "raw")),
    };
    match (dit, bf16(&tes, ""), vaes.first()) {
        (Some(d), Some(t), Some(v)) => Ok(vec![d.into(), t.into(), (*v).into()]),
        _ => Err(format!(
            "ComfyUI Krea 2 split files: only the bf16 transformer, text encoder and VAE load, not the float8 / nvfp4 / mxfp8 / int8 quantizations (found: {})",
            dits.iter().chain(&tes).chain(&vaes).copied().collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Weight files to fetch for this repo (primary first, then companions: shards / onnx external data).
/// Err(reason) when nothing the loader decodes exists. Single source of truth for download,
/// cached-locate and picker loadability.
///
/// Default ranking: highest predicted decode speed first (`model::variant::hub_rank`, 5% steps,
/// then smaller bytes) among `*.gguf` / single or sharded `*.safetensors`, then ONNX variants (smallest download first),
/// then PyTorch / Flax / NumPy files (single or `.index.json`-sharded; smallest download first),
/// then sub-4.5-bpw GGUFs. An explicit quant only ever yields a matching set.
pub fn select_weight_files(
    files: &[RepoFile],
    explicit_quant: Option<&str>,
) -> Result<Vec<String>, String> {
    if files.iter().any(|f| f.path == "model_index.json") {
        return diffusers_files(files);
    }
    if files
        .iter()
        .any(|f| f.path.starts_with("diffusion_models/krea2_"))
    {
        return comfy_krea2_files(files, explicit_quant);
    }
    let mut sets: Vec<WeightSet> = Vec::new();
    let mut problem = None;
    let size_of = |p: &str| files.iter().find(|f| f.path == p).and_then(|f| f.size);
    let policy = crate::model::variant::reencode_policy();
    // ponytail: ComfyUI packaging (`comfyui/` workflows or `text_encoders/` beside root GGUFs) ships
    // the same diffusion transformer again as root fp8 / int8 / NVFP4 / MLX / LoRA safetensors, in
    // forms no loader reads; those never compete with the GGUFs. Upgrade: read the Hub's GGUF arch.
    let comfy = files
        .iter()
        .any(|f| f.path.ends_with(".gguf") && !f.path.contains('/'))
        && files
            .iter()
            .any(|f| f.path.starts_with("comfyui/") || f.path.starts_with("text_encoders/"));
    for f in files {
        let p = f.path.as_str();
        let lower = p.to_lowercase();
        let root = !p.contains('/');
        if lower.ends_with(".gguf") {
            // `mmproj-*` files are vision projectors, `imatrix*.gguf` importance matrices and `mtp-*`
            // (MTP drafter) sidecars, not the language model.
            let name = lower.rsplit('/').next().unwrap_or(&lower);
            if is_mmproj(name)
                || name.starts_with("imatrix")
                || name.ends_with(".imatrix.gguf")
                || name.starts_with("mtp-")
            {
                continue;
            }
            // A split set (`-00001-of-00003.gguf`) is one weight set, listed once at its first shard.
            let members = match shard_of(&lower, ".gguf") {
                Some((_, 1, _, _)) => match gguf_split_set(files, p) {
                    Ok(set) => set,
                    Err(e) => {
                        problem = Some(e);
                        continue;
                    }
                },
                Some(_) => continue,
                None => vec![p],
            };
            let Some(q) = detect_quant_format(Path::new(p)) else {
                continue;
            };
            let bytes: Option<u64> = members.iter().map(|m| size_of(m)).sum();
            let rank = hub_rank(q, bytes, policy);
            sets.push(WeightSet {
                files: members,
                tier: 0,
                rank,
                matches: Box::new(move |eq| quant_matches(eq, q, p)),
            });
        } else if root && lower.ends_with(crate::model::ninfer::EXT) {
            // A NInfer artifact is its entry file plus the `.part-NNNN` volumes beside it; it wins
            // over the vision / imatrix safetensors some repos carry, the smallest variant first.
            let mut members = vec![p];
            members.extend(
                files
                    .iter()
                    .map(|d| d.path.as_str())
                    .filter(|d| d.strip_prefix(p).is_some_and(|r| r.starts_with(".part-"))),
            );
            let bytes: Option<u64> = members.iter().map(|m| size_of(m)).sum();
            sets.push(WeightSet {
                files: members,
                tier: 0,
                rank: bytes.unwrap_or(0) as f64 - 1e18,
                matches: Box::new(|eq| eq.eq_ignore_ascii_case("ninfer")),
            });
        } else if root
            && lower.ends_with(".safetensors")
            && !lower.starts_with("adapter_")
            // Original-format names (`layers.0.attention.wq`) beside the HF-format shards.
            && !lower.starts_with("consolidated")
            && shard_of(p, ".safetensors").is_none()
            && !comfy
        {
            let Some(q) = detect_quant_format(Path::new(p)) else {
                continue;
            };
            let bytes = size_of(p);
            sets.push(WeightSet {
                files: vec![p],
                tier: 0,
                // SD repos ship `*-inpainting.safetensors` beside the text-to-image checkpoint.
                rank: hub_rank(q, bytes, policy)
                    + if lower.contains("inpaint") { 1e6 } else { 0.0 },
                matches: Box::new(move |eq| quant_matches(eq, q, p)),
            });
        } else if let Some((prefix, ext)) =
            p.strip_suffix(".index.json")
                .filter(|_| root)
                .and_then(|f| {
                    let ext = Some(".safetensors")
                        .filter(|e| f.ends_with(e))
                        .or_else(|| fallback_weight_ext(f))?;
                    Some((f.strip_suffix(ext)?, ext))
                })
        {
            let mut shards: Vec<(u32, u32, usize, &str)> = files
                .iter()
                .filter_map(|s| {
                    let (pre, n, total, w) = shard_of(&s.path, ext)?;
                    // Some repos name shards `model.safetensors-00001-of-00001.safetensors`.
                    let pre = pre.strip_suffix(ext).unwrap_or(pre);
                    (pre == prefix).then_some((n, total, w, s.path.as_str()))
                })
                .collect();
            // An index without `-of-` shards (e.g. beside a single model.safetensors) adds nothing.
            let Some(&(_, total, width, first)) = shards.first() else {
                continue;
            };
            shards.sort_unstable();
            if let Some(n) = (1..=total).find(|n| !shards.iter().any(|s| s.0 == *n)) {
                problem = Some(format!(
                    "sharded index {p} lists missing shard {prefix}-{n:0width$}-of-{total:0width$}{ext}"
                ));
                continue;
            }
            let q = detect_quant_format(Path::new(first)).unwrap_or(QuantFormat::FP16);
            let mut members = vec![p];
            members.extend(shards.iter().map(|s| s.3));
            let bytes: Option<u64> = members.iter().map(|m| size_of(m)).sum();
            let tier = if ext == ".safetensors" { 0 } else { 2 };
            sets.push(WeightSet {
                files: members,
                tier,
                rank: if tier == 0 {
                    hub_rank(q, bytes, policy)
                } else {
                    bytes.unwrap_or(0) as f64
                },
                matches: Box::new(move |eq| quant_matches(eq, q, first)),
            });
        } else if fallback_weight_ext(p).is_some_and(|e| root && shard_of(p, e).is_none()) {
            // Pickle / Flax / NumPy: converted at load; smallest download first.
            let q = detect_quant_format(Path::new(p)).unwrap_or(QuantFormat::FP32);
            sets.push(WeightSet {
                files: vec![p],
                tier: 2,
                rank: size_of(p).unwrap_or(0) as f64,
                matches: Box::new(move |eq| quant_matches(eq, q, p)),
            });
        } else if let Some(variant) = onnx_variant(p) {
            let mut members = vec![p];
            members.extend(files.iter().map(|d| d.path.as_str()).filter(|d| {
                d.strip_prefix(p)
                    .is_some_and(|rest| rest == ".data" || rest.starts_with("_data"))
            }));
            let bytes: Option<u64> = members.iter().map(|m| size_of(m)).sum();
            let order = ONNX_VARIANTS
                .iter()
                .position(|v| *v == variant)
                .unwrap_or(0);
            sets.push(WeightSet {
                files: members,
                tier: 1,
                rank: bytes.map_or(order as f64, |b| b as f64),
                matches: Box::new(move |eq| {
                    let eq = eq.trim().to_lowercase();
                    eq == variant
                        || (variant.is_empty() && matches!(eq.as_str(), "fp32" | "f32"))
                        || (variant == "fp16" && eq == "f16")
                }),
            });
        }
    }
    if let Some(best) = best_set(sets, explicit_quant) {
        return Ok(best.into_iter().map(String::from).collect());
    }
    let weights: Vec<&str> = files
        .iter()
        .map(|f| f.path.as_str())
        .filter(|p| {
            let ext = p.rsplit('.').next().unwrap_or("").to_lowercase();
            matches!(
                ext.as_str(),
                "bin"
                    | "pt"
                    | "pth"
                    | "ckpt"
                    | "h5"
                    | "msgpack"
                    | "safetensors"
                    | "gguf"
                    | "onnx"
                    | "npz"
                    | "tflite"
                    | "ot"
                    | "mlmodel"
            )
        })
        .collect();
    if !weights.is_empty() && weights.iter().all(|p| is_mmproj(p)) {
        return Err(MMPROJ_ONLY.into());
    }
    let found = match weights.len() {
        0 => "no weight files".to_string(),
        n if n > 6 => format!("{}, … {} more", weights[..6].join(", "), n - 6),
        _ => weights.join(", "),
    };
    Err(match (explicit_quant, problem) {
        (Some(eq), _) => format!("no weights matching '{eq}' the loader reads (found: {found})"),
        (None, Some(p)) => p,
        (None, None) => format!("no weights the loader reads (found: {found})"),
    })
}

/// Why a repo holding only vision projectors is not a model.
pub const MMPROJ_ONLY: &str =
    "vision projector only (mmproj); load the language-model repo, it fetches this projector";

/// Vision/audio projector GGUFs (`mmproj-*.gguf`, `*-mmproj.gguf`): not language models.
fn is_mmproj(path: &str) -> bool {
    let p = path.to_lowercase();
    p.ends_with(".gguf") && p.rsplit('/').next().is_some_and(|n| n.contains("mmproj"))
}

/// The projector GGUF that goes with the language-model GGUF `primary`: the smallest `mmproj` file
/// in its directory.
pub fn select_mmproj(files: &[RepoFile], primary: &str) -> Option<String> {
    let dir = primary.rsplit_once('/').map_or("", |(d, _)| d);
    files
        .iter()
        .filter(|f| is_mmproj(&f.path) && f.path.rsplit_once('/').map_or("", |(d, _)| d) == dir)
        .min_by_key(|f| (f.size.unwrap_or(u64::MAX), f.path.as_str()))
        .map(|f| f.path.clone())
}

/// [`select_mmproj`] over the snapshot directory holding the cached weights file `weights`.
pub fn local_mmproj(weights: &Path) -> Option<PathBuf> {
    let dir = weights.parent()?;
    let name = weights.file_name()?.to_str()?;
    let files = list_local_files(dir);
    select_mmproj(&files, name).map(|f| dir.join(f))
}

/// Precision variant of a diffusers weight file (`<stem>[.<variant>][-NNNNN-of-NNNNN].safetensors`,
/// `<stem>.safetensors.index[.<variant>].json`), preferred first: `fp16`, `bf16`, none. Other
/// variants (`non_ema`, `ema_only`, ...) are `None`: never fetched or loaded.
fn diffusers_variant(name: &str) -> Option<u8> {
    let tag = match name.split_once(".safetensors.index") {
        Some((_, rest)) => rest.strip_suffix(".json")?.trim_start_matches('.'),
        None => name
            .strip_suffix(".safetensors")?
            .split_once('.')
            .map_or("", |(_, t)| t),
    };
    match tag.split('-').next().unwrap_or("") {
        "fp16" => Some(0),
        "bf16" => Some(1),
        "" => Some(2),
        _ => None,
    }
}

/// diffusers pipeline repo (root `model_index.json`): the index plus, for each component an image
/// pipeline runs (unet / transformer / vae / scheduler / text_encoder* / tokenizer*), its config and
/// tokenizer files and one safetensors set (single or sharded), the `fp16` variant when present.
/// Safety checkers, feature extractors, root single-file checkpoints and the duplicate
/// .bin/.ckpt/ONNX/Flax/OpenVINO copies are never fetched. Primary: the denoiser's weights.
fn diffusers_files(files: &[RepoFile]) -> Result<Vec<String>, String> {
    // `text_encoder`, `text_encoder_2`, ...; not ComfyUI's `text_encoders/` single files.
    let numbered =
        |c: &str, p: &str| c == p || c.strip_prefix(p).is_some_and(|s| s.starts_with('_'));
    let wanted = |c: &str| {
        matches!(
            c,
            "unet" | "transformer" | "vae" | "scheduler" | "processor"
        ) || numbered(c, "text_encoder")
            || numbered(c, "tokenizer")
    };
    let mut comps: Vec<&str> = files
        .iter()
        .filter_map(|f| f.path.split_once('/').map(|(c, _)| c))
        .filter(|c| wanted(c))
        .collect();
    comps.sort_unstable();
    comps.dedup();
    if !comps.iter().any(|c| matches!(*c, "unet" | "transformer")) {
        return Err("diffusers repo without unet/ or transformer/ component".into());
    }
    let (mut weights, mut meta) = (Vec::new(), vec!["model_index.json".to_string()]);
    for c in comps {
        let names: Vec<&str> = files
            .iter()
            .filter_map(|f| f.path.strip_prefix(c)?.strip_prefix('/'))
            .filter(|n| !n.contains('/'))
            .collect();
        let st: Vec<&str> = names
            .iter()
            .copied()
            .filter(|n| {
                n.ends_with(".safetensors")
                    || (n.contains(".safetensors.index") && n.ends_with(".json"))
            })
            .collect();
        let is_weight = |n: &str| {
            [".bin", ".ckpt", ".pt", ".pth", ".msgpack", ".onnx"]
                .iter()
                .any(|e| n.ends_with(e))
        };
        if st.is_empty() && names.iter().any(|n| is_weight(n)) {
            return Err(format!(
                "diffusers component {c}/ has no safetensors weights"
            ));
        }
        meta.extend(
            names
                .iter()
                .filter(|n| {
                    !n.contains(".safetensors")
                        && [".json", ".txt", ".model"].iter().any(|e| n.ends_with(e))
                })
                .map(|n| format!("{c}/{n}")),
        );
        if let Some(best) = st.iter().filter_map(|n| diffusers_variant(n)).min() {
            let mut set: Vec<String> = st
                .iter()
                .filter(|n| diffusers_variant(n) == Some(best))
                .map(|n| format!("{c}/{n}"))
                .collect();
            // Index first: it is the primary of a sharded set.
            set.sort_by_key(|n| !n.ends_with(".json"));
            if matches!(c, "unet" | "transformer") {
                set.extend(weights);
                weights = set;
            } else {
                weights.extend(set);
            }
        }
    }
    weights.extend(meta);
    Ok(weights)
}

/// The weight file a diffusers component directory holds, preferring the `fp16` variant: a single
/// `*.safetensors`, or the `*.safetensors.index*.json` of a sharded set.
pub fn component_weights(dir: &Path) -> Result<PathBuf, String> {
    let mut st: Vec<String> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| {
            n.ends_with(".safetensors")
                || (n.contains(".safetensors.index") && n.ends_with(".json"))
        })
        .collect();
    let best = st
        .iter()
        .filter_map(|n| diffusers_variant(n))
        .min()
        .ok_or_else(|| format!("{}: no safetensors weights", dir.display()))?;
    st.retain(|n| diffusers_variant(n) == Some(best));
    st.sort_by_key(|n| !n.ends_with(".json"));
    Ok(dir.join(&st[0]))
}

/// Local snapshot listing (sizes from disk, up to 2 directory levels deep like `onnx/`);
/// in-flight `.tmp` downloads are skipped.
pub(crate) fn list_local_files(dir: &Path) -> Vec<RepoFile> {
    let mut out = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), 0)];
    while let Some((d, depth)) = stack.pop() {
        for entry in fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                if depth < 2 {
                    stack.push((p, depth + 1));
                }
            } else if let Ok(rel) = p.strip_prefix(dir) {
                let path = rel.to_string_lossy().replace('\\', "/");
                if !path.ends_with(".tmp") {
                    let size = fs::metadata(&p).ok().map(|m| m.len());
                    out.push(RepoFile { path, size });
                }
            }
        }
    }
    out
}

/// Downloader for pulling Hugging Face model repositories and files.
#[derive(Debug, Clone)]
pub struct HuggingFaceDownloader {
    pub storage: StorageBackend,
    pub auth_token: Option<String>,
}

impl HuggingFaceDownloader {
    /// Creates a new HuggingFaceDownloader.
    pub fn new(storage: StorageBackend, auth_token: Option<String>) -> Self {
        Self {
            storage,
            auth_token,
        }
    }

    /// Extracts (org, repo) from "org/repo" or ("", repo) if no slash. Rejects empty string.
    pub fn parse_hf_repo(model_id: &str) -> Option<(String, String)> {
        let trimmed = model_id.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some((org, repo)) = trimmed.split_once('/') {
            if repo.is_empty() {
                return None;
            }
            Some((org.to_string(), repo.to_string()))
        } else {
            Some((String::new(), trimmed.to_string()))
        }
    }
    /// Invokes system `curl -fSL --connect-timeout 15 --retry 2` writing to `{destination}.tmp`,
    /// then atomically renames to `{destination}`. While curl runs, `progress.done` advances by
    /// the bytes written (polled every 200 ms).
    pub fn download_file(
        url: &str,
        destination: &Path,
        auth_token: Option<&str>,
        progress: Option<&DownloadProgress>,
    ) -> Result<(), String> {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
        }

        let tmp_destination = {
            let mut s = destination.as_os_str().to_os_string();
            s.push(".tmp");
            PathBuf::from(s)
        };

        let _ = fs::remove_file(&tmp_destination);

        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-fsSL")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--retry")
            .arg("2");

        if let Some(token) = auth_token {
            if !token.is_empty() {
                cmd.arg("-H")
                    .arg(format!("Authorization: Bearer {}", token));
            }
        }

        cmd.arg("-o").arg(&tmp_destination);
        cmd.arg(url);
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to execute curl: {}", e))?;
        let base = progress.map_or(0, |p| p.done.load(AtomicOrdering::Relaxed));
        let written = || fs::metadata(&tmp_destination).map_or(0, |m| m.len());
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if let Some(p) = progress {
                        p.done.store(base + written(), AtomicOrdering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                Err(e) => return Err(format!("curl wait failed: {e}")),
            }
        };
        if !status.success() {
            let _ = fs::remove_file(&tmp_destination);
            let mut stderr = String::new();
            if let Some(mut s) = child.stderr.take() {
                let _ = s.read_to_string(&mut stderr);
            }
            if let Some(p) = progress {
                p.done.store(base, AtomicOrdering::Relaxed);
            }
            return Err(format!(
                "curl download failed for {} (exit code {:?}): {}",
                url,
                status.code(),
                stderr.trim()
            ));
        }
        if let Some(p) = progress {
            p.done.store(base + written(), AtomicOrdering::Relaxed);
        }

        fs::rename(&tmp_destination, destination).map_err(|e| {
            let _ = fs::remove_file(&tmp_destination);
            format!(
                "Failed to rename {:?} to {:?}: {}",
                tmp_destination, destination, e
            )
        })?;

        Ok(())
    }

    /// Downloads the weights `select_weight_files` picks (or the explicit `<org>/<repo>/<file>`)
    /// plus the listed metadata files into `target_dir`, keeping repo subdirectories. Files already
    /// present with the listed size are skipped. Progress: `total` = bytes still to fetch.
    pub fn download_model_with_quant(
        &self,
        model_id: &str,
        target_dir: &Path,
        explicit_quant: Option<&str>,
        progress: Option<&DownloadProgress>,
    ) -> Result<(), String> {
        let (repo_id, explicit_file) = split_weight_file(model_id);
        let (repo, rev) = split_revision(repo_id);
        let token = self.auth_token.as_deref();
        let files = list_repo_files(repo_id, token)?;
        let mut weights = match explicit_file {
            // One shard of a split GGUF names the whole set.
            Some(file) if files.iter().any(|f| f.path == file) => match shard_of(file, ".gguf") {
                Some(_) => gguf_split_set(&files, file)?
                    .into_iter()
                    .map(String::from)
                    .collect(),
                None => vec![file.to_string()],
            },
            Some(file) => return Err(format!("no file '{file}' in {repo_id}")),
            None => select_weight_files(&files, explicit_quant).map_err(|e| {
                // Repos that keep each quantization in a branch hold only a README on main.
                let branches = match rev {
                    "main" => list_branches(repo, token).unwrap_or_default(),
                    _ => Vec::new(),
                };
                match branches.is_empty() {
                    true => format!("{repo_id}: {e}"),
                    false => format!(
                        "{repo_id}: {e}; main has no weights, load a branch as {repo}@<branch>: {}",
                        branches.join(", ")
                    ),
                }
            })?,
        };
        // ponytail: only GGUF families with `runtime.mmproj` run the repo's vision/audio projector
        // beside the language model; other architectures' projectors are not fetched.
        let wants_mmproj = weights[0].ends_with(".gguf")
            && hub_json(&format!("https://huggingface.co/api/models/{repo}"), token)
                .ok()
                .and_then(|m| {
                    let a = m["gguf"]["architecture"].as_str()?;
                    crate::arch::registry().ok().map(|r| r.gguf_uses_mmproj(a))
                })
                .unwrap_or(false);
        if wants_mmproj {
            weights.extend(select_mmproj(&files, &weights[0]));
        }
        let mut pending: Vec<&RepoFile> = files
            .iter()
            .filter(|f| is_metadata_file(&f.path) || weights.contains(&f.path))
            .filter(|f| {
                let local = fs::metadata(target_dir.join(&f.path)).ok().map(|m| m.len());
                local.is_none() || (f.size.is_some() && local != f.size)
            })
            .collect();
        // The primary weight file goes last: once it is on disk the rest of the set is too, so an
        // interrupted download is never mistaken for a complete cache.
        pending.sort_by_key(|f| f.path == weights[0]);
        if let Some(p) = progress {
            p.done.store(0, AtomicOrdering::Relaxed);
            let total = pending.iter().filter_map(|f| f.size).sum();
            p.total.store(total, AtomicOrdering::Relaxed);
        }
        for f in pending {
            if let Some(p) = progress {
                *p.file.lock() = f.path.clone();
            }
            let url = format!(
                "https://huggingface.co/{repo}/resolve/{}/{}",
                rev_segment(rev),
                f.path
            );
            Self::download_file(&url, &target_dir.join(&f.path), token, progress)?;
        }

        // Sync to remote cloud if configured
        let _ = self.storage.sync_to_remote(target_dir, repo_id);
        Ok(())
    }
}

/// HuggingFace Hub cache manager, validator, and tiny model resolver.
#[derive(Debug, Clone)]
pub struct HuggingFaceHub {
    pub cache_root: PathBuf,
    pub max_param_budget: usize,
    pub storage: StorageBackend,
    pub explicit_quant: Option<String>,
    /// Live byte counts of this hub's downloads, when someone watches them.
    pub progress: Option<Arc<DownloadProgress>>,
}

impl Default for HuggingFaceHub {
    fn default() -> Self {
        Self::new()
    }
}

impl HuggingFaceHub {
    /// Creates a new hub instance resolving cache_root from environment or defaulting to `~/.cache/huggingface/hub`.
    pub fn new() -> Self {
        let cache_root = if let Ok(val) = std::env::var("HF_HUB_CACHE") {
            PathBuf::from(val)
        } else if let Ok(val) = std::env::var("HF_HOME") {
            PathBuf::from(val).join("hub")
        } else if let Ok(val) = std::env::var("HOME") {
            PathBuf::from(val)
                .join(".cache")
                .join("huggingface")
                .join("hub")
        } else {
            PathBuf::from(".cache/huggingface/hub")
        };

        let explicit_quant = std::env::var("GIBBS_QUANT")
            .or_else(|_| std::env::var("FOUNDRY_QUANT"))
            .ok();

        Self {
            cache_root: cache_root.clone(),
            max_param_budget: DEFAULT_MAX_PARAM_BUDGET,
            storage: StorageBackend::LocalDisk {
                cache_dir: cache_root,
            },
            explicit_quant,
            progress: None,
        }
    }
    /// Returns the default local cache directory.
    pub fn cache_dir() -> PathBuf {
        Self::new().cache_root
    }

    /// Creates a hub instance with an explicit cache directory.
    pub fn with_cache_dir(dir: PathBuf) -> Self {
        let explicit_quant = std::env::var("GIBBS_QUANT")
            .or_else(|_| std::env::var("FOUNDRY_QUANT"))
            .ok();
        Self {
            cache_root: dir.clone(),
            max_param_budget: DEFAULT_MAX_PARAM_BUDGET,
            storage: StorageBackend::LocalDisk { cache_dir: dir },
            explicit_quant,
            progress: None,
        }
    }

    /// Sets explicit quantization override (e.g. "q4_k_m", "q8_0", "nvfp4").
    pub fn with_quant(mut self, quant: impl Into<String>) -> Self {
        self.explicit_quant = Some(quant.into());
        self
    }

    /// Sets optional explicit quantization override.
    pub fn with_optional_quant(mut self, quant: Option<String>) -> Self {
        self.explicit_quant = quant;
        self
    }

    /// Configures the storage backend.
    pub fn with_storage(mut self, storage: StorageBackend) -> Self {
        if let StorageBackend::LocalDisk { cache_dir } = &storage {
            self.cache_root = cache_dir.clone();
        }
        self.storage = storage;
        self
    }

    /// Automatically checks cache, downloads from Hugging Face if missing, or synthesizes a tiny model.
    pub fn auto_download_and_cache(
        &self,
        model_id: &str,
        auth_token: Option<&str>,
    ) -> Result<ModelArtifacts, String> {
        self.auto_download_and_cache_with_quant(
            model_id,
            auth_token,
            self.explicit_quant.as_deref(),
        )
    }

    /// Automatically checks cache, downloads from Hugging Face with explicit quantization filter if missing,
    /// or synthesizes a tiny model.
    pub fn auto_download_and_cache_with_quant(
        &self,
        model_id: &str,
        auth_token: Option<&str>,
        explicit_quant: Option<&str>,
    ) -> Result<ModelArtifacts, String> {
        // Explicit selection (file or quant) is enforced by locate: a cached file only counts when it matches.
        let (repo_id, _) = split_weight_file(model_id);
        let located = self.locate_model_with_quant(model_id, explicit_quant);
        let real_repo = !is_synthetic_model_id(repo_id);

        // A real repository is only usable with real weights; a lone config.json (e.g. a stale
        // synthetic directory) must never be executed with random initialization.
        if located.is_cached && (!real_repo || located.weights_path.is_some()) {
            return Ok(located);
        }
        if !real_repo {
            return self.resolve_or_synthesize_tiny(model_id);
        }

        let (repo_id, rev) = split_revision(repo_id);
        let (org, repo) = HuggingFaceDownloader::parse_hf_repo(repo_id)
            .filter(|(org, _)| !org.is_empty())
            .ok_or_else(|| {
                format!("Model '{model_id}' not found locally and is not an <org>/<repo> id")
            })?;
        let snapshot_dir = self
            .cache_root
            .join(format!("models--{}--{}", org, repo))
            .join("snapshots")
            .join(rev_dir(rev));
        let downloaded =
            HuggingFaceDownloader::new(self.storage.clone(), auth_token.map(|s| s.to_string()))
                .download_model_with_quant(
                    model_id,
                    &snapshot_dir,
                    explicit_quant,
                    self.progress.as_deref(),
                );
        if let Err(e) = downloaded {
            // Drop the directories the attempt created if nothing landed (remove_dir only removes empty dirs),
            // so a failed id never shows up as a cached model.
            for dir in snapshot_dir.ancestors().take(3) {
                let _ = fs::remove_dir(dir);
            }
            // curl -f reports a gate (or a missing token) as an HTTP 401/403.
            let denied = e.contains("returned error: 401") || e.contains("returned error: 403");
            return Err(match denied {
                true => format!(
                    "{e} (access denied: if {repo_id} is gated, accept its license at https://huggingface.co/{repo_id} with the account that owns HF_TOKEN)"
                ),
                false => format!(
                    "{e} (check the repo id exists on huggingface.co and HF_TOKEN for gated repos)"
                ),
            });
        }

        let relocated = self.locate_model_with_quant(model_id, explicit_quant);
        if relocated.weights_path.is_none() {
            return Err(format!(
                "Model '{model_id}': no weights the loader reads matching the request found in {}",
                snapshot_dir.display()
            ));
        }
        Ok(relocated)
    }

    /// Sets the maximum parameter budget ceiling.
    pub fn with_budget(mut self, max_params: usize) -> Self {
        self.max_param_budget = max_params;
        self
    }

    /// Downloads a diffusers pipeline repo's components (the set [`select_weight_files`] picks)
    /// except the `skip` component dirs (e.g. `["transformer"]` when a GGUF brings its own) into
    /// its snapshot dir and returns that dir. Offline with a cached `model_index.json`: the dir as is.
    pub fn fetch_diffusers(&self, repo: &str, skip: &[&str]) -> Result<PathBuf, String> {
        let dir = self
            .cache_root
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join("main");
        let token = std::env::var("HF_TOKEN").ok();
        let files = match list_repo_files(repo, token.as_deref()) {
            Ok(f) => f,
            Err(_) if dir.join("model_index.json").is_file() => return Ok(dir),
            Err(e) => return Err(e),
        };
        let wanted = diffusers_files(&files).map_err(|e| format!("{repo}: {e}"))?;
        let pending: Vec<&RepoFile> = files
            .iter()
            .filter(|f| wanted.contains(&f.path))
            .filter(|f| !skip.iter().any(|s| f.path.starts_with(&format!("{s}/"))))
            .filter(|f| {
                let local = fs::metadata(dir.join(&f.path)).ok().map(|m| m.len());
                local.is_none() || (f.size.is_some() && local != f.size)
            })
            .collect();
        if let Some(p) = self.progress.as_deref() {
            p.done.store(0, AtomicOrdering::Relaxed);
            p.total.store(
                pending.iter().filter_map(|f| f.size).sum(),
                AtomicOrdering::Relaxed,
            );
        }
        for f in pending {
            let url = format!("https://huggingface.co/{repo}/resolve/main/{}", f.path);
            HuggingFaceDownloader::download_file(
                &url,
                &dir.join(&f.path),
                token.as_deref(),
                self.progress.as_deref(),
            )?;
        }
        Ok(dir)
    }

    /// Fetches one non-weight file of `repo_id` (e.g. `tokenizer.json`, `<org>/<repo>@<revision>`) into
    /// `<cache>/models--<org>--<repo>/snapshots/<revision>/<filename>`. A cached file is returned without network.
    pub fn fetch_file(&self, repo_id: &str, filename: &str) -> Result<PathBuf, String> {
        let (repo_id, rev) = split_revision(repo_id);
        let path = self
            .cache_root
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("snapshots")
            .join(rev_dir(rev))
            .join(filename);
        if path.is_file() {
            return Ok(path);
        }
        let url = format!(
            "https://huggingface.co/{repo_id}/resolve/{}/{filename}",
            rev_segment(rev)
        );
        let token = std::env::var("HF_TOKEN").ok();
        HuggingFaceDownloader::download_file(
            &url,
            &path,
            token.as_deref(),
            self.progress.as_deref(),
        )?;
        Ok(path)
    }
    /// Inspects local filesystem and cache_root to locate model artifacts.
    ///
    /// Checks:
    /// 1. Direct local directory matching `model_id`.
    /// 2. `cache_root/models--<org>--<repo>`.
    /// 3. `cache_root/foundry--<name>`.
    ///
    /// If directory is found, inspects `snapshots/` folder for the newest commit hash.
    /// Inspects candidate directory for `config.json`, `tokenizer.json`, `model.safetensors`, and `*.gguf`.
    pub fn locate_model(&self, model_id: &str) -> ModelArtifacts {
        self.locate_model_with_quant(model_id, self.explicit_quant.as_deref())
    }

    /// Inspects local filesystem and cache_root to locate model artifacts with optional explicit quantization override.
    /// `<org>/<repo>/<file>` locates `<file>` in the `<org>/<repo>` snapshot; weights are `None` unless it is cached.
    pub fn locate_model_with_quant(
        &self,
        model_id: &str,
        explicit_quant: Option<&str>,
    ) -> ModelArtifacts {
        let (repo_id, explicit_file) = split_weight_file(model_id);
        // `<org>/<repo>@<revision>` reads exactly `snapshots/<revision>`; a bare id the newest snapshot.
        let (repo, rev) = split_revision(repo_id);
        let candidate_dir = self.find_candidate_dir(repo);

        if let Some(base_dir) = candidate_dir {
            let target_dir = match repo.len() < repo_id.len() {
                true => base_dir.join("snapshots").join(rev_dir(rev)),
                false => self.resolve_snapshot_dir(&base_dir),
            };

            // Locate config.json
            let config_path = {
                let p = target_dir.join("config.json");
                if p.is_file() {
                    Some(p)
                } else {
                    None
                }
            };

            // Locate tokenizer.json or tokenizer.model
            let tokenizer_path = {
                let p_json = target_dir.join("tokenizer.json");
                if p_json.is_file() {
                    Some(p_json)
                } else {
                    let p_model = target_dir.join("tokenizer.model");
                    if p_model.is_file() {
                        Some(p_model)
                    } else {
                        None
                    }
                }
            };

            // Locate weights: the explicit file, else the same selection a download would make.
            let selection = match explicit_file {
                Some(file) => vec![file.to_string()],
                None => self.local_selection(&target_dir, explicit_quant),
            };
            let weights_path = selection
                .first()
                .map(|f| target_dir.join(f))
                .filter(|p| p.is_file());
            let quant_format = weights_path.as_ref().and_then(|_| {
                selection
                    .iter()
                    .filter(|f| f.ends_with(".gguf") || f.ends_with(".safetensors"))
                    .find_map(|f| detect_quant_format(Path::new(f)))
            });

            let is_cached =
                config_path.is_some() || weights_path.is_some() || tokenizer_path.is_some();

            let estimated_params = config_path
                .as_deref()
                .and_then(estimate_params_from_config)
                .or_else(|| {
                    weights_path
                        .as_deref()
                        .and_then(estimate_params_from_weights)
                })
                .unwrap_or_else(|| estimate_params_from_id(model_id));

            ModelArtifacts {
                model_id: model_id.to_string(),
                local_dir: Some(target_dir),
                config_path,
                tokenizer_path,
                weights_path,
                estimated_params,
                is_cached,
                quant_format,
            }
        } else {
            let estimated_params = estimate_params_from_id(model_id);
            ModelArtifacts {
                model_id: model_id.to_string(),
                local_dir: None,
                config_path: None,
                tokenizer_path: None,
                weights_path: None,
                estimated_params,
                is_cached: false,
                quant_format: None,
            }
        }
    }

    /// Enforces the model parameter budget safeguard.
    ///
    /// If estimated_params > max_param_budget, returns an informative error
    /// explaining the budget safeguard and suggesting tiny models.
    pub fn validate_budget(&self, artifacts: &ModelArtifacts) -> Result<(), String> {
        if artifacts.estimated_params > self.max_param_budget {
            Err(format!(
                "Model '{}' exceeds parameter budget: estimated {} params exceeds maximum budget of {} params (10M ceiling safeguard). Suggested tiny models: 'gibbs/tiny-5m', 'gibbs/tiny-10m', 'gibbs/jev-decision-3m', 'foundry/tiny-5m'",
                artifacts.model_id, artifacts.estimated_params, self.max_param_budget
            ))
        } else {
            Ok(())
        }
    }

    /// Resolves a local model within budget, or synthesizes a cached tiny model directory.
    ///
    /// If local model is found and within budget, returns it.
    /// If not found or if model_id is a known/synthetic model ID, creates a cached tiny model
    /// directory under `cache_root/foundry--<name>` with synthetic `config.json` and returns it.
    pub fn resolve_or_synthesize_tiny(&self, model_id: &str) -> Result<ModelArtifacts, String> {
        let artifacts = self.locate_model(model_id);

        if artifacts.is_cached {
            self.validate_budget(&artifacts)?;
            return Ok(artifacts);
        }

        // If not cached, first enforce parameter budget on known non-tiny model names
        if artifacts.estimated_params > self.max_param_budget {
            return Err(format!(
                "Model '{}' exceeds parameter budget: estimated {} params exceeds maximum budget of {} params (10M ceiling safeguard). Suggested tiny models: 'gibbs/tiny-5m', 'gibbs/tiny-10m', 'gibbs/jev-decision-3m', 'foundry/tiny-5m'",
                artifacts.model_id, artifacts.estimated_params, self.max_param_budget
            ));
        }

        // Create cached tiny model directory under cache_root/gibbs--<name> (or backwards-compatible foundry--<name>)
        let clean_name = model_id
            .trim_start_matches("gibbs/")
            .trim_start_matches("gibbs--")
            .trim_start_matches("foundry/")
            .trim_start_matches("foundry--")
            .replace('/', "--");
        let gibbs_dir = self.cache_root.join(format!("gibbs--{}", clean_name));
        let foundry_dir = self.cache_root.join(format!("foundry--{}", clean_name));
        let tiny_dir = if foundry_dir.is_dir() {
            foundry_dir
        } else {
            gibbs_dir
        };
        fs::create_dir_all(&tiny_dir).map_err(|e| {
            format!(
                "Failed to create synthetic model directory '{}': {}",
                tiny_dir.display(),
                e
            )
        })?;

        let config_path = tiny_dir.join("config.json");
        if !config_path.exists() {
            let rt = crate::arch::registry()
                .ok()
                .and_then(|r| r.fallback_runtime())
                .unwrap_or_default();
            let preset = rt["synthetic_presets"]
                .as_array()
                .into_iter()
                .flatten()
                .find_map(|p| {
                    let pat = p[0].as_str()?;
                    crate::arch::glob(pat, &clean_name).then(|| p[1].clone())
                })
                .unwrap_or_default();
            let u = |i: usize, d: usize| preset[i].as_u64().map_or(d, |n| n as usize);
            let (
                num_params,
                hidden_dim,
                num_layers,
                num_heads,
                num_kv_heads,
                intermediate_dim,
                vocab_size,
            ) = (
                u(0, 1_000_000),
                u(1, 128),
                u(2, 2),
                u(3, 4),
                u(4, 2),
                u(5, 256),
                u(6, 512),
            );
            let model_type = preset[7].as_str().unwrap_or("default");
            let synth_arch = rt["synthetic_architecture"].as_str().unwrap_or_default();
            // Every synthetic id runs the fallback decoder; `model_type` only labels the size preset.
            let config_json = serde_json::json!({
                "architectures": [synth_arch],
                "model_type": model_type,
                "num_parameters": num_params,
                "hidden_size": hidden_dim,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "intermediate_size": intermediate_dim,
                "vocab_size": vocab_size,
                "max_position_embeddings": 2048,
                "torch_dtype": "float32",
                "bos_token_id": 1,
                "eos_token_id": 2,
                "rms_norm_eps": 1e-6
            });
            let config_str = serde_json::to_string_pretty(&config_json)
                .map_err(|e| format!("Failed to serialize synthetic config: {}", e))?;
            fs::write(&config_path, config_str).map_err(|e| {
                format!(
                    "Failed to write synthetic config.json to '{}': {}",
                    config_path.display(),
                    e
                )
            })?;
        }

        // Re-locate model from newly populated synthetic cache directory
        let synthesized = self.locate_model(model_id);
        self.validate_budget(&synthesized)?;
        Ok(synthesized)
    }

    /// Searches for a model directory candidate across standard HuggingFace Hub layouts.
    ///
    /// Checks:
    /// 1. Direct path check (as-is directory).
    /// 2. Relative to `cache_root`.
    /// 3. HuggingFace Hub format: `models--<org>--<repo>` or `models--<repo>`.
    /// 4. Foundry tiny model format: `foundry--<name>`.
    /// 5. Slash to double-dash format: `<org>--<repo>`.
    pub fn find_candidate(&self, model_id: &str) -> Option<PathBuf> {
        self.find_candidate_dir(model_id)
    }

    /// Searches for a model directory candidate across standard HuggingFace Hub layouts.
    pub fn find_candidate_dir(&self, model_id: &str) -> Option<PathBuf> {
        // 1. Direct path check
        let direct_path = PathBuf::from(model_id);
        if direct_path.is_dir() {
            return Some(direct_path);
        }

        // 2. Relative to cache_root as-is
        let as_is = self.cache_root.join(model_id);
        if as_is.is_dir() {
            return Some(as_is);
        }

        // 3. HuggingFace Hub format: models--<org>--<repo> or models--<repo>
        let hf_name = if model_id.starts_with("models--") {
            model_id.to_string()
        } else {
            format!("models--{}", model_id.replace('/', "--"))
        };
        let hf_path = self.cache_root.join(&hf_name);
        if hf_path.is_dir() {
            return Some(hf_path);
        }

        // 4-5. Engine-generated synthetic dirs (gibbs--<name>, legacy foundry--<name>) only serve synthetic
        // ids; a real <org>/<repo> must never resolve to a leftover random-weight config.
        let clean_name = model_id
            .trim_start_matches("gibbs/")
            .trim_start_matches("gibbs--")
            .trim_start_matches("foundry/")
            .trim_start_matches("foundry--")
            .replace('/', "--");
        if is_synthetic_model_id(model_id) {
            for prefix in ["gibbs--", "foundry--"] {
                let p = self.cache_root.join(format!("{prefix}{clean_name}"));
                if p.is_dir() {
                    return Some(p);
                }
            }
        }
        let dash_path = self.cache_root.join(model_id.replace('/', "--"));
        if dash_path.is_dir() {
            return Some(dash_path);
        }

        None
    }

    /// Inspects `snapshots/` folder for the newest commit hash subdirectory.
    /// Falls back to `base_dir` if no snapshots directory or valid snapshot subdirectories exist.
    fn resolve_snapshot_dir(&self, base_dir: &Path) -> PathBuf {
        let snapshots_dir = base_dir.join("snapshots");
        if !snapshots_dir.is_dir() {
            return base_dir.to_path_buf();
        }
        // `main` (what a bare id downloads) wins over branch snapshots.
        if snapshots_dir.join("main").is_dir() {
            return snapshots_dir.join("main");
        }

        let mut snapshots: Vec<(Option<std::time::SystemTime>, PathBuf)> = Vec::new();
        if let Ok(entries) = fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let mtime = entry.metadata().and_then(|m| m.modified()).ok();
                    snapshots.push((mtime, p));
                }
            }
        }

        if snapshots.is_empty() {
            return base_dir.to_path_buf();
        }

        // Sort descending: newest mtime first, or reverse path name if mtime unavailable
        snapshots.sort_by(|a, b| match (b.0, a.0) {
            (Some(tb), Some(ta)) => tb.cmp(&ta),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => b.1.cmp(&a.1),
        });

        snapshots
            .into_iter()
            .next()
            .map(|(_, p)| p)
            .unwrap_or_else(|| base_dir.to_path_buf())
    }

    /// Primary weight file the loader reads in `target_dir`, chosen exactly like a download
    /// (`select_weight_files` over the local snapshot).
    pub fn locate_weights(&self, target_dir: &Path) -> Option<PathBuf> {
        self.locate_weights_with_quant(target_dir, self.explicit_quant.as_deref())
    }

    /// `locate_weights` under an explicit quant: only a matching set counts (never a fallback).
    pub fn locate_weights_with_quant(
        &self,
        target_dir: &Path,
        explicit_quant: Option<&str>,
    ) -> Option<PathBuf> {
        let selection = self.local_selection(target_dir, explicit_quant);
        selection.first().map(|f| target_dir.join(f))
    }

    /// Local `select_weight_files`; a sharded set also needs every shard its index names on disk.
    fn local_selection(&self, target_dir: &Path, explicit_quant: Option<&str>) -> Vec<String> {
        let selection =
            select_weight_files(&list_local_files(target_dir), explicit_quant).unwrap_or_default();
        let complete = match selection.first() {
            Some(index) if index.ends_with(".safetensors.index.json") => {
                index_shards(&target_dir.join(index))
                    .is_some_and(|shards| shards.iter().all(|s| target_dir.join(s).is_file()))
            }
            _ => true,
        };
        if complete {
            selection
        } else {
            Vec::new()
        }
    }
}

/// Parses and estimates parameter count from a `config.json` file.
fn estimate_params_from_config(config_path: &Path) -> Option<usize> {
    let data = fs::read_to_string(config_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;

    // Check if explicitly given in config
    if let Some(n) = v
        .get("num_parameters")
        .or_else(|| v.get("n_params"))
        .or_else(|| v.get("params"))
    {
        if let Some(n) = n.as_u64() {
            return Some(n as usize);
        }
    }

    // Multimodal configs keep the language model's dims under `text_config`.
    let v = if v["text_config"].is_object() {
        &v["text_config"]
    } else {
        &v
    };
    let num = |keys: &[&str]| keys.iter().find_map(|k| v[*k].as_u64()).map(|n| n as usize);
    // DistilBERT names the model width `dim` and the FFN width `hidden_dim`.
    let hidden_size = num(&["hidden_size", "d_model", "n_embd", "dim", "hidden_dim"]).unwrap_or(0);
    let num_layers = num(&["num_hidden_layers", "n_layers", "num_layers", "n_layer"]).unwrap_or(0);

    if hidden_size == 0 || num_layers == 0 {
        return None;
    }

    let vocab_size = num(&["vocab_size"]).unwrap_or(512);

    let intermediate_size = num(&["intermediate_size", "n_inner", "ffn_dim"])
        .or_else(|| v.get("dim").and(num(&["hidden_dim"])))
        .or_else(|| num(&["intermediate_dim"]))
        .unwrap_or((hidden_size * 8) / 3);

    // Embeddings: input, plus lm_head unless tied. PretrainedConfig defaults to tied; families
    // in `runtime.untied_default_types` default to untied.
    let untied_default = v["model_type"].as_str().is_some_and(|mt| {
        crate::arch::registry()
            .ok()
            .and_then(|r| r.fallback_runtime())
            .and_then(|rt| {
                rt["untied_default_types"]
                    .as_array()?
                    .iter()
                    .find_map(|t| (t.as_str() == Some(mt)).then_some(true))
            })
            .unwrap_or(false)
    });
    let tied = v["tie_word_embeddings"]
        .as_bool()
        .unwrap_or(!untied_default);
    let embedding_params = if tied { 1 } else { 2 } * vocab_size * hidden_size;

    // Per-layer transformer params:
    // Attention: Q/O are hidden × heads·head_dim, K/V hidden × kv_heads·head_dim (`head_dim` may be
    // explicit in config and need not equal hidden / heads).
    // MLP (SwiGLU): gate + up + down -> 3 * hidden * intermediate
    // Layer norms: ~2 * hidden
    let heads = num(&["num_attention_heads", "n_heads", "n_head"])
        .unwrap_or(1)
        .max(1);
    let kv_heads = v["num_key_value_heads"]
        .as_u64()
        .map_or(heads, |k| k as usize);
    let head_dim = v["head_dim"]
        .as_u64()
        .map_or(hidden_size / heads, |d| d as usize);
    let layer_attn = 2 * hidden_size * (heads * head_dim) + 2 * hidden_size * (kv_heads * head_dim);
    let layer_mlp = 3 * hidden_size * intermediate_size;
    let layer_norm = 2 * hidden_size;
    let layer_params = layer_attn + layer_mlp + layer_norm;

    let total = embedding_params + num_layers * layer_params + hidden_size;
    Some(total)
}

/// Distinct shard files a `*.safetensors.index.json` names in its `weight_map`.
fn index_shards(index_path: &Path) -> Option<std::collections::BTreeSet<String>> {
    let index: serde_json::Value = serde_json::from_slice(&fs::read(index_path).ok()?).ok()?;
    Some(
        index["weight_map"]
            .as_object()?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
    )
}

/// Estimates parameter count from safetensors header(s) without loading tensor data; a sharded
/// `*.safetensors.index.json` sums the headers of every shard its `weight_map` names.
fn estimate_params_from_weights(weights_path: &Path) -> Option<usize> {
    if weights_path
        .to_string_lossy()
        .ends_with(".safetensors.index.json")
    {
        let dir = weights_path.parent()?;
        return index_shards(weights_path)?
            .iter()
            .map(|s| estimate_params_from_weights(&dir.join(s)))
            .sum();
    }
    if let Some(ext) = weights_path.extension() {
        if ext == "safetensors" {
            if let Ok(mut file) = fs::File::open(weights_path) {
                let mut len_buf = [0u8; 8];
                if file.read_exact(&mut len_buf).is_ok() {
                    let header_len = u64::from_le_bytes(len_buf) as usize;
                    if header_len > 0 && header_len <= 50 * 1024 * 1024 {
                        let mut header_buf = vec![0u8; header_len];
                        if file.read_exact(&mut header_buf).is_ok() {
                            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&header_buf)
                            {
                                if let serde_json::Value::Object(map) = v {
                                    let mut total_params = 0usize;
                                    for (k, val) in map {
                                        if k == "__metadata__" {
                                            continue;
                                        }
                                        if let Some(shape) =
                                            val.get("shape").and_then(|s| s.as_array())
                                        {
                                            let tensor_elems: usize = shape
                                                .iter()
                                                .filter_map(|d| d.as_u64().map(|x| x as usize))
                                                .product();
                                            total_params += tensor_elems;
                                        }
                                    }
                                    if total_params > 0 {
                                        return Some(total_params);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Ids that name engine-generated random-weight models (kernel benchmarking only):
/// `gibbs/…` / `foundry/…` (and `--` forms). Everything else is a real model and needs real weights.
pub fn is_synthetic_model_id(model_id: &str) -> bool {
    ["gibbs/", "gibbs--", "foundry/", "foundry--"]
        .iter()
        .any(|p| model_id.starts_with(p))
}

/// Heuristic parameter estimation from model identifier string.
fn estimate_params_from_id(model_id: &str) -> usize {
    let lower = model_id.to_lowercase();

    if let Some(params) = crate::arch::registry()
        .ok()
        .and_then(|r| r.fallback_runtime())
        .and_then(|rt| {
            rt["id_param_hints"]
                .as_array()?
                .iter()
                .find_map(|h| {
                    let pat = h[0].as_str()?;
                    crate::arch::glob(pat, &lower).then(|| h[1].as_u64()).flatten()
                })
        })
    {
        return params as usize;
    }
    // Parse heuristics like "10m", "5m", "3m", "7b", "70b", "8b", "1.5b", "0.5b"
    for part in lower.split(|c: char| !c.is_alphanumeric() && c != '.') {
        if let Some(m_str) = part.strip_suffix('m') {
            if let Ok(val) = m_str.parse::<f64>() {
                return (val * 1_000_000.0) as usize;
            }
        }
        if let Some(b_str) = part.strip_suffix('b') {
            if let Ok(val) = b_str.parse::<f64>() {
                return (val * 1_000_000_000.0) as usize;
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_thread_safety() {
        assert_send_sync::<ModelArtifacts>();
        assert_send_sync::<StorageBackend>();
        assert_send_sync::<HuggingFaceDownloader>();
        assert_send_sync::<HuggingFaceHub>();
    }

    #[test]
    fn test_hub_default_and_custom_cache() {
        let hub = HuggingFaceHub::new();
        assert_eq!(hub.max_param_budget, 10_000_000);

        let custom_dir = PathBuf::from("/tmp/custom_hf_cache");
        let hub_custom = HuggingFaceHub::with_cache_dir(custom_dir.clone());
        assert_eq!(hub_custom.cache_root, custom_dir);
        assert_eq!(hub_custom.max_param_budget, 10_000_000);

        let hub_budget = hub_custom.with_budget(5_000_000);
        assert_eq!(hub_budget.max_param_budget, 5_000_000);
    }

    #[test]
    fn test_fetch_file_cache_hit_skips_network() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        let cached = dir
            .path()
            .join("models--openai--clip-vit-large-patch14/snapshots/main/tokenizer.json");
        fs::create_dir_all(cached.parent().unwrap()).unwrap();
        fs::write(&cached, "{}").unwrap();
        assert_eq!(
            hub.fetch_file("openai/clip-vit-large-patch14", "tokenizer.json")
                .unwrap(),
            cached
        );
    }

    #[test]
    fn test_locate_model_not_cached() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        let artifacts = hub.locate_model("foundry/tiny-llama-5m");

        assert_eq!(artifacts.model_id, "foundry/tiny-llama-5m");
        assert!(!artifacts.is_cached);
        assert!(artifacts.local_dir.is_none());
        assert!(artifacts.config_path.is_none());
        assert!(artifacts.tokenizer_path.is_none());
        assert!(artifacts.weights_path.is_none());
        assert_eq!(artifacts.estimated_params, 5_000_000);
    }

    #[test]
    fn test_locate_model_hf_snapshots() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());

        // Create standard HF cache structure: models--org--repo/snapshots/<hash>/...
        let model_dir = dir.path().join("models--meta-llama--Llama-2-7b");
        let snap1 = model_dir.join("snapshots").join("hash111");
        let snap2 = model_dir.join("snapshots").join("hash222");

        fs::create_dir_all(&snap1).unwrap();
        fs::create_dir_all(&snap2).unwrap();

        // Write artifacts into snap2
        fs::write(
            snap2.join("config.json"),
            r#"{"hidden_size": 128, "num_hidden_layers": 2}"#,
        )
        .unwrap();
        fs::write(snap2.join("tokenizer.json"), "{}").unwrap();
        fs::write(snap2.join("model.safetensors"), "dummy weights").unwrap();

        let artifacts = hub.locate_model("meta-llama/Llama-2-7b");
        assert!(artifacts.is_cached);
        assert!(artifacts.local_dir.is_some());
        assert!(artifacts.config_path.is_some());
        assert!(artifacts.tokenizer_path.is_some());
        assert!(artifacts.weights_path.is_some());
    }

    #[test]
    fn test_validate_budget() {
        let hub = HuggingFaceHub::new();

        let tiny_artifacts = ModelArtifacts {
            model_id: "foundry/tiny-llama-5m".to_string(),
            estimated_params: 5_000_000,
            ..Default::default()
        };
        assert!(hub.validate_budget(&tiny_artifacts).is_ok());

        let big_artifacts = ModelArtifacts {
            model_id: "meta-llama/Llama-3-70B".to_string(),
            estimated_params: 70_000_000_000,
            ..Default::default()
        };
        let err = hub.validate_budget(&big_artifacts).unwrap_err();
        assert!(err.contains("exceeds parameter budget"));
        assert!(err.contains("10M ceiling safeguard"));
        assert!(err.contains("foundry/tiny-5m") || err.contains("foundry/tiny-llama-5m"));
    }

    /// `<repo>@<branch>` loads exactly that branch's snapshot; a bare id prefers `main`.
    #[test]
    fn revision_ids_resolve_their_own_snapshot() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        let snaps = dir.path().join("models--t--m-exl3/snapshots");
        for rev in ["main", "4.0bpw", "2.5bpw"] {
            fs::create_dir_all(snaps.join(rev)).unwrap();
            fs::write(snaps.join(rev).join("model.safetensors"), rev).unwrap();
        }
        let weights = |id: &str| hub.locate_model(id).weights_path;
        assert_eq!(
            weights("t/m-exl3@4.0bpw"),
            Some(snaps.join("4.0bpw/model.safetensors"))
        );
        assert_eq!(
            weights("t/m-exl3@2.5bpw"),
            Some(snaps.join("2.5bpw/model.safetensors"))
        );
        assert_eq!(
            weights("t/m-exl3"),
            Some(snaps.join("main/model.safetensors"))
        );
        assert!(!hub.locate_model("t/m-exl3@3.0bpw").is_cached);
    }

    #[test]
    fn test_real_repo_id_never_resolves_to_stale_synthetic_dir() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        // Leftovers of the old fallback that synthesized random-weight configs for real repo ids.
        for stale in ["gibbs--org--repo", "foundry--org--repo"] {
            let p = dir.path().join(stale);
            fs::create_dir_all(&p).unwrap();
            fs::write(p.join("config.json"), r#"{"hidden_size":256}"#).unwrap();
        }
        let located = hub.locate_model("org/repo");
        assert!(located.local_dir.is_none(), "{:?}", located.local_dir);
        assert!(!located.is_cached);
        // Synthetic ids still find their own directories; bare names are not synthetic.
        assert!(hub.locate_model("gibbs/org/repo").is_cached);
        assert!(!is_synthetic_model_id("TinyStories-Instruct-8M"));
    }

    #[test]
    fn test_resolve_or_synthesize_tiny() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());

        // Synthesize tiny-llama-5m
        let res = hub.resolve_or_synthesize_tiny("foundry/tiny-llama-5m");
        assert!(res.is_ok());
        let artifacts = res.unwrap();
        assert!(artifacts.is_cached);
        assert!(artifacts.local_dir.is_some());
        assert!(artifacts.config_path.is_some());
        assert_eq!(artifacts.estimated_params, 5_000_000);

        // Second call should locate existing cached model
        let res2 = hub.resolve_or_synthesize_tiny("foundry/tiny-llama-5m");
        assert!(res2.is_ok());
        let artifacts2 = res2.unwrap();
        assert_eq!(artifacts2.local_dir, artifacts.local_dir);

        // Synthesize smollm-10m
        let res_smol = hub
            .resolve_or_synthesize_tiny("foundry/smollm-10m")
            .unwrap();
        assert_eq!(res_smol.estimated_params, 10_000_000);

        // Synthesize jev-decision-3m
        let res_jev = hub
            .resolve_or_synthesize_tiny("foundry/jev-decision-3m")
            .unwrap();
        assert_eq!(res_jev.estimated_params, 3_000_000);

        // Reject model exceeding budget
        let res_big = hub.resolve_or_synthesize_tiny("meta-llama/Llama-3-70B");
        assert!(res_big.is_err());
        let err = res_big.unwrap_err();
        assert!(err.contains("exceeds parameter budget"));
    }

    #[test]
    fn test_storage_backend_from_uri() {
        let s3 = StorageBackend::from_uri("s3://my-bucket/models/llama").unwrap();
        assert_eq!(
            s3,
            StorageBackend::S3 {
                bucket: "my-bucket".to_string(),
                prefix: "models/llama".to_string(),
                endpoint: None,
            }
        );

        let s3_root = StorageBackend::from_uri("s3://my-bucket").unwrap();
        assert_eq!(
            s3_root,
            StorageBackend::S3 {
                bucket: "my-bucket".to_string(),
                prefix: "".to_string(),
                endpoint: None,
            }
        );

        let gs = StorageBackend::from_uri("gs://gcs-bucket/checkpoints").unwrap();
        assert_eq!(
            gs,
            StorageBackend::Gcs {
                bucket: "gcs-bucket".to_string(),
                prefix: "checkpoints".to_string(),
            }
        );

        let gcs = StorageBackend::from_uri("gcs://gcs-bucket/checkpoints/v1").unwrap();
        assert_eq!(
            gcs,
            StorageBackend::Gcs {
                bucket: "gcs-bucket".to_string(),
                prefix: "checkpoints/v1".to_string(),
            }
        );

        let local = StorageBackend::from_uri("/var/cache/models").unwrap();
        assert_eq!(
            local,
            StorageBackend::LocalDisk {
                cache_dir: PathBuf::from("/var/cache/models"),
            }
        );

        let local_file = StorageBackend::from_uri("file:///tmp/cache").unwrap();
        assert_eq!(
            local_file,
            StorageBackend::LocalDisk {
                cache_dir: PathBuf::from("/tmp/cache"),
            }
        );

        assert!(StorageBackend::from_uri("s3:///prefix").is_err());
        assert!(StorageBackend::from_uri("gs:///prefix").is_err());
    }

    #[test]
    fn test_parse_hf_repo() {
        assert_eq!(
            HuggingFaceDownloader::parse_hf_repo("unsloth/SmolLM2-135M-Instruct-GGUF"),
            Some((
                "unsloth".to_string(),
                "SmolLM2-135M-Instruct-GGUF".to_string()
            ))
        );

        assert_eq!(
            HuggingFaceDownloader::parse_hf_repo("roneneldan/TinyStories-1M"),
            Some(("roneneldan".to_string(), "TinyStories-1M".to_string()))
        );

        assert_eq!(
            HuggingFaceDownloader::parse_hf_repo("TinyStories-1M"),
            Some(("".to_string(), "TinyStories-1M".to_string()))
        );

        assert_eq!(
            HuggingFaceDownloader::parse_hf_repo("/repo"),
            Some(("".to_string(), "repo".to_string()))
        );

        assert_eq!(HuggingFaceDownloader::parse_hf_repo(""), None);
        assert_eq!(HuggingFaceDownloader::parse_hf_repo("   "), None);
        assert_eq!(HuggingFaceDownloader::parse_hf_repo("org/"), None);
    }

    #[test]
    fn test_auto_download_and_cache_checks_cache_first() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());

        // Pre-populate cache for unsloth/SmolLM2-135M-Instruct-GGUF
        let model_dir = dir
            .path()
            .join("models--unsloth--SmolLM2-135M-Instruct-GGUF")
            .join("snapshots")
            .join("main");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"hidden_size": 256, "num_hidden_layers": 4}"#,
        )
        .unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), "cached weights").unwrap();

        // auto_download_and_cache must hit cache immediately without downloading
        let res = hub.auto_download_and_cache("unsloth/SmolLM2-135M-Instruct-GGUF", None);
        assert!(res.is_ok());
        let artifacts = res.unwrap();
        assert!(artifacts.is_cached);
        assert_eq!(
            artifacts.weights_path,
            Some(model_dir.join("model.safetensors"))
        );
    }

    #[test]
    fn test_auto_download_and_cache_fallback_synthetic() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());

        // Synthetic model not in cache - download will fail (offline/synthetic), falling back to synthesis
        let res = hub.auto_download_and_cache("foundry/tiny-llama-5m", None);
        assert!(res.is_ok());
        let artifacts = res.unwrap();
        assert!(artifacts.is_cached);
        assert_eq!(artifacts.estimated_params, 5_000_000);
    }

    #[test]
    fn test_hub_with_storage() {
        let dir = tempdir().unwrap();
        let custom_cache = dir.path().join("custom_storage");
        let backend = StorageBackend::LocalDisk {
            cache_dir: custom_cache.clone(),
        };
        let hub = HuggingFaceHub::new().with_storage(backend.clone());
        assert_eq!(hub.storage, backend);
        assert_eq!(hub.cache_root, custom_cache);
    }
    #[test]
    fn test_auto_smallest_quant_resolution() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());

        let model_dir = dir
            .path()
            .join("models--test--llama-multi-quant")
            .join("snapshots")
            .join("main");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), "fp16 weights").unwrap();
        fs::write(model_dir.join("model.q8_0.gguf"), "q8 weights").unwrap();
        fs::write(model_dir.join("model.q4_k_m.gguf"), "q4 weights").unwrap();

        // Auto resolution selects smallest quant: Q4_K_M (~4.5 bpw) < Q8_0 (~8.5 bpw) < FP16 (16 bpw)
        let artifacts = hub.locate_model("test/llama-multi-quant");
        assert!(artifacts.is_cached);
        assert_eq!(
            artifacts.weights_path,
            Some(model_dir.join("model.q4_k_m.gguf"))
        );
        assert_eq!(artifacts.quant_format, Some(QuantFormat::GGUF_Q4_K_M));
    }

    /// Real configs: SmolLM2-135M (tied, GQA) counts exactly; DistilBERT's `hidden_dim` is its
    /// FFN width, not the model width (once read as 3072 wide, ~10x too many params).
    #[test]
    fn test_estimate_params_from_real_configs() {
        let dir = tempdir().unwrap();
        let est = |json: &str| {
            let p = dir.path().join("config.json");
            fs::write(&p, json).unwrap();
            estimate_params_from_config(&p).unwrap()
        };
        let smollm2 = r#"{"model_type":"llama","hidden_size":576,"intermediate_size":1536,
            "num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,
            "vocab_size":49152,"tie_word_embeddings":true}"#;
        assert_eq!(est(smollm2), 134_515_008);
        let distilbert = r#"{"model_type":"distilbert","dim":768,"hidden_dim":3072,"n_layers":6,
            "n_heads":12,"vocab_size":30522}"#;
        assert!((60_000_000..100_000_000).contains(&est(distilbert)));
    }

    #[test]
    fn test_explicit_quant_override() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf()).with_quant("q8_0");

        let model_dir = dir
            .path()
            .join("models--test--llama-override")
            .join("snapshots")
            .join("main");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), "fp16 weights").unwrap();
        fs::write(model_dir.join("model.q8_0.gguf"), "q8 weights").unwrap();
        fs::write(model_dir.join("model.q4_k_m.gguf"), "q4 weights").unwrap();

        // Explicit quant override honors requested format even if smaller quant is available
        let artifacts = hub.locate_model("test/llama-override");
        assert!(artifacts.is_cached);
        assert_eq!(
            artifacts.weights_path,
            Some(model_dir.join("model.q8_0.gguf"))
        );
        assert_eq!(artifacts.quant_format, Some(QuantFormat::GGUF_Q8_0));
    }

    #[test]
    fn test_selection_skips_formats_the_loader_cannot_decode() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        let model_dir = dir
            .path()
            .join("models--test--multi")
            .join("snapshots")
            .join("main");
        fs::create_dir_all(&model_dir).unwrap();
        for f in [
            "m-Q2_K.gguf",
            "m-Q3_K_M.gguf",
            "m-IQ4_XS.gguf",
            "m-Q4_0_4_4.gguf",
            "model.nvfp4",
            "m-Q8_0.gguf",
            "m-Q5_K_M.gguf",
            "model.safetensors",
        ] {
            fs::write(model_dir.join(f), "w").unwrap();
        }
        // Smallest decodable quant wins by default (Q2_K < Q3_K < Q5_K < Q8_0); repacked GGUFs and .nvfp4 are excluded.
        assert_eq!(
            hub.locate_weights(&model_dir),
            Some(model_dir.join("m-Q2_K.gguf"))
        );
        // An explicit request never falls back to a different quant.
        assert_eq!(
            hub.locate_weights_with_quant(&model_dir, Some("q6_k")),
            None
        );
        fs::remove_file(model_dir.join("m-Q2_K.gguf")).unwrap();
        fs::remove_file(model_dir.join("m-Q3_K_M.gguf")).unwrap();
        fs::remove_file(model_dir.join("m-IQ4_XS.gguf")).unwrap();
        fs::remove_file(model_dir.join("m-Q5_K_M.gguf")).unwrap();
        fs::remove_file(model_dir.join("m-Q8_0.gguf")).unwrap();
        assert_eq!(
            hub.locate_weights(&model_dir),
            Some(model_dir.join("model.safetensors"))
        );

        let q = |f: &str| detect_quant_format(Path::new(f));
        assert_eq!(
            q("SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            Some(QuantFormat::GGUF_Q4_K_M)
        );
        assert_eq!(q("stories15M-q4_0.gguf"), Some(QuantFormat::GGUF_Q4_0));
        assert_eq!(q("x.Q4_1.gguf"), Some(QuantFormat::GGUF_Q4_1));
        assert_eq!(q("x-Q5_1.gguf"), Some(QuantFormat::GGUF_Q5_1));
        assert_eq!(q("x-Q6_K.gguf"), Some(QuantFormat::GGUF_Q6_K));
        assert_eq!(q("liquid-f16.gguf"), Some(QuantFormat::FP16));
        assert_eq!(q("x-IQ4_NL.gguf"), Some(QuantFormat::GGUF_IQ4_NL));
        assert_eq!(q("x-TQ2_0.gguf"), Some(QuantFormat::GGUF_TQ2_0));
        assert_eq!(q("x.i1-IQ2_M.gguf"), Some(QuantFormat::GGUF_IQ2_M));
        assert_eq!(q("x-Q3_K_XL.gguf"), Some(QuantFormat::GGUF_Q3_K));
        assert_eq!(q("x-Q4_0_4_4.gguf"), None);
        assert_eq!(
            q("Qwen3.8-Flash-Next-Uncensored-IQ4XS-NGQ4.gguf"),
            Some(QuantFormat::GGUF_IQ4_XS)
        );
        assert_eq!(q("model-IQ3XXS.gguf"), Some(QuantFormat::GGUF_IQ3_XXS));
    }

    #[test]
    fn test_explicit_selection_beats_cache() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        let repo = "gpustack/stable-diffusion-v2-1-turbo-GGUF";
        let file = "stable-diffusion-v2-1-turbo-FP16.gguf";
        let file_id = format!("{repo}/{file}");
        let snap = dir
            .path()
            .join("models--gpustack--stable-diffusion-v2-1-turbo-GGUF/snapshots/main");
        fs::create_dir_all(&snap).unwrap();
        let q4 = snap.join("stable-diffusion-v2-1-turbo-Q4_0.gguf");
        let fp16 = snap.join(file);
        fs::write(&q4, "q4").unwrap();

        // 3-part id resolves to the repo snapshot and the named file.
        assert_eq!(split_weight_file(&file_id), (repo, Some(file)));
        assert_eq!(split_weight_file(repo), (repo, None));
        let listing: Vec<RepoFile> = [
            "stable-diffusion-v2-1-turbo-Q8_0.gguf",
            "stable-diffusion-v2-1-turbo-Q4_0.gguf",
            file,
        ]
        .map(|p| RepoFile {
            path: p.into(),
            size: None,
        })
        .to_vec();
        assert_eq!(select_weight_files(&listing, Some("FP16")).unwrap(), [file]);
        // Default download plan is smallest-first: Q4_0 before Q8_0.
        assert_eq!(
            select_weight_files(&listing, None).unwrap(),
            ["stable-diffusion-v2-1-turbo-Q4_0.gguf"]
        );
        // A specific tag never plans a sibling quant.
        assert!(select_weight_files(&listing, Some("Q4_K_M")).is_err());

        // Cached Q4_0 only serves the unselected request.
        assert_eq!(
            hub.locate_model_with_quant(repo, None).weights_path,
            Some(q4.clone())
        );
        let by_quant = hub.locate_model_with_quant(repo, Some("FP16"));
        assert_ne!(by_quant.weights_path, Some(q4.clone()));
        assert!(by_quant.weights_path.is_none());
        let by_file = hub.locate_model_with_quant(&file_id, None);
        assert_eq!(by_file.local_dir, Some(snap.clone()));
        assert!(by_file.weights_path.is_none());
        // A specific tag is not satisfied by a sibling quant or a substring (F16 vs BF16).
        assert!(hub
            .locate_model_with_quant(repo, Some("Q4_K_M"))
            .weights_path
            .is_none());
        assert!(!quant_matches("F16", QuantFormat::BF16, "x-BF16.gguf"));
        assert!(quant_matches("q4", QuantFormat::GGUF_Q4_0, "x-Q4_0.gguf"));

        // Cached FP16 is a hit for the quant (any case/alias) and for the explicit file, without network.
        fs::write(&fp16, "fp16").unwrap();
        for quant in ["FP16", "fp16", "f16"] {
            assert_eq!(
                hub.locate_model_with_quant(repo, Some(quant)).weights_path,
                Some(fp16.clone())
            );
        }
        assert_eq!(
            hub.locate_model_with_quant(&file_id, None).weights_path,
            Some(fp16.clone())
        );
        assert_eq!(
            hub.auto_download_and_cache_with_quant(&file_id, None, None)
                .unwrap()
                .weights_path,
            Some(fp16.clone())
        );
        assert_eq!(
            hub.auto_download_and_cache_with_quant(repo, None, Some("FP16"))
                .unwrap()
                .weights_path,
            Some(fp16)
        );
    }

    // The default ranking reads measured codec rates; pin them so the expectations do not follow the host.
    #[test]
    fn test_select_weight_files_policy() {
        crate::model::variant::with_unmeasured_host(select_weight_files_policy);
    }

    fn select_weight_files_policy() {
        let ls = |names: &[(&str, Option<u64>)]| -> Vec<RepoFile> {
            names
                .iter()
                .map(|(p, s)| RepoFile {
                    path: p.to_string(),
                    size: *s,
                })
                .collect()
        };
        let names =
            |n: &[&str]| -> Vec<RepoFile> { ls(&n.iter().map(|p| (*p, None)).collect::<Vec<_>>()) };

        // Sharded: the index plus every shard it names, in order; metadata is not a weight file.
        let sharded = names(&[
            "config.json",
            "model.safetensors.index.json",
            "model-00002-of-00002.safetensors",
            "model-00001-of-00002.safetensors",
        ]);
        assert_eq!(
            select_weight_files(&sharded, None).unwrap(),
            [
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        assert_eq!(
            select_weight_files(&sharded[..3], None).unwrap_err(),
            "sharded index model.safetensors.index.json lists missing shard model-00001-of-00002.safetensors"
        );
        // Qwen3.5 names its shards after the full `model.safetensors` stem.
        let qwen = names(&[
            "model.safetensors.index.json",
            "model.safetensors-00001-of-00001.safetensors",
        ]);
        assert_eq!(
            select_weight_files(&qwen, None).unwrap(),
            [
                "model.safetensors.index.json",
                "model.safetensors-00001-of-00001.safetensors"
            ]
        );

        // Mistral's native `consolidated.safetensors` (names the loader doesn't map) is never
        // chosen over the HF shards; GGUF vision projectors are never the language model.
        let mistral = names(&[
            "consolidated.safetensors",
            "model.safetensors.index.json",
            "model-00001-of-00001.safetensors",
        ]);
        assert_eq!(
            select_weight_files(&mistral, None).unwrap()[0],
            "model.safetensors.index.json"
        );
        let vlm = names(&["mmproj-model-f16.gguf", "qwen-vl-Q2_K.gguf"]);
        assert_eq!(
            select_weight_files(&vlm, None).unwrap(),
            ["qwen-vl-Q2_K.gguf"]
        );
        assert!(select_weight_files(&vlm, Some("f16")).is_err());
        // Projectors (`mmproj-*`, `*-mmproj`) are companions, never weights: the smallest one
        // beside the language model goes with it; a repo of projectors alone is no model.
        let g4 = ls(&[
            ("gemma-4-E2B-it-Q4_0.gguf", Some(2841)),
            ("mmproj-gemma-4-E2B-it-BF16.gguf", Some(986)),
            ("mmproj-gemma-4-E2B-it-Q8_0.gguf", Some(557)),
            ("other/mmproj-x.gguf", Some(1)),
        ]);
        assert_eq!(
            select_weight_files(&g4, None).unwrap(),
            ["gemma-4-E2B-it-Q4_0.gguf"]
        );
        assert_eq!(
            select_mmproj(&g4, "gemma-4-E2B-it-Q4_0.gguf").as_deref(),
            Some("mmproj-gemma-4-E2B-it-Q8_0.gguf")
        );
        let suffixed = names(&["g-q4_0.gguf", "g-mmproj.gguf"]);
        assert_eq!(
            select_weight_files(&suffixed, None).unwrap(),
            ["g-q4_0.gguf"]
        );
        assert_eq!(
            select_weight_files(&names(&["mmproj-a-F16.gguf"]), None).unwrap_err(),
            MMPROJ_ONLY
        );

        // Size variants share a tensor format; an explicit size never yields a sibling size.
        let iq = names(&["m-IQ3_M.gguf", "m-IQ3_XS.gguf"]);
        assert!(select_weight_files(&iq, Some("IQ3_S")).is_err());
        assert_eq!(
            select_weight_files(&iq, Some("IQ3_M")).unwrap(),
            ["m-IQ3_M.gguf"]
        );
        let k = names(&["m-Q3_K_L.gguf", "m-Q3_K_S.gguf", "m-Q4_K_S.gguf"]);
        assert!(select_weight_files(&k, Some("Q3_K_M")).is_err());
        assert!(select_weight_files(&k, Some("Q4_K_M")).is_err());
        assert_eq!(
            select_weight_files(&k, Some("Q3_K_S")).unwrap(),
            ["m-Q3_K_S.gguf"]
        );
        assert_eq!(
            select_weight_files(&k, Some("q4_k")).unwrap(),
            ["m-Q4_K_S.gguf"]
        );
        // candle's `-q4k` naming (google/madlad400-3b-mt): smallest quant (q2k) is the default.
        let candle = names(&["model-q2k.gguf", "model-q3k.gguf", "model-q4k.gguf"]);
        assert_eq!(
            select_weight_files(&candle, None).unwrap(),
            ["model-q2k.gguf"]
        );

        // An index beside a single file (mlx-community layout) is not a shard set.
        let single_indexed = names(&["model.safetensors.index.json", "model.safetensors"]);
        assert_eq!(
            select_weight_files(&single_indexed, None).unwrap(),
            ["model.safetensors"]
        );

        // On disk, a sharded set counts only once every shard its index names is present.
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());
        let snap = dir.path().join("models--o--sharded/snapshots/main");
        fs::create_dir_all(snap.join("parts")).unwrap();
        fs::write(
            snap.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00001.safetensors","b":"parts/extra.safetensors"}}"#,
        )
        .unwrap();
        fs::write(snap.join("model-00001-of-00001.safetensors"), "w").unwrap();
        assert_eq!(hub.locate_model("o/sharded").weights_path, None);
        fs::write(snap.join("parts/extra.safetensors"), "w").unwrap();
        assert_eq!(
            hub.locate_model("o/sharded").weights_path,
            Some(snap.join("model.safetensors.index.json"))
        );

        // ONNX: smallest download (graph + external data) among decodable variants.
        let mut onnx = ls(&[
            ("onnx/model.onnx", Some(1_000)),
            ("onnx/model.onnx_data", Some(4_000_000)),
            ("onnx/model_q4.onnx", Some(500_000)),
            ("onnx/model_fp16.onnx", Some(10)),
            ("onnx/model_fp16.onnx_data", Some(2_000_000)),
            ("onnx/decoder_model_merged.onnx", Some(1)),
        ]);
        assert_eq!(
            select_weight_files(&onnx, None).unwrap(),
            ["onnx/model_q4.onnx"]
        );
        assert_eq!(
            select_weight_files(&onnx, Some("fp16")).unwrap(),
            ["onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"]
        );
        assert_eq!(
            select_weight_files(&onnx, Some("fp32")).unwrap(),
            ["onnx/model.onnx", "onnx/model.onnx_data"]
        );
        // ONNX ranks after any safetensors/GGUF.
        onnx.push(RepoFile {
            path: "model.safetensors".into(),
            size: Some(1 << 30),
        });
        assert_eq!(
            select_weight_files(&onnx, None).unwrap(),
            ["model.safetensors"]
        );

        // GGUF: smallest bpw first; explicit quant strict.
        let gguf = names(&[
            "m-Q8_0.gguf",
            "m-Q2_K.gguf",
            "m-Q4_K_M.gguf",
            "model.safetensors",
        ]);
        assert_eq!(select_weight_files(&gguf, None).unwrap(), ["m-Q2_K.gguf"]);
        assert_eq!(
            select_weight_files(&gguf, Some("Q8_0")).unwrap(),
            ["m-Q8_0.gguf"]
        );
        assert!(select_weight_files(&gguf, Some("Q6_K"))
            .unwrap_err()
            .starts_with("no weights matching 'Q6_K'"));
        // Sub-4.5-bpw rule on the ranking itself (no GGUF tag below 4.5 bpw decodes before the
        // GgufQuants merge, so the sets are built by hand).
        let set = |file: &'static str, bpw: f64, tag: &'static str| WeightSet {
            files: vec![file],
            tier: 0,
            rank: bpw,
            matches: Box::new(move |eq| eq.eq_ignore_ascii_case(tag)),
        };
        let all = || {
            vec![
                set("m-Q2_K.gguf", 2.625, "q2_k"),
                set("model.safetensors", 16.0, "fp16"),
                set("m-Q8_0.gguf", 8.5, "q8_0"),
            ]
        };
        assert_eq!(best_set(all(), None), Some(vec!["m-Q2_K.gguf"]));
        assert_eq!(best_set(all(), Some("Q2_K")), Some(vec!["m-Q2_K.gguf"]));
        assert_eq!(best_set(all(), Some("q4_0")), None);
        assert_eq!(
            best_set(vec![set("m-Q2_K.gguf", 2.625, "q2_k")], None),
            Some(vec!["m-Q2_K.gguf"])
        );
        // Smallest quant wins by default.
        let low = names(&["m-Q2_K.gguf", "m-Q8_0.gguf"]);
        assert_eq!(select_weight_files(&low, None).unwrap(), ["m-Q2_K.gguf"]);

        // PyTorch / Flax / NumPy only when nothing better exists; trainer state is never weights.
        let torch = names(&[
            "pytorch_model.bin",
            "flax_model.msgpack",
            "optimizer.pt",
            "training_args.bin",
        ]);
        assert_eq!(
            select_weight_files(&torch, None).unwrap(),
            ["flax_model.msgpack"]
        );
        let mut both = torch.clone();
        both.extend(names(&["onnx/model.onnx"]));
        assert_eq!(
            select_weight_files(&both, None).unwrap(),
            ["onnx/model.onnx"]
        );
        let sharded = names(&[
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ]);
        assert_eq!(select_weight_files(&sharded, None).unwrap().len(), 3);
        assert_eq!(
            select_weight_files(&sharded[..2], None).unwrap_err(),
            "sharded index pytorch_model.bin.index.json lists missing shard pytorch_model-00002-of-00002.bin"
        );
        // Nothing decodable: the reason lists what the repo has.
        assert_eq!(
            select_weight_files(&names(&["tf_model.h5", "rust_model.ot", "README.md"]), None)
                .unwrap_err(),
            "no weights the loader reads (found: tf_model.h5, rust_model.ot)"
        );
        // OpenVINO IR / CTranslate2 / GGML `.bin` files are not pickles: never auto-picked.
        assert!(select_weight_files(
            &names(&[
                "openvino_model.bin",
                "openvino_model.xml",
                "model.bin",
                "ggml-model.bin"
            ]),
            None
        )
        .is_err());
    }

    /// A split GGUF (`-00001-of-00003.gguf`) is one weight set: every shard, in order, of the one
    /// quant chosen (never a mix), tagged by file or folder name; an incomplete set names its gap.
    #[test]
    fn test_select_weight_files_split_gguf() {
        let ls = |n: &[&str]| -> Vec<RepoFile> {
            n.iter()
                .map(|p| RepoFile {
                    path: p.to_string(),
                    size: Some(10),
                })
                .collect()
        };
        let repo = ls(&[
            "README.md",
            "Q8_0/m-00002-of-00002.gguf",
            "Q8_0/m-00001-of-00002.gguf",
            "Q4_0/m-00003-of-00003.gguf",
            "Q4_0/m-00001-of-00003.gguf",
            "Q4_0/m-00002-of-00003.gguf",
        ]);
        assert_eq!(
            select_weight_files(&repo, None).unwrap(),
            [
                "Q4_0/m-00001-of-00003.gguf",
                "Q4_0/m-00002-of-00003.gguf",
                "Q4_0/m-00003-of-00003.gguf"
            ]
        );
        assert_eq!(
            select_weight_files(&repo, Some("Q8_0")).unwrap(),
            ["Q8_0/m-00001-of-00002.gguf", "Q8_0/m-00002-of-00002.gguf"]
        );
        assert_eq!(
            detect_quant_format(Path::new("IQ2_S/GLM-00001-of-00003.gguf")),
            Some(QuantFormat::GGUF_IQ2_S)
        );
        // Missing shard: refused with the shard named.
        let gap = ls(&["m-Q4_0-00001-of-00003.gguf", "m-Q4_0-00003-of-00003.gguf"]);
        assert!(select_weight_files(&gap, None)
            .unwrap_err()
            .contains("missing shard m-Q4_0-00002-of-00003.gguf"));
    }

    /// A diffusers repo downloads the fp16 component set only: no non_ema / fp32 duplicates,
    /// ONNX, safety checker or root single-file checkpoint.
    #[test]
    fn test_diffusers_selection_prefers_fp16_components() {
        let files: Vec<RepoFile> = [
            "model_index.json",
            "v1-5-pruned-emaonly.safetensors",
            "safety_checker/model.safetensors",
            "scheduler/scheduler_config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder/model.onnx",
            "text_encoders/clip_g.safetensors",
            "text_encoder_2/model.safetensors",
            "tokenizer/special_tokens_map.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.non_ema.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.non_ema.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ]
        .iter()
        .map(|p| RepoFile {
            path: p.to_string(),
            size: None,
        })
        .collect();
        let mut sel = select_weight_files(&files, None).unwrap();
        assert_eq!(sel[0], "unet/diffusion_pytorch_model.fp16.safetensors");
        sel.sort();
        assert_eq!(
            sel,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/model.fp16.safetensors",
                "text_encoder_2/model.safetensors",
                "tokenizer/special_tokens_map.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ]
        );
    }

    #[test]
    fn test_gibbs_and_foundry_backwards_compatibility() {
        let dir = tempdir().unwrap();
        let hub = HuggingFaceHub::with_cache_dir(dir.path().to_path_buf());

        // Synthesizing with gibbs/ creates gibbs-- folder
        let art = hub.resolve_or_synthesize_tiny("gibbs/tiny-5m").unwrap();
        assert!(art.is_cached);
        assert!(art
            .local_dir
            .unwrap()
            .to_str()
            .unwrap()
            .contains("gibbs--tiny-5m"));

        // Existing foundry-- folder is found via backwards compatibility
        let legacy_dir = dir.path().join("foundry--legacy-model");
        fs::create_dir_all(&legacy_dir).unwrap();
        fs::write(legacy_dir.join("config.json"), "{}").unwrap();
        fs::write(legacy_dir.join("model.safetensors"), "legacy").unwrap();

        let legacy_art = hub.locate_model("gibbs/legacy-model");
        assert!(legacy_art.is_cached);
        assert_eq!(legacy_art.local_dir, Some(legacy_dir));
    }
}
