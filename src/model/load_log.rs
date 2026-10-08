//! Load-failure log. A model that fails to load appends one record to `<cache root>/load-failures.log`:
//! the error, the stage it died in, and what the snapshot actually holds (files, `config.json`
//! summary, safetensors tensor names), so "tensor not found" is diagnosable without a re-run.

use crate::model::hub::HuggingFaceHub;
use serde_json::Value;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Tensor names listed per safetensors file.
const TENSORS_SHOWN: usize = 24;
/// Files listed per snapshot.
const FILES_SHOWN: usize = 80;

pub fn path(hub: &HuggingFaceHub) -> PathBuf {
    hub.cache_root.join("load-failures.log")
}

/// Appends a record for `id` failing in `stage` with `err`; returns `err` plus the log path, the
/// text to show the user. A log that cannot be written is named in the returned text instead.
pub fn record(hub: &HuggingFaceHub, id: &str, stage: &str, err: &str) -> String {
    let mut r = format!(
        "=== {} load failed: {id} ===\nstage: {stage}\nerror: {}\ncache: {}\n",
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        err.replace('\n', "\n       "),
        hub.cache_root.display()
    );
    let art = hub.locate_model(id);
    match &art.local_dir {
        Some(dir) => describe_snapshot(&mut r, dir, &art.weights_path),
        None => r.push_str("snapshot: none on disk (the download did not start or was removed)\n"),
    }
    r.push('\n');
    let log = path(hub);
    let written = std::fs::create_dir_all(&hub.cache_root)
        .and_then(|_| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log)
        })
        .and_then(|mut f| f.write_all(r.as_bytes()));
    match written {
        Ok(()) => format!("{err}\ndetails: {}", log.display()),
        Err(e) => format!("{err}\n(could not write {}: {e})", log.display()),
    }
}

fn describe_snapshot(r: &mut String, dir: &Path, weights: &Option<PathBuf>) {
    let _ = writeln!(r, "snapshot: {}", dir.display());
    let mut files = Vec::new();
    walk(dir, dir, 0, &mut files);
    files.sort();
    let _ = writeln!(r, "files ({}):", files.len());
    for (name, size) in files.iter().take(FILES_SHOWN) {
        let _ = writeln!(r, "  {name}  {size} B");
    }
    if files.len() > FILES_SHOWN {
        let _ = writeln!(r, "  ... {} more", files.len() - FILES_SHOWN);
    }
    let _ = writeln!(
        r,
        "selected weights: {}",
        weights
            .as_ref()
            .map_or("none".into(), |w| w.display().to_string())
    );
    if let Some(c) = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
    {
        let _ = writeln!(
            r,
            "config: model_type={} architectures={} quantization_config={}",
            c["model_type"], c["architectures"], c["quantization_config"]
        );
    }
    for (name, _) in files
        .iter()
        .filter(|(n, _)| n.ends_with(".safetensors"))
        .take(3)
    {
        match safetensors_names(&dir.join(name)) {
            Ok(names) => {
                let _ = writeln!(r, "tensors in {name}: {} total", names.len());
                for n in names.iter().take(TENSORS_SHOWN) {
                    let _ = writeln!(r, "  {n}");
                }
            }
            Err(e) => {
                let _ = writeln!(r, "tensors in {name}: unreadable header ({e})");
            }
        }
    }
    // ponytail: GGUF tensor names are not listed (the reader takes the whole file); the error names the one that failed.
}

/// Files under `dir` (two levels deep) as (relative path, size).
fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, u64)>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        let Ok(meta) = std::fs::metadata(&p) else {
            continue; // dangling blob link: the file is listed as missing by its absence
        };
        if meta.is_dir() && depth < 2 {
            walk(root, &p, depth + 1, out);
        } else if meta.is_file() {
            let rel = p.strip_prefix(root).unwrap_or(&p).display().to_string();
            out.push((rel, meta.len()));
        }
    }
}

/// Sorted tensor names from a safetensors header (8-byte length, then JSON).
fn safetensors_names(p: &Path) -> Result<Vec<String>, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(p).map_err(|e| e.to_string())?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| e.to_string())?;
    let len = u64::from_le_bytes(len);
    if len > 1 << 28 {
        return Err(format!("header length {len} is implausible"));
    }
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let h: Value = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
    let mut names: Vec<String> = h
        .as_object()
        .ok_or("header is not an object")?
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect();
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_lists_what_the_snapshot_holds() {
        let cache = tempfile::tempdir().unwrap();
        let snap = cache.path().join("models--o--m/snapshots/main");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        let header = br#"{"a.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut st = (header.len() as u64).to_le_bytes().to_vec();
        st.extend_from_slice(header);
        st.extend_from_slice(&[0; 4]);
        std::fs::write(snap.join("model.safetensors"), st).unwrap();
        let hub = HuggingFaceHub::with_cache_dir(cache.path().to_path_buf());

        let shown = record(
            &hub,
            "o/m",
            "reading weights",
            "tensor 'b.weight' not found",
        );
        assert!(shown.contains("load-failures.log"));
        let log = std::fs::read_to_string(path(&hub)).unwrap();
        for want in [
            "o/m",
            "reading weights",
            "tensor 'b.weight' not found",
            "a.weight",
            "model_type=\"llama\"",
            "model.safetensors",
        ] {
            assert!(log.contains(want), "{want} missing from:\n{log}");
        }
        record(&hub, "o/m", "x", "second");
        assert_eq!(
            std::fs::read_to_string(path(&hub))
                .unwrap()
                .matches("load failed")
                .count(),
            2
        );
    }
}
