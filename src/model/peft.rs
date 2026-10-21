//! PEFT LoRA `CAUSAL_LM` adapters run as LLMs: the base model's safetensors with
//! `W + scale · B·A` (and any `modules_to_save` tensors) merged are written once into the
//! adapter repo's cache snapshot, next to the base's config and tokenizer (the adapter's own
//! tokenizer files win). Every LLM loader then reads the snapshot like any checkpoint.

use crate::model::hub::{
    is_synthetic_model_id, list_repo_files, split_weight_file, HuggingFaceHub,
};
use crate::model::loaders::read_checkpoint;
use crate::model::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, nn, UniversalModelLoader};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

/// Files the merged snapshot needs besides the weights (adapter's copy first, else the base's).
const METADATA: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
];

/// When `repo_id` is an uncached PEFT LoRA `CAUSAL_LM` adapter, merges it into its base model
/// and writes the result as the adapter's cached snapshot (`Ok(true)`); `Ok(false)` for anything
/// else (cached models, regular repos, other adapter tasks).
pub fn prepare_causal_lm(
    hub: &HuggingFaceHub,
    repo_id: &str,
    token: Option<&str>,
) -> Result<bool, String> {
    if is_synthetic_model_id(repo_id)
        || split_weight_file(repo_id).1.is_some()
        || Path::new(repo_id).is_dir()
        || hub.locate_model(repo_id).weights_path.is_some()
    {
        return Ok(false);
    }
    let Ok(ad_path) = hub.fetch_file(repo_id, "adapter_config.json") else {
        return Ok(false);
    };
    let ad: Value = std::fs::read(&ad_path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
        .map_err(|e| format!("{}: {e}", ad_path.display()))?;
    if ad["task_type"] != "CAUSAL_LM" {
        return Ok(false);
    }
    let scale = lora_scale(&ad).map_err(|e| format!("{repo_id}: {e}"))?;
    let base = ad["base_model_name_or_path"]
        .as_str()
        .filter(|b| b.split('/').count() == 2 && !b.starts_with('.') && !b.starts_with('/'))
        .ok_or_else(|| {
            format!(
                "{repo_id}: adapter base_model_name_or_path {} is not a Hub repo",
                ad["base_model_name_or_path"]
            )
        })?;

    let base_art = hub.auto_download_and_cache(base, token)?;
    let weights = base_art
        .weights_path
        .ok_or_else(|| format!("{base}: no weight file"))?;
    let base_dir = weights.parent().unwrap_or(Path::new(".")).to_path_buf();
    let name = weights.to_string_lossy();
    let shards: Vec<String> = if name.ends_with(".safetensors.index.json") {
        let json = std::fs::read(&weights).map_err(|e| format!("{}: {e}", weights.display()))?;
        UniversalModelLoader::parse_safetensors_index(&json)?
            .into_values()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    } else if name.ends_with(".safetensors") {
        vec![weights.file_name().unwrap().to_string_lossy().into_owned()]
    } else {
        return Err(format!(
            "{repo_id}: LoRA merges into safetensors bases only ({base} ships {})",
            weights.display()
        ));
    };

    let adapter = hub
        .fetch_file(repo_id, "adapter_model.safetensors")
        .or_else(|_| hub.fetch_file(repo_id, "adapter_model.bin"))?;
    let (am, ad_data) = read_checkpoint(&adapter)?;
    let dir = adapter.parent().unwrap_or(Path::new(".")).to_path_buf();
    let listed: Vec<String> = list_repo_files(repo_id, token)?
        .into_iter()
        .map(|f| f.path)
        .collect();
    for f in METADATA {
        if listed.iter().any(|p| p == f) {
            hub.fetch_file(repo_id, f)?;
        } else if base_dir.join(f).is_file() && !dir.join(f).is_file() {
            std::fs::copy(base_dir.join(f), dir.join(f))
                .map_err(|e| format!("copy {f} from {base}: {e}"))?;
        }
    }

    // Pass 1: every base tensor (shard, name, dtype, shape, byte range) and which adapter tensor
    // replaces or updates it; every adapter tensor must land somewhere.
    struct Entry {
        shard: usize,
        name: String,
        dtype: String,
        shape: Vec<usize>,
        range: (usize, usize),
    }
    let mut maps = Vec::new();
    let mut entries = Vec::new();
    let mut used = BTreeSet::new();
    for shard in &shards {
        let path = base_dir.join(shard);
        let file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: read-only mapping of a cache file nothing rewrites during the merge.
        let map =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header: BTreeMap<String, Value> = serde_json::from_slice(&map[8..8 + n])
            .map_err(|e| format!("{}: header: {e}", path.display()))?;
        for (name, info) in header {
            if name == "__metadata__" {
                continue;
            }
            let at = |i: usize| 8 + n + info["data_offsets"][i].as_u64().unwrap_or(0) as usize;
            let (saved, a_name) = adapter_names(&am, &name);
            used.extend(saved.into_iter().chain(a_name));
            entries.push(Entry {
                shard: maps.len(),
                dtype: info["dtype"].as_str().unwrap_or("").to_string(),
                shape: info["shape"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().map(|v| v as usize))
                            .collect()
                    })
                    .unwrap_or_default(),
                range: (at(0), at(1)),
                name,
            });
        }
        maps.push(map);
    }
    if let Some(stray) = am
        .tensors
        .keys()
        .find(|k| !k.contains(".lora_B.") && !used.contains(*k))
    {
        return Err(format!(
            "{repo_id}: adapter tensor {stray} matches no tensor of {base}"
        ));
    }
    let mut header = serde_json::Map::new();
    // Marks the snapshot as merged, so a loader that applies adapters at run time skips this one.
    header.insert(
        "__metadata__".into(),
        serde_json::json!({"peft_merged": "adapter_model"}),
    );
    let mut off = 0usize;
    for e in &entries {
        let len = e.range.1 - e.range.0;
        header.insert(
            e.name.clone(),
            serde_json::json!({"dtype": e.dtype, "shape": e.shape, "data_offsets": [off, off + len]}),
        );
        off += len;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).map_err(|e| e.to_string())?;
    // Data must start 8-byte aligned (readers borrow bf16/f32 slices straight from the map).
    header.resize(header.len().next_multiple_of(8), b' ');

    // Pass 2: stream the tensors, merged where the adapter touches them, to a temp name renamed
    // when complete so a partial merge never looks cached.
    let tmp = dir.join("model.safetensors.merging");
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
    );
    let io = |e: std::io::Error| format!("{}: {e}", tmp.display());
    out.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(io)?;
    out.write_all(&header).map_err(io)?;
    for e in &entries {
        let raw = &maps[e.shard][e.range.0..e.range.1];
        let (saved, a_name) = adapter_names(&am, &e.name);
        if let Some(saved) = saved {
            let v = UniversalModelLoader::extract_tensor_from_manifest(&am, &ad_data, &saved)?;
            if v.len() != e.shape.iter().product::<usize>() {
                return Err(format!(
                    "adapter {saved} does not match the base shape {:?}",
                    e.shape
                ));
            }
            out.write_all(&encode(&v, &e.dtype, &e.name)?).map_err(io)?;
        } else if let Some(a_name) = a_name {
            let b_name = a_name.replace(".lora_A.", ".lora_B.");
            let [out_f, in_f] = e.shape[..] else {
                return Err(format!(
                    "LoRA target {} is not a matrix ({:?})",
                    e.name, e.shape
                ));
            };
            let rank = am.tensors[&a_name].shape[0];
            if am.tensors[&a_name].shape != [rank, in_f]
                || am.tensors.get(&b_name).map(|t| &t.shape[..]) != Some(&[out_f, rank][..])
            {
                return Err(format!(
                    "LoRA factors of {} do not match [{out_f}, {in_f}]",
                    e.name
                ));
            }
            let a = UniversalModelLoader::extract_tensor_from_manifest(&am, &ad_data, &a_name)?;
            let b = UniversalModelLoader::extract_tensor_from_manifest(&am, &ad_data, &b_name)?;
            let mut w = decode(raw, &e.dtype, &e.name)?;
            let mut delta = vec![0.0; out_f * in_f];
            nn::matmul_nt(
                &b,
                &nn::transpose(&a, rank, in_f),
                None,
                out_f,
                rank,
                in_f,
                &mut delta,
            );
            w.iter_mut().zip(delta).for_each(|(w, d)| *w += scale * d);
            out.write_all(&encode(&w, &e.dtype, &e.name)?).map_err(io)?;
        } else {
            out.write_all(raw).map_err(io)?;
        }
    }
    out.flush().map_err(io)?;
    drop(out);
    std::fs::rename(&tmp, dir.join("model.safetensors"))
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    Ok(true)
}

/// The `scale` of a plain LoRA `adapter_config.json` (`lora_alpha / r`, rsLoRA `lora_alpha / √r`);
/// errs for adapter kinds a `W + scale · B·A` update cannot express.
pub fn lora_scale(ad: &Value) -> Result<f32, String> {
    if ad["peft_type"] != "LORA"
        || ad["use_dora"] == true
        || ad["fan_in_fan_out"] == true
        || ad["rank_pattern"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
        || ad["alpha_pattern"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
    {
        return Err(format!(
            "only plain LoRA adapters merge (not {}, DoRA, fan_in_fan_out or per-module rank/alpha)",
            ad["peft_type"]
        ));
    }
    let r = ad["r"].as_f64().ok_or("adapter_config.json has no r")?;
    let alpha = ad["lora_alpha"].as_f64().unwrap_or(r);
    Ok(if ad["use_rslora"] == true {
        alpha / r.sqrt()
    } else {
        alpha / r
    } as f32)
}

/// The adapter tensors that replace (`modules_to_save`) or update (`lora_A`) base tensor
/// `name`. A multimodal base keeps the text model under `model.language_model.`; an adapter
/// trained on its text-only causal LM names it `model.`.
pub(crate) fn adapter_names(
    am: &crate::model::ModelManifest,
    name: &str,
) -> (Option<String>, Option<String>) {
    let text = name.replacen("model.language_model.", "model.", 1);
    let find = |f: &dyn Fn(&str) -> String| {
        [name, text.as_str()]
            .into_iter()
            .map(f)
            .find(|k| am.tensors.contains_key(k))
    };
    (
        find(&|n| format!("base_model.model.{n}")),
        find(&|n| {
            format!(
                "base_model.model.{}.lora_A.weight",
                n.strip_suffix(".weight").unwrap_or("")
            )
        }),
    )
}

fn decode(raw: &[u8], dtype: &str, name: &str) -> Result<Vec<f32>, String> {
    let u16s = || {
        raw.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
    };
    Ok(match dtype {
        "BF16" => u16s().map(bf16_to_f32).collect(),
        "F16" => u16s().map(f16_to_f32).collect(),
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => {
            return Err(format!(
                "LoRA target {name} has dtype {other}; merges need BF16/F16/F32"
            ))
        }
    })
}

fn encode(v: &[f32], dtype: &str, name: &str) -> Result<Vec<u8>, String> {
    Ok(match dtype {
        "BF16" => v
            .iter()
            .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
            .collect(),
        "F16" => v
            .iter()
            .flat_map(|&x| f32_to_f16(x).to_le_bytes())
            .collect(),
        "F32" => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        other => {
            return Err(format!(
                "LoRA target {name} has dtype {other}; merges need BF16/F16/F32"
            ))
        }
    })
}
