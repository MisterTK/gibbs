//! PyTorch pickle checkpoints (`pytorch_model.bin`, `*.pt`, `*.pth`, `*.ckpt`): ZIP (torch >= 1.6),
//! legacy and TAR containers, via the `pytorch-reader` crate's restricted unpickler (it never runs
//! code; globals other than torch's tensor/storage rebuilders are refused or kept opaque).

use super::Checkpoint;
use pytorch_reader::{DType, PickleValue, PytorchReader};
use std::path::Path;

/// Top-level keys training scripts nest the state dict under (`{"state_dict": {...}}`).
const WRAPPERS: [&str; 3] = ["state_dict", "model_state_dict", "model"];

/// Appends every tensor of the checkpoint at `path` to `ck`, one tensor read at a time.
pub fn read(path: &Path, ck: &mut Checkpoint) -> Result<(), String> {
    // A wrapper is a nested dict in the pickle; flat HF keys (`model.embed_tokens.weight`) never are.
    let wrapper = match PytorchReader::read_pickle_data(path, None).map_err(|e| e.to_string())? {
        PickleValue::Dict(d) => WRAPPERS
            .into_iter()
            .find(|k| matches!(d.get(*k), Some(PickleValue::Dict(_)))),
        _ => None,
    };
    let reader = match wrapper {
        Some(k) => PytorchReader::with_top_level_key(path, k),
        None => PytorchReader::new(path),
    };
    let mut tensors: Vec<_> = reader
        .map_err(|e| e.to_string())?
        .into_tensors()
        .into_iter()
        .collect();
    if tensors.is_empty() {
        return Err("no tensors in checkpoint".into());
    }
    // DistributedDataParallel saves every name under `module.`.
    let strip = if tensors.iter().all(|(n, _)| n.starts_with("module.")) {
        "module.".len()
    } else {
        0
    };
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    ck.data
        .reserve_exact(tensors.iter().map(|(_, t)| t.byte_len()).sum());
    for (name, t) in tensors {
        let dtype = match t.dtype() {
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U64 => "U64",
            DType::U32 => "U32",
            DType::U16 => "U16",
            DType::U8 => "U8",
            DType::Bool => "BOOL",
        };
        ck.push(
            name[strip..].to_string(),
            t.shape().to_vec(),
            dtype,
            &t.read().map_err(|e| e.to_string())?,
        )?;
    }
    Ok(())
}
