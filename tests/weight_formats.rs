//! PyTorch pickle / Flax msgpack / NumPy npz checkpoints decode to the same tensors as the
//! reference file of the same weights, and broken files fail with Err instead of panicking.

use gibbs_engine::model::loaders::read_checkpoint;
use std::path::{Path, PathBuf};

fn snap(repo: &str, file: &str) -> Option<PathBuf> {
    let p = Path::new("disk")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots/main")
        .join(file);
    if p.exists() {
        Some(p)
    } else {
        println!("SKIP: {} not cached", p.display());
        None
    }
}

/// `reference` and `other` hold the same tensors with equal shapes and bitwise-equal values.
/// `rename` maps an `other` name to its `reference` name; names ending in an `unpaired` suffix
/// may exist on one side only (non-parameter buffers, tied copies). Tensors for which
/// `permuted` holds are stored with reordered axes in `other`: same multiset of values.
fn assert_same(
    reference: &Path,
    other: &Path,
    rename: impl Fn(&str) -> String,
    unpaired: &[&str],
    permuted: impl Fn(&str) -> bool,
) {
    let (rm, rd) = read_checkpoint(reference).unwrap();
    let (om, od) = read_checkpoint(other).unwrap();
    let skip = |n: &str| unpaired.iter().any(|u| n.ends_with(u));
    let mut paired = std::collections::HashSet::new();
    for (name, o) in &om.tensors {
        let rname = rename(name);
        let Some(t) = rm.tensors.get(&rname) else {
            assert!(skip(name), "{}: unexpected {name}", other.display());
            continue;
        };
        paired.insert(rname.clone());
        let bits = |m: &gibbs_engine::model::ModelManifest, d: &[u8], n: &str| -> Vec<u32> {
            let v = m.extract_tensor_f32(d, n).unwrap();
            v.iter().map(|x| x.to_bits()).collect()
        };
        let (mut a, mut b) = (bits(&rm, &rd, &rname), bits(&om, &od, name));
        if permuted(name) {
            a.sort_unstable();
            b.sort_unstable();
        } else {
            assert_eq!(t.shape, o.shape, "{name} shape");
        }
        assert!(
            a == b,
            "{name}: values differ between {} and {}",
            reference.display(),
            other.display()
        );
    }
    let missing: Vec<_> = rm
        .tensors
        .keys()
        .filter(|n| !paired.contains(*n) && !skip(n))
        .collect();
    assert!(
        missing.is_empty(),
        "{}: missing {missing:?}",
        other.display()
    );
    println!(
        "{} == {}: {} tensors",
        reference.display(),
        other.display(),
        paired.len()
    );
}

#[test]
fn pytorch_bin_matches_safetensors() {
    for repo in ["nickypro/tinyllama-15M", "distilbert/distilgpt2"] {
        let (Some(st), Some(bin)) = (
            snap(repo, "model.safetensors"),
            snap(repo, "pytorch_model.bin"),
        ) else {
            continue;
        };
        assert_same(&st, &bin, str::to_string, &["lm_head.weight"], |_| false);
    }
}

#[test]
fn sharded_pytorch_bin_matches_sharded_safetensors() {
    let repo = "hf-internal-testing/tiny-random-bert-sharded";
    let (Some(st), Some(bin)) = (
        snap(repo, "model.safetensors.index.json"),
        snap(repo, "pytorch_model.bin.index.json"),
    ) else {
        return;
    };
    assert_same(&st, &bin, str::to_string, &["lm_head.weight"], |_| false);
}

#[test]
fn flax_msgpack_matches_safetensors() {
    let repo = "distilbert/distilgpt2";
    let (Some(st), Some(flax)) = (
        snap(repo, "model.safetensors"),
        snap(repo, "flax_model.msgpack"),
    ) else {
        return;
    };
    assert_same(&st, &flax, str::to_string, &["attn.bias"], |_| false);
}

/// MLX's `weights.npz` for Whisper tiny against OpenAI's original `tiny.pt`
/// (`{"dims", "model_state_dict"}` wrapper); MLX stores conv kernels channels-last and renames
/// `mlp.0`/`mlp.2` to `mlp1`/`mlp2`.
#[test]
fn npz_matches_pytorch_checkpoint() {
    let (Some(pt), Some(npz)) = (
        snap("openai/whisper-tiny-pt", "tiny.pt"),
        snap("mlx-community/whisper-tiny", "weights.npz"),
    ) else {
        return;
    };
    let rename = |n: &str| n.replace(".mlp1.", ".mlp.0.").replace(".mlp2.", ".mlp.2.");
    // MLX recomputes Whisper's sinusoidal encoder positions instead of storing them.
    assert_same(&pt, &npz, rename, &["encoder.positional_embedding"], |n| {
        n.contains(".conv")
    });
}

/// Truncations of every format (and garbage) return Err; none panics.
#[test]
fn truncated_and_malformed_files_err() {
    let dir = tempfile::tempdir().unwrap();
    let sources = [
        snap("nickypro/tinyllama-15M", "pytorch_model.bin"),
        snap("openai/whisper-tiny-pt", "tiny.pt"),
        snap("mlx-community/whisper-tiny", "weights.npz"),
        snap("distilbert/distilgpt2", "flax_model.msgpack"),
    ];
    for src in sources.into_iter().flatten() {
        let bytes = std::fs::read(&src).unwrap();
        let name = src.file_name().unwrap();
        for cut in [0, 7, 64, 1000, bytes.len() / 2, bytes.len() - 1] {
            let p = dir.path().join(name);
            std::fs::write(&p, &bytes[..cut]).unwrap();
            assert!(
                read_checkpoint(&p).is_err(),
                "{} cut at {cut}",
                src.display()
            );
        }
    }
    for ext in ["bin", "pt", "msgpack", "npz", "npy"] {
        let p = dir.path().join(format!("junk.{ext}"));
        std::fs::write(
            &p,
            b"\x93NUMPY\x01\x00\xff\xff{'descr'\x80\x81\xdf\xff\xff\xff\xffPK",
        )
        .unwrap();
        assert!(read_checkpoint(&p).is_err(), "junk.{ext}");
    }
    // Deep nesting must not overflow the stack.
    let p = dir.path().join("deep.msgpack");
    std::fs::write(&p, [0x81, 0xa1, b'a'].repeat(100_000)).unwrap();
    assert!(read_checkpoint(&p).is_err());
}
