//! GLQ (github.com/cnygaard/glq, `quant_method: "glq"`): every linear stores `W_rht` in a
//! randomized-Hadamard domain and the dense weight is `diag(SU) · H_m · (Wscale · W_rht) · H_n · diag(SV)`
//! cropped to `[out, in]`, `H` the orthonormal Walsh–Hadamard transform per power-of-two block
//! (`_block_decompose`). `W_rht` comes from one of two codebooks:
//!
//! - `e8_shell` (`E8RHTLinear.dequantize`): `Σ_k c_k · CB[Qidxs_k]`, 8 columns per E8 index, with
//!   `c = [1, inv_resid_scale, inv_resid_scale2, inv_resid_scale3]`. `CB` is the 65536-entry
//!   `e8_codebook.pt` every checkpoint ships: the reference's unstable argsort scrambles the order
//!   within each shell, so the table can't be re-enumerated. 3 bpw layers index its first 256 entries
//!   (`make_small`), a prefix of the same table.
//! - `trellis` (`glq/trellis.py` `decode_layer`, QTIP bitshift trellis, L = 16, 16×16 tiles):
//!   `trellis_packed (+ inv_resid_scale2 · trellis_packed2)`, lookup-free `3inst` (V = 1, K = bpw)
//!   stored in the kernel (MMA-fragment) layout. `hyb` (V = 2, tlut) has no published checkpoint.

use crate::model::{decode_dtype, ModelManifest, VirtualTensor};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

/// Manifest metadata key holding the `e8_codebook.pt` path [`prepare`] found.
const CODEBOOK_KEY: &str = "glq.e8_codebook";

/// Exposes every GLQ module (`X.Qidxs`, `X.Qidxs_e8p` or `X.trellis_packed`, beside `X.SV`) as
/// `X.weight` with its RHT-domain shape `[len(SU), len(SV)]` (embeddings: `[len(Wscale), len(SV)]`);
/// [`prepare`] narrows padded shapes to the real ones.
pub(super) fn alias(m: &mut ModelManifest) {
    let add: Vec<VirtualTensor> = m
        .tensors
        .iter()
        .filter_map(|(name, t)| {
            let stem = name
                .strip_suffix(".Qidxs")
                .or_else(|| name.strip_suffix(".Qidxs_e8p"))
                .or_else(|| name.strip_suffix(".trellis_packed"))?;
            let len = |s: &str| -> Option<usize> {
                Some(
                    m.tensors
                        .get(&format!("{stem}.{s}"))?
                        .shape
                        .iter()
                        .product(),
                )
            };
            let rows = len("Wscale").filter(|&r| r > 1).or_else(|| len("SU"))?;
            Some(VirtualTensor {
                name: format!("{stem}.weight"),
                shape: vec![rows, len("SV")?],
                ..t.clone()
            })
        })
        .collect();
    for t in add {
        m.tensors.insert(t.name.clone(), t);
    }
}

/// Run on a GLQ manifest before reading weights, with the checkpoint's `config.json`:
/// legacy `e8_shell` layers pad to powers of two without storing the true size, so it comes from the
/// config by projection name (block-diagonal and trellis layers are unpadded), and the shipped
/// `e8_codebook.pt` beside the config is located for [`dequantize`].
pub fn prepare(m: &mut ModelManifest, config: Option<&Path>) -> Result<(), String> {
    let stems: Vec<String> = m
        .tensors
        .keys()
        .filter_map(|n| n.strip_suffix(".Qidxs").map(str::to_owned))
        .collect();
    if stems.is_empty() {
        return Ok(());
    }
    let cb = config
        .and_then(Path::parent)
        .unwrap_or(Path::new("."))
        .join("e8_codebook.pt");
    if !cb.exists() {
        return Err(format!(
            "GLQ e8_shell checkpoint needs its e8_codebook.pt beside the weights ({} missing)",
            cb.display()
        ));
    }
    m.metadata
        .insert(CODEBOOK_KEY.into(), cb.to_string_lossy().into_owned());
    let cfg: Value = config
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let cfg = cfg.get("text_config").unwrap_or(&cfg);
    let get = |k: &str| cfg[k].as_u64().map(|v| v as usize);
    let hidden = get("hidden_size");
    let heads = get("num_attention_heads");
    let hd = get("head_dim").or_else(|| Some(hidden? / heads?));
    let q = heads.zip(hd).map(|(a, b)| a * b);
    let kv = get("num_key_value_heads")
        .or(heads)
        .zip(hd)
        .map(|(a, b)| a * b);
    let inter = get("intermediate_size");
    // Per-layer embeddings: `[vocab, num_layers · hidden_size_per_layer_input]`.
    let ple = get("hidden_size_per_layer_input")
        .zip(get("num_hidden_layers"))
        .map(|(a, b)| a * b);
    for stem in stems {
        let Some(w) = m.tensors.get_mut(&format!("{stem}.weight")) else {
            return Err(format!("GLQ {stem}: Qidxs without SU/SV"));
        };
        let (mp, np) = (w.shape[0], w.shape[1]);
        if !(mp.is_power_of_two() && np.is_power_of_two()) {
            continue;
        }
        let proj = stem.rsplit('.').next().unwrap_or("");
        let dims = match proj {
            "q_proj" => q.zip(hidden),
            "k_proj" | "v_proj" => kv.zip(hidden),
            "o_proj" => hidden.zip(q),
            "gate_proj" | "up_proj" => inter.zip(hidden),
            "down_proj" => hidden.zip(inter),
            "embed_tokens_per_layer" => Some(mp).zip(ple),
            _ => None,
        };
        let (out, inp) = dims.ok_or_else(|| {
            format!("GLQ {stem}: padded [{mp}, {np}] and no config size for '{proj}'")
        })?;
        if out.next_power_of_two() != mp || inp.next_power_of_two() != np {
            return Err(format!(
                "GLQ {stem}: config size [{out}, {inp}] does not pad to [{mp}, {np}]"
            ));
        }
        w.shape = vec![out, inp];
    }
    Ok(())
}

/// The `[65536, 8]` table in `e8_codebook.pt`, cached for the last path read.
fn codebook(path: &str) -> Result<Arc<Vec<f32>>, String> {
    static CB: parking_lot::Mutex<Option<(PathBuf, Arc<Vec<f32>>)>> = parking_lot::Mutex::new(None);
    let mut slot = CB.lock();
    if let Some((_, cb)) = slot.as_ref().filter(|(p, _)| p.as_os_str() == path) {
        return Ok(cb.clone());
    }
    let (m, data) = crate::model::loaders::read_checkpoint(Path::new(path))?;
    let t = m
        .tensors
        .get("codebook")
        .ok_or_else(|| format!("{path}: no `codebook` tensor"))?;
    if t.shape.len() != 2 || t.shape[1] != 8 {
        return Err(format!("{path}: codebook shape {:?}, want [K, 8]", t.shape));
    }
    let cb = Arc::new(m.extract_tensor_f32(&data, "codebook")?);
    *slot = Some((PathBuf::from(path), cb.clone()));
    Ok(cb)
}

/// Dense f32 `[out, in]` for an aliased GLQ `X.weight`; `None` for other tensors.
pub fn dequantize(m: &ModelManifest, data: &[u8], name: &str) -> Result<Option<Vec<f32>>, String> {
    decode(m, data, name, |x| x)
}

/// [`dequantize`] straight to bf16, for the native loaders' owned [`crate::model::weights::Mat`]s
/// (a per-layer embedding table can be 5.6 GB as bf16, 11 GB as f32).
pub fn dequantize_bf16(
    m: &ModelManifest,
    data: &[u8],
    name: &str,
) -> Result<Option<Vec<u16>>, String> {
    decode(m, data, name, crate::model::f32_to_bf16)
}

fn decode<T: Copy + Default + Send + Sync>(
    m: &ModelManifest,
    data: &[u8],
    name: &str,
    conv: fn(f32) -> T,
) -> Result<Option<Vec<T>>, String> {
    let Some(stem) = name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let t = |s: &str| m.tensors.get(&format!("{stem}.{s}"));
    let (shell, trellis, e8p) = (t("Qidxs"), t("trellis_packed"), t("Qidxs_e8p"));
    if shell.is_none() && trellis.is_none() && e8p.is_none() {
        return Ok(None);
    }
    let err = |e: String| format!("{name}: {e}");
    let get = |s: &str| -> Result<Option<Vec<f32>>, String> {
        t(s).map(|t| decode_dtype(super::bytes(data, t)?, &t.dtype).map_err(err))
            .transpose()
    };
    let scalar = |s: &str| -> Result<f32, String> {
        Ok(get(s)?.and_then(|v| v.first().copied()).unwrap_or(0.0))
    };
    let sv = get("SV")?.ok_or_else(|| err("missing SV".into()))?;
    let wscale = get("Wscale")?.ok_or_else(|| err("missing Wscale".into()))?;
    // E8RHTEmbedding (PLE): per-row Wscale and a right-side-only RHT (`SU` unused).
    let emb = wscale.len() > 1;
    let su = if emb {
        Vec::new()
    } else {
        get("SU")?.ok_or_else(|| err("missing SU".into()))?
    };
    let (mp, np) = (if emb { wscale.len() } else { su.len() }, sv.len());
    let (out, inp) = (m.tensors[name].shape[0], m.tensors[name].shape[1]);
    if out > mp || inp > np {
        return Err(err(format!("[{out}, {inp}] exceeds RHT size [{mp}, {np}]")));
    }
    if emb {
        return embedding(&t, m, data, &sv, &wscale, out, inp, conv)
            .map(Some)
            .map_err(err);
    }
    let mut w = match (shell, trellis) {
        _ if e8p.is_some() => {
            e8p_decode(&t, data, mp, np, scalar("inv_resid_scale")?).map_err(err)?
        }
        (Some(q1), _) => {
            let path = m.metadata.get(CODEBOOK_KEY).ok_or_else(|| {
                err("GLQ e8_shell weights read without glq::prepare (no codebook)".into())
            })?;
            let cb = codebook(path).map_err(err)?;
            // Stage k counts when its indices exist and its scale is non-zero (2 bpw stores none).
            let mut stages = vec![(q1, 1.0)];
            for (q, s) in [
                ("Qidxs2", "inv_resid_scale"),
                ("Qidxs3", "inv_resid_scale2"),
                ("Qidxs4", "inv_resid_scale3"),
            ] {
                match (t(q), scalar(s)?) {
                    (Some(qt), c) if c != 0.0 => stages.push((qt, c)),
                    _ => break,
                }
            }
            let mut w = vec![0f32; mp * np];
            for (qt, c) in stages {
                if qt.dtype != "I16" || qt.shape != [mp, np / 8] || np % 8 != 0 {
                    return Err(err(format!(
                        "{} {} {:?} for [{mp}, {np}]",
                        qt.name, qt.dtype, qt.shape
                    )));
                }
                let raw = super::bytes(data, qt)?;
                for (g, dst) in w.chunks_exact_mut(8).enumerate() {
                    let i = u16::from_le_bytes([raw[2 * g], raw[2 * g + 1]]) as usize;
                    let e = cb.get(i * 8..i * 8 + 8).ok_or_else(|| {
                        err(format!(
                            "index {i} outside the {}-entry codebook",
                            cb.len() / 8
                        ))
                    })?;
                    for (d, v) in dst.iter_mut().zip(e) {
                        *d += c * v;
                    }
                }
            }
            w
        }
        (None, Some(p1)) => {
            if get("tlut")?.is_some_and(|l| !l.is_empty()) {
                return Err(err(
                    "GLQ trellis variant 'hyb' (tlut) is not supported (3inst is)".into(),
                ));
            }
            let mut w = trellis_decode(super::bytes(data, p1)?, &p1.shape, mp, np).map_err(err)?;
            if let Some(p2) = t("trellis_packed2").filter(|p| p.shape.iter().product::<usize>() > 0)
            {
                let c = scalar("inv_resid_scale2")?;
                if c == 0.0 {
                    return Err(err("trellis_packed2 with a zero inv_resid_scale2".into()));
                }
                let w2 = trellis_decode(super::bytes(data, p2)?, &p2.shape, mp, np).map_err(err)?;
                w.iter_mut().zip(w2).for_each(|(a, b)| *a += c * b);
            }
            w
        }
        (None, None) => unreachable!(),
    };
    let (bm, bn) = (blocks(mp), blocks(np));
    // (Wscale · W_rht) · H_n · diag(SV), row by row.
    crate::model::par::par_rows(&mut w, np, &|_, row| {
        fht(row, &bn);
        for (x, s) in row.iter_mut().zip(&sv) {
            *x *= s * wscale[0];
        }
    });
    // diag(SU) · H_m along columns: transpose the kept columns, transform rows, transpose back.
    let mut tr = vec![0f32; inp * mp];
    crate::model::par::par_rows(&mut tr, mp, &|c, col| {
        for (r, x) in col.iter_mut().enumerate() {
            *x = w[r * np + c];
        }
        fht(col, &bm);
    });
    drop(w);
    Ok(Some(
        (0..out * inp)
            .map(|i| conv(su[i / inp] * tr[(i % inp) * mp + i / inp]))
            .collect(),
    ))
}

/// `_dequant_embedding_rows` for every row: `(Σ_k irs_k[r] · CB[Qidxs_k[r]]) · Wscale[r]`, full FHT,
/// `· SV`, cropped to `inp`. Rows decode independently, straight into the output type.
#[allow(clippy::too_many_arguments)]
fn embedding<'a, T: Copy + Default + Send + Sync>(
    t: &dyn Fn(&str) -> Option<&'a VirtualTensor>,
    m: &ModelManifest,
    data: &[u8],
    sv: &[f32],
    wscale: &[f32],
    out: usize,
    inp: usize,
    conv: fn(f32) -> T,
) -> Result<Vec<T>, String> {
    let np = sv.len();
    let q1 =
        t("Qidxs").ok_or("GLQ embedding without Qidxs (only e8_shell tables are supported)")?;
    let path = m
        .metadata
        .get(CODEBOOK_KEY)
        .ok_or("GLQ e8_shell embedding read without glq::prepare (no codebook)")?;
    let cb = codebook(path)?;
    let mut stages = vec![(super::bytes(data, q1)?, vec![1.0f32; wscale.len()])];
    if let (Some(q2), Some(irs)) = (t("Qidxs2"), t("inv_resid_scale")) {
        let irs = decode_dtype(super::bytes(data, irs)?, &irs.dtype)?;
        if irs.iter().any(|&x| x != 0.0) {
            stages.push((super::bytes(data, q2)?, irs));
        }
    }
    for (raw, irs) in &stages {
        if raw.len() != wscale.len() * np / 4 || irs.len() != wscale.len() || !np.is_power_of_two()
        {
            return Err(format!(
                "embedding Qidxs {} bytes for [{}, {np}]",
                raw.len(),
                wscale.len()
            ));
        }
    }
    let mut res = vec![T::default(); out * inp];
    let threads = crate::model::par::threads();
    let chunk = out.div_ceil(threads);
    let ptr = res.as_mut_ptr() as usize;
    crate::model::par::par_for(threads, &|ti| {
        let mut row = vec![0f32; np];
        for r in ti * chunk..((ti + 1) * chunk).min(out) {
            row.iter_mut().for_each(|x| *x = 0.0);
            for (raw, irs) in &stages {
                for g in 0..np / 8 {
                    let o = (r * np / 8 + g) * 2;
                    let i = u16::from_le_bytes([raw[o], raw[o + 1]]) as usize;
                    for (d, v) in row[g * 8..][..8].iter_mut().zip(&cb[i * 8..i * 8 + 8]) {
                        *d += irs[r] * v;
                    }
                }
            }
            fht(&mut row, &[np]);
            // SAFETY: each task writes only rows [ti·chunk, (ti+1)·chunk) of `res`.
            let dst = unsafe { std::slice::from_raw_parts_mut((ptr as *mut T).add(r * inp), inp) };
            for ((d, x), s) in dst.iter_mut().zip(&row).zip(sv) {
                *d = conv(x * wscale[r] * s);
            }
        }
    });
    Ok(res)
}

/// `E8PCodebook` (QuIP# E8P12, `get_full_grid(get_packed_abs_grid())`): 65536 × 8.
static E8P: LazyLock<Vec<[f32; 8]>> = LazyLock::new(|| {
    // d8abs: |v| for v ∈ (Z + ½)^8, |v_i| ≤ 3.5, Σv even, ‖v‖² ≤ 10, unique and sorted (doubled units).
    let mut abs = std::collections::BTreeSet::new();
    let mut idx = [0i32; 8];
    loop {
        let u: [i32; 8] = idx.map(|i| 2 * i - 7); // 2v, v ∈ {-3.5..3.5}
        if (u.iter().sum::<i32>() / 2).rem_euclid(2) == 0
            && u.iter().map(|x| x * x).sum::<i32>() <= 40
        {
            abs.insert(u.map(i32::abs));
        }
        let Some(d) = (0..8).rev().find(|&d| idx[d] < 7) else {
            break;
        };
        idx[d] += 1;
        idx[d + 1..].iter_mut().for_each(|x| *x = 0);
    }
    const NORM12: [[i32; 8]; 29] = [
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ];
    // get_packed_abs_grid: columns [0,2,4,6,1,3,5,7], last negated when Σ is odd, nibbles 2v + 8.
    let packed: Vec<u32> = abs
        .into_iter()
        .chain(NORM12)
        .map(|a| {
            let mut c: [i32; 8] = std::array::from_fn(|i| a[[0, 2, 4, 6, 1, 3, 5, 7][i]]);
            if (c.iter().sum::<i32>() / 2) % 2 == 1 {
                c[7] = -c[7];
            }
            (0..8).fold(0u32, |acc, i| acc | ((c[i] + 8) as u32) << (4 * i))
        })
        .collect();
    assert_eq!(packed.len(), 256);
    // get_full_grid: index = abs << 8 | signs; bit 0 of signs absorbs the sign parity, ±¼ shift.
    const SHUFFLE: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    (0..1u32 << 16)
        .map(|c| {
            let parity = (c & 255).count_ones() & 1;
            let signs = (c & 255) ^ parity;
            let code = packed[(c >> 8) as usize];
            std::array::from_fn(|i| {
                let ii = SHUFFLE[i];
                let mut v = (((code >> (4 * ii)) & 15) as f32 - 8.0) * 0.5;
                if (signs >> ii) & 1 == 1 {
                    v = -v;
                }
                v + if parity == 1 { -0.25 } else { 0.25 }
            })
        })
        .collect()
});

/// `E81BCodebook` (`get_e81bgrid`): the norm² ≤ 2 even-sum E8 points (integer then half-integer
/// coordinates, lexicographic) followed by ±2·e_i (2e_0..2e_7, -2e_0..-2e_6).
static E81B: LazyLock<Vec<[f32; 8]>> = LazyLock::new(|| {
    let mut g = Vec::new();
    for half in [0, 1] {
        let mut idx = [0i32; 8];
        loop {
            let u: [i32; 8] = idx.map(|i| 2 * (i - 4) + half); // 2v
            if (u.iter().sum::<i32>() / 2).rem_euclid(2) == 0
                && u.iter().map(|x| x * x).sum::<i32>() <= 8
            {
                g.push(u.map(|x| x as f32 * 0.5));
            }
            let Some(d) = (0..8).rev().find(|&d| idx[d] < 7) else {
                break;
            };
            idx[d] += 1;
            idx[d + 1..].iter_mut().for_each(|x| *x = 0);
        }
    }
    for s in [2.0f32, -2.0] {
        for i in 0..if s > 0.0 { 8 } else { 7 } {
            let mut v = [0f32; 8];
            v[i] = s;
            g.push(v);
        }
    }
    g
});

/// RHT-domain `[m, n]` of an e8p layer: `E8P[Qidxs_e8p] + inv_resid_scale · (E8P[Qidxs2_e8p] |
/// E81B[Qidxs2_e81b])`. `Qidxs_e8p` is `maybe_pack_idxs`'s int64 `[m/16, n/64, 8, 4]`: entry
/// `(A, C, r, s)` packs the 2×2 index block at rows `2a, 2a+1`, column groups `2c, 2c+1`
/// (`a = 8A + r`, `c = 4C + s`), abs bytes in the low word and sign bytes interleaved in the high.
fn e8p_decode<'a>(
    t: &dyn Fn(&str) -> Option<&'a VirtualTensor>,
    data: &[u8],
    m: usize,
    n: usize,
    inv_rs: f32,
) -> Result<Vec<f32>, String> {
    let present = |s: &str| t(s).filter(|x| x.shape.iter().product::<usize>() > 0);
    if let Some(x) = ["Qidxs3_e8p", "Qidxs3_e81b", "Qidxs4_e8p", "Qidxs4_e81b"]
        .into_iter()
        .find(|s| present(s).is_some())
    {
        return Err(format!("e8p {x} (5–8 bpw) is not supported (2–4 bpw are)"));
    }
    if m % 16 != 0 || n % 64 != 0 {
        return Err(format!(
            "e8p RHT size [{m}, {n}] not a multiple of [16, 64]"
        ));
    }
    let mut w = vec![0f32; m * n];
    let mut add_e8p = |q: &VirtualTensor, scale: f32| -> Result<(), String> {
        let raw = super::bytes(data, q)?;
        if q.dtype != "I64" || raw.len() != m * n / 4 {
            return Err(format!(
                "{} {} {:?} for [{m}, {n}]",
                q.name, q.dtype, q.shape
            ));
        }
        let cg = n / 64;
        for (e, b) in raw.chunks_exact(8).enumerate() {
            let v = u64::from_le_bytes(b.try_into().unwrap());
            let (s, r, cc, aa) = (e % 4, (e / 4) % 8, (e / 32) % cg, e / 32 / cg);
            let (a, c) = (aa * 8 + r, cc * 4 + s);
            for k in 0..4 {
                let abs = (v >> (8 * k)) & 255;
                let sign = (0..8).fold(0u64, |acc, j| acc | ((v >> (32 + 4 * j + k)) & 1) << j);
                let (row, g) = (2 * a + k % 2, 2 * c + k / 2);
                for (d, x) in w[row * n + g * 8..][..8]
                    .iter_mut()
                    .zip(&E8P[(abs << 8 | sign) as usize])
                {
                    *d += scale * x;
                }
            }
        }
        Ok(())
    };
    add_e8p(t("Qidxs_e8p").unwrap(), 1.0)?;
    let (q2, q2b) = (present("Qidxs2_e8p"), present("Qidxs2_e81b"));
    if (q2.is_some() || q2b.is_some()) && inv_rs == 0.0 {
        return Err("e8p residual stage with a zero inv_resid_scale".into());
    }
    if let Some(q) = q2 {
        add_e8p(q, inv_rs)?;
    }
    if let Some(q) = q2b {
        // `pack_e81b`: byte i of `[r, j]` is the index of column group 8j + i.
        let raw = super::bytes(data, q)?;
        if q.dtype != "I64" || raw.len() != m * n / 8 {
            return Err(format!(
                "{} {} {:?} for [{m}, {n}]",
                q.name, q.dtype, q.shape
            ));
        }
        for (g, &i) in raw.iter().enumerate() {
            let cw = E81B
                .get(i as usize)
                .ok_or_else(|| format!("E81B index {i} outside {} entries", E81B.len()))?;
            for (d, x) in w[g * 8..][..8].iter_mut().zip(cw) {
                *d += inv_rs * x;
            }
        }
    }
    Ok(w)
}

/// `decode_layer(cb, packed, m, n, has_kernel=True)`: the normalized `[m, n]` trellis weight.
/// `packed` is int16 `[(m/16)·(n/16), 16·K]`, one tail-biting bitstream per 16×16 tile.
fn trellis_decode(raw: &[u8], shape: &[usize], m: usize, n: usize) -> Result<Vec<f32>, String> {
    let tiles = (m / 16) * (n / 16);
    let k = shape.get(1).copied().unwrap_or(0) / 16;
    if m % 32 != 0
        || n % 32 != 0
        || shape != [tiles, 16 * k]
        || k == 0
        || raw.len() != tiles * 32 * k
    {
        return Err(format!("trellis_packed {shape:?} for [{m}, {n}]"));
    }
    // Undo the MMA-fragment byte flip: kernel[i] = natural[perm[i]] with perm =
    // arange.view(-1,2).flip(-1).reshape(m/32,2,n/32,2,32,K).permute(0,2,4,3,1,5).flip(-1).
    let (m2, n2) = (m / 32, n / 32);
    let mut nat = vec![0u8; raw.len()];
    let mut i = 0;
    for a in 0..m2 {
        for c in 0..n2 {
            for e in 0..32 {
                for d in 0..2 {
                    for b in 0..2 {
                        for f in (0..k).rev() {
                            let src = ((((a * 2 + b) * n2 + c) * 2 + d) * 32 + e) * k + f;
                            nat[src ^ 1] = raw[i];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    // _PERMUTE = arange(256).reshape(2,8,2,4,2).permute(1,3,2,0,4): decoded step j is tile element perm[j].
    let mut perm = [0usize; 256];
    let mut j = 0;
    for b1 in 0..8 {
        for d3 in 0..4 {
            for c2 in 0..2 {
                for a0 in 0..2 {
                    for e4 in 0..2 {
                        perm[j] = a0 * 128 + b1 * 16 + c2 * 8 + d3 * 2 + e4;
                        j += 1;
                    }
                }
            }
        }
    }
    // decode_3inst: two masked f16 halves of s·89226354 + 64248484, summed in f16 (`lut`).
    static LUT: LazyLock<Vec<f32>> = LazyLock::new(|| {
        (0..1u32 << 16)
            .map(|s| {
                let r = (s.wrapping_mul(89226354).wrapping_add(64248484) & 0x8FFF_8FFF) ^ 996162400;
                let h = |b: u32| crate::model::f16_to_f32(b as u16);
                // torch adds the halves in f16: round the exact f32 sum to 10 mantissa bits, ties
                // to even (both halves have exponent 12..15, so the sum stays f16-normal).
                let b = (h(r >> 16) + h(r & 0xFFFF)).to_bits();
                f32::from_bits((b + 0xFFF + ((b >> 13) & 1)) & !0x1FFF)
            })
            .collect()
    });
    let words = 16 * k;
    let mut out = vec![0f32; m * n];
    let nt = n / 16;
    // One band of 16 rows per tile row; tiles in a band are consecutive in `nat`.
    crate::model::par::par_rows(&mut out, 16 * n, &|ti, band| {
        let mut wv = vec![0u16; words + 1];
        for tj in 0..nt {
            let bytes = &nat[(ti * nt + tj) * words * 2..][..words * 2];
            for (w, b) in wv.iter_mut().zip(bytes.chunks_exact(2)) {
                *w = u16::from_le_bytes([b[0], b[1]]);
            }
            wv[words] = wv[0]; // tail-biting wrap: the last windows read the opening bits
            for (st, &e) in perm.iter().enumerate() {
                let p = st * k;
                let win = (wv[p / 16] as u32) << 16 | wv[p / 16 + 1] as u32;
                let s = (win << (p % 16)) >> 16;
                band[(e / 16) * n + tj * 16 + e % 16] = LUT[s as usize];
            }
        }
    });
    Ok(out)
}

/// `_block_decompose`: descending powers of two summing to `n`.
fn blocks(mut n: usize) -> Vec<usize> {
    let mut b = Vec::new();
    while n > 0 {
        let p = 1 << (usize::BITS - 1 - n.leading_zeros());
        b.push(p);
        n -= p;
    }
    b
}

/// Orthonormal Walsh–Hadamard transform (Sylvester order) on each block of `x`.
fn fht(x: &mut [f32], blocks: &[usize]) {
    let mut off = 0;
    for &n in blocks {
        let v = &mut x[off..off + n];
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (v[j], v[j + h]);
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
            h *= 2;
        }
        let s = 1.0 / (n as f32).sqrt();
        v.iter_mut().for_each(|e| *e *= s);
        off += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::prequant::tests::{snapshot, HEAVY};

    /// `E8ShellCodebook.__init__` enumerates E8 with ‖v‖² ≤ 12.5 and asserts the shell counts
    /// 1, 240, 2160, 6720, 17520, 30240, 60480; the shell-sorted first 65536 are the table.
    #[test]
    fn e8_shell_enumeration_counts() {
        let mut counts = [0usize; 7];
        for (lo, k) in [(-6i32, 7usize), (-5, 6)] {
            for code in 0..k.pow(8) {
                let u: Vec<i32> = (0..8)
                    .map(|i| lo + 2 * ((code / k.pow(i)) % k) as i32)
                    .collect();
                let n2: i32 = u.iter().map(|x| x * x).sum();
                if n2 <= 50 && u.iter().sum::<i32>().rem_euclid(4) == 0 {
                    counts[n2 as usize / 8] += 1;
                }
            }
        }
        assert_eq!(counts, [1, 240, 2160, 6720, 17520, 30240, 60480]);
        assert_eq!(counts[..6].iter().sum::<usize>() + 8655, 65536);
    }

    /// E8P12 (QuIP#): 65536 distinct points, each `x ∓ ¼` an E8 point of the even-sum half-integer
    /// coset with norm² ≤ 12 (the 227 norm-≤10 abs patterns plus 29 norm-12 ones); E81B holds
    /// the 241 E8 points of norm² ≤ 2 plus 15 axis vectors of norm 4.
    #[test]
    fn e8p_and_e81b_grids() {
        let mut seen = std::collections::HashSet::new();
        for v in E8P.iter() {
            let q: [i32; 8] = v.map(|x| (x * 4.0) as i32);
            assert!(seen.insert(q));
            let ok = |s: i32| {
                let u: Vec<i32> = q.iter().map(|x| x - s).collect(); // 4·(x - s/4)
                u.iter().all(|x| x.rem_euclid(4) == 2)
                    && (u.iter().sum::<i32>() / 4).rem_euclid(2) == 0
                    && u.iter().map(|x| x * x).sum::<i32>() <= 12 * 16
            };
            assert!(ok(1) || ok(-1), "{v:?}");
        }
        assert_eq!(seen.len(), 65536);
        assert_eq!(E81B.len(), 256);
        let norms: Vec<f32> = E81B.iter().map(|v| v.iter().map(|x| x * x).sum()).collect();
        assert_eq!(norms.iter().filter(|&&n| n == 0.0).count(), 1);
        assert_eq!(norms.iter().filter(|&&n| n == 2.0).count(), 240);
        assert_eq!(norms.iter().filter(|&&n| n == 4.0).count(), 15);
    }

    /// The shipped table is `E8ShellCodebook`: 65536 distinct E8 points (all-integer or all-half-integer
    /// coordinates, even coordinate sum) filling shells 0–5 (the reference's asserted counts 1, 240, 2160,
    /// 6720, 17520, 30240 at norms 0–10) in norm order, then 8655 vectors of shell 6 (norm 12).
    #[test]
    fn e8_shell_codebook_shells() {
        let p = snapshot("xv0y5ncu/SmolLM2-135M-Instruct-GLQ-4bpw").join("e8_codebook.pt");
        if !p.exists() {
            eprintln!("SKIP: {} not downloaded", p.display());
            return;
        }
        let cb = codebook(p.to_str().unwrap()).unwrap();
        assert_eq!(cb.len(), 65536 * 8);
        let mut counts = [0usize; 7];
        let mut last = 0;
        let mut seen = std::collections::HashSet::new();
        for v in cb.chunks_exact(8) {
            let u: Vec<i32> = v.iter().map(|x| (x * 2.0) as i32).collect();
            assert!(v.iter().zip(&u).all(|(x, u)| *u as f32 == x * 2.0));
            assert!(
                u.iter().all(|x| x % 2 == 0) || u.iter().all(|x| x % 2 != 0),
                "{v:?}"
            );
            assert_eq!(u.iter().sum::<i32>().rem_euclid(4), 0, "{v:?}");
            let n2 = u.iter().map(|x| x * x).sum::<i32>() as usize / 4;
            assert!(n2 >= last, "not sorted by norm");
            last = n2;
            counts[n2 / 2] += 1;
            assert!(seen.insert(u));
        }
        assert_eq!(counts, [1, 240, 2160, 6720, 17520, 30240, 8655]);
    }

    /// Every dequantized projection vs the unquantized source model (all printed, worst asserted),
    /// then a greedy completion through the normal loader.
    fn glq_repo(repo: &str, reference: &str, min_cos: f64) {
        let (q, r) = (snapshot(repo), snapshot(reference));
        if !q.join("model.safetensors").exists() || !r.join("model.safetensors").exists() {
            eprintln!("SKIP: {repo} or {reference} not in disk/");
            return;
        }
        let _guard = HEAVY.lock();
        let cfg = q.join("config.json");
        crate::model::prequant::check(Some(&cfg), &q).unwrap();
        let (mut qm, qd) =
            crate::model::loaders::read_checkpoint(&q.join("model.safetensors")).unwrap();
        prepare(&mut qm, Some(&cfg)).unwrap();
        let (rm, rd) =
            crate::model::loaders::read_checkpoint(&r.join("model.safetensors")).unwrap();
        // The native loaders' view: mmap + owned bf16 per `mat` (`prequant::open_weights`).
        let nw =
            crate::model::prequant::open_weights(&q.join("model.safetensors"), Some(&cfg), false)
                .unwrap();
        let mut names: Vec<&String> = qm
            .tensors
            .keys()
            .filter(|n| {
                n.ends_with(".weight") && qm.tensors.contains_key(&n.replace(".weight", ".SU"))
            })
            .collect();
        names.sort();
        assert!(!names.is_empty());
        let mut worst = (1.0f64, String::new());
        for name in names {
            let a = qm.extract_tensor_f32(&qd, name).unwrap();
            let b = rm.extract_tensor_f32(&rd, name).unwrap();
            assert_eq!(a.len(), b.len(), "{name}");
            let native = nw.mat(name).unwrap().to_f32();
            assert_eq!(native.len(), b.len(), "{name} (native)");
            let (cos, cos_bf16) = (cosine(&a, &b), cosine(&native, &b));
            println!("{repo}\t{name}\t{cos:.5}\tbf16 {cos_bf16:.5}");
            if cos.min(cos_bf16) < worst.0 {
                worst = (cos.min(cos_bf16), name.clone());
            }
        }
        println!("{repo}: worst cos {:.5} ({})", worst.0, worst.1);
        assert!(
            worst.0 >= min_cos,
            "{repo}: {} cosine {} < {min_cos}",
            worst.1,
            worst.0
        );
        drop((qm, qd, rm, rd, nw));

        let artifacts = crate::model::hub::ModelArtifacts {
            model_id: repo.into(),
            config_path: Some(cfg),
            tokenizer_path: Some(q.join("tokenizer.json")),
            weights_path: Some(q.join("model.safetensors")),
            ..Default::default()
        };
        let out = crate::arch::testing::greedy_text(&artifacts, "The capital of France is", 12);
        println!("{repo}: {out:?}");
        assert!(out.contains(" Paris") && !out.contains('Ġ'), "{repo}: {out:?}");
    }

    const SMOL360: &str = "HuggingFaceTB/SmolLM2-360M-Instruct";

    #[test]
    fn glq_e8_shell_4bpw() {
        glq_repo(
            "xv0y5ncu/SmolLM2-135M-Instruct-GLQ-4bpw",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            0.98,
        );
    }

    #[test]
    fn glq_e8_shell_block_diagonal_4bpw() {
        glq_repo(
            "xv0y5ncu/SmolLM2-360M-Instruct-GLQ-block-diagonal-4bpw",
            SMOL360,
            0.98,
        );
    }

    #[test]
    fn glq_trellis_3inst_3bpw() {
        glq_repo(
            "xv0y5ncu/SmolLM2-360M-Instruct-GLQ-trellis-3inst-3bpw",
            SMOL360,
            0.95,
        );
    }

    #[test]
    fn glq_trellis_3inst_4bpw() {
        glq_repo(
            "xv0y5ncu/SmolLM2-360M-Instruct-GLQ-trellis-3inst-4bpw",
            SMOL360,
            0.98,
        );
    }

    #[test]
    fn glq_trellis_3inst_5bpw_two_stage() {
        glq_repo(
            "xv0y5ncu/SmolLM2-360M-Instruct-GLQ-trellis-3inst-5bpw",
            SMOL360,
            0.99,
        );
    }

    /// Gemma-4-12B e8p 3 bpw (E8P + E81B residual, block-diagonal RHT) vs google/gemma-4-12B-it on
    /// layers 0, 24, 47, read as the native loaders do (`Weights::mat`, owned bf16). Both files are
    /// range-extracted subsets (`disk/glq-extract/`).
    #[test]
    fn glq_e8p_3bpw_gemma4_12b_layers() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("disk/glq-extract");
        let (q, r) = (
            dir.join("gemma-4-12B-it-GLQ-3bpw-e8p-bd.safetensors"),
            dir.join("gemma-4-12B-it.safetensors"),
        );
        if !q.exists() || !r.exists() {
            eprintln!("SKIP: {} or {} missing", q.display(), r.display());
            return;
        }
        let _guard = HEAVY.lock();
        let qw = crate::model::weights::Weights::open(&q).unwrap();
        let (rm, rd) = crate::model::loaders::read_checkpoint(&r).unwrap();
        let mut names: Vec<&String> = rm
            .tensors
            .keys()
            .filter(|n| qw.has(&n.replace(".weight", ".SV")))
            .collect();
        names.sort();
        assert!(names.len() >= 15, "{} quantized projections", names.len());
        let mut worst = 1.0f64;
        for name in names {
            let a = qw.mat(name).unwrap().to_f32();
            let b = rm.extract_tensor_f32(&rd, name).unwrap();
            assert_eq!(a.len(), b.len(), "{name}");
            let cos = cosine(&a, &b);
            println!("gemma-4-12B e8p\t{name}\t{cos:.5}");
            worst = worst.min(cos);
        }
        println!("gemma-4-12B e8p 3bpw: worst cos {worst:.5}");
        assert!(worst >= 0.95, "worst cosine {worst}");
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }
}
