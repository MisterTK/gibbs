//! EXL3 (ExLlamaV3, `quant_method: "exl3"`) linear layers, decoded to f32 exactly as the reference
//! `LinearEXL3.get_weight_tensor` (exllamav3/modules/quant/exl3.py) does:
//!
//! `W[in, out] = diag(suh) · H · Ŵ · H · diag(svh)`, H = block-diagonal 128×128 Sylvester Hadamard / √128.
//!
//! Ŵ is stored as `trellis` `[in/16, out/16, 16·K]` int16: one 256-weight 16×16 tile per entry, a
//! tail-biting bitstream of K bits per weight (QTIP bitshift trellis). Weight `p` of a tile is the
//! procedural codebook (3INST, MCG or MUL1, chosen by an `mcg` / `mul1` marker tensor) applied to the
//! 16-bit window ending at bit `(p + 1)·K` (`exl3_dq.cuh::dq`), placed by the tensor-core permutation
//! (`quantize.py::tensor_core_perm`). Half-integer K (`16·K + 8` words, MUL1 only) alternates KA- and
//! (KA+1)-bit steps (`dq8_half`).

use crate::model::par::{par_for, OutPtr};
use crate::model::{f16_to_f32, VirtualTensor};
use std::sync::LazyLock;

const MCG: u32 = 0xCBAC_1FED;
const MUL1: u32 = 0x83DC_D12D;
const HAD: usize = 128;

/// fp16 round-to-nearest-even of `x`, kept as f32. Exact for the codebook's range: sums and fmas
/// of fp16 values in [2^-13, 16) whose f32 result is exact, so this is the kernel's single rounding.
fn round_f16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x0fff + ((b >> 13) & 1)) & !0x1fff)
}

/// `codebook.cuh::decode_3inst<cb>` of a 16-bit trellis state: cb 0 = 3INST, 1 = MCG, 2 = MUL1.
fn decode(cb: usize, state: u32) -> f32 {
    if cb == 2 {
        // dp4a byte sum on top of 0x6400 (fp16 1024.0), then one fp16 fma.
        let sum: u32 = state
            .wrapping_mul(MUL1)
            .to_le_bytes()
            .iter()
            .map(|&b| b as u32)
            .sum();
        let h = f16_to_f32((0x6400 + sum) as u16);
        return round_f16(h * f16_to_f32(0x1eee) + f16_to_f32(0xc931));
    }
    let x = match cb {
        0 => state.wrapping_mul(89_226_354).wrapping_add(64_248_484),
        _ => state.wrapping_mul(MCG),
    };
    // lop3 0x6a = (x & 0x8fff8fff) ^ 0x3b603b60, then __hadd of the two fp16 halves.
    let x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
    round_f16(f16_to_f32(x as u16) + f16_to_f32((x >> 16) as u16))
}

fn table(cb: usize) -> Vec<f32> {
    (0..1u32 << 16).map(|s| decode(cb, s)).collect()
}

/// All 65536 decoded values of each codebook.
static LUT: [LazyLock<Vec<f32>>; 3] = [
    LazyLock::new(|| table(0)),
    LazyLock::new(|| table(1)),
    LazyLock::new(|| table(2)),
];

/// Row-major `(row, col)` within a 16×16 tile of trellis position `p` (`tensor_core_perm`).
fn tile_pos(p: usize) -> (usize, usize) {
    let (t, j) = (p / 8, p % 8);
    let row = (t % 4) * 2 + (j & 1) + if j & 2 != 0 { 8 } else { 0 };
    (row, t / 4 + if j & 4 != 0 { 8 } else { 0 })
}

/// Decodes one packed tile (`words` little-endian u32) into `tile[row][col]`. `bits = (ka, half)`.
fn decode_tile(words: &[u32], ka: usize, half: bool, lut: &[f32], tile: &mut [[f32; 16]; 16]) {
    let n = words.len() * 32;
    for p in 0..256 {
        // End bit of position p's 16-bit window; the stream is MSB-first within each u32.
        let end = if half {
            (p >> 1) * (2 * ka + 1) + ka + (p & 1) * (ka + 1)
        } else {
            (p + 1) * ka
        };
        let s = (end + n - 16) % n;
        let i0 = s / 32;
        let merged = (words[i0] as u64) << 32 | words[(i0 + 1) % words.len()] as u64;
        let state = (merged >> (48 - s % 32)) as u32 & 0xffff;
        let (r, c) = tile_pos(p);
        tile[r][c] = lut[state as usize];
    }
}

/// In-place orthogonal-up-to-scale Walsh–Hadamard transform of each 128-chunk of `v`
/// (natural/Sylvester order, `H[i][j] = (-1)^popcount(i & j)`), unscaled.
fn fwht(v: &mut [f32]) {
    for blk in v.chunks_exact_mut(HAD) {
        let mut h = 1;
        while h < HAD {
            for i in (0..HAD).step_by(2 * h) {
                for j in i..i + h {
                    let (x, y) = (blk[j], blk[j + h]);
                    blk[j] = x + y;
                    blk[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }
}

/// A checkpoint tensor and its payload, by name.
pub(crate) type Lookup<'a, 'n> = &'n dyn Fn(&str) -> Option<(&'a VirtualTensor, &'a [u8])>;

/// `suh` / `svh` (fp16 scales) or the legacy packed sign bitfields `su` / `sv` (bit set = -1).
fn signs(get: Lookup, stem: &str, h: &str, n: usize) -> Result<Vec<f32>, String> {
    let v = if let Some((t, b)) = get(&format!("{stem}.{h}h")) {
        crate::model::decode_dtype(b, &t.dtype)?
    } else if let Some((_, b)) = get(&format!("{stem}.{h}")) {
        b.chunks_exact(2)
            .flat_map(|w| {
                let w = u16::from_le_bytes([w[0], w[1]]);
                (0..16).map(move |j| if w >> j & 1 == 1 { -1.0 } else { 1.0 })
            })
            .collect()
    } else {
        return Err(format!("{stem}: no {h}h / {h} tensor"));
    };
    if v.len() != n {
        return Err(format!("{stem}.{h}h has {} values, expected {n}", v.len()));
    }
    Ok(v)
}

/// Codebook of `stem`: the `mcg` / `mul1` marker (an int32 scalar holding the multiplier), else 3INST.
fn codebook(get: Lookup, stem: &str) -> Result<usize, String> {
    let mut cb = 0;
    for (i, key, mult) in [(1, "mcg", MCG), (2, "mul1", MUL1)] {
        let Some((t, b)) = get(&format!("{stem}.{key}")) else {
            continue;
        };
        if t.dtype != "I32" || b.len() != 4 || u32::from_le_bytes(b.try_into().unwrap()) != mult {
            return Err(format!(
                "{stem}.{key}: codebook marker is not the int32 {mult:#x}"
            ));
        }
        if cb != 0 {
            return Err(format!("{stem}: both mcg and mul1 codebook markers"));
        }
        cb = i;
    }
    Ok(cb)
}

/// `stem`'s EXL3 linear weight as row-major f32 `[out, in]` (the `nn.Linear` layout); `get` finds
/// `stem.trellis` and its companions (`suh`/`su`, `svh`/`sv`, `mcg`/`mul1`).
pub(crate) fn dequantize(get: Lookup, stem: &str) -> Result<Vec<f32>, String> {
    let (t, raw) =
        get(&format!("{stem}.trellis")).ok_or_else(|| format!("{stem}.trellis missing"))?;
    let [kt, nt, w] = t.shape[..] else {
        return Err(format!("{stem}.trellis: shape {:?} is not 3-D", t.shape));
    };
    let (inf, outf) = (kt * 16, nt * 16);
    let cb = codebook(get, stem)?;
    let (ka, half) = (w / 16, w % 16 == 8);
    if t.dtype != "I16" || !(1..=8).contains(&ka) || !(w % 16 == 0 || half && ka <= 3) {
        return Err(format!(
            "{stem}.trellis: {} [.., {w}] is no EXL3 bitrate (16·K int16 per tile, K 1-8 or 1.5/2.5/3.5)",
            t.dtype
        ));
    }
    if half && cb != 2 {
        return Err(format!(
            "{stem}: half-integer bitrate {ka}.5 requires the mul1 codebook"
        ));
    }
    if inf % HAD != 0 || outf % HAD != 0 {
        return Err(format!(
            "{stem}: [{outf}, {inf}] is not a multiple of the 128-wide Hadamard"
        ));
    }
    if raw.len() != kt * nt * w * 2 {
        return Err(format!(
            "{stem}.trellis: {} bytes for {:?} int16",
            raw.len(),
            t.shape
        ));
    }
    // Folds both 1/√128 Hadamard scales into suh.
    let suh: Vec<f32> = signs(get, stem, "su", inf)?
        .iter()
        .map(|s| s / HAD as f32)
        .collect();
    let svh = signs(get, stem, "sv", outf)?;
    let lut: &[f32] = &LUT[cb];
    let mut out = vec![0f32; outf * inf];
    let o = OutPtr::new(&mut out);

    // Ŵᵀ, 16 output rows per task, then the input-side Hadamard and suh on each row.
    par_for(nt, &|n| {
        // SAFETY: rows n·16..n·16+16 belong to task n.
        let rows = unsafe { o.slice(n * 16 * inf, 16 * inf) };
        let mut words = vec![0u32; w / 2];
        let mut tile = [[0f32; 16]; 16];
        for k in 0..kt {
            let src = &raw[(k * nt + n) * w * 2..][..w * 2];
            for (d, s) in words.iter_mut().zip(src.chunks_exact(4)) {
                *d = u32::from_le_bytes(s.try_into().unwrap());
            }
            decode_tile(&words, ka, half, lut, &mut tile);
            for (r, row) in tile.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    rows[c * inf + k * 16 + r] = v;
                }
            }
        }
        for row in rows.chunks_exact_mut(inf) {
            fwht(row);
            row.iter_mut().zip(&suh).for_each(|(v, s)| *v *= s);
        }
    });

    // Output-side Hadamard down each 128-row block, then svh; tasks own (block, column chunk).
    const COLS: usize = 256;
    let chunks = inf.div_ceil(COLS);
    par_for(outf / HAD * chunks, &|task| {
        let (b, c0) = (task / chunks, task % chunks * COLS);
        let len = COLS.min(inf - c0);
        // SAFETY: columns c0..c0+len of rows b·128..b·128+128 belong to this task.
        let row = |r: usize| unsafe { o.slice((b * HAD + r) * inf + c0, len) };
        let mut h = 1;
        while h < HAD {
            for i in (0..HAD).step_by(2 * h) {
                for j in i..i + h {
                    let (x, y) = (row(j), row(j + h));
                    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
                        (*x, *y) = (*x + *y, *x - *y);
                    }
                }
            }
            h *= 2;
        }
        for r in 0..HAD {
            let s = svh[b * HAD + r];
            row(r).iter_mut().for_each(|v| *v *= s);
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::QuantMode;
    use crate::model::hub::{split_revision, ModelArtifacts};
    use crate::model::{ModelManifest, UniversalModelLoader};
    use std::path::PathBuf;

    /// Each check holds a full model in memory.
    static HEAVY: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// `disk/` snapshot of `<org>/<repo>[@<revision>]`.
    fn snap(id: &str) -> PathBuf {
        let (repo, rev) = split_revision(id);
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "disk/models--{}/snapshots/{rev}",
            repo.replace('/', "--")
        ))
    }

    fn checkpoint(id: &str) -> Option<(ModelManifest, Vec<u8>)> {
        let dir = snap(id);
        let p = ["model.safetensors", "model.safetensors.index.json"]
            .map(|f| dir.join(f))
            .into_iter()
            .find(|p| p.exists())?;
        Some(crate::model::loaders::read_checkpoint(&p).unwrap())
    }

    fn artifacts(id: &str) -> ModelArtifacts {
        let dir = snap(id);
        ModelArtifacts {
            model_id: id.into(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            local_dir: Some(dir),
            ..Default::default()
        }
    }

    const LLAMA: &str = "turboderp/Llama-3.2-1B-Instruct-exl3";
    const QWEN35: &str = "Honkware/Qwen3.5-0.8B-exl3";
    const QWEN25: &str = "blockblockblock/Qwen2.5-0.5B-Instruct-exl3";
    const PROJS: [&str; 7] = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ];

    /// Per-tensor cosine of decoded EXL3 weights vs the unquantized original, printed as a table:
    /// 3INST (turboderp, one bitrate per branch, mixed 2-4 bit tensors at 2.5 bpw) and MUL1
    /// (Qwen3.5-0.8B) checkpoints. Floors per tensor bitrate: 4+ bit 0.99, 3 bit 0.98, 2 bit 0.94.
    #[test]
    fn exl3_weights_match_unquantized_original() {
        const Q3: &str = "turboderp/Qwen3-0.6B-exl3";
        #[rustfmt::skip]
        let cases = [
            (format!("{LLAMA}@4.0bpw"), "unsloth/Llama-3.2-1B-Instruct", "model", [0, 15]),
            (format!("{LLAMA}@2.5bpw"), "unsloth/Llama-3.2-1B-Instruct", "model", [0, 15]),
            (format!("{Q3}@4.0bpw"), "Qwen/Qwen3-0.6B", "model", [0, 27]),
            (format!("{Q3}@3.0bpw"), "Qwen/Qwen3-0.6B", "model", [0, 27]),
            (format!("{QWEN35}-4.0bpw"), "Qwen/Qwen3.5-0.8B", "model.language_model", [3, 23]),
            (format!("{QWEN25}-4.0bpw"), "Qwen/Qwen2.5-0.5B-Instruct", "model", [0, 23]),
            (format!("{QWEN25}-3.0bpw"), "Qwen/Qwen2.5-0.5B-Instruct", "model", [0, 23]),
        ];
        for (id, orig, prefix, layers) in cases {
            let _guard = HEAVY.lock();
            let (Some((qm, qd)), Some((om, od))) = (checkpoint(&id), checkpoint(orig)) else {
                eprintln!("SKIP: {id} or {orig} not in disk/");
                continue;
            };
            let mut pairs: Vec<(String, String)> = layers
                .iter()
                .flat_map(|l| PROJS.map(|p| format!("{prefix}.layers.{l}.{p}.weight")))
                .map(|n| (n.clone(), n))
                .collect();
            // The quantized head vs the original's (or the embedding of a tied-embedding model).
            if qm.tensors.contains_key("lm_head.trellis") {
                let head = match om.tensors.contains_key("lm_head.weight") {
                    true => "lm_head.weight".into(),
                    false => format!("{prefix}.embed_tokens.weight"),
                };
                pairs.push(("lm_head.weight".into(), head));
            }
            let mut low = Vec::new();
            for (q, o) in pairs {
                let a = UniversalModelLoader::extract_tensor_from_manifest(&qm, &qd, &q).unwrap();
                let b = UniversalModelLoader::extract_tensor_from_manifest(&om, &od, &o).unwrap();
                assert_eq!(a.len(), b.len(), "{q}");
                let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
                for (&x, &y) in a.iter().zip(&b) {
                    (dot, na, nb) = (
                        dot + x as f64 * y as f64,
                        na + x as f64 * x as f64,
                        nb + y as f64 * y as f64,
                    );
                }
                let cos = dot / (na.sqrt() * nb.sqrt());
                let bits = qm.tensors[&q.replace(".weight", ".trellis")].shape[2] as f64 / 16.0;
                eprintln!("| {id} | {q} | {bits} bit | {cos:.5} |");
                let floor = [0.94, 0.98, 0.99][(bits as usize).clamp(2, 4) - 2];
                if cos < floor {
                    low.push(format!("{q}: {bits} bit cos {cos:.5} < {floor}"));
                }
            }
            assert!(low.is_empty(), "{id}: {low:?}");
        }
    }

    /// Greedy completion through the llama-style loader (GraphLm, f32): Llama at 4 and 2.5 bpw,
    /// Qwen2.5 (q/k/v biases beside the trellis) at 4 and 3 bpw.
    #[test]
    fn tinyllm_exl3_answers_paris() {
        let ids = [
            format!("{LLAMA}@4.0bpw"),
            format!("{LLAMA}@2.5bpw"),
            format!("{QWEN25}-4.0bpw"),
            format!("{QWEN25}-3.0bpw"),
        ];
        for id in ids {
            if !snap(&id).join("model.safetensors").exists() {
                eprintln!("SKIP: {id} not in disk/");
                continue;
            }
            let _guard = HEAVY.lock();
            let llm =
                crate::model::hf_loader::load_causal_lm(&artifacts(&id), QuantMode::Uncompressed)
                    .unwrap();
            let out = llm.generate_text("The capital of France is", 8, 0.0);
            eprintln!("{id}: {out:?}");
            assert!(out.contains("Paris"), "{id}: {out:?}");
        }
    }

    /// Greedy completion through a native loader (Qwen3.5 reads bf16 `Weights::mat`) on MUL1 weights.
    #[test]
    fn qwen3_5_exl3_answers_paris() {
        let id = format!("{QWEN35}-4.0bpw");
        if !snap(&id).join("model.safetensors").exists() {
            eprintln!("SKIP: {id} not in disk/");
            return;
        }
        let _guard = HEAVY.lock();
        let lm = crate::model::native::load_native(&artifacts(&id), crate::hardware::Device::Cpu)
            .unwrap()
            .unwrap();
        let tok = lm.tokenizer();
        // The bf16 original answers the raw "The capital of France is" with prose too; ask in chat.
        let prompt = tok
            .apply_chat_template(&[("user", "What is the capital of France? Answer in one word.")]);
        let (ids, mut state) = lm.prepare(tok.encode(&prompt), &[]).unwrap();
        let mut logits = lm.forward(&ids, 0, &mut state).unwrap();
        let mut out = Vec::new();
        for pos in ids.len()..ids.len() + 6 {
            let next = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap();
            out.push(next as u32);
            logits = lm.forward(&[next as u32], pos, &mut state).unwrap();
        }
        let text = tok.decode(&out);
        eprintln!("{id}: {text:?}");
        assert!(text.contains("Paris"), "{id}: {text:?}");
    }

    /// Decodes real tiles and checks tile row 0, column 0 and the tile sum against values from a
    /// literal transcription of the CUDA path (`reconstruct_tile` → `dq` (integer K) / `dq8_half`
    /// (K + ½) → `decode_3inst<cb>` → the half2 warp shuffle), fp16 via JS `Float16Array`.
    ///
    /// Derivation of element (0, 0) = position 0 of the 3INST tile (first tile of
    /// turboderp/Llama-3.2-1B-Instruct-exl3@4.0bpw `model.layers.0.self_attn.q_proj.trellis`):
    ///   dq: b0 = 0·4 + 4 − 16 + 256·4 = 1012, b1 = 1028, i0 = 1012/32 = 31, i1 = 1027/32 % 32 = 0,
    ///       s0 = 33·32 − 1028 = 28; a = ptr[31] = 0xc608d84f, b = ptr[0] = 0x83e9f451;
    ///       funnelshift_r(b, a, 28) & 0xffff = 0x84f8.
    ///   3INST: 0x84f8·89226354 + 64248484 = 0x2eefb114; lop3 (x & 0x8fff8fff) ^ 0x3b603b60 = 0x358fba74;
    ///       __hadd(0xba74 = −0.806640625, 0x358f = 0.347412109375) = 0xb759 = −0.459228515625.
    /// MCG (0xzknw/LFM2.5-1.2B-Thinking-EXL3-4bpw `model.layers.0.feed_forward.w1`, tile 0):
    ///   ptr[31] = 0x756e62b6, ptr[0] = 0x0db1fa72
    ///   → state 0x2b60; ·0xcbac1fed = 0x49e8c7e0; lop3 → 0x3288bc80; −1.125 + 0.2041015625 = 0xbb5e.
    /// MUL1 (Honkware/Qwen3.5-0.8B-exl3-4.0bpw `layers.3.self_attn.q_proj`, tile 0): ptr[31] = 0x466b2aa5,
    ///   ptr[0] = 0x68a8377e → state 0xaa56; ·0x83dcd12d = 0xef16271e, byte sum 330; fp16 1024 + 330 =
    ///   1354; fma(1354, 0x1eee, 0xc931) = −1.21993… → fp16 0xbce1 = −1.2197265625.
    /// Half-integer K (MikeRoz/MiMo-V2.6-Flash-MOPD-2.52bpw-h6-exl3, MUL1): 3.5 bit
    ///   `model.layers.0.mlp.down_proj` [.., 56] and 2.5 bit `layers.4.mlp.experts.10.down_proj`
    ///   [.., 40], tile 0. dq8_half: even positions take KA bits, odd KA + 1, so 3.5-bit position 0
    ///   ends at bit 3 (state 0x325b → 0xbb0d) and position 1 at bit 7 (state 0x25b2 = 0x325b << 4).
    #[test]
    fn real_tiles_match_reference_kernel() {
        #[rustfmt::skip]
        let cases: [(usize, &[u16], [u16; 16], [u16; 16], f64); 5] = [
            (0, &[62545, 33769, 40878, 2869, 60200, 5236, 31554, 56187, 40194, 10796, 31636, 43686, 63259, 6313, 48624, 29062, 19971, 1419, 7475, 29558, 28022, 37150, 23351, 65191, 8280, 59432, 64760, 35106, 49940, 43691, 9921, 46191, 64311, 32530, 2923, 36060, 56768, 63206, 13923, 39684, 61074, 64530, 14691, 39050, 40856, 57734, 60376, 47358, 45619, 62800, 53198, 41697, 24123, 38725, 49266, 61454, 65022, 56365, 21379, 41967, 46528, 64160, 55375, 50696],
             [0xb759, 0x400e, 0xb81e, 0x3a7b, 0xbd32, 0x3d2f, 0x3f3f, 0x396a, 0x3cc6, 0x3b9d, 0xb638, 0x3cd5, 0xc09d, 0x3ac4, 0x2ae0, 0x4208],
             [0xb759, 0x38a6, 0x2f10, 0xb458, 0x3b75, 0xbd0e, 0x2588, 0xba23, 0x25e0, 0xc05c, 0xb4b0, 0x3e03, 0x3c27, 0xbdcf, 0xb8ae, 0x3b1e],
             29.5291748046875),
            (1, &[64114, 3505, 28361, 7934, 38358, 51070, 40544, 11474, 1103, 28109, 16673, 10234, 64249, 48373, 643, 56400, 61268, 18597, 32581, 25817, 17880, 50625, 23904, 55765, 56431, 28179, 39935, 42618, 35892, 56439, 58365, 1967, 23693, 10647, 60228, 19237, 38741, 13824, 23966, 51893, 36179, 64979, 57144, 57194, 48374, 13886, 60647, 4559, 63689, 26910, 16556, 57992, 53840, 51014, 57362, 28876, 17888, 52952, 45016, 17442, 47288, 682, 25270, 30062],
             [0xbb5e, 0xb402, 0xbcfa, 0xb96d, 0xb34e, 0xa928, 0x3b96, 0xb2ce, 0xb97b, 0x385a, 0xb7df, 0xbc9c, 0xbe13, 0xb588, 0xb73c, 0x2ad0],
             [0xbb5e, 0xbaec, 0xb6a8, 0xb8ba, 0xbc2c, 0xb0f0, 0x3c74, 0xbfe0, 0x2ccc, 0xb0c5, 0x3f33, 0x3e16, 0xb598, 0xb98c, 0xb333, 0xb978],
             -30.947509765625),
            (2, &[14206, 26792, 42449, 13971, 62356, 8493, 61383, 63754, 55887, 2989, 24845, 39033, 35346, 26523, 412, 2031, 55459, 42671, 21172, 40938, 45280, 63292, 1598, 10155, 51552, 24280, 64196, 2185, 61163, 30839, 46206, 50544, 31551, 4679, 36018, 32506, 43697, 61225, 3709, 39012, 52200, 63990, 14386, 60212, 53653, 14396, 19679, 418, 48678, 11492, 5002, 13683, 54952, 8903, 38999, 4797, 7969, 48060, 64023, 13404, 51429, 24142, 10917, 18027],
             [0xbce1, 0x2fae, 0x3678, 0xba59, 0xbe2e, 0xbd49, 0x2b90, 0x38ce, 0xb9a5, 0x9770, 0x39d5, 0xb952, 0xb21f, 0x2b90, 0xbed4, 0x3aa5],
             [0xbce1, 0x3cbb, 0xb10a, 0xb5a9, 0xbd57, 0x2677, 0x3f24, 0x3e01, 0xb821, 0x35ed, 0x3124, 0x3c91, 0x2fae, 0x3e3f, 0xaf0c, 0x3ceb],
             -15.986770629882812),
            (2, &[16324, 25946, 48351, 43762, 24729, 55331, 33901, 55899, 46954, 34989, 37209, 58274, 23643, 41240, 34094, 63788, 36962, 59721, 8643, 21921, 39803, 36278, 49659, 55130, 25768, 48972, 47285, 24847, 653, 24222, 5431, 39707, 15965, 35, 40679, 4582, 45461, 45695, 8155, 58929, 1751, 55038, 47851, 16035, 32207, 56303, 610, 14310, 20264, 33315, 44979, 40422, 382, 45816, 9803, 7638],
             [0xbb0d, 0x3469, 0xac72, 0xae9d, 0x3b14, 0xb997, 0x3d23, 0xbdb1, 0x3adc, 0xa3dc, 0xa065, 0xb7d3, 0xb0d2, 0xb28e, 0x3693, 0x34bc],
             [0xbb0d, 0x3c30, 0xbd20, 0xa3dc, 0xb858, 0xba59, 0x3d38, 0xb9cf, 0xb813, 0xbaf2, 0xb79c, 0xb6a2, 0xbf66, 0xbd88, 0xb3db, 0xc19f],
             -20.96405792236328),
            (2, &[193, 5968, 49498, 6739, 46306, 12555, 1031, 4782, 11239, 45741, 54165, 14155, 1424, 43044, 58170, 31926, 32817, 20552, 30508, 56772, 21786, 48315, 40397, 63037, 19180, 36558, 46619, 15380, 35711, 46265, 29326, 46632, 31115, 43825, 9551, 64017, 28050, 29899, 49341, 23787],
             [0x32a8, 0xb7ef, 0xbad6, 0x3485, 0xbddb, 0xb9a5, 0x3c37, 0xb8b9, 0x33bd, 0xb334, 0x38f7, 0xa065, 0x3e07, 0xbaba, 0x2ab2, 0x3b3d],
             [0x32a8, 0xb821, 0xbc72, 0x36af, 0x3888, 0xbc6b, 0xbdb8, 0x3e3f, 0xb09b, 0x352b, 0xb7b7, 0xb5e0, 0xb53a, 0x3124, 0xa065, 0xbbb4],
             -9.217002868652344),
        ];
        for (cb, u16s, row0, col0, sum) in cases {
            let words: Vec<u32> = u16s
                .chunks(2)
                .map(|p| p[0] as u32 | (p[1] as u32) << 16)
                .collect();
            let mut tile = [[0f32; 16]; 16];
            let (ka, half) = (u16s.len() / 16, u16s.len() % 16 == 8);
            decode_tile(&words, ka, half, &LUT[cb], &mut tile);
            for i in 0..16 {
                assert_eq!(tile[0][i], f16_to_f32(row0[i]), "cb {cb} K {ka} (0, {i})");
                assert_eq!(tile[i][0], f16_to_f32(col0[i]), "cb {cb} K {ka} ({i}, 0)");
            }
            let s: f64 = tile.iter().flatten().map(|&v| v as f64).sum();
            assert_eq!(s, sum, "cb {cb} K {ka} tile sum");
        }
    }
}
