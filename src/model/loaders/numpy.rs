//! NumPy `.npy` arrays and `.npz` archives (`np.savez` / `mx.savez`, e.g. MLX-era `weights.npz`).
//! Each `<name>.npy` member becomes tensor `<name>`.

use super::Checkpoint;
use std::path::Path;

/// Appends the array(s) in the `.npy` / `.npz` file at `path` to `ck`.
pub fn read(path: &Path, ck: &mut Checkpoint) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    // SAFETY: read-only map of a weight file nothing in this process writes.
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    if map.starts_with(b"\x93NUMPY") {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        return push_npy(ck, stem.to_string(), &map);
    }
    for (name, data) in zip_entries(&map)? {
        let Some(stem) = name.strip_suffix(".npy") else {
            return Err(format!("npz member {name} is not a .npy array"));
        };
        push_npy(ck, stem.to_string(), data).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

/// Parses one `.npy` payload (format 1.0–3.0, little-endian, C order).
fn push_npy(ck: &mut Checkpoint, name: String, b: &[u8]) -> Result<(), String> {
    let short = || "truncated .npy header".to_string();
    if !b.starts_with(b"\x93NUMPY") {
        return Err("not a .npy array".into());
    }
    let (len, start) = match b.get(6).ok_or_else(short)? {
        1 => (
            u16::from_le_bytes(b.get(8..10).ok_or_else(short)?.try_into().unwrap()) as usize,
            10,
        ),
        2 | 3 => (
            u32::from_le_bytes(b.get(8..12).ok_or_else(short)?.try_into().unwrap()) as usize,
            12,
        ),
        v => return Err(format!(".npy format version {v} unknown")),
    };
    let header = b
        .get(start..start + len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(short)?;
    // `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`
    let field = |key: &str| {
        let at = header.find(&format!("'{key}':"))? + key.len() + 3;
        Some(header[at..].trim_start())
    };
    let descr = field("descr")
        .and_then(|s| s.strip_prefix('\'')?.split('\'').next())
        .ok_or("npy header: no descr")?;
    if field("fortran_order").is_some_and(|s| s.starts_with("True")) {
        return Err("Fortran-ordered .npy arrays are not supported".into());
    }
    let shape = field("shape")
        .and_then(|s| s.strip_prefix('(')?.split(')').next())
        .ok_or("npy header: no shape")?
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse::<usize>().map_err(|e| format!("npy shape: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let (dtype, size) = match descr {
        "<f8" => ("F64", 8),
        "<f4" => ("F32", 4),
        "<f2" => ("F16", 2),
        "<i8" => ("I64", 8),
        "<i4" => ("I32", 4),
        "<i2" => ("I16", 2),
        "|i1" => ("I8", 1),
        "<u8" => ("U64", 8),
        "<u4" => ("U32", 4),
        "<u2" => ("U16", 2),
        "|u1" => ("U8", 1),
        "|b1" => ("BOOL", 1),
        other => return Err(format!("npy dtype '{other}' not supported")),
    };
    let n = shape
        .iter()
        .try_fold(size, |a: usize, &d| a.checked_mul(d))
        .ok_or("npy shape overflows")?;
    let data = b
        .get(start + len..)
        .and_then(|d| d.get(..n))
        .ok_or_else(|| format!("npy data shorter than {n} bytes"))?;
    ck.push(name, shape, dtype, data)
}

/// (name, stored bytes) of every member of a ZIP archive, from its central directory.
fn zip_entries(z: &[u8]) -> Result<Vec<(String, &[u8])>, String> {
    let u16_at = |o: usize| {
        z.get(o..o + 2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let u32_at = |o: usize| {
        z.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let bad = || "truncated or malformed npz (zip) archive".to_string();
    let eocd = (0..z.len().saturating_sub(21))
        .rev()
        .take(65_557)
        .find(|&o| u32_at(o) == Some(0x0605_4b50))
        .ok_or_else(bad)?;
    let (count, mut o) = (
        u16_at(eocd + 10).ok_or_else(bad)?,
        u32_at(eocd + 16).ok_or_else(bad)?,
    );
    if count == 0xFFFF || o == 0xFFFF_FFFF {
        // ponytail: no ZIP64 directory (> 65535 members or > 4 GiB archives); parse the zip64 EOCD if one appears.
        return Err("ZIP64 npz archives are not supported".into());
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if u32_at(o) != Some(0x0201_4b50) {
            return Err(bad());
        }
        let method = u16_at(o + 10).ok_or_else(bad)?;
        let size = u32_at(o + 20).ok_or_else(bad)?;
        let (n, e, c) = (
            u16_at(o + 28).ok_or_else(bad)?,
            u16_at(o + 30).ok_or_else(bad)?,
            u16_at(o + 32).ok_or_else(bad)?,
        );
        let local = u32_at(o + 42).ok_or_else(bad)?;
        let name = z
            .get(o + 46..o + 46 + n)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .ok_or_else(bad)?;
        if method != 0 {
            return Err(format!(
                "{name}: compressed npz members (np.savez_compressed) are not supported"
            ));
        }
        if size == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
            return Err("ZIP64 npz archives are not supported".into());
        }
        if u32_at(local) != Some(0x0403_4b50) {
            return Err(bad());
        }
        let start = local
            + 30
            + u16_at(local + 26).ok_or_else(bad)?
            + u16_at(local + 28).ok_or_else(bad)?;
        out.push((name, z.get(start..start + size).ok_or_else(bad)?));
        o += 46 + n + e + c;
    }
    Ok(out)
}
