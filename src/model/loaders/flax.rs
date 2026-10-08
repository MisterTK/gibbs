//! Flax `flax_model.msgpack` (`flax.serialization.to_bytes` of the params tree), renamed to the
//! HF PyTorch names the way transformers' `convert_flax_state_dict_to_pytorch` does: the nested
//! dict path joined with '.', `kernel` → `weight` (transposed), `embedding`/`scale` → `weight`.

use super::Checkpoint;
use std::path::Path;

/// Appends every array of the params tree at `path` to `ck`.
pub fn read(path: &Path, ck: &mut Checkpoint) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    // SAFETY: read-only map of a weight file nothing in this process writes.
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let mut r = Reader {
        b: &map,
        o: 0,
        depth: 0,
    };
    let root = r.value()?;
    if r.o != map.len() {
        return Err("trailing bytes after the msgpack params tree".into());
    }
    walk(ck, String::new(), root)
}

/// The msgpack subset Flax writes.
enum Value<'a> {
    Map(Vec<(String, Value<'a>)>),
    Array(Vec<Value<'a>>),
    Str(String),
    Bytes(&'a [u8]),
    Int(i64),
    Ext(i8, &'a [u8]),
    Other,
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
    depth: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self
            .b
            .get(self.o..self.o.checked_add(n).ok_or("msgpack length overflow")?)
            .ok_or("truncated msgpack")?;
        self.o += n;
        Ok(s)
    }

    fn uint(&mut self, n: usize) -> Result<usize, String> {
        Ok(self
            .take(n)?
            .iter()
            .fold(0usize, |a, &x| a << 8 | x as usize))
    }

    fn value(&mut self) -> Result<Value<'a>, String> {
        self.depth += 1;
        if self.depth > 64 {
            return Err("msgpack nested deeper than 64 levels".into());
        }
        let v = self.value_inner();
        self.depth -= 1;
        v
    }

    fn value_inner(&mut self) -> Result<Value<'a>, String> {
        let t = self.take(1)?[0];
        Ok(match t {
            0x00..=0x7f => Value::Int(t as i64),
            0xe0..=0xff => Value::Int(t as i8 as i64),
            0x80..=0x8f => self.map((t & 0x0f) as usize)?,
            0xde | 0xdf => {
                let n = self.uint(if t == 0xde { 2 } else { 4 })?;
                self.map(n)?
            }
            0x90..=0x9f => self.array((t & 0x0f) as usize)?,
            0xdc | 0xdd => {
                let n = self.uint(if t == 0xdc { 2 } else { 4 })?;
                self.array(n)?
            }
            0xa0..=0xbf => self.str((t & 0x1f) as usize)?,
            0xd9..=0xdb => {
                let n = self.uint(1 << (t - 0xd9))?;
                self.str(n)?
            }
            0xc4..=0xc6 => {
                let n = self.uint(1 << (t - 0xc4))?;
                Value::Bytes(self.take(n)?)
            }
            0xcc..=0xcf => Value::Int(self.uint(1 << (t - 0xcc))? as i64),
            0xd0..=0xd3 => {
                let n = 1usize << (t - 0xd0);
                let u = self.uint(n)? as u64;
                let shift = 64 - 8 * n as u32;
                Value::Int(((u << shift) as i64) >> shift)
            }
            0xd4..=0xd8 => {
                let ty = self.take(1)?[0] as i8;
                Value::Ext(ty, self.take(1 << (t - 0xd4))?)
            }
            0xc7..=0xc9 => {
                let n = self.uint(1 << (t - 0xc7))?;
                let ty = self.take(1)?[0] as i8;
                Value::Ext(ty, self.take(n)?)
            }
            0xc0 | 0xc2 | 0xc3 => Value::Other,
            0xca => {
                self.take(4)?;
                Value::Other
            }
            0xcb => {
                self.take(8)?;
                Value::Other
            }
            0xc1 => return Err("invalid msgpack byte 0xc1".into()),
        })
    }

    fn map(&mut self, n: usize) -> Result<Value<'a>, String> {
        let mut m = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            let Value::Str(k) = self.value()? else {
                return Err("msgpack map key is not a string".into());
            };
            m.push((k, self.value()?));
        }
        Ok(Value::Map(m))
    }

    fn array(&mut self, n: usize) -> Result<Value<'a>, String> {
        let mut v = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            v.push(self.value()?);
        }
        Ok(Value::Array(v))
    }

    fn str(&mut self, n: usize) -> Result<Value<'a>, String> {
        let s = std::str::from_utf8(self.take(n)?).map_err(|e| e.to_string())?;
        Ok(Value::Str(s.to_string()))
    }
}

fn walk(ck: &mut Checkpoint, path: String, v: Value) -> Result<(), String> {
    match v {
        Value::Map(m) => {
            for (k, child) in m {
                if k == "__msgpack_chunked_array__" {
                    // ponytail: arrays > 2^30 bytes are chunked by flax; reassemble chunks if a repo needs it.
                    return Err(format!("{path}: chunked flax arrays are not supported"));
                }
                let p = if path.is_empty() {
                    k
                } else {
                    format!("{path}.{k}")
                };
                walk(ck, p, child)?;
            }
            Ok(())
        }
        // flax `_MsgpackExtType`: 1 = ndarray (shape, dtype, bytes), 3 = numpy scalar (dtype, bytes).
        Value::Ext(ty @ (1 | 3), payload) => {
            let mut r = Reader {
                b: payload,
                o: 0,
                depth: 0,
            };
            let Value::Array(parts) = r.value()? else {
                return Err(format!("{path}: malformed flax array"));
            };
            let mut parts = parts.into_iter();
            let shape = if ty == 1 {
                match parts.next() {
                    Some(Value::Array(dims)) => dims
                        .into_iter()
                        .map(|d| match d {
                            Value::Int(n) if n >= 0 => Ok(n as usize),
                            _ => Err(format!("{path}: bad flax array shape")),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    _ => return Err(format!("{path}: bad flax array shape")),
                }
            } else {
                Vec::new()
            };
            let (Some(Value::Str(dtype)), Some(Value::Bytes(data))) = (parts.next(), parts.next())
            else {
                return Err(format!("{path}: malformed flax array"));
            };
            push(ck, path, shape, &dtype, data)
        }
        _ => Err(format!("{path}: not an array (flax ext type 1)")),
    }
}

fn push(
    ck: &mut Checkpoint,
    path: String,
    shape: Vec<usize>,
    dtype: &str,
    data: &[u8],
) -> Result<(), String> {
    let (dtype, size) = match dtype {
        "float64" => ("F64", 8),
        "float32" => ("F32", 4),
        "float16" => ("F16", 2),
        "bfloat16" => ("BF16", 2),
        "int64" => ("I64", 8),
        "int32" => ("I32", 4),
        "int8" => ("I8", 1),
        "uint8" => ("U8", 1),
        "bool" => ("BOOL", 1),
        other => return Err(format!("{path}: flax dtype '{other}' not supported")),
    };
    let n = shape.iter().try_fold(size, |a: usize, &d| a.checked_mul(d));
    if n != Some(data.len()) {
        return Err(format!(
            "{path}: {} bytes do not match shape {shape:?} × {dtype}",
            data.len()
        ));
    }
    let (stem, leaf) = path.rsplit_once('.').unwrap_or(("", &path));
    let renamed = |to: &str| {
        if stem.is_empty() {
            to.to_string()
        } else {
            format!("{stem}.{to}")
        }
    };
    match leaf {
        "kernel" => {
            // Dense (in, out) → Linear (out, in); Conv (H, W, in, out) → (out, in, H, W).
            let perm: Vec<usize> = if shape.len() == 4 {
                vec![3, 2, 0, 1]
            } else {
                (0..shape.len()).rev().collect()
            };
            let (shape, bytes) = permute(&shape, &perm, size, data);
            ck.push(renamed("weight"), shape, dtype, &bytes)
        }
        "embedding" | "scale" => ck.push(renamed("weight"), shape, dtype, data),
        "mean" => ck.push(renamed("running_mean"), shape, dtype, data),
        "var" => ck.push(renamed("running_var"), shape, dtype, data),
        _ => ck.push(path, shape, dtype, data),
    }
}

/// Row-major `data` of `shape` with axes reordered so output axis i is input axis `perm[i]`.
fn permute(shape: &[usize], perm: &[usize], size: usize, data: &[u8]) -> (Vec<usize>, Vec<u8>) {
    let mut strides = vec![size; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let out_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();
    let mut out = Vec::with_capacity(data.len());
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..data.len() / size {
        let src: usize = idx.iter().zip(&out_strides).map(|(i, s)| i * s).sum();
        out.extend_from_slice(&data[src..src + size]);
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    (out_shape, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_is_transposed_to_linear_layout() {
        // 2x3 f32 kernel [[0,1,2],[3,4,5]] → 3x2 [[0,3],[1,4],[2,5]].
        let data: Vec<u8> = (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let (shape, out) = permute(&[2, 3], &[1, 0], 4, &data);
        assert_eq!(shape, [3, 2]);
        let vals: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, [0., 3., 1., 4., 2., 5.]);
    }
}
