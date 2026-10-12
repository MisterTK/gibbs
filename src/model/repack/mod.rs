//! Source-checkpoint encodings repacked into GPU codec layouts (sprint-2 S3.7): bitsandbytes NF4/FP4
//! ([`bnb`]), FP8 ([`fp8`]) and the 4-bit affine formats MLX and compressed-tensors int4 ([`affine`]).
//!
//! Each family owns three things, one set per [`QuantFormat`] it produces:
//! * a CPU block dequantizer (`block_codec`), the reference the GPU codec must equal bit-for-bit
//!   (`tests/codec_parity.rs` style) and what `QuantizedWeight::to_f32` / the CPU GEMV use;
//! * an MSL `decode8` ([`crate::graph::codec::Spec`]) registered through [`crate::graph::codec::codec`];
//! * `repack`, which turns the checkpoint's own tensors into row-major `[rows, cols]` blocks of that
//!   format, or `Ok(None)` when the weight is not stored that way (the f32 dequantize path applies).
//!
//! Layout contract for every repacked format: a weight `[rows, cols]` is `rows * (cols / block_elems)`
//! consecutive blocks, row-major, so concatenating or repeating whole rows is a byte concatenation.

use crate::hardware::QuantFormat;
use crate::model::layers::GgufBlockFn;
use crate::model::ModelManifest;

pub mod affine;
pub mod bnb;
pub mod fp8;

/// A weight repacked into a GPU codec format.
pub struct Repacked {
    pub format: QuantFormat,
    pub rows: usize,
    pub cols: usize,
    pub bytes: Vec<u8>,
}

/// `(elements, bytes, dequantizer)` of a repacked block format.
pub fn block_codec(format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    bnb::block_codec(format)
        .or_else(|| fp8::block_codec(format))
        .or_else(|| affine::block_codec(format))
}

/// Repacks the checkpoint weight `weight` (the `X.weight` name `GraphLm::from_manifest` reads) when it
/// is stored in one of the encodings above; `data` is the buffer the manifest's offsets index.
pub fn repack(m: &ModelManifest, data: &[u8], weight: &str) -> Result<Option<Repacked>, String> {
    if let Some(r) = bnb::repack(m, data, weight)? {
        return Ok(Some(r));
    }
    if let Some(r) = fp8::repack(m, data, weight)? {
        return Ok(Some(r));
    }
    affine::repack(m, data, weight)
}
