//! See [`super`]. Formats of this family: none registered yet.

use super::Repacked;
use crate::graph::codec::Spec;
use crate::hardware::QuantFormat;
use crate::model::layers::GgufBlockFn;
use crate::model::ModelManifest;

/// GPU codecs of this family (MSL `decode8`), merged into the registry by `graph::codec`.
pub const SPECS: &[Spec] = &[];

pub fn block_codec(_format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    None
}

pub fn repack(_m: &ModelManifest, _data: &[u8], _weight: &str) -> Result<Option<Repacked>, String> {
    Ok(None)
}
