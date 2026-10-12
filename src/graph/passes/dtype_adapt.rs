//! GPU-side activation dtype adaptation: inserts `Cast` around F32-only kernels for F16/BF16 values.
//!
//! No `Cast` op or kernel exists yet (G4.1 step 9), so nothing is adapted and `can_cast` is false.

use crate::graph::{DType, Graph, Op};

/// Inserts casts so F16/BF16 operands reach F32-only kernels.
pub fn run(_g: &mut Graph) {}

/// Whether `op` at the reduced-precision `dtype` can run on the GPU through an inserted cast.
pub fn can_cast(_op: &Op, _dtype: DType) -> bool {
    false
}

/// Retypes weights whose quant format has no GPU kernel to F32. The Metal weight upload dequantizes
/// such tensors at load (`metal.rs`, logged per tensor), so the graph must declare what the GPU holds.
///
/// Fails when the dequantized tensor would exceed 2 GiB: the f32 kernels read garbage (NaN logits on
/// E4B's 2.7 GB `lm_head`) past that offset, so that is a gap, not an adaptation.
pub fn retype_uncoded_weights(
    g: &mut Graph,
    has_kernel: &dyn Fn(crate::hardware::QuantFormat) -> bool,
) -> Result<(), String> {
    for v in &mut g.values {
        if let (crate::graph::ValueKind::Weight, DType::Quant(f)) = (&v.kind, v.dtype) {
            if has_kernel(f) {
                continue;
            }
            let n: usize = v.shape.iter().product();
            if n * 4 > i32::MAX as usize {
                return Err(format!(
                    "gpu-only: weight '{}' ({f:?}, {n} elements) has no GPU kernel and its f32 dequantization exceeds 2 GiB; run with --backend cpu to accept the CPU executor",
                    v.name
                ));
            }
            v.dtype = DType::F32;
        }
    }
    Ok(())
}
