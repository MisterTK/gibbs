use crate::graph::codec::Codec;

#[derive(Debug, Clone, Copy)]
pub struct GemvParams {
    pub rows_per_sg: usize,
    pub k_unroll: usize,
}

impl Default for GemvParams {
    fn default() -> Self {
        Self {
            rows_per_sg: 2,
            k_unroll: 1,
        }
    }
}

/// Generates an MSL GEMV kernel function from a codec, parameters, prologue, and epilogue.
pub fn gemv(codec: &Codec, p: GemvParams, prologue: &str, epilogue: &str) -> String {
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let prologue_expr = if prologue.is_empty() {
        "x[col]"
    } else {
        prologue
    };
    let epilogue_stmt = if epilogue.is_empty() {
        "out[row] = sum;"
    } else {
        epilogue
    };

    format!(
        r#"
// Generated GEMV for {format_str}
{codec_msl}

struct GemvParams_{format_str} {{
    uint rows;
    uint cols;
    uint block_elems;
    uint block_bytes;
}};

kernel void gemv_{format_str}(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant GemvParams_{format_str}& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    uint row = tg * {rows_per_sg} + (sg % {rows_per_sg});
    if (row >= p.rows) return;

    float sum = 0.0f;
    uint num_sub = p.cols / 8;
    for (uint s = lane; s < num_sub; s += 32) {{
        float dec8[8];
        uint blk_idx = (s * 8) / p.block_elems;
        uint sub_in_blk = (s * 8) % p.block_elems / 8;
        device const uchar* blk = weights + (ulong)row * (p.cols / p.block_elems * p.block_bytes) + blk_idx * p.block_bytes;
        decode8(blk, sub_in_blk, dec8);

        for (uint i = 0; i < 8; i++) {{
            uint col = s * 8 + i;
            float x_val = {prologue_expr};
            sum += dec8[i] * x_val;
        }}
    }}

    sum = simd_sum(sum);
    if (lane == 0) {{
        {epilogue_stmt}
    }}
}}
"#,
        format_str = format_str,
        codec_msl = codec.msl,
        rows_per_sg = p.rows_per_sg,
        prologue_expr = prologue_expr,
        epilogue_stmt = epilogue_stmt,
    )
}

/// Generates an MSL Multi-GEMV kernel where N weights share the same input activation vector.
pub fn multi_gemv(codec: &Codec, n_weights: usize, prologue: &str, epilogue: &str) -> String {
    let format_str = format!("{:?}", codec.format).to_lowercase();
    let prologue_expr = if prologue.is_empty() {
        "x[col]"
    } else {
        prologue
    };
    let epilogue_stmt = if epilogue.is_empty() {
        "for (uint w = 0; w < N; w++) out[w * p.rows + row] = sum[w];"
    } else {
        epilogue
    };

    format!(
        r#"
// Generated Multi-GEMV (N={n_weights}) for {format_str}
{codec_msl}

struct MultiGemvParams_{format_str} {{
    uint rows;
    uint cols;
    uint block_elems;
    uint block_bytes;
}};

kernel void multi_gemv_{format_str}_{n_weights}(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MultiGemvParams_{format_str}& p [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    uint row = tg * 2 + (sg % 2);
    if (row >= p.rows) return;

    float sum[{n_weights}];
    for (uint w = 0; w < {n_weights}; w++) sum[w] = 0.0f;

    uint num_sub = p.cols / 8;
    for (uint s = lane; s < num_sub; s += 32) {{
        for (uint i = 0; i < 8; i++) {{
            uint col = s * 8 + i;
            float x_val = {prologue_expr};
            for (uint w = 0; w < {n_weights}; w++) {{
                float dec8[8];
                uint blk_idx = (s * 8) / p.block_elems;
                uint sub_in_blk = (s * 8) % p.block_elems / 8;
                device const uchar* blk = weights + (ulong)(w * p.rows + row) * (p.cols / p.block_elems * p.block_bytes) + blk_idx * p.block_bytes;
                decode8(blk, sub_in_blk, dec8);
                sum[w] += dec8[i] * x_val;
            }}
        }}
    }}

    for (uint w = 0; w < {n_weights}; w++) {{
        sum[w] = simd_sum(sum[w]);
    }}

    if (lane == 0) {{
        {epilogue_stmt}
    }}
}}
"#,
        format_str = format_str,
        codec_msl = codec.msl,
        n_weights = n_weights,
        prologue_expr = prologue_expr,
        epilogue_stmt = epilogue_stmt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::codec::CODEC_Q4_0;

    #[test]
    fn test_gemv_codegen() {
        let code = gemv(&CODEC_Q4_0, GemvParams::default(), "", "");
        assert!(code.contains("kernel void gemv_gguf_q4_0"));
        assert!(code.contains("decode8("));
        assert!(code.contains("out[row] = sum;"));
    }

    #[test]
    fn test_multi_gemv_codegen() {
        let code = multi_gemv(&CODEC_Q4_0, 2, "x[col]", "out[row] = sum[0] * sum[1];");
        assert!(code.contains("kernel void multi_gemv_gguf_q4_0_2"));
        assert!(code.contains("out[row] = sum[0] * sum[1];"));
    }
}

/// Whether the generated-kernel templates cover `op` at `dtype`. Nothing is covered until the
/// generated GEMV is wired into `GraphProgram::compile` (G3.2) and the movement/elementwise/reduction
/// templates exist (G4.1).
pub fn can_generate(_op: &crate::graph::Op, _dtype: crate::graph::DType) -> bool {
    false
}

/// Rows one threadgroup of [`gemv_packed`] covers: 8 simdgroups × `rows_per_sg`.
pub const fn gemv_rows_per_tg(p: GemvParams) -> usize {
    8 * p.rows_per_sg
}

/// MSL common to the hand kernels and generated GEMVs: GELU variants, `PackedGemvParams` and
/// `apply_gemv_epi`, taken from `metal_kernels.metal` between its `GEMV-COMMON` markers.
pub fn gemv_common(kernels_msl: &str) -> &str {
    let begin = kernels_msl
        .find("// GEMV-COMMON-BEGIN")
        .expect("GEMV-COMMON-BEGIN marker in metal_kernels.metal");
    let end = kernels_msl
        .find("// GEMV-COMMON-END")
        .expect("GEMV-COMMON-END marker in metal_kernels.metal");
    &kernels_msl[begin..end]
}

/// Name of the kernel [`gemv_packed`] emits for `codec` with `p`.
pub fn gemv_packed_name(codec: &Codec, p: GemvParams) -> String {
    format!("gemvgen_{:?}_r{}", codec.format, p.rows_per_sg)
}

/// A packed-weight GEMV with the `Step::GemvPacked` contract: buffers `w, x, y, PackedGemvParams,
/// bias, res`; grid `(ceil(rows / gemv_rows_per_tg), batch)` of 256 threads. Lane `l` of a
/// simdgroup walks each row's 8-element sub-blocks `l, l+32, …`, decodes them with the codec's
/// `decode8`, and the simdgroup sums the row. Bias and the fused epilogue are applied by lane 0.
pub fn gemv_packed(codec: &Codec, p: GemvParams) -> String {
    let name = gemv_packed_name(codec, p);
    // Every codec defines `decode8`; one library holds several codecs, so each gets its own name.
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = codec.msl.replace("decode8", &dec);
    let r = p.rows_per_sg;
    let (be, bb) = (codec.block_elems, codec.block_bytes);
    let acc_decl = (0..r).map(|i| format!("float acc{i} = 0.0f;")).collect::<Vec<_>>().join(" ");
    let rows_loop: String = (0..r)
        .map(|i| {
            format!(
                "        if (row0 + {i} < p.rows) {{\n            device const uchar* b = w + ((ulong)(row0 + {i}) * blocks_per_row + blk) * {bb};\n            float v[8];\n            {dec}(b, sub, v);\n            acc{i} += dot(float4(v[0], v[1], v[2], v[3]), x0) + dot(float4(v[4], v[5], v[6], v[7]), x1);\n        }}\n"
            )
        })
        .collect();
    let store: String = (0..r)
        .map(|i| {
            format!(
                "        if (row0 + {i} < p.rows) {{\n            float a = simd_sum_{i};\n            if (p.has_bias != 0) a += bias[row0 + {i}];\n            ulong o = (ulong)batch * p.rows + row0 + {i};\n            y[o] = apply_gemv_epi(a, o, p, res);\n        }}\n"
            )
        })
        .collect();
    let sums: String = (0..r)
        .map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n"))
        .collect();
    format!(
        r#"
{codec_msl}

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant PackedGemvParams& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device const float* res [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    const uint subs = p.cols / 8;
    const uint blocks_per_row = p.cols / {be};
    device const float* xb = x + (ulong)batch * p.cols;
    {acc_decl}
    for (uint s = lane; s < subs; s += 32) {{
        float4 x0 = *(device const float4*)(xb + s * 8);
        float4 x1 = *(device const float4*)(xb + s * 8 + 4);
        const uint blk = (s * 8) / {be};
        const uint sub = ((s * 8) % {be}) / 8;
{rows_loop}    }}
{sums}    if (lane == 0) {{
{store}    }}
}}
"#,
        rows_per_tg = gemv_rows_per_tg(p),
    )
}
