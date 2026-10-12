use crate::graph::codec::Codec;

#[derive(Debug, Clone, Copy)]
pub struct GemvParams {
    pub rows_per_sg: usize,
    pub k_unroll: usize,
}

impl Default for GemvParams {
    fn default() -> Self {
        Self {
            rows_per_sg: 4,
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
