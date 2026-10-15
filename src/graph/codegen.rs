use crate::graph::codec::{planar, tile, wide, Codec, Tile, HELPERS};

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

/// The MSL a GPU consumer of the device weight buffer includes: [`Codec::msl`], or the planar-layout decoder
/// for a codec stored [`planar`].
pub fn device_msl(codec: &Codec) -> String {
    // The planar decoders use the codec helpers (guarded, so a duplicate is harmless).
    planar(codec.format).map_or_else(|| codec.msl.to_string(), |p| format!("{HELPERS}{}", p.msl))
}

/// How MSL addresses block `idx` (a `ulong` expression) of weights `w`: a declaration of `b` (what `decode8`
/// takes as `blk`) and the prefixes to put before `sub` (resp. the wide group index): empty unless the codec is
/// stored [`planar`], where `b` is the block's group and the prefixes select the block within it.
pub struct BlockRef {
    pub decl: String,
    pub sub: String,
    pub group: String,
}

pub fn block_ref(codec: &Codec, w: &str, idx: &str) -> BlockRef {
    match planar(codec.format) {
        None => BlockRef {
            decl: format!("device const uchar* b = {w} + ({idx}) * {};", codec.block_bytes),
            sub: String::new(),
            group: String::new(),
        },
        Some(p) => {
            let groups = tile(codec.format)
                .map(|t| t.groups)
                .or_else(|| wide(codec.format).map(|w| w.groups))
                .unwrap_or(0);
            BlockRef {
                decl: format!(
                    "const ulong bi = {idx}; device const uchar* b = {w} + (bi / {}) * {};",
                    p.blocks,
                    p.blocks * codec.block_bytes
                ),
                sub: format!("uint(bi % {}) * {} + ", p.blocks, codec.block_elems / 8),
                group: format!("uint(bi % {}) * {groups} + ", p.blocks),
            }
        }
    }
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

/// Function-constant indices of the shape a [`gemv_packed`] pipeline may be specialized to: `rows` and blocks
/// per row (`cols / block_elems`). Left undefined, the kernel reads both from `PackedGemvParams`; defined, loop
/// trip counts and row guards fold when the pipeline is compiled.
pub const FC_ROWS: u64 = 0;
pub const FC_BLOCKS_PER_ROW: u64 = 1;

const GEMV_SPECIALIZATION: &str = r#"
constant uint fc_rows [[function_constant(0)]];
constant uint fc_bpr [[function_constant(1)]];
constant bool fc_has_rows = is_function_constant_defined(fc_rows);
constant bool fc_has_bpr = is_function_constant_defined(fc_bpr);
"#;

/// Name of the kernel [`gemv_packed`] emits for `codec` with `p`.
pub fn gemv_packed_name(codec: &Codec, p: GemvParams) -> String {
    format!("gemvgen_{:?}_r{}", codec.format, p.rows_per_sg)
}

/// Name of the kernel [`dequant_packed`] emits for `codec`.
pub fn dequant_packed_name(codec: &Codec) -> String {
    format!("dequantgen_{:?}", codec.format)
}

/// Expands a packed `[rows, cols]` weight to f32: buffers `w, out`, bytes `(rows, cols)`, one thread
/// per 8-element sub-block (`ceil(rows * cols / 8)` threads). Prefill GEMMs read the result.
pub fn dequant_packed(codec: &Codec) -> String {
    let name = dequant_packed_name(codec);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let (be, subs) = (codec.block_elems, codec.block_elems / 8);
    let br = block_ref(
        codec,
        "w",
        &format!("(ulong)row * (dims.y / {be}) + s / {subs}"),
    );
    let (decl, sub) = (&br.decl, &br.sub);
    format!(
        r#"
{codec_msl}

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint per_row = dims.y / 8;
    if (gid >= dims.x * per_row) return;
    const uint row = gid / per_row, s = gid % per_row;
    {decl}
    float v[8];
    {dec}(b, {sub}s % {subs}, v);
    device float* o = out + (ulong)row * dims.y + s * 8;
    for (uint i = 0; i < 8; ++i) o[i] = v[i];
}}
"#
    )
}

/// A packed-weight GEMV with the `Step::GemvPacked` contract: buffers `w, x, y, PackedGemvParams,
/// bias, res`; grid `(ceil(rows / gemv_rows_per_tg), batch)` of 256 threads. Lane `l` of a
/// simdgroup walks each row's 8-element sub-blocks `l, l+32, …`, decodes them with the codec's
/// `decode8`, and the simdgroup sums the row. Bias and the fused epilogue are applied by lane 0.
pub fn gemv_packed(codec: &Codec, p: GemvParams) -> String {
    let name = gemv_packed_name(codec, p);
    // Every codec defines `decode8`; one library holds several codecs, so each gets its own name.
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let r = p.rows_per_sg;
    let be = codec.block_elems;
    let loop_src = gemv_loop(codec, p, &dec);
    let store_row = |acc: &str, row: &str, ind: &str| {
        format!(
            "{ind}float a = {acc};\n{ind}if (p.has_bias != 0) a += bias[{row}];\n{ind}ulong o = (ulong)batch * n_rows + {row};\n{ind}y[o] = apply_gemv_epi(a, o, p, res);\n"
        )
    };
    // A tile codec splits the simdgroup's lanes across its `r` rows (`32 / r` lanes per row, summed by
    // xor-shuffles): its row has few 16-byte tiles, so letting every lane serve one row keeps all of them busy.
    let (acc_decl, sums, guard, store) = if tile(codec.format).is_some() {
        let lpr = 32 / r;
        let shuffles: String = std::iter::successors(Some(lpr / 2), |o| Some(o / 2))
            .take_while(|&o| o > 0)
            .map(|o| format!("    acc0 += simd_shuffle_xor(acc0, {o});\n"))
            .collect();
        (
            format!("const uint sl = lane % {lpr}, rl = lane / {lpr}, myrow = row0 + rl;\n    float acc0 = 0.0f;"),
            shuffles,
            format!("sl == 0 && myrow < n_rows"),
            store_row("acc0", "myrow", "        "),
        )
    } else {
        let store: String = (0..r)
            .map(|i| {
                format!(
                    "        if (row0 + {i} < n_rows) {{\n{}        }}\n",
                    store_row(&format!("simd_sum_{i}"), &format!("row0 + {i}"), "            ")
                )
            })
            .collect();
        (
            (0..r).map(|i| format!("float acc{i} = 0.0f;")).collect::<Vec<_>>().join(" "),
            (0..r).map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n")).collect(),
            "lane == 0".to_string(),
            store,
        )
    };
    format!(
        r#"
{codec_msl}
{wide_msl}
{GEMV_SPECIALIZATION}
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
    const uint n_rows = fc_has_rows ? fc_rows : p.rows;
    const uint n_cols = fc_has_bpr ? fc_bpr * {be} : p.cols;
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint batch = tg.y;
    if (row0 >= n_rows || batch >= p.batch_size) return;
    const uint blocks_per_row = n_cols / {be};
    device const float* xb = x + (ulong)batch * n_cols;
    {acc_decl}
{loop_src}{sums}    if ({guard}) {{
{store}    }}
}}
"#,
        rows_per_tg = gemv_rows_per_tg(p),
        wide_msl = wide(codec.format)
            .map(|w| planar(codec.format)
                .filter(|p| !p.wide_msl.is_empty())
                .map_or(w.msl, |p| p.wide_msl)
                .replace("decode_group", &dec.replace("decode8", "decode_group"))
                .replace("group_seg", &dec.replace("decode8", "group_seg")))
            .or_else(|| tile(codec.format).map(|t| {
                ["group_seg", "tile_load", "tile_scale", "tile_seg"]
                    .iter()
                    .fold(t.msl.to_string(), |m, n| m.replace(n, &dec.replace("decode8", n)))
            }))
            .unwrap_or_default(),
    )
}

/// The lane loop of [`gemv_packed`]: one 8-element sub-block per iteration (`decode8`), or, for a codec
/// with a [`wide`] decode, one group of `segs` sub-blocks (its bytes are read once, not once per sub-block).
fn gemv_loop(codec: &Codec, p: GemvParams, dec: &str) -> String {
    let (be, r) = (codec.block_elems, p.rows_per_sg);
    // `decode` gets the block's prefixes (`BlockRef::sub` for decode8, `group` for decode_group).
    let row = |i: usize, decode: &dyn Fn(&BlockRef) -> String, dots: &str, v_len: u32| {
        let br = block_ref(codec, "w", &format!("(ulong)(row0 + {i}) * blocks_per_row + blk"));
        format!(
            "        if (row0 + {i} < n_rows) {{\n            {}\n            float v[{v_len}];\n            {}\n{dots}        }}\n",
            br.decl,
            decode(&br)
        )
    };
    if let Some(t) = tile(codec.format) {
        return tile_loop(codec, t, p, dec);
    }
    let Some(wd) = wide(codec.format) else {
        let rows: String = (0..r)
            .map(|i| {
                row(
                    i,
                    &|br| format!("{dec}(b, {}sub, v);", br.sub),
                    &format!("            acc{i} += dot(float4(v[0], v[1], v[2], v[3]), x0) + dot(float4(v[4], v[5], v[6], v[7]), x1);\n"),
                    8,
                )
            })
            .collect();
        return format!(
            "    const uint subs = n_cols / 8;\n    for (uint s = lane; s < subs; s += 32) {{\n        float4 x0 = *(device const float4*)(xb + s * 8);\n        float4 x1 = *(device const float4*)(xb + s * 8 + 4);\n        const uint blk = (s * 8) / {be};\n        const uint sub = ((s * 8) % {be}) / 8;\n{rows}    }}\n"
        );
    };
    let (gdec, gseg) = (
        dec.replace("decode8", "decode_group"),
        dec.replace("decode8", "group_seg"),
    );
    let xs: String = (0..wd.segs)
        .map(|k| {
            format!(
                "        {{ const uint e = blk * {be} + {gseg}(g, {k}); xs[{}] = *(device const float4*)(xb + e); xs[{}] = *(device const float4*)(xb + e + 4); }}\n",
                2 * k,
                2 * k + 1
            )
        })
        .collect();
    let nseg = wd.segs as usize;
    let rows: String = (0..r)
        .map(|i| {
            let seg_dot = |k: usize| {
                let o = 8 * k;
                format!(
                    "dot(float4(v[{o}], v[{}], v[{}], v[{}]), xs[{}]) + dot(float4(v[{}], v[{}], v[{}], v[{}]), xs[{}])",
                    o + 1, o + 2, o + 3, 2 * k, o + 4, o + 5, o + 6, o + 7, 2 * k + 1
                )
            };
            let mut dots = String::new();
            if wd.uniform {
                let sum = (0..nseg).map(|k| format!("({})", seg_dot(k))).collect::<Vec<_>>().join(" + ");
                dots += &format!("            acc{i} += v[{}] * ({sum} - sxall);\n", 8 * nseg);
            } else {
                for k in 0..nseg {
                    dots += &format!(
                        "            acc{i} += v[{}] * ({}) + v[{}] * sx{k};\n",
                        8 * nseg + k, seg_dot(k), 9 * nseg + k
                    );
                }
            }
            row(i, &|br| format!("{gdec}(b, {}g, v);", br.group), &dots, 10 * wd.segs)
        })
        .collect();
    let groups = wd.groups;
    let sx: String = if wd.uniform {
        let all = (0..2 * nseg)
            .map(|j| format!("xs[{j}]"))
            .collect::<Vec<_>>()
            .join(" + ");
        format!("        const float sxall = dot({all}, float4(1.0f));\n")
    } else {
        (0..nseg)
            .map(|k| {
                format!(
                    "        const float sx{k} = dot(xs[{}], float4(1.0f)) + dot(xs[{}], float4(1.0f));\n",
                    2 * k,
                    2 * k + 1
                )
            })
            .collect()
    };
    format!(
        "    const uint groups = blocks_per_row * {groups};\n    for (uint s = lane; s < groups; s += 32) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        float4 xs[{}];\n{xs}{sx}{rows}    }}\n",
        2 * wd.segs
    )
}

/// The lane loop of [`gemv_packed`] for a [`Tile`] codec: lane `l` serves row `myrow` (`32 / rows_per_sg`
/// lanes per row), walks that row's tiles `sl, sl + lpr, …`, loads the tile and its scale, then decodes one
/// 8-element segment at a time against `x` and accumulates `scale * (dot(codes, x) - sum(x))`.
fn tile_loop(codec: &Codec, t: &Tile, p: GemvParams, dec: &str) -> String {
    let (be, r, segs) = (codec.block_elems, p.rows_per_sg, t.segs);
    let lpr = 32 / r;
    assert!(segs as usize % r == 0, "{} rows per simdgroup do not divide {segs} segments", r);
    let name = |n: &str| dec.replace("decode8", n);
    let (seg, load, scale, tseg) = (
        name("group_seg"),
        name("tile_load"),
        name("tile_scale"),
        name("tile_seg"),
    );
    let br = block_ref(codec, "w", "(ulong)myrow * blocks_per_row + blk");
    let body: String = (0..segs)
        .map(|k| {
            format!(
                "        {{\n            const uint e = blk * {be} + {seg}(g, {k});\n            const float4 xa = *(device const float4*)(xb + e), xc = *(device const float4*)(xb + e + 4);\n            float v[8];\n            {tseg}(q, {k}, v);\n            t += dot(float4(v[0], v[1], v[2], v[3]), xa) + dot(float4(v[4], v[5], v[6], v[7]), xc);\n        }}\n"
            )
        })
        .collect();
    // The lanes serving the same tile of different rows (`lane % lpr` equal) split sum(x) over the tile
    // between them (`segs / r` segments each) and add the parts with xor-shuffles.
    let spr = segs as usize / r;
    let sx_parts: String = (0..spr)
        .map(|m| {
            format!(
                "        {{\n            const uint e = blk * {be} + {seg}(g, rl * {spr} + {m});\n            sx += *(device const float4*)(xb + e) + *(device const float4*)(xb + e + 4);\n        }}\n"
            )
        })
        .collect();
    let sx_sum: String = std::iter::successors(Some(lpr), |o| Some(o * 2))
        .take_while(|&o| o < 32)
        .map(|o| format!("        sxs += simd_shuffle_xor(sxs, {o});\n"))
        .collect();
    let groups = t.groups;
    format!(
        "    const uint tiles = blocks_per_row * {groups};\n    for (uint s = sl; s < tiles; s += {lpr}) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        uint4 q = uint4(0);\n        float d = 0.0f, t = 0.0f;\n        if (myrow < p.rows) {{\n            {}\n            q = {load}(b, {}g);\n            d = {scale}(b, {}g);\n        }}\n        float4 sx = float4(0.0f);\n{sx_parts}        float sxs = dot(sx, float4(1.0f));\n{sx_sum}{body}        acc0 += d * ({a:?}f * t - {b:?}f * sxs);\n    }}\n",
        br.decl, br.group, br.group, a = t.a, b = t.b
    )
}

/// Name of the kernel [`gemv_indexed`] emits for `codec` with `p`.
pub fn gemv_indexed_name(codec: &Codec, p: GemvParams) -> String {
    format!("gemvgen_{:?}_r{}_ix", codec.format, p.rows_per_sg)
}

/// [`gemv_packed`] over an expert bank: the weight is `experts` stacked `[rows, cols]` matrices and
/// slot `s` (grid y) multiplies row `s` of `x` (`x_per_slot != 0`) or row `s / top_k` by the matrix
/// `ids[s]`, writing row `s` of `y` (`[slots, rows]`). Buffers `w, x, y, MoeGemvParams {rows, cols,
/// slots, top_k, experts, x_per_slot}, ids`; grid `(ceil(rows / gemv_rows_per_tg), slots)` of 256
/// threads. A slot whose id is not below `experts` writes zeros.
pub fn gemv_indexed(codec: &Codec, p: GemvParams) -> String {
    let name = gemv_indexed_name(codec, p);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let r = p.rows_per_sg;
    let be = codec.block_elems;
    let acc_decl = (0..r)
        .map(|i| format!("float acc{i} = 0.0f;"))
        .collect::<Vec<_>>()
        .join(" ");
    let rows_loop: String = (0..r)
        .map(|i| {
            let br = block_ref(codec, "w", &format!("(bank_row + row0 + {i}) * blocks_per_row + blk"));
            format!(
                "        if (row0 + {i} < p.rows) {{\n            {}\n            float v[8];\n            {dec}(b, {}sub, v);\n            acc{i} += dot(float4(v[0], v[1], v[2], v[3]), x0) + dot(float4(v[4], v[5], v[6], v[7]), x1);\n        }}\n",
                br.decl, br.sub
            )
        })
        .collect();
    let sums: String = (0..r)
        .map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n"))
        .collect();
    let store: String = (0..r)
        .map(|i| {
            format!("        if (row0 + {i} < p.rows) y[(ulong)slot * p.rows + row0 + {i}] = simd_sum_{i};\n")
        })
        .collect();
    format!(
        r#"
{codec_msl}

struct MoeGemvParams {{
    uint rows;
    uint cols;
    uint slots;
    uint top_k;
    uint experts;
    uint x_per_slot;
}};

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvParams& p [[buffer(3)]],
    device const uint* ids [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint slot = tg.y;
    if (row0 >= p.rows || slot >= p.slots) return;
    const uint expert = ids[slot];
    if (expert >= p.experts) {{
        if (lane == 0) {{
            for (uint i = 0; i < {r}; i++) if (row0 + i < p.rows) y[(ulong)slot * p.rows + row0 + i] = 0.0f;
        }}
        return;
    }}
    const ulong bank_row = (ulong)expert * p.rows;
    const uint subs = p.cols / 8;
    const uint blocks_per_row = p.cols / {be};
    device const float* xb = x + (ulong)(p.x_per_slot != 0 ? slot : slot / p.top_k) * p.cols;
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
