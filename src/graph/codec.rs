//! GPU weight codecs: one MSL `decode8` per GGUF block format, each a port of the CPU
//! `dequantize_*_block` in `src/model/layers.rs`. A codec is registered in [`codec`] only if
//! `tests/codec_parity.rs` proves its `decode8` reproduces the CPU dequantizer bit-for-bit.
//!
//! Contract: `inline void decode8(device const uchar* blk, uint sub, thread float* out8)` writes
//! elements `[sub*8, sub*8+8)` of the block. `blk` must be 2-byte aligned (every block size except
//! MXFP4's 17 is even, and MXFP4 reads single bytes only).

use crate::hardware::QuantFormat;
use crate::model::layers::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
    KVALUES_FP4, KVALUES_IQ4NL,
};
use std::fmt::LowerHex;
use std::sync::LazyLock;

pub struct Codec {
    pub format: QuantFormat,
    pub block_elems: u32,
    pub block_bytes: u32,
    /// MSL defining `inline void decode8(device const uchar* blk, uint sub, thread float* out8)`,
    /// preceded by every helper and lookup table it needs (guarded, so codecs can share a library).
    pub msl: &'static str,
}

/// Bumped whenever an encoder or codec layout changes: it keys the re-encode cache.
pub const CODEC_VERSION: u32 = 4;

pub fn codec(format: QuantFormat) -> Option<&'static Codec> {
    match format {
        QuantFormat::GGUF_Q4_0 => Some(&CODEC_Q4_0),
        QuantFormat::GGUF_Q8_0 => Some(&CODEC_Q8_0),
        // IQ2_S and IQ2_M share one block layout.
        QuantFormat::GGUF_IQ2_S => codec(QuantFormat::GGUF_IQ2_M),
        f => REGISTRY.iter().find(|c| c.format == f),
    }
}

/// Optional wide decode for formats whose 8-element sub-blocks share bytes (2-bit planes, K-quant nibble
/// and high-bit planes: several sub-blocks are bit fields of the same few bytes) and whose elements are
/// `scale * code + bias` per sub-block. `decode8` re-reads those bytes once per sub-block and applies the
/// affine map per element; a wide codec decodes a whole group once, as raw codes, and the GEMV applies
/// `scale * sum(code * x) + bias * sum(x)` per segment (the way ggml's mul_mv kernels do), which is a
/// different rounding than `decode8`, so GEMV parity is tolerance-based. A block has `groups` groups of
/// `segs` segments of 8 elements.
pub struct Wide {
    pub groups: u32,
    pub segs: u32,
    /// Every segment of a group shares one scale (`out[8 * segs]`) and has bias `-scale`: the GEMV sums
    /// the group's dots and x once and multiplies by the scale a single time.
    pub uniform: bool,
    /// MSL defining `inline uint group_seg(uint g, uint k)` (block offset of segment `k` of group `g`) and
    /// `inline void decode_group(device const uchar* blk, uint g, thread float* out)`, which writes the
    /// `8 * segs` raw codes of the group in segment order, then `segs` scales, then `segs` biases.
    pub msl: &'static str,
}

pub fn wide(format: QuantFormat) -> Option<&'static Wide> {
    match format {
        QuantFormat::GGUF_Q6_K => Some(&WIDE_Q6_K),
        QuantFormat::GGUF_Q4_K_M => Some(&WIDE_Q4_K),
        QuantFormat::GGUF_Q5_K_M => Some(&WIDE_Q5_K),
        QuantFormat::GGUF_Q3_K => Some(&WIDE_Q3_K),
        QuantFormat::GGUF_IQ4_XS => Some(&WIDE_IQ4_XS),
        QuantFormat::GGUF_IQ4_NL => Some(&WIDE_IQ4_NL),
        _ => None,
    }
}

/// A planar-layout codec decoded one lane-tile at a time (`groups` tiles per block, each loaded whole and
/// decoded per segment of 8 elements, so only a few registers are live at once) where every element is
/// `scale * (code - 1)` with one scale per tile. `tile_seg` writes a value `u` per element (not necessarily
/// the code: whatever the cheapest bit trick to make a float gives) with `code - 1 == a * u - b`, so the GEMV
/// applies `scale * (a * sum(u * x) - b * sum(x))`, the same algebra as a `uniform` [`Wide`] (and the same
/// non-bit-exact rounding vs `decode8`).
pub struct Tile {
    pub groups: u32,
    pub segs: u32,
    pub a: f32,
    pub b: f32,
    /// MSL defining `uint group_seg(uint g, uint k)` (block offset of segment `k` of tile `g`),
    /// `uint4 tile_load(device const uchar* grp, uint g)` (the tile's bytes), `float tile_scale(grp, g)` and
    /// `void tile_seg(uint4 q, uint k, thread float* out8)` (`u` of segment `k`; `k` is a compile-time
    /// constant at every call). `grp` is the block's [`Planar`] group and `g` continues across the group's
    /// blocks (`(i % blocks) * groups + tile`); `group_seg` takes the tile within the block.
    pub msl: &'static str,
}

pub fn tile(format: QuantFormat) -> Option<&'static Tile> {
    match format {
        QuantFormat::GGUF_TQ2_0 => Some(&TILE_TQ2_0),
        _ => None,
    }
}

/// TQ2_0: byte `j` of a 32-byte plane holds elements `j, j+32, j+64, j+96` of its 128-element half
/// (value = `(code - 1) * d`), so tile `g` of a block (16 bytes at `(g>>1)*32 + (g&1)*16`) is, for shift
/// level `k` of 4, two segments of 8 elements at `(g>>1)*128 + k*32 + (g&1)*16`; segment `s` is level `s>>1`,
/// chunk `s&1`. Under [`PLANAR_TQ2_0`] the tile is one aligned 16-byte load and the scale is in the group's
/// scale plane. `u` is the float `1 + code/4` (the 2-bit code moved into the top two mantissa bits of 1.0f:
/// one shift, one mask, one or per element instead of a byte-to-float conversion), so `code - 1 == 4u - 5`.
static TILE_TQ2_0: Tile = Tile {
    groups: 4,
    segs: 8,
    a: 4.0,
    b: 5.0,
    msl: r#"
inline uint group_seg(uint g, uint k) { return (g >> 1) * 128 + (k >> 1) * 32 + (g & 1) * 16 + (k & 1) * 8; }
inline uint4 tile_load(device const uchar* grp, uint g) {
    return *(device const uint4*)(grp + (g >> 2) * 64 + (g & 3) * 16);
}
inline float tile_scale(device const uchar* grp, uint g) {
    return float(*(device const half*)(grp + 512 + (g >> 2) * 2));
}
inline void tile_seg(uint4 q, uint k, thread float* out) {
    const int lv = int(k >> 1) * 2;
    for (uint w = 0; w < 2; w++) {
        const uint x = q[(k & 1) * 2 + w];
        for (int j = 0; j < 4; j++) {
            // The field of byte j at level lv is bits 8j+lv..+1; mantissa bits 22..21 are where it goes.
            const int sh = 21 - 8 * j - lv;
            const uint y = sh >= 0 ? (x << sh) : (x >> -sh);
            out[4 * w + j] = as_type<float>(0x3F800000u | (y & 0x00600000u));
        }
    }
}
"#,
};

/// Device layout of a codec whose block is `[lead][data][tail]` (`lead`/`tail` hold the scales; `data` the
/// quants, a multiple of 16 bytes): the GPU weight buffer stores `blocks` consecutive blocks as one group,
/// all their data bytes first and then all their lead + tail bytes (per block, lead first), `blocks *
/// block_bytes` bytes per group, so the quants of block `i` start at `(i / blocks) * group_bytes + (i %
/// blocks) * data`, which is 16-byte aligned, instead of at `i * block_bytes`; the scales of block `i` are at
/// `blocks * data + (i % blocks) * (block_bytes - data)` of its group. The last group is zero-padded. Blocks
/// are numbered across rows (and expert banks), so a row never has to be a whole number of groups.
/// [`Planar::msl`] replaces [`Codec::msl`] for every GPU consumer of the buffer: its `decode8(grp, sub, out8)`
/// takes the group base and a `sub` that continues across the group's blocks (`(i % blocks) * block_elems / 8
/// + sub_in_block`); [`Wide`] codecs replace their MSL by `wide_msl` (same functions, `decode_group` takes
/// `(i % blocks) * groups + g`), and [`Tile`]'s functions likewise take `(i % blocks) * groups + g`.
/// [`Codec::msl`] stays the as-stored (GGUF) layout that `tests/codec_parity.rs` proves against the CPU
/// dequantizer.
pub struct Planar {
    pub blocks: u32,
    pub data: u32,
    /// Scale bytes before `data` in the as-stored block.
    pub lead: u32,
    pub msl: &'static str,
    /// The [`Wide`] MSL for this layout; empty for a codec without a wide decode.
    pub wide_msl: &'static str,
}

pub fn planar(format: QuantFormat) -> Option<&'static Planar> {
    match format {
        QuantFormat::GGUF_TQ2_0 => Some(&PLANAR_TQ2_0),
        QuantFormat::GGUF_Q3_K => Some(&PLANAR_Q3_K),
        QuantFormat::GGUF_Q5_K_M => Some(&PLANAR_Q5_K),
        _ => None,
    }
}

impl Planar {
    /// `raw` (whole as-stored blocks of `block_bytes`) in the device layout.
    pub fn repack(&self, raw: &[u8], block_bytes: usize) -> Vec<u8> {
        let (g, data, lead) = (self.blocks as usize, self.data as usize, self.lead as usize);
        let scales = block_bytes - data;
        let group = g * block_bytes;
        let mut out = vec![0u8; (raw.len() / block_bytes).div_ceil(g) * group];
        for (dst, src) in out.chunks_exact_mut(group).zip(raw.chunks(group)) {
            for (j, b) in src.chunks_exact(block_bytes).enumerate() {
                dst[j * data..][..data].copy_from_slice(&b[lead..lead + data]);
                let s = &mut dst[g * data + j * scales..][..scales];
                s[..lead].copy_from_slice(&b[..lead]);
                s[lead..].copy_from_slice(&b[lead + data..]);
            }
        }
        out
    }
}

/// TQ2_0 blocks are 64 data bytes and an f16 scale: 8 blocks are 512 data bytes then 8 scales.
static PLANAR_TQ2_0: Planar = Planar {
    blocks: 8,
    data: 64,
    lead: 0,
    wide_msl: "",
    msl: r#"
inline void decode8(device const uchar* grp, uint sub, thread float* out8) {
    const uint n = sub >> 5;
    float d = float(*(device const half*)(grp + 512 + n * 2));
    uint e0 = (sub & 31) * 8;
    // Eight consecutive elements are eight consecutive bytes that share one shift: one aligned 8-byte load.
    uint2 q = *(device const uint2*)(grp + n * 64 + (e0 >> 7) * 32 + (e0 & 31));
    uint sh = ((e0 >> 5) & 3) * 2;
    uint a = (q.x >> sh) & 0x03030303u;
    uint b = (q.y >> sh) & 0x03030303u;
    float4 lo = (float4(as_type<uchar4>(a)) - 1.0f) * d;
    float4 hi = (float4(as_type<uchar4>(b)) - 1.0f) * d;
    out8[0] = lo.x; out8[1] = lo.y; out8[2] = lo.z; out8[3] = lo.w;
    out8[4] = hi.x; out8[5] = hi.y; out8[6] = hi.z; out8[7] = hi.w;
}
"#,
};

/// Q5_K blocks are `[d, dmin, 12 scale bytes][32 qh][128 qs]`: 8 blocks are 1280 data bytes (qh, qs) then 8
/// 16-byte scale heads. Every 8-byte group load of `decode_group` is one aligned `uint2`.
static PLANAR_Q5_K: Planar = Planar {
    blocks: 8,
    data: 160,
    lead: 16,
    msl: r#"
inline void decode8(device const uchar* grp, uint sub, thread float* out8) {
    #pragma clang fp contract(off)
    const uint bj = sub >> 5, s = sub & 31;
    device const uchar* sc = grp + 1280 + bj * 16;
    float d = cd_f16(sc), dmin = cd_f16(sc + 2);
    uint j = s >> 2;
    uint sv, mv;
    cd_k4(j, sc + 4, sv, mv);
    float d0 = d * float(sv), min0 = dmin * float(mv);
    device const uchar* data = grp + bj * 160;
    device const uchar* qh = data + (s & 3) * 8;
    device const uchar* q = data + 32 + (j >> 1) * 32 + (s & 3) * 8;
    uint sh = (j & 1) * 4;
    for (uint i = 0; i < 8; i++) {
        uint v = ((q[i] >> sh) & 0x0Fu) | (((uint(qh[i]) >> j) & 1u) << 4);
        out8[i] = d0 * float(v) - min0;
    }
}
"#,
    wide_msl: r#"
inline uint group_seg(uint g, uint k) { return ((g >> 2) * 2 + k) * 32 + (g & 3) * 8; }
inline void decode_group(device const uchar* grp, uint gg, thread float* out) {
    const uint bj = gg >> 4, g = gg & 15;
    device const uchar* sc = grp + 1280 + bj * 16;
    float d = cd_f16(sc), dmin = cd_f16(sc + 2);
    uint p = g >> 2, l0 = (g & 3) * 8;
    device const uchar* data = grp + bj * 160;
    const uint2 q = *(device const uint2*)(data + 32 + p * 32 + l0);
    const uint2 qh = *(device const uint2*)(data + l0);
    for (uint w = 0; w < 2; w++) {
        uint a = q[w];
        uint h = qh[w];
        float4 lo = float4(as_type<uchar4>((a & 0x0F0F0F0Fu) | (((h >> (2 * p)) & 0x01010101u) << 4)));
        float4 hi = float4(as_type<uchar4>(((a >> 4) & 0x0F0F0F0Fu) | (((h >> (2 * p + 1)) & 0x01010101u) << 4)));
        for (uint i = 0; i < 4; i++) { out[4 * w + i] = lo[i]; out[8 + 4 * w + i] = hi[i]; }
    }
    for (uint k = 0; k < 2; k++) {
        uint sv, mv;
        cd_k4(2 * p + k, sc + 4, sv, mv);
        out[16 + k] = d * float(sv);
        out[18 + k] = -(dmin * float(mv));
    }
}
"#,
};

/// Q3_K blocks are `[32 hmask][64 qs][12 scale bytes][d]`: 8 blocks are 768 data bytes (hmask, qs) then 8
/// 14-byte scale tails; every 8-byte group load of `decode_group` is one aligned `uint2`.
static PLANAR_Q3_K: Planar = Planar {
    blocks: 8,
    data: 96,
    lead: 0,
    msl: r#"
inline void decode8(device const uchar* grp, uint sub, thread float* out8) {
    const uint bj = sub >> 5, s = sub & 31;
    device const uchar* data = grp + bj * 96;
    device const uchar* tail = grp + 768 + bj * 14;
    float d_all = cd_f16(tail + 12);
    uint a0 = cd_u32(tail), a1 = cd_u32(tail + 4), a2 = cd_u32(tail + 8);
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    uint is = s >> 1;
    uint w = is >> 2;
    uint aux = w == 0 ? ((a0 & k2) | ((a2 & k1) << 4))
             : w == 1 ? ((a1 & k2) | (((a2 >> 2) & k1) << 4))
             : w == 2 ? (((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4))
                      : (((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4));
    int scale = int(as_type<char>(uchar((aux >> (8 * (is & 3))) & 0xFFu))) - 32;
    float dl = d_all * float(scale);
    uint n = s >> 4, si = (s >> 2) & 3, half_ = (s >> 1) & 1;
    uint m = 1u << (n * 4 + si);
    for (uint i = 0; i < 8; i++) {
        uint k = (s & 1) * 8 + i + 16 * half_;
        int h = (data[k] & m) ? 0 : 4;
        int qv = int((data[32 + n * 32 + k] >> (2 * si)) & 3u);
        out8[i] = dl * float(qv - h);
    }
}
"#,
    wide_msl: r#"
inline uint group_seg(uint g, uint k) { return (g >> 2) * 128 + k * 32 + (g & 3) * 8; }
inline void decode_group(device const uchar* grp, uint gg, thread float* out) {
    const uint bj = gg >> 3, g = gg & 7;
    device const uchar* data = grp + bj * 96;
    device const uchar* tail = grp + 768 + bj * 14;
    float d_all = cd_f16(tail + 12);
    uint n = g >> 2, l0 = (g & 3) * 8;
    const uint2 q = *(device const uint2*)(data + 32 + n * 32 + l0);
    const uint2 hm = *(device const uint2*)(data + l0);
    for (uint w = 0; w < 2; w++) {
        uint a = q[w];
        uint h = hm[w];
        for (uint k = 0; k < 4; k++) {
            float4 c = float4(as_type<uchar4>(((a >> (2 * k)) & 0x03030303u) | (((h >> (n * 4 + k)) & 0x01010101u) << 2)));
            for (uint i = 0; i < 4; i++) out[8 * k + 4 * w + i] = c[i];
        }
    }
    uint a0 = cd_u32(tail), a1 = cd_u32(tail + 4), a2 = cd_u32(tail + 8);
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    for (uint k = 0; k < 4; k++) {
        uint is = n * 8 + 2 * k + (l0 >> 4);
        uint w = is >> 2;
        uint aux = w == 0 ? ((a0 & k2) | ((a2 & k1) << 4))
                 : w == 1 ? ((a1 & k2) | (((a2 >> 2) & k1) << 4))
                 : w == 2 ? (((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4))
                          : (((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4));
        float s = d_all * float(int(as_type<char>(uchar((aux >> (8 * (is & 3))) & 0xFFu))) - 32);
        out[32 + k] = s;
        out[36 + k] = -4.0f * s;
    }
}
"#,
};

static WIDE_Q6_K: Wide = Wide {
    uniform: false,
    groups: 8,
    segs: 4,
    // Group g = (half n = g/4, 8 bytes at l0 = (g%4)*8): ql[l] and ql[l+32] hold the low / high nibbles of
    // segments 0,2 / 1,3, and qh[l] their top two bits; segment k is elements n*128 + k*32 + l0..l0+8 with
    // scale d * sc[l0/16 + 2k] and value (code - 32) * scale.
    msl: r#"
inline uint group_seg(uint g, uint k) { return (g >> 2) * 128 + k * 32 + (g & 3) * 8; }
inline void decode_group(device const uchar* blk, uint g, thread float* out) {
    float d = cd_f16(blk + 208);
    uint n = g >> 2, l0 = (g & 3) * 8;
    device const ushort* lo = (device const ushort*)(blk + n * 64 + l0);
    device const ushort* hi = (device const ushort*)(blk + n * 64 + 32 + l0);
    device const ushort* qh = (device const ushort*)(blk + 128 + n * 32 + l0);
    for (uint w = 0; w < 2; w++) {
        uint a = uint(lo[2 * w]) | (uint(lo[2 * w + 1]) << 16);
        uint b = uint(hi[2 * w]) | (uint(hi[2 * w + 1]) << 16);
        uint h = uint(qh[2 * w]) | (uint(qh[2 * w + 1]) << 16);
        float4 c0 = float4(as_type<uchar4>((a & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4)));
        float4 c1 = float4(as_type<uchar4>((b & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2)));
        float4 c2 = float4(as_type<uchar4>(((a >> 4) & 0x0F0F0F0Fu) | (h & 0x30303030u)));
        float4 c3 = float4(as_type<uchar4>(((b >> 4) & 0x0F0F0F0Fu) | ((h & 0xC0C0C0C0u) >> 2)));
        for (uint i = 0; i < 4; i++) {
            out[4 * w + i] = c0[i];
            out[8 + 4 * w + i] = c1[i];
            out[16 + 4 * w + i] = c2[i];
            out[24 + 4 * w + i] = c3[i];
        }
    }
    device const char* sc = (device const char*)(blk + 192 + n * 8 + (l0 >> 4));
    for (uint k = 0; k < 4; k++) {
        float s = d * float(sc[2 * k]);
        out[32 + k] = s;
        out[36 + k] = -32.0f * s;
    }
}
"#,
};

// Q4_K / Q5_K: group g = (pair p = g/4 of 32-element groups j = 2p, 2p+1, 8 bytes at l0 = (g%4)*8): the low
// nibbles are group 2p and the high nibbles group 2p+1 (Q5_K adds bit j of the 32 high-bit bytes as the
// fifth bit); value = d * sv_j * code - dmin * mv_j.
static WIDE_Q4_K: Wide = Wide {
    uniform: false,
    groups: 16,
    segs: 2,
    msl: r#"
inline uint group_seg(uint g, uint k) { return ((g >> 2) * 2 + k) * 32 + (g & 3) * 8; }
inline void decode_group(device const uchar* blk, uint g, thread float* out) {
    float d = cd_f16(blk), dmin = cd_f16(blk + 2);
    uint p = g >> 2;
    device const ushort* q = (device const ushort*)(blk + 16 + p * 32 + (g & 3) * 8);
    for (uint w = 0; w < 2; w++) {
        uint a = uint(q[2 * w]) | (uint(q[2 * w + 1]) << 16);
        float4 lo = float4(as_type<uchar4>(a & 0x0F0F0F0Fu));
        float4 hi = float4(as_type<uchar4>((a >> 4) & 0x0F0F0F0Fu));
        for (uint i = 0; i < 4; i++) { out[4 * w + i] = lo[i]; out[8 + 4 * w + i] = hi[i]; }
    }
    for (uint k = 0; k < 2; k++) {
        uint sv, mv;
        cd_k4(2 * p + k, blk + 4, sv, mv);
        out[16 + k] = d * float(sv);
        out[18 + k] = -(dmin * float(mv));
    }
}
"#,
};

static WIDE_Q5_K: Wide = Wide {
    uniform: false,
    groups: 16,
    segs: 2,
    msl: r#"
inline uint group_seg(uint g, uint k) { return ((g >> 2) * 2 + k) * 32 + (g & 3) * 8; }
inline void decode_group(device const uchar* blk, uint g, thread float* out) {
    float d = cd_f16(blk), dmin = cd_f16(blk + 2);
    uint p = g >> 2, l0 = (g & 3) * 8;
    device const ushort* q = (device const ushort*)(blk + 48 + p * 32 + l0);
    device const ushort* qh = (device const ushort*)(blk + 16 + l0);
    for (uint w = 0; w < 2; w++) {
        uint a = uint(q[2 * w]) | (uint(q[2 * w + 1]) << 16);
        uint h = uint(qh[2 * w]) | (uint(qh[2 * w + 1]) << 16);
        float4 lo = float4(as_type<uchar4>((a & 0x0F0F0F0Fu) | (((h >> (2 * p)) & 0x01010101u) << 4)));
        float4 hi = float4(as_type<uchar4>(((a >> 4) & 0x0F0F0F0Fu) | (((h >> (2 * p + 1)) & 0x01010101u) << 4)));
        for (uint i = 0; i < 4; i++) { out[4 * w + i] = lo[i]; out[8 + 4 * w + i] = hi[i]; }
    }
    for (uint k = 0; k < 2; k++) {
        uint sv, mv;
        cd_k4(2 * p + k, blk + 4, sv, mv);
        out[16 + k] = d * float(sv);
        out[18 + k] = -(dmin * float(mv));
    }
}
"#,
};

// Q3_K: group g = (half n = g/4, 8 bytes at l0 = (g%4)*8): segment k is the 2-bit plane k of the 32 quant bytes
// plus bit n*4+k of the high-mask byte as bit 2; value = d * (scale - 32) * (code - 4).
static WIDE_Q3_K: Wide = Wide {
    uniform: false,
    groups: 8,
    segs: 4,
    msl: r#"
inline uint group_seg(uint g, uint k) { return (g >> 2) * 128 + k * 32 + (g & 3) * 8; }
inline void decode_group(device const uchar* blk, uint g, thread float* out) {
    float d_all = cd_f16(blk + 108);
    uint n = g >> 2, l0 = (g & 3) * 8;
    device const ushort* q = (device const ushort*)(blk + 32 + n * 32 + l0);
    device const ushort* hm = (device const ushort*)(blk + l0);
    for (uint w = 0; w < 2; w++) {
        uint a = uint(q[2 * w]) | (uint(q[2 * w + 1]) << 16);
        uint h = uint(hm[2 * w]) | (uint(hm[2 * w + 1]) << 16);
        for (uint k = 0; k < 4; k++) {
            float4 c = float4(as_type<uchar4>(((a >> (2 * k)) & 0x03030303u) | (((h >> (n * 4 + k)) & 0x01010101u) << 2)));
            for (uint i = 0; i < 4; i++) out[8 * k + 4 * w + i] = c[i];
        }
    }
    uint a0 = cd_u32(blk + 96), a1 = cd_u32(blk + 100), a2 = cd_u32(blk + 104);
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    for (uint k = 0; k < 4; k++) {
        uint is = n * 8 + 2 * k + (l0 >> 4);
        uint w = is >> 2;
        uint aux = w == 0 ? ((a0 & k2) | ((a2 & k1) << 4))
                 : w == 1 ? ((a1 & k2) | (((a2 >> 2) & k1) << 4))
                 : w == 2 ? (((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4))
                          : (((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4));
        float s = d_all * float(int(as_type<char>(uchar((aux >> (8 * (is & 3))) & 0xFFu))) - 32);
        out[32 + k] = s;
        out[36 + k] = -4.0f * s;
    }
}
"#,
};

// IQ4_XS: group g = (32-element group ib = g/2, 8 bytes at (g%2)*8): low nibbles are elements 0..15 of the
// group's half, high nibbles 16..31; value = d * (ls - 32) * KVALUES[nibble] (a table, so no bias).
static WIDE_IQ4_XS: Wide = Wide {
    uniform: false,
    groups: 16,
    segs: 2,
    msl: r#"
inline uint group_seg(uint g, uint k) { return (g >> 1) * 32 + (g & 1) * 8 + k * 16; }
inline void decode_group(device const uchar* blk, uint g, thread float* out) {
    float d = cd_f16(blk);
    uint scales_h = cd_u16(blk + 2);
    uint ib = g >> 1;
    device const uchar* q = blk + 8 + 16 * ib + (g & 1) * 8;
    for (uint i = 0; i < 8; i++) {
        out[i] = float(KVALUES_IQ4NL[q[i] & 0x0Fu]);
        out[8 + i] = float(KVALUES_IQ4NL[q[i] >> 4]);
    }
    uint ls = ((uint(blk[4 + (ib >> 1)]) >> (4 * (ib & 1))) & 0x0Fu) | (((scales_h >> (2 * ib)) & 3u) << 4);
    float s = d * float(int(ls) - 32);
    out[16] = s; out[17] = s; out[18] = 0.0f; out[19] = 0.0f;
}
"#,
};

static WIDE_IQ4_NL: Wide = Wide {
    uniform: false,
    groups: 2,
    segs: 2,
    msl: r#"
inline uint group_seg(uint g, uint k) { return g * 8 + k * 16; }
inline void decode_group(device const uchar* blk, uint g, thread float* out) {
    float d = cd_f16(blk);
    device const uchar* q = blk + 2 + g * 8;
    for (uint i = 0; i < 8; i++) {
        out[i] = float(KVALUES_IQ4NL[q[i] & 0x0Fu]);
        out[8 + i] = float(KVALUES_IQ4NL[q[i] >> 4]);
    }
    out[16] = d; out[17] = d; out[18] = 0.0f; out[19] = 0.0f;
}
"#,
};

pub const CODEC_Q4_0: Codec = Codec {
    format: QuantFormat::GGUF_Q4_0,
    block_elems: 32,
    block_bytes: 18,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = float(*(device const half*)blk);
    device const uchar* q = blk + 2 + (sub & 1) * 8;
    uint sh = (sub >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(int((q[i] >> sh) & 0x0F) - 8) * d;
    }
}
"#,
};

pub const CODEC_Q8_0: Codec = Codec {
    format: QuantFormat::GGUF_Q8_0,
    block_elems: 32,
    block_bytes: 34,
    msl: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    half d = *(device const half*)blk;
    device const char* q = (device const char*)(blk + 2) + sub * 8;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(d) * float(q[i]);
    }
}
"#,
};

/// One registry entry. `tables` name entries of [`msl_prelude`]; `body` defines `decode8` (helpers from
/// `HELPERS` are in scope) and may declare its own `#ifndef`-guarded constants.
pub struct Spec {
    pub format: QuantFormat,
    pub block_elems: u32,
    pub block_bytes: u32,
    pub tables: &'static [&'static str],
    pub body: &'static str,
}

/// Every spec: the GGUF ones below plus the repacked source encodings of `model::repack`.
fn all_specs() -> impl Iterator<Item = &'static Spec> {
    use crate::model::repack::{affine, bnb, fp8};
    SPECS
        .iter()
        .chain(bnb::SPECS)
        .chain(fp8::SPECS)
        .chain(affine::SPECS)
}

/// Full MSL (helpers + tables + body) per entry of [`all_specs`], owned by a static so the codecs can borrow it.
static MSL: LazyLock<Vec<String>> = LazyLock::new(|| {
    all_specs()
        .map(|s| format!("{HELPERS}{}{}", tables_msl(s.tables), s.body))
        .collect()
});

static REGISTRY: LazyLock<Vec<Codec>> = LazyLock::new(|| {
    all_specs()
        .zip(MSL.iter())
        .map(|(s, msl)| Codec {
            format: s.format,
            block_elems: s.block_elems,
            block_bytes: s.block_bytes,
            msl: msl.as_str(),
        })
        .collect()
});

pub const HELPERS: &str = r#"
#ifndef GIBBS_CODEC_HELPERS
#define GIBBS_CODEC_HELPERS
inline uint cd_u16(device const uchar* p) { return uint(*(device const ushort*)p); }
inline uint cd_u32(device const uchar* p) { return cd_u16(p) | (cd_u16(p + 2) << 16); }
inline float cd_f16(device const uchar* p) { return float(*(device const half*)p); }
inline float cd_sgn(uint s, uint j) { return (s & (1u << j)) ? -1.0f : 1.0f; }
inline float cd_gb(ulong g, uint j) { return float(uint((g >> (8 * j)) & 0xFFul)); }
inline float cd_gb_i8(ulong g, uint j) { return float(int(as_type<char>(uchar((g >> (8 * j)) & 0xFFul)))); }
// ggml get_scale_min_k4: 6-bit (scale, min) pair j of a K-quant block's 12 scale bytes.
inline void cd_k4(uint j, device const uchar* s, thread uint& sv, thread uint& mv) {
    if (j < 4) {
        sv = s[j] & 63u;
        mv = s[j + 4] & 63u;
    } else {
        sv = (s[j + 4] & 0x0Fu) | ((s[j - 4] >> 6) << 4);
        mv = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}
#endif
"#;

fn arr<T: LowerHex>(name: &str, ty: &str, items: &[T], lit: impl Fn(&T) -> String) -> String {
    let body: Vec<String> = items.iter().map(lit).collect();
    format!(
        "#ifndef GIBBS_TBL_{name}\n#define GIBBS_TBL_{name}\nconstant {ty} {name}[{}] = {{\n{}\n}};\n#endif\n",
        items.len(),
        body.join(",")
    )
}

/// MSL for one named table, generated from the CPU tables in `src/model/gguf_tables.rs`.
fn table_msl(name: &str) -> String {
    let u64h = |v: &u64| format!("{v:#018x}ul");
    let u32h = |v: &u32| format!("{v:#010x}u");
    let dec = |v: &dyn std::fmt::Display| v.to_string();
    match name {
        "ksigns" => arr("KSIGNS_IQ2XS", "uchar", &KSIGNS_IQ2XS, |v| dec(v)),
        "iq2xxs" => arr("IQ2XXS_GRID", "ulong", &IQ2XXS_GRID, u64h),
        "iq2xs" => arr("IQ2XS_GRID", "ulong", &IQ2XS_GRID, u64h),
        "iq2s" => arr("IQ2S_GRID", "ulong", &IQ2S_GRID, u64h),
        "iq3xxs" => arr("IQ3XXS_GRID", "uint", &IQ3XXS_GRID, u32h),
        "iq3s" => arr("IQ3S_GRID", "uint", &IQ3S_GRID, u32h),
        "iq1s" => arr("IQ1S_GRID", "ulong", &IQ1S_GRID, u64h),
        "iq4nl" => arr("KVALUES_IQ4NL", "char", &KVALUES_IQ4NL, |v| dec(v)),
        "fp4" => arr("KVALUES_FP4", "char", &KVALUES_FP4, |v| dec(v)),
        _ => unreachable!("unknown codec table {name}"),
    }
}

const ALL_TABLES: &[&str] = &[
    "ksigns", "iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq1s", "iq4nl", "fp4",
];

fn tables_msl(names: &[&str]) -> String {
    names.iter().map(|n| table_msl(n)).collect()
}

/// Helpers plus every IQ grid, sign and kvalue table as `constant` arrays (each guarded).
pub fn msl_prelude() -> String {
    format!("{HELPERS}{}", tables_msl(ALL_TABLES))
}

const SPECS: &[Spec] = &[
    Spec {
        format: QuantFormat::GGUF_Q4_1,
        block_elems: 32,
        block_bytes: 20,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    #pragma clang fp contract(off)
    float d = cd_f16(blk), m = cd_f16(blk + 2);
    device const uchar* q = blk + 4 + (sub & 1) * 8;
    uint sh = (sub >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float((q[i] >> sh) & 0x0F) * d + m;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q5_0,
        block_elems: 32,
        block_bytes: 22,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint qh = cd_u32(blk + 2);
    device const uchar* q = blk + 6 + (sub & 1) * 8;
    uint sh = (sub >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        uint v = ((q[i] >> sh) & 0x0Fu) | (((qh >> (sub * 8 + i)) & 1u) << 4);
        out8[i] = float(int(v) - 16) * d;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q5_1,
        block_elems: 32,
        block_bytes: 24,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    #pragma clang fp contract(off)
    float d = cd_f16(blk), m = cd_f16(blk + 2);
    uint qh = cd_u32(blk + 4);
    device const uchar* q = blk + 8 + (sub & 1) * 8;
    uint sh = (sub >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        uint v = ((q[i] >> sh) & 0x0Fu) | (((qh >> (sub * 8 + i)) & 1u) << 4);
        out8[i] = float(v) * d + m;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q4_K_M,
        block_elems: 256,
        block_bytes: 144,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    #pragma clang fp contract(off)
    float d = cd_f16(blk), dmin = cd_f16(blk + 2);
    uint j = sub >> 2;  // 32-element group: 2 * (64-element group) + (high nibble)
    uint sv, mv;
    cd_k4(j, blk + 4, sv, mv);
    float d0 = d * float(sv), min0 = dmin * float(mv);
    device const uchar* q = blk + 16 + (j >> 1) * 32 + (sub & 3) * 8;
    uint sh = (j & 1) * 4;
    for (uint i = 0; i < 8; i++) {
        out8[i] = d0 * float((q[i] >> sh) & 0x0Fu) - min0;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q5_K_M,
        block_elems: 256,
        block_bytes: 176,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    #pragma clang fp contract(off)
    float d = cd_f16(blk), dmin = cd_f16(blk + 2);
    uint j = sub >> 2;
    uint sv, mv;
    cd_k4(j, blk + 4, sv, mv);
    float d0 = d * float(sv), min0 = dmin * float(mv);
    device const uchar* qh = blk + 16 + (sub & 3) * 8;
    device const uchar* q = blk + 48 + (j >> 1) * 32 + (sub & 3) * 8;
    uint sh = (j & 1) * 4;
    for (uint i = 0; i < 8; i++) {
        uint v = ((q[i] >> sh) & 0x0Fu) | (((uint(qh[i]) >> j) & 1u) << 4);
        out8[i] = d0 * float(v) - min0;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q6_K,
        block_elems: 256,
        block_bytes: 210,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk + 208);
    uint n = sub >> 4;           // 128-element half
    uint e = (sub & 15) * 8;     // offset within the half
    uint qd = e >> 5;            // quarter of the half
    uint l0 = e & 31;
    device const uchar* ql = blk + n * 64;
    device const uchar* qh = blk + 128 + n * 32;
    device const char* sc = (device const char*)(blk + 192 + n * 8);
    for (uint i = 0; i < 8; i++) {
        uint l = l0 + i;
        uint lo = (qd & 1) ? uint(ql[l + 32]) : uint(ql[l]);
        lo = (qd >> 1) ? (lo >> 4) : (lo & 0x0Fu);
        uint hb = (uint(qh[l]) >> (2 * qd)) & 3u;
        int q = int(lo | (hb << 4)) - 32;
        out8[i] = d * float(sc[(l >> 4) + 2 * qd]) * float(q);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q2_K,
        block_elems: 256,
        block_bytes: 84,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    #pragma clang fp contract(off)
    float d = cd_f16(blk + 80), mn = cd_f16(blk + 82);
    uint sc = blk[sub >> 1];
    float dl = d * float(sc & 0x0Fu), ml = mn * float(sc >> 4);
    device const uchar* q = blk + 16 + (sub >> 4) * 32 + ((sub >> 1) & 1) * 16 + (sub & 1) * 8;
    uint sh = ((sub >> 2) & 3) * 2;
    for (uint i = 0; i < 8; i++) {
        out8[i] = dl * float((q[i] >> sh) & 3u) - ml;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_Q3_K,
        block_elems: 256,
        block_bytes: 110,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d_all = cd_f16(blk + 108);
    uint a0 = cd_u32(blk + 96), a1 = cd_u32(blk + 100), a2 = cd_u32(blk + 104);
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    uint is = sub >> 1;
    uint w = is >> 2;
    uint aux = w == 0 ? ((a0 & k2) | ((a2 & k1) << 4))
             : w == 1 ? ((a1 & k2) | (((a2 >> 2) & k1) << 4))
             : w == 2 ? (((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4))
                      : (((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4));
    int scale = int(as_type<char>(uchar((aux >> (8 * (is & 3))) & 0xFFu))) - 32;
    float dl = d_all * float(scale);
    uint n = sub >> 4, si = (sub >> 2) & 3, half_ = (sub >> 1) & 1;
    uint m = 1u << (n * 4 + si);
    for (uint i = 0; i < 8; i++) {
        uint k = (sub & 1) * 8 + i + 16 * half_;
        int h = (blk[k] & m) ? 0 : 4;
        int qv = int((blk[32 + n * 32 + k] >> (2 * si)) & 3u);
        out8[i] = dl * float(qv - h);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ4_NL,
        block_elems: 32,
        block_bytes: 18,
        tables: &["iq4nl"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    device const uchar* q = blk + 2 + (sub & 1) * 8;
    uint sh = (sub >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        out8[i] = d * float(KVALUES_IQ4NL[(q[i] >> sh) & 0x0Fu]);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ4_XS,
        block_elems: 256,
        block_bytes: 136,
        tables: &["iq4nl"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint scales_h = cd_u16(blk + 2);
    uint ib = sub >> 2, t = sub & 3;
    uint ls = ((uint(blk[4 + (ib >> 1)]) >> (4 * (ib & 1))) & 0x0Fu) | (((scales_h >> (2 * ib)) & 3u) << 4);
    float dl = d * float(int(ls) - 32);
    device const uchar* q = blk + 8 + 16 * ib + (t & 1) * 8;
    uint sh = (t >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        out8[i] = dl * float(KVALUES_IQ4NL[(q[i] >> sh) & 0x0Fu]);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ3_XXS,
        block_elems: 256,
        block_bytes: 98,
        tables: &["ksigns", "iq3xxs"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint ib = sub >> 2, l = sub & 3;
    uint aux = cd_u32(blk + 66 + 4 * ib);
    float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    uint s = KSIGNS_IQ2XS[(aux >> (7 * l)) & 127u];
    ulong g1 = IQ3XXS_GRID[blk[2 + 2 * sub]];
    ulong g2 = IQ3XXS_GRID[blk[3 + 2 * sub]];
    for (uint j = 0; j < 4; j++) {
        out8[j] = db * cd_gb(g1, j) * cd_sgn(s, j);
        out8[j + 4] = db * cd_gb(g2, j) * cd_sgn(s, j + 4);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ3_S,
        block_elems: 256,
        block_bytes: 110,
        tables: &["iq3s"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint ib = sub >> 2, l = sub & 3;
    uint sc = (uint(blk[106 + (ib >> 1)]) >> (4 * (ib & 1))) & 0x0Fu;
    float db = d * float(1 + 2 * int(sc));
    uint h = blk[66 + ib];
    ulong g1 = IQ3S_GRID[uint(blk[2 + 2 * sub]) | ((h << (8 - 2 * l)) & 256u)];
    ulong g2 = IQ3S_GRID[uint(blk[3 + 2 * sub]) | ((h << (7 - 2 * l)) & 256u)];
    uint s = blk[74 + sub];
    for (uint j = 0; j < 4; j++) {
        out8[j] = db * cd_gb(g1, j) * cd_sgn(s, j);
        out8[j + 4] = db * cd_gb(g2, j) * cd_sgn(s, j + 4);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ2_XXS,
        block_elems: 256,
        block_bytes: 66,
        tables: &["ksigns", "iq2xxs"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint ib = sub >> 2, l = sub & 3;
    device const uchar* q = blk + 2 + 8 * ib;
    uint aux1 = cd_u32(q + 4);
    float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
    ulong g = IQ2XXS_GRID[q[l]];
    uint s = KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 127u];
    for (uint j = 0; j < 8; j++) {
        out8[j] = db * cd_gb(g, j) * cd_sgn(s, j);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ2_XS,
        block_elems: 256,
        block_bytes: 74,
        tables: &["ksigns", "iq2xs"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint ib = sub >> 2, l = sub & 3;
    uint sc = blk[66 + ib];
    float db = d * (0.5f + float((l < 2) ? (sc & 0x0Fu) : (sc >> 4))) * 0.25f;
    uint q = cd_u16(blk + 2 + 2 * sub);
    ulong g = IQ2XS_GRID[q & 511u];
    uint s = KSIGNS_IQ2XS[q >> 9];
    for (uint j = 0; j < 8; j++) {
        out8[j] = db * cd_gb(g, j) * cd_sgn(s, j);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ2_M,
        block_elems: 256,
        block_bytes: 82,
        tables: &["iq2s"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint ib = sub >> 2, l = sub & 3;
    uint sc = blk[74 + ib];
    float db = d * (0.5f + float((l < 2) ? (sc & 0x0Fu) : (sc >> 4))) * 0.25f;
    uint idx = uint(blk[2 + sub]) | ((uint(blk[66 + ib]) << (8 - 2 * l)) & 0x300u);
    ulong g = IQ2S_GRID[idx];
    uint s = blk[34 + sub];
    for (uint j = 0; j < 8; j++) {
        out8[j] = db * cd_gb(g, j) * cd_sgn(s, j);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ1_S,
        block_elems: 256,
        block_bytes: 50,
        tables: &["iq1s"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk);
    uint ib = sub >> 2, l = sub & 3;
    uint qh = cd_u16(blk + 34 + 2 * ib);
    float dl = d * float(2 * int((qh >> 12) & 7u) + 1);
    float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    ulong g = IQ1S_GRID[uint(blk[2 + sub]) | (((qh >> (3 * l)) & 7u) << 8)];
    for (uint j = 0; j < 8; j++) {
        out8[j] = dl * (cd_gb_i8(g, j) + delta);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_IQ1_M,
        block_elems: 256,
        block_bytes: 56,
        tables: &["iq1s"],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    uint sc[4] = {cd_u16(blk + 48), cd_u16(blk + 50), cd_u16(blk + 52), cd_u16(blk + 54)};
    ushort hb = ushort((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) | ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u));
    float d = float(as_type<half>(hb));
    uint ib = sub >> 2, l = sub & 3;
    uint s = sc[ib >> 1] >> (6 * (ib & 1));
    float dl = d * float(2 * int((s >> (3 * (l >> 1))) & 7u) + 1);
    uint hh = blk[32 + 2 * ib + (l >> 1)];
    uint idx = (l & 1) ? ((hh << 4) & 0x700u) : ((hh << 8) & 0x700u);
    bool neg = (l & 1) ? ((hh & 0x80u) != 0) : ((hh & 0x08u) != 0);
    float delta = neg ? -0.125f : 0.125f;
    ulong g = IQ1S_GRID[uint(blk[sub]) | idx];
    for (uint j = 0; j < 8; j++) {
        out8[j] = dl * (cd_gb_i8(g, j) + delta);
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_TQ1_0,
        block_elems: 256,
        block_bytes: 54,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk + 52);
    for (uint i = 0; i < 8; i++) {
        uint e = sub * 8 + i, n, bi;
        if (e < 160) { n = e >> 5; bi = e & 31; }
        else if (e < 240) { e -= 160; n = e >> 4; bi = 32 + (e & 15); }
        else { e -= 240; n = e >> 2; bi = 48 + (e & 3); }
        uint p = 1;
        for (uint k = 0; k < n; k++) p *= 3;
        uint t = (uint(blk[bi]) * p) & 0xFFu;
        out8[i] = (float((t * 3u) >> 8) - 1.0f) * d;
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_TQ2_0,
        block_elems: 256,
        block_bytes: 66,
        tables: &[],
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    float d = cd_f16(blk + 64);
    uint e0 = sub * 8;
    // Eight consecutive elements are eight consecutive bytes that share one shift. Blocks are 66 bytes,
    // so only 2-byte loads are aligned: four of them, shifted and masked four bytes at a time.
    device const ushort* q = (device const ushort*)(blk + (e0 >> 7) * 32 + (e0 & 31));
    uint sh = ((e0 >> 5) & 3) * 2;
    uint a = ((uint(q[0]) | (uint(q[1]) << 16)) >> sh) & 0x03030303u;
    uint b = ((uint(q[2]) | (uint(q[3]) << 16)) >> sh) & 0x03030303u;
    float4 lo = (float4(as_type<uchar4>(a)) - 1.0f) * d;
    float4 hi = (float4(as_type<uchar4>(b)) - 1.0f) * d;
    out8[0] = lo.x; out8[1] = lo.y; out8[2] = lo.z; out8[3] = lo.w;
    out8[4] = hi.x; out8[5] = hi.y; out8[6] = hi.z; out8[7] = hi.w;
}
"#,
    },
    Spec {
        format: QuantFormat::MXFP4,
        block_elems: 32,
        block_bytes: 17,
        tables: &["fp4"],
        // d = 2^(e-128). For e < 2 that scale (and every product with it) is an f32 subnormal, which the
        // GPU's float unit flushes to zero, so that case assembles the result bits with integer ops.
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    uint e = blk[0];
    float d = as_type<float>(e < 2 ? 0u : ((e - 1u) << 23));
    device const uchar* q = blk + 1 + (sub & 1) * 8;
    uint sh = (sub >> 1) * 4;
    for (uint i = 0; i < 8; i++) {
        int k = KVALUES_FP4[(q[i] >> sh) & 0x0Fu];
        if (e >= 2) {
            out8[i] = float(k) * d;
        } else {
            uint n = uint(k < 0 ? -k : k) << (21u + e);  // |k| * d in units of 2^-149
            uint bits = n < 0x800000u ? n : (as_type<uint>(float(n)) - (149u << 23));
            out8[i] = as_type<float>(bits | (k < 0 ? 0x80000000u : 0u));
        }
    }
}
"#,
    },
    Spec {
        format: QuantFormat::GGUF_NVFP4,
        block_elems: 64,
        block_bytes: 36,
        tables: &["fp4"],
        // d = UE4M3(scale byte) * 0.5 (0 and 0x7F give 0), assembled from bits like the CPU reference.
        body: r#"
inline void decode8(device const uchar* blk, uint sub, thread float* out8) {
    uint x = blk[sub >> 1];
    uint e = (x >> 3) & 0xFu, m = x & 7u;
    float d = (x == 0u || x == 0x7Fu) ? 0.0f : (e == 0u ? float(m) / 1024.0f : as_type<float>(((e + 119u) << 23) | (m << 20)));
    device const uchar* q = blk + 4 + (sub >> 1) * 8;
    uint sh = (sub & 1) * 4;
    for (uint i = 0; i < 8; i++) {
        out8[i] = float(KVALUES_FP4[(q[i] >> sh) & 0x0Fu]) * d;
    }
}
"#,
    },
];

/// Every format [`codec`] resolves directly (IQ2_S is IQ2_M's alias and not repeated).
pub fn formats() -> Vec<QuantFormat> {
    [QuantFormat::GGUF_Q4_0, QuantFormat::GGUF_Q8_0]
        .into_iter()
        .chain(all_specs().map(|s| s.format))
        .collect()
}

/// Hash of [`CODEC_VERSION`], every codec's MSL and every generated GEMV variant built from it:
/// the key of the measured codec efficiencies.
pub fn source_hash() -> u64 {
    use crate::graph::codegen::{gemv_packed, GemvParams};
    let mut h = 0xcbf29ce484222325u64 ^ u64::from(CODEC_VERSION);
    for f in formats() {
        let c = codec(f).expect("listed formats are registered");
        let generated = crate::graph::tune::GEN_ROWS.map(|r| {
            gemv_packed(
                c,
                GemvParams {
                    rows_per_sg: r as usize,
                    ..Default::default()
                },
            )
        });
        for b in format!("{f:?}")
            .bytes()
            .chain(c.msl.bytes())
            .chain(generated.iter().flat_map(|s| s.bytes()))
        {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    h
}
