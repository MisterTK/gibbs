// GraphLm single-token decode kernels (f32). Compiled at runtime by `MetalGraphLm::new`
// via `newLibraryWithSource`; every kernel mirrors the CPU math in `GraphLm::forward_token_with_kv_scratchpad`.
#include <metal_stdlib>
using namespace metal;

// GEMV: one simdgroup per output row, 8 simdgroups (256 threads) per threadgroup.
constant uint GEMV_ROWS_PER_TG = 8;
// Attention: per-lane head dims (head_dim <= 32 * ATTN_MAX_PER_LANE) and simdgroups per threadgroup.
constant uint ATTN_MAX_PER_LANE = 16;
constant uint ATTN_MAX_SG = 8;

struct EmbedParams { uint n; uint tok_off; uint pos_off; uint has_pos; };
struct NormParams { uint n; float eps; uint layernorm; uint has_bias; uint add_residual; };
struct GemvParams { uint rows; uint cols4; uint has_bias; };
struct RopeParams { uint n_heads; uint head_dim; uint pairs; uint pair_stride; uint pair_gap; uint k_offset; };
struct KvParams { uint n; uint src_k; uint src_v; uint dst; };
struct AttnParams { uint n_heads; uint head_dim; uint stride; uint len; uint group; float scale; uint window; float softcap; };

kernel void embed_f32(device const float* emb [[buffer(0)]],
                      device const float* pos_emb [[buffer(1)]],
                      device float* x [[buffer(2)]],
                      constant EmbedParams& p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float v = emb[p.tok_off + i];
    x[i] = p.has_pos ? v + pos_emb[p.pos_off + i] : v;
}

// Sum across the whole threadgroup; every thread gets the total. Safe to call repeatedly.
static inline float block_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// Optional residual add (x += res), then RMSNorm or LayerNorm of x into out (single threadgroup).
// out may alias x: each element is read and written by the same thread.
kernel void norm_f32(device float* x [[buffer(0)]],
                     device const float* res [[buffer(1)]],
                     device const float* w [[buffer(2)]],
                     device const float* b [[buffer(3)]],
                     device float* out [[buffer(4)]],
                     constant NormParams& p [[buffer(5)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint tpg [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    if (p.add_residual) {
        for (uint i = tid; i < p.n; i += tpg) x[i] += res[i];
    }
    float mean = 0.0f;
    if (p.layernorm) {
        float s = 0.0f;
        for (uint i = tid; i < p.n; i += tpg) s += x[i];
        mean = block_sum(s, scratch, sg, lane, nsg) / float(p.n);
    }
    float sq = 0.0f;
    for (uint i = tid; i < p.n; i += tpg) {
        float d = x[i] - mean;
        sq += d * d;
    }
    float scale = 1.0f / sqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += tpg) {
        float v = (x[i] - mean) * scale * w[i];
        out[i] = p.has_bias ? v + b[i] : v;
    }
}

// y[row] = dot(W[row, :], x) (+ bias[row]); W is row-major [rows, cols4 * 4].
kernel void gemv_f32(device const float4* w [[buffer(0)]],
                     device const float4* x [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     device const float* bias [[buffer(3)]],
                     constant GemvParams& p [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * GEMV_ROWS_PER_TG + sg;
    if (row >= p.rows) return;
    device const float4* wr = w + (ulong)row * p.cols4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint i = lane;
    for (; i + 32 < p.cols4; i += 64) {
        acc0 += dot(wr[i], x[i]);
        acc1 += dot(wr[i + 32], x[i + 32]);
    }
    if (i < p.cols4) acc0 += dot(wr[i], x[i]);
    float acc = simd_sum(acc0 + acc1);
    if (lane == 0) y[row] = p.has_bias ? acc + bias[row] : acc;
}

kernel void gemv_unroll4(device const float4* w [[buffer(0)]],
                         device const float4* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         device const float* bias [[buffer(3)]],
                         constant GemvParams& p [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * GEMV_ROWS_PER_TG + sg;
    if (row >= p.rows) return;
    device const float4* wr = w + (ulong)row * p.cols4;
    float4 acc = 0.0f;
    uint i = lane;
    for (; i + 96 < p.cols4; i += 128) {
        acc[0] += dot(wr[i], x[i]);
        acc[1] += dot(wr[i + 32], x[i + 32]);
        acc[2] += dot(wr[i + 64], x[i + 64]);
        acc[3] += dot(wr[i + 96], x[i + 96]);
    }
    float s = acc[0] + acc[1] + acc[2] + acc[3];
    for (; i < p.cols4; i += 32) {
        s += dot(wr[i], x[i]);
    }
    float total = simd_sum(s);
    if (lane == 0) y[row] = p.has_bias ? total + bias[row] : total;
}

static inline float4 unpack_bf16_4(ushort4 u) {
    uint4 u32 = uint4(u) << 16;
    return as_type<float4>(u32);
}

kernel void gemv_bf16(device const ushort4* w [[buffer(0)]],
                      device const float4* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      device const float* bias [[buffer(3)]],
                      constant GemvParams& p [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * GEMV_ROWS_PER_TG + sg;
    if (row >= p.rows) return;
    device const ushort4* wr = w + (ulong)row * p.cols4;
    float4 acc = 0.0f;
    uint i = lane;
    for (; i + 96 < p.cols4; i += 128) {
        acc[0] += dot(unpack_bf16_4(wr[i]), x[i]);
        acc[1] += dot(unpack_bf16_4(wr[i + 32]), x[i + 32]);
        acc[2] += dot(unpack_bf16_4(wr[i + 64]), x[i + 64]);
        acc[3] += dot(unpack_bf16_4(wr[i + 96]), x[i + 96]);
    }
    float s = acc[0] + acc[1] + acc[2] + acc[3];
    for (; i < p.cols4; i += 32) {
        s += dot(unpack_bf16_4(wr[i]), x[i]);
    }
    float total = simd_sum(s);
    if (lane == 0) y[row] = p.has_bias ? total + bias[row] : total;
}

// In-place rotary embedding of q (offset 0) and k (offset k_offset) for every head.
// cs = [cos[0..pairs], sin[0..pairs]]; pair d is (base + stride*d, base + stride*d + gap).
kernel void rope_f32(device float* qkv [[buffer(0)]],
                     constant float* cs [[buffer(1)]],
                     constant RopeParams& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint per = p.n_heads * p.pairs;
    if (gid >= 2 * per) return;
    uint which = gid / per;
    uint r = gid % per;
    uint head = r / p.pairs;
    uint d = r % p.pairs;
    uint i0 = which * p.k_offset + head * p.head_dim + p.pair_stride * d;
    uint i1 = i0 + p.pair_gap;
    float c = cs[d];
    float s = cs[p.pairs + d];
    float a = qkv[i0];
    float b = qkv[i1];
    qkv[i0] = a * c - b * s;
    qkv[i1] = b * c + a * s;
}

kernel void rope_single(device float* x [[buffer(0)]],
                        constant float* cs [[buffer(1)]],
                        constant RopeParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    uint pair = gid % p.pairs;
    uint head = gid / p.pairs;
    if (head >= p.n_heads) return;
    float cos_th = cs[pair];
    float sin_th = cs[p.pairs + pair];
    device float* h = x + (ulong)head * p.head_dim;
    uint i0 = pair * p.pair_stride;
    uint i1 = i0 + p.pair_gap;
    float v0 = h[i0];
    float v1 = h[i1];
    h[i0] = v0 * cos_th - v1 * sin_th;
    h[i1] = v1 * cos_th + v0 * sin_th;
}

// Append this token's K and V rows (from qkv) at row `dst` of the layer's KV cache.
kernel void kv_append_f32(device const float* qkv [[buffer(0)]],
                          device float* kc [[buffer(1)]],
                          device float* vc [[buffer(2)]],
                          constant KvParams& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    kc[p.dst + i] = qkv[p.src_k + i];
    vc[p.dst + i] = qkv[p.src_v + i];
}

// One query token against `len` cached positions; one threadgroup per query head.
// Each simdgroup runs an online softmax over a strided subset of positions, then partials merge.
// KV head = head / group (GQA); rows of the cache are `stride` floats apart.
kernel void attn_decode_f32(device const float* q [[buffer(0)]],
                            device const float* kc [[buffer(1)]],
                            device const float* vc [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AttnParams& p [[buffer(4)]],
                            uint head [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * ATTN_MAX_PER_LANE * 32];
    const uint hd = p.head_dim;
    const uint row = ATTN_MAX_PER_LANE * 32;
    device const float* qh = q + head * hd;
    const uint kv_off = (head / p.group) * hd;

    float qr[ATTN_MAX_PER_LANE];
    float acc[ATTN_MAX_PER_LANE];
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        qr[j] = d < hd ? qh[d] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    uint lo = (p.window > 0 && p.len > p.window) ? (p.len - p.window) : 0;
    for (uint t = lo + sg; t < p.len; t += nsg) {
        device const float* k = kc + (ulong)t * p.stride + kv_off;
        device const float* v = vc + (ulong)t * p.stride + kv_off;
        float s = 0.0f;
        for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) s += qr[j] * k[d];
        }
        s = simd_sum(s) * p.scale;
        if (p.softcap > 0.0f) {
            s = tanh(s / p.softcap) * p.softcap;
        }
        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float e = exp(s - m_new);
        l = l * corr + e;
        for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) acc[j] = acc[j] * corr + e * v[d];
        }
        m = m_new;
    }
    if (lane == 0) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        if (d < hd) tg_acc[sg * row + d] = acc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -FLT_MAX;
    for (uint i = 0; i < nsg; ++i) gm = max(gm, tg_m[i]);
    float gl = 0.0f;
    for (uint i = 0; i < nsg; ++i) gl += tg_l[i] * exp(tg_m[i] - gm);
    for (uint d = sg * 32 + lane; d < hd; d += nsg * 32) {
        float o = 0.0f;
        for (uint i = 0; i < nsg; ++i) o += tg_acc[i * row + d] * exp(tg_m[i] - gm);
        out[head * hd + d] = o / gl;
    }
}

kernel void add_f32(device float* x [[buffer(0)]],
                    device const float* y [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] += y[i];
}

// out[i] = silu(gate[i]) * up[i] with gate_up = [gate; up], each n long.
kernel void swiglu_f32(device const float* gu [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = gu[i];
    out[i] = g / (1.0f + exp(-g)) * gu[n + i];
}

// In-place tanh-approximated GELU (matches `simd::vector_gelu`).
kernel void gelu_f32(device float* x [[buffer(0)]],
                     constant uint& n [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i];
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanh(inner));
}

kernel void geglu_f32(device const float* gate [[buffer(0)]],
                      device const float* up [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = gate[i];
    float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    float gelu = 0.5f * g * (1.0f + tanh(inner));
    out[i] = gelu * up[i];
}

kernel void scale_f32(device float* x [[buffer(0)]],
                      constant float& s [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] *= s;
}

kernel void softcap_f32(device float* x [[buffer(0)]],
                        constant float& cap [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = tanh(x[i] / cap) * cap;
}

kernel void embed_direct(device const float* emb [[buffer(0)]],
                         device float* x [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         constant uint& tok [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = emb[(ulong)tok * n + i];
}

kernel void rms_norm_heads(device float* x [[buffer(0)]],
                           device const float* w [[buffer(1)]],
                           constant uint& head_dim [[buffer(2)]],
                           constant float& eps [[buffer(3)]],
                           constant uint& has_weight [[buffer(4)]],
                           uint head [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    device float* h = x + (ulong)head * head_dim;
    float sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tpg) {
        float d = h[i];
        sq += d * d;
    }
    float scale = 1.0f / sqrt(block_sum(sq, scratch, sg, lane, nsg) / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tpg) {
        float v = h[i] * scale;
        h[i] = has_weight ? v * w[i] : v;
    }
}

kernel void kv_append_heads(device const float* k_src [[buffer(0)]],
                            device const float* v_src [[buffer(1)]],
                            device float* k_cache [[buffer(2)]],
                            device float* v_cache [[buffer(3)]],
                            constant uint& width [[buffer(4)]],
                            constant uint& pos [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < width) {
        ulong dst = (ulong)pos * width + i;
        k_cache[dst] = k_src[i];
        v_cache[dst] = v_src[i];
    }
}

static inline float gelu_erf(float x) {
    float z = fabs(x) * M_SQRT1_2_F;
    float t = 1.0f / (1.0f + 0.5f * z);
    float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
        + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    float erfc_v = t * exp(poly);
    float erf_v = x >= 0.0f ? 1.0f - erfc_v : erfc_v - 1.0f;
    return 0.5f * x * (1.0f + erf_v);
}
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

struct PackedGemvParams {
    uint rows;
    uint cols;
    uint batch_size;
    uint has_bias;
    uint epi; // 0 = None, 1 = Gelu, 2 = Residual
};

kernel void gemv_q4_0(device const block_q4_0* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    device const block_q4_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint b = lane; b < n_blocks; b += 32) {
        block_q4_0 blk0 = r0_blocks[b];
        block_q4_0 blk1 = has_row1 ? r1_blocks[b] : blk0;
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        device const float* xb = xb_base + b * 32;

        float b_acc0 = 0.0f;
        float b_acc1 = 0.0f;

        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            uint32_t q0 = *reinterpret_cast<thread const uint32_t*>(&blk0.qs[j]);
            uint32_t q1 = *reinterpret_cast<thread const uint32_t*>(&blk1.qs[j]);

            float4 v0_0 = float4(int4((q0 & 0x0F), ((q0 >> 8) & 0x0F), ((q0 >> 16) & 0x0F), ((q0 >> 24) & 0x0F)) - 8);
            float4 v1_0 = float4(int4(((q0 >> 4) & 0x0F), ((q0 >> 12) & 0x0F), ((q0 >> 20) & 0x0F), ((q0 >> 28) & 0x0F)) - 8);

            float4 v0_1 = float4(int4((q1 & 0x0F), ((q1 >> 8) & 0x0F), ((q1 >> 16) & 0x0F), ((q1 >> 24) & 0x0F)) - 8);
            float4 v1_1 = float4(int4(((q1 >> 4) & 0x0F), ((q1 >> 12) & 0x0F), ((q1 >> 20) & 0x0F), ((q1 >> 28) & 0x0F)) - 8);

            float4 x0 = *reinterpret_cast<device const float4*>(xb + j);
            float4 x1 = *reinterpret_cast<device const float4*>(xb + j + 16);

            b_acc0 += dot(v0_0, x0) + dot(v1_0, x1);
            b_acc1 += dot(v0_1, x0) + dot(v1_1, x1);
        }
        acc0 += d0 * b_acc0;
        if (has_row1) {
            acc1 += d1 * b_acc1;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        if (p.epi == 2) {
            acc0 += res[(ulong)batch * p.rows + row0];
            if (has_row1) acc1 += res[(ulong)batch * p.rows + row1];
        } else if (p.epi == 1) {
            acc0 = gelu_erf(acc0);
            if (has_row1) acc1 = gelu_erf(acc1);
        }
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_q8_0(device const block_q8_0* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg.x * 8 + sg;
    uint batch = tg.y;
    if (row >= p.rows || batch >= p.batch_size) return;
    uint n_blocks = p.cols / 32;
    device const block_q8_0* row_blocks = w + (ulong)row * n_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;
    float acc = 0.0f;
    for (uint b = lane; b < n_blocks; b += 32) {
        block_q8_0 blk = row_blocks[b];
        float d = float(blk.d);
        device const float* xb = xb_base + b * 32;
        float b_acc = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 32; j += 4) {
            uint32_t q_raw = *reinterpret_cast<thread const uint32_t*>(&blk.qs[j]);
            char4 q = as_type<char4>(q_raw);
            float4 v = float4(q);
            float4 x_vec = *reinterpret_cast<device const float4*>(xb + j);
            b_acc += dot(v, x_vec);
        }
        acc += d * b_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        if (p.has_bias != 0) acc += bias[row];
        if (p.epi == 2) {
            acc += res[(ulong)batch * p.rows + row];
        } else if (p.epi == 1) {
            acc = gelu_erf(acc);
        }
        y[(ulong)batch * p.rows + row] = acc;
    }
}
