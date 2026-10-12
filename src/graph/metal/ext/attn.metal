// Kernels for the `attn` lowering group (see attn.rs). Shares the preamble of metal_kernels.metal.

// ---- Flash-style attention (`Op::Attention`) ---------------------------------------------------
// One threadgroup per (head, block of ATTNF_NSG query rows), one simdgroup per query row. The block's
// K/V rows are staged through threadgroup memory in tiles shared by its simdgroups; every simdgroup
// runs an online softmax over its own visible key range. Lane `l` owns head dims l, l+32, ...
// (`NJ` per lane, so head_dim <= 32 * NJ). Handles grouped kv heads, causal / windowed masks, soft
// cap and an additive bias; a query with no visible key (or only -inf bias) writes zeros.
// ponytail: scalar-FMA dot products, ~0.3 TFLOP/s measured (4096x4096, 24 heads, hd 128, M-series);
// a query-block simdgroup_matrix tile should be several times faster for long DiT/ViT sequences. A
// short query block (nq < ATTNF_NSG) leaves simdgroups idle and does not split keys across them.
constant uint ATTNF_NSG = 8;     // simdgroups (= query rows) per threadgroup; attn.rs `QB`
constant uint ATTNF_TILE = 2048; // floats per staged K or V tile (K + V = 16 KiB)

struct AttnFlashParams {
    uint nq;
    uint nk;
    uint heads;
    uint kv_heads;
    uint head_dim;
    uint flags;     // bit 0 causal, bit 1 window, bit 2 soft cap
    uint window;
    uint bias_mode; // 0 none, 1 [nk], 2 [nq, nk], 3 [heads, nq, nk]
    float scale;
    float soft_cap;
};

// Keys [lo, hi) visible to the query at `pos` (mirrors `cpu::attention_key_range`). Both bounds
// are non-decreasing in `pos`, which the threadgroup-wide tile range below relies on.
static inline void attn_key_range(uint pos, constant AttnFlashParams& p, thread uint& lo, thread uint& hi) {
    const bool causal = (p.flags & 1u) != 0;
    const bool win = (p.flags & 2u) != 0;
    const uint half_w = p.window / 2;
    if (causal) {
        hi = min(p.nk, pos + 1);
        lo = (win && pos + 1 > p.window) ? pos + 1 - p.window : 0;
    } else {
        hi = win ? min(p.nk, pos + half_w + 1) : p.nk;
        lo = (win && pos > half_w) ? pos - half_w : 0;
    }
}

template <uint NJ>
static inline void attn_flash(device const float* q, device const float* k, device const float* v,
                              device const float* bias, device float* out,
                              constant AttnFlashParams& p,
                              threadgroup float* tk, threadgroup float* tv,
                              uint2 tgp, uint sg, uint lane) {
    const uint hd = p.head_dim;
    const uint head = tgp.y;
    const uint kvh = head / (p.heads / p.kv_heads);
    const uint q0 = tgp.x * ATTNF_NSG;
    const uint qi = q0 + sg;
    const bool active = qi < p.nq;
    const ulong qstride = (ulong)p.heads * hd;
    const ulong kvstride = (ulong)p.kv_heads * hd;
    const bool has_cap = (p.flags & 4u) != 0;

    // Threadgroup-wide key range: the first row has the smallest `lo`, the last row the largest `hi`.
    uint tg_lo = 0, tg_hi = 0, lo = 0, hi = 0, unused = 0;
    attn_key_range(q0, p, tg_lo, unused);
    attn_key_range(min(q0 + ATTNF_NSG, p.nq) - 1, p, unused, tg_hi);
    if (active) attn_key_range(qi, p, lo, hi);

    float qr[NJ];
    float acc[NJ];
    for (uint j = 0; j < NJ; ++j) {
        const uint d = lane + 32 * j;
        qr[j] = (active && d < hd) ? q[(ulong)qi * qstride + (ulong)head * hd + d] : 0.0f;
        acc[j] = 0.0f;
    }
    ulong brow = 0;
    if (p.bias_mode == 2) brow = (ulong)qi * p.nk;
    else if (p.bias_mode == 3) brow = ((ulong)head * p.nq + qi) * p.nk;

    float m = -FLT_MAX;
    float l = 0.0f;
    const uint kt = ATTNF_TILE / hd;
    for (uint t0 = tg_lo; t0 < tg_hi; t0 += kt) {
        const uint n = min(kt, tg_hi - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup); // the previous tile is consumed
        for (uint r = sg; r < n; r += ATTNF_NSG) {
            device const float* kr = k + (ulong)(t0 + r) * kvstride + (ulong)kvh * hd;
            device const float* vr = v + (ulong)(t0 + r) * kvstride + (ulong)kvh * hd;
            for (uint d = lane; d < hd; d += 32) {
                tk[r * hd + d] = kr[d];
                tv[r * hd + d] = vr[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        const uint a = max(t0, lo);
        const uint b = min(t0 + n, hi);
        // Four keys per step: independent dot/simd_sum chains, one softmax rescale per step.
        for (uint t = a; t < b; t += 4) {
            float s[4];
            for (uint u = 0; u < 4; ++u) {
                float x = 0.0f;
                if (t + u < b) {
                    threadgroup const float* kr = tk + (t + u - t0) * hd;
                    for (uint j = 0; j < NJ; ++j) {
                        const uint d = lane + 32 * j;
                        if (d < hd) x += qr[j] * kr[d];
                    }
                }
                s[u] = x;
            }
            float m_new = m;
            for (uint u = 0; u < 4; ++u) {
                float x = simd_sum(s[u]) * p.scale;
                if (t + u < b) {
                    if (has_cap) x = p.soft_cap * tanh(x / p.soft_cap);
                    if (p.bias_mode != 0) x += bias[brow + t + u];
                } else {
                    x = -INFINITY;
                }
                s[u] = x;
                m_new = max(m_new, x);
            }
            const float corr = exp(m - m_new);
            l *= corr;
            for (uint j = 0; j < NJ; ++j) acc[j] *= corr;
            for (uint u = 0; u < 4; ++u) {
                const float e = exp(s[u] - m_new); // 0 for a -inf score
                l += e;
                if (t + u < b) {
                    threadgroup const float* vr = tv + (t + u - t0) * hd;
                    for (uint j = 0; j < NJ; ++j) {
                        const uint d = lane + 32 * j;
                        if (d < hd) acc[j] += e * vr[d];
                    }
                }
            }
            m = m_new;
        }
    }
    if (active) {
        device float* o = out + (ulong)qi * qstride + (ulong)head * hd;
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            if (d < hd) o[d] = l > 0.0f ? acc[j] / l : 0.0f;
        }
    }
}

#define ATTNF_KERNEL(NJ)                                                                         \
kernel void attn_flash_nj##NJ(device const float* q [[buffer(0)]],                               \
                              device const float* k [[buffer(1)]],                               \
                              device const float* v [[buffer(2)]],                               \
                              device const float* bias [[buffer(3)]],                            \
                              device float* out [[buffer(4)]],                                   \
                              constant AttnFlashParams& p [[buffer(5)]],                         \
                              uint2 tgp [[threadgroup_position_in_grid]],                        \
                              uint sg [[simdgroup_index_in_threadgroup]],                        \
                              uint lane [[thread_index_in_simdgroup]]) {                         \
    threadgroup float tk[ATTNF_TILE];                                                            \
    threadgroup float tv[ATTNF_TILE];                                                            \
    attn_flash<NJ>(q, k, v, bias, out, p, tk, tv, tgp, sg, lane);                               \
}

ATTNF_KERNEL(2)
ATTNF_KERNEL(3)
ATTNF_KERNEL(4)
ATTNF_KERNEL(8)
ATTNF_KERNEL(16)

// ---- RelPosBias ---------------------------------------------------------------------------------
// out[h, i, j] = table[bucket(k_pos[j] - q_pos[i]), h]; HF `_relative_position_bucket`, the same f32
// arithmetic as `cpu::rel_pos_bucket` (`scale` = ln(max_distance / exact) rounded from f64).
struct RelPosParams {
    uint n;
    uint m;
    uint heads;
    uint buckets;
    uint bidir;
    float scale;
};

static inline uint rel_pos_bucket(long rel, constant RelPosParams& p) {
    uint nb = p.buckets;
    uint ret = 0;
    uint n;
    if (p.bidir != 0) {
        nb /= 2;
        if (rel > 0) ret += nb;
        n = uint(rel < 0 ? -rel : rel);
    } else {
        n = rel < 0 ? uint(-rel) : 0u;
    }
    const uint exact = nb / 2;
    if (n < exact) return ret + n;
    const uint large = exact + uint(log(float(n) / float(exact)) / p.scale * float(nb - exact));
    return ret + min(large, nb - 1);
}

// One thread per (query, key); writes that position for every head.
kernel void rel_pos_bias_f32(device const float* table [[buffer(0)]],
                             device const uint* q_pos [[buffer(1)]],
                             device const uint* k_pos [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant RelPosParams& p [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    const ulong nm = (ulong)p.n * p.m;
    if (i >= nm) return;
    const uint qi = i / p.m;
    const uint ki = i - qi * p.m;
    const uint b = rel_pos_bucket(long(k_pos[ki]) - long(q_pos[qi]), p);
    for (uint h = 0; h < p.heads; ++h) out[(ulong)h * nm + i] = table[b * p.heads + h];
}

// ---- StateReorder -------------------------------------------------------------------------------
// out row b = state row parent[b], or zeros when parent[b] >= batch (as the CPU executor).
struct ReorderParams {
    uint batch;
    uint row;
};

kernel void state_reorder_f32(device const float* state [[buffer(0)]],
                              device const uint* parent [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant ReorderParams& p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= (ulong)p.batch * p.row) return;
    const uint b = i / p.row;
    const uint c = i - b * p.row;
    const uint src = parent[b];
    out[i] = src < p.batch ? state[(ulong)src * p.row + c] : 0.0f;
}
