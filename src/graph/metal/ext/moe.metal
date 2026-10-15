// Group `moe`: row top-k, column gather, and F16/BF16 embedding gathers (the expert GEMV is generated).

// Indices of the k largest values of each row, value descending, equal values lower index first, NaN
// below every number (`graph::cpu::topk_row` orders the same way). One threadgroup of 256 threads per
// row; round j picks the best element that sorts strictly after round j-1's pick, so no mask is needed.
struct MoeTopKParams { uint rows; uint cols; uint k; };

inline float moe_key(float v) { return isnan(v) ? -INFINITY : v; }

// (v1, i1) is the better candidate unless (v2, i2) is a real element that sorts before it.
inline void moe_pick(thread float& v1, thread uint& i1, float v2, uint i2) {
    if (i2 == 0xFFFFFFFFu) return;
    if (i1 == 0xFFFFFFFFu || v2 > v1 || (v2 == v1 && i2 < i1)) { v1 = v2; i1 = i2; }
}

// Rounds of the k-largest selection for one row, run by one 256-thread threadgroup (`sv`/`si`: 8
// slots, `prev`: 2 slots of threadgroup scratch).
inline void moe_topk_rounds(device const float* xr, uint cols, uint k, device uint* orow, uint t, uint sg,
                            uint lane, threadgroup float* sv, threadgroup uint* si,
                            threadgroup float* prev_v, threadgroup uint* prev_i) {
    if (t == 0) { *prev_v = INFINITY; *prev_i = 0xFFFFFFFFu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0; round < k; ++round) {
        const float pv = *prev_v;
        const uint pi = *prev_i;
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint i = t; i < cols; i += 256) {
            const float v = moe_key(xr[i]);
            // Eligible: sorts after the previous pick (round 0: pv = +inf, pi = none, everything).
            if (pi == 0xFFFFFFFFu || v < pv || (v == pv && i > pi)) moe_pick(bv, bi, v, i);
        }
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            uint oi = simd_shuffle_down(bi, off);
            moe_pick(bv, bi, ov, oi);
        }
        if (lane == 0) { sv[sg] = bv; si[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float fv = sv[0];
            uint fi = si[0];
            for (uint g = 1; g < 8; ++g) moe_pick(fv, fi, sv[g], si[g]);
            orow[round] = fi;
            *prev_v = fv;
            *prev_i = fi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_topk_f32(device const float* x [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant MoeTopKParams& p [[buffer(2)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup uint si[8];
    threadgroup float prev_v;
    threadgroup uint prev_i;
    moe_topk_rounds(x + (ulong)row * p.cols, p.cols, p.k, out + (ulong)row * p.k, t, sg, lane, sv, si, &prev_v, &prev_i);
}

// ---- Wide rows (k <= 64 of thousands of columns, e.g. a vocabulary): exact, three dispatches. ----
// A: each of 256 threadgroups per row writes the max key of its chunk. C: the k-th largest chunk max M
// is a lower bound of the row's k-th largest key (k distinct elements are >= M), so every element a
// full sort would keep has key >= M; those are appended to a bounded candidate list. D: one
// threadgroup ranks the candidates (key descending, lower index first) and writes the first k; a row
// with more than TK_CAP candidates (heavy ties) falls back to `moe_topk_rounds`.
// Scratch per row, in words: [0, 256) chunk max keys, [256] candidate count, [260, 260 + CAP) candidate
// keys, then CAP candidate indices.
constant uint TK_CHUNKS = 256;
constant uint TK_CAP = 1024;
constant uint TK_ROW_WORDS = 260 + 2 * TK_CAP;
struct MoeTopKWideParams { uint rows; uint cols; uint k; uint len; };

kernel void moe_topk_wide_max(device const float* x [[buffer(0)]],
                              device uint* scratch [[buffer(1)]],
                              constant MoeTopKWideParams& p [[buffer(2)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    const uint t = tid.x;
    const uint c = tg.x, row = tg.y;
    device const float* xr = x + (ulong)row * p.cols;
    const uint lo = min(c * p.len, p.cols), hi = min(lo + p.len, p.cols);
    float m = -INFINITY;
    for (uint i = lo + t; i < hi; i += 256) m = max(m, moe_key(xr[i]));
    m = simd_max(m);
    if (lane == 0) sv[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float r = sv[0];
        for (uint g = 1; g < 8; ++g) r = max(r, sv[g]);
        device uint* s = scratch + (ulong)row * TK_ROW_WORDS;
        s[c] = as_type<uint>(r);
        if (c == 0) atomic_store_explicit((device atomic_uint*)(s + TK_CHUNKS), 0u, memory_order_relaxed);
    }
}

kernel void moe_topk_wide_collect(device const float* x [[buffer(0)]],
                                  device uint* scratch [[buffer(1)]],
                                  constant MoeTopKWideParams& p [[buffer(2)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float cm[TK_CHUNKS];
    threadgroup float th;
    const uint t = tid.x;
    const uint c = tg.x, row = tg.y;
    device uint* s = scratch + (ulong)row * TK_ROW_WORDS;
    const float me = as_type<float>(s[t]);
    cm[t] = me;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint better = 0;
    for (uint j = 0; j < TK_CHUNKS; ++j) {
        const float o = cm[j];
        better += (o > me || (o == me && j < t)) ? 1u : 0u;
    }
    if (better == p.k - 1) th = me;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float M = th;
    device const float* xr = x + (ulong)row * p.cols;
    device atomic_uint* cnt = (device atomic_uint*)(s + TK_CHUNKS);
    device float* ck = (device float*)(s + 260);
    device uint* ci = s + 260 + TK_CAP;
    const uint lo = min(c * p.len, p.cols), hi = min(lo + p.len, p.cols);
    for (uint i = lo + t; i < hi; i += 256) {
        const float key = moe_key(xr[i]);
        if (key >= M) {
            const uint slot = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
            if (slot < TK_CAP) { ck[slot] = key; ci[slot] = i; }
        }
    }
}

kernel void moe_topk_wide_final(device const float* x [[buffer(0)]],
                                device uint* out [[buffer(1)]],
                                device uint* scratch [[buffer(2)]],
                                constant MoeTopKWideParams& p [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint t [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sk[TK_CAP];
    threadgroup uint sidx[TK_CAP];
    threadgroup float sv[8];
    threadgroup uint si[8];
    threadgroup float prev_v;
    threadgroup uint prev_i;
    device uint* s = scratch + (ulong)row * TK_ROW_WORDS;
    device uint* orow = out + (ulong)row * p.k;
    const uint n = atomic_load_explicit((device atomic_uint*)(s + TK_CHUNKS), memory_order_relaxed);
    if (n > TK_CAP) {
        moe_topk_rounds(x + (ulong)row * p.cols, p.cols, p.k, orow, t, sg, lane, sv, si, &prev_v, &prev_i);
        return;
    }
    for (uint i = t; i < n; i += 256) {
        sk[i] = as_type<float>(s[260 + i]);
        sidx[i] = s[260 + TK_CAP + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 256) {
        const float ki = sk[i];
        const uint ii = sidx[i];
        uint rank = 0;
        for (uint j = 0; j < n; ++j) rank += (sk[j] > ki || (sk[j] == ki && sidx[j] < ii)) ? 1u : 0u;
        if (rank < p.k) orow[rank] = ii;
    }
}

// out[r, j] = x[r, idx[r, j]] (one shared x row when `shared`); one thread per output element. An
// index past the row writes NaN.
struct MoeGatherParams { uint total; uint cols; uint k; uint shared; };

kernel void moe_gather_cols_f32(device const float* x [[buffer(0)]],
                                device const uint* idx [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant MoeGatherParams& p [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.total) return;
    uint c = idx[i];
    out[i] = c < p.cols ? x[(p.shared != 0 ? 0 : (ulong)(i / p.k) * p.cols) + c] : NAN;
}

// out[t, :] = table[ids[t], :]; grid (ceil(d / TG), n). ids are range-checked on the host.
struct MoeEmbedParams { uint n; uint d; };

kernel void moe_embed_f16(device const half* table [[buffer(0)]],
                          device const uint* ids [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant MoeEmbedParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.d || gid.y >= p.n) return;
    out[(ulong)gid.y * p.d + gid.x] = float(table[(ulong)ids[gid.y] * p.d + gid.x]);
}

kernel void moe_embed_bf16(device const ushort* table [[buffer(0)]],
                           device const uint* ids [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant MoeEmbedParams& p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.d || gid.y >= p.n) return;
    out[(ulong)gid.y * p.d + gid.x] = as_type<float>(uint(table[(ulong)ids[gid.y] * p.d + gid.x]) << 16);
}
