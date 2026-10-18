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
    device const float* xr = x + (ulong)row * p.cols;
    if (t == 0) { prev_v = INFINITY; prev_i = 0xFFFFFFFFu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0; round < p.k; ++round) {
        const float pv = prev_v;
        const uint pi = prev_i;
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint i = t; i < p.cols; i += 256) {
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
            out[(ulong)row * p.k + round] = fi;
            prev_v = fv;
            prev_i = fi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_topk_softmax_f32(device const float* x [[buffer(0)]],
                                 device uint* out_ids [[buffer(1)]],
                                 device float* out_weights [[buffer(2)]],
                                 device const float* scale [[buffer(3)]],
                                 constant uint4& p [[buffer(4)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_position_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup uint si[8];
    threadgroup float prev_v;
    threadgroup uint prev_i;
    float picked_v[32];
    uint picked_i[32];
    device const float* xr = x + (ulong)row * p.y;
    if (t == 0) { prev_v = INFINITY; prev_i = 0xFFFFFFFFu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0; round < p.z; ++round) {
        const float pv = prev_v;
        const uint pi = prev_i;
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint i = t; i < p.y; i += 256) {
            const float v = moe_key(xr[i]);
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
            out_ids[(ulong)row * p.z + round] = fi;
            picked_v[round] = fv;
            picked_i[round] = fi;
            prev_v = fv;
            prev_i = fi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0 && p.z > 0) {
        float max_v = picked_v[0];
        for (uint r = 1; r < p.z; ++r) max_v = max(max_v, picked_v[r]);
        float sum_exp = 0.0f;
        for (uint r = 0; r < p.z; ++r) {
            picked_v[r] = exp(picked_v[r] - max_v);
            sum_exp += picked_v[r];
        }
        for (uint r = 0; r < p.z; ++r) {
            float w = picked_v[r] / sum_exp;
            if (p.w != 0u) {
                w *= (picked_i[r] < p.y ? scale[picked_i[r]] : NAN);
            }
            out_weights[(ulong)row * p.z + r] = w;
        }
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
