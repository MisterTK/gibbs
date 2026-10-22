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

// TopK k = 1 over one wide row, in two passes: `moe_argmax_part_f32` gives threadgroup g the span
// [g * span, (g + 1) * span) and writes its best (value, index); `moe_argmax_reduce_f32` reduces
// the partials in one threadgroup. Same order as `moe_topk_f32`. 256 threads per threadgroup.
// Threadgroup best of every thread's (bv, bi), valid in simdgroup 0: simd_max of the values, then
// the lowest index holding it (keys are never NaN, empty lanes hold (-inf, none)).
inline void moe_argmax_tg(thread float& bv, thread uint& bi, threadgroup float* sv,
                          threadgroup uint* si, uint sg, uint lane) {
    float m = simd_max(bv);
    uint i = simd_min(bv == m ? bi : 0xFFFFFFFFu);
    if (lane == 0) { sv[sg] = m; si[sg] = i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float v = lane < 8 ? sv[lane] : -INFINITY;
        const uint j = lane < 8 ? si[lane] : 0xFFFFFFFFu;
        bv = simd_max(v);
        bi = simd_min(v == bv ? j : 0xFFFFFFFFu);
    }
}

kernel void moe_argmax_part_f32(device const float* x [[buffer(0)]],
                                device float* part_v [[buffer(1)]],
                                device uint* part_i [[buffer(2)]],
                                constant uint2& p [[buffer(3)]],  // cols, span
                                uint g [[threadgroup_position_in_grid]],
                                uint t [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup uint si[8];
    const uint hi = min((g + 1) * p.y, p.x);
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint i = g * p.y + t; i < hi; i += 256) moe_pick(bv, bi, moe_key(x[i]), i);
    moe_argmax_tg(bv, bi, sv, si, sg, lane);
    if (t == 0) { part_v[g] = bv; part_i[g] = bi; }
}

kernel void moe_argmax_reduce_f32(device const float* part_v [[buffer(0)]],
                                  device const uint* part_i [[buffer(1)]],
                                  device uint* out [[buffer(2)]],
                                  constant uint& parts [[buffer(3)]],
                                  uint t [[thread_position_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup uint si[8];
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint j = t; j < parts; j += 256) moe_pick(bv, bi, part_v[j], part_i[j]);
    moe_argmax_tg(bv, bi, sv, si, sg, lane);
    if (t == 0) out[0] = bi;
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

// out[t, :] = table[ids[t], :]; grid (ceil(d / TG), n). Host ids are range-checked, device-fed ids clamp.
struct MoeEmbedParams { uint n; uint d; uint rows; };

kernel void moe_embed_f16(device const half* table [[buffer(0)]],
                          device const uint* ids [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant MoeEmbedParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.d || gid.y >= p.n) return;
    out[(ulong)gid.y * p.d + gid.x] = float(table[(ulong)min(ids[gid.y], p.rows - 1) * p.d + gid.x]);
}

kernel void moe_embed_bf16(device const ushort* table [[buffer(0)]],
                           device const uint* ids [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant MoeEmbedParams& p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.d || gid.y >= p.n) return;
    out[(ulong)gid.y * p.d + gid.x] = as_type<float>(uint(table[(ulong)min(ids[gid.y], p.rows - 1) * p.d + gid.x]) << 16);
}

// Sparse block key selection (`Op::QsaSelect` / `Op::QsaBias`; `graph::cpu::qsa_select` is the
// reference). Row t sees v = pos[t] + 1 keys in nb = v / ratio complete blocks; with nb <= top every
// visible key is selected, else the `top` best blocks (f32 total order descending, equal scores lower
// block first), ascending, each expanded to its `ratio` keys, then the tail nb * ratio .. v. Unused
// slots hold id 0 and bias -inf. nb is clamped to the score width b (the CPU rejects nb > b).
struct MoeQsaParams { uint n; uint b; uint ratio; uint top; uint k; };

// Rust `f32::total_cmp` order as a signed integer.
inline int moe_total_key(float v) {
    const int i = as_type<int>(v);
    return i ^ int(uint(i >> 31) >> 1);
}

inline void moe_qsa_pick(thread int& v1, thread uint& i1, int v2, uint i2) {
    if (i2 == 0xFFFFFFFFu) return;
    if (i1 == 0xFFFFFFFFu || v2 > v1 || (v2 == v1 && i2 < i1)) { v1 = v2; i1 = i2; }
}

// One threadgroup of 256 threads per row: round r picks the best block that sorts strictly after
// round r-1's pick (as `moe_topk_f32`); thread 0 then sorts the picks and expands them in place.
kernel void moe_qsa_select_f32(device const float* score [[buffer(0)]],
                               device const uint* pos [[buffer(1)]],
                               device uint* out [[buffer(2)]],
                               constant MoeQsaParams& p [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int sv[8];
    threadgroup uint si[8];
    threadgroup int prev_v;
    threadgroup uint prev_i;
    device uint* o = out + (ulong)row * p.k;
    const uint v = pos[row] + 1;
    const uint nb = min(v / p.ratio, p.b);
    if (nb <= p.top) { // uniform per threadgroup
        for (uint j = t; j < p.k; j += 256) o[j] = j < v ? j : 0u;
        return;
    }
    device const float* sr = score + (ulong)row * p.b;
    if (t == 0) { prev_v = 0; prev_i = 0xFFFFFFFFu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < p.top; ++r) {
        const int pv = prev_v;
        const uint pi = prev_i;
        int bv = 0;
        uint bi = 0xFFFFFFFFu;
        for (uint i = t; i < nb; i += 256) {
            const int kv = moe_total_key(sr[i]);
            if (pi == 0xFFFFFFFFu || kv < pv || (kv == pv && i > pi)) moe_qsa_pick(bv, bi, kv, i);
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const int ov = simd_shuffle_down(bv, off);
            const uint oi = simd_shuffle_down(bi, off);
            moe_qsa_pick(bv, bi, ov, oi);
        }
        if (lane == 0) { sv[sg] = bv; si[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            int fv = sv[0];
            uint fi = si[0];
            for (uint g = 1; g < 8; ++g) moe_qsa_pick(fv, fi, sv[g], si[g]);
            o[r] = fi;
            prev_v = fv;
            prev_i = fi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t != 0) return;
    for (uint a = 1; a < p.top; ++a) {
        const uint x = o[a];
        uint c = a;
        for (; c > 0 && o[c - 1] > x; --c) o[c] = o[c - 1];
        o[c] = x;
    }
    // Last block first: block j's keys land at j * ratio >= j, past every unexpanded pick.
    for (uint j = p.top; j-- > 0;) {
        const uint blk = o[j];
        for (uint q = p.ratio; q-- > 0;) o[j * p.ratio + q] = blk * p.ratio + q;
    }
    uint c = p.top * p.ratio;
    for (uint key = nb * p.ratio; key < v && c < p.k; ++key) o[c++] = key;
    for (; c < p.k; ++c) o[c] = 0u;
}

kernel void moe_qsa_bias_f32(device const uint* pos [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant MoeQsaParams& p [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.k) return;
    const uint row = i / p.k;
    const uint v = pos[row] + 1;
    const uint nb = min(v / p.ratio, p.b);
    const uint cnt = nb <= p.top ? v : p.top * p.ratio + (v - nb * p.ratio);
    out[i] = i % p.k < cnt ? 0.0f : -INFINITY;
}

// Groups `ids[slots]` (expert per slot) into per-expert counts and `slot_list[experts][slots]`.
// Slots whose expert id is >= `experts` have their output row zeroed so `gemmgen_id_<fmt>` can skip them.
struct MoeMapParams { uint slots; uint experts; uint rows; };

kernel void moe_map_rows(device const uint* ids [[buffer(0)]],
                         device atomic_uint* counts [[buffer(1)]],
                         device uint* slot_list [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant MoeMapParams& p [[buffer(4)]],
                         uint tid [[thread_position_in_threadgroup]]) {
    for (uint e = tid; e < p.experts; e += 256) {
        atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint s = tid; s < p.slots; s += 256) {
        const uint e = ids[s];
        if (e < p.experts) {
            const uint idx = atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
            slot_list[(ulong)e * p.slots + idx] = s;
        } else {
            device float* y = out + (ulong)s * p.rows;
            for (uint r = 0; r < p.rows; ++r) y[r] = 0.0f;
        }
    }
}
