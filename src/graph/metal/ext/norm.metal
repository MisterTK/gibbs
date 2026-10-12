// Kernels for the `norm` lowering group (see norm.rs). Shares the preamble of metal_kernels.metal.

// Max across the whole threadgroup; every thread gets the result. Safe to call repeatedly.
static inline float block_max_nb(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = scratch[0];
    for (uint i = 1; i < nsg; ++i) m = max(m, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

// Reduce over a strided subset of axes. One simdgroup (32 threads) per output element. The host
// coalesces adjacent same-kind axes into at most 4 kept and 4 reduced runs (dim, stride).
// kind: 0 sum, 1 mean, 2 max, 3 min. Output index = row-major over the kept runs.
struct NormReduceParams {
    uint kind; uint rcount; uint nk; uint nr;
    uint kdim[4]; uint kstr[4]; uint rdim[4]; uint rstr[4];
};
kernel void norm_reduce_f32(device const float* x [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant NormReduceParams& p [[buffer(2)]],
                            uint o [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    ulong base = 0;
    uint rem = o;
    for (int i = int(p.nk) - 1; i >= 0; --i) {
        base += (ulong)(rem % p.kdim[i]) * p.kstr[i];
        rem /= p.kdim[i];
    }
    float acc = p.kind == 2 ? -INFINITY : (p.kind == 3 ? INFINITY : 0.0f);
    for (uint r = lane; r < p.rcount; r += 32) {
        ulong off = base;
        if (p.nr == 1) {
            off += (ulong)r * p.rstr[0];
        } else {
            uint rr = r;
            for (int i = int(p.nr) - 1; i >= 0; --i) {
                off += (ulong)(rr % p.rdim[i]) * p.rstr[i];
                rr /= p.rdim[i];
            }
        }
        float v = x[off];
        acc = p.kind < 2 ? acc + v : (p.kind == 2 ? max(acc, v) : min(acc, v));
    }
    acc = p.kind < 2 ? simd_sum(acc) : (p.kind == 2 ? simd_max(acc) : simd_min(acc));
    if (lane == 0) out[o] = p.kind == 1 ? acc / float(p.rcount) : acc;
}

// Softmax along one axis of an [outer, n, inner] view; one threadgroup per (outer, inner) pair.
struct NormSoftmaxParams { uint n; uint inner; };
kernel void norm_softmax_f32(device const float* x [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant NormSoftmaxParams& p [[buffer(2)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint nsg [[simdgroups_per_threadgroup]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    ulong base = (ulong)(tg / p.inner) * p.n * p.inner + (tg % p.inner);
    const uint st = p.inner;
    float m = -INFINITY;
    for (uint k = tid; k < p.n; k += tpg) m = max(m, x[base + (ulong)k * st]);
    m = block_max_nb(m, scratch, sg, lane, nsg);
    float s = 0.0f;
    for (uint k = tid; k < p.n; k += tpg) {
        float e = exp(x[base + (ulong)k * st] - m);
        out[base + (ulong)k * st] = e;
        s += e;
    }
    float inv = 1.0f / max(block_sum(s, scratch, sg, lane, nsg), 1e-12f);
    for (uint k = tid; k < p.n; k += tpg) out[base + (ulong)k * st] *= inv;
}

// LayerNorm over the last dim with gamma and no beta (the beta form is `layer_norm_f32`).
kernel void norm_layernorm_nobeta_f32(device const float* x [[buffer(0)]],
                                      device const float* g [[buffer(1)]],
                                      device float* out [[buffer(2)]],
                                      constant NormParams& p [[buffer(3)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint nsg [[simdgroups_per_threadgroup]],
                                      uint tid [[thread_position_in_threadgroup]],
                                      uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)row * p.d;
    device float* o = out + (ulong)row * p.d;
    float s = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) s += xr[i];
    float mean = block_sum(s, scratch, sg, lane, nsg) / float(p.d);
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float t = xr[i] - mean;
        sq += t * t;
    }
    float inv = 1.0f / sqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) o[i] = (xr[i] - mean) * inv * g[i];
}

// GroupNorm over x [n, c, hw] (InstanceNorm: groups == c); biased variance per (sample, group),
// per-channel affine. One threadgroup per (sample, group).
struct NormGroupParams { uint groups; uint cg; uint hw; float eps; };
kernel void norm_group_f32(device const float* x [[buffer(0)]],
                           device const float* g [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant NormGroupParams& p [[buffer(4)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    const uint grp = tg % p.groups;
    const uint cnt = p.cg * p.hw;
    // A group's channels are contiguous: [sample, grp * cg .. (grp + 1) * cg, hw].
    ulong base = (ulong)tg * cnt;
    float s = 0.0f;
    for (uint e = tid; e < cnt; e += tpg) s += x[base + e];
    float mean = block_sum(s, scratch, sg, lane, nsg) / float(cnt);
    float sq = 0.0f;
    for (uint e = tid; e < cnt; e += tpg) {
        float t = x[base + e] - mean;
        sq += t * t;
    }
    float inv = 1.0f / sqrt(block_sum(sq, scratch, sg, lane, nsg) / float(cnt) + p.eps);
    for (uint ci = 0; ci < p.cg; ++ci) {
        float ga = g[grp * p.cg + ci] * inv;
        float be = b[grp * p.cg + ci];
        ulong cb = base + (ulong)ci * p.hw;
        for (uint j = tid; j < p.hw; j += tpg) out[cb + j] = (x[cb + j] - mean) * ga + be;
    }
}
