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

// Reduce over a strided subset of axes. One thread per output element when `rcount <= 32`
// (coalesced across kept inner axes), else one simdgroup (32 threads) per output element.
// The host coalesces adjacent same-kind axes into at most 4 kept and 4 reduced runs (dim, stride).
// kind: 0 sum, 1 mean, 2 max, 3 min. Output index = row-major over the kept runs.
struct NormReduceParams {
    uint kind; uint rcount; uint nk; uint nr; uint n_out;
    uint kdim[4]; uint kstr[4]; uint rdim[4]; uint rstr[4];
};
kernel void norm_reduce_f32(device const float* x [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant NormReduceParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const bool scalar = p.rcount <= 32;
    const uint o = scalar ? gid : tg;
    if (o >= p.n_out) return;
    ulong base = 0;
    uint rem = o;
    for (int i = int(p.nk) - 1; i >= 0; --i) {
        base += (ulong)(rem % p.kdim[i]) * p.kstr[i];
        rem /= p.kdim[i];
    }
    float acc = p.kind == 2 ? -INFINITY : (p.kind == 3 ? INFINITY : 0.0f);
    for (uint r = scalar ? 0u : lane; r < p.rcount; r += scalar ? 1u : 32u) {
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
    if (!scalar) {
        acc = p.kind < 2 ? simd_sum(acc) : (p.kind == 2 ? simd_max(acc) : simd_min(acc));
        if (lane != 0) return;
    }
    out[o] = p.kind == 1 ? acc / float(p.rcount) : acc;
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
// per-channel affine (+ optional fused SiLU). Two dispatches: `norm_group_stats_f32` writes
// per-slice (sum, sum_sq) partials with `ceil(cg * hw / 4096)` threadgroups per (sample, group),
// then `norm_group_apply_f32` reduces those partials per group and writes 4 elements per thread.
struct NormGroupParams { uint n_groups; uint groups; uint cg; uint hw; uint splits; uint silu; float eps; };
kernel void norm_group_stats_f32(device const float* x [[buffer(0)]],
                                 device float2* stats [[buffer(1)]],
                                 constant NormGroupParams& p [[buffer(2)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tpg = 256;
    threadgroup float scratch[32];
    const uint cnt = p.cg * p.hw;
    const ulong base = (ulong)tg.y * cnt;
    const uint start = tg.x * 4096u;
    const uint end = min(cnt, start + 4096u);
    const float pivot = cnt > 0 ? x[base] : 0.0f;
    float s = 0.0f;
    float sq = 0.0f;
    for (uint e = start + tid; e < end; e += tpg) {
        float v = x[base + e] - pivot;
        s += v;
        sq += v * v;
    }
    s = block_sum(s, scratch, sg, lane, nsg);
    sq = block_sum(sq, scratch, sg, lane, nsg);
    if (tid == 0) stats[(ulong)tg.y * p.splits + tg.x] = float2(s, sq);
}

kernel void norm_group_apply_f32(device const float* x [[buffer(0)]],
                                 device const float* g [[buffer(1)]],
                                 device const float* b [[buffer(2)]],
                                 device const float2* stats [[buffer(3)]],
                                 device float* out [[buffer(4)]],
                                 constant NormGroupParams& p [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    const uint cnt = p.cg * p.hw;
    const ulong total = (ulong)p.n_groups * cnt;
    const ulong idx0 = (ulong)gid * 4;
    if (idx0 >= total) return;
    uint prev_sg = UINT_MAX;
    float mean = 0.0f, inv = 0.0f;
    #pragma unroll
    for (uint u = 0; u < 4; ++u) {
        const ulong idx = idx0 + u;
        if (idx >= total) break;
        const uint sg_idx = uint(idx / cnt);
        if (sg_idx != prev_sg) {
            prev_sg = sg_idx;
            device const float2* st = stats + (ulong)sg_idx * p.splits;
            float s = 0.0f, sq = 0.0f;
            for (uint k = 0; k < p.splits; ++k) {
                float2 pair = st[k];
                s += pair.x;
                sq += pair.y;
            }
            const float pivot = cnt > 0 ? x[(ulong)sg_idx * cnt] : 0.0f;
            float d_mean = s / float(cnt);
            mean = pivot + d_mean;
            float var = max(sq / float(cnt) - d_mean * d_mean, 0.0f);
            inv = rsqrt(var + p.eps);
        }
        const uint rem = uint(idx - (ulong)sg_idx * cnt);
        const uint ci = rem / p.hw;
        const uint ch = (sg_idx % p.groups) * p.cg + ci;
        float v = (x[idx] - mean) * (g[ch] * inv) + b[ch];
        if (p.silu != 0) v = v / (1.0f + exp(-v));
        out[idx] = v;
    }
}

// RmsNorm over the last dim without gamma (the gamma form is `rmsnorm_f32`).
kernel void norm_rmsnorm_nogamma_f32(device const float* x [[buffer(0)]],
                                     device float* out [[buffer(1)]],
                                     constant NormParams& p [[buffer(2)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint nsg [[simdgroups_per_threadgroup]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)row * p.d;
    device float* o = out + (ulong)row * p.d;
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) sq += xr[i] * xr[i];
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) o[i] = xr[i] * scale;
}
