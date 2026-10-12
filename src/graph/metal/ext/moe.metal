// Group `moe`: top-k softmax for any k, MoE combine helpers, F16/BF16 embedding gathers.

// Softmax over the k largest logits of each row (others 0); k >= cols is a plain softmax.
// One thread per row; selection is k passes over the row (routers are narrow), `out` marks picks with -1.
struct MoeTopKParams { uint rows; uint cols; uint k; };

kernel void moe_topk_softmax_f32(device const float* x [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 constant MoeTopKParams& p [[buffer(2)]],
                                 uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device const float* xr = x + (ulong)row * p.cols;
    device float* yr = out + (ulong)row * p.cols;
    uint n = p.cols;
    uint k = min(p.k, n);
    if (k >= n) {
        float mx = xr[0];
        for (uint i = 1; i < n; ++i) mx = max(mx, xr[i]);
        float sum = 0.0f;
        for (uint i = 0; i < n; ++i) {
            float e = exp(xr[i] - mx);
            yr[i] = e;
            sum += e;
        }
        float inv = 1.0f / max(sum, 1e-12f);
        for (uint i = 0; i < n; ++i) yr[i] *= inv;
        return;
    }
    for (uint i = 0; i < n; ++i) yr[i] = 0.0f;
    float top = 0.0f;
    float sum = 0.0f;
    for (uint s = 0; s < k; ++s) {
        uint bi = n;
        float best = 0.0f;
        for (uint i = 0; i < n; ++i) {
            if (yr[i] == 0.0f && (bi == n || xr[i] > best)) { best = xr[i]; bi = i; }
        }
        if (s == 0) top = best;
        sum += exp(best - top);
        yr[bi] = -1.0f;
    }
    float inv = 1.0f / max(sum, 1e-12f);
    for (uint i = 0; i < n; ++i) yr[i] = yr[i] < 0.0f ? exp(xr[i] - top) * inv : 0.0f;
}

kernel void moe_zero_f32(device float* out [[buffer(0)]],
                         constant uint& n [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = 0.0f;
}

// Per expert, indirect dispatch arguments `[gemv xyz, accumulate xyz]` (6 u32): the full grids when any
// row routes to the expert, else zero threadgroups, so unselected experts read no weights.
struct MoeArgsParams { uint m; uint experts; uint actual; uint g0; uint g1; uint g2; uint a0; uint a1; uint a2; };

kernel void moe_args(device const float* probs [[buffer(0)]],
                     device uint* args [[buffer(1)]],
                     constant MoeArgsParams& p [[buffer(2)]],
                     uint e [[thread_position_in_grid]]) {
    if (e >= p.actual) return;
    bool any = false;
    for (uint r = 0; r < p.m && !any; ++r) any = probs[(ulong)r * p.experts + e] != 0.0f;
    device uint* a = args + e * 6;
    a[0] = any ? p.g0 : 0u; a[1] = any ? p.g1 : 0u; a[2] = any ? p.g2 : 0u;
    a[3] = any ? p.a0 : 0u; a[4] = any ? p.a1 : 0u; a[5] = any ? p.a2 : 0u;
}

// out[r, :] += probs[r, e] * tmp[r, :]; grid (ceil(d / TG), m).
struct MoeAccumParams { uint m; uint d; uint experts; uint e; };

kernel void moe_accum_f32(device float* out [[buffer(0)]],
                          device const float* tmp [[buffer(1)]],
                          device const float* probs [[buffer(2)]],
                          constant MoeAccumParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.d || gid.y >= p.m) return;
    ulong i = (ulong)gid.y * p.d + gid.x;
    out[i] += probs[(ulong)gid.y * p.experts + p.e] * tmp[i];
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
