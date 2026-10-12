// Kernels for the `movement` lowering group (see movement.rs). Shares the preamble of metal_kernels.metal.
// All kernels move 4-byte elements; one thread per output element unless noted.

constant uint MV_TG = 256;

// out[obase + sum c_d*ostride_d] = x[ibase + sum ((c_d / idiv_d) % imod_d) * istride_d], c over odim (row-major).
// Covers Transpose (permuted istride), Slice (ibase), Concat part / Pad interior (obase, ostride),
// Tile (imod), Upsample-nearest-style repeats (idiv).
struct MvMap {
    uint n;
    uint rank;
    uint odim[6];
    uint ostride[6];
    uint istride[6];
    uint idiv[6];
    uint imod[6];
    uint ibase;
    uint obase;
};

kernel void mv_strided(device const float* x [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant MvMap& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint rem = gid;
    ulong io = p.ibase;
    ulong oo = p.obase;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        uint c = rem % p.odim[d];
        rem /= p.odim[d];
        oo += (ulong)c * p.ostride[d];
        io += (ulong)((c / p.idiv[d]) % p.imod[d]) * p.istride[d];
    }
    out[oo] = x[io];
}

kernel void mv_fill(device float* out [[buffer(0)]],
                    constant float& value [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = value;
}

// out[o, j, i] = x[o, idx[j], i] over x [outer, d, inner]; an out-of-range index yields 0.
struct MvGatherParams { uint n; uint m; uint d; uint inner; };

kernel void mv_gather(device const float* x [[buffer(0)]],
                      device const uint* idx [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant MvGatherParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint i = gid % p.inner;
    uint j = (gid / p.inner) % p.m;
    uint o = gid / (p.inner * p.m);
    uint r = idx[j];
    out[gid] = r < p.d ? x[((ulong)o * p.d + r) * p.inner + i] : 0.0f;
}

// One threadgroup per (o, j) update row: out[o, idx[j], :] = upd[o, j, :] (out already holds the base).
// With duplicate indices the last j wins, like the sequential reference; out-of-range rows are skipped.
struct MvScatterParams { uint m; uint d; uint inner; };

kernel void mv_scatter(device const uint* idx [[buffer(0)]],
                       device const float* upd [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant MvScatterParams& p [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint t [[thread_position_in_threadgroup]]) {
    uint o = row / p.m;
    uint j = row % p.m;
    uint target = idx[j];
    if (target >= p.d) return;
    threadgroup atomic_uint dup;
    if (t == 0) atomic_store_explicit(&dup, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = j + 1 + t; k < p.m; k += MV_TG) {
        if (idx[k] == target) atomic_store_explicit(&dup, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&dup, memory_order_relaxed) != 0u) return;
    ulong src = (ulong)row * p.inner;
    ulong dst = ((ulong)o * p.d + target) * p.inner;
    for (uint i = t; i < p.inner; i += MV_TG) out[dst + i] = upd[src + i];
}

// Upsample over [lead, d, h, w] (d == 1 for 2D); sd is the depth scale (1 for 2D), `scale` the h/w scale.
// mode 0 nearest, 1 bilinear, 2 bicubic (A = -0.75); half-pixel centres, border clamp.
struct MvUpParams { uint n; uint d; uint h; uint w; uint sd; uint scale; uint mode; };

static uint mv_taps(uint o, uint n, uint s, uint mode, thread int* idx, thread float* wt) {
    float src = ((float)o + 0.5f) / (float)s - 0.5f;
    if (mode == 0) {
        idx[0] = (int)min(o / s, n - 1);
        wt[0] = 1.0f;
        return 1;
    }
    if (mode == 1) {
        src = max(src, 0.0f);
        float i0f = floor(src);
        float l = src - i0f;
        uint i0 = min((uint)i0f, n - 1);
        idx[0] = (int)i0;
        wt[0] = 1.0f - l;
        idx[1] = (int)min(i0 + 1, n - 1);
        wt[1] = l;
        return 2;
    }
    const float A = -0.75f;
    float i0f = floor(src);
    float t = src - i0f;
    float x0 = t + 1.0f, x1 = t, x2 = 1.0f - t, x3 = 2.0f - t;
    wt[0] = ((A * x0 - 5.0f * A) * x0 + 8.0f * A) * x0 - 4.0f * A;
    wt[1] = ((A + 2.0f) * x1 - (A + 3.0f)) * x1 * x1 + 1.0f;
    wt[2] = ((A + 2.0f) * x2 - (A + 3.0f)) * x2 * x2 + 1.0f;
    wt[3] = ((A * x3 - 5.0f * A) * x3 + 8.0f * A) * x3 - 4.0f * A;
    for (int k = 0; k < 4; ++k) idx[k] = clamp((int)i0f + k - 1, 0, (int)n - 1);
    return 4;
}

kernel void mv_upsample(device const float* x [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant MvUpParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint od = p.d * p.sd, oh = p.h * p.scale, ow = p.w * p.scale;
    uint xx = gid % ow;
    uint y = (gid / ow) % oh;
    uint z = (gid / (ow * oh)) % od;
    uint l = gid / (ow * oh * od);
    int iz[4], iy[4], ix[4];
    float wz[4], wy[4], wx[4];
    uint nz = mv_taps(z, p.d, p.sd, p.mode, iz, wz);
    uint ny = mv_taps(y, p.h, p.scale, p.mode, iy, wy);
    uint nx = mv_taps(xx, p.w, p.scale, p.mode, ix, wx);
    device const float* plane = x + (ulong)l * p.d * p.h * p.w;
    float acc = 0.0f;
    for (uint a = 0; a < nz; ++a)
        for (uint b = 0; b < ny; ++b)
            for (uint c = 0; c < nx; ++c)
                acc += wz[a] * wy[b] * wx[c] * plane[((ulong)iz[a] * p.h + iy[b]) * p.w + ix[c]];
    out[gid] = acc;
}
