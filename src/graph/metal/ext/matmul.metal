// Kernels for the `matmul` lowering group (see matmul.rs). Shares the preamble of metal_kernels.metal.
//
// c[bz, i, j] = sum_k A(i, k) * B(k, j) with A(i, k) = a[bz * a_bs + i * a_row + k * a_col] and
// B(k, j) = b[bz * b_bs + k * b_row + j * b_col]. Transposes are just strides; a broadcast batch
// has batch stride 0. c is dense [batch, m, n]; the batch is the z grid dimension.

struct MatmulParams { uint m; uint n; uint k; uint a_row; uint a_col; uint b_row; uint b_col; uint a_bs; uint b_bs; };

// One (16·Q)² output tile per threadgroup of 4 simdgroups, each owning an (8·Q)² quadrant as Q×Q
// simdgroup_float8x8 accumulators. K advances 8 at a time through threadgroup memory, zero-filled
// past the matrix edges; the staging loops read along whichever operand axis is contiguous.
template <int Q>
kernel void matmul_sg(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant MatmulParams& p [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint T = 16 * Q;
    threadgroup float smem[T * T];  // as[T][8] then bs[8][T]; reused as the edge-store tile
    threadgroup float* as = smem;
    threadgroup float* bs = smem + T * 8;
    const uint m0 = tg.y * T, n0 = tg.x * T;
    const uint sm = (sg / 2) * 8 * Q, sn = (sg % 2) * 8 * Q;
    device const float* ap = a + (ulong)tg.z * p.a_bs;
    device const float* bp = b + (ulong)tg.z * p.b_bs;
    simdgroup_float8x8 acc[Q][Q];
    for (uint i = 0; i < Q; ++i)
        for (uint j = 0; j < Q; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = tid; e < T * 8; e += 128) {
            uint r, kk;
            if (p.a_col == 1) { r = e / 8; kk = e % 8; } else { r = e % T; kk = e / T; }
            uint gk = k0 + kk;
            as[r * 8 + kk] = (m0 + r < p.m && gk < p.k) ? ap[(ulong)(m0 + r) * p.a_row + (ulong)gk * p.a_col] : 0.0f;
            uint kb, cb;
            if (p.b_col == 1) { cb = e % T; kb = e / T; } else { kb = e % 8; cb = e / 8; }
            uint gkb = k0 + kb;
            bs[kb * T + cb] = (n0 + cb < p.n && gkb < p.k) ? bp[(ulong)gkb * p.b_row + (ulong)(n0 + cb) * p.b_col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 fa[Q], fb[Q];
        for (uint i = 0; i < Q; ++i) simdgroup_load(fa[i], as + (sm + 8 * i) * 8, 8);
        for (uint j = 0; j < Q; ++j) simdgroup_load(fb[j], bs + sn + 8 * j, T);
        for (uint i = 0; i < Q; ++i)
            for (uint j = 0; j < Q; ++j) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* cp = c + (ulong)tg.z * p.m * p.n;
    if (m0 + T <= p.m && n0 + T <= p.n) {  // uniform per threadgroup: store straight to memory
        for (uint i = 0; i < Q; ++i)
            for (uint j = 0; j < Q; ++j)
                simdgroup_store(acc[i][j], cp + (ulong)(m0 + sm + 8 * i) * p.n + n0 + sn + 8 * j, p.n);
        return;
    }
    for (uint i = 0; i < Q; ++i)
        for (uint j = 0; j < Q; ++j) simdgroup_store(acc[i][j], smem + (sm + 8 * i) * T + sn + 8 * j, T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < T * T; e += 128) {
        uint r = m0 + e / T, col = n0 + e % T;
        if (r < p.m && col < p.n) cp[(ulong)r * p.n + col] = smem[e];
    }
}

template [[host_name("matmul_sg32")]] [[kernel]] decltype(matmul_sg<2>) matmul_sg<2>;
template [[host_name("matmul_sg64")]] [[kernel]] decltype(matmul_sg<4>) matmul_sg<4>;

// Few rows (m <= 8) with n contiguous in b: a threadgroup is 32 columns (lanes, coalesced) x 8 k
// slices; every thread keeps all m accumulators and the slices are summed through threadgroup memory.
kernel void matmul_small_n(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           constant MatmulParams& p [[buffer(3)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8][8][32];
    const uint x = tid % 32, y = tid / 32, col = tg.x * 32 + x;
    device const float* ap = a + (ulong)tg.z * p.a_bs;
    device const float* bp = b + (ulong)tg.z * p.b_bs;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (col < p.n) {
        for (uint kk = y; kk < p.k; kk += 8) {
            float bv = bp[(ulong)kk * p.b_row + col];
            #pragma unroll
            for (uint i = 0; i < 8; ++i)
                if (i < p.m) acc[i] += ap[(ulong)i * p.a_row + (ulong)kk * p.a_col] * bv;
        }
    }
    #pragma unroll
    for (uint i = 0; i < 8; ++i) red[y][i][x] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.m * 32; e += 256) {
        uint i = e / 32, xx = e % 32;
        float s = 0.0f;
        for (uint yy = 0; yy < 8; ++yy) s += red[yy][i][xx];
        if (tg.x * 32 + xx < p.n) c[(ulong)tg.z * p.m * p.n + (ulong)i * p.n + tg.x * 32 + xx] = s;
    }
}

// Few rows (m <= 8), any b strides: one simdgroup per output column, lanes stride over k (coalesced
// when k is contiguous in b, the transposed-weight case), simd_sum reduces; 8 columns per threadgroup.
kernel void matmul_small_k(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           constant MatmulParams& p [[buffer(3)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint col = tg.x * 8 + sg;
    if (col >= p.n) return;  // uniform per simdgroup
    device const float* ap = a + (ulong)tg.z * p.a_bs;
    device const float* bp = b + (ulong)tg.z * p.b_bs + (ulong)col * p.b_col;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint kk = lane; kk < p.k; kk += 32) {
        float bv = bp[(ulong)kk * p.b_row];
        #pragma unroll
        for (uint i = 0; i < 8; ++i)
            if (i < p.m) acc[i] += ap[(ulong)i * p.a_row + (ulong)kk * p.a_col] * bv;
    }
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0 && i < p.m) c[(ulong)tg.z * p.m * p.n + (ulong)i * p.n + col] = s;
    }
}
