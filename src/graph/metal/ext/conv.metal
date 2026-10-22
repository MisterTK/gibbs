// Kernels for the `conv` lowering group (see conv.rs). Shares the preamble of metal_kernels.metal.
// Direct convolution (one thread per output element) for grouped/depthwise; groups == 1 runs the tiled
// implicit GEMM `conv_igemm` below.

struct ConvNdParams {
    uint n, in_c, in_h, in_w, out_c, kh, kw, sh, sw, pt, pl, dh, dw, groups, out_h, out_w, has_bias;
};

// x [n, in_c, in_h, in_w], w [out_c, in_c/groups, kh, kw], out [n, out_c, out_h, out_w].
// Conv1d runs here with in_h = kh = 1.
kernel void conv2d_gen_f32(device const float* x [[buffer(0)]],
                           device const float* w [[buffer(1)]],
                           device const float* bias [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant ConvNdParams& p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n * p.out_c * p.out_h * p.out_w) return;
    uint ow = gid % p.out_w;
    uint r = gid / p.out_w;
    uint oh = r % p.out_h;
    r /= p.out_h;
    uint oc = r % p.out_c;
    uint b = r / p.out_c;
    uint cpg = p.in_c / p.groups;
    uint g = oc / (p.out_c / p.groups);
    device const float* xb = x + (b * p.in_c + g * cpg) * p.in_h * p.in_w;
    device const float* wo = w + oc * cpg * p.kh * p.kw;
    float sum = p.has_bias ? bias[oc] : 0.0f;
    int ih0 = int(oh * p.sh) - int(p.pt);
    int iw0 = int(ow * p.sw) - int(p.pl);
    for (uint ic = 0; ic < cpg; ic++) {
        for (uint ky = 0; ky < p.kh; ky++) {
            int ih = ih0 + int(ky * p.dh);
            if (uint(ih) >= p.in_h) continue;
            for (uint kx = 0; kx < p.kw; kx++) {
                int iw = iw0 + int(kx * p.dw);
                if (uint(iw) >= p.in_w) continue;
                sum += xb[(ic * p.in_h + uint(ih)) * p.in_w + uint(iw)] * wo[(ic * p.kh + ky) * p.kw + kx];
            }
        }
    }
    out[gid] = sum;
}

struct Conv3dNdParams {
    uint n, in_c, in_d, in_h, in_w, out_c, kd, kh, kw, sd, sh, sw, pf, pt, pl, dd, dh, dw, groups,
        out_d, out_h, out_w, has_bias;
};

// x [n, in_c, d, h, w], w [out_c, in_c/groups, kd, kh, kw].
kernel void conv3d_gen_f32(device const float* x [[buffer(0)]],
                           device const float* w [[buffer(1)]],
                           device const float* bias [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant Conv3dNdParams& p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n * p.out_c * p.out_d * p.out_h * p.out_w) return;
    uint ow = gid % p.out_w;
    uint r = gid / p.out_w;
    uint oh = r % p.out_h;
    r /= p.out_h;
    uint od = r % p.out_d;
    r /= p.out_d;
    uint oc = r % p.out_c;
    uint b = r / p.out_c;
    uint cpg = p.in_c / p.groups;
    uint g = oc / (p.out_c / p.groups);
    uint plane = p.in_h * p.in_w;
    device const float* xb = x + (b * p.in_c + g * cpg) * p.in_d * plane;
    device const float* wo = w + oc * cpg * p.kd * p.kh * p.kw;
    float sum = p.has_bias ? bias[oc] : 0.0f;
    int id0 = int(od * p.sd) - int(p.pf);
    int ih0 = int(oh * p.sh) - int(p.pt);
    int iw0 = int(ow * p.sw) - int(p.pl);
    for (uint ic = 0; ic < cpg; ic++) {
        for (uint kz = 0; kz < p.kd; kz++) {
            int id = id0 + int(kz * p.dd);
            if (uint(id) >= p.in_d) continue;
            for (uint ky = 0; ky < p.kh; ky++) {
                int ih = ih0 + int(ky * p.dh);
                if (uint(ih) >= p.in_h) continue;
                for (uint kx = 0; kx < p.kw; kx++) {
                    int iw = iw0 + int(kx * p.dw);
                    if (uint(iw) >= p.in_w) continue;
                    sum += xb[((ic * p.in_d + uint(id)) * p.in_h + uint(ih)) * p.in_w + uint(iw)]
                         * wo[((ic * p.kd + kz) * p.kh + ky) * p.kw + kx];
                }
            }
        }
    }
    out[gid] = sum;
}

struct AvgPoolParams {
    uint planes, in_h, in_w, kh, kw, sh, sw, out_h, out_w;
};

// x [planes, in_h, in_w] -> [planes, out_h, out_w], valid windows only, mean over kh*kw.
kernel void avgpool2d_f32(device const float* x [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant AvgPoolParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.planes * p.out_h * p.out_w) return;
    uint ow = gid % p.out_w;
    uint r = gid / p.out_w;
    uint oh = r % p.out_h;
    uint c = r / p.out_h;
    device const float* xc = x + c * p.in_h * p.in_w;
    float sum = 0.0f;
    for (uint ky = 0; ky < p.kh; ky++)
        for (uint kx = 0; kx < p.kw; kx++)
            sum += xc[(oh * p.sh + ky) * p.in_w + ow * p.sw + kx];
    out[gid] = sum / float(p.kh * p.kw);
}

// Implicit-GEMM convolution (groups == 1, 1-3 spatial dims; 1-d/2-d run with the unused dims = 1).
// Per batch image the output is C[out_c, S] = W[out_c, K] * im2col(x)[K, S], K = in_c*kd*kh*kw and
// S = out_d*out_h*out_w, so the tiling is `matmul_sg`'s: a (16 Q)^2 tile per threadgroup of 4
// simdgroups (each an 8Q x 8Q quadrant of Q x Q simdgroup_float8x8 accumulators), K advancing BK at a
// time through threadgroup memory. The B operand is gathered on the fly (zero outside the padded
// input); the (channel, tap) decompositions of a K step are shared through `ktab`.
struct ConvGemmParams {
    uint n, in_c, in_d, in_h, in_w, out_c, kd, kh, kw, sd, sh, sw, pf, pt, pl, dd, dh, dw,
        out_d, out_h, out_w, has_bias, K, S;
};

template <int Q>
kernel void conv_igemm(device const float* x [[buffer(0)]],
                       device const float* w [[buffer(1)]],
                       device const float* bias [[buffer(2)]],
                       device float* out [[buffer(3)]],
                       constant ConvGemmParams& p [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint T = 16 * Q;
    constexpr uint BK = 32;         // K advances BK at a time: fewer barriers per multiply-accumulate
    const uint m0 = tg.y * T, n0 = tg.x * T;
    // 128 is a multiple of T, so every staging pass of a thread hits the same output pixel.
    const uint cb = tid % T;
    const uint s = n0 + cb;
    const bool pv = s < p.S;
    const uint ow = s % p.out_w, ohd = s / p.out_w;
    const uint oh = ohd % p.out_h, od = ohd / p.out_h;
    const int id0 = int(od * p.sd) - int(p.pf), ih0 = int(oh * p.sh) - int(p.pt), iw0 = int(ow * p.sw) - int(p.pl);
    device const float* xb = x + (ulong)tg.z * p.in_c * p.in_d * p.in_h * p.in_w;
    device float* op = out + (ulong)tg.z * p.out_c * p.S;
    threadgroup int4 ktab[BK];      // (channel, tap offset d/h/w) of each k in the step; channel -1 = past K
#if defined(GIBBS_MPP)
    constexpr uint AS = BK, BS = BK;
    threadgroup float as[T * AS];   // [m][k]
    threadgroup float bs[T * BS];   // [s][k]
    auto tA_proto = tensor(as, dextents<int32_t, 2>(int(BK), int(T)), array<int, 2>({1, int(AS)}));
    auto tB_proto = tensor(bs, dextents<int32_t, 2>(int(BK), int(T)), array<int, 2>({1, int(BS)}));
    constexpr auto desc = matmul2d_descriptor(
        int(T), int(T), static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto c = mm.template get_destination_cooperative_tensor<decltype(tA_proto), decltype(tB_proto), float>();
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (c.is_valid_element(i)) c[i] = 0.0f;
    }
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        if (tid < BK) {
            uint gk = k0 + tid;
            int4 t = int4(-1, 0, 0, 0);
            if (gk < p.K) {
                uint kx = gk % p.kw, r = gk / p.kw;
                uint ky = r % p.kh;
                r /= p.kh;
                t = int4(int(r / p.kd), int((r % p.kd) * p.dd), int(ky * p.dh), int(kx * p.dw));
            }
            ktab[tid] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < T * BK; e += 128) {
            uint r = e / BK, kk = e % BK;
            as[r * AS + kk] = (m0 + r < p.out_c && k0 + kk < p.K) ? w[(ulong)(m0 + r) * p.K + k0 + kk] : 0.0f;
            uint kb = e / T;
            int4 t = ktab[kb];
            float v = 0.0f;
            if (pv && t.x >= 0) {
                int id = id0 + t.y, ih = ih0 + t.z, iw = iw0 + t.w;
                if (uint(id) < p.in_d && uint(ih) < p.in_h && uint(iw) < p.in_w)
                    v = xb[(((ulong)t.x * p.in_d + uint(id)) * p.in_h + uint(ih)) * p.in_w + uint(iw)];
            }
            bs[cb * BS + kb] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(tA_proto, tB_proto, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            uint col = n0 + uint(idx[0]), r = m0 + uint(idx[1]);
            if (r < p.out_c && col < p.S) op[(ulong)r * p.S + col] = c[i] + (p.has_bias ? bias[r] : 0.0f);
        }
    }
#else
    constexpr uint AS = BK + 4, BS = T + 4;  // row strides padded off a multiple of 32 floats: no bank conflicts
    constexpr uint SM = (T * T > T * AS + BK * BS) ? T * T : T * AS + BK * BS;
    threadgroup float smem[SM];     // as[T][BK] then bs[BK][T]; reused as the output tile
    threadgroup float* as = smem;
    threadgroup float* bs = smem + T * AS;
    const uint sm = (sg / 2) * 8 * Q, sn = (sg % 2) * 8 * Q;
    simdgroup_float8x8 acc[Q][Q];
    for (uint i = 0; i < Q; ++i)
        for (uint j = 0; j < Q; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        if (tid < BK) {
            uint gk = k0 + tid;
            int4 t = int4(-1, 0, 0, 0);
            if (gk < p.K) {
                uint kx = gk % p.kw, r = gk / p.kw;
                uint ky = r % p.kh;
                r /= p.kh;
                t = int4(int(r / p.kd), int((r % p.kd) * p.dd), int(ky * p.dh), int(kx * p.dw));
            }
            ktab[tid] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < T * BK; e += 128) {
            uint r = e / BK, kk = e % BK;
            as[r * AS + kk] = (m0 + r < p.out_c && k0 + kk < p.K) ? w[(ulong)(m0 + r) * p.K + k0 + kk] : 0.0f;
            uint kb = e / T;
            int4 t = ktab[kb];
            float v = 0.0f;
            if (pv && t.x >= 0) {
                int id = id0 + t.y, ih = ih0 + t.z, iw = iw0 + t.w;
                if (uint(id) < p.in_d && uint(ih) < p.in_h && uint(iw) < p.in_w)
                    v = xb[(((ulong)t.x * p.in_d + uint(id)) * p.in_h + uint(ih)) * p.in_w + uint(iw)];
            }
            bs[kb * BS + cb] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_float8x8 fa[Q], fb[Q];
            for (uint i = 0; i < Q; ++i) simdgroup_load(fa[i], as + (sm + 8 * i) * AS + k8, AS);
            for (uint j = 0; j < Q; ++j) simdgroup_load(fb[j], bs + k8 * BS + sn + 8 * j, BS);
            for (uint i = 0; i < Q; ++i)
                for (uint j = 0; j < Q; ++j) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < Q; ++i)
        for (uint j = 0; j < Q; ++j) simdgroup_store(acc[i][j], smem + (sm + 8 * i) * T + sn + 8 * j, T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < T * T; e += 128) {
        uint r = m0 + e / T, col = n0 + e % T;
        if (r < p.out_c && col < p.S) op[(ulong)r * p.S + col] = smem[e] + (p.has_bias ? bias[r] : 0.0f);
    }
#endif
}

template [[host_name("conv_igemm_q2")]] [[kernel]] decltype(conv_igemm<2>) conv_igemm<2>;
template [[host_name("conv_igemm_q4")]] [[kernel]] decltype(conv_igemm<4>) conv_igemm<4>;
