// Kernels for the `conv` lowering group (see conv.rs). Shares the preamble of metal_kernels.metal.
// One thread per output element; direct convolution (ponytail: no implicit GEMM / tiling yet).

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
