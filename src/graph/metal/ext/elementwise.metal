// Kernels for the `elementwise` lowering group (see elementwise.rs). Shares the preamble of metal_kernels.metal
// (`gelu_erf` / `gelu_tanh` come from its GEMV-COMMON block). Kind codes mirror `UnaryCode` / `BinaryCode` in elementwise.rs.

struct EwUnaryParams {
    uint n;
    uint kind;
    float a; // Scale: factor; Clamp: min; SoftCap: cap
    float b; // Clamp: max
    uint xlen;    // x is rows of `xlen` elements `xstride` apart (dense: xlen == xstride)
    uint xstride;
};

kernel void unary_f32(device const float* x [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant EwUnaryParams& p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float v = x[p.xlen == p.xstride ? i : (i / p.xlen) * p.xstride + i % p.xlen];
    float r = v;
    switch (p.kind) {
        case 0: r = exp(v); break;
        case 1: r = 1.0f / (1.0f + exp(-v)); break;
        case 2: r = tanh(v); break;
        case 3: r = v > 20.0f ? v : log(1.0f + exp(v)); break; // Softplus
        case 4: r = -v; break;
        case 5: r = sqrt(v); break;
        case 6: r = 1.0f / sqrt(v); break; // Rsqrt
        case 7: r = fabs(v); break;
        case 8: r = max(v, 0.0f); break; // Relu
        case 9: { float t = max(v, 0.0f); r = t * t; break; } // Relu2
        case 10: r = v / (1.0f + exp(-v)); break; // Silu
        case 11: r = gelu_erf(v); break;
        case 12: r = gelu_tanh(v); break;
        case 13: r = v * p.a; break; // Scale
        case 14: { if (v < p.a) v = p.a; if (v > p.b) v = p.b; r = v; break; } // Clamp, NaN passes through like f32::clamp
        case 15: r = p.a * tanh(v / p.a); break; // SoftCap
    }
    out[i] = r;
}

// Broadcasting binary op. The planner coalesces the output into `rank <= 6` dims (outermost first);
// `sa`/`sb` are each operand's element stride per dim (0 where it is broadcast).
struct EwBinaryParams {
    uint n;
    uint kind;
    uint rank;
    uint pad;
    uint dims[6];
    uint sa[6];
    uint sb[6];
};

// ponytail: one uint div/mod per dim per element when rank > 1 (equal-shape and scalar broadcasts
// coalesce to rank 1 and skip it); specialise per broadcast pattern if a profile shows these bandwidth-bound.
kernel void binary_f32(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant EwBinaryParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint ia, ib;
    if (p.rank == 1) {
        ia = gid * p.sa[0];
        ib = gid * p.sb[0];
    } else {
        uint rem = gid;
        ia = 0;
        ib = 0;
        for (uint r = p.rank; r-- > 0;) {
            uint d = p.dims[r];
            uint c = rem % d;
            rem /= d;
            ia += c * p.sa[r];
            ib += c * p.sb[r];
        }
    }
    float x = a[ia];
    float y = b[ib];
    float r = x;
    switch (p.kind) {
        case 0: r = x + y; break;
        case 1: r = x - y; break;
        case 2: r = x * y; break;
        case 3: r = x / y; break;
        case 4: r = max(x, y); break;
        case 5: r = min(x, y); break;
    }
    out[gid] = r;
}
