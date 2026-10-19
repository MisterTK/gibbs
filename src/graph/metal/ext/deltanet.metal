// Kernels for the `deltanet` lowering group (see deltanet.rs): one decode token of a gated delta-rule
// recurrent layer in one dispatch. Shares the preamble of metal_kernels.metal (`block_sum`).

struct DeltaNetParams {
    uint channels;     // conv channels: 2 * key_heads * key_dim + val_heads * val_dim
    uint taps;         // conv taps (`kernel` is reserved in MSL)
    uint dilation;
    uint key_heads;
    uint key_dim;
    uint val_dim;
    uint rep;          // value heads per key head
    uint cw;           // state columns per row group (val_dim rounded up to 32)
    uint rgroups;      // row groups: DN_THREADS / cw
    uint rpt;          // state rows per thread: ceil(key_dim / rgroups) <= DN_MAX_RPT
    uint z_act;        // gate on z: 0 none (already applied), 1 SiLU, 2 sigmoid
    uint beta_sigmoid; // 1: beta = sigmoid(b), 0: b is beta
    uint has_dt_bias;  // 1: decay input is a + dt_bias
    float eps;         // output RMSNorm epsilon
};

constant uint DN_THREADS = 1024;
constant uint DN_MAX_KD = 256;
constant uint DN_MAX_VD = 256;
constant uint DN_MAX_RPT = 16;

// Recurrent-state storage; arithmetic is f32 throughout.
struct DnF32 {
    typedef float T;
    static float ld(float v) { return v; }
    static float st(float f) { return f; }
};
struct DnF16 {
    typedef half T;
    static float ld(half v) { return float(v); }
    static half st(float f) { return half(f); }
};
struct DnBF16 {
    typedef ushort T;
    static float ld(ushort v) { return as_type<float>(uint(v) << 16); }
    // Round to nearest even (no NaN special case).
    static ushort st(float f) {
        uint b = as_type<uint>(f);
        return ushort((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    }
};

// One-token causal conv of channel `c` over [state rows, x], shifting x into the state; then SiLU.
static inline float dn_conv_silu(device const float* x, device const float* w, device float* cs,
                                 uint c, constant DeltaNetParams& p) {
    const uint k = p.taps, d = p.dilation, hist = (k - 1) * d, n = p.channels;
    float acc = 0.0f;
    for (uint tap = 0; tap < k; ++tap) {
        const uint back = (k - 1 - tap) * d;
        const float v = back == 0 ? x[c] : cs[(ulong)(hist - back) * n + c];
        acc += w[(ulong)c * k + tap] * v;
    }
    for (uint j = 0; j + 1 < hist; ++j) cs[(ulong)j * n + c] = cs[(ulong)(j + 1) * n + c];
    if (hist > 0) cs[(ulong)(hist - 1) * n + c] = x[c];
    return acc / (1.0f + exp(-acc));
}

// One threadgroup per key head (it owns that head's q/k conv channels, so no other threadgroup touches
// them) looping over its `rep` value heads. Thread (column j = tid % cw, row group rg = tid / cw) keeps
// rows i = rg + t * rgroups of state column j in registers: the state is read and written once.
// The readout uses o_j = decay * (S^T q)_j + delta_j * (k.q), so one pass yields both column sums.
template <typename S>
[[kernel, max_total_threads_per_threadgroup(1024)]] void deltanet_step(
        device const float* x [[buffer(0)]],
        device const float* conv_w [[buffer(1)]],
        device float* conv_state [[buffer(2)]],
        device const float* z [[buffer(3)]],
        device const float* b [[buffer(4)]],
        device const float* a [[buffer(5)]],
        device const float* dt_bias [[buffer(6)]],
        device const float* neg_a [[buffer(7)]],
        device typename S::T* state [[buffer(8)]],
        device const float* norm_w [[buffer(9)]],
        device float* out [[buffer(10)]],
        constant DeltaNetParams& p [[buffer(11)]],
        uint kg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[DN_MAX_KD];
    threadgroup float ks[DN_MAX_KD];
    threadgroup float vs[DN_MAX_VD];
    threadgroup float part[DN_THREADS];
    threadgroup float red[32];
    const uint kd = p.key_dim, vd = p.val_dim, kk = p.key_heads * kd, nsg = DN_THREADS / 32;

    for (uint c = tid; c < 2 * kd; c += DN_THREADS) {
        const bool is_q = c < kd;
        const uint ch = is_q ? kg * kd + c : kk + kg * kd + (c - kd);
        const float v = dn_conv_silu(x, conv_w, conv_state, ch, p);
        if (is_q) {
            qs[c] = v;
        } else {
            ks[c - kd] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sqq = 0.0f, skk = 0.0f, sqk = 0.0f;
    for (uint i = tid; i < kd; i += DN_THREADS) {
        sqq += qs[i] * qs[i];
        skk += ks[i] * ks[i];
        sqk += qs[i] * ks[i];
    }
    sqq = block_sum(sqq, red, sg, lane, nsg);
    skk = block_sum(skk, red, sg, lane, nsg);
    sqk = block_sum(sqk, red, sg, lane, nsg);
    const float q_scale = (1.0f / sqrt(float(kd))) / max(sqrt(sqq), 1e-12f);
    const float k_scale = 1.0f / max(sqrt(skk), 1e-12f);
    const float qk = sqk * q_scale * k_scale;

    const uint j = tid % p.cw, rg = tid / p.cw;
    const bool col = j < vd;
    const bool live = col && rg < p.rgroups;
    for (uint r = 0; r < p.rep; ++r) {
        const uint h = kg * p.rep + r;
        // The previous head's last `vs` read precedes its final block_sum barriers.
        for (uint c = tid; c < vd; c += DN_THREADS) {
            vs[c] = dn_conv_silu(x, conv_w, conv_state, 2 * kk + h * vd + c, p);
        }
        const float bv = b[h];
        const float beta = p.beta_sigmoid != 0 ? 1.0f / (1.0f + exp(-bv)) : bv;
        const float av = p.has_dt_bias != 0 ? a[h] + dt_bias[h] : a[h];
        const float sp = av > 20.0f ? av : log(1.0f + exp(av));
        const float decay = exp(neg_a[h] * sp);

        device typename S::T* st = state + (ulong)h * kd * vd;
        float ls[DN_MAX_RPT];
        float kv = 0.0f, qsum = 0.0f;
        for (uint t = 0; t < DN_MAX_RPT; ++t) {
            const uint i = rg + t * p.rgroups;
            ls[t] = 0.0f;
            if (live && t < p.rpt && i < kd) {
                ls[t] = S::ld(st[(ulong)i * vd + j]);
                kv += ls[t] * ks[i];
                qsum += ls[t] * qs[i];
            }
        }
        part[tid] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kv_j = 0.0f;
        if (col) {
            for (uint gi = 0; gi < p.rgroups; ++gi) kv_j += part[gi * p.cw + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        part[tid] = qsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qs_j = 0.0f;
        if (col) {
            for (uint gi = 0; gi < p.rgroups; ++gi) qs_j += part[gi * p.cw + j];
        }
        const float delta = col ? beta * (vs[j] - decay * kv_j * k_scale) : 0.0f;
        const float o = decay * qs_j * q_scale + delta * qk;
        for (uint t = 0; t < DN_MAX_RPT; ++t) {
            const uint i = rg + t * p.rgroups;
            if (live && t < p.rpt && i < kd) {
                st[(ulong)i * vd + j] = S::st(decay * ls[t] + ks[i] * k_scale * delta);
            }
        }
        // Gated RMSNorm over the head's vd outputs; row group 0 holds each column once.
        const bool head = col && rg == 0;
        const float ss = block_sum(head ? o * o : 0.0f, red, sg, lane, nsg);
        if (head) {
            const float inv = 1.0f / sqrt(ss / float(vd) + p.eps);
            const float zv = z[(ulong)h * vd + j];
            const float gate = p.z_act == 1 ? zv / (1.0f + exp(-zv))
                             : (p.z_act == 2 ? 1.0f / (1.0f + exp(-zv)) : zv);
            out[(ulong)h * vd + j] = o * inv * norm_w[j] * gate;
        }
    }
}

template [[host_name("deltanet_step_f32")]] [[kernel]] decltype(deltanet_step<DnF32>) deltanet_step<DnF32>;
template [[host_name("deltanet_step_f16")]] [[kernel]] decltype(deltanet_step<DnF16>) deltanet_step<DnF16>;
template [[host_name("deltanet_step_bf16")]] [[kernel]] decltype(deltanet_step<DnBF16>) deltanet_step<DnBF16>;
