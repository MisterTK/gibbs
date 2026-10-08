// Graph-IR kernels for the Metal executor (`src/graph/metal.rs`), all f32. Compiled at runtime by
// `newLibraryWithSource`. `metal.rs` first compiles with `GIBBS_MPP` defined and MSL 4.0, which adds
// the MetalPerformancePrimitives GEMM; if that fails, without it at the default language version.
// Each kernel mirrors a `crate::model::nn` op; `metal.rs` tests compare them against it.
#include <metal_stdlib>
#if defined(GIBBS_MPP)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// Attention: head_dim <= 32 * ATTN_MAX_PER_LANE; simdgroups per threadgroup <= ATTN_MAX_SG.
constant uint ATTN_MAX_PER_LANE = 8;
constant uint ATTN_MAX_SG = 8;

struct GemmParams { uint m; uint n; uint k; uint has_bias; };
struct EmbedParams { uint n; uint d; };
struct NormParams { uint d; float eps; };
struct AttnParams { uint nq; uint nk; uint heads; uint head_dim; uint causal; float scale; };
struct PoolParams { uint n; uint d; };

// Exact GELU with the Numerical Recipes erfc fit used by `nn::gelu` (|erf error| < 1.2e-7), in f32.
static inline float gelu_erf(float x) {
    float z = fabs(x) * M_SQRT1_2_F;
    float t = 1.0f / (1.0f + 0.5f * z);
    float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
        + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    float erfc_v = t * exp(poly);
    float erf_v = x >= 0.0f ? 1.0f - erfc_v : erfc_v - 1.0f;
    return 0.5f * x * (1.0f + erf_v);
}

// Sum across the whole threadgroup; every thread gets the total. Safe to call repeatedly.
static inline float block_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// GEMM epilogues, applied to x·Wᵀ (+ bias) before the store.
enum : int { EPI_NONE = 0, EPI_GELU = 1, EPI_RESIDUAL = 2 };

template <int EPI>
static inline float epilogue(float acc, uint row, uint col, constant GemmParams& p,
                             device const float* bias, device const float* res) {
    float v = p.has_bias ? acc + bias[col] : acc;
    if (EPI == EPI_GELU) v = gelu_erf(v);
    if (EPI == EPI_RESIDUAL) v += res[(ulong)row * p.n + col];
    return v;
}

// out[m, n] = epilogue(x[m, k] · w[n, k]ᵀ): a 32×32 output tile per threadgroup of 4 simdgroups,
// each owning a 16×16 quadrant as 2×2 simdgroup_float8x8 accumulators. K advances 8 at a time
// through threadgroup memory, zero-filled past the matrix edges. Runs on every Apple GPU.
template <int EPI>
kernel void gemm_sg(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device const float* bias [[buffer(2)]],
                    device const float* res [[buffer(3)]],
                    device float* out [[buffer(4)]],
                    constant GemmParams& p [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[32 * 8];  // [row][k]
    threadgroup float ws[8 * 32];  // [k][col]
    threadgroup float cs[32 * 32];
    const uint m0 = tg.y * 32, n0 = tg.x * 32;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = tid; e < 256; e += 128) {
            // Consecutive threads read consecutive k of one row of x and of w.
            uint r = e / 8, kk = e % 8, gk = k0 + kk;
            xs[e] = (m0 + r < p.m && gk < p.k) ? x[(ulong)(m0 + r) * p.k + gk] : 0.0f;
            ws[kk * 32 + r] = (n0 + r < p.n && gk < p.k) ? w[(ulong)(n0 + r) * p.k + gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a[2], b[2];
        for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], xs + (sm + 8 * i) * 8, 8);
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], ws + sn + 8 * j, 32);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], cs + (sm + 8 * i) * 32 + sn + 8 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += 128) {
        uint r = m0 + e / 32, c = n0 + e % 32;
        if (r < p.m && c < p.n) out[(ulong)r * p.n + c] = epilogue<EPI>(cs[e], r, c, p, bias, res);
    }
}

#define GEMM_INSTANCE(name, kern, epi) \
    template [[host_name(name)]] [[kernel]] decltype(kern<epi>) kern<epi>;
GEMM_INSTANCE("gemm_sg_none", gemm_sg, EPI_NONE)
GEMM_INSTANCE("gemm_sg_gelu", gemm_sg, EPI_GELU)
GEMM_INSTANCE("gemm_sg_residual", gemm_sg, EPI_RESIDUAL)

// Same contract for few rows (decode-like m): one simdgroup per output column, i.e. per row of w,
// which is read once per 8 rows of x; 8 columns per threadgroup. Lanes take float4 chunks of k
// (lane 4i.., then +128), any k % 4 tail goes to lanes 0..2, and simd_sum reduces.
template <int EPI>
kernel void gemm_rows(device const float* x [[buffer(0)]],
                      device const float* w [[buffer(1)]],
                      device const float* bias [[buffer(2)]],
                      device const float* res [[buffer(3)]],
                      device float* out [[buffer(4)]],
                      constant GemmParams& p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint col = tg * 8 + sg;
    if (col >= p.n) return;  // uniform per simdgroup
    device const float* wr = w + (ulong)col * p.k;
    const uint k4 = p.k & ~3u;
    for (uint r0 = 0; r0 < p.m; r0 += 8) {
        const uint rows = min(8u, p.m - r0);
        device const float* xr = x + (ulong)r0 * p.k;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = lane * 4; i < k4; i += 128) {
            float4 wv = float4(*(device const packed_float4*)(wr + i));
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                if (j < rows) acc[j] += dot(wv, float4(*(device const packed_float4*)(xr + (ulong)j * p.k + i)));
            }
        }
        for (uint i = k4 + lane; i < p.k; i += 32) {
            float wv = wr[i];
            for (uint j = 0; j < rows; ++j) acc[j] += wv * xr[(ulong)j * p.k + i];
        }
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            float s = simd_sum(acc[j]);
            if (lane == j && j < rows) out[(ulong)(r0 + j) * p.n + col] = epilogue<EPI>(s, r0 + j, col, p, bias, res);
        }
    }
}

GEMM_INSTANCE("gemm_rows_none", gemm_rows, EPI_NONE)
GEMM_INSTANCE("gemm_rows_gelu", gemm_rows, EPI_GELU)
GEMM_INSTANCE("gemm_rows_residual", gemm_rows, EPI_RESIDUAL)

#if defined(GIBBS_MPP)
using namespace mpp::tensor_ops;

// Same contract as gemm_sg on the GPU's matrix units: MetalPerformancePrimitives `matmul2d`,
// a 64×32 output tile per threadgroup of 4 simdgroups, epilogue applied in registers on the
// cooperative destination tensor before one store (edges are clipped by the tensor extents).
template <int EPI>
kernel void gemm_mpp(device float* x [[buffer(0)]],
                     device float* w [[buffer(1)]],
                     device const float* bias [[buffer(2)]],
                     device const float* res [[buffer(3)]],
                     device float* out [[buffer(4)]],
                     constant GemmParams& p [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
    // extent(0) is the contiguous dimension: x is [m][k], w is [n][k] (so right-transposed), out [m][n].
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tx(x, dextents<int32_t, 2>(p.k, p.m));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tw(w, dextents<int32_t, 2>(p.k, p.n));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(out, dextents<int32_t, 2>(p.n, p.m));
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, tg.y * 64);
    auto mw = tw.slice(0, tg.x * 32);
    auto mo = to.slice(tg.x * 32, tg.y * 64);
    auto c = op.get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (c.is_valid_element(i)) c[i] = 0.0f;
    }
    op.run(mx, mw, c);
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            uint col = tg.x * 32 + uint(idx[0]), row = tg.y * 64 + uint(idx[1]);
            if (row < p.m && col < p.n) c[i] = epilogue<EPI>(c[i], row, col, p, bias, res);
        }
    }
    c.store(mo);
}

GEMM_INSTANCE("gemm_mpp_none", gemm_mpp, EPI_NONE)
GEMM_INSTANCE("gemm_mpp_gelu", gemm_mpp, EPI_GELU)
GEMM_INSTANCE("gemm_mpp_residual", gemm_mpp, EPI_RESIDUAL)
#endif

// out[r, :] = table[ids[r], :]; ids are range-checked on the host before dispatch.
kernel void embedding_f32(device const float* table [[buffer(0)]],
                          device const uint* ids [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant EmbedParams& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.d) return;
    out[i] = table[(ulong)ids[i / p.d] * p.d + i % p.d];
}

kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

kernel void gelu_erf_f32(device const float* x [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = gelu_erf(x[i]);
}

// LayerNorm of one row per threadgroup: mean, then biased variance, like `nn::layer_norm`.
kernel void layer_norm_f32(device const float* x [[buffer(0)]],
                           device const float* g [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant NormParams& p [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint nsg [[simdgroups_per_threadgroup]]) {
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
    for (uint i = tid; i < p.d; i += tpg) o[i] = (xr[i] - mean) * inv * g[i] + b[i];
}

// Scaled dot-product attention without a score matrix: one threadgroup per (head, query row). Its
// simdgroups split the keys, each keeping an online softmax; the partials merge at the end.
// Rows of q/k/v/out are heads*head_dim floats. Causal: query i sees keys 0..=i (as `nn::attention`).
kernel void attention_f32(device const float* q [[buffer(0)]],
                          device const float* k [[buffer(1)]],
                          device const float* v [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          constant AttnParams& p [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * ATTN_MAX_PER_LANE * 32];
    const uint head = tg.x, qi = tg.y;
    const uint hd = p.head_dim, stride = p.heads * hd, row = ATTN_MAX_PER_LANE * 32;
    device const float* qh = q + (ulong)qi * stride + head * hd;
    const uint len = p.causal ? min(p.nk, qi + 1) : p.nk;

    float qr[ATTN_MAX_PER_LANE];
    float acc[ATTN_MAX_PER_LANE];
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        qr[j] = d < hd ? qh[d] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    for (uint t = sg; t < len; t += nsg) {
        device const float* kt = k + (ulong)t * stride + head * hd;
        device const float* vt = v + (ulong)t * stride + head * hd;
        float s = 0.0f;
        for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) s += qr[j] * kt[d];
        }
        s = simd_sum(s) * p.scale;
        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float e = exp(s - m_new);
        l = l * corr + e;
        for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) acc[j] = acc[j] * corr + e * vt[d];
        }
        m = m_new;
    }
    if (lane == 0) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        if (d < hd) tg_acc[sg * row + d] = acc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -FLT_MAX;
    for (uint i = 0; i < nsg; ++i) gm = max(gm, tg_m[i]);
    float gl = 0.0f;
    for (uint i = 0; i < nsg; ++i) gl += tg_l[i] * exp(tg_m[i] - gm);
    device float* o = out + (ulong)qi * stride + head * hd;
    for (uint d = sg * 32 + lane; d < hd; d += nsg * 32) {
        float a = 0.0f;
        for (uint i = 0; i < nsg; ++i) a += tg_acc[i * row + d] * exp(tg_m[i] - gm);
        o[d] = a / gl;
    }
}

// Mean of the rows whose mask entry is non-zero (count clamped to 1), one thread per column.
kernel void mean_pool_f32(device const float* x [[buffer(0)]],
                          device const uint* mask [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant PoolParams& p [[buffer(3)]],
                          uint c [[thread_position_in_grid]]) {
    if (c >= p.d) return;
    float s = 0.0f;
    uint count = 0;
    for (uint r = 0; r < p.n; ++r) {
        if (mask[r] != 0) {
            s += x[(ulong)r * p.d + c];
            ++count;
        }
    }
    out[c] = s / float(max(count, 1u));
}

// x / max(‖x‖₂, 1e-12) in one threadgroup.
kernel void l2_normalize_f32(device const float* x [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    float s = 0.0f;
    for (uint i = tid; i < n; i += tpg) s += x[i] * x[i];
    float norm = max(sqrt(block_sum(s, scratch, sg, lane, nsg)), 1e-12f);
    for (uint i = tid; i < n; i += tpg) out[i] = x[i] / norm;
}

// Host profile: GPU cost of one dispatch boundary (an empty kernel back to back).
kernel void noop(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0xffffffffu) x[0] = 0.0f;
}

// Host profile: Streaming read throughput benchmark sweeping a memory buffer.
kernel void memory_sweep(device const ulong2* in [[buffer(0)]],
                         device atomic_uint* out [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint tid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    ulong2 s = 0;
    uint idx = tid * 4;
    if (idx + 3 < count) {
        s += in[idx];
        s += in[idx + 1];
        s += in[idx + 2];
        s += in[idx + 3];
    }
    uint acc = uint(s.x ^ s.y);
    acc = simd_sum(acc);
    if (lane == 0) {
        atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
    }
}

struct RmsNormParams { uint rows; uint d; float eps; };
kernel void rmsnorm_f32(device const float* x [[buffer(0)]],
                        device const float* g [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant RmsNormParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    uint row = tg;
    if (row >= p.rows) return;
    device const float* xr = x + (ulong)row * p.d;
    device float* outr = out + (ulong)row * p.d;
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) {
        outr[i] = xr[i] * scale * g[i];
    }
}

struct GraphRopeParams { uint n_heads; uint head_dim; uint pairs; uint pair_stride; uint pair_gap; };
kernel void graph_rope_f32(device const float* x [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant float* cs [[buffer(2)]],
                           constant GraphRopeParams& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    uint total = p.n_heads * p.head_dim;
    if (gid >= total) return;
    uint head = gid / p.head_dim;
    uint d = gid % p.head_dim;
    uint pairs = p.pairs;
    uint i0 = head * p.head_dim + d;
    if (d < pairs * 2) {
        uint pair_idx = d / 2;
        float c = cs[pair_idx];
        float s = cs[pairs + pair_idx];
        if (d % 2 == 0) {
            float a = x[head * p.head_dim + pair_idx * 2];
            float b = x[head * p.head_dim + pair_idx * 2 + 1];
            out[i0] = a * c - b * s;
        } else {
            float a = x[head * p.head_dim + pair_idx * 2];
            float b = x[head * p.head_dim + pair_idx * 2 + 1];
            out[i0] = b * c + a * s;
        }
    } else {
        out[i0] = x[i0];
    }
}
kernel void swiglu_f32(device const float* gate [[buffer(0)]],
                       device const float* up [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

kernel void geglu_f32(device const float* gate [[buffer(0)]],
                      device const float* up [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = gelu_erf(gate[i]) * up[i];
}

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

struct PackedGemvParams {
    uint rows;
    uint cols;
    uint batch_size;
    uint has_bias;
    uint epi; // 0 = None, 1 = Gelu, 2 = Residual
};

kernel void gemv_q4_0(device const block_q4_0* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    device const block_q4_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint b = lane; b < n_blocks; b += 32) {
        block_q4_0 blk0 = r0_blocks[b];
        block_q4_0 blk1 = has_row1 ? r1_blocks[b] : blk0;
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        device const float* xb = xb_base + b * 32;

        float b_acc0 = 0.0f;
        float b_acc1 = 0.0f;

        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            uint32_t q0 = *reinterpret_cast<thread const uint32_t*>(&blk0.qs[j]);
            uint32_t q1 = *reinterpret_cast<thread const uint32_t*>(&blk1.qs[j]);

            float4 v0_0 = float4(int4((q0 & 0x0F), ((q0 >> 8) & 0x0F), ((q0 >> 16) & 0x0F), ((q0 >> 24) & 0x0F)) - 8);
            float4 v1_0 = float4(int4(((q0 >> 4) & 0x0F), ((q0 >> 12) & 0x0F), ((q0 >> 20) & 0x0F), ((q0 >> 28) & 0x0F)) - 8);

            float4 v0_1 = float4(int4((q1 & 0x0F), ((q1 >> 8) & 0x0F), ((q1 >> 16) & 0x0F), ((q1 >> 24) & 0x0F)) - 8);
            float4 v1_1 = float4(int4(((q1 >> 4) & 0x0F), ((q1 >> 12) & 0x0F), ((q1 >> 20) & 0x0F), ((q1 >> 28) & 0x0F)) - 8);

            float4 x0 = *reinterpret_cast<device const float4*>(xb + j);
            float4 x1 = *reinterpret_cast<device const float4*>(xb + j + 16);

            b_acc0 += dot(v0_0, x0) + dot(v1_0, x1);
            b_acc1 += dot(v0_1, x0) + dot(v1_1, x1);
        }
        acc0 += d0 * b_acc0;
        if (has_row1) {
            acc1 += d1 * b_acc1;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        if (p.epi == 2) {
            acc0 += res[(ulong)batch * p.rows + row0];
            if (has_row1) acc1 += res[(ulong)batch * p.rows + row1];
        } else if (p.epi == 1) {
            acc0 = gelu_erf(acc0);
            if (has_row1) acc1 = gelu_erf(acc1);
        }
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

kernel void gemv_q8_0(device const block_q8_0* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg.x * 8 + sg;
    uint batch = tg.y;
    if (row >= p.rows || batch >= p.batch_size) return;
    uint n_blocks = p.cols / 32;
    device const block_q8_0* row_blocks = w + (ulong)row * n_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;
    float acc = 0.0f;
    for (uint b = lane; b < n_blocks; b += 32) {
        block_q8_0 blk = row_blocks[b];
        float d = float(blk.d);
        device const float* xb = xb_base + b * 32;
        float b_acc = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 32; j += 4) {
            uint32_t q_raw = *reinterpret_cast<thread const uint32_t*>(&blk.qs[j]);
            char4 q = as_type<char4>(q_raw);
            float4 v = float4(q);
            float4 x_vec = *reinterpret_cast<device const float4*>(xb + j);
            b_acc += dot(v, x_vec);
        }
        acc += d * b_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        if (p.has_bias != 0) acc += bias[row];
        if (p.epi == 2) {
            acc += res[(ulong)batch * p.rows + row];
        } else if (p.epi == 1) {
            acc = gelu_erf(acc);
        }
        y[(ulong)batch * p.rows + row] = acc;
    }
}

kernel void gemv_bf16(device const ushort* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    device const ushort* r0_w = w + (ulong)row0 * p.cols;
    device const ushort* r1_w = has_row1 ? (w + (ulong)row1 * p.cols) : r0_w;
    device const float* xb = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint col_rem = p.cols % 8;
    uint col_aligned = p.cols - col_rem;

    for (uint i = lane * 8; i < col_aligned; i += 32 * 8) {
        ushort4 w0_0 = *reinterpret_cast<device const ushort4*>(r0_w + i);
        ushort4 w0_1 = *reinterpret_cast<device const ushort4*>(r0_w + i + 4);
        ushort4 w1_0 = *reinterpret_cast<device const ushort4*>(r1_w + i);
        ushort4 w1_1 = *reinterpret_cast<device const ushort4*>(r1_w + i + 4);

        float4 xb0 = *reinterpret_cast<device const float4*>(xb + i);
        float4 xb1 = *reinterpret_cast<device const float4*>(xb + i + 4);

        float4 val0_0, val0_1;
        val0_0[0] = as_type<float>(uint(w0_0[0]) << 16);
        val0_0[1] = as_type<float>(uint(w0_0[1]) << 16);
        val0_0[2] = as_type<float>(uint(w0_0[2]) << 16);
        val0_0[3] = as_type<float>(uint(w0_0[3]) << 16);

        val0_1[0] = as_type<float>(uint(w0_1[0]) << 16);
        val0_1[1] = as_type<float>(uint(w0_1[1]) << 16);
        val0_1[2] = as_type<float>(uint(w0_1[2]) << 16);
        val0_1[3] = as_type<float>(uint(w0_1[3]) << 16);

        acc0 += dot(val0_0, xb0) + dot(val0_1, xb1);

        if (has_row1) {
            float4 val1_0, val1_1;
            val1_0[0] = as_type<float>(uint(w1_0[0]) << 16);
            val1_0[1] = as_type<float>(uint(w1_0[1]) << 16);
            val1_0[2] = as_type<float>(uint(w1_0[2]) << 16);
            val1_0[3] = as_type<float>(uint(w1_0[3]) << 16);

            val1_1[0] = as_type<float>(uint(w1_1[0]) << 16);
            val1_1[1] = as_type<float>(uint(w1_1[1]) << 16);
            val1_1[2] = as_type<float>(uint(w1_1[2]) << 16);
            val1_1[3] = as_type<float>(uint(w1_1[3]) << 16);

            acc1 += dot(val1_0, xb0) + dot(val1_1, xb1);
        }
    }
    for (uint i = col_aligned + lane; i < p.cols; i += 32) {
        uint u0 = uint(r0_w[i]) << 16;
        acc0 += as_type<float>(u0) * xb[i];
        if (has_row1) {
            uint u1 = uint(r1_w[i]) << 16;
            acc1 += as_type<float>(u1) * xb[i];
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        if (p.epi == 2) {
            acc0 += res[(ulong)batch * p.rows + row0];
            if (has_row1) acc1 += res[(ulong)batch * p.rows + row1];
        } else if (p.epi == 1) {
            acc0 = gelu_erf(acc0);
            if (has_row1) acc1 = gelu_erf(acc1);
        }
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_f16(device const half* w [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant PackedGemvParams& p [[buffer(3)]],
                     device const float* bias [[buffer(4)]],
                     device const float* res [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    device const half* r0_w = w + (ulong)row0 * p.cols;
    device const half* r1_w = has_row1 ? (w + (ulong)row1 * p.cols) : r0_w;
    device const float* xb = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint col_rem = p.cols % 8;
    uint col_aligned = p.cols - col_rem;

    for (uint i = lane * 8; i < col_aligned; i += 32 * 8) {
        half4 w0_0 = *reinterpret_cast<device const half4*>(r0_w + i);
        half4 w0_1 = *reinterpret_cast<device const half4*>(r0_w + i + 4);
        half4 w1_0 = *reinterpret_cast<device const half4*>(r1_w + i);
        half4 w1_1 = *reinterpret_cast<device const half4*>(r1_w + i + 4);

        float4 xb0 = *reinterpret_cast<device const float4*>(xb + i);
        float4 xb1 = *reinterpret_cast<device const float4*>(xb + i + 4);

        acc0 += dot(float4(w0_0), xb0) + dot(float4(w0_1), xb1);
        if (has_row1) {
            acc1 += dot(float4(w1_0), xb0) + dot(float4(w1_1), xb1);
        }
    }
    for (uint i = col_aligned + lane; i < p.cols; i += 32) {
        acc0 += float(r0_w[i]) * xb[i];
        if (has_row1) {
            acc1 += float(r1_w[i]) * xb[i];
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        if (p.epi == 2) {
            acc0 += res[(ulong)batch * p.rows + row0];
            if (has_row1) acc1 += res[(ulong)batch * p.rows + row1];
        } else if (p.epi == 1) {
            acc0 = gelu_erf(acc0);
            if (has_row1) acc1 = gelu_erf(acc1);
        }
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_tq2_0(device const uint8_t* w [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       constant PackedGemvParams& p [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 8 + sg;
    if (row >= p.rows) return;
    uint bytes = p.cols / 4;
    device const uint8_t* row_w = w + (ulong)row * bytes;
    float acc = 0.0f;
    for (uint b = lane; b < bytes; b += 32) {
        uint8_t byte = row_w[b];
        uint base = b * 4;
        for (uint j = 0; j < 4; ++j) {
            int code = int((byte >> (j * 2)) & 0x03);
            float val = float(code - 1);
            acc += val * x[base + j];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

enum : int {
    FMT_BF16 = 0,
    FMT_F16 = 1,
    FMT_Q4_0 = 2,
    FMT_Q8_0 = 3,
};

template <int FORMAT, int EPI>
kernel void gemm_batched_packed(device const float* x [[buffer(0)]],
                                device const void* w [[buffer(1)]],
                                device const float* bias [[buffer(2)]],
                                device const float* res [[buffer(3)]],
                                device float* out [[buffer(4)]],
                                constant GemmParams& p [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[32 * 8];  // [row][k]
    threadgroup float ws[8 * 32];  // [k][col]
    threadgroup float cs[32 * 32];
    const uint m0 = tg.y * 32, n0 = tg.x * 32;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    device const ushort* w_bf16 = (device const ushort*)w;
    device const half* w_f16 = (device const half*)w;
    device const block_q4_0* w_q4 = (device const block_q4_0*)w;
    device const block_q8_0* w_q8 = (device const block_q8_0*)w;
    const uint q4_n_blocks = p.k / 32;
    const uint q8_n_blocks = p.k / 32;

    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = tid; e < 256; e += 128) {
            uint r = e / 8, kk = e % 8, gk = k0 + kk;
            xs[e] = (m0 + r < p.m && gk < p.k) ? x[(ulong)(m0 + r) * p.k + gk] : 0.0f;

            uint col = n0 + r;
            float val = 0.0f;
            if (col < p.n && gk < p.k) {
                if (FORMAT == FMT_BF16) {
                    ushort u = w_bf16[(ulong)col * p.k + gk];
                    val = as_type<float>(uint(u) << 16);
                } else if (FORMAT == FMT_F16) {
                    val = float(w_f16[(ulong)col * p.k + gk]);
                } else if (FORMAT == FMT_Q8_0) {
                    device const block_q8_0* blk = w_q8 + (ulong)col * q8_n_blocks + (gk / 32);
                    val = float(blk->d) * float(blk->qs[gk % 32]);
                } else if (FORMAT == FMT_Q4_0) {
                    device const block_q4_0* blk = w_q4 + (ulong)col * q4_n_blocks + (gk / 32);
                    uint off = gk % 32;
                    int q = (off < 16) ? (int(blk->qs[off] & 0x0F) - 8) : (int(blk->qs[off - 16] >> 4) - 8);
                    val = float(blk->d) * float(q);
                }
            }
            ws[kk * 32 + r] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a[2], b[2];
        for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], xs + (sm + 8 * i) * 8, 8);
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], ws + sn + 8 * j, 32);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], cs + (sm + 8 * i) * 32 + sn + 8 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += 128) {
        uint r = m0 + e / 32, c = n0 + e % 32;
        if (r < p.m && c < p.n) out[(ulong)r * p.n + c] = epilogue<EPI>(cs[e], r, c, p, bias, res);
    }
}

#define GEMM_BATCHED_INSTANCES(fmt_name, fmt_val) \
    template [[host_name("gemm_batched_" fmt_name "_none")]] [[kernel]] decltype(gemm_batched_packed<fmt_val, EPI_NONE>) gemm_batched_packed<fmt_val, EPI_NONE>; \
    template [[host_name("gemm_batched_" fmt_name "_gelu")]] [[kernel]] decltype(gemm_batched_packed<fmt_val, EPI_GELU>) gemm_batched_packed<fmt_val, EPI_GELU>; \
    template [[host_name("gemm_batched_" fmt_name "_residual")]] [[kernel]] decltype(gemm_batched_packed<fmt_val, EPI_RESIDUAL>) gemm_batched_packed<fmt_val, EPI_RESIDUAL>;

GEMM_BATCHED_INSTANCES("bf16", FMT_BF16)
GEMM_BATCHED_INSTANCES("f16", FMT_F16)
GEMM_BATCHED_INSTANCES("q4_0", FMT_Q4_0)
GEMM_BATCHED_INSTANCES("q8_0", FMT_Q8_0)

struct SliceParams {
    uint rows;
    uint in_cols;
    uint start;
    uint out_cols;
};

kernel void slice_f32(device const float* x [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant SliceParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint total = p.rows * p.out_cols;
    if (gid >= total) return;
    uint r = gid / p.out_cols;
    uint c = gid % p.out_cols;
    out[gid] = x[(ulong)r * p.in_cols + (p.start + c)];
}

struct GraphRopeParams2 {
    uint n_heads;
    uint head_dim;
    uint pairs;
    uint pair_stride;
    uint pair_gap;
    uint rotary_dim;
    float base;
    uint n_seq;
};

kernel void rope_general_f32(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             device const uint* pos [[buffer(2)]],
                             constant GraphRopeParams2& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = p.n_seq * p.n_heads * p.head_dim;
    if (gid >= total) return;
    uint d = gid % p.head_dim;
    uint rem = gid / p.head_dim;
    uint head = rem % p.n_heads;
    uint seq = rem / p.n_heads;

    device const float* in_h = in + (ulong)seq * (p.n_heads * p.head_dim) + (ulong)head * p.head_dim;
    device float* out_h = out + (ulong)seq * (p.n_heads * p.head_dim) + (ulong)head * p.head_dim;

    if (d < p.rotary_dim) {
        if (p.pair_stride == 2) {
            uint pair_idx = d / 2;
            float freq = 1.0f / pow(p.base, float(2 * pair_idx) / float(p.rotary_dim));
            float theta = float(pos[seq]) * freq;
            float cos_th = cos(theta);
            float sin_th = sin(theta);
            float a = in_h[pair_idx * 2];
            float b = in_h[pair_idx * 2 + 1];
            out_h[d] = (d % 2 == 0) ? (a * cos_th - b * sin_th) : (b * cos_th + a * sin_th);
        } else {
            uint half_dim = p.rotary_dim / 2;
            if (d < half_dim) {
                float freq = 1.0f / pow(p.base, float(2 * d) / float(p.rotary_dim));
                float theta = float(pos[seq]) * freq;
                float cos_th = cos(theta);
                float sin_th = sin(theta);
                float a = in_h[d];
                float b = in_h[d + half_dim];
                out_h[d] = a * cos_th - b * sin_th;
                out_h[d + half_dim] = b * cos_th + a * sin_th;
            }
        }
    } else {
        out_h[d] = in_h[d];
    }
}

struct KvAppendParams {
    uint m;
    uint kv_dim;
    uint stride;
    uint start_pos;
};

kernel void kv_append_batched(device const float* k_in [[buffer(0)]],
                              device const float* v_in [[buffer(1)]],
                              device float* k_state [[buffer(2)]],
                              device float* v_state [[buffer(3)]],
                              constant KvAppendParams& p [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total = p.m * p.kv_dim;
    if (gid >= total) return;
    uint tok = gid / p.kv_dim;
    uint d = gid % p.kv_dim;
    ulong dst_idx = (ulong)(p.start_pos + tok) * p.stride + d;
    k_state[dst_idx] = k_in[gid];
    v_state[dst_idx] = v_in[gid];
}

struct AttnCachedParams {
    uint n_heads;
    uint head_dim;
    uint stride;
    uint group;
    float scale;
    uint window;
    float softcap;
    uint start_pos;
    uint m;
};

kernel void attn_cached_f32(device const float* q [[buffer(0)]],
                            device const float* kc [[buffer(1)]],
                            device const float* vc [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AttnCachedParams& p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * ATTN_MAX_PER_LANE * 32];
    const uint hd = p.head_dim;
    const uint row = ATTN_MAX_PER_LANE * 32;
    const uint head = tg.x;
    const uint q_tok = tg.y;
    if (head >= p.n_heads || q_tok >= p.m) return;

    device const float* qh = q + ((ulong)q_tok * p.n_heads + head) * hd;
    const uint kv_off = (head / p.group) * hd;
    const uint cur_pos = p.start_pos + q_tok;
    const uint len = cur_pos + 1;

    float qr[ATTN_MAX_PER_LANE];
    float acc[ATTN_MAX_PER_LANE];
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        qr[j] = d < hd ? qh[d] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    uint lo = (p.window > 0 && cur_pos >= p.window) ? (cur_pos - p.window) : 0;
    for (uint t = lo + sg; t < len; t += nsg) {
        device const float* k = kc + (ulong)t * p.stride + kv_off;
        device const float* v = vc + (ulong)t * p.stride + kv_off;
        float s = 0.0f;
        for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) s += qr[j] * k[d];
        }
        s = simd_sum(s) * p.scale;
        if (p.softcap > 0.0f) {
            s = tanh(s / p.softcap) * p.softcap;
        }
        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float e = exp(s - m_new);
        l = l * corr + e;
        for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) acc[j] = acc[j] * corr + e * v[d];
        }
        m = m_new;
    }
    if (lane == 0) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        if (d < hd) tg_acc[sg * row + d] = acc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -FLT_MAX;
    for (uint i = 0; i < nsg; ++i) gm = max(gm, tg_m[i]);
    float gl = 0.0f;
    for (uint i = 0; i < nsg; ++i) gl += tg_l[i] * exp(tg_m[i] - gm);
    for (uint d = sg * 32 + lane; d < hd; d += nsg * 32) {
        float o = 0.0f;
        for (uint i = 0; i < nsg; ++i) o += tg_acc[i * row + d] * exp(tg_m[i] - gm);
        out[((ulong)q_tok * p.n_heads + head) * hd + d] = o / gl;
    }
}

struct Conv1dParams {
    uint n;
    uint channels;
    uint kernel_size;
    uint dilation;
    uint hist;
    uint has_state;
};

kernel void causal_conv1d_f32(device const float* x [[buffer(0)]],
                              device const float* w [[buffer(1)]],
                              device float* state [[buffer(2)]],
                              device float* out [[buffer(3)]],
                              constant Conv1dParams& p [[buffer(4)]],
                              uint ch [[thread_position_in_grid]]) {
    if (ch >= p.channels) return;
    uint k = p.kernel_size;
    uint d = p.dilation;
    uint hist = p.hist;
    device const float* w_ch = w + ch * k;

    for (uint i = 0; i < p.n; ++i) {
        float acc = 0.0f;
        for (uint tap = 0; tap < k; ++tap) {
            uint offset_step = (k - 1 - tap) * d;
            int sample_idx = int(hist + i) - int(offset_step);
            if (sample_idx >= 0) {
                float val;
                if (sample_idx < int(hist)) {
                    val = (p.has_state != 0) ? state[(ulong)sample_idx * p.channels + ch] : 0.0f;
                } else {
                    val = x[(ulong)(sample_idx - hist) * p.channels + ch];
                }
                acc += w_ch[tap] * val;
            }
        }
        out[(ulong)i * p.channels + ch] = acc;
    }

    if (p.has_state != 0) {
        for (uint j = 0; j < hist; ++j) {
            int idx = int(p.n + j);
            float val;
            if (idx < int(hist)) {
                val = state[(ulong)idx * p.channels + ch];
            } else {
                val = x[(ulong)(idx - hist) * p.channels + ch];
            }
            state[(ulong)j * p.channels + ch] = val;
        }
    }
}

struct DeltaRuleParams {
    uint n;
    uint key_heads;
    uint val_heads;
    uint key_dim;
    uint val_dim;
    uint rep;
};

kernel void delta_rule_f32(device const float* q [[buffer(0)]],
                           device const float* k [[buffer(1)]],
                           device const float* v [[buffer(2)]],
                           device const float* beta [[buffer(3)]],
                           device const float* log_decay [[buffer(4)]],
                           device float* state [[buffer(5)]],
                           device float* out [[buffer(6)]],
                           constant DeltaRuleParams& p [[buffer(7)]],
                           uint h [[thread_position_in_grid]]) {
    if (h >= p.val_heads) return;
    uint kh = p.key_heads;
    uint kd = p.key_dim;
    uint vd = p.val_dim;
    uint vh = p.val_heads;
    uint rep = p.rep;
    uint k_h = h / rep;

    device float* ssm_state = state + (ulong)h * kd * vd;

    for (uint t = 0; t < p.n; ++t) {
        device const float* q_in = q + ((ulong)t * kh + k_h) * kd;
        device const float* k_in = k + ((ulong)t * kh + k_h) * kd;
        device const float* v_t = v + ((ulong)t * vh + h) * vd;
        float b = beta[(ulong)t * vh + h];
        float ld = log_decay[(ulong)t * vh + h];
        float decay = exp(ld);

        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (uint i = 0; i < kd; ++i) {
            q_norm += q_in[i] * q_in[i];
            k_norm += k_in[i] * k_in[i];
        }
        q_norm = max(sqrt(q_norm), 1e-12f);
        k_norm = max(sqrt(k_norm), 1e-12f);
        float q_scale = (1.0f / sqrt(float(kd))) / q_norm;
        float k_scale = 1.0f / k_norm;

        for (uint j = 0; j < vd; ++j) {
            float kv_j = 0.0f;
            for (uint i = 0; i < kd; ++i) {
                kv_j += ssm_state[i * vd + j] * (k_in[i] * k_scale);
            }
            float delta_j = b * (v_t[j] - decay * kv_j);
            for (uint i = 0; i < kd; ++i) {
                ssm_state[i * vd + j] = decay * ssm_state[i * vd + j] + (k_in[i] * k_scale) * delta_j;
            }
            float out_j = 0.0f;
            for (uint i = 0; i < kd; ++i) {
                out_j += ssm_state[i * vd + j] * (q_in[i] * q_scale);
            }
            out[((ulong)t * vh + h) * vd + j] = out_j;
        }
    }
}

struct TopKSoftmaxParams {
    uint rows;
    uint cols;
    uint k;
};

kernel void topk_softmax_f32(device const float* x [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant TopKSoftmaxParams& p [[buffer(2)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device const float* xr = x + (ulong)row * p.cols;
    device float* yr = out + (ulong)row * p.cols;
    uint n = p.cols;
    uint k = min(p.k, n);

    for (uint i = 0; i < n; ++i) yr[i] = 0.0f;

    float max_val = -1e38f;
    for (uint i = 0; i < n; ++i) {
        if (xr[i] > max_val) max_val = xr[i];
    }

    if (k >= n) {
        float sum_e = 0.0f;
        for (uint i = 0; i < n; ++i) {
            float e = exp(xr[i] - max_val);
            yr[i] = e;
            sum_e += e;
        }
        float inv = 1.0f / max(sum_e, 1e-12f);
        for (uint i = 0; i < n; ++i) yr[i] *= inv;
        return;
    }

    uint top_indices[32];
    float top_values[32];
    uint cap_k = min(k, 32u);
    for (uint ki = 0; ki < cap_k; ++ki) {
        top_values[ki] = -1e38f;
        top_indices[ki] = 0;
    }

    for (uint i = 0; i < n; ++i) {
        float v = xr[i];
        if (v > top_values[cap_k - 1]) {
            uint pos = cap_k - 1;
            while (pos > 0 && v > top_values[pos - 1]) {
                top_values[pos] = top_values[pos - 1];
                top_indices[pos] = top_indices[pos - 1];
                pos--;
            }
            top_values[pos] = v;
            top_indices[pos] = i;
        }
    }

    float top_max = top_values[0];
    float sum_e = 0.0f;
    float top_exp[32];
    for (uint ki = 0; ki < cap_k; ++ki) {
        float e = exp(top_values[ki] - top_max);
        top_exp[ki] = e;
        sum_e += e;
    }
    float inv = 1.0f / max(sum_e, 1e-12f);
    for (uint ki = 0; ki < cap_k; ++ki) {
        yr[top_indices[ki]] = top_exp[ki] * inv;
    }
}

struct MulParams {
    uint total;
    uint a_len;
    uint b_len;
    uint row_broadcast;
    uint d;
};

kernel void mul_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant MulParams& p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total) return;
    if (p.a_len == p.b_len) {
        out[gid] = a[gid] * b[gid];
    } else if (p.b_len == 1) {
        out[gid] = a[gid] * b[0];
    } else if (p.row_broadcast != 0) {
        uint r = gid / p.d;
        out[gid] = a[gid] * b[r];
    } else {
        uint c = gid % p.b_len;
        out[gid] = a[gid] * b[c];
    }
}

kernel void softcap_f32(device const float* x [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant float& cap [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float v = x[gid];
    out[gid] = tanh(v / cap) * cap;
}
