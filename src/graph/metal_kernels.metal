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
constant uint ATTN_MAX_PER_LANE = 16;
constant uint ATTN_MAX_SG = 8;

struct GemmParams { uint m; uint n; uint k; uint has_bias; };
struct EmbedParams { uint n; uint d; };
struct NormParams { uint d; float eps; };
struct AttnParams { uint nq; uint nk; uint heads; uint head_dim; uint causal; float scale; };
struct PoolParams { uint n; uint d; };

// GEMV-COMMON-BEGIN (also prepended to generated GEMV kernels by `codegen::gemv_packed`)
// Tanh-approximated GELU (`gelu_pytorch_tanh`), the activation of GeGlu.
static inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

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

struct PackedGemvParams {
    uint rows;
    uint cols;
    uint batch_size;
    uint has_bias;
    uint epi; // 0 = None, 1 = Gelu, 2 = Residual, 3 = Silu, 4 = SwiGlu, 5 = GeGlu
};

static inline float apply_gemv_epi(float acc, ulong out_idx, constant PackedGemvParams& p, device const float* res) {
    if (p.epi == 1) {
        return gelu_erf(acc);
    } else if (p.epi == 2) {
        return acc + res[out_idx];
    } else if (p.epi == 3) {
        return acc / (1.0f + exp(-acc));
    } else if (p.epi == 4) {
        float silu = acc / (1.0f + exp(-acc));
        return silu * res[out_idx];
    } else if (p.epi == 5) {
        return gelu_tanh(acc) * res[out_idx];
    } else if (p.epi == 6) {
        float gate = res[out_idx];
        return acc * (gate / (1.0f + exp(-gate)));
    } else if (p.epi == 7) {
        return acc * gelu_tanh(res[out_idx]);
    }
    return acc;
}
// GEMV-COMMON-END

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

// 2x ulong4 variant
kernel void memory_sweep_ulong4(device const ulong4* in [[buffer(0)]],
                                device atomic_uint* out [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                uint tid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    ulong4 s = 0;
    uint idx = tid * 2;
    if (idx + 1 < count) {
        s += in[idx];
        s += in[idx + 1];
    }
    uint acc = uint(s.x ^ s.y ^ s.z ^ s.w);
    acc = simd_sum(acc);
    if (lane == 0) {
        atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
    }
}

// Grid-stride 16 B/thread variant with 4 iterations per thread
kernel void memory_sweep_grid_stride(device const ulong2* in [[buffer(0)]],
                                     device atomic_uint* out [[buffer(1)]],
                                     constant uint& count [[buffer(2)]],
                                     uint tid [[thread_position_in_grid]],
                                     uint grid_size [[threads_per_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    ulong2 s = 0;
    for (uint idx = tid; idx < count; idx += grid_size) {
        s += in[idx];
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
// Horizontal batch of up to 4 independent RmsNorm segments (own x, gamma, out, rows, d, eps each);
// threadgroup `tg` handles one row of the segment its row index falls in.
struct RmsNormBatchParams { uint rows[4]; uint d[4]; float eps[4]; uint nseg; };
kernel void rmsnorm_batch_f32(device const float* x0 [[buffer(0)]], device const float* g0 [[buffer(1)]], device float* o0 [[buffer(2)]],
                              device const float* x1 [[buffer(3)]], device const float* g1 [[buffer(4)]], device float* o1 [[buffer(5)]],
                              device const float* x2 [[buffer(6)]], device const float* g2 [[buffer(7)]], device float* o2 [[buffer(8)]],
                              device const float* x3 [[buffer(9)]], device const float* g3 [[buffer(10)]], device float* o3 [[buffer(11)]],
                              constant RmsNormBatchParams& p [[buffer(12)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint nsg [[simdgroups_per_threadgroup]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xs[4] = {x0, x1, x2, x3};
    device const float* gs[4] = {g0, g1, g2, g3};
    device float* os[4] = {o0, o1, o2, o3};
    uint row = tg;
    uint s = 0;
    while (s < p.nseg && row >= p.rows[s]) { row -= p.rows[s]; s++; }
    if (s >= p.nseg) return;
    uint d = p.d[s];
    device const float* xr = xs[s] + (ulong)row * d;
    device float* outr = os[s] + (ulong)row * d;
    float sq = 0.0f;
    for (uint i = tid; i < d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(d) + p.eps[s]);
    for (uint i = tid; i < d; i += tpg) {
        outr[i] = xr[i] * scale * gs[s][i];
    }
}
kernel void rmsnorm_residual_f32(device const float* x [[buffer(0)]],
                                 device const float* g [[buffer(1)]],
                                 device const float* res [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant RmsNormParams& p [[buffer(4)]],
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
    device const float* resr = res + (ulong)row * p.d;
    device float* outr = out + (ulong)row * p.d;
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) {
        outr[i] = resr[i] + xr[i] * scale * g[i];
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
    out[i] = gelu_tanh(gate[i]) * up[i];
}

struct block_q4_0 {
    half d;
    uint8_t qs[16];
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
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_q4_0_split(device const block_q4_0* w [[buffer(0)]],
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

    const uint il = lane & 1;
    #pragma unroll 2
    for (uint b = lane >> 1; b < n_blocks; b += 16) {
        device const block_q4_0* blk0 = r0_blocks + b;
        device const block_q4_0* blk1 = r1_blocks + b;
        device const packed_ushort4* qp0 = (device const packed_ushort4*)(blk0->qs + il * 8);
        device const packed_ushort4* qp1 = (device const packed_ushort4*)(blk1->qs + il * 8);
        ushort4 h0 = ushort4(*qp0);
        ushort4 h1 = ushort4(*qp1);
        float d0 = float(blk0->d);
        float d1 = float(blk1->d);
        device const float* xb = xb_base + b * 32 + il * 8;
        float4 xa0 = *reinterpret_cast<device const float4*>(xb);
        float4 xa1 = *reinterpret_cast<device const float4*>(xb + 4);
        float4 xc0 = *reinterpret_cast<device const float4*>(xb + 16);
        float4 xc1 = *reinterpret_cast<device const float4*>(xb + 20);

        uint2 q0 = uint2(h0.x | (uint(h0.y) << 16), h0.z | (uint(h0.w) << 16));
        uint2 q1 = uint2(h1.x | (uint(h1.y) << 16), h1.z | (uint(h1.w) << 16));
        float4 lo0a = float4(int4(q0.x & 0x0F, (q0.x >> 8) & 0x0F, (q0.x >> 16) & 0x0F, (q0.x >> 24) & 0x0F) - 8);
        float4 hi0a = float4(int4((q0.x >> 4) & 0x0F, (q0.x >> 12) & 0x0F, (q0.x >> 20) & 0x0F, (q0.x >> 28) & 0x0F) - 8);
        float4 lo0b = float4(int4(q0.y & 0x0F, (q0.y >> 8) & 0x0F, (q0.y >> 16) & 0x0F, (q0.y >> 24) & 0x0F) - 8);
        float4 hi0b = float4(int4((q0.y >> 4) & 0x0F, (q0.y >> 12) & 0x0F, (q0.y >> 20) & 0x0F, (q0.y >> 28) & 0x0F) - 8);
        acc0 += d0 * (dot(lo0a, xa0) + dot(lo0b, xa1) + dot(hi0a, xc0) + dot(hi0b, xc1));
        float4 lo1a = float4(int4(q1.x & 0x0F, (q1.x >> 8) & 0x0F, (q1.x >> 16) & 0x0F, (q1.x >> 24) & 0x0F) - 8);
        float4 hi1a = float4(int4((q1.x >> 4) & 0x0F, (q1.x >> 12) & 0x0F, (q1.x >> 20) & 0x0F, (q1.x >> 28) & 0x0F) - 8);
        float4 lo1b = float4(int4(q1.y & 0x0F, (q1.y >> 8) & 0x0F, (q1.y >> 16) & 0x0F, (q1.y >> 24) & 0x0F) - 8);
        float4 hi1b = float4(int4((q1.y >> 4) & 0x0F, (q1.y >> 12) & 0x0F, (q1.y >> 20) & 0x0F, (q1.y >> 28) & 0x0F) - 8);
        if (has_row1) {
            acc1 += d1 * (dot(lo1a, xa0) + dot(lo1b, xa1) + dot(hi1a, xc0) + dot(hi1b, xc1));
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
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
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    device const block_q8_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q8_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint b = lane; b < n_blocks; b += 32) {
        block_q8_0 blk0 = r0_blocks[b];
        block_q8_0 blk1 = has_row1 ? r1_blocks[b] : blk0;
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        device const float* xb = xb_base + b * 32;

        float b_acc0 = 0.0f;
        float b_acc1 = 0.0f;

        #pragma unroll
        for (uint j = 0; j < 32; j += 4) {
            uint32_t q_raw0 = *reinterpret_cast<thread const uint32_t*>(&blk0.qs[j]);
            uint32_t q_raw1 = *reinterpret_cast<thread const uint32_t*>(&blk1.qs[j]);
            float4 v0 = float4(as_type<char4>(q_raw0));
            float4 v1 = float4(as_type<char4>(q_raw1));
            float4 x_vec = *reinterpret_cast<device const float4*>(xb + j);
            b_acc0 += dot(v0, x_vec);
            b_acc1 += dot(v1, x_vec);
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
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
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

        float4 val0_0 = as_type<float4>(uint4(w0_0) << 16);
        float4 val0_1 = as_type<float4>(uint4(w0_1) << 16);
        acc0 += dot(val0_0, xb0) + dot(val0_1, xb1);

        if (has_row1) {
            float4 val1_0 = as_type<float4>(uint4(w1_0) << 16);
            float4 val1_1 = as_type<float4>(uint4(w1_1) << 16);
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
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
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
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

constant const float kvalues_iq4nl[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
};

struct block_iq4_xs {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};

kernel void gemv_iq4_xs(device const block_iq4_xs* w [[buffer(0)]],
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

    uint n_blocks = p.cols / 256;
    uint n_sub_blocks = n_blocks * 8;
    device const block_iq4_xs* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_iq4_xs* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint sb = lane; sb < n_sub_blocks; sb += 32) {
        uint b = sb >> 3;
        uint ib = sb & 7;
        device const block_iq4_xs& blk0 = r0_blocks[b];
        device const block_iq4_xs& blk1 = r1_blocks[b];
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        int ls0 = int((blk0.scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                | (int((blk0.scales_h >> (2 * ib)) & 3) << 4);
        int ls1 = int((blk1.scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                | (int((blk1.scales_h >> (2 * ib)) & 3) << 4);
        float dl0 = d0 * float(ls0 - 32);
        float dl1 = d1 * float(ls1 - 32);
        device const float* xb_sub = xb_base + (ulong)sb * 32;

        float sb_acc0 = 0.0f;
        float sb_acc1 = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            uint8_t q0 = blk0.qs[ib * 16 + j];
            uint8_t q1 = blk1.qs[ib * 16 + j];
            float v0_0 = dl0 * kvalues_iq4nl[q0 & 0x0F];
            float v1_0 = dl0 * kvalues_iq4nl[q0 >> 4];
            float v0_1 = dl1 * kvalues_iq4nl[q1 & 0x0F];
            float v1_1 = dl1 * kvalues_iq4nl[q1 >> 4];
            float xj = xb_sub[j];
            float xj16 = xb_sub[j + 16];
            sb_acc0 += v0_0 * xj + v1_0 * xj16;
            sb_acc1 += v0_1 * xj + v1_1 * xj16;
        }
        acc0 += sb_acc0;
        acc1 += sb_acc1;
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

constant const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19, 0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919, 0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919, 0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b, 0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b, 0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808, 0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b, 0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08, 0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819, 0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908, 0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b, 0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808, 0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919, 0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808, 0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b, 0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919, 0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819, 0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908, 0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908, 0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908, 0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908, 0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819, 0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819, 0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819, 0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808, 0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08, 0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908, 0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819, 0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808, 0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908, 0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819, 0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19, 0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919, 0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908, 0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08, 0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808, 0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908, 0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819, 0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808, 0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808, 0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b, 0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19, 0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08, 0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908, 0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b, 0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808, 0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08, 0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b, 0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b, 0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08, 0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808, 0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919, 0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808, 0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819, 0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808, 0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919, 0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b, 0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808, 0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b, 0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b, 0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08, 0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b, 0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08, 0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919, 0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19, 0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919, 0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819, 0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919, 0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808, 0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919, 0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b, 0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919, 0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08, 0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808, 0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908, 0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808, 0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908, 0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808, 0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908, 0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919, 0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808, 0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b, 0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08, 0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919, 0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908, 0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808, 0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908, 0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808, 0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808, 0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908, 0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08, 0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b, 0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819, 0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819, 0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808, 0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908, 0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908, 0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819, 0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b, 0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819, 0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b, 0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819, 0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819, 0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808, 0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

struct block_iq2_s {
    half d;
    uint8_t qs[32];
    uint8_t signs[32];
    uint8_t qh[8];
    uint8_t scales[8];
};

kernel void gemv_iq2_m(device const block_iq2_s* w [[buffer(0)]],
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

    uint n_blocks = p.cols / 256;
    uint n_sub_blocks = n_blocks * 8;
    device const block_iq2_s* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_iq2_s* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint sb = lane; sb < n_sub_blocks; sb += 32) {
        uint b = sb >> 3;
        uint ib = sb & 7;
        device const block_iq2_s& blk0 = r0_blocks[b];
        device const block_iq2_s& blk1 = r1_blocks[b];
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        uint8_t sc0 = blk0.scales[ib];
        uint8_t sc1 = blk1.scales[ib];
        float db0_0 = d0 * (0.5f + float(sc0 & 0x0F)) * 0.25f;
        float db1_0 = d0 * (0.5f + float(sc0 >> 4)) * 0.25f;
        float db0_1 = d1 * (0.5f + float(sc1 & 0x0F)) * 0.25f;
        float db1_1 = d1 * (0.5f + float(sc1 >> 4)) * 0.25f;
        device const float* xb_sub = xb_base + (ulong)sb * 32;

        float sb_acc0 = 0.0f;
        float sb_acc1 = 0.0f;
        for (uint l = 0; l < 4; ++l) {
            float db_0 = (l < 2) ? db0_0 : db1_0;
            float db_1 = (l < 2) ? db0_1 : db1_1;
            uint idx0 = (uint)blk0.qs[4 * ib + l] | (((uint)blk0.qh[ib] << (8 - 2 * l)) & 0x300);
            uint idx1 = (uint)blk1.qs[4 * ib + l] | (((uint)blk1.qh[ib] << (8 - 2 * l)) & 0x300);
            uint64_t g0 = iq2s_grid[idx0];
            uint64_t g1 = iq2s_grid[idx1];
            uint8_t s0 = blk0.signs[4 * ib + l];
            uint8_t s1 = blk1.signs[4 * ib + l];
            device const float* x_el = xb_sub + l * 8;
            float4 x0 = *reinterpret_cast<device const float4*>(x_el);
            float4 x1 = *reinterpret_cast<device const float4*>(x_el + 4);
            char4 b0_lo = as_type<char4>((uint32_t)g0);
            char4 b0_hi = as_type<char4>((uint32_t)(g0 >> 32));
            char4 b1_lo = as_type<char4>((uint32_t)g1);
            char4 b1_hi = as_type<char4>((uint32_t)(g1 >> 32));

            for (uint j = 0; j < 4; ++j) {
                float sign0 = (s0 & (1 << j)) ? -1.0f : 1.0f;
                float sign1 = (s1 & (1 << j)) ? -1.0f : 1.0f;
                sb_acc0 += (db_0 * float(b0_lo[j]) * sign0) * x0[j];
                sb_acc1 += (db_1 * float(b1_lo[j]) * sign1) * x0[j];

                float sign0_hi = (s0 & (1 << (j + 4))) ? -1.0f : 1.0f;
                float sign1_hi = (s1 & (1 << (j + 4))) ? -1.0f : 1.0f;
                sb_acc0 += (db_0 * float(b0_hi[j]) * sign0_hi) * x1[j];
                sb_acc1 += (db_1 * float(b1_hi[j]) * sign1_hi) * x1[j];
            }
        }
        acc0 += sb_acc0;
        acc1 += sb_acc1;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

struct block_q4_k {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

inline float2 k4_scale_min(uint j, thread const uint8_t* s) {
    if (j < 4) {
        return float2(float(s[j] & 63), float(s[j + 4] & 63));
    } else {
        float sc = float((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        float m = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        return float2(sc, m);
    }
}

inline float2 k4_scale_min(uint j, device const uint8_t* s) {
    if (j < 4) {
        return float2(float(s[j] & 63), float(s[j + 4] & 63));
    } else {
        float sc = float((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        float m = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        return float2(sc, m);
    }
}

kernel void gemv_q4_k(device const block_q4_k* w [[buffer(0)]],
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

    uint n_blocks = p.cols / 256;
    uint n_groups = n_blocks * 4;
    device const block_q4_k* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_k* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint grp = lane; grp < n_groups; grp += 32) {
        uint b = grp >> 2;
        uint g = grp & 3;
        device const block_q4_k& blk0 = r0_blocks[b];
        device const block_q4_k& blk1 = has_row1 ? r1_blocks[b] : blk0;

        float d0 = float(blk0.d);
        float dmin0 = float(blk0.dmin);
        float2 sm0_0 = k4_scale_min(2 * g, blk0.scales);
        float2 sm1_0 = k4_scale_min(2 * g + 1, blk0.scales);
        float d0_0 = d0 * sm0_0.x;
        float min0_0 = dmin0 * sm0_0.y;
        float d1_0 = d0 * sm1_0.x;
        float min1_0 = dmin0 * sm1_0.y;

        float d1 = float(blk1.d);
        float dmin1 = float(blk1.dmin);
        float2 sm0_1 = k4_scale_min(2 * g, blk1.scales);
        float2 sm1_1 = k4_scale_min(2 * g + 1, blk1.scales);
        float d0_1 = d1 * sm0_1.x;
        float min0_1 = dmin1 * sm0_1.y;
        float d1_1 = d1 * sm1_1.x;
        float min1_1 = dmin1 * sm1_1.y;

        device const float* xb_g = xb_base + (ulong)grp * 64;

        float grp_acc0 = 0.0f;
        float grp_acc1 = 0.0f;
        for (uint l = 0; l < 32; ++l) {
            float x_lo = xb_g[l];
            float x_hi = xb_g[32 + l];

            uint8_t q0 = blk0.qs[g * 32 + l];
            float v0_0 = d0_0 * float(q0 & 0x0F) - min0_0;
            float v1_0 = d1_0 * float(q0 >> 4) - min1_0;
            grp_acc0 += v0_0 * x_lo + v1_0 * x_hi;

            uint8_t q1 = blk1.qs[g * 32 + l];
            float v0_1 = d0_1 * float(q1 & 0x0F) - min0_1;
            float v1_1 = d1_1 * float(q1 >> 4) - min1_1;
            grp_acc1 += v0_1 * x_lo + v1_1 * x_hi;
        }
        acc0 += grp_acc0;
        if (has_row1) acc1 += grp_acc1;
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

struct block_q2_k {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

kernel void gemv_q2_k(device const block_q2_k* w [[buffer(0)]],
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

    uint n_blocks = p.cols / 256;
    device const block_q2_k* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q2_k* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Each thread in SIMDgroup processes a subset of the 256-element block across all blocks in the row
    for (uint b = 0; b < n_blocks; ++b) {
        device const block_q2_k& blk0 = r0_blocks[b];
        device const block_q2_k& blk1 = has_row1 ? r1_blocks[b] : blk0;
        device const float* xb = xb_base + (ulong)b * 256;

        float d0 = float(blk0.d);
        float min0 = float(blk0.dmin);
        float d1 = float(blk1.d);
        float min1 = float(blk1.dmin);

        // 16 sub-blocks of 16 values = 256 values total.
        // lane (0..31) splits across the 16 sub-blocks: sub = lane / 2 (0..15), half = lane % 2 (8 values each).
        uint is = lane >> 1; // 0..15
        uint half_idx = lane & 1; // 0 or 1

        uint sc0 = blk0.scales[is];
        float dl0 = d0 * float(sc0 & 0x0F);
        float ml0 = min0 * float(sc0 >> 4);

        uint sc1 = blk1.scales[is];
        float dl1 = d1 * float(sc1 & 0x0F);
        float ml1 = min1 * float(sc1 >> 4);

        // n in 0..2, shift in [0, 2, 4, 6], half in 0..2
        // is = n * 8 + (shift / 2) * 2 + half
        uint n_part = is >> 3; // 0 or 1
        uint shift = (is & 6); // 0, 2, 4, or 6
        uint half_sub = is & 1; // 0 or 1

        uint q_off = n_part * 32 + half_sub * 16 + half_idx * 8;
        device const float* x_sub = xb + is * 16 + half_idx * 8;

        float b_acc0 = 0.0f;
        float b_acc1 = 0.0f;

        #pragma unroll
        for (uint l = 0; l < 8; ++l) {
            float x_val = x_sub[l];
            uint q0 = (uint(blk0.qs[q_off + l]) >> shift) & 3;
            uint q1 = (uint(blk1.qs[q_off + l]) >> shift) & 3;
            b_acc0 += (dl0 * float(q0) - ml0) * x_val;
            b_acc1 += (dl1 * float(q1) - ml1) * x_val;
        }

        acc0 += b_acc0;
        if (has_row1) {
            acc1 += b_acc1;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

constant const uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

struct block_iq3_s {
    half d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
};

kernel void gemv_iq3_s(device const block_iq3_s* w [[buffer(0)]],
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

    uint n_blocks = p.cols / 256;
    uint n_sub_blocks = n_blocks * 8;
    device const block_iq3_s* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_iq3_s* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint sb = lane; sb < n_sub_blocks; sb += 32) {
        uint b = sb >> 3;
        uint ib = sb & 7;
        device const block_iq3_s& blk0 = r0_blocks[b];
        device const block_iq3_s& blk1 = r1_blocks[b];
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        uint sc0 = (blk0.scales[ib / 2] >> (4 * (ib % 2))) & 0xF;
        uint sc1 = (blk1.scales[ib / 2] >> (4 * (ib % 2))) & 0xF;
        float db0 = d0 * float(1 + 2 * sc0);
        float db1 = d1 * float(1 + 2 * sc1);
        uint h0 = uint(blk0.qh[ib]);
        uint h1 = uint(blk1.qh[ib]);
        device const float* xb_sub = xb_base + (ulong)sb * 32;

        float sb_acc0 = 0.0f;
        float sb_acc1 = 0.0f;
        for (uint l = 0; l < 4; ++l) {
            uint g1_0 = iq3s_grid[uint(blk0.qs[8 * ib + 2 * l]) | ((h0 << (8 - 2 * l)) & 256)];
            uint g2_0 = iq3s_grid[uint(blk0.qs[8 * ib + 2 * l + 1]) | ((h0 << (7 - 2 * l)) & 256)];
            uint8_t s_0 = blk0.signs[4 * ib + l];

            uint g1_1 = iq3s_grid[uint(blk1.qs[8 * ib + 2 * l]) | ((h1 << (8 - 2 * l)) & 256)];
            uint g2_1 = iq3s_grid[uint(blk1.qs[8 * ib + 2 * l + 1]) | ((h1 << (7 - 2 * l)) & 256)];
            uint8_t s_1 = blk1.signs[4 * ib + l];
            device const float* x_el = xb_sub + l * 8;
            float4 x_lo = *reinterpret_cast<device const float4*>(x_el);
            float4 x_hi = *reinterpret_cast<device const float4*>(x_el + 4);
            char4 v1_0 = as_type<char4>(g1_0);
            char4 v2_0 = as_type<char4>(g2_0);
            char4 v1_1 = as_type<char4>(g1_1);
            char4 v2_1 = as_type<char4>(g2_1);

            for (uint j = 0; j < 4; ++j) {
                float xj = x_lo[j];
                float xj4 = x_hi[j];

                float sign0_0 = (s_0 & (1 << j)) ? -1.0f : 1.0f;
                float sign1_0 = (s_0 & (1 << (j + 4))) ? -1.0f : 1.0f;
                sb_acc0 += (db0 * float(v1_0[j]) * sign0_0) * xj;
                sb_acc0 += (db0 * float(v2_0[j]) * sign1_0) * xj4;

                float sign0_1 = (s_1 & (1 << j)) ? -1.0f : 1.0f;
                float sign1_1 = (s_1 & (1 << (j + 4))) ? -1.0f : 1.0f;
                sb_acc1 += (db1 * float(v1_1[j]) * sign0_1) * xj;
                sb_acc1 += (db1 * float(v2_1[j]) * sign1_1) * xj4;
            }
        }
        acc0 += sb_acc0;
        acc1 += sb_acc1;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_tq2_0(device const uint8_t* w [[buffer(0)]],
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

    uint bytes = p.cols / 4;
    device const uint8_t* r0_w = w + (ulong)row0 * bytes;
    device const uint8_t* r1_w = has_row1 ? (w + (ulong)row1 * bytes) : r0_w;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint b = lane; b < bytes; b += 32) {
        uint8_t byte0 = r0_w[b];
        uint8_t byte1 = r1_w[b];
        float4 xb = *reinterpret_cast<device const float4*>(xb_base + b * 4);
        for (uint j = 0; j < 4; ++j) {
            int code0 = int((byte0 >> (j * 2)) & 0x03);
            int code1 = int((byte1 >> (j * 2)) & 0x03);
            float xj = xb[j];
            acc0 += float(code0 - 1) * xj;
            acc1 += float(code1 - 1) * xj;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
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

struct GraphRopeParams2 {
    uint n_heads;
    uint head_dim;
    uint pairs;
    uint pair_stride;
    uint pair_gap;
    uint rotary_dim;
    float mscale;
    uint n_seq;
};

inline void rope_elem(device const float* in, device float* out, device const uint* pos,
                      device const float* inv_freq, uint n_heads, uint head_dim, uint rotary_dim,
                      uint pair_stride, float mscale, uint gid) {
    uint d = gid % head_dim;
    uint rem = gid / head_dim;
    uint head = rem % n_heads;
    uint seq = rem / n_heads;

    device const float* in_h = in + (ulong)seq * (n_heads * head_dim) + (ulong)head * head_dim;
    device float* out_h = out + (ulong)seq * (n_heads * head_dim) + (ulong)head * head_dim;

    if (d < rotary_dim) {
        if (pair_stride == 2) {
            uint pair_idx = d / 2;
            float freq = inv_freq[pair_idx];
            float theta = float(pos[seq]) * freq;
            float cos_th = cos(theta) * mscale;
            float sin_th = sin(theta) * mscale;
            float a = in_h[pair_idx * 2];
            float b = in_h[pair_idx * 2 + 1];
            out_h[d] = (d % 2 == 0) ? (a * cos_th - b * sin_th) : (b * cos_th + a * sin_th);
        } else {
            uint half_dim = rotary_dim / 2;
            if (d < half_dim) {
                float freq = inv_freq[d];
                float theta = float(pos[seq]) * freq;
                float cos_th = cos(theta) * mscale;
                float sin_th = sin(theta) * mscale;
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

kernel void rope_general_f32(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             device const uint* pos [[buffer(2)]],
                             device const float* inv_freq [[buffer(3)]],
                             constant GraphRopeParams2& p [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n_seq * p.n_heads * p.head_dim) return;
    rope_elem(in, out, pos, inv_freq, p.n_heads, p.head_dim, p.rotary_dim, p.pair_stride, p.mscale, gid);
}

// Horizontal batch of up to 4 Rope segments sharing pos/inv_freq/head_dim/rotary_dim/pair_stride/mscale/n_seq;
// each has its own input, output and head count. Thread ranges are consecutive per segment.
struct RopeBatchParams {
    uint n_heads[4];
    uint nseg;
    uint head_dim;
    uint rotary_dim;
    uint pair_stride;
    float mscale;
    uint n_seq;
};
kernel void rope_batch_f32(device const float* in0 [[buffer(0)]], device float* out0 [[buffer(1)]],
                           device const float* in1 [[buffer(2)]], device float* out1 [[buffer(3)]],
                           device const float* in2 [[buffer(4)]], device float* out2 [[buffer(5)]],
                           device const float* in3 [[buffer(6)]], device float* out3 [[buffer(7)]],
                           device const uint* pos [[buffer(8)]],
                           device const float* inv_freq [[buffer(9)]],
                           constant RopeBatchParams& p [[buffer(10)]],
                           uint gid [[thread_position_in_grid]]) {
    device const float* ins[4] = {in0, in1, in2, in3};
    device float* outs[4] = {out0, out1, out2, out3};
    uint g = gid;
    for (uint s = 0; s < p.nseg; s++) {
        uint total = p.n_seq * p.n_heads[s] * p.head_dim;
        if (g < total) {
            rope_elem(ins[s], outs[s], pos, inv_freq, p.n_heads[s], p.head_dim, p.rotary_dim,
                      p.pair_stride, p.mscale, g);
            return;
        }
        g -= total;
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
    uint has_vis;
    uint n_splits; // > 1: this launch covers one `chunk`-key slice of the row's keys (grid z = slice)
    uint chunk;
};

inline void attn_cached_f32_impl(device const float* q,
                                 device const float* kc,
                                 device const float* vc,
                                 device float* out,
                                 constant AttnCachedParams& p,
                                 device const uint* vis,
                                 uint2 tg,
                                 uint sg,
                                 uint lane,
                                 uint nsg,
                                 threadgroup float* tg_m,
                                 threadgroup float* tg_l,
                                 threadgroup float* tg_acc) {
    const uint hd = p.head_dim;
    const uint row = ATTN_MAX_PER_LANE * 32;
    const uint head = tg.x;
    const uint q_tok = tg.y;
    if (head >= p.n_heads || q_tok >= p.m) return;

    device const float* qh = q + ((ulong)q_tok * p.n_heads + head) * hd;
    const uint kv_off = (head / p.group) * hd;
    const uint cur_pos = p.start_pos + q_tok;
    const uint len = p.has_vis ? vis[q_tok] : cur_pos + 1;

    float qr[ATTN_MAX_PER_LANE];
    float acc[ATTN_MAX_PER_LANE];
    for (uint j = 0; j < ATTN_MAX_PER_LANE; ++j) {
        uint d = lane + 32 * j;
        qr[j] = d < hd ? qh[d] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    const uint lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;
    const uint hi = len;
    for (uint t = lo + sg; t < hi; t += nsg) {
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

kernel void attn_cached_f32(device const float* q [[buffer(0)]],
                            device const float* kc [[buffer(1)]],
                            device const float* vc [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AttnCachedParams& p [[buffer(4)]],
                            device const uint* vis [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * ATTN_MAX_PER_LANE * 32];
    attn_cached_f32_impl(q, kc, vc, out, p, vis, tg, sg, lane, nsg, tg_m, tg_l, tg_acc);
}

kernel void flash_attn_cached_f32(device const float* q [[buffer(0)]],
                                  device const float* kc [[buffer(1)]],
                                  device const float* vc [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  constant AttnCachedParams& p [[buffer(4)]],
                                  device const uint* vis [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * ATTN_MAX_PER_LANE * 32];
    attn_cached_f32_impl(q, kc, vc, out, p, vis, tg, sg, lane, nsg, tg_m, tg_l, tg_acc);
}

kernel void attn_cached_merge_f32(device const float* part [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant AttnCachedParams& p [[buffer(2)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint2 ntg [[threads_per_threadgroup]]) {
    const uint hd = p.head_dim, head = tg.x, q_tok = tg.y;
    if (head >= p.n_heads || q_tok >= p.m) return;
    device const float* base = part + ((ulong)q_tok * p.n_heads + head) * p.n_splits * (hd + 2);
    float gm = -FLT_MAX;
    for (uint s = 0; s < p.n_splits; ++s) gm = max(gm, base[s * (hd + 2)]);
    float gl = 0.0f;
    for (uint s = 0; s < p.n_splits; ++s) gl += base[s * (hd + 2) + 1] * exp(base[s * (hd + 2)] - gm);
    for (uint d = tid; d < hd; d += ntg.x) {
        float o = 0.0f;
        for (uint s = 0; s < p.n_splits; ++s) o += base[s * (hd + 2) + 2 + d] * exp(base[s * (hd + 2)] - gm);
        out[((ulong)q_tok * p.n_heads + head) * hd + d] = o / gl;
    }
}

// Decode attention for grouped KV heads: grid (kv_heads * ceil(group / ATTN_HPT), 1, slices). A threadgroup owns up to
// ATTN_HPT query heads that share one KV head and one `p.chunk`-key slice, so each key and value is loaded
// once for all of them (the per-head kernel above re-reads them `group` times). Partials have the layout of
// `attn_cached_split_f32`; `attn_cached_merge_f32` combines them.
constant uint ATTN_HPT = 4;
template <uint NJ>
kernel void attn_cached_gqa_split_t(device const float* q [[buffer(0)]],
                                      device const float* kc [[buffer(1)]],
                                      device const float* vc [[buffer(2)]],
                                      device float* part [[buffer(3)]],
                                      constant AttnCachedParams& p [[buffer(4)]],
                                      uint3 tg3 [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tq[ATTN_HPT * NJ * 32];
    threadgroup float tg_m[ATTN_HPT * ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_HPT * ATTN_MAX_SG];
    threadgroup float tg_red[ATTN_MAX_SG * NJ * 32];
    const uint hd = p.head_dim;
    const uint row = NJ * 32;
    const uint hgroups = (p.group + ATTN_HPT - 1) / ATTN_HPT;
    const uint kvh = tg3.x / hgroups;
    const uint head0 = kvh * p.group + (tg3.x % hgroups) * ATTN_HPT;
    const uint nh = min(ATTN_HPT, kvh * p.group + p.group - head0);
    const uint q_tok = 0;
    const uint split = tg3.z;
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < nh * row; i += nsg * 32) {
        const uint h = i / row, d = i % row;
        tq[i] = d < hd ? q[((ulong)q_tok * p.n_heads + head0 + h) * hd + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_off = kvh * hd;
    const uint cur_pos = p.start_pos + q_tok;
    const uint len = cur_pos + 1;
    uint lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;
    lo += split * p.chunk;
    const uint hi = min(len, lo + p.chunk);

    float acc[ATTN_HPT][NJ];
    float m[ATTN_HPT], l[ATTN_HPT];
    for (uint h = 0; h < ATTN_HPT; ++h) {
        m[h] = -FLT_MAX;
        l[h] = 0.0f;
        for (uint j = 0; j < NJ; ++j) acc[h][j] = 0.0f;
    }
    for (uint t = lo + sg; t < hi; t += nsg) {
        device const float* k = kc + (ulong)t * p.stride + kv_off;
        device const float* v = vc + (ulong)t * p.stride + kv_off;
        float kr[NJ];
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            kr[j] = d < hd ? k[d] : 0.0f;
        }
        float vr[NJ];
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            vr[j] = d < hd ? v[d] : 0.0f;
        }
        for (uint h = 0; h < ATTN_HPT; ++h) {
            if (h >= nh) break;
            float s = 0.0f;
            for (uint j = 0; j < NJ; ++j) s += tq[h * row + lane + 32 * j] * kr[j];
            s = simd_sum(s) * p.scale;
            if (p.softcap > 0.0f) s = tanh(s / p.softcap) * p.softcap;
            const float m_new = max(m[h], s);
            const float corr = exp(m[h] - m_new);
            const float e = exp(s - m_new);
            l[h] = l[h] * corr + e;
            for (uint j = 0; j < NJ; ++j) acc[h][j] = acc[h][j] * corr + e * vr[j];
            m[h] = m_new;
        }
    }
    for (uint h = 0; h < ATTN_HPT; ++h) {
        if (h >= nh) break;
        if (lane == 0) {
            tg_m[h * ATTN_MAX_SG + sg] = m[h];
            tg_l[h * ATTN_MAX_SG + sg] = l[h];
        }
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            if (d < hd) tg_red[sg * row + d] = acc[h][j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gm = -FLT_MAX;
        for (uint i = 0; i < nsg; ++i) gm = max(gm, tg_m[h * ATTN_MAX_SG + i]);
        device float* out = part + (((ulong)q_tok * p.n_heads + head0 + h) * p.n_splits + split) * (hd + 2);
        if (tid == 0) {
            float gl = 0.0f;
            for (uint i = 0; i < nsg; ++i) gl += tg_l[h * ATTN_MAX_SG + i] * exp(tg_m[h * ATTN_MAX_SG + i] - gm);
            out[0] = gm;
            out[1] = gl;
        }
        for (uint d = tid; d < hd; d += nsg * 32) {
            float o = 0.0f;
            for (uint i = 0; i < nsg; ++i) o += tg_red[i * row + d] * exp(tg_m[h * ATTN_MAX_SG + i] - gm);
            out[2 + d] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

typedef decltype(attn_cached_gqa_split_t<1>) attn_cached_gqa_split_k;
template [[host_name("attn_cached_gqa_split_f32_1")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<1>;
template [[host_name("attn_cached_gqa_split_f32_2")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<2>;
template [[host_name("attn_cached_gqa_split_f32_4")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<4>;
template [[host_name("attn_cached_gqa_split_f32_8")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<8>;
template [[host_name("attn_cached_gqa_split_f32_16")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<16>;

struct PagedAttnCachedParams {
    uint n_heads;
    uint head_dim;
    uint stride;
    uint group;
    float scale;
    uint window;
    float softcap;
    uint start_pos;
    uint m;
    uint tokens_per_block;
    uint max_blocks_per_seq;
};

kernel void paged_attn_cached_f32(device const float* q [[buffer(0)]],
                                  device const float* kc_pool [[buffer(1)]],
                                  device const float* vc_pool [[buffer(2)]],
                                  device const uint* block_table [[buffer(3)]],
                                  device float* out [[buffer(4)]],
                                  constant PagedAttnCachedParams& p [[buffer(5)]],
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
    uint lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;
    for (uint t = lo + sg; t < len; t += nsg) {
        uint blk_id = block_table[(ulong)q_tok * p.max_blocks_per_seq + (t / p.tokens_per_block)];
        uint blk_offset = t % p.tokens_per_block;
        ulong kv_idx = ((ulong)blk_id * p.tokens_per_block + blk_offset) * p.stride + kv_off;
        device const float* k = kc_pool + kv_idx;
        device const float* v = vc_pool + kv_idx;
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

// Gated delta rule, one threadgroup per (32 value columns, head): 4 simdgroups split the key rows of the
// state (`i = group + 4r`), lanes are the columns (coalesced state access), q/k are normalised once into
// threadgroup memory and the per-column sums are combined through it. Requires key_dim <= DELTA_MAX_KD.
constant uint DELTA_MAX_KD = 512;
kernel void delta_rule_f32(device const float* q [[buffer(0)]],
                           device const float* k [[buffer(1)]],
                           device const float* v [[buffer(2)]],
                           device const float* beta [[buffer(3)]],
                           device const float* log_decay [[buffer(4)]],
                           device float* state [[buffer(5)]],
                           device float* out [[buffer(6)]],
                           constant DeltaRuleParams& p [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float ks[DELTA_MAX_KD];
    threadgroup float qs[DELTA_MAX_KD];
    threadgroup float red[8];
    threadgroup float part[128];
    const uint lane = tid & 31, g = tid >> 5;
    const uint j = tg.x * 32 + lane;
    const uint h = tg.y;
    const bool live = j < p.val_dim;
    const uint kh = p.key_heads, kd = p.key_dim, vd = p.val_dim, vh = p.val_heads;
    const uint k_h = h / p.rep;

    device float* ssm_state = state + (ulong)h * kd * vd;

    for (uint t = 0; t < p.n; ++t) {
        device const float* q_in = q + ((ulong)t * kh + k_h) * kd;
        device const float* k_in = k + ((ulong)t * kh + k_h) * kd;
        device const float* v_t = v + ((ulong)t * vh + h) * vd;
        const float b = beta[(ulong)t * vh + h];
        const float decay = exp(log_decay[(ulong)t * vh + h]);

        float qn = 0.0f, kn = 0.0f;
        for (uint i = tid; i < kd; i += 128) {
            qn += q_in[i] * q_in[i];
            kn += k_in[i] * k_in[i];
        }
        qn = simd_sum(qn);
        kn = simd_sum(kn);
        if (lane == 0) {
            red[g] = qn;
            red[4 + g] = kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float q_norm = max(sqrt(red[0] + red[1] + red[2] + red[3]), 1e-12f);
        const float k_norm = max(sqrt(red[4] + red[5] + red[6] + red[7]), 1e-12f);
        const float q_scale = (1.0f / sqrt(float(kd))) / q_norm;
        const float k_scale = 1.0f / k_norm;
        for (uint i = tid; i < kd; i += 128) {
            ks[i] = k_in[i] * k_scale;
            qs[i] = q_in[i] * q_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float kv = 0.0f;
        if (live) {
            for (uint i = g; i < kd; i += 4) kv += ssm_state[i * vd + j] * ks[i];
        }
        part[tid] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_j = part[lane] + part[32 + lane] + part[64 + lane] + part[96 + lane];
        const float delta_j = live ? b * (v_t[j] - decay * kv_j) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float o = 0.0f;
        if (live) {
            for (uint i = g; i < kd; i += 4) {
                const float s = decay * ssm_state[i * vd + j] + ks[i] * delta_j;
                ssm_state[i * vd + j] = s;
                o += s * qs[i];
            }
        }
        part[tid] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0 && live) {
            out[((ulong)t * vh + h) * vd + j] = part[lane] + part[32 + lane] + part[64 + lane] + part[96 + lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct MulParams {
    uint total;
    uint a_len;
    uint b_len;
    uint row_broadcast;
    uint d;
};

constant const float nvfp4_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float e4m3_to_float(uchar x) {
    uchar m = x & 0x7F;
    if (m == 0x7F) return NAN;
    float mag;
    if (m < 8) {
        mag = float(m) / 512.0f;
    } else {
        uint bits = (((uint(m >> 3) + 120u) << 23) | ((uint(m & 7) << 20)));
        mag = as_type<float>(bits);
    }
    return (x & 0x80) ? -mag : mag;
}

kernel void gemv_nvfp4(device const uchar* wq [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       constant PackedGemvParams& p [[buffer(3)]],
                       device const float* bias [[buffer(4)]],
                       device const float* res [[buffer(5)]],
                       device const uchar* scales_e4m3 [[buffer(6)]],
                       constant float& global_scale [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 16;
    uint byte_stride = p.cols / 2;
    device const uchar* r0_bytes = wq + (ulong)row0 * byte_stride;
    device const uchar* r1_bytes = has_row1 ? (wq + (ulong)row1 * byte_stride) : r0_bytes;
    device const uchar* r0_scales = scales_e4m3 + (ulong)row0 * n_blocks;
    device const uchar* r1_scales = has_row1 ? (scales_e4m3 + (ulong)row1 * n_blocks) : r0_scales;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint blk = lane; blk < n_blocks; blk += 32) {
        float s0 = e4m3_to_float(r0_scales[blk]) * global_scale;
        float s1 = e4m3_to_float(r1_scales[blk]) * global_scale;
        device const uchar* b0_bytes = r0_bytes + blk * 8;
        device const uchar* b1_bytes = r1_bytes + blk * 8;
        device const float* xb = xb_base + blk * 16;

        float b_acc0 = 0.0f;
        float b_acc1 = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            uchar byte0 = b0_bytes[i];
            uchar byte1 = b1_bytes[i];
            float x0 = xb[2 * i];
            float x1 = xb[2 * i + 1];

            float v0_0 = nvfp4_e2m1_lut[byte0 & 0x0F] * s0;
            float v1_0 = nvfp4_e2m1_lut[byte0 >> 4] * s0;
            b_acc0 += v0_0 * x0 + v1_0 * x1;

            float v0_1 = nvfp4_e2m1_lut[byte1 & 0x0F] * s1;
            float v1_1 = nvfp4_e2m1_lut[byte1 >> 4] * s1;
            b_acc1 += v0_1 * x0 + v1_1 * x1;
        }
        acc0 += b_acc0;
        acc1 += b_acc1;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_jesse_lut(
    device const char* lut_tables [[buffer(0)]],
    device const float* hash_planes [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& address_bits [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= out_features) return;
    uint num_luts = (in_features + address_bits - 1) / address_bits;
    uint lut_size = 1u << address_bits;
    ulong lut_out_offset = (ulong)gid * num_luts * lut_size;

    int accum = 0;
    for (uint lut_idx = 0; lut_idx < num_luts; ++lut_idx) {
        uint addr = 0;
        for (uint b = 0; b < address_bits; ++b) {
            uint feat_idx = (lut_idx * address_bits + b) % in_features;
            float thresh = hash_planes[lut_idx * address_bits + b];
            if (x[feat_idx] > thresh) {
                addr |= (1u << b);
            }
        }
        accum += (int)lut_tables[lut_out_offset + (ulong)lut_idx * lut_size + addr];
    }
    y[gid] = float(accum);
}

struct Conv2dParams {
    uint in_channels;
    uint in_h;
    uint in_w;
    uint out_channels;
    uint kernel_size;
    uint stride;
    uint out_h;
    uint out_w;
    uint has_bias;
};

