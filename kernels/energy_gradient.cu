// A02 — Fused energy + gradient kernel for Gaussian splat EBM.
// Target: RTX 3090 (sm_86), compiled at runtime via cudarc+NVRTC (no nvcc).
//
// Math (matches src/ebm/energy_api.rs energy_gradient / energy):
//   w_i  = alpha_i * exp(-kappa_i * ||x - mu_i||^2)
//   Z    = sum_i w_i
//   E    = -log(Z)                                   (10.0 if Z < 1e-10)
//   G_d  = sum_i 2*kappa_i*w_i*(x_d - mu_i_d) / Z    (0 if Z <= 1e-10)
//
// FUSION TRADE-OFF (documented per task):
//   The gradient REQUIRES Z = sum w_i, which is the entire forward pass. A
//   separate forward kernel would re-read all of mu [N,D] from VRAM a second
//   time; this kernel is bandwidth-bound (measured, see report), so fusion
//   saves ~50% of the dominant traffic at the cost of one __logf and B floats
//   of extra output. Fused is strictly better here; a standalone forward
//   kernel only makes sense for callers that need E without G (e.g. A01's
//   use case), and both can coexist on the same math.
//
// Design (one block of 256 threads = 8 warps per query):
//   - query x cached in shared memory (pattern from distance.cu)
//   - each lane OWNS components d = lane + 32*c, c < COMPS -> the whole warp
//     reads each mu row coalesced; G accumulators live in REGISTERS
//   - warp-strided splat loop (warp w takes i = w, w+8, ...) -> consecutive
//     warps stream consecutive mu rows, full 128B transactions
//   - r2 via warp butterfly allreduce (__shfl_xor, 5 steps, no shared round-trip)
//   - only lane 0 accumulates Z per warp (r2/w are already replicated across
//     the warp by the butterfly)
//   - final cross-warp combine through shared memory, strided write (any D)
//
// Template COMPS = ceil(D/32), compile-time so g[]/diff[] are fixed-size
// register arrays and the d<D guards are loop-invariant across the splat
// loop (compiler hoists them). Entry points c1/c2/c4/c8/c16 cover D = 1..512;
// the Rust wrapper picks the name by D.
//
// Numerics: __expf/__logf (SFU, --use_fast_math) vs libm expf/lnf on CPU,
// both accumulating in fp32. Divergence vs CPU documented in the report;
// task tolerance for G is 1e-3 (gradient sums more terms than the forward).
//
// No UB: all global accesses guarded (i < N by loop condition, d < D checked,
// b < B guaranteed by gridDim), __syncthreads between shared write/read phases.

#include <cuda_runtime.h>

#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)  // 256, matches distance.cu pattern

template <int COMPS>
__device__ __forceinline__ void energy_gradient_fused_impl(
    const float* __restrict__ x,       // [B, D] queries
    const float* __restrict__ mu,      // [N, D] splat centers
    const float* __restrict__ alpha,   // [N]
    const float* __restrict__ kappa,   // [N]
    float* __restrict__ out_E,         // [B]      energies (fused forward)
    float* __restrict__ out_G,         // [B, D]   gradients
    int B, int N, int D
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    extern __shared__ float s_data[];
    float* s_x = s_data;                        // [D] query cache
    float* s_g = s_data + D;                    // [WARPS_PER_BLOCK * D]
    __shared__ float s_z[WARPS_PER_BLOCK];      // per-warp Z partials

    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        s_x[d] = x[b * D + d];
    }
    __syncthreads();

    float g[COMPS];   // this lane's gradient accumulators (registers)
    #pragma unroll
    for (int c = 0; c < COMPS; c++) g[c] = 0.0f;

    float z_partial = 0.0f;

    // Splat stream: warp w takes splats w, w+8, w+16, ...
    for (int i = warp; i < N; i += WARPS_PER_BLOCK) {
        const float a = alpha[i];   // uniform across warp -> 1 transaction
        const float k = kappa[i];

        float partial = 0.0f;
        float diff[COMPS];
        #pragma unroll
        for (int c = 0; c < COMPS; c++) {
            const int d = lane + 32 * c;
            if (d < D) {
                const float df = s_x[d] - mu[i * D + d];
                diff[c] = df;
                partial += df * df;
            }
        }

        // warp allreduce r2 (butterfly over 32 lanes)
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        }

        const float w = a * __expf(-k * partial);   // partial == r2 after reduce
        const float c_i = 2.0f * k * w;
        #pragma unroll
        for (int c = 0; c < COMPS; c++) {
            const int d = lane + 32 * c;
            if (d < D) g[c] += c_i * diff[c];
        }
        if (lane == 0) z_partial += w;
    }

    // Cross-warp combine via shared memory
    #pragma unroll
    for (int c = 0; c < COMPS; c++) {
        const int d = lane + 32 * c;
        if (d < D) s_g[warp * D + d] = g[c];
    }
    if (lane == 0) s_z[warp] = z_partial;
    __syncthreads();

    // Total Z computed by thread 0, published through s_z[0]
    if (threadIdx.x == 0) {
        float Z = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) Z += s_z[w];
        s_z[0] = Z;
        out_E[b] = (Z > 1e-10f) ? -__logf(Z) : 10.0f;   // CPU guard: E=10
    }
    __syncthreads();

    const float Z = s_z[0];
    const float inv = (Z > 1e-10f) ? (1.0f / Z) : 0.0f; // CPU guard: no normalize

    // Strided write so any D works even when D > BLOCK_SIZE
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) sum += s_g[w * D + d];
        out_G[b * D + d] = sum * inv;
    }
}

// extern "C" entry points selected by ceil(D/32); the Rust wrapper picks
// the name from D at runtime. __launch_bounds__ goes on the entry points.
#define EG_ENTRY(suffix, comps)                                                \
extern "C" __launch_bounds__(BLOCK_SIZE) __global__ void energy_gradient_fused_##suffix( \
    const float* x, const float* mu, const float* alpha, const float* kappa,   \
    float* out_E, float* out_G, int B, int N, int D)                           \
{ energy_gradient_fused_impl<comps>(x, mu, alpha, kappa, out_E, out_G, B, N, D); }

EG_ENTRY(c1, 1)
EG_ENTRY(c2, 2)
EG_ENTRY(c4, 4)
EG_ENTRY(c8, 8)
EG_ENTRY(c16, 16)
