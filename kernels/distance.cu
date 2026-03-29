// M2M Vector Search — CUDA Kernels
// Optimized for sm_86 (RTX 3090), CUDA 12.4
// All kernels assume row-major [N, D] layout

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// L2 Distance: single query vs N vectors
// ============================================================================
// Each block processes a tile of vectors. Query is cached in shared memory.
// Uses float4 vectorized loads for D divisible by 4.

extern "C" __launch_bounds__(256)
__global__ void l2_distance_kernel(
    const float* __restrict__ query,    // [D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [N]
    int N,
    int D
) {
    // Cache query in shared memory (up to 2048 floats = 8KB)
    extern __shared__ float s_query[];
    
    // Cooperative load of query into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_query[j] = query[j];
    }
    __syncthreads();
    
    // Each thread computes L2 distance for one vector
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const float* vec = dataset + i * D;
    float sum = 0.0f;
    
    // Vectorized path for D divisible by 4
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(s_query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 q = query4[j];
            float dx = q.x - v.x;
            float dy = q.y - v.y;
            float dz = q.z - v.z;
            float dw = q.w - v.w;
            sum += dx*dx + dy*dy + dz*dz + dw*dw;
        }
    } else {
        // Scalar fallback
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            float diff = s_query[j] - vec[j];
            sum += diff * diff;
        }
    }
    
    output[i] = sum;
}

// ============================================================================
// Batch L2 Distance: Q queries vs N vectors
// ============================================================================
// 2D grid: x = vector index (blockDim.x per block), y = query index
// Each thread computes one (query, vector) distance pair

extern "C" __launch_bounds__(256)
__global__ void batch_l2_distance_kernel(
    const float* __restrict__ queries,  // [Q, D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [Q, N]
    int Q,
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // vector index
    int q = blockIdx.y;                               // query index
    
    if (i >= N || q >= Q) return;
    
    const float* query = queries + q * D;
    const float* vec = dataset + i * D;
    
    float sum = 0.0f;
    
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 qu = query4[j];
            float dx = qu.x - v.x;
            float dy = qu.y - v.y;
            float dz = qu.z - v.z;
            float dw = qu.w - v.w;
            sum += dx*dx + dy*dy + dz*dz + dw*dw;
        }
    } else {
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            float diff = query[j] - vec[j];
            sum += diff * diff;
        }
    }
    
    output[q * N + i] = sum;
}

// ============================================================================
// Cosine Distance: single query vs N vectors
// ============================================================================
// Computes 1 - (q . v_i) / (||q|| * ||v_i||)
// Uses shared memory reduction for query norm.

extern "C" __launch_bounds__(256)
__global__ void cosine_distance_kernel(
    const float* __restrict__ query,    // [D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [N]
    int N,
    int D
) {
    extern __shared__ float s_buf[];  // [D + 256] = query + reduction workspace
    
    float* s_query = s_buf;
    float* s_reduce = s_buf + D;
    
    // Cooperative load query into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_query[j] = query[j];
    }
    __syncthreads();
    
    // Reduction: compute sum of query^2 across all threads
    float q_part = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        q_part += s_query[j] * s_query[j];
    }
    s_reduce[threadIdx.x] = q_part;
    __syncthreads();
    
    // Tree reduction for query norm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float q_norm = sqrtf(max(s_reduce[0], 1e-20f));
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const float* vec = dataset + i * D;
    float dot = 0.0f;
    float v_norm_sq = 0.0f;
    
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(s_query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 q = query4[j];
            dot += q.x*v.x + q.y*v.y + q.z*v.z + q.w*v.w;
            v_norm_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            dot += s_query[j] * vec[j];
            v_norm_sq += vec[j] * vec[j];
        }
    }
    
    float denom = q_norm * sqrtf(max(v_norm_sq, 1e-20f));
    output[i] = 1.0f - dot / max(denom, 1e-10f);
}

// ============================================================================
// Batch Cosine Distance: Q queries vs N vectors
// ============================================================================

extern "C" __launch_bounds__(256)
__global__ void batch_cosine_distance_kernel(
    const float* __restrict__ queries,  // [Q, D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [Q, N]
    int Q,
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y;
    
    if (i >= N || q >= Q) return;
    
    const float* query = queries + q * D;
    const float* vec = dataset + i * D;
    
    float dot = 0.0f;
    float q_norm_sq = 0.0f;
    float v_norm_sq = 0.0f;
    
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 qu = query4[j];
            dot += qu.x*v.x + qu.y*v.y + qu.z*v.z + qu.w*v.w;
            q_norm_sq += qu.x*qu.x + qu.y*qu.y + qu.z*qu.z + qu.w*qu.w;
            v_norm_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            dot += query[j] * vec[j];
            q_norm_sq += query[j] * query[j];
            v_norm_sq += vec[j] * vec[j];
        }
    }
    
    float denom = sqrtf(max(q_norm_sq, 1e-20f)) * sqrtf(max(v_norm_sq, 1e-20f));
    output[q * N + i] = 1.0f - dot / max(denom, 1e-10f);
}
