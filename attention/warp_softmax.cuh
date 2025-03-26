#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// --- Warp-level reductions ---
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// --- Warp-thread parallel softmax (for len <= 32) ---
__device__ inline float warp_softmax_parallel(float logit_i, float temperature = 1.0f) {
    float scaled = logit_i / temperature;
    float max_val = warp_reduce_max(scaled);
    float shifted = scaled - max_val;
    float exp_val = __expf(shifted);
    float sum_exp = warp_reduce_sum(exp_val);
    return exp_val / (sum_exp + 1e-6f);
}

// --- Legacy warp softmax fallback (serial for-loop) ---
__device__ inline void warp_softmax(float* scores, int len, float* softmax_out) {
    float maxval = -1e9f;
    for (int i = 0; i < len; ++i)
        maxval = fmaxf(maxval, scores[i]);

    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        softmax_out[i] = __expf(scores[i] - maxval);
        sum += softmax_out[i];
    }

    float inv = 1.0f / (sum + 1e-6f);
    for (int i = 0; i < len; ++i)
        softmax_out[i] *= inv;
}

// --- Block-wide parallel softmax (tile_size > 32) ---
__device__ inline void block_softmax(const float* __restrict__ logits,
                                     float* __restrict__ probs,
                                     int len,
                                     float temperature = 1.0f) {
    extern __shared__ float smem[];
    float* logits_shifted = smem;
    float* probs_smem = smem + len;

    int tid = threadIdx.x;

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < len; i += blockDim.x) {
        float val = logits[i] / temperature;
        logits_shifted[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();

    float max_val = warp_reduce_max(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
        float shifted = logits_shifted[i] - max_val;
        float exp_val = __expf(shifted);
        probs_smem[i] = exp_val;
        local_sum += exp_val;
    }
    __syncthreads();

    float sum_exp = warp_reduce_sum(local_sum);
    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    __syncthreads();

    for (int i = tid; i < len; i += blockDim.x)
        probs[i] = probs_smem[i] * inv_sum;
}

// --- Tile-fusion softmax kernel (for tile_size > 128), with register blocking ---
__device__ inline void block_softmax_fused(const float* __restrict__ logits,
                                           float* __restrict__ probs,
                                           int tile_size,
                                           float temperature = 1.0f,
                                           int tile_offset = 0) {
    // Each block handles a tile (e.g. one beam), offset by tile_offset
    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Register buffer: each thread handles N values (can tune N = 2/4)
    const int VEC = 2;
    float local_logits[VEC];
    float local_max[VEC];

    // Step 1: Load logits into registers + find local max
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int idx = tile_offset + tid * VEC + i;
        float val = (idx < tile_offset + tile_size) ? logits[idx] / temperature : -CUDART_INF_F;
        local_logits[i] = val;
        local_max[i] = val;
    }

    // Reduce to find max across all threads
    float thread_max = fmaxf(local_max[0], local_max[1]);
    float max_val = warp_reduce_max(thread_max);
    __syncthreads();

    // Step 2: compute exp(logit - max), accumulate local sum
    float local_exp[VEC];
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        float shifted = local_logits[i] - max_val;
        float exp_val = __expf(shifted);
        local_exp[i] = exp_val;
        local_sum += exp_val;
    }

    // Reduce to get global sum
    float sum_exp = warp_reduce_sum(local_sum);
    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    __syncthreads();

    // Step 3: Normalize
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int idx = tile_offset + tid * VEC + i;
        if (idx < tile_offset + tile_size)
            probs[idx] = local_exp[i] * inv_sum;
    }
}


// --- Unified smart dispatcher ---
__device__ inline void smart_softmax(float* logits,
                                     float* probs,
                                     int len,
                                     float temperature = 1.0f) {
    if (len <= 32) {
        int tid = threadIdx.x % 32;
        probs[tid] = warp_softmax_parallel(logits[tid], temperature);
    } else if (len <= 128) {
        block_softmax(logits, probs, len, temperature);
    } else {
        block_softmax_fused(logits, probs, len, temperature); // fused path
    }
}
