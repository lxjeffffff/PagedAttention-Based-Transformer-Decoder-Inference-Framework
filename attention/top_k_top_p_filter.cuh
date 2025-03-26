#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32

// Block-level reduction helpers
__device__ inline float block_reduce_max(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    float maxval = val;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, offset));

    if (lane == 0) shared[wid] = maxval;
    __syncthreads();

    if (wid == 0) {
        maxval = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, offset));
    }

    return __shfl_sync(0xffffffff, maxval, 0);
}

__device__ inline float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    float sumval = val;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        sumval += __shfl_down_sync(0xffffffff, sumval, offset);

    if (lane == 0) shared[wid] = sumval;
    __syncthreads();

    if (wid == 0) {
        sumval = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            sumval += __shfl_down_sync(0xffffffff, sumval, offset);
    }

    return __shfl_sync(0xffffffff, sumval, 0);
}

// Apply top-k/top-p filtering and sampling across block
__device__ inline void top_k_top_p_filter(
    float* logits,             // [vocab]
    int vocab_size,
    int top_k,
    float top_p,
    float temperature,
    int* top_indices,          // [top_k]
    float* top_logits,         // [top_k]
    int& sampled_token         // output: selected token
) {
    int tid = threadIdx.x;

    // Step 1: temperature
    for (int i = tid; i < vocab_size; i += blockDim.x)
        logits[i] /= temperature;
    __syncthreads();

    // Step 2: max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);
    float max_val = block_reduce_max(local_max);
    __syncthreads();

    // Step 3: softmax
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] = __expf(logits[i] - max_val);
        local_sum += logits[i];
    }
    float sum_val = block_reduce_sum(local_sum);
    __syncthreads();

    for (int i = tid; i < vocab_size; i += blockDim.x)
        logits[i] /= (sum_val + 1e-6f);
    __syncthreads();

    // Step 4: Top-k selection (serial within warp 0 only)
    if (tid == 0) {
        for (int k = 0; k < top_k; ++k) {
            float best_val = -1.0f;
            int best_idx = -1;
            for (int i = 0; i < vocab_size; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_idx = i;
                }
            }
            top_indices[k] = best_idx;
            top_logits[k] = best_val;
            logits[best_idx] = -1e9f;  // mask it out
        }

        // Top-p (optional - skip here)
        sampled_token = top_indices[rand() % top_k];
    }
}
