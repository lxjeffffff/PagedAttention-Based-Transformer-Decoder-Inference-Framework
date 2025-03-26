#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

// ---------------------------------------------
// Index utilities
// ---------------------------------------------

__device__ inline int get_qkv_index(int batch_id, int head_id, int token_idx, int head_dim, int seq_len, int num_heads) {
    // Returns flattened index for Q/K/V layout: [B, H, T, D]
    return ((batch_id * num_heads + head_id) * seq_len + token_idx) * head_dim;
}

// ---------------------------------------------
// Rotary Embedding (float version)
// ---------------------------------------------

__device__ inline void apply_rotary_embedding(float* q, float* k, const float* rotary_emb, int head_dim, int token_idx, bool apply_on_k = true) {
    for (int d = 0; d < head_dim; d += 2) {
        float cos = rotary_emb[token_idx * head_dim + d];
        float sin = rotary_emb[token_idx * head_dim + d + 1];

        float q0 = q[d], q1 = q[d + 1];
        q[d]     = q0 * cos - q1 * sin;
        q[d + 1] = q0 * sin + q1 * cos;

        if (apply_on_k) {
            float k0 = k[d], k1 = k[d + 1];
            k[d]     = k0 * cos - k1 * sin;
            k[d + 1] = k0 * sin + k1 * cos;
        }
    }
}

// ---------------------------------------------
// Rotary Embedding (half2 version)
// ---------------------------------------------

__device__ inline void apply_rotary_embedding_half2(__half2* q, __half2* k, const half2* rotary_emb, int head_dim_half2, int token_idx, bool apply_on_k = true) {
    for (int i = 0; i < head_dim_half2; ++i) {
        half2 cos_sin = rotary_emb[token_idx * head_dim_half2 + i];
        float cos = __half2float(__low2half(cos_sin));
        float sin = __half2float(__high2half(cos_sin));

        float2 q_val = __half22float2(q[i]);
        float q0 = q_val.x;
        float q1 = q_val.y;

        float q_rot0 = q0 * cos - q1 * sin;
        float q_rot1 = q0 * sin + q1 * cos;
        q[i] = __floats2half2_rn(q_rot0, q_rot1);

        if (apply_on_k) {
            float2 k_val = __half22float2(k[i]);
            float k0 = k_val.x;
            float k1 = k_val.y;
            float k_rot0 = k0 * cos - k1 * sin;
            float k_rot1 = k0 * sin + k1 * cos;
            k[i] = __floats2half2_rn(k_rot0, k_rot1);
        }
    }
}

// ---------------------------------------------
// Causal Mask
// ---------------------------------------------

template <typename T>
__device__ inline T apply_causal_mask(T score, int query_pos, int key_pos, bool causal) {
    if (causal && key_pos > query_pos) {
        if constexpr (std::is_same<T, float>::value)
            return -CUDART_INF_F;
        else if constexpr (std::is_same<T, __half>::value)
            return __float2half(-1e4f);
    }
    return score;
}

// ---------------------------------------------
// Safe exp / sqrt (template version)
// ---------------------------------------------

template <typename T>
__device__ inline T safe_exp(T x);

template <>
__device__ inline float safe_exp<float>(float x) {
    return __expf(fminf(x, 80.0f));
}

template <>
__device__ inline __half safe_exp<__half>(__half x) {
    return hexp(__hmin(x, __float2half(80.0f)));
}

template <typename T>
__device__ inline T fast_sqrt(T x);

template <>
__device__ inline float fast_sqrt<float>(float x) {
#if __CUDA_ARCH__ >= 300
    return rsqrtf(x) * x;
#else
    return sqrtf(x);
#endif
}

template <>
__device__ inline __half fast_sqrt<__half>(__half x) {
    return hsqrt(x);  // uses hardware instruction
}
