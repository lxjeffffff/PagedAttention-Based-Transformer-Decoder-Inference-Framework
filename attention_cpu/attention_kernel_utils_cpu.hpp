#pragma once

#include "vec_cpu.hpp"
#include <cstdint>
#include <cmath>
#include <cassert>

using cpuvec::Vec;

inline int get_cpu_qkv_index(int batch_id, int head_id, int token_idx, int head_dim, int seq_len, int num_heads) {
    return ((batch_id * num_heads + head_id) * seq_len + token_idx) * head_dim;
}

// === Float rotary embedding ===
inline void apply_rotary_qk(
    float* q, float* k,
    const float* rotary_emb,
    int head_dim,
    int token_idx)
{
    for (int d = 0; d < head_dim; d += 2) {
        float cos = rotary_emb[token_idx * head_dim + d];
        float sin = rotary_emb[token_idx * head_dim + d + 1];

        float q0 = q[d], q1 = q[d + 1];
        float k0 = k[d], k1 = k[d + 1];

        q[d]     = q0 * cos - q1 * sin;
        q[d + 1] = q0 * sin + q1 * cos;

        k[d]     = k0 * cos - k1 * sin;
        k[d + 1] = k0 * sin + k1 * cos;
    }
}

inline void apply_rotary_q_only(
    float* q,
    const float* rotary_emb,
    int head_dim,
    int token_idx)
{
    for (int d = 0; d < head_dim; d += 2) {
        float cos = rotary_emb[token_idx * head_dim + d];
        float sin = rotary_emb[token_idx * head_dim + d + 1];

        float q0 = q[d], q1 = q[d + 1];
        q[d]     = q0 * cos - q1 * sin;
        q[d + 1] = q0 * sin + q1 * cos;
    }
}

// === INT8 rotary embedding (with dequant) ===
inline void apply_rotary_qk_int8(
    int8_t* q, int8_t* k,
    const float* rotary_emb,
    int head_dim,
    int token_idx,
    float scale_q,
    float scale_k)
{
    for (int d = 0; d < head_dim; d += 2) {
        float cos = rotary_emb[token_idx * head_dim + d];
        float sin = rotary_emb[token_idx * head_dim + d + 1];

        float q0 = static_cast<float>(q[d]) * scale_q;
        float q1 = static_cast<float>(q[d + 1]) * scale_q;
        float k0 = static_cast<float>(k[d]) * scale_k;
        float k1 = static_cast<float>(k[d + 1]) * scale_k;

        q[d]     = static_cast<int8_t>(q0 * cos - q1 * sin);
        q[d + 1] = static_cast<int8_t>(q0 * sin + q1 * cos);

        k[d]     = static_cast<int8_t>(k0 * cos - k1 * sin);
        k[d + 1] = static_cast<int8_t>(k0 * sin + k1 * cos);
    }
}

// === Vectorized rotary for batch [B, H, D] ===
template<typename T>
inline void apply_rotary_batch_qk(
    T* q, T* k,
    const float* rotary_emb,
    int B, int H, int D,
    int token_idx)
{
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            T* q_ptr = q + (b * H + h) * D;
            T* k_ptr = k + (b * H + h) * D;
            apply_rotary_qk(q_ptr, k_ptr, rotary_emb, D, token_idx);
        }
    }
}

template<typename T>
inline void apply_rotary_batch_q_only(
    T* q,
    const float* rotary_emb,
    int B, int H, int D,
    int token_idx)
{
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            T* q_ptr = q + (b * H + h) * D;
            apply_rotary_q_only(q_ptr, rotary_emb, D, token_idx);
        }
    }
}
