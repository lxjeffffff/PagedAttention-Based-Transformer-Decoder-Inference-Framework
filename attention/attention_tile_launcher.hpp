#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "../kv_cache/kv_tile_cache.hpp"

template <typename T>
__global__ void paged_flash_attention_kernel_fused(
    const float* __restrict__ q,
    float* __restrict__ output,
    float* __restrict__ rerank_scores,
    int B, int T, int H, int D,
    const float* __restrict__ rotary_emb,
    const int* __restrict__ beam_ids,
    KVTileCache<T>* kv_cache,
    float temperature,
    int top_k,
    float top_p
);

template <typename T>
__global__ void paged_flash_attention_kernel_fused_overlap(
    const float* __restrict__ q,
    float* __restrict__ output,
    float* __restrict__ rerank_scores,
    int B, int T, int H, int D,
    int num_tiles,
    const float* __restrict__ rotary_emb,
    const int* __restrict__ beam_ids,
    KVTileCache<T>* kv_cache,
    float temperature,
    int top_k,
    float top_p
);

struct AttentionTileLauncher {
    template <typename T>
    static void launch(
        const float* q, 
        float* out,
        int B, int H, int D, int T,
        const int* beam_ids = nullptr,
        KVTileCache<T>* kv_cache = nullptr,
        const float* rotary_emb = nullptr,
        bool is_prefill = true,
        bool use_fp16 = false,
        bool use_overlap = false,
        float temperature = 1.0f,
        int top_k = 1,
        float top_p = 1.0f,
        float* rerank_scores = nullptr,
        bool debug = false
    ) {
        dim3 grid(B, H);
        dim3 block(128);
        size_t shared_mem_size = sizeof(float) * block.x

        if (use_overlap) {
            paged_flash_attention_kernel_fused_overlap<T><<<grid, block, shared_mem_size>>>(
                q, out,
                rerank_scores,
                B, T, H, D,
                1,  // num_tiles
                rotary_emb,
                beam_ids,
                kv_cache,
                temperature,
                top_k,
                top_p
            );
        } else {
            paged_flash_attention_kernel_fused<T><<<grid, block, shared_mem_size>>>(
                q, out,
                rerank_scores,
                B, T, H, D,
                rotary_emb,
                beam_ids,
                kv_cache,
                temperature,
                top_k,
                top_p
            );
        }

        if (debug) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("[AttentionTileLauncher] CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        }
    }
};
