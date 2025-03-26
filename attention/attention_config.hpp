#pragma once
#include <cstdint>
#include "../kv_cache/kv_tile_cache.hpp"

class AttentionCUDA {
public:
    static void forward(
        const float* q,             // Query tensor [B, H, T, D]
        float* out,                 // Output tensor [B, H, T, D]
        int B,                      // Batch size
        int H,                      // Number of heads
        int D,                      // Head dimension
        int T,                      // Sequence length
        const int* beam_ids = nullptr,          // [B] Beam id per sequence
        KVTileCache<float>* kv_cache = nullptr,         // KV cache (optional)
        const float* rotary_emb = nullptr,      // Rotary embedding [T, D]
        bool is_prefill = true,                 // Prefill or decode
        bool use_fp16 = false,                  // Use float16
        bool use_overlap = false,               // Use fused-overlap kernel
        float temperature = 1.0f,               // Sampling temperature
        int top_k = 1,                          // Top-K sampling
        float top_p = 1.0f,                     // Top-P nucleus sampling
        float* rerank_scores = nullptr,         // [B * H] output scores
        bool debug = false                      // CUDA debug print
    );
};
