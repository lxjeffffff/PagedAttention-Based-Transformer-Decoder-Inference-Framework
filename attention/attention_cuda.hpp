#pragma once

#include "attention_config.hpp"

#include "../kv_cache/kv_tile_cache.hpp"

// CUDA Attention launcher class
class AttentionCUDA {
public:
    AttentionCUDA(const AttentionConfig& config)
        : config_(config) {}

    // Launch paged attention kernel with fused overlap (optional)
    void forward(
        const float* q,                  // [B, H, D]
        float* out,                      // [B, H, D]
        int B, int H, int D, int T,      // Dynamic sequence length
        const int* beam_ids,             // [B], nullable
        KVTileCache<float>* kv_cache = nullptr,            // pointer to cache manager
        const float* rotary_emb = nullptr, // optional RoPE
        bool is_prefill = false,
        bool use_fp16 = false,
        bool use_overlap = false
    );

private:
    AttentionConfig config_;
};
