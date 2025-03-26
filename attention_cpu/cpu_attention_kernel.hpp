#pragma once

#include <vector>
#include <cstdint>
#include "../kv_cache/kv_tile_cache_cpu.hpp"

// Input structure for CPU PagedAttention forward
template <typename T>
struct CPUAttentionInput {
    const T* q = nullptr;                         // Query tensor [B, H, D]
    const std::vector<int>* beam_ids = nullptr;   // Optional beam ID routing [B]
    const float* rotary_emb = nullptr;            // Rotary embedding table [T, D] or flattened

    int B = 0;         // Batch size
    int H = 0;         // Number of heads
    int T = 0;         // Sequence length
    int D = 0;         // Head dimension
    int tile_size = 1; // Tile size for cached KV
    float temperature = 1.0f;     // Softmax temperature
    int top_k = 0;             // Top-k filter threshold
    float top_p = 1.0f;           // Nucleus sampling threshold
    int eos_token = -1;         // EOS token id
    float eos_threshold = 0.0f;   // Threshold to trigger EOS
    bool causal = false;           // Whether to apply causal masking

    KVTileCacheCPU<T>* kv_cache = nullptr;     // KV cache manager (optional)
    const std::vector<float>* lut = nullptr;  // Optional softmax lookup table
};

// Output structure for CPU PagedAttention forward
template <typename T>
struct CPUAttentionOutput {
    T* out = nullptr;  // Output tensor [B, H, D]

    // Optional attention weights [B * H][T]
    std::vector<std::vector<float>>* attention_weights = nullptr;

    // Optional logits before softmax [B * H][T]
    std::vector<std::vector<float>>* logits = nullptr;
};

// Forward function for paged attention with KV tile cache
template <typename T>
void cpu_paged_attention_forward(
    const CPUAttentionInput<T>& input,
    CPUAttentionOutput<T>& output
);
