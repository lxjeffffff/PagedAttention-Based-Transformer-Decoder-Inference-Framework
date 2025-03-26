#pragma once
#include "cpu_attention_config.hpp"
#include "../kv_cache/kv_tile_cache_cpu.hpp"
#include <vector>
#include <cstdint>

// Full input structure for CPU-based attention inference
template <typename T>
struct CPUAttentionInput {
    const T* q = nullptr;                      // Query [B, H, D]
    const T* k = nullptr;                      // Key   [B, T, H, D] (optional, unused if kv_cache provided)
    const T* v = nullptr;                      // Value [B, T, H, D] (optional, unused if kv_cache provided)
    T* output = nullptr;                       // Output [B, H, D]

    const float* rotary_emb = nullptr;         // Rotary embedding table [T, D]
    const std::vector<int>* beam_ids = nullptr;// Beam routing [B]

    KVTileCacheCPU<T>* kv_cache = nullptr;     // KV cache manager (optional)

    const std::vector<float>* lut = nullptr;   // Softmax lookup table (optional)

    int B = 1;          // Batch size
    int H = 1;          // Num heads
    int D = 64;         // Head dim
    int T = 1;          // Sequence length
    int tile_size = 32; // Tile granularity

    float temperature = 1.0f;
    bool causal = true;

    int top_k = 0;
    float top_p = 1.0f;
    int eos_token_id = -1;
    float eos_threshold = 0.0f;
};

// Output structure
template <typename T>
struct CPUAttentionOutput {
    T* out = nullptr; // Output tensor [B, H, D]
    std::vector<std::vector<float>>* attention_weights = nullptr; // [B*H][T]
    std::vector<std::vector<float>>* logits = nullptr;             // [B*H][T]
};

template <typename T>
class CPUAttention {
public:
    CPUAttention(const CPUAttentionConfig& config);
    void forward(const CPUAttentionInput<T>& input, CPUAttentionOutput<T>& output);

private:
    CPUAttentionConfig config_;
};
