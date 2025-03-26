#include "attention_cuda.hpp"
#include "attention_tile_launcher.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// Template dispatch helper
template <typename T>
void attention_forward_dispatch(
    const float* q,
    float* out,
    int B, int H, int D, int T,
    const int* beam_ids,
    const void* kv_cache,
    const float* rotary_emb,
    bool is_prefill,
    bool use_overlap,
    float temperature,
    int top_k,
    float top_p,
    float* rerank_scores,
    bool debug
) {
    AttentionTileLauncher::launch<T>(
        q, out,
        B, H, D, T,
        beam_ids,
        kv_cache,
        rotary_emb,
        is_prefill,
        std::is_same<T, __half>::value,
        use_overlap,
        temperature,
        top_k,
        top_p,
        rerank_scores,
        debug
    );
}

void AttentionCUDA::forward(
    const float* q,
    float* out,
    int B, int H, int D, int T,
    const int* beam_ids,
    const void* kv_cache,
    const float* rotary_emb,
    bool is_prefill,
    bool use_fp16,
    bool use_overlap,
    float temperature,
    int top_k,
    float top_p,
    float* rerank_scores,
    bool debug
) {
    // Precision priority: INT8 > BF16 > FP16 > FP32
    if (use_fp16 == true) {
        // Optional future logic can inspect data type tag
        attention_forward_dispatch<__half>(
            q, out, B, H, D, T,
            beam_ids, kv_cache, rotary_emb,
            is_prefill, use_overlap,
            temperature, top_k, top_p,
            rerank_scores, debug
        );
    } else if (D == 128 && rotary_emb != nullptr && temperature < 1e-4f) {
        // [Optional path] dispatch to bfloat16
        attention_forward_dispatch<__nv_bfloat16>(
            q, out, B, H, D, T,
            beam_ids, kv_cache, rotary_emb,
            is_prefill, use_overlap,
            temperature, top_k, top_p,
            rerank_scores, debug
        );
    } else if (top_k <= 2 && temperature > 1.0f) {
        // [Optional path] quantized inference (INT8)
        attention_forward_dispatch<int8_t>(
            q, out, B, H, D, T,
            beam_ids, kv_cache, rotary_emb,
            is_prefill, use_overlap,
            temperature, top_k, top_p,
            rerank_scores, debug
        );
    } else {
        // Default float
        attention_forward_dispatch<float>(
            q, out, B, H, D, T,
            beam_ids, kv_cache, rotary_emb,
            is_prefill, use_overlap,
            temperature, top_k, top_p,
            rerank_scores, debug
        );
    }
}
