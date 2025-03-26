#include "cpu_attention_kernel.hpp"
#include "vec_cpu.hpp"
#include "softmax_lut.hpp"
#include "../kv_cache/kv_tile_cache_cpu.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
inline void apply_rotary_embedding_tile(
    Vec<float>* q_vec, const float* rope_tile, int D
) {
    // Apply rotary embedding to a query vector (float)
    for (int d = 0; d < D; d += 2) {
        float cos = rope_tile[d];
        float sin = rope_tile[d + 1];
        float q0 = (*q_vec)[d], q1 = (*q_vec)[d + 1];
        (*q_vec)[d]     = q0 * cos - q1 * sin;
        (*q_vec)[d + 1] = q0 * sin + q1 * cos;
    }
}

template <>
inline void apply_rotary_embedding_tile<int8_t>(
    Vec<float>* q_vec, const float* rope_tile, int D
) {
    // Rotary embedding for int8_t is applied to its float-decoded query
    for (int d = 0; d < D; d += 2) {
        float cos = rope_tile[d];
        float sin = rope_tile[d + 1];
        float q0 = (*q_vec)[d], q1 = (*q_vec)[d + 1];
        (*q_vec)[d]     = q0 * cos - q1 * sin;
        (*q_vec)[d + 1] = q0 * sin + q1 * cos;
    }
}

template <typename T>
void cpu_paged_attention_forward(
    const CPUAttentionInput<T>& input,
    CPUAttentionOutput<T>& output
) {
    using cpuvec::Vec;

    const int B = input.B, H = input.H, T = input.T, D = input.D;
    const int tile_size = input.tile_size;
    const int num_tiles = (T + tile_size - 1) / tile_size;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const int beam_id = input.beam_ids ? (*input.beam_ids)[b] : b;
            const T* q_ptr = input.q + ((b * H + h) * D);
            Vec<float> q_vec;
            q_vec.load(q_ptr, D);  // Will decode if int8_t

            // Apply rotary embedding if provided
            if (input.rotary_emb) {
                const float* rope_tile = input.rotary_emb;
                apply_rotary_embedding_tile<T>(&q_vec, rope_tile, D);
            }

            std::vector<float> scores(T, -1e9f);
            std::vector<float> softmax_out(T, 0.0f);

            float* lut_buf = nullptr;
            if (input.lut) lut_buf = const_cast<float*>(input.lut->data());

            // Compute QK^T similarity across all tiles
            for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
                int tile_start = tile_id * tile_size;
                int tile_len = std::min(tile_size, T - tile_start);

                const T* k_tile = input.kv_cache->get(beam_id, h, tile_id, 'k');
                if (!k_tile) continue;

                for (int t = 0; t < tile_len; ++t) {
                    int token_id = tile_start + t;
                    Vec<float> k_vec;
                    k_vec.load(k_tile + t * D, D);

                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += q_vec[d] * k_vec[d];

                    if (!input.causal || token_id <= 0)
                        scores[token_id] = dot / input.temperature;
                }
            }

            // Apply softmax with LUT (vectorized)
            softmax_lut_vec(scores.data(), T, input.temperature, softmax_out.data(), lut_buf);

            // Apply top-k / top-p / EOS filtering
            apply_topk_topp_filter(
                softmax_out.data(), T,
                input.top_k, input.top_p,
                input.eos_token, input.eos_threshold
            );

            Vec<float> out_vec;
            out_vec.clear();

            // Compute AV = Softmax(QK^T) * V
            for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
                int tile_start = tile_id * tile_size;
                int tile_len = std::min(tile_size, T - tile_start);

                const T* v_tile = input.kv_cache->get(beam_id, h, tile_id, 'v');
                if (!v_tile) continue;

                for (int t = 0; t < tile_len; ++t) {
                    int token_id = tile_start + t;
                    Vec<float> v_vec;
                    v_vec.load(v_tile + t * D, D);
                    for (int d = 0; d < D; ++d)
                        out_vec[d] += softmax_out[token_id] * v_vec[d];
                }
            }

            // Write output
            out_vec.store(output.out + ((b * H + h) * D), D);

            // Optional: store attention weights and logits
            if (output.attention_weights)
                (*output.attention_weights)[b * H + h] = softmax_out;
            if (output.logits)
                (*output.logits)[b * H + h] = scores;
        }
    }
}

// Explicit instantiations
template void cpu_paged_attention_forward<float>(const CPUAttentionInput<float>&, CPUAttentionOutput<float>&);
template void cpu_paged_attention_forward<int8_t>(const CPUAttentionInput<int8_t>&, CPUAttentionOutput<int8_t>&);
