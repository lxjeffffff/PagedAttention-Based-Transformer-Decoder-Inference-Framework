#include <cuda_runtime.h>
#include "../kv_cache/kv_tile_cache.hpp"

// Paged Flash Attention kernel with overlap, integrated with GPU KV Cache
template <typename T>
__global__ void paged_flash_attention_kernel_fused_overlap(
    const float* __restrict__ q,
    float* __restrict__ output,
    float* __restrict__ rerank_scores,
    int B, int T, int H, int D,
    int num_tiles,
    const float* __restrict__ rotary_emb,
    const int* __restrict__ beam_ids,
    KVTileCache<T>* kv_cache, // Integrated KV cache
    float temperature,
    int top_k,
    float top_p
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int beam_id = beam_ids ? beam_ids[batch_idx] : batch_idx;

    extern __shared__ float shared_mem[];
    float* logits = shared_mem;  // Shared memory buffer for logits

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // MODIFICATION: Fetch k and v dynamically from kv_cache
        const float* k_tile = kv_cache->get(beam_id, head_idx, tile_id, 'k');
        const float* v_tile = kv_cache->get(beam_id, head_idx, tile_id, 'v');

        if (!k_tile || !v_tile) continue;

        // ===== Original Attention Logic Below (Unchanged) =====

        // Apply Rotary Embedding (if rotary_emb provided)
        float q_rot[D];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            q_rot[d] = rotary_emb ? q[batch_idx * H * D + head_idx * D + d] * rotary_emb[d]
                                  : q[batch_idx * H * D + head_idx * D + d];
        }

        // Compute QK^T similarity
        float logit = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            logit += q_rot[d] * k_tile[tid * D + d];
        }
        logit /= temperature;

        // Save logits (optional reranking)
        if (rerank_scores) {
            rerank_scores[batch_idx * num_tiles + tile_id] = logit;
        }

        logits[tid] = logit;
        __syncthreads();

        // Warp-level softmax computation
        float max_logit = -INFINITY;
        #pragma unroll
        for (int i = 0; i < D; ++i) {
            max_logit = max(max_logit, logits[i]);
        }

        float exp_sum = 0.f;
        #pragma unroll
        for (int i = 0; i < D; ++i) {
            logits[i] = expf(logits[i] - max_logit);
            exp_sum += logits[i];
        }

        float softmax_val = logits[tid] / (exp_sum + 1e-8f);

        // Optional Top-k / Top-p filtering
        if ((top_k > 0 && tid >= top_k) || (top_p < 1.0f && softmax_val < top_p)) {
            softmax_val = 0.f;
        }

        __syncthreads();

        // Compute final attention output: weighted sum of V
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            atomicAdd(&output[batch_idx * H * D + head_idx * D + d], softmax_val * v_tile[tid * D + d]);
        }
        __syncthreads();
    }
}

// Explicit instantiation
template __global__ void paged_flash_attention_kernel_fused_overlap<float>(
    const float*, KVTileCache<float>*, float*, float*, int, int, int, int, int,
    const float*, const int*, float, int, float);
