#include "attention_cpu.hpp"
#include "cpu_attention_kernel.hpp"

template <typename T>
CPUAttention<T>::CPUAttention(const CPUAttentionConfig& config)
    : config_(config) {}

template <typename T>
void CPUAttention<T>::forward(const CPUAttentionInput<T>& input, CPUAttentionOutput<T>& output) {
    cpu_paged_attention_forward<T>(
        input.q,
        input.k,
        input.v,
        input.rotary_emb,
        input.kv_cache,
        input.beam_ids,
        input.lut,
        input.B,
        input.H,
        input.D,
        input.T,
        input.tile_size,
        input.temperature,
        input.causal,
        input.top_k,
        input.top_p,
        input.eos_token_id,
        input.eos_threshold,
        output.out,
        output.attention_weights,
        output.logits
    );
}

template class CPUAttention<float>;
template class CPUAttention<int8_t>;
