#pragma once
#include "layer_norm.hpp"
#include "mlp.hpp"
#include "../attention_cuda/attention_cuda.hpp"
#include "../attention_cuda/attention_config.hpp"

#include <fstream>
#include <stdexcept>

template <typename T>
inline void load_vector_from_file(const std::string& path, std::vector<T>& buffer) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }
    fin.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(T));
    if (!fin) {
        throw std::runtime_error("Failed to read from file: " + path);
    }
}

template <typename T>
class DecoderBlock {
public:
    DecoderBlock(int num_heads, int head_dim, int hidden_dim, int max_seq_len)
        : num_heads_(num_heads), head_dim_(head_dim), hidden_dim_(hidden_dim), max_seq_len_(max_seq_len),
          layer_norm1_(hidden_dim), layer_norm2_(hidden_dim),
          mlp_(hidden_dim, hidden_dim * 4),
          attention_cuda_({
              .num_heads = num_heads,
              .head_dim = head_dim,
              .max_seq_len = max_seq_len
          }) {}

    void load_weights(const std::string& path) {
        layer_norm1_.load_weights(path + "/ln1.bin");
        layer_norm2_.load_weights(path + "/ln2.bin");
        mlp_.load_weights(path);
    }

    void forward(const T* input, T* output, KVTileCache<T>* kv_cache, int seq_len) {
        std::vector<T> norm_out(seq_len * hidden_dim_);
        layer_norm1_.forward(input, norm_out.data(), seq_len);

        attention_cuda_.forward(
            norm_out.data(),        // q
            norm_out.data(),        // out
            1,                      // B
            num_heads_,             // H
            head_dim_,              // D
            seq_len,                // T
            nullptr,                // beam_ids
            kv_cache,               // kv_cache
            nullptr,                // rotary_emb
            false,                  // is_prefill
            false,                  // use_fp16
            false                   // use_overlap
        );

        layer_norm2_.forward(norm_out.data(), norm_out.data(), seq_len);
        mlp_.forward(norm_out.data(), output, seq_len);
    }

private:
    int num_heads_, head_dim_, hidden_dim_, max_seq_len_;
    LayerNorm<T> layer_norm1_, layer_norm2_;
    MLP<T> mlp_;
    AttentionCUDA attention_cuda_;
};
