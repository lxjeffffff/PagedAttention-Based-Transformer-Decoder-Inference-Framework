#pragma once
#include "decoder_block.hpp"
#include "token_embedding.hpp"
#include "../kv_cache/kv_tile_cache.hpp"

template <typename T>
class CUDADecoder {
public:
    CUDADecoder(int num_layers, int num_heads, int head_dim, int hidden_dim, int vocab_size, int max_seq_len);

    void load_weights(const std::string& path);
    void generate(const std::vector<int>& input_ids, std::vector<int>& output_ids, int max_gen_len, float temperature = 1.0f);

private:
    int num_layers_;
    TokenEmbedding<T> embedding_;
    std::vector<DecoderBlock<T>> decoder_blocks_;
    KVTileCache<T> kv_cache_;
};
