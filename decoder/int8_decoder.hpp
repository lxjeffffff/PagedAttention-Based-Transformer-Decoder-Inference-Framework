#pragma once
#include "decoder_block.hpp"
#include "token_embedding.hpp"
#include "../kv_cache/kv_tile_cache_cpu.hpp"

class INT8Decoder {
public:
    INT8Decoder(int num_layers, int num_heads, int head_dim, int hidden_dim, int vocab_size, int max_seq_len);

    void quantize_weights(const std::string& path_fp32, const std::string& path_int8);
    void load_quantized_weights(const std::string& path_int8);
    void generate(const std::vector<int>& input_ids, std::vector<int>& output_ids, int max_gen_len, float temperature = 1.0f);

private:
    int num_layers_;
    TokenEmbedding<int8_t> embedding_;
    std::vector<DecoderBlock<int8_t>> decoder_blocks_;
    KVTileCacheCPU<int8_t> kv_cache_;
};
