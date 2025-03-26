// GPU decoder with Flash kernel routing
#include "cuda_decoder.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

int sample_from_logits(const float* logits, int vocab_size, float temperature) {
    std::vector<float> adjusted_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        adjusted_logits[i] = logits[i] / temperature;

    auto max_it = std::max_element(adjusted_logits.begin(), adjusted_logits.end());
    return std::distance(adjusted_logits.begin(), max_it);
}

template <typename T>
void load_vector_from_file(const std::string& path, std::vector<T>& buffer) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open file: " + path);
    fin.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(T));
}

template <typename T>
CUDADecoder<T>::CUDADecoder(int num_layers, int num_heads, int head_dim, int hidden_dim, int vocab_size, int max_seq_len)
    : num_layers_(num_layers),
      embedding_(vocab_size, hidden_dim),
      kv_cache_(num_layers, num_heads, head_dim, max_seq_len) {
    
    for (int i = 0; i < num_layers_; ++i) {
        decoder_blocks_.emplace_back(num_heads, head_dim, hidden_dim, max_seq_len);
    }
}

template <typename T>
void CUDADecoder<T>::load_weights(const std::string& path) {
    // Load embedding weights
    std::string embedding_path = path + "/embedding.bin";
    load_vector_from_file(embedding_path, embedding_.weights());

    // Load decoder block weights
    for (int i = 0; i < num_layers_; ++i) {
        std::string layer_path = path + "/layer_" + std::to_string(i);
        decoder_blocks_[i].load_weights(layer_path);
    }
}

template <typename T>
void CUDADecoder<T>::generate(const std::vector<int>& input_ids, std::vector<int>& output_ids, int max_gen_len, float temperature) {
    output_ids = input_ids;
    std::vector<T> embeddings;

    for (int step = 0; step < max_gen_len; ++step) {
        embedding_.forward(output_ids, embeddings);

        for (int layer = 0; layer < num_layers_; ++layer)
            decoder_blocks_[layer].forward(embeddings.data(), embeddings.data(), &kv_cache_, output_ids.size());

        int next_token = sample_from_logits(embeddings.data(), embedding_.vocab_size_, temperature);
        output_ids.push_back(next_token);
    }
}

template CUDADecoder<float>::CUDADecoder(int, int, int, int, int, int);
template void CUDADecoder<float>::load_weights(const std::string& path);
template class CUDADecoder<float>;
