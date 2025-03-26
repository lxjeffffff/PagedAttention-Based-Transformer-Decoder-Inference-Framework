// Token embedding with prefetch buffer
#pragma once
#include <vector>
#include <fstream>
#include <stdexcept>

template <typename T>
class TokenEmbedding {
public:
    TokenEmbedding(int vocab_size, int hidden_dim)
        : vocab_size_(vocab_size), hidden_dim_(hidden_dim), embedding_matrix_(vocab_size * hidden_dim) {}

    void load_weights(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("Cannot open embedding weights file");
        fin.read(reinterpret_cast<char*>(embedding_matrix_.data()), embedding_matrix_.size() * sizeof(T));
    }

    void forward(const std::vector<int>& input_ids, std::vector<T>& embeddings) {
        embeddings.resize(input_ids.size() * hidden_dim_);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            const T* src = &embedding_matrix_[input_ids[i] * hidden_dim_];
            T* dst = &embeddings[i * hidden_dim_];
            std::copy(src, src + hidden_dim_, dst);
        }
    }

private:
    int vocab_size_;
    int hidden_dim_;
    std::vector<T> embedding_matrix_;
};
