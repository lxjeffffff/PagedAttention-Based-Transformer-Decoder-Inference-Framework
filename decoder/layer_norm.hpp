// LayerNorm kernel
#pragma once
#include <vector>
#include <cmath>
#include <fstream>

template <typename T>
class LayerNorm {
public:
    LayerNorm(int hidden_dim, float epsilon = 1e-5f)
        : hidden_dim_(hidden_dim), epsilon_(epsilon), gamma_(hidden_dim, T(1)), beta_(hidden_dim, T(0)) {}

    void load_weights(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("Cannot open layer norm weights file");
        fin.read(reinterpret_cast<char*>(gamma_.data()), gamma_.size() * sizeof(T));
        fin.read(reinterpret_cast<char*>(beta_.data()), beta_.size() * sizeof(T));
    }

    void forward(const T* input, T* output, int batch_size) {
        for (int i = 0; i < batch_size; ++i) {
            const T* in = input + i * hidden_dim_;
            T* out = output + i * hidden_dim_;

            T mean = 0;
            for (int j = 0; j < hidden_dim_; ++j) mean += in[j];
            mean /= hidden_dim_;

            T var = 0;
            for (int j = 0; j < hidden_dim_; ++j) var += (in[j] - mean) * (in[j] - mean);
            var /= hidden_dim_;

            T inv_std = 1.0 / std::sqrt(var + epsilon_);
            for (int j = 0; j < hidden_dim_; ++j)
                out[j] = (in[j] - mean) * inv_std * gamma_[j] + beta_[j];
        }
    }

private:
    int hidden_dim_;
    float epsilon_;
    std::vector<T> gamma_, beta_;
};
