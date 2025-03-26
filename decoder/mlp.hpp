// MLP with quant/int8/float versions
#pragma once
#include <vector>
#include <fstream>

template <typename T>
class MLP {
public:
    MLP(int hidden_dim, int intermediate_dim)
        : hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim),
          fc1_weights_(hidden_dim * intermediate_dim), fc1_bias_(intermediate_dim),
          fc2_weights_(intermediate_dim * hidden_dim), fc2_bias_(hidden_dim) {}

    void load_weights(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("Cannot open MLP weights file");
        fin.read(reinterpret_cast<char*>(fc1_weights_.data()), fc1_weights_.size() * sizeof(T));
        fin.read(reinterpret_cast<char*>(fc1_bias_.data()), fc1_bias_.size() * sizeof(T));
        fin.read(reinterpret_cast<char*>(fc2_weights_.data()), fc2_weights_.size() * sizeof(T));
        fin.read(reinterpret_cast<char*>(fc2_bias_.data()), fc2_bias_.size() * sizeof(T));
    }

    void forward(const T* input, T* output, int batch_size) {
        std::vector<T> intermediate(intermediate_dim_);
        for (int b = 0; b < batch_size; ++b) {
            // fc1: hidden -> intermediate
            for (int i = 0; i < intermediate_dim_; ++i) {
                T sum = fc1_bias_[i];
                for (int j = 0; j < hidden_dim_; ++j)
                    sum += input[b * hidden_dim_ + j] * fc1_weights_[j * intermediate_dim_ + i];
                intermediate[i] = std::max(T(0), sum);  // ReLU
            }
            // fc2: intermediate -> hidden
            for (int i = 0; i < hidden_dim_; ++i) {
                T sum = fc2_bias_[i];
                for (int j = 0; j < intermediate_dim_; ++j)
                    sum += intermediate[j] * fc2_weights_[j * hidden_dim_ + i];
                output[b * hidden_dim_ + i] = sum;
            }
        }
    }

private:
    int hidden_dim_, intermediate_dim_;
    std::vector<T> fc1_weights_, fc1_bias_, fc2_weights_, fc2_bias_;
};
