// INT8 decoder with oneDNN matmul + GELU LUT
#include "int8_decoder.hpp"
#include "../attention_cpu/int8_quant.hpp"
#include <algorithm>

class INT8Quantizer {
public:
    void load_fp32(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        fin.seekg(0, fin.end);
        size_t size = fin.tellg() / sizeof(float);
        fin.seekg(0, fin.beg);
        fp32_weights_.resize(size);
        fin.read(reinterpret_cast<char*>(fp32_weights_.data()), size * sizeof(float));
    }

    void quantize() {
        int8_weights_.resize(fp32_weights_.size());
        float scale = *std::max_element(fp32_weights_.begin(), fp32_weights_.end());
        for (size_t i = 0; i < fp32_weights_.size(); ++i)
            int8_weights_[i] = static_cast<int8_t>(fp32_weights_[i] / scale * 127);
    }

    void save_int8(const std::string& path) {
        std::ofstream fout(path, std::ios::binary);
        fout.write(reinterpret_cast<char*>(int8_weights_.data()), int8_weights_.size());
    }

private:
    std::vector<float> fp32_weights_;
    std::vector<int8_t> int8_weights_;
};

INT8Decoder::INT8Decoder(int num_layers, int num_heads, int head_dim, int hidden_dim, int vocab_size, int max_seq_len)
    : num_layers_(num_layers),
      embedding_(vocab_size, hidden_dim),
      kv_cache_(num_layers, num_heads, head_dim, max_seq_len) {
    for (int i = 0; i < num_layers_; ++i) {
        decoder_blocks_.emplace_back(num_heads, head_dim, hidden_dim, max_seq_len);
    }
}

void INT8Decoder::quantize_weights(const std::string& path_fp32, const std::string& path_int8) {
    {
        std::ifstream fin(path_fp32 + "/embedding.bin", std::ios::binary);
        fin.seekg(0, std::ios::end);
        size_t size = fin.tellg() / sizeof(float);
        fin.seekg(0, std::ios::beg);

        std::vector<float> fp32(size);
        fin.read(reinterpret_cast<char*>(fp32.data()), size * sizeof(float));

        std::vector<int8_t> int8(size);
        float scale = *std::max_element(fp32.begin(), fp32.end());
        for (size_t i = 0; i < size; ++i)
            int8[i] = static_cast<int8_t>(fp32[i] / scale * 127);

        std::ofstream fout(path_int8 + "/embedding.bin", std::ios::binary);
        fout.write(reinterpret_cast<char*>(int8.data()), int8.size());
    }

    for (int i = 0; i < num_layers_; ++i) {
        std::string in_layer = path_fp32 + "/layer_" + std::to_string(i);
        std::string out_layer = path_int8 + "/layer_" + std::to_string(i);
        std::filesystem::create_directories(out_layer);

        for (const std::string& fname : {
            "ln1.bin", "ln2.bin", "mlp_fc1.bin", "mlp_fc2.bin", "mlp_biases.bin"
        }) {
            std::ifstream fin(in_layer + "/" + fname, std::ios::binary);
            fin.seekg(0, std::ios::end);
            size_t size = fin.tellg() / sizeof(float);
            fin.seekg(0, std::ios::beg);

            std::vector<float> fp32(size);
            fin.read(reinterpret_cast<char*>(fp32.data()), size * sizeof(float));

            std::vector<int8_t> int8(size);
            float scale = *std::max_element(fp32.begin(), fp32.end());
            for (size_t j = 0; j < size; ++j)
                int8[j] = static_cast<int8_t>(fp32[j] / scale * 127);

            std::ofstream fout(out_layer + "/" + fname, std::ios::binary);
            fout.write(reinterpret_cast<char*>(int8.data()), int8.size());
        }
    }

    std::cout << "[INT8Decoder] Quantized weights saved to " << path_int8 << std::endl;
}

void INT8Decoder::load_quantized_weights(const std::string& path_int8) {
    embedding_.load_weights(path_int8 + "/embedding.bin");
    for (int i = 0; i < num_layers_; ++i)
        decoder_blocks_[i].load_weights(path_int8 + "/layer_" + std::to_string(i));
}

int sample_from_int8_logits(const int8_t* logits, int vocab_size, float temperature) {
    std::vector<float> dequantized(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        dequantized[i] = static_cast<float>(logits[i]) * temperature;

    auto max_it = std::max_element(dequantized.begin(), dequantized.end());
    return std::distance(dequantized.begin(), max_it);
}

void INT8Decoder::generate(const std::vector<int>& input_ids, std::vector<int>& output_ids, int max_gen_len, float temperature) {
    output_ids = input_ids;
    std::vector<int8_t> embeddings;

    for (int step = 0; step < max_gen_len; ++step) {
        embedding_.forward(output_ids, embeddings);

        for (int layer = 0; layer < num_layers_; ++layer)
            decoder_blocks_[layer].forward(embeddings.data(), embeddings.data(), &kv_cache_, output_ids.size());

        int next_token = sample_from_int8_logits(embeddings.data(), embedding_.vocab_size_, temperature);
        output_ids.push_back(next_token);
    }
}
