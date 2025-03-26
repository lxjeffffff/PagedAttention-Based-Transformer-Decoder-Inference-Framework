#include "int8_quant.hpp"
#include <cmath>
#include <algorithm>

std::vector<int8_t> quantize_to_int8(const std::vector<float>& input, float scale) {
    std::vector<int8_t> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        int32_t q = static_cast<int32_t>(std::round(input[i] * scale));
        q = std::max(-128, std::min(127, q));
        output[i] = static_cast<int8_t>(q);
    }
    return output;
}

std::vector<int8_t> batch_quantize(const std::vector<float>& input, const std::vector<float>& scales, int dim) {
    int B = static_cast<int>(scales.size());
    std::vector<int8_t> output(B * dim);
    for (int b = 0; b < B; ++b) {
        float scale = scales[b];
        for (int i = 0; i < dim; ++i) {
            int idx = b * dim + i;
            int32_t q = static_cast<int32_t>(std::round(input[idx] * scale));
            q = std::max(-128, std::min(127, q));
            output[idx] = static_cast<int8_t>(q);
        }
    }
    return output;
}

float compute_absmax(const std::vector<float>& input) {
    float max_val = 0.f;
    for (float v : input) {
        max_val = std::max(max_val, std::abs(v));
    }
    return max_val;
}

std::vector<float> dequantize_from_int8(const std::vector<int8_t>& input, float scale) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]) / scale;
    }
    return output;
}

std::vector<float> batch_dequantize(const std::vector<int8_t>& input, const std::vector<float>& scales, int dim) {
    int B = static_cast<int>(scales.size());
    std::vector<float> output(B * dim);
    for (int b = 0; b < B; ++b) {
        float scale = scales[b];
        for (int i = 0; i < dim; ++i) {
            int idx = b * dim + i;
            output[idx] = static_cast<float>(input[idx]) / scale;
        }
    }
    return output;
}

float compute_minmax_scale(const std::vector<float>& input) {
    float min_val = *std::min_element(input.begin(), input.end());
    float max_val = *std::max_element(input.begin(), input.end());
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    return 127.f / (abs_max + 1e-6f);
}
