#pragma once
#include <vector>
#include <cstdint>

std::vector<int8_t> quantize_to_int8(const std::vector<float>& input, float scale);

std::vector<int8_t> batch_quantize(const std::vector<float>& input, const std::vector<float>& scales, int dim);

float compute_absmax(const std::vector<float>& input);

/// @brief Dequantize INT8 vector to FP32 using global scale
std::vector<float> dequantize_from_int8(const std::vector<int8_t>& input, float scale);

/// @brief Dequantize INT8 batch matrix using per-row scale
std::vector<float> batch_dequantize(const std::vector<int8_t>& input, const std::vector<float>& scales, int dim);

/// @brief Compute symmetric scale using min/max
float compute_minmax_scale(const std::vector<float>& input);
