#pragma once
#include <vector>
#include <cstdint>

std::vector<float> build_exp_lut(int resolution = 1024, float max_x = 10.0f);

std::vector<float> softmax_lut(
    const std::vector<int32_t>& input,
    float scale,
    const std::vector<float>& lut);

void fused_softmax_lut_inplace(
    std::vector<int32_t>& logits,
    float scale,
    const std::vector<float>& lut,
    std::vector<float>& output);

void softmax_batch_parallel(
    const std::vector<std::vector<int32_t>>& logits_batch,
    float scale,
    const std::vector<float>& lut,
    std::vector<std::vector<float>>& output_probs);

// Top-k / Top-p / EOS support
void apply_top_k(std::vector<float>& probs, int k);
void apply_top_p(std::vector<float>& probs, float p);
bool check_eos(const std::vector<float>& probs, int eos_token_id, float threshold);
size_t hash_tile(const float* data, int len);
void softmax_lut_tile(float* scores, int len, float temperature, float* out);
void softmax_lut_vec(
    float* scores, int len, float temperature,
    float* out, float* lut_cache = nullptr
);
void apply_topk_topp_filter(
    float* probs, int len,
    int top_k, float top_p,
    int eos_token_id = -1, float eos_thresh = 0.0f
);