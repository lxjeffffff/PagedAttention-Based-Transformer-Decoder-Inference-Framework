#include "softmax_lut.hpp"
#include "vec_cpu.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include<mutex>

using cpuvec::Vec;

std::vector<float> build_exp_lut(int resolution, float max_x) {
    std::vector<float> lut(resolution);
    for (int i = 0; i < resolution; ++i) {
        float x = -max_x + 2 * max_x * i / (resolution - 1);
        lut[i] = std::exp(x);
    }
    return lut;
}

// Softmax Core (Vec<T> optimization)
std::vector<float> softmax_lut(const std::vector<int32_t>& input, float scale, const std::vector<float>& lut) {
    const int resolution = static_cast<int>(lut.size());
    const float max_x = 10.0f;
    const float inv_range = (resolution - 1) / (2 * max_x);
    const size_t N = input.size();

    std::vector<float> exp_values(N);
    int32_t max_val = *std::max_element(input.begin(), input.end());

    for (size_t i = 0; i < N; i += Vec<float>::width) {
        Vec<float> v;
        for (int j = 0; j < Vec<float>::width && i + j < N; ++j) {
            float x = (static_cast<float>(input[i + j]) - max_val) * scale;
            x = std::max(-max_x, std::min(max_x, x));
            int idx = static_cast<int>((x + max_x) * inv_range);
            v[j] = lut[idx];
        }
        v.store(&exp_values[i]);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < N; i += Vec<float>::width) {
        Vec<float> x;
        x.load(&exp_values[i]);
        sum += x.sum();
    }

    float inv = 1.0f / (sum + 1e-6f);
    for (size_t i = 0; i < N; i += Vec<float>::width) {
        Vec<float> x;
        x.load(&exp_values[i]);
        x = x * Vec<float>(inv);
        x.store(&exp_values[i]);
    }

    return exp_values;
}

//In-place Softmax (scalar fallback)
void fused_softmax_lut_inplace(std::vector<int32_t>& logits, float scale, const std::vector<float>& lut, std::vector<float>& output) {
    size_t N = logits.size();
    output.resize(N);

    float max_x = 10.0f;
    int32_t max_val = *std::max_element(logits.begin(), logits.end());
    int resolution = static_cast<int>(lut.size());
    float inv_range = (resolution - 1) / (2 * max_x);

    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float x = (static_cast<float>(logits[i]) - max_val) * scale;
        x = std::max(-max_x, std::min(max_x, x));
        int idx = static_cast<int>((x + max_x) * inv_range);
        output[i] = lut[idx];
        sum += output[i];
    }

    float inv = 1.0f / (sum + 1e-6f);
    for (size_t i = 0; i < N; ++i) {
        output[i] *= inv;
    }
}

//Batch concurrency softmax
void softmax_batch_parallel(
    const std::vector<std::vector<int32_t>>& logits_batch,
    float scale,
    const std::vector<float>& lut,
    std::vector<std::vector<float>>& output_probs)
{
    size_t B = logits_batch.size();
    output_probs.resize(B);

#pragma omp parallel for
    for (size_t i = 0; i < B; ++i) {
        fused_softmax_lut_inplace(
            const_cast<std::vector<int32_t>&>(logits_batch[i]),
            scale, lut, output_probs[i]);
    }
}

//Top-k
void apply_top_k(std::vector<float>& probs, int k) {
    if (k <= 0 || k >= (int)probs.size()) return;

    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                     [&](size_t i, size_t j) { return probs[i] > probs[j]; });

    float sum = 0;
    for (int i = 0; i < k; ++i)
        sum += probs[indices[i]];

    for (size_t i = 0; i < probs.size(); ++i) {
        if (std::find(indices.begin(), indices.begin() + k, i) != indices.begin() + k) {
            probs[i] /= sum;
        } else {
            probs[i] = 0.0f;
        }
    }
}

//Top-p (nucleus)
void apply_top_p(std::vector<float>& probs, float p) {
    size_t N = probs.size();
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return probs[i] > probs[j]; });

    float cum_sum = 0.0f;
    size_t cutoff = N;
    for (size_t i = 0; i < N; ++i) {
        cum_sum += probs[indices[i]];
        if (cum_sum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    float new_sum = 0;
    for (size_t i = 0; i < cutoff; ++i)
        new_sum += probs[indices[i]];

    for (size_t i = 0; i < N; ++i) {
        if (i < cutoff) {
            probs[indices[i]] /= new_sum;
        } else {
            probs[indices[i]] = 0.0f;
        }
    }
}

// EOS Token detection
bool check_eos(const std::vector<float>& probs, int eos_token_id, float threshold) {
    if (eos_token_id < 0 || eos_token_id >= (int)probs.size()) return false;
    return probs[eos_token_id] >= threshold;
}

std::unordered_map<size_t, std::vector<float>> tile_softmax_cache;
std::mutex cache_mutex;

size_t hash_tile(const float* data, int len) {
    size_t hash = 0;
    for (int i = 0; i < len; ++i)
        hash ^= std::hash<float>()(std::round(data[i] * 1000)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

void softmax_lut_tile(float* scores, int len, float temperature, float* out) {
    size_t key = hash_tile(scores, len);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = tile_softmax_cache.find(key);
        if (it != tile_softmax_cache.end()) {
            std::copy(it->second.begin(), it->second.end(), out);
            return;
        }
    }

    float maxval = -1e9f;
    for (int i = 0; i < len; ++i)
        maxval = std::max(maxval, scores[i]);

    float sum = 0.0f;
    std::vector<float> tmp(len);
    for (int i = 0; i < len; ++i) {
        tmp[i] = std::exp((scores[i] - maxval) / temperature);
        sum += tmp[i];
    }

    for (int i = 0; i < len; ++i)
        out[i] = tmp[i] / (sum + 1e-6f);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        tile_softmax_cache[key] = std::vector<float>(out, out + len);
    }
}

void softmax_lut_vec(
    float* scores, int len, float temperature,
    float* out, float* lut_cache = nullptr
) {
    using cpuvec::Vec;
    const int step = Vec<float>::Size;

    float maxval = -1e9f;
    for (int i = 0; i < len; ++i)
        maxval = std::max(maxval, scores[i]);

    float sum = 0.0f;
    for (int i = 0; i < len; i += step) {
        Vec<float> x;
        x.load(scores + i);
        x = (x - maxval) / temperature;
        x = x.exp();  // fused LUT or exp
        x.store(out + i);
        sum += x.sum();
    }

    float inv = 1.0f / (sum + 1e-6f);
    for (int i = 0; i < len; i += step) {
        Vec<float> y;
        y.load(out + i);
        y = y * inv;
        y.store(out + i);
    }
}

void apply_topk_topp_filter(
    float* probs, int len,
    int top_k, float top_p,
    int eos_token_id = -1, float eos_thresh = 0.0f
) {
    std::vector<std::pair<float, int>> sorted;
    for (int i = 0; i < len; ++i)
        sorted.emplace_back(probs[i], i);

    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    float cum = 0.0f;
    for (int i = 0; i < len; ++i) {
        int idx = sorted[i].second;
        if ((top_k > 0 && i >= top_k) || (top_p < 1.0f && cum >= top_p))
            probs[idx] = 0.0f;
        cum += sorted[i].first;
    }

    // EOS hard threshold
    if (eos_token_id >= 0 && probs[eos_token_id] > eos_thresh)
        for (int i = 0; i < len; ++i)
            if (i != eos_token_id) probs[i] = 0.0f;
}