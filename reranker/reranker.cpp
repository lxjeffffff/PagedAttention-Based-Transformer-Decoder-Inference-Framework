#include "reranker.hpp"
#include "reranker_model.hpp"
#include <stdexcept>
#include <iterator>
#include <algorithm>

Reranker::Reranker(const std::string& model_path) {
    model_ = new RerankerModel(model_path);
}

std::vector<float> Reranker::rerank_scores(const std::string& context, const std::vector<std::string>& candidates) {
    if (candidates.empty())
        throw std::runtime_error("Candidates list cannot be empty.");

    std::vector<float> scores;
    for (const auto& candidate : candidates)
        scores.push_back(model_->score(context, candidate));

    return scores;
}

int Reranker::select_best(const std::string& context, const std::vector<std::string>& candidates) {
    auto scores = rerank_scores(context, candidates);
    return std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
}
