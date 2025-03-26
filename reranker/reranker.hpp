#pragma once
#include <string>
#include <vector>

class Reranker {
public:
    Reranker(const std::string& model_path);
    std::vector<float> rerank_scores(const std::string& context, const std::vector<std::string>& candidates);
    int select_best(const std::string& context, const std::vector<std::string>& candidates);

private:
    class RerankerModel* model_;
};
