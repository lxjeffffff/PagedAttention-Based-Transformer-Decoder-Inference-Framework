#pragma once
#include <string>

class RerankerModel {
public:
    RerankerModel(const std::string& model_path);
    ~RerankerModel();

    float score(const std::string& context, const std::string& candidate);

private:
    void load_model(const std::string& path);
    void* internal_model_;
};
