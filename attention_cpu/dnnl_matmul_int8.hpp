#pragma once
#include <cstdint>
#include <string>

/// @brief DNNL INT8 Batch MatMul with Bias + Activation + INT8 output
bool dnnl_matmul_int8(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int BATCH, int M, int N, int K,
    float scaleA, float scaleB, float scaleC = 1.0f,
    const float* bias = nullptr,
    const std::string& activation = ""
);
