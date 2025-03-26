// Vec<T> AVX2 / AVX512 / NEON platform specialization
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gpumath {

template<typename T>
struct Vec {
    T value;

    __device__ Vec() = default;
    __device__ Vec(T v) : value(v) {}

    __device__ inline Vec<T> operator+(const Vec<T>& other) const { return Vec<T>(value + other.value); }
    __device__ inline Vec<T> operator*(const Vec<T>& other) const { return Vec<T>(value * other.value); }
    __device__ inline Vec<T> operator/(const Vec<T>& other) const { return Vec<T>(value / other.value); }

    __device__ inline void store(T* ptr) const { *ptr = value; }
    __device__ inline static Vec<T> load(const T* ptr) { return Vec<T>(*ptr); }

    __device__ inline static Vec<T> zero() { return Vec<T>(T(0)); }
};

template<>
struct Vec<float> {
    float value;

    __device__ Vec() = default;
    __device__ Vec(float v) : value(v) {}

    __device__ inline Vec<float> operator+(const Vec<float>& other) const { return Vec<float>(value + other.value); }
    __device__ inline Vec<float> operator*(const Vec<float>& other) const { return Vec<float>(value * other.value); }
    __device__ inline Vec<float> operator/(const Vec<float>& other) const { return Vec<float>(value / other.value); }

    __device__ inline void store(float* ptr) const { *ptr = value; }
    __device__ inline static Vec<float> load(const float* ptr) { return Vec<float>(*ptr); }

    __device__ inline static Vec<float> zero() { return Vec<float>(0.f); }
};

#if __CUDA_ARCH__ >= 530
template<>
struct Vec<half2> {
    half2 value;

    __device__ Vec() = default;
    __device__ Vec(half2 v) : value(v) {}
    __device__ Vec(half v) : value(__halves2half2(v, v)) {}

    __device__ inline Vec<half2> operator+(const Vec<half2>& other) const {
        return Vec<half2>(__hadd2(value, other.value));
    }

    __device__ inline Vec<half2> operator*(const Vec<half2>& other) const {
        return Vec<half2>(__hmul2(value, other.value));
    }

    __device__ inline void store(half2* ptr) const {
        *ptr = value;
    }

    __device__ inline static Vec<half2> load(const half2* ptr) {
        return Vec<half2>(*ptr);
    }

    __device__ inline static Vec<half2> zero() {
        return Vec<half2>({__float2half(0.f), __float2half(0.f)});
    }
};
#endif

}  // namespace gpumath

