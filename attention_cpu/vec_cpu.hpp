#pragma once

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>
#include <cstdint>
#include <limits>

namespace cpuvec {

template<typename T, int N = 8>
struct Vec {
    T data[N];
    static constexpr int width = N;
    static constexpr int Size = N;

    Vec() { zero(); }
    explicit Vec(T val) { fill(val); }

    static Vec<T, N> from_scalar(T val) {
        Vec<T, N> v;
        v.fill(val);
        return v;
    }

    void fill(T val) {
        for (int i = 0; i < N; ++i) data[i] = val;
    }

    void zero() { fill(T(0)); }

    void load(const T* ptr) {
        std::memcpy(data, ptr, sizeof(T) * N);
    }

    void store(T* ptr) const {
        std::memcpy(ptr, data, sizeof(T) * N);
    }

    static void load_batch(const T* src, Vec<T, N>* dst, int num_vecs) {
        for (int i = 0; i < num_vecs; ++i)
            dst[i].load(src + i * N);
    }

    static void store_batch(T* dst, const Vec<T, N>* src, int num_vecs) {
        for (int i = 0; i < num_vecs; ++i)
            src[i].store(dst + i * N);
    }

    Vec<T, N> exp() const {
        return map([](T x) { return std::exp(x); });
    }
    
    Vec<T, N> log() const {
        return map([](T x) { return std::log(x); });
    }
    
    Vec<T, N> sqrt() const {
        return map([](T x) { return std::sqrt(x); });
    }

    Vec<T, N> operator+(const Vec<T, N>& other) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] + other.data[i];
        return out;
    }

    Vec<T, N> operator-(const Vec<T, N>& other) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] - other.data[i];
        return out;
    }

    Vec<T, N> operator-(const T& scalar) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] - scalar;
        return out;
    }

    Vec<T, N> operator*(const Vec<T, N>& other) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] * other.data[i];
        return out;
    }

    Vec<T, N> operator*(const T& scalar) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] * scalar;
        return out;
    }

    Vec<T, N> operator/(const Vec<T, N>& other) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] / other.data[i];
        return out;
    }

    Vec<T, N> operator/(const T& scalar) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] / scalar;
        return out;
    }

    Vec<T, N>& operator+=(const Vec<T, N>& other) {
        for (int i = 0; i < N; ++i)
            data[i] += other.data[i];
        return *this;
    }

    Vec<T, N>& operator-=(const Vec<T, N>& other) {
        for (int i = 0; i < N; ++i)
            data[i] -= other.data[i];
        return *this;
    }

    Vec<T, N>& operator*=(const Vec<T, N>& other) {
        for (int i = 0; i < N; ++i)
            data[i] *= other.data[i];
        return *this;
    }

    Vec<T, N>& operator/=(const T& scalar) {
        for (int i = 0; i < N; ++i)
            data[i] /= scalar;
        return *this;
    }

    Vec<T, N> map(std::function<T(T)> fn) const {
        Vec<T, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = fn(data[i]);
        return out;
    }

    void map_inplace(std::function<T(T)> fn) {
        for (int i = 0; i < N; ++i)
            data[i] = fn(data[i]);
    }

    Vec<T, N> relu() const {
        return map([](T x) { return std::max(T(0), x); });
    }

    Vec<T, N> gelu() const {
        return map([](T x) {
            const float a = std::sqrt(2.0f / 3.1415926f);
            return 0.5f * x * (1.0f + std::tanh(a * (x + 0.044715f * x * x * x)));
        });
    }

    T sum() const {
        T s = 0;
        for (int i = 0; i < N; ++i) s += data[i];
        return s;
    }

    T max() const {
        T m = data[0];
        for (int i = 1; i < N; ++i) m = std::max(m, data[i]);
        return m;
    }

    T min() const {
        T m = data[0];
        for (int i = 1; i < N; ++i) m = std::min(m, data[i]);
        return m;
    }

    std::vector<T> to_vector() const {
        return std::vector<T>(data, data + N);
    }

    T& operator[](int idx) { return data[idx]; }
    const T& operator[](int idx) const { return data[idx]; }
};


// ===== INT8 Specialization =====
template<int N>
struct Vec<int8_t, N> {
    int8_t data[N];

    void load(const int8_t* ptr) {
        std::memcpy(data, ptr, N);
    }

    void store(int8_t* ptr) const {
        std::memcpy(ptr, data, N);
    }

    void fill(int8_t val) {
        std::fill(data, data + N, val);
    }

    void zero() {
        std::memset(data, 0, N);
    }

    int sum() const {
        int s = 0;
        for (int i = 0; i < N; ++i) s += int(data[i]);
        return s;
    }

    int8_t min() const {
        int8_t m = data[0];
        for (int i = 1; i < N; ++i)
            m = std::min(m, data[i]);
        return m;
    }

    int8_t max() const {
        int8_t m = data[0];
        for (int i = 1; i < N; ++i)
            m = std::max(m, data[i]);
        return m;
    }

    Vec<int8_t, N> operator+(const Vec<int8_t, N>& other) const {
        Vec<int8_t, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = data[i] + other.data[i];
        return out;
    }

    Vec<int8_t, N> map(std::function<int8_t(int8_t)> fn) const {
        Vec<int8_t, N> out;
        for (int i = 0; i < N; ++i)
            out.data[i] = fn(data[i]);
        return out;
    }

    int8_t& operator[](int idx) { return data[idx]; }
    const int8_t& operator[](int idx) const { return data[idx]; }
};

}  // namespace cpuvec
