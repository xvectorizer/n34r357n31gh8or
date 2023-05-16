#pragma once

#include <vector>
#include <cmath>
#include <immintrin.h>
#include <iostream>


double dist_l2(const std::vector<float> *v1, const std::vector<float> *v2) {
    if (v1->size() != v2->size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    double dist = 0;
    for (size_t i = 0; i < v1->size(); i++) {
        dist += ((*v1)[i] - (*v2)[i]) * ((*v1)[i] - (*v2)[i]);
    }
    return std::sqrt(dist);
}

double l2_distance_avx256(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t size = a.size();
    size_t simd_size = size / 8; // 8 floats per AVX-256 register
    size_t remainder = size % 8;

    __m256 sum = _mm256_setzero_ps();

    // Process 8 floats at a time
    for (size_t i = 0; i < simd_size; i++) {
        __m256 a_reg = _mm256_loadu_ps(&a[i * 8]);
        __m256 b_reg = _mm256_loadu_ps(&b[i * 8]);

        __m256 diff = _mm256_sub_ps(a_reg, b_reg);
        __m256 square = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, square);
    }

    // Process remainder floats
    double remainder_sum = 0;
    for (size_t i = size - remainder; i < size; i++) {
        float diff = a[i] - b[i];
        remainder_sum += diff * diff;
    }

    // Sum all values in the AVX-256 register
    float square_sum[8];
    _mm256_storeu_ps(square_sum, sum);
    float distance = 0;
    for (size_t i = 0; i < 8; i++) {
        distance += square_sum[i];
    }

    // Add remainder sum and take square root
    distance += remainder_sum;
    return std::sqrt(distance);
}

double cosine_distance(const std::vector<float> *a, const std::vector<float> *b) {
    if (a->size() != b->size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    double sum = 0;
    double sum_a = 0;
    double sum_b = 0;
    for (size_t i = 0; i < a->size(); i++) {
        sum += (*a)[i] * (*b)[i];
        sum_a += (*a)[i] * (*a)[i];
        sum_b += (*b)[i] * (*b)[i];
    }
    return 1 - (sum / (sqrt(sum_a) * sqrt(sum_b)));
}