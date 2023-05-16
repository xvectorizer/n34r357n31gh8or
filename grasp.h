#pragma once

#include <vector>
#include <cmath>
#include <random>
#include "hnsw.h"
#include "typedef.h"
#include "space.h"
#include "util.h"

void binomial_weight_normalization_algorithm2(HNSW &hnsw, std::vector<std::vector<float>>& W, float lambda, float T) ;

void grasp_algorithm1(HNSW &hnsw, const std::vector<std::vector<float>> &Q, float eta, float T0, float beta, float sigma, int K, int L) {
    // check parameters
    if (eta <= 0) {
        throw std::invalid_argument("eta must be positive");
    }
    if (T0 <= 0) {
        throw std::invalid_argument("T0 must be positive");
    }
    if (beta <= 0 || beta >= 1) {
        throw std::invalid_argument("beta must be in (0, 1)");
    }
    if (sigma <= 0 || sigma >= 1) {
        throw std::invalid_argument("sigma must be in (0, 1)");
    }
    if (K <= 0) {
        throw std::invalid_argument("K must be positive");
    }

    // Stage 1: Construct Annealable Similarity Graph G(V, E, w)
    // Assume graph G is already constructed and stored in hnsw
    

    // Stage 2: Learning edge importance
    // initialize parameters
    float T = T0;
    int k = 0;
    float lambda0 = 0.98;
    int c = 3;

    // initialize w
    std::vector<std::vector<float>> w;
    w.resize(hnsw.id_to_level_to_neighbors.size());
    for (index_t i = 0; i < hnsw.id_to_level_to_neighbors.size(); i++) {
        w[i].resize(hnsw.id_to_level_to_neighbors[i][0].size());
        for (size_t j = 0; j < w[i].size(); j++) {
            w[i][j] = 1;
        }
    }

    // initialize edge mask
    std::vector<std::vector<bool>> edge_mask;
    std::vector<std::vector<bool>> all_edge_enabled; 
    edge_mask.resize(hnsw.id_to_level_to_neighbors.size());
    all_edge_enabled.resize(hnsw.id_to_level_to_neighbors.size());
    for (index_t i = 0; i < hnsw.id_to_level_to_neighbors.size(); i++) {
        edge_mask[i].resize(hnsw.id_to_level_to_neighbors[i][0].size());
        all_edge_enabled[i].resize(hnsw.id_to_level_to_neighbors[i][0].size());
        for (int j = 0; j < edge_mask[i].size(); j++) {
            edge_mask[i][j] = true;
            all_edge_enabled[i][j] = true;
        }
    }

    while (k <= K) {
        float lambda = sigma + (lambda0 - sigma) * std::pow(1.0 - float(k) / float(K), c);
        std::cout << "k = " << k << ", lambda = " << lambda << ", T = " << T << std::endl;

        // normalize w
        binomial_weight_normalization_algorithm2(hnsw, w, lambda, T);
        std::cout << "weight normalized" << std::endl;

        // random sample a subgraph
        std::random_device rd;
        std::default_random_engine engine(rd());
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (index_t i = 0; i < edge_mask.size(); i++) {
            for (size_t j = 0; j < edge_mask[i].size(); j++) {
                float random_float = distribution(engine);
                float p = 1 / (1 + std::exp(-w[i][j] / T));
                edge_mask[i][j] = random_float < p;
            }
        }
        std::cout << "subgraph sampled" << std::endl;

        int i = 0;
        for (const std::vector<float> &q : Q) {
            auto [p_prime, _] = hnsw.knn_search_grasp(q, 1, L, all_edge_enabled);
            auto [p, H] = hnsw.knn_search_grasp(q, 1, L, edge_mask);
            if (p_prime != p) {
                for (const auto &h : H) {
                    // update w
                    index_t start_point = h.first;
                    index_t end_point = h.second;

                    // find index of end_point in hnsw.id_to_level_to_neighbors[start_point][0]
                    const std::vector<index_t> &neighbors = hnsw.id_to_level_to_neighbors[start_point][0];
                    index_t index = std::find(neighbors.begin(), neighbors.end(), end_point) - neighbors.begin();
                    
                    float d_p_prime_q = l2_distance_avx256(hnsw.id_to_data[p_prime], q);
                    float d_p_q = l2_distance_avx256(hnsw.id_to_data[p], q) + 1e-3;
                    w[start_point][index] += (d_p_prime_q / d_p_q - 1) * eta;
                }
            }
            log_progress(i + 1, Q.size());
            i++;
        }
        T = T0 * std::pow(beta, k);
        k++;
    }

    // Stage 3: Pruning edges
    // select sigma * |E| edges with the largest weight
    
    // loop through all edges weights
    std::vector<float> all_weights;
    for (const std::vector<float> &weights : w) {
        for (float weight : weights) {
            all_weights.push_back(weight);
        }
    }
    std::sort(all_weights.begin(), all_weights.end(), std::greater<float>());
    // print 10%, 20%, ..., 100% of all weights
    for (int i = 1; i <= 10; i++) {
        std::cout << "all_weights[" << i * 10 << "%] = " << all_weights[all_weights.size() * i / 10] << std::endl;
    }
    float threshold = all_weights[all_weights.size() * sigma];

    print_histogram(all_weights);
    std::cout << "threshold = " << threshold << std::endl;

    // remove edges with weight < threshold
    for (index_t i = 0; i < w.size(); i++) {
        std::vector<index_t> indices_to_remove;
        for (index_t j = 0; j < w[i].size(); j++) {
            if (w[i][j] < threshold) {
                indices_to_remove.push_back(j);
            }
        }
        std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
        for (index_t index : indices_to_remove) {
            hnsw.id_to_level_to_neighbors[i][0].erase(hnsw.id_to_level_to_neighbors[i][0].begin() + index);
        }
    }
}

float binary_search_grasp(const std::vector<float>& arr, float left, float right, int num, float T) {
    while (std::abs(right - left) > 0.01) {
        float mid = (left + right) / 2;
        float count = 0;
        for (float x : arr) {
            count += 1 / (1 + std::exp(-(x + mid) / T));
        }
        if (count < num) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return (left + right) / 2;
}

void binomial_weight_normalization_algorithm2(HNSW &hnsw, std::vector<std::vector<float>>& W, float lambda, float T) {
    int num_edges = 0;
    for (const std::vector<std::vector<index_t>> &neighbors : hnsw.id_to_level_to_neighbors) {
        num_edges += neighbors[0].size();
    }
    float target = lambda * num_edges;

    // convert W to 1D vector
    std::vector<float> W_1D;
    for (const std::vector<float> &weights : W) {
        for (float weight : weights) {
            W_1D.push_back(weight);
        }
    }
    std::sort(W_1D.begin(), W_1D.end());

    // find max and min weight
    float max_w = W_1D[W_1D.size() - 1];
    float min_w = W_1D[0];
    float avg_w = T * log(lambda / (1 - lambda));

    std::cout << "max_w = " << max_w << ", min_w = " << min_w << ", avg_w = " << avg_w << std::endl;

    // print histogram
    print_histogram(W_1D);

    // derive search range
    float search_range_min = avg_w - max_w;
    float search_range_max = avg_w - min_w;

    std::cout << "search range: " << search_range_min << ", " << search_range_max << std::endl;

    float mu = binary_search_grasp(W_1D, search_range_min, search_range_max, target, T);
    std::cout << "mu = " << mu << std::endl;

    // update W
    for (index_t i = 0; i < W.size(); i++){
        for (size_t j = 0; j < W[i].size(); j++) {
            W[i][j] += mu;
        }
    }
}