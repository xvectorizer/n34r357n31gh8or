#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <iterator>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <map>
#include <string>
#include <sstream>
#include <unordered_map>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <immintrin.h>
#include "util.h"
#include "space.h"
#include "typedef.h"


class VAMANA {
public:
    // hyper parameters
    double alpha;
    int L;
    int R;

    // statistics
    unsigned long long int distance_calculation_count;           // count number of calling distance function

    // graph
    index_t enter_point;
    std::vector<std::vector<float>> id_to_data;
    std::vector<std::vector<index_t>> id_to_neighbors;
    std::unordered_map<index_t, std::unordered_map<index_t, index_t>> edge_to_frequency;
    std::map<std::vector<index_t>, index_t> new_edges_to_suggestion;
    std::vector<index_t> id_to_parent;

    // shared mutex
    std::vector<std::shared_mutex> id_to_neighbors_mutex;

    double distance(const std::vector<float> *v1, const std::vector<float> *v2, int layer = -1) {
        if (v1->size() != v2->size()) {
            throw std::runtime_error("distance: vectors sizes do not match");
        }
        this->distance_calculation_count++;
        return l2_distance_avx256(*v1, *v2);
    }


    VAMANA(double alpha, int L, int R) {
        this->alpha = alpha;
        this->L = L;
        this->R = R;
        this->distance_calculation_count = 0;
        this->enter_point = 0;
    }

    std::tuple<double, int, int> get_graph_parameters() {
        return std::make_tuple(this->alpha, this->L, this->R);
    }

    void print_graph_parameters() {
        std::cout << "alpha: " << this->alpha << std::endl;
        std::cout << "L: " << this->L << std::endl;
        std::cout << "R: " << this->R << std::endl;
    }

    void print_degree_distribution() {
        std::unordered_map<index_t, index_t> degree_to_count;
        for (auto neighbors: this->id_to_neighbors) {
            degree_to_count[neighbors.size()]++;
        }
        std::vector<std::pair<index_t, index_t>> degree_to_count_vector;
        for (auto degree_count: degree_to_count) {
            degree_to_count_vector.push_back(degree_count);
        }
        std::sort(degree_to_count_vector.begin(), degree_to_count_vector.end());
        std::cout << "degree distribution: {";
        for (auto degree_count: degree_to_count_vector) {
            std::cout << degree_count.first << ": " << degree_count.second << ", ";
        }
        std::cout << "}" << std::endl;
    }

    std::pair<std::vector<index_t>, std::vector<index_t>>
    greedy_search_vamana(index_t ep, const std::vector<float> &q, int k, int ef) {
        std::unordered_set<index_t> V;
        std::priority_queue<std::pair<double, index_t>> L;
        std::priority_queue<std::pair<double, index_t>> L_minus_V;
        std::unordered_set<index_t> del_L_minus_V;
        L.emplace(distance(&this->id_to_data[ep], &q), ep);
        L_minus_V.emplace(-distance(&this->id_to_data[ep], &q), ep);
        while (L_minus_V.size() > 0) {
            int p_id = L_minus_V.top().second;
            L_minus_V.pop();
            if (del_L_minus_V.find(p_id) != del_L_minus_V.end()) {
                continue;
            }
            for (auto neighbor: this->id_to_neighbors[p_id]) {
                L.emplace(distance(&this->id_to_data[neighbor], &q), neighbor);
                L_minus_V.emplace(-distance(&this->id_to_data[neighbor], &q), neighbor);
            }
            V.emplace(p_id);
            del_L_minus_V.emplace(p_id);
            while (L.size() > ef) {
                // update L to retain closest L points to q
                L.pop();
            }
        }

        // get k top elements from L, L is max heap
        std::vector<index_t> ids;
        while (L.size() > k) {
            L.pop();
        }
        while (!L.empty()) {
            ids.push_back(L.top().second);
            L.pop();
        }
        std::reverse(ids.begin(), ids.end());
        // convert V to vector
        std::vector<index_t> v_vector;
        for (int id: V) {
            v_vector.push_back(id);
        }
        return std::make_pair(ids, v_vector);

    }

    std::pair<std::vector<index_t>, std::vector<index_t>>
    greedy_search(index_t ep, const std::vector<float> &q, int k, int ef) {
        double d = distance(&this->id_to_data[ep], &q);
        std::unordered_set<index_t> v{ep};                          // set of visited elements
        std::priority_queue<std::pair<double, index_t>> candidates; // set of candidates
        std::priority_queue<std::pair<double, index_t>> w;          // dynamic list of found nearest neighbors
        candidates.emplace(-d, ep);
        w.emplace(d, ep);

        while (!candidates.empty()) {
            index_t c = candidates.top().second; // extract nearest element from c to q
            double c_dist = candidates.top().first;
            candidates.pop();
            index_t f = w.top().second; // get furthest element from w to q
            double f_dist = w.top().first;
            if (-c_dist > f_dist) {
                break;
            }
            for (index_t e: this->id_to_neighbors[c]) {
                if (v.find(e) == v.end()) {
                    v.emplace(e);
                    this->id_to_parent[e] = c; // record paren
                    f_dist = w.top().first;
                    double distance_e_q = distance(&this->id_to_data[e], &q);
                    if (distance_e_q < f_dist || w.size() < ef) {
                        candidates.emplace(-distance_e_q, e);
                        w.emplace(distance_e_q, e);
                        if (w.size() > ef) {
                            w.pop();
                        }
                    }
                }
            }
        }
        std::vector<index_t> ids;
        while (w.size() > k) {
            w.pop();
        }
        while (!w.empty()) {
            ids.push_back(w.top().second);
            w.pop();
        }
        std::reverse(ids.begin(), ids.end());
        // get k top elements from w
        int i = 0;
        while (!w.empty() && i < k) {
            ids.push_back(w.top().second);
            w.pop();
            i++;
        }
        // convert v to vector
        std::vector<index_t> v_vector;
        for (index_t node: v) {
            v_vector.push_back(node);
        }
        return std::make_pair(ids, v_vector);

    }

    index_t find_medoid() {
        std::vector<float> center = std::vector<float>(this->id_to_data[0].size(), 0);
        for (index_t i = 0; i < this->id_to_data.size(); i++) {
            for (index_t j = 0; j < this->id_to_data[i].size(); j++) {
                center[j] += this->id_to_data[i][j];
            }
        }
        for (index_t j = 0; j < center.size(); j++) {
            center[j] /= this->id_to_data.size();
        }
        std::vector<double> distances;
        for (index_t i = 0; i < this->id_to_data.size(); i++) {
            distances.push_back(distance(&(this->id_to_data[i]), &center));
        }
        index_t min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
        return min_index;
    }

    void robust_prune(index_t p, const std::vector<index_t> &V, double alpha, int R) {
        // convert vector to set
        std::unordered_set<index_t> V_set(V.begin(), V.end());
        for (index_t n: this->id_to_neighbors[p]) {
            V_set.emplace(n);
        }
        robust_prune(p, V_set, alpha, R);
    }

    void robust_prune(index_t p, std::unordered_set<index_t> V, double alpha, int R) {
        std::unordered_set<index_t> new_neighbors;
        {
            for (index_t n: this->id_to_neighbors[p]) {
                V.emplace(n);
            }
        }
        // remove p from V
        V.erase(p);

        std::unordered_map<index_t, double> distances;
        for (index_t v: V) {
            distances[v] = distance(&(this->id_to_data[p]), &(this->id_to_data[v]));
        }
        // convert V_set to vector
        std::vector<index_t> V_vector(V.begin(), V.end());
        // sort V by distance to p
        std::sort(V_vector.begin(), V_vector.end(), [&distances](index_t a, index_t b) {
            return distances[a] > distances[b];
        });

        std::unordered_set<index_t> removed;
        while (!V_vector.empty()) {
            // pick the node with the minimum distance to p
            index_t p_star = V_vector.back();
            if (removed.find(p_star) != removed.end()) {
                V_vector.pop_back();
                continue;
            }
            // add p* to p's neighbors
            new_neighbors.emplace(p_star);
            if (new_neighbors.size() >= R) {
                break;
            }
            for (index_t p_prime: V_vector) {
                if (alpha * distance(&(this->id_to_data[p_star]), &(this->id_to_data[p_prime])) <= distances[p_prime]) {
                    // remove p' from V
                    removed.emplace(p_prime);
                }
            }
        }
        {
            this->id_to_neighbors[p].clear();
            this->id_to_neighbors[p] = std::vector<index_t>(new_neighbors.begin(), new_neighbors.end());
        }
    }

    std::vector<index_t> knn_search(const std::vector<float> &q, int k, int ef) {
        return greedy_search(this->enter_point, q, k, ef).first;
    }

    void build_graph(const std::vector<std::vector<float>> &input) {
        std::cout << "building graph" << std::endl;
        this->id_to_data = std::move(input);
        this->id_to_neighbors = std::vector<std::vector<index_t>>(input.size());
        this->id_to_parent = std::vector<index_t>(input.size());
        this->id_to_neighbors_mutex = std::vector<std::shared_mutex>(input.size());


        // initialize G to a random R-regular directed graph
        std::mt19937 rng;
        rng.seed(42);
        std::uniform_int_distribution<index_t> dist(0, this->id_to_data.size() - 1);
        for (index_t i = 0; i < this->id_to_data.size(); i++) {
            while (this->id_to_neighbors[i].size() < this->R) {
                index_t random_index = dist(rng);
                if (random_index != i &&
                    std::find(this->id_to_neighbors[i].begin(), this->id_to_neighbors[i].end(), random_index) ==
                    this->id_to_neighbors[i].end()) {
                    this->id_to_neighbors[i].emplace_back(random_index);
                }
            }
        }
        std::cout << "  graph randomly initialized" << std::endl;

        this->enter_point = find_medoid();
        std::cout << "  found medoid " << this->enter_point << std::endl;
        // create threads double the number of cores
        size_t num_threads = std::thread::hardware_concurrency() * 2;
        num_threads = 1;
        std::vector<std::thread> threads;
        std::atomic<size_t> progress(0);
        for (size_t t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t]() {
                const size_t start = t * this->id_to_data.size() / num_threads;
                const size_t end = (t + 1) * this->id_to_data.size() / num_threads;
                for (index_t i = start; i < end; i++) {
                    std::vector<index_t> V;
                    {
                        // call greedy search and get L and V
                        V = greedy_search(this->enter_point, this->id_to_data[i], 1,
                                          this->L).second;
                    }

                    // call robust prune and update neighbors
                    robust_prune(i, V, this->alpha, this->R);
                    std::vector<index_t> neighbors_i;
                    {
                        neighbors_i = this->id_to_neighbors[i];
                    }
                    for (index_t j: neighbors_i) {
                        {
                            if (std::find(this->id_to_neighbors[j].begin(), this->id_to_neighbors[j].end(), i) ==
                                this->id_to_neighbors[j].end()) {
                                this->id_to_neighbors[j].emplace_back(i);
                            }
                        }
                        {
                            if (this->id_to_neighbors[j].size() > this->R) {
                                robust_prune(j, this->id_to_neighbors[j], this->alpha, this->R);
                            }
                        }
                    }
                    progress++;
                    log_progress(progress + 1, this->id_to_data.size());
                }
            });
        }
        for (auto &t: threads) {
            t.join();
        }

//        for (index_t i = 0; i < this->id_to_data.size(); i++) {
//            // call greedy search and get L and V
//            std::vector<index_t> V = greedy_search(this->enter_point, this->id_to_data[i], 1, this->L).second;
//            robust_prune(i, V, this->alpha, this->R);
//            for (index_t j: this->id_to_neighbors[i]) {
//                // this->id_to_neighbors[j].emplace(i);
//                std::unordered_set<index_t> N_out_j = this->id_to_neighbors[j];
//                N_out_j.emplace(j);
//                if (N_out_j.size() > this->R) {
//                    robust_prune(j, N_out_j, this->alpha, this->R);
//                } else {
//                    this->id_to_neighbors[j].emplace(i);
//                }
//            }
//            log_progress(i + 1, this->id_to_data.size());
//        }
    }


    void save_graph(std::string filename) {
        std::ofstream file;
        file.open(filename, std::ios::out | std::ios::binary);
        file.write((char *) &this->enter_point, 4);
        int num_nodes = this->id_to_data.size();
        file.write((char *) &num_nodes, 4);
        for (index_t i = 0; i < this->id_to_data.size(); i++) {
            index_t id = i;
            file.write((char *) &id, 4);
            index_t num_neighbors = this->id_to_neighbors[i].size();
            file.write((char *) &num_neighbors, 4);
            for (index_t neighbor: this->id_to_neighbors[i]) {
                file.write((char *) &neighbor, 4);
            }
        }
        file.close();
    }

    void load_graph(std::string filename, const std::vector<std::vector<float>> &base_load) {
        this->id_to_data = base_load;
        this->id_to_parent = std::vector<index_t>(this->id_to_data.size());
        this->id_to_neighbors = std::vector<std::vector<index_t>>(this->id_to_data.size());
        std::ifstream file;
        file.open(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "ERROR: could not open file" << std::endl;
            exit(1);
        }
        std::cout << "file opened" << std::endl;
        file.read((char *) &this->enter_point, 4);
        index_t num_nodes = 0;
        file.read((char *) &num_nodes, 4);
        if (num_nodes != this->id_to_data.size()) {
            std::cout << "ERROR: number of nodes in graph does not match number of nodes in base" << std::endl;
            std::cout << "  num nodes in graph: " << num_nodes << std::endl;
            std::cout << "  num nodes in base: " << this->id_to_data.size() << std::endl;
            exit(1);
        }
        std::cout << "num nodes: " << num_nodes << std::endl;
        for (index_t i = 0; i < num_nodes; i++) {
            index_t id;
            file.read((char *) &id, 4);
            index_t num_neighbors;
            file.read((char *) &num_neighbors, 4);
            for (int j = 0; j < num_neighbors; j++) {
                index_t neighbor_id;
                file.read((char *) &neighbor_id, 4);
                this->id_to_neighbors[id].emplace_back(neighbor_id);
            }
        }
        file.close();
    }

};

