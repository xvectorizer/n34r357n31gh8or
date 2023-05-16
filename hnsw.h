#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <map>
#include <random>
#include <tuple>
#include "util.h"
#include "space.h"
#include "typedef.h"


class HNSW {
public:
    // hyper parameters
    int m;                                   // number of neighbors to connect in algo1
    int m_max;                               // limit maximum number of neighbors in algo1
    int m_max_0;                             // limit maximum number of neighbors at layer0 in algo1
    int ef_construction;                     // size of dynamic candidate list
    double ml;                               // normalization factor for level generation
    std::string select_neighbors_mode;       // select which select neighbor algorithm to use

    // statistics
    unsigned long long int distance_calculation_count;           // count number of calling distance function
    unsigned long long int distance_calculation_count_0;
    unsigned long long int distance_calculation_count_1_to_inf;

    // graph
    index_t enter_point;
    std::vector<std::vector<float>> id_to_data;
    std::vector<std::vector<std::vector<index_t>>> id_to_level_to_neighbors;
    std::unordered_map<index_t, std::unordered_map<index_t, std::unordered_map<index_t, uint32_t>>> edge_to_frequency;
    std::map<std::vector<index_t>, uint32_t> new_edges_to_suggestion;
    std::vector<index_t> id_to_parent;
    std::vector<uint32_t> id_to_next;

    // rand
    std::mt19937 rng;
    std::uniform_real_distribution<double> unif_0_1;

    double distance(const std::vector<float> &v1, const std::vector<float> &v2, uint32_t level = -1) {
        if (v1.size() != v2.size()) {
            throw std::runtime_error("distance: vectors sizes do not match");
        }

        this->distance_calculation_count++;
        if (level == 0) {
            this->distance_calculation_count_0++;
        } else {
            this->distance_calculation_count_1_to_inf++;
        }

        return l2_distance_avx256(v1, v2);
    }

    HNSW(int m, int m_max, int m_max_0, int ef_construction, double ml, const std::string &select_neighbors_mode) {
        this->m = m;
        this->m_max = m_max;
        this->m_max_0 = m_max_0;
        this->ef_construction = ef_construction;
        this->ml = ml;
        this->select_neighbors_mode = select_neighbors_mode;
        this->distance_calculation_count = 0;
        this->distance_calculation_count_0 = 0;
        this->distance_calculation_count_1_to_inf = 0;
        this->enter_point = 0;
        this->unif_0_1 = std::uniform_real_distribution<double>(0.0000001, 0.9999999);
        this->rng.seed(42);

        if (select_neighbors_mode != "simple" && select_neighbors_mode != "heuristic") {
            throw std::runtime_error("select_neighbors_mode must be simple or heuristic");
        }
    }

    std::tuple<int, int, int, int, float, std::string> get_graph_parameters() {
        return std::make_tuple(m, m_max, m_max_0, ef_construction, ml, select_neighbors_mode);
    }

    void print_graph_parameters() {
        std::cout << "m=" << this->m << ", m_max=" << this->m_max << ", m_max_0=" << this->m_max_0
                  << ", ef_construction="
                  << this->ef_construction << ", ml=" << this->ml << ", select_neighbor=" << this->select_neighbors_mode
                  << std::endl;
    }

    void print_frequency() {
        std::unordered_map<int, std::unordered_map<int, int>> layer_to_frequency_to_count;
        for (auto &item: this->edge_to_frequency) {
            for (auto &item2: item.second) {
                for (auto &item3: item2.second) {
                    layer_to_frequency_to_count[item3.first][item3.second]++;
                    layer_to_frequency_to_count[-1][item3.second]++;
                }
            }
        }
        std::vector<int> layers;
        for (auto &item: layer_to_frequency_to_count) {
            layers.push_back(item.first);
        }
        std::sort(layers.begin(), layers.end());
        std::cout << "{";
        for (int l: layers) {
            std::cout << l << ":{";
            std::vector<uint32_t> frequencies;
            for (auto &item: layer_to_frequency_to_count[l]) {
                frequencies.push_back(item.first);
            }
            std::sort(frequencies.begin(), frequencies.end());
            for (uint32_t f: frequencies) {
                std::cout << f << ":" << layer_to_frequency_to_count[l][f] << ",";
            }
            std::cout << "}," << std::endl;
        }
        std::cout << "}" << std::endl;
        // print highest level
        std::cout << "highest level:" << this->id_to_level_to_neighbors[this->enter_point].size() - 1 << std::endl;
    }

    void visualize_frequency() {
        for (int l = 0; l < this->id_to_level_to_neighbors[this->enter_point].size(); l++) {
            std::cout << "level " << l << std::endl;
            for (index_t i = 0; i < this->id_to_data.size(); i++) {
                if (this->id_to_level_to_neighbors[i].size() - 1 >= l) {
                    std::cout << "node " << i << ": ";
                    std::cout << this->id_to_level_to_neighbors[i][l].size() << "    ";
                    for (index_t j: this->id_to_level_to_neighbors[i][l]) {
                        std::cout << this->edge_to_frequency[i][j][l] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    void build_graph(const std::vector<std::vector<float>> &input) {
        std::cout << "building graph" << std::endl;
        this->id_to_data = std::move(input);
        this->id_to_level_to_neighbors.resize(input.size());
        this->id_to_level_to_neighbors[0].resize(1);
        this->id_to_parent.resize(input.size());
        this->id_to_next.resize(input.size());

        // start from 1 because 0 is already inserted
        for (index_t i = 1; i < input.size(); i++) {
            insert(i);
            log_progress(i + 1, input.size());
        }
    }

    void insert(index_t q) {
        std::priority_queue<std::pair<double, index_t>> w;
        index_t ep = this->enter_point;
        int l = this->id_to_level_to_neighbors[ep].size() - 1;
        int l_new = floor(-log(unif_0_1(rng)) * this->ml);

        // update node to level l_new
        this->id_to_level_to_neighbors[q].resize(l_new + 1);

        for (int lc = l; lc > l_new; lc--) {
            w = search_layer_old(this->id_to_data[q], ep, 1, lc);
            ep = w.top().second; // ep = nearest element from W to q
        }

        for (int lc = std::min(l, l_new); lc >= 0; lc--) {
            w = search_layer_old(this->id_to_data[q], ep, this->ef_construction, lc);
            std::vector<index_t> neighbors;
            if (select_neighbors_mode == "simple") {
                neighbors = select_neighbors_simple(w, this->m);
            } else if (select_neighbors_mode == "heuristic") {
                neighbors = select_neighbors_heuristic(q, w, this->m, lc, true, true);
            } else {
                throw std::runtime_error("select_neighbors_mode should be simple/heuristic");
            }

            // add bidirectional connections from neighbors to q at layer lc
            for (index_t e: neighbors) {
                this->id_to_level_to_neighbors[e][lc].emplace_back(q);
                this->id_to_level_to_neighbors[q][lc].emplace_back(e);
            }

            // shrink connections if needed
            for (index_t e: neighbors) {
                // if lc = 0 then m_max = m_max_0
                int m_effective = lc == 0 ? this->m_max_0 : this->m_max;

                std::vector<index_t> e_conn = this->id_to_level_to_neighbors[e][lc];
                if (e_conn.size() > m_effective) // shrink connections if needed
                {
                    std::vector<index_t> e_new_conn;
                    if (select_neighbors_mode == "simple") {
                        e_new_conn = select_neighbors_simple(e, e_conn, m_effective);
                    } else if (select_neighbors_mode == "heuristic") {
                        e_new_conn = select_neighbors_heuristic(e, e_conn, m_effective, lc, true, true);
                    } else {
                        throw std::runtime_error("select_neighbors_mode should be simple/heuristic");
                    }
                    this->id_to_level_to_neighbors[e][lc] = e_new_conn;  // set neighborhood(e) at layer lc to e_new_conn
                }
            }
            ep = w.top().second;
        }
        if (l_new > l) {
            this->enter_point = q;
        }
    }

    std::priority_queue<std::pair<double, index_t>>
    search_layer_old(const std::vector<float> &q, index_t ep, int ef, int lc) {
        double d = distance(this->id_to_data[ep], q, lc);
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
            for (index_t e: this->id_to_level_to_neighbors[c][lc]) {
                if (v.find(e) == v.end()) {
                    v.emplace(e);
                    this->id_to_parent[e] = c;   // record parent
                    f_dist = w.top().first;
                    double distance_e_q = distance(this->id_to_data[e], q, lc);
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

        std::priority_queue<std::pair<double, index_t>> min_w;
        while (!w.empty()) {
            min_w.emplace(-w.top().first, w.top().second);
            w.pop();
        }
        return min_w;
    }

    std::priority_queue<std::pair<double, index_t>>
    search_layer_grasp(const std::vector<float> &q, index_t ep, int ef, int lc, const std::vector<std::vector<bool>> &edge_mask) {
        double d = distance(this->id_to_data[ep], q, lc);
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
            for (index_t e: this->id_to_level_to_neighbors[c][lc]) {
                // find index of e in neighbors of c
                int e_idx = std::find(this->id_to_level_to_neighbors[c][lc].begin(),
                                      this->id_to_level_to_neighbors[c][lc].end(), e) -
                            this->id_to_level_to_neighbors[c][lc].begin();
                if (edge_mask[c][e_idx] == false && lc == 0) {
                    continue;
                }
                if (v.find(e) == v.end()) {
                    v.emplace(e);
                    this->id_to_parent[e] = c;   // record parent
                    f_dist = w.top().first;
                    double distance_e_q = distance(this->id_to_data[e], q, lc);
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

        std::priority_queue<std::pair<double, index_t>> min_w;
        while (!w.empty()) {
            min_w.emplace(-w.top().first, w.top().second);
            w.pop();
        }
        return min_w;
    }

    std::priority_queue<std::pair<double, index_t>>
    search_layer_new(const std::vector<float> &q, index_t ep, int ef, int lc, float x) {
        uint32_t step = 0;
        auto step_to_x = [ef, x](int step) {
            if (step < ef) {
                return x;
            } else {
                return 1.0f;
            }
        };

        std::mt19937 rng;
        std::uniform_real_distribution<float> rand_0_1(0.000001, 0.999999);

        double d = distance(this->id_to_data[ep], q, lc);
        std::unordered_set<index_t> v{ep};                                   // set of visited elements
        std::priority_queue<std::pair<double, index_t>> candidates;
        std::vector<index_t> candidates_vec;
        std::priority_queue<std::pair<double, index_t>> w;                  // dynamic list of found nearest neighbors
        candidates.emplace(-d, ep);
        w.emplace(d, ep);
        candidates_vec.push_back(ep);
        this->id_to_next[ep] = 0;

        while (!candidates.empty() && !candidates_vec.empty()) {
            index_t c;
            double c_dist;
            if (unif_0_1(rng) < step_to_x(w.size())) {
                c = candidates.top().second;
                c_dist = candidates.top().first;
                if (this->id_to_next[c] >= this->id_to_level_to_neighbors[c][lc].size()) {
                    candidates.pop();
                    continue;
                }
            } else {
                // random choose an element from candidates_vec
                size_t index = floor(rand_0_1(rng) * candidates_vec.size());
                c = candidates_vec[index];
                c_dist = 0;
                if (this->id_to_next[c] >= this->id_to_level_to_neighbors[c][lc].size()) {
                    candidates_vec[index] = candidates_vec[candidates_vec.size() - 1];
                    candidates_vec.pop_back();
                    continue;
                }
            }

            index_t f = w.top().second; // get furthest element from w to q
            double f_dist = w.top().first;
            if (-c_dist > f_dist) {
                break;
            }

            index_t e = this->id_to_level_to_neighbors[c][lc][this->id_to_next[c]];
            this->id_to_next[c]++;
            if (v.find(e) == v.end()) {
                v.emplace(e);
                this->id_to_parent[e] = c; // record parent
                f_dist = w.top().first;
                double distance_e_q = distance(this->id_to_data[e], q, lc);
                if (distance_e_q < f_dist || w.size() < ef) {
                    step++;
                    this->id_to_next[e] = 0;
                    candidates.emplace(-distance_e_q, e);
                    candidates_vec.push_back(e);
                    w.emplace(distance_e_q, e);
                    if (w.size() > ef) {
                        w.pop();
                    }
                }
            }

        }

        std::priority_queue<std::pair<double, index_t>> min_w;
        while (!w.empty()) {
            min_w.emplace(-w.top().first, w.top().second);
            w.pop();
        }
        return min_w;
    }

    std::vector<index_t> select_neighbors_simple(std::priority_queue<std::pair<double, index_t>> c, int m) {
        std::vector<index_t> neighbors;
        while (neighbors.size() < m && !c.empty()) {
            neighbors.emplace_back(c.top().second);
            c.pop();
        }
        return neighbors;
    }

    std::vector<index_t> select_neighbors_simple(index_t q, const std::vector<index_t> &c, int m) {
        std::priority_queue<std::pair<double, index_t>> w;
        for (index_t e: c) {
            w.emplace(distance(this->id_to_data[e], this->id_to_data[q]), e);
            if (w.size() > m) {
                w.pop();
            }
        }
        return select_neighbors_simple(w, m);
    }

    std::vector<index_t> select_neighbors_heuristic(index_t q, std::priority_queue<std::pair<double, index_t>> c,
                                                    int m, int lc, bool extend_candidates,
                                                    bool keep_pruned_connections) {
        std::vector<index_t> v;
        while (!c.empty()) {
            v.push_back(c.top().second);
            c.pop();
        }
        return select_neighbors_heuristic(q, v, m, lc, extend_candidates, keep_pruned_connections);
    }

    std::vector<index_t> select_neighbors_heuristic(index_t q, const std::vector<index_t> &c,
                                                    int m, int lc, bool extend_candidates,
                                                    bool keep_pruned_connections) {
        std::vector<index_t> r; // (max heap)
        std::priority_queue<std::pair<double, index_t> > w; // working queue for the candidates (min_heap)
        std::unordered_set<index_t> w_set;                 // this is to help check if e_adj is in w

        for (index_t n: c) {
            w.emplace(-distance(this->id_to_data[q], this->id_to_data[n]), n);
            w_set.emplace(n);
        }

        if (extend_candidates) {
            for (index_t e: c) {
                for (index_t e_adj: this->id_to_level_to_neighbors[e][lc]) {
                    if (w_set.find(e_adj) == w_set.end()) {
                        w.emplace(-distance(this->id_to_data[q], this->id_to_data[e_adj]), e_adj);
                        w_set.emplace(e_adj);
                    }
                }
            }
        }

        std::priority_queue<std::pair<double, index_t>> w_d; // queue for the discarded candidates
        while (!w.empty() && r.size() < m) {
            index_t e = w.top().second;
            double distance_e_q = w.top().first;
            w.pop();
            bool good = true;
            for (index_t rr: r) {
                if (distance(this->id_to_data[rr], this->id_to_data[e]) < distance_e_q) {
                    good = false;
                    break;
                }
            }
            if (r.empty() || good) {
                r.push_back(e);
            } else {
                w_d.emplace(-distance_e_q, e);
            }
            if (keep_pruned_connections) { // add some of the discarded connections from w_d
                while (!w_d.empty() && r.size() < m) {
                    r.push_back(w_d.top().second);
                    w_d.pop();
                }
            }
        }

        return r;
    }


    std::vector<index_t> knn_search_new(const std::vector<float> &q, int k, int ef, float x) {
        std::priority_queue<std::pair<double, index_t>> w;      // set for the current nearest elements
        index_t ep = this->enter_point;                         // get enter point for hnsw
        int l = this->id_to_level_to_neighbors[ep].size() - 1;  // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer_new(q, ep, 1, lc, x);
            ep = w.top().second;
        }

        w = search_layer_new(q, ep, ef, 0, x);

        std::vector<index_t> result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second);
            w.pop();
        }
        return result; // return K nearest elements from W to q
    }

    std::vector<index_t> knn_search_old(const std::vector<float> &q, int k, int ef) {
        std::priority_queue<std::pair<double, index_t>> w; // set for the current nearest elements
        index_t ep = this->enter_point;                     // get enter point for hnsw
        int l = this->id_to_level_to_neighbors[ep].size() - 1;                 // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer_old(q, ep, 1, lc);
            ep = w.top().second;
        }

        w = search_layer_old(q, ep, ef, 0);

        std::vector<index_t> result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second);
            w.pop();
        }
        return result; // return K nearest elements from W to q
    }

    void knn_search_learn(const std::vector<float> &q, int k, int ef) {
        std::priority_queue<std::pair<double, index_t>> w;         // set for the current nearest elements
        index_t ep = this->enter_point;                            // get enter point for hnsw
        int l = this->id_to_level_to_neighbors[ep].size() - 1;     // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer_old(q, ep, 1, lc);
            index_t p = w.top().second;
            if (p == ep) {
                this->edge_to_frequency[p][p][lc]++;
            }
            while (p != ep) {
                this->edge_to_frequency[this->id_to_parent[p]][p][lc]++;
                p = this->id_to_parent[p];
            }
            ep = w.top().second;
        }

        w = search_layer_old(q, ep, ef, 0);

        std::vector<int> result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second);
            w.pop();
        }

        for (int i = 0; i < result.size(); i++) {
            index_t p = result[i];
            if (p == ep) {
                this->edge_to_frequency[p][p][0]++;
            }
            while (p != ep) {
                this->edge_to_frequency[this->id_to_parent[p]][p][0]++;
                p = this->id_to_parent[p];
            }
        }
        return;
    }

    void knn_search_add_edge(const std::vector<float> &q, int k, int ef) {
        std::priority_queue<std::pair<double, index_t>> w;         // set for the current nearest elements
        index_t ep = this->enter_point;                            // get enter point for hnsw
        int l = this->id_to_level_to_neighbors[ep].size() - 1;     // top level for hnsw

        for (index_t lc = l; lc > 0; lc--) {
            w = search_layer_old(q, ep, 1, lc);
            index_t p = w.top().second;
            std::vector<index_t> path;
            while (p != ep) {
                path.push_back(p);
                p = this->id_to_parent[p];
            }
            std::reverse(path.begin(), path.end());
            for (int j = 2; j < path.size(); j++) {
                std::vector<index_t> v{path[j - 2], path[j], lc};
                this->new_edges_to_suggestion[v]++;
            }
            ep = w.top().second;
        }

        w = search_layer_old(q, ep, ef, 0);

        std::vector<index_t> result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second);
            w.pop();
        }

//        for (int i = 0; i < result.size(); i++) {
//            int id = result[i];
//            Node *p = this->nodes[id];
//            std::vector<int> path;
//            while (p != ep) {
//                path.push_back(p->id);
//                p = p->parent;
//            }
//            std::reverse(path.begin(), path.end());
//            for (int j = 2; j < path.size(); j++) {
//                std::vector v{path[j - 2], path[j], 0};
//                this->new_edges_to_suggestion[v]++;
//            }
//            w.pop();
//        }
        std::vector<index_t> v{ep, result[0], 0};
        this->new_edges_to_suggestion[v]++;
        for (int i = 1; i < result.size(); i++) {
            std::vector<index_t> v{result[i - 1], result[i], 0};
            this->new_edges_to_suggestion[v]++;
        }
    }

    std::pair<index_t, std::vector<std::pair<index_t, index_t>>>
    knn_search_grasp(const std::vector<float> &q, int k, int ef, const std::vector<std::vector<bool>> &edge_mask) {
        std::priority_queue<std::pair<double, index_t>> w;        // set for the current nearest elements
        index_t ep = this->enter_point;                           // get enter point for hnsw
        int l = this->id_to_level_to_neighbors[ep].size() - 1;    // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer_old(q, ep, 1, lc);
            ep = w.top().second;
        }

        w = search_layer_grasp(q, ep, ef, 0, edge_mask);

        index_t result = w.top().second;

        std::vector<std::pair<index_t, index_t>> edge;
        index_t p = result;
        while (p != ep) {
            edge.emplace_back(this->id_to_parent[p], p);
            p = this->id_to_parent[p];
        }
        return std::make_pair(result, edge); // return K nearest elements from W to q
    }

    std::vector<index_t> knn_search_brute_force(const std::vector<float> &q, int k) {
        std::priority_queue<std::pair<float, int> > heap;
        for (index_t i = 0; i < this->id_to_data.size(); i++) {
            float dist = l2_distance_avx256(this->id_to_data[i], q);
            heap.emplace(dist, i);
            if (heap.size() > k) {
                heap.pop();
            }
        }
        std::vector<index_t> result;
        while (!heap.empty()) {
            result.emplace_back(heap.top().second);
            heap.pop();
        }
        // result from closest to furthest
        std::reverse(result.begin(), result.end());
        return result;
    }

    // build subgrah bidirectional
    std::unordered_map<index_t, std::vector<index_t>> build_subgraph(const std::vector<index_t> nodes) {
        std::unordered_map<index_t, std::vector<index_t>> subgraph;
        for (size_t i = 0; i < nodes.size(); i++) {
            subgraph[nodes[i]] = std::vector<index_t>();
            index_t id_a = nodes[i];
            for (index_t j = 0; j < this->id_to_level_to_neighbors[id_a][0].size(); j++) {
                index_t id_b = this->id_to_level_to_neighbors[id_a][0][j];
                // check if id_b is in nodes
                if (std::find(nodes.begin(), nodes.end(), id_b) == nodes.end()) {
                    continue;
                }
                // check if a->b is already in the subgraph
                // add a->b to the subgraph if not exist
                if (std::find(subgraph[id_a].begin(), subgraph[id_a].end(), id_b) != subgraph[id_a].end()) {
                    continue;
                }
                subgraph[id_a].push_back(id_b);
            }
        }
        return subgraph;
    }

    int get_number_of_connected_components(std::unordered_map<int, std::vector<int>> subgraph) {
        std::unordered_set<int> visited;
        int count = 0;
        for (auto it = subgraph.begin(); it != subgraph.end(); it++) {
            int id = it->first;
            if (visited.find(id) == visited.end()) {
                count++;
                std::queue<int> q;
                q.push(id);
                visited.insert(id);
                while (!q.empty()) {
                    int curr = q.front();
                    q.pop();
                    for (int i = 0; i < subgraph[curr].size(); i++) {
                        int neighbor_id = subgraph[curr][i];
                        if (visited.find(neighbor_id) == visited.end()) {
                            q.push(neighbor_id);
                            visited.insert(neighbor_id);
                        }
                    }
                }
            }
        }
        return count;
    }

    void dfs(int ancester, int node, const std::unordered_map<int, std::vector<int>> &graph,
             std::unordered_set<int> &visited, std::unordered_map<int, std::vector<int>> &sccs,
             std::vector<int> &stack) {
        if (visited.find(node) != visited.end()) {
            return;
        }
        visited.insert(node);
        sccs[ancester].push_back(node);
        if (graph.find(node) != graph.end()) {
            for (auto n: graph.at(node)) {
                dfs(ancester, n, graph, visited, sccs, stack);
            }
        }
        stack.push_back(node);
    };

    std::unordered_map<int, std::vector<int>>
    find_strongly_connected_components(const std::unordered_map<int, std::vector<int>> &graph) {
        std::unordered_map<int, std::vector<int>> sccs;
        std::vector<int> stack;
        std::unordered_set<int> visited;

        for (auto node: graph) {
            dfs(node.first, node.first, graph, visited, sccs, stack);
        }

        // reverse graph
        std::unordered_map<int, std::vector<int>> reversed_graph;
        // loop through the graph keys
        for (auto &item: graph) {
            int node = item.first;
            std::vector<int> neighbors = item.second;
            for (auto neighbor: neighbors) {
                reversed_graph[neighbor].push_back(node);
            }
        }

        // dfs visit reversed graph
        visited = {};
        sccs = {};
        while (!stack.empty()) {
            int node = stack.back();
            stack.pop_back();
            if (visited.find(node) == visited.end()) {
                dfs(node, node, reversed_graph, visited, sccs, stack);
            }
        }
        return sccs;
    }


    void save_graph(std::string file_name) {
        std::fstream file;
        file.open(file_name, std::ios::out | std::ios::binary);

        file.write((char *) &this->m, 4);
        file.write((char *) &this->m_max, 4);
        file.write((char *) &this->m_max_0, 4);
        file.write((char *) &this->ml, 4);
        file.write((char *) &this->ef_construction, 4);
        if (this->select_neighbors_mode == "simple") {
            int zero = 0;
            file.write((char *) &zero, 4);
        } else if (this->select_neighbors_mode == "heuristic") {
            int one = 1;
            file.write((char *) &one, 4);
        }

        index_t num_nodes = this->id_to_data.size();
        file.write((char *) &num_nodes, 4);

        for (index_t n = 0; n < this->id_to_data.size(); n++) {
            for (uint32_t l = 0; l < this->id_to_level_to_neighbors[n].size(); l++) {
                file.write((char *) &n, 4);
                file.write((char *) &l, 4);
                size_t num_neighbors = this->id_to_level_to_neighbors[n][l].size();
                file.write((char *) &num_neighbors, 4);
                for (index_t neighbor: this->id_to_level_to_neighbors[n][l]) {
                    file.write((char *) &neighbor, 4);
                }
            }
        }
        file.close();
    }

    void save_freq(std::string file_name) {
        std::fstream file;
        file.open(file_name, std::ios_base::out | std::ios_base::binary);
        for (index_t a = 0; a < this->id_to_data.size(); a++) {
            for (int l = 0; l < this->id_to_level_to_neighbors[a].size(); l++) {
                for (index_t b: this->id_to_level_to_neighbors[a][l]) {
                    file.write((char *) &a, 4);
                    file.write((char *) &b, 4);
                    file.write((char *) &l, 4);
                    file.write((char *) &this->edge_to_frequency[a][b][l], 4);
                }
            }
        }
        file.close();
    }

    void save_new_edge_suggestion(std::string file_name) {
        std::fstream file;
        file.open(file_name, std::ios::out);
        for (const auto &p: this->new_edges_to_suggestion) {
            file << p.first[0] << " " << p.first[1] << " " << p.first[2] << " " << p.second << std::endl;
        }
        file.close();
    }

    void load_graph(std::string file_name, const std::vector<std::vector<float>> &base_load) {
        this->id_to_data.clear();
        this->id_to_data = std::move(base_load);
        this->id_to_level_to_neighbors.clear();
        this->id_to_level_to_neighbors.resize(base_load.size());
        this->id_to_parent.clear();
        this->id_to_parent.resize(base_load.size());
        this->id_to_next.clear();
        this->id_to_next.resize(base_load.size());

        this->enter_point = -1;
        this->distance_calculation_count = 0;
        this->distance_calculation_count_0 = 0;
        this->distance_calculation_count_1_to_inf = 0;

        std::fstream file;
        file.open(file_name, std::ios::in | std::ios::binary);

        // ss >> this->m >> this->m_max >> this->m_max_0 >> this->ml >> this->ef_construction
        //   >> this->select_neighbors_mode;
        file.read((char *) &this->m, 4);
        file.read((char *) &this->m_max, 4);
        file.read((char *) &this->m_max_0, 4);
        file.read((char *) &this->ml, 4);
        file.read((char *) &this->ef_construction, 4);
        int select_neighbors_mode_int;
        file.read((char *) &select_neighbors_mode_int, 4);
        if (select_neighbors_mode_int == 0) {
            this->select_neighbors_mode = "simple";
        } else if (select_neighbors_mode_int == 1) {
            this->select_neighbors_mode = "heuristic";
        }
        index_t num_nodes;
        file.read((char *) &num_nodes, 4);
        if (num_nodes != this->id_to_data.size()) {
            std::cout << "ERROR: number of nodes in graph does not match number of nodes in base" << std::endl;
            std::cout << "  num nodes in graph: " << num_nodes << std::endl;
            std::cout << "  num nodes in base: " << this->id_to_data.size() << std::endl;
            exit(1);
        }
        std::cout << "num nodes: " << num_nodes << std::endl;

        while (!file.eof()) {
            index_t node_id, node_level;
            file.read((char *) &node_id, 4);
            file.read((char *) &node_level, 4);
            if (node_level >= this->id_to_level_to_neighbors[node_id].size()) {
                this->id_to_level_to_neighbors[node_id].resize(node_level + 1);
            }
            size_t num_neighbors;
            file.read((char *) &num_neighbors, 4);
            for (size_t j = 0; j < num_neighbors; j++) {
                index_t neighbor_id;
                file.read((char *) &neighbor_id, 4);
                this->id_to_level_to_neighbors[node_id][node_level].push_back(neighbor_id);
            }
            if (this->enter_point == -1 || this->id_to_level_to_neighbors[node_id].size() >
                                           this->id_to_level_to_neighbors[this->enter_point].size()) {
                this->enter_point = node_id;
            }
        }
        file.close();
    }

    void load_freq(std::string file_name) {
        this->edge_to_frequency.clear();
        std::fstream file;
        file.open(file_name, std::ios::in | std::ios::binary);
        index_t node_id_a;
        index_t node_id_b;
        int level;
        uint32_t freq;
        while (!file.eof()) {
            // ss >> node_id_a >> node_id_b >> level >> freq;
            file.read((char *) &node_id_a, 4);
            file.read((char *) &node_id_b, 4);
            file.read((char *) &level, 4);
            file.read((char *) &freq, 4);
            this->edge_to_frequency[node_id_a][node_id_b][level] = freq;
        }
        file.close();
    }

    void delete_edge(int min, int max, int remaining_edge) {
        for (index_t i = 0; i < this->id_to_data.size(); i++) {
            for (int l = 0; l < this->id_to_level_to_neighbors[i].size(); l++) {
                std::sort(this->id_to_level_to_neighbors[i][l].begin(), this->id_to_level_to_neighbors[i][l].end(),
                          [this, &i, l](index_t a, index_t b) {
                              return this->edge_to_frequency[i][a][l] > this->edge_to_frequency[i][b][l];
                          });
            }
        }
        //remove edges in hnsw.edge_to_frequency
        for (index_t n = 0; n < this->id_to_data.size(); n++) {
            for (int l = 0; l < this->id_to_level_to_neighbors[n].size(); l++) {
                std::vector<index_t> new_neighbors;
                if (this->id_to_level_to_neighbors[n][l].size() > max) {
                    auto v = this->id_to_level_to_neighbors[n][l];
                    this->id_to_level_to_neighbors[n][l] = std::vector<index_t>(v.begin(), v.begin() + max);
                }
            }
        }

        uint32_t edge_count = 0;
        for (index_t n = 0; n < this->id_to_data.size(); n++) {
            for (int l = 0; l < this->id_to_level_to_neighbors[n].size(); l++) {
                edge_count += this->id_to_level_to_neighbors[n][l].size();
            }
        }

        uint32_t removed_edge_count = 0;
        std::unordered_set<uint32_t> frequencies;
        for (const auto &i: this->edge_to_frequency) {
            for (const auto &j: i.second) {
                for (const auto &k: j.second) {
                    frequencies.insert(k.second);
                }
            }
        }
        std::vector<int> frequency(frequencies.begin(), frequencies.end());
        std::sort(frequency.begin(), frequency.end());

        int i = 0;

        while (removed_edge_count < (edge_count - remaining_edge)) {
            for (index_t n = 0; n < this->id_to_data.size(); i++) {
                for (int l = 0; l < this->id_to_level_to_neighbors[n].size(); l++) {
                    std::vector<index_t> new_neighbors;
                    for (index_t m: this->id_to_level_to_neighbors[n][l]) {
                        if (new_neighbors.size() < min ||
                            this->edge_to_frequency[n][m][l] > frequency[i]) {
                            new_neighbors.push_back(m);
                        } else {
                            removed_edge_count++;
                        }
                    }
                    this->id_to_level_to_neighbors[n][l] = new_neighbors;
                    if (removed_edge_count >= (edge_count - remaining_edge)) {
                        return;
                    }
                }
            }
            if (i == frequency.size() - 1) {
                return;
            }
            i++;
        }
    }

//    std::vector<float> generate_new_vector() {
//        std::mt19937 rng(time(0));
//        std::uniform_int_distribution<int> dist1(0, nodes.size());
//        int idx = dist1(rng);
//        std::vector<float> v = this->nodes[idx]->data;
//        std::vector<Node *> neighbors = this->nodes[idx]->neighbors[0];
//
//        // get the minimum and maximum distance from the neighbors
//        double min_dist;
//        double max_dist;
//        for (int i = 0; i < neighbors.size(); i++) {
//            double distance_v_n = distance(&v, &(neighbors[i]->data));
//            min_dist = std::min(min_dist, distance_v_n);
//            max_dist = std::max(max_dist, distance_v_n);
//        }
//
//        // generate d dimensional unit vector uniformly distributed on the surface of the sphere
//        std::uniform_real_distribution<float> dist2(-1, 1);
//        std::vector<float> unit_vector;
//        for (int i = 0; i < nodes[idx]->data.size(); i++) {
//            unit_vector.push_back(dist2(rng));
//        }
//        // normalize the unit vector
//        float norm = 0;
//        for (int i = 0; i < unit_vector.size(); i++) {
//            norm += unit_vector[i] * unit_vector[i];
//        }
//        norm = std::sqrt(norm);
//        for (int i = 0; i < unit_vector.size(); i++) {
//            unit_vector[i] /= norm;
//        }
//
//        // generate a random distance between 0 and min_dist
//        std::uniform_real_distribution<float> dist3(0, min_dist);
//        float random_dist = dist3(rng);
//
//        // scale the unit vector by the random distance
//        for (int i = 0; i < unit_vector.size(); i++) {
//            unit_vector[i] *= random_dist;
//        }
//
//        // add the scaled unit vector to the current vector
//        for (int i = 0; i < unit_vector.size(); i++) {
//            v[i] += unit_vector[i];
//        }
//        return v;
//    }
};


