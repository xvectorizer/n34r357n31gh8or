#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <string>
#include <random>
#include <tuple>

#include "hnsw.h"
#include "util.h"
#include "vamana.h"
#include "typedef.h"
#include "grasp.h"

std::vector<double>
query(HNSW &hnsw, const std::vector<std::vector<float>> &base_load, const std::vector<std::vector<float>> &query_load,
      const std::vector<std::vector<index_t>> &groundtruth_load, int k, int ef_k,
      const std::string &algo_mode, float x) {
    std::cout << std::endl << "testing: " << algo_mode << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    hnsw.distance_calculation_count = 0;
    hnsw.distance_calculation_count_1_to_inf = 0;
    hnsw.distance_calculation_count_0 = 0;
    std::vector<std::vector<index_t>> query_results;

    for (int i = 0; i < query_load.size(); i++) {
        if (algo_mode == "old") {
            query_results.emplace_back(hnsw.knn_search_old(query_load[i], k, ef_k));
        } else if (algo_mode == "new") {
            query_results.emplace_back(hnsw.knn_search_new(query_load[i], k, ef_k, x));
        }
        log_progress(i + 1, query_load.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(end - start);
    double query_time = (double) duration.count() / query_load.size();
    unsigned long long int query_count = hnsw.distance_calculation_count / query_load.size();
    std::cout << "time per query(ms): " << query_time << std::endl;
    std::cout << "distance count per query: " << query_count << std::endl;
    std::cout << "distance count_0 per query: " << hnsw.distance_calculation_count_0 / query_load.size() << std::endl;
    std::cout << "distance count_1_to_inf per query: " << hnsw.distance_calculation_count_1_to_inf / query_load.size()
              << std::endl;
    std::cout << "ef: " << ef_k << std::endl;

    // calculate recall
    std::vector<double> total_recall;
    for (int i = 0; i < query_load.size(); i++) {
        if (!groundtruth_load.empty()) {
            total_recall.emplace_back(calculate_recall(query_results[i], groundtruth_load[i], k));
        } else {
            throw std::runtime_error("query: groundtruth_load cannot be empty");
        }
    }
    double avg_recall = std::accumulate(total_recall.begin(), total_recall.end(), 0.0) / total_recall.size();
    std::cout << "recall: " << avg_recall << std::endl;
    std::vector<double> ans{query_time, (double) query_count, avg_recall};
    return ans;
}

int main(int argc, char **argv) {
    std::cout << system("whoami") << std::endl;
    std::vector<std::vector<float>> base_load;
    std::vector<std::vector<float>> query_load;
    std::vector<std::vector<float>> learn_load;
    std::vector<std::vector<index_t>> groundtruth_load;
    unsigned dim1, dim2, dim3, dim4;
    unsigned num1, num2, num3, num4;

//    load_fvecs_data("/ssd2/cs5522/dataset/gist/gist_base.fvecs", base_load, num1, dim1);
//    load_fvecs_data("/ssd2/cs5522/dataset/gist/gist_learn.fvecs", learn_load, num2, dim2);
//    load_fvecs_data("/ssd2/cs5522/dataset/gist/gist_query.fvecs", query_load, num3, dim3);
//    load_ivecs_data("/ssd2/cs5522/dataset/gist/gist_groundtruth.ivecs", groundtruth_load, num4, dim4);


//    load_fvecs_data("/ssd2/cs5522/dataset/siftsmall/siftsmall_base.fvecs", base_load, num1, dim1);
//    // load_txt_data("/ssd2/cs5522/dataset/siftsmall/siftsmall_train_generated.txt", learn_load, num2, dim2);
//    load_fvecs_data("/ssd2/cs5522/dataset/siftsmall/siftsmall_query.fvecs", query_load, num3, dim3);
//    load_ivecs_data("/ssd2/cs5522/dataset/siftsmall/siftsmall_groundtruth.ivecs", groundtruth_load, num4, dim4);

    load_fvecs_data("/ssd2/cs5522/dataset/sift/sift_base.fvecs", base_load, num1, dim1);
    load_txt_data("/ssd2/cs5522/dataset/sift/sift_train_generated.txt", learn_load, num2, dim2);
    load_fvecs_data("/ssd2/cs5522/dataset/sift/sift_query.fvecs", query_load, num3, dim3);
    load_ivecs_data("/ssd2/cs5522/dataset/sift/sift_groundtruth.ivecs", groundtruth_load, num4, dim4);

//    load_txt_data("/ssd2/cs5522/dataset/fashion-mnist/fashion-mnist_train.txt", base_load, num1, dim1);
//    load_txt_data("/ssd2/cs5522/dataset/fashion-mnist/fashion-mnist_train_generated.txt", learn_load, num2, dim2);
//    load_txt_data("/ssd2/cs5522/dataset/fashion-mnist/fashion-mnist_test.txt", query_load, num3, dim3);
    //load_txt_data("/ssd2/cs5522/dataset/fashion-mnist/fashion-mnist_groundtruth.txt",groundtruth_load, num4, dim4);

//    load_txt_data("/ssd2/cs5522/dataset/deep/deep_base.txt", base_load, num1, dim1);
//    load_txt_data("/ssd2/cs5522/dataset/deep/deep_learn.txt", learn_load, num2, dim2);
//    load_txt_data("/ssd2/cs5522/dataset/deep/deep_query.txt", query_load, num3, dim3);
//    load_txt_data("/ssd2/cs5522/dataset/deep/deep_groundtruth.txt", groundtruth_load, num4, dim4);

    std::cout << "base_num：" << num1 << std::endl
              << "base dimension：" << dim1 << std::endl;
    std::cout << "learn_num：" << num2 << std::endl
              << "learn dimension：" << dim2 << std::endl;
    std::cout << "query_num：" << num3 << std::endl
              << "query dimension：" << dim3 << std::endl;
    std::cout << "groundtruth_num：" << num4 << std::endl
              << "groundtruth dimension：" << dim4 << std::endl;



    // prepare csv file to write
    std::string file_name = "/home/cs5522/code/result/vamana_sift.csv";
    std::fstream output_file(file_name, std::ios_base::app);
    HNSW hnsw = HNSW(16, 16, 32, 300, 1.0, "simple");
    //hnsw.build_graph(base_load);
    hnsw.load_graph("/ssd2/cs5522/hnsw/graph/sift/graph_m=16_mmax=16_mmax0=32_ef=300_ml=1.0", base_load);

    int k = 10;
    int ef_k_learn = 100;

    float eta = 0.1;
    float beta = 0.8;
    float sigma = 0.7;
    float T0 = 1;
    int K = 10;


    grasp_algorithm1(hnsw, base_load, eta, T0, beta, sigma, K, ef_k_learn);

    //hnsw.load_graph("/ssd2/cs5522/hnsw/graph/sift/graph_m=16_mmax=16_mmax0=32_ef=300_ml=1.0_grasp_sigma=0.7_beta=0.8_eta=0.1_T0=1_K=20", base_load);

    int interval = 15;
    int end = 300;

//    VAMANA vamana = VAMANA(1.03, 125, 32);
//    vamana.build_graph(base_load);
//    std::cout << "graph build finished" << std::endl;
//    vamana.save_graph("/ssd2/cs5522/vamana/graph/sift/graph_alpha=1.03_L=125_R=32_l2");
//
//    //vamana.save_graph()
//    vamana.print_degree_distribution();
//
//    int interval = 20;
//    int end = 500;
//    for (int ef_k = 10; ef_k <= end; ef_k += interval) {
//        vamana.distance_calculation_count = 0;
//        std::vector<std::vector<index_t>> query_results;
//        for (index_t i = 0; i < query_load.size(); i++) {
//            query_results.emplace_back(vamana.knn_search(query_load[i], k, ef_k));
//            log_progress(i + 1, query_load.size());
//        }
//        // calculate recall
//        std::vector<double> total_recall;
//        for (index_t i = 0; i < query_load.size(); i++) {
//            if (!groundtruth_load.empty()) {
//                total_recall.emplace_back(calculate_recall(query_results[i], groundtruth_load[i]));
//            } else {
//                throw std::runtime_error("query: groundtruth_load cannot be empty");
//            }
//        }
//        double avg_recall = std::accumulate(total_recall.begin(), total_recall.end(), 0.0) / total_recall.size();
//        std::cout << "recall: " << avg_recall << std::endl;
//        long long total_distance_calculation = vamana.distance_calculation_count;
//        int avg_distance_calculation = total_distance_calculation / query_load.size();
//        // write to csv file
//        output_file << ef_k << "," << avg_recall << "," << avg_distance_calculation << std::endl;
//    }
//
//    return 0;
//}
//
    auto rng = std::default_random_engine{};
    rng.seed(999);
    float x = 1;

    std::vector<std::pair<std::string, int>> strategy;
    strategy.emplace_back("/", 0);
    //strategy.emplace_back("+", 0);
    //strategy.emplace_back("+", 34000000);
    int learn_size = 2000000;
//    strategy.emplace_back("-", 1000000);
//    strategy.emplace_back("-", 800000);
//    strategy.emplace_back("-", 600000);

    for (int iteration = 0; iteration < strategy.size(); iteration++) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(learn_load.begin(), learn_load.end(), g);
        if (strategy[iteration].first == "-") {
            // learn frequency count with learn and base vectors
            hnsw.edge_to_frequency.clear();
            std::cout << "learning base vectors..." << std::endl;
            for (size_t i = 0; i < base_load.size(); i++) {
                hnsw.knn_search_learn(base_load[i], 1, ef_k_learn);
                log_progress(i + 1, base_load.size());
            }
            std::cout << "learning generated vectors..." << std::endl;
            for (size_t i = 0; i < learn_size; i++) {
                hnsw.knn_search_learn(learn_load[i], 1, ef_k_learn);
                log_progress(i + 1, learn_size);
            }
            //remove edges in hnsw.edge_to_frequency
            // hnsw.delete_edge(0, 32, strategy[iteration].second);
            std::cout << "removing edges..." << std::endl;
            for (const auto &i : hnsw.edge_to_frequency) {
                for (const auto &j : i.second) {
                    for (const auto &k : j.second) {
                        if (k.second == 0) {
                            // remove edge from i to j at level k.first
                            std::vector<index_t> &v = hnsw.id_to_level_to_neighbors[i.first][k.first];
                            v.erase(std::remove(v.begin(), v.end(), j.first), v.end());
                        }
                    }
                }
            }
            std::stringstream ss;
            ss << "/ssd2/cs5522/hnsw/graph/sift/graph_m=16_mmax=16_mmax0=32_ef=300_ml=1.0_iter=" << iteration;
            hnsw.save_graph(ss.str());
        }
        if (strategy[iteration].first == "+") {
            std::cout << "adding new edges" << std::endl;
            for (int i = 0; i < base_load.size(); i++) {
                hnsw.knn_search_add_edge(base_load[i], 1, ef_k_learn);
                log_progress(i, base_load.size());
            }
            for (int i = 0; i < learn_size; i++) {
                hnsw.knn_search_add_edge(learn_load[i], 1, ef_k_learn);
                log_progress(i, learn_size);
            }

//            std::map<int, int> frequency;
//            for (const auto &i : hnsw.new_edges_to_suggestion) frequency[i.second]++;
//            std::vector<int> x;
//            std::vector<int> y;
//            for (const auto &i : frequency) x.push_back(i.first);
//            std::sort(x.begin(), x.end());
//            for (int i : x) y.push_back(frequency[i]);
//            int elbow_x = select_elbow(x, y);
//            elbow_x = -1;
            uint32_t added_edge_count = 0;
            for (const auto &p: hnsw.new_edges_to_suggestion) {
//                if (p.second <= elbow_x) continue;
                index_t a_id = p.first[0];
                index_t b_id = p.first[1];
                index_t level = p.first[2];

                auto v = hnsw.id_to_level_to_neighbors[a_id][level];

                if (std::find(v.begin(), v.end(), b_id) == v.end()) {
                    hnsw.id_to_level_to_neighbors[a_id][level].push_back(b_id);
                    added_edge_count++;
                }
            }
            std::cout << "added_edge_count: " << added_edge_count << " "
                      << (hnsw.new_edges_to_suggestion.size() == added_edge_count) << std::endl;
            hnsw.new_edges_to_suggestion.clear();
            std::stringstream ss;
            ss << "/ssd2/cs5522/hnsw/graph/sift/graph_m=16_mmax=16_mmax0=32_ef=300_ml=1.0_iter=" << iteration;
            hnsw.save_graph(ss.str());
        }

        std::cout << "iteration: " << iteration << std::endl;
        std::map<int, std::vector<double>> ef_to_old_result;
        std::map<int, std::vector<double>> ef_to_new_random_result;
        std::map<int, std::vector<double>> ef_to_new_result;

        for (int ef_k = 10; ef_k <= end; ef_k += interval) {
            std::vector<double> old_results = query(hnsw, base_load, query_load, groundtruth_load, k, ef_k, "old", 1);
            ef_to_old_result[ef_k] = old_results;
        }

        // shuffle neighbor order
        for (index_t i = 0; i < hnsw.id_to_data.size(); i++) {
            for (int l = 0; l < hnsw.id_to_level_to_neighbors[i].size(); l++) {
                std::shuffle(std::begin(hnsw.id_to_level_to_neighbors[i][l]),
                             std::end(hnsw.id_to_level_to_neighbors[i][l]), rng);
            }
        }


        // sort neighbors based on frequency (reverse order)
//        for (Node *i: hnsw.nodes) {
//            for (int l = 0; l < i->neighbors.size(); l++) {
//                std::sort(i->neighbors[l].begin(), i->neighbors[l].end(),
//                          [&hnsw, &i, l](Node *a, Node *b) {
//                              return hnsw.edge_to_frequency[i->id][a->id][l] <
//                                     hnsw.edge_to_frequency[i->id][b->id][l];
//                          });
//            }
//        }
        for (int ef_k = 10; ef_k <= end; ef_k += interval) {
            std::vector<double> new_random_results = query(hnsw, base_load, query_load, groundtruth_load, k, ef_k,
                                                           "new", x);
            ef_to_new_random_result[ef_k] = new_random_results;
        }

        // sort neighbors based on frequency
        for (index_t i = 0; i < hnsw.id_to_data.size(); i++) {
            for (int l = 0; l < hnsw.id_to_level_to_neighbors[i].size(); l++) {
                std::sort(hnsw.id_to_level_to_neighbors[i][l].begin(), hnsw.id_to_level_to_neighbors[i][l].end(),
                          [&hnsw, &i, l](index_t a, index_t b) {
                              return hnsw.edge_to_frequency[i][a][l] > hnsw.edge_to_frequency[i][b][l];
                          });
            }
        }
        for (int ef_k = 10; ef_k <= end; ef_k += interval) {
            std::vector<double> new_result = query(hnsw, base_load, query_load, groundtruth_load, k, ef_k, "new", x);
            ef_to_new_result[ef_k] = new_result;
        }

        uint32_t edge_count = 0;
        for (index_t n = 0; n < hnsw.id_to_data.size(); n++) {
            for (int l = 0; l < hnsw.id_to_level_to_neighbors[n].size(); l++) {
                edge_count += hnsw.id_to_level_to_neighbors[n][l].size();
            }
        }

        std::vector<int> keys;
        for (auto it = ef_to_old_result.begin(); it != ef_to_old_result.end(); it++) {
            keys.push_back(it->first);
        }
        std::sort(keys.begin(), keys.end());
        for (int i: keys) {
            if (!ef_to_old_result.contains(i) || !ef_to_new_result.contains(i) ||
                !ef_to_new_random_result.contains(i)) {
                continue;
            }
            //output_file << k << ef_k << old_results[0] << "," << old_results[1] << "," << old_results[2] << std::endl;
            output_file << ef_to_old_result[i][0] << "," << ef_to_old_result[i][1] << "," << ef_to_old_result[i][2]
                        << "," << ef_to_new_random_result[i][0] << "," << ef_to_new_random_result[i][1] << ","
                        << ef_to_new_random_result[i][2] << "," << ef_to_new_result[i][0] << ","
                        << ef_to_new_result[i][1] << "," << ef_to_new_result[i][2] << "," << i << "," << iteration
                        << "," << edge_count << std::endl;
        }
    }
    return 0;
}
