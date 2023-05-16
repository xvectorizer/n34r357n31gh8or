#pragma once

#include <unordered_set>
#include <sstream>
#include <iomanip>
#include "typedef.h"

template<typename T>
double calculate_recall(const std::vector<T> &sample, const std::vector<T> &base, int at) {
    std::unordered_set<T> s(base.begin(), base.begin() + at);
    uint32_t hit = 0;
    for (const auto &i : sample) {
        if (s.find(i) != s.end()) {
            hit++;
        }
    }
    return (double) hit / sample.size();
}

void load_fvecs_data(const char *filename,
                     std::vector<std::vector<float>> &results, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    // initialize results
    results.resize(num);
    for (unsigned i = 0; i < num; i++)
        results[i].resize(dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        float tmp[dim];
        in.read((char *) tmp, dim * 4);
        for (unsigned j = 0; j < dim; j++) {
            results[i][j] = (float) tmp[j];
        }
    }
    in.close();
}

void load_ivecs_data(const char *filename,
                     std::vector<std::vector<index_t>> &results, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    // initialize results
    results.resize(num);
    for (unsigned i = 0; i < num; i++)
        results[i].resize(dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        int tmp[dim];
        in.read((char *) tmp, dim * 4);
        for (unsigned j = 0; j < dim; j++) {
            results[i][j] = (int) tmp[j];
        }
    }
    in.close();
}

void load_txt_data(const char *filename, std::vector<std::vector<float>> &results, unsigned &num, unsigned &dim) {
    std::ifstream fd(filename);
    if (!fd.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    std::string temp;
    while (getline(fd, temp)) {
        std::vector<float> f;
        // split the line by space
        std::istringstream iss(temp);
        std::string token;
        while (getline(iss, token, ' ')) {
            f.push_back(stof(token));
        }
        results.push_back(f);
    }

    num = results.size();
    dim = results[0].size();
    fd.close();
}

void log_progress(int curr, int total) {
    int barWidth = 70;
    if (curr != total && curr % int(total / 100) != 0) {
        return;
    }
    float progress = (float) curr / total;
    std::cout << std::flush << "\r";
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0);

    if (curr >= total) {
        std::cout << std::endl;
    }
}


int select_elbow(std::vector<int> x, std::vector<int> y) {
    std::vector<int> d1;
    std::vector<int> d2;
    for (int i = 1; i < x.size(); i++) {
        d1.push_back((y[i-1] - y[i]) / (x[i-1] - x[i]));
    }
    for (int i = 1; i < x.size() - 1; i++) {
        d2.push_back((d1[i-1] - d1[i]) / (x[i-1] - x[i]));
    }

    int idx = std::distance(d2.begin(),std::max_element(d2.begin(), d2.end())) + 1;
    return x[idx];
}

void convert_int32_to_char_array_little_endian(int_fast32_t num, char* arr) {
    arr[0] = (num >> 0) & 0xFF;
    arr[1] = (num >> 8) & 0xFF;
    arr[2] = (num >> 16) & 0xFF;
    arr[3] = (num >> 24) & 0xFF;
}

int_fast32_t convert_char_array_to_int32_little_endian(const char* arr) {
    int_fast32_t num = 0;
    num |= (arr[0] & 0xFF) << 0;
    num |= (arr[1] & 0xFF) << 8;
    num |= (arr[2] & 0xFF) << 16;
    num |= (arr[3] & 0xFF) << 24;
    return num;
}

void print_histogram(const std::vector<float>& data) {
    const int NUM_BINS = 10;
    const int MAX_STARS = 50;

    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());

    float range = max_val - min_val;
    float bin_width = range / NUM_BINS;

    std::map<int, int> histogram;

    for (auto num : data) {
        int bin = (num - min_val) / bin_width;
        if (bin == NUM_BINS) bin--;
        histogram[bin]++;
    }

    int max_count = std::max_element(histogram.begin(), histogram.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.second < b.second;
                                     })->second;

    for (const auto& [bin, count] : histogram) {
        float bin_min = min_val + bin * bin_width;
        float bin_max = bin_min + bin_width;
        std::cout << "[" << std::fixed << std::setprecision(3) << std::setw(7) << bin_min
                  << " , " << std::setw(7) << bin_max << ") : ";
        int stars = (count * MAX_STARS) / max_count;
        for (int i = 0; i < stars; i++) {
            std::cout << "*";
        }
        std::cout << "\n";
    }
}
