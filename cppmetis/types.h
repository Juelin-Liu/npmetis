#pragma once

#include <cstdint>
#include <metis.h>
#include <vector>
#include <memory>
#include "tcb/span.hpp"

namespace cppmetis {
    using WeightType = idx_t;

    struct Dataset {
        std::vector<idx_t> vtxdist;
        std::vector<idx_t> indptr;
        std::vector<idx_t> indices;
        std::vector<WeightType> node_weight;
        std::vector<WeightType> edge_weight;
    };
    using DatasetPtr = std::unique_ptr<Dataset>;

    struct Args {
        int64_t num_partition;
        int64_t num_init_part;
        int64_t num_iteration;
        float unbalance_val;
        bool use_cut;
        std::string indptr_path;
        std::string indices_path;
        std::string node_weight_path;
        std::string edge_weight_path;
        std::string output_path;
    };
}

namespace std {
    using namespace tcb;
}