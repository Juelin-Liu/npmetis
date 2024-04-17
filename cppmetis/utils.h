#pragma once
#include "types.h"
#include "timer.h"
#include "cnpy.h"

#include <tuple>
#include <vector>
#include <memory>
#include <string>

namespace cppmetis
{
    Args parse_args(int argc, const char **argv);
    DatasetPtr load_dataset(const Args& args);
    std::vector<idx_t> expand_indptr(std::span<idx_t> indptr);
    std::vector<idx_t> compact_indptr(std::span<idx_t> in_indptr,
                                           std::span<uint8_t > flag);

    DatasetPtr remove_zero_weight_edges(const DatasetPtr& dataset);

    std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<WeightType>> make_sym(
        std::span<idx_t> in_indptr,
        std::span<idx_t> in_indices,
        std::span<WeightType> in_data);

    // helper functions for distributed version
    std::vector<idx_t> get_vtx_dist(const DatasetPtr &data, int world_size, bool balance_edge = true);
    DatasetPtr get_local_data(const DatasetPtr& global, int rank, int world_size);
}
