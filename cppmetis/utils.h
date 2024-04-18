#pragma once
#include "timer.h"
#include "types.h"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace cppmetis
{
    Args parse_args(int argc, const char **argv);
    DatasetPtr load_dataset(const Args &args);
    DatasetPtr make_sym(const DatasetPtr &dataset);
    DatasetPtr get_local_data(const Args &args, int rank, int world_size);

    std::vector<idx_t> expand_indptr(std::span<idx_t> indptr);
    std::vector<idx_t> compact_indptr(std::span<idx_t> in_indptr,
                                      std::span<uint8_t> flag);

    std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<WeightType>> remove_zero_weight_edges(std::span<idx_t> indptr,
                                                                                                         std::span<idx_t> indices,
                                                                                                         std::span<WeightType> edge_weight);
    int64_t non_zero_edges(std::span<idx_t> edge_weight);

    std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<WeightType>> make_sym(
        std::span<idx_t> in_indptr,
        std::span<idx_t> in_indices,
        std::span<WeightType> in_data);

    // helper functions for distributed version

    // balance number of nodes / edges in each parition
    std::vector<idx_t> get_vtx_dist(std::span<idx_t> indptr, int64_t num_edges, int world_size, bool balance_edges = true);
}
