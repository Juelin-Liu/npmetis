#pragma once
#include "types.h"
#include "timer.h"
#include "cnpy.h"

#include <tuple>
#include <vector>
#include <memory>
#include <string>

namespace pmetis
{
    struct Dataset
    {
        std::vector<IndptrType> vtxdist;
        std::vector<VertexIDType> indptr;
        std::vector<IndptrType> indices;
        std::vector<WeightType> node_weight;
        std::vector<WeightType> edge_weight;
    };

    struct Args
    {
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

    Args parse_args(int argc, const char **argv);
    std::unique_ptr<Dataset> load_dataset(const Args& args);

    std::vector<IndptrType> expand_indptr(std::span<IndptrType> indptr);
    std::vector<IndptrType> compact_indptr(std::span<IndptrType> in_indptr,
                                           std::span<uint8_t> flag);

    std::tuple<std::vector<IndptrType>, std::vector<VertexIDType>, std::vector<WeightType>> make_sym(
        std::span<IndptrType> in_indptr,
        std::span<VertexIDType> in_indices,
        std::span<WeightType> in_data);

}
