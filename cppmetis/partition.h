#pragma once
#include "types.h"
#include "utils.h"
#include <vector>

namespace cppmetis
{
    /**
     * @brief metis partition wrapper
     *
     * @param num_partition: number of partitions in the graph
     * @param num_iteration: number of iterations for
     * @param num_initpart: number of initial partitions
     * @param obj_cut: use vol / cut
     * @param vtxdist: number of vertex in each partition
     * @param indptr: local indptr for this rank
     * @param indices: local indices for this rank
     * @param node_weight: local node weight for this rank
     * @param edge_weight: local edge weights for this rank
     * @return std::vector<idx_t> local partition map
     */
    std::vector<idx_t> metis_assignment(int64_t num_partition,
                                                int64_t num_iteration,
                                                int64_t num_initpart,
                                                float unbalance_val,
                                                bool obj_cut,
                                                std::span<idx_t> vtxdist,
                                                std::span<idx_t> indptr,
                                                std::span<idx_t> indices,
                                                std::span<WeightType> node_weight,
                                                std::span<WeightType> edge_weight);

    inline std::vector<idx_t> metis_assignment(const Args& args,
                                                const DatasetPtr& dataset) {
        return metis_assignment(args.num_partition, args.num_iteration, args.num_init_part, args.unbalance_val, args.use_cut, dataset->vtxdist,
                                dataset->indptr, dataset->indices, dataset->node_weight, dataset->edge_weight);
    };
} // namespace cppmetis
