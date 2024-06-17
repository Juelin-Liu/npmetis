#pragma once
#include <vector>
#include <cstdint>
#include "common.h"
namespace pymetis
{
    /**
     * @brief Mult-threaded metis partition wrapper
     *
     * @param num_partition: number of partitions in the graph
     * @param num_iteration: number of iterations for
     * @param num_initpart: number of initial partitions
     * @param obj_cut: use vol / cut
     * @param indptr: local indptr for this rank
     * @param indices: local indices for this rank
     * @param node_weight: local node weight for this rank
     * @param edge_weight: local edge weights for this rank
     * @return std::vector<int32_t> local partition map
     */
    std::vector<uint32_t> mt_metis_assignment(int64_t num_partition,
                                              int64_t num_iteration,
                                              int64_t num_initpart,
                                              float unbalance_val,
                                              bool obj_cut,
                                              std::span<idx_t> indptr,
                                              std::span<id_t> indices,
                                              std::span<wgt_t> node_weight,
                                              std::span<wgt_t> edge_weight);

} // namespace pymetis
