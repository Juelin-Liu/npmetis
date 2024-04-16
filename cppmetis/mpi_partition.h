#pragma once
#include "types.h"
#include <mpi.h>
#include <vector>

namespace pmetis
{
    /**
     * @brief MPI metis partition wrapper
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
     * @return std::vector<VertexPIDType> local partition map
     */
    std::vector<VertexPIDType> mpi_metis_assignment(int64_t num_partition,
                                                    int64_t num_iteration,
                                                    int64_t num_initpart,
                                                    bool obj_cut,
                                                    std::span<IndptrType> vtxdist,
                                                    std::span<IndptrType> indptr,
                                                    std::span<VertexIDType> indices,
                                                    std::span<WeightType> node_weight,
                                                    std::span<WeightType> edge_weight);
} // namespace cppmetis
