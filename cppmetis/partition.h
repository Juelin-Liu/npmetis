#pragma once
#include "types.h"
#include <vector>
#include <span>

namespace pmetis
{
    std::vector<VertexPIDType> metis_assignment(int64_t num_partitions, bool obj_cut,
                                            std::span<IndptrType> indptr,
                                            std::span<VertexIDType> indices,
                                            std::span<WeightType> node_weight,
                                            std::span<WeightType> edge_weight
                                            );
} // namespace cppmetis
