#pragma once
#include "types.h"
#include "timer.h"
#include <tuple>
#include <vector>
#include <span>

namespace pmetis
{
    std::vector<IndptrType> expand_indptr(const std::span<IndptrType> indptr);
    std::vector<IndptrType> compact_indptr(const std::span<IndptrType> in_indptr,
                                           const std::span<uint8_t> flag);

    std::tuple<std::vector<IndptrType>, std::vector<VertexIDType>, std::vector<WeightType>> make_sym(
        const std::span<IndptrType> in_indptr, 
        const std::span<VertexIDType> in_indices, 
        const std::span<WeightType> in_data);
}