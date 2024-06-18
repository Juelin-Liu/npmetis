//
// Created by juelin on 6/17/24.
//

#ifndef CPPMETIS_MAKE_SYM_H
#define CPPMETIS_MAKE_SYM_H

#include <tuple>
#include <vector>
#include <cstdint>
#include "common.h"

namespace pymetis
{
        /**
        *
        * @param in_indptr input indptr
        * @param in_indices input indices
        * @param in_data input data
        * @return out_indptr, out_indices, out_data
        */
    std::tuple<std::vector<idx_t>, std::vector<id_t>, std::vector<wgt_t>> make_sym(
            std::span<idx_t> in_indptr,
            std::span<id_t> in_indices,
            std::span<wgt_t> in_data);

    std::tuple<std::vector<idx_t>, std::vector<id_t>> make_sym(
            std::span<idx_t> in_indptr,
            std::span<id_t> in_indices);
 
//     std::tuple<std::vector<metis_idx_t>, std::vector<metis_idx_t>, std::vector<metis_idx_t>> make_sym(
//             std::span<metis_idx_t> in_indptr,
//             std::span<metis_idx_t> in_indices,
//             std::span<metis_idx_t> in_data);

//     std::tuple<std::vector<metis_idx_t>, std::vector<metis_idx_t>> make_sym(
//             std::span<metis_idx_t> in_indptr,
//             std::span<metis_idx_t> in_indices);
}
#endif //CPPMETIS_MAKE_SYM_H
