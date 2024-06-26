#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include "metis_assignment.h"
#include "make_sym.h"

namespace py = pybind11;

namespace pymetis
{
    template<typename T>
    std::vector<metis_idx_t> to_64(std::vector<T> vec){
        std::vector<metis_idx_t> ret(vec.size());

        for (auto i = 0; i < vec.size(); i++){
            ret[i] = static_cast<metis_idx_t>(vec[i]);
        }

        return ret;
    }

    template<typename T>
    std::vector<metis_idx_t> to_64(std::span<T> vec){
        std::vector<metis_idx_t> ret(vec.size());

        for (auto i = 0; i < vec.size(); i++){
            ret[i] = static_cast<metis_idx_t>(vec[i]);
        }

        return ret;
    }

    py::array_t<int64_t> metis_assignment_wrapper(int64_t num_partition,
                                                      int64_t num_iteration,
                                                      int64_t num_initpart,
                                                      float unbalance_val,
                                                      bool obj_cut,
                                                      py::array_t<idx_t> indptr,
                                                      py::array_t<id_t > indices,
                                                      py::array_t<wgt_t> node_weight,
                                                      py::array_t<wgt_t> edge_weight)
    {
        py::buffer_info indptr_info = indptr.request();
        py::buffer_info indices_info = indices.request();
        py::buffer_info node_weight_info = node_weight.request();
        py::buffer_info edge_weight_info = edge_weight.request();

        if (indptr_info.ndim != 1 || indices_info.ndim != 1 ||
            node_weight_info.ndim != 1 || edge_weight_info.ndim != 1)
        {
            throw std::runtime_error("Input arrays must be 1-dimensional");
        }


        std::span<idx_t> indptr_span(static_cast<idx_t*>(indptr_info.ptr), indptr_info.size);
        std::span<id_t> indices_span(static_cast<id_t*>(indices_info.ptr), indices_info.size);
        std::span<wgt_t> node_weight_span(static_cast<wgt_t*>(node_weight_info.ptr), node_weight_info.size);
        std::span<wgt_t> edge_weight_span(static_cast<wgt_t*>(edge_weight_info.ptr), edge_weight_info.size);
        auto node_weight_vec = to_64(node_weight_span);

        if (edge_weight_info.size > 0) {
            auto [sym_indptr, sym_indices, sym_data] = make_sym(indptr_span, indices_span, edge_weight_span);
            auto indptr_vec = to_64(sym_indptr);
            auto indices_vec = to_64(sym_indices);
            auto edge_weight_vec = to_64(sym_data);
            std::cout << "start metis partitioning" << std::endl;
            std::vector<int64_t> result = metis_assignment(num_partition, num_iteration, num_initpart, unbalance_val, obj_cut,
                                                               indptr_vec, indices_vec, node_weight_vec, edge_weight_vec);
            return py::array_t<int64_t>(result.size(), result.data());

        } else {
            auto [sym_indptr, sym_indices] = make_sym(indptr_span, indices_span);
            auto indptr_vec = to_64(sym_indptr);
            auto indices_vec = to_64(sym_indices);
            auto edge_weight_vec = to_64(edge_weight_span);

            std::cout << "start metis partitioning" << std::endl;
            std::vector<int64_t> result = metis_assignment(num_partition, num_iteration, num_initpart, unbalance_val, obj_cut,
                                                               indptr_vec, indices_vec, node_weight_vec, edge_weight_vec);

            // Convert std::vector to py::array
            return py::array_t<int64_t>(result.size(), result.data());
        }
    }
} // namespace pymetis

PYBIND11_MODULE(pymetis, m)
{
    m.doc() = "Python bindings for pymetis using pybind11";

    m.def("metis_assignment", &pymetis::metis_assignment_wrapper,
          py::arg("num_partition"),
          py::arg("num_iteration"),
          py::arg("num_initpart"),
          py::arg("unbalance_val"),
          py::arg("obj_cut"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("node_weight"),
          py::arg("edge_weight"),
          "Single-threaded metis partition wrapper");
}
