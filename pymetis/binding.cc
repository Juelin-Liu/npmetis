#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include "mt_metis_assignment.h"
#include "make_sym.h"

namespace py = pybind11;

namespace pymetis
{
    py::array_t<uint32_t> mt_metis_assignment_wrapper(int64_t num_partition,
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

        if (edge_weight_info.size > 0) {
            auto [sym_indptr, sym_indices, sym_data] = make_sym(indptr_span, indices_span, edge_weight_span);
            indptr_span = sym_indptr;
            indices_span = sym_indices;
            edge_weight_span = sym_data;

            std::vector<uint32_t> result = mt_metis_assignment(num_partition, num_iteration, num_initpart, unbalance_val, obj_cut,
                                                               indptr_span, indices_span, node_weight_span, edge_weight_span);
            return py::array_t<uint32_t>(result.size(), result.data());

        } else {

            auto [sym_indptr, sym_indices] = make_sym(indptr_span, indices_span);
            indptr_span = sym_indptr;
            indices_span = sym_indices;

            std::vector<uint32_t> result = mt_metis_assignment(num_partition, num_iteration, num_initpart, unbalance_val, obj_cut,
                                                               indptr_span, indices_span, node_weight_span, edge_weight_span);

            // Convert std::vector to py::array
            return py::array_t<uint32_t>(result.size(), result.data());
        }


    }
} // namespace pymetis

PYBIND11_MODULE(pymetis, m)
{
    m.doc() = "Python bindings for pymetis using pybind11";

    m.def("mt_metis_assignment", &pymetis::mt_metis_assignment_wrapper,
          py::arg("num_partition"),
          py::arg("num_iteration"),
          py::arg("num_initpart"),
          py::arg("unbalance_val"),
          py::arg("obj_cut"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("node_weight"),
          py::arg("edge_weight"),
          "Multi-threaded metis partition wrapper");
}
