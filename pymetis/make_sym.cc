//
// Created by juelin on 6/17/24.
//
#include "make_sym.h"
#include <cassert>
#include <iostream>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>
#include <algorithm>
#include <numeric>

namespace pymetis {

    struct EdgeWithData {
        id_t _src{0};
        id_t _dst{0};
        wgt_t _data{0};

        EdgeWithData() = default;

        EdgeWithData(id_t src, id_t dst, wgt_t data) : _src{src}, _dst{dst}, _data{data} {};

        bool operator==(const EdgeWithData &other) const {
            return other._src == _src && other._dst == _dst;
        }

        bool operator<(const EdgeWithData &other) const {
            if (_src == other._src) {
                return _dst < other._dst;
            } else {
                return _src < other._src;
            }
        }
    };

    struct Edge {
        id_t _src{0};
        id_t _dst{0};

        Edge(id_t src, id_t dst) : _src{src}, _dst{dst} {};

        Edge() = default;

        bool operator==(const Edge &other) const {
            return other._src == _src && other._dst == _dst;
        }

        bool operator<(const Edge &other) const {
            if (_src == other._src) {
                return _dst < other._dst;
            } else {
                return _src < other._src;
            }
        }
    };

    // struct EdgeWithData {
    //     metis_idx_t _src{0};
    //     metis_idx_t _dst{0};
    //     metis_idx_t _data{0};

    //     EdgeWithData() = default;

    //     EdgeWithData(metis_idx_t src, metis_idx_t dst, metis_idx_t data) : _src{src}, _dst{dst}, _data{data} {};

    //     bool operator==(const EdgeWithData &other) const {
    //         return other._src == _src && other._dst == _dst;
    //     }

    //     bool operator<(const EdgeWithData &other) const {
    //         if (_src == other._src) {
    //             return _dst < other._dst;
    //         } else {
    //             return _src < other._src;
    //         }
    //     }
    // };

    // struct Edge {
    //     metis_idx_t _src{0};
    //     metis_idx_t _dst{0};

    //     Edge(metis_idx_t src, metis_idx_t dst) : _src{src}, _dst{dst} {};

    //     Edge() = default;

    //     bool operator==(const Edge &other) const {
    //         return other._src == _src && other._dst == _dst;
    //     }

    //     bool operator<(const Edge &other) const {
    //         if (_src == other._src) {
    //             return _dst < other._dst;
    //         } else {
    //             return _src < other._src;
    //         }
    //     }
    // };

    bool ends_with(const std::string &str, const std::string &suffix) {
        size_t index = str.rfind(suffix);
        return (index != std::string::npos) && (index == str.size() - suffix.size());
    }

    std::vector<idx_t> expand_indptr(const std::span<idx_t> indptr) {
        int64_t v_num = indptr.size() - 1;
        auto _indptr = indptr.data();
        int64_t e_num = _indptr[v_num];
        std::vector<idx_t> output_array(e_num);
        auto _ret = output_array.data();
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, v_num),
                          [&](tbb::blocked_range<int64_t> r) {
                              for (int64_t v = r.begin(); v < r.end(); v++) {
                                  int64_t start = _indptr[v];
                                  int64_t end = _indptr[v + 1];
                                  for (int64_t i = start; i < end; i++) {
                                      _ret[i] = v;
                                  }
                              }
                          });

        return output_array;
    };

    std::vector<idx_t> compact_indptr(const std::span<idx_t> in_indptr,
                                      const std::span<uint8_t> flag) {
        int64_t v_num = in_indptr.size() - 1;
        int64_t e_num = flag.size();
        std::cout << "ReindexCSR e_num before compact = " << e_num << std::endl;
        auto _in_indices = flag.data();
        auto _in_indptr = in_indptr.data();
        std::vector<idx_t> degree(v_num + 1, 0);

        tbb::parallel_for(tbb::blocked_range<int64_t>(0, v_num),
                          [&](tbb::blocked_range<int64_t> r) {
                              for (int64_t i = r.begin(); i < r.end(); i++) {
                                  int64_t start = _in_indptr[i];
                                  int64_t end = _in_indptr[i + 1];
                                  for (int64_t j = start; j < end; j++) {
                                      degree.at(i) += (_in_indices[j]);
                                  }
                              }
                          });

        std::vector<idx_t> ret(v_num + 1);
        auto out_indptr_start = ret.data();
        auto out_indptr_end = std::exclusive_scan(degree.begin(), degree.end(), out_indptr_start, 0ll);
        std::cout << "ReindexCSR e_num after = " << out_indptr_start[v_num] << std::endl;
        assert(*(out_indptr_end - 1) == out_indptr_start[v_num]);
        return ret;
    }


    int64_t non_zero_edges(std::span<wgt_t> edge_weight) {
        int64_t org_e_num = edge_weight.size();
        int64_t non_z_num = tbb::parallel_reduce(
                tbb::blocked_range<int64_t>(0, org_e_num),
                0llu,
                [&](tbb::blocked_range<int64_t> r, int64_t running_total) {
                    for (int64_t i = r.begin(); i < r.end(); i++) {
                        running_total += edge_weight[i] > 0;
                    }
                    return running_total;
                },
                std::plus<int64_t>());

        return non_z_num;
    }

    /**
     * @brief
     *
     * @param indptr input indptr
     * @param indices input adjacency lists
     * @param edge_weight input edge weight
     * @return std::tuple<std::vector<id_t>, std::vector<id_t>, std::vector<id_t>>
     * 1. pruned indptr
     * 2. pruned indices
     * 3. pruned edge weights
     */
    std::tuple<std::vector<idx_t>, std::vector<id_t>, std::vector<wgt_t>>
    remove_zero_weight_edges(std::span<idx_t> indptr,
                             std::span<id_t> indices,
                             std::span<wgt_t> edge_weight) {
        assert(edge_weight.size() == indices.size());
        int64_t org_e_num = indices.size();
        int64_t new_e_num = non_zero_edges(edge_weight);
        std::cout << "org_e_num: " << org_e_num << " new_e_num: " << new_e_num << " ("
                  << 1.0 * new_e_num / org_e_num * 100
                  << "%)" << std::endl;
        std::vector<uint8_t> flag(org_e_num, 0);
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, org_e_num),
                          [&](tbb::blocked_range<int64_t> r) {
                              for (int64_t i = r.begin(); i < r.end(); i++) {
                                  flag[i] = edge_weight[i] > 0;
                              }
                          });

        std::vector<idx_t> new_indptr;
        std::vector<id_t> new_indices;
        std::vector<wgt_t> new_edge_weight;

        new_indptr = compact_indptr(indptr, flag);
        new_edge_weight.reserve(new_e_num);
        new_indices.reserve(new_e_num);
        for (int64_t i = 0; i < org_e_num; i++) {
            if (flag.at(i)) {
                new_indices.push_back(indices[i]);
                new_edge_weight.push_back(edge_weight[i]);
            }
        }
        return std::make_tuple(new_indptr, new_indices, new_edge_weight);
    }

    std::tuple<std::vector<idx_t>, std::vector<id_t>> make_sym(const std::span<idx_t> in_indptr,
                                                               const std::span<id_t> in_indices) {
        int64_t e_num = in_indices.size();
        int64_t v_num = in_indptr.size() - 1;
        std::cout << "v_num: " << v_num << " | e_num: " << e_num << std::endl;
        typedef std::vector<Edge> EdgeVec;
        EdgeVec edge_vec;
        edge_vec.resize(e_num * 2);
        auto _in_indptr = in_indptr.data();
        auto _in_indices = in_indices.data();
        tbb::parallel_for(tbb::blocked_range<id_t>(0, v_num),
                          [&](tbb::blocked_range<id_t> r) {
                              for (auto v = r.begin(); v < r.end(); v++) {
                                  auto start = _in_indptr[v];
                                  auto end = _in_indptr[v + 1];
                                  for (auto i = start; i < end; i++) {
                                      auto u = _in_indices[i];
                                      if (u != v) {
                                          edge_vec.at(i * 2) = {v, u};
                                          edge_vec.at(i * 2 + 1) = {u, v};
                                      }
                                  }
                              }
                          });
        std::cout << "start sorting" << std::endl;
        tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
        edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
        edge_vec.shrink_to_fit();
        if (edge_vec.at(0)._src == edge_vec.at(0)._dst) {
            edge_vec.erase(edge_vec.begin());
        }
        int64_t cur_e_num = edge_vec.size();

        std::vector<idx_t> indptr(v_num + 1);
        std::vector<std::atomic<idx_t>> degree(v_num + 1);
        std::vector<id_t> indices(cur_e_num);
        auto indices_ptr = indices.data();
        std::cout << "compute degree" << std::endl;

        tbb::parallel_for(tbb::blocked_range<id_t>(0, cur_e_num),
                          [&](tbb::blocked_range<id_t> r) {
                              for (auto i = r.begin(); i < r.end(); i++) {
                                  const auto &e = edge_vec.at(i);
                                  degree.at(e._src)++;
                                  indices_ptr[i] = e._dst;
                              }
                          });

        std::cout << "compute indptr" << std::endl;
        auto out_start = indptr.data();
        auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
        static_cast<void>(out_end);
        std::cout << "final e_num: " << cur_e_num << std::endl;

        assert(out_start[v_num] == cur_e_num);
        return {indptr, indices};
    }

    // std::tuple<std::vector<metis_idx_t>, std::vector<metis_idx_t>> make_sym(const std::span<metis_idx_t> in_indptr,
    //                                                            const std::span<metis_idx_t> in_indices) {
    //     int64_t e_num = in_indices.size();
    //     int64_t v_num = in_indptr.size() - 1;
    //     std::cout << "v_num: " << v_num << " | e_num: " << e_num << std::endl;
    //     typedef std::vector<Edge> EdgeVec;
    //     EdgeVec edge_vec;
    //     edge_vec.resize(e_num * 2);
    //     auto _in_indptr = in_indptr.data();
    //     auto _in_indices = in_indices.data();
    //     tbb::parallel_for(tbb::blocked_range<metis_idx_t>(0, v_num),
    //                       [&](tbb::blocked_range<metis_idx_t> r) {
    //                           for (auto v = r.begin(); v < r.end(); v++) {
    //                               auto start = _in_indptr[v];
    //                               auto end = _in_indptr[v + 1];
    //                               for (auto i = start; i < end; i++) {
    //                                   auto u = _in_indices[i];
    //                                   if (u != v) {
    //                                       edge_vec.at(i * 2) = {v, u};
    //                                       edge_vec.at(i * 2 + 1) = {u, v};
    //                                   }
    //                               }
    //                           }
    //                       });
    //     std::cout << "start sorting" << std::endl;
    //     tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
    //     edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
    //     edge_vec.shrink_to_fit();
    //     if (edge_vec.at(0)._src == edge_vec.at(0)._dst) {
    //         edge_vec.erase(edge_vec.begin());
    //     }
    //     int64_t cur_e_num = edge_vec.size();

    //     std::vector<metis_idx_t> indptr(v_num + 1);
    //     std::vector<std::atomic<metis_idx_t>> degree(v_num + 1);
    //     std::vector<metis_idx_t> indices(cur_e_num);
    //     auto indices_ptr = indices.data();
    //     std::cout << "compute degree" << std::endl;

    //     tbb::parallel_for(tbb::blocked_range<metis_idx_t>(0, cur_e_num),
    //                       [&](tbb::blocked_range<metis_idx_t> r) {
    //                           for (auto i = r.begin(); i < r.end(); i++) {
    //                               const auto &e = edge_vec.at(i);
    //                               degree.at(e._src)++;
    //                               indices_ptr[i] = e._dst;
    //                           }
    //                       });

    //     std::cout << "compute indptr" << std::endl;
    //     auto out_start = indptr.data();
    //     auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
    //     static_cast<void>(out_end);
    //     std::cout << "final e_num: " << cur_e_num << std::endl;

    //     assert(out_start[v_num] == cur_e_num);
    //     return {indptr, indices};
    // }

    std::tuple<std::vector<idx_t>, std::vector<id_t>, std::vector<wgt_t>> make_sym(
            const std::span<idx_t> init_indptr,
            const std::span<id_t> init_indices,
            const std::span<wgt_t> init_data) {
        assert(init_data.size() == init_indices.size());
        std::cout << "init v_num: " << init_indptr.size() - 1 << " | e_num: " << init_indices.size() << std::endl;

        auto [in_indptr, in_indices, in_data] = remove_zero_weight_edges(init_indptr, init_indices, init_data);

        std::cout << "pruned v_num: " << in_indptr.size() - 1 << " | e_num: " << in_indices.size() << std::endl;

        int64_t e_num = in_indices.size();
        int64_t v_num = in_indptr.size() - 1;
        typedef std::vector<EdgeWithData> EdgeVec;
        EdgeVec edge_vec;
        edge_vec.resize(e_num * 2);
        auto _in_indptr = in_indptr.data();
        auto _in_indices = in_indices.data();
        auto _in_data = in_data.data();
        tbb::parallel_for(tbb::blocked_range<id_t>(0, v_num),
                          [&](tbb::blocked_range<id_t> r) {
                              for (auto v = r.begin(); v < r.end(); v++) {
                                  auto start = _in_indptr[v];
                                  auto end = _in_indptr[v + 1];
                                  for (auto i = start; i < end; i++) {
                                      auto u = _in_indices[i];
                                      if (u == v) continue;

                                      auto v_u_data = _in_data[i];
                                      auto u_v_data = 0;
                                      auto u_adj_start = _in_indptr[u];
                                      auto u_adj_end = _in_indptr[u + 1];

                                      for (auto j = u_adj_start; j < u_adj_end; j++) {
                                          if (_in_indices[j] == u) {
                                              u_v_data = _in_data[j];
                                              break;
                                          };
                                      }

                                      wgt_t d = u_v_data + v_u_data;
                                      edge_vec.at(i * 2) = {v, u, d};
                                      edge_vec.at(i * 2 + 1) = {u, v, d};
                                  }
                              }
                          });

        std::cout << "MakeSym start sorting" << std::endl;
        tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
        edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
        edge_vec.shrink_to_fit();

        if (edge_vec.at(0)._src == edge_vec.at(0)._dst) {
            edge_vec.erase(edge_vec.begin());
        }

        int64_t cur_e_num = edge_vec.size();
        std::vector<std::atomic<idx_t>> degree(v_num + 1);
        std::vector<idx_t> indptr(v_num + 1);
        std::vector<id_t> indices(cur_e_num);
        std::vector<wgt_t> retdata(cur_e_num);

        auto indices_ptr = indices.data();
        auto retdata_ptr = retdata.data();
        std::cout << "compute degree" << std::endl;

        tbb::parallel_for(tbb::blocked_range<idx_t>(0, cur_e_num),
                          [&](tbb::blocked_range<idx_t> r) {
                              for (auto i = r.begin(); i < r.end(); i++) {
                                  const auto &e = edge_vec.at(i);
                                  degree.at(e._src)++;
                                  indices_ptr[i] = e._dst;
                                  retdata_ptr[i] = e._data;
                              }
                          });

        std::cout << "compute indptr" << std::endl;
        auto out_start = indptr.data();
        auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
        static_cast<void>(out_end);
        std::cout << "final e_num: " << cur_e_num << std::endl;

        assert(out_start[v_num] == cur_e_num);
        return {indptr, indices, retdata};
    };

    // std::tuple<std::vector<metis_idx_t>, std::vector<metis_idx_t>, std::vector<metis_idx_t>> make_sym(
    //         const std::span<metis_idx_t> init_indptr,
    //         const std::span<metis_idx_t> init_indices,
    //         const std::span<metis_idx_t> init_data) {
    //     assert(init_data.size() == init_indices.size());
    //     std::cout << "init v_num: " << init_indptr.size() - 1 << " | e_num: " << init_indices.size() << std::endl;

    //     auto [in_indptr, in_indices, in_data] = remove_zero_weight_edges(init_indptr, init_indices, init_data);

    //     std::cout << "pruned v_num: " << in_indptr.size() - 1 << " | e_num: " << in_indices.size() << std::endl;

    //     int64_t e_num = in_indices.size();
    //     int64_t v_num = in_indptr.size() - 1;
    //     typedef std::vector<EdgeWithData> EdgeVec;
    //     EdgeVec edge_vec;
    //     edge_vec.resize(e_num * 2);
    //     auto _in_indptr = in_indptr.data();
    //     auto _in_indices = in_indices.data();
    //     auto _in_data = in_data.data();
    //     tbb::parallel_for(tbb::blocked_range<metis_idx_t>(0, v_num),
    //                       [&](tbb::blocked_range<metis_idx_t> r) {
    //                           for (auto v = r.begin(); v < r.end(); v++) {
    //                               auto start = _in_indptr[v];
    //                               auto end = _in_indptr[v + 1];
    //                               for (auto i = start; i < end; i++) {
    //                                   auto u = _in_indices[i];
    //                                   if (u == v) continue;

    //                                   auto v_u_data = _in_data[i];
    //                                   auto u_v_data = 0;
    //                                   auto u_adj_start = _in_indptr[u];
    //                                   auto u_adj_end = _in_indptr[u + 1];

    //                                   for (auto j = u_adj_start; j < u_adj_end; j++) {
    //                                       if (_in_indices[j] == u) {
    //                                           u_v_data = _in_data[j];
    //                                           break;
    //                                       };
    //                                   }

    //                                   metis_idx_t d = u_v_data + v_u_data;
    //                                   edge_vec.at(i * 2) = {v, u, d};
    //                                   edge_vec.at(i * 2 + 1) = {u, v, d};
    //                               }
    //                           }
    //                       });

    //     std::cout << "MakeSym start sorting" << std::endl;
    //     tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
    //     edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
    //     edge_vec.shrink_to_fit();

    //     if (edge_vec.at(0)._src == edge_vec.at(0)._dst) {
    //         edge_vec.erase(edge_vec.begin());
    //     }

    //     int64_t cur_e_num = edge_vec.size();
    //     std::vector<std::atomic<metis_idx_t>> degree(v_num + 1);
    //     std::vector<metis_idx_t> indptr(v_num + 1);
    //     std::vector<metis_idx_t> indices(cur_e_num);
    //     std::vector<metis_idx_t> retdata(cur_e_num);

    //     auto indices_ptr = indices.data();
    //     auto retdata_ptr = retdata.data();
    //     std::cout << "compute degree" << std::endl;

    //     tbb::parallel_for(tbb::blocked_range<metis_idx_t>(0, cur_e_num),
    //                       [&](tbb::blocked_range<metis_idx_t> r) {
    //                           for (auto i = r.begin(); i < r.end(); i++) {
    //                               const auto &e = edge_vec.at(i);
    //                               degree.at(e._src)++;
    //                               indices_ptr[i] = e._dst;
    //                               retdata_ptr[i] = e._data;
    //                           }
    //                       });

    //     std::cout << "compute indptr" << std::endl;
    //     auto out_start = indptr.data();
    //     auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
    //     static_cast<void>(out_end);
    //     std::cout << "final e_num: " << cur_e_num << std::endl;

    //     assert(out_start[v_num] == cur_e_num);
    //     return {indptr, indices, retdata};
    // };

}