#include "utils.h"
#include "cnpy.h"
#include "cnpy_mmap.h"
#include "command_line.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>

namespace cppmetis {
    struct EdgeWithData {
        idx_t _src{0};
        idx_t _dst{0};
        WeightType _data{0};

        EdgeWithData() = default;

        EdgeWithData(idx_t src, idx_t dst, WeightType data) : _src{src}, _dst{dst}, _data{data} {};

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
        idx_t _src{0};
        idx_t _dst{0};

        Edge(idx_t src, idx_t dst) : _src{src}, _dst{dst} {};

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

    bool ends_with(const std::string &str, const std::string &suffix) {
        size_t index = str.rfind(suffix);
        return (index != std::string::npos) && (index == str.size() - suffix.size());
    }

    Args parse_args(int argc, const char **argv, bool show_cmd) {
        auto cmd = CommandLine(argc, argv);
        Args args;

        cmd.get_cmd_line_argument<int64_t>("num_partition", args.num_partition, 4);
        cmd.get_cmd_line_argument<int64_t>("num_init_part", args.num_init_part, 1);
        cmd.get_cmd_line_argument<int64_t>("num_iteration", args.num_iteration, 10);
        cmd.get_cmd_line_argument<float>("unbalance_val", args.unbalance_val, 1.05);
        args.use_cut = cmd.check_cmd_line_flag("use_cut");
        cmd.get_cmd_line_argument<std::string>("indptr", args.indptr_path);
        cmd.get_cmd_line_argument<std::string>("indices", args.indices_path);
        cmd.get_cmd_line_argument<std::string>("output", args.output_path);
        cmd.get_cmd_line_argument<std::string>("node_weight", args.node_weight_path);
        cmd.get_cmd_line_argument<std::string>("edge_weight", args.edge_weight_path);

        assert(!args.indptr_path.empty());
        assert(!args.indices_path.empty());
        assert(!args.output_path.empty());
        assert(args.num_partition > 0);
        assert(args.unbalance_val <= args.num_partition);
        assert(args.unbalance_val >= 1);

        if (show_cmd) {
            std::cout << "Cmd options:\n";
            std::cout << "num_partition: " << args.num_partition << std::endl;
            std::cout << "num_init_part: " << args.num_init_part << std::endl;
            std::cout << "num_iteration: " << args.num_iteration << std::endl;
            std::cout << "unbalance_val: " << args.unbalance_val << std::endl;
            std::cout << "use_cut: " << args.use_cut << std::endl;
            std::cout << "indptr: " << args.indptr_path << std::endl;
            std::cout << "indices: " << args.indices_path << std::endl;
            std::cout << "node weight: " << args.node_weight_path << std::endl;
            std::cout << "edge weight: " << args.edge_weight_path << std::endl;
        }
        return args;
    };

    DatasetPtr load_dataset(const Args &args, bool to_sym) {
        cnpyMmap::NpyArray indptr = cnpyMmap::npy_load(args.indptr_path);
        cnpyMmap::NpyArray indices = cnpyMmap::npy_load(args.indices_path);
        cnpyMmap::NpyArray train_node, test_node, valid_node;
        cnpyMmap::NpyArray node_weight, edge_weight;
        auto ret = std::make_unique<Dataset>();

        if (!args.edge_weight_path.empty()) {
            edge_weight = cnpyMmap::npy_load(args.edge_weight_path);
            assert(edge_weight.num_vals == indices.num_vals);
        }

        if (!args.node_weight_path.empty()) {
            node_weight = cnpyMmap::npy_load(args.node_weight_path);
            assert(node_weight.num_vals % (indptr.num_vals - 1) == 0);
        }

        ret->indptr = indptr.as_vec<idx_t>();
        ret->indices = indices.as_vec<idx_t>();
        ret->edge_weight = edge_weight.as_vec<WeightType>();
        ret->node_weight = node_weight.as_vec<WeightType>();

        if (to_sym) {
            return std::move(make_sym(ret));
        } else {
            return std::move(ret);
        }
    }

    DatasetPtr make_sym(const DatasetPtr &dataset) {
        std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<WeightType>> ret;
        if (dataset->edge_weight.size() == dataset->indices.size()) {
            auto [pruned_indptr, pruned_indice, pruned_edge_weight] = remove_zero_weight_edges(dataset->indptr,
                                                                                               dataset->indices,
                                                                                               dataset->edge_weight);
            auto [sym_indptr, sym_indice, sym_edge_weight] = make_sym(pruned_indptr, pruned_indice, pruned_edge_weight);
            auto ret = std::make_unique<Dataset>();
            ret->indptr = std::move(sym_indptr);
            ret->indices = std::move(sym_indice);
            ret->edge_weight = std::move(sym_edge_weight);
            ret->node_weight = dataset->node_weight;
            ret->vtxdist = dataset->vtxdist;
            return std::move(ret);
        } else {
            auto [sym_indptr, sym_indice, sym_edge_weight] = make_sym(dataset->indptr, dataset->indices,
                                                                      dataset->edge_weight);
            auto ret = std::make_unique<Dataset>();
            ret->indptr = std::move(sym_indptr);
            ret->indices = std::move(sym_indice);
            ret->edge_weight = std::move(sym_edge_weight);
            ret->node_weight = dataset->node_weight;
            ret->vtxdist = dataset->vtxdist;
            return std::move(ret);
        }

    };

    int64_t non_zero_edges(std::span<idx_t> edge_weight) {
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
     * @return std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<idx_t>>
     * 1. pruned indptr
     * 2. pruned indices
     * 3. pruned edge weights
     */
    std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<WeightType>>
    remove_zero_weight_edges(std::span<idx_t> indptr,
                             std::span<idx_t> indices,
                             std::span<WeightType> edge_weight) {
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

        std::vector<idx_t> new_indptr, new_indices;
        std::vector<WeightType> new_edge_weight;

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
        std::vector<int64_t> degree(v_num + 1, 0);

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

    std::tuple<std::vector<idx_t>, std::vector<idx_t>, std::vector<WeightType>> make_sym(
            const std::span<idx_t> in_indptr,
            const std::span<idx_t> in_indices,
            const std::span<WeightType> in_data) {
        if (in_data.size() == 0) {
            int64_t e_num = in_indices.size();
            int64_t v_num = in_indptr.size() - 1;
            std::cout << "MakeSym v_num: " << v_num << " | e_num: " << e_num << std::endl;
            typedef std::vector<Edge> EdgeVec;
            EdgeVec edge_vec;
            edge_vec.resize(e_num * 2);
            auto _in_indptr = in_indptr.data();
            auto _in_indices = in_indices.data();
            tbb::parallel_for(tbb::blocked_range<idx_t>(0, v_num),
                              [&](tbb::blocked_range<idx_t> r) {
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
            std::cout << "MakeSym start sorting" << std::endl;
            tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
            edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
            edge_vec.shrink_to_fit();
            if (edge_vec.at(0)._src == edge_vec.at(0)._dst) {
                edge_vec.erase(edge_vec.begin());
            }
            int64_t cur_e_num = edge_vec.size();

            std::vector<idx_t> indptr(v_num + 1);
            std::vector<std::atomic<idx_t>> degree(v_num + 1);
            std::vector<idx_t> indices(cur_e_num);
            auto indices_ptr = indices.data();
            std::cout << "MakeSym compute degree" << std::endl;

            tbb::parallel_for(tbb::blocked_range<idx_t>(0, cur_e_num),
                              [&](tbb::blocked_range<idx_t> r) {
                                  for (auto i = r.begin(); i < r.end(); i++) {
                                      const auto &e = edge_vec.at(i);
                                      degree.at(e._src)++;
                                      indices_ptr[i] = e._dst;
                                  }
                              });

            std::cout << "MakeSym compute indptr" << std::endl;
            auto out_start = indptr.data();
            auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
            std::cout << "MakeSym e_num after convert " << cur_e_num << std::endl;

            assert(out_start[v_num] == cur_e_num);
            return {indptr, indices, std::vector<WeightType>()};
        } else {
            assert(in_data.size() == in_indices.size());
            int64_t e_num = in_indices.size();
            int64_t v_num = in_indptr.size() - 1;
            std::cout << "MakeSym v_num: " << v_num << " | e_num: " << e_num << std::endl;
            typedef std::vector<EdgeWithData> EdgeVec;
            EdgeVec edge_vec;
            edge_vec.resize(e_num * 2);
            auto _in_indptr = in_indptr.data();
            auto _in_indices = in_indices.data();
            auto _in_data = in_data.data();
            tbb::parallel_for(tbb::blocked_range<idx_t>(0, v_num),
                              [&](tbb::blocked_range<idx_t> r) {
                                  for (auto v = r.begin(); v < r.end(); v++) {
                                      auto start = _in_indptr[v];
                                      auto end = _in_indptr[v + 1];
                                      for (auto i = start; i < end; i++) {
                                          auto u = _in_indices[i];
                                          auto d = _in_data[i];
                                          if (u != v) {
                                              edge_vec.at(i * 2) = {v, u, d};
                                              edge_vec.at(i * 2 + 1) = {u, v, d};
                                          }
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
            std::vector<idx_t> indptr(v_num + 1);
            std::vector<std::atomic<idx_t>> degree(v_num + 1);
            std::vector<idx_t> indices(cur_e_num);
            std::vector<WeightType> retdata(cur_e_num);

            auto indices_ptr = indices.data();
            auto retdata_ptr = retdata.data();
            std::cout << "MakeSym compute degree" << std::endl;

            tbb::parallel_for(tbb::blocked_range<idx_t>(0, cur_e_num),
                              [&](tbb::blocked_range<idx_t> r) {
                                  for (auto i = r.begin(); i < r.end(); i++) {
                                      const auto &e = edge_vec.at(i);
                                      degree.at(e._src)++;
                                      indices_ptr[i] = e._dst;
                                      retdata_ptr[i] = e._data;
                                  }
                              });

            std::cout << "MakeSym compute indptr" << std::endl;
            auto out_start = indptr.data();
            auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
            std::cout << "MakeSym e_num after convert " << cur_e_num << std::endl;

            assert(out_start[v_num] == cur_e_num);
            return {indptr, indices, retdata};
        }
    };

    std::vector<idx_t> get_vtx_dist(std::span<idx_t> indptr, int64_t num_edges, int world_size, bool balance_edges) {
        std::vector<idx_t> vtxdist(1, 0);
        idx_t num_nodes = indptr.size() - 1;
        for (int r = 0; r < world_size; r++) {
            if (r == world_size - 1) {
                vtxdist.push_back(num_nodes);
            } else if (balance_edges) {
                idx_t loc_num_edges = (num_edges / world_size) * (r + 1);
                auto begin = indptr.begin();
                auto itr = std::upper_bound(indptr.begin(), indptr.end(), loc_num_edges);
                vtxdist.push_back(itr - begin);
            } else {
                vtxdist.push_back(num_nodes / world_size * (r + 1));
            }
        }

        return vtxdist;
    };

    DatasetPtr get_local_data(const Args &args, int rank, int world_size) {
        cnpyMmap::NpyArray indptr = cnpyMmap::npy_load(args.indptr_path);
        cnpyMmap::NpyArray indices = cnpyMmap::npy_load(args.indices_path);
        cnpyMmap::NpyArray train_node, test_node, valid_node;
        cnpyMmap::NpyArray node_weight, edge_weight;

        if (!args.edge_weight_path.empty()) {
            edge_weight = cnpyMmap::npy_load(args.edge_weight_path);
            assert(edge_weight.num_vals == indices.num_vals);
        }

        if (!args.node_weight_path.empty()) {
            node_weight = cnpyMmap::npy_load(args.node_weight_path);
            assert(node_weight.num_vals == indptr.num_vals - 1);
        }

        auto ret = std::make_unique<Dataset>();
        ret->vtxdist = get_vtx_dist({indptr.data<idx_t>(), indptr.num_vals}, indices.num_vals, world_size);

        idx_t total_e_num = indices.num_vals;
        idx_t total_v_num = indptr.num_vals - 1;
        idx_t start_v_idx = ret->vtxdist.at(rank);
        idx_t end_v_idx = ret->vtxdist.at(rank + 1);
        idx_t start_e_idx = indptr.data<idx_t>()[start_v_idx];
        idx_t end_e_idx = indptr.data<idx_t>()[end_v_idx];

        // copy node_weight
        if (node_weight.num_vals == total_v_num) {
            ret->node_weight = std::vector<WeightType>(node_weight.data<WeightType>() + start_v_idx,
                                                       node_weight.data<WeightType>() + end_v_idx);
        }
        // copy edge_weight
        if (edge_weight.num_vals == total_e_num) {
            ret->edge_weight = std::vector<WeightType>(edge_weight.data<WeightType>() + start_e_idx,
                                                       edge_weight.data<WeightType>() + end_e_idx);
        }
        // copy indices
        ret->indices = std::vector<idx_t>(indices.data<idx_t>() + start_e_idx,
                                          indices.data<idx_t>() + end_e_idx);

        // compute local indptr (start from 0)
        for (idx_t i = start_v_idx; i <= end_v_idx; i++)
            ret->indptr.push_back(indptr.data<idx_t>()[i] - start_e_idx);

        assert(ret->vtxdist.size() == world_size + 1);
        assert(ret->vtxdist.at(world_size) == total_v_num);
        assert(ret->indptr.at(0) == 0);
        assert(ret->indptr.at(end_v_idx - start_v_idx) == ret->indices.size());
        return std::move(ret);
    };
}
