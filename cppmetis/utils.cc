#include "utils.h"
#include "command_line.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <oneapi/tbb/parallel_sort.h>
#include <filesystem>

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

    Args parse_args(int argc, const char **argv) {
        auto cmd = CommandLine(argc, argv);
        Args args;

        cmd.get_cmd_line_argument<int64_t>("num_partition", args.num_partition, 4);
        cmd.get_cmd_line_argument<int64_t>("num_init_part", args.num_init_part, 4);
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
        return args;
    };

    std::unique_ptr<Dataset> load_dataset(const Args &args) {
        cnpy::NpyArray indptr = cnpy::npy_load(args.indptr_path);
        cnpy::NpyArray indices = cnpy::npy_load(args.indices_path);
        cnpy::NpyArray train_node, test_node, valid_node;
        cnpy::NpyArray node_weight, edge_weight;

        if (!args.node_weight_path.empty()) {
            node_weight = cnpy::npy_load(args.node_weight_path);
        }

        if (!args.edge_weight_path.empty()) {
            std::cout << "Use edge weight: " << args.edge_weight_path << std::endl;
            edge_weight = cnpy::npy_load(args.edge_weight_path);
        }

        std::vector<idx_t> assignment;
        if (ends_with(args.indptr_path, "indptr_xsym.npy")) {
            assert(ends_with(args.indices_path, "indices_xsym.npy"));
            auto ret = std::make_unique<Dataset>();
            std::span<WeightType> edge_weight_span;
            if (edge_weight.num_vals > 0)
                edge_weight_span = {edge_weight.data<WeightType>(), edge_weight.num_vals};

            auto [sym_indptr, sym_indice, sym_edge_weight] = make_sym({indptr.data<idx_t>(), indptr.num_vals},
                                                                      {indices.data<idx_t>(), indices.num_vals},
                                                                      edge_weight_span);
            ret->indptr = std::move(sym_indptr);
            ret->indices = std::move(sym_indice);
            ret->edge_weight = std::move(sym_edge_weight);
            if (node_weight.num_vals)
                ret->node_weight = node_weight.as_vec<WeightType>();
            return ret;
        } else {
            assert(ends_with(args.indices_path, "indices_sym.npy"));
            auto ret = std::make_unique<Dataset>();
            ret->indptr = indptr.as_vec<idx_t>();
            ret->indices = indices.as_vec<idx_t>();
            if (edge_weight.num_vals)
                ret->edge_weight = edge_weight.as_vec<WeightType>();
            if (node_weight.num_vals)
                ret->node_weight = node_weight.as_vec<WeightType>();
            return ret;
        }
    }

    DatasetPtr remove_zero_weight_edges(const DatasetPtr &dataset) {
        assert(dataset->edge_weight.size() == dataset->indices.size());
        int64_t org_e_num = dataset->indices.size();
        int64_t new_e_num{0};
        std::vector<uint8_t> flag(org_e_num, 0);

        for (int64_t i = 0; i < org_e_num; i++) {
            flag.at(i) = dataset->edge_weight.at(i) > 0;
            new_e_num += flag.at(i);
        }

        std::cout << "org e_num" << org_e_num << " new e_num" << new_e_num << std::endl;
        auto ret = std::make_unique<Dataset>();
        ret->indptr = compact_indptr(dataset->indptr, flag);
        ret->edge_weight.reserve(new_e_num);
        ret->indices.reserve(new_e_num);
        for (int64_t i = 0; i < org_e_num; i++) {
            if (flag.at(i)) {
                ret->indices.push_back(dataset->indices.at(i));
                ret->edge_weight.push_back(dataset->edge_weight.at(i));
            }
        }
        ret->vtxdist = dataset->vtxdist;
        ret->node_weight = dataset->node_weight;

        return std::move(ret);
    };

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

    std::vector<idx_t> get_vtx_dist(const DatasetPtr &data, int world_size, bool balance_edge) {
        std::vector<idx_t> vtxdist(1, 0);
        idx_t num_edges = data->indices.size();
        idx_t num_nodes = data->indptr.size() - 1;
        for (int r = 0; r < world_size; r++) {
            if (r == world_size - 1) {
                vtxdist.push_back(num_nodes);
            } else if (balance_edge) {
                idx_t loc_num_edges = (num_edges / world_size) * (r + 1);
                auto begin = data->indptr.begin();
                auto itr = std::upper_bound(data->indptr.begin(), data->indptr.end(), loc_num_edges);
                vtxdist.push_back(itr - begin);
            } else {
                vtxdist.push_back(num_nodes / world_size * (r + 1));
            }
        }
        data->vtxdist = vtxdist;
        return vtxdist;
    };

    DatasetPtr get_local_data(const DatasetPtr &data, int rank, int world_size) {
        auto ret = std::make_unique<Dataset>();
        ret->vtxdist = get_vtx_dist(data, world_size);
        idx_t total_e_num = data->indices.size();
        idx_t total_v_num = data->indptr.size() - 1;
        idx_t start_v_idx = ret->vtxdist.at(rank);
        idx_t end_v_idx = ret->vtxdist.at(rank + 1);
        idx_t start_e_idx = data->indptr.at(start_v_idx);
        idx_t end_e_idx = data->indptr.at(end_v_idx);



        // copy node_weight
        if (data->node_weight.size() == total_v_num) {
            ret->node_weight = std::vector<WeightType>(data->node_weight.begin() + start_v_idx,
                                                       data->node_weight.begin() + end_v_idx);
        }
        // copy edge_weight
        if (data->edge_weight.size() == total_e_num) {
            ret->edge_weight = std::vector<WeightType>(data->edge_weight.begin() + start_e_idx,
                                                       data->edge_weight.begin() + end_e_idx);
        }
        // copy indices
        ret->indices = std::vector<WeightType>(data->indices.begin() + start_e_idx,
                                               data->indices.begin() + end_e_idx);

        // compute local indptr (start from 0)
        for (idx_t i = start_v_idx; i <= end_v_idx; i++)
            ret->indptr.push_back(data->indptr.at(i) - start_e_idx);

        assert(ret->vtxdist.size() == world_size + 1);
        assert(ret->vtxdist.at(world_size) == total_v_num);
        assert(ret->indptr.at(0) == 0);
        assert(ret->indptr.at(end_v_idx - start_v_idx) == ret->indices.size());
        return std::move(ret);
    };
}