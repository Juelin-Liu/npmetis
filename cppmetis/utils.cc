#include "utils.h"
#include "command_line.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <oneapi/tbb/parallel_sort.h>
#include <filesystem>

namespace pmetis
{
    struct EdgeWithData
    {
        VertexIDType _src{0};
        VertexIDType _dst{0};
        WeightType _data{0};

        EdgeWithData() = default;

        EdgeWithData(VertexIDType src, VertexIDType dst, WeightType data) : _src{src}, _dst{dst}, _data{data} {};

        bool operator==(const EdgeWithData &other) const
        {
            return other._src == _src && other._dst == _dst;
        }

        bool operator<(const EdgeWithData &other) const
        {
            if (_src == other._src)
            {
                return _dst < other._dst;
            }
            else
            {
                return _src < other._src;
            }
        }
    };

    struct Edge
    {
        VertexIDType _src{0};
        VertexIDType _dst{0};

        Edge(VertexIDType src, VertexIDType dst) : _src{src}, _dst{dst} {};

        Edge() = default;

        bool operator==(const Edge &other) const
        {
            return other._src == _src && other._dst == _dst;
        }

        bool operator<(const Edge &other) const
        {
            if (_src == other._src)
            {
                return _dst < other._dst;
            }
            else
            {
                return _src < other._src;
            }
        }
    };

    bool ends_with(const std::string& str, const std::string& suffix) {
        size_t index = str.rfind(suffix);
        return (index != std::string::npos) && (index == str.size() - suffix.size());
    }

    Args parse_args(int argc, const char **argv){
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

    std::unique_ptr<Dataset> load_dataset(const Args& args)
    {
        cnpy::NpyArray indptr = cnpy::npy_load(args.indptr_path);
        cnpy::NpyArray indices = cnpy::npy_load(args.indices_path);
        cnpy::NpyArray train_node, test_node, valid_node;
        cnpy::NpyArray node_weight, edge_weight;

        if (!args.node_weight_path.empty())
        {
            node_weight = cnpy::npy_load(args.node_weight_path);
        }

        if (!args.edge_weight_path.empty())
        {
            std::cout << "Use edge weight: " << args.edge_weight_path << std::endl;
            edge_weight = cnpy::npy_load(args.edge_weight_path);
        }

        std::vector<VertexPIDType> assignment;
        if (ends_with(args.indptr_path, "indptr_xsym.npy"))
        {
            assert(ends_with(args.indices_path, "indices_xsym.npy"));
            auto ret = std::make_unique<Dataset>();
            std::span<WeightType> edge_weight_span;
            if (edge_weight.num_vals > 0)
                edge_weight_span = {edge_weight.data<WeightType>(), edge_weight.num_vals};

            auto [sym_indptr, sym_indice, sym_edge_weight] = pmetis::make_sym({indptr.data<IndptrType>(), indptr.num_vals},
                                                                              {indices.data<VertexIDType>(), indices.num_vals},
                                                                              edge_weight_span);
            ret->indptr = std::move(sym_indptr);
            ret->indices = std::move(sym_indice);
            ret->edge_weight = std::move(sym_edge_weight);
            if (node_weight.num_vals)
                ret->node_weight = node_weight.as_vec<WeightType>();
            return ret;
        }
        else
        {
            assert(ends_with(args.indices_path, "indices_sym.npy"));
            auto ret = std::make_unique<Dataset>();
            ret->indptr = indptr.as_vec<IndptrType>();
            ret->indices = indices.as_vec<VertexIDType>();
            if (edge_weight.num_vals)
                ret->edge_weight = edge_weight.as_vec<WeightType>();
            if (node_weight.num_vals)
                ret->node_weight = node_weight.as_vec<WeightType>();
            return ret;
        }
    }

    std::vector<IndptrType> expand_indptr(const std::span<IndptrType> indptr)
    {
        int64_t v_num = indptr.size() - 1;
        auto _indptr = indptr.data();
        int64_t e_num = _indptr[v_num];
        std::vector<IndptrType> output_array(e_num);
        auto _ret = output_array.data();
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, v_num),
                          [&](tbb::blocked_range<int64_t> r)
                          {
                              for (int64_t v = r.begin(); v < r.end(); v++)
                              {
                                  int64_t start = _indptr[v];
                                  int64_t end = _indptr[v + 1];
                                  for (int64_t i = start; i < end; i++)
                                  {
                                      _ret[i] = v;
                                  }
                              }
                          });

        return output_array;
    };

    std::vector<IndptrType> compact_indptr(const std::span<IndptrType> in_indptr,
                                           const std::span<uint8_t> flag)
    {
        int64_t v_num = in_indptr.size() - 1;
        int64_t e_num = flag.size();
        std::cout << "ReindexCSR e_num before compact = " << e_num << std::endl;
        auto _in_indices = flag.data();
        auto _in_indptr = in_indptr.data();
        std::vector<int64_t> degree(v_num + 1, 0);

        tbb::parallel_for(tbb::blocked_range<int64_t>(0, v_num),
                          [&](tbb::blocked_range<int64_t> r)
                          {
                              for (int64_t i = r.begin(); i < r.end(); i++)
                              {
                                  int64_t start = _in_indptr[i];
                                  int64_t end = _in_indptr[i + 1];
                                  for (int64_t j = start; j < end; j++)
                                  {
                                      degree.at(i) += (_in_indices[j]);
                                  }
                              }
                          });
        std::vector<IndptrType> ret(v_num + 1);
        auto out_indptr_start = ret.data();
        auto out_indptr_end = std::exclusive_scan(degree.begin(), degree.end(), out_indptr_start, 0ll);

        std::cout << "ReindexCSR e_num after = " << out_indptr_start[v_num] << std::endl;
        assert(*(out_indptr_end - 1) == out_indptr_start[v_num]);
        return ret;
    }

    std::tuple<std::vector<IndptrType>, std::vector<VertexIDType>, std::vector<WeightType>> make_sym(
        const std::span<IndptrType> in_indptr,
        const std::span<VertexIDType> in_indices,
        const std::span<WeightType> in_data)
    {
        if (in_data.size() == 0)
        {
            int64_t e_num = in_indices.size();
            int64_t v_num = in_indptr.size() - 1;
            std::cout << "MakeSym v_num: " << v_num << " | e_num: " << e_num << std::endl;
            typedef std::vector<Edge> EdgeVec;
            EdgeVec edge_vec;
            edge_vec.resize(e_num * 2);
            auto _in_indptr = in_indptr.data();
            auto _in_indices = in_indices.data();
            tbb::parallel_for(tbb::blocked_range<VertexIDType>(0, v_num),
                              [&](tbb::blocked_range<VertexIDType> r)
                              {
                                  for (auto v = r.begin(); v < r.end(); v++)
                                  {
                                      auto start = _in_indptr[v];
                                      auto end = _in_indptr[v + 1];
                                      for (auto i = start; i < end; i++)
                                      {
                                          auto u = _in_indices[i];
                                          if (u != v)
                                          {
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
            if (edge_vec.at(0)._src == edge_vec.at(0)._dst)
            {
                edge_vec.erase(edge_vec.begin());
            }
            int64_t cur_e_num = edge_vec.size();

            std::vector<IndptrType> indptr(v_num + 1);
            std::vector<std::atomic<IndptrType>> degree(v_num + 1);
            std::vector<VertexIDType> indices(cur_e_num);
            auto indices_ptr = indices.data();
            std::cout << "MakeSym compute degree" << std::endl;

            tbb::parallel_for(tbb::blocked_range<IndptrType>(0, cur_e_num),
                              [&](tbb::blocked_range<IndptrType> r)
                              {
                                  for (auto i = r.begin(); i < r.end(); i++)
                                  {
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
        }
        else
        {
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
            tbb::parallel_for(tbb::blocked_range<VertexIDType>(0, v_num),
                              [&](tbb::blocked_range<VertexIDType> r)
                              {
                                  for (auto v = r.begin(); v < r.end(); v++)
                                  {
                                      auto start = _in_indptr[v];
                                      auto end = _in_indptr[v + 1];
                                      for (auto i = start; i < end; i++)
                                      {
                                          auto u = _in_indices[i];
                                          auto d = _in_data[i];
                                          if (u != v)
                                          {
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
            if (edge_vec.at(0)._src == edge_vec.at(0)._dst)
            {
                edge_vec.erase(edge_vec.begin());
            }
            int64_t cur_e_num = edge_vec.size();
            std::vector<IndptrType> indptr(v_num + 1);
            std::vector<std::atomic<IndptrType>> degree(v_num + 1);
            std::vector<VertexIDType> indices(cur_e_num);
            std::vector<WeightType> retdata(cur_e_num);

            auto indices_ptr = indices.data();
            auto retdata_ptr = retdata.data();
            std::cout << "MakeSym compute degree" << std::endl;

            tbb::parallel_for(tbb::blocked_range<IndptrType>(0, cur_e_num),
                              [&](tbb::blocked_range<IndptrType> r)
                              {
                                  for (auto i = r.begin(); i < r.end(); i++)
                                  {
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
    }
}