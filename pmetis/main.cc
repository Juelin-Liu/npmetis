#include <cnpy.h>
#include <algorithm>
#include "utils.h"
#include "partition.h"
#include "command_line.h"

int main(int argc, const char** argv) {
    auto cmd = CommandLine(argc, argv);
    std::string indptr_path, indices_path, train_node_path, output_path, node_weight_path, edge_weight_path;
    bool obj_cut{false};
    int64_t num_partitions{-1};

    std::cout << "Cmd options:\n";

    cmd.get_cmd_line_argument<std::string>("indptr", indptr_path);
    cmd.get_cmd_line_argument<std::string>("indices", indices_path);
    cmd.get_cmd_line_argument<std::string>("train_node", train_node_path);
    cmd.get_cmd_line_argument<std::string>("output", output_path);
    cmd.get_cmd_line_argument<int64_t>("num_partition", num_partitions);
    obj_cut = cmd.check_cmd_line_flag("use_cut");

    cmd.get_cmd_line_argument<std::string>("node_weight", node_weight_path);
    cmd.get_cmd_line_argument<std::string>("edge_weight", edge_weight_path);


    assert(indptr_path.size() > 0);
    assert(indices_path.size() > 0);
    assert(output_path.size() > 0);
    assert(num_partitions > 0);

    std::cout << "indptr: " << indptr_path << std::endl;
    std::cout << "indices: " << indices_path << std::endl;
    std::cout << "use_cut: " << obj_cut << std::endl;
    std::cout << "num_partition: " << num_partitions << std::endl;
    
    cnpy::NpyArray indptr = cnpy::npy_load(indptr_path);
    cnpy::NpyArray indices = cnpy::npy_load(indices_path);
    cnpy::NpyArray train_node, node_weight, edge_weight;

    if (train_node_path.size()) {
        train_node = cnpy::npy_load(train_node_path);
    }

    if (node_weight_path.size()) {
        std::cout << "Use node weight: " << node_weight_path << std::endl;
        node_weight = cnpy::npy_load(node_weight_path);

        if (train_node.num_vals > 0) {
            std::cout << "Use train_node to mask node weight: " << train_node_path << std::endl;
            VertexIDType nvtxs = indptr.num_vals - 1;
            std::vector<bool> mask(nvtxs, 0);
            auto train_node_span = std::span<VertexIDType>{train_node.data<VertexIDType>(), train_node.num_vals};
            auto node_weight_span = std::span<WeightType>{node_weight.data<WeightType>(), node_weight.num_vals};

            for (auto tnode: train_node_span){
                mask.at(tnode) = 1;
            }

            for (VertexIDType i = 0; i < nvtxs; i++) {
                node_weight_span[i] *= mask.at(i);
            }
        }
    }

    if (edge_weight_path.size()) {
        std::cout << "Use edge weight: " << edge_weight_path << std::endl;
        edge_weight = cnpy::npy_load(edge_weight_path);
    }

    std::vector<VertexPIDType> assignment;
    std::cout << "start metis partitioning\n";
    if (indptr_path.ends_with("indptr_xsym.npy")) {
        assert(indices_path.ends_with("indices_xsym.npy"));
        auto [sym_indptr, sym_indice, sym_edge_weight] = pmetis::make_sym({indptr.data<IndptrType>(), indptr.num_vals},
                                                                          {indices.data<VertexIDType>(), indices.num_vals},
                                                                          {edge_weight.data<WeightType>(), indices.num_vals});

        assignment = pmetis::metis_assignment(num_partitions, obj_cut,
            sym_indptr,
            sym_indice, 
            {node_weight.data<WeightType>(), node_weight.num_vals}, 
            sym_edge_weight);
    } else {
        assert(indices_path.ends_with("indices_sym.npy"));
        assignment = pmetis::metis_assignment(num_partitions, obj_cut,
            {indptr.data<IndptrType>(), indptr.num_vals},
            {indices.data<VertexIDType>(), indices.num_vals}, 
            {node_weight.data<WeightType>(), node_weight.num_vals}, 
            {edge_weight.data<WeightType>(), edge_weight.num_vals});
    }

    cnpy::npy_save(output_path, assignment);
}
