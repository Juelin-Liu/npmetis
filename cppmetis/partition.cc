#include "partition.h"
#include <iostream>
#include <metis.h>
#include <cassert>

namespace pmetis
{

    std::vector<VertexPIDType> metis_assignment(int64_t num_partitions, bool obj_cut,
                                                std::span<IndptrType> indptr,
                                                std::span<VertexIDType> indices,
                                                std::span<WeightType> node_weight,
                                                std::span<WeightType> edge_weight
                                                )
    {
        VertexPIDType nparts = num_partitions;
        VertexPIDType nvtxs = indptr.size() - 1;
        IndptrType num_edge = indices.size();
        VertexIDType ncon = 1; // number of constraint
        if (node_weight.size())
        {
            VertexIDType nvwgt = node_weight.size();
            ncon = nvwgt / nvtxs;
            std::cout << "nvwgt: " << nvwgt << " nvtxs: " << nvtxs << std::endl;
            assert(nvwgt % nvtxs == 0);
        };

        if (edge_weight.size())
        {
            assert(edge_weight.size() == num_edge);
        }

        std::cout << "metis_assignment num_part: " << num_partitions << std::endl;
        // std::cout << "indptr: " << indptr << std::endl;
        // std::cout << "indices: " << indices << std::endl;
        // std::cout << "node_weight: " << node_weight << std::endl;
        // std::cout << "edge_weight: " << edge_weight << std::endl;
        //     auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
        //     torch::Tensor ret = torch::empty(nvtxs, tensor_opts);

        std::vector<VertexPIDType> ret(nvtxs);
        auto *part = ret.data();

        // std::vector<int64_t> ret(nvtxs, 0);
        // int64_t *part = ret.data();
        auto xadj = indptr.data();
        auto adjncy = indices.data();

        WeightType *vwgt = nullptr;
        WeightType *ewgt = nullptr;

        if (node_weight.size())
            vwgt = node_weight.data();
        if (edge_weight.size())
            ewgt = edge_weight.data();

        WeightType objval = 0;
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NITER] = 3;
        options[METIS_OPTION_OBJTYPE] = obj_cut ? METIS_OBJTYPE_CUT : METIS_OBJTYPE_VOL;
        //     options[METIS_OPTION_NIPARTS] = 1;
        //     options[METIS_OPTION_DROPEDGES] = edge_weight.size(0) == 0;
        // options[METIS_OPTION_DBGLVL] = METIS_DBG_COARSEN | METIS_DBG_INFO | METIS_DBG_TIME;

        int flag = METIS_PartGraphKway(&nvtxs,
                                       &ncon,
                                       xadj,
                                       adjncy,
                                       vwgt,
                                       NULL,
                                       ewgt,
                                       &nparts,
                                       NULL, // tpwgts
                                       NULL, // ubvec
                                       options,
                                       &objval,
                                       part);

        if (obj_cut)
        {
            std::cout << "Partition a graph with " << nvtxs << " nodes and "
                      << num_edge << " edges into " << num_partitions << " parts and "
                      << "get " << objval << " edge cuts" << std::endl;
        }
        else
        {
            std::cout << "Partition a graph with " << nvtxs << " nodes and "
                      << num_edge << " edges into " << num_partitions << " parts and "
                      << "the communication volume is " << objval << std::endl;
        }

        switch (flag)
        {
        case METIS_OK:
            return ret;
        case METIS_ERROR_INPUT:
            std::cerr << "Error in Metis partitioning: invalid input";
        case METIS_ERROR_MEMORY:
            std::cerr << "Error in Metis partitioning: not enough memory";
        };
        exit(-1);
    };
}