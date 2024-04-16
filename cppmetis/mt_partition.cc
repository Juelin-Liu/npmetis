#define MTMETIS_64BIT_VERTICES
#define MTMETIS_64BIT_EDGES
#define MTMETIS_64BIT_WEIGHTS
#define MTMETIS_64BIT_PARTITIONS

#include "mt_partition.h"
#include <cassert>
#include <iostream>
#include <mtmetis.h>
#include <thread>

namespace pmetis
{

    std::vector<int64_t> mt_metis_assignment(int64_t num_partition,
                                             int64_t num_iteration,
                                             int64_t num_initpart,
                                             float unbalance_val,
                                             bool obj_cut,
                                             std::span<IndptrType> vtxdist,
                                             std::span<int64_t> indptr,
                                             std::span<int64_t> indices,
                                             std::span<int64_t> node_weight,
                                             std::span<int64_t> edge_weight)
    {
        const mtmetis_vtx_type nparts = num_partition;
        const mtmetis_vtx_type nvtxs = indptr.size() - 1;
        const mtmetis_adj_type num_edge = indices.size();
        mtmetis_vtx_type ncon = 1; // number of constraint
        if (node_weight.size())
        {
            VertexIDType nvwgt = node_weight.size();
            ncon = nvwgt / nvtxs;
            // std::cout << "nvwgt: " << nvwgt << " nvtxs: " << nvtxs << std::endl;
            assert(nvwgt % nvtxs == 0);
        };

        if (edge_weight.size())
        {
            assert(edge_weight.size() == num_edge);
        }

        // std::cout << "metis_assignment num_part: " << num_partition << std::endl;
        // std::cout << "indptr: " << indptr << std::endl;
        // std::cout << "indices: " << indices << std::endl;
        // std::cout << "node_weight: " << node_weight << std::endl;
        // std::cout << "edge_weight: " << edge_weight << std::endl;
        //     auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
        //     torch::Tensor ret = torch::empty(nvtxs, tensor_opts);

        std::vector<int64_t> ret(nvtxs);
        auto part = reinterpret_cast<mtmetis_pid_type *>(ret.data());

        // std::vector<int64_t> ret(nvtxs, 0);
        // int64_t *part = ret.data();
        auto xadj = reinterpret_cast<const mtmetis_adj_type *>(indptr.data());
        auto adjncy = reinterpret_cast<const mtmetis_vtx_type *>(indices.data());

        WeightType *vwgt = nullptr;
        WeightType *ewgt = nullptr;

        if (node_weight.size())
            vwgt = node_weight.data();
        if (edge_weight.size())
            ewgt = edge_weight.data();

        mtmetis_wgt_type objval = 0;
        std::vector<double> options(MTMETIS_NOPTIONS, MTMETIS_VAL_OFF);
        options[MTMETIS_OPTION_NTHREADS] = std::thread::hardware_concurrency();
        options[MTMETIS_OPTION_NITER] = num_iteration;
        options[MTMETIS_OPTION_NINITSOLUTIONS] = num_initpart;
        options[MTMETIS_OPTION_NPARTS] = nparts;
        options[MTMETIS_OPTION_TIME] = 1;
        // tpwgts: array of size ncon × nparts that is used to specify the fraction of vertex weight that should
        // be distributed to each sub-domain for each balance constraint. If all of the sub-domains are to be of
        // the same size for every vertex weight, then each of the ncon ×nparts elements should be set to
        // a value of 1 / nparts. If ncon is greater than 1, the target sub-domain weights for each sub-domain
        // are stored contiguously (similar to the vwgt array). Note that the sum of all of the tpwgts for a
        // give vertex weight should be one.
        std::vector<mtmetis_real_type> tpwgts(ncon * nparts, 1.0 / nparts);

        // ubvec: An array of size ncon that is used to specify the imbalance tolerance for each vertex weight, with 1
        // being perfect balance and nparts being perfect imbalance. A value of 1.05 for each of the ncon
        // weights is recommended.
        std::vector<mtmetis_real_type> ubvec(ncon, unbalance_val);

        int flag = MTMETIS_PartGraphKway(&nvtxs,
                                         &ncon,
                                         xadj,
                                         adjncy,
                                         vwgt,
                                         NULL, // vsize not used
                                         ewgt,
                                         &nparts,
                                         tpwgts.data(), // tpwgts
                                         ubvec.data(),  // ubvec
                                         options.data(),
                                         &objval,
                                         part);

        if (obj_cut)
        {
            std::cout << "Partition a graph with " << nvtxs << " nodes and "
                      << num_edge << " edges into " << num_partition << " parts and "
                      << "get " << objval << " edge cuts" << std::endl;
        }
        else
        {
            std::cout << "Partition a graph with " << nvtxs << " nodes and "
                      << num_edge << " edges into " << num_partition << " parts and "
                      << "the communication volume is " << objval << std::endl;
        }

        switch (flag)
        {
        case MTMETIS_SUCCESS:
            return ret;
        case MTMETIS_ERROR_INVALIDINPUT:
            std::cerr << "Error in Metis partitioning: invalid input";
        case MTMETIS_ERROR_NOTENOUGHMEMORY:
            std::cerr << "Error in Metis partitioning: not enough memory";
        case MTMETIS_ERROR_THREADING:
            std::cerr << "Error in Metis partitioning: threading";
        };
        exit(-1);
    };
}