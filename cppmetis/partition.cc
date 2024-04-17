#include "partition.h"
#include <iostream>
#include <metis.h>
#include <cassert>
#include <numeric>

namespace cppmetis
{

    /**
     * @brief metis partition wrapper
     *
     * @param num_partition: number of partitions in the graph
     * @param num_iteration: number of iterations for
     * @param num_initpart: number of initial partitions
     * @param obj_cut: use vol / cut
     * @param vtxdist: number of vertex in each partition
     * @param indptr: local indptr for this rank
     * @param indices: local indices for this rank
     * @param node_weight: local node weight for this rank
     * @param edge_weight: local edge weights for this rank
     * @return std::vector<idx_t> local partition map
     */
    std::vector<idx_t> metis_assignment(int64_t num_partition,
                                                int64_t num_iteration,
                                                int64_t num_initpart,
                                                float unbalance_val,
                                                bool obj_cut,
                                                std::span<idx_t> vtxdist,
                                                std::span<idx_t> indptr,
                                                std::span<idx_t> indices,
                                                std::span<WeightType> node_weight,
                                                std::span<WeightType> edge_weight)
    {
        idx_t nparts = num_partition;
        idx_t nvtxs = indptr.size() - 1;
        idx_t num_edge = indices.size();
        idx_t ncon = 1; // number of constraint
        if (node_weight.size())
        {
            idx_t nvwgt = node_weight.size();
            ncon = nvwgt / nvtxs;
            std::cout << "nvwgt: " << nvwgt << " nvtxs: " << nvtxs << std::endl;
            assert(nvwgt % nvtxs == 0);
        };

        if (edge_weight.size())
        {
            assert(edge_weight.size() == num_edge);
        }

        std::cout << "metis_assignment num_part: " << num_partition << std::endl;
        // std::cout << "indptr: " << indptr << std::endl;
        // std::cout << "indices: " << indices << std::endl;
        // std::cout << "node_weight: " << node_weight << std::endl;
        // std::cout << "edge_weight: " << edge_weight << std::endl;
        //     auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
        //     torch::Tensor ret = torch::empty(nvtxs, tensor_opts);

        std::vector<idx_t> ret(nvtxs);
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
        options[METIS_OPTION_ONDISK] = 1;
        options[METIS_OPTION_NITER] = num_iteration;
        options[METIS_OPTION_OBJTYPE] = obj_cut ? METIS_OBJTYPE_CUT : METIS_OBJTYPE_VOL;
        options[METIS_OPTION_DROPEDGES] = ewgt == nullptr;
         options[METIS_OPTION_NIPARTS] = num_initpart;
        // options[METIS_OPTION_DBGLVL] = METIS_DBG_COARSEN | METIS_DBG_INFO | METIS_DBG_TIME;


        // tpwgts: array of size ncon × nparts that is used to specify the fraction of vertex weight that should
        // be distributed to each sub-domain for each balance constraint. If all of the sub-domains are to be of
        // the same size for every vertex weight, then each of the ncon ×nparts elements should be set to
        // a value of 1 / nparts. If ncon is greater than 1, the target sub-domain weights for each sub-domain
        // are stored contiguously (similar to the vwgt array). Note that the sum of all of the tpwgts for a
        // give vertex weight should be one.
        std::vector<float> tpwgts(ncon * num_partition, 1.0 / nparts);

        // ubvec: An array of size ncon that is used to specify the imbalance tolerance for each vertex weight, with 1
        // being perfect balance and nparts being perfect imbalance. A value of 1.05 for each of the ncon
        // weights is recommended.
        std::vector<float> ubvec(ncon, unbalance_val);

        int flag = METIS_PartGraphKway(&nvtxs,
                                       &ncon,
                                       xadj,
                                       adjncy,
                                       vwgt,
                                       NULL,
                                       ewgt,
                                       &nparts,
                                       tpwgts.data(), // tpwgts
                                       ubvec.data(), // ubvec
                                       options,
                                       &objval,
                                       part);
        float obj_scale = 1.0;
        if (ewgt != nullptr) {
            obj_scale *= std::accumulate(ewgt, ewgt + num_edge, 0ul) / num_edge;
        }
        objval /= obj_scale;

        if (obj_cut)
        {
            std::cout << "Partition a graph with " << nvtxs << " nodes and "
                      << num_edge << " edges into " << num_partition << " parts and "
                      << "get " << objval << " edge cuts with scale " << obj_scale << std::endl;
        }
        else
        {
            std::cout << "Partition a graph with " << nvtxs << " nodes and "
                      << num_edge << " edges into " << num_partition << " parts and "
                      << "the communication volume is " << objval << " with scale " << obj_scale << std::endl;
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