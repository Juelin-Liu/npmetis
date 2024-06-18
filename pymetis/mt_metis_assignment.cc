#define MTMETIS_64BIT_WEIGHTS
#define MTMETIS_64BIT_EDGES
#include "mt_metis_assignment.h"
#include <cassert>
#include <iostream>
#include <mtmetis.h>
#include <thread>
#include <numeric>

namespace pymetis
{

    std::vector<uint32_t> mt_metis_assignment(int64_t num_partition,
                                             int64_t num_iteration,
                                             int64_t num_initpart,
                                             float unbalance_val,
                                             bool obj_cut,
                                             std::span<idx_t> indptr,
                                             std::span<id_t> indices,
                                             std::span<wgt_t> node_weight,
                                             std::span<wgt_t> edge_weight)
    {
        const mtmetis_vtx_type nparts = num_partition;
        const mtmetis_vtx_type nvtxs = indptr.size() - 1;
        const int64_t num_edge = indices.size();
        mtmetis_vtx_type ncon = 1; // number of constraint
        if (node_weight.size())
        {
            mtmetis_vtx_type nvwgt = node_weight.size();
            ncon = nvwgt / nvtxs;
            assert(nvwgt % nvtxs == 0);
        };

        if (edge_weight.size())
        {
            assert(edge_weight.size() == num_edge);
        }

        std::vector<id_t> ret(nvtxs);
        auto part = reinterpret_cast<id_t *>(ret.data());
        auto xadj = reinterpret_cast<idx_t *>(indptr.data());
        auto adjncy = reinterpret_cast<const id_t *>(indices.data());

        wgt_t *vwgt = node_weight.empty() ? nullptr :  node_weight.data();
        wgt_t *ewgt = edge_weight.empty() ? nullptr : edge_weight.data();

        mtmetis_wgt_type objval = 0;
        std::vector<double> options(MTMETIS_NOPTIONS, MTMETIS_VAL_OFF);
        options[MTMETIS_OPTION_NTHREADS] = std::thread::hardware_concurrency();
        options[MTMETIS_OPTION_NITER] = num_iteration;
        options[MTMETIS_OPTION_NINITSOLUTIONS] = num_initpart;
        options[MTMETIS_OPTION_NPARTS] = nparts;
        options[MTMETIS_OPTION_VERBOSITY] = MTMETIS_VERBOSITY_HIGH;
        options[MTMETIS_OPTION_TIME] = 1;
        options[MTMETIS_OPTION_IGNORE] = MTMETIS_IGNORE_NONE;
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