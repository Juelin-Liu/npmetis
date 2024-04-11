#include "partition.h"
#include <cassert>
#include <iostream>
#include <parmetis.h>

namespace pmetis
{
    std::vector<VertexPIDType> mpi_metis_assignment(
                                                    int64_t num_partitions,
                                                    bool obj_cut,
                                                    std::span<IndptrType> vtxdist,
                                                    std::span<IndptrType> indptr,
                                                    std::span<VertexIDType> indices,
                                                    std::span<WeightType> node_weight,
                                                    std::span<WeightType> edge_weight)
    {
        VertexPIDType nparts = num_partitions;
        VertexPIDType nvtxs = indptr.size() - 1;
        IndptrType num_edge = indices.size();
        // ncon is used to specify the number of weights that each vertex has. It is also the number of balance
        // constraints that must be satisfied.
        VertexIDType ncon = 1;

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

        // std::cout << "metis_assignment num_part: " << num_partitions << std::endl;
        // std::cout << "indptr: " << indptr << std::endl;
        // std::cout << "indices: " << indices << std::endl;
        // std::cout << "node_weight: " << node_weight << std::endl;
        // std::cout << "edge_weight: " << edge_weight << std::endl;
        //     auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
        //     torch::Tensor ret = torch::empty(nvtxs, tensor_opts);

        std::vector<VertexPIDType> ret(nvtxs);
        auto part = ret.data();

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

        // options This is an array of integers that is used to pass additional parameters for the routine. The first element
        // (i.e., options[0]) can take either the value of 0 or 1. If it is 0, then the default values are used,
        // otherwise the remaining two elements of options are interpreted as follows:
        // options[1] This specifies the level of information to be returned during the execution of the
        // algorithm. Timing information can be obtained by setting this to 1. Additional
        // options for this parameter can be obtained by looking at parmetis.h. The nu-
        // merical values there should be added to obtain the correct value. The default value
        // is 0.
        // options[2] This is the random number seed for the routine.
        idx_t options[3] = {1, 1, 42};

        // wgtflag: This is used to indicate if the graph is weighted. wgtflag can take one of four values:
        // 0 No weights (vwgt and adjwgt are both NULL).
        // 1 Weights on the edges only (vwgt is NULL).
        // 2 Weights on the vertices only (adjwgt is NULL).
        // 3 Weights on both the vertices and edges.
        idx_t wgtflag{-1};
        if (edge_weight.size() && node_weight.size())
            wgtflag = 3;
        else if (edge_weight.size())
            wgtflag = 1;
        else if (node_weight.size())
            wgtflag = 2;
        else
            wgtflag = 0;

        // numflag is used to indicate the numbering scheme that is used for the vtxdist, xadj, adjncy, and part
        // arrays. numflag can take one of two values:
        // 0 C-style numbering that starts from 0.
        // 1 Fortran-style numbering that starts from 1.
        idx_t numflag = 0;

        // tpwgts: array of size ncon × nparts that is used to specify the fraction of vertex weight that should
        // be distributed to each sub-domain for each balance constraint. If all of the sub-domains are to be of
        // the same size for every vertex weight, then each of the ncon ×nparts elements should be set to
        // a value of 1 / nparts. If ncon is greater than 1, the target sub-domain weights for each sub-domain
        // are stored contiguously (similar to the vwgt array). Note that the sum of all of the tpwgts for a
        // give vertex weight should be one.
        std::vector<real_t> tpwgts(ncon * num_partitions, 1.0 / nparts);

        // ubvec: An array of size ncon that is used to specify the imbalance tolerance for each vertex weight, with 1
        // being perfect balance and nparts being perfect imbalance. A value of 1.05 for each of the ncon
        // weights is recommended.
        std::vector<real_t> ubvec(ncon, 1.05);
        MPI_Comm comm = MPI_COMM_WORLD;
        int flag = ParMETIS_V3_PartKway(vtxdist.data(),
                                        xadj,
                                        adjncy,
                                        vwgt,
                                        ewgt,
                                        &wgtflag,
                                        &numflag,
                                        &ncon,
                                        &nparts,
                                        tpwgts.data(), // tpwgts
                                        ubvec.data(),  // ubvec
                                        options,
                                        &objval,
                                        part,
                                        &comm);

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