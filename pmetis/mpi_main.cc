#include "command_line.h"
#include "metis.h"
#include "mpi_partition.h"
#include "utils.h"
#include <algorithm>
#include <cnpy.h>
#include <memory>

enum tag {
    indptr = 0,
    indices = 1,
    node_weight = 2,
    edge_weight = 3,
    partition_map = 4
};

struct dataset
{
    std::vector<IndptrType> vtxdist;
    std::vector<VertexIDType> indptr;
    std::vector<IndptrType> indices;
    std::vector<WeightType> node_weight;
    std::vector<WeightType> edge_weight;
};

std::unique_ptr<dataset> load_dataset(int argc, const char **argv)
{
    std::string data_dir = "/mnt/homes/juelinliu/sc24/pymetis/dataset/orkut/";
    std::string output_dir = "/mnt/homes/juelinliu/sc24/pymetis/dataset/partition_maps/";
    std::string indptr_path, indices_path, output_path, node_weight_path, edge_weight_path;
    bool obj_cut{false};
    int64_t num_partitions{-1};

    auto cmd = CommandLine(argc, argv);
    cmd.get_cmd_line_argument<int64_t>("num_partition", num_partitions, 4);
    obj_cut = cmd.check_cmd_line_flag("use_cut");

    cmd.get_cmd_line_argument<std::string>("indptr", indptr_path, data_dir + "indptr_sym.npy");
    cmd.get_cmd_line_argument<std::string>("indices", indices_path, data_dir + "indices_sym.npy");
    cmd.get_cmd_line_argument<std::string>("output", output_path, output_dir + "orkut_w4_ndst_efreq_xcut_xbal.npy");
    cmd.get_cmd_line_argument<std::string>("node_weight", node_weight_path, data_dir + "dst_node_weight.npy");
    cmd.get_cmd_line_argument<std::string>("edge_weight", edge_weight_path, data_dir + "edge_weight.npy");

    // std::string train_node_path, valid_node_path, test_node_path;
    // cmd.get_cmd_line_argument<std::string>("train_node", train_node_path);
    // cmd.get_cmd_line_argument<std::string>("valid_node", valid_node_path);
    // cmd.get_cmd_line_argument<std::string>("test_node", test_node_path);

    assert(indptr_path.size() > 0);
    assert(indices_path.size() > 0);
    assert(output_path.size() > 0);
    assert(num_partitions > 0);

    std::cout << "Cmd options:\n";
    std::cout << "use_cut: " << obj_cut << std::endl;
    std::cout << "num_partition: " << num_partitions << std::endl;
    std::cout << "indptr: " << indptr_path << std::endl;
    std::cout << "indices: " << indices_path << std::endl;

    cnpy::NpyArray indptr = cnpy::npy_load(indptr_path);
    cnpy::NpyArray indices = cnpy::npy_load(indices_path);
    cnpy::NpyArray train_node, test_node, valid_node;
    cnpy::NpyArray node_weight, edge_weight;

    if (node_weight_path.size())
    {
        std::cout << "Use node weight: " << node_weight_path << std::endl;
        node_weight = cnpy::npy_load(node_weight_path);
    }

    if (edge_weight_path.size())
    {
        std::cout << "Use edge weight: " << edge_weight_path << std::endl;
        edge_weight = cnpy::npy_load(edge_weight_path);
    }

    std::vector<VertexPIDType> assignment;
    if (indptr_path.ends_with("indptr_xsym.npy"))
    {
        assert(indices_path.ends_with("indices_xsym.npy"));
        auto [sym_indptr, sym_indice, sym_edge_weight] = pmetis::make_sym({indptr.data<IndptrType>(), indptr.num_vals},
                                                                          {indices.data<VertexIDType>(), indices.num_vals},
                                                                          {edge_weight.data<WeightType>(), indices.num_vals});

        auto ret = std::make_unique<dataset>();
        ret->indptr = std::move(sym_indptr);
        ret->indices = std::move(sym_indice);
        ret->edge_weight = std::move(sym_edge_weight);
        ret->node_weight = node_weight.as_vec<WeightType>();
        return ret;
    }
    else
    {
        assert(indices_path.ends_with("indices_sym.npy"));
        auto ret = std::make_unique<dataset>();
        ret->indptr = indptr.as_vec<IndptrType>();
        ret->indices = indices.as_vec<VertexIDType>();
        ret->edge_weight = edge_weight.as_vec<WeightType>();
        ret->node_weight = node_weight.as_vec<WeightType>();
        return ret;
    }
}

std::vector<idx_t> get_vtxdist(std::unique_ptr<dataset> &data, int world_size)
{
    std::vector<idx_t> vtxdist(1, 0);
    idx_t num_edges = data->indices.size();
    idx_t num_nodes = data->indptr.size() - 1;
    for (int r = 0; r < world_size; r++)
    {
        if (r == world_size - 1)
        {
            vtxdist.push_back(num_nodes);
        }
        else
        {
            idx_t loc_num_edges = (num_edges / world_size) * (r + 1);
            auto begin = data->indptr.begin();
            auto itr = std::upper_bound(data->indptr.begin(), data->indptr.end(), loc_num_edges);
            vtxdist.push_back(itr - begin);
        }
    }
    data->vtxdist = vtxdist;
    return vtxdist;
}

std::vector<idx_t> get_edgedist(std::unique_ptr<dataset> &data, int world_size)
{
    assert(data->vtxdist.size() == world_size + 1);
    std::vector<idx_t> edgedist;
    idx_t num_edges = data->indices.size();
    idx_t num_nodes = data->indptr.size() - 1;
    for (int recv_rank = 0; recv_rank < world_size; recv_rank++)
    {
        idx_t start_nd = data->vtxdist.at(recv_rank);
        idx_t end_nd = data->vtxdist.at(recv_rank + 1);

        idx_t start_eidx = data->indptr.at(start_nd);
        idx_t end_eidx = data->indptr.at(end_nd);
        edgedist.push_back(start_eidx);
    }
    edgedist.push_back(num_edges);
    return edgedist;
}

/**
 * @brief send indices / edge_weight to other ranks
 *
 * @param vtxdist
 * @param edata indices / edge_weight
 * @param send_rank
 * @param world_size
 * @return std::vector<MPI_Request* >
 */
std::shared_ptr<std::vector<MPI_Request>> send_data(int send_rank, int world_size, const std::unique_ptr<dataset> &data)
{
    assert(data->vtxdist.size() == world_size + 1);
    assert(send_rank == 0);
    auto requests = std::make_shared<std::vector<MPI_Request>>(2);
    idx_t num_nodes = data->indptr.size() - 1;
    idx_t num_edges = data->indices.size();

    for (int recv_rank = 0; recv_rank < world_size; recv_rank++)
    {
        idx_t start_nd = data->vtxdist.at(recv_rank);
        idx_t end_nd = data->vtxdist.at(recv_rank + 1);

        idx_t start_eidx = data->indptr.at(start_nd);
        idx_t end_eidx = data->indptr.at(end_nd);
        // send indptr
        std::vector<idx_t> send_indptr;
        for (size_t i = start_nd; i <= end_nd; i++)
            send_indptr.push_back(data->indptr.at(i) - start_eidx);
        MPI_Isend(send_indptr.data(), send_indptr.size(), MPI_INT64_T, recv_rank, tag::indptr, MPI_COMM_WORLD, &requests->at(0));

        // send indices
        MPI_Isend(data->indices.data() + start_eidx, end_eidx - start_eidx, MPI_INT64_T, recv_rank, tag::indices, MPI_COMM_WORLD, &requests->at(1));

        // send node_weight
        if (data->node_weight.size() > 0)
        {
            requests->push_back(0);
            assert(data->node_weight.size() == num_nodes);
            MPI_Isend(data->node_weight.data() + start_nd, end_nd - start_nd, MPI_INT64_T, recv_rank, tag::node_weight, MPI_COMM_WORLD, &requests->at(requests->size()-1));
        }

        // send edge_weight
        if (data->edge_weight.size() > 0)
        {
            requests->push_back(0);
            assert(data->edge_weight.size() == num_edges);
            MPI_Isend(data->edge_weight.data() + start_eidx, end_eidx - start_eidx, MPI_INT64_T, recv_rank, tag::edge_weight, MPI_COMM_WORLD, &requests->at(requests->size()-1));
        }
    }

    return requests;
};

std::shared_ptr<std::vector<MPI_Request>> recv_data(int send_rank, int recv_rank, bool use_edge_weight, bool use_node_weight,
                                    const std::vector<idx_t> &vtxdist, const std::vector<idx_t> &edgedist,
                                    std::unique_ptr<dataset>& data)
{
    std::cout << "rank " << recv_rank << " recv data from " << send_rank << std::endl; 
    auto requests = std::make_shared<std::vector<MPI_Request>>(2);
    int status;
    data->vtxdist = vtxdist;

    idx_t node_recv_num = vtxdist.at(recv_rank + 1) - vtxdist.at(recv_rank);
    idx_t edge_recv_num = edgedist.at(recv_rank + 1) - edgedist.at(recv_rank);

    // recv indptr
    data->indptr.resize(node_recv_num + 1);
    status = MPI_Irecv(data->indptr.data(), node_recv_num + 1, MPI_INT64_T, send_rank, tag::indptr, MPI_COMM_WORLD, &requests->at(0));
    assert(status == MPI_SUCCESS);

    // recv indices
    data->indices.resize(edge_recv_num);
    status = MPI_Irecv(data->indices.data(), edge_recv_num, MPI_INT64_T, send_rank, tag::indices, MPI_COMM_WORLD, &requests->at(1));
    assert(status == MPI_SUCCESS);

    if (use_node_weight) {
        requests->push_back(0);
        data->node_weight.reserve(node_recv_num);
        status = MPI_Irecv(data->node_weight.data(), node_recv_num, MPI_INT64_T, send_rank, tag::node_weight, MPI_COMM_WORLD, &requests->at(requests->size()-1));
        assert(status == MPI_SUCCESS);
    }
    if (use_edge_weight) {
        requests->push_back(0);
        data->edge_weight.reserve(edge_recv_num);
        status = MPI_Irecv(data->edge_weight.data(), edge_recv_num, MPI_INT64_T, send_rank, tag::edge_weight, MPI_COMM_WORLD, &requests->at(requests->size()-1));
        assert(status == MPI_SUCCESS);
    }
    std::cout << "rank " << recv_rank << " recv " << data->indptr.size() - 1 << " nodes and " << data->indices.size() << " edges from " << send_rank << std::endl; 
    return requests;
};

void log(int rank, std::string name, const std::vector<idx_t>& vtxdist)
{
    std::string log = name + " rank " + std::to_string(rank) + ": ";
    for (auto d : vtxdist)
    {
        log += std::to_string(d) + " ";
    }
    std::cout << log << std::endl;
}

int main(int argc, char **argv)
{
    int rank{0}, world_size{0};
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::unique_ptr<dataset> alldata = std::make_unique<dataset>();
    std::unique_ptr<dataset> locdata = std::make_unique<dataset>();
    std::vector<idx_t> vtxdist(world_size + 1, 0);
    std::vector<idx_t> edgedist(world_size + 1, 0);
    bool use_node_weight, use_edge_weight;

    if (rank == 0)
    {
        alldata = load_dataset(argc, const_cast<const char **>(argv));
        vtxdist = get_vtxdist(alldata, world_size);
        edgedist = get_edgedist(alldata, world_size);
        use_edge_weight = alldata->edge_weight.size() > 0;
        use_node_weight = alldata->node_weight.size() > 0;
    }
    // broadcast vtxdist to all ranks
    MPI_Bcast(vtxdist.data(), world_size + 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(edgedist.data(), world_size + 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_edge_weight, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_node_weight, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    log(rank, "num_adj_dist", vtxdist);
    log(rank, "num_edge_dist", edgedist);

    // rank 0 send partitioned graph to all the ranks
    pmetis::Timer t;
    t.start();

    std::shared_ptr<std::vector<MPI_Request>> send_requests, recv_requests;
    if (rank == 0) {
        send_requests = send_data(rank, world_size, alldata);
    } 

    recv_requests = recv_data(0, rank, use_edge_weight, use_node_weight, vtxdist, edgedist, locdata);

    std::cout << "rank " << rank << " start waiting" << std::endl;
    MPI_Status recv_statuses[recv_requests->size()];
    MPI_Waitall(recv_requests->size(), recv_requests->data(), recv_statuses);
    if (rank == 0) {
        MPI_Status send_statuses[send_requests->size()];
        MPI_Waitall(send_requests->size(), send_requests->data(), send_statuses);
    }
    t.end();

    size_t bytes_recv = locdata->indices.size() + locdata->indptr.size() + locdata->node_weight.size() + locdata->edge_weight.size();
    bytes_recv *= sizeof(idx_t);
    size_t mb_recv = bytes_recv / 1000000;
    std::cout << "rank " << rank << " recv " << mb_recv << " MB in " << t.nanosec() / 1e9 << " secs bandwidth: " << mb_recv * 1e9 / t.nanosec() << " MB/s" << std::endl;
    if (rank == 0) alldata.reset();


    // start metis partitioning
    auto cmd = CommandLine(argc, const_cast<const char**>(argv));
    MPI_Comm comm = MPI_COMM_WORLD;
    int64_t num_partitions = 0;
    std::string output_path;
    
    cmd.get_cmd_line_argument("output", output_path);
    bool use_cut = cmd.check_cmd_line_flag("use_cut");
    cmd.get_cmd_line_argument<int64_t>("num_partition", num_partitions, 4);
    
    // TODO: fix the bug in this function call    
    auto part = pmetis::mpi_metis_assignment(&comm, num_partitions, use_cut, locdata->vtxdist, locdata->indptr, locdata->indices, locdata->node_weight, locdata->edge_weight);

    MPI_Request partition_map_send_request;
    MPI_Isend(part.data(), part.size(), MPI_INT64_T, 0, tag::partition_map, MPI_COMM_WORLD, &partition_map_send_request);
    if (rank == 0) {
        idx_t num_nodes = locdata->vtxdist.at(world_size);
        std::vector<idx_t> partition_map(num_nodes);
        for (int send_rank = 0; send_rank < world_size; send_rank++) {
            idx_t start_idx = locdata->vtxdist.at(send_rank);
            idx_t end_idx = locdata->vtxdist.at(send_rank + 1);
            idx_t recv_cnt = end_idx - start_idx;
            int status = MPI_Recv(partition_map.data() + start_idx, recv_cnt, MPI_INT64_T, send_rank, tag::partition_map, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(status == MPI_SUCCESS);
        }
        cnpy::npy_save(output_path, partition_map);
    } 
    MPI_Finalize();

    return 0;
}
