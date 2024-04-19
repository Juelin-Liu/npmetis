#include "utils.h"
#include "mpi_partition.h"
#include "cnpy_mmap.h"
#include <oneapi/tbb/global_control.h>

using namespace cppmetis;
enum tag
{
    indptr = 0,
    indices = 1,
    node_weight = 2,
    edge_weight = 3,
    partition_map = 4
};

void log(int rank, const std::string& name, const std::vector<idx_t> &vec)
{
    std::string log = name + " [" + std::to_string(rank) + "]: ";
    for (auto d : vec)
    {
        log += std::to_string(d) + " ";
    }
    std::cout << log << std::endl;
}

void log(int rank, int world_size, std::string msg) {
    MPI_Barrier(MPI_COMM_WORLD);
   int log_rank = 0;
   while (log_rank < world_size)
   {
       if (rank == log_rank)
       {
           std::cout << msg << std::endl;
       };
       log_rank++;
       MPI_Barrier(MPI_COMM_WORLD);
   }
};

int main(int argc, const char** argv) {
    oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1); // limit to one thread per process
    int rank{0}, world_size{0};
    MPI_Init(&argc, const_cast<char ***>(&argv));
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the processor name (hostname)
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    Args args = parse_args(argc, argv, (rank==0));

    {
        std::stringstream ss;
        ss << "Rank " << rank << " of " << world_size << " on host " << processor_name;
        log(rank, world_size, ss.str());
    }

    auto local_data = get_local_data(args, rank, world_size);

    {
        std::stringstream ss;
        ss << "Rank " << rank << " loaded dataset " ;
        log(rank, world_size, ss.str());
    }

    auto local_partition_map = mpi_metis_assignment(args, local_data);

    {
        std::stringstream ss;
        ss << "Rank " << rank << " finished metis partitioning ";
        log(rank, world_size, ss.str());
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Request partition_map_send_request;
    MPI_Isend(local_partition_map.data(), local_partition_map.size(), MPI_LONG, 0, tag::partition_map, MPI_COMM_WORLD, &partition_map_send_request);
    if (rank == 0)
    {
        idx_t num_nodes = local_data->vtxdist.at(world_size);
        std::vector<idx_t> partition_map(num_nodes);
        for (int send_rank = 0; send_rank < world_size; send_rank++)
        {
            idx_t start_idx = local_data->vtxdist.at(send_rank);
            idx_t end_idx = local_data->vtxdist.at(send_rank + 1);
            idx_t recv_cnt = end_idx - start_idx;
            int status = MPI_Recv(partition_map.data() + start_idx, recv_cnt, MPI_LONG, send_rank, tag::partition_map, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(status == MPI_SUCCESS);
        }
        cnpyMmap::npy_save(args.output_path, partition_map);
    }
    MPI_Finalize();
}
