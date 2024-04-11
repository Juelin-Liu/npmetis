#include <cnpy.h>
#include "utils.h"
#include "mt_partition.h"
#include "command_line.h"

using namespace pmetis;

int main(int argc, const char** argv) {
    int64_t num_partitions = 0;
    std::string output_path;

    auto cmd = CommandLine(argc, argv);
    cmd.get_cmd_line_argument<int64_t>("num_partition", num_partitions);
    cmd.get_cmd_line_argument("output", output_path);
    bool obj_cut = cmd.check_cmd_line_flag("use_cut");

    auto locdata = load_dataset(argc, argv);
    auto partition_map = mt_metis_assignment(num_partitions, obj_cut, locdata->indptr, locdata->indices, locdata->node_weight, locdata->edge_weight);
    cnpy::npy_save(output_path, partition_map);
}
