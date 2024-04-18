#include "cnpy_mmap.h"
#include "utils.h"
#include "partition.h"

using namespace cppmetis;

int main(int argc, const char** argv) {
    Args args = parse_args(argc, argv);
    auto locdata = load_dataset(args);
    auto partition_map = metis_assignment(args, locdata);
    cnpyMmap::npy_save(args.output_path, partition_map);
}
