#include "cnpy_mmap.h"
#include "utils.h"
#include "partition.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace cppmetis;

int main(int argc, const char** argv) {
    Args args = parse_args(argc, argv);
    auto locdata = load_dataset(args, false);
    auto partition_map = metis_assignment(args, locdata);
    cnpyMmap::npy_save(args.output_path, partition_map);

    std::ifstream statusFile("/proc/self/status");
    std::string line;

    size_t maxVmSize = 0; 

    while (std::getline(statusFile, line)) {
        if (line.find("VmPeak:") == 0) {  // Find the VmPeak line
            size_t vmPeak = std::stoul(line.substr(line.find(':') + 1));
            maxVmSize = std::max(maxVmSize, vmPeak); 
        }
    }

    std::cout << "Maximum virtual memory usage: " << 1.0 * maxVmSize / 1e6 << " GB\n";

}
