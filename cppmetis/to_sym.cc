#include "cnpy_mmap.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

using namespace cppmetis;

int main(int argc, const char** argv) {
    Args args = parse_args(argc, argv);
    auto data = load_dataset(args, true);
    
    if (!std::filesystem::exists(args.output_path)) {
        std::cout << "Create directory: " << args.output_path << std::endl;
        std::filesystem::create_directories(args.output_path);
    }

    std::filesystem::path indptr_path = args.output_path;
    indptr_path /= "indptr_sym.npy";
    std::cout << "Saving indptr: " << indptr_path << std::endl;
    cnpyMmap::npy_save(indptr_path, data->indptr);

    std::filesystem::path indices_path = args.output_path;
    indices_path /= "indices_sym.npy";
    std::cout << "Saving indices: " << indices_path << std::endl;
    cnpyMmap::npy_save(indices_path, data->indices);


    if (!data->edge_weight.empty()) {
        std::filesystem::path edge_weight_path = args.output_path;
        edge_weight_path /= "edge_weight_sym.npy";
        std::cout << "Saving edge weight: " << edge_weight_path << std::endl;
        cnpyMmap::npy_save(edge_weight_path, data->edge_weight);
    }


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
