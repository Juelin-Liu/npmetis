# Project: Metis Wrapper

# Overview:
This project provides a user-friendly C++ wrapper around the robust Metis K-Way graph partitioning algorithm, enabling seamless usage with versatile NumPy arrays for input and output data representation.

# Objectives:
Metis Integration: Core functionality focuses on encapsulating the Metis K-Way partitioning API within a well-designed C++ interface and provides easy-to-use binary.
NumPy Compatibility: Implement conversion mechanisms between Metis data structures and NumPy arrays, ensuring effortless data flow in common scientific computing pipelines.

Components:
1. C++ Files `cppmetis`: Define classes, functions, and data types for representing graphs (adjacency structures) and interacting with the Metis library.
2. Wrapper functions for Metis API calls.
3. Build scripts: Manage project compilation and linking, including dependencies on Metis and NumPy libraries.
4. Metis v5.0, ParMetis v4.0, MT-Metis v0.7, cnpy, oneTBB

# Potential Use Cases:
You want to use Metis to conduct kWay partitioning but found it difficult to use/slow.

# Prerequisites:
## Single Threaded: 
C++ compiler (GCC >=9 ), Ninja, CMake.

## Multi-Threaded:
OpenMP

## Distributed:
MPI, possible solution:
```bash
sudo apt install mpich -y
```

# How to compile:

For single-threaded and multiple-threaded binaries:
```bash
./build.sh
```

To use ParMetis (distributed), you must agree to its [Licence](https://github.com/KarypisLab/ParMETIS/blob/main/LICENSE).
First download its source code from Github:
```bash
wget https://github.com/KarypisLab/ParMETIS/archive/refs/heads/main.zip -O third_party/parmetis.zip
pushd third_party && unzip parmetis.zip && mv ParMETIS-main parmetis && rm parmetis.zip && popd
```
Then, uncomment the code blocks in `build.sh` after `build ParMETIS`, also uncomment the part in `cppmetis/CMakeLists.txt` after `Build mpi_main start`
`.

The output binary files will be in the `./bin` directory.

# How to use:
```shell
./bin/main \
--num_partition="[number of partitions (default 4)]" \
--num_init_part="[number of initial partitions (default 4)]" \
--num_iteration="[number of iterations (default 10)]" \
--unbalance_val="[unbalance tolerance of each partition (default 1.05)]" \
--indptr="[path to indptr file (Required)]" \
--indices="[path to indices file (Required)]" \
--output="[path to output file (Required)]" \
--node_weight="[path to node_weight file (Optional)]" \
--edge_weight="[path to edge_weight file (Optional)]" \
```

To use the multi-threaded version, change `./bin/main` with `./bin/mt_main`.

To use the distributed version, change `./bin/main` with `mpirun -np [number of process] ./bin/mpi_main`. Note, `./bin/mpi_main` is buggy and any contribution is welcome! :smile:

All the input files must end with .npy and are NumPy arrays stored in int64_t format.

Metis assumes the graph is symmetrical, meaning that if `(u -> v)` then `(v -> u)` and the graph must be undirected.

`indptr` file must be ended with either `indptr_sym.npy` or `indptr_xsym.npy`. 
1. If the `indptr` file ends with `indptr_xsym.npy`, additional steps will be taken to convert it into symmetrical graphs.
2. If the `indptr` file ends with `indptr_sym.npy`, it will be provided to Metis directly.

`indices` file must end with either `indices_sym.npy` or `indices_xsym.npy`. 
1. If the `indices` file ends with `indices_xsym.npy`, additional steps will be taken to convert it into symmetrical graphs.
2. If the `indices` file ends with `indices_sym.npy`, it will be provided to Metis directly.

`node_weight`, if provided, must have a length equal to the number of nodes in the graph.

`edge_weight`, if provided, must have a length equal to the number of edges in the graph.

`output` path must end with `.npy` and the result will be an int64_t NumPy array.
