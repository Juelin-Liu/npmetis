# Project: C++ Metis K-Way Partitioning Wrapper with NumPy Integration

# Overview:
This project provides a user-friendly C++ wrapper library around the robust Metis K-Way graph partitioning algorithm, enabling seamless usage with versatile NumPy arrays for input and output data representation.

# Objectives:
Metis Integration: Core functionality focuses on encapsulating the Metis K-Way partitioning API within a well-designed C++ interface and provide easy to use binary.
NumPy Compatibility: Implement conversion mechanisms between Metis data structures and NumPy arrays, ensuring effortless data flow in common scientific computing pipelines.

Components:
1. C++ Files `cppmetis`: Define classes, functions, and data types for representing graphs (adjacency structures) and interacting with the Metis library.
2. Wrapper functions for Metis API calls.
3. Build scripts: Manage project compilation and linking, including dependencies on Metis and NumPy libraries.

# Potential Use Cases:
You want to use Metis to conduct kWay partitioning but found it is difficult to use / slow.

# Prerequisites:
## Single threaded: 
C++ compiler supoorting C++ standard 20. Ninja, CMake.

## Multi threaded:
OpenMP

## Distributed:
MPI

# Building:
```bash
./build.sh
```
The output binrary will be in the `./bin` directory.

# How to use:
```shell
./bin/main \
--num_partition="[number of partitions (Required)]" \
--indptr="[path to indptr file (Required)]" \
--indices="[path to indices file (Required)]" \
--output="[path to output file (Required)]" \
--node_weight="[path to node_weight file (Optional)]" \
--edge_weight="[path to edge_weight file (Optional)]" \
```
To use multi-threaded version, simply change `./bin/main` with `./bin/mt_main`.

To use distributed version, simply change `./bin/main` with `mpirun -np [number of process] ./bin/mpi_main`. Note, `mpi_main` is buggy. Any contribution is welcome.

All the input files must be ended with .npy and are numpy arrays stored in int64_t format.

The `indptr` file must be ended with either `indptr_sym.npy` or `indptr_xsym.npy`. 
Metis assumes the graph is symmetrical, meaning that if `(u -> v)` then `(v -> u)` and the graph must be undirected.
If the `indptr` file end with `indptr_xsym.npy`, additional step will be taken to convert it into symmetrical graphs.
If the `indptr` file end with `indptr_sym.npy`, it will be provided to Metis directly.

Similarly, the `indices` file must be ended with either `indices_sym.npy` or `indices_xsym.npy`. 
If the `indices` file end with `indices_xsym.npy`, additional step will be taken to convert it into symmetrical graphs.
If the `indices` file end with `indices_sym.npy`, it will be provided to Metis directly.

The `node_weight`, if provided, must have a length equal to the number of nodes in the graph.

The `edge_weight`, if provided, must have a length equal to the number of edges in the graph.

The output files must be ended with `.npy` and the output will be a int64_t numpy array.