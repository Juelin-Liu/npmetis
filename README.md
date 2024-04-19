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
MPI is required, recommend using MPICH v4.2 and above for it supports MPI standard v4.1, which is necessary for handling large graphs with more than 2 billion edges or nodes.

```bash
cd /tmp
wget https://www.mpich.org/static/downloads/4.2.1/mpich-4.2.1.tar.gz
tar -xvf mpich-4.2.1.tar.gz
cd mpich-4.2.1
export MPI_INSTALL_PREFIX=${HOME}/local # change this to your prefered place
./configure --prefix=$MPI_INSTALL_PREFIX --with-pmi=pmix # this might take a while
make > m.txt 2>&1 # this might take a while
make install
# add MPI to your path
export PATH=${MPI_INSTALL_PREFIX}/bin:$PATH
```

# How to compile:
To build main binaries:
```bash
./build.sh
```

To use ParMETIS (distributed), you must agree to its [Licence](https://github.com/KarypisLab/ParMETIS/blob/main/LICENSE).
<!-- First download its source code from Github:
```bash
wget https://github.com/KarypisLab/ParMETIS/archive/refs/heads/main.zip -O third_party/parmetis.zip
pushd third_party && unzip parmetis.zip && mv ParMETIS-main parmetis && rm parmetis.zip && popd
```
Then, uncomment the code blocks in `build.sh` after `build ParMETIS`, also uncomment the part in `cppmetis/CMakeLists.txt` after `Build mpi_main start`
`.

The output binary files will be in the `./bin` directory. -->

# How to use METIS binaries provided in this project:
```shell
./bin/main \
--num_partition="[number of partitions (default 4)]" \
--num_init_part="[number of initial partitions (default 1)]" \
--num_iteration="[number of iterations (default 10)]" \
--unbalance_val="[unbalance tolerance of each partition (default 1.05)]" \
--indptr="[path to indptr file (Required)]" \
--indices="[path to indices file (Required)]" \
--output="[path to output file (Required)]" \
--node_weight="[path to node_weight file (Optional)]" \
--edge_weight="[path to edge_weight file (Optional)]" \
```

Alternatively, to use the multi-threaded version, change `./bin/main` with `./bin/mt_main`.

To use the distributed version, change `./bin/main` with `mpirun -np [number of process] ./bin/mpi_main`. 

All the input files must end with .npy and are NumPy arrays stored in int64_t format.

Metis requires:
1. The graph is undirected (symmetrical), meaning that if `(u -> v)` then `(v -> u)`. 
2. The graph has no self-loops. 

So the input graph must be symmetric and undirected with no self-loops.

`indptr`: Indptr of the graph, assume ids are 0 indexed and contiguous.

`indices`: Adjacency lists in an array, must contain edges in both directions.

`node_weight`: If provided, must have a length equal to the number of nodes in the graph.

`edge_weight`: If provided, must have a length equal to the number of edges in the graph.

`output`: The path to save the result, must end with `.npy`, the result will be saved as an int64_t NumPy array.

# Graph Preprocessing
We also provide utilities `to_sym` for you to convert directed graphs to undirected graphs.

```shell
./bin/to_sym \
--indptr="[path to indptr file (Required)]" \
--indices="[path to indices file (Required)]" \
--edge_weight="[path to edge_weight file (Optional)]" \
--output="[path to output directory (Required)]" \
```

Suppose the output directory is `./output`, the resulting symmetrical graph will be saved as:
```
./output/
   indptr_sym.npy # indptr for the symmetrical graph
   indices_sym.npy # indices for the symmetrical graph
   edge_weight_sym.npy # edge weights for the symmetrical graph
```