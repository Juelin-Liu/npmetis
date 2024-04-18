#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

graph_name=friendster
sym_type=sym
data_dir=/data/juelin/dataset/gsplit/numpy/dataset/${graph_name}
output_dir=${SCRIPT_DIR}/dataset/partition_maps/${graph_name}

mkdir -p ${output_dir}

./bin/mt_main \
--num_partition=4 \
--num_iteration=1 \
--indptr=${data_dir}/indptr_${sym_type}.npy \
--indices=${data_dir}/indices_${sym_type}.npy \
--node_weight=${data_dir}/dst_node_weight.npy \
--edge_weight=${data_dir}/edge_weight.npy \
--output=${output_dir}/${graph_name}.npy
