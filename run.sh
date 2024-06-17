#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

graph_name=orkut
sym_type=sym
data_dir=/data/juelin/project/asplos25/dgl/dataset/graph/${graph_name}
output_dir=${SCRIPT_DIR}/partition_maps/${graph_name}

mkdir -p ${output_dir}

python3 run.py \
--num_partition=4 \
--num_iteration=10 \
--indptr_path=${data_dir}/indptr_${sym_type}.npy \
--indices_path=${data_dir}/indices_${sym_type}.npy \
--node_weight_path=${data_dir}/node_weight.npy \
--edge_weight_path=${data_dir}/edge_weight.npy \
--output_path=${output_dir}/w4_n10_ndst_efreq.npy