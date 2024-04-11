#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

data_dir=${SCRIPT_DIR}/dataset/products
output_dir=${SCRIPT_DIR}/dataset/partition_maps/

mkdir -p ${output_dir}

./bin/main \
--num_partition=4 \
--indptr=${data_dir}/indptr_xsym.npy \
--indices=${data_dir}/indices_xsym.npy \
--node_weight=${data_dir}/node_weight.npy \
--edge_weight=${data_dir}/edge_weight.npy \
--output=${output_dir}/result.npy