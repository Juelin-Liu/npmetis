#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

data_dir=${SCRIPT_DIR}/dataset/products
output_dir=${SCRIPT_DIR}/dataset/partition_maps/

mkdir -p ${output_dir}

./bin/pmetis \
--num_partition=4 \
--indptr=${data_dir}/indptr_xsym.npy \
--indices=${data_dir}/indices_xsym.npy \
--node_weight=${data_dir}/dst_node_weight.npy \
--edge_weight=${data_dir}/edge_weight.npy \
--output=${output_dir}/products_w4_ndst_efreq_unbal_vol.npy
