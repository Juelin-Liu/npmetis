from lib import pymetis
import numpy as np
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description='Metis partitioning algorithm')
    parser.add_argument("--indptr_path", required=True, type=str)
    parser.add_argument("--indices_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument('--num_partition', default=4, type=int, help='Number of partitions')
    parser.add_argument('--num_iteration', default=10, type=int, help='Number of partitions')
    parser.add_argument('--num_initpart', default=1, type=int, help='Number of initial partitions')
    parser.add_argument('--unbalance_val', default=1.05, type=float, help='Unbalance toleratnce')
    parser.add_argument('--node_weight_path', default="", type=str, help="Node weight path")
    parser.add_argument('--edge_weight_path', default="", type=str, help="Edge weight path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    num_partition:int = args.num_partition
    num_iteration: int = args.num_iteration
    num_initpart:int = args.num_initpart
    unbalance_val: float = args.unbalance_val
    indptr_path: str = args.indptr_path
    indices_path: str = args.indices_path
    obj_cut = True
    
    indptr = np.load(indptr_path).astype(np.uint64)
    indices = np.load(indices_path).astype(np.uint32)
    avg_deg = indices.shape[0] // indptr.shape[0]

    node_weight = np.array([], dtype=np.int64)
    edge_weight = np.array([], dtype=np.int64)

    if args.node_weight_path != "":
        node_weight = np.load(args.node_weight_path).astype(np.int64)
        if np.min(node_weight) == 0:
            node_weight += 1
        
    if args.edge_weight_path != "":
        edge_weight = np.load(args.edge_weight_path).astype(np.int64)
    
    start = time.time()
    result = pymetis.mt_metis_assignment(num_partition, num_iteration, num_initpart, unbalance_val, obj_cut, indptr, indices, node_weight, edge_weight)
    end = time.time()
    print(f"fininished metis in {round(end - start)} secs")
    np.save(args.output_path, result)
    print(f"saved to {args.output_path}")