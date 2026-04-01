import argparse
from utils.graph_utils import load_dgl_graph
import numpy as np
import dgl
import time
import torch

from configs.config import DGL_GRAPHS, REORDERED_GRAPHS


def parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-graph_name',  
        type=str, 
        help="The graph name to reorder",
        required=True
    )
    
    parser.add_argument(
        '-vid2pid',  
        type=str, 
        help="The absolute path to the vid2pid file",
        required=True
    )
    
    return parser


def vid2pid(vid2pid_file):
    try:
        with open(vid2pid_file) as f:
            content = f.readlines()
    except Exception as error:
        print("Error:", error)
        return None
    v2p = []
    for line in content:
        raw_pid = line.strip()
        if len(raw_pid) > 0:
            v2p.append(int(raw_pid))
    return np.array(v2p)



if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    # Load graph dgl graph
    g, num_classes = load_dgl_graph(args.graph_name)
    print(f"Graph {args.graph_name} loaded", g)

    # Load vid2pid file
    vid2pidMapping = vid2pid(args.vid2pid)
    output_file = REORDERED_GRAPHS + "/"+ args.vid2pid.split("/")[-1].replace(".vid2pid", ".reordered")
    print(f"output_file: {output_file}")
    # Compute permutation from vid2pid    
    partition_ids = np.unique(vid2pidMapping)
    print("partition ids:", partition_ids)
    print("vid2pidMapping:", vid2pidMapping)
    
    print("num nodes in graph:", g.num_nodes(), "num elements in mapping:", len(vid2pidMapping))

        # For each partition, get the list of vertex IDs (vids) in that partition
    permutation = []
    for pid in partition_ids:
        vids_in_partition = np.where(vid2pidMapping == pid)[0]
        permutation.extend(vids_in_partition)

    # Reorder graph using permutation
    g_reordered = dgl.reorder_graph(g, node_permute_algo='custom', permute_config={'nodes_perm': permutation})
     
    # Save reordered graph to disk
    dgl.save_graphs(output_file, [g_reordered], {'num_classes': torch.tensor([num_classes])})

