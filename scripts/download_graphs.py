import argparse
from tqdm import tqdm
from sparsification.GraphSparsifier import *
from utils.graph_utils import load_dgl_graph
import dgl
import torch as th

from configs.config import EDGELISTS, DGL_GRAPHS

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-graph_name", required=True,
                        type=str, help="The graph name.")
    return parser

if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    graph, num_classes = load_dgl_graph(args.graph_name) 
    print(graph)
    print(num_classes)
      
    conf = {"num_classes":  th.tensor([num_classes])}
    
    print(conf)
    src_nodes, dst_nodes = graph.edges()

    if not os.path.exists(DGL_GRAPHS):
        os.makedirs(DGL_GRAPHS)
    print(f"Saving graph to {DGL_GRAPHS}/{args.graph_name}.dgl")
    dgl.save_graphs(f"{DGL_GRAPHS}/{args.graph_name}.dgl", [graph], conf)   
    
    if not os.path.exists(EDGELISTS):
        os.makedirs(EDGELISTS)
    path = f"{EDGELISTS}/{args.graph_name}"
    
    print("computing isolated nodes")
    # Find all nodes and nodes that appear in edges
    all_nodes = set(range(graph.num_nodes()))
    nodes_in_edges = set(src_nodes.numpy()) | set(dst_nodes.numpy())
    isolated_nodes = all_nodes - nodes_in_edges
    print(f"Number of isolated nodes: {len(isolated_nodes)}")

    with open(path, "w") as f:
        # Write existing edges
        for u, v in tqdm(zip(src_nodes.numpy(), dst_nodes.numpy()), total=len(src_nodes), desc="Writing edges"):
            f.write(f"{u} {v}\n")
        
        # Write self-loops for isolated nodes
        for node in tqdm(isolated_nodes, desc="Writing self-loops"):
            f.write(f"{node} {node}\n")

    print(f"Edge list saved to {path}")
    print(f"Wrote {len(src_nodes)} edges and {len(isolated_nodes)} self-loops for isolated nodes.")