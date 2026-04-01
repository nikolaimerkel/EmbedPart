import argparse
from tqdm import tqdm  # Import tqdm for progress bar

from utils.graph_utils import load_dgl_graph

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-absolute_path', 
        type=str, 
        help="Path to edgelist.",
        required=True
    )
    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    graph,_ = load_dgl_graph("ogbn-products")
    
    in_deg = graph.in_degrees()
    out_deg = graph.out_degrees()

    isolated_mask = (in_deg == 0) & (out_deg == 0)
    isolated_nodes = isolated_mask.nonzero(as_tuple=True)[0]

    print(f"Number of isolated nodes: {len(isolated_nodes)}")

    
    src_nodes, dst_nodes = graph.edges()
     
    print(f"Graph loaded with {graph.num_nodes()} nodes and {graph.num_edges()} edges")
    print(f"Number of edges according to .edges(): {len(src_nodes)}, {len(dst_nodes)}")
    
    exit(0)
    all_nodes = set(range(graph.num_nodes()))
    nodes_in_edges = set(src_nodes.numpy()) | set(dst_nodes.numpy())
    isolated_nodes = all_nodes - nodes_in_edges
    print(f"Number of isolated nodes: {len(isolated_nodes)}")
         
    with open(args.absolute_path, "r") as f:
        lines = f.readlines()
        # each line contains two numbers: src and dst
        # src and dst are separated by a space
        # write code to see if the ids are consecutive
        max_id = 0
        loops = 0
        unique_ids = set()
        for line in tqdm(lines, desc="Processing lines"):  # Wrap lines with tqdm
            src, dst = map(int, line.split())
            if src == dst:
                loops += 1
            unique_ids.add(src)
            unique_ids.add(dst)
            max_id = max(max_id, src, dst)
        print(f"Maximum ID found: {max_id} number of unique ids: {len(unique_ids)} loops: {loops}")
        
        



