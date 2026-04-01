
import numpy as np
import argparse
import dgl
import os
import json
from datetime import datetime
import time 
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--absolute_path', 
        type=str, 
        help="Path to the dgl graph object.",
        required=True
    )
    
    parser.add_argument(
        '--results_dir', 
        type=str, 
        help="Where to store the metrics.",
        required=True
    )
       
    return parser

def compute_gaps(g):
    t = time.time()
    u, v = g.edges()
    print(f"got edges in {time.time() - t} seconds")
    t = time.time()
    gaps = np.abs(u - v).to(float)
    avg_gap = gaps.mean()
    print(f"computed avg_gap in {time.time() - t} seconds ")
    t = time.time()
    max_gaps = np.zeros(len(g.in_degrees()))
    np.maximum.at(max_gaps, u, gaps)
    np.maximum.at(max_gaps, v, gaps)
    avg_bandwidth = max_gaps.mean()
    print(f"computed avg_bandwidth in {time.time() - t} seconds ")
    
    return avg_gap.item(), avg_bandwidth.item()

if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    t = time.time()
    graph = dgl.load_graphs(args.absolute_path)[0][0]
    print(f"Graph loaded with {graph.num_nodes()} nodes and {graph.num_edges()} edges in {time.time() - t}")
    avg_gap, avg_bandwidth = compute_gaps(graph)
    
    metrics = {}
    metrics["absolute_path"] = args.absolute_path
    metrics["avg_gap"] = avg_gap
    metrics["avg_bandwidth"] = avg_bandwidth
    print(metrics)
    
    m = {key: int(value) if isinstance(value, np.integer) else value for key, value in metrics.items()}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(f"{args.results_dir}/{timestamp}.json", "w") as f:
        json.dump(m, f)  
        
    print(f"Average gap between connected nodes: {avg_gap:.2f}")
    exit(0)    
