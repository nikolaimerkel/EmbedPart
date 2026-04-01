import argparse
import json
import dgl
from partitioner.helper import get_partition_metrics, get_balance
import numpy as np
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-graph', 
        type=str, 
        help="Absolute path to the graph file.",
        required=True
    )
    parser.add_argument(
        '-vid2pid', 
        type=str, 
        help="Absolute path to the vid2pid file.",
        required=True
    )
    
    parser.add_argument(
        '-num_parts', 
        type=int, 
        help="The number of partitinos the graph was partitioned into.",
        required=True
    )
    
    parser.add_argument(
        '-partitioner', 
        type=str, 
        help="The partitioner used to partition the graph.",
        required=True
    )
    
    parser.add_argument(
        '-graph_name', 
        type=str, 
        help="The mane of the graph.",
        required=True
    )
        
       

    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    print(args)
 
    # python -m scripts.compute_partition_metrics_extern_no_cpp -graph /mnt/data/dgl/ogbn-arxiv.dgl -vid2pid /mnt/data/partitioned/ogbn-arxiv.metis.P4.vid2pid  -num_parts 4 -partitioner xx -graph_name ogbn-arxiv
    # python -m scripts.compute_partition_metrics_extern_no_cpp -graph /mnt/data/dgl/ogbn-arxiv.dgl -vid2pid /mnt/data/edgelists/ogbn-arxiv.cuttana.cuttana256.4.vid2pid  -num_parts 4 -partitioner xx -graph_name ogbn-arxiv
              
    graph = dgl.load_graphs(args.graph)[0][0]
    print(graph)
    
    
    vid2pid = np.loadtxt(args.vid2pid, dtype=int)
    print(vid2pid)
    print(len(vid2pid))
    metrics = get_partition_metrics(graph, vid2pid, args.num_parts)
    
    print("DGL metrics: ", metrics)
    
    metrics["partitioner"] = args.partitioner
    metrics["graph_name"] = args.graph_name
    metrics["num_parts"] = args.num_parts
    metrics["vid2pid"] = args.vid2pid
    metrics["graph"] = args.graph
    
    print(metrics)
      
      
    # Convert all values to native Python types
    metrics_clean = {k: int(v) if isinstance(v, (np.integer,)) 
                    else float(v) if isinstance(v, (np.floating,))
                    else v
                    for k, v in metrics.items()}
        
    with open(f"/mnt/data/gnn-partitioning/results/partitioning-metrics/cpp/{args.graph_name}.{args.partitioner}.P{args.num_parts}.json", "w") as f:
        f.write(json.dumps(metrics_clean, separators=(',', ':')) + '\n')
            
            
    
    
    
        
        
        
    
    
    
        
