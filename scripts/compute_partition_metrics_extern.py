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
        '-cpp_metrics', 
        type=str, 
        help="Partitioning metrics from C++, which is basically the runtime.",
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
    # python -m scripts.compute_partition_metrics_extern -graph /mnt/data/dgl/ogbn-arxiv.dgl -vid2pid /mnt/data/partitioned/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.P2.vid2pid -cpp_metrics /mnt/data/edgelists/ogbn-arxiv.directed.ldg.2.edgecut.partitioning.metrics.json -num_parts 2
              
    graph = dgl.load_graphs(args.graph)[0][0]
    print(graph)
    
    
    vid2pid = np.loadtxt(args.vid2pid, dtype=int)
    print(vid2pid)
    print(len(vid2pid))
    metrics = get_partition_metrics(graph, vid2pid, args.num_parts)
    
    print("DGL metrics: ", metrics)
    
    partitioning_time = None
    spinner_conf_time = None
    with open(args.cpp_metrics, 'r') as file:
        # Load the JSON object from the file and add to list
        metrics_cpp = json.load(file)
        print(args.cpp_metrics)
        print("cpp metrics: ",metrics_cpp)
        partitioning_time = metrics_cpp["partitioning_time"]
        if "spinner_conf_time" in metrics_cpp:
            spinner_conf_time = metrics_cpp["spinner_conf_time"]
        
        
        
        
    metrics["partitioning_time"] = partitioning_time
    metrics["spinner_conf_time"] = spinner_conf_time
    metrics["partitioner"] = args.partitioner
    metrics["graph_name"] = args.graph_name
    metrics["num_parts"] = args.num_parts
    metrics["vid2pid"] = args.vid2pid
    metrics["cpp_metrics"] = args.cpp_metrics
    metrics["graph"] = args.graph
    
    print(metrics)
      
      
    # Convert all values to native Python types
    metrics_clean = {k: int(v) if isinstance(v, (np.integer,)) 
                    else float(v) if isinstance(v, (np.floating,))
                    else v
                    for k, v in metrics.items()}
        
    with open(f"/mnt/data/gnn-partitioning/results/partitioning-metrics/cpp/{args.graph_name}.{args.partitioner}.P{args.num_parts}.json", "w") as f:
        f.write(json.dumps(metrics_clean, separators=(',', ':')) + '\n')
            
            
    
    
    
        
        
        
    
    
    
        
