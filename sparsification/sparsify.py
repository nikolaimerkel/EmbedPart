import argparse
from sparsification.GraphSparsifier import *
from utils.graph_utils import load_dgl_graph

from configs.config import DGL_GRAPHS 


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-graph_name", required=True,
                        type=str, help="The graph name.")
    parser.add_argument("-sparsifier", required=True,
                        type=str, help="The sparsifier. Options: rvs (random vertex sampling), res (random edge sampling), dbs (degree based sparsification), and gap", default="rvs")
    parser.add_argument("-sparsifier_level", required=True,
                        type=float, help="The number of features.", default=1.0)
    

    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    graph, num_classes = load_dgl_graph(args.graph_name) 
    
    sparsifier = GraphSparsifier(graph=graph, base_path=DGL_GRAPHS, graph_name = args.graph_name)
    
    if args.sparsifier == "rvs":
        print("rvs")
        sparsifier.random_vertex_sparsifier(keep_prob=args.sparsifier_level)
        
    if args.sparsifier == "res":
        print("res")
        sparsifier.random_edge_sparsifier(keep_prob=args.sparsifier_level)
        
    if args.sparsifier == "dbs":
        print("dbs")
        sparsifier.degree_based_sparsifier(degree_threshold=args.sparsifier_level)
        
    if args.sparsifier == "gap":
        print("gap")
        sparsifier.gap_sparsifier(args.sparsifier_level)
        

    