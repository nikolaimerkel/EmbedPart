import argparse
import numpy as np
import dgl
import time
import torch
import os

from configs.config import DGL_GRAPHS, PARTITIONED_GRAPHS








vid2pid_files = [
    "ogbn-papers100M.directed.cuttana.cuttana256.P32.vid2pid",
    "ogbn-arxiv.directed.cuttana.cuttana256.P32.vid2pid",
    "ogbn-products.directed.cuttana.cuttana256.P32.vid2pid",
    "reddit.directed.cuttana.cuttana256.P32.vid2pid",

    # "ogbn-arxiv.metis.P32.vid2pid",
    # "ogbn-arxiv.random.P32.vid2pid",
    # "ogbn-arxiv.ldg.P32.vid2pid",
    # "ogbn-arxiv.spinner.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.vid2pid",
    # 
    # "ogbn-products.metis.P32.vid2pid",
    # "ogbn-products.random.P32.vid2pid",
    # "ogbn-products.ldg.P32.vid2pid",
    # "ogbn-products.spinner.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.vid2pid",

    # "reddit.metis.P32.vid2pid",
    # "reddit.random.P32.vid2pid",
    # "reddit.ldg.P32.vid2pid",
    # "reddit.spinner.P32.vid2pid",
    # "reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.vid2pid",
    # "reddit.Dreddit.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "reddit.Dreddit.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.vid2pid",

    # "ogbn-papers100M.metis.P32.vid2pid",
    # "ogbn-papers100M.random.P32.vid2pid",
    # "ogbn-papers100M.ldg.P32.vid2pid",
    # "ogbn-papers100M.spinner.P32.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.vid2pid",   
    # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.vid2pid",
    # 
    # "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",

    # "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",


    # "reddit.Dreddit.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "reddit.Dreddit.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",
    # "reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",

    # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.vid2pid",

    # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P1024.vid2pid",
    # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P1024.vid2pid",
    
    
    #"ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P1024.vid2pid",
    #"ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P1024.vid2pid",
]
graphs = ["ogbn-arxiv", "ogbn-products", "reddit", "ogbn-papers100M"]





if __name__ == "__main__":
    commands = []
    graph_name = None
    for vid2pid_file in vid2pid_files:
        for graph in graphs:
            if graph in vid2pid_file:
                graph_name = graph
                break
        if graph_name is None:
            print("Graph name not found in vid2pid files.")
            exit(1)
            
        commands.append(f"python -m reordering.reorder -graph_name {graph_name} -vid2pid {PARTITIONED_GRAPHS}/{vid2pid_file}")    
        
    i = 1
    for command in commands:
        print(f"{i}/{len(commands)}", command)
        i+=1
        start = time.time()
        os.system(command)
        stop = time.time()
        print(f"Time: {stop-start:.4f}s")
