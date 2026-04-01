import argparse
import numpy as np

from datetime import datetime

if __name__ == "__main__":
        
    grahs = [
        
       # "ogbn-papers100M.directed.cuttana.cuttana256.P32.reordered",
        "ogbn-arxiv.directed.cuttana.cuttana256.P32.reordered",
        "ogbn-products.directed.cuttana.cuttana256.P32.reordered",
        "reddit.directed.cuttana.cuttana256.P32.reordered",
        
        "ogbn-arxiv.metis.P32.reordered",
        "ogbn-arxiv.random.P32.reordered",
        "ogbn-arxiv.ldg.P32.reordered",
        "ogbn-arxiv.spinner.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
        "ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",

        "ogbn-products.metis.P32.reordered",
        "ogbn-products.random.P32.reordered",
        "ogbn-products.ldg.P32.reordered",
        "ogbn-products.spinner.P32.reordered",
        "ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
        "ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
        "ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
        "ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
        "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
        "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
        "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
        "ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",

        "reddit.metis.P32.reordered",
        "reddit.random.P32.reordered",
        "reddit.ldg.P32.reordered",
        "reddit.spinner.P32.reordered",
        "reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
        "reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
        "reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
        "reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
        "reddit.Dreddit.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
        "reddit.Dreddit.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
        "reddit.Dreddit.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
        "reddit.Dreddit.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",

       # "ogbn-papers100M.metis.P32.reordered",
       # "ogbn-papers100M.random.P32.reordered",
       # "ogbn-papers100M.ldg.P32.reordered",
       # "ogbn-papers100M.spinner.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P1024.reordered",
       # "ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P1024.reordered",
        
        
     

]


    commands = []
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"/mnt/data/gnn-partitioning/results/single-node-training/{ts}"
    
    REP = 5
    for _ in range(REP):
        for mode in ["cpu", "cuda"]:
            for g in grahs:
                if "papers100M" in g and mode == "cuda":
                    print("we skipp papers100M non cuda ")
                    continue
                cmd = f"python -m reordering.train  --absolute_path_dgl_graph /mnt/reordered/{g} --results_dir {results_dir} --mode {mode}"
                commands.append(cmd)
        
    # write commands to file
    counter = 0
    all = len(commands)
    with open("reordering/train.sh", "w") as f:
        for c in commands:
            f.write(f"echo {counter}/{all} \n")
            f.write(c + "\n")
            counter += 1

        
        
    
    