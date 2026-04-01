import numpy as np
import os
from datetime import datetime

if __name__ == "__main__":
    
    def template(fn, results_dir):
        return f"python scripts/reordering_metrics.py --absolute_path {fn} --results_dir {results_dir}"
    
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"/mnt/data/gnn-partitioning/results/reordering-metrics/{ts}"
    os.makedirs(results_dir, exist_ok=True)
    
    commands = []



    for f in [
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.ldg.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.metis.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.random.P32.reordered",
       #  "/mnt/reordered/ogbn-arxiv.spinner.P32.reordered",
        "/mnt/reordered/ogbn-arxiv.directed.cuttana.cuttana256.P32.reordered",
        
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.ldg.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.metis.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.random.P32.reordered",
       # "/mnt/reordered/ogbn-papers100M.spinner.P32.reordered",
        "/mnt/reordered/ogbn-papers100M.directed.cuttana.cuttana256.P32.reordered",

        
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/ogbn-products.ldg.P32.reordered",
       # "/mnt/reordered/ogbn-products.metis.P32.reordered",
       # "/mnt/reordered/ogbn-products.random.P32.reordered",
       # "/mnt/reordered/ogbn-products.spinner.P32.reordered",
        "/mnt/reordered/ogbn-products.directed.cuttana.cuttana256.P32.reordered",

        
       # "/mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.linkgraphsage.H64.L2.O16.F25-10.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P32.reordered",
       # "/mnt/reordered/reddit.Dreddit.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB2048.0.VB2048.0.P32.reordered",
       # "/mnt/reordered/reddit.ldg.P32.reordered",
       # "/mnt/reordered/reddit.metis.P32.reordered",
       # "/mnt/reordered/reddit.random.P32.reordered",
       # "/mnt/reordered/reddit.spinner.P32.reordered",
        "/mnt/reordered/reddit.directed.cuttana.cuttana256.P32.reordered",

    ]:
        commands.append(template(fn=f, results_dir=results_dir))

            
    # write commands to file
    counter = 0
    all = len(commands)
    with open("scripts/reordering_metrics.sh", "w") as f:
        for c in commands:
            f.write(f"echo {counter}/{all} \n")
            f.write(c + "\n")
            counter += 1

        
        
        