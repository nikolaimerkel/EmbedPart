

import os
from datetime import datetime


commands = []
EPOCHS_TRAINING = 10
NUM_LAYERS = [2]
HIDDEN_DIMS = [64]
     
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create the directory name
results_dir_base = f"/mnt/data/gnn-partitioning/results/single-node-training/{timestamp}"

# Create the directory
os.makedirs(results_dir_base, exist_ok=True)

reordered_graphs = [
    "ogbn-arxiv.metis.P32.rerordered",
    "ogbn-arxiv.random.P32.rerordered",
    "ogbn-arxiv.ldg.P32.rerordered",
    "ogbn-arxiv.spinner.P32.rerordered",
    "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered",
    "ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered",
    "ogbn-products.metis.P32.rerordered",
    "ogbn-products.random.P32.rerordered",
    "ogbn-products.ldg.P32.rerordered",
    "ogbn-products.spinner.P32.rerordered",
    "ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered",
    "ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered",
    "reddit.metis.P32.rerordered",
    "reddit.random.P32.rerordered",
    "reddit.ldg.P32.rerordered",
    "reddit.spinner.P32.rerordered",
    "reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered",
    "reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered",
    "ogbn-papers100M.metis.P32.rerordered",
    "ogbn-papers100M.random.P32.rerordered",
    "ogbn-papers100M.ldg.P32.rerordered",
    "ogbn-papers100M.spinner.P32.rerordered",
    "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered",
    "ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered",
    "ogbn-papers100M.directed.cuttana.cuttana256.P32.reordered",
    "ogbn-arxiv.directed.cuttana.cuttana256.P32.reordered",
    "ogbn-products.directed.cuttana.cuttana256.P32.reordered",
    "reddit.directed.cuttana.cuttana256.P32.reordered",
]

counter = 0

for infrastructure in ["cuda", "cpu"]:
    for g in reordered_graphs:
        if "ogbn-papers100M" in g and infrastructure == "cuda":
            # Skip ogbn-papers100M with cuda
            print(f"Skipping {g} with {infrastructure} because it is too large")
            continue
        for num_layers in NUM_LAYERS:
            for hidden_dim in HIDDEN_DIMS:
                counter += 1
                results_dir = f"{results_dir_base}/{counter}"    
                template = [
                    'python3 train.py',
                    f'-absolute_path_to_graph /mnt/reordered/{g}',
                    f'-infrastructure {infrastructure}',
                    f'-num_epochs {EPOCHS_TRAINING}',
                    f'-num_layers {num_layers}',
                    f'-hidden_dim {hidden_dim}',
                    f'-results_dir {results_dir}'
                ]
                commands.append(" ".join(template))

counter = 0
all = len(commands)
with open("create_benchmark_single_node.sh", "w") as f:
    for c in commands:
        f.write(f"\n echo {counter}/{all} \n")
        f.write(c + "\n")
        counter += 1

    
    
    